"""H3-Image restoration with shared RGB objectives and native single-frame latents."""

from collections.abc import Iterator
from pathlib import Path

import torch
from torch import Tensor, nn

from unituner.adapters.encoder import EncoderAdapters
from unituner.config.training import TrainingConfig
from unituner.data.datasets.restoration import RestorationDataset
from unituner.inference.restoration.vae import encode_restoration_image, restoration_posterior
from unituner.models.features.loading import load_dino_convnext
from unituner.models.minimax_h3.image_cache import MiniMaxH3ImageRecord
from unituner.models.minimax_h3.image_vae import load_image_vae
from unituner.training.contracts import StepResult
from unituner.training.integrations.minimax_h3.image import MiniMaxH3ImageBinding
from unituner.training.losses.adversarial import frozen_discriminator, gan_loss
from unituner.training.losses.perceptual import DinoPerceptualLoss
from unituner.training.losses.restoration import generator_loss
from unituner.training.losses.wavelet import WaveletLoss
from unituner.training.methods.discriminator import MultilevelDiscriminator
from unituner.training.methods.mid_timestep import flow_target
from unituner.training.methods.snr_calibration import calibration_rng, estimate_mid_timestep
from unituner.training.session import TrainingSession


class MiniMaxH3ImageRestoration:
    """Own frozen VAE/features, trainable encoder factors and optional GAN heads."""

    def __init__(
        self, config: TrainingConfig, binding: MiniMaxH3ImageBinding, device: torch.device, session: TrainingSession
    ) -> None:
        options, weights = config.method.restoration, config.losses
        if options is None or weights is None or config.model.vae is None:
            raise ValueError("H3-Image restoration requires method, losses and VAE")
        self.binding, self.options, self.weights = binding, options, weights
        self.vae = (
            load_image_vae(
                Path(config.model.vae),
                config=config.model.minimax_h3.video_topology if config.model.minimax_h3 is not None else None,
                device=device,
            )
            .eval()
            .requires_grad_(False)
        )
        self.posterior = restoration_posterior(config.model.family, options.posterior)
        self.encoder = (
            session.own(EncoderAdapters(self.vae.encoder, rank=options.encoder_rank)) if options.encoder_rank else None
        )
        if weights.lrr and self.encoder is None:
            raise ValueError("Nonzero LRR requires encoder_rank > 0")
        self.state = nn.Module()
        self.state.register_buffer(
            "sigma",
            torch.tensor(
                options.sigma if options.sigma is not None else options.midtimestep / 1000, dtype=torch.float64
            ),
        )
        self.wavelet = None if config.method.wavelet is None else WaveletLoss(config.method.wavelet, device=device)
        self.modules: dict[str, nn.Module] = {"restoration": self.state}
        if self.encoder is not None:
            self.modules["encoder"] = self.encoder
        feature_dtype = torch.bfloat16 if options.feature_precision == "bf16" else torch.float32
        bf16 = options.feature_precision == "bf16"
        self.perceptual: DinoPerceptualLoss | None = None
        self.discriminator: MultilevelDiscriminator | None = None
        if weights.dv3d or weights.gan:
            if options.dino_weights is None:
                raise ValueError("H3-Image Dv3D/GAN requires DINO weights")
            backbone = load_dino_convnext(
                Path(options.dino_weights),
                widths=options.dino_widths,
                depths=options.dino_depths,
                device=device,
                dtype=feature_dtype,
            )
            self.modules["dino"] = backbone
            if weights.dv3d:
                self.perceptual = DinoPerceptualLoss(backbone, bf16_compute=bf16)
            if weights.gan:
                self.discriminator = (
                    MultilevelDiscriminator(
                        backbone, max(next(iter(config.dataset.groups.values())).resolution), bf16_compute=bf16
                    )
                    .to(device)
                    .train()
                )
                self.modules["discriminator"] = self.discriminator.heads

    @property
    def sigma(self) -> float:
        return float(self.state.get_buffer("sigma"))

    @torch.no_grad()
    def calibrate(self, config: TrainingConfig, dataset: RestorationDataset) -> None:
        if not self.options.calibrate:
            return
        from unituner.data.media.restoration import iter_aligned_images

        device = next(self.vae.parameters()).device

        def pairs() -> Iterator[tuple[Tensor, Tensor]]:
            seen = set()
            count = 0
            for index, pair in enumerate(dataset.pairs):
                if pair in seen:
                    continue
                seen.add(pair)
                data = next(
                    group
                    for group in config.dataset.groups.values()
                    if Path(group.cache_directory).resolve() == pair.cache_directory.resolve()
                )
                for low, high in iter_aligned_images(
                    pair.low,
                    pair.high,
                    resolution=data.resolution,
                    seed=config.train.seed + index,
                    flip_probability=data.horizontal_flip_probability,
                    degradation=pair.degradation,
                    sampling=self.options.calibration_sampling,
                ):
                    yield (
                        encode_restoration_image(self.vae, low.unsqueeze(0).to(device), posterior=self.posterior),
                        encode_restoration_image(self.vae, high.unsqueeze(0).to(device), posterior=self.posterior),
                    )
                    count += 1
                    if count >= self.options.calibration_pairs:
                        return

        with calibration_rng(config.train.seed, device), self.encoder.unadapted() if self.encoder else torch.no_grad():
            estimate = estimate_mid_timestep(
                pairs(), family="minimax-h3-image", resolution=next(iter(config.dataset.groups.values())).resolution
            )
        self.state.get_buffer("sigma").fill_(estimate.sigma)

    def steps(self, low: Tensor, high: Tensor, record: MiniMaxH3ImageRecord) -> Iterator[StepResult]:
        """Yield generator then optional discriminator graphs, retaining RGB gradients."""
        vae = self.vae
        if self.weights.lrr and self.encoder is not None:
            with self.encoder.unadapted():
                clean = encode_restoration_image(vae, high, posterior=self.posterior)
            noise = torch.randn_like(clean)
        else:
            clean = None
            noise = None
        low_latents = encode_restoration_image(vae, low, posterior=self.posterior)
        if noise is None:
            noise = torch.randn_like(low_latents)
        text = record.text.unsqueeze(0)
        tags = record.token_tags.unsqueeze(0)
        references = tuple(value.unsqueeze(0) for value in record.references)
        restored = self.binding.restore(low_latents, text, tags, sigma=self.sigma, references=references)
        generated = vae.decode(restored)[:, :, 0].float()
        logits = None
        if self.discriminator is not None:
            with frozen_discriminator(self.discriminator):
                logits = self.discriminator(generated)
        loss = generator_loss(
            generated,
            high,
            low_latents,
            flow_target(clean, noise, self.sigma) if clean is not None else None,
            self.weights,
            perceptual=self.perceptual(generated, high) if self.perceptual is not None else None,
            discriminator_logits=logits,
            histogram_bins=self.options.histogram_bins,
            phase_period=self.options.phase_period,
            phase_cycles=self.options.phase_cycles,
            phase_epsilon=self.options.phase_epsilon,
        )
        total = loss.total
        if self.wavelet is not None:
            wavelet = self.wavelet(generated, high.detach())
            total = total + wavelet
            loss.metrics["wavelet"] = wavelet.detach()
        yield StepResult(total, 1, "generator", loss.metrics)
        if self.discriminator is not None:
            fake = generated.detach()
            d_loss = self.weights.gan * (
                gan_loss(self.discriminator(fake), False) + gan_loss(self.discriminator(high.detach()), True)
            )
            yield StepResult(d_loss, 1, "discriminator")
