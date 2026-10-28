"""Compose native image restoration prediction and explicit G/D update phases."""

from collections.abc import Iterator, Sequence
from pathlib import Path
from time import monotonic
from typing import Literal

import torch
from torch import Tensor, nn

from unitunner.adapters.encoder import EncoderAdapters
from unitunner.artifacts.formats.precision import FloatingPrecision
from unitunner.config.training import TrainingConfig
from unitunner.data.cache.storage import read_caption_record
from unitunner.data.datasets.restoration import AlignedImagePair, RestorationDataset
from unitunner.inference.restoration.vae import (
    decode_restoration_latents,
    encode_restoration_image,
    load_restoration_vae,
    restoration_posterior,
)
from unitunner.logging import clear_context, set_context
from unitunner.models.components.conditioning import Caption, batch_captions
from unitunner.models.features.loading import load_dino_convnext, load_vgg19_features
from unitunner.models.flux.conditioning import FluxTextFeatures
from unitunner.models.flux2.vae import Flux2VAE
from unitunner.models.krea2.conditioning import Krea2TextFeatures, batch_text
from unitunner.models.qwen_image.vae import QwenImageVAE
from unitunner.precision import compute_context
from unitunner.progress import status
from unitunner.training.contracts import StepResult
from unitunner.training.integrations.flux.binding import FluxBinding
from unitunner.training.integrations.flux2.binding import Flux2Binding
from unitunner.training.integrations.flux3.binding import Flux3Binding
from unitunner.training.integrations.image_models import ImageBinding
from unitunner.training.integrations.krea2.binding import Krea2Binding
from unitunner.training.integrations.moss_image.binding import MossImageBinding
from unitunner.training.integrations.qwen_image.binding import QwenImageBinding
from unitunner.training.integrations.qwen_image_2_1.binding import QwenImage21Binding
from unitunner.training.losses.adversarial import frozen_discriminator, gan_loss
from unitunner.training.losses.frequency_distribution import FrequencyDistributionLoss
from unitunner.training.losses.perceptual import DinoPerceptualLoss
from unitunner.training.losses.restoration import generator_loss
from unitunner.training.losses.wavelet import WaveletLoss
from unitunner.training.methods.discriminator import MultilevelDiscriminator
from unitunner.training.methods.mid_timestep import flow_target, restoration_sigma
from unitunner.training.methods.snr_calibration import calibration_rng, estimate_mid_timestep
from unitunner.training.session import TrainingSession


class RestorationMethod:
    """Hold borrowed binding and owned frozen feature/VAE resources for image restoration steps.

    Session owns reversible encoder attachment. Checkpoint modules register each
    feature backbone once even when G/D share it. Caller owns optimizer creation
    and the shared engine; no independent training loop exists here.
    """

    def __init__(
        self,
        config: TrainingConfig,
        binding: ImageBinding,
        device: torch.device,
        session: TrainingSession,
    ) -> None:
        self.wavelet: WaveletLoss | None = (
            None if config.method.wavelet is None else WaveletLoss(config.method.wavelet, device=device)
        )
        options, weights = config.method.restoration, config.losses
        if options is None or weights is None or config.model.vae is None:
            raise ValueError("Restoration requires method settings, losses and a native vae")
        if weights.lrr and not options.encoder_rank:
            raise ValueError("Nonzero LRR requires encoder_rank > 0")
        self.binding, self.weights = binding, weights
        self._first_step = True
        self.compute_precision: FloatingPrecision = (
            "fp32" if config.execution.precision == "auto" else config.execution.precision
        )
        feature_bf16 = options.feature_precision == "bf16"
        feature_dtype = torch.bfloat16 if feature_bf16 else torch.float32
        self.state = nn.Module()
        self.state.register_buffer(
            "sigma",
            torch.tensor(
                options.sigma
                if options.sigma is not None
                else restoration_sigma(
                    options.midtimestep,
                    family=config.model.family,
                    resolution=next(iter(config.dataset.groups.values())).resolution,
                ),
                dtype=torch.float64,
            ),
        )
        self.histogram_bins = options.histogram_bins
        self.phase_period = options.phase_period
        self.phase_cycles = options.phase_cycles
        self.phase_epsilon = options.phase_epsilon
        status(f"[train] Loading restoration VAE: {config.model.vae}")
        self.vae = load_restoration_vae(Path(config.model.vae), family=config.model.family, device=device)
        status("[train] Restoration VAE loaded; installing encoder adapters")
        self.vae.compute_precision = self.compute_precision
        self.posterior: Literal["mean", "sample"] = restoration_posterior(config.model.family, options.posterior)
        vae_channels = self.vae.config.latent_channels * (4 if isinstance(self.vae, Flux2VAE) else 1)
        channels = (
            binding.model.config.out_channels // 4
            if isinstance(binding, FluxBinding)
            else binding.model.config.in_channels
            if isinstance(binding, Flux2Binding)
            else binding.model.config.image_channels
            if isinstance(binding, Flux3Binding)
            else binding.model.config.latent_channels
            if isinstance(binding, MossImageBinding)
            else binding.model.config.out_channels
            if isinstance(binding, QwenImageBinding | QwenImage21Binding)
            else binding.model.config.in_channels
        )
        if vae_channels != channels:
            raise ValueError("Restoration vae/model latent channels differ")
        if isinstance(self.vae, QwenImageVAE) and (
            self.vae.config.input_channels != 3
            or (not isinstance(binding, QwenImageBinding | Krea2Binding))
            or (isinstance(binding, QwenImageBinding) and binding.model.config.variant != "original")
        ):
            raise ValueError("Shared Qwen restoration vae requires RGB and original Qwen Image or Krea2")
        self.encoder = (
            session.own(EncoderAdapters(self.vae.encoder, rank=options.encoder_rank)) if options.encoder_rank else None
        )
        self.modules: dict[str, nn.Module] = {"restoration": self.state}
        if self.encoder is not None:
            self.modules["encoder"] = self.encoder
        self.perceptual: DinoPerceptualLoss | None = None
        self.discriminator: MultilevelDiscriminator | None = None
        self.frequency: FrequencyDistributionLoss | None = None
        if weights.dv3d or weights.gan:
            if options.dino_weights is None:
                raise ValueError("DINO weights are required for Dv3D/GAN")
            status(f"[train] Loading DINO backbone (Dv3D={weights.dv3d}, GAN={weights.gan}): {options.dino_weights}")
            backbone = load_dino_convnext(
                Path(options.dino_weights),
                widths=options.dino_widths,
                depths=options.dino_depths,
                device=device,
                dtype=feature_dtype,
            )
            self.modules["dino"] = backbone
            backbone.recompute = config.execution.recompute == "block"
            status(f"[train] DINO loaded ({options.feature_precision.upper()} storage and compute)")
            if weights.dv3d:
                self.perceptual = DinoPerceptualLoss(backbone, bf16_compute=feature_bf16)
            if weights.gan:
                self.discriminator = (
                    MultilevelDiscriminator(
                        backbone,
                        max(next(iter(config.dataset.groups.values())).resolution),
                        bf16_compute=feature_bf16,
                    )
                    .to(device)
                    .train()
                )
                self.discriminator.validate_resolution(*next(iter(config.dataset.groups.values())).resolution)
                self.modules["discriminator"] = self.discriminator.heads
        if weights.fdl:
            if options.vgg_weights is None:
                raise ValueError("VGG weights are required for FDL")
            if min(next(iter(config.dataset.groups.values())).resolution) < 80:
                raise ValueError("FDL requires image dimensions at least 80")
            status(f"[train] Loading VGG/FDL: {options.vgg_weights}")
            vgg = load_vgg19_features(
                Path(options.vgg_weights), widths=options.vgg_widths, device=device, dtype=feature_dtype
            )
            self.frequency = FrequencyDistributionLoss(
                vgg.feature_stages(), vgg.widths, config.train.seed, bf16_compute=feature_bf16
            ).to(device)
            self.modules["frequency"] = self.frequency
            status(f"[train] VGG/FDL ready ({options.feature_precision.upper()} feature storage and compute)")

    def _trace(self, label: str, device: torch.device) -> None:
        if self._first_step:
            memory = ""
            if device.type == "cuda":
                memory = f" | CUDA allocated={torch.cuda.memory_allocated(device) / 2**30:.2f}GiB peak={torch.cuda.max_memory_allocated(device) / 2**30:.2f}GiB"
            status(f"[train] First microbatch: {label}{memory}")

    @property
    def sigma(self) -> float:
        """Return the CPU FP64 effective noise fraction, including checkpointed calibration."""
        return float(self.state.get_buffer("sigma"))

    def encode(self, pixels: Tensor, *, posterior: Literal["mean", "sample"] = "mean") -> Tensor:
        """Encode restoration images with explicit mean/sample policy and native layout."""
        if posterior not in ("mean", "sample"):
            raise ValueError("Restoration posterior requires mean or sample")
        return encode_restoration_image(self.vae, pixels, posterior=posterior)

    def decode(self, latents: Tensor) -> Tensor:
        """Decode native normalized restoration latents with the differentiable frozen VAE."""
        return decode_restoration_latents(self.vae, latents)

    def restore(self, low_latents: Tensor, captions: Caption | Krea2TextFeatures | FluxTextFeatures) -> Tensor:
        """Dispatch native restoration with model-specific caption geometry and host lengths."""
        if isinstance(self.binding, FluxBinding):
            if not isinstance(captions, FluxTextFeatures):
                raise ValueError("FLUX restoration requires pooled CLIP and T5 features")
            return self.binding.restore(low_latents, captions, sigma=self.sigma)
        if isinstance(self.binding, Krea2Binding):
            if not isinstance(captions, Krea2TextFeatures):
                raise ValueError("Krea2 restoration requires selected-layer text features")
            return self.binding.restore(low_latents, captions, sigma=self.sigma)
        if not isinstance(captions, Caption):
            raise ValueError("Restoration selected-layer features require Krea2")
        if isinstance(self.binding, QwenImage21Binding | Flux2Binding | Flux3Binding | MossImageBinding):
            return self.binding.restore(low_latents, captions, sigma=self.sigma)
        return self.binding.restore(low_latents, captions.features, sigma=self.sigma, caption_mask=captions.mask)

    def captions(self, pairs: Sequence[AlignedImagePair]) -> Caption | Krea2TextFeatures | FluxTextFeatures:
        """Read immutable caption records and preserve each native text representation."""
        device = next(self.vae.parameters()).device
        records = [read_caption_record(pair.cache_directory, pair.caption, device=device) for pair in pairs]
        if isinstance(self.binding, FluxBinding):
            pooled = []
            for record in records:
                if record.pooled is None:
                    raise ValueError("FLUX restoration captions require pooled CLIP features")
                pooled.append(record.pooled)
            return FluxTextFeatures(torch.stack([record.caption for record in records]), torch.stack(pooled))
        if isinstance(self.binding, Krea2Binding):
            return batch_text(tuple(record.caption for record in records))
        return batch_captions([record.caption for record in records])

    @torch.no_grad()
    def calibrate(self, config: TrainingConfig, dataset: RestorationDataset) -> None:
        """Resolve sigma from deterministic first-pair crops with the frozen unadapted VAE.

        Evaluate at most calibration_pairs crops/tiles from unique indexed images. Use local crop
        seeds matching the initial training epoch, without advancing the training
        cursor or any RNG. Candidates use the FLUX.1 resolution-shifted grid
        or t/1000 for other families, t=1..998. Checkpoint owns the selected
        FP64 value; resume skips calibration.
        """
        options = config.method.restoration
        if options is None or not options.calibrate:
            raise ValueError("SNR calibration requires calibrate = true")
        device = next(self.vae.parameters()).device
        started = monotonic()
        status(
            f"[train] SNR calibration: up to {options.calibration_pairs} pairs, two VAE encodes per pair; "
            f"sampling={options.calibration_sampling}, compute={self.vae.compute_precision}; training starts afterward"
        )

        def latent_pairs() -> Iterator[tuple[Tensor, Tensor]]:
            seen: set[AlignedImagePair] = set()
            count = 0
            last_report = started
            for index, pair in enumerate(dataset.pairs):
                if pair in seen:
                    continue
                seen.add(pair)
                set_context(stage="calibration image loading/degradation", source=pair.high)
                data = next(
                    group
                    for group in config.dataset.groups.values()
                    if Path(group.cache_directory) == pair.cache_directory
                )
                if options.calibration_sampling == "random":
                    pixels = iter(
                        (
                            dataset.read(
                                pair,
                                data.resolution,
                                config.train.seed + index,
                                data.horizontal_flip_probability,
                            ),
                        )
                    )
                else:
                    from unitunner.data.media.restoration import iter_aligned_images

                    pixels = iter_aligned_images(
                        pair.low,
                        pair.high,
                        resolution=data.resolution,
                        seed=config.train.seed + index,
                        flip_probability=data.horizontal_flip_probability,
                        degradation=pair.degradation,
                        sampling=options.calibration_sampling,
                    )
                for low, high in pixels:
                    set_context(stage="calibration VAE encoding", calibration_pair=count + 1)
                    yield (
                        self.encode(low.unsqueeze(0).to(device), posterior=self.posterior),
                        self.encode(high.unsqueeze(0).to(device), posterior=self.posterior),
                    )
                    count += 1
                    now = monotonic()
                    if count == 1 or count == options.calibration_pairs or now - last_report >= 5:
                        memory = (
                            f", CUDA allocated={torch.cuda.memory_allocated(device) / 2**30:.2f}GiB"
                            if device.type == "cuda"
                            else ""
                        )
                        status(
                            f"[train] SNR calibration: {count}/{options.calibration_pairs} pairs, {now - started:.1f}s{memory}"
                        )
                        last_report = now
                    set_context(stage="calibration image loading/degradation")
                    if count == options.calibration_pairs:
                        return

        with (
            calibration_rng(config.train.seed, device),
            self.encoder.unadapted() if self.encoder is not None else torch.no_grad(),
        ):
            with torch.autocast(device.type, enabled=False):
                estimate = estimate_mid_timestep(
                    latent_pairs(),
                    family=config.model.family,
                    resolution=next(iter(config.dataset.groups.values())).resolution,
                )
        self.state.get_buffer("sigma").fill_(estimate.sigma)
        status(
            f"[train] SNR calibration complete: {estimate.pairs} pairs in {monotonic() - started:.1f}s; sigma={estimate.sigma:.8g}"
        )
        clear_context("calibration_pair", "source")
        set_context(stage="calibration complete")

    def steps(
        self, low: Tensor, high: Tensor, captions: Caption | Krea2TextFeatures | FluxTextFeatures
    ) -> Iterator[StepResult]:
        """Yield G then detached D graphs in original per-microbatch forward order.

        Use the explicit posterior policy, frozen unadapted HQ encoding for LRR,
        direct LQ denoising at native time and differentiable decoding under the configured compute precision.
        With LRR enabled, draw HQ posterior noise, LRR noise, then LQ posterior
        noise, matching Trainer. Without LRR skip the unused HQ branch; that
        variant does not preserve Trainer's unused HQ RNG draws. Shared D buffers
        retain training updates.
        """
        with compute_context(low.device, self.compute_precision):
            clean = None
            if self.weights.lrr and self.encoder is not None:
                self._trace("encoding HQ target for LRR", low.device)
                with self.encoder.unadapted():
                    clean = self.encode(high, posterior=self.posterior)
            noise = torch.randn_like(clean) if clean is not None else None
            self._trace("encoding LQ image", low.device)
            low_latent = self.encode(low, posterior=self.posterior)
            if noise is None:
                noise = torch.randn_like(low_latent)
            self._trace("denoiser forward", low.device)
            restored = self.restore(low_latent, captions)
            self._trace("VAE decoder forward", low.device)
            generated = self.decode(restored).float()
            self._trace("computing restoration losses (DINO/GAN/FDL as configured)", low.device)
            logits = None
            if self.discriminator is not None:
                with frozen_discriminator(self.discriminator):
                    logits = self.discriminator(generated)
            loss = generator_loss(
                generated,
                high,
                low_latent,
                flow_target(clean, noise, self.sigma) if clean is not None else None,
                self.weights,
                perceptual=self.perceptual(generated, high) if self.perceptual is not None else None,
                frequency_distribution=self.frequency(generated, high) if self.frequency is not None else None,
                discriminator_logits=logits,
                histogram_bins=self.histogram_bins,
                phase_period=self.phase_period,
                phase_cycles=self.phase_cycles,
                phase_epsilon=self.phase_epsilon,
            )
            total = loss.total
            if self.wavelet is not None:
                wavelet = self.wavelet(generated, high.detach())
                total = total + wavelet
                loss.metrics["wavelet"] = wavelet.detach()
            self._trace("generator loss ready; backward next", low.device)
            del clean, noise, low_latent, restored, logits
            yield StepResult(total, low.shape[0], "generator", loss.metrics)
            fake = generated.detach()
            del generated, total, loss
            if self.discriminator is not None:
                self._trace("generator backward complete; discriminator forward", low.device)
                discriminator_loss = self.weights.gan * (
                    gan_loss(self.discriminator(fake), False) + gan_loss(self.discriminator(high.detach()), True)
                )
                yield StepResult(discriminator_loss, low.shape[0], "discriminator")
            self._trace("microbatch backward complete", low.device)
            self._first_step = False
