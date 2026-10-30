"""Publish cached-text previews for FLUX, FLUX.2, FLUX3, MOSS and Z-Image."""

from pathlib import Path
from typing import TYPE_CHECKING, cast

import torch

from unituner.config.sampling import SamplingConfig
from unituner.config.training import TrainingConfig
from unituner.data.media.image import save_image
from unituner.inference.decoding import decode_on_device
from unituner.inference.sampling import generate_latents
from unituner.models.components.conditioning import Caption
from unituner.models.components.image_encoders import load_image_text, load_image_vae
from unituner.models.flux.conditioning import FluxTextFeatures
from unituner.models.flux.inference.conditioning import build_flux_condition
from unituner.models.flux.transformer import FluxTransformer
from unituner.models.flux.vae import FluxVAE
from unituner.models.flux2.inference.conditioning import build_flux2_references
from unituner.models.flux2.transformer import Flux2Transformer
from unituner.models.flux2.vae import Flux2VAE
from unituner.models.flux3.inference.conditioning import build_flux3_references
from unituner.models.flux3.transformer import Flux3Transformer
from unituner.models.moss_image.inference.conditioning import encode_references
from unituner.models.moss_image.transformer import MossImageTransformer
from unituner.models.z_image.transformer import ZImageTransformer
from unituner.models.z_image.vae import ZImageVAE
from unituner.training.previews.publication import CaptionCache, publish_images

if TYPE_CHECKING:
    from unituner.models.components.image_encoders import PromptTextEncoder
    from unituner.models.flux.text.encoder import FluxTextEncoder

type CachedImageModel = FluxTransformer | Flux2Transformer | Flux3Transformer | MossImageTransformer | ZImageTransformer
type PreviewCaptions = tuple[tuple[Caption | FluxTextFeatures, Caption | None], ...]


class ImagePreviews(CaptionCache[PreviewCaptions]):
    """Cache CPU captions and borrow the live model for each completed update."""

    def __init__(self, config: TrainingConfig, model: CachedImageModel) -> None:
        families = (
            (FluxTransformer, "flux"),
            (Flux2Transformer, "flux2"),
            (Flux3Transformer, "flux3"),
            (MossImageTransformer, "moss-image"),
            (ZImageTransformer, "z-image"),
        )
        if not any(isinstance(model, cls) and config.model.family == family for cls, family in families):
            raise ValueError("Preview configuration and native model family differ")
        if config.model.variant != model.config.variant:
            raise ValueError("Preview configuration and artifact variant differ")
        if isinstance(model, Flux2Transformer) and config.model.version != model.config.version:
            raise ValueError("Preview configuration and artifact version differ")
        self.config = config
        self.model = model
        self.captions = None

    def load_captions(self) -> PreviewCaptions:
        """Encode only the selected family's text features on CPU."""
        config, model = self.config, self.model
        if config.preview is None or config.model.text_encoder is None:
            raise ValueError("Image previews require text assets and samples")
        text = load_image_text(config.model, device=torch.device("cpu"))
        if isinstance(model, FluxTransformer):
            flux_text = cast("FluxTextEncoder", text)
            return tuple((flux_text.encode([sample.prompt]), None) for sample in config.preview.samples)
        encoder = cast("PromptTextEncoder", text)
        default_negative = " " if isinstance(model, Flux2Transformer) else ""
        requires_negative = isinstance(model, MossImageTransformer) or (
            isinstance(model, Flux2Transformer | Flux3Transformer) and model.config.variant == "base"
        )
        return tuple(
            (
                encoder.encode_prompt(sample.prompt),
                encoder.encode_prompt(default_negative if sample.negative_prompt is None else sample.negative_prompt)
                if requires_negative or sample.negative_prompt is not None
                else None,
            )
            for sample in config.preview.samples
        )

    def load_vae(self) -> tuple[FluxVAE | Flux2VAE | ZImageVAE, int]:
        """Load the frozen CPU decoder and validate native latent channels."""
        config, model = self.config, self.model
        if config.model.vae is None:
            raise ValueError("Image previews require VAE weights")
        vae = cast(
            FluxVAE | Flux2VAE | ZImageVAE,
            load_image_vae(Path(config.model.vae), family=config.model.family, device=torch.device("cpu")),
        )
        if isinstance(model, FluxTransformer):
            vae = cast(FluxVAE, vae)
            channels, vae_channels = model.config.out_channels // 4, vae.config.latent_channels
            factor = vae.config.spatial_factor
        elif isinstance(model, ZImageTransformer):
            vae = cast(ZImageVAE, vae)
            channels, vae_channels = model.config.in_channels, vae.config.latent_channels
            factor = 2 ** (len(vae.config.block_channels) - 1)
        else:
            vae = cast(Flux2VAE, vae)
            if isinstance(model, Flux2Transformer):
                channels = model.config.in_channels
            elif isinstance(model, Flux3Transformer):
                channels = model.config.image_channels
            else:
                channels = model.config.latent_channels
            vae_channels = 4 * vae.config.latent_channels
            factor = vae.config.spatial_factor
        if channels != vae_channels:
            raise ValueError("Preview vae/model latent channels differ")
        return vae, factor

    def __call__(self, update: int) -> None:
        """Render samples in order using native conditioning and pixel rounding."""
        config, model = self.config, self.model
        if config.preview is None or config.model.vae is None or config.model.text_encoder is None:
            raise ValueError("Image previews require samples, vae and text assets")
        device = next(model.parameters()).device
        captions = self.cached_captions(self.load_captions)
        vae, factor = self.load_vae()

        def render(
            entry: tuple[SamplingConfig, tuple[Caption | FluxTextFeatures, Caption | None]],
        ) -> tuple[torch.Tensor, ...]:
            sample, (positive, negative) = entry
            condition, references = None, ()
            if isinstance(model, FluxTransformer):
                condition = build_flux_condition(cast(FluxVAE, vae), sample, variant=model.config.variant)
            elif isinstance(model, Flux2Transformer):
                references = build_flux2_references(
                    cast(Flux2VAE, vae),
                    tuple(Path(path) for path in sample.reference_images),
                    original=sample.conditioning_resize == "original",
                )
            elif isinstance(model, Flux3Transformer):
                references = build_flux3_references(
                    cast(Flux2VAE, vae), tuple(Path(path) for path in sample.reference_images)
                )
            elif isinstance(model, MossImageTransformer):
                references = encode_references(cast(Flux2VAE, vae), sample.reference_images, device=device)
            latents = generate_latents(
                model,
                positive,
                sample,
                negative=negative,
                condition=condition,
                references=references,
                vae_factor=factor,
                allow_cpu_frozen=config.execution.offloads_frozen_weights,
            )
            return (decode_on_device(vae, latents, device=device)[0],)

        publish_images(
            zip(config.preview.samples, captions, strict=True),
            render,
            directory=Path(config.output.directory) / "previews",
            update=update,
            rounding="truncate" if isinstance(model, Flux2Transformer | Flux3Transformer) else "nearest",
            save=save_image,
        )
