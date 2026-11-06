"""Declare model composition and application providers without importing compute backends."""

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal


class ModelFamily(StrEnum):
    """Canonical family identities independent of Python import paths."""

    Z_IMAGE = "z-image"
    MOSS_IMAGE = "moss-image"
    FLUX = "flux"
    FLUX2 = "flux2"
    FLUX3 = "flux3"
    QWEN_IMAGE = "qwen-image"
    QWEN_IMAGE_2_1 = "qwen-image-2-1"
    KREA2 = "krea2"
    BLUEPIXEL = "bluepixel"
    WAN = "wan"
    MINIMAX_H3 = "minimax-h3"


@dataclass(frozen=True, slots=True, kw_only=True)
class ImageFamilySpec:
    """Compose public identity, reusable components and explicit lazy application entries.

    Backbone selects parameter layouts and execution policy. VAE and text select
    independent encoders. Fixed native selections describe derivatives such as
    BluePixel without changing the public TOML family/variant or cache identity.
    """

    family: ModelFamily
    backbone: ModelFamily
    vae: ModelFamily
    text: ModelFamily
    loader: str
    saver: str
    binding: str
    projections: str
    generation: str
    preview: str
    adapter_base: str
    adapter_module: str
    loader_variant: bool = True
    loader_version: bool = False
    native_variant: str | None = None
    native_version: str | None = None
    variant_in_weights: bool = True
    cache_alignment: int = 16
    restoration_training_alignment: int = 16
    restoration_variants: tuple[str, ...] | None = None
    latent_channels_field: Literal["in_channels", "out_channels", "image_channels", "latent_channels"] = "in_channels"
    latent_channels_divisor: int = 1
    inference_precision: Literal["fp32", "bf16"] = "fp32"
    tokenizer_in_assets: bool = False
    adapter_namespace: Literal["transformer", "diffusion_model"] = "transformer"

    def selected_variant(self, variant: str) -> str:
        """Map the public checkpoint selection to its native parameter topology."""
        return self.native_variant or variant

    def selected_version(self, version: str | None) -> str | None:
        """Resolve the text/backbone version while retaining the public identity."""
        return self.native_version or version


_SHARED_GENERATION = "unituner.inference.images:run"
_SHARED_PREVIEW = "unituner.training.previews.flow:ImagePreviews"

IMAGE_SPECS = MappingProxyType(
    {
        spec.family: spec
        for spec in (
            ImageFamilySpec(
                family=ModelFamily.Z_IMAGE,
                backbone=ModelFamily.Z_IMAGE,
                vae=ModelFamily.Z_IMAGE,
                text=ModelFamily.Z_IMAGE,
                loader="unituner.models.z_image.loading:load_z_image",
                saver="unituner.models.z_image.loading:save_z_image",
                binding="unituner.training.integrations.z_image.binding:ZImageBinding",
                projections="unituner.models.z_image.projections:adapter_projections",
                generation=_SHARED_GENERATION,
                preview=_SHARED_PREVIEW,
                adapter_base="z_image",
                adapter_module="networks.lora_zimage",
            ),
            ImageFamilySpec(
                family=ModelFamily.FLUX,
                backbone=ModelFamily.FLUX,
                vae=ModelFamily.FLUX,
                text=ModelFamily.FLUX,
                loader="unituner.models.flux.loading:load_flux",
                saver="unituner.models.flux.loading:save_flux",
                binding="unituner.training.integrations.flux.binding:FluxBinding",
                projections="unituner.models.flux.projections:adapter_projections",
                generation=_SHARED_GENERATION,
                preview=_SHARED_PREVIEW,
                adapter_base="flux",
                adapter_module="networks.lora_flux",
                restoration_variants=("dev",),
                latent_channels_field="out_channels",
                latent_channels_divisor=4,
            ),
            ImageFamilySpec(
                family=ModelFamily.FLUX2,
                backbone=ModelFamily.FLUX2,
                vae=ModelFamily.FLUX2,
                text=ModelFamily.FLUX2,
                loader="unituner.models.flux2.loading:load_flux2",
                saver="unituner.models.flux2.loading:save_flux2",
                binding="unituner.training.integrations.flux2.binding:Flux2Binding",
                projections="unituner.models.flux2.projections:adapter_projections",
                generation=_SHARED_GENERATION,
                preview=_SHARED_PREVIEW,
                adapter_base="flux2",
                adapter_module="networks.lora_flux2",
                loader_version=True,
            ),
            ImageFamilySpec(
                family=ModelFamily.FLUX3,
                backbone=ModelFamily.FLUX3,
                vae=ModelFamily.FLUX2,
                text=ModelFamily.FLUX3,
                loader="unituner.models.flux3.loading:load_flux3",
                saver="unituner.models.flux3.loading:save_flux3",
                binding="unituner.training.integrations.flux3.binding:Flux3Binding",
                projections="unituner.models.flux3.projections:adapter_projections",
                generation=_SHARED_GENERATION,
                preview=_SHARED_PREVIEW,
                adapter_base="flux3",
                adapter_module="networks.lora_flux3",
                latent_channels_field="image_channels",
                adapter_namespace="diffusion_model",
            ),
            ImageFamilySpec(
                family=ModelFamily.MOSS_IMAGE,
                backbone=ModelFamily.MOSS_IMAGE,
                vae=ModelFamily.FLUX2,
                text=ModelFamily.MOSS_IMAGE,
                loader="unituner.models.moss_image.loading:load_moss_image",
                saver="unituner.models.moss_image.loading:save_moss_image",
                binding="unituner.training.integrations.moss_image.binding:MossImageBinding",
                projections="unituner.models.moss_image.projections:adapter_projections",
                generation=_SHARED_GENERATION,
                preview=_SHARED_PREVIEW,
                adapter_base="moss_image",
                adapter_module="networks.lora_moss_image",
                restoration_variants=("base", "sft"),
                latent_channels_field="latent_channels",
                adapter_namespace="diffusion_model",
            ),
            ImageFamilySpec(
                family=ModelFamily.QWEN_IMAGE,
                backbone=ModelFamily.QWEN_IMAGE,
                vae=ModelFamily.QWEN_IMAGE,
                text=ModelFamily.QWEN_IMAGE,
                loader="unituner.models.qwen_image.loading:load_qwen_image",
                saver="unituner.models.qwen_image.loading:save_qwen_image",
                binding="unituner.training.integrations.qwen_image.binding:QwenImageBinding",
                projections="unituner.models.qwen_image.projections:adapter_projections",
                generation="unituner.models.qwen_image.inference.run:run",
                preview="unituner.training.integrations.qwen_image.previews:ImagePreviews",
                adapter_base="qwen_image",
                adapter_module="networks.lora_qwen_image",
                restoration_variants=("original",),
                latent_channels_field="out_channels",
            ),
            ImageFamilySpec(
                family=ModelFamily.QWEN_IMAGE_2_1,
                backbone=ModelFamily.QWEN_IMAGE_2_1,
                vae=ModelFamily.QWEN_IMAGE_2_1,
                text=ModelFamily.QWEN_IMAGE_2_1,
                loader="unituner.models.qwen_image_2_1.loading:load_qwen_image_21",
                saver="unituner.models.qwen_image_2_1.loading:save_qwen_image_21",
                binding="unituner.training.integrations.qwen_image_2_1.binding:QwenImage21Binding",
                projections="unituner.models.qwen_image_2_1.projections:adapter_projections",
                generation="unituner.models.qwen_image_2_1.inference.run:run",
                preview="unituner.training.integrations.qwen_image_2_1.previews:ImagePreviews",
                adapter_base="qwen_image_21",
                adapter_module="networks.lora_qwen_image_21",
                loader_variant=False,
                variant_in_weights=False,
                cache_alignment=32,
                latent_channels_field="out_channels",
            ),
            ImageFamilySpec(
                family=ModelFamily.KREA2,
                backbone=ModelFamily.KREA2,
                vae=ModelFamily.QWEN_IMAGE,
                text=ModelFamily.KREA2,
                loader="unituner.models.krea2.loading:load_krea2",
                saver="unituner.models.krea2.loading:save_krea2",
                binding="unituner.training.integrations.krea2.binding:Krea2Binding",
                projections="unituner.models.krea2.projections:adapter_projections",
                generation="unituner.models.krea2.inference.run:run",
                preview="unituner.training.integrations.krea2.previews:ImagePreviews",
                adapter_base="krea2",
                adapter_module="networks.lora_krea2",
            ),
            ImageFamilySpec(
                family=ModelFamily.BLUEPIXEL,
                backbone=ModelFamily.FLUX2,
                vae=ModelFamily.BLUEPIXEL,
                text=ModelFamily.FLUX2,
                loader="unituner.models.bluepixel.loading:load_bluepixel",
                saver="unituner.models.bluepixel.loading:save_bluepixel",
                binding="unituner.training.integrations.flux2.binding:Flux2Binding",
                projections="unituner.models.flux2.projections:adapter_projections",
                generation="unituner.models.bluepixel.inference.run:run",
                preview=_SHARED_PREVIEW,
                adapter_base="bluepixel_turbo",
                adapter_module="networks.lora_flux2",
                loader_variant=False,
                native_variant="base",
                native_version="klein-9b",
                cache_alignment=64,
                restoration_training_alignment=64,
                inference_precision="bf16",
                tokenizer_in_assets=True,
            ),
        )
    }
)

IMAGE_FAMILIES = frozenset(IMAGE_SPECS)


def image_family_spec(family: ModelFamily | str) -> ImageFamilySpec:
    """Resolve an image family explicitly; video families have their own application contracts."""
    selected = ModelFamily(family)
    if selected not in IMAGE_SPECS:
        raise ValueError(f"No image provider for {selected}")
    return IMAGE_SPECS[selected]


def cache_alignment(family: ModelFamily | str) -> int:
    """Declare pixel alignment once for image and video cache configuration."""
    selected = ModelFamily(family)
    if selected == ModelFamily.MINIMAX_H3:
        return 32
    if selected == ModelFamily.WAN:
        return 16
    return image_family_spec(selected).cache_alignment


def backbone_family(family: str) -> str:
    """Resolve image parameter layouts while retaining non-image artifact namespaces."""
    return image_family_spec(family).backbone if family in IMAGE_FAMILIES else family


def supports_restoration(family: ModelFamily, variant: str | None = None) -> bool:
    """Check native image restoration support, optionally for a selected variant."""
    if family not in IMAGE_FAMILIES:
        return False
    variants = image_family_spec(family).restoration_variants
    return variant is None or variants is None or variant in variants
