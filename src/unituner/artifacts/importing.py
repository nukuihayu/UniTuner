"""Import explicitly described external weights into reloadable native artifacts."""

import tempfile
from importlib import import_module
from pathlib import Path
from typing import Literal

from unituner.artifacts.tensors import resolve_shards
from unituner.models.flux.config.vae import FluxVAEConfig
from unituner.models.flux.loading import load_flux_vae, save_flux_vae
from unituner.models.flux2.config.vae import Flux2VAEConfig
from unituner.models.flux2.loading import load_flux2_vae, save_flux2_vae
from unituner.models.qwen_image.config.vae import QwenImageVAEConfig
from unituner.models.qwen_image.loading import (
    load_qwen_image_vae,
    save_qwen_image_vae,
)
from unituner.models.qwen_image_2_1.config.vae import QwenImage21VAEConfig
from unituner.models.qwen_image_2_1.loading import (
    load_qwen_image_21_vae,
    save_qwen_image_21_vae,
)
from unituner.models.z_image.config.vae import ZImageVAEConfig
from unituner.models.z_image.loading import load_z_image_vae, save_z_image_vae


def _import_transformer(source, destination, *, family, topology, source_format, prefix, fused_qkv, output_format):
    from unituner.artifacts.comfyui_weights import TRANSFORMER_TYPES
    from unituner.artifacts.conversion import convert_weights

    if family not in TRANSFORMER_TYPES or source_format not in ("auto", "native", "comfyui", "external", "diffusers"):
        raise ValueError("Unsupported transformer family or source format")
    if output_format not in ("native", "comfyui") or prefix == "vae.":
        raise ValueError("Transformer import requires native/comfyui output and a transformer prefix")
    protected = (*resolve_shards(source), *((topology,) if topology is not None else ()))
    if destination.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Weight import destination must differ from all source files")
    if fused_qkv and family != "moss-image":
        raise ValueError("--fused-qkv is a MOSS transformer import option")
    module, stem = TRANSFORMER_TYPES[family]
    config = None
    if topology is not None:
        cls = getattr(import_module(f"unituner.models.{module}.config.model"), stem + "Config")
        config = cls.model_validate_json(topology.read_text(encoding="utf-8"))
    loading = import_module(f"unituner.models.{module}.loading")
    suffix = "qwen_image_21" if family == "qwen-image-2-1" else module
    options = {}
    if family in ("moss-image", "qwen-image"):
        options["prefix"] = prefix
        if family == "moss-image":
            options["fused_qkv"] = fused_qkv
            options["source_format"] = "native" if source_format == "native" else "auto"
        else:
            options["source_format"] = "comfyui" if prefix or source_format == "external" else source_format
    model = getattr(loading, "load_" + suffix)(source, config=config, **options)
    save = getattr(loading, "save_" + suffix)
    if output_format == "native":
        save(model, destination)
    else:
        with tempfile.TemporaryDirectory(prefix="unituner-import-") as temporary:
            native = Path(temporary) / "model.safetensors"
            save(model, native)
            del model
            convert_weights(native, destination, precision="keep", family=family, output_format="comfyui")
    return 0


def import_weights(
    source: Path,
    destination: Path,
    *,
    family: Literal[
        "z-image", "moss-image", "flux", "flux2", "flux3", "qwen-image", "qwen-image-2-1", "krea2", "wan", "minimax-h3"
    ] = "z-image",
    kind: Literal["transformer", "vae"],
    topology: Path | None = None,
    source_format: Literal["auto", "native", "comfyui", "external", "diffusers"] = "auto",
    prefix: Literal["", "model.diffusion_model.", "diffusion_model.", "vae."] = "",
    fused_qkv: bool = False,
    output_format: Literal["native", "comfyui"] = "comfyui",
) -> int:
    """Validate external weights and atomically publish a consumer artifact.

    Transformer imports default to ComfyUI keys. Native metadata and local
    configuration are optional for released architectures. VAE imports use
    their component loaders and retain the native component layout.
    """
    if kind == "transformer":
        return _import_transformer(
            source,
            destination,
            family=family,
            topology=topology,
            source_format=source_format,
            prefix=prefix,
            fused_qkv=fused_qkv,
            output_format=output_format,
        )
    if kind != "vae" or family == "minimax-h3":
        raise ValueError("Weight import requires a supported family and transformer/vae kind")
    if fused_qkv or prefix not in ("", "vae."):
        raise ValueError("VAE import requires an empty/vae. prefix without QKV conversion")
    protected = (*resolve_shards(source), *((topology,) if topology is not None else ()))
    if destination.resolve() in {path.resolve() for path in protected}:
        raise ValueError("Weight import destination must differ from all source files")
    serialized = None if topology is None else topology.read_text(encoding="utf-8")
    if family in ("wan", "krea2", "qwen-image"):
        if source_format not in ("auto", "native", "comfyui", "external"):
            raise ValueError("Qwen/Wan VAE requires native/comfyui format")
        qwen_config = None if serialized is None else QwenImageVAEConfig.model_validate_json(serialized)
        if (
            family == "wan"
            and qwen_config is not None
            and (
                qwen_config.input_channels != 3
                or qwen_config.latent_channels != 16
                or qwen_config.channel_multipliers != (1, 2, 4, 4)
                or qwen_config.temporal_downsample != (False, True, True)
                or qwen_config.attention_scales
            )
        ):
            raise ValueError(
                "Wan shared vae requires RGB/16 channels, native four-stage geometry and no stage attention"
            )
        if (
            family == "krea2"
            and qwen_config is not None
            and (
                qwen_config.input_channels != 3 or qwen_config.latent_channels != 16 or qwen_config.spatial_factor != 8
            )
        ):
            raise ValueError("Krea2 requires the shared RGB Qwen vae with16 latent channels and spatial factor8")
        qwen = load_qwen_image_vae(
            source,
            config=qwen_config,
            source_format="auto" if source_format == "auto" else "native" if source_format == "native" else "comfyui",
            prefix=prefix,
        )
        save_qwen_image_vae(qwen, destination)
    elif family == "qwen-image-2-1":
        qwen21 = load_qwen_image_21_vae(
            source,
            config=None if serialized is None else QwenImage21VAEConfig.model_validate_json(serialized),
            source_format="comfyui" if source_format == "external" else source_format,
            prefix=prefix,
        )
        save_qwen_image_21_vae(qwen21, destination)
    elif family == "flux":
        if source_format not in ("auto", "native", "diffusers"):
            raise ValueError("FLUX VAE requires native/diffusers format")
        flux = load_flux_vae(
            source,
            config=None if serialized is None else FluxVAEConfig.model_validate_json(serialized),
            source_format=source_format,
            prefix=prefix,
        )
        save_flux_vae(flux, destination)
    elif family in ("moss-image", "flux2", "flux3"):
        if source_format not in ("auto", "native", "diffusers"):
            raise ValueError("FLUX.2 VAE requires native/diffusers format")
        flux2 = load_flux2_vae(
            source,
            config=None if serialized is None else Flux2VAEConfig.model_validate_json(serialized),
            source_format=source_format,
            prefix=prefix,
        )
        save_flux2_vae(flux2, destination)
    elif family == "z-image":
        if source_format not in ("auto", "native", "comfyui") or prefix:
            raise ValueError("Z-Image VAE requires native/comfyui format and no prefix")
        z = load_z_image_vae(
            source,
            config=None if serialized is None else ZImageVAEConfig.model_validate_json(serialized),
            source_format=source_format,
        )
        save_z_image_vae(z, destination)
    else:
        raise ValueError("Unsupported VAE family")
    return 0
