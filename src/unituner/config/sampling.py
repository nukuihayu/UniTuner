"""Declare native image sampling choices without compute backend imports."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SamplingConfig(BaseModel):
    """Configure one image in pixel units; omitted steps/guidance use native variant defaults."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid", allow_inf_nan=False)
    prompt: str = ""
    negative_prompt: str | None = None
    reference_images: tuple[str, ...] = ()
    conditioning_image: str | None = Field(default=None, min_length=1)
    mask_image: str | None = Field(default=None, min_length=1)
    condition_posterior: Literal["sample", "mean"] = "sample"
    conditioning_resize: Literal["preferred", "original", "output"] = "preferred"
    output_layers: int | None = Field(default=None, ge=0)
    caption_language: Literal["en", "zh"] | None = None
    width: int = Field(default=1024, ge=16)
    height: int = Field(default=1024, ge=16)
    seed: int = 42
    attention_backend: Literal["torch", "kitchen-int8"] = "torch"
    steps: int | None = Field(default=None, gt=0)
    shift: float | None = Field(default=None, gt=0)
    guidance: float | None = None
    training_shift: float = Field(default=1.0, gt=0)
    uncond_skip_layers: tuple[int, ...] | None = None
    slg_layers: tuple[int, ...] = ()
    slg_scale: float = Field(default=0.0, ge=0)
    slg_start: float = Field(default=0.0, ge=0, le=1)
    slg_stop: float = Field(default=1.0, ge=0, le=1)

    @model_validator(mode="after")
    def validate_resolution(self) -> Self:
        """Require dimensions divisible by the native 16-pixel patch granularity."""
        if self.width % 16 or self.height % 16:
            raise ValueError("Sampling width/height must be divisible by 16")
        if any(not path for path in self.reference_images):
            raise ValueError("Sampling reference_images must contain nonempty paths")
        if self.slg_start > self.slg_stop:
            raise ValueError("SLG requires slg_start <= slg_stop")
        if any(index < 0 for index in (*self.slg_layers, *(self.uncond_skip_layers or ()))):
            raise ValueError("Sampling skip-layer indices must be nonnegative")
        return self


def validate_image_sampling(
    sampling: SamplingConfig, *, family: str, variant: str | None = None, version: str | None = None
) -> None:
    """Reject unsupported family-specific controls before text/model artifact IO."""
    moss_controls = (
        sampling.uncond_skip_layers is not None
        or bool(sampling.slg_layers)
        or sampling.slg_scale != 0
        or sampling.slg_start != 0
        or sampling.slg_stop != 1
        or sampling.training_shift != 1
    )
    if sampling.reference_images and family not in ("flux2", "flux3", "qwen-image", "qwen-image-2-1", "moss-image"):
        raise ValueError("Sampling reference_images requires FLUX.2, FLUX3 or Qwen Image")
    if sampling.output_layers is not None and (family != "qwen-image" or variant != "layered"):
        raise ValueError("Sampling output_layers requires Qwen Image layered")
    if sampling.caption_language is not None and (family != "qwen-image" or variant != "layered"):
        raise ValueError("Sampling caption_language requires Qwen Image layered")
    if sampling.conditioning_resize == "output" and family not in ("qwen-image", "qwen-image-2-1"):
        raise ValueError("Sampling output-size references require Qwen Image")
    if family == "krea2":
        if variant not in ("raw", "turbo"):
            raise ValueError("Krea2 sampling requires raw or turbo variant")
        if moss_controls:
            raise ValueError("Krea2 does not support MOSS skip/SLG/training_shift options")
        if sampling.shift is not None and sampling.shift > 80:
            raise ValueError("Krea2 shift is a native log-shift and must not exceed80")
    if family == "qwen-image-2-1":
        if variant != "base" or sampling.width % 32 or sampling.height % 32:
            raise ValueError("Qwen Image2.1 sampling requires base variant and32-aligned dimensions")
        if sampling.conditioning_image is not None or sampling.condition_posterior != "sample":
            raise ValueError("Qwen Image2.1 uses reference_images and posterior means")
        if sampling.mask_image is not None and not sampling.reference_images:
            raise ValueError("Qwen Image2.1 inpainting requires a first reference image")
        if moss_controls:
            raise ValueError("Qwen Image2.1 does not support MOSS skip/SLG/training_shift options")
    if family == "qwen-image":
        if variant not in ("original", "edit", "edit-2509", "edit-2511", "layered"):
            raise ValueError("Qwen Image sampling requires an original/edit/layered variant")
        if sampling.steps is not None and sampling.steps < 2:
            raise ValueError("Qwen Image sampling requires at least two steps")
        if variant == "original" and sampling.reference_images:
            raise ValueError("Qwen original does not consume reference images")
        if variant.startswith("edit") and not sampling.reference_images:
            raise ValueError("Qwen edit requires reference_images")
        if variant == "layered" and len(sampling.reference_images) != 1:
            raise ValueError("Qwen layered requires exactly one source reference")
        if sampling.mask_image is not None and not variant.startswith("edit"):
            raise ValueError("Qwen inpainting masks require an edit variant")
        if sampling.conditioning_image is not None or sampling.condition_posterior != "sample":
            raise ValueError("Qwen uses reference_images and posterior means; Flux condition options do not apply")
        if moss_controls:
            raise ValueError("Qwen sampling does not support MOSS skip/SLG/training_shift options")
    if family == "flux2":
        if version not in ("dev", "klein-4b", "klein-9b") or variant not in ("base", "distilled"):
            raise ValueError("FLUX.2 sampling requires version and base/distilled variant")
        if version == "dev" and variant != "distilled":
            raise ValueError("FLUX.2 Dev requires its guidance-distilled variant")
        if moss_controls:
            raise ValueError("FLUX.2 sampling does not support MOSS skip/SLG/training_shift options")
        if variant == "distilled" and sampling.negative_prompt is not None:
            raise ValueError("FLUX.2 distilled does not consume negative prompts")
        if (
            version != "dev"
            and variant == "distilled"
            and (sampling.steps not in (None, 4) or sampling.guidance not in (None, 1.0))
        ):
            raise ValueError("FLUX.2 distilled Klein fixes steps4 and guidance1")
        if sampling.guidance is not None and sampling.guidance < 0:
            raise ValueError("FLUX.2 guidance must be nonnegative")
    if family == "flux3":
        if len(sampling.reference_images) > 10:
            raise ValueError("FLUX3 supports at most ten reference images")
        if variant not in ("base", "distilled"):
            raise ValueError("FLUX3 sampling requires base or distilled variant")
        if moss_controls:
            raise ValueError("FLUX3 sampling does not support MOSS skip/SLG/training_shift options")
        if variant == "distilled" and (sampling.negative_prompt is not None or sampling.guidance is not None):
            raise ValueError("FLUX3 distilled does not consume negative prompts or CFG guidance")
        if sampling.guidance is not None and sampling.guidance < 0:
            raise ValueError("FLUX3 CFG guidance must be nonnegative")
    if family == "z-image" and moss_controls:
        raise ValueError("Z-Image sampling does not support MOSS skip/SLG/training_shift options")
    if family == "moss-image" and bool(sampling.reference_images) != (variant == "edit"):
        raise ValueError("MOSS edit requires reference_images; base/sft do not accept them")
    if family == "moss-image" and sampling.guidance is not None and sampling.guidance < 0:
        raise ValueError("MOSS sampling guidance must be nonnegative")
    if family not in ("flux", "qwen-image", "qwen-image-2-1") and (
        sampling.conditioning_image is not None
        or sampling.mask_image is not None
        or sampling.condition_posterior != "sample"
    ):
        raise ValueError("Sampling condition images/masks currently require Flux")
    if family not in ("flux", "flux2", "qwen-image", "qwen-image-2-1") and sampling.conditioning_resize != "preferred":
        raise ValueError("conditioning_resize requires Flux Kontext or FLUX.2")
    if family == "flux":
        if sampling.negative_prompt is not None or moss_controls:
            raise ValueError("Flux sampling does not support negative prompts or MOSS skip/SLG/training_shift options")
        if sampling.guidance is not None and sampling.guidance < 0:
            raise ValueError("Flux embedded guidance must be nonnegative")
        if variant == "dev" and sampling.conditioning_image is not None:
            raise ValueError("Flux Dev does not accept a condition image")
        if variant in ("kontext", "fill") and sampling.conditioning_image is None:
            raise ValueError("Flux Kontext/Fill sampling requires conditioning_image")
        if sampling.mask_image is not None and (variant != "fill" or sampling.conditioning_image is None):
            raise ValueError("Flux sampling masks require a Fill condition image")
        if sampling.conditioning_resize != "preferred" and variant != "kontext":
            raise ValueError("conditioning_resize applies only to Flux Kontext")
