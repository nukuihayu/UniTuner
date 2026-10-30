"""Declare shared dataset conditions independently of native cache encoders."""

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DatasetInputConfig(BaseModel):
    """Describe ordered stem-matched references, masks, layers and aligned controls.

    Paths identify directories matched to target filename stems, not independent
    training samples. Repeated references retain their order. Training references
    independently use the dataset bucket settings. Equal source dimensions
    yield aligned geometry. Qwen2.1 can override the reference area budget.
    Capability checks reject unsupported conditions before artifact IO.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    reference_sources: tuple[str, ...] = ()
    reference_resolution: tuple[int, int] | None = Field(
        default=None,
        description="Qwen2.1 independent reference area budget (width,height); omitted references use the dataset budget and their own aspect",
    )
    control_source: str | None = Field(default=None, min_length=1)
    mask_source: str | None = Field(default=None, min_length=1)
    layer_sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        """Preserve ordered references and reject ambiguous image relationships."""
        if any(not path for path in (*self.reference_sources, *self.layer_sources)):
            raise ValueError("dataset reference_sources/layer_sources require nonempty paths")
        if self.reference_sources and self.layer_sources:
            raise ValueError("dataset reference_sources and layer_sources are mutually exclusive")
        if self.reference_resolution is not None:
            if not self.reference_sources:
                raise ValueError("dataset.reference_resolution requires reference_sources")
            if any(value < 32 or value % 32 for value in self.reference_resolution):
                raise ValueError("dataset.reference_resolution dimensions must be positive multiples of32")
        return self

    def validate_for(self, family: str, variant: str, *, kind: str = "image") -> None:
        """Check model capabilities without loading weights or touching source files."""
        references = self.reference_sources
        if self.reference_resolution is not None and (family != "qwen-image-2-1" or kind != "image"):
            raise ValueError("dataset.reference_resolution currently requires Qwen Image2.1 image data")
        if family == "flux":
            if references and (variant == "dev" or len(references) != 1):
                raise ValueError("Flux Kontext/Fill accepts exactly one reference source; Dev accepts none")
        elif family == "moss-image":
            if references and variant != "edit":
                raise ValueError("MOSS reference sources require model.variant=edit")
            if variant == "edit" and not references:
                raise ValueError("MOSS edit requires reference_sources")
        elif family == "flux3":
            if len(references) > 10:
                raise ValueError("FLUX3 accepts at most ten reference sources")
        elif family == "qwen-image":
            if references and variant in ("original", "layered"):
                raise ValueError("Qwen original/layered does not consume edit references")
        elif references and family not in ("flux2", "qwen-image-2-1"):
            raise ValueError(f"{family} does not support dataset.reference_sources")
        if self.control_source is not None and family != "wan":
            raise ValueError(f"{family} does not support aligned dataset.control_source")
        if self.mask_source is not None and (family, variant) != ("flux", "fill"):
            raise ValueError("dataset.mask_source requires Flux Fill")
        if self.layer_sources and (family, variant) != ("qwen-image", "layered"):
            raise ValueError("dataset.layer_sources requires Qwen layered")
        if kind != "image" and (references or self.mask_source is not None or self.layer_sources):
            raise ValueError("Reference/mask/layer directories currently require image data")
        return
