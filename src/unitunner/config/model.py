"""Share model selection and encoder artifact validation without training requirements."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitunner.config.text import TextEncodingConfig
from unitunner.models.catalog import ModelFamily
from unitunner.models.minimax_h3.config.training import MiniMaxH3EncoderConfig


class EncoderModelConfig(BaseModel):
    """Select an exact model variant and external weight source."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    family: ModelFamily
    variant: str = Field(min_length=1)
    version: Literal["dev", "klein-4b", "klein-9b", "2.1", "2.2"] | None = None
    vae: str | None = None
    text_encoder: str | None = None
    assets: str | None = Field(default=None, min_length=1)
    text: TextEncodingConfig = Field(default_factory=TextEncodingConfig)
    minimax_h3: MiniMaxH3EncoderConfig | None = None
    text_source_format: Literal["auto", "transformers", "qwen"] | None = None

    def asset_path(self, component: Literal["clip_text", "tokenizer", "image_encoder"]) -> Path:
        """Locate a required auxiliary asset in the declared local bundle, without IO.

        Encoder-only commands require an explicit bundle root. Component paths
        follow a fixed layout; no network lookup or ambiguous filename search is
        performed. Native Wan vision weights live in image_encoder/model.safetensors.
        Callers request only components actually used by the selected variant.
        """
        if self.assets is None:
            raise ValueError("model.assets is required to locate auxiliary encoder components")
        root = Path(self.assets)
        relative = {
            "clip_text": "text_encoder",
            "tokenizer": "tokenizer",
            "image_encoder": "image_encoder/model.safetensors",
        }
        return root / relative[component]

    @model_validator(mode="after")
    def validate_version(self) -> Self:
        """Require FLUX.2 size/version independently of checkpoint distillation."""
        if self.minimax_h3 is not None and self.family != ModelFamily.MINIMAX_H3:
            raise ValueError("model.minimax_h3 requires MiniMaxH3")
        if self.family == ModelFamily.MINIMAX_H3 and self.variant not in ("t2va", "fl2va", "ref2va"):
            raise ValueError("MiniMaxH3 requires model.variant=t2va/fl2va/ref2va")
        if self.text_source_format is not None and self.family not in {
            ModelFamily.QWEN_IMAGE,
            ModelFamily.QWEN_IMAGE_2_1,
            ModelFamily.KREA2,
            ModelFamily.MINIMAX_H3,
        }:
            raise ValueError("model.text_source_format requires Qwen Image, Krea2 or MiniMaxH3")
        if self.family == ModelFamily.MOSS_IMAGE and self.variant not in ("base", "sft", "edit"):
            raise ValueError("MOSS requires model.variant=base/sft/edit")
        self.text.validate_for(self.family)
        if self.family == ModelFamily.KREA2 and self.variant not in ("raw", "turbo"):
            raise ValueError("Krea2 requires model.variant=raw/turbo")
        if self.family == ModelFamily.QWEN_IMAGE_2_1 and self.variant != "base":
            raise ValueError("Qwen Image2.1 requires model.variant=base")
        if self.family == ModelFamily.FLUX2:
            if self.version not in ("dev", "klein-4b", "klein-9b") or self.variant not in ("base", "distilled"):
                raise ValueError("FLUX.2 requires model.version and base/distilled variant")
            if self.version == "dev" and self.variant != "distilled":
                raise ValueError("FLUX.2 Dev only publishes guidance-distilled weights")
        elif self.family == ModelFamily.WAN:
            if self.version not in ("2.1", "2.2") or self.variant not in ("t2v", "i2v"):
                raise ValueError("Wan requires model.version=2.1/2.2 and model.variant=t2v/i2v")
        elif self.version is not None:
            raise ValueError("model.version currently applies only to FLUX.2 and Wan")
        return self


class ModelConfig(EncoderModelConfig):
    """Require denoiser artifacts only for training."""

    weights: str = Field(min_length=1)

    def asset_path(self, component: Literal["clip_text", "tokenizer", "image_encoder"]) -> Path:
        """Use the explicit bundle or the denoiser parent for training assets."""
        model = self.model_copy(update={"assets": self.assets or str(Path(self.weights).parent)})
        return EncoderModelConfig.asset_path(model, component)
