"""Validate cache production independently of training adapters, losses and execution."""

from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from unitunner.config.datasets import CacheDataConfig, CacheDatasetConfig, DatasetConfig
from unitunner.config.diagnostics import ConfigRuleError
from unitunner.config.features import RepaConfig
from unitunner.config.model import EncoderModelConfig
from unitunner.config.runtime import RuntimeConfig
from unitunner.models.catalog import ModelFamily


class CacheOptions(BaseModel):
    """Set cache encoder batch size independently of training microbatches."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    batch_size: int = Field(default=1, gt=0)


class CacheConfig(BaseModel):
    """Own only encoder artifacts, input groups and optional feature targets."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    model: EncoderModelConfig
    dataset: CacheDataConfig
    cache: CacheOptions = Field(default_factory=CacheOptions)
    seed: int = 42
    repa: RepaConfig | None = None

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        """Reject model/data combinations before loading any encoder."""
        model = self.model
        if self.cache.batch_size > 1 and model.family != ModelFamily.Z_IMAGE:
            raise ConfigRuleError(
                "cache.batch_size",
                "Batched caching currently supports Z-Image only",
                "Use cache.batch_size=1 for this family.",
            )
        if model.vae is None or model.text_encoder is None:
            raise ConfigRuleError(
                "model",
                "Cache production requires model.vae and model.text_encoder",
                "Set both model.vae and model.text_encoder checkpoint paths before caching.",
            )
        if (
            model.family == ModelFamily.FLUX
            or (model.family == ModelFamily.WAN and model.version == "2.1" and model.variant == "i2v")
        ) and model.assets is None:
            raise ConfigRuleError(
                "model.assets",
                "Cache production for Flux/Wan requires model.assets",
                "Supply the family-specific model assets needed by the encoder.",
            )
        if model.family == ModelFamily.MINIMAX_H3 and model.minimax_h3 is None:
            raise ConfigRuleError(
                "model.minimax_h3",
                "MiniMaxH3 caching requires model.minimax_h3 encoder artifacts",
                "Supply the MiniMaxH3 encoder artifacts before caching.",
            )
        for name, data in self.dataset.groups.items():
            try:
                data.validate_for(model.family, model.variant, kind=data.kind)
                alignment = 32 if model.family in (ModelFamily.MINIMAX_H3, ModelFamily.QWEN_IMAGE_2_1) else 16
                if any(size < alignment or size % alignment for size in data.resolution):
                    raise ValueError(f"Cache resolution requires {alignment}-aligned dimensions")
                if data.wan is not None and model.family != ModelFamily.WAN:
                    raise ValueError("data.wan requires Wan")
                if data.minimax_h3 is not None and model.family != ModelFamily.MINIMAX_H3:
                    raise ValueError("data.minimax_h3 requires MiniMaxH3")
                if model.family in (ModelFamily.WAN, ModelFamily.MINIMAX_H3):
                    if data.kind not in ("image", "video"):
                        raise ValueError("Video model caches require image/video data")
                elif data.kind == "video":
                    raise ValueError("Image model caches do not support video data")
                if model.family == ModelFamily.WAN:
                    if data.video is not None and (
                        any((value - 1) % 4 for value in data.video.target_frames) or (data.video.max_frames - 1) % 4
                    ):
                        raise ValueError("Wan frame counts must follow 1+4k")
                    if data.wan is not None and data.wan.single_frame is not None:
                        if data.kind != "image" or model.variant != "i2v" or data.control_source is not None:
                            raise ValueError("Wan single_frame requires I2V images without control_source")
                if model.family == ModelFamily.MINIMAX_H3 and data.kind == "video":
                    video = data.video
                    if video is None or video.target_fps != 24 or video.fps_resampling != "timestamps":
                        raise ValueError("MiniMaxH3 video requires explicit 24Hz timestamps")
                    if any(count < 5 or (count - 5) % 17 for count in video.target_frames) or video.max_frames < 5:
                        raise ValueError("MiniMaxH3 frame counts must follow 17*n+5")
            except ValueError as error:
                raise ConfigRuleError(
                    f"dataset.groups.{name}",
                    str(error),
                    "Check this group against the selected model family and its resolution/frame requirements.",
                ) from None
        return self


def cache_document(document: dict[str, Any]) -> dict[str, Any]:
    """Select cache inputs from a shared run file without validating training sections.

    Known training sections are deliberately ignored except seed and REPA settings.
    Unknown sections still reach the strict cache schema and fail validation.
    """
    result = dict(document)
    train = result.pop("train", {})
    method = result.pop("method", {})
    for section in ("losses", "execution", "output", "preview"):
        result.pop(section, None)
    if isinstance(train, dict) and "seed" in train:
        result.setdefault("seed", train["seed"])
    if isinstance(method, dict) and "repa" in method:
        result.setdefault("repa", method["repa"])
    model = result.get("model")
    if isinstance(model, dict):
        model = dict(model)
        weights = model.pop("weights", None)
        if model.get("assets") is None and isinstance(weights, str):
            model["assets"] = str(Path(weights).parent)
        result["model"] = model
    dataset = result.get("dataset")
    if isinstance(dataset, dict) and isinstance(dataset.get("groups"), dict):
        unused = DatasetConfig.model_fields.keys() - CacheDatasetConfig.model_fields.keys()
        result["dataset"] = {
            **dataset,
            "groups": {
                name: {key: value for key, value in group.items() if key not in unused}
                if isinstance(group, dict)
                else group
                for name, group in dataset["groups"].items()
            },
        }
    return result
