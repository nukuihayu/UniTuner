"""Validate optional TOML and explicit overrides at one configuration boundary."""

import json
import math
import tomllib
from collections.abc import Callable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue, RootModel, ValidationError

from unituner.config.cache import CacheConfig, cache_document
from unituner.config.calibration import CalibrationConfig, calibration_document
from unituner.config.diagnostics import ConfigError as ConfigError
from unituner.config.diagnostics import (
    ConfigIssue,
    configuration_error,
    input_summary,
    override_source,
    validation_issues,
)
from unituner.config.generation import GenerationConfig
from unituner.config.restoration_inference import RestorationInferenceConfig
from unituner.config.training import TrainingConfig
from unituner.models.minimax_h3.config.generation import MiniMaxH3GenerationConfig
from unituner.models.wan.config.generation import WanGenerationConfig


class VideoGenerationConfig(RootModel[WanGenerationConfig | MiniMaxH3GenerationConfig]):
    """Validate a Wan or MiniMaxH3 generation document."""

    model_config = ConfigDict(frozen=True, strict=True)


def merge_config_overrides(document: dict[str, JsonValue], overrides: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Merge grouped fields recursively while replacing scalar and array values."""
    merged = dict(document)
    for key, value in overrides.items():
        previous = merged.get(key)
        merged[key] = (
            merge_config_overrides(previous, value) if isinstance(previous, dict) and isinstance(value, dict) else value
        )
    return merged


def _document_issues(
    value: object, overrides: Mapping[str, JsonValue], path: tuple[str | int, ...] = ()
) -> tuple[ConfigIssue, ...]:
    if isinstance(value, dict):
        return tuple(
            issue for key, child in value.items() for issue in _document_issues(child, overrides, (*path, key))
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            issue for index, child in enumerate(value) for issue in _document_issues(child, overrides, (*path, index))
        )
    if isinstance(value, float) and not math.isfinite(value):
        message, hint = "Nonfinite numbers are not supported", "Replace nan/inf with a finite number."
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        message, hint = (
            "Unsupported configuration value type",
            "Use TOML strings, numbers, booleans, arrays or tables; quote dates.",
        )
    else:
        return ()
    return (
        ConfigIssue(
            ".".join(map(str, path)),
            "invalid_value",
            message,
            override_source(path, overrides),
            input_summary(value),
            hint=hint,
        ),
    )


def _load_config[C: BaseModel](
    path: Path | None,
    schema: type[C],
    overrides: Mapping[str, JsonValue] | None,
    *,
    project: Callable[[dict[str, JsonValue]], dict[str, JsonValue]] | None = None,
) -> C:
    """Apply explicit overrides before one strict grouped schema validation.

    JSON-mode validation accepts TOML arrays as fixed tuples while retaining
    strict scalar checks. Unknown keys, nonfinite numbers and datetime objects
    fail. Paths remain relative to the caller's cwd. An omitted path validates overrides alone without reading a file.
    Diagnostics expose numeric inputs and types, while string contents stay hidden.
    """
    try:
        document = {}
        if path is not None:
            with path.open("rb") as stream:
                document = tomllib.load(stream)
        document = merge_config_overrides(document, overrides or {})
        document = document if project is None else project(document)
        issues = _document_issues(document, overrides or {})
        if issues:
            raise configuration_error(path or "CLI", issues)
        encoded = json.dumps(document, allow_nan=False)
        return schema.model_validate_json(encoded, strict=True)
    except ValidationError as error:
        raise configuration_error(path or "CLI", validation_issues(error, schema, overrides or {}, document)) from None
    except tomllib.TOMLDecodeError as error:
        detail = str(error)
        raise ConfigError(
            f"{path}: invalid TOML: {detail}. Check table names, quoting and duplicate assignments."
        ) from None
    except ConfigError:
        raise
    except UnicodeError:
        raise ConfigError(f"{path}: configuration must be valid UTF-8 text; save the TOML file as UTF-8.") from None
    except OSError as error:
        raise ConfigError(
            f"{path}: cannot read configuration ({error.strerror}); check the path and read permissions."
        ) from None
    except (TypeError, ValueError) as error:
        raise ConfigError(
            f"{path or 'CLI'}: invalid configuration ({type(error).__name__}); check value types and command-specific fields."
        ) from None


def load_training_config(
    path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None
) -> TrainingConfig:
    """Load strict grouped training configuration with explicit overrides."""
    return _load_config(path, TrainingConfig, overrides)


def load_generation_config(
    path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None
) -> GenerationConfig:
    """Load strict generation configuration with explicit overrides."""
    return _load_config(path, GenerationConfig, overrides)


def load_restoration_inference_config(
    path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None
) -> RestorationInferenceConfig:
    """Load strict one-step image restoration configuration with explicit overrides."""
    return _load_config(path, RestorationInferenceConfig, overrides)


def load_wan_generation_config(
    path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None
) -> WanGenerationConfig:
    """Load strict native Wan video-generation configuration with explicit overrides."""
    return _load_config(path, WanGenerationConfig, overrides)


def load_video_generation_config(
    path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None
) -> WanGenerationConfig | MiniMaxH3GenerationConfig:
    """Load a strict family-owned video generation configuration without compute imports."""
    return _load_config(path, VideoGenerationConfig, overrides).root


def load_cache_config(path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None) -> CacheConfig:
    """Read standalone or shared cache inputs without training validation."""
    return _load_config(path, CacheConfig, overrides, project=cache_document)


def load_calibration_config(
    path: Path | None = None, *, overrides: Mapping[str, JsonValue] | None = None
) -> CalibrationConfig:
    """Validate VAE calibration independently of training and cache production."""
    return _load_config(path, CalibrationConfig, overrides, project=calibration_document)
