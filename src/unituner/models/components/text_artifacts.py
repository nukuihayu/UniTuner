"""Resolve local text weights and topology independently of tokenizer assets."""

from pathlib import Path
from typing import Any

from unituner.artifacts.hashing import canonical_sha256, directory_sha256, file_sha256
from unituner.artifacts.tensors import weight_artifact_sha256
from unituner.config.text import TextEncodingConfig


def text_encoding_identity(config: TextEncodingConfig) -> dict[str, Any]:
    """Keep cache text policy separate from independently hashed tokenizer assets.

    Omit the section for the default configuration to preserve existing cache
    identities. An explicit tokenizer selection still records an empty policy
    section, matching the original producer contract.
    """
    if config == type(config)():
        return {}
    return {
        "text_encoding": config.model_dump(
            mode="json", exclude_defaults=True, exclude={"tokenizer", "clip_tokenizer", "processor"}
        )
    }


def text_config_path(path: Path, family: str, *, version: str | None = None) -> Path:
    """Prefer explicit local topology; use bundled released defaults for bare weights.

    A filename-specific JSON sidecar permits multiple encoders in one directory.
    Never infer architecture from the filename or download model configuration.
    Families with variable topology require a local sidecar.
    """
    if path.is_dir():
        config = path / "config.json"
    elif path.is_file():
        config = path.with_suffix(".json")
        if not config.is_file():
            config = path.parent / "config.json"
    else:
        raise FileNotFoundError(f"Text encoder does not exist: {path}")
    if config.is_file():
        return config
    if path.is_dir():
        raise ValueError(f"Text encoder directory requires config.json: {path}")
    defaults = {
        "z-image": "qwen3-4b",
        "flux": "t5-xxl",
        "flux-clip": "clip-l",
        "krea2": "qwen3-vl-4b",
        "flux3": "qwen3-vl-4b",
        "qwen-image": "qwen2-5-vl",
        "qwen-image-2-1": "qwen-image-2-1",
    }
    name = defaults.get(family)
    if family == "flux2":
        name = {"dev": "mistral3", "klein-4b": "qwen3-4b", "klein-9b": "qwen3-8b"}.get(version or "")
    if name is None:
        raise ValueError(
            f"{family} single-file text encoder requires {path.with_suffix('.json')} "
            f"or {path.parent / 'config.json'} describing its topology"
        )
    return Path(__file__).with_name("text_configs") / f"{name}.json"


def text_artifact_sha256(path: Path, family: str, *, version: str | None = None) -> str:
    """Include every weight shard and the effective topology in cache identity."""
    if path.is_dir():
        return directory_sha256(path)
    return canonical_sha256(
        {
            "weights": weight_artifact_sha256(path),
            "config": file_sha256(text_config_path(path, family, version=version)),
        }
    )
