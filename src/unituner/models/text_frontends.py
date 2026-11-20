"""Load only tokenizer/processor assets and retain them for a single preview-enabled run."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Literal

from unituner.config.model import ModelConfig
from unituner.config.text import TextEncodingConfig, TokenizerSourceConfig
from unituner.models.text_assets import tokenizer_directory

type Role = Literal["tokenizer", "clip_tokenizer", "processor"]
type Key = tuple[TextEncodingConfig, str, Path | None, str | None, Role]
_active = ContextVar[dict[Key, Any] | None]("preview_text_assets", default=None)


def load_frontend(
    config: TextEncodingConfig,
    family: str,
    directory: Path | None = None,
    *,
    version: str | None = None,
    role: Role = "tokenizer",
) -> Any:
    """Use each family's native parser, including independently configured secondary assets."""
    key = (config, family, directory, version, role)
    cache = _active.get()
    if cache is not None and key in cache:
        return cache[key]
    path = tokenizer_directory(config, family, directory, version=version, role=role)
    try:
        asset = _read_frontend(path, family, version, role)
    except (OSError, ValueError, ImportError) as error:
        raise ValueError(f"Cannot load {family} {role} from {path}: {error}") from error
    if cache is not None:
        cache[key] = asset
    return asset


def _read_frontend(path: Path, family: str, version: str | None, role: Role) -> Any:
    import transformers

    if family in ("minimax-h3", "minimax-h3-image"):
        from unituner.models.minimax_h3.text.processor import load_text_processor

        return load_text_processor(path)
    if family == "flux":
        loader = transformers.CLIPTokenizer if role == "clip_tokenizer" else transformers.T5Tokenizer
    elif family in ("z-image", "qwen-image") and role == "tokenizer":
        loader = transformers.Qwen2Tokenizer
    elif family in ("flux3", "qwen-image-2-1", "qwen-image") or (family == "flux2" and version == "dev"):
        loader = transformers.AutoProcessor
    else:
        loader = transformers.AutoTokenizer
    return loader.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)


@contextmanager
def preview_text_assets(model: ModelConfig) -> Iterator[None]:
    """Open native text frontends before training, releasing references even on failure."""
    family = model.family.value
    config = model.text
    directory = Path(model.text_encoder) if model.text_encoder else None
    version = model.version if family == "flux2" else None
    if family in ("wan", "minimax-h3", "minimax-h3-image"):
        directory = None
    if family == "minimax-h3" and model.minimax_h3 and model.minimax_h3.text_processor and not config.tokenizer.source:
        config = config.model_copy(
            update={"tokenizer": TokenizerSourceConfig(source=model.minimax_h3.text_processor, offline=True)}
        )
    token = _active.set({})
    try:
        load_frontend(config, family, directory, version=version)
        if family == "flux":
            load_frontend(config, family, model.asset_path("clip_text"), role="clip_tokenizer")
        elif family == "qwen-image":
            load_frontend(config, family, directory, role="processor")
        yield
    finally:
        _active.reset(token)
