"""Resolve tokenizer-only Hub snapshots and local assets without loading model weights."""

from pathlib import Path
from typing import Literal

from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

from unituner.artifacts.hashing import canonical_sha256, file_sha256
from unituner.config.text import TextEncodingConfig, TokenizerSourceConfig

# Exact filenames intentionally exclude encoder weights and Python code. config.json supplies processor type metadata.
TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "spiece.model",
    "tokenizer.model",
    "tekken.json",
    "chat_template.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "image_processor_config.json",
)


def default_tokenizer(
    family: str, version: str | None = None, *, role: str = "tokenizer"
) -> tuple[str, str | None] | None:
    """Match Musubi's released repository and subfolder selections."""
    if role == "clip_tokenizer":
        return "openai/clip-vit-large-patch14", None
    if role == "processor":
        return "Qwen/Qwen-Image-Edit", "processor"
    defaults = {
        "z-image": ("Tongyi-MAI/Z-Image", "tokenizer"),
        "flux": ("google/t5-v1_1-xxl", None),
        "qwen-image": ("Qwen/Qwen-Image", "tokenizer"),
        "krea2": ("Qwen/Qwen3-VL-4B-Instruct", None),
        "wan": ("Wan-AI/Wan2.1-T2V-14B", "google/umt5-xxl"),
        "minimax-h3": ("MiniMaxAI/MiniMax-H3", "processor"),
    }
    if family == "flux2":
        return {
            "dev": ("mistralai/Mistral-Small-3.1-24B-Instruct-2503", None),
            "klein-4b": ("Qwen/Qwen3-4B", None),
            "klein-9b": ("Qwen/Qwen3-8B", None),
        }.get(version or "")
    return defaults.get(family)


def tokenizer_directory(
    config: TextEncodingConfig,
    family: str,
    local_directory: Path | None = None,
    *,
    version: str | None = None,
    role: Literal["tokenizer", "clip_tokenizer", "processor"] = "tokenizer",
) -> Path:
    """Download only tokenizer assets, then let Transformers read the resulting local path."""
    selected = getattr(config, role)
    if selected is None:
        selected = TokenizerSourceConfig(offline=config.tokenizer.offline)
        # An explicit complete local processor directory can serve both Qwen inputs.
        if role == "processor" and config.tokenizer.source is not None:
            selected = config.tokenizer
    default = default_tokenizer(family, version, role=role)
    if local_directory is not None and local_directory.is_file():
        local_directory = local_directory.parent
    source = selected.source or (default[0] if default is not None else str(local_directory or ""))
    subfolder = selected.subfolder
    if selected.source is None and subfolder is None and default is not None:
        subfolder = default[1]
    if not source:
        raise ValueError(f"{family} requires an explicit tokenizer source")
    path = Path(source).expanduser()
    if path.is_dir():
        directory = path if subfolder is None else path / subfolder
        if not directory.is_dir():
            raise ValueError("Tokenizer subfolder does not exist in the selected local directory")
        return directory
    if path.is_absolute() or source.startswith((".", "~")) or "\\" in source:
        raise ValueError("Tokenizer local directory does not exist")
    prefix = "" if subfolder is None else subfolder + "/"
    try:
        snapshot = Path(
            snapshot_download(
                repo_id=source,
                revision=selected.revision,
                local_files_only=selected.offline,
                allow_patterns=[prefix + name for name in TOKENIZER_FILES] + [prefix + "chat_templates/*.jinja"],
            )
        )
    except (HfHubHTTPError, LocalEntryNotFoundError) as error:
        mode = "offline cache" if selected.offline else "Hub"
        raise ValueError(f"Cannot resolve tokenizer {source!r} from {mode}: {error}") from error
    return snapshot if subfolder is None else snapshot / subfolder


def tokenizer_sha256(directory: Path) -> str:
    """Hash tokenizer inputs only, including templates but excluding any colocated weights."""
    files = [directory / name for name in TOKENIZER_FILES if (directory / name).is_file()]
    files.extend(sorted((directory / "chat_templates").glob("*.jinja")))
    if not files:
        raise ValueError("Tokenizer source contains no tokenizer or processor assets")
    return canonical_sha256({path.relative_to(directory).as_posix(): file_sha256(path) for path in sorted(files)})


def pin_tokenizer_sources(
    config: TextEncodingConfig,
    family: str,
    local_directory: Path | None = None,
    *,
    version: str | None = None,
) -> tuple[TextEncodingConfig, dict[str, str]]:
    """Resolve and pin concrete snapshot paths once before cache hashing and encoder loading."""
    roles: tuple[Literal["tokenizer", "clip_tokenizer", "processor"], ...] = (
        ("tokenizer", "clip_tokenizer")
        if family == "flux"
        else ("tokenizer", "processor")
        if family == "qwen-image"
        else ("tokenizer",)
    )
    updates: dict[str, TokenizerSourceConfig] = {}
    digests = {}
    for role in roles:
        directory = tokenizer_directory(config, family, local_directory, version=version, role=role).resolve()
        digests[role] = tokenizer_sha256(directory)
        updates[role] = TokenizerSourceConfig(source=str(directory), offline=True)
    return config.model_copy(update=updates), digests
