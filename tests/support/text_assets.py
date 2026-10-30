"""Explicitly select tiny local tokenizer fixtures for offline model/application tests."""

import json
import tomllib
from collections.abc import Sequence
from pathlib import Path

from unituner.cli.main import main
from unituner.config.text import DEFAULT_TEXT_CONFIG, TextEncodingConfig, TokenizerSourceConfig


def local_text_config(
    directory: Path,
    *,
    base: TextEncodingConfig = DEFAULT_TEXT_CONFIG,
    clip: Path | None = None,
) -> TextEncodingConfig:
    """Keep numerical options while replacing production Hub defaults with fixture assets."""
    updates = {"tokenizer": TokenizerSourceConfig(source=str(directory), offline=True)}
    if clip is not None:
        updates["clip_tokenizer"] = TokenizerSourceConfig(source=str(clip), offline=True)
    return base.model_copy(update=updates)


def main_with_local_tokenizers(argv: Sequence[str]) -> int:
    """Run the public CLI with explicit local fixture sources, without changing production defaults."""
    if not argv or argv[0] not in ("cache", "train", "generate", "generate-video", "restore"):
        return main(argv)
    model = {}
    for index, arg in enumerate(argv[1:], 1):
        path = Path(arg) if str(arg).endswith(".toml") else None
        if path is not None and path.is_file():
            with path.open("rb") as stream:
                model = tomllib.load(stream).get("model", {})
            break
        if arg == "--model":
            model = json.loads(argv[index + 1])
            break
    text = model.get("text", {})
    source = model.get("text_encoder")
    family = model.get("family")
    if family == "minimax-h3":
        source = model.get("minimax_h3", {}).get("text_processor")
    if not source:
        return main(argv)
    extra = []
    if Path(source).is_dir() and not text.get("tokenizer", {}).get("source"):
        extra.extend(
            ["--set", "model.text.tokenizer.source=" + json.dumps(source), "--set", "model.text.tokenizer.offline=true"]
        )
    if family == "flux" and not (text.get("clip_tokenizer") or {}).get("source"):
        root = Path(model.get("assets") or Path(model.get("weights") or ".").parent)
        extra.extend(
            [
                "--set",
                "model.text.clip_tokenizer.source=" + json.dumps(str(root / "text_encoder")),
                "--set",
                "model.text.clip_tokenizer.offline=true",
            ]
        )
    if family == "wan" and not text.get("tokenizer", {}).get("source"):
        root = Path(model.get("assets") or Path(model.get("weights") or ".").parent)
        extra.extend(
            [
                "--set",
                "model.text.tokenizer.source=" + json.dumps(str(root / "tokenizer")),
                "--set",
                "model.text.tokenizer.offline=true",
            ]
        )
    return main([*argv, *extra])
