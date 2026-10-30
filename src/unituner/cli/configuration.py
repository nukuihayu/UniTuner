"""Expose grouped configuration fields as optional command-line overrides."""

import argparse
import json
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from typing import Any

from pydantic import BaseModel, JsonValue

from unituner.config.diagnostics import schema_field_help
from unituner.config.loading import ConfigError, merge_config_overrides


def _fields(
    node: dict[str, Any], definitions: dict[str, Any], prefix: str = "", refs: frozenset[str] = frozenset()
) -> Iterator[str]:
    reference = node.get("$ref")
    if reference is not None and reference not in refs:
        yield from _fields(definitions[reference.rsplit("/", 1)[-1]], definitions, prefix, refs | {reference})
    for kind in ("anyOf", "oneOf", "allOf"):
        for branch in node.get(kind, []):
            yield from _fields(branch, definitions, prefix, refs)
    for name, field in node.get("properties", {}).items():
        path = f"{prefix}.{name}" if prefix else name
        yield path
        yield from _fields(field, definitions, path, refs)


def _value(path: str, value: str) -> tuple[str, JsonValue]:
    try:
        parsed = json.loads(value)
    except ValueError:
        if value.startswith(("[", "{", '"')):
            raise argparse.ArgumentTypeError(
                f"{path}: invalid JSON value; use double-quoted strings, comma-separated arrays "
                "such as [512,512], or objects with double-quoted keys"
            ) from None
        parsed = value
    return path, parsed


def _assignment(value: str) -> tuple[str, JsonValue]:
    path, separator, raw = value.partition("=")
    if not separator or any(not part or not part.isidentifier() for part in path.replace("-", "_").split(".")):
        raise argparse.ArgumentTypeError("Expected a dotted configuration path followed by =VALUE")
    return _value(path.replace("-", "_"), raw)


def add_config_arguments(
    parser: argparse.ArgumentParser, schema: type[BaseModel] | tuple[type[BaseModel], ...]
) -> None:
    """Offer optional TOML and schema-derived dotted flags without loading model runtimes."""
    parser.add_argument("config", type=Path, nargs="?", help="Optional base TOML configuration")
    parser.add_argument("--config", dest="config_file", type=Path, help="Optional base TOML configuration")
    group = parser.add_argument_group(
        "configuration overrides", "Values use JSON or unquoted strings; CLI overrides TOML"
    )
    group.add_argument("--set", dest="config_values", action="append", type=_assignment, metavar="PATH=VALUE")
    paths = set()
    help_text = {}
    for choice in schema if isinstance(schema, tuple) else (schema,):
        document = choice.model_json_schema()
        fields = tuple(_fields(document, document.get("$defs", {})))
        paths.update(fields)
        for path in fields:
            help_text.setdefault(path, schema_field_help(document, path))
    for path in sorted(paths):
        option = "--" + path.replace("_", "-")
        aliases = [option] if "_" not in path else [option, "--" + path]
        if path == "execution.precision":
            aliases.append("--precision")
        group.add_argument(
            *aliases,
            dest="config_values",
            action="append",
            type=partial(_value, path),
            metavar="VALUE",
            help=help_text[path].replace("%", "%%"),
        )


def set_override(overrides: dict[str, JsonValue], path: str, value: JsonValue) -> None:
    """Apply one ordered assignment, rejecting conflicting scalar parent paths."""
    parts = path.split(".")
    target = overrides
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        child = target[part]
        if not isinstance(child, dict):
            raise ConfigError(
                f"CLI: {path}: cannot assign a field below a scalar value at {part!r}; "
                "remove the parent scalar assignment or assign a JSON object before setting child fields"
            )
        target = child
    previous = target.get(parts[-1])
    if isinstance(previous, dict) and isinstance(value, dict):
        target[parts[-1]] = merge_config_overrides(previous, value)
    else:
        target[parts[-1]] = value


def config_input(args: argparse.Namespace) -> tuple[Path | None, dict[str, JsonValue]]:
    """Resolve the optional file and ordered values shared by every configured command."""
    if args.config is not None and args.config_file is not None:
        raise ConfigError("Specify the TOML path either positionally or with --config")
    overrides: dict[str, JsonValue] = {}
    for path, value in args.config_values or ():
        set_override(overrides, path, value)
    return args.config_file or args.config, overrides
