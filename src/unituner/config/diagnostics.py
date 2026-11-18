"""Describe configuration failures without dumping user documents or string values."""

from collections.abc import Mapping
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any

from pydantic import BaseModel, JsonValue, ValidationError


class ConfigRuleError(ValueError):
    """Attach a field and an actionable remedy to a cross-field validation rule."""

    def __init__(self, field: str, message: str, hint: str) -> None:
        super().__init__(message)
        self.field = field
        self.hint = hint


@dataclass(frozen=True, slots=True)
class ConfigIssue:
    """Keep one diagnostic independently available to CLI and API consumers."""

    field: str
    code: str
    message: str
    source: str
    received: str
    expected: str = ""
    hint: str = ""
    description: str = ""

    def render(self) -> str:
        lines = [
            f"  - {self.field}: {self.message} [{self.code}]",
            f"    Source: {self.source}; received: {self.received}",
        ]
        for label, value in (("Expected", self.expected), ("Description", self.description), ("Fix", self.hint)):
            if value:
                lines.append(f"    {label}: {value}")
        return "\n".join(lines)


class ConfigError(ValueError):
    """Report concise configuration failures; structured issues support other frontends."""

    hide_traceback = True

    def __init__(self, message: str, *, issues: tuple[ConfigIssue, ...] = ()) -> None:
        super().__init__(message)
        self.issues = issues


def input_summary(value: object) -> str:
    """Expose scalar numbers, never arbitrary strings, paths or container contents."""
    if value is None or isinstance(value, (bool, int, float)):
        return repr(value)
    if isinstance(value, str):
        return f"string (length={len(value)}; contents hidden)"
    if isinstance(value, (list, tuple, dict)):
        return f"{type(value).__name__} ({len(value)} items)"
    return type(value).__name__


def _branches(node: dict[str, Any], definitions: dict[str, Any]):
    if "$ref" in node:
        yield from _branches(definitions[node["$ref"].rsplit("/", 1)[-1]], definitions)
    else:
        yield node
        for kind in ("anyOf", "oneOf", "allOf"):
            for branch in node.get(kind, ()):
                yield from _branches(branch, definitions)


def schema_nodes(schema: dict[str, Any], path: tuple[str | int, ...]) -> list[dict[str, Any]]:
    """Walk model properties, dynamic dataset groups and fixed/variable arrays."""
    definitions = schema.get("$defs", {})
    nodes = [schema]
    for part in path:
        children = []
        for node in nodes:
            for branch in _branches(node, definitions):
                if isinstance(part, int):
                    items = branch.get("prefixItems", ())
                    child = items[part] if part < len(items) else branch.get("items")
                else:
                    child = branch.get("properties", {}).get(part)
                    if child is None:
                        child = branch.get("additionalProperties")
                if isinstance(child, dict):
                    children.append(child)
        nodes = children
    return [branch for node in nodes for branch in _branches(node, definitions)]


def _configuration_path(schema: dict[str, Any], location: tuple[str | int, ...]) -> tuple[str | int, ...]:
    """Remove Pydantic union branch labels that are not user configuration keys."""
    path: tuple[str | int, ...] = ()
    for part in location:
        nodes = schema_nodes(schema, path)
        discriminator_tags = {tag for node in nodes for tag in node.get("discriminator", {}).get("mapping", {})}
        primitive_types = {node.get("type") for node in nodes if "type" in node}
        aliases = {"str": "string", "int": "integer", "float": "number", "bool": "boolean"}
        if part in discriminator_tags or (
            isinstance(part, str)
            and not schema_nodes(schema, (*path, part))
            and (
                aliases.get(part) in primitive_types
                or (part.startswith("literal[") and any("const" in node for node in nodes))
            )
        ):
            continue
        path += (part,)
    return path


def _expected(nodes: list[dict[str, Any]]) -> str:
    choices = []
    for node in nodes:
        parts = []
        if "enum" in node or "const" in node:
            parts.append("one of " + ", ".join(repr(v) for v in node.get("enum", [node.get("const")])))
        elif "type" in node:
            parts.append(node["type"])
        for key, symbol in (("exclusiveMinimum", ">"), ("minimum", ">="), ("exclusiveMaximum", "<"), ("maximum", "<=")):
            if key in node:
                parts.append(f"{symbol} {node[key]}")
        for key, label in (
            ("minItems", "minimum items"),
            ("maxItems", "maximum items"),
            ("minLength", "minimum length"),
            ("maxLength", "maximum length"),
            ("pattern", "pattern"),
        ):
            if key in node:
                parts.append(f"{label}: {node[key]}")
        description = "; ".join(parts)
        if description and description not in choices:
            choices.append(description)
    return " OR ".join(choices)


def override_source(path: tuple[str | int, ...], overrides: Mapping[str, JsonValue]) -> str:
    node: object = overrides
    for part in path:
        if isinstance(node, Mapping) and part in node:
            node = node[part]
        elif isinstance(node, list) and isinstance(part, int) and 0 <= part < len(node):
            node = node[part]
        else:
            return "TOML/default"
        if not isinstance(node, Mapping):
            return "CLI override"
    return "CLI override" if node else "TOML/default"


_MISSING = object()


def _lookup(values: object, path: tuple[str | int, ...]) -> object:
    for part in path:
        if isinstance(values, Mapping) and part in values:
            values = values[part]
        elif isinstance(values, list) and isinstance(part, int) and 0 <= part < len(values):
            values = values[part]
        else:
            return _MISSING
    return values


def _input_path(values: Mapping[str, JsonValue], path: tuple[str | int, ...]) -> tuple[str | int, ...]:
    if _lookup(values, path) is _MISSING and len(path) > 3 and path[:2] == ("dataset", "groups"):
        inherited = ("dataset", "defaults", *path[3:])
        if _lookup(values, inherited) is not _MISSING:
            return inherited
    return path


def validation_issues(
    error: ValidationError, schema: type[BaseModel], overrides: Mapping[str, JsonValue], values: Mapping[str, JsonValue]
) -> tuple[ConfigIssue, ...]:
    document = schema.model_json_schema()
    issues = []
    for item in error.errors(include_url=False):
        path = _configuration_path(document, tuple(item["loc"]))
        rule = item.get("ctx", {}).get("error")
        hint = ""
        if isinstance(rule, ConfigRuleError):
            path += tuple(rule.field.split(".")) if rule.field else ()
            hint = rule.hint
        nodes = schema_nodes(document, path)
        code, message = item["type"], item["msg"]
        if code == "extra_forbidden":
            parent = schema_nodes(document, path[:-1])
            names = {name for node in parent for name in node.get("properties", {})}
            matches = get_close_matches(str(path[-1]), sorted(names), n=1, cutoff=0.6)
            hint = f"Unknown parameter. Did you mean {matches[0]!r}?" if matches else "Remove this unknown parameter."
        elif code == "missing":
            hint = "Add this required field in TOML or pass --set PATH=VALUE using the field path above."
        elif code.endswith("_type"):
            hint = "Use the declared TOML/JSON type; quoted numbers and booleans are strings."
        elif code == "union_tag_invalid":
            message = "Unknown configuration variant"
            hint = "Choose a supported discriminator value for this model or method."
        elif code == "union_tag_not_found":
            hint = "Supply the model or method discriminator required by this configuration."
        elif not hint and code != "value_error":
            hint = "Set this parameter to a value satisfying the constraint above."
        actual_path = _input_path(values, path)
        value = _lookup(values, actual_path) if isinstance(rule, ConfigRuleError) else item.get("input")
        received = (
            "missing"
            if code == "missing"
            else "not supplied (default/inherited)"
            if value is _MISSING
            else input_summary(value)
        )
        source = override_source(actual_path, overrides)
        if actual_path != path:
            source += " (inherited from dataset.defaults)"
        issues.append(
            ConfigIssue(
                ".".join(map(str, path)) or "<configuration>",
                code,
                message,
                source,
                received,
                _expected(nodes)
                + (f"; supported variants: {item['ctx']['expected_tags']}" if code == "union_tag_invalid" else ""),
                hint,
                next((node["description"] for node in nodes if "description" in node), ""),
            )
        )
    return tuple(issues)


def configuration_error(source: object, issues: tuple[ConfigIssue, ...]) -> ConfigError:
    return ConfigError(
        f"{source}: {len(issues)} configuration error(s)\n" + "\n".join(issue.render() for issue in issues),
        issues=issues,
    )


def schema_field_help(schema: dict[str, Any], path: str) -> str:
    """Derive CLI parameter documentation from the same schema as diagnostics."""
    nodes = schema_nodes(schema, tuple(path.split(".")))
    description = next((node["description"] for node in nodes if "description" in node), "")
    expected = _expected(nodes)
    default = next((node["default"] for node in nodes if "default" in node), None)
    parts = [f"Set {path}"]
    if description:
        parts.append(description)
    if expected:
        parts.append(f"Expected: {expected}")
    if default is not None and isinstance(default, (int, float, bool)):
        parts.append(f"Default: {default}")
    return ". ".join(parts)
