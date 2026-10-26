"""Argument validation against a small, explicit subset of JSON Schema.

Decision: support only the keywords the tool definitions use, and reject any
other keyword when a tool is registered. A typo such as ``"requird"`` in a tool
schema then fails at startup instead of silently validating nothing.

Validation is strict (no coercion of ``"10"`` to ``10``, unknown properties
rejected): the model gets a precise error and corrects itself next turn, which
is cheaper than debugging a tool that ran with arguments it misread.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass

from agent.types import JSONObject, JSONValue

_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "enum",
        "items",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "default",
        "description",
    }
)
_TYPES = frozenset({"object", "array", "string", "integer", "number", "boolean", "null"})


class SchemaError(ValueError):
    """A tool schema itself is invalid (a programming error, not model error)."""


def check_schema(schema: JSONValue, where: str = "$") -> None:
    """Raise SchemaError if ``schema`` uses anything outside the supported subset."""
    if not isinstance(schema, dict):
        raise SchemaError(f"{where}: schema must be an object")
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise SchemaError(f"{where}: unsupported keywords {sorted(unknown)}")
    declared = schema.get("type")
    if not isinstance(declared, str) or declared not in _TYPES:
        raise SchemaError(f"{where}: 'type' must be one of {sorted(_TYPES)}")
    props = schema.get("properties", {})
    if not isinstance(props, dict):
        raise SchemaError(f"{where}: 'properties' must be an object")
    for name, sub in props.items():
        check_schema(sub, f"{where}.{name}")
    required = schema.get("required", [])
    if not isinstance(required, list) or not all(isinstance(r, str) and r in props for r in required):
        raise SchemaError(f"{where}: 'required' must list declared properties")
    if "items" in schema:
        check_schema(schema["items"], f"{where}[]")
    if "default" in schema and validate(schema, schema["default"]):
        raise SchemaError(f"{where}: default does not satisfy its own schema")


def _type_of(value: JSONValue) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    return "object"


def _show(value: JSONValue) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= 60 else text[:57] + "..."


def _label(path: str) -> str:
    return path or "arguments"


def validate(schema: JSONObject, value: JSONValue, path: str = "") -> list[str]:
    """Return human-readable errors; an empty list means ``value`` is valid."""
    expected = schema["type"]
    actual = _type_of(value)
    type_ok = actual == expected or (expected == "number" and actual == "integer")
    if not type_ok:
        return [f"{_label(path)}: expected {expected}, got {actual} {_show(value)}"]

    errors: list[str] = []
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        allowed = ", ".join(_show(e) for e in enum)
        errors.append(f"{_label(path)}: must be one of {allowed}; got {_show(value)}")

    if isinstance(value, str):
        min_len, max_len = schema.get("minLength"), schema.get("maxLength")
        if isinstance(min_len, int) and len(value) < min_len:
            errors.append(f"{_label(path)}: must be at least {min_len} characters")
        if isinstance(max_len, int) and len(value) > max_len:
            errors.append(f"{_label(path)}: must be at most {max_len} characters")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        low, high = schema.get("minimum"), schema.get("maximum")
        if isinstance(low, (int, float)) and value < low:
            errors.append(f"{_label(path)}: must be >= {low}; got {value}")
        if isinstance(high, (int, float)) and value > high:
            errors.append(f"{_label(path)}: must be <= {high}; got {value}")

    if isinstance(value, list):
        min_items, max_items = schema.get("minItems"), schema.get("maxItems")
        if isinstance(min_items, int) and len(value) < min_items:
            errors.append(f"{_label(path)}: must contain at least {min_items} item(s)")
        if isinstance(max_items, int) and len(value) > max_items:
            errors.append(f"{_label(path)}: must contain at most {max_items} item(s)")
        items = schema.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(validate(items, item, f"{path}[{index}]"))

    if isinstance(value, dict):
        props = schema.get("properties", {})
        assert isinstance(props, dict)
        required = schema.get("required", [])
        assert isinstance(required, list)
        for name in required:
            assert isinstance(name, str)
            if name not in value:
                errors.append(f"{_join(path, name)}: missing required property")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in props:
                    allowed = ", ".join(sorted(props)) or "none"
                    errors.append(f"{_join(path, name)}: unexpected property (allowed: {allowed})")
        for name, sub in props.items():
            if name in value and isinstance(sub, dict):
                errors.extend(validate(sub, value[name], _join(path, name)))
    return errors


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


def with_defaults(schema: JSONObject, value: JSONValue) -> JSONValue:
    """Return a copy of ``value`` with schema defaults filled in for absent properties."""
    if isinstance(value, dict) and schema.get("type") == "object":
        props = schema.get("properties", {})
        assert isinstance(props, dict)
        filled: JSONObject = {}
        for name, item in value.items():
            sub = props.get(name)
            filled[name] = with_defaults(sub, item) if isinstance(sub, dict) else copy.deepcopy(item)
        for name, sub in props.items():
            if name not in filled and isinstance(sub, dict) and "default" in sub:
                filled[name] = copy.deepcopy(sub["default"])
        return filled
    if isinstance(value, list):
        items = schema.get("items")
        if isinstance(items, dict):
            return [with_defaults(items, item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True, slots=True)
class ToolArgs:
    """Validated arguments with typed accessors.

    Accessors raise TypeError only if called with a key/type the schema does
    not guarantee, which is a bug in the tool, never in the model's output.
    """

    values: Mapping[str, JSONValue]

    def _get(self, key: str) -> JSONValue:
        if key not in self.values:
            raise TypeError(f"argument {key!r} absent; give it a default in the schema")
        return self.values[key]

    def has(self, key: str) -> bool:
        return key in self.values

    def text(self, key: str) -> str:
        value = self._get(key)
        if not isinstance(value, str):
            raise TypeError(f"argument {key!r} is not a string")
        return value

    def integer(self, key: str) -> int:
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"argument {key!r} is not an integer")
        return value

    def number(self, key: str) -> float:
        value = self._get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"argument {key!r} is not a number")
        return float(value)

    def boolean(self, key: str) -> bool:
        value = self._get(key)
        if not isinstance(value, bool):
            raise TypeError(f"argument {key!r} is not a boolean")
        return value

    def optional_integer(self, key: str) -> int | None:
        return self.integer(key) if key in self.values else None

    def text_list(self, key: str) -> list[str]:
        value = self._get(key)
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise TypeError(f"argument {key!r} is not a list of strings")
        return [v for v in value if isinstance(v, str)]

    def object_list(self, key: str) -> list[ToolArgs]:
        value = self._get(key)
        if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
            raise TypeError(f"argument {key!r} is not a list of objects")
        return [ToolArgs(v) for v in value if isinstance(v, dict)]


def bind(schema: JSONObject, raw: JSONObject) -> ToolArgs | list[str]:
    """Apply defaults, validate, and wrap. Returns errors instead of raising."""
    filled = with_defaults(schema, raw)
    errors = validate(schema, filled)
    if errors:
        return errors
    assert isinstance(filled, dict)
    return ToolArgs(filled)
