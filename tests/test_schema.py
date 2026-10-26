from __future__ import annotations

import pytest

from agent.schema import SchemaError, ToolArgs, bind, check_schema, validate
from agent.tools import build_registry
from agent.types import JSONObject, Limits

SCHEMA: JSONObject = {
    "type": "object",
    "properties": {
        "path": {"type": "string", "minLength": 1},
        "count": {"type": "integer", "minimum": 1, "default": 5},
        "mode": {"type": "string", "enum": ["a", "b"], "default": "a"},
        "items": {
            "type": "array",
            "items": {"type": "object", "properties": {"x": {"type": "number"}}, "required": ["x"],
                      "additionalProperties": False},
        },
    },
    "required": ["path"],
    "additionalProperties": False,
}


def test_defaults_are_applied_and_values_typed() -> None:
    bound = bind(SCHEMA, {"path": "a.py"})
    assert isinstance(bound, ToolArgs)
    assert bound.text("path") == "a.py"
    assert bound.integer("count") == 5
    assert bound.text("mode") == "a"


def test_every_problem_is_reported_with_its_location() -> None:
    errors = validate(SCHEMA, {"count": "10", "mode": "z", "extra": 1, "items": [{"x": 1}, {"y": 2}]})
    joined = "\n".join(errors)
    assert "path: missing required property" in joined
    assert 'count: expected integer, got string "10"' in joined
    assert "mode: must be one of" in joined
    assert "extra: unexpected property" in joined
    assert "items[1].x: missing required property" in joined
    assert "items[1].y: unexpected property" in joined


def test_booleans_are_not_integers_and_bounds_hold() -> None:
    assert validate(SCHEMA, {"path": "p", "count": True}) == ["count: expected integer, got boolean true"]
    assert validate(SCHEMA, {"path": "p", "count": 0}) == ["count: must be >= 1; got 0"]
    assert validate(SCHEMA, {"path": ""}) == ["path: must be at least 1 characters"]


def test_schema_typos_fail_at_registration() -> None:
    with pytest.raises(SchemaError, match="unsupported keywords"):
        check_schema({"type": "object", "requird": ["x"]})
    with pytest.raises(SchemaError, match="default does not satisfy"):
        check_schema({"type": "integer", "default": "five"})


def test_all_tool_schemas_are_valid() -> None:
    registry = build_registry(Limits(), allow_delegate=True)
    assert set(registry.names()) == {
        "read_file", "write_file", "edit_file", "list_files", "search", "run_command",
        "update_plan", "finish", "delegate",
    }
    assert "delegate" not in build_registry(Limits(), allow_delegate=False).names()
