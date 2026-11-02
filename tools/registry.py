"""Explicit registry, JSON argument validation, and authorization helpers."""
from __future__ import annotations
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from core.agent import ToolCall, ToolSpec

ToolHandler = Callable[[Mapping[str, object]], str]


@dataclass(frozen=True)
class _Entry:
    spec: ToolSpec
    handler: ToolHandler


class ToolRegistry:
    """Only explicitly registered handlers can be dispatched."""
    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("spec must be a ToolSpec")
        if not isinstance(spec.name, str) or not spec.name or spec.name != spec.name.strip():
            raise ValueError("tool name must be non-empty with no surrounding whitespace")
        if not isinstance(spec.description, str) or not isinstance(spec.parameters_schema, Mapping):
            raise ValueError("tool description and schema must be valid")
        if not isinstance(spec.requires_confirmation, bool) or not callable(handler):
            raise ValueError("confirmation flag or handler is invalid")
        if spec.name in self._entries:
            raise ValueError(f"tool is already registered: {spec.name}")
        safe = ToolSpec(spec.name, spec.description, deepcopy(dict(spec.parameters_schema)),
                        spec.requires_confirmation)
        self._entries[spec.name] = _Entry(safe, handler)

    def lookup(self, name: str) -> ToolSpec | None:
        entry = self._entries.get(name)
        if entry is None:
            return None
        spec = entry.spec
        return ToolSpec(spec.name, spec.description, deepcopy(dict(spec.parameters_schema)),
                        spec.requires_confirmation)

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self.lookup(name) for name in self._entries)

    def execute(self, call: ToolCall) -> str:
        entry = self._entries.get(call.name)
        if entry is None:
            raise KeyError(f"unknown tool: {call.name}")
        result = entry.handler(deepcopy(dict(call.arguments)))
        if not isinstance(result, str):
            raise TypeError("tool handlers must return a string")
        return result


class JsonSchemaArgumentValidator:
    """Validate object arguments and JSON Schema types used by the local tools."""
    def validate(self, schema: Mapping[str, object],
                 arguments: Mapping[str, object]) -> None:
        if not isinstance(schema, Mapping):
            raise ValueError("schema must be a mapping")
        self._validate_value(schema, arguments, "$")

    def _validate_value(self, schema: Mapping[str, object], value: object, path: str) -> None:
        expected = schema.get("type")
        if expected is not None and not self._matches(expected, value):
            raise ValueError(f"{path} has the wrong JSON type")
        if expected == "object" or isinstance(value, Mapping):
            if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
                raise ValueError(f"{path} must be an object")
            props, required = schema.get("properties", {}), schema.get("required", ())
            if not isinstance(props, Mapping) or not isinstance(required, Sequence) or isinstance(required, (str, bytes)):
                raise ValueError(f"{path} has an invalid object schema")
            if any(not isinstance(k, str) for k in required) or any(k not in value for k in required):
                raise ValueError(f"{path} is missing or has invalid required properties")
            if schema.get("additionalProperties", True) is False and any(k not in props for k in value):
                raise ValueError(f"{path} contains an unexpected property")
            for key, item in value.items():
                child = props.get(key)
                if child is not None:
                    if not isinstance(child, Mapping):
                        raise ValueError(f"{path}.{key} has an invalid schema")
                    self._validate_value(child, item, f"{path}.{key}")

    @staticmethod
    def _matches(expected: object, value: object) -> bool:
        if isinstance(expected, list):
            return any(JsonSchemaArgumentValidator._matches(item, value) for item in expected)
        if expected == "object":
            return isinstance(value, Mapping)
        if expected == "array":
            return isinstance(value, list)
        if expected == "string":
            return isinstance(value, str)
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if expected == "null":
            return value is None
        raise ValueError(f"unsupported JSON Schema type: {expected!r}")


class ToolAllowListAuthorizer:
    """Allows only calls whose tool name is explicitly configured."""
    def __init__(self, allowed_tool_names: Sequence[str]) -> None:
        names = tuple(allowed_tool_names)
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("allowed tool names must be non-empty strings")
        self._allowed = frozenset(names)

    def is_allowed(self, tool: ToolSpec, arguments: Mapping[str, object]) -> bool:
        del arguments
        return tool.name in self._allowed
