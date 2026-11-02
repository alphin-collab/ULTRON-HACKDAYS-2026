from __future__ import annotations
import json
import unittest
from collections.abc import Sequence
from core.agent import AgentCore, AgentStatus, HostApproval, ModelFinishReason, ModelResponse, ToolCall, ToolDefinition, ToolSpec
from tools.registry import JsonSchemaArgumentValidator, ToolAllowListAuthorizer, ToolRegistry
from tools.system_tools import create_default_tool_registry


class Provider:
    def __init__(self, call: ToolCall) -> None:
        self.call, self.tools, self.sent = call, (), False
    def complete(self, messages: Sequence[object], tools: Sequence[ToolDefinition]) -> ModelResponse:
        self.tools = tools
        if not self.sent:
            self.sent = True
            return ModelResponse(ModelFinishReason.TOOL_CALLS, tool_calls=(self.call,))
        return ModelResponse(ModelFinishReason.FINAL, content="done")


class Authenticator:
    def verify(self, approval: HostApproval, request: object) -> bool:
        return False


def make_core(provider: Provider, registry: ToolRegistry,
              allowed: Sequence[str] = ("get_system_info", "get_datetime")) -> AgentCore:
    return AgentCore(provider, JsonSchemaArgumentValidator(), ToolAllowListAuthorizer(allowed),
                     Authenticator(), tools=registry.specs, tool_executor=registry)


def make_fake_system_registry() -> ToolRegistry:
    registry = ToolRegistry()
    outputs = {
        "get_system_info": json.dumps({
            "os": "TestOS", "os_release": "1", "architecture": "test64",
            "processor": "Test CPU", "logical_cpu_count": 4,
            "total_memory_bytes": 1024,
        }),
        "get_datetime": json.dumps({"datetime": "2030-01-01T00:00:00+00:00", "timezone": "UTC"}),
    }
    for spec in create_default_tool_registry().specs:
        registry.register(spec, lambda _arguments, name=spec.name: outputs[name])
    return registry


class ToolRegistryTests(unittest.TestCase):
    def test_registration_lookup_and_unknown_lookup(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec("sample", "sample tool", {"type": "object"})
        registry.register(spec, lambda args: "ok")
        self.assertEqual(registry.lookup("sample"), spec)
        self.assertIsNone(registry.lookup("absent"))
        self.assertEqual(registry.specs, (spec,))

    def test_duplicate_registration_rejected(self) -> None:
        registry = ToolRegistry()
        spec = ToolSpec("sample", "sample", {"type": "object"})
        registry.register(spec, lambda args: "ok")
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(spec, lambda args: "again")

    def test_registered_schema_is_defensively_copied(self) -> None:
        registry, schema = ToolRegistry(), {"type": "object", "properties": {}}
        registry.register(ToolSpec("sample", "sample", schema), lambda args: "ok")
        schema["properties"]["bad"] = {"type": "string"}
        registry.lookup("sample").parameters_schema["properties"]["also_bad"] = {}
        self.assertEqual(registry.lookup("sample").parameters_schema,
                         {"type": "object", "properties": {}})

    def test_tools_have_closed_empty_argument_schemas(self) -> None:
        registry = make_fake_system_registry()
        self.assertEqual({item.name for item in registry.specs}, {"get_system_info", "get_datetime"})
        for spec in registry.specs:
            self.assertEqual(spec.parameters_schema, {"type": "object", "properties": {},
                "required": [], "additionalProperties": False})

    def test_argument_validation_required_types_and_extra_fields(self) -> None:
        validator = JsonSchemaArgumentValidator()
        schema = {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}},
            "required": ["query"], "additionalProperties": False}
        validator.validate(schema, {"query": "safe", "limit": 2})
        for bad in ({}, {"query": 5}, {"query": "safe", "limit": True},
                    {"query": "safe", "extra": "no"}):
            with self.subTest(arguments=bad), self.assertRaises(ValueError):
                validator.validate(schema, bad)

    def test_allow_list_authorization(self) -> None:
        auth = ToolAllowListAuthorizer(("get_datetime",))
        self.assertTrue(auth.is_allowed(ToolSpec("get_datetime", "", {}), {}))
        self.assertFalse(auth.is_allowed(ToolSpec("get_system_info", "", {}), {}))

    def test_unauthorized_agent_call_never_executes(self) -> None:
        registry = create_default_tool_registry()
        result = make_core(Provider(ToolCall("c1", "get_datetime", {})), registry, ()).run("time")
        self.assertEqual(result.status, AgentStatus.TOOL_NOT_AUTHORIZED)
        self.assertEqual(result.tool_executions[0].status.value, "not_authorized")

    def test_datetime_executes_through_agent_core_and_publishes_specs(self) -> None:
        registry, provider = make_fake_system_registry(), Provider(ToolCall("c1", "get_datetime", {}))
        result = make_core(provider, registry).run("time")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(json.loads(result.tool_executions[0].output),
                         {"datetime": "2030-01-01T00:00:00+00:00", "timezone": "UTC"})
        self.assertEqual({t.name for t in provider.tools}, {"get_system_info", "get_datetime"})

    def test_system_info_executes_through_agent_core(self) -> None:
        registry = make_fake_system_registry()
        result = make_core(Provider(ToolCall("c1", "get_system_info", {})), registry).run("info")
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        info = json.loads(result.tool_executions[0].output)
        self.assertEqual(info["os"], "TestOS")
        self.assertEqual(info["architecture"], "test64")
        self.assertEqual(info["logical_cpu_count"], 4)
        self.assertEqual(info["total_memory_bytes"], 1024)

    def test_invalid_arguments_do_not_execute(self) -> None:
        registry = make_fake_system_registry()
        result = make_core(Provider(ToolCall("c1", "get_datetime", {"extra": 1})), registry).run("time")
        self.assertEqual(result.status, AgentStatus.TOOL_ARGUMENT_ERROR)
        self.assertEqual(result.tool_executions[0].status.value, "invalid_arguments")

    def test_unregistered_dispatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(KeyError, "unknown tool"):
            create_default_tool_registry().execute(ToolCall("c1", "open_application", {}))


if __name__ == "__main__":
    unittest.main()
