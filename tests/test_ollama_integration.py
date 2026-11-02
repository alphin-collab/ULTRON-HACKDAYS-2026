from __future__ import annotations

import os
import unittest
from collections.abc import Mapping, Sequence

from ollama import Client

from config import OLLAMA_HOST
from core.agent import (
    AgentCore,
    AgentStatus,
    Message,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolSpec,
)
from intelligence.ollama_provider import OllamaProvider


MODEL = "qwen3:4b"
TOOL_NAME = "compatibility_probe"
PROBE_VALUE = "ultron-test"
PROBE_RESULT = "compatibility_probe_result: ultron-test"


class ProbeArgumentValidator:
    def validate(
        self,
        schema: Mapping[str, object],
        arguments: Mapping[str, object],
    ) -> None:
        required = schema.get("required", ())
        if any(name not in arguments for name in required):
            raise ValueError("required argument is missing")
        if not isinstance(arguments.get("value"), str):
            raise ValueError("value must be a string")


class ProbeAuthorizer:
    def is_allowed(
        self,
        tool: ToolSpec,
        arguments: Mapping[str, object],
    ) -> bool:
        return tool.name == TOOL_NAME and arguments == {"value": PROBE_VALUE}


class ProbeExecutor:
    def __init__(self) -> None:
        self.calls: list[ToolCall] = []

    def execute(self, call: ToolCall) -> str:
        self.calls.append(call)
        return PROBE_RESULT


class RecordingOllamaProvider(OllamaProvider):
    def __init__(self) -> None:
        super().__init__(model=MODEL, host=OLLAMA_HOST, client=Client(host=OLLAMA_HOST, timeout=180))
        self.requests: list[tuple[Message, ...]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        self.requests.append(tuple(messages))
        return super().complete(messages, tools)


class UnusedApprovalAuthenticator:
    def verify(self, approval: object, request: object) -> bool:
        return False


@unittest.skipUnless(
    os.getenv("ULTRON_RUN_OLLAMA_INTEGRATION") == "1",
    "Set ULTRON_RUN_OLLAMA_INTEGRATION=1 to run the local Ollama integration test.",
)
class OllamaAgentIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            installed_models = Client(host=OLLAMA_HOST, timeout=5).list().models
        except Exception as exc:
            raise unittest.SkipTest(f"Ollama is unavailable at {OLLAMA_HOST}: {exc}") from exc

        model_names = {model.model for model in installed_models}
        if MODEL not in model_names:
            raise unittest.SkipTest(f"Ollama model {MODEL!r} is not installed.")

    def test_qwen_tool_call_executes_and_returns_final_response(self) -> None:
        executor = ProbeExecutor()
        provider = RecordingOllamaProvider()
        core = AgentCore(
            provider=provider,
            argument_validator=ProbeArgumentValidator(),
            authorizer=ProbeAuthorizer(),
            approval_authenticator=UnusedApprovalAuthenticator(),
            tools=(
                ToolSpec(
                    name=TOOL_NAME,
                    description="Return a fixed harmless probe result.",
                    parameters_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
            ),
            tool_executor=executor,
            max_steps=4,
            max_tool_calls=2,
        )

        result = core.run(
            "Call compatibility_probe exactly once with value set to "
            "ultron-test. After receiving its result, respond with exactly "
            "that result and no other text."
        )

        self.assertEqual(result.status, AgentStatus.COMPLETED, result.error)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0].name, TOOL_NAME)
        self.assertEqual(executor.calls[0].arguments, {"value": PROBE_VALUE})
        self.assertEqual(len(result.tool_executions), 1)
        self.assertEqual(result.tool_executions[0].output, PROBE_RESULT)
        self.assertIn(PROBE_VALUE, result.output or "")
        self.assertEqual(len(provider.requests), 2)
        self.assertTrue(
            any(
                message.role == "tool" and message.content == PROBE_RESULT
                for message in provider.requests[1]
            )
        )


if __name__ == "__main__":
    unittest.main()