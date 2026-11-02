from __future__ import annotations

import unittest
from collections.abc import Sequence

from core.agent import (
    AgentCore,
    AgentStatus,
    HostApproval,
    ModelFinishReason,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)
from main import build_agent, run_chat
from tools.registry import ToolRegistry
from tools.system_tools import create_default_tool_registry


class FakeProvider:
    def __init__(self, *responses: ModelResponse) -> None:
        self.responses = list(responses)
        self.tools_seen: list[tuple[ToolDefinition, ...]] = []
        self.messages_seen: list[tuple[object, ...]] = []

    def complete(self, messages: Sequence[object], tools: Sequence[ToolDefinition]) -> ModelResponse:
        self.messages_seen.append(tuple(messages))
        self.tools_seen.append(tuple(tools))
        return self.responses.pop(0)


class NeverApprove:
    def verify(self, approval: HostApproval, request: object) -> bool:
        return False


class RuntimeTests(unittest.TestCase):
    def test_system_tool_call_is_published_and_dispatched_through_registry(self) -> None:
        defaults = create_default_tool_registry()
        registry, dispatched = ToolRegistry(), []
        for spec in defaults.specs:
            registry.register(spec, lambda args, name=spec.name: dispatched.append(name) or '{"fake":true}')

        provider = FakeProvider(
            ModelResponse(
                ModelFinishReason.TOOL_CALLS,
                tool_calls=(ToolCall("call-1", "get_datetime", {}),),
            ),
            ModelResponse(ModelFinishReason.FINAL, content="The fake time is ready."),
        )
        agent = build_agent(provider=provider, registry=registry)

        result = agent.run("What time is it?")

        self.assertIsInstance(agent, AgentCore)
        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.output, "The fake time is ready.")
        self.assertEqual(dispatched, ["get_datetime"])
        self.assertEqual({tool.name for tool in provider.tools_seen[0]}, {"get_datetime", "get_system_info"})
        self.assertEqual(provider.messages_seen[1][-1].content, '{"fake":true}')

    def test_terminal_loop_handles_multiple_requests_and_exit(self) -> None:
        class FakeAgent:
            def __init__(self) -> None:
                self.inputs: list[str] = []
                self.histories: list[tuple[object, ...]] = []

            def run(self, value: str, prior_messages: Sequence[object] = ()) -> object:
                self.inputs.append(value)
                self.histories.append(tuple(prior_messages))
                transcript = (*prior_messages, value)
                return type("Result", (), {
                    "status": AgentStatus.COMPLETED,
                    "output": f"reply to {value}",
                    "error": None,
                    "transcript": transcript,
                })()

        agent = FakeAgent()
        prompts = iter(["hello", "  ", "second message", "/q"])
        output: list[str] = []

        run_chat(agent, input_fn=lambda _prompt: next(prompts), output_fn=output.append)

        self.assertEqual(agent.inputs, ["hello", "second message"])
        self.assertEqual(agent.histories[0], ())
        self.assertEqual(len(agent.histories[1]), 1)
        self.assertTrue(any("reply to hello" in line for line in output))
        self.assertTrue(any("reply to second message" in line for line in output))
        self.assertEqual(output[-1], "Goodbye.")

    def test_model_errors_are_reported_and_loop_continues(self) -> None:
        class FailingAgent:
            def __init__(self) -> None:
                self.calls = 0

            def run(self, value: str, prior_messages: Sequence[object] = ()) -> object:
                self.calls += 1
                if self.calls == 1:
                    return type("Result", (), {
                        "status": AgentStatus.MODEL_ERROR,
                        "output": None,
                        "error": type("Error", (), {"message": "Model provider failed (OllamaProviderError)."})(),
                    })()
                return type("Result", (), {
                    "status": AgentStatus.COMPLETED,
                    "output": "recovered",
                    "error": None,
                })()

        prompts = iter(["first", "again", "/exit"])
        output: list[str] = []
        run_chat(FailingAgent(), input_fn=lambda _prompt: next(prompts), output_fn=output.append)
        self.assertTrue(any("Model request failed" in line for line in output))
        self.assertTrue(any("recovered" in line for line in output))

    def test_eof_exits_cleanly(self) -> None:
        output: list[str] = []

        def end_input(_prompt: str) -> str:
            raise EOFError

        run_chat(object(), input_fn=end_input, output_fn=output.append)
        self.assertEqual(output[-1], "\nGoodbye.")


if __name__ == "__main__":
    unittest.main()
