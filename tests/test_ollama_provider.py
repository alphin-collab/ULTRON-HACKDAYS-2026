from __future__ import annotations

import unittest
from typing import Mapping, Sequence
from unittest.mock import patch

from ollama import ResponseError

from core.agent import (
    Message,
    ModelFinishReason,
    ToolCall,
    ToolDefinition,
)
from intelligence.ollama_provider import OllamaProvider, OllamaProviderError


class FakeOllamaClient:
    def __init__(self, response: object = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        stream: bool = False,
    ) -> object:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": stream,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


def response(
    *,
    content: str | None = "hello",
    tool_calls: object = None,
    role: str = "assistant",
) -> dict[str, object]:
    message: dict[str, object] = {"role": role, "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message}


class OllamaProviderTests(unittest.TestCase):
    def test_normal_text_response_becomes_final_model_response(self) -> None:
        client = FakeOllamaClient(response())
        provider = OllamaProvider(model="test-model", host="http://localhost", client=client)

        result = provider.complete([Message("user", "Hi")], [])

        self.assertEqual(result.finish_reason, ModelFinishReason.FINAL)
        self.assertEqual(result.content, "hello")
        self.assertEqual(result.tool_calls, ())
        self.assertEqual(client.calls[0]["model"], "test-model")
        self.assertFalse(client.calls[0]["stream"])

    def test_tool_call_response_becomes_core_tool_call(self) -> None:
        client = FakeOllamaClient(
            response(
                content=None,
                tool_calls=[
                    {
                        "function": {
                            "name": "lookup",
                            "arguments": {"query": "weather"},
                        }
                    }
                ],
            )
        )
        provider = OllamaProvider(client=client)

        result = provider.complete([Message("user", "Look up weather")], [])

        self.assertEqual(result.finish_reason, ModelFinishReason.TOOL_CALLS)
        self.assertEqual(len(result.tool_calls), 1)
        self.assertIsInstance(result.tool_calls[0], ToolCall)
        self.assertEqual(result.tool_calls[0].name, "lookup")
        self.assertEqual(result.tool_calls[0].arguments, {"query": "weather"})
        self.assertTrue(result.tool_calls[0].id)

    def test_messages_and_tool_schemas_convert_to_ollama_format(self) -> None:
        client = FakeOllamaClient(response())
        provider = OllamaProvider(client=client)
        messages = [
            Message("user", "Find a result"),
            Message(
                "assistant",
                "",
                tool_calls=(ToolCall("call-1", "lookup", {"query": "example"}),),
            ),
            Message("tool", "found", tool_call_id="call-1"),
        ]
        tools = [
            ToolDefinition(
                "lookup",
                "Find a result",
                {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]

        provider.complete(messages, tools)

        outgoing = client.calls[0]
        self.assertEqual(
            outgoing["messages"],
            [
                {"role": "user", "content": "Find a result"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "lookup",
                                "arguments": {"query": "example"},
                            }
                        }
                    ],
                },
                {"role": "tool", "content": "found", "tool_name": "lookup"},
            ],
        )
        self.assertEqual(
            outgoing["tools"],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Find a result",
                        "parameters": {
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                        },
                    },
                }
            ],
        )

    def test_unmatched_tool_result_id_is_rejected_before_client_call(self) -> None:
        client = FakeOllamaClient(response())
        provider = OllamaProvider(client=client)

        with self.assertRaisesRegex(OllamaProviderError, "no earlier assistant tool call"):
            provider.complete(
                [Message("user", "Hi"), Message("tool", "result", tool_call_id="missing")],
                [],
            )

        self.assertEqual(client.calls, [])

    def test_malformed_tool_call_histories_are_rejected_before_client_call(self) -> None:
        malformed_histories = [
            [Message("tool", "result")],
            [
                Message(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("call-1", "lookup", {}),),
                ),
            ],
            [
                Message(
                    "assistant",
                    "",
                    tool_calls=(
                        ToolCall("duplicate", "lookup", {}),
                        ToolCall("duplicate", "lookup", {}),
                    ),
                ),
                Message("tool", "result", tool_call_id="duplicate"),
            ],
            [
                Message(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("call-1", "lookup", {}),),
                ),
                Message("tool", "first result", tool_call_id="call-1"),
                Message("tool", "duplicate result", tool_call_id="call-1"),
            ],
            [
                Message(
                    "assistant",
                    "",
                    tool_calls=(ToolCall("call-1", "lookup", {}),),
                ),
                Message("user", "Interleaved before tool result"),
                Message("tool", "result", tool_call_id="call-1"),
            ],
        ]

        for messages in malformed_histories:
            with self.subTest(messages=messages):
                client = FakeOllamaClient(response())
                provider = OllamaProvider(client=client)
                with self.assertRaises(OllamaProviderError):
                    provider.complete(messages, [])
                self.assertEqual(client.calls, [])

    def test_malformed_responses_raise_provider_error(self) -> None:
        malformed_responses = [
            {},
            {"message": {"role": "user", "content": "not assistant"}},
            response(content="", tool_calls=[]),
            response(
                content=None,
                tool_calls=[{"function": {"name": "lookup", "arguments": []}}],
            ),
            response(
                content=None,
                tool_calls=[{"function": {"name": "", "arguments": {}}}],
            ),
            response(content=None, tool_calls={"unexpected": "mapping"}),
        ]

        for malformed in malformed_responses:
            with self.subTest(malformed=malformed):
                provider = OllamaProvider(client=FakeOllamaClient(malformed))
                with self.assertRaises(OllamaProviderError):
                    provider.complete([Message("user", "Hi")], [])

    def test_connection_errors_are_wrapped_without_becoming_final_responses(self) -> None:
        provider = OllamaProvider(
            client=FakeOllamaClient(error=ConnectionError("offline"))
        )

        with self.assertRaisesRegex(OllamaProviderError, "ConnectionError"):
            provider.complete([Message("user", "Hi")], [])

    def test_model_rejection_of_tools_is_reported_as_provider_error(self) -> None:
        provider = OllamaProvider(
            client=FakeOllamaClient(
                error=ResponseError("model does not support tools", status_code=400)
            )
        )
        tools = [ToolDefinition("lookup", "Look up data", {"type": "object"})]

        with self.assertRaisesRegex(OllamaProviderError, "ResponseError"):
            provider.complete([Message("user", "Look up data")], tools)

    def test_default_configuration_is_used_to_construct_client(self) -> None:
        with patch("intelligence.ollama_provider.Client") as client_factory:
            provider = OllamaProvider()

        self.assertEqual(provider._host, "http://localhost:11434")
        self.assertEqual(provider._model, "qwen3:4b")
        client_factory.assert_called_once_with(host="http://localhost:11434")

    def test_explicit_blank_model_or_host_is_rejected(self) -> None:
        for kwargs in (
            {"model": ""},
            {"model": " \t "},
            {"host": ""},
            {"host": " \t "},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                OllamaProvider(client=FakeOllamaClient(), **kwargs)


if __name__ == "__main__":
    unittest.main()