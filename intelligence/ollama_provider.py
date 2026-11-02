from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Protocol

from ollama import Client

from config import OLLAMA_HOST, OLLAMA_MODEL
from core.agent import (
    Message,
    ModelFinishReason,
    ModelProvider,
    ModelResponse,
    ToolCall,
    ToolDefinition,
)


class OllamaProviderError(RuntimeError):
    """Raised when the Ollama request or response cannot satisfy the core contract."""


class _OllamaClient(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
        stream: bool = False,
    ) -> object:
        ...


_MISSING = object()


class OllamaProvider(ModelProvider):
    """Adapts the official Ollama chat client to Agent Core's model protocol."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        client: _OllamaClient | None = None,
    ) -> None:
        self._model = OLLAMA_MODEL if model is None else model
        self._host = OLLAMA_HOST if host is None else host
        if not isinstance(self._model, str) or not self._model.strip():
            raise ValueError("Ollama model name must be a non-empty string")
        if not isinstance(self._host, str) or not self._host.strip():
            raise ValueError("Ollama host must be a non-empty string")
        self._model = self._model.strip()
        self._host = self._host.strip()
        self._client = client if client is not None else Client(host=self._host)

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        ollama_messages = self._convert_messages(messages)
        ollama_tools = self._convert_tools(tools)

        try:
            response = self._client.chat(
                model=self._model,
                messages=ollama_messages,
                tools=ollama_tools,
                stream=False,
            )
        except Exception as exc:
            raise OllamaProviderError(
                f"Ollama chat request failed ({type(exc).__name__}): {exc}"
            ) from exc

        try:
            return self._convert_response(response)
        except OllamaProviderError:
            raise
        except Exception as exc:
            raise OllamaProviderError(
                f"Ollama returned a malformed response ({type(exc).__name__})."
            ) from exc

    @classmethod
    def _convert_messages(
        cls,
        messages: Sequence[Message],
    ) -> list[dict[str, object]]:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            raise OllamaProviderError("Agent messages must be a sequence.")

        converted: list[dict[str, object]] = []
        tool_names_by_id: dict[str, str] = {}
        seen_tool_call_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, Message):
                raise OllamaProviderError("Agent message has an unsupported shape.")
            if message.role not in ("user", "assistant", "tool"):
                raise OllamaProviderError("Agent message has an unsupported role.")
            if tool_names_by_id and message.role != "tool":
                raise OllamaProviderError(
                    "Tool-call history must resolve outstanding calls before another message."
                )
            if not isinstance(message.content, str):
                raise OllamaProviderError("Agent message content must be text.")
            if not isinstance(message.tool_calls, tuple):
                raise OllamaProviderError("Agent message tool calls must be a tuple.")

            item: dict[str, object] = {
                "role": message.role,
                "content": message.content,
            }
            if message.tool_calls:
                if message.role != "assistant":
                    raise OllamaProviderError(
                        "Only assistant messages may contain tool calls."
                    )
                converted_calls = []
                for call in message.tool_calls:
                    if not isinstance(call, ToolCall):
                        raise OllamaProviderError("Agent tool call has an unsupported shape.")
                    if not isinstance(call.id, str) or not call.id.strip():
                        raise OllamaProviderError("Agent tool call has an invalid ID.")
                    if call.id in seen_tool_call_ids:
                        raise OllamaProviderError("Agent tool-call history has duplicate IDs.")
                    if not isinstance(call.name, str) or not call.name.strip():
                        raise OllamaProviderError("Agent tool call has an invalid name.")
                    if not isinstance(call.arguments, Mapping):
                        raise OllamaProviderError("Agent tool arguments must be a mapping.")
                    arguments = deepcopy(dict(call.arguments))
                    if any(not isinstance(key, str) for key in arguments):
                        raise OllamaProviderError("Tool argument names must be strings.")
                    converted_calls.append(
                        {"function": {"name": call.name, "arguments": arguments}}
                    )
                    tool_names_by_id[call.id] = call.name
                    seen_tool_call_ids.add(call.id)
                item["tool_calls"] = converted_calls

            if message.role == "tool":
                if (
                    not isinstance(message.tool_call_id, str)
                    or not message.tool_call_id.strip()
                ):
                    raise OllamaProviderError(
                        "Tool-result message must reference a tool-call ID."
                    )
                tool_name = tool_names_by_id.get(message.tool_call_id)
                if tool_name is None:
                    raise OllamaProviderError(
                        "Tool-result message references no earlier assistant tool call."
                    )
                # Ollama associates tool results by tool name, not call ID.
                item["tool_name"] = tool_name
                del tool_names_by_id[message.tool_call_id]
            elif message.tool_call_id is not None:
                raise OllamaProviderError(
                    "Only tool-result messages may reference a tool-call ID."
                )
            converted.append(item)

        if tool_names_by_id:
            raise OllamaProviderError(
                "Agent tool-call history contains calls without corresponding results."
            )

        return converted

    @staticmethod
    def _convert_tools(
        tools: Sequence[ToolDefinition],
    ) -> list[dict[str, object]]:
        if isinstance(tools, (str, bytes)) or not isinstance(tools, Sequence):
            raise OllamaProviderError("Tool definitions must be a sequence.")

        converted: list[dict[str, object]] = []
        for tool in tools:
            if not isinstance(tool, ToolDefinition):
                raise OllamaProviderError("Tool definition has an unsupported shape.")
            if not isinstance(tool.name, str) or not tool.name.strip():
                raise OllamaProviderError("Tool definition has an invalid name.")
            if not isinstance(tool.description, str):
                raise OllamaProviderError("Tool description must be text.")
            if not isinstance(tool.parameters_schema, Mapping):
                raise OllamaProviderError("Tool parameters must be a JSON schema mapping.")
            converted.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": deepcopy(dict(tool.parameters_schema)),
                    },
                }
            )
        return converted

    @classmethod
    def _convert_response(cls, response: object) -> ModelResponse:
        message = cls._member(response, "message")
        if message is _MISSING:
            raise OllamaProviderError("Ollama response is missing its message.")

        role = cls._member(message, "role")
        if role != "assistant":
            raise OllamaProviderError("Ollama response message is not from the assistant.")

        content = cls._member(message, "content")
        if content is _MISSING or not isinstance(content, (str, type(None))):
            raise OllamaProviderError("Ollama response content has an invalid type.")

        raw_calls = cls._member(message, "tool_calls")
        if raw_calls is _MISSING or raw_calls is None:
            raw_calls = ()
        if isinstance(raw_calls, (str, bytes)) or not isinstance(raw_calls, Sequence):
            raise OllamaProviderError("Ollama tool calls have an invalid format.")

        tool_calls: list[ToolCall] = []
        for raw_call in raw_calls:
            function = cls._member(raw_call, "function")
            name = cls._member(function, "name")
            arguments = cls._member(function, "arguments")
            if not isinstance(name, str) or not name.strip():
                raise OllamaProviderError("Ollama returned a tool call with an invalid name.")
            if not isinstance(arguments, Mapping) or any(
                not isinstance(key, str) for key in arguments
            ):
                raise OllamaProviderError(
                    "Ollama returned tool arguments that are not a string-keyed mapping."
                )
            tool_calls.append(
                ToolCall(
                    id=uuid.uuid4().hex,
                    name=name,
                    arguments=deepcopy(dict(arguments)),
                )
            )

        if tool_calls:
            return ModelResponse(
                finish_reason=ModelFinishReason.TOOL_CALLS,
                content=content,
                tool_calls=tuple(tool_calls),
            )
        if not isinstance(content, str) or not content.strip():
            raise OllamaProviderError(
                "Ollama response contains neither text nor tool calls."
            )
        return ModelResponse(
            finish_reason=ModelFinishReason.FINAL,
            content=content,
        )

    @staticmethod
    def _member(value: object, name: str) -> object:
        if isinstance(value, Mapping):
            return value.get(name, _MISSING)
        return getattr(value, name, _MISSING)