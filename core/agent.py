from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from math import isfinite
from threading import Lock
from types import MappingProxyType
from typing import Callable, Literal, Protocol


class AgentStatus(Enum):
    COMPLETED = "completed"
    INVALID_INPUT = "invalid_input"
    MODEL_ERROR = "model_error"
    TOOL_ERROR = "tool_error"
    TOOL_NOT_FOUND = "tool_not_found"
    TOOL_ARGUMENT_ERROR = "tool_argument_error"
    TOOL_NOT_AUTHORIZED = "tool_not_authorized"
    TOOL_AUTHORIZATION_ERROR = "tool_authorization_error"
    NEEDS_CONFIRMATION = "needs_confirmation"
    CONFIRMATION_DENIED = "confirmation_denied"
    CONFIRMATION_EXPIRED = "confirmation_expired"
    CONFIRMATION_INVALID = "confirmation_invalid"
    PENDING_CONFIRMATION_LIMIT = "pending_confirmation_limit"
    APPROVAL_AUTHENTICATION_ERROR = "approval_authentication_error"
    TOOL_CALL_LIMIT = "tool_call_limit"
    STEP_LIMIT = "step_limit"


class ModelFinishReason(Enum):
    FINAL = "final"
    TOOL_CALLS = "tool_calls"


class ToolExecutionStatus(Enum):
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    NOT_AUTHORIZED = "not_authorized"
    AUTHORIZATION_ERROR = "authorization_error"
    DENIED_BY_USER = "denied_by_user"
    CONFIRMATION_EXPIRED = "confirmation_expired"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True)
class AgentError:
    code: str
    message: str


@dataclass(frozen=True)
class ToolDefinition:
    """The model-visible portion of a tool; core policy stays private."""

    name: str
    description: str
    parameters_schema: Mapping[str, object]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters_schema: Mapping[str, object]
    requires_confirmation: bool = False


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Mapping[str, object]


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ModelResponse:
    finish_reason: ModelFinishReason
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class ConfirmationRequest:
    id: str
    tool_name: str
    tool_description: str
    arguments: Mapping[str, object]
    expires_at: datetime


@dataclass(frozen=True)
class HostApproval:
    """A host-authenticated decision; model messages cannot create authority."""

    confirmation_id: str
    approved: bool
    principal_id: str
    proof: object


@dataclass(frozen=True)
class ToolExecutionRecord:
    call_id: str
    name: str
    arguments: Mapping[str, object]
    status: ToolExecutionStatus
    output: str | None = None
    error: AgentError | None = None


@dataclass(frozen=True)
class AgentResult:
    status: AgentStatus
    output: str | None = None
    error: AgentError | None = None
    tool_executions: tuple[ToolExecutionRecord, ...] = ()
    pending_confirmation: ConfirmationRequest | None = None
    transcript: tuple[Message, ...] = ()


class ModelProvider(Protocol):
    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        """Return a final response or a batch of tool calls."""


class ToolExecutor(Protocol):
    def execute(self, call: ToolCall) -> str:
        """Execute one validated and authorized tool call."""


class ToolArgumentValidator(Protocol):
    def validate(
        self,
        schema: Mapping[str, object],
        arguments: Mapping[str, object],
    ) -> None:
        """Return normally if valid; raise an exception otherwise."""


class ToolAuthorizer(Protocol):
    def is_allowed(
        self,
        tool: ToolSpec,
        arguments: Mapping[str, object],
    ) -> bool:
        """Return True only when this specific call is permitted."""


class HostApprovalAuthenticator(Protocol):
    def verify(
        self,
        approval: HostApproval,
        request: ConfirmationRequest,
    ) -> bool:
        """Verify host authentication, principal authority, and request binding."""


@dataclass
class _RunState:
    messages: list[Message]
    history_length: int = 0
    executions: list[ToolExecutionRecord] = field(default_factory=list)
    model_steps: int = 0
    tool_call_count: int = 0
    seen_tool_call_ids: set[str] = field(default_factory=set)
    batch: tuple[ToolCall, ...] = ()
    next_tool_index: int = 0


@dataclass(frozen=True)
class _PendingConfirmation:
    state: _RunState
    call: ToolCall
    request: ConfirmationRequest
    expires_at: datetime
    record_index: int


class AgentCore:
    """Synchronous orchestration core with injected, UI-independent ports."""

    def __init__(
        self,
        provider: ModelProvider,
        argument_validator: ToolArgumentValidator,
        authorizer: ToolAuthorizer,
        approval_authenticator: HostApprovalAuthenticator,
        tools: Sequence[ToolSpec] = (),
        tool_executor: ToolExecutor | None = None,
        max_steps: int = 8,
        max_tool_calls: int = 16,
        confirmation_ttl_seconds: float = 300,
        clock: Callable[[], datetime] | None = None,
        max_pending_confirmations: int = 100,
        max_history_messages: int = 40,
    ) -> None:
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if (
            isinstance(max_tool_calls, bool)
            or not isinstance(max_tool_calls, int)
            or max_tool_calls <= 0
        ):
            raise ValueError("max_tool_calls must be a positive integer")
        if (
            isinstance(confirmation_ttl_seconds, bool)
            or not isinstance(confirmation_ttl_seconds, (int, float))
            or not isfinite(confirmation_ttl_seconds)
            or confirmation_ttl_seconds <= 0
        ):
            raise ValueError("confirmation_ttl_seconds must be finite and positive")
        if (
            isinstance(max_pending_confirmations, bool)
            or not isinstance(max_pending_confirmations, int)
            or max_pending_confirmations <= 0
        ):
            raise ValueError("max_pending_confirmations must be a positive integer")
        if (
            isinstance(max_history_messages, bool)
            or not isinstance(max_history_messages, int)
            or max_history_messages <= 0
        ):
            raise ValueError("max_history_messages must be a positive integer")

        tool_list = tuple(tools)
        tool_map: dict[str, ToolSpec] = {}
        model_tools: list[ToolDefinition] = []
        for tool in tool_list:
            if not isinstance(tool, ToolSpec):
                raise ValueError("tools must contain ToolSpec instances")
            if (
                not isinstance(tool.name, str)
                or not tool.name
                or tool.name != tool.name.strip()
            ):
                raise ValueError("tool names must be non-empty with no surrounding whitespace")
            if tool.name in tool_map:
                raise ValueError(f"duplicate tool name: {tool.name}")
            if not isinstance(tool.description, str) or not isinstance(
                tool.parameters_schema, Mapping
            ):
                raise ValueError(f"invalid tool definition: {tool.name}")
            if not isinstance(tool.requires_confirmation, bool):
                raise ValueError("requires_confirmation must be a bool")

            schema = deepcopy(dict(tool.parameters_schema))
            trusted_tool = ToolSpec(
                name=tool.name,
                description=tool.description,
                parameters_schema=schema,
                requires_confirmation=tool.requires_confirmation,
            )
            tool_map[tool.name] = trusted_tool
            model_tools.append(
                ToolDefinition(tool.name, tool.description, deepcopy(schema))
            )

        if tool_list and tool_executor is None:
            raise ValueError("tool_executor is required when tools are configured")

        self._provider = provider
        self._argument_validator = argument_validator
        self._authorizer = authorizer
        self._approval_authenticator = approval_authenticator
        self._tools = tuple(model_tools)
        self._tool_map = tool_map
        self._tool_executor = tool_executor
        self._max_steps = max_steps
        self._max_tool_calls = max_tool_calls
        self._max_pending_confirmations = max_pending_confirmations
        self._max_history_messages = max_history_messages
        self._confirmation_ttl = float(confirmation_ttl_seconds)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._confirmation_lock = Lock()
        self._pending_confirmations: dict[str, _PendingConfirmation] = {}

        self._now()  # Fail fast if an injected clock does not return aware datetimes.

    def run(
        self,
        user_input: str,
        prior_messages: Sequence[Message] = (),
    ) -> AgentResult:
        if not isinstance(user_input, str) or not user_input.strip():
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error=AgentError("invalid_input", "Input must be a non-empty string."),
            )

        history = self._prepare_history(prior_messages)
        if history is None:
            return AgentResult(
                status=AgentStatus.INVALID_INPUT,
                error=AgentError("invalid_history", "Prior conversation history is invalid."),
            )

        state = _RunState(
            messages=[*history, Message(role="user", content=user_input)],
            history_length=len(history),
            seen_tool_call_ids={
                call.id for message in history for call in message.tool_calls
            },
        )
        return self._continue_run(state)

    def _prepare_history(
        self,
        messages: Sequence[Message],
    ) -> tuple[Message, ...] | None:
        if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
            return None
        try:
            history = tuple(deepcopy(message) for message in messages)
        except Exception:
            return None
        if self._turn_boundaries(history) is None:
            return None
        return self._trim_history(history)

    @staticmethod
    def _turn_boundaries(
        messages: Sequence[Message],
    ) -> list[tuple[int, int]] | None:
        # Reject incomplete turns and tool-call/result pairs.
        turns: list[tuple[int, int]] = []
        seen_call_ids: set[str] = set()
        index = 0
        while index < len(messages):
            start = index
            user = messages[index]
            if (
                not isinstance(user, Message)
                or user.role != "user"
                or not isinstance(user.content, str)
                or user.tool_calls
                or user.tool_call_id is not None
            ):
                return None
            index += 1
            while index < len(messages):
                assistant = messages[index]
                if (
                    not isinstance(assistant, Message)
                    or assistant.role != "assistant"
                    or not isinstance(assistant.content, str)
                    or assistant.tool_call_id is not None
                    or not isinstance(assistant.tool_calls, tuple)
                ):
                    return None
                index += 1
                if not assistant.tool_calls:
                    turns.append((start, index))
                    break
                calls = assistant.tool_calls
                if any(
                    not isinstance(call, ToolCall)
                    or not isinstance(call.id, str)
                    or not call.id.strip()
                    or call.id in seen_call_ids
                    for call in calls
                ):
                    return None
                ids = [call.id for call in calls]
                if len(ids) != len(set(ids)):
                    return None
                seen_call_ids.update(ids)
                for call_id in ids:
                    if index >= len(messages):
                        return None
                    result = messages[index]
                    if (
                        not isinstance(result, Message)
                        or result.role != "tool"
                        or not isinstance(result.content, str)
                        or result.tool_call_id != call_id
                        or result.tool_calls
                    ):
                        return None
                    index += 1
            else:
                return None
        return turns

    def _trim_history(self, messages: Sequence[Message]) -> tuple[Message, ...]:
        turns = self._turn_boundaries(messages)
        if turns is None:
            raise ValueError("only complete turns can be trimmed")
        start = 0
        count = len(messages)
        for _, end in turns[:-1]:
            if count <= self._max_history_messages:
                break
            start = end
            count = len(messages) - start
        return tuple(messages[start:])

    def _transcript_for_state(
        self,
        state: _RunState,
        *,
        include_current_turn: bool = False,
    ) -> tuple[Message, ...]:
        if include_current_turn:
            turns = self._turn_boundaries(state.messages)
            if turns is not None:
                return self._trim_history(state.messages)
        return self._trim_history(state.messages[:state.history_length])

    def resolve_confirmation(self, approval: HostApproval) -> AgentResult:
        if (
            not isinstance(approval, HostApproval)
            or not isinstance(approval.confirmation_id, str)
            or not approval.confirmation_id
            or not isinstance(approval.approved, bool)
            or not isinstance(approval.principal_id, str)
            or not approval.principal_id.strip()
        ):
            return self._failure(
                AgentStatus.CONFIRMATION_INVALID,
                "confirmation_invalid",
                "A valid host approval decision is required.",
            )

        with self._confirmation_lock:
            pending = self._pending_confirmations.get(approval.confirmation_id)
        if pending is None:
            return self._failure(
                AgentStatus.CONFIRMATION_INVALID,
                "confirmation_invalid",
                "Confirmation is unknown, expired, or already resolved.",
            )

        if self._is_expired(pending):
            if self._consume_if_current(approval.confirmation_id, pending):
                return self._expire_confirmation(pending)
            return self._failure(
                AgentStatus.CONFIRMATION_INVALID,
                "confirmation_invalid",
                "Confirmation is unknown, expired, or already resolved.",
            )

        try:
            authenticated = self._approval_authenticator.verify(
                approval, pending.request
            )
        except Exception:
            authenticated = False

        if authenticated is not True:
            return AgentResult(
                status=AgentStatus.APPROVAL_AUTHENTICATION_ERROR,
                error=AgentError(
                    "approval_authentication_error",
                    "The host could not authenticate this approval decision.",
                ),
                tool_executions=tuple(pending.state.executions),
                pending_confirmation=pending.request,
                transcript=self._transcript_for_state(pending.state),
            )

        # Authentication runs outside the lock. Recheck and remove atomically so
        # concurrent valid decisions cannot execute the same call twice.
        with self._confirmation_lock:
            current = self._pending_confirmations.get(approval.confirmation_id)
            if current is not pending:
                return self._failure(
                    AgentStatus.CONFIRMATION_INVALID,
                    "confirmation_invalid",
                    "Confirmation is unknown, expired, or already resolved.",
                    pending.state,
                )
            if self._is_expired(pending):
                del self._pending_confirmations[approval.confirmation_id]
                expired = True
            else:
                del self._pending_confirmations[approval.confirmation_id]
                expired = False

        if expired:
            return self._expire_confirmation(pending)

        state = pending.state
        if not approval.approved:
            self._replace_record(
                state,
                pending.record_index,
                pending.call,
                ToolExecutionStatus.DENIED_BY_USER,
                error=AgentError("confirmation_denied", "The host denied this tool call."),
            )
            return self._failure(
                AgentStatus.CONFIRMATION_DENIED,
                "confirmation_denied",
                "The host denied this tool call.",
                state,
            )

        tool, failure = self._prepare_call(pending.call)
        if failure is not None:
            status, record_status, error = failure
            self._replace_record(
                state, pending.record_index, pending.call, record_status, error=error
            )
            return self._failure(status, error.code, error.message, state)

        succeeded, output, error = self._execute(pending.call)
        if not succeeded:
            self._replace_record(
                state,
                pending.record_index,
                pending.call,
                ToolExecutionStatus.FAILED,
                error=error,
            )
            return self._failure(
                AgentStatus.TOOL_ERROR, error.code, error.message, state
            )

        self._replace_record(
            state,
            pending.record_index,
            pending.call,
            ToolExecutionStatus.SUCCEEDED,
            output=output,
        )
        state.messages.append(
            Message(role="tool", content=output, tool_call_id=pending.call.id)
        )
        state.next_tool_index += 1

        batch_result = self._advance_batch(state)
        if batch_result is not None:
            return batch_result
        return self._continue_run(state)

    def _continue_run(self, state: _RunState) -> AgentResult:
        while state.model_steps < self._max_steps:
            state.model_steps += 1
            try:
                messages = deepcopy(tuple(state.messages))
                response = self._provider.complete(messages, deepcopy(self._tools))
            except Exception as exc:
                return self._failure(
                    AgentStatus.MODEL_ERROR,
                    "model_error",
                    f"Model provider failed ({type(exc).__name__}).",
                    state,
                )

            if not isinstance(response, ModelResponse):
                return self._failure(
                    AgentStatus.MODEL_ERROR,
                    "invalid_model_response",
                    "Model provider returned an invalid response.",
                    state,
                )

            if response.finish_reason is ModelFinishReason.FINAL:
                if (
                    not isinstance(response.content, str)
                    or not response.content.strip()
                    or response.tool_calls != ()
                ):
                    return self._failure(
                        AgentStatus.MODEL_ERROR,
                        "invalid_final_response",
                        "A final response must contain text and no tool calls.",
                        state,
                    )
                state.messages.append(
                    Message(role="assistant", content=response.content)
                )
                return AgentResult(
                    status=AgentStatus.COMPLETED,
                    output=response.content,
                    tool_executions=tuple(state.executions),
                    transcript=self._transcript_for_state(
                        state, include_current_turn=True
                    ),
                )

            if (
                response.finish_reason is not ModelFinishReason.TOOL_CALLS
                or not isinstance(response.content, (str, type(None)))
            ):
                return self._failure(
                    AgentStatus.MODEL_ERROR,
                    "invalid_model_response",
                    "Model provider returned an invalid response.",
                    state,
                )

            calls = self._snapshot_tool_calls(response.tool_calls, state)
            if calls is None:
                return self._failure(
                    AgentStatus.MODEL_ERROR,
                    "invalid_tool_calls",
                    "Model provider returned invalid or repeated tool calls.",
                    state,
                )

            state.messages.append(
                Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=calls,
                )
            )
            state.batch = calls
            state.next_tool_index = 0

            batch_result = self._advance_batch(state)
            if batch_result is not None:
                return batch_result

        return self._failure(
            AgentStatus.STEP_LIMIT,
            "step_limit",
            "Agent reached the maximum number of model steps.",
            state,
        )

    def _advance_batch(self, state: _RunState) -> AgentResult | None:
        while state.next_tool_index < len(state.batch):
            call = state.batch[state.next_tool_index]
            if state.tool_call_count >= self._max_tool_calls:
                error = AgentError(
                    "tool_call_limit", "Agent reached the maximum number of tool calls."
                )
                state.executions.append(
                    self._record(call, ToolExecutionStatus.LIMIT_REACHED, error=error)
                )
                return self._failure(
                    AgentStatus.TOOL_CALL_LIMIT, error.code, error.message, state
                )

            state.tool_call_count += 1
            tool, failure = self._prepare_call(call)
            if failure is not None:
                status, record_status, error = failure
                state.executions.append(
                    self._record(call, record_status, error=error)
                )
                return self._failure(status, error.code, error.message, state)

            if tool.requires_confirmation:
                return self._request_confirmation(state, call, tool)

            succeeded, output, error = self._execute(call)
            if not succeeded:
                state.executions.append(
                    self._record(call, ToolExecutionStatus.FAILED, error=error)
                )
                return self._failure(
                    AgentStatus.TOOL_ERROR, error.code, error.message, state
                )

            state.executions.append(
                self._record(call, ToolExecutionStatus.SUCCEEDED, output=output)
            )
            state.messages.append(
                Message(role="tool", content=output, tool_call_id=call.id)
            )
            state.next_tool_index += 1

        state.batch = ()
        state.next_tool_index = 0
        return None

    def _request_confirmation(
        self,
        state: _RunState,
        call: ToolCall,
        tool: ToolSpec,
    ) -> AgentResult:
        try:
            now = self._now()
            expires_at = now + timedelta(seconds=self._confirmation_ttl)
            request_id = secrets.token_urlsafe(32)
            request = ConfirmationRequest(
                id=request_id,
                tool_name=tool.name,
                tool_description=tool.description,
                arguments=self._freeze_value(dict(call.arguments)),
                expires_at=expires_at,
            )
        except Exception as exc:
            error = AgentError(
                "confirmation_error",
                f"Could not create confirmation request ({type(exc).__name__}).",
            )
            state.executions.append(
                self._record(
                    call, ToolExecutionStatus.AUTHORIZATION_ERROR, error=error
                )
            )
            return self._failure(
                AgentStatus.TOOL_AUTHORIZATION_ERROR, error.code, error.message, state
            )

        record_index = len(state.executions)
        pending = _PendingConfirmation(
            state=state,
            call=call,
            request=request,
            expires_at=expires_at,
            record_index=record_index,
        )
        with self._confirmation_lock:
            expired = self._remove_expired_confirmations_locked(now)
            capacity_reached = (
                len(self._pending_confirmations) >= self._max_pending_confirmations
            )
            if not capacity_reached:
                state.executions.append(
                    self._record(call, ToolExecutionStatus.AWAITING_CONFIRMATION)
                )
                self._pending_confirmations[request_id] = pending

        for expired_pending in expired:
            self._mark_confirmation_expired(expired_pending)

        if capacity_reached:
            error = AgentError(
                "pending_confirmation_limit",
                "The maximum number of pending confirmations has been reached.",
            )
            state.executions.append(
                self._record(call, ToolExecutionStatus.LIMIT_REACHED, error=error)
            )
            return self._failure(
                AgentStatus.PENDING_CONFIRMATION_LIMIT,
                error.code,
                error.message,
                state,
            )

        return AgentResult(
            status=AgentStatus.NEEDS_CONFIRMATION,
            tool_executions=tuple(state.executions),
            pending_confirmation=request,
            transcript=self._transcript_for_state(state),
        )

    def _prepare_call(
        self,
        call: ToolCall,
    ) -> tuple[
        ToolSpec | None,
        tuple[AgentStatus, ToolExecutionStatus, AgentError] | None,
    ]:
        tool = self._tool_map.get(call.name)
        if tool is None:
            return None, (
                AgentStatus.TOOL_NOT_FOUND,
                ToolExecutionStatus.UNKNOWN_TOOL,
                AgentError("unknown_tool", "Requested tool is not available."),
            )

        try:
            self._argument_validator.validate(
                deepcopy(dict(tool.parameters_schema)),
                deepcopy(dict(call.arguments)),
            )
        except Exception:
            return None, (
                AgentStatus.TOOL_ARGUMENT_ERROR,
                ToolExecutionStatus.INVALID_ARGUMENTS,
                AgentError(
                    "invalid_tool_arguments", "Tool arguments did not pass validation."
                ),
            )

        try:
            allowed = self._authorizer.is_allowed(
                tool, deepcopy(dict(call.arguments))
            )
        except Exception:
            return None, (
                AgentStatus.TOOL_AUTHORIZATION_ERROR,
                ToolExecutionStatus.AUTHORIZATION_ERROR,
                AgentError(
                    "tool_authorization_error",
                    "Tool authorization could not be completed.",
                ),
            )

        if allowed is not True:
            return None, (
                AgentStatus.TOOL_NOT_AUTHORIZED,
                ToolExecutionStatus.NOT_AUTHORIZED,
                AgentError("tool_not_authorized", "Tool call was not authorized."),
            )

        return tool, None

    def _execute(self, call: ToolCall) -> tuple[bool, str | None, AgentError]:
        try:
            if self._tool_executor is None:
                raise RuntimeError("No tool executor is configured")
            output = self._tool_executor.execute(
                ToolCall(
                    id=call.id,
                    name=call.name,
                    arguments=deepcopy(dict(call.arguments)),
                )
            )
            if not isinstance(output, str):
                raise TypeError("Tool executor must return a string")
        except Exception as exc:
            return False, None, AgentError(
                "tool_execution_error",
                f"Tool execution failed ({type(exc).__name__}).",
            )
        return True, output, AgentError("", "")

    def _snapshot_tool_calls(
        self,
        calls: object,
        state: _RunState,
    ) -> tuple[ToolCall, ...] | None:
        if not isinstance(calls, tuple) or not calls:
            return None

        snapshots: list[ToolCall] = []
        call_ids: set[str] = set()
        try:
            for call in calls:
                if not isinstance(call, ToolCall):
                    return None
                if (
                    not isinstance(call.id, str)
                    or not call.id
                    or call.id in state.seen_tool_call_ids
                    or call.id in call_ids
                ):
                    return None
                if (
                    not isinstance(call.name, str)
                    or not call.name
                    or call.name != call.name.strip()
                ):
                    return None
                if not isinstance(call.arguments, Mapping) or any(
                    not isinstance(key, str) for key in call.arguments
                ):
                    return None
                snapshots.append(
                    ToolCall(
                        id=call.id,
                        name=call.name,
                        arguments=deepcopy(dict(call.arguments)),
                    )
                )
                call_ids.add(call.id)
        except Exception:
            return None

        state.seen_tool_call_ids.update(call_ids)
        return tuple(snapshots)

    def _replace_record(
        self,
        state: _RunState,
        index: int,
        call: ToolCall,
        status: ToolExecutionStatus,
        output: str | None = None,
        error: AgentError | None = None,
    ) -> None:
        state.executions[index] = self._record(
            call, status, output=output, error=error
        )

    @staticmethod
    def _record(
        call: ToolCall,
        status: ToolExecutionStatus,
        output: str | None = None,
        error: AgentError | None = None,
    ) -> ToolExecutionRecord:
        return ToolExecutionRecord(
            call_id=call.id,
            name=call.name,
            arguments=AgentCore._freeze_mapping(call.arguments),
            status=status,
            output=output,
            error=error,
        )

    def _expire_confirmation(self, pending: _PendingConfirmation) -> AgentResult:
        self._mark_confirmation_expired(pending)
        return self._failure(
            AgentStatus.CONFIRMATION_EXPIRED,
            "confirmation_expired",
            "The tool confirmation request expired.",
            pending.state,
        )

    def _mark_confirmation_expired(self, pending: _PendingConfirmation) -> None:
        self._replace_record(
            pending.state,
            pending.record_index,
            pending.call,
            ToolExecutionStatus.CONFIRMATION_EXPIRED,
            error=AgentError(
                "confirmation_expired", "The tool confirmation request expired."
            ),
        )

    def _remove_expired_confirmations_locked(
        self,
        now: datetime,
    ) -> list[_PendingConfirmation]:
        expired_ids = [
            request_id
            for request_id, pending in self._pending_confirmations.items()
            if now >= pending.expires_at
        ]
        return [self._pending_confirmations.pop(request_id) for request_id in expired_ids]

    def _consume_if_current(
        self,
        request_id: str,
        pending: _PendingConfirmation,
    ) -> bool:
        with self._confirmation_lock:
            if self._pending_confirmations.get(request_id) is not pending:
                return False
            del self._pending_confirmations[request_id]
            return True

    def _is_expired(self, pending: _PendingConfirmation) -> bool:
        return self._now() >= pending.expires_at

    def _now(self) -> datetime:
        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(timezone.utc)

    @classmethod
    def _freeze_value(cls, value: object) -> object:
        if isinstance(value, Mapping):
            return MappingProxyType(
                {key: cls._freeze_value(item) for key, item in value.items()}
            )
        if isinstance(value, (list, tuple)):
            return tuple(cls._freeze_value(item) for item in value)
        return deepcopy(value)

    @classmethod
    def _freeze_mapping(
        cls,
        value: Mapping[str, object],
    ) -> Mapping[str, object]:
        return MappingProxyType(
            {key: cls._freeze_value(item) for key, item in value.items()}
        )

    def _failure(
        self,
        status: AgentStatus,
        code: str,
        message: str,
        state: _RunState | None = None,
    ) -> AgentResult:
        return AgentResult(
            status=status,
            error=AgentError(code, message),
            tool_executions=tuple(state.executions) if state is not None else (),
            transcript=(
                self._transcript_for_state(state) if state is not None else ()
            ),
        )