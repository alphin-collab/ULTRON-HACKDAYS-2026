from __future__ import annotations

import threading
import unittest
from datetime import datetime, timedelta, timezone
from typing import Mapping, Sequence

from core.agent import (
    AgentCore,
    AgentStatus,
    ConfirmationRequest,
    HostApproval,
    ModelFinishReason,
    ModelResponse,
    Message,
    ToolCall,
    ToolDefinition,
    ToolExecutionStatus,
    ToolSpec,
)


class FakeProvider:
    def __init__(self, *responses: ModelResponse | Exception) -> None:
        self._responses = list(responses)
        self._lock = threading.Lock()
        self.calls: list[tuple[Sequence[Message], Sequence[ToolDefinition]]] = []

    def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolDefinition],
    ) -> ModelResponse:
        with self._lock:
            self.calls.append((messages, tools))
            if not self._responses:
                raise AssertionError("FakeProvider has no response queued")
            response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeExecutor:
    def __init__(self, outcomes: Mapping[str, str | Exception] | None = None) -> None:
        self.outcomes = dict(outcomes or {})
        self.calls: list[ToolCall] = []
        self._lock = threading.Lock()

    def execute(self, call: ToolCall) -> str:
        with self._lock:
            self.calls.append(call)
        outcome = self.outcomes.get(call.name, f"{call.name}-result")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeValidator:
    def validate(
        self,
        schema: Mapping[str, object],
        arguments: Mapping[str, object],
    ) -> None:
        if not isinstance(arguments, Mapping):
            raise ValueError("arguments must be a mapping")
        required = schema.get("required", ())
        if any(name not in arguments for name in required):
            raise ValueError("required argument is missing")
        if schema.get("additionalProperties") is False:
            properties = schema.get("properties", {})
            if any(name not in properties for name in arguments):
                raise ValueError("unexpected argument")


class FakeAuthorizer:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def is_allowed(
        self,
        tool: ToolSpec,
        arguments: Mapping[str, object],
    ) -> bool:
        self.calls.append((tool.name, dict(arguments)))
        return self.allowed


class FakeApprovalAuthenticator:
    def __init__(self, valid_proof: object = "host-proof") -> None:
        self.valid_proof = valid_proof
        self.calls: list[tuple[HostApproval, ConfirmationRequest]] = []

    def verify(
        self,
        approval: HostApproval,
        request: ConfirmationRequest,
    ) -> bool:
        self.calls.append((approval, request))
        return (
            approval.proof == self.valid_proof
            and approval.confirmation_id == request.id
            and approval.principal_id == "operator-1"
        )


class FakeClock:
    def __init__(self) -> None:
        self.current = datetime(2030, 1, 1, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


def final(text: str = "done") -> ModelResponse:
    return ModelResponse(ModelFinishReason.FINAL, content=text)


def tool_response(*calls: ToolCall) -> ModelResponse:
    return ModelResponse(ModelFinishReason.TOOL_CALLS, tool_calls=tuple(calls))


def call(name: str, call_id: str, **arguments: object) -> ToolCall:
    return ToolCall(call_id, name, arguments)


def tool(name: str, *, confirmation: bool = False) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"{name} description",
        parameters_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": [],
            "additionalProperties": False,
        },
        requires_confirmation=confirmation,
    )


class AgentCoreTests(unittest.TestCase):
    def make_core(
        self,
        provider: FakeProvider,
        *,
        tools: Sequence[ToolSpec] = (),
        executor: FakeExecutor | None = None,
        authorizer: FakeAuthorizer | None = None,
        authenticator: FakeApprovalAuthenticator | None = None,
        clock: FakeClock | None = None,
        max_steps: int = 8,
        max_tool_calls: int = 16,
        confirmation_ttl_seconds: float = 300,
        max_pending_confirmations: int = 100,
    ) -> AgentCore:
        return AgentCore(
            provider=provider,
            argument_validator=FakeValidator(),
            authorizer=authorizer or FakeAuthorizer(),
            approval_authenticator=authenticator or FakeApprovalAuthenticator(),
            tools=tools,
            tool_executor=executor,
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            confirmation_ttl_seconds=confirmation_ttl_seconds,
            max_pending_confirmations=max_pending_confirmations,
            clock=clock,
        )

    def approval(self, request: ConfirmationRequest, *, proof: object = "host-proof") -> HostApproval:
        return HostApproval(
            confirmation_id=request.id,
            approved=True,
            principal_id="operator-1",
            proof=proof,
        )

    def test_final_response_completes(self) -> None:
        core = self.make_core(FakeProvider(final("hello")))

        result = core.run("Hi")

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(result.output, "hello")

    def test_empty_input_does_not_call_provider(self) -> None:
        provider = FakeProvider(final())
        core = self.make_core(provider)

        result = core.run(" \n ")

        self.assertEqual(result.status, AgentStatus.INVALID_INPUT)
        self.assertEqual(provider.calls, [])

    def test_provider_exception_is_not_completion_and_keeps_records(self) -> None:
        provider = FakeProvider(
            tool_response(call("read", "read-1")),
            RuntimeError("provider unavailable"),
        )
        executor = FakeExecutor()
        core = self.make_core(provider, tools=[tool("read")], executor=executor)

        result = core.run("Read something")

        self.assertEqual(result.status, AgentStatus.MODEL_ERROR)
        self.assertEqual([r.status for r in result.tool_executions], [ToolExecutionStatus.SUCCEEDED])
        self.assertEqual(len(executor.calls), 1)

    def test_unknown_tool_is_never_executed(self) -> None:
        provider = FakeProvider(tool_response(call("missing", "missing-1")), final())
        executor = FakeExecutor()
        core = self.make_core(provider, executor=executor)

        result = core.run("Use a tool")

        self.assertEqual(result.status, AgentStatus.TOOL_NOT_FOUND)
        self.assertEqual(result.tool_executions[0].status, ToolExecutionStatus.UNKNOWN_TOOL)
        self.assertEqual(executor.calls, [])

    def test_invalid_arguments_are_not_executed(self) -> None:
        spec = ToolSpec(
            "write", "write", {"required": ["value"], "additionalProperties": False}
        )
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(tool_response(call("write", "write-1"))),
            tools=[spec],
            executor=executor,
        )

        result = core.run("Write")

        self.assertEqual(result.status, AgentStatus.TOOL_ARGUMENT_ERROR)
        self.assertEqual(executor.calls, [])
        self.assertEqual(
            result.tool_executions[0].status, ToolExecutionStatus.INVALID_ARGUMENTS
        )

    def test_authorization_denial_is_not_executed(self) -> None:
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(tool_response(call("read", "read-1"))),
            tools=[tool("read")],
            executor=executor,
            authorizer=FakeAuthorizer(allowed=False),
        )

        result = core.run("Read")

        self.assertEqual(result.status, AgentStatus.TOOL_NOT_AUTHORIZED)
        self.assertEqual(executor.calls, [])

    def test_executor_failure_is_not_completion(self) -> None:
        executor = FakeExecutor({"write": RuntimeError("failed")})
        core = self.make_core(
            FakeProvider(tool_response(call("write", "write-1")), final()),
            tools=[tool("write")],
            executor=executor,
        )

        result = core.run("Write")

        self.assertEqual(result.status, AgentStatus.TOOL_ERROR)
        self.assertEqual(result.tool_executions[0].status, ToolExecutionStatus.FAILED)
        self.assertEqual(len(executor.calls), 1)

    def test_later_tool_failure_preserves_successful_execution_records(self) -> None:
        executor = FakeExecutor({"fail": RuntimeError("failed")})
        provider = FakeProvider(
            tool_response(call("first", "first-1"), call("fail", "fail-1")),
            final(),
        )
        core = self.make_core(
            provider,
            tools=[tool("first"), tool("fail")],
            executor=executor,
        )

        result = core.run("Do both")

        self.assertEqual(result.status, AgentStatus.TOOL_ERROR)
        self.assertEqual(
            [record.status for record in result.tool_executions],
            [ToolExecutionStatus.SUCCEEDED, ToolExecutionStatus.FAILED],
        )
        self.assertEqual([item.name for item in executor.calls], ["first", "fail"])

    def test_sensitive_call_waits_for_host_approval_not_model_output(self) -> None:
        provider = FakeProvider(
            tool_response(call("delete", "delete-1", value="file")),
            final("the model says it is approved"),
        )
        executor = FakeExecutor()
        core = self.make_core(
            provider,
            tools=[tool("delete", confirmation=True)],
            executor=executor,
        )

        result = core.run("Delete file")

        self.assertEqual(result.status, AgentStatus.NEEDS_CONFIRMATION)
        self.assertIsNotNone(result.pending_confirmation)
        self.assertEqual(len(provider.calls), 1)
        self.assertEqual(executor.calls, [])
        self.assertEqual(
            result.tool_executions[0].status,
            ToolExecutionStatus.AWAITING_CONFIRMATION,
        )

    def test_confirmation_arguments_cannot_be_mutated_by_host_caller(self) -> None:
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(
                tool_response(call("delete", "delete-1", value="original")),
                final("deleted"),
            ),
            tools=[tool("delete", confirmation=True)],
            executor=executor,
        )

        pending = core.run("Delete")
        request = pending.pending_confirmation
        self.assertIsNotNone(request)
        with self.assertRaises(TypeError):
            request.arguments["value"] = "altered"

        result = core.resolve_confirmation(self.approval(request))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(executor.calls[0].arguments["value"], "original")

    def test_execution_record_arguments_are_deeply_immutable(self) -> None:
        nested_arguments = {"inner": {"items": ["original"]}}
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(
                tool_response(call("delete", "delete-1", value=nested_arguments)),
                final("deleted"),
            ),
            tools=[tool("delete", confirmation=True)],
            executor=executor,
        )

        pending = core.run("Delete")
        record = pending.tool_executions[0]
        with self.assertRaises(TypeError):
            record.arguments["value"]["inner"]["items"][0] = "altered"
        with self.assertRaises(TypeError):
            record.arguments["value"]["inner"]["new"] = "altered"

        result = core.resolve_confirmation(self.approval(pending.pending_confirmation))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual(
            result.tool_executions[0].arguments["value"]["inner"]["items"],
            ("original",),
        )

    def test_pending_confirmation_capacity_is_enforced(self) -> None:
        provider = FakeProvider(
            tool_response(call("delete", "delete-1")),
            tool_response(call("delete", "delete-2")),
            final("deleted"),
        )
        executor = FakeExecutor()
        core = self.make_core(
            provider,
            tools=[tool("delete", confirmation=True)],
            executor=executor,
            max_pending_confirmations=1,
        )

        first = core.run("Delete one")
        second = core.run("Delete two")

        self.assertEqual(first.status, AgentStatus.NEEDS_CONFIRMATION)
        self.assertEqual(second.status, AgentStatus.PENDING_CONFIRMATION_LIMIT)
        self.assertIsNone(second.pending_confirmation)
        self.assertEqual(
            [record.status for record in second.tool_executions],
            [ToolExecutionStatus.LIMIT_REACHED],
        )
        self.assertEqual(executor.calls, [])

        resolved = core.resolve_confirmation(self.approval(first.pending_confirmation))
        self.assertEqual(resolved.status, AgentStatus.COMPLETED)
        self.assertEqual(len(executor.calls), 1)

    def test_new_confirmation_cleans_up_expired_pending_confirmation(self) -> None:
        clock = FakeClock()
        provider = FakeProvider(
            tool_response(call("delete", "delete-1")),
            tool_response(call("delete", "delete-2")),
            final("deleted"),
        )
        executor = FakeExecutor()
        core = self.make_core(
            provider,
            tools=[tool("delete", confirmation=True)],
            executor=executor,
            clock=clock,
            confirmation_ttl_seconds=5,
            max_pending_confirmations=1,
        )

        expired = core.run("Delete one")
        clock.advance(timedelta(seconds=5))
        current = core.run("Delete two")

        self.assertEqual(current.status, AgentStatus.NEEDS_CONFIRMATION)
        self.assertEqual(
            core.resolve_confirmation(self.approval(expired.pending_confirmation)).status,
            AgentStatus.CONFIRMATION_INVALID,
        )
        resolved = core.resolve_confirmation(self.approval(current.pending_confirmation))
        self.assertEqual(resolved.status, AgentStatus.COMPLETED)
        self.assertEqual([item.id for item in executor.calls], ["delete-2"])

    def test_untrusted_host_decision_does_not_consume_confirmation(self) -> None:
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(
                tool_response(call("delete", "delete-1")),
                final("deleted"),
            ),
            tools=[tool("delete", confirmation=True)],
            executor=executor,
        )
        pending = core.run("Delete")
        request = pending.pending_confirmation
        self.assertIsNotNone(request)

        rejected = core.resolve_confirmation(self.approval(request, proof="forged"))

        self.assertEqual(rejected.status, AgentStatus.APPROVAL_AUTHENTICATION_ERROR)
        self.assertEqual(executor.calls, [])
        self.assertIsNotNone(rejected.pending_confirmation)

        accepted = core.resolve_confirmation(self.approval(request))
        self.assertEqual(accepted.status, AgentStatus.COMPLETED)
        self.assertEqual(len(executor.calls), 1)

    def test_expired_confirmation_never_executes(self) -> None:
        clock = FakeClock()
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(tool_response(call("delete", "delete-1"))),
            tools=[tool("delete", confirmation=True)],
            executor=executor,
            clock=clock,
            confirmation_ttl_seconds=5,
        )
        pending = core.run("Delete")
        request = pending.pending_confirmation
        self.assertIsNotNone(request)
        clock.advance(timedelta(seconds=5))

        result = core.resolve_confirmation(self.approval(request))

        self.assertEqual(result.status, AgentStatus.CONFIRMATION_EXPIRED)
        self.assertEqual(executor.calls, [])
        self.assertEqual(
            result.tool_executions[-1].status,
            ToolExecutionStatus.CONFIRMATION_EXPIRED,
        )

    def test_confirmation_id_is_atomic_and_one_use_under_concurrency(self) -> None:
        provider = FakeProvider(
            tool_response(call("delete", "delete-1")),
            final("deleted"),
        )
        executor = FakeExecutor()
        core = self.make_core(
            provider,
            tools=[tool("delete", confirmation=True)],
            executor=executor,
        )
        pending = core.run("Delete")
        approval = self.approval(pending.pending_confirmation)
        barrier = threading.Barrier(3)
        results = []

        def resolve() -> None:
            barrier.wait()
            results.append(core.resolve_confirmation(approval))

        threads = [threading.Thread(target=resolve) for _ in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertCountEqual(
            [result.status for result in results],
            [AgentStatus.COMPLETED, AgentStatus.CONFIRMATION_INVALID],
        )
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(len(provider.calls), 2)

    def test_pending_batch_resumes_at_next_call_without_repeating_successes(self) -> None:
        provider = FakeProvider(
            tool_response(
                call("first", "first-1"),
                call("sensitive", "sensitive-1"),
                call("last", "last-1"),
            ),
            final("all done"),
        )
        executor = FakeExecutor()
        core = self.make_core(
            provider,
            tools=[tool("first"), tool("sensitive", confirmation=True), tool("last")],
            executor=executor,
        )

        pending = core.run("Do all")
        self.assertEqual(pending.status, AgentStatus.NEEDS_CONFIRMATION)
        self.assertEqual([item.name for item in executor.calls], ["first"])
        self.assertEqual(
            [record.status for record in pending.tool_executions],
            [ToolExecutionStatus.SUCCEEDED, ToolExecutionStatus.AWAITING_CONFIRMATION],
        )

        result = core.resolve_confirmation(self.approval(pending.pending_confirmation))

        self.assertEqual(result.status, AgentStatus.COMPLETED)
        self.assertEqual([item.name for item in executor.calls], ["first", "sensitive", "last"])
        self.assertEqual(
            [record.status for record in result.tool_executions],
            [
                ToolExecutionStatus.SUCCEEDED,
                ToolExecutionStatus.SUCCEEDED,
                ToolExecutionStatus.SUCCEEDED,
            ],
        )
        self.assertEqual(len(provider.calls), 2)

    def test_denial_is_recorded_and_consumes_confirmation(self) -> None:
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(tool_response(call("delete", "delete-1"))),
            tools=[tool("delete", confirmation=True)],
            executor=executor,
        )
        pending = core.run("Delete")
        request = pending.pending_confirmation
        denial = HostApproval(request.id, False, "operator-1", "host-proof")

        result = core.resolve_confirmation(denial)
        replay = core.resolve_confirmation(denial)

        self.assertEqual(result.status, AgentStatus.CONFIRMATION_DENIED)
        self.assertEqual(
            result.tool_executions[-1].status, ToolExecutionStatus.DENIED_BY_USER
        )
        self.assertEqual(replay.status, AgentStatus.CONFIRMATION_INVALID)
        self.assertEqual(executor.calls, [])

    def test_tool_call_limit_counts_calls_in_a_batch(self) -> None:
        executor = FakeExecutor()
        core = self.make_core(
            FakeProvider(tool_response(call("one", "one-1"), call("two", "two-1"))),
            tools=[tool("one"), tool("two")],
            executor=executor,
            max_tool_calls=1,
        )

        result = core.run("Do both")

        self.assertEqual(result.status, AgentStatus.TOOL_CALL_LIMIT)
        self.assertEqual([item.name for item in executor.calls], ["one"])
        self.assertEqual(
            [record.status for record in result.tool_executions],
            [ToolExecutionStatus.SUCCEEDED, ToolExecutionStatus.LIMIT_REACHED],
        )

    def test_successful_tool_is_not_repeated_if_a_later_model_call_fails(self) -> None:
        executor = FakeExecutor()
        provider = FakeProvider(
            tool_response(call("read", "read-1")),
            RuntimeError("provider failed after tool ran"),
        )
        core = self.make_core(provider, tools=[tool("read")], executor=executor)

        result = core.run("Read")

        self.assertEqual(result.status, AgentStatus.MODEL_ERROR)
        self.assertEqual([item.name for item in executor.calls], ["read"])
        self.assertEqual(len(result.tool_executions), 1)
        self.assertEqual(result.tool_executions[0].status, ToolExecutionStatus.SUCCEEDED)

    def test_limits_and_confirmation_ttl_are_validated(self) -> None:
        provider = FakeProvider(final())
        dependencies = dict(
            provider=provider,
            argument_validator=FakeValidator(),
            authorizer=FakeAuthorizer(),
            approval_authenticator=FakeApprovalAuthenticator(),
        )

        for kwargs in (
            {"max_steps": 0},
            {"max_steps": True},
            {"max_tool_calls": 0},
            {"max_tool_calls": -1},
            {"confirmation_ttl_seconds": 0},
            {"confirmation_ttl_seconds": float("inf")},
            {"max_pending_confirmations": 0},
            {"max_pending_confirmations": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AgentCore(**dependencies, **kwargs)


    def test_transcript_carries_ordinary_turns_without_duplication(self) -> None:
        provider = FakeProvider(final("first reply"), final("follow-up reply"))
        core = self.make_core(provider)

        first = core.run("first question")
        second = core.run("follow-up", prior_messages=first.transcript)

        self.assertEqual(
            [(message.role, message.content) for message in second.transcript],
            [("user", "first question"), ("assistant", "first reply"),
             ("user", "follow-up"), ("assistant", "follow-up reply")],
        )
        self.assertEqual(
            [message.content for message in provider.calls[1][0]],
            ["first question", "first reply", "follow-up"],
        )

    def test_transcript_carries_tool_call_and_result_across_turns(self) -> None:
        provider = FakeProvider(
            tool_response(call("read", "read-1")), final("read completed"),
            final("that result was read-result"),
        )
        core = self.make_core(provider, tools=[tool("read")], executor=FakeExecutor())

        first = core.run("read something")
        second = core.run("what did it return?", prior_messages=first.transcript)

        messages = second.transcript
        self.assertEqual(
            [(message.role, message.content) for message in messages],
            [("user", "read something"), ("assistant", ""),
             ("tool", "read-result"), ("assistant", "read completed"),
             ("user", "what did it return?"),
             ("assistant", "that result was read-result")],
        )
        self.assertEqual(messages[1].tool_calls[0].id, messages[2].tool_call_id)
        self.assertEqual(len([m for m in messages if m.role == "user"]), 2)
        self.assertEqual(len([m for m in messages if m.role == "tool"]), 1)
        self.assertEqual(len(provider.calls[2][0]), len(first.transcript) + 1)

    def test_failed_turn_keeps_prior_history_and_drops_incomplete_tool_unit(self) -> None:
        provider = FakeProvider(final("kept"), tool_response(call("read", "read-2")),
                                RuntimeError("offline"), final("recovered"))
        core = self.make_core(provider, tools=[tool("read")], executor=FakeExecutor())
        first = core.run("first")
        failed = core.run("try tool", prior_messages=first.transcript)
        self.assertEqual(failed.status, AgentStatus.MODEL_ERROR)
        self.assertEqual(failed.transcript, first.transcript)

        recovered = core.run("retry", prior_messages=failed.transcript)
        self.assertEqual(recovered.status, AgentStatus.COMPLETED)
        self.assertEqual([m.content for m in recovered.transcript if m.role == "user"],
                         ["first", "retry"])

    def test_history_limit_trims_only_whole_turns_and_tool_units(self) -> None:
        provider = FakeProvider(final("one"), tool_response(call("read", "read-3")),
                                final("two"), final("three"))
        core = AgentCore(provider, FakeValidator(), FakeAuthorizer(),
                         FakeApprovalAuthenticator(), tools=[tool("read")],
                         tool_executor=FakeExecutor(), max_history_messages=7)
        first = core.run("first")
        second = core.run("second", prior_messages=first.transcript)
        third = core.run("third", prior_messages=second.transcript)

        self.assertEqual(
            [(message.role, message.content) for message in third.transcript],
            [("user", "second"), ("assistant", ""), ("tool", "read-result"),
             ("assistant", "two"), ("user", "third"), ("assistant", "three")],
        )
        self.assertEqual(len(third.transcript), 6)
        self.assertEqual(AgentCore._turn_boundaries(third.transcript), [(0, 4), (4, 6)])

    def test_incomplete_prior_transcript_is_rejected(self) -> None:
        core = self.make_core(FakeProvider(final()))
        result = core.run("next", prior_messages=(
            Message("user", "old"),
            Message("assistant", "", tool_calls=(call("read", "orphan"),)),
        ))
        self.assertEqual(result.status, AgentStatus.INVALID_INPUT)
        self.assertEqual(result.error.code, "invalid_history")



if __name__ == "__main__":
    unittest.main()