"""Terminal and optional push-to-talk runtime for ULTRON's local Ollama agent."""
from __future__ import annotations

import argparse
from collections.abc import Callable

from config import OLLAMA_HOST, OLLAMA_MODEL
from core.agent import AgentCore, AgentResult, AgentStatus, HostApproval
from intelligence.ollama_provider import OllamaProvider
from tools.registry import JsonSchemaArgumentValidator, ToolAllowListAuthorizer, ToolRegistry
from tools.system_tools import create_default_tool_registry
from voice import AudioRecorder, Speaker, Transcriber, VoiceUnavailableError, build_voice_adapters

EXIT_COMMANDS = frozenset({"/exit", "/quit", "/q"})
DEFAULT_WHISPER_MODEL = "tiny.en"


class _NoHostApproval:
    """System tools do not request approval; reject any unexpected request."""

    def verify(self, approval: HostApproval, request: object) -> bool:
        return False


def build_agent(
    *,
    provider: object | None = None,
    registry: ToolRegistry | None = None,
) -> AgentCore:
    """Build AgentCore with the configured provider and registered tools.

    Provider and registry injection keep runtime wiring testable without Ollama
    or real system calls.
    """
    tool_registry = registry if registry is not None else create_default_tool_registry()
    model_provider = provider if provider is not None else OllamaProvider()
    tool_names = tuple(spec.name for spec in tool_registry.specs)
    return AgentCore(
        provider=model_provider,
        argument_validator=JsonSchemaArgumentValidator(),
        authorizer=ToolAllowListAuthorizer(tool_names),
        approval_authenticator=_NoHostApproval(),
        tools=tool_registry.specs,
        tool_executor=tool_registry,
    )


def _result_text(result: AgentResult) -> str:
    if result.status is AgentStatus.COMPLETED:
        return result.output or ""
    if result.status is AgentStatus.MODEL_ERROR:
        return (
            f"Model request failed. Verify Ollama is running at {OLLAMA_HOST} "
            f"and that {OLLAMA_MODEL} is available."
        )
    if result.error is not None:
        return result.error.message
    return f"Request ended with status {result.status.value}."


def run_chat(
    agent: AgentCore,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], object] = print,
) -> None:
    """Run a synchronous terminal conversation until an exit command or EOF."""
    output_fn(f"ULTRON ready ({OLLAMA_MODEL} at {OLLAMA_HOST}). Type /exit to quit.")
    transcript = ()
    while True:
        try:
            user_text = input_fn("You: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye.")
            return

        command = user_text.strip().casefold()
        if command in EXIT_COMMANDS:
            output_fn("Goodbye.")
            return
        if not user_text.strip():
            continue

        try:
            result = agent.run(user_text, prior_messages=transcript)
        except KeyboardInterrupt:
            output_fn("\nRequest cancelled.")
            continue
        except Exception as exc:
            output_fn(f"Request error ({type(exc).__name__}). Check the Ollama service and try again.")
            continue

        transcript = getattr(result, "transcript", transcript)
        output_fn(f"ULTRON: {_result_text(result)}")


def run_voice_chat(
    agent: AgentCore,
    recorder: AudioRecorder,
    transcriber: Transcriber,
    speaker: Speaker,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], object] = print,
) -> bool:
    """Run one-utterance push-to-talk turns; return False if voice is unavailable.

    No capture begins until the user presses Enter at the explicit prompt.
    Each activation records one bounded sample and then stops.
    """
    output_fn("Push-to-talk ready. Press Enter to record one utterance, or type /exit to quit.")
    transcript = ()
    while True:
        try:
            activation = input_fn("[Enter] speak once | /exit: ")
        except (EOFError, KeyboardInterrupt):
            output_fn("\nGoodbye.")
            return True
        if activation.strip().casefold() in EXIT_COMMANDS:
            output_fn("Goodbye.")
            return True
        if activation.strip():
            output_fn("Press Enter to activate recording, or type /exit to quit.")
            continue

        output_fn("Recording one utterance now (maximum 30 seconds). Speak after this message.")
        try:
            pcm = recorder.capture()
            user_text = transcriber.transcribe(pcm).strip()
        except VoiceUnavailableError as exc:
            output_fn(f"Voice input unavailable: {exc}")
            return False
        except KeyboardInterrupt:
            output_fn("\nRecording cancelled.")
            continue
        except Exception as exc:
            output_fn(f"Voice input failed ({type(exc).__name__}); switching to terminal mode.")
            return False

        if not user_text:
            output_fn("No speech recognized. Press Enter to try again.")
            continue

        output_fn(f"You: {user_text}")
        try:
            result = agent.run(user_text, prior_messages=transcript)
        except KeyboardInterrupt:
            output_fn("\nRequest cancelled.")
            continue
        except Exception as exc:
            output_fn(f"Request error ({type(exc).__name__}). Check the Ollama service and try again.")
            continue

        transcript = getattr(result, "transcript", transcript)
        response = _result_text(result)
        output_fn(f"ULTRON: {response}")
        try:
            speaker.speak(response)
        except VoiceUnavailableError as exc:
            output_fn(f"Voice output unavailable: {exc}")
            return False
        except Exception as exc:
            output_fn(f"Voice output failed ({type(exc).__name__}); switching to terminal mode.")
            return False


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ULTRON local assistant.")
    parser.add_argument(
        "--voice",
        action="store_true",
        help="opt into one-utterance push-to-talk voice mode",
    )
    parser.add_argument(
        "--whisper-model",
        default=DEFAULT_WHISPER_MODEL,
        help=f"local Whisper model path or model name (default: {DEFAULT_WHISPER_MODEL})",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=8.0,
        help="maximum length of each explicit recording (default: 8 seconds)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        agent = build_agent()
    except Exception as exc:
        print(
            f"Startup error ({type(exc).__name__}): could not configure ULTRON for "
            f"{OLLAMA_MODEL} at {OLLAMA_HOST}. Check the Ollama settings and installation."
        )
        return 1

    if args.voice:
        try:
            recorder, transcriber, speaker = build_voice_adapters(
                args.whisper_model, seconds=args.record_seconds
            )
            voice_active = run_voice_chat(agent, recorder, transcriber, speaker)
        except (VoiceUnavailableError, ValueError) as exc:
            print(f"Voice mode unavailable: {exc}")
            voice_active = False
        if not voice_active:
            print("Continuing in terminal mode. Type /exit to quit.")
            run_chat(agent)
    else:
        run_chat(agent)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


