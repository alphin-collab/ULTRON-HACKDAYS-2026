from __future__ import annotations

import unittest
from collections.abc import Sequence

from core.agent import AgentStatus
from main import run_voice_chat
from voice import SoundDeviceRecorder, VoiceUnavailableError


class FakeAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def run(self, text: str, prior_messages: Sequence[object] = ()) -> object:
        self.calls.append((text, tuple(prior_messages)))
        return type("Result", (), {
            "status": AgentStatus.COMPLETED,
            "output": f"heard: {text}",
            "error": None,
            "transcript": (*prior_messages, text, f"heard: {text}"),
        })()


class FakeRecorder:
    def __init__(self, *utterances: bytes) -> None:
        self.utterances = list(utterances)
        self.activations = 0

    def capture(self) -> bytes:
        self.activations += 1
        return self.utterances.pop(0)


class FakeTranscriber:
    def __init__(self, results: Sequence[str]) -> None:
        self.results = list(results)
        self.audio_seen: list[bytes] = []

    def transcribe(self, pcm: bytes) -> str:
        self.audio_seen.append(pcm)
        return self.results.pop(0)


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def speak(self, text: str) -> None:
        self.spoken.append(text)


class VoiceRuntimeTests(unittest.TestCase):
    def test_activation_transcribes_runs_agent_speaks_and_keeps_history(self) -> None:
        agent = FakeAgent()
        recorder = FakeRecorder(b"first pcm", b"second pcm")
        transcriber = FakeTranscriber(("hello ultron", "follow up"))
        speaker = FakeSpeaker()
        prompts = iter(("", "", "/exit"))
        output: list[str] = []

        active = run_voice_chat(
            agent,
            recorder,
            transcriber,
            speaker,
            input_fn=lambda _prompt: next(prompts),
            output_fn=output.append,
        )

        self.assertTrue(active)
        self.assertEqual(recorder.activations, 2)
        self.assertEqual(transcriber.audio_seen, [b"first pcm", b"second pcm"])
        self.assertEqual([call[0] for call in agent.calls], ["hello ultron", "follow up"])
        self.assertEqual(agent.calls[0][1], ())
        self.assertEqual(agent.calls[1][1], ("hello ultron", "heard: hello ultron"))
        self.assertEqual(speaker.spoken, ["heard: hello ultron", "heard: follow up"])
        self.assertTrue(any("Press Enter" in line for line in output))

    def test_empty_transcription_does_not_call_agent(self) -> None:
        agent = FakeAgent()
        recorder = FakeRecorder(b"silence")
        transcriber = FakeTranscriber(("",))
        speaker = FakeSpeaker()
        prompts = iter(("", "/exit"))
        output: list[str] = []

        active = run_voice_chat(
            agent, recorder, transcriber, speaker,
            input_fn=lambda _prompt: next(prompts), output_fn=output.append,
        )

        self.assertTrue(active)
        self.assertEqual(agent.calls, [])
        self.assertEqual(speaker.spoken, [])
        self.assertTrue(any("No speech recognized" in line for line in output))

    def test_device_failure_requests_terminal_fallback(self) -> None:
        class MissingMicrophone:
            def capture(self) -> bytes:
                raise VoiceUnavailableError("No microphone found.")

        output: list[str] = []
        active = run_voice_chat(
            FakeAgent(), MissingMicrophone(), FakeTranscriber(()), FakeSpeaker(),
            input_fn=lambda _prompt: "", output_fn=output.append,
        )

        self.assertFalse(active)
        self.assertTrue(any("No microphone found" in line for line in output))

    def test_recording_duration_is_finite_and_capped_at_thirty_seconds(self) -> None:
        with self.assertRaisesRegex(ValueError, "between 0 and 30"):
            SoundDeviceRecorder(seconds=31)
        with self.assertRaisesRegex(ValueError, "between 0 and 30"):
            SoundDeviceRecorder(seconds=float("inf"))

    def test_nonempty_activation_input_never_starts_recording(self) -> None:
        agent = FakeAgent()
        recorder = FakeRecorder()
        output: list[str] = []
        prompts = iter(("anything else", "/exit"))

        active = run_voice_chat(
            agent, recorder, FakeTranscriber(()), FakeSpeaker(),
            input_fn=lambda _prompt: next(prompts), output_fn=output.append,
        )

        self.assertTrue(active)
        self.assertEqual(recorder.activations, 0)
        self.assertTrue(any("Press Enter to activate" in line for line in output))


if __name__ == "__main__":
    unittest.main()

