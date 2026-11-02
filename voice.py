"""Optional, offline push-to-talk adapters for ULTRON."""
from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import Protocol


class VoiceUnavailableError(RuntimeError):
    """A local voice dependency, model, or device is not available."""


class AudioRecorder(Protocol):
    def capture(self) -> bytes:
        """Capture one bounded utterance as mono signed 16-bit PCM."""


class Transcriber(Protocol):
    def transcribe(self, pcm: bytes) -> str:
        """Convert one PCM utterance to text locally."""


class Speaker(Protocol):
    def speak(self, text: str) -> None:
        """Speak text locally and return when playback completes."""


class SoundDeviceRecorder:
    """Record a single fixed-duration utterance after explicit activation."""

    def __init__(self, *, seconds: float = 8.0, sample_rate: int = 16_000) -> None:
        if not isfinite(seconds) or seconds <= 0 or seconds > 30:
            raise ValueError("seconds must be finite and between 0 and 30")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        self.seconds = seconds
        self.sample_rate = sample_rate
        try:
            import sounddevice
        except ImportError as exc:
            raise VoiceUnavailableError(
                "Voice input needs the optional packages. Install requirements-voice.txt."
            ) from exc
        self._sounddevice = sounddevice

    def capture(self) -> bytes:
        try:
            recording = self._sounddevice.rec(
                int(self.seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
            )
            self._sounddevice.wait()
            return recording.tobytes()
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Microphone capture failed ({type(exc).__name__})."
            ) from exc


class WhisperCppTranscriber:
    """Offline speech recognition using whisper.cpp and a local model."""

    def __init__(self, model_path: str | Path, *, sample_rate: int = 16_000) -> None:
        try:
            import numpy as np
            from pywhispercpp.model import Model
        except ImportError as exc:
            raise VoiceUnavailableError(
                "Speech recognition needs pywhispercpp. Install requirements-voice.txt."
            ) from exc
        self._numpy = np
        self.sample_rate = sample_rate
        try:
            # pywhispercpp accepts a local model path or a model name such as tiny.en.
            self._model = Model(str(model_path), print_realtime=False, print_progress=False)
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Could not load the local Whisper model ({type(exc).__name__})."
            ) from exc

    def transcribe(self, pcm: bytes) -> str:
        try:
            audio = self._numpy.frombuffer(pcm, dtype=self._numpy.int16)
            if audio.size == 0:
                return ""
            normalized = audio.astype(self._numpy.float32) / 32768.0
            segments = self._model.transcribe(normalized, language="en")
            return " ".join(segment.text.strip() for segment in segments).strip()
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Speech recognition failed ({type(exc).__name__})."
            ) from exc


class Pyttsx3Speaker:
    """Windows local speech output using SAPI through pyttsx3."""

    def __init__(self) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise VoiceUnavailableError(
                "Speech output needs pyttsx3. Install requirements-voice.txt."
            ) from exc
        try:
            self._engine = pyttsx3.init("sapi5")
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Windows speech output could not start ({type(exc).__name__})."
            ) from exc

    def speak(self, text: str) -> None:
        if not text:
            return
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception as exc:
            raise VoiceUnavailableError(
                f"Speech playback failed ({type(exc).__name__})."
            ) from exc


def build_voice_adapters(
    model_path: str | Path,
    *,
    seconds: float = 8.0,
    sample_rate: int = 16_000,
) -> tuple[AudioRecorder, Transcriber, Speaker]:
    """Load optional voice dependencies only when voice mode is selected."""
    recorder = SoundDeviceRecorder(seconds=seconds, sample_rate=sample_rate)
    transcriber = WhisperCppTranscriber(model_path, sample_rate=sample_rate)
    speaker = Pyttsx3Speaker()
    return recorder, transcriber, speaker


