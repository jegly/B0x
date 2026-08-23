"""Microphone recorder — captures audio to a WAV file for the engine."""
from __future__ import annotations

import threading
import wave
from pathlib import Path
from typing import Callable

from .config import CACHE_DIR

_SAMPLE_RATE = 16000  # 16 kHz — good for ASR / Gemma 4 audio understanding
_CHANNELS    = 1
_DTYPE       = "int16"
_WAV_PATH    = CACHE_DIR / "voice_input.wav"


class VoiceRecorder:
    """
    Usage:
        recorder = VoiceRecorder()
        recorder.start()          # begin capturing
        path = recorder.stop()    # finish, returns WAV path or None
    """

    def __init__(self) -> None:
        self._recording = False
        self._frames: list = []
        self._stream = None
        self._lock = threading.Lock()
        self._last_duration: float = 0.0

    @property
    def duration_s(self) -> float:
        """Duration of the last completed recording in seconds."""
        return self._last_duration

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        import sounddevice as sd
        import numpy as np

        with self._lock:
            if self._recording:
                return
            self._frames = []
            self._recording = True

            def _cb(indata, frames, time_info, status):
                if self._recording:
                    self._frames.append(indata.copy())

            self._stream = sd.InputStream(
                samplerate=_SAMPLE_RATE,
                channels=_CHANNELS,
                dtype=_DTYPE,
                callback=_cb,
            )
            self._stream.start()

    def stop(self) -> str | None:
        """Stop recording. Returns path to WAV or None if too short (<0.3 s)."""
        import numpy as np

        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None
            frames = list(self._frames)
            self._frames = []

        if not frames:
            return None

        audio = np.concatenate(frames, axis=0)

        # Reject clips shorter than 0.3 s — almost certainly accidental clicks.
        if len(audio) < _SAMPLE_RATE * 0.3:
            return None

        self._last_duration = len(audio) / _SAMPLE_RATE

        _WAV_PATH.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(_WAV_PATH), "wb") as wf:
            wf.setnchannels(_CHANNELS)
            wf.setsampwidth(2)          # int16 = 2 bytes
            wf.setframerate(_SAMPLE_RATE)
            wf.writeframes(audio.tobytes())

        return str(_WAV_PATH)
