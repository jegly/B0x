"""Voice Activity Detection wrapper for live mode.

Wraps WebRTC VAD (via ``webrtcvad-wheels``) and exposes a tiny
callback interface:

- ``feed(pcm_30ms_bytes)`` — call on every audio frame; the wrapper
  decides if speech is on/off based on a sliding window and fires
  ``on_speech_start()`` / ``on_speech_end(audio_pcm: bytes)``.
- The PCM buffer for the *current* utterance is accumulated internally
  and handed back via ``on_speech_end`` so the live-mode controller
  doesn't have to do its own bookkeeping.

If webrtcvad is unavailable we fall back to a dead-simple RMS-energy
threshold so live mode still functions (less accurately).
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Callable

from .audio_stream import BYTES_PER_FRAME, FRAME_MS, SAMPLE_RATE

log = logging.getLogger(__name__)

try:
    import webrtcvad as _webrtcvad  # type: ignore[import-not-found]
    _VAD_AVAILABLE = True
except Exception as _e:  # noqa: BLE001
    _webrtcvad = None  # type: ignore[assignment]
    _VAD_AVAILABLE = False
    log.warning("webrtcvad not available — falling back to RMS VAD: %s", _e)


class VoiceActivityDetector:
    """Emits ``on_speech_start`` / ``on_speech_end`` events.

    Tunables:

    - ``aggressiveness`` (0-3): higher = more eager to declare
      non-speech as silence. WebRTC's default is 3, Parlor uses 2.
    - ``speech_pad_ms``: how much of the pre-trigger audio to prepend
      to the captured utterance (so we don't clip the first phoneme).
    - ``silence_end_ms``: trailing silence required to declare speech
      ended. 800 ms is comfy for natural sentences without long pauses.
    - ``min_speech_ms``: discard utterances shorter than this (likely a
      cough, door click, etc.).
    """

    def __init__(
        self,
        on_speech_start: Callable[[], None] | None = None,
        on_speech_end: Callable[[bytes], None] | None = None,
        *,
        aggressiveness: int = 3,
        speech_pad_ms: int = 300,
        silence_end_ms: int = 800,
        min_speech_ms: int = 300,
        rms_threshold: int = 700,
    ) -> None:
        self._on_start = on_speech_start
        self._on_end = on_speech_end
        self._aggressiveness = max(0, min(3, int(aggressiveness)))
        self._silence_end_frames = max(1, silence_end_ms // FRAME_MS)
        self._min_speech_frames = max(1, min_speech_ms // FRAME_MS)
        self._rms_threshold = rms_threshold

        # Pre-roll buffer keeps the last `speech_pad_ms` of audio so we
        # can prepend it to the utterance once speech is detected.
        pad_frames = max(1, speech_pad_ms // FRAME_MS)
        self._preroll: deque[bytes] = deque(maxlen=pad_frames)

        self._vad = (
            _webrtcvad.Vad(self._aggressiveness)
            if _VAD_AVAILABLE else None
        )
        self._in_speech: bool = False
        self._silence_count: int = 0
        self._speech_frames: int = 0
        self._utterance: bytearray = bytearray()
        self._enabled: bool = True

    # ── frame processing ─────────────────────────────────────────────
    def feed(self, frame: bytes) -> None:
        if not self._enabled:
            return
        if len(frame) != BYTES_PER_FRAME:
            # Wrong cadence — silently skip. Better than throwing in a
            # hot audio callback.
            return

        is_speech = self._classify(frame)

        if self._in_speech:
            self._utterance.extend(frame)
            self._speech_frames += 1
            if is_speech:
                self._silence_count = 0
            else:
                self._silence_count += 1
                if self._silence_count >= self._silence_end_frames:
                    self._finish_utterance()
        else:
            self._preroll.append(frame)
            if is_speech:
                self._start_utterance()

    def _classify(self, frame: bytes) -> bool:
        if self._vad is not None:
            try:
                return bool(self._vad.is_speech(frame, SAMPLE_RATE))
            except Exception:
                log.exception("webrtcvad call failed; reverting to RMS")
        # RMS fallback — int16 little-endian.
        return _rms(frame) > self._rms_threshold

    def _start_utterance(self) -> None:
        self._in_speech = True
        self._silence_count = 0
        self._speech_frames = 0
        # Seed with the pre-roll so we don't clip the leading phoneme.
        self._utterance = bytearray(b"".join(self._preroll))
        if self._on_start is not None:
            try:
                self._on_start()
            except Exception:
                log.exception("on_speech_start raised")

    def _finish_utterance(self) -> None:
        utterance = bytes(self._utterance)
        long_enough = self._speech_frames >= self._min_speech_frames
        self._in_speech = False
        self._silence_count = 0
        self._speech_frames = 0
        self._utterance = bytearray()
        self._preroll.clear()
        if not long_enough:
            return
        if self._on_end is not None:
            try:
                self._on_end(utterance)
            except Exception:
                log.exception("on_speech_end raised")

    # ── controller knobs ─────────────────────────────────────────────
    def set_enabled(self, enabled: bool) -> None:
        """Temporarily mute the detector — useful during TTS playback
        so the synthesized speech doesn't trigger a barge-in unless we
        explicitly arm the detector again. (Live mode arms it after a
        short grace window once TTS starts.)"""
        self._enabled = enabled
        if not enabled:
            self._in_speech = False
            self._utterance = bytearray()
            self._silence_count = 0
            self._speech_frames = 0


def _rms(frame: bytes) -> float:
    """RMS energy of a 16-bit PCM frame. Cheap fallback when no VAD."""
    if not frame:
        return 0.0
    # int16 little-endian.
    n = len(frame) // 2
    total = 0
    for i in range(0, len(frame), 2):
        lo = frame[i]
        hi = frame[i + 1]
        # signed 16-bit
        v = lo | (hi << 8)
        if v & 0x8000:
            v -= 0x10000
        total += v * v
    return math.sqrt(total / n) if n else 0.0
