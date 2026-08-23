"""Continuous microphone frame stream for live mode.

Separate from :class:`VoiceRecorder` (which is a one-shot record/save
flow): this class emits short fixed-size PCM frames to a callback as
they arrive, so a VAD or live-conversation controller can react to
speech in real time.

Frame format: 16-bit mono 16 kHz PCM (matches `webrtcvad`'s constraints
and Gemma 4 audio backend). Frame duration is 30 ms by default → 480
samples → 960 bytes per callback invocation.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Callable

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480
BYTES_PER_FRAME = FRAME_SAMPLES * 2             # int16 → 2 bytes


def is_available() -> bool:
    """True if audio input can actually start on this machine.

    ``sounddevice`` raises at import time if the PortAudio system library
    (``libportaudio2``) is missing, so a successful import is a good proxy
    for "the mic / voice messages / live mode will work." Used by the UI to
    show a clear message instead of a dead button + console traceback.
    """
    try:
        import sounddevice  # noqa: F401
    except Exception as e:  # noqa: BLE001
        log.warning("audio input unavailable: %s", e)
        return False
    return True


class AudioStream:
    """Push-style audio capture.

    ``start(on_frame)`` begins capture; ``on_frame`` is invoked with a
    ``bytes`` payload of exactly :data:`BYTES_PER_FRAME` bytes each time
    a frame is ready. Callbacks run on sounddevice's stream thread —
    keep them cheap. The controller can also call
    :meth:`recent_pcm(seconds)` to grab the last N seconds as a single
    PCM byte-string when it decides to ship a turn to the model.
    """

    def __init__(self, ring_seconds: float = 30.0) -> None:
        self._stream = None
        self._on_frame: Callable[[bytes], None] | None = None
        self._lock = threading.Lock()
        self._running = False
        # Rolling buffer of recent frames so `recent_pcm()` can pull
        # the last N seconds at any time. Old frames evict from the
        # left when the deque hits its maxlen.
        ring_frames = max(1, int(ring_seconds * 1000 / FRAME_MS))
        self._ring: deque[bytes] = deque(maxlen=ring_frames)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self, on_frame: Callable[[bytes], None]) -> None:
        import sounddevice as sd
        with self._lock:
            if self._running:
                return
            self._on_frame = on_frame
            self._ring.clear()

            def _cb(indata, _frames, _time_info, _status) -> None:
                # indata is a numpy ndarray int16 of shape (FRAME_SAMPLES, 1)
                buf = indata.tobytes()
                self._ring.append(buf)
                cb = self._on_frame
                if cb is not None:
                    try:
                        cb(buf)
                    except Exception:
                        log.exception("on_frame callback raised")

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                blocksize=FRAME_SAMPLES,
                callback=_cb,
            )
            self._stream.start()
            self._running = True

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._on_frame = None
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    log.exception("Stream close failed")
                self._stream = None

    def recent_pcm(self, seconds: float) -> bytes:
        """Return up to ``seconds`` of the most recent audio as raw
        16-bit-PCM bytes. The result is suitable for wrapping in a WAV
        header and shipping to the engine."""
        n_frames = max(1, int(seconds * 1000 / FRAME_MS))
        frames = list(self._ring)[-n_frames:]
        return b"".join(frames)

    def clear_ring(self) -> None:
        """Drop the rolling buffer — call after consuming a turn so the
        next turn starts fresh."""
        self._ring.clear()
