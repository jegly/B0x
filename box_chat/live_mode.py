"""Parlor-style live conversation controller.

Owns the state graph::

    IDLE ──start()──▶ LISTENING ─speech_end─▶ PROCESSING
                                                  │
                                          EvtComplete
                                                  ▼
    LISTENING ◀──tts_done──── SPEAKING ◀──── (assistant text)
                                  │
                          speech_start (after grace window)
                                  ▼
                              BARGED ─→ cancel engine + TTS ─→ LISTENING

This class is pure orchestration — it does NOT touch GTK directly. All
state-change announcements come back to the caller via the
``on_state_change`` callback so the UI can mirror them in whatever
widget tree it wants. Audio/video plumbing goes through callbacks too,
so a future Qt/Tauri port can swap in different I/O.
"""
from __future__ import annotations

import enum
import logging
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

from .audio_stream import AudioStream, BYTES_PER_FRAME, SAMPLE_RATE
from .config import CACHE_DIR
from .vad import VoiceActivityDetector

log = logging.getLogger(__name__)

# Minimum captured audio (bytes of 16-bit mono PCM) for a push-to-talk turn
# to count — guards against an accidental tap producing an empty send.
# 16000 Hz * 2 bytes * 0.2 s = 6400 bytes ≈ 200 ms.
_MIN_PTT_BYTES = SAMPLE_RATE * 2 // 5


class LiveState(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# Steering prompts used in each turn — adapted from Parlor's
# server.py. Keep them short; longer instructions eat prefill budget.
_STEER_AUDIO_AND_IMAGE = (
    "The user just spoke to you (audio) while showing their camera (image). "
    "Respond to what they said, referencing what you see if relevant."
)
_STEER_AUDIO_ONLY = (
    "The user just spoke to you. Respond to what they said."
)


class LiveModeController:
    """Coordinator. Construct once per :class:`MainWindow`."""

    # Grace period after kicking off TTS during which barge-in is
    # suppressed — otherwise the speaker's own audio leaks back into
    # the mic and trips VAD instantly.
    BARGE_IN_GRACE_S = 0.8

    def __init__(
        self,
        *,
        engine,                          # EngineManager
        on_state_change: Callable[[LiveState], None] | None = None,
        on_user_audio_path: Callable[[str], None] | None = None,
        on_assistant_text: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
        send_audio_to_engine: Callable[[str, str | None], None] | None = None,
        cancel_generation: Callable[[], None] | None = None,
    ) -> None:
        self._engine = engine
        self._on_state_change = on_state_change
        self._on_user_audio_path = on_user_audio_path
        self._on_assistant_text = on_assistant_text
        self._on_error = on_error
        self._send_audio_to_engine = send_audio_to_engine
        self._cancel_generation = cancel_generation

        self._state: LiveState = LiveState.IDLE
        self._lock = threading.Lock()
        self._audio = AudioStream()
        self._vad = VoiceActivityDetector(
            on_speech_start=self._on_speech_start,
            on_speech_end=self._on_speech_end,
        )
        self._tts_started_at: float = 0.0
        self._camera_session: Any = None
        self._latest_jpeg_path: str = ""

        # Push-to-talk: when on, VAD is bypassed entirely and the user
        # drives capture via begin_talk()/end_talk() (a held button in the
        # live panel). Set per-session in start().
        self._push_to_talk: bool = False
        self._ptt_active: bool = False
        self._ptt_buffer: bytearray = bytearray()

    # ── lifecycle ────────────────────────────────────────────────────
    @property
    def state(self) -> LiveState:
        return self._state

    def start(
        self,
        camera_session: Any | None = None,
        *,
        push_to_talk: bool = False,
    ) -> None:
        """Begin a live session. ``camera_session`` is an already-open
        :class:`webcam.CameraSession`; the controller doesn't own it,
        the caller does (because the UI usually wants to bind a
        preview to it as well).

        ``push_to_talk`` selects manual capture (held Talk button) instead
        of VAD auto-listen — useful in noisy rooms where VAD keeps
        false-triggering."""
        with self._lock:
            if self._state is not LiveState.IDLE:
                return
            self._camera_session = camera_session
            self._push_to_talk = push_to_talk
            self._ptt_active = False
            self._ptt_buffer = bytearray()
            if push_to_talk:
                # No VAD in PTT mode — capture is driven by begin/end_talk.
                self._vad.set_enabled(False)
            else:
                # Re-arm VAD on every start. stop() turns it off so the
                # mic-stream tear-down doesn't leak speech events into a
                # closing-state controller; we need it back on for the
                # next session.
                self._vad.set_enabled(True)
            self._audio.start(self._feed_audio)
            self._transition(LiveState.LISTENING)

    def stop(self) -> None:
        with self._lock:
            if self._state is LiveState.IDLE:
                return
            self._audio.stop()
            self._vad.set_enabled(False)
            self._ptt_active = False
            self._ptt_buffer = bytearray()
            self._camera_session = None
            self._transition(LiveState.IDLE)

    # ── push-to-talk (only meaningful when started with push_to_talk) ─────
    def begin_talk(self) -> None:
        """User pressed/held the Talk button. Starts capturing a turn.

        Pressing while the model is speaking or processing acts as a
        barge-in — cancel the current turn first, then start listening."""
        if not self._push_to_talk:
            return
        if self._state in (LiveState.SPEAKING, LiveState.PROCESSING):
            log.info("live mode: PTT barge-in during %s", self._state.value)
            if self._cancel_generation is not None:
                try:
                    self._cancel_generation()
                except Exception:
                    log.exception("cancel_generation raised")
            with self._lock:
                self._transition(LiveState.LISTENING)
        with self._lock:
            if self._state is not LiveState.LISTENING:
                return
            self._ptt_buffer = bytearray()
            self._ptt_active = True

    def end_talk(self) -> None:
        """User released the Talk button — ship whatever was captured."""
        if not self._push_to_talk:
            return
        with self._lock:
            if not self._ptt_active:
                return
            self._ptt_active = False
            pcm = bytes(self._ptt_buffer)
            self._ptt_buffer = bytearray()
            if self._state is not LiveState.LISTENING:
                return
            if len(pcm) < _MIN_PTT_BYTES:
                # Too short — treat as an accidental tap, stay listening.
                return
            self._transition(LiveState.PROCESSING)
        self._ship_utterance(pcm)

    def notify_tts_started(self) -> None:
        """Caller hands TTS playback start back to us so we can mute
        VAD during the grace window and re-arm afterward."""
        self._tts_started_at = time.monotonic()
        if self._push_to_talk:
            # No VAD in PTT mode — barge-in is the Talk button, not voice.
            return
        self._vad.set_enabled(False)
        # Re-arm VAD after the grace window so the user can barge in.
        def _arm() -> None:
            time.sleep(self.BARGE_IN_GRACE_S)
            if self._state is LiveState.SPEAKING:
                self._vad.set_enabled(True)
        threading.Thread(target=_arm, daemon=True).start()

    def notify_tts_finished(self) -> None:
        """TTS finished playing naturally (no barge-in)."""
        with self._lock:
            if self._state is LiveState.SPEAKING:
                if not self._push_to_talk:
                    self._vad.set_enabled(True)
                self._transition(LiveState.LISTENING)

    def notify_engine_complete(self, full_text: str) -> None:
        """Engine emitted EvtComplete. Hand the text off to the UI for
        bubble rendering and (optionally) TTS."""
        with self._lock:
            if self._state is not LiveState.PROCESSING:
                return
            self._transition(LiveState.SPEAKING)
        if self._on_assistant_text is not None:
            try:
                self._on_assistant_text(full_text)
            except Exception:
                log.exception("on_assistant_text raised")

    def notify_engine_error(self, message: str) -> None:
        """Generation failed mid-turn. Drop back to LISTENING."""
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:
                log.exception("on_error callback raised")
        with self._lock:
            if self._state in (LiveState.PROCESSING, LiveState.SPEAKING):
                if not self._push_to_talk:
                    self._vad.set_enabled(True)
                self._transition(LiveState.LISTENING)

    # ── audio frames (run on the audio thread) ───────────────────────
    def _feed_audio(self, frame: bytes) -> None:
        """Single sink for mic frames. Routes to VAD in auto mode, or
        accumulates into the PTT buffer while the Talk button is held."""
        if self._push_to_talk:
            if self._ptt_active and len(frame) == BYTES_PER_FRAME:
                self._ptt_buffer.extend(frame)
            return
        self._vad.feed(frame)

    # ── VAD events (run on the audio thread) ─────────────────────────
    def _on_speech_start(self) -> None:
        # Barge-in only while the model is *speaking* (TTS playing). During
        # PROCESSING the mic is muted (see _on_speech_end) so the model can
        # finish thinking without the user's trailing speech / room noise
        # cancelling the turn — that storm of cancels is what broke live mode
        # on slow CPUs. To bail out of a genuinely stuck turn, use End.
        if self._state is LiveState.SPEAKING:
            log.info("live mode: barge-in during %s", self._state.value)
            if self._cancel_generation is not None:
                try:
                    self._cancel_generation()
                except Exception:
                    log.exception("cancel_generation raised")
            with self._lock:
                self._transition(LiveState.LISTENING)

    def _on_speech_end(self, pcm: bytes) -> None:
        """User stopped talking (VAD) — ship the audio + a webcam snapshot."""
        with self._lock:
            if self._state is not LiveState.LISTENING:
                return
            # Stop listening while the model thinks. On CPU a multimodal turn
            # (audio + webcam frame) can take many seconds; if VAD stayed live
            # it would pick up the user's trailing words or room noise and fire
            # a barge-in that cancels the reply before it ever arrives — which
            # made live mode unusable on slow hardware. VAD is re-armed for
            # barge-in only once TTS playback starts (after the grace window in
            # notify_tts_started), or back to LISTENING on tts-finished / error.
            self._vad.set_enabled(False)
            self._transition(LiveState.PROCESSING)
        self._ship_utterance(pcm)

    def _ship_utterance(self, pcm: bytes) -> None:
        """Persist a captured turn (VAD or PTT) and hand it to the engine.
        Caller must already have transitioned to PROCESSING."""
        # Persist the PCM as a WAV so we can hand it to the engine's
        # existing file-based audio attachment path.
        wav_path = _write_wav(pcm)
        if self._on_user_audio_path is not None:
            try:
                self._on_user_audio_path(wav_path)
            except Exception:
                log.exception("on_user_audio_path raised")

        # Grab one webcam frame if we have a session open.
        jpeg_path = ""
        if self._camera_session is not None:
            try:
                jpeg = self._camera_session.capture_jpeg()
                jpeg_path = str(
                    CACHE_DIR / "captures" / f"live_{int(time.time()*1000)}.jpg"
                )
                Path(jpeg_path).parent.mkdir(parents=True, exist_ok=True)
                Path(jpeg_path).write_bytes(jpeg)
                self._latest_jpeg_path = jpeg_path
            except Exception:
                log.exception("capture_jpeg failed during live turn")
                jpeg_path = ""

        if self._send_audio_to_engine is not None:
            try:
                self._send_audio_to_engine(wav_path, jpeg_path or None)
            except Exception as e:  # noqa: BLE001
                log.exception("send_audio_to_engine raised")
                self.notify_engine_error(str(e))

    # ── helpers ──────────────────────────────────────────────────────
    def _transition(self, new: LiveState) -> None:
        if new is self._state:
            return
        log.debug("live mode: %s → %s", self._state.value, new.value)
        self._state = new
        if self._on_state_change is not None:
            try:
                self._on_state_change(new)
            except Exception:
                log.exception("on_state_change raised")


def _write_wav(pcm_int16_le: bytes) -> str:
    """Wrap raw 16-bit PCM in a WAV container and return its path."""
    path = CACHE_DIR / f"live_turn_{int(time.time() * 1000)}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_int16_le)
    return str(path)
