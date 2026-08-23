"""Unit tests for push-to-talk in the live-mode controller.

These exercise the controller's PTT branch directly (white-box) without
touching real audio hardware: we never call ``start()`` (which would open a
sounddevice stream), instead we set the controller into LISTENING and drive
begin_talk / _feed_audio / end_talk by hand. The point is the state machine
and the "ship the captured PCM" path, not the mic.
"""
from __future__ import annotations

import unittest

from box_chat.audio_stream import BYTES_PER_FRAME
from box_chat.live_mode import _MIN_PTT_BYTES, LiveModeController, LiveState


def _frames(total_bytes: int) -> list[bytes]:
    """A list of BYTES_PER_FRAME-sized silence frames summing to >= total."""
    n = (total_bytes + BYTES_PER_FRAME - 1) // BYTES_PER_FRAME
    return [b"\x00" * BYTES_PER_FRAME for _ in range(max(1, n))]


class _Harness:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str | None]] = []
        self.cancelled = 0
        self.ctrl = LiveModeController(
            engine=object(),
            send_audio_to_engine=lambda wav, jpeg: self.sent.append((wav, jpeg)),
            cancel_generation=self._cancel,
        )
        # Put the controller into a started PTT session WITHOUT opening audio.
        self.ctrl._push_to_talk = True
        self.ctrl._state = LiveState.LISTENING

    def _cancel(self) -> None:
        self.cancelled += 1


class PushToTalkTests(unittest.TestCase):
    def test_full_turn_ships_audio(self) -> None:
        h = _Harness()
        h.ctrl.begin_talk()
        self.assertTrue(h.ctrl._ptt_active)
        for f in _frames(_MIN_PTT_BYTES + BYTES_PER_FRAME):
            h.ctrl._feed_audio(f)
        h.ctrl.end_talk()
        self.assertEqual(h.ctrl.state, LiveState.PROCESSING)
        self.assertEqual(len(h.sent), 1)
        wav, jpeg = h.sent[0]
        self.assertTrue(wav.endswith(".wav"))
        self.assertIsNone(jpeg)  # no camera session

    def test_too_short_tap_is_discarded(self) -> None:
        h = _Harness()
        h.ctrl.begin_talk()
        h.ctrl._feed_audio(b"\x00" * BYTES_PER_FRAME)  # ~30 ms, under the floor
        h.ctrl.end_talk()
        self.assertEqual(h.ctrl.state, LiveState.LISTENING)
        self.assertEqual(h.sent, [])

    def test_frames_ignored_when_not_holding(self) -> None:
        h = _Harness()
        # Not in a begin_talk window — frames must not accumulate.
        for f in _frames(_MIN_PTT_BYTES * 2):
            h.ctrl._feed_audio(f)
        self.assertEqual(len(h.ctrl._ptt_buffer), 0)

    def test_press_during_speaking_is_barge_in(self) -> None:
        h = _Harness()
        h.ctrl._state = LiveState.SPEAKING
        h.ctrl.begin_talk()
        self.assertEqual(h.cancelled, 1)
        self.assertEqual(h.ctrl.state, LiveState.LISTENING)
        self.assertTrue(h.ctrl._ptt_active)

    def test_ptt_methods_noop_without_ptt(self) -> None:
        # A VAD-mode controller should ignore begin/end_talk entirely.
        ctrl = LiveModeController(engine=object())
        ctrl._state = LiveState.LISTENING
        ctrl.begin_talk()
        self.assertFalse(ctrl._ptt_active)
        ctrl.end_talk()
        self.assertEqual(ctrl.state, LiveState.LISTENING)


class VadProcessingMuteTests(unittest.TestCase):
    """Regression: while the model thinks, VAD must be muted so trailing
    speech / room noise can't barge-in and cancel the reply (the bug that
    made live conversation unusable on slow CPU)."""

    def _ctrl(self):
        self.sent = []
        c = LiveModeController(
            engine=object(),
            send_audio_to_engine=lambda wav, jpeg: self.sent.append((wav, jpeg)),
        )
        c._state = LiveState.LISTENING
        c._vad.set_enabled(True)
        return c

    def test_speech_end_mutes_vad_and_processes(self) -> None:
        c = self._ctrl()
        c._on_speech_end(b"\x00" * 4000)
        self.assertEqual(c.state, LiveState.PROCESSING)
        self.assertFalse(c._vad._enabled)   # muted while thinking
        self.assertEqual(len(self.sent), 1)

    def test_no_barge_in_while_processing(self) -> None:
        c = self._ctrl()
        c._cancel_generation = lambda: self.fail("must not barge-in in PROCESSING")
        c._on_speech_end(b"\x00" * 4000)
        # A speech-start that slips through during PROCESSING must NOT cancel.
        c._on_speech_start()
        self.assertEqual(c.state, LiveState.PROCESSING)


if __name__ == "__main__":
    unittest.main()
