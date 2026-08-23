"""Model-reload debounce in PreferencesDialog.

Dragging a SpinRow (context window, temperature, top-k/p) used to queue a full
engine reload per tick. These tests drive the two reload methods directly with
a fake GLib so the cancel-and-reschedule logic is verified without a GTK main
loop or constructing the dialog (which needs a display).
"""
from __future__ import annotations

import types
import unittest

import box_chat.preferences as prefs


class _FakeGLib:
    def __init__(self) -> None:
        self.next_id = 0
        self.removed: list[int] = []
        self.scheduled: list[tuple[int, int, object]] = []

    def timeout_add(self, ms, cb):
        self.next_id += 1
        self.scheduled.append((self.next_id, ms, cb))
        return self.next_id

    def source_remove(self, sid):
        self.removed.append(sid)


class DebounceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._real_glib = prefs.GLib
        self.glib = _FakeGLib()
        prefs.GLib = self.glib
        self.reloads = 0

        def _main_win():
            return types.SimpleNamespace(on_model_changed=self._count_reload)

        self.stub = types.SimpleNamespace(
            _reload_source_id=0,
            _RELOAD_DEBOUNCE_MS=700,
            _main_win=_main_win,
            _settings=types.SimpleNamespace(model_path="/x.litertlm"),
        )
        # Bind the real one-shot reload method so the scheduled callback is the
        # same zero-arg bound method GLib would call.
        self.stub._do_model_reload = types.MethodType(
            prefs.PreferencesDialog._do_model_reload, self.stub
        )

    def tearDown(self) -> None:
        prefs.GLib = self._real_glib

    def _count_reload(self, _path) -> None:
        self.reloads += 1

    def _trigger(self) -> None:
        prefs.PreferencesDialog._trigger_model_reload(self.stub)

    def _fire_latest(self) -> None:
        # Simulate the GLib main loop firing the most-recently scheduled timeout.
        # GLib calls the callback with no args (it's a bound method).
        _id, _ms, cb = self.glib.scheduled[-1]
        cb()

    def test_single_change_schedules_one_reload(self) -> None:
        self._trigger()
        self.assertEqual(len(self.glib.scheduled), 1)
        self.assertEqual(self.glib.removed, [])
        self.assertEqual(self.reloads, 0)  # not fired yet — deferred
        self._fire_latest()
        self.assertEqual(self.reloads, 1)
        self.assertEqual(self.stub._reload_source_id, 0)

    def test_rapid_changes_collapse_to_one_reload(self) -> None:
        # 28 ticks like dragging the context slider 32k→4k.
        for _ in range(28):
            self._trigger()
        # Each new trigger cancels the previous pending source.
        self.assertEqual(len(self.glib.removed), 27)
        self.assertEqual(len(self.glib.scheduled), 28)
        # Only the last timeout is still pending; fire it.
        self._fire_latest()
        self.assertEqual(self.reloads, 1)  # ONE reload, not 28

    def test_uses_configured_delay(self) -> None:
        self._trigger()
        _id, ms, _cb = self.glib.scheduled[-1]
        self.assertEqual(ms, 700)


if __name__ == "__main__":
    unittest.main()
