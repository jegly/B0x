"""PermissionGate threading + trust semantics.

These tests stand in for the GTK dialog with an ``ask_user`` callback that
answers from a worker thread, so we exercise the full cross-thread wait /
notify cycle that the SDK handler will use in production.
"""
from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from box_chat.config import Settings
from box_chat.permissions import (
    BoxToolEventHandler,
    Decision,
    PermissionGate,
)


def _scripted_cb(answers: list[Decision], delay_s: float = 0.0):
    """Callback that returns the next scripted answer in a background thread.

    Captures every (tool_id, fn_name, args, risky) it sees in ``seen``.
    """
    seen: list[tuple[str, str, dict, bool]] = []
    lock = threading.Lock()

    def cb(tool_id, fn_name, args, risky, on_answer):
        with lock:
            seen.append((tool_id, fn_name, args, risky))
            try:
                d = answers.pop(0)
            except IndexError:
                d = Decision.DENY

        def respond():
            on_answer(d)

        # Always answer from a different thread so the gate's
        # threading.Event.wait() path actually runs.
        threading.Timer(delay_s, respond).start()

    return cb, seen


def _settings_in_tmp() -> tuple[Settings, tempfile.TemporaryDirectory]:
    """Settings instance backed by a temp settings.json so save() is real."""
    tmp = tempfile.TemporaryDirectory()
    # Point SETTINGS_PATH at the tmp dir for this test by patching at use.
    s = Settings()
    s.tool_always_allow = []  # ensure fresh
    return s, tmp


class GateBasicsTests(unittest.TestCase):
    def test_allow_once_returns_true_and_does_not_persist(self) -> None:
        s, _tmp = _settings_in_tmp()
        cb, seen = _scripted_cb([Decision.ALLOW_ONCE])
        gate = PermissionGate(s, cb)
        ok = gate.decide("fs_read", {"path": "x"}, risky=False, tool_id="filesystem")
        self.assertTrue(ok)
        self.assertEqual(s.tool_always_allow, [])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0][:2], ("filesystem", "fs_read"))

    def test_deny_returns_false(self) -> None:
        s, _ = _settings_in_tmp()
        cb, _seen = _scripted_cb([Decision.DENY])
        gate = PermissionGate(s, cb)
        ok = gate.decide("fs_read", {"path": "x"}, risky=False, tool_id="filesystem")
        self.assertFalse(ok)

    def test_chat_trust_skips_prompt_on_second_call(self) -> None:
        s, _ = _settings_in_tmp()
        cb, seen = _scripted_cb([Decision.ALLOW_CHAT])
        gate = PermissionGate(s, cb)
        gate.set_active_conversation(42)
        self.assertTrue(gate.decide("fs_read", {"a": 1}, risky=False, tool_id="filesystem"))
        self.assertTrue(gate.decide("fs_read", {"a": 2}, risky=False, tool_id="filesystem"))
        self.assertEqual(len(seen), 1, "second call should not prompt")

    def test_chat_trust_scoped_to_conversation(self) -> None:
        s, _ = _settings_in_tmp()
        cb, seen = _scripted_cb([Decision.ALLOW_CHAT, Decision.ALLOW_CHAT])
        gate = PermissionGate(s, cb)
        gate.set_active_conversation(1)
        gate.decide("fs_read", {}, risky=False, tool_id="filesystem")
        gate.set_active_conversation(2)
        gate.decide("fs_read", {}, risky=False, tool_id="filesystem")
        # Both conversations had to prompt independently
        self.assertEqual(len(seen), 2)

    def test_clear_chat_trust_re_prompts(self) -> None:
        s, _ = _settings_in_tmp()
        cb, seen = _scripted_cb([Decision.ALLOW_CHAT, Decision.ALLOW_CHAT])
        gate = PermissionGate(s, cb)
        gate.set_active_conversation(7)
        gate.decide("fs_read", {}, risky=False, tool_id="filesystem")
        gate.clear_chat_trust(7)
        gate.decide("fs_read", {}, risky=False, tool_id="filesystem")
        self.assertEqual(len(seen), 2)


class GateTrustModeTests(unittest.TestCase):
    def test_trust_mode_short_circuits_for_non_risky(self) -> None:
        s, _ = _settings_in_tmp()
        s.tool_web_search_permission = "trust"
        cb, seen = _scripted_cb([])  # would IndexError if reached
        gate = PermissionGate(s, cb)
        ok = gate.decide("web_search", {"query": "x"}, risky=False, tool_id="web_search")
        self.assertTrue(ok)
        self.assertEqual(seen, [], "trust mode should not prompt")

    def test_trust_mode_does_not_apply_to_risky(self) -> None:
        s, _ = _settings_in_tmp()
        s.tool_fs_permission = "trust"
        cb, seen = _scripted_cb([Decision.DENY])
        gate = PermissionGate(s, cb)
        ok = gate.decide(
            "fs_write", {"path": "x", "content": "y"},
            risky=True, tool_id="filesystem",
        )
        self.assertFalse(ok)
        self.assertEqual(len(seen), 1, "risky tools must always prompt")


class GatePersistentTrustTests(unittest.TestCase):
    def test_always_allow_persists_only_for_non_risky(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            sp = Path(tdir) / "settings.json"
            with patch("box_chat.config.SETTINGS_PATH", sp):
                s = Settings()
                s.tool_always_allow = []
                cb, _ = _scripted_cb([Decision.ALLOW_TRUST])
                gate = PermissionGate(s, cb)
                gate.decide(
                    "web_search", {"query": "x"},
                    risky=False, tool_id="web_search",
                )
                self.assertIn("web_search", s.tool_always_allow)
                # File was saved
                self.assertTrue(sp.exists())

    def test_always_allow_ignored_for_risky_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            sp = Path(tdir) / "settings.json"
            with patch("box_chat.config.SETTINGS_PATH", sp):
                s = Settings()
                s.tool_always_allow = []
                cb, _ = _scripted_cb([Decision.ALLOW_TRUST])
                gate = PermissionGate(s, cb)
                ok = gate.decide(
                    "fs_write", {"path": "x", "content": "y"},
                    risky=True, tool_id="filesystem",
                )
                # The call is allowed (single shot) but trust is NOT persisted.
                self.assertTrue(ok)
                self.assertEqual(s.tool_always_allow, [])

    def test_persistent_entry_short_circuits_prompt(self) -> None:
        s, _ = _settings_in_tmp()
        s.tool_always_allow = ["fs_read"]
        cb, seen = _scripted_cb([])
        gate = PermissionGate(s, cb)
        ok = gate.decide("fs_read", {"path": "x"}, risky=False, tool_id="filesystem")
        self.assertTrue(ok)
        self.assertEqual(seen, [])


class BoxToolEventHandlerTests(unittest.TestCase):
    def test_unknown_tool_denied(self) -> None:
        s, _ = _settings_in_tmp()
        cb, _ = _scripted_cb([])
        gate = PermissionGate(s, cb)
        handler = BoxToolEventHandler(gate, call_map={})
        self.assertFalse(handler.approve_tool_call({"name": "wat", "args": {}}))

    def test_routes_to_gate_with_correct_metadata(self) -> None:
        s, _ = _settings_in_tmp()
        cb, seen = _scripted_cb([Decision.ALLOW_ONCE])
        gate = PermissionGate(s, cb)
        call_map = {
            "fs_write": {
                "tool_id": "filesystem",
                "risky": True,
                "default_permission": "ask",
            }
        }
        handler = BoxToolEventHandler(gate, call_map=call_map)
        ok = handler.approve_tool_call(
            {"name": "fs_write", "args": {"path": "x", "content": "y"}}
        )
        self.assertTrue(ok)
        self.assertEqual(seen[0][0], "filesystem")
        self.assertEqual(seen[0][1], "fs_write")
        self.assertTrue(seen[0][3], "risky flag should propagate")

    def test_process_tool_response_passthrough(self) -> None:
        s, _ = _settings_in_tmp()
        gate = PermissionGate(s, _scripted_cb([])[0])
        handler = BoxToolEventHandler(gate, call_map={})
        payload = {"name": "fs_read", "result": "hello"}
        self.assertEqual(handler.process_tool_response(payload), payload)


class AgentIterationCapTests(unittest.TestCase):
    _CALL_MAP = {
        "web_search": {
            "tool_id": "web_search",
            "risky": False,
            "default_permission": "trust",
        }
    }

    def _approve(self, handler) -> bool:
        return handler.approve_tool_call(
            {"name": "web_search", "args": {"query": "x"}}
        )

    def test_no_cap_allows_unlimited_calls(self) -> None:
        s, _ = _settings_in_tmp()
        gate = PermissionGate(s, _scripted_cb([Decision.ALLOW_ONCE] * 10)[0])
        handler = BoxToolEventHandler(gate, self._CALL_MAP)  # max_iterations=None
        for _ in range(10):
            self.assertTrue(self._approve(handler))

    def test_cap_blocks_after_limit(self) -> None:
        s, _ = _settings_in_tmp()
        gate = PermissionGate(s, _scripted_cb([Decision.ALLOW_ONCE] * 5)[0])
        handler = BoxToolEventHandler(gate, self._CALL_MAP, max_iterations=3)
        self.assertTrue(self._approve(handler))   # 1
        self.assertTrue(self._approve(handler))   # 2
        self.assertTrue(self._approve(handler))   # 3
        self.assertFalse(self._approve(handler))  # 4 — capped
        self.assertFalse(self._approve(handler))  # stays capped

    def test_reset_iterations_reopens_budget(self) -> None:
        s, _ = _settings_in_tmp()
        gate = PermissionGate(s, _scripted_cb([Decision.ALLOW_ONCE] * 5)[0])
        handler = BoxToolEventHandler(gate, self._CALL_MAP, max_iterations=2)
        self.assertTrue(self._approve(handler))
        self.assertTrue(self._approve(handler))
        self.assertFalse(self._approve(handler))
        handler.reset_iterations()
        self.assertTrue(self._approve(handler))

    def test_progress_callback_fires(self) -> None:
        s, _ = _settings_in_tmp()
        gate = PermissionGate(s, _scripted_cb([Decision.ALLOW_ONCE] * 3)[0])
        seen: list[tuple[int, int | None]] = []
        handler = BoxToolEventHandler(
            gate, self._CALL_MAP, max_iterations=4,
            on_progress=lambda c, m: seen.append((c, m)),
        )
        self._approve(handler)
        self._approve(handler)
        handler.reset_iterations()
        self.assertEqual(seen, [(1, 4), (2, 4), (0, 4)])

    def test_denied_call_does_not_consume_budget(self) -> None:
        s, _ = _settings_in_tmp()
        gate = PermissionGate(
            s, _scripted_cb([Decision.DENY, Decision.ALLOW_ONCE,
                             Decision.ALLOW_ONCE])[0]
        )
        handler = BoxToolEventHandler(gate, self._CALL_MAP, max_iterations=2)
        self.assertFalse(self._approve(handler))  # denied — not counted
        self.assertTrue(self._approve(handler))   # 1
        self.assertTrue(self._approve(handler))   # 2


if __name__ == "__main__":
    unittest.main()
