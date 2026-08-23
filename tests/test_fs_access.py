"""On-the-fly file access: scoped per-path grants in the permission gate, and
the filesystem resolver that honors workspace + granted paths.

The security-relevant decisions under test: a path OUTSIDE the workspace must
be approved per-path (never via blanket fn-name trust), only reaches the tool
once granted, and the grant scope (turn / chat / persisted) is respected.
PermissionGate imports without the SDK, so these run standalone.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from box_chat.config import Settings
from box_chat.permissions import Decision, PermissionGate
from box_chat.tools import filesystem as fs


class _Recorder:
    """ask_user_cb stub that answers with a fixed decision and records calls."""
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        self.calls: list[tuple] = []

    def __call__(self, tool_id, fn_name, args, risky, on_answer) -> None:
        self.calls.append((tool_id, fn_name, dict(args), risky))
        on_answer(self.decision)


class OutsidePathGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory()
        self._out = tempfile.TemporaryDirectory()
        self.ws = self._ws.name
        self.secret = str(Path(self._out.name) / "blacklist.conf")
        Path(self.secret).write_text("blacklist nouveau")
        fs.clear_ephemeral_grants()
        fs.set_active_conversation(None)

    def tearDown(self) -> None:
        fs.clear_ephemeral_grants()
        fs.set_active_conversation(None)
        self._ws.cleanup()
        self._out.cleanup()

    def _settings(self, allow_outside: bool) -> Settings:
        s = Settings()
        s.tool_fs_root = self.ws
        s.tool_fs_enabled = True
        s.tool_fs_allow_outside = allow_outside
        s.tool_fs_extra_roots = []
        return s

    def _gate(self, s: Settings, decision: Decision) -> tuple[PermissionGate, _Recorder]:
        rec = _Recorder(decision)
        return PermissionGate(s, rec), rec

    def _read(self, args, s, gate):
        return gate.decide("fs_read", args, risky=False, tool_id="filesystem")

    def test_inside_path_not_handled_as_outside(self) -> None:
        s = self._settings(allow_outside=True)
        gate, rec = self._gate(s, Decision.DENY)
        self.assertIsNone(
            gate._decide_outside_path("fs_read", {"path": "a.txt"}, False, "filesystem")
        )
        self.assertEqual(rec.calls, [])

    def test_outside_ignored_when_feature_off(self) -> None:
        s = self._settings(allow_outside=False)
        gate, rec = self._gate(s, Decision.ALLOW_ONCE)
        self.assertIsNone(
            gate._decide_outside_path("fs_read", {"path": self.secret}, False, "filesystem")
        )
        self.assertEqual(rec.calls, [])

    def test_allow_once_is_turn_scoped(self) -> None:
        s = self._settings(allow_outside=True)
        gate, rec = self._gate(s, Decision.ALLOW_ONCE)
        self.assertTrue(self._read({"path": self.secret}, s, gate))
        self.assertEqual(len(rec.calls), 1)
        self.assertIsNotNone(fs.resolve_access(s, self.secret))  # granted this turn
        self.assertEqual(s.tool_fs_extra_roots, [])              # not persisted
        fs.clear_turn_grants()                                   # next send
        self.assertIsNone(fs.resolve_access(s, self.secret))     # re-prompts now

    def test_allow_chat_scoped_to_conversation(self) -> None:
        s = self._settings(allow_outside=True)
        gate, _ = self._gate(s, Decision.ALLOW_CHAT)
        gate.set_active_conversation(7)
        self.assertTrue(self._read({"path": self.secret}, s, gate))
        self.assertIsNotNone(fs.resolve_access(s, self.secret))  # visible in chat 7
        gate.set_active_conversation(8)
        self.assertIsNone(fs.resolve_access(s, self.secret))     # not in chat 8
        gate.set_active_conversation(7)
        self.assertIsNotNone(fs.resolve_access(s, self.secret))  # back in chat 7
        self.assertEqual(s.tool_fs_extra_roots, [])              # not persisted

    def test_deny_blocks_and_does_not_grant(self) -> None:
        s = self._settings(allow_outside=True)
        gate, _ = self._gate(s, Decision.DENY)
        self.assertFalse(self._read({"path": self.secret}, s, gate))
        self.assertIsNone(fs.resolve_access(s, self.secret))

    def test_always_allow_persists(self) -> None:
        s = self._settings(allow_outside=True)
        gate, _ = self._gate(s, Decision.ALLOW_TRUST)
        self.assertTrue(self._read({"path": self.secret}, s, gate))
        self.assertIn(str(Path(self.secret)), s.tool_fs_extra_roots)
        fs.clear_ephemeral_grants()                              # survives session wipe
        self.assertIsNotNone(fs.resolve_access(s, self.secret))

    def test_forget_persisted_path(self) -> None:
        s = self._settings(allow_outside=True)
        fs.grant_path_persist(s, self.secret)
        self.assertIsNotNone(fs.resolve_access(s, self.secret))
        fs.forget_persisted_path(s, str(Path(self.secret)))
        self.assertIsNone(fs.resolve_access(s, self.secret))

    def test_already_granted_does_not_reprompt(self) -> None:
        s = self._settings(allow_outside=True)
        fs.grant_path_turn(self.secret)
        gate, rec = self._gate(s, Decision.DENY)  # would deny IF it prompted
        self.assertTrue(self._read({"path": self.secret}, s, gate))
        self.assertEqual(rec.calls, [])

    def test_grant_for_one_path_does_not_grant_another(self) -> None:
        s = self._settings(allow_outside=True)
        other = str(Path(self._out.name) / "other.conf")
        Path(other).write_text("x")
        gate, _ = self._gate(s, Decision.ALLOW_TRUST)
        self._read({"path": self.secret}, s, gate)
        self.assertIsNone(fs.resolve_access(s, other))  # sibling never approved

    def test_tool_reads_granted_outside_file(self) -> None:
        s = self._settings(allow_outside=True)
        fs.grant_path_turn(self.secret)
        read = fs._make_fs_read(s)
        self.assertEqual(read(self.secret).strip(), "blacklist nouveau")


if __name__ == "__main__":
    unittest.main()
