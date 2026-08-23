"""Box Code runtime: gate, sessions, agent state machine, live loop.

The live class runs the real CodeAgent against llama-server +
qwen2.5-0.5b (same guards as tests/test_llama_tools.py) — skipped when
the binary or model is missing.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path

from box_chat.code_mode.gate import CodePermissionGate
from box_chat.code_mode.sessions import (
    CODE_SESSIONS_DIR,
    CodeSession,
    delete_session,
    list_sessions,
)
from box_chat.config import Settings
from box_chat.llama_server import LlamaServerError, find_server_binary

QWEN = Path(__file__).parent.parent / "vendor" / "test-models" / (
    "qwen2.5-0.5b-instruct-q5_k_m.gguf"
)


class GateTests(unittest.TestCase):
    def test_non_risky_always_allowed_in_ask(self):
        gate = CodePermissionGate("ask", ask_cb=None)
        self.assertTrue(
            gate.decide("read_file", {}, risky=False, tool_id="code")
        )

    def test_risky_denied_without_ui(self):
        gate = CodePermissionGate("ask", ask_cb=None)
        self.assertFalse(gate.decide("bash", {}, risky=True, tool_id="code"))

    def test_auto_allows_everything(self):
        gate = CodePermissionGate("auto", ask_cb=None)
        self.assertTrue(gate.decide("bash", {}, risky=True, tool_id="code"))

    def test_ask_once_and_session_memory(self):
        answers = iter(["once", "session", "deny"])
        asked: list[str] = []

        def cb(fn, args, on_answer):
            asked.append(fn)
            on_answer(next(answers))

        gate = CodePermissionGate("ask", ask_cb=cb)
        # "once" allows but doesn't remember
        self.assertTrue(gate.decide("bash", {}, risky=True, tool_id="code"))
        # "session" allows and remembers bash
        self.assertTrue(gate.decide("bash", {}, risky=True, tool_id="code"))
        # remembered — no third prompt for bash
        self.assertTrue(gate.decide("bash", {}, risky=True, tool_id="code"))
        self.assertEqual(asked, ["bash", "bash"])
        # a different risky fn still prompts (and gets the "deny")
        self.assertFalse(
            gate.decide("write_file", {}, risky=True, tool_id="code")
        )

    def test_ask_cb_from_other_thread(self):
        def cb(fn, args, on_answer):
            threading.Timer(0.05, on_answer, args=("once",)).start()

        gate = CodePermissionGate("ask", ask_cb=cb)
        self.assertTrue(gate.decide("bash", {}, risky=True, tool_id="code"))

    def test_invalid_mode_falls_back_to_ask(self):
        gate = CodePermissionGate("yolo", ask_cb=None)
        self.assertEqual(gate.mode, "ask")


class SessionTests(unittest.TestCase):
    def setUp(self):
        self._made: list[str] = []

    def tearDown(self):
        for sid in self._made:
            delete_session(sid)

    def _create(self) -> CodeSession:
        s = CodeSession.create("/tmp/proj", "/tmp/model.gguf")
        self._made.append(s.meta.session_id)
        return s

    def test_roundtrip_and_title(self):
        s = self._create()
        s.append({"type": "user", "text": "fix the bug in main.py"})
        s.append({"type": "tool", "name": "bash", "args": {"command": "ls"},
                  "result": "main.py", "denied": False})
        s.append({"type": "assistant", "text": "done", "completed": True})
        s2 = CodeSession.open(s.meta.session_id)
        self.assertEqual(s2.meta.title, "fix the bug in main.py")
        self.assertEqual(len(s2.events()), 3)
        self.assertEqual(s2.history(), [
            {"role": "user", "content": "fix the bug in main.py"},
            {"role": "assistant", "content": "done"},
        ])

    def test_listed_newest_first(self):
        a = self._create()
        b = self._create()
        ids = [m.session_id for m in list_sessions()]
        self.assertLess(ids.index(b.meta.session_id),
                        ids.index(a.meta.session_id))

    def test_torn_tail_line_ignored(self):
        s = self._create()
        s.append({"type": "user", "text": "hi"})
        with (CODE_SESSIONS_DIR / s.meta.session_id / "events.jsonl").open(
            "a", encoding="utf-8"
        ) as f:
            f.write('{"type": "assist')  # crash mid-write
        self.assertEqual(len(s.events()), 1)

    def test_delete(self):
        s = self._create()
        delete_session(s.meta.session_id)
        self.assertNotIn(
            s.meta.session_id, [m.session_id for m in list_sessions()]
        )


class AgentStateMachineTests(unittest.TestCase):
    """No server: exercise error path, busy rejection, shutdown."""

    def tearDown(self):
        delete_session(self._session.meta.session_id)

    def test_bad_model_reaches_error_state(self):
        from box_chat.code_mode.agent_loop import CodeAgent, CodeAgentCallbacks

        proj = tempfile.mkdtemp(prefix="codeloop-")
        self._session = CodeSession.create(proj, "/nonexistent.gguf")
        states, errors = [], []
        agent = CodeAgent(Settings(), self._session, CodeAgentCallbacks(
            on_state=lambda s, d: states.append(s),
            on_error=lambda m: errors.append(m),
        ))
        self.assertTrue(agent.send("hello"))
        for _ in range(200):
            if agent.state == "error":
                break
            time.sleep(0.05)
        self.assertEqual(agent.state, "error")
        self.assertTrue(errors and "bad_model" in errors[0])
        # busy/shutdown behavior
        agent.shutdown()
        self.assertFalse(agent.send("again"))  # worker gone
        # error event persisted
        kinds = [e.get("type") for e in self._session.events()]
        self.assertIn("error", kinds)


def _live_ready() -> str | None:
    try:
        find_server_binary()
    except LlamaServerError:
        return "no llama-server binary bundled"
    if not QWEN.is_file():
        return "qwen2.5-0.5b test model not present"
    return None


@unittest.skipIf(_live_ready() is not None, _live_ready() or "")
class LiveAgentLoopTests(unittest.TestCase):
    """Real model, real sandboxed tools, real agentic loop."""

    def tearDown(self):
        delete_session(self._session.meta.session_id)

    def test_write_file_via_agent(self):
        from box_chat.code_mode.agent_loop import CodeAgent, CodeAgentCallbacks

        proj = Path(tempfile.mkdtemp(prefix="codeloop-live-"))
        self._session = CodeSession.create(str(proj), str(QWEN))
        tools_used: list[str] = []
        settings = Settings()
        settings.code_permission_mode = "auto"
        settings.temperature = 0.0  # greedy — deterministic for a 0.5B model
        agent = CodeAgent(settings, self._session, CodeAgentCallbacks(
            on_tool_event=lambda fn, a, r, d: tools_used.append(fn),
        ))
        try:
            agent.send(
                "Use the write_file tool to create hello.txt containing "
                "exactly: hi"
            )
            deadline = time.time() + 300
            time.sleep(0.5)
            while agent.state in ("idle", "loading", "running"):
                if time.time() > deadline:
                    self.fail("agent did not finish in 300s")
                time.sleep(0.5)
            self.assertIn("write_file", tools_used)
            self.assertTrue((proj / "hello.txt").is_file())
            self.assertIn("hi", (proj / "hello.txt").read_text())
            # transcript persisted: user + tool + assistant
            kinds = [e["type"] for e in self._session.events()]
            self.assertIn("tool", kinds)
            self.assertIn("assistant", kinds)
        finally:
            agent.shutdown()


if __name__ == "__main__":
    unittest.main()
