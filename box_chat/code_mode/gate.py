"""Permission gate for Box Code — Claude Code-style modes.

Much simpler than chat's PermissionGate because code mode has exactly two
policies and no cross-conversation state:

- ``"ask"``  — non-risky tools (read/list/glob/grep/todo/ask_user) run
  freely; risky ones (bash / write_file / edit_file) prompt the user with
  deny / allow once / allow for this session.
- ``"auto"`` — everything runs without prompting. The sandbox and the
  project-root scoping still confine what the tools can touch; this mode
  is what lets you hand the agent a task and walk away.

``decide()`` has the same signature LlamaToolRunner expects, so this
slots straight into the existing tool loop. Pure stdlib — no gi.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable

log = logging.getLogger(__name__)

# UI callback: (fn_name, args, on_answer). on_answer takes one of
# "deny" | "once" | "session" and may be called from any thread.
PermissionAskCB = Callable[[str, dict, Callable[[str], None]], None]


class CodePermissionGate:
    """Owns the mode + per-session trust for one agent session."""

    def __init__(
        self,
        mode: str = "ask",
        ask_cb: PermissionAskCB | None = None,
    ) -> None:
        self._mode = mode if mode in ("ask", "auto") else "ask"
        self._ask_cb = ask_cb
        self._session_allowed: set[str] = set()
        self._lock = threading.Lock()

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        if mode in ("ask", "auto"):
            self._mode = mode

    def decide(self, fn_name: str, args: dict, *, risky: bool, tool_id: str) -> bool:
        """Block until decided. Same contract as PermissionGate.decide."""
        if not risky or self._mode == "auto":
            return True
        with self._lock:
            if fn_name in self._session_allowed:
                return True
        if self._ask_cb is None:
            log.warning("no permission UI wired; denying %s", fn_name)
            return False

        result: dict[str, str] = {"d": "deny"}
        done = threading.Event()

        def on_answer(decision: str) -> None:
            result["d"] = str(decision)
            done.set()

        try:
            self._ask_cb(fn_name, dict(args), on_answer)
        except Exception:  # noqa: BLE001
            log.exception("permission ask callback raised; denying")
            return False
        done.wait()

        decision = result["d"]
        if decision == "session":
            with self._lock:
                self._session_allowed.add(fn_name)
        return decision in ("once", "session")
