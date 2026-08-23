"""Cross-thread permission gate for Phase 4 tool calls.

The LiteRT-LM SDK calls :meth:`BoxToolEventHandler.approve_tool_call` from
the engine worker thread. We turn that into a synchronous prompt by:

1. Shipping a description of the call to the UI thread via an ``ask_user_cb``
   the caller supplies (the UI layer wires this to ``GLib.idle_add`` and an
   :class:`Adw.AlertDialog`).
2. Blocking on a ``threading.Event`` until the user clicks a response.
3. Returning the boolean the SDK expects.

The gate decision logic itself is pure stdlib — no GTK in here — so the
GTK4 UI can be swapped for Qt/Tauri without touching this file.
"""
from __future__ import annotations

import enum
import hashlib
import json
import logging
import threading
from collections.abc import Callable
from typing import Any

# IMPORTANT: do NOT import litert_lm at module top.
# The engine worker thread is the only place that should trigger its first
# import — litert_lm uses XNNPACK threadpools that get bound to the thread
# that initialises them, and if the GTK main thread imports it first
# (because window.py imports permissions.py at startup) inference slows
# down dramatically. ``_handler_class()`` below defers the import so it
# happens on whichever thread actually instantiates the handler — by
# which point the worker has already imported and cached the SDK.

from .config import Settings
from .tools import permission_mode

log = logging.getLogger(__name__)


class Decision(enum.Enum):
    DENY = "deny"
    ALLOW_ONCE = "allow_once"
    ALLOW_CHAT = "allow_chat"
    ALLOW_TRUST = "allow_trust"


# UI-side callback. Receives:
#   tool_id      — e.g. "filesystem"
#   fn_name      — e.g. "fs_read"
#   args         — dict of the call's arguments
#   risky        — True if the call is write/execute (no "trust always")
#   on_answer    — invoke this with a Decision when the user clicks
#
# The callback returns immediately; it must hand the question to the main
# thread and let the gate's ``done`` event block the worker.
AskUserCB = Callable[
    [str, str, dict, bool, Callable[[Decision], None]],
    None,
]


def _args_hash(fn_name: str, args: dict) -> str:
    blob = json.dumps(
        {"fn": fn_name, "args": args}, sort_keys=True, default=str
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _granular_trust_key(fn_name: str, args: dict) -> str:
    return f"{fn_name}::{_args_hash(fn_name, args)}"


class PermissionGate:
    """Owns trust state and routes prompts to the UI.

    Trust scopes:

    - **persistent** — entries in ``settings.tool_always_allow``. Either a
      bare ``fn_name`` (blanket) or ``fn_name::<hash>`` (this exact call).
      Written automatically when the user clicks "Always allow" on a
      non-risky tool. Risky tools never auto-write here.
    - **per-chat** — in-memory ``set[str]`` of function names keyed by
      conversation id. Cleared when the chat is closed or deleted.
    """

    def __init__(self, settings: Settings, ask_user_cb: AskUserCB) -> None:
        self._settings = settings
        self._ask_user = ask_user_cb
        self._lock = threading.Lock()
        self._chat_trust: dict[int, set[str]] = {}
        self._active_conv_id: int | None = None

    # ── conversation lifecycle ────────────────────────────────────────
    def set_active_conversation(self, conv_id: int | None) -> None:
        with self._lock:
            self._active_conv_id = conv_id
        # Keep the filesystem grant registry's notion of the active chat in
        # sync so per-chat path grants resolve against the right conversation.
        from .tools import filesystem as fsmod
        fsmod.set_active_conversation(conv_id)

    def clear_chat_trust(self, conv_id: int) -> None:
        with self._lock:
            self._chat_trust.pop(conv_id, None)

    # ── main entry point — called from the engine worker thread ───────
    def decide(
        self,
        fn_name: str,
        args: dict,
        *,
        risky: bool,
        tool_id: str,
    ) -> bool:
        """Block until a decision is reached. Returns True to allow."""
        # On-the-fly access: a filesystem call targeting a path OUTSIDE the
        # workspace is decided per-PATH, before the blanket fn-name trust
        # scopes — otherwise one "always allow fs_read" would silently grant
        # every path on disk. Returns None when this isn't an outside-path
        # call, so we fall through to the normal logic below.
        outside = self._decide_outside_path(fn_name, args, risky, tool_id)
        if outside is not None:
            return outside

        if self._is_persistent_trust(fn_name, args):
            return True

        if not risky:
            if self._is_chat_trust(fn_name):
                return True
            if permission_mode(self._settings, tool_id) == "trust":
                return True

        decision = self._prompt(tool_id, fn_name, args, risky)

        # Record any persistent / per-chat trust the user just granted.
        # Risky tools are deliberately excluded from "always allow" /
        # "allow for chat" — the UI shouldn't even surface those buttons
        # for them, but enforce here too as defense in depth.
        if not risky:
            if decision == Decision.ALLOW_CHAT:
                self._remember_chat(fn_name)
            elif decision == Decision.ALLOW_TRUST:
                self._remember_always(fn_name)

        return decision in (
            Decision.ALLOW_ONCE,
            Decision.ALLOW_CHAT,
            Decision.ALLOW_TRUST,
        )

    def _decide_outside_path(
        self, fn_name: str, args: dict, risky: bool, tool_id: str
    ) -> bool | None:
        """Per-path decision for filesystem calls that target a path outside
        the workspace. Returns True/False once decided, or None if this isn't
        such a call (so :meth:`decide` continues with the normal logic)."""
        if tool_id != "filesystem":
            return None
        if not getattr(self._settings, "tool_fs_allow_outside", False):
            return None
        path = args.get("path")
        if not isinstance(path, str) or not path:
            return None
        from .tools import filesystem as fsmod
        kind, full = fsmod.classify_request(self._settings, path)
        if kind != "outside" or full is None:
            return None  # inside workspace or invalid → normal handling
        # Already granted this path (session or persisted "always allow").
        if fsmod.is_path_granted(self._settings, full):
            return True
        # Prompt for THIS path. Out-of-workspace is sensitive, so even reads
        # prompt; writes/deletes stay risky (no "trust always").
        decision = self._prompt(tool_id, fn_name, args, risky)
        if decision == Decision.DENY:
            return False
        norm = str(full)
        if decision == Decision.ALLOW_TRUST and not risky:
            fsmod.grant_path_persist(self._settings, norm)   # global, persisted
        elif decision == Decision.ALLOW_CHAT and not risky:
            with self._lock:
                cid = self._active_conv_id
            fsmod.grant_path_chat(cid, norm)                 # this chat only
        else:
            # Allow once (or a risky tool, which only ever gets "once") → cover
            # just this user turn; re-prompts on the next send.
            fsmod.grant_path_turn(norm)
        return True

    def _prompt(
        self,
        tool_id: str,
        fn_name: str,
        args: dict,
        risky: bool,
    ) -> Decision:
        result: dict[str, Decision] = {"d": Decision.DENY}
        done = threading.Event()

        def on_answer(d: Decision) -> None:
            result["d"] = d
            done.set()

        try:
            self._ask_user(tool_id, fn_name, args, risky, on_answer)
        except Exception:
            log.exception("ask_user_cb raised; denying tool call")
            return Decision.DENY

        done.wait()
        return result["d"]

    # ── persistent trust ──────────────────────────────────────────────
    def _is_persistent_trust(self, fn_name: str, args: dict) -> bool:
        rules = set(getattr(self._settings, "tool_always_allow", []) or [])
        if fn_name in rules:
            return True
        return _granular_trust_key(fn_name, args) in rules

    def _remember_always(self, fn_name: str) -> None:
        with self._lock:
            entries = list(
                getattr(self._settings, "tool_always_allow", []) or []
            )
            if fn_name in entries:
                return
            entries.append(fn_name)
            self._settings.tool_always_allow = entries
            try:
                self._settings.save()
            except Exception:
                log.exception("Could not persist tool_always_allow")

    # ── per-chat trust ────────────────────────────────────────────────
    def _is_chat_trust(self, fn_name: str) -> bool:
        with self._lock:
            cid = self._active_conv_id
            if cid is None:
                return False
            return fn_name in self._chat_trust.get(cid, set())

    def _remember_chat(self, fn_name: str) -> None:
        with self._lock:
            cid = self._active_conv_id
            if cid is None:
                return
            self._chat_trust.setdefault(cid, set()).add(fn_name)


# UI-facing callback fired AFTER each tool call resolves (approved+ran, OR
# denied). Same threading rules as the permission prompt: the UI must
# bounce to the main thread via GLib.idle_add before touching widgets.
ToolEventCB = Callable[[str, dict, str, bool], None]
#                       fn   args  result denied

# UI-facing callback fired when the agent iteration counter changes (a tool
# call was approved, or the counter was reset for a new send). Args are
# ``(current, maximum)`` where ``maximum`` is None when no cap is in force.
# Same threading rules as ToolEventCB — bounce to the main thread.
ProgressCB = Callable[[int, "int | None"], None]


def _extract_response_text(tool_response: Any) -> str:
    """Pull a human-readable string out of whatever the SDK gives us.

    The SDK doesn't document the shape of tool_response, so this is
    defensive — try a few likely keys, fall back to repr.
    """
    if isinstance(tool_response, str):
        return tool_response
    if isinstance(tool_response, dict):
        # Common LiteRT shape: content list with {"type":"text","text":...}
        content = tool_response.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    txt = item.get("text")
                    if isinstance(txt, str):
                        parts.append(txt)
            if parts:
                return "\n".join(parts)
        for k in ("result", "output", "text", "value"):
            v = tool_response.get(k)
            if isinstance(v, str):
                return v
    return repr(tool_response)


_handler_cls = None


def _handler_class():
    """Define BoxToolEventHandler lazily on first instantiation so the
    ``import litert_lm`` happens *after* the engine worker thread has
    already imported and initialised the SDK."""
    global _handler_cls
    if _handler_cls is not None:
        return _handler_cls

    import litert_lm

    class _BoxToolEventHandlerImpl(litert_lm.ToolEventHandler):
        def __init__(
            self,
            gate: PermissionGate,
            call_map: dict[str, dict[str, Any]],
            on_tool_event: ToolEventCB | None = None,
            max_iterations: int | None = None,
            on_progress: ProgressCB | None = None,
        ) -> None:
            super().__init__()
            self._gate = gate
            self._call_map = dict(call_map)
            self._on_tool_event = on_tool_event
            # Agent-mode tool-call budget. None = no cap (plain tool use).
            # Counts *approved* (about-to-run) calls per user-send so a
            # confused agent can't loop forever. Reset by the window before
            # each send via ``reset_iterations``.
            self._max_iterations = max_iterations
            self._on_progress = on_progress
            self._iter_count = 0
            self._iter_lock = threading.Lock()
            # Pending approvals waiting on their corresponding response.
            # The SDK appears to fire approve → run → process for each
            # call serially, so a list-as-queue is sufficient. If/when
            # parallel tool calling lands we'd switch to a dict keyed
            # by call_id.
            self._pending: list[tuple[str, dict]] = []
            self._pending_lock = threading.Lock()

        def reset_iterations(self) -> None:
            """Zero the per-send tool-call counter. Called on the main
            thread right before the engine starts a new generation."""
            with self._iter_lock:
                self._iter_count = 0
            self._emit_progress(0)

        # Read-only accessors so the llama.cpp path can build its own
        # LlamaToolRunner from the same ingredients without re-plumbing
        # window.py — the SDK push-model and the HTTP agentic loop share the
        # gate, call map and UI callbacks, only the driving differs.
        @property
        def gate(self):
            return self._gate

        @property
        def call_map(self) -> dict[str, Any]:
            return self._call_map

        @property
        def on_tool_event(self):
            return self._on_tool_event

        @property
        def on_progress(self):
            return self._on_progress

        @property
        def max_iterations(self) -> int | None:
            return self._max_iterations

        def approve_tool_call(self, tool_call: dict[str, Any]) -> bool:
            # The SDK is loose about tool_call shape — Gemma 4 sometimes
            # emits flat {"name", "args"}, sometimes nested under
            # "function"/"function_call"/"tool_use". Check all known layouts
            # so a stray empty {"name": ""} dict doesn't lock the user out.
            fn_name = (
                tool_call.get("name")
                or (tool_call.get("function") or {}).get("name")
                or (tool_call.get("function_call") or {}).get("name")
                or (tool_call.get("tool_use") or {}).get("name")
                or ""
            )
            fn_name = str(fn_name or "")
            args = (
                tool_call.get("args")
                or tool_call.get("arguments")
                or (tool_call.get("function") or {}).get("arguments")
                or (tool_call.get("function_call") or {}).get("arguments")
                or (tool_call.get("tool_use") or {}).get("input")
                or {}
            )
            if not isinstance(args, dict):
                args = {"_raw": args}
            meta = self._call_map.get(fn_name)
            if meta is None:
                log.warning("Model invoked unknown tool: %s", fn_name)
                self._emit(fn_name, args, "Unknown tool — denied.", True)
                return False
            tool_id = str(meta.get("tool_id") or fn_name)
            risky = bool(meta.get("risky"))
            # Agent iteration cap — checked BEFORE the permission prompt so a
            # capped-out chain doesn't keep nagging the user. We've already
            # run `self._max_iterations` calls this send; refuse the next.
            if self._max_iterations is not None:
                with self._iter_lock:
                    capped = self._iter_count >= self._max_iterations
                if capped:
                    self._emit(
                        fn_name, args,
                        f"Agent iteration cap reached "
                        f"({self._max_iterations}). Stopping.",
                        True,
                    )
                    return False
            approved = self._gate.decide(
                fn_name, args, risky=risky, tool_id=tool_id
            )
            if not approved:
                self._emit(fn_name, args, "Permission denied by user.", True)
                return False
            with self._iter_lock:
                self._iter_count += 1
                count = self._iter_count
            self._emit_progress(count)
            with self._pending_lock:
                self._pending.append((fn_name, args))
            return True

        def process_tool_response(
            self, tool_response: dict[str, Any]
        ) -> dict[str, Any]:
            with self._pending_lock:
                fn_name, args = (
                    self._pending.pop(0)
                    if self._pending else ("?", {})
                )
            result_text = _extract_response_text(tool_response)
            self._emit(fn_name, args, result_text, False)
            return tool_response

        def _emit(
            self, fn_name: str, args: dict, result: str, denied: bool
        ) -> None:
            if self._on_tool_event is None:
                return
            try:
                self._on_tool_event(fn_name, args, result, denied)
            except Exception:
                log.exception("on_tool_event callback raised")

        def _emit_progress(self, current: int) -> None:
            if self._on_progress is None:
                return
            try:
                self._on_progress(current, self._max_iterations)
            except Exception:
                log.exception("on_progress callback raised")

    _handler_cls = _BoxToolEventHandlerImpl
    return _handler_cls


def BoxToolEventHandler(
    gate: PermissionGate,
    call_map: dict[str, dict[str, Any]],
    on_tool_event: ToolEventCB | None = None,
    max_iterations: int | None = None,
    on_progress: ProgressCB | None = None,
):
    """LiteRT-LM SDK adapter that delegates to a :class:`PermissionGate`.

    This is a factory (defined as a function, not a class) so the SDK
    subclass is only built once the SDK has been imported elsewhere — see
    the module docstring for why that matters.

    ``on_tool_event`` is invoked from the engine worker thread after each
    tool call resolves (approved+ran, OR denied). The UI layer must
    schedule its widget updates via ``GLib.idle_add``.

    ``max_iterations`` caps approved tool calls per user-send (agent mode);
    None disables the cap. ``on_progress`` reports ``(current, max)`` as the
    counter advances or resets. The returned handler exposes
    ``reset_iterations()`` for the window to call before each send.
    """
    return _handler_class()(
        gate, call_map, on_tool_event, max_iterations, on_progress
    )
