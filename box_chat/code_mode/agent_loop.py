"""The Box Code agent runtime — one worker thread, one llama-server.

Deliberately independent of EngineManager: code mode owns its own
:class:`~box_chat.llama_backend.LlamaBackend` (which supervises its own
sandboxed llama-server child), so an agent session never touches chat
state. The agentic loop itself IS ``LlamaBackend.send`` — the same
multi-round OpenAI tool loop the chat side live-verified — driven here
with the code toolset, the code permission gate and session persistence.

Threading contract (mirrors EngineManager): every callback fires on the
worker thread; the UI layer must bounce to the main thread via
``GLib.idle_add``. Pure stdlib — no gi.
"""
from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..config import CACHE_DIR
from ..llama_backend import LlamaBackend
from ..llama_server import LlamaServerError
from ..llama_tools import LlamaToolRunner
from .agent_tools import AgentToolbox, AskUserCB
from .gate import CodePermissionGate, PermissionAskCB
from .prompts import build_system_prompt
from .sessions import CodeSession

log = logging.getLogger(__name__)

# Keep roughly this many chars of tool-result payload in the live message
# list before old tool results get stubbed out (v1 compaction — crude but
# prevents a long session from overflowing the server context).
_CHARS_PER_TOKEN_EST = 4
_TRIM_STUB = "[old tool output trimmed to save context]"

STATES = ("idle", "loading", "ready", "running", "stopped", "error")

LITERT_SUFFIXES = (".litertlm", ".task")


def is_litert_model(path: str) -> bool:
    return str(path).lower().endswith(LITERT_SUFFIXES)


class _LitertCodeBackend:
    """In-process LiteRT backend for Box Code (worker-thread only).

    Mirrors EngineManager's litert path: the SDK drives the tool loop
    itself (``automatic_tool_calling`` + BoxToolEventHandler), so unlike
    the llama path there's no HTTP loop here — just send and stream.
    """

    def __init__(self) -> None:
        self._engine = None
        self._conv = None
        self._loaded_key: tuple | None = None

    def is_loaded(self) -> bool:
        return self._conv is not None

    def load(
        self, path: str, system_prompt: str, history: list[dict],
        settings: Any, callables: list, handler: Any,
        max_num_tokens: int,
    ) -> None:
        import litert_lm

        from ..config import LITERTLM_CACHE
        from ..engine import _build_initial_messages

        key = (str(Path(path).resolve()), max_num_tokens)
        if self._loaded_key != key:
            self.unload()
            sdk_backend = (
                litert_lm.Backend.GPU
                if getattr(settings, "backend", "cpu") == "gpu"
                else litert_lm.Backend.CPU
            )
            engine = litert_lm.Engine(
                path, backend=sdk_backend, cache_dir=str(LITERTLM_CACHE),
                max_num_tokens=max_num_tokens,
            )
            engine.__enter__()
            self._engine = engine
            self._loaded_key = key
        else:
            # Same engine, fresh conversation state below.
            if self._conv is not None:
                try:
                    self._conv.__exit__(None, None, None)
                except Exception:  # noqa: BLE001
                    log.exception("error closing litert conversation")
                self._conv = None

        sampler = None
        if settings.temperature is not None or settings.top_k is not None \
                or settings.top_p is not None:
            sampler = litert_lm.SamplerConfig(
                temperature=settings.temperature,
                top_k=settings.top_k,
                top_p=settings.top_p,
            )
        conv = self._engine.create_conversation(
            messages=_build_initial_messages(
                system_prompt, history, max_tokens=max_num_tokens
            ),
            sampler_config=sampler,
            tools=callables,
            automatic_tool_calling=True,
            tool_event_handler=handler,
        )
        conv.__enter__()
        self._conv = conv

    def send(
        self, user_text: str, on_token, stop_flag, register_active,
    ) -> tuple[str, bool]:
        import litert_lm

        from ..engine import _build_message

        conv = self._conv
        if conv is None:
            raise RuntimeError("LiteRT conversation not loaded")
        message = _build_message(user_text, [], litert_lm)
        stream = conv.send_message_async(message)
        register_active(conv)  # cancel_process() aborts the stream
        full: list[str] = []
        try:
            for chunk in stream:
                if stop_flag.is_set():
                    try:
                        conv.cancel_process()
                    except Exception:  # noqa: BLE001
                        pass
                    return "".join(full), False
                for item in chunk.get("content", []) or []:
                    if item.get("type") == "text":
                        t = item.get("text", "")
                        if t:
                            full.append(t)
                            on_token(t)
        except Exception:
            if stop_flag.is_set():
                return "".join(full), False
            raise
        finally:
            register_active(None)
        return "".join(full), True

    def unload(self) -> None:
        if self._conv is not None:
            try:
                self._conv.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                log.exception("error closing litert conversation")
            self._conv = None
        if self._engine is not None:
            try:
                self._engine.__exit__(None, None, None)
            except Exception:  # noqa: BLE001
                log.exception("error closing litert engine")
            self._engine = None
        self._loaded_key = None


@dataclass
class CodeAgentCallbacks:
    """UI hooks. All fire on the agent worker thread."""

    on_state: Callable[[str, str], None] | None = None      # (state, detail)
    on_token: Callable[[str], None] | None = None
    on_tool_event: Callable[[str, dict, str, bool], None] | None = None
    on_progress: Callable[[int, int | None], None] | None = None
    on_todo: Callable[[str], None] | None = None
    on_turn_done: Callable[[str, bool], None] | None = None  # (text, completed)
    on_error: Callable[[str], None] | None = None
    ask_user: AskUserCB | None = None                        # ask_user tool
    ask_permission: PermissionAskCB | None = None            # risky-tool gate


@dataclass
class _CmdSend:
    text: str


@dataclass
class _CmdShutdown:
    done: threading.Event = field(default_factory=threading.Event)


class CodeAgent:
    """One agent session: project + model + persisted transcript."""

    def __init__(
        self,
        settings: Any,
        session: CodeSession,
        callbacks: CodeAgentCallbacks | None = None,
    ) -> None:
        self._settings = settings
        self._session = session
        self._cb = callbacks or CodeAgentCallbacks()
        self._state = "idle"
        self._q: queue.Queue = queue.Queue()
        self._stop_flag = threading.Event()
        self._active_lock = threading.Lock()
        self._active: Any = None  # cancel shim for the in-flight stream

        scratch = (
            CACHE_DIR / "code_scratch" / session.meta.session_id
        )
        self._toolbox = AgentToolbox(
            session.meta.project_dir,
            scratch,
            ask_user_cb=self._cb.ask_user,
            on_todo=self._on_todo,
            bash_timeout=getattr(settings, "code_bash_timeout", 120),
            web_enabled=getattr(settings, "code_web_enabled", False),
        )
        self._gate = CodePermissionGate(
            mode=getattr(settings, "code_permission_mode", "ask"),
            ask_cb=self._cb.ask_permission,
        )
        self._backend = LlamaBackend()
        self._lt: _LitertCodeBackend | None = None
        self._lt_handler: Any = None
        self._runner = LlamaToolRunner(
            self._toolbox.callables(),
            self._gate,
            self._toolbox.call_map(),
            on_tool_event=self._on_tool_event,
            on_progress=self._cb.on_progress,
            max_iterations=getattr(settings, "code_max_iterations", 100),
        )
        self._thread = threading.Thread(
            target=self._worker, name="box-code-agent", daemon=True
        )
        self._thread.start()

    # ── public API (any thread) ───────────────────────────────────────────
    @property
    def state(self) -> str:
        return self._state

    @property
    def session(self) -> CodeSession:
        return self._session

    @property
    def toolbox(self) -> AgentToolbox:
        return self._toolbox

    @property
    def gate(self) -> CodePermissionGate:
        return self._gate

    @property
    def backend_kind(self) -> str:
        return (
            "litert" if is_litert_model(self._session.meta.model_path)
            else "llama"
        )

    @property
    def sandbox_report(self):
        if self.backend_kind == "litert":
            return None  # in-process SDK — no subprocess to sandbox
        return self._backend.sandbox_report

    def set_permission_mode(self, mode: str) -> None:
        self._gate.set_mode(mode)

    def apply_tweaks(self) -> None:
        """Re-read the live-tunable code_* settings (Box Code settings
        dialog). Iteration cap + bash timeout apply immediately; context
        size and AGENTS.md apply at the next model (re)load."""
        self._runner._max_iterations = getattr(
            self._settings, "code_max_iterations", 100
        )
        if self._lt_handler is not None:
            self._lt_handler._max_iterations = getattr(
                self._settings, "code_max_iterations", 100
            )
        self._toolbox._bash_timeout = max(
            1, int(getattr(self._settings, "code_bash_timeout", 120))
        )

    def send(self, text: str) -> bool:
        """Queue one user turn. False if the agent is busy or shut down."""
        if self._state in ("running", "loading") or not self._thread.is_alive():
            return False
        self._q.put(_CmdSend(text=text))
        return True

    def stop(self) -> None:
        """Abort the in-flight turn (stream + tool loop)."""
        self._stop_flag.set()
        with self._active_lock:
            shim = self._active
        if shim is not None:
            try:
                shim.cancel_process()
            except Exception:  # noqa: BLE001
                pass

    def shutdown(self, timeout: float = 15.0) -> None:
        """Stop everything and kill the llama-server child."""
        self.stop()
        cmd = _CmdShutdown()
        self._q.put(cmd)
        cmd.done.wait(timeout)

    # ── worker thread ─────────────────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            cmd = self._q.get()
            if isinstance(cmd, _CmdShutdown):
                try:
                    self._backend.unload()
                except Exception:  # noqa: BLE001
                    log.exception("backend unload failed")
                if self._lt is not None:
                    try:
                        self._lt.unload()
                    except Exception:  # noqa: BLE001
                        log.exception("litert unload failed")
                self._set_state("idle", "shut down")
                cmd.done.set()
                return
            if isinstance(cmd, _CmdSend):
                self._handle_send(cmd.text)

    def _handle_send(self, text: str) -> None:
        self._stop_flag.clear()
        lt = self.backend_kind == "litert"
        try:
            loaded = (
                self._lt is not None and self._lt.is_loaded()
                if lt else self._backend.is_loaded()
            )
            if not loaded:
                self._set_state(
                    "loading",
                    Path(self._session.meta.model_path).name,
                )
                if lt:
                    self._load_litert()
                else:
                    self._load_backend()
            if not lt:
                self._trim_old_tool_results()
            self._session.append({"type": "user", "text": text})
            self._set_state("running", "")
            if lt:
                self._lt_handler.reset_iterations()
                reply, completed = self._lt.send(
                    text,
                    on_token=self._on_token,
                    stop_flag=self._stop_flag,
                    register_active=self._register_active,
                )
            else:
                self._runner.reset()
                reply, completed = self._backend.send(
                    text,
                    on_token=self._on_token,
                    stop_flag=self._stop_flag,
                    register_active=self._register_active,
                )
            self._session.append({
                "type": "assistant", "text": reply, "completed": completed,
            })
            if not completed and self._stop_flag.is_set() and not lt:
                # A cancelled turn never reached backend bookkeeping — record
                # the partial turn ourselves so resume history stays honest.
                # (The LiteRT conversation keeps its own state.)
                self._backend.record_turn(text, reply)
            if self._cb.on_turn_done is not None:
                self._cb.on_turn_done(reply, completed)
            self._set_state(
                "stopped" if not completed else "ready", ""
            )
        except LlamaServerError as exc:
            log.error("agent turn failed: %s: %s", exc.kind, exc)
            self._session.append({
                "type": "error", "kind": exc.kind, "text": str(exc),
            })
            self._emit_error(f"{exc.kind}: {exc}")
            self._set_state("error", exc.kind)
        except Exception as exc:  # noqa: BLE001
            log.exception("agent turn crashed")
            self._session.append({"type": "error", "text": str(exc)})
            self._emit_error(str(exc))
            self._set_state("error", "internal")

    def _load_backend(self) -> None:
        meta = self._session.meta
        self._backend.load(
            meta.model_path,
            build_system_prompt(
                meta.project_dir,
                self._project_instructions(),
                web_enabled=self._toolbox.web_enabled,
            ),
            self._session.history(),
            self._settings,
            temperature=(
                self._settings.code_temperature
                if getattr(self._settings, "code_temperature", -1.0) >= 0
                else self._settings.temperature
            ),
            top_k=self._settings.top_k,
            top_p=self._settings.top_p,
            max_num_tokens=getattr(self._settings, "code_max_context", 8192),
            tool_runner=self._runner,
        )

    def _load_litert(self) -> None:
        """In-process LiteRT: the SDK owns the tool loop; the shared gate,
        call map and iteration cap ride in via BoxToolEventHandler — the
        same adapter chat's agent mode uses."""
        from ..permissions import BoxToolEventHandler

        meta = self._session.meta
        if self._lt_handler is None:
            self._lt_handler = BoxToolEventHandler(
                self._gate,
                self._toolbox.call_map(),
                on_tool_event=self._on_tool_event,
                max_iterations=getattr(
                    self._settings, "code_max_iterations", 100
                ),
                on_progress=self._cb.on_progress,
            )
        if self._lt is None:
            self._lt = _LitertCodeBackend()
        self._lt.load(
            meta.model_path,
            build_system_prompt(
                meta.project_dir,
                self._project_instructions(),
                web_enabled=self._toolbox.web_enabled,
            ),
            self._session.history(),
            self._settings,
            self._toolbox.callables(),
            self._lt_handler,
            max_num_tokens=getattr(self._settings, "code_max_context", 8192),
        )

    def _project_instructions(self) -> str:
        """AGENTS.md / CLAUDE.md from the project root (first found, capped),
        same per-repo convention as Claude Code and opencode."""
        if not getattr(self._settings, "code_read_agents_md", True):
            return ""
        root = Path(self._session.meta.project_dir)
        for name in ("AGENTS.md", "CLAUDE.md", ".agents.md"):
            p = root / name
            try:
                if p.is_file():
                    return p.read_text(
                        encoding="utf-8", errors="replace"
                    )[:8000]
            except OSError:
                continue
        return ""

    def _trim_old_tool_results(self) -> None:
        """v1 compaction: stub out old tool payloads once the live message
        list outgrows the context estimate. Touches the backend's message
        list directly — same package, documented liberty."""
        msgs = self._backend._messages
        budget = self._backend.context_estimate * _CHARS_PER_TOKEN_EST
        total = sum(len(m.get("content") or "") for m in msgs)
        if total <= budget:
            return
        # Oldest first; never touch the trailing 10 messages (live work).
        for m in msgs[:-10]:
            if m.get("role") == "tool" and m.get("content") not in (
                None, _TRIM_STUB
            ):
                total -= len(m["content"]) - len(_TRIM_STUB)
                m["content"] = _TRIM_STUB
                if total <= budget:
                    break

    # ── worker-side callback shims ────────────────────────────────────────
    def _register_active(self, shim: Any) -> None:
        with self._active_lock:
            self._active = shim

    def _set_state(self, state: str, detail: str) -> None:
        self._state = state
        if self._cb.on_state is not None:
            try:
                self._cb.on_state(state, detail)
            except Exception:  # noqa: BLE001
                log.exception("on_state callback raised")

    def _on_token(self, token: str) -> None:
        if self._cb.on_token is not None:
            try:
                self._cb.on_token(token)
            except Exception:  # noqa: BLE001
                log.exception("on_token callback raised")

    def _on_tool_event(
        self, fn_name: str, args: dict, result: str, denied: bool
    ) -> None:
        self._session.append({
            "type": "tool", "name": fn_name, "args": args,
            "result": result[:20_000], "denied": denied,
        })
        if self._cb.on_tool_event is not None:
            try:
                self._cb.on_tool_event(fn_name, args, result, denied)
            except Exception:  # noqa: BLE001
                log.exception("on_tool_event callback raised")

    def _on_todo(self, todos: str) -> None:
        self._session.append({"type": "todo", "text": todos})
        if self._cb.on_todo is not None:
            try:
                self._cb.on_todo(todos)
            except Exception:  # noqa: BLE001
                log.exception("on_todo callback raised")

    def _emit_error(self, message: str) -> None:
        if self._cb.on_error is not None:
            try:
                self._cb.on_error(message)
            except Exception:  # noqa: BLE001
                log.exception("on_error callback raised")
