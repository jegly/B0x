"""Threaded wrapper around `litert_lm.Engine` (SDK 0.12.0+).

Why a worker thread?
--------------------
- `Engine(...)` constructor blocks for 10-30 s while weights load.
- `conversation.send_message_async(...)` returns a blocking iterator.
- GTK widgets must only be touched from the main thread, so streamed tokens are
  shuttled back via `GLib.idle_add(...)`.

Design
------
- One persistent worker thread per `EngineManager` instance.
- Commands are submitted via a `queue.Queue`; the worker processes them serially.
- Events are published back through user-supplied callbacks wrapped in
  `GLib.idle_add` by the GUI layer.
"""
from __future__ import annotations

import logging
import queue
import threading
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

log = logging.getLogger(__name__)


# ──── Event types streamed back to the UI ────────────────────────────────────

@dataclass
class EvtLoading:
    model_path: str

@dataclass
class EvtReady:
    model_path: str
    # Which inference backend serves this model, plus what it can do — the
    # UI keys off these instead of guessing from the file extension.
    backend_kind: str = "litert"      # "litert" | "llama"
    supports_tools: bool = True
    supports_vision: bool = True

@dataclass
class EvtToken:
    text: str

@dataclass
class EvtComplete:
    full_text: str

@dataclass
class EvtError:
    message: str
    # Structured error class so the UI can branch without string-matching
    # backend-specific text. kinds: "error" | "load_failed" | "unsupported"
    # | plus llama_server.LlamaServerError kinds (oom, crash, bad_model, …).
    kind: str = "error"

@dataclass
class EvtStopped:
    full_text: str

Event = (
    EvtLoading | EvtReady | EvtToken | EvtComplete | EvtError | EvtStopped
)

EventCallback = Callable[[Event], None]


# ──── Commands ────────────────────────────────────────────────────────────────

class _Cmd:
    pass

@dataclass
class _CmdLoadModel(_Cmd):
    path: str
    system_prompt: str
    history: list[dict]
    cb: EventCallback
    backend: str = "cpu"
    enable_speculative_decoding: bool = False
    enable_vision: bool = False
    enable_audio: bool = False
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_num_tokens: int | None = None
    # Phase 4 — tool calling. ``tools`` is the SDK's expected list (plain
    # Python callables or litert_lm.Tool objects). ``tool_event_handler``
    # is a litert_lm.ToolEventHandler subclass driving the permission UI.
    tools: list[Any] | None = None
    tool_event_handler: Any | None = None
    # Settings object carrying the llama_* fields — only read when ``path``
    # is a .gguf (the llama.cpp backend); the litert path ignores it.
    llama_settings: Any | None = None

@dataclass
class _CmdSend(_Cmd):
    user_text: str
    cb: EventCallback
    # Media-only attachments (text/PDF content is already folded into user_text).
    # Each dict: {"type": "image"|"audio", "path": str}
    attachments: list[dict] = field(default_factory=list)

class _CmdStop(_Cmd):
    pass

class _CmdUnload(_Cmd):
    """Free the loaded model (litert or llama) without shutting down the
    worker. Used by Box Code to reclaim RAM before loading its own model."""
    pass

class _CmdShutdown(_Cmd):
    pass

@dataclass
class _CmdCaptionImage(_Cmd):
    """Caption an image using a fresh, throwaway conversation.

    The temp conversation isolates the captioning from the user's real chat
    history — no pollution either direction. ``cb(caption, error)`` is
    invoked on the engine worker thread (callers bounce to the main
    thread via ``GLib.idle_add`` themselves).
    """
    image_path: str
    cb: Callable[[str | None, str | None], None]
    prompt: str = (
        "Describe this image in detail for search. Cover the main subject, "
        "scene, notable objects, any visible text, and overall mood. Be "
        "factual and concise — about 3-5 sentences."
    )


@dataclass
class _CmdAuditFile(_Cmd):
    """Audit a (potentially large) text file via a chunked map-reduce pass.

    The file is read off disk, split into window-sized sections, and each
    section is scored by the model in its own throwaway conversation — so the
    file never has to fit the chat's context window and the live chat
    conversation is left intact (rebuilt afterwards, like the captioning
    detour). ``on_progress(done, total, phase)`` fires per section on the
    worker thread; ``cb(report, error)`` delivers the final report.
    """
    path: str
    focus: str
    cb: Callable[[str | None, str | None], None]
    on_progress: Callable[[int, int, str], None]
    user_text: str = ""
    max_chunks: int = 40


# ──── Manager ────────────────────────────────────────────────────────────────

class EngineManager:
    """Owns the worker thread, the Engine, and the active Conversation."""

    def __init__(self) -> None:
        self._q: queue.Queue[_Cmd] = queue.Queue()
        self._stop_flag = threading.Event()
        self._ready = threading.Event()
        # Handle to the currently-streaming conversation, set by the worker
        # while a send is in flight. ``stop()`` (UI thread) reads this to
        # invoke ``cancel_process()`` *immediately* instead of waiting for
        # the worker to notice ``_stop_flag`` at the next chunk boundary —
        # at <2 tok/s that boundary can be 500-800 ms away.
        self._active_conversation: Any = None
        self._active_conv_lock = threading.Lock()
        self._current_model_path: str = ""
        self._current_backend: str = ""
        self._current_spec_decoding: bool = False
        self._current_vision: bool = False
        self._current_audio: bool = False
        self._current_max_num_tokens: int | None = None
        # Snapshot of the most recent LoadModel + messages exchanged since,
        # so we can rebuild the main conversation after a captioning detour
        # (the SDK only allows one live conversation per Engine).
        self._last_system_prompt: str = ""
        self._last_history: list[dict] = []
        self._session_messages: list[dict] = []
        self._last_sampler_kwargs: dict | None = None
        # Tools snapshot — rebuilt onto the new conversation after a
        # captioning detour, since the SDK only allows one live conversation
        # per Engine and a fresh create_conversation() defaults to no tools.
        self._last_tools: list[Any] | None = None
        self._last_tool_event_handler: Any | None = None

        self._thread = threading.Thread(
            target=self._run, name="litertlm-worker", daemon=True
        )
        self._thread.start()

    # ── public API ────────────────────────────────────────────────────────

    @property
    def current_model_path(self) -> str:
        return self._current_model_path

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def load_model(
        self,
        path: str,
        system_prompt: str,
        history: list[dict],
        cb: EventCallback,
        backend: str = "cpu",
        enable_speculative_decoding: bool = False,
        enable_vision: bool = False,
        enable_audio: bool = False,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_num_tokens: int | None = None,
        tools: list[Any] | None = None,
        tool_event_handler: Any | None = None,
        llama_settings: Any | None = None,
    ) -> None:
        self._q.put(_CmdLoadModel(
            path=path,
            system_prompt=system_prompt,
            history=history,
            cb=cb,
            backend=backend,
            enable_speculative_decoding=enable_speculative_decoding,
            enable_vision=enable_vision,
            enable_audio=enable_audio,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_num_tokens=max_num_tokens,
            tools=tools,
            tool_event_handler=tool_event_handler,
            llama_settings=llama_settings,
        ))

    def send(
        self,
        user_text: str,
        cb: EventCallback,
        attachments: list[dict] | None = None,
    ) -> None:
        if not self._ready.is_set():
            cb(EvtError(
                "No model loaded. Open Preferences and pick a .litertlm "
                "or .gguf model file."
            ))
            return
        self._stop_flag.clear()
        self._q.put(_CmdSend(
            user_text=user_text,
            cb=cb,
            attachments=attachments or [],
        ))

    def caption_image(
        self,
        image_path: str,
        cb: Callable[[str | None, str | None], None],
        prompt: str | None = None,
    ) -> None:
        """Queue an image-captioning request. Vision must be enabled on the
        loaded model (Preferences → Multimodal). ``cb(caption, error)`` is
        invoked on the engine worker thread."""
        if not self._ready.is_set():
            cb(None, "No model loaded.")
            return
        if not self._current_vision:
            cb(None, "Vision is not enabled — turn it on in Preferences → Multimodal.")
            return
        kwargs = {"image_path": image_path, "cb": cb}
        if prompt is not None:
            kwargs["prompt"] = prompt
        self._q.put(_CmdCaptionImage(**kwargs))

    def audit_file(
        self,
        path: str,
        focus: str,
        cb: Callable[[str | None, str | None], None],
        on_progress: Callable[[int, int, str], None],
        user_text: str = "",
        max_chunks: int = 40,
    ) -> None:
        """Queue a chunked map-reduce audit of ``path``. The caller is
        responsible for having resolved ``path`` inside the workspace.
        ``cb`` / ``on_progress`` are invoked on the engine worker thread."""
        if not self._ready.is_set():
            cb(None, "No model loaded.")
            return
        # A prior Stop leaves the flag set; clear it so the audit isn't
        # cancelled before it starts (mirrors send()).
        self._stop_flag.clear()
        self._q.put(_CmdAuditFile(
            path=path,
            focus=focus,
            cb=cb,
            on_progress=on_progress,
            user_text=user_text,
            max_chunks=max_chunks,
        ))

    def stop(self) -> None:
        """Abort the in-flight generation as quickly as possible.

        Both flags AND a direct ``cancel_process()`` call from this thread:
        the flag is the cooperative signal the worker checks between chunks,
        but the SDK's cancel call is what actually interrupts the C++
        generate loop mid-token. Without the direct cancel, the user sees
        ~one token of lag on fast hardware and seconds of lag at 1 tok/s.
        """
        self._stop_flag.set()
        with self._active_conv_lock:
            conv = self._active_conversation
        if conv is not None:
            # cancel_process() can block until the worker's generate/tool
            # loop reaches a cancellation point. If the worker is wedged
            # inside a slow tool call (e.g. a web_search), calling it inline
            # on the GTK main thread freezes the UI ("not responding") until
            # the tool returns. Fire it from a throwaway daemon thread so the
            # main thread returns immediately; the cooperative _stop_flag +
            # queued _CmdStop still drive the actual teardown.
            def _cancel() -> None:
                try:
                    conv.cancel_process()
                except Exception:
                    log.debug("cancel_process failed", exc_info=True)
            threading.Thread(
                target=_cancel, name="box-cancel", daemon=True
            ).start()
        self._q.put(_CmdStop())

    def unload_model(self) -> None:
        """Free whichever model is loaded (async; worker-queued). The next
        chat send needs a model reload — the window handles that already
        for conversation switches and model picks."""
        self._q.put(_CmdUnload())

    def shutdown(self) -> None:
        self._stop_flag.set()
        self._q.put(_CmdShutdown())
        self._thread.join(timeout=5)

    # ── worker loop ───────────────────────────────────────────────────────

    def _run(self) -> None:
        try:
            import litert_lm
            litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)
        except Exception as e:
            litert_lm = None
            import_error = (
                "Failed to import litert_lm. Install it with:\n\n"
                "    pip install litert-lm-api\n\n"
                f"Original error: {e}"
            )
        else:
            import_error = None

        engine = None
        conversation = None
        # GGUF path — lazily constructed so litert-only sessions never pay
        # for it. Which backend owns the loaded model is tracked in
        # ``backend_kind`` ("litert" | "llama" | None).
        llama_backend = None
        backend_kind: str | None = None

        def _get_llama_backend():
            nonlocal llama_backend
            if llama_backend is None:
                from .llama_backend import LlamaBackend
                llama_backend = LlamaBackend()
            return llama_backend

        def _release_llama() -> None:
            if llama_backend is not None:
                llama_backend.unload()

        def _release_conv() -> None:
            nonlocal conversation
            if conversation is not None:
                try:
                    conversation.__exit__(None, None, None)
                except Exception:
                    log.exception("Error closing conversation")
                conversation = None

        def _release_engine() -> None:
            nonlocal engine
            _release_conv()
            if engine is not None:
                try:
                    engine.__exit__(None, None, None)
                except Exception:
                    log.exception("Error closing engine")
                engine = None

        while True:
            cmd = self._q.get()

            if isinstance(cmd, _CmdShutdown):
                _release_engine()
                _release_llama()
                return

            if isinstance(cmd, _CmdStop):
                continue

            if isinstance(cmd, _CmdUnload):
                _release_engine()
                _release_llama()
                backend_kind = None
                self._ready.clear()
                self._current_model_path = ""
                continue

            if isinstance(cmd, _CmdLoadModel) and _is_gguf_path(cmd.path):
                # llama.cpp path. Free the litert engine first — the whole
                # point of one-model-at-a-time is not holding both in RAM.
                cmd.cb(EvtLoading(model_path=cmd.path))
                _release_engine()
                self._current_model_path = ""
                try:
                    from .llama_server import LlamaServerError

                    tool_runner = None
                    if cmd.tools and cmd.tool_event_handler is not None:
                        from .llama_tools import LlamaToolRunner
                        h = cmd.tool_event_handler
                        tool_runner = LlamaToolRunner(
                            callables=cmd.tools,
                            gate=h.gate,
                            call_map=h.call_map,
                            on_tool_event=h.on_tool_event,
                            on_progress=h.on_progress,
                            max_iterations=h.max_iterations,
                        )

                    backend = _get_llama_backend()
                    backend.load(
                        path=cmd.path,
                        system_prompt=cmd.system_prompt,
                        history=cmd.history,
                        settings=cmd.llama_settings,
                        temperature=cmd.temperature,
                        top_k=cmd.top_k,
                        top_p=cmd.top_p,
                        max_num_tokens=cmd.max_num_tokens,
                        tool_runner=tool_runner,
                    )
                except LlamaServerError as e:
                    self._ready.clear()
                    backend_kind = None
                    detail = f"\n\nserver log:\n{e.log_tail}" if e.log_tail else ""
                    cmd.cb(EvtError(
                        f"Failed to load GGUF model:\n\n{e}{detail}",
                        kind="load_failed",
                    ))
                    continue
                except Exception as e:  # noqa: BLE001
                    self._ready.clear()
                    backend_kind = None
                    cmd.cb(EvtError(
                        f"Failed to load GGUF model:\n\n{e}\n\n"
                        f"{traceback.format_exc(limit=2)}",
                        kind="load_failed",
                    ))
                    continue
                backend_kind = "llama"
                self._current_model_path = cmd.path
                self._current_vision = False
                self._ready.set()
                cmd.cb(EvtReady(
                    model_path=cmd.path,
                    backend_kind="llama",
                    supports_tools=backend.has_tools,
                    supports_vision=False,
                ))
                continue

            if isinstance(cmd, _CmdSend) and backend_kind == "llama":
                from .llama_server import LlamaServerError

                backend = _get_llama_backend()
                if cmd.attachments:
                    cmd.cb(EvtError(
                        "This model runs on the llama.cpp backend, which "
                        "doesn't support image/audio attachments yet — "
                        "remove the attachment or switch to a .litertlm "
                        "model with vision enabled.",
                        kind="unsupported",
                    ))
                    continue

                def _register(shim: Any) -> None:
                    with self._active_conv_lock:
                        self._active_conversation = shim

                try:
                    full, completed = backend.send(
                        cmd.user_text,
                        on_token=lambda t: cmd.cb(EvtToken(text=t)),
                        stop_flag=self._stop_flag,
                        register_active=_register,
                    )
                    if completed:
                        cmd.cb(EvtComplete(full_text=full))
                    else:
                        cmd.cb(EvtStopped(full_text=full))
                except LlamaServerError as e:
                    if e.kind in ("oom", "crash", "corruption", "assert"):
                        self._ready.clear()
                        backend_kind = None
                    detail = f"\n\nserver log:\n{e.log_tail}" if e.log_tail else ""
                    cmd.cb(EvtError(f"Generation failed:\n\n{e}{detail}",
                                    kind=e.kind))
                except Exception as e:  # noqa: BLE001
                    cmd.cb(EvtError(
                        f"Generation failed:\n\n{e}\n\n"
                        f"{traceback.format_exc(limit=2)}",
                    ))
                continue

            if isinstance(cmd, _CmdAuditFile) and backend_kind == "llama":
                from pathlib import Path as _Path

                from . import audit as auditmod

                backend = _get_llama_backend()
                try:
                    p = _Path(cmd.path)
                    total_size = p.stat().st_size
                    with p.open("r", encoding="utf-8", errors="replace") as fp:
                        data = fp.read(auditmod.MAX_AUDIT_BYTES)
                    truncated_bytes = total_size > auditmod.MAX_AUDIT_BYTES
                except OSError as e:
                    cmd.cb(None, f"Could not read {cmd.path}: {e}")
                    continue
                if not data.strip():
                    cmd.cb(None, "The file is empty or has no readable text.")
                    continue

                chunk_chars = auditmod.chunk_chars_for_context(
                    backend.context_estimate
                )

                def _llama_overflow(e: Exception) -> bool:
                    m = str(e).lower()
                    return ("context" in m or "n_ctx" in m
                            or "exceed" in m or "too long" in m)

                report: str | None = None
                error: str | None = None
                try:
                    report, error = auditmod.run_map_reduce_audit(
                        data=data, focus=cmd.focus, file_label=p.name,
                        chunk_chars=chunk_chars, max_chunks=cmd.max_chunks,
                        truncated_bytes=truncated_bytes,
                        audit_pass=backend.audit_pass,
                        is_token_overflow=_llama_overflow,
                        is_cancelled=self._stop_flag.is_set,
                        on_progress=cmd.on_progress,
                    )
                except Exception as e:  # noqa: BLE001
                    error = (
                        f"Audit failed: {e}\n\n{traceback.format_exc(limit=2)}"
                    )
                if report and not error:
                    backend.record_turn(
                        cmd.user_text or f"audit {p.name}", report
                    )
                cmd.cb(report, error)
                continue

            if isinstance(cmd, _CmdCaptionImage) and backend_kind == "llama":
                cmd.cb(None, "Vision isn't available on GGUF models.")
                continue

            if isinstance(cmd, _CmdLoadModel):
                if import_error is not None:
                    cmd.cb(EvtError(import_error, kind="load_failed"))
                    continue

                # Coming back from a GGUF model? Free its server process.
                if backend_kind == "llama":
                    _release_llama()
                backend_kind = "litert"

                cmd.cb(EvtLoading(model_path=cmd.path))
                try:
                    engine_params_changed = (
                        self._current_model_path != cmd.path
                        or self._current_backend != cmd.backend
                        or self._current_spec_decoding != cmd.enable_speculative_decoding
                        or self._current_vision != cmd.enable_vision
                        or self._current_audio != cmd.enable_audio
                        or self._current_max_num_tokens != cmd.max_num_tokens
                    )

                    if engine is not None and engine_params_changed:
                        _release_engine()

                    if engine is None:
                        from .config import LITERTLM_CACHE

                        sdk_backend = (
                            litert_lm.Backend.GPU
                            if cmd.backend == "gpu"
                            else litert_lm.Backend.CPU
                        )
                        # Vision encoder follows the user's main backend
                        # choice — Parlor's `vision_backend=GPU` setup
                        # cuts image prefill from ~1.5 s to ~0.5 s. Audio
                        # stays on CPU (Parlor keeps it there too;
                        # audio path doesn't benefit from GPU compute).
                        vision_be = sdk_backend if cmd.enable_vision else None
                        audio_be  = litert_lm.Backend.CPU if cmd.enable_audio  else None

                        engine_kwargs: dict[str, Any] = dict(
                            backend=sdk_backend,
                            cache_dir=str(LITERTLM_CACHE),
                            vision_backend=vision_be,
                            audio_backend=audio_be,
                            enable_speculative_decoding=(
                                cmd.enable_speculative_decoding or None
                            ),
                        )
                        if cmd.max_num_tokens:
                            engine_kwargs["max_num_tokens"] = cmd.max_num_tokens
                        engine = litert_lm.Engine(cmd.path, **engine_kwargs)
                        engine.__enter__()
                        self._current_model_path = cmd.path
                        self._current_backend = cmd.backend
                        self._current_spec_decoding = cmd.enable_speculative_decoding
                        self._current_vision = cmd.enable_vision
                        self._current_audio = cmd.enable_audio
                        self._current_max_num_tokens = cmd.max_num_tokens

                    _release_conv()
                    messages = _build_initial_messages(
                        cmd.system_prompt, cmd.history,
                        max_tokens=cmd.max_num_tokens or 4096,
                    )
                    sampler = None
                    sampler_kwargs: dict | None = None
                    if any(v is not None for v in (cmd.temperature, cmd.top_k, cmd.top_p)):
                        sampler_kwargs = dict(
                            temperature=cmd.temperature,
                            top_k=cmd.top_k,
                            top_p=cmd.top_p,
                        )
                        sampler = litert_lm.SamplerConfig(**sampler_kwargs)
                    conv_kwargs: dict[str, Any] = dict(
                        messages=messages,
                        sampler_config=sampler,
                    )
                    if cmd.tools:
                        conv_kwargs["tools"] = cmd.tools
                        conv_kwargs["automatic_tool_calling"] = True
                        if cmd.tool_event_handler is not None:
                            conv_kwargs["tool_event_handler"] = cmd.tool_event_handler
                    conversation = engine.create_conversation(**conv_kwargs)
                    conversation.__enter__()

                    # Snapshot for the captioning detour. Reset session messages
                    # on every fresh load so the rebuild starts from a clean base.
                    self._last_system_prompt = cmd.system_prompt
                    self._last_history = list(cmd.history)
                    self._last_sampler_kwargs = sampler_kwargs
                    self._last_tools = cmd.tools
                    self._last_tool_event_handler = cmd.tool_event_handler
                    self._session_messages = []

                    self._ready.set()
                    cmd.cb(EvtReady(model_path=cmd.path, backend_kind="litert"))

                except Exception as e:
                    self._ready.clear()
                    _release_engine()
                    self._current_model_path = ""
                    cmd.cb(EvtError(
                        f"Failed to load model:\n\n{e}\n\n"
                        f"{traceback.format_exc(limit=2)}",
                        kind="load_failed",
                    ))
                continue

            if isinstance(cmd, _CmdSend):
                if conversation is None:
                    cmd.cb(EvtError("Engine is not ready."))
                    continue

                full = []
                try:
                    message = _build_message(cmd.user_text, cmd.attachments, litert_lm)
                    stream: Iterable[dict[str, Any]] = (
                        conversation.send_message_async(message)
                    )
                    with self._active_conv_lock:
                        self._active_conversation = conversation
                    completed = False
                    try:
                        for chunk in stream:
                            if self._stop_flag.is_set():
                                try:
                                    conversation.cancel_process()
                                except Exception:
                                    pass
                                cmd.cb(EvtStopped(full_text="".join(full)))
                                break
                            for item in chunk.get("content", []) or []:
                                if item.get("type") == "text":
                                    t = item.get("text", "")
                                    if t:
                                        full.append(t)
                                        cmd.cb(EvtToken(text=t))
                        else:
                            completed = True
                            cmd.cb(EvtComplete(full_text="".join(full)))
                    except Exception:
                        # If stop() ran cancel_process() from the UI thread,
                        # the iterator may raise instead of yielding empty.
                        # Treat that as a clean stop, not an error.
                        if self._stop_flag.is_set():
                            cmd.cb(EvtStopped(full_text="".join(full)))
                        else:
                            raise
                    finally:
                        with self._active_conv_lock:
                            self._active_conversation = None
                    # Track session messages so we can rebuild the conversation
                    # after a captioning detour. Skip stopped runs (incomplete
                    # assistant turn would distort future context).
                    if completed:
                        if cmd.user_text:
                            self._session_messages.append({
                                "role": "user", "content": cmd.user_text,
                            })
                        self._session_messages.append({
                            "role": "assistant", "content": "".join(full),
                        })
                except Exception as e:
                    cmd.cb(EvtError(
                        f"Generation failed:\n\n{e}\n\n"
                        f"{traceback.format_exc(limit=2)}"
                    ))
                continue

            if isinstance(cmd, _CmdCaptionImage):
                if engine is None or not self._current_vision:
                    cmd.cb(None, "Vision support not enabled.")
                    continue
                # The SDK allows only one live conversation per Engine. We
                # close the main convo, do the captioning, then rebuild the
                # main convo from the snapshotted system_prompt + history +
                # any session messages exchanged since LoadModel.
                _release_conv()
                temp_conv = None
                caption_result: tuple[str | None, str | None] = (None, "Unknown")
                try:
                    sys_msg = [{
                        "role": "system",
                        "content": [{
                            "type": "text",
                            "text": "You are a precise image captioner. "
                                    "Describe images factually for search.",
                        }],
                    }]
                    temp_conv = engine.create_conversation(messages=sys_msg)
                    temp_conv.__enter__()
                    message = _build_message(
                        cmd.prompt,
                        [{"type": "image", "path": cmd.image_path}],
                        litert_lm,
                    )
                    parts: list[str] = []
                    for chunk in temp_conv.send_message_async(message):
                        for item in chunk.get("content", []) or []:
                            if item.get("type") == "text":
                                t = item.get("text", "")
                                if t:
                                    parts.append(t)
                    caption = "".join(parts).strip()
                    caption_result = (caption or None,
                                      None if caption else "Empty caption returned.")
                except Exception as e:
                    caption_result = (
                        None,
                        f"Captioning failed: {e}\n\n"
                        f"{traceback.format_exc(limit=2)}",
                    )
                finally:
                    if temp_conv is not None:
                        try:
                            temp_conv.__exit__(None, None, None)
                        except Exception:
                            log.exception("Error closing temp caption conversation")
                # Rebuild the main conversation so subsequent _CmdSend works.
                try:
                    combined_history = list(self._last_history) + list(self._session_messages)
                    messages = _build_initial_messages(
                        self._last_system_prompt, combined_history,
                        max_tokens=self._current_max_num_tokens or 4096,
                    )
                    sampler = (
                        litert_lm.SamplerConfig(**self._last_sampler_kwargs)
                        if self._last_sampler_kwargs else None
                    )
                    conv_kwargs: dict[str, Any] = dict(
                        messages=messages, sampler_config=sampler,
                    )
                    if self._last_tools:
                        conv_kwargs["tools"] = self._last_tools
                        conv_kwargs["automatic_tool_calling"] = True
                        if self._last_tool_event_handler is not None:
                            conv_kwargs["tool_event_handler"] = self._last_tool_event_handler
                    conversation = engine.create_conversation(**conv_kwargs)
                    conversation.__enter__()
                except Exception as e:
                    log.exception("Failed to rebuild main conversation after captioning")
                    self._ready.clear()
                    # Surface the rebuild failure so the user knows the chat is broken.
                    cmd.cb(None, f"Captioning OK but chat conv lost: {e}")
                    continue
                cmd.cb(*caption_result)
                continue

            if isinstance(cmd, _CmdAuditFile):
                if engine is None or not self._ready.is_set():
                    cmd.cb(None, "No model loaded.")
                    continue
                from pathlib import Path as _Path

                from . import audit as auditmod

                # 1. Read the file off disk (bounded). The window already
                #    resolved the path inside the workspace; here we just read.
                try:
                    p = _Path(cmd.path)
                    total_size = p.stat().st_size
                    with p.open("r", encoding="utf-8", errors="replace") as fp:
                        data = fp.read(auditmod.MAX_AUDIT_BYTES)
                    truncated_bytes = total_size > auditmod.MAX_AUDIT_BYTES
                except OSError as e:
                    cmd.cb(None, f"Could not read {cmd.path}: {e}")
                    continue
                if not data.strip():
                    cmd.cb(None, "The file is empty or has no readable text.")
                    continue

                # Size sections to fit the model's window. Log content (kernel
                # hex, timestamps, base64) tokenizes FAR denser than prose —
                # ~1.6 chars/token observed on dmesg, vs ~4 for English — so
                # size for a worst-case 1.5 chars/token and reserve room for
                # the instruction + the model's own findings. A section that
                # STILL overflows is bisected on the fly (see _scan_chunk).
                ctx = self._current_max_num_tokens or 4096
                _min_cpt = 1.5                  # densest expected chars/token
                _sys_tok = 320                  # instruction overhead
                _out_tok = max(700, ctx // 4)   # reserve for the model's reply
                _budget_tok = max(256, ctx - _out_tok - _sys_tok)
                chunk_chars = max(1200, int(_budget_tok * _min_cpt))
                reduce_chars = chunk_chars
                chunks = auditmod.chunk_lines(data, chunk_chars)
                chunks, sampled = auditmod.sample_chunks(chunks, cmd.max_chunks)
                total = len(chunks)
                file_label = p.name

                # SDK allows one live conversation per Engine — close the main
                # chat conv for the duration, rebuild it afterwards (captioning
                # pattern). Each audit pass runs in its own throwaway conv so
                # sections stay independent.
                _release_conv()

                # A single pass is bounded by max_out_chars so a runaway /
                # repetitive generation (small models do this) can't hang the
                # audit indefinitely. ~8000 chars (~2000 tokens) is ample for a
                # section finding or a combined report.
                def _audit_pass(system_text: str, user_text: str,
                                max_out_chars: int = 8000) -> str:
                    tconv = engine.create_conversation(messages=[{
                        "role": "system",
                        "content": [{"type": "text", "text": system_text}],
                    }])
                    tconv.__enter__()
                    parts: list[str] = []
                    out_len = 0
                    try:
                        with self._active_conv_lock:
                            self._active_conversation = tconv
                        for ch in tconv.send_message_async(user_text):
                            if self._stop_flag.is_set():
                                try:
                                    tconv.cancel_process()
                                except Exception:
                                    pass
                                break
                            for item in ch.get("content", []) or []:
                                if item.get("type") == "text":
                                    t = item.get("text", "")
                                    if t:
                                        parts.append(t)
                                        out_len += len(t)
                            if out_len >= max_out_chars:
                                try:
                                    tconv.cancel_process()
                                except Exception:
                                    pass
                                break
                    finally:
                        with self._active_conv_lock:
                            self._active_conversation = None
                        try:
                            tconv.__exit__(None, None, None)
                        except Exception:
                            log.exception("Error closing audit temp conversation")
                    return "".join(parts).strip()

                def _is_token_overflow(e: Exception) -> bool:
                    m = str(e).lower()
                    return ("too long" in m
                            or "exceeding the maximum number of tokens" in m)

                def _scan_chunk(ch, depth: int = 0) -> str:
                    """Audit one section; if it overflows the window even at
                    the conservative size, bisect by lines and recurse.
                    Returns the finding text ('' when nothing notable)."""
                    try:
                        out = _audit_pass(
                            map_sys, auditmod.map_user_message(file_label, ch)
                        )
                    except Exception as e:
                        if not _is_token_overflow(e) or depth >= 5:
                            raise
                        out = None
                    if out is not None:
                        return "" if auditmod.is_none_finding(out) else out
                    parts: list[str] = []
                    for h in auditmod.split_chunk(ch):
                        if self._stop_flag.is_set():
                            break
                        sub = _scan_chunk(h, depth + 1)
                        if sub:
                            parts.append(sub)
                    return "\n".join(parts)

                def _reduce_pass(items: list[str], depth: int = 0) -> str:
                    """Reduce findings into one report; on overflow, split the
                    list and reduce the halves, stitching text together if it
                    can't shrink any further (guarantees termination)."""
                    if not items:
                        return ""
                    if len(items) == 1:
                        try:
                            return _audit_pass(
                                reduce_sys,
                                auditmod.reduce_user_message(
                                    file_label, [items[0][:reduce_chars]]
                                ),
                            )
                        except Exception as e:
                            if _is_token_overflow(e):
                                return items[0][:reduce_chars]
                            raise
                    try:
                        return _audit_pass(
                            reduce_sys,
                            auditmod.reduce_user_message(file_label, items),
                        )
                    except Exception as e:
                        if not _is_token_overflow(e) or depth >= 6:
                            return "\n\n".join(items)
                        mid = len(items) // 2
                        left = _reduce_pass(items[:mid], depth + 1)
                        right = _reduce_pass(items[mid:], depth + 1)
                        try:
                            return _audit_pass(
                                reduce_sys,
                                auditmod.reduce_user_message(
                                    file_label, [left, right]
                                ),
                            )
                        except Exception as e2:
                            if _is_token_overflow(e2):
                                return left + "\n\n" + right
                            raise

                report: str | None = None
                error: str | None = None
                try:
                    map_sys = auditmod.map_system(cmd.focus)
                    reduce_sys = auditmod.reduce_system(cmd.focus)
                    findings: list[str] = []
                    cancelled = False
                    for i, ch in enumerate(chunks, 1):
                        if self._stop_flag.is_set():
                            cancelled = True
                            break
                        cmd.on_progress(i, total, "scan")
                        out = _scan_chunk(ch)
                        if out:
                            findings.append(
                                f"[lines {ch.start_line}–{ch.end_line}]\n{out}"
                            )

                    header = [
                        f"{total} section{'s' if total != 1 else ''}",
                        f"focus: {auditmod.focus_label(cmd.focus)}",
                    ]
                    if sampled:
                        header.append("large file — sampled evenly")
                    if truncated_bytes:
                        header.append("truncated to 8 MB")
                    if cancelled:
                        header.append("stopped early")

                    head = f"**Audit of {file_label}** ({', '.join(header)})\n\n"
                    if not findings:
                        # Only call the log clean if we actually finished. A
                        # stop with no findings means the scan was incomplete,
                        # NOT that the log is safe.
                        if cancelled:
                            report = (
                                head + "⚠ Stopped before the audit finished — "
                                "nothing was flagged in the sections scanned so "
                                "far, but this is NOT a complete audit."
                            )
                        else:
                            report = head + auditmod.clean_report(cmd.focus)
                    else:
                        # Findings exist. Summarise them — but NEVER fall back
                        # to the "clean" message here: if the reduce is stopped
                        # or fails, show the RAW findings, otherwise we would
                        # falsely tell the user the log is safe.
                        raw_findings = "\n\n".join(findings)
                        batches = auditmod.batch_findings(findings, reduce_chars)
                        n_reduce = len(batches) + (1 if len(batches) > 1 else 0)
                        if len(batches) == 1:
                            cmd.on_progress(1, n_reduce, "report")
                            body = _reduce_pass(batches[0])
                        else:
                            partials: list[str] = []
                            for bi, b in enumerate(batches, 1):
                                if self._stop_flag.is_set():
                                    break
                                cmd.on_progress(bi, n_reduce, "report")
                                partials.append(_reduce_pass(b))
                            if partials and not self._stop_flag.is_set():
                                cmd.on_progress(n_reduce, n_reduce, "report")
                                body = _reduce_pass(partials)
                            else:
                                body = "\n\n".join(partials)
                        if body and body.strip():
                            report = head + body
                            if cancelled:
                                report += (
                                    "\n\n---\n*Audit stopped early — this "
                                    "summary covers the sections scanned so far.*"
                                )
                        else:
                            note = (
                                "⚠ Stopped before the summary was written — "
                                "raw findings collected so far:\n\n"
                                if cancelled else
                                "⚠ Could not generate a summary — raw "
                                "findings below:\n\n"
                            )
                            report = head + note + raw_findings
                except Exception as e:
                    error = (
                        f"Audit failed: {e}\n\n{traceback.format_exc(limit=2)}"
                    )

                # Rebuild the main conversation so the chat keeps working
                # (identical to the captioning detour's rebuild).
                try:
                    combined_history = (
                        list(self._last_history) + list(self._session_messages)
                    )
                    messages = _build_initial_messages(
                        self._last_system_prompt, combined_history,
                        max_tokens=self._current_max_num_tokens or 4096,
                    )
                    sampler = (
                        litert_lm.SamplerConfig(**self._last_sampler_kwargs)
                        if self._last_sampler_kwargs else None
                    )
                    conv_kwargs: dict[str, Any] = dict(
                        messages=messages, sampler_config=sampler,
                    )
                    if self._last_tools:
                        conv_kwargs["tools"] = self._last_tools
                        conv_kwargs["automatic_tool_calling"] = True
                        if self._last_tool_event_handler is not None:
                            conv_kwargs["tool_event_handler"] = (
                                self._last_tool_event_handler
                            )
                    conversation = engine.create_conversation(**conv_kwargs)
                    conversation.__enter__()
                except Exception as e:
                    log.exception("Failed to rebuild main conversation after audit")
                    self._ready.clear()
                    cmd.cb(None, f"Audit done but chat conv lost: {e}")
                    continue

                # Keep the audit turn in the session so a follow-up question
                # ("explain finding 2") has context and the next captioning
                # rebuild stays consistent.
                if report and not error:
                    self._session_messages.append({
                        "role": "user",
                        "content": cmd.user_text or f"audit {file_label}",
                    })
                    self._session_messages.append({
                        "role": "assistant", "content": report,
                    })
                cmd.cb(report, error)
                continue


# ──── Helpers ────────────────────────────────────────────────────────────────

def _is_gguf_path(path: str) -> bool:
    return path.lower().endswith(".gguf")


_MAX_HISTORY_MSG_CHARS = 20_000  # ≈ 5000 tokens; safe for 4k-context models
_CHARS_PER_TOKEN_EST = 4         # conservative English estimate used for trimming


def _build_initial_messages(
    system_prompt: str,
    history: list[dict],
    max_tokens: int = 4096,
) -> list[dict]:
    out: list[dict] = []
    if system_prompt.strip():
        out.append({
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        })

    history_msgs: list[dict] = []
    for m in history:
        role = m["role"]
        if role not in ("user", "assistant"):
            continue
        sdk_role = "model" if role == "assistant" else role
        # Strip embedded voice path (\x00/path) before sending to SDK.
        content = m["content"].split("\x00", 1)[0]
        # Cap any individual message at ~5000 tokens.
        if len(content) > _MAX_HISTORY_MSG_CHARS:
            content = (
                content[:_MAX_HISTORY_MSG_CHARS]
                + "\n\n[… message truncated to fit context window …]"
            )
        history_msgs.append({
            "role": sdk_role,
            "content": [{"type": "text", "text": content}],
        })

    # Trim oldest turn-pairs until the estimated total fits the window.
    # Reserve 1024 tokens for the current user message, RAG context, and reply.
    # Allow trimming all the way to zero — a long single exchange can exceed
    # the entire context budget (e.g. a large dataset analysis reply).
    sys_tokens = len(system_prompt) // _CHARS_PER_TOKEN_EST
    history_token_budget = max(256, max_tokens - sys_tokens - 1024)
    char_budget = history_token_budget * _CHARS_PER_TOKEN_EST
    while len(history_msgs) >= 2:
        total_chars = sum(len(m["content"][0]["text"]) for m in history_msgs)
        if total_chars <= char_budget:
            break
        history_msgs = history_msgs[2:]  # drop oldest user+assistant pair
        log.debug(
            "History trimmed: dropped oldest turn pair to fit %d-token window",
            max_tokens,
        )

    out.extend(history_msgs)
    return out


def _build_message(user_text: str, attachments: list[dict], litert_lm) -> Any:
    """Build a Contents object when media attachments are present, else plain str."""
    if not attachments:
        return user_text

    parts: list = []
    if user_text.strip():
        parts.append(user_text)

    for att in attachments:
        if att["type"] == "image":
            parts.append(litert_lm.Content.ImageFile(absolute_path=att["path"]))
        elif att["type"] == "audio":
            parts.append(litert_lm.Content.AudioFile(absolute_path=att["path"]))

    if not parts:
        return user_text

    return litert_lm.Contents.of(*parts)
