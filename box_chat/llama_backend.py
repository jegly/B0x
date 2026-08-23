"""GGUF chat backend: llama-server supervisor + OpenAI-style HTTP client.

Runs entirely on EngineManager's worker thread (same threading contract as
the litert path). The server process is one-model-per-process; this module
owns the chat-level state on top of it:

- History is client state — the full conversation is resent every request
  and the server's prompt-prefix cache (``--cache-reuse``) absorbs the
  repeat cost. No slot save/restore, no client reprime.
- Sampling parameters ride each request. Everything that maps to argv
  participates in the load signature and triggers a restart.
- Tools: llama-server native OpenAI tool-calling via ``--jinja`` (the
  agentic loop lives in :meth:`LlamaBackend.send`). No vision in v1.

The HTTP calls go to 127.0.0.1 on a per-session random port with a
per-session bearer token: local IPC with our own child, deliberately
outside net.require_https()'s remote-download policy.
"""
from __future__ import annotations

import http.client
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Callable

from .llama_server import LlamaServer, LlamaServerError, find_server_binary

log = logging.getLogger(__name__)

GGUF_SUFFIXES = (".gguf",)

_THINK_RE = re.compile(r"<(think|thought|thinking)>.*?</\1>\s*", re.DOTALL)

_MAX_HISTORY_MSG_CHARS = 20_000
_CHARS_PER_TOKEN_EST = 4


def is_gguf(path: str) -> bool:
    return Path(path).suffix.lower() in GGUF_SUFFIXES


def _auto_threads() -> int:
    return max(4, (os.cpu_count() or 8) // 2)


def build_server_args(s: Any, with_tools: bool = False) -> list[str]:
    """Translate llama_* Settings into llama-server argv.

    Sentinels ("auto"/""/0/-1 per config.py) mean "omit the flag". With
    ``with_tools`` adds ``--jinja`` (native OpenAI tool-calling; verified
    required on b10001).
    """
    a: list[str] = []
    if with_tools:
        a += ["--jinja"]
    # Memory & context
    if s.llama_ctx_mode == "manual" and s.llama_ctx_size > 0:
        a += ["--ctx-size", str(s.llama_ctx_size)]
    else:
        a += ["--fit", "on", "--fit-target", str(max(0, s.llama_fit_target_mib)),
              "--fit-ctx", str(max(256, s.llama_fit_ctx_min))]
    if s.llama_cache_type_k != "auto":
        a += ["--cache-type-k", s.llama_cache_type_k]
    if s.llama_cache_type_v != "auto":
        a += ["--cache-type-v", s.llama_cache_type_v]
    if s.llama_kv_unified == "on":
        a += ["--kv-unified"]
    elif s.llama_kv_unified == "off":
        a += ["--no-kv-unified"]
    if s.llama_swa_full:
        a += ["--swa-full"]
    if s.llama_keep_tokens:
        a += ["--keep", str(s.llama_keep_tokens)]
    if s.llama_cache_reuse:
        a += ["--cache-reuse", str(s.llama_cache_reuse)]
    if s.llama_cache_ram_mib:
        a += ["--cache-ram", str(s.llama_cache_ram_mib)]
    # Performance
    threads = s.llama_threads if s.llama_threads > 0 else _auto_threads()
    a += ["--threads", str(threads)]
    if s.llama_threads_batch > 0:
        a += ["--threads-batch", str(s.llama_threads_batch)]
    if s.llama_batch_size > 0:
        a += ["--batch-size", str(s.llama_batch_size)]
    if s.llama_ubatch_size > 0:
        a += ["--ubatch-size", str(s.llama_ubatch_size)]
    if s.llama_flash_attn in ("on", "off"):
        a += ["--flash-attn", s.llama_flash_attn]
    if not s.llama_cont_batching:
        a += ["--no-cont-batching"]
    if s.llama_parallel > 0:
        a += ["--parallel", str(s.llama_parallel)]
    if not s.llama_mmap:
        a += ["--no-mmap"]
    if s.llama_mlock:
        a += ["--mlock"]
    # Advanced CPU
    if s.llama_cpu_range:
        a += ["--cpu-range", s.llama_cpu_range]
    if s.llama_cpu_strict:
        a += ["--cpu-strict", "1"]
    if s.llama_priority:
        a += ["--prio", str(s.llama_priority)]
    if s.llama_poll >= 0:
        a += ["--poll", str(s.llama_poll)]
    if s.llama_numa:
        a += ["--numa", s.llama_numa]
    if s.llama_cpu_moe:
        a += ["--cpu-moe"]
    elif s.llama_n_cpu_moe > 0:
        a += ["--n-cpu-moe", str(s.llama_n_cpu_moe)]
    # GPU
    if s.llama_gpu_layers > 0:
        a += ["--gpu-layers", str(s.llama_gpu_layers)]
    # Speculative decoding / MTP
    if s.llama_spec_type and s.llama_spec_type != "none":
        a += ["--spec-type", s.llama_spec_type]
        if s.llama_spec_type.startswith("draft-") and s.llama_draft_model:
            a += ["--spec-draft-model", s.llama_draft_model]
        if s.llama_spec_n_max > 0:
            a += ["--spec-draft-n-max", str(s.llama_spec_n_max)]
        if s.llama_spec_n_min > 0:
            a += ["--spec-draft-n-min", str(s.llama_spec_n_min)]
        if s.llama_draft_cache_type_k != "auto":
            a += ["--cache-type-k-draft", s.llama_draft_cache_type_k]
        if s.llama_draft_cache_type_v != "auto":
            a += ["--cache-type-v-draft", s.llama_draft_cache_type_v]
    # RoPE
    if s.llama_rope_scaling:
        a += ["--rope-scaling", s.llama_rope_scaling]
    if s.llama_rope_scale > 0:
        a += ["--rope-scale", str(s.llama_rope_scale)]
    if s.llama_rope_freq_base > 0:
        a += ["--rope-freq-base", str(s.llama_rope_freq_base)]
    if s.llama_rope_freq_scale > 0:
        a += ["--rope-freq-scale", str(s.llama_rope_freq_scale)]
    return a


def pick_variant(s: Any) -> str:
    """CPU binary whenever GPU layers are 0 — never the Vulkan binary with
    zero layers (Box Android measured that prefill trap)."""
    if s.llama_variant in ("cpu", "vulkan"):
        return s.llama_variant
    if s.llama_gpu_layers > 0:
        try:
            find_server_binary("vulkan")
            return "vulkan"
        except LlamaServerError:
            log.warning("GPU layers requested but no vulkan build bundled; using CPU")
    return "cpu"


def build_sampling(s: Any, temperature, top_k, top_p) -> dict:
    out: dict[str, float | int] = {}
    if temperature is not None:
        out["temperature"] = temperature
    if top_k is not None:
        out["top_k"] = top_k
    if top_p is not None:
        out["top_p"] = top_p
    if s.llama_min_p > 0:
        out["min_p"] = s.llama_min_p
    if s.llama_repeat_penalty > 0:
        out["repeat_penalty"] = s.llama_repeat_penalty
    if s.llama_presence_penalty != 0:
        out["presence_penalty"] = s.llama_presence_penalty
    if s.llama_frequency_penalty != 0:
        out["frequency_penalty"] = s.llama_frequency_penalty
    return out


def build_openai_messages(
    system_prompt: str, history: list[dict], max_tokens: int = 4096,
    strip_reasoning: bool = True,
) -> list[dict]:
    out: list[dict] = []
    if system_prompt.strip():
        out.append({"role": "system", "content": system_prompt})
    msgs: list[dict] = []
    for m in history:
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        content = m["content"].split("\x00", 1)[0]
        if strip_reasoning and role == "assistant":
            content = _THINK_RE.sub("", content)
        if len(content) > _MAX_HISTORY_MSG_CHARS:
            content = (
                content[:_MAX_HISTORY_MSG_CHARS]
                + "\n\n[… message truncated to fit context window …]"
            )
        msgs.append({"role": role, "content": content})
    sys_tokens = len(system_prompt) // _CHARS_PER_TOKEN_EST
    budget_chars = max(256, max_tokens - sys_tokens - 1024) * _CHARS_PER_TOKEN_EST
    while len(msgs) >= 2 and sum(len(m["content"]) for m in msgs) > budget_chars:
        msgs = msgs[2:]
    out.extend(msgs)
    return out


class _CancelShim:
    """Registered as EngineManager's active conversation during a stream —
    cancel_process() closes the socket, which llama-server treats as
    client-gone and aborts generation immediately."""

    def __init__(self, conn: http.client.HTTPConnection) -> None:
        self._conn = conn

    def cancel_process(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


class LlamaBackend:
    """Worker-thread-side GGUF backend. One instance per EngineManager."""

    supports_vision = False
    supports_audit = False

    def __init__(self) -> None:
        self._server: LlamaServer | None = None
        self._signature: tuple | None = None
        self._messages: list[dict] = []
        self._sampling: dict = {}
        self._strip_reasoning = True
        self._ctx_estimate = 4096
        self._tool_runner: Any = None

    def is_loaded(self) -> bool:
        return self._server is not None and self._server.is_running()

    @property
    def has_tools(self) -> bool:
        return self._tool_runner is not None

    @property
    def sandbox_report(self):
        return self._server.sandbox_report if self._server else None

    @property
    def context_estimate(self) -> int:
        return self._ctx_estimate

    def load(
        self, path, system_prompt, history, settings,
        temperature, top_k, top_p, max_num_tokens, tool_runner=None,
    ) -> None:
        """(Re)start the server if argv-relevant config changed, and rebuild
        chat state either way. Raises LlamaServerError on failure."""
        self._tool_runner = tool_runner
        args = build_server_args(settings, with_tools=tool_runner is not None)
        variant = pick_variant(settings)
        signature = (str(Path(path).resolve()), tuple(args), variant)
        if not (self.is_loaded() and signature == self._signature):
            self.unload()
            server = LlamaServer(binary=find_server_binary(variant))
            self._start_with_vocab_heal(server, path, args)
            self._server = server
            self._signature = signature
        self._strip_reasoning = bool(settings.llama_strip_reasoning)
        self._ctx_estimate = (
            settings.llama_ctx_size
            if settings.llama_ctx_mode == "manual"
            else max(settings.llama_fit_ctx_min, max_num_tokens or 4096)
        )
        self._messages = build_openai_messages(
            system_prompt, history, self._ctx_estimate, self._strip_reasoning
        )
        self._sampling = build_sampling(settings, temperature, top_k, top_p)

    def _start_with_vocab_heal(self, server, path, args) -> None:
        """Start the server; if it dies on llama.cpp's token-bijection assert
        (some official GGUFs ship duplicate token strings — e.g. gemma-4 QAT),
        de-duplicate the vocab in place and retry once. A no-op for any other
        failure, and for vocabs that are already fine."""
        try:
            server.start(path, extra_args=args)
            return
        except LlamaServerError as exc:
            if exc.kind != "assert":
                raise
            try:
                from .gguf_vocab_fix import dedup_gguf_vocab
                fixed = dedup_gguf_vocab(path)
            except Exception:  # noqa: BLE001 — never let the repair mask the load error
                log.exception("vocab dedup attempt failed")
                raise exc
            if not fixed:
                raise
            log.warning(
                "llama-server asserted loading %s; deduplicated %d vocab "
                "token(s) and retrying", Path(path).name, fixed,
            )
        server.start(path, extra_args=args)

    def unload(self) -> None:
        if self._server is not None:
            try:
                self._server.stop()
            except Exception:  # noqa: BLE001
                log.exception("error stopping llama-server")
        self._server = None
        self._signature = None
        self._messages = []

    # ── stateless passes (file audit) ────────────────────────────────────
    def audit_pass(self, system_text: str, user_text: str, max_out_chars: int = 8000) -> str:
        """One stateless completion for the map-reduce audit — no chat
        history touched. Raises LlamaServerError so the orchestrator can
        classify overflow/errors."""
        server = self._server
        if server is None or not server.is_running():
            raise (server.classify_exit() if server
                   else LlamaServerError("not_running", "no GGUF model loaded"))
        body = json.dumps({
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "max_tokens": max(64, max_out_chars // 3),
            **self._sampling,
        })
        port = int(server.base_url.rsplit(":", 1)[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        try:
            conn.request(
                "POST", "/v1/chat/completions", body=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {server.api_key}",
                },
            )
            resp = conn.getresponse()
            payload = resp.read()
            if resp.status != 200:
                detail = payload.decode("utf-8", "replace")
                raise LlamaServerError(
                    "request_failed",
                    f"llama-server HTTP {resp.status}: {detail}", server.log_tail(),
                )
            data = json.loads(payload)
            return (data["choices"][0]["message"].get("content") or "").strip()
        finally:
            conn.close()

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self._messages.append({"role": "user", "content": user_text})
        self._messages.append({"role": "assistant", "content": assistant_text})

    # ── chat ────────────────────────────────────────────────────────────
    def send(
        self, user_text: str, on_token: Callable[[str], None],
        stop_flag: threading.Event, register_active: Callable[[Any], None],
    ) -> tuple[str, bool]:
        """Stream one turn, running the tool loop if a runner is loaded.
        Returns (assistant_text, completed)."""
        server = self._server
        if server is None or not server.is_running():
            raise (server.classify_exit() if server
                   else LlamaServerError("not_running", "no GGUF model loaded"))

        if self._tool_runner is not None:
            self._tool_runner.reset()

        convo = self._messages + [{"role": "user", "content": user_text}]
        final_text = ""

        for _round in range(64):
            text, tool_calls, completed = self._stream_once(
                server, convo, on_token, stop_flag, register_active
            )
            if stop_flag.is_set() or not completed:
                return final_text + text, False
            if tool_calls and self._tool_runner is not None:
                convo.append({
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": tc["id"], "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        }
                        for tc in tool_calls
                    ],
                })
                for tc in tool_calls:
                    result = self._tool_runner.run_call(tc["name"], tc["arguments"])
                    convo.append({
                        "role": "tool", "tool_call_id": tc["id"], "content": result,
                    })
                    if stop_flag.is_set():
                        return final_text, False
                continue
            final_text += text
            break

        self._messages = convo
        assistant = _THINK_RE.sub("", final_text) if self._strip_reasoning else final_text
        self._messages.append({"role": "assistant", "content": assistant})
        return final_text, True

    def _stream_once(
        self, server, messages, on_token, stop_flag, register_active,
    ) -> tuple[str, list[dict], bool]:
        body_obj: dict[str, Any] = {
            "messages": messages, "stream": True, **self._sampling,
        }
        if self._tool_runner is not None:
            body_obj["tools"] = self._tool_runner.schemas
        body = json.dumps(body_obj)
        port = int(server.base_url.rsplit(":", 1)[1])
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        shim = _CancelShim(conn)
        register_active(shim)
        parts: list[str] = []
        tc_acc: dict[int, dict[str, Any]] = {}
        completed = False
        try:
            conn.request(
                "POST", "/v1/chat/completions", body=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {server.api_key}",
                },
            )
            resp = conn.getresponse()
            if resp.status != 200:
                detail = resp.read(2000).decode("utf-8", "replace")
                raise LlamaServerError(
                    "request_failed",
                    f"llama-server HTTP {resp.status}: {detail}", server.log_tail(),
                )
            buffer = b""
            while not stop_flag.is_set():
                chunk = resp.read1(65536)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == b"[DONE]":
                        completed = True
                        break
                    try:
                        delta = json.loads(payload)
                    except ValueError:
                        continue
                    choices = delta.get("choices") or []
                    if not choices:
                        continue
                    d = choices[0].get("delta") or {}
                    text = d.get("content")
                    if text:
                        parts.append(text)
                        on_token(text)
                    for tc in d.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        slot = tc_acc.setdefault(
                            idx, {"id": "", "name": "", "arg_parts": []}
                        )
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["arg_parts"].append(fn["arguments"])
                    if choices[0].get("finish_reason"):
                        completed = True
                if completed:
                    break
        except (OSError, http.client.HTTPException, AttributeError, ValueError) as exc:
            # AttributeError/ValueError here are http.client's closed-response
            # symptoms ("'NoneType' has no attribute 'read1'") — the cancel
            # shim closes the connection from the UI's stop path mid-read.
            # Only a set stop_flag makes that a clean stop.
            if stop_flag.is_set():
                pass
            elif not server.is_running():
                raise server.classify_exit() from exc
            else:
                raise LlamaServerError(
                    "request_failed", f"stream failed: {exc}", server.log_tail()
                ) from exc
        finally:
            register_active(None)
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

        tool_calls = [
            {
                "id": v["id"] or f"call_{i}",
                "name": v["name"],
                "arguments": "".join(v["arg_parts"]),
            }
            for i, v in sorted(tc_acc.items())
            if v["name"]
        ]
        return "".join(parts), tool_calls, completed
