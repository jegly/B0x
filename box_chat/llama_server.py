"""llama-server subprocess supervisor.

Engine-tier, pure Python (no gi). Owns the lifecycle of one sandboxed
`llama-server` process serving one GGUF: spawn → wait ready → serve →
SIGTERM → grace → SIGKILL. One model per process by design.

Security invariants (do not loosen):
- ``--host 127.0.0.1`` is hardcoded, never configurable.
- The API key travels via the LLAMA_API_KEY environment variable (or a
  0600 EnvironmentFile on the systemd path), never argv.
- The web UI is always disabled (``--no-webui``).
- Model paths are canonicalized and handed to the sandbox policy.

Crash handling: a reader thread drains merged stdout/stderr into a ring
buffer and pattern-matches known fatal signatures. No automatic respawn.
"""
from __future__ import annotations

import collections
import http.client
import json
import os
import random
import re
import threading
import time
from pathlib import Path

from .sandbox import LaunchedProcess, Policy, SandboxReport, launch

__all__ = ["LlamaServer", "LlamaServerError", "find_server_binary"]

_EPHEMERAL = (49152, 65535)
_LOG_RING = 400
_BIND_RETRIES = 3

_FATAL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("oom", r"out of memory|failed to allocate|cannot allocate memory"),
    ("corruption", r"double free|heap corruption|stack smashing|buffer overflow"),
    ("assert", r"ggml_assert|assertion.*failed"),
    ("bad_model", r"failed to load model|invalid magic|unknown model architecture"),
)


class LlamaServerError(Exception):
    """Supervisor-level failure with a machine-usable ``kind``.

    kinds: spawn_failed | not_ready | bind_failed | oom | corruption |
    assert | bad_model | crash | not_running | request_failed
    """

    def __init__(self, kind: str, message: str, log_tail: str = "") -> None:
        super().__init__(message)
        self.kind = kind
        self.log_tail = log_tail


def find_server_binary(variant: str = "cpu") -> Path:
    """Locate the bundled llama-server for a variant ("cpu" | "vulkan")."""
    candidates: list[Path] = []
    if env_dir := os.environ.get("BOX_LLAMA_SERVER_DIR"):
        candidates.append(Path(env_dir))
    suffix = "" if variant == "cpu" else f"-{variant}"
    candidates.append(Path(f"/opt/box/libexec/llama.cpp{suffix}"))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "vendor" / f"llama.cpp{suffix}")
    if variant == "cpu":
        candidates.append(repo_root / "vendor" / "llama.cpp")
    for d in candidates:
        binary = d / "llama-server"
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
    raise LlamaServerError(
        "spawn_failed",
        f"no llama-server binary found for variant {variant!r} "
        f"(searched: {', '.join(str(c) for c in candidates)})",
    )


class LlamaServer:
    """Supervise one sandboxed llama-server process serving one GGUF."""

    def __init__(self, binary: Path | None = None) -> None:
        self._binary = binary
        self._proc: LaunchedProcess | None = None
        self._port = 0
        self._api_key = ""
        self._log: collections.deque[str] = collections.deque(maxlen=_LOG_RING)
        self._log_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._model_path: Path | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def sandbox_report(self) -> SandboxReport | None:
        return self._proc.report if self._proc else None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.popen.poll() is None

    def log_tail(self, n: int = 40) -> str:
        with self._log_lock:
            return "\n".join(list(self._log)[-n:])

    def start(
        self,
        model_path: str | Path,
        extra_args: list[str] | None = None,
        ready_timeout: float = 600.0,
    ) -> None:
        if self.is_running():
            raise LlamaServerError("spawn_failed", "server already running")

        model = Path(model_path).resolve()
        if not model.is_file():
            raise LlamaServerError("bad_model", f"model file not found: {model}")
        binary = self._binary or find_server_binary()
        bin_dir = binary.parent
        self._model_path = model
        self._api_key = os.urandom(24).hex()

        last_error: LlamaServerError | None = None
        for _attempt in range(_BIND_RETRIES):
            port = random.randint(*_EPHEMERAL)
            argv = [
                str(binary), "-m", str(model),
                "--host", "127.0.0.1", "--port", str(port), "--no-webui",
                *(extra_args or []),
            ]
            policy = Policy.for_local_server(
                exec_dir=str(bin_dir), model_files=(str(model),), port=port,
            )
            with self._log_lock:
                self._log.clear()
            try:
                proc = launch(
                    argv, policy,
                    env={"LD_LIBRARY_PATH": str(bin_dir)},
                    secret_env={"LLAMA_API_KEY": self._api_key},
                )
            except Exception as exc:  # noqa: BLE001
                raise LlamaServerError("spawn_failed", str(exc)) from exc
            self._proc = proc
            self._port = port
            self._start_reader(proc)
            try:
                self._wait_ready(ready_timeout)
                proc.cleanup_env_file()
                return
            except LlamaServerError as exc:
                proc.cleanup_env_file()
                self.stop()
                if exc.kind != "bind_failed":
                    raise
                last_error = exc
        assert last_error is not None
        raise last_error

    def stop(self, grace: float = 5.0) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.popen.poll() is None:
            proc.terminate()
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and proc.popen.poll() is None:
                time.sleep(0.05)
            if proc.popen.poll() is None:
                proc.kill()
                proc.popen.wait(timeout=10)
        proc.cleanup_env_file()
        if self._reader is not None:
            self._reader.join(timeout=2)
            self._reader = None
        self._proc = None
        self._port = 0

    def check_health(self) -> bool:
        if not self.is_running():
            return False
        try:
            return self._http_health() == 200
        except OSError:
            return False

    def classify_exit(self) -> LlamaServerError:
        tail = self.log_tail()
        low = tail.lower()
        for kind, pattern in _FATAL_PATTERNS:
            if re.search(pattern, low):
                return LlamaServerError(kind, f"llama-server died ({kind})", tail)
        return LlamaServerError("crash", "llama-server exited unexpectedly", tail)

    def _start_reader(self, proc: LaunchedProcess) -> None:
        def drain() -> None:
            stream = proc.popen.stdout
            if stream is None:
                return
            for raw in stream:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                with self._log_lock:
                    self._log.append(line)

        self._reader = threading.Thread(target=drain, daemon=True)
        self._reader.start()

    def _http_health(self) -> int:
        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=5)
        try:
            conn.request("GET", "/health")
            return conn.getresponse().status
        finally:
            conn.close()

    def _wait_ready(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            code = self._proc.popen.poll() if self._proc else 0
            if code is not None:
                tail = self.log_tail()
                if re.search(
                    r"couldn't bind|address already in use", tail, re.IGNORECASE
                ):
                    raise LlamaServerError(
                        "bind_failed", f"port {self._port} unavailable", tail
                    )
                raise self.classify_exit()
            try:
                if self._http_health() == 200:
                    return
            except OSError:
                pass
            time.sleep(0.15)
        raise LlamaServerError(
            "not_ready", f"server not ready after {timeout:.0f}s", self.log_tail()
        )

    def quick_completion(self, prompt: str, n_predict: int = 8) -> str:
        if not self.is_running():
            raise LlamaServerError("not_running", "server is not running")
        body = json.dumps({
            "model": "box",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict,
            "stream": False,
        })
        conn = http.client.HTTPConnection("127.0.0.1", self._port, timeout=120)
        try:
            conn.request(
                "POST", "/v1/chat/completions", body=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
            )
            resp = conn.getresponse()
            data = json.loads(resp.read())
            if resp.status != 200:
                raise LlamaServerError(
                    "crash", f"completion HTTP {resp.status}: {data}", self.log_tail()
                )
            return data["choices"][0]["message"]["content"]
        finally:
            conn.close()
