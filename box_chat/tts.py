"""Piper TTS wrapper — multi-voice download, synthesis, playback."""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tarfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

# Sentence boundary: after .!? followed by optional markdown chars then newline,
# or after .!? followed by whitespace, or double newline.
# The [*_`\s]* handles markdown endings like ?** before a newline.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])[*_`\s]*\n|\n{2,}|(?<=[.!?])\s+')

# Markdown → plain text for speech synthesis.
_MD_PATTERNS = [
    # Math first — display and inline LaTeX blocks are dropped entirely
    # rather than spoken. The user already SEES the rendered equation; the
    # raw "$$ \\oint \\partial S …" stream would otherwise be pronounced as
    # "dollar dollar slash slash oint slash slash partial S", which is the
    # bug Jegly hit.
    (re.compile(r'\$\$.+?\$\$',     re.DOTALL),       ' '),   # $$ display $$
    (re.compile(r'\\\[.+?\\\]',     re.DOTALL),       ' '),   # \[ display \]
    (re.compile(r'\$(?!\$)[^$\n]+?\$'),               ' '),   # $ inline $
    (re.compile(r'\\\(.+?\\\)',     re.DOTALL),       ' '),   # \( inline \)
    # Markdown formatting.
    (re.compile(r'\*{2,3}(.*?)\*{2,3}', re.DOTALL), r'\1'),  # **bold** / ***bold***
    (re.compile(r'\*(.*?)\*',            re.DOTALL), r'\1'),  # *italic*
    (re.compile(r'_(.*?)_',              re.DOTALL), r'\1'),  # _italic_
    (re.compile(r'`+([^`]*)`+'),                     r'\1'),  # `code`
    (re.compile(r'^\s*#{1,6}\s+',   re.MULTILINE),  ''),     # ## headers
    (re.compile(r'^\s*[*\-+]\s+',   re.MULTILINE),  ''),     # bullet points
    (re.compile(r'^\s*\d+\.\s+',    re.MULTILINE),  ''),     # numbered lists
    (re.compile(r'\[([^\]]+)\]\([^\)]*\)'),          r'\1'),  # [text](url)
    # Stray LaTeX outside math delimiters (model emitted raw \command
    # without wrapping). Drop the \command and braces; the prose around
    # it usually carries the meaning.
    (re.compile(r'\\\\'),                            ''),     # \\ → ''
    (re.compile(r'\\[a-zA-Z]+\b'),                   ''),     # \alpha, \oint …
    (re.compile(r'[{}]'),                            ''),     # stray braces
    (re.compile(r'\s+'),                             ' '),    # collapse whitespace
]


def _clean_for_speech(text: str) -> str:
    for pat, repl in _MD_PATTERNS:
        text = pat.sub(repl, text)
    return text.strip()

from .config import DATA_DIR
from .net import require_https

PIPER_DIR = DATA_DIR / "piper"
PIPER_BIN = PIPER_DIR / "piper"

# Curated voice catalog — voice_id → {display, onnx_url, json_url}
VOICES: dict[str, dict] = {
    "en_US-lessac-medium": {
        "display": "Lessac · US · Female",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    },
    "en_US-ryan-high": {
        "display": "Ryan · US · Male",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/ryan/high/en_US-ryan-high.onnx.json",
    },
    "en_US-amy-medium": {
        "display": "Amy · US · Female",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    },
    "en_US-joe-medium": {
        "display": "Joe · US · Male",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/medium/en_US-joe-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/joe/medium/en_US-joe-medium.onnx.json",
    },
    "en_GB-alan-medium": {
        "display": "Alan · GB · Male",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    },
    "en_GB-jenny_dioco-medium": {
        "display": "Jenny · GB · Female",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium.onnx.json",
    },
}

DEFAULT_VOICE = "en_US-lessac-medium"

_PIPER_TAR_URL = (
    "https://github.com/rhasspy/piper/releases/download/2023.11.14-2/"
    "piper_linux_x86_64.tar.gz"
)


def get_voice_paths(voice_id: str) -> tuple[Path, Path]:
    return PIPER_DIR / f"{voice_id}.onnx", PIPER_DIR / f"{voice_id}.onnx.json"


def is_ready(voice_id: str = DEFAULT_VOICE) -> bool:
    onnx, cfg = get_voice_paths(voice_id)
    return PIPER_BIN.is_file() and onnx.is_file() and cfg.is_file()


class PiperTTS:
    """Thread-safe Piper TTS — one-shot and sentence-streaming modes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()

        # One-shot mode
        self._thread: threading.Thread | None = None

        # Streaming mode
        self._stream_queue: queue.Queue | None = None
        self._stream_thread: threading.Thread | None = None
        self._stream_buf: str = ""
        self._stream_voice: str = DEFAULT_VOICE
        self._stream_on_done: Callable[[], None] | None = None

        # Playback gain provider — called at synth time so the user's slider
        # changes take effect on the next sentence without restarting TTS.
        self._gain_provider: Callable[[], float] = lambda: 1.0

    def set_gain_provider(self, fn: Callable[[], float]) -> None:
        """Inject a callable that returns the desired playback gain (float).
        Called once per played sentence — adjust the slider mid-stream and
        the next sentence picks up the new level."""
        self._gain_provider = fn

    # ── One-shot speak ─────────────────────────────────────────────────────

    def speak(
        self,
        text: str,
        voice_id: str = DEFAULT_VOICE,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        self.stop()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._do_speak, args=(text, voice_id, on_done), daemon=True
        )
        self._thread.start()

    def _do_speak(
        self,
        text: str,
        voice_id: str,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        onnx_path, json_path = get_voice_paths(voice_id)
        if not (PIPER_BIN.is_file() and onnx_path.is_file() and json_path.is_file()):
            if on_done:
                on_done()
            return
        try:
            self._synthesize_and_play(text, voice_id)
        except Exception:
            with self._lock:
                self._proc = None
        finally:
            if on_done:
                on_done()

    # ── Streaming speak ────────────────────────────────────────────────────

    def start_stream(
        self,
        voice_id: str = DEFAULT_VOICE,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        """Begin streaming mode. Sentences queued via push_chunk() are spoken in order."""
        self.stop()
        self._stop.clear()
        self._stream_buf = ""
        self._stream_voice = voice_id
        self._stream_on_done = on_done
        self._stream_queue = queue.Queue()
        self._stream_thread = threading.Thread(
            target=self._stream_worker, daemon=True
        )
        self._stream_thread.start()

    def push_chunk(self, text: str) -> None:
        """Add a token to the streaming buffer; complete sentences are queued immediately."""
        if self._stream_queue is None or self._stop.is_set():
            return
        self._stream_buf += text
        while True:
            m = _SENT_BOUNDARY.search(self._stream_buf)
            if not m:
                break
            sentence = self._stream_buf[: m.end()].strip()
            self._stream_buf = self._stream_buf[m.end():]
            if sentence:
                self._stream_queue.put(sentence)

    def finish_stream(self) -> None:
        """Flush remaining buffered text and signal end of stream."""
        if self._stream_queue is None:
            return
        remaining = self._stream_buf.strip()
        self._stream_buf = ""
        if remaining:
            self._stream_queue.put(remaining)
        self._stream_queue.put(None)  # sentinel

    def _stream_worker(self) -> None:
        while True:
            item = self._stream_queue.get()
            if item is None or self._stop.is_set():
                break
            try:
                self._synthesize_and_play(item, self._stream_voice)
            except Exception:
                with self._lock:
                    self._proc = None
        if not self._stop.is_set() and self._stream_on_done:
            self._stream_on_done()

    # ── Shared core ────────────────────────────────────────────────────────

    def _synthesize_and_play(self, text: str, voice_id: str) -> None:
        text = _clean_for_speech(text)
        if not text:
            return
        onnx_path, json_path = get_voice_paths(voice_id)
        if not (PIPER_BIN.is_file() and onnx_path.is_file() and json_path.is_file()):
            return
        with open(json_path) as f:
            cfg = json.load(f)
        sample_rate = cfg.get("audio", {}).get("sample_rate", 22050)

        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = str(PIPER_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")

        with self._lock:
            if self._stop.is_set():
                return
            self._proc = subprocess.Popen(
                [str(PIPER_BIN), "--model", str(onnx_path), "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
            )
        raw, _ = self._proc.communicate(input=text.encode(), timeout=60)
        with self._lock:
            self._proc = None

        if self._stop.is_set() or not raw:
            return

        import numpy as np
        import sounddevice as sd

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        # Apply user gain. Clip to [-1, 1] so a >1.0 boost doesn't fold/distort.
        try:
            gain = float(self._gain_provider())
        except Exception:
            gain = 1.0
        if gain <= 0.0:
            return  # muted — skip playback entirely
        if gain != 1.0:
            audio = np.clip(audio * gain, -1.0, 1.0)
        sd.play(audio, samplerate=sample_rate, blocking=False)
        while sd.get_stream().active:
            if self._stop.is_set():
                sd.stop()
                return
            time.sleep(0.05)

    # ── Stop ───────────────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop.set()
        # Drain stream queue and unblock worker
        if self._stream_queue is not None:
            while True:
                try:
                    self._stream_queue.get_nowait()
                except queue.Empty:
                    break
            self._stream_queue.put(None)
        # Kill piper process
        with self._lock:
            if self._proc is not None:
                try:
                    self._proc.kill()
                except Exception:
                    pass
                self._proc = None
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        if self._stream_thread and self._stream_thread.is_alive():
            self._stream_thread.join(timeout=2)
        self._stream_queue = None
        self._stream_thread = None
        self._thread = None


def download(
    voice_id: str,
    on_progress: Callable[[str, int, int], None],
    on_done: Callable[[], None],
    on_error: Callable[[str], None],
) -> None:
    threading.Thread(
        target=_do_download,
        args=(voice_id, on_progress, on_done, on_error),
        daemon=True,
    ).start()


def _do_download(
    voice_id: str,
    on_progress: Callable[[str, int, int], None],
    on_done: Callable[[], None],
    on_error: Callable[[str], None],
) -> None:
    PIPER_DIR.mkdir(parents=True, exist_ok=True)
    voice = VOICES.get(voice_id)
    if voice is None:
        on_error(f"Unknown voice: {voice_id}")
        return
    try:
        if not PIPER_BIN.is_file():
            tar_tmp = PIPER_DIR / "piper.tar.gz.tmp"
            _dl_with_progress(_PIPER_TAR_URL, tar_tmp, "Piper binary", on_progress)

            on_progress("Extracting piper…", 0, 0)
            with tarfile.open(tar_tmp) as tf:
                for member in tf.getmembers():
                    parts = Path(member.name).parts
                    if len(parts) < 2:
                        continue
                    member.name = str(Path(*parts[1:]))
                    tf.extract(member, path=PIPER_DIR)
            tar_tmp.unlink(missing_ok=True)
            PIPER_BIN.chmod(PIPER_BIN.stat().st_mode | 0o111)

        onnx_path, json_path = get_voice_paths(voice_id)
        _dl_with_progress(voice["json_url"], json_path, "Voice config", on_progress)
        _dl_with_progress(voice["onnx_url"], onnx_path, "Voice model", on_progress)

        on_done()
    except Exception as e:
        on_error(str(e))


def _dl_with_progress(
    url: str,
    dest: Path,
    label: str,
    on_progress: Callable[[str, int, int], None],
) -> None:
    require_https(url)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    resume = tmp.stat().st_size if tmp.exists() else 0

    headers: dict[str, str] = {"Accept-Encoding": "identity"}
    if resume:
        headers["Range"] = f"bytes={resume}-"

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        mode = "ab" if resp.status == 206 else "wb"
        if mode == "wb":
            resume = 0
        content_length = int(resp.headers.get("Content-Length", 0))
        total = resume + content_length if content_length else 0
        done = resume

        last_update = 0.0
        with open(tmp, mode) as f:
            while True:
                chunk = resp.read(4 * 1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last_update >= 0.25:
                    on_progress(label, done, total)
                    last_update = now

    tmp.rename(dest)
