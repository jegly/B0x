"""Model catalog + reusable download widgets, shown in Preferences → Models.

To add a new downloadable model: append a dict to MODELS below. That's the
whole job — the Models page picks it up automatically. Include "sha256" and
"size" from the Hugging Face API so the download is verified — never guess
them:

    curl -s -X POST https://huggingface.co/api/models/<repo>/paths-info/main \\
        -H 'Content-Type: application/json' -d '{"paths": ["<filename>"]}'

→ ``lfs.oid`` is the SHA-256, ``lfs.size`` the byte count. HTTPS protects
the transfer; the hash is what protects against the *source* being swapped
out later — several catalog entries are community uploads, so this is the
trust anchor, not a formality.

Two formats are tracked, both runnable:
- "litertlm": the litert-lm chat engine, in-process.
- "gguf": the llama.cpp engine — a sandboxed llama-server subprocess
  (llama_backend.py / llama_server.py); tunable on the Preferences →
  Llama.cpp page.
"""
from __future__ import annotations

import hashlib
import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from .config import MODELS_DIR
from .net import require_https

FORMAT_LABELS = {
    "litertlm": "LiteRT-LM",
    "gguf": "GGUF · llama.cpp",
    "sd-gguf": "GGUF · stable-diffusion.cpp",
}

MODELS: list[dict] = [
    {
        "name": "Gemma 4 E2B",
        "subtitle": "2.59 GB · 2B effective params · faster",
        "filename": "gemma-4-E2B-it.litertlm",
        "sha256": "181938105e0eefd105961417e8da75903eacda102c4fce9ce90f50b97139a63c",
        "size": 2588147712,
        "format": "litertlm",
        "url": (
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm"
            "/resolve/main/gemma-4-E2B-it.litertlm?download=true"
        ),
    },
    {
        "name": "Gemma 4 E4B",
        "subtitle": "3.66 GB · 4B effective params · higher quality",
        "filename": "gemma-4-E4B-it.litertlm",
        "sha256": "0b2a8980ce155fd97673d8e820b4d29d9c7d99b8fa6806f425d969b145bd52e0",
        "size": 3659530240,
        "format": "litertlm",
        "url": (
            "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm"
            "/resolve/main/gemma-4-E4B-it.litertlm?download=true"
        ),
    },
    {
        "name": "Gemma 4 26B-A4B",
        "subtitle": "14.70 GB · 26B MoE, 4B active · needs ~20 GB free RAM",
        "filename": "gemma-4-26B-A4B-it-web.litertlm",
        "sha256": "4523b2de695a7f22dc675b716253ba9a8512ca9f4852fb6a682fe4c2eb859c16",
        "size": 15786524672,
        "format": "litertlm",
        "url": (
            "https://huggingface.co/litert-community/gemma-4-26B-A4B-it-litert-lm"
            "/resolve/main/gemma-4-26B-A4B-it-web.litertlm?download=true"
        ),
    },
    {
        "name": "Gemma 4 31B",
        "subtitle": "17.90 GB · 31B dense · needs ~24 GB free RAM",
        "filename": "gemma-4-31B-it-web.litertlm",
        "sha256": "5131b21f70265632e7545abe21dc6b52ff6a690779033eef48d368957374035b",
        "size": 19217989632,
        "format": "litertlm",
        "url": (
            "https://huggingface.co/litert-community/gemma-4-31B-it-litert-lm"
            "/resolve/main/gemma-4-31B-it-web.litertlm?download=true"
        ),
    },
    # Official Google Gemma 4 QAT GGUFs — quantization-aware-trained q4_0,
    # for the llama.cpp engine (the .litertlm variants above are the LiteRT
    # ones). These hit the duplicate-token vocab assert on load; the backend
    # self-heal (LlamaBackend._start_with_vocab_heal / gguf_vocab_fix) dedups
    # and retries automatically. sha256 is the PRISTINE HF file — the heal
    # runs after download, at load time. mmproj (vision) files live in the
    # same repos but the catalog is single-file per entry, so text-only here.
    {
        "name": "Gemma 4 E2B (QAT GGUF)",
        "subtitle": "3.35 GB · 2B effective params · GGUF for llama.cpp",
        "filename": "gemma-4-E2B_q4_0-it.gguf",
        "sha256": "25194efbf8a53268241e5ffa6d5490edc08b3faaa6ead24478c8b025a986d556",
        "size": 3349515840,
        "format": "gguf",
        "url": (
            "https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf"
            "/resolve/main/gemma-4-E2B_q4_0-it.gguf?download=true"
        ),
    },
    {
        "name": "Gemma 4 E4B (QAT GGUF)",
        "subtitle": "5.15 GB · 4B effective params · GGUF for llama.cpp",
        "filename": "gemma-4-E4B_q4_0-it.gguf",
        "sha256": "09f6f2a1d9ff4a1b7db9cc1aad9c55a9df2f5ec133327a92eab593fcf4360ed0",
        "size": 5154940864,
        "format": "gguf",
        "url": (
            "https://huggingface.co/google/gemma-4-E4B-it-qat-q4_0-gguf"
            "/resolve/main/gemma-4-E4B_q4_0-it.gguf?download=true"
        ),
    },
    {
        "name": "Gemma 4 12B (QAT GGUF)",
        "subtitle": "6.98 GB · 12B dense · GGUF for llama.cpp",
        "filename": "gemma-4-12b-it-qat-q4_0.gguf",
        "sha256": "1e76e46623deaa4db97d4ef272ceab0dfb767c0f34c2c76524837edf2b57a510",
        "size": 6975878912,
        "format": "gguf",
        "url": (
            "https://huggingface.co/google/gemma-4-12B-it-qat-q4_0-gguf"
            "/resolve/main/gemma-4-12b-it-qat-q4_0.gguf?download=true"
        ),
    },
    {
        "name": "Gemma 4 26B-A4B (QAT GGUF)",
        "subtitle": "14.44 GB · 26B MoE, 4B active · GGUF · ~20 GB RAM",
        "filename": "gemma-4-26B_q4_0-it.gguf",
        "sha256": "17e6bd3ade3c6bfc57165bf6790a5d27166961d5d663fdb20decd0bd357883c8",
        "size": 14439363168,
        "format": "gguf",
        "url": (
            "https://huggingface.co/google/gemma-4-26B-A4B-it-qat-q4_0-gguf"
            "/resolve/main/gemma-4-26B_q4_0-it.gguf?download=true"
        ),
    },
    {
        "name": "Gemma 4 31B (QAT GGUF)",
        "subtitle": "17.65 GB · 31B dense · GGUF · ~24 GB RAM",
        "filename": "gemma-4-31B_q4_0-it.gguf",
        "sha256": "561b08eeb83b5fa9fc2d39e3330270cf1d94979fb82af4a8822a76367d471564",
        "size": 17651001184,
        "format": "gguf",
        "url": (
            "https://huggingface.co/google/gemma-4-31B-it-qat-q4_0-gguf"
            "/resolve/main/gemma-4-31B_q4_0-it.gguf?download=true"
        ),
    },
    # Community GGUF finetunes/merges — not from Google. Run on the
    # llama.cpp engine (checksums above are the trust anchor for these).
    {
        "name": "Qwythos 9B (Claude-Mythos merge)",
        "subtitle": "5.63 GB · 9B · Q4_K_M",
        "filename": "Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf",
        "sha256": "5c09e7f207d2fd9069c802ff77632fa7cdaffe7fb5ba40ff9f060aaeaa09acd5",
        "size": 5629108896,
        "format": "gguf",
        "url": (
            "https://huggingface.co/empero-ai/Qwythos-9B-Claude-Mythos-5-1M-GGUF"
            "/resolve/main/Qwythos-9B-Claude-Mythos-5-1M-Q4_K_M.gguf?download=true"
        ),
    },
    {
        # "Abliterated" = safety/refusal training stripped out. Flagged plainly.
        "name": "Qwythos 9B (abliterated, uncensored)",
        "subtitle": "5.38 GB · 9B · 1M context · Q4_K, refusal training removed",
        "filename": "Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-Q4_K.gguf",
        "sha256": "11d81ce53c9f1003e9332652d7be00b7f343bbb4345aafeba450ec1beb2d6dd5",
        "size": 5780091040,
        "format": "gguf",
        "url": (
            "https://huggingface.co/huihui-ai/"
            "Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-GGUF"
            "/resolve/main/Huihui-Qwythos-9B-Claude-Mythos-5-1M-abliterated-Q4_K.gguf"
            "?download=true"
        ),
    },
    {
        "name": "Gemma 4 12B Coder (Fable5 Composer 2.5)",
        "subtitle": "6.87 GB · 12B · coding-focused finetune · Q4_K_M",
        "filename": "gemma4-coding-Q4_K_M.gguf",
        "sha256": "1fe90b72e105d7bc71650aa59883edece3e84751af489075217a7ae717b1fe8d",
        "size": 7381381664,
        "format": "gguf",
        "url": (
            "https://huggingface.co/yuxinlu1/"
            "gemma-4-12B-coder-fable5-composer2.5-v1-GGUF"
            "/resolve/main/gemma4-coding-Q4_K_M.gguf?download=true"
        ),
    },
    {
        "name": "Qwythos 9B v2",
        "subtitle": "5.34 GB · 9B · Q4_K_M",
        "filename": "Qwythos-9B-v2-Q4_K_M.gguf",
        "sha256": "c0a588704f422b713eca29b2c1f192ae6f69aea3f9e7cb64f9ecdb76ff7a85f4",
        "size": 5736063744,
        "format": "gguf",
        "url": (
            "https://huggingface.co/empero-ai/Qwythos-9B-v2-GGUF"
            "/resolve/main/Qwythos-9B-v2-Q4_K_M.gguf?download=true"
        ),
    },
    # ── 2026 Qwen local coders (lmstudio-community GGUFs, Q4_K_M, single-file).
    # MoE variants (…-A3B) activate ~3B params/token → fast on CPU despite big
    # total size; the dense 27B has top benchmarks but runs all 27B/token.
    {
        "name": "Qwen3-Coder 30B-A3B (coder)",
        "subtitle": "18.63 GB · 30B MoE, 3B active · dedicated coder · needs ~24 GB RAM",
        "filename": "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        "sha256": "79ad15a5ee3caddc3f4ff0db33a14454a5a3eb503d7fa1c1e35feafc579de486",
        "size": 18632186176,
        "format": "gguf",
        "url": (
            "https://huggingface.co/lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF"
            "/resolve/main/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf?download=true"
        ),
    },
    {
        "name": "Qwen3.6 35B-A3B (MoE)",
        "subtitle": "21.17 GB · 35B MoE, 3B active · fast on CPU · needs ~27 GB RAM",
        "filename": "Qwen3.6-35B-A3B-Q4_K_M.gguf",
        "sha256": "4ac6a06bce551257267f49ad2226f8671a22519ccc1a4dde9d5b433d1f2a410d",
        "size": 21166757728,
        "format": "gguf",
        "url": (
            "https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF"
            "/resolve/main/Qwen3.6-35B-A3B-Q4_K_M.gguf?download=true"
        ),
    },
    {
        # unsloth UD (Dynamic) quant — keeps sensitive layers at higher precision,
        # so noticeably better quality than a plain Q4_K_M at ~the same size.
        "name": "Qwen3.6 27B (dense)",
        "subtitle": "17.61 GB · 27B dense · top quality, slower on CPU · unsloth UD-Q4_K_XL · needs ~21 GB RAM",
        "filename": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
        "sha256": "ff6941ded525b34eb159496762c29dd0ec6e71dc31b74d57e75d871a03eec259",
        "size": 17612564704,
        "format": "gguf",
        "url": (
            "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF"
            "/resolve/main/Qwen3.6-27B-UD-Q4_K_XL.gguf?download=true"
        ),
    },
    # Add new models here — name/subtitle/filename/sha256/size/format/url
    # (sha256+size from the HF paths-info API, see module docstring).
]


# ── Stable Diffusion (stable-diffusion.cpp) image models ────────────────────
# Same single-file download+verify path as MODELS, but "used" by registering
# the file as an SD model (settings.add_sd_model) rather than as the chat LLM.
# SD 1.x / 2.x GGUFs are self-contained (UNet + VAE + CLIP in one file), so
# stable-diffusion.cpp runs them standalone. SDXL/SD3.5/FLUX need separate text
# encoders + VAE, so those stay Import-only for now.
SD_MODELS: list[dict] = [
    {
        "name": "Stable Diffusion 1.5 (Q4_0)",
        "subtitle": "1.57 GB · SD 1.5 · smallest, fast",
        "filename": "stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf",
        "sha256": "b8944e9fe0b69b36ae1b5bb0185b3a7b8ef14347fe0fa9af6c64c4829022261f",
        "size": 1566768416,
        "format": "sd-gguf",
        "url": (
            "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF"
            "/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q4_0.gguf?download=true"
        ),
    },
    {
        "name": "Stable Diffusion 1.5 (Q8_0)",
        "subtitle": "1.76 GB · SD 1.5 · higher quality",
        "filename": "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
        "sha256": "d0555243938c62faeefb4ac93f6c7a053ad373a4290c5256bce229aeb193bf94",
        "size": 1763578176,
        "format": "sd-gguf",
        "url": (
            "https://huggingface.co/second-state/stable-diffusion-v1-5-GGUF"
            "/resolve/main/stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf?download=true"
        ),
    },
    {
        "name": "Stable Diffusion 2.1 (Q8_0)",
        "subtitle": "2.01 GB · SD 2.1 · 768px",
        "filename": "v2-1_768-nonema-pruned-Q8_0.gguf",
        "sha256": "173287e9974beba1c100c6dc52c3afbb8e07038d5cfd972a34243346a2c1bf7c",
        "size": 2014680768,
        "format": "sd-gguf",
        "url": (
            "https://huggingface.co/second-state/stable-diffusion-2-1-GGUF"
            "/resolve/main/v2-1_768-nonema-pruned-Q8_0.gguf?download=true"
        ),
    },
]


def _fmt(n: int) -> str:
    if n < 1 << 20:
        return f"{n / 1024:.1f} KB"
    if n < 1 << 30:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


class SizeMismatch(Exception):
    """Completed download's byte count differs from the catalog's size."""

    def __init__(self, expected: int, actual: int) -> None:
        super().__init__(f"expected {_fmt(expected)}, got {_fmt(actual)}")
        self.expected = expected
        self.actual = actual


class ChecksumMismatch(Exception):
    """Completed download doesn't hash to the catalog's published SHA-256."""


def verify_file(
    path: Path,
    sha256: str | None,
    size: int | None,
    on_progress: Callable[[int, int], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
) -> bool:
    """Check a finished download against the catalog's size and SHA-256.

    Size is checked first (cheap); then the file is hashed in chunks.
    ``on_progress(done, total)`` fires per chunk and ``is_cancelled()`` is
    honoured between chunks (returns False when cancelled). Raises
    SizeMismatch/ChecksumMismatch on failure; returns True when the file
    checks out. Entries without a published hash pass size-only or
    unconditionally.
    """
    actual_size = path.stat().st_size
    if size is not None and actual_size != size:
        raise SizeMismatch(size, actual_size)
    if not sha256:
        return True
    digest = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            if is_cancelled is not None and is_cancelled():
                return False
            digest.update(chunk)
            done += len(chunk)
            if on_progress is not None:
                on_progress(done, actual_size)
    if digest.hexdigest() != sha256.lower():
        raise ChecksumMismatch(
            "checksum mismatch — file doesn't match its published SHA-256"
        )
    return True


class _Downloader:
    def __init__(
        self,
        url: str,
        dest: Path,
        on_progress: Callable[[int, int], None],
        on_done: Callable[[str], None],
        on_error: Callable[[str], None],
        sha256: str | None = None,
        size: int | None = None,
        on_status: Callable[[str], None] | None = None,
    ) -> None:
        self._url = url
        self._dest = dest
        self._on_progress = on_progress
        self._on_done = on_done
        self._on_error = on_error
        self._sha256 = sha256
        self._size = size
        self._on_status = on_status or (lambda _msg: None)
        self._cancelled = False
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self) -> None:
        self._cancelled = True

    def _run(self) -> None:
        tmp = self._dest.with_suffix(self._dest.suffix + ".tmp")
        try:
            require_https(self._url)
            resume_pos = tmp.stat().st_size if tmp.exists() else 0

            headers = {
                "User-Agent": "Box/0.3 (LiteRT-LM desktop)",
                "Accept-Encoding": "identity",
            }
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"

            req = urllib.request.Request(self._url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 206:
                    done = resume_pos
                    file_mode = "ab"
                else:
                    done = 0
                    resume_pos = 0
                    file_mode = "wb"

                content_length = int(resp.headers.get("Content-Length", 0))
                total = resume_pos + content_length if content_length else 0

                last_ui_update = 0.0
                with open(tmp, file_mode) as f:
                    while not self._cancelled:
                        chunk = resp.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_ui_update >= 0.25:
                            GLib.idle_add(self._on_progress, done, total)
                            last_ui_update = now

            if self._cancelled:
                tmp.unlink(missing_ok=True)
                return
            if not self._verify(tmp):
                return
            tmp.rename(self._dest)
            GLib.idle_add(self._on_done, str(self._dest))
        except Exception as exc:  # noqa: BLE001
            if not self._cancelled:
                GLib.idle_add(self._on_error, str(exc))

    def _verify(self, tmp: Path) -> bool:
        """Verify tmp against the catalog hash/size; False = don't rename.

        Three distinct outcomes surface distinctly: short download → keep
        .tmp (resumable); over-long or wrong-hash → delete .tmp; cancel
        mid-hash → delete .tmp, silent.
        """
        GLib.idle_add(self._on_status, "Verifying checksum…")
        last_status = 0.0

        def hash_progress(done: int, total: int) -> None:
            nonlocal last_status
            now = time.monotonic()
            if now - last_status >= 0.25:
                GLib.idle_add(
                    self._on_status,
                    f"Verifying checksum…  {done / total * 100:.0f}%",
                )
                last_status = now

        try:
            ok = verify_file(
                tmp, self._sha256, self._size,
                on_progress=hash_progress,
                is_cancelled=lambda: self._cancelled,
            )
        except SizeMismatch as exc:
            if exc.actual < exc.expected:
                GLib.idle_add(
                    self._on_error,
                    f"incomplete download ({exc}) — press Download to resume",
                )
            else:
                tmp.unlink(missing_ok=True)
                GLib.idle_add(self._on_error, f"size mismatch ({exc}) — deleted, try again")
            return False
        except ChecksumMismatch:
            tmp.unlink(missing_ok=True)
            GLib.idle_add(
                self._on_error,
                "checksum mismatch — file may be corrupted or tampered with; "
                "deleted, try again",
            )
            return False
        if not ok:  # cancelled mid-hash
            tmp.unlink(missing_ok=True)
            return False
        return True


class ModelRow(Adw.ActionRow):
    """One catalog entry — Download → progress → Use.

    ``on_use`` is called with the local file path once it's downloaded (or
    immediately, if it already exists on disk).
    """

    def __init__(self, model: dict, on_use: Callable[[str], None]) -> None:
        fmt = model.get("format", "litertlm")
        subtitle = f"{model['subtitle']}  ·  {FORMAT_LABELS.get(fmt, fmt)}"
        super().__init__(title=model["name"], subtitle=subtitle)
        self._model = model
        self._base_subtitle = subtitle
        self._on_use = on_use
        self._dl: _Downloader | None = None
        self._dest = MODELS_DIR / model["filename"]

        self._stack = Gtk.Stack(
            transition_type=Gtk.StackTransitionType.CROSSFADE,
            transition_duration=120,
        )

        dl_btn = Gtk.Button(label="Download", valign=Gtk.Align.CENTER)
        dl_btn.add_css_class("suggested-action")
        dl_btn.connect("clicked", self._start)
        self._stack.add_named(dl_btn, "idle")

        cancel_btn = Gtk.Button(label="Cancel", valign=Gtk.Align.CENTER)
        cancel_btn.add_css_class("destructive-action")
        cancel_btn.connect("clicked", self._cancel)
        self._stack.add_named(cancel_btn, "busy")

        use_btn = Gtk.Button(label="Use", valign=Gtk.Align.CENTER)
        use_btn.add_css_class("suggested-action")
        use_btn.connect("clicked", lambda *_: on_use(str(self._dest)))
        self._stack.add_named(use_btn, "done")

        self.add_suffix(self._stack)

        if self._dest.exists():
            self._set_done()

    def _start(self, _btn) -> None:
        self._stack.set_visible_child_name("busy")
        tmp = self._dest.with_suffix(self._dest.suffix + ".tmp")
        if tmp.exists() and (sz := tmp.stat().st_size) > 0:
            self.set_subtitle(f"Resuming from {_fmt(sz)}…")
        else:
            self.set_subtitle("Connecting…")

        # `token` pins down which _Downloader these callbacks belong to; each
        # one no-ops unless it's still current (cancel/replace race guard).
        token: list[_Downloader | None] = [None]

        def guarded_progress(done: int, total: int) -> None:
            if self._dl is token[0]:
                self._on_progress(done, total)

        def guarded_done(path: str) -> None:
            if self._dl is token[0]:
                self._on_done(path)

        def guarded_error(msg: str) -> None:
            if self._dl is token[0]:
                self._on_error(msg)

        def guarded_status(msg: str) -> None:
            if self._dl is token[0]:
                self.set_subtitle(msg)

        token[0] = _Downloader(
            self._model["url"],
            self._dest,
            guarded_progress,
            guarded_done,
            guarded_error,
            sha256=self._model.get("sha256"),
            size=self._model.get("size"),
            on_status=guarded_status,
        )
        self._dl = token[0]

    def _cancel(self, _btn) -> None:
        if self._dl:
            self._dl.cancel()
            self._dl = None
        self._stack.set_visible_child_name("idle")
        self.set_subtitle(self._base_subtitle)

    def _on_progress(self, done: int, total: int) -> None:
        if total:
            self.set_subtitle(
                f"{_fmt(done)} / {_fmt(total)}  ·  {done/total*100:.0f}%"
            )
        else:
            self.set_subtitle(f"{_fmt(done)} downloaded…")

    def _on_done(self, _path: str) -> None:
        self._dl = None
        self._set_done()

    def _on_error(self, msg: str) -> None:
        self._dl = None
        self._stack.set_visible_child_name("idle")
        self.set_subtitle(f"Error: {msg[:100]}")

    def _set_done(self) -> None:
        self._stack.set_visible_child_name("done")
        self.set_subtitle(f"{self._base_subtitle}  ·  ready")
