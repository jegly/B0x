"""Embedding model downloader.

Pulls EmbeddingGemma .tflite + sentencepiece tokenizer from
huggingface.co/jegly/mirror on demand. Same resumable-download pattern as
``tts.py``: stream to a .tmp file, atomically rename on completion.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
import urllib.request
from collections.abc import Callable
from pathlib import Path

from .config import EMBED_DIR
from .net import require_https

log = logging.getLogger(__name__)

_HF_BASE = "https://huggingface.co/jegly/mirror/resolve/main"

# Embedding variants the user can choose from. The tokenizer file is shared
# across every variant, so it lives outside this map.
VARIANTS: dict[str, dict[str, str]] = {
    "seq1024": {
        "display": "Seq 1024 (recommended, 183 MB)",
        "filename": "embeddinggemma-300M_seq1024_mixed-precision.tflite",
        "url": f"{_HF_BASE}/embeddinggemma-300M_seq1024_mixed-precision.tflite",
        "size_mb": 183,
    },
    "seq2048": {
        "display": "Seq 2048 (longer chunks, 196 MB)",
        "filename": "embeddinggemma-300M_seq2048_mixed-precision.tflite",
        "url": f"{_HF_BASE}/embeddinggemma-300M_seq2048_mixed-precision.tflite",
        "size_mb": 196,
    },
}

TOKENIZER_FILENAME = "sentencepiece.model"
TOKENIZER_URL = f"{_HF_BASE}/{TOKENIZER_FILENAME}"


def model_path(variant: str) -> Path:
    spec = VARIANTS[variant]
    return EMBED_DIR / spec["filename"]


def tokenizer_path() -> Path:
    return EMBED_DIR / TOKENIZER_FILENAME


def is_variant_ready(variant: str) -> bool:
    return model_path(variant).is_file() and tokenizer_path().is_file()


# ── download orchestration ──────────────────────────────────────────────
def download_variant(
    variant: str,
    on_progress: Callable[[str, int, int], None],
    on_done: Callable[[], None],
    on_error: Callable[[str], None],
) -> None:
    """Download ``variant``'s .tflite and the shared tokenizer in a worker thread."""
    threading.Thread(
        target=_do_download,
        args=(variant, on_progress, on_done, on_error),
        daemon=True,
    ).start()


def _do_download(
    variant: str,
    on_progress: Callable[[str, int, int], None],
    on_done: Callable[[], None],
    on_error: Callable[[str], None],
) -> None:
    spec = VARIANTS.get(variant)
    if spec is None:
        on_error(f"Unknown embed variant: {variant}")
        return
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    try:
        if not tokenizer_path().is_file():
            _dl_with_progress(TOKENIZER_URL, tokenizer_path(), "Tokenizer", on_progress)
        if not model_path(variant).is_file():
            _dl_with_progress(spec["url"], model_path(variant), "Embed model", on_progress)
        on_done()
    except Exception as e:  # noqa: BLE001
        log.exception("RAG download failed")
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

    try:
        # shutil.move handles cross-device moves and is more robust than
        # Path.rename() which can fail with ENOENT on some Python 3.14 builds.
        shutil.move(str(tmp), str(dest))
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
