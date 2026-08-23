"""Downloadable LiteRT diffusion models (Z-Image Turbo + FLUX.2-klein).

These models are *directories* of chunked ``.tflite`` graphs plus a shared
Qwen tokenizer — not the single-file GGUF/litertlm entries in
``model_catalog.py``. This module carries the per-file manifests (copied
verbatim from Box Android's ZImageModels.kt / FluxKleinModels.kt, sizes from
the Hugging Face repos) and a gi-free, resumable multi-file downloader.

The four ``qwen_*`` tokenizer files are byte-for-byte identical across both
models, so when one is already on disk the downloader hard-links (falling
back to copy) instead of re-fetching ~0.8GB.

Engine-tier and ``gi``-free — the UI tier drives it on a worker thread.
Size is the integrity check (the HF repos publish sizes, not per-file
SHA-256); HTTPS protects the transfer.
"""
from __future__ import annotations

import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .net import require_https

log = logging.getLogger(__name__)

__all__ = [
    "LiterFile", "LiterModel", "LITERT_MODELS", "get_model",
    "download_model", "LiterDownloadError", "LiterDownloadCancelled",
    "SHARED_TOKENIZER_FILES",
]

_ZIMAGE_REPO = "https://huggingface.co/litert-community/Z-Image-Turbo-LiteRT/resolve/main"
_KLEIN_REPO = "https://huggingface.co/litert-community/FLUX.2-klein-4B-LiteRT/resolve/main"

# Bit-identical across Z-Image and klein — shared/hard-linked, never re-fetched.
SHARED_TOKENIZER_FILES = (
    "qwen_embed_fp16.bin", "qwen_vocab.txt", "qwen_merges.txt", "qwen_special.txt",
)


class LiterDownloadError(Exception):
    """A file failed to download or didn't match its published size."""


class LiterDownloadCancelled(Exception):
    """The user cancelled the download."""


@dataclass(frozen=True)
class LiterFile:
    name: str          # local filename inside the model directory
    url: str
    size: int          # bytes (from the HF repo — the integrity check)
    shared: bool = False   # part of the shared qwen tokenizer set


@dataclass(frozen=True)
class LiterModel:
    key: str           # directory name + settings key
    name: str
    engine: str        # "zimage" | "fluxklein"
    info: str
    files: tuple[LiterFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)


def _tok(repo: str) -> tuple[LiterFile, ...]:
    """The shared Qwen tokenizer files, served from ``repo``/tokenizer/."""
    sizes = {
        "qwen_embed_fp16.bin": 777_912_320,
        "qwen_vocab.txt": 1_521_491,
        "qwen_merges.txt": 1_671_838,
        "qwen_special.txt": 547,
    }
    return tuple(
        LiterFile(name, f"{repo}/tokenizer/{name}?download=true", size, shared=True)
        for name, size in sizes.items()
    )


def _f(repo: str, name: str, size: int) -> LiterFile:
    return LiterFile(name, f"{repo}/{name}?download=true", size)


# ── Z-Image Turbo 6B (~10.6GB) — 13 graphs + shared tokenizer ───────────────
_ZIMAGE = LiterModel(
    key="zimage-turbo-6b",
    name="Z-Image Turbo 6B",
    engine="zimage",
    info=(
        "Alibaba Tongyi's Z-Image-Turbo 6B single-stream diffusion transformer "
        "as chunked int8 LiteRT graphs. 256×256, 9 steps, guidance-free. "
        "~10.6GB (0.8GB shared with FLUX klein if installed)."
    ),
    files=(
        _f(_ZIMAGE_REPO, "qwen_enc.tflite", 3_547_652_208),
        _f(_ZIMAGE_REPO, "z_embx.tflite", 308_272),
        _f(_ZIMAGE_REPO, "z_refx.tflite", 363_386_160),
        _f(_ZIMAGE_REPO, "z_embc.tflite", 9_906_096),
        _f(_ZIMAGE_REPO, "z_refc.tflite", 355_025_040),
        _f(_ZIMAGE_REPO, "zc_main0.tflite", 908_480_960),
        _f(_ZIMAGE_REPO, "zc_main1.tflite", 908_480_960),
        _f(_ZIMAGE_REPO, "zc_main2.tflite", 908_480_960),
        _f(_ZIMAGE_REPO, "zc_main3.tflite", 908_480_960),
        _f(_ZIMAGE_REPO, "zc_main4.tflite", 908_480_960),
        _f(_ZIMAGE_REPO, "zc_main5.tflite", 908_480_960),
        _f(_ZIMAGE_REPO, "zc_final.tflite", 1_295_936),
        _f(_ZIMAGE_REPO, "zvae.tflite", 50_139_872),
        *_tok(_KLEIN_REPO),   # Z-Image pulls the tokenizer from klein's repo
    ),
)

# ── FLUX.2 klein 4B (~7.4GB) — 12 graphs + shared tokenizer ─────────────────
_KLEIN = LiterModel(
    key="flux2-klein-4b",
    name="FLUX.2 klein 4B",
    engine="fluxklein",
    info=(
        "Black Forest Labs FLUX.2 [klein] 4B rectified-flow transformer as "
        "chunked int8 LiteRT graphs. 256×256, 4 steps, guidance-free. ~7.4GB."
    ),
    files=(
        _f(_KLEIN_REPO, "ke_enc0.tflite", 912_190_032),
        _f(_KLEIN_REPO, "ke_enc1.tflite", 912_190_032),
        _f(_KLEIN_REPO, "ke_enc2.tflite", 912_190_032),
        _f(_KLEIN_REPO, "kc_prep.tflite", 166_174_032),
        _f(_KLEIN_REPO, "kc_double0.tflite", 738_688_720),
        _f(_KLEIN_REPO, "kc_double1.tflite", 492_460_800),
        _f(_KLEIN_REPO, "kc_single0.tflite", 615_367_264),
        _f(_KLEIN_REPO, "kc_single1.tflite", 615_367_264),
        _f(_KLEIN_REPO, "kc_single2.tflite", 615_367_264),
        _f(_KLEIN_REPO, "kc_single3.tflite", 615_367_264),
        _f(_KLEIN_REPO, "kc_final.tflite", 19_348_608),
        _f(_KLEIN_REPO, "kv_vae.tflite", 50_207_984),
        *_tok(_KLEIN_REPO),
    ),
)

LITERT_MODELS: tuple[LiterModel, ...] = (_ZIMAGE, _KLEIN)


def get_model(key: str) -> LiterModel | None:
    return next((m for m in LITERT_MODELS if m.key == key), None)


# ── downloader ──────────────────────────────────────────────────────────────
_CHUNK = 4 * 1024 * 1024


def _try_share(f: LiterFile, target: Path, share_from) -> bool:
    """Hard-link (or copy) a shared tokenizer file from a sibling model dir."""
    for d in share_from:
        src = Path(d) / f.name
        try:
            if not (src.is_file() and src.stat().st_size == f.size):
                continue
            if src.resolve() == target.resolve():
                return True
        except OSError:
            continue
        target.unlink(missing_ok=True)
        try:
            os.link(src, target)
        except OSError:
            shutil.copy2(src, target)
        log.info("shared %s from %s", f.name, d)
        return True
    return False


def _download_one(
    f: LiterFile, target: Path,
    on_chunk: Callable[[int], None], cancelled: Callable[[], bool],
) -> None:
    """Fetch one file into ``target`` with resume + size verification.

    A short download keeps its ``.tmp`` (resumable next time); an over-long or
    otherwise size-wrong download is deleted. Raises on failure/cancel.
    """
    tmp = target.with_suffix(target.suffix + ".tmp")
    require_https(f.url)
    resume = tmp.stat().st_size if tmp.exists() else 0
    if resume > f.size:
        tmp.unlink(missing_ok=True)
        resume = 0

    headers = {
        "User-Agent": "Box/0.3 (LiteRT desktop)",
        "Accept-Encoding": "identity",
    }
    if resume > 0:
        headers["Range"] = f"bytes={resume}-"

    req = urllib.request.Request(f.url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        if resp.status == 206:
            file_done, mode = resume, "ab"
        else:
            file_done, mode = 0, "wb"
        with open(tmp, mode) as out:
            while not cancelled():
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                file_done += len(chunk)
                on_chunk(file_done)

    if cancelled():
        raise LiterDownloadCancelled()

    sz = tmp.stat().st_size
    if sz != f.size:
        if sz < f.size:
            raise LiterDownloadError(
                f"{f.name}: incomplete ({sz}/{f.size} bytes) — press Download to resume"
            )
        tmp.unlink(missing_ok=True)
        raise LiterDownloadError(f"{f.name}: size mismatch ({sz} != {f.size}) — deleted")
    tmp.rename(target)


def download_model(
    model: LiterModel,
    dest_dir: str | Path,
    *,
    on_overall: Callable[[int, int], None] | None = None,
    on_status: Callable[[str], None] | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    share_from=(),
) -> None:
    """Download every file of ``model`` into ``dest_dir``.

    Files already present at the right size are skipped; shared tokenizer
    files are hard-linked from any ``share_from`` sibling dir that has them.
    ``on_overall(done_bytes, total_bytes)`` tracks the whole set;
    ``on_status(text)`` names the current file; ``is_cancelled()`` is polled
    between chunks. Raises LiterDownloadError / LiterDownloadCancelled.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    total = model.total_bytes
    done = 0
    n = len(model.files)

    def cancelled() -> bool:
        return bool(is_cancelled and is_cancelled())

    for i, f in enumerate(model.files, 1):
        if cancelled():
            raise LiterDownloadCancelled()
        target = dest / f.name

        if target.is_file() and target.stat().st_size == f.size:
            done += f.size
            if on_overall:
                on_overall(done, total)
            continue

        if f.shared and _try_share(f, target, share_from):
            done += f.size
            if on_overall:
                on_overall(done, total)
            continue

        if on_status:
            on_status(f"Downloading {f.name}  ({i}/{n})")
        base = done

        def on_chunk(file_done: int, _base=base) -> None:
            if on_overall:
                on_overall(_base + file_done, total)

        _download_one(f, target, on_chunk, cancelled)
        done = base + f.size

    if on_overall:
        on_overall(total, total)
