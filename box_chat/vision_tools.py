"""On-device image tools: Erase (MI-GAN inpainting) and Upscale (EDSR 4x).

Two small LiteRT ``.tflite`` models run on the ``ai-edge-litert`` runtime
Box already ships for RAG embedding. Both are bundled in the .deb
(``data/vision_models/``), verified by SHA-256 at load.

Engine-tier and ``gi``-free: the UI passes a PIL image in, gets a PIL image
back. Numerics are ported verbatim from Box Android's device-tested Kotlin
(MiGanEngine.kt / UpscaleEngine.kt).

MI-GAN 512 (Places2): input [1,4,512,512] NCHW = concat(mask−0.5, rgb·mask),
rgb in [-1,1] via x/127.5−1, mask binary (1=keep, 0=erase). Output
[1,3,512,512] in [-1,1]; kept pixels are pasted back bit-identical.

EDSR 4x: input [1,3,128,128] NCHW in [0,1], output [1,3,512,512]. Larger
images are tiled with 16px overlap and stitched with a half-overlap crop.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "EraseEngine", "UpscaleEngine", "VisionModelError",
    "MIGAN_SHA256", "EDSR_SHA256", "models_dir",
]

MIGAN_SHA256 = "ef53f8dca69e5ce29629441128322c4d9a0a527b2a043703e0ab497f2463c25f"
EDSR_SHA256 = "77fdb8655436844e94187ce6675f5c8c73ed5bf3080327d8ebe96e15da829bde"

_MIGAN_SIDE = 512
_EDSR_IN = 128
_EDSR_OUT = 512
_EDSR_OVERLAP = 16
_EDSR_MAX_SRC_LONG = 1024


class VisionModelError(Exception):
    """Model file missing, hash mismatch, or inference failed."""


def models_dir() -> Path:
    for d in (
        Path("/opt/box/vision_models"),
        Path(__file__).resolve().parent.parent / "data" / "vision_models",
    ):
        if d.is_dir():
            return d
    return Path(__file__).resolve().parent.parent / "data" / "vision_models"


def _verify(path: Path, sha256: str) -> None:
    if not path.is_file():
        raise VisionModelError(f"vision model not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    if h.hexdigest() != sha256:
        raise VisionModelError(f"checksum mismatch for {path.name}")


def _load_interpreter(path: Path):
    from ai_edge_litert.interpreter import Interpreter
    import os

    n_threads = max(1, (os.cpu_count() or 4) - 2)
    interp = Interpreter(model_path=str(path), num_threads=n_threads)
    interp.allocate_tensors()
    return interp


def _to_rgb_array(image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _resize(image, size: tuple[int, int]):
    from PIL import Image

    return image.convert("RGB").resize(size, Image.BILINEAR)


class EraseEngine:
    """MI-GAN inpainting. Thread-safe (Interpreter isn't — calls serialised)."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._path = model_path or (models_dir() / "migan_fp16.tflite")
        self._interp = None
        self._in_idx: int | None = None
        self._out_idx: int | None = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self._path.is_file()

    def _ensure(self) -> None:
        if self._interp is not None:
            return
        with self._lock:
            if self._interp is not None:
                return
            _verify(self._path, MIGAN_SHA256)
            interp = _load_interpreter(self._path)
            self._in_idx = interp.get_input_details()[0]["index"]
            self._out_idx = interp.get_output_details()[0]["index"]
            self._interp = interp
            log.info("Erase (MI-GAN) model ready")

    def erase(self, image, mask):
        """Inpaint the erased regions of ``image``.

        ``mask``: PIL image where DARK pixels (<128) mark regions to erase.
        Returns a PIL RGB image the size of the input; kept pixels pasted
        back exactly (no quality loss on untouched areas).
        """
        from PIL import Image

        self._ensure()
        assert self._interp is not None
        orig_size = image.size

        img512 = _to_rgb_array(_resize(image, (_MIGAN_SIDE, _MIGAN_SIDE)))
        mask512 = np.asarray(
            _resize(mask, (_MIGAN_SIDE, _MIGAN_SIDE)).convert("L"), dtype=np.uint8
        )
        keep = (mask512 >= 128).astype(np.float32)

        rgb = img512.astype(np.float32) / 127.5 - 1.0
        inp = np.empty((1, 4, _MIGAN_SIDE, _MIGAN_SIDE), dtype=np.float32)
        inp[0, 0] = keep - 0.5
        for c in range(3):
            inp[0, c + 1] = rgb[:, :, c] * keep

        with self._lock:
            self._interp.set_tensor(self._in_idx, inp)
            self._interp.invoke()
            out = np.array(
                self._interp.get_tensor(self._out_idx)[0], dtype=np.float32
            )

        gen = np.clip(out / 2.0 + 0.5, 0.0, 1.0) * 255.0
        gen_hwc = np.transpose(gen, (1, 2, 0))
        keep3 = keep[:, :, None]
        composited = np.where(
            keep3 >= 0.5, img512.astype(np.float32), gen_hwc
        ).astype(np.uint8)

        result512 = Image.fromarray(composited, mode="RGB")
        if orig_size != (_MIGAN_SIDE, _MIGAN_SIDE):
            result512 = result512.resize(orig_size, Image.BILINEAR)
        return result512


class UpscaleEngine:
    """EDSR 4x super-resolution via overlap tiling. Thread-safe."""

    scale = _EDSR_OUT // _EDSR_IN

    def __init__(self, model_path: Path | None = None) -> None:
        self._path = model_path or (models_dir() / "edsr.tflite")
        self._interp = None
        self._in_idx: int | None = None
        self._out_idx: int | None = None
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self._path.is_file()

    def _ensure(self) -> None:
        if self._interp is not None:
            return
        with self._lock:
            if self._interp is not None:
                return
            _verify(self._path, EDSR_SHA256)
            interp = _load_interpreter(self._path)
            self._in_idx = interp.get_input_details()[0]["index"]
            self._out_idx = interp.get_output_details()[0]["index"]
            self._interp = interp
            log.info("Upscale (EDSR) model ready")

    @staticmethod
    def _tile_starts(total: int, tile: int, step: int) -> list[int]:
        if total <= tile:
            return [0]
        starts: list[int] = []
        p = 0
        while p + tile < total:
            starts.append(p)
            p += step
        starts.append(total - tile)
        seen: set[int] = set()
        return [s for s in starts if not (s in seen or seen.add(s))]

    def _run_tile(self, tile_rgb: np.ndarray) -> np.ndarray:
        x = tile_rgb.astype(np.float32) / 255.0
        inp = np.empty((1, 3, _EDSR_IN, _EDSR_IN), dtype=np.float32)
        for c in range(3):
            inp[0, c] = x[:, :, c]
        with self._lock:
            self._interp.set_tensor(self._in_idx, inp)
            self._interp.invoke()
            out = np.array(
                self._interp.get_tensor(self._out_idx)[0], dtype=np.float32
            )
        return np.transpose(out, (1, 2, 0))

    def upscale(self, image, on_progress=None):
        """4x super-resolve ``image`` (PIL). Returns a PIL RGB image."""
        from PIL import Image

        self._ensure()
        assert self._interp is not None

        src = image.convert("RGB")
        long_edge = max(src.size)
        if long_edge > _EDSR_MAX_SRC_LONG:
            r = _EDSR_MAX_SRC_LONG / long_edge
            src = src.resize(
                (max(1, int(src.width * r)), max(1, int(src.height * r))),
                Image.BILINEAR,
            )
        src_arr = _to_rgb_array(src)
        src_h, src_w = src_arr.shape[:2]
        s = self.scale

        if src_w < _EDSR_IN or src_h < _EDSR_IN:
            tile = np.asarray(
                Image.fromarray(src_arr).resize((_EDSR_IN, _EDSR_IN), Image.BILINEAR),
                dtype=np.uint8,
            )
            out = self._run_tile(tile)
            out_img = Image.fromarray(
                (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB"
            )
            if on_progress:
                on_progress(1.0)
            return out_img.resize((src_w * s, src_h * s), Image.BILINEAR)

        overlap = min(_EDSR_OVERLAP, min(src_w, src_h) // 2)
        step_x = max(1, _EDSR_IN - overlap)
        step_y = max(1, _EDSR_IN - overlap)
        xs = self._tile_starts(src_w, _EDSR_IN, step_x)
        ys = self._tile_starts(src_h, _EDSR_IN, step_y)

        dst = np.zeros((src_h * s, src_w * s, 3), dtype=np.float32)
        half = overlap // 2
        total_tiles = len(xs) * len(ys)
        done = 0
        for y0 in ys:
            for x0 in xs:
                tile = src_arr[y0:y0 + _EDSR_IN, x0:x0 + _EDSR_IN]
                out = self._run_tile(tile)
                cx0 = 0 if x0 == 0 else half
                cy0 = 0 if y0 == 0 else half
                cx1 = _EDSR_IN if (x0 + _EDSR_IN) >= src_w else _EDSR_IN - half
                cy1 = _EDSR_IN if (y0 + _EDSR_IN) >= src_h else _EDSR_IN - half
                sx0, sy0 = (x0 + cx0) * s, (y0 + cy0) * s
                dst[sy0:sy0 + (cy1 - cy0) * s, sx0:sx0 + (cx1 - cx0) * s] = (
                    out[cy0 * s:cy1 * s, cx0 * s:cx1 * s]
                )
                done += 1
                if on_progress:
                    on_progress(done / total_tiles)

        return Image.fromarray((np.clip(dst, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")
