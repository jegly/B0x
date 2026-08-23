"""On-device vision tools: Erase (MI-GAN) and Upscale (EDSR).

Covers box_chat/vision_tools.py: SHA-256 gating rejects a tampered model,
the bundled models match their pinned hashes, tile-start math is correct, and
— when the LiteRT runtime + models are present — a full-keep erase leaves the
image bit-identical (0.0 diff) and upscale returns exactly 4x dimensions.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from box_chat import vision_tools as vt

REPO = Path(__file__).resolve().parent.parent
MIGAN = vt.models_dir() / "migan_fp16.tflite"
EDSR = vt.models_dir() / "edsr.tflite"


def _litert_available() -> bool:
    try:
        import ai_edge_litert.interpreter  # noqa: F401
        return True
    except Exception:
        return False


class VerifyTests(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with self.assertRaises(vt.VisionModelError):
            vt._verify(Path("/no/such/model.tflite"), "0" * 64)

    def test_wrong_hash_rejected(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "fake.tflite"
            p.write_bytes(b"not a real model")
            with self.assertRaises(vt.VisionModelError):
                vt._verify(p, "0" * 64)

    def test_correct_hash_passes(self) -> None:
        import hashlib
        with TemporaryDirectory() as td:
            p = Path(td) / "ok.tflite"
            data = b"content here"
            p.write_bytes(data)
            vt._verify(p, hashlib.sha256(data).hexdigest())  # no raise


class BundledModelIntegrityTests(unittest.TestCase):
    @unittest.skipUnless(MIGAN.is_file(), "migan model not bundled")
    def test_migan_matches_pin(self) -> None:
        vt._verify(MIGAN, vt.MIGAN_SHA256)  # must not raise

    @unittest.skipUnless(EDSR.is_file(), "edsr model not bundled")
    def test_edsr_matches_pin(self) -> None:
        vt._verify(EDSR, vt.EDSR_SHA256)


class TileMathTests(unittest.TestCase):
    def test_small_total_single_tile(self) -> None:
        self.assertEqual(vt.UpscaleEngine._tile_starts(100, 128, 112), [0])

    def test_tiles_cover_and_end_flush(self) -> None:
        starts = vt.UpscaleEngine._tile_starts(300, 128, 112)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 300 - 128)  # last tile ends flush
        self.assertEqual(starts, sorted(set(starts)))  # ascending, deduped

    def test_scale_is_four(self) -> None:
        self.assertEqual(vt.UpscaleEngine.scale, 4)


@unittest.skipUnless(
    MIGAN.is_file() and _litert_available(),
    "MI-GAN model or LiteRT runtime not present",
)
class LiveEraseTests(unittest.TestCase):
    def test_full_keep_mask_is_identity(self) -> None:
        from PIL import Image

        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
        # All-white mask → keep everything (mask>=128 everywhere).
        mask = Image.new("L", (512, 512), 255)

        out = vt.EraseEngine().erase(img, mask)
        self.assertEqual(out.size, (512, 512))
        diff = np.abs(np.asarray(out, np.int16) - arr.astype(np.int16))
        # Kept pixels are pasted back bit-identical: zero difference.
        self.assertEqual(int(diff.max()), 0)


@unittest.skipUnless(
    EDSR.is_file() and _litert_available(),
    "EDSR model or LiteRT runtime not present",
)
class LiveUpscaleTests(unittest.TestCase):
    def test_output_is_exactly_4x(self) -> None:
        from PIL import Image

        rng = np.random.default_rng(1)
        img = Image.fromarray(
            rng.integers(0, 256, (48, 64, 3), dtype=np.uint8), "RGB"
        )  # 64x48 (WxH)
        seen: list[float] = []
        out = vt.UpscaleEngine().upscale(img, on_progress=seen.append)
        self.assertEqual(out.size, (64 * 4, 48 * 4))  # exact 4x
        self.assertTrue(seen and seen[-1] == 1.0)  # progress reached 100%


if __name__ == "__main__":
    unittest.main()
