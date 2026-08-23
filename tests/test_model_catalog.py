"""Model catalog integrity + download verification.

Covers box_chat/model_catalog.py's non-GTK core: every catalog entry carries
a sha256 + size (the trust anchor for community GGUFs), and verify_file
enforces both — SizeMismatch first (cheap), then ChecksumMismatch — while
honouring cancellation and progress callbacks.
"""
from __future__ import annotations

import hashlib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from box_chat import model_catalog as mc


class CatalogIntegrityTests(unittest.TestCase):
    def test_catalog_entry_count(self) -> None:
        # 8 original (4 Gemma litertlm + 4 community gguf) + 5 Gemma 4 QAT gguf
        # + 3 Qwen 2026 coders (Qwen3-Coder-30B-A3B, Qwen3.6-35B-A3B, Qwen3.6-27B).
        self.assertEqual(len(mc.MODELS), 16)
        qat = [m for m in mc.MODELS if "QAT GGUF" in m["name"]]
        self.assertEqual(len(qat), 5)
        qwen = [m for m in mc.MODELS if m["name"].startswith("Qwen3")]
        self.assertEqual(len(qwen), 3)

    def test_every_entry_has_sha256_and_size(self) -> None:
        for m in mc.MODELS:
            with self.subTest(model=m["name"]):
                self.assertIn("sha256", m)
                self.assertIn("size", m)
                self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$")
                self.assertIsInstance(m["size"], int)
                self.assertGreater(m["size"], 0)

    def test_every_entry_has_required_fields(self) -> None:
        for m in mc.MODELS:
            with self.subTest(model=m.get("name")):
                for field in ("name", "subtitle", "filename", "format", "url"):
                    self.assertIn(field, m)
                self.assertIn(m["format"], ("litertlm", "gguf"))
                # HTTPS is the transport-trust half of the anchor.
                self.assertTrue(m["url"].startswith("https://"))

    def test_filenames_unique(self) -> None:
        names = [m["filename"] for m in mc.MODELS]
        self.assertEqual(len(names), len(set(names)))


class SDCatalogIntegrityTests(unittest.TestCase):
    def test_sd_catalog_shape(self) -> None:
        self.assertEqual(len(mc.SD_MODELS), 3)
        for m in mc.SD_MODELS:
            with self.subTest(model=m.get("name")):
                for field in ("name", "subtitle", "filename", "sha256", "size", "format", "url"):
                    self.assertIn(field, m)
                self.assertEqual(m["format"], "sd-gguf")
                self.assertRegex(m["sha256"], r"^[0-9a-f]{64}$")
                self.assertIsInstance(m["size"], int)
                self.assertGreater(m["size"], 0)
                self.assertTrue(m["url"].startswith("https://"))
                self.assertIn("sd-gguf", mc.FORMAT_LABELS)

    def test_sd_filenames_unique_and_disjoint_from_chat(self) -> None:
        sd = [m["filename"] for m in mc.SD_MODELS]
        self.assertEqual(len(sd), len(set(sd)))
        chat = {m["filename"] for m in mc.MODELS}
        self.assertTrue(chat.isdisjoint(sd))


def _write(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


class VerifyFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_good_file_passes(self) -> None:
        p = self.dir / "m.gguf"
        digest = _write(p, b"hello box" * 1000)
        self.assertTrue(mc.verify_file(p, digest, len(b"hello box" * 1000)))

    def test_size_mismatch_raises_first(self) -> None:
        p = self.dir / "m.gguf"
        _write(p, b"x" * 100)
        # Wrong size AND wrong hash → SizeMismatch wins (checked first, cheap).
        with self.assertRaises(mc.SizeMismatch) as ctx:
            mc.verify_file(p, "0" * 64, 999)
        self.assertEqual(ctx.exception.expected, 999)
        self.assertEqual(ctx.exception.actual, 100)

    def test_checksum_mismatch_raises(self) -> None:
        p = self.dir / "m.gguf"
        _write(p, b"abcd" * 50)
        with self.assertRaises(mc.ChecksumMismatch):
            mc.verify_file(p, "0" * 64, len(b"abcd" * 50))

    def test_no_hash_is_size_only(self) -> None:
        p = self.dir / "m.gguf"
        _write(p, b"z" * 42)
        self.assertTrue(mc.verify_file(p, None, 42))
        with self.assertRaises(mc.SizeMismatch):
            mc.verify_file(p, None, 43)

    def test_no_hash_no_size_passes_unconditionally(self) -> None:
        p = self.dir / "m.gguf"
        _write(p, b"whatever")
        self.assertTrue(mc.verify_file(p, None, None))

    def test_cancel_returns_false_mid_hash(self) -> None:
        p = self.dir / "m.gguf"
        digest = _write(p, b"y" * (8 * 1024 * 1024))  # 2 chunks
        self.assertFalse(
            mc.verify_file(p, digest, 8 * 1024 * 1024, is_cancelled=lambda: True)
        )

    def test_progress_callback_fires(self) -> None:
        p = self.dir / "m.gguf"
        data = b"p" * (6 * 1024 * 1024)
        digest = _write(p, data)
        seen: list[tuple[int, int]] = []
        self.assertTrue(
            mc.verify_file(p, digest, len(data), on_progress=lambda d, t: seen.append((d, t)))
        )
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], len(data))  # last progress == total

    def test_uppercase_hash_accepted(self) -> None:
        p = self.dir / "m.gguf"
        digest = _write(p, b"caseinsensitive")
        self.assertTrue(mc.verify_file(p, digest.upper(), len(b"caseinsensitive")))


class FmtTests(unittest.TestCase):
    def test_human_sizes(self) -> None:
        self.assertIn("KB", mc._fmt(2048))
        self.assertIn("MB", mc._fmt(5 * 1024 * 1024))
        self.assertIn("GB", mc._fmt(3 * 1024 ** 3))


if __name__ == "__main__":
    unittest.main()
