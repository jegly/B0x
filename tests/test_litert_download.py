"""LiteRT diffusion model catalog + multi-file downloader.

Covers box_chat/litert_diffusion_models.py: the Z-Image / klein manifests are
well-formed (sizes, https, shared tokenizer set), and download_model handles
the real cases — fresh fetch, skip-if-present, resume, size verification
(short vs over-long), shared-file hard-linking, and cancel — all against a
faked urlopen (no network).
"""
from __future__ import annotations

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from box_chat import litert_diffusion_models as ldm
from box_chat.litert_diffusion_models import (
    LiterFile, LiterModel, LiterDownloadCancelled, LiterDownloadError,
    LITERT_MODELS, SHARED_TOKENIZER_FILES, download_model, get_model,
)


class CatalogTests(unittest.TestCase):
    def test_two_models_present(self) -> None:
        keys = {m.key for m in LITERT_MODELS}
        self.assertEqual(keys, {"zimage-turbo-6b", "flux2-klein-4b"})

    def test_manifests_well_formed(self) -> None:
        for m in LITERT_MODELS:
            with self.subTest(model=m.key):
                self.assertIn(m.engine, ("zimage", "fluxklein"))
                self.assertTrue(m.files)
                for f in m.files:
                    self.assertTrue(f.url.startswith("https://"))
                    self.assertGreater(f.size, 0)
                # Every model ships the same 4 shared tokenizer files.
                shared = {f.name for f in m.files if f.shared}
                self.assertEqual(shared, set(SHARED_TOKENIZER_FILES))

    def test_totals_in_expected_range(self) -> None:
        z = get_model("zimage-turbo-6b").total_bytes / 1024 ** 3
        k = get_model("flux2-klein-4b").total_bytes / 1024 ** 3
        self.assertAlmostEqual(z, 9.8, delta=0.5)   # ~10.6GB decimal
        self.assertAlmostEqual(k, 6.9, delta=0.5)   # ~7.4GB decimal

    def test_main_graphs_present(self) -> None:
        znames = {f.name for f in get_model("zimage-turbo-6b").files}
        knames = {f.name for f in get_model("flux2-klein-4b").files}
        self.assertIn("qwen_enc.tflite", znames)
        self.assertIn("zvae.tflite", znames)
        self.assertIn("ke_enc0.tflite", knames)
        self.assertIn("kv_vae.tflite", knames)


# ── downloader (faked network) ──────────────────────────────────────────────
def _mk_model(files: dict[str, bytes]) -> tuple[LiterModel, dict[str, bytes]]:
    registry = {f"https://host/{n}": data for n, data in files.items()}
    model = LiterModel(
        key="fake", name="Fake", engine="zimage", info="",
        files=tuple(
            LiterFile(n, f"https://host/{n}", len(data),
                      shared=n in SHARED_TOKENIZER_FILES)
            for n, data in files.items()
        ),
    )
    return model, registry


class _FakeResp:
    def __init__(self, data: bytes, status: int) -> None:
        self._buf = io.BytesIO(data)
        self.status = status

    def read(self, n: int = -1) -> bytes:
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_urlopen_factory(registry: dict[str, bytes], hits: list[str],
                          over: dict | None = None):
    def fake(req, timeout=60):
        url = req.full_url
        hits.append(url)
        data = registry[url]
        if over and url in over:
            data = over[url]
        rng = req.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            return _FakeResp(data[start:], 206)
        return _FakeResp(data, 200)
    return fake


class DownloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)
        self._patch = mock.patch.object(ldm, "require_https", lambda _u: None)
        self._patch.start()

    def tearDown(self) -> None:
        self._patch.stop()
        self._td.cleanup()

    def _run(self, model, registry, dest, over=None, **kw):
        hits: list[str] = []
        with mock.patch.object(
            ldm.urllib.request, "urlopen",
            _fake_urlopen_factory(registry, hits, over),
        ):
            download_model(model, dest, **kw)
        return hits

    def test_fresh_download_writes_all(self) -> None:
        model, reg = _mk_model({"a.tflite": b"A" * 100, "qwen_vocab.txt": b"vocab"})
        dest = self.dir / "m"
        seen: list = []
        self._run(model, reg, dest, on_overall=lambda d, t: seen.append((d, t)))
        self.assertEqual((dest / "a.tflite").read_bytes(), b"A" * 100)
        self.assertEqual((dest / "qwen_vocab.txt").read_bytes(), b"vocab")
        self.assertEqual(seen[-1], (105, 105))  # overall reached total

    def test_skip_present_file(self) -> None:
        model, reg = _mk_model({"a.tflite": b"A" * 50})
        dest = self.dir / "m"
        dest.mkdir()
        (dest / "a.tflite").write_bytes(b"A" * 50)  # already correct size
        hits = self._run(model, reg, dest)
        self.assertEqual(hits, [])  # nothing fetched

    def test_shared_file_hardlinked(self) -> None:
        model, reg = _mk_model({"qwen_special.txt": b"special-token-data"})
        sibling = self.dir / "other"
        sibling.mkdir()
        (sibling / "qwen_special.txt").write_bytes(b"special-token-data")
        dest = self.dir / "m"
        hits = self._run(model, reg, dest, share_from=[str(sibling)])
        self.assertEqual(hits, [])  # linked, not downloaded
        self.assertEqual((dest / "qwen_special.txt").read_bytes(), b"special-token-data")

    def test_resume_partial_tmp(self) -> None:
        model, reg = _mk_model({"a.tflite": b"0123456789"})
        dest = self.dir / "m"
        dest.mkdir()
        (dest / "a.tflite.tmp").write_bytes(b"01234")  # 5 of 10 already there
        hits = self._run(model, reg, dest)
        self.assertEqual((dest / "a.tflite").read_bytes(), b"0123456789")
        # Server saw a Range request (resumed, not restarted).
        self.assertEqual(len(hits), 1)

    def test_over_long_deletes_tmp(self) -> None:
        model, reg = _mk_model({"a.tflite": b"X" * 10})
        dest = self.dir / "m"
        with self.assertRaises(LiterDownloadError):
            self._run(model, reg, dest, over={"https://host/a.tflite": b"X" * 15})
        self.assertFalse((dest / "a.tflite.tmp").exists())  # deleted
        self.assertFalse((dest / "a.tflite").exists())

    def test_short_keeps_tmp(self) -> None:
        model, reg = _mk_model({"a.tflite": b"X" * 20})
        dest = self.dir / "m"
        with self.assertRaises(LiterDownloadError):
            self._run(model, reg, dest, over={"https://host/a.tflite": b"X" * 8})
        self.assertTrue((dest / "a.tflite.tmp").exists())  # kept for resume

    def test_cancel_raises(self) -> None:
        model, reg = _mk_model({"a.tflite": b"A" * 100})
        dest = self.dir / "m"
        with self.assertRaises(LiterDownloadCancelled):
            self._run(model, reg, dest, is_cancelled=lambda: True)


if __name__ == "__main__":
    unittest.main()
