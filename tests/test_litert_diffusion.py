"""LiteRT diffusion pipelines (Z-Image Turbo + FLUX.2-klein) — pure math.

Covers box_chat/litert_diffusion.py's host-side numerics that don't need the
~10GB .tflite graphs: patchify/unpatchify roundtrip, the sigma schedule, the
t-embedding, RoPE table shapes, klein's causal+pad attention mask and
bn-denorm unpatchify, and that the bundled staged assets match the sizes the
loaders assert. The graph-driven end-to-end render needs the model download
(see TODO_after_recovery P4e) and is out of scope here.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from box_chat import litert_diffusion as ld
from box_chat.litert_diffusion import (
    FluxKleinPipeline, ZImagePipeline, assets_dir, _load_floats,
)


class StagedAssetTests(unittest.TestCase):
    def test_zimage_assets_match_loader(self) -> None:
        ad = assets_dir() / "zimage"
        cap = _load_floats(ad / "zimage_cap_pad_token.bin")
        self.assertEqual(cap.size, ld._DIT_DIM)
        mlp = _load_floats(ad / "zimage_temb_mlp.bin")
        self.assertEqual(mlp.size, 1024 * 256 + 1024 + 256 * 1024 + 256)

    def test_fluxklein_assets_match_loader(self) -> None:
        ad = assets_dir() / "fluxklein"
        expected = {
            "enc_cos.bin": 512 * 128, "enc_sin.bin": 512 * 128,
            "dit_cos.bin": 768 * 64, "dit_sin.bin": 768 * 64,
            "sigmas.bin": 5, "bn_mean.bin": 128, "bn_std.bin": 128,
            "temb.bin": 4 * 3072,
        }
        for name, size in expected.items():
            with self.subTest(asset=name):
                self.assertEqual(_load_floats(ad / name).size, size)


class ZImageMathTests(unittest.TestCase):
    def test_patchify_roundtrip(self) -> None:
        lat = np.random.default_rng(0).standard_normal(16 * 32 * 32).astype(np.float32)
        toks = ZImagePipeline._patchify(lat)
        self.assertEqual(toks.size, 256 * 64)
        back = ZImagePipeline._unpatchify(toks)
        self.assertTrue(np.array_equal(lat, back))

    def test_sigma_schedule_monotone_to_zero(self) -> None:
        sig = ZImagePipeline._sigma_schedule(9)
        self.assertEqual(sig.size, 10)
        self.assertAlmostEqual(sig[0], 1.0, places=5)
        self.assertEqual(sig[-1], 0.0)
        for i in range(9):
            self.assertGreaterEqual(sig[i], sig[i + 1])

    def test_t_freq_embedding_at_zero(self) -> None:
        emb = ZImagePipeline._t_freq_embedding(0.0)
        self.assertEqual(emb.size, 256)
        # cos(0)=1 for the first half, sin(0)=0 for the second.
        self.assertTrue(np.allclose(emb[:128], 1.0))
        self.assertTrue(np.allclose(emb[128:], 0.0))

    def test_id_grids_and_rope_shapes(self) -> None:
        self.assertEqual(ZImagePipeline._cap_ids().shape, (32, 3))
        self.assertEqual(ZImagePipeline._img_ids().shape, (256, 3))
        cos, sin = ZImagePipeline._rope_for_ids(ZImagePipeline._cap_ids())
        self.assertEqual(cos.size, 32 * 64)
        self.assertEqual(sin.size, 32 * 64)


class FluxKleinMaskTests(unittest.TestCase):
    def test_causal_and_pad(self) -> None:
        m = FluxKleinPipeline._build_enc_mask(3).reshape(32, 512, 512)
        self.assertEqual(m[0, 0, 0], 0.0)          # q0,k0 allowed
        self.assertEqual(m[0, 5, 0], 0.0)          # k<=q and k<n
        self.assertEqual(m[0, 5, 4], ld._K_NEG_MASK)   # k>=n → masked
        self.assertEqual(m[0, 1, 2], ld._K_NEG_MASK)   # k>q → masked
        self.assertTrue(np.array_equal(m[0], m[31]))   # 32 heads identical


class FluxKleinUnpatchifyTests(unittest.TestCase):
    def test_bn_denorm_and_placement(self) -> None:
        pipe = FluxKleinPipeline.__new__(FluxKleinPipeline)
        pipe._assets = {
            "bn_mean.bin": np.zeros(128, np.float32),
            "bn_std.bin": np.ones(128, np.float32),
        }
        packed = np.zeros(256 * 128, dtype=np.float32)
        # c4=0 → c=0, dy=0, dx=0; place at h=1, w=2.
        packed[(1 * 16 + 2) * 128 + 0] = 5.0
        out = pipe._unpack_bn_unpatchify(packed)
        self.assertEqual(out.size, 32 * 32 * 32)
        # out index = c*1024 + (h*2+dy)*32 + (w*2+dx) = 0 + 2*32 + 4 = 68
        self.assertEqual(out[68], 5.0)
        self.assertEqual(np.count_nonzero(out), 1)

    def test_bn_affine_applied(self) -> None:
        pipe = FluxKleinPipeline.__new__(FluxKleinPipeline)
        pipe._assets = {
            "bn_mean.bin": np.full(128, 2.0, np.float32),
            "bn_std.bin": np.full(128, 3.0, np.float32),
        }
        packed = np.zeros(256 * 128, dtype=np.float32)
        packed[0] = 1.0  # c4=0, h=0, w=0
        out = pipe._unpack_bn_unpatchify(packed)
        # 1.0 * 3.0 + 2.0 = 5.0 at index 0; rest = mean 2.0
        self.assertAlmostEqual(out[0], 5.0, places=5)


class AvailabilityTests(unittest.TestCase):
    def test_zimage_unavailable_on_empty_dir(self) -> None:
        self.assertFalse(ZImagePipeline("/no/such/zimage/dir").is_available())

    def test_fluxklein_unavailable_on_empty_dir(self) -> None:
        self.assertFalse(FluxKleinPipeline("/no/such/klein/dir").is_available())


class ConfigLitertDirTests(unittest.TestCase):
    def _settings(self):
        from box_chat.config import Settings
        return Settings()

    def test_add_dir_is_mru(self) -> None:
        s = self._settings()
        s.add_litert_dir("/a")
        s.add_litert_dir("/b")
        s.add_litert_dir("/a")  # re-add moves to front
        self.assertEqual(s.litert_diffusion_dirs[0], "/a")
        self.assertEqual(s.litert_last_dir, "/a")
        self.assertEqual(s.litert_diffusion_dirs.count("/a"), 1)

    def test_add_dir_caps_at_eight(self) -> None:
        s = self._settings()
        for i in range(12):
            s.add_litert_dir(f"/d{i}")
        self.assertEqual(len(s.litert_diffusion_dirs), 8)

    def test_forget_dir_updates_last(self) -> None:
        s = self._settings()
        s.add_litert_dir("/a")
        s.add_litert_dir("/b")   # last = /b, list = [/b, /a]
        s.forget_litert_dir("/b")
        self.assertNotIn("/b", s.litert_diffusion_dirs)
        self.assertEqual(s.litert_last_dir, "/a")

    def test_engine_default_is_sdcpp(self) -> None:
        self.assertEqual(self._settings().sd_engine, "sdcpp")


if __name__ == "__main__":
    unittest.main()
