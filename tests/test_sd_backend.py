"""stable-diffusion.cpp backend: argv translation + binary discovery.

Covers box_chat/sd_backend.py's pure surface — build_argv sentinel handling
(‑1 / "" / "auto" omit their flags), the required-flag set, and find_sd_binary
discovery. A live single-image generation runs only if an SD model is pointed
to via BOX_SD_TEST_MODEL (none is bundled in the test tree).
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from box_chat import sd_backend as sd

REPO = Path(__file__).resolve().parent.parent


def _find_sd_binary() -> Path | None:
    try:
        return sd.find_sd_binary("cpu")
    except sd.SDError:
        return None


class ConstantsTests(unittest.TestCase):
    def test_option_lists_populated(self) -> None:
        self.assertIn("euler_a", sd.SAMPLERS)
        self.assertIn("karras", sd.SCHEDULERS)
        self.assertIn("auto", sd.WEIGHT_TYPES)

    def test_error_carries_log_tail(self) -> None:
        e = sd.SDError("boom", "last lines")
        self.assertEqual(e.log_tail, "last lines")


class BuildArgvTests(unittest.TestCase):
    def _argv(self, **over) -> list[str]:
        params = sd.SDGenParams(model="/m/model.gguf", prompt="a cat", **over)
        return sd.build_argv(Path("/bin/sd-cli"), params, "/out/img.png")

    def test_required_flags_present(self) -> None:
        a = self._argv()
        self.assertEqual(a[0], "/bin/sd-cli")
        for flag in ("-M", "-m", "-p", "-o", "-W", "-H", "--steps",
                     "--cfg-scale", "--sampling-method", "--scheduler", "-s", "-b"):
            self.assertIn(flag, a)
        self.assertEqual(a[a.index("-M") + 1], "img_gen")
        self.assertEqual(a[a.index("-m") + 1], "/m/model.gguf")
        self.assertEqual(a[a.index("-p") + 1], "a cat")
        self.assertEqual(a[a.index("-o") + 1], "/out/img.png")

    def test_default_sentinels_omit_flags(self) -> None:
        a = self._argv()  # defaults: neg="", clip_skip=-1, weight=auto, threads=-1
        self.assertNotIn("-n", a)
        self.assertNotIn("--clip-skip", a)
        self.assertNotIn("--type", a)
        self.assertNotIn("-t", a)
        self.assertNotIn("--init-img", a)
        self.assertNotIn("--control-net", a)
        # guidance default 3.5 (>0) IS included
        self.assertIn("--guidance", a)

    def test_negative_prompt_and_clip_skip(self) -> None:
        a = self._argv(negative_prompt="blurry", clip_skip=2)
        self.assertEqual(a[a.index("-n") + 1], "blurry")
        self.assertEqual(a[a.index("--clip-skip") + 1], "2")

    def test_weight_type_and_threads(self) -> None:
        a = self._argv(weight_type="q4_0", threads=6)
        self.assertEqual(a[a.index("--type") + 1], "q4_0")
        self.assertEqual(a[a.index("-t") + 1], "6")

    def test_img2img_adds_init_and_strength(self) -> None:
        a = self._argv(init_image="/i/in.png", strength=0.6)
        self.assertEqual(a[a.index("--init-img") + 1], "/i/in.png")
        self.assertEqual(a[a.index("--strength") + 1], "0.6")

    def test_controlnet_triple(self) -> None:
        a = self._argv(control_net="/c/cn.safetensors", control_image="/c/edge.png",
                        control_strength=0.8)
        self.assertIn("--control-net", a)
        self.assertIn("--control-image", a)
        self.assertEqual(a[a.index("--control-strength") + 1], "0.8")

    def test_controlnet_requires_both(self) -> None:
        # net without image → omitted entirely.
        a = self._argv(control_net="/c/cn.safetensors")
        self.assertNotIn("--control-net", a)

    def test_boolean_flags(self) -> None:
        a = self._argv(vae_tiling=True, vae_on_cpu=True, clip_on_cpu=True,
                       diffusion_fa=True)
        for flag in ("--vae-tiling", "--vae-on-cpu", "--clip-on-cpu", "--diffusion-fa"):
            self.assertIn(flag, a)

    def test_optional_paths(self) -> None:
        a = self._argv(vae="/v/vae.pt", lora_dir="/l", upscale_model="/u/up.pth")
        self.assertEqual(a[a.index("--vae") + 1], "/v/vae.pt")
        self.assertEqual(a[a.index("--lora-model-dir") + 1], "/l")
        self.assertEqual(a[a.index("--upscale-model") + 1], "/u/up.pth")


class FindBinaryTests(unittest.TestCase):
    def test_env_override(self) -> None:
        bin_ = _find_sd_binary()
        if bin_ is None:
            self.skipTest("no sd-cli binary")
        os.environ["BOX_SD_DIR"] = str(bin_.parent)
        try:
            self.assertEqual(sd.find_sd_binary("cpu"), bin_)
        finally:
            del os.environ["BOX_SD_DIR"]

    def test_missing_raises(self) -> None:
        # A variant with no bundled build and no cpu fallback → nothing found.
        with self.assertRaises(sd.SDError):
            sd.find_sd_binary("rocm")


_SD_MODEL = os.environ.get("BOX_SD_TEST_MODEL", "")


@unittest.skipUnless(
    _SD_MODEL and Path(_SD_MODEL).is_file() and _find_sd_binary() is not None,
    "set BOX_SD_TEST_MODEL to an SD .gguf to run the live generation test",
)
class LiveGenerateTests(unittest.TestCase):
    def test_single_image(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "out.png")
            params = sd.SDGenParams(
                model=_SD_MODEL, prompt="a red apple on a table",
                width=256, height=256, steps=4, seed=42,
            )
            written = sd.SDBackend("cpu").generate(params, out)
            self.assertTrue(written)
            self.assertTrue(Path(written[0]).is_file())


if __name__ == "__main__":
    unittest.main()


class ComponentModeArgvTests(unittest.TestCase):
    """New in 0.4.0: component layout + caching + preview + offload."""

    def _argv(self, **kw) -> list[str]:
        from box_chat.sd_backend import SDGenParams, build_argv
        from pathlib import Path
        p = SDGenParams(model="/m/ckpt.gguf", prompt="cat", **kw)
        return build_argv(Path("/bin/sd-cli"), p, "/out/x.png")

    def test_checkpoint_mode_uses_dash_m(self):
        a = self._argv()
        self.assertIn("-m", a)
        self.assertNotIn("--diffusion-model", a)

    def test_component_mode_replaces_checkpoint(self):
        a = self._argv(diffusion_model="/m/dit.gguf", vae="/m/ae.st",
                       llm="/m/qwen.gguf")
        self.assertIn("--diffusion-model", a)
        self.assertIn("--llm", a)
        self.assertIn("--vae", a)
        self.assertNotIn("-m", a)

    def test_cache_and_offload_and_preview(self):
        a = self._argv(cache_mode="easycache", cache_option="threshold=0.3",
                       offload_to_cpu=True, preview_mode="proj",
                       preview_path="/out/p.png", preview_interval=2)
        self.assertIn("--cache-mode", a)
        self.assertIn("easycache", a)
        self.assertIn("--cache-option", a)
        self.assertIn("--offload-to-cpu", a)
        self.assertIn("--preview", a)
        self.assertIn("--preview-interval", a)
        # cache_mode "none" and empty preview add nothing
        b = self._argv(cache_mode="none")
        self.assertNotIn("--cache-mode", b)
        self.assertNotIn("--preview", b)

    def test_seed_regex(self):
        from box_chat.sd_backend import _SEED_RE
        m = _SEED_RE.search("[INFO ] generating image: 1/1 - seed 42")
        self.assertIsNotNone(m)
        self.assertEqual(m.group(1), "42")


class SDBundleTests(unittest.TestCase):
    def test_bundles_wellformed(self):
        from box_chat.sd_components import SD_BUNDLES, bundle_dir
        self.assertEqual(len(SD_BUNDLES), 2)
        for b in SD_BUNDLES:
            names = {f.name for f in b.files}
            self.assertIn(b.diffusion_file, names)
            self.assertIn(b.vae_file, names)
            self.assertIn(b.llm_file, names)
            for f in b.files:
                self.assertTrue(f.url.startswith("https://huggingface.co/"))
                self.assertGreater(f.size, 1_000_000)
            self.assertGreater(b.default_steps, 0)
            self.assertIn("sd_components", str(bundle_dir(b.key)))

    def test_incomplete_bundle_detected(self):
        from box_chat.sd_components import SD_BUNDLES, is_complete
        import tempfile
        from pathlib import Path
        d = Path(tempfile.mkdtemp())
        self.assertFalse(is_complete(SD_BUNDLES[0], d))


class InpaintHiresArgvTests(unittest.TestCase):
    def _argv(self, **kw) -> list[str]:
        from box_chat.sd_backend import SDGenParams, build_argv
        from pathlib import Path
        p = SDGenParams(model="/m/ckpt.gguf", prompt="cat", **kw)
        return build_argv(Path("/bin/sd-cli"), p, "/out/x.png")

    def test_mask_requires_init_image(self):
        # mask without init image adds nothing
        a = self._argv(mask_image="/m/mask.png")
        self.assertNotIn("--mask", a)
        b = self._argv(init_image="/m/src.png", mask_image="/m/mask.png")
        self.assertIn("--mask", b)
        self.assertIn("--init-img", b)

    def test_hires_fix_flags(self):
        a = self._argv(hires_scale=2.0, hires_steps=12,
                       hires_upscaler="Lanczos")
        self.assertIn("--hires-scale", a)
        self.assertIn("--hires-steps", a)
        self.assertIn("--hires-upscaler", a)
        b = self._argv(hires_scale=0.0)
        self.assertNotIn("--hires-scale", b)
