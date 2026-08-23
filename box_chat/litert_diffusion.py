"""LiteRT .tflite diffusion pipelines: Z-Image Turbo (and FLUX.2-klein).

Text-to-image on the ``ai-edge-litert`` runtime Box already ships. Chunked
LiteRT graph pipelines (NOT stable-diffusion.cpp / GGUF) from the
litert-community HF repos — a separate image-gen backend alongside sd.cpp.

Numerics ported verbatim from Box Android's device-tested Kotlin
(ZImageEngine.kt / FluxKleinEngine.kt). The hard-won staged assets — the
t_embedder MLP (the "mesh bug" fix: the graphs' temb input is the MLP
OUTPUT, not the raw sinusoidal), the learned caption-pad token, RoPE/sigma
tables — are bundled in ``data/litert_diffusion/``.

Engine-tier and ``gi``-free. Sequential graph residency: one graph is
created/run/closed at a time. FP32 compute (fp16 produces NaNs in adaLN).
Model files download from litert-community (Z-Image ~10.6GB).
"""
from __future__ import annotations

import logging
import math
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "ZImagePipeline", "FluxKleinPipeline", "LiterDiffusionError", "assets_dir",
]

_ENC_SEQ = 64
_CAP_LEN = 32
_IMG_TOKENS = 256
_PATCH_DIM = 64
_HIDDEN = 2560
_DIT_DIM = 3840
_PAD_ID = 151643
_EMBED_ROWS = 151_936
_ROPE_THETA = 256.0
_T_SCALE = 1000.0
_VAE_SCALING = 0.3611
_VAE_SHIFT = 0.1159
_AXES_DIMS = (32, 48, 48)


class LiterDiffusionError(Exception):
    """Model files missing, asset mismatch, or graph inference failed."""


def assets_dir() -> Path:
    for d in (
        Path("/opt/box/litert_diffusion"),
        Path(__file__).resolve().parent.parent / "data" / "litert_diffusion",
    ):
        if d.is_dir():
            return d
    return Path(__file__).resolve().parent.parent / "data" / "litert_diffusion"


def _load_floats(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype="<f4").astype(np.float32)


class _GraphRunner:
    """Sequential-residency runner: one interpreter created/run/closed per
    call. Inputs are matched to the graph's declared input order."""

    def __init__(self, model_dir: Path, n_threads: int) -> None:
        self._dir = model_dir
        self._n_threads = n_threads

    def run(self, name: str, inputs: list[np.ndarray]) -> list[np.ndarray]:
        from ai_edge_litert.interpreter import Interpreter

        path = self._dir / name
        if not path.is_file():
            raise LiterDiffusionError(f"missing graph: {path}")
        interp = Interpreter(model_path=str(path), num_threads=self._n_threads)
        try:
            interp.allocate_tensors()
            in_details = interp.get_input_details()
            if len(inputs) != len(in_details):
                raise LiterDiffusionError(
                    f"{name}: expected {len(in_details)} inputs, got {len(inputs)}"
                )
            for det, arr in zip(in_details, inputs):
                interp.set_tensor(
                    det["index"], arr.astype(det["dtype"]).reshape(det["shape"]),
                )
            interp.invoke()
            return [
                np.array(interp.get_tensor(d["index"]), dtype=np.float32).reshape(-1)
                for d in interp.get_output_details()
            ]
        finally:
            del interp


class ZImagePipeline:
    """Z-Image-Turbo text-to-image. 256×256 output, 9 steps, guidance-free."""

    def __init__(self, model_dir: str | Path, n_threads: int | None = None) -> None:
        import os

        self._dir = Path(model_dir)
        self._threads = n_threads or max(1, (os.cpu_count() or 4) - 2)
        self._runner = _GraphRunner(self._dir, self._threads)
        self._tokenizer = None
        self._embed = None
        self._cap_pad = None
        self._temb_mlp = None
        self._lock = threading.Lock()
        self._cancelled = False

    def is_available(self) -> bool:
        return (self._dir / "qwen_enc.tflite").is_file() and (
            self._dir / "qwen_embed_fp16.bin"
        ).is_file()

    def cancel(self) -> None:
        self._cancelled = True

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        with self._lock:
            if self._tokenizer is not None:
                return
            ad = assets_dir() / "zimage"
            self._cap_pad = _load_floats(ad / "zimage_cap_pad_token.bin")
            if self._cap_pad.size != _DIT_DIM:
                raise LiterDiffusionError(
                    f"cap_pad_token size {self._cap_pad.size} != {_DIT_DIM}"
                )
            self._temb_mlp = _load_floats(ad / "zimage_temb_mlp.bin")
            expected = 1024 * 256 + 1024 + 256 * 1024 + 256
            if self._temb_mlp.size != expected:
                raise LiterDiffusionError(
                    f"temb_mlp size {self._temb_mlp.size} != {expected}"
                )
            self._embed = np.memmap(self._dir / "qwen_embed_fp16.bin", dtype="<f2", mode="r")
            from .qwen_tokenizer import QwenBpeTokenizer

            self._tokenizer = QwenBpeTokenizer(
                vocab_file=self._dir / "qwen_vocab.txt",
                merges_file=self._dir / "qwen_merges.txt",
                specials_file=self._dir / "qwen_special.txt",
            )

    def generate(self, prompt: str, seed: int = 0, steps: int = 9,
                 guidance: float = 0.0, on_progress=None):
        """Generate a 256×256 PIL image. steps=9, guidance=0 = official Turbo."""
        from PIL import Image  # noqa: F401

        self._ensure_loaded()
        self._cancelled = False

        def prog(stage, frac):
            if on_progress:
                on_progress(stage, frac)

        use_cfg = guidance > 0.0
        branches = 2 if use_cfg else 1
        enc_frac = 0.15 * branches
        step_frac = (0.93 - enc_frac) / steps

        prog("Encoding prompt", 0.0)
        cap_pos, n_pos = self._encode_branch(prompt)
        cap_neg = None
        n_neg = 0
        if use_cfg:
            self._check_cancel()
            prog("Encoding negative prompt", 0.15)
            cap_neg, n_neg = self._encode_branch("")

        latent = self._gaussian(16 * 32 * 32, seed)
        sig = self._sigma_schedule(steps)

        for step in range(steps):
            self._check_cancel()
            base = enc_frac + step * step_frac
            temb = self._t_adaln_input(1.0 - sig[step])
            tokens = self._patchify(latent)
            prog(f"Step {step + 1}/{steps}", base)
            pos = self._dit_branch(tokens, cap_pos, n_pos, temb)
            if use_cfg:
                self._check_cancel()
                neg = self._dit_branch(tokens, cap_neg, n_neg, temb)
                pred = pos + guidance * (pos - neg)
            else:
                pred = pos
            dsig = sig[step + 1] - sig[step]
            noise_img = self._unpatchify(pred)
            latent = latent + dsig * (-noise_img)

        self._check_cancel()
        prog("Decoding image", 0.95)
        lat = latent / _VAE_SCALING + _VAE_SHIFT
        rgb = self._runner.run("zvae.tflite", [lat])[0]
        prog("Done", 1.0)
        return self._to_image(rgb)

    def _encode_branch(self, prompt: str) -> tuple[np.ndarray, int]:
        templated = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
        raw_ids = self._tokenizer.encode(templated)
        n = min(len(raw_ids), _ENC_SEQ)
        ids = np.full(_ENC_SEQ, _PAD_ID, dtype=np.int64)
        ids[:n] = raw_ids[:n]
        embeds = self._embed_lookup(ids)
        cap_feats = self._runner.run("qwen_enc.tflite", [embeds])[0]
        cap32 = np.zeros(_CAP_LEN * _HIDDEN, dtype=np.float32)
        n_cap = min(n, _CAP_LEN)
        cap32[: n_cap * _HIDDEN] = cap_feats[: n_cap * _HIDDEN]
        return cap32, n_cap

    def _dit_branch(self, tokens, cap32, n_cap, temb) -> np.ndarray:
        cap_emb = self._runner.run("z_embc.tflite", [cap32])[0].copy()
        for i in range(n_cap, _CAP_LEN):
            cap_emb[i * _DIT_DIM:(i + 1) * _DIT_DIM] = self._cap_pad
        ccos, csin = self._rope_for_ids(self._cap_ids())
        cap_ref = self._runner.run("z_refc.tflite", [cap_emb, ccos, csin])[0]
        self._check_cancel()

        icos, isin = self._rope_for_ids(self._img_ids())
        img_emb = self._runner.run("z_embx.tflite", [tokens])[0]
        img_ref = self._runner.run("z_refx.tflite", [img_emb, icos, isin, temb])[0]
        self._check_cancel()

        hidden = np.concatenate([img_ref, cap_ref]).astype(np.float32)
        ucos = np.concatenate([icos, ccos]).astype(np.float32)
        usin = np.concatenate([isin, csin]).astype(np.float32)
        for i in range(6):
            self._check_cancel()
            hidden = self._runner.run(
                f"zc_main{i}.tflite", [hidden, ucos, usin, temb]
            )[0]
        out = self._runner.run("zc_final.tflite", [hidden, temb])[0]
        return out[: _IMG_TOKENS * _PATCH_DIM].copy()

    @staticmethod
    def _sigma_schedule(steps: int) -> np.ndarray:
        out = np.zeros(steps + 1, dtype=np.float32)
        for i in range(steps):
            s = 1.0 - i * (1.0 - 1.0 / steps) / (steps - 1)
            out[i] = 3.0 * s / (1.0 + 2.0 * s)
        out[steps] = 0.0
        return out

    @staticmethod
    def _t_freq_embedding(t_norm: float) -> np.ndarray:
        half = 128
        out = np.zeros(256, dtype=np.float32)
        for i in range(half):
            freq = math.exp(-math.log(10000.0) * i / half)
            arg = t_norm * _T_SCALE * freq
            out[i] = math.cos(arg)
            out[half + i] = math.sin(arg)
        return out

    def _t_adaln_input(self, t_norm: float) -> np.ndarray:
        """adaln_input = t_embedder MLP over the sinusoidal — NOT the raw
        sinusoidal. This is the mesh-bug fix."""
        sinus = self._t_freq_embedding(t_norm)
        m = self._temb_mlp
        w0 = 0
        b0 = w0 + 1024 * 256
        w2 = b0 + 1024
        b2 = w2 + 256 * 1024
        w0m = m[w0:b0].reshape(1024, 256)
        b0v = m[b0:w2]
        w2m = m[w2:b2].reshape(256, 1024)
        b2v = m[b2:b2 + 256]
        acc = w0m @ sinus + b0v
        hid = acc / (1.0 + np.exp(-acc))
        out = w2m @ hid + b2v
        return out.astype(np.float32)

    @staticmethod
    def _cap_ids() -> np.ndarray:
        return np.array([[1 + i, 0, 0] for i in range(_CAP_LEN)], dtype=np.int64)

    @staticmethod
    def _img_ids() -> np.ndarray:
        return np.array(
            [[_CAP_LEN + 1, i // 16, i % 16] for i in range(_IMG_TOKENS)],
            dtype=np.int64,
        )

    @staticmethod
    def _rope_for_ids(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = ids.shape[0]
        cos_out = np.zeros(s * 64, dtype=np.float32)
        sin_out = np.zeros(s * 64, dtype=np.float32)
        for p in range(s):
            off = 0
            for a, d in enumerate(_AXES_DIMS):
                halves = d // 2
                for k in range(halves):
                    inv = 1.0 / (_ROPE_THETA ** (2.0 * k / d))
                    ang = ids[p][a] * inv
                    cos_out[p * 64 + off + k] = math.cos(ang)
                    sin_out[p * 64 + off + k] = math.sin(ang)
                off += halves
        return cos_out, sin_out

    @staticmethod
    def _patchify(latent: np.ndarray) -> np.ndarray:
        out = np.zeros(_IMG_TOKENS * _PATCH_DIM, dtype=np.float32)
        for ht in range(16):
            for wt in range(16):
                tok_base = (ht * 16 + wt) * _PATCH_DIM
                for ph in range(2):
                    for pw in range(2):
                        for c in range(16):
                            out[tok_base + (ph * 2 + pw) * 16 + c] = latent[
                                c * 32 * 32 + (ht * 2 + ph) * 32 + (wt * 2 + pw)
                            ]
        return out

    @staticmethod
    def _unpatchify(tokens: np.ndarray) -> np.ndarray:
        out = np.zeros(16 * 32 * 32, dtype=np.float32)
        for ht in range(16):
            for wt in range(16):
                tok_base = (ht * 16 + wt) * _PATCH_DIM
                for ph in range(2):
                    for pw in range(2):
                        for c in range(16):
                            out[c * 32 * 32 + (ht * 2 + ph) * 32 + (wt * 2 + pw)] = (
                                tokens[tok_base + (ph * 2 + pw) * 16 + c]
                            )
        return out

    def _embed_lookup(self, ids: np.ndarray) -> np.ndarray:
        table = self._embed
        out = np.zeros(_ENC_SEQ * _HIDDEN, dtype=np.float32)
        for s in range(_ENC_SEQ):
            row = int(ids[s])
            if not (0 <= row < _EMBED_ROWS):
                raise LiterDiffusionError(f"token id {row} out of range")
            base = row * _HIDDEN
            out[s * _HIDDEN:(s + 1) * _HIDDEN] = np.asarray(
                table[base:base + _HIDDEN], dtype=np.float32
            )
        return out

    @staticmethod
    def _gaussian(n: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(n).astype(np.float32)

    @staticmethod
    def _to_image(rgb: np.ndarray):
        from PIL import Image

        plane = 256 * 256
        chw = rgb[: 3 * plane].reshape(3, 256, 256)
        arr = np.clip(chw / 2.0 + 0.5, 0.0, 1.0)
        hwc = np.transpose(arr, (1, 2, 0))
        return Image.fromarray((hwc * 255.0 + 0.5).astype(np.uint8), "RGB")

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise LiterDiffusionError("generation cancelled")


# ── FLUX.2-klein ────────────────────────────────────────────────────────────
_K_MAX_SEQ = 512
_K_IMG_TOKENS = 256      # 16×16 latent patches
_K_LATENT_CH = 128       # packed/patchified channels
_K_NUM_STEPS = 4
_K_HIDDEN = 2560
_K_ENC_HEADS = 32
_K_INTERLEAVE = 7680     # 3 taps × 2560
_K_DIT_DIM = 3072
_K_PAD_ID = 151643       # <|endoftext|>
_K_EMBED_ROWS = 151_936
_K_NEG_MASK = -3.4028235e38  # float32 min — matches the HF attention-mask fill


class FluxKleinPipeline:
    """FLUX.2-klein-4B text-to-image over 12 chunked LiteRT graphs.

    Ported verbatim from Box Android's device-tested FluxKleinEngine.kt (which
    itself mirrors the desktop reference that reproduced the model card's
    sample). Sequential graph residency (one .tflite loaded/run/closed at a
    time), FP32 compute, 4-step guidance-free flow-matching. All
    prompt-independent tensors (RoPE tables, per-step temb, sigmas, bn stats)
    are staged assets in ``data/litert_diffusion/fluxklein/``.

    Live-verified on Linux (2026-07-16): rendered a coherent 256×256 image from
    the real ~7GB klein weights in ~130s CPU-only. The model download isn't
    bundled with the source tree — fetch it via the in-app LiteRT downloader.
    """

    def __init__(self, model_dir: str | Path, n_threads: int | None = None) -> None:
        import os

        self._dir = Path(model_dir)
        self._threads = n_threads or max(1, (os.cpu_count() or 4) - 2)
        self._runner = _GraphRunner(self._dir, self._threads)
        self._tokenizer = None
        self._embed = None
        self._assets: dict[str, np.ndarray] = {}
        self._temb = None
        self._lock = threading.Lock()
        self._cancelled = False

    def is_available(self) -> bool:
        return (self._dir / "ke_enc0.tflite").is_file() and (
            self._dir / "qwen_embed_fp16.bin"
        ).is_file()

    def cancel(self) -> None:
        self._cancelled = True

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None:
            return
        with self._lock:
            if self._tokenizer is not None:
                return
            ad = assets_dir() / "fluxklein"
            expected = {
                "enc_cos.bin": _K_MAX_SEQ * 128,
                "enc_sin.bin": _K_MAX_SEQ * 128,
                "dit_cos.bin": 768 * 64,
                "dit_sin.bin": 768 * 64,
                "sigmas.bin": _K_NUM_STEPS + 1,
                "bn_mean.bin": _K_LATENT_CH,
                "bn_std.bin": _K_LATENT_CH,
            }
            for name, size in expected.items():
                arr = _load_floats(ad / name)
                if arr.size != size:
                    raise LiterDiffusionError(
                        f"{name} size {arr.size} != {size}"
                    )
                self._assets[name] = arr
            temb_flat = _load_floats(ad / "temb.bin")
            if temb_flat.size != _K_NUM_STEPS * _K_DIT_DIM:
                raise LiterDiffusionError(
                    f"temb.bin size {temb_flat.size} != {_K_NUM_STEPS * _K_DIT_DIM}"
                )
            self._temb = temb_flat.reshape(_K_NUM_STEPS, _K_DIT_DIM)
            self._embed = np.memmap(
                self._dir / "qwen_embed_fp16.bin", dtype="<f2", mode="r"
            )
            from .qwen_tokenizer import QwenBpeTokenizer

            self._tokenizer = QwenBpeTokenizer(
                vocab_file=self._dir / "qwen_vocab.txt",
                merges_file=self._dir / "qwen_merges.txt",
                specials_file=self._dir / "qwen_special.txt",
            )

    def generate(self, prompt: str, seed: int = 0, on_progress=None):
        """Generate a 256×256 PIL image (4 steps, guidance-free)."""
        self._ensure_loaded()
        self._cancelled = False

        def prog(stage, frac):
            if on_progress:
                on_progress(stage, frac)

        enc_cos = self._assets["enc_cos.bin"]
        enc_sin = self._assets["enc_sin.bin"]
        dit_cos = self._assets["dit_cos.bin"]
        dit_sin = self._assets["dit_sin.bin"]
        sigmas = self._assets["sigmas.bin"]

        # ── 1. prompt → ids → embeds + causal/pad mask
        prog("Encoding prompt", 0.0)
        templated = (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        raw_ids = self._tokenizer.encode(templated)
        n_valid = min(len(raw_ids), _K_MAX_SEQ)
        ids = np.full(_K_MAX_SEQ, _K_PAD_ID, dtype=np.int64)
        ids[:n_valid] = raw_ids[:n_valid]
        embeds = self._embed_lookup(ids)
        mask = self._build_enc_mask(n_valid)

        # ── 2. text encoder: 3 chunks, taps interleaved to 7680 channels
        hidden = embeds
        prompt_embeds = np.zeros(_K_MAX_SEQ * _K_INTERLEAVE, dtype=np.float32)
        pe = prompt_embeds.reshape(_K_MAX_SEQ, _K_INTERLEAVE)
        for i in range(3):
            self._check_cancel()
            prog(f"Text encoder {i + 1}/3", 0.02 + 0.09 * i)
            hidden = self._runner.run(
                f"ke_enc{i}.tflite", [hidden, mask, enc_cos, enc_sin]
            )[0]
            pe[:, i * _K_HIDDEN:(i + 1) * _K_HIDDEN] = hidden.reshape(
                _K_MAX_SEQ, _K_HIDDEN
            )

        # ── 3. seeded gaussian latent, packed [256,128]
        latents = self._gaussian(_K_IMG_TOKENS * _K_LATENT_CH, seed)

        # ── 4. denoise: 4 steps × (prep → 2 double → 4 single → final)
        for step in range(_K_NUM_STEPS):
            base = 0.30 + 0.16 * step
            self._check_cancel()
            prog(f"Step {step + 1}/{_K_NUM_STEPS} · prep", base)
            temb = self._temb[step]
            prep = self._runner.run(
                "kc_prep.tflite", [latents, prompt_embeds, temb]
            )
            image, text, mod_img, mod_txt, mod_single = prep[:5]
            for i in range(2):
                self._check_cancel()
                prog(f"Step {step + 1}/{_K_NUM_STEPS} · double {i + 1}/2",
                     base + 0.03 + 0.02 * i)
                o = self._runner.run(
                    f"kc_double{i}.tflite",
                    [image, text, dit_cos, dit_sin, mod_img, mod_txt],
                )
                image, text = o[0], o[1]
            # joint = [text; image] (text first)
            joint = np.concatenate([text, image]).astype(np.float32)
            for i in range(4):
                self._check_cancel()
                prog(f"Step {step + 1}/{_K_NUM_STEPS} · single {i + 1}/4",
                     base + 0.08 + 0.015 * i)
                joint = self._runner.run(
                    f"kc_single{i}.tflite", [joint, dit_cos, dit_sin, mod_single]
                )[0]
            self._check_cancel()
            prog(f"Step {step + 1}/{_K_NUM_STEPS} · final", base + 0.145)
            noise_pred = self._runner.run("kc_final.tflite", [joint, temb])[0]
            dsigma = sigmas[step + 1] - sigmas[step]
            latents = latents + dsigma * noise_pred[: latents.size]

        # ── 5. unpack → bn denorm → unpatchify → VAE → image
        self._check_cancel()
        prog("Decoding image", 0.95)
        latent_img = self._unpack_bn_unpatchify(latents)
        rgb = self._runner.run("kv_vae.tflite", [latent_img])[0]
        prog("Done", 1.0)
        return ZImagePipeline._to_image(rgb)

    def _embed_lookup(self, ids: np.ndarray) -> np.ndarray:
        table = self._embed
        out = np.zeros(_K_MAX_SEQ * _K_HIDDEN, dtype=np.float32)
        for s in range(_K_MAX_SEQ):
            row = int(ids[s])
            if not (0 <= row < _K_EMBED_ROWS):
                raise LiterDiffusionError(f"token id {row} out of range")
            base = row * _K_HIDDEN
            out[s * _K_HIDDEN:(s + 1) * _K_HIDDEN] = np.asarray(
                table[base:base + _K_HIDDEN], dtype=np.float32
            )
        return out

    @staticmethod
    def _build_enc_mask(n_valid: int) -> np.ndarray:
        """Causal + right-padding additive mask, pre-expanded across 32 heads."""
        q = np.arange(_K_MAX_SEQ)[:, None]
        k = np.arange(_K_MAX_SEQ)[None, :]
        allowed = (k <= q) & (k < n_valid)
        plane = np.where(allowed, 0.0, _K_NEG_MASK).astype(np.float32)
        return np.broadcast_to(
            plane, (_K_ENC_HEADS, _K_MAX_SEQ, _K_MAX_SEQ)
        ).reshape(-1).copy()

    def _unpack_bn_unpatchify(self, packed: np.ndarray) -> np.ndarray:
        """[256,128] packed latents → ×bn_std+bn_mean → unpatchify → [32,32,32]."""
        bn_mean = self._assets["bn_mean.bin"]
        bn_std = self._assets["bn_std.bin"]
        pk = packed.reshape(_K_IMG_TOKENS, _K_LATENT_CH)  # [h*w, c4]
        out = np.zeros(32 * 32 * 32, dtype=np.float32)
        for c4 in range(_K_LATENT_CH):
            c = c4 // 4
            dy = (c4 % 4) // 2
            dx = c4 % 2
            vals = pk[:, c4] * bn_std[c4] + bn_mean[c4]  # [256] in raster h*16+w
            grid = vals.reshape(16, 16)
            for h in range(16):
                base = c * 32 * 32 + (h * 2 + dy) * 32
                out[base + dx:base + 32:2] = grid[h]
        return out

    @staticmethod
    def _gaussian(n: int, seed: int) -> np.ndarray:
        return np.random.default_rng(seed).standard_normal(n).astype(np.float32)

    def _check_cancel(self) -> None:
        if self._cancelled:
            raise LiterDiffusionError("generation cancelled")
