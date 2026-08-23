"""Component-model bundles for the sd.cpp engine (Z-Image, FLUX.2-klein).

sd.cpp runs these natively from split GGUF components — a diffusion
transformer + a VAE + an LLM text encoder — at ANY resolution, unlike the
fixed-256×256 LiteRT graphs (which stay available as their own engine).

Bundles reuse the LiteRT multi-file downloader (resumable, size-verified)
and land in ``MODELS_DIR/sd_components/<key>/``. File sizes are exact
bytes from the HuggingFace API (2026-07-16); all repos are public.
Pure stdlib — no gi.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from .config import MODELS_DIR
from .litert_diffusion_models import LiterFile, LiterModel, download_model

__all__ = [
    "SD_BUNDLES",
    "SDBundle",
    "bundle_dir",
    "bundle_for_dir",
    "download_bundle",
    "installed_bundles",
    "is_complete",
]

SD_COMPONENTS_DIR = MODELS_DIR / "sd_components"


def _hf(repo: str, path: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/main/{path}"


@dataclasses.dataclass(frozen=True)
class SDBundle:
    """One downloadable component set + its generation defaults."""

    key: str
    name: str
    info: str
    files: tuple[LiterFile, ...]
    diffusion_file: str
    vae_file: str
    llm_file: str
    default_steps: int
    default_cfg: float
    default_sampler: str = "euler"

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def as_liter_model(self) -> LiterModel:
        """Adapter for the shared multi-file downloader."""
        return LiterModel(
            key=self.key, name=self.name, engine="sdcpp",
            info=self.info, files=self.files,
        )


SD_BUNDLES: tuple[SDBundle, ...] = (
    SDBundle(
        key="zimage-turbo-gguf",
        name="Z-Image Turbo (GGUF, any resolution)",
        info="6B distilled, 8 steps, cfg 1.0 — ~6.1GB total",
        files=(
            LiterFile(
                name="z_image_turbo-Q4_0.gguf",
                url=_hf("leejet/Z-Image-Turbo-GGUF",
                        "z_image_turbo-Q4_0.gguf"),
                size=3_683_370_944,
            ),
            LiterFile(
                name="ae.safetensors",
                url=_hf("black-forest-labs/FLUX.1-schnell",
                        "ae.safetensors"),
                size=335_304_388,
            ),
            LiterFile(
                name="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                url=_hf("unsloth/Qwen3-4B-Instruct-2507-GGUF",
                        "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"),
                size=2_497_281_120,
            ),
        ),
        diffusion_file="z_image_turbo-Q4_0.gguf",
        vae_file="ae.safetensors",
        llm_file="Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        default_steps=8,
        default_cfg=1.0,
    ),
    SDBundle(
        key="flux2-klein-gguf",
        name="FLUX.2 klein 4B (GGUF, any resolution)",
        info="4B distilled, 4 steps, cfg 1.0 — ~4.9GB total",
        files=(
            LiterFile(
                name="flux-2-klein-4b-Q4_0.gguf",
                url=_hf("leejet/FLUX.2-klein-4B-GGUF",
                        "flux-2-klein-4b-Q4_0.gguf"),
                size=2_460_378_560,
            ),
            LiterFile(
                name="flux2-vae.safetensors",
                url=_hf("Comfy-Org/flux2-klein-4B",
                        "split_files/vae/flux2-vae.safetensors"),
                size=336_211_292,
            ),
            LiterFile(
                name="Qwen3-4B-Q4_K_M.gguf",
                url=_hf("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
                size=2_497_281_312,
            ),
        ),
        diffusion_file="flux-2-klein-4b-Q4_0.gguf",
        vae_file="flux2-vae.safetensors",
        llm_file="Qwen3-4B-Q4_K_M.gguf",
        default_steps=4,
        default_cfg=1.0,
    ),
)

_BY_KEY = {b.key: b for b in SD_BUNDLES}


def bundle_dir(key: str) -> Path:
    return SD_COMPONENTS_DIR / key


def is_complete(bundle: SDBundle, d: Path | None = None) -> bool:
    d = d or bundle_dir(bundle.key)
    for f in bundle.files:
        p = d / f.name
        try:
            if p.stat().st_size != f.size:
                return False
        except OSError:
            return False
    return True


def installed_bundles() -> list[SDBundle]:
    return [b for b in SD_BUNDLES if is_complete(b)]


def bundle_for_dir(path: str | Path) -> SDBundle | None:
    return _BY_KEY.get(Path(path).name)


def download_bundle(
    bundle: SDBundle,
    *,
    on_overall=None,
    on_status=None,
    is_cancelled=None,
) -> Path:
    """Fetch every file (resumable, size-verified). Returns the dir."""
    dest = bundle_dir(bundle.key)
    download_model(
        bundle.as_liter_model(), dest,
        on_overall=on_overall, on_status=on_status,
        is_cancelled=is_cancelled,
    )
    return dest
