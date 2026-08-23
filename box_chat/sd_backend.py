"""stable-diffusion.cpp image generation via a sandboxed CLI subprocess.

Box wraps leejet/stable-diffusion.cpp's ``sd-cli`` as a one-shot subprocess
(one process per image, not a server). Engine-tier and ``gi``-free; the UI
drives ``generate`` on a worker thread.

Security: the subprocess runs under :mod:`box_chat.sandbox` with a
compute-subprocess policy — read-only on the model files + the bundled
binary dir, read-write only on the output directory, and NO network at all.

Flag ground truth: brain/box_linux/REFERENCE_sd_cli_flags.txt.
Pinned: master-778-c00a9e9.
"""
from __future__ import annotations

import dataclasses
import logging
import os
import re
import threading
from pathlib import Path

from .sandbox import Policy, launch

log = logging.getLogger(__name__)

__all__ = ["SDGenParams", "SDBackend", "SDError", "find_sd_binary"]

SAMPLERS = [
    "euler", "euler_a", "heun", "dpm2", "dpm++2s_a", "dpm++2m", "dpm++2mv2",
    "dpm++2m_sde", "ipndm", "ipndm_v", "lcm", "ddim_trailing", "tcd",
    "res_multistep", "euler_cfg_pp", "euler_a_cfg_pp",
]
SCHEDULERS = [
    "discrete", "karras", "exponential", "ays", "gits", "smoothstep",
    "sgm_uniform", "simple", "kl_optimal", "beta",
]
WEIGHT_TYPES = [
    "auto", "f32", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0", "q3_K", "q2_K",
]


class SDError(Exception):
    """sd-cli missing, failed to launch, or exited non-zero."""

    def __init__(self, message: str, log_tail: str = "") -> None:
        super().__init__(message)
        self.log_tail = log_tail


def find_sd_binary(variant: str = "cpu") -> Path:
    candidates: list[Path] = []
    if env := os.environ.get("BOX_SD_DIR"):
        candidates.append(Path(env))
    suffix = "" if variant == "cpu" else f"-{variant}"
    candidates.append(Path(f"/opt/box/libexec/stable-diffusion.cpp{suffix}"))
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "vendor" / f"stable-diffusion.cpp{suffix}")
    if variant == "cpu":
        candidates.append(repo_root / "vendor" / "stable-diffusion.cpp")
    for d in candidates:
        binary = d / "sd-cli"
        if binary.is_file() and os.access(binary, os.X_OK):
            return binary
    raise SDError(
        f"no sd-cli binary found for variant {variant!r} "
        f"(searched: {', '.join(str(c) for c in candidates)})"
    )


CACHE_MODES = ["none", "ucache", "easycache", "dbcache", "taylorseer"]


@dataclasses.dataclass
class SDGenParams:
    """One image-generation request.

    Two model layouts:
    - checkpoint: ``model`` = a single full checkpoint (SD1.5/2.1 …).
    - components: ``diffusion_model`` + ``vae`` + ``llm`` (Z-Image,
      FLUX.2-klein, Qwen-Image … — the split GGUF layout sd.cpp
      supports natively). ``model`` stays empty then.
    """
    model: str
    prompt: str
    negative_prompt: str = ""
    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    guidance: float = 3.5
    seed: int = -1
    sampler: str = "euler_a"
    scheduler: str = "discrete"
    batch_count: int = 1
    clip_skip: int = -1
    weight_type: str = "auto"
    threads: int = -1
    init_image: str = ""
    strength: float = 0.75
    vae_tiling: bool = False
    vae_on_cpu: bool = False
    clip_on_cpu: bool = False
    diffusion_fa: bool = False
    vae: str = ""
    lora_dir: str = ""
    upscale_model: str = ""
    control_net: str = ""
    control_image: str = ""
    control_strength: float = 0.9
    # Component layout (Z-Image / FLUX.2-klein / Qwen-Image …)
    diffusion_model: str = ""
    llm: str = ""
    # Step caching (real speedup; docs/caching.md upstream)
    cache_mode: str = "none"
    cache_option: str = ""
    # Keep weights in RAM, load into VRAM on demand (GPU builds)
    offload_to_cpu: bool = False
    # Progressive preview ("proj" needs no extra files)
    preview_mode: str = ""          # "" | "proj" | "tae" | "vae"
    preview_path: str = ""
    preview_interval: int = 1
    # Inpainting: white areas of the mask are regenerated (needs init_image)
    mask_image: str = ""
    # Hires fix (A1111-style two-pass): 0 scale = off
    hires_scale: float = 0.0
    hires_steps: int = 0            # 0 = reuse --steps
    hires_upscaler: str = "Latent"


def build_argv(binary: Path, params: SDGenParams, output_path: str) -> list[str]:
    """Translate params into an sd-cli argv. Sentinels (-1 / "" / "auto")
    omit their flag so sd.cpp's own defaults hold."""
    a: list[str] = [
        str(binary), "-M", "img_gen", "-p", params.prompt,
        "-o", output_path, "-W", str(params.width), "-H", str(params.height),
        "--steps", str(params.steps), "--cfg-scale", str(params.cfg_scale),
        "--sampling-method", params.sampler, "--scheduler", params.scheduler,
        "-s", str(params.seed), "-b", str(params.batch_count),
    ]
    if params.diffusion_model:
        a += ["--diffusion-model", params.diffusion_model]
    else:
        a += ["-m", params.model]
    if params.llm:
        a += ["--llm", params.llm]
    if params.negative_prompt:
        a += ["-n", params.negative_prompt]
    if params.guidance and params.guidance > 0:
        a += ["--guidance", str(params.guidance)]
    if params.clip_skip >= 1:
        a += ["--clip-skip", str(params.clip_skip)]
    if params.weight_type != "auto":
        a += ["--type", params.weight_type]
    if params.threads > 0:
        a += ["-t", str(params.threads)]
    if params.init_image:
        # img2img still runs in img_gen mode (already set) — just add init.
        a += ["--init-img", params.init_image, "--strength", str(params.strength)]
        if params.mask_image:
            a += ["--mask", params.mask_image]
    if params.hires_scale and params.hires_scale > 1.0:
        a += ["--hires-scale", str(params.hires_scale)]
        if params.hires_steps > 0:
            a += ["--hires-steps", str(params.hires_steps)]
        if params.hires_upscaler:
            a += ["--hires-upscaler", params.hires_upscaler]
    if params.vae:
        a += ["--vae", params.vae]
    if params.lora_dir:
        a += ["--lora-model-dir", params.lora_dir]
    if params.upscale_model:
        a += ["--upscale-model", params.upscale_model]
    if params.control_net and params.control_image:
        a += ["--control-net", params.control_net,
              "--control-image", params.control_image,
              "--control-strength", str(params.control_strength)]
    if params.vae_tiling:
        a += ["--vae-tiling"]
    if params.vae_on_cpu:
        a += ["--vae-on-cpu"]
    if params.clip_on_cpu:
        a += ["--clip-on-cpu"]
    if params.diffusion_fa:
        a += ["--diffusion-fa"]
    if params.cache_mode and params.cache_mode != "none":
        a += ["--cache-mode", params.cache_mode]
        if params.cache_option:
            a += ["--cache-option", params.cache_option]
    if params.offload_to_cpu:
        a += ["--offload-to-cpu"]
    if params.preview_mode and params.preview_path:
        a += ["--preview", params.preview_mode,
              "--preview-path", params.preview_path,
              "--preview-interval", str(max(1, params.preview_interval))]
    return a


# sd.cpp prints denoising progress like "  |=====> | 5/20 - 2.34s/it" on a
# carriage-return-updated line; the step is "N/M" followed by an it-rate
# suffix (tensor-loader bars use "MB/s" instead).
_STEP_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s*-\s*[\d.]+\s*(?:s/it|it/s)")
# sd.cpp logs the effective seed (also with -s -1) — capture it so the UI
# can offer "reuse seed" (webui habit).
_SEED_RE = re.compile(r"seed[:= ]+(\d+)", re.IGNORECASE)


class SDBackend:
    """Runs sd-cli under the sandbox and reports progress. One-shot per call."""

    def __init__(self, variant: str = "cpu") -> None:
        self._variant = variant
        self._proc = None
        self._cancelled = False
        self._lock = threading.Lock()
        self.last_seed: int | None = None

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            proc = self._proc
        if proc is not None:
            proc.terminate()

    def generate(self, params: SDGenParams, output_path: str,
                 on_progress=None, on_log=None) -> list[str]:
        """Generate synchronously (call from a worker thread). Returns the
        list of written image paths. Raises SDError on failure/cancel."""
        binary = find_sd_binary(self._variant)
        bin_dir = binary.parent
        read_files: list[str] = []
        if params.diffusion_model:
            resolved = dataclasses.replace(params, model="")
        else:
            model = Path(params.model).resolve()
            if not model.is_file():
                raise SDError(f"model not found: {model}")
            resolved = dataclasses.replace(params, model=str(model))
            read_files.append(str(model))
        out = Path(output_path).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)

        # The sandboxed unit runs from a different cwd, so every path handed
        # to sd-cli must be absolute. Resolve into a copy.
        read_dirs: list[str] = []
        for attr in ("vae", "init_image", "upscale_model",
                     "control_net", "control_image", "diffusion_model",
                     "llm", "mask_image"):
            val = getattr(params, attr)
            if val:
                absp = str(Path(val).resolve())
                setattr(resolved, attr, absp)
                read_files.append(absp)
        if params.lora_dir:
            resolved.lora_dir = str(Path(params.lora_dir).resolve())
            read_dirs.append(resolved.lora_dir)
        if params.diffusion_model and not Path(resolved.diffusion_model).is_file():
            raise SDError(f"diffusion model not found: {params.diffusion_model}")
        if params.preview_path:
            resolved.preview_path = str(Path(params.preview_path).resolve())

        argv = build_argv(binary, resolved, str(out))
        policy = Policy.for_compute_subprocess(
            exec_dir=str(bin_dir),
            read_files=tuple(read_files),
            read_dirs=tuple(read_dirs),
            write_dirs=(str(out.parent),),
        )
        with self._lock:
            self._cancelled = False
        self.last_seed = None

        proc = launch(argv, policy, env={"LD_LIBRARY_PATH": str(bin_dir)})
        with self._lock:
            self._proc = proc
        log_lines: list[str] = []
        try:
            stream = proc.popen.stdout
            if stream is not None:
                buf = ""
                while True:
                    raw = stream.read1(4096) if hasattr(stream, "read1") else stream.read(4096)
                    if not raw:
                        break
                    buf += raw.decode("utf-8", "replace")
                    parts = re.split(r"[\r\n]", buf)
                    buf = parts.pop()
                    for line in parts:
                        line = line.rstrip()
                        if not line:
                            continue
                        log_lines.append(line)
                        if len(log_lines) > 500:
                            del log_lines[:100]
                        if on_log:
                            on_log(line)
                        if self.last_seed is None:
                            sm = _SEED_RE.search(line)
                            if sm:
                                self.last_seed = int(sm.group(1))
                        if on_progress:
                            m = _STEP_RE.search(line)
                            if m:
                                step, total = int(m.group(1)), int(m.group(2))
                                if 0 < total <= 10000 and step <= total:
                                    on_progress(step, total)
            code = proc.popen.wait()
        finally:
            proc.cleanup_env_file()
            with self._lock:
                self._proc = None

        if self._cancelled:
            raise SDError("generation cancelled")
        if code != 0:
            tail = "\n".join(log_lines[-25:])
            raise SDError(f"sd-cli exited with code {code}", tail)

        written: list[str] = []
        if out.is_file():
            written.append(str(out))
        else:
            stem = out.stem
            for f in sorted(out.parent.glob(f"{stem}*{out.suffix}")):
                written.append(str(f))
        if not written:
            raise SDError(
                "sd-cli finished but no output image was written",
                "\n".join(log_lines[-25:]),
            )
        return written
