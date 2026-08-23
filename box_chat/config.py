"""Filesystem paths and persistent settings (JSON).

Follows XDG Base Directory spec:
  ~/.config/box/settings.json
  ~/.local/share/box/{models,chats.db,cache}
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


# ──── XDG paths ──────────────────────────────────────────────────────────────

def _xdg(env: str, fallback: Path) -> Path:
    return Path(os.environ.get(env, str(fallback))).expanduser()


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", Path.home() / ".config") / "box"
DATA_DIR = _xdg("XDG_DATA_HOME", Path.home() / ".local" / "share") / "box"
CACHE_DIR = _xdg("XDG_CACHE_HOME", Path.home() / ".cache") / "box"

MODELS_DIR = DATA_DIR / "models"
VOICE_DIR  = DATA_DIR / "voice_messages"
EMBED_DIR  = DATA_DIR / "embeddings"      # embedding model + tokenizer files
CAPTURES_DIR = CACHE_DIR / "captures"     # webcam JPEG snapshots; transient
DB_PATH = DATA_DIR / "chats.db"
RAG_DB_PATH = DATA_DIR / "rag_index.db"   # separate sqlite for chunk metadata
VECTOR_INDEX_DIR = DATA_DIR / "indexes"   # TurboVec .tvim index files
SETTINGS_PATH = CONFIG_DIR / "settings.json"
LITERTLM_CACHE = CACHE_DIR / "litert-lm"

for _p in (CONFIG_DIR, DATA_DIR, CACHE_DIR, MODELS_DIR, LITERTLM_CACHE,
           VOICE_DIR, EMBED_DIR, CAPTURES_DIR, VECTOR_INDEX_DIR):
    _p.mkdir(parents=True, exist_ok=True)


# ──── Agent-mode prompt presets (Phase 5) ────────────────────────────────────
# The text prepended to the system prompt while agent mode is on. "balanced"
# is the default — it explicitly tells the model to focus on the latest
# message and not restate earlier answers, which small models (e.g. Gemma 4
# E2B) otherwise do when a follow-up is topically similar to a prior turn.
# Insertion order is the dropdown order; "custom" is handled separately (it
# uses Settings.agent_system_prompt, the user's own text).
AGENT_PROMPT_PRESETS: dict[str, str] = {
    "balanced": (
        "You are an autonomous research assistant. Focus on the user's most "
        "recent message and answer that specific request — do not repeat or "
        "restate earlier answers in this conversation. Break the task into "
        "steps and use your tools (web search, filesystem read/list/grep) "
        "across multiple turns to gather what you need. Think before each "
        "tool call and cite the specific sources you used. Keep going until "
        "you can give a useful, grounded reply, then stop and summarise."
    ),
    "fresh": (
        "Answer only the user's most recent message, treating it as a fresh "
        "request. Ignore the framing of any earlier answers in this chat — if "
        "the new question is different, research it from scratch with new tool "
        "calls rather than reusing previous results. Use your tools as needed, "
        "cite your sources, then give a focused answer to exactly what was "
        "asked."
    ),
    "concise": (
        "Use tools sparingly — only when you genuinely cannot answer from what "
        "you already know, and prefer one or two targeted calls at most. "
        "Answer the user's latest question directly and briefly, cite any "
        "sources you used, and don't pad the reply."
    ),
}

AGENT_PROMPT_PRESET_LABELS: dict[str, str] = {
    "balanced": "Balanced research (recommended)",
    "fresh": "Fresh each turn",
    "concise": "Concise",
    "custom": "Custom",
}


# ──── Settings ───────────────────────────────────────────────────────────────

@dataclass
class Settings:
    # Path to a .litertlm file. Empty until the user picks one.
    model_path: str = ""

    # "cpu" | "gpu"
    backend: str = "cpu"

    # Inference options
    enable_speculative_decoding: bool = False
    enable_vision: bool = False
    enable_audio: bool = False

    # Sampling (None = use model defaults)
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None

    # TTS
    enable_tts: bool = False
    tts_auto_speak: bool = False
    tts_voice: str = "en_US-lessac-medium"
    # Playback gain applied to the raw Piper output. 1.0 = native level,
    # 0.0 = silent. Range allowed up to 2.0 so quiet voices can be boosted;
    # the sample buffer is clipped to [-1, 1] before sounddevice playback.
    tts_volume: float = 1.0

    # Voice input
    voice_auto_send: bool = False

    # Live conversation mode: when True, the mic is NOT auto-listened via
    # VAD. Instead the user holds a "Talk" button in the live panel to
    # capture each turn. Covers noisy environments where VAD (even at
    # aggressiveness 3) keeps false-triggering on background sound/music.
    live_push_to_talk: bool = False

    # System / assistant behaviour
    system_prompt: str = (
        "You are a helpful, concise assistant running locally on the user's device."
    )

    # LLM context window (`max_num_tokens` passed to litert_lm.Engine).
    # Gemma 4 E2B/E4B support up to 128k. We default to 4096 to keep CPU
    # inference fast on entry-level hardware — at 32k the KV cache is
    # ~8x larger and every generated token has to re-stream that buffer,
    # which dropped throughput from ~12 tok/s to ~1.3 tok/s on a Gemma 4
    # E2B + i5 7th-gen test rig. Users who need long context (notebook
    # retrieval, long PDFs in chat) can raise this in Preferences →
    # Behaviour → Context window; the engine reloads on change.
    max_context_tokens: int = 4096
    # Optional UI: show an estimated token-usage bar above the composer.
    show_context_bar: bool = True

    # ── App Lock (2026-07-14) ──────────────────────────────────────────────
    # Argon2id-hashed password gate on the app window. NOT encryption — Box
    # has no encrypted volumes; it only gates casual UI access.
    app_lock_enabled: bool = False
    app_lock_hash: str = ""

    # ── llama.cpp (GGUF) engine ────────────────────────────────────────────
    # Only applied when the loaded model is a .gguf. Sentinels
    # ("auto"/""/0/-1) mean "omit the flag" so the pinned binary's defaults
    # stay authoritative.
    llama_ctx_mode: str = "auto"        # "auto" = --fit sizing | "manual"
    llama_ctx_size: int = 8192
    llama_fit_target_mib: int = 1024
    llama_fit_ctx_min: int = 4096
    llama_cache_type_k: str = "auto"
    llama_cache_type_v: str = "auto"
    llama_kv_unified: str = "auto"
    llama_swa_full: bool = False
    llama_keep_tokens: int = 0
    llama_cache_reuse: int = 256
    llama_cache_ram_mib: int = 0
    llama_threads: int = -1
    llama_threads_batch: int = -1
    llama_batch_size: int = 0
    llama_ubatch_size: int = 0
    llama_flash_attn: str = "auto"
    llama_cont_batching: bool = True
    llama_parallel: int = 0
    llama_mmap: bool = True
    llama_mlock: bool = False
    llama_cpu_range: str = ""
    llama_cpu_strict: bool = False
    llama_priority: int = 0
    llama_poll: int = -1
    llama_numa: str = ""
    llama_cpu_moe: bool = False
    llama_n_cpu_moe: int = 0
    llama_variant: str = "auto"
    llama_gpu_layers: int = 0
    llama_spec_type: str = "none"
    llama_draft_model: str = ""
    llama_spec_n_max: int = 0
    llama_spec_n_min: int = 0
    llama_draft_cache_type_k: str = "auto"
    llama_draft_cache_type_v: str = "auto"
    llama_rope_scaling: str = ""
    llama_rope_scale: float = 0.0
    llama_rope_freq_base: float = 0.0
    llama_rope_freq_scale: float = 0.0
    llama_min_p: float = 0.0
    llama_repeat_penalty: float = 0.0
    llama_presence_penalty: float = 0.0
    llama_frequency_penalty: float = 0.0
    llama_strip_reasoning: bool = True

    # ── stable-diffusion.cpp image generation ──────────────────────────────
    sd_models: list[str] = field(default_factory=list)
    sd_last_model: str = ""
    sd_variant: str = "auto"
    sd_width: int = 512
    sd_height: int = 512
    sd_steps: int = 20
    sd_cfg_scale: float = 7.0
    sd_guidance: float = 3.5
    sd_sampler: str = "euler_a"
    sd_scheduler: str = "discrete"
    sd_batch_count: int = 1
    sd_clip_skip: int = -1
    sd_weight_type: str = "auto"
    # Component-model mode for the sd.cpp engine (Z-Image/klein GGUFs).
    sd_gen_mode: str = "checkpoint"    # "checkpoint" | "components"
    sd_component_dir: str = ""         # selected bundle directory
    sd_cache_mode: str = "none"        # none|ucache|easycache|dbcache|taylorseer
    # Hand-imported component files (shown as "Custom" in the bundle list).
    sd_custom_diffusion: str = ""
    sd_custom_vae: str = ""
    sd_custom_llm: str = ""
    # Hires fix (two-pass upscale during generation)
    sd_hires_scale: float = 0.0        # 0/1 = off
    sd_hires_steps: int = 0            # 0 = reuse steps
    sd_hires_upscaler: str = "Latent"
    sd_offload_cpu: bool = False       # GPU builds: weights in RAM
    sd_live_preview: bool = True       # progressive preview during gen
    sd_lora_dir: str = ""              # LoRA folder (<lora:name:w> syntax)
    sd_vae_tiling: bool = False
    sd_vae_on_cpu: bool = False
    sd_clip_on_cpu: bool = False
    sd_diffusion_fa: bool = False

    # Which engine the Generate page drives: "sdcpp" (stable-diffusion.cpp,
    # GGUF/safetensors) or "litert" (Z-Image / FLUX.2-klein .tflite pipelines).
    sd_engine: str = "sdcpp"
    # Imported LiteRT diffusion model *directories* (each holds the chunked
    # .tflite graphs + tokenizer for one Z-Image / klein model).
    litert_diffusion_dirs: list[str] = field(default_factory=list)
    litert_last_dir: str = ""
    litert_steps: int = 9
    litert_guidance: float = 0.0

    # ── Box Code (standalone local coding-agent mode) ──────────────────────
    code_project_dir: str = ""          # last project folder
    code_model_path: str = ""           # GGUF used for agent sessions
    code_permission_mode: str = "ask"   # "ask" | "auto"
    code_max_iterations: int = 100      # tool-call budget per send
    code_bash_timeout: int = 120        # default seconds per bash command
    code_max_context: int = 8192        # token budget for agent context
    code_read_agents_md: bool = True    # fold AGENTS.md/CLAUDE.md into prompt
    code_web_enabled: bool = False      # opt-in web_search/fetch_url tools
    code_temperature: float = -1.0      # sampling temp; < 0 = use global
    code_notify_done: bool = True       # desktop notification when a turn ends

    # One-time welcome text on the empty chat; "Got it" hides it forever.
    welcome_dismissed: bool = False

    # Font for headers app-wide (window titles, sidebar titles, section
    # headings, New Chat/New Notebook). "" = theme default.
    header_font_family: str = ""

    # iOS-style glass mode: translucent surfaces + luminous borders.
    glass_mode: bool = False
    glass_opacity: float = 0.82         # 0.5–1.0; lower = more see-through
    # Liquid glass: glass + pill lens controls, speculars, accent glow.
    glass_liquid: bool = False

    # Nav rail (ATK-inspired): position + labels, applied live.
    nav_position: str = "bottom"        # left | right | top | bottom
    nav_labels: bool = False            # show names with the icons

    # UI preferences
    theme: str = "aura"               # any key in themes.ALL_THEMES
    accent_name: str = ""             # named accent within the theme; "" = theme default
    bubble_accent_name: str = ""      # bubble-specific accent (border, role labels); "" = follow theme accent
    font_size: int = 14

    # Window geometry (auto-saved on close)
    window_width: int = 1100
    window_height: int = 720

    # Left sidebar (Chats / Notebooks) — width is restored from the user's
    # last drag; visibility is toggled from the content header.
    sidebar_width: int = 280
    sidebar_visible: bool = True

    # Most recently used .litertlm paths (max 8)
    recent_models: list[str] = field(default_factory=list)
    # Imported .gguf model paths (runnable via the llama.cpp backend).
    imported_gguf_models: list[str] = field(default_factory=list)

    # Composer input box — False = blends with theme, True = accent colour background
    composer_use_accent: bool = False

    # Bubble colour palette key (see themes.BUBBLE_PALETTES)
    bubble_style: str = "default"
    bubble_opacity: float = 1.0          # 0.1–1.0; applied to bubble backgrounds
    user_bubble_text_color: str = ""     # hex override e.g. "#ff0000"; "" = palette default
    asst_bubble_text_color: str = ""     # hex override; "" = palette default
    chat_font_family: str = ""           # e.g. "Noto Sans"; "" = system default

    # ── RAG (retrieval-augmented generation) ───────────────────────────────
    rag_enabled: bool = False             # global on/off
    rag_embed_variant: str = "seq1024"    # "seq1024" | "seq2048"
    rag_top_k: int = 5                    # number of chunks injected per query
    rag_chunk_size: int = 500             # approximate words per chunk
    rag_chunk_overlap: int = 100          # word overlap between adjacent chunks
    rag_max_chunks: int = 100             # cap chunks per doc (0 = unlimited)
                                          # when over, evenly sample across the doc
    rag_auto_attach_notebooks: bool = True  # apply notebooks.auto_attach=1 to new chats

    # ── Tools (Phase 4) — every tool OFF by default ────────────────────────
    # Web search (ddgs / DuckDuckGo).
    tool_web_search_enabled: bool = False
    tool_web_search_permission: str = "chat"   # "ask"|"chat"|"trust"
    tool_web_search_results: int = 5           # max results returned per query
    tool_web_search_first_enable_acknowledged: bool = False

    # Filesystem (read-only by default; writes are a nested toggle).
    tool_fs_enabled: bool = False
    tool_fs_writable: bool = False
    tool_fs_root: str = "~/Documents/box-workspace"
    tool_fs_permission: str = "ask"
    tool_fs_first_enable_acknowledged: bool = False
    # On-the-fly access: when on, the model may request paths OUTSIDE the
    # workspace; the permission gate prompts for each one (per-path, never
    # blanket-trusted). The workspace stays the always-allowed base.
    tool_fs_allow_outside: bool = False
    # Paths the user chose "Always allow" for, outside the workspace. A grant
    # on a directory also covers files beneath it. Session-only grants
    # ("Allow once/for this chat") live in memory and aren't persisted here.
    tool_fs_extra_roots: list[str] = field(default_factory=list)

    # (Code interpreter is deferred — Linux kernel locks unprivileged user
    # namespaces on this user's systems, so bwrap can't sandbox; ship the
    # other tools without it for Phase 4.)

    # ── Webcam / vision capture (Phase 4.5) ───────────────────────────────
    # Master switch for the 📷 button in the composer + Camera tab in Prefs.
    # Independent of `enable_vision` — that one controls whether the model
    # loads its vision encoder; this one controls whether Box exposes any
    # camera UI at all. Both must be on for the feature to be useful.
    webcam_enabled: bool = False
    webcam_first_enable_acknowledged: bool = False
    webcam_device: str = ""              # "" = system default; else /dev/videoN
    webcam_capture_width: int = 640      # pixels (height derived from cam)
    webcam_capture_jpeg_quality: int = 80
    # Tier 2 (sticky vision mode): per-chat override mirrors the tools
    # tri-state. NULL/follow-global = vision mode off; 1 = always attach
    # the latest captured frame to every send in this chat.
    vision_mode_active: bool = False

    # ── Phase 5: agent mode ───────────────────────────────────────────────
    # Master switch for "agentic" multi-step chaining. When effectively on
    # for a chat, window prepends a stanza to the system prompt telling
    # the model to use tools across multiple steps, AND BoxToolEventHandler
    # caps the number of tool calls per user-send at agent_max_iterations
    # so a confused model can't loop forever.
    agent_enabled: bool = False
    agent_first_enable_acknowledged: bool = False
    agent_max_iterations: int = 6
    # Which preset supplies the agent prompt: "balanced" | "fresh" |
    # "concise" | "custom". See AGENT_PROMPT_PRESETS. "custom" uses the
    # free-text agent_system_prompt below.
    agent_prompt_preset: str = "balanced"
    # User's own agent prompt — only used when agent_prompt_preset == "custom".
    # Seeded with the balanced text so switching to Custom starts sensibly.
    agent_system_prompt: str = AGENT_PROMPT_PRESETS["balanced"]

    # Persisted "always allow" decisions. Each entry is either a bare tool id
    # ("web_search") for blanket trust, or "tool_id::arg-hash" for scoped trust.
    tool_always_allow: list[str] = field(default_factory=list)

    # ── Phase 6: persistent memory ────────────────────────────────────────
    # A long-term, cross-chat store of facts the user explicitly saves
    # ("Remember this"). Distinct from RAG (which is per-chat / per-notebook
    # documents). When enabled, the top-K most relevant memories are embedded-
    # searched and prepended to each message — so the assistant carries
    # context between conversations. Off by default (every feature off until
    # the user opts in); capture is ALWAYS explicit (nothing is stored without
    # the user saving it). Retrieval reuses the RAG embedder + vector store.
    memory_enabled: bool = False
    memory_top_k: int = 3                  # memories injected per message

    # ── File / log audit (chunked map-reduce) ─────────────────────────────
    # "audit /var/log/dmesg for security issues" reads a file the filesystem
    # tool can reach, splits it into window-sized sections, flags notable
    # events per section, then writes one report — so a log far bigger than
    # the context window can still be audited locally. Gated on the filesystem
    # tool being enabled for the chat (workspace scope + no-root still apply).
    # The cap bounds runtime: a file with more sections than this is sampled
    # evenly across its length rather than read end-to-end.
    audit_max_chunks: int = 40

    # Generic wall-clock deadline (seconds) applied to EVERY tool callable
    # before the SDK ever sees the result. A tool that wedges on I/O (a slow
    # web_search, a huge fs_grep, a hung future tool) returns a fast "timed
    # out" string to the model instead of blocking the engine worker for tens
    # of seconds — which previously made the app look "not responding" and
    # locked out Stop/barge-in. 0 disables the cap (advanced; not recommended).
    tool_timeout_s: int = 20

    # ── persistence ────────────────────────────────────────────────────────
    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_PATH.exists():
            s = cls()
            s.save()
            return s
        try:
            raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in raw.items() if k in valid})

    def save(self) -> None:
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        # settings.json now holds the App Lock Argon2id hash — make it
        # owner-only before the atomic rename so no other local account can
        # read the hash off disk and brute-force it offline (2026-07-14 fix).
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        tmp.replace(SETTINGS_PATH)

    def resolved_agent_prompt(self) -> str:
        """The agent stanza actually prepended to the system prompt: the
        selected preset's text, or the user's custom text when the preset
        is 'custom'."""
        if self.agent_prompt_preset == "custom":
            return self.agent_system_prompt
        return AGENT_PROMPT_PRESETS.get(
            self.agent_prompt_preset, AGENT_PROMPT_PRESETS["balanced"]
        )

    def add_recent_model(self, path: str) -> None:
        if not path:
            return
        if path in self.recent_models:
            self.recent_models.remove(path)
        self.recent_models.insert(0, path)
        del self.recent_models[8:]

    def add_imported_gguf(self, path: str) -> None:
        if not path:
            return
        if path in self.imported_gguf_models:
            self.imported_gguf_models.remove(path)
        self.imported_gguf_models.insert(0, path)
        del self.imported_gguf_models[16:]

    def forget_imported_gguf(self, path: str) -> None:
        if path in self.imported_gguf_models:
            self.imported_gguf_models.remove(path)

    def add_sd_model(self, path: str) -> None:
        if path in self.sd_models:
            self.sd_models.remove(path)
        self.sd_models.insert(0, path)
        del self.sd_models[8:]
        self.sd_last_model = path

    def forget_sd_model(self, path: str) -> None:
        if path in self.sd_models:
            self.sd_models.remove(path)
        if self.sd_last_model == path:
            self.sd_last_model = self.sd_models[0] if self.sd_models else ""

    def add_litert_dir(self, path: str) -> None:
        if path in self.litert_diffusion_dirs:
            self.litert_diffusion_dirs.remove(path)
        self.litert_diffusion_dirs.insert(0, path)
        del self.litert_diffusion_dirs[8:]
        self.litert_last_dir = path

    def forget_litert_dir(self, path: str) -> None:
        if path in self.litert_diffusion_dirs:
            self.litert_diffusion_dirs.remove(path)
        if self.litert_last_dir == path:
            self.litert_last_dir = (
                self.litert_diffusion_dirs[0] if self.litert_diffusion_dirs else ""
            )
