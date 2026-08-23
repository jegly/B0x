"""Preferences dialog — model picker, system prompt, theme, font size."""
from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, GLib, Gio, Gtk, Pango  # noqa: E402

from .config import (
    AGENT_PROMPT_PRESET_LABELS,
    AGENT_PROMPT_PRESETS,
    MODELS_DIR,
    Settings,
)


class PreferencesDialog(Adw.PreferencesWindow):
    # A resizable window (not the adaptive, fixed-size Adw.PreferencesDialog):
    # with ~10 pages the dialog collapsed its tabs into a cramped bottom bar,
    # and users couldn't widen it. A window opens wide enough for the top view
    # switcher to breathe and can be resized freely.
    def __init__(self, app, settings: Settings, window=None) -> None:
        super().__init__()
        self._app = app
        self._settings = settings
        self._window = window
        # Debounce handle for model reloads (see _trigger_model_reload).
        self._reload_source_id = 0

        self.set_title("Preferences")
        self.set_search_enabled(True)
        self.set_default_size(980, 720)
        if window is not None:
            self.set_transient_for(window)
            self.set_modal(True)

        self.add(self._build_model_page())
        self.add(self._build_models_page())
        self.add(self._build_behaviour_page())
        self.add(self._build_llama_page())
        self.add(self._build_multimodal_page())
        self.add(self._build_knowledge_page())
        self.add(self._build_memory_page())
        self.add(self._build_tools_page())
        self.add(self._build_appearance_page())
        self.add(self._build_security_page())

    # ──────────────────────────────────────────────────────────────────────
    def _build_model_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Model", icon_name="application-x-executable-symbolic")

        group = Adw.PreferencesGroup(
            title="LiteRT-LM model",
            description=(
                "Pick any .litertlm file. Gemma-3n E2B / E4B work out of the box. "
                "Re-loading the same model after the first run is fast (cached)."
            ),
        )
        page.add(group)

        # Current model row
        self._model_row = Adw.ActionRow(
            title="Selected model",
            subtitle=self._settings.model_path or "(none — pick a file below)",
        )
        choose_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        choose_btn.connect("clicked", self._on_choose_model)
        self._model_row.add_suffix(choose_btn)
        self._model_row.set_activatable_widget(choose_btn)
        group.add(self._model_row)

        # Recent models
        if self._settings.recent_models:
            recents = Adw.PreferencesGroup(title="Recent models")
            page.add(recents)
            for p in self._settings.recent_models:
                row = Adw.ActionRow(title=Path(p).name, subtitle=p)
                use_btn = Gtk.Button(label="Use", valign=Gtk.Align.CENTER)
                use_btn.connect("clicked", lambda _b, path=p: self._set_model(path))
                row.add_suffix(use_btn)
                recents.add(row)

        # Download from HuggingFace
        dl_group = Adw.PreferencesGroup(title="Get models")
        page.add(dl_group)

        dl_row = Adw.ActionRow(
            title="Download from HuggingFace",
            subtitle="Gemma 4 E2B and E4B · no account required",
            activatable=True,
        )
        dl_row.add_suffix(Gtk.Image(icon_name="go-next-symbolic", valign=Gtk.Align.CENTER))
        dl_row.connect("activated", self._on_download_models)
        dl_group.add(dl_row)

        # Inference backend + options
        backend_group = Adw.PreferencesGroup(title="Inference backend")
        page.add(backend_group)

        self._backend_ids = ["cpu", "gpu"]
        backend_row = Adw.ComboRow(
            title="Backend",
            subtitle="GPU requires compatible drivers. NPU is C++ SDK only.",
            model=Gtk.StringList.new(["CPU", "GPU"]),
        )
        try:
            backend_row.set_selected(self._backend_ids.index(self._settings.backend))
        except ValueError:
            backend_row.set_selected(0)
        backend_row.connect("notify::selected", self._on_backend_changed)
        backend_group.add(backend_row)

        spec_row = Adw.SwitchRow(
            title="Speculative decoding (MTP)",
            subtitle=(
                "Google recommends this for GPU backends only. On CPU it "
                "typically halves throughput because the draft model's "
                "compute cost outweighs the parallel-verification gain. "
                "Leave OFF unless you're on GPU."
            ),
        )
        spec_row.set_active(self._settings.enable_speculative_decoding)
        spec_row.connect("notify::active", self._on_spec_decoding_changed)
        backend_group.add(spec_row)

        return page

    # ──────────────────────────────────────────────────────────────────────
    def _build_behaviour_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Behaviour", icon_name="emblem-system-symbolic")

        group = Adw.PreferencesGroup(
            title="System prompt",
            description="Sent at the start of every conversation as the system role.",
        )
        page.add(group)

        sp_row = Adw.PreferencesRow(activatable=False)
        sp_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6,
                         margin_top=8, margin_bottom=8, margin_start=12, margin_end=12)
        sp_label = Gtk.Label(label="System prompt", xalign=0)
        sp_label.add_css_class("caption-heading")
        sp_box.append(sp_label)
        self._sp_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR, accepts_tab=False,
        )
        self._sp_view.get_buffer().set_text(self._settings.system_prompt)
        self._sp_view.get_buffer().connect("changed", self._on_system_prompt_changed)
        # Wrap in a fixed-height ScrolledWindow so a long prompt scrolls inside
        # the field instead of stretching it tall on first layout (the field
        # otherwise opened over-sized and only snapped back after a relayout).
        sp_scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=120, max_content_height=120,
        )
        sp_scroller.set_child(self._sp_view)
        sp_frame = Gtk.Frame(child=sp_scroller)
        sp_box.append(sp_frame)
        sp_row.set_child(sp_box)
        group.add(sp_row)

        sampling_group = Adw.PreferencesGroup(
            title="Sampling",
            description="Toggle a parameter on to override the model's built-in default.",
        )
        page.add(sampling_group)

        t = self._settings.temperature
        self._temp_switch = Adw.SwitchRow(title="Temperature")
        self._temp_switch.set_active(t is not None)
        self._temp_switch.connect("notify::active", self._on_temp_switch)
        sampling_group.add(self._temp_switch)

        self._temp_spin = Adw.SpinRow.new_with_range(0.0, 2.0, 0.05)
        self._temp_spin.set_title("Temperature value")
        self._temp_spin.set_digits(2)
        self._temp_spin.set_value(t if t is not None else 1.0)
        self._temp_spin.set_sensitive(t is not None)
        self._temp_spin.connect("notify::value", self._on_temperature_changed)
        sampling_group.add(self._temp_spin)

        tk = self._settings.top_k
        self._topk_switch = Adw.SwitchRow(title="Top-K")
        self._topk_switch.set_active(tk is not None)
        self._topk_switch.connect("notify::active", self._on_topk_switch)
        sampling_group.add(self._topk_switch)

        self._topk_spin = Adw.SpinRow.new_with_range(1, 200, 1)
        self._topk_spin.set_title("Top-K value")
        self._topk_spin.set_value(tk if tk is not None else 40)
        self._topk_spin.set_sensitive(tk is not None)
        self._topk_spin.connect("notify::value", self._on_topk_changed)
        sampling_group.add(self._topk_spin)

        tp = self._settings.top_p
        self._topp_switch = Adw.SwitchRow(title="Top-P")
        self._topp_switch.set_active(tp is not None)
        self._topp_switch.connect("notify::active", self._on_topp_switch)
        sampling_group.add(self._topp_switch)

        self._topp_spin = Adw.SpinRow.new_with_range(0.0, 1.0, 0.01)
        self._topp_spin.set_title("Top-P value")
        self._topp_spin.set_digits(2)
        self._topp_spin.set_value(tp if tp is not None else 0.95)
        self._topp_spin.set_sensitive(tp is not None)
        self._topp_spin.connect("notify::value", self._on_topp_changed)
        sampling_group.add(self._topp_spin)

        # ── Context window ──
        ctx_group = Adw.PreferencesGroup(
            title="Context window",
            description=(
                "How much of the conversation + injected context the model "
                "can see at once. Higher = remembers more but uses more RAM "
                "and loads slower. Changing this reloads the model."
            ),
        )
        page.add(ctx_group)

        ctx_row = Adw.SpinRow.new_with_range(1024, 32768, 1024)
        ctx_row.set_title("Maximum tokens")
        ctx_row.set_subtitle(
            "Default 4096. For RAG / long chats try 8192–16384. "
            "Very large values (32k) need 8+ GB free RAM — may crash on low-memory systems."
        )
        ctx_row.set_value(self._settings.max_context_tokens)
        ctx_row.connect("notify::value", self._on_max_context_changed)
        ctx_group.add(ctx_row)

        bar_row = Adw.SwitchRow(
            title="Show context-usage bar",
            subtitle="Thin progress bar above the composer with estimated token usage",
        )
        bar_row.set_active(self._settings.show_context_bar)
        bar_row.connect("notify::active", self._on_context_bar_changed)
        ctx_group.add(bar_row)

        return page

    # ──────────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────────
    def _build_llama_page(self) -> Adw.PreferencesPage:
        """Llama.cpp (GGUF) engine settings — the full surface.

        Data-driven: every knob is (settings attr → row) with one shared
        write-back path (save + debounced reload). Sentinel values
        ("auto"/""/0/-1) mean "server default".
        """
        page = Adw.PreferencesPage(
            title="Llama.cpp",
            icon_name="applications-engineering-symbolic",
            description=(
                "Settings for GGUF models (llama.cpp engine). LiteRT-LM "
                "models use the Model and Behaviour pages instead."
            ),
        )
        s = self._settings

        def _changed(attr, value) -> None:
            setattr(s, attr, value)
            s.save()
            self._trigger_model_reload()

        def switch(group, attr, title, subtitle="") -> Adw.SwitchRow:
            row = Adw.SwitchRow(title=title, subtitle=subtitle)
            row.set_active(bool(getattr(s, attr)))
            row.connect("notify::active",
                        lambda r, _p, a=attr: _changed(a, bool(r.get_active())))
            group.add(row)
            return row

        def spin(group, attr, title, subtitle, lo, hi, step, digits=0):
            row = Adw.SpinRow.new_with_range(lo, hi, step)
            row.set_title(title)
            row.set_subtitle(subtitle)
            row.set_digits(digits)
            row.set_value(getattr(s, attr))
            cast = float if digits else int
            row.connect("notify::value",
                        lambda r, _p, a=attr, c=cast: _changed(a, c(r.get_value())))
            group.add(row)
            return row

        def combo(group, attr, title, subtitle, values, labels=None):
            row = Adw.ComboRow(title=title, subtitle=subtitle)
            row.set_model(Gtk.StringList.new(labels or list(values)))
            current = getattr(s, attr)
            row.set_selected(values.index(current) if current in values else 0)
            row.connect("notify::selected",
                        lambda r, _p, a=attr, v=values: _changed(a, v[r.get_selected()]))
            group.add(row)
            return row

        def entry(group, attr, title, subtitle="") -> Adw.EntryRow:
            row = Adw.EntryRow(title=title)
            if subtitle:
                row.set_tooltip_text(subtitle)
            row.set_text(getattr(s, attr))
            row.connect("notify::text",
                        lambda r, _p, a=attr: _changed(a, r.get_text().strip()))
            group.add(row)
            return row

        kv_types = ["auto", "f16", "bf16", "q8_0", "q5_1", "q5_0",
                    "q4_1", "q4_0", "iq4_nl", "f32"]

        # ── Memory & context ──
        mem = Adw.PreferencesGroup(
            title="Memory and context",
            description=(
                "Auto sizing asks llama.cpp to fit context/settings into "
                "available memory — the safe choice on any RAM size."
            ),
        )
        page.add(mem)
        ctx_mode = combo(mem, "llama_ctx_mode", "Context size",
                         "Auto = fit to available memory",
                         ["auto", "manual"], ["Auto (fit to memory)", "Manual"])
        ctx_spin = spin(mem, "llama_ctx_size", "Manual context tokens",
                        "Used only in Manual mode", 1024, 1_048_576, 1024)
        fit_margin = spin(mem, "llama_fit_target_mib", "Auto-fit safety margin (MiB)",
                          "RAM headroom the auto-sizer leaves free · default 1024",
                          0, 16384, 128)
        fit_floor = spin(mem, "llama_fit_ctx_min", "Auto-fit context floor",
                         "Auto sizing never goes below this · default 4096",
                         256, 131072, 256)

        def _ctx_sensitivity(*_a) -> None:
            manual = s.llama_ctx_mode == "manual"
            ctx_spin.set_sensitive(manual)
            fit_margin.set_sensitive(not manual)
            fit_floor.set_sensitive(not manual)

        ctx_mode.connect("notify::selected", _ctx_sensitivity)
        _ctx_sensitivity()

        combo(mem, "llama_cache_type_k", "KV cache type (K)",
              "q8_0 halves KV memory; pair with flash attention · default f16",
              kv_types)
        combo(mem, "llama_cache_type_v", "KV cache type (V)",
              "Quantized V cache needs flash attention on · default f16", kv_types)
        combo(mem, "llama_kv_unified", "Unified KV buffer",
              "Shared KV buffer across slots · server default: on",
              ["auto", "on", "off"], ["Auto", "On", "Off"])
        switch(mem, "llama_swa_full", "Full sliding-window cache",
               "For SWA models (Gemma family) — bigger cache, better reuse")
        spin(mem, "llama_keep_tokens", "Tokens kept on context shift",
             "-1 keeps the whole initial prompt · default 0", -1, 8192, 64)
        spin(mem, "llama_cache_reuse", "Prompt-cache reuse chunk",
             "Min matching prefix reused via KV shifting · default 256", 0, 4096, 64)
        spin(mem, "llama_cache_ram_mib", "Prompt-cache RAM cap (MiB)",
             "0 = server default (8192) · -1 = unlimited", -1, 65536, 512)

        # ── Performance ──
        perf = Adw.PreferencesGroup(title="Performance")
        page.add(perf)
        spin(perf, "llama_threads", "CPU threads",
             "-1 = auto (physical cores, min 4)", -1, 256, 1)
        spin(perf, "llama_threads_batch", "Batch threads",
             "Threads for prompt processing · -1 = same as CPU threads", -1, 256, 1)
        spin(perf, "llama_batch_size", "Logical batch size",
             "0 = server default (2048)", 0, 8192, 128)
        spin(perf, "llama_ubatch_size", "Physical batch size",
             "0 = server default (512)", 0, 4096, 64)
        combo(perf, "llama_flash_attn", "Flash attention",
              "Needed for quantized KV cache · default auto",
              ["auto", "on", "off"], ["Auto", "On", "Off"])
        switch(perf, "llama_cont_batching", "Continuous batching", "Server default: on")
        spin(perf, "llama_parallel", "Parallel slots",
             "Concurrent request slots · 0 = server default", 0, 8, 1)
        switch(perf, "llama_mmap", "Memory-map model file",
               "Lets the OS page weights in/out under memory pressure — "
               "recommended on")
        switch(perf, "llama_mlock", "Lock model in RAM (mlock)",
               "Prevents swapping; needs enough free RAM for the whole model")

        # ── Advanced CPU ──
        cpu = Adw.PreferencesGroup(
            title="Advanced CPU",
            description="Affinity, scheduling and NUMA — leave at defaults "
                        "unless you know your machine's topology.",
        )
        page.add(cpu)
        entry(cpu, "llama_cpu_range", "CPU affinity range (lo-hi, empty = off)")
        switch(cpu, "llama_cpu_strict", "Strict CPU placement")
        combo(cpu, "llama_priority", "Process priority",
              "Higher = more CPU time, may starve the UI · default normal",
              [-1, 0, 1, 2, 3], ["Low", "Normal", "Medium", "High", "Realtime"])
        spin(cpu, "llama_poll", "Polling level",
             "0 = no busy-wait, 100 = always poll · -1 = default (50)", -1, 100, 5)
        combo(cpu, "llama_numa", "NUMA optimization",
              "For multi-socket machines · default off",
              ["", "distribute", "isolate", "numactl"],
              ["Off", "Distribute", "Isolate", "numactl map"])
        switch(cpu, "llama_cpu_moe", "All MoE experts on CPU",
               "Keeps Mixture-of-Experts weights in system RAM")
        spin(cpu, "llama_n_cpu_moe", "First N layers' experts on CPU",
             "Finer alternative to the switch above · 0 = off", 0, 128, 1)

        # ── GPU ──
        gpu = Adw.PreferencesGroup(
            title="GPU",
            description=(
                "Box bundles CPU and Vulkan builds of llama.cpp. With GPU "
                "layers at 0 the pure-CPU build runs — the Vulkan build is "
                "only used when layers > 0, so an idle GPU never slows "
                "prefill down."
            ),
        )
        page.add(gpu)
        combo(gpu, "llama_variant", "Engine build", "Auto picks by GPU layers",
              ["auto", "cpu", "vulkan"], ["Auto", "CPU only", "Vulkan"])
        spin(gpu, "llama_gpu_layers", "GPU layers",
             "Layers offloaded to VRAM · 0 = pure CPU", 0, 999, 1)

        # ── Speculative decoding ──
        spec = Adw.PreferencesGroup(
            title="Speculative decoding",
            description=(
                "Drafts tokens cheaply, then verifies them in one pass — "
                "same output, often faster. Self-speculation (ngram) needs "
                "no extra model and is free to try. MTP uses the model's own "
                "multi-token head (model support required)."
            ),
        )
        page.add(spec)
        spec_types = ["none", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
                      "ngram-mod", "ngram-cache", "draft-simple", "draft-eagle3",
                      "draft-mtp", "draft-dflash"]
        spec_labels = ["Off", "Self: ngram simple", "Self: ngram map-K",
                       "Self: ngram map-K4V", "Self: ngram mod",
                       "Self: ngram cache", "Draft model: simple",
                       "Draft model: EAGLE-3", "MTP (multi-token prediction)",
                       "Draft model: dflash"]
        spec_combo = combo(spec, "llama_spec_type", "Speculation mode",
                           "Off = standard decoding", spec_types, spec_labels)

        draft_row = Adw.ComboRow(
            title="Draft model",
            subtitle="Small GGUF that drafts for the main model (draft-* modes)",
        )
        draft_paths = [""]
        for p in list(s.imported_gguf_models) + sorted(
            str(x) for x in MODELS_DIR.glob("*.gguf")
        ):
            if p and p not in draft_paths:
                draft_paths.append(p)
        draft_row.set_model(Gtk.StringList.new(
            ["None"] + [Path(p).name for p in draft_paths[1:]]
        ))
        cur_draft = s.llama_draft_model
        draft_row.set_selected(
            draft_paths.index(cur_draft) if cur_draft in draft_paths else 0
        )
        draft_row.connect(
            "notify::selected",
            lambda r, _p: _changed("llama_draft_model", draft_paths[r.get_selected()]),
        )
        spec.add(draft_row)
        spec_nmax = spin(spec, "llama_spec_n_max", "Draft tokens per step",
                         "0 = server default (3)", 0, 64, 1)
        spec_nmin = spin(spec, "llama_spec_n_min", "Min draft tokens",
                         "0 = server default", 0, 64, 1)
        draft_ctk = combo(spec, "llama_draft_cache_type_k",
                          "Draft KV cache type (K)", "", kv_types)
        draft_ctv = combo(spec, "llama_draft_cache_type_v",
                          "Draft KV cache type (V)", "", kv_types)

        def _spec_sensitivity(*_a) -> None:
            mode = s.llama_spec_type
            on = mode != "none"
            uses_draft = mode.startswith("draft-") and mode != "draft-mtp"
            for w in (spec_nmax, spec_nmin):
                w.set_sensitive(on)
            for w in (draft_row, draft_ctk, draft_ctv):
                w.set_sensitive(uses_draft)

        spec_combo.connect("notify::selected", _spec_sensitivity)
        _spec_sensitivity()

        # ── RoPE ──
        rope = Adw.PreferencesGroup(
            title="RoPE scaling",
            description="Context-extension parameters. 0 / Model default means "
                        "the GGUF's own metadata wins.",
        )
        page.add(rope)
        combo(rope, "llama_rope_scaling", "Scaling method", "",
              ["", "none", "linear", "yarn"],
              ["Model default", "None", "Linear", "YaRN"])
        spin(rope, "llama_rope_scale", "Context scale factor",
             "0 = model default", 0.0, 32.0, 0.5, digits=1)
        spin(rope, "llama_rope_freq_base", "Frequency base",
             "0 = model default", 0.0, 10_000_000.0, 1000.0, digits=0)
        spin(rope, "llama_rope_freq_scale", "Frequency scale",
             "0 = model default", 0.0, 8.0, 0.05, digits=2)

        # ── Sampling extras ──
        samp = Adw.PreferencesGroup(
            title="Sampling (GGUF extras)",
            description=(
                "Applied per request — no model reload. Temperature, Top-K "
                "and Top-P from Behaviour → Sampling apply here too."
            ),
        )
        page.add(samp)
        spin(samp, "llama_min_p", "Min-P", "0 = server default (0.05)",
             0.0, 1.0, 0.01, digits=2)
        spin(samp, "llama_repeat_penalty", "Repeat penalty",
             "0 = server default · 1.0 disables", 0.0, 2.0, 0.05, digits=2)
        spin(samp, "llama_presence_penalty", "Presence penalty", "0 = off",
             -2.0, 2.0, 0.1, digits=1)
        spin(samp, "llama_frequency_penalty", "Frequency penalty", "0 = off",
             -2.0, 2.0, 0.1, digits=1)

        # ── History ──
        hist = Adw.PreferencesGroup(title="History")
        page.add(hist)
        switch(hist, "llama_strip_reasoning", "Strip reasoning from history",
               "Trims chain-of-thought (think) blocks from past turns before "
               "resending — better cache reuse; some models answer multi-turn "
               "better with reasoning kept")

        return page

    # ──────────────────────────────────────────────────────────────────────
    def _build_multimodal_page(self) -> Adw.PreferencesPage:
        from .tts import is_ready as tts_ready, VOICES, DEFAULT_VOICE

        page = Adw.PreferencesPage(
            title="Multimodal", icon_name="applications-multimedia-symbolic"
        )

        # Vision
        vis_group = Adw.PreferencesGroup(
            title="Vision",
            description="Let the model see images you attach. Requires model restart.",
        )
        page.add(vis_group)
        vis_row = Adw.SwitchRow(title="Vision backend", subtitle="CPU backend for image understanding")
        vis_row.set_active(self._settings.enable_vision)
        vis_row.connect("notify::active", self._on_vision_changed)
        vis_group.add(vis_row)

        # Camera (Phase 4.5) — sits next to Vision because they're a pair:
        # Vision enables the model's image encoder; Camera adds a webcam
        # capture button to the composer. Both are needed for live use.
        cam_group = Adw.PreferencesGroup(
            title="Camera",
            description=(
                "Adds a 📷 button to the composer so you can capture a "
                "webcam frame and attach it to the next message. "
                "Requires Vision (above) to be on for the model to "
                "actually look at the image."
            ),
        )
        page.add(cam_group)

        cam_switch = Adw.SwitchRow(title="Enable camera")
        cam_switch.set_active(self._settings.webcam_enabled)
        cam_switch.connect("notify::active", self._on_cam_enabled_changed)
        cam_group.add(cam_switch)
        self._cam_switch = cam_switch

        # Device picker — populated lazily when the user enables the
        # feature (probes for cameras only when they ask for it).
        self._cam_device_row = Adw.ComboRow(
            title="Camera device",
            subtitle="System default until you pick one.",
        )
        self._cam_device_ids: list[str] = []
        self._cam_device_row.connect(
            "notify::selected", self._on_cam_device_changed
        )
        cam_group.add(self._cam_device_row)
        refresh_btn = Gtk.Button(
            label="Refresh", valign=Gtk.Align.CENTER,
        )
        refresh_btn.add_css_class("flat")
        refresh_btn.connect("clicked", lambda *_: self._refresh_cam_devices())
        self._cam_device_row.add_suffix(refresh_btn)

        cam_w_row = Adw.SpinRow.new_with_range(160, 1920, 16)
        cam_w_row.set_title("Capture width")
        cam_w_row.set_subtitle(
            "Lower = faster prefill, smaller image. 640 is a good default."
        )
        cam_w_row.set_value(self._settings.webcam_capture_width)
        cam_w_row.connect("notify::value", self._on_cam_width_changed)
        cam_group.add(cam_w_row)

        cam_q_row = Adw.SpinRow.new_with_range(40, 95, 5)
        cam_q_row.set_title("JPEG quality")
        cam_q_row.set_subtitle(
            "70-80 is usually plenty for the model to read text + objects."
        )
        cam_q_row.set_value(self._settings.webcam_capture_jpeg_quality)
        cam_q_row.connect("notify::value", self._on_cam_quality_changed)
        cam_group.add(cam_q_row)

        self._cam_sub_rows = (self._cam_device_row, cam_w_row, cam_q_row)
        self._refresh_cam_devices()
        self._sync_cam_sensitivity()

        # Audio input
        audio_group = Adw.PreferencesGroup(
            title="Audio input",
            description="Let the model hear audio you record or attach. Requires model restart.",
        )
        page.add(audio_group)
        audio_row = Adw.SwitchRow(title="Audio backend", subtitle="CPU backend for audio understanding")
        audio_row.set_active(self._settings.enable_audio)
        audio_row.connect("notify::active", self._on_audio_changed)
        audio_group.add(audio_row)

        auto_send_row = Adw.SwitchRow(
            title="Auto-send voice messages",
            subtitle="Send immediately when recording stops, no button press needed",
        )
        auto_send_row.set_active(self._settings.voice_auto_send)
        auto_send_row.connect("notify::active", self._on_voice_auto_send_changed)
        audio_group.add(auto_send_row)

        ptt_row = Adw.SwitchRow(
            title="Push-to-talk in live mode",
            subtitle=(
                "Hold a Talk button to speak instead of auto-listening. "
                "Use this in noisy rooms where the voice detector "
                "false-triggers."
            ),
        )
        ptt_row.set_active(self._settings.live_push_to_talk)
        ptt_row.connect("notify::active", self._on_push_to_talk_changed)
        audio_group.add(ptt_row)

        # TTS
        tts_group = Adw.PreferencesGroup(
            title="Voice output (TTS)",
            description="Piper — high-quality offline neural TTS. Each voice is ~60–120 MB.",
        )
        page.add(tts_group)

        tts_row = Adw.SwitchRow(title="Enable voice output")
        tts_row.set_active(self._settings.enable_tts)
        tts_row.connect("notify::active", self._on_tts_changed)
        tts_group.add(tts_row)

        auto_row = Adw.SwitchRow(
            title="Auto-speak responses",
            subtitle="Speak every completed AI response automatically",
        )
        auto_row.set_active(self._settings.tts_auto_speak)
        auto_row.set_sensitive(self._settings.enable_tts)
        auto_row.connect("notify::active", self._on_tts_auto_changed)
        tts_group.add(auto_row)
        self._tts_auto_row = auto_row

        # Voice picker
        self._voice_ids = list(VOICES.keys())
        voice_names = [VOICES[v]["display"] for v in self._voice_ids]
        voice_row = Adw.ComboRow(
            title="Voice",
            model=Gtk.StringList.new(voice_names),
        )
        current = self._settings.tts_voice or DEFAULT_VOICE
        try:
            voice_row.set_selected(self._voice_ids.index(current))
        except ValueError:
            voice_row.set_selected(0)
        voice_row.connect("notify::selected", self._on_voice_changed)
        self._tts_voice_row = voice_row
        tts_group.add(voice_row)

        selected_voice = self._voice_ids[voice_row.get_selected()]
        ready = tts_ready(selected_voice)
        self._tts_dl_row = Adw.ActionRow(
            title="Voice model",
            subtitle="Ready ✓" if ready else "Not downloaded",
        )
        dl_btn = Gtk.Button(
            label="Download" if not ready else "Re-download",
            valign=Gtk.Align.CENTER,
        )
        dl_btn.add_css_class("suggested-action" if not ready else "flat")
        dl_btn.connect("clicked", self._on_tts_download)
        self._tts_dl_btn = dl_btn
        self._tts_dl_row.add_suffix(dl_btn)
        tts_group.add(self._tts_dl_row)

        return page

    # ──────────────────────────────────────────────────────────────────────
    def _build_knowledge_page(self) -> Adw.PreferencesPage:
        from . import rag_models

        page = Adw.PreferencesPage(
            title="Knowledge", icon_name="folder-documents-symbolic"
        )

        # ── RAG group ──
        rag_group = Adw.PreferencesGroup(
            title="Retrieval-Augmented Generation (RAG)",
            description=(
                "Index attached documents into a per-chat vector store. When "
                "enabled, the most relevant chunks are pulled into the prompt "
                "for each query."
            ),
        )
        page.add(rag_group)

        rag_switch = Adw.SwitchRow(title="Enable RAG")
        rag_switch.set_active(self._settings.rag_enabled)
        rag_switch.connect("notify::active", self._on_rag_enabled_changed)
        rag_group.add(rag_switch)

        # Variant picker
        self._rag_variant_ids = list(rag_models.VARIANTS.keys())
        variant_names = [rag_models.VARIANTS[v]["display"] for v in self._rag_variant_ids]
        variant_row = Adw.ComboRow(
            title="Embed model variant",
            subtitle="Higher seq = longer chunks but slower",
            model=Gtk.StringList.new(variant_names),
        )
        try:
            variant_row.set_selected(
                self._rag_variant_ids.index(self._settings.rag_embed_variant)
            )
        except ValueError:
            variant_row.set_selected(0)
        variant_row.connect("notify::selected", self._on_rag_variant_changed)
        rag_group.add(variant_row)
        self._rag_variant_row = variant_row

        # Download row (mirrors the TTS download pattern)
        variant = self._settings.rag_embed_variant
        ready = rag_models.is_variant_ready(variant)
        self._rag_dl_row = Adw.ActionRow(
            title="Embed model",
            subtitle="Ready ✓" if ready else "Not downloaded",
        )
        dl_btn = Gtk.Button(
            label="Download" if not ready else "Re-download",
            valign=Gtk.Align.CENTER,
        )
        dl_btn.add_css_class("suggested-action" if not ready else "flat")
        dl_btn.connect("clicked", self._on_rag_download)
        self._rag_dl_btn = dl_btn
        self._rag_dl_row.add_suffix(dl_btn)
        rag_group.add(self._rag_dl_row)

        # Pieces of context fetched per question.
        topk_row = Adw.SpinRow.new_with_range(1, 20, 1)
        topk_row.set_title("Pieces of context per question")
        topk_row.set_subtitle(
            "How many relevant document sections are pulled into the prompt "
            "each time you ask something. Higher = more context, larger prompt."
        )
        topk_row.set_value(self._settings.rag_top_k)
        topk_row.connect("notify::value", self._on_rag_topk_changed)
        rag_group.add(topk_row)

        # Master kill switch for the per-notebook "auto-attach to new chats"
        # flag. ON (default) means notebooks flagged in their detail view get
        # auto-attached when you create a new chat. OFF disables the behavior
        # wholesale without having to flip every notebook.
        autoattach_row = Adw.SwitchRow(
            title="Auto-attach default notebooks to new chats",
            subtitle=(
                "When a new chat starts, attach any notebooks you flagged "
                "with “Attach to new chats by default”. Turn this off to "
                "disable the behaviour for every notebook at once."
            ),
        )
        autoattach_row.set_active(self._settings.rag_auto_attach_notebooks)
        autoattach_row.connect("notify::active", self._on_rag_auto_attach_changed)
        rag_group.add(autoattach_row)

        # ── Indexing depth ──
        index_group = Adw.PreferencesGroup(
            title="Indexing",
            description=(
                "How much of an attached document gets stored for retrieval. "
                "Indexing is slow on CPU (~5 sec per section)."
            ),
        )
        page.add(index_group)

        # Full vs sampled indexing.
        full_index_row = Adw.SwitchRow(
            title="Index the whole document",
            subtitle=(
                "ON: every section of an attached document is indexed — "
                "accurate but slow.\n"
                "OFF: only a fast sample is indexed — covers start, middle and "
                "end but may miss specific details."
            ),
        )
        full_index_row.set_active(self._settings.rag_max_chunks == 0)
        full_index_row.connect("notify::active", self._on_rag_full_index_changed)
        index_group.add(full_index_row)
        self._rag_full_index_row = full_index_row

        # Sample size — only active when "Index the whole document" is OFF.
        sample_row = Adw.SpinRow.new_with_range(10, 1000, 10)
        sample_row.set_title("Sections to sample")
        sample_row.set_subtitle(
            "When 'Index the whole document' is OFF, this many sections are "
            "evenly chosen across the document. ~5 sec each on CPU."
        )
        # Show the saved value, or default to 100 if currently set to "whole doc".
        sample_row.set_value(
            self._settings.rag_max_chunks if self._settings.rag_max_chunks > 0 else 100
        )
        sample_row.set_sensitive(not full_index_row.get_active())
        sample_row.connect("notify::value", self._on_rag_sample_size_changed)
        index_group.add(sample_row)
        self._rag_sample_row = sample_row

        # Section size — advanced; bigger = fewer sections but each is less specific.
        section_size_row = Adw.SpinRow.new_with_range(100, 2000, 50)
        section_size_row.set_title("Section size (words)")
        section_size_row.set_subtitle(
            "How long each indexed section is. Bigger sections = fewer of "
            "them but each piece of retrieved context is less specific. "
            "Default 500 works well for most documents."
        )
        section_size_row.set_value(self._settings.rag_chunk_size)
        section_size_row.connect("notify::value", self._on_rag_chunk_size_changed)
        index_group.add(section_size_row)

        return page

    # ── Memory (Phase 6) ───────────────────────────────────────────────────
    def _build_memory_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(
            title="Memory", icon_name="user-bookmarks-symbolic"
        )

        intro = Adw.PreferencesGroup(
            title="Persistent memory",
            description=(
                "Long-term facts you explicitly save, recalled across all "
                "chats. Nothing is stored automatically — you choose what to "
                "remember (the 🧠 button in the composer, or below). Uses the "
                "same embedding model as Knowledge/RAG."
            ),
        )
        page.add(intro)

        # Shown when the embedding model isn't downloaded — memory can't save
        # or recall without it, and the controls below are disabled until it is.
        self._memory_model_notice = Adw.ActionRow(
            title="Embedding model not downloaded",
            subtitle=(
                "Memory can't save or recall yet. Download the embedding model "
                "on the Knowledge page first."
            ),
        )
        self._memory_model_notice.add_prefix(
            Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
        )
        self._memory_model_notice.set_visible(False)
        intro.add(self._memory_model_notice)

        mem_switch = Adw.SwitchRow(
            title="Use saved memory",
            subtitle="Inject the most relevant saved memories into each message.",
        )
        mem_switch.set_active(self._settings.memory_enabled)
        mem_switch.connect("notify::active", self._on_memory_enabled_changed)
        intro.add(mem_switch)
        self._memory_switch_row = mem_switch

        topk_row = Adw.SpinRow.new_with_range(1, 10, 1)
        topk_row.set_title("Memories per message")
        topk_row.set_subtitle(
            "How many of the most relevant memories to recall each time."
        )
        topk_row.set_value(self._settings.memory_top_k)
        topk_row.connect("notify::value", self._on_memory_topk_changed)
        intro.add(topk_row)
        self._memory_topk_row = topk_row

        # Add a memory by hand — multi-line so long memories stay visible
        # while typing (a single-line entry hides everything past the width).
        add_group = Adw.PreferencesGroup(title="Add a memory")
        page.add(add_group)
        add_row = Adw.PreferencesRow(activatable=False)
        add_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=8,
            margin_top=10, margin_bottom=10, margin_start=12, margin_end=12,
        )
        self._memory_add_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR, accepts_tab=False,
        )
        add_scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=84, max_content_height=84,
        )
        add_scroller.set_child(self._memory_add_view)
        add_frame = Gtk.Frame(child=add_scroller)
        add_box.append(add_frame)
        add_btn = Gtk.Button(label="Save", halign=Gtk.Align.END)
        add_btn.add_css_class("suggested-action")
        add_btn.connect("clicked", lambda *_: self._on_memory_add())
        add_box.append(add_btn)
        add_row.set_child(add_box)
        add_group.add(add_row)
        self._memory_add_row = add_row

        # Stored memories — searchable, individually deletable.
        list_group = Adw.PreferencesGroup(title="Saved memories")
        page.add(list_group)

        search_row = Adw.EntryRow(title="Search memories")
        search_row.connect("changed", lambda *_: self._refresh_memory_list())
        list_group.add(search_row)
        self._memory_search_row = search_row

        self._memory_listbox = Gtk.ListBox()
        self._memory_listbox.add_css_class("boxed-list")
        self._memory_listbox.set_selection_mode(Gtk.SelectionMode.NONE)
        list_group.add(self._memory_listbox)

        clear_row = Adw.ActionRow(
            title="Forget everything",
            subtitle="Delete all saved memories. Cannot be undone.",
        )
        clear_btn = Gtk.Button(label="Clear all", valign=Gtk.Align.CENTER)
        clear_btn.add_css_class("destructive-action")
        clear_btn.connect("clicked", self._on_memory_clear_all)
        clear_row.add_suffix(clear_btn)
        list_group.add(clear_row)

        # Re-check model readiness whenever the page is shown (the user may
        # download the embed model on the Knowledge page, then come back).
        page.connect("map", lambda *_: self._refresh_memory_availability())
        self._refresh_memory_availability()
        return page

    def _memory_toast(self, text: str) -> None:
        """Show feedback ON the Preferences dialog itself — a plain
        window toast would appear behind the modal and be invisible."""
        try:
            self.add_toast(Adw.Toast.new(text))
        except Exception:
            win = self._main_win()
            if win is not None:
                win._show_toast(text)

    def _memory_model_ready(self) -> bool:
        win = self._main_win()
        rag = getattr(win, "_rag", None) if win is not None else None
        return bool(rag and rag.is_model_ready())

    def _refresh_memory_availability(self) -> None:
        """Show the 'download the embed model' notice and disable save / use
        controls until the embedding model is present."""
        ready = self._memory_model_ready()
        if getattr(self, "_memory_model_notice", None) is not None:
            self._memory_model_notice.set_visible(not ready)
        for attr in ("_memory_switch_row", "_memory_topk_row", "_memory_add_row"):
            w = getattr(self, attr, None)
            if w is not None:
                w.set_sensitive(ready)
        self._refresh_memory_list()

    def _refresh_memory_list(self) -> None:
        """Repopulate the saved-memories list from the store, honouring the
        search box."""
        listbox = getattr(self, "_memory_listbox", None)
        if listbox is None:
            return
        child = listbox.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            listbox.remove(child)
            child = nxt

        win = self._main_win()
        rag = getattr(win, "_rag", None) if win is not None else None
        if rag is None:
            return
        query = self._memory_search_row.get_text().strip()
        mems = rag.search_memories(query) if query else rag.list_memories()
        if not mems:
            placeholder = Adw.ActionRow(
                title="No memories yet" if not query else "No matches",
                subtitle=(
                    "Save one with the 🧠 button in a chat, or above."
                    if not query else None
                ),
            )
            listbox.append(placeholder)
            return
        for m in mems:
            when = datetime.fromtimestamp(m.created_at).strftime("%Y-%m-%d")
            row = Adw.ActionRow(title=m.text, subtitle=when)
            row.set_title_lines(3)
            del_btn = Gtk.Button(
                icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER,
            )
            del_btn.add_css_class("flat")
            del_btn.connect("clicked", self._on_memory_delete, m.id)
            row.add_suffix(del_btn)
            listbox.append(row)

    def _on_memory_enabled_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        active = row.get_active()
        win = self._main_win()
        rag = getattr(win, "_rag", None) if win is not None else None
        if active and rag is not None and not rag.is_model_ready():
            # Revert — memory needs the embed model.
            row.set_active(False)
            self._memory_toast(
                "Memory needs the embedding model — download it on the "
                "Knowledge page first."
            )
            return
        self._settings.memory_enabled = active
        self._settings.save()
        if win is not None:
            win.refresh_memory_button()

    def _on_memory_topk_changed(self, row: Adw.SpinRow, _pspec) -> None:
        self._settings.memory_top_k = int(row.get_value())
        self._settings.save()

    def _on_memory_add(self, *_a) -> None:
        view = getattr(self, "_memory_add_view", None)
        if view is None:
            return
        buf = view.get_buffer()
        text = buf.get_text(
            buf.get_start_iter(), buf.get_end_iter(), False
        ).strip()
        if not text:
            return
        win = self._main_win()
        rag = getattr(win, "_rag", None) if win is not None else None
        if rag is None:
            return
        if not rag.is_model_ready():
            self._memory_toast(
                "Memory needs the embedding model — download it on the "
                "Knowledge page first."
            )
            return
        view.set_sensitive(False)

        def _work() -> None:
            err = None
            try:
                rag.remember(text)
            except Exception as e:  # noqa: BLE001
                err = str(e)

            def _done() -> bool:
                view.set_sensitive(True)
                if err is None:
                    buf.set_text("", -1)
                    self._refresh_memory_list()
                    self._memory_toast("Memory saved.")
                else:
                    self._memory_toast(f"Couldn't save memory: {err}")
                return False

            GLib.idle_add(_done)

        threading.Thread(target=_work, daemon=True).start()

    def _on_memory_delete(self, _btn, memory_id: int) -> None:
        win = self._main_win()
        rag = getattr(win, "_rag", None) if win is not None else None
        if rag is None:
            return
        rag.delete_memory(memory_id)
        self._refresh_memory_list()

    def _on_memory_clear_all(self, _btn) -> None:
        win = self._main_win()
        rag = getattr(win, "_rag", None) if win is not None else None
        if rag is None:
            return
        dlg = Adw.AlertDialog(
            heading="Forget all memories?",
            body="This deletes every saved memory. It cannot be undone.",
        )
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("clear", "Clear all")
        dlg.set_response_appearance("clear", Adw.ResponseAppearance.DESTRUCTIVE)

        def _resp(_d, resp: str) -> None:
            if resp == "clear":
                rag.clear_memories()
                self._refresh_memory_list()

        dlg.connect("response", _resp)
        dlg.present(self._main_win())

    # ──────────────────────────────────────────────────────────────────────
    _PERMISSION_IDS = ("ask", "chat", "trust")
    _PERMISSION_LABELS = (
        "Ask every time",
        "Ask once per chat",
        "Trust always",
    )
    # Dropdown order for the agent style picker. "custom" last; its text
    # comes from Settings.agent_system_prompt instead of a fixed preset.
    _AGENT_PRESET_KEYS = ("balanced", "fresh", "concise", "custom")

    def _build_tools_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(
            title="Tools", icon_name="utilities-terminal-symbolic"
        )

        # ── Web search ──
        ws_group = Adw.PreferencesGroup(
            title="Web search",
            description=(
                "Lets the assistant search the public web through DuckDuckGo. "
                "Free, no account, no API key. Every result URL is checked to "
                "be HTTPS before being shown to the model."
            ),
        )
        page.add(ws_group)

        ws_switch = Adw.SwitchRow(title="Enable web search")
        ws_switch.set_active(self._settings.tool_web_search_enabled)
        ws_switch.connect("notify::active", self._on_ws_enabled_changed)
        ws_group.add(ws_switch)
        self._ws_switch = ws_switch

        ws_results_row = Adw.SpinRow.new_with_range(1, 10, 1)
        ws_results_row.set_title("Results per query")
        ws_results_row.set_subtitle(
            "How many search hits are returned to the model each call."
        )
        ws_results_row.set_value(self._settings.tool_web_search_results)
        ws_results_row.connect("notify::value", self._on_ws_results_changed)
        ws_group.add(ws_results_row)

        ws_perm_row = self._build_permission_combo(
            title="Permission",
            subtitle="When to prompt before each search.",
            current=self._settings.tool_web_search_permission,
            on_changed=self._on_ws_permission_changed,
        )
        ws_group.add(ws_perm_row)
        self._ws_results_row = ws_results_row
        self._ws_perm_row = ws_perm_row

        # ── Filesystem ──
        fs_group = Adw.PreferencesGroup(
            title="Filesystem",
            description=(
                "Lets the assistant read files inside a workspace folder you "
                "choose. Read-only by default; writes are a separate toggle. "
                "The assistant cannot access files outside the workspace."
            ),
        )
        page.add(fs_group)

        fs_switch = Adw.SwitchRow(title="Enable filesystem access")
        fs_switch.set_active(self._settings.tool_fs_enabled)
        fs_switch.connect("notify::active", self._on_fs_enabled_changed)
        fs_group.add(fs_switch)
        self._fs_switch = fs_switch

        fs_root_row = Adw.ActionRow(
            title="Workspace folder",
            subtitle=self._settings.tool_fs_root,
        )
        fs_root_btn = Gtk.Button(label="Choose…", valign=Gtk.Align.CENTER)
        fs_root_btn.connect("clicked", self._on_fs_choose_root)
        fs_root_row.add_suffix(fs_root_btn)
        fs_root_row.set_activatable_widget(fs_root_btn)
        fs_group.add(fs_root_row)
        self._fs_root_row = fs_root_row

        fs_writable_switch = Adw.SwitchRow(
            title="Allow writing and deleting files",
            subtitle=(
                "When OFF (default), the assistant can only read. When ON, "
                "fs_write / fs_delete are available — each call always "
                "prompts for permission (cannot be trusted permanently)."
            ),
        )
        fs_writable_switch.set_active(self._settings.tool_fs_writable)
        fs_writable_switch.connect("notify::active", self._on_fs_writable_changed)
        fs_group.add(fs_writable_switch)
        self._fs_writable_switch = fs_writable_switch

        fs_outside_switch = Adw.SwitchRow(
            title="Allow access outside the workspace",
            subtitle=(
                "When OFF (default), only the workspace folder is reachable. "
                "When ON, the assistant can request any path; Box prompts you "
                "for each one (per-path — approving one path doesn't grant "
                "others). The workspace stays the always-allowed base."
            ),
        )
        fs_outside_switch.set_active(self._settings.tool_fs_allow_outside)
        fs_outside_switch.connect(
            "notify::active", self._on_fs_allow_outside_changed
        )
        fs_group.add(fs_outside_switch)
        self._fs_outside_switch = fs_outside_switch

        # Persisted "Always allow" path grants — view and forget them here.
        fs_grants_expander = Adw.ExpanderRow(title="Always-allowed paths")
        fs_group.add(fs_grants_expander)
        self._fs_grants_expander = fs_grants_expander
        self._fs_grant_rows: list = []
        self._rebuild_fs_grants_rows()

        fs_perm_row = self._build_permission_combo(
            title="Permission (reads)",
            subtitle=(
                "When to prompt before each read. Writes always prompt."
            ),
            current=self._settings.tool_fs_permission,
            on_changed=self._on_fs_permission_changed,
        )
        fs_group.add(fs_perm_row)
        self._fs_perm_row = fs_perm_row

        # ── Agent mode (Phase 5) ──
        agent_group = Adw.PreferencesGroup(
            title="Agent mode",
            description=(
                "Lets the assistant chain several tool calls together to "
                "work through a multi-step task on its own. Needs at least "
                "one tool above enabled. A per-send cap stops it looping "
                "forever."
            ),
        )
        page.add(agent_group)

        agent_switch = Adw.SwitchRow(title="Enable agent mode")
        agent_switch.set_active(self._settings.agent_enabled)
        agent_switch.connect("notify::active", self._on_agent_enabled_changed)
        agent_group.add(agent_switch)
        self._agent_switch = agent_switch

        agent_iter_row = Adw.SpinRow.new_with_range(3, 15, 1)
        agent_iter_row.set_title("Max tool calls per message")
        agent_iter_row.set_subtitle(
            "How many tool calls the assistant may make while answering one "
            "message before it must stop and reply."
        )
        agent_iter_row.set_value(self._settings.agent_max_iterations)
        agent_iter_row.connect("notify::value", self._on_agent_iter_changed)
        agent_group.add(agent_iter_row)
        self._agent_iter_row = agent_iter_row

        # Agent style preset picker — fills the instructions box below.
        agent_preset_row = Adw.ComboRow(
            title="Agent style",
            subtitle=(
                "How the assistant approaches a task. 'Balanced' keeps it "
                "from restating earlier answers — the usual culprit when a "
                "follow-up gets the same reply."
            ),
            model=Gtk.StringList.new(
                [AGENT_PROMPT_PRESET_LABELS[k] for k in self._AGENT_PRESET_KEYS]
            ),
        )
        try:
            agent_preset_row.set_selected(
                self._AGENT_PRESET_KEYS.index(self._settings.agent_prompt_preset)
            )
        except ValueError:
            agent_preset_row.set_selected(0)
        agent_preset_row.connect("notify::selected", self._on_agent_preset_changed)
        agent_group.add(agent_preset_row)
        self._agent_preset_row = agent_preset_row

        agent_prompt_row = Adw.PreferencesRow(activatable=False)
        ap_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6,
                         margin_top=8, margin_bottom=8,
                         margin_start=12, margin_end=12)
        ap_label = Gtk.Label(label="Agent instructions", xalign=0)
        ap_label.add_css_class("caption-heading")
        ap_box.append(ap_label)
        self._agent_prompt_view = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR, accepts_tab=False,
        )
        self._agent_prompt_view.get_buffer().connect(
            "changed", self._on_agent_prompt_changed
        )
        ap_scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=120, max_content_height=120,
        )
        ap_scroller.set_child(self._agent_prompt_view)
        ap_frame = Gtk.Frame(child=ap_scroller)
        ap_box.append(ap_frame)
        ap_hint = Gtk.Label(
            label=(
                "Prepended to your system prompt while agent mode is on. "
                "Editable only when style is 'Custom'."
            ),
            xalign=0, wrap=True,
        )
        ap_hint.add_css_class("caption")
        ap_hint.add_css_class("dim-label")
        ap_box.append(ap_hint)
        agent_prompt_row.set_child(ap_box)
        agent_group.add(agent_prompt_row)
        self._agent_prompt_row = agent_prompt_row
        # Populate the box + lock/unlock it to match the selected preset.
        self._refresh_agent_prompt_view()

        # ── Tool safety (applies to every tool) ──
        safety_group = Adw.PreferencesGroup(
            title="Tool safety",
            description=(
                "Limits that apply to every tool, so a slow or stuck call "
                "can't freeze the app."
            ),
        )
        page.add(safety_group)

        timeout_row = Adw.SpinRow.new_with_range(0, 120, 1)
        timeout_row.set_title("Tool timeout (seconds)")
        timeout_row.set_subtitle(
            "Give up on any single tool call after this long and tell the "
            "model it timed out. 0 = no limit (not recommended)."
        )
        timeout_row.set_value(self._settings.tool_timeout_s)
        timeout_row.connect("notify::value", self._on_tool_timeout_changed)
        safety_group.add(timeout_row)

        # Sensitise per-tool sub-rows to their master switch.
        self._sync_tools_sensitivity()
        return page

    def _build_permission_combo(
        self,
        *,
        title: str,
        subtitle: str,
        current: str,
        on_changed,
    ) -> Adw.ComboRow:
        row = Adw.ComboRow(
            title=title,
            subtitle=subtitle,
            model=Gtk.StringList.new(list(self._PERMISSION_LABELS)),
        )
        try:
            row.set_selected(self._PERMISSION_IDS.index(current))
        except ValueError:
            row.set_selected(0)
        row.connect("notify::selected", on_changed)
        return row

    def _sync_tools_sensitivity(self) -> None:
        ws_on = self._settings.tool_web_search_enabled
        for row in (self._ws_results_row, self._ws_perm_row):
            row.set_sensitive(ws_on)
        fs_on = self._settings.tool_fs_enabled
        for row in (self._fs_root_row, self._fs_writable_switch,
                    self._fs_outside_switch, self._fs_grants_expander,
                    self._fs_perm_row):
            row.set_sensitive(fs_on)
        if hasattr(self, "_agent_iter_row"):
            agent_on = self._settings.agent_enabled
            for row in (self._agent_iter_row, self._agent_preset_row,
                        self._agent_prompt_row):
                row.set_sensitive(agent_on)

    # ── Web search handlers ──────────────────────────────────────────────
    def _on_ws_enabled_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        target = row.get_active()
        if target and not self._settings.tool_web_search_first_enable_acknowledged:
            self._confirm_first_enable(
                row,
                heading="Enable web search?",
                body=(
                    "The assistant will be able to search the public web "
                    "through DuckDuckGo whenever it judges a query needs "
                    "it. Results are filtered to HTTPS only. No API key "
                    "leaves your machine."
                ),
                ack_attr="tool_web_search_first_enable_acknowledged",
                enable_attr="tool_web_search_enabled",
            )
            return
        self._settings.tool_web_search_enabled = target
        self._settings.save()
        self._sync_tools_sensitivity()
        self._trigger_tools_reload()

    def _on_ws_results_changed(self, row: Adw.SpinRow, _pspec) -> None:
        self._settings.tool_web_search_results = int(row.get_value())
        self._settings.save()
        # No reload needed — closure reads settings live on each call.

    def _on_tool_timeout_changed(self, row: Adw.SpinRow, _pspec) -> None:
        # The timeout is baked into each tool callable when the engine
        # builds its tool list, so this applies on the NEXT engine load
        # (toggling a tool, switching chat/model, or restart) — same
        # deferred-apply discipline as the agent cap, which avoids a burst
        # of reloads while the user drags the spinner.
        self._settings.tool_timeout_s = int(row.get_value())
        self._settings.save()

    def _on_ws_permission_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._PERMISSION_IDS):
            self._settings.tool_web_search_permission = self._PERMISSION_IDS[idx]
            self._settings.save()

    # ── Filesystem handlers ──────────────────────────────────────────────
    def _on_fs_enabled_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        target = row.get_active()
        if target and not self._settings.tool_fs_first_enable_acknowledged:
            self._confirm_first_enable(
                row,
                heading="Enable filesystem access?",
                body=(
                    "The assistant will be able to read files inside the "
                    "workspace folder shown below. By default it cannot "
                    "write or delete anything — that's a separate toggle. "
                    "Paths outside the workspace are always refused."
                ),
                ack_attr="tool_fs_first_enable_acknowledged",
                enable_attr="tool_fs_enabled",
            )
            return
        self._settings.tool_fs_enabled = target
        self._settings.save()
        self._sync_tools_sensitivity()
        self._trigger_tools_reload()

    def _on_fs_writable_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.tool_fs_writable = row.get_active()
        self._settings.save()
        self._trigger_tools_reload()  # exposes/removes fs_write + fs_delete

    def _on_fs_allow_outside_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        active = row.get_active()
        self._settings.tool_fs_allow_outside = active
        self._settings.save()
        # Turning it off drops any in-memory grants so a later re-enable
        # starts clean (persisted "Always allow" grants survive in settings).
        if not active:
            from .tools.filesystem import clear_ephemeral_grants
            clear_ephemeral_grants()

    def _rebuild_fs_grants_rows(self) -> None:
        """(Re)populate the 'Always-allowed paths' expander from settings."""
        exp = self._fs_grants_expander
        for row in self._fs_grant_rows:
            exp.remove(row)
        self._fs_grant_rows = []
        roots = list(self._settings.tool_fs_extra_roots or [])
        if not roots:
            exp.set_subtitle(
                "None yet — paths you choose “Always allow” for appear here."
            )
            exp.set_enable_expansion(False)
            return
        exp.set_subtitle(
            f"{len(roots)} path(s) granted outside the workspace."
        )
        exp.set_enable_expansion(True)
        for path in roots:
            r = Adw.ActionRow(title=path)
            btn = Gtk.Button(
                icon_name="user-trash-symbolic", valign=Gtk.Align.CENTER
            )
            btn.add_css_class("flat")
            btn.set_tooltip_text("Forget this path")
            btn.connect("clicked", self._on_forget_fs_grant, path)
            r.add_suffix(btn)
            exp.add_row(r)
            self._fs_grant_rows.append(r)
        clear_row = Adw.ActionRow(title="Forget all granted paths")
        clear_btn = Gtk.Button(label="Forget all", valign=Gtk.Align.CENTER)
        clear_btn.add_css_class("destructive-action")
        clear_btn.connect("clicked", self._on_forget_all_fs_grants)
        clear_row.add_suffix(clear_btn)
        exp.add_row(clear_row)
        self._fs_grant_rows.append(clear_row)

    def _on_forget_fs_grant(self, _btn: Gtk.Button, path: str) -> None:
        from .tools.filesystem import forget_persisted_path
        forget_persisted_path(self._settings, path)
        self._rebuild_fs_grants_rows()

    def _on_forget_all_fs_grants(self, _btn: Gtk.Button) -> None:
        self._settings.tool_fs_extra_roots = []
        self._settings.save()
        self._rebuild_fs_grants_rows()

    def _on_fs_permission_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._PERMISSION_IDS):
            self._settings.tool_fs_permission = self._PERMISSION_IDS[idx]
            self._settings.save()

    # ── Agent-mode handlers (Phase 5) ────────────────────────────────────
    def _on_agent_enabled_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        target = row.get_active()
        if target and not self._settings.agent_first_enable_acknowledged:
            dlg = Adw.AlertDialog(
                heading="Enable agent mode?",
                body=(
                    "The assistant will chain several tool calls together to "
                    "work through a task on its own — searching the web or "
                    "reading files across multiple steps before replying. "
                    "Each enabled tool still asks for permission as configured "
                    "above, and a per-message cap stops runaway loops."
                ),
            )
            dlg.add_response("cancel", "Cancel")
            dlg.add_response("enable", "Enable")
            dlg.set_response_appearance("enable", Adw.ResponseAppearance.SUGGESTED)
            dlg.set_default_response("enable")
            dlg.set_close_response("cancel")

            def _on_response(_dlg, response: str) -> None:
                if response == "enable":
                    self._settings.agent_first_enable_acknowledged = True
                    self._settings.agent_enabled = True
                    self._settings.save()
                    self._sync_tools_sensitivity()
                    self._trigger_agent_reload()
                else:
                    row.handler_block_by_func(self._on_agent_enabled_changed)
                    row.set_active(False)
                    row.handler_unblock_by_func(self._on_agent_enabled_changed)

            dlg.connect("response", _on_response)
            dlg.present(self._main_win())
            return
        self._settings.agent_enabled = target
        self._settings.save()
        self._sync_tools_sensitivity()
        self._trigger_agent_reload()

    def _on_agent_iter_changed(self, row: Adw.SpinRow, _pspec) -> None:
        # Saved live; the cap is baked into the handler at engine-load time,
        # so it takes effect on the next reload (e.g. next agent-mode toggle
        # or model switch). We don't reload here to avoid SpinRow drag-spam.
        self._settings.agent_max_iterations = int(row.get_value())
        self._settings.save()

    def _on_agent_prompt_changed(self, buf: Gtk.TextBuffer) -> None:
        # Only persist edits in Custom mode — in preset mode the box mirrors
        # a fixed preset and must not overwrite the user's saved custom text.
        # Saved on every edit, applied on next engine load (per-keystroke
        # reloads would be pathological).
        if self._settings.agent_prompt_preset != "custom":
            return
        self._settings.agent_system_prompt = buf.get_text(
            buf.get_start_iter(), buf.get_end_iter(), False
        )
        self._settings.save()

    def _on_agent_preset_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if not (0 <= idx < len(self._AGENT_PRESET_KEYS)):
            return
        key = self._AGENT_PRESET_KEYS[idx]
        prev = self._settings.agent_prompt_preset
        # Seed the custom box from whatever was showing, so switching to
        # Custom gives the user an editable starting point rather than blank.
        if key == "custom" and prev != "custom":
            self._settings.agent_system_prompt = AGENT_PROMPT_PRESETS.get(
                prev, AGENT_PROMPT_PRESETS["balanced"]
            )
        self._settings.agent_prompt_preset = key
        self._settings.save()
        self._refresh_agent_prompt_view()
        self._trigger_agent_reload()

    def _refresh_agent_prompt_view(self) -> None:
        """Sync the instructions box to the selected preset and lock it
        unless the preset is Custom."""
        key = self._settings.agent_prompt_preset
        custom = key == "custom"
        text = (
            self._settings.agent_system_prompt if custom
            else AGENT_PROMPT_PRESETS.get(key, AGENT_PROMPT_PRESETS["balanced"])
        )
        buf = self._agent_prompt_view.get_buffer()
        buf.handler_block_by_func(self._on_agent_prompt_changed)
        buf.set_text(text)
        buf.handler_unblock_by_func(self._on_agent_prompt_changed)
        self._agent_prompt_view.set_editable(custom)
        # Visually signal the locked state without hiding the text.
        if custom:
            self._agent_prompt_view.remove_css_class("dim-label")
        else:
            self._agent_prompt_view.add_css_class("dim-label")

    def _trigger_agent_reload(self) -> None:
        win = self._main_win()
        if win is not None and hasattr(win, "refresh_agent_for_current_conv"):
            win.refresh_agent_for_current_conv()

    def _on_fs_choose_root(self, _btn: Gtk.Button) -> None:
        dlg = Gtk.FileDialog(
            title="Choose workspace folder",
            modal=True,
        )
        current = Path(self._settings.tool_fs_root).expanduser()
        if current.is_dir():
            dlg.set_initial_folder(Gio.File.new_for_path(str(current)))

        def _done(_dialog, result):
            try:
                folder = dlg.select_folder_finish(result)
            except Exception:
                return
            if folder is None:
                return
            path = folder.get_path()
            if not path:
                return
            self._settings.tool_fs_root = path
            self._settings.save()
            self._fs_root_row.set_subtitle(path)

        dlg.select_folder(self._main_win(), None, _done)

    # ── First-time-enable confirm ────────────────────────────────────────
    def _confirm_first_enable(
        self,
        row: Adw.SwitchRow,
        *,
        heading: str,
        body: str,
        ack_attr: str,
        enable_attr: str,
    ) -> None:
        dlg = Adw.AlertDialog(heading=heading, body=body)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("enable", "Enable")
        dlg.set_response_appearance("enable", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("enable")
        dlg.set_close_response("cancel")

        def _on_response(_dlg, response: str) -> None:
            if response == "enable":
                setattr(self._settings, ack_attr, True)
                setattr(self._settings, enable_attr, True)
                self._settings.save()
                self._sync_tools_sensitivity()
                self._trigger_tools_reload()
            else:
                # User declined — flip the switch back without re-firing
                # the handler.
                row.handler_block_by_func(
                    self._on_ws_enabled_changed
                    if enable_attr == "tool_web_search_enabled"
                    else self._on_fs_enabled_changed
                )
                row.set_active(False)
                row.handler_unblock_by_func(
                    self._on_ws_enabled_changed
                    if enable_attr == "tool_web_search_enabled"
                    else self._on_fs_enabled_changed
                )

        dlg.connect("response", _on_response)
        dlg.present(self._main_win())

    def _trigger_tools_reload(self) -> None:
        win = self._main_win()
        if win is not None and hasattr(win, "refresh_tools_for_current_conv"):
            win.refresh_tools_for_current_conv()

    # ──────────────────────────────────────────────────────────────────────
    def _build_appearance_page(self) -> Adw.PreferencesPage:
        from .themes import ALL_THEMES, THEME_ACCENTS

        page = Adw.PreferencesPage(title="Appearance", icon_name="applications-graphics-symbolic")

        # ── Navigation rail (ATK-inspired) ────────────────────────────────
        nav_group = Adw.PreferencesGroup(
            title="Navigation",
            description="The app-wide nav rail. Changes apply instantly.",
        )
        page.add(nav_group)

        _nav_positions = ["left", "right", "top", "bottom"]
        nav_pos_row = Adw.ComboRow(
            title="Rail position",
            model=Gtk.StringList.new(["Left", "Right", "Top", "Bottom"]),
        )
        try:
            nav_pos_row.set_selected(
                _nav_positions.index(self._settings.nav_position)
            )
        except ValueError:
            nav_pos_row.set_selected(0)

        def _on_nav_pos(row, *_a) -> None:
            self._settings.nav_position = _nav_positions[row.get_selected()]
            self._settings.save()
            if self._window is not None:
                self._window.rebuild_nav_rail()

        nav_pos_row.connect("notify::selected", _on_nav_pos)
        nav_group.add(nav_pos_row)

        nav_labels_row = Adw.SwitchRow(
            title="Show labels",
            subtitle="Display each item's name with its icon",
        )
        nav_labels_row.set_active(self._settings.nav_labels)

        def _on_nav_labels(row, *_a) -> None:
            self._settings.nav_labels = row.get_active()
            self._settings.save()
            if self._window is not None:
                self._window.rebuild_nav_rail()

        nav_labels_row.connect("notify::active", _on_nav_labels)
        nav_group.add(nav_labels_row)

        # ── Header font ───────────────────────────────────────────────────
        hf_group = Adw.PreferencesGroup(
            title="Header font",
            description="Typeface for titles and headings app-wide "
            "(window titles, sidebar, New Chat…). Bundled fonts like "
            "DotGothic16 work too.",
        )
        page.add(hf_group)
        hf_row = Adw.ActionRow(title="Header font")
        hf_dialog = Gtk.FontDialog(title="Choose a header font")
        hf_btn = Gtk.FontDialogButton(dialog=hf_dialog)
        hf_btn.set_valign(Gtk.Align.CENTER)
        if self._settings.header_font_family:
            hf_btn.set_font_desc(Pango.FontDescription.from_string(
                self._settings.header_font_family
            ))

        def _on_hf_picked(btn, *_a) -> None:
            desc = btn.get_font_desc()
            self._settings.header_font_family = (
                desc.get_family() or "" if desc else ""
            )
            self._settings.save()
            self._app.refresh_theme()

        hf_btn.connect("notify::font-desc", _on_hf_picked)
        hf_row.add_suffix(hf_btn)
        hf_reset = Gtk.Button(label="Reset", valign=Gtk.Align.CENTER)
        hf_reset.add_css_class("flat")

        def _on_hf_reset(*_a) -> None:
            self._settings.header_font_family = ""
            self._settings.save()
            self._app.refresh_theme()

        hf_reset.connect("clicked", _on_hf_reset)
        hf_row.add_suffix(hf_reset)
        hf_group.add(hf_row)

        # ── Glass mode (iOS-style translucency) ──────────────────────────
        glass_group = Adw.PreferencesGroup(
            title="Glass",
            description="Translucent, luminous surfaces over your desktop. "
            "Blur behind windows depends on the compositor.",
        )
        page.add(glass_group)

        glass_row = Adw.SwitchRow(
            title="Glass mode",
            subtitle="See-through window with hairline glass edges",
        )
        glass_row.set_active(self._settings.glass_mode)

        def _on_glass(row, *_a) -> None:
            self._settings.glass_mode = row.get_active()
            self._settings.save()
            self._app.refresh_theme()

        glass_row.connect("notify::active", _on_glass)
        glass_group.add(glass_row)

        liquid_row = Adw.SwitchRow(
            title="Liquid glass",
            subtitle="Lens-shaped pill controls, specular highlights and an "
            "accent light-wash (includes Glass mode)",
        )
        liquid_row.set_active(self._settings.glass_liquid)

        def _on_liquid(row, *_a) -> None:
            self._settings.glass_liquid = row.get_active()
            self._settings.save()
            self._app.refresh_theme()

        liquid_row.connect("notify::active", _on_liquid)
        glass_group.add(liquid_row)

        glass_op_row = Adw.SpinRow.new_with_range(0.5, 1.0, 0.02)
        glass_op_row.set_title("Glass opacity")
        glass_op_row.set_subtitle("Lower is more transparent")
        glass_op_row.set_digits(2)
        glass_op_row.set_value(self._settings.glass_opacity)

        def _on_glass_op(row, *_a) -> None:
            self._settings.glass_opacity = round(row.get_value(), 2)
            self._settings.save()
            if self._settings.glass_mode:
                self._app.refresh_theme()

        glass_op_row.connect("notify::value", _on_glass_op)
        glass_group.add(glass_op_row)

        group = Adw.PreferencesGroup(
            title="Theme",
            description="Catppuccin, Dracula, and 44 Ptyxis terminal palettes.",
        )
        page.add(group)

        # Derive the picker straight from ALL_THEMES: the 6 curated base themes
        # first (stable order), then every Ptyxis palette by insertion order.
        _base_order = [
            "catppuccin-latte",
            "catppuccin-frappe",
            "catppuccin-macchiato",
            "catppuccin-mocha",
            "dracula",
            "dracula-pro",
        ]
        _rest = [tid for tid in ALL_THEMES if tid not in _base_order]
        self._theme_ids = _base_order + _rest
        names = [ALL_THEMES[tid].name for tid in self._theme_ids]
        theme_row = Adw.ComboRow(title="Theme", model=Gtk.StringList.new(names))
        try:
            theme_row.set_selected(self._theme_ids.index(self._settings.theme))
        except ValueError:
            theme_row.set_selected(0)
        theme_row.connect("notify::selected", self._on_theme_changed)
        group.add(theme_row)

        # Accent picker — theme-specific; first entry "Theme default" snaps back.
        self._accent_names: list[str] = []
        self._accent_row = Adw.ComboRow(
            title="Accent colour",
            subtitle="Buttons, links, sidebar selection, composer focus",
        )
        self._accent_row.connect("notify::selected", self._on_accent_changed)
        group.add(self._accent_row)

        # Bubble accent picker — affects user bubble border + role labels only.
        self._bubble_accent_names: list[str] = []
        self._bubble_accent_row = Adw.ComboRow(
            title="Bubble accent colour",
            subtitle="User bubble border + role labels (uses main accent if 'Theme default')",
        )
        self._bubble_accent_row.connect("notify::selected", self._on_bubble_accent_changed)
        group.add(self._bubble_accent_row)

        self._refresh_accent_model()   # populate both for current theme

        font_group = Adw.PreferencesGroup(title="Text")
        page.add(font_group)

        font_row = Adw.SpinRow.new_with_range(10, 22, 1)
        font_row.set_title("Message font size")
        font_row.set_value(self._settings.font_size)
        font_row.connect("notify::value", self._on_font_size_changed)
        font_group.add(font_row)

        font_family_row = Adw.ActionRow(
            title="Message font family",
            subtitle="Click to pick from all installed fonts; Reset for system default",
        )
        font_dialog = Gtk.FontDialog(title="Choose a font for chat messages")
        self._font_btn = Gtk.FontDialogButton(dialog=font_dialog)
        self._font_btn.set_valign(Gtk.Align.CENTER)
        # Apply the saved font (if any) as the button's initial display.
        if self._settings.chat_font_family:
            desc = Pango.FontDescription.from_string(self._settings.chat_font_family)
            self._font_btn.set_font_desc(desc)
        self._font_btn.connect("notify::font-desc", self._on_font_family_picked)
        font_family_row.add_suffix(self._font_btn)
        font_reset_btn = Gtk.Button(label="Reset", valign=Gtk.Align.CENTER)
        font_reset_btn.add_css_class("flat")
        font_reset_btn.connect("clicked", self._on_font_family_reset)
        font_family_row.add_suffix(font_reset_btn)
        font_group.add(font_family_row)

        composer_group = Adw.PreferencesGroup(title="Composer")
        page.add(composer_group)

        accent_input_row = Adw.SwitchRow(
            title="Accent colour input box",
            subtitle="Use the accent colour as the chat input background",
        )
        accent_input_row.set_active(self._settings.composer_use_accent)
        accent_input_row.connect("notify::active", self._on_composer_accent_changed)
        composer_group.add(accent_input_row)

        bubble_group = Adw.PreferencesGroup(title="Bubbles")
        page.add(bubble_group)

        from .themes import BUBBLE_PALETTES
        # Derive bubble picker from BUBBLE_PALETTES: "default" (None) → "Theme
        # Default", every other entry uses its own "display" label.
        self._bubble_ids = list(BUBBLE_PALETTES.keys())
        bubble_names = [
            "Theme Default" if bid == "default"
            else (BUBBLE_PALETTES[bid] or {}).get("display", bid.title())
            for bid in self._bubble_ids
        ]
        bubble_row = Adw.ComboRow(
            title="Colour palette",
            subtitle="Preset colour scheme for chat bubbles",
            model=Gtk.StringList.new(bubble_names),
        )
        try:
            bubble_row.set_selected(self._bubble_ids.index(self._settings.bubble_style))
        except ValueError:
            bubble_row.set_selected(0)
        bubble_row.connect("notify::selected", self._on_bubble_style_changed)
        bubble_group.add(bubble_row)

        # Opacity slider row
        opacity_pref_row = Adw.PreferencesRow()
        opacity_pref_row.set_activatable(False)
        opacity_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=12,
            margin_top=10, margin_bottom=10, margin_start=16, margin_end=16,
            valign=Gtk.Align.CENTER,
        )
        opacity_label = Gtk.Label(label="Opacity", xalign=0.0, width_chars=8)
        opacity_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0.1, 1.0, 0.05)
        opacity_scale.set_value(self._settings.bubble_opacity)
        opacity_scale.set_draw_value(True)
        opacity_scale.set_digits(2)
        opacity_scale.set_hexpand(True)
        opacity_scale.connect("value-changed", self._on_bubble_opacity_changed)
        opacity_box.append(opacity_label)
        opacity_box.append(opacity_scale)
        opacity_pref_row.set_child(opacity_box)
        bubble_group.add(opacity_pref_row)

        # User bubble text colour
        user_color_row = Adw.ActionRow(
            title="User text colour",
            subtitle="Custom text colour; Reset to follow palette",
        )
        user_color_dialog = Gtk.ColorDialog(title="User bubble text", with_alpha=False)
        self._user_color_btn = Gtk.ColorDialogButton(dialog=user_color_dialog)
        init_rgba = Gdk.RGBA()
        init_rgba.parse(self._settings.user_bubble_text_color or "#888888")
        self._user_color_btn.set_rgba(init_rgba)
        self._user_color_btn.set_valign(Gtk.Align.CENTER)
        self._user_color_btn.connect("notify::rgba", self._on_user_text_color_changed)
        user_color_row.add_suffix(self._user_color_btn)
        user_reset_btn = Gtk.Button(label="Reset", valign=Gtk.Align.CENTER)
        user_reset_btn.add_css_class("flat")
        user_reset_btn.connect("clicked", self._on_user_text_color_reset)
        user_color_row.add_suffix(user_reset_btn)
        bubble_group.add(user_color_row)

        # AI bubble text colour
        asst_color_row = Adw.ActionRow(
            title="AI text colour",
            subtitle="Custom text colour; Reset to follow palette",
        )
        asst_color_dialog = Gtk.ColorDialog(title="AI bubble text", with_alpha=False)
        self._asst_color_btn = Gtk.ColorDialogButton(dialog=asst_color_dialog)
        init_rgba2 = Gdk.RGBA()
        init_rgba2.parse(self._settings.asst_bubble_text_color or "#888888")
        self._asst_color_btn.set_rgba(init_rgba2)
        self._asst_color_btn.set_valign(Gtk.Align.CENTER)
        self._asst_color_btn.connect("notify::rgba", self._on_asst_text_color_changed)
        asst_color_row.add_suffix(self._asst_color_btn)
        asst_reset_btn = Gtk.Button(label="Reset", valign=Gtk.Align.CENTER)
        asst_reset_btn.add_css_class("flat")
        asst_reset_btn.connect("clicked", self._on_asst_text_color_reset)
        asst_color_row.add_suffix(asst_reset_btn)
        bubble_group.add(asst_color_row)

        return page

    def _refresh_accent_model(self) -> None:
        from .themes import THEME_ACCENTS
        accents = THEME_ACCENTS.get(self._settings.theme, {})
        # First entry is the "snap back" option; subsequent entries are the
        # named palette accents.  An empty saved accent_name maps to index 0.
        self._accent_names = ["Theme default"] + list(accents.keys())

        def _resolve_idx(saved: str) -> int:
            if not saved:
                return 0
            try:
                return self._accent_names.index(saved)
            except ValueError:
                return 0

        # Main accent
        self._accent_row.handler_block_by_func(self._on_accent_changed)
        self._accent_row.set_model(Gtk.StringList.new(self._accent_names))
        self._accent_row.set_selected(_resolve_idx(self._settings.accent_name))
        self._accent_row.handler_unblock_by_func(self._on_accent_changed)

        # Bubble accent uses the same palette list
        self._bubble_accent_names = self._accent_names
        self._bubble_accent_row.handler_block_by_func(self._on_bubble_accent_changed)
        self._bubble_accent_row.set_model(Gtk.StringList.new(self._bubble_accent_names))
        self._bubble_accent_row.set_selected(_resolve_idx(self._settings.bubble_accent_name))
        self._bubble_accent_row.handler_unblock_by_func(self._on_bubble_accent_changed)

    # ── Handlers ───────────────────────────────────────────────────────────
    def _on_download_models(self, _row) -> None:
        from .download_dialog import DownloadDialog
        DownloadDialog(on_model_selected=self._set_model).present(self.get_root())

    def _on_choose_model(self, _btn) -> None:
        dlg = Gtk.FileDialog(title="Choose a model")
        try:
            dlg.set_initial_folder(Gio.File.new_for_path(str(MODELS_DIR)))
        except Exception:  # noqa: BLE001
            pass

        f = Gtk.FileFilter(name="Models (*.litertlm, *.task, *.gguf)")
        f.add_pattern("*.litertlm")
        f.add_pattern("*.task")
        f.add_pattern("*.gguf")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dlg.set_filters(filters)

        dlg.open(self.get_root(), None, self._on_file_chosen)

    def _on_file_chosen(self, dialog: Gtk.FileDialog, result) -> None:
        try:
            file = dialog.open_finish(result)
        except Exception:  # cancelled / failed
            return
        if file is None:
            return
        path = file.get_path()
        if path:
            self._set_model(path)

    def _set_model(self, path: str) -> None:
        self._settings.model_path = path
        self._settings.add_recent_model(path)
        self._settings.save()
        if hasattr(self, "_model_row"):
            self._model_row.set_subtitle(path)
        win = self._main_win()
        if win is not None:
            win.on_model_changed(path)

    def _register_sd_model(self, path: str) -> None:
        """A downloaded SD model's 'Use' button — register it for the Generate
        page (it's an image model, not the chat LLM)."""
        self._settings.add_sd_model(path)
        self._settings.sd_last_model = path
        self._settings.save()
        self._memory_toast(f"Added {Path(path).name} to Image Tools → Generate")

    # ── Models page (download catalog + import) ─────────────────────────
    def _build_models_page(self) -> Adw.PreferencesPage:
        from .model_catalog import MODELS, SD_MODELS, ModelRow

        page = Adw.PreferencesPage(title="Models", icon_name="folder-download-symbolic")

        dl_group = Adw.PreferencesGroup(
            title="Chat models",
            description=(
                "No account required. Saved to ~/.local/share/box/models/. "
                "Downloads are verified by SHA-256."
            ),
        )
        page.add(dl_group)
        for model in MODELS:
            dl_group.add(ModelRow(model, on_use=self._set_model))

        # Stable Diffusion image models — single-file, same verified downloader
        # as chat models, but "Use" registers them as SD models for the
        # Image Tools → Generate page.
        sd_group = Adw.PreferencesGroup(
            title="Image models — Stable Diffusion",
            description="For Image Tools → Generate (stable-diffusion.cpp engine).",
        )
        page.add(sd_group)
        for model in SD_MODELS:
            sd_group.add(ModelRow(model, on_use=self._register_sd_model))

        # On-device text-to-image (LiteRT) — big multi-file directory downloads
        # (Z-Image / FLUX klein). Registered for the Generate page's LiteRT engine.
        from .litert_diffusion_models import LITERT_MODELS
        from .litert_download_dialog import LiterModelRow
        litert_group = Adw.PreferencesGroup(
            title="Image models — LiteRT (Z-Image / FLUX klein)",
            description=(
                "Large multi-file downloads that resume if interrupted. The Qwen "
                "tokenizer (~0.8 GB) is shared across both models."
            ),
        )
        page.add(litert_group)
        for model in LITERT_MODELS:
            litert_group.add(LiterModelRow(model, self._settings))

        import_group = Adw.PreferencesGroup(
            title="Import",
            description="Already have a model file? Point Box at it directly.",
        )
        page.add(import_group)

        import_litertlm_row = Adw.ActionRow(
            title="Import a LiteRT-LM file",
            subtitle="*.litertlm, *.task — used immediately",
            activatable=True,
        )
        import_litertlm_row.add_suffix(
            Gtk.Image(icon_name="go-next-symbolic", valign=Gtk.Align.CENTER))
        import_litertlm_row.connect("activated", lambda _r: self._on_choose_model(None))
        import_group.add(import_litertlm_row)

        import_gguf_row = Adw.ActionRow(
            title="Import a GGUF file",
            subtitle="*.gguf — used immediately via the llama.cpp engine",
            activatable=True,
        )
        import_gguf_row.add_suffix(
            Gtk.Image(icon_name="go-next-symbolic", valign=Gtk.Align.CENTER))
        import_gguf_row.connect("activated", self._on_import_gguf)
        import_group.add(import_gguf_row)

        self._models_page = page
        self._gguf_group = None
        self._gguf_rows: dict[str, Adw.ActionRow] = {}
        if self._settings.imported_gguf_models:
            self._ensure_gguf_group()
            for p in self._settings.imported_gguf_models:
                self._add_gguf_row(p)
        return page

    def _ensure_gguf_group(self) -> Adw.PreferencesGroup:
        if self._gguf_group is None:
            self._gguf_group = Adw.PreferencesGroup(
                title="Imported GGUF models",
                description="Run on the llama.cpp engine — tune it on the "
                            "Llama.cpp page.",
            )
            self._models_page.add(self._gguf_group)
        return self._gguf_group

    def _add_gguf_row(self, path: str) -> None:
        existing = self._gguf_rows.pop(path, None)
        if existing is not None and self._gguf_group is not None:
            self._gguf_group.remove(existing)
        row = self._build_gguf_row(path)
        self._gguf_rows[path] = row
        self._ensure_gguf_group().add(row)

    def _build_gguf_row(self, path: str) -> Adw.ActionRow:
        row = Adw.ActionRow(title=Path(path).name, subtitle=path)
        use_btn = Gtk.Button(label="Use", valign=Gtk.Align.CENTER)
        use_btn.add_css_class("suggested-action")
        use_btn.connect("clicked", lambda _b: self._set_model(path))
        row.add_suffix(use_btn)
        forget_btn = Gtk.Button(label="Forget", valign=Gtk.Align.CENTER)
        forget_btn.add_css_class("flat")

        def _forget(_b) -> None:
            self._settings.forget_imported_gguf(path)
            self._settings.save()
            self._gguf_rows.pop(path, None)
            if self._gguf_group is not None:
                self._gguf_group.remove(row)
                if not self._gguf_rows:
                    self._models_page.remove(self._gguf_group)
                    self._gguf_group = None

        forget_btn.connect("clicked", _forget)
        row.add_suffix(forget_btn)
        return row

    def _on_import_gguf(self, _row) -> None:
        dlg = Gtk.FileDialog(title="Choose a GGUF model")
        try:
            dlg.set_initial_folder(Gio.File.new_for_path(str(MODELS_DIR)))
        except Exception:  # noqa: BLE001
            pass
        f = Gtk.FileFilter(name="GGUF models (*.gguf)")
        f.add_pattern("*.gguf")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dlg.set_filters(filters)

        def _done(dialog, result) -> None:
            try:
                file = dialog.open_finish(result)
            except Exception:
                return
            if file is None:
                return
            path = file.get_path()
            if path:
                self._on_import_gguf_path(path)

        dlg.open(self._main_win() or self.get_root(), None, _done)

    def _on_import_gguf_path(self, path: str) -> None:
        self._settings.add_imported_gguf(path)
        self._settings.save()
        self._add_gguf_row(path)
        self._set_model(path)

    # ── Security page (App Lock + sandbox status) ───────────────────────
    def _build_security_page(self) -> Adw.PreferencesPage:
        page = Adw.PreferencesPage(title="Security", icon_name="channel-secure-symbolic")

        group = Adw.PreferencesGroup(
            title="App Lock",
            description=(
                "A password gate on the app window. This is not encryption — "
                "Box has no encrypted volumes or files; it only protects "
                "casual access to the app UI."
            ),
        )
        page.add(group)

        has_pw = bool(self._settings.app_lock_hash)
        applock_enable = Adw.SwitchRow(
            title="Lock the app behind a password",
            active=self._settings.app_lock_enabled and has_pw,
        )
        group.add(applock_enable)

        applock_set = Adw.ActionRow(
            title="App password", subtitle="Set" if has_pw else "Not set",
        )
        applock_btn = Gtk.Button(
            label="Change password…" if has_pw else "Set password…",
            valign=Gtk.Align.CENTER,
        )
        applock_btn.add_css_class("pill")
        applock_set.add_suffix(applock_btn)
        group.add(applock_set)

        applock_btn.connect(
            "clicked",
            lambda _b: self._open_set_app_password_dialog(
                applock_enable, applock_set, applock_btn),
        )

        def _on_applock_toggled(row: Adw.SwitchRow, _pspec) -> None:
            if row.get_active():
                if not self._settings.app_lock_hash:
                    self._open_set_app_password_dialog(
                        applock_enable, applock_set, applock_btn)
                else:
                    self._settings.app_lock_enabled = True
                    self._settings.save()
            else:
                self._settings.app_lock_enabled = False
                self._settings.app_lock_hash = ""
                self._settings.save()
                applock_set.set_subtitle("Not set")
                applock_btn.set_label("Set password…")

        applock_enable.connect("notify::active", _on_applock_toggled)

        note_group = Adw.PreferencesGroup(
            description="Lock the app any time from the header menu (☰ → Lock Now) or Ctrl+L."
        )
        page.add(note_group)

        # ── Inference sandbox status (probed, not assumed) ──
        sbx_group = Adw.PreferencesGroup(
            title="Inference sandbox",
            description=(
                "GGUF models and image generation run in separate processes, "
                "locked down with the strongest mechanism this machine "
                "supports. Status below is probed, not assumed."
            ),
        )
        page.add(sbx_group)
        sbx_row = Adw.ActionRow(title="Sandbox mechanism", subtitle="Probing…")
        sbx_group.add(sbx_row)
        sbx_group.add(Adw.ActionRow(title="Always on", subtitle=(
            "Loopback-only API · per-session random port and token · "
            "no-new-privileges · token never on the command line"
        )))

        def _probe_sandbox() -> None:
            from . import sandbox as sbx

            abi = sbx.landlock_abi()
            if abi > 0:
                headline = f"Landlock LSM (kernel ABI v{abi})"
                detail = ("Per-model file grants · bind restricted to one port · "
                          "outbound connections denied")
            elif sbx.systemd_user_available():
                verdicts = sbx.probe_systemd_properties()
                ok = [k for k, v in verdicts.items() if v]
                no = [k for k, v in verdicts.items() if not v]
                headline = (f"systemd user sandbox — {len(ok)} of "
                            f"{len(verdicts)} properties verified enforced")
                parts = []
                if ok:
                    parts.append("Enforced: " + ", ".join(ok))
                if no:
                    parts.append("Not enforceable here: " + ", ".join(no))
                detail = " · ".join(parts)
            else:
                headline = "Baseline hardening only"
                detail = ("No kernel sandbox available on this system — the "
                          "always-on protections below still apply")

            def _apply() -> bool:
                sbx_row.set_subtitle(headline)
                sbx_row.set_tooltip_text(detail)
                return False

            GLib.idle_add(_apply)

        threading.Thread(target=_probe_sandbox, daemon=True).start()
        return page

    def _open_set_app_password_dialog(self, enable_row, set_row, btn) -> None:
        from .applock import hash_password

        dlg = Adw.AlertDialog(
            heading="Set app password",
            body="Enter a password to lock the app window. This is not "
                 "encryption — it only gates casual access to the UI.",
        )
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        pw1 = Gtk.PasswordEntry(show_peek_icon=True)
        pw1.set_property("placeholder-text", "Password")
        pw2 = Gtk.PasswordEntry(show_peek_icon=True)
        pw2.set_property("placeholder-text", "Confirm password")
        box.append(pw1)
        box.append(pw2)
        dlg.set_extra_child(box)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("save", "Set password")
        dlg.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("save")
        dlg.set_close_response("cancel")

        def _resp(_d, response: str) -> None:
            if response != "save":
                return
            p1, p2 = pw1.get_text(), pw2.get_text()
            if not p1 or p1 != p2:
                self._show_pref_toast("Passwords empty or don't match.")
                return
            self._settings.app_lock_hash = hash_password(p1)
            self._settings.app_lock_enabled = True
            self._settings.save()
            set_row.set_subtitle("Set")
            btn.set_label("Change password…")
            enable_row.set_active(True)

        dlg.connect("response", _resp)
        dlg.present(self._main_win() or self.get_root())

    def _show_pref_toast(self, text: str) -> None:
        win = self._main_win()
        if win is not None and hasattr(win, "_show_toast"):
            win._show_toast(text)

    def _on_system_prompt_changed(self, buf: Gtk.TextBuffer) -> None:
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        self._settings.system_prompt = text
        self._settings.save()
        # Take effect on the next conversation reload — we don't yank an
        # in-flight one out from under the user.

    def _on_theme_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._theme_ids):
            self._settings.theme = self._theme_ids[idx]
            self._settings.accent_name = ""        # reset to theme default on theme switch
            self._settings.bubble_accent_name = ""
            self._settings.save()
            self._refresh_accent_model()
            self._app.refresh_theme()

    def _on_accent_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._accent_names):
            # Index 0 is "Theme default" — store as empty string.
            self._settings.accent_name = "" if idx == 0 else self._accent_names[idx]
            self._settings.save()
            self._app.refresh_theme()

    def _on_bubble_accent_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._bubble_accent_names):
            self._settings.bubble_accent_name = "" if idx == 0 else self._bubble_accent_names[idx]
            self._settings.save()
            self._app.refresh_theme()

    def _on_font_size_changed(self, row: Adw.SpinRow, _pspec) -> None:
        size = int(row.get_value())
        self._settings.font_size = size
        self._settings.save()
        self._app.refresh_theme()

    def _on_composer_accent_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.composer_use_accent = row.get_active()
        self._settings.save()
        self._app.refresh_theme()

    def _on_bubble_style_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._bubble_ids):
            self._settings.bubble_style = self._bubble_ids[idx]
            self._settings.save()
            self._app.refresh_theme()

    def _on_bubble_opacity_changed(self, scale: Gtk.Scale) -> None:
        self._settings.bubble_opacity = round(scale.get_value(), 2)
        self._settings.save()
        self._app.refresh_theme()

    def _on_font_family_picked(self, btn: Gtk.FontDialogButton, _pspec) -> None:
        desc = btn.get_font_desc()
        # Strip size — we use the SpinRow above for that. Just keep family/style.
        family = desc.get_family() if desc else ""
        self._settings.chat_font_family = family or ""
        self._settings.save()
        self._app.refresh_theme()

    def _on_font_family_reset(self, _btn: Gtk.Button) -> None:
        self._settings.chat_font_family = ""
        self._settings.save()
        # Reset the button to a neutral default so the UI reflects the change.
        self._font_btn.set_font_desc(Pango.FontDescription.from_string("Sans"))
        self._app.refresh_theme()

    def _on_user_text_color_changed(self, btn: Gtk.ColorDialogButton, _pspec) -> None:
        rgba = btn.get_rgba()
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba.red * 255), int(rgba.green * 255), int(rgba.blue * 255),
        )
        self._settings.user_bubble_text_color = hex_color
        self._settings.save()
        self._app.refresh_theme()

    def _on_user_text_color_reset(self, _btn) -> None:
        self._user_color_btn.handler_block_by_func(self._on_user_text_color_changed)
        grey = Gdk.RGBA()
        grey.parse("#888888")
        self._user_color_btn.set_rgba(grey)
        self._user_color_btn.handler_unblock_by_func(self._on_user_text_color_changed)
        self._settings.user_bubble_text_color = ""
        self._settings.save()
        self._app.refresh_theme()

    def _on_asst_text_color_changed(self, btn: Gtk.ColorDialogButton, _pspec) -> None:
        rgba = btn.get_rgba()
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(rgba.red * 255), int(rgba.green * 255), int(rgba.blue * 255),
        )
        self._settings.asst_bubble_text_color = hex_color
        self._settings.save()
        self._app.refresh_theme()

    def _on_asst_text_color_reset(self, _btn) -> None:
        self._asst_color_btn.handler_block_by_func(self._on_asst_text_color_changed)
        grey = Gdk.RGBA()
        grey.parse("#888888")
        self._asst_color_btn.set_rgba(grey)
        self._asst_color_btn.handler_unblock_by_func(self._on_asst_text_color_changed)
        self._settings.asst_bubble_text_color = ""
        self._settings.save()
        self._app.refresh_theme()

    def _on_backend_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._backend_ids):
            self._settings.backend = self._backend_ids[idx]
            self._settings.save()
            self._trigger_model_reload()

    def _on_spec_decoding_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.enable_speculative_decoding = row.get_active()
        self._settings.save()
        self._trigger_model_reload()

    def _on_temp_switch(self, row: Adw.SwitchRow, _pspec) -> None:
        enabled = row.get_active()
        self._temp_spin.set_sensitive(enabled)
        self._settings.temperature = self._temp_spin.get_value() if enabled else None
        self._settings.save()
        self._trigger_model_reload()

    def _on_temperature_changed(self, row: Adw.SpinRow, _pspec) -> None:
        if self._temp_switch.get_active():
            self._settings.temperature = row.get_value()
            self._settings.save()
            self._trigger_model_reload()

    def _on_topk_switch(self, row: Adw.SwitchRow, _pspec) -> None:
        enabled = row.get_active()
        self._topk_spin.set_sensitive(enabled)
        self._settings.top_k = int(self._topk_spin.get_value()) if enabled else None
        self._settings.save()
        self._trigger_model_reload()

    def _on_topk_changed(self, row: Adw.SpinRow, _pspec) -> None:
        if self._topk_switch.get_active():
            self._settings.top_k = int(row.get_value())
            self._settings.save()
            self._trigger_model_reload()

    def _on_topp_switch(self, row: Adw.SwitchRow, _pspec) -> None:
        enabled = row.get_active()
        self._topp_spin.set_sensitive(enabled)
        self._settings.top_p = round(self._topp_spin.get_value(), 2) if enabled else None
        self._settings.save()
        self._trigger_model_reload()

    def _on_topp_changed(self, row: Adw.SpinRow, _pspec) -> None:
        if self._topp_switch.get_active():
            self._settings.top_p = round(row.get_value(), 2)
            self._settings.save()
            self._trigger_model_reload()

    def _on_max_context_changed(self, row: Adw.SpinRow, _pspec) -> None:
        new = int(row.get_value())
        if new == self._settings.max_context_tokens:
            return
        self._settings.max_context_tokens = new
        self._settings.save()
        # Update the composer's context-usage bar total right away — the
        # engine reload below is debounced and EvtReady doesn't refresh the
        # bar, so without this the bar shows the old "/ N" until the next send.
        win = self._main_win()
        if win is not None and hasattr(win, "refresh_context_bar"):
            win.refresh_context_bar()
        self._trigger_model_reload()

    def _on_context_bar_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.show_context_bar = row.get_active()
        self._settings.save()
        win = self._main_win()
        if win is not None and hasattr(win, "refresh_context_bar"):
            win.refresh_context_bar()

    def _on_vision_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.enable_vision = row.get_active()
        self._settings.save()
        self._trigger_model_reload()

    def _on_audio_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.enable_audio = row.get_active()
        self._settings.save()
        self._trigger_model_reload()

    def _on_voice_auto_send_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.voice_auto_send = row.get_active()
        self._settings.save()

    def _on_push_to_talk_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        # Takes effect on the next live-mode session (the controller reads
        # the flag in start()); no engine reload needed.
        self._settings.live_push_to_talk = row.get_active()
        self._settings.save()

    # ── Camera (Phase 4.5) ────────────────────────────────────────────────
    def _refresh_cam_devices(self) -> None:
        try:
            from . import webcam
            devices = webcam.list_devices()
        except Exception:
            devices = []
        self._cam_device_ids = [""] + [d.id for d in devices]
        labels = ["System default"] + [d.label for d in devices]
        self._cam_device_row.set_model(Gtk.StringList.new(labels))
        # Reflect saved setting if it's still around.
        saved = self._settings.webcam_device or ""
        if saved in self._cam_device_ids:
            self._cam_device_row.set_selected(self._cam_device_ids.index(saved))
        else:
            self._cam_device_row.set_selected(0)

    def _sync_cam_sensitivity(self) -> None:
        on = self._settings.webcam_enabled
        for row in self._cam_sub_rows:
            row.set_sensitive(on)

    def _on_cam_enabled_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        target = row.get_active()
        if target and not self._settings.webcam_first_enable_acknowledged:
            self._confirm_camera_first_enable(row)
            return
        self._apply_cam_enable(target)

    def _confirm_camera_first_enable(self, row: Adw.SwitchRow) -> None:
        dlg = Adw.AlertDialog(
            heading="Enable camera?",
            body=(
                "Box will add a 📷 button to the composer so you can snap "
                "a webcam frame and attach it to the next message. The "
                "camera only opens while the capture modal is open — no "
                "continuous recording. Make sure Vision (above) is also "
                "on so the model can see what you capture."
            ),
        )
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("enable", "Enable")
        dlg.set_response_appearance("enable", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("enable")
        dlg.set_close_response("cancel")

        def _on_response(_dlg, response: str) -> None:
            if response == "enable":
                self._settings.webcam_first_enable_acknowledged = True
                self._apply_cam_enable(True)
            else:
                row.handler_block_by_func(self._on_cam_enabled_changed)
                row.set_active(False)
                row.handler_unblock_by_func(self._on_cam_enabled_changed)

        dlg.connect("response", _on_response)
        dlg.present(self._main_win())

    def _apply_cam_enable(self, target: bool) -> None:
        self._settings.webcam_enabled = target
        self._settings.save()
        self._sync_cam_sensitivity()
        win = self._main_win()
        if win is not None and hasattr(win, "refresh_camera_button"):
            win.refresh_camera_button()

    def _on_cam_device_changed(self, row: Adw.ComboRow, _pspec) -> None:
        idx = row.get_selected()
        if 0 <= idx < len(self._cam_device_ids):
            self._settings.webcam_device = self._cam_device_ids[idx]
            self._settings.save()

    def _on_cam_width_changed(self, row: Adw.SpinRow, _pspec) -> None:
        self._settings.webcam_capture_width = int(row.get_value())
        self._settings.save()

    def _on_cam_quality_changed(self, row: Adw.SpinRow, _pspec) -> None:
        self._settings.webcam_capture_jpeg_quality = int(row.get_value())
        self._settings.save()

    def _on_tts_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.enable_tts = row.get_active()
        self._settings.save()
        self._tts_auto_row.set_sensitive(self._settings.enable_tts)
        win = self._main_win()
        if win is not None and hasattr(win, "on_tts_changed"):
            win.on_tts_changed(self._settings.enable_tts)

    def _on_tts_auto_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.tts_auto_speak = row.get_active()
        self._settings.save()

    def _on_voice_changed(self, row: Adw.ComboRow, _pspec) -> None:
        from .tts import is_ready as tts_ready
        voice_id = self._voice_ids[row.get_selected()]
        self._settings.tts_voice = voice_id
        self._settings.save()
        ready = tts_ready(voice_id)
        self._tts_dl_row.set_subtitle("Ready ✓" if ready else "Not downloaded")
        self._tts_dl_btn.set_label("Re-download" if ready else "Download")
        if ready:
            self._tts_dl_btn.remove_css_class("suggested-action")
            self._tts_dl_btn.add_css_class("flat")
        else:
            self._tts_dl_btn.add_css_class("suggested-action")
            self._tts_dl_btn.remove_css_class("flat")

    # ── RAG handlers ──────────────────────────────────────────────────────
    def _on_rag_enabled_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.rag_enabled = row.get_active()
        self._settings.save()

    def _on_rag_variant_changed(self, row: Adw.ComboRow, _pspec) -> None:
        from . import rag_models
        idx = row.get_selected()
        if not (0 <= idx < len(self._rag_variant_ids)):
            return
        self._settings.rag_embed_variant = self._rag_variant_ids[idx]
        self._settings.save()
        # Refresh download row state for the new variant.
        ready = rag_models.is_variant_ready(self._settings.rag_embed_variant)
        self._rag_dl_row.set_subtitle("Ready ✓" if ready else "Not downloaded")
        self._rag_dl_btn.set_label("Re-download" if ready else "Download")
        if ready:
            self._rag_dl_btn.remove_css_class("suggested-action")
            self._rag_dl_btn.add_css_class("flat")
        else:
            self._rag_dl_btn.add_css_class("suggested-action")
            self._rag_dl_btn.remove_css_class("flat")

    def _on_rag_topk_changed(self, row: Adw.SpinRow, _pspec) -> None:
        self._settings.rag_top_k = int(row.get_value())
        self._settings.save()

    def _on_rag_auto_attach_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        self._settings.rag_auto_attach_notebooks = row.get_active()
        self._settings.save()

    def _on_rag_chunk_size_changed(self, row: Adw.SpinRow, _pspec) -> None:
        self._settings.rag_chunk_size = int(row.get_value())
        self._settings.save()

    def _on_rag_full_index_changed(self, row: Adw.SwitchRow, _pspec) -> None:
        if row.get_active():
            # "Index the whole document" ON → no cap.
            self._settings.rag_max_chunks = 0
            self._rag_sample_row.set_sensitive(False)
        else:
            # OFF → use the sample-row's value as the cap.
            self._settings.rag_max_chunks = int(self._rag_sample_row.get_value())
            self._rag_sample_row.set_sensitive(True)
        self._settings.save()

    def _on_rag_sample_size_changed(self, row: Adw.SpinRow, _pspec) -> None:
        # Only persist while sampling is the active mode.
        if not self._rag_full_index_row.get_active():
            self._settings.rag_max_chunks = int(row.get_value())
            self._settings.save()

    def _on_rag_download(self, _btn) -> None:
        from gi.repository import GLib as _GLib
        from . import rag_models

        variant = self._settings.rag_embed_variant
        self._rag_dl_btn.set_sensitive(False)
        self._rag_dl_row.set_subtitle("Starting download…")

        def _progress(label: str, done: int, total: int) -> None:
            def _ui():
                if total:
                    mb_done = done / (1024 * 1024)
                    mb_total = total / (1024 * 1024)
                    self._rag_dl_row.set_subtitle(
                        f"{label}  {mb_done:.0f}/{mb_total:.0f} MB"
                    )
                else:
                    self._rag_dl_row.set_subtitle(label)
                return False
            _GLib.idle_add(_ui)

        def _done() -> None:
            def _ui():
                self._rag_dl_row.set_subtitle("Ready ✓")
                self._rag_dl_btn.set_label("Re-download")
                self._rag_dl_btn.remove_css_class("suggested-action")
                self._rag_dl_btn.add_css_class("flat")
                self._rag_dl_btn.set_sensitive(True)
                return False
            _GLib.idle_add(_ui)

        def _error(msg: str) -> None:
            def _ui():
                self._rag_dl_row.set_subtitle(f"Error: {msg[:80]}")
                self._rag_dl_btn.set_sensitive(True)
                return False
            _GLib.idle_add(_ui)

        rag_models.download_variant(variant, _progress, _done, _error)

    def _on_tts_download(self, _btn) -> None:
        from gi.repository import GLib as _GLib
        from .tts import download as tts_download

        voice_id = self._settings.tts_voice or "en_US-lessac-medium"
        self._tts_dl_btn.set_sensitive(False)
        self._tts_dl_row.set_subtitle("Starting download…")

        def _progress(label: str, done: int, total: int) -> None:
            def _ui():
                if total:
                    pct = done / total * 100
                    self._tts_dl_row.set_subtitle(f"{label}  {pct:.0f}%")
                else:
                    self._tts_dl_row.set_subtitle(label)
                return False
            _GLib.idle_add(_ui)

        def _done() -> None:
            def _ui():
                self._tts_dl_row.set_subtitle("Ready ✓")
                self._tts_dl_btn.set_label("Re-download")
                self._tts_dl_btn.remove_css_class("suggested-action")
                self._tts_dl_btn.add_css_class("flat")
                self._tts_dl_btn.set_sensitive(True)
                return False
            _GLib.idle_add(_ui)

        def _error(msg: str) -> None:
            def _ui():
                self._tts_dl_row.set_subtitle(f"Error: {msg[:80]}")
                self._tts_dl_btn.set_sensitive(True)
                return False
            _GLib.idle_add(_ui)

        tts_download(voice_id, _progress, _done, _error)

    def _main_win(self):
        if self._window is not None:
            return self._window
        win = self._app.get_active_window()
        if win is not None and hasattr(win, "on_model_changed"):
            return win
        for w in self._app.get_windows():
            if hasattr(w, "on_model_changed"):
                return w
        return None

    # Wait this long after the last inference-setting change before reloading
    # the engine. Dragging a SpinRow (context window, temperature, top-k/p)
    # emits one value change per tick; without debouncing, each tick queued a
    # full engine reload — ~28 of them dragging the context slider 32k→4k, each
    # re-prefilling history. We collapse a burst of changes into a single
    # reload once the value settles. A 700 ms gap is well past a drag but
    # imperceptible for a single click.
    _RELOAD_DEBOUNCE_MS = 700

    def _trigger_model_reload(self) -> None:
        if self._reload_source_id:
            GLib.source_remove(self._reload_source_id)
            self._reload_source_id = 0
        self._reload_source_id = GLib.timeout_add(
            self._RELOAD_DEBOUNCE_MS, self._do_model_reload
        )

    def _do_model_reload(self) -> bool:
        self._reload_source_id = 0
        win = self._main_win()
        if win is not None:
            win.on_model_changed(self._settings.model_path)
        return False  # GLib.SOURCE_REMOVE — one-shot
