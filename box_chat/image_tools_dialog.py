"""Image Tools dialog — Generate (SD), Erase (paint-mask), Upscale (4x).

The UI tier for :mod:`box_chat.sd_backend` and :mod:`box_chat.vision_tools`.
Three pages in a view switcher. Inference runs on a worker thread; results
come back via ``GLib.idle_add``. All model math lives in the backends
(gi-free); this file is pure UI.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from .vision_tools import EraseEngine, UpscaleEngine, VisionModelError

log = logging.getLogger(__name__)


def _pil_to_texture(img) -> Gdk.Texture:
    rgba = img.convert("RGBA")
    w, h = rgba.size
    data = GLib.Bytes.new(rgba.tobytes())
    return Gdk.MemoryTexture.new(w, h, Gdk.MemoryFormat.R8G8B8A8, data, w * 4)


class ImageToolsDialog(Adw.Window):
    """Image Tools as a real window (was Adw.Dialog — fixed size, no
    maximize; the canvas needs room). Same conversion Preferences got."""

    def __init__(self, parent_window, settings=None) -> None:
        super().__init__(
            title="Image Tools",
            default_width=1100,
            default_height=760,
        )
        if parent_window is not None:
            self.set_transient_for(parent_window)
        self.add_css_class("aux-solid")  # opaque under glass modes
        self._parent = parent_window
        self._settings = settings
        # Ctrl+W closes, like any window.
        kc = Gtk.ShortcutController()
        kc.add_shortcut(Gtk.Shortcut.new(
            Gtk.ShortcutTrigger.parse_string("<Primary>w"),
            Gtk.CallbackAction.new(lambda *_: bool(self.close()) or True),
        ))
        self.add_controller(kc)
        # Closing mid-generation must not orphan a grinding sd-cli child or
        # leave the preview timer poking destroyed widgets.
        self.connect("close-request", self._on_close_request)

        self._erase_engine = EraseEngine()
        self._upscale_engine = UpscaleEngine()

        self._sd_backend = None
        self._litert_pipe = None
        self._sd_result = None
        self._sd_result_path = None   # original PNG (keeps sd.cpp metadata)
        self._sd_busy = False
        self._sd_preview_timer = 0    # GLib source polling the preview PNG
        self._sd_preview_file = ""
        self._sd_preview_mtime = 0.0

        self._erase_orig = None
        self._erase_surface = None
        self._erase_surface_data = None
        self._mask = None
        self._brush = 28
        self._display_scale = 1.0
        self._busy = False

        self._upscale_orig = None
        self._upscale_result = None

        toolbar = Adw.ToolbarView()
        header = Adw.HeaderBar()
        self._switcher = Adw.ViewSwitcher(policy=Adw.ViewSwitcherPolicy.WIDE)
        header.set_title_widget(self._switcher)
        toolbar.add_top_bar(header)

        self._stack = Adw.ViewStack()
        self._switcher.set_stack(self._stack)
        self._stack.add_titled_with_icon(
            self._build_generate_page(), "generate", "Generate",
            "applications-graphics-symbolic")
        self._stack.add_titled_with_icon(
            self._build_erase_page(), "erase", "Erase", "edit-clear-symbolic")
        self._stack.add_titled_with_icon(
            self._build_upscale_page(), "upscale", "Upscale", "zoom-in-symbolic")
        toolbar.set_content(self._stack)
        self.set_content(toolbar)

    # ── Generate page (stable-diffusion.cpp) ────────────────────────────
    def _build_generate_page(self) -> Gtk.Widget:
        from .sd_backend import SAMPLERS, SCHEDULERS, find_sd_binary, SDError

        s = self._settings
        scroll = Gtk.ScrolledWindow(hscrollbar_policy=Gtk.PolicyType.NEVER, vexpand=True)
        page = Adw.PreferencesPage()
        scroll.set_child(page)

        try:
            find_sd_binary(self._sd_variant())
            self._sd_available = True
        except SDError:
            self._sd_available = False

        model_group = Adw.PreferencesGroup(title="Model")
        page.add(model_group)

        # Engine selector: stable-diffusion.cpp (GGUF/safetensors) vs the
        # LiteRT .tflite pipelines (Z-Image Turbo / FLUX.2-klein).
        self._engine_ids = ["sdcpp", "litert"]
        engine_names = ["stable-diffusion.cpp (GGUF)", "LiteRT (Z-Image / FLUX klein)"]
        self._engine_row = Adw.ComboRow(
            title="Engine",
            subtitle="LiteRT runs chunked .tflite graphs — 256×256, several minutes/image",
            model=Gtk.StringList.new(engine_names))
        cur_engine = getattr(s, "sd_engine", "sdcpp") if s else "sdcpp"
        self._engine_row.set_selected(
            self._engine_ids.index(cur_engine) if cur_engine in self._engine_ids else 0)
        self._engine_row.connect("notify::selected", self._on_engine_changed)
        model_group.add(self._engine_row)

        # sd.cpp model layout: single checkpoint vs component bundle
        # (Z-Image / FLUX.2-klein GGUFs — any resolution, unlike LiteRT).
        self._sd_mode_ids = ["checkpoint", "components"]
        self._sd_mode_row = Adw.ComboRow(
            title="Model layout",
            model=Gtk.StringList.new(
                ["Single checkpoint", "Components (Z-Image / FLUX.2)"]
            ))
        cur_mode = getattr(s, "sd_gen_mode", "checkpoint") if s else "checkpoint"
        self._sd_mode_row.set_selected(
            self._sd_mode_ids.index(cur_mode)
            if cur_mode in self._sd_mode_ids else 0)
        self._sd_mode_row.connect("notify::selected", self._on_sd_mode_changed)
        model_group.add(self._sd_mode_row)

        # stable-diffusion.cpp model rows.
        self._sd_model_row = Adw.ComboRow(title="Diffusion model")
        self._sd_refresh_model_list()
        model_group.add(self._sd_model_row)
        self._sd_import_row = Adw.ActionRow(
            title="Import a model…",
            subtitle="*.gguf or *.safetensors (SD1.x / SDXL / SD3.5 / FLUX)",
            activatable=True)
        self._sd_import_row.add_suffix(Gtk.Image(icon_name="go-next-symbolic",
                                                 valign=Gtk.Align.CENTER))
        self._sd_import_row.connect("activated", lambda _r: self._sd_import_model())
        model_group.add(self._sd_import_row)

        # Component-bundle rows (sd.cpp components layout).
        self._sd_bundle_row = Adw.ComboRow(title="Component bundle")
        self._sd_bundle_row.connect(
            "notify::selected", self._on_sd_bundle_changed)
        model_group.add(self._sd_bundle_row)
        self._sd_bundle_dl_row = Adw.ActionRow(
            title="Download a bundle…",
            subtitle="Z-Image Turbo (~6.1GB) or FLUX.2 klein (~4.9GB), "
                     "resumable — runs at any resolution",
            activatable=True)
        self._sd_bundle_dl_row.add_suffix(Gtk.Image(
            icon_name="folder-download-symbolic", valign=Gtk.Align.CENTER))
        self._sd_bundle_dl_row.connect(
            "activated", lambda _r: self._sd_download_bundle())
        model_group.add(self._sd_bundle_dl_row)
        self._sd_comp_import_row = Adw.ActionRow(
            title="Import components…",
            subtitle="Pick a diffusion GGUF, a VAE, then a text-encoder "
                     "GGUF you downloaded yourself",
            activatable=True)
        self._sd_comp_import_row.add_suffix(Gtk.Image(
            icon_name="go-next-symbolic", valign=Gtk.Align.CENTER))
        self._sd_comp_import_row.connect(
            "activated", lambda _r: self._sd_import_components())
        model_group.add(self._sd_comp_import_row)
        self._sd_refresh_bundle_list()

        # LiteRT model-directory rows (each dir holds the chunked graphs +
        # tokenizer for one Z-Image / klein model).
        self._litert_row = Adw.ComboRow(title="LiteRT model folder")
        self._litert_refresh_dir_list()
        model_group.add(self._litert_row)
        self._litert_download_row = Adw.ActionRow(
            title="Download a model…",
            subtitle="Z-Image Turbo (~10.6GB) or FLUX.2 klein (~7.4GB), resumable",
            activatable=True)
        self._litert_download_row.add_suffix(Gtk.Image(icon_name="folder-download-symbolic",
                                                       valign=Gtk.Align.CENTER))
        self._litert_download_row.connect("activated", lambda _r: self._litert_download())
        model_group.add(self._litert_download_row)
        self._litert_import_row = Adw.ActionRow(
            title="Import a LiteRT model folder…",
            subtitle="Folder of *.tflite graphs + qwen tokenizer (Z-Image / klein)",
            activatable=True)
        self._litert_import_row.add_suffix(Gtk.Image(icon_name="go-next-symbolic",
                                                     valign=Gtk.Align.CENTER))
        self._litert_import_row.connect("activated", lambda _r: self._litert_import_dir())
        model_group.add(self._litert_import_row)

        prompt_group = Adw.PreferencesGroup(title="Prompt")
        page.add(prompt_group)
        self._sd_prompt = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD, top_margin=8, bottom_margin=8,
            left_margin=8, right_margin=8, height_request=70)
        pw = Gtk.Frame()
        pw.set_child(self._sd_prompt)
        prow = Adw.PreferencesRow(activatable=False)
        prow.set_child(pw)
        prompt_group.add(prow)
        self._sd_negative = Adw.EntryRow(title="Negative prompt (optional)")
        prompt_group.add(self._sd_negative)

        pg = Adw.PreferencesGroup(title="Parameters")
        page.add(pg)
        self._sd_width = self._sd_spin(pg, "Width", s.sd_width if s else 512, 64, 2048, 64)
        self._sd_height = self._sd_spin(pg, "Height", s.sd_height if s else 512, 64, 2048, 64)
        self._sd_steps = self._sd_spin(pg, "Steps", s.sd_steps if s else 20, 1, 150, 1)
        self._sd_cfg = self._sd_spin(pg, "CFG scale", s.sd_cfg_scale if s else 7.0,
                                     1.0, 20.0, 0.5, digits=1)
        self._sd_seed = self._sd_spin(pg, "Seed (-1 = random)", -1, -1, 2_000_000_000, 1)
        self._sd_batch = self._sd_spin(pg, "Batch count", s.sd_batch_count if s else 1, 1, 8, 1)
        self._sd_sampler = self._sd_combo(pg, "Sampler", SAMPLERS,
                                          s.sd_sampler if s else "euler_a")
        self._sd_scheduler = self._sd_combo(pg, "Scheduler", SCHEDULERS,
                                            s.sd_scheduler if s else "discrete")

        extras = Adw.PreferencesGroup(
            title="Extras",
            description="Speed, previews, img2img and LoRA.")
        page.add(extras)
        from .sd_backend import CACHE_MODES
        self._sd_cache = self._sd_combo(
            extras, "Step caching (speed boost)", CACHE_MODES,
            getattr(s, "sd_cache_mode", "none") if s else "none")
        self._sd_preview_sw = self._sd_switch(
            extras, "Live preview while generating",
            getattr(s, "sd_live_preview", True) if s else True)
        self._sd_init_row = Adw.ActionRow(
            title="img2img source (optional)", subtitle="No image set",
            activatable=True)
        self._sd_init_row.add_suffix(Gtk.Image(icon_name="document-open-symbolic",
                                               valign=Gtk.Align.CENTER))
        clear_btn = Gtk.Button(icon_name="edit-clear-symbolic",
                               valign=Gtk.Align.CENTER)
        clear_btn.add_css_class("flat")
        clear_btn.set_tooltip_text("Clear the img2img source")
        clear_btn.connect("clicked", lambda *_: self._set_init_image(""))
        self._sd_init_row.add_suffix(clear_btn)
        self._sd_init_row.connect("activated", lambda _r: self._pick_init_image())
        self._sd_init_image = ""
        extras.add(self._sd_init_row)
        self._sd_strength = self._sd_spin(
            extras, "img2img strength", 0.75, 0.0, 1.0, 0.05, digits=2)
        self._sd_mask_image = ""
        self._sd_mask_row = Adw.ActionRow(
            title="Inpaint mask (optional)",
            subtitle="Paint the areas to regenerate — needs an img2img source",
            activatable=True)
        self._sd_mask_row.add_suffix(Gtk.Image(
            icon_name="applications-graphics-symbolic",
            valign=Gtk.Align.CENTER))
        mask_clear = Gtk.Button(icon_name="edit-clear-symbolic",
                                valign=Gtk.Align.CENTER)
        mask_clear.add_css_class("flat")
        mask_clear.set_tooltip_text("Clear the inpaint mask")
        mask_clear.connect("clicked", lambda *_: self._set_mask_image(""))
        self._sd_mask_row.add_suffix(mask_clear)
        self._sd_mask_row.connect("activated", lambda _r: self._edit_mask())
        extras.add(self._sd_mask_row)
        hires_vals = [0.0, 1.5, 2.0, 3.0, 4.0]
        self._sd_hires = self._sd_combo(
            extras, "Hires fix (two-pass scale)",
            ["off", "1.5×", "2×", "3×", "4×"], "off")
        self._sd_hires._scales = hires_vals  # parallel to the labels
        cur_hs = getattr(s, "sd_hires_scale", 0.0) if s else 0.0
        if cur_hs in hires_vals:
            self._sd_hires.set_selected(hires_vals.index(cur_hs))
        self._sd_lora_row = Adw.ActionRow(
            title="LoRA folder (optional)",
            subtitle=(getattr(s, "sd_lora_dir", "") if s else "")
            or "Prompt syntax: lora:name:0.8 in angle brackets",
            activatable=True)
        self._sd_lora_row.add_suffix(Gtk.Image(icon_name="folder-symbolic",
                                               valign=Gtk.Align.CENTER))
        self._sd_lora_row.connect("activated", lambda _r: self._pick_lora_dir())
        extras.add(self._sd_lora_row)

        mem = Adw.PreferencesGroup(
            title="Memory and performance",
            description="Turn these on if generation runs out of RAM.")
        page.add(mem)
        self._sd_vae_tiling = self._sd_switch(mem, "VAE tiling", s.sd_vae_tiling if s else False)
        self._sd_vae_cpu = self._sd_switch(mem, "VAE on CPU", s.sd_vae_on_cpu if s else False)
        self._sd_clip_cpu = self._sd_switch(mem, "Text encoder on CPU", s.sd_clip_on_cpu if s else False)
        self._sd_fa = self._sd_switch(mem, "Diffusion flash attention", s.sd_diffusion_fa if s else False)
        self._sd_offload = self._sd_switch(
            mem, "Offload weights to RAM (GPU builds)",
            getattr(s, "sd_offload_cpu", False) if s else False)

        # ── Two-pane layout: settings left, canvas right — the divider is
        # user-draggable (Jegly: the image section needs to be resizable).
        outer = Gtk.Paned(
            orientation=Gtk.Orientation.HORIZONTAL, wide_handle=True,
            resize_start_child=False, shrink_start_child=False,
            shrink_end_child=False,
            margin_top=6, margin_bottom=10, margin_start=10, margin_end=10,
        )
        scroll.set_size_request(380, -1)
        outer.set_start_child(scroll)
        outer.set_position(430)

        canvas = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                         hexpand=True, vexpand=True,
                         margin_start=10)
        self._sd_pic = Gtk.Picture(vexpand=True, hexpand=True)
        self._sd_pic.add_css_class("card")
        self._sd_placeholder = Adw.StatusPage(
            icon_name="applications-graphics-symbolic",
            title="No image yet",
            description="Set a prompt and press Generate.",
            vexpand=True, hexpand=True,
        )
        self._sd_canvas_stack = Gtk.Stack(vexpand=True, hexpand=True)
        self._sd_canvas_stack.add_named(self._sd_placeholder, "empty")
        self._sd_canvas_stack.add_named(self._sd_pic, "image")
        canvas.append(self._sd_canvas_stack)

        self._sd_progress = Gtk.ProgressBar(visible=False, show_text=True)
        canvas.append(self._sd_progress)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.CENTER)
        self._sd_gen_btn = Gtk.Button(label="Generate")
        self._sd_gen_btn.add_css_class("suggested-action")
        self._sd_gen_btn.connect("clicked", self._on_generate)
        btns.append(self._sd_gen_btn)
        self._sd_cancel_btn = Gtk.Button(label="Cancel", sensitive=False)
        self._sd_cancel_btn.connect("clicked", self._on_cancel_generate)
        btns.append(self._sd_cancel_btn)
        self._sd_save_btn = Gtk.Button(label="Save…", sensitive=False)
        self._sd_save_btn.connect(
            "clicked",
            lambda _b: self._save(self._sd_result, self._sd_result_path),
        )
        btns.append(self._sd_save_btn)
        canvas.append(btns)

        self._sd_seed_label = Gtk.Label(label="", visible=False)
        self._sd_seed_label.add_css_class("dim-label")
        seed_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
                           halign=Gtk.Align.CENTER)
        seed_box.append(self._sd_seed_label)
        self._sd_seed_reuse = Gtk.Button(label="Reuse seed", visible=False)
        self._sd_seed_reuse.add_css_class("flat")
        self._sd_seed_reuse.connect("clicked", self._on_reuse_seed)
        seed_box.append(self._sd_seed_reuse)
        canvas.append(seed_box)

        self._gen_note = Gtk.Label(wrap=True, visible=False)
        self._gen_note.add_css_class("dim-label")
        canvas.append(self._gen_note)
        outer.set_end_child(canvas)

        self._apply_engine_visibility()
        return outer

    # ── engine selection ────────────────────────────────────────────────
    def _current_engine(self) -> str:
        return self._engine_ids[self._engine_row.get_selected()]

    def _litert_available(self) -> bool:
        try:
            import ai_edge_litert.interpreter  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    # ── component-bundle helpers (sd.cpp Z-Image / FLUX.2-klein) ────────
    def _current_sd_mode(self) -> str:
        idx = self._sd_mode_row.get_selected()
        return self._sd_mode_ids[idx if 0 <= idx < 2 else 0]

    def _on_sd_mode_changed(self, *_a) -> None:
        if self._settings is not None:
            self._settings.sd_gen_mode = self._current_sd_mode()
            self._settings.save()
        self._apply_engine_visibility()

    def _sd_refresh_bundle_list(self) -> None:
        from .sd_components import bundle_dir, installed_bundles
        s = self._settings
        # Items: ("bundle", SDBundle) for downloaded sets, plus a "custom"
        # entry when the user hand-imported component files.
        self._sd_bundle_items: list[tuple] = [
            ("bundle", b) for b in installed_bundles()
        ]
        if s is not None and s.sd_custom_diffusion:
            self._sd_bundle_items.append(("custom", (
                s.sd_custom_diffusion, s.sd_custom_vae, s.sd_custom_llm)))
        names: list[str] = []
        for kind, data in self._sd_bundle_items:
            if kind == "bundle":
                names.append(data.name)
            else:
                names.append(f"Custom: {Path(data[0]).name}")
        if not names:
            names = ["(none downloaded)"]
        self._sd_bundle_row.set_model(Gtk.StringList.new(names))
        want = getattr(s, "sd_component_dir", "") if s else ""
        for i, (kind, data) in enumerate(self._sd_bundle_items):
            if kind == "bundle" and str(bundle_dir(data.key)) == want:
                self._sd_bundle_row.set_selected(i)
                break

    def _selected_bundle_item(self) -> tuple | None:
        items = getattr(self, "_sd_bundle_items", None)
        if not items:
            return None
        idx = self._sd_bundle_row.get_selected()
        if 0 <= idx < len(items):
            return items[idx]
        return None

    def _selected_bundle(self):
        item = self._selected_bundle_item()
        return item[1] if item and item[0] == "bundle" else None

    def _sd_import_components(self) -> None:
        """Three chained pickers: diffusion GGUF → VAE → text encoder."""
        picks: list[str] = []
        titles = [
            "1/3 — Choose the diffusion model (*.gguf / *.safetensors)",
            "2/3 — Choose the VAE (*.safetensors)",
            "3/3 — Choose the text encoder (*.gguf / *.safetensors)",
        ]

        def ask(i: int) -> None:
            dlg = Gtk.FileDialog(title=titles[i])

            def done(d, res) -> None:
                try:
                    f = d.open_finish(res)
                except GLib.Error:
                    return
                if not (f and f.get_path()):
                    return
                picks.append(f.get_path())
                if len(picks) < 3:
                    ask(len(picks))
                    return
                s = self._settings
                if s is not None:
                    s.sd_custom_diffusion = picks[0]
                    s.sd_custom_vae = picks[1]
                    s.sd_custom_llm = picks[2]
                    s.save()
                self._sd_refresh_bundle_list()
                # Select the custom entry we just created.
                self._sd_bundle_row.set_selected(
                    len(self._sd_bundle_items) - 1)
                self._toast("Component set imported.")

            dlg.open(self, None, done)

        ask(0)

    def _on_sd_bundle_changed(self, *_a) -> None:
        b = self._selected_bundle()
        if b is None:
            return
        from .sd_components import bundle_dir
        if self._settings is not None:
            self._settings.sd_component_dir = str(bundle_dir(b.key))
            self._settings.save()
        # Distilled models want their own defaults — prefill, user can edit.
        self._sd_steps.set_value(b.default_steps)
        self._sd_cfg.set_value(b.default_cfg)

    def _sd_download_bundle(self) -> None:
        from .sd_components import SD_BUNDLES, download_bundle, is_complete
        pending = [b for b in SD_BUNDLES if not is_complete(b)]
        if not pending:
            self._toast("All bundles are already downloaded.")
            self._sd_refresh_bundle_list()
            return
        dlg = Adw.AlertDialog(
            heading="Download a component bundle",
            body="Downloads run in the background (resumable, verified by "
                 "size). Progress shows in the generation progress bar.",
        )
        dd = Gtk.DropDown.new_from_strings(
            [f"{b.name} — {b.total_bytes/1e9:.1f}GB" for b in pending])
        dlg.set_extra_child(dd)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("go", "Download")
        dlg.set_response_appearance("go", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_close_response("cancel")

        def on_response(_d, rid: str) -> None:
            if rid != "go":
                return
            bundle = pending[dd.get_selected()]
            self._sd_progress.set_visible(True)
            self._sd_progress.set_fraction(0.0)
            self._sd_progress.set_text(f"Downloading {bundle.name}…")

            def overall(done: int, total: int) -> None:
                GLib.idle_add(self._sd_progress.set_fraction,
                              done / max(1, total))
                GLib.idle_add(
                    self._sd_progress.set_text,
                    f"{bundle.name}: {done/1e9:.2f} / {total/1e9:.2f} GB")

            def work() -> None:
                try:
                    download_bundle(bundle, on_overall=overall)
                    GLib.idle_add(self._toast, f"{bundle.name} ready.")
                except Exception as e:  # noqa: BLE001
                    log.exception("bundle download failed")
                    GLib.idle_add(self._toast, f"Download failed: {e}")
                GLib.idle_add(self._sd_progress.set_visible, False)
                GLib.idle_add(self._sd_refresh_bundle_list)

            threading.Thread(target=work, daemon=True).start()

        dlg.connect("response", on_response)
        dlg.present(self._parent)

    def _pick_init_image(self) -> None:
        dlg = Gtk.FileDialog(title="Choose the img2img source image")

        def done(d, res) -> None:
            try:
                f = d.open_finish(res)
            except GLib.Error:
                return
            if f and f.get_path():
                self._set_init_image(f.get_path())

        dlg.open(self._parent, None, done)

    def _set_init_image(self, path: str) -> None:
        self._sd_init_image = path
        self._sd_init_row.set_subtitle(
            Path(path).name if path else "No image set")

    def _set_mask_image(self, path: str) -> None:
        self._sd_mask_image = path
        self._sd_mask_row.set_subtitle(
            "Mask set — white areas regenerate" if path
            else "Paint the areas to regenerate — needs an img2img source")

    def _edit_mask(self) -> None:
        if not self._sd_init_image:
            self._toast("Pick an img2img source image first.")
            return
        _InpaintMaskWindow(self, self._sd_init_image, self._set_mask_image)

    def _pick_lora_dir(self) -> None:
        dlg = Gtk.FileDialog(title="Choose the LoRA folder")

        def done(d, res) -> None:
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            if f and f.get_path():
                self._settings.sd_lora_dir = f.get_path()
                self._settings.save()
                self._sd_lora_row.set_subtitle(f.get_path())

        dlg.select_folder(self._parent, None, done)

    def _on_reuse_seed(self, *_a) -> None:
        if self._sd_backend is not None and self._sd_backend.last_seed is not None:
            self._sd_seed.set_value(self._sd_backend.last_seed)

    def _on_engine_changed(self, _row, _param) -> None:
        if self._settings is not None:
            self._settings.sd_engine = self._current_engine()
            self._settings.save()
        self._apply_engine_visibility()

    def _apply_engine_visibility(self) -> None:
        """Show only the rows for the active engine and set button state."""
        litert = self._current_engine() == "litert"
        components = (not litert) and self._current_sd_mode() == "components"
        self._sd_mode_row.set_visible(not litert)
        self._sd_model_row.set_visible(not litert and not components)
        self._sd_import_row.set_visible(not litert and not components)
        self._sd_bundle_row.set_visible(components)
        self._sd_bundle_dl_row.set_visible(components)
        self._sd_comp_import_row.set_visible(components)
        self._litert_row.set_visible(litert)
        self._litert_download_row.set_visible(litert)
        self._litert_import_row.set_visible(litert)

        if litert:
            available = self._litert_available()
            note = ("" if available else
                    "LiteRT runtime (ai-edge-litert) not available in this build.")
        else:
            available = self._sd_available
            note = ("" if available else
                    "Image generation engine not installed in this build.")
        self._gen_active_available = available
        self._sd_gen_btn.set_sensitive(available and not self._sd_busy)
        self._gen_note.set_text(note)
        self._gen_note.set_visible(bool(note))

    def _sd_variant(self) -> str:
        v = getattr(self._settings, "sd_variant", "auto") if self._settings else "auto"
        return "cpu" if v == "auto" else v

    def _sd_spin(self, group, title, value, lo, hi, step, digits=0):
        row = Adw.SpinRow.new_with_range(lo, hi, step)
        row.set_title(title)
        row.set_digits(digits)
        row.set_value(value)
        group.add(row)
        return row

    def _sd_combo(self, group, title, values, current):
        row = Adw.ComboRow(title=title)
        row.set_model(Gtk.StringList.new(values))
        row.set_selected(values.index(current) if current in values else 0)
        row._values = values
        group.add(row)
        return row

    def _sd_switch(self, group, title, active):
        row = Adw.SwitchRow(title=title, active=active)
        group.add(row)
        return row

    def _sd_refresh_model_list(self) -> None:
        import os
        models = list(self._settings.sd_models) if self._settings else []
        self._sd_model_paths = models
        names = [os.path.basename(p) for p in models] or ["(no model imported)"]
        self._sd_model_row.set_model(Gtk.StringList.new(names))
        if models and self._settings and self._settings.sd_last_model in models:
            self._sd_model_row.set_selected(models.index(self._settings.sd_last_model))

    def _sd_import_model(self) -> None:
        dlg = Gtk.FileDialog(title="Choose a diffusion model")
        f = Gtk.FileFilter(name="Models (*.gguf, *.safetensors, *.ckpt)")
        for pat in ("*.gguf", "*.safetensors", "*.ckpt"):
            f.add_pattern(pat)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dlg.set_filters(filters)

        def done(dialog, result) -> None:
            try:
                gfile = dialog.open_finish(result)
            except Exception:
                return
            if gfile and gfile.get_path() and self._settings:
                self._settings.add_sd_model(gfile.get_path())
                self._settings.save()
                self._sd_refresh_model_list()
                self._toast(f"Imported {gfile.get_path().split('/')[-1]}")

        dlg.open(self._parent, None, done)

    def _selected_sd_model(self):
        if not self._sd_model_paths:
            return None
        idx = self._sd_model_row.get_selected()
        if 0 <= idx < len(self._sd_model_paths):
            return self._sd_model_paths[idx]
        return None

    # ── LiteRT model folders ────────────────────────────────────────────
    def _litert_refresh_dir_list(self) -> None:
        import os
        dirs = list(self._settings.litert_diffusion_dirs) if self._settings else []
        self._litert_dir_paths = dirs
        names = [os.path.basename(p.rstrip("/")) for p in dirs] or ["(no folder imported)"]
        self._litert_row.set_model(Gtk.StringList.new(names))
        if dirs and self._settings and self._settings.litert_last_dir in dirs:
            self._litert_row.set_selected(dirs.index(self._settings.litert_last_dir))

    def _litert_import_dir(self) -> None:
        dlg = Gtk.FileDialog(title="Choose a LiteRT model folder")

        def done(dialog, result) -> None:
            try:
                gfile = dialog.select_folder_finish(result)
            except Exception:  # noqa: BLE001
                return
            if gfile and gfile.get_path() and self._settings:
                self._settings.add_litert_dir(gfile.get_path())
                self._settings.save()
                self._litert_refresh_dir_list()
                self._toast(f"Imported {gfile.get_path().split('/')[-1]}")

        dlg.select_folder(self._parent, None, done)

    def _litert_download(self) -> None:
        from .litert_download_dialog import LiterDownloadDialog

        def on_done() -> None:
            self._litert_refresh_dir_list()

        LiterDownloadDialog(self._parent, self._settings, on_done=on_done).present(
            self._parent or self)

    def _selected_litert_dir(self):
        if not self._litert_dir_paths:
            return None
        idx = self._litert_row.get_selected()
        if 0 <= idx < len(self._litert_dir_paths):
            return self._litert_dir_paths[idx]
        return None

    def _on_generate(self, _btn) -> None:
        if self._sd_busy:
            return
        if self._current_engine() == "litert":
            self._on_generate_litert()
            return
        components = self._current_sd_mode() == "components"
        bundle_item = self._selected_bundle_item() if components else None
        bundle = self._selected_bundle() if components else None
        if components:
            if bundle_item is None:
                self._toast("Download or import a component set first.")
                return
            model = ""
        else:
            model = self._selected_sd_model()
            if not model:
                self._toast("Import a diffusion model first.")
                return
        buf = self._sd_prompt.get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        if not prompt:
            self._toast("Enter a prompt.")
            return

        import time
        from .sd_backend import SDBackend, SDGenParams
        from .config import DATA_DIR

        params = SDGenParams(
            model=model, prompt=prompt,
            negative_prompt=self._sd_negative.get_text().strip(),
            width=int(self._sd_width.get_value()),
            height=int(self._sd_height.get_value()),
            steps=int(self._sd_steps.get_value()),
            cfg_scale=float(self._sd_cfg.get_value()),
            seed=int(self._sd_seed.get_value()),
            batch_count=int(self._sd_batch.get_value()),
            sampler=self._sd_sampler._values[self._sd_sampler.get_selected()],
            scheduler=self._sd_scheduler._values[self._sd_scheduler.get_selected()],
            vae_tiling=self._sd_vae_tiling.get_active(),
            vae_on_cpu=self._sd_vae_cpu.get_active(),
            clip_on_cpu=self._sd_clip_cpu.get_active(),
            diffusion_fa=self._sd_fa.get_active(),
            cache_mode=self._sd_cache._values[self._sd_cache.get_selected()],
            offload_to_cpu=self._sd_offload.get_active(),
            init_image=self._sd_init_image,
            strength=float(self._sd_strength.get_value()),
            mask_image=self._sd_mask_image if self._sd_init_image else "",
            hires_scale=self._sd_hires._scales[self._sd_hires.get_selected()],
            hires_steps=int(getattr(self._settings, "sd_hires_steps", 0) or 0),
            hires_upscaler=getattr(
                self._settings, "sd_hires_upscaler", "Latent") or "Latent",
            lora_dir=getattr(self._settings, "sd_lora_dir", "") or "")
        if bundle is not None:
            from .sd_components import bundle_dir
            d = bundle_dir(bundle.key)
            params.diffusion_model = str(d / bundle.diffusion_file)
            params.vae = str(d / bundle.vae_file)
            params.llm = str(d / bundle.llm_file)
        elif bundle_item is not None and bundle_item[0] == "custom":
            diff, vae, llm = bundle_item[1]
            params.diffusion_model = diff
            if vae:
                params.vae = vae
            if llm:
                params.llm = llm
        self._persist_sd_settings(params)

        outdir = DATA_DIR / "generated"
        stamp = int(time.time())
        outpath = str(outdir / f"box-{stamp}.png")
        if self._sd_preview_sw.get_active():
            params.preview_mode = "proj"
            params.preview_path = str(outdir / f"preview-{stamp}.png")
            self._start_preview_poll(params.preview_path)
        self._sd_backend = SDBackend(self._sd_variant())
        self._set_sd_busy(True)
        self._sd_progress.set_visible(True)
        self._sd_progress.set_fraction(0.0)
        self._sd_progress.set_text("Loading model…")

        def prog(step: int, total: int) -> None:
            GLib.idle_add(self._sd_set_progress, step, total)

        def work() -> None:
            try:
                out = self._sd_backend.generate(params, outpath, on_progress=prog)
            except Exception as e:  # noqa: BLE001
                log.exception("generation failed")
                GLib.idle_add(self._sd_failed, str(e))
                return
            GLib.idle_add(self._sd_done, out)

        threading.Thread(target=work, daemon=True).start()

    def _sd_set_progress(self, step: int, total: int) -> bool:
        self._sd_progress.set_fraction(step / total)
        self._sd_progress.set_text(f"Step {step}/{total}")
        return False

    # ── live preview polling (sd.cpp --preview) ──────────────────────────
    def _start_preview_poll(self, path: str) -> None:
        self._stop_preview_poll()
        self._sd_preview_file = path
        self._sd_preview_mtime = 0.0
        self._sd_preview_timer = GLib.timeout_add(600, self._poll_preview)

    def _stop_preview_poll(self) -> None:
        if self._sd_preview_timer:
            GLib.source_remove(self._sd_preview_timer)
            self._sd_preview_timer = 0
        if self._sd_preview_file:
            try:
                Path(self._sd_preview_file).unlink(missing_ok=True)
            except OSError:
                pass
            self._sd_preview_file = ""

    def _poll_preview(self) -> bool:
        p = Path(self._sd_preview_file) if self._sd_preview_file else None
        if p is None:
            self._sd_preview_timer = 0
            return False
        try:
            mtime = p.stat().st_mtime
        except OSError:
            return True  # not written yet — keep polling
        if mtime != self._sd_preview_mtime:
            self._sd_preview_mtime = mtime
            try:
                self._sd_pic.set_paintable(Gdk.Texture.new_from_filename(str(p)))
                self._sd_canvas_stack.set_visible_child_name("image")
            except GLib.Error:
                pass  # torn write — next tick will catch a whole file
        return True

    def _sd_done(self, out_paths) -> bool:
        from PIL import Image
        self._set_sd_busy(False)
        self._sd_progress.set_visible(False)
        self._stop_preview_poll()
        if out_paths:
            self._sd_result_path = out_paths[0]
            self._sd_result = Image.open(out_paths[0])
            self._sd_pic.set_paintable(_pil_to_texture(self._sd_result))
            self._sd_canvas_stack.set_visible_child_name("image")
            self._sd_save_btn.set_sensitive(True)
            self._show_used_seed()
            self._toast(f"Generated {len(out_paths)} image(s).")
        return False

    def _show_used_seed(self) -> None:
        seed = getattr(self._sd_backend, "last_seed", None)
        if seed is None:
            self._sd_seed_label.set_visible(False)
            self._sd_seed_reuse.set_visible(False)
            return
        self._sd_seed_label.set_text(f"seed {seed}")
        self._sd_seed_label.set_visible(True)
        self._sd_seed_reuse.set_visible(True)

    def _sd_failed(self, msg: str) -> bool:
        self._set_sd_busy(False)
        self._sd_progress.set_visible(False)
        self._stop_preview_poll()
        self._toast(f"Generation failed: {msg[:140]}")
        return False

    def _on_close_request(self, *_a) -> bool:
        if self._sd_busy:
            if self._sd_backend is not None:
                self._sd_backend.cancel()
            if getattr(self, "_litert_pipe", None) is not None:
                try:
                    self._litert_pipe.cancel()
                except Exception:  # noqa: BLE001
                    pass
        self._stop_preview_poll()
        return False  # proceed with close

    def _on_cancel_generate(self, _btn) -> None:
        if self._sd_backend is not None:
            self._sd_backend.cancel()
        if getattr(self, "_litert_pipe", None) is not None:
            self._litert_pipe.cancel()
        self._sd_progress.set_text("Cancelling…")

    # ── Generate page (LiteRT: Z-Image / FLUX klein) ────────────────────
    def _on_generate_litert(self) -> None:
        model_dir = self._selected_litert_dir()
        if not model_dir:
            self._toast("Import a LiteRT model folder first.")
            return
        buf = self._sd_prompt.get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        if not prompt:
            self._toast("Enter a prompt.")
            return

        from pathlib import Path
        from .litert_diffusion import (
            ZImagePipeline, FluxKleinPipeline, LiterDiffusionError,
        )

        d = Path(model_dir)
        # Z-Image ships qwen_enc.tflite; klein ships ke_enc0.tflite.
        if (d / "qwen_enc.tflite").is_file():
            pipe = ZImagePipeline(d)
            kind = "Z-Image"
        elif (d / "ke_enc0.tflite").is_file():
            pipe = FluxKleinPipeline(d)
            kind = "FLUX klein"
        else:
            self._toast("Folder has no Z-Image or klein graphs.")
            return
        if not pipe.is_available():
            self._toast(f"{kind} model files are incomplete in that folder.")
            return

        seed = int(self._sd_seed.get_value())
        if seed < 0:
            import random
            seed = random.randint(0, 2_000_000_000)

        self._litert_pipe = pipe
        self._sd_backend = None
        self._set_sd_busy(True)
        self._sd_progress.set_visible(True)
        self._sd_progress.set_fraction(0.0)
        self._sd_progress.set_text(f"Loading {kind}…")

        def on_progress(stage: str, frac: float) -> None:
            GLib.idle_add(self._litert_set_progress, stage, frac)

        def work() -> None:
            try:
                img = pipe.generate(prompt, seed=seed, on_progress=on_progress)
            except LiterDiffusionError as e:
                GLib.idle_add(self._sd_failed, str(e))
                return
            except Exception as e:  # noqa: BLE001
                log.exception("litert generation failed")
                GLib.idle_add(self._sd_failed, str(e))
                return
            GLib.idle_add(self._litert_done, img)

        threading.Thread(target=work, daemon=True).start()

    def _litert_set_progress(self, stage: str, frac: float) -> bool:
        self._sd_progress.set_fraction(max(0.0, min(1.0, frac)))
        self._sd_progress.set_text(stage)
        return False

    def _litert_done(self, img) -> bool:
        import time
        from .config import DATA_DIR

        self._litert_pipe = None
        self._set_sd_busy(False)
        self._sd_progress.set_visible(False)
        if img is None:
            self._toast("Generation produced no image.")
            return False
        self._sd_result = img
        self._sd_result_path = None  # PIL result — save re-encodes
        self._sd_pic.set_paintable(_pil_to_texture(img))
        self._sd_canvas_stack.set_visible_child_name("image")
        self._sd_save_btn.set_sensitive(True)
        # Auto-save alongside the sd.cpp outputs.
        try:
            outdir = DATA_DIR / "generated"
            outdir.mkdir(parents=True, exist_ok=True)
            img.save(outdir / f"box-litert-{int(time.time())}.png")
        except Exception:  # noqa: BLE001
            log.exception("litert autosave failed")
        self._toast("Generated 1 image.")
        return False

    def _set_sd_busy(self, busy: bool) -> None:
        self._sd_busy = busy
        available = getattr(self, "_gen_active_available", self._sd_available)
        self._sd_gen_btn.set_sensitive(not busy and available)
        self._sd_cancel_btn.set_sensitive(busy)
        # Lock the engine switch mid-run so the two paths can't tangle.
        self._engine_row.set_sensitive(not busy)

    def _persist_sd_settings(self, params) -> None:
        s = self._settings
        if s is None:
            return
        s.sd_width = params.width
        s.sd_height = params.height
        s.sd_steps = params.steps
        s.sd_cfg_scale = params.cfg_scale
        s.sd_sampler = params.sampler
        s.sd_scheduler = params.scheduler
        s.sd_batch_count = params.batch_count
        s.sd_vae_tiling = params.vae_tiling
        s.sd_vae_on_cpu = params.vae_on_cpu
        s.sd_clip_on_cpu = params.clip_on_cpu
        s.sd_diffusion_fa = params.diffusion_fa
        s.sd_cache_mode = params.cache_mode
        s.sd_offload_cpu = params.offload_to_cpu
        s.sd_live_preview = self._sd_preview_sw.get_active()
        s.sd_hires_scale = params.hires_scale
        s.save()

    # ── Erase page ──────────────────────────────────────────────────────
    def _build_erase_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                      margin_top=10, margin_bottom=10, margin_start=10, margin_end=10)

        self._erase_canvas = Gtk.DrawingArea(vexpand=True, hexpand=True)
        self._erase_canvas.set_draw_func(self._draw_erase, None)
        self._erase_canvas.add_css_class("card")
        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._on_paint_begin)
        drag.connect("drag-update", self._on_paint_update)
        self._erase_canvas.add_controller(drag)
        box.append(self._erase_canvas)

        self._erase_hint = Gtk.Label(
            label="Open an image, then drag to paint over what you want removed.",
            wrap=True)
        self._erase_hint.add_css_class("dim-label")
        box.append(self._erase_hint)

        brush_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        brush_row.append(Gtk.Label(label="Brush"))
        adj = Gtk.Adjustment(value=28, lower=6, upper=90, step_increment=2)
        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL, adjustment=adj,
                          hexpand=True, draw_value=False)
        scale.connect("value-changed", lambda sc: setattr(self, "_brush", int(sc.get_value())))
        brush_row.append(scale)
        box.append(brush_row)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.CENTER)
        open_btn = Gtk.Button(label="Open image…")
        open_btn.connect("clicked", lambda _b: self._open_image(for_erase=True))
        btns.append(open_btn)
        self._erase_clear_btn = Gtk.Button(label="Clear mask", sensitive=False)
        self._erase_clear_btn.connect("clicked", self._on_clear_mask)
        btns.append(self._erase_clear_btn)
        self._erase_run_btn = Gtk.Button(label="Erase", sensitive=False)
        self._erase_run_btn.add_css_class("suggested-action")
        self._erase_run_btn.connect("clicked", self._on_run_erase)
        btns.append(self._erase_run_btn)
        self._erase_save_btn = Gtk.Button(label="Save…", sensitive=False)
        self._erase_save_btn.connect("clicked", lambda _b: self._save(self._erase_orig))
        btns.append(self._erase_save_btn)
        box.append(btns)

        self._erase_spinner = Gtk.Spinner()
        box.append(self._erase_spinner)
        return box

    def _draw_erase(self, area, cr, width, height, _data) -> None:
        if self._erase_surface is None:
            return
        sw = self._erase_surface.get_width()
        sh = self._erase_surface.get_height()
        scale = min(width / sw, height / sh)
        self._display_scale = scale
        dw, dh = sw * scale, sh * scale
        ox, oy = (width - dw) / 2, (height - dh) / 2
        self._erase_origin = (ox, oy)

        cr.save()
        cr.translate(ox, oy)
        cr.scale(scale, scale)
        cr.set_source_surface(self._erase_surface, 0, 0)
        cr.paint()
        if self._mask is not None:
            import numpy as np
            if (self._mask == 0).any():
                overlay = self._mask_overlay_surface()
                if overlay is not None:
                    cr.set_source_surface(overlay, 0, 0)
                    cr.paint_with_alpha(0.5)
        cr.restore()

    def _mask_overlay_surface(self):
        import cairo
        import numpy as np
        if self._mask is None:
            return None
        h, w = self._mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[self._mask == 0] = (60, 60, 255, 255)
        buf = bytearray(rgba.tobytes())
        surface = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_ARGB32, w, h, w * 4)
        self._overlay_buf = buf
        return surface

    def _canvas_to_image(self, x, y):
        ox, oy = getattr(self, "_erase_origin", (0, 0))
        s = self._display_scale or 1.0
        return (x - ox) / s, (y - oy) / s

    def _paint_at(self, cx, cy) -> None:
        if self._mask is None:
            return
        import numpy as np
        ix, iy = self._canvas_to_image(cx, cy)
        h, w = self._mask.shape
        r = self._brush
        y0, y1 = max(0, int(iy - r)), min(h, int(iy + r) + 1)
        x0, x1 = max(0, int(ix - r)), min(w, int(ix + r) + 1)
        if y0 >= y1 or x0 >= x1:
            return
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - ix) ** 2 + (yy - iy) ** 2 <= r * r
        self._mask[y0:y1, x0:x1][circle] = 0
        self._erase_canvas.queue_draw()

    def _on_paint_begin(self, gesture, x, y) -> None:
        if self._erase_orig is None or self._busy:
            return
        self._paint_at(x, y)

    def _on_paint_update(self, gesture, dx, dy) -> None:
        if self._erase_orig is None or self._busy:
            return
        ok, sx, sy = gesture.get_start_point()
        if ok:
            self._paint_at(sx + dx, sy + dy)

    def _on_clear_mask(self, _btn) -> None:
        if self._mask is not None:
            self._mask[:] = 255
            self._erase_canvas.queue_draw()

    def _on_run_erase(self, _btn) -> None:
        if self._erase_orig is None or self._mask is None or self._busy:
            return
        import numpy as np
        from PIL import Image
        if (self._mask == 0).sum() == 0:
            self._toast("Paint over something to erase first.")
            return
        self._set_erase_busy(True)
        img = self._erase_orig
        mask_img = Image.fromarray(self._mask, "L")

        def work() -> None:
            try:
                result = self._erase_engine.erase(img, mask_img)
            except VisionModelError as e:
                GLib.idle_add(self._erase_failed, str(e))
                return
            except Exception as e:  # noqa: BLE001
                log.exception("erase failed")
                GLib.idle_add(self._erase_failed, str(e))
                return
            GLib.idle_add(self._erase_done, result)

        threading.Thread(target=work, daemon=True).start()

    def _erase_done(self, result) -> bool:
        import numpy as np
        self._erase_orig = result
        self._cache_erase_surface(result)
        self._mask = np.full((result.size[1], result.size[0]), 255, np.uint8)
        self._set_erase_busy(False)
        self._erase_save_btn.set_sensitive(True)
        self._erase_canvas.queue_draw()
        self._toast("Erased. Paint again to remove more, or Save.")
        return False

    def _erase_failed(self, msg: str) -> bool:
        self._set_erase_busy(False)
        self._toast(f"Erase failed: {msg[:120]}")
        return False

    def _set_erase_busy(self, busy: bool) -> None:
        self._busy = busy
        self._erase_run_btn.set_sensitive(not busy and self._erase_orig is not None)
        self._erase_clear_btn.set_sensitive(not busy and self._erase_orig is not None)
        if busy:
            self._erase_spinner.start()
        else:
            self._erase_spinner.stop()

    def _cache_erase_surface(self, img) -> None:
        import cairo
        rgba = img.convert("RGBA")
        w, h = rgba.size
        buf = bytearray(rgba.tobytes("raw", "BGRA"))
        self._erase_surface_data = buf
        self._erase_surface = cairo.ImageSurface.create_for_data(
            buf, cairo.FORMAT_ARGB32, w, h, w * 4)

    # ── Upscale page ────────────────────────────────────────────────────
    def _build_upscale_page(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                      margin_top=10, margin_bottom=10, margin_start=10, margin_end=10)
        # Hidden until a result exists (same fix as the Generate page).
        self._upscale_pic = Gtk.Picture(
            vexpand=True, hexpand=True, visible=False
        )
        self._upscale_pic.add_css_class("card")
        box.append(self._upscale_pic)

        self._upscale_info = Gtk.Label(label="Open an image to upscale it 4×.")
        self._upscale_info.add_css_class("dim-label")
        box.append(self._upscale_info)

        self._upscale_progress = Gtk.ProgressBar(visible=False)
        box.append(self._upscale_progress)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8, halign=Gtk.Align.CENTER)
        open_btn = Gtk.Button(label="Open image…")
        open_btn.connect("clicked", lambda _b: self._open_image(for_erase=False))
        btns.append(open_btn)
        self._upscale_run_btn = Gtk.Button(label="Upscale 4×", sensitive=False)
        self._upscale_run_btn.add_css_class("suggested-action")
        self._upscale_run_btn.connect("clicked", self._on_run_upscale)
        btns.append(self._upscale_run_btn)
        self._upscale_save_btn = Gtk.Button(label="Save…", sensitive=False)
        self._upscale_save_btn.connect("clicked", lambda _b: self._save(self._upscale_result))
        btns.append(self._upscale_save_btn)
        box.append(btns)
        return box

    def _on_run_upscale(self, _btn) -> None:
        if self._upscale_orig is None or self._busy:
            return
        self._busy = True
        self._upscale_run_btn.set_sensitive(False)
        self._upscale_progress.set_visible(True)
        self._upscale_progress.set_fraction(0.0)
        img = self._upscale_orig

        def prog(f: float) -> None:
            GLib.idle_add(self._upscale_progress.set_fraction, f)

        def work() -> None:
            try:
                result = self._upscale_engine.upscale(img, on_progress=prog)
            except Exception as e:  # noqa: BLE001
                log.exception("upscale failed")
                GLib.idle_add(self._upscale_failed, str(e))
                return
            GLib.idle_add(self._upscale_done, result)

        threading.Thread(target=work, daemon=True).start()

    def _upscale_done(self, result) -> bool:
        self._upscale_result = result
        self._upscale_pic.set_paintable(_pil_to_texture(result))
        self._upscale_pic.set_visible(True)
        self._upscale_progress.set_visible(False)
        self._upscale_info.set_label(
            f"Upscaled to {result.size[0]}×{result.size[1]}. Save to keep it.")
        self._upscale_save_btn.set_sensitive(True)
        self._upscale_run_btn.set_sensitive(True)
        self._busy = False
        return False

    def _upscale_failed(self, msg: str) -> bool:
        self._upscale_progress.set_visible(False)
        self._upscale_run_btn.set_sensitive(True)
        self._busy = False
        self._toast(f"Upscale failed: {msg[:120]}")
        return False

    # ── shared: open / save ─────────────────────────────────────────────
    def _open_image(self, for_erase: bool) -> None:
        dlg = Gtk.FileDialog(title="Choose an image")
        f = Gtk.FileFilter(name="Images")
        for pat in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            f.add_pattern(pat)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dlg.set_filters(filters)

        def done(dialog, result) -> None:
            try:
                gfile = dialog.open_finish(result)
            except Exception:
                return
            if gfile and gfile.get_path():
                self._load_image(gfile.get_path(), for_erase)

        dlg.open(self._parent, None, done)

    def _load_image(self, path: str, for_erase: bool) -> None:
        from PIL import Image
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:  # noqa: BLE001
            self._toast(f"Couldn't open image: {e}")
            return
        if for_erase:
            import numpy as np
            self._erase_orig = img
            self._cache_erase_surface(img)
            self._mask = np.full((img.size[1], img.size[0]), 255, np.uint8)
            self._erase_run_btn.set_sensitive(True)
            self._erase_clear_btn.set_sensitive(True)
            self._erase_save_btn.set_sensitive(False)
            self._erase_hint.set_label(
                "Drag to paint over what you want removed, then press Erase.")
            self._erase_canvas.queue_draw()
        else:
            self._upscale_orig = img
            self._upscale_result = None
            self._upscale_pic.set_paintable(_pil_to_texture(img))
            self._upscale_info.set_label(
                f"{img.size[0]}×{img.size[1]} → {img.size[0]*4}×{img.size[1]*4} on Upscale.")
            self._upscale_run_btn.set_sensitive(True)
            self._upscale_save_btn.set_sensitive(False)

    def _save(self, img, src_path: str | None = None) -> None:
        if img is None and not src_path:
            return
        dlg = Gtk.FileDialog(title="Save image", initial_name="box-image.png")

        def done(dialog, result) -> None:
            try:
                gfile = dialog.save_finish(result)
            except Exception:
                return
            if not (gfile and gfile.get_path()):
                return
            path = gfile.get_path()
            if not path.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                path += ".png"
            try:
                # Copy the original bytes when we have them — sd.cpp embeds
                # webui-compatible generation params in its PNGs, and a PIL
                # re-encode would strip that metadata.
                if src_path and Path(src_path).is_file() and path.lower().endswith(".png"):
                    import shutil
                    shutil.copyfile(src_path, path)
                else:
                    img.save(path)
                self._toast(f"Saved to {path}")
            except Exception as e:  # noqa: BLE001
                self._toast(f"Save failed: {e}")

        dlg.save(self._parent, None, done)

    def _toast(self, text: str) -> None:
        try:
            self._parent._show_toast(text)
        except Exception:  # noqa: BLE001
            log.info("toast: %s", text)


class _InpaintMaskWindow(Adw.Window):
    """Paint an inpaint mask over the img2img source. White = regenerate.

    Deliberately simple: brush circles onto an overlay; "Use mask" writes
    a black/white PNG (source resolution) to the cache dir and hands the
    path back via ``on_done(path)``.
    """

    def __init__(self, parent, image_path: str, on_done) -> None:
        super().__init__(
            title="Paint inpaint mask", default_width=820,
            default_height=640, transient_for=parent, modal=True,
        )
        self.add_css_class("aux-solid")  # opaque under glass modes
        from PIL import Image
        self._img = Image.open(image_path).convert("RGB")
        self._tex = _pil_to_texture(self._img)
        self._dots: list[tuple[float, float, float]] = []  # x, y, r (image px)
        self._brush = 40
        self._on_done = on_done

        tv = Adw.ToolbarView()
        tv.add_top_bar(Adw.HeaderBar())
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8,
                      margin_top=8, margin_bottom=10, margin_start=10,
                      margin_end=10)
        self._area = Gtk.DrawingArea(vexpand=True, hexpand=True)
        self._area.set_draw_func(self._draw, None)
        drag = Gtk.GestureDrag()
        drag.connect("drag-begin", self._paint_begin)
        drag.connect("drag-update", self._paint_update)
        self._area.add_controller(drag)
        box.append(self._area)

        brush_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        brush_row.append(Gtk.Label(label="Brush"))
        adj = Gtk.Adjustment(value=self._brush, lower=8, upper=160,
                             step_increment=4)
        scale = Gtk.Scale(orientation=Gtk.Orientation.HORIZONTAL,
                          adjustment=adj, hexpand=True, draw_value=False)
        scale.connect(
            "value-changed",
            lambda sc: setattr(self, "_brush", int(sc.get_value())))
        brush_row.append(scale)
        box.append(brush_row)

        btns = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
                       halign=Gtk.Align.CENTER)
        clear = Gtk.Button(label="Clear")
        clear.connect("clicked", self._on_clear)
        btns.append(clear)
        use = Gtk.Button(label="Use mask")
        use.add_css_class("suggested-action")
        use.connect("clicked", self._on_use)
        btns.append(use)
        box.append(btns)
        tv.set_content(box)
        self.set_content(tv)
        self.present()

    # ── geometry: fit image into the widget, keep aspect ────────────────
    def _fit(self) -> tuple[float, float, float]:
        aw = max(1, self._area.get_width())
        ah = max(1, self._area.get_height())
        iw, ih = self._img.size
        scale = min(aw / iw, ah / ih)
        ox = (aw - iw * scale) / 2
        oy = (ah - ih * scale) / 2
        return scale, ox, oy

    def _to_image_xy(self, wx: float, wy: float) -> tuple[float, float] | None:
        scale, ox, oy = self._fit()
        ix = (wx - ox) / scale
        iy = (wy - oy) / scale
        iw, ih = self._img.size
        if 0 <= ix < iw and 0 <= iy < ih:
            return ix, iy
        return None

    def _draw(self, _area, cr, w, h, _data) -> None:
        scale, ox, oy = self._fit()
        iw, ih = self._img.size
        cr.save()
        cr.translate(ox, oy)
        cr.scale(scale, scale)
        Gdk.cairo_set_source_pixbuf if False else None  # noqa: B018
        # Draw the texture via a Gdk snapshot-free path: use cairo surface
        # from PIL bytes (cheap enough at dialog size).
        import cairo
        rgba = self._img.convert("RGBA").tobytes()
        surf = cairo.ImageSurface.create_for_data(
            bytearray(rgba), cairo.FORMAT_ARGB32, iw, ih, iw * 4)
        cr.set_source_surface(surf, 0, 0)
        cr.paint()
        cr.set_source_rgba(1.0, 0.2, 0.2, 0.45)
        for x, y, r in self._dots:
            cr.arc(x, y, r, 0, 6.283185)
            cr.fill()
        cr.restore()

    def _paint_at(self, wx: float, wy: float) -> None:
        pt = self._to_image_xy(wx, wy)
        if pt is None:
            return
        scale, _o, _o2 = self._fit()
        self._dots.append((pt[0], pt[1], self._brush / max(scale, 1e-6) / 2))
        self._area.queue_draw()

    def _paint_begin(self, gesture, x, y) -> None:
        self._start = (x, y)
        self._paint_at(x, y)

    def _paint_update(self, gesture, dx, dy) -> None:
        x0, y0 = self._start
        self._paint_at(x0 + dx, y0 + dy)

    def _on_clear(self, *_a) -> None:
        self._dots.clear()
        self._area.queue_draw()

    def _on_use(self, *_a) -> None:
        if not self._dots:
            self._on_done("")
            self.close()
            return
        from PIL import Image, ImageDraw
        from .config import CACHE_DIR
        mask = Image.new("L", self._img.size, 0)
        d = ImageDraw.Draw(mask)
        for x, y, r in self._dots:
            d.ellipse((x - r, y - r, x + r, y + r), fill=255)
        out = CACHE_DIR / "inpaint-mask.png"
        mask.save(out)
        self._on_done(str(out))
        self.close()
