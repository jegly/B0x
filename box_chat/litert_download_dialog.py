"""Download dialog for the LiteRT diffusion models (Z-Image / FLUX klein).

UI tier for :mod:`box_chat.litert_diffusion_models`. Each model is a big
multi-file directory download (~7–11GB); this shows one row per model with a
whole-set progress bar, resume, and cancel. On completion it registers the
model directory in settings so the Generate page picks it up.
"""
from __future__ import annotations

import logging
import threading

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from .config import MODELS_DIR
from .litert_diffusion_models import (
    LITERT_MODELS, LiterDownloadCancelled, LiterModel, download_model,
)

log = logging.getLogger(__name__)


def _fmt(n: int) -> str:
    if n < 1 << 20:
        return f"{n / 1024:.0f} KB"
    if n < 1 << 30:
        return f"{n / 1024 ** 2:.0f} MB"
    return f"{n / 1024 ** 3:.2f} GB"


def litert_dir_for(model: LiterModel):
    return MODELS_DIR / "litert" / model.key


def is_complete(model: LiterModel) -> bool:
    d = litert_dir_for(model)
    return all(
        (d / f.name).is_file() and (d / f.name).stat().st_size == f.size
        for f in model.files
    )


class LiterModelRow(Adw.ActionRow):
    """One multi-file LiteRT diffusion model — Download → progress → Ready.

    Reusable outside :class:`LiterDownloadDialog` (e.g. the Preferences → Models
    page). ``settings`` receives the completed model dir via ``add_litert_dir``;
    ``on_done`` (optional) fires once after a successful download.
    """

    def __init__(self, model: LiterModel, settings=None, on_done=None) -> None:
        super().__init__(title=model.name)
        self._model = model
        self._settings = settings
        self._on_done = on_done
        self._base_subtitle = f"{model.info}  ·  {_fmt(model.total_bytes)}"
        self.set_subtitle(self._base_subtitle)
        self.set_use_markup(False)
        self._cancel_flag = False

        self._stack = Gtk.Stack()
        dl = Gtk.Button(label="Download", valign=Gtk.Align.CENTER)
        dl.add_css_class("suggested-action")
        dl.connect("clicked", self._start)
        self._stack.add_named(dl, "idle")
        cancel = Gtk.Button(label="Cancel", valign=Gtk.Align.CENTER)
        cancel.add_css_class("destructive-action")
        cancel.connect("clicked", self._cancel)
        self._stack.add_named(cancel, "busy")
        ready = Gtk.Label(label="Ready", valign=Gtk.Align.CENTER)
        ready.add_css_class("dim-label")
        self._stack.add_named(ready, "done")
        self.add_suffix(self._stack)

        if is_complete(model):
            # Already fully on disk (a prior download, or the Models page opened
            # after one finished). Make sure it's registered so the Generate page
            # can actually select it — but only touch settings if it's new, and
            # don't fire the on_done refresh (nothing just changed on screen).
            if self._settings is not None:
                dirp = str(litert_dir_for(model))
                if dirp not in self._settings.litert_diffusion_dirs:
                    self._settings.add_litert_dir(dirp)
                    self._settings.save()
            self._set_done(register=False)

    def _start(self, _btn) -> None:
        self._cancel_flag = False
        self._stack.set_visible_child_name("busy")
        self.set_subtitle("Preparing…")
        dest = litert_dir_for(self._model)
        # Share the tokenizer files from any other already-downloaded model.
        share_from = [
            str(litert_dir_for(m)) for m in LITERT_MODELS if m.key != self._model.key
        ]
        if self._settings is not None:
            share_from += list(self._settings.litert_diffusion_dirs)

        def on_overall(done: int, total: int) -> None:
            GLib.idle_add(self._progress, done, total)

        def on_status(text: str) -> None:
            GLib.idle_add(self.set_subtitle, text)

        def work() -> None:
            try:
                download_model(
                    self._model, dest,
                    on_overall=on_overall, on_status=on_status,
                    is_cancelled=lambda: self._cancel_flag,
                    share_from=share_from,
                )
            except LiterDownloadCancelled:
                GLib.idle_add(self._reset, "Cancelled")
                return
            except Exception as e:  # noqa: BLE001
                log.exception("litert download failed")
                GLib.idle_add(self._reset, f"Error: {str(e)[:100]}")
                return
            GLib.idle_add(self._set_done, True)

        threading.Thread(target=work, daemon=True).start()

    def _progress(self, done: int, total: int) -> bool:
        pct = (done / total * 100) if total else 0
        self.set_subtitle(f"{_fmt(done)} / {_fmt(total)}  ·  {pct:.0f}%")
        return False

    def _cancel(self, _btn) -> None:
        self._cancel_flag = True
        self.set_subtitle("Cancelling…")

    def _reset(self, msg: str) -> bool:
        self._stack.set_visible_child_name("idle")
        self.set_subtitle(f"{msg}  ·  {self._base_subtitle}")
        return False

    def _set_done(self, register: bool = True) -> bool:
        self._stack.set_visible_child_name("done")
        self.set_subtitle(f"Ready  ·  {_fmt(self._model.total_bytes)}")
        if register and self._settings is not None:
            self._settings.add_litert_dir(str(litert_dir_for(self._model)))
            self._settings.save()
            if self._on_done is not None:
                try:
                    self._on_done()
                except Exception:  # noqa: BLE001
                    log.exception("litert row on_done callback failed")
        return False


class LiterDownloadDialog(Adw.Dialog):
    def __init__(self, parent_window, settings=None, on_done=None) -> None:
        super().__init__()
        self.set_title("Download LiteRT models")
        self.set_content_width(640)
        self.set_content_height(420)
        self._settings = settings
        self._on_done = on_done

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())
        page = Adw.PreferencesPage()
        group = Adw.PreferencesGroup(
            title="On-device text-to-image (LiteRT)",
            description=(
                "Large multi-file downloads that resume if interrupted. The Qwen "
                "tokenizer (~0.8GB) is shared — downloaded once across both models."
            ),
        )
        page.add(group)
        for model in LITERT_MODELS:
            group.add(LiterModelRow(model, self._settings, self._notify_done))
        toolbar.set_content(page)
        self.set_child(toolbar)

    def _notify_done(self) -> None:
        if self._on_done is not None:
            try:
                self._on_done()
            except Exception:  # noqa: BLE001
                log.exception("litert download on_done callback failed")
