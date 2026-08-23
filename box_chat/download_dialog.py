"""HuggingFace model download dialog."""
from __future__ import annotations

import threading
import time
import urllib.request
from pathlib import Path
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from .config import MODELS_DIR
from .net import require_https

MODELS: list[dict] = [
    {
        "name": "Gemma 4 E2B",
        "subtitle": "2.59 GB · 2B effective params · faster",
        "filename": "gemma-4-E2B-it.litertlm",
        "url": (
            "https://huggingface.co/litert-community/gemma-4-E2B-it-litert-lm"
            "/resolve/main/gemma-4-E2B-it.litertlm?download=true"
        ),
    },
    {
        "name": "Gemma 4 E4B",
        "subtitle": "3.66 GB · 4B effective params · higher quality",
        "filename": "gemma-4-E4B-it.litertlm",
        "url": (
            "https://huggingface.co/litert-community/gemma-4-E4B-it-litert-lm"
            "/resolve/main/gemma-4-E4B-it.litertlm?download=true"
        ),
    },
]


def _fmt(n: int) -> str:
    if n < 1 << 20:
        return f"{n / 1024:.1f} KB"
    if n < 1 << 30:
        return f"{n / 1024**2:.1f} MB"
    return f"{n / 1024**3:.2f} GB"


class _Downloader:
    def __init__(
        self,
        url: str,
        dest: Path,
        on_progress: Callable[[int, int], None],
        on_done: Callable[[str], None],
        on_error: Callable[[str], None],
    ) -> None:
        self._url = url
        self._dest = dest
        self._on_progress = on_progress
        self._on_done = on_done
        self._on_error = on_error
        self._cancelled = False
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self) -> None:
        self._cancelled = True

    def _run(self) -> None:
        tmp = self._dest.with_suffix(".tmp")
        try:
            require_https(self._url)
            resume_pos = tmp.stat().st_size if tmp.exists() else 0

            headers = {
                "User-Agent": "Box/0.1 (LiteRT-LM desktop)",
                "Accept-Encoding": "identity",
            }
            if resume_pos > 0:
                headers["Range"] = f"bytes={resume_pos}-"

            req = urllib.request.Request(self._url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp:
                if resp.status == 206:
                    done = resume_pos
                    file_mode = "ab"
                else:
                    done = 0
                    resume_pos = 0
                    file_mode = "wb"

                content_length = int(resp.headers.get("Content-Length", 0))
                total = resume_pos + content_length if content_length else 0

                last_ui_update = 0.0
                with open(tmp, file_mode) as f:
                    while not self._cancelled:
                        chunk = resp.read(4 * 1024 * 1024)  # 4 MB
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        now = time.monotonic()
                        if now - last_ui_update >= 0.25:
                            GLib.idle_add(self._on_progress, done, total)
                            last_ui_update = now

            if self._cancelled:
                tmp.unlink(missing_ok=True)
                return
            tmp.rename(self._dest)
            GLib.idle_add(self._on_done, str(self._dest))
        except Exception as exc:  # noqa: BLE001
            # Keep the .tmp so the next attempt can resume — only delete on cancel.
            if not self._cancelled:
                GLib.idle_add(self._on_error, str(exc))


class _ModelRow(Adw.ActionRow):
    """One model entry — Download → progress text → Use."""

    def __init__(self, model: dict, on_use: Callable[[str], None]) -> None:
        super().__init__(title=model["name"], subtitle=model["subtitle"])
        self._model = model
        self._on_use = on_use
        self._dl: _Downloader | None = None
        self._dest = MODELS_DIR / model["filename"]

        self._stack = Gtk.Stack(
            transition_type=Gtk.StackTransitionType.CROSSFADE,
            transition_duration=120,
        )

        dl_btn = Gtk.Button(label="Download", valign=Gtk.Align.CENTER)
        dl_btn.add_css_class("suggested-action")
        dl_btn.connect("clicked", self._start)
        self._stack.add_named(dl_btn, "idle")

        cancel_btn = Gtk.Button(label="Cancel", valign=Gtk.Align.CENTER)
        cancel_btn.add_css_class("destructive-action")
        cancel_btn.connect("clicked", self._cancel)
        self._stack.add_named(cancel_btn, "busy")

        use_btn = Gtk.Button(label="Use", valign=Gtk.Align.CENTER)
        use_btn.add_css_class("suggested-action")
        use_btn.connect("clicked", lambda *_: on_use(str(self._dest)))
        self._stack.add_named(use_btn, "done")

        self.add_suffix(self._stack)

        if self._dest.exists():
            self._set_done()

    def _start(self, _btn) -> None:
        self._stack.set_visible_child_name("busy")
        tmp = self._dest.with_suffix(".tmp")
        if tmp.exists() and (sz := tmp.stat().st_size) > 0:
            self.set_subtitle(f"Resuming from {_fmt(sz)}…")
        else:
            self.set_subtitle("Connecting…")
        self._dl = _Downloader(
            self._model["url"],
            self._dest,
            self._on_progress,
            self._on_done,
            self._on_error,
        )

    def _cancel(self, _btn) -> None:
        if self._dl:
            self._dl.cancel()
            self._dl = None
        self._stack.set_visible_child_name("idle")
        self.set_subtitle(self._model["subtitle"])

    def _on_progress(self, done: int, total: int) -> None:
        if total:
            self.set_subtitle(
                f"{_fmt(done)} / {_fmt(total)}  ·  {done/total*100:.0f}%"
            )
        else:
            self.set_subtitle(f"{_fmt(done)} downloaded…")

    def _on_done(self, _path: str) -> None:
        self._dl = None
        self._set_done()

    def _on_error(self, msg: str) -> None:
        self._dl = None
        self._stack.set_visible_child_name("idle")
        self.set_subtitle(f"Error: {msg[:100]}")

    def _set_done(self) -> None:
        self._stack.set_visible_child_name("done")
        self.set_subtitle("Ready · click Use to load")


class DownloadDialog(Adw.Dialog):
    """Modal dialog for downloading Gemma 4 models from HuggingFace."""

    def __init__(self, on_model_selected: Callable[[str], None]) -> None:
        super().__init__(title="Download models")
        self.set_content_width(500)
        self._on_model_selected = on_model_selected

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())

        page = Adw.PreferencesPage()
        group = Adw.PreferencesGroup(
            title="Gemma 4  ·  litert-community on HuggingFace",
            description=(
                "No account required. Files are saved to "
                "~/.local/share/box/models/. "
                "First load takes 10–30 s while the engine caches weights."
            ),
        )
        for model in MODELS:
            group.add(_ModelRow(model, on_use=self._use))
        page.add(group)

        toolbar.set_content(page)
        self.set_child(toolbar)

    def _use(self, path: str) -> None:
        self._on_model_selected(path)
        self.close()
