"""The Adw.Application — entry point, actions, theme loading."""
from __future__ import annotations

import logging
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

from . import APP_ID, APP_NAME, __version__
from .config import DB_PATH, Settings
from .database import Database
from .engine import EngineManager
from .themes import build_css, get_accent_hex, get_theme
from .window import MainWindow

log = logging.getLogger(__name__)

# Icon search path. In a source checkout data/icons sits two levels above
# this file; in the packaged build box_chat is a compiled .so and the icons
# ship at /usr/share/box/icons (first match wins).
def _icons_dir() -> Path:
    packaged = Path("/usr/share/box/icons")
    if packaged.is_dir():
        return packaged
    return Path(__file__).parent.parent / "data" / "icons"


_DATA_DIR = _icons_dir()


class BoxApp(Adw.Application):
    def __init__(self) -> None:
        super().__init__(
            application_id=APP_ID,
            flags=Gio.ApplicationFlags.DEFAULT_FLAGS,
        )

        # Shared singletons constructed lazily in do_startup.
        self.settings: Settings | None = None
        self.db: Database | None = None
        self.engine: EngineManager | None = None
        self._main_window: MainWindow | None = None
        self._css_provider: Gtk.CssProvider | None = None

    # ── Application lifecycle ──────────────────────────────────────────────
    def do_startup(self) -> None:  # noqa: D401
        Adw.Application.do_startup(self)

        self.settings = Settings.load()
        self.db = Database(DB_PATH)
        self.engine = EngineManager()

        self._setup_icon_theme()
        try:
            from .fonts import register_bundled_fonts
            n = register_bundled_fonts()
            log.info("registered %d bundled fonts", n)
        except Exception:  # noqa: BLE001 — never block startup on fonts
            log.exception("bundled font registration failed")
        self._apply_theme()
        self._install_actions()

    def do_activate(self) -> None:  # noqa: D401
        if self._main_window is None:
            assert self.settings and self.db and self.engine
            self._main_window = MainWindow(
                application=self,
                settings=self.settings,
                db=self.db,
                engine=self.engine,
            )
            # Apply the textview background now that the window exists.
            theme = get_theme(self.settings.theme)
            accent_hex = get_accent_hex(self.settings.theme, self.settings.accent_name)
            ac = accent_hex or theme.accent
            composer_bg = ac if self.settings.composer_use_accent else theme.base
            self._main_window._chat_view.set_input_bg(composer_bg)
            # System tray — close-to-tray parks + locks the app (see Tray).
            try:
                from .tray import Tray
                self._tray = Tray(self)
                self._main_window.connect("close-request", self._on_window_close)
            except Exception:  # noqa: BLE001 — tray is best-effort
                log.exception("tray init failed")
        self._main_window.present()

    def _on_window_close(self, _win) -> bool:
        """Close-to-tray: if a tray is registered, hide + lock instead of
        quitting. Returns True to stop the default close (which would quit)."""
        if getattr(self, "_tray", None) is not None and self._tray.ok:
            self._main_window.lock_now()
            self._main_window.set_visible(False)
            return True  # swallow the close
        return False  # no tray → normal close/quit

    def tray_toggle(self) -> None:
        w = self._main_window
        if w is None:
            return
        if w.get_visible():
            w.set_visible(False)
        else:
            w.set_visible(True)
            w.present()

    def hide_aux_windows(self) -> None:
        """Hide Box Code + Image Tools when App Lock engages — they're
        separate toplevels the lock screen doesn't cover."""
        for attr in ("_code_window", "_image_tools"):
            win = getattr(self, attr, None)
            if win is not None:
                try:
                    win.set_visible(False)
                except Exception:  # noqa: BLE001
                    log.exception("could not hide %s", attr)

    def tray_code(self) -> None:
        """Open Box Code from the tray. While locked, surface the lock
        screen instead — App Lock gates everything."""
        w = self._main_window
        if w is None:
            return
        if w.is_locked():
            w.set_visible(True)
            w.present()
            return
        self._on_code_mode()

    def tray_lock(self) -> None:
        if self._main_window is not None:
            self._main_window.lock_now()

    def tray_quit(self) -> None:
        self._tray = None  # let close actually quit
        self.quit()

    def do_shutdown(self) -> None:  # noqa: D401
        # Graceful teardown for the Box Code agent's llama-server child
        # (the sandbox lifeline would reap it anyway; graceful is better).
        code_win = getattr(self, "_code_window", None)
        if code_win is not None:
            try:
                code_win._teardown_agent()
            except Exception:  # noqa: BLE001
                log.exception("code window teardown failed")
        if self.settings and self._main_window:
            w, h = self._main_window.get_default_size()
            self.settings.window_width = w
            self.settings.window_height = h
            self.settings.save()
        if self.engine:
            self.engine.shutdown()
        if self.db:
            self.db.close()
        Adw.Application.do_shutdown(self)

    # ── Icon theme ────────────────────────────────────────────────────────
    def _setup_icon_theme(self) -> None:
        display = Gdk.Display.get_default()
        if display is None:
            return
        if _DATA_DIR.exists():
            Gtk.IconTheme.get_for_display(display).add_search_path(str(_DATA_DIR))

    # ── Theme application ──────────────────────────────────────────────────
    def _apply_theme(self) -> None:
        """Apply the current theme: tell Adw light/dark, then push CSS overrides."""
        assert self.settings
        theme = get_theme(self.settings.theme)

        sm = Adw.StyleManager.get_default()
        sm.set_color_scheme(
            Adw.ColorScheme.FORCE_DARK if theme.is_dark else Adw.ColorScheme.FORCE_LIGHT
        )

        display = Gdk.Display.get_default()
        if display is None:
            return  # headless / before display is ready

        accent_hex = get_accent_hex(self.settings.theme, self.settings.accent_name)
        bubble_accent_hex = get_accent_hex(self.settings.theme, self.settings.bubble_accent_name)
        css_text = build_css(
            theme,
            font_size=self.settings.font_size,
            accent_hex=accent_hex,
            bubble_accent_hex=bubble_accent_hex,
            bubble_style=self.settings.bubble_style,
            bubble_opacity=self.settings.bubble_opacity,
            user_bubble_text_color=self.settings.user_bubble_text_color,
            asst_bubble_text_color=self.settings.asst_bubble_text_color,
            chat_font_family=self.settings.chat_font_family,
            composer_use_accent=self.settings.composer_use_accent,
            glass=self.settings.glass_mode,
            glass_opacity=self.settings.glass_opacity,
            glass_liquid=self.settings.glass_liquid,
            header_font=self.settings.header_font_family,
        )

        # First time: create provider and attach. Subsequent times: just replace data.
        if self._css_provider is None:
            self._css_provider = Gtk.CssProvider()
            Gtk.StyleContext.add_provider_for_display(
                display,
                self._css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_USER + 1000,
            )
        self._css_provider.load_from_data(css_text.encode("utf-8"))

        if self._main_window is not None:
            ac = accent_hex or theme.accent
            composer_bg = ac if self.settings.composer_use_accent else theme.mantle
            self._main_window._chat_view.set_input_bg(composer_bg)
            # Display-math images are rendered in the assistant text colour so
            # they match the surrounding text on the current theme.
            from .chat_view import set_math_color
            set_math_color(self.settings.asst_bubble_text_color or theme.text)

    def refresh_theme(self) -> None:
        """Called by the preferences dialog after the user changes theme/font."""
        self._apply_theme()

    # ── Actions (menus, shortcuts) ─────────────────────────────────────────
    def _install_actions(self) -> None:
        actions = [
            ("preferences", self._on_preferences, ["<Primary>comma"]),
            ("about",       self._on_about,       None),
            ("quit",        self._on_quit,        ["<Primary>q"]),
            ("new-chat",    self._on_new_chat,    ["<Primary>n"]),
            ("image-tools", self._on_image_tools, ["<Primary>i"]),
            ("code-mode",   self._on_code_mode,   ["<Primary>grave"]),
            ("lock",        self._on_lock,        ["<Primary>l"]),
        ]
        for name, cb, accels in actions:
            act = Gio.SimpleAction.new(name, None)
            act.connect("activate", cb)
            self.add_action(act)
            if accels:
                self.set_accels_for_action(f"app.{name}", accels)

        # SIGTERM/SIGINT → orderly quit. Without this, GTK's default handling
        # can kill the process before do_shutdown runs, so the engine never
        # stops its llama-server child (the sandbox lifeline still reaps it,
        # but graceful should mean graceful).
        for signum in (2, 15):  # SIGINT, SIGTERM
            GLib.unix_signal_add(
                GLib.PRIORITY_DEFAULT, signum, self._on_terminate_signal
            )

    def _on_terminate_signal(self) -> bool:
        log.info("termination signal received — shutting down cleanly")
        self.quit()
        return GLib.SOURCE_REMOVE

    def _on_lock(self, *_args) -> None:
        if self._main_window is not None:
            self._main_window.lock_now()

    def _on_image_tools(self, *_args) -> None:
        if not self._main_window:
            return
        if getattr(self._main_window, "is_locked", lambda: False)():
            return
        from .image_tools_dialog import ImageToolsDialog

        win = getattr(self, "_image_tools", None)
        if win is None:
            win = ImageToolsDialog(self._main_window, self.settings)
            self._image_tools = win
            win.connect(
                "close-request",
                lambda *_: setattr(self, "_image_tools", None) or False,
            )
        win.present()

    def _on_code_mode(self, *_args) -> None:
        if not self._main_window or not self.settings:
            return
        if getattr(self._main_window, "is_locked", lambda: False)():
            return
        win = getattr(self, "_code_window", None)
        if win is None:
            from .code_mode.code_window import CodeWindow

            win = CodeWindow(self, self.settings)
            self._code_window = win
            win.connect(
                "close-request", lambda *_: setattr(self, "_code_window", None)
                or False
            )
        win.present()

    def _on_preferences(self, *_args) -> None:
        if not self._main_window or not self.settings:
            return
        # Preferences (incl. the Security page that can disable App Lock) must
        # never be reachable while locked — Ctrl+, is a global accelerator,
        # not gated by which content is visible.
        if self._main_window.is_locked():
            return
        # Imported lazily — preferences pulls in every page's dependencies
        # (model catalog, themes tables, tools) and isn't needed at startup.
        from .preferences import PreferencesDialog
        dlg = PreferencesDialog(self, self.settings, self._main_window)
        dlg.present()

    def _on_about(self, *_args) -> None:
        about = Adw.AboutDialog(
            application_name=APP_NAME,
            application_icon=APP_ID,
            developer_name="Jegly",
            version=__version__,
            comments=(
                "The private, on-device AI workbench for Linux.\n\n"
                "Chat with text, images and audio · Box Code, a local "
                "coding agent · Image Studio for generation, inpainting "
                "and upscaling · notebooks and document Q&A · voice "
                "conversations · 52 themes with Glass modes.\n\n"
                "LiteRT-LM first (.litertlm/.task), with a bundled "
                "llama.cpp engine for GGUF models — Gemma, Qwen, Phi, "
                "Llama and more. Kernel-sandboxed inference, no account, "
                "no telemetry."
            ),
            release_notes_version=__version__,
            release_notes=(
                "<p>Box 0.4.0 — the workbench release:</p>"
                "<ul>"
                "<li>Box Code: a local coding agent (LiteRT + GGUF) with a "
                "sandboxed shell, sessions, and permission modes</li>"
                "<li>Image Studio: Z-Image and FLUX.2-klein at any "
                "resolution, inpainting, hires fix, LoRA, live previews</li>"
                "<li>Nav rail, Glass and Liquid Glass modes, 52 themes, "
                "header fonts, syntax-highlighted code</li>"
                "<li>GGUF engine with a 16-model hub and vocab self-heal; "
                "App Lock now covers every window</li>"
                "</ul>"
            ),
            website="https://www.jegly.xyz",
            issue_url="https://github.com/jegly",
            license_type=Gtk.License.CUSTOM,
            license=(
                "Proprietary software. © 2026 Jegly. All rights reserved.\n"
                "Source code is not published."
            ),
            copyright="© 2026 Jegly. All rights reserved.",
        )
        if self._main_window:
            about.present(self._main_window)

    def _on_quit(self, *_args) -> None:
        self.quit()

    def _on_new_chat(self, *_args) -> None:
        if self._main_window:
            self._main_window.start_new_chat()
