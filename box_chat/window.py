"""Main window — split view, engine events, TTS, voice recording, attachments."""
from __future__ import annotations

import logging
import shutil
import threading
import time
from copy import copy
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, GObject, Gio, Gtk  # noqa: E402

from . import APP_NAME
from . import audit
from .chat_view import ChatView
from .config import Settings, VOICE_DIR, MODELS_DIR
from .database import Database, Message
from .kb_panel import KbPanel
from .notebook_view import NotebookView
from .notebooks_list import NotebooksList
from .rag import RagController
from .engine import (
    EngineManager, Event, EvtComplete, EvtError, EvtLoading, EvtReady,
    EvtStopped, EvtToken,
)
from .live_mode import LiveModeController, LiveState
from .live_panel import LivePanel
from .permissions import BoxToolEventHandler, Decision, PermissionGate
from .sidebar import Sidebar
from .tools import call_map_for_callables, callables_for_tool_ids
from .tts import PiperTTS, is_ready as tts_is_ready
from .voice_recorder import VoiceRecorder

log = logging.getLogger(__name__)

# Supported attachment types
_TEXT_EXTS  = {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".xml",
               ".yaml", ".yml", ".toml", ".html", ".css", ".sh", ".c", ".cpp",
               ".h", ".rs", ".go", ".java", ".kt", ".swift"}
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
_PDF_EXT    = ".pdf"
# Phase 3b: extensions that go through the image-RAG (caption-then-embed) flow.
_IMAGE_RAG_EXTS = _IMAGE_EXTS


def _format_tool_call_for_prompt(
    tool_id: str, fn_name: str, args: dict, risky: bool
) -> str:
    """Build the body text of the permission dialog.

    Shows the tool id, the function name, and the arguments in compact form.
    Risky calls get an extra warning line so the user reads more carefully.
    """
    head = f"The assistant wants to call:\n\n    {fn_name}("
    parts = []
    for k, v in (args or {}).items():
        s = repr(v)
        if len(s) > 80:
            s = s[:77] + "…"
        parts.append(f"{k}={s}")
    args_str = ", ".join(parts)
    if len(args_str) > 200:
        args_str = args_str[:197] + "…"
    body = head + args_str + f")\n\nTool: {tool_id}"
    if risky:
        body += "\n\nThis call can MODIFY or DELETE files."
    return body


class MainWindow(Adw.ApplicationWindow):
    def __init__(
        self,
        application: Adw.Application,
        settings: Settings,
        db: Database,
        engine: EngineManager,
    ) -> None:
        super().__init__(application=application, title=APP_NAME)
        self._settings = settings
        self._db = db
        self._engine = engine
        self._current_conv_id: int | None = None
        self._toast_overlay: Adw.ToastOverlay
        self._model_status_label: Gtk.Label
        self._gen_start: float | None = None
        self._gen_tokens: int = 0

        self._tts = PiperTTS()
        self._tts.set_gain_provider(lambda: self._settings.tts_volume)
        self._recorder = VoiceRecorder()
        self._rag = RagController(settings, db)
        # Running estimate of LLM context usage (system + history); updated on
        # message add and conversation switch. Current composer text is added
        # on top in _refresh_context_bar so user sees usage update as they type.
        self._ctx_used_base: int = 0
        self._tts_speak_gen = 0
        self._tts_streaming = False
        # Set when the user presses the TTS stop button. EvtComplete checks
        # this before falling back to a one-shot _on_speak(text) — otherwise
        # stopping mid-stream would re-launch the whole reply 10-15s later
        # when generation finishes. Cleared on each new _on_user_send.
        self._tts_user_stopped: bool = False
        self._model_loading = False
        self._model_menu = Gio.Menu()
        # Hits returned by the last RAG retrieval — consumed when the
        # assistant message is committed (EvtComplete) so we can persist
        # which sources were used.
        self._pending_context_hits: list = []
        # Background-generation tracking. The engine has one queue and one
        # in-flight generation; we record which conversation owns it so
        # tokens routed back can be persisted to the originating chat
        # even if the user switches away mid-stream. Cleared on the
        # terminal event (Complete/Stopped/Error).
        self._active_gen_conv_id: int | None = None
        self._active_gen_text: str = ""
        self._active_gen_context: list[dict] = []
        self._active_gen_tool_calls: list[tuple] = []
        # Latest agent-progress (current, max). Persists across chat
        # switches so the pill re-shows when returning to the gen's conv.
        self._agent_progress_state: tuple[int, int | None] = (0, None)
        # File-audit state (chunked map-reduce; see _maybe_start_audit). Runs
        # as its own engine command, not a normal generation.
        self._auditing: bool = False
        self._audit_conv_id: int | None = None
        self._audit_name: str = ""
        self._audit_start: float = 0.0
        # Phase 4 — permission gate is owned by the window so the dialog
        # callback has access to the active GTK window. The engine worker
        # thread blocks inside gate.decide() until the dialog answers.
        self._perm_gate = PermissionGate(
            self._settings, self._ask_tool_permission_on_main
        )
        # Snapshot of the SDK-exposed callables at last load_model. We use
        # this to skip reloads when prefs toggles end up producing the same
        # tools list (e.g. flip a switch on then back off). Conversation
        # rebuilds appear to leak ~600 MB each in the SDK, so avoiding
        # unnecessary ones is load-bearing.
        self._tools_fingerprint: frozenset[str] = frozenset()
        # The live BoxToolEventHandler for the current conversation (or None
        # when no tools are active). Held so the window can reset the agent
        # iteration counter before each send.
        self._tool_handler = None
        # True while ``_sync_tools_popover`` is mass-setting widget values
        # so we don't echo each set into a real reload trigger.
        self._syncing_popover: bool = False

        self.set_default_size(settings.window_width, settings.window_height)
        self.set_resizable(True)
        self._install_window_actions()
        self._build_ui()
        self._update_model_menu()

        if self._settings.model_path and Path(self._settings.model_path).is_file():
            self._reload_engine_for_current_conv(force=True)
        self.refresh_tools_badge()

        convs = self._db.list_conversations()
        if convs:
            self._sidebar.refresh(select_id=convs[0].id)
            self._open_conversation(convs[0].id)
        else:
            self.start_new_chat()

        # Initial context-bar state.
        self.refresh_context_bar()

    # ──────────────────────────────────────────────────────────────────────
    def rebuild_nav_rail(self) -> None:
        """(Re)assemble the window around the nav rail per Settings.

        ``nav_position``: left | right | top | bottom. ``nav_labels``:
        show the item name with the icon. Called at build time and again
        by Preferences when either option changes — applies instantly.
        """
        pos = getattr(self._settings, "nav_position", "left")
        if pos not in ("left", "right", "top", "bottom"):
            pos = "left"
        horizontal_rail = pos in ("top", "bottom")

        outer = self._nav_outer
        while (child := outer.get_first_child()) is not None:
            outer.remove(child)
        outer.set_orientation(
            Gtk.Orientation.VERTICAL if horizontal_rail
            else Gtk.Orientation.HORIZONTAL
        )
        rail = self._build_nav_rail(horizontal_rail)
        sep = Gtk.Separator(orientation=(
            Gtk.Orientation.HORIZONTAL if horizontal_rail
            else Gtk.Orientation.VERTICAL
        ))
        if pos in ("left", "top"):
            outer.append(rail)
            outer.append(sep)
            outer.append(self._toast_overlay)
        else:
            outer.append(self._toast_overlay)
            outer.append(sep)
            outer.append(rail)
        # Re-apply the active highlight lost in the rebuild.
        try:
            active = self._sidebar_stack.get_visible_child_name() or "chats"
        except AttributeError:  # first build — sidebar not constructed yet
            active = "chats"
        self._sync_rail_highlight(active)

    def _build_nav_rail(self, horizontal: bool) -> Gtk.Box:
        """The icon (± label) rail, vertical or horizontal."""
        labels = bool(getattr(self._settings, "nav_labels", False))
        rail = Gtk.Box(
            orientation=(Gtk.Orientation.HORIZONTAL if horizontal
                         else Gtk.Orientation.VERTICAL),
            spacing=4,
            margin_top=6, margin_bottom=6, margin_start=6, margin_end=6,
        )
        rail.add_css_class("nav-rail")

        def add(icon: str, name: str, cb) -> Gtk.Button:
            btn = Gtk.Button()
            if labels:
                inner = Gtk.Box(
                    orientation=Gtk.Orientation.VERTICAL, spacing=2,
                    halign=Gtk.Align.CENTER,
                )
                inner.append(Gtk.Image.new_from_icon_name(icon))
                lbl = Gtk.Label(label=name)
                lbl.add_css_class("caption")
                inner.append(lbl)
                btn.set_child(inner)
            else:
                btn.set_icon_name(icon)
            btn.add_css_class("flat")
            btn.set_tooltip_text(name)
            btn.connect("clicked", cb)
            rail.append(btn)
            return btn

        self._rail_buttons = {}
        self._rail_buttons["chats"] = add(
            "chat-message-new-symbolic", "Chats",
            lambda *_: self._nav_show_sidebar_tab("chats"))
        self._rail_buttons["notebooks"] = add(
            "accessories-dictionary-symbolic", "Notebooks",
            lambda *_: self._nav_show_sidebar_tab("notebooks"))
        add("image-x-generic-symbolic", "Image Tools",
            lambda *_: self.get_application().activate_action(
                "image-tools", None))
        add("utilities-terminal-symbolic", "Box Code",
            lambda *_: self.get_application().activate_action(
                "code-mode", None))
        spacer = Gtk.Box(vexpand=not horizontal, hexpand=horizontal)
        rail.append(spacer)
        add("emblem-system-symbolic", "Preferences",
            lambda *_: self.get_application().activate_action(
                "preferences", None))
        return rail

    def _sync_rail_highlight(self, active: str) -> None:
        """Fill the active section's rail icon with the accent colour."""
        for name, btn in getattr(self, "_rail_buttons", {}).items():
            if name == active:
                btn.add_css_class("suggested-action")
                btn.remove_css_class("flat")
            else:
                btn.remove_css_class("suggested-action")
                btn.add_css_class("flat")

    def _nav_show_sidebar_tab(self, name: str) -> None:
        """Rail click: reveal the sidebar (if hidden) and switch its tab."""
        if self.is_locked():
            return
        try:
            self._sidebar_toolbar.set_visible(True)
            self._sidebar_stack.set_visible_child_name(name)
            if name == "chats":
                self._show_chat_pane()
        except Exception:  # noqa: BLE001 — nav must never crash the window
            log.exception("nav rail switch failed")

    def _build_ui(self) -> None:
        self._toast_overlay = Adw.ToastOverlay()

        # Nav rail: one place to reach every part of the app (Chats,
        # Notebooks, Image Tools, Box Code, Preferences). Repositionable
        # (left/right/top/bottom) with optional labels — ATK-style, driven
        # by Settings.nav_position / nav_labels and relayoutable live.
        self._nav_outer = Gtk.Box()
        self._toast_overlay.set_hexpand(True)
        self._toast_overlay.set_vexpand(True)
        self.rebuild_nav_rail()
        self.set_content(self._nav_outer)

        # Top-level split is a Gtk.Paned so the divider is user-draggable
        # (was Adw.NavigationSplitView — no resize handle). Width + visible
        # state are persisted in Settings.
        self._sidebar_split = Gtk.Paned(
            orientation=Gtk.Orientation.HORIZONTAL,
            wide_handle=True,
            resize_start_child=False,  # content takes new space on window resize
            shrink_start_child=False,
            shrink_end_child=False,
        )
        self._toast_overlay.set_child(self._sidebar_split)

        # Sidebar: Chats / Notebooks tab switcher
        sidebar_toolbar = Adw.ToolbarView()
        sidebar_header = Adw.HeaderBar()
        sidebar_header.set_show_end_title_buttons(False)

        self._sidebar = Sidebar(
            db=self._db,
            on_select=self._open_conversation,
            on_create=self.start_new_chat,
        )
        self._notebooks_list = NotebooksList(
            list_notebooks=self._rag.list_notebooks,
            notebook_file_count=lambda nb_id: len(self._rag.notebook_sources(nb_id)),
            on_select=self._open_notebook,
            on_create=self._on_create_notebook,
        )

        self._sidebar_stack = Adw.ViewStack()
        chats_page = self._sidebar_stack.add_titled(
            self._sidebar, "chats", "Chats")
        chats_page.set_icon_name("user-available-symbolic")
        nbs_page = self._sidebar_stack.add_titled(
            self._notebooks_list, "notebooks", "Notebooks")
        nbs_page.set_icon_name("accessories-dictionary-symbolic")

        # The nav rail is the one place that switches sections — the old
        # Chats/Notebooks tab switcher here duplicated it. Plain title now.
        self._sidebar_title = Adw.WindowTitle(title="Chats", subtitle="")
        sidebar_header.set_title_widget(self._sidebar_title)
        sidebar_toolbar.add_top_bar(sidebar_header)
        sidebar_toolbar.set_content(self._sidebar_stack)
        # Floor sidebar at 200 px so the user can't drag it into uselessness;
        # the toggle button is the right way to hide it entirely.
        sidebar_toolbar.set_size_request(200, -1)

        def _on_sidebar_tab_changed(*_) -> None:
            name = self._sidebar_stack.get_visible_child_name()
            self._sidebar_title.set_title(
                "Notebooks" if name == "notebooks" else "Chats"
            )
            self._sync_rail_highlight(name)
            if (name == "chats"
                    and self._main_stack.get_visible_child_name() == "notebook"):
                self._show_chat_pane()

        self._sidebar_stack.connect("notify::visible-child", _on_sidebar_tab_changed)
        self._sync_rail_highlight("chats")  # initial child fires no notify
        self._sidebar_toolbar = sidebar_toolbar
        self._sidebar_split.set_start_child(sidebar_toolbar)

        # Content
        content_toolbar = Adw.ToolbarView()
        content_header = Adw.HeaderBar()
        # Custom macOS-style traffic-light buttons — we hide the native window
        # controls and pack our own so we have full control over size + shape.
        content_header.set_show_end_title_buttons(False)
        content_header.set_show_start_title_buttons(False)
        traffic = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
                          valign=Gtk.Align.CENTER, halign=Gtk.Align.CENTER)
        traffic.add_css_class("traffic-lights")
        for css_cls, action in (
            ("traffic-min", lambda *_: self.minimize()),
            ("traffic-max", lambda *_: (
                self.unmaximize() if self.is_maximized() else self.maximize()
            )),
            ("traffic-close", lambda *_: self.close()),
        ):
            btn = Gtk.Button()
            btn.add_css_class("traffic-light")
            btn.add_css_class(css_cls)
            btn.set_size_request(13, 13)
            btn.connect("clicked", action)
            traffic.append(btn)
        content_header.pack_end(traffic)

        # Sidebar toggle — hides/shows the left Chats/Notebooks panel.
        # The user can also drag the panel divider to resize.
        self._sidebar_toggle_btn = Gtk.ToggleButton(
            icon_name="sidebar-show-symbolic",
            tooltip_text="Hide chat list",
            active=bool(self._settings.sidebar_visible),
        )
        self._sidebar_toggle_btn.add_css_class("flat")
        self._sidebar_toggle_btn.connect("toggled", self._on_sidebar_toggle)
        content_header.pack_start(self._sidebar_toggle_btn)

        # KB panel toggle — packed at the start of the content header. Reveals
        # the right-side knowledge base pane.
        self._kb_toggle_btn = Gtk.ToggleButton(
            icon_name="folder-documents-symbolic",
            tooltip_text="Knowledge Base",
        )
        self._kb_toggle_btn.add_css_class("flat")
        self._kb_toggle_btn.connect("toggled", self._on_kb_toggle)
        content_header.pack_start(self._kb_toggle_btn)

        # Tools popover button — shown only when at least one tool is
        # effectively enabled for the current chat. Click opens a quick-
        # toggle popover so the user can flip web search / filesystem /
        # the per-chat override without diving into Preferences.
        self._tools_btn = Gtk.MenuButton(
            icon_name="applications-utilities-symbolic",
            tooltip_text="Tools — quick toggles",
        )
        self._tools_btn.add_css_class("flat")
        self._tools_btn.set_popover(self._build_tools_popover())
        # Always visible (so user can enable a tool from here). Active
        # state is signalled via a CSS class set by refresh_tools_badge.
        content_header.pack_start(self._tools_btn)

        # Agent progress pill — shows "Agent 3/6" while a tool-chain is
        # mid-generation. Hidden unless a capped agent run is in flight.
        self._agent_pill = Gtk.Label(label="", visible=False)
        self._agent_pill.add_css_class("agent-pill")
        content_header.pack_start(self._agent_pill)

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2,
                            halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
        self._title_label = Gtk.Label(label=APP_NAME)
        self._title_label.add_css_class("title")

        # Model picker — clicking the subtitle drops a menu of recent models.
        picker_child = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=3)
        self._model_status_label = Gtk.Label(label="No model loaded")
        self._model_status_label.add_css_class("dim-label")
        self._model_status_label.add_css_class("caption")
        picker_child.append(self._model_status_label)
        picker_child.append(Gtk.Image(icon_name="pan-down-symbolic", pixel_size=10))
        self._model_picker_btn = Gtk.MenuButton(menu_model=self._model_menu)
        self._model_picker_btn.add_css_class("flat")
        self._model_picker_btn.set_child(picker_child)
        # Rebuild the list every time the dropdown opens so models downloaded
        # or imported (even without a Settings → Use) always show up.
        self._model_picker_btn.set_create_popup_func(
            lambda *_: self._update_model_menu()
        )

        title_box.append(self._title_label)
        title_box.append(self._model_picker_btn)
        content_header.set_title_widget(title_box)

        menu = Gio.Menu()
        menu.append("New Chat", "app.new-chat")
        menu.append("Image Tools…", "app.image-tools")
        menu.append("Box Code…", "app.code-mode")
        menu.append("Lock Now", "app.lock")
        menu.append("Preferences", "app.preferences")
        menu.append("About Box", "app.about")
        menu.append("Quit", "app.quit")
        menu_btn = Gtk.MenuButton(icon_name="open-menu-symbolic", menu_model=menu)
        content_header.pack_end(menu_btn)
        content_toolbar.add_top_bar(content_header)

        # Persistent banner for RAG indexing progress (hidden by default).
        self._rag_banner = Adw.Banner(title="", revealed=False)
        content_toolbar.add_top_bar(self._rag_banner)

        self._chat_view = ChatView(
            on_send=self._on_user_send,
            on_stop=self._on_user_stop,
            on_attach=self._on_attach_clicked,
            on_mic_toggle=self._on_mic_toggle,
            on_speak=self._on_speak if self._settings.enable_tts else None,
            on_stop_tts=self._on_stop_tts_btn,
            on_camera=self._on_camera_clicked,
            on_live_toggle=self._on_live_toggle,
            on_agent_toggle=self._on_agent_toggle,
            settings=self._settings,
            on_voice_live_toggle=self._on_voice_live_toggle,
            on_websearch_toggle=self._on_websearch_toggle,
            on_fs_toggle=self._on_fs_toggle,
            on_tts_toggle=self._on_tts_toggle,
            on_tts_volume_changed=self._on_tts_volume_changed,
            on_memory_toggle=self._on_memory_toggle,
            on_memory_save=self._on_memory_save,
            on_save_memory=self._on_save_bubble_as_memory,
            initial_tts_enabled=bool(self._settings.tts_auto_speak),
            initial_tts_volume=float(self._settings.tts_volume),
            initial_memory_enabled=bool(self._settings.memory_enabled),
        )
        self.refresh_camera_button()
        self.refresh_agent_button()
        self.refresh_websearch_button()
        self.refresh_fs_button()
        self.refresh_tts_button()
        self.refresh_voice_live_button()
        self.refresh_memory_button()

        # Live-mode panel (Phase 4.5 Tier 3) — wrapped in a Revealer
        # that slides it in/out above the chat scroller while the
        # controller is active.
        self._live_panel = LivePanel(
            on_end=lambda: self._on_live_toggle(False),
            on_talk_start=lambda: self._live_controller.begin_talk(),
            on_talk_end=lambda: self._live_controller.end_talk(),
        )
        self._live_revealer = Gtk.Revealer(reveal_child=False)
        self._live_revealer.set_transition_type(Gtk.RevealerTransitionType.SLIDE_DOWN)
        self._live_revealer.set_child(self._live_panel)
        content_toolbar.add_top_bar(self._live_revealer)

        self._live_camera_session = None
        self._live_controller = LiveModeController(
            engine=self._engine,
            on_state_change=self._on_live_state,
            on_user_audio_path=self._on_live_user_audio,
            on_assistant_text=self._on_live_assistant_text,
            on_error=self._on_live_error,
            send_audio_to_engine=self._live_send_to_engine,
            cancel_generation=self._engine.stop,
        )

        # Notebook detail page — used when the user opens a notebook from
        # the sidebar. Sits next to the ChatView in a Gtk.Stack.
        self._notebook_view = NotebookView(
            list_sources=self._rag.notebook_sources,
            on_add_files=self._on_notebook_add_files,
            on_remove_source=self._on_notebook_remove_source,
            on_rename=self._on_rename_notebook_clicked,
            on_delete=self._on_delete_notebook_clicked,
            on_auto_attach_changed=self._on_notebook_auto_attach_changed,
            on_back=self._show_chat_pane,
        )
        self._main_stack = Gtk.Stack(
            transition_type=Gtk.StackTransitionType.CROSSFADE,
        )
        self._main_stack.add_named(self._chat_view, "chat")
        self._main_stack.add_named(self._notebook_view, "notebook")

        # Knowledge Base panel — wraps the chat/notebook stack in an
        # OverlaySplitView so the KB pane slides in from the right when
        # toggled. KB panel is only meaningful for chats; we hide the
        # toggle button while a notebook is open.
        self._kb_panel = KbPanel(
            list_sources_for=self._rag.sources,
            on_add_files=self._on_kb_add_files,
            on_remove_source=self._on_kb_remove_source,
            on_rag_override_changed=self._on_kb_rag_override_changed,
            list_attached_notebooks=self._rag.attached_notebooks,
            list_all_notebooks=self._rag.list_notebooks,
            on_detach_notebook=self._on_kb_detach_notebook,
        )
        self._kb_split = Adw.OverlaySplitView(
            sidebar=self._kb_panel,
            content=self._main_stack,
            sidebar_position=Gtk.PackType.END,
            show_sidebar=False,
            min_sidebar_width=240,
            max_sidebar_width=380,
            sidebar_width_fraction=0.28,
        )
        self._kb_split.connect("notify::show-sidebar", self._on_kb_split_notify)

        content_toolbar.set_content(self._kb_split)
        self._sidebar_split.set_end_child(content_toolbar)
        # Restore last-used sidebar geometry. Position has to be applied
        # after the children are attached (GtkPaned defers layout otherwise).
        initial_pos = max(200, int(self._settings.sidebar_width or 280))
        self._sidebar_split.set_position(initial_pos)
        self._apply_sidebar_visible(self._settings.sidebar_visible)
        # Persist the user's drag — debounce via a single idle callback to
        # avoid hammering the JSON file on every pixel.
        self._sidebar_save_pending = False
        self._sidebar_split.connect(
            "notify::position", self._on_sidebar_position_changed,
        )

    def _install_window_actions(self) -> None:
        for name, cb in (("rename-conv", self._on_rename_conv),
                         ("delete-conv", self._on_delete_conv),
                         ("rename-notebook", self._on_rename_notebook_action),
                         ("delete-notebook", self._on_delete_notebook_action),
                         ("detach-notebook", self._on_detach_notebook_action)):
            act = Gio.SimpleAction.new(name, GLib.VariantType.new("i"))
            act.connect("activate", cb)
            self.add_action(act)

        pick_act = Gio.SimpleAction.new("pick-model", GLib.VariantType.new("s"))
        pick_act.connect("activate", self._on_pick_model_action)
        self.add_action(pick_act)

        browse_act = Gio.SimpleAction.new("browse-model", None)
        browse_act.connect("activate", self._on_browse_model_action)
        self.add_action(browse_act)

        attach_act = Gio.SimpleAction.new("attach-notebook", GLib.VariantType.new("i"))
        attach_act.connect("activate", self._on_attach_notebook_action)
        self.add_action(attach_act)

    # ── Conversation lifecycle ─────────────────────────────────────────────

    def start_new_chat(self) -> None:
        model_name = Path(self._settings.model_path).name if self._settings.model_path else ""
        conv = self._db.create_conversation(title="New chat", model=model_name)
        # Auto-attach default notebooks if the master switch is on.
        if self._settings.rag_auto_attach_notebooks:
            for nb_id in self._db.list_auto_attach_notebook_ids():
                self._rag.attach_notebook(conv.id, nb_id)
        self._sidebar.refresh(select_id=conv.id)
        self._open_conversation(conv.id)
        self._chat_view.focus_input()

    def _open_conversation(self, conv_id: int) -> None:
        # Always switch to the chat pane (user may be viewing a notebook).
        self._show_chat_pane()
        if conv_id == self._current_conv_id:
            return
        self._current_conv_id = conv_id
        self._perm_gate.set_active_conversation(conv_id)
        messages = self._db.list_messages(conv_id)
        self._chat_view.load_messages(messages)
        # If a background generation belongs to this conv, re-attach its
        # streaming bubble pre-loaded with whatever text + tool-call cards
        # have already arrived. Without this the user would land in an
        # in-progress chat with no visible bubble.
        if self._active_gen_conv_id == conv_id:
            self._chat_view.restore_assistant_stream(
                self._active_gen_text,
                context=self._active_gen_context or None,
                tool_calls=list(self._active_gen_tool_calls),
            )
        # Engine reload: skip while a generation is in flight. Reloading
        # now would (a) queue a load_model behind the active send with a
        # stale history snapshot (the in-progress reply hasn't been
        # persisted yet), and (b) tear down the live SDK conv that the
        # gen is using. We reload at gen-end if _current_conv_id differs
        # from the gen's conv — see the EvtComplete/Stopped/Error
        # handlers in _handle_engine_event.
        if self._active_gen_conv_id is None:
            self._reload_engine_for_current_conv()
        self._recompute_context_base()
        self._refresh_kb_panel()
        self.refresh_tools_badge()
        self.refresh_agent_button()
        self.refresh_websearch_button()
        # Re-evaluate the agent pill: it should only show while the
        # generating conv is on-screen.
        cur, mx = self._agent_progress_state
        self._set_agent_progress(cur, mx)

    def _reload_engine_for_current_conv(self, force: bool = False) -> None:
        if not self._settings.model_path:
            return
        if not Path(self._settings.model_path).is_file():
            self._show_toast(f"Model file not found: {self._settings.model_path}")
            return

        # If using GPU or speculative decoding, pre-write a CPU-safe copy of
        # settings to disk. If the process segfaults (e.g. no GPU hardware),
        # the next launch reads CPU settings and opens cleanly. Only the
        # litert path loads in-process and can take Box down; GGUF models run
        # in a separate llama-server process, so their load can't segfault us.
        is_litert = not self._settings.model_path.lower().endswith(".gguf")
        risky = is_litert and (
            self._settings.backend != "cpu"
            or self._settings.enable_speculative_decoding
        )
        if risky:
            safe = copy(self._settings)
            safe.backend = "cpu"
            safe.enable_speculative_decoding = False
            safe.save()

        self._model_loading = True
        history: list[dict] = []
        if self._current_conv_id is not None:
            history = [
                {"role": m.role, "content": m.content}
                for m in self._db.list_messages(self._current_conv_id)
                if m.role in ("user", "assistant")
            ]
        # Agent mode gates tool exposure: agent ON → all enabled tools;
        # agent OFF → web search only (if its quick toggle is on).
        active_ids = self.tools_for_model(self._current_conv_id)
        tools_list = callables_for_tool_ids(self._settings, active_ids)
        call_map = call_map_for_callables(tools_list)
        # Agent mode: only meaningful when there are tools to chain. When on,
        # prepend the agent stanza to the system prompt and cap the per-send
        # tool-call count so a confused model can't loop forever.
        agent_on = bool(tools_list) and self.effective_agent_enabled(
            self._current_conv_id
        )
        system_prompt = self._settings.system_prompt
        if agent_on:
            system_prompt = (
                self._settings.resolved_agent_prompt() + "\n\n" + system_prompt
            )
        handler = (
            BoxToolEventHandler(
                self._perm_gate, call_map,
                on_tool_event=self._on_tool_event,
                max_iterations=(
                    self._settings.agent_max_iterations if agent_on else None
                ),
                on_progress=self._on_agent_progress if agent_on else None,
            )
            if tools_list else None
        )
        self._tool_handler = handler
        self._set_agent_progress(0, None)  # clear any stale pill on reload
        self._tools_fingerprint = frozenset(fn.__name__ for fn in tools_list)
        self._engine.load_model(
            path=self._settings.model_path,
            system_prompt=system_prompt,
            history=history,
            cb=self._engine_callback,
            backend=self._settings.backend,
            enable_speculative_decoding=self._settings.enable_speculative_decoding,
            enable_vision=self._settings.enable_vision,
            enable_audio=self._settings.enable_audio,
            temperature=self._settings.temperature,
            top_k=self._settings.top_k,
            top_p=self._settings.top_p,
            max_num_tokens=self._settings.max_context_tokens,
            tools=tools_list or None,
            tool_event_handler=handler,
            llama_settings=self._settings,
        )

    # ── Preference change hooks ────────────────────────────────────────────

    def on_model_changed(self, _path: str) -> None:
        self._update_model_menu()
        self._reload_engine_for_current_conv(force=True)

    # ── Tools popover (header quick-toggle menu) ──────────────────────────
    _PER_CHAT_OPTIONS = ("Follow global", "Always on", "Always off")
    _OPT_TO_OVR: dict[int, int | None] = {0: None, 1: 1, 2: 0}
    _OVR_TO_OPT: dict[int | None, int] = {None: 0, 1: 1, 0: 2}

    def _build_tools_popover(self) -> Gtk.Popover:
        pop = Gtk.Popover()
        outer = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6,
            margin_top=8, margin_bottom=8, margin_start=8, margin_end=8,
        )
        outer.set_size_request(280, -1)

        head = Gtk.Label(label="Tools for this chat", xalign=0.0)
        head.add_css_class("heading")
        outer.append(head)

        # Web search row — global on/off + per-chat override.
        self._pop_ws_switch = self._build_pop_global_switch(
            "Web search",
            "tool_web_search_enabled",
        )
        outer.append(self._pop_ws_switch[0])
        self._pop_ws_override = self._build_pop_override_combo(
            "web_search", "Per-chat",
        )
        outer.append(self._pop_ws_override[0])

        outer.append(Gtk.Separator())

        # Filesystem row.
        self._pop_fs_switch = self._build_pop_global_switch(
            "Filesystem",
            "tool_fs_enabled",
        )
        outer.append(self._pop_fs_switch[0])
        self._pop_fs_override = self._build_pop_override_combo(
            "filesystem", "Per-chat",
        )
        outer.append(self._pop_fs_override[0])

        outer.append(Gtk.Separator())

        prefs_btn = Gtk.Button(label="Open full Preferences…")
        prefs_btn.add_css_class("flat")
        prefs_btn.connect(
            "clicked",
            lambda *_: (
                self.get_application().activate_action("preferences", None),
                pop.popdown(),
            ),
        )
        outer.append(prefs_btn)

        pop.set_child(outer)
        pop.connect("show", lambda *_: self._sync_tools_popover())
        return pop

    def _build_pop_global_switch(
        self, label: str, settings_field: str,
    ) -> tuple[Gtk.Box, Gtk.Switch]:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        lbl = Gtk.Label(label=label, xalign=0.0, hexpand=True)
        sw = Gtk.Switch(valign=Gtk.Align.CENTER)
        sw.set_active(bool(getattr(self._settings, settings_field, False)))
        sw.connect(
            "notify::active",
            lambda s, _p: self._on_pop_global_changed(settings_field, s),
        )
        row.append(lbl)
        row.append(sw)
        return row, sw

    def _build_pop_override_combo(
        self, tool_id: str, label: str,
    ) -> tuple[Gtk.Box, Gtk.DropDown]:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
                      margin_start=12)
        lbl = Gtk.Label(label=label, xalign=0.0, hexpand=True)
        lbl.add_css_class("caption")
        lbl.add_css_class("dim-label")
        dd = Gtk.DropDown.new_from_strings(list(self._PER_CHAT_OPTIONS))
        dd.set_valign(Gtk.Align.CENTER)
        dd.connect(
            "notify::selected",
            lambda d, _p: self._on_pop_override_changed(tool_id, d),
        )
        row.append(lbl)
        row.append(dd)
        return row, dd

    def _on_pop_global_changed(self, settings_field: str, sw: Gtk.Switch) -> None:
        if getattr(self, "_syncing_popover", False):
            return
        target = sw.get_active()
        if bool(getattr(self._settings, settings_field, False)) == target:
            return
        setattr(self._settings, settings_field, target)
        self._settings.save()
        self._dismiss_tools_popover_and_reload()

    def _on_pop_override_changed(self, tool_id: str, dd: Gtk.DropDown) -> None:
        if getattr(self, "_syncing_popover", False):
            return
        if self._current_conv_id is None:
            return
        idx = dd.get_selected()
        override = self._OPT_TO_OVR.get(idx)
        try:
            self._db.set_tool_override(self._current_conv_id, tool_id, override)
        except ValueError:
            return
        self._dismiss_tools_popover_and_reload()

    def _dismiss_tools_popover_and_reload(self) -> None:
        """Close the popover first, then queue the reload on the next idle
        cycle. Reloading synchronously inside a popover-change signal keeps
        the popover modal until the SDK finishes a fresh prefill, which
        looks like an app hang on slow CPUs."""
        if hasattr(self, "_tools_btn"):
            pop = self._tools_btn.get_popover()
            if pop is not None:
                pop.popdown()

        def _later() -> bool:
            self.refresh_tools_for_current_conv()
            self.refresh_tools_badge()
            return False
        GLib.idle_add(_later)

    def _sync_tools_popover(self) -> None:
        """Re-read state into the popover widgets each time it opens. We
        flip a ``_syncing_popover`` flag so the dropdown/switch change
        handlers don't treat the sync as a user-initiated change and
        trigger a needless engine reload."""
        self._syncing_popover = True
        try:
            for sw, field in (
                (self._pop_ws_switch[1], "tool_web_search_enabled"),
                (self._pop_fs_switch[1], "tool_fs_enabled"),
            ):
                sw.set_active(bool(getattr(self._settings, field, False)))

            conv = (
                self._db.get_conversation(self._current_conv_id)
                if self._current_conv_id is not None else None
            )
            for dd, tool_id, field in (
                (self._pop_ws_override[1], "web_search", "tool_web_override"),
                (self._pop_fs_override[1], "filesystem", "tool_fs_override"),
            ):
                ovr = getattr(conv, field, None) if conv is not None else None
                dd.set_selected(self._OVR_TO_OPT.get(ovr, 0))
                dd.set_sensitive(conv is not None)
        finally:
            self._syncing_popover = False

    def refresh_tools_badge(self) -> None:
        """Mark the header tools button as active iff any tool is
        effectively enabled for the current chat. The button itself
        is always visible so users can flip a tool on from the popover.
        """
        if not hasattr(self, "_tools_btn"):
            return
        active = bool(self.effective_enabled_tool_ids(self._current_conv_id))
        if active and not getattr(self, "_backend_supports_tools", True):
            self._tools_btn.remove_css_class("tools-btn-active")
            self._tools_btn.set_tooltip_text(
                "Tools enabled, but the llama.cpp engine doesn't support "
                "tool calling yet — they won't be used by this model"
            )
        elif active:
            self._tools_btn.add_css_class("tools-btn-active")
            self._tools_btn.set_tooltip_text(
                f"Tools active: {', '.join(self.effective_enabled_tool_ids(self._current_conv_id))}"
            )
        else:
            self._tools_btn.remove_css_class("tools-btn-active")
            self._tools_btn.set_tooltip_text("Tools — quick toggles (none active)")

    def refresh_tools_for_current_conv(self) -> None:
        """Preferences → Tools toggle hook. Reloads the conversation ONLY
        when the resulting set of exposed callables actually changed —
        otherwise switch-flipping in the Tools tab would queue a stream
        of redundant reloads, each leaking SDK memory."""
        if not self._settings.model_path:
            return
        active_ids = self.tools_for_model(self._current_conv_id)
        new_fp = frozenset(
            fn.__name__ for fn in callables_for_tool_ids(self._settings, active_ids)
        )
        if new_fp == self._tools_fingerprint:
            self._refresh_composer_toggles()
            return
        self._reload_engine_for_current_conv(force=True)
        self._refresh_composer_toggles()

    def refresh_agent_for_current_conv(self) -> None:
        """Preferences → Agent toggle hook. Agent mode changes the system
        prompt and the iteration cap but NOT the tools fingerprint, so the
        fingerprint short-circuit in ``refresh_tools_for_current_conv``
        would skip the needed reload — force one here."""
        if self._settings.model_path:
            self._reload_engine_for_current_conv(force=True)
        self._refresh_composer_toggles()

    # ── Webcam capture (Phase 4.5) ────────────────────────────────────────
    def refresh_camera_button(self) -> None:
        """Toggle the composer 📷 button based on settings + backend
        availability. Called on init, when Preferences toggles
        ``webcam_enabled``, and when the active conversation changes."""
        visible = False
        if getattr(self._settings, "webcam_enabled", False):
            try:
                from . import webcam
                visible = webcam.AVAILABLE
            except Exception:
                log.exception("webcam backend import failed")
                visible = False
        self._chat_view.set_camera_visible(visible)

    def _on_camera_clicked(self) -> None:
        """Open the capture modal. Handler is invoked from the composer
        camera button; the dialog handles everything from there and
        calls back into ``_on_camera_captured`` on success."""
        from .capture_dialog import CaptureDialog
        from . import webcam
        ok, reason = webcam.probe()
        if not ok:
            self._show_toast(f"Camera unavailable: {reason}")
            return
        dlg = CaptureDialog(
            self._settings,
            on_captured=self._on_camera_captured,
        )
        dlg.present(self)

    def _on_camera_captured(self, path: str) -> None:
        """A frame was just written to ``path``. Drop it into the
        composer as a regular image attachment — the existing send path
        will then ship it to the model on the user's next send."""
        from pathlib import Path as _P
        p = _P(path)
        att = {"type": "image", "path": str(p), "name": p.name}
        self._chat_view.add_attachment(att)

    # ── Live mode (Phase 4.5 Tier 3) ──────────────────────────────────────
    def _on_live_toggle(self, active: bool) -> None:
        """Composer toggle / panel End button. Starts or stops the
        :class:`LiveModeController` and reveals / hides the panel."""
        if active:
            self._start_live_mode()
        else:
            self._stop_live_mode()

    def _on_voice_live_toggle(self, active: bool) -> None:
        """Voice-only sibling of Live mode — no camera session. Same
        listening/thinking/speaking pill flow, just blank preview tile."""
        if active:
            self._start_live_mode(with_camera=False)
        else:
            self._stop_live_mode()

    def refresh_voice_live_button(self) -> None:
        """Visible whenever the audio backend is enabled — that's all
        voice-only mode needs."""
        visible = bool(self._settings.enable_audio)
        self._chat_view.set_voice_live_visible(visible)

    def _start_live_mode(self, *, with_camera: bool = True) -> None:
        from . import audio_stream, webcam
        if not self._settings.enable_audio:
            self._show_toast(
                "Audio is off — turn it on in Preferences → Multimodal."
            )
            self._chat_view.set_live_active(False)
            self._chat_view.set_voice_live_active(False)
            return
        if not audio_stream.is_available():
            self._show_toast(
                "Microphone unavailable — install PortAudio "
                "(sudo apt install libportaudio2), then restart Box."
            )
            self._chat_view.set_live_active(False)
            self._chat_view.set_voice_live_active(False)
            return
        session = None
        if with_camera:
            if not getattr(self._settings, "webcam_enabled", False):
                self._show_toast("Enable the camera in Preferences first.")
                self._chat_view.set_live_active(False)
                return
            if not self._settings.enable_vision:
                self._show_toast(
                    "Vision is off — turn it on in Preferences → Multimodal."
                )
                self._chat_view.set_live_active(False)
                return
            ok, reason = webcam.probe()
            if not ok:
                self._show_toast(f"Camera unavailable: {reason}")
                self._chat_view.set_live_active(False)
                return
            try:
                session = webcam.open_session()
                session.start()
            except Exception as e:  # noqa: BLE001
                log.exception("live mode: camera open failed")
                self._show_toast(f"Camera failed: {e}")
                self._chat_view.set_live_active(False)
                return

            paintable = session.paintable() if hasattr(session, "paintable") else None
            if paintable is not None:
                self._live_panel.bind_preview(paintable)
            elif hasattr(session, "set_preview_callback"):
                session.set_preview_callback(self._live_panel.push_preview_texture)

        ptt = bool(getattr(self._settings, "live_push_to_talk", False))
        self._live_camera_session = session
        self._live_panel.set_preview_visible(session is not None)
        self._live_panel.set_ptt_visible(ptt)
        try:
            self._live_controller.start(camera_session=session, push_to_talk=ptt)
        except Exception as e:  # noqa: BLE001
            log.exception("live mode: audio capture failed to start")
            self._show_toast(f"Couldn't start the microphone: {e}")
            if session is not None:
                try:
                    session.close()
                except Exception:
                    log.exception("live mode: camera close failed")
                self._live_camera_session = None
            self._chat_view.set_live_active(False)
            self._chat_view.set_voice_live_active(False)
            return
        self._live_revealer.set_reveal_child(True)
        # Sync the right toggle button without echoing back.
        if with_camera:
            self._chat_view.set_live_active(True)
        else:
            self._chat_view.set_voice_live_active(True)

    def _stop_live_mode(self) -> None:
        self._live_controller.stop()
        self._live_revealer.set_reveal_child(False)
        self._chat_view.set_live_active(False)
        self._chat_view.set_voice_live_active(False)
        if self._live_camera_session is not None:
            try:
                self._live_camera_session.close()
            except Exception:
                log.exception("live mode: camera close failed")
            self._live_camera_session = None

    def _on_live_state(self, state: LiveState) -> None:
        def _ui() -> bool:
            self._live_panel.set_state(state)
            return False
        GLib.idle_add(_ui)

    def _on_live_user_audio(self, wav_path: str) -> None:
        """The controller just wrote the user's spoken turn to a WAV.
        Stub for now — the engine-send path in task 26 will pick this
        up and ship it. We could also drop a bubble here showing
        'You said…' once we have transcription."""
        log.info("live mode: user audio at %s", wav_path)

    def _on_live_assistant_text(self, text: str) -> None:
        """Drop the assistant's response into the chat AND kick TTS so
        the user actually hears the reply. If TTS isn't ready, fall
        straight back to LISTENING so we don't get stuck."""
        def _ui() -> bool:
            self._chat_view.add_assistant_message(text)
            voice = self._settings.tts_voice or "en_US-lessac-medium"
            if self._settings.enable_tts and tts_is_ready(voice):
                # Mute VAD during the grace window, speak, re-arm on done.
                self._live_controller.notify_tts_started()
                self._tts.speak(
                    text,
                    voice_id=voice,
                    on_done=self._live_controller.notify_tts_finished,
                )
            else:
                # No TTS available — head back to listening immediately
                # so the conversation can continue.
                self._live_controller.notify_tts_finished()
            return False
        GLib.idle_add(_ui)

    def _on_live_error(self, message: str) -> None:
        def _ui() -> bool:
            self._show_toast(f"Live mode error: {message}")
            return False
        GLib.idle_add(_ui)

    def _live_send_to_engine(
        self, wav_path: str, jpeg_path: str | None,
    ) -> None:
        """Build the multimodal turn (audio + optional webcam frame +
        steering text) and ship it to the engine with a dedicated
        callback that pipes EvtComplete back to the live controller."""
        from pathlib import Path as _P
        atts: list[dict] = [
            {"type": "audio", "path": wav_path, "name": _P(wav_path).name},
        ]
        if jpeg_path:
            atts.append({
                "type": "image", "path": jpeg_path, "name": _P(jpeg_path).name,
            })
        # Short steering text — Parlor uses ~1 sentence; longer ones eat
        # prefill budget on CPU.
        steer = (
            "The user just spoke to you while showing their camera. "
            "Reply in 1-2 short sentences."
            if jpeg_path else
            "The user just spoke to you. Reply in 1-2 short sentences."
        )
        self._live_response_buffer: list[str] = []
        self._engine.send(
            steer,
            cb=self._live_engine_callback,
            attachments=atts,
        )

    def _live_engine_callback(self, evt) -> None:
        """Engine event callback for live-mode turns. Runs on the
        worker thread; UI work bounces via :func:`GLib.idle_add`."""
        if isinstance(evt, EvtToken):
            self._live_response_buffer.append(evt.text)
        elif isinstance(evt, EvtComplete):
            full = "".join(self._live_response_buffer)
            self._live_response_buffer = []
            self._live_controller.notify_engine_complete(full)
        elif isinstance(evt, EvtStopped):
            # Barge-in cancelled the generation; controller already
            # transitioned to LISTENING via its own _on_speech_start.
            self._live_response_buffer = []
        elif isinstance(evt, EvtError):
            self._live_response_buffer = []
            self._live_controller.notify_engine_error(evt.message)

    # ── Tool-call event bridge (runs on engine worker thread) ─────────────
    def _on_tool_event(
        self, fn_name: str, args: dict, result: str, denied: bool
    ) -> None:
        """Worker-thread callback fired by BoxToolEventHandler after each
        tool resolves. Bounces to the main thread to append a tool-call
        card to the streaming assistant bubble — OR stashes it on the
        background-gen state if the user has switched away (so the cards
        replay when they return to the originating chat)."""
        def _ui() -> bool:
            # Always record the call against the active gen so a later
            # restore_assistant_stream() can replay it.
            self._active_gen_tool_calls.append((fn_name, args, result, denied))
            if (self._active_gen_conv_id is not None
                    and self._active_gen_conv_id == self._current_conv_id):
                bubble = self._chat_view._streaming_bubble
                if bubble is not None:
                    bubble.add_tool_call(fn_name, args, result, denied=denied)
            return False
        GLib.idle_add(_ui)

    # ── Agent progress pill (counter fires from engine worker thread) ──────
    def _on_agent_progress(self, current: int, maximum: int | None) -> None:
        """Worker-thread callback from BoxToolEventHandler. Bounces to the
        main thread to update the header pill."""
        GLib.idle_add(self._set_agent_progress, current, maximum)

    def _set_agent_progress(self, current: int, maximum: int | None) -> bool:
        """Main-thread pill update. current==0 hides the pill (idle/reset).
        Also hides when the active generation belongs to a different chat
        than the one on screen — the pill reflects the visible chat's
        agent, not background generations elsewhere."""
        if not hasattr(self, "_agent_pill"):
            return False
        self._agent_progress_state = (current, maximum)
        if current <= 0:
            self._agent_pill.set_visible(False)
            return False
        if (self._active_gen_conv_id is not None
                and self._active_gen_conv_id != self._current_conv_id):
            self._agent_pill.set_visible(False)
            return False
        label = f"Agent {current}/{maximum}" if maximum else f"Agent {current}"
        self._agent_pill.set_text(label)
        self._agent_pill.set_visible(True)
        return False

    # ── Permission prompt bridge (runs on engine worker thread) ────────────
    def _ask_tool_permission_on_main(
        self,
        tool_id: str,
        fn_name: str,
        args: dict,
        risky: bool,
        on_answer,
    ) -> None:
        """Bridge from PermissionGate (worker thread) to GTK (main thread)."""
        def _ui() -> bool:
            # For filesystem calls, lead with the actual path so the user sees
            # exactly what's being accessed (esp. for out-of-workspace grants).
            heading = f"Allow {fn_name}?"
            if tool_id == "filesystem" and isinstance(args.get("path"), str):
                p = args["path"]
                p_short = p if len(p) <= 60 else "…" + p[-59:]
                heading = f"Allow access to {p_short}?"
            dlg = Adw.AlertDialog(
                heading=heading,
                body=_format_tool_call_for_prompt(tool_id, fn_name, args, risky),
            )
            dlg.add_response("deny", "Deny")
            dlg.set_response_appearance("deny", Adw.ResponseAppearance.DESTRUCTIVE)
            dlg.add_response("allow_once", "Allow once")
            dlg.set_response_appearance(
                "allow_once", Adw.ResponseAppearance.SUGGESTED
            )
            if not risky:
                dlg.add_response("allow_chat", "Allow for this chat")
                dlg.add_response("allow_trust", "Always allow")
            dlg.set_default_response("allow_once")
            dlg.set_close_response("deny")

            answered = {"done": False}

            def _on_response(_d, response_id: str) -> None:
                if answered["done"]:
                    return
                answered["done"] = True
                decision_map = {
                    "deny": Decision.DENY,
                    "allow_once": Decision.ALLOW_ONCE,
                    "allow_chat": Decision.ALLOW_CHAT,
                    "allow_trust": Decision.ALLOW_TRUST,
                }
                on_answer(decision_map.get(response_id, Decision.DENY))

            dlg.connect("response", _on_response)
            dlg.present(self)
            return False

        GLib.idle_add(_ui)

    def on_tts_changed(self, enabled: bool) -> None:
        pass  # TTS state is read from settings directly at speak time

    # ── Model picker (header dropdown) ────────────────────────────────────

    def _update_model_menu(self) -> None:
        self._model_menu.remove_all()
        seen: set[str] = set()

        def _add_section(label, paths) -> None:
            if not paths:
                return
            section = Gio.Menu()
            for path in paths:
                item = Gio.MenuItem.new(Path(path).name, None)
                item.set_action_and_target_value(
                    "win.pick-model", GLib.Variant("s", path)
                )
                section.append_item(item)
            self._model_menu.append_section(label, section)

        # 1. Recently used — most useful, on top.
        recent = [
            p for p in self._settings.recent_models if Path(p).is_file()
        ]
        seen.update(recent)
        _add_section("Recent", recent)

        # 2. Every other model available on disk — so a freshly downloaded or
        #    imported model shows up here without a trip through Settings.
        available: list[str] = []

        def _offer(path: str) -> None:
            if path not in seen and Path(path).is_file():
                seen.add(path)
                available.append(path)

        for p in self._settings.imported_gguf_models:
            _offer(p)
        try:
            from .model_catalog import MODELS
            for m in MODELS:
                _offer(str(MODELS_DIR / m["filename"]))
        except Exception:  # noqa: BLE001 — never let the catalog break the menu
            pass
        for pattern in ("*.litertlm", "*.task", "*.gguf"):
            for f in sorted(MODELS_DIR.glob(pattern)):
                _offer(str(f))
        # ~/Downloads too — same courtesy the Box Code picker extends.
        try:
            for f in sorted((Path.home() / "Downloads").glob("*.gguf")):
                _offer(str(f))
        except OSError:
            pass

        _add_section("Available", available)

        browse_section = Gio.Menu()
        browse_section.append("Browse for model…", "win.browse-model")
        self._model_menu.append_section(None, browse_section)

    def _on_pick_model_action(self, _action, param: GLib.Variant) -> None:
        path = param.get_string()
        if not Path(path).is_file():
            self._show_toast(f"File not found: {Path(path).name}")
            return
        self._settings.model_path = path
        self._settings.add_recent_model(path)
        self._settings.save()
        self._update_model_menu()
        self._reload_engine_for_current_conv(force=True)

    def _on_browse_model_action(self, *_) -> None:
        dlg = Gtk.FileDialog(title="Choose a model")
        try:
            dlg.set_initial_folder(Gio.File.new_for_path(str(MODELS_DIR)))
        except Exception:
            pass
        f = Gtk.FileFilter(name="Models (*.litertlm, *.gguf, *.task)")
        for pat in ("*.litertlm", "*.gguf", "*.task"):
            f.add_pattern(pat)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dlg.set_filters(filters)
        dlg.open(self, None, self._on_browse_model_done)

    def _on_browse_model_done(self, dialog, result) -> None:
        try:
            file = dialog.open_finish(result)
        except Exception:
            return
        if file is None:
            return
        path = file.get_path()
        if path:
            self._settings.model_path = path
            self._settings.add_recent_model(path)
            self._settings.save()
            self._update_model_menu()
            self._reload_engine_for_current_conv(force=True)

    # ── Attachments ───────────────────────────────────────────────────────

    def _on_attach_clicked(self) -> None:
        dlg = Gtk.FileDialog(title="Attach file")

        f_text = Gtk.FileFilter(name="Text & code")
        for ext in _TEXT_EXTS:
            f_text.add_pattern(f"*{ext}")
        f_pdf = Gtk.FileFilter(name="PDF")
        f_pdf.add_pattern("*.pdf")
        f_img = Gtk.FileFilter(name="Images")
        for ext in _IMAGE_EXTS:
            f_img.add_pattern(f"*{ext}")
        f_audio = Gtk.FileFilter(name="Audio")
        for ext in _AUDIO_EXTS:
            f_audio.add_pattern(f"*{ext}")
        f_all = Gtk.FileFilter(name="All supported")
        for ext in _TEXT_EXTS | _IMAGE_EXTS | _AUDIO_EXTS | {_PDF_EXT}:
            f_all.add_pattern(f"*{ext}")

        filters = Gio.ListStore.new(Gtk.FileFilter)
        for f in (f_all, f_text, f_pdf, f_img, f_audio):
            filters.append(f)
        dlg.set_filters(filters)

        dlg.open(self, None, self._on_file_attached)

    def _on_file_attached(self, dialog: Gtk.FileDialog, result) -> None:
        try:
            file = dialog.open_finish(result)
        except Exception:
            return
        if file is None:
            return
        path = file.get_path()
        if not path:
            return

        ext = Path(path).suffix.lower()
        name = Path(path).name

        if ext in _IMAGE_EXTS:
            if not self._settings.enable_vision:
                self._show_toast("Enable Vision backend in Preferences → Multimodal first")
                return
            self._chat_view.add_attachment({"type": "image", "name": name, "path": path})

        elif ext in _AUDIO_EXTS:
            if not self._settings.enable_audio:
                self._show_toast("Enable Audio backend in Preferences → Multimodal first")
                return
            self._chat_view.add_attachment({"type": "audio", "name": name, "path": path})

        elif ext == _PDF_EXT:
            content = _extract_pdf(path)
            if content:
                self._chat_view.add_attachment(
                    {"type": "text", "name": name, "path": path, "content": content}
                )
                self._maybe_rag_index(path, name)
            else:
                self._show_toast("Could not extract text from PDF")

        elif ext in _TEXT_EXTS:
            try:
                raw = Path(path).read_bytes()
                content = raw.decode("utf-8", errors="replace")
                # Cap inline content at 512 KB to avoid blocking the UI with
                # giant files (e.g. large JSON datasets). The full file is still
                # indexed via RAG so the assistant can retrieve it semantically.
                _INLINE_CAP = 512 * 1024
                if len(raw) > _INLINE_CAP:
                    content = content[:_INLINE_CAP] + (
                        f"\n\n[… file truncated for context window — "
                        f"{len(raw) // 1024} KB total, full content indexed via RAG …]"
                    )
                self._chat_view.add_attachment(
                    {"type": "text", "name": name, "path": path, "content": content}
                )
                self._maybe_rag_index(path, name)
            except Exception as e:
                self._show_toast(f"Could not read file: {e}")

        else:
            self._show_toast(f"Unsupported file type: {ext}")

    # ── Context-usage bar ──────────────────────────────────────────────────
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Cheap LLM-token estimator. Gemma's SentencePiece averages roughly
        4 chars/token for English; close enough for a usage meter."""
        return max(0, (len(text) + 3) // 4)

    def _recompute_context_base(self) -> None:
        """Recount tokens for system prompt + current conversation history."""
        used = self._estimate_tokens(self._settings.system_prompt)
        if self._current_conv_id is not None:
            for m in self._db.list_messages(self._current_conv_id):
                # Strip embedded voice path before estimating.
                content = m.content.split("\x00", 1)[0]
                used += self._estimate_tokens(content)
        self._ctx_used_base = used
        self.refresh_context_bar()

    def refresh_context_bar(self) -> None:
        """Push current usage to the bar; show/hide per settings."""
        self._chat_view.set_context_bar_visible(self._settings.show_context_bar)
        total = max(1, int(self._settings.max_context_tokens))
        self._chat_view.set_context_usage(self._ctx_used_base, total)

    # ── Knowledge Base panel ──────────────────────────────────────────────

    def _refresh_kb_panel(self) -> None:
        if self._current_conv_id is None:
            self._kb_panel.set_conversation(None, None, self._settings.rag_enabled)
            return
        conv = self._db.get_conversation(self._current_conv_id)
        override = conv.rag_override if conv is not None else None
        self._kb_panel.set_conversation(
            self._current_conv_id, override, self._settings.rag_enabled,
        )

    def _on_kb_toggle(self, btn: Gtk.ToggleButton) -> None:
        self._kb_split.set_show_sidebar(btn.get_active())

    # ── Left sidebar (chats list) hide/show + drag-resize ─────────────────
    def _apply_sidebar_visible(self, visible: bool) -> None:
        """Show or hide the chats sidebar. Hiding stashes the current
        width in settings so the show path can restore it."""
        if visible:
            self._sidebar_toolbar.set_visible(True)
            width = max(200, int(self._settings.sidebar_width or 280))
            self._sidebar_split.set_position(width)
            self._sidebar_toggle_btn.set_tooltip_text("Hide chat list")
        else:
            # Remember the current divider so we can restore exactly.
            pos = self._sidebar_split.get_position()
            if pos and pos >= 200:
                self._settings.sidebar_width = pos
            self._sidebar_toolbar.set_visible(False)
            self._sidebar_split.set_position(0)
            self._sidebar_toggle_btn.set_tooltip_text("Show chat list")

    def _on_sidebar_toggle(self, btn: Gtk.ToggleButton) -> None:
        active = btn.get_active()
        self._apply_sidebar_visible(active)
        if self._settings.sidebar_visible != active:
            self._settings.sidebar_visible = active
            self._settings.save()

    def _on_sidebar_position_changed(self, *_args) -> None:
        """Drag handler — persist the new width once GTK settles. Ignored
        while the sidebar is hidden (the position is forced to 0)."""
        if not self._sidebar_toggle_btn.get_active():
            return
        if self._sidebar_save_pending:
            return
        self._sidebar_save_pending = True

        def _save() -> bool:
            self._sidebar_save_pending = False
            pos = self._sidebar_split.get_position()
            if pos and pos >= 200 and pos != self._settings.sidebar_width:
                self._settings.sidebar_width = pos
                self._settings.save()
            return False

        # 400 ms debounce so a single drag doesn't write the JSON 50 times.
        GLib.timeout_add(400, _save)

    def _on_kb_split_notify(self, *_args) -> None:
        showing = self._kb_split.get_show_sidebar()
        if self._kb_toggle_btn.get_active() != showing:
            self._kb_toggle_btn.set_active(showing)

    def _on_kb_add_files(self, paths: list[str]) -> None:
        if self._current_conv_id is None:
            self._show_toast("Open a chat first")
            return
        if not self._rag.is_model_ready():
            self._show_toast("Download the embed model in Preferences → Knowledge first")
            return
        conv_id = self._current_conv_id
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext in _IMAGE_RAG_EXTS:
                self._index_image_into(
                    path, Path(path).name,
                    conversation_id=conv_id,
                    on_done=self._refresh_kb_panel,
                )
            else:
                # force=True: indexing via KB panel bypasses the global
                # rag_enabled gate (user intent is explicit).
                self._maybe_rag_index(
                    path, Path(path).name,
                    on_done=self._refresh_kb_panel, force=True,
                )

    def _on_kb_remove_source(self, source_path: str) -> None:
        if self._current_conv_id is None:
            return
        n = self._rag.delete_source(self._current_conv_id, source_path)
        self._refresh_kb_panel()
        if n:
            self._show_toast(f"Removed {Path(source_path).name} ({n} chunks)")

    def _on_kb_detach_notebook(self, nb_id: int) -> None:
        if self._current_conv_id is None:
            return
        self._rag.detach_notebook(self._current_conv_id, nb_id)
        self._refresh_kb_panel()

    def _on_kb_rag_override_changed(self, override: int | None) -> None:
        if self._current_conv_id is None:
            return
        self._db.set_rag_override(self._current_conv_id, override)
        # Sync the subtitle hint (recomputes effective on/off).
        self._refresh_kb_panel()

    @staticmethod
    def _hits_to_payload(hits) -> list[dict]:
        """Convert vector_store.Chunk objects into JSON-safe dicts for DB+UI."""
        if not hits:
            return []
        out: list[dict] = []
        for h in hits:
            out.append({
                "label": h.source_label,
                "path": h.source_path,
                "chunk_idx": h.chunk_idx,
                "text": h.text,
                "score": float(h.score) if h.score is not None else None,
            })
        return out

    def _consume_pending_context_json(self) -> str | None:
        """Pop the pending RAG hits and return them as a JSON string (or None)."""
        if not self._pending_context_hits:
            return None
        import json
        payload = self._hits_to_payload(self._pending_context_hits)
        self._pending_context_hits = []
        return json.dumps(payload, ensure_ascii=False) if payload else None

    def _active_gen_context_json(self) -> str | None:
        """Serialise the active generation's RAG-hit payload for DB storage."""
        if not self._active_gen_context:
            return None
        import json
        return json.dumps(self._active_gen_context, ensure_ascii=False)

    def _clear_active_gen(self) -> None:
        self._active_gen_conv_id = None
        self._active_gen_text = ""
        self._active_gen_context = []
        self._active_gen_tool_calls = []

    # ── Notebooks ─────────────────────────────────────────────────────────

    def _show_chat_pane(self) -> None:
        self._main_stack.set_visible_child_name("chat")
        self._kb_toggle_btn.set_visible(True)

    def _show_notebook_pane(self, nb_id: int) -> None:
        nb = self._db.get_notebook(nb_id)
        if nb is None:
            return
        self._notebook_view.load(nb.id, nb.name, auto_attach=bool(nb.auto_attach))
        self._main_stack.set_visible_child_name("notebook")
        # KB panel is per-chat — hide its toggle while a notebook is open
        # and close the panel if it was open.
        self._kb_toggle_btn.set_visible(False)
        if self._kb_split.get_show_sidebar():
            self._kb_split.set_show_sidebar(False)

    def _open_notebook(self, nb_id: int) -> None:
        self._show_notebook_pane(nb_id)
        # Deselect any conversation in the chats sidebar so the visual
        # state is consistent.

    def _on_create_notebook(self) -> None:
        dlg = Adw.AlertDialog(heading="New notebook", body="Name this notebook.")
        entry = Gtk.Entry(placeholder_text="e.g. Engineering docs")
        entry.set_margin_top(8)
        entry.set_activates_default(True)
        dlg.set_extra_child(entry)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("create", "Create")
        dlg.set_response_appearance("create", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("create")
        def on_response(_d, r):
            if r == "create":
                name = entry.get_text().strip() or "Untitled notebook"
                nb = self._rag.create_notebook(name)
                self._notebooks_list.refresh(select_id=nb.id)
                self._sidebar_stack.set_visible_child_name("notebooks")
                self._show_notebook_pane(nb.id)
        dlg.connect("response", on_response)
        dlg.present(self)

    def _on_rename_notebook_clicked(self, nb_id: int) -> None:
        self._on_rename_notebook_action(None, GLib.Variant("i", nb_id))

    def _on_delete_notebook_clicked(self, nb_id: int) -> None:
        self._on_delete_notebook_action(None, GLib.Variant("i", nb_id))

    def _on_rename_notebook_action(self, _action, param: GLib.Variant) -> None:
        nb_id = param.get_int32()
        nb = self._db.get_notebook(nb_id)
        if nb is None:
            return
        dlg = Adw.AlertDialog(heading="Rename notebook", body="Enter a new name.")
        entry = Gtk.Entry(text=nb.name)
        entry.set_margin_top(8)
        entry.set_activates_default(True)
        dlg.set_extra_child(entry)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("ok", "Rename")
        dlg.set_response_appearance("ok", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("ok")
        def on_response(_d, r):
            if r == "ok":
                new = entry.get_text().strip() or nb.name
                self._rag.rename_notebook(nb_id, new)
                self._notebooks_list.refresh(select_id=nb_id)
                # If the user has this notebook open, refresh the header.
                if (self._main_stack.get_visible_child_name() == "notebook"
                        and self._notebook_view._nb_id == nb_id):
                    nb_after = self._db.get_notebook(nb_id)
                    self._notebook_view.load(
                        nb_id, new,
                        auto_attach=bool(nb_after.auto_attach) if nb_after else False,
                    )
                # KB panel may also show this notebook in its attached list.
                self._refresh_kb_panel()
        dlg.connect("response", on_response)
        dlg.present(self)

    def _on_delete_notebook_action(self, _action, param: GLib.Variant) -> None:
        nb_id = param.get_int32()
        nb = self._db.get_notebook(nb_id)
        if nb is None:
            return
        n_files = len(self._rag.notebook_sources(nb_id))
        body = (f"Permanently deletes the notebook “{nb.name}” "
                f"and its {n_files} source{'s' if n_files != 1 else ''}. "
                "Chats that had this notebook attached will lose access to "
                "its sources but otherwise are unaffected.")
        dlg = Adw.AlertDialog(heading="Delete this notebook?", body=body)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("delete", "Delete")
        dlg.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_default_response("cancel")
        def on_response(_d, r):
            if r == "delete":
                self._rag.delete_notebook(nb_id)
                self._notebooks_list.refresh()
                if (self._main_stack.get_visible_child_name() == "notebook"
                        and self._notebook_view._nb_id == nb_id):
                    self._show_chat_pane()
                self._refresh_kb_panel()
        dlg.connect("response", on_response)
        dlg.present(self)

    def _on_notebook_add_files(self, nb_id: int, paths: list[str]) -> None:
        if not self._rag.is_model_ready():
            self._show_toast("Download the embed model in Preferences → Knowledge first")
            return
        for path in paths:
            ext = Path(path).suffix.lower()
            if ext in _IMAGE_RAG_EXTS:
                self._index_image_into(
                    path, Path(path).name,
                    notebook_id=nb_id,
                    on_done=lambda nid=nb_id: (
                        self._notebook_view.refresh(),
                        self._notebooks_list.refresh(select_id=nid),
                        self._refresh_kb_panel(),
                    ),
                )
            else:
                self._index_into_notebook(nb_id, path, Path(path).name)

    def _on_notebook_auto_attach_changed(self, nb_id: int, on: bool) -> None:
        self._db.set_notebook_auto_attach(nb_id, on)
        # Reflect the new sort order in the sidebar (touch updates updated_at).
        self._notebooks_list.refresh(select_id=nb_id)

    def _on_notebook_remove_source(self, nb_id: int, source_path: str) -> None:
        n = self._rag.delete_notebook_source(nb_id, source_path)
        self._notebook_view.refresh()
        self._notebooks_list.refresh(select_id=nb_id)
        self._refresh_kb_panel()
        if n:
            self._show_toast(f"Removed {Path(source_path).name} ({n} chunks)")

    def _index_into_notebook(self, nb_id: int, path: str, name: str) -> None:
        """Background-index a file into a notebook with banner progress."""
        start_t = time.monotonic()

        def _set_banner(text: str, revealed: bool = True) -> bool:
            self._rag_banner.set_title(text)
            self._rag_banner.set_revealed(revealed)
            return False

        GLib.idle_add(_set_banner, f"Indexing {name}: preparing…", True)

        def _on_progress(done: int, total: int) -> None:
            if done < 10 or done % 5 == 0 or done == total:
                elapsed = time.monotonic() - start_t
                rate = done / elapsed if elapsed > 0 else 0
                remain = (total - done) / rate if rate > 0 else 0
                m, s = divmod(int(remain), 60)
                eta = f"~{m}m {s}s" if m else f"~{s}s"
                GLib.idle_add(
                    _set_banner,
                    f"Indexing {name}: {done}/{total} chunks ({eta} left)",
                    True,
                )

        def _worker() -> None:
            try:
                label, n = self._rag.index_file_into_notebook(
                    nb_id, path, on_progress=_on_progress,
                )
            except Exception as e:
                log.exception("Notebook index failed for %s", path)
                GLib.idle_add(_set_banner, f"Index failed: {str(e)[:80]}", True)
                return
            if label is None:
                GLib.idle_add(_set_banner, f"{name}: nothing to index", False)
                GLib.idle_add(self._show_toast, f"{name}: nothing to index")
            else:
                GLib.idle_add(_set_banner, f"Indexed {name} ({n} chunks) ✓", False)
                GLib.idle_add(self._show_toast, f"Indexed {name} ({n} chunks)")
            def _refresh_ui() -> bool:
                # Main-thread side-effects: bump notebook timestamp and
                # refresh the UI lists. Done here because Database's
                # sqlite3 connection is bound to this thread.
                if n > 0:
                    self._db.touch_notebook(nb_id)
                self._notebook_view.refresh()
                self._notebooks_list.refresh(select_id=nb_id)
                self._refresh_kb_panel()
                return False
            GLib.idle_add(_refresh_ui)

        threading.Thread(target=_worker, daemon=True).start()

    # ── Image RAG: caption then embed ─────────────────────────────────────
    def _index_image_into(
        self,
        path: str,
        name: str,
        *,
        conversation_id: int | None = None,
        notebook_id: int | None = None,
        on_done=None,
    ) -> None:
        """Caption the image with the active LLM, then embed the caption.

        Steps run sequentially across two threads:
          1. Engine worker thread: caption (slow; minutes on CPU-only).
          2. RAG: embed the caption (fast; one chunk).
          3. Main thread: refresh UI + bump notebook timestamp if applicable.
        """
        if not self._engine.is_ready:
            self._show_toast("Model still loading…")
            return
        if not self._settings.enable_vision:
            self._show_toast("Enable Vision in Preferences → Multimodal first")
            return

        scope_label = "notebook" if notebook_id is not None else "chat"

        def _set_banner(text: str, revealed: bool = True) -> bool:
            self._rag_banner.set_title(text)
            self._rag_banner.set_revealed(revealed)
            return False

        GLib.idle_add(_set_banner, f"Captioning {name} (this can take a minute)…", True)

        def _on_caption(caption: str | None, error: str | None) -> None:
            # Runs on the engine worker thread.
            if error or not caption:
                msg = error or "Empty caption."
                log.warning("Image caption failed for %s: %s", path, msg)
                GLib.idle_add(_set_banner, f"Caption failed: {msg[:80]}", True)
                GLib.idle_add(self._show_toast, f"Caption failed: {msg[:80]}")
                return
            GLib.idle_add(_set_banner, f"Embedding {name}…", True)
            try:
                self._rag.index_image(
                    path, caption,
                    conversation_id=conversation_id,
                    notebook_id=notebook_id,
                )
            except Exception as e:
                log.exception("Image embed failed")
                GLib.idle_add(_set_banner, f"Embed failed: {str(e)[:80]}", True)
                return
            GLib.idle_add(_set_banner, f"Indexed {name} ({scope_label}) ✓", False)
            GLib.idle_add(self._show_toast, f"Indexed {name}")
            def _ui_finish() -> bool:
                if notebook_id is not None:
                    self._db.touch_notebook(notebook_id)
                if on_done is not None:
                    try:
                        on_done()
                    except Exception:
                        log.exception("on_done failed")
                return False
            GLib.idle_add(_ui_finish)

        self._engine.caption_image(path, _on_caption)

    # ── Attach / detach notebooks to the current chat ─────────────────────
    def _on_attach_notebook_action(self, _action, param: GLib.Variant) -> None:
        nb_id = param.get_int32()
        if self._current_conv_id is None:
            return
        self._rag.attach_notebook(self._current_conv_id, nb_id)
        self._refresh_kb_panel()
        nb = self._db.get_notebook(nb_id)
        if nb:
            self._show_toast(f"Attached notebook “{nb.name}”")

    def _on_detach_notebook_action(self, _action, param: GLib.Variant) -> None:
        nb_id = param.get_int32()
        if self._current_conv_id is None:
            return
        self._rag.detach_notebook(self._current_conv_id, nb_id)
        self._refresh_kb_panel()

    def effective_rag_enabled(self, conv_id: int | None) -> bool:
        """Per-chat override wins; otherwise fall back to the global toggle."""
        if conv_id is None:
            return self._settings.rag_enabled
        conv = self._db.get_conversation(conv_id)
        if conv is not None and conv.rag_override is not None:
            return bool(conv.rag_override)
        return self._settings.rag_enabled

    # ── Phase 4 per-chat tool overrides ────────────────────────────────────
    _TOOL_OVERRIDE_FIELDS = {
        "web_search": "tool_web_override",
        "filesystem": "tool_fs_override",
    }
    _TOOL_GLOBAL_FIELDS = {
        "web_search": "tool_web_search_enabled",
        "filesystem": "tool_fs_enabled",
    }

    def effective_tool_enabled(
        self, conv_id: int | None, tool_id: str
    ) -> bool:
        """Per-chat override wins; otherwise the global tool switch."""
        global_field = self._TOOL_GLOBAL_FIELDS.get(tool_id)
        if global_field is None:
            return False
        global_on = bool(getattr(self._settings, global_field, False))
        if conv_id is None:
            return global_on
        conv = self._db.get_conversation(conv_id)
        override_field = self._TOOL_OVERRIDE_FIELDS.get(tool_id)
        if conv is not None and override_field is not None:
            ovr = getattr(conv, override_field, None)
            if ovr is not None:
                return bool(ovr)
        return global_on

    def effective_enabled_tool_ids(self, conv_id: int | None) -> list[str]:
        return [
            tid for tid in self._TOOL_GLOBAL_FIELDS
            if self.effective_tool_enabled(conv_id, tid)
        ]

    # ── Phase 5: agent-mode helpers ────────────────────────────────────────
    def effective_agent_enabled(self, conv_id: int | None) -> bool:
        """Per-chat override wins, else the global agent_enabled flag."""
        if conv_id is None:
            return bool(self._settings.agent_enabled)
        conv = self._db.get_conversation(conv_id)
        if conv is not None and getattr(conv, "agent_override", None) is not None:
            return bool(conv.agent_override)
        return bool(self._settings.agent_enabled)

    def tools_for_model(self, conv_id: int | None) -> list[str]:
        """The tool ids actually exposed to the model for this chat.

        Agent mode is the master gate for full autonomous behaviour:
        - Agent ON  → every enabled tool (web search, filesystem, …) plus the
          research prompt + iteration cap.
        - Agent OFF → the per-chat composer quick toggles still expose
          individual tools (web search via 🌐, filesystem via 📁). No
          research prompt, no cap — model can call them once if useful.
        """
        if self.effective_agent_enabled(conv_id):
            return self.effective_enabled_tool_ids(conv_id)
        ids: list[str] = []
        if self.effective_tool_enabled(conv_id, "web_search"):
            ids.append("web_search")
        if self.effective_tool_enabled(conv_id, "filesystem"):
            ids.append("filesystem")
        return ids

    # ── composer toggle buttons (agent + web search) ───────────────────────
    def refresh_agent_button(self) -> None:
        """The 🤖 toggle is always visible; its active state reflects the
        effective agent setting for the current chat."""
        self._chat_view.set_agent_visible(True)
        self._chat_view.set_agent_active(
            self.effective_agent_enabled(self._current_conv_id)
        )

    def refresh_websearch_button(self) -> None:
        """The 🌐 toggle is always visible; active = web search effectively
        enabled for the current chat."""
        self._chat_view.set_websearch_visible(True)
        self._chat_view.set_websearch_active(
            self.effective_tool_enabled(self._current_conv_id, "web_search")
        )

    def refresh_fs_button(self) -> None:
        """The 📁 toggle is always visible; active = filesystem effectively
        enabled for the current chat."""
        self._chat_view.set_fs_visible(True)
        self._chat_view.set_fs_active(
            self.effective_tool_enabled(self._current_conv_id, "filesystem")
        )

    def refresh_tts_button(self) -> None:
        """The 🔊 composer button is always visible; the popover state
        mirrors settings (tts_auto_speak + tts_volume)."""
        self._chat_view.set_tts_visible(True)
        self._chat_view.set_tts_state(
            enabled=bool(self._settings.tts_auto_speak),
            volume=float(self._settings.tts_volume),
        )

    def refresh_memory_button(self) -> None:
        """The 🧠 composer button is always visible; the popover switch
        mirrors settings.memory_enabled."""
        self._chat_view.set_memory_visible(True)
        self._chat_view.set_memory_state(bool(self._settings.memory_enabled))

    def _refresh_composer_toggles(self) -> None:
        self.refresh_agent_button()
        self.refresh_websearch_button()
        self.refresh_fs_button()
        self.refresh_tts_button()
        self.refresh_memory_button()
        self.refresh_tools_badge()

    def _set_chat_flag(self, *, global_attr: str, override_tool: str | None,
                       agent: bool, active: bool) -> None:
        """Per-chat-that-remembers write: set the current chat's override AND
        update the global default so new chats inherit this choice."""
        if self._current_conv_id is not None:
            ovr = 1 if active else 0
            if agent:
                self._db.set_agent_override(self._current_conv_id, ovr)
            elif override_tool is not None:
                self._db.set_tool_override(
                    self._current_conv_id, override_tool, ovr
                )
        setattr(self._settings, global_attr, active)
        self._settings.save()
        if self._settings.model_path:
            self._reload_engine_for_current_conv(force=True)
        self._refresh_composer_toggles()

    def _on_agent_toggle(self, active: bool) -> None:
        if active and not self._settings.agent_first_enable_acknowledged:
            self._confirm_composer_enable(
                heading="Enable agent mode?",
                body=(
                    "The assistant will be able to chain several tool calls "
                    "together to work through a task on its own — searching "
                    "the web or reading files across multiple steps. Each "
                    "tool still asks permission as configured, and a per-"
                    "message cap stops runaway loops."
                ),
                ack_attr="agent_first_enable_acknowledged",
                revert=lambda: self._chat_view.set_agent_active(False),
                on_ok=lambda: self._set_chat_flag(
                    global_attr="agent_enabled", override_tool=None,
                    agent=True, active=True,
                ),
            )
            return
        self._set_chat_flag(
            global_attr="agent_enabled", override_tool=None,
            agent=True, active=active,
        )

    def _on_websearch_toggle(self, active: bool) -> None:
        if active and not self._settings.tool_web_search_first_enable_acknowledged:
            self._confirm_composer_enable(
                heading="Enable web search?",
                body=(
                    "The assistant will be able to search the public web "
                    "through DuckDuckGo when it judges a query needs it. "
                    "Results are filtered to HTTPS only; no API key leaves "
                    "your machine."
                ),
                ack_attr="tool_web_search_first_enable_acknowledged",
                revert=lambda: self._chat_view.set_websearch_active(False),
                on_ok=lambda: self._set_chat_flag(
                    global_attr="tool_web_search_enabled",
                    override_tool="web_search", agent=False, active=True,
                ),
            )
            return
        self._set_chat_flag(
            global_attr="tool_web_search_enabled",
            override_tool="web_search", agent=False, active=active,
        )

    def _on_fs_toggle(self, active: bool) -> None:
        """Composer 📁 toggle. Same per-chat-that-remembers pattern as
        web search. Filesystem reads always prompt for permission."""
        if active and not self._settings.tool_fs_first_enable_acknowledged:
            self._confirm_composer_enable(
                heading="Enable filesystem access?",
                body=(
                    "The assistant will be able to read, list, and search "
                    "files under your workspace folder "
                    f"({self._settings.tool_fs_root}). Each call still "
                    "asks for permission. Writes/deletes stay off unless "
                    "you turn them on in Preferences → Tools."
                ),
                ack_attr="tool_fs_first_enable_acknowledged",
                revert=lambda: self._chat_view.set_fs_active(False),
                on_ok=lambda: self._set_chat_flag(
                    global_attr="tool_fs_enabled",
                    override_tool="filesystem", agent=False, active=True,
                ),
            )
            return
        self._set_chat_flag(
            global_attr="tool_fs_enabled",
            override_tool="filesystem", agent=False, active=active,
        )

    def _on_tts_toggle(self, active: bool) -> None:
        """Composer 🔊 popover switch — flips settings.tts_auto_speak.
        No engine reload needed (TTS is decoupled from generation)."""
        if bool(self._settings.tts_auto_speak) == active:
            return
        self._settings.tts_auto_speak = active
        # Auto-enable the parent TTS feature when the user flips this on for
        # the first time — otherwise the switch silently does nothing.
        if active and not self._settings.enable_tts:
            self._settings.enable_tts = True
        self._settings.save()
        self.refresh_tts_button()

    def _on_tts_volume_changed(self, value: float) -> None:
        """Slider drag — write the new gain to settings. The PiperTTS gain
        provider reads it lazily per sentence, so the next sentence picks
        up the new level without restarting playback."""
        v = max(0.0, min(2.0, float(value)))
        if abs(self._settings.tts_volume - v) < 1e-3:
            return
        self._settings.tts_volume = v
        self._settings.save()

    # ── memory (Phase 6) ──────────────────────────────────────────────────
    def _on_memory_toggle(self, active: bool) -> None:
        """Composer 🧠 popover switch — flips settings.memory_enabled. Recall
        happens at send time, so no engine reload is needed."""
        if bool(self._settings.memory_enabled) == active:
            return
        if active and not self._rag.is_model_ready():
            self._show_toast(
                "Memory needs the embedding model — download it in "
                "Preferences → Knowledge."
            )
            self.refresh_memory_button()  # revert the switch
            return
        self._settings.memory_enabled = active
        self._settings.save()
        self.refresh_memory_button()

    def _on_memory_save(self) -> None:
        """Save the current composer text as a memory (explicit capture)."""
        text = self._chat_view.get_composer_text().strip()
        if not text:
            self._show_toast("Type something first, then Remember it.")
            return
        if not self._rag.is_model_ready():
            self._show_toast(
                "Memory needs the embedding model — download it in "
                "Preferences → Knowledge."
            )
            return
        preview = text if len(text) <= 60 else text[:57] + "…"

        def _work() -> None:
            try:
                self._rag.remember(text)
                GLib.idle_add(
                    lambda: (self._show_toast(f"Remembered: {preview}"), False)[1]
                )
            except Exception as e:  # noqa: BLE001
                log.exception("memory save failed")
                GLib.idle_add(
                    lambda: (self._show_toast(f"Couldn't save memory: {e}"), False)[1]
                )

        threading.Thread(target=_work, daemon=True).start()

    def _on_save_bubble_as_memory(self, text: str) -> None:
        """Save a message bubble's text as a memory (user-triggered from bubble)."""
        text = text.strip()
        if not text:
            return
        if not self._rag.is_model_ready():
            self._show_toast(
                "Memory needs the embedding model — download it in "
                "Preferences → Knowledge."
            )
            return
        preview = text if len(text) <= 60 else text[:57] + "…"

        def _work() -> None:
            try:
                self._rag.remember(text)
                GLib.idle_add(
                    lambda: (self._show_toast(f"Remembered: {preview}"), False)[1]
                )
            except Exception as e:  # noqa: BLE001
                log.exception("memory save failed")
                GLib.idle_add(
                    lambda: (self._show_toast(f"Couldn't save memory: {e}"), False)[1]
                )

        threading.Thread(target=_work, daemon=True).start()

    def _confirm_composer_enable(self, *, heading: str, body: str,
                                 ack_attr: str, revert, on_ok) -> None:
        """First-time-enable confirm for a composer toggle. On cancel, flips
        the button back (via ``revert``) without re-firing the handler."""
        dlg = Adw.AlertDialog(heading=heading, body=body)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("enable", "Enable")
        dlg.set_response_appearance("enable", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("enable")
        dlg.set_close_response("cancel")

        def _resp(_d, r: str) -> None:
            if r == "enable":
                setattr(self._settings, ack_attr, True)
                self._settings.save()
                on_ok()
            else:
                revert()

        dlg.connect("response", _resp)
        dlg.present(self)

    def _maybe_rag_index(
        self,
        path: str,
        name: str,
        on_done=None,
        force: bool = False,
    ) -> None:
        """If RAG is enabled and ready, index ``path`` on a background thread.

        ``force=True`` bypasses the global rag_enabled gate — used when the
        user adds a file directly through the KB panel.
        """
        if not (force or self._settings.rag_enabled):
            return
        if not self._rag.is_model_ready():
            return
        if self._current_conv_id is None:
            return

        conv_id = self._current_conv_id
        start_t = time.monotonic()

        def _set_banner(text: str, revealed: bool = True) -> bool:
            self._rag_banner.set_title(text)
            self._rag_banner.set_revealed(revealed)
            return False

        GLib.idle_add(_set_banner, f"Indexing {name}: preparing…", True)

        def _on_progress(done: int, total: int) -> None:
            # Throttle UI updates: every chunk for the first 10, then every 5.
            if done < 10 or done % 5 == 0 or done == total:
                elapsed = time.monotonic() - start_t
                rate = done / elapsed if elapsed > 0 else 0
                remain = (total - done) / rate if rate > 0 else 0
                m, s = divmod(int(remain), 60)
                eta = f"~{m}m {s}s" if m else f"~{s}s"
                GLib.idle_add(
                    _set_banner,
                    f"Indexing {name}: {done}/{total} chunks ({eta} left)",
                    True,
                )

        def _worker() -> None:
            try:
                label, n = self._rag.index_file(conv_id, path, on_progress=_on_progress)
            except Exception as e:
                log.exception("RAG index failed for %s", path)
                GLib.idle_add(_set_banner, f"Index failed: {str(e)[:80]}", True)
                return
            if label is None:
                GLib.idle_add(_set_banner, f"{name}: nothing to index", False)
                GLib.idle_add(self._show_toast, f"{name}: nothing to index")
            else:
                GLib.idle_add(_set_banner, f"Indexed {name} ({n} chunks) ✓", False)
                GLib.idle_add(self._show_toast, f"Indexed {name} ({n} chunks)")
            if on_done is not None:
                GLib.idle_add(lambda: (on_done(), False)[1])

        threading.Thread(target=_worker, daemon=True).start()

    # ── Voice recording ────────────────────────────────────────────────────

    def _on_mic_toggle(self) -> None:
        if self._recorder.is_recording:
            wav_path = self._recorder.stop()
            self._chat_view.set_recording(False)
            if wav_path:
                if not self._settings.enable_audio:
                    self._show_toast("Enable Audio backend in Preferences → Multimodal first")
                    return
                secs = self._recorder.duration_s
                dur = f"{int(secs // 60)}:{int(secs % 60):02d}"
                # Copy to permanent storage so the user can play it back later.
                permanent = VOICE_DIR / f"voice_{int(time.time() * 1000)}.wav"
                shutil.copy2(wav_path, permanent)
                self._chat_view.add_attachment({
                    "type": "audio",
                    "name": f"Voice message ({dur})",
                    "path": wav_path,
                    "permanent_path": str(permanent),
                })
                if self._settings.voice_auto_send:
                    GLib.idle_add(self._chat_view.trigger_send)
            else:
                self._show_toast("Recording too short — try again")
        else:
            from . import audio_stream
            if not audio_stream.is_available():
                self._show_toast(
                    "Microphone unavailable — install PortAudio "
                    "(sudo apt install libportaudio2), then restart Box."
                )
                return
            try:
                self._recorder.start()
            except Exception as e:  # noqa: BLE001
                log.exception("voice recorder failed to start")
                self._show_toast(f"Couldn't start the microphone: {e}")
                return
            self._chat_view.set_recording(True)

    # ── TTS ────────────────────────────────────────────────────────────────

    def _on_speak(self, text: str) -> None:
        if not self._settings.enable_tts:
            return
        if not tts_is_ready(self._settings.tts_voice):
            self._show_toast("Download Piper TTS in Preferences → Multimodal first")
            return
        self._tts_speak_gen += 1
        gen = self._tts_speak_gen
        self._chat_view.set_tts_speaking(True)

        def _on_done():
            def _ui():
                if self._tts_speak_gen == gen:
                    self._chat_view.set_tts_speaking(False)
                return False
            GLib.idle_add(_ui)

        self._tts.speak(text, self._settings.tts_voice, on_done=_on_done)

    def _on_stop_tts_btn(self) -> None:
        self._tts_streaming = False
        self._tts_speak_gen += 1
        self._tts_user_stopped = True
        self._tts.stop()
        self._chat_view.set_tts_speaking(False)

    # ── Send / Stop ────────────────────────────────────────────────────────

    def _on_user_send(self, text: str, attachments: list[dict]) -> None:
        if self._current_conv_id is None:
            self.start_new_chat()
            assert self._current_conv_id is not None

        if not self._engine.is_ready:
            self._show_toast("Model still loading…")
            return

        # Engine is single-threaded — block sending while another chat is
        # mid-generation. Interleaving streams would put A's tokens into
        # B's transcript (the bug this whole block defends against). The
        # composer keeps input enabled so the user can draft; Send just
        # toasts and returns.
        gen_id = self._active_gen_conv_id
        if gen_id is not None and gen_id != self._current_conv_id:
            src = self._db.get_conversation(gen_id)
            src_name = src.title if src and src.title else "another chat"
            self._show_toast(
                f"Wait — assistant is still replying in “{src_name}”"
            )
            return

        # "audit /var/log/dmesg for security issues" — when the message reads
        # like an audit request AND names a readable file in the workspace,
        # route to the chunked map-reduce audit instead of a normal send.
        # Falls through to normal chat otherwise (e.g. no path → model asks).
        if self._maybe_start_audit(text):
            return

        # Split attachments. When RAG is enabled for this chat, text/PDF
        # content is NOT folded into the message — we rely on retrieval
        # instead. Folding the full doc would overflow the model's context
        # window (e.g. a novel = ~150k tokens vs. a typical 4096-token window).
        rag_on = self.effective_rag_enabled(self._current_conv_id)
        full_text = text
        media_attachments: list[dict] = []
        display_parts: list[str] = []
        fold_text_content = not rag_on

        for att in attachments:
            if att["type"] == "text":
                if fold_text_content:
                    full_text = f'[{att["name"]}]\n{att["content"]}\n\n{full_text}'
                display_parts.append(f"📄 {att['name']}")
            elif att["type"] == "image":
                media_attachments.append(att)
                display_parts.append(f"🖼 {att['name']}")
            elif att["type"] == "audio":
                media_attachments.append(att)
                display_parts.append(f"🎤 {att['name']}")

        display_text = text or ""
        if display_parts:
            display_text = ", ".join(display_parts) + (" — " + text if text else "")

        if not full_text.strip() and not media_attachments:
            return

        # Embed permanent voice path so the bubble can show a play button.
        voice_att = next(
            (a for a in attachments if a["type"] == "audio" and a.get("permanent_path")),
            None,
        )
        db_content = display_text or "(voice)"
        if voice_att:
            db_content = db_content + "\x00" + voice_att["permanent_path"]

        self._db.add_message(self._current_conv_id, "user", db_content)
        self._chat_view.append_user_message(db_content)
        self._recompute_context_base()

        msgs = self._db.list_messages(self._current_conv_id)
        if len([m for m in msgs if m.role == "user"]) == 1:
            title = (display_text or "Voice message")[:60].strip().splitlines()[0]
            self._db.rename_conversation(self._current_conv_id, title)
            self._sidebar.refresh(select_id=self._current_conv_id)

        self._gen_start = None
        self._gen_tokens = 0
        self._tts_streaming = False
        # Fresh send → previous user-stop no longer suppresses this gen's
        # auto-speak fallback.
        self._tts_user_stopped = False
        if (self._settings.enable_tts and self._settings.tts_auto_speak
                and tts_is_ready(self._settings.tts_voice)):
            self._tts_speak_gen += 1
            gen = self._tts_speak_gen
            self._tts_streaming = True
            self._chat_view.set_tts_speaking(True)

            def _on_stream_done():
                def _ui():
                    self._tts_streaming = False
                    if self._tts_speak_gen == gen:
                        self._chat_view.set_tts_speaking(False)
                    return False
                GLib.idle_add(_ui)

            self._tts.start_stream(self._settings.tts_voice, on_done=_on_stream_done)

        # RAG: prepend retrieved context if enabled, model is ready, and the
        # current conversation has indexed chunks. Failures are non-fatal —
        # we just send the original text and toast the error.
        send_text = full_text
        self._pending_context_hits = []
        if (rag_on
                and self._rag.is_model_ready()
                and self._current_conv_id is not None
                and self._rag.count_scope(self._current_conv_id) > 0):
            try:
                ctx, hits = self._rag.retrieve(self._current_conv_id, text or full_text)
                if ctx:
                    send_text = ctx + "\n" + full_text
                    self._pending_context_hits = hits
            except Exception as e:
                log.exception("RAG retrieval failed")
                self._show_toast(f"RAG retrieval failed: {e}")

        # Persistent memory (Phase 6): independent of RAG scope — recall the
        # most relevant saved memories and prepend them too. Failures are
        # non-fatal. Memory hits render in the same "sources" card, labelled
        # "🧠 Memory".
        if (self._settings.memory_enabled
                and self._rag.is_model_ready()
                and self._rag.count_memories() > 0):
            try:
                mctx, mhits = self._rag.recall(text or full_text)
                if mctx:
                    send_text = mctx + "\n" + send_text
                    self._pending_context_hits = (
                        list(self._pending_context_hits) + mhits
                    )
            except Exception as e:
                log.exception("memory recall failed")
                self._show_toast(f"Memory recall failed: {e}")

        ctx_payload = self._hits_to_payload(self._pending_context_hits)
        self._chat_view.start_assistant_stream(context=ctx_payload or None)
        # Capture this generation's identity + initial state so the
        # engine_callback can route tokens to the right conv (and persist
        # to it) even if the user switches chats mid-stream.
        self._active_gen_conv_id = self._current_conv_id
        self._active_gen_text = ""
        self._active_gen_context = ctx_payload or []
        self._active_gen_tool_calls = []
        # Agent mode: reset the per-send tool-call budget so the cap applies
        # fresh to this generation.
        if self._tool_handler is not None:
            self._tool_handler.reset_iterations()
        # "Allow once" file-access grants only cover one user turn — drop them
        # so the next send re-prompts for any out-of-workspace path.
        from .tools import filesystem as _fsmod
        _fsmod.clear_turn_grants()
        self._engine.send(
            send_text,
            cb=self._engine_callback,
            attachments=media_attachments,
        )

    # ── File / log audit (chunked map-reduce) ─────────────────────────────
    def _banner(self, text: str, revealed: bool = True) -> bool:
        """Set the shared content-area banner. Returns False so it can also
        be used directly as a GLib.idle_add callback."""
        self._rag_banner.set_title(text)
        self._rag_banner.set_revealed(revealed)
        return False

    def _maybe_start_audit(self, text: str) -> bool:
        """If ``text`` is an audit request naming a readable workspace file,
        kick off the audit and return True. Otherwise return False so the
        caller continues with a normal send."""
        if self._auditing:
            self._show_toast("An audit is already running — please wait.")
            return True  # swallow the send while one is in flight
        if not text or not text.strip():
            return False
        conv_id = self._current_conv_id
        # Gated on the filesystem tool: same opt-in, workspace scope, no-root.
        if not self.effective_tool_enabled(conv_id, "filesystem"):
            return False
        if not audit.is_audit_request(text):
            return False
        abs_path = self._resolve_audit_path(text)
        if abs_path is None:
            return False  # no readable file named → let the model handle it
        focus = audit.resolve_focus(text)

        # Record the user turn exactly like a normal send.
        self._db.add_message(conv_id, "user", text)
        self._chat_view.append_user_message(text)
        msgs = self._db.list_messages(conv_id)
        if len([m for m in msgs if m.role == "user"]) == 1:
            title = (text[:60].strip().splitlines() or ["Audit"])[0]
            self._db.rename_conversation(conv_id, title)
            self._sidebar.refresh(select_id=conv_id)
        self._recompute_context_base()

        self._auditing = True
        self._audit_conv_id = conv_id
        self._audit_name = abs_path.name
        self._audit_start = time.monotonic()
        self._chat_view.set_generating(True)
        self._banner(f"Auditing {abs_path.name}: reading…", True)
        self._engine.audit_file(
            str(abs_path),
            focus,
            cb=self._on_audit_done,
            on_progress=self._on_audit_progress,
            user_text=text,
            max_chunks=self._settings.audit_max_chunks,
        )
        return True

    def _resolve_audit_path(self, text: str):
        """First readable file named in ``text`` that resolves inside the
        filesystem-tool workspace, or None. Reuses the tool's own path
        boundary so the audit can't read outside the workspace."""
        from .tools.filesystem import resolve_within
        root = Path(
            self._settings.tool_fs_root or "~/Documents/box-workspace"
        ).expanduser()
        for tok in audit.extract_path_tokens(text):
            full = resolve_within(root, tok)
            if full is not None and full.is_file():
                return full
        return None

    def _on_audit_progress(self, done: int, total: int, phase: str) -> None:
        """Worker-thread progress callback → bounce to the banner."""
        GLib.idle_add(self._update_audit_banner, done, total, phase)

    def _update_audit_banner(self, done: int, total: int, phase: str) -> bool:
        name = self._audit_name
        if phase == "report":
            label = (
                f"Auditing {name}: writing report ({done}/{total})…"
                if total > 1 else f"Auditing {name}: writing report…"
            )
            self._banner(label, True)
            return False
        eta = ""
        elapsed = time.monotonic() - self._audit_start
        if done > 0 and total > done and elapsed > 0:
            remain = int(elapsed / done * (total - done))
            if remain >= 1:
                m, s = divmod(remain, 60)
                eta = f" (~{m}m {s:02d}s left)" if m else f" (~{s}s left)"
        self._banner(f"Auditing {name}: section {done}/{total}{eta}", True)
        return False

    def _on_audit_done(self, report: str | None, error: str | None) -> None:
        """Worker-thread completion callback → bounce to the main thread."""
        GLib.idle_add(self._finish_audit, report, error)

    def _finish_audit(self, report: str | None, error: str | None) -> bool:
        self._auditing = False
        conv_id = self._audit_conv_id
        self._audit_conv_id = None
        self._chat_view.set_generating(False)
        self._chat_view.focus_input()
        self._banner("", False)
        if error:
            self._show_error_dialog("Audit failed", error)
            return False
        if not report:
            self._show_toast("Audit produced no report.")
            return False
        if conv_id is not None:
            self._db.add_message(conv_id, "assistant", report)
            if conv_id == self._current_conv_id:
                self._chat_view.add_assistant_message(report)
            self._sidebar.refresh(select_id=self._current_conv_id)
            self._recompute_context_base()
        return False

    def _on_user_stop(self) -> None:
        self._engine.stop()

    # ── Context menu ──────────────────────────────────────────────────────

    def _on_rename_conv(self, _action, param: GLib.Variant) -> None:
        conv_id = param.get_int32()
        conv = self._db.get_conversation(conv_id)
        if conv is None:
            return
        dlg = Adw.AlertDialog(heading="Rename chat", body="Enter a new title.")
        entry = Gtk.Entry(text=conv.title)
        entry.set_margin_top(8)
        entry.set_activates_default(True)
        dlg.set_extra_child(entry)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("ok", "Rename")
        dlg.set_response_appearance("ok", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("ok")
        def on_response(_d, r):
            if r == "ok":
                self._db.rename_conversation(conv_id, entry.get_text().strip() or "Untitled")
                self._sidebar.refresh(select_id=conv_id)
        dlg.connect("response", on_response)
        dlg.present(self)

    def _on_delete_conv(self, _action, param: GLib.Variant) -> None:
        conv_id = param.get_int32()
        dlg = Adw.AlertDialog(
            heading="Delete this chat?",
            body="Permanently deletes the conversation and all its messages.",
        )
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("delete", "Delete")
        dlg.set_response_appearance("delete", Adw.ResponseAppearance.DESTRUCTIVE)
        dlg.set_default_response("cancel")
        def on_response(_d, r):
            if r == "delete":
                # Abort any background generation tied to this conv —
                # otherwise EvtComplete/EvtStopped would try to persist
                # into a row that no longer exists (FK violation). The
                # SDK still holds the doomed conv, so we also force a
                # reload so the next send doesn't run against it.
                gen_was_here = self._active_gen_conv_id == conv_id
                if gen_was_here:
                    self._engine.stop()
                    self._clear_active_gen()
                    self._set_agent_progress(0, None)
                self._db.delete_conversation(conv_id)
                # Drop the conversation's RAG chunks too — orphaned rows
                # would otherwise stay in rag_index.db indefinitely.
                try:
                    self._rag.clear_conversation(conv_id)
                except Exception:
                    log.exception("RAG cleanup failed for conv %s", conv_id)
                if self._current_conv_id == conv_id:
                    self._current_conv_id = None
                    self._chat_view.clear()
                    self._refresh_kb_panel()
                self._sidebar.refresh()
                if not self._db.list_conversations():
                    self.start_new_chat()
                elif gen_was_here and self._current_conv_id is not None:
                    # Force the SDK to drop the now-deleted conv before
                    # the user sends in whatever chat they're on.
                    self._reload_engine_for_current_conv(force=True)
        dlg.connect("response", on_response)
        dlg.present(self)

    # ── Engine event routing ───────────────────────────────────────────────

    def _engine_callback(self, evt: Event) -> None:
        GLib.idle_add(self._handle_engine_event, evt)

    def _handle_engine_event(self, evt: Event) -> bool:
        if isinstance(evt, EvtLoading):
            name = Path(evt.model_path).name
            self._model_status_label.set_text(f"Loading {name}…")
            self._chat_view.set_input_sensitive(False)

        elif isinstance(evt, EvtReady):
            name = Path(evt.model_path).name
            self._model_status_label.set_text(name)
            self._gen_start = None
            self._gen_tokens = 0
            self._model_loading = False
            # Backend capability — the tools badge tooltip tells the truth
            # when the loaded model's engine can't run tools (GGUF w/o tools).
            self._backend_supports_tools = evt.supports_tools
            self.refresh_tools_badge()
            if not evt.supports_tools and self.effective_enabled_tool_ids(
                self._current_conv_id
            ):
                self._show_toast(
                    "Tools are enabled but this model's engine (llama.cpp) "
                    "doesn't support tool calling yet — chat works normally."
                )
            # Load succeeded — restore real settings (re-saves GPU/spec-decoding
            # after the crash-safe CPU copy we may have written before loading).
            self._settings.save()
            self._chat_view.set_input_sensitive(True)
            self._chat_view.focus_input()

        elif isinstance(evt, EvtToken):
            if self._gen_start is None:
                self._gen_start = time.monotonic()
                self._gen_tokens = 0
            self._gen_tokens += 1
            elapsed = time.monotonic() - self._gen_start
            if elapsed > 0 and self._gen_tokens % 5 == 0:
                tps = self._gen_tokens / elapsed
                self._model_status_label.set_text(
                    f"{Path(self._settings.model_path).name}  ·  {tps:.1f} tok/s"
                )
            # Always accumulate into the originating gen's buffer so we can
            # persist on completion and restore the bubble on chat switch.
            if self._active_gen_conv_id is not None:
                self._active_gen_text += evt.text
            # Render live + speak only when the user is still viewing the
            # originating chat. Otherwise the token is "background" — saved
            # to _active_gen_text but invisible until they switch back.
            if self._active_gen_conv_id == self._current_conv_id:
                self._chat_view.append_token(evt.text)
                if self._tts_streaming:
                    self._tts.push_chunk(evt.text)

        elif isinstance(evt, EvtComplete):
            self._finish_tps()
            gen_conv = self._active_gen_conv_id
            text = self._active_gen_text
            viewing_gen = (gen_conv is not None
                           and gen_conv == self._current_conv_id)
            if viewing_gen:
                # User is still on the originating chat — finalise the
                # visible bubble (this also rebuilds it as Markdown/LaTeX).
                finished = self._chat_view.finish_stream()
                if finished:
                    text = finished
            if gen_conv is not None and text:
                ctx_json = self._active_gen_context_json()
                self._db.add_message(
                    gen_conv, "assistant", text, context_json=ctx_json,
                )
                # Preserve the user's CURRENT selection — don't yank focus
                # back to the gen conv if they've switched away.
                self._sidebar.refresh(select_id=self._current_conv_id)
            if viewing_gen:
                if self._tts_streaming:
                    self._tts.finish_stream()  # flush + drain
                elif (self._settings.enable_tts and self._settings.tts_auto_speak
                        and text and not self._tts_user_stopped):
                    # Fallback: streaming wasn't initialised for this gen
                    # (voice not ready at send time). Skip if the user
                    # explicitly pressed stop earlier — otherwise stopping
                    # mid-stream would relaunch the whole reply when
                    # generation finishes.
                    self._on_speak(text)
                self._chat_view.focus_input()
            else:
                # User switched away — drop any in-flight TTS so the reply
                # for a now-hidden chat doesn't keep talking.
                if self._tts_streaming:
                    self._tts_streaming = False
                    self._tts_speak_gen += 1
                    self._tts.stop()
            self._recompute_context_base()
            self._clear_active_gen()
            self._set_agent_progress(0, None)
            # If the user switched chats during this generation, the SDK
            # still holds the gen's conv — rebuild for whatever chat
            # they're on now so the next send has the right context.
            if gen_conv is not None and gen_conv != self._current_conv_id:
                self._reload_engine_for_current_conv(force=True)

        elif isinstance(evt, EvtStopped):
            self._finish_tps()
            gen_conv = self._active_gen_conv_id
            text = self._active_gen_text
            viewing_gen = (gen_conv is not None
                           and gen_conv == self._current_conv_id)
            if viewing_gen:
                finished = self._chat_view.finish_stream()
                if finished:
                    text = finished
            if gen_conv is not None and text:
                ctx_json = self._active_gen_context_json()
                self._db.add_message(
                    gen_conv, "assistant", text + " [stopped]",
                    context_json=ctx_json,
                )
                self._sidebar.refresh(select_id=self._current_conv_id)
            self._recompute_context_base()
            if self._tts_streaming:
                self._tts_streaming = False
                self._tts_speak_gen += 1
                self._tts.stop()
                if viewing_gen:
                    self._chat_view.set_tts_speaking(False)
            self._show_toast("Generation stopped")
            self._clear_active_gen()
            self._set_agent_progress(0, None)
            if gen_conv is not None and gen_conv != self._current_conv_id:
                self._reload_engine_for_current_conv(force=True)

        elif isinstance(evt, EvtError):
            gen_conv = self._active_gen_conv_id
            if gen_conv is not None and gen_conv == self._current_conv_id:
                self._chat_view.finish_stream()
            self._clear_active_gen()
            self._set_agent_progress(0, None)
            self._chat_view.set_input_sensitive(True)
            self._model_status_label.set_text("Error")
            # If the failed send belonged to a chat the user no longer
            # has on screen, queue a reload so the SDK doesn't keep a
            # half-broken conv around for the wrong chat. (gen_conv is
            # always None for load errors — no need to special-case.)
            if gen_conv is not None and gen_conv != self._current_conv_id:
                self._reload_engine_for_current_conv(force=True)
            msg = evt.message or ""
            # A load error can arrive when _model_loading is already False
            # (e.g. a debounced retry fires a second EvtError) — the
            # structured kind covers that for both backends without
            # string-matching engine text.
            is_load_error = (
                self._model_loading
                or getattr(evt, "kind", "error") == "load_failed"
            )
            self._model_loading = False

            if is_load_error:
                # "backend constraint mismatch" only appears in C++ stderr,
                # not in the Python exception. Detect GPU-required models by
                # checking that we're on CPU and the error is a load failure —
                # that pattern reliably means the model needs GPU.
                on_cpu = self._settings.backend == "cpu"
                if on_cpu and "failed to create" in msg.lower():
                    self._show_error_dialog(
                        "Model requires GPU",
                        "This model only runs on a dedicated GPU and cannot be used on CPU.\n\n"
                        "Recommended: choose a CPU-compatible model such as Gemma 4 E2B or E4B "
                        "(Preferences → Model → Download models).\n\n"
                        "If you have a dedicated NVIDIA or AMD GPU, you can switch the "
                        "Inference backend to GPU in Preferences → Model. "
                        "Do not switch to GPU if you only have integrated graphics — "
                        "it will crash the app.",
                    )
                    return False

                # Non-GPU load failure — reset risky settings so the next
                # startup doesn't try GPU/spec-decoding and loop.
                was_gpu = self._settings.backend != "cpu"
                self._settings.backend = "cpu"
                self._settings.enable_speculative_decoding = False
                self._settings.save()
                suffix = (
                    "\n\nBackend has been reset to CPU. If you were using GPU or "
                    "Speculative Decoding, your hardware may not support it — "
                    "disable it in Preferences → Model."
                    if was_gpu else ""
                )
                self._show_error_dialog("Failed to load model", msg + suffix)
            else:
                # Title by engine — a llama.cpp failure labelled "LiteRT-LM
                # error" sends the user debugging the wrong backend.
                is_gguf = self._settings.model_path.lower().endswith(".gguf")
                self._show_error_dialog(
                    "Llama.cpp error" if is_gguf else "LiteRT-LM error", msg
                )

        return False

    # ── Helpers ────────────────────────────────────────────────────────────

    def _finish_tps(self) -> None:
        if self._gen_start is not None and self._gen_tokens > 0:
            elapsed = time.monotonic() - self._gen_start
            tps = self._gen_tokens / elapsed if elapsed > 0 else 0
            name = Path(self._settings.model_path).name
            self._model_status_label.set_text(f"{name}  ·  {tps:.1f} tok/s")
        self._gen_start = None
        self._gen_tokens = 0

    # ── App Lock (2026-07-14) ───────────────────────────────────────────
    def is_locked(self) -> bool:
        return getattr(self, "_locked", False)

    def lock_now(self) -> None:
        """Swap the window content for a password gate. No-op if no password
        is set or already locked. NOT encryption — just a UI gate."""
        if self.is_locked() or not self._settings.app_lock_hash:
            return
        self._locked = True
        self._toast_overlay.set_child(self._build_lock_screen())
        # App Lock must cover the WHOLE app: the Box Code and Image Tools
        # windows are separate toplevels and would otherwise stay usable
        # (agent + filesystem access!) behind the lock screen.
        app = self.get_application()
        if app is not None and hasattr(app, "hide_aux_windows"):
            app.hide_aux_windows()

    def _build_lock_screen(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=14,
                      halign=Gtk.Align.CENTER, valign=Gtk.Align.CENTER)
        title = Gtk.Label(label="Box is locked")
        title.add_css_class("title-1")
        box.append(title)
        entry = Gtk.PasswordEntry(show_peek_icon=True, width_request=260)
        entry.set_property("placeholder-text", "Password")
        box.append(entry)
        err = Gtk.Label()
        err.add_css_class("error")
        box.append(err)
        btn = Gtk.Button(label="Unlock")
        btn.add_css_class("suggested-action")
        btn.add_css_class("pill")
        box.append(btn)

        def _try_unlock(*_a) -> None:
            from .applock import verify_password
            if verify_password(self._settings.app_lock_hash, entry.get_text()):
                self._locked = False
                self._toast_overlay.set_child(self._sidebar_split)
            else:
                err.set_text("Incorrect password")
                entry.set_text("")

        btn.connect("clicked", _try_unlock)
        entry.connect("activate", _try_unlock)
        return box

    def _show_toast(self, msg: str, timeout: int = 4) -> None:
        toast = Adw.Toast(title=msg, timeout=timeout)
        self._toast_overlay.add_toast(toast)

    def _show_error_dialog(self, heading: str, body: str) -> None:
        dlg = Adw.AlertDialog(heading=heading, body=body)
        dlg.add_response("ok", "OK")
        dlg.set_default_response("ok")
        dlg.present(self)


# ── File helpers ──────────────────────────────────────────────────────────────

def _extract_pdf(path: str) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p for p in pages if p.strip())
    except Exception as e:
        log.warning("PDF extraction failed: %s", e)
        return ""
