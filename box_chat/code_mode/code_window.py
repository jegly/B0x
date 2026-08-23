"""Box Code — the standalone agent window (the only gi module in code_mode).

One window, singleton per app. Left: persisted sessions (click to resume).
Right: transcript (streamed assistant text, collapsible tool rows, pinned
todo list) + composer. Header: project + model + permission mode + Stop.
A status strip reports the sandbox that actually ran the last command —
honestly, including whether network was really blocked on this machine.

All agent callbacks arrive on the agent worker thread and bounce to the
GTK main thread via ``GLib.idle_add`` (same contract as MainWindow).
"""
from __future__ import annotations

import logging
from pathlib import Path

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk, Pango  # noqa: E402

from ..config import MODELS_DIR, Settings
from .agent_loop import CodeAgent, CodeAgentCallbacks
from .agent_tools import network_blocked
from .sessions import CodeSession, list_sessions

log = logging.getLogger(__name__)

_MODE_LABELS = ("Ask before risky tools", "Auto-approve (sandboxed)")
_MODE_IDS = ("ask", "auto")

# Claude Code-style working verbs — cycled while the agent thinks.
_WORK_VERBS = (
    "Pondering", "Scheming", "Brewing", "Noodling", "Percolating",
    "Cogitating", "Tinkering", "Riffling through files", "Untangling",
    "Conjuring", "Sanity-checking", "Herding electrons", "Mulling it over",
    "Connecting dots", "Sharpening pencils", "Consulting the rubber duck",
    "Warming neurons", "Plotting", "Assembling thoughts", "Crunching",
)


class CodeWindow(Adw.Window):
    """The Box Code workspace window."""

    def __init__(self, application: Adw.Application, settings: Settings) -> None:
        super().__init__(
            application=application,
            title="Box Code",
            default_width=1150,
            default_height=760,
        )
        self._settings = settings
        self._agent: CodeAgent | None = None
        self._session: CodeSession | None = None
        self._project_dir: str = settings.code_project_dir or ""
        self._model_path: str = settings.code_model_path or ""
        self._assistant_buf: list[str] = []
        self._assistant_label: Gtk.Label | None = None
        self._stream_flush_id: int = 0  # 33 ms streaming repaint throttle
        # Working-status state (spinner + cycling verbs + elapsed + tools).
        self._work_timer: int = 0
        self._work_started: float = 0.0
        self._work_verb: str = ""
        self._work_verb_at: float = 0.0
        self._work_tools: int = 0
        self._work_last_tool: str = ""
        # Messages typed while the agent is busy — sent when it frees up.
        self._queued: list[str] = []
        self._todo_label: Gtk.Label | None = None

        self.add_css_class("aux-solid")  # opaque under glass modes
        self._build_ui()
        self._refresh_sessions()
        self._refresh_header()
        self.connect("close-request", self._on_close_request)
        kc = Gtk.ShortcutController()
        kc.add_shortcut(Gtk.Shortcut.new(
            Gtk.ShortcutTrigger.parse_string("<Primary>w"),
            Gtk.CallbackAction.new(lambda *_: bool(self.close()) or True),
        ))
        kc.add_shortcut(Gtk.Shortcut.new(
            Gtk.ShortcutTrigger.parse_string("Escape"),
            Gtk.CallbackAction.new(lambda *_: self._on_escape()),
        ))
        self.add_controller(kc)

    # ── UI construction ───────────────────────────────────────────────────
    def _build_ui(self) -> None:
        tv = Adw.ToolbarView()
        header = Adw.HeaderBar()
        self._title = Adw.WindowTitle(title="Box Code", subtitle="")
        header.set_title_widget(self._title)

        back_btn = Gtk.Button(icon_name="go-previous-symbolic")
        back_btn.set_tooltip_text("Back to Box chat")
        back_btn.connect("clicked", self._on_back_to_box)
        header.pack_start(back_btn)

        self._project_btn = Gtk.Button(label="Project…")
        self._project_btn.set_tooltip_text("Choose the project folder")
        self._project_btn.connect("clicked", self._on_pick_project)
        header.pack_start(self._project_btn)

        self._model_btn = Gtk.MenuButton(label="Model…")
        self._model_btn.set_tooltip_text(
            "Choose a model — LiteRT (.litertlm/.task) or GGUF"
        )
        # Refresh the menu whenever the window is (re)shown — NEVER while
        # the popover is opening: swapping the menu model mid-activation
        # closes the popover instantly ("clicking does nothing" bug).
        self.connect("map", lambda *_: self._rebuild_model_menu())
        header.pack_start(self._model_btn)

        self._stop_btn = Gtk.Button(label="Stop", visible=False)
        self._stop_btn.add_css_class("destructive-action")
        self._stop_btn.connect("clicked", lambda *_: self._on_stop())
        header.pack_end(self._stop_btn)

        self._mode_dd = Gtk.DropDown.new_from_strings(list(_MODE_LABELS))
        self._mode_dd.set_tooltip_text("Permission mode")
        try:
            self._mode_dd.set_selected(
                _MODE_IDS.index(self._settings.code_permission_mode)
            )
        except ValueError:
            self._mode_dd.set_selected(0)
        self._mode_dd.connect("notify::selected", self._on_mode_changed)
        header.pack_end(self._mode_dd)

        gear_btn = Gtk.Button(icon_name="emblem-system-symbolic")
        gear_btn.set_tooltip_text("Box Code settings")
        gear_btn.connect("clicked", lambda *_: self._on_settings())
        header.pack_end(gear_btn)

        tv.add_top_bar(header)

        body = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)

        # Sessions sidebar
        side = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, width_request=250)
        new_btn = Gtk.Button(label="New session")
        new_btn.set_margin_top(8)
        new_btn.set_margin_start(8)
        new_btn.set_margin_end(8)
        new_btn.connect("clicked", lambda *_: self._start_new_session())
        side.append(new_btn)
        self._session_list = Gtk.ListBox()
        self._session_list.add_css_class("navigation-sidebar")
        self._session_list.connect("row-activated", self._on_session_row)
        side_scroll = Gtk.ScrolledWindow(vexpand=True, child=self._session_list)
        side.append(side_scroll)
        body.append(side)
        body.append(Gtk.Separator(orientation=Gtk.Orientation.VERTICAL))

        # Main column
        main = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        self._transcript = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=10,
            margin_top=12, margin_bottom=12, margin_start=14, margin_end=14,
        )
        self._scroller = Gtk.ScrolledWindow(vexpand=True, child=self._transcript)
        main.append(self._scroller)

        # Working indicator (spinner + verb + elapsed + tools) — the
        # Claude Code-style "Pondering… 12s · 3 tools" line.
        self._work_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            margin_start=14, margin_end=14, visible=False,
        )
        self._work_spinner = Gtk.Spinner()
        self._work_row.append(self._work_spinner)
        self._work_label = Gtk.Label(label="", xalign=0)
        self._work_label.add_css_class("dim-label")
        self._work_row.append(self._work_label)
        main.append(self._work_row)
        self._queue_label = Gtk.Label(
            label="", xalign=0, visible=False,
            margin_start=14, margin_end=14,
        )
        self._queue_label.add_css_class("dim-label")
        main.append(self._queue_label)

        # Status strip
        status = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        status.set_margin_start(14)
        status.set_margin_end(14)
        status.set_margin_bottom(4)
        self._sandbox_badge = Gtk.Label(label="", xalign=0, hexpand=True)
        self._sandbox_badge.add_css_class("dim-label")
        self._sandbox_badge.set_ellipsize(Pango.EllipsizeMode.END)
        status.append(self._sandbox_badge)
        self._iter_label = Gtk.Label(label="")
        self._iter_label.add_css_class("dim-label")
        status.append(self._iter_label)
        main.append(status)

        # Composer
        comp = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        comp.set_margin_start(14)
        comp.set_margin_end(14)
        comp.set_margin_bottom(12)
        self._input = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR, hexpand=True,
            top_margin=8, bottom_margin=8, left_margin=8, right_margin=8,
        )
        self._input.set_size_request(-1, 64)
        input_frame = Gtk.Frame(child=self._input)
        comp.append(input_frame)
        self._send_btn = Gtk.Button(label="Send")
        self._send_btn.add_css_class("suggested-action")
        self._send_btn.set_valign(Gtk.Align.END)
        self._send_btn.connect("clicked", lambda *_: self._on_send())
        comp.append(self._send_btn)
        main.append(comp)

        kc = Gtk.EventControllerKey()
        kc.connect("key-pressed", self._on_key)
        self._input.add_controller(kc)

        body.append(main)
        tv.set_content(body)
        self.set_content(tv)
        self._rebuild_model_menu()

    # ── header state ──────────────────────────────────────────────────────
    def _refresh_header(self) -> None:
        proj = Path(self._project_dir).name if self._project_dir else "no project"
        model = Path(self._model_path).name if self._model_path else "no model"
        self._title.set_subtitle(f"{proj} · {model}")
        self._project_btn.set_label(
            Path(self._project_dir).name if self._project_dir else "Project…"
        )
        self._model_btn.set_label(
            Path(self._model_path).stem[:40] if self._model_path else "Model…"
        )

    def _rebuild_model_menu(self) -> None:
        import os
        menu = Gio.Menu()
        seen: set[str] = set()  # realpaths — kills duplicate entries
        gguf_sec = Gio.Menu()
        litert_sec = Gio.Menu()

        def offer(path: str) -> None:
            p = Path(path)
            if not p.is_file():
                return
            real = os.path.realpath(path)
            if real in seen:
                return
            seen.add(real)
            item = Gio.MenuItem.new(p.name, None)
            item.set_action_and_target_value(
                "codewin.pick-model", GLib.Variant("s", str(p))
            )
            (litert_sec if p.suffix in (".litertlm", ".task")
             else gguf_sec).append_item(item)

        for p in self._settings.recent_models:
            offer(p)
        for p in self._settings.imported_gguf_models:
            offer(p)
        for pattern in ("*.gguf", "*.litertlm", "*.task"):
            for f in sorted(MODELS_DIR.glob(pattern)):
                offer(str(f))
            try:
                for f in sorted((Path.home() / "Downloads").glob(pattern)):
                    offer(str(f))
            except OSError:
                pass
        menu.append_section("LiteRT models", litert_sec)
        menu.append_section("GGUF models", gguf_sec)
        browse = Gio.Menu()
        browse.append("Browse…", "codewin.browse-model")
        menu.append_section(None, browse)
        self._model_btn.set_menu_model(menu)

        group = Gio.SimpleActionGroup()
        pick = Gio.SimpleAction.new("pick-model", GLib.VariantType.new("s"))
        pick.connect("activate", self._on_pick_model)
        group.add_action(pick)
        browse_act = Gio.SimpleAction.new("browse-model", None)
        browse_act.connect("activate", self._on_browse_model)
        group.add_action(browse_act)
        self.insert_action_group("codewin", group)

    # ── header handlers ───────────────────────────────────────────────────
    def _on_back_to_box(self, *_a) -> None:
        app = self.get_application()
        main = getattr(app, "_main_window", None)
        if main is not None:
            main.set_visible(True)
            main.present()

    def _on_settings(self) -> None:
        """Small in-window tweaks dialog for the code_* settings."""
        dlg = Adw.AlertDialog(
            heading="Box Code settings",
            body="Iteration cap and shell timeout apply immediately. "
            "Context size and project instructions apply when the model "
            "next (re)loads — start a new session to force it.",
        )
        grid = Gtk.Grid(
            column_spacing=12, row_spacing=8,
            margin_top=8, margin_bottom=4,
        )

        def add_row(row: int, label: str, lo: int, hi: int, step: int,
                    value: int) -> Gtk.SpinButton:
            lbl = Gtk.Label(label=label, xalign=0, hexpand=True)
            spin = Gtk.SpinButton.new_with_range(lo, hi, step)
            spin.set_value(value)
            grid.attach(lbl, 0, row, 1, 1)
            grid.attach(spin, 1, row, 1, 1)
            return spin

        s = self._settings
        it_spin = add_row(0, "Tool calls per message (cap)", 10, 500, 10,
                          s.code_max_iterations)
        to_spin = add_row(1, "Shell command timeout (seconds)", 10, 600, 10,
                          s.code_bash_timeout)
        ctx_spin = add_row(2, "Context size (tokens)", 2048, 131072, 1024,
                           s.code_max_context)
        agents_lbl = Gtk.Label(
            label="Read AGENTS.md / CLAUDE.md from the project",
            xalign=0, hexpand=True,
        )
        agents_sw = Gtk.Switch(
            active=s.code_read_agents_md, halign=Gtk.Align.END
        )
        grid.attach(agents_lbl, 0, 3, 1, 1)
        grid.attach(agents_sw, 1, 3, 1, 1)
        web_lbl = Gtk.Label(
            label="Web research (web_search + fetch_url; shell stays offline)",
            xalign=0, hexpand=True,
        )
        web_sw = Gtk.Switch(
            active=s.code_web_enabled, halign=Gtk.Align.END
        )
        grid.attach(web_lbl, 0, 4, 1, 1)
        grid.attach(web_sw, 1, 4, 1, 1)
        temp_lbl = Gtk.Label(
            label="Temperature (−1 = use the chat setting; 0 = precise)",
            xalign=0, hexpand=True,
        )
        temp_spin = Gtk.SpinButton.new_with_range(-1.0, 2.0, 0.05)
        temp_spin.set_digits(2)
        temp_spin.set_value(s.code_temperature)
        grid.attach(temp_lbl, 0, 5, 1, 1)
        grid.attach(temp_spin, 1, 5, 1, 1)
        notify_lbl = Gtk.Label(
            label="Desktop notification when the agent finishes",
            xalign=0, hexpand=True,
        )
        notify_sw = Gtk.Switch(
            active=s.code_notify_done, halign=Gtk.Align.END
        )
        grid.attach(notify_lbl, 0, 6, 1, 1)
        grid.attach(notify_sw, 1, 6, 1, 1)
        dlg.set_extra_child(grid)
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("save", "Save")
        dlg.set_response_appearance("save", Adw.ResponseAppearance.SUGGESTED)
        dlg.set_default_response("save")
        dlg.set_close_response("cancel")

        def on_response(_d, rid: str) -> None:
            if rid != "save":
                return
            s.code_max_iterations = int(it_spin.get_value())
            s.code_bash_timeout = int(to_spin.get_value())
            s.code_max_context = int(ctx_spin.get_value())
            s.code_read_agents_md = agents_sw.get_active()
            s.code_temperature = round(temp_spin.get_value(), 2)
            s.code_notify_done = notify_sw.get_active()
            web_changed = web_sw.get_active() != s.code_web_enabled
            s.code_web_enabled = web_sw.get_active()
            s.save()
            if self._agent is not None:
                if web_changed:
                    # The tool schemas are fixed at agent start — rebuild
                    # the agent so the model actually gains/loses the tools.
                    self._teardown_agent()
                else:
                    self._agent.apply_tweaks()

        dlg.connect("response", on_response)
        dlg.present(self)

    def _on_pick_project(self, *_a) -> None:
        dlg = Gtk.FileDialog(title="Choose the project folder")
        if self._project_dir:
            try:
                dlg.set_initial_folder(
                    Gio.File.new_for_path(self._project_dir)
                )
            except Exception:  # noqa: BLE001
                pass

        def done(d, res) -> None:
            try:
                f = d.select_folder_finish(res)
            except GLib.Error:
                return
            if f and f.get_path():
                self._project_dir = f.get_path()
                self._settings.code_project_dir = self._project_dir
                self._settings.save()
                self._start_new_session()
                self._refresh_header()

        dlg.select_folder(self, None, done)

    def _on_pick_model(self, _a, param: GLib.Variant) -> None:
        self._set_model(param.get_string())

    def _on_browse_model(self, *_a) -> None:
        dlg = Gtk.FileDialog(title="Choose a model")
        f = Gtk.FileFilter(name="Models (*.litertlm, *.task, *.gguf)")
        for pat in ("*.litertlm", "*.task", "*.gguf"):
            f.add_pattern(pat)
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dlg.set_filters(filters)

        def done(d, res) -> None:
            try:
                gf = d.open_finish(res)
            except GLib.Error:
                return
            if gf and gf.get_path():
                self._set_model(gf.get_path())

        dlg.open(self, None, done)

    def _set_model(self, path: str) -> None:
        if not Path(path).is_file():
            return
        self._model_path = path
        self._settings.code_model_path = path
        self._settings.save()
        # Model changes apply to the NEXT agent start: drop the live one.
        self._teardown_agent()
        if self._session is not None:
            self._session.update_model_path(path)
        self._refresh_header()

    def _on_mode_changed(self, *_a) -> None:
        mode = _MODE_IDS[min(self._mode_dd.get_selected(), 1)]
        self._settings.code_permission_mode = mode
        self._settings.save()
        if self._agent is not None:
            self._agent.set_permission_mode(mode)

    # ── sessions ──────────────────────────────────────────────────────────
    def _refresh_sessions(self) -> None:
        while (row := self._session_list.get_row_at_index(0)) is not None:
            self._session_list.remove(row)
        for meta in list_sessions():
            label = Gtk.Label(
                label=meta.title or "(untitled)", xalign=0, hexpand=True,
                margin_top=6, margin_bottom=6, margin_start=8,
            )
            label.set_ellipsize(Pango.EllipsizeMode.END)
            label.set_tooltip_text(
                f"{meta.created}\n{meta.project_dir}\n{meta.model_path}"
            )
            trash = Gtk.Button(icon_name="user-trash-symbolic")
            trash.add_css_class("flat")
            trash.set_tooltip_text("Delete this session")
            trash.connect(
                "clicked", self._on_delete_session, meta.session_id
            )
            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            box.append(label)
            box.append(trash)
            row = Gtk.ListBoxRow(child=box)
            row._session_id = meta.session_id  # type: ignore[attr-defined]
            self._session_list.append(row)

    def _on_delete_session(self, _btn, session_id: str) -> None:
        dlg = Adw.AlertDialog(
            heading="Delete this session?",
            body="Its transcript will be removed permanently.",
        )
        dlg.add_response("cancel", "Cancel")
        dlg.add_response("delete", "Delete")
        dlg.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dlg.set_close_response("cancel")

        def on_response(_d, rid: str) -> None:
            if rid != "delete":
                return
            from .sessions import delete_session
            if (
                self._session is not None
                and self._session.meta.session_id == session_id
            ):
                self._start_new_session()
            delete_session(session_id)
            self._refresh_sessions()

        dlg.connect("response", on_response)
        dlg.present(self)

    def _on_session_row(self, _list, row) -> None:
        sid = getattr(row, "_session_id", None)
        if not sid:
            return
        try:
            session = CodeSession.open(sid)
        except Exception:  # noqa: BLE001
            log.exception("could not open session %s", sid)
            return
        self._teardown_agent()
        self._session = session
        self._project_dir = session.meta.project_dir or self._project_dir
        if session.meta.model_path and Path(session.meta.model_path).is_file():
            self._model_path = session.meta.model_path
        self._clear_transcript()
        for ev in session.events():
            t = ev.get("type")
            if t == "user":
                self._add_user_row(ev.get("text", ""))
            elif t == "assistant":
                self._add_assistant_row(ev.get("text", ""), markdown=True)
            elif t == "tool":
                self._add_tool_row(
                    ev.get("name", "?"), ev.get("args", {}),
                    ev.get("result", ""), bool(ev.get("denied")),
                )
            elif t == "todo":
                self._show_todos(ev.get("text", ""))
        self._refresh_header()

    def _start_new_session(self) -> None:
        self._teardown_agent()
        self._session = None
        self._clear_transcript()
        self._refresh_header()

    # ── transcript rows (main thread only) ────────────────────────────────
    def _clear_transcript(self) -> None:
        while (child := self._transcript.get_first_child()) is not None:
            self._transcript.remove(child)
        self._assistant_label = None
        self._todo_label = None
        self._assistant_buf = []

    def _scroll_to_end(self) -> None:
        adj = self._scroller.get_vadjustment()
        GLib.idle_add(
            lambda: adj.set_value(adj.get_upper() - adj.get_page_size())
            and False
        )

    def _add_user_row(self, text: str) -> None:
        lbl = Gtk.Label(label=text, xalign=0, wrap=True, selectable=True)
        lbl.add_css_class("heading")
        self._transcript.append(lbl)
        self._scroll_to_end()

    def _add_assistant_row(self, text: str, markdown: bool = False) -> Gtk.Label:
        lbl = Gtk.Label(label=text, xalign=0, wrap=True, selectable=True)
        if markdown and text:
            self._render_markdown(lbl, text)
        self._transcript.append(lbl)
        self._scroll_to_end()
        return lbl

    @staticmethod
    def _render_markdown(lbl: Gtk.Label, text: str) -> None:
        """Upgrade a finished assistant label to rendered markdown; fall
        back to plain text on any Pango markup hiccup."""
        try:
            from ..mdrender import to_pango_markup
            lbl.set_markup(to_pango_markup(text))
        except Exception:  # noqa: BLE001
            lbl.set_text(text)

    def _add_tool_row(
        self, fn: str, args: dict, result: str, denied: bool
    ) -> None:
        # Lead with the most meaningful argument, plain text — no emoji.
        key_arg = args.get("command") or args.get("path") or args.get(
            "pattern"
        ) or ""
        summary = " ".join(str(key_arg).split())[:70]
        title = f"{fn}: {summary}" if summary else f"{fn}()"
        if denied:
            title += "  — denied"
        body = Gtk.Label(
            label=result[:4000], xalign=0, wrap=True, selectable=True,
        )
        body.add_css_class("monospace")
        body.add_css_class("dim-label")
        exp = Gtk.Expander(label=title, child=body)
        self._transcript.append(exp)
        self._scroll_to_end()

    def _show_todos(self, todos: str) -> None:
        if self._todo_label is None:
            frame_child = Gtk.Label(
                label="", xalign=0, wrap=True, selectable=True,
                margin_top=6, margin_bottom=6, margin_start=8, margin_end=8,
            )
            frame = Gtk.Frame(label="Tasks", child=frame_child)
            self._transcript.append(frame)
            self._todo_label = frame_child
        self._todo_label.set_text(todos)
        self._scroll_to_end()

    # ── working indicator ────────────────────────────────────────────────
    def _work_start(self, fixed_verb: str = "") -> None:
        import random
        import time as _t
        if not self._work_timer:
            self._work_started = _t.monotonic()
            self._work_tools = 0
            self._work_last_tool = ""
            self._work_timer = GLib.timeout_add(1000, self._work_tick)
        self._work_verb = fixed_verb or random.choice(_WORK_VERBS)
        self._work_verb_at = _t.monotonic()
        self._work_spinner.start()
        self._work_row.set_visible(True)
        self._work_refresh()

    def _work_stop(self) -> None:
        if self._work_timer:
            GLib.source_remove(self._work_timer)
            self._work_timer = 0
        self._work_spinner.stop()
        self._work_row.set_visible(False)

    def _work_tick(self) -> bool:
        import random
        import time as _t
        # A fresh verb every ~5s keeps it alive without being noisy.
        if _t.monotonic() - self._work_verb_at > 5:
            self._work_verb = random.choice(_WORK_VERBS)
            self._work_verb_at = _t.monotonic()
        self._work_refresh()
        return True

    def _work_refresh(self) -> None:
        import time as _t
        secs = int(_t.monotonic() - self._work_started)
        parts = [f"{self._work_verb}… {secs}s"]
        if self._work_tools:
            plural = "s" if self._work_tools != 1 else ""
            parts.append(f"{self._work_tools} tool{plural}")
        if self._work_last_tool:
            parts.append(f"last: {self._work_last_tool}")
        parts.append("Esc to interrupt")
        self._work_label.set_text("  ·  ".join(parts))

    def _on_escape(self) -> bool:
        if self._agent is not None and self._agent.state in (
            "running", "loading"
        ):
            self._agent.stop()
            return True
        return False

    def _update_queue_label(self) -> None:
        n = len(self._queued)
        if not n:
            self._queue_label.set_visible(False)
            return
        head = " ".join(self._queued[0].split())[:60]
        self._queue_label.set_text(
            f"{n} message{'s' if n > 1 else ''} queued — sending when the "
            f"agent finishes (next: {head}…)")
        self._queue_label.set_visible(True)

    # ── send / agent lifecycle ────────────────────────────────────────────
    def _on_key(self, _c, keyval, _code, state) -> bool:
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter) and (
            state & Gtk.accelerator_get_default_mod_mask()
        ) == Gdk.ModifierType.CONTROL_MASK:
            self._on_send()
            return True
        return False

    def _on_send(self) -> None:
        buf = self._input.get_buffer()
        text = buf.get_text(
            buf.get_start_iter(), buf.get_end_iter(), False
        ).strip()
        if not text:
            return
        if not self._project_dir or not Path(self._project_dir).is_dir():
            self._toast("Choose a project folder first.")
            return
        if not self._model_path or not Path(self._model_path).is_file():
            self._toast("Choose a GGUF model first.")
            return
        if self._agent is not None and self._agent.state in (
            "running", "loading"
        ):
            # Claude Code/opencode habit: type ahead while it works.
            self._queued.append(text)
            buf.set_text("")
            self._update_queue_label()
            return
        buf.set_text("")
        self._dispatch(text)

    def _dispatch(self, text: str) -> None:
        try:
            agent = self._ensure_agent()
        except Exception as e:  # noqa: BLE001
            log.exception("could not start agent")
            self._toast(f"Could not start the agent: {e}")
            return
        self._add_user_row(text)
        self._assistant_buf = []
        self._assistant_label = None
        agent.send(text)
        self._refresh_sessions_later()

    def _ensure_agent(self) -> CodeAgent:
        if self._agent is not None:
            return self._agent
        # RAM guard: one big model at a time on this machine. If the chat
        # side is holding one, free it before our llama-server loads.
        app = self.get_application()
        engine = getattr(app, "engine", None)
        if engine is not None and getattr(engine, "is_ready", False):
            try:
                engine.unload_model()
                self._sandbox_badge.set_text(
                    "Unloaded the chat model to free RAM "
                    "(chat reloads it when you go back)."
                )
            except Exception:  # noqa: BLE001
                log.exception("chat model unload failed")
        if self._session is None:
            self._session = CodeSession.create(
                self._project_dir, self._model_path
            )
        elif self._session.meta.model_path != self._model_path:
            self._session.update_model_path(self._model_path)
        cb = CodeAgentCallbacks(
            on_state=self._cb_state,
            on_token=self._cb_token,
            on_tool_event=self._cb_tool_event,
            on_progress=self._cb_progress,
            on_todo=self._cb_todo,
            on_turn_done=self._cb_turn_done,
            on_error=self._cb_error,
            ask_user=self._cb_ask_user,
            ask_permission=self._cb_ask_permission,
        )
        self._agent = CodeAgent(self._settings, self._session, cb)
        return self._agent

    def _teardown_agent(self) -> None:
        if self._agent is not None:
            agent = self._agent
            self._agent = None
            agent.shutdown(timeout=10)

    def _on_stop(self) -> None:
        if self._agent is not None:
            self._agent.stop()

    def _on_close_request(self, *_a) -> bool:
        if self._stream_flush_id:
            GLib.source_remove(self._stream_flush_id)
            self._stream_flush_id = 0
        self._work_stop()
        self._teardown_agent()
        return False  # proceed with close

    def _refresh_sessions_later(self) -> None:
        GLib.timeout_add(600, lambda: (self._refresh_sessions(), False)[1])

    def _toast(self, text: str) -> None:
        dlg = Adw.AlertDialog(heading="Box Code", body=text)
        dlg.add_response("ok", "OK")
        dlg.present(self)

    # ── agent callbacks (worker thread → idle_add) ────────────────────────
    def _cb_state(self, state: str, detail: str) -> None:
        def ui() -> bool:
            busy = state in ("loading", "running")
            self._stop_btn.set_visible(busy)
            self._send_btn.set_label("Queue" if busy else "Send")
            if state == "loading":
                self._work_start(fixed_verb=f"Loading {detail}")
                self._sandbox_badge.set_text(
                    f"Loading {detail}… (large models can take several "
                    "minutes on this machine — don't close the window)"
                )
            elif state == "running":
                self._work_start()
            elif state in ("ready", "stopped", "error"):
                self._work_stop()
                self._update_sandbox_badge()
            # Auto-send anything queued while it was busy (only on a clean
            # finish — after Stop or an error the user should decide).
            if state == "ready" and self._queued:
                nxt = self._queued.pop(0)
                self._update_queue_label()
                self._dispatch(nxt)
            return False
        GLib.idle_add(ui)

    def _update_sandbox_badge(self) -> None:
        agent = self._agent
        if agent is None:
            self._sandbox_badge.set_text("")
            return
        parts: list[str] = []
        rep = agent.sandbox_report
        if rep is not None:
            parts.append(f"model: {rep.summary()}")
        elif agent.backend_kind == "litert":
            parts.append("model: LiteRT (in-process)")
        brep = agent.toolbox.last_bash_report
        if brep is not None:
            net = "blocked" if network_blocked(brep) else "NOT blocked"
            parts.append(f"shell: {brep.summary()} · network {net}")
        if agent.toolbox.web_enabled:
            parts.append("web research: ON")
        self._sandbox_badge.set_text("  |  ".join(parts))

    def _cb_token(self, token: str) -> None:
        def ui() -> bool:
            if self._assistant_label is None:
                self._assistant_label = self._add_assistant_row("")
            self._assistant_buf.append(token)
            # Repaint at most every 33 ms — per-token full-join set_text
            # is O(n²) and floods the main loop on long generations.
            if not self._stream_flush_id:
                self._stream_flush_id = GLib.timeout_add(
                    33, self._flush_stream
                )
            return False
        GLib.idle_add(ui)

    def _flush_stream(self) -> bool:
        self._stream_flush_id = 0
        if self._assistant_label is not None:
            self._assistant_label.set_text("".join(self._assistant_buf))
            self._scroll_to_end()
        return False

    def _flush_stream_now(self) -> None:
        if self._stream_flush_id:
            GLib.source_remove(self._stream_flush_id)
        self._flush_stream()

    def _cb_tool_event(
        self, fn: str, args: dict, result: str, denied: bool
    ) -> None:
        def ui() -> bool:
            # Tokens streamed before this tool call belong to the reasoning
            # segment — paint any pending tail, then freeze that label so
            # the next text starts fresh.
            self._flush_stream_now()
            self._assistant_label = None
            self._assistant_buf = []
            self._add_tool_row(fn, args, result, denied)
            self._update_sandbox_badge()
            if not denied:
                import random
                self._work_tools += 1
                self._work_last_tool = fn
                self._work_verb = random.choice(_WORK_VERBS)
                self._work_refresh()
            return False
        GLib.idle_add(ui)

    def _cb_progress(self, current: int, maximum: int | None) -> None:
        def ui() -> bool:
            self._iter_label.set_text(
                f"tools: {current}/{maximum}" if maximum else f"tools: {current}"
            )
            return False
        GLib.idle_add(ui)

    def _cb_todo(self, todos: str) -> None:
        GLib.idle_add(lambda: (self._show_todos(todos), False)[1])

    def _cb_turn_done(self, text: str, completed: bool) -> None:
        def ui() -> bool:
            # Paint the streamed tail first (a pending repaint after the
            # markdown render below would clobber markup with raw text).
            self._flush_stream_now()
            if completed and self._assistant_label is not None:
                self._render_markdown(self._assistant_label, text)
            if not completed:
                self._add_assistant_row("Stopped.")
            self._refresh_sessions()
            # Desktop ping when the user is elsewhere — long local runs.
            if (
                completed
                and getattr(self._settings, "code_notify_done", True)
                and not self.is_active()
            ):
                try:
                    n = Gio.Notification.new("Box Code — task finished")
                    body = " ".join(text.split())[:120]
                    if body:
                        n.set_body(body)
                    self.get_application().send_notification(
                        "box-code-done", n
                    )
                except Exception:  # noqa: BLE001 — notify is best-effort
                    log.debug("notification failed", exc_info=True)
            return False
        GLib.idle_add(ui)

    def _cb_error(self, message: str) -> None:
        def ui() -> bool:
            lbl = self._add_assistant_row(f"Error: {message}")
            lbl.add_css_class("error")
            return False
        GLib.idle_add(ui)

    def _cb_ask_user(self, question: str, on_answer) -> None:
        def ui() -> bool:
            dlg = Adw.AlertDialog(
                heading="The agent has a question", body=question
            )
            entry = Gtk.Entry(placeholder_text="Your answer…")
            dlg.set_extra_child(entry)
            dlg.add_response("skip", "Skip")
            dlg.add_response("answer", "Answer")
            dlg.set_response_appearance(
                "answer", Adw.ResponseAppearance.SUGGESTED
            )
            dlg.set_default_response("answer")
            dlg.set_close_response("skip")

            def on_response(_d, rid: str) -> None:
                on_answer(
                    entry.get_text()
                    if rid == "answer"
                    else "(the user skipped the question — use your judgment)"
                )

            dlg.connect("response", on_response)
            dlg.present(self)
            return False
        GLib.idle_add(ui)

    def _cb_ask_permission(self, fn_name: str, args: dict, on_answer) -> None:
        def ui() -> bool:
            if fn_name == "bash":
                heading = "Run this command?"
                body = str(args.get("command", ""))[:600]
            elif fn_name == "edit_file":
                heading = f"Edit {str(args.get('path', ''))[:80]}?"
                old = str(args.get("old_string", ""))[:400]
                new = str(args.get("new_string", ""))[:400]
                body = f"Replace:\n{old}\n\nWith:\n{new}"
            elif fn_name == "write_file":
                content = str(args.get("content", ""))
                heading = f"Write {str(args.get('path', ''))[:80]}?"
                body = (
                    f"{len(content)} characters. Starts with:\n"
                    + content[:400]
                )
            elif fn_name == "web_search":
                heading = "Search the web?"
                body = str(args.get("query", ""))[:300]
            elif fn_name == "fetch_url":
                heading = "Fetch this page?"
                body = str(args.get("url", ""))[:300]
            else:
                heading = f"Allow {fn_name}?"
                body = str(args.get("path", ""))[:300]
            dlg = Adw.AlertDialog(heading=heading, body=body)
            dlg.add_response("deny", "Deny")
            dlg.set_response_appearance(
                "deny", Adw.ResponseAppearance.DESTRUCTIVE
            )
            dlg.add_response("once", "Allow once")
            dlg.set_response_appearance(
                "once", Adw.ResponseAppearance.SUGGESTED
            )
            dlg.add_response("session", "Allow for this session")
            dlg.set_default_response("once")
            dlg.set_close_response("deny")
            dlg.connect(
                "response", lambda _d, rid: on_answer(rid)
            )
            dlg.present(self)
            return False
        GLib.idle_add(ui)

