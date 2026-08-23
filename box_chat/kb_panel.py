"""Knowledge Base panel — right-side pane listing indexed sources per chat.

Public API the window uses:

  panel.set_conversation(conv_id, rag_override, is_global_rag_on)
      Reload the sources list and sync the per-chat RAG switch.

  panel.refresh_sources()
      Re-query the vector store for the current conversation and redraw.

Callbacks (set via constructor):
  on_add_files(list[str])     — user picked / dropped one or more file paths
  on_remove_source(str)       — user clicked the X on a source row
  on_rag_override_changed(int|None) — None=follow global, 0=off, 1=on
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, GObject, Gtk  # noqa: E402


# Tri-state option presented in the per-chat RAG dropdown.
# Kept short so the ComboRow value column doesn't truncate on a narrow pane.
_OPTION_LABELS = ["Follow global", "Always on", "Always off"]
_OPTION_TO_OVERRIDE = {0: None, 1: 1, 2: 0}
_OVERRIDE_TO_OPTION = {None: 0, 1: 1, 0: 2}


class KbPanel(Gtk.Box):
    def __init__(
        self,
        list_sources_for: Callable[[int], list[tuple[str | None, str | None, int]]],
        on_add_files: Callable[[list[str]], None],
        on_remove_source: Callable[[str], None],
        on_rag_override_changed: Callable[[int | None], None],
        # Phase 3 — notebooks integration. Optional so the panel still
        # works without notebook-aware callbacks (defensive).
        list_attached_notebooks: Callable[[int], list] | None = None,
        list_all_notebooks: Callable[[], list] | None = None,
        on_detach_notebook: Callable[[int], None] | None = None,
        on_attach_notebook: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("kb-panel")
        self._list_sources_for = list_sources_for
        self._on_add_files = on_add_files
        self._on_remove_source = on_remove_source
        self._on_rag_override_changed = on_rag_override_changed
        self._list_attached_notebooks = list_attached_notebooks
        self._list_all_notebooks = list_all_notebooks
        self._on_detach_notebook = on_detach_notebook
        self._on_attach_notebook = on_attach_notebook
        self._current_conv_id: int | None = None
        self._suppress_override_signal = False

        # ── Header: title + per-chat RAG control ──────────────────────────
        header = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6,
            margin_top=12, margin_bottom=8, margin_start=12, margin_end=12,
        )
        title = Gtk.Label(label="Knowledge Base", xalign=0.0)
        title.add_css_class("title-4")
        header.append(title)

        subtitle = Gtk.Label(
            label="Files indexed for this chat. The assistant will retrieve "
                  "relevant snippets when answering.",
            xalign=0.0, wrap=True,
        )
        subtitle.add_css_class("caption")
        subtitle.add_css_class("dim-label")
        header.append(subtitle)

        # Per-chat RAG override dropdown.
        rag_row = Adw.PreferencesGroup()
        self._override_row = Adw.ComboRow(title="RAG for this chat")
        sm = Gtk.StringList()
        for lbl in _OPTION_LABELS:
            sm.append(lbl)
        self._override_row.set_model(sm)
        self._override_row.connect("notify::selected", self._on_override_changed)
        rag_row.add(self._override_row)
        header.append(rag_row)
        self.append(header)

        # ── Attached notebooks section ────────────────────────────────────
        nb_header_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
            margin_start=12, margin_end=12, margin_top=4, margin_bottom=4,
        )
        nb_title = Gtk.Label(label="Attached notebooks", xalign=0.0, hexpand=True)
        nb_title.add_css_class("heading")
        nb_header_row.append(nb_title)
        self._attach_btn = Gtk.MenuButton(
            icon_name="list-add-symbolic",
            tooltip_text="Attach a notebook to this chat",
        )
        self._attach_btn.add_css_class("flat")
        self._attach_menu = Gio.Menu()
        self._attach_btn.set_menu_model(self._attach_menu)
        nb_header_row.append(self._attach_btn)
        self.append(nb_header_row)

        self._attached_list = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self._attached_list.add_css_class("boxed-list")
        self._attached_list.set_margin_start(12)
        self._attached_list.set_margin_end(12)
        self._attached_list.set_margin_bottom(8)
        self._attached_empty = Gtk.Label(
            label="No notebooks attached. Click + to share a notebook's "
                  "sources with this chat.",
            xalign=0.0, wrap=True,
            margin_start=12, margin_end=12, margin_bottom=8,
        )
        self._attached_empty.add_css_class("caption")
        self._attached_empty.add_css_class("dim-label")
        self.append(self._attached_list)
        self.append(self._attached_empty)

        # Section header for the per-chat sources list below.
        priv_title = Gtk.Label(label="Private to this chat", xalign=0.0)
        priv_title.add_css_class("heading")
        priv_title.set_margin_start(12)
        priv_title.set_margin_end(12)
        priv_title.set_margin_top(4)
        priv_title.set_margin_bottom(4)
        self.append(priv_title)

        # ── Sources list ──────────────────────────────────────────────────
        self._list_box = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("boxed-list")
        self._list_box.set_margin_start(12)
        self._list_box.set_margin_end(12)
        self._list_box.set_margin_top(4)
        self._list_box.set_margin_bottom(4)

        self._empty = Adw.StatusPage(
            icon_name="folder-documents-symbolic",
            title="No sources yet",
            description="Drop a file here or click \"Add files…\" below.",
            vexpand=True,
        )
        self._empty.add_css_class("compact")

        self._stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE)
        sources_scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
        )
        sources_scroller.set_child(self._list_box)
        self._stack.add_named(sources_scroller, "list")
        self._stack.add_named(self._empty, "empty")
        self._stack.set_vexpand(True)
        self.append(self._stack)

        # ── Add files button ──────────────────────────────────────────────
        footer = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
            margin_top=8, margin_bottom=12, margin_start=12, margin_end=12,
        )
        add_btn = Gtk.Button(label="Add files…")
        add_btn.add_css_class("suggested-action")
        add_btn.set_hexpand(True)
        add_btn.connect("clicked", self._on_add_clicked)
        footer.append(add_btn)
        self.append(footer)

        # ── Drag-drop target on the whole panel ──────────────────────────
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

    # ── Public API ───────────────────────────────────────────────────────
    def set_conversation(
        self,
        conv_id: int | None,
        rag_override: int | None,
        is_global_rag_on: bool,
    ) -> None:
        self._current_conv_id = conv_id
        # Sync dropdown without re-emitting the change signal.
        self._suppress_override_signal = True
        self._override_row.set_selected(_OVERRIDE_TO_OPTION.get(rag_override, 0))
        self._suppress_override_signal = False
        effective = "on" if (rag_override == 1 or (rag_override is None and is_global_rag_on)) else "off"
        self._override_row.set_subtitle(f"Currently {effective}")
        self.refresh_attached_notebooks()
        self.refresh_sources()

    def refresh_attached_notebooks(self) -> None:
        # Drain attached list.
        child = self._attached_list.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._attached_list.remove(child)
            child = nxt
        # Rebuild attach-menu (all - already-attached).
        self._attach_menu.remove_all()

        if self._current_conv_id is None or self._list_attached_notebooks is None:
            self._attached_list.set_visible(False)
            self._attached_empty.set_visible(True)
            self._attach_btn.set_sensitive(False)
            return

        attached = self._list_attached_notebooks(self._current_conv_id)
        if attached:
            for nb in attached:
                self._attached_list.append(self._build_notebook_row(nb))
            self._attached_list.set_visible(True)
            self._attached_empty.set_visible(False)
        else:
            self._attached_list.set_visible(False)
            self._attached_empty.set_visible(True)

        # Attach-menu shows notebooks that aren't already attached.
        self._attach_btn.set_sensitive(True)
        if self._list_all_notebooks is None:
            return
        attached_ids = {nb.id for nb in attached}
        candidates = [nb for nb in self._list_all_notebooks() if nb.id not in attached_ids]
        if not candidates:
            item = Gio.MenuItem.new("(No other notebooks)", None)
            self._attach_menu.append_item(item)
            return
        for nb in candidates:
            item = Gio.MenuItem.new(nb.name, None)
            item.set_action_and_target_value(
                "win.attach-notebook", GLib.Variant("i", nb.id),
            )
            self._attach_menu.append_item(item)

    def _build_notebook_row(self, nb) -> Gtk.Widget:
        row = Adw.ActionRow(title=GLib.markup_escape_text(nb.name))
        icon = Gtk.Image.new_from_icon_name("accessories-dictionary-symbolic")
        row.add_prefix(icon)
        rm = Gtk.Button(
            icon_name="window-close-symbolic",
            tooltip_text="Detach this notebook from the chat",
            valign=Gtk.Align.CENTER,
        )
        rm.add_css_class("flat")
        rm.connect("clicked", lambda *_: self._handle_detach(nb.id))
        row.add_suffix(rm)
        return row

    def _handle_detach(self, nb_id: int) -> None:
        if self._on_detach_notebook is not None:
            self._on_detach_notebook(nb_id)

    def refresh_sources(self) -> None:
        # Drain list.
        child = self._list_box.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._list_box.remove(child)
            child = nxt

        if self._current_conv_id is None:
            self._stack.set_visible_child_name("empty")
            return

        sources = self._list_sources_for(self._current_conv_id)
        if not sources:
            self._stack.set_visible_child_name("empty")
            return

        for path, label, count in sources:
            self._list_box.append(self._build_row(path, label, count))
        self._stack.set_visible_child_name("list")

    # ── Internals ────────────────────────────────────────────────────────
    def _build_row(self, path: str | None, label: str | None, count: int) -> Gtk.Widget:
        row = Adw.ActionRow(
            title=GLib.markup_escape_text(label or "(unnamed)"),
            subtitle=f"{count} chunk{'s' if count != 1 else ''}"
                     + (f"  ·  {GLib.markup_escape_text(_short_path(path))}" if path else ""),
        )
        icon = Gtk.Image.new_from_icon_name(_icon_for(path))
        row.add_prefix(icon)

        rm = Gtk.Button(
            icon_name="user-trash-symbolic",
            tooltip_text="Remove from knowledge base",
            valign=Gtk.Align.CENTER,
        )
        rm.add_css_class("flat")
        rm.connect("clicked", lambda *_: self._handle_remove(path))
        row.add_suffix(rm)
        return row

    def _handle_remove(self, source_path: str | None) -> None:
        if source_path is None:
            return
        self._on_remove_source(source_path)

    def _on_override_changed(self, *_args) -> None:
        if self._suppress_override_signal:
            return
        idx = self._override_row.get_selected()
        self._on_rag_override_changed(_OPTION_TO_OVERRIDE.get(idx))

    def _on_add_clicked(self, *_args) -> None:
        # Multi-select file dialog.
        dlg = Gtk.FileDialog(title="Add files to knowledge base")
        f_all = Gtk.FileFilter(name="Text, code, PDF, images")
        for ext in (".txt", ".md", ".pdf", ".py", ".js", ".ts", ".json",
                    ".csv", ".xml", ".yaml", ".yml", ".toml", ".html",
                    ".css", ".sh", ".c", ".cpp", ".h", ".rs", ".go",
                    ".java", ".kt", ".swift",
                    # Image RAG (Phase 3b): captioned via the active LLM.
                    ".jpg", ".jpeg", ".png", ".webp", ".gif"):
            f_all.add_pattern(f"*{ext}")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f_all)
        dlg.set_filters(filters)
        win = self.get_root()
        dlg.open_multiple(win if isinstance(win, Gtk.Window) else None, None,
                          self._on_files_picked)

    def _on_files_picked(self, dialog, result) -> None:
        try:
            files = dialog.open_multiple_finish(result)
        except Exception:
            return
        if files is None:
            return
        paths: list[str] = []
        n = files.get_n_items()
        for i in range(n):
            f = files.get_item(i)
            p = f.get_path() if f else None
            if p:
                paths.append(p)
        if paths:
            self._on_add_files(paths)

    def _on_drop(self, _target, value, _x, _y) -> bool:
        if value is None:
            return False
        paths: list[str] = []
        for f in value.get_files():
            p = f.get_path()
            if p:
                paths.append(p)
        if not paths:
            return False
        self._on_add_files(paths)
        return True


def _short_path(path: str) -> str:
    p = Path(path)
    home = Path.home()
    try:
        return "~/" + str(p.relative_to(home))
    except ValueError:
        return str(p)


def _icon_for(path: str | None) -> str:
    if path is None:
        return "text-x-generic-symbolic"
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "application-pdf-symbolic"
    if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        return "image-x-generic-symbolic"
    if ext in (".py", ".js", ".ts", ".c", ".cpp", ".h", ".rs", ".go",
               ".java", ".kt", ".swift", ".sh"):
        return "text-x-script-symbolic"
    return "text-x-generic-symbolic"
