"""Notebook detail page — full-pane view for managing a notebook's sources.

Shown in the main content stack when the user opens a notebook from the
sidebar. Mirrors the file-list shape used in `KbPanel` but with more space
and notebook-level actions (rename / delete).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gdk, Gio, GLib, Gtk  # noqa: E402

# Re-use the icon picker / short-path helpers from kb_panel.
from .kb_panel import _icon_for, _short_path


class NotebookView(Gtk.Box):
    """Body of the notebook detail page.

    Public API:
      load(nb_id, name, sources)  — switch to a notebook and render its sources
      refresh()                   — re-fetch sources for the current notebook

    Callbacks (constructor):
      list_sources(nb_id)         — () → list[(path, label, count)]
      on_add_files(nb_id, paths)  — user picked / dropped files
      on_remove_source(nb_id, path)
      on_rename(nb_id)            — host opens a rename dialog
      on_delete(nb_id)            — host opens a confirm-delete dialog
    """

    def __init__(
        self,
        list_sources: Callable[[int], list],
        on_add_files: Callable[[int, list[str]], None],
        on_remove_source: Callable[[int, str], None],
        on_rename: Callable[[int], None],
        on_delete: Callable[[int], None],
        on_auto_attach_changed: Callable[[int, bool], None] | None = None,
        on_back: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add_css_class("notebook-view")
        self._list_sources = list_sources
        self._on_add_files = on_add_files
        self._on_remove_source = on_remove_source
        self._on_rename = on_rename
        self._on_delete = on_delete
        self._on_auto_attach_changed = on_auto_attach_changed
        self._on_back = on_back
        self._nb_id: int | None = None
        self._suppress_auto_attach = False

        # ── Header: back button + title + rename / delete buttons ────────
        header = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            margin_top=16, margin_bottom=8, margin_start=20, margin_end=20,
        )

        if on_back is not None:
            back_btn = Gtk.Button(
                icon_name="go-previous-symbolic",
                tooltip_text="Back to chat",
                valign=Gtk.Align.START,
            )
            back_btn.add_css_class("flat")
            back_btn.connect("clicked", lambda *_: on_back())
            header.append(back_btn)

        title_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        self._title_label = Gtk.Label(label="Notebook", xalign=0.0)
        self._title_label.add_css_class("title-1")
        sub = Gtk.Label(
            label="Sources here are searchable from any chat that attaches "
                  "this notebook.",
            xalign=0.0, wrap=True,
        )
        sub.add_css_class("caption")
        sub.add_css_class("dim-label")
        title_box.append(self._title_label)
        title_box.append(sub)
        header.append(title_box)

        rename_btn = Gtk.Button(icon_name="document-edit-symbolic",
                                tooltip_text="Rename notebook",
                                valign=Gtk.Align.START)
        rename_btn.add_css_class("flat")
        rename_btn.connect("clicked", self._on_rename_clicked)
        header.append(rename_btn)

        del_btn = Gtk.Button(icon_name="user-trash-symbolic",
                             tooltip_text="Delete notebook",
                             valign=Gtk.Align.START)
        del_btn.add_css_class("flat")
        del_btn.add_css_class("destructive-action")
        del_btn.connect("clicked", self._on_delete_clicked)
        header.append(del_btn)
        self.append(header)

        # ── Auto-attach toggle ────────────────────────────────────────────
        aa_group = Adw.PreferencesGroup(
            margin_start=20, margin_end=20, margin_bottom=4,
        )
        self._auto_attach_row = Adw.SwitchRow(
            title="Attach to new chats by default",
            subtitle="Every new chat you create will start with this notebook attached. "
                     "Existing chats are not affected.",
        )
        self._auto_attach_row.connect("notify::active", self._on_auto_attach_toggled)
        aa_group.add(self._auto_attach_row)
        self.append(aa_group)

        # ── Sources list ──────────────────────────────────────────────────
        self._list_box = Gtk.ListBox(selection_mode=Gtk.SelectionMode.NONE)
        self._list_box.add_css_class("boxed-list")
        self._list_box.set_margin_start(20)
        self._list_box.set_margin_end(20)
        self._list_box.set_margin_top(8)
        self._list_box.set_margin_bottom(8)

        self._empty = Adw.StatusPage(
            icon_name="folder-documents-symbolic",
            title="No sources in this notebook",
            description="Drop files here or click \"Add files…\" below.",
            vexpand=True,
        )
        self._empty.add_css_class("compact")

        self._stack = Gtk.Stack(transition_type=Gtk.StackTransitionType.CROSSFADE,
                                vexpand=True)
        scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC, vexpand=True,
        )
        scroller.set_child(self._list_box)
        self._stack.add_named(scroller, "list")
        self._stack.add_named(self._empty, "empty")
        self.append(self._stack)

        # ── Add files footer ──────────────────────────────────────────────
        footer = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            margin_top=8, margin_bottom=16, margin_start=20, margin_end=20,
        )
        add_btn = Gtk.Button(label="Add files…")
        add_btn.add_css_class("suggested-action")
        add_btn.set_hexpand(True)
        add_btn.connect("clicked", self._on_add_clicked)
        footer.append(add_btn)
        self.append(footer)

        # Drop target on the whole view.
        drop = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        drop.connect("drop", self._on_drop)
        self.add_controller(drop)

    # ── Public API ───────────────────────────────────────────────────────
    def load(self, nb_id: int, name: str, auto_attach: bool = False) -> None:
        self._nb_id = nb_id
        self._title_label.set_text(name or "Notebook")
        self._suppress_auto_attach = True
        self._auto_attach_row.set_active(bool(auto_attach))
        self._suppress_auto_attach = False
        self.refresh()

    def _on_auto_attach_toggled(self, *_args) -> None:
        if self._suppress_auto_attach:
            return
        if self._nb_id is None or self._on_auto_attach_changed is None:
            return
        self._on_auto_attach_changed(self._nb_id, self._auto_attach_row.get_active())

    def refresh(self) -> None:
        # Drain list.
        child = self._list_box.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._list_box.remove(child)
            child = nxt

        if self._nb_id is None:
            self._stack.set_visible_child_name("empty")
            return
        sources = self._list_sources(self._nb_id)
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
        row.add_prefix(Gtk.Image.new_from_icon_name(_icon_for(path)))
        rm = Gtk.Button(
            icon_name="user-trash-symbolic",
            tooltip_text="Remove from notebook",
            valign=Gtk.Align.CENTER,
        )
        rm.add_css_class("flat")
        rm.connect("clicked", lambda *_: self._handle_remove(path))
        row.add_suffix(rm)
        return row

    def _handle_remove(self, source_path: str | None) -> None:
        if source_path is None or self._nb_id is None:
            return
        self._on_remove_source(self._nb_id, source_path)

    def _on_add_clicked(self, *_args) -> None:
        if self._nb_id is None:
            return
        dlg = Gtk.FileDialog(title="Add files to notebook")
        f_all = Gtk.FileFilter(name="Text, code, PDF, images")
        for ext in (".txt", ".md", ".pdf", ".py", ".js", ".ts", ".json",
                    ".csv", ".xml", ".yaml", ".yml", ".toml", ".html",
                    ".css", ".sh", ".c", ".cpp", ".h", ".rs", ".go",
                    ".java", ".kt", ".swift",
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
        if files is None or self._nb_id is None:
            return
        paths: list[str] = []
        for i in range(files.get_n_items()):
            f = files.get_item(i)
            p = f.get_path() if f else None
            if p:
                paths.append(p)
        if paths:
            self._on_add_files(self._nb_id, paths)

    def _on_drop(self, _target, value, _x, _y) -> bool:
        if value is None or self._nb_id is None:
            return False
        paths: list[str] = []
        for f in value.get_files():
            p = f.get_path()
            if p:
                paths.append(p)
        if not paths:
            return False
        self._on_add_files(self._nb_id, paths)
        return True

    def _on_rename_clicked(self, *_args) -> None:
        if self._nb_id is not None:
            self._on_rename(self._nb_id)

    def _on_delete_clicked(self, *_args) -> None:
        if self._nb_id is not None:
            self._on_delete(self._nb_id)
