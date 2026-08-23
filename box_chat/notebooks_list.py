"""Notebooks sidebar pane — listed alongside the Chats pane via a tab switcher.

Mirrors the Sidebar widget's shape (new button at top + Gtk.ListView body)
so the two panes feel like siblings inside the Adw.ViewSwitcher.
"""
from __future__ import annotations

from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import GLib, GObject, Gio, Gtk  # noqa: E402


class _NotebookRow(GObject.GObject):
    __gtype_name__ = "BoxNotebookRow"

    nb_id = GObject.Property(type=int, default=0)
    name = GObject.Property(type=str, default="")
    subtitle = GObject.Property(type=str, default="")
    auto_attach = GObject.Property(type=bool, default=False)

    def __init__(self, nb_id: int, name: str, file_count: int, auto_attach: bool = False):
        super().__init__()
        self.nb_id = nb_id
        self.name = name
        # "📌 " prefix makes auto-attach notebooks instantly recognisable
        # in the sidebar list without adding a column.
        prefix = "📌  " if auto_attach else ""
        self.subtitle = prefix + f"{file_count} file{'s' if file_count != 1 else ''}"
        self.auto_attach = auto_attach


class NotebooksList(Gtk.Box):
    """List of notebooks with new/rename/delete and an on-select callback.

    Callbacks:
      on_select(nb_id)  — user clicked a notebook row
      on_create()       — user clicked "+ New Notebook"
    """

    def __init__(
        self,
        list_notebooks: Callable[[], list],
        notebook_file_count: Callable[[int], int],
        on_select: Callable[[int], None],
        on_create: Callable[[], None],
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self._list_notebooks = list_notebooks
        self._file_count = notebook_file_count
        self._on_select = on_select
        self._on_create = on_create

        # New-notebook button at top.
        toolbar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
            margin_top=8, margin_bottom=8, margin_start=8, margin_end=8,
        )
        new_btn = Gtk.Button(halign=Gtk.Align.FILL, hexpand=True)
        new_btn.set_child(_label_with_icon("list-add-symbolic", "New Notebook"))
        new_btn.add_css_class("header-font")
        new_btn.add_css_class("suggested-action")
        new_btn.connect("clicked", lambda *_: self._on_create())
        toolbar.append(new_btn)
        self.append(toolbar)

        # ListView of notebooks.
        self._store = Gio.ListStore(item_type=_NotebookRow)
        self._selection = Gtk.SingleSelection(model=self._store, autoselect=False)

        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._row_setup)
        factory.connect("bind", self._row_bind)

        listview = Gtk.ListView(
            model=self._selection,
            factory=factory,
            single_click_activate=True,
            hexpand=True, vexpand=True,
        )
        listview.add_css_class("navigation-sidebar")
        listview.connect("activate", self._on_row_activated)

        scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            child=listview,
            vexpand=True,
        )
        self.append(scroller)

        self.refresh()

    # ── Public API ────────────────────────────────────────────────────────
    def refresh(self, select_id: int | None = None) -> None:
        nbs = self._list_notebooks()
        self._store.remove_all()
        for nb in nbs:
            self._store.append(_NotebookRow(
                nb.id, nb.name, self._file_count(nb.id),
                auto_attach=bool(getattr(nb, "auto_attach", 0)),
            ))
        if select_id is not None:
            for i, nb in enumerate(nbs):
                if nb.id == select_id:
                    self._selection.set_selected(i)
                    break

    # ── Internals ─────────────────────────────────────────────────────────
    def _row_setup(self, _factory, list_item: Gtk.ListItem) -> None:
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
            margin_top=4, margin_bottom=4, margin_start=8, margin_end=4,
        )
        icon = Gtk.Image.new_from_icon_name("accessories-dictionary-symbolic")
        box.append(icon)
        text_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        title = Gtk.Label(xalign=0, ellipsize=3)
        title.add_css_class("title")
        sub = Gtk.Label(xalign=0)
        sub.add_css_class("dim-label")
        sub.add_css_class("caption")
        text_box.append(title)
        text_box.append(sub)
        box.append(text_box)

        menu_btn = Gtk.MenuButton(
            icon_name="view-more-symbolic",
            has_frame=False,
            valign=Gtk.Align.CENTER,
        )
        menu_btn.add_css_class("flat")
        box.append(menu_btn)
        list_item.set_child(box)

    def _row_bind(self, _factory, list_item: Gtk.ListItem) -> None:
        box: Gtk.Box = list_item.get_child()
        # children: [icon, text_box, menu_btn]
        icon = box.get_first_child()
        text_box = icon.get_next_sibling()
        title = text_box.get_first_child()
        sub = text_box.get_last_child()
        menu_btn = box.get_last_child()

        row: _NotebookRow = list_item.get_item()
        title.set_text(row.name or "(Untitled)")
        sub.set_text(row.subtitle)

        menu = Gio.Menu()
        menu.append("Rename…", f"win.rename-notebook({row.nb_id})")
        menu.append("Delete",  f"win.delete-notebook({row.nb_id})")
        menu_btn.set_menu_model(menu)

    def _on_row_activated(self, _listview, position: int) -> None:
        row = self._store.get_item(position)
        if row is not None:
            self._selection.set_selected(position)
            self._on_select(row.nb_id)


def _label_with_icon(icon_name: str, label: str) -> Gtk.Widget:
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6, halign=Gtk.Align.CENTER)
    box.append(Gtk.Image.new_from_icon_name(icon_name))
    box.append(Gtk.Label(label=label))
    return box
