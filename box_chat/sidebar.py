"""Sidebar — conversation list with search, rename, delete, new-chat."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, GObject, Gio, Gtk  # noqa: E402

from .database import Conversation, Database


class _ConvRow(GObject.GObject):
    """Boxed row object so Gtk.ListView can hold it."""
    __gtype_name__ = "BoxChatConvRow"

    conv_id = GObject.Property(type=int, default=0)
    title = GObject.Property(type=str, default="")
    subtitle = GObject.Property(type=str, default="")

    def __init__(self, conv: Conversation):
        super().__init__()
        self.conv_id = conv.id
        self.title = conv.title
        ts = datetime.fromtimestamp(conv.updated_at)
        self.subtitle = ts.strftime("%b %d, %H:%M")


class Sidebar(Gtk.Box):
    """Adw.NavigationPage-friendly sidebar.

    Emits a Python-level callback when the user selects, creates, renames, or
    deletes a conversation. The window owns the chat-view state machine.
    """

    def __init__(
        self,
        db: Database,
        on_select: Callable[[int], None],
        on_create: Callable[[], None],
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        self._db = db
        self._on_select = on_select
        self._on_create = on_create

        # ── New-chat button (sticky at top) ────────────────────────────────
        toolbar = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_top=8, margin_bottom=8, margin_start=8, margin_end=8,
        )
        new_btn = Gtk.Button(
            label="New Chat",
            halign=Gtk.Align.FILL,
            hexpand=True,
        )
        new_btn.set_child(_label_with_icon("list-add-symbolic", "New Chat"))
        new_btn.add_css_class("header-font")
        new_btn.add_css_class("suggested-action")
        new_btn.connect("clicked", lambda *_: self._on_create())
        toolbar.append(new_btn)
        self.append(toolbar)

        # ── Search entry ───────────────────────────────────────────────────
        self._search = Gtk.SearchEntry(
            placeholder_text="Search chats…",
            margin_start=8, margin_end=8, margin_bottom=6,
        )
        self._search.connect("search-changed", lambda *_: self.refresh())
        self.append(self._search)

        # ── ListView of conversations ──────────────────────────────────────
        self._store = Gio.ListStore(item_type=_ConvRow)
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

    # ── Public API ─────────────────────────────────────────────────────────
    def refresh(self, select_id: int | None = None) -> None:
        query = self._search.get_text().strip()
        convs = self._db.list_conversations(query=query)
        self._store.remove_all()
        for c in convs:
            self._store.append(_ConvRow(c))
        # Restore selection if requested.
        if select_id is not None:
            for i, c in enumerate(convs):
                if c.id == select_id:
                    self._selection.set_selected(i)
                    break

    def selected_id(self) -> int | None:
        idx = self._selection.get_selected()
        if idx == Gtk.INVALID_LIST_POSITION:
            return None
        row = self._store.get_item(idx)
        return row.conv_id if row else None

    # ── Internal helpers ───────────────────────────────────────────────────
    def _row_setup(self, _factory, list_item: Gtk.ListItem) -> None:
        box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_top=4, margin_bottom=4, margin_start=8, margin_end=4,
        )
        text_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, hexpand=True)
        title = Gtk.Label(xalign=0, ellipsize=3)  # ELLIPSIZE_END
        title.add_css_class("title")
        subtitle = Gtk.Label(xalign=0)
        subtitle.add_css_class("dim-label")
        subtitle.add_css_class("caption")
        text_box.append(title)
        text_box.append(subtitle)
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
        text_box: Gtk.Box = box.get_first_child()
        title: Gtk.Label = text_box.get_first_child()
        subtitle: Gtk.Label = text_box.get_last_child()
        menu_btn: Gtk.MenuButton = box.get_last_child()

        row: _ConvRow = list_item.get_item()
        title.set_text(row.title or "Untitled")
        subtitle.set_text(row.subtitle)

        # Per-row menu (rename / delete). Build fresh each bind because the
        # conv_id changes when the row is recycled.
        menu = Gio.Menu()
        menu.append("Rename…", f"win.rename-conv({row.conv_id})")
        menu.append("Delete", f"win.delete-conv({row.conv_id})")
        menu_btn.set_menu_model(menu)

    def _on_row_activated(self, _listview, position: int) -> None:
        row = self._store.get_item(position)
        if row is not None:
            self._selection.set_selected(position)
            self._on_select(row.conv_id)


def _label_with_icon(icon_name: str, label: str) -> Gtk.Widget:
    box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6, halign=Gtk.Align.CENTER)
    box.append(Gtk.Image.new_from_icon_name(icon_name))
    box.append(Gtk.Label(label=label))
    return box
