"""System tray via StatusNotifierItem (+ com.canonical.dbusmenu).

Ported from Frequency's ``_Tray`` (/opt/frequency/gui.py) — pure ``Gio.DBus``
because GTK4 has no tray API and the appindicator libraries would drag GTK3
into the process. Close-to-tray parks the app in the top-bar tray and locks
it (closing the loop with App Lock).

If no StatusNotifierWatcher is running (tray extension off), ``ok`` stays
False and the window's close handler just quits normally instead.
"""
from __future__ import annotations

import logging
import os

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib  # noqa: E402

from . import APP_NAME

log = logging.getLogger(__name__)

_ICONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "icons")


class Tray:
    """Minimal StatusNotifierItem so closing the window parks Box in the tray."""

    _SNI_XML = """<node><interface name="org.kde.StatusNotifierItem">
      <property name="Category" type="s" access="read"/>
      <property name="Id" type="s" access="read"/>
      <property name="Title" type="s" access="read"/>
      <property name="Status" type="s" access="read"/>
      <property name="IconName" type="s" access="read"/>
      <property name="IconThemePath" type="s" access="read"/>
      <property name="Menu" type="o" access="read"/>
      <property name="ItemIsMenu" type="b" access="read"/>
      <method name="Activate"><arg type="i"/><arg type="i"/></method>
      <method name="SecondaryActivate"><arg type="i"/><arg type="i"/></method>
      <method name="ContextMenu"><arg type="i"/><arg type="i"/></method>
      <method name="Scroll"><arg type="i"/><arg type="s"/></method>
    </interface></node>"""

    _MENU_XML = """<node><interface name="com.canonical.dbusmenu">
      <property name="Version" type="u" access="read"/>
      <property name="Status" type="s" access="read"/>
      <method name="GetLayout">
        <arg type="i" direction="in"/><arg type="i" direction="in"/>
        <arg type="as" direction="in"/>
        <arg type="u" direction="out"/><arg type="(ia{sv}av)" direction="out"/>
      </method>
      <method name="GetGroupProperties">
        <arg type="ai" direction="in"/><arg type="as" direction="in"/>
        <arg type="a(ia{sv})" direction="out"/>
      </method>
      <method name="Event">
        <arg type="i" direction="in"/><arg type="s" direction="in"/>
        <arg type="v" direction="in"/><arg type="u" direction="in"/>
      </method>
      <method name="AboutToShow"><arg type="i" direction="in"/>
        <arg type="b" direction="out"/></method>
    </interface></node>"""

    def __init__(self, app) -> None:
        self.app = app
        self.ok = False
        self._rev = 1
        self._name = f"org.kde.StatusNotifierItem-{os.getpid()}-1"
        try:
            self._bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            sni = Gio.DBusNodeInfo.new_for_xml(self._SNI_XML).interfaces[0]
            menu = Gio.DBusNodeInfo.new_for_xml(self._MENU_XML).interfaces[0]
            self._bus.register_object("/StatusNotifierItem", sni,
                                      self._sni_call, self._sni_get, None)
            self._bus.register_object("/MenuBar", menu,
                                      self._menu_call, self._menu_get, None)
            Gio.bus_own_name_on_connection(self._bus, self._name,
                                           Gio.BusNameOwnerFlags.NONE,
                                           self._on_name, None)
        except Exception:  # noqa: BLE001
            self.ok = False

    def _on_name(self, _bus, _name) -> None:
        try:
            self._bus.call_sync(
                "org.kde.StatusNotifierWatcher", "/StatusNotifierWatcher",
                "org.kde.StatusNotifierWatcher", "RegisterStatusNotifierItem",
                GLib.Variant("(s)", (self._name,)), None,
                Gio.DBusCallFlags.NONE, 2000, None)
            self.ok = True
        except Exception:  # noqa: BLE001 — no watcher → close quits normally
            self.ok = False

    def _sni_get(self, _c, _s, _p, _i, prop):
        vals = {
            "Category": ("s", "ApplicationStatus"),
            "Id": ("s", "box"),
            "Title": ("s", APP_NAME),
            "Status": ("s", "Active"),
            "IconName": ("s", "box-tray-symbolic"),
            "IconThemePath": ("s", os.path.abspath(_ICONS_DIR)),
            "Menu": ("o", "/MenuBar"),
            "ItemIsMenu": ("b", False),
        }
        t, v = vals.get(prop, ("s", ""))
        return GLib.Variant(t, v)

    def _sni_call(self, _c, _s, _p, _i, method, _params, inv) -> None:
        if method in ("Activate", "SecondaryActivate"):
            self.app.tray_toggle()
        inv.return_value(None)

    def _items(self):
        # id 1 = show/hide, 5 = Box Code, 4 = lock, 2 = separator, 3 = quit
        return [(1, "Show / hide window", True), (5, "Box Code", True),
                (4, "Lock now", True), (2, None, False),
                (3, "Quit Box", True)]

    def _item_props(self, iid):
        for i, label, enabled in self._items():
            if i == iid:
                if label is None:
                    return {"type": GLib.Variant("s", "separator")}
                return {"label": GLib.Variant("s", label),
                        "enabled": GLib.Variant("b", enabled)}
        return {}

    def _menu_call(self, _c, _s, _p, _i, method, params, inv) -> None:
        if method == "GetLayout":
            children = [GLib.Variant("(ia{sv}av)", (i, self._item_props(i), []))
                        for i, _l, _e in self._items()]
            root = (0, {"children-display": GLib.Variant("s", "submenu")}, children)
            inv.return_value(GLib.Variant("(u(ia{sv}av))", (self._rev, root)))
        elif method == "GetGroupProperties":
            ids = params.unpack()[0]
            out = [(i, self._item_props(i)) for i in ids]
            inv.return_value(GLib.Variant("(a(ia{sv}))", (out,)))
        elif method == "Event":
            iid, event, _d, _t = params.unpack()
            if event == "clicked":
                if iid == 1:
                    self.app.tray_toggle()
                elif iid == 5:
                    self.app.tray_code()
                elif iid == 4:
                    self.app.tray_lock()
                elif iid == 3:
                    self.app.tray_quit()
            inv.return_value(None)
        elif method == "AboutToShow":
            inv.return_value(GLib.Variant("(b)", (False,)))
        else:
            inv.return_value(None)

    def _menu_get(self, _c, _s, _p, _i, prop):
        if prop == "Version":
            return GLib.Variant("u", 3)
        return GLib.Variant("s", "normal")
