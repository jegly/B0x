"""Snap entry point: run Box as a non-unique GApplication.

A strict snap may only own a D-Bus name (com.jegly.box) through a dbus slot,
and the store holds those for manual review. Without the name, Gtk.Application
registration fails and Box quits — so skip uniqueness instead. A second launch
opens a second window rather than focusing the first.
"""

import sys

from gi.repository import Gio

import box_chat.app
from box_chat.__main__ import main

_init = box_chat.app.BoxApp.__init__


def _init_non_unique(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.set_flags(self.get_flags() | Gio.ApplicationFlags.NON_UNIQUE)


box_chat.app.BoxApp.__init__ = _init_non_unique

sys.argv[0] = "box"
sys.exit(main())
