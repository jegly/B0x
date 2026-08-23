"""Bundled font registration via fontconfig (2026-07-14).

Box ships a set of .ttf families (``data/fonts/``) and registers them with
the running process's fontconfig config through ``FcConfigAppFontAddFile``
(ctypes) — no PyGObject binding exists for this, and no system install is
needed. Once registered, the families resolve through Pango's normal
font-family lookup, so the chat/UI font pickers can offer them.

Pure ctypes + stdlib (the ``gi`` UI layer calls ``register_bundled_fonts``
once at startup). Failure is non-fatal — the app just falls back to system
fonts.
"""
from __future__ import annotations

import ctypes
import logging
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["register_bundled_fonts", "fonts_dir", "BUNDLED_FONT_FAMILIES"]

# Human-facing family names of the bundled fonts, for a curated picker. The
# actual resolution is by whatever family name each .ttf declares — this list
# is the display/curation layer.
BUNDLED_FONT_FAMILIES: list[str] = [
    "Nunito", "IBM Plex Serif", "Playfair Display", "Cormorant Garamond",
    "Source Code Pro", "JetBrains Mono", "Inter", "Lora", "Merriweather",
    "Roboto Slab", "Fira Code", "Space Mono", "EB Garamond",
]


def fonts_dir() -> Path:
    for d in (
        Path("/usr/share/box/fonts"),
        Path(__file__).resolve().parent.parent / "data" / "fonts",
    ):
        if d.is_dir():
            return d
    return Path(__file__).resolve().parent.parent / "data" / "fonts"


def register_bundled_fonts() -> int:
    """Register every bundled .ttf with the process fontconfig. Returns the
    count successfully added. Non-fatal on any error."""
    d = fonts_dir()
    if not d.is_dir():
        return 0
    try:
        fc = ctypes.CDLL("libfontconfig.so.1")
    except OSError:
        try:
            fc = ctypes.CDLL("libfontconfig.so")
        except OSError:
            log.info("fontconfig not loadable; skipping bundled font registration")
            return 0

    fc.FcConfigGetCurrent.restype = ctypes.c_void_p
    fc.FcConfigAppFontAddFile.restype = ctypes.c_int
    fc.FcConfigAppFontAddFile.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

    config = fc.FcConfigGetCurrent()
    count = 0
    for ttf in sorted(d.glob("*.ttf")):
        if fc.FcConfigAppFontAddFile(config, str(ttf).encode("utf-8")):
            count += 1
    log.info("registered %d bundled fonts from %s", count, d)
    return count
