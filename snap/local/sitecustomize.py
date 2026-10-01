"""Snap shim, loaded automatically via PYTHONPATH.

ctypes.util.find_library on Linux only consults ldconfig's cache and binutils'
``ld``; neither sees libraries staged inside $SNAP, so sounddevice can't find
libportaudio and voice mode fails. Fall back to scanning LD_LIBRARY_PATH.
"""

import ctypes.util
import glob
import os

_find_library = ctypes.util.find_library


def find_library(name):
    found = _find_library(name)
    if found:
        return found
    for d in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if d:
            hits = sorted(glob.glob(os.path.join(d, f"lib{name}.so.*")))
            if hits:
                return hits[0]
    return None


ctypes.util.find_library = find_library
