"""OS-abstracted webcam capture.

Public surface:

- :data:`AVAILABLE` — best-effort flag set at import time (cheap check
  only; the real verification is :func:`probe`).
- :data:`INSTALL_HINT` — actionable string for the user if AVAILABLE is
  False on their platform.
- :func:`probe` — run a real pipeline open/close cycle and update
  AVAILABLE / unavailable_reason. Cached after first call; pass
  ``force=True`` to retry (e.g. user just installed a missing plugin).
- :func:`list_devices` — enumerate connected cameras.
- :func:`open_session(device_id=None)` — context-manager that yields a
  :class:`CameraSession` (paintable + capture_jpeg).

This package is the only place in box_chat that knows about
GStreamer / PipeWire / V4L2. The UI layer talks to it through plain
Python — keeps the engine / RAG / notebook core portable, in line with
the "core stays portable" project rule.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CameraDevice:
    """A camera we can open."""
    id: str           # OS-specific handle (V4L2 path / PipeWire node-id)
    label: str        # Human-readable name for the picker


# The backend module is selected at import time so the rest of the app
# can import these symbols statically. Each backend exposes the same
# public surface; only the Linux one actually works today.
if sys.platform.startswith("linux"):
    from . import _linux_gst as _impl  # noqa: F401
elif sys.platform == "darwin":
    from . import _macos as _impl  # noqa: F401
elif sys.platform == "win32":
    from . import _windows as _impl  # noqa: F401
else:
    from . import _unsupported as _impl  # noqa: F401


# Re-export the backend's public surface. Each value is read live from
# the impl module so calls like ``probe()`` can mutate them.
def _get(name: str, default: Any = None) -> Any:
    return getattr(_impl, name, default)


def __getattr__(name: str) -> Any:
    """Lazy attribute lookup so AVAILABLE / unavailable_reason / INSTALL_HINT
    always reflect the impl module's *current* value, not a snapshot."""
    if name in (
        "AVAILABLE",
        "INSTALL_HINT",
        "unavailable_reason",
    ):
        return getattr(_impl, name)
    raise AttributeError(name)


def probe(force: bool = False) -> tuple[bool, str]:
    """Verify the backend can actually open + close a pipeline.

    Returns ``(ok, reason)``. ``reason`` is an empty string when ok.
    """
    return _impl.probe(force=force)


def list_devices() -> list[CameraDevice]:
    """Enumerate connected cameras. Empty list if the backend can't
    enumerate (try opening the system-default device anyway)."""
    return _impl.list_devices()


def open_session(device_id: str | None = None) -> "CameraSession":
    """Open a camera session. Use as a context manager:

    .. code-block:: python

        with webcam.open_session() as cam:
            cam.attach_picture(my_gtk_picture)
            jpeg = cam.capture_jpeg(quality=80)
    """
    return _impl.open_session(device_id)


# Re-export the class so type hints in callers work. The backend module
# defines the concrete class; the abstract surface lives below as a
# protocol-style stub for documentation.
CameraSession = _impl.CameraSession  # type: ignore[misc]
