"""Windows webcam backend — placeholder.

A real impl should use Media Foundation or DirectShow, or GStreamer's
``mfvideosrc``/``dshowvideosrc`` when shipping a bundled GStreamer.
Until that lands, the webcam feature is unavailable on Windows.
"""
from __future__ import annotations

from typing import Any

from . import CameraDevice

AVAILABLE: bool = False
INSTALL_HINT: str = "Webcam support is not yet implemented on Windows."
unavailable_reason: str = INSTALL_HINT


class CameraSession:
    def __init__(self, *_a: Any, **_kw: Any) -> None:
        raise NotImplementedError(unavailable_reason)

    def __enter__(self):
        raise NotImplementedError(unavailable_reason)

    def __exit__(self, *_a):
        return False


def probe(force: bool = False) -> tuple[bool, str]:  # noqa: ARG001
    return False, unavailable_reason


def list_devices() -> list[CameraDevice]:
    return []


def open_session(device_id: str | None = None) -> CameraSession:  # noqa: ARG001
    raise NotImplementedError(unavailable_reason)
