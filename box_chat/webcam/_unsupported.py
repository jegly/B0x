"""Fallback backend for platforms with no webcam impl yet (BSD, etc.)."""
from __future__ import annotations

import sys
from typing import Any

from . import CameraDevice

AVAILABLE: bool = False
INSTALL_HINT: str = ""
unavailable_reason: str = f"No webcam backend for platform: {sys.platform}"


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
