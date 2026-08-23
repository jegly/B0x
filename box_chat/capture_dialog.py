"""Webcam capture modal (Adw.Dialog).

Opens the live preview, lets the user pick a camera if more than one is
attached, and saves the captured JPEG to ``CAPTURES_DIR/box_<ts>.jpg``.
On every close path (Capture, Cancel, OS dismiss) the GStreamer pipeline
is torn down — the camera light must never stay on after the dialog
goes away.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gtk  # noqa: E402

from . import webcam
from .config import CAPTURES_DIR, Settings

log = logging.getLogger(__name__)


class CaptureDialog(Adw.Dialog):
    """Modal camera dialog. Construct, then ``present(parent)``.

    Args:
        settings: live Settings instance — read for default device,
            capture width, JPEG quality.
        on_captured: called with the absolute JPEG path on a successful
            capture. Always invoked from the GTK main thread.
    """

    def __init__(
        self,
        settings: Settings,
        on_captured: Callable[[str], None],
    ) -> None:
        super().__init__()
        self.set_title("Camera")
        self.set_content_width(560)
        self.set_content_height(480)

        self._settings = settings
        self._on_captured = on_captured
        self._session: webcam.CameraSession | None = None
        self._devices: list[webcam.CameraDevice] = []
        self._preview_picture: Gtk.Picture | None = None

        toolbar = Adw.ToolbarView()
        toolbar.add_top_bar(Adw.HeaderBar())

        outer = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=12,
            margin_top=12, margin_bottom=12, margin_start=12, margin_end=12,
        )

        # Preview area — will be filled with either a Gtk.Picture bound
        # to the pipeline's paintable, or a status page hint when no
        # paintable-capable sink is installed.
        self._preview_slot = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            hexpand=True, vexpand=True,
        )
        self._preview_slot.add_css_class("card")
        outer.append(self._preview_slot)

        # Device picker — only added if >1 camera shows up.
        self._device_combo: Adw.ComboRow | None = None

        # Buttons row.
        btn_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            halign=Gtk.Align.END,
        )
        cancel_btn = Gtk.Button(label="Cancel")
        cancel_btn.connect("clicked", lambda *_: self.close())
        btn_row.append(cancel_btn)

        self._capture_btn = Gtk.Button(label="Capture")
        self._capture_btn.add_css_class("suggested-action")
        self._capture_btn.connect("clicked", self._on_capture_clicked)
        btn_row.append(self._capture_btn)
        outer.append(btn_row)

        toolbar.set_content(outer)
        self.set_child(toolbar)

        # Make sure the camera releases on every close path. Adw.Dialog
        # fires 'closed' regardless of how it was dismissed.
        self.connect("closed", lambda *_: self._teardown())

        # Defer pipeline start until the dialog has actually mapped, so
        # we don't hold the camera open during construction.
        GLib.idle_add(self._open_pipeline)

    # ── Pipeline lifecycle ────────────────────────────────────────────
    def _open_pipeline(self) -> bool:
        ok, reason = webcam.probe()
        if not ok:
            self._render_error(reason or "Webcam backend unavailable.")
            return False

        self._devices = webcam.list_devices()
        # If the saved device isn't connected this session, fall back to
        # the system default (None) and warn in the picker subtitle.
        saved = self._settings.webcam_device or ""
        device_id: str | None = saved if saved else None
        if saved and not any(d.id == saved for d in self._devices):
            log.info("Saved camera %s not found, using default", saved)
            device_id = None

        try:
            self._session = webcam.open_session(device_id)
            self._session.start()
        except Exception as e:  # noqa: BLE001
            log.exception("Could not start camera")
            self._render_error(f"Could not start camera: {e}")
            self._session = None
            return False

        self._render_preview()
        self._maybe_add_device_picker()
        return False  # do not re-fire idle_add

    def _teardown(self) -> None:
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                log.exception("Camera teardown failed")
            self._session = None

    # ── Preview rendering ─────────────────────────────────────────────
    def _clear_preview(self) -> None:
        child = self._preview_slot.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._preview_slot.remove(child)
            child = nxt

    def _render_preview(self) -> None:
        self._clear_preview()
        if self._session is None:
            return

        # Always render a Gtk.Picture. If the pipeline has a native
        # paintable sink, bind it directly; otherwise rely on the
        # session's set_preview_callback to push Gdk.Texture frames at
        # us (the slower but always-available manual fallback).
        picture = Gtk.Picture()
        picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        picture.set_hexpand(True)
        picture.set_vexpand(True)
        self._preview_slot.append(picture)
        self._preview_picture = picture

        paintable = self._session.paintable()
        if paintable is not None:
            picture.set_paintable(paintable)
            return

        # Manual decoded-frame path — works without the rs-gtk4 plugin.
        if hasattr(self._session, "set_preview_callback"):
            self._session.set_preview_callback(self._on_preview_texture)

    def _on_preview_texture(self, texture) -> None:
        """Called on the GTK main thread once per decoded preview frame."""
        if self._preview_picture is not None:
            self._preview_picture.set_paintable(texture)

    def _render_error(self, message: str) -> None:
        self._clear_preview()
        err = Adw.StatusPage(
            icon_name="dialog-error-symbolic",
            title="Camera unavailable",
            description=message,
        )
        self._preview_slot.append(err)
        self._capture_btn.set_sensitive(False)

    # ── Device picker ────────────────────────────────────────────────
    def _maybe_add_device_picker(self) -> None:
        if len(self._devices) <= 1 or self._device_combo is not None:
            return
        labels = [d.label or d.id for d in self._devices]
        combo = Adw.ComboRow(
            title="Camera",
            model=Gtk.StringList.new(labels),
        )
        # Highlight the current device.
        current = self._settings.webcam_device or ""
        for i, d in enumerate(self._devices):
            if d.id == current:
                combo.set_selected(i)
                break
        combo.connect("notify::selected", self._on_device_changed)

        group = Adw.PreferencesGroup()
        group.add(combo)
        self._preview_slot.get_parent().insert_child_after(group, self._preview_slot)
        self._device_combo = combo

    def _on_device_changed(self, combo: Adw.ComboRow, _pspec) -> None:
        idx = combo.get_selected()
        if not (0 <= idx < len(self._devices)):
            return
        new_id = self._devices[idx].id
        if new_id == (self._settings.webcam_device or ""):
            return
        self._settings.webcam_device = new_id
        self._settings.save()
        # Reopen with the new device. _teardown() then _open_pipeline().
        self._teardown()
        GLib.idle_add(self._open_pipeline)

    # ── Capture ──────────────────────────────────────────────────────
    def _on_capture_clicked(self, _btn: Gtk.Button) -> None:
        if self._session is None:
            return
        try:
            jpeg = self._session.capture_jpeg()
        except Exception as e:  # noqa: BLE001
            log.exception("Capture failed")
            self._render_error(f"Capture failed: {e}")
            return
        ts = int(time.time() * 1000)
        path = CAPTURES_DIR / f"box_{ts}.jpg"
        try:
            path.write_bytes(jpeg)
        except OSError as e:
            log.exception("Could not write capture")
            self._render_error(f"Could not save snapshot: {e}")
            return
        log.info("Captured frame: %s (%d bytes)", path, len(jpeg))
        self._on_captured(str(path))
        # Close after a tick so the click animation finishes naturally.
        GLib.idle_add(self.close)
