"""Inline live-mode panel (Phase 4.5 Tier 3).

A small bar that appears between the chat header and the message scroll
area while live mode is active. Hosts the webcam preview, a status pill
(Listening / Processing / Speaking), and an End button. The actual
audio + state machinery lives in :mod:`live_mode`; this file is purely
visual.
"""
from __future__ import annotations

from typing import Callable

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, Gtk  # noqa: E402

from .live_mode import LiveState


_STATE_LABELS = {
    LiveState.IDLE: "Idle",
    LiveState.LISTENING: "Listening…",
    LiveState.PROCESSING: "Thinking…",
    LiveState.SPEAKING: "Speaking",
}

_STATE_CLASSES = {
    LiveState.IDLE: "live-idle",
    LiveState.LISTENING: "live-listening",
    LiveState.PROCESSING: "live-processing",
    LiveState.SPEAKING: "live-speaking",
}


class LivePanel(Gtk.Box):
    """Compact UI: webcam preview on the left, status + End on the right."""

    def __init__(
        self,
        on_end: Callable[[], None],
        on_talk_start: Callable[[], None] | None = None,
        on_talk_end: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=12,
            margin_top=8, margin_bottom=8, margin_start=12, margin_end=12,
        )
        self.add_css_class("live-panel")
        self._on_talk_start = on_talk_start
        self._on_talk_end = on_talk_end
        self._ptt = False

        # Preview Picture — bound either to a paintable or pushed via
        # texture callbacks from the manual-decode fallback.
        self._picture = Gtk.Picture()
        self._picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self._picture.set_size_request(220, 124)
        self._picture.add_css_class("live-preview")
        self.append(self._picture)

        right = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=8,
            hexpand=True, valign=Gtk.Align.CENTER,
        )

        title = Gtk.Label(label="Live conversation", xalign=0.0)
        title.add_css_class("heading")
        right.append(title)

        self._status_label = Gtk.Label(
            label=_STATE_LABELS[LiveState.IDLE], xalign=0.0,
        )
        self._status_label.add_css_class("live-status")
        right.append(self._status_label)

        btn_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6,
        )
        # Hold-to-talk button — only shown in push-to-talk mode. A
        # GestureClick gives us press/release edges so capture runs exactly
        # while the button is held down.
        self._talk_btn = Gtk.Button(label="Hold to talk")
        self._talk_btn.add_css_class("suggested-action")
        self._talk_btn.set_visible(False)
        talk_gesture = Gtk.GestureClick()
        # Capture phase so our press/release edges fire even though the
        # Button has its own built-in click gesture in the bubble phase.
        talk_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        talk_gesture.connect("pressed", self._on_talk_pressed)
        talk_gesture.connect("released", self._on_talk_released)
        talk_gesture.connect("cancel", self._on_talk_cancelled)
        self._talk_btn.add_controller(talk_gesture)
        btn_row.append(self._talk_btn)

        end_btn = Gtk.Button(label="End")
        end_btn.add_css_class("destructive-action")
        end_btn.connect("clicked", lambda *_: on_end())
        btn_row.append(end_btn)
        right.append(btn_row)

        self.append(right)

    # ── push-to-talk button edges ─────────────────────────────────────
    def _on_talk_pressed(self, *_a) -> None:
        self._talk_btn.set_label("Listening… release to send")
        if self._on_talk_start is not None:
            self._on_talk_start()

    def _on_talk_released(self, *_a) -> None:
        self._talk_btn.set_label("Hold to talk")
        if self._on_talk_end is not None:
            self._on_talk_end()

    def _on_talk_cancelled(self, *_a) -> None:
        # Pointer left the button / gesture aborted — treat as release so
        # we don't leave capture stuck on.
        self._talk_btn.set_label("Hold to talk")
        if self._on_talk_end is not None:
            self._on_talk_end()

    # ── External API used by MainWindow ──────────────────────────────
    def set_preview_visible(self, visible: bool) -> None:
        """Voice-only live mode hides the preview tile entirely so the
        panel becomes a compact status banner."""
        self._picture.set_visible(visible)

    def set_ptt_visible(self, visible: bool) -> None:
        """Show the hold-to-talk button (push-to-talk mode) and adjust the
        idle status copy so the user knows the mic isn't auto-listening."""
        self._ptt = visible
        self._talk_btn.set_visible(visible)
        # Refresh the label in case we're already in LISTENING.
        self.set_state(LiveState.LISTENING if visible else LiveState.IDLE)

    def bind_preview(self, paintable) -> None:
        """Attach a Gdk.Paintable (or texture) for live frames."""
        if paintable is not None:
            self._picture.set_paintable(paintable)

    def push_preview_texture(self, texture) -> None:
        """Manual-fallback path: a per-frame texture from
        :class:`webcam.CameraSession.set_preview_callback`."""
        if texture is not None:
            self._picture.set_paintable(texture)

    def set_state(self, state: LiveState) -> None:
        label = _STATE_LABELS.get(state, "?")
        # In push-to-talk mode, "Listening…" is misleading (the mic isn't
        # auto-listening) — prompt the user to hold the button instead.
        if self._ptt and state is LiveState.LISTENING:
            label = "Hold the button to talk"
        self._status_label.set_text(label)
        # Cycle the CSS classes so themes can colour-code states.
        for cls in _STATE_CLASSES.values():
            self._status_label.remove_css_class(cls)
        cls = _STATE_CLASSES.get(state)
        if cls:
            self._status_label.add_css_class(cls)
