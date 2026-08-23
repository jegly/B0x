"""Chat view — message bubbles, composer, attachments, voice recording."""
from __future__ import annotations

import logging
import threading
import time
import wave
from pathlib import Path
from typing import Callable, Optional

import gi
gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")
from gi.repository import Adw, GLib, Gdk, Gtk, Pango  # noqa: E402

from .database import Message

log = logging.getLogger(__name__)

# Foreground colour used when rendering display-math images (matplotlib).
# Updated from the active theme via set_math_color(); the default suits the
# dark themes Box ships by default.
_MATH_COLOR = "#cdd6f4"

# Catppuccin soft-pastel palette used for the per-letter thinking animation.
_THINKING_COLORS = [
    "#cba6f7",  # mauve
    "#f5c2e7",  # pink
    "#f38ba8",  # red
    "#fab387",  # peach
    "#f9e2af",  # yellow
    "#a6e3a1",  # green
    "#94e2d5",  # teal
    "#89dceb",  # sky
    "#89b4fa",  # blue
    "#b4befe",  # lavender
]
_THINKING_TEXT = "thinking"


def set_math_color(hex_color: str) -> None:
    global _MATH_COLOR
    if hex_color:
        _MATH_COLOR = hex_color


class _Bubble(Gtk.Box):
    """Single message bubble.

    Outer box is a full-width horizontal row; inner _box holds the visible
    bubble with CSS styling. A spacer on the opposite side pushes the bubble
    left (assistant) or right (user), splitting available width 50/50.
    """

    def __init__(
        self,
        role: str,
        text: str = "",
        on_speak: Callable[[str], None] | None = None,
        on_save_memory: Callable[[str], None] | None = None,
        context: list[dict] | None = None,
    ):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, hexpand=True)
        self.role = role
        # Streaming throttle: tokens accumulate in _buffer and the label
        # repaints at most every 33 ms (per-token full-buffer set_text was
        # O(n²) and flooded the main loop on long replies).
        self._stream_flush_id: int = 0

        # Parse \x00-separated voice path embedded in user message content.
        voice_path: str | None = None
        display_text = text
        if role == "user" and text and "\x00" in text:
            display_text, voice_path = text.split("\x00", 1)
            if not Path(voice_path).is_file():
                voice_path = None  # file was deleted

        # Inner box carries all visible styling — NOT the outer row.
        self._box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self._box.add_css_class("bubble")
        self._box.add_css_class(f"bubble-{role}")
        self._box.set_hexpand(True)

        spacer = Gtk.Box(hexpand=True)

        if role == "user":
            self.append(spacer)
            self.append(self._box)
        else:
            self.append(self._box)
            self.append(spacer)

        # No "Assistant" tag on model bubbles — the alignment and palette
        # already say who's talking (Jegly's call). User/system keep theirs.
        if role != "assistant":
            role_label = Gtk.Label(
                label="You" if role == "user" else "System",
                xalign=0.0,
            )
            role_label.add_css_class("bubble-role")
            self._box.append(role_label)

        # Slot for the retrieved-context expander on assistant bubbles. We
        # always reserve the slot so set_context() can swap it in mid-stream.
        self._context_expander: Gtk.Expander | None = None
        # Slot for tool-call expanders, one per call. Stacked vertically
        # below the context expander, above the streaming body text. Each
        # call is a small Gtk.Expander; mid-stream additions just append.
        self._tool_calls_slot: Gtk.Box | None = None
        if role == "assistant":
            self._context_slot = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
            self._box.append(self._context_slot)
            self._tool_calls_slot = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL, spacing=4
            )
            self._box.append(self._tool_calls_slot)
            if context:
                self.set_context(context)
        else:
            self._context_slot = None

        self._buffer = display_text
        # Body holds the message content. During streaming it's a single
        # plain Gtk.Label (self._text). On finish/load it's rebuilt from
        # rendered Markdown segments — text labels + display-math widgets.
        self._body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self._body.set_hexpand(True)
        self._box.append(self._body)
        self._text = self._make_text_label()
        if display_text:
            self._text.set_text(display_text)
        self._body.append(self._text)

        # Thinking animation — shown while waiting for the first token.
        # Per-letter Catppuccin colours cycling as a wave via Pango markup.
        self._thinking_row: Gtk.Box | None = None
        self._thinking_anim: Gtk.Label | None = None
        self._thinking_timer: int = 0
        self._thinking_step: int = 0
        if role == "assistant" and not display_text:
            self._thinking_row = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=0,
                margin_top=4,
            )
            self._thinking_anim = Gtk.Label(xalign=0.0)
            self._thinking_anim.add_css_class("thinking-dots")
            self._thinking_row.append(self._thinking_anim)
            self._body.append(self._thinking_row)
            self._update_thinking_markup()
            self._thinking_timer = GLib.timeout_add(150, self._tick_thinking)

        if voice_path is not None:
            self._voice_path = voice_path
            self._voice_playing = False
            self._voice_stop = threading.Event()
            self._play_btn = Gtk.Button(
                icon_name="media-playback-start-symbolic",
                tooltip_text="Play voice message",
                halign=Gtk.Align.START,
                has_frame=False,
            )
            self._play_btn.add_css_class("flat")
            self._play_btn.connect("clicked", self._on_play_clicked)
            self._box.append(self._play_btn)
        else:
            self._voice_path = None
            self._play_btn = None

        # ── Action row (copy / speak / save-memory) ─────────────────────────
        # Always present as a child of _box so hovering over the buttons
        # doesn't trigger a leave event on _box. CSS opacity hides them
        # at rest — no height change means no bubble jump on hover.
        self._action_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        self._action_row.add_css_class("bubble-actions")

        self._copy_btn = Gtk.Button(
            icon_name="edit-copy-symbolic",
            tooltip_text="Copy text",
            has_frame=False,
        )
        self._copy_btn.add_css_class("flat")
        self._copy_btn.connect("clicked", self._on_copy_clicked)
        self._action_row.append(self._copy_btn)

        if role == "assistant" and on_speak is not None:
            speak_btn = Gtk.Button(
                icon_name="audio-speakers-symbolic",
                tooltip_text="Speak this message",
                has_frame=False,
            )
            speak_btn.add_css_class("flat")
            speak_btn.connect("clicked", lambda *_: on_speak(self.get_text()))
            self._action_row.append(speak_btn)

        if on_save_memory is not None:
            save_mem_btn = Gtk.Button(
                icon_name="user-bookmarks-symbolic",
                tooltip_text="Save as memory",
                has_frame=False,
            )
            save_mem_btn.add_css_class("flat")
            save_mem_btn.connect("clicked", lambda *_: on_save_memory(self.get_text()))
            self._action_row.append(save_mem_btn)

        self._box.append(self._action_row)

        motion = Gtk.EventControllerMotion()
        motion.connect("enter", lambda *_: self._action_row.add_css_class("bubble-actions-visible"))
        motion.connect("leave", lambda *_: self._action_row.remove_css_class("bubble-actions-visible"))
        self._box.add_controller(motion)

    # ── Voice playback ─────────────────────────────────────────────────────

    def _on_play_clicked(self, _btn) -> None:
        if self._voice_playing:
            self._voice_stop.set()
        else:
            self._voice_stop.clear()
            self._voice_playing = True
            self._play_btn.set_icon_name("media-playback-stop-symbolic")
            threading.Thread(target=self._do_play_voice, daemon=True).start()

    def _do_play_voice(self) -> None:
        try:
            import numpy as np
            import sounddevice as sd
            with wave.open(self._voice_path, "rb") as wf:
                sr = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(audio, samplerate=sr, blocking=False)
            while sd.get_stream().active:
                if self._voice_stop.is_set():
                    sd.stop()
                    break
                time.sleep(0.05)
        except Exception:
            pass
        finally:
            GLib.idle_add(self._on_play_done)

    def _on_play_done(self) -> bool:
        self._voice_playing = False
        if self._play_btn is not None:
            self._play_btn.set_icon_name("media-playback-start-symbolic")
        return False

    # ── Copy ───────────────────────────────────────────────────────────────

    def _on_copy_clicked(self, _btn) -> None:
        text = self.get_text()
        if not text:
            return
        Gdk.Display.get_default().get_clipboard().set(text)
        self._copy_btn.set_icon_name("object-select-symbolic")
        self._copy_btn.set_tooltip_text("Copied!")
        GLib.timeout_add(1500, self._reset_copy_btn)

    def _reset_copy_btn(self) -> bool:
        self._copy_btn.set_icon_name("edit-copy-symbolic")
        self._copy_btn.set_tooltip_text("Copy text")
        return False

    # ── Thinking animation ─────────────────────────────────────────────────

    def _update_thinking_markup(self) -> None:
        if self._thinking_anim is None:
            return
        parts: list[str] = []
        color_i = 0
        for ch in _THINKING_TEXT:
            if ch == " ":
                parts.append(" ")
            else:
                color = _THINKING_COLORS[
                    (self._thinking_step + color_i) % len(_THINKING_COLORS)
                ]
                parts.append(
                    f'<span color="{color}" weight="bold">'
                    f"{GLib.markup_escape_text(ch)}</span>"
                )
                color_i += 1
        self._thinking_anim.set_markup("".join(parts))

    def _tick_thinking(self) -> bool:
        if self._thinking_anim is None:
            return False
        self._thinking_step = (self._thinking_step + 1) % len(_THINKING_COLORS)
        self._update_thinking_markup()
        return True

    # ── Text accessors ─────────────────────────────────────────────────────

    def _make_text_label(self) -> Gtk.Label:
        lbl = Gtk.Label(
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
            xalign=0.0,
            yalign=0.0,
            selectable=True,
            hexpand=True,
        )
        lbl.set_size_request(0, -1)
        lbl.add_css_class("bubble-body")
        return lbl

    def append_text(self, chunk: str) -> None:
        if self._thinking_row is not None:
            if self._thinking_timer:
                GLib.source_remove(self._thinking_timer)
                self._thinking_timer = 0
            self._body.remove(self._thinking_row)
            self._thinking_row = None
            self._thinking_anim = None
        self._buffer += chunk
        if self._text is not None and not self._stream_flush_id:
            self._stream_flush_id = GLib.timeout_add(33, self._flush_stream)

    def _flush_stream(self) -> bool:
        self._stream_flush_id = 0
        if self._text is not None:
            self._text.set_text(self._buffer)
        return False

    def _cancel_stream_flush(self) -> None:
        if self._stream_flush_id:
            GLib.source_remove(self._stream_flush_id)
            self._stream_flush_id = 0

    def set_text(self, text: str) -> None:
        self._buffer = text
        self._clear_body()
        self._text = self._make_text_label()
        self._text.set_text(text)
        self._body.append(self._text)

    def get_text(self) -> str:
        return self._buffer

    def _clear_body(self) -> None:
        child = self._body.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._body.remove(child)
            child = nxt
        self._text = None

    def render_markdown(self) -> None:
        """Rebuild the body from rendered Markdown (assistant only). Text runs
        become Gtk.Labels with Pango markup; display-math blocks become
        rendered images (matplotlib) or a Unicode fallback label."""
        if self.role != "assistant":
            return
        # A pending stream flush would clobber the rendered markup with the
        # raw buffer text — kill it before rebuilding.
        self._cancel_stream_flush()
        md = self._buffer
        if not md.strip():
            return
        from . import mdrender
        try:
            segments = mdrender.to_segments(md)
        except Exception:
            log.debug("markdown segmentation failed", exc_info=True)
            return  # leave the plain streaming label in place
        self._clear_body()
        for kind, content in segments:
            if kind == "math":
                self._body.append(self._make_math_widget(content))
                continue
            lbl = self._make_text_label()
            try:
                lbl.set_markup(content)
            except Exception:
                # Malformed markup — Pango rejects the whole string. Fall back
                # to the raw markdown text for this run so nothing is lost.
                log.debug("set_markup failed; using plain text", exc_info=True)
                lbl.set_text(md)
            self._body.append(lbl)
        # Keep self._text pointing at a label so late append_text is harmless.
        last = self._body.get_last_child()
        self._text = last if isinstance(last, Gtk.Label) else None

    def _make_math_widget(self, tex: str) -> Gtk.Widget:
        from . import mdrender
        png = mdrender.render_math_png(tex, color=_MATH_COLOR)
        if png:
            try:
                texture = Gdk.Texture.new_from_bytes(GLib.Bytes.new(png))
                pic = Gtk.Picture.new_for_paintable(texture)
                pic.set_can_shrink(False)
                pic.set_halign(Gtk.Align.START)
                pic.add_css_class("math-display")
                return pic
            except Exception:
                log.debug("math image build failed; Unicode fallback", exc_info=True)
        lbl = self._make_text_label()
        try:
            lbl.set_markup(mdrender._display_math(tex))
        except Exception:
            lbl.set_text(tex)
        return lbl

    # ── Retrieved-context card (assistant only) ────────────────────────────

    def set_context(self, chunks: list[dict]) -> None:
        """Render a collapsible expander showing the chunks the RAG layer
        injected into this reply. ``chunks`` is a list of dicts with keys:
            label   — source filename (e.g. 'foo.pdf')
            chunk_idx — int
            text    — full chunk text
            score   — float (optional)
        """
        if self._context_slot is None or not chunks:
            return
        # Drain prior expander (allowed to re-set during stream).
        child = self._context_slot.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._context_slot.remove(child)
            child = nxt

        n = len(chunks)
        # Distinct source count for a nicer label.
        sources = sorted({c.get("label") or "inline" for c in chunks})
        if len(sources) == 1:
            title = f"Used 1 source · {n} snippet{'s' if n != 1 else ''}"
        else:
            title = f"Used {len(sources)} sources · {n} snippets"
        exp = Gtk.Expander(label=title)
        exp.add_css_class("rag-context")
        body = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4,
                       margin_top=4, margin_start=8, margin_end=8, margin_bottom=4)
        for c in chunks:
            row = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            head_text = f"{c.get('label') or 'inline'}  ·  chunk {c.get('chunk_idx', 0)}"
            score = c.get("score")
            if isinstance(score, (int, float)):
                head_text += f"  ·  score {score:.2f}"
            head = Gtk.Label(label=head_text, xalign=0.0)
            head.add_css_class("caption-heading")
            head.add_css_class("dim-label")
            row.append(head)
            preview = (c.get("text") or "").strip()
            if len(preview) > 400:
                preview = preview[:400].rstrip() + "…"
            body_lbl = Gtk.Label(
                label=preview, xalign=0.0, wrap=True,
                wrap_mode=Pango.WrapMode.WORD_CHAR, selectable=True,
            )
            body_lbl.add_css_class("caption")
            row.append(body_lbl)
            body.append(row)
        exp.set_child(body)
        self._context_slot.append(exp)
        self._context_expander = exp

    # ── Tool-call cards (assistant only) ───────────────────────────────────
    def add_tool_call(
        self,
        fn_name: str,
        args: dict,
        result: str,
        *,
        denied: bool = False,
    ) -> None:
        """Append a collapsible card showing a tool call the model made.

        Same visual pattern as the RAG context expander: a one-line header
        that summarises the call, click-to-expand body showing the args
        and the tool's full response.
        """
        if self._tool_calls_slot is None:
            return

        args_summary = ", ".join(
            f"{k}={_short_repr(v)}" for k, v in (args or {}).items()
        )
        if len(args_summary) > 90:
            args_summary = args_summary[:87] + "…"
        icon = "🚫" if denied else "🛠"
        title = f"{icon} {fn_name}({args_summary})"

        exp = Gtk.Expander(label=title)
        exp.add_css_class("tool-call-card")
        if denied:
            exp.add_css_class("tool-call-denied")

        body = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=6,
            margin_top=6, margin_start=8, margin_end=8, margin_bottom=6,
        )

        if args:
            args_head = Gtk.Label(label="Arguments", xalign=0.0)
            args_head.add_css_class("caption-heading")
            args_head.add_css_class("dim-label")
            body.append(args_head)
            for k, v in args.items():
                row_lbl = Gtk.Label(
                    label=f"{k} = {v!r}",
                    xalign=0.0,
                    wrap=True,
                    wrap_mode=Pango.WrapMode.WORD_CHAR,
                    selectable=True,
                )
                row_lbl.add_css_class("caption")
                body.append(row_lbl)

        result_head_text = "Denied" if denied else "Result"
        result_head = Gtk.Label(label=result_head_text, xalign=0.0)
        result_head.add_css_class("caption-heading")
        result_head.add_css_class("dim-label")
        body.append(result_head)

        result_text = (result or "").strip() or "(no output)"
        if len(result_text) > 2000:
            result_text = result_text[:2000].rstrip() + "\n…(truncated)"
        result_lbl = Gtk.Label(
            label=result_text,
            xalign=0.0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
            selectable=True,
        )
        result_lbl.add_css_class("caption")
        body.append(result_lbl)

        exp.set_child(body)
        self._tool_calls_slot.append(exp)


def _short_repr(v) -> str:
    """Compact value preview for tool-call header labels."""
    if isinstance(v, str):
        s = v if len(v) <= 40 else v[:37] + "…"
        return repr(s)
    return repr(v)


class ChatView(Gtk.Box):
    """Scrolling message list + multi-line input + send/stop/attach/mic."""

    def __init__(
        self,
        on_send: Callable[[str, list[dict]], None],
        on_stop: Callable[[], None],
        on_attach: Callable[[], None] | None = None,
        on_mic_toggle: Callable[[], None] | None = None,
        on_speak: Callable[[str], None] | None = None,
        on_stop_tts: Callable[[], None] | None = None,
        on_camera: Callable[[], None] | None = None,
        on_live_toggle: Callable[[bool], None] | None = None,
        on_agent_toggle: Callable[[bool], None] | None = None,
        on_voice_live_toggle: Callable[[bool], None] | None = None,
        on_websearch_toggle: Callable[[bool], None] | None = None,
        on_fs_toggle: Callable[[bool], None] | None = None,
        on_tts_toggle: Callable[[bool], None] | None = None,
        on_tts_volume_changed: Callable[[float], None] | None = None,
        on_memory_toggle: Callable[[bool], None] | None = None,
        on_memory_save: Callable[[], None] | None = None,
        on_save_memory: Callable[[str], None] | None = None,
        initial_tts_enabled: bool = False,
        initial_tts_volume: float = 1.0,
        initial_memory_enabled: bool = False,
        settings=None,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0)

        self._settings = settings
        self._on_send = on_send
        self._on_stop = on_stop
        self._on_attach = on_attach
        self._on_camera = on_camera
        self._on_live_toggle = on_live_toggle
        self._on_agent_toggle = on_agent_toggle
        self._on_voice_live_toggle = on_voice_live_toggle
        self._on_websearch_toggle = on_websearch_toggle
        self._on_fs_toggle = on_fs_toggle
        self._on_tts_toggle = on_tts_toggle
        self._on_tts_volume_changed = on_tts_volume_changed
        self._on_memory_toggle = on_memory_toggle
        self._on_memory_save = on_memory_save
        self._on_save_memory = on_save_memory
        self._on_mic_toggle = on_mic_toggle
        self._on_speak = on_speak
        self._on_stop_tts = on_stop_tts
        self._streaming_bubble: Optional[_Bubble] = None
        self._is_generating = False
        self._is_recording = False

        # Pending attachments — cleared on each send.
        self._attachments: list[dict] = []

        # ── Scrolling message list ─────────────────────────────────────────
        self._messages_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=2,
            margin_start=16, margin_end=16, margin_top=12, margin_bottom=12,
            hexpand=True,
        )
        scroller = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
        )
        scroller.set_propagate_natural_height(False)
        scroller.set_propagate_natural_width(False)
        scroller.add_css_class("chat-scroller")
        self._scroller = scroller

        vadj = scroller.get_vadjustment()
        vadj.connect("changed", self._maybe_scroll_to_bottom)
        vadj.connect("value-changed", self._on_user_scroll)
        self._stick_to_bottom = True

        self._empty_state = self._build_empty_state()
        scroller.set_child(self._empty_state)
        self.append(scroller)

        # ── Composer area ──────────────────────────────────────────────────
        composer_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        composer_box.add_css_class("input-area")
        composer_box.append(Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL))

        # Context-usage bar — thin row showing estimated token budget usage.
        # Hidden by default; revealed via set_context_bar_visible(True).
        ctx_row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            margin_top=2, margin_bottom=2, margin_start=10, margin_end=10,
        )
        self._ctx_bar = Gtk.ProgressBar(hexpand=True, valign=Gtk.Align.CENTER)
        self._ctx_bar.add_css_class("context-bar")
        self._ctx_label = Gtk.Label(label="", xalign=1.0)
        self._ctx_label.add_css_class("caption")
        self._ctx_label.add_css_class("dim-label")
        ctx_row.append(self._ctx_bar)
        ctx_row.append(self._ctx_label)
        ctx_row.set_visible(False)
        self._ctx_row = ctx_row
        composer_box.append(ctx_row)

        # Attachment chips row (hidden when empty).
        self._chips_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            margin_top=6, margin_start=8, margin_end=8,
        )
        self._chips_box.set_visible(False)
        composer_box.append(self._chips_box)

        # Input row
        row = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            margin_top=8, margin_bottom=8, margin_start=8, margin_end=8,
        )

        # Attach button
        if on_attach is not None:
            attach_btn = Gtk.Button(
                icon_name="mail-attachment-symbolic",
                tooltip_text="Attach file",
                valign=Gtk.Align.END,
            )
            attach_btn.add_css_class("flat")
            attach_btn.connect("clicked", lambda *_: self._on_attach())
            row.append(attach_btn)

        # Camera button (Phase 4.5) — shown only when on_camera wired AND
        # the window has marked the webcam feature as available for this
        # session via ``set_camera_visible(True)``. Default hidden so
        # users without a working backend never see a dead button.
        self._camera_btn: Gtk.Button | None = None
        if on_camera is not None:
            self._camera_btn = Gtk.Button(
                icon_name="camera-photo-symbolic",
                tooltip_text="Capture from webcam",
                valign=Gtk.Align.END,
            )
            self._camera_btn.add_css_class("flat")
            self._camera_btn.set_visible(False)
            self._camera_btn.connect("clicked", lambda *_: self._on_camera())
            row.append(self._camera_btn)

        # Live-mode toggle (Phase 4.5 Tier 3) — Parlor-style hands-free
        # conversation. Hidden by default; window flips it on when the
        # webcam feature is enabled.
        self._live_btn: Gtk.ToggleButton | None = None
        if on_live_toggle is not None:
            self._live_btn = Gtk.ToggleButton(
                icon_name="video-display-symbolic",
                tooltip_text="Live conversation mode",
                valign=Gtk.Align.END,
            )
            self._live_btn.add_css_class("flat")
            self._live_btn.set_visible(False)
            self._suppress_live_toggle = False
            def _live_toggled(b: Gtk.ToggleButton) -> None:
                if self._suppress_live_toggle:
                    return
                self._on_live_toggle(b.get_active())  # type: ignore[misc]
            self._live_btn.connect("toggled", _live_toggled)

        # Agent mode toggle (Phase 5). Multi-step autonomous mode that
        # encourages the model to chain tools across iterations. Same
        # visibility rules as the camera/live buttons — only shown when
        # tools are available (otherwise there's nothing to chain).
        self._agent_btn: Gtk.ToggleButton | None = None
        if on_agent_toggle is not None:
            self._agent_btn = Gtk.ToggleButton(
                icon_name="emblem-system-symbolic",
                tooltip_text="Agent mode (multi-step chaining)",
                valign=Gtk.Align.END,
            )
            self._agent_btn.add_css_class("flat")
            self._agent_btn.set_visible(False)
            self._suppress_agent_toggle = False
            def _agent_toggled(b: Gtk.ToggleButton) -> None:
                if self._suppress_agent_toggle:
                    return
                self._on_agent_toggle(b.get_active())  # type: ignore[misc]
            self._agent_btn.connect("toggled", _agent_toggled)

        # Web-search quick toggle. Lets the user allow a one-off web search
        # without turning on full agent mode. When agent mode is OFF this is
        # the only way tools reach the model.
        self._websearch_btn: Gtk.ToggleButton | None = None
        if on_websearch_toggle is not None:
            self._websearch_btn = Gtk.ToggleButton(
                icon_name="applications-internet-symbolic",
                tooltip_text="Web search",
                valign=Gtk.Align.END,
            )
            self._websearch_btn.add_css_class("flat")
            self._websearch_btn.set_visible(False)
            self._suppress_websearch_toggle = False
            def _websearch_toggled(b: Gtk.ToggleButton) -> None:
                if self._suppress_websearch_toggle:
                    return
                self._on_websearch_toggle(b.get_active())  # type: ignore[misc]
            self._websearch_btn.connect("toggled", _websearch_toggled)
            row.append(self._websearch_btn)

        # Filesystem quick toggle — mirrors 🌐. Lets the user grant file
        # access without turning on full agent mode. Permission prompts
        # still fire for every fs_read/list/grep/write/delete call.
        self._fs_btn: Gtk.ToggleButton | None = None
        if on_fs_toggle is not None:
            self._fs_btn = Gtk.ToggleButton(
                icon_name="folder-symbolic",
                tooltip_text="Filesystem access",
                valign=Gtk.Align.END,
            )
            self._fs_btn.add_css_class("flat")
            self._fs_btn.set_visible(False)
            self._suppress_fs_toggle = False
            def _fs_toggled(b: Gtk.ToggleButton) -> None:
                if self._suppress_fs_toggle:
                    return
                self._on_fs_toggle(b.get_active())  # type: ignore[misc]
            self._fs_btn.connect("toggled", _fs_toggled)
            row.append(self._fs_btn)

        # TTS auto-speak quick control — MenuButton opens a small popover with
        # a Speak-replies switch + a volume slider. One composer slot, both
        # controls. Icon reflects (enabled, volume) so the user can read
        # state at a glance.
        self._tts_btn: Gtk.MenuButton | None = None
        self._tts_switch: Gtk.Switch | None = None
        self._tts_volume_scale: Gtk.Scale | None = None
        if on_tts_toggle is not None:
            self._tts_btn = Gtk.MenuButton(
                icon_name="audio-volume-high-symbolic",
                tooltip_text="Speak replies — click for volume",
                valign=Gtk.Align.END,
            )
            self._tts_btn.add_css_class("flat")
            self._tts_btn.set_visible(False)
            pop = Gtk.Popover()
            pop_box = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL, spacing=10,
                margin_top=10, margin_bottom=10,
                margin_start=12, margin_end=12,
            )
            pop_box.set_size_request(220, -1)

            switch_row = Gtk.Box(
                orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
            )
            switch_label = Gtk.Label(label="Speak replies", xalign=0.0,
                                     hexpand=True)
            self._tts_switch = Gtk.Switch(
                active=bool(initial_tts_enabled), valign=Gtk.Align.CENTER,
            )
            self._suppress_tts_switch = False

            def _switch_changed(sw: Gtk.Switch, *_a) -> bool:
                if self._suppress_tts_switch:
                    return False
                self._on_tts_toggle(sw.get_active())  # type: ignore[misc]
                self._update_tts_icon()
                return False

            self._tts_switch.connect("notify::active", _switch_changed)
            switch_row.append(switch_label)
            switch_row.append(self._tts_switch)
            pop_box.append(switch_row)

            vol_label = Gtk.Label(label="Volume", xalign=0.0)
            vol_label.add_css_class("caption")
            vol_label.add_css_class("dim-label")
            pop_box.append(vol_label)
            # 0% – 150% in 5% steps. 100% = native Piper level.
            self._tts_volume_scale = Gtk.Scale.new_with_range(
                Gtk.Orientation.HORIZONTAL, 0.0, 1.5, 0.05,
            )
            self._tts_volume_scale.set_value(float(initial_tts_volume))
            self._tts_volume_scale.set_draw_value(False)
            self._tts_volume_scale.set_hexpand(True)
            for mark, label in ((0.0, "0%"), (1.0, "100%"), (1.5, "150%")):
                self._tts_volume_scale.add_mark(
                    mark, Gtk.PositionType.BOTTOM, label,
                )
            self._suppress_tts_volume = False

            def _volume_changed(scale: Gtk.Scale) -> None:
                if self._suppress_tts_volume:
                    return
                v = float(scale.get_value())
                if self._on_tts_volume_changed is not None:
                    self._on_tts_volume_changed(v)
                self._update_tts_icon()

            self._tts_volume_scale.connect("value-changed", _volume_changed)
            pop_box.append(self._tts_volume_scale)

            pop.set_child(pop_box)
            self._tts_btn.set_popover(pop)
            row.append(self._tts_btn)

        # Memory quick control (Phase 6) — MenuButton popover with a
        # "Use saved memory" switch + a button to save the typed message as a
        # memory. Capture is always explicit; nothing is stored automatically.
        self._memory_btn: Gtk.MenuButton | None = None
        self._memory_switch: Gtk.Switch | None = None
        if on_memory_toggle is not None:
            self._memory_btn = Gtk.MenuButton(
                icon_name="user-bookmarks-symbolic",
                tooltip_text="Memory — use & save long-term facts",
                valign=Gtk.Align.END,
            )
            self._memory_btn.add_css_class("flat")
            self._memory_btn.set_visible(False)
            mpop = Gtk.Popover()
            mbox = Gtk.Box(
                orientation=Gtk.Orientation.VERTICAL, spacing=10,
                margin_top=10, margin_bottom=10, margin_start=12, margin_end=12,
            )
            mbox.set_size_request(240, -1)

            msw_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            msw_label = Gtk.Label(label="Use saved memory", xalign=0.0,
                                  hexpand=True)
            self._memory_switch = Gtk.Switch(
                active=bool(initial_memory_enabled), valign=Gtk.Align.CENTER,
            )
            self._suppress_memory_switch = False

            def _mem_switch_changed(sw: Gtk.Switch, *_a) -> bool:
                if self._suppress_memory_switch:
                    return False
                self._on_memory_toggle(sw.get_active())  # type: ignore[misc]
                self._update_memory_icon()
                return False

            self._memory_switch.connect("notify::active", _mem_switch_changed)
            msw_row.append(msw_label)
            msw_row.append(self._memory_switch)
            mbox.append(msw_row)

            save_btn = Gtk.Button(label="Remember typed message")
            save_btn.add_css_class("flat")

            def _mem_save_clicked(*_a) -> None:
                if self._on_memory_save is not None:
                    self._on_memory_save()

            save_btn.connect("clicked", _mem_save_clicked)
            mbox.append(save_btn)

            mhint = Gtk.Label(
                label="Saves what's in the box right now. Manage in "
                      "Preferences → Memory.",
                xalign=0.0, wrap=True,
            )
            mhint.add_css_class("caption")
            mhint.add_css_class("dim-label")
            mbox.append(mhint)

            mpop.set_child(mbox)
            self._memory_btn.set_popover(mpop)
            row.append(self._memory_btn)

        if self._agent_btn is not None:
            row.append(self._agent_btn)
        if self._live_btn is not None:
            row.append(self._live_btn)

        # Text input
        self._input = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            accepts_tab=False,
            hexpand=True,
            top_margin=6, bottom_margin=6, left_margin=8, right_margin=8,
        )
        self._input.set_name("box-composer-input")
        self._input.add_css_class("composer")

        self._input_bg_provider = Gtk.CssProvider()
        self._input.get_style_context().add_provider(
            self._input_bg_provider, 1000
        )

        key_ctrl = Gtk.EventControllerKey()
        key_ctrl.connect("key-pressed", self._on_input_key)
        self._input.add_controller(key_ctrl)

        self._copy_conv_btn = Gtk.Button(
            icon_name="edit-copy-symbolic",
            tooltip_text="Copy conversation",
            valign=Gtk.Align.END,
        )
        self._copy_conv_btn.add_css_class("flat")
        self._copy_conv_btn.connect("clicked", lambda *_: self._copy_conversation())
        row.append(self._copy_conv_btn)

        input_frame = Gtk.Box()
        input_frame.add_css_class("composer-frame")
        input_frame.set_size_request(-1, 60)
        input_frame.append(self._input)
        row.append(input_frame)

        # Voice-only live mode toggle (Phase 4.5 follow-up). Sits right
        # before the mic so the user reads "voice conversation" next to
        # "one-shot record". Same panel + state-pill flow as Live mode,
        # just no webcam preview.
        self._voice_live_btn: Gtk.ToggleButton | None = None
        if on_voice_live_toggle is not None:
            self._voice_live_btn = Gtk.ToggleButton(
                icon_name="audio-headphones-symbolic",
                tooltip_text="Voice conversation mode (no camera)",
                valign=Gtk.Align.END,
            )
            self._voice_live_btn.add_css_class("flat")
            self._voice_live_btn.set_visible(False)
            self._suppress_voice_live_toggle = False
            def _voice_live_toggled(b: Gtk.ToggleButton) -> None:
                if self._suppress_voice_live_toggle:
                    return
                self._on_voice_live_toggle(b.get_active())  # type: ignore[misc]
            self._voice_live_btn.connect("toggled", _voice_live_toggled)
            row.append(self._voice_live_btn)

        # Mic button
        if on_mic_toggle is not None:
            self._mic_btn = Gtk.Button(
                icon_name="audio-input-microphone-symbolic",
                tooltip_text="Hold to record voice",
                valign=Gtk.Align.END,
            )
            self._mic_btn.add_css_class("flat")
            self._mic_btn.connect("clicked", lambda *_: self._on_mic_toggle())
            row.append(self._mic_btn)
        else:
            self._mic_btn = None

        # TTS stop button — visible only while speaking
        self._tts_stop_btn = Gtk.Button(
            icon_name="media-playback-stop-symbolic",
            tooltip_text="Stop speaking",
            valign=Gtk.Align.END,
            visible=False,
        )
        self._tts_stop_btn.add_css_class("flat")
        self._tts_stop_btn.connect("clicked", lambda *_: self._on_stop_tts and self._on_stop_tts())
        row.append(self._tts_stop_btn)

        # Send / Stop
        self._send_btn = Gtk.Button(
            icon_name="document-send-symbolic",
            tooltip_text="Send (Enter)",
            valign=Gtk.Align.END,
        )
        self._send_btn.add_css_class("circular")
        self._send_btn.add_css_class("suggested-action")
        self._send_btn.connect("clicked", lambda *_: self._send_clicked())
        row.append(self._send_btn)

        self._stop_btn = Gtk.Button(
            icon_name="media-playback-stop-symbolic",
            tooltip_text="Stop generating",
            valign=Gtk.Align.END,
            visible=False,
        )
        self._stop_btn.add_css_class("circular")
        self._stop_btn.add_css_class("destructive-action")
        self._stop_btn.connect("clicked", lambda *_: self._on_stop())
        row.append(self._stop_btn)

        composer_box.append(row)
        self.append(composer_box)

    # ── Public API ─────────────────────────────────────────────────────────

    def clear(self) -> None:
        child = self._messages_box.get_first_child()
        while child is not None:
            nxt = child.get_next_sibling()
            self._messages_box.remove(child)
            child = nxt
        self._scroller.set_child(self._empty_state)
        self._streaming_bubble = None

    def load_messages(self, messages: list[Message]) -> None:
        import json
        self.clear()
        # clear() drops the streaming bubble; ensure the composer's
        # stop-button / input-sensitive state matches (a generation may
        # still be in flight for a different conv — the window restores
        # the bubble via restore_assistant_stream() in that case).
        self._set_generating(False)
        if messages:
            self._scroller.set_child(self._messages_box)
        for m in messages:
            if m.role not in ("user", "assistant", "system"):
                continue
            ctx = None
            cj = getattr(m, "context_json", None)
            if m.role == "assistant" and cj:
                try:
                    ctx = json.loads(cj)
                except Exception:
                    ctx = None
            bubble = _Bubble(
                m.role, m.content, on_speak=self._on_speak,
                on_save_memory=self._on_save_memory, context=ctx,
            )
            self._messages_box.append(bubble)
            if m.role == "assistant":
                bubble.render_markdown()
        self._stick_to_bottom = True
        GLib.idle_add(self._scroll_to_bottom_now)

    def append_user_message(self, text: str) -> None:
        if self._messages_box.get_first_child() is None:
            self._scroller.set_child(self._messages_box)
        self._messages_box.append(
            _Bubble("user", text, on_save_memory=self._on_save_memory)
        )
        self._stick_to_bottom = True

    def add_assistant_message(self, text: str) -> None:
        """One-shot assistant bubble — for live mode and other paths
        that already have the full response text in hand. Bypasses the
        streaming machinery."""
        if self._messages_box.get_first_child() is None:
            self._scroller.set_child(self._messages_box)
        bubble = _Bubble(
            "assistant", text, on_speak=self._on_speak,
            on_save_memory=self._on_save_memory,
        )
        self._messages_box.append(bubble)
        bubble.render_markdown()
        self._stick_to_bottom = True

    def start_assistant_stream(self, context: list[dict] | None = None) -> None:
        self._streaming_bubble = _Bubble(
            "assistant", "", on_speak=self._on_speak,
            on_save_memory=self._on_save_memory, context=context,
        )
        self._messages_box.append(self._streaming_bubble)
        self._set_generating(True)
        self._stick_to_bottom = True

    def restore_assistant_stream(
        self,
        text: str,
        context: list[dict] | None = None,
        tool_calls: list[tuple] | None = None,
    ) -> None:
        """Re-attach a streaming bubble pre-loaded with accumulated text
        and tool-call cards. Used when the user switches into a chat that
        has a background generation in flight."""
        if self._messages_box.get_first_child() is None:
            self._scroller.set_child(self._messages_box)
        self._streaming_bubble = _Bubble(
            "assistant", text, on_speak=self._on_speak,
            on_save_memory=self._on_save_memory, context=context,
        )
        if tool_calls:
            for fn_name, args, result, denied in tool_calls:
                self._streaming_bubble.add_tool_call(
                    fn_name, args, result, denied=denied,
                )
        self._messages_box.append(self._streaming_bubble)
        self._set_generating(True)
        self._stick_to_bottom = True
        GLib.idle_add(self._scroll_to_bottom_now)

    def append_token(self, text: str) -> None:
        if self._streaming_bubble is None:
            self.start_assistant_stream()
        self._streaming_bubble.append_text(text)

    def finish_stream(self) -> str:
        bubble = self._streaming_bubble
        text = bubble.get_text() if bubble else ""
        if bubble is not None:
            # If the model produced no visible text (e.g. it only emitted
            # tool-call chunks, or the audio path produced an empty result)
            # the bubble would otherwise be a blank ghost. Surface that as
            # an explicit "(no reply)" so the user knows what happened.
            if not text.strip():
                bubble.set_text(
                    "(no reply — model produced no text. Try rephrasing.)"
                )
            else:
                # Swap the plain streaming text for rendered Markdown/LaTeX.
                bubble.render_markdown()
        self._streaming_bubble = None
        self._set_generating(False)
        return text

    def set_input_bg(self, color: str) -> None:
        css = (
            f"* {{ background: {color}; background-color: {color}; }}\n"
            f"text {{ background: {color}; background-color: {color}; }}"
        )
        self._input_bg_provider.load_from_data(css.encode())

    def set_input_sensitive(self, sensitive: bool) -> None:
        self._input.set_sensitive(sensitive)
        self._send_btn.set_sensitive(sensitive)

    def focus_input(self) -> None:
        self._input.grab_focus()

    # ── Attachments ────────────────────────────────────────────────────────

    def add_attachment(self, att: dict) -> None:
        """Add an attachment dict and show its chip. att must have 'name' key."""
        self._attachments.append(att)
        self._add_chip(att)
        self._chips_box.set_visible(True)

    def trigger_send(self) -> None:
        self._send_clicked()

    def set_tts_speaking(self, speaking: bool) -> None:
        self._tts_stop_btn.set_visible(speaking)

    def set_camera_visible(self, visible: bool) -> None:
        """Window calls this when it decides the webcam feature is
        available + enabled for this chat. Default-hidden keeps the
        composer clean for users who never enable it."""
        if self._camera_btn is not None:
            self._camera_btn.set_visible(visible)
        if self._live_btn is not None:
            self._live_btn.set_visible(visible)

    def set_live_active(self, active: bool) -> None:
        """Reflect the controller's view of live mode in the toggle
        button without firing the handler back into the controller."""
        if self._live_btn is None:
            return
        if self._live_btn.get_active() == active:
            return
        self._suppress_live_toggle = True
        try:
            self._live_btn.set_active(active)
        finally:
            self._suppress_live_toggle = False

    def set_agent_visible(self, visible: bool) -> None:
        if self._agent_btn is not None:
            self._agent_btn.set_visible(visible)

    def set_voice_live_visible(self, visible: bool) -> None:
        if self._voice_live_btn is not None:
            self._voice_live_btn.set_visible(visible)

    def set_voice_live_active(self, active: bool) -> None:
        if self._voice_live_btn is None:
            return
        if self._voice_live_btn.get_active() == active:
            return
        self._suppress_voice_live_toggle = True
        try:
            self._voice_live_btn.set_active(active)
        finally:
            self._suppress_voice_live_toggle = False

    def set_agent_active(self, active: bool) -> None:
        """Reflect the window's effective-agent state in the toggle
        without echoing back to the handler."""
        if self._agent_btn is None:
            return
        if self._agent_btn.get_active() == active:
            return
        self._suppress_agent_toggle = True
        try:
            self._agent_btn.set_active(active)
        finally:
            self._suppress_agent_toggle = False

    def set_websearch_visible(self, visible: bool) -> None:
        if self._websearch_btn is not None:
            self._websearch_btn.set_visible(visible)

    def set_websearch_active(self, active: bool) -> None:
        """Reflect the window's effective web-search state without echoing
        back to the handler."""
        if self._websearch_btn is None:
            return
        if self._websearch_btn.get_active() == active:
            return
        self._suppress_websearch_toggle = True
        try:
            self._websearch_btn.set_active(active)
        finally:
            self._suppress_websearch_toggle = False

    def set_fs_visible(self, visible: bool) -> None:
        if self._fs_btn is not None:
            self._fs_btn.set_visible(visible)

    def set_fs_active(self, active: bool) -> None:
        if self._fs_btn is None:
            return
        if self._fs_btn.get_active() == active:
            return
        self._suppress_fs_toggle = True
        try:
            self._fs_btn.set_active(active)
        finally:
            self._suppress_fs_toggle = False

    def set_tts_visible(self, visible: bool) -> None:
        if self._tts_btn is not None:
            self._tts_btn.set_visible(visible)

    def set_tts_state(self, enabled: bool, volume: float) -> None:
        """Sync the popover's switch + slider with persisted settings —
        called on chat-view init and after Preferences changes. Both
        widgets fire suppression flags so the bound handlers don't echo
        back into _on_tts_toggle / _on_tts_volume_changed."""
        if self._tts_switch is not None and self._tts_switch.get_active() != bool(enabled):
            self._suppress_tts_switch = True
            try:
                self._tts_switch.set_active(bool(enabled))
            finally:
                self._suppress_tts_switch = False
        if self._tts_volume_scale is not None:
            self._suppress_tts_volume = True
            try:
                self._tts_volume_scale.set_value(float(volume))
            finally:
                self._suppress_tts_volume = False
        self._update_tts_icon()

    def _update_tts_icon(self) -> None:
        """Pick the audio-volume icon based on (enabled, volume) so a glance
        at the composer tells the user whether they'll hear replies."""
        if self._tts_btn is None:
            return
        enabled = bool(self._tts_switch.get_active()) if self._tts_switch else False
        vol = float(self._tts_volume_scale.get_value()) if self._tts_volume_scale else 1.0
        if not enabled or vol <= 0.0:
            icon = "audio-volume-muted-symbolic"
        elif vol < 0.34:
            icon = "audio-volume-low-symbolic"
        elif vol < 0.85:
            icon = "audio-volume-medium-symbolic"
        else:
            icon = "audio-volume-high-symbolic"
        self._tts_btn.set_icon_name(icon)
        # Tint when enabled so it reads as "armed" at a glance.
        if enabled:
            self._tts_btn.add_css_class("tts-on")
        else:
            self._tts_btn.remove_css_class("tts-on")

    # ── memory (Phase 6) ──────────────────────────────────────────────────
    def set_memory_visible(self, visible: bool) -> None:
        if self._memory_btn is not None:
            self._memory_btn.set_visible(visible)

    def set_memory_state(self, enabled: bool) -> None:
        """Sync the popover switch with the persisted setting (suppressed
        so it doesn't echo back into _on_memory_toggle)."""
        if (self._memory_switch is not None
                and self._memory_switch.get_active() != bool(enabled)):
            self._suppress_memory_switch = True
            try:
                self._memory_switch.set_active(bool(enabled))
            finally:
                self._suppress_memory_switch = False
        self._update_memory_icon()

    def _update_memory_icon(self) -> None:
        if self._memory_btn is None:
            return
        enabled = bool(self._memory_switch.get_active()) if self._memory_switch else False
        if enabled:
            self._memory_btn.add_css_class("tts-on")  # reuse accent-tint class
        else:
            self._memory_btn.remove_css_class("tts-on")

    def _copy_conversation(self) -> None:
        parts: list[str] = []
        child = self._messages_box.get_first_child()
        while child is not None:
            if isinstance(child, _Bubble):
                label = "You" if child.role == "user" else "Assistant"
                text = child.get_text().strip()
                if text:
                    parts.append(f"{label}: {text}")
            child = child.get_next_sibling()
        if not parts:
            return
        Gdk.Display.get_default().get_clipboard().set("\n\n".join(parts))
        self._copy_conv_btn.set_icon_name("object-select-symbolic")
        self._copy_conv_btn.set_tooltip_text("Copied!")
        GLib.timeout_add(1500, self._reset_copy_conv_btn)

    def _reset_copy_conv_btn(self) -> bool:
        self._copy_conv_btn.set_icon_name("edit-copy-symbolic")
        self._copy_conv_btn.set_tooltip_text("Copy conversation")
        return False

    def get_composer_text(self) -> str:
        """Current text in the input box (unstripped)."""
        buf = self._input.get_buffer()
        return buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

    def set_context_bar_visible(self, visible: bool) -> None:
        self._ctx_row.set_visible(visible)

    def set_context_usage(self, used: int, total: int) -> None:
        """Update the context-usage bar with estimated tokens used / total."""
        if total <= 0:
            self._ctx_bar.set_fraction(0.0)
            self._ctx_label.set_text("")
            return
        frac = max(0.0, min(1.0, used / total))
        self._ctx_bar.set_fraction(frac)
        # Friendly label: "2,847 / 32,768  (9%)"
        self._ctx_label.set_text(f"{used:,} / {total:,}  ({frac*100:.0f}%)")
        # Tint the bar based on usage via CSS state classes.
        for cls in ("ctx-warn", "ctx-crit"):
            self._ctx_bar.remove_css_class(cls)
        if frac >= 0.90:
            self._ctx_bar.add_css_class("ctx-crit")
        elif frac >= 0.70:
            self._ctx_bar.add_css_class("ctx-warn")

    def set_recording(self, recording: bool) -> None:
        """Update mic button appearance to reflect recording state."""
        self._is_recording = recording
        if self._mic_btn is None:
            return
        if recording:
            self._mic_btn.set_icon_name("media-record-symbolic")
            self._mic_btn.add_css_class("destructive-action")
            self._mic_btn.set_tooltip_text("Stop recording")
        else:
            self._mic_btn.set_icon_name("audio-input-microphone-symbolic")
            self._mic_btn.remove_css_class("destructive-action")
            self._mic_btn.set_tooltip_text("Record voice message")

    # ── Internal ───────────────────────────────────────────────────────────

    def _add_chip(self, att: dict) -> None:
        chip = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=4,
        )
        chip.add_css_class("attachment-chip")

        icon = {
            "image": "image-x-generic-symbolic",
            "audio": "audio-x-generic-symbolic",
        }.get(att["type"], "text-x-generic-symbolic")
        chip.append(Gtk.Image(icon_name=icon))

        label = Gtk.Label(label=att["name"], max_width_chars=20, ellipsize=3)
        chip.append(label)

        rm_btn = Gtk.Button(
            icon_name="window-close-symbolic",
            has_frame=False,
            valign=Gtk.Align.CENTER,
        )
        rm_btn.add_css_class("flat")
        rm_btn.connect(
            "clicked",
            lambda _b, a=att, c=chip: self._remove_attachment(a, c),
        )
        chip.append(rm_btn)
        self._chips_box.append(chip)

    def _remove_attachment(self, att: dict, chip: Gtk.Box) -> None:
        if att in self._attachments:
            self._attachments.remove(att)
        self._chips_box.remove(chip)
        if not self._attachments:
            self._chips_box.set_visible(False)

    def set_generating(self, generating: bool) -> None:
        """Public busy-state toggle for non-streaming long ops (e.g. a file
        audit): shows the Stop button and disables input, no bubble needed.
        Stop routes through the composer's existing on_stop → engine.stop()."""
        self._set_generating(generating)

    def _set_generating(self, generating: bool) -> None:
        self._is_generating = generating
        self._send_btn.set_visible(not generating)
        self._stop_btn.set_visible(generating)
        self._input.set_sensitive(not generating)

    def _send_clicked(self) -> None:
        if self._is_generating:
            return
        buf = self._input.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False).strip()
        attachments = list(self._attachments)
        if not text and not attachments:
            return
        buf.set_text("", -1)
        self._attachments.clear()
        # Clear chips
        child = self._chips_box.get_first_child()
        while child:
            nxt = child.get_next_sibling()
            self._chips_box.remove(child)
            child = nxt
        self._chips_box.set_visible(False)
        self._on_send(text, attachments)

    def _on_input_key(self, _ctrl, keyval: int, _keycode: int, state: Gdk.ModifierType) -> bool:
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            if state & Gdk.ModifierType.SHIFT_MASK:
                return False
            self._send_clicked()
            return True
        return False

    def _build_empty_state(self) -> Gtk.Widget:
        # After the one-time welcome is dismissed, the empty chat is just
        # the quiet app icon — no copy.
        if self._settings is not None and getattr(
            self._settings, "welcome_dismissed", False
        ):
            return Adw.StatusPage(icon_name="com.jegly.box")
        page = Adw.StatusPage(
            icon_name="com.jegly.box",
            title="Start a conversation",
            description="Pick a model from the header dropdown — LiteRT "
            "(.litertlm/.task) or GGUF — then chat with text, images, "
            "audio and tools. Image generation and Box Code live in the "
            "nav rail.",
        )

        def dismiss(*_a) -> None:
            if self._settings is not None:
                self._settings.welcome_dismissed = True
                try:
                    self._settings.save()
                except Exception:  # noqa: BLE001
                    log.exception("could not persist welcome_dismissed")
            # The ✕ only exists while the welcome is on screen, so swap
            # unconditionally. (ScrolledWindow wraps children in a
            # Viewport — never compare get_child() to the page.)
            self._empty_state = self._build_empty_state()
            self._scroller.set_child(self._empty_state)

        close = Gtk.Button(icon_name="window-close-symbolic")
        close.add_css_class("flat")
        close.add_css_class("circular")
        close.set_tooltip_text("Dismiss (won't show again)")
        close.set_halign(Gtk.Align.END)
        close.set_valign(Gtk.Align.START)
        close.set_margin_top(12)
        close.set_margin_end(12)
        close.connect("clicked", dismiss)

        overlay = Gtk.Overlay(child=page)
        overlay.add_overlay(close)
        return overlay

    def _maybe_scroll_to_bottom(self, adj: Gtk.Adjustment) -> None:
        if self._stick_to_bottom:
            adj.set_value(adj.get_upper() - adj.get_page_size())

    def _on_user_scroll(self, adj: Gtk.Adjustment) -> None:
        at_bottom = (adj.get_value() + adj.get_page_size()) >= (adj.get_upper() - 4)
        self._stick_to_bottom = at_bottom

    def _scroll_to_bottom_now(self) -> bool:
        adj = self._scroller.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())
        return False
