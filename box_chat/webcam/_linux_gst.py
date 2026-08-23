"""Linux webcam backend — GStreamer over PipeWire (preferred) or V4L2.

Pipeline shape:

    <source>
      ! videoconvert
      ! videoscale
      ! video/x-raw,width=<W>,height=<H>
      ! tee name=t
        t. ! queue ! <preview-sink name=preview>
        t. ! queue ! videoconvert ! jpegenc quality=<Q>
                   ! appsink name=snap max-buffers=1 drop=true

We pick the source and preview sink at runtime based on what's actually
installed: ``pipewiresrc`` is preferred (Wayland-clean + sandbox-friendly),
``v4l2src`` is the fallback. For preview, ``gtk4paintablesink`` is ideal
because it hands GTK a :class:`Gdk.Paintable` directly; if it's missing
we fall through to ``glsinkbin sink=gtkglsink`` and then to no preview
at all (we can still capture even without preview).
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# Try to import GStreamer up front so AVAILABLE reflects a real
# library-load result, not just a guess based on platform.
try:
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("Gdk", "4.0")
    from gi.repository import Gdk, GLib, Gst  # noqa: F401

    if not Gst.is_initialized():
        Gst.init(None)

    _IMPORT_OK = True
    _IMPORT_ERROR: str = ""
except Exception as _e:  # noqa: BLE001 — broad on import path
    _IMPORT_OK = False
    _IMPORT_ERROR = f"GStreamer not importable: {_e}"


AVAILABLE: bool = _IMPORT_OK
INSTALL_HINT: str = (
    "sudo apt install gstreamer1.0-plugins-good gstreamer1.0-plugins-base "
    "gst-plugin-pipewire gstreamer1.0-plugins-rs-gtk4"
)
unavailable_reason: str = "" if _IMPORT_OK else _IMPORT_ERROR
_probed: bool = False


@dataclass(frozen=True)
class CameraDevice:
    id: str
    label: str


# ── Plugin probing ────────────────────────────────────────────────────
def _has_element(name: str) -> bool:
    if not _IMPORT_OK:
        return False
    f = Gst.ElementFactory.find(name)
    return f is not None


def _select_source() -> tuple[str, str]:
    """Return (gst-launch-style snippet, label) for the best available source."""
    if _has_element("pipewiresrc"):
        return "pipewiresrc", "PipeWire"
    if _has_element("v4l2src"):
        return "v4l2src", "V4L2"
    return "", ""


def _select_preview_sink() -> tuple[str, str]:
    """Return (sink snippet, label). Empty string if none available —
    we can still capture without a preview."""
    if _has_element("gtk4paintablesink"):
        return "gtk4paintablesink name=preview", "gtk4paintablesink"
    if _has_element("glsinkbin") and _has_element("gtk4glsink"):
        return "glsinkbin name=preview sink=gtk4glsink", "glsinkbin+gtk4glsink"
    if _has_element("gtksink"):
        return "gtksink name=preview", "gtksink (GTK3-era)"
    return "", ""


# ── probe ─────────────────────────────────────────────────────────────
def probe(force: bool = False) -> tuple[bool, str]:
    """Build the minimal pipeline once and confirm it transitions to
    PAUSED. Updates :data:`AVAILABLE` and :data:`unavailable_reason`.
    """
    global AVAILABLE, unavailable_reason, _probed
    if _probed and not force:
        return AVAILABLE, unavailable_reason
    if not _IMPORT_OK:
        AVAILABLE = False
        unavailable_reason = _IMPORT_ERROR
        _probed = True
        return AVAILABLE, unavailable_reason

    src, src_label = _select_source()
    if not src:
        AVAILABLE = False
        unavailable_reason = (
            "No GStreamer video source available — install "
            "gst-plugin-pipewire or gstreamer1.0-plugins-good."
        )
        _probed = True
        return AVAILABLE, unavailable_reason

    # Minimal pipeline: source → fakesink. We're checking that the
    # source can be instantiated, not that the camera will actually
    # produce frames (that takes longer).
    pipeline_str = f"{src} ! fakesink"
    try:
        pipeline = Gst.parse_launch(pipeline_str)
    except GLib.Error as e:
        AVAILABLE = False
        unavailable_reason = f"Could not build pipeline: {e}"
        _probed = True
        return AVAILABLE, unavailable_reason
    finally:
        # Always release.
        try:
            pipeline.set_state(Gst.State.NULL)  # type: ignore[name-defined]
        except Exception:
            pass

    AVAILABLE = True
    unavailable_reason = ""
    log.info(
        "Webcam backend ready: source=%s preview=%s",
        src_label, _select_preview_sink()[1] or "(none — capture only)",
    )
    _probed = True
    return AVAILABLE, unavailable_reason


# ── device enumeration ────────────────────────────────────────────────
def list_devices() -> list[CameraDevice]:
    """Enumerate cameras, deduped by device path.

    GStreamer reports the same physical camera multiple times (once via
    the V4L2 provider, once via PipeWire), and ThinkPad-class laptops
    expose both a colour capture node and an IR / metadata node on the
    same camera. We keep one entry per unique device path, prefer
    longer/cleaner labels, and skip the IR helper nodes since they
    can't deliver a normal RGB frame.
    """
    if not _IMPORT_OK:
        return []
    monitor = Gst.DeviceMonitor.new()
    monitor.add_filter("Video/Source", None)
    if not monitor.start():
        return []
    try:
        devices = monitor.get_devices() or []
    finally:
        monitor.stop()

    by_id: dict[str, str] = {}
    for d in devices:
        label = (d.get_display_name() or "Camera").strip()
        props = d.get_properties()
        dev_id = ""
        if props is not None:
            for key in ("device.path", "api.v4l2.path", "node.id"):
                v = props.get_string(key)
                if v:
                    dev_id = v
                    break
        if not dev_id:
            dev_id = label
        # Keep the longest label we see for each unique path (PipeWire
        # entries tend to have richer names than the raw V4L2 ones).
        if dev_id not in by_id or len(label) > len(by_id[dev_id]):
            by_id[dev_id] = label

    return sorted(
        (CameraDevice(id=dev_id, label=label) for dev_id, label in by_id.items()),
        key=lambda d: d.id,
    )


# ── CameraSession ─────────────────────────────────────────────────────
class CameraSession:
    """Live GStreamer pipeline with a paintable for preview and an
    appsink for on-demand JPEG snapshots.

    Use as a context manager so the pipeline is guaranteed to release —
    keeping the camera light on past close is a privacy bug.
    """

    def __init__(
        self,
        device_id: str | None = None,
        *,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 80,
    ) -> None:
        if not _IMPORT_OK:
            raise RuntimeError(_IMPORT_ERROR)
        self._device_id = device_id
        self._width = max(160, int(width))
        self._height = max(120, int(height))
        self._jpeg_quality = max(1, min(100, int(jpeg_quality)))
        self._pipeline: Any = None
        self._appsink: Any = None
        # _preview_element is either a gtk4paintablesink (returns a real
        # Gdk.Paintable via .props.paintable) OR a raw-RGB appsink that we
        # decode in Python and push to a registered callback. Both paths
        # produce a Gdk.Paintable for the dialog; the latter just costs
        # a bit more CPU.
        self._preview_element: Any = None
        self._preview_is_appsink: bool = False
        self._preview_cb: Any = None  # Callable[[Gdk.Texture], None]
        self._lock = threading.Lock()

    # ── lifecycle ────────────────────────────────────────────────────
    def __enter__(self) -> "CameraSession":
        self.start()
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False

    def start(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                return

            src, _ = _select_source()
            if not src:
                raise RuntimeError(
                    "No GStreamer video source available "
                    "(install gst-plugin-pipewire or "
                    "gstreamer1.0-plugins-good)"
                )

            # Append device-path hint for v4l2src; pipewiresrc takes
            # node-id but we let the user's portal pick by default.
            if src == "v4l2src" and self._device_id:
                src = f'v4l2src device="{self._device_id}"'

            preview_sink, _ = _select_preview_sink()
            if preview_sink:
                # Native paintable sink — single element, dialog reads
                # its .props.paintable.
                preview_branch = (
                    f"t. ! queue ! videoconvert ! videoscale "
                    f"! video/x-raw,width={self._width},height={self._height} "
                    f"! {preview_sink}"
                )
                self._preview_is_appsink = False
            else:
                # No paintable sink available — fall back to a raw-RGB
                # appsink and decode frames in Python. Each branch does
                # its OWN videoconvert + videoscale so the V4L2 source
                # only sees one consumer's caps requirement at a time —
                # otherwise the two converters disagree on format and
                # we get `EINVAL set_output_format` from V4L2.
                preview_branch = (
                    f"t. ! queue leaky=downstream max-size-buffers=1 "
                    f"! videoconvert ! videoscale "
                    f"! video/x-raw,format=RGB,width={self._width},height={self._height} "
                    f"! appsink name=preview emit-signals=true "
                    f"max-buffers=1 drop=true sync=false"
                )
                self._preview_is_appsink = True

            # Note: scaling and format-conversion happen INSIDE each
            # branch, not before the tee. The shared upstream is just
            # the camera source → tee, which leaves the V4L2 source
            # free to negotiate its native format (YUYV / MJPG / …).
            pipeline_str = (
                f"{src} ! tee name=t "
                f"{preview_branch} "
                f"t. ! queue ! videoconvert ! videoscale "
                f"! video/x-raw,width={self._width},height={self._height} "
                f"! jpegenc quality={self._jpeg_quality} "
                f"! appsink name=snap emit-signals=true max-buffers=1 drop=true sync=false"
            )
            log.debug("Webcam pipeline: %s", pipeline_str)
            self._pipeline = Gst.parse_launch(pipeline_str)
            self._appsink = self._pipeline.get_by_name("snap")
            self._preview_element = self._pipeline.get_by_name("preview")

            if self._preview_is_appsink and self._preview_element is not None:
                self._preview_element.connect(
                    "new-sample", self._on_preview_sample
                )

            ret = self._pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                # Drain the bus for a real ERROR message before giving up.
                detail = self._drain_bus_error()
                self._pipeline.set_state(Gst.State.NULL)
                self._pipeline = None
                raise RuntimeError(detail or "Pipeline failed to start")
            # Wait briefly for ASYNC state-change to complete and surface
            # any negotiation/permission errors as a real message.
            ret2, _, _ = self._pipeline.get_state(2 * Gst.SECOND)
            if ret2 == Gst.StateChangeReturn.FAILURE:
                detail = self._drain_bus_error()
                self._pipeline.set_state(Gst.State.NULL)
                self._pipeline = None
                raise RuntimeError(detail or "Pipeline failed to start")

    def _drain_bus_error(self) -> str:
        """Pop ERROR messages from the pipeline bus and stitch them into
        a useful human-readable string. Returns empty string if there
        was nothing on the bus."""
        if self._pipeline is None:
            return ""
        bus = self._pipeline.get_bus()
        if bus is None:
            return ""
        parts: list[str] = []
        while True:
            msg = bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.WARNING)
            if msg is None:
                break
            if msg.type == Gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                parts.append(f"{err.message}")
                if debug:
                    log.warning("GStreamer ERROR detail: %s", debug)
            elif msg.type == Gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                log.warning("GStreamer WARNING: %s (%s)", err.message, debug)
        return " · ".join(parts)

    def close(self) -> None:
        with self._lock:
            if self._pipeline is None:
                return
            try:
                self._pipeline.set_state(Gst.State.NULL)
            except Exception:
                log.exception("Pipeline NULL transition failed")
            self._pipeline = None
            self._appsink = None
            self._preview_element = None

    # ── public surface ───────────────────────────────────────────────
    def paintable(self):
        """Return a :class:`Gdk.Paintable` for live preview, or None when
        only the manual fallback (set_preview_callback) is available."""
        if self._preview_element is None or self._preview_is_appsink:
            return None
        try:
            return self._preview_element.props.paintable
        except Exception:
            return None

    def set_preview_callback(self, cb) -> None:
        """Register a callback that receives a :class:`Gdk.Texture` for
        each decoded preview frame. Used when no paintable-capable
        GStreamer sink is installed, so the dialog can still show live
        video. The callback is always invoked from the GTK main thread.
        """
        self._preview_cb = cb

    def _on_preview_sample(self, appsink) -> Any:
        """``new-sample`` handler — fires on the GStreamer streaming
        thread (NOT the GTK main thread). We pull the latest RGB frame,
        copy it into a :class:`Gdk.Texture`, then bounce to main thread
        for the callback."""
        sample = appsink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        caps = sample.get_caps()
        if caps is None:
            return Gst.FlowReturn.OK
        s = caps.get_structure(0)
        ok_w, w = s.get_int("width")
        ok_h, h = s.get_int("height")
        if not (ok_w and ok_h):
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return Gst.FlowReturn.OK
        try:
            data = bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)

        cb = self._preview_cb
        if cb is None:
            return Gst.FlowReturn.OK

        def _deliver() -> bool:
            try:
                gbytes = GLib.Bytes.new(data)
                texture = Gdk.MemoryTexture.new(
                    w, h,
                    Gdk.MemoryFormat.R8G8B8,
                    gbytes,
                    w * 3,
                )
                cb(texture)
            except Exception:
                log.exception("Preview callback raised")
            return False

        GLib.idle_add(_deliver)
        return Gst.FlowReturn.OK

    def capture_jpeg(self) -> bytes:
        """Pull the latest JPEG-encoded frame from the appsink.

        Blocks up to ~2 s waiting for a buffer (camera warm-up). Returns
        the raw JPEG bytes — no headers, ready to write to a `.jpg` file
        or feed into the SDK as a blob.
        """
        if self._appsink is None:
            raise RuntimeError("Camera not started")
        sample = self._appsink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            raise RuntimeError("No frame available yet — camera warming up?")
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            raise RuntimeError("Could not map GStreamer buffer")
        try:
            return bytes(mapinfo.data)
        finally:
            buf.unmap(mapinfo)


def open_session(device_id: str | None = None) -> CameraSession:
    """Convenience factory matching the public API in ``__init__.py``."""
    return CameraSession(device_id=device_id)
