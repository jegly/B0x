"""Tool registry for Phase 4 function calling.

Each tool module under ``box_chat.tools`` exposes:

- ``TOOL_ID: str``                – short identifier (matches the registry key).
- ``DEFAULT_PERMISSION: str``     – one of ``"ask" | "chat" | "trust"``.
- ``RISKY: bool``                 – does the tool touch user state (write / exec)?
- ``get_callables(settings) -> list[Callable]`` – the SDK-facing functions to
                                   register; may return ``[]`` if disabled.

Callables themselves should be wrapped with the :func:`tool` decorator below
so the engine layer can read their metadata without re-importing the module.

This package is pure stdlib + portable Python — it must never import ``gi``
(see project-core-must-stay-portable memory).
"""
from __future__ import annotations

import concurrent.futures
import functools
import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..config import Settings

log = logging.getLogger(__name__)


# ── Tool registry ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ToolSpec:
    """Static metadata about a tool module."""
    tool_id: str
    module_name: str         # relative to this package
    enable_attr: str         # Settings field for master on/off
    permission_attr: str     # Settings field for "ask"|"chat"|"trust"
    ack_attr: str            # Settings field for first-enable acknowledgement


_REGISTRY: tuple[ToolSpec, ...] = (
    ToolSpec(
        tool_id="web_search",
        module_name="web_search",
        enable_attr="tool_web_search_enabled",
        permission_attr="tool_web_search_permission",
        ack_attr="tool_web_search_first_enable_acknowledged",
    ),
    ToolSpec(
        tool_id="filesystem",
        module_name="filesystem",
        enable_attr="tool_fs_enabled",
        permission_attr="tool_fs_permission",
        ack_attr="tool_fs_first_enable_acknowledged",
    ),
)

_REGISTRY_BY_ID: dict[str, ToolSpec] = {s.tool_id: s for s in _REGISTRY}


def all_specs() -> tuple[ToolSpec, ...]:
    return _REGISTRY


def spec(tool_id: str) -> ToolSpec | None:
    return _REGISTRY_BY_ID.get(tool_id)


def is_enabled(settings: Settings, tool_id: str) -> bool:
    s = _REGISTRY_BY_ID.get(tool_id)
    return bool(s and getattr(settings, s.enable_attr, False))


def permission_mode(settings: Settings, tool_id: str) -> str:
    s = _REGISTRY_BY_ID.get(tool_id)
    if not s:
        return "ask"
    return getattr(settings, s.permission_attr, "ask")


# ── Decorator stamped on every tool callable ───────────────────────────────
def tool(*, tool_id: str, risky: bool, default_permission: str = "ask"):
    """Stamp metadata onto a tool function.

    The engine layer reads these attributes off the callable so it can
    decide whether to gate the call behind a permission prompt without
    re-importing the tool module.
    """
    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._box_tool_id = tool_id              # type: ignore[attr-defined]
        fn._box_tool_risky = bool(risky)       # type: ignore[attr-defined]
        fn._box_tool_default_permission = default_permission  # type: ignore[attr-defined]
        return fn
    return deco


def with_timeout(
    fn: Callable[..., Any], timeout_s: float | int | None
) -> Callable[..., Any]:
    """Wrap a tool callable so it can't block the engine worker forever.

    Runs ``fn`` on a throwaway worker thread and waits at most ``timeout_s``
    for it. On timeout the model gets a plain "timed out" string (tools
    return strings, so this slots straight into the conversation) while the
    runaway thread is abandoned via ``shutdown(wait=False)`` — we never join
    it, so the engine worker returns immediately.

    ``functools.wraps`` copies ``__name__``/``__doc__``/``__annotations__``
    AND the function's ``__dict__`` (which carries the ``_box_tool_*``
    metadata stamped by :func:`tool`), and ``inspect.signature`` follows
    ``__wrapped__`` — so the SDK still derives the same schema and the
    permission gate still resolves the call. ``timeout_s`` of 0/None returns
    ``fn`` unwrapped.
    """
    if not timeout_s or timeout_s <= 0:
        return fn

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        fut = ex.submit(fn, *args, **kwargs)
        try:
            return fut.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            fut.cancel()
            log.warning(
                "tool %s timed out after %ss",
                getattr(fn, "__name__", "?"), timeout_s,
            )
            return (
                f"Error: tool '{getattr(fn, '__name__', '?')}' timed out "
                f"after {timeout_s:g}s. Try a narrower request."
            )
        finally:
            # Don't wait on the (possibly stuck) worker thread.
            ex.shutdown(wait=False)

    return wrapper


def tool_metadata(fn: Callable[..., Any]) -> dict[str, Any]:
    """Return metadata stamped on a tool callable. Empty dict if not a tool."""
    if not hasattr(fn, "_box_tool_id"):
        return {}
    return {
        "tool_id": fn._box_tool_id,                    # type: ignore[attr-defined]
        "risky": bool(fn._box_tool_risky),             # type: ignore[attr-defined]
        "default_permission": fn._box_tool_default_permission,  # type: ignore[attr-defined]
    }


# ── Aggregation for the engine ─────────────────────────────────────────────
def enabled_tools(settings: Settings) -> list[Callable[..., Any]]:
    """Return the flat list of callables the SDK should expose to the model.

    Walks every registered tool, skips disabled ones, imports each enabled
    module lazily, and concatenates the callables it offers. A tool module
    that fails to import (e.g. missing optional dep like ``ddgs``) is
    skipped with a warning so one broken tool doesn't kill the whole chat.
    """
    out: list[Callable[..., Any]] = []
    timeout_s = getattr(settings, "tool_timeout_s", 0)
    for s in _REGISTRY:
        if not is_enabled(settings, s.tool_id):
            continue
        try:
            mod = importlib.import_module(f".{s.module_name}", __package__)
        except ImportError as e:
            log.warning("Tool %s unavailable: %s", s.tool_id, e)
            continue
        get_callables = getattr(mod, "get_callables", None)
        if get_callables is None:
            log.warning("Tool %s missing get_callables()", s.tool_id)
            continue
        try:
            out.extend(
                with_timeout(fn, timeout_s) for fn in get_callables(settings)
            )
        except Exception:  # noqa: BLE001
            log.exception("Tool %s get_callables() raised", s.tool_id)
    return out


def enabled_tool_ids(settings: Settings) -> list[str]:
    """Tool ids that are currently active (master switch on)."""
    return [s.tool_id for s in _REGISTRY if is_enabled(settings, s.tool_id)]


def callables_for_tool_ids(
    settings: Settings, tool_ids: list[str] | set[str]
) -> list[Callable[..., Any]]:
    """Like :func:`enabled_tools` but driven by an explicit ID set rather
    than the global master switches. Used by per-chat overrides — the
    window resolves global + per-chat tri-state into the effective ID set
    and passes it here.
    """
    wanted = set(tool_ids)
    out: list[Callable[..., Any]] = []
    timeout_s = getattr(settings, "tool_timeout_s", 0)
    for s in _REGISTRY:
        if s.tool_id not in wanted:
            continue
        try:
            mod = importlib.import_module(f".{s.module_name}", __package__)
        except ImportError as e:
            log.warning("Tool %s unavailable: %s", s.tool_id, e)
            continue
        get_callables = getattr(mod, "get_callables", None)
        if get_callables is None:
            continue
        try:
            out.extend(
                with_timeout(fn, timeout_s) for fn in get_callables(settings)
            )
        except Exception:  # noqa: BLE001
            log.exception("Tool %s get_callables() raised", s.tool_id)
    return out


def call_map_for_callables(
    callables: list[Callable[..., Any]],
) -> dict[str, dict[str, Any]]:
    """Like :func:`enabled_tool_call_map` but for an already-resolved list."""
    m: dict[str, dict[str, Any]] = {}
    for fn in callables:
        meta = tool_metadata(fn)
        if meta:
            m[fn.__name__] = meta
    return m


def enabled_tool_call_map(settings: Settings) -> dict[str, dict[str, Any]]:
    """Map function name → tool metadata for every currently-enabled callable.

    The SDK reports tool calls by function name (``"fs_read"``,
    ``"web_search"`` etc.); the permission gate needs that to resolve back
    to a tool_id and the risky flag.
    """
    m: dict[str, dict[str, Any]] = {}
    for fn in enabled_tools(settings):
        meta = tool_metadata(fn)
        if meta:
            m[fn.__name__] = meta
    return m
