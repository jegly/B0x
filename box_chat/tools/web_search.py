"""Web search tool — DuckDuckGo via the ``ddgs`` library.

Free, no API key, no signup. Every returned URL is run through the HTTPS guard
in :mod:`box_chat.net` so non-HTTPS hits never reach the model.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..config import Settings
from ..net import require_https
from . import tool

log = logging.getLogger(__name__)

TOOL_ID = "web_search"
DEFAULT_PERMISSION = "chat"   # "ask" | "chat" | "trust"
RISKY = False                 # read-only on the network

_RESULT_CAP_FALLBACK = 5
_MAX_BODY_CHARS = 280         # snippet length per hit
# NOTE: the wall-clock deadline now lives in the generic tools.with_timeout
# wrapper (Settings.tool_timeout_s), applied to every tool callable at
# registration. web_search no longer runs its own ThreadPoolExecutor — the
# old `with`-block version actually blocked on shutdown(wait=True) after a
# timeout, so it never truly bounded the engine worker. The backend pinning
# below is the real fix for the runaway-cascade hang; the generic timeout is
# the safety net on top.
# Bound the backend set instead of ddgs' default "auto", which cascades
# through every engine (DuckDuckGo → Wikipedia → Startpage → Mojeek →
# Grokipedia → …) when one returns nothing, stacking 4-5 sequential requests
# into a 30-80 s blocking tool call — this is what made the app "not
# responding" when Stop was pressed mid-search. A short, fixed list keeps the
# worst case to a couple of attempts (well under _SEARCH_TIMEOUT_S) while
# still having a fallback, since the lone DuckDuckGo backend is unreliable in
# current ddgs. All are HTTPS engines; require_https() still gates every
# result URL as defense in depth.
_SEARCH_BACKEND = "duckduckgo,brave,mojeek"


def get_callables(settings: Settings) -> list[Callable[..., Any]]:
    """SDK entry point — returns the web_search callable."""
    return [_make_web_search(settings)]


def _make_web_search(settings: Settings):
    """Build a closure that reads ``settings.tool_web_search_results`` live."""

    @tool(tool_id=TOOL_ID, risky=RISKY, default_permission=DEFAULT_PERMISSION)
    def web_search(query: str) -> str:
        """Search the web and return the top results.

        Use this when you need information from the public internet — news,
        documentation, definitions, recent events. Only HTTPS URLs are
        returned.

        Args:
            query: Natural-language search query, e.g.
                "python asyncio cancel task".

        Returns:
            A numbered list of results. Each entry has the page title, URL,
            and a short snippet of body text.
        """
        cap = max(1, int(getattr(settings, "tool_web_search_results", _RESULT_CAP_FALLBACK)))
        return _do_search(query, cap)

    return web_search


def _do_search(query: str, max_results: int) -> str:
    q = (query or "").strip()
    if not q:
        return "Error: empty search query."

    try:
        from ddgs import DDGS
    except ImportError as e:
        return f"Error: web search backend unavailable ({e})."

    try:
        results = DDGS().text(
            q, max_results=max_results, backend=_SEARCH_BACKEND
        )
    except Exception as e:  # noqa: BLE001
        log.exception("DuckDuckGo search failed")
        return f"Error running search: {e}"

    if not results:
        return "No results."

    kept: list[dict[str, Any]] = []
    dropped_non_https = 0
    for r in results:
        href = r.get("href") or r.get("url") or ""
        try:
            require_https(href)
        except ValueError:
            dropped_non_https += 1
            continue
        kept.append(r)

    if not kept:
        return (
            f"No HTTPS results (had {len(results)} hits but all were non-HTTPS)."
        )

    lines: list[str] = []
    for i, r in enumerate(kept, 1):
        href = r.get("href") or r.get("url") or ""
        title = (r.get("title") or "").strip() or href
        body = (r.get("body") or "").strip()
        if len(body) > _MAX_BODY_CHARS:
            body = body[: _MAX_BODY_CHARS - 1].rstrip() + "…"
        lines.append(f"{i}. {title}\n   {href}\n   {body}")

    suffix = ""
    if dropped_non_https:
        suffix = f"\n\n(Dropped {dropped_non_https} non-HTTPS result(s).)"
    return "\n\n".join(lines) + suffix
