"""Filesystem tool — scoped to a workspace folder, read-only by default.

Every path the model passes goes through :func:`resolve_within` first, which:

- normalises ``..`` (canonical path),
- follows symlinks (so links escaping the root are caught),
- rejects anything that doesn't resolve inside the workspace.

Writes (``fs_write`` / ``fs_delete``) are exposed only when
``settings.tool_fs_writable`` is True and are stamped ``RISKY=True`` so the
permission layer refuses "trust always" for them.
"""
from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..config import Settings
from . import tool

log = logging.getLogger(__name__)

TOOL_ID = "filesystem"
DEFAULT_PERMISSION = "ask"
RISKY = False  # read-only callables; write pair stamps RISKY=True per-fn

_READ_CAP_BYTES = 200 * 1024     # cap fs_read at 200 KB
_WRITE_CAP_BYTES = 1 * 1024 * 1024  # cap fs_write at 1 MB
_LIST_CAP = 500                  # entries returned by fs_list
_GREP_MAX_MATCHES = 200          # global cap across the walk
_GREP_MAX_FILE_BYTES = 2 * 1024 * 1024  # skip files larger than this when grepping


# ── Path validation ───────────────────────────────────────────────────────
def resolve_within(root: Path, requested: str) -> Path | None:
    """Resolve ``requested`` against ``root`` and return the canonical path
    only if it lands inside (or is) ``root``. Returns None otherwise.

    Handles ``..``, absolute paths, and symlink escapes by always calling
    :meth:`Path.resolve` (with ``strict=False`` so non-existent destinations
    used by ``fs_write`` still validate).
    """
    try:
        root_abs = root.resolve(strict=False)
        target = requested if requested else "."
        full = (root_abs / target).resolve(strict=False)
        if full == root_abs:
            return full
        # str(parent) + os.sep prefix check stops "/foo/barbaz" passing
        # for root "/foo/bar".
        if str(full).startswith(str(root_abs) + os.sep):
            return full
        return None
    except (OSError, ValueError):
        return None


def _root(settings: Settings) -> Path:
    raw = settings.tool_fs_root or "~/Documents/box-workspace"
    p = Path(raw).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.warning("Could not create workspace %s: %s", p, e)
    return p


def _rel(root: Path, p: Path) -> str:
    """Format a resolved path as workspace-relative for display."""
    try:
        return str(p.relative_to(root)) or "."
    except ValueError:
        return str(p)


# ── On-the-fly access grants (paths OUTSIDE the workspace) ─────────────────
# When ``settings.tool_fs_allow_outside`` is on, the model can request paths
# outside the workspace; the permission gate prompts for each one and records
# the grant here so the tool can then resolve it. Three scopes:
#
# - **turn**  ("Allow once")        — in-memory; cleared at the start of every
#                                      user send, so it only covers the current
#                                      request/agent-loop, then re-prompts.
# - **chat**  ("Allow for this chat")— in-memory, keyed by conversation id; only
#                                      applies while that chat is the active one.
# - **persist** ("Always allow")     — saved to ``settings.tool_fs_extra_roots``;
#                                      global, survives restart.
#
# A grant on a directory also covers files beneath it. This is process-global
# state synced to the gate's active conversation — there is one app instance.
_turn_grants: set[str] = set()
_chat_grants: dict[int | None, set[str]] = {}
_active_conv_id: int | None = None
_grants_lock = threading.Lock()


def set_active_conversation(conv_id: int | None) -> None:
    """Track which chat is active so per-chat grants resolve correctly. The
    permission gate forwards its own active-conversation changes here."""
    global _active_conv_id
    with _grants_lock:
        _active_conv_id = conv_id


def grant_path_turn(abs_path: str) -> None:
    """Grant access for the current user turn only (cleared on next send)."""
    with _grants_lock:
        _turn_grants.add(str(Path(abs_path)))


def grant_path_chat(conv_id: int | None, abs_path: str) -> None:
    """Grant access for the duration of one chat (in-memory, this session)."""
    with _grants_lock:
        _chat_grants.setdefault(conv_id, set()).add(str(Path(abs_path)))


def grant_path_persist(settings: Settings, abs_path: str) -> None:
    """Grant access to ``abs_path`` permanently (persisted to settings)."""
    norm = str(Path(abs_path))
    roots = list(getattr(settings, "tool_fs_extra_roots", []) or [])
    if norm not in roots:
        roots.append(norm)
        settings.tool_fs_extra_roots = roots
        try:
            settings.save()
        except Exception:
            log.exception("Could not persist tool_fs_extra_roots")


def forget_persisted_path(settings: Settings, abs_path: str) -> None:
    """Remove one persisted 'Always allow' grant (Preferences management)."""
    norm = str(Path(abs_path))
    roots = [
        r for r in (getattr(settings, "tool_fs_extra_roots", []) or [])
        if r != norm
    ]
    settings.tool_fs_extra_roots = roots
    try:
        settings.save()
    except Exception:
        log.exception("Could not persist tool_fs_extra_roots")


def clear_turn_grants() -> None:
    """Drop "Allow once" grants — called at the start of each user send."""
    with _grants_lock:
        _turn_grants.clear()


def clear_chat_grants(conv_id: int | None) -> None:
    """Drop per-chat grants for a conversation (e.g. when it's deleted)."""
    with _grants_lock:
        _chat_grants.pop(conv_id, None)


def clear_ephemeral_grants() -> None:
    """Drop all in-memory grants (turn + every chat). Persisted grants stay.
    Used when the feature is toggled off, and by tests."""
    with _grants_lock:
        _turn_grants.clear()
        _chat_grants.clear()


def _granted_roots(settings: Settings) -> list[Path]:
    roots: list[Path] = []
    for r in getattr(settings, "tool_fs_extra_roots", []) or []:
        try:
            roots.append(Path(r).expanduser().resolve(strict=False))
        except (OSError, ValueError):
            continue
    with _grants_lock:
        cid = _active_conv_id
        ephemeral = set(_turn_grants) | set(_chat_grants.get(cid, set()))
    for r in ephemeral:
        try:
            roots.append(Path(r).resolve(strict=False))
        except (OSError, ValueError):
            continue
    return roots


def is_path_granted(settings: Settings, full: Path) -> bool:
    """True if ``full`` (a canonical absolute path) is inside a granted path."""
    try:
        full = full.resolve(strict=False)
    except (OSError, ValueError):
        return False
    for root in _granted_roots(settings):
        if full == root or str(full).startswith(str(root) + os.sep):
            return True
    return False


def classify_request(settings: Settings, requested: str) -> tuple[str, Path | None]:
    """Classify a requested path for the permission gate.

    Returns ``(kind, canonical_path)`` where kind is:
    - ``"inside"``  — resolves inside the workspace (canonical path returned),
    - ``"outside"`` — a valid path outside the workspace (canonical returned),
    - ``"invalid"`` — couldn't be resolved at all (None returned).
    """
    root = _root(settings)
    inside = resolve_within(root, requested)
    if inside is not None:
        return ("inside", inside)
    try:
        full = Path(requested).expanduser()
        if not full.is_absolute():
            full = root / requested
        full = full.resolve(strict=False)
    except (OSError, ValueError):
        return ("invalid", None)
    return ("outside", full)


def resolve_access(settings: Settings, requested: str) -> Path | None:
    """Resolve a path the tool may touch: inside the workspace, OR inside a
    granted out-of-workspace path (only when ``tool_fs_allow_outside`` is on).
    Returns the canonical path, or None if access isn't allowed."""
    kind, full = classify_request(settings, requested)
    if kind == "inside":
        return full
    if kind == "invalid" or full is None:
        return None
    if not getattr(settings, "tool_fs_allow_outside", False):
        return None
    return full if is_path_granted(settings, full) else None


def _resolve_or_error(
    settings: Settings, root: Path, path: str
) -> tuple[Path | None, str | None]:
    """Resolve ``path`` for a tool callable, or return a model-readable error."""
    full = resolve_access(settings, path)
    if full is not None:
        return full, None
    if getattr(settings, "tool_fs_allow_outside", False):
        return None, (
            f"Error: {path!r} is outside the workspace and access wasn't "
            "granted. Ask again and approve the access prompt."
        )
    return None, (
        f"Error: {path!r} is outside the workspace. Turn on 'Allow access "
        "outside the workspace' in Preferences → Tools → Filesystem to let "
        "the model request paths like this."
    )


# ── SDK entry point ───────────────────────────────────────────────────────
def get_callables(settings: Settings) -> list[Callable[..., Any]]:
    fns: list[Callable[..., Any]] = [
        _make_fs_read(settings),
        _make_fs_list(settings),
        _make_fs_grep(settings),
    ]
    if getattr(settings, "tool_fs_writable", False):
        fns.append(_make_fs_write(settings))
        fns.append(_make_fs_delete(settings))
    return fns


# ── Read-only callables ───────────────────────────────────────────────────
def _make_fs_read(settings: Settings):
    @tool(tool_id=TOOL_ID, risky=False, default_permission=DEFAULT_PERMISSION)
    def fs_read(path: str) -> str:
        """Read a text file from the workspace and return its contents.

        Args:
            path: Path relative to the workspace root, e.g. ``"notes.md"``
                or ``"src/main.py"``.

        Returns:
            File contents as UTF-8 text (replacement for invalid bytes).
            Files larger than 200 KB are refused — ask the user to split
            them.
        """
        root = _root(settings)
        full, _err = _resolve_or_error(settings, root, path)
        if _err:
            return _err
        if not full.exists():
            return f"Error: no such file: {_rel(root, full)}"
        if full.is_dir():
            return f"Error: {_rel(root, full)} is a directory; use fs_list."
        try:
            size = full.stat().st_size
        except OSError as e:
            return f"Error: cannot stat {_rel(root, full)}: {e}"
        if size > _READ_CAP_BYTES:
            return (
                f"Error: {_rel(root, full)} is {size} bytes; "
                f"fs_read is capped at {_READ_CAP_BYTES} bytes."
            )
        try:
            return full.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error reading {_rel(root, full)}: {e}"

    return fs_read


def _make_fs_list(settings: Settings):
    @tool(tool_id=TOOL_ID, risky=False, default_permission=DEFAULT_PERMISSION)
    def fs_list(path: str = ".") -> str:
        """List entries inside a workspace directory.

        Args:
            path: Directory path relative to the workspace root. Defaults to
                the workspace root itself.

        Returns:
            One entry per line. Directories get a trailing ``/`` so they're
            visually distinct from files, but the name itself is the literal
            path you'd pass back to fs_read or fs_list.
        """
        root = _root(settings)
        full, _err = _resolve_or_error(settings, root, path)
        if _err:
            return _err
        if not full.exists():
            return f"Error: no such directory: {_rel(root, full)}"
        if not full.is_dir():
            return f"Error: {_rel(root, full)} is not a directory."
        try:
            entries = sorted(full.iterdir(), key=lambda p: p.name.lower())
        except OSError as e:
            return f"Error listing {_rel(root, full)}: {e}"
        lines: list[str] = []
        for child in entries[:_LIST_CAP]:
            suffix = "/" if child.is_dir() else ""
            lines.append(f"{child.name}{suffix}")
        if len(entries) > _LIST_CAP:
            lines.append(f"… ({len(entries) - _LIST_CAP} more entries hidden)")
        if not lines:
            return f"(empty: {_rel(root, full)})"
        return "\n".join(lines)

    return fs_list


def _make_fs_grep(settings: Settings):
    @tool(tool_id=TOOL_ID, risky=False, default_permission=DEFAULT_PERMISSION)
    def fs_grep(pattern: str, path: str = ".") -> str:
        """Search files in the workspace for a regular-expression pattern.

        Args:
            pattern: Python ``re``-syntax regex. Matched per line.
            path: Start directory (relative to workspace), or a single file.
                Defaults to the workspace root.

        Returns:
            Lines of the form ``relative/file:lineno: matching line``.
            Capped at 200 matches in total. Binary-looking files are
            skipped.
        """
        root = _root(settings)
        full, _err = _resolve_or_error(settings, root, path)
        if _err:
            return _err
        if not full.exists():
            return f"Error: no such path: {_rel(root, full)}"
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return f"Error: invalid regex: {e}"

        targets: list[Path] = [full] if full.is_file() else sorted(
            p for p in full.rglob("*") if p.is_file()
        )

        hits: list[str] = []
        for f in targets:
            try:
                size = f.stat().st_size
            except OSError:
                continue
            if size > _GREP_MAX_FILE_BYTES:
                continue
            try:
                with f.open("r", encoding="utf-8", errors="strict") as fp:
                    for lineno, line in enumerate(fp, 1):
                        if rx.search(line):
                            hits.append(
                                f"{_rel(root, f)}:{lineno}: {line.rstrip()}"
                            )
                            if len(hits) >= _GREP_MAX_MATCHES:
                                hits.append(
                                    f"… (more than {_GREP_MAX_MATCHES} matches, "
                                    "stopped here)"
                                )
                                return "\n".join(hits)
            except (UnicodeDecodeError, OSError):
                # Binary file or unreadable — skip silently.
                continue
        return "\n".join(hits) if hits else "No matches."

    return fs_grep


# ── Write callables (opt-in via tool_fs_writable) ─────────────────────────
def _make_fs_write(settings: Settings):
    @tool(tool_id=TOOL_ID, risky=True, default_permission="ask")
    def fs_write(path: str, content: str) -> str:
        """Write text to a file inside the workspace, creating or replacing it.

        Args:
            path: Path relative to the workspace root.
            content: Full file contents to write (UTF-8). Replaces any
                existing file.

        Returns:
            Confirmation string with the resolved path and byte count.
        """
        root = _root(settings)
        full, _err = _resolve_or_error(settings, root, path)
        if _err:
            return _err
        if full.is_dir():
            return f"Error: {_rel(root, full)} is a directory."
        data = content.encode("utf-8")
        if len(data) > _WRITE_CAP_BYTES:
            return (
                f"Error: refusing to write {len(data)} bytes; "
                f"fs_write is capped at {_WRITE_CAP_BYTES} bytes."
            )
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            tmp = full.with_suffix(full.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(full)
        except OSError as e:
            return f"Error writing {_rel(root, full)}: {e}"
        return f"Wrote {len(data)} bytes to {_rel(root, full)}."

    return fs_write


def _make_fs_delete(settings: Settings):
    @tool(tool_id=TOOL_ID, risky=True, default_permission="ask")
    def fs_delete(path: str) -> str:
        """Delete a file inside the workspace. Directories are NOT supported.

        Args:
            path: Path relative to the workspace root.

        Returns:
            Confirmation or error string.
        """
        root = _root(settings)
        full, _err = _resolve_or_error(settings, root, path)
        if _err:
            return _err
        if not full.exists():
            return f"Error: no such file: {_rel(root, full)}"
        if full.is_dir():
            return f"Error: {_rel(root, full)} is a directory; "\
                "fs_delete only removes files."
        try:
            full.unlink()
        except OSError as e:
            return f"Error deleting {_rel(root, full)}: {e}"
        return f"Deleted {_rel(root, full)}."

    return fs_delete
