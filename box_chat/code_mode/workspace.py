"""Project-root path scoping for Box Code tools.

Same discipline as ``tools/filesystem.resolve_within`` but stricter in
spirit: code mode has NO outside-the-project grant mechanism at all. If a
path doesn't canonically resolve inside the project folder, the tool
refuses — full stop.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["JUNK_DIRS", "iter_project_files", "rel", "resolve_in_project"]

# Directories skipped by glob/grep walks — build junk and VCS internals the
# model never wants and that would blow the output caps instantly.
JUNK_DIRS: frozenset[str] = frozenset({
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist",
    "build", "target", ".tox", ".idea", ".vscode",
})


def resolve_in_project(root: Path, requested: str) -> Path | None:
    """Resolve ``requested`` against ``root``; None unless it lands inside.

    Handles ``..``, absolute paths and symlink escapes by canonicalizing
    with :meth:`Path.resolve` (``strict=False`` so not-yet-existing write
    targets still validate). An absolute path that happens to live inside
    the project is accepted — models flip between relative and absolute
    constantly and there's no reason to punish the accurate case.
    """
    try:
        root_abs = root.resolve(strict=False)
        target = requested if requested else "."
        candidate = Path(target)
        if not candidate.is_absolute():
            candidate = root_abs / target
        full = candidate.resolve(strict=False)
        if full == root_abs:
            return full
        if str(full).startswith(str(root_abs) + os.sep):
            return full
        return None
    except (OSError, ValueError):
        return None


def rel(root: Path, p: Path) -> str:
    """Format a resolved path project-relative for display."""
    try:
        return str(p.relative_to(root.resolve(strict=False))) or "."
    except ValueError:
        return str(p)


def iter_project_files(start: Path):
    """Yield files under ``start`` depth-first, pruning JUNK_DIRS.

    ``start`` must already be a validated in-project path. Symlinked
    directories are not followed (escape + cycle safety).
    """
    if start.is_file():
        yield start
        return
    stack = [start]
    while stack:
        d = stack.pop()
        try:
            entries = sorted(d.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        for child in entries:
            try:
                if child.is_symlink():
                    if child.is_file():
                        yield child
                    continue
                if child.is_dir():
                    if child.name not in JUNK_DIRS:
                        stack.append(child)
                elif child.is_file():
                    yield child
            except OSError:
                continue
