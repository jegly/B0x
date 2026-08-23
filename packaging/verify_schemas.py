#!/usr/bin/env python3
"""Build gate: prove the LiteRT-LM tool schemas survive compilation.

The SDK derives each tool's JSON schema from the callable's type hints and
docstring. Nuitka must not strip ``__doc__`` / ``__annotations__`` or break
``inspect.signature`` (which follows ``functools.wraps``' ``__wrapped__``
through the timeout wrapper). If it does, web_search / fs_* silently lose
their schemas at runtime.

Run AGAINST THE COMPILED build: the caller puts the dir holding the compiled
``box_chat`` extension on PYTHONPATH, then runs this. Exits non-zero on any
regression so the build aborts before packaging.
"""
from __future__ import annotations

import inspect
import sys


def main() -> int:
    import box_chat  # noqa: F401  — verifies the compiled package imports
    from box_chat.config import Settings
    from box_chat.tools import enabled_tools

    s = Settings()
    s.tool_web_search_enabled = True
    s.tool_fs_enabled = True
    s.tool_fs_writable = True  # also exposes fs_write / fs_delete

    tools = enabled_tools(s)
    if not tools:
        print("FAIL: enabled_tools() returned nothing", file=sys.stderr)
        return 1

    ok = True
    for fn in tools:
        name = getattr(fn, "__name__", repr(fn))
        if not getattr(fn, "__doc__", None):
            print(f"FAIL: {name} lost its docstring under compilation",
                  file=sys.stderr)
            ok = False
        try:
            sig = inspect.signature(fn)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL: {name} — inspect.signature raised: {exc}",
                  file=sys.stderr)
            ok = False
            continue
        if not sig.parameters:
            print(f"FAIL: {name} lost its parameter signature", file=sys.stderr)
            ok = False
        else:
            print(f"  ok: {name}{tuple(sig.parameters)}")

    if ok:
        print(f"PASS: {len(tools)} tool schemas intact under compilation")
        return 0
    print("\nGATE FAILED — do not ship this build.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
