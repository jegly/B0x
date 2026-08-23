"""Console-script entry point for the `box` command."""
from __future__ import annotations

import logging
import os
import sys


def main() -> int:
    # --verbose / -v or BOX_DEBUG=1 → DEBUG logging across all box_chat modules.
    verbose = (
        "-v" in sys.argv
        or "--verbose" in sys.argv
        or os.environ.get("BOX_DEBUG", "").lower() in ("1", "true", "yes")
    )
    for flag in ("-v", "--verbose"):
        try:
            sys.argv.remove(flag)
        except ValueError:
            pass

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(levelname)s] %(name)s: %(message)s",
        force=True,   # override any earlier basicConfig (e.g. from imports)
    )

    from .app import BoxApp

    app = BoxApp()
    return app.run(sys.argv)


if __name__ == "__main__":
    raise SystemExit(main())
