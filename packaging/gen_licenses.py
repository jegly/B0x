#!/usr/bin/env python3
"""Generate the THIRD-PARTY-LICENSES attribution file.

Apache-2.0 / MIT / BSD all require preserving copyright + license notices
for redistributed code. Box vendors its pip dependencies into the .deb, so
this walks the bundled site-packages dir and emits a per-package attribution
block (name, version, license, homepage, and the license text when present).

Usage: gen_licenses.py <site-packages-dir>   (writes the document to stdout)
"""
from __future__ import annotations

import sys
from importlib.metadata import distributions


def _license_text(dist) -> str:
    """Return the bundled license file's text, or '' if none is present."""
    for path in dist.files or []:
        base = path.name.lower()
        if base.startswith(("license", "licence", "copying")):
            try:
                return dist.read_text(str(path)) or ""
            except (OSError, UnicodeDecodeError):
                continue
    return ""


def _license_name(dist) -> str:
    lic = (dist.metadata.get("License") or "").strip()
    if lic and len(lic) <= 120 and "\n" not in lic:
        return lic
    classifiers = [
        c for c in (dist.metadata.get_all("Classifier") or [])
        if c.startswith("License ::")
    ]
    if classifiers:
        return "; ".join(c.split("::")[-1].strip() for c in classifiers)
    return "(see license text below / package metadata)"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: gen_licenses.py <site-packages-dir>", file=sys.stderr)
        return 2
    site = sys.argv[1]

    blocks: dict[str, str] = {}
    for dist in distributions(path=[site]):
        name = dist.metadata.get("Name")
        if not name or name.lower() in blocks:
            continue
        lines = [
            f"{name} {dist.version or '?'}",
            f"  License: {_license_name(dist)}",
        ]
        url = dist.metadata.get("Home-page") or ""
        if url:
            lines.append(f"  Homepage: {url}")
        text = _license_text(dist)
        if text:
            lines.append("")
            lines.append("  --- license text ---")
            lines.extend("  " + ln for ln in text.rstrip().splitlines())
        blocks[name.lower()] = "\n".join(lines)

    out = [
        "THIRD-PARTY LICENSES",
        "====================",
        "",
        "Box bundles the third-party Python packages listed below. Each remains",
        "under its own license and copyright, held by its respective authors.",
        "GTK 4, libadwaita, GStreamer and the PyGObject bindings are NOT bundled",
        "— they are provided by the system (apt) and remain under the LGPL.",
        "",
        "=" * 72,
        "",
    ]
    for key in sorted(blocks):
        out.append(blocks[key])
        out.append("")
        out.append("-" * 72)
        out.append("")

    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
