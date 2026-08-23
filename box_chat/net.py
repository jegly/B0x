"""Network policy helpers — enforce HTTPS across every download/fetch."""
from __future__ import annotations


def require_https(url: str) -> str:
    """Raise ValueError if url isn't https://. Returns url unchanged on pass."""
    if not url.lower().startswith("https://"):
        raise ValueError(f"Box only allows HTTPS URLs; refused: {url[:80]}")
    return url
