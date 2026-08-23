"""Unit tests for the HTTPS guard in box_chat.net.

The guard is the boundary between the rest of the app and the network — every
download path must call it, so the contract has to hold for a wide range of
junk inputs.
"""
from __future__ import annotations

import unittest

from box_chat.net import require_https


class RequireHttpsTests(unittest.TestCase):
    def test_passes_https_url_unchanged(self) -> None:
        url = "https://huggingface.co/jegly/mirror/resolve/main/x.bin"
        self.assertEqual(require_https(url), url)

    def test_passes_https_with_uppercase_scheme(self) -> None:
        # urllib accepts mixed-case schemes; the guard normalises via .lower()
        url = "HTTPS://example.com/path"
        self.assertEqual(require_https(url), url)

    def test_rejects_plain_http(self) -> None:
        for url in (
            "http://example.com",
            "http://localhost:8080/x",
            "http://127.0.0.1/x",
            "HTTP://example.com",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    require_https(url)

    def test_rejects_other_schemes(self) -> None:
        for url in (
            "ftp://example.com/file",
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "ws://example.com/socket",
            "ssh://git@example.com/repo",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    require_https(url)

    def test_rejects_schemeless_or_empty(self) -> None:
        for url in ("", "  ", "example.com/foo", "//example.com/x", "/relative"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    require_https(url)

    def test_rejects_leading_whitespace(self) -> None:
        # "  https://..." would be treated as http(s) by some parsers but
        # startswith('https://') correctly rejects it. Belt-and-braces check.
        with self.assertRaises(ValueError):
            require_https("  https://example.com")


if __name__ == "__main__":
    unittest.main()
