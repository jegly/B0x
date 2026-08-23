"""Unit tests for the generic per-tool timeout wrapper (tools.with_timeout).

The wrapper is the safety net that stops a wedged tool call (slow web search,
huge fs_grep, a hung future tool) from blocking the engine worker for tens of
seconds. It MUST also stay transparent to the SDK — preserve the function's
name, docstring, type hints, and the @tool metadata — or schema derivation and
the permission gate break.
"""
from __future__ import annotations

import inspect
import time
import unittest

from box_chat.tools import tool, tool_metadata, with_timeout


class WithTimeoutTests(unittest.TestCase):
    def test_zero_timeout_returns_function_unwrapped(self) -> None:
        def f(x: int) -> int:
            return x + 1

        self.assertIs(with_timeout(f, 0), f)
        self.assertIs(with_timeout(f, None), f)

    def test_fast_call_passes_through_result(self) -> None:
        def f(query: str) -> str:
            return f"ok:{query}"

        wrapped = with_timeout(f, 5)
        self.assertEqual(wrapped("hi"), "ok:hi")

    def test_slow_call_returns_timeout_string(self) -> None:
        def slow() -> str:
            time.sleep(2.0)
            return "should not see this"

        wrapped = with_timeout(slow, 0.2)
        start = time.monotonic()
        result = wrapped()
        elapsed = time.monotonic() - start
        self.assertIn("timed out", result.lower())
        # The wrapper must NOT block until the slow call finishes.
        self.assertLess(elapsed, 1.0)

    def test_preserves_name_and_doc(self) -> None:
        def web_search(query: str) -> str:
            """Search the web."""
            return ""

        wrapped = with_timeout(web_search, 5)
        self.assertEqual(wrapped.__name__, "web_search")
        self.assertEqual(wrapped.__doc__, "Search the web.")

    def test_preserves_signature_via_wrapped(self) -> None:
        def fs_read(path: str) -> str:
            return ""

        wrapped = with_timeout(fs_read, 5)
        sig = inspect.signature(wrapped)
        self.assertEqual(list(sig.parameters), ["path"])
        # `from __future__ import annotations` stringifies annotations here
        # (and in the real tool modules too), so the annotation is "str".
        self.assertEqual(sig.parameters["path"].annotation, "str")

    def test_preserves_tool_metadata(self) -> None:
        @tool(tool_id="filesystem", risky=True, default_permission="ask")
        def fs_write(path: str, content: str) -> str:
            return ""

        wrapped = with_timeout(fs_write, 5)
        meta = tool_metadata(wrapped)
        self.assertEqual(meta.get("tool_id"), "filesystem")
        self.assertTrue(meta.get("risky"))
        self.assertEqual(meta.get("default_permission"), "ask")


if __name__ == "__main__":
    unittest.main()
