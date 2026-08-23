"""Backend-neutral map-reduce orchestration (run_map_reduce_audit).

Covers box_chat/audit.py's orchestrator with a fake audit_pass — the four
behaviours that matter: a clean file (all sections NONE), a file with
findings (map → reduce), a cancelled run that must NOT be reported as clean,
and on-the-fly bisection when a section overflows the context window.
"""
from __future__ import annotations

import unittest

from box_chat import audit


class _Overflow(Exception):
    pass


def _collect_progress():
    events: list = []
    return events, (lambda i, total, phase: events.append((i, total, phase)))


class CleanFileTests(unittest.TestCase):
    def test_all_none_reports_clean(self) -> None:
        events, on_progress = _collect_progress()
        report, err = audit.run_map_reduce_audit(
            data="routine line 1\nroutine line 2\nroutine line 3\n",
            focus="security",
            file_label="test.log",
            chunk_chars=1000,
            max_chunks=40,
            truncated_bytes=False,
            audit_pass=lambda sysp, user, m: "NONE",
            is_token_overflow=lambda e: False,
            is_cancelled=lambda: False,
            on_progress=on_progress,
        )
        self.assertIsNone(err)
        self.assertIn(audit.clean_report("security"), report)
        self.assertIn("**Audit of test.log**", report)
        self.assertTrue(events)  # progress was emitted


class FindingsTests(unittest.TestCase):
    def test_findings_flow_to_reduce(self) -> None:
        calls = {"map": 0, "reduce": 0}

        def audit_pass(sysp: str, user: str, m: int) -> str:
            if user.startswith("Findings from"):
                calls["reduce"] += 1
                return "FINAL REPORT: two issues found."
            calls["map"] += 1
            return "FAIL: something bad" if "ERROR" in user else "NONE"

        events, on_progress = _collect_progress()
        data = "ok\n" * 5 + "ERROR here\n" + "ok\n" * 5
        report, err = audit.run_map_reduce_audit(
            data=data, focus="errors", file_label="app.log",
            chunk_chars=1000, max_chunks=40, truncated_bytes=False,
            audit_pass=audit_pass,
            is_token_overflow=lambda e: False,
            is_cancelled=lambda: False,
            on_progress=on_progress,
        )
        self.assertIsNone(err)
        self.assertIn("FINAL REPORT", report)
        self.assertGreaterEqual(calls["map"], 1)
        self.assertEqual(calls["reduce"], 1)
        self.assertIn("report", [e[2] for e in events])  # reduce phase reported


class CancelledTests(unittest.TestCase):
    def test_cancel_never_reports_clean(self) -> None:
        _events, on_progress = _collect_progress()
        report, err = audit.run_map_reduce_audit(
            data="a\nb\nc\nd\n",
            focus="security", file_label="x.log",
            chunk_chars=1000, max_chunks=40, truncated_bytes=False,
            audit_pass=lambda s, u, m: "NONE",
            is_token_overflow=lambda e: False,
            is_cancelled=lambda: True,   # cancelled before any section
            on_progress=on_progress,
        )
        self.assertIsNone(err)
        # Critical: a stopped run with no findings is NOT "clean".
        self.assertNotIn(audit.clean_report("security"), report)
        self.assertIn("NOT a", report)
        self.assertIn("Stopped", report)


class OverflowBisectTests(unittest.TestCase):
    def test_overflowing_section_is_bisected(self) -> None:
        tokens = ("L1", "L2", "L3", "L4")

        def audit_pass(sysp: str, user: str, m: int) -> str:
            if user.startswith("Findings from"):
                return "REDUCED"
            present = [t for t in tokens if t in user]
            if len(present) > 2:          # whole section overflows the window
                raise _Overflow()
            return "found " + ",".join(present)

        # Four fat lines that land in a single chunk, then must be split.
        data = "".join(f"{t} " + "x" * 250 + "\n" for t in tokens)
        report, err = audit.run_map_reduce_audit(
            data=data, focus="summary", file_label="big.log",
            chunk_chars=4000, max_chunks=40, truncated_bytes=False,
            audit_pass=audit_pass,
            is_token_overflow=lambda e: isinstance(e, _Overflow),
            is_cancelled=lambda: False,
            on_progress=lambda *a: None,
        )
        self.assertIsNone(err)
        # The reduce ran over findings that only exist because the bisect
        # succeeded on the half-sections.
        self.assertIn("REDUCED", report)

    def test_overflow_without_recovery_still_returns(self) -> None:
        # If it overflows even after max bisection depth, the orchestrator
        # degrades to raw findings rather than raising.
        def audit_pass(sysp: str, user: str, m: int) -> str:
            if user.startswith("Findings from"):
                raise _Overflow()  # reduce also overflows
            return "finding text"

        report, err = audit.run_map_reduce_audit(
            data="line one\nline two\n",
            focus="errors", file_label="y.log",
            chunk_chars=1000, max_chunks=40, truncated_bytes=False,
            audit_pass=audit_pass,
            is_token_overflow=lambda e: isinstance(e, _Overflow),
            is_cancelled=lambda: False,
            on_progress=lambda *a: None,
        )
        self.assertIsNone(err)
        self.assertIn("finding text", report)


class HeaderTests(unittest.TestCase):
    def test_truncation_and_sampling_noted(self) -> None:
        # Many small chunks + a low cap → "sampled evenly" in the header.
        data = "".join(f"line {i}\n" for i in range(200))
        report, _err = audit.run_map_reduce_audit(
            data=data, focus="security", file_label="huge.log",
            chunk_chars=500, max_chunks=3, truncated_bytes=True,
            audit_pass=lambda s, u, m: "NONE",
            is_token_overflow=lambda e: False,
            is_cancelled=lambda: False,
            on_progress=lambda *a: None,
        )
        self.assertIn("sampled evenly", report)
        self.assertIn("truncated to 8 MB", report)


if __name__ == "__main__":
    unittest.main()
