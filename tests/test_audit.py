"""Chunked-audit core: trigger detection, focus routing, path extraction,
line-aligned chunking, even sampling, and reduce batching.

These cover the pure policy in box_chat/audit.py — the engine's model passes
are out of scope (they need a loaded .litertlm), but every decision that drives
them is here and unit-testable.
"""
from __future__ import annotations

import unittest

from box_chat import audit


class TriggerDetectionTests(unittest.TestCase):
    def test_audit_log_phrasings_match(self) -> None:
        for msg in (
            "audit /var/log/dmesg for security issues",
            "scan this log for anything suspicious",
            "go through dmesg and find anything malicious",
            "review the syslog file",
            "check the log for errors",
            "find anything suspicious in /var/log/auth.log",
        ):
            self.assertTrue(audit.is_audit_request(msg), msg)

    def test_plain_chat_does_not_match(self) -> None:
        for msg in (
            "hello, how are you?",
            "what is the capital of France?",
            "write me a poem about logs in a fireplace",
            "summarize our conversation",  # no log/file noun, no path
        ):
            self.assertFalse(audit.is_audit_request(msg), msg)


class FocusRoutingTests(unittest.TestCase):
    def test_security_focus(self) -> None:
        self.assertEqual(
            audit.resolve_focus("audit dmesg for anything suspicious"),
            "security",
        )
        self.assertEqual(
            audit.resolve_focus("scan the log for security problems"), "security"
        )

    def test_errors_focus(self) -> None:
        self.assertEqual(
            audit.resolve_focus("check this log for errors and failures"),
            "errors",
        )

    def test_summary_focus(self) -> None:
        self.assertEqual(
            audit.resolve_focus("give me an overview of this log"), "summary"
        )

    def test_default_is_security(self) -> None:
        self.assertEqual(audit.resolve_focus("audit the log"), "security")


class PathExtractionTests(unittest.TestCase):
    def test_slash_path_ranks_first(self) -> None:
        toks = audit.extract_path_tokens(
            "audit /var/log/dmesg for security issues"
        )
        self.assertEqual(toks[0], "/var/log/dmesg")

    def test_strips_surrounding_punctuation(self) -> None:
        toks = audit.extract_path_tokens("please scan `/var/log/auth.log`.")
        self.assertIn("/var/log/auth.log", toks)

    def test_dotted_filename_before_bare_word(self) -> None:
        toks = audit.extract_path_tokens("scan dmesg.log now")
        self.assertLess(toks.index("dmesg.log"), toks.index("now"))

    def test_dedup(self) -> None:
        toks = audit.extract_path_tokens("audit a.log and a.log again")
        self.assertEqual(toks.count("a.log"), 1)


class ChunkingTests(unittest.TestCase):
    # chunk_lines clamps max_chars up to a 500-char floor, so tests use
    # realistic section sizes (production uses >= 2000).
    def test_lines_kept_intact_and_line_numbers(self) -> None:
        text = "".join(f"line {i:04d} " + "x" * 50 + "\n" for i in range(1, 41))
        chunks = audit.chunk_lines(text, max_chars=500)
        self.assertGreater(len(chunks), 1)  # actually split into sections
        # Reassembling all chunk texts reproduces the file exactly.
        self.assertEqual("".join(c.text for c in chunks), text)
        # First chunk starts at line 1; chunks are contiguous and ordered.
        self.assertEqual(chunks[0].start_line, 1)
        for a, b in zip(chunks, chunks[1:]):
            self.assertEqual(b.start_line, a.end_line + 1)
        self.assertEqual(chunks[-1].end_line, 40)

    def test_each_chunk_within_cap(self) -> None:
        text = "".join(f"{'x' * 70}\n" for _ in range(50))
        chunks = audit.chunk_lines(text, max_chars=600)
        for c in chunks:
            self.assertLessEqual(len(c.text), 600)

    def test_overlong_single_line_is_hard_split(self) -> None:
        text = "y" * 1500 + "\n"
        chunks = audit.chunk_lines(text, max_chars=600)
        self.assertTrue(len(chunks) >= 3)
        for c in chunks:
            self.assertLessEqual(len(c.text), 600)
            self.assertEqual((c.start_line, c.end_line), (1, 1))


class SplitChunkTests(unittest.TestCase):
    def test_splits_multiline_in_half_preserving_text(self) -> None:
        text = "".join(f"line {i}\n" for i in range(10))
        ch = audit.Chunk(1, 10, text)
        halves = audit.split_chunk(ch)
        self.assertEqual(len(halves), 2)
        self.assertEqual(halves[0].text + halves[1].text, text)

    def test_splits_single_giant_line_by_chars(self) -> None:
        ch = audit.Chunk(5, 5, "z" * 2000)
        halves = audit.split_chunk(ch)
        self.assertEqual(len(halves), 2)
        self.assertEqual(halves[0].text + halves[1].text, "z" * 2000)
        self.assertTrue(all(len(h.text) < 2000 for h in halves))


class SamplingTests(unittest.TestCase):
    def _chunks(self, n: int) -> list[audit.Chunk]:
        return [audit.Chunk(i, i, f"c{i}") for i in range(n)]

    def test_no_sampling_under_cap(self) -> None:
        chunks, sampled = audit.sample_chunks(self._chunks(10), 40)
        self.assertFalse(sampled)
        self.assertEqual(len(chunks), 10)

    def test_samples_evenly_with_endpoints(self) -> None:
        chunks, sampled = audit.sample_chunks(self._chunks(100), 10)
        self.assertTrue(sampled)
        self.assertLessEqual(len(chunks), 10)
        # endpoints preserved
        self.assertEqual(chunks[0].start_line, 0)
        self.assertEqual(chunks[-1].start_line, 99)

    def test_zero_cap_means_unlimited(self) -> None:
        chunks, sampled = audit.sample_chunks(self._chunks(100), 0)
        self.assertFalse(sampled)
        self.assertEqual(len(chunks), 100)


class NoneFindingTests(unittest.TestCase):
    def test_recognises_none_variants(self) -> None:
        for s in ("NONE", "none", "None.", "NONE\n", "nothing notable here"):
            self.assertTrue(audit.is_none_finding(s), s)

    def test_real_finding_is_kept(self) -> None:
        self.assertFalse(audit.is_none_finding(
            "Line 42: sudo: 3 incorrect password attempts for root"
        ))


class ReduceBatchingTests(unittest.TestCase):
    def test_single_batch_when_small(self) -> None:
        findings = ["a", "b", "c"]
        batches = audit.batch_findings(findings, max_chars=1000)
        self.assertEqual(len(batches), 1)

    def test_splits_when_over_budget(self) -> None:
        findings = ["x" * 90 for _ in range(10)]
        batches = audit.batch_findings(findings, max_chars=200)
        self.assertGreater(len(batches), 1)
        # every finding survives exactly once
        flat = [f for b in batches for f in b]
        self.assertEqual(len(flat), 10)


if __name__ == "__main__":
    unittest.main()
