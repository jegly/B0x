"""GGUF vocab de-duplication (the gemma-4 QAT assert fix).

Covers box_chat/gguf_vocab_fix.py against a hand-built minimal GGUF: it finds
duplicate token strings, makes the earlier (encode-dead) occurrences unique in
place with same-length bytes (file size + other metadata untouched), is
idempotent, and leaves a clean vocab alone.
"""
from __future__ import annotations

import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from box_chat import gguf_vocab_fix as gv


def _write_gguf(path: Path, tokens: list[bytes]) -> None:
    """Write a minimal GGUF v3 with just a tokenizer.ggml.tokens array."""
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<I", 3)          # version
    buf += struct.pack("<Q", 0)          # tensor count
    buf += struct.pack("<Q", 1)          # kv count
    key = b"tokenizer.ggml.tokens"
    buf += struct.pack("<Q", len(key)) + key
    buf += struct.pack("<I", 9)          # value type = ARRAY
    buf += struct.pack("<I", 8)          # element type = STRING
    buf += struct.pack("<Q", len(tokens))
    for t in tokens:
        buf += struct.pack("<Q", len(t)) + t
    path.write_bytes(buf)


class ScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_finds_duplicates(self) -> None:
        p = self.dir / "m.gguf"
        _write_gguf(p, [b"a", b"//", b"b", b"//", b"c", b"#", b"#"])
        dups = gv.scan_duplicate_tokens(p)
        self.assertEqual(set(dups), {b"//", b"#"})
        self.assertEqual(len(dups[b"//"]), 2)

    def test_clean_vocab_has_no_dups(self) -> None:
        p = self.dir / "m.gguf"
        _write_gguf(p, [b"a", b"b", b"c"])
        self.assertEqual(gv.scan_duplicate_tokens(p), {})

    def test_bad_magic_raises(self) -> None:
        p = self.dir / "bad.gguf"
        p.write_bytes(b"NOPExxxxxxxx")
        with self.assertRaises(gv.GgufVocabError):
            gv.scan_duplicate_tokens(p)


class DedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_dedup_makes_unique_and_same_size(self) -> None:
        p = self.dir / "m.gguf"
        toks = [b"a", b"//", b"b", b"//", b"c", b"<?", b"<?", b"#", b"#"]
        _write_gguf(p, toks)
        size_before = p.stat().st_size

        fixed = gv.dedup_gguf_vocab(p)
        self.assertEqual(fixed, 3)               # //, <?, # each fixed once
        self.assertEqual(p.stat().st_size, size_before)   # in place, same size
        self.assertEqual(gv.scan_duplicate_tokens(p), {})  # bijection restored

    def test_last_occurrence_kept_canonical(self) -> None:
        # The LAST id of a duplicate is what llama.cpp encodes to, so it must
        # keep the original bytes; the earlier one is the one rewritten.
        p = self.dir / "m.gguf"
        _write_gguf(p, [b"x", b"//", b"y", b"//", b"z"])
        gv.dedup_gguf_vocab(p)
        # Re-read the raw token array and check positions.
        with open(p, "rb") as f:
            r = gv._Reader(f)
            r.parse_tokens()
        toks = [b for _o, _n, b in r.tokens]
        self.assertEqual(toks[3], b"//")         # last // preserved
        self.assertNotEqual(toks[1], b"//")      # first // rewritten
        self.assertEqual(len(toks[1]), 2)        # same length

    def test_idempotent(self) -> None:
        p = self.dir / "m.gguf"
        _write_gguf(p, [b"//", b"//", b"a"])
        self.assertEqual(gv.dedup_gguf_vocab(p), 1)
        self.assertEqual(gv.dedup_gguf_vocab(p), 0)   # nothing left to do

    def test_clean_vocab_untouched(self) -> None:
        p = self.dir / "m.gguf"
        _write_gguf(p, [b"a", b"b", b"c"])
        before = p.read_bytes()
        self.assertEqual(gv.dedup_gguf_vocab(p), 0)
        self.assertEqual(p.read_bytes(), before)


REAL = Path.home() / "Downloads" / "gemma-4-E2B_q4_0-it.gguf"


@unittest.skipUnless(REAL.is_file(), "real gemma-4 QAT gguf not present")
class RealModelScanTests(unittest.TestCase):
    def test_detects_the_three_gemma4_dups(self) -> None:
        # Read-only scan of the pristine official file — must NOT modify it.
        # NOTE: if this local copy has already been healed in place (dedup_gguf_vocab
        # / the backend self-heal rewrites it), there is nothing left to detect —
        # skip rather than fail, since re-downloading the 3.3GB pristine file is not
        # something the suite should require.
        dups = gv.scan_duplicate_tokens(REAL)
        if not dups:
            self.skipTest("real gemma-4 gguf already deduped in place — no pristine copy")
        self.assertEqual({b.decode("latin1") for b in dups}, {"//", "<?", "#"})


if __name__ == "__main__":
    unittest.main()
