"""Repair GGUF vocabularies that trip llama.cpp's token-bijection assert.

Some official GGUFs — notably google/gemma-4-*-qat-q4_0-gguf — ship a vocab
where a few token *strings* appear at more than one id (e.g. ``//``, ``<?``,
``#`` in gemma-4). llama.cpp builds ``token_to_id`` as a map keyed by the
string and ``id_to_token`` as a size-``n_tokens`` vector, then asserts the two
are the same size (llama-vocab.cpp): the duplicates collapse the map, so
``GGML_ASSERT(id_to_token.size() == token_to_id.size())`` aborts the process
before the model ever loads.

The duplicates are harmless — encoding maps the string to one id (llama.cpp
keeps the last), and every id still decodes to its own text — so the fix is to
make the *earlier* (encode-dead) occurrences unique. We rewrite them in place
with a unique, same-byte-length string drawn from otherwise-unused byte
values, so the file layout, tensor data and weights are untouched: only a few
metadata bytes for tokens the tokenizer never emits change. The bijection then
holds and the stock (tagged) llama-server loads the model.

Engine-tier and ``gi``-free. Pure stdlib. Idempotent: a second pass finds
nothing to do.
"""
from __future__ import annotations

import logging
import struct
from collections import defaultdict
from pathlib import Path

log = logging.getLogger(__name__)

__all__ = ["scan_duplicate_tokens", "dedup_gguf_vocab", "GgufVocabError"]

_TOKENS_KEY = "tokenizer.ggml.tokens"


class GgufVocabError(Exception):
    """The GGUF couldn't be parsed as expected (bad magic / truncated)."""


class _Reader:
    def __init__(self, f) -> None:
        self._f = f
        self.tokens: list[tuple[int, int, bytes]] = []  # (offset, length, bytes)
        self._capture = False

    def _u(self, fmt: str):
        return struct.unpack("<" + fmt, self._f.read(struct.calcsize(fmt)))[0]

    def _value(self, vt: int):
        if vt == 8:  # STRING
            n = self._u("Q")
            off = self._f.tell()
            b = self._f.read(n)
            if self._capture:
                self.tokens.append((off, n, b))
            return b
        if vt in (0, 7):
            return self._u("B")
        if vt == 1:
            return self._u("b")
        if vt == 2:
            return self._u("H")
        if vt == 3:
            return self._u("h")
        if vt == 4:
            return self._u("I")
        if vt == 5:
            return self._u("i")
        if vt == 6:
            return self._u("f")
        if vt == 10:
            return self._u("Q")
        if vt == 11:
            return self._u("q")
        if vt == 12:
            return self._u("d")
        if vt == 9:  # ARRAY
            et = self._u("I")
            count = self._u("Q")
            for _ in range(count):
                self._value(et)
            return None
        raise GgufVocabError(f"unknown GGUF value type {vt}")

    def parse_tokens(self) -> None:
        if self._f.read(4) != b"GGUF":
            raise GgufVocabError("not a GGUF file (bad magic)")
        version = self._u("I")
        if version not in (2, 3):
            raise GgufVocabError(f"unsupported GGUF version {version}")
        self._u("Q")             # tensor count (unused here)
        n_kv = self._u("Q")
        for _ in range(n_kv):
            klen = self._u("Q")
            key = self._f.read(klen).decode("utf-8", "replace")
            vt = self._u("I")
            self._capture = key == _TOKENS_KEY
            self._value(vt)
            self._capture = False
            # We only need the tokens array; keep scanning KVs though, since it
            # may not be first and there's no index.


def _scan(path: Path) -> list[tuple[int, int, bytes]]:
    with open(path, "rb") as f:
        r = _Reader(f)
        r.parse_tokens()
    if not r.tokens:
        raise GgufVocabError("no tokenizer.ggml.tokens array found")
    return r.tokens


def scan_duplicate_tokens(path: str | Path) -> dict[bytes, list[int]]:
    """Return ``{token_bytes: [file_offsets…]}`` for every token string that
    occurs at more than one id. Empty dict = vocab is already a bijection.

    Read-only. Raises :class:`GgufVocabError` if the file isn't parseable.
    """
    tokens = _scan(Path(path))
    groups: dict[bytes, list[int]] = defaultdict(list)
    for off, _n, b in tokens:
        groups[b].append(off)
    return {b: offs for b, offs in groups.items() if len(offs) > 1}


def dedup_gguf_vocab(path: str | Path) -> int:
    """Make duplicate token strings unique in place; return how many were fixed.

    Rewrites the *earlier* (encode-dead — llama.cpp's ``token_to_id`` keeps the
    last id) occurrence of each duplicate with a unique, same-length byte
    string built from values not used by any single-byte token. Returns 0 when
    the vocab is already fine (safe to call unconditionally). Same-length edits
    mean tensor data and offsets are untouched.
    """
    p = Path(path)
    tokens = _scan(p)

    groups: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
    used = set()
    for off, n, b in tokens:
        groups[b].append((off, n))
        used.add(b)
    dups = {b: locs for b, locs in groups.items() if len(locs) > 1}
    if not dups:
        return 0

    # Byte values not used as a single-byte token — building blocks for unique
    # same-length replacements. Avoid NUL and newline to stay well-behaved.
    single = {b for b in used if len(b) == 1}
    pool = [bytes([v]) for v in range(1, 256)
            if v != 0x0a and bytes([v]) not in single]
    if not pool:
        log.warning("gguf dedup: no spare byte values to build replacements")
        return 0
    pool_iter = iter(pool)
    taken = set(used)

    def unique_of_length(length: int) -> bytes | None:
        # Greedy: draw fresh spare bytes; retry a few times on the rare clash.
        for _ in range(64):
            try:
                cand = b"".join(next(pool_iter) for _ in range(length))
            except StopIteration:
                return None
            if cand not in taken:
                taken.add(cand)
                return cand
        return None

    patches: list[tuple[int, bytes]] = []
    for b, locs in dups.items():
        for off, n in sorted(locs)[:-1]:   # keep the last id canonical
            rep = unique_of_length(n)
            if rep is None:
                log.warning("gguf dedup: ran out of unique replacements for %r", b)
                continue
            patches.append((off, rep))

    if not patches:
        return 0
    with open(p, "r+b") as out:
        for off, rep in patches:
            out.seek(off)
            out.write(rep)
    log.info("gguf dedup: made %d duplicate token(s) unique in %s",
             len(patches), p.name)
    return len(patches)
