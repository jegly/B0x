"""Qwen2 byte-level BPE tokenizer for the LiteRT diffusion pipelines.

Faithful port of Box Android's ``QwenBpeTokenizer.kt`` (which the repo notes
was verified 20/20 byte-for-byte against the fixture and against HF
``tokenizers`` on the full chat-template encoding). Fed by the plain-text
files shipped in the litert-community model repos:

- ``qwen_vocab.txt`` — one token per line, line number = id (GPT-2
  byte-to-unicode form),
- ``qwen_merges.txt`` — one ``a b`` merge pair per line, line number = rank,
- ``qwen_special.txt`` — ``token\tid`` pairs.

Pipeline: NFC normalize → split off special tokens (longest-first) → Qwen2
split regex → GPT-2 byte-level map → lowest-rank-first BPE merges → vocab
lookup. Uses the ``regex`` module for the Unicode property classes (\\p{L},
\\p{N}, \\p{White_Space}) the stdlib ``re`` can't express.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import regex

# Qwen2 pre-tokenizer regex (from tokenizer.json). The Android port rewrote
# \s/\S as \p{White_Space} for ICU; the `regex` module spells that property
# \p{White_Space} directly (Java's "IsWhite_Space" prefix isn't used here).
_SPLIT_PATTERN = regex.compile(
    r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\p{White_Space}\p{L}\p{N}]+[\r\n]*|\p{White_Space}*[\r\n]+|\p{White_Space}+(?!\P{White_Space})|\p{White_Space}+"""
)


def _build_byte_to_unicode() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class QwenBpeTokenizer:
    def __init__(
        self,
        vocab_file: str | Path,
        merges_file: str | Path,
        specials_file: str | Path,
    ) -> None:
        self._vocab: dict[str, int] = {}
        self._merges: dict[tuple[str, str], int] = {}
        self._specials: dict[str, int] = {}
        self._byte_to_unicode = _build_byte_to_unicode()

        with open(vocab_file, encoding="utf-8") as f:
            # Preserve exact line content minus the trailing newline (the
            # GPT-2 byte-level form can contain spaces/odd glyphs). Split on
            # \n and drop a single trailing empty element if the file ended
            # with a newline, so ids line up with the Kotlin's forEachIndexed.
            lines = f.read().split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            for i, line in enumerate(lines):
                self._vocab[line] = i
        with open(merges_file, encoding="utf-8") as f:
            for i, line in enumerate(f.read().split("\n")):
                if line:
                    sp = line.index(" ")
                    self._merges[(line[:sp], line[sp + 1:])] = i
        with open(specials_file, encoding="utf-8") as f:
            for line in f.read().split("\n"):
                if line:
                    tab = line.index("\t")
                    self._specials[line[:tab]] = int(line[tab + 1:].strip())

        # Longest-first so e.g. "<|im_start|>" wins over any prefix.
        keys = sorted(self._specials.keys(), key=len, reverse=True)
        self._specials_pattern = (
            regex.compile("|".join(regex.escape(k) for k in keys)) if keys else None
        )

    def encode(self, text: str) -> list[int]:
        """Encode ``text``, recognizing special tokens as single ids."""
        out: list[int] = []
        if self._specials_pattern is None:
            self._encode_plain(text, out)
            return out
        pos = 0
        for m in self._specials_pattern.finditer(text):
            if m.start() > pos:
                self._encode_plain(text[pos:m.start()], out)
            out.append(self._specials[m.group()])
            pos = m.end()
        if pos < len(text):
            self._encode_plain(text[pos:], out)
        return out

    def _encode_plain(self, raw: str, out: list[int]) -> None:
        text = unicodedata.normalize("NFC", raw)
        for m in _SPLIT_PATTERN.finditer(text):
            word = m.group()
            chars = "".join(
                self._byte_to_unicode[b] for b in word.encode("utf-8")
            )
            for piece in self._bpe(chars):
                idx = self._vocab.get(piece)
                if idx is None:
                    raise ValueError(f"BPE piece not in vocab: {piece!r}")
                out.append(idx)

    def _bpe(self, word: str) -> list[str]:
        parts = list(word)
        while len(parts) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(parts) - 1):
                r = self._merges.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            parts[best_i] = parts[best_i] + parts[best_i + 1]
            del parts[best_i + 1]
        return parts
