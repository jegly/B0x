"""Chunked, local file/log auditing (map-reduce over a small context window).

Dumping a whole log into the model overflows a small context window — a 50k-token
`dmesg` versus a 4k window is the failure the user hit. This module drives a
map-reduce pass instead:

- **map:** split the file into window-sized sections and have the model flag the
  notable events in each one,
- **reduce:** combine those per-section findings into one report.

Box's engine layer runs the actual model passes (see ``engine._CmdAuditFile``) so
the audit never re-enters the live chat conversation. This module is the pure,
``gi``-free policy: chunking, the focus presets, the prompts, and the
natural-language trigger. Keep it import-light — it must never import ``gi``
(see the project-core-must-stay-portable rule).
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

# Hard ceiling on how much of a file we'll read, regardless of settings — a
# runaway multi-GB log would otherwise eat memory and hours of CPU.
MAX_AUDIT_BYTES = 8 * 1024 * 1024  # 8 MB
DEFAULT_MAX_CHUNKS = 40

# Literal sentinel a section returns when it has nothing worth reporting.
_NONE = "NONE"
DEFAULT_FOCUS = "security"


@dataclass(frozen=True)
class Chunk:
    start_line: int      # 1-based, inclusive
    end_line: int        # 1-based, inclusive
    text: str


# ── Focus presets ──────────────────────────────────────────────────────────
# Each preset supplies the "map" instruction (applied to every section) and the
# "reduce" instruction (applied to the collected findings).
FOCUS_PRESETS: dict[str, dict[str, str]] = {
    "security": {
        "label": "security",
        "map": (
            "You are a security log auditor. Examine the log excerpt below and "
            "list ONLY genuinely notable or security-relevant events: "
            "authentication failures, privilege escalation (sudo/su/root), "
            "unexpected USB/device or kernel-module events, segfaults or "
            "crashes, OOM kills, kernel taints, firewall/network anomalies, "
            "permission denials, and anything that looks like tampering or "
            "intrusion. For each, quote the relevant line and say briefly why "
            "it stands out. Do not speculate beyond the lines shown. If nothing "
            "in this excerpt is noteworthy, reply with exactly: " + _NONE
        ),
        "reduce": (
            "You are a security analyst writing the final audit report for a "
            "log file. Below are notable findings extracted from consecutive "
            "sections of the file. Write one concise report with: (1) an overall "
            "assessment, (2) the suspicious or security-relevant events grouped "
            "and de-duplicated, most important first, with the evidence, and "
            "(3) recommended next steps. If the findings are trivial, say the "
            "log looks clean and why."
        ),
        "clean": (
            "No notable security issues were found across the audited sections."
        ),
    },
    "errors": {
        "label": "errors & failures",
        "map": (
            "You are a log triage assistant. From the log excerpt below, list "
            "ONLY errors, warnings, failures, crashes, and anything indicating "
            "something went wrong. Quote the relevant line for each and note "
            "what it suggests. If the excerpt has nothing of that kind, reply "
            "with exactly: " + _NONE
        ),
        "reduce": (
            "Combine the per-section findings below into one concise report of "
            "the errors and failures in this log: group related issues, order "
            "by severity, quote representative evidence, and suggest likely "
            "causes or next steps. If findings are empty, say the log shows no "
            "errors."
        ),
        "clean": "No errors or failures were found across the audited sections.",
    },
    "summary": {
        "label": "summary",
        "map": (
            "Summarize the key events in the log excerpt below as a few short "
            "bullet points — what happened, in order. Omit routine noise. If "
            "the excerpt is entirely routine or empty, reply with exactly: "
            + _NONE
        ),
        "reduce": (
            "Combine the per-section notes below into one concise, ordered "
            "summary of what this log shows overall. Group related events and "
            "keep it brief."
        ),
        "clean": "The log appears routine — nothing notable to summarize.",
    },
}


def _preset(focus: str) -> dict[str, str]:
    return FOCUS_PRESETS.get(focus, FOCUS_PRESETS[DEFAULT_FOCUS])


def none_token() -> str:
    return _NONE


def focus_label(focus: str) -> str:
    return _preset(focus)["label"]


def map_system(focus: str) -> str:
    return _preset(focus)["map"]


def reduce_system(focus: str) -> str:
    return _preset(focus)["reduce"]


def clean_report(focus: str) -> str:
    return _preset(focus)["clean"]


def is_none_finding(s: str) -> bool:
    """True when a section's reply means 'nothing notable here'."""
    low = s.strip().lower()
    if not low:
        return True
    if low.rstrip(".!").strip() == _NONE.lower():
        return True
    if low.startswith("none"):
        return True
    if len(low) < 60 and (
        "nothing notable" in low
        or "no notable" in low
        or "no suspicious" in low
        or "no errors" in low
        or "nothing of" in low
    ):
        return True
    return False


# ── Natural-language trigger ────────────────────────────────────────────────
_AUDIT_VERBS = (
    "audit", "scan", "analyse", "analyze", "inspect", "review",
    "go through", "look through", "comb through", "check", "examine",
)
_SUSPICIOUS_HINTS = (
    "suspicious", "anomal", "intrusion", "compromise", "breach",
    "malicious", "security", "vulnerab",
)
_ERROR_HINTS = ("error", "warning", "failure", "failed", "crash", "fault")
_SUMMARY_HINTS = (
    "summar", "overview", "what happened", "rundown", "tldr", "tl;dr",
)
_NOUN_HINTS = (
    "log", "logs", "file", "journal", "dmesg", "syslog", "output",
)


def is_audit_request(text: str) -> bool:
    """Heuristic: does this message read like 'audit/scan this log'?

    Conservative on purpose — the caller ALSO requires a real, readable file
    in the workspace before routing here, so a false positive is harmless
    unless a matching file also happens to be named.
    """
    t = text.lower()
    has_verb = any(v in t for v in _AUDIT_VERBS)
    has_noun = any(n in t for n in _NOUN_HINTS)
    has_suspect = any(h in t for h in _SUSPICIOUS_HINTS)
    return (has_verb and (has_noun or has_suspect)) or (has_suspect and has_noun)


def resolve_focus(text: str) -> str:
    t = text.lower()
    if any(h in t for h in _SUSPICIOUS_HINTS):
        return "security"
    if any(h in t for h in _SUMMARY_HINTS):
        return "summary"
    if any(h in t for h in _ERROR_HINTS):
        return "errors"
    return DEFAULT_FOCUS


_PATH_TOKEN = re.compile(r"""^["'`(<]*([^\s"'`)>]+?)["'`)>.,;:]*$""")


def extract_path_tokens(text: str) -> list[str]:
    """Pull candidate path tokens from a message, most-specific first.

    Slash paths rank first (explicit), then dotted filenames, then bare words.
    The caller resolves each against the workspace root and keeps the first
    that is a readable file.
    """
    cands: list[str] = []
    for raw in text.split():
        m = _PATH_TOKEN.match(raw)
        if not m:
            continue
        tok = m.group(1)
        if not tok or tok in (".", ".."):
            continue
        cands.append(tok)

    def rank(tok: str) -> int:
        if "/" in tok:
            return 0
        if "." in tok:
            return 1
        return 2

    # dict.fromkeys dedupes while preserving order; sorted() is stable.
    return sorted(dict.fromkeys(cands), key=rank)


# ── Chunking ────────────────────────────────────────────────────────────────
def chunk_lines(text: str, max_chars: int) -> list[Chunk]:
    """Split text into line-aligned chunks of at most ``max_chars`` characters.

    Whole log lines are kept intact so the model never sees a line cut in half
    — except a single line longer than ``max_chars``, which is hard-split so it
    can't exceed the window on its own.
    """
    if max_chars < 500:
        max_chars = 500
    lines = text.splitlines(keepends=True)
    chunks: list[Chunk] = []
    buf: list[str] = []
    size = 0
    start = 1
    for i, line in enumerate(lines, 1):
        if len(line) > max_chars:
            if buf:
                chunks.append(Chunk(start, i - 1, "".join(buf)))
                buf, size = [], 0
            for j in range(0, len(line), max_chars):
                chunks.append(Chunk(i, i, line[j:j + max_chars]))
            start = i + 1
            continue
        if size + len(line) > max_chars and buf:
            chunks.append(Chunk(start, i - 1, "".join(buf)))
            buf, size = [], 0
            start = i
        buf.append(line)
        size += len(line)
    if buf:
        chunks.append(Chunk(start, len(lines), "".join(buf)))
    return chunks


def split_chunk(chunk: Chunk) -> list[Chunk]:
    """Split one chunk into two at a line boundary — the on-the-fly fallback
    for a section that still overflows the window (dense hex/base64). A
    single unsplittable line is halved by characters. Line spans are
    approximate for the sub-pieces; they're only used for display labels.
    """
    lines = chunk.text.splitlines(keepends=True)
    if len(lines) < 2:
        half = max(400, len(chunk.text) // 2)
        if half >= len(chunk.text):
            return [chunk]
        return [
            Chunk(chunk.start_line, chunk.end_line, chunk.text[:half]),
            Chunk(chunk.start_line, chunk.end_line, chunk.text[half:]),
        ]
    mid = len(lines) // 2
    span = max(0, chunk.end_line - chunk.start_line)
    mid_line = chunk.start_line + span * mid // len(lines)
    return [
        Chunk(chunk.start_line, mid_line, "".join(lines[:mid])),
        Chunk(min(mid_line + 1, chunk.end_line), chunk.end_line,
              "".join(lines[mid:])),
    ]


def sample_chunks(
    chunks: list[Chunk], max_chunks: int
) -> tuple[list[Chunk], bool]:
    """Cap chunk count by sampling evenly across the file (endpoints kept).

    Returns ``(chunks, sampled)`` where ``sampled`` is True if anything was
    dropped — the caller surfaces that in the report header.
    """
    n = len(chunks)
    if max_chunks <= 0 or n <= max_chunks:
        return chunks, False
    if max_chunks == 1:
        return [chunks[0]], True
    idxs = sorted({
        round(k * (n - 1) / (max_chunks - 1)) for k in range(max_chunks)
    })
    return [chunks[i] for i in idxs], True


# ── Prompt bodies ───────────────────────────────────────────────────────────
def map_user_message(file_label: str, chunk: Chunk) -> str:
    return (
        f"Log excerpt from {file_label} "
        f"(lines {chunk.start_line}–{chunk.end_line}):\n\n{chunk.text}"
    )


def reduce_user_message(file_label: str, findings: list[str]) -> str:
    joined = "\n\n---\n\n".join(findings)
    return f"Findings from {file_label}:\n\n{joined}"


def batch_findings(findings: list[str], max_chars: int) -> list[list[str]]:
    """Group findings so each batch fits the reduce window. Two-level reduce
    uses this when there are too many findings for a single pass."""
    if max_chars < 500:
        max_chars = 500
    batches: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for f in findings:
        flen = len(f) + 8  # account for the "\n\n---\n\n" separator
        if cur and size + flen > max_chars:
            batches.append(cur)
            cur, size = [], 0
        cur.append(f)
        size += flen
    if cur:
        batches.append(cur)
    return batches


# ── Backend-agnostic orchestration ──────────────────────────────────────────
# ``AuditPass`` runs one model pass: (system_prompt, user_text, max_out_chars)
# → the model's reply text. The litert path creates a throwaway conversation;
# the llama path does a stateless completion. Everything else lives here and
# both engine paths call it.
AuditPass = Callable[[str, str, int], str]
OverflowCheck = Callable[[Exception], bool]
Cancelled = Callable[[], bool]
Progress = Callable[[int, int, str], None]


def chunk_chars_for_context(ctx_tokens: int) -> int:
    """Section size (chars) that fits a model with ``ctx_tokens`` of context.

    Log content tokenizes far denser than prose (~1.5 chars/token worst case
    on dmesg vs ~4 for English), so size conservatively and reserve room for
    the instruction + the model's findings. A section that still overflows is
    bisected on the fly by :func:`run_map_reduce_audit`.
    """
    min_cpt = 1.5
    sys_tok = 320
    out_tok = max(700, ctx_tokens // 4)
    budget_tok = max(256, ctx_tokens - out_tok - sys_tok)
    return max(1200, int(budget_tok * min_cpt))


def run_map_reduce_audit(
    *,
    data: str,
    focus: str,
    file_label: str,
    chunk_chars: int,
    max_chunks: int,
    truncated_bytes: bool,
    audit_pass: AuditPass,
    is_token_overflow: OverflowCheck,
    is_cancelled: Cancelled,
    on_progress: Progress,
) -> tuple[str | None, str | None]:
    """Run the full map-reduce audit. Returns ``(report, error)``.

    Backend-neutral: the caller supplies ``audit_pass`` and
    ``is_token_overflow``. ``is_cancelled()`` is polled between sections and
    batches for responsive Stop.
    """
    reduce_chars = chunk_chars
    chunks = chunk_lines(data, chunk_chars)
    chunks, sampled = sample_chunks(chunks, max_chunks)
    total = len(chunks)

    map_sys = map_system(focus)
    reduce_sys = reduce_system(focus)

    def _scan_chunk(ch: Chunk, depth: int = 0) -> str:
        try:
            out = audit_pass(map_sys, map_user_message(file_label, ch), 8000)
        except Exception as e:  # noqa: BLE001
            if not is_token_overflow(e) or depth >= 5:
                raise
            out = None
        if out is not None:
            return "" if is_none_finding(out) else out
        parts: list[str] = []
        for h in split_chunk(ch):
            if is_cancelled():
                break
            sub = _scan_chunk(h, depth + 1)
            if sub:
                parts.append(sub)
        return "\n".join(parts)

    def _reduce_pass(items: list[str], depth: int = 0) -> str:
        if not items:
            return ""
        if len(items) == 1:
            try:
                return audit_pass(
                    reduce_sys,
                    reduce_user_message(file_label, [items[0][:reduce_chars]]),
                    8000,
                )
            except Exception as e:  # noqa: BLE001
                if is_token_overflow(e):
                    return items[0][:reduce_chars]
                raise
        try:
            return audit_pass(
                reduce_sys, reduce_user_message(file_label, items), 8000
            )
        except Exception as e:  # noqa: BLE001
            if not is_token_overflow(e) or depth >= 6:
                return "\n\n".join(items)
            mid = len(items) // 2
            left = _reduce_pass(items[:mid], depth + 1)
            right = _reduce_pass(items[mid:], depth + 1)
            try:
                return audit_pass(
                    reduce_sys, reduce_user_message(file_label, [left, right]),
                    8000,
                )
            except Exception as e2:  # noqa: BLE001
                if is_token_overflow(e2):
                    return left + "\n\n" + right
                raise

    findings: list[str] = []
    cancelled = False
    for i, ch in enumerate(chunks, 1):
        if is_cancelled():
            cancelled = True
            break
        on_progress(i, total, "scan")
        out = _scan_chunk(ch)
        if out:
            findings.append(f"[lines {ch.start_line}–{ch.end_line}]\n{out}")

    header = [
        f"{total} section{'s' if total != 1 else ''}",
        f"focus: {focus_label(focus)}",
    ]
    if sampled:
        header.append("large file — sampled evenly")
    if truncated_bytes:
        header.append("truncated to 8 MB")
    if cancelled:
        header.append("stopped early")
    head = f"**Audit of {file_label}** ({', '.join(header)})\n\n"

    if not findings:
        if cancelled:
            return (
                head + "⚠ Stopped before the audit finished — nothing was "
                "flagged in the sections scanned so far, but this is NOT a "
                "complete audit."
            ), None
        return head + clean_report(focus), None

    raw_findings = "\n\n".join(findings)
    batches = batch_findings(findings, reduce_chars)
    n_reduce = len(batches) + (1 if len(batches) > 1 else 0)
    if len(batches) == 1:
        on_progress(1, n_reduce, "report")
        body = _reduce_pass(batches[0])
    else:
        partials: list[str] = []
        for bi, b in enumerate(batches, 1):
            if is_cancelled():
                break
            on_progress(bi, n_reduce, "report")
            partials.append(_reduce_pass(b))
        if partials and not is_cancelled():
            on_progress(n_reduce, n_reduce, "report")
            body = _reduce_pass(partials)
        else:
            body = "\n\n".join(partials)

    if body and body.strip():
        report = head + body
        if cancelled:
            report += (
                "\n\n---\n*Audit stopped early — this summary covers the "
                "sections scanned so far.*"
            )
        return report, None
    note = (
        "⚠ Stopped before the summary was written — raw findings collected "
        "so far:\n\n"
        if cancelled else
        "⚠ Could not generate a summary — raw findings below:\n\n"
    )
    return head + note + raw_findings, None
