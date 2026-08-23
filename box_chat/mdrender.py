"""Markdown → Pango markup for chat bubbles.

The assistant emits Markdown (headings, **bold**, lists, `code`, fenced code
blocks, pipe tables). ``Gtk.Label`` can't render Markdown but it does render
Pango markup — a small XML-ish subset (``<b>``, ``<i>``, ``<tt>``, ``<span>``
…). This module converts the common Markdown the model produces into valid
Pango markup.

Deliberately dependency-free: a Markdown library would only give us HTML, and
we'd still have to translate to Pango. We control the (small) subset here.

Pango has no table primitive, so pipe tables are rendered as aligned
monospace text. Links render as underlined text (the URL is dropped — the
model usually also writes the bare URL in prose, and we don't want clickable
links auto-launching a browser).

``to_pango_markup`` always returns a string; if anything looks off the caller
should still wrap ``Gtk.Label.set_markup`` in try/except and fall back to
plain text, because Pango rejects the whole string on a single malformed tag.
"""
from __future__ import annotations

import html
import re

# ── block-level patterns ────────────────────────────────────────────────────
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_HR = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_ULIST = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_OLIST = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_BQ = re.compile(r"^\s*>\s?(.*)$")
_FENCE = re.compile(r"^\s*```")

# Heading level → Pango size keyword.
_HEADING_SIZE = {1: "xx-large", 2: "x-large", 3: "large",
                 4: "medium", 5: "medium", 6: "medium"}

# ── inline patterns ─────────────────────────────────────────────────────────
_HOLD_RE = re.compile(r"\x00(\d+)\x00")
_CODE = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_BOLD_ITALIC = re.compile(r"\*\*\*(.+?)\*\*\*")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<!\w)\*(.+?)\*(?!\w)|(?<!\w)_(.+?)_(?!\w)")
_STRIKE = re.compile(r"~~(.+?)~~")


def _esc(s: str) -> str:
    # Escape &, <, > — the three chars Pango treats as markup. Leave quotes;
    # they're only special inside attribute values, which we never build from
    # user/model text.
    return html.escape(s, quote=False)


# ── LaTeX-subset → Unicode/Pango ─────────────────────────────────────────────
# A pragmatic math converter: maps the common LaTeX commands the model emits
# to Unicode glyphs, and uses Pango <sup>/<sub> for powers/indices. Not a
# typesetter — no real fraction/integral layout — but covers the bulk of
# inline math. Display ($$…$$) blocks get this too as a fallback; when
# matplotlib is available the bubble swaps them for rendered images.
_MATH_SYMBOLS = {
    # greek lower
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "vartheta": "ϑ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ",
    "pi": "π", "varpi": "ϖ", "rho": "ρ", "varrho": "ϱ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ", "varphi": "ϕ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    # greek upper
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Upsilon": "Υ", "Phi": "Φ", "Psi": "Ψ",
    "Omega": "Ω",
    # operators / relations
    "times": "×", "div": "÷", "cdot": "·", "pm": "±", "mp": "∓", "ast": "∗",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠", "ne": "≠",
    "approx": "≈", "equiv": "≡", "cong": "≅", "sim": "∼", "simeq": "≃",
    "propto": "∝", "ll": "≪", "gg": "≫",
    # big operators / calculus
    "infty": "∞", "partial": "∂", "nabla": "∇", "sum": "∑", "prod": "∏",
    "int": "∫", "iint": "∬", "iiint": "∭", "oint": "∮", "surd": "√",
    # sets / logic
    "in": "∈", "notin": "∉", "ni": "∋", "subset": "⊂", "subseteq": "⊆",
    "supset": "⊃", "supseteq": "⊇", "cup": "∪", "cap": "∩", "emptyset": "∅",
    "varnothing": "∅", "forall": "∀", "exists": "∃", "nexists": "∄",
    "neg": "¬", "lnot": "¬", "land": "∧", "wedge": "∧", "lor": "∨",
    "vee": "∨", "oplus": "⊕", "otimes": "⊗", "perp": "⊥", "parallel": "∥",
    # arrows
    "rightarrow": "→", "to": "→", "leftarrow": "←", "gets": "←",
    "leftrightarrow": "↔", "Rightarrow": "⇒", "implies": "⇒",
    "Leftarrow": "⇐", "Leftrightarrow": "⇔", "iff": "⇔", "mapsto": "↦",
    "uparrow": "↑", "downarrow": "↓",
    # delimiters / misc
    "langle": "⟨", "rangle": "⟩", "lfloor": "⌊", "rfloor": "⌋",
    "lceil": "⌈", "rceil": "⌉", "hbar": "ℏ", "ell": "ℓ", "Re": "ℜ",
    "Im": "ℑ", "aleph": "ℵ", "prime": "′", "circ": "∘", "bullet": "•",
    "star": "⋆", "dagger": "†", "ddagger": "‡", "angle": "∠",
    "triangle": "△", "cdots": "⋯", "ldots": "…", "dots": "…", "vdots": "⋮",
    "ddots": "⋱", "deg": "°",
}
# Replace longer command names first so e.g. \subseteq isn't eaten by \subset.
_MATH_SYMBOLS_SORTED = sorted(
    _MATH_SYMBOLS.items(), key=lambda kv: len(kv[0]), reverse=True
)
_ACCENTS = {"vec": "⃗", "hat": "̂", "bar": "̄",
            "tilde": "̃", "dot": "̇"}


def _accent(text: str, comb: str) -> str:
    # Put the combining mark after the first character (best effort).
    return (text[:1] + comb + text[1:]) if text else ""


def _normalize_latex(s: str) -> str:
    """Collapse LLM-style doubled backslashes so \\\\frac becomes \\frac. LLMs
    often double-escape commands when emitting LaTeX inside markdown, which
    breaks both matplotlib parsing and the regex-based Unicode fallback.
    Iterates to handle 3-4× over-escaping."""
    prev = None
    while s != prev:
        prev = s
        s = s.replace("\\\\", "\\")
    return s


def _read_braced(s: str, i: int) -> tuple[str, int] | None:
    """If s[i] is '{', return (inner_text, end_index_after_close); else None.
    Handles nested braces — essential for \\frac{Q_{\\text{enc}}}{...} where
    the simple regex [^{}]* gives up at the first inner '{'."""
    if i >= len(s) or s[i] != "{":
        return None
    depth = 1
    j = i + 1
    while j < len(s) and depth > 0:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return s[i + 1 : j - 1], j


def _expand_command(s: str, cmd_pattern: str,
                    wrap: "Callable[[list[str]], str]",
                    nargs: int = 1) -> str:
    """Find every occurrence of cmd_pattern and consume the next ``nargs``
    brace-balanced groups. ``wrap`` builds the replacement from the groups.
    Recurses into each captured group so nested commands resolve too."""
    out: list[str] = []
    i = 0
    n = len(s)
    pat = re.compile(cmd_pattern)
    while i < n:
        m = pat.match(s, i)
        if not m:
            out.append(s[i])
            i += 1
            continue
        j = m.end()
        args: list[str] = []
        ok = True
        for _ in range(nargs):
            while j < n and s[j].isspace():
                j += 1
            br = _read_braced(s, j)
            if br is None:
                ok = False
                break
            args.append(br[0])
            j = br[1]
        if not ok:
            out.append(s[i])
            i += 1
            continue
        out.append(wrap(args))
        i = j
    return "".join(out)


def _math_to_pango(tex: str) -> str:
    """Convert a LaTeX math fragment to Pango markup (Unicode subset).

    Brace-balanced for the common multi-arg / nested-brace constructs the
    LLM emits (\\frac{Q_{\\text{enc}}}{...}, \\mathbf{\\nabla\\cdot E}, etc.),
    iterating until the structure converges. Falls back to a flat sweep of
    named-symbol replacements after that."""
    s = _esc(_normalize_latex(tex.strip()))
    # \left( \right) delimiters — drop the command, keep the delimiter. Guard
    # the negative-lookahead so \leftarrow / \rightarrow survive.
    s = re.sub(r"\\left(?![a-zA-Z])\s*", "", s)
    s = re.sub(r"\\right(?![a-zA-Z])\s*", "", s)
    # \sqrt[n]{x} — drop the optional [n] root index (no clean Unicode form).
    s = re.sub(r"\\sqrt\s*\[[^\]]*\]\s*", r"\\sqrt", s)

    # Brace-balanced expansions. Iterate until a pass produces no change so
    # nested constructs (\\frac inside \\mathbf, \\mathbf inside \\boxed) all
    # resolve. 8 iterations is generous — three-level nesting is rare.
    for _ in range(8):
        prev = s
        s = _expand_command(
            s, r"\\[dt]?frac\b\s*",
            lambda a: f"({a[0]})/({a[1]})", nargs=2,
        )
        s = _expand_command(
            s, r"\\sqrt\b\s*",
            lambda a: f"√({a[0]})", nargs=1,
        )
        for name, comb in _ACCENTS.items():
            s = _expand_command(
                s, rf"\\{name}\b\s*",
                lambda a, c=comb: _accent(a[0], c), nargs=1,
            )
        # \text / \mathrm / \operatorname / \boxed — unwrap, drop the command.
        s = _expand_command(
            s, r"\\(?:text|mathrm|operatorname|boxed)\b\s*",
            lambda a: a[0], nargs=1,
        )
        s = _expand_command(
            s, r"\\mathbf\b\s*",
            lambda a: f"<b>{a[0]}</b>", nargs=1,
        )
        s = _expand_command(
            s, r"\\mathit\b\s*",
            lambda a: f"<i>{a[0]}</i>", nargs=1,
        )
        # Braced super/subscripts (brace-balanced — single-char form below).
        s = _expand_command(
            s, r"\^\s*",
            lambda a: f"<sup>{a[0]}</sup>", nargs=1,
        )
        s = _expand_command(
            s, r"_\s*",
            lambda a: f"<sub>{a[0]}</sub>", nargs=1,
        )
        if s == prev:
            break

    # Single-character sub/sup (e.g. x^2, a_i).
    s = re.sub(r"\^(\S)", lambda m: f"<sup>{m.group(1)}</sup>", s)
    s = re.sub(r"_(\S)", lambda m: f"<sub>{m.group(1)}</sub>", s)
    # Named symbols (\\alpha → α, …).
    for name, ch in _MATH_SYMBOLS_SORTED:
        s = s.replace("\\" + name, ch)
    # Spacing commands.
    s = re.sub(r"\\[,;:! ]", " ", s)
    s = re.sub(r"\\q?quad", "  ", s)
    # Cleanup — drop leftover \\command tokens, stray braces, lone backslashes.
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("\\", "")
    s = s.replace("{", "").replace("}", "")
    return s


# Inline math: $…$ (not $$) or \(…\). Display $$…$$ / \[…\] handled per-block.
_MATH_INLINE = re.compile(r"\$(?!\$)([^$\n]+?)\$|\\\((.+?)\\\)")


def _inline(s: str) -> str:
    """Render inline Markdown (bold/italic/code/strike/links) to Pango."""
    holds: list[str] = []

    def hold(markup: str) -> str:
        holds.append(markup)
        return f"\x00{len(holds) - 1}\x00"

    # Pull out spans whose *content* must not be re-parsed (code), already
    # produce markup (math, links), before escaping the rest. Code wins over
    # math so a `$x$` inside backticks stays literal.
    s = _CODE.sub(lambda m: hold(f"<tt>{_esc(m.group(1))}</tt>"), s)
    s = _MATH_INLINE.sub(
        lambda m: hold(_math_to_pango(m.group(1) or m.group(2) or "")), s
    )
    s = _LINK.sub(lambda m: hold(f"<u>{_esc(m.group(1))}</u>"), s)
    s = _esc(s)
    s = _BOLD_ITALIC.sub(lambda m: f"<b><i>{m.group(1)}</i></b>", s)
    s = _BOLD.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", s)
    s = _ITALIC.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", s)
    s = _STRIKE.sub(lambda m: f"<s>{m.group(1)}</s>", s)
    s = _HOLD_RE.sub(lambda m: holds[int(m.group(1))], s)
    return s


def _is_table_sep(line: str) -> bool:
    s = line.strip()
    if "-" not in s:
        return False
    cells = [c.strip() for c in s.strip("|").split("|")]
    return bool(cells) and all(
        re.fullmatch(r":?-{1,}:?", c or "") for c in cells
    )


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _render_table(block: list[str]) -> str:
    rows = [_split_row(ln) for ln in block]
    header = rows[0]
    body = rows[2:] if len(rows) > 2 else []
    grid = [header] + body
    ncol = max((len(r) for r in grid), default=0)
    for r in grid:
        r.extend([""] * (ncol - len(r)))
    widths = [
        max(len(grid[ri][ci]) for ri in range(len(grid)))
        for ci in range(ncol)
    ]

    def fmt(r: list[str]) -> str:
        return "  ".join(c.ljust(widths[ci]) for ci, c in enumerate(r))

    lines = [fmt(header), "  ".join("-" * widths[ci] for ci in range(ncol))]
    lines.extend(fmt(r) for r in body)
    return "<tt>" + "\n".join(_esc(ln) for ln in lines) + "</tt>"


def _display_math(expr: str) -> str:
    """Display-math fallback: Unicode math in a larger span (used by the
    single-string renderer and when matplotlib isn't available)."""
    return f'<span size="x-large">{_math_to_pango(expr.replace(chr(10), " "))}</span>'


def _scan_display_math(lines: list[str], i: int) -> tuple[str, int]:
    """Given that lines[i] opens a display-math block, return (expr, next_i)."""
    stripped = lines[i].strip()
    close = "$$" if stripped.startswith("$$") else "\\]"
    rest = stripped[2:]
    if close in rest:
        return rest[: rest.index(close)], i + 1
    parts = [rest]
    i += 1
    n = len(lines)
    while i < n and close not in lines[i]:
        parts.append(lines[i])
        i += 1
    if i < n:
        parts.append(lines[i][: lines[i].index(close)])
        i += 1
    return "\n".join(parts), i


def _opens_display_math(stripped: str) -> bool:
    return stripped.startswith("$$") or stripped.startswith("\\[")


def to_segments(md: str) -> list[tuple[str, str]]:
    """Split Markdown into renderable segments for the bubble:

    - ``("text", pango_markup)`` — a run of normal Markdown (incl. inline math)
    - ``("math", latex_expr)`` — a display-math block, for the caller to render
      as an image (matplotlib) or fall back to Unicode.

    Lets the UI interleave Gtk.Label (text) and Gtk.Picture (math image)
    widgets while keeping all parsing here.
    """
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[tuple[str, str]] = []
    text_buf: list[str] = []
    i, n = 0, len(lines)

    def flush() -> None:
        if text_buf:
            chunk = "\n".join(text_buf)
            if chunk.strip():
                segments.append(("text", to_pango_markup(chunk)))
            text_buf.clear()

    while i < n:
        if _opens_display_math(lines[i].strip()):
            expr, i = _scan_display_math(lines, i)
            flush()
            if expr.strip():
                segments.append(("math", expr.strip()))
            continue
        text_buf.append(lines[i])
        i += 1
    flush()
    return segments


def render_math_png(tex: str, color: str = "#000000",
                    fontsize: int = 16, dpi: int = 160) -> bytes | None:
    """Render a display-math LaTeX expression to PNG bytes via matplotlib's
    mathtext (no TeX install needed). Returns None if matplotlib is missing
    or the expression can't be parsed — the caller then falls back to the
    Unicode renderer.
    """
    try:
        import io
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib import mathtext
        from matplotlib.font_manager import FontProperties
    except Exception:
        return None
    expr = _normalize_latex(tex.strip().replace("\n", " "))
    if not expr:
        return None
    # mathtext wants a single $…$-wrapped math string.
    if not expr.startswith("$"):
        expr = f"${expr}$"
    buf = io.BytesIO()
    try:
        # Size is set via FontProperties — math_to_image has no fontsize arg.
        mathtext.math_to_image(
            expr, buf, prop=FontProperties(size=fontsize),
            dpi=dpi, format="png", color=color,
        )
    except Exception:
        return None
    return buf.getvalue()


def to_pango_markup(md: str) -> str:
    """Convert a Markdown string to Pango markup for a Gtk.Label."""
    lines = md.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    i, n = 0, len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # Display math: $$ … $$ or \[ … \] (possibly spanning lines).
        if _opens_display_math(stripped):
            expr, i = _scan_display_math(lines, i)
            out.append(_display_math(expr))
            continue

        # Fenced code block — syntax-highlighted (see highlight.py).
        if _FENCE.match(line):
            lang = line.strip().lstrip("`").strip()
            i += 1
            code: list[str] = []
            while i < n and not _FENCE.match(lines[i]):
                code.append(lines[i])
                i += 1
            i += 1  # consume closing fence (or run off the end harmlessly)
            body = "\n".join(code)
            try:
                from .highlight import highlight_pango
                out.append("<tt>" + highlight_pango(body, lang) + "</tt>")
            except Exception:  # noqa: BLE001 — colouring must never lose code
                out.append("<tt>" + _esc(body) + "</tt>")
            continue

        # Pipe table (needs a dashes separator on the next line).
        if "|" in line and i + 1 < n and _is_table_sep(lines[i + 1]):
            block: list[str] = []
            while i < n and "|" in lines[i]:
                block.append(lines[i])
                i += 1
            out.append(_render_table(block))
            continue

        m = _HEADING.match(line)
        if m:
            size = _HEADING_SIZE[len(m.group(1))]
            out.append(
                f'<span size="{size}" weight="bold">'
                f"{_inline(m.group(2).strip())}</span>"
            )
            i += 1
            continue

        if _HR.match(line):
            out.append('<span foreground="#888888">────────────────</span>')
            i += 1
            continue

        m = _BQ.match(line)
        if m:
            out.append(f"<i>▌ {_inline(m.group(1))}</i>")
            i += 1
            continue

        m = _ULIST.match(line)
        if m:
            pad = "    " * (len(m.group(1)) // 2)
            out.append(f"{pad}• {_inline(m.group(2))}")
            i += 1
            continue

        m = _OLIST.match(line)
        if m:
            pad = "    " * (len(m.group(1)) // 2)
            out.append(f"{pad}{m.group(2)}. {_inline(m.group(3))}")
            i += 1
            continue

        if line.strip() == "":
            out.append("")
            i += 1
            continue

        out.append(_inline(line))
        i += 1

    return "\n".join(out)
