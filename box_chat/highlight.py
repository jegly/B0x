"""Syntax highlighting for fenced code blocks — pure stdlib → Pango markup.

GtkSourceView isn't available on the target systems, so this is a small
regex tokenizer: strings, comments, numbers, keywords, decorators and
function calls get colour spans; everything else is escaped verbatim.
Good-enough Claude Code-style colouring for the languages a local coding
agent actually meets, with a graceful generic fallback for the rest.

The palette is mid-saturation so it reads on both dark and light themes.
Entry point: :func:`highlight_pango` (used by mdrender's fence renderer).
"""
from __future__ import annotations

import re
from html import escape as _esc

__all__ = ["highlight_pango"]

# One Palette, believable on dark and light backgrounds.
_C_KEYWORD = "#c678dd"   # purple
_C_STRING = "#7fb069"    # green
_C_COMMENT = "#8a8f98"   # grey (italic)
_C_NUMBER = "#d19a66"    # orange
_C_FUNC = "#61afef"      # blue
_C_TYPE = "#e5c07b"      # yellow
_C_DECOR = "#e06c75"     # red-ish (decorators, preprocessor, selectors)

_PY_KW = (
    "False None True and as assert async await break class continue def del "
    "elif else except finally for from global if import in is lambda nonlocal "
    "not or pass raise return try while with yield match case self cls"
)
_JS_KW = (
    "abstract arguments async await break case catch class const continue "
    "debugger default delete do else enum export extends false finally for "
    "from function get if implements import in instanceof interface let new "
    "null of private protected public return set static super switch this "
    "throw true try type typeof undefined var void while with yield"
)
_SH_KW = (
    "if then else elif fi for while until do done case esac function in "
    "select time coproc break continue return exit export local readonly "
    "declare unset shift trap set source alias true false"
)
_C_KW = (
    "auto break case char const continue default do double else enum extern "
    "float for goto if inline int long register restrict return short signed "
    "sizeof static struct switch typedef union unsigned void volatile while "
    "bool class catch delete explicit friend mutable namespace new operator "
    "private protected public template this throw try typename using virtual "
    "nullptr constexpr override final noexcept"
)
_GO_KW = (
    "break case chan const continue default defer else fallthrough for func "
    "go goto if import interface map package range return select struct "
    "switch type var nil true false iota make new len cap append"
)
_RS_KW = (
    "as async await break const continue crate dyn else enum extern false fn "
    "for if impl in let loop match mod move mut pub ref return self Self "
    "static struct super trait true type unsafe use where while"
)
_SQL_KW = (
    "select from where insert into values update delete set create table "
    "drop alter index view join left right inner outer on as and or not "
    "null primary key foreign references group by order having limit offset "
    "distinct union all exists between like in is asc desc"
)


def _kwset(s: str) -> frozenset[str]:
    return frozenset(s.split())


class _Lang:
    def __init__(
        self,
        keywords: frozenset[str],
        line_comment: str | None = "#",
        block_comment: tuple[str, str] | None = None,
        triple_strings: bool = False,
        case_insensitive: bool = False,
    ) -> None:
        self.keywords = keywords
        self.line_comment = line_comment
        self.block_comment = block_comment
        self.triple_strings = triple_strings
        self.case_insensitive = case_insensitive


_LANGS: dict[str, _Lang] = {
    "python": _Lang(_kwset(_PY_KW), "#", None, triple_strings=True),
    "javascript": _Lang(_kwset(_JS_KW), "//", ("/*", "*/")),
    "bash": _Lang(_kwset(_SH_KW), "#"),
    "c": _Lang(_kwset(_C_KW), "//", ("/*", "*/")),
    "go": _Lang(_kwset(_GO_KW), "//", ("/*", "*/")),
    "rust": _Lang(_kwset(_RS_KW), "//", ("/*", "*/")),
    "sql": _Lang(_kwset(_SQL_KW), "--", ("/*", "*/"), case_insensitive=True),
    "json": _Lang(frozenset({"true", "false", "null"})),
    "yaml": _Lang(frozenset({"true", "false", "null", "yes", "no"}), "#"),
    "toml": _Lang(frozenset({"true", "false"}), "#"),
    "css": _Lang(frozenset(), None, ("/*", "*/")),
    "generic": _Lang(frozenset(), None),
}

_ALIASES = {
    "py": "python", "python3": "python",
    "js": "javascript", "jsx": "javascript", "ts": "javascript",
    "tsx": "javascript", "typescript": "javascript", "java": "javascript",
    "kotlin": "javascript", "kt": "javascript", "swift": "javascript",
    "sh": "bash", "shell": "bash", "zsh": "bash", "console": "bash",
    "cpp": "c", "c++": "c", "h": "c", "hpp": "c", "cc": "c",
    "rs": "rust", "golang": "go",
    "yml": "yaml", "jsonc": "json",
    "scss": "css", "less": "css",
}


_RX_CACHE: dict[str, re.Pattern] = {}


def _resolve(lang: str) -> tuple[str, _Lang]:
    key = (lang or "").strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _LANGS:
        key = "generic"
    return key, _LANGS[key]


def _span(text: str, color: str, italic: bool = False) -> str:
    style = ' style="italic"' if italic else ""
    return f'<span foreground="{color}"{style}>{_esc(text)}</span>'


def _build_regex(lang: _Lang) -> re.Pattern:
    parts: list[str] = []
    if lang.triple_strings:
        parts.append(r'(?P<tstr>"""(?:\\.|[^\\])*?"""|\'\'\'(?:\\.|[^\\])*?\'\'\')')
    parts.append(r'(?P<str>"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\')')
    if lang.block_comment:
        o, c = (re.escape(x) for x in lang.block_comment)
        parts.append(rf"(?P<bcom>{o}.*?{c})")
    if lang.line_comment:
        lc = re.escape(lang.line_comment)
        parts.append(rf"(?P<lcom>{lc}[^\n]*)")
    parts.append(r"(?P<dec>^[ \t]*@[\w.]+|^[ \t]*#(?:include|define|ifdef|ifndef|endif|pragma)\b[^\n]*)")
    parts.append(r"(?P<num>\b(?:0[xXbBoO][0-9a-fA-F_]+|\d[\d_]*(?:\.\d+)?(?:[eE][+-]?\d+)?)\b)")
    parts.append(r"(?P<word>[A-Za-z_][A-Za-z0-9_]*)")
    flags = re.DOTALL | re.MULTILINE
    if lang.case_insensitive:
        flags |= re.IGNORECASE
    return re.compile("|".join(parts), flags)


def highlight_pango(code: str, lang: str = "") -> str:
    """Return ``code`` as Pango markup with syntax colouring.

    Always escapes everything; on any internal failure the caller should
    fall back to a plain escaped block (mdrender does).
    """
    key, spec = _resolve(lang)
    rx = _RX_CACHE.get(key)
    if rx is None:
        rx = _RX_CACHE[key] = _build_regex(spec)
    kw = spec.keywords
    lower = spec.case_insensitive
    out: list[str] = []
    pos = 0
    for m in rx.finditer(code):
        if m.start() > pos:
            out.append(_esc(code[pos:m.start()]))
        pos = m.end()
        text = m.group(0)
        kind = m.lastgroup
        if kind in ("tstr", "str"):
            out.append(_span(text, _C_STRING))
        elif kind in ("bcom", "lcom"):
            out.append(_span(text, _C_COMMENT, italic=True))
        elif kind == "dec":
            out.append(_span(text, _C_DECOR))
        elif kind == "num":
            out.append(_span(text, _C_NUMBER))
        elif kind == "word":
            check = text.lower() if lower else text
            if check in kw:
                out.append(_span(text, _C_KEYWORD))
            elif code[m.end():m.end() + 1] == "(":
                out.append(_span(text, _C_FUNC))
            elif text[:1].isupper() and any(c.islower() for c in text):
                out.append(_span(text, _C_TYPE))
            else:
                out.append(_esc(text))
        else:
            out.append(_esc(text))
    out.append(_esc(code[pos:]))
    return "".join(out)
