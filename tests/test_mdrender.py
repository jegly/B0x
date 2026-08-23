"""Markdown + LaTeX-subset rendering to Pango markup.

The hard guarantee is that every output of to_pango_markup parses as valid
Pango markup — Pango rejects the *entire* string on one malformed tag, which
would blank the bubble. We validate with Pango.parse_markup.
"""
from __future__ import annotations

import unittest

import gi

gi.require_version("Pango", "1.0")
from gi.repository import Pango  # noqa: E402

from box_chat.mdrender import to_pango_markup, to_segments


def _valid(markup: str) -> bool:
    try:
        Pango.parse_markup(markup, -1, "\x00")
        return True
    except Exception:
        return False


class MarkupValidityTests(unittest.TestCase):
    SAMPLES = [
        "# Heading\n\n**bold** _italic_ `code`",
        "- a\n- b\n1. one\n2. two",
        "| A | B |\n| - | - |\n| 1 | 2 |",
        "```python\nx = 1 < 2 & 3 > 0\n```",
        "raw < > & chars and <fake>tags</fake>",
        "inline $E = mc^2$ and $\\alpha \\leq \\beta$",
        "$$\\nabla \\cdot \\vec{E} = \\frac{\\rho}{\\epsilon_0}$$",
        "***bolditalic*** ~~strike~~ [link](https://x.test)",
        "",
        "unbalanced ** asterisks and _ underscores",
    ]

    def test_all_samples_produce_valid_markup(self) -> None:
        for s in self.SAMPLES:
            with self.subTest(sample=s[:30]):
                self.assertTrue(_valid(to_pango_markup(s)),
                                f"invalid markup for: {s!r}")


class ConversionTests(unittest.TestCase):
    def test_bold_italic_code(self) -> None:
        out = to_pango_markup("**b** *i* `c`")
        self.assertIn("<b>b</b>", out)
        self.assertIn("<i>i</i>", out)
        self.assertIn("<tt>c</tt>", out)

    def test_heading(self) -> None:
        self.assertIn('weight="bold"', to_pango_markup("## Title"))

    def test_html_escaped(self) -> None:
        out = to_pango_markup("a < b & c")
        self.assertIn("&lt;", out)
        self.assertIn("&amp;", out)
        self.assertNotIn("<b", out)  # no stray real tags from the text

    def test_inline_math_superscript_and_symbols(self) -> None:
        out = to_pango_markup("$E = mc^2$, $\\alpha \\leq \\beta$")
        self.assertIn("<sup>2</sup>", out)
        self.assertIn("α", out)
        self.assertIn("≤", out)
        self.assertIn("β", out)

    def test_code_block_not_markdown_parsed(self) -> None:
        out = to_pango_markup("```\n**not bold** $x$\n```")
        # Inside a code block, markdown/math must stay literal.
        self.assertIn("**not bold**", out)
        self.assertNotIn("<b>not bold</b>", out)

    def test_table_renders_monospace(self) -> None:
        out = to_pango_markup("| A | B |\n| - | - |\n| 11 | 2 |")
        self.assertIn("<tt>", out)
        self.assertIn("A", out)
        self.assertIn("11", out)


class LatexEdgeCaseTests(unittest.TestCase):
    """Regression: LLM-emitted LaTeX with doubled backslashes or nested
    braces used to leave 'frac', 'mathbf', stray '\\' in the output."""

    def test_doubled_backslash_normalises(self) -> None:
        # `$$\\\\frac{a}{b}$$` is what models emit when they double-escape.
        from box_chat.mdrender import _math_to_pango
        doubled = "\\\\frac{a}{b}"
        out = _math_to_pango(doubled)
        self.assertIn("(a)/(b)", out)
        self.assertNotIn("frac", out)
        self.assertNotIn("\\", out)

    def test_frac_with_nested_braces(self) -> None:
        from box_chat.mdrender import _math_to_pango
        out = _math_to_pango(r"\frac{Q_{\text{enc}}}{\epsilon_0}")
        self.assertNotIn("frac", out)
        self.assertIn("(Q<sub>enc</sub>)/(ε<sub>0</sub>)", out)

    def test_mathbf_unwrapped(self) -> None:
        from box_chat.mdrender import _math_to_pango
        out = _math_to_pango(r"\mathbf{E} \cdot d\mathbf{A}")
        self.assertIn("<b>E</b>", out)
        self.assertIn("<b>A</b>", out)
        self.assertNotIn("mathbf", out)

    def test_boxed_unwrapped(self) -> None:
        from box_chat.mdrender import _math_to_pango
        out = _math_to_pango(r"\boxed{E = mc^2}")
        self.assertNotIn("boxed", out)
        self.assertIn("E = mc<sup>2</sup>", out)

    def test_no_stray_backslashes_in_pango(self) -> None:
        from box_chat.mdrender import _math_to_pango
        for c in [
            r"\oint_{\partial S} \mathbf{E} \cdot d\mathbf{A}",
            "\\\\oint_{\\\\partial S} \\\\mathbf{E}",
            r"\sum_{i=1}^n \frac{1}{i^2}",
        ]:
            out = _math_to_pango(c)
            self.assertNotIn("\\", out, f"backslash leaked from {c!r} → {out!r}")
            self.assertTrue(_valid(out), f"Pango rejected: {out!r}")


class SegmentTests(unittest.TestCase):
    def test_display_math_split_out(self) -> None:
        segs = to_segments("before\n$$x^2$$\nafter")
        kinds = [k for k, _ in segs]
        self.assertEqual(kinds, ["text", "math", "text"])
        math = next(c for k, c in segs if k == "math")
        self.assertEqual(math, "x^2")

    def test_no_display_math_is_single_text_segment(self) -> None:
        segs = to_segments("just **text** here")
        self.assertEqual([k for k, _ in segs], ["text"])

    def test_bracket_display_math(self) -> None:
        segs = to_segments("a\n\\[ y = x \\]\nb")
        self.assertIn("math", [k for k, _ in segs])


if __name__ == "__main__":
    unittest.main()
