"""TTS text-cleaning tests.

The voice should never pronounce LaTeX raw — `$$`, `\\oint`, `\\mathbf` etc.
would otherwise come out as "dollar dollar slash slash oint slash slash
mathbf". These tests pin the stripping rules in _clean_for_speech.
"""
from __future__ import annotations

import unittest

from box_chat.tts import _clean_for_speech


class StripLatexTests(unittest.TestCase):
    def test_display_math_block_dropped(self) -> None:
        out = _clean_for_speech(
            r"Gauss's Law: $$\oint_{\partial S} \mathbf{E} \cdot d\mathbf{A} "
            r"= \frac{Q_{\text{enc}}}{\epsilon_0}$$ is the integral form."
        )
        self.assertNotIn("$", out)
        self.assertNotIn("\\", out)
        self.assertNotIn("oint", out)
        self.assertNotIn("mathbf", out)
        self.assertIn("Gauss", out)
        self.assertIn("integral form", out)

    def test_inline_math_dropped(self) -> None:
        out = _clean_for_speech(
            r"For example $\nabla \cdot \mathbf{E} = \rho / \epsilon_0$ here."
        )
        self.assertNotIn("$", out)
        self.assertNotIn("nabla", out)
        self.assertNotIn("\\", out)
        self.assertIn("For example", out)
        self.assertIn("here", out)

    def test_doubled_backslash_in_block(self) -> None:
        # Models often double-escape inside markdown. Both forms must be
        # silenced — the stripper runs after pattern collapse implicitly.
        out = _clean_for_speech("Equation: $$\\\\frac{a}{b}$$ rest.")
        self.assertNotIn("$", out)
        self.assertNotIn("\\", out)
        self.assertNotIn("frac", out)
        self.assertIn("Equation:", out)
        self.assertIn("rest.", out)

    def test_bracket_display_math_dropped(self) -> None:
        out = _clean_for_speech(r"See \[ y = mx + b \] for the line.")
        self.assertNotIn("\\", out)
        self.assertNotIn("=", out)  # belongs to the equation we stripped
        self.assertIn("See", out)
        self.assertIn("for the line.", out)

    def test_stray_latex_command_dropped(self) -> None:
        # Model emits raw LaTeX without delimiters.
        out = _clean_for_speech(r"Use \mathbf{E} for the field and \alpha here.")
        self.assertNotIn("mathbf", out)
        self.assertNotIn("alpha", out)
        self.assertNotIn("\\", out)
        self.assertIn("Use", out)
        self.assertIn("field", out)

    def test_plain_prose_unchanged(self) -> None:
        out = _clean_for_speech("Plain text with **bold** and `code`.")
        self.assertEqual(out, "Plain text with bold and code.")


if __name__ == "__main__":
    unittest.main()
