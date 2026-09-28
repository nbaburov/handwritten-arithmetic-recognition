"""Single source of truth for all colour constants used in the GUI.

Role colours (#RRGGBB hex) satisfy three simultaneous surfaces:
  - Box stroke on white image background (WCAG AA large, >= 3:1 vs white)
  - Text accent on Gradio dark card (~#1f2937, >= 4.5:1)
  - Gradio AnnotatedImage legend tint (~53% alpha over white): dark label text >= 4.5:1

GT overlay and semantic colours are hardcoded by meaning, not theme.

Tuning notes (all values pass all three constraints above):
  digit_main:     #58a6ff -> #3996ff  (stroke 3.01, dark_text 4.88)
  digit_carry:    #f78166 -> #f56949  (stroke 3.00, dark_text 4.89)
  digit_borrow:   #ffa657 -> #ed7000  (stroke 3.04, dark_text 4.84)
  operator:       #d2a8ff -> #b672ff  (stroke 3.07, dark_text 4.78)
  result_bar:     #3fb950 -> #39a849  (stroke 3.06, dark_text 4.80)
  divide_bracket: #79c0ff -> #2297ff  (stroke 3.03, dark_text 4.85)
  low_conf:       #f0883e -> #ed7118  (stroke 3.01, dark_text 4.88)
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Role palette -- box stroke + legend tint + text accent (single unified set)
# ---------------------------------------------------------------------------

ROLE_COLOURS: dict[str, str] = {
    "digit_main":     "#3996ff",
    "digit_carry":    "#f56949",
    "digit_borrow":   "#ed7000",
    "operator":       "#b672ff",
    "result_bar":     "#39a849",
    "divide_bracket": "#2297ff",
}

LOW_CONF_COLOUR: str = "#ed7118"
LOW_CONF_THRESHOLD: float = 0.5

# Cycling palette for row cluster IDs (row 0=blue, 1=orange-red, 2=green, ...)
ROW_PALETTE: list[str] = [
    "#3996ff", "#f56949", "#56d364", "#b672ff",
    "#ed7000", "#2297ff", "#e3b341", "#ff7b72",
]

# ---------------------------------------------------------------------------
# GT overlay semantic colours (hardcoded by meaning)
# ---------------------------------------------------------------------------

GT_CORRECT: str = "#3fb950"    # green checkmark
GT_WRONG: str = "#f85149"      # red X
GT_MISSED: str = "#ffa657"     # amber missing
GT_EXTRA: str = "#f0883e"      # orange false positive

# ---------------------------------------------------------------------------
# Score / status semantic colours (hardcoded; work in both themes)
# ---------------------------------------------------------------------------

SCORE_OK: str = "#22c55e"
SCORE_ERR: str = "#ef4444"
STATUS_OK: str = "#22c55e"
STATUS_WARN: str = "#f59e0b"
STATUS_ERR: str = "#ef4444"

# ---------------------------------------------------------------------------
# Showcase hero buttons (semantic CTA -- intentionally hardcoded)
# ---------------------------------------------------------------------------

SHOWCASE_HERO_YES_BG: str = "#16a34a"
SHOWCASE_HERO_NO_BG: str = "#dc2626"
SHOWCASE_HERO_FG: str = "#ffffff"

# ---------------------------------------------------------------------------
# Glyph display map -- SSoT for how each fine label renders as a visible symbol
# Shared by Stage 2 panel and Result panel.
# ---------------------------------------------------------------------------

# Maps fine label -> (display_glyph, human_name)
# display_glyph is the character/string rendered in the panel
# human_name is used in hover tooltip (title=)
GLYPH_MAP: dict[str, tuple[str, str]] = {
    # Digits
    "main_0": ("0", "digit 0"),
    "main_1": ("1", "digit 1"),
    "main_2": ("2", "digit 2"),
    "main_3": ("3", "digit 3"),
    "main_4": ("4", "digit 4"),
    "main_5": ("5", "digit 5"),
    "main_6": ("6", "digit 6"),
    "main_7": ("7", "digit 7"),
    "main_8": ("8", "digit 8"),
    "main_9": ("9", "digit 9"),
    # Operators
    "op_plus":   ("+",  "plus"),
    "op_minus":  ("−", "minus"),
    "op_times":  ("×", "times"),
    "op_divide": ("÷", "divide"),
    # Structural tokens
    "result_bar":  ("─────", "result bar"),
    # div_bracket uses a CSS-drawn bracket shape (engine-chip--bracket modifier).
    # The glyph string is intentionally empty; the visual is pure CSS border.
    "div_bracket": ("",  "division bracket"),
}

# Readable coarse class names for Stage 1 panel (spaces, not underscores)
COARSE_DISPLAY_NAMES: dict[str, str] = {
    "digit_main":     "digit",
    "digit_carry":    "carry",
    "digit_borrow":   "borrow",
    "operator":       "operator",
    "result_bar":     "result bar",
    "divide_bracket": "div bracket",
}


def display_glyph(fine_label: str) -> str:
    """Return the display glyph for a fine label (digit/operator/structural)."""
    return GLYPH_MAP.get(fine_label, (fine_label, fine_label))[0]


def display_name(fine_label: str) -> str:
    """Return the human-readable name for a fine label (for tooltips)."""
    return GLYPH_MAP.get(fine_label, (fine_label, fine_label))[1]


# ---------------------------------------------------------------------------
# Equation-kind display names -- full words for the Result panel.
# The assembler emits short kinds (add/subtract/...); show them in full.
# ---------------------------------------------------------------------------

EQ_KIND_DISPLAY: dict[str, str] = {
    "add": "Addition",            "addition": "Addition",
    "subtract": "Subtraction",    "subtraction": "Subtraction",
    "multiply": "Multiplication", "multiplication": "Multiplication",
    "divide": "Division",         "division": "Division",
    "bare_digits": "Bare digits (no equation)",
    "unknown": "Unknown (out of distribution)",
}


def equation_kind_display(kind: str) -> str:
    """Full human label for an equation kind (add -> 'Addition')."""
    return EQ_KIND_DISPLAY.get(kind, kind.replace("_", " ").capitalize())
