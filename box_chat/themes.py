"""Themes — Catppuccin (Mocha + Latte) and Dracula.

Each theme is a dict of CSS variable overrides that get spliced into a CSS
template at load time. We override libadwaita's named colors so EVERYTHING
(headerbar, sidebar, dialogs, toasts, popovers, AlertDialog, …) follows the
palette — not just our chat bubbles.

Palette references:
  Catppuccin: https://github.com/catppuccin/catppuccin
  Dracula:    https://draculatheme.com/contribute
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    id: str            # internal key, also persisted in settings
    name: str          # display name
    is_dark: bool      # tells Adw whether to use dark icons etc.

    # ── core palette ──
    base:      str     # main background
    mantle:    str     # slightly lower (sidebar / cards)
    crust:     str     # deepest (headerbar)
    surface:   str     # raised surface (input fields, popovers)
    overlay:   str     # borders / dividers
    chat_bg:   str     # chat panel + composer background (neutral grey for dark themes)
    text:      str     # default foreground
    subtext:   str     # dim labels / captions
    accent:    str     # primary accent (send button, selection)
    accent_fg: str     # text on accent
    success:   str
    warning:   str
    error:     str

    # ── chat bubble colors ──
    user_bubble:        str
    user_bubble_border: str
    asst_bubble:        str
    asst_bubble_border: str


# Catppuccin Mocha (dark) — the default.
CATPPUCCIN_MOCHA = Theme(
    id="catppuccin-mocha",
    name="Catppuccin Mocha",
    is_dark=True,
    base="#1e1e2e",       # base
    mantle="#181825",     # mantle
    crust="#11111b",      # crust
    surface="#313244",    # surface0
    overlay="#45475a",    # surface1
    chat_bg="#181825",    # same as mantle — chat matches theme
    text="#cdd6f4",       # text
    subtext="#a6adc8",    # subtext0
    accent="#fab387",     # peach — warm "mocha/coffee" feel, distinct from Dracula purples
    accent_fg="#11111b",
    success="#a6e3a1",    # green
    warning="#f9e2af",    # yellow
    error="#f38ba8",      # red
    user_bubble="#313244",
    user_bubble_border="#fab387",
    asst_bubble="#181825",
    asst_bubble_border="#45475a",
)

# Catppuccin Latte (light).
CATPPUCCIN_LATTE = Theme(
    id="catppuccin-latte",
    name="Catppuccin Latte",
    is_dark=False,
    base="#eff1f5",
    mantle="#e6e9ef",
    crust="#dce0e8",
    surface="#ccd0da",
    overlay="#bcc0cc",
    chat_bg="#e6e9ef",    # same as mantle — light theme is fine as-is
    text="#4c4f69",
    subtext="#6c6f85",
    accent="#8839ef",     # mauve
    accent_fg="#eff1f5",
    success="#40a02b",
    warning="#df8e1d",
    error="#d20f39",
    user_bubble="#ccd0da",
    user_bubble_border="#8839ef",
    asst_bubble="#e6e9ef",
    asst_bubble_border="#bcc0cc",
)

# Catppuccin Frappé (medium-dark) — softer than Mocha, harmonious tinted darks.
CATPPUCCIN_FRAPPE = Theme(
    id="catppuccin-frappe",
    name="Catppuccin Frappé",
    is_dark=True,
    base="#303446",
    mantle="#292c3c",
    crust="#232634",
    surface="#414559",    # surface0
    overlay="#51576d",    # surface1
    chat_bg="#292c3c",    # tinted mantle — harmonious
    text="#c6d0f5",
    subtext="#a5adce",
    accent="#ca9ee6",     # mauve
    accent_fg="#232634",
    success="#a6d189",
    warning="#e5c890",
    error="#e78284",
    user_bubble="#414559",
    user_bubble_border="#ca9ee6",
    asst_bubble="#292c3c",
    asst_bubble_border="#51576d",
)

# Catppuccin Macchiato (dark) — between Frappé and Mocha.
CATPPUCCIN_MACCHIATO = Theme(
    id="catppuccin-macchiato",
    name="Catppuccin Macchiato",
    is_dark=True,
    base="#24273a",
    mantle="#1e2030",
    crust="#181926",
    surface="#363a4f",    # surface0
    overlay="#494d64",    # surface1
    chat_bg="#1e2030",    # tinted mantle — harmonious
    text="#cad3f5",
    subtext="#a5adcb",
    accent="#c6a0f6",     # mauve
    accent_fg="#181926",
    success="#a6da95",
    warning="#eed49f",
    error="#ed8796",
    user_bubble="#363a4f",
    user_bubble_border="#c6a0f6",
    asst_bubble="#1e2030",
    asst_bubble_border="#494d64",
)

# Dracula.
DRACULA = Theme(
    id="dracula",
    name="Dracula",
    is_dark=True,
    base="#282a36",
    mantle="#21222c",
    crust="#191a21",
    surface="#44475a",
    overlay="#6272a4",
    chat_bg="#21222c",    # same as mantle — chat matches theme
    text="#f8f8f2",
    subtext="#bd93f9",
    accent="#ff79c6",     # pink — iconic Dracula, distinct from Mocha/Dracula Pro
    accent_fg="#282a36",
    success="#50fa7b",
    warning="#f1fa8c",
    error="#ff5555",
    user_bubble="#44475a",
    user_bubble_border="#ff79c6",
    asst_bubble="#21222c",
    asst_bubble_border="#6272a4",
)

# Dracula Pro — saturated pastels on cool near-black (Van Helsing variant).
DRACULA_PRO = Theme(
    id="dracula-pro",
    name="Dracula Pro",
    is_dark=True,
    base="#22212c",
    mantle="#1d1c26",
    crust="#15141b",
    surface="#454158",    # selection
    overlay="#7970a9",    # comment
    chat_bg="#1d1c26",    # tinted mantle — harmonious
    text="#f8f8f2",
    subtext="#bcbac3",
    accent="#80ffea",     # cyan — modern Dracula Pro feel, distinct from purples/pinks
    accent_fg="#22212c",
    success="#8aff80",
    warning="#ffff80",
    error="#ff9580",
    user_bubble="#454158",
    user_bubble_border="#80ffea",
    asst_bubble="#1d1c26",
    asst_bubble_border="#454158",
)


# ──────────────────────────────────────────────────────────────────────────
# Ptyxis terminal palettes.
#
# Ported from GNOME Ptyxis's bundled .palette resources (themselves from
# Gogh, https://github.com/Gogh-Co/Gogh), matching the Android app's
# PtyxisPalette / PtyxisPaletteExtended enums.  Rather than hand-write 44
# `Theme` entries, we keep a compact table of each palette's semantic colours
# (all copied verbatim from the palettes — nothing invented) and derive the
# libadwaita surface ramp the same way the Android side does: lerp the
# background toward black for the darker crust/mantle, toward the foreground
# for the raised surface/overlay.  Every Ptyxis palette is a dark scheme.
#
# Semantic colour sourcing:
#   • base (Ptyxis.kt)  — accent=primary, success=secondary, warning=tertiary,
#     error=error, straight from each entry's signature accent set.
#   • extended (PtyxisPalettesExtended.kt) — accent=ANSI blue (slot 4),
#     success=ANSI green (slot 2), warning=ANSI yellow (slot 3),
#     error=ANSI red (slot 1): the standard terminal colour convention every
#     one of these palettes was authored against.
#
# Known cosmetic quirk (flagged, no clean fix): a handful of the curated base
# palettes have no dedicated success/warning slot, so those semantic colours
# fall back to the palette's secondary/tertiary accents and can read
# "swapped" (e.g. Homebrew Ocean's warning is a green-ish tertiary).  This
# only tints toasts/badges; the palette's own colours are still faithful.


def _lerp_hex(a: str, b: str, t: float) -> str:
    """Linear-interpolate two #RRGGBB colours; t=0 → a, t=1 → b."""
    ah, bh = a.lstrip("#"), b.lstrip("#")
    ar, ag, ab = int(ah[0:2], 16), int(ah[2:4], 16), int(ah[4:6], 16)
    br, bg, bb = int(bh[0:2], 16), int(bh[2:4], 16), int(bh[4:6], 16)
    r = round(ar + (br - ar) * t)
    g = round(ag + (bg - ag) * t)
    bl = round(ab + (bb - ab) * t)
    return f"#{r:02X}{g:02X}{bl:02X}"


def _luminance(hex_color: str) -> float:
    """Relative luminance (0..1) of a #RRGGBB colour."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255


def _on_color(hex_color: str) -> str:
    """Legible foreground for a fill: near-white on dark, near-black on light."""
    return "#0A0A0A" if _luminance(hex_color) > 0.5 else "#F5F5F5"


_BLACK = "#000000"

# (key, name, bg, fg, accent, success, warning, error) — colours verbatim from
# the .kt palette enums; see parser provenance in the block comment above.
_PTYXIS_BASE: list[tuple[str, ...]] = [
    ('fairy_floss', 'Fairy Floss', '#5A5475', '#C2FFDF', '#FFB8D1', '#AE81FF', '#C2FFDF', '#FF857F'),
    ('nord', 'Nord', '#2E3440', '#D8DEE9', '#88C0D0', '#81A1C1', '#8FBCBB', '#BF616A'),
    ('bim', 'Bim', '#012849', '#A9BED8', '#5EA2EC', '#F557A0', '#A9EE55', '#F557A0'),
    ('borland', 'Borland', '#0000A4', '#FFFF4E', '#FFFF4E', '#FF73FD', '#96CBFE', '#FF6C60'),
    ('c64', 'C64', '#40318D', '#7869C4', '#67B6BD', '#BFCE72', '#8B3F96', '#883932'),
    ('cobalt_neon', 'Cobalt Neon', '#142838', '#8FF586', '#8FF586', '#3BA5FF', '#E9E75C', '#FF2320'),
    ('grass', 'Grass', '#13773D', '#FFF0A5', '#E7B000', '#00BBBB', '#FFF0A5', '#BB0000'),
    ('homebrew_ocean', 'Homebrew Ocean', '#224FBC', '#FFFFFF', '#00A6B2', '#00A600', '#999900', '#990000'),
    ('mono_amber', 'Mono Amber', '#2B1900', '#FF9400', '#FF9400', '#FF9400', '#FF9400', '#FF9400'),
    ('mono_red', 'Mono Red', '#2B0C00', '#FF3600', '#FF3600', '#FF3600', '#FF3600', '#FF3600'),
    ('synthwave', 'Synthwave', '#262335', '#FFFFFF', '#FF7EDB', '#03EDF9', '#FEDE5D', '#FE4450'),
]

# (key, name, bg, fg, accent[ANSI blue], success[ANSI green],
#  warning[ANSI yellow], error[ANSI red]) — verbatim from the extended enum.
_PTYXIS_EXTENDED: list[tuple[str, ...]] = [
    ('aci', 'Aci', '#0D1926', '#B4E1FD', '#0883FF', '#83FF08', '#FF8308', '#FF0883'),
    ('afterglow', 'Afterglow', '#222222', '#D0D0D0', '#6C99BB', '#7B9246', '#D3A04D', '#A53C23'),
    ('argonaut', 'Argonaut', '#0E1019', '#FFFAF4', '#008DF8', '#8CE10B', '#FFB900', '#FF000F'),
    ('aura', 'Aura', '#15141B', '#EDECEE', '#A277FF', '#61FFCA', '#FFCA85', '#FF6767'),
    ('ayu_mirage', 'Ayu Mirage', '#1F2430', '#CBCCC6', '#73D0FF', '#BAE67E', '#FFA759', '#FF3333'),
    ('belafonte', 'Belafonte', '#20111B', '#968C83', '#426A79', '#858162', '#EAA549', '#BE100E'),
    ('birds_of_paradise', 'Birds Of Paradise', '#2A1F1D', '#E0DBB7', '#5A86AD', '#6BA18A', '#E99D2A', '#BE2D26'),
    ('blazer', 'Blazer', '#0D1926', '#D9E6F2', '#7A7AB8', '#7AB87A', '#B8B87A', '#B87A7A'),
    ('brogrammer', 'Brogrammer', '#131313', '#D6DBE5', '#2A84D2', '#2DC55E', '#ECBA0F', '#F81118'),
    ('chalkboard', 'Chalkboard', '#29262F', '#D9E6F2', '#7372C3', '#72C373', '#C2C372', '#C37372'),
    ('espresso_libre', 'Espresso Libre', '#2A211C', '#B8A898', '#0066FF', '#1A921C', '#F0E53A', '#CC0000'),
    ('everforest', 'Everforest', '#2D353B', '#D3C6AA', '#7FBBB3', '#A7C080', '#DBBC7F', '#E67E80'),
    ('flatland', 'Flatland', '#1D1F21', '#B8DBEF', '#5096BE', '#9FD364', '#F4EF6D', '#F18339'),
    ('github', 'GitHub', '#101216', '#8B949E', '#6CA4F8', '#56D364', '#E3B341', '#F78166'),
    ('ibm3270', 'IBM3270', '#000000', '#FDFDFD', '#7890F0', '#24D830', '#F0D824', '#F01818'),
    ('ic_green_ppl', 'IC Green PPL', '#3A3D3F', '#D9EFD3', '#149B45', '#339C24', '#659B25', '#FB002A'),
    ('kanagawa', 'Kanagawa', '#1F1F28', '#DCD7BA', '#7E9CD8', '#76946A', '#C0A36E', '#C34043'),
    ('material', 'Material', '#1E282C', '#C3C7D1', '#80CBC3', '#C3E88D', '#F7EB95', '#EB606B'),
    ('mona_lisa', 'Mona Lisa', '#120B0D', '#F7D66A', '#515C5D', '#636232', '#C36E28', '#9B291C'),
    ('mono_cyan', 'Mono Cyan', '#00222B', '#00CCFF', '#00CCFF', '#00CCFF', '#00CCFF', '#00CCFF'),
    ('monokai_pro', 'Monokai Pro', '#363537', '#FDF9F3', '#FC9867', '#A9DC76', '#FFD866', '#FF6188'),
    ('omni', 'Omni', '#191622', '#ABB2BF', '#78D1E1', '#67E480', '#E89E64', '#E96379'),
    ('paraiso_dark', 'Paraiso Dark', '#2F1E2E', '#A39E9B', '#06B6EF', '#48B685', '#FEC418', '#EF6155'),
    ('pixiefloss', 'Pixiefloss', '#241F33', '#D1CAE8', '#AE81FF', '#48B685', '#E6C000', '#FF857F'),
    ('powershell', 'Powershell', '#052454', '#F6F6F7', '#010083', '#098003', '#C4A000', '#7E0008'),
    ('relaxed', 'Relaxed', '#353A44', '#D9D9D9', '#6A8799', '#909D63', '#EBC17A', '#BC5653'),
    ('sea_shells', 'Sea Shells', '#09141B', '#DEB88D', '#1E4950', '#027C9B', '#FCA02F', '#D15123'),
    ('solarized', 'Solarized', '#002B36', '#839496', '#268BD2', '#859900', '#CF9A6B', '#DC322F'),
    ('spacedust', 'Spacedust', '#0A1E24', '#ECF0C1', '#0F548B', '#5CAB96', '#E3CD7B', '#E35B00'),
    ('spring', 'Spring', '#0A1E24', '#ECF0C1', '#1DD3EE', '#1F8C3B', '#1FC95B', '#FF4D83'),
    ('twilight', 'Twilight', '#141414', '#FFFFD4', '#44474A', '#AFB97A', '#C2A86C', '#C06D44'),
    ('urple', 'Urple', '#1B1B23', '#877A9B', '#564D9B', '#37A415', '#AD5C42', '#B0425B'),
    ('xterm', 'Xterm', '#000000', '#FFFFFF', '#0000EE', '#00CD00', '#CDCD00', '#CD0000'),
]


def _ptyxis_theme(
    key: str, name: str, bg: str, fg: str,
    accent: str, success: str, warning: str, error: str,
) -> Theme:
    """Build a Theme from a Ptyxis palette's semantic colours.

    Surfaces are derived from bg/fg the same way the Android side does:
    darker crust/mantle by lerping toward black, raised surface/overlay by
    lerping toward the foreground.
    """
    return Theme(
        id=key,
        name=name,
        is_dark=True,
        base=bg,
        mantle=_lerp_hex(bg, _BLACK, 0.20),
        crust=_lerp_hex(bg, _BLACK, 0.40),
        surface=_lerp_hex(bg, fg, 0.09),
        overlay=_lerp_hex(bg, fg, 0.20),
        chat_bg=_lerp_hex(bg, _BLACK, 0.20),
        text=fg,
        subtext=_lerp_hex(fg, bg, 0.30),
        accent=accent,
        accent_fg=_on_color(accent),
        success=success,
        warning=warning,
        error=error,
        user_bubble=_lerp_hex(bg, fg, 0.09),
        user_bubble_border=accent,
        asst_bubble=_lerp_hex(bg, _BLACK, 0.20),
        asst_bubble_border=_lerp_hex(bg, fg, 0.20),
    )


_PTYXIS_THEMES: list[Theme] = [
    _ptyxis_theme(*row) for row in (_PTYXIS_BASE + _PTYXIS_EXTENDED)
]


ALL_THEMES: dict[str, Theme] = {
    t.id: t for t in (
        CATPPUCCIN_MOCHA,
        CATPPUCCIN_LATTE,
        CATPPUCCIN_FRAPPE,
        CATPPUCCIN_MACCHIATO,
        DRACULA,
        DRACULA_PRO,
    )
}
# Append the 44 Ptyxis palettes (11 base + 33 extended) → 50 themes total.
ALL_THEMES.update({t.id: t for t in _PTYXIS_THEMES})

# ── Glass-mode companion themes ─────────────────────────────────────────────
# Tuned for translucency: desaturated cool surfaces look right with the
# desktop showing through, and the accents are picked to read as "lit
# glass". Both glass modes work with ANY theme — these just flatter them.
GLASS = Theme(
    id="glass",
    name="Glass",
    is_dark=True,
    base="#10151a",
    mantle="#0c1116",
    crust="#080d11",
    surface="#1a2128",
    overlay="#303a44",
    chat_bg="#0e1318",
    text="#eaf1f7",
    subtext="#9fb0bd",
    accent="#7fd6e8",     # ice — reads as light through glass
    accent_fg="#06252c",
    success="#8fd9a8",
    warning="#eed49f",
    error="#ee99a0",
    user_bubble="#1a2128",
    user_bubble_border="#7fd6e8",
    asst_bubble="#12181e",
    asst_bubble_border="#303a44",
)

LIQUID_GLASS = Theme(
    id="liquid-glass",
    name="Liquid Glass",
    is_dark=True,
    base="#12101e",
    mantle="#0d0b18",
    crust="#090713",
    surface="#1d1a2e",
    overlay="#363150",
    chat_bg="#100e1b",
    text="#eeeaff",
    subtext="#a89fc9",
    accent="#8a9bf8",     # indigo — the liquid light-wash colour
    accent_fg="#0b0f2a",
    success="#9ae6b4",
    warning="#f5d08c",
    error="#f28fad",
    user_bubble="#1d1a2e",
    user_bubble_border="#8a9bf8",
    asst_bubble="#14111f",
    asst_bubble_border="#363150",
)

ALL_THEMES.update({GLASS.id: GLASS, LIQUID_GLASS.id: LIQUID_GLASS})

DEFAULT_THEME_ID = "catppuccin-mocha"

# Per-theme accent palettes.  First entry is the theme's canonical default.
THEME_ACCENTS: dict[str, dict[str, str]] = {
    "catppuccin-mocha": {
        "Mauve":        "#cba6f7",
        "Rosewater":    "#f5e0dc",
        "Flamingo":     "#f2cdcd",
        "Pink":         "#f5c2e7",
        "Red":          "#f38ba8",
        "Maroon":       "#eba0ac",
        "Peach":        "#fab387",
        "Yellow":       "#f9e2af",
        "Green":        "#a6e3a1",
        "Teal":         "#94e2d5",
        "Sky":          "#89dceb",
        "Sapphire":     "#74c7ec",
        "Blue":         "#89b4fa",
        "Lavender":     "#b4befe",
        "White":        "#f5f5f5",
        "Dark Grey":    "#6c7086",
        "Indigo":       "#5c5f9c",
        "Charcoal":     "#45475a",
    },
    "catppuccin-latte": {
        "Mauve":        "#8839ef",
        "Rosewater":    "#dc8a78",
        "Flamingo":     "#dd7878",
        "Pink":         "#ea76cb",
        "Red":          "#d20f39",
        "Maroon":       "#e64553",
        "Peach":        "#fe640b",
        "Yellow":       "#df8e1d",
        "Green":        "#40a02b",
        "Teal":         "#179299",
        "Sky":          "#04a5e5",
        "Sapphire":     "#209fb5",
        "Blue":         "#1e66f5",
        "Lavender":     "#7287fd",
        "White":        "#f5f5f5",
        "Dark Grey":    "#7c7f93",
        "Indigo":       "#5c5f9c",
        "Charcoal":     "#5c5f77",
    },
    "dracula": {
        "Purple":       "#bd93f9",
        "Pink":         "#ff79c6",
        "Red":          "#ff5555",
        "Orange":       "#ffb86c",
        "Yellow":       "#f1fa8c",
        "Green":        "#50fa7b",
        "Cyan":         "#8be9fd",
        "White":        "#f8f8f2",
        "Dark Grey":    "#6272a4",
        "Indigo":       "#5555aa",
        "Charcoal":     "#44475a",
    },
    "catppuccin-frappe": {
        "Mauve":        "#ca9ee6",
        "Rosewater":    "#f2d5cf",
        "Flamingo":     "#eebebe",
        "Pink":         "#f4b8e4",
        "Red":          "#e78284",
        "Maroon":       "#ea999c",
        "Peach":        "#ef9f76",
        "Yellow":       "#e5c890",
        "Green":        "#a6d189",
        "Teal":         "#81c8be",
        "Sky":          "#99d1db",
        "Sapphire":     "#85c1dc",
        "Blue":         "#8caaee",
        "Lavender":     "#babbf1",
        "White":        "#f5f5f5",
        "Dark Grey":    "#737994",
        "Indigo":       "#5c5f9c",
        "Charcoal":     "#51576d",
    },
    "catppuccin-macchiato": {
        "Mauve":        "#c6a0f6",
        "Rosewater":    "#f4dbd6",
        "Flamingo":     "#f0c6c6",
        "Pink":         "#f5bde6",
        "Red":          "#ed8796",
        "Maroon":       "#ee99a0",
        "Peach":        "#f5a97f",
        "Yellow":       "#eed49f",
        "Green":        "#a6da95",
        "Teal":         "#8bd5ca",
        "Sky":          "#91d7e3",
        "Sapphire":     "#7dc4e4",
        "Blue":         "#8aadf4",
        "Lavender":     "#b7bdf8",
        "White":        "#f5f5f5",
        "Dark Grey":    "#6e738d",
        "Indigo":       "#5c5f9c",
        "Charcoal":     "#494d64",
    },
    "dracula-pro": {
        "Purple":       "#9580ff",
        "Pink":         "#ff80bf",
        "Red":          "#ff9580",
        "Orange":       "#ffca80",
        "Yellow":       "#ffff80",
        "Green":        "#8aff80",
        "Cyan":         "#80ffea",
        "White":        "#f8f8f2",
        "Dark Grey":    "#7970a9",
        "Indigo":       "#5555aa",
        "Charcoal":     "#454158",
    },
}


# iOS-style solid bubble colour palettes.
# Each non-default entry: user_bg/fg and asst_bg/fg.
# "default" (None) uses the per-theme user_bubble / asst_bubble colours.
# Optional "role_clr" overrides the role-label colour (default: white 65%).
_DARK_ROLE = "rgba(30,30,46,0.60)"

BUBBLE_PALETTES: dict[str, dict | None] = {
    "default":  None,
    "imessage": {
        "display": "Classic",
        "user_bg": "#34C759", "user_fg": "#FFFFFF",
        "asst_bg": "#007AFF", "asst_fg": "#FFFFFF",
    },
    "candy": {
        "display": "Candy",
        "user_bg": "#FF6B9D", "user_fg": "#FFFFFF",
        "asst_bg": "#9B59B6", "asst_fg": "#FFFFFF",
    },
    "ocean": {
        "display": "Ocean",
        "user_bg": "#00B4D8", "user_fg": "#FFFFFF",
        "asst_bg": "#0077B6", "asst_fg": "#FFFFFF",
    },
    "sunset": {
        "display": "Sunset",
        "user_bg": "#FF6B35", "user_fg": "#FFFFFF",
        "asst_bg": "#7B2D8B", "asst_fg": "#FFFFFF",
    },
    "catppuccin": {
        "display": "Catppuccin",
        "user_bg": "#cba6f7", "user_fg": "#1e1e2e",
        "asst_bg": "#74c7ec", "asst_fg": "#1e1e2e",
        "role_clr": _DARK_ROLE,
    },
    "mocha-bloom": {
        "display": "Mocha Bloom",
        "user_bg": "#f5c2e7", "user_fg": "#1e1e2e",
        "asst_bg": "#a6e3a1", "asst_fg": "#1e1e2e",
        "role_clr": _DARK_ROLE,
    },
    "mocha-flame": {
        "display": "Mocha Flame",
        "user_bg": "#fab387", "user_fg": "#1e1e2e",
        "asst_bg": "#89b4fa", "asst_fg": "#1e1e2e",
        "role_clr": _DARK_ROLE,
    },
    "dracula-bubble": {
        "display": "Dracula",
        "user_bg": "#ff79c6", "user_fg": "#282a36",
        "asst_bg": "#bd93f9", "asst_fg": "#282a36",
        "role_clr": _DARK_ROLE,
    },
    "nord-bubble": {
        "display": "Nord",
        "user_bg": "#88c0d0", "user_fg": "#2e3440",
        "asst_bg": "#81a1c1", "asst_fg": "#2e3440",
        "role_clr": _DARK_ROLE,
    },
    "everforest-bubble": {
        "display": "Everforest",
        "user_bg": "#a7c080", "user_fg": "#2d353b",
        "asst_bg": "#7fbbb3", "asst_fg": "#2d353b",
        "role_clr": _DARK_ROLE,
    },
    "monokai-bubble": {
        "display": "Monokai",
        "user_bg": "#ff6188", "user_fg": "#2d2a2e",
        "asst_bg": "#78dce8", "asst_fg": "#2d2a2e",
        "role_clr": _DARK_ROLE,
    },
    "solarized-bubble": {
        "display": "Solarized",
        "user_bg": "#268bd2", "user_fg": "#FFFFFF",
        "asst_bg": "#2aa198", "asst_fg": "#FFFFFF",
    },
    "forest": {
        "display": "Forest",
        "user_bg": "#52b788", "user_fg": "#FFFFFF",
        "asst_bg": "#40916c", "asst_fg": "#FFFFFF",
    },
    "grape": {
        "display": "Grape",
        "user_bg": "#9b5de5", "user_fg": "#FFFFFF",
        "asst_bg": "#7b2cbf", "asst_fg": "#FFFFFF",
    },
    "ember": {
        "display": "Ember",
        "user_bg": "#e76f51", "user_fg": "#FFFFFF",
        "asst_bg": "#bc4749", "asst_fg": "#FFFFFF",
    },
}


def get_accent_hex(theme_id: str, accent_name: str) -> str | None:
    """Return the hex for a named accent in a given theme, or None if unknown."""
    return THEME_ACCENTS.get(theme_id, {}).get(accent_name)


def get_theme(theme_id: str) -> Theme:
    return ALL_THEMES.get(theme_id, ALL_THEMES[DEFAULT_THEME_ID])


def _accent_fg(accent_hex: str, theme: Theme) -> str:
    """Pick a legible foreground for accent — dark crust on light accents."""
    h = accent_hex.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return theme.crust if luminance > 0.60 else theme.accent_fg


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert #RRGGBB hex to CSS rgba(r,g,b,alpha)."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.2f})"


def build_css(
    theme: Theme,
    font_size: int = 14,
    accent_hex: str | None = None,
    bubble_accent_hex: str | None = None,
    bubble_style: str = "default",
    bubble_opacity: float = 1.0,
    user_bubble_text_color: str = "",
    asst_bubble_text_color: str = "",
    chat_font_family: str = "",
    composer_use_accent: bool = False,
    glass: bool = False,
    glass_opacity: float = 0.82,
    glass_liquid: bool = False,
    header_font: str = "",
) -> str:
    """Generate full CSS for the given theme + palette + font size.

    ``accent_hex`` overrides the theme's main accent (buttons, links, sidebar
    selection, composer focus, attachment chips). ``bubble_accent_hex``
    overrides the bubble-specific accent (user bubble border + role labels);
    falls back to ``accent_hex``/``theme.accent`` when None.
    """
    ac    = accent_hex or theme.accent
    ac_fg = _accent_fg(ac, theme)
    bub_ac = bubble_accent_hex or ac

    # Resolve bubble colours from the active palette.
    palette = BUBBLE_PALETTES.get(bubble_style)
    if palette:
        ub_bg          = palette["user_bg"]
        ub_fg          = palette["user_fg"]
        ub_bdr         = palette["user_bg"]
        ab_bg          = palette["asst_bg"]
        ab_fg          = palette["asst_fg"]
        ab_bdr         = palette["asst_bg"]
        user_role_clr  = palette.get("role_clr", "rgba(255,255,255,0.65)")
        asst_role_clr  = palette.get("role_clr", "rgba(255,255,255,0.65)")
    else:
        ub_bg          = theme.user_bubble
        ub_fg          = theme.text
        ub_bdr         = bub_ac
        ab_bg          = theme.asst_bubble
        ab_fg          = theme.text
        ab_bdr         = theme.asst_bubble_border
        user_role_clr  = bub_ac
        asst_role_clr  = bub_ac

    # Custom text colour overrides.
    if user_bubble_text_color:
        ub_fg = user_bubble_text_color
    if asst_bubble_text_color:
        ab_fg = asst_bubble_text_color

    # Background opacity — convert hex → rgba.
    if bubble_opacity < 0.99:
        ub_bg  = _hex_to_rgba(ub_bg,  bubble_opacity)
        ab_bg  = _hex_to_rgba(ab_bg,  bubble_opacity)
        ub_bdr = _hex_to_rgba(ub_bdr, min(bubble_opacity + 0.15, 1.0))
        ab_bdr = _hex_to_rgba(ab_bdr, min(bubble_opacity + 0.15, 1.0))

    font_family_rule = f'font-family: "{chat_font_family}";' if chat_font_family else ""

    # Composer colours
    if composer_use_accent:
        comp_bg        = ac
        comp_fg        = ac_fg
        comp_border    = ac
        comp_border_w  = "2px"
        comp_focus_bdr = ac
    else:
        comp_bg        = theme.chat_bg
        comp_fg        = theme.text
        comp_border    = theme.overlay
        comp_border_w  = "1px"
        comp_focus_bdr = ac

    return f"""
/* ──────────────────────────────────────────────────────────────────────
 * {theme.name}  —  applied to the entire app via @define-color overrides.
 * libadwaita reads these named colors throughout: headerbar, sidebar,
 * popovers, dialogs, AlertDialog responses, scrollbars, etc.
 * ────────────────────────────────────────────────────────────────────── */

@define-color window_bg_color    {theme.base};
@define-color window_fg_color    {theme.text};
@define-color view_bg_color      {theme.chat_bg};
@define-color view_fg_color      {theme.text};
@define-color headerbar_bg_color {theme.crust};
@define-color headerbar_fg_color {theme.text};
@define-color headerbar_border_color {theme.overlay};
@define-color headerbar_backdrop_color {theme.crust};
@define-color sidebar_bg_color   {theme.mantle};
@define-color sidebar_fg_color   {theme.text};
@define-color sidebar_backdrop_color {theme.mantle};
@define-color sidebar_border_color {theme.overlay};
@define-color secondary_sidebar_bg_color {theme.mantle};
@define-color secondary_sidebar_fg_color {theme.text};
@define-color card_bg_color      {theme.surface};
@define-color card_fg_color      {theme.text};
@define-color popover_bg_color   {theme.surface};
@define-color popover_fg_color   {theme.text};
@define-color dialog_bg_color    {theme.base};
@define-color dialog_fg_color    {theme.text};
@define-color shade_color        alpha(black, 0.30);
@define-color scrollbar_outline_color alpha({theme.overlay}, 0.5);

@define-color accent_color       {ac};
@define-color accent_bg_color    {ac};
@define-color accent_fg_color    {ac_fg};

@define-color destructive_color    {theme.error};
@define-color destructive_bg_color {theme.error};
@define-color destructive_fg_color {ac_fg};

@define-color success_color    {theme.success};
@define-color success_bg_color {theme.success};
@define-color success_fg_color {theme.crust};

@define-color warning_color    {theme.warning};
@define-color warning_bg_color {theme.warning};
@define-color warning_fg_color {theme.crust};

@define-color error_color    {theme.error};
@define-color error_bg_color {theme.error};
@define-color error_fg_color {theme.accent_fg};

@define-color borders alpha({theme.overlay}, 0.7);

/* ── Force theme colours on libadwaita widgets ────────────────────────────
 * libadwaita's FORCE_DARK / FORCE_LIGHT color scheme applies its own dark
 * palette via internal CSS that wins over @define-color in some places.
 * These rules target the concrete widget classes used by NavigationSplitView
 * + ToolbarView so the theme's hues propagate to sidebar, headerbar, and
 * window background — not just the chat panel. */

window, window.background {{
    background-color: {theme.base};
    color: {theme.text};
}}

/* Aux toplevels (Box Code, Image Tools, …) stay opaque even in glass
 * modes — stacked translucent windows read as ghosting/broken text.
 * The class outranks the glass/liquid `window` rules appended later. */
window.aux-solid, window.aux-solid.background {{
    background-color: {theme.base};
    background-image: none;
}}

/* NavigationSplitView sidebar pane */
.sidebar-pane,
.sidebar-pane > .background,
navigation-split-view > .sidebar-pane,
.navigation-sidebar {{
    background-color: {theme.mantle};
    color: {theme.text};
}}

/* NavigationSplitView content pane */
.content-pane,
.content-pane > .background {{
    background-color: {theme.base};
    color: {theme.text};
}}

/* Headerbars in both panes */
headerbar,
.toolbar headerbar,
toolbarview > headerbar {{
    background-color: {theme.crust};
    color: {theme.text};
    box-shadow: none;
}}

/* ToolbarView's container (the body between header and bottom bars) */
toolbarview {{
    background-color: transparent;
}}

/* AdwNavigationPage content background */
navigationpage {{
    background-color: {theme.base};
}}

/* Popovers — force theme colours so headerbar dropdowns/menus/quick toggles
 * don't fall through to the system (Yaru / Adwaita default) palette.
 *
 * Only the inner `> contents` and `> arrow` nodes draw; the outer
 * `popover` element is the wrapper surface and must stay transparent
 * (otherwise you get a hard rectangle around the rounded card). */
popover {{
    background: transparent;
    border: none;
    box-shadow: none;
}}
popover > contents,
popover.menu > contents {{
    background-color: {theme.mantle};
    color: {theme.text};
    border: 1px solid {theme.overlay};
    border-radius: 12px;
    padding: 6px;
}}
popover > arrow {{
    background-color: {theme.mantle};
    border: 1px solid {theme.overlay};
}}
popover separator {{
    background-color: alpha({theme.overlay}, 0.7);
}}
popover button,
popover button.flat,
popover modelbutton {{
    color: {theme.text};
    background-color: transparent;
    border-radius: 6px;
}}
popover button:hover,
popover button.flat:hover,
popover modelbutton:hover {{
    background-color: alpha({theme.overlay}, 0.5);
}}
popover label {{
    color: {theme.text};
}}
popover label.caption,
popover label.dim-label {{
    color: alpha({theme.text}, 0.65);
}}

/* Chat scroller — explicitly tint to chat_bg */
.chat-scroller, .chat-scroller viewport, .chat-scroller > viewport {{
    background-color: {theme.chat_bg};
}}

/* ── Context-usage progress bar ──────────────────────────────────────────
 * Lives above the composer. Colour shifts via state classes added in code:
 *   default → accent  |  ctx-warn → yellow  |  ctx-crit → red */

progressbar.context-bar trough {{
    min-height: 4px;
    background: alpha({theme.overlay}, 0.5);
    border-radius: 2px;
}}
progressbar.context-bar progress {{
    background-color: {ac};
    border-radius: 2px;
}}
progressbar.context-bar.ctx-warn progress {{
    background-color: {theme.warning};
}}
progressbar.context-bar.ctx-crit progress {{
    background-color: {theme.error};
}}

/* ── macOS-style traffic lights (custom buttons) ─────────────────────────
 * Native AdwHeaderBar window controls have a libadwaita-enforced min-height
 * that we cannot reliably override, so we pack three plain Gtk.Buttons and
 * style them ourselves — full control over size, shape, and colour. */

box.traffic-lights {{
    margin: 0 8px;
}}

button.traffic-light {{
    min-width: 12px;
    min-height: 12px;
    padding: 0;
    margin: 0;
    border-radius: 6px;
    border: none;
    background: #888;
    background-image: none;
    box-shadow: none;
    outline: none;
}}

/* Pastel palette (Catppuccin Frappé, matching ATK's titlebar) — flat,
 * borderless discs read crisper than the old dark-ringed macOS colours. */
button.traffic-light.traffic-close {{ background: #e78284; background-image: none; }}
button.traffic-light.traffic-min   {{ background: #e5c890; background-image: none; }}
button.traffic-light.traffic-max   {{ background: #a6d189; background-image: none; }}

button.traffic-light:hover  {{ filter: brightness(1.12); }}
button.traffic-light:active {{ filter: brightness(0.9); }}

/* ── Chat bubbles ─────────────────────────────────────────────────────── */

.bubble {{
    padding: 10px 14px;
    border-radius: 18px;
    margin: 4px 0;
    font-size: {font_size}px;
}}

.bubble-user {{
    background: {ub_bg};
    border: 1px solid {ub_bdr};
}}

.bubble-assistant {{
    background: {ab_bg};
    border: 1px solid {ab_bdr};
}}

.bubble-system {{
    background: alpha({theme.warning}, 0.18);
    border: 1px solid alpha({theme.warning}, 0.45);
    font-style: italic;
}}

.bubble-role {{
    font-size: 0.78em;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}}

.bubble-user .bubble-role   {{ color: {user_role_clr}; }}
.bubble-assistant .bubble-role {{ color: {asst_role_clr}; opacity: 0.7; }}
.bubble-system .bubble-role {{ color: {theme.warning}; }}

/* Bubble body label — per bubble type so palette text colours work */
label.bubble-body {{ font-size: {font_size}px; {font_family_rule} }}
.bubble-user      label.bubble-body {{ color: {ub_fg}; }}
.bubble-assistant label.bubble-body {{ color: {ab_fg}; }}
.bubble-system    label.bubble-body {{ color: {theme.text}; }}

/* Thinking animation label inside streaming assistant bubbles */
.thinking-dots {{ font-size: 0.82em; letter-spacing: 1px; }}

/* Hover action buttons — invisible at rest, fade in on hover */
.bubble-actions {{ opacity: 0; transition: opacity 120ms ease; }}
.bubble-actions-visible {{ opacity: 1; }}

/* Allow the chat scroller to shrink freely — overrides StatusPage minimum */
.chat-scroller {{ min-height: 0; min-width: 0; }}

/* ── Retrieved-context expander (RAG card on assistant bubbles) ───────── */
expander.rag-context {{
    margin-top: 4px;
    margin-bottom: 4px;
    padding: 4px 6px;
    border-radius: 10px;
    background: alpha({theme.overlay}, 0.35);
    border: 1px solid alpha({theme.overlay}, 0.6);
    font-size: 0.85em;
}}
expander.rag-context > title {{ padding: 2px 4px; opacity: 0.85; }}

/* ── Tool-call card (Phase 4: one per fn call in the assistant bubble) ── */
expander.tool-call-card {{
    margin-top: 4px;
    margin-bottom: 4px;
    padding: 4px 6px;
    border-radius: 10px;
    background: alpha({theme.overlay}, 0.30);
    border: 1px solid alpha({theme.overlay}, 0.55);
    font-size: 0.85em;
}}
expander.tool-call-card > title {{ padding: 2px 4px; opacity: 0.9; }}
expander.tool-call-card.tool-call-denied {{
    background: alpha({theme.error}, 0.18);
    border-color: alpha({theme.error}, 0.55);
}}

/* Header tools button — active class set when any tool is on for this chat */
button.tools-btn-active {{
    color: {ac};
}}

/* Composer 🔊 button — tinted when TTS auto-speak is on. */
button.tts-on {{
    color: {ac};
}}

/* Display-math image (LaTeX $$…$$ rendered via matplotlib) */
picture.math-display {{
    margin-top: 4px;
    margin-bottom: 4px;
}}

/* Agent progress pill (Phase 5) — shown while a tool-chain is mid-run */
.agent-pill {{
    background-color: alpha({ac}, 0.18);
    color: {ac};
    border: 1px solid alpha({ac}, 0.45);
    border-radius: 11px;
    padding: 1px 9px;
    margin-left: 2px;
    font-size: 0.82em;
    font-weight: 600;
}}

/* Live-mode panel (Phase 4.5 Tier 3) */
.live-panel {{
    background-color: {theme.mantle};
    border-bottom: 1px solid {theme.overlay};
}}
.live-preview {{
    border-radius: 8px;
    background-color: {theme.crust};
}}
.live-status {{
    font-weight: 500;
}}
.live-status.live-listening {{ color: {ac}; }}
.live-status.live-processing {{ color: {theme.warning}; }}
.live-status.live-speaking {{ color: {theme.success}; }}

/* ── Knowledge Base side pane ─────────────────────────────────────────── */
.kb-panel {{
    background: {theme.chat_bg};
    border-left: 1px solid {theme.overlay};
}}
.kb-panel listbox.boxed-list {{
    background: transparent;
}}


/* ── Composer ─────────────────────────────────────────────────────────── */

.input-area {{
    background: {theme.chat_bg};
    border-top: 1px solid {theme.overlay};
}}

.composer-frame {{
    background-color: {comp_bg};
    border-style: solid;
    border-width: {comp_border_w};
    border-color: {comp_border};
    border-radius: 8px;
}}

.composer-frame:focus-within {{
    border-color: {comp_focus_bdr};
    border-width: 2px;
}}

#box-composer-input,
#box-composer-input text,
textview.composer,
textview.composer text {{
    background: {comp_bg};
    background-color: {comp_bg};
    color: {comp_fg};
    caret-color: {comp_fg};
}}

textview.composer:focus,
textview.composer:focus-visible {{
    outline-style: none;
    outline-width: 0;
    box-shadow: none;
}}

/* ── Attachment chips ─────────────────────────────────────────────────── */

.attachment-chip {{
    background: alpha({ac}, 0.18);
    border: 1px solid alpha({ac}, 0.4);
    border-radius: 6px;
    padding: 2px 6px;
    color: {theme.text};
}}

.attachment-chip label {{
    font-size: 0.85em;
    color: {theme.text};
}}

/* ── Misc polish ──────────────────────────────────────────────────────── */

.empty-state-title {{
    font-size: 1.5em;
    font-weight: 800;
    color: {theme.text};
}}

.empty-state-subtitle {{
    color: {theme.subtext};
}}

/* Selection highlight in the sidebar */
row:selected, row:selected:hover {{
    background: alpha({ac}, 0.20);
    color: {theme.text};
}}

/* Suggested-action buttons should use our accent, not Adwaita's blue */
button.suggested-action {{
    background: {ac};
    color: {ac_fg};
}}
button.suggested-action:hover {{
    background: shade({ac}, 1.1);
}}

button.destructive-action {{
    background: {theme.error};
    color: {ac_fg};
}}

/* ── Accent wiring ────────────────────────────────────────────────────── */

/* Links anywhere in the app (about dialog, etc.) */
link, a {{ color: @accent_color; }}

/* Loading spinner follows accent */
spinner {{ color: @accent_color; }}


""" + (
        _glass_css(theme, glass_opacity) if (glass or glass_liquid) else ""
    ) + (_liquid_css(theme, ac) if glass_liquid else "") + (
        _header_font_css(header_font) if header_font else ""
    )


def _header_font_css(family: str) -> str:
    """App-wide header typography (Appearance → Header font)."""
    fam = family.replace('"', "")
    return f"""
/* ── Header font ──────────────────────────────────────────────────────── */
headerbar label.title,
headerbar label.subtitle,
statuspage label.title,
window .title-1, window .title-2, window .title-3, window .title-4,
label.heading,
.header-font, .header-font label {{
    font-family: "{fam}";
}}
"""


def _glass_css(theme: Theme, opacity: float) -> str:
    """iOS-style glass overrides, appended after the theme CSS so the
    cascade lets them win. Real translucency (the compositor shows what's
    behind the window) + luminous hairline borders. GNOME can't blur
    behind arbitrary windows, so alpha is kept moderate for readability —
    ``opacity`` is the main surface's alpha, panels go a step clearer.
    """
    a = max(0.5, min(1.0, float(opacity)))
    a_side = max(0.40, a - 0.20)   # sidebar sees more through
    a_head = max(0.35, a - 0.25)   # headerbar clearest — the "glass edge"
    a_chat = max(0.45, a - 0.15)
    edge = "rgba(255,255,255,0.09)" if theme.is_dark else "rgba(0,0,0,0.10)"
    glow = "rgba(255,255,255,0.05)" if theme.is_dark else "rgba(255,255,255,0.35)"
    return f"""
/* ── Glass mode ────────────────────────────────────────────────────────
 * Later rules at equal specificity override the opaque theme blocks. */

window, window.background {{
    background-color: {_hex_to_rgba(theme.base, a)};
}}

.sidebar-pane,
.sidebar-pane > .background,
navigation-split-view > .sidebar-pane,
.navigation-sidebar {{
    background-color: {_hex_to_rgba(theme.mantle, a_side)};
}}

.content-pane,
.content-pane > .background {{
    background-color: {_hex_to_rgba(theme.base, a_side)};
}}

headerbar,
.toolbar headerbar,
toolbarview > headerbar {{
    background-color: {_hex_to_rgba(theme.crust, a_head)};
    border-bottom: 1px solid {edge};
    box-shadow: inset 0 1px 0 {glow};
}}

.chat-scroller, .chat-scroller viewport, .chat-scroller > viewport {{
    background-color: {_hex_to_rgba(theme.chat_bg, a_chat)};
}}

.input-area {{
    background-color: {_hex_to_rgba(theme.chat_bg, min(1.0, a + 0.06))};
    border: 1px solid {edge};
}}

.card, preferencesgroup > box.card {{
    background-color: {_hex_to_rgba(theme.surface, a)};
    border: 1px solid {edge};
}}

popover > contents {{
    background-color: {_hex_to_rgba(theme.surface, min(1.0, a + 0.10))};
    border: 1px solid {edge};
}}

.nav-rail {{
    background-color: {_hex_to_rgba(theme.crust, a_head)};
}}
"""


def _liquid_css(theme: Theme, accent: str) -> str:
    """Liquid-glass layer on top of glass mode (v3 — geometry + restraint).

    v2's gradient-sheen "lens" bevels read as Win9x chrome; gone. Liquid
    now keeps glass mode's flatness and adds only: pill/rounded geometry
    everywhere, hairline edges, floating ambient shadows, a whisper of
    accent light-wash on the window, accent-tinted flat selection, and
    responsive light (hover wash, press dim, accent focus glow). Pure
    GTK4 CSS, no runtime cost.
    """
    if theme.is_dark:
        spec_soft = "rgba(255,255,255,0.08)"
        lens_edge = "rgba(255,255,255,0.10)"
        ambient = "rgba(0,0,0,0.22)"
        hover_wash = "rgba(255,255,255,0.06)"
    else:
        spec_soft = "rgba(255,255,255,0.45)"
        lens_edge = "rgba(0,0,0,0.09)"
        ambient = "rgba(0,0,0,0.10)"
        hover_wash = "rgba(255,255,255,0.30)"
    wash = _hex_to_rgba(accent, 0.06 if theme.is_dark else 0.05)
    glow = _hex_to_rgba(accent, 0.25)
    sel = _hex_to_rgba(accent, 0.16)
    sel_edge = _hex_to_rgba(accent, 0.30)
    return f"""
/* ── Liquid glass v3 ───────────────────────────────────────────────────
 * Geometry + restraint: pills, hairlines, a whisper of accent light and
 * interactive hover washes. NO gradient bevels — v2's sheen gradients
 * read as Win9x chrome (Jegly's verdict). Glass mode's flatness stays;
 * liquid adds only shape and responsive light. Appended last: wins. */

window, window.background {{
    background-image: linear-gradient(
        to bottom right, {wash}, transparent 50%);
}}

headerbar,
.toolbar headerbar,
toolbarview > headerbar {{
    box-shadow: inset 0 1px 0 {spec_soft};
}}

/* ── Buttons: pill shape + responsive light. Flat as glass itself. ───── */
button {{
    border-radius: 9999px;
    transition: background-color 140ms ease, box-shadow 140ms ease,
                filter 140ms ease;
}}

button:not(.traffic-light):hover {{
    background-color: {hover_wash};
}}

button:not(.traffic-light):active {{
    filter: brightness(0.92);
}}

button.suggested-action, button.destructive-action {{
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.16);
    border: none;
}}

button:focus-visible, entry:focus-within {{
    outline: none;
    box-shadow: 0 0 0 3px {glow};
}}

/* ── Chat bubbles: rounded glass panes, no chrome ────────────────────── */
.bubble {{
    border-radius: 20px;
    border: 1px solid {lens_edge};
    box-shadow: 0 6px 18px {ambient};
}}

/* ── Composer + entries ──────────────────────────────────────────────── */
entry {{
    border-radius: 9999px;
}}

.input-area {{
    border-radius: 22px;
    border: 1px solid {lens_edge};
    box-shadow: 0 4px 14px {ambient};
}}

/* ── Cards, popovers, tooltips: rounded, hairlined, floating ─────────── */
.card {{
    border-radius: 18px;
    border: 1px solid {lens_edge};
    box-shadow: 0 8px 22px {ambient};
}}

popover > contents {{
    border-radius: 16px;
    border: 1px solid {lens_edge};
    box-shadow: 0 10px 26px {ambient};
}}

popover modelbutton,
popover .menuitem {{
    border-radius: 9999px;
    padding: 6px 12px;
}}

popover modelbutton:hover {{
    background-color: {hover_wash};
}}

tooltip, tooltip.background {{
    border-radius: 12px;
    border: 1px solid {lens_edge};
}}

/* ── Selection: one flat accent tint, no gradients ───────────────────── */
.navigation-sidebar row {{
    border-radius: 9999px;
    margin-left: 6px;
    margin-right: 6px;
    transition: background-color 140ms ease;
}}

.navigation-sidebar row:hover {{
    background-color: {hover_wash};
}}

.navigation-sidebar row:selected {{
    background-color: {sel};
    border: 1px solid {sel_edge};
}}

.nav-rail button {{
    border-radius: 9999px;
    padding: 8px;
}}

.nav-rail button.suggested-action {{
    background-color: {sel};
    background-image: none;
    border: 1px solid {sel_edge};
    box-shadow: none;
    color: {theme.text};
}}

/* ── Small furniture ─────────────────────────────────────────────────── */
scrollbar slider {{
    border-radius: 9999px;
    min-width: 6px;
    min-height: 6px;
    background-color: rgba({'255,255,255' if theme.is_dark else '0,0,0'},0.25);
}}

scrollbar slider:hover {{
    background-color: rgba({'255,255,255' if theme.is_dark else '0,0,0'},0.40);
}}

switch, switch slider {{
    border-radius: 9999px;
}}

scale trough, progressbar trough,
scale highlight, progressbar progress {{
    border-radius: 9999px;
}}
"""
