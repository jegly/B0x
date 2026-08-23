#!/usr/bin/env bash
# Box — DEVELOPMENT setup for Ubuntu 26.04.
#
# This is the non-packaging path: it just gets you to a working venv so you
# can iterate. No system-wide install, no ~/.local/bin shim, no .desktop file.
#
# When you're ready to actually package & install, we'll write install.sh.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "=== Box dev setup ==="
echo "Project: $PROJECT_ROOT"
echo "Venv:    $VENV_DIR"
echo

# ── System packages ──────────────────────────────────────────────────────────
echo "[1/3] Installing system packages (sudo apt)…"
sudo apt update
sudo apt install -y \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-4.0 \
    gir1.2-adw-1 \
    libgtk-4-1 \
    libadwaita-1-0 \
    python3-venv \
    python3-pip

# ── Venv ─────────────────────────────────────────────────────────────────────
echo "[2/3] Creating venv at $VENV_DIR…"
if [[ ! -d "$VENV_DIR" ]]; then
    # --system-site-packages so the venv can see apt's PyGObject
    # (which is built against the system GTK).
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel setuptools

# ── Python deps ──────────────────────────────────────────────────────────────
echo "[3/3] Installing litert-lm-api-nightly and Box (editable)…"
pip install --upgrade litert-lm-api-nightly
pip install --upgrade -e "$PROJECT_ROOT"

deactivate || true

echo
echo "✅  Done."
echo
echo "Run Box with:"
echo "    source $VENV_DIR/bin/activate && box"
echo
echo "Or directly:"
echo "    $VENV_DIR/bin/box"
echo
echo "First launch: open Preferences (Ctrl+,) and point at your .litertlm file."
