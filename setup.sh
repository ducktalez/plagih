#!/bin/bash
# Minimal setup for plagih (Linux/macOS).
# Windows users: see README.md "Quick Start" — use PowerShell instead.
set -e

if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found." >&2
    exit 1
fi

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "Upgrading pip ..."
pip install --upgrade pip --quiet

echo "Installing plagih (editable) + dev extras ..."
pip install -e ".[dev]"

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Run the minimal demo with:"
echo "  python plagih_gp.py"
