#!/bin/bash
# Direktes Setup Script fuer PLAGIH ohne venv

set -e

echo "PLAGIH Setup wird gestartet..."
echo ""

# Pruefe Python
if ! command -v python3 &> /dev/null; then
    echo "Fehler: Python3 ist nicht installiert!"
    exit 1
fi

echo "Python $(python3 --version) gefunden"
echo ""

# Installiere requirements direkt mit --user
echo "Installiere Abhaengigkeiten..."
pip3 install --user -r requirements.txt

echo ""
echo "Setup abgeschlossen!"
echo ""
echo "Naechste Schritte:"
echo "   1. Starte das Projekt: python3 plagih_gp.py"
echo ""

