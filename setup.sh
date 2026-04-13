#!/bin/bash
# Setup Script fuer PLAGIH

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

# Erstelle Virtual Environment
if [ ! -d "venv" ]; then
    echo "Erstelle Virtual Environment..."
    python3 -m venv venv
    echo "Virtual Environment erstellt"
else
    echo "Virtual Environment existiert bereits"
fi
echo ""

# Aktiviere Environment
echo "Aktiviere Environment und installiere Pakete..."
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip --quiet

# Installiere requirements
echo "Installiere Abhaengigkeiten..."
pip install -r requirements.txt

echo ""
echo "Setup abgeschlossen!"
echo ""
echo "Naechste Schritte:"
echo "   1. Aktiviere das Environment: source venv/bin/activate"
echo "   2. Starte das Projekt: python plagih_gp.py"
echo "   3. Deaktiviere spaeter mit: deactivate"
echo ""

