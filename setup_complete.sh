#!/bin/bash
# Setup Script mit System-Paket-Installation

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

# Installiere notwendige System-Pakete
echo "Installiere notwendige System-Pakete..."
echo "Bitte sudo-Passwort eingeben wenn noetig:"
sudo apt update
sudo apt install -y python3-venv python3-pip

echo ""
echo "Erstelle Virtual Environment..."
rm -rf venv
python3 -m venv venv

echo "Aktiviere Environment und installiere Python-Pakete..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup abgeschlossen!"
echo ""
echo "Naechste Schritte:"
echo "   1. Aktiviere das Environment: source venv/bin/activate"
echo "   2. Starte das Projekt: python plagih_gp.py"
echo "   3. Deaktiviere spaeter mit: deactivate"
echo ""

