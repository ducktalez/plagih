# Setup-Anleitung für PLAGIH

Es gibt zwei empfohlene Wege, um das Projekt einzurichten:

## Option 1: Python Virtual Environment (Empfohlen für Linux)

Diese Methode ist leichtgewichtiger und direkt in Python integriert.

### Schritt-für-Schritt:

```bash
# 1. Navigiere zum Projektverzeichnis
cd /home/schlechti/PycharmProjects/plagih

# 2. Erstelle ein virtuelles Environment
python3 -m venv venv

# 3. Aktiviere das Environment
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Installiere alle Abhängigkeiten
pip install -r requirements.txt

# 6. (Optional) Installiere LaTeX für Visualisierungen
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# 7. Starte das Projekt
python plagih_gp.py
```

### Environment deaktivieren:
```bash
deactivate
```

---

## Option 2: Anaconda/Miniconda

Anaconda ist nützlich, wenn du:
- Mehrere Data-Science-Projekte verwaltest
- Eine GUI für Package-Management bevorzugst
- Bereits Anaconda installiert hast

### Schritt-für-Schritt:

```bash
# 1. Erstelle ein neues Conda Environment
conda create -n plagih python=3.9

# 2. Aktiviere das Environment
conda activate plagih

# 3. Installiere Conda-Pakete
conda install matplotlib sympy tensorflow pandas scipy pyyaml scikit-learn

# 4. Installiere Pip-Pakete (die nicht in Conda verfügbar sind)
pip install apted tikzplotlib graphviz gym

# 5. (Optional) Installiere LaTeX für Visualisierungen
sudo apt-get install texlive-latex-extra texlive-fonts-recommended dvipng cm-super

# 6. Starte das Projekt
python plagih_gp.py
```

### Environment deaktivieren:
```bash
conda deactivate
```

---

## Welche Option wählen?

### Wähle **Virtual Environment (venv)**, wenn:
- ✅ Du ein schlankes Setup möchtest
- ✅ Du nur Python-Projekte hast
- ✅ Du schnell starten möchtest
- ✅ Du keine zusätzliche Software installieren willst

### Wähle **Anaconda**, wenn:
- ✅ Du bereits Anaconda nutzt
- ✅ Du häufig Data-Science/ML-Projekte machst
- ✅ Du komplexe wissenschaftliche Bibliotheken nutzt
- ✅ Du eine GUI bevorzugst

---

## Problembehebung

### TensorFlow-Probleme:
Wenn TensorFlow Probleme macht (häufig auf Linux):
```bash
pip install tensorflow-cpu  # Nur CPU-Version
```

### Import-Fehler:
Stelle sicher, dass du im Projektverzeichnis bist und das Environment aktiviert ist.

### PyYAML-Warnung:
Falls eine Warnung erscheint:
```bash
pip install --upgrade PyYAML
```

---

## Nach dem Setup

### Testlauf:
```bash
python plagih_gp.py
```

Dies startet automatisch einen Beispiel-Run (wahrscheinlich Mountain Car).

### Ergebnisse anschauen:
- `plots/average-fitness.png` - Fitness-Verlauf
- `plots/best_candidate.png` - Beste Lösung
- `plots/pareto.png` - Pareto-Front
- `info/pareto.txt` - Pareto-Kandidaten als Text

---

## Für PyCharm-Nutzer

1. Öffne PyCharm Settings: `File` → `Settings` → `Project: plagih` → `Python Interpreter`
2. Klicke auf das Zahnrad-Symbol → `Add...`
3. Wähle:
   - **Für venv**: `Existing environment` und navigiere zu `venv/bin/python`
   - **Für Conda**: `Conda Environment` und wähle `plagih`
4. Klicke `OK`

Jetzt kann PyCharm das richtige Environment nutzen!

