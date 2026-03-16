# Benchmarks für plagih GP

Das Framework enthält mehrere Benchmark-Environments für Tests und Demonstrationen.

## Übersicht

| Benchmark | Ordner | Komplexität | Status | Beschreibung |
|-----------|--------|-------------|--------|--------------|
| **MountainCar** | `benchmarks/mc/` | ⭐ Einfach | ✅ Standard | Kontinuierliche Steuerung, 2 Inputs |
| **CartPole** | `benchmarks/cp/` | ⭐ Einfach | ✅ Vorhanden | Balancieren, 4 Inputs, diskrete Ausgabe |
| **Symbolic Regression** | `benchmarks/sr/` | ⭐ Einfach | ✅ Neu | Klassisches GP-Benchmark, 1 Input |
| **Industrial Benchmark** | `benchmarks/ib/` | ⭐⭐⭐ Komplex | ⚠️ Experimentell | Viele Inputs, industrielles Szenario |

## Parallel-Performance-Diagnose

Die aktuelle detaillierte Analyse der Windows-Parallelisierung, inklusive
Shared-Memory-Test, Batch-Größen-Vergleich und Populationsvergleich,
steht in:

- `docs/PARALLEL_BENCHMARK_DIAGNOSIS.md`

Direktes CPU-/RAM-Profiling für den aktuellen Parallelpfad steht in:

- `plagih/test/benchmarks/bench_parallel_resources.py`
- Output-Datei: `plagih/test/benchmarks/bench_resources_output.txt`

---

## MountainCar (Standard-Benchmark)

**Verwendet in:** `demo_minimal()`, Standard-Testläufe

**Problem:**
- Ein Auto muss aus einem Tal auf einen Berg fahren
- Das Auto hat nicht genug Kraft, direkt hochzufahren
- Es muss Schwung holen (hin und her fahren)

**Spezifikation:**
```
Inputs:  cartPos (Position: -1.2 bis 0.6)
         cartVel (Geschwindigkeit: -0.07 bis 0.07)
Output:  action (0=links, 1=nichts, 2=rechts)
```

**Dateien:**
```
benchmarks/mc/
├── gp_files/
│   ├── samples200.csv              # Kleine Trainingsmenge (200 Samples)
│   ├── samples75.csv               # Sehr kleine Menge (75 Samples)
│   ├── behaviour_samples.csv       # Vollständige Samples (~2000)
│   └── tree_*.csv                  # Verschiedene Startbäume
└── agents/
    └── ...                         # Evaluation-Agenten
```

**Beispiel-Aufruf:**
```python
from plagih_gp import demo_minimal
demo_minimal()
```

**Bekannte gute Lösungen:**
- `sign(cartVel)` - Sehr einfach, funktioniert grundlegend
- `sign(cartPos + cartVel)` - Besser

---

## CartPole (Alternatives Benchmark)

**Verwendet in:** `demo_cartpole()`

**Problem:**
- Ein Wagen mit einer Stange muss balanciert werden
- Die Stange darf nicht umfallen
- Klassisches Reinforcement Learning Problem

**Spezifikation:**
```
Inputs:  cartPos       (Wagenposition)
         cartVel       (Wagengeschwindigkeit)
         poleAngle     (Stangenwinkel, früher observation2)
         poleVel       (Winkelgeschwindigkeit, früher observation3)
Output:  action (0=links, 1=rechts) - binär!
```

**Dateien:**
```
benchmarks/cp/
├── gp_files/
│   ├── samples.csv                    # Trainingsdaten (~5800 Samples)
│   ├── operators.csv                  # Operatoren-Set
│   └── tree_labels(simple).csv        # Einfacher Startbaum
└── agents/
    ├── cartpole_eval.py               # Gymnasium-Evaluation
    └── yingzwang.py                   # Literatur-Agent
```

**Beispiel-Aufruf:**
```python
from plagih_gp import demo_cartpole
demo_cartpole()
```

**Bekannte gute Lösungen:**
- `poleAngle < 0` - Nur Winkel (einfachste Lösung)
- `poleVel < 0` - Nur Winkelgeschwindigkeit

---

## Symbolic Regression (Klassisches GP-Benchmark)

**Verwendet in:** `demo_symbolic_regression()`

**Problem:**
- Finde eine mathematische Formel, die Daten approximiert
- Klassisches GP-Standard-Benchmark
- Kein externes Environment nötig

**Spezifikation:**
```
Inputs:  x (Wert zwischen -2 und 2)
Output:  target = x³ + x² + x (Zielfunktion)
```

**Dateien:**
```
benchmarks/sr/
└── gp_files/
    └── polynomial.csv    # f(x) = x³ + x² + x
```

**Beispiel-Aufruf:**
```python
from plagih_gp import demo_symbolic_regression
demo_symbolic_regression()
```

**Zielfunktion:**
Die zu findende Formel ist: `f(x) = x³ + x² + x`

Dies ist ein klassisches Benchmark, da:
- Es eine eindeutige Lösung gibt
- Die Lösung relativ einfach ist
- Man den Erfolg leicht messen kann

---

## Industrial Benchmark (Komplex)

**Empfohlen für:** Fortgeschrittene Tests, Skalierbarkeit, Publikationen

⚠️ **Hinweis:** Dieses Benchmark ist deutlich komplexer als die anderen.

**Problem:**
- Simulation eines industriellen Prozesses
- Viele Eingangsvariablen
- Komplexe, nicht-lineare Dynamik

**Spezifikation:**
```
Inputs:  Shift_0, Shift_1, ...     (Verschiebungen)
         Gain_0, Gain_1, ...       (Verstärkungen)
         Setpoint, Velocity, ...   (Weitere Variablen)
Output:  Steuerungsaktion
```

**Dateien:**
```
benchmarks/ib/
├── gp_files/
│   ├── samples_prepared.csv    # Vorbereitete Daten (sehr groß!)
│   └── samples_raw.csv         # Rohdaten
├── ib_eval_agents.py           # Evaluation
└── IDS.py                      # Industrial Data Simulator
```

---

## Benchmark hinzufügen

Um ein neues Benchmark hinzuzufügen:

1. **Ordner erstellen:**
   ```
   benchmarks/BENCHMARK_NAME/
   └── gp_files/
       └── samples.csv          # Trainingsdaten
   ```

2. **Samples-Format:**
   ```csv
   input1:float,input2:float,target:float
   0.1,0.2,1.0
   0.3,-0.1,0.0
   ...
   ```

3. **Demo-Funktion hinzufügen** (in `plagih_gp.py`):
   ```python
   def demo_BENCHMARK_NAME():
       df = pd.read_csv('benchmarks/BENCHMARK_NAME/gp_files/samples.csv')
       gp = ExplainableGP.create(
           symbols=['input1', 'input2'],
           df_train=df_train,
           ...
       )
   ```
