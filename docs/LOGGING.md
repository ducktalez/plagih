# Logging in Plagih

Plagih verwendet ein Hybrid-Logging-System, das Python's `logging`-Modul mit benutzerfreundlichem Console-Output kombiniert.

## Quick Start

```python
from plagih.logging_utils import setup_logging
from pathlib import Path

# Einmaliger Aufruf zu Beginn deines Scripts
setup_logging(
    log_file=Path('./logs/my_run.log'),
    console_level=logging.INFO,
    verbose=False
)
```

## Was wird geloggt?

### Console (Benutzer-Feedback)
- **INFO**: Wichtige Fortschrittsmeldungen (Generationen, Paretofront-Updates)
- **WARNING**: Nicht-kritische Probleme (zu große Bäume, Sympy-Fehler)
- **ERROR**: Kritische Fehler

### Log-Datei (Debug-Informationen)
- Alle Console-Meldungen
- **DEBUG**: Detaillierte interne Informationen (Tree-Konstruktion, Evaluierungen)
- Timestamps, Modul-Namen, Funktions-Namen

## Verwendung

### 1. Automatisches Logging (empfohlen)

Die existierenden `printpl()`, `print_warning()`, etc. funktionen nutzen automatisch das Logging-Backend:

```python
from plagih.util import printpl, print_warning

printpl('i', 'Generation 5 completed')  # INFO
print_warning('w', 'Tree too complex')  # WARNING
```

### 2. Direktes Logging (für Framework-Entwicklung)

```python
from plagih.logging_utils import log_debug, log_info, log_warning

log_debug("Created tree with %d nodes", len(tree))
log_info("Pareto front: %d candidates", len(front))
log_warning("Sympy simplification failed for: %s", expr)
```

## Konfiguration

### Verbose Mode
Zeigt auch DEBUG-Meldungen in der Console:

```python
setup_logging(log_file=Path('./run.log'), verbose=True)
```

### Nur Console (kein Log-File)
```python
setup_logging(console_level=logging.INFO)
```

### Verschiedene Log-Level
```python
setup_logging(
    log_file=Path('./run.log'),
    console_level=logging.WARNING,  # Nur Warnungen in Console
    file_level=logging.DEBUG         # Alles in Datei
)
```

## Vorteile

✅ **Produktionsbereit** - Logs können in Dateien gespeichert werden  
✅ **Performance** - DEBUG-Logs werden nur ausgeführt wenn Level aktiv  
✅ **Filterbar** - Einfache Kontrolle über Verbosity  
✅ **Rückwärts-kompatibel** - Existierender Code funktioniert weiter  
✅ **Strukturiert** - Automatische Timestamps und Modul-Informationen  

## Beispiel Log-Datei

```
[2026-01-27 14:23:10][INFO   ][plagih] Starting test run: MTC200_RMSE_scratch
[2026-01-27 14:23:10][DEBUG  ][plagih] Options - chained_on=False, simplicate=False
[2026-01-27 14:23:11][INFO   ][plagih] Preparing to create first Generation. Gen 0.
[2026-01-27 14:23:11][DEBUG  ][plagih] Created tree with 15 nodes
[2026-01-27 14:23:12][INFO   ][plagih] Created 10/3 (10 unique) in generation 0
[2026-01-27 14:23:12][WARNING][plagih] (w) Tree too complex: 52 > 50, pruning 2 nodes
```

## Migration bestehender Prints

Bestehende `printpl()`, `printez()`, `print_warning()` Aufrufe funktionieren **automatisch** mit dem neuen System.

Für neue Code-Stellen empfohlen:
```python
# Statt:
printpl('i', f'Generation {gen} completed')

# Besser:
from plagih.logging_utils import log_info
log_info("Generation %d completed", gen)
```

## Best Practices

1. **Setup einmalig** zu Beginn deines Scripts
2. **log_debug()** für interne Details
3. **log_info()** für wichtige Meilensteine
4. **log_warning()** für Probleme, die nicht kritisch sind
5. **printpl()** wenn du sowohl Console als auch Log brauchst
