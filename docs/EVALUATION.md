# Unified Evaluation System

Plagih bietet ein optionales, vereinheitlichtes Evaluierungssystem, das alle drei
Evaluierungsmethoden unter einer einheitlichen API zusammenfasst.

## Überblick

### Die drei Evaluierungsmethoden

| Methode | Beschreibung | Geschwindigkeit | Anwendungsfall |
|---------|--------------|-----------------|----------------|
| `get_sympy_expr()` | Symbolische SymPy-Evaluierung | Langsam | Analyse, Export, Vereinfachung |
| `eval_predict_numpy_now()` | Direkte NumPy-Evaluierung (eager) | Schnell | Fitness-Berechnung, Debugging |
| `eval_np_lambdas()` | Lambda-basierte Evaluierung (lazy) | Mittel | Wiederverwendung, Graph-Aufbau |

### Warum EvaluationContext?

Das `EvaluationContext`-System bietet:

- **Einheitliche Schnittstelle**: Eine API für alle drei Methoden
- **Optionales LUT-Caching**: Automatisches Caching von Ergebnissen
- **Multi-Mode-Evaluation**: Mehrere Modi gleichzeitig evaluieren
- **Statistiken**: Tracking von Cache-Hits, Evaluierungen, etc.
- **Backward-Kompatibilität**: Alte Methoden funktionieren unverändert

## Quick Start

### Einfachste Nutzung

```python
from plagih.evaluation_context import evaluate_tree

# One-shot evaluation
result = evaluate_tree(tree, df, mode='numpy_eager')
```

### Mit EvaluationContext

```python
from plagih.evaluation_context import EvaluationContext

# Einfacher Fall: NumPy-Evaluierung
ctx = EvaluationContext(modes=['numpy_eager'])
result = ctx.evaluate(tree, df)

# Mehrere Modi gleichzeitig
ctx = EvaluationContext(modes=['sympy', 'numpy_eager', 'numpy_lambda'])
results = ctx.evaluate(tree, df)
# results = {'sympy': <expr>, 'numpy_eager': <array>, 'numpy_lambda': <callable>}
```

## Detaillierte Nutzung

### EvaluationContext erstellen

```python
from plagih.evaluation_context import EvaluationContext, create_context

# Vollständig konfiguriert
ctx = EvaluationContext(
    modes=['numpy_eager', 'numpy_lambda'],  # Welche Modi
    use_lut=True,                           # LUT-Caching aktivieren
    df=my_dataframe                         # Optional: DataFrame speichern
)

# Factory-Funktion für einzelnen Modus
ctx = create_context('numpy_eager', use_lut=True)
```

### Fluent Interface

```python
ctx = (EvaluationContext()
       .with_modes(['numpy_eager', 'numpy_lambda'])
       .with_df(df)
       .with_lut(True))
```

### Convenience-Methoden

```python
ctx = EvaluationContext(modes=['sympy', 'numpy_eager', 'numpy_lambda'])

# Einzelne Modi
sympy_result = ctx.eval_sympy(tree)
numpy_result = ctx.eval_numpy(tree, df)
lambda_fn = ctx.eval_lambda(tree)

# Alle Modi auf einmal (mit Error-Handling)
result = ctx.eval_all(tree, df)
# result.sympy, result.numpy_eager, result.numpy_lambda, result.errors
```

## LUT-Caching

### Aktivieren/Deaktivieren

```python
# Mit Caching (default)
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=True)

# Ohne Caching
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=False)
```

### Cache-Statistiken

```python
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=True)

# Mehrere Evaluierungen
ctx.evaluate(tree, df)
ctx.evaluate(tree, df)  # Cache-Hit!
ctx.evaluate(tree, df)  # Cache-Hit!

# Statistiken abrufen
print(ctx.summary())
# EvaluationContext Statistics:
#   Modes: ['numpy_eager']
#   LUT enabled: True
#   numpy_eager:
#     Evaluations: 3, Cache hits: 2 (66.7%)
#     Errors: 0, Cache size: 1

# Detaillierte Stats
stats = ctx.get_stats()
hit_rate = ctx.get_cache_hit_rate('numpy_eager')
cache_size = ctx.get_cache_size()
```

### Cache leeren

```python
# Alle Modi leeren
ctx.clear_cache()

# Nur bestimmten Modus leeren
ctx.clear_cache('numpy_eager')
```

## Backward-Kompatibilität

Die bestehenden Methoden funktionieren weiterhin unverändert:

```python
# Diese funktionieren wie bisher!
sy_expr = tree.get_sympy_expr()
np_result = tree.eval_predict_numpy_now(df)
lambda_fn = tree.eval_np_lambdas()
```

Das EvaluationContext-System **delegiert** zu diesen Methoden - es ersetzt sie nicht!

## Integration mit Node-Klasse

Optional kann eine `evaluate_unified` Methode zur Node-Klasse hinzugefügt werden:

```python
from plagih.trees import Node
from plagih.evaluation_context import add_unified_evaluation_to_node

# Einmalig aufrufen
add_unified_evaluation_to_node(Node)

# Danach verfügbar für alle Nodes
ctx = EvaluationContext(modes=['numpy_eager'])
result = tree.evaluate_unified(ctx, df)
```

## Praktische Beispiele

### Fitness-Berechnung mit Caching

```python
from plagih.evaluation_context import EvaluationContext

# Context für die gesamte Generation
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=True)

for tree in population:
    # Gleiche Bäume werden nur einmal evaluiert!
    result = ctx.evaluate(tree, df_train)
    fitness = calculate_fitness(result, target)

# Am Ende: Cache-Statistiken
print(f"Cache hit rate: {ctx.get_cache_hit_rate('numpy_eager'):.1%}")
```

### Vergleich aller drei Modi

```python
from plagih.evaluation_context import EvaluationContext

ctx = EvaluationContext()
result = ctx.eval_all(tree, df)

if result.errors:
    print(f"Errors: {result.errors}")
else:
    # Vergleiche Ergebnisse
    print(f"SymPy: {result.sympy}")
    print(f"NumPy eager: {result.numpy_eager[:5]}...")
    
    # Lambda ausführen
    lambda_result = result.numpy_lambda(df)
    print(f"Lambda: {lambda_result[:5]}...")
    
    # Ergebnisse sollten übereinstimmen
    assert np.allclose(result.numpy_eager, lambda_result)
```

## Zukünftige Features

### Gradient Tracking (Placeholder)

```python
# Vorbereitet für zukünftige JAX/PyTorch Integration
ctx = EvaluationContext(track_gradients=True)
ctx.enable_gradient_tracking()  # Gibt FutureWarning aus

# Später:
gradients = ctx.get_gradients()
```

### Parallelisierung

Das EvaluationContext-System ist so designed, dass es später leicht parallelisiert
werden kann:

```python
# Zukünftig möglich:
from concurrent.futures import ThreadPoolExecutor

ctx = EvaluationContext(modes=['numpy_eager'])

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(ctx.evaluate, tree, df) 
        for tree in population
    ]
    results = [f.result() for f in futures]
```

## API-Referenz

### EvaluationContext

```python
class EvaluationContext:
    def __init__(
        self,
        modes: List[str] = ['numpy_eager'],
        use_lut: bool = True,
        track_gradients: bool = False,
        df: pd.DataFrame = None
    )
    
    # Fluent interface
    def with_modes(self, modes: List[str]) -> EvaluationContext
    def with_df(self, df: pd.DataFrame) -> EvaluationContext
    def with_lut(self, use_lut: bool) -> EvaluationContext
    
    # Main evaluation
    def evaluate(self, node, df=None, single_mode=None) -> Union[Any, Dict]
    
    # Convenience methods
    def eval_sympy(self, node) -> sympy.Basic
    def eval_numpy(self, node, df) -> np.ndarray
    def eval_lambda(self, node) -> Callable
    def eval_all(self, node, df) -> EvaluationResult
    
    # Cache management
    def clear_cache(self, mode=None) -> None
    def get_cache_size(self) -> Dict[str, int]
    
    # Statistics
    def get_stats(self) -> Dict[str, Dict[str, int]]
    def get_cache_hit_rate(self, mode: str) -> float
    def summary(self) -> str
```

### Utility Functions

```python
# Factory for single-mode context
create_context(mode='numpy_eager', use_lut=True, df=None) -> EvaluationContext

# One-shot evaluation
evaluate_tree(node, df=None, mode='numpy_eager', use_lut=False) -> Any

# Add method to Node class
add_unified_evaluation_to_node(node_class) -> None
```

### EvaluationResult

```python
@dataclass
class EvaluationResult:
    sympy: Optional[sympy.Basic]
    numpy_eager: Optional[np.ndarray]
    numpy_lambda: Optional[Callable]
    errors: Dict[str, str]
    
    def get(self, mode: str) -> Any
    def has_error(self, mode: str) -> bool
    def successful_modes(self) -> List[str]
```
