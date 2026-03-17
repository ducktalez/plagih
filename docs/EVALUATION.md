# Unified Evaluation System

Plagih provides an optional, unified evaluation system that combines all three
evaluation methods under a single API.

## Overview

### The three evaluation methods

| Method | Description | Speed | Use case |
|--------|-------------|-------|----------|
| `get_sympy_expr()` | Symbolic SymPy evaluation | Slow | Analysis, export, simplification |
| `eval_predict_numpy_now()` | Direct NumPy evaluation (eager) | Fast | Fitness calculation, debugging |
| `eval_np_lambdas()` | Lambda-based evaluation (lazy) | Medium | Reuse, graph construction |

### Why EvaluationContext?

The `EvaluationContext` system provides:

- **Unified interface**: One API for all three methods
- **Optional LUT caching**: Automatic result caching
- **Multi-mode evaluation**: Evaluate multiple modes simultaneously
- **Statistics**: Tracking of cache hits, evaluations, etc.
- **Backward compatibility**: Existing methods continue to work unchanged

## Quick Start

### Simplest usage

```python
from plagih.evaluation_context import evaluate_tree

# One-shot evaluation
result = evaluate_tree(tree, df, mode='numpy_eager')
```

### With EvaluationContext

```python
from plagih.evaluation_context import EvaluationContext

# Simple case: NumPy evaluation
ctx = EvaluationContext(modes=['numpy_eager'])
result = ctx.evaluate(tree, df)

# Multiple modes at once
ctx = EvaluationContext(modes=['sympy', 'numpy_eager', 'numpy_lambda'])
results = ctx.evaluate(tree, df)
# results = {'sympy': <expr>, 'numpy_eager': <array>, 'numpy_lambda': <callable>}
```

## Detailed Usage

### Creating an EvaluationContext

```python
from plagih.evaluation_context import EvaluationContext, create_context

# Fully configured
ctx = EvaluationContext(
    modes=['numpy_eager', 'numpy_lambda'],  # Which modes
    use_lut=True,                           # Enable LUT caching
    df=my_dataframe                         # Optional: store DataFrame
)

# Factory function for a single mode
ctx = create_context('numpy_eager', use_lut=True)
```

### Fluent Interface

```python
ctx = (EvaluationContext()
       .with_modes(['numpy_eager', 'numpy_lambda'])
       .with_df(df)
       .with_lut(True))
```

### Convenience Methods

```python
ctx = EvaluationContext(modes=['sympy', 'numpy_eager', 'numpy_lambda'])

# Individual modes
sympy_result = ctx.eval_sympy(tree)
numpy_result = ctx.eval_numpy(tree, df)
lambda_fn = ctx.eval_lambda(tree)

# All modes at once (with error handling)
result = ctx.eval_all(tree, df)
# result.sympy, result.numpy_eager, result.numpy_lambda, result.errors
```

## LUT Caching

### Enable / Disable

```python
# With caching (default)
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=True)

# Without caching
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=False)
```

### Cache Statistics

```python
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=True)

# Multiple evaluations
ctx.evaluate(tree, df)
ctx.evaluate(tree, df)  # Cache hit!
ctx.evaluate(tree, df)  # Cache hit!

# Retrieve statistics
print(ctx.summary())
# EvaluationContext Statistics:
#   Modes: ['numpy_eager']
#   LUT enabled: True
#   numpy_eager:
#     Evaluations: 3, Cache hits: 2 (66.7%)
#     Errors: 0, Cache size: 1

# Detailed stats
stats = ctx.get_stats()
hit_rate = ctx.get_cache_hit_rate('numpy_eager')
cache_size = ctx.get_cache_size()
```

### Clear Cache

```python
# Clear all modes
ctx.clear_cache()

# Clear specific mode only
ctx.clear_cache('numpy_eager')
```

## Backward Compatibility

Existing methods continue to work unchanged:

```python
# These still work as before!
sy_expr = tree.get_sympy_expr()
np_result = tree.eval_predict_numpy_now(df)
lambda_fn = tree.eval_np_lambdas()
```

The EvaluationContext system **delegates** to these methods — it does not replace them.

## Node Class Integration

Optionally, an `evaluate_unified` method can be added to the Node class:

```python
from plagih.trees import Node
from plagih.evaluation_context import add_unified_evaluation_to_node

# Call once
add_unified_evaluation_to_node(Node)

# Then available for all nodes
ctx = EvaluationContext(modes=['numpy_eager'])
result = tree.evaluate_unified(ctx, df)
```

## Practical Examples

### Fitness Calculation with Caching

```python
from plagih.evaluation_context import EvaluationContext

# Context for the entire generation
ctx = EvaluationContext(modes=['numpy_eager'], use_lut=True)

for tree in population:
    # Identical trees are evaluated only once!
    result = ctx.evaluate(tree, df_train)
    fitness = calculate_fitness(result, target)

# Cache statistics
print(f"Cache hit rate: {ctx.get_cache_hit_rate('numpy_eager'):.1%}")
```

### Comparing All Three Modes

```python
from plagih.evaluation_context import EvaluationContext

ctx = EvaluationContext()
result = ctx.eval_all(tree, df)

if result.errors:
    print(f"Errors: {result.errors}")
else:
    print(f"SymPy: {result.sympy}")
    print(f"NumPy eager: {result.numpy_eager[:5]}...")

    lambda_result = result.numpy_lambda(df)
    print(f"Lambda: {lambda_result[:5]}...")

    # Results should match
    assert np.allclose(result.numpy_eager, lambda_result)
```

## Future Features

### Gradient Tracking (Placeholder)

```python
# Prepared for future JAX/PyTorch integration
ctx = EvaluationContext(track_gradients=True)
ctx.enable_gradient_tracking()  # Emits FutureWarning

# Later:
gradients = ctx.get_gradients()
```

### Parallelization

The EvaluationContext system is designed for easy future parallelization:

```python
# Possible in the future:
from concurrent.futures import ThreadPoolExecutor

ctx = EvaluationContext(modes=['numpy_eager'])

with ThreadPoolExecutor() as executor:
    futures = [
        executor.submit(ctx.evaluate, tree, df)
        for tree in population
    ]
    results = [f.result() for f in futures]
```

## API Reference

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
