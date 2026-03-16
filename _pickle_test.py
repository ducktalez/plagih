"""Quick check: how expensive is pickling Candidate/Node/TaskResult objects?"""

import pickle
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import sympy

from plagih.parallel import TaskResult
from plagih.trees import Add, Candidate, Cos, Mul, Number, Sin, Symbol

# Build a tree of typical complexity
x = Symbol(sympy.Symbol("cartPos"))
y = Symbol(sympy.Symbol("cartVel"))
tree = Add(Sin(Mul(x, Number(2.5))), Cos(Add(y, Number(1.3))))

# Create a Candidate
cand = Candidate(tree, fitness=1.23, parsimony=7, tag="test")

# Create a TaskResult with one candidate
result = TaskResult(
    candidates=[cand],
    lut_tree_entries={},
    lut_symex_entries={},
    timing={"create": 0.01, "evaluate": 0.02, "total": 0.03},
    error=None,
    tag="mutation",
)

# Measure pickle sizes
data = pickle.dumps(cand)
print(f"Candidate pickle size: {len(data):,} bytes")

data = pickle.dumps(tree)
print(f"Tree pickle size:      {len(data):,} bytes")

data = pickle.dumps(result)
print(f"TaskResult pickle size:{len(data):,} bytes")

# Build a batch of 25 results (typical batch for 4 workers, 100 pop)
batch = []
for i in range(25):
    t = Add(Sin(Mul(x, Number(float(i)))), Cos(Add(y, Number(float(i) * 0.1))))
    c = Candidate(t, fitness=float(i) * 0.1, parsimony=i % 10 + 3, tag="mutation")
    r = TaskResult(
        candidates=[c], lut_tree_entries={}, lut_symex_entries={}, timing={"total": 0.03}, error=None, tag="mutation"
    )
    batch.append(r)

data = pickle.dumps(batch)
print(f"Batch (25 results) pickle size: {len(data):,} bytes")

# Measure pickle TIME
N = 100
t0 = time.perf_counter()
for _ in range(N):
    pickle.dumps(batch)
t_dumps = (time.perf_counter() - t0) / N

t0 = time.perf_counter()
for _ in range(N):
    pickle.loads(data)
t_loads = (time.perf_counter() - t0) / N

print(f"\nPickle dumps (batch of 25): {t_dumps * 1000:.1f}ms")
print(f"Pickle loads (batch of 25): {t_loads * 1000:.1f}ms")
print(f"Total roundtrip per batch:  {(t_dumps + t_loads) * 1000:.1f}ms")
print(f"Est. overhead for 4 batches: {(t_dumps + t_loads) * 4 * 1000:.1f}ms")

# Compare: single candidate
data_single = pickle.dumps(cand)
t0 = time.perf_counter()
for _ in range(N):
    pickle.dumps(cand)
t_single = (time.perf_counter() - t0) / N

print(f"\nPickle dumps (1 candidate): {t_single * 1000:.1f}ms")
print(f"Est. overhead for 100 individual tasks: {t_single * 2 * 100 * 1000:.1f}ms")

# Test: what if we strip sympy and send just strings?
tree_str = tree.represent_str(show_all=False)
lightweight = {"tree_str": tree_str, "fitness": 1.23, "parsimony": 7, "tag": "test"}
data_light = pickle.dumps(lightweight)
print(f"\nLightweight dict pickle size: {len(data_light):,} bytes (vs {len(pickle.dumps(cand)):,} for Candidate)")

light_batch = [
    {
        "tree_str": f"Add(Sin(Mul(cartPos, {i})), Cos(Add(cartVel, {i * 0.1})))",
        "fitness": float(i) * 0.1,
        "parsimony": i % 10 + 3,
        "tag": "mut",
    }
    for i in range(25)
]
data_lb = pickle.dumps(light_batch)
t0 = time.perf_counter()
for _ in range(N):
    pickle.dumps(light_batch)
t_lb = (time.perf_counter() - t0) / N
print(f"Lightweight batch (25 dicts): {len(data_lb):,} bytes, {t_lb * 1000:.1f}ms dumps")
print(f"Ratio: {len(data) / len(data_lb):.1f}x size, {t_dumps / t_lb:.1f}x time")
