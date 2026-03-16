"""
Pickle performance benchmark for Candidate/Node/TaskResult objects.

Measures pickle size and serialization time for typical GP objects.
Helps identify serialization bottlenecks in the parallel pipeline.

Run directly:
    python plagih/test/benchmarks/bench_pickle.py
"""

import pickle
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import sympy

from plagih.parallel import TaskResult
from plagih.trees import Add, Candidate, Cos, Mul, Number, Sin, Symbol


def _build_test_tree():
    """Build a tree of typical GP complexity."""
    x = Symbol(sympy.Symbol("cartPos"))
    y = Symbol(sympy.Symbol("cartVel"))
    return Add(Sin(Mul(x, Number(2.5))), Cos(Add(y, Number(1.3))))


def _build_test_batch(n=25):
    """Build a batch of TaskResults like a worker would return."""
    x = Symbol(sympy.Symbol("cartPos"))
    y = Symbol(sympy.Symbol("cartVel"))
    batch = []
    for i in range(n):
        t = Add(Sin(Mul(x, Number(float(i)))), Cos(Add(y, Number(float(i) * 0.1))))
        c = Candidate(t, fitness=float(i) * 0.1, parsimony=i % 10 + 3, tag="mutation")
        r = TaskResult(
            candidates=[c],
            lut_tree_entries={},
            lut_symex_entries={},
            timing={"total": 0.03},
            error=None,
            tag="mutation",
        )
        batch.append(r)
    return batch


N_ITERATIONS = 100


def bench_pickle_sizes():
    """Measure pickle sizes of Candidate, Tree, TaskResult, and batch."""
    tree = _build_test_tree()
    cand = Candidate(tree, fitness=1.23, parsimony=7, tag="test")
    result = TaskResult(
        candidates=[cand],
        lut_tree_entries={},
        lut_symex_entries={},
        timing={"create": 0.01, "evaluate": 0.02, "total": 0.03},
        error=None,
        tag="mutation",
    )
    batch = _build_test_batch(25)

    sz_tree = len(pickle.dumps(tree))
    sz_cand = len(pickle.dumps(cand))
    sz_result = len(pickle.dumps(result))
    sz_batch = len(pickle.dumps(batch))

    print(f"\nTree pickle size:          {sz_tree:,} bytes")
    print(f"Candidate pickle size:     {sz_cand:,} bytes")
    print(f"TaskResult pickle size:    {sz_result:,} bytes")
    print(f"Batch (25 results) size:   {sz_batch:,} bytes")

    assert sz_tree > 0
    assert sz_cand > sz_tree
    assert sz_batch > sz_result


def bench_pickle_roundtrip_time():
    """Measure pickle dumps/loads time for a typical batch."""
    batch = _build_test_batch(25)
    data = pickle.dumps(batch)

    t0 = time.perf_counter()
    for _ in range(N_ITERATIONS):
        pickle.dumps(batch)
    t_dumps = (time.perf_counter() - t0) / N_ITERATIONS

    t0 = time.perf_counter()
    for _ in range(N_ITERATIONS):
        pickle.loads(data)
    t_loads = (time.perf_counter() - t0) / N_ITERATIONS

    print(f"\nPickle dumps (batch of 25): {t_dumps * 1000:.1f}ms")
    print(f"Pickle loads (batch of 25): {t_loads * 1000:.1f}ms")
    print(f"Total roundtrip per batch:  {(t_dumps + t_loads) * 1000:.1f}ms")
    print(f"Est. overhead for 4 batches: {(t_dumps + t_loads) * 4 * 1000:.1f}ms")


def bench_single_candidate_pickle_time():
    """Measure pickle time for a single candidate."""
    tree = _build_test_tree()
    cand = Candidate(tree, fitness=1.23, parsimony=7, tag="test")

    t0 = time.perf_counter()
    for _ in range(N_ITERATIONS):
        pickle.dumps(cand)
    t_single = (time.perf_counter() - t0) / N_ITERATIONS

    print(f"\nPickle dumps (1 candidate): {t_single * 1000:.1f}ms")
    print(f"Est. overhead for 100 individual tasks: {t_single * 2 * 100 * 1000:.1f}ms")


def bench_lightweight_dict_comparison():
    """Compare pickle size of full Candidate vs lightweight dict."""
    tree = _build_test_tree()
    cand = Candidate(tree, fitness=1.23, parsimony=7, tag="test")

    tree_str = tree.represent_str(show_all=False)
    lightweight = {"tree_str": tree_str, "fitness": 1.23, "parsimony": 7, "tag": "test"}

    sz_cand = len(pickle.dumps(cand))
    sz_light = len(pickle.dumps(lightweight))

    batch = _build_test_batch(25)
    light_batch = [
        {
            "tree_str": f"Add(Sin(Mul(cartPos, {i})), Cos(Add(cartVel, {i * 0.1})))",
            "fitness": float(i) * 0.1,
            "parsimony": i % 10 + 3,
            "tag": "mut",
        }
        for i in range(25)
    ]

    data_full = pickle.dumps(batch)
    data_light = pickle.dumps(light_batch)

    t0 = time.perf_counter()
    for _ in range(N_ITERATIONS):
        pickle.dumps(batch)
    t_full = (time.perf_counter() - t0) / N_ITERATIONS

    t0 = time.perf_counter()
    for _ in range(N_ITERATIONS):
        pickle.dumps(light_batch)
    t_light = (time.perf_counter() - t0) / N_ITERATIONS

    print(f"\nCandidate pickle:   {sz_cand:,} bytes")
    print(f"Lightweight dict:   {sz_light:,} bytes")
    print(f"Batch full:         {len(data_full):,} bytes, {t_full * 1000:.1f}ms")
    print(f"Batch lightweight:  {len(data_light):,} bytes, {t_light * 1000:.1f}ms")
    print(f"Ratio: {len(data_full) / len(data_light):.1f}x size, {t_full / t_light:.1f}x time")


if __name__ == "__main__":
    print("=" * 60)
    print("plagih GP — Pickle Performance Benchmark")
    print("=" * 60)
    bench_pickle_sizes()
    bench_pickle_roundtrip_time()
    bench_single_candidate_pickle_time()
    bench_lightweight_dict_comparison()
