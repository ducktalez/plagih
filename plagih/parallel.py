"""
Parallel execution engine for plagih GP.

Provides a declarative Strategy API and parallel/sequential execution
for population generation. Workers receive evolve, df_train, and
pop_genepool as global variables via ProcessPoolExecutor initializer
(no per-task serialization overhead for large objects).

Design decisions documented inline.

Usage:
    # Declarative strategy list
    gp.run_generation([
        Strategy("reproduction", rate=0.2, tournament_n=3),
        Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3),
        Strategy("random_new", rate=0.2, depths=[2, 3, 4]),
        Strategy("crossover", rate=0.2, tournament_n=3),
    ])

    # Custom strategy
    def my_strategy(evolve, pop_genepool, **params):
        tree = selection_tournament(pop_genepool, n=3)
        return evolve.evolve_mutate_point(tree)

    gp.register_strategy("my_mutation", my_strategy)
"""

import atexit
import os
import random
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from typing import Any, Callable, Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from plagih.util import (
    FLOAT_PRECISION,
    SympyError,
    TreeError,
    TreeLutError,
    TreeSizeError,
    print_caution,
    print_warning,
    printpl,
)

# =============================================================================
# Strategy Dataclass
# =============================================================================


@dataclass
class Strategy:
    """Declarative specification of an evolution strategy.

    Args:
        name: Name of a builtin strategy or a registered custom strategy.
        rate: Fraction of pop_max_size to create (0.0 to 1.0).
        crossover: If True, strategy returns two trees per call.
        simplicate: If True, apply tree simplification before evaluation.
        params: Additional keyword arguments passed to the strategy function.

    Example:
        Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3)
        Strategy("crossover", rate=0.2, tournament_n=3)
    """

    name: str
    rate: float = 0.0
    crossover: bool = False
    simplicate: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, name: str, rate: float = 0.0, crossover: bool = False, simplicate: bool = False, **params):
        self.name = name
        self.rate = rate
        self.crossover = crossover
        self.simplicate = simplicate
        self.params = params


# =============================================================================
# Task and Result Dataclasses
# =============================================================================


@dataclass
class TaskSpec:
    """Specification for a single tree-creation task sent to a worker.

    Attributes:
        strategy_name: Which strategy function to call.
        strategy_params: Parameters for the strategy function.
        crossover: Whether this task produces two trees.
        simplicate: Whether to simplify trees before evaluation.
        tag: Label for tracking (usually strategy name).
        task_index: Unique index for seed computation.
        seed: Optional base seed for reproducibility.
        selected_trees: Pre-selected parent trees (parallel mode only).
            When set, workers use these instead of pop_genepool, eliminating
            the need for _update_worker_state IPC.
    """

    strategy_name: str
    strategy_params: Dict[str, Any]
    crossover: bool
    simplicate: bool
    tag: str
    task_index: int
    seed: Optional[int] = None
    selected_trees: Optional[List] = None


@dataclass
class TaskResult:
    """Result returned by a worker for a single task.

    Attributes:
        candidates: List of Candidate objects (1 for normal, 2 for crossover).
        lut_tree_entries: Worker-local LUT entries to merge (tree_id -> info dict).
        lut_symex_entries: Worker-local symex->fitness entries to merge.
        timing: Per-phase timing breakdown.
        error: Error info if task failed, None on success.
        tag: Strategy tag for tracking.
    """

    candidates: list  # List[Candidate]
    lut_tree_entries: Dict[str, dict]
    lut_symex_entries: dict
    timing: Dict[str, float]
    error: Optional[str]
    tag: str
    debug: Optional[Dict[str, Any]] = None


# =============================================================================
# Performance Tracker
# =============================================================================


class PerformanceTracker:
    """Aggregates per-strategy timing and error statistics.

    Collects TaskResults and computes summary statistics per strategy tag.
    Results can be exported as dict for inclusion in monitor_df.
    """

    def __init__(self):
        self._results: Dict[str, List[Dict[str, float]]] = {}  # tag -> [timing dicts]
        self._errors: Dict[str, int] = {}  # tag -> error count
        self._successes: Dict[str, int] = {}  # tag -> success count
        self._generation_time: float = 0.0

    def record(self, result: TaskResult):
        """Record a single task result."""
        tag = result.tag
        if tag not in self._results:
            self._results[tag] = []
            self._errors[tag] = 0
            self._successes[tag] = 0

        if result.error is not None:
            self._errors[tag] += 1
        else:
            self._successes[tag] += 1
            self._results[tag].append(result.timing)

    def set_generation_time(self, elapsed: float):
        """Set the total wall-clock time for the generation."""
        self._generation_time = elapsed

    @property
    def total_ok(self) -> int:
        """Total number of successful tasks across all strategies."""
        return sum(self._successes.values())

    @property
    def total_fail(self) -> int:
        """Total number of failed tasks across all strategies."""
        return sum(self._errors.values())

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics per strategy tag.

        Returns:
            Dict with keys like 'strategy_{tag}_avg_time', 'strategy_{tag}_fail_rate', etc.
            Also includes 'generation_total_time'.
        """
        out = {"generation_total_time": self._generation_time}
        for tag in self._results:
            timings = self._results[tag]
            total_tasks = self._successes[tag] + self._errors[tag]
            if timings:
                totals = [t.get("total", 0.0) for t in timings]
                out[f"strategy_{tag}_avg_time"] = np.mean(totals)
                out[f"strategy_{tag}_median_time"] = np.median(totals)
            else:
                out[f"strategy_{tag}_avg_time"] = 0.0
                out[f"strategy_{tag}_median_time"] = 0.0
            out[f"strategy_{tag}_success"] = self._successes[tag]
            out[f"strategy_{tag}_fails"] = self._errors[tag]
            out[f"strategy_{tag}_fail_rate"] = self._errors[tag] / total_tasks if total_tasks > 0 else 0.0
        return out

    def print_summary(self):
        """Print a human-readable performance summary."""
        summary = self.summary()
        printpl("pp", "=== Strategy Performance Summary ===")
        printpl("pp", f"  Generation total: {summary['generation_total_time']:.3f}s")
        tags = sorted(set(t.rsplit("_", 2)[0].replace("strategy_", "") for t in summary if t.startswith("strategy_")))
        # Deduplicate properly
        seen_tags = set()
        for tag in self._results:
            if tag in seen_tags:
                continue
            seen_tags.add(tag)
            avg = summary.get(f"strategy_{tag}_avg_time", 0)
            med = summary.get(f"strategy_{tag}_median_time", 0)
            ok = summary.get(f"strategy_{tag}_success", 0)
            fail = summary.get(f"strategy_{tag}_fails", 0)
            rate = summary.get(f"strategy_{tag}_fail_rate", 0)
            printpl(
                "pp",
                f"  {tag:20s}: avg={avg * 1000:6.1f}ms  med={med * 1000:6.1f}ms  "
                f"ok={ok:3d}  fail={fail:3d}  fail_rate={rate:.1%}",
            )

    def reset(self):
        """Reset all tracking data for the next generation."""
        self._results.clear()
        self._errors.clear()
        self._successes.clear()
        self._generation_time = 0.0


@dataclass
class WorkerInitResources:
    """Parent-side resources owned for worker initialization."""

    df_train_shm: Optional[shared_memory.SharedMemory] = None
    uses_shared_df_train: bool = False


DEFAULT_TASKS_PER_BATCH = 128
PROGRESS_TIMEOUT_SECONDS = 120.0
DEBUG_TRACEBACK_LINES = 20


# =============================================================================
# Worker Global State
# =============================================================================
# Design: Global variables in worker via initializer, not per-task serialization.
# df_train and evolve are copied once per pool-start, not per task (~1000 tasks).

_worker_evolve = None
_worker_df_train = None
_worker_pop_genepool = None
_worker_paretofront = None
_worker_eval_autocast = None
_worker_eval_error_metric = None
_worker_allow_chain = None
_worker_target_column = None
_worker_nodes_max = None
_worker_complexity_metric = None
_worker_strategy_registry = None
_worker_df_train_shm = None
_worker_df_train_shm_registered = False


def _close_worker_df_train_shm():
    """Close the worker-local shared memory handle if one is attached."""
    global _worker_df_train_shm

    shm = _worker_df_train_shm
    if shm is not None:
        try:
            shm.close()
        except FileNotFoundError:
            pass
        _worker_df_train_shm = None


def _is_shared_memory_compatible_df(df_train) -> bool:
    """Return True if df_train can be represented as one numeric ndarray."""
    if df_train is None or len(df_train.columns) == 0 or len(df_train.index) == 0:
        return False

    arr = df_train.to_numpy(copy=False)
    if arr.size == 0 or arr.nbytes == 0:
        return False

    return bool(np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_))


def _create_df_train_shared_memory(df_train) -> Tuple[Optional[Tuple], WorkerInitResources]:
    """Create shared memory for df_train and return worker metadata plus resources."""
    resources = WorkerInitResources()
    if not _is_shared_memory_compatible_df(df_train):
        return None, resources

    arr = np.ascontiguousarray(df_train.to_numpy(copy=False))
    if arr.size == 0 or arr.nbytes == 0:
        return None, resources

    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr

    df_train_shm_meta = (
        shm.name,
        arr.shape,
        arr.dtype.str,
        df_train.columns.copy(),
        df_train.index.copy(),
    )
    resources.df_train_shm = shm
    resources.uses_shared_df_train = True
    return df_train_shm_meta, resources


def cleanup_worker_init_resources(resources: Optional[WorkerInitResources]) -> None:
    """Release parent-side resources created for worker initialization."""
    if resources is None or resources.df_train_shm is None:
        return

    try:
        resources.df_train_shm.close()
    finally:
        try:
            resources.df_train_shm.unlink()
        except FileNotFoundError:
            pass
        resources.df_train_shm = None
        resources.uses_shared_df_train = False


def build_worker_init_config(
    evolve,
    df_train,
    pop_genepool,
    paretofront,
    eval_autocast,
    eval_error_metric,
    allow_chain,
    target_column,
    nodes_max,
    complexity_metric,
    strategy_registry,
) -> Tuple[Tuple, WorkerInitResources]:
    """Build initializer args for a worker pool, using shared memory for df_train when possible."""
    df_train_shm_meta, resources = _create_df_train_shared_memory(df_train)
    df_train_init = None if df_train_shm_meta is not None else df_train
    initargs = (
        evolve,
        df_train_init,
        pop_genepool,
        paretofront,
        eval_autocast,
        eval_error_metric,
        allow_chain,
        target_column,
        nodes_max,
        complexity_metric,
        strategy_registry,
        df_train_shm_meta,
    )
    return initargs, resources


def _safe_tree_debug_id(tree) -> str:
    """Return a fast, robust identifier for debug output."""
    if tree is None:
        return "<none>"

    try:
        return tree.get_lut_id()
    except Exception:
        try:
            return repr(tree)
        except Exception:
            return f"<{type(tree).__name__}>"


def _task_debug_payload(task: TaskSpec, stage: str, trees=None, exc: Optional[BaseException] = None) -> Dict[str, Any]:
    """Build structured debug context for a failed task."""
    selected_tree_ids = []
    if task.selected_trees is not None:
        selected_tree_ids = [_safe_tree_debug_id(tree) for tree in task.selected_trees]

    tree_ids = []
    if trees:
        tree_ids = [_safe_tree_debug_id(tree) for tree in trees]

    payload: Dict[str, Any] = {
        "worker_pid": os.getpid(),
        "stage": stage,
        "strategy": task.strategy_name,
        "task_index": task.task_index,
        "seed": task.seed,
        "crossover": task.crossover,
        "simplicate": task.simplicate,
        "strategy_params": dict(task.strategy_params),
        "selected_tree_ids": selected_tree_ids,
        "tree_ids": tree_ids,
    }
    if exc is not None:
        payload["traceback"] = traceback.format_exc(limit=DEBUG_TRACEBACK_LINES)
    return payload


def _format_task_error_message(exc: BaseException, debug: Dict[str, Any]) -> str:
    """Format a concise but actionable error message for TaskResult.error."""
    selected = len(debug.get("selected_tree_ids", []))
    trees = len(debug.get("tree_ids", []))
    return (
        f"{type(exc).__name__}: {exc} "
        f"[stage={debug.get('stage')}, strategy={debug.get('strategy')}, "
        f"task_index={debug.get('task_index')}, seed={debug.get('seed')}, "
        f"selected={selected}, trees={trees}, pid={debug.get('worker_pid')}]"
    )


def _debug_should_print(result: TaskResult) -> bool:
    """Return True for failures that deserve explicit runtime diagnostics."""
    if result.error is None:
        return False
    return any(
        token in result.error for token in ("RecursionError", "WorkerCrash", "BatchTimeout", "BrokenProcessPool")
    )


def _print_parallel_failure_debug(result: TaskResult) -> None:
    """Print detailed diagnostics for severe worker failures."""
    if not _debug_should_print(result):
        return

    debug = result.debug or {}
    print_warning(
        "ww",
        "Parallel task failure: "
        f"tag={result.tag}, error={result.error}, "
        f"stage={debug.get('stage')}, task_index={debug.get('task_index')}, "
        f"strategy={debug.get('strategy')}, pid={debug.get('worker_pid')}",
    )
    tb = debug.get("traceback")
    if tb:
        print_warning("ww", f"Parallel task traceback:\n{tb}")


def _resolve_target_tasks_per_batch(n_tasks: int, n_workers: int) -> int:
    """Return a conservative runtime batch size.

    Smaller chunks avoid long silent waits in as_completed/wait, isolate bad
    tasks better, and matched the benchmark results more closely than one huge
    batch per worker.
    """
    env_value = os.environ.get("PLAGIH_TASKS_PER_BATCH")
    if env_value:
        try:
            return max(1, min(int(env_value), n_tasks))
        except ValueError:
            pass

    auto_floor = max(1, n_tasks // max(1, n_workers * 8))
    return max(auto_floor, min(DEFAULT_TASKS_PER_BATCH, n_tasks))


def _build_task_batches(tasks: List[TaskSpec], n_workers: int) -> List[List[TaskSpec]]:
    """Split tasks into mixed-strategy chunks for runtime execution."""
    if not tasks:
        return []

    target_batch_size = _resolve_target_tasks_per_batch(len(tasks), n_workers)
    n_batches = max(n_workers, int(np.ceil(len(tasks) / target_batch_size)))
    batches: List[List[TaskSpec]] = [[] for _ in range(n_batches)]
    for i, task in enumerate(tasks):
        batches[i % n_batches].append(task)
    return [batch for batch in batches if batch]


def _summarize_batch(batch: List[TaskSpec]) -> Dict[str, Any]:
    """Build a compact batch summary for timeout/crash diagnostics."""
    strategy_counts = dict(Counter(task.strategy_name for task in batch))
    return {
        "size": len(batch),
        "task_index_min": min((task.task_index for task in batch), default=None),
        "task_index_max": max((task.task_index for task in batch), default=None),
        "strategies": strategy_counts,
    }


def _progress_timeout_seconds() -> float:
    """Return the maximum allowed time without any finished batch."""
    env_value = os.environ.get("PLAGIH_PROGRESS_TIMEOUT_S")
    if env_value:
        try:
            return max(1.0, float(env_value))
        except ValueError:
            pass
    return PROGRESS_TIMEOUT_SECONDS


def _init_worker(
    evolve,
    df_train,
    pop_genepool,
    paretofront,
    eval_autocast,
    eval_error_metric,
    allow_chain,
    target_column,
    nodes_max,
    complexity_metric,
    strategy_registry,
    df_train_shm_meta=None,
):
    """Initialize global state in each worker process.

    Called once per worker by ProcessPoolExecutor(initializer=...).
    """
    global _worker_evolve, _worker_df_train, _worker_pop_genepool
    global _worker_paretofront, _worker_eval_autocast, _worker_eval_error_metric
    global _worker_allow_chain, _worker_target_column
    global _worker_nodes_max, _worker_complexity_metric, _worker_strategy_registry
    global _worker_df_train_shm, _worker_df_train_shm_registered

    _worker_evolve = evolve
    _close_worker_df_train_shm()
    if df_train_shm_meta is not None:
        shm_name, shape, dtype_str, columns, index = df_train_shm_meta
        _worker_df_train_shm = shared_memory.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_worker_df_train_shm.buf)
        _worker_df_train = pd.DataFrame(arr, columns=columns, index=index, copy=False)
        if not _worker_df_train_shm_registered:
            atexit.register(_close_worker_df_train_shm)
            _worker_df_train_shm_registered = True
    else:
        _worker_df_train = df_train
    _worker_pop_genepool = pop_genepool
    _worker_paretofront = paretofront
    _worker_eval_autocast = eval_autocast
    _worker_eval_error_metric = eval_error_metric
    _worker_allow_chain = allow_chain
    _worker_target_column = target_column
    _worker_nodes_max = nodes_max
    _worker_complexity_metric = complexity_metric
    _worker_strategy_registry = strategy_registry


def _update_worker_state(pop_genepool, paretofront):
    """Update worker state per generation (persistent pool mode).

    Only pop_genepool and paretofront change between generations.
    Called as a task on each worker before the actual batch.

    After deserialization, tree back-references (parent_node, root_node)
    are None (excluded from pickle for performance). We repair them here
    so strategies that need them (e.g. set_new_node with repair=True) work.
    """
    global _worker_pop_genepool, _worker_paretofront
    # Repair tree back-references stripped during pickling
    for candidate in pop_genepool:
        candidate.tree.repair_all()
    for candidate in paretofront:
        candidate.tree.repair_all()
    _worker_pop_genepool = pop_genepool
    _worker_paretofront = paretofront
    return True


# =============================================================================
# Standalone Tree Evaluation (extracted from ExplainableGP.tree_to_candidate)
# =============================================================================


def evaluate_tree_standalone(
    evotree,
    evolve,
    df_train,
    eval_autocast,
    eval_error_metric,
    target_column,
    nodes_max,
    complexity_metric,
    local_lut_tree,
    local_lut_symex,
    tag="",
    origin_tree=None,
    raise_if_useless=True,
):
    """Evaluate a tree into a Candidate, using worker-local LUTs.

    This is the extracted core of ExplainableGP.tree_to_candidate,
    operating without reference to a GP instance.

    Args:
        evotree: The tree Node to evaluate.
        evolve: Evolution instance (for pruning, force_input_node).
        df_train: Training DataFrame.
        eval_autocast: Autocast function for predictions.
        eval_error_metric: Error metric function(pred, true) -> float.
        target_column: Name of target column in df_train.
        nodes_max: Maximum node count.
        complexity_metric: Complexity metric name.
        local_lut_tree: Worker-local tree info LUT (mutated in place).
        local_lut_symex: Worker-local symex->fitness LUT (mutated in place).
        origin_tree: Optional reference tree for edit distance.
        raise_if_useless: Raise TreeSizeError for oversized trees.

    Returns:
        Candidate with tree, fitness, and parsimony.
    """
    # Lazy import to avoid circular dependency
    from plagih.trees import (
        Candidate,
        eval_parsimony,
    )

    # Prepare tree for evaluation
    evotree.force_input_node(evolve)
    evotree = evolve.evolve_prune_tree(evotree)
    evotree.repair_depth()

    tree_id = evotree.get_lut_id()

    if tree_id in local_lut_tree:
        sy_expr = local_lut_tree[tree_id].get("sy_expr")
        parsimony = local_lut_tree[tree_id].get("parsimony")
        fitness = local_lut_tree[tree_id].get("fitness")
        if any(v is None for v in [sy_expr, parsimony, fitness]):
            _err = local_lut_tree[tree_id].get("error")
            raise TreeLutError(f"Tree LUT Entry implies Problem: {_err}")
    else:
        local_lut_tree[tree_id] = {}

        parsimony = eval_parsimony(evotree, complexity_metric, origin_tree=origin_tree)
        if raise_if_useless and parsimony > nodes_max:
            err_txt = f"Tree too complex: {parsimony} > {nodes_max}"
            local_lut_tree[tree_id]["error"] = err_txt
            raise TreeSizeError(err_txt)

        try:
            sy_expr = evotree.get_sympy_expr()
        except SympyError as e:
            local_lut_tree[tree_id]["error"] = str(e)
            raise

        if sy_expr in local_lut_symex:
            fitness = local_lut_symex[sy_expr]
        else:
            true_values = df_train[target_column].to_numpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                np_results_raw = evotree.eval_predict_numpy_now(df_train)
                np_results = eval_autocast(np_results_raw)
                np_fitness = eval_error_metric(np_results, true_values)
                np_fitness = round(np_fitness, FLOAT_PRECISION)

                if "nan" in str(np_fitness) or np_fitness != np_fitness or np_fitness == float("inf"):
                    err_txt = "NaN in results"
                    local_lut_tree[tree_id]["error"] = err_txt
                    raise TreeError(err_txt)

            fitness = np_fitness
            local_lut_symex[sy_expr] = fitness

        local_lut_tree[tree_id]["sy_expr"] = sy_expr
        local_lut_tree[tree_id]["parsimony"] = parsimony
        local_lut_tree[tree_id]["fitness"] = fitness

    candidate = Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
    return candidate


# =============================================================================
# Builtin Strategy Functions
# =============================================================================
# All strategies are top-level functions (picklable).
# Signature: fn(evolve, pop_genepool, paretofront, allow_chain, **params) -> Node | Tuple[Node, Node]


def _strategy_reproduction(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a good tree via tournament selection and copy it."""
    pre = params.pop("_pre_selected", None)
    if pre:
        return pre[0]
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
    return selection_tournament(pop_genepool, n=tournament_n)


def _strategy_mutation(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate a branch."""
    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    depth_goal = params.get("depth_goal", 3)
    p_term = params.get("p_term", 0.3)
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_branch_depth(tree, depth_goal, allow_chain, p_term=p_term)


def _strategy_mutation_point(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate a single point."""
    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_point(tree)


def _strategy_mutation_branch_nodes(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate by target node count."""
    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    nodes_goal = params.get("nodes_goal", 4)
    p_term = params.get("p_term", 0.2)
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_branch_nodes(tree, nodes_goal, p_term=p_term)


def _strategy_mutation_filter(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and apply filter mutation (constant tuning)."""
    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_filter(tree)


def _strategy_random_new(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Create a new random tree with a random depth."""
    depths = params.get("depths", [2, 3, 4])
    p_term = params.get("p_term", 0.1)
    depth = np.random.choice(depths)
    return evolve.evolve_new_tree_depth(float, int(depth), p_term=p_term)


def _strategy_crossover(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select two trees and perform subtree crossover."""
    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    if pre:
        tree_a, tree_b = pre[0], pre[1]
    else:
        from plagih.trees import selection_tournament

        tree_a = selection_tournament(pop_genepool, n=tournament_n)
        tree_b = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_crossover(tree_a, tree_b)


def _strategy_simplicate(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and simplify it via sympy."""
    from plagih.trees import evolve_reduce_simplicate

    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    completely = params.get("completely", True)
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve_reduce_simplicate(tree, allow_chain, completely=completely)


def _strategy_pareto_revive(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Revive a random candidate from the Pareto front."""
    pre = params.pop("_pre_selected", None)
    if pre:
        return pre[0]
    import copy as _copy

    if not paretofront:
        raise TreeError("Pareto front is empty, cannot revive")
    candidate = np.random.choice(paretofront)
    return _copy.deepcopy(candidate.get_evotree())


# Registry of builtin strategies
BUILTIN_STRATEGIES: Dict[str, Callable] = {
    "reproduction": _strategy_reproduction,
    "mutation": _strategy_mutation,
    "mutation_point": _strategy_mutation_point,
    "mutation_branch_nodes": _strategy_mutation_branch_nodes,
    "mutation_filter": _strategy_mutation_filter,
    "random_new": _strategy_random_new,
    "crossover": _strategy_crossover,
    "simplicate": _strategy_simplicate,
    "pareto_revive": _strategy_pareto_revive,
}

# Strategies grouped by how many parent trees they need from the genepool.
# Used by pre_select_for_tasks() to move tournament selection to the main process.
_STRATEGIES_ONE_PARENT = frozenset(
    {
        "reproduction",
        "mutation",
        "mutation_point",
        "mutation_branch_nodes",
        "mutation_filter",
        "simplicate",
    }
)
_STRATEGIES_TWO_PARENTS = frozenset({"crossover"})
_STRATEGIES_NO_PARENT = frozenset({"random_new"})
_STRATEGIES_PARETO = frozenset({"pareto_revive"})


def pre_select_for_tasks(tasks, pop_genepool, paretofront):
    """Pre-select parent trees in the main process to avoid IPC of pop_genepool.

    For each task, performs tournament selection (or appropriate selection)
    and stores the result in task.selected_trees. Workers then use these
    pre-selected trees instead of needing the full population.

    This eliminates _update_worker_state overhead (~950ms/gen for pop=1000, 4w)
    which was the main parallelization bottleneck on Windows.

    Args:
        tasks: List of TaskSpec objects (modified in place).
        pop_genepool: Current generation's population.
        paretofront: Current Pareto front.

    Returns:
        bool: True if any task could NOT be pre-selected (needs pop_genepool
              sent to workers via _update_worker_state).
    """
    import copy as _copy

    from plagih.trees import selection_tournament

    needs_full_pop = False

    for task in tasks:
        name = task.strategy_name
        params = task.strategy_params
        tournament_n = params.get("tournament_n", 3)

        if name in _STRATEGIES_ONE_PARENT:
            tree = selection_tournament(pop_genepool, n=tournament_n)
            task.selected_trees = [tree]

        elif name in _STRATEGIES_TWO_PARENTS:
            tree_a = selection_tournament(pop_genepool, n=tournament_n)
            tree_b = selection_tournament(pop_genepool, n=tournament_n)
            task.selected_trees = [tree_a, tree_b]

        elif name in _STRATEGIES_NO_PARENT:
            task.selected_trees = []

        elif name in _STRATEGIES_PARETO:
            if paretofront:
                candidate = np.random.choice(paretofront)
                task.selected_trees = [_copy.deepcopy(candidate.get_evotree())]
            else:
                task.selected_trees = []

        else:
            # Unknown/custom strategy — cannot pre-select, worker needs pop_genepool
            task.selected_trees = None
            needs_full_pop = True

    return needs_full_pop


# =============================================================================
# Worker Task Execution
# =============================================================================


def _worker_run_task(task: TaskSpec, shared_lut_tree=None, shared_lut_symex=None) -> TaskResult:
    """Execute a single tree-creation task in a worker process.

    Uses global worker state (set by _init_worker) for evolve, df_train, etc.
    When called from _worker_run_batch, shared LUT dicts are passed in
    to enable intra-batch cache hits.

    Args:
        task: TaskSpec describing what to do.
        shared_lut_tree: Optional shared LUT dict (from batch). Creates new if None.
        shared_lut_symex: Optional shared LUT dict (from batch). Creates new if None.

    Returns:
        TaskResult with candidate(s), LUT deltas, timing, and error info.
    """
    from plagih.trees import tree_simplification

    timing = {}
    lut_tree = shared_lut_tree if shared_lut_tree is not None else {}
    lut_symex = shared_lut_symex if shared_lut_symex is not None else {}
    t0 = time.perf_counter()  # Start timing before try block
    stage = "setup"
    created_trees = []
    trees = []

    # Set seed if provided (for reproducibility)
    if task.seed is not None:
        task_seed = task.seed * 10000 + task.task_index
        random.seed(task_seed)
        np.random.seed(task_seed % (2**31))

    try:
        # Phase 1: Create tree(s) via strategy function
        stage = "create"

        # Look up strategy function
        registry = _worker_strategy_registry or BUILTIN_STRATEGIES
        if task.strategy_name not in registry:
            raise ValueError(f"Unknown strategy: {task.strategy_name}")

        strategy_fn = registry[task.strategy_name]

        # Inject pre-selected trees if available (parallel pre-selection mode).
        # Trees were selected in the main process and pickled with the task,
        # so we need to repair back-references after deserialization.
        call_params = task.strategy_params
        if task.selected_trees is not None and len(task.selected_trees) > 0:
            for tree in task.selected_trees:
                tree.repair_all()
            call_params = dict(task.strategy_params)
            call_params["_pre_selected"] = task.selected_trees

        result = strategy_fn(
            _worker_evolve,
            _worker_pop_genepool,
            _worker_paretofront,
            _worker_allow_chain,
            **call_params,
        )
        if task.crossover:
            created_trees = [result[0], result[1]]
        else:
            created_trees = [result]

        timing["create"] = time.perf_counter() - t0

        # Phase 2: Simplify if requested
        stage = "simplify"
        t1 = time.perf_counter()
        if task.crossover:
            tree_a, tree_b = result
            if task.simplicate:
                tree_a = tree_simplification(tree_a, allow_chain=_worker_allow_chain)
                tree_b = tree_simplification(tree_b, allow_chain=_worker_allow_chain)
            trees = [tree_a, tree_b]
        else:
            tree = result
            if task.simplicate:
                tree = tree_simplification(tree, allow_chain=_worker_allow_chain)
            trees = [tree]
        timing["simplify"] = time.perf_counter() - t1

        # Phase 3: Evaluate each tree
        stage = "evaluate"
        t2 = time.perf_counter()
        candidates = []
        for tree in trees:
            candidate = evaluate_tree_standalone(
                evotree=tree,
                evolve=_worker_evolve,
                df_train=_worker_df_train,
                eval_autocast=_worker_eval_autocast,
                eval_error_metric=_worker_eval_error_metric,
                target_column=_worker_target_column,
                nodes_max=_worker_nodes_max,
                complexity_metric=_worker_complexity_metric,
                local_lut_tree=lut_tree,
                local_lut_symex=lut_symex,
                tag=task.tag,
            )
            candidates.append(candidate)
        timing["evaluate"] = time.perf_counter() - t2
        timing["total"] = time.perf_counter() - t0

        return TaskResult(
            candidates=candidates,
            lut_tree_entries={},  # LUT entries stay in shared dicts (or local if standalone)
            lut_symex_entries={},
            timing=timing,
            error=None,
            tag=task.tag,
            debug=None,
        )

    except Exception as e:
        timing["total"] = time.perf_counter() - t0
        debug = _task_debug_payload(task, stage=stage, trees=trees or created_trees, exc=e)
        return TaskResult(
            candidates=[],
            lut_tree_entries={},  # LUT entries stay in shared dicts
            lut_symex_entries={},
            timing=timing,
            error=_format_task_error_message(e, debug),
            tag=task.tag,
            debug=debug,
        )


def _worker_run_batch(tasks: List[TaskSpec]) -> List[TaskResult]:
    """Execute a batch of tasks in a single worker call.

    Processes multiple tasks sequentially within the same worker process,
    sharing LUT dicts across tasks for intra-batch cache hits.
    Returns only TaskResults (no LUT dicts) — LUTs contain sympy objects
    which are extremely expensive to pickle. Since the pool is destroyed
    per generation anyway, worker LUTs don't need to persist.

    Args:
        tasks: List of TaskSpecs for this batch.

    Returns:
        List of TaskResults for tracking and candidate collection.
    """
    batch_lut_tree = {}
    batch_lut_symex = {}
    results = []

    for task in tasks:
        try:
            result = _worker_run_task(task, shared_lut_tree=batch_lut_tree, shared_lut_symex=batch_lut_symex)
        except Exception as e:
            debug = _task_debug_payload(task, stage="batch_wrapper", trees=None, exc=e)
            result = TaskResult(
                candidates=[],
                lut_tree_entries={},
                lut_symex_entries={},
                timing={"total": 0.0},
                error=_format_task_error_message(e, debug),
                tag=task.tag,
                debug=debug,
            )
        results.append(result)

    return results


# =============================================================================
# Sequential Task Execution (same logic, no pool)
# =============================================================================


def run_task_sequential(
    task: TaskSpec,
    evolve,
    df_train,
    pop_genepool,
    paretofront,
    eval_autocast,
    eval_error_metric,
    allow_chain,
    target_column,
    nodes_max,
    complexity_metric,
    strategy_registry,
    lut_tree,
    lut_symex,
) -> TaskResult:
    """Execute a single task sequentially in the main process.

    Same logic as _worker_run_task but uses passed-in objects directly
    and writes to the main-process LUTs. Allows full debugging with breakpoints.

    Args:
        task: TaskSpec describing what to do.
        evolve, df_train, ...: GP state passed directly (not via globals).
        lut_tree, lut_symex: Main-process LUTs (mutated in place).
        strategy_registry: Combined builtin + custom strategies.

    Returns:
        TaskResult with candidate(s), timing, and error info.
    """
    from plagih.trees import tree_simplification

    timing = {}
    t0 = time.perf_counter()

    # Set seed if provided
    if task.seed is not None:
        task_seed = task.seed * 10000 + task.task_index
        random.seed(task_seed)
        np.random.seed(task_seed % (2**31))

    try:
        # Phase 1: Create tree(s)

        registry = strategy_registry
        if task.strategy_name not in registry:
            raise ValueError(f"Unknown strategy: {task.strategy_name}")

        strategy_fn = registry[task.strategy_name]
        result = strategy_fn(evolve, pop_genepool, paretofront, allow_chain, **task.strategy_params)
        timing["create"] = time.perf_counter() - t0

        # Phase 2: Simplify
        t1 = time.perf_counter()
        if task.crossover:
            tree_a, tree_b = result
            if task.simplicate:
                tree_a = tree_simplification(tree_a, allow_chain=allow_chain)
                tree_b = tree_simplification(tree_b, allow_chain=allow_chain)
            trees = [tree_a, tree_b]
        else:
            tree = result
            if task.simplicate:
                tree = tree_simplification(tree, allow_chain=allow_chain)
            trees = [tree]
        timing["simplify"] = time.perf_counter() - t1

        # Phase 3: Evaluate
        t2 = time.perf_counter()
        candidates = []
        for tree in trees:
            candidate = evaluate_tree_standalone(
                evotree=tree,
                evolve=evolve,
                df_train=df_train,
                eval_autocast=eval_autocast,
                eval_error_metric=eval_error_metric,
                target_column=target_column,
                nodes_max=nodes_max,
                complexity_metric=complexity_metric,
                local_lut_tree=lut_tree,
                local_lut_symex=lut_symex,
                tag=task.tag,
            )
            candidates.append(candidate)
        timing["evaluate"] = time.perf_counter() - t2
        timing["total"] = time.perf_counter() - t0

        return TaskResult(
            candidates=candidates,
            lut_tree_entries={},  # Already written to main LUTs
            lut_symex_entries={},
            timing=timing,
            error=None,
            tag=task.tag,
        )

    except (TreeError, TreeSizeError, SympyError, ValueError, ArithmeticError, KeyError, RecursionError) as e:
        timing["total"] = time.perf_counter() - t0
        return TaskResult(
            candidates=[],
            lut_tree_entries={},
            lut_symex_entries={},
            timing=timing,
            error=f"{type(e).__name__}: {e}",
            tag=task.tag,
        )


# =============================================================================
# Generation Runner (orchestrates parallel or sequential execution)
# =============================================================================


def build_task_list(strategies: List[Strategy], pop_max_size: int, seed: Optional[int] = None) -> List[TaskSpec]:
    """Convert a list of Strategy objects into individual TaskSpecs.

    Args:
        strategies: List of Strategy objects defining the generation.
        pop_max_size: Maximum population size (used to compute task counts from rates).
        seed: Optional base seed for reproducibility.

    Returns:
        List of TaskSpec objects, one per tree to create.
    """
    tasks = []
    task_index = 0
    for strategy in strategies:
        n = int(strategy.rate * pop_max_size)
        # For crossover, each call produces 2 trees, so halve the task count
        if strategy.crossover:
            n_tasks = max(1, n // 2)
        else:
            n_tasks = n

        for _ in range(n_tasks):
            tasks.append(
                TaskSpec(
                    strategy_name=strategy.name,
                    strategy_params=strategy.params,
                    crossover=strategy.crossover,
                    simplicate=strategy.simplicate,
                    tag=strategy.name,
                    task_index=task_index,
                    seed=seed,
                )
            )
            task_index += 1
    return tasks


def run_generation_parallel(
    tasks: List[TaskSpec],
    n_workers: int,
    evolve,
    df_train,
    pop_genepool,
    paretofront,
    eval_autocast,
    eval_error_metric,
    allow_chain,
    target_column,
    nodes_max,
    complexity_metric,
    strategy_registry,
    pool: Optional[ProcessPoolExecutor] = None,
) -> Tuple[list, Dict, Dict, PerformanceTracker]:
    """Run all tasks in parallel using ProcessPoolExecutor with batch submission.

    Design: Instead of submitting N individual tasks (N pickle roundtrips),
    tasks are split into n_workers batches. Each worker processes its entire
    batch in a single call, sharing LUTs within the batch. This reduces
    serialization overhead from O(N) to O(n_workers).

    Pre-selection: Tournament selection is performed in the main process and
    the selected trees are attached to each TaskSpec. This eliminates the
    need for _update_worker_state (which pickles the entire pop_genepool to
    every worker every generation — the main IPC bottleneck on Windows).

    LUT dicts are NOT returned from workers because they contain sympy objects
    which are extremely expensive to pickle. Worker LUTs are used only for
    intra-batch deduplication.

    Args:
        tasks: List of TaskSpec objects.
        n_workers: Number of worker processes.
        evolve, df_train, ...: GP state for worker initialization.
        strategy_registry: Combined builtin + custom strategies.
        pool: Optional persistent ProcessPoolExecutor. If None, creates a new one.

    Returns:
        Tuple of (candidates, lut_tree_delta, lut_symex_delta, tracker).
        Note: lut_tree_delta and lut_symex_delta are always empty in parallel mode.
    """
    tracker = PerformanceTracker()
    all_candidates = []
    lut_tree_delta = {}
    lut_symex_delta = {}

    gen_start = time.perf_counter()

    # Pre-select parent trees in the main process so workers don't need pop_genepool.
    # This eliminates _update_worker_state IPC overhead (~950ms/gen for pop=1000, 4w).
    needs_full_pop = pre_select_for_tasks(tasks, pop_genepool, paretofront)

    # Split tasks into mixed-strategy chunks rather than one giant batch/worker.
    # This improves load balancing and makes pathological tasks debuggable.
    batches = _build_task_batches(tasks, n_workers)

    owns_pool = pool is None
    init_resources = None
    fast_shutdown = False
    if owns_pool:
        initargs, init_resources = build_worker_init_config(
            evolve=evolve,
            df_train=df_train,
            pop_genepool=pop_genepool if needs_full_pop else [],
            paretofront=paretofront if needs_full_pop else [],
            eval_autocast=eval_autocast,
            eval_error_metric=eval_error_metric,
            allow_chain=allow_chain,
            target_column=target_column,
            nodes_max=nodes_max,
            complexity_metric=complexity_metric,
            strategy_registry=strategy_registry,
        )
        try:
            pool = ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=cast(tuple[Any, ...], initargs),
            )
        except Exception:
            cleanup_worker_init_resources(init_resources)
            raise

    try:
        # Only send pop_genepool/paretofront to workers if a custom strategy
        # needs them (i.e. could not be pre-selected). For builtin strategies,
        # pre-selected trees are already attached to each TaskSpec.
        if needs_full_pop:
            update_futures = [pool.submit(_update_worker_state, pop_genepool, paretofront) for _ in range(n_workers)]
            for f in update_futures:
                f.result()  # Wait for all workers to update

        futures = {pool.submit(_worker_run_batch, batch): batch for batch in batches}
        pending = set(futures)
        timeout_s = _progress_timeout_seconds()

        while pending:
            done, pending = wait(pending, timeout=timeout_s, return_when=FIRST_COMPLETED)
            if not done:
                pending_debug = [_summarize_batch(futures[future]) for future in pending]
                fast_shutdown = True
                print_warning(
                    "ww",
                    "Parallel batch timeout: no worker batch completed within "
                    f"{timeout_s:.1f}s; pending_batches={pending_debug}",
                )
                if not owns_pool and pool is not None:
                    try:
                        pool.shutdown(wait=False, cancel_futures=True)
                    except Exception:
                        pass
                raise TimeoutError(
                    f"BatchTimeout: no worker batch completed within {timeout_s:.1f}s; pending_batches={pending_debug}"
                )

            for future in done:
                batch = futures[future]

                try:
                    batch_results = future.result()
                except Exception as e:
                    batch_summary = _summarize_batch(batch)
                    debug = {
                        "batch_summary": batch_summary,
                        "traceback": traceback.format_exc(limit=DEBUG_TRACEBACK_LINES),
                    }
                    for task in batch:
                        result = TaskResult(
                            candidates=[],
                            lut_tree_entries={},
                            lut_symex_entries={},
                            timing={"total": 0.0},
                            error=(
                                f"WorkerCrash: {type(e).__name__}: {e} "
                                f"[strategy={task.strategy_name}, task_index={task.task_index}, batch={batch_summary}]"
                            ),
                            tag=task.tag,
                            debug=debug,
                        )
                        tracker.record(result)
                        _print_parallel_failure_debug(result)
                    continue

                # Process individual task results for tracking
                for result in batch_results:
                    tracker.record(result)
                    if result.error is not None:
                        _print_parallel_failure_debug(result)
                        continue

                    # Repair tree back-references stripped during pickling
                    for candidate in result.candidates:
                        candidate.tree.repair_all()
                    all_candidates.extend(result.candidates)

    finally:
        if owns_pool:
            pool.shutdown(wait=not fast_shutdown, cancel_futures=fast_shutdown)
            cleanup_worker_init_resources(init_resources)

    tracker.set_generation_time(time.perf_counter() - gen_start)
    return all_candidates, lut_tree_delta, lut_symex_delta, tracker


def run_generation_sequential(
    tasks: List[TaskSpec],
    evolve,
    df_train,
    pop_genepool,
    paretofront,
    eval_autocast,
    eval_error_metric,
    allow_chain,
    target_column,
    nodes_max,
    complexity_metric,
    strategy_registry,
    lut_tree,
    lut_symex,
) -> Tuple[list, PerformanceTracker]:
    """Run all tasks sequentially in the main process.

    Identical logic to parallel execution but without a pool.
    Allows full debugging with breakpoints.

    Args:
        tasks: List of TaskSpec objects.
        evolve, df_train, ...: GP state.
        strategy_registry: Combined builtin + custom strategies.
        lut_tree, lut_symex: Main-process LUTs (mutated in place).

    Returns:
        Tuple of (candidates, tracker).
    """
    tracker = PerformanceTracker()
    all_candidates = []

    fail_counts: Dict[str, int] = {}
    tag_expected: Dict[str, int] = {}
    for task in tasks:
        tag_expected[task.tag] = tag_expected.get(task.tag, 0) + 1

    gen_start = time.perf_counter()

    for task in tasks:
        tag = task.tag

        # Check failure budget before executing
        n_expected = tag_expected.get(tag, 1)
        budget = 2 * n_expected + 5
        if fail_counts.get(tag, 0) > budget:
            continue  # Skip remaining tasks for this strategy

        result = run_task_sequential(
            task=task,
            evolve=evolve,
            df_train=df_train,
            pop_genepool=pop_genepool,
            paretofront=paretofront,
            eval_autocast=eval_autocast,
            eval_error_metric=eval_error_metric,
            allow_chain=allow_chain,
            target_column=target_column,
            nodes_max=nodes_max,
            complexity_metric=complexity_metric,
            strategy_registry=strategy_registry,
            lut_tree=lut_tree,
            lut_symex=lut_symex,
        )

        tracker.record(result)

        if result.error is not None:
            fail_counts[tag] = fail_counts.get(tag, 0) + 1
            if fail_counts[tag] > budget:
                print_caution(
                    f"Strategy '{tag}' exceeded failure budget "
                    f"({fail_counts[tag]}/{budget}). Skipping remaining tasks.",
                )
        else:
            all_candidates.extend(result.candidates)

    tracker.set_generation_time(time.perf_counter() - gen_start)
    return all_candidates, tracker
