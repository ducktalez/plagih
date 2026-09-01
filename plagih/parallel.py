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

from plagih.config import cfg as _cfg
from plagih.exceptions import SympyError, TreeError, TreeLutError, TreeSizeError
from plagih.logging_utils import log, log_error

# =============================================================================
# Strategy Dataclass
# =============================================================================


@dataclass
class Strategy:
    """Declarative specification of an evolution strategy.

    Args:
        name: Name of a builtin strategy or a registered custom strategy.
        rate: Fraction of pop_max_size to create (0.0 to 1.0).
        count: Exact number of candidate slots for this strategy. If set,
            overrides ``rate`` and is especially useful for generation-0 plans.
        crossover: If True, strategy returns two trees per call.
        simplicate: If True, apply tree simplification before evaluation.
        params: Additional keyword arguments passed to the strategy function.

    Example:
        Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3)
        Strategy("crossover", rate=0.2, tournament_n=3)
    """

    name: str
    rate: float = 0.0
    count: Optional[int] = None
    crossover: bool = False
    simplicate: bool = False
    params: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        rate: float = 0.0,
        count: Optional[int] = None,
        crossover: bool = False,
        simplicate: bool = False,
        **params,
    ):
        self.name = name
        self.rate = rate
        self.count = count
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
    tree_timings: Optional[List[Dict[str, Any]]] = None


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
        self.tree_timings: List[Dict[str, Any]] = []

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

        if result.tree_timings:
            self.tree_timings.extend(result.tree_timings)

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
        """Print a compact one-line performance summary per strategy."""
        summary = self.summary()
        parts = []
        for tag in dict.fromkeys(self._results):  # insertion-order dedup
            avg = summary.get(f"strategy_{tag}_avg_time", 0) * 1000
            ok = summary.get(f"strategy_{tag}_success", 0)
            fail = summary.get(f"strategy_{tag}_fails", 0)
            seg = f"{tag}: avg={avg:.0f}ms ok={ok}"
            if fail:
                seg += f" fail={fail}"
            parts.append(seg)
        total = summary["generation_total_time"]
        log("ppp", f"[Perf] {total:.2f}s | {' | '.join(parts)}")

    def merge(self, other: "PerformanceTracker") -> None:
        """Merge another tracker into this one."""
        for tag, timings in other._results.items():
            self._results.setdefault(tag, []).extend(timings)
            self._errors[tag] = self._errors.get(tag, 0) + other._errors.get(tag, 0)
            self._successes[tag] = self._successes.get(tag, 0) + other._successes.get(tag, 0)

        self._generation_time += other._generation_time
        self.tree_timings.extend(other.tree_timings)

    def reset(self):
        """Reset all tracking data for the next generation."""
        self._results.clear()
        self._errors.clear()
        self._successes.clear()
        self._generation_time = 0.0
        self.tree_timings.clear()


@dataclass
class WorkerInitResources:
    """Parent-side resources owned for worker initialization."""

    df_train_shm: Optional[shared_memory.SharedMemory] = None
    uses_shared_df_train: bool = False


DEFAULT_TASKS_PER_BATCH = 128
PROGRESS_TIMEOUT_SECONDS = 120.0
DEBUG_TRACEBACK_LINES = 20


def _validate_post_simplify_tree_requirements(task: TaskSpec, tree) -> None:
    """Apply generic post-simplification validation declared on a task."""
    min_depth = task.strategy_params.get("require_min_depth_after_simplify")
    if min_depth is None:
        return
    if tree.get_max_depth() < int(min_depth):
        raise TreeSizeError("Tree did not get complex enough after simplification (only root node).")


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
_worker_true_values = None  # Cached df_train[target_column].to_numpy() — avoids per-eval copy
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
    log(
        "ww",
        "Parallel task failure: "
        f"tag={result.tag}, error={result.error}, "
        f"stage={debug.get('stage')}, task_index={debug.get('task_index')}, "
        f"strategy={debug.get('strategy')}, pid={debug.get('worker_pid')}",
    )
    tb = debug.get("traceback")
    if tb:
        log("ww", f"Parallel task traceback:\n{tb}")


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
    global _worker_true_values

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
    # Cache target values once — avoids df_train[target_column].to_numpy() per evaluation.
    _worker_true_values = _worker_df_train[target_column].to_numpy() if _worker_df_train is not None else None


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
    true_values=None,
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
        true_values: Pre-computed ``df_train[target_column].to_numpy()``.
            If None, computed on demand (backward compatible).

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
    evotree = evolve.evolve_prune_tree(evotree, df_train=df_train)  # I16: semantic replacement
    evotree.repair_depth()
    tree_id = evotree.canonicalize_and_get_lut_id()  # Fused: canonicalize + LUT key in one traversal

    if tree_id in local_lut_tree:
        entry = local_lut_tree[tree_id]
        parsimony = entry.get("parsimony")
        fitness = entry.get("fitness")
        if parsimony is not None and fitness is not None:
            return Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
        _err = entry.get("error")
        if _err is not None:
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
        if true_values is None:
            true_values = df_train[target_column].to_numpy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            np_results_raw = evotree.eval_predict_numpy_now(df_train)
            np_results = eval_autocast(np_results_raw)

            # --- NaN-tolerant evaluation (I1 replacement) ---
            np_results_arr = np.asarray(np_results, dtype=np.float64)
            finite_mask = np.isfinite(np_results_arr)
            n_bad = int((~finite_mask).sum())
            if n_bad > 0:
                n_total = len(np_results_arr)
                if n_bad > n_total // 2:
                    err_txt = f"NaN in results ({n_bad}/{n_total} non-finite)"
                    local_lut_tree[tree_id]["error"] = err_txt
                    raise TreeError(err_txt)
                tv_min, tv_max = float(true_values.min()), float(true_values.max())
                tv_range = tv_max - tv_min if tv_max > tv_min else 1.0
                penalty_value = tv_max + tv_range
                np_results = np.where(finite_mask, np_results_arr, penalty_value)

            np_fitness = eval_error_metric(np_results, true_values)
            np_fitness = round(np_fitness, _cfg.float_precision)

            if not np.isfinite(np_fitness):
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


def _strategy_mutation_terminal(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate only its terminal nodes (structure-preserving)."""
    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 3)
    n_terminals = params.get("n_terminals", 1)
    p_symbol = params.get("p_symbol", 0.5)
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_terminals(tree, n_terminals=n_terminals, p_symbol=p_symbol)


def _strategy_random_new(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Create a new random tree with a random depth."""
    depth_sampler = params.get("depth_sampler", "choice")
    p_term = params.get("p_term", 0.1)
    if depth_sampler == "normal":
        mean = params.get("mean", 3.0)
        sigma = params.get("sigma", 1.0)
        min_depth = params.get("min_depth", 2)
        max_depth = params.get("max_depth", evolve.depth_max)
        depth = np.clip(int(random.normalvariate(mean, sigma)), int(min_depth), int(max_depth))
    else:
        depths = params.get("depths", [2, 3, 4])
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
    from plagih.trees._nodes import fast_tree_copy

    pre = params.pop("_pre_selected", None)
    if pre:
        return pre[0]

    if not paretofront:
        raise TreeError("Pareto front is empty, cannot revive")
    candidate = np.random.choice(paretofront)
    return fast_tree_copy(candidate.get_evotree())


def _strategy_targeted_ifte(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree containing Ifte/Piecewise and mutate its weakest component.

    Phase 2b of Targeted Evolutionary Optimization (D5 in IMPLEMENTATION_PLAN).
    Uses pseudo-backpropagation scoring to identify the weakest Ifte component
    (condition, then, or else) and applies focused mutation only to that subtree.

    Falls back to standard branch mutation if no Ifte nodes are found.
    """
    from plagih.trees._nodes import fast_tree_copy

    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 5)
    depth_goal = params.get("depth_goal", 3)
    p_term = params.get("p_term", 0.2)
    df_train = params.get("_df_train")
    target = params.get("_target")

    # Select a tree — prefer trees with Ifte/Piecewise nodes
    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        # Try multiple tournaments to find a tree with Ifte/Piecewise
        best_tree = None
        for _ in range(3):
            candidate_tree = selection_tournament(pop_genepool, n=tournament_n)
            if _tree_has_ifte(candidate_tree):
                best_tree = candidate_tree
                break
            if best_tree is None:
                best_tree = candidate_tree
        tree = best_tree

    evotree = fast_tree_copy(tree)

    # If no df_train/target available, or tree has no Ifte — fall back to standard mutation
    if df_train is None or target is None or not _tree_has_ifte(evotree):
        return evolve.evolve_mutate_branch_depth(evotree, depth_goal, allow_chain, p_term=p_term)

    # Find weakest Ifte component via pseudo-backpropagation
    from plagih.targeted_optimization import ifte_component_scores

    try:
        analyses = ifte_component_scores(evotree, df_train, target)
    except Exception:
        return evolve.evolve_mutate_branch_depth(evotree, depth_goal, allow_chain, p_term=p_term)

    if not analyses:
        return evolve.evolve_mutate_branch_depth(evotree, depth_goal, allow_chain, p_term=p_term)

    # Pick the Ifte node with the worst condition accuracy
    worst_analysis = min(analyses, key=lambda a: a.condition_accuracy)

    # Find the actual Ifte node in the tree by matching node_id
    target_node = _find_node_by_id(evotree, worst_analysis.node_id)
    if target_node is None or not hasattr(target_node, "get_childs") or len(target_node.get_childs()) < 3:
        return evolve.evolve_mutate_branch_depth(evotree, depth_goal, allow_chain, p_term=p_term)

    # Determine which component to mutate
    weakest = worst_analysis.weakest  # "condition", "then", or "else"
    childs = target_node.get_childs()
    if weakest == "condition":
        mutation_target = childs[0]
    elif weakest == "then":
        mutation_target = childs[1]
    else:
        mutation_target = childs[2]

    # Focused mutation: replace only the weakest component
    n_init = len(evotree)
    xtype_out = mutation_target.get_xtype_self()
    branch = evolve.evolve_create_random(
        xtype_out,
        depth_goal,
        num_rest=max(1, evolve.nodes_max - n_init),
        depth=mutation_target.depth or 0,
        p_term=p_term,
    )
    mutation_target.set_new_node(branch)

    return evotree


def _strategy_targeted_gap(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Mutate the subtree with the largest node-level optimization gap.

    Phase 3 of Targeted Evolutionary Optimization (D5 in IMPLEMENTATION_PLAN).
    Pushes the ideal output value backwards through invertible operators
    (Add, Sub, Mul, Div, Scale, Usub, DivFraction) and replaces the child
    whose actual output deviates most from its ideal value.

    Falls back to standard branch mutation when no gap can be computed
    (no training data, non-invertible root, or evaluation failure).
    """
    from plagih.trees._nodes import fast_tree_copy

    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 5)
    depth_goal = params.get("depth_goal", 3)
    p_term = params.get("p_term", 0.2)
    df_train = params.get("_df_train")
    target = params.get("_target")

    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)

    evotree = fast_tree_copy(tree)

    def _fallback():
        return evolve.evolve_mutate_branch_depth(evotree, depth_goal, allow_chain, p_term=p_term)

    if df_train is None or target is None:
        return _fallback()

    from plagih.targeted_optimization import largest_gap_node, node_optimization_gaps

    try:
        gaps = node_optimization_gaps(evotree, df_train, target)
    except Exception:
        return _fallback()

    worst = largest_gap_node(gaps)
    if worst is None:
        return _fallback()

    mutation_target = _find_node_by_id(evotree, worst.node_id)
    if mutation_target is None or mutation_target.is_fix:
        return _fallback()

    # Focused mutation: regrow only the weakest subtree
    n_init = len(evotree)
    branch = evolve.evolve_create_random(
        mutation_target.get_xtype_self(),
        depth_goal,
        num_rest=max(1, evolve.nodes_max - n_init),
        depth=mutation_target.depth or 0,
        p_term=p_term,
    )
    mutation_target.set_new_node(branch)

    return evotree


def _strategy_chain_mutation(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Targeted mutation of chainable operators (Add/Mul/Min/Max/And/Or).

    Phase 4 of Targeted Evolutionary Optimization (§3.4, D5).  Picks a
    chainable node and applies one of:

    - ``add``: append a new random operand (grows the chain; requires
      ``allow_chain`` to exceed arity 2)
    - ``remove``: drop one operand (arity never falls below 2)
    - ``replace``: regrow one operand

    With training data available, the operand to remove/replace is the one
    with the largest optimization gap (Phase 3); otherwise random.
    Falls back to standard branch mutation when the tree has no mutable
    chainable node.
    """
    import random as _random

    from plagih.trees._nodes import ChainableOp, Terminal, fast_tree_copy

    pre = params.pop("_pre_selected", None)
    tournament_n = params.get("tournament_n", 5)
    depth_goal = params.get("depth_goal", 3)
    operand_depth = params.get("operand_depth", 2)
    p_term = params.get("p_term", 0.2)
    df_train = params.get("_df_train")
    target = params.get("_target")

    if pre:
        tree = pre[0]
    else:
        from plagih.trees import selection_tournament

        tree = selection_tournament(pop_genepool, n=tournament_n)

    evotree = fast_tree_copy(tree)

    def _fallback():
        return evolve.evolve_mutate_branch_depth(evotree, depth_goal, allow_chain, p_term=p_term)

    # Collect mutable chainable nodes
    chain_nodes = []

    def _collect(node):
        if isinstance(node, Terminal):
            return
        if isinstance(node, ChainableOp) and not node.is_fix:
            chain_nodes.append(node)
        for cc in node.get_childs():
            _collect(cc)

    _collect(evotree)
    if not chain_nodes:
        return _fallback()

    chain_node = _random.choice(chain_nodes)
    childs = chain_node.get_childs()

    # Feasible actions
    actions = ["replace"]
    if allow_chain or len(childs) < 2:
        actions.append("add")
    if len(childs) > 2:
        actions.append("remove")
    action = _random.choice(actions)

    def _pick_operand_index() -> int:
        """Worst operand by gap when data present, else random."""
        if df_train is not None and target is not None:
            from plagih.targeted_optimization import node_optimization_gaps

            try:
                gaps = node_optimization_gaps(evotree, df_train, target)
                by_id = {g.node_id: g.gap_mean for g in gaps if g.n_finite > 0 and np.isfinite(g.gap_mean)}
                scored = [(by_id.get(id(cc), -1.0), ii) for ii, cc in enumerate(childs)]
                best_gap, best_ii = max(scored)
                if best_gap >= 0:
                    return best_ii
            except Exception:
                pass
        return _random.randrange(len(childs))

    operand_xtype = getattr(type(chain_node), "xtype_input", None) or float

    if action == "add":
        n_init = len(evotree)
        branch = evolve.evolve_create_random(
            operand_xtype,
            operand_depth,
            num_rest=max(1, evolve.nodes_max - n_init),
            depth=(chain_node.depth or 0) + 1,
            p_term=p_term,
        )
        chain_node.add_child(branch)
    elif action == "remove":
        idx = _pick_operand_index()
        new_childs = [cc for ii, cc in enumerate(childs) if ii != idx]
        chain_node.set_childs(new_childs)
    else:  # replace
        idx = _pick_operand_index()
        n_init = len(evotree)
        branch = evolve.evolve_create_random(
            childs[idx].get_xtype_self(),
            operand_depth,
            num_rest=max(1, evolve.nodes_max - n_init),
            depth=(chain_node.depth or 0) + 1,
            p_term=p_term,
        )
        childs[idx].set_new_node(branch)

    evotree.repair_all()
    if len(evotree) > evolve.nodes_max:
        evotree = evolve.evolve_prune_tree(evotree, df_train=df_train)  # I16: semantic replacement

    return evotree


def _tree_has_ifte(tree) -> bool:
    """Check if a tree contains any Ifte or Piecewise nodes."""
    from plagih.trees._nodes import Ifte, Piecewise, Terminal

    if isinstance(tree, Terminal):
        return False
    if isinstance(tree, (Ifte, Piecewise)):
        return True
    if hasattr(tree, "get_childs"):
        return any(_tree_has_ifte(c) for c in tree.get_childs())
    return False


def _find_node_by_id(tree, node_id: int):
    """Find a node in a tree by its id()."""
    if id(tree) == node_id:
        return tree
    if hasattr(tree, "get_childs"):
        for child in tree.get_childs():
            found = _find_node_by_id(child, node_id)
            if found is not None:
                return found
    return None


# Registry of builtin strategies
BUILTIN_STRATEGIES: Dict[str, Callable] = {
    "reproduction": _strategy_reproduction,
    "mutation": _strategy_mutation,
    "mutation_point": _strategy_mutation_point,
    "mutation_branch_nodes": _strategy_mutation_branch_nodes,
    "mutation_filter": _strategy_mutation_filter,
    "mutation_terminal": _strategy_mutation_terminal,
    "random_new": _strategy_random_new,
    "crossover": _strategy_crossover,
    "simplicate": _strategy_simplicate,
    "pareto_revive": _strategy_pareto_revive,
    "targeted_ifte": _strategy_targeted_ifte,
    "targeted_gap": _strategy_targeted_gap,
    "chain_mutation": _strategy_chain_mutation,
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
        "mutation_terminal",
        "simplicate",
        "targeted_ifte",
        "targeted_gap",
        "chain_mutation",
    }
)
_STRATEGIES_TWO_PARENTS = frozenset({"crossover"})
_STRATEGIES_NO_PARENT = frozenset({"random_new"})
_STRATEGIES_PARETO = frozenset({"pareto_revive"})

# Strategies that need `_df_train` / `_target` injected at task runtime.
_STRATEGIES_NEEDING_TRAINING_DATA = frozenset({"targeted_ifte", "targeted_gap", "chain_mutation"})


def _batch_tournament_select(
    pop_genepool: list,
    n_selections: int,
    tournament_n: int = 3,
) -> np.ndarray:
    """Vectorized tournament selection for multiple tasks at once.

    Uses NumPy to generate all tournament indices and find winners
    in a single vectorized pass, replacing N individual Python-level
    ``random.choices()`` + ``min()`` calls.

    Args:
        pop_genepool: List of Candidate objects.
        n_selections: Number of winners to select.
        tournament_n: Tournament size per selection.

    Returns:
        NumPy int array of winner indices into pop_genepool (shape: ``(n_selections,)``).
    """
    pop_size = len(pop_genepool)
    k = min(tournament_n, pop_size)

    # Pre-compute fitness array once — eliminates N*k get_fitness() calls
    fitness = np.array([c.get_fitness() for c in pop_genepool], dtype=np.float64)

    # Generate all tournament indices at once: (n_selections, k)
    indices = np.random.randint(0, pop_size, size=(n_selections, k))

    # Vectorized winner selection (lowest fitness = best)
    tournament_fitness = fitness[indices]
    winner_local = np.argmin(tournament_fitness, axis=1)
    winner_indices = indices[np.arange(n_selections), winner_local]

    return winner_indices


def pre_select_for_tasks(tasks, pop_genepool, paretofront):
    """Pre-select parent trees in the main process to avoid IPC of pop_genepool.

    For each task, performs tournament selection (or appropriate selection)
    and stores the result in task.selected_trees. Workers then use these
    pre-selected trees instead of needing the full population.

    This eliminates _update_worker_state overhead (~950ms/gen for pop=1000, 4w)
    which was the main parallelization bottleneck on Windows.

    Optimization (H2): Tournament selection is vectorized via NumPy —
    all tournaments for same-sized groups are generated in a single
    ``np.random.randint`` + ``np.argmin`` pass, replacing N individual
    ``random.choices()`` + ``min()`` calls.

    Args:
        tasks: List of TaskSpec objects (modified in place).
        pop_genepool: Current generation's population.
        paretofront: Current Pareto front.

    Returns:
        bool: True if any task could NOT be pre-selected (needs pop_genepool
              sent to workers via _update_worker_state).
    """
    from plagih.trees._nodes import fast_tree_copy

    needs_full_pop = False

    if not pop_genepool:
        for task in tasks:
            if task.strategy_name in _STRATEGIES_NO_PARENT or task.strategy_name in _STRATEGIES_PARETO:
                task.selected_trees = []
            else:
                task.selected_trees = None
                needs_full_pop = True
        return needs_full_pop

    # ── Classify tasks by selection type and tournament_n ──
    one_parent_groups: Dict[int, List[TaskSpec]] = {}  # tournament_n -> [task, ...]
    two_parent_groups: Dict[int, List[TaskSpec]] = {}  # tournament_n -> [task, ...]

    for task in tasks:
        name = task.strategy_name
        tournament_n = task.strategy_params.get("tournament_n", 3)

        if name in _STRATEGIES_ONE_PARENT:
            one_parent_groups.setdefault(tournament_n, []).append(task)

        elif name in _STRATEGIES_TWO_PARENTS:
            two_parent_groups.setdefault(tournament_n, []).append(task)

        elif name in _STRATEGIES_NO_PARENT:
            task.selected_trees = []

        elif name in _STRATEGIES_PARETO:
            if paretofront:
                candidate = np.random.choice(paretofront)
                task.selected_trees = [fast_tree_copy(candidate.get_evotree())]
            else:
                task.selected_trees = []

        else:
            # Unknown/custom strategy — cannot pre-select, worker needs pop_genepool
            task.selected_trees = None
            needs_full_pop = True

    # ── Batch tournament selection for one-parent strategies ──
    for tournament_n, group_tasks in one_parent_groups.items():
        winner_indices = _batch_tournament_select(pop_genepool, len(group_tasks), tournament_n)
        for task, winner_idx in zip(group_tasks, winner_indices):
            task.selected_trees = [fast_tree_copy(pop_genepool[winner_idx].get_evotree())]

    # ── Batch tournament selection for two-parent strategies ──
    for tournament_n, group_tasks in two_parent_groups.items():
        n = len(group_tasks)
        winners_a = _batch_tournament_select(pop_genepool, n, tournament_n)
        winners_b = _batch_tournament_select(pop_genepool, n, tournament_n)
        for task, idx_a, idx_b in zip(group_tasks, winners_a, winners_b):
            tree_a = fast_tree_copy(pop_genepool[idx_a].get_evotree())
            tree_b = fast_tree_copy(pop_genepool[idx_b].get_evotree())
            task.selected_trees = [tree_a, tree_b]

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
    tree_timing_records = []
    current_tree_index = None

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

        # Inject runtime context for targeted strategies (they need training
        # data).  Without this they silently degrade to plain branch mutation
        # in parallel mode.
        if task.strategy_name in _STRATEGIES_NEEDING_TRAINING_DATA:
            if call_params is task.strategy_params:
                call_params = dict(task.strategy_params)
            call_params.setdefault("_df_train", _worker_df_train)
            _wrk_target = _worker_true_values
            if _wrk_target is None and _worker_df_train is not None and _worker_target_column:
                _wrk_target = _worker_df_train[_worker_target_column].values
            call_params.setdefault("_target", _wrk_target)

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
        trees = []
        simplify_durations = []
        raw_trees = list(created_trees)
        t1 = time.perf_counter()
        for raw_tree in raw_trees:
            tree_simplify_start = time.perf_counter()
            tree_now = tree_simplification(raw_tree, allow_chain=_worker_allow_chain) if task.simplicate else raw_tree
            _validate_post_simplify_tree_requirements(task, tree_now)
            simplify_durations.append(time.perf_counter() - tree_simplify_start)
            trees.append(tree_now)
        timing["simplify"] = time.perf_counter() - t1

        # Phase 3: Evaluate each tree
        stage = "evaluate"
        t2 = time.perf_counter()
        candidates = []
        for tree_index, tree in enumerate(trees):
            current_tree_index = tree_index
            tree_eval_start = time.perf_counter()
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
                true_values=_worker_true_values,
            )
            candidates.append(candidate)
            evaluate_duration = time.perf_counter() - tree_eval_start
            tree_timing_records.append(
                {
                    "tag": task.tag,
                    "task_index": task.task_index,
                    "tree_index": tree_index,
                    "status": "ok",
                    "failed_stage": None,
                    "create_ms_shared": timing.get("create", 0.0) * 1000,
                    "simplify_ms": simplify_durations[tree_index] * 1000,
                    "evaluate_ms": evaluate_duration * 1000,
                    "total_ms": (timing.get("create", 0.0) + simplify_durations[tree_index] + evaluate_duration) * 1000,
                    "fitness": candidate.fitness,
                    "parsimony": candidate.parsimony,
                    "expr_short": str(candidate.tree),
                }
            )
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
            tree_timings=tree_timing_records,
        )

    except Exception as e:
        timing["total"] = time.perf_counter() - t0
        debug = _task_debug_payload(task, stage=stage, trees=trees or created_trees, exc=e)
        tree_timing_records.append(
            {
                "tag": task.tag,
                "task_index": task.task_index,
                "tree_index": current_tree_index,
                "status": "error",
                "failed_stage": stage,
                "create_ms_shared": timing.get("create", 0.0) * 1000,
                "simplify_ms": timing.get("simplify", 0.0) * 1000,
                "evaluate_ms": timing.get("evaluate", 0.0) * 1000,
                "total_ms": timing["total"] * 1000,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "expr_short": str((trees or created_trees)[current_tree_index])
                if current_tree_index is not None and (trees or created_trees)
                else None,
            }
        )
        return TaskResult(
            candidates=[],
            lut_tree_entries={},  # LUT entries stay in shared dicts
            lut_symex_entries={},
            timing=timing,
            error=_format_task_error_message(e, debug),
            tag=task.tag,
            debug=debug,
            tree_timings=tree_timing_records,
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
    tree_timing_records = []
    current_tree_index = None
    stage = "setup"

    # Set seed if provided
    if task.seed is not None:
        task_seed = task.seed * 10000 + task.task_index
        random.seed(task_seed)
        np.random.seed(task_seed % (2**31))

    try:
        # Phase 1: Create tree(s)
        stage = "create"

        registry = strategy_registry
        if task.strategy_name not in registry:
            raise ValueError(f"Unknown strategy: {task.strategy_name}")

        strategy_fn = registry[task.strategy_name]

        # Inject runtime context for strategies that need training data
        _effective_params = dict(task.strategy_params)
        if task.strategy_name in _STRATEGIES_NEEDING_TRAINING_DATA:
            _effective_params.setdefault("_df_train", df_train)
            if target_column and df_train is not None:
                _effective_params.setdefault("_target", df_train[target_column].values)

        result = strategy_fn(evolve, pop_genepool, paretofront, allow_chain, **_effective_params)
        timing["create"] = time.perf_counter() - t0

        # Phase 2: Simplify
        stage = "simplify"
        t1 = time.perf_counter()
        simplify_durations = []
        if task.crossover:
            tree_a, tree_b = result
            raw_trees = [tree_a, tree_b]
        else:
            tree = result
            raw_trees = [tree]

        trees = []
        for raw_tree in raw_trees:
            tree_simplify_start = time.perf_counter()
            tree_now = tree_simplification(raw_tree, allow_chain=allow_chain) if task.simplicate else raw_tree
            _validate_post_simplify_tree_requirements(task, tree_now)
            simplify_durations.append(time.perf_counter() - tree_simplify_start)
            trees.append(tree_now)
        timing["simplify"] = time.perf_counter() - t1

        # Phase 3: Evaluate
        stage = "evaluate"
        t2 = time.perf_counter()
        candidates = []
        # Pre-compute true_values once for this task (avoid per-tree DataFrame→NumPy copy)
        _seq_true_values = df_train[target_column].to_numpy() if df_train is not None else None
        for tree_index, tree in enumerate(trees):
            current_tree_index = tree_index
            tree_eval_start = time.perf_counter()
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
                true_values=_seq_true_values,
            )
            candidates.append(candidate)
            evaluate_duration = time.perf_counter() - tree_eval_start
            tree_timing_records.append(
                {
                    "tag": task.tag,
                    "task_index": task.task_index,
                    "tree_index": tree_index,
                    "status": "ok",
                    "failed_stage": None,
                    "create_ms_shared": timing.get("create", 0.0) * 1000,
                    "simplify_ms": simplify_durations[tree_index] * 1000,
                    "evaluate_ms": evaluate_duration * 1000,
                    "total_ms": (timing.get("create", 0.0) + simplify_durations[tree_index] + evaluate_duration) * 1000,
                    "fitness": candidate.fitness,
                    "parsimony": candidate.parsimony,
                    "expr_short": str(candidate.tree),
                }
            )
        timing["evaluate"] = time.perf_counter() - t2
        timing["total"] = time.perf_counter() - t0

        return TaskResult(
            candidates=candidates,
            lut_tree_entries={},  # Already written to main LUTs
            lut_symex_entries={},
            timing=timing,
            error=None,
            tag=task.tag,
            tree_timings=tree_timing_records,
        )

    except (TreeError, TreeSizeError, SympyError, ValueError, ArithmeticError, KeyError, RecursionError) as e:
        timing["total"] = time.perf_counter() - t0
        tree_timing_records.append(
            {
                "tag": task.tag,
                "task_index": task.task_index,
                "tree_index": current_tree_index,
                "status": "error",
                "failed_stage": stage,
                "create_ms_shared": timing.get("create", 0.0) * 1000,
                "simplify_ms": timing.get("simplify", 0.0) * 1000,
                "evaluate_ms": timing.get("evaluate", 0.0) * 1000,
                "total_ms": timing["total"] * 1000,
                "error_type": type(e).__name__,
                "error_message": str(e),
                "expr_short": str(trees[current_tree_index])
                if current_tree_index is not None and current_tree_index < len(trees)
                else None,
            }
        )
        return TaskResult(
            candidates=[],
            lut_tree_entries={},
            lut_symex_entries={},
            timing=timing,
            error=f"{type(e).__name__}: {e}",
            tag=task.tag,
            tree_timings=tree_timing_records,
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
        tag = str(strategy.params.get("init_label", strategy.name))
        n = int(strategy.count) if strategy.count is not None else int(strategy.rate * pop_max_size)
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
                    tag=tag,
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
    progress_callback: Optional[Callable[[int, int, int, Optional[str]], None]] = None,
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
    total_candidates_expected = sum(2 if task.crossover else 1 for task in tasks)
    created_candidates = 0
    failed_tasks = 0

    gen_start = time.perf_counter()

    if progress_callback is not None:
        progress_callback(created_candidates, total_candidates_expected, failed_tasks, None)

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
                log(
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
                        failed_tasks += 1
                        _print_parallel_failure_debug(result)
                        continue

                    # Repair tree back-references stripped during pickling
                    for candidate in result.candidates:
                        candidate.tree.repair_all()
                        if _cfg.lut_enabled:
                            lut_tree_delta[candidate.tree.get_lut_id()] = {
                                "parsimony": candidate.parsimony,
                                "fitness": candidate.fitness,
                            }
                    all_candidates.extend(result.candidates)
                    created_candidates += len(result.candidates)

                if progress_callback is not None:
                    batch_label = batch[0].tag if len(batch) == 1 else None
                    progress_callback(created_candidates, total_candidates_expected, failed_tasks, batch_label)

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
    progress_callback: Optional[Callable[[int, int, int, Optional[str]], None]] = None,
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
    total_candidates_expected = sum(2 if task.crossover else 1 for task in tasks)
    created_candidates = 0
    failed_tasks = 0

    fail_counts: Dict[str, int] = {}
    tag_expected: Dict[str, int] = {}
    for task in tasks:
        tag_expected[task.tag] = tag_expected.get(task.tag, 0) + 1

    gen_start = time.perf_counter()

    if progress_callback is not None:
        progress_callback(created_candidates, total_candidates_expected, failed_tasks, None)

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
            failed_tasks += 1
            fail_counts[tag] = fail_counts.get(tag, 0) + 1
            if fail_counts[tag] > budget:
                log_error(
                    f"Strategy '{tag}' exceeded failure budget "
                    f"({fail_counts[tag]}/{budget}). Skipping remaining tasks.",
                )
        else:
            all_candidates.extend(result.candidates)
            created_candidates += len(result.candidates)

        if progress_callback is not None:
            progress_callback(created_candidates, total_candidates_expected, failed_tasks, task.tag)

    tracker.set_generation_time(time.perf_counter() - gen_start)
    return all_candidates, tracker
