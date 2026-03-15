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

import random
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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
    """

    strategy_name: str
    strategy_params: Dict[str, Any]
    crossover: bool
    simplicate: bool
    tag: str
    task_index: int
    seed: Optional[int] = None


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
):
    """Initialize global state in each worker process.

    Called once per worker by ProcessPoolExecutor(initializer=...).
    """
    global _worker_evolve, _worker_df_train, _worker_pop_genepool
    global _worker_paretofront, _worker_eval_autocast, _worker_eval_error_metric
    global _worker_allow_chain, _worker_target_column
    global _worker_nodes_max, _worker_complexity_metric, _worker_strategy_registry

    _worker_evolve = evolve
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
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
    return selection_tournament(pop_genepool, n=tournament_n)


def _strategy_mutation(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate a branch."""
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
    depth_goal = params.get("depth_goal", 3)
    p_term = params.get("p_term", 0.3)
    tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_branch_depth(tree, depth_goal, allow_chain, p_term=p_term)


def _strategy_mutation_point(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate a single point."""
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
    tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_point(tree)


def _strategy_mutation_branch_nodes(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and mutate by target node count."""
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
    nodes_goal = params.get("nodes_goal", 4)
    p_term = params.get("p_term", 0.2)
    tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_mutate_branch_nodes(tree, nodes_goal, p_term=p_term)


def _strategy_mutation_filter(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and apply filter mutation (constant tuning)."""
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
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
    from plagih.trees import selection_tournament

    tournament_n = params.get("tournament_n", 3)
    tree_a = selection_tournament(pop_genepool, n=tournament_n)
    tree_b = selection_tournament(pop_genepool, n=tournament_n)
    return evolve.evolve_crossover(tree_a, tree_b)


def _strategy_simplicate(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Select a tree and simplify it via sympy."""
    from plagih.trees import evolve_reduce_simplicate, selection_tournament

    tournament_n = params.get("tournament_n", 3)
    completely = params.get("completely", True)
    tree = selection_tournament(pop_genepool, n=tournament_n)
    return evolve_reduce_simplicate(tree, allow_chain, completely=completely)


def _strategy_pareto_revive(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Revive a random candidate from the Pareto front."""
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


# =============================================================================
# Worker Task Execution
# =============================================================================


def _worker_run_task(task: TaskSpec) -> TaskResult:
    """Execute a single tree-creation task in a worker process.

    Uses global worker state (set by _init_worker) for evolve, df_train, etc.
    Builds a worker-local LUT to avoid cross-process synchronization.

    Args:
        task: TaskSpec describing what to do.

    Returns:
        TaskResult with candidate(s), LUT deltas, timing, and error info.
    """
    from plagih.trees import tree_simplification

    timing = {}
    lut_tree = {}
    lut_symex = {}
    t0 = time.perf_counter()  # Start timing before try block

    # Set seed if provided (for reproducibility)
    if task.seed is not None:
        task_seed = task.seed * 10000 + task.task_index
        random.seed(task_seed)
        np.random.seed(task_seed % (2**31))

    try:
        # Phase 1: Create tree(s) via strategy function

        # Look up strategy function
        registry = _worker_strategy_registry or BUILTIN_STRATEGIES
        if task.strategy_name not in registry:
            raise ValueError(f"Unknown strategy: {task.strategy_name}")

        strategy_fn = registry[task.strategy_name]
        result = strategy_fn(
            _worker_evolve,
            _worker_pop_genepool,
            _worker_paretofront,
            _worker_allow_chain,
            **task.strategy_params,
        )

        timing["create"] = time.perf_counter() - t0

        # Phase 2: Simplify if requested
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
            lut_tree_entries=lut_tree,
            lut_symex_entries=lut_symex,
            timing=timing,
            error=None,
            tag=task.tag,
        )

    except (TreeError, TreeSizeError, SympyError, ValueError, ArithmeticError, KeyError, RecursionError) as e:
        timing["total"] = time.perf_counter() - t0
        return TaskResult(
            candidates=[],
            lut_tree_entries=lut_tree,
            lut_symex_entries=lut_symex,
            timing=timing,
            error=f"{type(e).__name__}: {e}",
            tag=task.tag,
        )


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
) -> Tuple[list, Dict, Dict, PerformanceTracker]:
    """Run all tasks in parallel using ProcessPoolExecutor.

    Design: Pool is created fresh per generation (~100ms overhead).
    This guarantees consistent genepool state without needing
    a broadcast update mechanism for persistent pools.

    Args:
        tasks: List of TaskSpec objects.
        n_workers: Number of worker processes.
        evolve, df_train, ...: GP state for worker initialization.
        strategy_registry: Combined builtin + custom strategies.

    Returns:
        Tuple of (candidates, lut_tree_delta, lut_symex_delta, tracker).
    """
    tracker = PerformanceTracker()
    all_candidates = []
    lut_tree_delta = {}
    lut_symex_delta = {}

    # Track failures per strategy tag for budget enforcement
    fail_counts: Dict[str, int] = {}
    success_counts: Dict[str, int] = {}
    # Compute expected counts for budget calculation
    tag_expected: Dict[str, int] = {}
    for task in tasks:
        tag_expected[task.tag] = tag_expected.get(task.tag, 0) + 1

    gen_start = time.perf_counter()

    # Design: Pool-Neustart pro Generation statt persistent.
    # ~100ms Overhead, dafür garantiert konsistenter Genepool-State.
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(
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
        ),
    ) as executor:
        futures = {executor.submit(_worker_run_task, task): task for task in tasks}

        for future in as_completed(futures):
            task = futures[future]
            tag = task.tag

            try:
                result = future.result()
            except Exception as e:
                # Unexpected worker crash
                result = TaskResult(
                    candidates=[],
                    lut_tree_entries={},
                    lut_symex_entries={},
                    timing={"total": 0.0},
                    error=f"WorkerCrash: {e}",
                    tag=tag,
                )

            tracker.record(result)

            if result.error is not None:
                fail_counts[tag] = fail_counts.get(tag, 0) + 1
                n_expected = tag_expected.get(tag, 1)
                budget = 2 * n_expected + 5
                if fail_counts[tag] > budget:
                    print_warning(
                        "ww",
                        f"Strategy '{tag}' exceeded failure budget "
                        f"({fail_counts[tag]}/{budget}). Some tasks may not complete.",
                    )
            else:
                success_counts[tag] = success_counts.get(tag, 0) + 1
                all_candidates.extend(result.candidates)
                # Merge worker-local LUTs: same key = same result, simple update
                lut_tree_delta.update(result.lut_tree_entries)
                lut_symex_delta.update(result.lut_symex_entries)

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
