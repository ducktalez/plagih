"""GP Engine module: ExplainableGP and picklable helper callables."""

import pickle
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import sympy

from plagih.config import cfg as _cfg
from plagih.logging_utils import _flush_progress_line
from plagih.monitoring import GPMonitor
from plagih.paretofront import *
from plagih.trees._evolution import *
from plagih.trees._nodes import *
from plagih.util import *

# =============================================================================
# Picklable helper callables for ProcessPoolExecutor compatibility
# =============================================================================
# Closures and lambdas are NOT picklable. These classes wrap the same
# logic in a picklable __call__ object.


class _ClipAutocast:
    """Picklable autocast that clips predictions to [lo, hi]."""

    def __init__(self, lo: float, hi: float):
        self.lo = lo
        self.hi = hi

    def __call__(self, x):
        return np.clip(np.asarray(x, dtype=np.float64), self.lo, self.hi)


def _error_rmse(pred, true):
    """RMSE error metric (picklable top-level function)."""
    return np.sqrt(np.mean((pred - true) ** 2))


def _error_mse(pred, true):
    """MSE error metric (picklable top-level function)."""
    return np.mean((pred - true) ** 2)


def _error_mae(pred, true):
    """MAE error metric (picklable top-level function)."""
    return np.mean(np.abs(pred - true))


def _default_autocast(x):
    """Default autocast: convert to float64 numpy array (picklable top-level function)."""
    return np.asarray(x, dtype=np.float64)


def _warmup_noop():
    """Trivial picklable function for pool warmup (lambdas can't be pickled on Windows)."""
    return True


class ExplainableGP:
    """Main class for explainable genetic programming.

    Manages the complete GP workflow including population evolution,
    fitness evaluation, Pareto front maintenance, and monitoring.

    Attributes:
        evolve: Evolution instance for tree operations.
        df_train: Training DataFrame with input features and target.
        rootdir: Output directory for logs, plots, and backups.
        pop_max_size: Maximum population size per generation.
        gen_end: Target number of generations.
        paretofront: List of non-dominated Candidates.
        pop_genepool: Current generation's population.
        lut_tree_infos: Cache for tree metadata (sympy expr, fitness, parsimony).
        lut_symex_fitness: Cache mapping sympy expressions to fitness values.
        monitor_df: DataFrame tracking generation statistics.

    Example:
        # Simple usage with defaults
        gp = ExplainableGP.create(
            symbols=['x', 'y'],
            df_train=my_data,
            rootdir='./results'
        )

        # Or with custom Evolution
        gp = ExplainableGP(
            evolve=my_evolution,
            df_train=my_data,
            rootdir=Path('./results'),
            pop_max_size=50,
            gen_end=20
        )
    """

    # Default error metric and autocast as picklable top-level references
    # (lambdas and closures are NOT picklable for ProcessPoolExecutor on Windows)
    DEFAULT_ERROR_METRIC = staticmethod(_error_rmse)
    DEFAULT_AUTOCAST = staticmethod(_default_autocast)

    def __init__(
        self,
        evolve: Evolution,
        df_train: pd.DataFrame,
        rootdir: Union[Path, str],
        *,  # Force keyword-only args after this
        pop_max_size: int = 100,
        gen_end: int = 100,
        eval_autocast: Optional[Callable] = None,
        eval_error_metric: Optional[Callable] = None,
        allow_chain: bool = False,
        target_column: str = "action",
        verbose: bool = True,
        parallel: Union[bool, int] = None,
        enable_analysis: Optional[bool] = None,
    ):
        """Initialize the GP system.

        Args:
            evolve: Evolution instance with operator pool and constraints.
            df_train: Training data with target column.
            rootdir: Path for output files (str or Path).
            pop_max_size: Maximum individuals per generation. Default: 100.
            gen_end: Number of generations to run. Default: 100.
            eval_autocast: Function to cast predictions. Default: np.array.
            eval_error_metric: Error function(pred, true) -> float. Default: RMSE.
            allow_chain: Whether to allow chained operators. Default: False.
            target_column: Name of target column in df_train. Default: 'action'.
            verbose: Print info. Default: True.
            parallel: None=use .env default, False/0=sequential,
                True=auto-detect, int=explicit workers.
            enable_analysis: Enable plots, backups, visualizations.
                None=use .env default. Set to False for benchmarks.

        Returns:
            Configured ExplainableGP instance.

        Example:
            gp = ExplainableGP.create(
                symbols=['x', 'y'],
                df_train=data,
                rootdir='./run_001',
                preset='math_simple',
                pop_max_size=50,
                gen_end=20
            )
        """
        self.time_start = time.perf_counter()

        # Handle rootdir as str or Path
        self.rootdir = Path(rootdir) if isinstance(rootdir, str) else rootdir
        self.rootdir.mkdir(parents=True, exist_ok=True)

        self.df_train = df_train
        self.target_column = target_column

        self.evolve = evolve
        self.gen_end = gen_end
        self.pop_max_size = pop_max_size
        self.gen_id: int = 0

        # Use defaults if not provided
        self.eval_autocast = eval_autocast or self.DEFAULT_AUTOCAST
        self.eval_error_metric = eval_error_metric or self.DEFAULT_ERROR_METRIC

        self.allow_chain = allow_chain

        # Analysis control: when False, skips plots, backups, and merged tree rendering.
        # None → resolve from .env / PlagihConfig
        self.enable_analysis = enable_analysis if enable_analysis is not None else _cfg.visualization

        if verbose:
            print(
                f"\n"
                f"\tInitializing Plagih.\n"
                f"\tName: {BColors.CYAN}{self.rootdir.name}{BColors.RESET_COLOR}.\n"
                f"\tLocated in: \n"
                f"\t{self.rootdir}\n"
            )

        self.paretofront = []
        self.pop_genepool = []
        self.pop_next = []

        self.lut_tree_infos = {}
        self.lut_symex_fitness = {}

        # monitoring
        self.time_genstart = time.perf_counter()
        self.monitor = GPMonitor()

        # Parallel execution config
        # None → resolve from .env / PlagihConfig

        from plagih.parallel import BUILTIN_STRATEGIES
        from plagih.util import cpu_count_physical

        if parallel is None:
            # Resolve from config: 0 = sequential
            parallel = _cfg.parallel
        if parallel is True:
            self._parallel_workers = cpu_count_physical()
        elif isinstance(parallel, int) and parallel > 0:
            self._parallel_workers = parallel
        else:
            self._parallel_workers = 0  # 0 = sequential mode
        self._custom_strategies: Dict[str, Callable] = {}
        self._strategy_registry = dict(BUILTIN_STRATEGIES)  # copy builtin registry
        self._performance_tracker = None  # Set per generation
        self._generation_tree_timings = []
        self._latest_generation_tree_timings = []

        # Persistent worker pool (avoids ~2s Windows spawn overhead per generation).
        # Created lazily on first parallel generation, shutdown on close().
        self._pool = None
        self._pool_init_resources = None

    def _get_or_create_pool(self):
        """Get or lazily create the persistent worker pool.

        On Windows, spawning new processes is expensive (~2s due to Python
        interpreter startup + module imports). A persistent pool amortizes
        this cost across all generations.
        """
        if self._pool is None and self._parallel_workers > 0:
            from concurrent.futures import ProcessPoolExecutor

            from plagih.parallel import _init_worker, build_worker_init_config, cleanup_worker_init_resources

            initargs, self._pool_init_resources = build_worker_init_config(
                evolve=self.evolve,
                df_train=self.df_train,
                pop_genepool=[],  # pop_genepool: not needed — pre-selection in main process
                paretofront=[],  # paretofront: not needed — pre-selection in main process
                eval_autocast=self.eval_autocast,
                eval_error_metric=self.eval_error_metric,
                allow_chain=self.allow_chain,
                target_column=self.target_column,
                nodes_max=self.evolve.nodes_max,
                complexity_metric=self.evolve.complexity_metric,
                strategy_registry=self._strategy_registry,
            )
            try:
                self._pool = ProcessPoolExecutor(
                    max_workers=self._parallel_workers,
                    initializer=_init_worker,
                    initargs=cast(tuple[Any, ...], initargs),
                )
                # Warm up: submit a trivial task to force all workers to start
                # and import modules. This way the first real generation doesn't
                # pay the import cost.
                warmup_futures = [self._pool.submit(_warmup_noop) for _ in range(self._parallel_workers)]
                for f in warmup_futures:
                    f.result()
                log("i", f"Worker pool created with {self._parallel_workers} workers")
            except Exception:
                if self._pool is not None:
                    self._pool.shutdown(wait=True)
                    self._pool = None
                cleanup_worker_init_resources(self._pool_init_resources)
                self._pool_init_resources = None
                raise
        return self._pool

    def close(self):
        """Shutdown the persistent worker pool. Call when done with GP."""
        from plagih.parallel import cleanup_worker_init_resources

        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        cleanup_worker_init_resources(self._pool_init_resources)
        self._pool_init_resources = None

    def __del__(self):
        """Ensure pool is cleaned up on garbage collection."""
        try:
            self.close()
        except Exception:
            pass

    # =========================================================================
    # Backwards Compatibility Properties
    # =========================================================================

    @property
    def monitor_df(self):
        """Backwards compatible access to monitoring DataFrame.

        Returns the monitor data as a pandas DataFrame with old column names.
        """
        return self.monitor.to_dataframe()

    @property
    def gens_since_last_pareto(self):
        """Backwards compatible access to generations since last Pareto update."""
        return self.monitor.gens_since_last_pareto

    @classmethod
    def create(
        cls,
        symbols: List[Union[str, sympy.Symbol]],
        df_train: pd.DataFrame,
        rootdir: Union[Path, str],
        *,
        operators: Optional[Dict] = None,
        preset: str = "math_full",
        depth_max: int = 7,
        nodes_max: int = 40,
        pop_max_size: int = 100,
        gen_end: int = 50,
        clip_range: Optional[Tuple[float, float]] = None,
        error_metric: str = "rmse",
        allow_chain: bool = False,
        target_column: str = "action",
        verbose: bool = True,
        parallel: Union[bool, int] = None,
        enable_analysis: Optional[bool] = None,
    ) -> "ExplainableGP":
        """Factory method for easy GP creation with sensible defaults.

        All feature-flag defaults (parallel, enable_analysis, …) are resolved
        from the ``.env`` / ``PlagihConfig`` unless explicitly overridden here.

        Args:
            symbols: List of input variable names or sympy Symbols.
            df_train: Training DataFrame with features and target.
            rootdir: Output directory path.
            operators: Custom operator dict. If None, uses preset.
            preset: Operator preset name ('math_simple', 'math_full', 'with_logic').
            depth_max: Maximum tree depth. Default: 7.
            nodes_max: Maximum nodes per tree. Default: 40.
            pop_max_size: Population size. Default: 100.
            gen_end: Number of generations. Default: 50.
            clip_range: Optional (min, max) to clip predictions.
            error_metric: 'rmse', 'mse', 'mae', or custom callable.
            allow_chain: Allow chained operators. Default: False.
            target_column: Target column name. Default: 'action'.
            verbose: Print info. Default: True.
            parallel: None=use .env default, False/0=sequential,
                True=auto-detect, int=explicit workers.
            enable_analysis: Enable plots, backups, visualizations.
                None=use .env default. Set to False for benchmarks.

        Returns:
            Configured ExplainableGP instance.

        Example:
            gp = ExplainableGP.create(
                symbols=['x', 'y'],
                df_train=data,
                rootdir='./run_001',
                preset='math_simple',
                pop_max_size=50,
                gen_end=20
            )
        """
        # Create Evolution with preset or custom operators
        # Evolution accepts: operators as dict, string (preset name), or list
        ops = operators if operators is not None else preset
        evolve = Evolution(
            symbol_list=symbols, operators=ops, depth_max=depth_max, nodes_max=nodes_max, allow_chain=allow_chain
        )

        # Setup autocast
        if clip_range:
            eval_autocast = _ClipAutocast(clip_range[0], clip_range[1])
        else:
            eval_autocast = None  # Will use default

        # Setup error metric
        # Uses picklable top-level functions (not lambdas) for ProcessPoolExecutor.
        if callable(error_metric):
            eval_error_metric = error_metric
        elif error_metric == "rmse":
            eval_error_metric = _error_rmse
        elif error_metric == "mse":
            eval_error_metric = _error_mse
        elif error_metric == "mae":
            eval_error_metric = _error_mae
        else:
            raise ValueError(f"Unknown error_metric: {error_metric}. Use 'rmse', 'mse', 'mae', or callable.")

        return cls(
            evolve=evolve,
            df_train=df_train,
            rootdir=rootdir,
            pop_max_size=pop_max_size,
            gen_end=gen_end,
            eval_autocast=eval_autocast,
            eval_error_metric=eval_error_metric,
            allow_chain=allow_chain,
            target_column=target_column,
            verbose=verbose,
            parallel=parallel,
            enable_analysis=enable_analysis,
        )

    @classmethod
    def from_config(cls, config: dict, df_train: pd.DataFrame) -> "ExplainableGP":
        """Create GP instance from configuration dictionary.

        Args:
            config: Dictionary with GP configuration.
            df_train: Training DataFrame.

        Returns:
            Configured ExplainableGP instance.

        Example:
            config = {
                'symbols': ['x', 'y'],
                'rootdir': './results',
                'preset': 'math_simple',
                'pop_max_size': 50,
                'gen_end': 20
            }
            gp = ExplainableGP.from_config(config, my_data)
        """
        return cls.create(df_train=df_train, **config)

    def get_name(self):
        """Returns the name of this GP run (derived from rootdir)."""
        if isinstance(self.rootdir, Path):
            s = self.rootdir.name
        else:
            s = None
        return s

    def run_update_paretofront(self, pop):
        """Updates the Pareto front with non-dominated candidates from pop.

        Minimizes both fitness and parsimony. A candidate dominates another
        if it is better in at least one objective and not worse in any.

        Args:
            pop: Population to extract Pareto-optimal candidates from.

        Returns:
            True if the Pareto front changed, False otherwise.
        """
        new_cands = pareto_from_pop(pop) or []
        if not new_cands:
            return False

        old_front = list(self.paretofront) if self.paretofront else []
        old_best_par = min((c.get_parsim() for c in old_front), default=float("inf"))
        old_best_fit = min((c.get_fitness() for c in old_front), default=float("inf"))

        combined = old_front + new_cands

        def dominated(a, b):
            pa, fa = a.get_parsim(), a.get_fitness()
            pb, fb = b.get_parsim(), b.get_fitness()
            return (pb <= pa and fb <= fa) and (pb < pa or fb < fa)

        new_front = []
        for c in combined:
            if any(dominated(c, o) for o in combined if o is not c):
                continue
            # Duplikate (gleiche Metriken) vermeiden
            if not any((c.get_parsim() == e.get_parsim() and c.get_fitness() == e.get_fitness()) for e in new_front):
                new_front.append(c)

        # Sortierung für stabile Ausgabe/Weiterverarbeitung
        new_front.sort(key=lambda x: (x.get_parsim(), x.get_fitness()))

        # Änderung erkennen
        old_keys = {(cand.get_parsim(), cand.get_fitness(), cand.full_string()) for cand in old_front}
        new_keys = {(cand.get_parsim(), cand.get_fitness(), cand.full_string()) for cand in new_front}
        changed = new_keys != old_keys

        if changed:
            self.paretofront = new_front
            new_best_par = min(c.get_parsim() for c in new_front)
            new_best_fit = min(c.get_fitness() for c in new_front)

            if old_front and new_best_par < old_best_par:
                log(
                    "a",
                    f"Paretofront: Neuer simpelster Eintrag. parsimony: {new_best_par} old simplest had {old_best_par}",
                )
            if old_front and new_best_fit < old_best_fit:
                log(
                    "a",
                    f"Paretofront: Neuer fittester Eintrag. fitness: {new_best_fit:6.4f} "
                    f"old best had {old_best_fit:6.4f}",
                )
            if not old_front:
                log("a", f"Paretofront initialisiert mit {len(new_front)} Kandidaten.")

            # Note: gens_since_last_pareto is now tracked by self.monitor

        return changed

    def end_generation(self, _suppress_analyze_print: bool = False):
        """Finalizes the current generation and prepares for the next.

        Actions:
        - Updates Pareto front with new candidates
        - Moves pop_next to pop_genepool
        - Prints population summary (only at high verbosity, expensive for large pops)
        - Runs analysis and monitoring
        - Increments generation counter

        Args:
            _suppress_analyze_print: If True, suppress the generation summary print
                in analyze_generation (used by run_generation which prints its own
                progress-style summary).
        """
        pareto_updated = self.run_update_paretofront(self.pop_next)

        self.pop_genepool = self.pop_next[:]
        # Pitfall P3: print_pop calls get_sympy_expr() per candidate (~40ms/tree).
        # Guarded by "gggg" in verbosity - skipped at default verbosity.
        if "gggg" in _cfg.verbosity:
            print_pop(self.pop_next)
        self.pop_next = []
        self.analyze_generation(pareto_updated=pareto_updated, _suppress_print=_suppress_analyze_print)
        self.gen_id += 1

        self.time_genstart = time.perf_counter()

    # =========================================================================
    # Declarative Strategy API
    # =========================================================================

    def register_strategy(self, name: str, fn: Callable):
        """Register a custom strategy function.

        The function must be a top-level function (picklable for parallel mode).
        Signature: fn(evolve, pop_genepool, paretofront, allow_chain, **params) -> Node | Tuple[Node, Node]

        Args:
            name: Name to register the strategy under.
            fn: Strategy function.

        Example:
            def my_strategy(evolve, pop_genepool, paretofront, allow_chain, **params):
                tree = selection_tournament(pop_genepool, n=3)
                return evolve.evolve_mutate_point(tree)

            gp.register_strategy("my_mutation", my_strategy)
        """
        self._custom_strategies[name] = fn
        self._strategy_registry[name] = fn

    def run_generation(self, strategies, *, parallel: Optional[bool] = None, seed: Optional[int] = None):
        """Run a complete generation using declarative strategies.

        Creates trees according to the strategy list, evaluates them,
        updates the Pareto front, and prepares for the next generation.

        Supports both sequential (debuggable) and parallel execution.
        In sequential mode, the same code path is used as in parallel,
        just without a ProcessPoolExecutor — allowing full debugging
        with breakpoints.

        Args:
            strategies: List of Strategy objects defining the generation.
            parallel: Override parallel mode. None=use self._parallel_workers.
                      True=parallel with auto-detect, False=sequential.
            seed: Optional base seed for reproducibility. Each task gets
                  seed = base_seed * 10000 + task_index.

        Example:
            from plagih.parallel import Strategy

            gp.run_generation([
                Strategy("reproduction", rate=0.2, tournament_n=3),
                Strategy("mutation", rate=0.4, depth_goal=3, p_term=0.3),
                Strategy("random_new", rate=0.2, depths=[2, 3, 4]),
                Strategy("crossover", rate=0.2, crossover=True, tournament_n=3),
            ])
        """
        from plagih.parallel import (
            build_task_list,
            run_generation_parallel,
            run_generation_sequential,
        )

        # Determine execution mode
        if parallel is True:
            from plagih.util import cpu_count_physical

            n_workers = cpu_count_physical()
        elif parallel is False or parallel == 0:
            n_workers = 0
        elif isinstance(parallel, int) and parallel > 0:
            n_workers = parallel
        elif parallel is None:
            n_workers = self._parallel_workers
        else:
            n_workers = 0

        # Build task list from strategies
        tasks = build_task_list(strategies, self.pop_max_size, seed=seed)
        total_candidates_expected = sum(2 if task.crossover else 1 for task in tasks)
        progress_started = time.perf_counter()
        self._generation_tree_timings = []

        def _update_generation_progress(created: int, total: int, fail: int, label: Optional[str]) -> None:
            print_generation_progress(
                gen_id=self.gen_id,
                gen_end=self.gen_end,
                created=created,
                total=total if total > 0 else total_candidates_expected,
                label=label or "create",
                fail=fail,
                elapsed_s=time.perf_counter() - progress_started,
            )

        # Progress start — will be overwritten by the done-line below
        current_gen_id = self.gen_id
        print_generation_start(current_gen_id, self.gen_end)

        if n_workers > 0:
            # Parallel execution — use persistent pool to avoid Windows spawn overhead
            pool = self._get_or_create_pool()
            candidates, lut_tree_delta, lut_symex_delta, tracker = run_generation_parallel(
                tasks=tasks,
                n_workers=n_workers,
                evolve=self.evolve,
                df_train=self.df_train,
                pop_genepool=self.pop_genepool,
                paretofront=self.paretofront,
                eval_autocast=self.eval_autocast,
                eval_error_metric=self.eval_error_metric,
                allow_chain=self.allow_chain,
                target_column=self.target_column,
                nodes_max=self.evolve.nodes_max,
                complexity_metric=self.evolve.complexity_metric,
                strategy_registry=self._strategy_registry,
                pool=pool,
                progress_callback=_update_generation_progress,
            )
            # Note: LUT deltas are empty in parallel mode.
            # Worker LUTs contain sympy objects (too expensive to pickle back).
            # Worker LUTs are used only for intra-batch deduplication.
            # Main-process LUTs won't grow during parallel generations, which
            # causes some redundant computation but avoids pickle overhead.
        else:
            # Sequential execution — identical logic, no pool, full debugging
            candidates, tracker = run_generation_sequential(
                tasks=tasks,
                evolve=self.evolve,
                df_train=self.df_train,
                pop_genepool=self.pop_genepool,
                paretofront=self.paretofront,
                eval_autocast=self.eval_autocast,
                eval_error_metric=self.eval_error_metric,
                allow_chain=self.allow_chain,
                target_column=self.target_column,
                nodes_max=self.evolve.nodes_max,
                complexity_metric=self.evolve.complexity_metric,
                strategy_registry=self._strategy_registry,
                lut_tree=self.lut_tree_infos,
                lut_symex=self.lut_symex_fitness,
                progress_callback=_update_generation_progress,
            )

        # Store candidates as next population
        self.pop_next = candidates
        self._generation_tree_timings = list(tracker.tree_timings)

        # Progress done — overwrites the start-line.
        # Must happen BEFORE end_generation() because end_generation may
        # trigger plots/backups/prints that would interrupt the \r line.
        tracker_total_ms = tracker.summary().get("generation_total_time", 0.0) * 1000
        print_generation_done(
            gen_id=current_gen_id,
            gen_end=self.gen_end,
            time_ms=tracker_total_ms,
            created=len(candidates),
            pareto_pre=len(self.paretofront),
            ok=tracker.total_ok,
            fail=tracker.total_fail,
            tracker_total_ms=tracker_total_ms,
        )

        # Print detailed performance summary (only at higher verbosity)
        tracker.print_summary()

        # Finalize generation (Pareto update, monitoring, plots, backups).
        # This may print file/plot messages — safe now because progress line
        # is already finished.
        self.end_generation(_suppress_analyze_print=True)

        # Store tracker for inspection
        self._performance_tracker = tracker

    def gen_create_initial(self, origin_tree=None):
        """Creates the initial population (generation 0).

        If an origin_tree is provided, adds it as a candidate.
        Otherwise generates random trees with varying depths.

        Args:
            origin_tree: Optional seed _tree to include in initial population.

        Returns:
            The initial population (pop_genepool).
        """
        log("gg", f"generation {self.gen_id}/{self.gen_end} start: create initial population")
        self._generation_tree_timings = []

        if origin_tree is not None:
            cand_origin = self.tree_to_candidate(origin_tree, raise_if_useless=False, tag="origin")
            self.pop_next_append(cand_origin)
        else:
            if self.allow_chain:

                @self.create_trees(rate=0.5)
                def init_rand1():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 4, 6)
                    tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    tree = tree_simplification(tree, allow_chain=self.allow_chain)

                    if tree.get_max_depth() == 0:
                        raise TreeSizeError("Tree did not get complex enough (only root node).")
                    return tree

                @self.create_trees(rate=0.5)
                def init_rand2():
                    n = np.clip(int(random.normalvariate(4.5, 1.0)), 3, self.evolve.depth_max)
                    tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    tree = tree_simplification(tree, allow_chain=self.allow_chain)
                    return tree
            else:

                @self.create_trees(rate=0.5)
                def init_rand1a():
                    n = np.clip(int(random.normalvariate(4.0, 1.0)), 3, 5)
                    _tree = self.evolve.evolve_new_tree_depth(float, n, p_term=0)
                    return _tree

                @self.create_trees(rate=0.5)
                def init_rand2a():
                    n = np.clip(int(random.normalvariate(3.5, 1.0)), 3, self.evolve.depth_max)
                    return self.evolve.evolve_new_tree_depth(float, n, p_term=0)

        self.paretofront = pareto_from_pop(self.pop_next)
        self.pop_genepool = self.pop_next[:]
        self.pop_next = []
        self.analyze_generation(pareto_updated=True)  # Initial population always updates Pareto
        self.gen_id += 1
        return self.pop_genepool

    def pop_next_append(self, ct: Candidate, force=False):
        """Appends a candidate to the next generation's population.

        Logs the tree expression and adds it to pop_next.

        Args:
            ct: The Candidate to add.
            force: If True, skips minimum parsimony check.
        """
        evotree = ct.get_evotree()
        # from visualization.pygraphviz import render_pygraphviz
        if force and ct.get_parsim() < _cfg.tree_min_parsimony:
            # raise ValueError(f'Tree not complex enough for population, sfeh')
            return
        # Guard expensive debug formatting locally: f-strings are evaluated
        # before log() can check verbosity.
        if "gggg" in _cfg.verbosity:
            log("gggg", f"|->{evotree.len_nodecount_fair():2.0f}: {evotree.str_as_expr()}")
        self.pop_next.append(ct)

    def create_trees(self, rate=0.0, crossover=False, simplicate=False, allow_chain=False):
        """Decorator factory for safely creating and adding trees to the population.

        Wraps a tree creation function to handle errors, apply simplification,
        and convert trees to Candidates.

        Args:
            rate: Fraction of pop_max_size to create (0.0 to 1.0).
            crossover: If True, expects function to return two trees.
            simplicate: If True, applies tree_simplification before evaluation.
            allow_chain: Whether to allow chained operators in simplification.

        Returns:
            Decorator function that wraps tree creation logic.
        """

        def loop(create_tree_f):
            n = int(rate * self.pop_max_size)
            n_success = 0
            fails_list = []
            tag = create_tree_f.__name__
            log("ggg", f"->Evolving {n}x '{tag}'...")
            progress_started = time.perf_counter()

            def _update_creation_progress() -> None:
                print_generation_progress(
                    gen_id=self.gen_id,
                    gen_end=self.gen_end,
                    created=min(n_success, n),
                    total=n,
                    label=tag,
                    fail=len(fails_list),
                    elapsed_s=time.perf_counter() - progress_started,
                )

            if n > 0:
                _update_creation_progress()

            while n_success < n:
                attempt_started = time.perf_counter()
                failed_stage = "create"
                try:
                    if crossover:
                        create_started = time.perf_counter()
                        t1, t2 = create_tree_f()
                        create_duration = time.perf_counter() - create_started
                        if simplicate:
                            failed_stage = "simplify"
                            simplify_started = time.perf_counter()
                            t1 = tree_simplification(t1, allow_chain=self.allow_chain)
                            simplify_duration_1 = time.perf_counter() - simplify_started
                        else:
                            simplify_duration_1 = 0.0
                        failed_stage = "evaluate"
                        evaluate_started = time.perf_counter()
                        ctree1 = self.tree_to_candidate(t1, tag=tag)
                        evaluate_duration_1 = time.perf_counter() - evaluate_started
                        self._generation_tree_timings.append(
                            {
                                "tag": tag,
                                "task_index": n_success,
                                "tree_index": 0,
                                "status": "ok",
                                "failed_stage": None,
                                "create_ms_shared": create_duration * 1000,
                                "simplify_ms": simplify_duration_1 * 1000,
                                "evaluate_ms": evaluate_duration_1 * 1000,
                                "total_ms": (create_duration + simplify_duration_1 + evaluate_duration_1) * 1000,
                                "fitness": ctree1.fitness,
                                "parsimony": ctree1.parsimony,
                                "expr_short": str(ctree1.tree),
                            }
                        )
                        self.pop_next_append(ctree1)
                        n_success += 1
                        _update_creation_progress()
                        if simplicate:
                            failed_stage = "simplify"
                            simplify_started = time.perf_counter()
                            t2 = tree_simplification(t2, allow_chain=self.allow_chain)
                            simplify_duration_2 = time.perf_counter() - simplify_started
                        else:
                            simplify_duration_2 = 0.0
                        failed_stage = "evaluate"
                        evaluate_started = time.perf_counter()
                        ctree2 = self.tree_to_candidate(t2, tag=tag)
                        evaluate_duration_2 = time.perf_counter() - evaluate_started
                        self._generation_tree_timings.append(
                            {
                                "tag": tag,
                                "task_index": n_success,
                                "tree_index": 1,
                                "status": "ok",
                                "failed_stage": None,
                                "create_ms_shared": create_duration * 1000,
                                "simplify_ms": simplify_duration_2 * 1000,
                                "evaluate_ms": evaluate_duration_2 * 1000,
                                "total_ms": (create_duration + simplify_duration_2 + evaluate_duration_2) * 1000,
                                "fitness": ctree2.fitness,
                                "parsimony": ctree2.parsimony,
                                "expr_short": str(ctree2.tree),
                            }
                        )
                        self.pop_next_append(ctree2)
                        n_success += 1
                        _update_creation_progress()
                    else:
                        create_started = time.perf_counter()
                        evotree = create_tree_f()
                        create_duration = time.perf_counter() - create_started
                        if simplicate:
                            failed_stage = "simplify"
                            simplify_started = time.perf_counter()
                            evotree = tree_simplification(evotree, allow_chain=self.allow_chain)
                            simplify_duration = time.perf_counter() - simplify_started
                        else:
                            simplify_duration = 0.0
                        failed_stage = "evaluate"
                        evaluate_started = time.perf_counter()
                        ctree = self.tree_to_candidate(evotree, tag=tag)
                        evaluate_duration = time.perf_counter() - evaluate_started
                        self._generation_tree_timings.append(
                            {
                                "tag": tag,
                                "task_index": n_success,
                                "tree_index": 0,
                                "status": "ok",
                                "failed_stage": None,
                                "create_ms_shared": create_duration * 1000,
                                "simplify_ms": simplify_duration * 1000,
                                "evaluate_ms": evaluate_duration * 1000,
                                "total_ms": (create_duration + simplify_duration + evaluate_duration) * 1000,
                                "fitness": ctree.fitness,
                                "parsimony": ctree.parsimony,
                                "expr_short": str(ctree.tree),
                            }
                        )
                        self.pop_next_append(ctree)
                        n_success += 1
                        _update_creation_progress()

                except (TreeError, TreeSizeError, SympyError) as e:
                    self._generation_tree_timings.append(
                        {
                            "tag": tag,
                            "task_index": n_success,
                            "tree_index": None,
                            "status": "error",
                            "failed_stage": failed_stage,
                            "total_ms": (time.perf_counter() - attempt_started) * 1000,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                        }
                    )
                    fails_list.append(e)
                    log("www", f"Failed evolution tag '{tag}': {e}")
                    if n > 0:
                        _update_creation_progress()
                    if len(fails_list) > 2 * n_success + 5:  # allow more fails: fails_list > n
                        _flush_progress_line()
                        log_error(f"Evolution fails too often: {tag}, failed: {len(fails_list)}x. ({n_success} ok).")
                        return  # optional raise

                except (ValueError, ArithmeticError) as e:
                    # if 'Crossover tree 1 has no mutable nodes!' in str(ex):
                    if "'a' cannot be empty unless no samples are taken" in str(
                        e
                    ) or "The argument 'zoo' is not comparable" in str(e):
                        log("ww", f"OnlyPrintException: {e}")
                        if n > 0:
                            _update_creation_progress()

                except KeyError as e:
                    # KeyError(re) -> okay?, real part implies complex numbers, ignoring is okay
                    # (probably sympy.lambdify expression not evaluable)
                    log("ww", f"OnlyPrintException: Keyerror?: {e}")
                    if n > 0:
                        _update_creation_progress()
                except RecursionError as e:
                    log("ww", f"OnlyPrintException: RecursionError (probably Piecewise/relational combination?): {e}")
                    if n > 0:
                        _update_creation_progress()
                # 2026: (ww) OnlyPrintException: RecursionError (probably Piecewise/relational combination?): maximum recursion depth exceeded
                # -> sfeh: how did this happen
                # except NotImplementedError as nie:
                #     log_error(f'Notimplemented? {nie}')
                # except Exception as ex:
                #     print(f'OnlyPrintException: Why are we not here??? {ex}')

            _flush_progress_line()

        return loop

    def tree_to_candidate(
        self, evotree: Node, origin_tree=None, tag=None, raise_if_useless=True, compare_with_sympy=None
    ):  # compare_with_sympy defaults to cfg.debug
        """Converts a tree to a fully evaluated Candidate.

        Process:
        1. Ensures tree has input variables
        2. Prunes if necessary
        3. Computes sympy expression
        4. Evaluates fitness using NumPy
        5. Computes parsimony

        Uses lookup tables to avoid redundant computation.

        Args:
            evotree: The tree to convert.
            origin_tree: Reference tree for edit distance (if used).
            tag: Label indicating which evolution created this tree.
            raise_if_useless: If True, raises error for oversized trees.
            compare_with_sympy: If True, validates NumPy results against SymPy.

        Returns:
            A Candidate object with tree, fitness, and parsimony.

        Raises:
            TreeLutError: If cached tree had errors.
            TreeSizeError: If tree exceeds max nodes.
            SympyError: If sympy expression cannot be created.
        """

        # Make this tree usable for evaluation
        if compare_with_sympy is None:
            compare_with_sympy = _cfg.debug
        evotree.force_input_node(self.evolve)
        evotree = self.evolve.evolve_prune_tree(evotree)
        evotree.repair_depth()
        evotree.canonicalize_children()  # Deterministic child order for commutative ops → consistent LUT keys

        tree_id = evotree.get_lut_id()

        if _cfg.lut_enabled and tree_id in self.lut_tree_infos:
            sy_expr = self.lut_tree_infos[tree_id].get("sy_expr")  # Attention: can be "False"
            parsimony = self.lut_tree_infos[tree_id].get("parsimony")
            fitness = self.lut_tree_infos[tree_id].get("fitness")
            if any(v is None for v in [sy_expr, parsimony, fitness]):
                _err = self.lut_tree_infos[tree_id].get("error")
                raise TreeLutError(f"Tree LUT Entry implies Problem: {_err}")
        else:
            # requires: valid, sympy expr, parsimony, fitness
            if _cfg.lut_enabled:
                self.lut_tree_infos[tree_id] = {}  # empty placeholder, if correctly filled later

            parsimony = eval_parsimony(evotree, self.evolve.complexity_metric, origin_tree=origin_tree)
            if raise_if_useless and parsimony > self.evolve.nodes_max:
                err_txt = f"Tree too complex: {parsimony} > {self.evolve.nodes_max}"
                if _cfg.lut_enabled:
                    self.lut_tree_infos[tree_id]["error"] = err_txt
                raise TreeSizeError(err_txt)
            try:
                sy_expr = evotree.get_sympy_expr()
            except SympyError as e:
                log("www", f"Could not create sympy expression for tree: {e}")
                if _cfg.lut_enabled:
                    self.lut_tree_infos[tree_id]["error"] = str(e)
                raise

            if _cfg.lut_enabled and sy_expr in self.lut_symex_fitness:
                # other tree might have same expression -> lookup fitness
                fitness = self.lut_symex_fitness[sy_expr]
            else:
                """Numpy eval"""
                perf_t = {0: time.perf_counter()}
                true_values = self.df_train[self.target_column].to_numpy()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)  # sfeh:discuss
                        np_results_raw = evotree.eval_predict_numpy_now(
                            self.df_train
                        )  # exception? -> check np.isnan(sym_results).any()
                        np_results = self.eval_autocast(np_results_raw)
                        np_fitness = self.eval_error_metric(np_results, true_values)
                        np_fitness = round(np_fitness, _cfg.float_precision)

                        if (
                            "nan" in str(np_fitness) or np_fitness == np.nan or np_fitness == np.inf
                        ):  # sfeh:code not so good looking
                            err_txt = "NaN in results"
                            self.lut_tree_infos[tree_id]["error"] = err_txt
                            raise TreeError(f"{err_txt}")

                        perf_t[1] = time.perf_counter()

                        if compare_with_sympy:
                            # =========================================================
                            # BENCHMARK: New EvaluationContext System vs. Old Methods
                            # =========================================================
                            import time as bench_time

                            from plagih.evaluation_context import EvaluationContext, create_context

                            bench_results = {}

                            # 1. NumPy Eager (new context)
                            t0 = bench_time.perf_counter()
                            ctx_np = create_context("numpy_eager", use_lut=False)
                            result_np = ctx_np.evaluate(evotree, self.df_train)
                            bench_results["1_numpy_ctx"] = bench_time.perf_counter() - t0

                            # 2. NumPy Lambda (new context)
                            t0 = bench_time.perf_counter()
                            ctx_lambda = create_context("numpy_lambda", use_lut=False)
                            result_lambda_fn = ctx_lambda.evaluate(evotree)
                            result_lambda = result_lambda_fn(self.df_train)
                            bench_results["2_lambda_ctx"] = bench_time.perf_counter() - t0

                            # 3. SymPy (new context)
                            t0 = bench_time.perf_counter()
                            ctx_sympy = create_context("sympy", use_lut=False)
                            result_sympy = ctx_sympy.evaluate(evotree)
                            bench_results["3_sympy_ctx"] = bench_time.perf_counter() - t0

                            # 4. All together without LUT
                            t0 = bench_time.perf_counter()
                            ctx_all = EvaluationContext(modes=["numpy_eager", "numpy_lambda", "sympy"], use_lut=False)
                            results_all = ctx_all.evaluate(evotree, self.df_train)
                            bench_results["4_all_no_lut"] = bench_time.perf_counter() - t0

                            # 5. All together WITH LUT (second call should be faster)
                            t0 = bench_time.perf_counter()
                            ctx_lut = EvaluationContext(modes=["numpy_eager", "numpy_lambda", "sympy"], use_lut=True)
                            results_lut1 = ctx_lut.evaluate(evotree, self.df_train)  # First call (cache miss)
                            results_lut2 = ctx_lut.evaluate(evotree, self.df_train)  # Second call (cache hit!)
                            bench_results["5_all_with_lut"] = bench_time.perf_counter() - t0

                            # Compare with OLD methods timing
                            t0 = bench_time.perf_counter()
                            old_np = evotree.eval_predict_numpy_now(self.df_train)
                            bench_results["OLD_numpy"] = bench_time.perf_counter() - t0

                            t0 = bench_time.perf_counter()
                            old_lambda = evotree.eval_np_lambdas()(self.df_train)
                            bench_results["OLD_lambda"] = bench_time.perf_counter() - t0

                            t0 = bench_time.perf_counter()
                            old_sympy = evotree.get_sympy_expr()
                            bench_results["OLD_sympy"] = bench_time.perf_counter() - t0

                            # Print benchmark results
                            log("pp", "=== EvaluationContext Benchmark ===")
                            log(
                                "pp",
                                f"  1. NumPy (ctx):     {bench_results['1_numpy_ctx'] * 1000:6.2f}ms | OLD: {bench_results['OLD_numpy'] * 1000:6.2f}ms",
                            )
                            log(
                                "pp",
                                f"  2. Lambda (ctx):    {bench_results['2_lambda_ctx'] * 1000:6.2f}ms | OLD: {bench_results['OLD_lambda'] * 1000:6.2f}ms",
                            )
                            log(
                                "pp",
                                f"  3. SymPy (ctx):     {bench_results['3_sympy_ctx'] * 1000:6.2f}ms | OLD: {bench_results['OLD_sympy'] * 1000:6.2f}ms",
                            )
                            log("pp", f"  4. All (no LUT):    {bench_results['4_all_no_lut'] * 1000:6.2f}ms")
                            log(
                                "pp", f"  5. All (with LUT):  {bench_results['5_all_with_lut'] * 1000:6.2f}ms (2 evals)"
                            )
                            log(
                                "pp",
                                f"  LUT Stats: {ctx_lut.get_cache_size()} entries, hit-rate: {ctx_lut.get_cache_hit_rate('numpy_eager'):.0%}",
                            )

                            # Verify results match
                            np.testing.assert_array_almost_equal(
                                result_np, old_np, decimal=6, err_msg="NumPy context result doesn't match old method!"
                            )
                            np.testing.assert_array_almost_equal(
                                result_lambda,
                                old_lambda,
                                decimal=6,
                                err_msg="Lambda context result doesn't match old method!",
                            )
                            # =========================================================

                            """Numpy eager eval"""
                            # sfeh _lambda verion comparisson, functionality-wise and time-wise, eval sympy first
                            nplambda_results_raw = evotree.eval_np_lambdas()
                            nplambda_results_raw = nplambda_results_raw(self.df_train)

                            perf_t[2] = time.perf_counter()

                            """Sympy lambdify"""
                            sym_results_raw = eval_predict_sympyBatch(sy_expr, self.df_train, self.evolve.symbol_list)
                            sym_results = self.eval_autocast(sym_results_raw)
                            sym_fitness = self.eval_error_metric(sym_results, self.df_train[self.target_column])
                            sym_fitness = round(sym_fitness, _cfg.float_precision)

                            perf_t[3] = time.perf_counter()

                            log(
                                "pp",
                                f"NP: {perf_t[1] - perf_t[0]:4.4f}s, NE: {perf_t[2] - perf_t[1]:4.4f}s, SY: {perf_t[3] - perf_t[2]:4.4f}s. "
                                f"Fitness NP: {np_fitness}, SY: {sym_fitness} ({sy_expr}), eval tree id {tree_id}",
                            )

                            try:
                                sum(nplambda_results_raw - np_results_raw)
                            except Exception:
                                raise TreeError("SFEH THESE ARE [True] trees")

                            if sum(nplambda_results_raw - np_results_raw) > 0.001:
                                diffs = np.abs(nplambda_results_raw - np_results_raw)
                                mask = diffs > 0.001
                                if np.any(mask):
                                    indices = np.where(mask)[0]
                                    log("w", f"{len(indices)} differences found above tolerance 0.001: (NP VERSION)")
                                print(
                                    f"Different in (NP VERSION): {sum(nplambda_results_raw - np_results)} ({sy_expr})"
                                )

                            if np.sum(np.abs(nplambda_results_raw - np_results_raw)) > 0.001:
                                sym_results_raw_np = sym_results_raw.to_numpy()
                                diffs = np.abs(sym_results_raw_np - np_results_raw)
                                mask = diffs > 0.001
                                if np.any(mask):
                                    indices = np.where(mask)[0]
                                    log("w", f"{len(indices)} differences found above tolerance 0.001:")
                                # results_syraw_df = eval_predict_df_sympy_only(sy_expr, self.df_train)  #  takes forever
                                result_diffs = sum(sym_results_raw - np_results_raw)
                                log("w", f"Different results in evaluation: {result_diffs} sy-expr: ({sy_expr})")

                            if _cfg.lut_enabled:
                                self.lut_tree_infos[tree_id]["fitness-sympy"] = sym_fitness
                except (SympyError, TreeError, ValueError) as e:
                    log("wwww", f"Could not evaluate fitness for tree {sy_expr}: {e}")
                    if _cfg.lut_enabled:
                        self.lut_tree_infos[tree_id]["error"] = str(e)
                    raise

                fitness = np_fitness

                if _cfg.lut_enabled:
                    self.lut_symex_fitness[sy_expr] = fitness

            if _cfg.lut_enabled:
                self.lut_tree_infos[tree_id]["sy_expr"] = sy_expr
                self.lut_tree_infos[tree_id]["parsimony"] = parsimony
                self.lut_tree_infos[tree_id]["fitness"] = fitness

        candidate = Candidate(evotree, fitness=fitness, parsimony=parsimony, tag=tag)
        return candidate

    def _save_merged_population_tree(self):
        """Creates and saves a merged population tree visualization.

        Merges all trees in the current population into a single DAG
        and saves a PNG visualization using the unified tree_renderer.

        Output files:
        - population_merged/Population-merged-gen-XXX.png (visualization)
        """
        from plagih.population_merge import build_one_evaluation_tree
        from visualization.tree_renderer import render_merged_tree

        try:
            # Build merged graph from current population
            graph = build_one_evaluation_tree(self.pop_genepool)

            # Create output directory
            merged_dir = self.rootdir / "population_merged"
            merged_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with generation number
            base_filename = f"Population-merged-gen-{self.gen_id:03d}"

            # Render using the unified tree_renderer (produces only .png, no .dot)
            render_merged_tree(
                graph=graph,
                filename=base_filename,
                output_dir=merged_dir,
                orientation="BT",  # Bottom-up for merged trees
                display_mode="expression",
                title=f"Merged Population Tree - Generation {self.gen_id}",
                show_statistics=True,
            )
            log("ggg", f"Saved merged population tree: {base_filename}.png")

        except Exception as e:
            log("w", f"Could not create merged population tree: {e}")

    def evoloop_monitoring_plots(self):
        """Creates all monitoring visualizations for the GP run.

        Generates:
        - Performance plot (fitness/parsimony over generations)
        - Pareto front plot
        - Pareto front tree visualization
        - Parsimony histogram (for first 20 generations)
        """
        # Use monitor's built-in plotting or convert to DataFrame for compatibility
        self.monitor.plot_performance(self.rootdir / "monitoring.png")
        plot_paretofront(self.paretofront, self.rootdir, self.evolve.nodes_max)

        from plagih.visualization.tree_renderer import visualize_paretofront

        visualize_paretofront(self.paretofront, filename="paretofront_trees", output_dir=self.rootdir)

        if self.gen_id <= 20:
            gen_filename = f"monitoring_parsimony_histogram_{self.gen_id:03d}.png"
            # Use pop_size as max_population and nodes_max as max_parsimony for fixed scaling
            plot_parsimony_histogram(
                self.pop_genepool,
                self.rootdir / gen_filename,
                max_population=self.pop_max_size,
                max_parsimony=self.evolve.nodes_max,
            )

    def backup_save(self, opt_path_backup=None):
        """Saves a pickle backup of the current GP run state.

        Saves: generation ID, population, Pareto front, and monitoring data.

        Args:
            opt_path_backup: Optional custom path. Defaults to rootdir/backup/backup.pkl.
        """

        path_backup = opt_path_backup or self.rootdir / "backup/backup.pkl"

        # {} is the help_dict; include this, even if empty, to store/load successfully after future updates
        # Save monitor as DataFrame for backwards compatibility
        monitor_df = self.monitor.to_dataframe()
        run_backup_data = {}, self.gen_id, self.pop_genepool, self.paretofront, monitor_df
        path_backup = path_make_dir(path_backup)
        pickle_dump(path_backup, run_backup_data)

    def backup_load(self, opt_path_backup=None):
        """Loads a pickle backup of a previous GP run.

        Restores: generation ID, population, Pareto front, and monitoring data.
        Also creates a timestamped copy of the backup.

        Args:
            opt_path_backup: Optional custom path. Defaults to rootdir/backup/backup.pkl.

        Raises:
            FileNotFoundError: If backup file doesn't exist.
            Exception: If backup file is corrupted (EOFError).
        """

        path_backup = opt_path_backup or self.rootdir / "backup/backup.pkl"

        if Path.is_file(path_backup):
            log("g", f"Loading data from backup-file {path_backup}")
            try:
                with Path.open(path_backup, "rb") as file:
                    run_data = pickle.load(file)
            except EOFError as e:
                raise Exception(f"EOFError: \n{e}")

            help_dict, self.gen_id, self.pop_genepool, self.paretofront, loaded_monitor_df = run_data
            # Repair tree back-references stripped during pickling
            for candidate in self.pop_genepool:
                candidate.tree.repair_all()
            for candidate in self.paretofront:
                candidate.tree.repair_all()
            # Recreate monitor from loaded DataFrame (for backwards compatibility)
            # The monitor will be repopulated if evolution continues
            self.monitor = GPMonitor()
            self.backup_save(opt_path_backup=self.rootdir / f"backup/backup-{self.gen_id}.pkl")
            log("g", f"Successfully loaded backup file. Generation: {self.gen_id}")
        else:
            raise FileNotFoundError(f"No backup-file found at {path_backup}")

    def analyze_generation(self, pareto_updated: bool = False, _suppress_print: bool = False):
        """Analyzes and logs statistics for the current generation.

        Computes population metrics and stores them in monitor.
        Triggers scheduled IO operations (plots, backups) based on intervals.

        Args:
            pareto_updated: Whether the Pareto front was updated this generation.
            _suppress_print: If True, suppress the generation summary printpl.
                Used by run_generation which prints its own progress-style summary.
        """
        gen_time = time.perf_counter() - self.time_genstart

        # Record generation metrics using GPMonitor
        self.monitor.record_generation(
            gen_id=self.gen_id,
            population=self.pop_genepool,
            gen_time=gen_time,
            pareto_updated=pareto_updated,
            lut_size=len(self.lut_symex_fitness),
        )

        # Get latest metrics for logging (only when not suppressed by run_generation)
        if not _suppress_print:
            latest = self.monitor.latest
            log(
                "gg",
                f"generation {self.gen_id}/{self.gen_end} summary: "
                f"created {latest.get('pop_size', 0)}/{self.pop_max_size}"
                f" | unique={latest.get('pop_unique', 0)}"
                f" | lut={len(self.lut_symex_fitness)}"
                f" | time={gen_time:4.2f}s",
            )

        self._persist_generation_tree_timings(gen_id=self.gen_id)

        # Generate merged population tree visualization
        if self.enable_analysis and _cfg.merged_tree:
            self._save_merged_population_tree()

        if not _suppress_print:
            log("ggg", f"generation {self.gen_id}/{self.gen_end} summary done: {gen_time:4.2f}s")

        # def monitoring_scheduled_io(self, gen_id, plots_interval=10, backup_interval=10):
        """
        Every x generations, save a backup and/or save plots
        """
        if self.enable_analysis:
            if self.gen_id >= _cfg.plots_interval and self.gen_id % _cfg.plots_interval == 0:
                self.evoloop_monitoring_plots()

            if (self.gen_id >= _cfg.backup_interval and self.gen_id % _cfg.backup_interval == 0) or self.gen_id == 10:
                self.backup_save()

    def _persist_generation_tree_timings(self, gen_id: int) -> None:
        """Persist per-tree timing records and emit warnings only for real anomalies."""
        records = list(self._generation_tree_timings)
        self._latest_generation_tree_timings = records
        if not records:
            return

        perf_dir = self.rootdir / "performance"
        perf_dir.mkdir(parents=True, exist_ok=True)
        csv_path = perf_dir / f"tree_timings_gen_{gen_id:04d}.csv"
        pd.DataFrame(records).to_csv(csv_path, index=False)

        ok_records = [record for record in records if record.get("status") == "ok"]
        error_records = [record for record in records if record.get("status") == "error"]

        max_create_ms = max((record.get("create_ms_shared", 0.0) for record in ok_records), default=0.0)
        max_simplify_ms = max((record.get("simplify_ms", 0.0) for record in ok_records), default=0.0)
        max_evaluate_ms = max((record.get("evaluate_ms", 0.0) for record in ok_records), default=0.0)
        slowest_total = max(ok_records, key=lambda record: record.get("total_ms", 0.0), default=None)
        timing_stats = self._tree_timing_stats(ok_records)
        outliers = self._tree_timing_outliers(ok_records, stats=timing_stats)

        stage_counts = {"create": 0, "simplify": 0, "evaluate": 0, "other": 0}
        for record in error_records:
            stage = record.get("failed_stage")
            if stage in stage_counts:
                stage_counts[stage] += 1
            else:
                stage_counts["other"] += 1

        slowest_part = "n/a"
        if slowest_total is not None:
            slowest_part = (
                f"{slowest_total.get('total_ms', 0.0):.1f}ms"
                f" ({slowest_total.get('tag', '?')}#{slowest_total.get('tree_index', '?')})"
            )

        interesting_error_count = stage_counts["simplify"] + stage_counts["evaluate"] + stage_counts["other"]
        excessive_create_errors = stage_counts["create"] >= max(10, len(records) // 8)
        should_log_warning = bool(outliers) or interesting_error_count > 0 or excessive_create_errors

        if should_log_warning:
            log(
                "w",
                f"generation {gen_id}/{self.gen_end} tree timing warning: "
                f"records={len(records)} | outliers={len(outliers)} | slowest_total={slowest_part} | "
                f"median={timing_stats['median_total_ms']:.1f}ms | p95={timing_stats['p95_total_ms']:.1f}ms | "
                f"max(create/simplify/evaluate)={max_create_ms:.1f}/{max_simplify_ms:.1f}/{max_evaluate_ms:.1f}ms | "
                f"errors(create/simplify/evaluate/other)="
                f"{stage_counts['create']}/{stage_counts['simplify']}/{stage_counts['evaluate']}/{stage_counts['other']}",
            )
            self._log_tree_timing_outliers(gen_id=gen_id, records=outliers)
            log("f", f"Tree timing CSV saved to: {csv_path}")

    @staticmethod
    def _tree_timing_dominant_phase(record: Dict[str, Any]) -> str:
        """Return the dominant measured phase for one successful tree timing record."""
        phase_values = {
            "create": record.get("create_ms_shared", 0.0),
            "simplify": record.get("simplify_ms", 0.0),
            "evaluate": record.get("evaluate_ms", 0.0),
        }
        return max(phase_values, key=phase_values.get)

    @staticmethod
    def _tree_timing_expr_short(record: Dict[str, Any], limit: int = 90) -> str:
        """Return a compact single-line expression preview for logging."""
        expr = str(record.get("expr_short") or "")
        expr = expr.replace("\n", " ")
        if len(expr) <= limit:
            return expr
        return expr[: limit - 3] + "..."

    @staticmethod
    def _tree_timing_stats(records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Return robust summary statistics for successful tree timings."""
        totals = np.asarray([record.get("total_ms", 0.0) for record in records], dtype=np.float64)
        if totals.size == 0:
            return {"median_total_ms": 0.0, "p95_total_ms": 0.0, "max_total_ms": 0.0}
        return {
            "median_total_ms": float(np.median(totals)),
            "p95_total_ms": float(np.percentile(totals, 95)) if totals.size > 1 else float(totals[0]),
            "max_total_ms": float(np.max(totals)),
        }

    def _tree_timing_outliers(
        self,
        records: List[Dict[str, Any]],
        *,
        stats: Optional[Dict[str, float]] = None,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return only clearly anomalous slow trees instead of every generation's top-N list."""
        if not records:
            return []

        stats = stats or self._tree_timing_stats(records)
        median_total_ms = stats.get("median_total_ms", 0.0)
        p95_total_ms = stats.get("p95_total_ms", 0.0)
        outliers: List[Dict[str, Any]] = []

        for record in records:
            total_ms = record.get("total_ms", 0.0)
            dominant_phase = self._tree_timing_dominant_phase(record)
            dominant_phase_ms = {
                "create": record.get("create_ms_shared", 0.0),
                "simplify": record.get("simplify_ms", 0.0),
                "evaluate": record.get("evaluate_ms", 0.0),
            }.get(dominant_phase, 0.0)
            is_extreme_absolute = total_ms >= 5000.0
            is_large_absolute = total_ms >= 1000.0
            is_large_relative = total_ms >= max(3.0 * p95_total_ms, 8.0 * median_total_ms, 1000.0)
            is_phase_dominated = (
                total_ms >= 1000.0 and dominant_phase_ms >= 0.85 * total_ms and dominant_phase_ms >= 750.0
            )

            if is_extreme_absolute or (is_large_absolute and (is_large_relative or is_phase_dominated)):
                outliers.append(record)

        return sorted(outliers, key=lambda record: record.get("total_ms", 0.0), reverse=True)[:limit]

    def _log_tree_timing_outliers(self, gen_id: int, records: List[Dict[str, Any]]) -> None:
        """Log only genuinely slow outlier trees for long-run usability diagnostics."""
        if not records:
            return

        for rank, record in enumerate(records, start=1):
            dominant_phase = self._tree_timing_dominant_phase(record)
            expr_short = self._tree_timing_expr_short(record)
            log(
                "w",
                f"Tree timing outlier #{rank} in generation {gen_id}/{self.gen_end}: "
                f"total={record.get('total_ms', 0.0):7.1f}ms | "
                f"phase={dominant_phase:<8} | tag={record.get('tag', '?')} | "
                f"fit={record.get('fitness', float('nan')):.4f} | "
                f"parsim={record.get('parsimony', '?')} | expr={expr_short}",
            )

    def run_custom_exit_condition(self):
        """Checks for early termination conditions.

        Currently checks: No new Pareto entries in 100 generations.

        Returns:
            True if evolution should stop early, False otherwise.
        """
        if self.monitor.gens_since_last_pareto > 100:
            log("i", "Custom Condition made your program exit! (No new pareto entries in 100 generations)")
            return True
        else:
            return False


if __name__ == "__main__":
    """
    Tests have been moved to plagih/test/ directory.

    Run tests with:
        pytest plagih/test/ -v

    Or run a quick sanity check:
        python -c "from plagih.trees import *; print('Import successful')"
    """
    print("Tests have been moved to plagih/test/")
    print("Run: pytest plagih/test/ -v")
    print()
    print("Quick sanity check:")

    # Quick sanity check
    tree = Add(Symbol(sympy.Symbol("a")), Number(1.0))
    expr = tree.get_sympy_expr()
    print(f"  Tree: {tree}")
    print(f"  SymPy: {expr}")
    print(f"  Length: {len(tree)}")
    print()
    print("All basic imports and operations work correctly.")
