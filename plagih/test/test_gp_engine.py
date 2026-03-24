"""Regression tests for ExplainableGP engine wiring."""

from pathlib import Path

import numpy as np
import pandas as pd
import sympy

import plagih.parallel as parallel_mod
import plagih.trees._gp_engine as gp_engine_mod
from plagih.parallel import Strategy, build_task_list
from plagih.trees import Add, ExplainableGP, Number, Symbol


class TestTargetColumnRegression:
    """Ensure ExplainableGP uses the configured target column everywhere."""

    @staticmethod
    def _make_gp(tmp_path: Path, cartpole_evolution) -> ExplainableGP:
        df_train = pd.DataFrame(
            {
                "cartPos": [1.0, 2.0, 3.0, 4.0],
                "cartVel": [0.0, 0.0, 0.0, 0.0],
                "target": [2.0, 3.0, 4.0, 5.0],
                # Deliberately wrong target values: using this column would produce non-zero loss.
                "action": [100.0, 100.0, 100.0, 100.0],
            }
        )
        return ExplainableGP(
            evolve=cartpole_evolution,
            df_train=df_train,
            rootdir=tmp_path,
            pop_max_size=5,
            gen_end=2,
            target_column="target",
            eval_autocast=lambda x: np.asarray(x, dtype=np.float64),
            eval_error_metric=lambda pred, true: float(np.mean(np.abs(np.asarray(pred) - np.asarray(true)))),
            verbose=False,
            parallel=False,
            enable_analysis=False,
        )

    def test_tree_to_candidate_uses_target_column_without_sympy_compare(self, tmp_path, cartpole_evolution):
        """Base evaluation path must use self.target_column instead of hardcoded 'action'."""
        gp = self._make_gp(tmp_path, cartpole_evolution)
        tree = Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))

        candidate = gp.tree_to_candidate(tree, compare_with_sympy=False)

        assert candidate.fitness == 0.0

    def test_tree_to_candidate_uses_target_column_with_sympy_compare(self, tmp_path, cartpole_evolution):
        """SymPy comparison path must also use self.target_column."""
        gp = self._make_gp(tmp_path, cartpole_evolution)
        tree = Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))

        candidate = gp.tree_to_candidate(tree, compare_with_sympy=True)

        assert candidate.fitness == 0.0


class TestGenerationSummaryLogging:
    """Ensure canonical generation summary/start logs stay consistent."""

    def test_distribute_weighted_counts_preserves_exact_total(self, gp_instance):
        """Weighted init/backfill counts should sum exactly to the requested target."""
        counts = gp_instance._distribute_weighted_counts(9, [0.5, 0.5])

        assert sum(counts) == 9
        assert sorted(counts) == [4, 5]

    def test_build_task_list_honours_exact_strategy_count(self):
        """Declarative task building must support exact counts, not only rate-derived counts."""
        tasks = build_task_list([Strategy("random_new", count=7, depths=[3, 4])], pop_max_size=50, seed=123)

        assert len(tasks) == 7
        assert all(task.strategy_name == "random_new" for task in tasks)

    def test_build_task_list_prefers_init_label_for_task_tag(self):
        """Generation-0 sampler labels should survive task expansion for progress/timing attribution."""
        tasks = build_task_list(
            [Strategy("random_new", count=2, init_label="init_rand_probe")], pop_max_size=50, seed=123
        )

        assert len(tasks) == 2
        assert {task.tag for task in tasks} == {"init_rand_probe"}

    def test_build_initial_strategy_plan_respects_remaining_slots(self, gp_instance):
        """Generation-0 strategy plans should reserve already filled slots and still sum exactly."""
        gp_instance.pop_next = [object()]

        strategies = gp_instance._build_initial_strategy_plan()

        counts = [strategy.count for strategy in strategies]
        assert counts == [4, 5]
        assert sum(counts) == gp_instance.pop_max_size - 1
        assert all(strategy.name == "random_new" for strategy in strategies)

    def test_gen_create_initial_with_origin_tree_still_fills_population(self, gp_instance):
        """Generation 0 should reserve the origin tree and still fill the remaining slots automatically."""
        origin_tree = Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))

        gp_instance.gen_create_initial(origin_tree=origin_tree)

        assert len(gp_instance.pop_genepool) == gp_instance.pop_max_size
        assert gp_instance.pop_genepool[0].tree.get_sympy_expr() == origin_tree.get_sympy_expr()

    def test_analyze_generation_logs_created_against_pop_max_size(self, gp_instance, monkeypatch):
        """Regression: summary should use canonical wording and pop_max_size."""
        messages = []

        def _capture_log(msg_type, message):
            messages.append((msg_type, message))

        monkeypatch.setattr(gp_engine_mod, "log", _capture_log)

        gp_instance.gen_create_initial()

        summary_messages = [
            message
            for msg_type, message in messages
            if msg_type == "gg" and message.startswith(f"generation 0/{gp_instance.gen_end} summary:")
        ]
        assert summary_messages
        assert any(
            f"created {gp_instance.pop_max_size}/{gp_instance.pop_max_size}" in message for message in summary_messages
        )
        assert all(f"/{gp_instance.gen_end} (" not in message for message in summary_messages)

    def test_gen_create_initial_logs_canonical_start_message(self, gp_instance, monkeypatch):
        """Initial population creation should use the canonical generation start wording."""
        messages = []

        def _capture_log(msg_type, message):
            messages.append((msg_type, message))

        monkeypatch.setattr(gp_engine_mod, "log", _capture_log)

        gp_instance.gen_create_initial()

        assert any(
            msg_type == "gg" and message == f"generation 0/{gp_instance.gen_end} start: create initial population"
            for msg_type, message in messages
        )

    def test_gen_create_initial_uses_shared_task_runner(self, gp_instance, monkeypatch):
        """Generation 0 should now execute through the declarative sequential task runner."""
        calls = []
        original = parallel_mod.run_generation_sequential

        def _spy_run_generation_sequential(**kwargs):
            calls.append(kwargs)
            return original(**kwargs)

        monkeypatch.setattr(parallel_mod, "run_generation_sequential", _spy_run_generation_sequential)

        gp_instance.gen_create_initial(parallel=False, seed=7)

        assert calls
        assert sum(len(call["tasks"]) for call in calls) >= gp_instance.pop_max_size
        assert all(task.strategy_name == "random_new" for call in calls for task in call["tasks"])

    def test_tree_to_candidate_reuses_partial_exact_tree_lut_without_sympy(self, gp_instance, monkeypatch):
        """Exact-tree cache entries from parallel backfill should be usable even without a stored SymPy object."""
        base_tree = Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))
        first = gp_instance.tree_to_candidate(base_tree)
        tree_id = first.tree.get_lut_id()

        gp_instance.lut_tree_infos = {tree_id: {"parsimony": first.parsimony, "fitness": first.fitness}}
        gp_instance.lut_symex_fitness = {}

        cached_tree = Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))
        monkeypatch.setattr(
            cached_tree,
            "get_sympy_expr",
            lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("SymPy should not be recomputed")),
        )

        cached = gp_instance.tree_to_candidate(cached_tree)

        assert cached.fitness == first.fitness
        assert cached.parsimony == first.parsimony

    def test_gen_create_initial_persists_tree_timing_csv(self, gp_instance):
        """Generation analysis should persist per-tree timing records for later inspection."""
        gp_instance.gen_create_initial()

        timing_path = gp_instance.rootdir / "performance" / "tree_timings_gen_0000.csv"
        assert timing_path.exists()

        df = pd.read_csv(timing_path)
        assert not df.empty
        assert {"tag", "status", "create_ms_shared", "total_ms"}.issubset(df.columns)
        assert len(gp_instance._latest_generation_tree_timings) == len(df)

    def test_persist_generation_tree_timings_logs_only_real_outliers(self, gp_instance, monkeypatch):
        """Only genuinely anomalous slow trees should be logged for diagnosis."""
        messages = []

        def _capture_log(msg_type, message):
            messages.append((msg_type, message))

        monkeypatch.setattr(gp_engine_mod, "log", _capture_log)
        gp_instance._generation_tree_timings = [
            {
                "tag": "mutation",
                "status": "ok",
                "create_ms_shared": 5.0,
                "simplify_ms": 7.0,
                "evaluate_ms": 22.0,
                "total_ms": 34.0,
                "fitness": 0.5,
                "parsimony": 9,
                "expr_short": "Add(cartPos, cartVel)",
            },
            {
                "tag": "random_new",
                "status": "ok",
                "create_ms_shared": 3.0,
                "simplify_ms": 50.0,
                "evaluate_ms": 2350.0,
                "total_ms": 2403.0,
                "fitness": 0.4,
                "parsimony": 7,
                "expr_short": "Mul(cartPos, cartVel)",
            },
        ]

        gp_instance._persist_generation_tree_timings(gen_id=0)

        warning_messages = [message for msg_type, message in messages if msg_type == "w"]
        assert any("tree timing warning" in message for message in warning_messages)
        assert any("Tree timing outlier #1" in message and "phase=evaluate" in message for message in warning_messages)
        assert any("expr=Mul(cartPos, cartVel)" in message for message in warning_messages)

    def test_persist_generation_tree_timings_skips_normal_cases(self, gp_instance, monkeypatch):
        """Normal generations should persist CSVs silently without extra per-tree spam."""
        messages = []

        def _capture_log(msg_type, message):
            messages.append((msg_type, message))

        monkeypatch.setattr(gp_engine_mod, "log", _capture_log)
        gp_instance._generation_tree_timings = [
            {
                "tag": "mutation",
                "status": "ok",
                "create_ms_shared": 5.0,
                "simplify_ms": 7.0,
                "evaluate_ms": 22.0,
                "total_ms": 34.0,
                "fitness": 0.5,
                "parsimony": 9,
                "expr_short": "Add(cartPos, cartVel)",
            },
            {
                "tag": "random_new",
                "status": "ok",
                "create_ms_shared": 3.0,
                "simplify_ms": 1.0,
                "evaluate_ms": 40.0,
                "total_ms": 44.0,
                "fitness": 0.4,
                "parsimony": 7,
                "expr_short": "Mul(cartPos, cartVel)",
            },
        ]

        gp_instance._persist_generation_tree_timings(gen_id=0)

        assert not [message for msg_type, message in messages if msg_type in {"w", "f", "pp"}]

    def test_persist_generation_tree_timings_skips_single_noncreate_error(self, gp_instance, monkeypatch):
        """A single isolated simplify/evaluate error should not already trigger a warning in long-run mode."""
        messages = []

        def _capture_log(msg_type, message):
            messages.append((msg_type, message))

        monkeypatch.setattr(gp_engine_mod, "log", _capture_log)
        gp_instance._generation_tree_timings = [
            {
                "tag": "mutation",
                "status": "ok",
                "create_ms_shared": 5.0,
                "simplify_ms": 7.0,
                "evaluate_ms": 22.0,
                "total_ms": 34.0,
                "fitness": 0.5,
                "parsimony": 9,
                "expr_short": "Add(cartPos, cartVel)",
            },
            {
                "tag": "mutation",
                "status": "error",
                "failed_stage": "evaluate",
                "total_ms": 49.0,
                "error_type": "TreeError",
                "error_message": "isolated evaluation failure",
            },
        ]

        gp_instance._persist_generation_tree_timings(gen_id=0)

        assert not [message for msg_type, message in messages if msg_type in {"w", "f", "pp"}]
