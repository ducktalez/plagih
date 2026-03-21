"""Regression tests for ExplainableGP engine wiring."""

from pathlib import Path

import numpy as np
import pandas as pd
import sympy

import plagih.trees._gp_engine as gp_engine_mod
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
