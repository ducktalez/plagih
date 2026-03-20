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
