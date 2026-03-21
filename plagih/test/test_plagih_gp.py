"""Tests for the top-level plagih_gp entry point helpers."""

import pandas as pd
import pytest
import sympy

import plagih_gp


class _FakeTree:
    def get_sympy_expr(self):
        return sympy.Symbol("cartPos") + 1


class _FakeCandidate:
    def __init__(self, fitness=0.5, parsimony=3):
        self.fitness = fitness
        self.parsimony = parsimony

    def get_evotree(self):
        return _FakeTree()


class _FakeGP:
    def __init__(self):
        self.gen_id = 0
        self.gen_end = 1000
        self.pop_max_size = 5000
        self.time_start = 0.0
        self.pop_genepool = []
        self.paretofront = []
        self.run_generation_calls = 0
        self.backup_saved = False
        self.plots_saved = False

    def gen_create_initial(self):
        self.gen_id = 1
        self.pop_genepool = [object() for _ in range(5)]
        self.paretofront = [_FakeCandidate(fitness=0.8, parsimony=5)]

    def run_generation(self, strategies):
        self.run_generation_calls += 1
        self.gen_id += 1
        self.pop_genepool = [object() for _ in range(5)]
        self.paretofront = [_FakeCandidate(fitness=0.25, parsimony=2)]

        # Stop the smoke-test quickly after the first observed generation.
        self.gen_id = self.gen_end + 1

    def run_custom_exit_condition(self):
        return False

    def backup_save(self):
        self.backup_saved = True

    def evoloop_monitoring_plots(self):
        self.plots_saved = True


class TestLongrunHelpers:
    def test_demo_active_usability_test_smoke_uses_nonstandard_defaults(self, monkeypatch, tmp_path):
        fake_gp = _FakeGP()
        create_kwargs = {}
        logged_messages = []

        df = pd.DataFrame(
            {
                "cartVel": [0.1, 0.2, 0.3],
                "cartPos": [1.0, 1.1, 1.2],
                "action": [0.0, 1.0, 2.0],
            }
        )

        def fake_create(cls, **kwargs):
            create_kwargs.update(kwargs)
            fake_gp.gen_end = kwargs["gen_end"]
            fake_gp.pop_max_size = kwargs["pop_max_size"]
            return fake_gp

        monkeypatch.setattr(plagih_gp, "setup_logging", lambda **kwargs: None)
        monkeypatch.setattr(plagih_gp, "log_info", lambda message: logged_messages.append(("info", message)))
        monkeypatch.setattr(plagih_gp, "log", lambda level, message: logged_messages.append((level, message)))
        monkeypatch.setattr(plagih_gp.pd, "read_csv", lambda *args, **kwargs: df.copy())
        monkeypatch.setattr(
            plagih_gp, "train_test_split", lambda data, test_size, random_state: (data.copy(), data.copy())
        )
        monkeypatch.setattr(plagih_gp.ExplainableGP, "create", classmethod(fake_create))
        monkeypatch.setattr(plagih_gp.Path, "cwd", lambda: tmp_path)

        gp = plagih_gp.demo_active_usability_test(run_name="pytest-active-test", enable_analysis=False)

        assert gp is fake_gp
        assert fake_gp.run_generation_calls == 1
        assert fake_gp.backup_saved is True
        assert fake_gp.plots_saved is True
        assert create_kwargs["rootdir"] == tmp_path / ".testruns" / "pytest-active-test"
        assert create_kwargs["pop_max_size"] == 5000
        assert create_kwargs["gen_end"] == 1000
        assert create_kwargs["error_metric"] == "rmse"
        assert create_kwargs["enable_analysis"] is False
        assert create_kwargs["parallel"] == 0
        assert create_kwargs["depth_max"] == 7
        assert create_kwargs["nodes_max"] == 35
        assert any("observe generation 1/1000" in message for level, message in logged_messages if level == "gg")
        assert any("generation limit 1000 reached" in message for level, message in logged_messages if level == "g")
        assert any(
            message.startswith("Starting non-standard active usability test")
            for level, message in logged_messages
            if level == "info"
        )
        assert "simplicate" not in [strategy.name for strategy in plagih_gp._build_active_test_strategies()]

    def test_demo_active_usability_test_validates_positive_limits(self):
        with pytest.raises(ValueError):
            plagih_gp.demo_active_usability_test(pop_max_size=0)

        with pytest.raises(ValueError):
            plagih_gp.demo_active_usability_test(gen_end=0)

    def test_test_simple_uses_planned_generation_count_for_gen_end(self, monkeypatch, tmp_path):
        """Regression: `_test_simple()` must not log beyond `gen_end` due to mismatched fixed loop counts."""
        captured = {}

        class _FakeGPForSimple:
            def __init__(self, *args, **kwargs):
                captured["gen_end"] = kwargs["gen_end"]
                self.run_generation_calls = 0

            def gen_create_initial(self):
                return None

            def run_generation(self, strategies):
                self.run_generation_calls += 1
                captured["run_generation_calls"] = self.run_generation_calls

            def evoloop_monitoring_plots(self):
                captured["plots_called"] = True

        monkeypatch.setattr(plagih_gp, "setup_logging", lambda **kwargs: None)
        monkeypatch.setattr(plagih_gp, "log_info", lambda *args, **kwargs: None)
        monkeypatch.setattr(plagih_gp, "rootdir", tmp_path, raising=False)
        monkeypatch.setattr(plagih_gp, "operator_dict", {}, raising=False)
        monkeypatch.setattr(
            plagih_gp, "df_train", pd.DataFrame({"cartPos": [1.0], "cartVel": [0.0], "action": [1.0]}), raising=False
        )
        monkeypatch.setattr(plagih_gp, "eval_autocast", lambda x: x, raising=False)
        monkeypatch.setattr(plagih_gp, "eval_error_metric", lambda pred, true: 0.0, raising=False)
        monkeypatch.setattr(plagih_gp, "Evolution", lambda *args, **kwargs: object())
        monkeypatch.setattr(plagih_gp, "ExplainableGP", _FakeGPForSimple)

        plagih_gp._test_simple(dir_name="pytest-simple", chained_on=False)

        assert captured["run_generation_calls"] == 24
        assert captured["gen_end"] == 24
        assert captured["plots_called"] is True
