"""Tests for logging/progress helpers."""

import logging

import sympy

import plagih.trees._gp_engine as gp_engine_mod
from plagih.config import cfg
from plagih.logging_utils import (
    _flush_progress_line,
    log_info,
    print_generation_done,
    print_generation_progress,
    setup_logging,
)
from plagih.parallel import Strategy
from plagih.trees import Add, Number, Symbol


def _deterministic_strategy(evolve, pop_genepool, paretofront, allow_chain, **params):
    """Return a simple deterministic tree for progress tests."""
    return Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))


class TestGenerationProgress:
    """Tests for generation progress output."""

    def test_print_generation_progress_outputs_bar(self, monkeypatch, capsys):
        """Progress helper should print a visible in-place bar when enabled."""
        monkeypatch.setattr(cfg, "verbosity", "gg")

        print_generation_progress(
            gen_id=1,
            gen_end=3,
            created=4,
            total=10,
            label="mutation",
            fail=2,
            elapsed_s=1.25,
        )
        _flush_progress_line()

        captured = capsys.readouterr().out
        assert "generation 1/3 mutation" in captured
        assert "4/10" in captured
        assert "fail=2" in captured
        assert "#" in captured

    def test_print_generation_done_uses_canonical_done_fields(self, monkeypatch, capsys):
        """Done helper should report created candidates and pre-update Pareto count explicitly."""
        monkeypatch.setattr(cfg, "verbosity", "gg")

        print_generation_done(
            gen_id=2,
            gen_end=5,
            time_ms=123.4,
            created=10,
            pareto_pre=3,
            ok=9,
            fail=1,
            tracker_total_ms=123.4,
        )

        captured = capsys.readouterr().out
        assert "generation 2/5 done:" in captured
        assert "created=10" in captured
        assert "pareto_pre=3" in captured

    def test_print_generation_progress_throttles_same_bucket_updates(self, monkeypatch, capsys):
        """Repeated updates inside the same progress bucket should not spam extra console lines."""
        monkeypatch.setattr(cfg, "verbosity", "gg")

        print_generation_progress(gen_id=9, gen_end=10, created=1, total=100, label="throttled", elapsed_s=0.0)
        print_generation_progress(gen_id=9, gen_end=10, created=2, total=100, label="throttled", elapsed_s=0.1)
        _flush_progress_line()

        captured = capsys.readouterr().out
        assert captured.count("generation 9/10 throttled") == 1

    def test_log_info_flushes_open_progress_line(self, monkeypatch, capsys):
        """Normal console logs should flush an open progress line instead of sticking to it."""
        monkeypatch.setattr(cfg, "verbosity", "gg")
        setup_logging(console_level=logging.INFO, verbose=False)

        print_generation_progress(gen_id=3, gen_end=7, created=1, total=10, label="random_new", elapsed_s=0.0)
        log_info("separate log line")

        captured = capsys.readouterr()
        assert "separate log line" in captured.out
        assert captured.err == ""
        assert "0.0s\n" in captured.out

    def test_create_trees_reports_progress(self, gp_instance, monkeypatch):
        """Legacy create_trees path should emit progress updates while adding trees."""
        monkeypatch.setattr(cfg, "verbosity", "gg")
        calls = []

        def _spy_progress(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(gp_engine_mod, "print_generation_progress", _spy_progress)
        monkeypatch.setattr(gp_engine_mod, "_flush_progress_line", lambda: None)

        @gp_instance.create_trees(rate=0.2)
        def deterministic_tree():
            return Add(Symbol(sympy.Symbol("cartPos")), Number(1.0))

        assert len(gp_instance.pop_next) == 2
        assert calls
        assert calls[0]["created"] == 0
        assert calls[-1]["created"] == 2
        assert calls[-1]["total"] == 2
        assert calls[-1]["label"] == "deterministic_tree"

    def test_run_generation_reports_progress(self, gp_instance, monkeypatch):
        """Declarative run_generation path should emit progress updates, too."""
        gp_instance.gen_create_initial()
        gp_instance.register_strategy("deterministic_progress", _deterministic_strategy)

        monkeypatch.setattr(cfg, "verbosity", "gg")
        calls = []

        def _spy_progress(**kwargs):
            calls.append(kwargs)

        monkeypatch.setattr(gp_engine_mod, "print_generation_progress", _spy_progress)
        monkeypatch.setattr(gp_engine_mod, "print_generation_start", lambda *args, **kwargs: None)
        monkeypatch.setattr(gp_engine_mod, "print_generation_done", lambda *args, **kwargs: None)

        gp_instance.run_generation(
            [Strategy("deterministic_progress", rate=0.2)],
            parallel=False,
            seed=1,
        )

        assert calls
        assert calls[0]["created"] == 0
        assert calls[-1]["created"] == 2
        assert calls[-1]["total"] == 2
        assert any(call["label"] == "deterministic_progress" for call in calls[1:])
