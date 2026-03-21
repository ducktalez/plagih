"""Smoke tests for the tree-creation benchmark harness."""

from pathlib import Path

from plagih.test.benchmarks._tree_creation_harness import (
    bench_active_generation,
    bench_initial_population,
    bench_raw_random_creation,
    run_tree_creation_benchmarks,
    save_benchmark_summary,
)


class TestTreeCreationBenchmarkHarness:
    """Structure checks for focused tree-creation performance diagnostics."""

    def test_raw_random_creation_summary_schema(self):
        summary = bench_raw_random_creation(iterations=10, depth_max_local=4, seed=7)

        assert summary["scenario"] == "raw_random_creation"
        assert summary["records"] == 10
        assert summary["ok_records"] == 10
        assert summary["error_records"] == 0
        assert summary["mean_total_ms"] >= 0.0
        assert summary["max_create_ms"] >= 0.0
        assert "create" in summary["dominant_phase_counts"]
        assert summary["top_slowest"]

    def test_initial_population_summary_contains_evaluation_phase_data(self):
        summary = bench_initial_population(pop_max_size=30, seed=11)

        assert summary["scenario"] == "initial_population"
        assert summary["records"] >= summary["ok_records"]
        assert summary["ok_records"] > 0
        assert summary["max_evaluate_ms"] >= 0.0
        assert set(summary["failed_stage_counts"]).issuperset({"create", "simplify", "evaluate", "other"})

    def test_active_generation_summary_has_nonempty_records(self):
        summary = bench_active_generation(pop_max_size=30, seed=13)

        assert summary["scenario"] == "active_generation"
        assert summary["records"] > 0
        assert summary["ok_records"] > 0
        assert summary["p95_total_ms"] >= summary["p50_total_ms"]

    def test_run_tree_creation_benchmarks_and_save_json(self, tmp_path: Path):
        suite = run_tree_creation_benchmarks(pop_max_size=25, seed=5)
        output_path = tmp_path / "tree_creation_summary.json"
        save_benchmark_summary(suite, output_path)

        assert set(suite) == {
            "raw_random_creation",
            "depth_goal_creation",
            "initial_population",
            "active_generation",
        }
        assert output_path.exists()
