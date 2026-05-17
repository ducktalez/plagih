"""Smoke test for the NN+GP pipeline (baseline-only, fast).

Runs ``benchmarks/nn_gp/run_mc.py`` end-to-end in the cheapest possible
mode (``--baseline-only --fast`` equivalent) and asserts that:

* ``experiment.json`` is written by the tracker
* ``PAPER_BLUEPRINT.md`` is generated
* the blueprint contains **no unresolved ``{{...}}`` placeholders**
  (catches regressions in the placeholder wiring; see IMPLEMENTATION_PLAN
  I10.2 / I10.3).

Marked as ``performance`` because it still trains several small NNs
(~30-60 s wall time). Run via::

    pytest plagih/test/test_nn_gp_pipeline.py --run-perf -q
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import pytest


@pytest.mark.performance
def test_nn_gp_baseline_smoke(tmp_path, monkeypatch):
    # Lazy imports - pulls in torch, only happens when the test actually runs
    pytest.importorskip("torch")
    from benchmarks.nn_gp import run_mc

    # Redirect RESULTS_BASE into the temporary directory so we do not
    # pollute .results/ on every CI run.
    monkeypatch.setattr(run_mc, "RESULTS_BASE", tmp_path / "nn_gp")

    blueprint_path = run_mc.run(
        n_iterations=1,
        gp_pop_size=10,
        gp_gen_end=2,
        baseline_only=True,  # skip GP entirely - fastest path
        show_figures=False,
        fast=True,
    )

    assert isinstance(blueprint_path, Path)
    assert blueprint_path.exists(), f"blueprint not written: {blueprint_path}"
    assert blueprint_path.name == "PAPER_BLUEPRINT.md"

    # experiment.json must exist alongside it
    exp_json = blueprint_path.parent / "experiment.json"
    assert exp_json.exists(), f"experiment.json missing: {exp_json}"

    text = blueprint_path.read_text(encoding="utf-8")

    # No unresolved placeholders. Matches anything of the form {{...}}.
    unresolved = re.findall(r"\{\{[^}]+\}\}", text)
    assert not unresolved, f"unresolved placeholders in blueprint: {unresolved[:5]} (total {len(unresolved)})"

    # Sanity: baseline section must be filled with a real number
    assert "Baseline" in text
    assert "Parameters" in text

    # Tidy up - tmp_path is auto-cleaned, but be explicit about the gp_runs
    # subtree (PyTorch holds onto some weights / file handles otherwise).
    gp_runs = blueprint_path.parent / "gp_runs"
    if gp_runs.exists():
        shutil.rmtree(gp_runs, ignore_errors=True)
