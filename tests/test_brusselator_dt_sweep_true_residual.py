"""Consistency checks for the Brusselator stopping-convention follow-up."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.brusselator_dt_sweep_true_residual import render

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


def test_true_residual_report_is_current_and_decision_is_robust() -> None:
    """All committed cells retain count parity and the decision."""
    payload = json.loads(
        (RESULTS / "brusselator_dt_sweep_true_residual.json").read_text(encoding="utf-8")
    )
    assert len(payload["cells"]) == 200
    assert all(
        row["preconditioned_stop_count"] == row["committed_preconditioned_stop_count"]
        for row in payload["cells"].values()
    )
    assert payload["robustness_criterion_met"]
    assert payload["decision_true_stop"]["decision"] == "BUILD"
    expected = (RESULTS / "BRUSSELATOR_DT_SWEEP_TRUE_RESIDUAL.md").read_text(encoding="utf-8")
    assert render(payload) == expected
    assert render(payload) == render(payload)


def test_posthoc_result_has_all_states_and_correlations() -> None:
    """The labelled post-hoc result contains ten states and eight correlations."""
    payload = json.loads(
        (RESULTS / "brusselator_dt_sweep_posthoc.json").read_text(encoding="utf-8")
    )
    assert len(payload["states"]) == 10
    correlations = payload["spearman_at_dt_1"]
    assert (
        sum(
            len(conventions)
            for methods in correlations.values()
            for conventions in methods.values()
        )
        == 8
    )
