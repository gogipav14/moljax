"""Smoke and trend checks for the nonlinear Diffrax work--precision harness."""

from __future__ import annotations

import importlib.util

import pytest

if importlib.util.find_spec("diffrax") is None:
    pytest.skip("diffrax benchmark extra is not installed", allow_module_level=True)

from benchmarks.work_precision_nonlinear_diffusion import (
    WorkPrecisionConfig,
    run_work_precision,
)


@pytest.mark.slow
def test_work_precision_smoke_returns_finite_shared_reference_data(tmp_path) -> None:
    """Both solvers run on both nonlinear RHSs and return crossover structures."""
    report = run_work_precision(
        WorkPrecisionConfig(
            nx=24,
            pme_be_dt_values=(0.1, 0.05),
            porous_fisher_be_dt_values=(0.02, 0.01),
            diffrax_rtol_values=(1.0e-2, 1.0e-5),
            timing_runs=1,
            output_path=str(tmp_path / "work_precision_nonlinear_diffusion.json"),
        )
    )

    assert set(report["problems"]) == {"pme_m2", "porous_fisher_r1"}
    for problem in report["problems"].values():
        assert len(problem["be_jfnk_frozen_bulk"]) == 2
        assert len(problem["diffrax_tsit5_pid"]) == 2
        assert len(problem["matched_accuracy_crossovers"]) == 4
        for method in ("be_jfnk_frozen_bulk", "diffrax_tsit5_pid"):
            assert all(record["error_inf"] >= 0.0 for record in problem[method])
            assert all(record["runtime"]["median_seconds"] >= 0.0 for record in problem[method])


@pytest.mark.slow
def test_work_precision_time_refinement_has_nonincreasing_errors(tmp_path) -> None:
    """Smaller BE steps and tighter PID tolerances do not worsen the PME error."""
    report = run_work_precision(
        WorkPrecisionConfig(
            nx=64,
            pme_be_dt_values=(0.1, 0.05, 0.025),
            porous_fisher_be_dt_values=(0.02, 0.01, 0.005),
            diffrax_rtol_values=(1.0e-1, 1.0e-4, 1.0e-6),
            timing_runs=1,
            output_path=str(tmp_path / "work_precision_nonlinear_diffusion_trends.json"),
        )
    )
    pme = report["problems"]["pme_m2"]
    be_errors = [record["error_inf"] for record in pme["be_jfnk_frozen_bulk"]]
    adaptive_errors = [record["error_inf"] for record in pme["diffrax_tsit5_pid"]]

    assert be_errors == sorted(be_errors, reverse=True)
    assert adaptive_errors == sorted(adaptive_errors, reverse=True)
