#!/usr/bin/env python3
"""Test a dense pseudospectral criterion on the PME DST/tau blend.

This experiment compares exact eigenvalues, the existing field-of-values disk
diagnostic, and a grid-resolved pseudospectral certificate on the same
512-node left-preconditioned Newton operators.  It records the full resolvent
sampling cost alongside counted GMRES so certification cost is not hidden.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pme_breakdown import _initial_state
from pme_dst_tau_blend_stage1_validation import (
    Stage1Config,
    _project_physical_state,
)
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_dst_tau_blend import (
    build_pme_tau_blend_linearization,
    measure_pme_tau_blend_gmres_iterations,
)
from moljax.experimental.pseudospectral_criterion import (
    assess_dense_pseudospectral_criterion,
    materialize_dense_operator,
)


class CriterionConfig(NamedTuple):
    """Configuration for the dense pseudospectral tau-blend experiment."""

    nx: int = 512
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    epsilon: float = 1.0e-5
    halfwidth: float = 3.0
    analysis_dt: float = 2.0e-2
    reference_count: int = 3
    m_values: tuple[int, ...] = (2, 4, 8)
    grid_points_per_axis: int = 61
    domain_padding_fraction: float = 0.75
    polynomial_degrees: tuple[int, ...] = (4, 8, 12, 16, 20)
    epsilon_samples: int = 6
    gmres_tolerance: float = 1.0e-8
    gmres_max_iters: int = 400
    spectral_results_path: str = "benchmarks/results/pme_dst_tau_blend_spectral_analysis.json"
    work_precision_results_path: str = "benchmarks/results/pme_dst_tau_blend_stage1_validation.json"
    output_path: str = "benchmarks/results/pme_dst_tau_blend_pseudospectral_criterion.json"


def _stage_config(config: CriterionConfig) -> Stage1Config:
    return Stage1Config(
        x_min=config.x_min,
        x_max=config.x_max,
        t0=config.t0,
        epsilon=config.epsilon,
        evolved_nx=config.nx,
    )


def _stored_tau_diagnostics(config: CriterionConfig) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(config.spectral_results_path).read_text(encoding="utf-8"))
    stored = payload["config"]
    required = {
        "nx": config.nx,
        "halfwidth": config.halfwidth,
        "analysis_dt": config.analysis_dt,
        "reference_count": config.reference_count,
    }
    if any(stored[name] != value for name, value in required.items()):
        raise ValueError("stored spectral configuration does not match this experiment")
    return {
        int(record["m"]): record
        for record in payload["records"]
        if record["case"] == "initial_wide_front" and record["method"] == "tau_blend"
    }


def _stored_gmres_timings(config: CriterionConfig) -> dict[int, dict[str, Any]]:
    """Load matching warmed solve costs from the prior validation study."""
    payload = json.loads(Path(config.work_precision_results_path).read_text(encoding="utf-8"))
    records = {}
    for record in payload["work_precision_records"]:
        if (
            record["nx"] == config.nx
            and record["analysis_dt"] == config.analysis_dt
            and record["requested_relative_residual"] == config.gmres_tolerance
        ):
            records[int(record["m"])] = record["methods"]["tau_blend"]
    return records


def _one_record(
    m: int,
    grid: NodeCenteredDirichletGrid,
    stored: dict[str, Any],
    gmres_timing: dict[str, Any],
    config: CriterionConfig,
) -> dict[str, Any]:
    state, _ = _project_physical_state(
        _initial_state(grid, m, config.t0, config.halfwidth),
        _stage_config(config),
    )
    linearization = build_pme_tau_blend_linearization(
        state,
        grid,
        float(m),
        config.analysis_dt,
        config.epsilon,
        reference_count=config.reference_count,
    )
    matrix_started = perf_counter()
    matrix = materialize_dense_operator(
        linearization.operator.matvec,
        linearization.operator.n,
    )
    matrix_seconds = perf_counter() - matrix_started
    criterion = assess_dense_pseudospectral_criterion(
        matrix,
        grid_points_per_axis=config.grid_points_per_axis,
        domain_padding_fraction=config.domain_padding_fraction,
        polynomial_degrees=config.polynomial_degrees,
        epsilon_samples=config.epsilon_samples,
        target_tolerance=config.gmres_tolerance,
    )
    gmres = measure_pme_tau_blend_gmres_iterations(
        state,
        grid,
        float(m),
        config.analysis_dt,
        config.epsilon,
        reference_count=config.reference_count,
        tol=config.gmres_tolerance,
        max_iters=config.gmres_max_iters,
    )
    if gmres["iterations"] not in gmres_timing["iterations"]:
        raise RuntimeError("counted GMRES does not reproduce the stored timing study")
    eigenvalues = np.linalg.eigvals(matrix)
    field_of_values = stored["field_of_values"]
    return {
        "m": m,
        "operator_dimension": matrix.shape[0],
        "matrix_materialization_seconds": matrix_seconds,
        "eigenvalue_clustering": {
            "minimum_eigenvalue_modulus": float(np.min(np.abs(eigenvalues))),
            "maximum_eigenvalue_modulus": float(np.max(np.abs(eigenvalues))),
            "near_zero_eigenvalue_count": int(np.count_nonzero(np.abs(eigenvalues) < 0.1)),
            "near_one_fraction": float(np.mean(np.abs(eigenvalues - 1.0) < 0.5)),
        },
        "disk_criterion": {
            "disk_rate": float(field_of_values["disk_rate"]),
            "origin_enclosed": bool(field_of_values["origin_enclosed"]),
            "verdict": field_of_values["disk_only_verdict"],
        },
        "pseudospectral_criterion": asdict(criterion),
        "actual_counted_gmres": gmres,
        "gmres_timing_reference": {
            "source": config.work_precision_results_path,
            "median_seconds": float(gmres_timing["median_seconds"]),
            "iqr_seconds": float(gmres_timing["iqr_seconds"]),
            "repetitions": int(gmres_timing["repetitions"]),
            "certificate_sigma_min_time_over_gmres_median": (
                criterion.sigma_min_seconds / float(gmres_timing["median_seconds"])
            ),
        },
    }


def run_experiment(config: CriterionConfig | None = None) -> dict[str, Any]:
    """Run the three-criterion comparison on selected wide-front states."""
    if config is None:
        config = CriterionConfig()
    if config.reference_count != 3:
        raise ValueError("this experiment is fixed to the selected l=3 blend")
    stored = _stored_tau_diagnostics(config)
    timings = _stored_gmres_timings(config)
    missing = (set(config.m_values) - set(stored)) | (set(config.m_values) - set(timings))
    if missing:
        raise ValueError(f"stored spectral records are missing m values: {sorted(missing)}")
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    started = perf_counter()
    records = [_one_record(m, grid, stored[m], timings[m], config) for m in config.m_values]
    report = {
        "schema": "pme_dst_tau_blend_pseudospectral_criterion_v1",
        "provenance": provenance_revisions(),
        "scope": (
            "Experimental dense 512-node comparison of eigenvalue clustering, the "
            "field-of-values disk diagnostic, and a grid-resolved pseudospectral "
            "certificate. The resolvent grid is finite and Lipschitz-corrected; this "
            "is not interval arithmetic or a scalable production diagnostic."
        ),
        "bound": {
            "rate_prefactor": "L(Gamma_epsilon)/(2*pi*epsilon)",
            "formula": "L(Gamma_epsilon)/(2*pi*epsilon) * rho(polynomial)**k",
            "crouzeix_constant_used": False,
        },
        "config": config._asdict(),
        "records": records,
        "runtime_seconds": perf_counter() - started,
        "complete": True,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    require_x64("PME DST/tau-blend dense pseudospectral benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=CriterionConfig().output_path)
    parser.add_argument("--m", type=int, nargs="+", default=list(CriterionConfig().m_values))
    parser.add_argument("--grid-points", type=int, default=CriterionConfig().grid_points_per_axis)
    parser.add_argument("--epsilon-samples", type=int, default=CriterionConfig().epsilon_samples)
    args = parser.parse_args()
    report = run_experiment(
        CriterionConfig(
            m_values=tuple(args.m),
            grid_points_per_axis=args.grid_points,
            epsilon_samples=args.epsilon_samples,
            output_path=args.output,
        )
    )
    for record in report["records"]:
        criterion = record["pseudospectral_criterion"]
        actual = record["actual_counted_gmres"]["iterations"]
        print(
            f"m={record['m']} certified={criterion['certified']} "
            f"eps_connect_upper={criterion['eps_connect_upper']:.6g} "
            f"epsilon_zero={criterion['epsilon_zero']:.6g} "
            f"predicted={criterion['predicted_iterations']} actual={actual}"
        )
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")


if __name__ == "__main__":
    main()
