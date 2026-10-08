#!/usr/bin/env python3
"""Measure a sparse matrix-free pseudospectral estimator on PME operators.

The experiment separates two modes.  Validation mode supplies the stored
exact spectrum and dense origin singular value, but replaces the old dense
resolvent grid with adaptive matrix-free path probes.  Runtime mode uses only
a short Arnoldi Ritz spectrum and matrix-free origin estimate and is therefore
provisional.  Both modes expose their cost relative to the measured GMRES
solve.  A frozen-coefficient operator supplies an inadequate comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
from pme_breakdown import _initial_state
from pme_dst_tau_blend_stage1_validation import Stage1Config, _project_physical_state
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import build_pme_linearization
from moljax.experimental.pme_dst_tau_blend import build_pme_tau_blend_linearization
from moljax.experimental.pseudospectral_criterion import (
    dense_sigma_min,
    materialize_dense_operator,
)
from moljax.experimental.pseudospectral_criterion_matrix_free import (
    assess_targeted_pseudospectral_connectivity,
    estimate_operator_norm,
    estimate_sigma_min_matrix_free,
    matrix_free_ritz_values,
)


class RuntimeEstimatorConfig(NamedTuple):
    """Configuration for the sparse pseudospectral estimator experiment."""

    nx: int = 512
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    epsilon: float = 1.0e-5
    halfwidth: float = 3.0
    analysis_dt: float = 2.0e-2
    reference_count: int = 3
    m_values: tuple[int, ...] = (2, 4, 8)
    sigma_arnoldi_steps: int = 40
    spectrum_arnoldi_steps: int = 12
    maximum_sigma_min_evaluations: int = 48
    bisection_iterations: int = 10
    power_steps: int = 12
    gmres_tolerance: float = 1.0e-8
    spectral_results_path: str = "benchmarks/results/pme_dst_tau_blend_spectral_analysis.json"
    work_precision_results_path: str = "benchmarks/results/pme_dst_tau_blend_stage1_validation.json"
    dense_criterion_results_path: str = (
        "benchmarks/results/pme_dst_tau_blend_pseudospectral_criterion.json"
    )
    output_path: str = "benchmarks/results/pme_dst_tau_blend_pseudospectral_runtime_estimator.json"


def _stored_spectral_records(
    config: RuntimeEstimatorConfig,
) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(Path(config.spectral_results_path).read_text(encoding="utf-8"))
    return {
        (int(record["m"]), record["method"]): record
        for record in payload["records"]
        if record["case"] == "initial_wide_front"
    }


def _stored_work_records(config: RuntimeEstimatorConfig) -> dict[tuple[int, str], dict[str, Any]]:
    payload = json.loads(Path(config.work_precision_results_path).read_text(encoding="utf-8"))
    records: dict[tuple[int, str], dict[str, Any]] = {}
    for record in payload["work_precision_records"]:
        if (
            record["nx"] == config.nx
            and record["analysis_dt"] == config.analysis_dt
            and record["requested_relative_residual"] == config.gmres_tolerance
        ):
            for method, measurement in record["methods"].items():
                records[(int(record["m"]), method)] = measurement
    return records


def _stored_dense_records(config: RuntimeEstimatorConfig) -> dict[int, dict[str, Any]]:
    payload = json.loads(Path(config.dense_criterion_results_path).read_text(encoding="utf-8"))
    return {int(record["m"]): record for record in payload["records"]}


def _state(grid: NodeCenteredDirichletGrid, m: int, config: RuntimeEstimatorConfig) -> jax.Array:
    state, _ = _project_physical_state(
        _initial_state(grid, m, config.t0, config.halfwidth),
        Stage1Config(evolved_nx=config.nx),
    )
    return state


def _operator(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    method: str,
    config: RuntimeEstimatorConfig,
) -> Any:
    if method == "tau_blend":
        return build_pme_tau_blend_linearization(
            state,
            grid,
            float(m),
            config.analysis_dt,
            config.epsilon,
            reference_count=config.reference_count,
        ).operator
    return build_pme_linearization(
        state,
        grid,
        float(m),
        config.analysis_dt,
        config.epsilon,
        method,
    ).operator


def _complex_values(pairs: list[list[float]]) -> np.ndarray:
    return np.asarray([complex(*pair) for pair in pairs], dtype=np.complex128)


def _dense_timed_sigma_min(matrix: np.ndarray, point: complex) -> tuple[float, float]:
    started = perf_counter()
    value = dense_sigma_min(matrix, point)
    return value, perf_counter() - started


def _sigma_validation(
    operator: Any,
    matrix: np.ndarray,
    eigenvalues: np.ndarray,
    operator_norm_scale: float,
    config: RuntimeEstimatorConfig,
) -> dict[str, Any]:
    real_span = max(float(np.ptp(eigenvalues.real)), 1.0)
    points = (0.0j, complex(float(np.median(eigenvalues.real)), 0.15 * real_span))
    records = []
    for offset, point in enumerate(points):
        matrix_free = estimate_sigma_min_matrix_free(
            operator,
            point,
            arnoldi_steps=config.sigma_arnoldi_steps,
            operator_norm_scale=operator_norm_scale,
            seed=20261030 + offset,
            refine_with_propack=True,
        )
        dense_value, dense_seconds = _dense_timed_sigma_min(matrix, point)
        matrix_free_payload = asdict(matrix_free)
        matrix_free_payload["point"] = [matrix_free.point.real, matrix_free.point.imag]
        records.append(
            {
                "point": [point.real, point.imag],
                "dense_sigma_min": dense_value,
                "matrix_free": matrix_free_payload,
                "absolute_error": abs(matrix_free.estimate - dense_value),
                "relative_error": abs(matrix_free.estimate - dense_value)
                / max(dense_value, np.finfo(float).tiny),
                "dense_svd_seconds": dense_seconds,
                "matrix_free_time_over_dense_svd": matrix_free.elapsed_seconds / dense_seconds,
            }
        )
    return {
        "points": records,
        "median_dense_svd_seconds": median(record["dense_svd_seconds"] for record in records),
        "median_matrix_free_seconds": median(
            record["matrix_free"]["elapsed_seconds"] for record in records
        ),
        "maximum_relative_error": max(record["relative_error"] for record in records),
    }


def _one_record(
    m: int,
    method: str,
    grid: NodeCenteredDirichletGrid,
    spectral: dict[str, Any],
    work: dict[str, Any],
    dense_prior: dict[str, Any] | None,
    config: RuntimeEstimatorConfig,
) -> dict[str, Any]:
    operator = _operator(_state(grid, m, config), grid, m, method, config)
    matrix = materialize_dense_operator(operator.matvec, operator.n)
    eigenvalues = _complex_values(spectral["eigenvalues"])
    norm_scale, norm_scale_applications, norm_scale_seconds = estimate_operator_norm(
        operator,
        power_steps=config.power_steps,
        seed=20261040 + m,
    )
    sigma_validation = _sigma_validation(
        operator,
        matrix,
        eigenvalues,
        norm_scale,
        config,
    )
    exact_paths = assess_targeted_pseudospectral_connectivity(
        operator,
        eigenvalues,
        spectrum_source="stored exact dense eigendecomposition",
        spectrum_complete=True,
        exact_spectrum_points=True,
        epsilon_zero_lower_bound=float(spectral["spectrum"]["minimum_singular_value"]),
        arnoldi_steps=config.sigma_arnoldi_steps,
        operator_norm_scale=norm_scale,
        anchor_count=eigenvalues.size,
        maximum_sigma_min_evaluations=config.maximum_sigma_min_evaluations,
        bisection_iterations=config.bisection_iterations,
        seed=20261050 + m,
    )
    ritz, ritz_steps, ritz_seconds = matrix_free_ritz_values(
        operator,
        arnoldi_steps=config.spectrum_arnoldi_steps,
        seed=20261060 + m,
    )
    runtime_paths = assess_targeted_pseudospectral_connectivity(
        operator,
        ritz,
        spectrum_source=f"{ritz_steps}-step matrix-free Arnoldi Ritz values",
        spectrum_complete=False,
        exact_spectrum_points=False,
        arnoldi_steps=config.sigma_arnoldi_steps,
        operator_norm_scale=norm_scale,
        anchor_count=ritz.size,
        maximum_sigma_min_evaluations=config.maximum_sigma_min_evaluations,
        bisection_iterations=config.bisection_iterations,
        seed=20261070 + m,
    )
    solve_seconds = float(work["median_seconds"])
    exact_total = norm_scale_seconds + exact_paths.elapsed_seconds
    runtime_total = norm_scale_seconds + ritz_seconds + runtime_paths.elapsed_seconds
    result: dict[str, Any] = {
        "m": m,
        "method": method,
        "operator_dimension": operator.n,
        "operator_norm_scale": {
            "matrix_free_estimate_with_safety_factor": norm_scale,
            "dense_operator_norm": float(spectral["spectrum"]["maximum_singular_value"]),
            "operator_applications": norm_scale_applications,
            "seconds": norm_scale_seconds,
        },
        "sigma_min_validation": sigma_validation,
        "exact_spectrum_sparse_paths": asdict(exact_paths),
        "fully_matrix_free_runtime_estimate": {
            **asdict(runtime_paths),
            "ritz_arnoldi_steps": ritz_steps,
            "ritz_operator_applications": ritz_steps,
            "ritz_seconds": ritz_seconds,
        },
        "actual_gmres": {
            "iterations": sorted(set(int(value) for value in work["iterations"])),
            "all_converged": bool(work["all_converged"]),
            "median_seconds": solve_seconds,
            "iqr_seconds": float(work["iqr_seconds"]),
        },
        "cost": {
            "exact_spectrum_sparse_path_total_seconds": exact_total,
            "exact_spectrum_sparse_path_time_over_gmres": exact_total / solve_seconds,
            "runtime_estimate_total_seconds": runtime_total,
            "runtime_estimate_time_over_gmres": runtime_total / solve_seconds,
        },
    }
    if dense_prior is not None:
        dense = dense_prior["pseudospectral_criterion"]
        result["dense_grid_reference"] = {
            "certified": bool(dense["certified"]),
            "epsilon_zero": float(dense["epsilon_zero"]),
            "eps_connect_upper": float(dense["eps_connect_upper"]),
            "sigma_min_evaluations": int(dense["sigma_min_evaluations"]),
            "sigma_min_seconds": float(dense["sigma_min_seconds"]),
            "predicted_iterations": dense["predicted_iterations"],
            "actual_iterations": dense_prior["actual_counted_gmres"]["iterations"],
        }
        result["cost"]["old_dense_time_over_gmres"] = float(
            dense_prior["gmres_timing_reference"]["certificate_sigma_min_time_over_gmres_median"]
        )
    return result


def run_experiment(config: RuntimeEstimatorConfig | None = None) -> dict[str, Any]:
    """Run sparse-path validation on tau operators and one inadequate baseline."""
    if config is None:
        config = RuntimeEstimatorConfig()
    spectral = _stored_spectral_records(config)
    work = _stored_work_records(config)
    dense = _stored_dense_records(config)
    cases = [(m, "tau_blend") for m in config.m_values] + [(8, "frozen_mean")]
    missing = [case for case in cases if case not in spectral or case not in work]
    if missing:
        raise ValueError(f"stored validation records are missing cases: {missing}")
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    started = perf_counter()
    records = [
        _one_record(
            m,
            method,
            grid,
            spectral[(m, method)],
            work[(m, method)],
            dense.get(m) if method == "tau_blend" else None,
            config,
        )
        for m, method in cases
    ]
    report = {
        "schema": "pme_dst_tau_blend_pseudospectral_runtime_estimator_v1",
        "provenance": provenance_revisions(),
        "scope": (
            "Experimental sparse-path pseudospectral connectivity estimate. Exact-spectrum "
            "mode is a sufficient qualitative certificate; the fully matrix-free reduced-"
            "spectrum mode is provisional. Sparse paths do not provide a closed-contour "
            "arc length, so no residual-rate certificate is claimed."
        ),
        "bound": {
            "rate_prefactor": "L(Gamma_epsilon)/(2*pi*epsilon)",
            "crouzeix_constant_used": False,
            "rate_available_in_sparse_estimator": False,
            "reason": (
                "the proven bound requires a closed contour and its arc length; targeted "
                "connectivity paths alone do not supply either"
            ),
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
    require_x64("PME DST/tau-blend sparse pseudospectral benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=RuntimeEstimatorConfig().output_path)
    args = parser.parse_args()
    report = run_experiment(RuntimeEstimatorConfig(output_path=args.output))
    for record in report["records"]:
        exact = record["exact_spectrum_sparse_paths"]
        runtime = record["fully_matrix_free_runtime_estimate"]
        print(
            f"m={record['m']} method={record['method']} "
            f"exact={exact['verdict']} ({exact['sigma_min_evaluations']} evals) "
            f"runtime={runtime['verdict']} ({runtime['sigma_min_evaluations']} evals) "
            f"cost/solve={record['cost']['runtime_estimate_time_over_gmres']:.2f}"
        )
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")


if __name__ == "__main__":
    main()
