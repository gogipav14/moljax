#!/usr/bin/env python3
"""Compare the PME tau blend with active-support single-reference oracles.

This measurement-only study reuses the cleaned wide-front and evolved-state
protocol from ``pme_dst_tau_blend_stage1_validation``.  Every non-identity
single-reference method applies the same DST-I Helmholtz inverse; only its
scalar ``d0`` differs.  The optimized reference is a per-case log-grid oracle
chosen solely by counted GMRES iterations at the requested tolerance.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pme_breakdown import _initial_state
from pme_dst_tau_blend_stage1_validation import (
    Stage1Config,
    _advance_one_step,
    _project_physical_state,
)
from tau_blend_reproducibility import (
    load_or_generate_states,
    provenance_revisions,
    source_state_artifact,
)

from moljax._precision import require_x64
from moljax.conditioning import numerical_range
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import d0_variant, measure_gmres_iterations
from moljax.experimental.pme_dst_tau_blend import (
    build_pme_single_reference_linearization,
    build_pme_tau_blend_linearization,
    measure_pme_single_reference_gmres_iterations,
    measure_pme_tau_blend_gmres_iterations,
    pme_active_single_reference_values,
    porous_medium_diffusivity,
)

METHODS = (
    "identity",
    "frozen_mean",
    "frozen_bulk",
    "floor",
    "const",
    "geometric_mean",
    "harmonic_mean",
    "optimized_d0",
    "tau_blend",
)
SPECTRAL_METHODS = METHODS


class BaselineConfig(NamedTuple):
    """Configuration for the full single-reference comparison."""

    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    halfwidth: float = 3.0
    epsilon: float = 1.0e-5
    analysis_dt: float = 2.0e-2
    reference_count: int = 3
    max_krylov_iters: int = 400
    active_absolute_floor: float = 1.0e-14
    active_relative_floor: float = 1.0e-6
    work_nx_values: tuple[int, ...] = (512, 1024)
    work_m_values: tuple[int, ...] = (2, 4, 8)
    work_tolerances: tuple[float, ...] = (1.0e-2, 1.0e-5, 1.0e-8)
    oracle_scan_points: int = 51
    timing_warmups: int = 2
    timing_repetitions_512: int = 7
    timing_repetitions_1024: int = 3
    contrast_nx: int = 512
    contrast_m_values: tuple[int, ...] = (2, 8)
    contrast_tolerance: float = 1.0e-8
    evolved_state_dt: float = 1.0e-2
    evolved_steps: int = 8
    evolved_sample_steps: tuple[int, ...] = (0, 2, 5, 8)
    evolved_max_newton_iters: int = 12
    positivity_absolute_floor: float = 1.0e-14
    positivity_relative_floor: float = 1.0e-8
    spectral_nx: int = 512
    fov_angles: int = 32
    fov_max_iters: int = 120
    fov_residual_tolerance: float = 1.0e-3
    fov_restarts: int = 2
    source_state_cache_dir: str = "/tmp/moljax-tau-blend-source-states-v4"
    output_path: str = "benchmarks/results/pme_dst_tau_blend_single_reference_baselines.json"


def _json_config(config: BaselineConfig) -> dict[str, Any]:
    """Return a JSON-normalized configuration."""
    return json.loads(json.dumps(config._asdict()))


def _stage_config(config: BaselineConfig) -> Stage1Config:
    """Return the matching positivity-projected trajectory configuration."""
    return Stage1Config(
        x_min=config.x_min,
        x_max=config.x_max,
        t0=config.t0,
        epsilon=config.epsilon,
        analysis_dt=config.analysis_dt,
        max_krylov_iters=config.max_krylov_iters,
        evolved_nx=config.contrast_nx,
        evolved_m_values=config.contrast_m_values,
        evolved_state_dt=config.evolved_state_dt,
        evolved_steps=config.evolved_steps,
        evolved_sample_steps=config.evolved_sample_steps,
        evolved_max_newton_iters=config.evolved_max_newton_iters,
        positivity_absolute_floor=config.positivity_absolute_floor,
        positivity_relative_floor=config.positivity_relative_floor,
        active_absolute_floor=config.active_absolute_floor,
        active_relative_floor=config.active_relative_floor,
        source_state_cache_dir=config.source_state_cache_dir,
    )


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return median, first quartile, and third quartile."""
    ordered = sorted(values)
    center = float(median(ordered))
    if len(ordered) == 1:
        return center, center, center
    return (
        center,
        float(median(ordered[: len(ordered) // 2])),
        float(median(ordered[(len(ordered) + 1) // 2 :])),
    )


def _active_profile(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    config: BaselineConfig,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return active-support statistics and values used by all new references."""
    summary = dict(
        pme_active_single_reference_values(
            state,
            float(m),
            grid,
            config.epsilon,
            active_relative=config.active_relative_floor,
            active_absolute=config.active_absolute_floor,
        )
    )
    diffusivity = np.asarray(
        jax.block_until_ready(
            jnp.maximum(
                porous_medium_diffusivity(state, float(m), config.epsilon),
                0.0,
            )
        ),
        dtype=np.float64,
    )
    active_values = diffusivity[diffusivity > float(summary["active_threshold"])]
    quantile_levels = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    summary["active_d95_over_d05"] = float(
        np.quantile(active_values, 0.95) / np.quantile(active_values, 0.05)
    )
    summary["active_quantiles"] = {
        str(level): float(np.quantile(active_values, level)) for level in quantile_levels
    }
    summary["literal_minimum"] = float(np.min(diffusivity))
    summary["literal_maximum"] = float(np.max(diffusivity))
    return summary, active_values


def _named_d0(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    method: str,
    active: dict[str, Any],
    config: BaselineConfig,
    optimized_d0: float | None,
) -> float | None:
    """Return one method's scalar reference, or ``None`` for the tau blend."""
    if method == "tau_blend":
        return None
    if method == "identity":
        return 0.0
    if method in {"frozen_mean", "frozen_bulk", "floor", "const"}:
        return d0_variant(
            state,
            grid,
            float(m),
            config.epsilon,
            method,
        )
    if method == "geometric_mean":
        return float(active["geometric_mean"])
    if method == "harmonic_mean":
        return float(active["harmonic_mean"])
    if method == "optimized_d0":
        if optimized_d0 is None:
            raise ValueError("optimized_d0 requires a completed oracle scan")
        return optimized_d0
    raise ValueError(f"unknown method {method!r}")


def _measure_method(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    method: str,
    active: dict[str, Any],
    config: BaselineConfig,
    optimized_d0: float | None,
) -> dict[str, Any]:
    """Measure one counted-GMRES solve for the full comparison set."""
    if method == "tau_blend":
        return dict(
            measure_pme_tau_blend_gmres_iterations(
                state,
                grid,
                float(m),
                config.analysis_dt,
                config.epsilon,
                reference_count=config.reference_count,
                tol=tolerance,
                max_iters=config.max_krylov_iters,
                active_relative=config.active_relative_floor,
                active_absolute=config.active_absolute_floor,
            )
        )
    d0 = _named_d0(state, grid, m, method, active, config, optimized_d0)
    if method == "identity":
        return measure_gmres_iterations(
            state,
            grid,
            float(m),
            config.analysis_dt,
            config.epsilon,
            "identity",
            tol=tolerance,
            max_iters=config.max_krylov_iters,
        )
    if d0 is None:
        raise AssertionError("single-reference method has no d0")
    return dict(
        measure_pme_single_reference_gmres_iterations(
            state,
            grid,
            float(m),
            config.analysis_dt,
            config.epsilon,
            d0=d0,
            tol=tolerance,
            max_iters=config.max_krylov_iters,
        )
    )


def _oracle_scan(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    active: dict[str, Any],
    active_values: np.ndarray,
    config: BaselineConfig,
) -> dict[str, Any]:
    """Scan active D on a log grid and choose the minimum counted iteration count."""
    candidates = np.geomspace(
        float(active["active_minimum"]),
        float(active["active_maximum"]),
        config.oracle_scan_points,
    )
    rows: list[dict[str, Any]] = []
    for d0 in candidates:
        measured = measure_pme_single_reference_gmres_iterations(
            state,
            grid,
            float(m),
            config.analysis_dt,
            config.epsilon,
            d0=float(d0),
            tol=tolerance,
            max_iters=config.max_krylov_iters,
        )
        rows.append(dict(measured))
    converged = [row for row in rows if bool(row["converged"])]
    if converged:
        best_iterations = min(int(row["iterations"]) for row in converged)
        tied = [row for row in converged if int(row["iterations"]) == best_iterations]
        selected = tied[len(tied) // 2]
    else:
        best_iterations = min(int(row["iterations"]) for row in rows)
        tied = [row for row in rows if int(row["iterations"]) == best_iterations]
        selected = min(tied, key=lambda row: float(row["final_relative_residual"]))
    selected_d0 = float(selected["d0"])
    quantile_rank = float(np.mean(active_values <= selected_d0))
    return {
        "candidate_count": len(rows),
        "candidate_range": [float(candidates[0]), float(candidates[-1])],
        "selected_d0": selected_d0,
        "selected_converged": bool(selected["converged"]),
        "selected_iterations": int(selected["iterations"]),
        "selected_final_relative_residual": float(selected["final_relative_residual"]),
        "minimum_iteration_tie_count": len(tied),
        "minimum_iteration_tie_d0_range": [
            min(float(row["d0"]) for row in tied),
            max(float(row["d0"]) for row in tied),
        ],
        "active_empirical_quantile_rank": quantile_rank,
        "ratio_to_active_minimum": selected_d0 / float(active["active_minimum"]),
        "ratio_to_active_maximum": selected_d0 / float(active["active_maximum"]),
        "ratio_to_geometric_mean": selected_d0 / float(active["geometric_mean"]),
        "ratio_to_harmonic_mean": selected_d0 / float(active["harmonic_mean"]),
        "candidates": rows,
    }


def _time_method(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    method: str,
    active: dict[str, Any],
    config: BaselineConfig,
    optimized_d0: float,
) -> dict[str, Any]:
    """Time synchronized fresh linearization, preconditioner, and counted GMRES."""
    repetitions = (
        config.timing_repetitions_1024 if grid.nx >= 1024 else config.timing_repetitions_512
    )
    for _ in range(config.timing_warmups):
        warmup = _measure_method(
            state,
            grid,
            m,
            tolerance,
            method,
            active,
            config,
            optimized_d0,
        )
        jax.block_until_ready(jnp.asarray(warmup["final_relative_residual"]))
    elapsed: list[float] = []
    measurements: list[dict[str, Any]] = []
    for _ in range(repetitions):
        started = perf_counter()
        measured = _measure_method(
            state,
            grid,
            m,
            tolerance,
            method,
            active,
            config,
            optimized_d0,
        )
        jax.block_until_ready(jnp.asarray(measured["final_relative_residual"]))
        elapsed.append(perf_counter() - started)
        measurements.append(measured)
    center, lower, upper = _quartiles(elapsed)
    return {
        "d0": _named_d0(state, grid, m, method, active, config, optimized_d0),
        "repetitions": repetitions,
        "warmups_excluded": config.timing_warmups,
        "median_seconds": center,
        "iqr_seconds": upper - lower,
        "q1_seconds": lower,
        "q3_seconds": upper,
        "iterations": sorted({int(row["iterations"]) for row in measurements}),
        "all_converged": all(bool(row["converged"]) for row in measurements),
        "max_final_relative_residual": max(
            float(row["final_relative_residual"]) for row in measurements
        ),
        "includes": "fresh linearization/preconditioner construction plus counted GMRES",
    }


def _dense_operator_matrix(operator: Any) -> np.ndarray:
    """Materialize one matrix-free operator at the 512-node spectral size."""
    basis = jnp.eye(operator.n, dtype=jnp.float64)
    actions = jax.jit(jax.vmap(operator.matvec))(basis)
    return np.asarray(jax.block_until_ready(actions.T), dtype=np.complex128)


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    """Return complex values in a stable JSON representation."""
    return [[float(value.real), float(value.imag)] for value in values]


def _spectral_summary(matrix: np.ndarray) -> dict[str, Any]:
    """Return the requested exact eigenvalue and 2-norm diagnostics."""
    eigenvalues = np.linalg.eigvals(matrix)
    magnitudes = np.abs(eigenvalues)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    hermitian_part = 0.5 * (matrix + matrix.conj().T)
    numerical_abscissa = float(np.linalg.eigvalsh(hermitian_part)[-1])
    spectral_abscissa = float(np.max(eigenvalues.real))
    return {
        "condition_number_2": float(singular_values[0] / singular_values[-1]),
        "minimum_singular_value": float(singular_values[-1]),
        "maximum_singular_value": float(singular_values[0]),
        "near_zero_eigenvalue_count": int(np.count_nonzero(magnitudes < 0.1)),
        "minimum_eigenvalue_modulus": float(np.min(magnitudes)),
        "maximum_eigenvalue_modulus": float(np.max(magnitudes)),
        "sorted_eigenvalue_magnitudes": [float(value) for value in np.sort(magnitudes)],
        "eigenvalues": _complex_pairs(eigenvalues),
        "spectral_abscissa": spectral_abscissa,
        "numerical_abscissa": numerical_abscissa,
        "non_normality_gap": numerical_abscissa - spectral_abscissa,
    }


def _fov_summary(operator: Any, config: BaselineConfig) -> dict[str, Any]:
    """Return a corroborated matrix-free FOV summary for one spectral record."""
    result = numerical_range(
        operator.matvec,
        operator.matvec_adjoint,
        operator.n,
        n_angles=config.fov_angles,
        max_iters=config.fov_max_iters,
        residual_tolerance=config.fov_residual_tolerance,
        n_restarts=config.fov_restarts,
    )
    return {
        "boundary": _complex_pairs(np.asarray(result.boundary, dtype=np.complex128)),
        "center": [float(result.center.real), float(result.center.imag)],
        "radius": float(result.radius),
        "disk_rate": float(result.disk_rate),
        "origin_enclosed": bool(result.origin_enclosed),
        "supports_consistent": bool(result.supports_consistent),
        "corroboration_attempted": bool(result.corroboration_attempted),
        "supports_converged": bool(result.supports_converged),
        "supports_corroborated": bool(result.supports_corroborated),
        "max_support_residual": float(result.max_support_residual),
        "n_angles": config.fov_angles,
        "max_iters": config.fov_max_iters,
        "residual_tolerance": config.fov_residual_tolerance,
        "n_restarts": config.fov_restarts,
    }


def _linearization(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    method: str,
    active: dict[str, Any],
    optimized_d0: float,
    config: BaselineConfig,
) -> Any:
    """Build one operator through the same paths used by the solve benchmark."""
    if method == "tau_blend":
        return build_pme_tau_blend_linearization(
            state,
            grid,
            float(m),
            config.analysis_dt,
            config.epsilon,
            reference_count=config.reference_count,
            active_relative=config.active_relative_floor,
            active_absolute=config.active_absolute_floor,
        )
    d0 = _named_d0(state, grid, m, method, active, config, optimized_d0)
    if d0 is None:
        raise AssertionError("single-reference method has no d0")
    return build_pme_single_reference_linearization(
        state,
        grid,
        float(m),
        config.analysis_dt,
        config.epsilon,
        d0=d0,
    )


def _atomic_write(path: Path, report: dict[str, Any]) -> None:
    """Atomically persist the resumable report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _new_report(config: BaselineConfig) -> dict[str, Any]:
    """Create an empty measurement report."""
    return {
        "schema": "pme_dst_tau_blend_single_reference_baselines_v1",
        "scope": (
            "Measurement-only one-dimensional comparison of an l=3 DST/tau blend "
            "against active-support and per-case optimized single DST-I references."
        ),
        "timing_protocol": (
            "Two synchronized warmups excluded; median and IQR include fresh "
            "linearization/preconditioner construction plus counted GMRES."
        ),
        "oracle_protocol": (
            "Per (state, N, m, tolerance), scan 51 logarithmically spaced d0 values "
            "from active D_min to D_max and select the middle member of the "
            "minimum-count tie."
        ),
        "provenance": provenance_revisions(),
        "config": _json_config(config),
        "work_precision_records": [],
        "contrast_records": [],
        "spectral_records": [],
        "complete": False,
    }


def _load_report(config: BaselineConfig, resume: bool) -> dict[str, Any]:
    """Load a compatible checkpoint or start a fresh report."""
    output = Path(config.output_path)
    if not resume or not output.exists():
        return _new_report(config)
    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("config") != _json_config(config):
        raise ValueError("existing checkpoint uses a different configuration")
    return report


def _work_record(
    report: dict[str, Any], nx: int, m: int, tolerance: float
) -> dict[str, Any] | None:
    """Find one work-precision record in a checkpoint."""
    return next(
        (
            row
            for row in report["work_precision_records"]
            if int(row["nx"]) == nx
            and int(row["m"]) == m
            and float(row["requested_relative_residual"]) == tolerance
        ),
        None,
    )


def _wide_front_state(
    nx: int,
    m: int,
    config: BaselineConfig,
) -> tuple[
    NodeCenteredDirichletGrid,
    jax.Array,
    dict[str, Any],
    dict[str, Any],
    bool,
]:
    """Load or generate one fingerprinted projected wide-front source state."""
    grid = NodeCenteredDirichletGrid.uniform(nx, config.x_min, config.x_max)
    fingerprint = {
        "schema": "pme_tau_blend_generation_fingerprint_v4",
        "label": f"projected-wide-front-m{m}-n{nx}",
        "study": "pme_tau_blend_projected_wide_front",
        "m": m,
        "grid": {
            "nx": nx,
            "x_min": config.x_min,
            "x_max": config.x_max,
            "centering": "node",
            "boundary": "homogeneous_dirichlet",
        },
        "initial_state": {
            "kind": "barenblatt_wide_front",
            "time": config.t0,
            "halfwidth": config.halfwidth,
        },
        "epsilon": config.epsilon,
        "positivity_projection": {
            "absolute_floor": config.positivity_absolute_floor,
            "relative_floor": config.positivity_relative_floor,
        },
        "model_version": provenance_revisions()["base_revision"],
    }

    def generate() -> tuple[list[jax.Array], list[dict[str, Any]]]:
        state, projection = _project_physical_state(
            _initial_state(grid, m, config.t0, config.halfwidth),
            _stage_config(config),
        )
        return [state], [{"step": 0, "projection": projection}]

    states, identities, _, cache_reused = load_or_generate_states(
        config.source_state_cache_dir,
        fingerprint["label"],
        fingerprint,
        generate,
    )
    artifact = source_state_artifact(
        config.source_state_cache_dir,
        fingerprint["label"],
        fingerprint,
        0,
        identities[0],
    )
    return grid, states[0], fingerprint, artifact, cache_reused


def _run_work_precision(
    report: dict[str, Any], config: BaselineConfig, checkpoint: Callable[[], None]
) -> None:
    """Measure every wide-front method and target tolerance."""
    for nx in config.work_nx_values:
        for m in config.work_m_values:
            grid, state, fingerprint, artifact, cache_reused = _wide_front_state(nx, m, config)
            active, active_values = _active_profile(state, grid, m, config)
            for tolerance in config.work_tolerances:
                record = _work_record(report, nx, m, tolerance)
                if record is None:
                    record = {
                        "nx": nx,
                        "m": m,
                        "state": "initial_wide_front",
                        "analysis_dt": config.analysis_dt,
                        "requested_relative_residual": tolerance,
                        "coefficient": active,
                        "source_state_artifact": artifact,
                        "source_state_cache_reused": cache_reused,
                        "record_config": {
                            "state": "initial_wide_front",
                            "nx": nx,
                            "m": m,
                            "domain": [config.x_min, config.x_max],
                            "initial_time": config.t0,
                            "halfwidth": config.halfwidth,
                            "analysis_dt": config.analysis_dt,
                            "epsilon": config.epsilon,
                            "reference_count": config.reference_count,
                            "generation_fingerprint": fingerprint,
                        },
                        "methods": {},
                    }
                    report["work_precision_records"].append(record)
                if "optimized_d0_scan" not in record:
                    record["optimized_d0_scan"] = _oracle_scan(
                        state,
                        grid,
                        m,
                        tolerance,
                        active,
                        active_values,
                        config,
                    )
                    checkpoint()
                    print(
                        f"WORK_ORACLE nx={nx} m={m} tol={tolerance:.1e} "
                        f"d0={record['optimized_d0_scan']['selected_d0']:.9e} "
                        f"iters={record['optimized_d0_scan']['selected_iterations']}",
                        flush=True,
                    )
                optimized_d0 = float(record["optimized_d0_scan"]["selected_d0"])
                for method in METHODS:
                    if method in record["methods"]:
                        continue
                    record["methods"][method] = _time_method(
                        state,
                        grid,
                        m,
                        tolerance,
                        method,
                        active,
                        config,
                        optimized_d0,
                    )
                    checkpoint()
                    values = record["methods"][method]
                    print(
                        f"WORK_METHOD nx={nx} m={m} tol={tolerance:.1e} "
                        f"method={method} iters={values['iterations']} "
                        f"median={values['median_seconds']:.6f} "
                        f"converged={values['all_converged']}",
                        flush=True,
                    )


def _contrast_record(report: dict[str, Any], m: int, step: int) -> dict[str, Any] | None:
    """Find one evolved contrast record in a checkpoint."""
    return next(
        (
            row
            for row in report["contrast_records"]
            if int(row["m"]) == m and int(row["step"]) == step
        ),
        None,
    )


def _run_contrast(
    report: dict[str, Any], config: BaselineConfig, checkpoint: Callable[[], None]
) -> None:
    """Measure the full method set across the established evolved-state contrast range."""
    grid = NodeCenteredDirichletGrid.uniform(
        config.contrast_nx,
        config.x_min,
        config.x_max,
    )
    stage_config = _stage_config(config)
    for m in config.contrast_m_values:
        fingerprint = {
            "schema": "pme_tau_blend_generation_fingerprint_v4",
            "label": f"projected-wide-front-m{m}-n{config.contrast_nx}",
            "study": "pme_tau_blend_projected_evolved_trajectory",
            "m": m,
            "grid": {
                "nx": config.contrast_nx,
                "x_min": config.x_min,
                "x_max": config.x_max,
                "centering": "node",
                "boundary": "homogeneous_dirichlet",
            },
            "initial_state": {
                "kind": "barenblatt_wide_front",
                "time": config.t0,
                "halfwidth": config.halfwidth,
            },
            "epsilon": config.epsilon,
            "state_dt": config.evolved_state_dt,
            "steps": config.evolved_steps,
            "solver": {
                "preconditioner": "frozen_bulk",
                "max_newton_iters": config.evolved_max_newton_iters,
                "max_krylov_iters": config.max_krylov_iters,
                "newton_tol": 1.0e-8,
                "krylov_tol": 1.0e-8,
                "max_backtrack": 8,
            },
            "positivity_projection": {
                "absolute_floor": config.positivity_absolute_floor,
                "relative_floor": config.positivity_relative_floor,
            },
            "model_version": provenance_revisions()["base_revision"],
        }

        def generate() -> tuple[list[jax.Array], list[dict[str, Any]]]:
            initial_state, projection = _project_physical_state(
                _initial_state(grid, m, config.t0, config.halfwidth),
                stage_config,
            )
            generated = [initial_state]
            metadata: list[dict[str, Any]] = [
                {"step": 0, "advance_to_state": None, "projection": projection}
            ]
            current = initial_state
            for step_index in range(1, config.evolved_steps + 1):
                current, advance = _advance_one_step(current, grid, m, stage_config)
                generated.append(current)
                metadata.append({"step": step_index, "advance_to_state": advance})
            return generated, metadata

        states, identities, _, cache_reused = load_or_generate_states(
            config.source_state_cache_dir,
            fingerprint["label"],
            fingerprint,
            generate,
        )
        for step, state in enumerate(states):
            if step in config.evolved_sample_steps:
                active, active_values = _active_profile(state, grid, m, config)
                record = _contrast_record(report, m, step)
                if record is None:
                    record = {
                        "nx": config.contrast_nx,
                        "m": m,
                        "step": step,
                        "time": config.t0 + step * config.evolved_state_dt,
                        "requested_relative_residual": config.contrast_tolerance,
                        "coefficient": active,
                        "source_state_artifact": source_state_artifact(
                            config.source_state_cache_dir,
                            fingerprint["label"],
                            fingerprint,
                            step,
                            identities[step],
                        ),
                        "record_config": {
                            "m": m,
                            "step": step,
                            "grid": fingerprint["grid"],
                            "analysis_dt": config.analysis_dt,
                            "epsilon": config.epsilon,
                            "reference_count": config.reference_count,
                            "generation_fingerprint": fingerprint,
                        },
                        "source_state_cache_reused": cache_reused,
                        "methods": {},
                    }
                    report["contrast_records"].append(record)
                if "optimized_d0_scan" not in record:
                    record["optimized_d0_scan"] = _oracle_scan(
                        state,
                        grid,
                        m,
                        config.contrast_tolerance,
                        active,
                        active_values,
                        config,
                    )
                    checkpoint()
                    print(
                        f"CONTRAST_ORACLE m={m} step={step} "
                        f"d0={record['optimized_d0_scan']['selected_d0']:.9e} "
                        f"iters={record['optimized_d0_scan']['selected_iterations']}",
                        flush=True,
                    )
                optimized_d0 = float(record["optimized_d0_scan"]["selected_d0"])
                for method in METHODS:
                    if method in record["methods"]:
                        continue
                    record["methods"][method] = _measure_method(
                        state,
                        grid,
                        m,
                        config.contrast_tolerance,
                        method,
                        active,
                        config,
                        optimized_d0,
                    )
                    checkpoint()
                    values = record["methods"][method]
                    print(
                        f"CONTRAST_METHOD m={m} step={step} method={method} "
                        f"iters={values['iterations']} converged={values['converged']}",
                        flush=True,
                    )


def _spectral_record(report: dict[str, Any], m: int, method: str) -> dict[str, Any] | None:
    """Find one dense spectral record in a checkpoint."""
    return next(
        (
            row
            for row in report["spectral_records"]
            if int(row["m"]) == m and row["method"] == method
        ),
        None,
    )


def _run_spectra(
    report: dict[str, Any], config: BaselineConfig, checkpoint: Callable[[], None]
) -> None:
    """Materialize exact N=512 spectra for the decisive single references and blend."""
    for m in config.work_m_values:
        grid, state, fingerprint, artifact, cache_reused = _wide_front_state(
            config.spectral_nx,
            m,
            config,
        )
        active, _ = _active_profile(state, grid, m, config)
        work = _work_record(report, grid.nx, m, 1.0e-8)
        if work is None:
            raise RuntimeError("tight-tolerance work record is required before spectra")
        optimized_d0 = float(work["optimized_d0_scan"]["selected_d0"])
        for method in SPECTRAL_METHODS:
            if _spectral_record(report, m, method) is not None:
                continue
            linearization = _linearization(
                state,
                grid,
                m,
                method,
                active,
                optimized_d0,
                config,
            )
            matrix = _dense_operator_matrix(linearization.operator)
            report["spectral_records"].append(
                {
                    "nx": grid.nx,
                    "m": m,
                    "state": "initial_wide_front",
                    "method": method,
                    "coefficient": active,
                    "source_state_artifact": artifact,
                    "source_state_cache_reused": cache_reused,
                    "record_config": {
                        "state": "initial_wide_front",
                        "nx": grid.nx,
                        "m": m,
                        "analysis_dt": config.analysis_dt,
                        "epsilon": config.epsilon,
                        "reference_count": config.reference_count,
                        "method": method,
                        "generation_fingerprint": fingerprint,
                    },
                    "d0": _named_d0(
                        state,
                        grid,
                        m,
                        method,
                        active,
                        config,
                        optimized_d0,
                    ),
                    "maximum_imaginary_matrix_entry_abs": float(np.max(np.abs(matrix.imag))),
                    "spectrum": _spectral_summary(matrix),
                    "field_of_values": _fov_summary(linearization.operator, config),
                }
            )
            checkpoint()
            spectrum = report["spectral_records"][-1]["spectrum"]
            print(
                f"SPECTRAL m={m} method={method} "
                f"kappa2={spectrum['condition_number_2']:.9e} "
                f"near_zero={spectrum['near_zero_eigenvalue_count']}",
                flush=True,
            )


def _summaries(report: dict[str, Any]) -> dict[str, Any]:
    """Derive compact comparisons without replacing the underlying measurements."""
    work: list[dict[str, Any]] = []
    for record in report["work_precision_records"]:
        if any(method not in record["methods"] for method in METHODS):
            continue
        singles = {
            name: values
            for name, values in record["methods"].items()
            if name not in {"identity", "tau_blend"}
        }
        converged = {name: values for name, values in singles.items() if values["all_converged"]}
        tau = record["methods"]["tau_blend"]
        if not converged:
            work.append(
                {
                    "nx": record["nx"],
                    "m": record["m"],
                    "requested_relative_residual": record["requested_relative_residual"],
                    "best_single_reference": None,
                    "best_single_reached_target": False,
                    "best_single_iterations": None,
                    "best_single_median_seconds": None,
                    "tau_iterations": tau["iterations"][0],
                    "tau_median_seconds": tau["median_seconds"],
                    "tau_iteration_ratio_over_best_single": None,
                    "tau_time_ratio_over_best_single": None,
                }
            )
            continue
        best_name = min(
            converged,
            key=lambda name: (
                converged[name]["iterations"][0],
                converged[name]["median_seconds"],
            ),
        )
        best = record["methods"][best_name]
        work.append(
            {
                "nx": record["nx"],
                "m": record["m"],
                "requested_relative_residual": record["requested_relative_residual"],
                "best_single_reference": best_name,
                "best_single_reached_target": True,
                "best_single_iterations": best["iterations"][0],
                "best_single_median_seconds": best["median_seconds"],
                "tau_iterations": tau["iterations"][0],
                "tau_median_seconds": tau["median_seconds"],
                "tau_iteration_ratio_over_best_single": tau["iterations"][0]
                / best["iterations"][0],
                "tau_time_ratio_over_best_single": tau["median_seconds"] / best["median_seconds"],
            }
        )
    contrast: list[dict[str, Any]] = []
    for record in report["contrast_records"]:
        if any(method not in record["methods"] for method in METHODS):
            continue
        singles = {
            name: values
            for name, values in record["methods"].items()
            if name not in {"identity", "tau_blend"} and values["converged"]
        }
        best_name = min(singles, key=lambda name: singles[name]["iterations"]) if singles else None
        contrast.append(
            {
                "m": record["m"],
                "step": record["step"],
                "active_d95_over_d05": record["coefficient"]["active_d95_over_d05"],
                "degeneracy_fraction": record["coefficient"]["degeneracy_fraction"],
                "best_single_reference": best_name,
                "best_single_reached_target": best_name is not None,
                "best_single_iterations": (
                    singles[best_name]["iterations"] if best_name is not None else None
                ),
                "tau_iterations": record["methods"]["tau_blend"]["iterations"],
            }
        )
    return {"work_precision": work, "contrast": contrast}


def run(
    config: BaselineConfig | None = None,
    *,
    resume: bool = False,
    redo_spectra: bool = False,
) -> dict[str, Any]:
    """Run or resume the full single-reference comparison."""
    if config is None:
        config = BaselineConfig()
    if config.oracle_scan_points < 40:
        raise ValueError("oracle_scan_points must preserve the requested 40-60 point scan")
    report = _load_report(config, resume)
    output = Path(config.output_path)
    started = perf_counter()
    if redo_spectra:
        report["spectral_records"] = []
        report["complete"] = False

    def checkpoint() -> None:
        report["runtime_seconds_latest_invocation"] = perf_counter() - started
        report["summaries"] = _summaries(report) if report["work_precision_records"] else {}
        _atomic_write(output, report)

    _run_work_precision(report, config, checkpoint)
    _run_contrast(report, config, checkpoint)
    _run_spectra(report, config, checkpoint)
    report["summaries"] = _summaries(report)
    report["runtime_seconds_latest_invocation"] = perf_counter() - started
    report["complete"] = True
    _atomic_write(output, report)
    return report


def main() -> None:
    """Run the checkpointed single-reference comparison."""
    require_x64("PME DST/tau-blend single-reference benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=BaselineConfig().output_path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--redo-spectra", action="store_true")
    args = parser.parse_args()
    report = run(
        BaselineConfig(output_path=args.output),
        resume=args.resume,
        redo_spectra=args.redo_spectra,
    )
    print(
        "single-reference baseline study complete: "
        f"work={len(report['work_precision_records'])} "
        f"contrast={len(report['contrast_records'])} "
        f"spectral={len(report['spectral_records'])}"
    )


if __name__ == "__main__":
    main()
