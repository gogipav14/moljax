#!/usr/bin/env python3
"""Dense spectral analysis of experimental DST/tau PME preconditioners.

The spectra are exact dense eigendecompositions of the same left-preconditioned
Newton actions used by the counted-GMRES measurements.  This benchmark remains
limited to one-dimensional experimental linear systems; it is not a solver or
a full time-integration accuracy study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
    _physical_coefficient_summary,
    _project_physical_state,
)
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.conditioning import adjoint_identity, numerical_range
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import build_pme_linearization
from moljax.experimental.pme_dst_tau_blend import build_pme_tau_blend_linearization

METHODS = ("tau_blend", "frozen_mean", "frozen_bulk", "identity")


class SpectralConfig(NamedTuple):
    """Configuration for the dense one-dimensional spectral analysis."""

    nx: int = 512
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    epsilon: float = 1.0e-5
    halfwidth: float = 3.0
    analysis_dt: float = 2.0e-2
    reference_count: int = 3
    fov_angles: int = 32
    fov_max_iters: int = 60
    evolved_cases: tuple[tuple[int, int], ...] = ((2, 5), (8, 5))
    output_path: str = "benchmarks/results/pme_dst_tau_blend_spectral_analysis.json"


def _stage_config(config: SpectralConfig) -> Stage1Config:
    """Return the matching positivity-projected trajectory configuration."""
    return Stage1Config(
        x_min=config.x_min,
        x_max=config.x_max,
        t0=config.t0,
        epsilon=config.epsilon,
        evolved_nx=config.nx,
    )


def _complex_pairs(values: np.ndarray) -> list[list[float]]:
    """Serialize complex values without losing their real and imaginary parts."""
    return [[float(value.real), float(value.imag)] for value in values]


def _dense_operator_matrix(operator: Any) -> np.ndarray:
    """Materialize an operator by applying its action to all standard basis vectors."""
    basis = jnp.eye(operator.n, dtype=jnp.float64)
    basis_actions = jax.jit(jax.vmap(operator.matvec))(basis)
    return np.asarray(jax.block_until_ready(basis_actions.T), dtype=np.complex128)


def _dense_action_error(operator: Any, matrix: np.ndarray, seed: int) -> float:
    """Verify a dense materialization against the source matrix-free action."""
    generator = np.random.default_rng(seed)
    vector = generator.normal(size=operator.n) + 1j * generator.normal(size=operator.n)
    matrix_action = matrix @ vector
    source_action = np.asarray(
        jax.block_until_ready(operator.matvec(jnp.asarray(vector, dtype=jnp.complex128)))
    )
    return float(np.linalg.norm(matrix_action - source_action) / np.linalg.norm(source_action))


def _spectrum_summary(matrix: np.ndarray) -> tuple[np.ndarray, dict[str, float | int]]:
    """Return exact dense eigenvalues and their clustering/conditioning statistics."""
    eigenvalues = np.linalg.eigvals(matrix)
    magnitudes = np.abs(eigenvalues)
    distances_from_one = np.abs(eigenvalues - 1.0)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return eigenvalues, {
        "eigenvalue_real_mean": float(np.mean(eigenvalues.real)),
        "eigenvalue_real_median": float(np.median(eigenvalues.real)),
        "eigenvalue_real_min": float(np.min(eigenvalues.real)),
        "eigenvalue_real_max": float(np.max(eigenvalues.real)),
        "eigenvalue_imag_max_abs": float(np.max(np.abs(eigenvalues.imag))),
        "eigenvalue_abs_min": float(np.min(magnitudes)),
        "eigenvalue_abs_max": float(np.max(magnitudes)),
        "near_zero_eigenvalue_count": int(np.count_nonzero(magnitudes < 0.1)),
        "near_one_fraction": float(np.mean(distances_from_one < 0.5)),
        "distance_from_one_median": float(np.median(distances_from_one)),
        "distance_from_one_p95": float(np.quantile(distances_from_one, 0.95)),
        "minimum_singular_value": float(np.min(singular_values)),
        "maximum_singular_value": float(np.max(singular_values)),
        "condition_number_2": float(np.max(singular_values) / np.min(singular_values)),
    }


def _linearization(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    method: str,
    config: SpectralConfig,
) -> Any:
    """Build the same preconditioned action measured by the staged GMRES harness."""
    if method == "tau_blend":
        return build_pme_tau_blend_linearization(
            state,
            grid,
            float(m),
            config.analysis_dt,
            config.epsilon,
            reference_count=config.reference_count,
        )
    return build_pme_linearization(
        state,
        grid,
        float(m),
        config.analysis_dt,
        config.epsilon,
        method,
    )


def _tau_fov_summary(operator: Any, config: SpectralConfig, seed: int) -> dict[str, Any]:
    """Trace the existing matrix-free numerical-range diagnostic for one tau action."""
    field_of_values = numerical_range(
        operator.matvec,
        operator.matvec_adjoint,
        operator.n,
        n_angles=config.fov_angles,
        max_iters=config.fov_max_iters,
    )
    boundary = np.asarray(field_of_values.boundary, dtype=np.complex128)
    return {
        "adjoint_error": float(adjoint_identity(operator, jax.random.PRNGKey(seed), operator.n)),
        "boundary": _complex_pairs(boundary),
        "center": [float(field_of_values.center.real), float(field_of_values.center.imag)],
        "radius": float(field_of_values.radius),
        "disk_rate": float(field_of_values.disk_rate),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "disk_only_verdict": (
            "indeterminate"
            if field_of_values.origin_enclosed or field_of_values.disk_rate >= 1.0
            else "not_abstaining"
        ),
        "real_min": float(np.min(boundary.real)),
        "real_max": float(np.max(boundary.real)),
        "imag_min": float(np.min(boundary.imag)),
        "imag_max": float(np.max(boundary.imag)),
    }


def _state_for_evolved_case(
    grid: NodeCenteredDirichletGrid,
    m: int,
    steps: int,
    config: SpectralConfig,
) -> tuple[jax.Array, float]:
    """Regenerate one positivity-projected state using the Stage-1 advancement policy."""
    stage_config = _stage_config(config)
    state, _ = _project_physical_state(
        _initial_state(grid, m, config.t0, config.halfwidth),
        stage_config,
    )
    for _ in range(steps):
        state, _ = _advance_one_step(state, grid, m, stage_config)
    return state, config.t0 + steps * stage_config.evolved_state_dt


def _spectral_record(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    method: str,
    case: str,
    state_time: float,
    config: SpectralConfig,
    seed: int,
    *,
    include_fov: bool,
) -> dict[str, Any]:
    """Form one exact dense spectrum and optional matrix-free numerical range."""
    linearization = _linearization(state, grid, m, method, config)
    matrix = _dense_operator_matrix(linearization.operator)
    eigenvalues, summary = _spectrum_summary(matrix)
    record: dict[str, Any] = {
        "case": case,
        "m": m,
        "method": method,
        "state_time": state_time,
        "operator_dimension": linearization.operator.n,
        "dense_action_relative_error": _dense_action_error(linearization.operator, matrix, seed),
        "eigenvalues": _complex_pairs(eigenvalues),
        "spectrum": summary,
    }
    if method == "tau_blend":
        record["reference_values"] = [
            float(value) for value in linearization.preconditioner.reference_values
        ]
        record["inactive_node_count"] = int(
            jnp.sum(linearization.preconditioner.inactive_weight > 0.0)
        )
    if include_fov:
        field_of_values = _tau_fov_summary(linearization.operator, config, seed + 1000)
        field_of_values["spectral_real_min"] = summary["eigenvalue_real_min"]
        field_of_values["spectral_real_max"] = summary["eigenvalue_real_max"]
        field_of_values["spectral_abscissa"] = summary["eigenvalue_real_max"]
        field_of_values["numerical_abscissa"] = field_of_values["real_max"]
        field_of_values["numerical_minus_spectral_abscissa"] = (
            field_of_values["real_max"] - summary["eigenvalue_real_max"]
        )
        record["field_of_values"] = field_of_values
    return record


def run_spectral_analysis(config: SpectralConfig | None = None) -> dict[str, Any]:
    """Regenerate exact dense spectra for wide-front PME preconditioned actions."""
    if config is None:
        config = SpectralConfig()
    if config.reference_count != 3:
        raise ValueError("This analysis is fixed to the selected l=3 blend")
    started = perf_counter()
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    records: list[dict[str, Any]] = []
    stage_config = _stage_config(config)
    for index, m in enumerate((2, 4, 8)):
        state, _ = _project_physical_state(
            _initial_state(grid, m, config.t0, config.halfwidth),
            stage_config,
        )
        coefficient = _physical_coefficient_summary(state, m, stage_config)
        for offset, method in enumerate(METHODS):
            record = _spectral_record(
                state,
                grid,
                m,
                method,
                "initial_wide_front",
                config.t0,
                config,
                20261100 + 100 * index + offset,
                include_fov=method == "tau_blend",
            )
            record["coefficient"] = coefficient
            records.append(record)
    for index, (m, steps) in enumerate(config.evolved_cases):
        state, state_time = _state_for_evolved_case(grid, m, steps, config)
        record = _spectral_record(
            state,
            grid,
            m,
            "tau_blend",
            f"positivity_projected_step_{steps}",
            state_time,
            config,
            20261600 + 100 * index,
            include_fov=False,
        )
        record["coefficient"] = _physical_coefficient_summary(
            state,
            m,
            stage_config,
        )
        records.append(record)
    report = {
        "schema": "pme_dst_tau_blend_spectral_analysis_v1",
        "provenance": provenance_revisions(),
        "scope": (
            "Experimental one-dimensional dense spectral analysis of the same "
            "left-preconditioned Newton operators used by counted GMRES. "
            "Dense eigendecompositions are exact for these materialized 512-node actions; "
            "the field of values remains the existing matrix-free diagnostic trace."
        ),
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
    """Run the dense spectral analysis."""
    require_x64("PME DST/tau-blend spectral benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=SpectralConfig().output_path)
    args = parser.parse_args()
    report = run_spectral_analysis(SpectralConfig(output_path=args.output))
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")


if __name__ == "__main__":
    main()
