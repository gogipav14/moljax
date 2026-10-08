#!/usr/bin/env python3
"""Validate scaling, solve time, and evolved-state robustness of a DST/tau PME blend."""

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
from pme_breakdown import _initial_state
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.core.newton_krylov import NKParams, newton_krylov_solve
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import (
    make_backward_euler_residual,
    measure_gmres_iterations,
    pme_preconditioner_variant,
)
from moljax.experimental.pme_dst_tau_blend import (
    measure_pme_tau_blend_gmres_iterations,
    porous_medium_diffusivity,
)

METHODS = ("tau_blend", "frozen_mean", "frozen_bulk", "identity")


class ValidationConfig(NamedTuple):
    """Configuration for the one-dimensional wide-front validation study."""

    nx_values: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
    m_values: tuple[int, ...] = (2, 4, 8)
    halfwidth: float = 3.0
    analysis_dt: float = 2.0e-2
    stress_m: int = 8
    stress_dt: float = 2.0
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    epsilon: float = 1.0e-5
    reference_count: int = 3
    krylov_tol: float = 1.0e-8
    max_krylov_iters: int = 400
    timing_repetitions: int = 31
    large_n_timing_repetitions: int = 15
    large_n_threshold: int = 2048
    timing_warmups: int = 2
    trajectory_nx: int = 512
    trajectory_state_dt: float = 1.0e-2
    trajectory_steps: int = 8
    trajectory_sample_steps: tuple[int, ...] = (0, 2, 5, 8)
    trajectory_solver_preconditioner: str = "frozen_bulk"
    trajectory_max_newton_iters: int = 12
    output_path: str = "benchmarks/results/pme_dst_tau_blend_validation.json"


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return median, lower quartile, and upper quartile without dependencies."""
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    center = float(median(ordered))
    if len(ordered) == 1:
        return center, center, center
    lower = float(median(ordered[: len(ordered) // 2]))
    upper = float(median(ordered[(len(ordered) + 1) // 2 :]))
    return center, lower, upper


def _coefficient_summary(state: jax.Array, m: int, epsilon: float) -> dict[str, float | int | str]:
    """Summarize the nonzero diagonal coefficient without hiding compact support."""
    diffusivity = porous_medium_diffusivity(state, float(m), epsilon)
    positive = diffusivity[diffusivity > 0.0]
    minimum = float(jnp.min(diffusivity))
    maximum = float(jnp.max(diffusivity))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "literal_contrast": "infinite" if minimum == 0.0 and maximum > 0.0 else maximum / minimum,
        "active_node_count": int(positive.size),
        "active_d95_over_d05": float(jnp.quantile(positive, 0.95) / jnp.quantile(positive, 0.05)),
    }


def _measure_method(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    dt: float,
    config: ValidationConfig,
    method: str,
) -> dict[str, Any]:
    """Build and solve one exact first-Newton system with a named method."""
    if method == "tau_blend":
        return dict(
            measure_pme_tau_blend_gmres_iterations(
                state,
                grid,
                float(m),
                dt,
                config.epsilon,
                reference_count=config.reference_count,
                tol=config.krylov_tol,
                max_iters=config.max_krylov_iters,
            )
        )
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return measure_gmres_iterations(
        state,
        grid,
        float(m),
        dt,
        config.epsilon,
        method,
        tol=config.krylov_tol,
        max_iters=config.max_krylov_iters,
    )


def _timed_method(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    dt: float,
    config: ValidationConfig,
    method: str,
) -> dict[str, Any]:
    """Time fresh full GMRES solves after synchronized warmups."""
    repetitions = (
        config.large_n_timing_repetitions
        if grid.nx >= config.large_n_threshold
        else config.timing_repetitions
    )
    for _ in range(config.timing_warmups):
        warmup = _measure_method(state, grid, m, dt, config, method)
        jax.block_until_ready(jnp.asarray(warmup["final_relative_residual"], dtype=jnp.float64))

    elapsed_seconds: list[float] = []
    iterations: list[int] = []
    converged: list[bool] = []
    residuals: list[float] = []
    for _ in range(repetitions):
        started = perf_counter()
        measurement = _measure_method(state, grid, m, dt, config, method)
        jax.block_until_ready(
            jnp.asarray(measurement["final_relative_residual"], dtype=jnp.float64)
        )
        elapsed_seconds.append(perf_counter() - started)
        iterations.append(int(measurement["iterations"]))
        converged.append(bool(measurement["converged"]))
        residuals.append(float(measurement["final_relative_residual"]))

    center, lower, upper = _quartiles(elapsed_seconds)
    return {
        "repetitions": repetitions,
        "warmups_excluded": config.timing_warmups,
        "median_seconds": center,
        "iqr_seconds": upper - lower,
        "q1_seconds": lower,
        "q3_seconds": upper,
        "iteration_values": sorted(set(iterations)),
        "all_converged": all(converged),
        "max_final_relative_residual": max(residuals),
        "includes": "fresh linearization and preconditioner construction plus counted GMRES",
    }


def _scaling_cases(config: ValidationConfig) -> tuple[tuple[str, int, float], ...]:
    """Return standard wide-front cases plus the prescribed stiff case."""
    standard = tuple((f"m{m}_wide", m, config.analysis_dt) for m in config.m_values)
    return standard + ((f"m{config.stress_m}_wide_stiff", config.stress_m, config.stress_dt),)


def _scaling_records(
    config: ValidationConfig,
    records: list[dict[str, Any]],
    checkpoint: Callable[[list[dict[str, Any]] | None, list[dict[str, Any]] | None], None],
) -> list[dict[str, Any]]:
    """Collect scaling records and checkpoint each fully timed resolution case."""
    completed = {(str(record["case"]), int(record["nx"])) for record in records}
    for label, m, dt in _scaling_cases(config):
        for nx in config.nx_values:
            if (label, nx) in completed:
                continue
            grid = NodeCenteredDirichletGrid.uniform(nx, config.x_min, config.x_max)
            state = _initial_state(grid, m, config.t0, config.halfwidth)
            jax.block_until_ready(state)
            records.append(
                {
                    "case": label,
                    "nx": nx,
                    "m": m,
                    "analysis_dt": dt,
                    "state_kind": "wide_front_barenblatt_initial_first_newton_system",
                    "coefficient": _coefficient_summary(state, m, config.epsilon),
                    "methods": {
                        method: {
                            "actual_gmres": _measure_method(state, grid, m, dt, config, method),
                            "wall_clock": _timed_method(state, grid, m, dt, config, method),
                        }
                        for method in METHODS
                    },
                }
            )
            checkpoint(records, None)
    return records


def _advance_one_step(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    config: ValidationConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    """Advance one backward-Euler state with an existing frozen preconditioner."""
    residual = make_backward_euler_residual(
        state,
        grid,
        float(m),
        config.trajectory_state_dt,
        config.epsilon,
    )
    preconditioner, d0 = pme_preconditioner_variant(
        state,
        grid,
        float(m),
        config.trajectory_state_dt,
        config.epsilon,
        config.trajectory_solver_preconditioner,
    )
    result = newton_krylov_solve(
        residual,
        state,
        grid,
        params={},
        preconditioner=preconditioner,
        nk_params=NKParams(
            max_newton_iters=config.trajectory_max_newton_iters,
            max_krylov_iters=config.max_krylov_iters,
            newton_tol=1.0e-8,
            krylov_tol=config.krylov_tol,
        ),
        dt=config.trajectory_state_dt,
    )
    solution = jax.block_until_ready(result.solution)
    return solution, {
        "converged": bool(result.stats.converged),
        "newton_iters": int(result.stats.newton_iters),
        "configured_krylov_budget_total": int(result.stats.lin_iters),
        "final_residual_l2": float(result.stats.final_res_norm),
        "d0": float(d0),
    }


def _trajectory_records(
    config: ValidationConfig,
    records: list[dict[str, Any]],
    checkpoint: Callable[[list[dict[str, Any]] | None, list[dict[str, Any]] | None], None],
) -> list[dict[str, Any]]:
    """Measure evolved states and checkpoint each fully measured trajectory state."""
    completed = {(int(record["m"]), int(record["step"])) for record in records}
    grid = NodeCenteredDirichletGrid.uniform(config.trajectory_nx, config.x_min, config.x_max)
    for m in (2, 8):
        initial = _initial_state(grid, m, config.t0, config.halfwidth)
        state = jax.block_until_ready(initial)
        advance_stats: dict[str, Any] | None = None
        for step in range(config.trajectory_steps + 1):
            if step in config.trajectory_sample_steps:
                if (m, step) not in completed:
                    records.append(
                        {
                            "m": m,
                            "step": step,
                            "time": config.t0 + step * config.trajectory_state_dt,
                            "developedness_max_abs_from_initial": float(
                                jnp.max(jnp.abs(state - initial))
                            ),
                            "coefficient": _coefficient_summary(state, m, config.epsilon),
                            "advance_to_state": advance_stats,
                            "methods": {
                                method: _measure_method(
                                    state,
                                    grid,
                                    m,
                                    config.analysis_dt,
                                    config,
                                    method,
                                )
                                for method in METHODS
                            },
                        }
                    )
                    checkpoint(None, records)
            if step == config.trajectory_steps:
                continue
            state, advance_stats = _advance_one_step(state, grid, m, config)
            if not advance_stats["converged"]:
                raise RuntimeError(f"Trajectory step {step + 1} did not converge for m={m}")
    return records


def _crossover_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report the first measured tau-versus-identity time crossover per case."""
    summary: dict[str, Any] = {}
    for case in sorted({str(record["case"]) for record in records}):
        case_records = sorted(
            (record for record in records if record["case"] == case),
            key=lambda record: int(record["nx"]),
        )
        faster_than_identity = [
            int(record["nx"])
            for record in case_records
            if record["methods"]["tau_blend"]["wall_clock"]["median_seconds"]
            < record["methods"]["identity"]["wall_clock"]["median_seconds"]
        ]
        faster_than_frozen = {
            method: [
                int(record["nx"])
                for record in case_records
                if record["methods"]["tau_blend"]["wall_clock"]["median_seconds"]
                < record["methods"][method]["wall_clock"]["median_seconds"]
            ]
            for method in ("frozen_mean", "frozen_bulk")
        }
        summary[case] = {
            "first_tau_faster_than_identity_nx": (
                faster_than_identity[0] if faster_than_identity else None
            ),
            "tau_faster_than_identity_nx_values": faster_than_identity,
            "tau_faster_than_frozen_nx_values": faster_than_frozen,
        }
    return summary


def _write_report(output: Path, report: dict[str, Any]) -> None:
    """Atomically persist a valid report after each completed measurement case."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def run_validation(
    config: ValidationConfig | None = None, *, resume: bool = False
) -> dict[str, Any]:
    """Regenerate the full scaling, timing, and trajectory validation dataset."""
    if config is None:
        config = ValidationConfig()
    if config.reference_count != 3:
        raise ValueError("This validation is fixed to the l=3 blend selected by the prior study")
    if not config.nx_values or min(config.nx_values) < 2:
        raise ValueError("nx_values must contain node counts of at least two")
    started = perf_counter()
    output = Path(config.output_path)
    existing: dict[str, Any] = {}
    if resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("config") != config._asdict():
            raise ValueError("The existing checkpoint uses a different validation configuration")
    scaling = list(existing.get("scaling_records", []))
    trajectory = list(existing.get("trajectory_records", []))

    def checkpoint(
        updated_scaling: list[dict[str, Any]] | None,
        updated_trajectory: list[dict[str, Any]] | None,
    ) -> None:
        report = _report(
            config,
            scaling if updated_scaling is None else updated_scaling,
            trajectory if updated_trajectory is None else updated_trajectory,
            started,
            complete=False,
        )
        _write_report(output, report)

    scaling = _scaling_records(config, scaling, checkpoint)
    trajectory = _trajectory_records(config, trajectory, checkpoint)
    report = _report(config, scaling, trajectory, started, complete=True)
    _write_report(output, report)
    return report


def _report(
    config: ValidationConfig,
    scaling: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    started: float,
    *,
    complete: bool,
) -> dict[str, Any]:
    """Build a self-describing report from completed checkpoint records."""
    return {
        "schema": "pme_dst_tau_blend_validation_v1",
        "provenance": provenance_revisions(),
        "scope": (
            "Experimental one-dimensional scaling, solve-time, and evolved-state validation; "
            "it is not a public solver component."
        ),
        "device": {
            "default_backend": jax.default_backend(),
            "devices": [
                {"platform": device.platform, "device_kind": device.device_kind}
                for device in jax.devices()
            ],
        },
        "timing_protocol": (
            "Two JIT warmups excluded; synchronized full linearization/preconditioner "
            "construction plus counted GMRES; median and IQR over the reported repetitions."
        ),
        "config": config._asdict(),
        "scaling_records": scaling,
        "trajectory_records": trajectory,
        "wall_clock_crossover": _crossover_summary(scaling),
        "runtime_seconds": perf_counter() - started,
        "complete": complete,
    }


def main() -> None:
    """Run the full DST/tau blend validation experiment."""
    require_x64("PME DST/tau-blend validation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=ValidationConfig().output_path)
    parser.add_argument("--nx", nargs="+", type=int, default=None)
    parser.add_argument("--timing-repetitions", type=int, default=None)
    parser.add_argument("--large-n-timing-repetitions", type=int, default=None)
    parser.add_argument("--large-n-threshold", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = ValidationConfig(
        nx_values=tuple(args.nx) if args.nx is not None else ValidationConfig().nx_values,
        timing_repetitions=(
            args.timing_repetitions
            if args.timing_repetitions is not None
            else ValidationConfig().timing_repetitions
        ),
        large_n_timing_repetitions=(
            args.large_n_timing_repetitions
            if args.large_n_timing_repetitions is not None
            else ValidationConfig().large_n_timing_repetitions
        ),
        large_n_threshold=(
            args.large_n_threshold
            if args.large_n_threshold is not None
            else ValidationConfig().large_n_threshold
        ),
        output_path=args.output,
    )
    report = run_validation(config, resume=args.resume)
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    for case, values in report["wall_clock_crossover"].items():
        print(
            f"{case}: first_tau_faster_than_identity_nx={values['first_tau_faster_than_identity_nx']}"
        )


if __name__ == "__main__":
    main()
