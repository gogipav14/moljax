#!/usr/bin/env python3
"""Validate cleaned PME trajectories and linear-solve work precision for a DST/tau blend.

This benchmark is deliberately limited to preconditioned Newton linear systems.  It does
not compare full nonlinear time-integration accuracy.
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
from pme_breakdown import _initial_state
from tau_blend_reproducibility import (
    load_or_generate_states,
    provenance_revisions,
    source_state_artifact,
)

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


class Stage1Config(NamedTuple):
    """Configuration for cleaned-state and linear-solve validation."""

    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    epsilon: float = 1.0e-5
    reference_count: int = 3
    analysis_dt: float = 2.0e-2
    max_krylov_iters: int = 400
    evolved_nx: int = 512
    evolved_m_values: tuple[int, ...] = (2, 8)
    evolved_state_dt: float = 1.0e-2
    evolved_steps: int = 8
    evolved_sample_steps: tuple[int, ...] = (0, 2, 5, 8)
    evolved_max_newton_iters: int = 12
    positivity_absolute_floor: float = 1.0e-14
    positivity_relative_floor: float = 1.0e-8
    active_absolute_floor: float = 1.0e-14
    active_relative_floor: float = 1.0e-6
    work_nx_values: tuple[int, ...] = (512, 1024)
    work_m_values: tuple[int, ...] = (2, 4, 8)
    work_tolerances: tuple[float, ...] = (1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10)
    timing_warmups: int = 2
    timing_repetitions_512: int = 7
    timing_repetitions_1024: int = 3
    source_state_cache_dir: str = "/tmp/moljax-tau-blend-source-states-v4"
    output_path: str = "benchmarks/results/pme_dst_tau_blend_stage1_validation.json"


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return median and quartiles without a numerical dependency."""
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    center = float(median(ordered))
    if len(ordered) == 1:
        return center, center, center
    return (
        center,
        float(median(ordered[: len(ordered) // 2])),
        float(median(ordered[(len(ordered) + 1) // 2 :])),
    )


def _project_physical_state(
    state: jax.Array,
    config: Stage1Config,
) -> tuple[jax.Array, dict[str, float | int]]:
    """Project negligible front undershoots to an exact non-negative compact support.

    Values below ``max(absolute_floor, relative_floor * max(u))`` are set to zero after
    each converged backward-Euler update.  The threshold is 1e-8 of the state maximum,
    so it removes floating-point front debris rather than resolved profile values.
    """
    values = jnp.asarray(state, dtype=jnp.float64)
    nonnegative = jnp.maximum(values, 0.0)
    state_maximum = float(jnp.max(nonnegative))
    cutoff = max(
        config.positivity_absolute_floor,
        config.positivity_relative_floor * state_maximum,
    )
    projected = jnp.where(nonnegative >= cutoff, nonnegative, 0.0)
    projected = jax.block_until_ready(projected)
    return projected, {
        "maximum_negative_undershoot_before_projection": float(jnp.max(jnp.maximum(-values, 0.0))),
        "support_cutoff": cutoff,
        "projected_to_zero_node_count": int(jnp.sum(projected == 0.0)),
        "minimum_state_after_projection": float(jnp.min(projected)),
    }


def _physical_coefficient_summary(
    state: jax.Array,
    m: int,
    config: Stage1Config,
) -> dict[str, float | int | str]:
    """Summarize a non-negative coefficient on a finite, physically active support."""
    diffusivity = jnp.maximum(
        porous_medium_diffusivity(state, float(m), config.epsilon),
        0.0,
    )
    maximum = float(jnp.max(diffusivity))
    threshold = max(config.active_absolute_floor, config.active_relative_floor * maximum)
    active = diffusivity > threshold
    active_values = diffusivity[active]
    if active_values.size == 0:
        raise RuntimeError("The cleaned state has no active diffusion support")
    minimum = float(jnp.min(diffusivity))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "literal_contrast": "infinite" if minimum == 0.0 and maximum > 0.0 else maximum / minimum,
        "active_threshold": threshold,
        "active_node_count": int(active_values.size),
        "active_d95_over_d05": float(
            jnp.quantile(active_values, 0.95) / jnp.quantile(active_values, 0.05)
        ),
    }


def _measure_method(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    config: Stage1Config,
    method: str,
) -> dict[str, Any]:
    """Measure one real counted-GMRES solve using an existing staged implementation."""
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
            )
        )
    return measure_gmres_iterations(
        state,
        grid,
        float(m),
        config.analysis_dt,
        config.epsilon,
        method,
        tol=tolerance,
        max_iters=config.max_krylov_iters,
    )


def _time_method(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    config: Stage1Config,
    method: str,
) -> dict[str, Any]:
    """Time synchronized fresh counted-GMRES solves at one requested residual target."""
    repetitions = (
        config.timing_repetitions_1024 if grid.nx >= 1024 else config.timing_repetitions_512
    )
    for _ in range(config.timing_warmups):
        warmup = _measure_method(state, grid, m, tolerance, config, method)
        jax.block_until_ready(jnp.asarray(warmup["final_relative_residual"], dtype=jnp.float64))

    elapsed: list[float] = []
    measurements: list[dict[str, Any]] = []
    for _ in range(repetitions):
        started = perf_counter()
        measurement = _measure_method(state, grid, m, tolerance, config, method)
        jax.block_until_ready(
            jnp.asarray(measurement["final_relative_residual"], dtype=jnp.float64)
        )
        elapsed.append(perf_counter() - started)
        measurements.append(measurement)
    center, lower, upper = _quartiles(elapsed)
    return {
        "repetitions": repetitions,
        "warmups_excluded": config.timing_warmups,
        "median_seconds": center,
        "iqr_seconds": upper - lower,
        "q1_seconds": lower,
        "q3_seconds": upper,
        "iterations": sorted({int(value["iterations"]) for value in measurements}),
        "all_converged": all(bool(value["converged"]) for value in measurements),
        "max_final_relative_residual": max(
            float(value["final_relative_residual"]) for value in measurements
        ),
        "includes": "fresh linearization and preconditioner construction plus counted GMRES",
    }


def _advance_one_step(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    config: Stage1Config,
) -> tuple[jax.Array, dict[str, Any]]:
    """Advance one converged backward-Euler step and apply the positivity projection."""
    residual = make_backward_euler_residual(
        state,
        grid,
        float(m),
        config.evolved_state_dt,
        config.epsilon,
    )
    preconditioner, d0 = pme_preconditioner_variant(
        state,
        grid,
        float(m),
        config.evolved_state_dt,
        config.epsilon,
        "frozen_bulk",
    )
    result = newton_krylov_solve(
        residual,
        state,
        grid,
        params={},
        preconditioner=preconditioner,
        nk_params=NKParams(
            max_newton_iters=config.evolved_max_newton_iters,
            max_krylov_iters=config.max_krylov_iters,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        ),
        dt=config.evolved_state_dt,
    )
    if not result.stats.converged:
        raise RuntimeError(f"Backward-Euler step did not converge for m={m}")
    projected, projection = _project_physical_state(result.solution, config)
    return projected, {
        "converged": True,
        "newton_iters": int(result.stats.newton_iters),
        "final_residual_l2": float(result.stats.final_res_norm),
        "frozen_bulk_d0": float(d0),
        "positivity_projection": projection,
    }


def _evolved_records(config: Stage1Config) -> list[dict[str, Any]]:
    """Measure real GMRES counts on positivity-projected evolved PME states."""
    grid = NodeCenteredDirichletGrid.uniform(config.evolved_nx, config.x_min, config.x_max)
    records: list[dict[str, Any]] = []
    for m in config.evolved_m_values:
        fingerprint = {
            "schema": "pme_tau_blend_generation_fingerprint_v4",
            "label": f"projected-wide-front-m{m}-n{config.evolved_nx}",
            "study": "pme_tau_blend_projected_evolved_trajectory",
            "m": m,
            "grid": {
                "nx": config.evolved_nx,
                "x_min": config.x_min,
                "x_max": config.x_max,
                "centering": "node",
                "boundary": "homogeneous_dirichlet",
            },
            "initial_state": {
                "kind": "barenblatt_wide_front",
                "time": config.t0,
                "halfwidth": 3.0,
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
                _initial_state(grid, m, config.t0, 3.0), config
            )
            generated = [initial_state]
            metadata: list[dict[str, Any]] = [
                {"step": 0, "advance_to_state": None, "projection": projection}
            ]
            current = initial_state
            for step_index in range(1, config.evolved_steps + 1):
                current, advance_stats = _advance_one_step(current, grid, m, config)
                generated.append(current)
                metadata.append({"step": step_index, "advance_to_state": advance_stats})
            return generated, metadata

        states, identities, metadata, cache_reused = load_or_generate_states(
            config.source_state_cache_dir,
            fingerprint["label"],
            fingerprint,
            generate,
        )
        initial = states[0]
        for step, state in enumerate(states):
            if step in config.evolved_sample_steps:
                methods = {
                    method: _measure_method(state, grid, m, 1.0e-8, config, method)
                    for method in METHODS
                }
                if not all(bool(value["converged"]) for value in methods.values()):
                    raise RuntimeError(f"GMRES did not converge on cleaned m={m}, step={step}")
                records.append(
                    {
                        "m": m,
                        "step": step,
                        "time": config.t0 + step * config.evolved_state_dt,
                        "developedness_max_abs_from_initial": float(
                            jnp.max(jnp.abs(state - initial))
                        ),
                        "coefficient": _physical_coefficient_summary(state, m, config),
                        "advance_to_state": metadata[step].get("advance_to_state"),
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
                        "methods": methods,
                    }
                )
    return records


def _work_precision_records(
    config: Stage1Config,
    records: list[dict[str, Any]],
    checkpoint: Callable[[list[dict[str, Any]]], None],
) -> list[dict[str, Any]]:
    """Collect and checkpoint linear-solve cost versus achieved residual."""
    completed = {
        (int(record["nx"]), int(record["m"]), float(record["requested_relative_residual"]))
        for record in records
    }
    for nx in config.work_nx_values:
        grid = NodeCenteredDirichletGrid.uniform(nx, config.x_min, config.x_max)
        for m in config.work_m_values:
            state, _ = _project_physical_state(_initial_state(grid, m, config.t0, 3.0), config)
            for tolerance in config.work_tolerances:
                if (nx, m, tolerance) in completed:
                    continue
                records.append(
                    {
                        "nx": nx,
                        "m": m,
                        "analysis_dt": config.analysis_dt,
                        "requested_relative_residual": tolerance,
                        "coefficient": _physical_coefficient_summary(state, m, config),
                        "methods": {
                            method: _time_method(state, grid, m, tolerance, config, method)
                            for method in METHODS
                        },
                    }
                )
                checkpoint(records)
    return records


def _matched_accuracy_summary(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Summarize costs only where each requested preconditioned residual target is met."""
    summaries: list[dict[str, Any]] = []
    for record in records:
        target = float(record["requested_relative_residual"])
        methods = record["methods"]
        tau = methods["tau_blend"]
        comparisons: dict[str, Any] = {}
        for baseline in ("frozen_mean", "frozen_bulk", "identity"):
            other = methods[baseline]
            comparable = bool(tau["all_converged"] and other["all_converged"])
            comparisons[baseline] = {
                "both_reach_requested_target": comparable
                and tau["max_final_relative_residual"] <= target
                and other["max_final_relative_residual"] <= target,
                "tau_time_ratio_over_baseline": (
                    tau["median_seconds"] / other["median_seconds"] if comparable else None
                ),
            }
        summaries.append(
            {
                "nx": record["nx"],
                "m": record["m"],
                "requested_relative_residual": target,
                "comparisons": comparisons,
            }
        )
    return summaries


def _write_report(output: Path, report: dict[str, Any]) -> None:
    """Atomically write a self-consistent measured-data checkpoint."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def _report(
    config: Stage1Config,
    evolved: list[dict[str, Any]],
    work_precision: list[dict[str, Any]],
    started: float,
    *,
    complete: bool,
) -> dict[str, Any]:
    """Build a checkpointable report containing only measured values."""
    return {
        "schema": "pme_dst_tau_blend_stage1_validation_v1",
        "scope": (
            "Experimental one-dimensional preconditioned linear-solve validation. "
            "It does not establish full nonlinear time-integration work precision."
        ),
        "positivity_policy": (
            "After each converged backward-Euler update, values below max(1e-14, "
            "1e-8 * max(u)) are set to exactly zero.  Coefficient contrast is computed "
            "only over D > max(1e-14, 1e-6 * max(D))."
        ),
        "timing_protocol": (
            "Two warmups excluded; synchronized full linearization/preconditioner "
            "construction plus counted GMRES; median and IQR over reported repetitions."
        ),
        "provenance": provenance_revisions(),
        "config": config._asdict(),
        "evolved_records": evolved,
        "work_precision_records": work_precision,
        "matched_accuracy_summary": _matched_accuracy_summary(work_precision),
        "runtime_seconds": perf_counter() - started,
        "complete": complete,
    }


def run_stage1_validation(
    config: Stage1Config | None = None,
    *,
    include_evolved: bool = True,
    include_work_precision: bool = True,
    resume: bool = False,
) -> dict[str, Any]:
    """Regenerate cleaned-state and linear-solve work-precision measurements."""
    if config is None:
        config = Stage1Config()
    if config.reference_count != 3:
        raise ValueError("This validation is fixed to the selected l=3 blend")
    started = perf_counter()
    output = Path(config.output_path)
    existing: dict[str, Any] = {}
    if resume and output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        serialized_config = json.loads(json.dumps(config._asdict()))
        if existing.get("config") != serialized_config:
            raise ValueError("The existing checkpoint uses a different validation configuration")
    evolved = list(existing.get("evolved_records", []))
    work_precision = list(existing.get("work_precision_records", []))
    if include_evolved and not evolved:
        evolved = _evolved_records(config)

    def checkpoint(updated_work_precision: list[dict[str, Any]]) -> None:
        _write_report(
            output,
            _report(config, evolved, updated_work_precision, started, complete=False),
        )

    if include_work_precision:
        work_precision = _work_precision_records(config, work_precision, checkpoint)
    report = _report(config, evolved, work_precision, started, complete=True)
    _write_report(output, report)
    return report


def main() -> None:
    """Run the cleaned-state and work-precision validation."""
    require_x64("PME DST/tau-blend stage-1 validation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=Stage1Config().output_path)
    parser.add_argument("--skip-evolved", action="store_true")
    parser.add_argument("--skip-work-precision", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run_stage1_validation(
        Stage1Config(output_path=args.output),
        include_evolved=not args.skip_evolved,
        include_work_precision=not args.skip_work_precision,
        resume=args.resume,
    )
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")


if __name__ == "__main__":
    main()
