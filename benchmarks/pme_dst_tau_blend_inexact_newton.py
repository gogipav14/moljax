#!/usr/bin/env python3
"""Measure complete PME inexact-Newton solves with the experimental DST/tau blend."""

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

import jax.numpy as jnp
from pme_breakdown import _initial_state
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_inexact_newton import (
    InexactNewtonConfig,
    InexactNewtonResult,
    PMEBackwardEulerProblem,
    solve_pme_inexact_newton,
)

METHODS = ("tau_blend", "frozen_mean", "frozen_bulk", "identity")


class MeasurementConfig(NamedTuple):
    """Configuration for the matched-nonlinear-accuracy measurement."""

    nx: int = 512
    m_values: tuple[int, ...] = (2, 4, 8)
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    halfwidth: float = 3.0
    dt: float = 2.0e-2
    epsilon: float = 1.0e-5
    reference_count: int = 3
    nonlinear_tolerance: float = 1.0e-8
    max_newton_iters: int = 20
    max_gmres_iters: int = 400
    eta_initial: float = 0.5
    eta_min: float = 1.0e-10
    eta_max: float = 0.5
    timing_warmups: int = 1
    timing_repetitions: int = 5
    output_path: str = "benchmarks/results/pme_dst_tau_blend_inexact_newton.json"


def _quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return median and quartiles for a nonempty timing sample."""
    if not values:
        raise ValueError("values must be nonempty")
    ordered = sorted(values)
    center = float(median(ordered))
    if len(ordered) == 1:
        return center, center, center
    return (
        center,
        float(median(ordered[: len(ordered) // 2])),
        float(median(ordered[(len(ordered) + 1) // 2 :])),
    )


def _serialize_result(result: InexactNewtonResult) -> dict[str, Any]:
    """Serialize one deterministic work trace without embedding its state vector."""
    return {
        "converged": result.converged,
        "nonlinear_iterations": result.nonlinear_iterations,
        "total_gmres_iterations": result.total_gmres_iterations,
        "initial_residual_norm": result.initial_residual_norm,
        "final_residual_norm": result.final_residual_norm,
        "steps": [asdict(step) for step in result.steps],
    }


def _write_report(output: Path, report: dict[str, Any]) -> None:
    """Write an atomic measured-data checkpoint."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)


def _measurement_controls(config: MeasurementConfig) -> InexactNewtonConfig:
    """Map report configuration to the solver's Eisenstat--Walker controls."""
    return InexactNewtonConfig(
        nonlinear_tolerance=config.nonlinear_tolerance,
        max_newton_iters=config.max_newton_iters,
        max_gmres_iters=config.max_gmres_iters,
        eta_initial=config.eta_initial,
        eta_min=config.eta_min,
        eta_max=config.eta_max,
    )


def _solve(
    problem: PMEBackwardEulerProblem,
    method: str,
    config: MeasurementConfig,
) -> InexactNewtonResult:
    """Run one synchronized full nonlinear solve."""
    result = solve_pme_inexact_newton(
        problem,
        method=method,
        config=_measurement_controls(config),
        reference_count=config.reference_count,
    )
    jax.block_until_ready(result.solution)
    return result


def _time_methods(
    problem: PMEBackwardEulerProblem,
    config: MeasurementConfig,
) -> tuple[dict[str, Any], dict[str, jax.Array]]:
    """Warm up then time methods in rotating order to limit order bias."""
    for method in METHODS:
        for _ in range(config.timing_warmups):
            warmup = _solve(problem, method, config)
            if not warmup.converged:
                raise RuntimeError(f"Warmup did not converge for {method}")

    elapsed = {method: [] for method in METHODS}
    results: dict[str, list[InexactNewtonResult]] = {method: [] for method in METHODS}
    for repetition in range(config.timing_repetitions):
        offset = repetition % len(METHODS)
        order = METHODS[offset:] + METHODS[:offset]
        for method in order:
            started = perf_counter()
            result = _solve(problem, method, config)
            elapsed[method].append(perf_counter() - started)
            results[method].append(result)
            if not result.converged:
                raise RuntimeError(f"Timed solve did not converge for {method}")

    summaries: dict[str, Any] = {}
    solutions: dict[str, jax.Array] = {}
    for method in METHODS:
        center, lower, upper = _quartiles(elapsed[method])
        traces = results[method]
        iteration_pairs = sorted(
            {(result.nonlinear_iterations, result.total_gmres_iterations) for result in traces}
        )
        if len(iteration_pairs) != 1:
            raise RuntimeError(f"Nondeterministic iteration trace for {method}: {iteration_pairs}")
        summaries[method] = {
            "representative": _serialize_result(traces[0]),
            "timing": {
                "warmups_excluded": config.timing_warmups,
                "repetitions": config.timing_repetitions,
                "median_seconds": center,
                "q1_seconds": lower,
                "q3_seconds": upper,
                "iqr_seconds": upper - lower,
                "samples_seconds": elapsed[method],
            },
        }
        solutions[method] = traces[0].solution
    return summaries, solutions


def _matched_solution_summary(
    solutions: dict[str, jax.Array],
    methods: dict[str, Any],
    nonlinear_tolerance: float,
) -> dict[str, Any]:
    """Quantify whether all methods reached the same nonlinear root."""
    reference = solutions["tau_blend"]
    reference_norm = max(float(jnp.linalg.norm(reference)), float(jnp.finfo(jnp.float64).tiny))
    comparisons: dict[str, Any] = {}
    for method in METHODS:
        difference = solutions[method] - reference
        comparisons[method] = {
            "relative_l2_vs_tau_blend": float(jnp.linalg.norm(difference)) / reference_norm,
            "max_abs_vs_tau_blend": float(jnp.max(jnp.abs(difference))),
            "final_residual_norm": methods[method]["representative"]["final_residual_norm"],
        }
    return {
        "nonlinear_tolerance": nonlinear_tolerance,
        "all_converged_to_tolerance": all(
            value["representative"]["converged"]
            and value["representative"]["final_residual_norm"] <= nonlinear_tolerance
            for value in methods.values()
        ),
        "comparisons": comparisons,
    }


def _forcing_schedule_description() -> str:
    """Return the exact adaptive forcing contract used by the harness."""
    return (
        "Eisenstat-Walker choice 1: eta_0=0.5; after accepted step s_k, raw eta is "
        "abs(||F(x_{k+1})||-||F(x_k)+J_k s_k||)/||F(x_k)||. If "
        "eta_k**phi>0.1 (phi=(1+sqrt(5))/2), the next eta is at least eta_k**phi; "
        "the result is clamped to [1e-10,0.5]. Each linear solve is accepted only "
        "when the true residual ||J_k s_k+F(x_k)||/||F(x_k)|| is at most eta_k."
    )


def run_measurement(config: MeasurementConfig | None = None) -> dict[str, Any]:
    """Run the complete m=2/4/8 matched-accuracy nonlinear measurement."""
    if config is None:
        config = MeasurementConfig()
    if config.reference_count != 3:
        raise ValueError("This measurement is fixed to the selected l=3 blend")
    started = perf_counter()
    records: list[dict[str, Any]] = []
    output = Path(config.output_path)
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    for m in config.m_values:
        previous_state = _initial_state(grid, m, config.t0, config.halfwidth)
        problem = PMEBackwardEulerProblem(
            previous_state=previous_state,
            grid=grid,
            m=float(m),
            dt=config.dt,
            epsilon=config.epsilon,
        )
        methods, solutions = _time_methods(problem, config)
        records.append(
            {
                "m": m,
                "nx": config.nx,
                "dt": config.dt,
                "methods": methods,
                "matched_solution": _matched_solution_summary(
                    solutions,
                    methods,
                    config.nonlinear_tolerance,
                ),
            }
        )
        _write_report(
            output,
            _report(config, records, perf_counter() - started, complete=False),
        )
    report = _report(config, records, perf_counter() - started, complete=True)
    _write_report(output, report)
    return report


def _report(
    config: MeasurementConfig,
    records: list[dict[str, Any]],
    runtime_seconds: float,
    *,
    complete: bool,
) -> dict[str, Any]:
    """Assemble the self-describing result document."""
    return {
        "schema": "pme_dst_tau_blend_inexact_newton_v1",
        "provenance": provenance_revisions(),
        "scope": (
            "Experimental one-dimensional full backward-Euler nonlinear-solve measurement. "
            "All methods solve the same nonlinear equation to the same residual tolerance."
        ),
        "forcing_schedule": _forcing_schedule_description(),
        "timing_protocol": (
            "One complete nonlinear-solve warmup per method is excluded. Timed methods are "
            "rotated by repetition; each solve is synchronized; medians and IQRs include "
            "preconditioner rebuilding at every Newton iterate and all GMRES work."
        ),
        "device": {
            "default_backend": jax.default_backend(),
            "devices": [
                {"platform": device.platform, "device_kind": device.device_kind}
                for device in jax.devices()
            ],
        },
        "config": config._asdict(),
        "records": records,
        "runtime_seconds": runtime_seconds,
        "complete": complete,
    }


def main() -> None:
    """Run the full inexact-Newton measurement."""
    require_x64("PME DST/tau-blend inexact-Newton benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=MeasurementConfig().output_path)
    parser.add_argument("--nx", type=int, default=MeasurementConfig().nx)
    parser.add_argument(
        "--timing-repetitions",
        type=int,
        default=MeasurementConfig().timing_repetitions,
    )
    parser.add_argument(
        "--timing-warmups",
        type=int,
        default=MeasurementConfig().timing_warmups,
    )
    args = parser.parse_args()
    report = run_measurement(
        MeasurementConfig(
            nx=args.nx,
            timing_repetitions=args.timing_repetitions,
            timing_warmups=args.timing_warmups,
            output_path=args.output,
        )
    )
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    for record in report["records"]:
        print(f"m={record['m']}")
        for method, values in record["methods"].items():
            result = values["representative"]
            timing = values["timing"]
            print(
                f"  {method}: newton={result['nonlinear_iterations']} "
                f"gmres={result['total_gmres_iterations']} "
                f"median_seconds={timing['median_seconds']:.6f}"
            )


if __name__ == "__main__":
    main()
