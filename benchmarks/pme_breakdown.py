"""Decision-grade conditioning study for experimental porous-medium backward Euler."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from math import isfinite, sqrt
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from moljax.core.grid import Grid1D
from moljax.core.newton_krylov import NKParams, newton_krylov_solve
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_conditioning import (
    assess_pme_state,
    interior_values,
    make_backward_euler_residual,
    measure_gmres_iterations,
    padded_values,
    pme_preconditioner_variant,
    predicted_iterations_from_envelope,
)
from moljax.experimental.pme_preconditioner import helmholtz_inverse_relative_residual


class BreakdownConfig(NamedTuple):
    """Static parameters for the one-dimensional PME conditioning sweep.

    The study starts from an early-time compact profile and analyses the first
    three backward-Euler states. PME smooths rather than sharpens in forward
    time, so the schedule retains a range of steep fronts while
    ``dt / dx**2`` remains order one for the linear control.
    """

    nx: int = 48
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    dt: float = 0.02
    epsilon: float = 1.0e-5
    visited_steps: tuple[int, ...] = (1, 2, 3)
    n_angles: int = 3
    fov_max_iters: int = 10
    arnoldi_steps: int = 6
    const_d0: float = 1.0
    max_newton_iters: int = 8
    max_krylov_iters: int = 64
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    m_values: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    d0_kinds: tuple[str, ...] = ("frozen_mean", "frozen_bulk", "floor", "const", "identity")
    output_path: str = "benchmarks/results/pme_breakdown.json"


def _heat_gaussian(x: jax.Array, t: float) -> jax.Array:
    """Return the linear-control heat kernel, negligible at the domain faces."""
    return jnp.exp(-(x**2) / (4.0 * t)) / jnp.sqrt(4.0 * jnp.pi * t)


def _barenblatt_b(m: int, target_radius: float = 1.8) -> float:
    """Choose ``b`` so the unit-time Barenblatt support has ``target_radius``."""
    beta = 1.0 / (m - 1.0 + 2.0)
    return target_radius**2 * (m - 1.0) * beta / (2.0 * m)


def _initial_state(
    grid: NodeCenteredDirichletGrid,
    m: int,
    t0: float,
) -> tuple[jax.Array, float | None]:
    """Return a node-centred control or Barenblatt initial state and its ``b``."""
    x = grid.x_coords()
    if m == 1:
        return _heat_gaussian(x, t0), None
    b = _barenblatt_b(m)
    return barenblatt(x, t0, float(m), b=b), b


def _exact_solution(
    grid: NodeCenteredDirichletGrid,
    m: int,
    time: float,
    b: float | None,
) -> jax.Array:
    """Return the unregularized analytic comparison profile on interior nodes."""
    x = grid.x_coords()
    if m == 1:
        return _heat_gaussian(x, time)
    if b is None:
        raise ValueError("Barenblatt comparison requires b")
    return barenblatt(x, time, float(m), b=b)


def _solve_one_step(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    config: BreakdownConfig,
    d0_kind: str,
) -> tuple[jax.Array, dict[str, Any]]:
    """Take one BE solve and return its state and nonlinear-solver statistics."""
    epsilon = 0.0 if m == 1 else config.epsilon
    interior = interior_values(state, grid)
    residual = make_backward_euler_residual(interior, grid, float(m), config.dt, epsilon)
    preconditioner, d0 = pme_preconditioner_variant(
        interior,
        grid,
        float(m),
        config.dt,
        epsilon,
        d0_kind,
        const_value=config.const_d0,
    )
    result = newton_krylov_solve(
        residual,
        interior,
        grid,
        params={},
        preconditioner=preconditioner,
        nk_params=NKParams(
            max_newton_iters=config.max_newton_iters,
            max_krylov_iters=config.max_krylov_iters,
            newton_tol=config.newton_tol,
            krylov_tol=config.krylov_tol,
        ),
        dt=config.dt,
    )
    solution = jax.block_until_ready(result.solution)
    return padded_values(solution, grid), {
        "d0": float(d0),
        "converged": bool(result.stats.converged),
        "newton_iters": int(result.stats.newton_iters),
        "configured_krylov_budget_total": int(result.stats.lin_iters),
        "final_residual_l2": float(result.stats.final_res_norm),
    }


def _centering_report(config: BreakdownConfig) -> dict[str, Any]:
    """Quantify the legacy cell/DST mismatch and validate the selected grid."""
    legacy_grid = Grid1D.uniform(config.nx, config.x_min, config.x_max)
    node_grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    settings = ((1.0, config.dt), (1.0, 5.0 * config.dt), (0.16, config.dt))
    cases = []
    for index, (d0, dt) in enumerate(settings):
        key = jax.random.PRNGKey(20260950 + index)
        legacy = helmholtz_inverse_relative_residual(d0, dt, legacy_grid, key)
        node = helmholtz_inverse_relative_residual(d0, dt, node_grid, key)
        cases.append(
            {
                "d0": d0,
                "dt": dt,
                "legacy_cell_centered_relative_residual": legacy,
                "node_centered_relative_residual": node,
            }
        )
    largest_legacy = max(case["legacy_cell_centered_relative_residual"] for case in cases)
    return {
        "cases": cases,
        "adopted_discretization": "node_centered_dirichlet",
        "decision": (
            "The cell-centred residual is material relative to conditioning-rate differences; "
            "the experimental study therefore uses the node-centred operator that the DST-I "
            "Helmholtz inverse diagonalizes exactly."
        ),
        "largest_legacy_relative_residual": largest_legacy,
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    """Return dependency-free count/minimum/median/maximum statistics."""
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": float(min(values)),
        "median": float(median(values)),
        "max": float(max(values)),
    }


def _ranges_overlap(
    first: dict[str, float | int | None], second: dict[str, float | int | None]
) -> bool:
    """Return whether two non-empty closed summary intervals overlap."""
    if first["count"] == 0 or second["count"] == 0:
        return False
    return max(float(first["min"]), float(second["min"])) <= min(
        float(first["max"]), float(second["max"])
    )


def _pearson(first: list[float], second: list[float]) -> float | None:
    """Compute a simple Pearson correlation without external dependencies."""
    if len(first) != len(second):
        raise ValueError("Pearson inputs must have matching lengths")
    if len(first) < 2:
        return None
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (first_value - first_mean) * (second_value - second_mean)
        for first_value, second_value in zip(first, second, strict=True)
    )
    denominator = sqrt(
        sum((value - first_mean) ** 2 for value in first)
        * sum((value - second_mean) ** 2 for value in second)
    )
    if denominator == 0.0:
        return None
    return numerator / denominator


def _regime_claim(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize actual iteration work by the decision-procedure verdict."""
    by_verdict: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_verdict.setdefault(record["verdict"], []).append(record)

    iteration_by_verdict = {
        verdict: _summary([float(row["actual_gmres"]["iterations"]) for row in rows])
        for verdict, rows in sorted(by_verdict.items())
    }
    gradient_by_verdict = {
        verdict: _summary([float(row["front_max_gradient"]) for row in rows])
        for verdict, rows in sorted(by_verdict.items())
    }
    adequate = iteration_by_verdict.get("adequate", _summary([]))
    investigate = iteration_by_verdict.get("investigate", _summary([]))
    adequate_gradient = gradient_by_verdict.get("adequate", _summary([]))
    investigate_gradient = gradient_by_verdict.get("investigate", _summary([]))
    state_verdicts: dict[tuple[int, int], set[str]] = {}
    for record in records:
        key = (int(record["m"]), int(record["visited_step"]))
        state_verdicts.setdefault(key, set()).add(str(record["verdict"]))

    supports_cost_separation = (
        adequate["count"] > 0
        and investigate["count"] > 0
        and float(investigate["median"]) >= float(adequate["median"])
    )
    iteration_overlap = _ranges_overlap(adequate, investigate)
    gradient_overlap = _ranges_overlap(adequate_gradient, investigate_gradient)
    return {
        "description": "Actual counted-GMRES work grouped by decision-procedure verdict.",
        "iteration_by_verdict": iteration_by_verdict,
        "front_gradient_by_verdict": gradient_by_verdict,
        "adequate_vs_investigate": {
            "supports_cost_separation": supports_cost_separation,
            "iteration_ranges_overlap": iteration_overlap,
            "front_gradient_ranges_overlap": gradient_overlap,
            "mixed_verdict_states": sum(len(verdicts) > 1 for verdicts in state_verdicts.values()),
            "total_states": len(state_verdicts),
            "coincident_clear_jump": supports_cost_separation
            and not iteration_overlap
            and not gradient_overlap,
        },
    }


def _rank_claim(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare disk-rate and actual-iteration orderings within each visited state."""
    by_state: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (int(record["m"]), int(record["visited_step"]))
        by_state.setdefault(key, []).append(record)

    states = []
    concordant = 0
    discordant = 0
    tied = 0
    exact_order_count = 0
    for (m, visited_step), rows in sorted(by_state.items()):
        disk_order = [
            row["d0_kind"]
            for row in sorted(rows, key=lambda row: (row["disk_rate"], row["d0_kind"]))
        ]
        iteration_order = [
            row["d0_kind"]
            for row in sorted(
                rows,
                key=lambda row: (row["actual_gmres"]["iterations"], row["d0_kind"]),
            )
        ]
        state_concordant = 0
        state_discordant = 0
        state_tied = 0
        for first, second in combinations(rows, 2):
            disk_difference = float(first["disk_rate"] - second["disk_rate"])
            iteration_difference = float(
                first["actual_gmres"]["iterations"] - second["actual_gmres"]["iterations"]
            )
            product = disk_difference * iteration_difference
            if product > 0.0:
                state_concordant += 1
            elif product < 0.0:
                state_discordant += 1
            else:
                state_tied += 1
        concordant += state_concordant
        discordant += state_discordant
        tied += state_tied
        exact_order_count += disk_order == iteration_order
        states.append(
            {
                "m": m,
                "visited_step": visited_step,
                "disk_rate_order": disk_order,
                "actual_iteration_order": iteration_order,
                "concordant_pairs": state_concordant,
                "discordant_pairs": state_discordant,
                "tied_pairs": state_tied,
            }
        )
    compared = concordant + discordant
    return {
        "description": "Within-state pairwise ranking of five D0 variants, lower is better.",
        "states": states,
        "state_exact_order_agreement_fraction": exact_order_count / len(states) if states else None,
        "pairwise": {
            "concordant": concordant,
            "discordant": discordant,
            "tied": tied,
            "concordance_fraction": concordant / compared if compared else None,
        },
    }


def _predictor_quality(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare CP-envelope estimates and raw disk rates against measured work."""
    raw_pairs = [
        {
            "m": record["m"],
            "visited_step": record["visited_step"],
            "d0_kind": record["d0_kind"],
            "disk_rate": record["disk_rate"],
            "actual_iterations": record["actual_gmres"]["iterations"],
        }
        for record in records
    ]
    predictor_pairs = [
        {
            "m": record["m"],
            "visited_step": record["visited_step"],
            "d0_kind": record["d0_kind"],
            "predicted_iterations_from_envelope": record["predicted_iterations_from_envelope"],
            "actual_iterations": record["actual_gmres"]["iterations"],
        }
        for record in records
        if isfinite(float(record["predicted_iterations_from_envelope"]))
    ]
    raw_disk_rates = [float(pair["disk_rate"]) for pair in raw_pairs]
    raw_actual = [float(pair["actual_iterations"]) for pair in raw_pairs]
    predicted = [float(pair["predicted_iterations_from_envelope"]) for pair in predictor_pairs]
    predicted_actual = [float(pair["actual_iterations"]) for pair in predictor_pairs]
    return {
        "description": "Crouzeix--Palencia envelope count estimate versus measured GMRES work.",
        "raw_disk_rate_pearson": _pearson(raw_disk_rates, raw_actual),
        "envelope_predicted_pearson": _pearson(predicted, predicted_actual),
        "median_absolute_error": (
            float(
                median(
                    [
                        abs(estimate - actual)
                        for estimate, actual in zip(predicted, predicted_actual, strict=True)
                    ]
                )
            )
            if predicted
            else None
        ),
        "finite_predictor_pairs": len(predictor_pairs),
        "non_predictive_records": len(records) - len(predictor_pairs),
        "pairs": predictor_pairs,
    }


def _correlation_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return raw disk-rate/iteration pairs for direct inspection."""
    return {
        "description": "Raw disk-rate and counted-GMRES values for direct inspection.",
        "pairs": [
            {
                "m": record["m"],
                "visited_step": record["visited_step"],
                "d0_kind": record["d0_kind"],
                "disk_rate": record["disk_rate"],
                "actual_gmres_iterations": record["actual_gmres"]["iterations"],
            }
            for record in records
        ],
    }


def _record_state(
    records: list[dict[str, Any]],
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    b: float | None,
    visited_step: int,
    config: BreakdownConfig,
    centering: dict[str, Any],
) -> None:
    """Measure every D0 variant at one solver-visited state."""
    epsilon = 0.0 if m == 1 else config.epsilon
    visited_interior = interior_values(state, grid)
    comparison = _exact_solution(grid, m, config.t0 + (visited_step + 1) * config.dt, b)
    front_gradient = float(jnp.max(jnp.abs(jnp.diff(visited_interior))) / grid.dx)
    for index, d0_kind in enumerate(config.d0_kinds):
        diagnostics = assess_pme_state(
            state,
            grid,
            float(m),
            config.dt,
            epsilon,
            d0_kind,
            const_value=config.const_d0,
            n_angles=config.n_angles,
            fov_max_iters=config.fov_max_iters,
            arnoldi_steps=config.arnoldi_steps,
            seed=20260900 + 1000 * m + 10 * visited_step + index,
        )
        actual_gmres = measure_gmres_iterations(
            state,
            grid,
            float(m),
            config.dt,
            epsilon,
            d0_kind,
            tol=config.krylov_tol,
            max_iters=config.max_krylov_iters,
            const_value=config.const_d0,
        )
        next_state, solve = _solve_one_step(state, grid, m, config, d0_kind)
        numerical = interior_values(next_state, grid)
        records.append(
            {
                "m": m,
                "visited_step": visited_step,
                "visited_time": config.t0 + visited_step * config.dt,
                "front_max_gradient": front_gradient,
                "d0_kind": d0_kind,
                "d0_used": diagnostics["d0"],
                "sigma": float(diagnostics["d0"] * config.dt / grid.dx**2),
                "adjoint_identity": diagnostics["adjoint_error"],
                "adjoint_tolerance": diagnostics["adjoint_tolerance"],
                "verdict": diagnostics["verdict"],
                "disk_rate": diagnostics["disk_rate"],
                "predicted_iterations_from_envelope": predicted_iterations_from_envelope(
                    diagnostics["disk_rate"], tol=config.krylov_tol
                ),
                "epsilon_zero": diagnostics["epsilon_zero"],
                "predicted_gmres_factor": diagnostics["predicted_gmres_factor"],
                "origin_enclosed": diagnostics["origin_enclosed"],
                "n_right_real_outliers": diagnostics["n_right_real_outliers"],
                "rates": diagnostics["rates"],
                "actual_gmres": actual_gmres,
                "solver": solve,
                "centering_mismatch_note": centering["decision"],
                "exact_linf_error": float(jnp.max(jnp.abs(numerical - comparison))),
            }
        )


def run_breakdown_study(config: BreakdownConfig | None = None) -> dict[str, Any]:
    """Run the powered D0-variant sweep and write the decision-grade JSON report."""
    if config is None:
        config = BreakdownConfig()
    if not config.visited_steps or min(config.visited_steps) < 1:
        raise ValueError("visited_steps must contain positive step indices")

    started_at = perf_counter()
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
        centering = _centering_report(config)
        records: list[dict[str, Any]] = []
        selected_steps = set(config.visited_steps)
        for m in config.m_values:
            state, b = _initial_state(grid, m, config.t0)
            for visited_step in range(1, max(selected_steps) + 1):
                state, _ = _solve_one_step(state, grid, m, config, "frozen_bulk")
                if visited_step in selected_steps:
                    _record_state(records, state, grid, m, b, visited_step, config, centering)

    runtime_seconds = perf_counter() - started_at
    report = {
        "description": (
            "Experimental PME conditioning study. It evaluates a preconditioner decision "
            "procedure; it does not claim to fix stiffness degradation."
        ),
        "config": config._asdict(),
        "metric": {"accuracy": "L_inf", "timing": "total runtime only"},
        "runtime_seconds": runtime_seconds,
        "gmres_measurement_note": (
            "actual_gmres is an explicit residual-history count for the fixed system "
            "P^-1 J delta = P^-1 (-R), not NKStats.lin_iters."
        ),
        "centering": centering,
        "regime_claim": _regime_claim(records),
        "rank_claim": _rank_claim(records),
        "predictor_quality": _predictor_quality(records),
        "correlation": _correlation_pairs(records),
        "records": records,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a short table separating regime, rank, and predictor evidence."""
    regime = report["regime_claim"]
    rank = report["rank_claim"]
    predictor = report["predictor_quality"]
    print(f"records={len(report['records'])} runtime_seconds={report['runtime_seconds']:.3f}")
    print("verdict       count  min  median  max")
    for verdict, values in regime["iteration_by_verdict"].items():
        print(
            f"{verdict:12} {values['count']:5d} {values['min']:4.0f} "
            f"{values['median']:7.2f} {values['max']:4.0f}"
        )
    pairwise = rank["pairwise"]
    print(
        "rank pairwise: "
        f"concordant={pairwise['concordant']} discordant={pairwise['discordant']} "
        f"tied={pairwise['tied']} fraction={pairwise['concordance_fraction']:.3f}"
    )
    print(
        "predictor: "
        f"raw_pearson={predictor['raw_disk_rate_pearson']:.3f} "
        f"envelope_pearson={predictor['envelope_predicted_pearson']:.3f} "
        f"median_absolute_error={predictor['median_absolute_error']:.3f}"
    )


def main() -> None:
    """Run the default study or write to an explicitly supplied JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=BreakdownConfig().output_path)
    args = parser.parse_args()
    result = run_breakdown_study(BreakdownConfig(output_path=args.output))
    _print_summary(result)


if __name__ == "__main__":
    main()
