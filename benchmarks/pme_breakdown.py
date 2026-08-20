"""Conditioning breakdown study for experimental porous-medium backward Euler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
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
)
from moljax.experimental.pme_preconditioner import helmholtz_inverse_relative_residual


class BreakdownConfig(NamedTuple):
    """Static parameters for the one-dimensional PME conditioning sweep.

    The study starts from an early-time compact profile and takes only three
    short backward-Euler steps. PME smooths rather than sharpens in forward
    time, so this schedule preserves an initially steep front while
    ``dt / dx**2`` remains order one for the linear control.
    """

    nx: int = 48
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    dt: float = 0.02
    epsilon: float = 1.0e-5
    visited_steps: int = 3
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


def _correlation_pairs(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return paired diagnostic and observed iteration values without extra dependencies."""
    pairs = [
        {
            "m": record["m"],
            "d0_kind": record["d0_kind"],
            "disk_rate": record["disk_rate"],
            "actual_gmres_iterations": record["actual_gmres"]["iterations"],
        }
        for record in records
    ]
    return {
        "description": "Paired disk-rate and counted-GMRES values for direct inspection.",
        "pairs": pairs,
    }


def run_breakdown_study(config: BreakdownConfig | None = None) -> dict[str, Any]:
    """Run the D0-variant sweep and write its JSON report.

    The comparison metric is the actual count from a matrix-free GMRES
    residual history. No timing is reported; any future timing extension must
    use warmup, synchronization, and a 50--100 sample median/IQR protocol.
    """
    if config is None:
        config = BreakdownConfig()

    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
        centering = _centering_report(config)
        records: list[dict[str, Any]] = []
        for m in config.m_values:
            state, b = _initial_state(grid, m, config.t0)
            for _ in range(config.visited_steps):
                state, _ = _solve_one_step(state, grid, m, config, "frozen_bulk")

            epsilon = 0.0 if m == 1 else config.epsilon
            visited_interior = interior_values(state, grid)
            comparison = _exact_solution(
                grid,
                m,
                config.t0 + (config.visited_steps + 1) * config.dt,
                b,
            )
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
                    seed=20260900 + 100 * m + index,
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
                        "d0_kind": d0_kind,
                        "visited_time": config.t0 + config.visited_steps * config.dt,
                        "front_max_gradient": front_gradient,
                        "d0_used": diagnostics["d0"],
                        "sigma": float(diagnostics["d0"] * config.dt / grid.dx**2),
                        "adjoint_identity": diagnostics["adjoint_error"],
                        "adjoint_tolerance": diagnostics["adjoint_tolerance"],
                        "verdict": diagnostics["verdict"],
                        "disk_rate": diagnostics["disk_rate"],
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

    report = {
        "description": (
            "Experimental PME conditioning breakdown study. It detects "
            "preconditioner adequacy or breakdown; it does not claim to fix stiffness degradation."
        ),
        "config": config._asdict(),
        "metric": {"accuracy": "L_inf", "timing": "not measured"},
        "gmres_measurement_note": (
            "actual_gmres is an explicit residual-history count for the fixed system "
            "P^-1 J delta = P^-1 (-R), not NKStats.lin_iters."
        ),
        "centering": centering,
        "correlation": _correlation_pairs(records),
        "records": records,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the default study or write to an explicitly supplied JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=BreakdownConfig().output_path)
    args = parser.parse_args()
    result = run_breakdown_study(BreakdownConfig(output_path=args.output))
    for record in result["records"]:
        actual = record["actual_gmres"]
        print(
            "m={m} d0={d0_kind} verdict={verdict} disk_rate={disk_rate:.6g} "
            "epsilon_zero={epsilon_zero:.6g} gmres={iterations} residual={residual:.3e}".format(
                **record,
                iterations=actual["iterations"],
                residual=actual["final_relative_residual"],
            )
        )


if __name__ == "__main__":
    main()
