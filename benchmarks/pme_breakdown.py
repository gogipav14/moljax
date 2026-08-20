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
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_conditioning import (
    assess_pme_state,
    interior_values,
    make_backward_euler_residual,
    padded_values,
    pme_preconditioner_variant,
)
from moljax.experimental.pme_preconditioner import d0_frozen_mean


class BreakdownConfig(NamedTuple):
    """Static parameters for the one-dimensional PME conditioning sweep."""

    nx: int = 64
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 1.0
    dt: float = 0.01
    epsilon: float = 1.0e-5
    visited_steps: int = 2
    n_angles: int = 6
    fov_max_iters: int = 30
    arnoldi_steps: int = 8
    const_d0: float = 1.0
    max_newton_iters: int = 5
    max_krylov_iters: int = 16
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-7
    output_path: str = "benchmarks/results/pme_breakdown.json"


_M_VALUES = (1, 2, 3, 4, 5, 6)
_D0_KINDS = ("frozen_mean", "floor", "const", "identity")


def _heat_gaussian(x: jax.Array, t: float) -> jax.Array:
    """Return the linear-control heat kernel, negligible at the domain faces."""
    return jnp.exp(-(x**2) / (4.0 * t)) / jnp.sqrt(4.0 * jnp.pi * t)


def _barenblatt_b(m: int, target_radius: float = 1.8) -> float:
    """Choose ``b`` so the unit-time Barenblatt support has ``target_radius``."""
    beta = 1.0 / (m - 1.0 + 2.0)
    return target_radius**2 * (m - 1.0) * beta / (2.0 * m)


def _initial_state(grid: Grid1D, m: int, t0: float) -> tuple[jax.Array, float | None]:
    """Return a padded control or Barenblatt initial state and its ``b`` value."""
    x = grid.x_coords()
    if m == 1:
        return padded_values(_heat_gaussian(x, t0), grid), None
    b = _barenblatt_b(m)
    return padded_values(barenblatt(x, t0, float(m), b=b), grid), b


def _exact_solution(grid: Grid1D, m: int, time: float, b: float | None) -> jax.Array:
    """Return the unregularized analytic comparison profile on the interior."""
    x = grid.x_coords()
    if m == 1:
        return _heat_gaussian(x, time)
    if b is None:
        raise ValueError("Barenblatt comparison requires b")
    return barenblatt(x, time, float(m), b=b)


def _solve_one_step(
    state: jax.Array,
    grid: Grid1D,
    m: int,
    config: BreakdownConfig,
    d0_kind: str,
) -> tuple[jax.Array, dict[str, Any]]:
    """Take one BE solve and return its padded state and solver statistics."""
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
        "gmres_lin_iters": int(result.stats.lin_iters),
        "final_residual_l2": float(result.stats.final_res_norm),
    }


def run_breakdown_study(config: BreakdownConfig | None = None) -> dict[str, Any]:
    """Run the D0-variant sweep and write its JSON report.

    The sweep deliberately reports no timings: its comparison metric is the
    solver-reported Krylov count plus conditioning geometry.  Any future timing
    extension must use warmup, synchronization, and a 50--100 sample median/IQR
    protocol.
    """
    if config is None:
        config = BreakdownConfig()

    with jax.enable_x64(True):
        grid = Grid1D.uniform(config.nx, config.x_min, config.x_max)
        records: list[dict[str, Any]] = []
        for m in _M_VALUES:
            state, b = _initial_state(grid, m, config.t0)
            for _ in range(config.visited_steps):
                state, _ = _solve_one_step(state, grid, m, config, "frozen_mean")

            epsilon = 0.0 if m == 1 else config.epsilon
            visited_interior = interior_values(state, grid)
            frozen_d0 = d0_frozen_mean(visited_interior, float(m))
            sigma = frozen_d0 * config.dt / grid.dx**2
            comparison = _exact_solution(
                grid,
                m,
                config.t0 + config.visited_steps * config.dt + config.dt,
                b,
            )

            for index, d0_kind in enumerate(_D0_KINDS):
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
                next_state, solve = _solve_one_step(state, grid, m, config, d0_kind)
                numerical = interior_values(next_state, grid)
                records.append(
                    {
                        "m": m,
                        "d0_kind": d0_kind,
                        "sigma_frozen_mean": float(sigma),
                        "visited_time": config.t0 + config.visited_steps * config.dt,
                        "d0_used": diagnostics["d0"],
                        "adjoint_identity": diagnostics["adjoint_error"],
                        "adjoint_tolerance": diagnostics["adjoint_tolerance"],
                        "verdict": diagnostics["verdict"],
                        "disk_rate": diagnostics["disk_rate"],
                        "epsilon_zero": diagnostics["epsilon_zero"],
                        "predicted_gmres_factor": diagnostics["predicted_gmres_factor"],
                        "origin_enclosed": diagnostics["origin_enclosed"],
                        "n_right_real_outliers": diagnostics["n_right_real_outliers"],
                        "rates": diagnostics["rates"],
                        "solver": solve,
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
        "solver_stat_note": (
            "gmres_lin_iters is NKStats.lin_iters from newton_krylov_solve; the current "
            "solver reports its configured Krylov budget per Newton solve rather than a callback count."
        ),
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
        print(
            "m={m} d0={d0_kind} verdict={verdict} disk_rate={disk_rate:.6g} "
            "epsilon_zero={epsilon_zero:.6g} gmres_lin_iters={solver[gmres_lin_iters]}".format(
                **record
            )
        )


if __name__ == "__main__":
    main()
