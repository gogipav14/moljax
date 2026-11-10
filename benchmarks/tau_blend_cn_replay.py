#!/usr/bin/env python3
"""Replay the delivered PME trajectory with CN beside BE."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from benchmarks.tau_blend_benchmark_support import benchmark_provenance, write_json
from benchmarks.tau_blend_cfl_fidelity import (
    CflFidelityConfig,
    cfl_mask,
    gmres,
    jacobian,
    make_case,
    preconditioner,
)


def _residual(
    case: dict[str, Any],
    method: str,
    state: np.ndarray,
    previous: np.ndarray,
    dt: float,
) -> np.ndarray:
    if method == "BE":
        return state - previous - dt * (case["L"] @ case["phi"](state))
    if method == "CN":
        return (
            state - previous - 0.5 * dt * (case["L"] @ (case["phi"](state) + case["phi"](previous)))
        )
    raise ValueError(f"unknown method {method!r}")


def _step(
    case: dict[str, Any],
    method: str,
    previous: np.ndarray,
    dt: float,
    eta: float,
    config: CflFidelityConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    effective_dt = dt if method == "BE" else 0.5 * dt
    state = previous.copy()
    current_residual = _residual(case, method, state, previous, dt)
    initial_norm = float(np.linalg.norm(current_residual))
    target = config.e3_newton_tol * initial_norm
    newton_iterations = 0
    linear_iterations = 0
    line_search_failed = False
    while np.linalg.norm(current_residual) > target and newton_iterations < config.e3_max_newton:
        diffusivity = case["Dfun"](state)
        operator = jacobian(case["L"], diffusivity, effective_dt)
        preconditioner_matrix = preconditioner(
            case,
            diffusivity,
            effective_dt,
            "Z1",
            eta,
        )
        solve = gmres(
            operator @ preconditioner_matrix,
            -current_residual,
            1.0e-4,
            config.e3_linear_max,
        )
        update = preconditioner_matrix @ solve["x"]
        linear_iterations += solve["iterations"]
        newton_iterations += 1
        residual_norm = float(np.linalg.norm(current_residual))
        step_length = 1.0
        accepted = None
        for _ in range(config.e3_backtracks):
            trial = state + step_length * update
            trial_residual = _residual(case, method, trial, previous, dt)
            if (
                np.linalg.norm(trial_residual)
                <= (1.0 - config.e3_armijo * step_length) * residual_norm
            ):
                accepted = (trial, trial_residual)
                break
            step_length *= 0.5
        if accepted is None:
            line_search_failed = True
            break
        state, current_residual = accepted
    diffusivity = case["Dfun"](state)
    active_operator = ~cfl_mask(case, diffusivity, eta, effective_dt)
    active_physical_dt = ~cfl_mask(case, diffusivity, eta, dt)
    return state, {
        "converged": bool(np.linalg.norm(current_residual) <= target),
        "line_search_failed": line_search_failed,
        "newton_iterations": newton_iterations,
        "gmres_iterations": linear_iterations,
        "initial_residual": initial_norm,
        "final_residual": float(np.linalg.norm(current_residual)),
        "min_u": float(state.min()),
        "negative_undershoot": float(min(0.0, state.min())),
        "active_operator": active_operator,
        "active_physical_dt": active_physical_dt,
        "effective_jacobian_dt": effective_dt,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--m", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/tau_blend_cn_replay.json"),
    )
    args = parser.parse_args()
    started = perf_counter()
    config = CflFidelityConfig(
        nx=args.nx,
        m=tuple(args.m),
        e3_dts=(args.dt,),
        e3_steps=args.steps,
    )
    rows = []
    summaries = []
    for m in config.m:
        eta = 0.3 if m == 2 else 0.1
        case = make_case(config.nx, m, config)
        histories = {}
        for method in ("BE", "CN"):
            state = case["u"].copy()
            history = []
            for _ in range(config.e3_steps):
                state, info = _step(case, method, state, args.dt, eta, config)
                history.append((state.copy(), info))
            histories[method] = history
        for step, ((_, be), (_, cn)) in enumerate(
            zip(histories["BE"], histories["CN"], strict=True),
            start=1,
        ):
            rows.append(
                {
                    "nx": config.nx,
                    "m": m,
                    "dt": args.dt,
                    "eta": eta,
                    "step": step,
                    "BE": {
                        key: value
                        for key, value in be.items()
                        if key not in ("active_operator", "active_physical_dt")
                    },
                    "CN": {
                        key: value
                        for key, value in cn.items()
                        if key not in ("active_operator", "active_physical_dt")
                    },
                    "active_size_BE_operator_mask": int(be["active_operator"].sum()),
                    "active_size_CN_operator_mask": int(cn["active_operator"].sum()),
                    "operator_mask_symmetric_difference": int(
                        np.count_nonzero(be["active_operator"] ^ cn["active_operator"])
                    ),
                    "active_size_BE_same_physical_dt_mask": int(be["active_physical_dt"].sum()),
                    "active_size_CN_same_physical_dt_mask": int(cn["active_physical_dt"].sum()),
                    "physical_dt_mask_symmetric_difference": int(
                        np.count_nonzero(be["active_physical_dt"] ^ cn["active_physical_dt"])
                    ),
                }
            )
        summary = {"nx": config.nx, "m": m, "dt": args.dt, "eta": eta}
        for method in ("BE", "CN"):
            summary[method] = {
                "newton_total": sum(row[1]["newton_iterations"] for row in histories[method]),
                "gmres_total": sum(row[1]["gmres_iterations"] for row in histories[method]),
                "all_steps_converged": all(row[1]["converged"] for row in histories[method]),
                "minimum_u": min(row[1]["min_u"] for row in histories[method]),
            }
        summaries.append(summary)
    payload = {
        "schema": "tau_blend_cn_replay_v1",
        "protocol": {
            "preconditioner": "Z1, right-preconditioned",
            "linear_forcing": 1.0e-4,
            "steps": config.e3_steps,
            "mask_note": (
                "operator masks use BE dt and CN dt/2, matching their Jacobians; "
                "physical-dt masks are also reported for a same-dt comparison."
            ),
        },
        "summaries": summaries,
        "steps": rows,
        "provenance": benchmark_provenance(__file__, perf_counter() - started),
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
