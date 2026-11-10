#!/usr/bin/env python3
"""Falsifiable numerical checks of tau-blend mask Lemmas 2, 3, 4, and 6."""

from __future__ import annotations

import argparse
import json
import math
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
    KINDS,
    CflFidelityConfig,
    make_case,
    newton_step,
    preconditioner,
    residual,
)
from benchmarks.tau_blend_mask_analysis import (
    MaskAnalysisConfig,
    build_state,
    cfl_mask,
    jacobian,
    local_cfl,
    one_inverse_matrix,
)
from moljax.experimental.tau_blend_active_block import (
    extract_active_block,
    lemma6_tail_lower_bound,
)

DEFAULT_OUTPUT = Path("benchmarks/results/tau_blend_lemma_checks.json")


def check_lemma6() -> list[dict[str, Any]]:
    """Compare Lemma 6's lower bound with every delivered scalar tail."""
    committed = json.loads(Path("benchmarks/results/tau_blend_mask_analysis.json").read_text())
    measured = {
        (row["nx"], row["m"], row["case"]): row
        for row in committed["cases"]
        if row["family"] == "scalar"
    }
    config = MaskAnalysisConfig()
    rows = []
    for nx in (512, 1024):
        for m in (2, 4, 8):
            state = build_state(nx, m, config.dt, config)
            diffusivity = state["D"]
            floor = float(diffusivity.min())
            rank = int(np.count_nonzero(diffusivity > floor))
            for case in ("scalar_max", "scalar_median", "scalar_geomean"):
                record = measured[(nx, m, case)]
                bound = lemma6_tail_lower_bound(
                    -state["lam"],
                    diffusivity_floor=floor,
                    scalar_reference=float(record["references"][0]),
                    dt=config.dt,
                    magnitude_threshold=0.1,
                    rank_correction=rank,
                )
                observed = int(record["eigenvalues_below_0p1"])
                rows.append(
                    {
                        "nx": nx,
                        "m": m,
                        "case": case,
                        "diffusivity_floor": floor,
                        "reference": float(record["references"][0]),
                        "rank_correction": rank,
                        "modal_threshold": bound.modal_threshold,
                        "modes_above_threshold": bound.modes_above_threshold,
                        "lower_bound": bound.lower_bound,
                        "measured_below_0p1": observed,
                        "margin": observed - bound.lower_bound,
                        "passes": observed >= bound.lower_bound,
                    }
                )
    return rows


def check_lemma2_and_identity_block() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check perturbation norms and the regularized identity-block deviation."""
    config = CflFidelityConfig()
    lemma_rows = []
    identity_rows = []
    for m in config.m:
        case = make_case(config.nx, m, config)
        diffusivity = case["D"]
        for dt in config.e1_dts:
            operator = jacobian(case["L"], diffusivity, dt)
            chi = local_cfl(case, diffusivity, dt)
            for eta in config.e1_etas:
                masked = chi <= eta
                zeroed = jacobian(case["L"], np.where(masked, 0.0, diffusivity), dt)
                difference = operator - zeroed
                eta_z = float(chi[masked].max()) if np.any(masked) else 0.0
                norm1 = float(np.linalg.norm(difference, 1))
                norminf = float(np.linalg.norm(difference, np.inf))
                allowance = 2.0e-12 * max(1.0, eta_z)
                lemma_rows.append(
                    {
                        "nx": config.nx,
                        "m": m,
                        "dt": dt,
                        "eta": eta,
                        "masked": int(masked.sum()),
                        "eta_Z": eta_z,
                        "norm_1": norm1,
                        "norm_inf": norminf,
                        "margin_1": eta_z - norm1,
                        "margin_inf": eta_z - norminf,
                        "passes": norm1 <= eta_z + allowance and norminf <= eta_z + allowance,
                    }
                )
        eta = 0.3 if m == 2 else 0.1
        mask = cfl_mask(case, diffusivity, eta, 0.02)
        preconditioner_matrix = one_inverse_matrix(case, diffusivity, mask, 0.02)
        blocks = extract_active_block(preconditioner_matrix @ case["J"], mask)
        floor = float(diffusivity.min())
        measured = float(np.max(np.abs(blocks.E))) if blocks.E.size else 0.0
        expected = 2.0 * 0.02 * floor / case["dx"] ** 2
        identity_rows.append(
            {
                "nx": config.nx,
                "m": m,
                "eta": eta,
                "diffusivity_floor": floor,
                "max_abs_identity_block_deviation": measured,
                "expected_2_dt_floor_over_h2": expected,
                "absolute_difference": abs(measured - expected),
            }
        )
    return lemma_rows, identity_rows


def check_lemma4() -> list[dict[str, Any]]:
    """Measure sigma_min and kappa_2 of every delivered preconditioner."""
    config = CflFidelityConfig()
    rows = []
    for m in config.m:
        case = make_case(config.nx, m, config)
        for eta in ((0.1, 0.3) if m == 2 else (0.1,)):
            for kind in KINDS:
                matrix = preconditioner(case, case["D"], 0.02, kind, eta)
                singular = np.linalg.svd(matrix, compute_uv=False)
                rows.append(
                    {
                        "nx": config.nx,
                        "m": m,
                        "dt": 0.02,
                        "eta": eta if kind in ("Z1", "Zf") else None,
                        "kind": kind,
                        "sigma_min_M": float(singular[-1]),
                        "sigma_max_M": float(singular[0]),
                        "kappa2_M": float(singular[0] / singular[-1]),
                    }
                )
    return rows


def _reference_trajectory(case: dict[str, Any], dt: float, config: CflFidelityConfig):
    states = [case["u"].copy()]
    residuals = []
    for _ in range(config.e3_steps):
        previous = states[-1]
        atol = 64.0 * np.finfo(float).eps * (1.0 + float(np.linalg.norm(previous)))
        state, info = newton_step(
            case,
            previous,
            dt,
            "direct",
            math.nan,
            0.0,
            "",
            config,
            nl_tol=config.e3_reference_tol,
            max_newton=config.e3_reference_max_newton,
            atol=atol,
        )
        states.append(state)
        residuals.append(float(np.linalg.norm(residual(case, state, previous, dt), 1)))
        if not info["converged"] or info["line_search_failed"]:
            raise RuntimeError(f"reference failed for m={case['m']} dt={dt}: {info}")
    return states, residuals


def check_lemma3() -> dict[str, Any]:
    """Check the true-residual one-norm a posteriori bound, including failures."""
    config = CflFidelityConfig()
    rows = []
    for m in config.m:
        case = make_case(config.nx, m, config)
        for dt in config.e3_dts:
            exact, reference_residuals = _reference_trajectory(case, dt, config)
            for side in config.e3_sides:
                for forcing in config.e3_forcings:
                    for kind in KINDS:
                        eta = config.e3_eta if kind in ("Z1", "Zf") else math.nan
                        state = case["u"].copy()
                        initial_difference = float(np.linalg.norm(state - exact[0], 1))
                        cumulative = 0.0
                        steps = []
                        for step in range(config.e3_steps):
                            previous = state.copy()
                            state, info = newton_step(
                                case,
                                previous,
                                dt,
                                kind,
                                eta,
                                forcing,
                                side,
                                config,
                                nl_tol=config.e3_newton_tol,
                                max_newton=config.e3_max_newton,
                            )
                            rho1 = float(np.linalg.norm(residual(case, state, previous, dt), 1))
                            cumulative += rho1
                            bound = initial_difference + cumulative
                            lhs = float(np.linalg.norm(state - exact[step + 1], 1))
                            allowance = 5.0e-9 * max(1.0, lhs, bound)
                            steps.append(
                                {
                                    "step": step + 1,
                                    "lhs_error_1": lhs,
                                    "rho_1": rho1,
                                    "initial_difference_1": initial_difference,
                                    "cumulative_rho_1": cumulative,
                                    "bound_rhs_1": bound,
                                    "margin": bound - lhs,
                                    "reference_residual_1": reference_residuals[step],
                                    "solver_converged": info["converged"],
                                    "line_search_failed": info["line_search_failed"],
                                    "passes": lhs <= bound + allowance,
                                }
                            )
                        rows.append(
                            {
                                "nx": config.nx,
                                "m": m,
                                "dt": dt,
                                "kind": kind,
                                "side": side,
                                "forcing": forcing,
                                "failed_steps": sum(
                                    not row["solver_converged"] or row["line_search_failed"]
                                    for row in steps
                                ),
                                "minimum_margin": min(row["margin"] for row in steps),
                                "passes": all(row["passes"] for row in steps),
                                "steps": steps,
                            }
                        )
    return {
        "rows": rows,
        "violations": sum(not row["passes"] for row in rows),
        "failed_step_trajectories": sum(row["failed_steps"] > 0 for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = perf_counter()
    lemma2, identity = check_lemma2_and_identity_block()
    payload = {
        "schema": "tau_blend_lemma_checks_v1",
        "lemma2": lemma2,
        "lemma3": check_lemma3(),
        "lemma4": check_lemma4(),
        "lemma6": check_lemma6(),
        "identity_block_deviation": identity,
    }
    payload["provenance"] = benchmark_provenance(__file__, perf_counter() - started)
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
