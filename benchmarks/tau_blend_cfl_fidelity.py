#!/usr/bin/env python3
"""CFL analysis of the identity mask in the DST/tau blend, and solution fidelity.

This script is the committed form of the private study "A10".  It uses the
dense operators of ``benchmarks/tau_blend_mask_analysis.py``.

The mask sets the preconditioner to the identity on node j when the local
diffusive CFL number ``chi_j = 4 dt D_j / h^2`` is at or below ``eta``.  The
global CFL number is ``c = 4 dt Dmax / h^2``.

Preconditioners (each one is built at the current Newton iterate):

- ``S``:  H(Dmax), the scalar inverse, no mask.
- ``Z1``: Q + H(Dmax) (I - Q), the mask and one inverse.
- ``Zf``: Q + H(Dmax) diag(Dmax / D) (I - Q), the mask, one inverse, the frame.
- ``B3``: the repository blend (three quantile references, Gaussian log
  weights, the repository threshold).

Here Q is the diagonal mask and Dmax is the largest unmasked coefficient.

Parts:

- ``e1``: GMRES iterations against c for S, B3, Z1 and Zf, with the time
  step from 2e-7 to 2.  The same loop gives E2, the cutoff sweep over eta,
  with the masked-node count, ||J - J0||_2 and ||M (J - J0)||_2.  Here J0 is
  J with D set to zero on the masked nodes.  The same loop gives E4: the
  ratio of the true relative residual to the preconditioned relative residual
  when the preconditioned residual first reaches each tolerance.
- ``floor``: the N = 2048 floor case (m = 2, dt = 0.02, eta = 0.1, 0.3, 1).
- ``e3``: five backward-Euler steps by globalized inexact Newton, compared
  with the trajectory of exact (direct) Newton solves.

Every GMRES count is given for two residual conventions:
"preconditioned", ||M (b - J x)|| <= tol ||M b||, and "true",
||b - J x|| <= tol ||b||.  The E3 linear solves use left preconditioning
(the forcing applies to ||M r||) or right preconditioning (the forcing
applies to ||r||).

CAUTION: A full run takes hours on one core.  Use ``--parts`` and small
values of ``--nx`` for a check.

Usage::

    JAX_PLATFORMS=cpu python benchmarks/tau_blend_cfl_fidelity.py
    JAX_PLATFORMS=cpu python benchmarks/tau_blend_cfl_fidelity.py --parts e1 \
        --nx 64 --m 4 --output /tmp/check.json
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from benchmarks.tau_blend_mask_analysis import (
    MaskAnalysisConfig,
    blend_matrix,
    build_state,
    cfl_mask,
    floor_cfl,
    frame_matrix,
    gaussian_log_weights,
    gmres,
    gmres_two_conventions,
    helmholtz,
    jacobian,
    local_cfl,
    one_inverse_matrix,
    provenance,
    quantile_references,
    repository_mask,
    write_outputs,
)
from moljax.experimental.nonlinear_diffusion import (
    porous_medium_diffusivity,
    regularized_porous_medium_potential,
)

SCHEMA = "tau_blend_cfl_fidelity_v1"
KINDS = ("S", "B3", "Z1", "Zf")


@dataclass(frozen=True)
class CflFidelityConfig:
    """The full configuration of the CFL and fidelity study."""

    parts: tuple[str, ...] = ("e1", "floor", "e3")
    nx: int = 512
    m: tuple[int, ...] = (2, 4, 8)
    epsilon: float = 1.0e-5
    t0: float = 0.1
    halfwidth: float = 3.0
    x_min: float = -4.0
    x_max: float = 4.0
    gmres_tol: float = 1.0e-8
    e1_dts: tuple[float, ...] = (2.0e-7, 2.0e-6, 2.0e-5, 2.0e-4, 2.0e-3, 2.0e-2, 2.0e-1, 2.0)
    e1_etas: tuple[float, ...] = (1.0e-3, 1.0e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    e1_gmres_max: int = 600
    e4_tolerances: tuple[float, ...] = (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-8)
    floor_nx: int = 2048
    floor_m: int = 2
    floor_dt: float = 0.02
    floor_etas: tuple[float, ...] = (0.1, 0.3, 1.0)
    floor_gmres_max: int = 300
    e3_dts: tuple[float, ...] = (2.0e-3, 2.0e-2, 2.0e-1)
    e3_steps: int = 5
    e3_eta: float = 0.1
    e3_sides: tuple[str, ...] = ("left", "right")
    e3_forcings: tuple[float, ...] = (1.0e-1, 1.0e-2, 1.0e-4)
    e3_linear_max: int = 600
    e3_newton_tol: float = 1.0e-8
    e3_max_newton: int = 60
    e3_reference_tol: float = 1.0e-10
    e3_reference_max_newton: int = 200
    e3_armijo: float = 1.0e-4
    e3_backtracks: int = 30
    e3_front_level: float = 1.0e-3
    output: str = "benchmarks/results/tau_blend_cfl_fidelity.json"


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------


def make_case(nx: int, m: int, config: CflFidelityConfig) -> dict[str, Any]:
    """Return the dense model of the wide Barenblatt state and its potential."""
    state = build_state(nx, m, 0.02, config)
    eps = config.epsilon
    potential = jax.jit(lambda u: regularized_porous_medium_potential(u, float(m), epsilon=eps))
    diffusivity = jax.jit(lambda u: porous_medium_diffusivity(u, float(m), epsilon=eps))
    state["phi"] = lambda u: np.asarray(potential(jnp.asarray(u)))
    state["Dfun"] = lambda u: np.asarray(diffusivity(jnp.asarray(u)))
    state["x"] = np.asarray(state["grid"].x_coords())
    return state


def preconditioner(case: dict[str, Any], D: np.ndarray, dt: float, kind: str, eta: float) -> np.ndarray:
    """Return the dense preconditioner of one kind at the coefficient D."""
    if kind == "S":
        return helmholtz(case, float(D.max()), dt)
    if kind == "B3":
        masked = repository_mask(D)
        refs = quantile_references(D, masked, 3)
        return blend_matrix(case, masked, refs, gaussian_log_weights(D, refs, ~masked), dt)
    masked = cfl_mask(case, D, eta, dt)
    if kind == "Z1":
        return one_inverse_matrix(case, D, masked, dt)
    if kind == "Zf":
        return frame_matrix(case, D, masked, dt)
    raise ValueError(f"unknown preconditioner kind {kind!r}")


def residual(case: dict[str, Any], u: np.ndarray, u_prev: np.ndarray, dt: float) -> np.ndarray:
    """Return the backward-Euler residual R(u) = u - u_prev - dt L phi(u)."""
    return u - u_prev - dt * (case["L"] @ case["phi"](u))


# ---------------------------------------------------------------------------
# E1, E2 and E4
# ---------------------------------------------------------------------------


def _linear_row(case, dt, kind, eta, M, J, b, D, config, maxit) -> dict[str, Any]:
    chi = local_cfl(case, D, dt)
    counts = gmres_two_conventions(M, J, b, config.gmres_tol, maxit, checkpoints=config.e4_tolerances)
    masked = chi <= eta if kind in ("Z1", "Zf") else np.zeros(case["nx"], dtype=bool)
    row: dict[str, Any] = {
        "experiment": "E1E2",
        "nx": case["nx"],
        "m": case["m"],
        "dt": dt,
        "global_cfl": float(chi.max()),
        "floor_cfl": floor_cfl(case["m"], config.epsilon, dt, case["dx"]),
        "kind": kind,
        "eta": eta if kind in ("Z1", "Zf") else None,
        "masked": int(np.count_nonzero(masked)),
        "norm_b": float(np.linalg.norm(b)),
        "norm_Mb": float(np.linalg.norm(M @ b)),
    }
    row.update({k: v for k, v in counts.items() if k != "checkpoints"})
    singular = np.linalg.svd(M @ J, compute_uv=False)
    row["kappa2"] = float(singular[0] / singular[-1])
    if np.any(masked):
        J0 = jacobian(case["L"], np.where(masked, 0.0, D), dt)
        row["J_minus_J0_norm"] = float(np.linalg.norm(J - J0, 2))
        row["mask_perturbation"] = float(np.linalg.norm(M @ (J - J0), 2))
    else:
        row["J_minus_J0_norm"] = 0.0
        row["mask_perturbation"] = 0.0
    if kind == "Zf" and not np.all(masked):
        active = D[~masked]
        row["frame_gain_max"] = float(active.max() / active.min())
    e4 = []
    for level, point in counts["checkpoints"].items():
        prec = point["primary_ratio"]
        true = point["secondary_ratio"]
        e4.append(
            {
                "tolerance": float(level),
                "iterations": point["iterations"],
                "preconditioned_ratio": prec,
                "true_ratio": true,
                "true_over_preconditioned": true / prec if prec > 0.0 else None,
                "absolute_ratio_r_over_Mr": (true * row["norm_b"]) / (prec * row["norm_Mb"])
                if prec > 0.0
                else None,
            }
        )
    row["e4"] = e4
    return row


def run_e1(case: dict[str, Any], config: CflFidelityConfig) -> list[dict[str, Any]]:
    """Return the E1, E2 and E4 rows of one state."""
    rows = []
    u, D = case["u"], case["D"]
    phi = case["phi"](u)
    for dt in config.e1_dts:
        J = jacobian(case["L"], D, dt)
        b = dt * (case["L"] @ phi)  # -R(u) for the step from u_prev = u
        plan = [("S", math.nan), ("B3", math.nan)]
        plan += [(kind, eta) for kind in ("Z1", "Zf") for eta in config.e1_etas]
        for kind, eta in plan:
            M = preconditioner(case, D, dt, kind, eta)
            row = _linear_row(case, dt, kind, eta, M, J, b, D, config, config.e1_gmres_max)
            rows.append(row)
            print(
                f"E1 N={case['nx']} m={case['m']} dt={dt:7.1e} c={row['global_cfl']:9.3g} "
                f"{kind:2s} eta={eta:7.1e} masked={row['masked']:5d} "
                f"gmres(prec/true)={row['gmres_preconditioned']}/{row['gmres_true']} "
                f"|M(J-J0)|={row['mask_perturbation']:.2g}",
                flush=True,
            )
    return rows


def best_cutoffs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return, for each state, dt and masked kind, the cutoff with the fewest iterations.

    Ties go to the smallest cutoff.  A count that did not converge counts as
    infinite.
    """
    out = []
    keys = sorted({(r["nx"], r["m"], r["dt"], r["kind"]) for r in rows if r["kind"] in ("Z1", "Zf")})
    for nx, m, dt, kind in keys:
        group = [r for r in rows if (r["nx"], r["m"], r["dt"], r["kind"]) == (nx, m, dt, kind)]
        record = {"nx": nx, "m": m, "dt": dt, "kind": kind, "global_cfl": group[0]["global_cfl"]}
        for convention in ("gmres_preconditioned", "gmres_true"):
            def score(r: dict[str, Any], convention: str = convention) -> tuple[float, float]:
                count = r[convention]
                return (math.inf if count is None else count, r["eta"])

            best = min(group, key=score)
            record[f"best_eta_{convention}"] = best["eta"]
            record[f"best_{convention}"] = best[convention]
            record[f"best_masked_{convention}"] = best["masked"]
        out.append(record)
    return out


# ---------------------------------------------------------------------------
# The N = 2048 floor case
# ---------------------------------------------------------------------------


def run_floor(config: CflFidelityConfig) -> list[dict[str, Any]]:
    """Return the rows of the floor case at N = floor_nx."""
    case = make_case(config.floor_nx, config.floor_m, config)
    dt = config.floor_dt
    D = case["D"]
    J = jacobian(case["L"], D, dt)
    b = dt * (case["L"] @ case["phi"](case["u"]))
    rows = []
    for eta in config.floor_etas:
        for kind in ("Z1", "Zf"):
            M = preconditioner(case, D, dt, kind, eta)
            counts = gmres_two_conventions(M, J, b, config.gmres_tol, config.floor_gmres_max)
            counts.pop("checkpoints")
            row = {
                "experiment": "floor",
                "nx": case["nx"],
                "m": case["m"],
                "dt": dt,
                "kind": kind,
                "eta": eta,
                "floor_cfl": floor_cfl(case["m"], config.epsilon, dt, case["dx"]),
                "global_cfl": float(local_cfl(case, D, dt).max()),
                "masked": int(np.count_nonzero(cfl_mask(case, D, eta, dt))),
                "gmres_max": config.floor_gmres_max,
                **counts,
            }
            rows.append(row)
            print(
                f"floor N={case['nx']} m={case['m']} eta={eta:4.2g} {kind} masked={row['masked']} "
                f"gmres(prec/true)={row['gmres_preconditioned']}/{row['gmres_true']} "
                f"final(prec/true)={row['final_preconditioned_ratio']:.3g}/{row['final_true_ratio']:.3g}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# E3: inexact-Newton fidelity
# ---------------------------------------------------------------------------


def front_halfwidth(case: dict[str, Any], u: np.ndarray, level: float) -> float:
    """Return half the distance between the outer nodes with u > level max(u)."""
    idx = np.nonzero(u > level * u.max())[0]
    return float(case["x"][idx[-1]] - case["x"][idx[0]]) / 2.0


def newton_step(
    case: dict[str, Any],
    u_prev: np.ndarray,
    dt: float,
    kind: str,
    eta: float,
    forcing: float,
    side: str,
    config: CflFidelityConfig,
    *,
    nl_tol: float,
    max_newton: int,
    atol: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Take one backward-Euler step by globalized (inexact) Newton.

    The step stops when ||R|| <= max(nl_tol ||R0||, atol).  The step accepts a
    Newton update only when the Armijo test on ||R|| passes.  When all
    backtracking trials fail, the step stops with the last accepted iterate,
    so that the returned state and residual always agree.  ``kind`` "direct"
    uses a dense direct solve.
    """
    u = u_prev.copy()
    r = residual(case, u, u_prev, dt)
    r0 = float(np.linalg.norm(r))
    target = max(nl_tol * r0, atol)
    newton = linear = 0
    first_1e8 = 0 if r0 == 0.0 else None
    true_ratios: list[float] = []
    line_search_failed = False
    while np.linalg.norm(r) > target and newton < max_newton:
        D = case["Dfun"](u)
        J = jacobian(case["L"], D, dt)
        if kind == "direct":
            du = np.linalg.solve(J, -r)
        else:
            M = preconditioner(case, D, dt, kind, eta)
            if side == "left":
                out = gmres(M @ J, -(M @ r), forcing, config.e3_linear_max)
                du = out["x"]
            elif side == "right":
                out = gmres(J @ M, -r, forcing, config.e3_linear_max)
                du = M @ out["x"]
            else:
                raise ValueError(f"unknown side {side!r}")
            linear += out["iterations"]
            true_ratios.append(float(np.linalg.norm(J @ du + r) / np.linalg.norm(r)))
        newton += 1
        step, rn = 1.0, float(np.linalg.norm(r))
        accepted = None
        for _ in range(config.e3_backtracks):
            trial = residual(case, u + step * du, u_prev, dt)
            if np.linalg.norm(trial) <= (1.0 - config.e3_armijo * step) * rn:
                accepted = (u + step * du, trial)
                break
            step *= 0.5
        if accepted is None:
            line_search_failed = True
            break
        u, r = accepted
        if first_1e8 is None and np.linalg.norm(r) <= 1.0e-8 * r0:
            first_1e8 = newton
    if not np.array_equal(r, residual(case, u, u_prev, dt)):
        raise AssertionError("the returned residual does not belong to the returned state")
    final = float(np.linalg.norm(r))
    return u, {
        "residual_initial": r0,
        "residual_final": final,
        "residual_target": target,
        "converged": bool(final <= target),
        "line_search_failed": line_search_failed,
        "newton": newton,
        "linear_iterations": linear,
        "newton_to_1e-8": first_1e8,
        "max_true_linear_ratio": max(true_ratios) if true_ratios else 0.0,
    }


def run_e3(case: dict[str, Any], config: CflFidelityConfig) -> tuple[list[dict], list[dict]]:
    """Return the E3 trajectory rows and the per-step rows of one state."""
    h = case["dx"]
    steps = config.e3_steps
    level = config.e3_front_level
    summaries, step_rows = [], []
    for dt in config.e3_dts:
        c_global = float(local_cfl(case, case["D"], dt).max())
        reference = [case["u"].copy()]
        for step in range(steps):
            u_prev = reference[-1]
            atol = 64.0 * np.finfo(float).eps * (1.0 + float(np.linalg.norm(u_prev)))
            u, info = newton_step(
                case, u_prev, dt, "direct", math.nan, 0.0, "", config,
                nl_tol=config.e3_reference_tol,
                max_newton=config.e3_reference_max_newton,
                atol=atol,
            )
            reference.append(u)
            step_rows.append(
                {"experiment": "E3_step", "nx": case["nx"], "m": case["m"], "dt": dt,
                 "global_cfl": c_global, "kind": "direct", "side": None, "forcing": None,
                 "eta": None, "step": step + 1, "absolute_floor": atol, **info}
            )
        reference_ok = all(
            r["converged"] and not r["line_search_failed"]
            for r in step_rows
            if r["kind"] == "direct" and r["dt"] == dt and r["m"] == case["m"]
        )
        ref_counts = [
            r["newton_to_1e-8"]
            for r in step_rows
            if r["kind"] == "direct" and r["dt"] == dt and r["m"] == case["m"]
        ]
        cells = [abs(front_halfwidth(case, reference[k + 1], level) - front_halfwidth(case, reference[k], level)) / h
                 for k in range(steps)]
        mass0 = float(np.sum(case["u"]) * h)
        base = {
            "experiment": "E3",
            "nx": case["nx"],
            "m": case["m"],
            "dt": dt,
            "global_cfl": c_global,
            "steps": steps,
            "reference_converged": reference_ok,
            "reference_newton_to_1e-8_per_step": ref_counts,
            "front_cells_per_step_mean": float(np.mean(cells)),
            "front_cells_per_step_max": float(np.max(cells)),
        }
        print(f"E3 N={case['nx']} m={case['m']} dt={dt:6.0e} c={c_global:8.3g} reference "
              f"converged={reference_ok} newton to 1e-8 per step={ref_counts}", flush=True)
        for side in config.e3_sides:
            for forcing in config.e3_forcings:
                for kind in KINDS:
                    eta = config.e3_eta if kind in ("Z1", "Zf") else math.nan
                    u = case["u"].copy()
                    records = []
                    for step in range(steps):
                        # A failed step keeps its last accepted iterate and the march
                        # continues from it: this models a solver without step rejection.
                        u, info = newton_step(
                            case, u, dt, kind, eta, forcing, side, config,
                            nl_tol=config.e3_newton_tol, max_newton=config.e3_max_newton,
                        )
                        records.append(info)
                        step_rows.append(
                            {"experiment": "E3_step", "nx": case["nx"], "m": case["m"], "dt": dt,
                             "global_cfl": c_global, "kind": kind, "side": side, "forcing": forcing,
                             "eta": eta if kind in ("Z1", "Zf") else None, "step": step + 1, **info}
                        )
                    final_ref = reference[-1]
                    row = {
                        **base,
                        "kind": kind,
                        "eta": eta if kind in ("Z1", "Zf") else None,
                        "side": side,
                        "forcing": forcing,
                        "converged": all(r["converged"] for r in records),
                        "steps_converged": sum(r["converged"] for r in records),
                        "line_search_failures": sum(r["line_search_failed"] for r in records),
                        "newton_total": sum(r["newton"] for r in records),
                        "linear_iterations_total": sum(r["linear_iterations"] for r in records),
                        "newton_to_1e-8_per_step": [r["newton_to_1e-8"] for r in records],
                        "max_true_linear_ratio": max(r["max_true_linear_ratio"] for r in records),
                        "relative_l2_difference_to_reference": float(
                            np.linalg.norm(u - final_ref) / np.linalg.norm(final_ref)
                        ),
                        "front_halfwidth_difference": front_halfwidth(case, u, level)
                        - front_halfwidth(case, final_ref, level),
                        "mass_difference_relative": float(np.sum(u) * h - np.sum(final_ref) * h) / mass0,
                        "min_u": float(u.min()),
                    }
                    summaries.append(row)
                    print(
                        f"E3 N={case['nx']} m={case['m']} c={c_global:8.3g} {side:5s} f={forcing:5.0e} "
                        f"{kind:2s} ok={row['steps_converged']}/{steps} lsfail={row['line_search_failures']} "
                        f"newton={row['newton_total']} linear={row['linear_iterations_total']} "
                        f"maxtrue={row['max_true_linear_ratio']:.2e} "
                        f"err={row['relative_l2_difference_to_reference']:.2e} "
                        f"front={row['front_halfwidth_difference']:+.2g}",
                        flush=True,
                    )
    return summaries, step_rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _e4_rows(e1_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in e1_rows:
        for point in row["e4"]:
            out.append(
                {
                    "experiment": "E4",
                    "nx": row["nx"],
                    "m": row["m"],
                    "dt": row["dt"],
                    "global_cfl": row["global_cfl"],
                    "kind": row["kind"],
                    "eta": row["eta"],
                    **point,
                }
            )
    return out


def main(argv: list[str] | None = None) -> None:
    """Run the selected parts and write the JSON and CSV results."""
    defaults = CflFidelityConfig()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--parts", nargs="+", choices=("e1", "floor", "e3"), default=list(defaults.parts))
    parser.add_argument("--nx", type=int, default=defaults.nx)
    parser.add_argument("--m", type=int, nargs="+", default=list(defaults.m))
    parser.add_argument("--e1-dts", type=float, nargs="+", default=list(defaults.e1_dts))
    parser.add_argument("--e3-dts", type=float, nargs="+", default=list(defaults.e3_dts))
    parser.add_argument("--e3-steps", type=int, default=defaults.e3_steps)
    parser.add_argument("--e3-forcings", type=float, nargs="+", default=list(defaults.e3_forcings))
    parser.add_argument("--floor-nx", type=int, default=defaults.floor_nx)
    parser.add_argument("--output", default=defaults.output)
    args = parser.parse_args(argv)
    config = CflFidelityConfig(
        parts=tuple(args.parts),
        nx=args.nx,
        m=tuple(args.m),
        e1_dts=tuple(args.e1_dts),
        e3_dts=tuple(args.e3_dts),
        e3_steps=args.e3_steps,
        e3_forcings=tuple(args.e3_forcings),
        floor_nx=args.floor_nx,
        output=args.output,
    )
    e1_rows: list[dict] = []
    floor_rows: list[dict] = []
    e3_rows: list[dict] = []
    e3_steps: list[dict] = []
    states = []
    if "e1" in config.parts or "e3" in config.parts:
        for m in config.m:
            case = make_case(config.nx, m, config)
            states.append(
                {
                    "nx": case["nx"],
                    "m": m,
                    "h": case["dx"],
                    "D_max": float(case["D"].max()),
                    "rhs_dense_relative_difference_dt_0.02": case["rhs_dense_relative_difference"],
                }
            )
            if "e1" in config.parts:
                e1_rows.extend(run_e1(case, config))
            if "e3" in config.parts:
                summaries, steps = run_e3(case, config)
                e3_rows.extend(summaries)
                e3_steps.extend(steps)
    if "floor" in config.parts:
        floor_rows = run_floor(config)
    output = Path(config.output)
    if not output.is_absolute():
        output = REPOSITORY / output
    e4 = _e4_rows(e1_rows)
    payload = {
        "schema": SCHEMA,
        "provenance": provenance(config),
        "mask_rule": MaskAnalysisConfig().mask_rule,
        "residual_conventions": {
            "gmres_preconditioned": "first k with ||M(b - J x_k)|| <= tol ||M b|| (left preconditioning)",
            "gmres_true": "first k with ||b - J x_k|| <= tol ||b||, same Krylov sequence",
            "e3_left": "forcing on ||M r|| / ||M r0|| (left-preconditioned GMRES)",
            "e3_right": "forcing on ||r|| / ||r0|| (right-preconditioned GMRES)",
            "e3_reference": "direct Newton solves to max(1e-10 ||R0||, 64 eps (1 + ||u_prev||))",
            "e3_error_norm": "relative 2-norm difference of the final state to the reference trajectory",
        },
        "states": states,
        "e1_e2": [{k: v for k, v in r.items() if k != "e4"} for r in e1_rows],
        "e2_best_cutoffs": best_cutoffs(e1_rows),
        "e4": e4,
        "floor": floor_rows,
        "e3": e3_rows,
        "e3_steps": e3_steps,
    }
    flat = (
        [{k: v for k, v in r.items() if k != "e4"} for r in e1_rows]
        + e4
        + floor_rows
        + [{k: v for k, v in r.items() if not isinstance(v, list)} for r in e3_rows]
        + e3_steps
    )
    write_outputs(output, payload, flat)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
