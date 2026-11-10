#!/usr/bin/env python3
"""Active-block certificates, dense FoV policy, and true-residual work."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.linalg

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from benchmarks.tau_blend_benchmark_support import benchmark_provenance, write_json
from benchmarks.tau_blend_mask_analysis import (
    blend_matrix,
    build_state,
    cfl_mask,
    gaussian_log_weights,
    jacobian,
    one_inverse_matrix,
    quantile_references,
    repository_mask,
)
from moljax.experimental.pseudospectral_criterion import (
    assess_dense_pseudospectral_criterion,
)
from moljax.experimental.tau_blend_active_block import (
    dense_exact_support_fov,
    emulate_merged_policy,
    extract_active_block,
    lemma1_iteration_bound,
    true_residual_gmres,
)

DEGREES = tuple(range(4, 41, 4))
GRID_LADDER = (121, 241, 481)
CASE_CAP_SECONDS = 4.0 * 60.0 * 60.0
RATE_TIE_THRESHOLD = 0.05
METHODS_BY_M = {
    2: ("one_inverse_cfl_mask", "repository_blend_B3", "cfl_reference_blend_B3"),
    4: ("one_inverse_cfl_mask", "cfl_reference_blend_B3"),
    8: ("one_inverse_cfl_mask", "cfl_reference_blend_B3"),
}


def _state(nx: int, m: int) -> dict[str, Any]:
    config = SimpleNamespace(
        x_min=-4.0,
        x_max=4.0,
        t0=0.1,
        halfwidth=3.0,
        epsilon=1.0e-5,
    )
    return build_state(nx, m, 0.02, config)


def _preconditioner(
    state: dict[str, Any],
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diffusivity = state["D"]
    eta = 0.3 if state["m"] == 2 else 0.1
    if method == "one_inverse_cfl_mask":
        mask = cfl_mask(state, diffusivity, eta)
        references = np.asarray([diffusivity[~mask].max()])
        return one_inverse_matrix(state, diffusivity, mask), mask, references
    if method == "repository_blend_B3":
        mask = repository_mask(diffusivity)
    elif method == "cfl_reference_blend_B3":
        mask = cfl_mask(state, diffusivity, eta)
    else:
        raise ValueError(f"unknown method {method!r}")
    references = quantile_references(diffusivity, mask, 3)
    weights = gaussian_log_weights(diffusivity, references, ~mask)
    return blend_matrix(state, mask, references, weights), mask, references


def _spectrum(matrix: np.ndarray) -> dict[str, Any]:
    values = scipy.linalg.eigvals(matrix, check_finite=False)
    singular = scipy.linalg.svdvals(matrix, check_finite=False)
    return {
        "minimum_real": float(values.real.min()),
        "maximum_real": float(values.real.max()),
        "minimum_abs": float(np.abs(values).min()),
        "maximum_abs": float(np.abs(values).max()),
        "maximum_abs_imag": float(np.abs(values.imag).max()),
        "all_real_to_1e-10": bool(np.abs(values.imag).max() <= 1.0e-10),
        "sigma_min": float(singular[-1]),
        "sigma_max": float(singular[0]),
    }


def _certificate_summary(result, wall_seconds: float, block_constant: float | None) -> dict:
    predicted = None
    if (
        block_constant is not None
        and result.polynomial_degree is not None
        and result.contour_modulus_bound is not None
        and result.theorem_prefactor is not None
    ):
        predicted = lemma1_iteration_bound(
            polynomial_degree=result.polynomial_degree,
            contour_modulus_bound=result.contour_modulus_bound,
            theorem_prefactor=result.theorem_prefactor,
            block_constant=block_constant,
            target_tolerance=1.0e-8,
        )
    return {
        "certified": bool(result.certified),
        "reason": result.reason,
        "grid": result.grid_points_per_axis,
        "degree": result.polynomial_degree,
        "rho": result.polynomial_effective_rate,
        "prefactor": result.theorem_prefactor,
        "contour_modulus_bound": result.contour_modulus_bound,
        "contour_floor": result.contour_sigma_min_lower_bound,
        "epsilon_zero": result.epsilon_zero,
        "eps_connect_grid": result.eps_connect_grid,
        "eps_connect_upper": result.eps_connect_upper,
        "window_width": result.epsilon_zero - result.eps_connect_upper,
        "criterion_predicted_iterations": result.predicted_iterations,
        "lemma1_full_operator_iteration_bound": predicted,
        "constraint_points": result.polynomial_constraint_point_count,
        "sigma_min_evaluations": result.sigma_min_evaluations,
        "sigma_min_seconds": result.sigma_min_seconds,
        "search_seconds": result.polynomial_search_seconds,
        "wall_seconds": wall_seconds,
    }


def _certificate_ladder(
    matrix: np.ndarray,
    *,
    block_constant: float | None,
    first_grid: int = 121,
) -> list[dict[str, Any]]:
    # Match the archived B0 certificate protocol exactly.  The B0 scratch
    # runner passed complex128 matrices even when the operator was real; that
    # dtype is part of the numerical protocol because LAPACK's real and
    # complex SVD paths differ at roundoff level and can select different
    # degenerate LP minimizers.
    matrix = np.asarray(matrix, dtype=np.complex128)
    rows = []
    accumulated = 0.0
    grids = [first_grid, *(grid for grid in GRID_LADDER if grid > first_grid)]
    for grid in grids:
        projected = 0.0
        if rows and not rows[-1].get("stopped_before_run"):
            previous_grid = int(rows[-1]["grid"])
            projected = float(rows[-1]["wall_seconds"]) * (grid / previous_grid) ** 2
        if accumulated >= CASE_CAP_SECONDS or (
            projected > 0.0 and accumulated + projected > CASE_CAP_SECONDS
        ):
            rows.append(
                {
                    "certified": False,
                    "reason": "estimated cumulative compute exceeds four-hour per-case cap",
                    "grid": grid,
                    "projected_wall_seconds": projected,
                    "stopped_before_run": True,
                }
            )
            break
        started = perf_counter()
        result = assess_dense_pseudospectral_criterion(
            matrix,
            grid_points_per_axis=grid,
            domain_padding_fraction=0.75,
            polynomial_degrees=DEGREES,
            epsilon_samples=6,
            target_tolerance=1.0e-8,
        )
        wall = perf_counter() - started
        accumulated += wall
        rows.append(_certificate_summary(result, wall, block_constant))
        if result.certified:
            break
    return rows


def _gmres_summary(result: dict[str, Any], operator: str) -> dict[str, Any]:
    """Return the auditable stopping counts for one named operator."""
    return {
        "operator": operator,
        "iterations_run": result["iterations"],
        "true_stop_count": result["true_stop_count"],
        "preconditioned_stop_count": result["preconditioned_stop_count"],
        "true_relative_residual": result["secondary_ratio"],
        "preconditioned_relative_residual": result["primary_ratio"],
        "converged": result["converged"],
    }


def analyze_case(nx: int, m: int, method: str) -> dict[str, Any]:
    """Run one unique method/state case."""
    started = perf_counter()
    state = _state(nx, m)
    preconditioner, mask, references = _preconditioner(state, method)
    zeroed_diffusivity = np.where(mask, 0.0, state["D"])
    j0 = jacobian(state["L"], zeroed_diffusivity, state["dt"])
    mj0 = preconditioner @ j0
    mj = preconditioner @ state["J"]
    blocks0 = extract_active_block(mj0, mask)
    blocks = extract_active_block(mj, mask)
    fov_mj = dense_exact_support_fov(mj, n_angles=32)
    fov_y = dense_exact_support_fov(blocks0.Y, n_angles=32)
    # The active-block certificate is for J0, so its measured comparator is
    # the same J0 solve used by the archived A5/B0 table.  Direct certificates
    # for the regularized full operator are reported separately below.
    gmres_j0 = true_residual_gmres(
        preconditioner,
        j0,
        state["b"],
        tolerance=1.0e-8,
    )
    gmres_regularized = true_residual_gmres(
        preconditioner,
        state["J"],
        state["b"],
        tolerance=1.0e-8,
    )
    y_ladder = _certificate_ladder(
        blocks0.Y,
        block_constant=blocks0.lemma1_constant,
    )
    selected = next((row for row in y_ladder if row["certified"]), y_ladder[-1])
    disk_policy = emulate_merged_policy(mj, fov_mj)
    union = (
        "certified" if disk_policy["verdict"] == "adequate" or selected["certified"] else "abstains"
    )
    return {
        "nx": nx,
        "m": m,
        "method": method,
        "eta": 0.3 if m == 2 else 0.1,
        "references": [float(value) for value in references],
        "inactive_count": int(mask.sum()),
        "active_count": int((~mask).sum()),
        "J0_blocks": {
            "max_abs_E": float(np.max(np.abs(blocks0.E))) if blocks0.E.size else 0.0,
            "max_abs_F": float(np.max(np.abs(blocks0.F))) if blocks0.F.size else 0.0,
            "norm_X": float(np.linalg.norm(blocks0.X, 2)) if blocks0.X.size else 0.0,
            "lemma1_constant": blocks0.lemma1_constant,
        },
        "regularized_blocks": {
            "max_abs_E": float(np.max(np.abs(blocks.E))) if blocks.E.size else 0.0,
            "max_abs_F": float(np.max(np.abs(blocks.F))) if blocks.F.size else 0.0,
            "Y_byte_identical_to_J0": bool(np.array_equal(blocks0.Y, blocks.Y)),
        },
        "Y_spectrum": _spectrum(blocks0.Y),
        "MJ_spectrum": _spectrum(mj),
        "Y_certificate_ladder": y_ladder,
        "Y_certificate_selected": selected,
        "Y_fov": asdict(fov_y),
        "MJ_fov": asdict(fov_mj),
        "merged_policy": disk_policy,
        "union": union,
        "gmres_J0": _gmres_summary(gmres_j0, "J0"),
        "gmres_regularized": _gmres_summary(gmres_regularized, "regularized_J"),
        "gmres": _gmres_summary(gmres_regularized, "regularized_J (headline)"),
        "wall_seconds": perf_counter() - started,
    }


def full_regularized_certificate(nx: int) -> dict[str, Any]:
    """Certify the regularized full MJ directly for the m=2 CFL blend."""
    state = _state(nx, 2)
    preconditioner, mask, references = _preconditioner(state, "cfl_reference_blend_B3")
    matrix = preconditioner @ state["J"]
    # B0 escalated this sole wide-window full-operator case to 565 points.
    # Keep that selected grid explicit so regeneration is idempotent.
    first_grid = 565 if nx == 128 else 121
    return {
        "nx": nx,
        "m": 2,
        "method": "cfl_reference_blend_B3",
        "operator": "regularized_full_MJ",
        "dimension": nx,
        "inactive_count": int(mask.sum()),
        "references": [float(value) for value in references],
        "certificate_ladder": _certificate_ladder(
            matrix,
            block_constant=None,
            first_grid=first_grid,
        ),
    }


def refresh_case_protocol(row: dict[str, Any]) -> dict[str, Any]:
    """Refresh protocol-sensitive J0 work and an already-selected certificate.

    This is the checkpoint-resume path used after tightening the certificate
    evaluation enclosure.  Structural abstentions retain their completed grid
    ladders; certified rows are re-evaluated at the already-selected B0 grid
    and degree, exactly as in the independent Part-0 soundness recheck.
    """
    nx, m, method = int(row["nx"]), int(row["m"]), str(row["method"])
    state = _state(nx, m)
    preconditioner, mask, _ = _preconditioner(state, method)
    zeroed_diffusivity = np.where(mask, 0.0, state["D"])
    j0 = jacobian(state["L"], zeroed_diffusivity, state["dt"])
    blocks0 = extract_active_block(preconditioner @ j0, mask)
    gmres_j0 = true_residual_gmres(preconditioner, j0, state["b"], tolerance=1.0e-8)
    gmres_regularized = true_residual_gmres(
        preconditioner,
        state["J"],
        state["b"],
        tolerance=1.0e-8,
    )
    row["gmres_J0"] = _gmres_summary(gmres_j0, "J0")
    row["gmres_regularized"] = _gmres_summary(gmres_regularized, "regularized_J")
    row["gmres"] = _gmres_summary(gmres_regularized, "regularized_J (headline)")
    ladder = row["Y_certificate_ladder"]
    selected_index = next((index for index, item in enumerate(ladder) if item["certified"]), None)
    if selected_index is not None:
        selected = ladder[selected_index]
        started = perf_counter()
        result = assess_dense_pseudospectral_criterion(
            np.asarray(blocks0.Y, dtype=np.complex128),
            grid_points_per_axis=int(selected["grid"]),
            domain_padding_fraction=0.75,
            polynomial_degrees=(int(selected["degree"]),),
            epsilon_samples=6,
            target_tolerance=1.0e-8,
        )
        if not result.certified:
            raise RuntimeError(f"protocol refresh lost certificate for {(nx, m, method)}")
        ladder[selected_index] = _certificate_summary(
            result,
            perf_counter() - started,
            blocks0.lemma1_constant,
        )
    row["Y_certificate_selected"] = next(
        (item for item in ladder if item["certified"]),
        ladder[-1],
    )
    return row


def refresh_gmres_counts(row: dict[str, Any]) -> dict[str, Any]:
    """Refresh both GMRES operators without rerunning any certificate."""
    nx, m, method = int(row["nx"]), int(row["m"]), str(row["method"])
    state = _state(nx, m)
    preconditioner, mask, _ = _preconditioner(state, method)
    zeroed_diffusivity = np.where(mask, 0.0, state["D"])
    j0 = jacobian(state["L"], zeroed_diffusivity, state["dt"])
    gmres_j0 = true_residual_gmres(preconditioner, j0, state["b"], tolerance=1.0e-8)
    gmres_regularized = true_residual_gmres(
        preconditioner,
        state["J"],
        state["b"],
        tolerance=1.0e-8,
    )
    row["gmres_J0"] = _gmres_summary(gmres_j0, "J0")
    row["gmres_regularized"] = _gmres_summary(gmres_regularized, "regularized_J")
    row["gmres"] = _gmres_summary(gmres_regularized, "regularized_J (headline)")
    row["MJ_spectrum"] = _spectrum(preconditioner @ state["J"])
    return row


def refresh_full_protocol(row: dict[str, Any]) -> dict[str, Any]:
    """Refresh one selected direct full-operator certificate in place."""
    nx = int(row["nx"])
    state = _state(nx, 2)
    preconditioner, _, _ = _preconditioner(state, "cfl_reference_blend_B3")
    ladder = row["certificate_ladder"]
    selected_index = next(index for index, item in enumerate(ladder) if item["certified"])
    selected = ladder[selected_index]
    started = perf_counter()
    result = assess_dense_pseudospectral_criterion(
        np.asarray(preconditioner @ state["J"], dtype=np.complex128),
        grid_points_per_axis=int(selected["grid"]),
        domain_padding_fraction=0.75,
        polynomial_degrees=(int(selected["degree"]),),
        epsilon_samples=6,
        target_tolerance=1.0e-8,
    )
    if not result.certified:
        raise RuntimeError(f"protocol refresh lost full certificate for N={nx}")
    ladder[selected_index] = _certificate_summary(result, perf_counter() - started, None)
    return row


def ordering_check(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare union/rate ordering with measured true-residual GMRES."""
    reports = []
    for nx in sorted({row["nx"] for row in rows}):
        for m in sorted({row["m"] for row in rows if row["nx"] == nx}):
            group = [row for row in rows if row["nx"] == nx and row["m"] == m]
            fastest = min(row["gmres"]["true_stop_count"] for row in group)
            certified = [row for row in group if row["union"] == "certified"]
            certified_fastest = any(row["gmres"]["true_stop_count"] == fastest for row in certified)
            rates = [
                (row["method"], row["Y_certificate_selected"].get("rho"))
                for row in group
                if row["Y_certificate_selected"]["certified"]
            ]
            nonnull_rates = [(method, rate) for method, rate in rates if rate is not None]
            predicted_tie = False
            if len(nonnull_rates) > 1:
                values = [rate for _, rate in nonnull_rates]
                predicted_tie = max(values) - min(values) < RATE_TIE_THRESHOLD
            reports.append(
                {
                    "nx": nx,
                    "m": m,
                    "fastest_true_stop_count": fastest,
                    "union_contains_fastest": certified_fastest,
                    "certified_rates": nonnull_rates,
                    "rate_differences_below_0p05_treated_as_ties": predicted_tie,
                }
            )
    return reports


def decision_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Make the union convention and its counts explicit in the artifact."""
    return {
        "case_count": len(rows),
        "Y_certified_count": sum(row["Y_certificate_selected"]["certified"] for row in rows),
        "disk_adequate_count": sum(row["merged_policy"]["verdict"] == "adequate" for row in rows),
        "union_certified_count": sum(row["union"] == "certified" for row in rows),
        "union_positive_rule": (
            "Y certificate is certified OR merged dense-disk policy verdict is adequate"
        ),
        "disk_positive_categories": ["adequate"],
        "disk_nonpositive_categories": ["provisional", "investigate", "indeterminate"],
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, nargs="+", default=[128, 512, 1024])
    parser.add_argument("--m", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--method", nargs="+", default=None)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--refresh-protocol",
        action="store_true",
        help="refresh J0 GMRES and selected certificates in a completed checkpoint",
    )
    parser.add_argument(
        "--refresh-gmres",
        action="store_true",
        help="refresh J0 and regularized GMRES counts without rerunning certificates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/tau_blend_active_block.json"),
    )
    args = parser.parse_args()
    started = perf_counter()
    selected = [
        (nx, m, method)
        for nx in args.nx
        for m in args.m
        for method in METHODS_BY_M[m]
        if args.method is None or method in args.method
    ]
    rows: list[dict[str, Any]] = []
    full: list[dict[str, Any]] = []
    if args.resume and args.output.is_file():
        previous = json.loads(args.output.read_text())
        rows = previous.get("cases", [])
        full = previous.get("direct_regularized_full_MJ_certificates", [])
    completed = {(row["nx"], row["m"], row["method"]) for row in rows}
    configuration = {
        "grid_ladder": list(GRID_LADDER),
        "polynomial_degrees": list(DEGREES),
        "epsilon_samples": 6,
        "target_tolerance": 1.0e-8,
        "fov_directions": 32,
        "case_cap_seconds": CASE_CAP_SECONDS,
        "true_residual_protocol": "physical Newton RHS, unrestarted, tolerance 1e-8",
        "gmres_operators": ["J0", "regularized_J"],
        "gmres_headline_operator": "regularized_J",
        "certificate_matrix_dtype": "complex128",
        "full_operator_selected_grid_N128": 565,
        "rewrite_base_revision": "ad478fc",
        "rate_tie_threshold": RATE_TIE_THRESHOLD,
        "certificate_changes": [
            {
                "commit": "61fa907",
                "change": "centred derivative enclosure and numerical-range domain fallback",
            },
            {
                "commit": "a7d6dae",
                "change": "Horner floating-point evaluation allowance",
            },
            {
                "commit": "c9c4a8e",
                "change": "soundness and monotonicity validation for the tightened certificate",
            },
        ],
    }

    def checkpoint() -> None:
        payload = {
            "schema": "tau_blend_active_block_v1",
            "configuration": configuration,
            "cases": rows,
            "direct_regularized_full_MJ_certificates": full,
            "decision_summary": decision_summary(rows),
            "ordering_check": ordering_check(rows),
            "provenance": benchmark_provenance(__file__, perf_counter() - started),
        }
        write_json(args.output, _json_ready(payload))

    if args.refresh_protocol:
        if not args.resume or not rows:
            raise ValueError("--refresh-protocol requires --resume and a completed checkpoint")
        for index, row in enumerate(rows):
            rows[index] = refresh_case_protocol(row)
            checkpoint()
        if not args.skip_full:
            for index, row in enumerate(full):
                full[index] = refresh_full_protocol(row)
                checkpoint()
        checkpoint()
        return

    if args.refresh_gmres:
        if not args.resume or not rows:
            raise ValueError("--refresh-gmres requires --resume and a completed checkpoint")
        for index, row in enumerate(rows):
            rows[index] = refresh_gmres_counts(row)
        checkpoint()
        return

    for case in selected:
        if case in completed:
            continue
        rows.append(analyze_case(*case))
        checkpoint()
    if not args.skip_full:
        completed_full = {row["nx"] for row in full}
        for nx in args.nx:
            if 2 in args.m and nx not in completed_full:
                full.append(full_regularized_certificate(nx))
                checkpoint()
    checkpoint()


if __name__ == "__main__":
    main()
