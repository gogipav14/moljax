#!/usr/bin/env python3
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

jax.config.update("jax_enable_x64", True)
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

    The study advances exact-profile initializations by a stable backward-Euler
    step.  Their target support halfwidths provide a solver-visited
    front-sharpness axis, while ``analysis_dt / dx**2`` spans a deliberately
    wide stiffness range.
    """

    nx: int = 512
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    state_dt: float = 0.02
    analysis_dt_values: tuple[float, ...] = (2.0e-4, 2.0e-2, 2.0)
    epsilon: float = 1.0e-5
    front_target_halfwidths: tuple[float, ...] = (0.25, 0.75, 3.0)
    n_angles: int = 3
    fov_max_iters: int = 10
    arnoldi_steps: int = 6
    const_d0: float = 1.0
    max_newton_iters: int = 8
    max_krylov_iters: int = 400
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    m_values: tuple[int, ...] = (1, 2, 3, 4, 6, 8)
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
    target_halfwidth: float,
) -> jax.Array:
    """Return a control or Barenblatt state with a selected front scale."""
    x = grid.x_coords()
    if m == 1:
        return _heat_gaussian(x, t0)
    b = _barenblatt_b(m, target_halfwidth)
    return barenblatt(x, t0, float(m), b=b)


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
    residual = make_backward_euler_residual(interior, grid, float(m), config.state_dt, epsilon)
    preconditioner, d0 = pme_preconditioner_variant(
        interior,
        grid,
        float(m),
        config.state_dt,
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
        dt=config.state_dt,
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
    largest_dt = max(config.analysis_dt_values)
    settings = ((1.0, largest_dt), (1.0, 0.1 * largest_dt), (0.16, largest_dt))
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
    state_verdicts: dict[tuple[int, float, int], set[str]] = {}
    for record in records:
        key = _state_key(record)
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
    by_state: dict[tuple[int, float, int], list[dict[str, Any]]] = {}
    for record in records:
        key = _state_key(record)
        by_state.setdefault(key, []).append(record)

    states = []
    concordant = 0
    discordant = 0
    tied = 0
    exact_order_count = 0
    for (m, analysis_dt, front_case), rows in sorted(by_state.items()):
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
                "analysis_dt": analysis_dt,
                "front_case": front_case,
                "target_support_halfwidth": rows[0]["target_support_halfwidth"],
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
            "analysis_dt": record["analysis_dt"],
            "front_case": record["front_case"],
            "target_support_halfwidth": record["target_support_halfwidth"],
            "d0_kind": record["d0_kind"],
            "disk_rate": record["disk_rate"],
            "actual_iterations": record["actual_gmres"]["iterations"],
        }
        for record in records
    ]
    predictor_pairs = [
        {
            "m": record["m"],
            "analysis_dt": record["analysis_dt"],
            "front_case": record["front_case"],
            "target_support_halfwidth": record["target_support_halfwidth"],
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
                "analysis_dt": record["analysis_dt"],
                "front_case": record["front_case"],
                "target_support_halfwidth": record["target_support_halfwidth"],
                "d0_kind": record["d0_kind"],
                "disk_rate": record["disk_rate"],
                "actual_gmres_iterations": record["actual_gmres"]["iterations"],
            }
            for record in records
        ],
    }


def _state_key(record: dict[str, Any]) -> tuple[int, float, int]:
    """Return the common physical-state and candidate-step identifier."""
    return int(record["m"]), float(record["analysis_dt"]), int(record["front_case"])


def _cell_label(m: int, analysis_dt: float) -> str:
    """Format one exponent/stiffness cell for a compact report statement."""
    return f"(m={m}, dt={analysis_dt:g})"


def _regime_map(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Map identity-conditioning evidence across exponent and stiffness cells."""
    by_cell: dict[tuple[int, float], list[dict[str, Any]]] = {}
    for record in records:
        if record["d0_kind"] != "identity":
            continue
        key = int(record["m"]), float(record["analysis_dt"])
        by_cell.setdefault(key, []).append(record)

    cells: list[dict[str, Any]] = []
    reliable_cells: list[str] = []
    unreliable_or_benign_cells: list[str] = []
    for (m, analysis_dt), identity_records in sorted(by_cell.items()):
        iterations = [float(record["actual_gmres"]["iterations"]) for record in identity_records]
        disk_rates = [float(record["disk_rate"]) for record in identity_records]
        threshold = float(median(iterations))
        hard_records = [
            record
            for record in identity_records
            if float(record["actual_gmres"]["iterations"]) >= threshold
        ]
        easy_records = [
            record
            for record in identity_records
            if float(record["actual_gmres"]["iterations"]) < threshold
        ]
        minimum = min(iterations)
        maximum = max(iterations)
        ratio = float("inf") if minimum == 0.0 and maximum > 0.0 else maximum / minimum
        pearson = _pearson(disk_rates, iterations)
        label = _cell_label(m, analysis_dt)
        if pearson is not None and pearson > 0.8:
            reliable_cells.append(label)
        else:
            unreliable_or_benign_cells.append(label)
        cells.append(
            {
                "m": m,
                "analysis_dt": analysis_dt,
                "identity_records": len(identity_records),
                "identity_disk_rate_vs_iterations_pearson": pearson,
                "identity_hard_iteration_threshold": threshold,
                "identity_hard_disk_rate": _summary(
                    [float(record["disk_rate"]) for record in hard_records]
                ),
                "identity_easy_disk_rate": _summary(
                    [float(record["disk_rate"]) for record in easy_records]
                ),
                "identity_investigate_fraction": (
                    sum(record["verdict"] == "investigate" for record in identity_records)
                    / len(identity_records)
                ),
                "identity_hard_investigate_fraction": (
                    sum(record["verdict"] == "investigate" for record in hard_records)
                    / len(hard_records)
                    if hard_records
                    else None
                ),
                "identity_easy_investigate_fraction": (
                    sum(record["verdict"] == "investigate" for record in easy_records)
                    / len(easy_records)
                    if easy_records
                    else None
                ),
                "identity_iteration_range": {
                    **_summary(iterations),
                    "ratio": ratio,
                },
            }
        )

    reliable_text = ", ".join(reliable_cells) if reliable_cells else "none"
    unreliable_text = (
        ", ".join(unreliable_or_benign_cells) if unreliable_or_benign_cells else "none"
    )
    return {
        "description": (
            "Identity-preconditioned evidence by porous-medium exponent and candidate "
            "implicit-step size; hard and easy are split at each cell's identity median."
        ),
        "cells": cells,
        "reliable_cells_pearson_gt_0_8": reliable_cells,
        "unreliable_or_too_benign_cells_pearson_le_0_8": unreliable_or_benign_cells,
        "statement": (
            f"Identity Pearson > 0.8: {reliable_text}. "
            f"Identity Pearson <= 0.8: {unreliable_text}."
        ),
    }


def _verdict_on_decision_procedure(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Test the decision procedure against the identity-cost hard-state label.

    A state is hard when its identity-preconditioned counted-GMRES cost is at
    least the median identity cost over the sweep.  Comparing variants only
    within that label avoids conflating different right-hand sides across
    separate physical states.
    """
    identity_records = [record for record in records if record["d0_kind"] == "identity"]
    identity_iterations = [
        float(record["actual_gmres"]["iterations"]) for record in identity_records
    ]
    if not identity_iterations:
        raise ValueError("The stress study requires an identity D0 variant")

    minimum = min(identity_iterations)
    maximum = max(identity_iterations)
    ratio = float("inf") if minimum == 0.0 and maximum > 0.0 else maximum / minimum
    hard_threshold = float(median(identity_iterations))
    hard_state_keys = {
        _state_key(record)
        for record in identity_records
        if float(record["actual_gmres"]["iterations"]) >= hard_threshold
    }

    by_variant: dict[str, dict[str, Any]] = {}
    for d0_kind in sorted({str(record["d0_kind"]) for record in records}):
        variant_records = [record for record in records if record["d0_kind"] == d0_kind]
        hard_records = [
            record for record in variant_records if _state_key(record) in hard_state_keys
        ]
        easy_records = [
            record for record in variant_records if _state_key(record) not in hard_state_keys
        ]
        hard_disk_rates = [float(record["disk_rate"]) for record in hard_records]
        easy_disk_rates = [float(record["disk_rate"]) for record in easy_records]
        by_variant[d0_kind] = {
            "hard_records": len(hard_records),
            "easy_records": len(easy_records),
            "hard_disk_rate": _summary(hard_disk_rates),
            "easy_disk_rate": _summary(easy_disk_rates),
            "hard_actual_iterations": _summary(
                [float(record["actual_gmres"]["iterations"]) for record in hard_records]
            ),
            "easy_actual_iterations": _summary(
                [float(record["actual_gmres"]["iterations"]) for record in easy_records]
            ),
            "hard_investigate_fraction": (
                sum(record["verdict"] == "investigate" for record in hard_records)
                / len(hard_records)
                if hard_records
                else None
            ),
            "easy_investigate_fraction": (
                sum(record["verdict"] == "investigate" for record in easy_records)
                / len(easy_records)
                if easy_records
                else None
            ),
        }

    identity_disk_rates = [float(record["disk_rate"]) for record in identity_records]
    identity_pearson = _pearson(identity_disk_rates, identity_iterations)
    identity_stats = by_variant["identity"]
    hard_median = identity_stats["hard_disk_rate"]["median"]
    easy_median = identity_stats["easy_disk_rate"]["median"]
    hard_fraction = identity_stats["hard_investigate_fraction"]
    easy_fraction = identity_stats["easy_investigate_fraction"]
    separates = (
        identity_pearson is not None
        and hard_median is not None
        and easy_median is not None
        and hard_fraction is not None
        and easy_fraction is not None
        and float(hard_median) > float(easy_median)
        and hard_fraction >= easy_fraction
    )
    nonconvergence_cases = [
        {
            "m": record["m"],
            "analysis_dt": record["analysis_dt"],
            "front_case": record["front_case"],
            "target_support_halfwidth": record["target_support_halfwidth"],
            "d0_kind": record["d0_kind"],
            "iterations": record["actual_gmres"]["iterations"],
            "final_relative_residual": record["actual_gmres"]["final_relative_residual"],
            "disk_rate": record["disk_rate"],
            "verdict": record["verdict"],
        }
        for record in records
        if not record["actual_gmres"]["converged"]
    ]
    answer = "yes" if separates else "no"
    return {
        "description": (
            "Identity-preconditioned cost defines hard states; all D0 variants are then "
            "compared on the same state and candidate implicit step."
        ),
        "hard_label": "identity actual GMRES iterations >= median identity iterations",
        "hard_identity_iteration_threshold": hard_threshold,
        "identity_iteration_dynamic_range": {
            "min": minimum,
            "max": maximum,
            "ratio": ratio,
            "meets_tenfold_gate": ratio >= 10.0,
        },
        "identity_disk_rate_vs_iterations_pearson": identity_pearson,
        "by_variant": by_variant,
        "nonconvergence_cases": nonconvergence_cases,
        "data_supported_separation": separates,
        "answer": answer,
        "summary": (
            "yes: the identity disk rate rises from easy to hard states and the investigate "
            "fraction does not fall."
            if separates
            else "no: the identity disk-rate and verdict statistics do not jointly separate "
            "the hard-state label."
        ),
    }


def _record_state(
    records: list[dict[str, Any]],
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    front_case: int,
    target_halfwidth: float,
    analysis_dt: float,
    config: BreakdownConfig,
    centering: dict[str, Any],
    state_solver: dict[str, Any],
) -> None:
    """Measure every D0 variant at one state and candidate implicit step."""
    epsilon = 0.0 if m == 1 else config.epsilon
    visited_interior = interior_values(state, grid)
    front_gradient = float(jnp.max(jnp.abs(jnp.diff(visited_interior))) / grid.dx)
    for index, d0_kind in enumerate(config.d0_kinds):
        diagnostics = assess_pme_state(
            state,
            grid,
            float(m),
            analysis_dt,
            epsilon,
            d0_kind,
            const_value=config.const_d0,
            n_angles=config.n_angles,
            fov_max_iters=config.fov_max_iters,
            arnoldi_steps=config.arnoldi_steps,
            seed=20260900 + 1000 * m + 10 * front_case + index,
        )
        actual_gmres = measure_gmres_iterations(
            state,
            grid,
            float(m),
            analysis_dt,
            epsilon,
            d0_kind,
            tol=config.krylov_tol,
            max_iters=config.max_krylov_iters,
            const_value=config.const_d0,
        )
        records.append(
            {
                "m": m,
                "front_case": front_case,
                "target_support_halfwidth": target_halfwidth,
                "visited_step": 1,
                "visited_time": config.t0 + config.state_dt,
                "reference_state_dt": config.state_dt,
                "analysis_dt": analysis_dt,
                "front_max_gradient": front_gradient,
                "d0_kind": d0_kind,
                "d0_used": diagnostics["d0"],
                "sigma": float(diagnostics["d0"] * analysis_dt / grid.dx**2),
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
                "reference_state_solver": dict(state_solver),
                "centering_mismatch_note": centering["decision"],
            }
        )


def run_breakdown_study(config: BreakdownConfig | None = None) -> dict[str, Any]:
    """Run the powered D0-variant sweep and write the decision-grade JSON report."""
    if config is None:
        config = BreakdownConfig()
    if not config.front_target_halfwidths or min(config.front_target_halfwidths) <= 0.0:
        raise ValueError("front_target_halfwidths must contain positive halfwidths")
    if not config.analysis_dt_values or min(config.analysis_dt_values) <= 0.0:
        raise ValueError("analysis_dt_values must contain positive step sizes")

    started_at = perf_counter()
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    centering = _centering_report(config)
    records: list[dict[str, Any]] = []
    for m in config.m_values:
        for front_case, target_halfwidth in enumerate(config.front_target_halfwidths, start=1):
            state = _initial_state(grid, m, config.t0, target_halfwidth)
            state, state_solver = _solve_one_step(state, grid, m, config, "frozen_bulk")
            for analysis_dt in config.analysis_dt_values:
                _record_state(
                    records,
                    state,
                    grid,
                    m,
                    front_case,
                    target_halfwidth,
                    analysis_dt,
                    config,
                    centering,
                    state_solver,
                )

    runtime_seconds = perf_counter() - started_at
    decision = _verdict_on_decision_procedure(records)
    dynamic_range = decision["identity_iteration_dynamic_range"]
    if not dynamic_range["meets_tenfold_gate"]:
        raise RuntimeError(
            "Identity GMRES dynamic-range gate failed: "
            f"min={dynamic_range['min']}, max={dynamic_range['max']}, "
            f"ratio={dynamic_range['ratio']}"
        )
    report = {
        "description": (
            "Experimental PME conditioning study. It evaluates a preconditioner decision "
            "procedure; it does not claim to fix stiffness degradation."
        ),
        "config": config._asdict(),
        "metric": {
            "accuracy": "not evaluated in this conditioning-only stress study",
            "timing": "total runtime only",
        },
        "runtime_seconds": runtime_seconds,
        "gmres_measurement_note": (
            "actual_gmres is an explicit residual-history count for the fixed system "
            "P^-1 J delta = P^-1 (-R), not NKStats.lin_iters."
        ),
        "state_schedule_note": (
            "Each state is produced by the stable reference backward-Euler step size state_dt "
            "with the frozen_bulk preconditioner.  Each analysis_dt then defines a separate "
            "fixed linearized backward-Euler system at that genuinely visited state."
        ),
        "centering": centering,
        "regime_claim": _regime_claim(records),
        "rank_claim": _rank_claim(records),
        "predictor_quality": _predictor_quality(records),
        "correlation": _correlation_pairs(records),
        "regime_map": _regime_map(records),
        "verdict_on_decision_procedure": decision,
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
    decision = report["verdict_on_decision_procedure"]
    regime_map = report["regime_map"]
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
        f"raw_pearson={predictor['raw_disk_rate_pearson']} "
        f"envelope_pearson={predictor['envelope_predicted_pearson']} "
        f"median_absolute_error={predictor['median_absolute_error']}"
    )
    dynamic_range = decision["identity_iteration_dynamic_range"]
    print(
        "identity stress: "
        f"min={dynamic_range['min']:.0f} max={dynamic_range['max']:.0f} "
        f"ratio={dynamic_range['ratio']:.3f} "
        f"pearson={decision['identity_disk_rate_vs_iterations_pearson']} "
        f"answer={decision['answer']}"
    )
    print(f"regime map: {regime_map['statement']}")


def main() -> None:
    """Run the default study or write to an explicitly supplied JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=BreakdownConfig().output_path)
    args = parser.parse_args()
    result = run_breakdown_study(BreakdownConfig(output_path=args.output))
    _print_summary(result)


if __name__ == "__main__":
    main()
