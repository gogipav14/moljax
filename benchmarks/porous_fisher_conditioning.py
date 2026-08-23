#!/usr/bin/env python3
"""Reaction-axis conditioning study for the experimental Porous--Fisher equation."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

from moljax.core.newton_krylov import NKParams, newton_krylov_solve
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import pme_preconditioner_variant
from moljax.experimental.porous_fisher import porous_fisher_traveling_wave, wave_speed
from moljax.experimental.porous_fisher_conditioning import (
    assess_porous_fisher_state,
    make_porous_fisher_residual,
    measure_porous_fisher_gmres_iterations,
)


class ReactionStudyConfig(NamedTuple):
    """Static parameters for the reaction-axis Porous--Fisher study."""

    nx: int = 256
    x_min: float = -8.0
    x_max: float = 8.0
    initial_time: float = 0.25
    state_dt: float = 2.0e-3
    analysis_dt_values: tuple[float, ...] = (2.0e-4, 2.0e-2, 2.0)
    reaction_values: tuple[float, ...] = (0.0, 1.0, 100.0)
    reference_wave_r: float = 1.0
    epsilon: float = 1.0e-5
    d0_kinds: tuple[str, ...] = ("frozen_mean", "frozen_bulk", "floor", "const", "identity")
    const_d0: float = 1.0
    n_angles: int = 3
    fov_max_iters: int = 10
    arnoldi_steps: int = 6
    max_newton_iters: int = 8
    max_krylov_iters: int = 400
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    output_path: str = "benchmarks/results/porous_fisher_conditioning.json"


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


def _pearson(first: list[float], second: list[float]) -> float | None:
    """Compute Pearson correlation without an external statistics dependency."""
    if len(first) != len(second):
        raise ValueError("Pearson inputs must have matching lengths")
    if len(first) < 2:
        return None
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second, strict=True)
    )
    denominator = sqrt(
        sum((value - first_mean) ** 2 for value in first)
        * sum((value - second_mean) ** 2 for value in second)
    )
    return numerator / denominator if denominator else None


def _advance_to_visited_state(
    initial_state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    config: ReactionStudyConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    """Advance one stable BE step so each analysed state is solver-visited."""
    residual = make_porous_fisher_residual(
        initial_state,
        grid,
        r=r,
        dt=config.state_dt,
        epsilon=config.epsilon,
    )
    preconditioner, d0 = pme_preconditioner_variant(
        initial_state,
        grid,
        2.0,
        config.state_dt,
        config.epsilon,
        "frozen_bulk",
        const_value=config.const_d0,
    )
    result = newton_krylov_solve(
        residual,
        initial_state,
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
    return solution, {
        "d0": float(d0),
        "converged": bool(result.stats.converged),
        "newton_iters": int(result.stats.newton_iters),
        "configured_krylov_budget_total": int(result.stats.lin_iters),
        "final_residual_l2": float(result.stats.final_res_norm),
    }


def _identity_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize hard/easy geometry for one reaction strength."""
    identity_records = [record for record in records if record["d0_kind"] == "identity"]
    iterations = [float(record["actual_gmres"]["iterations"]) for record in identity_records]
    disk_rates = [float(record["disk_rate"]) for record in identity_records]
    threshold = float(median(iterations))
    hard = [
        record
        for record in identity_records
        if float(record["actual_gmres"]["iterations"]) >= threshold
    ]
    easy = [
        record
        for record in identity_records
        if float(record["actual_gmres"]["iterations"]) < threshold
    ]
    minimum = min(iterations)
    maximum = max(iterations)
    ratio = float("inf") if minimum == 0.0 and maximum > 0.0 else maximum / minimum
    pearson = _pearson(disk_rates, iterations)
    hard_disk_rate = _summary([float(record["disk_rate"]) for record in hard])
    easy_disk_rate = _summary([float(record["disk_rate"]) for record in easy])
    hard_fraction = sum(record["verdict"] == "investigate" for record in hard) / len(hard)
    easy_fraction = sum(record["verdict"] == "investigate" for record in easy) / len(easy)
    hard_non_adequate_fraction = sum(record["verdict"] != "adequate" for record in hard) / len(hard)
    easy_non_adequate_fraction = sum(record["verdict"] != "adequate" for record in easy) / len(easy)
    separates = (
        pearson is not None
        and hard_disk_rate["median"] is not None
        and easy_disk_rate["median"] is not None
        and float(hard_disk_rate["median"]) > float(easy_disk_rate["median"])
        and hard_non_adequate_fraction >= easy_non_adequate_fraction
    )
    return {
        "identity_records": len(identity_records),
        "identity_disk_rate_vs_iterations_pearson": pearson,
        "identity_hard_iteration_threshold": threshold,
        "identity_iteration_range": {**_summary(iterations), "ratio": ratio},
        "identity_hard_disk_rate": hard_disk_rate,
        "identity_easy_disk_rate": easy_disk_rate,
        "identity_investigate_fraction": sum(
            record["verdict"] == "investigate" for record in identity_records
        )
        / len(identity_records),
        "identity_hard_investigate_fraction": hard_fraction,
        "identity_easy_investigate_fraction": easy_fraction,
        "identity_non_adequate_fraction": sum(
            record["verdict"] != "adequate" for record in identity_records
        )
        / len(identity_records),
        "identity_hard_non_adequate_fraction": hard_non_adequate_fraction,
        "identity_easy_non_adequate_fraction": easy_non_adequate_fraction,
        "hard_easy_separation": separates,
    }


def _regime_claim(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Group actual GMRES work by the decision-procedure verdict."""
    by_verdict: dict[str, list[float]] = {}
    for record in records:
        by_verdict.setdefault(record["verdict"], []).append(
            float(record["actual_gmres"]["iterations"])
        )
    return {
        "description": "Actual counted-GMRES iterations grouped by diagnostic verdict.",
        "iteration_by_verdict": {
            verdict: _summary(values) for verdict, values in sorted(by_verdict.items())
        },
    }


def _verdict_on_decision_procedure(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the same global identity-cost hard/easy test as the PME study."""
    identity = _identity_effect(records)
    nonconvergence_cases = [
        {
            "r": record["r"],
            "analysis_dt": record["analysis_dt"],
            "d0_kind": record["d0_kind"],
            "iterations": record["actual_gmres"]["iterations"],
            "final_relative_residual": record["actual_gmres"]["final_relative_residual"],
            "disk_rate": record["disk_rate"],
            "verdict": record["verdict"],
        }
        for record in records
        if not record["actual_gmres"]["converged"]
    ]
    answer = "yes" if identity["hard_easy_separation"] else "no"
    return {
        "description": "Global identity hard/easy decision-procedure evidence.",
        **identity,
        "nonconvergence_cases": nonconvergence_cases,
        "answer": answer,
        "summary": (
            "yes: identity disk rate rises from easy to hard systems and non-adequate verdicts "
            "do not fall."
            if answer == "yes"
            else "no: the aggregated identity geometry does not separate hard and easy systems."
        ),
    }


def _reaction_effect(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Report reaction-strength-specific identity hard/easy separation evidence."""
    by_reaction: dict[float, list[dict[str, Any]]] = {}
    for record in records:
        by_reaction.setdefault(float(record["r"]), []).append(record)
    effects = []
    for r, rows in sorted(by_reaction.items()):
        effects.append({"r": r, **_identity_effect(rows)})
    robust = all(effect["hard_easy_separation"] for effect in effects)
    return {
        "description": (
            "Identity-only geometry and actual GMRES work by reaction strength; the diffusion "
            "Helmholtz preconditioner does not include this reaction term."
        ),
        "by_reaction": effects,
        "robust_to_unpreconditioned_reaction": robust,
        "answer": "yes" if robust else "no",
        "summary": (
            "yes: hard/easy identity separation survives every tested reaction strength, with "
            "indeterminate verdicts retained as non-adequate warnings."
            if robust
            else "no: at least one tested reaction strength loses identity hard/easy separation."
        ),
    }


def run_reaction_study(config: ReactionStudyConfig | None = None) -> dict[str, Any]:
    """Run the reaction-axis conditioning study and write its JSON report."""
    if config is None:
        config = ReactionStudyConfig()
    if not config.analysis_dt_values or min(config.analysis_dt_values) <= 0.0:
        raise ValueError("analysis_dt_values must contain positive step sizes")
    if not config.reaction_values or min(config.reaction_values) < 0.0:
        raise ValueError("reaction_values must contain non-negative strengths")

    started_at = perf_counter()
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    c = wave_speed(config.reference_wave_r)
    initial_state = porous_fisher_traveling_wave(
        grid.x_coords(),
        config.initial_time,
        r=config.reference_wave_r,
        c=c,
    )
    records: list[dict[str, Any]] = []
    for r in config.reaction_values:
        state, state_solver = _advance_to_visited_state(initial_state, grid, r=r, config=config)
        for analysis_dt in config.analysis_dt_values:
            for index, d0_kind in enumerate(config.d0_kinds):
                diagnostics = assess_porous_fisher_state(
                    state,
                    grid,
                    r=r,
                    dt=analysis_dt,
                    epsilon=config.epsilon,
                    d0_kind=d0_kind,
                    const_value=config.const_d0,
                    n_angles=config.n_angles,
                    fov_max_iters=config.fov_max_iters,
                    arnoldi_steps=config.arnoldi_steps,
                    seed=20260880 + 1000 * int(100 * r) + 10 * int(100 * analysis_dt) + index,
                )
                if diagnostics["adjoint_error"] > diagnostics["adjoint_tolerance"]:
                    raise RuntimeError(
                        "Porous--Fisher adjoint gate failed: "
                        f"r={r}, dt={analysis_dt}, d0_kind={d0_kind}, "
                        f"error={diagnostics['adjoint_error']}"
                    )
                actual_gmres = measure_porous_fisher_gmres_iterations(
                    state,
                    grid,
                    r=r,
                    dt=analysis_dt,
                    epsilon=config.epsilon,
                    d0_kind=d0_kind,
                    tol=config.krylov_tol,
                    max_iters=config.max_krylov_iters,
                    const_value=config.const_d0,
                )
                records.append(
                    {
                        "r": r,
                        "analysis_dt": analysis_dt,
                        "d0_kind": d0_kind,
                        "d0_used": diagnostics["d0"],
                        "sigma": float(diagnostics["d0"] * analysis_dt / grid.dx**2),
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
                        "state_solver": dict(state_solver),
                    }
                )

    runtime_seconds = perf_counter() - started_at
    identity_records = [record for record in records if record["d0_kind"] == "identity"]
    identity_iterations = [
        float(record["actual_gmres"]["iterations"]) for record in identity_records
    ]
    minimum = min(identity_iterations)
    maximum = max(identity_iterations)
    ratio = float("inf") if minimum == 0.0 and maximum > 0.0 else maximum / minimum
    if ratio < 10.0:
        raise RuntimeError(
            "Identity GMRES dynamic-range gate failed: "
            f"min={minimum}, max={maximum}, ratio={ratio}"
        )

    report = {
        "description": (
            "Experimental Porous--Fisher conditioning study. It tests whether diagnostics flag "
            "hard states when a logistic reaction remains outside the diffusion preconditioner."
        ),
        "config": config._asdict(),
        "runtime_seconds": runtime_seconds,
        "physical_model": {
            "equation": "u_t = d_xx(u**2 + epsilon**2) + r*u*(1-u)",
            "diffusivity": "D(u)=2*u",
            "preconditioner_scope": "diffusion-only D0 Helmholtz; reaction is unpreconditioned",
        },
        "gmres_measurement_note": (
            "actual_gmres is the explicit residual-history count for P^-1 J delta=P^-1(-R), "
            "not NKStats.lin_iters."
        ),
        "regime_claim": _regime_claim(records),
        "verdict_on_decision_procedure": _verdict_on_decision_procedure(records),
        "reaction_effect": _reaction_effect(records),
        "records": records,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print the compact reaction-axis conclusion."""
    verdict = report["verdict_on_decision_procedure"]
    effect = report["reaction_effect"]
    iteration_range = verdict["identity_iteration_range"]
    print(f"records={len(report['records'])} runtime_seconds={report['runtime_seconds']:.3f}")
    print(
        "identity stress: "
        f"min={iteration_range['min']:.0f} max={iteration_range['max']:.0f} "
        f"ratio={iteration_range['ratio']:.3f} "
        f"pearson={verdict['identity_disk_rate_vs_iterations_pearson']} "
        f"answer={verdict['answer']}"
    )
    for item in effect["by_reaction"]:
        item_range = item["identity_iteration_range"]
        print(
            f"r={item['r']:g}: pearson={item['identity_disk_rate_vs_iterations_pearson']} "
            f"iters={item_range['min']:.0f}/{item_range['median']:.1f}/{item_range['max']:.0f} "
            f"separates={item['hard_easy_separation']}"
        )
    print(f"reaction effect: {effect['summary']}")


def main() -> None:
    """Run the default reaction-axis study or write an explicitly supplied JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=ReactionStudyConfig().output_path)
    args = parser.parse_args()
    _print_summary(run_reaction_study(ReactionStudyConfig(output_path=args.output)))


if __name__ == "__main__":
    main()
