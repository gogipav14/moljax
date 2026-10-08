#!/usr/bin/env python3
"""Measure an experimental DST/tau blend on variable-coefficient PME systems."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax

# Enable float64 before importing JAX-dependent benchmark modules.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from pme_breakdown import _initial_state
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import assess_pme_state, measure_gmres_iterations
from moljax.experimental.pme_dst_tau_blend import (
    assess_pme_tau_blend_state,
    measure_pme_tau_blend_gmres_iterations,
    pme_dst_tau_blend_preconditioner,
    porous_medium_diffusivity,
)


class TauBlendConfig(NamedTuple):
    """Configuration for the compact first-Newton-state DST/tau study."""

    nx: int = 512
    x_min: float = -4.0
    x_max: float = 4.0
    t0: float = 0.1
    epsilon: float = 1.0e-5
    reference_counts: tuple[int, ...] = (3, 5)
    n_angles: int = 3
    fov_max_iters: int = 8
    arnoldi_steps: int = 6
    krylov_tol: float = 1.0e-8
    max_krylov_iters: int = 400
    output_path: str = "benchmarks/results/pme_dst_tau_blend_poc.json"


CASES = (
    ("linear_control", 1, 3.0, 2.0e-2),
    ("m2_narrow", 2, 0.25, 2.0e-2),
    ("m2_wide", 2, 3.0, 2.0e-2),
    ("m4_wide", 4, 3.0, 2.0e-2),
    ("m8_wide", 8, 3.0, 2.0e-2),
    ("m8_wide_stiff", 8, 3.0, 2.0),
)


def _coefficient_summary(
    state: jax.Array,
    m: int,
    epsilon: float,
) -> dict[str, float | int | str]:
    """Describe the degenerate coefficient without hiding its zero region."""
    if m == 1:
        diffusivity = jnp.ones_like(state)
    else:
        diffusivity = porous_medium_diffusivity(state, float(m), epsilon)
    positive = diffusivity[diffusivity > 0.0]
    minimum = float(jnp.min(diffusivity))
    maximum = float(jnp.max(diffusivity))
    return {
        "minimum": minimum,
        "maximum": maximum,
        "literal_contrast": "infinite" if minimum == 0.0 and maximum > 0.0 else maximum / minimum,
        "active_node_count": int(positive.size),
        "active_p95_over_p05": float(jnp.quantile(positive, 0.95) / jnp.quantile(positive, 0.05)),
    }


def _baseline_measurement(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    dt: float,
    epsilon: float,
    variant: str,
    config: TauBlendConfig,
    seed: int,
) -> dict[str, Any]:
    """Measure one existing frozen or identity reference system."""
    gmres = measure_gmres_iterations(
        state,
        grid,
        float(m),
        dt,
        epsilon,
        variant,
        tol=config.krylov_tol,
        max_iters=config.max_krylov_iters,
    )
    if variant == "identity":
        return {"actual_gmres": gmres}
    diagnostic = assess_pme_state(
        state,
        grid,
        float(m),
        dt,
        epsilon,
        variant,
        n_angles=config.n_angles,
        fov_max_iters=config.fov_max_iters,
        arnoldi_steps=config.arnoldi_steps,
        seed=seed,
    )
    return {"diagnostic": diagnostic, "actual_gmres": gmres}


def _tau_measurement(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    dt: float,
    epsilon: float,
    reference_count: int,
    config: TauBlendConfig,
    seed: int,
) -> dict[str, Any]:
    """Measure one reference-count setting of the staged DST/tau blend."""
    preconditioner = pme_dst_tau_blend_preconditioner(
        state,
        float(m),
        dt,
        grid,
        epsilon,
        reference_count=reference_count,
    )
    diagnostic = assess_pme_tau_blend_state(
        state,
        grid,
        float(m),
        dt,
        epsilon,
        reference_count=reference_count,
        n_angles=config.n_angles,
        fov_max_iters=config.fov_max_iters,
        arnoldi_steps=config.arnoldi_steps,
        seed=seed,
    )
    gmres = measure_pme_tau_blend_gmres_iterations(
        state,
        grid,
        float(m),
        dt,
        epsilon,
        reference_count=reference_count,
        tol=config.krylov_tol,
        max_iters=config.max_krylov_iters,
    )
    return {
        "reference_values": [float(value) for value in preconditioner.reference_values],
        "active_nodes": int(jnp.sum(preconditioner.inactive_weight == 0.0)),
        "diagnostic": diagnostic,
        "actual_gmres": gmres,
    }


def run_tau_blend_study(config: TauBlendConfig | None = None) -> dict[str, Any]:
    """Regenerate the small variable-coefficient DST/tau comparison table."""
    if config is None:
        config = TauBlendConfig()
    if not config.reference_counts or min(config.reference_counts) < 1:
        raise ValueError("reference_counts must contain positive values")

    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    started = perf_counter()
    records: list[dict[str, Any]] = []
    for case_index, (label, m, halfwidth, dt) in enumerate(CASES):
        state = _initial_state(grid, m, config.t0, halfwidth)
        epsilon = 0.0 if m == 1 else config.epsilon
        baselines = {
            variant: _baseline_measurement(
                state,
                grid,
                m,
                dt,
                epsilon,
                variant,
                config,
                seed=20261030 + 100 * case_index + offset,
            )
            for offset, variant in enumerate(("frozen_mean", "frozen_bulk", "identity"))
        }
        tau_blends = {
            str(reference_count): _tau_measurement(
                state,
                grid,
                m,
                dt,
                epsilon,
                reference_count,
                config,
                seed=20261130 + 100 * case_index + reference_count,
            )
            for reference_count in config.reference_counts
        }
        identity_iterations = baselines["identity"]["actual_gmres"]["iterations"]
        frozen_mean_iterations = baselines["frozen_mean"]["actual_gmres"]["iterations"]
        frozen_bulk_iterations = baselines["frozen_bulk"]["actual_gmres"]["iterations"]
        best_baseline_iterations = min(
            identity_iterations,
            frozen_mean_iterations,
            frozen_bulk_iterations,
        )
        records.append(
            {
                "case": label,
                "m": m,
                "target_support_halfwidth": halfwidth,
                "analysis_dt": dt,
                "state_kind": "initial_barenblatt_first_newton_system",
                "coefficient": _coefficient_summary(state, m, epsilon),
                "baselines": baselines,
                "dst_tau_blends": tau_blends,
                "beats_identity": {
                    str(reference_count): values["actual_gmres"]["iterations"] < identity_iterations
                    for reference_count, values in tau_blends.items()
                },
                "beats_all_baselines": {
                    str(reference_count): values["actual_gmres"]["iterations"]
                    < best_baseline_iterations
                    for reference_count, values in tau_blends.items()
                },
            }
        )

    target_records = [record for record in records if "wide" in str(record["case"])]
    success_by_reference_count = {
        str(reference_count): all(
            record["beats_all_baselines"][str(reference_count)] for record in target_records
        )
        for reference_count in config.reference_counts
    }
    report = {
        "schema": "pme_dst_tau_blend_poc_v1",
        "provenance": provenance_revisions(),
        "scope": (
            "Experimental input-weighted DST/tau blend for first Newton systems at "
            "Barenblatt initial states; it is not a public solver component."
        ),
        "design": {
            "operator": "I - dt * L_D * diag(D(u))",
            "ordering": "input weights approximate the right diagonal coefficient ordering",
            "zero_coefficient_policy": (
                "D <= max(1e-6 * max(D), 1e-14) retains an identity source term and "
                "does not select a zero-coefficient reference inverse."
            ),
            "cost": "reference_count DST-I Helmholtz solves plus O(reference_count * N) work",
        },
        "config": config._asdict(),
        "runtime_seconds": perf_counter() - started,
        "records": records,
        "success_bar": (
            "tau GMRES iterations must be lower than identity, frozen_mean, and frozen_bulk "
            "on every wide-front target"
        ),
        "success_by_reference_count": success_by_reference_count,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    """Run the compact DST/tau blend comparison."""
    require_x64("PME DST/tau-blend proof-of-concept benchmark")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=TauBlendConfig().output_path)
    args = parser.parse_args()
    report = run_tau_blend_study(TauBlendConfig(output_path=args.output))
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    for record in report["records"]:
        identity = record["baselines"]["identity"]["actual_gmres"]["iterations"]
        values = [
            f"l={count}:{row['actual_gmres']['iterations']}"
            for count, row in record["dst_tau_blends"].items()
        ]
        print(f"{record['case']}: identity={identity}; " + ", ".join(values))


if __name__ == "__main__":
    main()
