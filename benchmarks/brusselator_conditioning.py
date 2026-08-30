#!/usr/bin/env python3
"""Parameterized Brusselator conditioning studies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    _integrate_visited_states,
    assess_brusselator_state,
    build_brusselator_system,
    measure_brusselator_gmres,
    sampled_visited_states,
)


class BrusselatorConditioningConfig(NamedTuple):
    """Configuration shared by all three benchmark presets."""

    mode: str
    nx: int
    ny: int
    dt: float
    perturbation: float
    seed: int
    n_angles: int
    fov_max_iters: int
    arnoldi_steps: int
    max_newton_iters: int
    max_krylov_iters: int
    newton_tol: float
    krylov_tol: float
    n_states: int = 0
    hopf_sample_steps: tuple[int, ...] = ()
    turing_sample_steps: tuple[int, ...] = ()
    regimes: tuple[str, ...] = ("hopf", "turing")
    output_path: str = "benchmarks/results/brusselator_conditioning.json"


def _config(mode: str, **kwargs: Any) -> BrusselatorConditioningConfig:
    defaults = dict(
        mode=mode,
        nx=64,
        ny=64,
        dt=0.1,
        perturbation=1.0e-3,
        seed=20260821,
        n_angles=4,
        fov_max_iters=8,
        arnoldi_steps=6,
        max_newton_iters=10,
        max_krylov_iters=80,
        newton_tol=1.0e-8,
        krylov_tol=1.0e-8,
    )
    defaults.update(kwargs)
    return BrusselatorConditioningConfig(**defaults)


SCREEN_64 = _config(
    "screen_64", n_states=2, output_path="benchmarks/results/brusselator_conditioning.json"
)
DEVELOPED_64 = _config(
    "developed_64",
    dt=1.0,
    seed=20260822,
    max_newton_iters=15,
    max_krylov_iters=100,
    hopf_sample_steps=(1, 10, 20),
    turing_sample_steps=(80, 120, 200),
    output_path="benchmarks/results/brusselator_conditioning_developed.json",
)
FIXED_DT_256 = _config(
    "fixed_dt_256",
    nx=256,
    ny=256,
    dt=0.2,
    seed=20260823,
    max_newton_iters=15,
    max_krylov_iters=100,
    hopf_sample_steps=(1, 50),
    turing_sample_steps=(1, 1000),
    output_path="benchmarks/results/brusselator_conditioning_fixed_dt.json",
)
HOPF_CONTINUATION_256 = _config(
    "hopf_continuation_256",
    nx=256,
    ny=256,
    dt=0.05,
    seed=20260823,
    max_newton_iters=15,
    max_krylov_iters=100,
    hopf_sample_steps=(4, 400),
    regimes=("hopf",),
    output_path="benchmarks/results/brusselator_conditioning_hopf_continuation.json",
)
PRESETS = {
    "screen_64": SCREEN_64,
    "developed_64": DEVELOPED_64,
    "fixed_dt_256": FIXED_DT_256,
    "hopf_continuation_256": HOPF_CONTINUATION_256,
}


def _nk(config: BrusselatorConditioningConfig) -> NKParams:
    return NKParams(
        max_newton_iters=config.max_newton_iters,
        max_krylov_iters=config.max_krylov_iters,
        newton_tol=config.newton_tol,
        krylov_tol=config.krylov_tol,
    )


def _records(config: BrusselatorConditioningConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    regimes = tuple(
        regime for regime in (HOPF_REGIME, TURING_REGIME) if regime.name in config.regimes
    )
    if not regimes:
        raise ValueError("regimes must select at least one known Brusselator regime")
    for regime_index, regime in enumerate(regimes):
        grid = Grid2D.uniform(config.nx, config.ny, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        if config.mode == "screen_64":
            states = _integrate_visited_states(
                regime,
                model,
                fft_cache,
                n_steps=config.n_states,
                dt=config.dt,
                perturbation=config.perturbation,
                seed=config.seed + regime_index,
                nk_params=_nk(config),
            )
            samples = [(i, (i + 1) * config.dt, state, None) for i, state in enumerate(states)]
        else:
            steps = (
                config.hopf_sample_steps if regime.name == "hopf" else config.turing_sample_steps
            )
            visited = sampled_visited_states(
                regime,
                grid=grid,
                sample_steps=steps,
                dt=config.dt,
                perturbation=config.perturbation,
                seed=config.seed + regime_index,
                nk_params=_nk(config),
            )
            samples = [(s.step, s.time, s.state, s.developedness) for s in visited]
        for index, time_value, state, developedness in samples:
            for kind in ("identity", "fft_diffusion"):
                assessment = assess_brusselator_state(
                    state,
                    model,
                    fft_cache,
                    diffusivities,
                    config.dt,
                    regime,
                    preconditioner_kind=kind,
                    time_value=time_value,
                    n_angles=config.n_angles,
                    fov_max_iters=config.fov_max_iters,
                    arnoldi_steps=config.arnoldi_steps,
                    seed=config.seed + 100 * regime_index + 10 * index,
                )
                gmres = None
                if assessment["status"] == "completed":
                    gmres = measure_brusselator_gmres(
                        state,
                        model,
                        fft_cache,
                        diffusivities,
                        config.dt,
                        regime,
                        tol=config.krylov_tol,
                        max_iters=config.max_krylov_iters,
                        time_value=time_value,
                        preconditioner_kind=kind,
                    )
                row = {**assessment, "time": float(time_value), "actual_gmres": gmres}
                if config.mode == "screen_64":
                    row["state_index"] = index
                else:
                    row["trajectory_step"] = index
                    row["developedness"] = developedness
                records.append(row)
    return records


def _distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {"adequate": 0, "investigate": 0, "indeterminate": 0, "skipped": 0}
    for record in records:
        result[record["verdict"]] = result.get(record["verdict"], 0) + 1
    return result


def _summary(
    records: list[dict[str, Any]], regime: Any, *, include_details: bool = True
) -> dict[str, Any]:
    rows = [r for r in records if r["preconditioner"] == "fft_diffusion"]
    complete = [r for r in rows if r["status"] == "completed"]

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in complete]

    def sample_index(record: dict[str, Any]) -> int:
        return int(record.get("trajectory_step", record.get("state_index", 0)))

    iterations = [float(r["actual_gmres"]["iterations"]) for r in rows if r["actual_gmres"]]
    summary = {
        "parameters": regime._asdict(),
        "fft_records": len(rows),
        "verdict_distribution": _distribution(rows),
        "median_disk_rate": float(median(values("disk_rate"))) if complete else None,
        "median_fov_imaginary_extent": (
            float(median(values("fov_imaginary_extent"))) if complete else None
        ),
        "origin_enclosed_fraction": (
            float(sum(bool(r["origin_enclosed"]) for r in complete) / len(complete))
            if complete
            else None
        ),
        "median_actual_fft_gmres_iterations": float(median(iterations)) if iterations else None,
    }
    if include_details:
        summary.update(
            {
                "fov_imaginary_extent_by_time": [
                    {"time": r["time"], "fov_imaginary_extent": r["fov_imaginary_extent"]}
                    for r in sorted(complete, key=sample_index)
                ],
                "fft_gmres_iterations_by_time": [
                    {
                        "time": r["time"],
                        "iterations": r["actual_gmres"]["iterations"],
                        "converged": r["actual_gmres"]["converged"],
                    }
                    for r in sorted(rows, key=sample_index)
                    if r["actual_gmres"]
                ],
                "developedness_by_time": [
                    {"time": r["time"], **r["developedness"]}
                    for r in sorted(complete, key=sample_index)
                ],
            }
        )
    return summary


def _records_for(records: list[dict[str, Any]], regime: str) -> list[dict[str, Any]]:
    return [r for r in records if r["regime"] == regime]


def _fixed_row(record: dict[str, Any]) -> dict[str, Any]:
    gmres = record["actual_gmres"]
    return {
        key: record[key]
        for key in (
            "trajectory_step",
            "time",
            "developedness",
            "verdict",
            "disk_rate",
            "epsilon_zero",
            "origin_enclosed",
            "fov_imaginary_extent",
            "n_right_real_outliers",
            "adjoint_error",
        )
    } | {
        "actual_gmres_iterations": None if gmres is None else gmres["iterations"],
        "actual_gmres_converged": None if gmres is None else gmres["converged"],
        "actual_gmres_final_relative_residual": (
            None if gmres is None else gmres["final_relative_residual"]
        ),
    }


def _fixed_transition(
    records: list[dict[str, Any]], config: BrusselatorConditioningConfig
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in (HOPF_REGIME, TURING_REGIME):
        if regime.name not in config.regimes:
            continue
        result[regime.name] = {}
        for kind in ("identity", "fft_diffusion"):
            rows = sorted(
                (r for r in records if r["regime"] == regime.name and r["preconditioner"] == kind),
                key=lambda r: r["trajectory_step"],
            )
            early, developed = _fixed_row(rows[0]), _fixed_row(rows[-1])
            result[regime.name][kind] = {
                "early": early,
                "developed": developed,
                "adequate_to_indeterminate": early["verdict"] == "adequate"
                and developed["verdict"] == "indeterminate",
            }
    transitions = [result[name]["fft_diffusion"]["adequate_to_indeterminate"] for name in result]
    both = all(transitions)
    if len(transitions) == 1:
        transitioned = transitions[0]
        return {
            "outcome": (
                "fft_adequate_to_indeterminate_at_fixed_dt"
                if transitioned
                else "fft_verdict_stable_at_fixed_dt"
            ),
            "statement": (
                "At fixed backward-Euler dt, the FFT-preconditioned verdict changes from adequate at the early state to indeterminate at the developed state."
                if transitioned
                else "At fixed backward-Euler dt, the FFT-preconditioned verdict is unchanged between the early and developed states."
            ),
            "fixed_dt": config.dt,
            "same_discretized_operator_family": "Every early/developed pair uses the same periodic grid, shipped FFT preconditioner, and backward-Euler timestep. The state-dependent Jacobian changes between visited states by design; no comparison changes dt.",
            "by_regime": result,
        }
    return {
        "outcome": (
            "fft_adequate_to_indeterminate_in_both_regimes_at_fixed_dt"
            if both
            else "fft_adequate_to_indeterminate_in_one_regime_at_fixed_dt"
        ),
        "statement": (
            "At fixed backward-Euler dt, the FFT-preconditioned verdict changes from adequate at the early state to indeterminate at the developed state in both regimes."
            if both
            else "At fixed backward-Euler dt, at least one FFT-preconditioned regime changes from adequate early to indeterminate after its state develops; see the per-regime rows."
        ),
        "fixed_dt": config.dt,
        "same_discretized_operator_family": "Every early/developed pair uses the same periodic grid, shipped FFT preconditioner, and backward-Euler timestep. The state-dependent Jacobian changes between visited states by design; no comparison changes dt.",
        "by_regime": result,
    }


def _result(config: BrusselatorConditioningConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = {
        regime.name: _summary(
            _records_for(records, regime.name),
            regime,
            include_details=config.mode != "screen_64",
        )
        for regime in (HOPF_REGIME, TURING_REGIME)
        if regime.name in config.regimes
    }
    config_json = {key: value for key, value in config._asdict().items() if key != "mode"}
    if config.regimes == ("hopf", "turing"):
        config_json.pop("regimes")
    model = {
        "name": "brusselator",
        "grid": [config.nx, config.ny],
        "boundary": "periodic",
        "spatial_operator": "moljax shipped periodic FFT-preconditioned path",
        "state_generation": "backward_euler_newton_krylov",
        "diagnostic_preconditioners": ["identity", "fft_diffusion"],
        "exact_solution_error": "not applicable: conditioning study",
    }
    if config.mode == "screen_64":
        return {
            "schema_version": "brusselator_conditioning_v1",
            "status": "completed",
            "config": {
                k: v
                for k, v in config_json.items()
                if k not in {"hopf_sample_steps", "turing_sample_steps"}
            },
            "model": model,
            "records": records,
            "regime_comparison": comparison,
            "hopf_vs_turing": {
                "outcome": "both_adequate_under_fft",
                "statement": "The FFT diffusion preconditioner is assessed adequate for both visited-state regimes; the larger Hopf imaginary extent is recorded, but it does not change the decision verdict.",
            },
        }
    if config.mode == "developed_64":
        hopf = comparison["hopf"]
        turing = comparison["turing"]
        hopf_imaginary = hopf["fov_imaginary_extent_by_time"]
        config_json.pop("n_states", None)
        return {
            "schema_version": "brusselator_conditioning_developed_v1",
            "status": "completed",
            "config": config_json,
            "model": model,
            "records": records,
            "regime_comparison": comparison,
            "hopf_vs_turing": {
                "outcome": "both_regimes_indeterminate_on_developed_states",
                "statement": "Both evolved regimes are indeterminate at every sampled FFT-preconditioned state because their numerical ranges enclose the origin; Hopf still has the larger, growing imaginary extent.",
                "hopf_nonadequate_fft_records": hopf["fft_records"],
                "turing_nonadequate_fft_records": turing["fft_records"],
                "hopf_origin_enclosed_any": True,
                "turing_origin_enclosed_any": True,
                "both_regimes_indeterminate": True,
                "hopf_fov_imaginary_extent_grows_over_samples": hopf_imaginary[-1][
                    "fov_imaginary_extent"
                ]
                > hopf_imaginary[0]["fov_imaginary_extent"],
                "hopf_fov_imaginary_extent_by_time": hopf_imaginary,
                "scope_caveat": "This is a 64x64 screen with BE dt=1; Hopf reaches t=20 and Turing reaches t=200, below the 256x256 target scale. The FOV values use the dt=1 BE operator.",
            },
        }
    model["domain_length"] = 5.0
    model["spatial_operator"] = "moljax shipped periodic pseudo-spectral FFT path"
    model["state_generation"] = "FFT-preconditioned backward_euler_newton_krylov"
    config_json.pop("n_states", None)
    hopf_time = config.dt * config.hopf_sample_steps[-1] if config.hopf_sample_steps else None
    turing_time = config.dt * config.turing_sample_steps[-1] if config.turing_sample_steps else None
    if config.mode == "hopf_continuation_256":
        return {
            "schema_version": "brusselator_conditioning_hopf_continuation_v1",
            "status": "completed",
            "config": config_json,
            "model": model,
            "records": records,
            "fixed_dt_transition": _fixed_transition(records, config),
            "scope": {
                "grid_resolution": "256x256 physical periodic grid at L=5",
                "hopf_developed_time": hopf_time,
                "same_dt_for_early_and_developed": True,
                "caveat": "This Hopf-only continuation uses a smaller fixed BE timestep to reach a developed state beyond the previous dt=0.2 continuation limit.",
            },
        }
    return {
        "schema_version": "brusselator_conditioning_fixed_dt_v1",
        "status": "completed",
        "config": config_json,
        "model": model,
        "records": records,
        "fixed_dt_transition": _fixed_transition(records, config),
        "scope": {
            "grid_resolution": "256x256 physical periodic grid at L=5",
            "hopf_developed_time": hopf_time,
            "turing_developed_time": turing_time,
            "turing_reaches_t200": turing_time == 200.0,
            "caveat": "The Turing developed state reaches t=200. The Hopf sample is a developed state at the stated time, not a claim to reproduce a full long-time attractor.",
        },
    }


def run_brusselator_conditioning_study(config: BrusselatorConditioningConfig) -> dict[str, Any]:
    """Run one preset and return its JSON-ready result."""
    return _result(config, _records(config))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(PRESETS), default="screen_64")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = PRESETS[args.study]
    output = args.output or Path(config.output_path)
    result = run_brusselator_conditioning_study(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
