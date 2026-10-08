#!/usr/bin/env python3
"""Audit and benchmark batching in the experimental PME DST/tau blend.

The production implementation uses ``jax.vmap`` across its static reference
axis.  This measurement harness retains that path and supplies a deliberately
unrolled sequential comparator so the transform count, numerical equivalence,
and performance effect of batching can be measured without changing solver
logic.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from pme_breakdown import _initial_state
from pme_dst_tau_blend_single_reference_baselines import (
    BaselineConfig,
    _json_config,
    _project_physical_state,
    _quartiles,
    _stage_config,
)
from tau_blend_reproducibility import provenance_revisions

from moljax._precision import require_x64
from moljax.conditioning import linearized_operator
from moljax.core.fft_nonperiodic import solve_helmholtz_dirichlet
from moljax.core.preconditioners import PrecondContext
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import (
    _counted_gmres,
    interior_values,
    make_backward_euler_residual,
)
from moljax.experimental.pme_dst_tau_blend import (
    DSTTauBlendPreconditioner,
    measure_pme_tau_blend_gmres_iterations,
    pme_dst_tau_blend_preconditioner,
)


class AuditConfig(NamedTuple):
    """Fixed Phase-3 measurement grid."""

    nx_values: tuple[int, ...] = (512, 1024)
    m_values: tuple[int, ...] = (2, 4, 8)
    tolerances: tuple[float, ...] = (1.0e-2, 1.0e-8)
    apply_warmups: int = 10
    apply_repetitions: int = 101
    solve_warmups: int = 2
    solve_repetitions_512: int = 7
    solve_repetitions_1024: int = 3


@dataclass(frozen=True)
class SequentialDSTTauBlendPreconditioner:
    """Deliberately unrolled comparator for the production batched path."""

    base: DSTTauBlendPreconditioner
    name: str = "pme_dst_tau_blend_forced_sequential_audit"

    def apply(self, residual: jax.Array, context: Any = None) -> jax.Array:
        """Apply each reference solve separately, solely for measurement."""
        del context
        values = jnp.asarray(residual, dtype=jnp.float64)
        output = self.base.inactive_weight * values
        for index in range(self.base.reference_count):
            output = output + solve_helmholtz_dirichlet(
                self.base.input_weights[index] * values,
                self.base.laplacian_symbol,
                self.base.dt,
                self.base.reference_values[index],
            )
        return output


def _state(
    nx: int, m: int, baseline: BaselineConfig
) -> tuple[NodeCenteredDirichletGrid, jax.Array]:
    """Return the exact projected Phase-0 wide-front state."""
    grid = NodeCenteredDirichletGrid.uniform(nx, baseline.x_min, baseline.x_max)
    state, _ = _project_physical_state(
        _initial_state(grid, m, baseline.t0, baseline.halfwidth),
        _stage_config(baseline),
    )
    return grid, state


def _preconditioners(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    baseline: BaselineConfig,
) -> tuple[DSTTauBlendPreconditioner, SequentialDSTTauBlendPreconditioner]:
    """Build production and forced-sequential views of one preconditioner."""
    batched = pme_dst_tau_blend_preconditioner(
        state,
        float(m),
        baseline.analysis_dt,
        grid,
        baseline.epsilon,
        reference_count=baseline.reference_count,
        active_relative=baseline.active_relative_floor,
        active_absolute=baseline.active_absolute_floor,
    )
    return batched, SequentialDSTTauBlendPreconditioner(batched)


def _time_apply(
    batched: DSTTauBlendPreconditioner,
    sequential: SequentialDSTTauBlendPreconditioner,
    rhs: jax.Array,
    config: AuditConfig,
) -> dict[str, Any]:
    """Time synchronized applications in alternating order."""
    functions = {
        "batched": jax.jit(batched.apply),
        "forced_sequential": jax.jit(sequential.apply),
    }
    for _ in range(config.apply_warmups):
        for function in functions.values():
            jax.block_until_ready(function(rhs))
    samples: dict[str, list[float]] = {name: [] for name in functions}
    names = tuple(functions)
    for repetition in range(config.apply_repetitions):
        order = names if repetition % 2 == 0 else names[::-1]
        for name in order:
            started = perf_counter()
            jax.block_until_ready(functions[name](rhs))
            samples[name].append(perf_counter() - started)
    summaries: dict[str, Any] = {}
    for name, elapsed in samples.items():
        center, lower, upper = _quartiles(elapsed)
        summaries[name] = {
            "median_seconds": center,
            "iqr_seconds": upper - lower,
            "q1_seconds": lower,
            "q3_seconds": upper,
            "repetitions": len(elapsed),
            "warmups_excluded": config.apply_warmups,
        }
    summaries["speedup_batched_over_forced_sequential"] = (
        summaries["forced_sequential"]["median_seconds"] / summaries["batched"]["median_seconds"]
    )
    return summaries


def _equivalence(
    batched: DSTTauBlendPreconditioner,
    sequential: SequentialDSTTauBlendPreconditioner,
    nx: int,
    seed: int,
) -> dict[str, Any]:
    """Compare the two actions on deterministic, oscillatory, and random inputs."""
    x = jnp.linspace(-1.0, 1.0, nx, dtype=jnp.float64)
    vectors = {
        "linear": x,
        "oscillatory": jnp.sin(17.0 * jnp.pi * x) + 0.3 * jnp.cos(5.0 * jnp.pi * x),
        "random": jax.random.normal(jax.random.PRNGKey(seed), (nx,), dtype=jnp.float64),
    }
    relative: dict[str, float] = {}
    absolute: dict[str, float] = {}
    apply_batched = jax.jit(batched.apply)
    apply_sequential = jax.jit(sequential.apply)
    for name, vector in vectors.items():
        actual = jax.block_until_ready(apply_batched(vector))
        reference = jax.block_until_ready(apply_sequential(vector))
        difference = jnp.linalg.norm(actual - reference)
        scale = jnp.maximum(jnp.linalg.norm(reference), jnp.finfo(jnp.float64).tiny)
        relative[name] = float(difference / scale)
        absolute[name] = float(jnp.max(jnp.abs(actual - reference)))
    return {
        "relative_l2": relative,
        "maximum_absolute": absolute,
        "maximum_relative_l2": max(relative.values()),
        "maximum_absolute_difference": max(absolute.values()),
    }


def _stablehlo_audit(
    batched: DSTTauBlendPreconditioner,
    sequential: SequentialDSTTauBlendPreconditioner,
    rhs: jax.Array,
) -> dict[str, Any]:
    """Record call-graph evidence for the transform invocation counts."""
    result: dict[str, Any] = {}
    for name, function in {
        "batched": batched.apply,
        "forced_sequential": sequential.apply,
    }.items():
        ir = str(jax.jit(function).lower(rhs).compiler_ir(dialect="stablehlo"))
        solve_calls = ir.count("call @solve_helmholtz_dirichlet(")
        result[name] = {
            "top_level_helmholtz_solve_calls": solve_calls,
            "dst_calls_per_helmholtz_solve": 2,
            "transform_invocations_per_apply": 2 * solve_calls,
            "stablehlo_map_count": ir.count("stablehlo.map"),
            "stablehlo_while_count": ir.count("stablehlo.while"),
            "batched_transform_shape_present": f"tensor<{batched.reference_count}x{2 * (rhs.size + 1)}"
            in ir,
        }
    return result


def _measure_sequential_gmres(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    baseline: BaselineConfig,
) -> dict[str, float | int | bool]:
    """Run counted GMRES with the deliberately sequential comparator."""
    previous = interior_values(state, grid)
    residual = make_backward_euler_residual(
        previous,
        grid,
        float(m),
        baseline.analysis_dt,
        baseline.epsilon,
    )
    batched, sequential = _preconditioners(previous, grid, m, baseline)
    del batched
    context = PrecondContext(grid=grid, dt=baseline.analysis_dt, params={})
    operator = linearized_operator(
        residual,
        previous,
        preconditioner=sequential,
        context=context,
    )
    rhs = sequential.apply(-residual(previous), context)
    return _counted_gmres(
        operator.matvec,
        rhs,
        tol=tolerance,
        max_iters=baseline.max_krylov_iters,
    )


def _time_gmres(
    state: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: int,
    tolerance: float,
    baseline: BaselineConfig,
    config: AuditConfig,
    *,
    sequential: bool,
) -> dict[str, Any]:
    """Time fresh construction plus counted GMRES, matching Phase 0."""
    repetitions = config.solve_repetitions_1024 if grid.nx >= 1024 else config.solve_repetitions_512

    def measure() -> dict[str, float | int | bool]:
        if sequential:
            return _measure_sequential_gmres(state, grid, m, tolerance, baseline)
        return measure_pme_tau_blend_gmres_iterations(
            state,
            grid,
            float(m),
            baseline.analysis_dt,
            baseline.epsilon,
            reference_count=baseline.reference_count,
            tol=tolerance,
            max_iters=baseline.max_krylov_iters,
            active_relative=baseline.active_relative_floor,
            active_absolute=baseline.active_absolute_floor,
        )

    for _ in range(config.solve_warmups):
        warmup = measure()
        jax.block_until_ready(jnp.asarray(warmup["final_relative_residual"]))
    elapsed: list[float] = []
    measurements: list[dict[str, float | int | bool]] = []
    for _ in range(repetitions):
        started = perf_counter()
        measured = measure()
        jax.block_until_ready(jnp.asarray(measured["final_relative_residual"]))
        elapsed.append(perf_counter() - started)
        measurements.append(measured)
    center, lower, upper = _quartiles(elapsed)
    return {
        "median_seconds": center,
        "iqr_seconds": upper - lower,
        "q1_seconds": lower,
        "q3_seconds": upper,
        "repetitions": repetitions,
        "warmups_excluded": config.solve_warmups,
        "iterations": sorted({int(row["iterations"]) for row in measurements}),
        "all_converged": all(bool(row["converged"]) for row in measurements),
        "max_final_relative_residual": max(
            float(row["final_relative_residual"]) for row in measurements
        ),
    }


def _phase0_record(phase0: dict[str, Any], nx: int, m: int, tolerance: float) -> dict[str, Any]:
    """Select one completed Phase-0 work-precision record."""
    return next(
        row
        for row in phase0["work_precision_records"]
        if int(row["nx"]) == nx
        and int(row["m"]) == m
        and float(row["requested_relative_residual"]) == tolerance
    )


def _best_scalar(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return Phase 0's fastest converged scalar competitor."""
    candidates = [
        (float(values["median_seconds"]), name, values)
        for name, values in record["methods"].items()
        if name != "tau_blend" and bool(values["all_converged"])
    ]
    if not candidates:
        return None
    elapsed, name, values = min(candidates)
    return {
        "name": name,
        "median_seconds": elapsed,
        "iqr_seconds": float(values["iqr_seconds"]),
        "iterations": list(values["iterations"]),
    }


def run(
    output: Path,
    phase0_path: Path,
    *,
    apply_only: bool,
) -> dict[str, Any]:
    """Run the audit on the process-selected JAX backend."""
    config = AuditConfig()
    baseline = BaselineConfig()
    phase0 = json.loads(phase0_path.read_text(encoding="utf-8"))
    if phase0["config"] != _json_config(baseline):
        raise ValueError("Phase-0 result uses an unexpected configuration")
    report: dict[str, Any] = {
        "schema": "pme_dst_tau_blend_batching_audit_v1",
        "provenance": provenance_revisions(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "production_path": "jax.vmap over the static reference axis",
        "symbols_precomputed": False,
        "symbol_note": (
            "laplacian_symbol and reference values are stored, but the (l,N) "
            "Helmholtz denominator is broadcast/recomputed inside each apply"
        ),
        "reference_count_static_shape": baseline.reference_count,
        "config": json.loads(json.dumps(config._asdict())),
        "apply_records": [],
        "solve_records": [],
    }
    for nx in config.nx_values:
        for m in config.m_values:
            grid, state = _state(nx, m, baseline)
            batched, sequential = _preconditioners(state, grid, m, baseline)
            rhs = jax.random.normal(
                jax.random.PRNGKey(30_000 + nx + m),
                (nx,),
                dtype=jnp.float64,
            )
            row = {
                "nx": nx,
                "m": m,
                "reference_count": batched.reference_count,
                "stablehlo": _stablehlo_audit(batched, sequential, rhs),
                "equivalence": _equivalence(batched, sequential, nx, 40_000 + nx + m),
                "timing": _time_apply(batched, sequential, rhs, config),
            }
            report["apply_records"].append(row)
            print(
                f"APPLY backend={report['backend']} nx={nx} m={m} "
                f"speedup={row['timing']['speedup_batched_over_forced_sequential']:.3f} "
                f"max_rel={row['equivalence']['maximum_relative_l2']:.3e}",
                flush=True,
            )
    if not apply_only:
        for nx in config.nx_values:
            for m in config.m_values:
                grid, state = _state(nx, m, baseline)
                for tolerance in config.tolerances:
                    old = _phase0_record(phase0, nx, m, tolerance)
                    batched = _time_gmres(
                        state,
                        grid,
                        m,
                        tolerance,
                        baseline,
                        config,
                        sequential=False,
                    )
                    sequential = _time_gmres(
                        state,
                        grid,
                        m,
                        tolerance,
                        baseline,
                        config,
                        sequential=True,
                    )
                    best = _best_scalar(old)
                    old_blend = old["methods"]["tau_blend"]
                    row = {
                        "nx": nx,
                        "m": m,
                        "requested_relative_residual": tolerance,
                        "phase0_batched": old_blend,
                        "rerun_batched": batched,
                        "forced_sequential": sequential,
                        "best_phase0_scalar": best,
                        "full_solve_speedup_batched_over_forced_sequential": (
                            sequential["median_seconds"] / batched["median_seconds"]
                        ),
                        "iteration_sets_identical": (
                            list(batched["iterations"])
                            == list(sequential["iterations"])
                            == list(old_blend["iterations"])
                        ),
                    }
                    if best is not None:
                        row["phase0_advantage_percent_vs_best_scalar"] = (
                            100.0
                            * (best["median_seconds"] - old_blend["median_seconds"])
                            / best["median_seconds"]
                        )
                        row["rerun_advantage_percent_vs_best_scalar"] = (
                            100.0
                            * (best["median_seconds"] - batched["median_seconds"])
                            / best["median_seconds"]
                        )
                    report["solve_records"].append(row)
                    print(
                        f"SOLVE nx={nx} m={m} tol={tolerance:.1e} "
                        f"batched={batched['median_seconds']:.6f}s "
                        f"sequential={sequential['median_seconds']:.6f}s "
                        f"iters={batched['iterations']}",
                        flush=True,
                    )
    report["maximum_relative_action_difference"] = max(
        row["equivalence"]["maximum_relative_l2"] for row in report["apply_records"]
    )
    report["all_solve_iteration_sets_identical"] = (
        all(row["iteration_sets_identical"] for row in report["solve_records"])
        if report["solve_records"]
        else None
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return report


def main() -> None:
    """Parse CLI arguments and run the measurement."""
    require_x64("PME DST/tau-blend batching audit")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/pme_dst_tau_blend_batching_audit_gpu.json"),
    )
    parser.add_argument(
        "--phase0",
        type=Path,
        default=Path("benchmarks/results/pme_dst_tau_blend_single_reference_baselines.json"),
    )
    parser.add_argument("--apply-only", action="store_true")
    args = parser.parse_args()
    report = run(args.output, args.phase0, apply_only=args.apply_only)
    print(
        f"COMPLETE backend={report['backend']} output={args.output} "
        f"max_rel={report['maximum_relative_action_difference']:.3e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
