"""Decision-procedure demo for a diffusion-dominated periodic model.

The demo integrates a small Gray--Scott system with backward Euler and the
moljax FFT diffusion preconditioner.  It then applies the matrix-free
conditioning diagnostics to states actually visited by those implicit steps.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from moljax.conditioning import (
    adjoint_identity,
    arnoldi,
    assess_preconditioner,
    epsilon_zero,
    estimate_rates,
    linearized_operator,
    numerical_range,
    plot_numerical_range,
    plot_pseudospectrum,
    plot_residual_envelope,
    reduced_pseudospectrum,
    ritz_values,
)
from moljax.core.grid import Grid2D
from moljax.core.model import create_gray_scott_periodic_fft
from moljax.core.newton_krylov import NKParams, create_implicit_residual
from moljax.core.preconditioners import (
    IdentityPreconditioner,
    PrecondContext,
    create_gray_scott_fft_preconditioner,
)
from moljax.core.stepping import be_step


class DemoConfig(NamedTuple):
    """Static configuration for the importable decision demo."""

    nx: int = 64
    ny: int = 64
    domain_length: float = 2.5
    du: float = 0.16
    dv: float = 0.08
    feed: float = 0.01
    kill: float = 0.01
    dt: float = 0.5
    n_states: int = 3
    n_angles: int = 8
    fov_max_iters: int = 120
    arnoldi_steps: int = 8
    pseudospectrum_points: int = 5
    overhead_runs: int = 50
    max_newton_iters: int = 10
    max_krylov_iters: int = 18
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-7
    seed: int = 20260819
    figure_dir: str | None = "benchmarks/figures"


class FigureData(NamedTuple):
    """Already-computed diagnostic data used to render one report figure family."""

    state_index: int
    preconditioner: str
    fov: Any
    ritz: jax.Array
    real_grid: jax.Array
    imag_grid: jax.Array
    sigma_min: jax.Array
    predicted_gmres_factor: float


def _ready_state(state: dict[str, jax.Array]) -> dict[str, jax.Array]:
    """Block until every field in a state has completed execution."""
    return jax.tree_util.tree_map(jax.block_until_ready, state)


def _initial_condition(grid: Grid2D) -> dict[str, jax.Array]:
    """Return a periodic Gray--Scott perturbation with a small deterministic seed."""
    x, y = grid.meshgrid(include_ghost=True)
    u = jnp.ones((grid.ny_total, grid.nx_total), dtype=jnp.float64)
    v = jnp.zeros((grid.ny_total, grid.nx_total), dtype=jnp.float64)
    cx = 0.5 * (grid.x_min + grid.x_max)
    cy = 0.5 * (grid.y_min + grid.y_max)
    patch = (jnp.abs(x - cx) < 0.12) & (jnp.abs(y - cy) < 0.12)
    u = jnp.where(patch, 0.50, u)
    v = jnp.where(patch, 0.25, v)
    key_u, key_v = jax.random.split(jax.random.PRNGKey(7))
    u = u + 1.0e-3 * jax.random.uniform(key_u, u.shape, minval=-1.0, maxval=1.0)
    v = v + 1.0e-3 * jax.random.uniform(key_v, v.shape, minval=-1.0, maxval=1.0)
    return {"u": u, "v": v}


def _summary_statistics(values: list[float]) -> dict[str, float | list[float]]:
    """Return median and IQR summary for wall-clock samples."""
    samples = np.asarray(values, dtype=np.float64)
    q25, median, q75 = np.percentile(samples, [25.0, 50.0, 75.0])
    return {
        "median_s": float(median),
        "q25_s": float(q25),
        "q75_s": float(q75),
        "iqr_s": float(q75 - q25),
        "min_s": float(np.min(samples)),
        "max_s": float(np.max(samples)),
    }


def _measure_operator_action(
    action: Any,
    probe: jax.Array,
    runs: int,
) -> dict[str, float | list[float]]:
    """Measure one warmed operator action with explicit synchronization."""
    if runs < 1:
        raise ValueError("overhead_runs must be positive")
    jax.block_until_ready(action(probe))
    timings: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        result = action(probe)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - started)
    return _summary_statistics(timings)


def _spectral_grid(ritz: jax.Array, points: int) -> tuple[jax.Array, jax.Array]:
    """Build a small reduced pseudospectrum grid around the Ritz values."""
    values = np.asarray(ritz, dtype=np.complex128)
    real_span = max(float(np.ptp(values.real)), 0.25)
    imag_span = max(float(np.ptp(values.imag)), 0.25)
    real_mid = float(np.mean(values.real))
    imag_mid = float(np.mean(values.imag))
    return (
        jnp.linspace(real_mid - real_span, real_mid + real_span, points),
        jnp.linspace(imag_mid - imag_span, imag_mid + imag_span, points),
    )


def _run_state_diagnostics(
    state: dict[str, jax.Array],
    model: Any,
    preconditioner: Any,
    preconditioner_name: str,
    config: DemoConfig,
    state_index: int,
    time_value: float,
) -> tuple[dict[str, Any], FigureData | None]:
    """Run the gate and all matrix-free diagnostics for one visited state."""
    residual_fn = create_implicit_residual(
        model,
        state,
        time_value + config.dt,
        config.dt,
        method="be",
    )
    context = PrecondContext(grid=model.grid, dt=config.dt, params=model.params)
    operator = linearized_operator(
        residual_fn,
        state,
        preconditioner=preconditioner,
        context=context,
    )
    identity_error = adjoint_identity(
        operator,
        jax.random.PRNGKey(config.seed + state_index),
        operator.n,
    )
    common = {
        "state_index": state_index,
        "time": float(time_value),
        "preconditioner": preconditioner_name,
        "operator_dimension": operator.n,
        "adjoint_identity": identity_error,
        "adjoint_tolerance": 1.0e-8,
    }
    if identity_error > 1.0e-8:
        return {**common, "status": "adjoint_failed", "verdict": "skipped"}, None

    counter = {"matvec": 0, "adjoint": 0}

    def counted_matvec(value: jax.Array) -> jax.Array:
        counter["matvec"] += 1
        return operator.matvec(value)

    def counted_adjoint(value: jax.Array) -> jax.Array:
        counter["adjoint"] += 1
        return operator.matvec_adjoint(value)

    key = jax.random.PRNGKey(config.seed + 1000 + state_index)
    key_real, key_imag = jax.random.split(key)
    v0 = jax.random.normal(key_real, (operator.n,), dtype=jnp.float64)
    v0 = v0 + 1j * jax.random.normal(key_imag, (operator.n,), dtype=jnp.float64)

    diagnostic_started = time.perf_counter()
    q_basis, hessenberg = arnoldi(
        counted_matvec,
        v0,
        min(config.arnoldi_steps, operator.n),
    )
    del q_basis
    ritz = ritz_values(hessenberg)
    epsilon = epsilon_zero(hessenberg)
    real_grid, imag_grid = _spectral_grid(ritz, config.pseudospectrum_points)
    reduced = reduced_pseudospectrum(hessenberg, real_grid, imag_grid)
    fov = numerical_range(
        counted_matvec,
        counted_adjoint,
        operator.n,
        n_angles=config.n_angles,
        max_iters=config.fov_max_iters,
    )
    rates = estimate_rates(fov, ritz)
    assessment = assess_preconditioner(fov, ritz, epsilon)
    jax.block_until_ready(reduced)
    diagnostic_elapsed = time.perf_counter() - diagnostic_started

    probe = jnp.ones(operator.n, dtype=jnp.complex128)
    baseline = _measure_operator_action(operator.matvec, probe, config.overhead_runs)
    diagnostic_cost = float(diagnostic_elapsed)
    baseline_median = float(baseline["median_s"])

    record = {
        **common,
        "status": "completed",
        "verdict": assessment.verdict,
        "field_of_values": {
            "center_real": float(np.real(fov.center)),
            "center_imag": float(np.imag(fov.center)),
            "radius": float(fov.radius),
            "disk_rate": float(fov.disk_rate),
            "origin_enclosed": bool(fov.origin_enclosed),
        },
        "arnoldi": {
            "requested_steps": config.arnoldi_steps,
            "returned_steps": int(hessenberg.shape[1]),
            "hessenberg_shape": list(hessenberg.shape),
        },
        "reduced_pseudospectrum": {
            "grid_shape": list(reduced.shape),
            "epsilon_zero": float(epsilon),
            "sigma_minimum": float(jnp.min(reduced)),
            "sigma_maximum": float(jnp.max(reduced)),
        },
        "rates": rates._asdict(),
        "assessment": assessment._asdict(),
        "overhead": {
            "warmup_excluded": True,
            "block_until_ready": True,
            "baseline_runs": config.overhead_runs,
            "one_preconditioned_action": baseline,
            "diagnostic_wall_time_s": diagnostic_cost,
            "diagnostic_to_median_action": diagnostic_cost / max(baseline_median, 1.0e-30),
            # Host-side invocation counts only.  LOBPCG runs its iterations
            # inside a staged JAX loop, which traces the operator once no
            # matter how many iterations execute, so these numbers exclude the
            # bulk of the eigensolver work and must not be read as a device
            # operation count.  diagnostic_wall_time_s is the honest cost.
            "host_forward_calls": counter["matvec"],
            "host_adjoint_calls": counter["adjoint"],
            "host_call_counts_exclude_staged_loops": True,
            "arnoldi_steps": int(hessenberg.shape[1]),
        },
    }
    return (
        record,
        FigureData(
            state_index=state_index,
            preconditioner=preconditioner_name,
            fov=fov,
            ritz=ritz,
            real_grid=real_grid,
            imag_grid=imag_grid,
            sigma_min=reduced,
            predicted_gmres_factor=rates.predicted_gmres_factor,
        ),
    )


def _save_figure(figure: Any, path: Path) -> str:
    """Save and clear a figure produced by the generic plotting API."""
    figure.savefig(path, dpi=180)
    figure.clear()
    return str(path)


def _render_figures(data: list[FigureData], figure_dir: str | None) -> list[str]:
    """Render numerical-range, pseudospectrum, and envelope figures per diagnostic case."""
    if figure_dir is None:
        return []
    directory = Path(figure_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for item in data:
        stem = f"conditioning_state_{item.state_index}_{item.preconditioner}"
        paths.append(
            _save_figure(
                plot_numerical_range(item.fov),
                directory / f"{stem}_numerical_range.png",
            )
        )
        paths.append(
            _save_figure(
                plot_pseudospectrum((item.real_grid, item.imag_grid, item.sigma_min, item.ritz)),
                directory / f"{stem}_pseudospectrum.png",
            )
        )
        residuals = item.predicted_gmres_factor ** np.arange(0, 9)
        paths.append(
            _save_figure(
                plot_residual_envelope(residuals, item.fov.disk_rate),
                directory / f"{stem}_residual_envelope.png",
            )
        )
    return paths


def run_decision_demo(config: DemoConfig | None = None) -> dict[str, Any]:
    """Integrate and diagnose visited states; return the JSON-ready result."""
    if config is None:
        config = DemoConfig()
    if config.n_states < 1:
        raise ValueError("n_states must be positive")
    if config.n_angles < 3:
        raise ValueError("n_angles must be at least three")
    if config.pseudospectrum_points < 1:
        raise ValueError("pseudospectrum_points must be positive")

    grid = Grid2D.uniform(
        config.nx,
        config.ny,
        0.0,
        config.domain_length,
        0.0,
        config.domain_length,
        n_ghost=1,
    )
    model, fft_cache, _ = create_gray_scott_periodic_fft(
        grid=grid,
        Du=config.du,
        Dv=config.dv,
        F=config.feed,
        k=config.kill,
        dtype=jnp.float64,
    )
    preconditioner = create_gray_scott_fft_preconditioner(fft_cache)
    diagnostic_preconditioners = (
        ("fft_diffusion", preconditioner),
        ("identity", IdentityPreconditioner()),
    )
    nk_params = NKParams(
        max_newton_iters=config.max_newton_iters,
        max_krylov_iters=config.max_krylov_iters,
        newton_tol=config.newton_tol,
        krylov_tol=config.krylov_tol,
    )
    state = _initial_condition(grid)
    states: list[dict[str, Any]] = []
    figure_data: list[FigureData] = []
    time_value = 0.0
    failures: list[dict[str, Any]] = []
    for state_index in range(config.n_states):
        stepped, stats = be_step(
            model,
            state,
            time_value,
            config.dt,
            preconditioner=preconditioner,
            nk_params=nk_params,
        )
        # Diagnosing a state the implicit solve failed to produce would report
        # conditioning for a point the trajectory never reaches, and the run
        # would still be labelled completed.  Stop instead of advancing from an
        # invalid state.
        finite = all(
            bool(jnp.all(jnp.isfinite(value))) for value in jax.tree.leaves(stepped)
        )
        if not bool(stats.converged) or not finite:
            failures.append(
                {
                    "state_index": state_index,
                    "time": time_value + config.dt,
                    "converged": bool(stats.converged),
                    "finite": finite,
                    "newton_iters": int(stats.newton_iters),
                    "final_residual_norm": float(stats.final_res_norm),
                }
            )
            break
        state = stepped
        _ready_state(state)
        time_value += config.dt
        for preconditioner_name, diagnostic_preconditioner in diagnostic_preconditioners:
            record, plot_data = _run_state_diagnostics(
                state,
                model,
                diagnostic_preconditioner,
                preconditioner_name,
                config,
                state_index,
                time_value,
            )
            record["implicit_step"] = {
                "converged": bool(stats.converged),
                "newton_iters": int(stats.newton_iters),
                "linear_iters": int(stats.lin_iters),
                "final_residual_norm": float(stats.final_res_norm),
            }
            states.append(record)
            if state_index == 0 and plot_data is not None:
                figure_data.append(plot_data)

    diagnostics_complete = all(row["status"] == "completed" for row in states)
    if failures:
        status = "failed"
    elif not diagnostics_complete:
        status = "completed_with_skips"
    else:
        status = "completed"
    return {
        "schema_version": "conditioning_decision_demo_v1",
        "status": status,
        "implicit_step_failures": failures,
        "config": config._asdict(),
        "model": {
            "name": "gray_scott",
            "grid": [config.nx, config.ny],
            "boundary": "periodic",
            "integrator": "backward_euler_newton_krylov",
            "step_preconditioner": "fft_diffusion",
            "diagnostic_preconditioners": ["fft_diffusion", "identity"],
            "diffusion_dominated_parameters": {
                "Du": config.du,
                "Dv": config.dv,
                "feed": config.feed,
                "kill": config.kill,
            },
        },
        "states": states,
        "figures": _render_figures(figure_data, config.figure_dir),
    }


def main() -> None:
    """Run the default demo and write its diagnostics JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/conditioning_decision_demo.json"),
    )
    parser.add_argument("--nx", type=int, default=64)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--n-states", type=int, default=3)
    parser.add_argument("--overhead-runs", type=int, default=50)
    args = parser.parse_args()
    config = DemoConfig(
        nx=args.nx,
        ny=args.ny,
        n_states=args.n_states,
        overhead_runs=args.overhead_runs,
    )
    result = run_decision_demo(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
