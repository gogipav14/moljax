#!/usr/bin/env python3
"""Work--precision comparisons for nonlinear diffusion models.

This benchmark compares the local fixed-step backward-Euler JFNK path against
Diffrax's adaptive Tsit5/PID integrator on identical node-centred spatial
discretizations.  Diffrax is intentionally imported only in this benchmark:
it is an external comparator rather than a moljax runtime dependency.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, NamedTuple

import diffrax
import jax
import jax.numpy as jnp

from moljax.core.fft_nonperiodic import laplacian_symbol_dirichlet
from moljax.core.newton_krylov import NKParams, newton_krylov_solve
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import (
    barenblatt,
    porous_medium_node_centered_rhs,
    support_halfwidth,
)
from moljax.experimental.pme_conditioning import (
    make_backward_euler_residual,
    pme_preconditioner_variant,
)
from moljax.experimental.pme_preconditioner import (
    PMEHelmholtzPreconditioner,
    _d0_frozen_bulk_value,
)
from moljax.experimental.porous_fisher import (
    porous_fisher_rhs,
    porous_fisher_traveling_wave,
    wave_front_position,
    wave_speed,
)
from moljax.experimental.porous_fisher_conditioning import make_porous_fisher_residual


class WorkPrecisionConfig(NamedTuple):
    """Configuration for the nonlinear work--precision comparison.

    The default has 50 timed repeats per configuration.  Tests intentionally
    override that count with one repeat; the published benchmark protocol does
    not.  The exact references are used only to assess time-integrator error;
    both methods use the same node-centred spatial discretization.
    """

    nx: int = 64
    pme_x_min: float = -4.0
    pme_x_max: float = 4.0
    pme_t0: float = 1.0
    pme_t1: float = 1.2
    pme_b: float = 0.30
    porous_fisher_x_min: float = -8.0
    porous_fisher_x_max: float = 8.0
    porous_fisher_t0: float = 0.25
    porous_fisher_t1: float = 0.29
    porous_fisher_r: float = 1.0
    epsilon: float = 1.0e-5
    pme_be_dt_values: tuple[float, ...] = (0.1, 0.05, 0.025)
    porous_fisher_be_dt_values: tuple[float, ...] = (0.02, 0.01, 0.005)
    diffrax_rtol_values: tuple[float, ...] = (1.0e-1, 1.0e-4, 1.0e-6)
    diffrax_atol_ratio: float = 1.0e-2
    timing_runs: int = 50
    max_newton_iters: int = 8
    max_krylov_iters: int = 64
    newton_tol: float = 1.0e-8
    krylov_tol: float = 1.0e-8
    max_diffrax_steps: int = 10_000
    output_path: str = "benchmarks/results/work_precision_nonlinear_diffusion.json"


class ProblemSetup(NamedTuple):
    """One exact-reference problem on a shared node-centred grid."""

    name: str
    grid: NodeCenteredDirichletGrid
    t0: float
    t1: float
    initial_state: jax.Array
    exact_final_state: jax.Array
    error_mask: jax.Array
    rhs: Callable[[jax.Array], jax.Array]
    be_dt_values: tuple[float, ...]


def _median_iqr(values: list[float]) -> dict[str, float]:
    """Return the median and interquartile range without another dependency."""
    if not values:
        raise ValueError("timing values must not be empty")
    sorted_values = sorted(values)
    lower = sorted_values[len(sorted_values) // 4]
    upper = sorted_values[(3 * len(sorted_values)) // 4]
    return {
        "median_seconds": float(median(sorted_values)),
        "iqr_seconds": float(upper - lower),
        "minimum_seconds": float(sorted_values[0]),
        "maximum_seconds": float(sorted_values[-1]),
    }


def _validate_config(config: WorkPrecisionConfig) -> None:
    """Reject schedules that cannot represent fixed full-step integrations."""
    if config.nx < 8:
        raise ValueError("nx must be at least 8")
    if config.timing_runs < 1:
        raise ValueError("timing_runs must be positive")
    if not config.pme_be_dt_values or min(config.pme_be_dt_values) <= 0.0:
        raise ValueError("pme_be_dt_values must contain positive values")
    if not config.porous_fisher_be_dt_values or min(config.porous_fisher_be_dt_values) <= 0.0:
        raise ValueError("porous_fisher_be_dt_values must contain positive values")
    if not config.diffrax_rtol_values or min(config.diffrax_rtol_values) <= 0.0:
        raise ValueError("diffrax_rtol_values must contain positive values")
    if not 0.0 < config.diffrax_atol_ratio <= 1.0:
        raise ValueError("diffrax_atol_ratio must lie in (0, 1]")
    for start, end, label, dts in (
        (config.pme_t0, config.pme_t1, "PME", config.pme_be_dt_values),
        (
            config.porous_fisher_t0,
            config.porous_fisher_t1,
            "Porous--Fisher",
            config.porous_fisher_be_dt_values,
        ),
    ):
        if end <= start:
            raise ValueError(f"{label} final time must exceed its initial time")
        for dt in dts:
            steps = (end - start) / dt
            if abs(steps - round(steps)) > 1.0e-12:
                raise ValueError(f"{label} window must contain an integer number of dt={dt} steps")


def _pme_setup(config: WorkPrecisionConfig) -> ProblemSetup:
    """Build the Barenblatt-reference ``m=2`` porous-medium problem."""
    grid = NodeCenteredDirichletGrid.uniform(config.nx, config.pme_x_min, config.pme_x_max)
    x = grid.x_coords()
    initial = barenblatt(x, config.pme_t0, 2.0, b=config.pme_b)
    exact = barenblatt(x, config.pme_t1, 2.0, b=config.pme_b)
    radius = support_halfwidth(config.pme_t1, 2.0, config.pme_b)
    mask = jnp.abs(x) < radius - 4.0 * grid.dx
    if not bool(jnp.any(mask)):
        raise ValueError("PME edge exclusion left no nodes for the max-norm error")

    def rhs(u: jax.Array) -> jax.Array:
        return porous_medium_node_centered_rhs(u, grid, 2.0, epsilon=config.epsilon)

    return ProblemSetup(
        "pme_m2",
        grid,
        config.pme_t0,
        config.pme_t1,
        initial,
        exact,
        mask,
        rhs,
        config.pme_be_dt_values,
    )


def _porous_fisher_setup(config: WorkPrecisionConfig) -> ProblemSetup:
    """Build the sharp-wave ``r=1`` Porous--Fisher reference problem."""
    grid = NodeCenteredDirichletGrid.uniform(
        config.nx,
        config.porous_fisher_x_min,
        config.porous_fisher_x_max,
    )
    x = grid.x_coords()
    speed = wave_speed(config.porous_fisher_r)
    initial = porous_fisher_traveling_wave(
        x,
        config.porous_fisher_t0,
        r=config.porous_fisher_r,
        c=speed,
    )
    exact = porous_fisher_traveling_wave(
        x,
        config.porous_fisher_t1,
        r=config.porous_fisher_r,
        c=speed,
    )
    edge = wave_front_position(config.porous_fisher_t1, c=speed)
    mask = (x > grid.x_min + 1.5) & (x < edge - 4.0 * grid.dx)
    if not bool(jnp.any(mask)):
        raise ValueError("Porous--Fisher edge exclusion left no nodes for the max-norm error")

    def rhs(u: jax.Array) -> jax.Array:
        return porous_fisher_rhs(u, grid, r=config.porous_fisher_r, epsilon=config.epsilon)

    return ProblemSetup(
        "porous_fisher_r1",
        grid,
        config.porous_fisher_t0,
        config.porous_fisher_t1,
        initial,
        exact,
        mask,
        rhs,
        config.porous_fisher_be_dt_values,
    )


def _infinity_error(solution: jax.Array, setup: ProblemSetup) -> float:
    """Measure the identical masked final-time max-norm error for either method."""
    difference = jnp.abs(solution[setup.error_mask] - setup.exact_final_state[setup.error_mask])
    return float(jnp.max(difference))


def _be_step(
    state: jax.Array,
    setup: ProblemSetup,
    dt: float,
    config: WorkPrecisionConfig,
) -> tuple[jax.Array, dict[str, float | int | bool]]:
    """Take one frozen-bulk-preconditioned backward-Euler nonlinear solve."""
    if setup.name == "pme_m2":
        residual = make_backward_euler_residual(state, setup.grid, 2.0, dt, config.epsilon)
    else:
        residual = make_porous_fisher_residual(
            state,
            setup.grid,
            r=config.porous_fisher_r,
            dt=dt,
            epsilon=config.epsilon,
        )
    preconditioner, d0 = pme_preconditioner_variant(
        state,
        setup.grid,
        2.0,
        dt,
        config.epsilon,
        "frozen_bulk",
    )
    parameters = NKParams(
        max_newton_iters=config.max_newton_iters,
        max_krylov_iters=config.max_krylov_iters,
        newton_tol=config.newton_tol,
        krylov_tol=config.krylov_tol,
    )
    result = newton_krylov_solve(
        residual,
        state,
        setup.grid,
        params={},
        preconditioner=preconditioner,
        nk_params=parameters,
        dt=dt,
    )
    solution = jax.block_until_ready(result.solution)
    newton_iterations = int(result.stats.newton_iters)
    return solution, {
        "converged": bool(result.stats.converged),
        "newton_iterations": newton_iterations,
        "configured_krylov_budget_total": int(result.stats.lin_iters),
        "final_residual_l2": float(result.stats.final_res_norm),
        "d0": float(d0),
        "nonlinear_rhs_evaluations": 1 + newton_iterations * (1 + parameters.max_backtrack),
    }


def integrate_backward_euler(
    setup: ProblemSetup,
    dt: float,
    config: WorkPrecisionConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    """Integrate one problem with fixed-step BE-JFNK and frozen-bulk D0.

    ``nonlinear_rhs_evaluations`` counts calls implied by the core solver's
    initial residual, one residual per Newton step, and its statically bounded
    line-search residual calls.  ``NKStats.lin_iters`` is retained separately
    because current core GMRES reports its configured budget rather than a
    converged Krylov count.
    """
    n_steps = int(round((setup.t1 - setup.t0) / dt))
    state = setup.initial_state
    step_records: list[dict[str, float | int | bool]] = []
    for _ in range(n_steps):
        state, step = _be_step(state, setup, dt, config)
        if not step["converged"]:
            raise RuntimeError(
                f"BE-JFNK did not converge for {setup.name}, dt={dt}, "
                f"residual={step['final_residual_l2']}"
            )
        step_records.append(step)
    return state, {
        "steps": n_steps,
        "newton_iterations": sum(int(step["newton_iterations"]) for step in step_records),
        "configured_krylov_budget_total": sum(
            int(step["configured_krylov_budget_total"]) for step in step_records
        ),
        "nonlinear_rhs_evaluations": sum(
            int(step["nonlinear_rhs_evaluations"]) for step in step_records
        ),
        "d0": {
            "min": float(min(float(step["d0"]) for step in step_records)),
            "median": float(median(float(step["d0"]) for step in step_records)),
            "max": float(max(float(step["d0"]) for step in step_records)),
        },
    }


def _build_jitted_be_integrator(
    setup: ProblemSetup,
    dt: float,
    config: WorkPrecisionConfig,
) -> Callable[[jax.Array], tuple[jax.Array, tuple[jax.Array, ...]]]:
    """Create one cached fixed-step BE runner for a timing configuration.

    The runner accepts the initial state as an argument so JAX compiles the
    nested residual/preconditioner construction once per problem and time
    step.  The benchmark's one excluded warmup then pays that compilation cost
    before all repeated samples use the same executable.
    """
    n_steps = int(round((setup.t1 - setup.t0) / dt))
    parameters = NKParams(
        max_newton_iters=config.max_newton_iters,
        max_krylov_iters=config.max_krylov_iters,
        newton_tol=config.newton_tol,
        krylov_tol=config.krylov_tol,
    )
    laplacian_symbol = laplacian_symbol_dirichlet(
        setup.grid.nx,
        setup.grid.dx,
        dtype=jnp.float64,
    )

    def advance(
        state: jax.Array,
        _: None,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
        if setup.name == "pme_m2":
            residual = make_backward_euler_residual(state, setup.grid, 2.0, dt, config.epsilon)
        else:
            residual = make_porous_fisher_residual(
                state,
                setup.grid,
                r=config.porous_fisher_r,
                dt=dt,
                epsilon=config.epsilon,
            )
        d0 = _d0_frozen_bulk_value(state, 2.0)
        preconditioner = PMEHelmholtzPreconditioner(
            d0=d0,
            dt=dt,
            laplacian_symbol=laplacian_symbol,
        )
        result = newton_krylov_solve(
            residual,
            state,
            setup.grid,
            params={},
            preconditioner=preconditioner,
            nk_params=parameters,
            dt=dt,
        )
        return result.solution, (
            result.stats.converged,
            result.stats.newton_iters,
            result.stats.lin_iters,
            result.stats.final_res_norm,
            jnp.asarray(d0, dtype=jnp.float64),
        )

    @jax.jit
    def integrate(initial_state: jax.Array) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        return jax.lax.scan(advance, initial_state, xs=None, length=n_steps)

    return integrate


def _summarize_jitted_be_work(
    history: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    config: WorkPrecisionConfig,
) -> dict[str, Any]:
    """Convert one cached BE runner's per-step JAX statistics to JSON data."""
    converged, newton_iterations, linear_budgets, final_residuals, d0_values = history
    if not bool(jnp.all(converged)):
        raise RuntimeError("BE-JFNK did not converge during the timed integration")
    newton = [int(value) for value in newton_iterations]
    d0 = [float(value) for value in d0_values]
    return {
        "steps": len(newton),
        "newton_iterations": sum(newton),
        "configured_krylov_budget_total": sum(int(value) for value in linear_budgets),
        "nonlinear_rhs_evaluations": sum(
            1 + value * (1 + NKParams().max_backtrack) for value in newton
        ),
        "d0": {
            "min": min(d0),
            "median": float(median(d0)),
            "max": max(d0),
        },
        "final_residual_l2": float(final_residuals[-1]),
    }


def integrate_diffrax(
    setup: ProblemSetup,
    rtol: float,
    config: WorkPrecisionConfig,
) -> tuple[jax.Array, dict[str, Any]]:
    """Integrate one shared RHS with Tsit5 and a PID error controller."""
    atol = rtol * config.diffrax_atol_ratio
    initial_dt = (setup.t1 - setup.t0) / 4.0
    term = diffrax.ODETerm(lambda _time, state, _args: setup.rhs(state))
    solution = diffrax.diffeqsolve(
        term,
        diffrax.Tsit5(),
        setup.t0,
        setup.t1,
        initial_dt,
        setup.initial_state,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=config.max_diffrax_steps,
        throw=False,
    )
    final_state = jax.block_until_ready(solution.ys[-1])
    steps = int(solution.stats["num_steps"])
    return final_state, {
        "rtol": rtol,
        "atol": atol,
        "accepted_steps": int(solution.stats["num_accepted_steps"]),
        "rejected_steps": int(solution.stats["num_rejected_steps"]),
        "attempted_steps": steps,
        "nfe_estimate": 6 * steps + 1,
        "nfe_note": (
            "Diffrax reports attempted/accepted/rejected steps but not an exact RHS counter; "
            "this is the Tsit5 FSAL stage estimate of six new RHS calls per attempted step "
            "plus the initial evaluation."
        ),
        "result": str(solution.result),
    }


def _build_jitted_diffrax_integrator(
    setup: ProblemSetup,
    rtol: float,
    config: WorkPrecisionConfig,
) -> Callable[[jax.Array], tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]]:
    """Create one cached Tsit5/PID runner for a timing configuration."""
    atol = rtol * config.diffrax_atol_ratio
    initial_dt = (setup.t1 - setup.t0) / 4.0
    term = diffrax.ODETerm(lambda _time, state, _args: setup.rhs(state))
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(t1=True)

    @jax.jit
    def integrate(
        initial_state: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array]]:
        solution = diffrax.diffeqsolve(
            term,
            diffrax.Tsit5(),
            setup.t0,
            setup.t1,
            initial_dt,
            initial_state,
            saveat=saveat,
            stepsize_controller=controller,
            max_steps=config.max_diffrax_steps,
            throw=False,
        )
        return solution.ys[-1], (
            solution.stats["num_accepted_steps"],
            solution.stats["num_rejected_steps"],
            solution.stats["num_steps"],
        )

    return integrate


def _summarize_jitted_diffrax_work(
    history: tuple[jax.Array, jax.Array, jax.Array],
    rtol: float,
    config: WorkPrecisionConfig,
) -> dict[str, Any]:
    """Convert cached Tsit5/PID statistics to the benchmark's work schema."""
    accepted_steps, rejected_steps, attempted_steps = history
    attempted = int(attempted_steps)
    return {
        "rtol": rtol,
        "atol": rtol * config.diffrax_atol_ratio,
        "accepted_steps": int(accepted_steps),
        "rejected_steps": int(rejected_steps),
        "attempted_steps": attempted,
        "nfe_estimate": 6 * attempted + 1,
        "nfe_note": (
            "Diffrax reports attempted/accepted/rejected steps but not an exact RHS counter; "
            "this is the Tsit5 FSAL stage estimate of six new RHS calls per attempted step "
            "plus the initial evaluation."
        ),
        "result": "completed",
    }


def _timed_runs(
    integrate: Callable[[], tuple[jax.Array, dict[str, Any]]],
    timing_runs: int,
) -> tuple[jax.Array, dict[str, Any], dict[str, float]]:
    """Warm up once, then measure fully synchronized repeated integrations."""
    warmup_state, _ = integrate()
    jax.block_until_ready(warmup_state)
    timings: list[float] = []
    final_state = warmup_state
    work: dict[str, Any] = {}
    for _ in range(timing_runs):
        started_at = perf_counter()
        final_state, work = integrate()
        jax.block_until_ready(final_state)
        timings.append(perf_counter() - started_at)
    return final_state, work, _median_iqr(timings)


def _be_records(setup: ProblemSetup, config: WorkPrecisionConfig) -> list[dict[str, Any]]:
    """Measure the fixed-step BE work--precision points for one problem."""
    records = []
    for dt in setup.be_dt_values:
        runner = _build_jitted_be_integrator(setup, dt, config)
        state, work, timing = _timed_runs(
            lambda runner=runner: runner(setup.initial_state),
            config.timing_runs,
        )
        records.append(
            {
                "method": "be_jfnk_frozen_bulk",
                "dt": dt,
                "error_inf": _infinity_error(state, setup),
                "runtime": timing,
                "work": _summarize_jitted_be_work(work, config),
            }
        )
    return records


def _diffrax_records(setup: ProblemSetup, config: WorkPrecisionConfig) -> list[dict[str, Any]]:
    """Measure the adaptive Tsit5/PID work--precision points for one problem."""
    records = []
    for rtol in config.diffrax_rtol_values:
        runner = _build_jitted_diffrax_integrator(setup, rtol, config)
        state, work, timing = _timed_runs(
            lambda runner=runner: runner(setup.initial_state),
            config.timing_runs,
        )
        records.append(
            {
                "method": "diffrax_tsit5_pid",
                "rtol": rtol,
                "atol": rtol * config.diffrax_atol_ratio,
                "error_inf": _infinity_error(state, setup),
                "runtime": timing,
                "work": _summarize_jitted_diffrax_work(work, rtol, config),
            }
        )
    return records


def _fastest_at_or_below(
    records: list[dict[str, Any]], target_error: float
) -> dict[str, Any] | None:
    """Return the fastest measured point whose max-norm error meets a target."""
    candidates = [record for record in records if float(record["error_inf"]) <= target_error]
    if not candidates:
        return None
    return min(candidates, key=lambda item: float(item["runtime"]["median_seconds"]))


def _matched_accuracy_crossovers(
    be_records: list[dict[str, Any]],
    diffrax_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare fastest measured configurations at standard target accuracies."""
    crossovers = []
    for target in (1.0e-3, 1.0e-5, 1.0e-6, 1.0e-8):
        be = _fastest_at_or_below(be_records, target)
        adaptive = _fastest_at_or_below(diffrax_records, target)
        if be is None or adaptive is None:
            crossovers.append(
                {
                    "target_error": target,
                    "status": "unreached_by_one_or_both_methods",
                    "be_jfnk": be,
                    "diffrax_tsit5_pid": adaptive,
                }
            )
            continue
        be_seconds = float(be["runtime"]["median_seconds"])
        adaptive_seconds = float(adaptive["runtime"]["median_seconds"])
        winner = "be_jfnk_frozen_bulk" if be_seconds <= adaptive_seconds else "diffrax_tsit5_pid"
        speedup = max(be_seconds, adaptive_seconds) / min(be_seconds, adaptive_seconds)
        crossovers.append(
            {
                "target_error": target,
                "status": "matched",
                "winner": winner,
                "speedup": speedup,
                "be_jfnk": be,
                "diffrax_tsit5_pid": adaptive,
            }
        )
    return crossovers


def _problem_summary(crossovers: list[dict[str, Any]]) -> str:
    """State the measured matched-accuracy result without assuming a winner."""
    matched = [item for item in crossovers if item["status"] == "matched"]
    if not matched:
        return "No common requested target was reached by both measured sweeps."
    statements = []
    for item in matched:
        statements.append(
            f"at error <= {item['target_error']:.0e}, {item['winner']} was "
            f"{item['speedup']:.2f}x faster"
        )
    return "; ".join(statements) + "."


def run_work_precision(config: WorkPrecisionConfig | None = None) -> dict[str, Any]:
    """Run both nonlinear exact-reference work--precision comparisons.

    The scope is deliberately narrow: it compares local preconditioned
    first-order BE-JFNK with an adaptive higher-order external integrator on
    the same nonlinear spatial discretization.  It is not a reproduction of a
    linear FFT-Crank--Nicolson comparison.
    """
    if config is None:
        config = WorkPrecisionConfig()
    _validate_config(config)
    started_at = perf_counter()
    with jax.enable_x64(True):
        setups = (_pme_setup(config), _porous_fisher_setup(config))
        problems: dict[str, dict[str, Any]] = {}
        for setup in setups:
            be_records = _be_records(setup, config)
            adaptive_records = _diffrax_records(setup, config)
            crossovers = _matched_accuracy_crossovers(be_records, adaptive_records)
            problems[setup.name] = {
                "time_window": {"t0": setup.t0, "t1": setup.t1},
                "grid": {
                    "nx": setup.grid.nx,
                    "x_min": setup.grid.x_min,
                    "x_max": setup.grid.x_max,
                    "dx": setup.grid.dx,
                },
                "error_metric": {
                    "name": "masked_linf",
                    "description": (
                        "max |u_numeric(t1)-u_exact(t1)| on the common smooth interior; "
                        "the compact-support edge and the finite-domain left boundary layer "
                        "are excluded identically for both methods."
                    ),
                    "porous_fisher_left_boundary_exclusion": 1.5,
                    "masked_nodes": int(jnp.sum(setup.error_mask)),
                },
                "be_jfnk_frozen_bulk": be_records,
                "diffrax_tsit5_pid": adaptive_records,
                "matched_accuracy_crossovers": crossovers,
                "summary": _problem_summary(crossovers),
            }

    report = {
        "description": "Nonlinear work--precision data for frozen-bulk BE-JFNK versus Diffrax.",
        "scope": (
            "This compares the local preconditioned BE-JFNK path with Diffrax adaptive Tsit5/PID "
            "on the same nonlinear node-centred spatial discretization. It differs from a linear "
            "FFT-Crank--Nicolson comparison: only the time integrator and nonlinear solver vary."
        ),
        "timing_protocol": {
            "warmup_excluded": True,
            "block_until_ready": True,
            "timed_runs_per_configuration": config.timing_runs,
            "statistic": "median and IQR wall-clock seconds",
        },
        "config": config._asdict(),
        "problems": problems,
        "summary": {name: problem["summary"] for name, problem in problems.items()},
        "runtime_seconds": perf_counter() - started_at,
    }
    output = Path(config.output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _print_summary(report: dict[str, Any]) -> None:
    """Print a compact table suitable for an interactive benchmark invocation."""
    print(f"runtime_seconds={report['runtime_seconds']:.3f}")
    for name, problem in report["problems"].items():
        print(name)
        for method in ("be_jfnk_frozen_bulk", "diffrax_tsit5_pid"):
            for record in problem[method]:
                runtime = record["runtime"]
                print(
                    f"  {method}: error={record['error_inf']:.3e} "
                    f"median={runtime['median_seconds']:.6f}s "
                    f"IQR={runtime['iqr_seconds']:.6f}s"
                )
        print(f"  {problem['summary']}")


def main() -> None:
    """Run the default benchmark or write to an explicitly supplied JSON path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=WorkPrecisionConfig().output_path)
    args = parser.parse_args()
    _print_summary(run_work_precision(WorkPrecisionConfig(output_path=args.output)))


if __name__ == "__main__":
    main()
