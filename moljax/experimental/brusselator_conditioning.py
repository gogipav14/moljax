"""Visited-state conditioning study helpers for the periodic Brusselator.

This experimental module deliberately reuses moljax's shipped Brusselator
factory, periodic FFT diffusion preconditioner, and generic conditioning
toolbox.  It contributes only the study wiring: a two-field backward-Euler
linearization, diagnostic evaluation, and a counted GMRES measurement on the
same left-preconditioned Newton system.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from moljax.conditioning import (
    LinearizedOperator,
    adjoint_identity,
    arnoldi,
    assess_preconditioner,
    epsilon_zero,
    estimate_rates,
    linearized_operator,
    numerical_range,
    ritz_values,
)
from moljax.core.grid import Grid2D
from moljax.core.model import MOLModel, create_brusselator_periodic_fft
from moljax.core.newton_krylov import NKParams, create_implicit_residual
from moljax.core.preconditioners import (
    FFTDiffusionPreconditioner,
    IdentityPreconditioner,
    PrecondContext,
    Preconditioner,
    create_fft_preconditioner,
)
from moljax.core.state import StateDict
from moljax.core.stepping import be_step
from moljax.experimental.pme_conditioning import _counted_gmres


class BrusselatorRegime(NamedTuple):
    """Physical parameters and reference horizon for one Brusselator regime."""

    name: str
    du: float
    dv: float
    a: float
    b: float
    domain_length: float
    reference_final_time: float


HOPF_REGIME = BrusselatorRegime(
    name="hopf",
    du=0.01,
    dv=0.02,
    a=1.0,
    b=3.4,
    domain_length=5.0,
    reference_final_time=50.0,
)
"""Oscillatory Brusselator regime with ``b > 1 + a**2``."""

TURING_REGIME = BrusselatorRegime(
    name="turing",
    du=0.01,
    dv=0.1,
    a=1.0,
    b=1.8,
    domain_length=5.0,
    reference_final_time=200.0,
)
"""Diffusion-driven Brusselator regime with the paper's ``L=5, t=200`` target."""

REGIMES = {HOPF_REGIME.name: HOPF_REGIME, TURING_REGIME.name: TURING_REGIME}


class BrusselatorLinearization(NamedTuple):
    """A BE linearization and its consistently left-preconditioned RHS."""

    operator: LinearizedOperator
    residual: Any
    preconditioner: Preconditioner
    context: PrecondContext
    rhs: jax.Array


class TrajectorySample(NamedTuple):
    """One converged trajectory state and its distance from the steady state."""

    step: int
    time: float
    state: StateDict
    developedness: dict[str, float]


def resolve_regime(regime: str | BrusselatorRegime) -> BrusselatorRegime:
    """Return a named standard regime or pass through an explicit parameter set."""
    if isinstance(regime, BrusselatorRegime):
        return regime
    try:
        return REGIMES[regime]
    except KeyError as error:
        choices = ", ".join(sorted(REGIMES))
        raise ValueError(
            f"Unknown Brusselator regime {regime!r}; choose one of {choices}"
        ) from error


def build_brusselator_system(
    regime: str | BrusselatorRegime,
    grid: Grid2D,
) -> tuple[MOLModel, Any, dict[str, float]]:
    """Build the shipped periodic-FFT Brusselator model for one study regime."""
    selected = resolve_regime(regime)
    return create_brusselator_periodic_fft(
        grid,
        Du=selected.du,
        Dv=selected.dv,
        a=selected.a,
        b=selected.b,
        dtype=jnp.float64,
    )


def _ready_state(state: StateDict) -> StateDict:
    """Synchronize every array in a PyTree state before returning it."""
    return jax.tree_util.tree_map(jax.block_until_ready, state)


def _initial_state(
    model: MOLModel,
    regime: BrusselatorRegime,
    perturbation: float,
    seed: int,
) -> StateDict:
    """Return a small, reproducible perturbation of the homogeneous steady state."""
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive so the analysed trajectory is dynamical")
    grid = model.grid
    if not isinstance(grid, Grid2D):
        raise TypeError("The Brusselator study requires a two-dimensional grid")
    key_u, key_v = jax.random.split(jax.random.PRNGKey(seed))
    shape = (grid.ny_total, grid.nx_total)
    u = jnp.full(shape, regime.a, dtype=jnp.float64)
    v = jnp.full(shape, regime.b / regime.a, dtype=jnp.float64)
    u = u + perturbation * jax.random.normal(key_u, shape, dtype=jnp.float64)
    v = v + perturbation * jax.random.normal(key_v, shape, dtype=jnp.float64)
    return model.apply_bcs({"u": u, "v": v}, 0.0)


def _integrate_visited_states(
    regime: BrusselatorRegime,
    model: MOLModel,
    fft_cache: Any,
    *,
    n_steps: int,
    dt: float,
    perturbation: float,
    seed: int,
    nk_params: NKParams,
) -> list[StateDict]:
    """Advance a perturbed state by converged FFT-preconditioned BE steps."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    state = _initial_state(model, regime, perturbation, seed)
    preconditioner = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    visited: list[StateDict] = []
    time_value = 0.0
    for _ in range(n_steps):
        state, stats = be_step(
            model,
            state,
            time_value,
            dt,
            preconditioner=preconditioner,
            nk_params=nk_params,
        )
        state = _ready_state(state)
        if not bool(stats.converged):
            raise RuntimeError(
                "The FFT-preconditioned backward-Euler step did not converge "
                f"at t={time_value + dt:g}; no nonconverged iterate is analysed."
            )
        visited.append(state)
        time_value += dt
    return visited


def visited_states(
    regime: str | BrusselatorRegime,
    *,
    grid: Grid2D,
    n_steps: int,
    dt: float,
    perturbation: float,
    seed: int,
) -> list[StateDict]:
    """Return genuinely visited states from FFT-preconditioned BE integration.

    The starting state is the homogeneous Brusselator steady state
    ``(a, b / a)`` plus a small deterministic random perturbation.  Each
    returned state is the converged result of one shipped
    :func:`moljax.core.stepping.be_step`, not an analytic fixed point.
    """
    selected = resolve_regime(regime)
    model, fft_cache, _ = build_brusselator_system(selected, grid)
    return _integrate_visited_states(
        selected,
        model,
        fft_cache,
        n_steps=n_steps,
        dt=dt,
        perturbation=perturbation,
        seed=seed,
        nk_params=NKParams(
            max_newton_iters=10,
            max_krylov_iters=50,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        ),
    )


def state_developedness(
    state: StateDict,
    grid: Grid2D,
    regime: str | BrusselatorRegime,
) -> dict[str, float]:
    """Measure interior departure from the homogeneous Brusselator steady state."""
    selected = resolve_regime(regime)
    slice_y, slice_x = grid.interior_slice
    u_interior = jnp.asarray(state["u"], dtype=jnp.float64)[slice_y, slice_x]
    v_interior = jnp.asarray(state["v"], dtype=jnp.float64)[slice_y, slice_x]
    return {
        "max_abs_u_minus_steady": float(jnp.max(jnp.abs(u_interior - selected.a))),
        "max_abs_v_minus_steady": float(jnp.max(jnp.abs(v_interior - selected.b / selected.a))),
    }


def _sampled_visited_states(
    regime: BrusselatorRegime,
    model: MOLModel,
    fft_cache: Any,
    *,
    sample_steps: tuple[int, ...],
    dt: float,
    perturbation: float,
    seed: int,
    nk_params: NKParams,
) -> list[TrajectorySample]:
    """Advance BE steps and retain only requested, converged trajectory states."""
    if not sample_steps:
        raise ValueError("sample_steps must be nonempty")
    if any(step < 1 for step in sample_steps):
        raise ValueError("sample_steps must contain positive step numbers")
    if tuple(sorted(set(sample_steps))) != sample_steps:
        raise ValueError("sample_steps must be sorted and contain no duplicates")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not isinstance(model.grid, Grid2D):
        raise TypeError("The Brusselator study requires a two-dimensional grid")

    state = _initial_state(model, regime, perturbation, seed)
    preconditioner = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    requested = set(sample_steps)
    samples: list[TrajectorySample] = []
    time_value = 0.0
    for step in range(1, sample_steps[-1] + 1):
        state, stats = be_step(
            model,
            state,
            time_value,
            dt,
            preconditioner=preconditioner,
            nk_params=nk_params,
        )
        state = _ready_state(state)
        time_value += dt
        if not bool(stats.converged):
            raise RuntimeError(
                "The FFT-preconditioned backward-Euler step did not converge "
                f"at t={time_value:g}; no nonconverged iterate is analysed."
            )
        if step in requested:
            samples.append(
                TrajectorySample(
                    step=step,
                    time=time_value,
                    state=state,
                    developedness=state_developedness(state, model.grid, regime),
                )
            )
    return samples


def sampled_visited_states(
    regime: str | BrusselatorRegime,
    *,
    grid: Grid2D,
    sample_steps: tuple[int, ...],
    dt: float,
    perturbation: float,
    seed: int,
    nk_params: NKParams | None = None,
) -> list[TrajectorySample]:
    """Return selected converged states along a developed BE trajectory.

    This retains only the requested samples while integrating every preceding
    step.  The per-sample ``developedness`` metric is the maximum interior
    departure from ``(a, b/a)``, ensuring that a diagnostic can distinguish a
    developed state from a nearly unchanged perturbation.
    """
    selected = resolve_regime(regime)
    model, fft_cache, _ = build_brusselator_system(selected, grid)
    if nk_params is None:
        nk_params = NKParams(
            max_newton_iters=15,
            max_krylov_iters=100,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        )
    return _sampled_visited_states(
        selected,
        model,
        fft_cache,
        sample_steps=sample_steps,
        dt=dt,
        perturbation=perturbation,
        seed=seed,
        nk_params=nk_params,
    )


def _preconditioner(
    kind: str,
    fft_cache: Any,
) -> FFTDiffusionPreconditioner | IdentityPreconditioner:
    """Return the requested baseline or shipped diffusion preconditioner."""
    if kind == "fft_diffusion":
        return create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    if kind == "identity":
        return IdentityPreconditioner()
    raise ValueError("preconditioner_kind must be 'fft_diffusion' or 'identity'")


def build_brusselator_linearization(
    state: StateDict,
    model: MOLModel,
    fft_cache: Any,
    diffusivities: Mapping[str, float],
    dt: float,
    *,
    time_value: float = 0.0,
    preconditioner_kind: str = "fft_diffusion",
) -> BrusselatorLinearization:
    """Build ``P^-1 J`` and ``P^-1(-R)`` for the next BE Brusselator solve.

    The state remains a two-field ``StateDict`` through the public residual,
    JVP/VJP, and preconditioner actions.  The shared
    :func:`moljax.conditioning.linearized_operator` performs the only
    flattening, at the conditioning-toolbox boundary.
    """
    if set(diffusivities) != {"u", "v"}:
        raise ValueError("Brusselator FFT diffusivities must contain exactly 'u' and 'v'")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    residual = create_implicit_residual(model, state, time_value + dt, dt, method="be")
    preconditioner = _preconditioner(preconditioner_kind, fft_cache)
    context = PrecondContext(grid=model.grid, dt=dt, params=model.params)
    operator = linearized_operator(
        residual,
        state,
        preconditioner=preconditioner,
        context=context,
    )
    negative_residual = jax.tree_util.tree_map(lambda value: -value, residual(state))
    rhs_state = preconditioner.apply(negative_residual, context)
    rhs, _ = ravel_pytree(rhs_state)
    return BrusselatorLinearization(operator, residual, preconditioner, context, rhs)


def assess_brusselator_state(
    state: StateDict,
    model: MOLModel,
    fft_cache: Any,
    diffusivities: Mapping[str, float],
    dt: float,
    regime_params: str | BrusselatorRegime,
    *,
    preconditioner_kind: str = "fft_diffusion",
    time_value: float = 0.0,
    n_angles: int = 4,
    fov_max_iters: int = 8,
    arnoldi_steps: int = 6,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Apply the generic diagnostics to one visited two-field Brusselator state.

    The adjoint identity is a required gate for numerical-range diagnostics.
    If it fails, the result is explicitly flagged and no field-of-values
    verdict is inferred.
    """
    selected = resolve_regime(regime_params)
    linearization = build_brusselator_linearization(
        state,
        model,
        fft_cache,
        diffusivities,
        dt,
        time_value=time_value,
        preconditioner_kind=preconditioner_kind,
    )
    operator = linearization.operator
    adjoint_error = adjoint_identity(operator, jax.random.PRNGKey(seed), operator.n)
    common = {
        "regime": selected.name,
        "preconditioner": preconditioner_kind,
        "operator_dimension": operator.n,
        "adjoint_error": float(adjoint_error),
        "adjoint_tolerance": 1.0e-8,
    }
    if adjoint_error > 1.0e-8:
        return {
            **common,
            "status": "adjoint_failed",
            "verdict": "skipped",
            "disk_rate": None,
            "epsilon_zero": None,
            "predicted_gmres_factor": None,
            "origin_enclosed": None,
            "n_right_real_outliers": None,
            "fov_imaginary_extent": None,
        }

    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed + 1))
    start = jax.random.normal(key_real, (operator.n,), dtype=jnp.float64)
    start = start + 1j * jax.random.normal(key_imag, (operator.n,), dtype=jnp.float64)
    _, hessenberg = arnoldi(operator.matvec, start, min(arnoldi_steps, operator.n))
    ritz = ritz_values(hessenberg)
    epsilon_at_zero = epsilon_zero(hessenberg)
    field_of_values = numerical_range(
        operator.matvec,
        operator.matvec_adjoint,
        operator.n,
        n_angles=n_angles,
        max_iters=fov_max_iters,
    )
    rates = estimate_rates(field_of_values, ritz)
    assessment = assess_preconditioner(field_of_values, ritz, epsilon_at_zero)
    jax.block_until_ready(field_of_values.boundary)

    return {
        **common,
        "status": "completed",
        "verdict": assessment.verdict,
        "disk_rate": float(assessment.disk_rate),
        "epsilon_zero": float(assessment.epsilon_zero),
        "predicted_gmres_factor": float(assessment.predicted_gmres_factor),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "n_right_real_outliers": int(assessment.n_right_real_outliers),
        "fov_imaginary_extent": float(jnp.max(jnp.abs(jnp.imag(field_of_values.boundary)))),
        "rates": rates._asdict(),
    }


def measure_brusselator_gmres(
    state: StateDict,
    model: MOLModel,
    fft_cache: Any,
    diffusivities: Mapping[str, float],
    dt: float,
    regime_params: str | BrusselatorRegime,
    *,
    tol: float,
    max_iters: int,
    time_value: float = 0.0,
    preconditioner_kind: str = "fft_diffusion",
) -> dict[str, Any]:
    """Measure true GMRES work on the same ``P^-1 J`` system as the diagnostics.

    This directly reuses the experimental residual-history GMRES loop from
    the nonlinear-diffusion study.  Only the state PyTree and its flattening
    differ; those are handled by the shared linearization adapter.
    """
    selected = resolve_regime(regime_params)
    linearization = build_brusselator_linearization(
        state,
        model,
        fft_cache,
        diffusivities,
        dt,
        time_value=time_value,
        preconditioner_kind=preconditioner_kind,
    )
    measurement = _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=tol,
        max_iters=max_iters,
    )
    return {
        **measurement,
        "regime": selected.name,
        "preconditioner": preconditioner_kind,
        "operator_dimension": linearization.operator.n,
    }


__all__ = [
    "BrusselatorLinearization",
    "BrusselatorRegime",
    "HOPF_REGIME",
    "REGIMES",
    "TURING_REGIME",
    "TrajectorySample",
    "assess_brusselator_state",
    "build_brusselator_linearization",
    "build_brusselator_system",
    "measure_brusselator_gmres",
    "resolve_regime",
    "sampled_visited_states",
    "state_developedness",
    "visited_states",
]
