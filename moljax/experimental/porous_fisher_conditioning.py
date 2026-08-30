"""Conditioning adapters for experimental Porous--Fisher backward-Euler steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

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
from moljax.core.preconditioners import PrecondContext
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import (
    ExperimentalPreconditioner,
    _counted_gmres,
    pme_preconditioner_variant,
)
from moljax.experimental.porous_fisher import porous_fisher_rhs

Residual = Callable[[jax.Array], jax.Array]


class PorousFisherLinearization(NamedTuple):
    """A Porous--Fisher BE linearization and its preconditioned Newton RHS."""

    operator: LinearizedOperator
    residual: Residual
    preconditioner: ExperimentalPreconditioner
    context: PrecondContext
    d0: float
    rhs: jax.Array


def porous_fisher_backward_euler_residual(
    u: jax.Array,
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    dt: float,
    epsilon: float,
) -> jax.Array:
    """Return ``u-u_prev-dt*F(u)`` for one Porous--Fisher BE step."""
    candidate = jnp.asarray(u, dtype=jnp.float64)
    previous = jnp.asarray(u_prev, dtype=jnp.float64)
    if candidate.shape != (grid.nx,) or previous.shape != (grid.nx,):
        raise ValueError(
            f"Expected interior shape {(grid.nx,)}, got {candidate.shape} and {previous.shape}"
        )
    return candidate - previous - dt * porous_fisher_rhs(candidate, grid, r=r, epsilon=epsilon)


def make_porous_fisher_residual(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    dt: float,
    epsilon: float,
) -> Residual:
    """Build the public residual callable consumed by the shared JFNK adapter."""
    previous = jnp.asarray(u_prev, dtype=jnp.float64)

    def residual(u: jax.Array) -> jax.Array:
        return porous_fisher_backward_euler_residual(
            u,
            previous,
            grid,
            r=r,
            dt=dt,
            epsilon=epsilon,
        )

    return residual


def build_porous_fisher_linearization(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    const_value: float = 1.0,
) -> PorousFisherLinearization:
    """Build ``P^-1 J`` for a Porous--Fisher BE residual.

    The reused ``m=2`` D0 variants approximate only the diffusion derivative
    ``2*u``.  The logistic reaction derivative remains in ``J`` and is not
    represented in the Helmholtz preconditioner.
    """
    previous = jnp.asarray(u_prev, dtype=jnp.float64)
    residual = make_porous_fisher_residual(
        previous,
        grid,
        r=r,
        dt=dt,
        epsilon=epsilon,
    )
    preconditioner, d0 = pme_preconditioner_variant(
        previous,
        grid,
        2.0,
        dt,
        epsilon,
        d0_kind,
        const_value=const_value,
    )
    context = PrecondContext(grid=grid, dt=dt, params={})
    operator = linearized_operator(
        residual,
        previous,
        preconditioner=preconditioner,
        context=context,
    )
    rhs = preconditioner.apply(-residual(previous), context)
    return PorousFisherLinearization(operator, residual, preconditioner, context, d0, rhs)


def measure_porous_fisher_gmres_iterations(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    tol: float,
    max_iters: int,
    const_value: float = 1.0,
) -> dict[str, Any]:
    """Return the true residual-history GMRES count for one fixed PF system."""
    linearization = build_porous_fisher_linearization(
        u_prev,
        grid,
        r=r,
        dt=dt,
        epsilon=epsilon,
        d0_kind=d0_kind,
        const_value=const_value,
    )
    measurement = _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=tol,
        max_iters=max_iters,
    )
    return {**measurement, "d0": linearization.d0}


def assess_porous_fisher_state(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    const_value: float = 1.0,
    n_angles: int = 6,
    fov_max_iters: int = 30,
    arnoldi_steps: int = 8,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Assess a Porous--Fisher state with the shared conditioning toolbox."""
    linearization = build_porous_fisher_linearization(
        u_prev,
        grid,
        r=r,
        dt=dt,
        epsilon=epsilon,
        d0_kind=d0_kind,
        const_value=const_value,
    )
    operator = linearization.operator
    adjoint_error = adjoint_identity(operator, jax.random.PRNGKey(seed), operator.n)
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

    return {
        "d0_kind": d0_kind,
        "d0": float(linearization.d0),
        "operator_dimension": operator.n,
        "adjoint_error": float(adjoint_error),
        "adjoint_tolerance": 1.0e-8,
        "verdict": assessment.verdict,
        "disk_rate": float(assessment.disk_rate),
        "epsilon_zero": float(assessment.epsilon_zero),
        "predicted_gmres_factor": float(assessment.predicted_gmres_factor),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "n_right_real_outliers": int(assessment.n_right_real_outliers),
        "rates": {name: value for name, value in rates._asdict().items()},
    }


__all__ = [
    "PorousFisherLinearization",
    "assess_porous_fisher_state",
    "build_porous_fisher_linearization",
    "make_porous_fisher_residual",
    "measure_porous_fisher_gmres_iterations",
    "porous_fisher_backward_euler_residual",
]
