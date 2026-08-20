"""Experimental conditioning diagnostics for backward-Euler porous-medium steps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax
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
from moljax.core.grid import Grid1D
from moljax.core.preconditioners import IdentityPreconditioner, PrecondContext
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import (
    porous_medium_flux_rhs,
    porous_medium_node_centered_rhs,
)
from moljax.experimental.pme_preconditioner import (
    PMEHelmholtzPreconditioner,
    d0_const,
    d0_floor,
    d0_frozen_bulk,
    d0_frozen_mean,
    pme_helmholtz_preconditioner,
)

Residual = Callable[[jax.Array], jax.Array]
ExperimentalPreconditioner = PMEHelmholtzPreconditioner | IdentityPreconditioner
PMEGrid = Grid1D | NodeCenteredDirichletGrid


class PMELinearization(NamedTuple):
    """A backward-Euler PME linearization and its consistently preconditioned RHS."""

    operator: LinearizedOperator
    residual: Residual
    preconditioner: ExperimentalPreconditioner
    context: PrecondContext
    d0: float
    rhs: jax.Array


def interior_values(u: jax.Array, grid: PMEGrid) -> jax.Array:
    """Return interior degrees of freedom from an interior or padded field."""
    values = jnp.asarray(u, dtype=jnp.float64)
    if values.ndim != 1:
        raise ValueError("PME fields must be one-dimensional")
    if values.shape[0] == grid.nx:
        return values
    if isinstance(grid, NodeCenteredDirichletGrid):
        raise ValueError(f"PME field length must be {grid.nx}, got {values.shape[0]}")
    if values.shape[0] == grid.nx_total:
        return values[grid.interior_slice]
    raise ValueError(
        f"PME field length must be {grid.nx} or {grid.nx_total}, got {values.shape[0]}"
    )


def padded_values(u_interior: jax.Array, grid: PMEGrid) -> jax.Array:
    """Embed a cell field in ghosts, or retain a node-centred interior field."""
    values = jnp.asarray(u_interior, dtype=jnp.float64)
    if values.shape != (grid.nx,):
        raise ValueError(f"Expected interior shape {(grid.nx,)}, got {values.shape}")
    if isinstance(grid, NodeCenteredDirichletGrid):
        return values
    return jnp.zeros(grid.nx_total, dtype=values.dtype).at[grid.interior_slice].set(values)


def backward_euler_residual(
    u: jax.Array,
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
) -> jax.Array:
    """Return the interior backward-Euler residual for the regularized PME."""
    candidate = interior_values(u, grid)
    previous = interior_values(u_prev, grid)
    if isinstance(grid, NodeCenteredDirichletGrid):
        rhs = porous_medium_node_centered_rhs(candidate, grid, m, epsilon=epsilon)
    else:
        rhs = porous_medium_flux_rhs(padded_values(candidate, grid), grid, m, epsilon=epsilon)
        rhs = rhs[grid.interior_slice]
    return candidate - previous - dt * rhs


def make_backward_euler_residual(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
) -> Residual:
    """Build an interior-vector residual ``R(u) = u-u_prev-dt*F(u)``."""
    previous = interior_values(u_prev, grid)

    def residual(u: jax.Array) -> jax.Array:
        return backward_euler_residual(u, previous, grid, m, dt, epsilon)

    return residual


def d0_variant(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    epsilon: float,
    d0_kind: str,
    *,
    const_value: float = 1.0,
) -> float:
    """Select one named frozen-coefficient variant from an interior state."""
    previous = interior_values(u_prev, grid)
    if d0_kind == "frozen_mean":
        return d0_frozen_mean(previous, m)
    if d0_kind == "frozen_bulk":
        return d0_frozen_bulk(previous, m)
    if d0_kind == "floor":
        return d0_floor(m, epsilon)
    if d0_kind == "const":
        return d0_const(const_value)
    if d0_kind == "identity":
        return 0.0
    raise ValueError(f"Unknown D0 variant: {d0_kind!r}")


def pme_preconditioner_variant(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    *,
    const_value: float = 1.0,
) -> tuple[ExperimentalPreconditioner, float]:
    """Return the requested D0 preconditioner and the coefficient it uses."""
    d0 = d0_variant(u_prev, grid, m, epsilon, d0_kind, const_value=const_value)
    if d0_kind == "identity":
        return IdentityPreconditioner(), d0
    return pme_helmholtz_preconditioner(d0, dt, grid), d0


def build_pme_linearization(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    *,
    const_value: float = 1.0,
) -> PMELinearization:
    """Build a preconditioned PME Jacobian and the matching Newton right-hand side.

    The returned action is ``P^-1 J`` from the shared conditioning adapter and
    ``rhs`` is exactly ``P^-1 (-R(u_prev))``.  This makes the experimental
    counted GMRES measurement use the same left-preconditioned linear system
    as a Newton update, without relying on the core solver's budget counter.
    """
    with jax.enable_x64(True):
        previous = interior_values(u_prev, grid)
        residual = make_backward_euler_residual(previous, grid, m, dt, epsilon)
        preconditioner, d0 = pme_preconditioner_variant(
            previous,
            grid,
            m,
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
    return PMELinearization(operator, residual, preconditioner, context, d0, rhs)


def _counted_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    tol: float,
    max_iters: int,
) -> dict[str, Any]:
    """Run unrestarted matrix-free GMRES and retain its true residual history.

    This small experimental implementation uses Arnoldi least-squares updates
    and evaluates the true preconditioned residual after every iteration.
    It intentionally trades JIT fusion for an explicit, trustworthy iteration
    count; the operator itself remains matrix-free and JIT-compatible.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")

    with jax.enable_x64(True):
        vector_rhs = jnp.asarray(rhs, dtype=jnp.float64)
        norm_rhs = float(jnp.linalg.norm(vector_rhs))
        if norm_rhs == 0.0:
            return {"converged": True, "iterations": 0, "final_relative_residual": 0.0}

        basis = [vector_rhs / norm_rhs]
        hessenberg = jnp.zeros((max_iters + 1, max_iters), dtype=jnp.float64)
        threshold = float(jnp.sqrt(jnp.finfo(jnp.float64).eps))
        final_relative_residual = 1.0

        for column in range(max_iters):
            candidate_vector = jnp.real(matvec(basis[column]))
            coefficients = []
            for basis_vector in basis:
                coefficient = jnp.vdot(basis_vector, candidate_vector)
                coefficients.append(coefficient)
                candidate_vector = candidate_vector - coefficient * basis_vector
            for row, basis_vector in enumerate(basis):
                correction = jnp.vdot(basis_vector, candidate_vector)
                coefficients[row] = coefficients[row] + correction
                candidate_vector = candidate_vector - correction * basis_vector

            subdiagonal = jnp.linalg.norm(candidate_vector)
            hessenberg = hessenberg.at[: column + 1, column].set(jnp.asarray(coefficients))
            hessenberg = hessenberg.at[column + 1, column].set(subdiagonal)
            reduced_rhs = jnp.zeros(column + 2, dtype=jnp.float64).at[0].set(norm_rhs)
            least_squares = jnp.linalg.lstsq(
                hessenberg[: column + 2, : column + 1],
                reduced_rhs,
                rcond=None,
            )[0]
            correction = jnp.zeros_like(vector_rhs)
            for row, basis_vector in enumerate(basis):
                correction = correction + least_squares[row] * basis_vector
            residual = vector_rhs - jnp.real(matvec(correction))
            final_relative_residual = float(jnp.linalg.norm(residual) / norm_rhs)
            if final_relative_residual <= tol:
                return {
                    "converged": True,
                    "iterations": column + 1,
                    "final_relative_residual": final_relative_residual,
                }
            if float(subdiagonal) <= threshold:
                break
            basis.append(candidate_vector / subdiagonal)

    return {
        "converged": False,
        "iterations": len(basis),
        "final_relative_residual": final_relative_residual,
    }


def measure_gmres_iterations(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    *,
    tol: float,
    max_iters: int,
    const_value: float = 1.0,
) -> dict[str, Any]:
    """Measure the actual GMRES count for one preconditioned PME Newton system.

    The fixed system is ``P^-1 J delta = P^-1 (-R(u_prev))``.  Its count is
    determined from the true relative residual, unlike ``NKStats.lin_iters``
    which records only the configured budget in the current core solver.
    """
    linearization = build_pme_linearization(
        u_prev,
        grid,
        m,
        dt,
        epsilon,
        d0_kind,
        const_value=const_value,
    )
    measurement = _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=tol,
        max_iters=max_iters,
    )
    return {**measurement, "d0": linearization.d0}


def assess_pme_state(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    *,
    const_value: float = 1.0,
    n_angles: int = 6,
    fov_max_iters: int = 30,
    arnoldi_steps: int = 8,
    seed: int = 20260820,
) -> dict[str, Any]:
    """Assess one visited PME state under a named D0 preconditioner variant.

    The residual is differentiated through public JAX JVP/VJP calls by
    :func:`moljax.conditioning.linearized_operator`.  The returned operator is
    the left-preconditioned ``P^-1 J`` action and its Euclidean adjoint.
    """
    with jax.enable_x64(True):
        linearization = build_pme_linearization(
            u_prev,
            grid,
            m,
            dt,
            epsilon,
            d0_kind,
            const_value=const_value,
        )
        operator = linearization.operator
        d0 = linearization.d0
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
        "d0": float(d0),
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
