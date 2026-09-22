"""Experimental conditioning diagnostics for backward-Euler porous-medium steps."""

from __future__ import annotations

from collections.abc import Callable
from math import hypot, log, sqrt
from time import perf_counter
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
    full_operator_epsilon_zero,
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
        return d0_frozen_mean(previous, m, epsilon=epsilon)
    if d0_kind == "frozen_bulk":
        return d0_frozen_bulk(previous, m, epsilon=epsilon)
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


def _candidate_relative_residual(
    matvec: Callable[[jax.Array], jax.Array],
    vector_rhs: jax.Array,
    norm_rhs: float,
    basis: list[jax.Array],
    triangular: list[list[float]],
    rotated_rhs: list[float],
    n_columns: int,
) -> float:
    """Return the measured ``||b - A x|| / ||b||`` of one GMRES candidate.

    ``triangular`` holds the rotated Hessenberg columns, so the candidate
    coefficients follow from back substitution against the leading
    ``n_columns`` rotated right-hand-side entries.  This costs one extra
    matrix-vector product and is only used where the rotated residual estimate
    is not trustworthy.
    """
    if n_columns == 0:
        return float(jnp.linalg.norm(vector_rhs)) / norm_rhs
    weights = [0.0] * n_columns
    for row in reversed(range(n_columns)):
        total = rotated_rhs[row]
        for column in range(row + 1, n_columns):
            total -= triangular[column][row] * weights[column]
        weights[row] = total / triangular[row][row]
    candidate = jnp.zeros_like(vector_rhs)
    for weight, basis_vector in zip(weights, basis[:n_columns], strict=True):
        candidate = candidate + weight * basis_vector
    residual = vector_rhs - jnp.real(matvec(candidate))
    return float(jnp.linalg.norm(residual)) / norm_rhs


def _counted_gmres(
    matvec: Callable[[jax.Array], jax.Array],
    rhs: jax.Array,
    *,
    tol: float,
    max_iters: int,
) -> dict[str, Any]:
    """Run unrestarted matrix-free GMRES and retain its true residual history.

    This small experimental implementation applies Givens rotations to the
    Arnoldi Hessenberg columns.  The final rotated right-hand-side entry is
    the GMRES residual norm, so no dense least-squares solve is needed at each
    step.  It intentionally trades JIT fusion for an explicit, trustworthy
    iteration count; the operator itself remains matrix-free and JIT-compatible.

    The Krylov space stops growing only when the Arnoldi subdiagonal
    ``h[k+1, k] = ||w||`` left after orthogonalizing ``w = A v_k`` vanishes
    relative to that column's own scale ``||A v_k||``; the cutoff is
    ``h[k+1, k] <= 64 eps ||A v_k||``, the same scale-relative factor the
    Arnoldi factorization in ``moljax.conditioning.pseudospectra`` uses.  A
    small rotated pivot is never a reason to stop on its own: the pivot
    ``hypot(a, h[k+1, k])`` is nonzero whenever the subdiagonal is, and a
    pivot that is small relative to a global historical scale says nothing
    about whether the Krylov space can still grow.

    At such an invariant-subspace ("happy") breakdown the space cannot grow
    further.  If the rotated diagonal of that column also vanishes relative to
    ``||A v_k||``, the operator restricted to the Krylov space is singular:
    zeroing the Givens sine there would make the rotated right-hand-side entry
    vanish whatever the right-hand side is, so the stats instead report
    ``breakdown`` with ``converged`` False and the measured
    ``||b - A x|| / ||b||`` of the best candidate the previous columns
    support.  Otherwise the upper-triangular system is solved and convergence
    is decided by the measured ``||b - A x|| / ||b||`` of its candidate, not
    by the rotated estimate, whose last factor is a roundoff-level sine.  A
    candidate that meets ``tol`` is convergence, reported with the rotated
    GMRES residual like every other converged return; one that misses it is a
    breakdown, reported with the measured residual.  Exhausting ``max_iters``
    without meeting ``tol`` is non-convergence, not breakdown.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if max_iters <= 0:
        raise ValueError("max_iters must be positive")

    vector_rhs = jnp.asarray(rhs, dtype=jnp.float64)
    norm_rhs = float(jnp.linalg.norm(vector_rhs))
    if norm_rhs == 0.0:
        return {
            "converged": True,
            "iterations": 0,
            "final_relative_residual": 0.0,
            "breakdown": False,
        }

    basis = [vector_rhs / norm_rhs]
    cosines: list[float] = []
    sines: list[float] = []
    triangular: list[list[float]] = []
    rotated_rhs = [norm_rhs] + [0.0] * max_iters
    breakdown_factor = 64.0 * float(jnp.finfo(jnp.float64).eps)
    final_relative_residual = 1.0

    for column in range(max_iters):
        candidate_vector = jnp.real(matvec(basis[column]))
        column_scale = float(jnp.linalg.norm(candidate_vector))
        coefficients: list[float] = []
        for basis_vector in basis:
            coefficient = float(jnp.vdot(basis_vector, candidate_vector))
            coefficients.append(coefficient)
            candidate_vector = candidate_vector - coefficient * basis_vector
        for row, basis_vector in enumerate(basis):
            correction = float(jnp.vdot(basis_vector, candidate_vector))
            coefficients[row] = coefficients[row] + correction
            candidate_vector = candidate_vector - correction * basis_vector

        arnoldi_subdiagonal = float(jnp.linalg.norm(candidate_vector))
        invariant = arnoldi_subdiagonal <= breakdown_factor * column_scale
        hessenberg_column = coefficients + [arnoldi_subdiagonal]
        for row, (cosine, sine) in enumerate(zip(cosines, sines, strict=True)):
            upper = cosine * hessenberg_column[row] + sine * hessenberg_column[row + 1]
            hessenberg_column[row + 1] = (
                -sine * hessenberg_column[row] + cosine * hessenberg_column[row + 1]
            )
            hessenberg_column[row] = upper

        diagonal = hessenberg_column[column]
        if invariant and abs(diagonal) <= breakdown_factor * column_scale:
            measured = _candidate_relative_residual(
                matvec,
                vector_rhs,
                norm_rhs,
                basis,
                triangular,
                rotated_rhs,
                column,
            )
            return {
                "converged": False,
                "iterations": column + 1,
                "final_relative_residual": measured,
                "breakdown": True,
            }

        normalization = hypot(diagonal, arnoldi_subdiagonal)
        cosine, sine = diagonal / normalization, arnoldi_subdiagonal / normalization
        cosines.append(cosine)
        sines.append(sine)
        hessenberg_column[column] = normalization
        hessenberg_column[column + 1] = 0.0
        triangular.append(hessenberg_column[: column + 1])
        previous_rhs = rotated_rhs[column]
        rotated_rhs[column] = cosine * previous_rhs
        rotated_rhs[column + 1] = -sine * previous_rhs
        final_relative_residual = abs(rotated_rhs[column + 1]) / norm_rhs
        if invariant:
            measured = _candidate_relative_residual(
                matvec,
                vector_rhs,
                norm_rhs,
                basis,
                triangular,
                rotated_rhs,
                column + 1,
            )
            if measured <= tol:
                return {
                    "converged": True,
                    "iterations": column + 1,
                    "final_relative_residual": final_relative_residual,
                    "breakdown": False,
                }
            return {
                "converged": False,
                "iterations": column + 1,
                "final_relative_residual": measured,
                "breakdown": True,
            }
        if final_relative_residual <= tol:
            return {
                "converged": True,
                "iterations": column + 1,
                "final_relative_residual": final_relative_residual,
                "breakdown": False,
            }
        basis.append(candidate_vector / arnoldi_subdiagonal)

    return {
        "converged": False,
        "iterations": len(basis),
        "final_relative_residual": final_relative_residual,
        "breakdown": False,
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


def predicted_iterations_from_envelope(
    disk_rate: float,
    *,
    tol: float,
    prefactor: float = 1.0 + sqrt(2.0),
) -> float:
    """Return the Crouzeix--Palencia envelope iteration estimate.

    :func:`moljax.conditioning.crouzeix_palencia_envelope` bounds a residual
    after ``k`` iterations by ``prefactor * disk_rate**k``.  Solving that
    bound for the first ``k`` at or below ``tol`` gives
    ``log(tol / prefactor) / log(disk_rate)``.  The result is a real-valued
    estimate; callers take its ceiling when comparing it with an integer
    iteration count.  A non-positive disk rate predicts zero iterations,
    while a rate at least one returns ``inf`` to mark the bound as
    non-predictive.
    """
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    if prefactor <= 0.0:
        raise ValueError("prefactor must be positive")
    if disk_rate <= 0.0:
        return 0.0
    if disk_rate >= 1.0:
        return float("inf")
    return max(0.0, log(tol / prefactor) / log(disk_rate))


def assess_pme_state(
    u_prev: jax.Array,
    grid: PMEGrid,
    m: float,
    dt: float,
    epsilon: float,
    d0_kind: str,
    *,
    const_value: float = 1.0,
    n_angles: int = 16,
    fov_max_iters: int = 60,
    fov_residual_tolerance: float = 1.0e-3,
    fov_n_restarts: int = 2,
    arnoldi_steps: int = 8,
    seed: int = 20260820,
    full_operator_epsilon_evidence: bool = False,
) -> dict[str, Any]:
    """Assess one visited PME state under a named D0 preconditioner variant.

    The residual is differentiated through public JAX JVP/VJP calls by
    :func:`moljax.conditioning.linearized_operator`.  The returned operator is
    the left-preconditioned ``P^-1 J`` action and its Euclidean adjoint.
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
    operator = linearization.operator
    d0 = linearization.d0
    adjoint_error = adjoint_identity(operator, jax.random.PRNGKey(seed), operator.n)
    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed + 1))
    start = jax.random.normal(key_real, (operator.n,), dtype=jnp.float64)
    start = start + 1j * jax.random.normal(key_imag, (operator.n,), dtype=jnp.float64)
    arnoldi_result = arnoldi(operator.matvec, start, min(arnoldi_steps, operator.n))
    hessenberg = arnoldi_result.hessenberg
    ritz = ritz_values(hessenberg)
    reduced_epsilon_at_zero = epsilon_zero(hessenberg)
    field_of_values = numerical_range(
        operator.matvec,
        operator.matvec_adjoint,
        operator.n,
        n_angles=n_angles,
        max_iters=fov_max_iters,
        residual_tolerance=fov_residual_tolerance,
        n_restarts=fov_n_restarts,
    )
    rates = estimate_rates(field_of_values, ritz)
    full_epsilon_seconds = None
    if full_operator_epsilon_evidence:
        full_epsilon_started_at = perf_counter()
        full_epsilon_at_zero = full_operator_epsilon_zero(operator.matvec, operator.n)
        full_epsilon_seconds = perf_counter() - full_epsilon_started_at
    else:
        full_epsilon_at_zero = None
    epsilon_at_zero = (
        full_epsilon_at_zero if full_epsilon_at_zero is not None else reduced_epsilon_at_zero
    )
    assessment = assess_preconditioner(
        field_of_values,
        ritz,
        epsilon_at_zero,
        coverage=arnoldi_result,
        full_operator_lower_bound=full_operator_epsilon_evidence,
    )

    return {
        "d0_kind": d0_kind,
        "d0": float(d0),
        "operator_dimension": operator.n,
        "adjoint_error": float(adjoint_error),
        "adjoint_tolerance": 1.0e-8,
        "verdict": assessment.verdict,
        "disk_rate": float(assessment.disk_rate),
        "epsilon_zero": float(assessment.epsilon_zero),
        "epsilon_zero_reduced_arnoldi": float(reduced_epsilon_at_zero),
        "full_operator_epsilon_zero": (
            None if full_epsilon_at_zero is None else float(full_epsilon_at_zero)
        ),
        "full_operator_epsilon_zero_seconds": full_epsilon_seconds,
        "epsilon_zero_full_operator_evidence": bool(assessment.epsilon_zero_full_operator_evidence),
        "verdict_reason": assessment.verdict_reason,
        "arnoldi_k_requested": int(arnoldi_result.k_requested),
        "arnoldi_k_achieved": int(arnoldi_result.k_achieved),
        "arnoldi_breakdown": bool(arnoldi_result.breakdown),
        "arnoldi_residual_norm": float(arnoldi_result.residual_norm),
        "predicted_gmres_factor": (
            None
            if assessment.predicted_gmres_factor is None
            else float(assessment.predicted_gmres_factor)
        ),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "n_right_real_outliers": (
            None
            if assessment.n_right_real_outliers is None
            else int(assessment.n_right_real_outliers)
        ),
        "max_support_residual": float(field_of_values.max_support_residual),
        "supports_converged": bool(field_of_values.supports_converged),
        "supports_corroborated": bool(field_of_values.supports_corroborated),
        "supports_consistent": bool(field_of_values.supports_consistent),
        "corroboration_attempted": bool(field_of_values.corroboration_attempted),
        "rates": {name: value for name, value in rates._asdict().items()},
    }
