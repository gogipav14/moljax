"""DST/tau-style variable-coefficient preconditioners for experimental PME systems.

The node-centred porous-medium backward-Euler Jacobian has the ordering
``I - dt * L_D * diag(D(u))``, where ``L_D`` is the fixed Dirichlet
second-difference and ``D(u)`` is a state-dependent diagonal coefficient.
This module approximates its inverse by an input-weighted blend of
constant-coefficient DST-I Helmholtz inverses.  It is deliberately staged as
an experimental feasibility probe rather than a general solver component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from moljax._precision import require_x64
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
from moljax.core.fft_nonperiodic import laplacian_symbol_dirichlet, solve_helmholtz_dirichlet
from moljax.core.preconditioners import PrecondContext
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import (
    Residual,
    _counted_gmres,
    interior_values,
    make_backward_euler_residual,
)
from moljax.experimental.pme_preconditioner import (
    PMEHelmholtzPreconditioner,
    pme_helmholtz_preconditioner,
)


class TauBlendLinearization(NamedTuple):
    """One consistently preconditioned PME linear system."""

    operator: LinearizedOperator
    residual: Residual
    preconditioner: DSTTauBlendPreconditioner
    context: PrecondContext
    rhs: jax.Array


class SingleReferenceLinearization(NamedTuple):
    """One PME system preconditioned by a single explicit DST-I reference."""

    operator: LinearizedOperator
    residual: Residual
    preconditioner: PMEHelmholtzPreconditioner
    context: PrecondContext
    d0: float
    rhs: jax.Array


def porous_medium_diffusivity(u: jax.Array, m: float, epsilon: float) -> jax.Array:
    """Return the diagonal coefficient ``phi'(u)`` of the staged PME flux."""
    values = jnp.asarray(u, dtype=jnp.float64)
    return m * values * (values**2 + epsilon**2) ** (m / 2.0 - 1.0)


def _active_reference_values(
    diffusivity: jax.Array,
    reference_count: int,
    active_relative: float,
    active_absolute: float,
) -> tuple[jax.Array, jax.Array]:
    """Return positive reference values and the compact-support active mask."""
    if reference_count < 1:
        raise ValueError("reference_count must be positive")
    if not 0.0 < active_relative < 1.0:
        raise ValueError("active_relative must lie in (0, 1)")
    if active_absolute <= 0.0:
        raise ValueError("active_absolute must be positive")

    nonnegative = jnp.maximum(jnp.asarray(diffusivity, dtype=jnp.float64), 0.0)
    maximum = jnp.max(nonnegative)
    threshold = jnp.maximum(active_absolute, active_relative * maximum)
    active = nonnegative > threshold
    fallback = jnp.maximum(threshold, active_absolute)
    quantiles = jnp.linspace(0.0, 1.0, reference_count, dtype=jnp.float64)
    if not bool(jnp.any(active)):
        return jnp.full((reference_count,), fallback, dtype=jnp.float64), active
    references = jnp.quantile(nonnegative[active], quantiles)
    return references, active


def pme_active_single_reference_values(
    u: jax.Array,
    m: float,
    grid: NodeCenteredDirichletGrid,
    epsilon: float,
    *,
    active_relative: float = 1.0e-6,
    active_absolute: float = 1.0e-14,
) -> dict[str, float | int]:
    """Return active-support geometric and harmonic PME diffusivity references.

    The active support is exactly the one used by the tau blend.  Both values
    therefore differ from ``frozen_mean`` only in the scalar ``d0`` supplied
    to the same DST-I Helmholtz inverse.
    """
    require_x64("PME active-support reference calculation")
    values = interior_values(u, grid)
    diffusivity = jnp.maximum(porous_medium_diffusivity(values, m, epsilon), 0.0)
    maximum = float(jnp.max(diffusivity))
    threshold = max(active_absolute, active_relative * maximum)
    active = diffusivity > threshold
    active_values = diffusivity[active]
    if active_values.size == 0:
        raise ValueError("PME state has no active diffusivity values")
    minimum_active = float(jnp.min(active_values))
    maximum_active = float(jnp.max(active_values))
    geometric = float(jnp.sqrt(minimum_active * maximum_active))
    harmonic = float(active_values.size / jnp.sum(1.0 / active_values))
    return {
        "active_threshold": threshold,
        "active_node_count": int(active_values.size),
        "inactive_node_count": int(diffusivity.size - active_values.size),
        "degeneracy_fraction": float(jnp.mean(diffusivity < threshold)),
        "active_minimum": minimum_active,
        "active_maximum": maximum_active,
        "geometric_mean": geometric,
        "harmonic_mean": harmonic,
    }


def _log_partition_weights(
    diffusivity: jax.Array,
    references: jax.Array,
    active: jax.Array,
) -> jax.Array:
    """Build an input-node log-coefficient partition of unity."""
    values = jnp.maximum(jnp.asarray(diffusivity, dtype=jnp.float64), jnp.finfo(jnp.float64).tiny)
    log_values = jnp.log(values)[:, None]
    log_references = jnp.log(jnp.asarray(references, dtype=jnp.float64))[None, :]
    width = jnp.maximum(jnp.ptp(log_references) / max(references.size - 1, 1), 1.0e-12)
    unnormalized = jnp.exp(-(((log_values - log_references) / width) ** 2))
    normalized = unnormalized / jnp.sum(unnormalized, axis=1, keepdims=True)
    return jnp.where(active[:, None], normalized, 0.0).T


@dataclass(frozen=True)
class DSTTauBlendPreconditioner:
    """Input-weighted blend of Dirichlet Helmholtz inverses.

    ``apply`` approximates the inverse of ``I - dt * L_D * diag(D)`` as
    ``W0 + sum_w H(d_w) W_w``.  The right-side weights ``W_w`` follow the
    column-wise coefficient ordering of ``L_D * diag(D)``.  Nodes below the
    active threshold retain an identity source term in ``W0`` and never select
    a zero-coefficient reference inverse; DST propagation from active sources
    is intentionally retained.
    """

    dt: float
    laplacian_symbol: jax.Array
    reference_values: jax.Array
    input_weights: jax.Array
    inactive_weight: jax.Array
    name: str = "pme_dst_tau_blend"

    @property
    def reference_count(self) -> int:
        """Return the static number of DST Helmholtz inverses per application."""
        return int(self.reference_values.size)

    def apply(self, residual: jax.Array, context: Any = None) -> jax.Array:
        """Apply the matrix-free input-weighted DST blend."""
        del context
        values = jnp.asarray(residual, dtype=jnp.float64)
        weighted_rhs = self.input_weights * values[None, :]

        def solve_one(reference: jax.Array, rhs: jax.Array) -> jax.Array:
            return solve_helmholtz_dirichlet(rhs, self.laplacian_symbol, self.dt, reference)

        solved = jax.vmap(solve_one)(self.reference_values, weighted_rhs)
        return self.inactive_weight * values + jnp.sum(solved, axis=0)


def pme_dst_tau_blend_preconditioner(
    u: jax.Array,
    m: float,
    dt: float,
    grid: NodeCenteredDirichletGrid,
    epsilon: float,
    *,
    reference_count: int,
    active_relative: float = 1.0e-6,
    active_absolute: float = 1.0e-14,
) -> DSTTauBlendPreconditioner:
    """Build an experimental input-weighted DST/tau blend at one PME state."""
    require_x64("PME DST/tau-blend preconditioner")
    values = interior_values(u, grid)
    diffusivity = porous_medium_diffusivity(values, m, epsilon)
    references, active = _active_reference_values(
        diffusivity,
        reference_count,
        active_relative,
        active_absolute,
    )
    weights = _log_partition_weights(diffusivity, references, active)
    symbol = laplacian_symbol_dirichlet(grid.nx, grid.dx, dtype=jnp.float64)
    return DSTTauBlendPreconditioner(
        dt=float(dt),
        laplacian_symbol=symbol,
        reference_values=references,
        input_weights=weights,
        inactive_weight=(~active).astype(jnp.float64),
    )


def build_pme_tau_blend_linearization(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: float,
    dt: float,
    epsilon: float,
    *,
    reference_count: int,
    active_relative: float = 1.0e-6,
    active_absolute: float = 1.0e-14,
) -> TauBlendLinearization:
    """Build the shared-conditioning linearization for a DST/tau blend."""
    previous = interior_values(u_prev, grid)
    residual = make_backward_euler_residual(previous, grid, m, dt, epsilon)
    preconditioner = pme_dst_tau_blend_preconditioner(
        previous,
        m,
        dt,
        grid,
        epsilon,
        reference_count=reference_count,
        active_relative=active_relative,
        active_absolute=active_absolute,
    )
    context = PrecondContext(grid=grid, dt=dt, params={})
    operator = linearized_operator(
        residual,
        previous,
        preconditioner=preconditioner,
        context=context,
    )
    rhs = preconditioner.apply(-residual(previous), context)
    return TauBlendLinearization(operator, residual, preconditioner, context, rhs)


def build_pme_single_reference_linearization(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: float,
    dt: float,
    epsilon: float,
    *,
    d0: float,
) -> SingleReferenceLinearization:
    """Build a PME linearization using one explicit DST-I Helmholtz reference."""
    require_x64("PME single-reference linearization")
    if d0 < 0.0 or not jnp.isfinite(d0):
        raise ValueError("d0 must be finite and non-negative")
    previous = interior_values(u_prev, grid)
    residual = make_backward_euler_residual(previous, grid, m, dt, epsilon)
    preconditioner = pme_helmholtz_preconditioner(d0, dt, grid)
    context = PrecondContext(grid=grid, dt=dt, params={})
    operator = linearized_operator(
        residual,
        previous,
        preconditioner=preconditioner,
        context=context,
    )
    rhs = preconditioner.apply(-residual(previous), context)
    return SingleReferenceLinearization(
        operator,
        residual,
        preconditioner,
        context,
        float(d0),
        rhs,
    )


def measure_pme_tau_blend_gmres_iterations(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: float,
    dt: float,
    epsilon: float,
    *,
    reference_count: int,
    tol: float,
    max_iters: int,
    active_relative: float = 1.0e-6,
    active_absolute: float = 1.0e-14,
) -> dict[str, float | int | bool]:
    """Measure a true residual-history GMRES count for the DST/tau blend."""
    require_x64("PME DST/tau-blend GMRES measurement")
    linearization = build_pme_tau_blend_linearization(
        u_prev,
        grid,
        m,
        dt,
        epsilon,
        reference_count=reference_count,
        active_relative=active_relative,
        active_absolute=active_absolute,
    )
    return _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=tol,
        max_iters=max_iters,
    )


def measure_pme_single_reference_gmres_iterations(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: float,
    dt: float,
    epsilon: float,
    *,
    d0: float,
    tol: float,
    max_iters: int,
) -> dict[str, float | int | bool]:
    """Measure counted GMRES for one explicit single-reference DST-I solve."""
    require_x64("PME single-reference GMRES measurement")
    linearization = build_pme_single_reference_linearization(
        u_prev,
        grid,
        m,
        dt,
        epsilon,
        d0=d0,
    )
    measurement = _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=tol,
        max_iters=max_iters,
    )
    return {**measurement, "d0": float(d0)}


def assess_pme_tau_blend_state(
    u_prev: jax.Array,
    grid: NodeCenteredDirichletGrid,
    m: float,
    dt: float,
    epsilon: float,
    *,
    reference_count: int,
    n_angles: int = 6,
    fov_max_iters: int = 30,
    arnoldi_steps: int = 8,
    seed: int = 20261010,
    active_relative: float = 1.0e-6,
    active_absolute: float = 1.0e-14,
) -> dict[str, float | int | bool | str | dict[str, float]]:
    """Assess an experimental DST/tau blend through the shared toolbox."""
    require_x64("PME DST/tau-blend assessment")
    linearization = build_pme_tau_blend_linearization(
        u_prev,
        grid,
        m,
        dt,
        epsilon,
        reference_count=reference_count,
        active_relative=active_relative,
        active_absolute=active_absolute,
    )
    operator = linearization.operator
    adjoint_error = adjoint_identity(operator, jax.random.PRNGKey(seed), operator.n)
    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed + 1))
    start = jax.random.normal(key_real, (operator.n,), dtype=jnp.float64)
    start = start + 1j * jax.random.normal(key_imag, (operator.n,), dtype=jnp.float64)
    coverage = arnoldi(operator.matvec, start, min(arnoldi_steps, operator.n))
    ritz = ritz_values(coverage.hessenberg)
    epsilon_at_zero = epsilon_zero(coverage.hessenberg)
    field_of_values = numerical_range(
        operator.matvec,
        operator.matvec_adjoint,
        operator.n,
        n_angles=n_angles,
        max_iters=fov_max_iters,
    )
    rates = estimate_rates(field_of_values, ritz)
    assessment = assess_preconditioner(
        field_of_values,
        ritz,
        epsilon_at_zero,
        coverage=coverage,
    )
    predicted = rates.predicted_gmres_factor
    return {
        "reference_count": reference_count,
        "adjoint_error": float(adjoint_error),
        "adjoint_tolerance": 1.0e-8,
        "disk_rate": float(assessment.disk_rate),
        "epsilon_zero": float(assessment.epsilon_zero),
        "epsilon_zero_full_operator_evidence": bool(assessment.epsilon_zero_full_operator_evidence),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "supports_consistent": bool(field_of_values.supports_consistent),
        "corroboration_attempted": bool(field_of_values.corroboration_attempted),
        "verdict": assessment.verdict,
        "verdict_reason": assessment.verdict_reason,
        "n_right_real_outliers": (
            None
            if assessment.n_right_real_outliers is None
            else int(assessment.n_right_real_outliers)
        ),
        "rates": {
            "r1": float(rates.r1),
            "r2": float(rates.r2),
            "r3": float(rates.r3),
            "predicted_gmres_factor": None if predicted is None else float(predicted),
            "agree": bool(rates.agree),
            "supports_consistent": bool(rates.supports_consistent),
            "corroboration_attempted": bool(rates.corroboration_attempted),
        },
    }


__all__ = [
    "DSTTauBlendPreconditioner",
    "SingleReferenceLinearization",
    "TauBlendLinearization",
    "assess_pme_tau_blend_state",
    "build_pme_single_reference_linearization",
    "build_pme_tau_blend_linearization",
    "measure_pme_single_reference_gmres_iterations",
    "measure_pme_tau_blend_gmres_iterations",
    "pme_active_single_reference_values",
    "pme_dst_tau_blend_preconditioner",
    "porous_medium_diffusivity",
]
