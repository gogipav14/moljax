"""Node-centred experimental operators for the degenerate Porous--Fisher equation."""

from __future__ import annotations

from math import sqrt

import jax
import jax.numpy as jnp

from moljax.experimental.node_centered import (
    NodeCenteredDirichletGrid,
    node_centered_dirichlet_laplacian,
)


def porous_fisher_potential(u: jax.Array, *, epsilon: float = 1.0e-5) -> jax.Array:
    """Return ``phi(u) = u**2 + epsilon**2`` for degenerate diffusion.

    Its derivative is ``phi'(u) = 2*u``, so the flux form
    ``d_xx(phi(u))`` has density-dependent diffusivity ``D(u) = 2*u``.
    The additive ``epsilon**2`` is retained for a common regularized-potential
    interface and cancels under the Laplacian for this ``m=2`` case.
    """
    values = jnp.asarray(u, dtype=jnp.float64)
    return values**2 + epsilon**2


def porous_fisher_rhs(
    u: jax.Array,
    grid: NodeCenteredDirichletGrid,
    *,
    r: float,
    epsilon: float = 1.0e-5,
) -> jax.Array:
    """Return the node-centred Porous--Fisher RHS.

    This discretizes ``u_t = d_xx(u**2 + epsilon**2) + r*u*(1-u)``.
    The diffusion is the degenerate flux form with ``D(u)=2*u``.  The
    node-centred stencil enforces homogeneous Dirichlet values at both finite
    endpoints.  This matches the zero state ahead of a right-moving front;
    the left endpoint is placed far behind the front, where validation excludes
    its artificial boundary layer.

    The arithmetic path is branchless and remains differentiable with respect
    to the interior state.
    """
    values = jnp.asarray(u, dtype=jnp.float64)
    if values.shape != (grid.nx,):
        raise ValueError(f"Expected shape {(grid.nx,)}, got {values.shape}")
    diffusion = node_centered_dirichlet_laplacian(
        porous_fisher_potential(values, epsilon=epsilon), grid
    )
    return diffusion + r * values * (1.0 - values)


def wave_speed(r: float, *, m: int = 2) -> float:
    """Return the admissible sharp-wave speed for the exact ``m=2`` profile.

    For ``u_t = (u**2)_xx + r*u*(1-u)``, the explicit sharp profile below has
    ``c = sqrt(r)``.  The closed form is specific to the logistic ``m=2``
    equation, so other exponents are rejected instead of being mislabeled as
    exact solutions.
    """
    if m != 2:
        raise ValueError("The implemented logistic travelling-wave reference is exact only for m=2")
    if r < 0.0:
        raise ValueError("r must be non-negative")
    return sqrt(r)


def porous_fisher_traveling_wave(
    x: jax.Array,
    t: float,
    *,
    r: float,
    c: float,
    m: int = 2,
    xi0: float = 0.0,
) -> jax.Array:
    """Return the exact sharp ``m=2`` Porous--Fisher travelling wave.

    With ``xi = x - c*t - xi0`` and ``c=sqrt(r)``, this is
    ``u(xi) = max(1 - exp((c/2)*xi), 0)``.  It solves the unregularized
    ``u_t=(u**2)_xx + r*u*(1-u)`` away from its sharp compact-support edge.
    The positive part is analytic-reference logic, not an operator hot path.
    """
    admissible_speed = wave_speed(r, m=m)
    if abs(c - admissible_speed) > 1.0e-12 * max(1.0, admissible_speed):
        raise ValueError(f"Exact m=2 travelling wave requires c={admissible_speed}")
    coordinate = jnp.asarray(x, dtype=jnp.float64) - c * t - xi0
    return jnp.maximum(1.0 - jnp.exp(0.5 * c * coordinate), 0.0)


def wave_front_position(t: float, *, c: float, xi0: float = 0.0) -> float:
    """Return the compact-support edge position ``xi=0`` of the wave."""
    return xi0 + c * t


__all__ = [
    "porous_fisher_potential",
    "porous_fisher_rhs",
    "porous_fisher_traveling_wave",
    "wave_front_position",
    "wave_speed",
]
