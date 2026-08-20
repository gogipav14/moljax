"""Regularized porous-medium diffusion utilities for experimental validation."""

from __future__ import annotations

import jax.numpy as jnp

from moljax.core.grid import Grid1D
from moljax.core.operators import laplacian_1d
from moljax.experimental.node_centered import NodeCenteredDirichletGrid


def _regularized_potential_with_dirichlet_zero(
    u: jnp.ndarray,
    grid: Grid1D,
    m: float,
    epsilon: float,
) -> jnp.ndarray:
    """Build ``(u**2 + epsilon**2)**(m/2)`` with zero-state boundary data."""
    ng = grid.n_ghost
    interior = grid.interior_slice
    phi_boundary = jnp.asarray(epsilon, dtype=u.dtype) ** m
    phi = jnp.zeros_like(u).at[interior].set((u[interior] ** 2 + epsilon**2) ** (m / 2.0))

    for offset in range(ng):
        phi = phi.at[ng - 1 - offset].set(2.0 * phi_boundary - phi[ng + offset])
        phi = phi.at[ng + grid.nx + offset].set(2.0 * phi_boundary - phi[ng + grid.nx - 1 - offset])

    return phi


def porous_medium_flux_rhs(
    u: jnp.ndarray,
    grid: Grid1D,
    m: float,
    *,
    epsilon: float = 1.0e-5,
    bc: str = "dirichlet0",
) -> jnp.ndarray:
    """Apply a regularized one-dimensional porous-medium diffusion operator.

    This returns a padded array containing the semi-discrete right-hand side
    for ``u_t = d²phi/dx²`` with ``phi = (u² + epsilon²)**(m/2)``.  The
    interior stencil is the conservative central-flux form
    ``((phi[i+1] - phi[i]) - (phi[i] - phi[i-1])) / dx²``, supplied by
    :func:`moljax.core.operators.laplacian_1d`.  Ghost values of ``phi`` are
    chosen so that the cell-face state boundary condition is ``u = 0``;
    consequently ``phi`` takes the regularized boundary value ``epsilon**m``.

    The state-based regularization is smooth and branchless, which keeps the
    operator differentiable with respect to ``u``.  It perturbs the unregularized
    porous-medium equation only near the zero state.

    Args:
        u: Padded field with shape ``(grid.nx_total,)``.
        grid: One-dimensional uniform grid describing ``u``.
        m: Porous-medium exponent.
        epsilon: Positive state regularization scale; zero recovers the
            positive-state linear control when ``m=1``.
        bc: Boundary condition selector.  Only homogeneous Dirichlet state
            data, ``"dirichlet0"``, are currently staged.

    Returns:
        Padded semi-discrete right-hand side with zeroed ghost entries.
    """
    if bc != "dirichlet0":
        raise ValueError(f"Unsupported boundary condition: {bc!r}")

    phi = _regularized_potential_with_dirichlet_zero(u, grid, m, epsilon)
    return laplacian_1d(phi, grid)


def porous_medium_node_centered_rhs(
    u: jnp.ndarray,
    grid: NodeCenteredDirichletGrid,
    m: float,
    *,
    epsilon: float = 1.0e-5,
) -> jnp.ndarray:
    """Apply the regularized PME operator on interior Dirichlet nodes.

    The unknowns occupy the interior nodes of a uniform grid whose boundary
    nodes are fixed at zero.  With
    ``phi(u) = (u**2 + epsilon**2)**(m/2)``, this uses the three-point stencil
    for ``d²phi/dx²`` after adjoining the boundary value ``epsilon**m``.  It
    therefore shares the node-centred Dirichlet convention of the DST-I
    Helmholtz inverse exactly.  The state regularization is smooth and
    branchless, so JAX can differentiate the operator directly.
    """
    interior = jnp.asarray(u)
    if interior.shape != (grid.nx,):
        raise ValueError(f"Expected shape {(grid.nx,)}, got {interior.shape}")
    phi = (interior**2 + epsilon**2) ** (m / 2.0)
    boundary = jnp.asarray(epsilon, dtype=interior.dtype) ** m
    padded = jnp.concatenate((boundary[None], phi, boundary[None]))
    return (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / grid.dx**2


def barenblatt(
    x: jnp.ndarray,
    t: float,
    m: float,
    *,
    d: int = 1,
    b: float,
    D0: float = 1.0,
) -> jnp.ndarray:
    """Evaluate the Barenblatt similarity solution of the unregularized PME.

    For ``u_t = D0 * Delta(u**m)``, the one-dimensional profile is
    ``s**(-alpha) * (b - ((m - 1) * beta / (2*m)) * y**2)_+**(1/(m - 1))``,
    with ``s=D0*t``, ``y=x/s**beta``, ``alpha=d/(d*(m-1)+2)``, and
    ``beta=1/(d*(m-1)+2)``.  This is an analytic reference for the true PME;
    ``porous_medium_flux_rhs`` instead uses a smooth state regularization.

    Args:
        x: Evaluation coordinates.
        t: Positive time.
        m: Porous-medium exponent greater than one.
        d: Spatial dimension in the similarity exponents.
        b: Similarity-profile height parameter.
        D0: Constant diffusion scale.

    Returns:
        Non-negative compactly supported similarity profile.
    """
    alpha = d / (d * (m - 1.0) + 2.0)
    beta = 1.0 / (d * (m - 1.0) + 2.0)
    scaled_time = D0 * t
    y = x / scaled_time**beta
    profile = b - ((m - 1.0) / (2.0 * m)) * beta * y**2
    return scaled_time ** (-alpha) * jnp.maximum(profile, 0.0) ** (1.0 / (m - 1.0))


def support_halfwidth(t: float, m: float, b: float, *, d: int = 1) -> jnp.ndarray:
    """Return the Barenblatt support radius for unit diffusion scale.

    The profile vanishes when ``b = ((m - 1) * beta / (2*m)) * y**2``.  Thus
    ``R(t) = t**beta * sqrt(2*m*b / ((m - 1)*beta))``, where
    ``beta = 1/(d*(m-1)+2)``.
    """
    beta = 1.0 / (d * (m - 1.0) + 2.0)
    return t**beta * jnp.sqrt(2.0 * m * b / ((m - 1.0) * beta))
