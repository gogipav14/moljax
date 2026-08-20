"""Constant-coefficient Dirichlet preconditioners for experimental PME solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from moljax.core.fft_nonperiodic import laplacian_symbol_dirichlet, solve_helmholtz_dirichlet
from moljax.core.grid import Grid1D
from moljax.experimental.node_centered import (
    NodeCenteredDirichletGrid,
    node_centered_dirichlet_laplacian,
)

PMEGrid = Grid1D | NodeCenteredDirichletGrid


def d0_frozen_mean(u: jax.Array, m: float) -> float:
    """Return ``m * mean(u)**(m - 1)`` for an interior PME state."""
    with jax.enable_x64(True):
        mean = jnp.mean(jnp.asarray(u, dtype=jnp.float64))
        return float(m * mean ** (m - 1.0))


def d0_frozen_bulk(u: jax.Array, m: float, *, quantile: float = 0.9) -> float:
    """Return ``m * q(|u|)**(m - 1)`` using a robust bulk-state quantile.

    The default 90th percentile avoids diluting the frozen diffusivity by the
    large zero-state region surrounding a compact porous-medium profile.
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must lie in (0, 1]")
    with jax.enable_x64(True):
        reference = jnp.quantile(jnp.abs(jnp.asarray(u, dtype=jnp.float64)), quantile)
        return float(m * reference ** (m - 1.0))


def d0_floor(m: float, epsilon: float) -> float:
    """Return the regularization-floor coefficient ``m * epsilon**(m - 1)``."""
    return float(m * epsilon ** (m - 1.0))


def d0_const(value: float) -> float:
    """Return a fixed user-supplied constant diffusion coefficient."""
    return float(value)


@dataclass(frozen=True)
class PMEHelmholtzPreconditioner:
    """Dirichlet Helmholtz inverse with a fixed porous-medium coefficient.

    This stages the paper's ``M = I - dt * D * Laplacian`` preconditioner
    (Section 3.3.3), with ``D`` frozen to ``d0``.  It acts on interior
    Dirichlet-0 degrees of freedom using the DST-I solver, rather than the
    periodic FFT diffusion preconditioner.
    """

    d0: float
    dt: float
    laplacian_symbol: jax.Array

    def apply(self, residual: jax.Array, context: Any = None) -> jax.Array:
        """Apply ``(I - dt * d0 * Laplacian_h)^-1`` to an interior residual."""
        del context
        return solve_helmholtz_dirichlet(residual, self.laplacian_symbol, self.dt, self.d0)

    def __call__(self, residual: jax.Array) -> jax.Array:
        """Apply the same linear map through a simple one-argument callable."""
        return self.apply(residual)


def pme_helmholtz_preconditioner(
    d0: float,
    dt: float,
    grid: PMEGrid,
) -> PMEHelmholtzPreconditioner:
    """Build the fixed-``d0`` DST Helmholtz preconditioner for one PME step."""
    with jax.enable_x64(True):
        symbol = laplacian_symbol_dirichlet(grid.nx, grid.dx, dtype=jnp.float64)
    return PMEHelmholtzPreconditioner(d0=float(d0), dt=float(dt), laplacian_symbol=symbol)


def cell_centered_dirichlet_laplacian(values: jax.Array, grid: Grid1D) -> jax.Array:
    """Apply the cell-centred zero-face Laplacian used by the original PME RHS."""
    interior = jnp.asarray(values)
    if interior.shape != (grid.nx,):
        raise ValueError(f"Expected shape {(grid.nx,)}, got {interior.shape}")
    padded = jnp.zeros(grid.nx_total, dtype=interior.dtype).at[grid.interior_slice].set(interior)
    left_ghost = grid.n_ghost - 1
    right_ghost = grid.n_ghost + grid.nx
    padded = padded.at[left_ghost].set(-interior[0])
    padded = padded.at[right_ghost].set(-interior[-1])
    start = grid.n_ghost
    return (
        padded[start + 1 : start + grid.nx + 1]
        - 2.0 * padded[start : start + grid.nx]
        + padded[start - 1 : start + grid.nx - 1]
    ) / grid.dx**2


def helmholtz_inverse_relative_residual(
    d0: float,
    dt: float,
    grid: PMEGrid,
    key: jax.Array,
) -> float:
    """Measure the residual of the DST inverse against a grid's Laplacian.

    A node-centred grid should give roundoff-level residual because its
    Laplacian diagonalizes in DST-I.  Passing a cell-centred :class:`Grid1D`
    deliberately quantifies the former centring mismatch instead.
    """
    with jax.enable_x64(True):
        rhs = jax.random.normal(key, (grid.nx,), dtype=jnp.float64)
        solution = pme_helmholtz_preconditioner(d0, dt, grid).apply(rhs)
        if isinstance(grid, Grid1D):
            laplacian = cell_centered_dirichlet_laplacian(solution, grid)
        else:
            laplacian = node_centered_dirichlet_laplacian(solution, grid)
        residual = solution - dt * d0 * laplacian - rhs
        return float(jnp.linalg.norm(residual) / jnp.linalg.norm(rhs))
