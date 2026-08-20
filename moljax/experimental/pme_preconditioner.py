"""Constant-coefficient Dirichlet preconditioners for experimental PME solves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from moljax.core.fft_nonperiodic import laplacian_symbol_dirichlet, solve_helmholtz_dirichlet
from moljax.core.grid import Grid1D


def d0_frozen_mean(u: jax.Array, m: float) -> float:
    """Return ``m * mean(u)**(m - 1)`` for an interior PME state."""
    with jax.enable_x64(True):
        mean = jnp.mean(jnp.asarray(u, dtype=jnp.float64))
        return float(m * mean ** (m - 1.0))


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
    grid: Grid1D,
) -> PMEHelmholtzPreconditioner:
    """Build the fixed-``d0`` DST Helmholtz preconditioner for one PME step."""
    with jax.enable_x64(True):
        symbol = laplacian_symbol_dirichlet(grid.nx, grid.dx, dtype=jnp.float64)
    return PMEHelmholtzPreconditioner(d0=float(d0), dt=float(dt), laplacian_symbol=symbol)
