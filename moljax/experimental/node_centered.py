"""Node-centred Dirichlet discretization helpers for experimental PME studies."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NodeCenteredDirichletGrid:
    """Uniform interior nodes with homogeneous Dirichlet boundary nodes.

    ``nx`` is the number of unknowns at ``x_min + dx, ..., x_max - dx`` and
    ``dx = (x_max - x_min) / (nx + 1)``.  This matches the DST-I Dirichlet
    Helmholtz solver used by the experimental frozen-coefficient preconditioner.
    """

    nx: int
    x_min: float
    x_max: float

    @classmethod
    def uniform(cls, nx: int, x_min: float, x_max: float) -> NodeCenteredDirichletGrid:
        """Create a uniform node-centred Dirichlet grid."""
        return cls(nx=nx, x_min=x_min, x_max=x_max)

    @property
    def dx(self) -> float:
        """Spacing between adjacent nodes, including the boundary nodes."""
        return (self.x_max - self.x_min) / (self.nx + 1)

    @property
    def min_dx(self) -> float:
        """Return the uniform spacing."""
        return self.dx

    @property
    def min_dx2(self) -> float:
        """Return the squared uniform spacing."""
        return self.dx**2

    def x_coords(self) -> jax.Array:
        """Return the ``nx`` interior-node coordinates."""
        return jnp.linspace(self.x_min + self.dx, self.x_max - self.dx, self.nx)


def node_centered_dirichlet_laplacian(
    values: jax.Array,
    grid: NodeCenteredDirichletGrid,
) -> jax.Array:
    """Apply the three-point Laplacian with zero values at boundary nodes."""
    interior = jnp.asarray(values)
    if interior.shape != (grid.nx,):
        raise ValueError(f"Expected shape {(grid.nx,)}, got {interior.shape}")
    padded = jnp.concatenate(
        (jnp.zeros(1, dtype=interior.dtype), interior, jnp.zeros(1, dtype=interior.dtype))
    )
    return (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / grid.dx**2


__all__ = ["NodeCenteredDirichletGrid", "node_centered_dirichlet_laplacian"]
