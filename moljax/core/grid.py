"""
Grid definitions with ghost cell support for MOL-JAX.

This module provides Grid1D and Grid2D dataclasses that define computational
domains with ghost cells for boundary condition handling. All arrays in the
solver are stored with ghost cells: shape (nx + 2*n_ghost,) for 1D and
(ny + 2*n_ghost, nx + 2*n_ghost) for 2D.

Design decisions:
- Grids are immutable dataclasses (frozen=True)
- Ghost cell count is configurable (default n_ghost=1 for 2nd-order stencils)
- Properties provide convenient access to derived quantities (dx, dy, etc.)
- interior_slice property provides slicing for extracting interior points
"""

from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class Grid1D:
    """
    1D uniform Cartesian grid with ghost cells.

    Arrays on this grid have shape (nx_total,) = (nx + 2*n_ghost,).
    Interior points are at indices [n_ghost : nx + n_ghost].

    Attributes:
        nx: Number of interior grid points
        x_min: Left boundary of domain
        x_max: Right boundary of domain
        n_ghost: Number of ghost cells on each side (default 1)

    Example:
        >>> grid = Grid1D.uniform(100, 0.0, 1.0)
        >>> grid.dx
        0.01
        >>> grid.nx_total
        102
    """
    nx: int
    x_min: float
    x_max: float
    n_ghost: int = 1

    @classmethod
    def uniform(cls, nx: int, x_min: float, x_max: float, n_ghost: int = 1) -> "Grid1D":
        """
        Create a uniform 1D grid.

        Args:
            nx: Number of interior grid points
            x_min: Left boundary
            x_max: Right boundary
            n_ghost: Number of ghost cells per side

        Returns:
            Grid1D instance
        """
        return cls(nx=nx, x_min=x_min, x_max=x_max, n_ghost=n_ghost)

    @property
    def dx(self) -> float:
        """Grid spacing."""
        return (self.x_max - self.x_min) / self.nx

    @property
    def nx_total(self) -> int:
        """Total number of points including ghost cells."""
        return self.nx + 2 * self.n_ghost

    @property
    def interior_slice(self) -> slice:
        """Slice for extracting interior points from padded arrays."""
        return slice(self.n_ghost, self.n_ghost + self.nx)

    @property
    def min_dx(self) -> float:
        """Minimum grid spacing (same as dx for uniform grids)."""
        return self.dx

    @property
    def min_dx2(self) -> float:
        """Minimum grid spacing squared."""
        return self.dx ** 2

    def x_coords(self, include_ghost: bool = False) -> "jax.Array":
        """
        Get x coordinates of grid points.

        Args:
            include_ghost: If True, include ghost cell coordinates

        Returns:
            Array of x coordinates
        """
        import jax.numpy as jnp

        if include_ghost:
            # Ghost cells extend beyond domain
            x_start = self.x_min - self.n_ghost * self.dx
            return jnp.linspace(
                x_start + 0.5 * self.dx,
                x_start + (self.nx_total - 0.5) * self.dx,
                self.nx_total
            )
        else:
            return jnp.linspace(
                self.x_min + 0.5 * self.dx,
                self.x_max - 0.5 * self.dx,
                self.nx
            )


@dataclass(frozen=True)
class Grid2D:
    """
    2D uniform Cartesian grid with ghost cells.

    Arrays on this grid have shape (ny_total, nx_total) =
    (ny + 2*n_ghost, nx + 2*n_ghost).
    Interior points are at indices [n_ghost:ny+n_ghost, n_ghost:nx+n_ghost].

    Convention: First index is y (rows), second index is x (columns).
    This matches numpy/matplotlib conventions for image-like arrays.

    Attributes:
        nx: Number of interior grid points in x direction
        ny: Number of interior grid points in y direction
        x_min: Left boundary of domain
        x_max: Right boundary of domain
        y_min: Bottom boundary of domain
        y_max: Top boundary of domain
        n_ghost: Number of ghost cells on each side (default 1)

    Example:
        >>> grid = Grid2D.uniform(100, 100, 0.0, 1.0, 0.0, 1.0)
        >>> grid.dx, grid.dy
        (0.01, 0.01)
        >>> grid.interior_slice
        (slice(1, 101), slice(1, 101))
    """
    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    n_ghost: int = 1

    @classmethod
    def uniform(
        cls,
        nx: int,
        ny: int,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        n_ghost: int = 1
    ) -> "Grid2D":
        """
        Create a uniform 2D grid.

        Args:
            nx: Number of interior grid points in x
            ny: Number of interior grid points in y
            x_min, x_max: x domain bounds
            y_min, y_max: y domain bounds
            n_ghost: Number of ghost cells per side

        Returns:
            Grid2D instance
        """
        return cls(
            nx=nx, ny=ny,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            n_ghost=n_ghost
        )

    @property
    def dx(self) -> float:
        """Grid spacing in x direction."""
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        """Grid spacing in y direction."""
        return (self.y_max - self.y_min) / self.ny

    @property
    def nx_total(self) -> int:
        """Total number of points in x including ghost cells."""
        return self.nx + 2 * self.n_ghost

    @property
    def ny_total(self) -> int:
        """Total number of points in y including ghost cells."""
        return self.ny + 2 * self.n_ghost

    @property
    def interior_slice(self) -> Tuple[slice, slice]:
        """Tuple of slices for extracting interior from padded arrays."""
        return (
            slice(self.n_ghost, self.n_ghost + self.ny),
            slice(self.n_ghost, self.n_ghost + self.nx)
        )

    @property
    def min_dx(self) -> float:
        """Minimum grid spacing in any direction."""
        return min(self.dx, self.dy)

    @property
    def min_dx2(self) -> float:
        """Minimum grid spacing squared."""
        return self.min_dx ** 2

    def meshgrid(self, include_ghost: bool = False) -> Tuple["jax.Array", "jax.Array"]:
        """
        Get 2D coordinate arrays.

        Args:
            include_ghost: If True, include ghost cell coordinates

        Returns:
            Tuple (X, Y) of 2D coordinate arrays with shape matching grid
        """
        import jax.numpy as jnp

        if include_ghost:
            x_start = self.x_min - self.n_ghost * self.dx
            y_start = self.y_min - self.n_ghost * self.dy
            x = jnp.linspace(
                x_start + 0.5 * self.dx,
                x_start + (self.nx_total - 0.5) * self.dx,
                self.nx_total
            )
            y = jnp.linspace(
                y_start + 0.5 * self.dy,
                y_start + (self.ny_total - 0.5) * self.dy,
                self.ny_total
            )
        else:
            x = jnp.linspace(
                self.x_min + 0.5 * self.dx,
                self.x_max - 0.5 * self.dx,
                self.nx
            )
            y = jnp.linspace(
                self.y_min + 0.5 * self.dy,
                self.y_max - 0.5 * self.dy,
                self.ny
            )

        # Note: indexing='xy' gives X varying along columns (axis=1)
        # and Y varying along rows (axis=0)
        return jnp.meshgrid(x, y, indexing='xy')


# Type alias for grid union
GridType = Union[Grid1D, Grid2D]
