"""
Boundary condition specification and ghost cell filling for MOL-JAX.

This module implements ghost-cell based boundary conditions:
- PERIODIC: Ghost cells wrap around from opposite boundary
- DIRICHLET: Ghost cells set to enforce boundary value (2nd order extrapolation)
- NEUMANN: Ghost cells set to enforce zero gradient (mirror interior)

Design decisions:
- All BCs are applied by filling ghost cells, not by modifying stencils
- BC application is JIT-compatible using lax.switch on BC type
- Per-field BC specification allows different BCs for different fields
- 2nd-order accurate ghost cell formulas for Dirichlet BCs
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Optional, Callable, Any
import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.grid import Grid1D, Grid2D, GridType


class BCType(IntEnum):
    """Boundary condition types."""
    PERIODIC = 0
    DIRICHLET = 1
    NEUMANN = 2


@dataclass(frozen=True)
class FieldBCSpec:
    """
    Boundary condition specification for a single field.

    Attributes:
        kind: BCType enum value (PERIODIC, DIRICHLET, or NEUMANN)
        value: Optional callable (x, t, params) -> boundary value for Dirichlet
               If None, uses 0.0 for Dirichlet
        flux: Optional callable (x, t, params) -> normal flux for Neumann
              If None, uses zero gradient (homogeneous Neumann)

    For 1D:
        - value/flux return scalars or arrays of shape (2,) for [left, right]
    For 2D:
        - value/flux return dict with keys 'left', 'right', 'bottom', 'top'
          or a single scalar applied to all boundaries
    """
    kind: BCType
    value: Optional[Callable] = None
    flux: Optional[Callable] = None

    @classmethod
    def periodic(cls) -> "FieldBCSpec":
        """Create periodic BC spec."""
        return cls(kind=BCType.PERIODIC)

    @classmethod
    def dirichlet(cls, value: Optional[Callable] = None) -> "FieldBCSpec":
        """
        Create Dirichlet BC spec.

        Args:
            value: Callable (x, t, params) -> boundary value, or None for zero
        """
        return cls(kind=BCType.DIRICHLET, value=value)

    @classmethod
    def neumann(cls, flux: Optional[Callable] = None) -> "FieldBCSpec":
        """
        Create Neumann BC spec.

        Args:
            flux: Callable (x, t, params) -> normal derivative, or None for zero
        """
        return cls(kind=BCType.NEUMANN, flux=flux)


# Type alias for full BC specification
BCSpec = Dict[str, FieldBCSpec]


def _apply_bc_1d_periodic(field: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """Apply periodic BC in 1D by wrapping ghost cells."""
    ng = grid.n_ghost
    nx = grid.nx

    # Left ghost cells get values from right interior
    # Right ghost cells get values from left interior
    field = field.at[:ng].set(field[nx:nx + ng])
    field = field.at[nx + ng:].set(field[ng:2 * ng])
    return field


def _apply_bc_1d_dirichlet(
    field: jnp.ndarray,
    grid: Grid1D,
    bc_value_left: float,
    bc_value_right: float
) -> jnp.ndarray:
    """
    Apply Dirichlet BC in 1D.

    Uses linear extrapolation for 2nd order accuracy:
    ghost = 2 * bc_value - interior_adjacent

    This places the boundary value at the cell face between
    ghost and first interior cell.
    """
    ng = grid.n_ghost

    # Left boundary: ghost cell such that (ghost + interior)/2 = bc_value
    # => ghost = 2*bc_value - interior
    for i in range(ng):
        interior_idx = ng + i  # First interior cells
        ghost_idx = ng - 1 - i  # Ghost cells from boundary inward
        field = field.at[ghost_idx].set(2.0 * bc_value_left - field[interior_idx])

    # Right boundary
    for i in range(ng):
        interior_idx = grid.nx + ng - 1 - i  # Last interior cells
        ghost_idx = grid.nx + ng + i  # Ghost cells from boundary outward
        field = field.at[ghost_idx].set(2.0 * bc_value_right - field[interior_idx])

    return field


def _apply_bc_1d_neumann(
    field: jnp.ndarray,
    grid: Grid1D,
    flux_left: float = 0.0,
    flux_right: float = 0.0
) -> jnp.ndarray:
    """
    Apply Neumann BC in 1D.

    For zero gradient (flux=0): ghost = interior (mirror)
    For non-zero gradient: ghost = interior - 2*dx*flux (outward normal convention)
    """
    ng = grid.n_ghost
    dx = grid.dx

    # Left boundary: du/dx = flux_left at left face
    # ghost = interior - 2*dx*flux_left
    for i in range(ng):
        interior_idx = ng + i
        ghost_idx = ng - 1 - i
        field = field.at[ghost_idx].set(field[interior_idx] - 2.0 * dx * flux_left)

    # Right boundary: du/dx = flux_right at right face (outward normal is +x)
    for i in range(ng):
        interior_idx = grid.nx + ng - 1 - i
        ghost_idx = grid.nx + ng + i
        field = field.at[ghost_idx].set(field[interior_idx] + 2.0 * dx * flux_right)

    return field


def _apply_bc_field_1d(
    field: jnp.ndarray,
    grid: Grid1D,
    bc_spec: FieldBCSpec,
    t: float,
    params: Dict[str, Any]
) -> jnp.ndarray:
    """Apply BC to a single 1D field using lax.switch for JIT compatibility."""

    def apply_periodic(_):
        return _apply_bc_1d_periodic(field, grid)

    def apply_dirichlet(_):
        if bc_spec.value is not None:
            val = bc_spec.value(grid.x_coords(), t, params)
            if jnp.ndim(val) == 0:
                val_left = val_right = val
            else:
                val_left, val_right = val[0], val[-1]
        else:
            val_left = val_right = 0.0
        return _apply_bc_1d_dirichlet(field, grid, val_left, val_right)

    def apply_neumann(_):
        if bc_spec.flux is not None:
            flux = bc_spec.flux(grid.x_coords(), t, params)
            if jnp.ndim(flux) == 0:
                flux_left = flux_right = flux
            else:
                flux_left, flux_right = flux[0], flux[-1]
        else:
            flux_left = flux_right = 0.0
        return _apply_bc_1d_neumann(field, grid, flux_left, flux_right)

    return lax.switch(
        int(bc_spec.kind),
        [apply_periodic, apply_dirichlet, apply_neumann],
        None
    )


def _apply_bc_2d_periodic(field: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """Apply periodic BC in 2D by wrapping ghost cells in both directions."""
    ng = grid.n_ghost
    nx, ny = grid.nx, grid.ny

    # X direction (columns): left/right
    field = field.at[:, :ng].set(field[:, nx:nx + ng])
    field = field.at[:, nx + ng:].set(field[:, ng:2 * ng])

    # Y direction (rows): bottom/top
    field = field.at[:ng, :].set(field[ny:ny + ng, :])
    field = field.at[ny + ng:, :].set(field[ng:2 * ng, :])

    return field


def _apply_bc_2d_dirichlet(
    field: jnp.ndarray,
    grid: Grid2D,
    bc_values: Dict[str, float]
) -> jnp.ndarray:
    """
    Apply Dirichlet BC in 2D.

    bc_values should have keys: 'left', 'right', 'bottom', 'top'
    """
    ng = grid.n_ghost
    nx, ny = grid.nx, grid.ny

    val_left = bc_values.get('left', 0.0)
    val_right = bc_values.get('right', 0.0)
    val_bottom = bc_values.get('bottom', 0.0)
    val_top = bc_values.get('top', 0.0)

    # X direction (columns)
    for i in range(ng):
        # Left: ghost columns
        interior_col = ng + i
        ghost_col = ng - 1 - i
        field = field.at[:, ghost_col].set(2.0 * val_left - field[:, interior_col])

        # Right: ghost columns
        interior_col = nx + ng - 1 - i
        ghost_col = nx + ng + i
        field = field.at[:, ghost_col].set(2.0 * val_right - field[:, interior_col])

    # Y direction (rows)
    for i in range(ng):
        # Bottom: ghost rows
        interior_row = ng + i
        ghost_row = ng - 1 - i
        field = field.at[ghost_row, :].set(2.0 * val_bottom - field[interior_row, :])

        # Top: ghost rows
        interior_row = ny + ng - 1 - i
        ghost_row = ny + ng + i
        field = field.at[ghost_row, :].set(2.0 * val_top - field[interior_row, :])

    return field


def _apply_bc_2d_neumann(
    field: jnp.ndarray,
    grid: Grid2D,
    flux_values: Dict[str, float]
) -> jnp.ndarray:
    """
    Apply Neumann BC in 2D.

    flux_values should have keys: 'left', 'right', 'bottom', 'top'
    Default is zero gradient.
    """
    ng = grid.n_ghost
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    flux_left = flux_values.get('left', 0.0)
    flux_right = flux_values.get('right', 0.0)
    flux_bottom = flux_values.get('bottom', 0.0)
    flux_top = flux_values.get('top', 0.0)

    # X direction
    for i in range(ng):
        # Left
        interior_col = ng + i
        ghost_col = ng - 1 - i
        field = field.at[:, ghost_col].set(field[:, interior_col] - 2.0 * dx * flux_left)

        # Right
        interior_col = nx + ng - 1 - i
        ghost_col = nx + ng + i
        field = field.at[:, ghost_col].set(field[:, interior_col] + 2.0 * dx * flux_right)

    # Y direction
    for i in range(ng):
        # Bottom
        interior_row = ng + i
        ghost_row = ng - 1 - i
        field = field.at[ghost_row, :].set(field[interior_row, :] - 2.0 * dy * flux_bottom)

        # Top
        interior_row = ny + ng - 1 - i
        ghost_row = ny + ng + i
        field = field.at[ghost_row, :].set(field[interior_row, :] + 2.0 * dy * flux_top)

    return field


def _apply_bc_field_2d(
    field: jnp.ndarray,
    grid: Grid2D,
    bc_spec: FieldBCSpec,
    t: float,
    params: Dict[str, Any]
) -> jnp.ndarray:
    """Apply BC to a single 2D field using lax.switch for JIT compatibility."""

    def apply_periodic(_):
        return _apply_bc_2d_periodic(field, grid)

    def apply_dirichlet(_):
        if bc_spec.value is not None:
            X, Y = grid.meshgrid(include_ghost=False)
            val = bc_spec.value(X, Y, t, params)
            if isinstance(val, dict):
                bc_values = val
            elif jnp.ndim(val) == 0:
                bc_values = {'left': val, 'right': val, 'bottom': val, 'top': val}
            else:
                # Assume it's a constant
                bc_values = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}
        else:
            bc_values = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}
        return _apply_bc_2d_dirichlet(field, grid, bc_values)

    def apply_neumann(_):
        if bc_spec.flux is not None:
            X, Y = grid.meshgrid(include_ghost=False)
            flux = bc_spec.flux(X, Y, t, params)
            if isinstance(flux, dict):
                flux_values = flux
            elif jnp.ndim(flux) == 0:
                flux_values = {'left': flux, 'right': flux, 'bottom': flux, 'top': flux}
            else:
                flux_values = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}
        else:
            flux_values = {'left': 0.0, 'right': 0.0, 'bottom': 0.0, 'top': 0.0}
        return _apply_bc_2d_neumann(field, grid, flux_values)

    return lax.switch(
        int(bc_spec.kind),
        [apply_periodic, apply_dirichlet, apply_neumann],
        None
    )


def apply_bc(
    state: Dict[str, jnp.ndarray],
    grid: GridType,
    bc_spec: BCSpec,
    t: float = 0.0,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, jnp.ndarray]:
    """
    Apply boundary conditions to all fields in state.

    This function fills ghost cells according to the BC specification
    for each field. It is JIT-compatible.

    Args:
        state: Multi-field state dict with padded arrays
        grid: Grid defining the domain
        bc_spec: Dict mapping field names to FieldBCSpec
        t: Current time (for time-dependent BCs)
        params: Additional parameters passed to BC value/flux functions

    Returns:
        New state dict with ghost cells filled according to BCs

    Example:
        >>> grid = Grid2D.uniform(64, 64, 0, 1, 0, 1)
        >>> state = {'u': jnp.ones((66, 66)), 'v': jnp.zeros((66, 66))}
        >>> bc_spec = {'u': FieldBCSpec.periodic(), 'v': FieldBCSpec.dirichlet()}
        >>> state = apply_bc(state, grid, bc_spec, t=0.0)
    """
    if params is None:
        params = {}

    new_state = {}
    for name, field in state.items():
        if name not in bc_spec:
            # No BC specified, keep field unchanged
            new_state[name] = field
            continue

        field_bc = bc_spec[name]

        if isinstance(grid, Grid1D):
            new_state[name] = _apply_bc_field_1d(field, grid, field_bc, t, params)
        else:
            new_state[name] = _apply_bc_field_2d(field, grid, field_bc, t, params)

    return new_state


def create_default_bc_spec(
    field_names: list,
    bc_type: BCType = BCType.PERIODIC
) -> BCSpec:
    """
    Create a uniform BC spec for all fields.

    Args:
        field_names: List of field names
        bc_type: BC type to apply to all fields

    Returns:
        BCSpec dict
    """
    return {name: FieldBCSpec(kind=bc_type) for name in field_names}
