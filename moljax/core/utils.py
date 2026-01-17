"""
Utility functions for MOL-JAX.

This module provides helper functions for:
- Interior extraction from padded arrays
- Pre-allocated output buffer management
- Numerical constants tied to dtype
- Safe norm computation
"""

from typing import Dict, Tuple, Any, Union
import jax
import jax.numpy as jnp

from moljax.core.grid import Grid1D, Grid2D, GridType
from moljax.core.state import StateDict


# Numerical constants
def eps_for_dtype(dtype: jnp.dtype = jnp.float64) -> float:
    """Get machine epsilon for a given dtype."""
    return float(jnp.finfo(dtype).eps)


def safe_eps(dtype: jnp.dtype = jnp.float64) -> float:
    """Get a safe small epsilon for divisions (larger than machine eps)."""
    return 1e-10 if dtype == jnp.float32 else 1e-14


def get_interior(field: jnp.ndarray, grid: GridType) -> jnp.ndarray:
    """
    Extract interior points from a padded field array.

    Args:
        field: Padded array with ghost cells
        grid: Grid defining interior region

    Returns:
        Interior-only array (view, not copy)
    """
    return field[grid.interior_slice]


def set_interior(
    padded: jnp.ndarray,
    interior: jnp.ndarray,
    grid: GridType
) -> jnp.ndarray:
    """
    Set interior points in a padded array.

    Args:
        padded: Padded array to modify
        interior: Interior values to set
        grid: Grid defining interior region

    Returns:
        New padded array with interior set
    """
    return padded.at[grid.interior_slice].set(interior)


def allocate_scalar_history(max_steps: int, dtype: jnp.dtype = jnp.float64) -> jnp.ndarray:
    """
    Allocate buffer for scalar time history.

    Args:
        max_steps: Maximum number of steps to store
        dtype: Data type

    Returns:
        Zero-initialized array of shape (max_steps,)
    """
    return jnp.zeros(max_steps, dtype=dtype)


def allocate_field_history(
    field_shape: Tuple[int, ...],
    max_steps: int,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Allocate buffer for field history.

    Args:
        field_shape: Shape of field (interior only typically)
        max_steps: Maximum number of steps to store
        dtype: Data type

    Returns:
        Zero-initialized array of shape (max_steps,) + field_shape
    """
    return jnp.zeros((max_steps,) + field_shape, dtype=dtype)


def allocate_state_history(
    state: StateDict,
    grid: GridType,
    max_steps: int,
    interior_only: bool = True,
    dtype: jnp.dtype = jnp.float64
) -> Dict[str, jnp.ndarray]:
    """
    Allocate history buffers for all fields in state.

    Args:
        state: Template state for structure and shapes
        grid: Grid for interior extraction
        max_steps: Maximum number of steps to store
        interior_only: If True, only store interior points
        dtype: Data type

    Returns:
        Dict of history buffers, one per field
    """
    history = {}
    for name, field in state.items():
        if interior_only:
            field_shape = get_interior(field, grid).shape
        else:
            field_shape = field.shape
        history[name] = allocate_field_history(field_shape, max_steps, dtype)
    return history


def save_to_history(
    history: Dict[str, jnp.ndarray],
    state: StateDict,
    grid: GridType,
    idx: int,
    interior_only: bool = True
) -> Dict[str, jnp.ndarray]:
    """
    Save current state to history buffers at given index.

    Args:
        history: History buffers to update
        state: Current state to save
        grid: Grid for interior extraction
        idx: Index in history to write
        interior_only: If True, only save interior points

    Returns:
        Updated history buffers
    """
    new_history = {}
    for name, field in state.items():
        if interior_only:
            data = get_interior(field, grid)
        else:
            data = field
        new_history[name] = history[name].at[idx].set(data)
    return new_history


def safe_divide(numerator: jnp.ndarray, denominator: jnp.ndarray, eps: float = 1e-14) -> jnp.ndarray:
    """
    Safe division that avoids division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        eps: Small value to add to denominator

    Returns:
        numerator / (denominator + eps)
    """
    return numerator / (denominator + eps)


def clip_array(x: jnp.ndarray, x_min: float, x_max: float) -> jnp.ndarray:
    """
    Clip array values to [x_min, x_max].

    Args:
        x: Array to clip
        x_min: Minimum value
        x_max: Maximum value

    Returns:
        Clipped array
    """
    return jnp.clip(x, x_min, x_max)


def is_finite(state: StateDict) -> jnp.ndarray:
    """
    Check if all values in state are finite (no NaN or Inf).

    Args:
        state: State to check

    Returns:
        Boolean scalar: True if all finite
    """
    checks = [jnp.all(jnp.isfinite(f)) for f in state.values()]
    return jnp.all(jnp.array(checks))


def create_padded_array(
    grid: GridType,
    fill_value: float = 0.0,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Create a padded array filled with a constant value.

    Args:
        grid: Grid defining array shape
        fill_value: Value to fill array with
        dtype: Data type

    Returns:
        Padded array
    """
    if isinstance(grid, Grid1D):
        return jnp.full(grid.nx_total, fill_value, dtype=dtype)
    else:
        return jnp.full((grid.ny_total, grid.nx_total), fill_value, dtype=dtype)


def max_abs_interior(field: jnp.ndarray, grid: GridType) -> jnp.ndarray:
    """
    Get maximum absolute value in interior of field.

    Args:
        field: Padded field array
        grid: Grid for interior extraction

    Returns:
        Scalar max absolute value
    """
    return jnp.max(jnp.abs(get_interior(field, grid)))


def mean_interior(field: jnp.ndarray, grid: GridType) -> jnp.ndarray:
    """
    Get mean value in interior of field.

    Args:
        field: Padded field array
        grid: Grid for interior extraction

    Returns:
        Scalar mean value
    """
    return jnp.mean(get_interior(field, grid))


def sum_interior(field: jnp.ndarray, grid: GridType) -> jnp.ndarray:
    """
    Get sum of values in interior of field.

    Args:
        field: Padded field array
        grid: Grid for interior extraction

    Returns:
        Scalar sum
    """
    return jnp.sum(get_interior(field, grid))


# Status codes for adaptive integrator
class StatusCode:
    """Status codes for adaptive integration."""
    SUCCESS = 0
    RUNNING = 1
    MAX_STEPS_REACHED = 2
    DT_TOO_SMALL = 3
    NON_FINITE_VALUES = 4
    NK_FAILED = 5
