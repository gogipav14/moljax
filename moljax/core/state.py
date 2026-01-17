"""
Multi-field PyTree state utilities for MOL-JAX.

This module provides utilities for working with multi-field states represented
as dictionaries of JAX arrays. All operations are designed to be JIT-compatible
and work with interior points only by default (ignoring ghost cells).

Design decisions:
- StateDict = dict[str, jnp.ndarray] is the canonical state type
- All norms/dots operate on interior points by default
- flatten_to_vec uses jax.flatten_util.ravel_pytree for Krylov solver compatibility
- PyTree structure (keys) must remain invariant across all code paths
"""

from typing import Callable, Tuple, Union, Dict
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from moljax.core.grid import Grid1D, Grid2D, GridType


# Type alias for multi-field state
StateDict = Dict[str, jnp.ndarray]


def tree_add(a: StateDict, b: StateDict) -> StateDict:
    """
    Element-wise addition of two states: a + b.

    Args:
        a: First state
        b: Second state (must have same keys as a)

    Returns:
        New state with element-wise sum
    """
    return jax.tree_util.tree_map(jnp.add, a, b)


def tree_sub(a: StateDict, b: StateDict) -> StateDict:
    """
    Element-wise subtraction of two states: a - b.

    Args:
        a: First state
        b: Second state (must have same keys as a)

    Returns:
        New state with element-wise difference
    """
    return jax.tree_util.tree_map(jnp.subtract, a, b)


def tree_scale(a: StateDict, c: float) -> StateDict:
    """
    Scalar multiplication of state: c * a.

    Args:
        a: State to scale
        c: Scalar multiplier

    Returns:
        New state with all fields scaled by c
    """
    return jax.tree_util.tree_map(lambda x: c * x, a)


def tree_axpy(y: StateDict, a: float, x: StateDict) -> StateDict:
    """
    Compute y + a*x (AXPY operation).

    Args:
        y: State
        a: Scalar multiplier
        x: State to scale and add

    Returns:
        New state: y + a*x
    """
    return jax.tree_util.tree_map(lambda yi, xi: yi + a * xi, y, x)


def tree_zeros_like(state: StateDict) -> StateDict:
    """
    Create a state of zeros with same structure and shape.

    Args:
        state: Template state

    Returns:
        New state with all zeros
    """
    return jax.tree_util.tree_map(jnp.zeros_like, state)


def tree_ones_like(state: StateDict) -> StateDict:
    """
    Create a state of ones with same structure and shape.

    Args:
        state: Template state

    Returns:
        New state with all ones
    """
    return jax.tree_util.tree_map(jnp.ones_like, state)


def _get_interior(field: jnp.ndarray, grid: GridType) -> jnp.ndarray:
    """Extract interior points from a padded field array."""
    return field[grid.interior_slice]


def tree_norm2(
    state: StateDict,
    grid: GridType,
    interior_only: bool = True
) -> jnp.ndarray:
    """
    Compute L2 norm of state (sqrt of sum of squares).

    Args:
        state: Multi-field state
        grid: Grid for interior extraction
        interior_only: If True (default), only count interior points

    Returns:
        Scalar L2 norm
    """
    def field_norm2_sq(field: jnp.ndarray) -> jnp.ndarray:
        if interior_only:
            field = _get_interior(field, grid)
        return jnp.sum(field ** 2)

    total_sq = sum(field_norm2_sq(f) for f in state.values())
    return jnp.sqrt(total_sq)


def tree_norm_inf(
    state: StateDict,
    grid: GridType,
    interior_only: bool = True
) -> jnp.ndarray:
    """
    Compute L-infinity norm of state (max absolute value).

    Args:
        state: Multi-field state
        grid: Grid for interior extraction
        interior_only: If True (default), only count interior points

    Returns:
        Scalar L-inf norm
    """
    def field_norm_inf(field: jnp.ndarray) -> jnp.ndarray:
        if interior_only:
            field = _get_interior(field, grid)
        return jnp.max(jnp.abs(field))

    return jnp.max(jnp.array([field_norm_inf(f) for f in state.values()]))


def tree_vdot(
    a: StateDict,
    b: StateDict,
    grid: GridType,
    interior_only: bool = True
) -> jnp.ndarray:
    """
    Compute inner product of two states: sum over all fields of sum(a*b).

    Args:
        a: First state
        b: Second state
        grid: Grid for interior extraction
        interior_only: If True (default), only count interior points

    Returns:
        Scalar inner product
    """
    def field_vdot(fa: jnp.ndarray, fb: jnp.ndarray) -> jnp.ndarray:
        if interior_only:
            fa = _get_interior(fa, grid)
            fb = _get_interior(fb, grid)
        return jnp.sum(fa * fb)

    total = sum(field_vdot(a[k], b[k]) for k in a.keys())
    return total


def flatten_to_vec(state: StateDict) -> Tuple[jnp.ndarray, Callable[[jnp.ndarray], StateDict]]:
    """
    Flatten state PyTree to 1D vector for Krylov solvers.

    Uses jax.flatten_util.ravel_pytree which maintains a consistent
    ordering and provides an unravel function.

    Args:
        state: Multi-field state to flatten

    Returns:
        Tuple of:
        - flat_vec: 1D array containing all field values
        - unravel_fn: Function to reconstruct state from flat vector

    Example:
        >>> state = {'u': jnp.ones((10,)), 'v': jnp.zeros((10,))}
        >>> vec, unravel = flatten_to_vec(state)
        >>> vec.shape
        (20,)
        >>> reconstructed = unravel(vec)
        >>> jnp.allclose(reconstructed['u'], state['u'])
        True
    """
    flat_vec, unravel_fn = ravel_pytree(state)
    return flat_vec, unravel_fn


def unflatten_from_vec(
    flat_vec: jnp.ndarray,
    template_state: StateDict
) -> StateDict:
    """
    Unflatten a 1D vector back to state using a template.

    This is a convenience wrapper when you have a template state
    but not the unravel function.

    Args:
        flat_vec: 1D array to unflatten
        template_state: State with same structure and shapes

    Returns:
        Reconstructed state
    """
    _, unravel_fn = ravel_pytree(template_state)
    return unravel_fn(flat_vec)


def state_total_size(state: StateDict) -> int:
    """
    Get total number of elements across all fields.

    Args:
        state: Multi-field state

    Returns:
        Total element count
    """
    return sum(f.size for f in state.values())


def state_interior_size(state: StateDict, grid: GridType) -> int:
    """
    Get total number of interior elements across all fields.

    Args:
        state: Multi-field state
        grid: Grid for interior extraction

    Returns:
        Total interior element count
    """
    return sum(_get_interior(f, grid).size for f in state.values())


def create_state_from_interior(
    interior_arrays: Dict[str, jnp.ndarray],
    grid: GridType,
    dtype: jnp.dtype = jnp.float64
) -> StateDict:
    """
    Create a padded state from interior-only arrays.

    Ghost cells are initialized to zero. Use apply_bc to fill them properly.

    Args:
        interior_arrays: Dict of interior arrays (without ghost cells)
        grid: Grid defining the domain
        dtype: Data type for arrays

    Returns:
        Padded state with ghost cells (zeros)
    """
    state = {}
    for name, interior in interior_arrays.items():
        if isinstance(grid, Grid1D):
            padded = jnp.zeros(grid.nx_total, dtype=dtype)
            padded = padded.at[grid.interior_slice].set(interior)
        else:
            padded = jnp.zeros((grid.ny_total, grid.nx_total), dtype=dtype)
            padded = padded.at[grid.interior_slice].set(interior)
        state[name] = padded
    return state


def get_interior_state(state: StateDict, grid: GridType) -> StateDict:
    """
    Extract interior points from all fields in state.

    Args:
        state: Padded state with ghost cells
        grid: Grid for interior extraction

    Returns:
        State with interior-only arrays (no ghost cells)
    """
    return {name: _get_interior(field, grid) for name, field in state.items()}


def scaled_error_norm(
    error: StateDict,
    y: StateDict,
    grid: GridType,
    atol: float,
    rtol: float
) -> jnp.ndarray:
    """
    Compute scaled error norm for adaptive time stepping.

    Computes: sqrt(mean((error / (atol + rtol * |y|))^2))
    over all interior points and all fields.

    Args:
        error: Error estimate state
        y: Current solution state
        grid: Grid for interior extraction
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        Scaled error norm (should be <= 1 for accepted step)
    """
    def field_scaled_error_sq(err_f: jnp.ndarray, y_f: jnp.ndarray) -> jnp.ndarray:
        err_int = _get_interior(err_f, grid)
        y_int = _get_interior(y_f, grid)
        scale = atol + rtol * jnp.abs(y_int)
        return jnp.sum((err_int / scale) ** 2)

    total_sq = sum(field_scaled_error_sq(error[k], y[k]) for k in error.keys())
    n_dof = state_interior_size(error, grid)
    return jnp.sqrt(total_sq / n_dof)
