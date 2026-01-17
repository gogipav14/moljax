"""
Spatial differential operators for MOL-JAX.

This module provides finite difference operators for the Method of Lines:
- D1: First derivative (central and upwind schemes)
- D2: Second derivative
- Laplacian: Sum of second derivatives in all directions
- Laplacian symbol functions for FFT-based solvers

All operators work on padded arrays with ghost cells and return arrays
of the same shape. The interior values are computed; ghost cells may
contain stale or uninitialized values.

Design decisions:
- Operators use simple indexing on padded arrays (no roll/shift needed)
- Upwind schemes use the sign of velocity for flux direction
- LinearOp and NonlinearOp dataclasses wrap operators with metadata
- Coupled multi-field operators return StateDict
- FFT symbol functions match finite difference Laplacian for periodic BCs
"""

from dataclasses import dataclass
from typing import Callable, Optional, Dict, Tuple, Any, Union, List
import jax
import jax.numpy as jnp

from moljax.core.grid import Grid1D, Grid2D, GridType
from moljax.core.state import StateDict, tree_add, tree_scale, tree_zeros_like


# =============================================================================
# 1D Operators
# =============================================================================

def d1_central_1d(f: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """
    Central difference first derivative in 1D: (f[i+1] - f[i-1]) / (2*dx).

    Second-order accurate. Returns array of same shape; interior values
    are computed derivatives, ghost cell values are undefined.

    Args:
        f: Padded 1D field array of shape (nx + 2*n_ghost,)
        grid: Grid1D defining spacing

    Returns:
        Array of same shape with derivative in interior
    """
    dx = grid.dx
    ng = grid.n_ghost

    # Create output array
    df = jnp.zeros_like(f)

    # Compute central difference in interior
    # Interior indices: ng to ng + nx
    # df[i] = (f[i+1] - f[i-1]) / (2*dx)
    interior = slice(ng, ng + grid.nx)
    df = df.at[interior].set((f[ng + 1:ng + grid.nx + 1] - f[ng - 1:ng + grid.nx - 1]) / (2.0 * dx))

    return df


def d1_upwind_1d(
    f: jnp.ndarray,
    v: Union[float, jnp.ndarray],
    grid: Grid1D
) -> jnp.ndarray:
    """
    Upwind first derivative in 1D based on velocity sign.

    First-order accurate. Uses:
    - Backward difference (f[i] - f[i-1])/dx when v > 0 (information from left)
    - Forward difference (f[i+1] - f[i])/dx when v < 0 (information from right)

    Args:
        f: Padded 1D field array of shape (nx + 2*n_ghost,)
        v: Velocity (scalar or array of same shape as f)
        grid: Grid1D defining spacing

    Returns:
        Array of same shape with upwind derivative in interior
    """
    dx = grid.dx
    ng = grid.n_ghost
    nx = grid.nx

    # Create output array
    df = jnp.zeros_like(f)

    # Compute both forward and backward differences
    interior = slice(ng, ng + nx)

    # Backward difference: (f[i] - f[i-1]) / dx
    df_backward = (f[ng:ng + nx] - f[ng - 1:ng + nx - 1]) / dx

    # Forward difference: (f[i+1] - f[i]) / dx
    df_forward = (f[ng + 1:ng + nx + 1] - f[ng:ng + nx]) / dx

    # Get velocity at interior points
    if jnp.ndim(v) == 0:
        v_interior = v
    else:
        v_interior = v[interior]

    # Select based on velocity sign
    df_interior = jnp.where(v_interior >= 0, df_backward, df_forward)
    df = df.at[interior].set(df_interior)

    return df


def d2_central_1d(f: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """
    Central difference second derivative in 1D: (f[i+1] - 2*f[i] + f[i-1]) / dx^2.

    Second-order accurate.

    Args:
        f: Padded 1D field array of shape (nx + 2*n_ghost,)
        grid: Grid1D defining spacing

    Returns:
        Array of same shape with second derivative in interior
    """
    dx2 = grid.dx ** 2
    ng = grid.n_ghost
    nx = grid.nx

    d2f = jnp.zeros_like(f)

    # d2f[i] = (f[i+1] - 2*f[i] + f[i-1]) / dx^2
    interior = slice(ng, ng + nx)
    d2f = d2f.at[interior].set(
        (f[ng + 1:ng + nx + 1] - 2.0 * f[ng:ng + nx] + f[ng - 1:ng + nx - 1]) / dx2
    )

    return d2f


def laplacian_1d(f: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """
    Laplacian in 1D (same as d2_central_1d).

    Args:
        f: Padded 1D field array
        grid: Grid1D defining spacing

    Returns:
        Array of same shape with Laplacian in interior
    """
    return d2_central_1d(f, grid)


# =============================================================================
# 2D Operators
# =============================================================================

def d1_central_2d(f: jnp.ndarray, grid: Grid2D, axis: int) -> jnp.ndarray:
    """
    Central difference first derivative in 2D along specified axis.

    axis=0: y-direction (rows)
    axis=1: x-direction (columns)

    Args:
        f: Padded 2D field array of shape (ny + 2*ng, nx + 2*ng)
        grid: Grid2D defining spacing
        axis: 0 for y-derivative, 1 for x-derivative

    Returns:
        Array of same shape with derivative in interior
    """
    ng = grid.n_ghost
    nx, ny = grid.nx, grid.ny

    if axis == 0:
        # y-derivative: d/dy
        dy = grid.dy
        df = jnp.zeros_like(f)

        # Interior region: [ng:ng+ny, ng:ng+nx]
        # df[j,i] = (f[j+1,i] - f[j-1,i]) / (2*dy)
        df = df.at[ng:ng + ny, ng:ng + nx].set(
            (f[ng + 1:ng + ny + 1, ng:ng + nx] - f[ng - 1:ng + ny - 1, ng:ng + nx]) / (2.0 * dy)
        )
    else:
        # x-derivative: d/dx
        dx = grid.dx
        df = jnp.zeros_like(f)

        # df[j,i] = (f[j,i+1] - f[j,i-1]) / (2*dx)
        df = df.at[ng:ng + ny, ng:ng + nx].set(
            (f[ng:ng + ny, ng + 1:ng + nx + 1] - f[ng:ng + ny, ng - 1:ng + nx - 1]) / (2.0 * dx)
        )

    return df


def d1_upwind_2d(
    f: jnp.ndarray,
    v: Union[float, jnp.ndarray],
    grid: Grid2D,
    axis: int
) -> jnp.ndarray:
    """
    Upwind first derivative in 2D along specified axis.

    axis=0: y-direction
    axis=1: x-direction

    Args:
        f: Padded 2D field array
        v: Velocity component in the direction of axis (scalar or array)
        grid: Grid2D defining spacing
        axis: 0 for y-derivative, 1 for x-derivative

    Returns:
        Array of same shape with upwind derivative in interior
    """
    ng = grid.n_ghost
    nx, ny = grid.nx, grid.ny

    df = jnp.zeros_like(f)

    if axis == 0:
        # y-derivative
        dy = grid.dy
        interior_y = slice(ng, ng + ny)
        interior_x = slice(ng, ng + nx)

        # Backward difference: (f[j,i] - f[j-1,i]) / dy
        df_backward = (f[ng:ng + ny, ng:ng + nx] - f[ng - 1:ng + ny - 1, ng:ng + nx]) / dy

        # Forward difference: (f[j+1,i] - f[j,i]) / dy
        df_forward = (f[ng + 1:ng + ny + 1, ng:ng + nx] - f[ng:ng + ny, ng:ng + nx]) / dy

        # Velocity at interior
        if jnp.ndim(v) == 0:
            v_interior = v
        else:
            v_interior = v[interior_y, interior_x]

        df_interior = jnp.where(v_interior >= 0, df_backward, df_forward)
        df = df.at[interior_y, interior_x].set(df_interior)

    else:
        # x-derivative
        dx = grid.dx
        interior_y = slice(ng, ng + ny)
        interior_x = slice(ng, ng + nx)

        # Backward difference: (f[j,i] - f[j,i-1]) / dx
        df_backward = (f[ng:ng + ny, ng:ng + nx] - f[ng:ng + ny, ng - 1:ng + nx - 1]) / dx

        # Forward difference: (f[j,i+1] - f[j,i]) / dx
        df_forward = (f[ng:ng + ny, ng + 1:ng + nx + 1] - f[ng:ng + ny, ng:ng + nx]) / dx

        # Velocity at interior
        if jnp.ndim(v) == 0:
            v_interior = v
        else:
            v_interior = v[interior_y, interior_x]

        df_interior = jnp.where(v_interior >= 0, df_backward, df_forward)
        df = df.at[interior_y, interior_x].set(df_interior)

    return df


def d2_central_2d(f: jnp.ndarray, grid: Grid2D, axis: int) -> jnp.ndarray:
    """
    Central difference second derivative in 2D along specified axis.

    axis=0: d^2/dy^2
    axis=1: d^2/dx^2

    Args:
        f: Padded 2D field array
        grid: Grid2D defining spacing
        axis: 0 for y-derivative, 1 for x-derivative

    Returns:
        Array of same shape with second derivative in interior
    """
    ng = grid.n_ghost
    nx, ny = grid.nx, grid.ny

    d2f = jnp.zeros_like(f)

    if axis == 0:
        # d^2/dy^2
        dy2 = grid.dy ** 2
        d2f = d2f.at[ng:ng + ny, ng:ng + nx].set(
            (f[ng + 1:ng + ny + 1, ng:ng + nx] -
             2.0 * f[ng:ng + ny, ng:ng + nx] +
             f[ng - 1:ng + ny - 1, ng:ng + nx]) / dy2
        )
    else:
        # d^2/dx^2
        dx2 = grid.dx ** 2
        d2f = d2f.at[ng:ng + ny, ng:ng + nx].set(
            (f[ng:ng + ny, ng + 1:ng + nx + 1] -
             2.0 * f[ng:ng + ny, ng:ng + nx] +
             f[ng:ng + ny, ng - 1:ng + nx - 1]) / dx2
        )

    return d2f


def laplacian_2d(f: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """
    Laplacian in 2D: d^2f/dx^2 + d^2f/dy^2.

    Args:
        f: Padded 2D field array
        grid: Grid2D defining spacing

    Returns:
        Array of same shape with Laplacian in interior
    """
    return d2_central_2d(f, grid, axis=0) + d2_central_2d(f, grid, axis=1)


# =============================================================================
# Operator Classes
# =============================================================================

@dataclass(frozen=True)
class LinearOp:
    """
    Linear operator wrapper with metadata.

    A linear operator L satisfies L(a*u + b*v) = a*L(u) + b*L(v).
    Examples: Laplacian, advection with constant velocity, coupled linear systems.

    Attributes:
        name: Identifier for the operator
        apply: Function (state, grid, t, params) -> state that applies the operator
        dt_bound: Optional function (grid, params) -> float giving CFL/stability bound
    """
    name: str
    apply: Callable[[StateDict, GridType, float, Dict[str, Any]], StateDict]
    dt_bound: Optional[Callable[[GridType, Dict[str, Any]], float]] = None


@dataclass(frozen=True)
class NonlinearOp:
    """
    Nonlinear operator wrapper with metadata.

    Examples: Reaction terms, nonlinear fluxes, variable-coefficient operators.

    Attributes:
        name: Identifier for the operator
        apply: Function (state, grid, t, params) -> state that applies the operator
        dt_bound: Optional function (grid, state, params) -> float giving stiffness bound
    """
    name: str
    apply: Callable[[StateDict, GridType, float, Dict[str, Any]], StateDict]
    dt_bound: Optional[Callable[[GridType, StateDict, Dict[str, Any]], float]] = None


def op_sum(ops: List[Union[LinearOp, NonlinearOp]]) -> Callable:
    """
    Create a combined operator that sums the results of multiple operators.

    Args:
        ops: List of LinearOp or NonlinearOp to combine

    Returns:
        Function (state, grid, t, params) -> StateDict that applies sum of all ops
    """
    def combined_apply(state: StateDict, grid: GridType, t: float, params: Dict[str, Any]) -> StateDict:
        result = ops[0].apply(state, grid, t, params)
        for op in ops[1:]:
            result = tree_add(result, op.apply(state, grid, t, params))
        return result

    return combined_apply


def state_zero_like(state: StateDict) -> StateDict:
    """Create a state of zeros with same structure."""
    return tree_zeros_like(state)


# =============================================================================
# Common Linear Operators
# =============================================================================

def make_diffusion_op_1d(field_name: str, D_param: str = 'D') -> LinearOp:
    """
    Create a 1D diffusion operator for a single field.

    Args:
        field_name: Name of the field to apply diffusion to
        D_param: Key in params dict for diffusion coefficient

    Returns:
        LinearOp that applies D * Laplacian to the field
    """
    def apply_diffusion(state: StateDict, grid: Grid1D, t: float, params: Dict) -> StateDict:
        D = params.get(D_param, 1.0)
        f = state[field_name]
        lap_f = laplacian_1d(f, grid)
        result = state_zero_like(state)
        result = {k: jnp.zeros_like(v) for k, v in state.items()}
        result[field_name] = D * lap_f
        return result

    def dt_bound(grid: Grid1D, params: Dict) -> float:
        D = params.get(D_param, 1.0)
        # Diffusion CFL: dt <= 0.5 * dx^2 / D for 1D
        return 0.5 * grid.min_dx2 / (D + 1e-14)

    return LinearOp(name=f"diffusion_{field_name}", apply=apply_diffusion, dt_bound=dt_bound)


def make_diffusion_op_2d(field_name: str, D_param: str = 'D') -> LinearOp:
    """
    Create a 2D diffusion operator for a single field.

    Args:
        field_name: Name of the field to apply diffusion to
        D_param: Key in params dict for diffusion coefficient

    Returns:
        LinearOp that applies D * Laplacian to the field
    """
    def apply_diffusion(state: StateDict, grid: Grid2D, t: float, params: Dict) -> StateDict:
        D = params.get(D_param, 1.0)
        f = state[field_name]
        lap_f = laplacian_2d(f, grid)
        result = {k: jnp.zeros_like(v) for k, v in state.items()}
        result[field_name] = D * lap_f
        return result

    def dt_bound(grid: Grid2D, params: Dict) -> float:
        D = params.get(D_param, 1.0)
        # Diffusion CFL: dt <= 0.25 * dx^2 / D for 2D (more restrictive)
        return 0.25 * grid.min_dx2 / (D + 1e-14)

    return LinearOp(name=f"diffusion_{field_name}", apply=apply_diffusion, dt_bound=dt_bound)


def make_advection_op_1d(
    field_name: str,
    vx_param: str = 'vx',
    use_upwind: bool = True
) -> LinearOp:
    """
    Create a 1D advection operator for a single field.

    Args:
        field_name: Name of the field to advect
        vx_param: Key in params dict for x-velocity
        use_upwind: If True, use upwind scheme; else central

    Returns:
        LinearOp that applies -vx * df/dx
    """
    def apply_advection(state: StateDict, grid: Grid1D, t: float, params: Dict) -> StateDict:
        vx = params.get(vx_param, 0.0)
        f = state[field_name]

        if use_upwind:
            df_dx = d1_upwind_1d(f, vx, grid)
        else:
            df_dx = d1_central_1d(f, grid)

        result = {k: jnp.zeros_like(v) for k, v in state.items()}
        result[field_name] = -vx * df_dx
        return result

    def dt_bound(grid: Grid1D, params: Dict) -> float:
        vx = params.get(vx_param, 0.0)
        v_max = jnp.abs(vx) if jnp.ndim(vx) == 0 else jnp.max(jnp.abs(vx))
        return grid.min_dx / (v_max + 1e-14)

    return LinearOp(name=f"advection_{field_name}", apply=apply_advection, dt_bound=dt_bound)


def make_advection_op_2d(
    field_name: str,
    vx_param: str = 'vx',
    vy_param: str = 'vy',
    use_upwind: bool = True
) -> LinearOp:
    """
    Create a 2D advection operator for a single field.

    Args:
        field_name: Name of the field to advect
        vx_param: Key in params dict for x-velocity
        vy_param: Key in params dict for y-velocity
        use_upwind: If True, use upwind scheme; else central

    Returns:
        LinearOp that applies -(vx * df/dx + vy * df/dy)
    """
    def apply_advection(state: StateDict, grid: Grid2D, t: float, params: Dict) -> StateDict:
        vx = params.get(vx_param, 0.0)
        vy = params.get(vy_param, 0.0)
        f = state[field_name]

        if use_upwind:
            df_dx = d1_upwind_2d(f, vx, grid, axis=1)
            df_dy = d1_upwind_2d(f, vy, grid, axis=0)
        else:
            df_dx = d1_central_2d(f, grid, axis=1)
            df_dy = d1_central_2d(f, grid, axis=0)

        result = {k: jnp.zeros_like(v) for k, v in state.items()}
        result[field_name] = -vx * df_dx - vy * df_dy
        return result

    def dt_bound(grid: Grid2D, params: Dict) -> float:
        vx = params.get(vx_param, 0.0)
        vy = params.get(vy_param, 0.0)
        vx_max = jnp.abs(vx) if jnp.ndim(vx) == 0 else jnp.max(jnp.abs(vx))
        vy_max = jnp.abs(vy) if jnp.ndim(vy) == 0 else jnp.max(jnp.abs(vy))
        v_max = vx_max + vy_max  # More conservative than sqrt(vx^2 + vy^2)
        return grid.min_dx / (v_max + 1e-14)

    return LinearOp(name=f"advection_{field_name}", apply=apply_advection, dt_bound=dt_bound)


# =============================================================================
# Coupled Multi-Field Operators
# =============================================================================

def acoustics_1d_linear_op() -> LinearOp:
    """
    Create coupled linear operator for 1D linear acoustics.

    System:
        dp/dt = -K * dv/dx
        dv/dt = -(1/rho) * dp/dx

    where K is bulk modulus, rho is density.
    Wave speed c = sqrt(K/rho).

    Expected params:
        'K': bulk modulus (default 1.0)
        'rho': density (default 1.0)

    State must have fields: 'p' (pressure) and 'v' (velocity)
    """
    def apply_acoustics(state: StateDict, grid: Grid1D, t: float, params: Dict) -> StateDict:
        K = params.get('K', 1.0)
        rho = params.get('rho', 1.0)

        p = state['p']
        v = state['v']

        # Use central differences for symmetric system
        dp_dx = d1_central_1d(p, grid)
        dv_dx = d1_central_1d(v, grid)

        # dp/dt = -K * dv/dx
        # dv/dt = -(1/rho) * dp/dx
        return {
            'p': -K * dv_dx,
            'v': -(1.0 / rho) * dp_dx
        }

    def dt_bound(grid: Grid1D, params: Dict) -> float:
        K = params.get('K', 1.0)
        rho = params.get('rho', 1.0)
        c = jnp.sqrt(K / rho)  # Wave speed
        # CFL condition: dt <= dx / c
        return grid.min_dx / (c + 1e-14)

    return LinearOp(name="acoustics_1d", apply=apply_acoustics, dt_bound=dt_bound)


def acoustics_2d_linear_op() -> LinearOp:
    """
    Create coupled linear operator for 2D linear acoustics.

    System:
        dp/dt = -K * (dvx/dx + dvy/dy)
        dvx/dt = -(1/rho) * dp/dx
        dvy/dt = -(1/rho) * dp/dy

    State must have fields: 'p', 'vx', 'vy'
    """
    def apply_acoustics(state: StateDict, grid: Grid2D, t: float, params: Dict) -> StateDict:
        K = params.get('K', 1.0)
        rho = params.get('rho', 1.0)

        p = state['p']
        vx = state['vx']
        vy = state['vy']

        # Central differences
        dp_dx = d1_central_2d(p, grid, axis=1)
        dp_dy = d1_central_2d(p, grid, axis=0)
        dvx_dx = d1_central_2d(vx, grid, axis=1)
        dvy_dy = d1_central_2d(vy, grid, axis=0)

        return {
            'p': -K * (dvx_dx + dvy_dy),
            'vx': -(1.0 / rho) * dp_dx,
            'vy': -(1.0 / rho) * dp_dy
        }

    def dt_bound(grid: Grid2D, params: Dict) -> float:
        K = params.get('K', 1.0)
        rho = params.get('rho', 1.0)
        c = jnp.sqrt(K / rho)
        # 2D CFL: dt <= dx / (c * sqrt(2)) for diagonal propagation
        return grid.min_dx / (c * jnp.sqrt(2.0) + 1e-14)

    return LinearOp(name="acoustics_2d", apply=apply_acoustics, dt_bound=dt_bound)


# =============================================================================
# Common Nonlinear Operators
# =============================================================================

def make_reaction_op(
    reactions: Dict[str, Callable[[StateDict, Dict], jnp.ndarray]]
) -> NonlinearOp:
    """
    Create a nonlinear reaction operator from a dict of reaction functions.

    Args:
        reactions: Dict mapping field name to function (state, params) -> reaction_rate

    Returns:
        NonlinearOp that applies the reaction terms
    """
    def apply_reaction(state: StateDict, grid: GridType, t: float, params: Dict) -> StateDict:
        result = {k: jnp.zeros_like(v) for k, v in state.items()}
        for field_name, reaction_fn in reactions.items():
            result[field_name] = reaction_fn(state, params)
        return result

    return NonlinearOp(name="reactions", apply=apply_reaction, dt_bound=None)


def gray_scott_reaction_op() -> NonlinearOp:
    """
    Gray-Scott reaction-diffusion reaction terms.

    du/dt = ... - u*v^2 + F*(1-u)
    dv/dt = ... + u*v^2 - (F+k)*v

    Expected params:
        'F': feed rate (default 0.04)
        'k': kill rate (default 0.06)
    """
    def apply_reaction(state: StateDict, grid: GridType, t: float, params: Dict) -> StateDict:
        F = params.get('F', 0.04)
        k = params.get('k', 0.06)

        u = state['u']
        v = state['v']

        uvv = u * v * v

        return {
            'u': -uvv + F * (1.0 - u),
            'v': uvv - (F + k) * v
        }

    def dt_bound(grid: GridType, state: StateDict, params: Dict) -> float:
        # Estimate reaction stiffness from Jacobian eigenvalues
        # For Gray-Scott: |df/du| ~ v^2 + F, |df/dv| ~ 2*u*v + (F+k)
        F = params.get('F', 0.04)
        k = params.get('k', 0.06)
        # Use a heuristic bound
        return 0.5 / (F + k + 1e-14)

    return NonlinearOp(name="gray_scott_reaction", apply=apply_reaction, dt_bound=dt_bound)


def fisher_kpp_reaction_op(field_name: str = 'u', r_param: str = 'r') -> NonlinearOp:
    """
    Fisher-KPP reaction term: r * u * (1 - u).

    Args:
        field_name: Field to apply reaction to
        r_param: Key in params for growth rate
    """
    def apply_reaction(state: StateDict, grid: GridType, t: float, params: Dict) -> StateDict:
        r = params.get(r_param, 1.0)
        u = state[field_name]

        result = {k: jnp.zeros_like(v) for k, v in state.items()}
        result[field_name] = r * u * (1.0 - u)
        return result

    def dt_bound(grid: GridType, state: StateDict, params: Dict) -> float:
        r = params.get(r_param, 1.0)
        # |df/du| = r * |1 - 2u| <= r
        return 0.5 / (r + 1e-14)

    return NonlinearOp(name=f"fisher_kpp_{field_name}", apply=apply_reaction, dt_bound=dt_bound)


# =============================================================================
# FFT-Based Laplacian Symbols
# =============================================================================

def fd_laplacian_symbol_1d(
    nx: int,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build discrete Laplacian symbol for 1D periodic domain.

    The symbol matches the finite difference stencil:
    lam(k) = (2*cos(k*dx) - 2) / dx^2

    This gives the eigenvalue of the FD Laplacian for the k-th Fourier mode.

    Args:
        nx: Number of interior points
        dx: Grid spacing
        dtype: Data type

    Returns:
        Laplacian symbol array of shape (nx,)
    """
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    lam = (2.0 * jnp.cos(k * dx) - 2.0) / (dx * dx)
    return lam.astype(dtype)


def fd_laplacian_symbol_2d(
    ny: int,
    nx: int,
    dy: float,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build discrete Laplacian symbol for 2D periodic domain.

    The symbol is:
    lam(kx, ky) = (2*cos(kx*dx) - 2)/dx^2 + (2*cos(ky*dy) - 2)/dy^2

    Args:
        ny: Number of interior points in y
        nx: Number of interior points in x
        dy: Grid spacing in y
        dx: Grid spacing in x
        dtype: Data type

    Returns:
        Laplacian symbol array of shape (ny, nx)
    """
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)

    kx = jnp.broadcast_to(kx_1d, (ny, nx))
    ky = jnp.broadcast_to(ky_1d[:, None], (ny, nx))

    lam_x = (2.0 * jnp.cos(kx * dx) - 2.0) / (dx * dx)
    lam_y = (2.0 * jnp.cos(ky * dy) - 2.0) / (dy * dy)

    return (lam_x + lam_y).astype(dtype)
