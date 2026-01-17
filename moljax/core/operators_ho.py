"""
Higher-order finite difference operators for MOL-JAX.

This module provides 4th-order accurate finite difference operators:
- D1: First derivative (4th-order central)
- D2: Second derivative (4th-order central)
- Laplacian: 4th-order accurate
- FFT symbol functions matching 4th-order stencils

4th-order stencils:
- D1: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
- D2: (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*dx^2)

These require 2 ghost cells on each side (n_ghost >= 2).

Reference: Fornberg, "Generation of Finite Difference Formulas" (1988)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Any, Union, Optional

import jax
import jax.numpy as jnp

from moljax.core.grid import Grid1D, Grid2D, GridType


# =============================================================================
# 4th-Order 1D Operators
# =============================================================================

@partial(jax.jit, static_argnums=(2, 3))
def d1_fourth_order_1d(f: jnp.ndarray, dx: float, ng: int, nx: int) -> jnp.ndarray:
    """
    4th-order central difference first derivative in 1D.

    Stencil: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)

    Args:
        f: Padded 1D field array of shape (nx + 2*ng,)
        dx: Grid spacing
        ng: Number of ghost cells (must be >= 2)
        nx: Number of interior points

    Returns:
        Array of same shape with derivative in interior
    """
    df = jnp.zeros_like(f)

    # 4th-order central difference
    # df[i] = (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
    interior = slice(ng, ng + nx)
    df = df.at[interior].set(
        (-f[ng + 2:ng + nx + 2] + 8.0 * f[ng + 1:ng + nx + 1]
         - 8.0 * f[ng - 1:ng + nx - 1] + f[ng - 2:ng + nx - 2]) / (12.0 * dx)
    )

    return df


@partial(jax.jit, static_argnums=(2, 3))
def d2_fourth_order_1d(f: jnp.ndarray, dx: float, ng: int, nx: int) -> jnp.ndarray:
    """
    4th-order central difference second derivative in 1D.

    Stencil: (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*dx^2)

    Args:
        f: Padded 1D field array of shape (nx + 2*ng,)
        dx: Grid spacing
        ng: Number of ghost cells (must be >= 2)
        nx: Number of interior points

    Returns:
        Array of same shape with second derivative in interior
    """
    dx2 = dx * dx
    d2f = jnp.zeros_like(f)

    # 4th-order central difference for second derivative
    interior = slice(ng, ng + nx)
    d2f = d2f.at[interior].set(
        (-f[ng + 2:ng + nx + 2] + 16.0 * f[ng + 1:ng + nx + 1]
         - 30.0 * f[ng:ng + nx]
         + 16.0 * f[ng - 1:ng + nx - 1] - f[ng - 2:ng + nx - 2]) / (12.0 * dx2)
    )

    return d2f


def d1_ho_1d(f: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """
    4th-order first derivative with Grid1D interface.

    Args:
        f: Padded 1D field array
        grid: Grid1D (must have n_ghost >= 2)

    Returns:
        Array with 4th-order first derivative
    """
    if grid.n_ghost < 2:
        raise ValueError("4th-order operators require n_ghost >= 2")
    return d1_fourth_order_1d(f, grid.dx, grid.n_ghost, grid.nx)


def d2_ho_1d(f: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """
    4th-order second derivative with Grid1D interface.

    Args:
        f: Padded 1D field array
        grid: Grid1D (must have n_ghost >= 2)

    Returns:
        Array with 4th-order second derivative
    """
    if grid.n_ghost < 2:
        raise ValueError("4th-order operators require n_ghost >= 2")
    return d2_fourth_order_1d(f, grid.dx, grid.n_ghost, grid.nx)


def laplacian_ho_1d(f: jnp.ndarray, grid: Grid1D) -> jnp.ndarray:
    """
    4th-order Laplacian in 1D (same as d2_ho_1d).
    """
    return d2_ho_1d(f, grid)


# =============================================================================
# 4th-Order 2D Operators
# =============================================================================

@partial(jax.jit, static_argnums=(2, 3, 4))
def d1_fourth_order_2d_x(
    f: jnp.ndarray, dx: float, ng: int, ny: int, nx: int
) -> jnp.ndarray:
    """
    4th-order central difference first derivative in x-direction (2D).
    """
    df = jnp.zeros_like(f)

    df = df.at[ng:ng + ny, ng:ng + nx].set(
        (-f[ng:ng + ny, ng + 2:ng + nx + 2]
         + 8.0 * f[ng:ng + ny, ng + 1:ng + nx + 1]
         - 8.0 * f[ng:ng + ny, ng - 1:ng + nx - 1]
         + f[ng:ng + ny, ng - 2:ng + nx - 2]) / (12.0 * dx)
    )

    return df


@partial(jax.jit, static_argnums=(2, 3, 4))
def d1_fourth_order_2d_y(
    f: jnp.ndarray, dy: float, ng: int, ny: int, nx: int
) -> jnp.ndarray:
    """
    4th-order central difference first derivative in y-direction (2D).
    """
    df = jnp.zeros_like(f)

    df = df.at[ng:ng + ny, ng:ng + nx].set(
        (-f[ng + 2:ng + ny + 2, ng:ng + nx]
         + 8.0 * f[ng + 1:ng + ny + 1, ng:ng + nx]
         - 8.0 * f[ng - 1:ng + ny - 1, ng:ng + nx]
         + f[ng - 2:ng + ny - 2, ng:ng + nx]) / (12.0 * dy)
    )

    return df


@partial(jax.jit, static_argnums=(2, 3, 4))
def d2_fourth_order_2d_x(
    f: jnp.ndarray, dx: float, ng: int, ny: int, nx: int
) -> jnp.ndarray:
    """
    4th-order central difference second derivative in x-direction (2D).
    """
    dx2 = dx * dx
    d2f = jnp.zeros_like(f)

    d2f = d2f.at[ng:ng + ny, ng:ng + nx].set(
        (-f[ng:ng + ny, ng + 2:ng + nx + 2]
         + 16.0 * f[ng:ng + ny, ng + 1:ng + nx + 1]
         - 30.0 * f[ng:ng + ny, ng:ng + nx]
         + 16.0 * f[ng:ng + ny, ng - 1:ng + nx - 1]
         - f[ng:ng + ny, ng - 2:ng + nx - 2]) / (12.0 * dx2)
    )

    return d2f


@partial(jax.jit, static_argnums=(2, 3, 4))
def d2_fourth_order_2d_y(
    f: jnp.ndarray, dy: float, ng: int, ny: int, nx: int
) -> jnp.ndarray:
    """
    4th-order central difference second derivative in y-direction (2D).
    """
    dy2 = dy * dy
    d2f = jnp.zeros_like(f)

    d2f = d2f.at[ng:ng + ny, ng:ng + nx].set(
        (-f[ng + 2:ng + ny + 2, ng:ng + nx]
         + 16.0 * f[ng + 1:ng + ny + 1, ng:ng + nx]
         - 30.0 * f[ng:ng + ny, ng:ng + nx]
         + 16.0 * f[ng - 1:ng + ny - 1, ng:ng + nx]
         - f[ng - 2:ng + ny - 2, ng:ng + nx]) / (12.0 * dy2)
    )

    return d2f


def d1_ho_2d(f: jnp.ndarray, grid: Grid2D, axis: int) -> jnp.ndarray:
    """
    4th-order first derivative in 2D along specified axis.

    Args:
        f: Padded 2D field array
        grid: Grid2D (must have n_ghost >= 2)
        axis: 0 for y-derivative, 1 for x-derivative

    Returns:
        Array with 4th-order first derivative
    """
    if grid.n_ghost < 2:
        raise ValueError("4th-order operators require n_ghost >= 2")

    if axis == 0:
        return d1_fourth_order_2d_y(f, grid.dy, grid.n_ghost, grid.ny, grid.nx)
    else:
        return d1_fourth_order_2d_x(f, grid.dx, grid.n_ghost, grid.ny, grid.nx)


def d2_ho_2d(f: jnp.ndarray, grid: Grid2D, axis: int) -> jnp.ndarray:
    """
    4th-order second derivative in 2D along specified axis.

    Args:
        f: Padded 2D field array
        grid: Grid2D (must have n_ghost >= 2)
        axis: 0 for y-derivative, 1 for x-derivative

    Returns:
        Array with 4th-order second derivative
    """
    if grid.n_ghost < 2:
        raise ValueError("4th-order operators require n_ghost >= 2")

    if axis == 0:
        return d2_fourth_order_2d_y(f, grid.dy, grid.n_ghost, grid.ny, grid.nx)
    else:
        return d2_fourth_order_2d_x(f, grid.dx, grid.n_ghost, grid.ny, grid.nx)


def laplacian_ho_2d(f: jnp.ndarray, grid: Grid2D) -> jnp.ndarray:
    """
    4th-order Laplacian in 2D: d^2f/dx^2 + d^2f/dy^2.
    """
    return d2_ho_2d(f, grid, axis=0) + d2_ho_2d(f, grid, axis=1)


# =============================================================================
# FFT Symbols for 4th-Order Stencils
# =============================================================================

def fd_laplacian_symbol_ho_1d(
    nx: int,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build discrete Laplacian symbol for 4th-order 1D periodic domain.

    The 4th-order stencil:
        d²f/dx² ≈ (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*dx²)

    Has Fourier symbol:
        λ(k) = (-2*cos(2*k*dx) + 32*cos(k*dx) - 30) / (12*dx²)

    Args:
        nx: Number of interior points
        dx: Grid spacing
        dtype: Data type

    Returns:
        Laplacian symbol array of shape (nx,)
    """
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    lam = (-2.0 * jnp.cos(2.0 * k * dx) + 32.0 * jnp.cos(k * dx) - 30.0) / (12.0 * dx * dx)
    return lam.astype(dtype)


def fd_laplacian_symbol_ho_2d(
    ny: int,
    nx: int,
    dy: float,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build discrete Laplacian symbol for 4th-order 2D periodic domain.

    The symbol is the sum of 1D symbols in each direction:
        λ(kx, ky) = λ_x(kx) + λ_y(ky)

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

    # 4th-order symbol in each direction
    lam_x = (-2.0 * jnp.cos(2.0 * kx * dx) + 32.0 * jnp.cos(kx * dx) - 30.0) / (12.0 * dx * dx)
    lam_y = (-2.0 * jnp.cos(2.0 * ky * dy) + 32.0 * jnp.cos(ky * dy) - 30.0) / (12.0 * dy * dy)

    return (lam_x + lam_y).astype(dtype)


def fd_d1_symbol_ho_1d(
    nx: int,
    dx: float,
    dtype: jnp.dtype = jnp.complex128
) -> jnp.ndarray:
    """
    Build discrete first derivative symbol for 4th-order 1D periodic domain.

    The 4th-order stencil:
        df/dx ≈ (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)

    Has Fourier symbol:
        λ(k) = i * (8*sin(k*dx) - sin(2*k*dx)) / (6*dx)

    Args:
        nx: Number of interior points
        dx: Grid spacing
        dtype: Data type

    Returns:
        First derivative symbol array of shape (nx,)
    """
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    lam = 1j * (8.0 * jnp.sin(k * dx) - jnp.sin(2.0 * k * dx)) / (6.0 * dx)
    return lam.astype(dtype)


# =============================================================================
# 6th-Order Operators (for completeness)
# =============================================================================

@partial(jax.jit, static_argnums=(2, 3))
def d2_sixth_order_1d(f: jnp.ndarray, dx: float, ng: int, nx: int) -> jnp.ndarray:
    """
    6th-order central difference second derivative in 1D.

    Stencil: (2*f[i+3] - 27*f[i+2] + 270*f[i+1] - 490*f[i]
              + 270*f[i-1] - 27*f[i-2] + 2*f[i-3]) / (180*dx^2)

    Requires n_ghost >= 3.
    """
    dx2 = dx * dx
    d2f = jnp.zeros_like(f)

    interior = slice(ng, ng + nx)
    d2f = d2f.at[interior].set(
        (2.0 * f[ng + 3:ng + nx + 3]
         - 27.0 * f[ng + 2:ng + nx + 2]
         + 270.0 * f[ng + 1:ng + nx + 1]
         - 490.0 * f[ng:ng + nx]
         + 270.0 * f[ng - 1:ng + nx - 1]
         - 27.0 * f[ng - 2:ng + nx - 2]
         + 2.0 * f[ng - 3:ng + nx - 3]) / (180.0 * dx2)
    )

    return d2f


def fd_laplacian_symbol_6th_1d(
    nx: int,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build discrete Laplacian symbol for 6th-order 1D periodic domain.

    The 6th-order stencil has Fourier symbol:
        λ(k) = (4*cos(3*k*dx) - 54*cos(2*k*dx) + 540*cos(k*dx) - 490) / (180*dx²)
    """
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    lam = (4.0 * jnp.cos(3.0 * k * dx)
           - 54.0 * jnp.cos(2.0 * k * dx)
           + 540.0 * jnp.cos(k * dx)
           - 490.0) / (180.0 * dx * dx)
    return lam.astype(dtype)


# =============================================================================
# Operator Order Enum
# =============================================================================

class OperatorOrder:
    """Constants for finite difference order selection."""
    SECOND = 2
    FOURTH = 4
    SIXTH = 6


def get_laplacian_symbol_1d(
    nx: int,
    dx: float,
    order: int = 2,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Get Laplacian symbol for specified order.

    Args:
        nx: Number of points
        dx: Grid spacing
        order: Finite difference order (2, 4, or 6)
        dtype: Data type

    Returns:
        Laplacian symbol array
    """
    from moljax.core.operators import fd_laplacian_symbol_1d

    if order == 2:
        return fd_laplacian_symbol_1d(nx, dx, dtype)
    elif order == 4:
        return fd_laplacian_symbol_ho_1d(nx, dx, dtype)
    elif order == 6:
        return fd_laplacian_symbol_6th_1d(nx, dx, dtype)
    else:
        raise ValueError(f"Unsupported order: {order}. Use 2, 4, or 6.")


def get_laplacian_symbol_2d(
    ny: int,
    nx: int,
    dy: float,
    dx: float,
    order: int = 2,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Get 2D Laplacian symbol for specified order.

    Args:
        ny, nx: Number of points
        dy, dx: Grid spacings
        order: Finite difference order (2 or 4)
        dtype: Data type

    Returns:
        Laplacian symbol array
    """
    from moljax.core.operators import fd_laplacian_symbol_2d

    if order == 2:
        return fd_laplacian_symbol_2d(ny, nx, dy, dx, dtype)
    elif order == 4:
        return fd_laplacian_symbol_ho_2d(ny, nx, dy, dx, dtype)
    else:
        raise ValueError(f"Unsupported order: {order}. Use 2 or 4.")


# =============================================================================
# Convergence Test Utilities
# =============================================================================

def test_convergence_order_d1(
    f_exact: Callable[[jnp.ndarray], jnp.ndarray],
    df_exact: Callable[[jnp.ndarray], jnp.ndarray],
    x_range: tuple = (0.0, 2 * jnp.pi),
    n_values: tuple = (32, 64, 128, 256),
    order: int = 4
) -> Dict[str, Any]:
    """
    Test convergence order of first derivative operator.

    Args:
        f_exact: Exact function f(x)
        df_exact: Exact derivative f'(x)
        x_range: Domain (x_min, x_max)
        n_values: Grid sizes to test
        order: Expected order (4 for 4th-order)

    Returns:
        Dict with errors and measured order
    """
    x_min, x_max = x_range
    errors = []
    dxs = []

    for n in n_values:
        ng = 2 if order == 4 else (3 if order == 6 else 1)
        dx = (x_max - x_min) / n
        dxs.append(dx)

        # Create padded array
        x_interior = jnp.linspace(x_min + dx, x_max - dx, n)
        x_padded = jnp.linspace(x_min - ng * dx, x_max + ng * dx, n + 2 * ng)

        f_padded = f_exact(x_padded)
        df_numerical = d1_fourth_order_1d(f_padded, dx, ng, n) if order == 4 else None

        # Extract interior
        df_num_interior = df_numerical[ng:ng + n]
        df_exact_interior = df_exact(x_interior)

        error = jnp.max(jnp.abs(df_num_interior - df_exact_interior))
        errors.append(float(error))

    # Compute convergence rates
    rates = []
    for i in range(1, len(errors)):
        rate = jnp.log(errors[i-1] / errors[i]) / jnp.log(dxs[i-1] / dxs[i])
        rates.append(float(rate))

    return {
        'n_values': n_values,
        'errors': errors,
        'dxs': dxs,
        'rates': rates,
        'mean_rate': sum(rates) / len(rates) if rates else 0.0
    }
