"""
FFT-based solvers for MOL-JAX.

This module provides FFT-based implicit solvers for periodic domains:
- Laplacian symbol builders for 1D and 2D
- Exact Helmholtz solver: (I - dt * D * Δ) u = rhs
- Multi-field diffusion inverse for StateDict

Design decisions:
- FFT operates only on interior points (no ghost cells in FFT)
- Symbols are precomputed once for JIT efficiency
- All operations are compatible with lax.while_loop
- Uses jax.numpy.fft (fft, ifft, fft2, ifft2)

The discrete Laplacian symbol for second-order central differences:
  1D: lam(k) = (2*cos(k*dx) - 2) / dx^2
  2D: lam(kx, ky) = (2*cos(kx*dx) - 2)/dx^2 + (2*cos(ky*dy) - 2)/dy^2

This matches the finite difference stencil: (u[i+1] - 2*u[i] + u[i-1]) / dx^2
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.grid import Grid1D, Grid2D, GridType
from moljax.core.state import StateDict


class FFTCache1D(NamedTuple):
    """Cached FFT data for 1D grids."""
    laplacian_symbol: jnp.ndarray  # Shape (nx,)
    k: jnp.ndarray  # Wavenumbers


class FFTCache2D(NamedTuple):
    """Cached FFT data for 2D grids."""
    laplacian_symbol: jnp.ndarray  # Shape (ny, nx)
    kx: jnp.ndarray  # x-wavenumbers
    ky: jnp.ndarray  # y-wavenumbers


# =============================================================================
# Symbol Builders
# =============================================================================

def build_wavenumbers_1d(nx: int, dx: float, dtype: jnp.dtype = jnp.float64) -> jnp.ndarray:
    """
    Build wavenumbers for 1D FFT.

    Args:
        nx: Number of interior points
        dx: Grid spacing
        dtype: Data type

    Returns:
        Wavenumber array k of shape (nx,)
    """
    # fftfreq returns frequencies in cycles/sample
    # k = 2*pi * fftfreq(n, d=dx) gives angular wavenumber
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    return k.astype(dtype)


def build_wavenumbers_2d(
    ny: int,
    nx: int,
    dy: float,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Build wavenumbers for 2D FFT.

    Args:
        ny: Number of interior points in y
        nx: Number of interior points in x
        dy: Grid spacing in y
        dx: Grid spacing in x
        dtype: Data type

    Returns:
        Tuple of (kx, ky) wavenumber arrays, each 2D with shape (ny, nx)
    """
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)

    # Create 2D arrays (broadcasting)
    kx = jnp.broadcast_to(kx_1d, (ny, nx)).astype(dtype)
    ky = jnp.broadcast_to(ky_1d[:, None], (ny, nx)).astype(dtype)

    return kx, ky


def laplacian_symbol_1d(
    nx: int,
    dx: float,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build discrete Laplacian symbol for 1D periodic domain.

    The symbol is: lam(k) = (2*cos(k*dx) - 2) / dx^2

    This corresponds to the finite difference operator:
    (u[i+1] - 2*u[i] + u[i-1]) / dx^2

    Args:
        nx: Number of interior points
        dx: Grid spacing
        dtype: Data type

    Returns:
        Laplacian symbol array of shape (nx,)
    """
    k = build_wavenumbers_1d(nx, dx, dtype)
    # lam = (2*cos(k*dx) - 2) / dx^2
    lam = (2.0 * jnp.cos(k * dx) - 2.0) / (dx * dx)
    return lam


def laplacian_symbol_2d(
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
    kx, ky = build_wavenumbers_2d(ny, nx, dy, dx, dtype)

    # lam_x = (2*cos(kx*dx) - 2) / dx^2
    # lam_y = (2*cos(ky*dy) - 2) / dy^2
    lam_x = (2.0 * jnp.cos(kx * dx) - 2.0) / (dx * dx)
    lam_y = (2.0 * jnp.cos(ky * dy) - 2.0) / (dy * dy)

    return lam_x + lam_y


# =============================================================================
# FFT Cache Creation
# =============================================================================

def create_fft_cache_1d(grid: Grid1D, dtype: jnp.dtype = jnp.float64) -> FFTCache1D:
    """
    Create FFT cache for 1D grid.

    Args:
        grid: Grid1D instance
        dtype: Data type

    Returns:
        FFTCache1D containing precomputed symbols
    """
    nx = grid.nx
    dx = grid.dx

    k = build_wavenumbers_1d(nx, dx, dtype)
    lam = laplacian_symbol_1d(nx, dx, dtype)

    return FFTCache1D(laplacian_symbol=lam, k=k)


def create_fft_cache_2d(grid: Grid2D, dtype: jnp.dtype = jnp.float64) -> FFTCache2D:
    """
    Create FFT cache for 2D grid.

    Args:
        grid: Grid2D instance
        dtype: Data type

    Returns:
        FFTCache2D containing precomputed symbols
    """
    nx, ny = grid.nx, grid.ny
    dx, dy = grid.dx, grid.dy

    kx, ky = build_wavenumbers_2d(ny, nx, dy, dx, dtype)
    lam = laplacian_symbol_2d(ny, nx, dy, dx, dtype)

    return FFTCache2D(laplacian_symbol=lam, kx=kx, ky=ky)


def create_fft_cache(grid: GridType, dtype: jnp.dtype = jnp.float64):
    """
    Create FFT cache for any grid type.

    Args:
        grid: Grid1D or Grid2D instance
        dtype: Data type

    Returns:
        FFTCache1D or FFTCache2D
    """
    if isinstance(grid, Grid1D):
        return create_fft_cache_1d(grid, dtype)
    else:
        return create_fft_cache_2d(grid, dtype)


# =============================================================================
# Helmholtz Solvers
# =============================================================================

def solve_helmholtz_1d(
    rhs_interior: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float
) -> jnp.ndarray:
    """
    Solve (I - dt * D * Δ) u = rhs using FFT for 1D periodic domain.

    In Fourier space: (1 - dt * D * lam) * U_hat = RHS_hat
    So: U_hat = RHS_hat / (1 - dt * D * lam)

    Args:
        rhs_interior: Right-hand side, interior only, shape (nx,)
        laplacian_symbol: Precomputed Laplacian symbol
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution u on interior, shape (nx,)
    """
    # Compute denominator: 1 - dt * D * lam
    denom = 1.0 - dt * D * laplacian_symbol

    # Avoid division by zero (shouldn't happen for positive D, dt)
    denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)

    # Forward FFT
    rhs_hat = jnp.fft.fft(rhs_interior)

    # Solve in Fourier space
    u_hat = rhs_hat / denom

    # Inverse FFT
    u = jnp.fft.ifft(u_hat).real

    return u


def solve_helmholtz_2d(
    rhs_interior: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float
) -> jnp.ndarray:
    """
    Solve (I - dt * D * Δ) u = rhs using FFT for 2D periodic domain.

    Args:
        rhs_interior: Right-hand side, interior only, shape (ny, nx)
        laplacian_symbol: Precomputed Laplacian symbol
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution u on interior, shape (ny, nx)
    """
    # Compute denominator: 1 - dt * D * lam
    denom = 1.0 - dt * D * laplacian_symbol

    # Avoid division by zero
    denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)

    # Forward FFT
    rhs_hat = jnp.fft.fft2(rhs_interior)

    # Solve in Fourier space
    u_hat = rhs_hat / denom

    # Inverse FFT
    u = jnp.fft.ifft2(u_hat).real

    return u


def solve_helmholtz(
    rhs_interior: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float
) -> jnp.ndarray:
    """
    Solve (I - dt * D * Δ) u = rhs using FFT.

    Automatically dispatches to 1D or 2D based on array shape.

    Args:
        rhs_interior: Right-hand side, interior only
        laplacian_symbol: Precomputed Laplacian symbol
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution u on interior
    """
    if rhs_interior.ndim == 1:
        return solve_helmholtz_1d(rhs_interior, laplacian_symbol, dt, D)
    else:
        return solve_helmholtz_2d(rhs_interior, laplacian_symbol, dt, D)


# =============================================================================
# Ghost Cell Integration
# =============================================================================

def extract_interior(f_padded: jnp.ndarray, grid: GridType) -> jnp.ndarray:
    """
    Extract interior values from padded array.

    Args:
        f_padded: Padded array with ghost cells
        grid: Grid defining the interior region

    Returns:
        Interior values only
    """
    if isinstance(grid, Grid1D):
        return f_padded[grid.interior_slice]
    else:
        sl_y, sl_x = grid.interior_slice
        return f_padded[sl_y, sl_x]


def embed_interior(
    u_interior: jnp.ndarray,
    grid: GridType,
    padded_template: jnp.ndarray
) -> jnp.ndarray:
    """
    Embed interior solution back into padded array.

    Args:
        u_interior: Solution on interior
        grid: Grid defining the interior region
        padded_template: Template padded array (for shape/ghost values)

    Returns:
        Padded array with updated interior
    """
    if isinstance(grid, Grid1D):
        return padded_template.at[grid.interior_slice].set(u_interior)
    else:
        sl_y, sl_x = grid.interior_slice
        return padded_template.at[sl_y, sl_x].set(u_interior)


def solve_diffusion_fft_field(
    rhs_padded: jnp.ndarray,
    grid: GridType,
    dt: float,
    D: float,
    fft_cache
) -> jnp.ndarray:
    """
    Solve (I - dt * D * Δ) u = rhs for a single field using FFT.

    Operates on interior only, then writes back to padded array.

    Args:
        rhs_padded: Right-hand side with ghost cells
        grid: Grid instance
        dt: Time step
        D: Diffusion coefficient
        fft_cache: Precomputed FFT cache

    Returns:
        Solution with ghost cells (ghost values from rhs_padded)
    """
    # Extract interior
    rhs_interior = extract_interior(rhs_padded, grid)

    # Solve using FFT
    u_interior = solve_helmholtz(rhs_interior, fft_cache.laplacian_symbol, dt, D)

    # Embed back into padded array
    u_padded = embed_interior(u_interior, grid, rhs_padded)

    return u_padded


# =============================================================================
# Multi-Field Operations
# =============================================================================

def apply_diffusion_inverse_fft(
    state_rhs: StateDict,
    grid: GridType,
    dt: float,
    diffusivities: Dict[str, float],
    fft_cache
) -> StateDict:
    """
    Apply (I - dt * D * Δ)^{-1} to each diffusive field in state.

    For each field with diffusivity D > 0, solves:
    (I - dt * D * Δ) u = rhs

    Non-diffusive fields (D = 0 or not in diffusivities) pass through unchanged.

    Args:
        state_rhs: Right-hand side StateDict (with ghost cells)
        grid: Grid instance
        dt: Time step
        diffusivities: Dict mapping field name to diffusion coefficient
        fft_cache: Precomputed FFT cache

    Returns:
        Solution StateDict with each field solved
    """
    result = {}

    for name, rhs_field in state_rhs.items():
        D = diffusivities.get(name, 0.0)

        if D > 1e-14:
            # Solve using FFT
            result[name] = solve_diffusion_fft_field(
                rhs_field, grid, dt, D, fft_cache
            )
        else:
            # Pass through unchanged
            result[name] = rhs_field

    return result


def apply_diffusion_fft(
    state: StateDict,
    grid: GridType,
    dt: float,
    diffusivities: Dict[str, float],
    fft_cache
) -> StateDict:
    """
    Apply implicit diffusion step: u_new = (I - dt * D * Δ)^{-1} u_old.

    This is for an implicit Euler diffusion step.

    Args:
        state: Current state (with ghost cells)
        grid: Grid instance
        dt: Time step
        diffusivities: Dict mapping field name to diffusion coefficient
        fft_cache: Precomputed FFT cache

    Returns:
        New state after implicit diffusion
    """
    return apply_diffusion_inverse_fft(state, grid, dt, diffusivities, fft_cache)


# =============================================================================
# Spectral Filters (Optional)
# =============================================================================

def exponential_filter_1d(
    nx: int,
    strength: float = 36.0,
    order: int = 36,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build exponential spectral filter for 1D.

    Filter: sigma(k) = exp(-strength * (|k|/k_max)^order)

    Args:
        nx: Number of points
        strength: Filter strength (higher = more damping)
        order: Filter order (higher = sharper cutoff)
        dtype: Data type

    Returns:
        Filter array of shape (nx,)
    """
    k_idx = jnp.arange(nx)
    # Fold indices for symmetric filter
    k_norm = jnp.minimum(k_idx, nx - k_idx) / (nx // 2)
    k_norm = jnp.minimum(k_norm, 1.0)  # Clamp to [0, 1]

    sigma = jnp.exp(-strength * jnp.power(k_norm, order))
    return sigma.astype(dtype)


def exponential_filter_2d(
    ny: int,
    nx: int,
    strength: float = 36.0,
    order: int = 36,
    dtype: jnp.dtype = jnp.float64
) -> jnp.ndarray:
    """
    Build exponential spectral filter for 2D.

    Args:
        ny: Number of points in y
        nx: Number of points in x
        strength: Filter strength
        order: Filter order
        dtype: Data type

    Returns:
        Filter array of shape (ny, nx)
    """
    sigma_x = exponential_filter_1d(nx, strength, order, dtype)
    sigma_y = exponential_filter_1d(ny, strength, order, dtype)

    # 2D filter is product of 1D filters
    sigma = jnp.outer(sigma_y, sigma_x)
    return sigma


def apply_spectral_filter_interior(
    u_interior: jnp.ndarray,
    filter_kernel: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply spectral filter to interior array.

    Args:
        u_interior: Interior values (no ghost cells)
        filter_kernel: Precomputed filter in Fourier space

    Returns:
        Filtered interior values
    """
    if u_interior.ndim == 1:
        u_hat = jnp.fft.fft(u_interior)
        u_hat_filtered = u_hat * filter_kernel
        return jnp.fft.ifft(u_hat_filtered).real
    else:
        u_hat = jnp.fft.fft2(u_interior)
        u_hat_filtered = u_hat * filter_kernel
        return jnp.fft.ifft2(u_hat_filtered).real


def apply_spectral_filter_field(
    f_padded: jnp.ndarray,
    grid: GridType,
    filter_kernel: jnp.ndarray
) -> jnp.ndarray:
    """
    Apply spectral filter to a padded field.

    Args:
        f_padded: Padded field with ghost cells
        grid: Grid instance
        filter_kernel: Precomputed filter kernel

    Returns:
        Filtered padded field
    """
    # Extract interior
    f_interior = extract_interior(f_padded, grid)

    # Apply filter
    f_filtered = apply_spectral_filter_interior(f_interior, filter_kernel)

    # Embed back
    return embed_interior(f_filtered, grid, f_padded)


def apply_spectral_filter_state(
    state: StateDict,
    grid: GridType,
    filter_kernel: jnp.ndarray,
    filter_fields: Optional[Dict[str, bool]] = None
) -> StateDict:
    """
    Apply spectral filter to selected fields in state.

    Args:
        state: State with ghost cells
        grid: Grid instance
        filter_kernel: Precomputed filter kernel
        filter_fields: Dict indicating which fields to filter (None = all)

    Returns:
        Filtered state
    """
    result = {}

    for name, field in state.items():
        should_filter = filter_fields is None or filter_fields.get(name, True)

        if should_filter:
            result[name] = apply_spectral_filter_field(field, grid, filter_kernel)
        else:
            result[name] = field

    return result


# =============================================================================
# IMEX Support Functions
# =============================================================================

def diffusion_rhs_fft(
    state: StateDict,
    grid: GridType,
    diffusivities: Dict[str, float],
    fft_cache
) -> StateDict:
    """
    Compute diffusion RHS: D * Δu using FFT.

    This is for computing the explicit diffusion term in IMEX schemes.
    Uses the spectral Laplacian for accuracy.

    Args:
        state: Current state
        grid: Grid instance
        diffusivities: Dict mapping field name to D
        fft_cache: Precomputed FFT cache

    Returns:
        Diffusion RHS for each field
    """
    result = {}

    for name, field in state.items():
        D = diffusivities.get(name, 0.0)

        if D > 1e-14:
            # Extract interior
            f_interior = extract_interior(field, grid)

            # Compute Laplacian in Fourier space
            f_hat = jnp.fft.fft2(f_interior) if f_interior.ndim == 2 else jnp.fft.fft(f_interior)
            lap_hat = fft_cache.laplacian_symbol * f_hat

            if f_interior.ndim == 2:
                lap_f = jnp.fft.ifft2(lap_hat).real
            else:
                lap_f = jnp.fft.ifft(lap_hat).real

            # Embed back and scale
            lap_padded = embed_interior(lap_f, grid, jnp.zeros_like(field))
            result[name] = D * lap_padded
        else:
            result[name] = jnp.zeros_like(field)

    return result
