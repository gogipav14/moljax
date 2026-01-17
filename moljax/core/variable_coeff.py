"""
Variable coefficient operators with FFT-based preconditioning.

This module provides FFT-accelerated solvers for PDEs with spatially-varying
coefficients:
    ∂u/∂t = ∇·(D(x)∇u) + f(u)

For nearly-constant coefficients D(x) ≈ D̄ + δD(x), we use:
1. Circulant approximation: replace D(x) with its mean D̄
2. FFT solves the constant-coefficient problem exactly
3. Use result as preconditioner for iterative refinement

Key insight: Even when D(x) varies significantly, the constant-coefficient
FFT solve provides an excellent preconditioner for GMRES/conjugate gradient.

Reference:
- Chan & Olkin, "Circulant Preconditioners for Toeplitz-Block Matrices" (1994)
- Strang, "A Proposal for Toeplitz Matrix Calculations" (1986)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.grid import Grid1D, Grid2D


# =============================================================================
# Variable Coefficient Data Structures
# =============================================================================

class VariableCoeffStats(NamedTuple):
    """Statistics for a variable coefficient field."""
    mean: float
    min: float
    max: float
    std: float
    variation_ratio: float  # std / mean


@dataclass(frozen=True)
class CirculantApprox:
    """
    Circulant approximation data for a variable coefficient operator.

    Stores the mean coefficient and FFT symbol for the constant-coefficient
    approximation, along with quality metrics.
    """
    D_mean: float
    fft_symbol: jnp.ndarray  # Laplacian symbol scaled by D_mean
    stats: VariableCoeffStats
    is_valid_approx: bool  # True if variation is small enough


def compute_coeff_stats(D: jnp.ndarray) -> VariableCoeffStats:
    """
    Compute statistics for a variable coefficient field.

    Args:
        D: Coefficient field (can be 1D or 2D)

    Returns:
        VariableCoeffStats with mean, min, max, std, and variation ratio
    """
    D_mean = float(jnp.mean(D))
    D_min = float(jnp.min(D))
    D_max = float(jnp.max(D))
    D_std = float(jnp.std(D))

    # Variation ratio: how much D varies relative to its mean
    variation_ratio = D_std / (jnp.abs(D_mean) + 1e-14)

    return VariableCoeffStats(
        mean=D_mean,
        min=D_min,
        max=D_max,
        std=D_std,
        variation_ratio=float(variation_ratio)
    )


# =============================================================================
# Circulant Approximation Factory
# =============================================================================

def create_circulant_approx_1d(
    D: jnp.ndarray,
    dx: float,
    threshold: float = 0.3,
    dtype=jnp.float64
) -> CirculantApprox:
    """
    Create circulant approximation for 1D variable-coefficient diffusion.

    For the operator L[u] = (D(x) u_x)_x, approximates with D̄ * u_xx
    where D̄ is the mean of D(x).

    Args:
        D: Variable diffusion coefficient, shape (nx,)
        dx: Grid spacing
        threshold: Maximum variation ratio for "valid" approximation
        dtype: Data type

    Returns:
        CirculantApprox with FFT symbol and quality metrics
    """
    stats = compute_coeff_stats(D)
    nx = len(D)

    # Build FFT symbol for constant-coefficient Laplacian
    k = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    laplacian_symbol = (2.0 * jnp.cos(k * dx) - 2.0) / (dx * dx)

    # Scale by mean coefficient
    fft_symbol = stats.mean * laplacian_symbol

    is_valid = stats.variation_ratio < threshold

    return CirculantApprox(
        D_mean=stats.mean,
        fft_symbol=fft_symbol.astype(dtype),
        stats=stats,
        is_valid_approx=is_valid
    )


def create_circulant_approx_2d(
    D: jnp.ndarray,
    dy: float,
    dx: float,
    threshold: float = 0.3,
    dtype=jnp.float64
) -> CirculantApprox:
    """
    Create circulant approximation for 2D variable-coefficient diffusion.

    Args:
        D: Variable diffusion coefficient, shape (ny, nx)
        dy, dx: Grid spacings
        threshold: Maximum variation ratio for "valid" approximation
        dtype: Data type

    Returns:
        CirculantApprox with 2D FFT symbol
    """
    stats = compute_coeff_stats(D)
    ny, nx = D.shape

    # Build 2D FFT symbol
    kx_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=dx)
    ky_1d = 2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=dy)

    kx = jnp.broadcast_to(kx_1d, (ny, nx))
    ky = jnp.broadcast_to(ky_1d[:, None], (ny, nx))

    lam_x = (2.0 * jnp.cos(kx * dx) - 2.0) / (dx * dx)
    lam_y = (2.0 * jnp.cos(ky * dy) - 2.0) / (dy * dy)

    fft_symbol = stats.mean * (lam_x + lam_y)

    is_valid = stats.variation_ratio < threshold

    return CirculantApprox(
        D_mean=stats.mean,
        fft_symbol=fft_symbol.astype(dtype),
        stats=stats,
        is_valid_approx=is_valid
    )


# =============================================================================
# Variable Coefficient Operators (Finite Difference)
# =============================================================================

@partial(jax.jit, static_argnums=(3, 4))
def apply_variable_diffusion_1d(
    u: jnp.ndarray,
    D: jnp.ndarray,
    dx: float,
    ng: int,
    nx: int
) -> jnp.ndarray:
    """
    Apply variable-coefficient diffusion: (D(x) u_x)_x.

    Uses conservative discretization:
        L[u]_i = (D_{i+1/2}(u_{i+1} - u_i) - D_{i-1/2}(u_i - u_{i-1})) / dx²

    where D_{i+1/2} = (D_i + D_{i+1}) / 2.

    Args:
        u: Solution field, shape (nx + 2*ng,)
        D: Diffusion coefficient, shape (nx + 2*ng,) or (nx,)
        dx: Grid spacing
        ng: Number of ghost cells
        nx: Number of interior points

    Returns:
        Diffusion term at interior points, shape (nx + 2*ng,)
    """
    dx2 = dx * dx
    result = jnp.zeros_like(u)

    # Handle D shape
    if D.shape[0] == nx:
        # D is interior-only, pad with edge values
        D_full = jnp.pad(D, ng, mode='edge')
    else:
        D_full = D

    # Interface diffusion coefficients
    D_plus = 0.5 * (D_full[ng:ng + nx] + D_full[ng + 1:ng + nx + 1])
    D_minus = 0.5 * (D_full[ng:ng + nx] + D_full[ng - 1:ng + nx - 1])

    # Fluxes
    flux_plus = D_plus * (u[ng + 1:ng + nx + 1] - u[ng:ng + nx])
    flux_minus = D_minus * (u[ng:ng + nx] - u[ng - 1:ng + nx - 1])

    # Conservative form
    Lu = (flux_plus - flux_minus) / dx2

    result = result.at[ng:ng + nx].set(Lu)
    return result


@partial(jax.jit, static_argnums=(4, 5, 6))
def apply_variable_diffusion_2d(
    u: jnp.ndarray,
    D: jnp.ndarray,
    dy: float,
    dx: float,
    ng: int,
    ny: int,
    nx: int
) -> jnp.ndarray:
    """
    Apply 2D variable-coefficient diffusion: ∇·(D(x,y)∇u).

    Conservative discretization in both directions.

    Args:
        u: Solution field, shape (ny + 2*ng, nx + 2*ng)
        D: Diffusion coefficient, same shape or interior-only
        dy, dx: Grid spacings
        ng: Number of ghost cells
        ny, nx: Number of interior points

    Returns:
        Diffusion term, shape (ny + 2*ng, nx + 2*ng)
    """
    dx2 = dx * dx
    dy2 = dy * dy
    result = jnp.zeros_like(u)

    # Handle D shape
    if D.shape == (ny, nx):
        D_full = jnp.pad(D, ng, mode='edge')
    else:
        D_full = D

    # Interior slices
    interior = (slice(ng, ng + ny), slice(ng, ng + nx))

    # X-direction diffusion
    D_x_plus = 0.5 * (D_full[ng:ng + ny, ng:ng + nx] + D_full[ng:ng + ny, ng + 1:ng + nx + 1])
    D_x_minus = 0.5 * (D_full[ng:ng + ny, ng:ng + nx] + D_full[ng:ng + ny, ng - 1:ng + nx - 1])

    flux_x_plus = D_x_plus * (u[ng:ng + ny, ng + 1:ng + nx + 1] - u[ng:ng + ny, ng:ng + nx])
    flux_x_minus = D_x_minus * (u[ng:ng + ny, ng:ng + nx] - u[ng:ng + ny, ng - 1:ng + nx - 1])
    Lx = (flux_x_plus - flux_x_minus) / dx2

    # Y-direction diffusion
    D_y_plus = 0.5 * (D_full[ng:ng + ny, ng:ng + nx] + D_full[ng + 1:ng + ny + 1, ng:ng + nx])
    D_y_minus = 0.5 * (D_full[ng:ng + ny, ng:ng + nx] + D_full[ng - 1:ng + ny - 1, ng:ng + nx])

    flux_y_plus = D_y_plus * (u[ng + 1:ng + ny + 1, ng:ng + nx] - u[ng:ng + ny, ng:ng + nx])
    flux_y_minus = D_y_minus * (u[ng:ng + ny, ng:ng + nx] - u[ng - 1:ng + ny - 1, ng:ng + nx])
    Ly = (flux_y_plus - flux_y_minus) / dy2

    result = result.at[interior].set(Lx + Ly)
    return result


# =============================================================================
# FFT-Based Solvers with Circulant Approximation
# =============================================================================

@jax.jit
def solve_helmholtz_circulant_1d(
    rhs: jnp.ndarray,
    fft_symbol: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Solve (I - dt*D̄*Δ)u = rhs using circulant (constant-coefficient) approximation.

    This is an approximate solve when D(x) varies, but exact when D is constant.

    Args:
        rhs: Right-hand side, shape (nx,)
        fft_symbol: Precomputed D̄ * laplacian_symbol
        dt: Time step

    Returns:
        Approximate solution u
    """
    rhs_hat = jnp.fft.fft(rhs)
    denom = 1.0 - dt * fft_symbol
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft(u_hat))


@jax.jit
def solve_helmholtz_circulant_2d(
    rhs: jnp.ndarray,
    fft_symbol: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    Solve 2D Helmholtz with circulant approximation.

    Args:
        rhs: Right-hand side, shape (ny, nx)
        fft_symbol: Precomputed 2D FFT symbol
        dt: Time step

    Returns:
        Approximate solution
    """
    rhs_hat = jnp.fft.fft2(rhs)
    denom = 1.0 - dt * fft_symbol
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft2(u_hat))


# =============================================================================
# Iterative Refinement with FFT Preconditioner
# =============================================================================

@partial(jax.jit, static_argnums=(4, 5, 6))
def richardson_iteration_varcoeff_1d(
    rhs: jnp.ndarray,
    D: jnp.ndarray,
    fft_symbol: jnp.ndarray,
    dx: float,
    ng: int,
    nx: int,
    n_iters: int,
    dt: float = 1.0,
    omega: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Richardson iteration with FFT preconditioner for variable-coefficient diffusion.

    Solves (I - dt * L_D)u = rhs where L_D is the variable-coefficient operator.

    Uses M⁻¹ = (I - dt * D̄ * Δ)⁻¹ as preconditioner.

    Args:
        rhs: Right-hand side, interior points only
        D: Variable diffusion coefficient (interior + ghost or interior)
        fft_symbol: Precomputed FFT symbol (approx.fft_symbol)
        dx: Grid spacing
        ng: Ghost cells (for padding during FD application)
        nx: Interior points
        n_iters: Number of iterations
        dt: Time step
        omega: Relaxation parameter (0 < omega <= 1)

    Returns:
        (solution, residual_history)
    """
    def residual(u_interior):
        # Pad for FD stencil
        u_padded = jnp.pad(u_interior, ng, mode='edge')
        Lu = apply_variable_diffusion_1d(u_padded, D, dx, ng, nx)
        Lu_interior = Lu[ng:ng + nx]
        return rhs - (u_interior - dt * Lu_interior)

    def iteration_step(carry, _):
        u, _ = carry
        r = residual(u)
        z = solve_helmholtz_circulant_1d(r, fft_symbol, dt)
        u_new = u + omega * z
        return (u_new, jnp.linalg.norm(r)), jnp.linalg.norm(r)

    # Initial guess: FFT solve
    u0 = solve_helmholtz_circulant_1d(rhs, fft_symbol, dt)

    (u_final, _), residuals = lax.scan(iteration_step, (u0, 0.0), None, length=n_iters)

    return u_final, residuals


@partial(jax.jit, static_argnums=(5, 6, 7, 8))
def richardson_iteration_varcoeff_2d(
    rhs: jnp.ndarray,
    D: jnp.ndarray,
    fft_symbol: jnp.ndarray,
    dy: float,
    dx: float,
    ng: int,
    ny: int,
    nx: int,
    n_iters: int,
    dt: float = 1.0,
    omega: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    2D Richardson iteration with FFT preconditioner.
    """
    def residual(u_interior):
        u_padded = jnp.pad(u_interior, ng, mode='edge')
        Lu = apply_variable_diffusion_2d(u_padded, D, dy, dx, ng, ny, nx)
        Lu_interior = Lu[ng:ng + ny, ng:ng + nx]
        return rhs - (u_interior - dt * Lu_interior)

    def iteration_step(carry, _):
        u, _ = carry
        r = residual(u)
        z = solve_helmholtz_circulant_2d(r, fft_symbol, dt)
        u_new = u + omega * z
        return (u_new, jnp.linalg.norm(r)), jnp.linalg.norm(r)

    u0 = solve_helmholtz_circulant_2d(rhs, fft_symbol, dt)
    (u_final, _), residuals = lax.scan(iteration_step, (u0, 0.0), None, length=n_iters)

    return u_final, residuals


# =============================================================================
# Quality Assessment
# =============================================================================

def assess_circulant_quality(
    approx: CirculantApprox,
    test_u: Optional[jnp.ndarray] = None,
    D: Optional[jnp.ndarray] = None,
    dx: float = 0.1
) -> dict:
    """
    Assess quality of circulant approximation.

    Args:
        approx: CirculantApprox to assess
        test_u: Optional test vector for error estimation
        D: Original variable coefficient (for comparison)
        dx: Grid spacing

    Returns:
        Dictionary with quality metrics
    """
    quality = {
        'D_mean': approx.D_mean,
        'variation_ratio': approx.stats.variation_ratio,
        'D_range': (approx.stats.min, approx.stats.max),
        'is_valid_approx': approx.is_valid_approx,
    }

    # Approximate condition number of Helmholtz operator
    dt = 0.01  # Reference dt
    denom_min = float(jnp.min(jnp.abs(1.0 - dt * approx.fft_symbol)))
    denom_max = float(jnp.max(jnp.abs(1.0 - dt * approx.fft_symbol)))
    quality['approx_condition_number'] = denom_max / (denom_min + 1e-14)

    # Error estimate based on variation
    # Expected relative error ≈ variation_ratio for small variations
    quality['expected_relative_error'] = approx.stats.variation_ratio

    # Recommendation
    if approx.stats.variation_ratio < 0.1:
        quality['recommendation'] = 'excellent - direct FFT solve recommended'
    elif approx.stats.variation_ratio < 0.3:
        quality['recommendation'] = 'good - FFT preconditioner effective'
    elif approx.stats.variation_ratio < 0.5:
        quality['recommendation'] = 'moderate - consider iterative refinement'
    else:
        quality['recommendation'] = 'poor - use full iterative solver'

    return quality


# =============================================================================
# ETD with Variable Coefficients
# =============================================================================

@jax.jit
def etd1_varcoeff_approx_1d(
    u: jnp.ndarray,
    N_u: jnp.ndarray,
    fft_symbol: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    ETD1 step using circulant (mean-coefficient) approximation.

    This is an approximate ETD step for variable coefficients:
        u^{n+1} ≈ exp(dt*D̄*Δ)*u^n + φ₁(dt*D̄*Δ)*dt*N(u^n)

    Args:
        u: Current solution (interior points)
        N_u: Nonlinear term N(u)
        fft_symbol: D̄ * laplacian_symbol
        dt: Time step

    Returns:
        Updated solution
    """
    z = dt * fft_symbol
    exp_z = jnp.exp(z)

    # φ₁(z) with Taylor for small |z|
    phi1_z = jnp.where(
        jnp.abs(z) < 1e-4,
        1.0 + z / 2.0 + z**2 / 6.0,
        (jnp.exp(z) - 1.0) / z
    )

    u_hat = jnp.fft.fft(u)
    N_hat = jnp.fft.fft(N_u)

    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat

    return jnp.real(jnp.fft.ifft(u_new_hat))


def print_varcoeff_report(approx: CirculantApprox) -> None:
    """Print a summary report for variable coefficient approximation."""
    print("=" * 60)
    print("Variable Coefficient Approximation Report")
    print("=" * 60)
    print(f"Mean coefficient D̄:      {approx.D_mean:.6f}")
    print(f"Range:                   [{approx.stats.min:.6f}, {approx.stats.max:.6f}]")
    print(f"Standard deviation:      {approx.stats.std:.6f}")
    print(f"Variation ratio (σ/μ):   {approx.stats.variation_ratio:.4f}")
    print(f"Valid approximation:     {approx.is_valid_approx}")
    print("-" * 60)

    quality = assess_circulant_quality(approx)
    print(f"Recommendation: {quality['recommendation']}")
    print(f"Expected relative error: {quality['expected_relative_error']:.2%}")
    print("=" * 60)
