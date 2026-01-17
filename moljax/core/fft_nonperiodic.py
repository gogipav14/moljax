"""
FFT-based solvers for non-periodic boundary conditions.

This module extends FFT acceleration to Dirichlet and Neumann BCs using:
- DST (Discrete Sine Transform) for homogeneous Dirichlet BCs
- DCT (Discrete Cosine Transform) for homogeneous Neumann BCs

The key insight is that DST/DCT diagonalize the Laplacian on domains with
these boundary conditions, similar to how FFT diagonalizes on periodic domains.

Eigenvalue formulas:
- Dirichlet: λ_k = -4/dx² · sin²(πk/(2(N+1))), k = 1, ..., N
- Neumann:   λ_k = -4/dx² · sin²(πk/(2N)),     k = 0, ..., N-1

Reference: Trefethen, "Spectral Methods in MATLAB" (2000)
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.fft import dct, idct


class BCType(Enum):
    """Boundary condition types for non-periodic domains."""
    DIRICHLET = 'dirichlet'  # u = 0 at boundaries
    NEUMANN = 'neumann'      # du/dx = 0 at boundaries
    MIXED_DN = 'mixed_dn'    # Dirichlet at left, Neumann at right
    MIXED_ND = 'mixed_nd'    # Neumann at left, Dirichlet at right


class NonperiodicFFTCache(NamedTuple):
    """Cached data for non-periodic FFT operations."""
    N: int
    dx: float
    bc_type: str
    laplacian_symbol: jnp.ndarray
    k: jnp.ndarray  # Wavenumber indices


# =============================================================================
# DST/DCT Implementation
# =============================================================================

def dst_I(x: jnp.ndarray) -> jnp.ndarray:
    """
    Type-I Discrete Sine Transform (DST-I).

    DST-I is the appropriate transform for homogeneous Dirichlet BCs.
    It transforms a function that vanishes at both endpoints.

    DST-I[k] = Σ_{n=1}^{N} x[n] · sin(πkn/(N+1)), k = 1, ..., N

    Args:
        x: Input array of length N (interior points only)

    Returns:
        DST-I coefficients
    """
    N = len(x)
    # Embed in a larger array for scipy dct
    # DST-I can be computed via DCT-I of antisymmetric extension
    x_ext = jnp.concatenate([jnp.array([0.0]), x, jnp.array([0.0])])
    # Use relationship: DST-I(x) = -imag(FFT(antisym_ext))
    n = jnp.arange(N + 2)
    # Direct computation
    k = jnp.arange(1, N + 1)
    result = jnp.zeros(N, dtype=x.dtype)
    for ki in range(N):
        result = result.at[ki].set(
            jnp.sum(x * jnp.sin(jnp.pi * (ki + 1) * jnp.arange(1, N + 1) / (N + 1)))
        )
    return result * 2.0


def idst_I(X: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse Type-I Discrete Sine Transform.

    IDST-I is proportional to DST-I (self-inverse up to scaling).

    Args:
        X: DST-I coefficients

    Returns:
        Reconstructed interior values
    """
    N = len(X)
    # DST-I is self-inverse up to scaling: IDST-I = 2/(N+1) * DST-I
    return dst_I(X) / (N + 1)


@jax.jit
def dst_I_fast(x: jnp.ndarray) -> jnp.ndarray:
    """
    Fast DST-I using FFT.

    DST-I of x[0..N-1] computes:
        X[k] = Σ_{n=0}^{N-1} x[n] * sin(π(k+1)(n+1)/(N+1))

    Implemented via FFT of antisymmetric extension.
    """
    N = len(x)
    # Embed x in a 2(N+1) point array with odd extension
    # [0, x[0], x[1], ..., x[N-1], 0, -x[N-1], ..., -x[1], -x[0]]
    x_ext = jnp.concatenate([
        jnp.array([0.0]),
        x,
        jnp.array([0.0]),
        -x[::-1]
    ])

    # FFT gives us the DST coefficients in the imaginary part
    fft_result = jnp.fft.fft(x_ext)

    # Extract DST coefficients (imaginary part, scaled)
    # The DST-I coefficients are at indices 1..N
    return -0.5 * jnp.imag(fft_result[1:N+1])


@jax.jit
def idst_I_fast(X: jnp.ndarray) -> jnp.ndarray:
    """
    Fast inverse DST-I.

    IDST-I = (2/(N+1)) * DST-I for type-I transform.
    """
    N = len(X)
    return dst_I_fast(X) * 2.0 / (N + 1)


# =============================================================================
# Eigenvalue Formulas
# =============================================================================

def laplacian_symbol_dirichlet(N: int, dx: float, dtype=jnp.float64) -> jnp.ndarray:
    """
    Laplacian eigenvalues for Dirichlet BCs.

    For the second-order centered difference Laplacian with u(0) = u(L) = 0:
        λ_k = -4/dx² · sin²(πk/(2(N+1))), k = 1, ..., N

    These are the eigenvalues of the tridiagonal matrix:
        [-2, 1, 0, ..., 0,  1]
        [ 1,-2, 1, ..., 0,  0]
        ...
        [ 0, 0, 0, ..., 1, -2]

    Args:
        N: Number of interior points
        dx: Grid spacing
        dtype: Data type

    Returns:
        Array of N eigenvalues (all negative)
    """
    k = jnp.arange(1, N + 1, dtype=dtype)
    return -4.0 / (dx * dx) * jnp.sin(jnp.pi * k / (2 * (N + 1))) ** 2


def laplacian_symbol_neumann(N: int, dx: float, dtype=jnp.float64) -> jnp.ndarray:
    """
    Laplacian eigenvalues for Neumann BCs.

    For the second-order centered difference Laplacian with du/dx(0) = du/dx(L) = 0:
        λ_k = -4/dx² · sin²(πk/(2N)), k = 0, ..., N-1

    Note: λ_0 = 0 corresponds to the constant mode (null space).

    Args:
        N: Number of grid points
        dx: Grid spacing
        dtype: Data type

    Returns:
        Array of N eigenvalues (λ_0 = 0, rest negative)
    """
    k = jnp.arange(N, dtype=dtype)
    return -4.0 / (dx * dx) * jnp.sin(jnp.pi * k / (2 * N)) ** 2


def laplacian_symbol_mixed_dn(N: int, dx: float, dtype=jnp.float64) -> jnp.ndarray:
    """
    Laplacian eigenvalues for mixed Dirichlet-Neumann BCs.

    Dirichlet at x=0, Neumann at x=L:
        λ_k = -4/dx² · sin²(π(2k-1)/(4N)), k = 1, ..., N
    """
    k = jnp.arange(1, N + 1, dtype=dtype)
    return -4.0 / (dx * dx) * jnp.sin(jnp.pi * (2 * k - 1) / (4 * N)) ** 2


# =============================================================================
# FFT Cache Creation
# =============================================================================

def create_nonperiodic_fft_cache(
    N: int,
    dx: float,
    bc_type: BCType | str = BCType.DIRICHLET,
    dtype=jnp.float64
) -> NonperiodicFFTCache:
    """
    Create FFT cache for non-periodic boundary conditions.

    Args:
        N: Number of interior points (Dirichlet) or grid points (Neumann)
        dx: Grid spacing
        bc_type: Boundary condition type
        dtype: Data type

    Returns:
        NonperiodicFFTCache with precomputed eigenvalues
    """
    if isinstance(bc_type, str):
        bc_type = BCType(bc_type.lower())

    if bc_type == BCType.DIRICHLET:
        lap_sym = laplacian_symbol_dirichlet(N, dx, dtype)
        k = jnp.arange(1, N + 1, dtype=dtype)
    elif bc_type == BCType.NEUMANN:
        lap_sym = laplacian_symbol_neumann(N, dx, dtype)
        k = jnp.arange(N, dtype=dtype)
    elif bc_type == BCType.MIXED_DN:
        lap_sym = laplacian_symbol_mixed_dn(N, dx, dtype)
        k = jnp.arange(1, N + 1, dtype=dtype)
    else:
        raise ValueError(f"Unsupported BC type: {bc_type}")

    return NonperiodicFFTCache(
        N=N,
        dx=dx,
        bc_type=bc_type.value,
        laplacian_symbol=lap_sym,
        k=k
    )


# =============================================================================
# Solvers
# =============================================================================

@jax.jit
def solve_poisson_dirichlet(
    rhs: jnp.ndarray,
    laplacian_symbol: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve Poisson equation with Dirichlet BCs: Δu = f, u(0) = u(L) = 0.

    Uses DST for O(N log N) complexity.

    Args:
        rhs: Right-hand side f (interior points only)
        laplacian_symbol: Precomputed eigenvalues

    Returns:
        Solution u at interior points
    """
    # Transform to spectral space
    rhs_hat = dst_I_fast(rhs)

    # Divide by eigenvalues (solve in spectral space)
    u_hat = rhs_hat / laplacian_symbol

    # Transform back
    return idst_I_fast(u_hat)


@jax.jit
def solve_poisson_neumann(
    rhs: jnp.ndarray,
    laplacian_symbol: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve Poisson equation with Neumann BCs: Δu = f, du/dx(0) = du/dx(L) = 0.

    Uses DCT-II for O(N log N) complexity.

    Note: Solution is unique only up to a constant. We impose ∫u = 0.

    Args:
        rhs: Right-hand side f
        laplacian_symbol: Precomputed eigenvalues (λ_0 = 0)

    Returns:
        Solution u with zero mean
    """
    # Transform to spectral space using DCT-II
    rhs_hat = dct(rhs, type=2, norm='ortho')

    # Handle null space: set DC component to zero
    # (compatibility condition: ∫f = 0 for existence)
    u_hat = jnp.where(
        jnp.abs(laplacian_symbol) < 1e-14,
        0.0,
        rhs_hat / laplacian_symbol
    )

    # Transform back using inverse DCT (DCT-III)
    return idct(u_hat, type=2, norm='ortho')


@jax.jit
def solve_helmholtz_dirichlet(
    rhs: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float
) -> jnp.ndarray:
    """
    Solve Helmholtz equation: (I - dt·D·Δ)u = f with Dirichlet BCs.

    Args:
        rhs: Right-hand side f
        laplacian_symbol: Precomputed Laplacian eigenvalues
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution u
    """
    rhs_hat = dst_I_fast(rhs)
    denom = 1.0 - dt * D * laplacian_symbol
    u_hat = rhs_hat / denom
    return idst_I_fast(u_hat)


@jax.jit
def solve_helmholtz_neumann(
    rhs: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float
) -> jnp.ndarray:
    """
    Solve Helmholtz equation: (I - dt·D·Δ)u = f with Neumann BCs.

    Args:
        rhs: Right-hand side f
        laplacian_symbol: Precomputed Laplacian eigenvalues
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution u
    """
    rhs_hat = dct(rhs, type=2, norm='ortho')
    denom = 1.0 - dt * D * laplacian_symbol
    u_hat = rhs_hat / denom
    return idct(u_hat, type=2, norm='ortho')


# =============================================================================
# ETD Kernels for Non-Periodic BCs
# =============================================================================

@jax.jit
def etd1_dirichlet(
    u: jnp.ndarray,
    N_u: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    ETD1 step for diffusion with Dirichlet BCs.

    Args:
        u: Current solution (interior points)
        N_u: Nonlinear term N(u)
        eigenvalues: Diffusion eigenvalues (D * laplacian_symbol)
        dt: Time step

    Returns:
        Updated solution
    """
    z = dt * eigenvalues
    exp_z = jnp.exp(z)

    # φ₁(z) = (exp(z) - 1) / z with Taylor for small |z|
    phi1_z = jnp.where(
        jnp.abs(z) < 1e-4,
        1.0 + z/2.0 + z**2/6.0,
        (jnp.exp(z) - 1.0) / z
    )

    # Transform to spectral
    u_hat = dst_I_fast(u)
    N_hat = dst_I_fast(N_u)

    # ETD1 formula in spectral space
    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat

    return idst_I_fast(u_new_hat)


@jax.jit
def etd1_neumann(
    u: jnp.ndarray,
    N_u: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float
) -> jnp.ndarray:
    """
    ETD1 step for diffusion with Neumann BCs.

    Args:
        u: Current solution
        N_u: Nonlinear term N(u)
        eigenvalues: Diffusion eigenvalues
        dt: Time step

    Returns:
        Updated solution
    """
    z = dt * eigenvalues

    # Handle λ_0 = 0 case
    exp_z = jnp.exp(z)
    phi1_z = jnp.where(
        jnp.abs(z) < 1e-4,
        1.0 + z/2.0 + z**2/6.0,
        (jnp.exp(z) - 1.0) / z
    )

    # Transform using DCT
    u_hat = dct(u, type=2, norm='ortho')
    N_hat = dct(N_u, type=2, norm='ortho')

    # ETD1 formula
    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat

    return idct(u_new_hat, type=2, norm='ortho')


# =============================================================================
# Utility Functions
# =============================================================================

def check_compatibility_neumann(rhs: jnp.ndarray, tol: float = 1e-10) -> bool:
    """
    Check if RHS satisfies Neumann compatibility condition: ∫f = 0.

    For Neumann BCs, the Poisson equation Δu = f has a solution only if
    the integral of f over the domain is zero.
    """
    integral = jnp.sum(rhs)
    return float(jnp.abs(integral)) < tol


def project_to_compatible(rhs: jnp.ndarray) -> jnp.ndarray:
    """
    Project RHS to be compatible with Neumann BCs.

    Subtracts the mean to ensure ∫f = 0.
    """
    return rhs - jnp.mean(rhs)
