"""
FFT-diagonalizable linear operators for Method-of-Lines PDE solvers.

This module provides a clean interface for operators that can be efficiently
applied and inverted using FFT on periodic domains. Key classes:

- FFTLinearOperator: Protocol defining the interface
- DiffusionOperator: L = D * Δ (Laplacian with diffusivity D)
- AdvectionDiffusionOperator: L = v·∇ + D*Δ

All operators provide:
- matvec(u): Apply L*u via FFT (O(N log N))
- solve(rhs, dt): Solve (I - dt*L)*u = rhs via FFT (O(N log N))
- exp_matvec(u, dt): Apply exp(dt*L)*u via FFT (for ETD methods)
- spectral_bounds(): Exact bounds from eigenvalues

Design:
- Operators are immutable (frozen dataclass)
- FFT cache is computed once at construction
- All operations work on interior points (no ghost cells)
- Compatible with JAX JIT compilation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, Union

import jax.numpy as jnp
from jax import jit

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.fft_solvers import (
    FFTCache1D,
    FFTCache2D,
    create_fft_cache_1d,
    create_fft_cache_2d,
    build_wavenumbers_1d,
    build_wavenumbers_2d,
)
from moljax.laplace.spectral_bounds import SpectralBounds


GridType = Union[Grid1D, Grid2D]


@runtime_checkable
class FFTLinearOperator(Protocol):
    """Protocol for FFT-diagonalizable linear operators.

    Any operator implementing this protocol can be used with ETD integrators
    and FFT-based implicit solvers.
    """

    name: str
    grid: GridType

    @property
    def eigenvalues(self) -> jnp.ndarray:
        """Eigenvalues λ(k) in Fourier space."""
        ...

    def matvec(self, u: jnp.ndarray) -> jnp.ndarray:
        """Apply L*u via FFT: ifft(λ * fft(u))."""
        ...

    def solve(self, rhs: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Solve (I - dt*L)*u = rhs via FFT."""
        ...

    def exp_matvec(self, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Apply exp(dt*L)*u via FFT (for ETD methods)."""
        ...

    def spectral_bounds(self) -> SpectralBounds:
        """Exact spectral bounds from eigenvalues."""
        ...


@dataclass(frozen=True)
class DiffusionOperator:
    """FFT-diagonalized diffusion operator: L = D*Δ.

    For periodic domains with second-order central differences:
    - Eigenvalues: λ(k) = D * (2*cos(k*dx) - 2) / dx²
    - All eigenvalues are real and ≤ 0 (stable)
    - Spectral radius: ρ = 4D/dx²

    Args:
        grid: Grid1D or Grid2D instance
        D: Diffusion coefficient (must be > 0)
        dtype: Data type for computations

    Example:
        >>> grid = Grid1D.uniform(128, 0, 1)
        >>> op = DiffusionOperator(grid, D=0.01)
        >>> u_new = op.solve(rhs, dt=0.1)  # Solve (I - dt*D*Δ)u = rhs
        >>> Lu = op.matvec(u)  # Apply D*Δu
    """

    grid: GridType
    D: float
    dtype: jnp.dtype = jnp.float64
    name: str = field(default="diffusion", repr=False)
    _cache: FFTCache1D | FFTCache2D = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if self.D <= 0:
            raise ValueError(f"Diffusion coefficient D must be > 0, got {self.D}")

        if isinstance(self.grid, Grid1D):
            cache = create_fft_cache_1d(self.grid, self.dtype)
        else:
            cache = create_fft_cache_2d(self.grid, self.dtype)
        object.__setattr__(self, '_cache', cache)

    @property
    def eigenvalues(self) -> jnp.ndarray:
        """Eigenvalues λ(k) = D * (2*cos(k*dx) - 2) / dx²."""
        return self.D * self._cache.laplacian_symbol

    def matvec(self, u: jnp.ndarray) -> jnp.ndarray:
        """Apply D*Δu via FFT."""
        if u.ndim == 1:
            u_hat = jnp.fft.fft(u)
            Lu_hat = self.eigenvalues * u_hat
            return jnp.real(jnp.fft.ifft(Lu_hat))
        else:
            u_hat = jnp.fft.fft2(u)
            Lu_hat = self.eigenvalues * u_hat
            return jnp.real(jnp.fft.ifft2(Lu_hat))

    def solve(self, rhs: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Solve (I - dt*D*Δ)*u = rhs via FFT.

        In Fourier space: u_hat = rhs_hat / (1 - dt*λ(k))
        """
        denom = 1.0 - dt * self.eigenvalues
        # Avoid division by zero (shouldn't happen for positive D, dt)
        denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)

        if rhs.ndim == 1:
            rhs_hat = jnp.fft.fft(rhs)
            u_hat = rhs_hat / denom
            return jnp.real(jnp.fft.ifft(u_hat))
        else:
            rhs_hat = jnp.fft.fft2(rhs)
            u_hat = rhs_hat / denom
            return jnp.real(jnp.fft.ifft2(u_hat))

    def exp_matvec(self, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Apply exp(dt*D*Δ)*u via FFT (for ETD methods).

        In Fourier space: u_new_hat = exp(dt*λ(k)) * u_hat
        """
        exp_lam = jnp.exp(dt * self.eigenvalues)

        if u.ndim == 1:
            u_hat = jnp.fft.fft(u)
            u_new_hat = exp_lam * u_hat
            return jnp.real(jnp.fft.ifft(u_new_hat))
        else:
            u_hat = jnp.fft.fft2(u)
            u_new_hat = exp_lam * u_hat
            return jnp.real(jnp.fft.ifft2(u_new_hat))

    def spectral_bounds(self) -> SpectralBounds:
        """Exact spectral bounds from eigenvalues."""
        lam = self.eigenvalues
        return SpectralBounds(
            rho=float(jnp.max(jnp.abs(lam))),
            re_max=float(jnp.max(jnp.real(lam))),  # ≤ 0 for diffusion
            im_max=0.0,  # Diffusion has real eigenvalues
            methods_used={'exact_fft': f'DiffusionOperator(D={self.D})'},
            warnings=[]
        )


@dataclass(frozen=True)
class AdvectionDiffusionOperator:
    """FFT-diagonalized advection-diffusion operator: L = -v·∇ + D*Δ.

    For the PDE u_t = Lu = -v*u_x + D*u_xx (advection to the right for v > 0):
    - 1D eigenvalues: λ(k) = -i*v*k + D*(2*cos(k*dx) - 2)/dx²
    - Re(λ) = D*(2*cos(k*dx) - 2)/dx² ≤ 0 (diffusive damping)
    - Im(λ) = -v*k (advective transport)

    Note: This operator represents L in u_t = Lu, where positive v moves
    waves to the RIGHT (positive x direction).

    Args:
        grid: Grid1D instance (2D not yet supported)
        v: Advection velocity (positive = rightward motion)
        D: Diffusion coefficient (can be 0 for pure advection)
        dtype: Data type for computations

    Example:
        >>> grid = Grid1D.uniform(256, 0, 1)
        >>> op = AdvectionDiffusionOperator(grid, v=1.0, D=0.01)
        >>> bounds = op.spectral_bounds()  # Get exact CFL info
    """

    grid: Grid1D
    v: float
    D: float = 0.0
    dtype: jnp.dtype = jnp.float64
    name: str = field(default="advection_diffusion", repr=False)
    _cache: FFTCache1D = field(init=False, repr=False, compare=False)
    _k: jnp.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.grid, Grid1D):
            raise NotImplementedError("AdvectionDiffusionOperator only supports 1D grids currently")

        cache = create_fft_cache_1d(self.grid, self.dtype)
        k = build_wavenumbers_1d(self.grid.nx, self.grid.dx, self.dtype)
        object.__setattr__(self, '_cache', cache)
        object.__setattr__(self, '_k', k)

    @property
    def eigenvalues(self) -> jnp.ndarray:
        """Eigenvalues λ(k) = -i*v*k + D*(2*cos(k*dx) - 2)/dx²."""
        # Negative sign on advection: u_t = -v*u_x means exp(-i*v*k*t) phase shift
        lam_adv = -1j * self.v * self._k
        lam_diff = self.D * self._cache.laplacian_symbol if self.D > 0 else 0.0
        return lam_adv + lam_diff

    def matvec(self, u: jnp.ndarray) -> jnp.ndarray:
        """Apply (v·∇ + D*Δ)u via FFT."""
        u_hat = jnp.fft.fft(u)
        Lu_hat = self.eigenvalues * u_hat
        return jnp.real(jnp.fft.ifft(Lu_hat))

    def solve(self, rhs: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Solve (I - dt*L)*u = rhs via FFT."""
        denom = 1.0 - dt * self.eigenvalues
        denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)

        rhs_hat = jnp.fft.fft(rhs)
        u_hat = rhs_hat / denom
        return jnp.real(jnp.fft.ifft(u_hat))

    def exp_matvec(self, u: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Apply exp(dt*L)*u via FFT (for ETD methods)."""
        exp_lam = jnp.exp(dt * self.eigenvalues)

        u_hat = jnp.fft.fft(u)
        u_new_hat = exp_lam * u_hat
        return jnp.real(jnp.fft.ifft(u_new_hat))

    def spectral_bounds(self) -> SpectralBounds:
        """Exact spectral bounds from eigenvalues."""
        lam = self.eigenvalues
        return SpectralBounds(
            rho=float(jnp.max(jnp.abs(lam))),
            re_max=float(jnp.max(jnp.real(lam))),  # ≤ 0 if D > 0
            im_max=float(jnp.max(jnp.abs(jnp.imag(lam)))),
            methods_used={'exact_fft': f'AdvectionDiffusionOperator(v={self.v}, D={self.D})'},
            warnings=[]
        )


def exact_cfl_dt(
    op: FFTLinearOperator,
    method: str = 'explicit',
    safety: float = 0.9
) -> float:
    """Compute exact CFL timestep from FFT eigenvalues.

    Args:
        op: FFT linear operator
        method: Integration method ('explicit', 'imex', 'etd')
        safety: Safety factor (0 < safety ≤ 1)

    Returns:
        Maximum stable timestep for the given method
    """
    lam = op.eigenvalues

    if method == 'explicit':
        # Forward Euler stability: |1 + dt*λ| ≤ 1 for all λ
        # For purely real negative λ: dt ≤ 2/|λ_max|
        # For complex λ: more conservative
        rho = float(jnp.max(jnp.abs(lam)))
        if rho < 1e-14:
            return float('inf')
        return safety * 2.0 / rho

    elif method == 'imex':
        # Only explicit part matters (diffusion handled implicitly)
        # Return large dt if fully implicit
        return safety * 1.0

    elif method == 'etd':
        # No stability limit from L (exact exponential)
        # Limited only by accuracy of nonlinear terms
        return safety * 1.0

    else:
        raise ValueError(f"Unknown method: {method}")
