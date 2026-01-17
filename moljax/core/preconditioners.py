"""
Preconditioners for Newton-Krylov solver in MOL-JAX.

This module provides preconditioners for the implicit time stepping:
- IdentityPreconditioner: No preconditioning (baseline)
- BlockJacobiPreconditioner: Per-field diagonal scaling
- DiffusionPreconditioner: Approximate (I - dt*D*Laplacian)^-1 via Jacobi
- FFTDiffusionPreconditioner: Exact (I - dt*D*Laplacian)^-1 via FFT for periodic

Design decisions:
- Preconditioners operate on StateDict (multi-field PyTree)
- Apply method takes (residual, context) and returns preconditioned residual
- Context contains grid, dt, params needed for preconditioning
- All operations are JIT-compatible with static iteration counts
- FFT preconditioner requires periodic BCs and precomputed FFT cache
"""

from dataclasses import dataclass
from typing import Dict, Any, Callable, Protocol, Optional
import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.grid import Grid1D, Grid2D, GridType
from moljax.core.state import StateDict
from moljax.core.operators import laplacian_1d, laplacian_2d


class PrecondContext:
    """Context passed to preconditioner apply method."""

    def __init__(
        self,
        grid: GridType,
        dt: float,
        params: Dict[str, Any],
        **kwargs
    ):
        self.grid = grid
        self.dt = dt
        self.params = params
        for k, v in kwargs.items():
            setattr(self, k, v)


class Preconditioner(Protocol):
    """Protocol for preconditioners."""
    name: str

    def apply(self, r: StateDict, context: PrecondContext) -> StateDict:
        """Apply M^-1 to residual r."""
        ...


@dataclass(frozen=True)
class IdentityPreconditioner:
    """
    Identity preconditioner (no preconditioning).

    M^-1 = I, so apply just returns the input unchanged.
    """
    name: str = "identity"

    def apply(self, r: StateDict, context: PrecondContext) -> StateDict:
        """Return residual unchanged."""
        return r


@dataclass(frozen=True)
class BlockJacobiPreconditioner:
    """
    Block Jacobi preconditioner.

    Applies diagonal scaling to each field independently.
    The scaling factors are derived from the dominant term in the Jacobian.

    For diffusion-dominated problems: scale ~ 1 / (1 + dt * D / dx^2)
    For advection-dominated problems: scale ~ 1 / (1 + dt * v / dx)

    Attributes:
        diffusion_keys: Dict mapping field name to diffusion coeff param key
        advection_keys: Dict mapping field name to velocity param key
    """
    name: str = "block_jacobi"
    diffusion_keys: Dict[str, str] = None
    advection_keys: Dict[str, str] = None

    def apply(self, r: StateDict, context: PrecondContext) -> StateDict:
        """Apply block Jacobi scaling to each field."""
        grid = context.grid
        dt = context.dt
        params = context.params

        dx2 = grid.min_dx2
        dx = grid.min_dx

        result = {}
        for name, r_field in r.items():
            scale = 1.0

            # Diffusion contribution
            if self.diffusion_keys is not None and name in self.diffusion_keys:
                D_key = self.diffusion_keys[name]
                D = params.get(D_key, 0.0)
                # Jacobian diagonal: 1 - dt * D * (-4/dx^2) for 2D Laplacian
                scale += dt * D * 4.0 / dx2

            # Advection contribution
            if self.advection_keys is not None and name in self.advection_keys:
                v_key = self.advection_keys[name]
                v = params.get(v_key, 0.0)
                v_abs = jnp.abs(v) if jnp.ndim(v) == 0 else jnp.max(jnp.abs(v))
                scale += dt * v_abs / dx

            # Apply scaling
            result[name] = r_field / scale

        return result


@dataclass(frozen=True)
class DiffusionPreconditioner:
    """
    Diffusion-based preconditioner.

    Approximately inverts (I - dt * D * Laplacian) for each diffusive field
    using a fixed number of weighted Jacobi iterations.

    For periodic BCs, this could use FFT (implemented in Step 2).
    For non-periodic BCs, uses iterative weighted Jacobi.

    Attributes:
        field_diffusivities: Dict mapping field name to diffusion param key
        n_iterations: Number of Jacobi iterations (default 5)
        omega: Relaxation parameter (default 2/3 for 2D Jacobi)
    """
    name: str = "diffusion"
    field_diffusivities: Dict[str, str] = None
    n_iterations: int = 5
    omega: float = 0.6667

    def apply(self, r: StateDict, context: PrecondContext) -> StateDict:
        """Apply approximate diffusion inverse to each field."""
        grid = context.grid
        dt = context.dt
        params = context.params

        result = {}
        for name, r_field in r.items():
            if self.field_diffusivities is not None and name in self.field_diffusivities:
                D_key = self.field_diffusivities[name]
                D = params.get(D_key, 0.0)

                if D > 1e-14:
                    # Apply weighted Jacobi iterations to approximately solve
                    # (I - dt*D*Laplacian) * x = r
                    x = self._jacobi_solve(r_field, grid, dt, D)
                    result[name] = x
                else:
                    result[name] = r_field
            else:
                result[name] = r_field

        return result

    def _jacobi_solve(
        self,
        rhs: jnp.ndarray,
        grid: GridType,
        dt: float,
        D: float
    ) -> jnp.ndarray:
        """
        Weighted Jacobi iterations for (I - dt*D*Laplacian) * x = rhs.

        The iteration is:
            x_new = (1-omega)*x + omega * (rhs + dt*D*Laplacian(x)) / diag

        where diag = 1 + dt*D*(4/dx^2) for 2D (stencil diagonal).
        """
        if isinstance(grid, Grid1D):
            diag = 1.0 + dt * D * 2.0 / grid.dx ** 2
            lap_fn = lambda x: laplacian_1d(x, grid)
        else:
            diag = 1.0 + dt * D * (2.0 / grid.dx ** 2 + 2.0 / grid.dy ** 2)
            lap_fn = lambda x: laplacian_2d(x, grid)

        def iteration(x, _):
            # x_new = omega * (rhs + dt*D*Laplacian(x)) / diag + (1-omega)*x
            lap_x = lap_fn(x)
            x_new = self.omega * (rhs + dt * D * lap_x) / diag + (1.0 - self.omega) * x
            return x_new, None

        # Initial guess is rhs (corresponds to M=I)
        x0 = rhs
        x_final, _ = lax.scan(iteration, x0, None, length=self.n_iterations)

        return x_final


@dataclass(frozen=True)
class FFTDiffusionPreconditioner:
    """
    FFT-based exact diffusion preconditioner for periodic domains.

    Exactly inverts (I - dt * D * Laplacian) for each diffusive field
    using FFT. This is the optimal preconditioner for diffusion-dominated
    problems with periodic boundary conditions.

    Attributes:
        field_diffusivity_keys: Dict mapping field name to diffusivity param key
        fft_cache: Precomputed FFT cache (laplacian symbol)
        fallback: Optional fallback preconditioner for non-FFT compatible fields
    """
    name: str = "fft_diffusion"
    field_diffusivity_keys: Dict[str, str] = None
    fft_cache: Any = None  # FFTCache1D or FFTCache2D
    fallback: Optional[Any] = None  # Fallback preconditioner

    def apply(self, r: StateDict, context: PrecondContext) -> StateDict:
        """
        Apply FFT-based diffusion inverse to each field.

        For each field with diffusivity > 0, exactly solves:
        (I - dt*D*Laplacian) * x = r

        Args:
            r: Residual StateDict
            context: Context with grid, dt, params

        Returns:
            Preconditioned residual
        """
        from moljax.core.fft_solvers import solve_helmholtz

        grid = context.grid
        dt = context.dt
        params = context.params

        if self.fft_cache is None:
            # No FFT cache, fall back to identity or fallback preconditioner
            if self.fallback is not None:
                return self.fallback.apply(r, context)
            return r

        result = {}
        for name, r_field in r.items():
            D_key = self.field_diffusivity_keys.get(name, None) if self.field_diffusivity_keys else None
            D = params.get(D_key, 0.0) if D_key else 0.0

            if D > 1e-14:
                # Check if array is already interior-sized (no ghost cells)
                interior_size = grid.nx if isinstance(grid, Grid1D) else (grid.ny, grid.nx)
                is_interior_only = (r_field.shape == (interior_size,) if isinstance(grid, Grid1D)
                                    else r_field.shape == interior_size)

                # Extract interior if needed
                if is_interior_only:
                    r_interior = r_field
                elif hasattr(grid, 'interior_slice'):
                    if isinstance(grid.interior_slice, tuple):
                        sl_y, sl_x = grid.interior_slice
                        r_interior = r_field[sl_y, sl_x]
                    else:
                        r_interior = r_field[grid.interior_slice]
                else:
                    r_interior = r_field

                # Solve using FFT
                x_interior = solve_helmholtz(
                    r_interior,
                    self.fft_cache.laplacian_symbol,
                    dt,
                    D
                )

                # Embed back if needed
                if is_interior_only:
                    x_field = x_interior
                elif hasattr(grid, 'interior_slice'):
                    if isinstance(grid.interior_slice, tuple):
                        sl_y, sl_x = grid.interior_slice
                        x_field = r_field.at[sl_y, sl_x].set(x_interior)
                    else:
                        x_field = r_field.at[grid.interior_slice].set(x_interior)
                else:
                    x_field = x_interior

                result[name] = x_field
            else:
                # No diffusion, pass through
                result[name] = r_field

        return result


@dataclass(frozen=True)
class FFTAdvectionDiffusionPreconditioner:
    """
    FFT-based exact advection-diffusion preconditioner for periodic domains.

    Exactly inverts (I - dt * L) where L = -v·∇ + D*Δ for each field
    using FFT. This handles complex eigenvalues from the advection term.

    For advection-diffusion:
        λ(k) = -i*v*k + D*(2*cos(k*dx) - 2)/dx²
        (I - dt*L)^{-1} in Fourier space: u_hat / (1 - dt*λ(k))

    Attributes:
        field_diffusivity_keys: Dict mapping field name to diffusivity param key
        field_velocity_keys: Dict mapping field name to velocity param key
        fft_cache: Precomputed FFT cache (laplacian symbol and wavenumbers)
        fallback: Optional fallback preconditioner
    """
    name: str = "fft_advection_diffusion"
    field_diffusivity_keys: Dict[str, str] = None
    field_velocity_keys: Dict[str, str] = None
    fft_cache: Any = None  # FFTCache1D or FFTCache2D
    fallback: Optional[Any] = None

    def apply(self, r: StateDict, context: PrecondContext) -> StateDict:
        """
        Apply FFT-based advection-diffusion inverse to each field.

        For each field, exactly solves:
        (I - dt*(-v·∇ + D*Δ)) * x = r

        Args:
            r: Residual StateDict
            context: Context with grid, dt, params

        Returns:
            Preconditioned residual
        """
        from moljax.core.fft_solvers import build_wavenumbers_1d, build_wavenumbers_2d

        grid = context.grid
        dt = context.dt
        params = context.params

        if self.fft_cache is None:
            if self.fallback is not None:
                return self.fallback.apply(r, context)
            return r

        result = {}
        for name, r_field in r.items():
            D_key = self.field_diffusivity_keys.get(name, None) if self.field_diffusivity_keys else None
            D = params.get(D_key, 0.0) if D_key else 0.0

            v_key = self.field_velocity_keys.get(name, None) if self.field_velocity_keys else None
            v = params.get(v_key, 0.0) if v_key else 0.0

            if D > 1e-14 or abs(v) > 1e-14:
                # Check if array is already interior-sized (no ghost cells)
                interior_size = grid.nx if isinstance(grid, Grid1D) else (grid.ny, grid.nx)
                is_interior_only = (r_field.shape == (interior_size,) if isinstance(grid, Grid1D)
                                    else r_field.shape == interior_size)

                # Extract interior if needed
                if is_interior_only:
                    r_interior = r_field
                elif hasattr(grid, 'interior_slice'):
                    if isinstance(grid.interior_slice, tuple):
                        sl_y, sl_x = grid.interior_slice
                        r_interior = r_field[sl_y, sl_x]
                    else:
                        r_interior = r_field[grid.interior_slice]
                else:
                    r_interior = r_field

                # Build eigenvalues: λ(k) = -i*v*k + D*laplacian_symbol
                lam_diff = D * self.fft_cache.laplacian_symbol if D > 1e-14 else 0.0

                if isinstance(grid, Grid1D):
                    k = self.fft_cache.k
                    lam_adv = -1j * v * k if abs(v) > 1e-14 else 0.0
                    eigenvalues = lam_diff + lam_adv

                    # Solve in Fourier space
                    r_hat = jnp.fft.fft(r_interior)
                    denom = 1.0 - dt * eigenvalues
                    denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)
                    x_hat = r_hat / denom
                    x_interior = jnp.real(jnp.fft.ifft(x_hat))
                else:
                    # 2D case
                    kx, ky = self.fft_cache.kx, self.fft_cache.ky
                    # For 2D, v could be tuple (vx, vy)
                    if isinstance(v, (tuple, list)):
                        vx, vy = v
                    else:
                        vx, vy = v, 0.0
                    lam_adv = -1j * (vx * kx + vy * ky) if (abs(vx) + abs(vy)) > 1e-14 else 0.0
                    eigenvalues = lam_diff + lam_adv

                    r_hat = jnp.fft.fft2(r_interior)
                    denom = 1.0 - dt * eigenvalues
                    denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)
                    x_hat = r_hat / denom
                    x_interior = jnp.real(jnp.fft.ifft2(x_hat))

                # Embed back if needed
                if is_interior_only:
                    x_field = x_interior
                elif hasattr(grid, 'interior_slice'):
                    if isinstance(grid.interior_slice, tuple):
                        sl_y, sl_x = grid.interior_slice
                        x_field = r_field.at[sl_y, sl_x].set(x_interior)
                    else:
                        x_field = r_field.at[grid.interior_slice].set(x_interior)
                else:
                    x_field = x_interior

                result[name] = x_field
            else:
                # No advection or diffusion, pass through
                result[name] = r_field

        return result


def create_gray_scott_preconditioner() -> BlockJacobiPreconditioner:
    """Create preconditioner for Gray-Scott model."""
    return BlockJacobiPreconditioner(
        diffusion_keys={'u': 'Du', 'v': 'Dv'}
    )


def create_advdiff_preconditioner(field_names: list) -> BlockJacobiPreconditioner:
    """Create preconditioner for advection-diffusion model."""
    diff_keys = {name: 'D' for name in field_names}
    adv_keys = {name: 'vx' for name in field_names}
    return BlockJacobiPreconditioner(
        diffusion_keys=diff_keys,
        advection_keys=adv_keys
    )


def create_diffusion_preconditioner(
    field_diffusivities: Dict[str, str],
    n_iterations: int = 5
) -> DiffusionPreconditioner:
    """
    Create diffusion-based preconditioner.

    Args:
        field_diffusivities: Dict mapping field name to param key for D
        n_iterations: Number of Jacobi iterations

    Returns:
        DiffusionPreconditioner instance
    """
    return DiffusionPreconditioner(
        field_diffusivities=field_diffusivities,
        n_iterations=n_iterations
    )


def create_fft_preconditioner(
    field_diffusivity_keys: Dict[str, str],
    fft_cache,
    fallback: Optional[Any] = None
) -> FFTDiffusionPreconditioner:
    """
    Create FFT-based diffusion preconditioner for periodic domains.

    Args:
        field_diffusivity_keys: Dict mapping field name to diffusivity param key
        fft_cache: Precomputed FFT cache (from create_fft_cache)
        fallback: Optional fallback preconditioner

    Returns:
        FFTDiffusionPreconditioner instance
    """
    return FFTDiffusionPreconditioner(
        field_diffusivity_keys=field_diffusivity_keys,
        fft_cache=fft_cache,
        fallback=fallback
    )


def create_gray_scott_fft_preconditioner(fft_cache) -> FFTDiffusionPreconditioner:
    """Create FFT preconditioner for Gray-Scott model."""
    return FFTDiffusionPreconditioner(
        field_diffusivity_keys={'u': 'Du', 'v': 'Dv'},
        fft_cache=fft_cache
    )


def create_advdiff_fft_preconditioner(
    field_diffusivity_keys: Dict[str, str],
    field_velocity_keys: Dict[str, str],
    fft_cache,
    fallback: Optional[Any] = None
) -> FFTAdvectionDiffusionPreconditioner:
    """
    Create FFT-based advection-diffusion preconditioner.

    Args:
        field_diffusivity_keys: Dict mapping field name to diffusivity param key
        field_velocity_keys: Dict mapping field name to velocity param key
        fft_cache: Precomputed FFT cache
        fallback: Optional fallback preconditioner

    Returns:
        FFTAdvectionDiffusionPreconditioner instance
    """
    return FFTAdvectionDiffusionPreconditioner(
        field_diffusivity_keys=field_diffusivity_keys,
        field_velocity_keys=field_velocity_keys,
        fft_cache=fft_cache,
        fallback=fallback
    )
