"""
Hybrid spectral bounds estimation for NILT autotuning.

Provides analytic/stencil-based bounds when operator structure is known,
with matrix-free fallback estimators (power iteration, symmetric-part bounds).

Used to determine:
- rho_est: spectral radius estimate/bound
- re_max: spectral abscissa (max real part of eigenvalues)
- im_max: max imaginary part magnitude
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional

import jax
import jax.numpy as jnp


class SpectralBounds(NamedTuple):
    """Spectral bounds for an operator."""
    rho: float  # Spectral radius estimate
    re_max: float  # Max real part of eigenvalues
    im_max: float  # Max |imaginary part| of eigenvalues
    methods_used: dict[str, str]  # Which method was used for each bound
    warnings: list[str]  # Any warnings generated


@dataclass
class BoundContext:
    """Context for computing spectral bounds.

    Attributes:
        grid_spacings: Tuple of (dx,) for 1D or (dx, dy) for 2D
        operator_type: One of 'FD_LAPLACIAN', 'FD_ADVECTION_UPWIND',
                       'FD_ADVECTION_CENTERED', 'WAVE', 'COUPLED', 'UNKNOWN'
        diffusivity: Diffusion coefficient D (scalar or per-field dict)
        velocities: Advection velocities (scalar, tuple, or per-field dict)
        wave_speed: Wave speed c for acoustic/wave operators
        stencil_coeffs: Optional explicit stencil coefficients
        matvec: A(v) function for matrix-free methods
        matvec_adjoint: A^T(v) function for symmetric-part bounds
        ndim: Spatial dimension (1 or 2)
        n_points: Number of grid points per dimension
        dtype: Data type for computations
    """
    grid_spacings: tuple[float, ...]
    operator_type: str = 'UNKNOWN'
    diffusivity: float | dict[str, float] | None = None
    velocities: float | tuple[float, ...] | dict[str, Any] | None = None
    wave_speed: float | None = None
    stencil_coeffs: dict[str, jnp.ndarray] | None = None
    matvec: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    matvec_adjoint: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    ndim: int = 1
    n_points: tuple[int, ...] | None = None
    dtype: jnp.dtype = field(default_factory=lambda: jnp.float32)


# =============================================================================
# Analytic / Stencil-based bounds (preferred)
# =============================================================================

def fd_laplacian_bounds(dx: float, dy: float | None = None,
                        D: float = 1.0) -> tuple[float, float, float]:
    """
    Analytic spectral bounds for FD Laplacian with diffusivity D.

    For second-order central differences:
    - 1D: eigenvalues are -4D/dx² * sin²(kπ/(2n)) for k=1..n-1
    - Maximum magnitude: |λ_max| ≈ 4D/dx² (at k=n-1)

    Args:
        dx: Grid spacing in x
        dy: Grid spacing in y (None for 1D)
        D: Diffusion coefficient

    Returns:
        (rho, re_max, im_max) bounds
    """
    if dy is None:
        # 1D case
        rho = 4.0 * D / (dx * dx)
        re_max = 0.0  # Diffusion eigenvalues are non-positive
        im_max = 0.0  # Laplacian is self-adjoint, real eigenvalues
    else:
        # 2D case: eigenvalues sum
        rho = 4.0 * D * (1.0 / (dx * dx) + 1.0 / (dy * dy))
        re_max = 0.0
        im_max = 0.0

    return rho, re_max, im_max


def fd_advection_upwind_bounds(dx: float, dy: float | None = None,
                                vx: float = 0.0, vy: float = 0.0
                                ) -> tuple[float, float, float]:
    """
    Analytic spectral bounds for first-order upwind advection.

    Upwind discretization: (u_i - u_{i-1})/dx for v > 0
    Eigenvalues: λ = v/dx * (1 - e^{-ikdx})

    Args:
        dx: Grid spacing in x
        dy: Grid spacing in y (None for 1D)
        vx: Velocity in x direction
        vy: Velocity in y direction

    Returns:
        (rho, re_max, im_max) bounds
    """
    # Upwind advection has eigenvalues with both real and imaginary parts
    # |λ| ≤ 2|v|/dx, Re(λ) ∈ [-2|v|/dx, 0], |Im(λ)| ≤ |v|/dx

    scale_x = abs(vx) / dx if dx > 0 else 0.0
    scale_y = abs(vy) / dy if dy is not None and dy > 0 else 0.0

    rho = 2.0 * (scale_x + scale_y)  # Conservative bound
    re_max = 0.0  # Upwind is dissipative (stable)
    im_max = scale_x + scale_y  # Max imaginary part

    return rho, re_max, im_max


def fd_advection_centered_bounds(dx: float, dy: float | None = None,
                                  vx: float = 0.0, vy: float = 0.0
                                  ) -> tuple[float, float, float]:
    """
    Analytic spectral bounds for centered difference advection.

    Centered: (u_{i+1} - u_{i-1})/(2dx)
    Eigenvalues: λ = i*v*sin(kdx)/dx (purely imaginary!)

    Args:
        dx: Grid spacing in x
        dy: Grid spacing in y (None for 1D)
        vx: Velocity in x direction
        vy: Velocity in y direction

    Returns:
        (rho, re_max, im_max) bounds
    """
    scale_x = abs(vx) / dx if dx > 0 else 0.0
    scale_y = abs(vy) / dy if dy is not None and dy > 0 else 0.0

    im_max = scale_x + scale_y
    rho = im_max  # |λ| = |Im(λ)| for purely imaginary
    re_max = 0.0  # Centered advection is neutrally stable

    return rho, re_max, im_max


def wave_operator_bounds(dx: float, dy: float | None = None,
                         c: float = 1.0) -> tuple[float, float, float]:
    """
    Spectral bounds for first-order wave/acoustic system.

    Wave equation as first-order system has eigenvalues ±i*c*k
    where k is the wavenumber.

    Args:
        dx: Grid spacing in x
        dy: Grid spacing in y (None for 1D)
        c: Wave speed

    Returns:
        (rho, re_max, im_max) bounds
    """
    # Max wavenumber k_max ≈ π/dx
    k_max_x = jnp.pi / dx
    k_max_y = jnp.pi / dy if dy is not None else 0.0

    # For 2D, use Euclidean norm of wavenumber
    k_max = jnp.sqrt(k_max_x**2 + k_max_y**2)

    im_max = float(c * k_max)
    rho = im_max
    re_max = 0.0  # Wave equation is energy-conserving

    return rho, re_max, im_max


def gershgorin_bound_from_stencil(coeffs: jnp.ndarray) -> float:
    """
    Gershgorin row-sum bound from stencil coefficients.

    For a stencil operator, each row has the same pattern of coefficients.
    ρ(A) ≤ max_i Σ_j |a_ij| = Σ |stencil_coeffs|

    Args:
        coeffs: 1D array of stencil coefficients

    Returns:
        Gershgorin bound on spectral radius
    """
    return float(jnp.sum(jnp.abs(coeffs)))


# =============================================================================
# Matrix-free fallback estimators
# =============================================================================

def power_iteration_rho(matvec: Callable[[jnp.ndarray], jnp.ndarray],
                        x0: jnp.ndarray,
                        max_iters: int = 50,
                        tol: float = 1e-4) -> float:
    """
    Estimate spectral radius via power iteration.

    Args:
        matvec: Function computing A @ x
        x0: Initial vector (should be random)
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        Estimated spectral radius
    """
    x = x0 / jnp.linalg.norm(x0.ravel())
    rho_prev = 0.0

    for _ in range(max_iters):
        y = matvec(x)
        rho = jnp.linalg.norm(y.ravel())

        if rho < 1e-14:
            return 0.0

        x = y / rho

        if abs(rho - rho_prev) < tol * max(rho, 1.0):
            break
        rho_prev = rho

    return float(rho)


def symmetric_part_bound_remax(matvec: Callable[[jnp.ndarray], jnp.ndarray],
                                matvec_adjoint: Callable[[jnp.ndarray], jnp.ndarray],
                                x0: jnp.ndarray,
                                max_iters: int = 30) -> float:
    """
    Estimate max Re(λ) via symmetric part S = (A + A^T)/2.

    max Re(λ(A)) ≤ λ_max(S)

    Args:
        matvec: Function computing A @ x
        matvec_adjoint: Function computing A^T @ x
        x0: Initial vector
        max_iters: Maximum iterations for power iteration

    Returns:
        Estimated upper bound on max Re(λ)
    """
    def symmetric_matvec(v):
        return 0.5 * (matvec(v) + matvec_adjoint(v))

    return power_iteration_rho(symmetric_matvec, x0, max_iters)


def antisymmetric_part_bound_immax(matvec: Callable[[jnp.ndarray], jnp.ndarray],
                                    matvec_adjoint: Callable[[jnp.ndarray], jnp.ndarray],
                                    x0: jnp.ndarray,
                                    max_iters: int = 30) -> float:
    """
    Estimate max |Im(λ)| via antisymmetric part K = (A - A^T)/2.

    max |Im(λ(A))| ≤ ||K||_2

    Args:
        matvec: Function computing A @ x
        matvec_adjoint: Function computing A^T @ x
        x0: Initial vector
        max_iters: Maximum iterations

    Returns:
        Estimated upper bound on max |Im(λ)|
    """
    def antisymmetric_matvec(v):
        return 0.5 * (matvec(v) - matvec_adjoint(v))

    return power_iteration_rho(antisymmetric_matvec, x0, max_iters)


# =============================================================================
# Main hybrid bounds function
# =============================================================================

def compute_spectral_bounds(ctx: BoundContext) -> SpectralBounds:
    """
    Compute hybrid spectral bounds using analytic methods when possible,
    with matrix-free fallback otherwise.

    Args:
        ctx: BoundContext with operator information

    Returns:
        SpectralBounds with estimates and method info
    """
    methods_used = {}
    warnings = []

    rho = float('nan')
    re_max = float('nan')
    im_max = float('nan')

    dx = ctx.grid_spacings[0]
    dy = ctx.grid_spacings[1] if len(ctx.grid_spacings) > 1 else None

    # Try analytic bounds based on operator type
    if ctx.operator_type == 'FD_LAPLACIAN':
        D = ctx.diffusivity if ctx.diffusivity is not None else 1.0
        if isinstance(D, dict):
            D = max(D.values())
        rho, re_max, im_max = fd_laplacian_bounds(dx, dy, D)
        methods_used['rho'] = 'analytic_laplacian'
        methods_used['re_max'] = 'analytic_laplacian'
        methods_used['im_max'] = 'analytic_laplacian'

    elif ctx.operator_type == 'FD_ADVECTION_UPWIND':
        vx = 0.0
        vy = 0.0
        if ctx.velocities is not None:
            if isinstance(ctx.velocities, (int, float)):
                vx = float(ctx.velocities)
            elif isinstance(ctx.velocities, tuple):
                vx = ctx.velocities[0]
                vy = ctx.velocities[1] if len(ctx.velocities) > 1 else 0.0
            elif isinstance(ctx.velocities, dict):
                # Take max velocity
                all_vels = []
                for v in ctx.velocities.values():
                    if isinstance(v, tuple):
                        all_vels.extend(v)
                    else:
                        all_vels.append(v)
                vx = max(abs(v) for v in all_vels) if all_vels else 0.0

        rho, re_max, im_max = fd_advection_upwind_bounds(dx, dy, vx, vy)
        methods_used['rho'] = 'analytic_advection_upwind'
        methods_used['re_max'] = 'analytic_advection_upwind'
        methods_used['im_max'] = 'analytic_advection_upwind'

    elif ctx.operator_type == 'FD_ADVECTION_CENTERED':
        vx = 0.0
        vy = 0.0
        if ctx.velocities is not None:
            if isinstance(ctx.velocities, (int, float)):
                vx = float(ctx.velocities)
            elif isinstance(ctx.velocities, tuple):
                vx = ctx.velocities[0]
                vy = ctx.velocities[1] if len(ctx.velocities) > 1 else 0.0

        rho, re_max, im_max = fd_advection_centered_bounds(dx, dy, vx, vy)
        methods_used['rho'] = 'analytic_advection_centered'
        methods_used['re_max'] = 'analytic_advection_centered'
        methods_used['im_max'] = 'analytic_advection_centered'

    elif ctx.operator_type == 'WAVE':
        c = ctx.wave_speed if ctx.wave_speed is not None else 1.0
        rho, re_max, im_max = wave_operator_bounds(dx, dy, c)
        methods_used['rho'] = 'analytic_wave'
        methods_used['re_max'] = 'analytic_wave'
        methods_used['im_max'] = 'analytic_wave'

    elif ctx.operator_type == 'COUPLED':
        # For coupled systems, aggregate bounds from components
        rho_total = 0.0
        re_max_total = 0.0
        im_max_total = 0.0

        # Diffusion contribution
        if ctx.diffusivity is not None:
            D = ctx.diffusivity
            if isinstance(D, dict):
                D = max(D.values())
            rho_d, re_d, im_d = fd_laplacian_bounds(dx, dy, D)
            rho_total += rho_d
            re_max_total = max(re_max_total, re_d)
            im_max_total = max(im_max_total, im_d)

        # Advection contribution
        if ctx.velocities is not None:
            vx = 0.0
            vy = 0.0
            if isinstance(ctx.velocities, (int, float)):
                vx = float(ctx.velocities)
            elif isinstance(ctx.velocities, tuple):
                vx = ctx.velocities[0]
                vy = ctx.velocities[1] if len(ctx.velocities) > 1 else 0.0
            rho_a, re_a, im_a = fd_advection_upwind_bounds(dx, dy, vx, vy)
            rho_total += rho_a
            re_max_total = max(re_max_total, re_a)
            im_max_total = max(im_max_total, im_a)

        # Wave contribution
        if ctx.wave_speed is not None:
            rho_w, re_w, im_w = wave_operator_bounds(dx, dy, ctx.wave_speed)
            rho_total = max(rho_total, rho_w)
            re_max_total = max(re_max_total, re_w)
            im_max_total = max(im_max_total, im_w)

        rho = rho_total
        re_max = re_max_total
        im_max = im_max_total
        methods_used['rho'] = 'analytic_coupled'
        methods_used['re_max'] = 'analytic_coupled'
        methods_used['im_max'] = 'analytic_coupled'

    # Gershgorin bound from explicit stencil if available
    if ctx.stencil_coeffs is not None and jnp.isnan(rho):
        gershgorin_bounds = []
        for name, coeffs in ctx.stencil_coeffs.items():
            gershgorin_bounds.append(gershgorin_bound_from_stencil(coeffs))
        if gershgorin_bounds:
            rho = max(gershgorin_bounds)
            methods_used['rho'] = 'gershgorin_stencil'

    # Matrix-free fallback for remaining unknowns
    if ctx.matvec is not None:
        # Create random initial vector
        if ctx.n_points is not None:
            shape = ctx.n_points
        else:
            shape = (64,) if ctx.ndim == 1 else (32, 32)

        key = jax.random.PRNGKey(42)
        x0 = jax.random.normal(key, shape, dtype=ctx.dtype)

        # Spectral radius via power iteration
        if jnp.isnan(rho):
            try:
                rho = power_iteration_rho(ctx.matvec, x0)
                methods_used['rho'] = 'power_iteration'
            except Exception as e:
                warnings.append(f"Power iteration failed: {e}")
                rho = float('nan')

        # Symmetric/antisymmetric part bounds if adjoint available
        if ctx.matvec_adjoint is not None:
            if jnp.isnan(re_max):
                try:
                    re_max = symmetric_part_bound_remax(
                        ctx.matvec, ctx.matvec_adjoint, x0
                    )
                    methods_used['re_max'] = 'symmetric_part_power'
                except Exception as e:
                    warnings.append(f"Symmetric part estimation failed: {e}")

            if jnp.isnan(im_max):
                try:
                    im_max = antisymmetric_part_bound_immax(
                        ctx.matvec, ctx.matvec_adjoint, x0
                    )
                    methods_used['im_max'] = 'antisymmetric_part_power'
                except Exception as e:
                    warnings.append(f"Antisymmetric part estimation failed: {e}")

    # Fallback defaults
    if jnp.isnan(re_max):
        if not jnp.isnan(rho):
            # Conservative: assume could be anywhere in disk
            re_max = rho
            methods_used['re_max'] = 'fallback_rho'
            warnings.append("re_max estimated from rho (conservative)")
        else:
            re_max = 0.0
            methods_used['re_max'] = 'fallback_zero'
            warnings.append("re_max defaulted to 0 (no information)")

    if jnp.isnan(im_max):
        if not jnp.isnan(rho):
            im_max = rho
            methods_used['im_max'] = 'fallback_rho'
            warnings.append("im_max estimated from rho (conservative)")
        else:
            im_max = 1.0
            methods_used['im_max'] = 'fallback_default'
            warnings.append("im_max defaulted to 1.0 (no information)")

    if jnp.isnan(rho):
        rho = 1.0
        methods_used['rho'] = 'fallback_default'
        warnings.append("rho defaulted to 1.0 (no information)")

    return SpectralBounds(
        rho=rho,
        re_max=re_max,
        im_max=im_max,
        methods_used=methods_used,
        warnings=warnings
    )


def bounds_from_mol_model(model, field_name: str | None = None) -> SpectralBounds:
    """
    Extract spectral bounds from a MOL model's metadata.

    Args:
        model: A MOL Model object from moljax.core.model
        field_name: Optional specific field to analyze

    Returns:
        SpectralBounds for the operator
    """
    # Build context from model metadata
    dx = model.grid.dx
    dy = getattr(model.grid, 'dy', None)
    grid_spacings = (dx,) if dy is None else (dx, dy)

    # Determine operator type from model metadata
    operator_type = 'UNKNOWN'
    diffusivity = None
    velocities = None

    if hasattr(model, 'metadata'):
        meta = model.metadata
        if 'operator_type' in meta:
            operator_type = meta['operator_type']
        if 'diffusivities' in meta:
            diffusivity = meta['diffusivities']
        if 'velocities' in meta:
            velocities = meta['velocities']

    # Infer from model type
    if hasattr(model, 'D_u') or hasattr(model, 'diffusivities'):
        if operator_type == 'UNKNOWN':
            operator_type = 'FD_LAPLACIAN'
        if diffusivity is None:
            if hasattr(model, 'diffusivities'):
                diffusivity = model.diffusivities
            elif hasattr(model, 'D_u'):
                diffusivity = {'u': model.D_u}
                if hasattr(model, 'D_v'):
                    diffusivity['v'] = model.D_v

    if hasattr(model, 'velocity') or hasattr(model, 'vx'):
        if operator_type == 'UNKNOWN':
            operator_type = 'FD_ADVECTION_UPWIND'
        if velocities is None:
            if hasattr(model, 'velocity'):
                velocities = model.velocity
            elif hasattr(model, 'vx'):
                velocities = (model.vx, getattr(model, 'vy', 0.0))

    ndim = 1 if model.grid.ndim == 1 else 2
    n_points = model.grid.shape

    ctx = BoundContext(
        grid_spacings=grid_spacings,
        operator_type=operator_type,
        diffusivity=diffusivity,
        velocities=velocities,
        ndim=ndim,
        n_points=n_points,
        dtype=model.dtype
    )

    return compute_spectral_bounds(ctx)
