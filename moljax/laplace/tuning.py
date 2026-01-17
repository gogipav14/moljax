"""
NILT parameter autotuner using hybrid spectral bounds.

Automatically selects NILT parameters (a, dt, N, T) based on
operator spectral properties and grid constraints, analogous
to CFL/Heisenberg-style time-frequency coupling.

Key insight: dt determines ω_max = π/dt (Nyquist frequency).
For accurate inversion, ω_max must cover the significant
frequency content of F(s), which is determined by the
spectral bounds of the underlying operator.
"""

from __future__ import annotations

from typing import Literal, NamedTuple
import warnings as _warnings

import jax.numpy as jnp
import numpy as np

from .spectral_bounds import BoundContext, SpectralBounds, compute_spectral_bounds


class TunedNILTParams(NamedTuple):
    """Autotuned NILT parameters."""
    dt: float  # Time step
    N: int  # FFT size (power of 2)
    T: float  # Half-period (2T = N*dt)
    a: float  # Bromwich shift
    omega_max: float  # Maximum frequency = π/dt
    omega_req: float  # Required frequency coverage
    bound_sources: dict[str, str]  # Methods used for bounds
    warnings: list[str]  # Any tuning warnings
    diagnostics: dict  # Additional diagnostic info


def next_power_of_two(n: int) -> int:
    """Return smallest power of 2 >= n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


def _log_max_float(dtype, safety_log_margin: float = 10.0) -> float:
    """
    Return L = log(maxfloat(dtype)) - safety_log_margin.

    This is the maximum safe exponent for exp() to avoid overflow.

    Args:
        dtype: JAX dtype (float32 or float64) or string like 'float32', 'float64'
        safety_log_margin: Safety margin in log units (default: 10.0)

    Returns:
        Maximum safe log value for exponential operations
    """
    # Convert dtype to string and normalize
    dtype_str = str(dtype)
    if 'float32' in dtype_str:
        dt = np.dtype('float32')
    elif 'float64' in dtype_str or 'float_' in dtype_str:
        dt = np.dtype('float64')
    else:
        # Default to float64 for unknown types
        dt = np.dtype('float64')

    return float(np.log(np.finfo(dt).max) - safety_log_margin)


def tune_nilt_params(
    *,
    t_end: float,
    bounds: SpectralBounds | dict | None = None,
    ctx: BoundContext | None = None,
    dtype=jnp.float64,
    shift_mode: Literal["auto", "never", "always"] = "auto",
    delta_min: float = 1e-3,
    eps_tail: float = 1e-8,
    period_factor: float = 4.0,
    omega_factor: float = 1.5,
    safety_log_margin: float = 10.0,
    N_min: int = 256,
    N_max: int = 8192,
) -> TunedNILTParams:
    """
    Autotune NILT parameters using three-guardrail spectral framework.

    Guardrails:
        1. Dynamic-range: a * t_max ≤ L_dtype
        2. Spectral placement: a ≥ alpha + delta_min
        3. Wraparound: exp(-(a-alpha)*t_max) ≤ eps_tail

    Args:
        t_end: End time for inversion
        bounds: SpectralBounds or dict with {rho, re_max, im_max}
        ctx: BoundContext for computing bounds (if bounds not provided)
        dtype: Data type for overflow budget calculation
        shift_mode: Bromwich shift policy ("auto", "never", "always")
        delta_min: Minimal spectral margin for correctness
        eps_tail: Wraparound suppression threshold
        period_factor: T = period_factor * t_end / 2
        omega_factor: Frequency oversampling factor
        safety_log_margin: Safety margin in log units for overflow
        N_min: Minimum FFT size
        N_max: Maximum FFT size

    Returns:
        TunedNILTParams with selected parameters and diagnostics
    """
    warnings = []

    # Get spectral bounds
    if bounds is None and ctx is not None:
        bounds = compute_spectral_bounds(ctx)
    elif bounds is None:
        # No bounds provided - use defaults
        bounds = SpectralBounds(
            rho=10.0,
            re_max=0.0,
            im_max=10.0,
            methods_used={'all': 'default'},
            warnings=['No bounds provided, using defaults']
        )
        warnings.append("No spectral bounds provided; using conservative defaults")

    # Extract bound values
    if isinstance(bounds, SpectralBounds):
        alpha = float(bounds.re_max)  # Spectral abscissa
        rho = float(bounds.rho)       # Spectral radius
        im_max = float(bounds.im_max) # Max imaginary part
        bound_sources = bounds.methods_used
        warnings.extend(bounds.warnings)
    else:
        # Dict format
        alpha = float(bounds.get('re_max', 0.0))
        rho = float(bounds.get('rho', 10.0))
        im_max = float(bounds.get('im_max', rho))
        bound_sources = bounds.get('methods_used', {'all': 'dict_input'})

    # Period selection
    T = period_factor * t_end / 2.0
    t_max = 2.0 * T

    # Dynamic-range budget
    L = _log_max_float(dtype, safety_log_margin=safety_log_margin)

    # Initialize variables that may or may not be computed
    a_min = None

    # Mode selection and shift computation
    if shift_mode == "never":
        a = 0.0

    elif shift_mode == "auto" and alpha <= 0:
        # Stable operator: try imaginary-axis mode (a=0) first
        # But check wraparound criterion
        tail_margin = np.log(1.0 / eps_tail) / t_max

        if tail_margin > abs(alpha):
            # a=0 insufficient for wraparound; use minimal shift
            a = max(0.0, alpha + tail_margin)
            warnings.append(
                f"Stable operator but minimal shift a={a:.2e} needed for wraparound "
                f"(eps_tail={eps_tail:.1e}, t_max={t_max:.2f})"
            )
        else:
            # a=0 is fine (MATLAB-faithful)
            a = 0.0

    else:
        # Shifted mode (unstable operator or always mode)
        # Compute feasible interval
        tail_term = np.log(1.0 / eps_tail) / t_max
        a_min = alpha + max(delta_min, tail_term)
        a_max = L / t_max

        if a_min <= a_max:
            # Feasible: choose minimal shift
            a = float(a_min)
        else:
            # Infeasible: recommend remedies
            # Try dtype upgrade first
            if dtype == jnp.float32 or str(dtype) == 'float32':
                L_f64 = _log_max_float(jnp.float64, safety_log_margin)
                a_max_f64 = L_f64 / t_max
                if a_min <= a_max_f64:
                    raise ValueError(
                        f"Overflow infeasible with float32 (L={L:.2f}). "
                        f"Switch to float64 (L={L_f64:.2f}) to enable a={a_min:.2e}."
                    )

            # Recommend time-domain splitting
            T_chunk = 0.9 * L / (2.0 * (alpha + delta_min))
            K_chunks = int(np.ceil(t_end / T_chunk))

            raise ValueError(
                f"NILT-CFL infeasible for single-shot (a_min={a_min:.2e} > a_max={a_max:.2e}).\n"
                f"  Stiffness × duration × precision constraint violated.\n"
                f"  Recommend: Split into {K_chunks} windows of T≈{T_chunk:.2f} each.\n"
                f"  Parameters: alpha={alpha:.2e}, t_max={t_max:.2f}, L={L:.2f}"
            )

    # Nyquist / sampling guardrail
    omega_req = max(im_max, 0.5 * rho)
    dt_nyquist = np.pi / (omega_factor * omega_req) if omega_req > 0 else t_max / N_min

    # Compute N
    N_target = int(np.ceil(t_max / dt_nyquist))
    N = 2 ** int(np.ceil(np.log2(N_target)))  # Next power of 2
    N = max(N_min, min(N, N_max))

    if N == N_max and N_target > N_max:
        warnings.append(
            f"N clamped to {N_max}. Consider splitting time domain for better resolution."
        )

    # Final dt (consistent with 2T = N*dt)
    dt = t_max / N

    # Emit warnings
    for w in warnings:
        _warnings.warn(w, UserWarning)

    return TunedNILTParams(
        dt=dt,
        N=N,
        T=T,
        a=a,
        omega_max=np.pi / dt,
        omega_req=omega_req,
        bound_sources=bound_sources,
        warnings=warnings,
        diagnostics={
            "alpha": alpha,
            "rho": rho,
            "im_max": im_max,
            "a_min": a_min if shift_mode != "never" and not (shift_mode == "auto" and alpha <= 0 and a == 0) else None,
            "a_max": L / t_max if shift_mode != "never" else None,
            "shift_mode": shift_mode,
            "L_dtype": L,
            "t_max": t_max,
        }
    )


def tune_nilt_from_mol_model(
    model,
    t_end: float,
    **kwargs
) -> TunedNILTParams:
    """
    Tune NILT parameters from a MOL model.

    Extracts spectral bounds from model metadata and grid,
    then tunes NILT parameters accordingly.

    Args:
        model: MOL Model object
        t_end: End time for inversion
        **kwargs: Additional arguments for tune_nilt_params

    Returns:
        TunedNILTParams
    """
    from .spectral_bounds import bounds_from_mol_model

    bounds = bounds_from_mol_model(model)
    return tune_nilt_params(t_end=t_end, bounds=bounds, **kwargs)


def tune_for_diffusion(
    dx: float,
    D: float,
    t_end: float,
    ndim: int = 1,
    **kwargs
) -> TunedNILTParams:
    """
    Tune NILT parameters for diffusion operator.

    Args:
        dx: Grid spacing
        D: Diffusion coefficient
        t_end: End time
        ndim: Spatial dimension (1 or 2)
        **kwargs: Additional tuning parameters

    Returns:
        TunedNILTParams
    """
    grid_spacings = (dx,) * ndim

    ctx = BoundContext(
        grid_spacings=grid_spacings,
        operator_type='FD_LAPLACIAN',
        diffusivity=D,
        ndim=ndim
    )

    bounds = compute_spectral_bounds(ctx)
    return tune_nilt_params(t_end=t_end, bounds=bounds, **kwargs)


def tune_for_advection(
    dx: float,
    v: float,
    t_end: float,
    scheme: str = 'upwind',
    ndim: int = 1,
    **kwargs
) -> TunedNILTParams:
    """
    Tune NILT parameters for advection operator.

    Args:
        dx: Grid spacing
        v: Advection velocity
        t_end: End time
        scheme: 'upwind' or 'centered'
        ndim: Spatial dimension
        **kwargs: Additional tuning parameters

    Returns:
        TunedNILTParams
    """
    grid_spacings = (dx,) * ndim

    if scheme == 'upwind':
        op_type = 'FD_ADVECTION_UPWIND'
    else:
        op_type = 'FD_ADVECTION_CENTERED'

    ctx = BoundContext(
        grid_spacings=grid_spacings,
        operator_type=op_type,
        velocities=v,
        ndim=ndim
    )

    bounds = compute_spectral_bounds(ctx)
    return tune_nilt_params(t_end=t_end, bounds=bounds, **kwargs)


def tune_for_advection_diffusion(
    dx: float,
    v: float,
    D: float,
    t_end: float,
    ndim: int = 1,
    **kwargs
) -> TunedNILTParams:
    """
    Tune NILT parameters for advection-diffusion operator.

    Args:
        dx: Grid spacing
        v: Advection velocity
        D: Diffusion coefficient
        t_end: End time
        ndim: Spatial dimension
        **kwargs: Additional tuning parameters

    Returns:
        TunedNILTParams
    """
    grid_spacings = (dx,) * ndim

    ctx = BoundContext(
        grid_spacings=grid_spacings,
        operator_type='COUPLED',
        diffusivity=D,
        velocities=v,
        ndim=ndim
    )

    bounds = compute_spectral_bounds(ctx)
    return tune_nilt_params(t_end=t_end, bounds=bounds, **kwargs)


# =============================================================================
# Diagnostic utilities
# =============================================================================

def diagnose_tuning(params: TunedNILTParams) -> dict:
    """
    Generate diagnostic report for tuned parameters.

    Args:
        params: TunedNILTParams from autotuner

    Returns:
        Dict with diagnostic information and recommendations
    """
    report = {
        'parameters': {
            'dt': params.dt,
            'N': params.N,
            'T': params.T,
            'a': params.a
        },
        'coverage': {
            'omega_max': params.omega_max,
            'omega_req': params.omega_req,
            'coverage_ratio': params.omega_max / params.omega_req if params.omega_req > 0 else float('inf')
        },
        'bounds': params.bound_sources,
        'warnings': params.warnings,
        'recommendations': []
    }

    # Generate recommendations
    coverage_ratio = report['coverage']['coverage_ratio']

    if coverage_ratio < 1.5:
        report['recommendations'].append(
            "Frequency coverage is marginal. Consider decreasing dt or increasing N."
        )

    if params.a > 1.0:
        report['recommendations'].append(
            f"Large Bromwich shift a={params.a:.2f}. "
            "Verify operator stability or use smaller a_margin."
        )

    if params.N == 16384:
        report['recommendations'].append(
            "FFT size at maximum. For higher resolution, increase N_max."
        )

    if len(params.warnings) > 0:
        report['recommendations'].append(
            "Review warnings in params.warnings for potential issues."
        )

    return report
