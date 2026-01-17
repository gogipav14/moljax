"""
Spectral smoothing methods for Gibbs artifact reduction in NILT.

This module implements σ-factor windowing methods to reduce Gibbs phenomena
(ringing artifacts) that occur when inverting functions with discontinuities.

Key methods:
- Fejér summation: σ_k = 1 - k/N (linear taper)
- Lanczos σ-factor: σ_k = sinc(kπ/N) (preserves bandwidth better)
- Hamming window: 0.54 - 0.46*cos(2πk/N) (good sidelobe suppression)
- Raised cosine: smooth transition at edges

Trade-off: Smoothing reduces ringing but also reduces effective bandwidth,
which can smear sharp features. Use the mildest smoothing that achieves
acceptable artifact levels.

Reference: Gibbs phenomenon and sigma factors in Fourier series.
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class SmoothingMethod(Enum):
    """Available spectral smoothing methods."""
    NONE = 'none'
    FEJER = 'fejer'
    LANCZOS = 'lanczos'
    HAMMING = 'hamming'
    RAISED_COSINE = 'raised_cosine'
    EXPONENTIAL = 'exponential'


class SmoothingResult(NamedTuple):
    """Result of spectral smoothing."""
    F_smoothed: jnp.ndarray
    sigma_factors: jnp.ndarray
    method: str
    bandwidth_retention: float  # Fraction of original bandwidth preserved


# =============================================================================
# σ-Factor Functions
# =============================================================================

def fejer_sigma(N: int, dtype=jnp.float64) -> jnp.ndarray:
    """
    Fejér σ-factors: σ_k = 1 - |k|/N

    The simplest σ-factor, provides linear taper from 1 at k=0 to 0 at k=N.
    Equivalent to convolving with Fejér kernel (triangular window).

    Properties:
    - Always non-negative
    - Bandwidth retention: ~50% effective
    - Good for moderate Gibbs reduction

    Args:
        N: Number of frequency points
        dtype: Data type

    Returns:
        Array of σ-factors for k = 0, 1, ..., N-1
    """
    k = jnp.arange(N, dtype=dtype)
    # For FFT convention, k goes 0..N/2, then wraps to negative frequencies
    # We want |k| effect, so use min(k, N-k) for wrapped frequencies
    k_effective = jnp.minimum(k, N - k)
    sigma = 1.0 - k_effective / (N / 2)
    return jnp.maximum(sigma, 0.0)


def lanczos_sigma(N: int, order: int = 1, dtype=jnp.float64) -> jnp.ndarray:
    """
    Lanczos σ-factors: σ_k = sinc(k·π/N)^order

    Better bandwidth preservation than Fejér while still reducing Gibbs.
    Higher order gives stronger smoothing.

    Properties:
    - sinc(0) = 1, smooth decay
    - Bandwidth retention: ~75% (order=1), ~60% (order=2)
    - Excellent for functions with jump discontinuities

    Args:
        N: Number of frequency points
        order: Lanczos filter order (1, 2, or 3)
        dtype: Data type

    Returns:
        Array of σ-factors
    """
    k = jnp.arange(N, dtype=dtype)
    k_effective = jnp.minimum(k, N - k)

    # Normalized argument: kπ/N for k ∈ [0, N/2]
    # Avoid division by zero at k=0
    arg = k_effective * jnp.pi / (N / 2)

    # sinc(x) = sin(x)/x
    sinc_val = jnp.where(
        arg < 1e-10,
        1.0,
        jnp.sin(arg) / (arg + 1e-30)
    )

    return jnp.power(sinc_val, order)


def hamming_window(N: int, dtype=jnp.float64) -> jnp.ndarray:
    """
    Hamming window σ-factors: σ_k = 0.54 - 0.46·cos(2πk/N)

    Classic window with excellent sidelobe suppression (~43 dB).

    Properties:
    - Non-zero at endpoints (unlike Hann)
    - Bandwidth retention: ~60%
    - Good general-purpose smoothing

    Args:
        N: Number of frequency points
        dtype: Data type

    Returns:
        Array of σ-factors
    """
    k = jnp.arange(N, dtype=dtype)
    k_effective = jnp.minimum(k, N - k)

    # Normalized to [0, 1] range in effective frequency
    norm_k = k_effective / (N / 2)

    # Hamming formula (shifted for frequency domain)
    sigma = 0.54 + 0.46 * jnp.cos(jnp.pi * norm_k)

    return sigma


def raised_cosine_sigma(N: int, rolloff: float = 0.5, dtype=jnp.float64) -> jnp.ndarray:
    """
    Raised cosine σ-factors with adjustable rolloff.

    Provides smooth transition from passband to stopband.

    Properties:
    - rolloff=0: brick-wall (no smoothing)
    - rolloff=1: full cosine rolloff
    - Bandwidth retention: 1 - rolloff/2

    Args:
        N: Number of frequency points
        rolloff: Rolloff factor β ∈ [0, 1]
        dtype: Data type

    Returns:
        Array of σ-factors
    """
    k = jnp.arange(N, dtype=dtype)
    k_effective = jnp.minimum(k, N - k)

    # Normalize to [0, 1]
    f = k_effective / (N / 2)

    # Raised cosine regions
    f_low = (1 - rolloff) / 2
    f_high = (1 + rolloff) / 2

    sigma = jnp.where(
        f <= f_low,
        1.0,
        jnp.where(
            f >= f_high,
            0.0,
            0.5 * (1 + jnp.cos(jnp.pi * (f - f_low) / rolloff))
        )
    )

    return sigma


def exponential_sigma(N: int, alpha: float = 2.0, dtype=jnp.float64) -> jnp.ndarray:
    """
    Exponential σ-factors: σ_k = exp(-α·(k/N)²)

    Gaussian-like taper for very smooth transitions.

    Properties:
    - α controls decay rate (higher = more smoothing)
    - Preserves low frequencies very well
    - Bandwidth retention: depends on α

    Args:
        N: Number of frequency points
        alpha: Decay rate parameter
        dtype: Data type

    Returns:
        Array of σ-factors
    """
    k = jnp.arange(N, dtype=dtype)
    k_effective = jnp.minimum(k, N - k)

    # Normalized squared frequency
    f_sq = (k_effective / (N / 2)) ** 2

    return jnp.exp(-alpha * f_sq)


# =============================================================================
# Smoothing Application
# =============================================================================

def get_sigma_factors(
    N: int,
    method: SmoothingMethod | str = SmoothingMethod.LANCZOS,
    dtype=jnp.float64,
    **kwargs
) -> jnp.ndarray:
    """
    Get σ-factors for the specified smoothing method.

    Args:
        N: Number of frequency points
        method: Smoothing method (enum or string)
        dtype: Data type
        **kwargs: Method-specific parameters
            - lanczos: order (default 1)
            - raised_cosine: rolloff (default 0.5)
            - exponential: alpha (default 2.0)

    Returns:
        Array of σ-factors
    """
    if isinstance(method, str):
        method = SmoothingMethod(method.lower())

    if method == SmoothingMethod.NONE:
        return jnp.ones(N, dtype=dtype)
    elif method == SmoothingMethod.FEJER:
        return fejer_sigma(N, dtype)
    elif method == SmoothingMethod.LANCZOS:
        order = kwargs.get('order', 1)
        return lanczos_sigma(N, order, dtype)
    elif method == SmoothingMethod.HAMMING:
        return hamming_window(N, dtype)
    elif method == SmoothingMethod.RAISED_COSINE:
        rolloff = kwargs.get('rolloff', 0.5)
        return raised_cosine_sigma(N, rolloff, dtype)
    elif method == SmoothingMethod.EXPONENTIAL:
        alpha = kwargs.get('alpha', 2.0)
        return exponential_sigma(N, alpha, dtype)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")


def apply_spectral_smoothing(
    F_vals: jnp.ndarray,
    method: SmoothingMethod | str = SmoothingMethod.LANCZOS,
    **kwargs
) -> SmoothingResult:
    """
    Apply spectral smoothing (σ-factors) to frequency domain data.

    This multiplies F[k] by σ[k] before IFFT, which is equivalent
    to convolving f(t) with the corresponding kernel in time domain.

    Args:
        F_vals: Frequency domain values F[k] for k = 0, ..., N-1
        method: Smoothing method
        **kwargs: Method-specific parameters

    Returns:
        SmoothingResult with smoothed spectrum and diagnostics
    """
    N = len(F_vals)
    dtype = F_vals.dtype

    sigma = get_sigma_factors(N, method, dtype, **kwargs)
    F_smoothed = F_vals * sigma

    # Estimate bandwidth retention: energy ratio
    energy_original = jnp.sum(jnp.abs(F_vals) ** 2)
    energy_smoothed = jnp.sum(jnp.abs(F_smoothed) ** 2)
    bandwidth_retention = float(energy_smoothed / (energy_original + 1e-30))

    method_name = method.value if isinstance(method, SmoothingMethod) else method

    return SmoothingResult(
        F_smoothed=F_smoothed,
        sigma_factors=sigma,
        method=method_name,
        bandwidth_retention=bandwidth_retention
    )


# =============================================================================
# NILT Integration
# =============================================================================

def nilt_with_smoothing(
    F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    dt: float,
    N: int,
    a: float = 0.0,
    smoothing: SmoothingMethod | str = SmoothingMethod.LANCZOS,
    t_end: float | None = None,
    dtype=jnp.float64,
    return_diagnostics: bool = False,
    **smoothing_kwargs
):
    """
    NILT with spectral smoothing for Gibbs artifact reduction.

    This is a wrapper around the standard NILT that applies σ-factor
    windowing before the IFFT to reduce ringing artifacts.

    Args:
        F_eval: Laplace domain function F(s)
        dt: Time step
        N: Number of points (power of 2)
        a: Bromwich shift parameter
        smoothing: Smoothing method to apply
        t_end: End time (default: N*dt)
        dtype: Data type
        return_diagnostics: If True, include diagnostics
        **smoothing_kwargs: Method-specific smoothing parameters

    Returns:
        NILTResult or (NILTResult, dict) if return_diagnostics=True
    """
    from .nilt_fft import NILTResult

    T = N * dt / 2  # Period T = N*dt/2
    t_end_actual = t_end if t_end is not None else 2 * T

    # Create time grid
    t = jnp.linspace(0, 2 * T, N, endpoint=False, dtype=dtype)
    valid_mask = t <= t_end_actual
    n_valid = int(jnp.sum(valid_mask))

    # Create frequency grid: ω_k = k·Δω where Δω = π/T
    delta_omega = jnp.pi / T
    k = jnp.arange(N, dtype=dtype)
    omega = k * delta_omega

    # Evaluate F(s) at s = a + i·ω_k
    s = a + 1j * omega
    F_vals = F_eval(s).astype(jnp.complex128 if dtype == jnp.float64 else jnp.complex64)

    # Apply spectral smoothing
    smoothing_result = apply_spectral_smoothing(F_vals, smoothing, **smoothing_kwargs)
    F_smoothed = smoothing_result.F_smoothed

    # Apply exponential shift: multiply by exp(a·t_k) after IFFT
    # First do IFFT
    f_shifted = jnp.fft.ifft(F_smoothed)

    # Scale and shift
    f_values = jnp.real(f_shifted) * (delta_omega / jnp.pi) * jnp.exp(a * t)

    # Create result
    result = NILTResult(
        t=t[:n_valid],
        f=f_values[:n_valid],
        dt=dt,
        N=N,
        a=a,
        T=T,
        diagnostics={
            'smoothing_method': smoothing_result.method,
            'bandwidth_retention': smoothing_result.bandwidth_retention,
        } if return_diagnostics else None
    )

    if return_diagnostics:
        return result, {
            'smoothing': smoothing_result,
            'sigma_factors': smoothing_result.sigma_factors,
        }

    return result


# =============================================================================
# Utility: Compare Smoothing Methods
# =============================================================================

def compare_smoothing_methods(
    F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    f_exact: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    dt: float = 0.01,
    N: int = 512,
    a: float = 0.0,
    t_eval: jnp.ndarray | None = None,
    dtype=jnp.float64
) -> dict:
    """
    Compare different smoothing methods on the same problem.

    Args:
        F_eval: Laplace domain function
        f_exact: Exact time domain solution (optional, for error calculation)
        dt: Time step
        N: Number of points
        a: Bromwich shift
        t_eval: Times to evaluate at (default: full grid)
        dtype: Data type

    Returns:
        Dictionary with results for each method
    """
    methods = [
        SmoothingMethod.NONE,
        SmoothingMethod.FEJER,
        SmoothingMethod.LANCZOS,
        SmoothingMethod.HAMMING,
        SmoothingMethod.RAISED_COSINE,
        SmoothingMethod.EXPONENTIAL,
    ]

    results = {}

    for method in methods:
        result, diag = nilt_with_smoothing(
            F_eval, dt, N, a,
            smoothing=method,
            dtype=dtype,
            return_diagnostics=True
        )

        entry = {
            'f': result.f,
            't': result.t,
            'bandwidth_retention': diag['smoothing'].bandwidth_retention,
        }

        if f_exact is not None:
            f_true = f_exact(result.t)
            error = jnp.abs(result.f - f_true)
            entry['max_error'] = float(jnp.max(error))
            entry['rms_error'] = float(jnp.sqrt(jnp.mean(error**2)))
            entry['f_exact'] = f_true

        results[method.value] = entry

    return results


def print_smoothing_comparison(comparison: dict):
    """Print formatted comparison table."""
    print("\nSpectral Smoothing Comparison")
    print("=" * 70)
    print(f"{'Method':<15} | {'Bandwidth %':>12} | {'Max Error':>12} | {'RMS Error':>12}")
    print("-" * 70)

    for method, data in comparison.items():
        bw = data['bandwidth_retention'] * 100
        max_err = data.get('max_error', float('nan'))
        rms_err = data.get('rms_error', float('nan'))

        if not jnp.isnan(max_err):
            print(f"{method:<15} | {bw:>11.1f}% | {max_err:>12.2e} | {rms_err:>12.2e}")
        else:
            print(f"{method:<15} | {bw:>11.1f}% | {'N/A':>12} | {'N/A':>12}")

    print("=" * 70)
