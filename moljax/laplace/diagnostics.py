"""
NILT diagnostics for assessing inversion quality.

Provides metrics for:
- Frequency coverage adequacy
- Truncation/ringing artifacts
- Wraparound contamination
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp


class NILTDiagnostics(NamedTuple):
    """Comprehensive NILT quality diagnostics."""
    frequency_coverage: float  # |F(a+i*ω_max)| / |F(a)|
    frequency_slope: float  # Rate of decay at ω_max
    ringing_metric: float  # Oscillatory energy in tail
    wraparound_metric: float  # Tail vs head comparison
    energy_ratio: float  # Energy in [0,T] vs [T,2T]
    quality_score: float  # Overall quality (0-1, higher is better)
    warnings: list[str]


# =============================================================================
# Frequency domain diagnostics
# =============================================================================

def frequency_coverage_metric(
    F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    a: float,
    omega_max: float,
    n_samples: int = 10
) -> tuple[float, float]:
    """
    Assess frequency coverage by examining |F(s)| decay.

    Good coverage requires |F(a + i*ω_max)| << |F(a)|,
    indicating the transform has decayed sufficiently at the cutoff.

    Args:
        F_eval: Laplace domain function
        a: Bromwich shift
        omega_max: Maximum frequency (π/dt)
        n_samples: Number of samples for slope estimation

    Returns:
        (magnitude_ratio, slope) where:
        - magnitude_ratio = |F(a+i*ω_max)| / |F(a)|
        - slope = d/dω log|F(a+i*ω)| at ω_max (should be negative)
    """
    # Sample frequencies
    omega = jnp.linspace(0, omega_max, n_samples)
    s = a + 1j * omega

    F_vals = F_eval(s)
    F_mag = jnp.abs(F_vals)

    # Magnitude ratio at endpoints
    F_at_zero = F_mag[0] + 1e-20
    F_at_max = F_mag[-1]
    magnitude_ratio = float(F_at_max / F_at_zero)

    # Slope estimation via finite difference of log magnitude
    log_F = jnp.log(F_mag + 1e-20)
    d_omega = omega_max / (n_samples - 1)

    # Slope at high frequency end
    slope = float((log_F[-1] - log_F[-2]) / d_omega)

    return magnitude_ratio, slope


def estimate_truncation_error(
    F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    a: float,
    omega_max: float
) -> float:
    """
    Estimate the error from truncating the inverse integral at ω_max.

    The truncation error is approximately:
        E_trunc ~ |∫_{ω_max}^∞ F(a+iω) dω| ~ |F(a+iω_max)| / |slope|

    Args:
        F_eval: Laplace domain function
        a: Bromwich shift
        omega_max: Cutoff frequency

    Returns:
        Estimated truncation error magnitude
    """
    # Sample at and slightly below cutoff
    omega_sample = jnp.array([omega_max * 0.95, omega_max])
    s = a + 1j * omega_sample
    F_vals = F_eval(s)

    F_at_max = jnp.abs(F_vals[1])
    dF = jnp.abs(F_vals[1]) - jnp.abs(F_vals[0])
    d_omega = 0.05 * omega_max

    # Slope magnitude
    slope_mag = jnp.abs(dF / d_omega) + 1e-20

    # Truncation error estimate: remaining area under curve
    # Assuming exponential decay beyond ω_max
    error_est = float(F_at_max / slope_mag)

    return error_est


# =============================================================================
# Time domain diagnostics
# =============================================================================

def ringing_metric(
    f: jnp.ndarray,
    tail_fraction: float = 0.2
) -> float:
    """
    Quantify oscillatory energy near end of interval.

    Ringing artifacts from truncation appear as oscillations
    in the tail of f(t). This metric measures the RMS of
    the signal in the tail region relative to the overall signal.

    Args:
        f: Inverse transform values f(t)
        tail_fraction: Fraction of interval to examine at end

    Returns:
        Ringing metric (lower is better, should be << 1)
    """
    n = len(f)
    tail_start = int(n * (1 - tail_fraction))

    tail_vals = f[tail_start:]

    # RMS of tail
    tail_rms = jnp.sqrt(jnp.mean(tail_vals**2))

    # RMS of full signal
    full_rms = jnp.sqrt(jnp.mean(f**2)) + 1e-20

    return float(tail_rms / full_rms)


def wraparound_metric(f: jnp.ndarray) -> float:
    """
    Detect periodic wraparound contamination.

    In FFT-based NILT, the result is periodic with period 2T.
    If f(t) hasn't decayed by t=T, the signal "wraps around"
    and contaminates the [0,T] interval from the [T,2T] part.

    This metric compares early and late values.

    Args:
        f: Inverse transform values (length N, covering [0, 2T))

    Returns:
        Wraparound metric (should be close to 0)
    """
    n = len(f)
    half_n = n // 2

    # Energy in first half [0, T)
    first_half_energy = jnp.sum(f[:half_n]**2)

    # Energy in second half [T, 2T)
    second_half_energy = jnp.sum(f[half_n:]**2)

    total = first_half_energy + second_half_energy + 1e-20

    # Ratio of second half energy to total
    # Should be small if f decays properly
    return float(second_half_energy / total)


def energy_distribution(
    f: jnp.ndarray,
    t: jnp.ndarray,
    t_end: float
) -> tuple[float, float]:
    """
    Compute energy distribution relative to target interval.

    Args:
        f: Inverse transform values
        t: Time points
        t_end: Target end time (should be < T)

    Returns:
        (energy_in_target, energy_outside) as fractions
    """
    mask_in = t <= t_end
    mask_out = t > t_end

    energy_in = jnp.sum(f[mask_in]**2)
    energy_out = jnp.sum(f[mask_out]**2)
    total = energy_in + energy_out + 1e-20

    return float(energy_in / total), float(energy_out / total)


def oscillation_count(
    f: jnp.ndarray,
    threshold: float = 0.0
) -> int:
    """
    Count zero crossings in the signal.

    High oscillation count in tail may indicate ringing.

    Args:
        f: Signal values
        threshold: Threshold for crossing detection

    Returns:
        Number of crossings
    """
    above = f > threshold
    crossings = jnp.sum(jnp.abs(jnp.diff(above.astype(jnp.int32))))
    return int(crossings)


# =============================================================================
# Comprehensive diagnostics
# =============================================================================

def compute_nilt_diagnostics(
    F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    f: jnp.ndarray,
    t: jnp.ndarray,
    a: float,
    T: float,
    t_end: float | None = None
) -> NILTDiagnostics:
    """
    Compute comprehensive NILT quality diagnostics.

    Args:
        F_eval: Laplace domain function
        f: Computed inverse transform values
        t: Time points
        a: Bromwich shift used
        T: Half-period used
        t_end: Target end time (defaults to T)

    Returns:
        NILTDiagnostics with quality metrics
    """
    if t_end is None:
        t_end = T

    warnings = []

    # Frequency domain metrics
    omega_max = jnp.pi * len(f) / (2 * T)
    freq_ratio, freq_slope = frequency_coverage_metric(
        F_eval, a, float(omega_max)
    )

    if freq_ratio > 0.1:
        warnings.append(
            f"High frequency content at cutoff: |F(ω_max)|/|F(0)| = {freq_ratio:.3f}"
        )

    if freq_slope > -0.1:
        warnings.append(
            f"Slow frequency decay: slope = {freq_slope:.3f} at ω_max"
        )

    # Time domain metrics
    ring_metric = ringing_metric(f)
    wrap_metric = wraparound_metric(f)

    if ring_metric > 0.3:
        warnings.append(f"Significant tail ringing: {ring_metric:.3f}")

    if wrap_metric > 0.2:
        warnings.append(f"Wraparound contamination: {wrap_metric:.3f}")

    # Energy distribution
    energy_in, energy_out = energy_distribution(f, t, t_end)

    if energy_out > 0.3:
        warnings.append(
            f"{100*energy_out:.1f}% of energy outside target interval"
        )

    # Quality score (heuristic combination)
    # Higher is better, range [0, 1]
    quality_score = 1.0

    # Penalize high frequency ratio
    quality_score *= jnp.exp(-10 * freq_ratio)

    # Penalize ringing
    quality_score *= jnp.exp(-3 * ring_metric)

    # Penalize wraparound
    quality_score *= jnp.exp(-5 * wrap_metric)

    return NILTDiagnostics(
        frequency_coverage=freq_ratio,
        frequency_slope=freq_slope,
        ringing_metric=ring_metric,
        wraparound_metric=wrap_metric,
        energy_ratio=energy_in,
        quality_score=float(quality_score),
        warnings=warnings
    )


def compare_to_reference(
    f_computed: jnp.ndarray,
    f_reference: jnp.ndarray,
    t: jnp.ndarray,
    t_end: float | None = None
) -> dict:
    """
    Compare computed inverse to reference solution.

    Args:
        f_computed: NILT result
        f_reference: Analytic or high-accuracy reference
        t: Time points
        t_end: End of comparison interval

    Returns:
        Dict with error metrics
    """
    if t_end is not None:
        mask = t <= t_end
        f_comp = f_computed[mask]
        f_ref = f_reference[mask]
        t_used = t[mask]
    else:
        f_comp = f_computed
        f_ref = f_reference
        t_used = t

    # Error metrics
    abs_error = jnp.abs(f_comp - f_ref)
    rel_error = abs_error / (jnp.abs(f_ref) + 1e-10)

    # Norms
    l2_error = jnp.sqrt(jnp.mean(abs_error**2))
    linf_error = jnp.max(abs_error)
    l2_rel = l2_error / (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)

    # Where is max error?
    max_idx = jnp.argmax(abs_error)
    t_max_error = float(t_used[max_idx])

    return {
        'l2_error': float(l2_error),
        'linf_error': float(linf_error),
        'l2_relative': float(l2_rel),
        'mean_abs_error': float(jnp.mean(abs_error)),
        'max_rel_error': float(jnp.max(rel_error)),
        't_max_error': t_max_error,
        'n_points': len(f_comp)
    }


# =============================================================================
# Parameter sensitivity analysis
# =============================================================================

def sensitivity_analysis(
    F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    f_reference: Callable[[jnp.ndarray], jnp.ndarray],
    base_dt: float,
    base_N: int,
    base_a: float,
    t_end: float,
    perturbation: float = 0.2
) -> dict:
    """
    Analyze sensitivity of NILT to parameter variations.

    Args:
        F_eval: Laplace domain function
        f_reference: Reference time-domain function
        base_dt: Baseline time step
        base_N: Baseline FFT size
        base_a: Baseline Bromwich shift
        t_end: End time for comparison
        perturbation: Fractional perturbation to apply

    Returns:
        Dict with sensitivity information for each parameter
    """
    from .nilt_fft import nilt_fft_uniform

    def compute_error(dt, N, a):
        result = nilt_fft_uniform(F_eval, dt=dt, N=N, a=a)
        f_ref = f_reference(result.t)
        mask = result.t <= t_end
        error = jnp.sqrt(jnp.mean((result.f[mask] - f_ref[mask])**2))
        return float(error)

    # Baseline error
    base_error = compute_error(base_dt, base_N, base_a)

    # Perturb dt
    dt_plus = base_dt * (1 + perturbation)
    dt_minus = base_dt * (1 - perturbation)
    error_dt_plus = compute_error(dt_plus, base_N, base_a)
    error_dt_minus = compute_error(dt_minus, base_N, base_a)

    # Perturb a
    a_plus = base_a * (1 + perturbation) if base_a > 0 else base_a + 0.1
    a_minus = max(0, base_a * (1 - perturbation))
    error_a_plus = compute_error(base_dt, base_N, a_plus)
    error_a_minus = compute_error(base_dt, base_N, a_minus)

    # Perturb N (use powers of 2)
    N_double = min(base_N * 2, 32768)
    N_half = max(base_N // 2, 64)
    error_N_double = compute_error(base_dt, N_double, base_a)
    error_N_half = compute_error(base_dt, N_half, base_a)

    return {
        'base_error': base_error,
        'dt_sensitivity': {
            'plus': error_dt_plus,
            'minus': error_dt_minus,
            'ratio_plus': error_dt_plus / (base_error + 1e-20),
            'ratio_minus': error_dt_minus / (base_error + 1e-20)
        },
        'a_sensitivity': {
            'plus': error_a_plus,
            'minus': error_a_minus,
            'ratio_plus': error_a_plus / (base_error + 1e-20),
            'ratio_minus': error_a_minus / (base_error + 1e-20)
        },
        'N_sensitivity': {
            'double': error_N_double,
            'half': error_N_half,
            'ratio_double': error_N_double / (base_error + 1e-20),
            'ratio_half': error_N_half / (base_error + 1e-20)
        }
    }
