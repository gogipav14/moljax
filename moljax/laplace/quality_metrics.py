"""
Unified quality metrics for NILT autotuning.

Quality is decided on two frequency-domain sensors that nilt_fft_uniform
reports in its diagnostics:

1. Bandwidth: band_edge_ratio = |F(a + i pi/dt)| / max_k |F(a + i omega_k)|
   and tail_energy_fraction, the RMS share of the top tenth of the resolved
   band. Both measure what truncating the Bromwich integral at pi/dt
   discards.
2. Wraparound: tail_ratio, the energy of the damped inverse f(t) e^{-a t}
   beyond t_end relative to [0, t_end], which the periodic extension folds
   back onto the valid interval.

eps_Im (imaginary leakage) and eps_sym (Hermitian symmetry) are still
computed as numerical-health indicators, but they are reported rather than
decided on: the spectrum is mirrored into exact Hermitian symmetry before
the ifft, so both are zero by construction and cannot signal a badly
resolved transform.

Levels (band_edge_ratio on 1/(s+1) at N = 256 in parentheses):
- excellent: band edge below 1% (sin t at dt = 0.05: 9e-6)
- good: below 10% (dt = 0.05: 0.016, RMS error 3.9%)
- acceptable: below 30% (dt = 0.5: 0.157, RMS error 16%)
- poor: above (dt = 2: 0.537, RMS error 78%)

The thresholds match the balanced policy of
moljax.laplace.classify_quality_tier.

Reference:
- Dubner & Abate, "Numerical Inversion of Laplace Transforms" (1968)
- Weideman & Trefethen, "Parabolic and Hyperbolic Contours" (2007)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, NamedTuple

import jax.numpy as jnp

from .nilt_fft import compute_bandwidth_sensors


class QualityLevel(Enum):
    """Quality level classification."""
    EXCELLENT = 'excellent'  # band edge < 1%
    GOOD = 'good'            # band edge < 10%
    ACCEPTABLE = 'acceptable'  # band edge < 30%
    POOR = 'poor'            # band edge >= 30% or wraparound
    FAILED = 'failed'        # Numerical issues


class RetuningAction(Enum):
    """Recommended retuning actions."""
    NONE = 'none'
    REDUCE_DT = 'reduce_dt'         # Bandwidth insufficient
    INCREASE_T = 'increase_T'       # Wraparound detected
    INCREASE_N = 'increase_N'       # Resolution insufficient
    INCREASE_A = 'increase_a'       # Shift too small
    APPLY_PROJECTION = 'apply_projection'  # Hermitian projection
    APPLY_SMOOTHING = 'apply_smoothing'    # Gibbs artifacts


class QualityMetrics(NamedTuple):
    """Comprehensive quality metrics for NILT."""
    # Decision sensors
    band_edge_ratio: float         # |F| at the band edge relative to its peak (nan without F_vals)
    tail_energy_fraction: float    # RMS share of the top tenth of the band (nan without F_vals)
    tail_ratio: float              # Wraparound indicator

    # Numerical health
    eps_im: float                  # Imaginary leakage (full grid)
    eps_im_valid: float            # Imaginary leakage in [0, t_end]
    eps_sym: float                 # Hermitian symmetry residual

    # Localization of the leakage
    r_early: float                 # Early-time leakage ratio
    r_late: float                  # Late-time leakage ratio

    # Percentile metrics
    leakage_p50: float             # Median leakage
    leakage_p95: float             # 95th percentile
    leakage_p99: float             # 99th percentile

    # Derived indicators
    quality_level: str             # QualityLevel enum value
    dominant_issue: str            # 'bandwidth', 'wraparound', 'none'
    spike_detected: bool           # p95 >> p50 (informational)

    # Recommended actions
    actions: list[str]             # List of RetuningAction values


class AutotunerFeedback(NamedTuple):
    """Feedback for autotuner iteration."""
    should_retune: bool
    priority_action: str           # Most important action
    all_actions: list[str]         # All recommended actions
    confidence: float              # Confidence in recommendation (0-1)
    reason: str                    # Human-readable explanation
    suggested_params: dict[str, float]  # Suggested parameter changes


# =============================================================================
# Core Quality Computation
# =============================================================================

def compute_eps_im(
    ifft_result: jnp.ndarray,
    t: jnp.ndarray | None = None,
    t_end: float | None = None
) -> tuple[float, dict[str, float]]:
    """
    Compute imaginary leakage metrics.

    The imaginary part of IFFT should be zero for real-valued f(t).
    Non-zero imaginary indicates numerical error or parameter mismatch.

    Args:
        ifft_result: Complex IFFT output
        t: Time grid (optional, for localization)
        t_end: End time of interest

    Returns:
        (eps_im, localization_dict)
    """
    real_part = jnp.real(ifft_result)
    imag_part = jnp.imag(ifft_result)

    # Compute full-grid metric
    norm_real = jnp.sqrt(jnp.mean(real_part ** 2))
    norm_imag = jnp.sqrt(jnp.mean(imag_part ** 2))
    eps_im = float(norm_imag / (norm_real + 1e-14))

    # Localization if time grid provided
    localization = {
        'eps_im_valid': eps_im,
        'r_early': 0.0,
        'r_late': 0.0,
        'tail_ratio': 0.0,
        'leakage_p50': 0.0,
        'leakage_p95': 0.0,
        'leakage_p99': 0.0,
        'dominant': 'none'
    }

    if t is not None and len(t) > 10:
        N = len(t)

        # Pointwise leakage
        pointwise = jnp.abs(imag_part) / (jnp.abs(real_part) + 1e-14)

        # Percentiles
        localization['leakage_p50'] = float(jnp.percentile(pointwise, 50))
        localization['leakage_p95'] = float(jnp.percentile(pointwise, 95))
        localization['leakage_p99'] = float(jnp.percentile(pointwise, 99))

        # Region-based analysis
        early_idx = N // 4
        late_idx = 3 * N // 4

        early_leakage = jnp.sqrt(jnp.mean(imag_part[:early_idx] ** 2))
        mid_leakage = jnp.sqrt(jnp.mean(imag_part[early_idx:late_idx] ** 2))
        late_leakage = jnp.sqrt(jnp.mean(imag_part[late_idx:] ** 2))

        total_leakage = early_leakage + mid_leakage + late_leakage + 1e-14

        localization['r_early'] = float(early_leakage / total_leakage)
        localization['r_late'] = float(late_leakage / total_leakage)
        localization['tail_ratio'] = float(late_leakage / (norm_real + 1e-14))

        # Determine dominant region
        if localization['r_early'] > 0.5:
            localization['dominant'] = 'early'
        elif localization['r_late'] > 0.5:
            localization['dominant'] = 'late'
        else:
            localization['dominant'] = 'mid'

        # Valid region metric
        if t_end is not None:
            valid_mask = t <= t_end
            if jnp.sum(valid_mask) > 0:
                real_valid = real_part[valid_mask]
                imag_valid = imag_part[valid_mask]
                norm_real_valid = jnp.sqrt(jnp.mean(real_valid ** 2))
                norm_imag_valid = jnp.sqrt(jnp.mean(imag_valid ** 2))
                localization['eps_im_valid'] = float(
                    norm_imag_valid / (norm_real_valid + 1e-14)
                )

    return eps_im, localization


def compute_eps_sym(F_vals: jnp.ndarray) -> float:
    """
    Compute Hermitian symmetry residual.

    For real f(t), F(s) should satisfy F[N-k] = conj(F[k]).
    This measures violation of that symmetry.

    Args:
        F_vals: Frequency-domain values

    Returns:
        eps_sym: Symmetry residual (zero for the mirrored spectrum)
    """
    N = len(F_vals)
    if N < 4:
        return 0.0

    # Compare F[k] with conj(F[N-k])
    k_range = jnp.arange(1, N // 2)
    F_pos = F_vals[k_range]
    F_neg = jnp.conj(F_vals[N - k_range])

    diff = F_pos - F_neg
    norm_diff = jnp.sqrt(jnp.mean(jnp.abs(diff) ** 2))
    norm_F = jnp.sqrt(jnp.mean(jnp.abs(F_pos) ** 2))

    return float(norm_diff / (norm_F + 1e-14))


# =============================================================================
# Quality Classification
# =============================================================================

_DEFAULT_THRESHOLDS = {
    'band_edge_excellent': 0.01,
    'band_edge_good': 0.10,
    'band_edge_acceptable': 0.30,
    'tail_energy_good': 0.08,
    'tail_energy_acceptable': 0.18,
    'tail_ratio_good': 0.09,
    'tail_ratio_acceptable': 0.12,
}


def classify_quality(
    band_edge_ratio: float,
    tail_ratio: float,
    tail_energy_fraction: float = 0.0,
    thresholds: dict[str, float] | None = None
) -> tuple[QualityLevel, list[RetuningAction], str]:
    """
    Classify quality and determine retuning actions from the two sensors.

    A sensor at or above its ``*_acceptable`` threshold is poor, at or above
    its ``*_good`` threshold acceptable, otherwise good; the level is the
    worst of the bandwidth pair (band_edge_ratio, tail_energy_fraction) and
    the wraparound sensor (tail_ratio), and excellent when everything is good
    and the band edge is below ``band_edge_excellent``.

    A NaN band_edge_ratio and tail_energy_fraction *together* is the
    documented "no F samples available" case (see assess_nilt_quality) and
    counts as no evidence for bandwidth, deferring entirely to tail_ratio.
    tail_ratio itself is always computed from the time-domain result, so a
    non-finite tail_ratio, or a non-finite bandwidth sensor whose counterpart
    is finite, means the underlying transform is not finite (F_eval returned
    NaN/inf somewhere on the contour): that is QualityLevel.FAILED, never a
    passing grade, since every ``>=`` comparison against NaN is False and
    would otherwise fall through to good.

    Args:
        band_edge_ratio: |F| at the band edge relative to its peak
        tail_ratio: Energy of the damped inverse beyond t_end
        tail_energy_fraction: RMS share of the top tenth of the band
        thresholds: Overrides for the keys of the default thresholds

    Returns:
        (quality_level, recommended_actions, dominant_issue)
    """
    th = dict(_DEFAULT_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    no_bandwidth_evidence = math.isnan(band_edge_ratio) and math.isnan(tail_energy_fraction)

    non_finite = []
    if not no_bandwidth_evidence:
        if not math.isfinite(band_edge_ratio):
            non_finite.append('band_edge_ratio')
        if not math.isfinite(tail_energy_fraction):
            non_finite.append('tail_energy_fraction')
    if not math.isfinite(tail_ratio):
        non_finite.append('tail_ratio')

    if non_finite:
        return QualityLevel.FAILED, [], f"non-finite sensor(s): {', '.join(non_finite)}"

    def rank(value: float, good: float, acceptable: float) -> int:
        if value >= acceptable:
            return 3
        if value >= good:
            return 2
        return 1

    if no_bandwidth_evidence:
        bandwidth_rank = 1
    else:
        bandwidth_rank = max(
            rank(band_edge_ratio, th['band_edge_good'], th['band_edge_acceptable']),
            rank(tail_energy_fraction, th['tail_energy_good'], th['tail_energy_acceptable']),
        )
    wraparound_rank = rank(tail_ratio, th['tail_ratio_good'], th['tail_ratio_acceptable'])
    worst = max(bandwidth_rank, wraparound_rank)

    if worst == 1 and band_edge_ratio < th['band_edge_excellent']:
        level = QualityLevel.EXCELLENT
    else:
        level = [QualityLevel.GOOD, QualityLevel.ACCEPTABLE, QualityLevel.POOR][worst - 1]

    actions = []
    dominant = 'none'
    if bandwidth_rank >= 2:
        actions.append(RetuningAction.REDUCE_DT)
        dominant = 'bandwidth'
    if wraparound_rank >= 2:
        actions.append(RetuningAction.INCREASE_T)
        if wraparound_rank > bandwidth_rank:
            dominant = 'wraparound'

    return level, actions, dominant


# =============================================================================
# Comprehensive Quality Assessment
# =============================================================================

def assess_nilt_quality(
    ifft_result: jnp.ndarray,
    F_vals: jnp.ndarray | None = None,
    t: jnp.ndarray | None = None,
    t_end: float | None = None,
    params: dict[str, Any] | None = None
) -> QualityMetrics:
    """
    Comprehensive quality assessment for NILT result.

    The bandwidth sensors need the sampled spectrum; without F_vals they
    are NaN and only the wraparound sensor decides.

    Args:
        ifft_result: Complex IFFT output
        F_vals: Full frequency-domain values (length N, Hermitian order)
        t: Time grid
        t_end: End time of interest
        params: NILT parameters (dt, N, a, T)

    Returns:
        QualityMetrics with all quality indicators
    """
    # Compute imaginary leakage
    eps_im, localization = compute_eps_im(ifft_result, t, t_end)

    if F_vals is not None:
        eps_sym = compute_eps_sym(F_vals)
        band_edge_ratio, tail_energy_fraction = compute_bandwidth_sensors(F_vals[:len(F_vals) // 2 + 1])
    else:
        eps_sym = 0.0
        band_edge_ratio = tail_energy_fraction = float('nan')

    level, actions, dominant = classify_quality(
        band_edge_ratio, localization['tail_ratio'], tail_energy_fraction
    )

    p50 = localization['leakage_p50']
    p95 = localization['leakage_p95']
    spike_detected = (p95 / (p50 + 1e-14)) > 5.0

    return QualityMetrics(
        band_edge_ratio=band_edge_ratio,
        tail_energy_fraction=tail_energy_fraction,
        tail_ratio=localization['tail_ratio'],
        eps_im=eps_im,
        eps_im_valid=localization['eps_im_valid'],
        eps_sym=eps_sym,
        r_early=localization['r_early'],
        r_late=localization['r_late'],
        leakage_p50=p50,
        leakage_p95=p95,
        leakage_p99=localization['leakage_p99'],
        quality_level=level.value,
        dominant_issue=dominant,
        spike_detected=spike_detected,
        actions=[a.value for a in actions]
    )


# =============================================================================
# Autotuner Feedback Loop
# =============================================================================

def generate_autotuner_feedback(
    metrics: QualityMetrics,
    current_params: dict[str, float],
    iteration: int = 0,
    max_iterations: int = 3
) -> AutotunerFeedback:
    """
    Generate feedback for autotuner based on quality metrics.

    Args:
        metrics: Quality metrics from assessment
        current_params: Current NILT parameters {dt, N, a, T}
        iteration: Current iteration number
        max_iterations: Maximum allowed iterations

    Returns:
        AutotunerFeedback with retuning recommendations
    """
    # Check if retuning needed
    should_retune = metrics.quality_level in ['poor', 'acceptable']

    if iteration >= max_iterations:
        should_retune = False
        reason = f"Max iterations ({max_iterations}) reached"
        priority_action = 'none'
        suggested_params = {}
        confidence = 0.5
    elif metrics.quality_level == 'excellent':
        should_retune = False
        reason = f"Excellent quality (band_edge_ratio = {metrics.band_edge_ratio:.2e})"
        priority_action = 'none'
        suggested_params = {}
        confidence = 1.0
    elif metrics.quality_level == 'good':
        should_retune = False
        reason = f"Good quality (band_edge_ratio = {metrics.band_edge_ratio:.3f})"
        priority_action = 'none'
        suggested_params = {}
        confidence = 0.9
    else:
        # Determine priority action and parameter adjustments
        priority_action = metrics.actions[0] if metrics.actions else 'none'
        suggested_params = {}

        dt = current_params.get('dt', 0.01)
        N = current_params.get('N', 256)
        T = current_params.get('T', N * dt / 2)

        if metrics.dominant_issue == 'bandwidth':
            # Reduce dt to increase bandwidth
            suggested_params['dt'] = dt * 0.5
            suggested_params['N'] = N * 2  # Keep same T
            reason = (f"Bandwidth issue detected (band_edge_ratio = {metrics.band_edge_ratio:.3f}, "
                      f"tail_energy_fraction = {metrics.tail_energy_fraction:.3f})")
            confidence = 0.85
        elif metrics.dominant_issue == 'wraparound':
            # Increase T to reduce wraparound
            suggested_params['T'] = T * 2.0
            suggested_params['N'] = N * 2  # Double N to maintain dt
            reason = f"Wraparound detected (tail_ratio = {metrics.tail_ratio:.2%})"
            confidence = 0.85
        else:
            # General poor quality - try increasing N
            suggested_params['N'] = min(N * 2, 8192)
            reason = f"Poor quality ({metrics.quality_level})"
            confidence = 0.7

    return AutotunerFeedback(
        should_retune=should_retune,
        priority_action=priority_action,
        all_actions=metrics.actions,
        confidence=confidence,
        reason=reason,
        suggested_params=suggested_params
    )


# =============================================================================
# Utility Functions
# =============================================================================

def print_quality_report(metrics: QualityMetrics, feedback: AutotunerFeedback) -> None:
    """Print a formatted quality report."""
    print("=" * 60)
    print("NILT Quality Assessment Report")
    print("=" * 60)
    print(f"Quality Level:     {metrics.quality_level.upper()}")
    print(f"band_edge_ratio:   {metrics.band_edge_ratio:.4f}")
    print(f"tail_energy_frac:  {metrics.tail_energy_fraction:.4f}")
    print(f"tail_ratio:        {metrics.tail_ratio:.4%}")
    print("-" * 60)
    print(f"ε_Im (full):       {metrics.eps_im:.4%}")
    print(f"ε_Im (valid):      {metrics.eps_im_valid:.4%}")
    print(f"ε_sym:             {metrics.eps_sym:.4f}")
    print(f"r_early:           {metrics.r_early:.3f}")
    print(f"r_late:            {metrics.r_late:.3f}")
    print(f"Spike detected:    {metrics.spike_detected}")
    print(f"Dominant issue:    {metrics.dominant_issue}")
    print("-" * 60)
    print(f"Recommended actions: {metrics.actions}")
    print("=" * 60)
    print("Autotuner Feedback:")
    print(f"  Should retune:   {feedback.should_retune}")
    print(f"  Priority action: {feedback.priority_action}")
    print(f"  Confidence:      {feedback.confidence:.0%}")
    print(f"  Reason:          {feedback.reason}")
    if feedback.suggested_params:
        print(f"  Suggested params: {feedback.suggested_params}")
    print("=" * 60)


def quality_meets_threshold(
    metrics: QualityMetrics,
    threshold: str = 'good'
) -> bool:
    """Check if quality meets a given threshold."""
    levels = ['excellent', 'good', 'acceptable', 'poor', 'failed']
    current_idx = levels.index(metrics.quality_level)
    threshold_idx = levels.index(threshold)
    return current_idx <= threshold_idx
