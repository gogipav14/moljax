"""
Unified quality metrics for NILT autotuning.

This module consolidates quality assessment into a single framework:
1. ε_Im (imaginary leakage) - primary quality indicator
2. ε_sym (Hermitian symmetry) - numerical health indicator
3. CFL-style endpoint conditions - physical constraints
4. Automatic feedback for retuning decisions

The quality metrics form a hierarchy:
- ε_Im < 1% → excellent (direct use)
- ε_Im < 5% → good (may benefit from projection)
- ε_Im < 10% → acceptable (projection recommended)
- ε_Im > 10% → poor (retuning required)

Reference:
- Dubner & Abate, "Numerical Inversion of Laplace Transforms" (1968)
- Weideman & Trefethen, "Parabolic and Hyperbolic Contours" (2007)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Any

import jax
import jax.numpy as jnp


class QualityLevel(Enum):
    """Quality level classification."""
    EXCELLENT = 'excellent'  # ε_Im < 1%
    GOOD = 'good'            # ε_Im < 5%
    ACCEPTABLE = 'acceptable'  # ε_Im < 10%
    POOR = 'poor'            # ε_Im > 10%
    FAILED = 'failed'        # Numerical issues


class RetuningAction(Enum):
    """Recommended retuning actions."""
    NONE = 'none'
    REDUCE_DT = 'reduce_dt'         # Bandwidth insufficient
    INCREASE_T = 'increase_T'       # Wraparound detected
    INCREASE_N = 'increase_N'       # Resolution insufficient
    INCREASE_A = 'increase_a'       # Shift too small
    APPLY_PROJECTION = 'apply_projection'  # Spike detected
    APPLY_SMOOTHING = 'apply_smoothing'    # Gibbs artifacts


class QualityMetrics(NamedTuple):
    """Comprehensive quality metrics for NILT."""
    # Primary metrics
    eps_im: float                  # Imaginary leakage (full grid)
    eps_im_valid: float            # Imaginary leakage in [0, t_end]
    eps_sym: float                 # Hermitian symmetry residual

    # Localization metrics
    r_early: float                 # Early-time leakage ratio
    r_late: float                  # Late-time leakage ratio
    tail_ratio: float              # Wraparound indicator

    # Percentile metrics
    leakage_p50: float             # Median leakage
    leakage_p95: float             # 95th percentile
    leakage_p99: float             # 99th percentile

    # Derived indicators
    quality_level: str             # QualityLevel enum value
    dominant_issue: str            # 'bandwidth', 'wraparound', 'spike', 'none'
    spike_detected: bool           # p95 >> p50

    # Recommended actions
    actions: List[str]             # List of RetuningAction values


class AutotunerFeedback(NamedTuple):
    """Feedback for autotuner iteration."""
    should_retune: bool
    priority_action: str           # Most important action
    all_actions: List[str]         # All recommended actions
    confidence: float              # Confidence in recommendation (0-1)
    reason: str                    # Human-readable explanation
    suggested_params: Dict[str, float]  # Suggested parameter changes


# =============================================================================
# Core Quality Computation
# =============================================================================

def compute_eps_im(
    ifft_result: jnp.ndarray,
    t: Optional[jnp.ndarray] = None,
    t_end: Optional[float] = None
) -> Tuple[float, Dict[str, float]]:
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
        eps_sym: Symmetry residual (expected ~1.0 for validated grid)
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

def classify_quality(
    eps_im: float,
    eps_im_valid: float,
    localization: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> Tuple[QualityLevel, List[RetuningAction], str]:
    """
    Classify quality and determine retuning actions.

    Args:
        eps_im: Full-grid imaginary leakage
        eps_im_valid: Valid-region imaginary leakage
        localization: Localization metrics dict
        thresholds: Custom thresholds (optional)

    Returns:
        (quality_level, recommended_actions, dominant_issue)
    """
    if thresholds is None:
        thresholds = {
            'excellent': 0.01,
            'good': 0.05,
            'acceptable': 0.10,
            'r_early_thresh': 0.5,
            'r_late_thresh': 0.5,
            'tail_thresh': 0.1,
            'spike_ratio': 5.0,  # p95/p50 threshold
        }

    # Use valid-region metric if available
    eps = eps_im_valid if eps_im_valid > 0 else eps_im

    # Determine quality level
    if eps < thresholds['excellent']:
        level = QualityLevel.EXCELLENT
    elif eps < thresholds['good']:
        level = QualityLevel.GOOD
    elif eps < thresholds['acceptable']:
        level = QualityLevel.ACCEPTABLE
    else:
        level = QualityLevel.POOR

    # Determine dominant issue and actions
    actions = []
    dominant = 'none'

    r_early = localization.get('r_early', 0.0)
    r_late = localization.get('r_late', 0.0)
    tail_ratio = localization.get('tail_ratio', 0.0)
    p50 = localization.get('leakage_p50', 0.0)
    p95 = localization.get('leakage_p95', 0.0)

    # Check for bandwidth issue (high early leakage)
    if r_early > thresholds['r_early_thresh']:
        dominant = 'bandwidth'
        actions.append(RetuningAction.REDUCE_DT)

    # Check for wraparound (high late leakage or tail)
    if r_late > thresholds['r_late_thresh'] or tail_ratio > thresholds['tail_thresh']:
        if dominant == 'none':
            dominant = 'wraparound'
        actions.append(RetuningAction.INCREASE_T)

    # Check for spike (p95 >> p50)
    spike_ratio = p95 / (p50 + 1e-14)
    if spike_ratio > thresholds['spike_ratio'] and level != QualityLevel.EXCELLENT:
        if dominant == 'none':
            dominant = 'spike'
        actions.append(RetuningAction.APPLY_PROJECTION)
        actions.append(RetuningAction.INCREASE_N)

    # For acceptable/poor quality, suggest projection or smoothing
    if level in [QualityLevel.ACCEPTABLE, QualityLevel.POOR]:
        if RetuningAction.APPLY_PROJECTION not in actions:
            actions.append(RetuningAction.APPLY_PROJECTION)

    return level, actions, dominant


# =============================================================================
# Comprehensive Quality Assessment
# =============================================================================

def assess_nilt_quality(
    ifft_result: jnp.ndarray,
    F_vals: Optional[jnp.ndarray] = None,
    t: Optional[jnp.ndarray] = None,
    t_end: Optional[float] = None,
    params: Optional[Dict[str, Any]] = None
) -> QualityMetrics:
    """
    Comprehensive quality assessment for NILT result.

    Args:
        ifft_result: Complex IFFT output
        F_vals: Frequency-domain values (for symmetry check)
        t: Time grid
        t_end: End time of interest
        params: NILT parameters (dt, N, a, T)

    Returns:
        QualityMetrics with all quality indicators
    """
    # Compute imaginary leakage
    eps_im, localization = compute_eps_im(ifft_result, t, t_end)

    # Compute symmetry residual
    eps_sym = compute_eps_sym(F_vals) if F_vals is not None else 0.0

    # Classify quality
    level, actions, dominant = classify_quality(
        eps_im, localization['eps_im_valid'], localization
    )

    # Detect spike
    p50 = localization['leakage_p50']
    p95 = localization['leakage_p95']
    spike_detected = (p95 / (p50 + 1e-14)) > 5.0

    return QualityMetrics(
        eps_im=eps_im,
        eps_im_valid=localization['eps_im_valid'],
        eps_sym=eps_sym,
        r_early=localization['r_early'],
        r_late=localization['r_late'],
        tail_ratio=localization['tail_ratio'],
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
    current_params: Dict[str, float],
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
        reason = f"Excellent quality (ε_Im = {metrics.eps_im_valid:.2%})"
        priority_action = 'none'
        suggested_params = {}
        confidence = 1.0
    elif metrics.quality_level == 'good':
        should_retune = False
        reason = f"Good quality (ε_Im = {metrics.eps_im_valid:.2%})"
        priority_action = 'none'
        suggested_params = {}
        confidence = 0.9
    else:
        # Determine priority action and parameter adjustments
        priority_action = metrics.actions[0] if metrics.actions else 'none'
        suggested_params = {}

        dt = current_params.get('dt', 0.01)
        N = current_params.get('N', 256)
        a = current_params.get('a', 0.0)
        T = current_params.get('T', N * dt)

        if metrics.dominant_issue == 'bandwidth':
            # Reduce dt to increase bandwidth
            suggested_params['dt'] = dt * 0.5
            suggested_params['N'] = N * 2  # Keep same T
            reason = f"Bandwidth issue detected (r_early = {metrics.r_early:.2f})"
            confidence = 0.85
        elif metrics.dominant_issue == 'wraparound':
            # Increase T to reduce wraparound
            suggested_params['T'] = T * 2.0
            suggested_params['N'] = N * 2  # Double N to maintain dt
            reason = f"Wraparound detected (tail_ratio = {metrics.tail_ratio:.2%})"
            confidence = 0.85
        elif metrics.dominant_issue == 'spike':
            # Increase N or apply projection
            suggested_params['N'] = min(N * 2, 8192)
            reason = f"Spike detected (p95/p50 = {metrics.leakage_p95/(metrics.leakage_p50+1e-14):.1f}x)"
            confidence = 0.75
        else:
            # General poor quality - try increasing N
            suggested_params['N'] = min(N * 2, 8192)
            reason = f"Poor quality (ε_Im = {metrics.eps_im_valid:.2%})"
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
# Integration with Existing Autotuner
# =============================================================================

def integrate_with_adaptive_tuner(
    result,
    F_eval: Callable,
    params: Dict[str, float],
    t_end: float,
    iteration: int = 0
) -> Tuple[QualityMetrics, AutotunerFeedback, bool]:
    """
    Integrate quality metrics with the adaptive tuner.

    This function is designed to be called from tune_nilt_adaptive()
    to provide quality-based feedback for retuning decisions.

    Args:
        result: NILTResult from nilt_fft_* function
        F_eval: Transfer function
        params: Current parameters
        t_end: End time of interest
        iteration: Current iteration

    Returns:
        (metrics, feedback, projection_recommended)
    """
    # Get IFFT result (need to recompute with complex output)
    # This assumes result has the raw complex values available
    ifft_result = result.f.astype(jnp.complex128)
    if hasattr(result, 'diagnostics') and result.diagnostics:
        # Use pre-computed diagnostics if available
        diag = result.diagnostics
        eps_im = diag.get('eps_im', 0.0)
        eps_sym = diag.get('eps_sym', 0.0)

    # Assess quality
    metrics = assess_nilt_quality(
        ifft_result=ifft_result,
        t=result.t,
        t_end=t_end,
        params=params
    )

    # Generate feedback
    feedback = generate_autotuner_feedback(
        metrics=metrics,
        current_params=params,
        iteration=iteration
    )

    # Determine if projection should be applied
    projection_recommended = (
        metrics.spike_detected or
        RetuningAction.APPLY_PROJECTION.value in metrics.actions
    )

    return metrics, feedback, projection_recommended


# =============================================================================
# Utility Functions
# =============================================================================

def print_quality_report(metrics: QualityMetrics, feedback: AutotunerFeedback) -> None:
    """Print a formatted quality report."""
    print("=" * 60)
    print("NILT Quality Assessment Report")
    print("=" * 60)
    print(f"Quality Level:     {metrics.quality_level.upper()}")
    print(f"ε_Im (full):       {metrics.eps_im:.4%}")
    print(f"ε_Im (valid):      {metrics.eps_im_valid:.4%}")
    print(f"ε_sym:             {metrics.eps_sym:.4f}")
    print("-" * 60)
    print(f"r_early:           {metrics.r_early:.3f}")
    print(f"r_late:            {metrics.r_late:.3f}")
    print(f"tail_ratio:        {metrics.tail_ratio:.4%}")
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
