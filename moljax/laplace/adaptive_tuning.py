"""
Adaptive NILT parameter tuning with closed-loop quality feedback.

This module implements a bounded 2-iteration controller that uses ε_Im (imaginary
leakage) diagnostics to adaptively refine NILT parameters when initial autotuning
produces suboptimal quality.

Key features:
- Feedforward guardrails (tune_nilt_params) + feedback diagnostics (ε_Im)
- Localization-guided retuning (early → bandwidth, late → wraparound)
- Deterministic escalation ladder (dt/T/N → a → projection → splitting)
- Triad re-normalization (dt = 2T/N preserved after adjustments)
- Bounded iteration (max 2 refinement steps)
"""

from __future__ import annotations

from typing import NamedTuple, Literal
import warnings as _warnings

import jax.numpy as jnp

from .tuning import tune_nilt_params, TunedNILTParams, next_power_of_two
from .nilt_fft import nilt_fft_uniform, NILTResult
from .spectral_bounds import SpectralBounds, BoundContext


# =============================================================================
# Quality classification
# =============================================================================

class QualityTier(NamedTuple):
    """Three-tier quality classification based on ε_Im diagnostics."""
    tier: Literal['good', 'acceptable', 'poor']
    reason: str
    eps_im_valid: float
    r_early: float
    r_late: float
    tail_ratio: float
    spike_ratio: float  # p95/p50


def classify_quality(
    diagnostics: dict,
    tier: Literal['conservative', 'balanced', 'aggressive'] = 'balanced'
) -> QualityTier:
    """
    Classify NILT quality based on ε_Im diagnostics.

    Three-tier thresholds:
    - **Good**: All metrics within normal range
    - **Acceptable**: Moderate degradation, projection may help
    - **Poor**: Significant issues, retuning recommended

    Args:
        diagnostics: Dict from nilt_fft_uniform(..., return_diagnostics=True)
        tier: Threshold policy ('conservative', 'balanced', 'aggressive')

    Returns:
        QualityTier with classification and reason
    """
    loc = diagnostics.get('leakage_localization', {})

    eps_im_valid = loc.get('eps_im_valid', diagnostics.get('eps_im_before', 1.0))
    r_early = loc.get('r_early', 0.0)
    r_late = loc.get('r_late', 0.0)
    tail_ratio = loc.get('tail_ratio', 0.0)

    p50 = loc.get('leakage_p50', 1.0)
    p95 = loc.get('leakage_p95', 1.0)
    spike_ratio = p95 / (p50 + 1e-10)

    # Threshold sets (conservative → aggressive)
    if tier == 'conservative':
        eps_im_warn = 1.5
        eps_im_crit = 2.0
        r_warn = 0.45
        r_crit = 0.50
        tail_warn = 0.08
        tail_crit = 0.10
        spike_warn = 3.0
        spike_crit = 5.0
    elif tier == 'aggressive':
        eps_im_warn = 2.0
        eps_im_crit = 3.0
        r_warn = 0.50
        r_crit = 0.60
        tail_warn = 0.10
        tail_crit = 0.15
        spike_warn = 5.0
        spike_crit = 10.0
    else:  # balanced (default)
        eps_im_warn = 1.8
        eps_im_crit = 2.5
        r_warn = 0.48
        r_crit = 0.55
        tail_warn = 0.09
        tail_crit = 0.12
        spike_warn = 4.0
        spike_crit = 7.0

    # Check critical thresholds first (poor quality)
    if eps_im_valid > eps_im_crit:
        return QualityTier('poor', f'eps_im_valid={eps_im_valid:.2f} > {eps_im_crit}',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)
    if r_early > r_crit:
        return QualityTier('poor', f'r_early={r_early:.2f} > {r_crit} (bandwidth issue)',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)
    if r_late > r_crit or tail_ratio > tail_crit:
        return QualityTier('poor', f'r_late={r_late:.2f} or tail_ratio={tail_ratio:.3f} (wraparound)',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)
    if spike_ratio > spike_crit:
        return QualityTier('poor', f'spike_ratio={spike_ratio:.2f} > {spike_crit} (resolution issue)',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)

    # Check warning thresholds (acceptable quality)
    if eps_im_valid > eps_im_warn:
        return QualityTier('acceptable', f'eps_im_valid={eps_im_valid:.2f} slightly elevated',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)
    if r_early > r_warn:
        return QualityTier('acceptable', f'r_early={r_early:.2f} slightly high',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)
    if r_late > r_warn or tail_ratio > tail_warn:
        return QualityTier('acceptable', f'r_late={r_late:.2f} or tail_ratio={tail_ratio:.3f} slightly high',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)
    if spike_ratio > spike_warn:
        return QualityTier('acceptable', f'spike_ratio={spike_ratio:.2f} slightly high',
                           eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)

    # All metrics within normal range
    return QualityTier('good', 'all metrics within normal range',
                       eps_im_valid, r_early, r_late, tail_ratio, spike_ratio)


# =============================================================================
# Retuning logic
# =============================================================================

def retune_based_on_diagnostics(
    current_params: TunedNILTParams,
    quality: QualityTier,
    bounds: SpectralBounds | dict | None = None,
    max_N: int = 8192
) -> tuple[TunedNILTParams, str]:
    """
    Apply ONE corrective action based on quality diagnostics.

    Deterministic escalation ladder:
    1. Early-time leakage → reduce dt (bandwidth)
    2. Late-time leakage → increase T (wraparound)
    3. Spike/resolution → increase N
    4. General degradation → reduce a if feasible

    Always re-normalizes triad: dt = 2T/N after adjustments.

    Args:
        current_params: Current NILT parameters
        quality: Quality classification from classify_quality()
        bounds: Spectral bounds (for a reduction feasibility)
        max_N: Maximum allowed N

    Returns:
        new_params: Adjusted parameters with triad re-normalized
        action: Description of action taken
    """
    # Extract current values
    dt = current_params.dt
    N = current_params.N
    T = current_params.T
    a = current_params.a

    # Determine action based on localization
    if quality.r_early > 0.50:
        # Early-time leakage dominant → bandwidth issue
        # Action: reduce dt (increase Nyquist frequency)
        dt_new = dt / 2.0
        T_new = T  # Keep period
        N_new = next_power_of_two(int(2 * T_new / dt_new))
        N_new = min(N_new, max_N)
        dt_new = 2 * T_new / N_new  # Re-normalize
        action = f"reduced dt: {dt:.4f} → {dt_new:.4f} (bandwidth)"

    elif quality.r_late > 0.50 or quality.tail_ratio > 0.10:
        # Late-time leakage or tail energy dominant → wraparound
        # Action: increase T (finer frequency resolution)
        T_new = 2.0 * T
        dt_new = dt  # Try to keep dt
        N_new = next_power_of_two(int(2 * T_new / dt_new))
        N_new = min(N_new, max_N)
        dt_new = 2 * T_new / N_new  # Re-normalize
        action = f"increased T: {T:.2f} → {T_new:.2f} (wraparound)"

    elif quality.spike_ratio > 5.0:
        # Spike/resolution issue
        # Action: increase N
        N_new = min(2 * N, max_N)
        T_new = T  # Keep period
        dt_new = 2 * T_new / N_new  # Re-normalize
        action = f"increased N: {N} → {N_new} (resolution)"

    else:
        # General quality degradation
        # Action: try to reduce a if feasible
        if a > 0.01:
            a_new = max(0.0, a / 2.0)
            # Keep dt, T, N
            dt_new = dt
            T_new = T
            N_new = N
            action = f"reduced a: {a:.3f} → {a_new:.3f} (general quality)"
            a = a_new
        else:
            # Can't reduce a further, increase N as fallback
            N_new = min(2 * N, max_N)
            T_new = T
            dt_new = 2 * T_new / N_new
            action = f"increased N: {N} → {N_new} (fallback)"

    # Create new params (preserve other fields)
    new_params = TunedNILTParams(
        dt=dt_new,
        N=N_new,
        T=T_new,
        a=a,
        omega_max=float(jnp.pi / dt_new),
        omega_req=current_params.omega_req,
        bound_sources=current_params.bound_sources,
        warnings=current_params.warnings + [f"Retuned: {action}"],
        diagnostics=current_params.diagnostics
    )

    return new_params, action


# =============================================================================
# Adaptive tuning with closed-loop feedback
# =============================================================================

class AdaptiveTuningResult(NamedTuple):
    """Result from adaptive NILT tuning with quality feedback."""
    params: TunedNILTParams
    result: NILTResult
    quality: QualityTier
    iterations: int
    actions: list[str]  # List of retuning actions taken


def tune_nilt_adaptive(
    F_eval,
    *,
    t_end: float,
    bounds: SpectralBounds | dict | None = None,
    ctx: BoundContext | None = None,
    dtype=jnp.float64,
    max_iterations: int = 2,
    quality_tier: Literal['conservative', 'balanced', 'aggressive'] = 'balanced',
    enable_projection_fallback: bool = True,
    **tune_kwargs
) -> AdaptiveTuningResult:
    """
    Adaptive NILT parameter tuning with closed-loop quality feedback.

    Process:
    1. Feedforward: tune_nilt_params() (feasibility guardrails)
    2. Pilot run: nilt_fft_uniform() with diagnostics
    3. Quality check: classify_quality()
    4. If poor: retune based on localization (max 2 iterations)
    5. If still poor: offer projection as fallback

    Args:
        F_eval: Laplace-domain function F(s)
        t_end: End time for inversion
        bounds: Spectral bounds (or dict with {rho, re_max, im_max})
        ctx: BoundContext for computing bounds (if bounds not provided)
        dtype: Data type for computation
        max_iterations: Maximum retuning iterations (default: 2)
        quality_tier: Threshold policy for quality classification
        enable_projection_fallback: If True, enable projection when retuning fails
        **tune_kwargs: Additional kwargs for tune_nilt_params()

    Returns:
        AdaptiveTuningResult with final parameters, result, and quality info
    """
    actions = []

    # Step 1: Feedforward autotuning (feasibility)
    params = tune_nilt_params(
        t_end=t_end,
        bounds=bounds,
        ctx=ctx,
        dtype=dtype,
        **tune_kwargs
    )
    actions.append("initial autotuning (feedforward)")

    # Step 2-4: Iterative refinement loop (bounded)
    for iteration in range(max_iterations + 1):
        # Run NILT with diagnostics
        result = nilt_fft_uniform(
            F_eval,
            dt=params.dt,
            N=params.N,
            a=params.a,
            dtype=dtype,
            t_end=t_end,
            return_diagnostics=True
        )

        # Classify quality
        quality = classify_quality(result.diagnostics, tier=quality_tier)

        # If quality is good or acceptable, we're done
        if quality.tier in ['good', 'acceptable']:
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        # If poor quality and we've exhausted iterations, try projection fallback
        if iteration >= max_iterations:
            if enable_projection_fallback:
                # Last resort: apply projection
                result_proj = nilt_fft_uniform(
                    F_eval,
                    dt=params.dt,
                    N=params.N,
                    a=params.a,
                    dtype=dtype,
                    t_end=t_end,
                    apply_projection=True,
                    return_diagnostics=True
                )
                quality_proj = classify_quality(result_proj.diagnostics, tier=quality_tier)
                actions.append("projection fallback (max iterations exhausted)")

                if quality_proj.tier in ['good', 'acceptable']:
                    return AdaptiveTuningResult(
                        params=params,
                        result=result_proj,
                        quality=quality_proj,
                        iterations=iteration + 1,
                        actions=actions
                    )

            # Still poor after all attempts
            _warnings.warn(
                f"NILT quality remains poor after {max_iterations} retuning attempts. "
                f"Reason: {quality.reason}. Consider time-domain splitting.",
                UserWarning
            )
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        # Retune based on diagnostics
        params, action = retune_based_on_diagnostics(
            params, quality, bounds=bounds, max_N=tune_kwargs.get('N_max', 8192)
        )
        actions.append(action)

    # Should never reach here due to loop structure, but for completeness
    return AdaptiveTuningResult(
        params=params,
        result=result,
        quality=quality,
        iterations=max_iterations,
        actions=actions
    )


# =============================================================================
# CFL-guided adaptive tuning
# =============================================================================

def tune_nilt_adaptive_cfl(
    F_eval,
    *,
    t_end: float,
    bounds: SpectralBounds | dict | None = None,
    ctx: BoundContext | None = None,
    dtype=jnp.float64,
    max_iterations: int = 3,
    use_halfstep_ivt: bool = True,
    tau_end: float = 0.01,
    tau_tail: float = 1e-4,
    tau_chi: float = 0.3,
    A_max: float = 1e6,
    **tune_kwargs
) -> AdaptiveTuningResult:
    """
    Adaptive NILT tuning with CFL-like spectral conditions as primary guidance.

    Unlike tune_nilt_adaptive which uses ε_Im quality classification,
    this function uses principled spectral CFL conditions (from Hsu & Dranoff)
    for parameter adjustment decisions.

    CFL Condition Hierarchy:
    1. Endpoint compatibility (J): Controls use of halfstep + IVT
    2. Bandwidth coverage (R_tail): Controls dt refinement
    3. Quadrature resolution (χ): Controls T expansion
    4. Conditioning (A_exp): Controls Bromwich shift a

    Args:
        F_eval: Laplace-domain function F(s)
        t_end: End time for inversion
        bounds: Spectral bounds for parameter selection
        ctx: BoundContext if bounds not provided
        dtype: Data type for computation
        max_iterations: Maximum refinement iterations
        use_halfstep_ivt: If True, use halfstep + IVT when endpoint jump detected
        tau_end: Endpoint jump tolerance (relative)
        tau_tail: Bandwidth tail energy tolerance
        tau_chi: Quadrature phase-step tolerance
        A_max: Maximum exponential amplification
        **tune_kwargs: Additional kwargs for tune_nilt_params()

    Returns:
        AdaptiveTuningResult with CFL-based diagnostics
    """
    from .endpoint_diagnostics import check_spectral_cfl_conditions, suggest_parameter_adjustments
    from .nilt_fft import nilt_fft_uniform, nilt_fft_halfstep_ivt

    actions = []

    # Step 1: Initial feedforward autotuning
    params = tune_nilt_params(
        t_end=t_end,
        bounds=bounds,
        ctx=ctx,
        dtype=dtype,
        **tune_kwargs
    )
    actions.append("initial autotuning (CFL-guided)")

    for iteration in range(max_iterations + 1):
        # Step 2: Run NILT with diagnostics
        result = nilt_fft_uniform(
            F_eval,
            dt=params.dt,
            N=params.N,
            a=params.a,
            dtype=dtype,
            t_end=t_end,
            return_diagnostics=True
        )

        # Step 3: Check CFL conditions
        cfl = check_spectral_cfl_conditions(
            result=result,
            F_eval=F_eval,
            t_end=t_end,
            a=params.a,
            T=params.T,
            dt=params.dt,
            tau_end=tau_end,
            tau_tail=tau_tail,
            tau_chi=tau_chi,
            A_max=A_max,
        )

        # Step 4: Check if endpoint jump detected → use halfstep + IVT
        if not cfl.endpoint_compatible and use_halfstep_ivt:
            result = nilt_fft_halfstep_ivt(
                F_eval,
                dt=params.dt,
                N=params.N,
                a=params.a,
                dtype=dtype,
                t_end=t_end,
                return_diagnostics=True
            )
            actions.append(f"endpoint jump J={cfl.endpoint_jump:.2e} → halfstep + IVT")

            # Re-check CFL with halfstep result
            cfl = check_spectral_cfl_conditions(
                result=result,
                F_eval=F_eval,
                t_end=t_end,
                a=params.a,
                T=params.T,
                dt=params.dt,
                tau_end=tau_end,
                tau_tail=tau_tail,
                tau_chi=tau_chi,
                A_max=A_max,
            )

        # Step 5: All conditions met → done
        if cfl.all_conditions_met:
            quality = QualityTier('good', 'all CFL conditions satisfied',
                                  0.0, 0.0, 0.0, cfl.tail_energy_ratio, 1.0)
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        # Step 6: Max iterations reached
        if iteration >= max_iterations:
            tier = 'acceptable' if len(cfl.violated_conditions) == 1 else 'poor'
            quality = QualityTier(
                tier,
                f"CFL violations: {', '.join(cfl.violated_conditions)}",
                0.0, 0.0, 0.0, cfl.tail_energy_ratio, 1.0
            )
            if tier == 'poor':
                _warnings.warn(
                    f"NILT CFL conditions not fully satisfied after {max_iterations} iterations. "
                    f"Violations: {cfl.violated_conditions}",
                    UserWarning
                )
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        # Step 7: Apply CFL-guided parameter adjustment
        adjustments, action = suggest_parameter_adjustments(
            cfl, params, bounds, max_N=tune_kwargs.get('N_max', 8192)
        )

        if not adjustments:
            # No adjustments suggested, done
            quality = QualityTier('acceptable', 'no further adjustments available',
                                  0.0, 0.0, 0.0, cfl.tail_energy_ratio, 1.0)
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        # Apply adjustments
        actions.append(action)
        new_dt = adjustments.get('dt', params.dt)
        new_T = adjustments.get('T', params.T)
        new_a = adjustments.get('a', params.a)

        # Re-normalize triad: dt = 2T/N
        new_N = next_power_of_two(int(2 * new_T / new_dt))
        new_N = min(new_N, tune_kwargs.get('N_max', 8192))
        new_dt = 2 * new_T / new_N

        params = TunedNILTParams(
            dt=new_dt,
            N=new_N,
            T=new_T,
            a=new_a,
            omega_max=float(jnp.pi / new_dt),
            omega_req=params.omega_req,
            bound_sources=params.bound_sources,
            warnings=params.warnings + [f"CFL-adjusted: {action}"],
            diagnostics=params.diagnostics
        )

    # Should never reach here
    return AdaptiveTuningResult(
        params=params,
        result=result,
        quality=QualityTier('poor', 'max iterations exceeded', 0.0, 0.0, 0.0, 0.0, 1.0),
        iterations=max_iterations,
        actions=actions
    )
