"""
Adaptive NILT parameter tuning with closed-loop quality feedback.

The feedforward tuner (tune_nilt_params) picks dt, N, T and a from spectral
bounds. This module runs a pilot inversion, reads two kinds of sensor from
it, and applies one corrective action per iteration:

- band_edge_ratio and tail_energy_fraction, computed from the sampled
  transform F(a + i omega_k), measure how much of F sits at the edge of the
  resolved band [0, pi/dt]. When they are large the grid is too coarse:
  reduce dt.
- tail_ratio, computed from the damped inverse f(t) e^{-a t}, measures the
  energy beyond t_end that the periodic extension folds back onto
  [0, t_end]. When it is large the period is too short: increase T.

Earlier versions classified on the imaginary leakage eps_Im and its
localization. That quantity is zero by construction, because the spectrum is
mirrored into exact Hermitian symmetry before the ifft, so those sensors
never fired on a badly resolved transform; the ones above do.

The triad dt = 2T/N is re-normalized after every adjustment, and the loop is
bounded: max_iterations refinements, then an optional projection fallback.
"""

from __future__ import annotations

import warnings as _warnings
from typing import Literal, NamedTuple

import jax.numpy as jnp

from .nilt_fft import NILTResult, nilt_fft_uniform
from .spectral_bounds import BoundContext, SpectralBounds
from .tuning import TunedNILTParams, next_power_of_two, tune_nilt_params

# =============================================================================
# Quality classification
# =============================================================================

class QualityTier(NamedTuple):
    """Three-tier quality classification from the bandwidth and wraparound sensors."""
    tier: Literal['good', 'acceptable', 'poor']
    reason: str
    band_edge_ratio: float       # |F(a + i pi/dt)| / max_k |F(a + i omega_k)|
    tail_energy_fraction: float  # RMS share of the top tenth of the resolved band
    tail_ratio: float            # energy of f e^{-at} beyond t_end, relative to [0, t_end]
    r_late: float                # share of imaginary leakage in the last third of [0, t_end]


# (warn, crit) per sensor and policy. Calibrated on 1/(s+1) at N = 256:
# dt = 0.05 gives band_edge 0.016 and tail_energy 0.031 (RMS error 3.9%),
# dt = 0.5 gives 0.157 and 0.110 (16%), dt = 2 gives 0.537 and 0.221 (78%).
# tail_ratio keeps the levels the wraparound branch always used.
_TIER_THRESHOLDS = {
    'conservative': {'band_edge': (0.05, 0.20), 'tail_energy': (0.05, 0.12), 'tail_ratio': (0.08, 0.10)},
    'balanced': {'band_edge': (0.10, 0.30), 'tail_energy': (0.08, 0.18), 'tail_ratio': (0.09, 0.12)},
    'aggressive': {'band_edge': (0.15, 0.40), 'tail_energy': (0.10, 0.25), 'tail_ratio': (0.10, 0.15)},
}


def _read_sensors(diagnostics: dict | None) -> tuple[float, float, float, float]:
    """Pull the sensors out of an nilt_fft_* diagnostics dict; a missing sensor reads 0."""
    diagnostics = diagnostics or {}
    loc = diagnostics.get('leakage_localization', {})
    return (
        float(diagnostics.get('band_edge_ratio', 0.0)),
        float(diagnostics.get('tail_energy_fraction', 0.0)),
        float(loc.get('tail_ratio', 0.0)),
        float(loc.get('r_late', 0.0)),
    )


def _quality_from_diagnostics(tier: str, reason: str, diagnostics: dict | None) -> QualityTier:
    band_edge, tail_energy, tail_ratio, r_late = _read_sensors(diagnostics)
    return QualityTier(tier, reason, band_edge, tail_energy, tail_ratio, r_late)


def classify_quality(
    diagnostics: dict,
    tier: Literal['conservative', 'balanced', 'aggressive'] = 'balanced'
) -> QualityTier:
    """
    Classify NILT quality from nilt_fft_uniform(..., return_diagnostics=True).

    - **good**: every sensor below its warning level
    - **acceptable**: a sensor above its warning level; a retune is optional
    - **poor**: a sensor above its critical level; retune_based_on_diagnostics
      picks the remedy

    Bandwidth is judged on band_edge_ratio and tail_energy_fraction,
    wraparound on tail_ratio. r_late is carried for information only.
    Exported from the package as classify_quality_tier.

    Args:
        diagnostics: Dict from nilt_fft_uniform(..., return_diagnostics=True)
        tier: Threshold policy ('conservative', 'balanced', 'aggressive')

    Returns:
        QualityTier with classification, reason and the sensor values
    """
    if tier not in _TIER_THRESHOLDS:
        raise ValueError(f"unknown quality tier {tier!r}; expected one of {sorted(_TIER_THRESHOLDS)}")
    th = _TIER_THRESHOLDS[tier]
    band_edge, tail_energy, tail_ratio, r_late = _read_sensors(diagnostics)

    def verdict(level: str, reason: str) -> QualityTier:
        return QualityTier(level, reason, band_edge, tail_energy, tail_ratio, r_late)

    bandwidth = f'band_edge_ratio={band_edge:.3f}, tail_energy_fraction={tail_energy:.3f}'
    wraparound = f'tail_ratio={tail_ratio:.3f}'

    if band_edge > th['band_edge'][1] or tail_energy > th['tail_energy'][1]:
        return verdict('poor', f'{bandwidth} (bandwidth)')
    if tail_ratio > th['tail_ratio'][1]:
        return verdict('poor', f'{wraparound} (wraparound)')
    if band_edge > th['band_edge'][0] or tail_energy > th['tail_energy'][0]:
        return verdict('acceptable', f'{bandwidth} (bandwidth marginal)')
    if tail_ratio > th['tail_ratio'][0]:
        return verdict('acceptable', f'{wraparound} (wraparound marginal)')
    return verdict('good', 'all sensors within normal range')


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
    Apply ONE corrective action based on the quality sensors.

    Deterministic escalation ladder:
    1. Bandwidth sensors above their warning level -> halve dt, keep T
    2. Wraparound sensor above its warning level -> double T, keep dt
    3. Otherwise -> halve a if it is not already small, else double N

    dt = 2T/N is re-normalized after the adjustment and N is capped at
    max_N. When the cap leaves every parameter as it was, the current
    parameters are returned unchanged (same object) with the action
    "at max_N, no change", so a caller's loop can stop instead of repeating
    the same inversion until its iteration budget runs out.

    Args:
        current_params: Current NILT parameters
        quality: Quality classification from classify_quality()
        bounds: Spectral bounds (unused, kept for the call signature)
        max_N: Maximum allowed N

    Returns:
        new_params: Adjusted parameters with the triad re-normalized
        action: Description of the action taken
    """
    dt = current_params.dt
    N = current_params.N
    T = current_params.T
    a = current_params.a
    a_new = a
    th = _TIER_THRESHOLDS['balanced']

    if quality.band_edge_ratio > th['band_edge'][0] or quality.tail_energy_fraction > th['tail_energy'][0]:
        # Too much of F at the band edge: halve dt (double the Nyquist frequency)
        T_new = T
        N_new = min(next_power_of_two(int(round(4.0 * T / dt))), max_N)
        dt_new = 2 * T_new / N_new
        action = f"reduced dt: {dt:.4f} → {dt_new:.4f} (bandwidth)"

    elif quality.tail_ratio > th['tail_ratio'][0]:
        # Damped inverse has not decayed by t_end: double T (finer frequency resolution)
        T_new = 2.0 * T
        N_new = min(next_power_of_two(int(round(2.0 * T_new / dt))), max_N)
        dt_new = 2 * T_new / N_new
        action = f"increased T: {T:.2f} → {T_new:.2f} (wraparound)"

    elif a > 0.01:
        # General degradation: reduce the exponential amplification
        a_new = a / 2.0
        dt_new, T_new, N_new = dt, T, N
        action = f"reduced a: {a:.3f} → {a_new:.3f} (general quality)"

    else:
        N_new = min(2 * N, max_N)
        T_new = T
        dt_new = 2 * T_new / N_new
        action = f"increased N: {N} → {N_new} (fallback)"

    if (dt_new, N_new, T_new, a_new) == (dt, N, T, a):
        return current_params, "at max_N, no change"

    new_params = TunedNILTParams(
        dt=dt_new,
        N=N_new,
        T=T_new,
        a=a_new,
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
    4. If poor: one corrective action from retune_based_on_diagnostics(),
       at most max_iterations times; the loop also stops when the N cap
       leaves nothing to change
    5. If still poor: the Hermitian projection is tried as a last resort.
       The projection cannot move the bandwidth or wraparound sensors (the
       spectrum is Hermitian by construction), so this rarely changes the
       verdict; it is kept so ``actions`` records that everything was tried,
       and a UserWarning names the remaining problem.

    Args:
        F_eval: Laplace-domain function F(s)
        t_end: End time for inversion
        bounds: Spectral bounds (or dict with {rho, re_max, im_max})
        ctx: BoundContext for computing bounds (if bounds not provided)
        dtype: Data type for computation
        max_iterations: Maximum retuning iterations (default: 2)
        quality_tier: Threshold policy for quality classification
        enable_projection_fallback: If True, try the projection when retuning fails
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

    # Steps 2-4: bounded refinement loop
    for iteration in range(max_iterations + 1):
        result = nilt_fft_uniform(
            F_eval,
            dt=params.dt,
            N=params.N,
            a=params.a,
            dtype=dtype,
            t_end=t_end,
            return_diagnostics=True
        )
        quality = classify_quality(result.diagnostics, tier=quality_tier)

        if quality.tier in ['good', 'acceptable']:
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        if iteration < max_iterations:
            new_params, action = retune_based_on_diagnostics(
                params, quality, bounds=bounds, max_N=tune_kwargs.get('N_max', 8192)
            )
            actions.append(action)
            if new_params is not params:
                params = new_params
                continue
        break

    # Step 5: still poor with nothing left to adjust
    if enable_projection_fallback:
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
        actions.append("projection fallback (retuning exhausted)")

        if quality_proj.tier in ['good', 'acceptable']:
            return AdaptiveTuningResult(
                params=params,
                result=result_proj,
                quality=quality_proj,
                iterations=iteration + 1,
                actions=actions
            )

    _warnings.warn(
        f"NILT quality remains poor after {iteration} retuning attempt(s). "
        f"Reason: {quality.reason}. Consider time-domain splitting.",
        UserWarning, stacklevel=2
    )
    return AdaptiveTuningResult(
        params=params,
        result=result,
        quality=quality,
        iterations=iteration,
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
    tau_tail: float = 1e-2,
    tau_chi: float = 2.0,
    A_max: float = 1e6,
    **tune_kwargs
) -> AdaptiveTuningResult:
    """
    Adaptive NILT tuning with CFL-like spectral conditions as primary guidance.

    Unlike tune_nilt_adaptive, which classifies on the bandwidth and
    wraparound sensors of the pilot inversion, this function uses the
    spectral CFL conditions of endpoint_diagnostics (after Hsu & Dranoff)
    for its parameter adjustments.

    CFL condition hierarchy:
    1. Endpoint compatibility (J): controls the switch to halfstep + IVT.
       The jump is a property of f, not of the grid, so once half-step
       sampling is in use the condition counts as settled and is not
       re-evaluated.
    2. Bandwidth coverage (R_tail): controls dt refinement
    3. Quadrature resolution (chi): controls T expansion
    4. Conditioning (A_exp): controls the Bromwich shift a

    The default tolerances are met by tune_nilt_params' own defaults on a
    well-posed transform (tau_chi = 2.0 admits chi = pi/2 at
    period_factor = 4; tau_tail = 1e-2 admits the 5e-3 tail of 1/(s+1) at
    the tuned dt), so the loop converges at iteration 0 there and iterates
    only when a condition is actually violated.

    Args:
        F_eval: Laplace-domain function F(s)
        t_end: End time for inversion
        bounds: Spectral bounds for parameter selection
        ctx: BoundContext if bounds not provided
        dtype: Data type for computation
        max_iterations: Maximum refinement iterations
        use_halfstep_ivt: If True, use halfstep + IVT when an endpoint jump is detected
        tau_end: Endpoint jump tolerance (relative to the signal scale)
        tau_tail: Bandwidth tail energy tolerance
        tau_chi: Quadrature phase-step tolerance
        A_max: Maximum exponential amplification
        **tune_kwargs: Additional kwargs for tune_nilt_params()

    Returns:
        AdaptiveTuningResult with CFL-based diagnostics
    """
    from .endpoint_diagnostics import check_spectral_cfl_conditions, suggest_parameter_adjustments
    from .nilt_fft import nilt_fft_halfstep_ivt

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

        # Step 4: Endpoint jump -> half-step sampling with IVT. CFL-2 to 4
        # depend only on F and the parameters, so they stand as computed;
        # CFL-1 is settled by the switch.
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
            remaining = [v for v in cfl.violated_conditions if not v.startswith('endpoint_jump')]
            cfl = cfl._replace(
                endpoint_compatible=True,
                violated_conditions=remaining,
                all_conditions_met=not remaining,
            )

        # Step 5: All conditions met -> done
        if cfl.all_conditions_met:
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=_quality_from_diagnostics('good', 'all CFL conditions satisfied', result.diagnostics),
                iterations=iteration,
                actions=actions
            )

        # Step 6: Max iterations reached
        if iteration >= max_iterations:
            tier = 'acceptable' if len(cfl.violated_conditions) == 1 else 'poor'
            quality = _quality_from_diagnostics(
                tier, f"CFL violations: {', '.join(cfl.violated_conditions)}", result.diagnostics
            )
            if tier == 'poor':
                _warnings.warn(
                    f"NILT CFL conditions not fully satisfied after {max_iterations} iterations. "
                    f"Violations: {cfl.violated_conditions}",
                    UserWarning, stacklevel=2
                )
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=quality,
                iterations=iteration,
                actions=actions
            )

        # Step 7: CFL-guided parameter adjustment
        adjustments, action = suggest_parameter_adjustments(
            cfl, params, bounds, max_N=tune_kwargs.get('N_max', 8192)
        )

        if not any(key in adjustments for key in ('dt', 'T', 'a')):
            # Nothing that changes the parameters is left (an endpoint jump
            # with use_halfstep_ivt=False, or a already minimal).
            tier = 'acceptable' if len(cfl.violated_conditions) == 1 else 'poor'
            return AdaptiveTuningResult(
                params=params,
                result=result,
                quality=_quality_from_diagnostics(tier, 'no further adjustments available', result.diagnostics),
                iterations=iteration,
                actions=actions
            )

        actions.append(action)
        new_dt = adjustments.get('dt', params.dt)
        new_T = adjustments.get('T', params.T)
        new_a = adjustments.get('a', params.a)

        # Re-normalize triad: dt = 2T/N
        new_N = next_power_of_two(int(round(2 * new_T / new_dt)))
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

    # Not reachable: the loop returns at iteration == max_iterations
    return AdaptiveTuningResult(
        params=params,
        result=result,
        quality=_quality_from_diagnostics('poor', 'max iterations exceeded', result.diagnostics),
        iterations=max_iterations,
        actions=actions
    )
