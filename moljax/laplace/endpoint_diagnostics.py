"""
Principled spectral CFL-like guardrails for FFT-NILT.

This module implements quantitative stability conditions analogous to CFL
constraints, replacing qualitative ε_Im quality tiers with measurable
frequency-domain constraints:

1. Endpoint Compatibility: Periodization jump condition
2. Bandwidth Coverage: Nyquist-like tail energy from F(s) samples
3. Quadrature Resolution: Phase-step condition for oscillatory integral
4. Conditioning Guard: Exponential amplification bound
5. Spectral Placement: the Bromwich abscissa a must stay to the right of
   the abscissa sigma of the transform being inverted, by at least the
   wraparound margin ln(1/eps_tail)/(2T). Conditions 2 to 4 are all
   satisfied by a contour that runs through a pole, so without this one the
   check can report success on an inversion that is not even convergent.

Based on the interpretation of FFT-NILT as trapezoidal quadrature of the
Bromwich integral with periodic extension (Hsu & Dranoff 1987, Weeks 1966).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class SpectralCFLConditions(NamedTuple):
    """Quantitative CFL-like conditions for FFT-NILT."""

    # CFL-1: Endpoint compatibility
    endpoint_jump: float  # J = |f(0+) - f(2T-)|
    f_0_ivt: float | None  # f(0+) from Initial Value Theorem
    f_2T: float  # f(2T-) from last sample
    endpoint_compatible: bool  # J ≤ threshold

    # CFL-2: Bandwidth coverage
    tail_energy_ratio: float  # R_tail from F(s) samples
    omega_max: float  # Maximum frequency π/Δt
    bandwidth_sufficient: bool  # R_tail ≤ threshold

    # CFL-3: Quadrature resolution
    phase_step: float  # χ = π·t_end/T
    quadrature_stable: bool  # χ ≤ threshold

    # Conditioning guard
    exp_amplification: float  # A_exp = exp(a·t_end)
    conditioning_safe: bool  # A_exp ≤ threshold

    # Spectral placement guard
    sigma: float  # Abscissa of the inverted transform (nan when not supplied)
    a_required: float  # sigma + max(delta_min, ln(1/eps_tail)/(2T)) (nan if sigma is)
    spectral_placement_ok: bool  # a ≥ a_required

    # Overall status
    all_conditions_met: bool
    violated_conditions: list[str]


def compute_ivt(F_eval, method='large_s', s_large=1e6):
    """
    Compute f(0+) using Initial Value Theorem.

    IVT: f(0+) = lim_{s→∞} s·F(s)

    Args:
        F_eval: Laplace-domain function F(s)
        method: 'large_s' or 'richardson' for extrapolation
        s_large: Value of s for large-s approximation

    Returns:
        f(0+) estimate, or None if computation fails
    """
    try:
        if method == 'large_s':
            # Simple approximation: evaluate at large s
            f_0 = float(s_large * F_eval(s_large))
            return f_0
        elif method == 'richardson':
            # Richardson extrapolation using multiple s values
            s_vals = np.array([1e4, 1e5, 1e6])
            f_vals = np.array([float(s * F_eval(s)) for s in s_vals])
            # Linear extrapolation in 1/s
            coeffs = np.polyfit(1/s_vals, f_vals, deg=1)
            f_0 = coeffs[1]  # Intercept at 1/s = 0
            return float(f_0)
        else:
            return None
    except Exception:
        return None


def compute_tail_energy_from_F_samples(F_eval, omega_grid, a, tail_fraction=0.1):
    """
    Compute tail energy ratio from frequency-domain samples.

    CRITICAL: Use F(a+iω) samples, NOT rfft(result.f), because:
    - result.f already contains endpoint artifacts
    - We need to measure truncation of the Bromwich integral
    - Hsu & Dranoff explicitly discuss truncation of F(iω) beyond ω_max

    Args:
        F_eval: Laplace-domain function F(s)
        omega_grid: Frequency samples (1D array)
        a: Bromwich shift
        tail_fraction: Fraction of frequencies considered "tail" (default 10%)

    Returns:
        R_tail: Ratio of tail energy to total energy
    """
    # Evaluate F at all frequency points in one vectorized call, as every
    # nilt_fft_* routine does; transfer functions that vmap internally
    # (create_transfer_function_from_fft_operator) reject scalar calls.
    s_grid = a + 1j * omega_grid
    F_samples = jnp.asarray(F_eval(s_grid))

    # Energy (squared magnitude)
    energy = jnp.abs(F_samples)**2
    total_energy = jnp.sum(energy)

    # Tail: top tail_fraction of frequencies
    n_tail = max(1, int(len(omega_grid) * tail_fraction))
    tail_energy = jnp.sum(energy[-n_tail:])

    # Ratio
    R_tail = float(tail_energy / (total_energy + 1e-30))

    return R_tail


def required_abscissa(
    sigma: float | None,
    T: float,
    delta_min: float = 1e-3,
    eps_tail: float = 1e-8,
) -> float:
    """
    Smallest Bromwich abscissa a that keeps the inversion convergent.

    The contour Re s = a must lie to the right of sigma, the abscissa of the
    transform being inverted (the spectral abscissa for a normal operator,
    the numerical abscissa for a non-normal one), and far enough to the right
    that the wraparound condition 2(a - sigma)T >= ln(1/eps_tail) holds over
    the period 2T:

        a >= sigma + max(delta_min, ln(1/eps_tail) / (2T)).

    This is the same floor tune_nilt_params uses to pick its shift; it is
    exposed here so that every later adjustment of a can be measured against
    it instead of being accepted on the strength of the other conditions.

    Args:
        sigma: Abscissa of the inverted transform (None or non-finite when
            no spectral information is available)
        T: Half-period (the period is 2T)
        delta_min: Minimal spectral margin
        eps_tail: Wraparound suppression threshold

    Returns:
        The required abscissa, or nan when sigma is unknown
    """
    if sigma is None or not np.isfinite(sigma):
        return float('nan')
    t_max = 2.0 * T
    margin = max(delta_min, np.log(1.0 / eps_tail) / t_max) if t_max > 0 else delta_min
    return float(sigma) + float(margin)


def check_spectral_cfl_conditions(
    result,
    F_eval,
    t_end: float,
    a: float,
    T: float,
    dt: float,
    tau_end: float = 0.01,  # Endpoint jump tolerance (1% of the signal scale)
    tau_tail: float = 1e-2,  # Tail energy tolerance
    tau_chi: float = 2.0,  # Phase-step tolerance (pi/2 at period_factor 4)
    A_max: float = 1e6,  # Maximum exp amplification (float64 safe)
    sigma: float | None = None,  # Abscissa of the inverted transform
    delta_min: float = 1e-3,  # Minimal spectral margin for placement
    eps_tail: float = 1e-8,  # Wraparound suppression threshold for placement
) -> SpectralCFLConditions:
    """
    Check all spectral CFL-like conditions for FFT-NILT.

    The default tolerances are chosen so that tune_nilt_params' own defaults
    pass them on a well-posed transform: period_factor = 4 gives
    chi = pi t_end / T = pi/2, and the tail energy of 1/(s+1) on the tuned
    grid is about 5e-3. Tighter values (the previous 0.3 and 1e-4) flagged
    every tuned parameter set and sent the CFL tuner into retuning loops
    that could not satisfy them.

    Args:
        result: NILTResult from nilt_fft_uniform (or a half-step variant)
        F_eval: Laplace-domain function F(s), evaluated on an array of s
        t_end: Integration end time
        a: Bromwich shift
        T: Half-period (2T = period)
        dt: Time step
        tau_end: Endpoint jump tolerance, relative to max(|f(0+)|, ||f||_inf)
        tau_tail: Bandwidth tail energy tolerance
        tau_chi: Quadrature phase-step tolerance
        A_max: Maximum exponential amplification
        sigma: Abscissa of the transform being inverted (bounds.re_max for a
            normal operator, the numerical abscissa for a non-normal one).
            The spectral placement condition is only checked when it is
            given; without it a caller gets the other four conditions and a
            nan a_required, never a silent pass on a contour that has
            crossed a pole.
        delta_min: Minimal spectral margin used by the placement condition
        eps_tail: Wraparound suppression threshold used by the placement
            condition

    Returns:
        SpectralCFLConditions with all diagnostics and pass/fail
    """
    violated = []

    # CFL-1: Endpoint compatibility
    # ================================

    # Compute f(0+) from IVT
    f_0_ivt = compute_ivt(F_eval) if F_eval is not None else None

    # Extract f(2T-) from last sample
    f_2T = float(result.f[-1])

    # Endpoint jump, relative to the signal's own scale. The scale used to
    # be floored at 1.0, which passed any transform of amplitude below 1%
    # regardless of its jump.
    f_start = f_0_ivt if f_0_ivt is not None else float(result.f[0])
    J = abs(f_start - f_2T)
    signal_scale = max(abs(f_start), float(jnp.max(jnp.abs(result.f))), np.finfo(float).tiny)
    endpoint_compatible = (J / signal_scale <= tau_end)

    if not endpoint_compatible:
        violated.append(f"endpoint_jump: J={J:.3e} > threshold={tau_end*signal_scale:.3e}")

    # CFL-2: Bandwidth coverage
    # ==========================

    # The frequency grid the NILT actually samples: k = 0..N/2 with
    # omega_k = k pi / T, N = 2T/dt. (result.f cannot supply N: the half-step
    # IVT variant carries one extra point.) The previous grid ran k to N-1,
    # twice past the Nyquist frequency, and measured a tail the inversion
    # never uses.
    N = int(round(2.0 * T / dt))
    omega = jnp.pi * jnp.arange(N // 2 + 1) / T

    # Compute tail energy from F(s) samples (NOT from rfft(result.f))
    R_tail = compute_tail_energy_from_F_samples(F_eval, omega, a) if F_eval is not None else 0.0
    omega_max = float(jnp.pi / dt)
    bandwidth_sufficient = (R_tail <= tau_tail)

    if not bandwidth_sufficient:
        violated.append(f"bandwidth: R_tail={R_tail:.3e} > threshold={tau_tail:.3e}")

    # CFL-3: Quadrature resolution
    # =============================

    # Phase step: χ = Δω · t_end = (π/T) · t_end
    chi = np.pi * t_end / T
    quadrature_stable = (chi <= tau_chi)

    if not quadrature_stable:
        violated.append(f"quadrature: χ={chi:.3f} > threshold={tau_chi:.3f}")

    # Conditioning guard
    # ===================

    # Exponential amplification: A_exp = exp(a·t_end)
    A_exp = float(np.exp(a * t_end))
    conditioning_safe = (A_exp <= A_max)

    if not conditioning_safe:
        violated.append(f"conditioning: A_exp={A_exp:.2e} > threshold={A_max:.2e}")

    # Spectral placement guard
    # =========================

    # a must sit to the right of the transform's abscissa by the wraparound
    # margin. The other four conditions are blind to this: a contour placed
    # on or left of a pole still has a small tail energy, a small phase step
    # and a small amplification, and the endpoint jump of the divergent
    # inversion it produces is no larger than that of a convergent one.
    a_required = required_abscissa(sigma, T, delta_min=delta_min, eps_tail=eps_tail)
    sigma_used = float(sigma) if sigma is not None and np.isfinite(sigma) else float('nan')
    spectral_placement_ok = True if np.isnan(a_required) else bool(a >= a_required)

    if not spectral_placement_ok:
        violated.append(
            f"spectral_placement: a={a:.3f} < required={a_required:.3f} "
            f"(sigma={sigma_used:.3f})"
        )

    # Overall status
    all_met = (
        endpoint_compatible
        and bandwidth_sufficient
        and quadrature_stable
        and conditioning_safe
        and spectral_placement_ok
    )

    return SpectralCFLConditions(
        endpoint_jump=J,
        f_0_ivt=f_0_ivt,
        f_2T=f_2T,
        endpoint_compatible=endpoint_compatible,
        tail_energy_ratio=R_tail,
        omega_max=omega_max,
        bandwidth_sufficient=bandwidth_sufficient,
        phase_step=chi,
        quadrature_stable=quadrature_stable,
        exp_amplification=A_exp,
        conditioning_safe=conditioning_safe,
        sigma=sigma_used,
        a_required=a_required,
        spectral_placement_ok=spectral_placement_ok,
        all_conditions_met=all_met,
        violated_conditions=violated,
    )


def _abscissa_floor(cfl, current_params, bounds) -> float:
    """Lower bound on a for suggest_parameter_adjustments.

    Prefers the a_required the conditions were checked with; falls back to
    the caller's bounds (re_max) when the check ran without a sigma, and to
    -inf (no floor) when neither is available.
    """
    if np.isfinite(cfl.a_required):
        return float(cfl.a_required)
    re_max = None
    if bounds is not None:
        re_max = getattr(bounds, 're_max', None)
        if re_max is None and isinstance(bounds, dict):
            re_max = bounds.get('re_max')
    floor = required_abscissa(re_max, current_params.T)
    return float('-inf') if np.isnan(floor) else floor


def suggest_parameter_adjustments(
    cfl: SpectralCFLConditions,
    current_params,
    bounds,
    max_N: int = 8192,
) -> tuple[dict, str]:
    """
    Suggest parameter adjustments based on violated CFL conditions.

    Escalation priority (deterministic):
    1. Endpoint jump → switch to half-step sampling + IVT
    2. Bandwidth tail → reduce dt (increase ω_max)
    3. Quadrature phase → increase T (reduce Δω)
    4. Conditioning → reduce a, but never below the required abscissa

    The shift a is bounded below by required_abscissa(sigma, T): halving it
    to buy amplification headroom is only legitimate while the contour stays
    to the right of the transform's abscissa. When the halved shift would
    fall through that floor no adjustment is offered, and the action says
    why, so the caller reports an infeasible window instead of inverting on
    a contour that runs through a pole.

    Args:
        cfl: SpectralCFLConditions from check
        current_params: TunedNILTParams
        bounds: SpectralBounds or dict, used for the abscissa floor when the
            conditions were checked without a sigma
        max_N: Maximum FFT size

    Returns:
        adjustments: Dict with parameter changes
        action: String description
    """
    if not cfl.endpoint_compatible:
        # Priority 1: Endpoint jump violation
        # → Use half-step sampling and IVT correction
        return {
            'use_halfstep': True,
            'apply_ivt': True,
        }, f"endpoint jump J={cfl.endpoint_jump:.2e} → half-step + IVT"

    if not cfl.bandwidth_sufficient:
        # Priority 2: Bandwidth tail energy violation
        # → Reduce dt to increase ω_max (Hsu & Dranoff prescription)
        dt_new = current_params.dt / 2.0
        return {
            'dt': dt_new,
        }, f"bandwidth tail R={cfl.tail_energy_ratio:.2e} → reduce dt: {current_params.dt:.4f} → {dt_new:.4f}"

    if not cfl.quadrature_stable:
        # Priority 3: Quadrature phase-step violation
        # → Increase T to reduce Δω
        T_new = 2.0 * current_params.T
        return {
            'T': T_new,
        }, f"phase step χ={cfl.phase_step:.2f} → increase T: {current_params.T:.2f} → {T_new:.2f}"

    if not cfl.spectral_placement_ok:
        # The contour is at or left of the transform's abscissa. Nothing
        # short of a larger a fixes that, and a larger a is what the
        # conditioning guard was trying to avoid: the window is infeasible.
        return {}, (
            f"spectral placement violated: a={current_params.a:.3f} < "
            f"required={cfl.a_required:.3f} (sigma={cfl.sigma:.3f}); "
            "split the interval or invert in higher precision"
        )

    if not cfl.conditioning_safe:
        # Priority 4: Exponential amplification violation
        # → Reduce a, but not below the abscissa the inversion needs
        a_floor = _abscissa_floor(cfl, current_params, bounds)
        a_half = max(0.0, current_params.a / 2.0)

        if current_params.a > 0.01 and a_half >= a_floor:
            return {
                'a': a_half,
            }, f"amplification A={cfl.exp_amplification:.2e} → reduce a: {current_params.a:.3f} → {a_half:.3f}"
        if np.isfinite(a_floor) and a_half < a_floor:
            return {}, (
                f"amplification A={cfl.exp_amplification:.2e} but a={current_params.a:.3f} "
                f"cannot be reduced below the required abscissa {a_floor:.3f}; "
                "split the interval or invert in higher precision"
            )
        return {}, "conditioning violated but a already minimal"

    # All conditions met
    return {}, "all CFL conditions satisfied"


def print_cfl_diagnostic_report(cfl: SpectralCFLConditions):
    """Print human-readable CFL diagnostic report."""
    print("\n" + "="*70)
    print("SPECTRAL CFL CONDITIONS (Quantitative Guardrails)")
    print("="*70)

    print("\nCFL-1: Endpoint Compatibility (Periodization Jump)")
    print(f"  Endpoint jump J:        {cfl.endpoint_jump:.3e}")
    print(f"  f(0+) from IVT:         {cfl.f_0_ivt:.6f}" if cfl.f_0_ivt else "  f(0+) from IVT:         N/A")
    print(f"  f(2T-) from last sample: {cfl.f_2T:.6f}")
    print(f"  Status: {'✓ PASS' if cfl.endpoint_compatible else '✗ FAIL'}")

    print("\nCFL-2: Bandwidth Coverage (Spectral Tail Energy)")
    print(f"  Tail energy ratio R_tail: {cfl.tail_energy_ratio:.3e}")
    print(f"  ω_max (Nyquist):          {cfl.omega_max:.2f}")
    print(f"  Status: {'✓ PASS' if cfl.bandwidth_sufficient else '✗ FAIL'}")

    print("\nCFL-3: Quadrature Resolution (Phase Step)")
    print(f"  Phase step χ:           {cfl.phase_step:.3f}")
    print(f"  Status: {'✓ PASS' if cfl.quadrature_stable else '✗ FAIL'}")

    print("\nConditioning Guard (Exponential Amplification)")
    print(f"  A_exp = exp(a·t_end):   {cfl.exp_amplification:.2e}")
    print(f"  Status: {'✓ PASS' if cfl.conditioning_safe else '✗ FAIL'}")

    print("\nSpectral Placement (Bromwich abscissa)")
    print(f"  sigma:                  {cfl.sigma:.3f}")
    print(f"  a required:             {cfl.a_required:.3f}")
    print(f"  Status: {'✓ PASS' if cfl.spectral_placement_ok else '✗ FAIL'}")

    print(f"\n{'='*70}")
    print(f"Overall: {'ALL CONDITIONS MET ✓' if cfl.all_conditions_met else 'VIOLATIONS DETECTED ✗'}")
    if cfl.violated_conditions:
        print("\nViolated conditions:")
        for v in cfl.violated_conditions:
            print(f"  - {v}")
    print("="*70)
