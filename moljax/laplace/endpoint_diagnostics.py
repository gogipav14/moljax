"""
Principled spectral CFL-like guardrails for FFT-NILT.

This module implements quantitative stability conditions analogous to CFL
constraints, replacing qualitative ε_Im quality tiers with measurable
frequency-domain constraints:

1. Endpoint Compatibility: Periodization jump condition
2. Bandwidth Coverage: Nyquist-like tail energy from F(s) samples
3. Quadrature Resolution: Phase-step condition for oscillatory integral
4. Conditioning Guard: Exponential amplification bound

Based on the interpretation of FFT-NILT as trapezoidal quadrature of the
Bromwich integral with periodic extension (Hsu & Dranoff 1987, Weeks 1966).
"""

from __future__ import annotations

from typing import NamedTuple, Optional
import jax.numpy as jnp
import numpy as np


class SpectralCFLConditions(NamedTuple):
    """Quantitative CFL-like conditions for FFT-NILT."""

    # CFL-1: Endpoint compatibility
    endpoint_jump: float  # J = |f(0+) - f(2T-)|
    f_0_ivt: Optional[float]  # f(0+) from Initial Value Theorem
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
    except:
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
    # Evaluate F at all frequency points
    s_grid = a + 1j * omega_grid
    F_samples = jnp.array([F_eval(s) for s in s_grid])

    # Energy (squared magnitude)
    energy = jnp.abs(F_samples)**2
    total_energy = jnp.sum(energy)

    # Tail: top tail_fraction of frequencies
    n_tail = max(1, int(len(omega_grid) * tail_fraction))
    tail_energy = jnp.sum(energy[-n_tail:])

    # Ratio
    R_tail = float(tail_energy / (total_energy + 1e-30))

    return R_tail


def check_spectral_cfl_conditions(
    result,
    F_eval,
    t_end: float,
    a: float,
    T: float,
    dt: float,
    tau_end: float = 0.01,  # Endpoint jump tolerance (1%)
    tau_tail: float = 1e-4,  # Tail energy tolerance (0.01%)
    tau_chi: float = 0.3,  # Phase-step tolerance
    A_max: float = 1e6,  # Maximum exp amplification (float64 safe)
) -> SpectralCFLConditions:
    """
    Check all spectral CFL-like conditions for FFT-NILT.

    Args:
        result: NILTResult from nilt_fft_uniform
        F_eval: Laplace-domain function F(s)
        t_end: Integration end time
        a: Bromwich shift
        T: Half-period (2T = period)
        dt: Time step
        tau_end: Endpoint jump tolerance (relative to signal)
        tau_tail: Bandwidth tail energy tolerance
        tau_chi: Quadrature phase-step tolerance
        A_max: Maximum exponential amplification

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

    # Compute endpoint jump
    if f_0_ivt is not None:
        J = abs(f_0_ivt - f_2T)
        # Normalize by signal scale
        signal_scale = max(1.0, abs(f_0_ivt))
        J_relative = J / signal_scale
        endpoint_compatible = (J_relative <= tau_end)
    else:
        # Fallback: use first sample as f(0) estimate
        J = abs(result.f[0] - f_2T)
        signal_scale = max(1.0, abs(result.f[0]))
        J_relative = J / signal_scale
        endpoint_compatible = (J_relative <= tau_end)

    if not endpoint_compatible:
        violated.append(f"endpoint_jump: J={J:.3e} > threshold={tau_end*signal_scale:.3e}")

    # CFL-2: Bandwidth coverage
    # ==========================

    # Construct frequency grid (same as used in NILT)
    N = len(result.f)
    omega = jnp.pi * jnp.arange(N) / T

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

    # Overall status
    all_met = endpoint_compatible and bandwidth_sufficient and quadrature_stable and conditioning_safe

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
        all_conditions_met=all_met,
        violated_conditions=violated,
    )


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
    4. Conditioning → reduce a (if feasible)

    Args:
        cfl: SpectralCFLConditions from check
        current_params: TunedNILTParams
        bounds: SpectralBounds
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

    if not cfl.conditioning_safe:
        # Priority 4: Exponential amplification violation
        # → Reduce a if feasible
        if current_params.a > 0.01:
            a_new = max(0.0, current_params.a / 2.0)
            return {
                'a': a_new,
            }, f"amplification A={cfl.exp_amplification:.2e} → reduce a: {current_params.a:.3f} → {a_new:.3f}"
        else:
            return {}, "conditioning violated but a already minimal"

    # All conditions met
    return {}, "all CFL conditions satisfied"


def print_cfl_diagnostic_report(cfl: SpectralCFLConditions):
    """Print human-readable CFL diagnostic report."""
    print("\n" + "="*70)
    print("SPECTRAL CFL CONDITIONS (Quantitative Guardrails)")
    print("="*70)

    print(f"\nCFL-1: Endpoint Compatibility (Periodization Jump)")
    print(f"  Endpoint jump J:        {cfl.endpoint_jump:.3e}")
    print(f"  f(0+) from IVT:         {cfl.f_0_ivt:.6f}" if cfl.f_0_ivt else "  f(0+) from IVT:         N/A")
    print(f"  f(2T-) from last sample: {cfl.f_2T:.6f}")
    print(f"  Status: {'✓ PASS' if cfl.endpoint_compatible else '✗ FAIL'}")

    print(f"\nCFL-2: Bandwidth Coverage (Spectral Tail Energy)")
    print(f"  Tail energy ratio R_tail: {cfl.tail_energy_ratio:.3e}")
    print(f"  ω_max (Nyquist):          {cfl.omega_max:.2f}")
    print(f"  Status: {'✓ PASS' if cfl.bandwidth_sufficient else '✗ FAIL'}")

    print(f"\nCFL-3: Quadrature Resolution (Phase Step)")
    print(f"  Phase step χ:           {cfl.phase_step:.3f}")
    print(f"  Status: {'✓ PASS' if cfl.quadrature_stable else '✗ FAIL'}")

    print(f"\nConditioning Guard (Exponential Amplification)")
    print(f"  A_exp = exp(a·t_end):   {cfl.exp_amplification:.2e}")
    print(f"  Status: {'✓ PASS' if cfl.conditioning_safe else '✗ FAIL'}")

    print(f"\n{'='*70}")
    print(f"Overall: {'ALL CONDITIONS MET ✓' if cfl.all_conditions_met else 'VIOLATIONS DETECTED ✗'}")
    if cfl.violated_conditions:
        print(f"\nViolated conditions:")
        for v in cfl.violated_conditions:
            print(f"  - {v}")
    print("="*70)
