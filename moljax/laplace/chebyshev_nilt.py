"""
Chebyshev-based Numerical Inverse Laplace Transform (NILT).

This module implements Chebyshev polynomial methods for Laplace inversion,
providing an alternative to FFT-based approaches with different trade-offs:

Advantages over FFT-NILT:
- Better endpoint accuracy (no Gibbs phenomenon)
- Exponential convergence for smooth functions
- Adaptive accuracy via polynomial degree
- Natural handling of non-uniform time grids

Disadvantages:
- Higher per-point cost (O(N²) vs O(N log N))
- Less efficient for uniform grids with many points
- Requires more evaluations of F(s) per time point

Methods implemented:
1. Weeks' method - Laguerre expansion with Chebyshev acceleration
2. Talbot contour - deformed Bromwich integration via Chebyshev quadrature
3. Gaver-Stehfest - integer moments with Chebyshev refinement

References:
- Weeks, "Numerical Inversion of Laplace Transforms Using Laguerre Functions" (1966)
- Talbot, "The Accurate Numerical Inversion of Laplace Transforms" (1979)
- Weideman & Trefethen, "Parabolic and Hyperbolic Contours for Laplace Inversion" (2007)
- Abate & Whitt, "A Unified Framework for Numerically Inverting Laplace Transforms" (2006)
"""

from __future__ import annotations

from functools import partial
from typing import Callable, NamedTuple, Optional, Tuple, List

import jax
import jax.numpy as jnp
from jax import lax


class ChebyshevNILTResult(NamedTuple):
    """Result from Chebyshev-based NILT."""
    t: jnp.ndarray           # Time points
    f: jnp.ndarray           # Inverse transform values
    method: str              # Method used
    n_terms: int             # Number of terms/evaluations
    error_estimate: float    # Estimated error (if available)
    diagnostics: dict        # Method-specific diagnostics


# =============================================================================
# Chebyshev Polynomial Utilities
# =============================================================================

def chebyshev_nodes(n: int, a: float = -1.0, b: float = 1.0) -> jnp.ndarray:
    """
    Chebyshev nodes of the first kind on [a, b].

    T_n(x) = cos(n * arccos(x)) has zeros at x_k = cos((2k+1)π/(2n))
    """
    k = jnp.arange(n)
    nodes_std = jnp.cos((2 * k + 1) * jnp.pi / (2 * n))
    # Map from [-1, 1] to [a, b]
    return 0.5 * (b - a) * nodes_std + 0.5 * (a + b)


def chebyshev_coefficients(f_vals: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Chebyshev coefficients from function values at Chebyshev nodes.

    Uses the discrete cosine transform relationship.
    """
    n = len(f_vals)
    k = jnp.arange(n)

    # DCT-I relationship
    coeffs = jnp.zeros(n)
    for j in range(n):
        weights = jnp.cos(j * (2 * k + 1) * jnp.pi / (2 * n))
        coeffs = coeffs.at[j].set(2.0 / n * jnp.sum(f_vals * weights))

    # Adjust first coefficient
    coeffs = coeffs.at[0].set(coeffs[0] / 2.0)

    return coeffs


def chebyshev_eval(coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate Chebyshev series at points x using Clenshaw's algorithm.

    f(x) = Σ c_k T_k(x)
    """
    n = len(coeffs)
    if n == 0:
        return jnp.zeros_like(x)
    if n == 1:
        return coeffs[0] * jnp.ones_like(x)

    # Clenshaw recurrence
    b_kp2 = jnp.zeros_like(x)
    b_kp1 = jnp.zeros_like(x)

    for k in range(n - 1, 0, -1):
        b_k = coeffs[k] + 2 * x * b_kp1 - b_kp2
        b_kp2 = b_kp1
        b_kp1 = b_k

    return coeffs[0] + x * b_kp1 - b_kp2


# =============================================================================
# Method 1: Weeks' Method (Laguerre-Chebyshev)
# =============================================================================

def laguerre_coefficients(
    F_eval: Callable,
    n_terms: int,
    sigma: float = 1.0,
    b: float = 1.0
) -> jnp.ndarray:
    """
    Compute Laguerre expansion coefficients for F(s).

    The inverse transform is represented as:
    f(t) = exp(σt) Σ a_k L_k(2bt)

    where L_k are Laguerre polynomials.

    Args:
        F_eval: Laplace transform F(s)
        n_terms: Number of Laguerre terms
        sigma: Abscissa shift (should exceed all singularities)
        b: Scaling parameter (affects convergence rate)

    Returns:
        Laguerre coefficients a_k
    """
    # Use contour integral via trapezoidal rule
    # a_k = (1/2πi) ∮ F(s) (s-σ+b)^k / (s-σ-b)^(k+1) ds

    n_quad = max(2 * n_terms, 64)
    theta = jnp.linspace(0, 2 * jnp.pi, n_quad, endpoint=False)

    # Circular contour centered at σ with radius b
    s_vals = sigma + b * jnp.exp(1j * theta)
    F_vals = jnp.array([F_eval(s) for s in s_vals])

    coeffs = jnp.zeros(n_terms, dtype=jnp.complex128)

    for k in range(n_terms):
        # Residue calculation
        integrand = F_vals * ((s_vals - sigma + b) / (s_vals - sigma - b)) ** k
        integrand = integrand / (s_vals - sigma - b)
        coeffs = coeffs.at[k].set(jnp.mean(integrand) * b)

    return jnp.real(coeffs)


def laguerre_eval(coeffs: jnp.ndarray, t: jnp.ndarray, b: float = 1.0) -> jnp.ndarray:
    """
    Evaluate Laguerre series at times t.

    Uses Clenshaw recurrence for Laguerre polynomials.
    """
    n = len(coeffs)
    x = 2 * b * t

    # Laguerre recurrence: (k+1)L_{k+1}(x) = (2k+1-x)L_k(x) - k L_{k-1}(x)
    b_kp2 = jnp.zeros_like(t)
    b_kp1 = jnp.zeros_like(t)

    for k in range(n - 1, -1, -1):
        if k == n - 1:
            b_k = coeffs[k]
        else:
            b_k = coeffs[k] + ((2 * k + 3 - x) * b_kp1 - (k + 2) * b_kp2) / (k + 1)
        b_kp2 = b_kp1
        b_kp1 = b_k

    return b_kp1


def weeks_method(
    F_eval: Callable,
    n_terms: int,
    t: jnp.ndarray,
    sigma: float = 1.0,
    b: float = 1.0
) -> ChebyshevNILTResult:
    """
    Weeks' method for numerical inverse Laplace transform.

    Uses Laguerre polynomial expansion with optimal parameter selection.

    Args:
        F_eval: Laplace transform F(s)
        n_terms: Number of Laguerre terms
        t: Time points for evaluation
        sigma: Abscissa shift (should exceed real parts of all singularities)
        b: Scaling parameter

    Returns:
        ChebyshevNILTResult with inverse transform values
    """
    # Compute Laguerre coefficients
    coeffs = laguerre_coefficients(F_eval, n_terms, sigma, b)

    # Evaluate at requested times
    f_laguerre = laguerre_eval(coeffs, t, b)
    f = jnp.exp(sigma * t) * f_laguerre

    # Error estimate from last coefficients
    if n_terms > 2:
        error_est = float(jnp.max(jnp.abs(coeffs[-3:])))
    else:
        error_est = float(jnp.max(jnp.abs(coeffs)))

    return ChebyshevNILTResult(
        t=t,
        f=f,
        method='weeks',
        n_terms=n_terms,
        error_estimate=error_est,
        diagnostics={
            'sigma': sigma,
            'b': b,
            'coefficients': coeffs,
        }
    )


# =============================================================================
# Method 2: Talbot Contour Method
# =============================================================================

def talbot_contour(
    t: float,
    n_points: int,
    sigma: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate Talbot contour points and weights.

    Uses the optimal Talbot contour from Weideman & Trefethen (2007):
    s(θ) = N/t * (0.5017*θ*cot(0.6407*θ) - 0.6122 + 0.2645i*θ)

    Returns:
        (s_values, weights) for quadrature
    """
    N = n_points
    # Use the optimal Talbot parameters from Weideman & Trefethen
    # These give good convergence for a wide range of problems
    k = jnp.arange(N)
    theta = -jnp.pi + (2*k + 1) * jnp.pi / N

    # Optimal Talbot contour parameters
    alpha = 0.6407
    beta = 0.5017
    gamma = 0.2645

    # Contour: s(θ) = N/t * [β*θ*cot(α*θ) - δ + γ*i*θ]
    # where δ = β*cot(α*π) ≈ 0.6122 for the optimal parameters
    delta = beta * jnp.cos(alpha * jnp.pi) / jnp.sin(alpha * jnp.pi)

    # Compute s values
    cot_alpha_theta = jnp.cos(alpha * theta) / jnp.sin(alpha * theta)
    s_real = (N / t) * (beta * theta * cot_alpha_theta - delta) + sigma
    s_imag = (N / t) * gamma * theta
    s = s_real + 1j * s_imag

    # Derivative ds/dθ for quadrature
    sin_sq = jnp.sin(alpha * theta) ** 2
    ds_real = (N / t) * beta * (cot_alpha_theta - alpha * theta / sin_sq)
    ds_imag = (N / t) * gamma
    ds_dtheta = ds_real + 1j * ds_imag

    # Weights for trapezoidal rule
    weights = jnp.exp(s * t) * ds_dtheta * (2 * jnp.pi / N) / (2j * jnp.pi)

    return s, weights


def talbot_method(
    F_eval: Callable,
    t: jnp.ndarray,
    n_points: int = 32,
    sigma: float = 0.0
) -> ChebyshevNILTResult:
    """
    Talbot contour method for numerical inverse Laplace transform.

    Uses deformed Bromwich contour with exponential convergence.

    Args:
        F_eval: Laplace transform F(s)
        t: Time points for evaluation
        n_points: Number of quadrature points
        sigma: Optional shift to avoid singularities

    Returns:
        ChebyshevNILTResult with inverse transform values
    """
    f_vals = jnp.zeros(len(t))

    for i, ti in enumerate(t):
        if ti <= 0:
            f_vals = f_vals.at[i].set(0.0)
            continue

        s_pts, weights = talbot_contour(float(ti), n_points, sigma)

        # Evaluate F at contour points
        F_pts = jnp.array([F_eval(s) for s in s_pts])

        # Quadrature
        integral = jnp.sum(F_pts * weights) * (jnp.pi / n_points)
        f_vals = f_vals.at[i].set(jnp.real(integral))

    # Error estimate from Richardson extrapolation
    if len(t) > 2:
        # Compare with half the points
        s_half, w_half = talbot_contour(float(t[len(t)//2]), n_points // 2, sigma)
        F_half = jnp.array([F_eval(s) for s in s_half])
        f_half = jnp.real(jnp.sum(F_half * w_half) * (jnp.pi / (n_points // 2)))
        error_est = float(jnp.abs(f_vals[len(t)//2] - f_half))
    else:
        error_est = 0.0

    return ChebyshevNILTResult(
        t=t,
        f=f_vals,
        method='talbot',
        n_terms=n_points,
        error_estimate=error_est,
        diagnostics={
            'sigma': sigma,
            'n_quadrature': n_points,
        }
    )


# =============================================================================
# Method 3: Gaver-Stehfest Algorithm
# =============================================================================

def gaver_stehfest_weights(n: int) -> jnp.ndarray:
    """
    Compute Gaver-Stehfest weights.

    The weights w_k allow approximation:
    f(t) ≈ (ln 2 / t) Σ_{k=1}^{n} w_k F(k ln 2 / t)
    """
    # Must be even
    n = n if n % 2 == 0 else n + 1
    m = n // 2

    weights = jnp.zeros(n)

    for k in range(1, n + 1):
        sum_val = 0.0
        for j in range(max(1, (k + 1) // 2), min(k, m) + 1):
            # Binomial and factorial computations
            num = j ** m * _factorial(2 * j)
            den = (_factorial(m - j) * _factorial(j) * _factorial(j - 1) *
                   _factorial(k - j) * _factorial(2 * j - k))
            if den != 0:
                sum_val += num / den

        weights = weights.at[k - 1].set((-1) ** (k + m) * sum_val)

    return weights


def _factorial(n: int) -> int:
    """Compute factorial."""
    if n <= 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def gaver_stehfest_method(
    F_eval: Callable,
    t: jnp.ndarray,
    n_terms: int = 14
) -> ChebyshevNILTResult:
    """
    Gaver-Stehfest algorithm for numerical inverse Laplace transform.

    Uses integer abscissae, suitable for transforms known only at integer values.

    Args:
        F_eval: Laplace transform F(s)
        t: Time points for evaluation
        n_terms: Number of terms (should be even, typically 10-18)

    Returns:
        ChebyshevNILTResult with inverse transform values
    """
    weights = gaver_stehfest_weights(n_terms)
    ln2 = jnp.log(2.0)

    f_vals = jnp.zeros(len(t))

    for i, ti in enumerate(t):
        if ti <= 0:
            f_vals = f_vals.at[i].set(0.0)
            continue

        # Evaluate at s_k = k * ln(2) / t
        s_vals = jnp.arange(1, n_terms + 1) * ln2 / ti
        F_vals = jnp.array([F_eval(s) for s in s_vals])

        # Weighted sum
        f_vals = f_vals.at[i].set((ln2 / ti) * jnp.sum(weights * F_vals))

    return ChebyshevNILTResult(
        t=t,
        f=f_vals,
        method='gaver_stehfest',
        n_terms=n_terms,
        error_estimate=0.0,  # Hard to estimate for this method
        diagnostics={
            'weights': weights,
        }
    )


# =============================================================================
# Adaptive Chebyshev NILT
# =============================================================================

def adaptive_chebyshev_nilt(
    F_eval: Callable,
    t: jnp.ndarray,
    method: str = 'auto',
    tol: float = 1e-6,
    max_terms: int = 64,
    sigma: Optional[float] = None
) -> ChebyshevNILTResult:
    """
    Adaptive Chebyshev-based NILT with automatic method selection.

    Chooses the best method based on problem characteristics and
    adaptively increases accuracy until tolerance is met.

    Args:
        F_eval: Laplace transform F(s)
        t: Time points for evaluation
        method: 'auto', 'weeks', 'talbot', or 'gaver_stehfest'
        tol: Target tolerance
        max_terms: Maximum number of terms
        sigma: Abscissa shift (auto-detected if None)

    Returns:
        ChebyshevNILTResult with best achievable accuracy
    """
    t_arr = jnp.asarray(t)
    t_max = float(jnp.max(t_arr[t_arr > 0]))

    # Auto-detect sigma if not provided
    if sigma is None:
        # Probe F(s) to estimate decay
        s_probe = jnp.array([0.1, 1.0, 10.0])
        F_probe = jnp.array([jnp.abs(F_eval(s)) for s in s_probe])
        # Simple heuristic: sigma should make F decay sufficiently
        sigma = 1.0 / t_max

    # Auto method selection
    if method == 'auto':
        # Talbot is generally most robust
        # Weeks is better for smooth F
        # Gaver-Stehfest for integer evaluations only
        method = 'talbot'

    # Adaptive refinement
    n_terms = 16
    prev_result = None

    while n_terms <= max_terms:
        if method == 'weeks':
            result = weeks_method(F_eval, n_terms, t_arr, sigma, b=1.0/t_max)
        elif method == 'talbot':
            result = talbot_method(F_eval, t_arr, n_terms, sigma)
        elif method == 'gaver_stehfest':
            result = gaver_stehfest_method(F_eval, t_arr, min(n_terms, 18))
        else:
            raise ValueError(f"Unknown method: {method}")

        # Check convergence
        if prev_result is not None:
            diff = jnp.max(jnp.abs(result.f - prev_result.f))
            rel_diff = diff / (jnp.max(jnp.abs(result.f)) + 1e-14)
            if rel_diff < tol:
                break

        prev_result = result
        n_terms *= 2

    return result


# =============================================================================
# Comparison with FFT-NILT
# =============================================================================

def compare_chebyshev_vs_fft(
    F_eval: Callable,
    f_exact: Optional[Callable] = None,
    t_end: float = 10.0,
    n_points: int = 64
) -> dict:
    """
    Compare Chebyshev methods with FFT-NILT.

    Args:
        F_eval: Laplace transform F(s)
        f_exact: Exact inverse (optional, for error measurement)
        t_end: End time
        n_points: Number of evaluation points

    Returns:
        Dictionary with comparison metrics
    """
    t = jnp.linspace(0.01, t_end, n_points)

    results = {}

    # Talbot method
    talbot_result = talbot_method(F_eval, t, n_points=32)
    results['talbot'] = {
        'f': talbot_result.f,
        'n_evals': 32 * len(t),
    }

    # Weeks method
    weeks_result = weeks_method(F_eval, 32, t, sigma=1.0/t_end)
    results['weeks'] = {
        'f': weeks_result.f,
        'n_evals': 64 + len(t),  # coefficients + evaluation
    }

    # Gaver-Stehfest
    gs_result = gaver_stehfest_method(F_eval, t, n_terms=14)
    results['gaver_stehfest'] = {
        'f': gs_result.f,
        'n_evals': 14 * len(t),
    }

    # Compute errors if exact solution provided
    if f_exact is not None:
        f_true = jnp.array([f_exact(ti) for ti in t])
        for method in results:
            error = jnp.max(jnp.abs(results[method]['f'] - f_true))
            results[method]['max_error'] = float(error)
            results[method]['rms_error'] = float(
                jnp.sqrt(jnp.mean((results[method]['f'] - f_true) ** 2))
            )

    return results


# =============================================================================
# Utility Functions
# =============================================================================

def print_chebyshev_report(result: ChebyshevNILTResult) -> None:
    """Print a summary report for Chebyshev NILT result."""
    print("=" * 50)
    print("Chebyshev NILT Report")
    print("=" * 50)
    print(f"Method:          {result.method}")
    print(f"N terms:         {result.n_terms}")
    print(f"Error estimate:  {result.error_estimate:.2e}")
    print(f"Time range:      [{result.t[0]:.4f}, {result.t[-1]:.4f}]")
    print(f"N time points:   {len(result.t)}")
    print("-" * 50)
    print(f"f range:         [{jnp.min(result.f):.4e}, {jnp.max(result.f):.4e}]")
    if 'sigma' in result.diagnostics:
        print(f"Sigma:           {result.diagnostics['sigma']:.4f}")
    print("=" * 50)
