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
1. Weeks' method - Laguerre-function expansion, coefficients by FFT on the
   Mobius image of the unit circle (Weideman 1999)
2. Talbot contour - the fixed Talbot contour with the trapezoidal rule
   (Weideman & Trefethen 2007)
3. Gaver-Stehfest - integer abscissae with exact rational weights

References:
- Weeks, "Numerical Inversion of Laplace Transforms Using Laguerre Functions" (1966)
- Weideman, "Algorithms for Parameter Selection in the Weeks Method for
  Inverting the Laplace Transform", SIAM J. Sci. Comput. 21 (1999)
- Talbot, "The Accurate Numerical Inversion of Laplace Transforms" (1979)
- Weideman & Trefethen, "Parabolic and Hyperbolic Contours for Computing the
  Bromwich Integral", Math. Comp. 76 (2007)
- Abate & Whitt, "A Unified Framework for Numerically Inverting Laplace Transforms" (2006)
"""

from __future__ import annotations

from collections.abc import Callable
from fractions import Fraction
from math import factorial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from moljax._precision import require_x64


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
    Laguerre coefficients a_n, n = 0..n_terms-1, of Weeks' expansion

        f(t) = e^{σt} Σ_n a_n e^{-bt} L_n(2bt).

    Following Weideman (1999): the Mobius map s = σ + b (1 + z)/(1 - z)
    sends the unit circle to the line Re s = σ, and the a_n are the Taylor
    coefficients on the unit disk of

        G(z) = 2b F(s(z)) / (1 - z).

    They are computed by the trapezoidal rule on the midpoint grid
    z_j = e^{iθ_j}, θ_j = (j + 1/2) 2π/M - π, M = 2 n_terms, which avoids
    z = 1 (s = infinity) and turns the sum into an FFT:

        a_n = Re[(1/M) Σ_j G(z_j) z_j^{-n}].

    The previous implementation integrated on a circle in the s-plane
    through the point s = σ + b, where its integrand has a pole, and every
    coefficient came out NaN.

    Args:
        F_eval: Laplace transform F(s), evaluated on an array of s
        n_terms: Number of Laguerre terms
        sigma: Abscissa of the line Re s = σ; must exceed the real part of
            every singularity of F
        b: Scaling parameter (affects convergence rate)

    Returns:
        Real Laguerre coefficients a_n, shape (n_terms,)
    """
    M = 2 * n_terms
    j = jnp.arange(M)
    theta = (j + 0.5) * (2 * jnp.pi / M) - jnp.pi
    z = jnp.exp(1j * theta)
    s = sigma + b * (1 + z) / (1 - z)
    G = 2 * b * F_eval(s) / (1 - z)

    # Σ_j G_j z_j^{-n} = Σ_j G_j e^{-inθ_j} = e^{in(π - π/M)} FFT(G)[n]
    n = jnp.arange(n_terms)
    coeffs = jnp.fft.fft(G)[:n_terms] * jnp.exp(1j * n * (jnp.pi - jnp.pi / M)) / M
    return jnp.real(coeffs)


def laguerre_eval(coeffs: jnp.ndarray, t: jnp.ndarray, b: float = 1.0) -> jnp.ndarray:
    """
    Evaluate the Laguerre-function series Σ_n a_n e^{-bt} L_n(2bt) at times t.

    Weeks' basis functions are the Laguerre functions e^{-x/2} L_n(x) with
    x = 2bt, not the polynomials alone; the e^{-bt} factor was missing
    before. The polynomials come from the three-term recurrence
    (n + 1) L_{n+1}(x) = (2n + 1 - x) L_n(x) - n L_{n-1}(x) run forward.
    """
    x = 2.0 * b * t
    n = len(coeffs)

    L_prev = jnp.zeros_like(x)
    L = jnp.ones_like(x)
    total = coeffs[0] * L
    for k in range(1, n):
        L_next = ((2 * k - 1 - x) * L - (k - 1) * L_prev) / k
        L_prev, L = L, L_next
        total = total + coeffs[k] * L

    return jnp.exp(-b * t) * total


def weeks_method(
    F_eval: Callable,
    n_terms: int,
    t: jnp.ndarray,
    sigma: float = 1.0,
    b: float = 1.0
) -> ChebyshevNILTResult:
    """
    Weeks' method for numerical inverse Laplace transform.

    f(t) = e^{σt} Σ_n a_n e^{-bt} L_n(2bt), with the a_n from
    laguerre_coefficients (Weideman 1999). For 1/(s+1) with σ = 0.5, b = 1
    and 32 terms the error is at rounding level.

    Args:
        F_eval: Laplace transform F(s), evaluated on an array of s
        n_terms: Number of Laguerre terms
        t: Time points for evaluation
        sigma: Abscissa of the Weeks line; must exceed the real part of
            every singularity of F
        b: Scaling parameter

    Returns:
        ChebyshevNILTResult with inverse transform values
    """
    t = jnp.asarray(t)
    coeffs = laguerre_coefficients(F_eval, n_terms, sigma, b)
    f = jnp.exp(sigma * t) * laguerre_eval(coeffs, t, b)

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
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Talbot contour points and trapezoidal weights for one time t.

    The fixed Talbot contour in the optimized form of Weideman & Trefethen
    (2007),

        s(θ) = (N/t) (-0.6122 + 0.5017 θ cot(0.6407 θ) + 0.2645 i θ) + σ,

    sampled at the midpoints θ_k = -π + (2k + 1) π/N. The trapezoidal rule
    for (1/2πi) ∫ e^{st} F(s) s'(θ) dθ gives

        w_k = e^{s_k t} s'(θ_k) / (iN),   f(t) = Re Σ_k w_k F(s_k),

    with no further factor: the 2π/N quadrature step and the 1/(2πi) of the
    Bromwich integral are already in w_k. The previous implementation used
    β cot(απ) = -0.24 in place of the paper's -0.6122 and its caller
    multiplied the sum by another π/N; together they returned 0.098 of the
    true value at N = 32 and diverged at N = 64.

    Args:
        t: Time (positive scalar; may be traced)
        n_points: Number of contour points N
        sigma: Optional shift of the whole contour to the right

    Returns:
        (s_values, weights), each of shape (N,)
    """
    N = n_points
    k = jnp.arange(N)
    theta = -jnp.pi + (2 * k + 1) * jnp.pi / N

    alpha, beta, gamma, delta = 0.6407, 0.5017, 0.2645, 0.6122

    cot = jnp.cos(alpha * theta) / jnp.sin(alpha * theta)
    s = (N / t) * (beta * theta * cot - delta + 1j * gamma * theta) + sigma
    ds_dtheta = (N / t) * (beta * (cot - alpha * theta / jnp.sin(alpha * theta) ** 2) + 1j * gamma)

    weights = jnp.exp(s * t) * ds_dtheta / (1j * N)
    return s, weights


def talbot_method(
    F_eval: Callable,
    t: jnp.ndarray,
    n_points: int = 32,
    sigma: float = 0.0
) -> ChebyshevNILTResult:
    """
    Talbot contour method for numerical inverse Laplace transform.

    Trapezoidal rule on the Weideman-Trefethen contour, vectorized over
    time points with jax.vmap; F is evaluated on the N contour points of
    each time in one call. Converges geometrically for transforms analytic
    off the negative real axis: exp(-t) to 6e-10 at N = 16 and 5e-14 at
    N = 32.

    Args:
        F_eval: Laplace transform F(s), evaluated on an array of s
        t: Time points for evaluation (t <= 0 returns 0)
        n_points: Number of quadrature points
        sigma: Optional shift of the contour to the right

    Returns:
        ChebyshevNILTResult with inverse transform values; error_estimate is
        the difference from the same quadrature with N/2 points at the
        middle time.
    """
    t = jnp.atleast_1d(jnp.asarray(t))
    positive = t > 0
    t_safe = jnp.where(positive, t, 1.0)

    def invert_at(ti, n):
        s, w = talbot_contour(ti, n, sigma)
        return jnp.real(jnp.sum(F_eval(s) * w))

    f_vals = jnp.where(positive, jax.vmap(lambda ti: invert_at(ti, n_points))(t_safe), 0.0)

    if len(t) > 2 and n_points >= 4:
        mid = len(t) // 2
        error_est = float(jnp.abs(f_vals[mid] - invert_at(t_safe[mid], n_points // 2)))
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

def gaver_stehfest_weights(n: int) -> np.ndarray:
    """
    Gaver-Stehfest weights V_k, k = 1..n (n even; an odd n is rounded up),
    as a float64 numpy array.

        f(t) ≈ (ln 2 / t) Σ_{k=1}^{n} V_k F(k ln 2 / t)

    The weights are alternating integers that reach 1.7e8 at n = 14, and
    the sum cancels down to an O(1) value, so they are formed exactly in
    fractions.Fraction and rounded once. Forming them in floating point and
    storing them as a JAX array made them float32 whenever x64 was off, and
    the inversion then returned values off by 0.65 to 1.28 with no warning.
    """
    n = n if n % 2 == 0 else n + 1
    m = n // 2

    weights = np.zeros(n, dtype=np.float64)
    for k in range(1, n + 1):
        total = Fraction(0)
        for j in range((k + 1) // 2, min(k, m) + 1):
            num = Fraction(j ** m * factorial(2 * j))
            den = (factorial(m - j) * factorial(j) * factorial(j - 1) *
                   factorial(k - j) * factorial(2 * j - k))
            total += num / den
        weights[k - 1] = float((-1) ** (k + m) * total)

    return weights


def gaver_stehfest_method(
    F_eval: Callable,
    t: jnp.ndarray,
    n_terms: int = 14
) -> ChebyshevNILTResult:
    """
    Gaver-Stehfest algorithm for numerical inverse Laplace transform.

    Uses real abscissae only, so it suits transforms that can only be
    evaluated on the real axis. It needs 64-bit precision: the weights
    alternate in sign with magnitudes near 1e8 and cancel to an O(1) result,
    which float32 cannot carry (errors of 0.65 to 1.28 on exp(-t)). With
    x64 enabled 1/(s+1) at t = 1 comes out within 1e-6 at 14 terms.

    Args:
        F_eval: Laplace transform F(s), evaluated on an array of s
        t: Time points for evaluation (t <= 0 returns 0)
        n_terms: Number of terms (even, typically 10-18; odd is rounded up)

    Returns:
        ChebyshevNILTResult with inverse transform values

    Raises:
        RuntimeError: If JAX is not running with 64-bit precision.
    """
    require_x64("Gaver-Stehfest inversion")

    weights_np = gaver_stehfest_weights(n_terms)
    weights = jnp.asarray(weights_np)
    n = len(weights_np)
    ln2 = float(np.log(2.0))

    t = jnp.atleast_1d(jnp.asarray(t, dtype=jnp.float64))
    positive = t > 0
    t_safe = jnp.where(positive, t, 1.0)
    k = jnp.arange(1, n + 1, dtype=jnp.float64)

    def invert_at(ti):
        return (ln2 / ti) * jnp.sum(weights * F_eval(k * ln2 / ti))

    f_vals = jnp.where(positive, jax.vmap(invert_at)(t_safe), 0.0)

    return ChebyshevNILTResult(
        t=t,
        f=f_vals,
        method='gaver_stehfest',
        n_terms=n,
        error_estimate=0.0,  # Hard to estimate for this method
        diagnostics={
            'weights': weights_np,
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
    sigma: float | None = None
) -> ChebyshevNILTResult:
    """
    Adaptive Chebyshev-based NILT with automatic method selection.

    The number of terms doubles from 16 until two successive results agree
    to ``tol`` in the max norm relative to the result's size, or
    ``max_terms`` is reached; the returned error_estimate is that measured
    difference. Gaver-Stehfest is capped at 18 terms (float64 cancellation
    grows beyond it), so its refinement stops there instead of comparing
    two identical 18-term runs and declaring convergence.

    Args:
        F_eval: Laplace transform F(s)
        t: Time points for evaluation
        method: 'auto' (Talbot), 'weeks', 'talbot', or 'gaver_stehfest'
        tol: Target relative tolerance between successive refinements
        max_terms: Maximum number of terms
        sigma: Shift of the Talbot contour or abscissa of the Weeks line.
            Defaults to 1/t_max: for a transform whose singularities lie in
            Re s <= 0 this puts the line a distance 1/t_max to their right,
            and the factor e^{σt} it introduces is at most e over the
            requested times. A transform with singularities in Re s > 0
            needs sigma passed explicitly.

    Returns:
        ChebyshevNILTResult with the last refinement
    """
    t_arr = jnp.atleast_1d(jnp.asarray(t))
    t_max = float(jnp.max(t_arr[t_arr > 0]))

    if sigma is None:
        sigma = 1.0 / t_max

    if method == 'auto':
        method = 'talbot'
    if method not in ('weeks', 'talbot', 'gaver_stehfest'):
        raise ValueError(f"Unknown method: {method}")

    def run(n: int) -> ChebyshevNILTResult:
        if method == 'weeks':
            return weeks_method(F_eval, n, t_arr, sigma, b=1.0 / t_max)
        if method == 'talbot':
            return talbot_method(F_eval, t_arr, n, sigma)
        return gaver_stehfest_method(F_eval, t_arr, min(n, 18))

    n_terms = 16
    result = run(n_terms)
    while 2 * n_terms <= max_terms:
        n_terms *= 2
        refined = run(n_terms)
        diff = float(jnp.max(jnp.abs(refined.f - result.f)))
        rel_diff = diff / (float(jnp.max(jnp.abs(refined.f))) + 1e-14)
        result = refined._replace(error_estimate=rel_diff)
        if rel_diff < tol or (method == 'gaver_stehfest' and n_terms >= 18):
            break

    return result


# =============================================================================
# Comparison with FFT-NILT
# =============================================================================

def compare_chebyshev_vs_fft(
    F_eval: Callable,
    f_exact: Callable | None = None,
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
