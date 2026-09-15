"""
FFT-NILT Bridge: Connecting FFT-diagonalizable MOL operators to NILT.

This module bridges the FFT operator infrastructure (Milestone 1-2) with
the NILT (Numerical Inverse Laplace Transform) solver for linear PDEs.

Key features:
1. Exact spectral bounds from FFT eigenvalues (no power iteration needed)
2. NILT parameter tuning optimized for FFT-diagonalizable operators
3. Transfer function construction from FFT operators
4. Performance comparison: NILT vs time-stepping (ETD/IMEX)

For linear PDEs of the form:
    u_t = L*u + f(x)  where L is FFT-diagonalizable

The Laplace-domain solution is:
    U(s) = (sI - L)^{-1} * (u0 + F(s))

where in Fourier space this becomes:
    U_hat(k, s) = (s - λ(k))^{-1} * (u0_hat(k) + F_hat(k, s))
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from moljax._precision import require_x64
from moljax.core.jit_kernels import phi1
from moljax.laplace.nilt_fft import nilt_fft_batch
from moljax.laplace.spectral_bounds import SpectralBounds
from moljax.laplace.tuning import TunedNILTParams, tune_nilt_params

# tau: nilt_solve_linear_pde hands a mode to the NILT only when
# |lambda_k| t_end exceeds this; below it the mode is evaluated entirely in
# closed form. See nilt_solve_linear_pde's docstring for the measurement
# that sets the value.
TRANSIENT_TAU = 1e-2


class FFTSpectralBounds(NamedTuple):
    """Exact spectral bounds from FFT eigenvalues."""
    rho: float  # Spectral radius = max|λ(k)|
    re_max: float  # Spectral abscissa = max(Re(λ(k)))
    im_max: float  # Max imaginary magnitude = max|Im(λ(k))|
    eigenvalues: jnp.ndarray  # Full eigenvalue array λ(k)
    methods_used: dict[str, str]  # Method description
    warnings: list[str]  # Any warnings


def exact_spectral_bounds_from_fft(
    eigenvalues: jnp.ndarray,
    operator_name: str = "FFT operator"
) -> FFTSpectralBounds:
    """
    Compute exact spectral bounds from FFT eigenvalues.

    This is the key advantage of FFT-diagonalizable operators: we know
    the exact eigenvalues λ(k), so spectral bounds are computed exactly
    in O(N) time without power iteration.

    Args:
        eigenvalues: Complex eigenvalue array λ(k) from FFT operator
        operator_name: Name for diagnostics

    Returns:
        FFTSpectralBounds with exact rho, re_max, im_max
    """
    # Exact bounds from eigenvalues
    rho = float(jnp.max(jnp.abs(eigenvalues)))
    re_max = float(jnp.max(jnp.real(eigenvalues)))
    im_max = float(jnp.max(jnp.abs(jnp.imag(eigenvalues))))

    return FFTSpectralBounds(
        rho=rho,
        re_max=re_max,
        im_max=im_max,
        eigenvalues=eigenvalues,
        methods_used={'exact_fft': f'{operator_name}, N={len(eigenvalues)}'},
        warnings=[]
    )


def fft_bounds_to_spectral_bounds(fft_bounds: FFTSpectralBounds) -> SpectralBounds:
    """Convert FFTSpectralBounds to standard SpectralBounds for NILT tuning."""
    return SpectralBounds(
        rho=fft_bounds.rho,
        re_max=fft_bounds.re_max,
        im_max=fft_bounds.im_max,
        methods_used=fft_bounds.methods_used,
        warnings=fft_bounds.warnings
    )


def tune_nilt_for_fft_operator(
    eigenvalues: jnp.ndarray,
    t_end: float,
    *,
    operator_name: str = "FFT operator",
    dtype=jnp.float64,
    delta_min: float = 1e-3,
    eps_tail: float = 1e-8,
    period_factor: float = 4.0,
    omega_factor: float = 1.5,
    N_min: int = 256,
    N_max: int = 8192,
    re_max_override: float | None = None,
) -> TunedNILTParams:
    """
    Tune NILT parameters using exact FFT eigenvalues.

    This is the optimized path for FFT-diagonalizable operators:
    - Uses exact spectral bounds (no estimation needed)
    - Guarantees coverage of all eigenvalue frequencies
    - Optimal Bromwich shift placement

    Args:
        eigenvalues: Complex eigenvalue array λ(k) from FFT operator
        t_end: End time for NILT inversion
        operator_name: Name for diagnostics
        dtype: Output data type
        delta_min: Minimum shift above spectral abscissa
        eps_tail: Tail truncation tolerance
        period_factor: Multiplier for period T relative to t_end
        omega_factor: Multiplier for frequency coverage
        N_min: Minimum FFT size
        N_max: Maximum FFT size
        re_max_override: If given, replaces the spectral abscissa (re_max)
            computed from eigenvalues before tuning. nilt_solve_linear_pde
            uses this to tune against the abscissa of the transform it
            actually inverts (sigma_H, the transient's abscissa) rather
            than the abscissa of the full eigenvalue spectrum, which can
            include modes handled entirely in closed form.

    Returns:
        TunedNILTParams optimized for the FFT operator
    """
    # Get exact bounds
    fft_bounds = exact_spectral_bounds_from_fft(eigenvalues, operator_name)
    spectral_bounds = fft_bounds_to_spectral_bounds(fft_bounds)
    if re_max_override is not None:
        spectral_bounds = spectral_bounds._replace(re_max=re_max_override)

    # Use standard tuner with exact bounds
    params = tune_nilt_params(
        t_end=t_end,
        bounds=spectral_bounds,
        dtype=dtype,
        delta_min=delta_min,
        eps_tail=eps_tail,
        period_factor=period_factor,
        omega_factor=omega_factor,
        N_min=N_min,
        N_max=N_max,
    )

    return params


def create_transfer_function_from_fft_operator(
    eigenvalues: jnp.ndarray,
    u0_hat: jnp.ndarray,
    source_hat: jnp.ndarray | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Create Laplace-domain transfer function from FFT operator.

    For the linear PDE:  u_t = L*u + f(x)  with  u(0) = u0

    The Laplace transform gives:
        s*U(s) - u0 = L*U(s) + F(s)
        U(s) = (sI - L)^{-1} * (u0 + F(s)/s)

    In Fourier space, this becomes diagonal:
        U_hat(k, s) = (s - λ(k))^{-1} * (u0_hat(k) + source_hat(k)/s)

    For constant source f(x), F(s) = f/s, so:
        U_hat(k, s) = (s - λ(k))^{-1} * (u0_hat(k) + source_hat(k)/s)

    Args:
        eigenvalues: FFT eigenvalues λ(k)
        u0_hat: FFT of initial condition
        source_hat: FFT of source term (None for zero source)

    Returns:
        Transfer function F(s) that maps s (complex array) to U_hat(s)
    """
    def transfer_function(s: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluate U_hat(s) for given s values.

        Args:
            s: Complex s values (can be array)

        Returns:
            U_hat values (averaged over wavenumbers for scalar output)
        """
        # For each s, compute (s - λ(k))^{-1} * (u0_hat + source_hat/s)
        # We return the DC component (k=0) or spatial average

        # s is array of complex values, eigenvalues is array of size N
        # We need to compute for each s the full spectrum then average

        def eval_single_s(s_val):
            denom = s_val - eigenvalues
            # Regularize to avoid division by zero
            denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)

            if source_hat is not None:
                # Include source term: (u0_hat + source_hat/s) / (s - λ)
                numerator = u0_hat + source_hat / s_val
            else:
                numerator = u0_hat

            U_hat_k = numerator / denom

            # Return DC component (spatial mean in physical space)
            return U_hat_k[0]

        # Vectorize over s array
        return jax.vmap(eval_single_s)(s)

    return transfer_function


def nilt_solve_linear_pde(
    eigenvalues: jnp.ndarray,
    u0: jnp.ndarray,
    t_end: float,
    *,
    source: jnp.ndarray | None = None,
    nilt_params: TunedNILTParams | None = None,
    return_full_history: bool = False,
    dtype=jnp.float64,
) -> dict:
    """
    Solve u_t = L*u + f(x), u(0) = u0, by evaluating everything the closed
    form already knows and inverting only the residual-driven transient.

    In Fourier space the PDE decouples into scalar ODEs
    u_k' = λ_k u_k + f_k, one per wavenumber, whose exact solution is

        u_k(t) = e^{λ_k t} u0_k + t φ₁(λ_k t) f_k,  φ₁(z) = (e^z - 1)/z

    for every λ_k, λ_k = 0 included: φ₁(0) = 1, so the one formula
    degenerates to u0_k + f_k t on its own and needs no separate branch. It
    contains no 1/λ_k. The algebraically equivalent split
    u_k(t) = w_k e^{λ_k t} - f_k/λ_k, w_k = u0_k + f_k/λ_k, instead
    evaluates an O(1) answer as the difference of two terms of size
    |f_k/λ_k|, and for a λ_k small but not exactly zero that difference is
    all roundoff: eight eigenvalues -1e-16 with u0 = 0, f = 1, t_end = 1
    returned 2 instead of 1, and the same case at -1e-8 in float32 returned 0.

    Collecting the per-mode residual

        r_k = λ_k u0_k + f_k,   u_k(t) = u0_k + r_k t φ₁(λ_k t)

    (substitute f_k = r_k - λ_k u0_k above) instead groups the solution
    around the quantity that drives the mode away from its initial value: a
    stationary mode has r_k = 0 and stands still. That form is what the
    inverted weight is built from, and only that; every value this function
    evaluates is evaluated in the e^{λ_k t} u0_k + t φ₁(λ_k t) f_k form,
    where an f_k far smaller than λ_k u0_k survives instead of being
    rounded away inside r_k.

    Everything above is evaluated in the time domain (via
    moljax.core.jit_kernels.phi1, whose Taylor branch below |z| = 0.5 in
    double precision carries the small-|λ_k t| limit). What the NILT is
    given is the transient

        H_k(s) = w_k * [1/(s - λ_k) - 1/(s + c_k)],   c_k = -Re(λ_k)

    whose inverse is w_k (e^{λ_k t} - e^{-c_k t}), all modes in one
    nilt_fft_batch call, with the residual-weighted

        w_k = r_k/λ_k  if |λ_k| t_end > τ,   w_k = 0  otherwise

    so a mode with no residual is not inverted at all and the inversion's
    own error cannot reach it. Only the weight is written in the residual.
    The rest is added back in closed form, in u0_k and f_k: substituting
    w_k = u0_k + f_k/λ_k (the same number as r_k/λ_k) into
    u_k(t) = e^{λ_k t} u0_k + t φ₁(λ_k t) f_k,

        u_k(t) - w_k (e^{λ_k t} - e^{-c_k t})
            = u0_k e^{-c_k t}
              + f_k [t φ₁(λ_k t) - (e^{λ_k t} - e^{-c_k t})/λ_k]
            = u0_k e^{-c_k t} + f_k (e^{-c_k t} - 1)/λ_k
            = u0_k e^{-c_k t} + f_k t φ₁(-c_k t) Re(λ_k)/λ_k

    (the middle step is t φ₁(λ_k t) = (e^{λ_k t} - 1)/λ_k, which cancels
    the e^{λ_k t} the transient took away) on the inverted modes, and the
    whole exact solution e^{λ_k t} u0_k + t φ₁(λ_k t) f_k on the modes with
    w_k = 0. Neither branch cancels: the u0_k term is a plain decay,
    Re(λ_k)/λ_k is bounded by 1 in modulus, φ₁ carries its own
    small-argument branch, and the division by λ_k happens only on modes
    whose |λ_k| t_end clears τ.

    Writing that remainder in the residual instead, as
    u0_k + r_k t φ₁(-c_k t) Re(λ_k)/λ_k, is algebraically the same and
    loses f_k whenever |λ_k u0_k| ≫ |f_k|: r_k = λ_k u0_k + f_k rounds to
    λ_k u0_k, and what is left is u0_k against a second term of size
    |r_k/λ_k| = |u0_k|, an O(1) answer read off the difference of two
    O(u0_k) terms. Four eigenvalues -1, u0 = 1e16, f = 1, t_end = 50
    returned 0 everywhere, u_analytical included, against the exact
    1.000001928749848 = 1e16 e^{-50} + (1 - e^{-50}); at λ = -1 + 5j with
    u0_k = 1e12, f_k = 1 and t_end = 30 the closed form was off by 1.3e-4
    relative. In the form above both are exact to rounding.

    The closed form (the full solution, not just the part evaluated here)
    is returned alongside as ``u_analytical`` so the two can be compared;
    it is not what ``u_final`` reports.

    The weight has to be r_k-proportional rather than u0_k-proportional,
    which is what it was between the removal of the 1/λ_k cancellation and
    this fix. With w_k = u0_k a stationary mode still had its homogeneous
    part e^{λ_k t} u0_k inverted numerically, and nothing cancelled the
    inversion's error any more, because the forcing that used to cancel it
    is now evaluated in closed form: λ = [0, -1+100j, 0, -1-100j],
    u0 = [1, 0, -1, 0], f = [1, 100, -1, -100], t_end = 1 is stationary
    (r_k = 0 for every k) and returned [0.98499821, 0.00121119, -0.98499821,
    -0.00121119] instead of u0, a max error of 1.5e-2, the raw NILT error
    on the lightly damped 100j pair.

    τ = TRANSIENT_TAU = 1e-2, a module constant. Both branches are exact
    in exact arithmetic, so τ is not a choice between two approximations:
    it decides which modes are handed to the NILT at all, and every mode
    handed over is charged the inversion's own error, which the closed form
    below τ does not pay.

    That error does not shrink as λ_k → 0. It is tempting to read the
    weight r_k/λ_k as amplifying the inversion's truncation error by
    1/(|λ_k| t_end), but the two poles of H_k coalesce in the same limit:

        H_k(s) = w_k (λ_k + c_k)/((s - λ_k)(s + c_k))
               = r_k [i Im(λ_k)/λ_k] / ((s - λ_k)(s + c_k))

    and |i Im(λ_k)/λ_k| ≤ 1, so what the NILT is handed is O(|r_k|), not
    O(|w_k|), and its truncation error is flat in |λ_k| t_end. Measured
    with λ = ±ib, u0 = 0, f_hat = 2, t_end = 1, the inverted branch is
    1.861e-5 away from the exact answer at every b from 1e-7 to 3, while
    the closed form is exact. What the weight does amplify is the
    floating-point cancellation in forming 1/(s - λ_k) - 1/(s + c_k), of
    relative size eps |s|/|Im(λ_k)| at the contour's top frequency
    π/dt ≈ 200 in that case: 3e-6 at sqrt(eps), 5e-12 at τ = 1e-2. The old
    τ = sqrt(eps) bounded that roundoff correctly; it was simply not the
    binding term.

    A flat error cannot be made small by moving τ, so τ goes where the
    jump it creates is no larger than the answer's own sensitivity to λ_k
    there. In the real field a conjugate pair's first-order term i Im(λ_k)t
    cancels against its partner, so below τ the mode is the straight line
    the closed form would draw to within (|λ_k| t_end)^2/6 of itself: at
    τ = 1e-2 that is 1.667e-5, against the 1.861e-5 the inversion costs.
    The two sides of τ therefore differ by about as much as the answer
    itself differs across the band τ separates, which is the most that can
    be asked of a threshold with an exact branch on one side of it.
    sqrt(eps) instead put that same 1.9e-5 jump where the exact answer
    changes by 4e-17: b = 1.49e-8 returned 1.0 and b = 1.491e-8 returned
    0.9999813. τ is not tied to eps_tail = 1e-8, the tuner's wraparound
    tolerance, which the inversion of a transient this slowly decaying does
    not reach in the first place.

    The 1/(s + c_k) term is the t = 0 jump correction, placed to cancel
    exactly at the abscissa of the pole it is removing instead of at an
    unrelated c = 1/t_end. For a real λ_k, c_k = -λ_k, so both poles of H_k
    sit at the same point s = λ_k and H_k is identically zero; in the
    remainder Re(λ_k)/λ_k is then exactly 1 and φ₁(-c_k t) is φ₁(λ_k t), so
    the remainder is the whole exact solution, and the two branches of the
    remainder agree bit for bit, so a real spectrum has no discontinuity at
    τ at all. On a real spectrum the NILT inverts nothing and the bridge
    reproduces the closed form to rounding.

    The NILT budget is spent entirely on modes with a nonzero imaginary
    part (or, more precisely, on any mode whose weight w_k is nonzero and
    whose two poles therefore do not coincide); a purely real, decaying
    spectrum needs no numerical inversion at all. This also removes the
    pole G_k(s) = U_k(s) - u0_k/(s + c) used to carry, with c = 1/t_end: it
    added a pole at -1/t_end that the tuner never accounted for, and left
    the source pole at s = 0 in place, which together could throw the
    Bromwich contour off by orders of magnitude (see the module's git log
    for the a = full(8, -10), u0 = ones(8), t_end = 1 regression this fixed).

    Two details make the inversion accurate to the grid's own truncation
    error instead of first order in dt:

    - The t = 0 jump is removed analytically, as above: the uniform-grid
      inversion samples the periodic extension at the jump between h_k(0+)
      and h_k(2T-), where the Fourier partial sums converge to the
      midpoint, and the ringing this leaves is multiplied by e^{a t} (about
      100 at the tuned shift). The inverse of H_k(s) vanishes at t = 0 by
      construction, so there is no jump to ring; the -e^{-c_k t} it
      subtracts is restored by the closed-form remainder above, not by a
      separate add-back.
    - Complex modes are inverted as two real transforms. For real u,
      u_{-k}(t) = conj(u_k(t)), so P_k = (H_k + H_{-k})/2 and
      Q_k = (H_k - H_{-k})/(2i) are the transforms of the real functions
      Re h_k(t) and Im h_k(t), and the real-valued NILT (which enforces
      Hermitian symmetry of the sampled spectrum) applies to each. Inverting
      Re H_k(s) alone is wrong by O(1).

    There is no spectral-zero branch in the reconstruction: the same
    continuous formula is evaluated on both sides of τ, and τ decides only
    which part of it the NILT is asked to supply. When no |λ_k| t_end
    clears τ, every w_k is zero, no grid is built at all, and the closed
    form is returned directly; that subsumes the old "the spectrum is
    numerically the origin" early return, which tested max|λ_k| t_end
    against eps.

    64-bit precision is required, as everywhere else in the NILT stack: the
    Bromwich contour's e^{a t} factor (about 100 at the tuned shift)
    multiplies the inversion's roundoff, and φ₁'s direct branch already
    costs a decade of relative accuracy per decade below |z| = 1.

    A self-paired mode (index 0, or N//2 on an even grid) has no distinct
    conjugate partner; a real field cannot have a nonzero imaginary part
    there; eigenvalues with one raise a ValueError before any of the above
    is attempted.

    Restricted to 1D spectra: eigenvalues and u0 must each be a 1D array of
    length n_modes (FFT ordering). n_modes is read from eigenvalues.shape[0]
    and a 1D fft/irfft is applied throughout, so a multi-dimensional
    spectrum (e.g. from a 2D DiffusionOperator) either fails to broadcast
    against the 1D frequency grid or silently reconstructs a field of the
    wrong shape; both are rejected up front with a ValueError rather than
    attempted. eigenvalues must be nonempty, u0.shape must equal
    eigenvalues.shape, and source (when given) must too: every mode of u0
    and source is combined with the eigenvalue at the same index, so a
    mismatched length would otherwise fabricate modes (broadcasting a
    shorter array against a longer eigenvalues) or silently drop them
    (returning a field the length of the shorter array) instead of raising.

    Args:
        eigenvalues: FFT eigenvalues λ(k), FFT ordering, 1D only
        u0: Initial condition (real space, interior only), 1D only
        t_end: End time
        source: Optional source term f(x), constant in time. A
            time-dependent or polynomial-in-t source is not accepted by
            this signature, so only φ₁ is needed here; the higher φ₂, φ₃
            of moljax.core.jit_kernels would be the ones to reach for if
            it ever were.
        nilt_params: Pre-tuned NILT parameters (auto-tuned if None). The
            Bromwich shift must exceed sigma_H = max Re(λ_k) over modes
            with a nonzero weight w_k = r_k/λ_k, the abscissa of the
            transform H_k actually being inverted; stationary modes
            (r_k = 0) and modes below τ carry no transient and do not
            constrain it, and there is no source-pole positivity
            requirement (H_k has no pole at the origin regardless of
            source).
        return_full_history: If True, also return u on every NILT grid
            time. Meaningless (and not populated) when there is no
            transient to invert, since no NILT grid is built in that case.
        dtype: Output data type

    Raises:
        RuntimeError: If JAX is not running with 64-bit precision.

    Returns:
        Dict with:
            - u_final: solution at t_final (the NILT-inverted transient
              plus the closed-form remainder)
            - t_final: The NILT grid time nearest t_end (the tuned grid,
              2T = 4 t_end = N dt, contains t_end exactly), or exactly
              t_end when there is no transient to invert
            - u_analytical: Closed form e^{λ t} u0 + t φ₁(λ t) f at
              t_final
            - nilt_dc: value of the k = 0 mode at t_final
            - nilt_result: The batch NILTResult (rows: Re h_k then Im h_k,
              k = 0..n//2, before the closed-form remainder is added
              back), or None when there is no transient to invert
            - params: NILT parameters used, or None when there is no
              transient to invert
            - note: only present when there is nothing to invert (every
              residual r_k is zero, or every |λ_k| t_end is at or below τ);
              explains that the closed form was returned directly
            - t_history, u_history: if return_full_history, the NILT grid
              and the solution on it, shape (N, n)
    """
    require_x64("nilt_solve_linear_pde")

    eigenvalues = jnp.asarray(eigenvalues)
    u0 = jnp.asarray(u0)
    if eigenvalues.ndim != 1:
        raise ValueError(
            f"nilt_solve_linear_pde only supports a 1D spectrum; got "
            f"eigenvalues.shape={eigenvalues.shape}. n_modes is read from "
            f"eigenvalues.shape[0] and a 1D fft/irfft is applied throughout, "
            f"so a multi-dimensional spectrum (e.g. from a 2D "
            f"DiffusionOperator) is not supported here."
        )
    if u0.ndim != 1:
        raise ValueError(
            f"nilt_solve_linear_pde only supports a 1D initial condition; "
            f"got u0.shape={u0.shape}. Flatten a multi-dimensional field "
            f"before calling this function; it is not supported here."
        )
    if eigenvalues.shape[0] == 0:
        raise ValueError("nilt_solve_linear_pde requires a nonempty eigenvalues array.")
    if u0.shape != eigenvalues.shape:
        raise ValueError(
            f"u0.shape={u0.shape} must match eigenvalues.shape={eigenvalues.shape}: "
            f"n_modes is read from eigenvalues.shape[0] and every mode of u0 is "
            f"transformed against the eigenvalue at the same index, so a mismatched "
            f"length would fabricate or drop modes rather than raise."
        )
    if source is not None:
        source = jnp.asarray(source)
        if source.shape != eigenvalues.shape:
            raise ValueError(
                f"source.shape={source.shape} must match eigenvalues.shape={eigenvalues.shape} "
                f"for the same reason as u0: source is transformed mode-by-mode against "
                f"eigenvalues."
            )
    n_modes = eigenvalues.shape[0]
    u0_hat = jnp.fft.fft(u0)
    # Zeros rather than None when there is no source: every formula below
    # carries an f_k term, and a zero one costs one multiply.
    source_hat = (
        jnp.fft.fft(source) if source is not None
        else jnp.zeros_like(u0_hat)
    )

    # Only the self-paired Hermitian check below uses this: an imaginary
    # part this far under the spectrum's own scale is roundoff in the symbol
    # rather than a complex eigenvalue. The reconstruction itself has no
    # spectral-zero threshold to set.
    max_abs_lambda = float(jnp.max(jnp.abs(eigenvalues)))
    im_tol = max(1e-12 * max_abs_lambda, 1e-300)

    # A self-paired mode (its own conjugate partner) cannot carry a nonzero
    # imaginary eigenvalue for a real field: index 0 always, and N//2 on an
    # even grid (the Nyquist mode of an odd first-derivative symbol such as
    # AdvectionDiffusionOperator's -i*v*k).
    self_paired = [0] if n_modes % 2 else [0, n_modes // 2]
    self_paired_idx = jnp.array(self_paired)
    im_self_paired = jnp.imag(eigenvalues)[self_paired_idx]
    if bool(jnp.any(jnp.abs(im_self_paired) > im_tol)):
        bad = int(self_paired_idx[int(jnp.argmax(jnp.abs(im_self_paired)))])
        raise ValueError(
            f"eigenvalues[{bad}] = {complex(eigenvalues[bad])!r} has a nonzero "
            f"imaginary part at a self-paired mode (index 0, or N//2 on an "
            f"even grid). A self-paired mode is its own conjugate partner, so "
            f"a real field cannot have a complex eigenvalue there."
        )

    real_dtype = u0_hat.real.dtype

    # r_k = lambda_k u0_k + f_k, the residual that drives the mode off its
    # initial value: u_k(t) = u0_k + r_k t phi1(lambda_k t). A stationary
    # mode has r_k = 0 to the last bit and must invert nothing at all,
    # which is why the weight below is proportional to r_k and not to u0_k.
    # Only the weight is: what is added back in closed form is written in
    # u0_k and f_k, where it has no cancellation.
    residual_hat = eigenvalues * u0_hat + source_hat

    c = -jnp.real(eigenvalues)  # per-mode jump-cancellation rate

    # tau = TRANSIENT_TAU = 1e-2 on |lambda_k| t_end. Both branches are
    # exact in exact arithmetic, so tau trades the inversion's own error
    # against the answer's sensitivity to lambda_k: a mode below tau is a
    # straight line to within (|lambda_k| t_end)^2/6 of itself, which at
    # tau is 1.7e-5, the same size as the 1.9e-5 the NILT costs on such a
    # mode. Below tau the closed form is therefore not merely cheaper but
    # indistinguishable, and the division by lambda_k never happens there.
    inverted = jnp.abs(eigenvalues) * abs(float(t_end)) > TRANSIENT_TAU
    lambda_safe = jnp.where(inverted, eigenvalues, 1.0)  # placeholder; w is 0 there
    # w_k = r_k/lambda_k, with r_k = lambda_k u0_k + f_k formed first and
    # not as the algebraically equal u0_k + f_k/lambda_k: a stationary mode
    # has r_k = 0 to the last bit, so its weight is exactly zero and it is
    # not inverted at all.
    w = jnp.where(inverted, residual_hat / lambda_safe, 0.0)

    def closed_form_hat(t: jnp.ndarray) -> jnp.ndarray:
        """e^{lambda_k t} u0_k + t phi1(lambda_k t) f_k, shape (n_modes, len(t))."""
        t = jnp.atleast_1d(jnp.asarray(t)).astype(real_dtype)
        z = eigenvalues[:, None] * t[None, :]
        return jnp.exp(z) * u0_hat[:, None] + (
            t[None, :] * phi1(z) * source_hat[:, None]
        )

    def remainder_hat(t: jnp.ndarray) -> jnp.ndarray:
        """u_k(t) minus the inverted transient, per mode, shape (n_modes, len(t)).

        e^{-c_k t} u0_k + t phi1(-c_k t) Re(lambda_k)/lambda_k f_k on the
        inverted modes (the exact solution less
        w_k (e^{lambda_k t} - e^{-c_k t})), and the whole exact solution
        e^{lambda_k t} u0_k + t phi1(lambda_k t) f_k on the rest. Both are
        written in u0_k and f_k rather than in the residual r_k: the
        residual is what the weight must be built from, but a remainder
        built from it reads r_k/lambda_k against u0_k, two terms of the
        same size whose difference is the answer, and that difference is
        where an f_k much smaller than lambda_k u0_k is lost. Here the u0_k
        term is a plain decay, Re(lambda_k)/lambda_k is bounded by 1 in
        modulus, and phi1 carries its own small-argument branch, so neither
        term is a cancellation. For a real lambda_k the two branches are
        the same expression (c_k = -lambda_k, the ratio is 1).
        """
        t = jnp.atleast_1d(jnp.asarray(t)).astype(real_dtype)
        decay = -c[:, None] * t[None, :]  # Re(lambda_k) t
        damped = jnp.exp(decay) * u0_hat[:, None] + (
            t[None, :]
            * phi1(decay)
            * (jnp.real(eigenvalues) / lambda_safe)[:, None]
            * source_hat[:, None]
        )
        return jnp.where(inverted[:, None], damped, closed_form_hat(t))

    def closed_form(t: float) -> jnp.ndarray:
        """The exact field at a single time t."""
        return jnp.real(jnp.fft.ifft(closed_form_hat(jnp.asarray(t))[:, 0]))

    # Nothing is inverted for a mode with no residual, and nothing at all is
    # inverted when no |lambda_k| t_end clears tau: that subsumes the old
    # "the spectrum is numerically the origin" early return, which compared
    # max|lambda_k| t_end against eps.
    transient_mask = w != 0
    has_transient = bool(jnp.any(transient_mask))

    if not has_transient:
        # Nothing to invert: either every residual r_k is zero (the field is
        # stationary), or no |lambda_k| t_end clears tau. The closed form is
        # the exact solution in both cases.
        u_final = closed_form(t_end)
        result = {
            'u_final': u_final,
            't_final': float(t_end),
            'u_analytical': u_final,
            'nilt_dc': float(jnp.real(closed_form_hat(jnp.asarray(t_end))[0, 0])),
            'nilt_result': None,
            'params': None,
            'note': (
                "empty transient: every residual lambda_k u0_k + f_k is zero, or "
                f"no |lambda_k| t_end clears tau = {TRANSIENT_TAU:g}, so "
                "nilt_solve_linear_pde returned the closed form "
                "e^{lambda t} u0 + t phi1(lambda t) f directly with no NILT "
                "inversion."
            ),
        }
        return result

    sigma_H = float(jnp.max(jnp.real(eigenvalues[transient_mask])))

    if nilt_params is None:
        nilt_params = tune_nilt_for_fft_operator(
            eigenvalues, t_end, dtype=dtype, re_max_override=sigma_H
        )

    a = nilt_params.a
    if a <= sigma_H:
        raise ValueError(
            f"Bromwich shift a={a:.3e} must exceed sigma_H={sigma_H:.3e}, the "
            f"abscissa of H_k(s) (max Re(lambda_k) over modes with a nonzero "
            f"transient); the contour has to pass to the right of every pole "
            f"of H_k(s)."
        )

    n_half = n_modes // 2 + 1
    k_pos = jnp.arange(n_half)
    k_neg = (-k_pos) % n_modes

    def transfer_pairs(s: jnp.ndarray) -> jnp.ndarray:
        """P_k rows then Q_k rows of H_k(s) = w_k*[1/(s-lambda_k) - 1/(s+c_k)]."""
        s = s[None, :]
        # Modes with no transient are zeroed by w_k anyway; the placeholder
        # denominators keep a 0 * inf out of that product if the contour ever
        # passes through such a mode's pole.
        denom1 = jnp.where(transient_mask[:, None], s - eigenvalues[:, None], 1.0)
        denom2 = jnp.where(transient_mask[:, None], s + c[:, None], 1.0)
        H = w[:, None] * (1.0 / denom1 - 1.0 / denom2)
        P = 0.5 * (H[k_pos] + H[k_neg])
        Q = -0.5j * (H[k_pos] - H[k_neg])
        return jnp.concatenate([P, Q], axis=0)

    batch = nilt_fft_batch(
        transfer_pairs,
        dt=nilt_params.dt,
        N=nilt_params.N,
        a=a,
        n_batch=2 * n_half,
        dtype=dtype,
    )

    # Reassemble the transient h_k(t) = Re + i Im, which is
    # w_k (e^{lambda_k t} - e^{-c_k t}) and vanishes at t = 0, then add the
    # closed-form remainder u_k(t) - h_k(t), mode by mode. The -e^{-c_k t}
    # that H_k subtracts is restored inside that remainder, not by a
    # separate add-back: doing it separately would rebuild w_k e^{lambda_k t}
    # and leave the remainder u0_k - r_k/lambda_k, an O(1/lambda_k) term
    # again.
    transient_pos = batch.f[:n_half] + 1j * batch.f[n_half:]
    u_hat_pos = transient_pos + remainder_hat(batch.t)[k_pos]

    t_idx = int(jnp.argmin(jnp.abs(batch.t - t_end)))
    t_final = float(batch.t[t_idx])
    # irfft rebuilds u_{-k} = conj(u_k) and returns the real field.
    u_final = jnp.fft.irfft(u_hat_pos[:, t_idx], n=n_modes).astype(dtype)

    result = {
        'u_final': u_final,
        't_final': t_final,
        'u_analytical': closed_form(t_final),
        'nilt_dc': float(jnp.real(u_hat_pos[0, t_idx])),
        'nilt_result': batch,
        'params': nilt_params,
    }

    if return_full_history:
        result['t_history'] = batch.t
        result['u_history'] = jnp.fft.irfft(u_hat_pos, n=n_modes, axis=0).T.astype(dtype)

    return result


@dataclass
class NILTvsTSSComparison:
    """Results from NILT vs time-stepping comparison."""
    t_end: float
    grid_size: int

    # Accuracy
    nilt_error: float  # RMS error vs analytical
    tss_error: float  # RMS error vs analytical
    analytical_norm: float  # Norm of analytical solution

    # Timing
    nilt_time_ms: float
    tss_time_ms: float
    speedup: float  # tss_time / nilt_time

    # Parameters
    nilt_params: TunedNILTParams
    tss_dt: float
    tss_steps: int  # Steps actually taken: floor(t_end / tss_dt)
    tss_method: str


def compare_nilt_vs_timestepping(
    eigenvalues: jnp.ndarray,
    u0: jnp.ndarray,
    t_end: float,
    *,
    tss_dt: float | None = None,
    tss_method: str = 'etd1',
    n_warmup: int = 3,
    n_runs: int = 10,
) -> NILTvsTSSComparison:
    """
    Compare NILT vs time-stepping for linear PDE.

    For linear PDEs (u_t = L*u), both methods should give the same
    answer to high precision. This comparison measures:
    - Accuracy vs analytical solution
    - Wall-clock time

    NILT is expected to be faster for long time horizons
    (t_end > 100 * dt_cfl) because it solves in Laplace domain
    with O(N log N) cost independent of t_end.

    Args:
        eigenvalues: FFT eigenvalues λ(k)
        u0: Initial condition
        t_end: End time
        tss_dt: Time step for time-stepping (auto if None)
        tss_method: 'etd1', 'etd2', or 'etdrk4'
        n_warmup: Warmup iterations for timing
        n_runs: Number of timing runs

    Raises:
        RuntimeError: If JAX is not running with 64-bit precision.

    Returns:
        NILTvsTSSComparison with accuracy and timing results
    """
    require_x64("compare_nilt_vs_timestepping")

    import time

    from moljax.core.fft_integrators import etd_integrate

    N = len(eigenvalues)

    # Analytical solution for linear PDE: u(t) = ifft(exp(λt) * u0_hat)
    u0_hat = jnp.fft.fft(u0)

    def analytical_solution(t: float) -> jnp.ndarray:
        exp_lam_t = jnp.exp(eigenvalues * t)
        u_hat_t = exp_lam_t * u0_hat
        return jnp.real(jnp.fft.ifft(u_hat_t))

    u_exact = analytical_solution(t_end)
    analytical_norm = float(jnp.linalg.norm(u_exact))

    # --- NILT solution ---
    nilt_params = tune_nilt_for_fft_operator(eigenvalues, t_end)

    # Warmup
    for _ in range(n_warmup):
        nilt_result = nilt_solve_linear_pde(eigenvalues, u0, t_end, nilt_params=nilt_params)

    # Timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        nilt_result = nilt_solve_linear_pde(eigenvalues, u0, t_end, nilt_params=nilt_params)
    nilt_time = (time.perf_counter() - t0) / n_runs * 1000  # ms

    u_nilt = nilt_result['u_final']
    nilt_error = float(jnp.linalg.norm(u_nilt - u_exact) / analytical_norm)

    # --- Time-stepping solution ---
    # Auto-select dt based on spectral radius
    rho = float(jnp.max(jnp.abs(eigenvalues)))
    if tss_dt is None:
        # ETD methods are unconditionally stable for linear part,
        # so we use moderate dt for accuracy
        tss_dt = min(0.1, 0.1 / (rho + 1e-10))

    # Create a simple operator wrapper for etd_integrate
    class SimpleOp:
        def __init__(self, eig):
            self.eigenvalues = eig
            self.grid = type('Grid', (), {'nx': len(eig)})()

        def exp_matvec(self, u, dt):
            u_hat = jnp.fft.fft(u)
            return jnp.real(jnp.fft.ifft(jnp.exp(dt * self.eigenvalues) * u_hat))

    op = SimpleOp(eigenvalues)

    def zero_rhs(state, t):
        return {name: jnp.zeros_like(v) for name, v in state.items()}

    # Only the final state is used below, so ask etd_integrate to retain
    # just the endpoint. Otherwise it materializes every intermediate
    # step, which for long horizons on fine grids is hundreds of MB that
    # are immediately discarded. (etd_integrate always returns the final
    # state as the last history entry regardless of save_every, so this is
    # purely a memory choice, not a correctness requirement.)
    #
    # etd_integrate now takes round(t_end / tss_dt) steps and requires that
    # to divide the interval exactly (matching stepping.py's fixed-step
    # schedule), raising rather than silently flooring and falling short of
    # t_end. The auto-selected tss_dt above has no reason to divide t_end
    # evenly, so it is snapped to the nearest step count that does; the
    # comparison reports the same, adjusted dt.
    n_steps_taken = max(round(t_end / tss_dt), 1)
    tss_dt = t_end / n_steps_taken
    save_every = n_steps_taken

    def _integrate():
        return etd_integrate(
            {'u': u0}, (0.0, t_end), tss_dt, {'u': op}, zero_rhs,
            method=tss_method, save_every=save_every,
        )

    # Warmup (also pays the one-time JIT compilation)
    for _ in range(n_warmup):
        _, hist = _integrate()
        jax.block_until_ready(hist[-1]['u'])

    # Timing
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _, hist = _integrate()
        jax.block_until_ready(hist[-1]['u'])
    tss_time = (time.perf_counter() - t0) / n_runs * 1000  # ms

    u_tss = hist[-1]['u']
    tss_error = float(jnp.linalg.norm(u_tss - u_exact) / analytical_norm)

    return NILTvsTSSComparison(
        t_end=t_end,
        grid_size=N,
        nilt_error=nilt_error,
        tss_error=tss_error,
        analytical_norm=analytical_norm,
        nilt_time_ms=nilt_time,
        tss_time_ms=tss_time,
        speedup=tss_time / nilt_time,
        nilt_params=nilt_params,
        tss_dt=tss_dt,
        tss_steps=n_steps_taken,
        tss_method=tss_method,
    )


def print_comparison_table(comparisons: list[NILTvsTSSComparison]) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 90)
    print("NILT vs Time-Stepping Comparison")
    print("=" * 90)
    print(f"{'t_end':>10} | {'N':>6} | {'NILT Err':>10} | {'TSS Err':>10} | "
          f"{'NILT ms':>8} | {'TSS ms':>8} | {'Speedup':>8}")
    print("-" * 90)

    for c in comparisons:
        print(f"{c.t_end:>10.2f} | {c.grid_size:>6} | {c.nilt_error:>10.2e} | "
              f"{c.tss_error:>10.2e} | {c.nilt_time_ms:>8.2f} | "
              f"{c.tss_time_ms:>8.2f} | {c.speedup:>8.2f}x")

    print("=" * 90)
