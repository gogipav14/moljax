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

from moljax.laplace.nilt_fft import nilt_fft_batch
from moljax.laplace.spectral_bounds import SpectralBounds
from moljax.laplace.tuning import TunedNILTParams, tune_nilt_params


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

    Returns:
        TunedNILTParams optimized for the FFT operator
    """
    # Get exact bounds
    fft_bounds = exact_spectral_bounds_from_fft(eigenvalues, operator_name)
    spectral_bounds = fft_bounds_to_spectral_bounds(fft_bounds)

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
    Solve u_t = L*u + f(x), u(0) = u0, by inverting the Laplace transform
    of every Fourier mode numerically.

    In Fourier space the PDE decouples into scalar transforms
    U_k(s) = (u0_k + f_k/s) / (s - λ_k), one per wavenumber, and all of them
    are inverted in one nilt_fft_batch call. The closed form
    u_k(t) = e^{λ_k t} u0_k + t φ₁(λ_k t) f_k is returned alongside as
    ``u_analytical`` so the two can be compared; it is not what ``u_final``
    reports.

    Two details make the inversion accurate to the tuner's design tolerance
    instead of first order in dt:

    - The t = 0 jump is removed analytically. The uniform-grid inversion
      samples the periodic extension at the jump between u_k(0+) and
      u_k(2T-), where the Fourier partial sums converge to the midpoint, and
      the ringing this leaves is multiplied by e^{a t} (about 100 at the
      tuned shift). Inverting G_k(s) = U_k(s) - u0_k/(s + c) with
      c = 1/t_end, whose inverse vanishes at t = 0, and adding u0_k e^{-c t}
      back afterwards, removes the jump.
    - Complex modes are inverted as two real transforms. For real u,
      u_{-k}(t) = conj(u_k(t)), so P_k = (U_k + U_{-k})/2 and
      Q_k = (U_k - U_{-k})/(2i) are the transforms of the real functions
      Re u_k(t) and Im u_k(t), and the real-valued NILT (which enforces
      Hermitian symmetry of the sampled spectrum) applies to each. Inverting
      Re U_k(s) alone is wrong by O(1).

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
        source: Optional constant source term f(x)
        nilt_params: Pre-tuned NILT parameters (auto-tuned if None). The
            Bromwich shift must exceed max Re λ, and must be positive when a
            source is given (f_k/s adds a pole at the origin).
        return_full_history: If True, also return u on every NILT grid time
        dtype: Output data type

    Returns:
        Dict with:
            - u_final: NILT solution at t_final
            - t_final: The NILT grid time nearest t_end (the tuned grid,
              2T = 4 t_end = N dt, contains t_end exactly)
            - u_analytical: Closed form e^{λ t} u0 + t φ₁(λ t) f at t_final
            - nilt_dc: NILT value of the k = 0 mode at t_final
            - nilt_result: The batch NILTResult (rows: Re u_k then Im u_k,
              k = 0..n//2, before the jump term is added back)
            - params: NILT parameters used
            - t_history, u_history: if return_full_history, the NILT grid
              and the solution on it, shape (N, n)
    """
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
    source_hat = jnp.fft.fft(source) if source is not None else None

    if nilt_params is None:
        nilt_params = tune_nilt_for_fft_operator(eigenvalues, t_end, dtype=dtype)

    a = nilt_params.a
    re_max = float(jnp.max(jnp.real(eigenvalues)))
    if a <= re_max:
        raise ValueError(
            f"Bromwich shift a={a:.3e} must exceed the spectral abscissa "
            f"max Re(lambda)={re_max:.3e}; the contour has to pass to the right "
            f"of every pole of U_k(s)."
        )
    if source_hat is not None and a <= 0.0:
        raise ValueError(
            "a constant source adds the pole f_k/s at the origin; the Bromwich "
            f"shift must be positive, got a={a:.3e}."
        )

    c = 1.0 / t_end
    n_half = n_modes // 2 + 1
    k_pos = jnp.arange(n_half)
    k_neg = (-k_pos) % n_modes

    def transfer_pairs(s: jnp.ndarray) -> jnp.ndarray:
        """P_k rows then Q_k rows of G_k(s) = U_k(s) - u0_k/(s + c), shape (2 n_half, len(s))."""
        s = s[None, :]
        numerator = u0_hat[:, None]
        if source_hat is not None:
            numerator = numerator + source_hat[:, None] / s
        G = numerator / (s - eigenvalues[:, None]) - u0_hat[:, None] / (s + c)
        P = 0.5 * (G[k_pos] + G[k_neg])
        Q = -0.5j * (G[k_pos] - G[k_neg])
        return jnp.concatenate([P, Q], axis=0)

    batch = nilt_fft_batch(
        transfer_pairs,
        dt=nilt_params.dt,
        N=nilt_params.N,
        a=a,
        n_batch=2 * n_half,
        dtype=dtype,
    )

    # Reassemble u_k(t) = Re + i Im, adding back the jump term u0_k e^{-c t}.
    decay = jnp.exp(-c * batch.t)
    u_hat_pos = batch.f[:n_half] + 1j * batch.f[n_half:] + u0_hat[k_pos][:, None] * decay[None, :]

    t_idx = int(jnp.argmin(jnp.abs(batch.t - t_end)))
    t_final = float(batch.t[t_idx])
    # irfft rebuilds u_{-k} = conj(u_k) and returns the real field.
    u_final = jnp.fft.irfft(u_hat_pos[:, t_idx], n=n_modes).astype(dtype)

    def closed_form(t: float) -> jnp.ndarray:
        """u(t) = ifft(e^{λt} u0_hat + t φ₁(λt) f_hat), φ₁(z) = (e^z - 1)/z."""
        z = eigenvalues * t
        exp_z = jnp.exp(z)
        u_hat_t = exp_z * u0_hat
        if source_hat is not None:
            small = jnp.abs(z) < 1e-10
            phi1 = jnp.where(
                small,
                1.0 + z / 2 + z**2 / 6,
                (exp_z - 1.0) / jnp.where(small, 1.0, z),
            )
            u_hat_t = u_hat_t + t * phi1 * source_hat
        return jnp.real(jnp.fft.ifft(u_hat_t))

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

    Returns:
        NILTvsTSSComparison with accuracy and timing results
    """
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
    # are immediately discarded.
    #
    # etd_integrate floors (t_end - t_start)/dt internally. save_every must
    # match the number of steps actually taken, or no sample is ever retained
    # and the "final" state is still u0; the same count is what the
    # comparison reports.
    n_steps_taken = int(t_end / tss_dt)
    save_every = max(n_steps_taken, 1)

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
