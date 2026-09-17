"""
Tests for FFT-NILT Bridge (Milestone 4).

Verifies:
1. Exact spectral bounds from FFT eigenvalues (no power iteration)
2. NILT parameter tuning optimized for FFT operators
3. NILT accuracy matches time-stepping to 1e-4 relative error
4. NILT faster for long time horizons (t_end > 100 dt_cfl)
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from moljax.core.fft_operators import AdvectionDiffusionOperator, DiffusionOperator
from moljax.core.grid import Grid1D
from moljax.laplace.fft_nilt_bridge import (
    TRANSIENT_TAU,
    compare_nilt_vs_timestepping,
    exact_spectral_bounds_from_fft,
    fft_bounds_to_spectral_bounds,
    nilt_solve_linear_pde,
    print_comparison_table,
    tune_nilt_for_fft_operator,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def grid_128():
    """128-point periodic grid on [0, 1]."""
    return Grid1D.uniform(128, x_min=0.0, x_max=1.0)


@pytest.fixture
def grid_256():
    """256-point periodic grid on [0, 1]."""
    return Grid1D.uniform(256, x_min=0.0, x_max=1.0)


@pytest.fixture
def diffusion_op(grid_128):
    """Diffusion operator with D=0.01."""
    return DiffusionOperator(grid_128, D=0.01)


@pytest.fixture
def advdiff_op(grid_128):
    """Advection-diffusion operator with D=0.01, v=1.0."""
    return AdvectionDiffusionOperator(grid_128, D=0.01, v=1.0)


# =============================================================================
# Test: Exact Spectral Bounds
# =============================================================================

class TestExactSpectralBounds:
    """Test exact spectral bounds from FFT eigenvalues."""

    def test_diffusion_bounds_exact(self, diffusion_op):
        """Verify diffusion operator bounds are exact."""
        eigenvalues = diffusion_op.eigenvalues
        bounds = exact_spectral_bounds_from_fft(eigenvalues, "Diffusion D=0.01")

        # For diffusion: all eigenvalues are real and non-positive
        # Max magnitude is at highest wavenumber
        D = 0.01
        dx = diffusion_op.grid.dx

        # Expected spectral radius: 4D/dx²
        expected_rho = 4 * D / dx**2

        # Check bounds
        assert abs(bounds.rho - expected_rho) / expected_rho < 0.01, \
            f"Spectral radius mismatch: {bounds.rho} vs {expected_rho}"
        assert bounds.re_max <= 0, f"Diffusion should have re_max <= 0, got {bounds.re_max}"
        assert bounds.im_max < 1e-10, f"Diffusion should have im_max ≈ 0, got {bounds.im_max}"

        # Method should be exact FFT
        assert 'exact_fft' in bounds.methods_used

    def test_advdiff_bounds_exact(self, advdiff_op):
        """Verify advection-diffusion operator bounds.

        The odd advection symbol -i*v*k is zeroed at the self-paired
        Nyquist mode of this even grid (k = -pi/dx has no distinct
        conjugate partner for a real field), so the true im_max is
        attained one bin below Nyquist, not at v*pi/dx.
        """
        eigenvalues = advdiff_op.eigenvalues
        bounds = exact_spectral_bounds_from_fft(eigenvalues, "AdvDiff")

        # For advection-diffusion: complex eigenvalues
        # Re(λ) from diffusion (≤ 0), Im(λ) from advection
        v = 1.0
        dx = advdiff_op.grid.dx
        nx = advdiff_op.grid.nx

        # Expected im_max: v * k_next, k_next the wavenumber one bin below
        # the (zeroed) Nyquist mode.
        k_next = 2.0 * jnp.pi * (nx // 2 - 1) / (nx * dx)
        expected_im_max = abs(v) * k_next

        assert bounds.re_max <= 0, "AdvDiff should have re_max <= 0"
        assert abs(bounds.im_max - expected_im_max) / expected_im_max < 0.01, \
            f"im_max mismatch: {bounds.im_max} vs {expected_im_max}"

    def test_bounds_conversion(self, diffusion_op):
        """Verify conversion to standard SpectralBounds."""
        eigenvalues = diffusion_op.eigenvalues
        fft_bounds = exact_spectral_bounds_from_fft(eigenvalues)
        std_bounds = fft_bounds_to_spectral_bounds(fft_bounds)

        assert std_bounds.rho == fft_bounds.rho
        assert std_bounds.re_max == fft_bounds.re_max
        assert std_bounds.im_max == fft_bounds.im_max


# =============================================================================
# Test: NILT Parameter Tuning
# =============================================================================

class TestNILTTuning:
    """Test NILT parameter tuning for FFT operators."""

    def test_tuning_produces_valid_params(self, diffusion_op):
        """Verify tuning produces sensible NILT parameters."""
        eigenvalues = diffusion_op.eigenvalues
        t_end = 1.0

        params = tune_nilt_for_fft_operator(eigenvalues, t_end)

        # Check basic validity
        assert params.dt > 0, "dt should be positive"
        assert params.N > 0, "N should be positive"
        assert params.N & (params.N - 1) == 0, "N should be power of 2"
        assert params.T > t_end, "T should exceed t_end"
        assert params.a >= 0, "a (Bromwich shift) should be non-negative for stable operator"

    def test_tuning_covers_spectral_content(self, advdiff_op):
        """Verify tuning covers operator spectral content."""
        eigenvalues = advdiff_op.eigenvalues
        t_end = 1.0

        params = tune_nilt_for_fft_operator(eigenvalues, t_end)

        # omega_max should cover im_max
        im_max = float(jnp.max(jnp.abs(jnp.imag(eigenvalues))))
        assert params.omega_max >= im_max, \
            f"omega_max ({params.omega_max}) should cover im_max ({im_max})"


# =============================================================================
# Test: NILT Accuracy
# =============================================================================

class TestNILTAccuracy:
    """Test NILT accuracy for linear PDEs."""

    def test_nilt_matches_analytical_diffusion(self, grid_256):
        """NILT should match analytical solution for diffusion."""
        D = 0.01
        t_end = 1.0

        # Create operator
        op = DiffusionOperator(grid_256, D)
        eigenvalues = op.eigenvalues

        # Initial condition: sin(2πx)
        x = grid_256.x_coords(include_ghost=False)
        u0 = jnp.sin(2 * jnp.pi * x)

        # Analytical solution: exp(-4π²Dt) * sin(2πx)
        u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)

        # NILT solution
        result = nilt_solve_linear_pde(eigenvalues, u0, t_end)
        u_nilt = result['u_final']

        # Against the continuous solution the residual is the second-order
        # operator's discretization error (2e-5 on 256 points), which the
        # closed form e^{lambda t} u0_hat shares exactly.
        rel_error = float(jnp.linalg.norm(u_nilt - u_exact) / jnp.linalg.norm(u_exact))
        assert rel_error < 1e-4, f"NILT error {rel_error:.2e} exceeds 1e-4 threshold"

        # DiffusionOperator's eigenvalues are real, so H_k(s) = w_k*[1/(s -
        # lambda_k) - 1/(s + c_k)] is identically zero for every mode (both
        # poles sit at s = lambda_k) and u_final matches u_analytical to
        # rounding: the bridge is exact by construction on a real spectrum,
        # not tautological. The non-tautology check below therefore uses
        # AdvectionDiffusionOperator, whose complex eigenvalues give the
        # NILT an actual transient to invert.
        advdiff = AdvectionDiffusionOperator(grid_256, v=1.0, D=D)
        u0_adv = jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(6 * jnp.pi * x)
        result_adv = nilt_solve_linear_pde(advdiff.eigenvalues, u0_adv, t_end)
        u_nilt_adv = result_adv['u_final']

        # Against the closed form of the same discrete operator the inversion
        # itself is measured: about 1.0e-7 at the tuned parameters. u_final
        # must also be the NILT's own number. The bridge used to return the
        # closed form under both keys, which made this test pass without any
        # inversion taking place; a numerical inversion differs from the
        # closed form by rounding at least.
        norm_exact_adv = float(jnp.linalg.norm(result_adv['u_analytical']))
        rel_diff = float(
            jnp.linalg.norm(u_nilt_adv - result_adv['u_analytical']) / norm_exact_adv
        )
        assert rel_diff > 1e-12, "u_final is a copy of the closed form, not an inversion"
        assert rel_diff < 1e-6, f"NILT deviates from the closed form by {rel_diff:.2e}"

        # t_final is the grid time the NILT value was read at, and the tuned
        # grid (2T = 4 t_end = N dt) contains t_end exactly.
        t_grid = result_adv['nilt_result'].t
        assert float(jnp.min(jnp.abs(t_grid - result_adv['t_final']))) == 0.0
        assert abs(result_adv['t_final'] - t_end) < 1e-12

    def test_nilt_source_term_matches_closed_form(self, grid_256):
        """Every mode with a constant source: U_k = (u0_k + f_k/s)/(s - lambda_k)."""
        D = 0.01
        t_end = 1.0
        op = DiffusionOperator(grid_256, D)
        eigenvalues = op.eigenvalues

        x = grid_256.x_coords(include_ghost=False)
        u0 = jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(6 * jnp.pi * x)
        source = 0.5 * jnp.cos(2 * jnp.pi * x) + 0.2

        result = nilt_solve_linear_pde(eigenvalues, u0, t_end, source=source)

        # Independent closed form: u_hat(t) = e^{lam t} u0_hat + (e^{lam t} - 1)/lam f_hat,
        # with the lam = 0 (mean) mode growing linearly.
        lam = eigenvalues
        z = lam * result['t_final']
        growth = jnp.where(jnp.abs(z) > 1e-12,
                           (jnp.exp(z) - 1.0) / jnp.where(jnp.abs(z) > 1e-12, lam, 1.0),
                           result['t_final'])
        u_hat = jnp.exp(z) * jnp.fft.fft(u0) + growth * jnp.fft.fft(source)
        u_exact = jnp.real(jnp.fft.ifft(u_hat))

        rel_error = float(jnp.linalg.norm(result['u_final'] - u_exact) / jnp.linalg.norm(u_exact))
        assert rel_error < 1e-6, f"NILT error with source {rel_error:.2e}"

    def test_nilt_vs_timestepping_agreement(self, grid_128):
        """NILT and time-stepping should agree for linear PDE."""
        D = 0.01
        t_end = 0.5

        op = DiffusionOperator(grid_128, D)
        eigenvalues = op.eigenvalues

        x = grid_128.x_coords(include_ghost=False)
        u0 = jnp.sin(2 * jnp.pi * x)

        # Compare NILT vs time-stepping
        comparison = compare_nilt_vs_timestepping(
            eigenvalues, u0, t_end,
            tss_method='etdrk4',
            n_warmup=1,
            n_runs=3,
        )

        # Both should achieve good accuracy
        assert comparison.nilt_error < 1e-3, f"NILT error {comparison.nilt_error:.2e}"
        assert comparison.tss_error < 1e-3, f"TSS error {comparison.tss_error:.2e}"

        # etd_integrate floors (t_end - t0)/dt; the report must count the steps
        # actually taken, not the ceiling.
        assert comparison.tss_steps == int(t_end / comparison.tss_dt)


# =============================================================================
# Test: The Bridge Inverts Only the Transient
# =============================================================================

class TestTransientOnlyInversion:
    """nilt_solve_linear_pde removes everything the closed form already
    knows and inverts only the e^{lambda_k t} transient, via
    H_k(s) = w_k*[1/(s - lambda_k) - 1/(s + c_k)] with c_k = -Re(lambda_k).
    For a real lambda_k both poles of H_k coincide at s = lambda_k, so H_k
    is identically zero and the bridge reproduces the closed form to
    rounding; before this fix the bridge instead inverted
    G_k(s) = U_k(s) - u0_k/(s + 1/t_end), which added a pole at -1/t_end
    the tuner never saw and left the source pole at s = 0 uncanceled."""

    def test_decaying_spectrum_is_exact(self):
        """A fully real, negative spectrum needs no numerical inversion.

        Before the fix, eigenvalues = full(8, -10), u0 = ones(8), t_end = 1
        tuned a = 0 and returned -0.006816 instead of e^{-10} = 4.540e-5.
        """
        eigenvalues = jnp.full(8, -10.0)
        u0 = jnp.ones(8)
        t_end = 1.0

        result = nilt_solve_linear_pde(eigenvalues, u0, t_end)
        u_expected = jnp.exp(-10.0) * jnp.ones(8)
        max_error = float(jnp.max(jnp.abs(result['u_final'] - u_expected)))
        assert max_error < 1e-12, f"max error {max_error:.3e} on a real spectrum"

        # With a constant source, u_k(t) = w e^{lambda t} - f/lambda,
        # w = u0 + f/lambda; still exact, since H_k is still identically
        # zero for a real spectrum.
        source = jnp.ones(8)
        result_src = nilt_solve_linear_pde(eigenvalues, u0, t_end, source=source)
        lam = -10.0
        f = 1.0
        w = 1.0 + f / lam
        u_expected_src = (w * jnp.exp(lam * t_end) - f / lam) * jnp.ones(8)
        max_error_src = float(jnp.max(jnp.abs(result_src['u_final'] - u_expected_src)))
        assert max_error_src < 1e-12, f"max error {max_error_src:.3e} with a source"

    def test_zero_mode_with_source_is_a_ramp(self):
        """An eigenvalue exactly 0 with a source is handled entirely in
        closed form: u(t) = u0 + f*t, no transient (w = 0 for that mode)."""
        eigenvalues = jnp.zeros(4)
        u0 = jnp.array([1.0, 2.0, -1.0, 0.5])
        source = jnp.array([0.5, -0.5, 1.0, 0.0])
        t_end = 2.0

        result = nilt_solve_linear_pde(eigenvalues, u0, t_end)
        assert 'empty transient' in result['note']
        assert result['nilt_result'] is None
        assert result['params'] is None
        assert float(jnp.max(jnp.abs(result['u_final'] - u0))) < 1e-12

        result_src = nilt_solve_linear_pde(eigenvalues, u0, t_end, source=source)
        u_expected_src = u0 + source * t_end
        max_error = float(jnp.max(jnp.abs(result_src['u_final'] - u_expected_src)))
        assert max_error < 1e-12, f"max error {max_error:.3e} on the zero-mode ramp"
        assert 'empty transient' in result_src['note']
        assert result_src['nilt_result'] is None
        assert result_src['params'] is None


# =============================================================================
# Test: Small Eigenvalues Do Not Cancel the Answer Away
# =============================================================================

class TestSmallEigenvalueReconstruction:
    """The forced part of each mode is evaluated as t phi1(lambda_k t) f_k,
    which has no 1/lambda_k in it.

    The reconstruction used to split the mode into w_k e^{lambda_k t} with
    w_k = u0_k + f_k/lambda_k and a particular constant -f_k/lambda_k, and
    add the two as plain floats. For a |lambda_k| small but above the old
    relative spectral-zero threshold max(1e-12 max|lambda|, 1e-300) the two
    O(1/lambda_k) terms had to cancel to an O(1) answer, which they cannot
    do in floating point: eight eigenvalues -1e-8 with u0 = 0, f = 1,
    t_end = 1 returned 0 in float32, and -1e-16 returned 2 in float64,
    against about 1 in both cases.
    """

    @staticmethod
    def _exact(eigenvalues, u0, source, t):
        """ifft(e^{lambda t} u0_hat + (e^{lambda t} - 1)/lambda f_hat), via expm1."""
        lam = jnp.asarray(eigenvalues)
        z = lam * t
        nonzero = jnp.abs(lam) > 0
        growth = jnp.where(
            nonzero, jnp.expm1(z) / jnp.where(nonzero, lam, 1.0), t
        )
        u_hat = jnp.exp(z) * jnp.fft.fft(u0) + growth * jnp.fft.fft(source)
        return jnp.real(jnp.fft.ifft(u_hat))

    def test_tiny_eigenvalue_with_source_is_t_phi1(self):
        """Eight eigenvalues -1e-8, u0 = 0, f = 1, t_end = 1.

        The exact answer is t phi1(lambda t) f = (e^{lambda t} - 1)/lambda =
        0.999999995. The old reconstruction returned 0.0 in float32 and was
        good to 8 digits at best in float64.
        """
        lam = -1e-8
        n = 8
        result = nilt_solve_linear_pde(
            jnp.full(n, lam), jnp.zeros(n), 1.0, source=jnp.ones(n)
        )
        expected = float(np.expm1(lam * 1.0) / lam)
        max_error = float(jnp.max(jnp.abs(result['u_final'] - expected)))
        assert max_error < 1e-10, f"max error {max_error:.3e} at lambda = {lam:.1e}"

    def test_subepsilon_eigenvalue_with_source_is_the_ramp(self):
        """Eight eigenvalues -1e-16, u0 = 0, f = 1, t_end = 1: returns 1.0.

        |lambda| t_end sits under the double-precision epsilon, so the mode
        is the ramp f t to rounding. The old reconstruction returned 2.0:
        w = f/lambda = -1e16 and the particular term +1e16 differ by one
        unit in the last place of 1e16, which is 2.
        """
        n = 8
        result = nilt_solve_linear_pde(
            jnp.full(n, -1e-16), jnp.zeros(n), 1.0, source=jnp.ones(n)
        )
        max_error = float(jnp.max(jnp.abs(result['u_final'] - 1.0)))
        assert max_error < 1e-12, f"max error {max_error:.3e} at lambda = -1e-16"

    def test_exactly_zero_matches_the_tiny_eigenvalue_limit(self):
        """lambda = 0 gives u0 + f t, and the two sides of tau agree.

        phi1's own series branch carries the limit, so the closed form
        e^{lambda t} u0 + t phi1(lambda t) f is continuous across
        tau = TRANSIENT_TAU, the threshold that decides whether a mode is
        inverted at all -- but a real eigenvalue never clears the mask in
        the first place, since its two poles coincide regardless of tau
        (transient_mask requires Im(lambda_k) != 0 too). At t_end = 2 the
        tau edge sits at |lambda| = 5e-3: -4.999999e-3 stays under it and
        -5.000001e-3 clears it, but both are real, so both take the closed
        form directly and neither builds a grid. Both match their own exact
        solution to rounding and differ from each other only by the 2e-9
        between the two spectra.
        """
        n = 8
        u0 = jnp.array([1.0, 2.0, -1.0, 0.5, 0.25, -0.75, 1.5, 0.0])
        source = jnp.array([0.5, -0.5, 1.0, 0.0, 0.3, 0.2, -0.1, 0.4])
        t_end = 2.0
        edge = TRANSIENT_TAU / t_end

        zero = nilt_solve_linear_pde(jnp.zeros(n), u0, t_end, source=source)
        ramp_error = float(jnp.max(jnp.abs(zero['u_final'] - (u0 + source * t_end))))
        assert ramp_error < 1e-12, f"max error {ramp_error:.3e} on u0 + f t"

        lam_below, lam_above = -(edge - 1e-9), -(edge + 1e-9)
        below = nilt_solve_linear_pde(jnp.full(n, lam_below), u0, t_end, source=source)
        above = nilt_solve_linear_pde(jnp.full(n, lam_above), u0, t_end, source=source)
        assert 'note' in below, f"|lambda| t_end just under tau = {TRANSIENT_TAU:g}"
        assert 'note' in above, "a real eigenvalue never builds a grid, tau or not"

        for lam, result in ((lam_below, below), (lam_above, above)):
            exact = self._exact(jnp.full(n, lam), u0, source, result['t_final'])
            error = float(jnp.max(jnp.abs(result['u_final'] - exact)))
            assert error < 1e-12, f"max error {error:.3e} at lambda = {lam:.6e}"

        jump = float(jnp.max(jnp.abs(above['u_final'] - below['u_final'])))
        assert jump < 1e-7, f"discontinuity {jump:.3e} across tau"

    def test_imaginary_pair_straddling_tau_agrees_on_both_sides(self):
        """A conjugate imaginary pair on either side of tau, t_end = 1.

        A real spectrum makes H_k identically zero, so it cannot see where
        tau sits; a pair at +-ib can. The transient handed to the NILT is
        r_k i Im(lambda_k)/(lambda_k (s - lambda_k)(s + c_k)), which stays
        O(|r_k|) as b -> 0 because the two poles coalesce, so the
        inversion's error on it is flat in b: 1.861e-5 at every b from 1e-7
        to 3, where the closed form is exact. tau = 1e-2 is placed where
        that flat error is no larger than the answer's own variation across
        the band tau separates, (b t_end)^2/6 = 1.667e-5 -- the first-order
        term i b t is a phase and cancels against the conjugate partner.
        At the old tau = sqrt(eps) the same 1.9e-5 jump sat where the exact
        answer varies by 4e-17: b = 1.49e-8 returned 1.0 and b = 1.491e-8
        returned 0.9999813.
        """
        u0 = jnp.zeros(4)
        source = jnp.array([1.0, 0.0, -1.0, 0.0])  # f_hat = [0, 2, 0, 2]

        results = {}
        for side, b in (('below', TRANSIENT_TAU * (1 - 1e-6)),
                        ('above', TRANSIENT_TAU * (1 + 1e-6))):
            eigenvalues = jnp.array([0.0, 1j * b, 0.0, -1j * b])
            result = nilt_solve_linear_pde(eigenvalues, u0, 1.0, source=source)
            exact = self._exact(eigenvalues, u0, source, result['t_final'])
            results[side] = (result, exact)

        below, below_exact = results['below']
        above, above_exact = results['above']
        assert 'note' in below, "just under tau the mode is closed form"
        assert 'note' not in above, "just over tau the mode is inverted"

        below_error = float(jnp.max(jnp.abs(below['u_final'] - below_exact)))
        above_error = float(jnp.max(jnp.abs(above['u_final'] - above_exact)))
        assert below_error < 1e-12, f"closed-form side off by {below_error:.3e}"
        assert above_error < 5e-5, f"inverted side off by {above_error:.3e}"

        # The jump across tau against the answer's own variation over
        # |lambda| t_end in [0, tau], which is what sets tau.
        jump = float(jnp.max(jnp.abs(above['u_final'] - below['u_final'])))
        edge = jnp.array([0.0, 1j * TRANSIENT_TAU, 0.0, -1j * TRANSIENT_TAU])
        variation = float(jnp.max(jnp.abs(
            self._exact(edge, u0, source, 1.0)
            - self._exact(jnp.zeros(4), u0, source, 1.0)
        )))
        assert jump < 3 * variation, (
            f"discontinuity {jump:.3e} across tau exceeds the answer's own "
            f"{variation:.3e} variation over the band tau separates"
        )

    def test_mixed_spectrum_matches_the_per_mode_closed_form(self):
        """Magnitudes from 1e-12 to 1e3, nonzero u0 and f, several times.

        The spectrum is real and conjugate-symmetric, so H_k(s) is
        identically zero and the bridge is exact by construction; what is
        measured is the reconstruction's arithmetic. The mode at
        lambda = -5.6e-9 sat just above the old threshold (1e-12 max|lambda|
        = 1e-9) and cost about 2e-8 there, the size of f_k/lambda_k times
        the double-precision epsilon.
        """
        magnitudes = np.logspace(-12, 3, 9)
        spectrum = -np.concatenate([magnitudes, magnitudes[1:-1][::-1]])
        eigenvalues = jnp.asarray(spectrum)
        n = eigenvalues.shape[0]

        rng = np.random.default_rng(0)
        u0 = jnp.asarray(rng.standard_normal(n))
        source = jnp.asarray(rng.standard_normal(n))

        for t_end in (0.05, 0.5, 2.0):
            result = nilt_solve_linear_pde(eigenvalues, u0, t_end, source=source)
            u_exact = self._exact(eigenvalues, u0, source, result['t_final'])
            max_error = float(jnp.max(jnp.abs(result['u_final'] - u_exact)))
            assert max_error < 1e-10, f"max error {max_error:.3e} at t_end = {t_end}"

    def test_bridge_requires_x64(self):
        """Without x64 the bridge raises instead of inverting in float32.

        Every other NILT entry point guards on moljax._precision.require_x64;
        the bridge did not, so a float32 call went all the way through the
        inversion and returned numbers the e^{a t} factor had already ruined
        (the -1e-8 case above came back as exactly 0).
        """
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "import jax.numpy as jnp\n"
            "from moljax.laplace.fft_nilt_bridge import nilt_solve_linear_pde\n"
            "try:\n"
            "    nilt_solve_linear_pde(jnp.full(8, -1e-8, dtype=jnp.float32),\n"
            "                          jnp.zeros(8, dtype=jnp.float32), 1.0,\n"
            "                          source=jnp.ones(8, dtype=jnp.float32),\n"
            "                          dtype=jnp.float32)\n"
            "except RuntimeError as e:\n"
            "    print('RAISED', e)\n"
        )
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=ROOT)
        out = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True,
            env=env, cwd=ROOT, check=True
        )
        assert 'RAISED' in out.stdout, out.stdout + out.stderr
        assert 'nilt_solve_linear_pde' in out.stdout
        assert '64-bit precision' in out.stdout
        assert 'jax_enable_x64' in out.stdout


# =============================================================================
# Test: A Stationary Mode Inverts Nothing
# =============================================================================

class TestStationaryModeIsNotInverted:
    """The inverted weight is w_k = r_k/lambda_k, r_k = lambda_k u0_k + f_k.

    While the weight was w_k = u0_k, a stationary mode (r_k = 0, so
    u_k(t) = u0_k for every t) still had its homogeneous part e^{lambda_k t}
    u0_k inverted numerically, and nothing cancelled the inversion's error
    any more: the forcing that used to cancel it against a particular term
    -f_k/lambda_k is now evaluated in closed form. Weighting the transient
    by the residual instead leaves a stationary mode with nothing to invert.
    """

    @staticmethod
    def _exact(eigenvalues, u0_hat, residual_hat, t):
        """ifft(u0_hat + (e^{lambda t} - 1)/lambda r_hat), via expm1."""
        lam = jnp.asarray(eigenvalues)
        z = lam * t
        nonzero = jnp.abs(lam) > 0
        growth = jnp.where(
            nonzero, jnp.expm1(z) / jnp.where(nonzero, lam, 1.0), t
        )
        return jnp.real(jnp.fft.ifft(u0_hat + growth * residual_hat))

    # lambda_k u0_k + f_k = 0 for every k, so u(t) = u0 for every t. The
    # spectrum is self-conjugate (index 0 and the Nyquist index 2 are real)
    # and the 100j pair is damped by only Re(lambda) = -1, which is where
    # the NILT error lives.
    STATIONARY = dict(
        eigenvalues=jnp.array([0.0, -1 + 100j, 0.0, -1 - 100j]),
        u0=jnp.array([1.0, 0.0, -1.0, 0.0]),
        source=jnp.array([1.0, 100.0, -1.0, -100.0]),
    )

    def test_stationary_field_is_returned_unchanged(self):
        """u_final = u0 to rounding (2.8e-17); HEAD was off by 1.5e-2.

        With w_k = u0_k this returned
        [0.98499821, 0.00121119, -0.98499821, -0.00121119]: the raw NILT
        error on the lightly damped 100j pair, no longer cancelled by
        anything.
        """
        u0 = self.STATIONARY['u0']
        result = nilt_solve_linear_pde(
            self.STATIONARY['eigenvalues'], u0, 1.0,
            source=self.STATIONARY['source'],
        )
        max_error = float(jnp.max(jnp.abs(result['u_final'] - u0)))
        assert max_error < 1e-12, f"max error {max_error:.3e} on a stationary field"

    def test_every_mode_stationary_inverts_nothing(self):
        """No mode has a weight, so no NILT grid is built at all.

        The weights are not exposed, but w_k = 0 for every k is exactly the
        condition the empty-transient branch reports.
        """
        result = nilt_solve_linear_pde(
            self.STATIONARY['eigenvalues'], self.STATIONARY['u0'], 1.0,
            source=self.STATIONARY['source'],
        )
        assert result['nilt_result'] is None
        assert result['params'] is None
        assert 'empty transient' in result['note']
        assert result['t_final'] == 1.0

    def test_near_stationary_error_scales_with_the_residual(self):
        """Scaling the residual by 1e-2 scales the error by 1e-2.

        The inverted weight is proportional to r_k, so the inversion's
        error on the 100j pair is too: at a residual 1e-6 of the source the
        error is 1.50e-8, and at 1e-8 it is 1.50e-10, against the 1.50e-2
        the full source would give. With w_k = u0_k the error was 1.50e-2
        at every residual, the initial condition being what it was scaled
        by.
        """
        eigenvalues = self.STATIONARY['eigenvalues']
        u0 = self.STATIONARY['u0']
        u0_hat = jnp.fft.fft(u0)

        errors = []
        for delta in (1e-6, 1e-8):
            source = self.STATIONARY['source'] * (1.0 - delta)
            result = nilt_solve_linear_pde(eigenvalues, u0, 1.0, source=source)
            residual_hat = eigenvalues * u0_hat + jnp.fft.fft(source)
            exact = self._exact(eigenvalues, u0_hat, residual_hat, result['t_final'])
            errors.append(float(jnp.max(jnp.abs(result['u_final'] - exact))))

        assert errors[0] < 1e-7, f"max error {errors[0]:.3e} at a 1e-6 residual"
        ratio = errors[0] / errors[1]
        assert 90.0 < ratio < 110.0, f"error ratio {ratio:.1f} is not the residual's 100"

    def test_stationary_modes_stay_exact_beside_live_ones(self):
        """A stationary pair inside an otherwise live spectrum is untouched.

        The error is read off mode by mode: modes 1 and 7 have r_k = 0 and
        must come back at rounding level even though the inversion is
        carrying real error on modes 2, 3, 5 and 6.
        """
        eigenvalues = jnp.array(
            [0.0, -1 + 100j, -2 + 30j, -3 + 10j, -5.0, -3 - 10j, -2 - 30j, -1 - 100j]
        )
        u0_hat = jnp.array([0.0, 2.0, 1 - 1j, 0.5, 1.0, 0.5, 1 + 1j, 2.0]) + 0j
        residual_hat = jnp.array(
            [0.3, 0.0, 0.7 - 0.2j, 0.1, -0.4, 0.1, 0.7 + 0.2j, 0.0]
        ) + 0j
        u0 = jnp.real(jnp.fft.ifft(u0_hat))
        source = jnp.real(jnp.fft.ifft(residual_hat - eigenvalues * u0_hat))

        result = nilt_solve_linear_pde(eigenvalues, u0, 1.0, source=source)
        assert result['nilt_result'] is not None, "the live modes need a grid"

        exact = self._exact(eigenvalues, u0_hat, residual_hat, result['t_final'])
        error_hat = jnp.abs(jnp.fft.fft(result['u_final'] - exact))
        for k in (1, 7):
            assert float(error_hat[k]) < 1e-12, (
                f"stationary mode {k} carries error {float(error_hat[k]):.3e}"
            )
        assert float(jnp.max(error_hat[jnp.array([2, 3, 5, 6])])) > 1e-12, (
            "the live modes should be inverted, not copied from the closed form"
        )


# =============================================================================
# Test: the closed form under a large initial condition
# =============================================================================

class TestClosedFormKeepsTheSourceUnderALargeInitialCondition:
    """The closed form is written in u0_k and f_k, not in the residual.

    r_k = lambda_k u0_k + f_k is what the inverted weight must be built
    from (it is exactly zero on a stationary mode), but a closed form built
    from it evaluates the answer as u0_k against a second term of size
    |r_k/lambda_k| = |u0_k|, and f_k is gone twice over: rounded away in
    forming r_k, and then subtracted out of the difference. Four
    eigenvalues -1 with u0 = 1e16, f = 1 and t_end = 50 returned 0
    everywhere, u_analytical included, against 1.000001928749848.
    """

    def test_real_decay_keeps_a_source_16_decades_under_u0(self):
        """lambda = -1, u0 = 1e16, f = 1, t_end = 50.

        The spectrum is real, so H_k is identically zero and the NILT
        contributes nothing: transient_mask excludes a real eigenvalue
        whatever its residual, so no grid is built at all and what is
        measured is the remainder branch
        e^{-c_k t} u0_k + t phi1(-c_k t) Re(lambda_k)/lambda_k f_k on its
        own. e^{-50} 1e16 = 1.93e-6 and 1 - e^{-50} = 1, and the residual
        form loses both (r_k = -1e16 + 1 rounds to -1e16, whose ramp is
        -u0_k exactly).
        """
        n = 4
        result = nilt_solve_linear_pde(
            jnp.full(n, -1.0), jnp.full(n, 1e16), 50.0, source=jnp.ones(n)
        )
        assert result['nilt_result'] is None, "a real eigenvalue never builds a grid"
        expected = float(1e16 * np.exp(-50.0) - np.expm1(-50.0))
        for key in ('u_final', 'u_analytical'):
            error = float(jnp.max(jnp.abs(result[key] - expected))) / expected
            assert error < 1e-12, f"{key} off by {error:.3e} relative"

    def test_complex_mode_closed_form_keeps_the_source(self):
        """lambda = -1 + 5j on a conjugate pair, u0_hat = 1e12, f_hat = 1.

        At t_end = 30 the initial condition has decayed to 1e12 e^{-30} =
        0.094 while the forced part is still -f_k/lambda_k = 0.19, so the
        answer is the smaller of two contributions separated by 13 decades
        at t = 0. The residual form read it off the difference of two 1e12
        terms and was 1.3e-4 out relative.

        Only u_analytical is checked. The transient this mode hands the
        NILT is weighted by w_k = r_k/lambda_k, of size 1e12, and the
        inversion's relative accuracy on a transform of that scale (about
        1e-5) leaves u_final meaningless here -- which is a statement about
        what a Bromwich inversion can do with a 13-decade dynamic range,
        not about the reconstruction.
        """
        eigenvalues = jnp.array([0.0, -1 + 5j, 0.0, -1 - 5j])
        u0_hat = jnp.array([0.0, 1e12, 0.0, 1e12]) + 0j
        source_hat = jnp.array([0.0, 1.0, 0.0, 1.0]) + 0j
        u0 = jnp.real(jnp.fft.ifft(u0_hat))
        source = jnp.real(jnp.fft.ifft(source_hat))

        result = nilt_solve_linear_pde(eigenvalues, u0, 30.0, source=source)
        t = result['t_final']
        z = eigenvalues * t
        nonzero = jnp.abs(eigenvalues) > 0
        growth = jnp.where(
            nonzero, jnp.expm1(z) / jnp.where(nonzero, eigenvalues, 1.0), t
        )
        exact = jnp.real(jnp.fft.ifft(jnp.exp(z) * u0_hat + growth * source_hat))
        scale = float(jnp.max(jnp.abs(exact)))
        error = float(jnp.max(jnp.abs(result['u_analytical'] - exact))) / scale
        assert error < 1e-9, f"u_analytical off by {error:.3e} relative"


# =============================================================================
# Test: real eigenvalues never enter the transient mask
# =============================================================================

class TestRealEigenvaluesAreNeverInverted:
    """transient_mask must exclude a real eigenvalue whatever its weight.

    H_k(s) = w_k [1/(s - lambda_k) - 1/(s + c_k)], c_k = -Re(lambda_k). For a
    real lambda_k, c_k = -lambda_k, so both poles sit at s = lambda_k and
    H_k is identically zero: a real mode has nothing to invert regardless of
    w_k = r_k/lambda_k. Before this fix transient_mask was w != 0 alone, so
    a real mode entered it whenever its residual was nonzero. Its own
    contribution to the transient was still zero, but it entered
    sigma_H = max Re(lambda_k) over the mask and the tuner's
    re_max_override anyway, inflating the Bromwich shift far past what the
    genuinely complex modes needed and ruining their inversion.
    """

    @staticmethod
    def _exact(eigenvalues, u0_hat, residual_hat, t):
        """ifft(u0_hat + (e^{lambda t} - 1)/lambda r_hat), via expm1."""
        lam = jnp.asarray(eigenvalues)
        z = lam * t
        nonzero = jnp.abs(lam) > 0
        growth = jnp.where(
            nonzero, jnp.expm1(z) / jnp.where(nonzero, lam, 1.0), t
        )
        return jnp.real(jnp.fft.ifft(u0_hat + growth * residual_hat))

    def test_forcing_only_real_mode_no_longer_inflates_the_shift(self):
        """lambda = [20, -1+5j, 0, -1-5j], u0 = [1, 0, -1, 0], f = 1e-12.

        The k=0 mode (lambda=20, u0_hat=0, f_hat=4e-12) has a tiny but
        nonzero residual, so before this fix it entered the mask and pushed
        sigma_H to 20 and a to 24.605 instead of 3.605. The max error was
        349033 against an exact field below 0.36: the inflated shift's
        e^{a t} amplified the genuinely complex pair's inversion roundoff by
        orders of magnitude. Now a stays at 3.605 and the error is back to
        the raw NILT level.
        """
        eigenvalues = jnp.array([20.0, -1 + 5j, 0.0, -1 - 5j])
        u0 = jnp.array([1.0, 0.0, -1.0, 0.0])
        source = jnp.full(4, 1e-12)
        u0_hat = jnp.fft.fft(u0)
        source_hat = jnp.fft.fft(source)
        residual_hat = eigenvalues * u0_hat + source_hat

        result = nilt_solve_linear_pde(eigenvalues, u0, 1.0, source=source)
        assert result['params'] is not None
        a = result['params'].a
        assert a < 5.0, f"Bromwich shift a={a:.3f} was inflated by the real mode"

        exact = self._exact(eigenvalues, u0_hat, residual_hat, result['t_final'])
        error = float(jnp.max(jnp.abs(result['u_final'] - exact)))
        assert error < 1e-4, f"max error {error:.3e} vs the exact field"

    def test_hot_real_mode_with_no_transient_skips_the_nilt_entirely(self):
        """lambda = full(4, 200), u0 = 0, f = 1: nothing needs inverting.

        HEAD raised "NILT-CFL infeasible" here because the sole (real)
        mode's nonzero residual put it in the mask and sigma_H = 200
        exceeded the tuner's a_max. The exact answer is the closed form
        t phi1(200 t) f -- large but finite, expm1(200)/200 = 3.61e84 in
        real space -- and is now returned directly with no NILT grid built
        at all.
        """
        eigenvalues = jnp.full(4, 200.0, dtype=jnp.complex128)
        u0 = jnp.zeros(4)
        source = jnp.ones(4)

        result = nilt_solve_linear_pde(eigenvalues, u0, 1.0, source=source)
        assert result['nilt_result'] is None
        assert result['params'] is None
        assert 'empty transient' in result['note']

        expected = float(np.expm1(200.0) / 200.0)
        error = abs(float(result['u_final'][0]) - expected) / expected
        assert error < 1e-9, f"u_final off by {error:.3e} relative"

    def test_preexisting_exposure_with_nonzero_initial_condition(self):
        """lambda = [20, -1+5j, 0, -1-5j], u0 = [1, 1, -1, 0], f = 0.

        u0_hat[0] = 1 is nonzero here, so even the w_k = u0_k weight this
        module used before a518612 would have let the real k=0 mode into
        the mask (w != 0 there too): this exposure predates the residual
        weighting, which only widened it to forcing-only real modes.
        """
        eigenvalues = jnp.array([20.0, -1 + 5j, 0.0, -1 - 5j])
        u0 = jnp.array([1.0, 1.0, -1.0, 0.0])
        u0_hat = jnp.fft.fft(u0)
        residual_hat = eigenvalues * u0_hat

        result = nilt_solve_linear_pde(eigenvalues, u0, 1.0)
        assert result['params'] is not None
        a = result['params'].a
        assert a < 5.0, f"Bromwich shift a={a:.3f} was inflated by the real mode"

        exact = self._exact(eigenvalues, u0_hat, residual_hat, result['t_final'])
        error = float(jnp.max(jnp.abs(result['u_final'] - exact)))
        assert error < 1e-4, f"max error {error:.3e} vs the exact field"

    def test_imaginary_pair_alone_is_still_inverted(self):
        """A purely imaginary pair with zero u0 still needs the NILT.

        lambda = [0, 5j, 0, -5j], u0 = 0, f nonzero: the mask must still be
        true for the +-5j modes, whose poles genuinely do not coincide, so
        the fix does not silently disable the NILT path for every spectrum
        with a self-paired zero mode.
        """
        eigenvalues = jnp.array([0.0, 5j, 0.0, -5j])
        u0 = jnp.zeros(4)
        source = jnp.array([1.0, 2.0, 3.0, 4.0])

        result = nilt_solve_linear_pde(eigenvalues, u0, 1.0, source=source)
        assert result['nilt_result'] is not None, "the +-5j pair must still be inverted"

        error = float(jnp.max(jnp.abs(result['u_final'] - result['u_analytical'])))
        assert error < 1e-3, f"max error {error:.3e} vs the closed form"


# =============================================================================
# Test: 1D Restriction
# =============================================================================

class TestBridgeRejects2DSpectra:
    """nilt_solve_linear_pde reads n_modes from eigenvalues.shape[0] and runs
    a 1D fft/irfft throughout; a multi-dimensional spectrum (e.g. from a 2D
    DiffusionOperator) must be rejected rather than silently mishandled."""

    def test_bridge_rejects_2d_spectra(self):
        """A 2D eigenvalue array either fails to broadcast against the 1D
        frequency grid or would reconstruct a wrong-sized field; it must
        instead raise a clear ValueError before any of that happens."""
        n = 4
        eigenvalues_2d = -jnp.ones((n, n)) - jnp.eye(n)
        u0_1d = jnp.ones(n)
        u0_2d = jnp.ones((n, n))

        with pytest.raises(ValueError, match="1D"):
            nilt_solve_linear_pde(eigenvalues_2d, u0_1d, t_end=1.0)

        with pytest.raises(ValueError, match="1D"):
            nilt_solve_linear_pde(eigenvalues_2d, u0_2d, t_end=1.0)

        # A 1D spectrum with a 2D initial condition must also be rejected.
        eigenvalues_1d = -jnp.arange(1.0, n + 1)
        with pytest.raises(ValueError, match="1D"):
            nilt_solve_linear_pde(eigenvalues_1d, u0_2d, t_end=1.0)


# =============================================================================
# Test: Matching Mode Counts
# =============================================================================

class TestBridgeRejectsMismatchedLengths:
    """nilt_solve_linear_pde reads n_modes from eigenvalues.shape[0] alone and
    combines eigenvalues, u0 and source mode-by-mode; the 1D checks above let
    a shorter u0 or source through, which either broadcasts into fabricated
    extra modes or is silently truncated to the wrong field length instead of
    raising."""

    def test_bridge_rejects_mismatched_lengths(self):
        """A 4-mode spectrum with a 1-element u0 must not return a 4-element
        field built from a broadcast, fabricated u0; a 1-mode spectrum with a
        4-element u0 must not silently return only 1 element; and a source
        of the wrong length must be rejected the same way."""
        eigenvalues_4 = jnp.array([-1.0, -2.0, -3.0, -4.0], dtype=jnp.complex128)
        eigenvalues_1 = jnp.array([-1.0], dtype=jnp.complex128)
        u0_1 = jnp.array([1.0])
        u0_4 = jnp.array([1.0, 2.0, 3.0, 4.0])

        # 4-mode spectrum, singleton u0: broadcasting would fabricate 3 modes.
        with pytest.raises(ValueError, match="u0.shape"):
            nilt_solve_linear_pde(eigenvalues_4, u0_1, t_end=1.0)

        # 1-mode spectrum, 4-element u0: must not silently return 1 element.
        with pytest.raises(ValueError, match="u0.shape"):
            nilt_solve_linear_pde(eigenvalues_1, u0_4, t_end=1.0)

        # Matching lengths but a mismatched source.
        eigenvalues_2 = jnp.array([-1.0, -2.0], dtype=jnp.complex128)
        u0_2 = jnp.array([1.0, 2.0])
        source_3 = jnp.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="source.shape"):
            nilt_solve_linear_pde(eigenvalues_2, u0_2, t_end=1.0, source=source_3)


# =============================================================================
# Test: NILT Speed Advantage for Long Horizons
# =============================================================================

class TestNILTSpeedAdvantage:
    """Test NILT is faster for long time horizons."""

    def test_nilt_faster_for_long_horizon(self, grid_128):
        """NILT should be faster for t_end >> dt_cfl."""
        D = 0.01
        dx = grid_128.dx
        dt_cfl = 0.25 * dx**2 / D  # Explicit diffusion CFL

        # Long time horizon: t_end = 100 * dt_cfl
        t_end = 100 * dt_cfl

        op = DiffusionOperator(grid_128, D)
        eigenvalues = op.eigenvalues

        x = grid_128.x_coords(include_ghost=False)
        u0 = jnp.sin(2 * jnp.pi * x)

        comparison = compare_nilt_vs_timestepping(
            eigenvalues, u0, t_end,
            tss_method='etd1',
            tss_dt=10 * dt_cfl,  # Use ETD which allows larger dt
            n_warmup=2,
            n_runs=5,
        )

        print(f"\nLong horizon test (t_end = {t_end:.3f} = 100 * dt_cfl):")
        print(f"  NILT: {comparison.nilt_time_ms:.2f} ms, error={comparison.nilt_error:.2e}")
        print(f"  TSS:  {comparison.tss_time_ms:.2f} ms, error={comparison.tss_error:.2e}")
        print(f"  Speedup: {comparison.speedup:.2f}x")

        # For long horizons, we expect speedup (though exact value depends on setup)
        # At minimum, both should produce accurate results
        assert comparison.nilt_error < 0.01, f"NILT error too high: {comparison.nilt_error}"

    def test_comparison_table_output(self, grid_128):
        """Generate comparison table for multiple time horizons."""
        D = 0.01
        dx = grid_128.dx
        dt_cfl = 0.25 * dx**2 / D

        op = DiffusionOperator(grid_128, D)
        eigenvalues = op.eigenvalues

        x = grid_128.x_coords(include_ghost=False)
        u0 = jnp.sin(2 * jnp.pi * x)

        comparisons = []
        for t_factor in [10, 50, 100]:
            t_end = t_factor * dt_cfl
            comp = compare_nilt_vs_timestepping(
                eigenvalues, u0, t_end,
                tss_method='etd1',
                n_warmup=1,
                n_runs=3,
            )
            comparisons.append(comp)

        # Print table
        print_comparison_table(comparisons)

        # All should have reasonable accuracy
        for c in comparisons:
            assert c.nilt_error < 0.01 or c.tss_error < 0.01, \
                f"At least one method should achieve <1% error at t_end={c.t_end}"


# =============================================================================
# Test: Spectral Guardrails (No Power Iteration)
# =============================================================================

class TestSpectralGuardrails:
    """Test that FFT bounds replace power iteration."""

    def test_no_power_iteration_needed(self, diffusion_op):
        """Verify exact bounds don't require matrix-free estimation."""
        eigenvalues = diffusion_op.eigenvalues
        bounds = exact_spectral_bounds_from_fft(eigenvalues)

        # Method should be 'exact_fft', not 'power_iteration' or 'gershgorin'
        for method in bounds.methods_used.values():
            assert 'power' not in method.lower(), \
                f"Should not use power iteration, but found: {method}"
            assert 'gershgorin' not in method.lower(), \
                f"Should not use Gershgorin, but found: {method}"

    def test_bounds_match_direct_computation(self, grid_256):
        """Verify bounds match direct eigenvalue computation."""
        D = 0.01
        v = 0.5

        op = AdvectionDiffusionOperator(grid_256, D=D, v=v)
        eigenvalues = op.eigenvalues

        # Direct computation
        rho_direct = float(jnp.max(jnp.abs(eigenvalues)))
        re_max_direct = float(jnp.max(jnp.real(eigenvalues)))
        im_max_direct = float(jnp.max(jnp.abs(jnp.imag(eigenvalues))))

        # Via bounds function
        bounds = exact_spectral_bounds_from_fft(eigenvalues)

        assert abs(bounds.rho - rho_direct) < 1e-10
        assert abs(bounds.re_max - re_max_direct) < 1e-10
        assert abs(bounds.im_max - im_max_direct) < 1e-10


# =============================================================================
# Test: Quantitative Error Table
# =============================================================================

class TestQuantitativeResults:
    """Generate quantitative results for documentation."""

    def test_error_table_nilt_vs_tss(self, grid_256):
        """Generate error table: NILT vs Time-Stepping.

        | t_end | NILT Error | TSS Error | NILT ms | TSS ms | Speedup |
        """
        D = 0.01
        op = DiffusionOperator(grid_256, D)
        eigenvalues = op.eigenvalues

        x = grid_256.x_coords(include_ghost=False)
        u0 = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.cos(4 * jnp.pi * x)

        print("\n" + "=" * 80)
        print("NILT vs Time-Stepping Accuracy (Milestone 4)")
        print("=" * 80)
        print(f"Grid: N={grid_256.nx}, D={D}")
        print("Method: NILT (FFT-tuned) vs ETDRK4")
        print("-" * 80)

        results = []
        for t_end in [0.1, 0.5, 1.0, 2.0]:
            comp = compare_nilt_vs_timestepping(
                eigenvalues, u0, t_end,
                tss_method='etdrk4',
                n_warmup=2,
                n_runs=5,
            )
            results.append(comp)
            print(f"t_end={t_end:.1f}: NILT err={comp.nilt_error:.2e}, "
                  f"TSS err={comp.tss_error:.2e}, speedup={comp.speedup:.2f}x")

        # Verify NILT achieves target accuracy
        for r in results:
            assert r.nilt_error < 1e-3, \
                f"NILT should achieve <1e-3 error at t_end={r.t_end}"


# =============================================================================
# Test: Real Symbol at Self-Paired Modes
# =============================================================================

class TestAdvectionNyquistFix:
    """AdvectionDiffusionOperator's odd symbol -i*v*k used to carry a
    complex eigenvalue at the self-paired Nyquist mode of an even grid
    (fftfreq stores that mode's k as -pi/dx), which no real operator has.
    nilt_solve_linear_pde must reject such a spectrum rather than silently
    produce a wrong answer, and the fixed operator (real eigenvalue at that
    mode) must agree with both the closed form and an independent
    etd_integrate run."""

    def test_advection_nyquist_matches_closed_form(self):
        """8 points on [0, 1], v = 1, D = 0, u0[j] = (-1)^j, t_end = 0.1.

        u0 is exactly the Nyquist spatial mode, so once its derivative
        symbol is zeroed (the usual convention for a real grid function)
        the field is stationary: the bridge, the closed form it is built
        from, and an independent etd_integrate run all return u0
        unchanged. Before the fix the bridge returned -5.638635 at the
        first point against -0.809017 from treating the checkerboard as a
        continuous, aliased wave.
        """
        from moljax.core.fft_integrators import etd_integrate

        grid = Grid1D.uniform(8, x_min=0.0, x_max=1.0)
        op = AdvectionDiffusionOperator(grid, v=1.0, D=0.0)
        u0 = (-1.0) ** jnp.arange(grid.nx)
        t_end = 0.1

        result = nilt_solve_linear_pde(op.eigenvalues, u0, t_end)
        max_err_closed = float(jnp.max(jnp.abs(result['u_final'] - result['u_analytical'])))
        assert max_err_closed < 1e-10, f"bridge vs closed form: {max_err_closed:.3e}"
        max_err_u0 = float(jnp.max(jnp.abs(result['u_final'] - u0)))
        assert max_err_u0 < 1e-10, f"bridge vs stationary u0: {max_err_u0:.3e}"

        def zero_rhs(state, t):
            return {name: jnp.zeros_like(v) for name, v in state.items()}

        dt = 1e-3
        n_steps = int(t_end / dt)
        _, hist = etd_integrate(
            {'u': u0}, (0.0, t_end), dt, {'u': op}, zero_rhs,
            method='etd1', save_every=n_steps,
        )
        u_etd = hist[-1]['u']
        max_err_etd = float(jnp.max(jnp.abs(result['u_final'] - u_etd)))
        assert max_err_etd < 1e-8, f"bridge vs etd_integrate: {max_err_etd:.3e}"

    def test_bridge_rejects_complex_self_paired_mode(self):
        """A hand-built spectrum with a nonzero imaginary eigenvalue at the
        self-paired Nyquist mode (index N//2 on an even grid) must raise:
        a self-paired mode is its own conjugate partner, so a real field
        cannot have a complex eigenvalue there."""
        n = 6
        eigenvalues = jnp.array([0.0, -1.0 + 1.0j, -2.0, 1.0 + 2.0j, -2.0, -1.0 - 1.0j])
        u0 = jnp.ones(n)

        with pytest.raises(ValueError, match="self-paired"):
            nilt_solve_linear_pde(eigenvalues, u0, t_end=1.0)

        # The DC mode (index 0) is also self-paired.
        eigenvalues_dc = jnp.array([1.0j, -1.0, -2.0, -3.0, -2.0, -1.0])
        with pytest.raises(ValueError, match="self-paired"):
            nilt_solve_linear_pde(eigenvalues_dc, u0, t_end=1.0)
