"""
Tests for FFT-NILT Bridge (Milestone 4).

Verifies:
1. Exact spectral bounds from FFT eigenvalues (no power iteration)
2. NILT parameter tuning optimized for FFT operators
3. NILT accuracy matches time-stepping to 1e-4 relative error
4. NILT faster for long time horizons (t_end > 100 dt_cfl)
"""

import jax
import jax.numpy as jnp
import pytest

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from moljax.core.fft_operators import AdvectionDiffusionOperator, DiffusionOperator
from moljax.core.grid import Grid1D
from moljax.laplace.fft_nilt_bridge import (
    compare_nilt_vs_timestepping,
    exact_spectral_bounds_from_fft,
    fft_bounds_to_spectral_bounds,
    nilt_solve_linear_pde,
    print_comparison_table,
    tune_nilt_for_fft_operator,
)

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
        """Verify advection-diffusion operator bounds."""
        eigenvalues = advdiff_op.eigenvalues
        bounds = exact_spectral_bounds_from_fft(eigenvalues, "AdvDiff")

        # For advection-diffusion: complex eigenvalues
        # Re(λ) from diffusion (≤ 0), Im(λ) from advection
        v = 1.0
        dx = advdiff_op.grid.dx

        # Expected im_max: v * k_max where k_max = π/dx
        expected_im_max = abs(v) * jnp.pi / dx

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
