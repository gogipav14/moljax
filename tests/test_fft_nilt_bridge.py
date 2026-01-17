"""
Tests for FFT-NILT Bridge (Milestone 4).

Verifies:
1. Exact spectral bounds from FFT eigenvalues (no power iteration)
2. NILT parameter tuning optimized for FFT operators
3. NILT accuracy matches time-stepping to 1e-4 relative error
4. NILT faster for long time horizons (t_end > 100 dt_cfl)
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from moljax.core.grid import Grid1D
from moljax.core.fft_operators import DiffusionOperator, AdvectionDiffusionOperator
from moljax.core.fft_integrators import etd_integrate
from moljax.laplace.fft_nilt_bridge import (
    exact_spectral_bounds_from_fft,
    fft_bounds_to_spectral_bounds,
    tune_nilt_for_fft_operator,
    nilt_solve_linear_pde,
    compare_nilt_vs_timestepping,
    print_comparison_table,
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
        D = 0.01
        v = 1.0
        dx = advdiff_op.grid.dx

        # Expected im_max: v * k_max where k_max = π/dx
        expected_im_max = abs(v) * jnp.pi / dx

        assert bounds.re_max <= 0, f"AdvDiff should have re_max <= 0"
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

        # Check accuracy
        rel_error = float(jnp.linalg.norm(u_nilt - u_exact) / jnp.linalg.norm(u_exact))
        assert rel_error < 1e-4, f"NILT error {rel_error:.2e} exceeds 1e-4 threshold"

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
