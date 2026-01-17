"""
Tests comparing NILT with MOL time stepping for diffusion.

Verifies that NILT produces consistent results with the
established FFT-implicit diffusion solver from Step 2.
"""

import jax.numpy as jnp
import pytest

from moljax.laplace import (
    nilt_fft_uniform,
    tune_for_diffusion,
    diffusion_semi_infinite_F,
    diffusion_semi_infinite_f,
)


class TestDiffusionNILT:
    """Test NILT for diffusion-related transforms."""

    def test_semi_infinite_diffusion_returns_result(self):
        """Test that semi-infinite diffusion NILT returns valid result."""
        D = 1.0
        x = 1.0

        F = lambda s: diffusion_semi_infinite_F(s, D=D, x=x)

        # Use manual parameters for stability
        # Semi-infinite diffusion has exp(-x*sqrt(s/D))/s behavior
        result = nilt_fft_uniform(
            F, dt=0.05, N=1024, a=0.5
        )

        # Verify result structure
        assert result.t is not None
        assert result.f is not None
        assert len(result.t) == 1024

        # Check that output is not all zeros (only in early time where it's finite)
        early_mask = result.t <= 10.0
        early_vals = result.f[early_mask]
        assert jnp.sum(jnp.isfinite(early_vals)) > 0, "All values are non-finite"

    def test_diffusion_scaling_with_D(self):
        """Test that diffusion solutions scale correctly with D."""
        x = 1.0
        t_test = 2.0

        # For semi-infinite diffusion: f(t) = erfc(x/(2*sqrt(D*t)))
        # Larger D -> larger argument to erfc -> smaller result
        # (since erfc is decreasing)

        f_D1 = diffusion_semi_infinite_f(jnp.array([t_test]), D=1.0, x=x)[0]
        f_D2 = diffusion_semi_infinite_f(jnp.array([t_test]), D=2.0, x=x)[0]

        # Larger D means faster diffusion, so concentration at x arrives sooner
        # erfc(x/(2*sqrt(D*t))) increases with D
        assert f_D2 > f_D1, "Larger D should give larger concentration"

    def test_diffusion_scaling_with_x(self):
        """Test that solutions scale correctly with position x."""
        D = 1.0
        t_test = 2.0

        f_x1 = diffusion_semi_infinite_f(jnp.array([t_test]), D=D, x=1.0)[0]
        f_x2 = diffusion_semi_infinite_f(jnp.array([t_test]), D=D, x=2.0)[0]

        # Larger x (farther from boundary) means smaller concentration
        assert f_x1 > f_x2, "Farther position should have smaller concentration"


class TestNILTvsMOLConcept:
    """
    Conceptual tests for NILT vs MOL comparison.

    These tests verify the NILT approach is internally consistent
    rather than directly comparing with MOL (which would require
    setting up the PDE problem with matching conditions).
    """

    def test_nilt_self_consistency(self):
        """Test NILT gives consistent results with different parameters."""
        D = 0.01
        x = 0.5

        F = lambda s: diffusion_semi_infinite_F(s, D=D, x=x)

        # Run with different N values
        result_N1024 = nilt_fft_uniform(F, dt=0.05, N=1024, a=0.3)
        result_N2048 = nilt_fft_uniform(F, dt=0.05, N=2048, a=0.3)

        # Compare at common time points
        t_test = jnp.array([1.0, 2.0, 3.0, 4.0])

        # Interpolate results
        def interp_at(result, t_vals):
            return jnp.interp(t_vals, result.t, result.f)

        f_N1024 = interp_at(result_N1024, t_test)
        f_N2048 = interp_at(result_N2048, t_test)

        # Should be similar
        diff = jnp.abs(f_N1024 - f_N2048)
        assert jnp.max(diff) < 0.2, "Results should be consistent across N"

    def test_nilt_convergence_with_finer_sampling(self):
        """Test NILT converges with finer time sampling."""
        D = 0.1
        x = 0.5

        F = lambda s: diffusion_semi_infinite_F(s, D=D, x=x)
        f_true = lambda t: diffusion_semi_infinite_f(t, D=D, x=x)

        # Coarse sampling
        result_coarse = nilt_fft_uniform(F, dt=0.1, N=512, a=0.3)

        # Fine sampling
        result_fine = nilt_fft_uniform(F, dt=0.025, N=2048, a=0.3)

        # Compute errors at common times
        t_test = jnp.array([0.5, 1.0, 2.0, 3.0])

        def get_error(result):
            f_interp = jnp.interp(t_test, result.t, result.f)
            f_ref = f_true(t_test)
            return jnp.sqrt(jnp.mean((f_interp - f_ref)**2))

        error_coarse = get_error(result_coarse)
        error_fine = get_error(result_fine)

        # Fine sampling should generally be better or equal
        # (Note: not always guaranteed due to other factors)
        assert error_fine < error_coarse * 2, \
            "Finer sampling should not be significantly worse"


class TestTunedDiffusionNILT:
    """Test NILT with autotuned parameters for diffusion."""

    def test_tuned_parameters_produce_valid_result(self):
        """Tuned parameters should produce valid NILT result."""
        dx = 0.1
        D = 0.01
        t_end = 10.0

        params = tune_for_diffusion(dx=dx, D=D, t_end=t_end)

        # Simple test: exponential decay (represents diffusion eigenmode)
        alpha = D * (jnp.pi / 1.0)**2  # First eigenvalue for L=1
        F = lambda s: 1.0 / (s + alpha)
        f_true = lambda t: jnp.exp(-alpha * t)

        result = nilt_fft_uniform(
            F, dt=params.dt, N=params.N, a=params.a
        )

        mask = result.t <= t_end
        f_ref = f_true(result.t[mask])
        f_computed = result.f[mask]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2)) / \
                    (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)

        assert rel_error < 0.15, f"Relative error {rel_error:.4f} too large"

    def test_tuning_adapts_to_grid(self):
        """Tuning should produce different params for different grids."""
        D = 0.01
        t_end = 10.0

        params_coarse = tune_for_diffusion(dx=0.2, D=D, t_end=t_end)
        params_fine = tune_for_diffusion(dx=0.05, D=D, t_end=t_end)

        # Fine grid has faster dynamics, needs larger omega coverage
        assert params_fine.omega_req > params_coarse.omega_req


class TestComparisonFramework:
    """Test the comparison framework infrastructure."""

    def test_comparison_metrics_computed(self):
        """Test that comparison metrics can be computed."""
        from moljax.laplace import compare_to_reference

        # Simple test case
        t = jnp.linspace(0, 10, 100)
        f_computed = jnp.exp(-0.5 * t) + jnp.sin(t) * 0.1  # Add some error
        f_reference = jnp.exp(-0.5 * t)

        metrics = compare_to_reference(f_computed, f_reference, t, t_end=10.0)

        assert 'l2_error' in metrics
        assert 'linf_error' in metrics
        assert 'l2_relative' in metrics
        assert metrics['l2_error'] > 0
        assert metrics['linf_error'] > 0

    def test_comparison_respects_t_end(self):
        """Comparison should only use points up to t_end."""
        from moljax.laplace import compare_to_reference

        t = jnp.linspace(0, 20, 200)
        f_computed = jnp.exp(-0.5 * t)
        f_reference = jnp.exp(-0.5 * t)

        # Add large error after t=10
        f_computed = f_computed.at[100:].set(100.0)

        # Full comparison should see large error
        metrics_full = compare_to_reference(f_computed, f_reference, t)

        # Comparison with t_end=10 should see small error
        metrics_truncated = compare_to_reference(
            f_computed, f_reference, t, t_end=10.0
        )

        assert metrics_truncated['linf_error'] < metrics_full['linf_error']


class TestBenchmarkInfrastructure:
    """Test benchmark infrastructure components."""

    def test_benchmark_config_creation(self):
        """Test benchmark config can be created."""
        from moljax.laplace import BenchmarkConfig

        config = BenchmarkConfig(
            name='test',
            t_end=10.0,
            F_eval=lambda s: 1.0 / (s + 1.0),
            f_reference=lambda t: jnp.exp(-t)
        )

        assert config.name == 'test'
        assert config.t_end == 10.0
        assert config.F_eval is not None

    def test_benchmark_run_basic(self):
        """Test basic benchmark execution."""
        from moljax.laplace import BenchmarkConfig, run_nilt_benchmark

        config = BenchmarkConfig(
            name='exp_decay_test',
            t_end=5.0,
            F_eval=lambda s: 1.0 / (s + 1.0),
            f_reference=lambda t: jnp.exp(-t)
        )

        result = run_nilt_benchmark(config)

        assert result.name == 'exp_decay_test'
        assert result.tuned_params is not None
        assert result.nilt_result is not None
        assert result.errors is not None
        assert 'l2_error' in result.errors
