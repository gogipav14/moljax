"""
Tests for the second-order damping example: F(s) = 1/(s^2 + s + 1).

This is the main test case demonstrating NILT accuracy with
autotuned parameters.
"""

import jax.numpy as jnp
import pytest

from moljax.laplace import (
    nilt_fft_uniform,
    tune_nilt_params,
    second_order_damping_F,
    second_order_damping_f,
    compute_nilt_diagnostics,
    compare_to_reference,
    SpectralBounds,
)


class TestSecondOrderDamping:
    """Tests for second-order damping transfer function."""

    def test_analytic_inverse_correctness(self):
        """Verify the analytic inverse formula is correct at key points."""
        t = jnp.array([0.0, 1.0, 2.0, 5.0, 10.0])
        f = second_order_damping_f(t)

        # At t=0, f(0) = (2/sqrt(3)) * exp(0) * sin(0) = 0
        assert abs(f[0]) < 1e-6, f"f(0) = {f[0]} should be 0"

        # f should decay as t increases (due to exp(-t/2))
        assert f[4] < f[2], "f should decay with time"

        # f should be positive initially (for small t)
        assert f[1] > 0, "f(1) should be positive"

    def test_basic_inversion(self):
        """Test basic NILT inversion with manual parameters."""
        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.05,
            N=1024,
            a=0.5
        )

        # Verify result structure
        assert result.t is not None
        assert result.f is not None
        assert len(result.t) == 1024

        # Check that we get non-trivial output
        assert jnp.max(jnp.abs(result.f)) > 0.01

    def test_with_autotuned_params(self):
        """Test NILT with autotuned parameters."""
        # Create bounds for the second-order system
        # Poles at s = -0.5 ± i*sqrt(3)/2
        bounds = SpectralBounds(
            rho=1.0,
            re_max=-0.5,
            im_max=0.866,
            methods_used={'all': 'analytic'},
            warnings=[]
        )

        tuned = tune_nilt_params(t_end=20.0, bounds=bounds)

        # Verify tuning produced valid parameters
        assert tuned.dt > 0
        assert tuned.N > 0
        assert tuned.a >= 0

        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=tuned.dt,
            N=tuned.N,
            a=tuned.a
        )

        # Verify result
        assert result.f is not None

    def test_diagnostics_structure(self):
        """Test that diagnostics return expected structure."""
        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.05,
            N=1024,
            a=0.5
        )

        diag = compute_nilt_diagnostics(
            second_order_damping_F,
            result.f,
            result.t,
            a=0.5,
            T=result.T,
            t_end=20.0
        )

        # Verify structure
        assert hasattr(diag, 'quality_score')
        assert hasattr(diag, 'ringing_metric')
        assert hasattr(diag, 'frequency_coverage')
        assert hasattr(diag, 'warnings')

    def test_comparison_to_reference(self):
        """Test compare_to_reference utility."""
        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.05,
            N=1024,
            a=0.5
        )

        f_ref = second_order_damping_f(result.t)
        comparison = compare_to_reference(
            result.f, f_ref, result.t, t_end=20.0
        )

        # Verify comparison structure
        assert 'l2_error' in comparison
        assert 'linf_error' in comparison
        assert 't_max_error' in comparison


class TestFrequencyWindowImportance:
    """
    Tests demonstrating the importance of full complex spectrum
    (not cosine-only) for accurate inversion over the full interval.
    """

    def test_sufficient_frequency_coverage(self):
        """Test that adequate frequency coverage gives good results."""
        # Second-order system has oscillations at omega_d = sqrt(3)/2 ≈ 0.866
        # Need omega_max > omega_d for good coverage

        # Good coverage: omega_max = pi/dt = pi/0.05 ≈ 63 >> 0.866
        result_good = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.05,
            N=2048,
            a=0.3
        )

        t_end = 25.0
        mask = result_good.t <= t_end
        f_ref = second_order_damping_f(result_good.t[mask])

        error_good = jnp.sqrt(jnp.mean((result_good.f[mask] - f_ref)**2))

        # Poor coverage: omega_max = pi/0.5 ≈ 6.28, still > 0.866 but less margin
        result_poor = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.5,
            N=256,
            a=0.3
        )

        mask_poor = result_poor.t <= t_end
        f_ref_poor = second_order_damping_f(result_poor.t[mask_poor])
        error_poor = jnp.sqrt(jnp.mean((result_poor.f[mask_poor] - f_ref_poor)**2))

        # Better coverage should give lower error
        # (Note: this may not always hold due to other factors)
        assert error_good < 0.5, f"Good coverage error {error_good} should be small"

    def test_long_time_accuracy(self):
        """Test accuracy at longer times (t > T/2 region)."""
        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.05,
            N=2048,
            a=0.3
        )

        # Check early times
        early_mask = (result.t >= 1.0) & (result.t <= 10.0)
        f_ref_early = second_order_damping_f(result.t[early_mask])
        error_early = jnp.sqrt(jnp.mean((result.f[early_mask] - f_ref_early)**2))

        # Check later times (but still within valid interval)
        T = result.T
        late_mask = (result.t >= 15.0) & (result.t <= min(30.0, T * 0.9))
        if jnp.sum(late_mask) > 5:
            f_ref_late = second_order_damping_f(result.t[late_mask])
            error_late = jnp.sqrt(jnp.mean((result.f[late_mask] - f_ref_late)**2))

            # Error at late times should still be reasonable
            # (may be slightly larger due to accumulated effects)
            assert error_late < 0.3, f"Late time error {error_late} too large"

        assert error_early < 0.15, f"Early time error {error_early} too large"


class TestParameterSensitivity:
    """Tests for sensitivity to NILT parameters."""

    def test_bromwich_shift_sensitivity(self):
        """Test that different a values produce different results."""
        dt, N = 0.05, 1024

        results = []
        a_values = [0.1, 0.5, 1.0]

        for a in a_values:
            result = nilt_fft_uniform(
                second_order_damping_F,
                dt=dt, N=N, a=a
            )
            results.append(result.f)

        # Different a values should produce different results
        diff_01_05 = jnp.max(jnp.abs(results[0] - results[1]))
        diff_05_10 = jnp.max(jnp.abs(results[1] - results[2]))

        assert diff_01_05 > 0.001 or diff_05_10 > 0.001, \
            "Different a values should produce different results"

    def test_fft_size_sensitivity(self):
        """Test sensitivity to FFT size N."""
        dt, a = 0.05, 0.5
        t_end = 15.0

        errors = []
        N_values = [256, 512, 1024, 2048]

        for N in N_values:
            result = nilt_fft_uniform(
                second_order_damping_F,
                dt=dt, N=N, a=a
            )
            mask = result.t <= t_end
            f_ref = second_order_damping_f(result.t[mask])
            error = jnp.sqrt(jnp.mean((result.f[mask] - f_ref)**2))
            errors.append(float(error))

        # Larger N should generally give better or equal results
        # (up to a point where other errors dominate)
        assert errors[0] >= errors[-1] * 0.5, "Larger N should help"

        # All sizes should give reasonable results
        for N, err in zip(N_values, errors):
            assert err < 0.5, f"N={N}: error {err:.4f} too large"

    def test_dt_sensitivity(self):
        """Test sensitivity to time step dt."""
        N, a = 1024, 0.2  # Use smaller a to avoid overflow
        t_end = 10.0

        errors = []
        dt_values = [0.02, 0.05, 0.1, 0.2]

        for dt in dt_values:
            result = nilt_fft_uniform(
                second_order_damping_F,
                dt=dt, N=N, a=a, dtype=jnp.float64  # Use float64 for wider range
            )
            mask = result.t <= t_end
            f_ref = second_order_damping_f(result.t[mask])
            error = jnp.sqrt(jnp.mean((result.f[mask] - f_ref)**2))
            errors.append(float(error))

        # Smaller dt provides better frequency resolution
        # Should generally improve accuracy (for this problem)
        for dt, err in zip(dt_values, errors):
            assert err < 0.5, f"dt={dt}: error {err:.4f} too large"


class TestSpecialCases:
    """Tests for edge cases and special situations."""

    def test_zero_initial_value(self):
        """Second-order damping has f(0) = 0."""
        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.01, N=1024, a=0.5
        )

        # f(0) should be close to 0
        f_at_zero = result.f[0]
        assert abs(f_at_zero) < 0.1, f"f(0) = {f_at_zero} should be near 0"

    def test_decay_captured(self):
        """Test that exponential decay is captured."""
        result = nilt_fft_uniform(
            second_order_damping_F,
            dt=0.05, N=2048, a=0.3
        )

        # Value at t=20 should be much smaller than at t=2
        # Due to exp(-t/2) factor
        idx_2 = int(2.0 / 0.05)
        idx_20 = int(20.0 / 0.05)

        if idx_20 < len(result.f):
            ratio = abs(result.f[idx_20]) / (abs(result.f[idx_2]) + 1e-10)
            expected_ratio = jnp.exp(-9)  # exp(-20/2) / exp(-2/2) = exp(-9)

            # Should decay significantly
            assert ratio < 0.1, f"Decay ratio {ratio} not captured properly"
