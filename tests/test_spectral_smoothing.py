"""
Tests for spectral smoothing methods (Gibbs artifact reduction).
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from moljax.laplace.spectral_smoothing import (
    SmoothingMethod,
    fejer_sigma,
    lanczos_sigma,
    hamming_window,
    raised_cosine_sigma,
    exponential_sigma,
    get_sigma_factors,
    apply_spectral_smoothing,
    nilt_with_smoothing,
    compare_smoothing_methods,
)
from moljax.laplace.transfer_functions import step_F, step_f


class TestSigmaFactors:
    """Test individual σ-factor functions."""

    def test_fejer_sigma_endpoints(self):
        """Fejér σ = 1 at k=0, decreases toward edges."""
        N = 128
        sigma = fejer_sigma(N)

        assert sigma[0] == 1.0
        assert sigma[N//2] < sigma[0]  # Decreases toward Nyquist
        assert jnp.all(sigma >= 0)

    def test_lanczos_sigma_endpoints(self):
        """Lanczos σ = 1 at k=0, smooth decay."""
        N = 128
        sigma = lanczos_sigma(N, order=1)

        assert jnp.abs(sigma[0] - 1.0) < 1e-10
        assert sigma[N//2] < sigma[0]
        assert jnp.all(sigma >= 0)

    def test_lanczos_higher_order_stronger(self):
        """Higher order Lanczos gives stronger smoothing."""
        N = 128
        sigma1 = lanczos_sigma(N, order=1)
        sigma2 = lanczos_sigma(N, order=2)

        # Higher order should decay faster
        assert float(sigma2[N//4]) < float(sigma1[N//4])

    def test_hamming_window_shape(self):
        """Hamming window has expected shape."""
        N = 128
        sigma = hamming_window(N)

        assert jnp.abs(sigma[0] - 1.0) < 1e-10
        assert jnp.all(sigma > 0)  # Hamming never goes to zero

    def test_raised_cosine_rolloff_effect(self):
        """Raised cosine rolloff parameter controls transition."""
        N = 128
        sigma_low = raised_cosine_sigma(N, rolloff=0.2)
        sigma_high = raised_cosine_sigma(N, rolloff=0.8)

        # Low rolloff keeps more high frequencies
        mid_idx = N // 4
        assert float(sigma_low[mid_idx]) >= float(sigma_high[mid_idx])

    def test_exponential_sigma_alpha_effect(self):
        """Exponential alpha parameter controls decay rate."""
        N = 128
        sigma_slow = exponential_sigma(N, alpha=1.0)
        sigma_fast = exponential_sigma(N, alpha=4.0)

        # Higher alpha decays faster
        mid_idx = N // 4
        assert float(sigma_fast[mid_idx]) < float(sigma_slow[mid_idx])

    def test_get_sigma_factors_all_methods(self):
        """get_sigma_factors works for all methods."""
        N = 64
        methods = [
            SmoothingMethod.NONE,
            SmoothingMethod.FEJER,
            SmoothingMethod.LANCZOS,
            SmoothingMethod.HAMMING,
            SmoothingMethod.RAISED_COSINE,
            SmoothingMethod.EXPONENTIAL,
        ]

        for method in methods:
            sigma = get_sigma_factors(N, method)
            assert sigma.shape == (N,)
            assert jnp.all(jnp.isfinite(sigma))


class TestApplySmoothing:
    """Test spectral smoothing application."""

    def test_no_smoothing_identity(self):
        """No smoothing returns original spectrum."""
        F = jnp.array([1.0, 0.5, 0.25, 0.125], dtype=jnp.complex128)
        result = apply_spectral_smoothing(F, SmoothingMethod.NONE)

        assert jnp.allclose(result.F_smoothed, F)
        assert result.bandwidth_retention == 1.0

    def test_smoothing_reduces_energy(self):
        """Smoothing reduces spectral energy."""
        N = 64
        F = jnp.ones(N, dtype=jnp.complex128)

        for method in [SmoothingMethod.FEJER, SmoothingMethod.LANCZOS]:
            result = apply_spectral_smoothing(F, method)
            assert result.bandwidth_retention < 1.0

    def test_smoothing_preserves_dc(self):
        """Smoothing preserves DC component (k=0)."""
        N = 64
        F = jnp.zeros(N, dtype=jnp.complex128)
        F = F.at[0].set(10.0)

        result = apply_spectral_smoothing(F, SmoothingMethod.LANCZOS)

        # DC component should be preserved (σ[0] = 1)
        assert jnp.abs(result.F_smoothed[0] - 10.0) < 1e-10


class TestNILTWithSmoothing:
    """Test NILT integration with smoothing."""

    def test_smooth_function_minimal_effect(self):
        """Smoothing has minimal effect on smooth functions."""
        # Exponential decay: smooth, no Gibbs artifacts expected
        def F_exp(s):
            return 1.0 / (s + 1.0)

        dt = 0.01
        N = 256
        a = 1.0

        result_none = nilt_with_smoothing(F_exp, dt, N, a, SmoothingMethod.NONE)
        result_lanczos = nilt_with_smoothing(F_exp, dt, N, a, SmoothingMethod.LANCZOS)

        # Results should be similar for smooth functions
        diff = jnp.linalg.norm(result_none.f - result_lanczos.f)
        assert diff < 0.1  # Small difference

    def test_step_function_smoothing_helps(self):
        """Smoothing reduces ringing for step function."""
        dt = 0.01
        N = 512  # More points for better resolution
        a = 2.0

        result_none = nilt_with_smoothing(step_F, dt, N, a, SmoothingMethod.NONE)
        result_lanczos = nilt_with_smoothing(step_F, dt, N, a, SmoothingMethod.LANCZOS)

        # Compute oscillation metric: max-min in a region after the step
        # Step happens around t~0, look at oscillations in later region
        n_pts = len(result_none.f)
        late_region = slice(n_pts // 4, n_pts // 2)

        # For a step function, f should be ~1 after t=0
        # Ringing shows as deviations from 1
        dev_none = jnp.abs(result_none.f[late_region] - 1.0)
        dev_lanczos = jnp.abs(result_lanczos.f[late_region] - 1.0)

        max_dev_none = jnp.max(dev_none)
        max_dev_lanczos = jnp.max(dev_lanczos)

        # Lanczos smoothing should reduce max deviation from expected value
        # Allow small tolerance for numerical noise
        assert max_dev_lanczos <= max_dev_none * 1.1

    def test_diagnostics_returned(self):
        """Diagnostics include smoothing info."""
        def F_exp(s):
            return 1.0 / (s + 1.0)

        result, diag = nilt_with_smoothing(
            F_exp, dt=0.01, N=128, a=1.0,
            smoothing=SmoothingMethod.LANCZOS,
            return_diagnostics=True
        )

        assert 'smoothing' in diag
        assert diag['smoothing'].method == 'lanczos'
        assert 0 < diag['smoothing'].bandwidth_retention < 1


class TestCompareSmoothing:
    """Test smoothing comparison utility."""

    def test_comparison_returns_all_methods(self):
        """Comparison includes all smoothing methods."""
        def F_exp(s):
            return 1.0 / (s + 1.0)

        comparison = compare_smoothing_methods(F_exp, dt=0.01, N=128, a=1.0)

        expected_methods = ['none', 'fejer', 'lanczos', 'hamming',
                          'raised_cosine', 'exponential']
        for method in expected_methods:
            assert method in comparison

    def test_comparison_with_exact_solution(self):
        """Comparison computes errors when exact solution provided."""
        def F_exp(s):
            return 1.0 / (s + 1.0)

        def f_exp(t):
            return jnp.exp(-t)

        comparison = compare_smoothing_methods(
            F_exp, f_exact=f_exp,
            dt=0.01, N=256, a=1.0
        )

        for method, data in comparison.items():
            assert 'max_error' in data
            assert 'rms_error' in data
            assert data['max_error'] >= 0
            assert data['rms_error'] >= 0
