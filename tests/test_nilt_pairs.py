"""
Tests for NILT inversion of standard Laplace transform pairs.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from moljax.laplace import (
    cosine_F,
    cosine_f,
    damped_cosine_F,
    damped_cosine_f,
    damped_sine_F,
    damped_sine_f,
    estimate_nilt_truncation_error,
    exponential_decay_F,
    exponential_decay_f,
    get_standard_laplace_pairs,
    invert_laplace,
    nilt_fft_halfstep,
    nilt_fft_uniform,
    nilt_fft_with_pole_at_origin,
    nilt_with_smoothing,
    sine_F,
    sine_f,
    step_F,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestExponentialDecay:
    """Test NILT for exponential decay functions.

    Note: NILT accuracy depends on careful parameter tuning (a, dt, N, T).
    These tests verify qualitative correctness with relaxed tolerances.
    """

    def test_exp_decay_returns_valid_result(self):
        """Test that NILT returns a valid result structure."""
        def F(s):
            return exponential_decay_F(s, alpha=1.0)
        result = nilt_fft_uniform(F, dt=0.05, N=1024, a=0.5)

        assert result.t is not None
        assert result.f is not None
        assert len(result.t) == len(result.f) == 1024
        assert result.dt == 0.05
        assert result.N == 1024

    def test_exp_decay_qualitative_behavior(self):
        """Test that decay behavior is captured qualitatively."""
        def F(s):
            return exponential_decay_F(s, alpha=1.0)
        result = nilt_fft_uniform(F, dt=0.05, N=1024, a=0.5)

        # Early values should be larger than late values (decay)
        early_mean = jnp.mean(jnp.abs(result.f[10:50]))
        late_mean = jnp.mean(jnp.abs(result.f[100:200]))

        assert early_mean > late_mean, "Decay behavior not captured"

    def test_exp_decay_different_alpha(self):
        """Test that different decay rates produce different results."""
        def F_slow(s):
            return exponential_decay_F(s, alpha=0.5)
        def F_fast(s):
            return exponential_decay_F(s, alpha=2.0)

        result_slow = nilt_fft_uniform(F_slow, dt=0.05, N=1024, a=0.3)
        result_fast = nilt_fft_uniform(F_fast, dt=0.05, N=1024, a=0.3)

        # Fast decay should have smaller values at later times
        late_slow = jnp.mean(jnp.abs(result_slow.f[100:200]))
        late_fast = jnp.mean(jnp.abs(result_fast.f[100:200]))

        assert late_slow > late_fast, "Different decay rates not distinguished"


class TestOscillatory:
    """Test NILT for oscillatory functions."""

    def test_cosine_omega1(self):
        """Test L^{-1}[s/(s^2+1)] = cos(t)."""
        omega = 1.0
        def F(s):
            return cosine_F(s, omega=omega)
        def f_true(t):
            return cosine_f(t, omega=omega)

        # Oscillatory functions need good frequency resolution
        result = nilt_fft_uniform(F, dt=0.05, N=2048, a=0.2)

        t_end = 20.0
        mask = result.t <= t_end
        f_ref = f_true(result.t[mask])
        f_computed = result.f[mask]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2)) / (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)
        assert rel_error < 0.1, f"Cosine relative error {rel_error:.4f} too large"

    def test_sine_omega1(self):
        """Test L^{-1}[1/(s^2+1)] = sin(t)."""
        omega = 1.0
        def F(s):
            return sine_F(s, omega=omega)
        def f_true(t):
            return sine_f(t, omega=omega)

        result = nilt_fft_uniform(F, dt=0.05, N=2048, a=0.2)

        t_end = 20.0
        mask = result.t <= t_end
        f_ref = f_true(result.t[mask])
        f_computed = result.f[mask]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2)) / (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)
        assert rel_error < 0.1, f"Sine relative error {rel_error:.4f} too large"

    def test_damped_sine(self):
        """Test damped sinusoidal: exp(-alpha*t)*sin(omega*t)."""
        alpha = 0.5
        omega = 2.0
        def F(s):
            return damped_sine_F(s, alpha=alpha, omega=omega)
        def f_true(t):
            return damped_sine_f(t, alpha=alpha, omega=omega)

        result = nilt_fft_uniform(F, dt=0.02, N=2048, a=0.3)

        t_end = 10.0
        mask = result.t <= t_end
        f_ref = f_true(result.t[mask])
        f_computed = result.f[mask]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2)) / (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)
        assert rel_error < 0.1, f"Damped sine relative error {rel_error:.4f} too large"

    def test_damped_cosine(self):
        """Test damped cosinusoidal: exp(-alpha*t)*cos(omega*t)."""
        alpha = 0.3
        omega = 1.5
        def F(s):
            return damped_cosine_F(s, alpha=alpha, omega=omega)
        def f_true(t):
            return damped_cosine_f(t, alpha=alpha, omega=omega)

        result = nilt_fft_uniform(F, dt=0.03, N=2048, a=0.2)

        t_end = 12.0
        mask = result.t <= t_end
        f_ref = f_true(result.t[mask])
        f_computed = result.f[mask]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2)) / (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)
        assert rel_error < 0.1, f"Damped cosine relative error {rel_error:.4f} too large"


class TestPoleAtOrigin:
    """Test convolution handling for poles at s=0.

    Note: The pole-at-origin method multiplies F(s) by s to remove the pole,
    inverts, then integrates. This is a structural test - numerical accuracy
    depends on careful parameter tuning.
    """

    def test_pole_at_origin_returns_valid_result(self):
        """Test that pole-at-origin method returns valid structure."""
        result = nilt_fft_with_pole_at_origin(
            step_F, dt=0.05, N=1024, a=0.5
        )

        # Verify result structure
        assert result.t is not None
        assert result.f is not None
        assert len(result.t) == 1024
        assert result.dt == 0.05
        assert result.N == 1024

    def test_pole_at_origin_integrates(self):
        """Test that convolution method performs integration."""
        # For F(s) = 1/(s*(s+1)), we have G(s) = s*F(s) = 1/(s+1)
        # g(t) = exp(-t), and f(t) = integral of exp(-t) = 1 - exp(-t)

        def F(s):
            return 1.0 / (s * (s + 1.0))

        result = nilt_fft_with_pole_at_origin(F, dt=0.05, N=1024, a=0.5)

        # The result should start near 0 and grow toward 1
        assert result.f[0] < 0.5, "Should start near zero"
        # Later values should be larger (integration effect)
        assert jnp.mean(result.f[100:200]) > jnp.mean(result.f[0:10]), \
            "Integration should accumulate value"

    def test_pole_at_origin_requires_positive_shift(self):
        """G(s) = s F(s) is evaluated at s = a + i omega; at a = 0 the k = 0 sample is 0 * inf."""
        def F(s):
            return 1.0 / (s * (s + 1.0))

        with pytest.raises(ValueError, match="positive"):
            nilt_fft_with_pole_at_origin(F, dt=0.05, N=1024, a=0.0, dtype=jnp.float64)

        result = nilt_fft_with_pole_at_origin(F, dt=0.05, N=1024, a=0.5, dtype=jnp.float64)
        mask = result.t <= 10.0
        exact = 1.0 - jnp.exp(-result.t[mask])
        assert bool(jnp.all(jnp.isfinite(result.f)))
        # Measured 3.0e-3 (trapezoidal integration of the inverted e^{-t})
        assert float(jnp.max(jnp.abs(result.f[mask] - exact))) < 5e-3


class TestHighLevelInterface:
    """Test the high-level invert_laplace interface."""

    def test_auto_parameters(self):
        """Test automatic parameter selection."""
        def F(s):
            return exponential_decay_F(s, alpha=1.0)

        result = invert_laplace(F, t_end=10.0)

        # Check result structure
        assert result.t is not None
        assert result.f is not None
        assert len(result.t) == len(result.f)

        # Check that output covers requested interval
        assert result.t[-1] >= 10.0 or result.T >= 10.0

    def test_long_horizon_defaults_are_feasible(self):
        """The defaults come from tune_nilt_params, so a t_max stays inside the float32 budget at t_end = 50.

        The old fixed default a = 0.5 with 2T = 4 t_end gave a t_max = 100,
        past the float32 overflow guard, and invert_laplace(t_end=50) raised.
        """
        def F(s):
            return exponential_decay_F(s, alpha=1.0)

        with pytest.warns(UserWarning):
            result = invert_laplace(F, t_end=50.0)

        assert bool(jnp.all(jnp.isfinite(result.f)))
        assert result.t[-1] >= 50.0
        assert result.a > 0.0
        mask = (result.t > 0) & (result.t <= 5.0)
        assert float(jnp.max(jnp.abs(result.f[mask] - jnp.exp(-result.t[mask])))) < 0.1

    def test_with_pole_at_origin_flag(self):
        """Test has_pole_at_origin flag routes to convolution method."""
        # Use F(s) = 1/(s*(s+1)) which has a simple pole at origin
        def F(s):
            return 1.0 / (s * (s + 1.0))

        result = invert_laplace(
            F, t_end=10.0, has_pole_at_origin=True
        )

        # Verify result structure
        assert result.t is not None
        assert result.f is not None
        assert len(result.t) == len(result.f)


class TestOverflowGuard:
    """exp(a t) must never overflow silently."""

    def test_overflow_guard_without_x64(self):
        """With x64 off a float64 request yields a float32 grid; the guard judges that dtype.

        a t_max = 127.5 is inside the float64 budget (699.8) but past the
        float32 one (78.7). Before, the guard read the declared dtype and let
        the float32 computation return inf.
        """
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "import jax.numpy as jnp\n"
            "from moljax.laplace import nilt_fft_uniform\n"
            "try:\n"
            "    r = nilt_fft_uniform(lambda s: 1 / (s + 1), dt=0.1, N=256, a=5.0, dtype=jnp.float64)\n"
            "except ValueError as e:\n"
            "    print('RAISED', str(e).splitlines()[0])\n"
            "else:\n"
            "    print('FINITE', bool(jnp.all(jnp.isfinite(r.f))), str(r.f.dtype))\n"
        )
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=ROOT)
        out = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=ROOT, check=True
        )
        assert out.stdout.startswith('RAISED') or 'FINITE True' in out.stdout, out.stdout + out.stderr
        if out.stdout.startswith('RAISED'):
            assert 'float32' in out.stdout, out.stdout

    def test_halfstep_and_smoothing_have_the_guard(self):
        """a t_max = 766 exceeds the float64 budget; the half-step and smoothing variants must refuse, not return inf."""
        def F(s):
            return exponential_decay_F(s, alpha=1.0)

        with pytest.raises(ValueError, match="overflow"):
            nilt_fft_halfstep(F, dt=0.1, N=256, a=30.0, dtype=jnp.float64)
        with pytest.raises(ValueError, match="overflow"):
            nilt_with_smoothing(F, 0.1, 256, 30.0)


class TestTruncationErrorEstimate:
    def test_estimate_honors_dtype(self):
        """The cutoff sample is evaluated in the complex type matching dtype, not always complex64."""
        def F(s):
            return exponential_decay_F(s, alpha=1.0)

        exact = 1.0 / (1.0 + 100.0 ** 2) ** 0.5
        est64 = estimate_nilt_truncation_error(F, 0.0, 100.0, dtype=jnp.float64)
        assert abs(est64 - exact) < 1e-13, est64 - exact
        est32 = estimate_nilt_truncation_error(F, 0.0, 100.0, dtype=jnp.float32)
        assert abs(est32 - exact) < 1e-7


class TestStandardPairsSuite:
    """Run through all standard Laplace pairs."""

    def test_pairs_without_origin_pole(self):
        """Test all pairs that don't have poles at origin."""
        pairs = get_standard_laplace_pairs()

        for pair in pairs:
            if pair.has_pole_at_origin:
                continue

            result = nilt_fft_uniform(pair.F, dt=0.05, N=1024, a=0.5)

            t_end = min(10.0, result.T * 0.8)
            mask = (result.t >= 0.1) & (result.t <= t_end)

            if jnp.sum(mask) < 10:
                continue

            f_ref = pair.f(result.t[mask])
            f_computed = result.f[mask]

            # Check for reasonable accuracy
            ref_rms = jnp.sqrt(jnp.mean(f_ref**2))
            if ref_rms < 1e-6:
                continue  # Skip near-zero signals

            abs_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2))
            rel_error = abs_error / ref_rms

            assert rel_error < 0.5, f"Pair {pair.name}: relative error {rel_error:.4f} too large"


class TestNILTProperties:
    """Test fundamental NILT properties."""

    def test_linearity(self):
        """Test that NILT is linear: L^{-1}[aF + bG] = aL^{-1}[F] + bL^{-1}[G]."""
        a, b = 2.0, 0.5
        def F1(s):
            return exponential_decay_F(s, alpha=1.0)
        def F2(s):
            return exponential_decay_F(s, alpha=2.0)
        def F_combined(s):
            return a * F1(s) + b * F2(s)

        # Compute individual inversions
        result1 = nilt_fft_uniform(F1, dt=0.05, N=1024, a=0.5)
        result2 = nilt_fft_uniform(F2, dt=0.05, N=1024, a=0.5)
        result_combined = nilt_fft_uniform(F_combined, dt=0.05, N=1024, a=0.5)

        # Linear combination of individual results
        f_linear = a * result1.f + b * result2.f

        # Compare with direct inversion of combined function
        # Focus on early times where signal is strong (not decayed to noise)
        t_end = 15.0
        mask = result_combined.t <= t_end
        diff = jnp.abs(result_combined.f[mask] - f_linear[mask])

        # Use normalized error relative to signal RMS
        signal_rms = jnp.sqrt(jnp.mean(f_linear[mask]**2))
        normalized_error = jnp.sqrt(jnp.mean(diff**2)) / (signal_rms + 1e-10)

        # The FFT-based NILT should be linear up to floating-point precision
        assert normalized_error < 0.01, f"Linearity violated: normalized error = {normalized_error}"

    def test_time_shift(self):
        """Test that L^{-1}[e^{-as}F(s)] = f(t-a) for t > a."""
        # Shifted exponential: e^{-as} / (s+1) corresponds to
        # exp(-(t-a)) for t > a, 0 for t < a

        shift = 2.0
        def F_shifted(s):
            return jnp.exp(-shift * s) * exponential_decay_F(s, alpha=1.0)

        result = nilt_fft_uniform(F_shifted, dt=0.05, N=2048, a=0.5)

        # For t > shift, should match exp(-(t-shift))
        t_test = result.t[(result.t > shift + 0.5) & (result.t < 10.0)]
        f_expected = jnp.exp(-(t_test - shift))

        idx = jnp.searchsorted(result.t, t_test)
        f_computed = result.f[idx]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_expected)**2)) / (jnp.sqrt(jnp.mean(f_expected**2)) + 1e-10)
        assert rel_error < 0.15, f"Time shift test failed with error {rel_error:.4f}"

    def test_scaling(self):
        """Test that L^{-1}[F(as)] = (1/a)f(t/a) for a > 0."""
        scale = 2.0
        def F_orig(s):
            return exponential_decay_F(s, alpha=1.0)
        def F_scaled(s):
            return F_orig(scale * s)

        def f_orig(t):
            return exponential_decay_f(t, alpha=1.0)
        def f_scaled_true(t):
            return (1.0 / scale) * f_orig(t / scale)

        result = nilt_fft_uniform(F_scaled, dt=0.05, N=1024, a=0.5)

        t_end = 8.0
        mask = result.t <= t_end
        f_ref = f_scaled_true(result.t[mask])
        f_computed = result.f[mask]

        rel_error = jnp.sqrt(jnp.mean((f_computed - f_ref)**2)) / (jnp.sqrt(jnp.mean(f_ref**2)) + 1e-10)
        assert rel_error < 0.15, f"Scaling test failed with error {rel_error:.4f}"
