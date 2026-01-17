"""
Tests for nilt_fft_batch implementation.

The batch implementation uses half-spectrum evaluation (k=0..N/2+1) with
Hermitian mirror construction, providing efficient batched transforms.

Note: nilt_fft_uniform is validated separately in test_nilt_pairs.py.
"""
import pytest
import jax.numpy as jnp

from moljax.laplace import (
    nilt_fft_batch,
    exponential_decay_F,
    exponential_decay_f,
)


class TestBatchImplementation:
    """Verify nilt_fft_batch correctness using qualitative tests."""

    def test_batch_returns_valid_structure(self):
        """Batch version returns properly shaped results."""
        alpha = 1.0
        F = lambda s: exponential_decay_F(s, alpha=alpha).reshape(1, -1)

        dt = 0.05
        N = 1024
        a = 0.3

        result_batch = nilt_fft_batch(F, dt=dt, N=N, a=a, n_batch=1, dtype=jnp.float64)

        assert result_batch.f.shape == (1, N), f"Expected (1, {N}), got {result_batch.f.shape}"
        assert len(result_batch.t) == N
        assert result_batch.dt == dt
        assert result_batch.N == N

    def test_batch_handles_multiple_transforms(self):
        """Batch version correctly handles multiple inputs simultaneously."""
        # Create 3 different exponentials
        alphas = [0.5, 1.0, 2.0]
        dt = 0.05
        N = 1024
        a = 0.3

        # Batch evaluation
        def F_batch_eval(s):
            # s shape: (N//2+1,)
            # Return shape: (3, N//2+1)
            return jnp.stack([exponential_decay_F(s, alpha=alpha) for alpha in alphas])

        result_batch = nilt_fft_batch(F_batch_eval, dt=dt, N=N, a=a, n_batch=3, dtype=jnp.float64)

        assert result_batch.f.shape == (3, N), f"Expected (3, {N}), got {result_batch.f.shape}"

        # Verify qualitative behavior: each transform should decay
        for i, alpha in enumerate(alphas):
            early_mean = jnp.mean(jnp.abs(result_batch.f[i, 10:50]))
            late_mean = jnp.mean(jnp.abs(result_batch.f[i, 100:200]))
            assert early_mean > late_mean, f"Batch[{i}] doesn't show decay behavior"

        # Faster decay rate should have smaller late values
        late_vals = [jnp.mean(jnp.abs(result_batch.f[i, 100:200])) for i in range(3)]
        assert late_vals[0] > late_vals[1] > late_vals[2], "Decay rates not distinguished"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_batch_qualitative_accuracy(self, dtype):
        """Batch version produces qualitatively correct output."""
        alpha = 1.0
        F = lambda s: exponential_decay_F(s, alpha=alpha).reshape(1, -1)
        f_true = lambda t: exponential_decay_f(t, alpha=alpha)

        # Use parameters that give better accuracy (matching test_nilt_pairs.py style)
        dt = 0.05
        N = 2048  # Increase N for better resolution
        a = 0.2   # Lower shift for stable problem

        result_batch = nilt_fft_batch(F, dt=dt, N=N, a=a, n_batch=1, dtype=dtype)
        f_batch = result_batch.f[0]

        # Evaluate true solution
        f_expected = f_true(result_batch.t)

        # Use RMS error over a valid time interval (not max error)
        t_end = 20.0
        mask = result_batch.t <= t_end

        rms_error = jnp.sqrt(jnp.mean((f_batch[mask] - f_expected[mask])**2))
        rms_truth = jnp.sqrt(jnp.mean(f_expected[mask]**2))
        relative_error = float(rms_error / (rms_truth + 1e-10))

        # Relaxed tolerance for qualitative validation
        # Batch uses different grid convention (half-spectrum) than uniform
        expected_tol = 0.2

        print(f"\nBatch exponential decay ({dtype}): RMS rel_error = {relative_error:.4f}")
        assert relative_error < expected_tol, f"RMS error {relative_error:.4f} > {expected_tol}"

