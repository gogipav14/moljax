"""
Tests for Hermitian projection (Option B) frequency-domain denoising.

Validates that the projection:
1. Is idempotent on valid Hermitian spectra
2. Removes non-Hermitian noise
3. Preserves or improves accuracy
4. Provides useful diagnostics
"""
import pytest
import jax.numpy as jnp
import numpy as np

from moljax.laplace import (
    nilt_fft_uniform,
    exponential_decay_F,
    exponential_decay_f,
    sine_F,
    sine_f,
    cosine_F,
    cosine_f,
)
from moljax.laplace.nilt_fft import (
    compute_symmetry_residual,
    apply_hermitian_projection,
)


class TestSymmetryResidual:
    """Test computation of Hermitian symmetry residual."""

    def test_perfect_hermitian_has_zero_residual(self):
        """Perfectly Hermitian spectrum should have eps_sym ≈ 0."""
        N = 128
        # Create perfectly Hermitian spectrum
        F_vals = jnp.zeros(N, dtype=jnp.complex64)

        # Set k=1..N/2-1 with conjugate symmetry
        for k in range(1, N // 2):
            val = 1.0 + 0.5j * k
            F_vals = F_vals.at[k].set(val)
            F_vals = F_vals.at[N - k].set(jnp.conj(val))

        # DC and Nyquist are real
        F_vals = F_vals.at[0].set(2.0)
        F_vals = F_vals.at[N // 2].set(1.0)

        eps_sym = compute_symmetry_residual(F_vals)

        # Should be machine precision
        assert eps_sym < 1e-6, f"Perfect Hermitian has eps_sym={eps_sym:.2e}"

    def test_asymmetric_spectrum_has_nonzero_residual(self):
        """Breaking Hermitian symmetry should increase eps_sym."""
        N = 128
        # Start with Hermitian spectrum
        F_vals = jnp.zeros(N, dtype=jnp.complex64)
        for k in range(1, N // 2):
            val = 1.0 + 0.5j * k
            F_vals = F_vals.at[k].set(val)
            F_vals = F_vals.at[N - k].set(jnp.conj(val))

        F_vals = F_vals.at[0].set(2.0)
        F_vals = F_vals.at[N // 2].set(1.0)

        eps_sym_before = compute_symmetry_residual(F_vals)

        # Break symmetry by adding noise to one side
        noise = 0.1 * (1.0 + 1.0j)
        F_vals = F_vals.at[10].add(noise)

        eps_sym_after = compute_symmetry_residual(F_vals)

        assert eps_sym_after > eps_sym_before, "Breaking symmetry should increase eps_sym"
        assert eps_sym_after > 1e-4, f"Asymmetric spectrum has eps_sym={eps_sym_after:.2e}"


class TestHermitianProjection:
    """Test Hermitian projection operator."""

    def test_projection_is_idempotent(self):
        """Projection should be idempotent: P(P(F)) = P(F)."""
        N = 128
        # Create arbitrary spectrum
        rng = np.random.default_rng(42)
        F_vals = jnp.array(
            rng.standard_normal(N) + 1j * rng.standard_normal(N),
            dtype=jnp.complex64
        )

        # Apply projection twice
        F_proj1 = apply_hermitian_projection(F_vals)
        F_proj2 = apply_hermitian_projection(F_proj1)

        # Should be identical
        max_diff = float(jnp.max(jnp.abs(F_proj2 - F_proj1)))
        assert max_diff < 1e-6, f"Projection not idempotent: max_diff={max_diff:.2e}"

    def test_projection_enforces_hermitian_symmetry(self):
        """Projected spectrum should have eps_sym ≈ 0."""
        N = 128
        # Create arbitrary asymmetric spectrum
        rng = np.random.default_rng(42)
        F_vals = jnp.array(
            rng.standard_normal(N) + 1j * rng.standard_normal(N),
            dtype=jnp.complex64
        )

        eps_sym_before = compute_symmetry_residual(F_vals)
        F_proj = apply_hermitian_projection(F_vals)
        eps_sym_after = compute_symmetry_residual(F_proj)

        assert eps_sym_before > 1e-2, "Input should be asymmetric"
        assert eps_sym_after < 1e-6, f"Projected spectrum has eps_sym={eps_sym_after:.2e}"

    def test_projection_preserves_hermitian_spectrum(self):
        """Projection should not change already Hermitian spectrum."""
        N = 128
        # Create perfectly Hermitian spectrum
        F_vals = jnp.zeros(N, dtype=jnp.complex64)
        for k in range(1, N // 2):
            val = 1.0 + 0.5j * k
            F_vals = F_vals.at[k].set(val)
            F_vals = F_vals.at[N - k].set(jnp.conj(val))

        F_vals = F_vals.at[0].set(2.0)
        F_vals = F_vals.at[N // 2].set(1.0)

        F_proj = apply_hermitian_projection(F_vals)

        max_diff = float(jnp.max(jnp.abs(F_proj - F_vals)))
        assert max_diff < 1e-6, f"Hermitian spectrum changed: max_diff={max_diff:.2e}"


class TestNILTWithProjection:
    """Test NILT with optional Hermitian projection."""

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_one_sided_grid_has_high_eps_sym(self, dtype):
        """
        After Fix B (half-spectrum construction with explicit conjugate mirroring),
        nilt_fft_uniform now produces perfectly Hermitian spectra by construction.

        This test validates that eps_sym ≈ 0 (previously it was ~1.0 before Fix B).
        """
        alpha = 1.0
        F = lambda s: exponential_decay_F(s, alpha=alpha)

        dt = 0.05
        N = 1024
        a = 0.3

        result = nilt_fft_uniform(
            F, dt=dt, N=N, a=a, dtype=dtype,
            return_diagnostics=True
        )

        eps_sym = result.diagnostics['eps_sym_before']
        print(f"\nExponential decay ({dtype}): eps_sym = {eps_sym:.2e}")

        # After Fix B: spectrum is constructed with explicit Hermitian symmetry
        # eps_sym should be essentially machine precision
        expected_eps = 1e-5 if dtype == jnp.float32 else 1e-12
        assert eps_sym < expected_eps, f"Fix B spectrum has eps_sym={eps_sym:.2e} > {expected_eps:.2e}"

    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
    def test_projection_reduces_eps_sym(self, dtype):
        """Projection should reduce or maintain eps_sym."""
        alpha = 1.0
        F = lambda s: exponential_decay_F(s, alpha=alpha)

        dt = 0.05
        N = 1024
        a = 0.3

        # Without projection
        result_no_proj = nilt_fft_uniform(
            F, dt=dt, N=N, a=a, dtype=dtype,
            apply_projection=False,
            return_diagnostics=True
        )

        # With projection
        result_proj = nilt_fft_uniform(
            F, dt=dt, N=N, a=a, dtype=dtype,
            apply_projection=True,
            return_diagnostics=True
        )

        eps_sym_before = result_proj.diagnostics['eps_sym_before']
        eps_sym_after = result_proj.diagnostics['eps_sym_after']

        print(f"\n{dtype}: eps_sym before={eps_sym_before:.2e}, after={eps_sym_after:.2e}")

        # Projection should reduce or maintain eps_sym
        assert eps_sym_after <= eps_sym_before * 1.01, "Projection increased eps_sym"

    @pytest.mark.parametrize("test_case", [
        ("exponential", lambda s: exponential_decay_F(s, alpha=1.0), lambda t: exponential_decay_f(t, alpha=1.0)),
        ("sine", lambda s: sine_F(s, omega=1.0), lambda t: sine_f(t, omega=1.0)),
        ("cosine", lambda s: cosine_F(s, omega=1.0), lambda t: cosine_f(t, omega=1.0)),
    ])
    def test_projection_preserves_accuracy(self, test_case):
        """Projection should not degrade accuracy on validated cases."""
        name, F, f_true = test_case

        dt = 0.05
        N = 2048
        a = 0.2

        # Without projection
        result_no_proj = nilt_fft_uniform(
            F, dt=dt, N=N, a=a, dtype=jnp.float64,
            apply_projection=False
        )

        # With projection
        result_proj = nilt_fft_uniform(
            F, dt=dt, N=N, a=a, dtype=jnp.float64,
            apply_projection=True
        )

        # Evaluate true solution
        f_expected = f_true(result_no_proj.t)

        # Compare RMS errors over valid interval
        t_end = 20.0
        mask = result_no_proj.t <= t_end

        rms_no_proj = jnp.sqrt(jnp.mean((result_no_proj.f[mask] - f_expected[mask])**2))
        rms_proj = jnp.sqrt(jnp.mean((result_proj.f[mask] - f_expected[mask])**2))

        rms_truth = jnp.sqrt(jnp.mean(f_expected[mask]**2))
        rel_err_no_proj = float(rms_no_proj / (rms_truth + 1e-10))
        rel_err_proj = float(rms_proj / (rms_truth + 1e-10))

        print(f"\n{name}: RMS error without projection = {rel_err_no_proj:.4f}")
        print(f"{name}: RMS error with projection = {rel_err_proj:.4f}")

        # Projection should not significantly degrade accuracy
        # Allow up to 10% degradation for numerical noise
        assert rel_err_proj < rel_err_no_proj * 1.1, \
            f"{name}: Projection degraded accuracy by {(rel_err_proj/rel_err_no_proj - 1)*100:.1f}%"

    def test_diagnostics_structure(self):
        """Diagnostics should have expected structure."""
        F = lambda s: exponential_decay_F(s, alpha=1.0)

        result = nilt_fft_uniform(
            F, dt=0.05, N=1024, a=0.3,
            apply_projection=True,
            return_diagnostics=True
        )

        diag = result.diagnostics
        assert diag is not None, "Diagnostics should be present"
        assert 'eps_sym_before' in diag
        assert 'eps_sym_after' in diag
        assert 'omega_max' in diag
        assert 'delta_omega' in diag
        assert 'projection_applied' in diag
        assert diag['projection_applied'] is True

        # Check values are reasonable
        # Note: One-sided grid can have eps_sym ≈ 1.0 due to DC halving asymmetry
        assert 0 <= diag['eps_sym_before'] < 2.0
        assert 0 <= diag['eps_sym_after'] <= diag['eps_sym_before']
        assert diag['omega_max'] > 0
        assert diag['delta_omega'] > 0


class TestDiagnosticsOnly:
    """Test diagnostics without projection."""

    def test_diagnostics_without_projection(self):
        """Can request diagnostics without applying projection."""
        F = lambda s: exponential_decay_F(s, alpha=1.0)

        result = nilt_fft_uniform(
            F, dt=0.05, N=1024, a=0.3,
            apply_projection=False,
            return_diagnostics=True
        )

        diag = result.diagnostics
        assert diag is not None
        assert 'eps_sym_before' in diag
        assert 'eps_sym_after' not in diag  # Only computed if projection applied
        assert diag['projection_applied'] is False

    def test_no_diagnostics_by_default(self):
        """Default behavior should not include diagnostics."""
        F = lambda s: exponential_decay_F(s, alpha=1.0)

        result = nilt_fft_uniform(
            F, dt=0.05, N=1024, a=0.3
        )

        assert result.diagnostics is None, "Default should not include diagnostics"
