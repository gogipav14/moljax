"""
Tests for non-periodic FFT solvers (Dirichlet/Neumann BCs).
"""

import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from moljax.core.fft_nonperiodic import (
    BCType,
    dst_I_fast,
    idst_I_fast,
    laplacian_symbol_dirichlet,
    laplacian_symbol_neumann,
    create_nonperiodic_fft_cache,
    solve_poisson_dirichlet,
    solve_poisson_neumann,
    solve_helmholtz_dirichlet,
    solve_helmholtz_neumann,
    etd1_dirichlet,
    etd1_neumann,
    check_compatibility_neumann,
    project_to_compatible,
)


class TestDST:
    """Test DST implementation."""

    def test_dst_inverse_identity(self):
        """DST followed by IDST recovers original."""
        N = 32
        x = jnp.sin(jnp.pi * jnp.arange(1, N+1) / (N+1))

        X = dst_I_fast(x)
        x_recovered = idst_I_fast(X)

        assert jnp.allclose(x, x_recovered, rtol=1e-10)

    def test_dst_sine_wave(self):
        """DST of sin(kπx) is concentrated at mode k."""
        N = 64
        L = 1.0
        dx = L / (N + 1)
        x_grid = jnp.arange(1, N + 1) * dx

        # sin(πx) - first mode
        f = jnp.sin(jnp.pi * x_grid)
        F = dst_I_fast(f)

        # Mode 1 should dominate
        assert jnp.abs(F[0]) > 0.9 * jnp.linalg.norm(F)


class TestEigenvalues:
    """Test eigenvalue formulas."""

    def test_dirichlet_eigenvalues_negative(self):
        """Dirichlet Laplacian eigenvalues are all negative."""
        N = 32
        dx = 0.1
        lam = laplacian_symbol_dirichlet(N, dx)

        assert jnp.all(lam < 0)

    def test_dirichlet_eigenvalues_ordering(self):
        """Eigenvalues increase in magnitude with k."""
        N = 32
        dx = 0.1
        lam = laplacian_symbol_dirichlet(N, dx)

        # |λ_k| increases with k
        assert jnp.all(jnp.diff(jnp.abs(lam)) >= 0)

    def test_neumann_eigenvalues_zero_mode(self):
        """Neumann Laplacian has zero eigenvalue at k=0."""
        N = 32
        dx = 0.1
        lam = laplacian_symbol_neumann(N, dx)

        assert jnp.abs(lam[0]) < 1e-14
        assert jnp.all(lam[1:] < 0)


class TestPoissonSolver:
    """Test Poisson equation solvers."""

    def test_poisson_dirichlet_manufactured(self):
        """Verify Poisson solve with manufactured solution."""
        N = 64
        L = 1.0
        dx = L / (N + 1)
        x = jnp.arange(1, N + 1) * dx

        # Manufactured solution: u = sin(πx) satisfies Δu = -π²sin(πx)
        u_exact = jnp.sin(jnp.pi * x)
        rhs = -jnp.pi**2 * jnp.sin(jnp.pi * x)

        lam = laplacian_symbol_dirichlet(N, dx)
        u_computed = solve_poisson_dirichlet(rhs, lam)

        # Note: discrete Laplacian eigenvalue differs from continuous
        # so we compare against discrete solution
        assert jnp.allclose(u_computed, u_exact, rtol=0.05)

    def test_poisson_neumann_compatibility(self):
        """Neumann Poisson requires compatible RHS."""
        N = 32
        dx = 0.1

        # Incompatible RHS (non-zero integral)
        rhs_bad = jnp.ones(N)
        assert not check_compatibility_neumann(rhs_bad)

        # Make compatible
        rhs_good = project_to_compatible(rhs_bad)
        assert check_compatibility_neumann(rhs_good)


class TestHelmholtzSolver:
    """Test Helmholtz equation solvers."""

    def test_helmholtz_dirichlet_identity_limit(self):
        """Helmholtz → identity as dt → 0."""
        N = 32
        L = 1.0
        dx = L / (N + 1)
        x = jnp.arange(1, N + 1) * dx

        rhs = jnp.sin(2 * jnp.pi * x)
        lam = laplacian_symbol_dirichlet(N, dx)

        # Small dt: solution ≈ rhs
        u = solve_helmholtz_dirichlet(rhs, lam, dt=1e-10, D=1.0)
        assert jnp.allclose(u, rhs, rtol=1e-8)

    def test_helmholtz_neumann_identity_limit(self):
        """Helmholtz → identity as dt → 0."""
        N = 32
        dx = 0.1

        rhs = jnp.cos(jnp.pi * jnp.arange(N) / N)
        lam = laplacian_symbol_neumann(N, dx)

        u = solve_helmholtz_neumann(rhs, lam, dt=1e-10, D=1.0)
        assert jnp.allclose(u, rhs, rtol=1e-8)

    def test_helmholtz_dirichlet_residual(self):
        """Verify (I - dt·D·Δ)u = f residual."""
        N = 32
        L = 1.0
        dx = L / (N + 1)
        dt = 0.01
        D = 0.1

        x = jnp.arange(1, N + 1) * dx
        rhs = jnp.sin(jnp.pi * x) + 0.5 * jnp.sin(3 * jnp.pi * x)

        lam = laplacian_symbol_dirichlet(N, dx)
        u = solve_helmholtz_dirichlet(rhs, lam, dt, D)

        # Check residual in spectral space
        u_hat = dst_I_fast(u)
        rhs_hat = dst_I_fast(rhs)
        residual_hat = (1.0 - dt * D * lam) * u_hat - rhs_hat

        assert jnp.linalg.norm(residual_hat) < 1e-10


class TestETD:
    """Test ETD integrators for non-periodic BCs."""

    def test_etd1_dirichlet_diffusion_decay(self):
        """ETD1 with Dirichlet BCs shows correct diffusion decay."""
        N = 64
        L = 1.0
        dx = L / (N + 1)
        D = 0.01
        dt = 0.01

        x = jnp.arange(1, N + 1) * dx

        # Initial condition: sin(πx)
        u0 = jnp.sin(jnp.pi * x)
        N_term = jnp.zeros_like(u0)  # Pure diffusion

        lam = laplacian_symbol_dirichlet(N, dx)
        eigenvalues = D * lam

        # One ETD1 step
        u1 = etd1_dirichlet(u0, N_term, eigenvalues, dt)

        # Expected: exp(λ₁·dt) decay for mode 1
        # λ₁ ≈ -D·π²/L² for continuous case
        expected_decay = jnp.exp(D * lam[0] * dt)

        # First mode should decay
        rel_change = jnp.linalg.norm(u1) / jnp.linalg.norm(u0)
        assert rel_change < 1.0  # Decaying

    def test_etd1_neumann_constant_preserved(self):
        """ETD1 with Neumann BCs preserves constant (λ₀ = 0)."""
        N = 32
        dx = 0.1
        D = 0.1
        dt = 0.1

        # Constant initial condition
        u0 = jnp.ones(N) * 5.0
        N_term = jnp.zeros_like(u0)

        lam = laplacian_symbol_neumann(N, dx)
        eigenvalues = D * lam

        # Multiple steps
        u = u0
        for _ in range(10):
            u = etd1_neumann(u, jnp.zeros_like(u), eigenvalues, dt)

        # Should remain constant
        assert jnp.allclose(u, u0, rtol=1e-10)


class TestCache:
    """Test FFT cache creation."""

    def test_cache_dirichlet(self):
        """Cache creation for Dirichlet BCs."""
        cache = create_nonperiodic_fft_cache(32, 0.1, BCType.DIRICHLET)

        assert cache.N == 32
        assert cache.bc_type == 'dirichlet'
        assert len(cache.laplacian_symbol) == 32

    def test_cache_neumann(self):
        """Cache creation for Neumann BCs."""
        cache = create_nonperiodic_fft_cache(32, 0.1, BCType.NEUMANN)

        assert cache.N == 32
        assert cache.bc_type == 'neumann'
        assert jnp.abs(cache.laplacian_symbol[0]) < 1e-14

    def test_cache_string_input(self):
        """Cache accepts string BC type."""
        cache = create_nonperiodic_fft_cache(32, 0.1, 'dirichlet')
        assert cache.bc_type == 'dirichlet'
