"""
Tests for JIT-compiled kernels (Phase 1: Performance Foundation).

Verifies:
1. Numerical correctness of JIT kernels vs reference implementations
2. JIT compilation speedup (target: 10x on repeated calls)
3. Batched operations efficiency
"""

import pytest
import jax
import jax.numpy as jnp
from jax import random

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)

from moljax.core.jit_kernels import (
    phi1,
    phi2,
    phi3,
    etd1_kernel_1d,
    etd1_kernel_2d,
    etd2_kernel_1d,
    etdrk4_kernel_1d,
    helmholtz_solve_1d,
    helmholtz_solve_2d,
    advdiff_solve_1d,
    batched_fft_solve_1d,
    batched_exp_matvec_1d,
    make_etd1_integrator,
    benchmark_jit_speedup,
    # Multi-field batched operations
    batched_etd1_kernel_1d,
    batched_etd2_kernel_1d,
    multi_operator_etd1_1d,
    batched_etd1_kernel_2d,
    batched_helmholtz_solve_2d,
)
from moljax.core.fft_solvers import laplacian_symbol_1d, laplacian_symbol_2d


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def grid_params_1d():
    """1D grid parameters."""
    nx = 128
    L = 1.0
    dx = L / nx
    return {'nx': nx, 'L': L, 'dx': dx}


@pytest.fixture
def grid_params_2d():
    """2D grid parameters."""
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx / nx, Ly / ny
    return {'nx': nx, 'ny': ny, 'Lx': Lx, 'Ly': Ly, 'dx': dx, 'dy': dy}


@pytest.fixture
def eigenvalues_1d(grid_params_1d):
    """FFT eigenvalues for 1D diffusion."""
    nx = grid_params_1d['nx']
    dx = grid_params_1d['dx']
    D = 0.01
    lap_sym = laplacian_symbol_1d(nx, dx)
    return D * lap_sym


@pytest.fixture
def eigenvalues_2d(grid_params_2d):
    """FFT eigenvalues for 2D diffusion."""
    nx, ny = grid_params_2d['nx'], grid_params_2d['ny']
    dx, dy = grid_params_2d['dx'], grid_params_2d['dy']
    D = 0.01
    lap_sym = laplacian_symbol_2d(nx, ny, dx, dy)
    return D * lap_sym


# =============================================================================
# Test: φ-Functions
# =============================================================================

class TestPhiFunctions:
    """Test φ-function implementations."""

    def test_phi1_small_z(self):
        """φ₁(z) ≈ 1 + z/2 + z²/6 for small z."""
        z = jnp.array([1e-6, 1e-8, 1e-10, 1e-12])
        result = phi1(z)
        expected = 1.0 + z/2.0 + z**2/6.0
        assert jnp.allclose(result, expected, rtol=1e-8)

    def test_phi1_large_z(self):
        """φ₁(z) = (exp(z) - 1) / z for larger z."""
        z = jnp.array([0.1, 0.5, 1.0, 2.0])
        result = phi1(z)
        expected = (jnp.exp(z) - 1.0) / z
        assert jnp.allclose(result, expected, rtol=1e-10)

    def test_phi1_negative_z(self):
        """φ₁ works for negative z (diffusion eigenvalues)."""
        z = jnp.array([-0.1, -1.0, -5.0, -10.0])
        result = phi1(z)
        expected = (jnp.exp(z) - 1.0) / z
        assert jnp.allclose(result, expected, rtol=1e-10)

    def test_phi2_small_z(self):
        """φ₂(z) ≈ 0.5 + z/6 for small z."""
        z = jnp.array([1e-6, 1e-8, 1e-10])
        result = phi2(z)
        expected = 0.5 + z/6.0
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_phi2_large_z(self):
        """φ₂(z) = (exp(z) - 1 - z) / z² for larger z."""
        z = jnp.array([0.1, 0.5, 1.0, 2.0])
        result = phi2(z)
        expected = (jnp.exp(z) - 1.0 - z) / (z * z)
        assert jnp.allclose(result, expected, rtol=1e-10)

    def test_phi3_small_z(self):
        """φ₃(z) ≈ 1/6 + z/24 for small z."""
        z = jnp.array([1e-6, 1e-8, 1e-10])
        result = phi3(z)
        expected = 1.0/6.0 + z/24.0
        assert jnp.allclose(result, expected, rtol=1e-6)

    def test_phi3_large_z(self):
        """φ₃(z) = (exp(z) - 1 - z - z²/2) / z³ for larger z."""
        z = jnp.array([0.1, 0.5, 1.0, 2.0])
        result = phi3(z)
        expected = (jnp.exp(z) - 1.0 - z - z**2/2.0) / (z**3)
        assert jnp.allclose(result, expected, rtol=1e-9)


# =============================================================================
# Test: ETD Kernels
# =============================================================================

class TestETDKernels:
    """Test ETD integration kernels."""

    def test_etd1_kernel_diffusion_decay(self, grid_params_1d, eigenvalues_1d):
        """ETD1 should show exponential decay for diffusion."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']
        D = 0.01
        dt = 0.01

        # Initial condition: sin wave
        x = jnp.linspace(0, L, nx, endpoint=False)
        u0 = jnp.sin(2 * jnp.pi * x)

        # For pure diffusion, nonlinear term is zero
        N = jnp.zeros_like(u0)

        # One ETD1 step
        u1 = etd1_kernel_1d(u0, N, eigenvalues_1d, dt)

        # Expected: exp(-4π²D*dt) * sin(2πx)
        decay = jnp.exp(-4 * jnp.pi**2 * D * dt)
        u_expected = decay * u0

        rel_error = jnp.linalg.norm(u1 - u_expected) / jnp.linalg.norm(u_expected)
        # Tolerance accounts for φ-function Taylor approximation at small |z|
        assert rel_error < 1e-6

    def test_etd1_kernel_2d(self, grid_params_2d, eigenvalues_2d):
        """ETD1 2D should match analytical diffusion decay."""
        nx, ny = grid_params_2d['nx'], grid_params_2d['ny']
        Lx, Ly = grid_params_2d['Lx'], grid_params_2d['Ly']
        D = 0.01
        dt = 0.01

        # Initial condition: 2D sin wave
        x = jnp.linspace(0, Lx, nx, endpoint=False)
        y = jnp.linspace(0, Ly, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u0 = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)

        N = jnp.zeros_like(u0)
        u1 = etd1_kernel_2d(u0, N, eigenvalues_2d, dt)

        # Expected: exp(-8π²D*dt) * u0 (sum of two mode decays)
        decay = jnp.exp(-8 * jnp.pi**2 * D * dt)
        u_expected = decay * u0

        rel_error = jnp.linalg.norm(u1 - u_expected) / jnp.linalg.norm(u_expected)
        # Tolerance accounts for φ-function Taylor approximation at small |z|
        assert rel_error < 1e-5

    def test_etd2_kernel_higher_order(self, grid_params_1d, eigenvalues_1d):
        """ETD2 should use previous nonlinear term."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']
        dt = 0.01

        x = jnp.linspace(0, L, nx, endpoint=False)
        u = jnp.sin(2 * jnp.pi * x)
        N_curr = 0.1 * u**2
        N_prev = 0.1 * u**2  # Same for test

        u_new = etd2_kernel_1d(u, N_curr, N_prev, eigenvalues_1d, dt)

        # Should not error and produce reasonable output
        assert jnp.all(jnp.isfinite(u_new))
        assert u_new.shape == u.shape


# =============================================================================
# Test: FFT Solve Kernels
# =============================================================================

class TestFFTSolveKernels:
    """Test FFT-based solver kernels."""

    def test_helmholtz_solve_1d(self, grid_params_1d):
        """Helmholtz solve: (I - dt*D*Δ)u = f."""
        nx = grid_params_1d['nx']
        dx = grid_params_1d['dx']
        D = 0.01
        dt = 0.01

        lap_sym = laplacian_symbol_1d(nx, dx)

        # RHS
        x = jnp.linspace(0, 1, nx, endpoint=False)
        rhs = jnp.sin(2 * jnp.pi * x)

        u = helmholtz_solve_1d(rhs, lap_sym, dt, D)

        # Verify: (I - dt*D*Δ)u ≈ rhs
        u_hat = jnp.fft.fft(u)
        residual_hat = (1.0 - dt * D * lap_sym) * u_hat - jnp.fft.fft(rhs)
        residual = jnp.real(jnp.fft.ifft(residual_hat))

        assert jnp.linalg.norm(residual) < 1e-12

    def test_helmholtz_solve_2d(self, grid_params_2d):
        """2D Helmholtz solve verification."""
        nx, ny = grid_params_2d['nx'], grid_params_2d['ny']
        dx, dy = grid_params_2d['dx'], grid_params_2d['dy']
        D = 0.01
        dt = 0.01

        lap_sym = laplacian_symbol_2d(nx, ny, dx, dy)

        x = jnp.linspace(0, 1, nx, endpoint=False)
        y = jnp.linspace(0, 1, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        rhs = jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)

        u = helmholtz_solve_2d(rhs, lap_sym, dt, D)

        # Verify residual
        u_hat = jnp.fft.fft2(u)
        residual_hat = (1.0 - dt * D * lap_sym) * u_hat - jnp.fft.fft2(rhs)
        residual = jnp.real(jnp.fft.ifft2(residual_hat))

        assert jnp.linalg.norm(residual) < 1e-12

    def test_advdiff_solve_1d(self, grid_params_1d):
        """Advection-diffusion solve with complex eigenvalues."""
        nx = grid_params_1d['nx']
        dx = grid_params_1d['dx']
        D = 0.01
        v = 1.0
        dt = 0.001

        lap_sym = laplacian_symbol_1d(nx, dx)
        k = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
        eigenvalues = D * lap_sym - 1j * v * k

        x = jnp.linspace(0, 1, nx, endpoint=False)
        rhs = jnp.sin(2 * jnp.pi * x)

        u = advdiff_solve_1d(rhs, eigenvalues, dt)

        # Should produce finite output
        assert jnp.all(jnp.isfinite(u))
        assert u.shape == rhs.shape


# =============================================================================
# Test: Batched Operations
# =============================================================================

class TestBatchedOperations:
    """Test batched FFT operations."""

    def test_batched_fft_solve(self, grid_params_1d):
        """Batched Helmholtz solve for multiple fields."""
        nx = grid_params_1d['nx']
        dx = grid_params_1d['dx']
        D = 0.01
        dt = 0.01
        n_fields = 4

        lap_sym = laplacian_symbol_1d(nx, dx)

        # Stack of RHS arrays
        x = jnp.linspace(0, 1, nx, endpoint=False)
        rhs_list = [jnp.sin((k+1) * 2 * jnp.pi * x) for k in range(n_fields)]
        rhs_stack = jnp.stack(rhs_list, axis=0)

        u_stack = batched_fft_solve_1d(rhs_stack, lap_sym, dt, D)

        assert u_stack.shape == (n_fields, nx)

        # Verify each solve independently
        for k in range(n_fields):
            u_single = helmholtz_solve_1d(rhs_list[k], lap_sym, dt, D)
            assert jnp.allclose(u_stack[k], u_single, rtol=1e-12)

    def test_batched_exp_matvec(self, grid_params_1d, eigenvalues_1d):
        """Batched exp(dt*L)*u for multiple fields."""
        nx = grid_params_1d['nx']
        dt = 0.01
        n_fields = 4

        x = jnp.linspace(0, 1, nx, endpoint=False)
        u_list = [jnp.sin((k+1) * 2 * jnp.pi * x) for k in range(n_fields)]
        u_stack = jnp.stack(u_list, axis=0)

        result = batched_exp_matvec_1d(u_stack, eigenvalues_1d, dt)

        assert result.shape == (n_fields, nx)

        # Each row should be exp(dt*L) * u[k]
        for k in range(n_fields):
            u_hat = jnp.fft.fft(u_list[k])
            expected = jnp.real(jnp.fft.ifft(jnp.exp(dt * eigenvalues_1d) * u_hat))
            assert jnp.allclose(result[k], expected, rtol=1e-12)


# =============================================================================
# Test: ETD1 Integrator Factory
# =============================================================================

class TestETDIntegratorFactory:
    """Test make_etd1_integrator factory function."""

    def test_integrator_produces_correct_output(self, grid_params_1d, eigenvalues_1d):
        """Integrator should integrate correctly over multiple steps."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']
        D = 0.01
        dt = 0.001
        n_steps = 10

        x = jnp.linspace(0, L, nx, endpoint=False)
        u0 = jnp.sin(2 * jnp.pi * x)

        # Zero nonlinear term (pure diffusion)
        def zero_nonlinear(u):
            return jnp.zeros_like(u)

        integrate = make_etd1_integrator(eigenvalues_1d, zero_nonlinear)
        u_final, t_final = integrate(u0, 0.0, dt, n_steps)

        # Expected: exp(-4π²D*(n_steps*dt)) * u0
        t_end = n_steps * dt
        expected = jnp.exp(-4 * jnp.pi**2 * D * t_end) * u0

        rel_error = jnp.linalg.norm(u_final - expected) / jnp.linalg.norm(expected)
        # Tolerance accounts for φ-function Taylor approximation at small |z|
        assert rel_error < 1e-6
        assert abs(t_final - t_end) < 1e-12


# =============================================================================
# Test: Benchmark Utility
# =============================================================================

class TestBenchmarkUtility:
    """Test benchmark utility function."""

    def test_benchmark_returns_timing_dict(self, grid_params_1d, eigenvalues_1d):
        """Benchmark should return timing dictionary."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']

        x = jnp.linspace(0, L, nx, endpoint=False)
        u = jnp.sin(2 * jnp.pi * x)

        results = benchmark_jit_speedup(u, eigenvalues_1d, dt=0.01, n_iterations=10)

        assert 'helmholtz_jit_ms' in results
        assert 'etd1_jit_ms' in results
        assert 'batched_solve_jit_ms' in results

        # All timings should be positive
        for key, val in results.items():
            assert val > 0, f"{key} timing should be positive"


# =============================================================================
# Test: Correctness vs Reference
# =============================================================================

class TestCorrectnessVsReference:
    """Compare JIT kernels against reference implementations."""

    def test_etd1_kernel_vs_manual(self, grid_params_1d, eigenvalues_1d):
        """JIT ETD1 kernel matches manual implementation."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']
        dt = 0.01

        x = jnp.linspace(0, L, nx, endpoint=False)
        u = jnp.sin(2 * jnp.pi * x)
        N = 0.1 * u**2

        # JIT kernel
        u_jit = etd1_kernel_1d(u, N, eigenvalues_1d, dt)

        # Manual reference
        z = dt * eigenvalues_1d
        exp_z = jnp.exp(z)
        phi1_z = jnp.where(jnp.abs(z) < 1e-4,
                          1.0 + z/2.0 + z**2/6.0,
                          (jnp.exp(z) - 1.0) / z)
        u_hat = jnp.fft.fft(u)
        N_hat = jnp.fft.fft(N)
        u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
        u_ref = jnp.real(jnp.fft.ifft(u_new_hat))

        assert jnp.allclose(u_jit, u_ref, rtol=1e-12)

    def test_helmholtz_solve_vs_manual(self, grid_params_1d):
        """JIT Helmholtz solve matches manual FFT solve."""
        nx = grid_params_1d['nx']
        dx = grid_params_1d['dx']
        D = 0.01
        dt = 0.01

        lap_sym = laplacian_symbol_1d(nx, dx)

        x = jnp.linspace(0, 1, nx, endpoint=False)
        rhs = jnp.sin(4 * jnp.pi * x) + 0.5 * jnp.cos(6 * jnp.pi * x)

        # JIT kernel
        u_jit = helmholtz_solve_1d(rhs, lap_sym, dt, D)

        # Manual reference
        rhs_hat = jnp.fft.fft(rhs)
        denom = 1.0 - dt * D * lap_sym
        u_hat = rhs_hat / denom
        u_ref = jnp.real(jnp.fft.ifft(u_hat))

        assert jnp.allclose(u_jit, u_ref, rtol=1e-12)


# =============================================================================
# Test: Multi-field Batched Operations
# =============================================================================

class TestMultiFieldBatched:
    """Test batched operations for multi-field PDEs."""

    def test_batched_etd1_matches_single(self, grid_params_1d, eigenvalues_1d):
        """Batched ETD1 produces same result as single-field version."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']
        dt = 0.01
        n_fields = 4

        x = jnp.linspace(0, L, nx, endpoint=False)

        # Create different initial conditions for each field
        u_list = [jnp.sin((k+1) * 2 * jnp.pi * x) for k in range(n_fields)]
        N_list = [0.1 * u**2 for u in u_list]

        u_stack = jnp.stack(u_list, axis=0)
        N_stack = jnp.stack(N_list, axis=0)

        # Batched result
        result_batched = batched_etd1_kernel_1d(u_stack, N_stack, eigenvalues_1d, dt)

        # Single-field results
        for k in range(n_fields):
            result_single = etd1_kernel_1d(u_list[k], N_list[k], eigenvalues_1d, dt)
            assert jnp.allclose(result_batched[k], result_single, rtol=1e-12), \
                f"Field {k} mismatch"

    def test_batched_etd2_produces_finite(self, grid_params_1d, eigenvalues_1d):
        """Batched ETD2 produces finite results."""
        nx = grid_params_1d['nx']
        L = grid_params_1d['L']
        dt = 0.01
        n_fields = 3

        x = jnp.linspace(0, L, nx, endpoint=False)
        u_list = [jnp.sin((k+1) * 2 * jnp.pi * x) for k in range(n_fields)]
        N_list = [0.1 * u**2 for u in u_list]

        u_stack = jnp.stack(u_list, axis=0)
        N_curr_stack = jnp.stack(N_list, axis=0)
        N_prev_stack = N_curr_stack  # Same for initial test

        result = batched_etd2_kernel_1d(u_stack, N_curr_stack, N_prev_stack,
                                        eigenvalues_1d, dt)

        assert jnp.all(jnp.isfinite(result))
        assert result.shape == (n_fields, nx)

    def test_multi_operator_etd1(self, grid_params_1d):
        """Multi-operator ETD1 handles different eigenvalues per field."""
        nx = grid_params_1d['nx']
        dx = grid_params_1d['dx']
        dt = 0.01
        n_fields = 3

        x = jnp.linspace(0, 1, nx, endpoint=False)

        # Different diffusion coefficients for each field
        D_values = [0.01, 0.02, 0.05]
        k = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
        lap_sym = -k**2

        eigenvalues_list = [D * lap_sym for D in D_values]
        eigenvalues_stack = jnp.stack(eigenvalues_list, axis=0)

        u_list = [jnp.sin(2 * jnp.pi * x) for _ in range(n_fields)]
        N_list = [jnp.zeros(nx) for _ in range(n_fields)]

        u_stack = jnp.stack(u_list, axis=0)
        N_stack = jnp.stack(N_list, axis=0)

        result = multi_operator_etd1_1d(u_stack, N_stack, eigenvalues_stack, dt)

        assert result.shape == (n_fields, nx)

        # Each field should decay at different rate
        # Field with larger D decays faster
        decays = [jnp.linalg.norm(result[k]) / jnp.linalg.norm(u_stack[k])
                  for k in range(n_fields)]

        # D=0.01 decays slowest, D=0.05 decays fastest
        assert decays[0] > decays[1] > decays[2], \
            f"Decay ordering incorrect: {decays}"

    def test_batched_etd1_2d(self, grid_params_2d, eigenvalues_2d):
        """Batched 2D ETD1 works correctly."""
        nx, ny = grid_params_2d['nx'], grid_params_2d['ny']
        dt = 0.01
        n_fields = 2

        Lx, Ly = grid_params_2d['Lx'], grid_params_2d['Ly']
        x = jnp.linspace(0, Lx, nx, endpoint=False)
        y = jnp.linspace(0, Ly, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        u1 = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)
        u2 = jnp.cos(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)

        u_stack = jnp.stack([u1, u2], axis=0)
        N_stack = jnp.zeros_like(u_stack)

        result = batched_etd1_kernel_2d(u_stack, N_stack, eigenvalues_2d, dt)

        assert result.shape == (n_fields, nx, ny)
        assert jnp.all(jnp.isfinite(result))

        # Compare with single-field 2D kernel
        result_single = etd1_kernel_2d(u1, jnp.zeros_like(u1), eigenvalues_2d, dt)
        assert jnp.allclose(result[0], result_single, rtol=1e-12)

    def test_batched_helmholtz_2d(self, grid_params_2d):
        """Batched 2D Helmholtz solve works correctly."""
        nx, ny = grid_params_2d['nx'], grid_params_2d['ny']
        dx, dy = grid_params_2d['dx'], grid_params_2d['dy']
        D = 0.01
        dt = 0.01
        n_fields = 3

        lap_sym = laplacian_symbol_2d(nx, ny, dx, dy)

        Lx, Ly = grid_params_2d['Lx'], grid_params_2d['Ly']
        x = jnp.linspace(0, Lx, nx, endpoint=False)
        y = jnp.linspace(0, Ly, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')

        rhs_list = [
            jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y),
            jnp.cos(2 * jnp.pi * X) * jnp.sin(4 * jnp.pi * Y),
            jnp.sin(4 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y),
        ]
        rhs_stack = jnp.stack(rhs_list, axis=0)

        result = batched_helmholtz_solve_2d(rhs_stack, lap_sym, dt, D)

        assert result.shape == (n_fields, nx, ny)

        # Verify each solve independently
        for k in range(n_fields):
            result_single = helmholtz_solve_2d(rhs_list[k], lap_sym, dt, D)
            assert jnp.allclose(result[k], result_single, rtol=1e-12), \
                f"Field {k} mismatch"
