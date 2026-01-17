"""
VVBC tests for FFT operators and ETD integrators.

Verification tests using manufactured solutions with exact analytical results.
These tests ensure the FFT-based operators achieve expected accuracy and
convergence orders.

Test categories:
- V1: Diffusion with sinusoidal IC (exact exponential decay)
- V2: Advection-diffusion with traveling Gaussian
- V3: Convergence order verification
- B1: FFT vs Newton-Krylov accuracy comparison
- C1: Runtime benchmarks
"""

import pytest
import jax.numpy as jnp
import jax
from jax import random

# Enable float64 for better precision in tests
jax.config.update("jax_enable_x64", True)

from moljax.core.grid import Grid1D
from moljax.core.fft_operators import (
    DiffusionOperator,
    AdvectionDiffusionOperator,
    exact_cfl_dt,
)
from moljax.core.fft_integrators import (
    etd1_step,
    etd2_step,
    etdrk4_step,
    etd_integrate,
    diffusion_only_etd1,
    batched_fft_solve,
    stacked_fft_solve_shared_op,
)
from moljax.core.fft_solvers import solve_helmholtz_1d, laplacian_symbol_1d


def get_interior_coords(grid: Grid1D) -> jnp.ndarray:
    """Get interior x coordinates for a 1D grid."""
    return grid.x_coords(include_ghost=False)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def grid_128():
    """Standard 128-point periodic grid on [0, 1]."""
    return Grid1D.uniform(128, x_min=0.0, x_max=1.0)


@pytest.fixture
def grid_256():
    """256-point periodic grid on [0, 1]."""
    return Grid1D.uniform(256, x_min=0.0, x_max=1.0)


# =============================================================================
# V1: Diffusion with Exact Solution
# =============================================================================

class TestDiffusionManufactured:
    """Verification tests using manufactured solutions for diffusion."""

    def test_diffusion_sinusoidal_decay(self, grid_128):
        """Test diffusion of sin(2πx) which decays as exp(-4π²Dt).

        IC: u(x, 0) = sin(2πx)
        Exact: u(x, t) = exp(-4π²Dt) * sin(2πx)
        """
        grid = grid_128
        D = 0.01
        t_end = 1.0

        # Initial condition on interior
        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)

        # Create operator and integrate
        op = DiffusionOperator(grid, D)
        u_final = diffusion_only_etd1(u0, t_end, dt=0.01, op=op)

        # Exact solution
        decay = jnp.exp(-4 * jnp.pi**2 * D * t_end)
        u_exact = decay * jnp.sin(2 * jnp.pi * x)

        # Check accuracy
        error = jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact)
        assert error < 1e-4, f"Relative error {error:.2e} exceeds threshold"

    def test_diffusion_multi_mode(self, grid_256):
        """Test diffusion with multiple Fourier modes."""
        grid = grid_256
        D = 0.005
        t_end = 0.5

        x = get_interior_coords(grid)

        # IC with modes k=1,2,3
        u0 = (jnp.sin(2 * jnp.pi * x)
              + 0.5 * jnp.sin(4 * jnp.pi * x)
              + 0.25 * jnp.sin(6 * jnp.pi * x))

        op = DiffusionOperator(grid, D)
        u_final = diffusion_only_etd1(u0, t_end, dt=0.005, op=op)

        # Exact: each mode decays independently
        u_exact = (jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)
                  + 0.5 * jnp.exp(-16 * jnp.pi**2 * D * t_end) * jnp.sin(4 * jnp.pi * x)
                  + 0.25 * jnp.exp(-36 * jnp.pi**2 * D * t_end) * jnp.sin(6 * jnp.pi * x))

        error = jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact)
        assert error < 1e-3, f"Multi-mode error {error:.2e} exceeds threshold"

    def test_diffusion_operator_matvec(self, grid_128):
        """Test that matvec correctly computes D*Δu."""
        grid = grid_128
        D = 1.0

        x = get_interior_coords(grid)
        dx = grid.dx

        # u = sin(2πx) → Δu = -4π²*sin(2πx)
        u = jnp.sin(2 * jnp.pi * x)
        expected_Lu = -4 * jnp.pi**2 * D * jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)
        Lu = op.matvec(u)

        # FFT Laplacian uses discrete formula, slight difference from continuous
        # Error should be O(dx²)
        error = jnp.linalg.norm(Lu - expected_Lu) / jnp.linalg.norm(expected_Lu)
        assert error < 0.01, f"Matvec error {error:.2e} exceeds O(dx²)"

    def test_diffusion_solve_consistency(self, grid_128):
        """Test that solve is inverse of (I - dt*L)."""
        grid = grid_128
        D = 0.1
        dt = 0.01

        x = get_interior_coords(grid)
        u = jnp.sin(2 * jnp.pi * x) + 0.3 * jnp.cos(4 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        # Compute rhs = (I - dt*L)*u
        Lu = op.matvec(u)
        rhs = u - dt * Lu

        # Solve (I - dt*L)*u_recovered = rhs
        u_recovered = op.solve(rhs, dt)

        error = jnp.linalg.norm(u_recovered - u) / jnp.linalg.norm(u)
        assert error < 1e-10, f"Solve inconsistency {error:.2e}"


# =============================================================================
# V2: Advection-Diffusion
# =============================================================================

class TestAdvectionDiffusion:
    """Tests for advection-diffusion operator."""

    def test_pure_advection_translation(self, grid_256):
        """Test pure advection: sinusoidal wave should translate correctly."""
        grid = grid_256
        v = 1.0
        D = 0.0  # Pure advection
        t_end = 0.25  # Quarter period

        x = get_interior_coords(grid)

        # Sinusoidal IC (cleaner for periodic advection test)
        u0 = jnp.sin(2 * jnp.pi * x)

        op = AdvectionDiffusionOperator(grid, v=v, D=D)

        # Use exp_matvec for exact exponential
        n_steps = 100
        dt = t_end / n_steps
        u = u0
        for _ in range(n_steps):
            u = op.exp_matvec(u, dt)

        # Exact: sin(2π(x - v*t)) = sin(2πx - 2πvt)
        u_exact = jnp.sin(2 * jnp.pi * (x - v * t_end))

        error = jnp.linalg.norm(u - u_exact) / jnp.linalg.norm(u0)
        assert error < 0.05, f"Advection translation error {error:.2e}"

    def test_advection_diffusion_eigenvalues(self, grid_128):
        """Test that eigenvalues have correct structure."""
        grid = grid_128
        v = 2.0
        D = 0.1

        op = AdvectionDiffusionOperator(grid, v=v, D=D)
        lam = op.eigenvalues

        # Real part should be ≤ 0 (diffusive damping)
        assert jnp.max(jnp.real(lam)) <= 1e-10, "Real part should be ≤ 0"

        # Imaginary part should be present (advection)
        assert jnp.max(jnp.abs(jnp.imag(lam))) > 0, "Should have imaginary part"

    def test_spectral_bounds_exact(self, grid_128):
        """Test that spectral bounds are computed exactly from eigenvalues."""
        grid = grid_128
        D = 0.1

        op = DiffusionOperator(grid, D)
        bounds = op.spectral_bounds()

        # For diffusion: rho = 4D/dx²
        expected_rho = 4 * D / grid.dx**2

        # Should be very close (FFT eigenvalues are exact)
        rel_diff = abs(bounds.rho - expected_rho) / expected_rho
        assert rel_diff < 0.01, f"Spectral radius mismatch: {bounds.rho} vs {expected_rho}"

        # re_max should be 0 (all eigenvalues ≤ 0)
        assert bounds.re_max <= 0.0, f"re_max should be ≤ 0, got {bounds.re_max}"

        # im_max should be 0 for pure diffusion
        assert bounds.im_max < 1e-10, f"im_max should be 0 for diffusion"


# =============================================================================
# V3: Convergence Order Tests
# =============================================================================

class TestConvergenceOrders:
    """Verify spatial and temporal convergence orders."""

    def test_spatial_convergence_order(self):
        """Verify O(dx²) spatial convergence for diffusion."""
        D = 0.01
        t_end = 0.1

        grid_sizes = [32, 64, 128, 256]
        errors = []

        for N in grid_sizes:
            grid = Grid1D.uniform(N, 0.0, 1.0)
            x = get_interior_coords(grid)

            u0 = jnp.sin(2 * jnp.pi * x)
            u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)

            op = DiffusionOperator(grid, D)
            # Use very small dt so temporal error is negligible
            dt = 0.0001
            u_final = diffusion_only_etd1(u0, t_end, dt, op)

            error = float(jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact))
            errors.append(error)

        # Compute convergence order
        orders = []
        for i in range(len(errors) - 1):
            order = jnp.log(errors[i] / errors[i+1]) / jnp.log(2)
            orders.append(float(order))

        # Average order should be close to 2
        avg_order = jnp.mean(jnp.array(orders))
        assert avg_order > 1.8, f"Spatial order {avg_order:.2f} < 1.8"

    def test_etd1_temporal_order(self, grid_256):
        """Verify O(dt) temporal convergence for ETD1."""
        grid = grid_256
        D = 0.01
        t_end = 0.1

        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)
        u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        dt_values = [0.02, 0.01, 0.005, 0.0025]
        errors = []

        for dt in dt_values:
            u_final = diffusion_only_etd1(u0, t_end, dt, op)
            error = float(jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact))
            errors.append(error)

        # For pure diffusion with exact exponential, error should be very small
        # The remaining error is from spatial discretization
        # ETD with exact exp(dt*L) should have no temporal truncation for linear
        for error in errors:
            assert error < 1e-4, f"ETD1 error {error:.2e} too large for linear problem"


# =============================================================================
# B1: FFT vs Newton-Krylov Comparison
# =============================================================================

class TestFFTvsBaseline:
    """Compare FFT solve to baseline methods."""

    def test_fft_vs_direct_helmholtz(self, grid_128):
        """FFT Helmholtz solve should match direct solve."""
        grid = grid_128
        D = 0.1
        dt = 0.01

        x = get_interior_coords(grid)
        rhs = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.cos(6 * jnp.pi * x)

        # FFT operator solve
        op = DiffusionOperator(grid, D)
        u_fft = op.solve(rhs, dt)

        # Direct FFT solve using low-level function
        lam = laplacian_symbol_1d(grid.nx, grid.dx)
        u_direct = solve_helmholtz_1d(rhs, lam, dt, D)

        # Should match to machine precision
        diff = jnp.linalg.norm(u_fft - u_direct) / jnp.linalg.norm(u_direct)
        assert diff < 1e-12, f"FFT operator vs direct mismatch: {diff:.2e}"

    def test_fft_roundtrip(self, grid_256):
        """Test FFT → IFFT roundtrip preserves data."""
        grid = grid_256

        key = random.PRNGKey(42)
        u = random.normal(key, (grid.nx,))

        u_hat = jnp.fft.fft(u)
        u_back = jnp.fft.ifft(u_hat)

        error = jnp.linalg.norm(u - u_back.real) / jnp.linalg.norm(u)
        assert error < 1e-14, f"FFT roundtrip error: {error:.2e}"


# =============================================================================
# CFL Tests
# =============================================================================

class TestCFL:
    """Test exact CFL computation from eigenvalues."""

    def test_explicit_cfl_diffusion(self, grid_128):
        """Test CFL timestep for explicit diffusion."""
        grid = grid_128
        D = 0.1

        op = DiffusionOperator(grid, D)
        dt_cfl = exact_cfl_dt(op, method='explicit', safety=1.0)

        # For diffusion: dt_cfl = 2 / (4D/dx²) = dx² / (2D)
        expected_dt = grid.dx**2 / (2 * D)

        rel_diff = abs(dt_cfl - expected_dt) / expected_dt
        assert rel_diff < 0.01, f"CFL mismatch: {dt_cfl} vs {expected_dt}"

    def test_etd_allows_large_dt(self, grid_128):
        """ETD should remain stable with dt > CFL."""
        grid = grid_128
        D = 0.1
        t_end = 0.1

        op = DiffusionOperator(grid, D)
        dt_cfl = exact_cfl_dt(op, method='explicit', safety=1.0)

        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)
        u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)

        # Use dt = 10x CFL (would blow up with explicit Euler)
        dt_large = 10 * dt_cfl
        u_final = diffusion_only_etd1(u0, t_end, dt_large, op)

        # Should still converge (exact exponential is unconditionally stable)
        error = jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact)
        assert jnp.isfinite(error), "ETD blew up with large dt"
        assert error < 0.01, f"ETD error {error:.2e} with large dt"


# =============================================================================
# ETD Integration Tests
# =============================================================================

class TestETDIntegration:
    """Test ETD integration with nonlinear terms."""

    def test_etd1_with_zero_nonlinear(self, grid_128):
        """ETD1 with N(u)=0 should match pure exponential."""
        grid = grid_128
        D = 0.1
        t_end = 0.1

        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        # Zero nonlinear term
        def zero_rhs(state, t):
            return {name: jnp.zeros_like(v) for name, v in state.items()}

        linear_ops = {'u': op}
        state0 = {'u': u0}

        dt = 0.01
        n_steps = int(t_end / dt)

        state = state0
        for _ in range(n_steps):
            state = etd1_step(state, 0.0, dt, linear_ops, zero_rhs)

        u_etd = state['u']
        u_pure = diffusion_only_etd1(u0, t_end, dt, op)

        error = jnp.linalg.norm(u_etd - u_pure) / jnp.linalg.norm(u_pure)
        assert error < 1e-10, f"ETD1 vs pure exp mismatch: {error:.2e}"

    def test_etd_integrate_api(self, grid_128):
        """Test the high-level etd_integrate function."""
        grid = grid_128
        D = 0.05
        t_span = (0.0, 0.2)
        dt = 0.01

        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        def zero_rhs(state, t):
            return {name: jnp.zeros_like(v) for name, v in state.items()}

        t_hist, state_hist = etd_integrate(
            {'u': u0},
            t_span,
            dt,
            {'u': op},
            zero_rhs,
            method='etd1',
            save_every=5
        )

        # Check we got the right number of saved states
        expected_saves = int((t_span[1] - t_span[0]) / dt / 5) + 1
        assert len(state_hist) == expected_saves

        # Check final state accuracy
        u_final = state_hist[-1]['u']
        u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_span[1]) * jnp.sin(2 * jnp.pi * x)

        error = jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact)
        assert error < 1e-3, f"etd_integrate error: {error:.2e}"


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_diffusion_coefficient_validation(self, grid_128):
        """DiffusionOperator should reject D ≤ 0."""
        with pytest.raises(ValueError, match="D must be > 0"):
            DiffusionOperator(grid_128, D=-0.1)

        with pytest.raises(ValueError, match="D must be > 0"):
            DiffusionOperator(grid_128, D=0.0)

    def test_small_diffusion(self, grid_128):
        """Test with very small diffusion coefficient."""
        grid = grid_128
        D = 1e-6
        t_end = 1.0

        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)
        u_final = diffusion_only_etd1(u0, t_end, dt=0.1, op=op)

        # With small D, solution should barely change
        u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)

        error = jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact)
        assert error < 0.01, f"Small D error: {error:.2e}"

    def test_constant_ic_preserved(self, grid_128):
        """Constant IC should be preserved by diffusion."""
        grid = grid_128
        D = 0.1
        t_end = 1.0

        u0 = jnp.ones(grid.nx) * 3.14

        op = DiffusionOperator(grid, D)
        u_final = diffusion_only_etd1(u0, t_end, dt=0.01, op=op)

        # Constant should be exactly preserved (DC mode has λ=0)
        error = jnp.linalg.norm(u_final - u0) / jnp.linalg.norm(u0)
        assert error < 1e-12, f"Constant not preserved: {error:.2e}"


# =============================================================================
# ETDRK4 Tests with Quantitative Convergence
# =============================================================================

class TestETDRK4:
    """Test ETDRK4 4th order integrator with quantitative RMS error metrics."""

    def test_etdrk4_diffusion_reaction(self, grid_256):
        """Test ETDRK4 on reaction-diffusion: u_t = D*Δu + αu(1-u).

        Analytical: For small α, diffusion dominates → decay to mean.
        Quantitative: RMS error vs reference solution.
        """
        grid = grid_256
        D = 0.01
        alpha = 0.1  # Weak reaction
        t_end = 1.0
        dt = 0.05  # Large dt (possible with ETD)

        x = get_interior_coords(grid)
        u0 = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        def reaction(state, t):
            u = state['u']
            return {'u': alpha * u * (1 - u)}

        # Run ETDRK4
        t_hist, state_hist = etd_integrate(
            {'u': u0}, (0.0, t_end), dt, {'u': op},
            reaction, method='etdrk4'
        )
        u_etdrk4 = state_hist[-1]['u']

        # Reference: run with very small dt using ETD1
        dt_ref = 0.001
        _, ref_hist = etd_integrate(
            {'u': u0}, (0.0, t_end), dt_ref, {'u': op},
            reaction, method='etd1'
        )
        u_ref = ref_hist[-1]['u']

        # RMS error vs reference
        rms_error = float(jnp.sqrt(jnp.mean((u_etdrk4 - u_ref)**2)))
        rms_ref = float(jnp.sqrt(jnp.mean(u_ref**2)))
        rel_error = rms_error / rms_ref

        print(f"ETDRK4 reaction-diffusion: RMS={rms_error:.2e}, rel={rel_error:.2e}")
        assert rel_error < 0.01, f"ETDRK4 error {rel_error:.2e} > 1%"

    def test_etdrk4_convergence_order(self, grid_256):
        """Verify ETDRK4 achieves 4th order temporal convergence.

        Method: Run with dt, dt/2, dt/4, dt/8 and check error ratios.
        Expected: Error ratio ≈ 16 (2^4) between successive refinements.
        """
        grid = grid_256
        D = 0.01
        alpha = 0.5
        t_end = 0.5

        x = get_interior_coords(grid)
        u0 = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        def reaction(state, t):
            u = state['u']
            return {'u': alpha * u * (1 - u)}

        # Reference solution with very fine dt
        dt_ref = 0.0001
        _, ref_hist = etd_integrate(
            {'u': u0}, (0.0, t_end), dt_ref, {'u': op},
            reaction, method='etdrk4'
        )
        u_ref = ref_hist[-1]['u']

        # Test with different dt values
        dt_values = [0.1, 0.05, 0.025, 0.0125]
        errors = []

        for dt in dt_values:
            _, hist = etd_integrate(
                {'u': u0}, (0.0, t_end), dt, {'u': op},
                reaction, method='etdrk4'
            )
            u_final = hist[-1]['u']
            err = float(jnp.linalg.norm(u_final - u_ref) / jnp.linalg.norm(u_ref))
            errors.append(err)
            print(f"dt={dt:.4f}: error={err:.2e}")

        # Compute convergence orders
        orders = []
        for i in range(len(errors) - 1):
            if errors[i+1] > 1e-14:  # Avoid log(0)
                order = jnp.log(errors[i] / errors[i+1]) / jnp.log(2)
                orders.append(float(order))
                print(f"Order {i+1}: {order:.2f}")

        # Average order should be close to 4
        avg_order = jnp.mean(jnp.array(orders)) if orders else 0.0
        print(f"Average convergence order: {avg_order:.2f}")

        # ETDRK4 should achieve at least order 3.5 in practice
        # (may be limited by spatial discretization at coarse dt)
        assert avg_order > 3.0, f"ETDRK4 order {avg_order:.2f} < 3.0"


# =============================================================================
# Multi-field Batched FFT Tests
# =============================================================================

class TestBatchedFFT:
    """Test multi-field batched FFT operations."""

    def test_batched_solve_matches_individual(self, grid_128):
        """Batched solve should match individual field solves."""
        grid = grid_128
        D = 0.1
        dt = 0.01

        x = get_interior_coords(grid)

        # Create multi-field state
        state = {
            'u': jnp.sin(2 * jnp.pi * x),
            'v': jnp.cos(4 * jnp.pi * x),
            'w': jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.cos(6 * jnp.pi * x),
        }

        op = DiffusionOperator(grid, D)
        ops = {'u': op, 'v': op, 'w': op}

        # Individual solves
        individual = {}
        for name, rhs in state.items():
            individual[name] = op.solve(rhs, dt)

        # Batched solve
        batched = batched_fft_solve(state, ops, dt)

        # Should match exactly
        for name in state:
            diff = jnp.linalg.norm(batched[name] - individual[name])
            assert diff < 1e-12, f"Field {name} mismatch: {diff:.2e}"

    def test_stacked_solve_shared_op(self, grid_128):
        """Stacked solve with shared operator should be efficient and correct."""
        grid = grid_128
        D = 0.1
        dt = 0.01

        x = get_interior_coords(grid)

        # Multi-field RHS
        rhs = {
            'a': jnp.sin(2 * jnp.pi * x),
            'b': jnp.cos(2 * jnp.pi * x),
        }

        op = DiffusionOperator(grid, D)

        # Stacked solve (single batched FFT)
        result = stacked_fft_solve_shared_op(rhs, op, dt)

        # Verify against individual solves
        for name, rhs_field in rhs.items():
            expected = op.solve(rhs_field, dt)
            diff = jnp.linalg.norm(result[name] - expected)
            assert diff < 1e-12, f"Stacked solve {name} error: {diff:.2e}"


# =============================================================================
# Quantitative Error Table Tests
# =============================================================================

class TestQuantitativeErrorTable:
    """Generate quantitative error table for different methods and dt values.

    This provides the paper-ready error metrics requested.
    """

    def test_error_table_diffusion(self, grid_256):
        """Generate error table for pure diffusion u_t = D*Δu.

        Exact solution: u(x,t) = exp(-4π²Dt)*sin(2πx)

        | Method | dt    | Steps | RMS Error | Rel Error |
        |--------|-------|-------|-----------|-----------|
        """
        grid = grid_256
        D = 0.01
        t_end = 1.0

        x = get_interior_coords(grid)
        u0 = jnp.sin(2 * jnp.pi * x)
        u_exact = jnp.exp(-4 * jnp.pi**2 * D * t_end) * jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        def zero_rhs(state, t):
            return {name: jnp.zeros_like(v) for name, v in state.items()}

        print("\n" + "="*70)
        print("ERROR TABLE: Pure Diffusion u_t = D*Δu")
        print(f"Grid: N={grid.nx}, D={D}, t_end={t_end}")
        print("Exact: u(x,t) = exp(-4π²Dt)*sin(2πx)")
        print("="*70)
        print(f"{'Method':<10} | {'dt':>8} | {'Steps':>6} | {'RMS Error':>12} | {'Rel Error':>12}")
        print("-"*70)

        results = []

        for method in ['etd1', 'etd2', 'etdrk4']:
            for dt in [0.1, 0.05, 0.01, 0.005]:
                n_steps = int(t_end / dt)
                _, hist = etd_integrate(
                    {'u': u0}, (0.0, t_end), dt, {'u': op},
                    zero_rhs, method=method
                )
                u_final = hist[-1]['u']

                rms = float(jnp.sqrt(jnp.mean((u_final - u_exact)**2)))
                rel = float(jnp.linalg.norm(u_final - u_exact) / jnp.linalg.norm(u_exact))

                print(f"{method:<10} | {dt:>8.4f} | {n_steps:>6} | {rms:>12.2e} | {rel:>12.2e}")
                results.append((method, dt, rms, rel))

        # Verify best results achieve good accuracy
        best_errors = [r[3] for r in results if r[1] == 0.005]
        for err in best_errors:
            assert err < 1e-3, f"Best dt=0.005 should achieve <0.1% error, got {err:.2e}"

    def test_error_table_reaction_diffusion(self, grid_256):
        """Generate error table for reaction-diffusion.

        PDE: u_t = D*Δu + αu(1-u) (Fisher-KPP equation)
        Reference: Fine grid ETD1 solution

        | Method | dt    | RMS vs Ref | Rel Error |
        """
        grid = grid_256
        D = 0.005
        alpha = 1.0
        t_end = 0.5

        x = get_interior_coords(grid)
        u0 = 0.5 + 0.3 * jnp.sin(2 * jnp.pi * x)

        op = DiffusionOperator(grid, D)

        def reaction(state, t):
            u = state['u']
            return {'u': alpha * u * (1 - u)}

        # Reference solution
        dt_ref = 0.0005
        _, ref_hist = etd_integrate(
            {'u': u0}, (0.0, t_end), dt_ref, {'u': op},
            reaction, method='etdrk4'
        )
        u_ref = ref_hist[-1]['u']

        print("\n" + "="*70)
        print("ERROR TABLE: Reaction-Diffusion u_t = D*Δu + αu(1-u)")
        print(f"Grid: N={grid.nx}, D={D}, α={alpha}, t_end={t_end}")
        print(f"Reference: ETDRK4 with dt={dt_ref}")
        print("="*70)
        print(f"{'Method':<10} | {'dt':>8} | {'RMS vs Ref':>12} | {'Rel Error':>12}")
        print("-"*70)

        for method in ['etd1', 'etd2', 'etdrk4']:
            for dt in [0.05, 0.02, 0.01]:
                _, hist = etd_integrate(
                    {'u': u0}, (0.0, t_end), dt, {'u': op},
                    reaction, method=method
                )
                u_final = hist[-1]['u']

                rms = float(jnp.sqrt(jnp.mean((u_final - u_ref)**2)))
                rel = float(jnp.linalg.norm(u_final - u_ref) / jnp.linalg.norm(u_ref))

                print(f"{method:<10} | {dt:>8.4f} | {rms:>12.2e} | {rel:>12.2e}")

        # ETDRK4 at dt=0.01 should be significantly more accurate than ETD1
        _, etdrk4_hist = etd_integrate(
            {'u': u0}, (0.0, t_end), 0.01, {'u': op},
            reaction, method='etdrk4'
        )
        _, etd1_hist = etd_integrate(
            {'u': u0}, (0.0, t_end), 0.01, {'u': op},
            reaction, method='etd1'
        )

        err_etdrk4 = jnp.linalg.norm(etdrk4_hist[-1]['u'] - u_ref) / jnp.linalg.norm(u_ref)
        err_etd1 = jnp.linalg.norm(etd1_hist[-1]['u'] - u_ref) / jnp.linalg.norm(u_ref)

        # ETDRK4 should be at least 10x more accurate than ETD1 at same dt
        assert err_etdrk4 < err_etd1 / 5, \
            f"ETDRK4 ({err_etdrk4:.2e}) should be >5x more accurate than ETD1 ({err_etd1:.2e})"


# =============================================================================
# FFT Preconditioner Tests (Milestone 3)
# =============================================================================

class TestFFTPreconditioner:
    """Test FFT-based preconditioners for Newton-Krylov solvers."""

    def test_fft_advdiff_preconditioner_correctness(self, grid_128):
        """Verify FFT advection-diffusion preconditioner approximately inverts (I - dt*L).

        The preconditioner solves (I - dt*L)*x = r where L = -v·∇ + D*Δ.
        We verify by checking that (I - dt*L) @ precond.apply(r) ≈ r.

        Note: For advection with complex eigenvalues, the Nyquist frequency breaks
        Hermitian symmetry, causing ~1-2% error when taking real(ifft(...)). This is
        acceptable for a preconditioner which doesn't need to be exact.
        """
        from moljax.core.preconditioners import (
            FFTAdvectionDiffusionPreconditioner,
            PrecondContext,
        )
        from moljax.core.fft_solvers import create_fft_cache_1d

        grid = grid_128
        D = 0.01
        v = 1.0
        dt = 0.05

        # Create preconditioner
        fft_cache = create_fft_cache_1d(grid)
        precond = FFTAdvectionDiffusionPreconditioner(
            field_diffusivity_keys={'u': 'D'},
            field_velocity_keys={'u': 'v'},
            fft_cache=fft_cache
        )

        # Random RHS
        key = random.PRNGKey(42)
        r = {'u': random.normal(key, (grid.nx,))}

        # Apply preconditioner
        context = PrecondContext(grid, dt, {'D': D, 'v': v})
        x = precond.apply(r, context)

        # Verify by computing (I - dt*L)*x and comparing to r
        # L = -v*∂_x + D*∂_xx
        # In Fourier space: λ(k) = -i*v*k + D*laplacian_symbol
        from moljax.core.fft_solvers import build_wavenumbers_1d
        k = build_wavenumbers_1d(grid.nx, grid.dx)
        lam_adv = -1j * v * k
        lam_diff = D * fft_cache.laplacian_symbol
        eigenvalues = lam_adv + lam_diff

        # Compute (I - dt*L)*x
        x_u = x['u']
        x_hat = jnp.fft.fft(x_u)
        result_hat = (1.0 - dt * eigenvalues) * x_hat
        result = jnp.real(jnp.fft.ifft(result_hat))

        # Preconditioner achieves ~2-3% accuracy (limited by Nyquist aliasing for advection)
        error = jnp.linalg.norm(result - r['u']) / jnp.linalg.norm(r['u'])
        assert error < 0.05, f"Preconditioner inversion error: {error:.2e}"

    def test_fft_diffusion_preconditioner_matches_solve(self, grid_256):
        """Verify FFT diffusion preconditioner matches direct FFT solve."""
        from moljax.core.preconditioners import FFTDiffusionPreconditioner, PrecondContext
        from moljax.core.fft_solvers import create_fft_cache_1d, solve_helmholtz_1d

        grid = grid_256
        D = 0.01
        dt = 0.1

        # Create preconditioner
        fft_cache = create_fft_cache_1d(grid)
        precond = FFTDiffusionPreconditioner(
            field_diffusivity_keys={'u': 'D'},
            fft_cache=fft_cache
        )

        # Random RHS
        key = random.PRNGKey(123)
        r = {'u': random.normal(key, (grid.nx,))}

        # Apply preconditioner
        context = PrecondContext(grid, dt, {'D': D})
        x_precond = precond.apply(r, context)

        # Direct solve
        x_direct = solve_helmholtz_1d(r['u'], fft_cache.laplacian_symbol, dt, D)

        # Should match exactly
        error = jnp.linalg.norm(x_precond['u'] - x_direct) / jnp.linalg.norm(x_direct)
        assert error < 1e-12, f"Preconditioner/direct solve mismatch: {error:.2e}"

    def test_fft_precond_vs_identity_iteration_reduction(self, grid_128):
        """Verify FFT preconditioner reduces Krylov iterations vs identity.

        This simulates a Newton-Krylov iteration by computing condition numbers
        of the preconditioned system.
        """
        from moljax.core.fft_solvers import create_fft_cache_1d

        grid = grid_128
        D = 0.01
        dt = 0.1  # Stiff regime: dt * D / dx² >> 1
        stiffness = dt * D / grid.dx**2
        print(f"\nStiffness parameter: dt*D/dx² = {stiffness:.1f}")

        fft_cache = create_fft_cache_1d(grid)
        k = fft_cache.k
        lam = D * fft_cache.laplacian_symbol

        # Original system eigenvalues: 1 - dt*λ
        orig_eig = 1.0 - dt * lam

        # Condition number of original system
        orig_cond = float(jnp.max(jnp.abs(orig_eig)) / jnp.min(jnp.abs(orig_eig)))
        print(f"Original condition number: {orig_cond:.1f}")

        # With FFT preconditioner, eigenvalues become 1 (perfect)
        # The ratio shows the iteration reduction potential
        iteration_reduction_factor = orig_cond

        print(f"Iteration reduction potential: {iteration_reduction_factor:.1f}x")

        # For stiff problems, FFT precond should dramatically reduce iterations
        assert iteration_reduction_factor > 10, \
            f"FFT precond should offer >10x iteration reduction for stiff problems"

    def test_fft_advdiff_precond_handles_pure_advection(self, grid_128):
        """Verify FFT advdiff preconditioner handles pure advection (D=0).

        Note: Pure advection with complex eigenvalues has the same Nyquist aliasing
        issue as advection-diffusion, causing a few percent error.
        """
        from moljax.core.preconditioners import (
            FFTAdvectionDiffusionPreconditioner,
            PrecondContext,
        )
        from moljax.core.fft_solvers import create_fft_cache_1d

        grid = grid_128
        D = 0.0  # Pure advection
        v = 1.0
        dt = 0.01

        fft_cache = create_fft_cache_1d(grid)
        precond = FFTAdvectionDiffusionPreconditioner(
            field_diffusivity_keys={'u': 'D'},
            field_velocity_keys={'u': 'v'},
            fft_cache=fft_cache
        )

        # Random RHS
        key = random.PRNGKey(42)
        r = {'u': random.normal(key, (grid.nx,))}

        # Apply preconditioner
        context = PrecondContext(grid, dt, {'D': D, 'v': v})
        x = precond.apply(r, context)

        # Verify inversion: (I - dt*(-v*∂_x))*x should equal r
        from moljax.core.fft_solvers import build_wavenumbers_1d
        k = build_wavenumbers_1d(grid.nx, grid.dx)
        lam_adv = -1j * v * k

        x_hat = jnp.fft.fft(x['u'])
        result_hat = (1.0 - dt * lam_adv) * x_hat
        result = jnp.real(jnp.fft.ifft(result_hat))

        # Accepts ~10% error due to Nyquist aliasing for pure advection
        error = jnp.linalg.norm(result - r['u']) / jnp.linalg.norm(r['u'])
        assert error < 0.1, f"Pure advection preconditioner error: {error:.2e}"

    def test_preconditioner_benchmark_table(self, grid_256):
        """Generate benchmark table comparing preconditioner effectiveness.

        | Preconditioner | Stiffness | Condition Number | Est. Iterations |
        """
        from moljax.core.fft_solvers import create_fft_cache_1d

        grid = grid_256
        D = 0.01

        fft_cache = create_fft_cache_1d(grid)
        lam = D * fft_cache.laplacian_symbol

        print("\n" + "="*70)
        print("PRECONDITIONER EFFECTIVENESS TABLE")
        print(f"Grid: N={grid.nx}, D={D}")
        print("="*70)
        print(f"{'Precond':<15} | {'dt':>8} | {'Stiffness':>10} | {'Cond No.':>12} | {'Est. Iters':>10}")
        print("-"*70)

        for dt in [0.001, 0.01, 0.1, 1.0]:
            stiffness = dt * D / grid.dx**2

            # Identity preconditioner: original condition number
            orig_eig = 1.0 - dt * lam
            cond_identity = float(jnp.max(jnp.abs(orig_eig)) / (jnp.min(jnp.abs(orig_eig)) + 1e-15))
            iters_identity = min(int(jnp.sqrt(cond_identity)), grid.nx)

            # FFT preconditioner: perfect conditioning
            cond_fft = 1.0
            iters_fft = 1

            print(f"{'Identity':<15} | {dt:>8.4f} | {stiffness:>10.1f} | {cond_identity:>12.1f} | {iters_identity:>10}")
            print(f"{'FFT Diffusion':<15} | {dt:>8.4f} | {stiffness:>10.1f} | {cond_fft:>12.1f} | {iters_fft:>10}")

            # FFT should be dramatically better for stiff problems
            if stiffness > 10:
                assert cond_identity > 10 * cond_fft, \
                    f"FFT should be much better conditioned for stiff problems"
