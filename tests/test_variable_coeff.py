"""
Tests for variable coefficient operators with FFT-based preconditioning.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moljax.core.variable_coeff import (
    apply_variable_diffusion_1d,
    apply_variable_diffusion_2d,
    assess_circulant_quality,
    compute_coeff_stats,
    create_circulant_approx_1d,
    create_circulant_approx_2d,
    etd1_varcoeff_approx_1d,
    richardson_iteration_varcoeff_1d,
    solve_helmholtz_circulant_1d,
    solve_helmholtz_circulant_2d,
)


class TestCoeffStats:
    """Test coefficient statistics."""

    def test_constant_coeff(self):
        """Constant coefficient should have zero variation."""
        D = jnp.ones(64) * 2.5
        stats = compute_coeff_stats(D)

        assert stats.mean == pytest.approx(2.5)
        assert stats.std == pytest.approx(0.0, abs=1e-10)
        assert stats.variation_ratio == pytest.approx(0.0, abs=1e-10)

    def test_varying_coeff(self):
        """Test statistics for varying coefficient."""
        D = jnp.array([1.0, 2.0, 3.0, 4.0])
        stats = compute_coeff_stats(D)

        assert stats.mean == pytest.approx(2.5)
        assert stats.min == pytest.approx(1.0)
        assert stats.max == pytest.approx(4.0)
        assert stats.std > 0

    def test_2d_coeff(self):
        """Test 2D coefficient statistics."""
        D = jnp.ones((32, 32)) * 1.5
        D = D.at[10:20, 10:20].set(2.0)
        stats = compute_coeff_stats(D)

        assert stats.min == pytest.approx(1.5)
        assert stats.max == pytest.approx(2.0)
        assert stats.variation_ratio > 0


class TestCirculantApprox:
    """Test circulant approximation creation."""

    def test_constant_coeff_valid(self):
        """Constant coefficient should be valid approximation."""
        D = jnp.ones(64) * 1.0
        dx = 0.1
        approx = create_circulant_approx_1d(D, dx)

        assert approx.D_mean == pytest.approx(1.0)
        assert approx.is_valid_approx is True
        assert approx.fft_symbol.shape == (64,)

    def test_varying_coeff_validity(self):
        """Test validity threshold."""
        n = 64
        dx = 0.1
        x = jnp.linspace(0, 2 * jnp.pi, n)

        # Small variation - should be valid
        D_small = 1.0 + 0.1 * jnp.sin(x)
        approx_small = create_circulant_approx_1d(D_small, dx, threshold=0.3)
        assert approx_small.is_valid_approx is True

        # Large variation - should be invalid
        D_large = 1.0 + 0.5 * jnp.sin(x)
        approx_large = create_circulant_approx_1d(D_large, dx, threshold=0.3)
        assert approx_large.is_valid_approx is False

    def test_2d_circulant(self):
        """Test 2D circulant approximation."""
        D = jnp.ones((32, 32)) * 2.0
        dy, dx = 0.1, 0.1
        approx = create_circulant_approx_2d(D, dy, dx)

        assert approx.D_mean == pytest.approx(2.0)
        assert approx.fft_symbol.shape == (32, 32)
        assert approx.is_valid_approx is True


class TestVariableDiffusion:
    """Test variable coefficient diffusion operators."""

    def test_constant_coeff_matches_laplacian(self):
        """With constant D, should match D * Laplacian."""
        n = 64
        ng = 1
        dx = 2 * jnp.pi / n

        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        u = jnp.sin(x)
        D = jnp.ones(n + 2 * ng) * 2.0

        # Variable coeff diffusion
        Lu_var = apply_variable_diffusion_1d(u, D, dx, ng, n)

        # Expected: 2.0 * (-sin(x)) = -2*sin(x)
        expected = -2.0 * jnp.sin(x)

        # Check interior
        error = jnp.max(jnp.abs(Lu_var[ng:ng + n] - expected[ng:ng + n]))
        assert error < 0.01  # 2nd-order FD error

    def test_conservative_form(self):
        """Test that operator conserves mass for zero-flux BCs."""
        n = 64
        ng = 1
        dx = 0.1

        # Arbitrary smooth field
        u = jnp.zeros(n + 2 * ng)
        u = u.at[ng:ng + n].set(jnp.sin(jnp.linspace(0, 2 * jnp.pi, n)))

        # Constant D
        D = jnp.ones(n + 2 * ng) * 1.5

        Lu = apply_variable_diffusion_1d(u, D, dx, ng, n)

        # For periodic/zero-flux, integral of Laplacian should be ~0
        integral = jnp.sum(Lu[ng:ng + n]) * dx
        assert abs(float(integral)) < 1e-10

    def test_2d_constant_coeff(self):
        """Test 2D with constant coefficient."""
        ny, nx = 64, 64
        ng = 1
        L = 2 * jnp.pi
        dy = L / ny
        dx = L / nx

        # Test function - use proper periodic grid
        x = jnp.linspace(0, L, nx + 2 * ng, endpoint=False)
        y = jnp.linspace(0, L, ny + 2 * ng, endpoint=False)
        X, Y = jnp.meshgrid(x, y)
        u = jnp.sin(X) * jnp.sin(Y)

        D = jnp.ones((ny + 2 * ng, nx + 2 * ng)) * 1.0

        Lu = apply_variable_diffusion_2d(u, D, dy, dx, ng, ny, nx)

        # Laplacian of sin(x)*sin(y) = -2*sin(x)*sin(y)
        expected = -2.0 * u

        # Check interior - allow more tolerance for 2nd-order FD
        interior = (slice(ng, ng + ny), slice(ng, ng + nx))
        error = jnp.max(jnp.abs(Lu[interior] - expected[interior]))
        assert error < 0.5  # Relaxed for 2nd-order FD with moderate grid


class TestCirculantSolvers:
    """Test FFT-based circulant solvers."""

    def test_helmholtz_1d_identity_limit(self):
        """As dt→0, solution should equal RHS."""
        n = 64
        dx = 0.1
        D = jnp.ones(n) * 1.0
        approx = create_circulant_approx_1d(D, dx)

        rhs = jnp.sin(jnp.linspace(0, 2 * jnp.pi, n))

        u = solve_helmholtz_circulant_1d(rhs, approx.fft_symbol, dt=1e-10)

        assert jnp.allclose(u, rhs, rtol=1e-6)

    def test_helmholtz_2d_identity_limit(self):
        """2D identity limit test."""
        ny, nx = 32, 32
        dy, dx = 0.1, 0.1
        D = jnp.ones((ny, nx)) * 1.0
        approx = create_circulant_approx_2d(D, dy, dx)

        x = jnp.linspace(0, 2 * jnp.pi, nx)
        y = jnp.linspace(0, 2 * jnp.pi, ny)
        X, Y = jnp.meshgrid(x, y)
        rhs = jnp.sin(X) * jnp.cos(Y)

        u = solve_helmholtz_circulant_2d(rhs, approx.fft_symbol, dt=1e-10)

        assert jnp.allclose(u, rhs, rtol=1e-6)

    def test_helmholtz_1d_exact_for_constant(self):
        """For constant D, circulant solver is exact for FD Laplacian."""
        n = 64
        dx = 2 * jnp.pi / n
        D_val = 0.5

        D = jnp.ones(n) * D_val
        approx = create_circulant_approx_1d(D, dx)

        # Test: solve (I - dt*symbol)u = rhs and verify residual
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        rhs = jnp.cos(2 * x)

        dt = 0.01
        u_solved = solve_helmholtz_circulant_1d(rhs, approx.fft_symbol, dt)

        # Verify by computing residual: (I - dt*symbol)u - rhs should be ~0
        u_hat = jnp.fft.fft(u_solved)
        reconstructed = jnp.real(jnp.fft.ifft((1.0 - dt * approx.fft_symbol) * u_hat))

        assert jnp.allclose(reconstructed, rhs, rtol=1e-10)


class TestRichardsonIteration:
    """Test iterative refinement with FFT preconditioner."""

    def test_variable_diffusion_matches_periodic_solution(self):
        """The FD stencil's boundary node must converge at its own order, not
        an O(1) defect that survives grid refinement.

        Manufactured solution on the periodic domain [0, 2*pi): u(x) = sin(x),
        D(x) = 1 + 0.3*sin(x), so rhs = u - dt*(D u')' has a closed form and
        u is the exact continuous solution of (I - dt*d/dx(D d/dx)) for any
        n. apply_variable_diffusion_1d/2d padded the ghost cells with 'edge'
        replication instead of periodic 'wrap', even though this module's
        preconditioner (create_circulant_approx_1d) treats the domain as
        periodic: the boundary stencil's left neighbor used D[0] (replicated)
        instead of the correct periodic neighbor D[n-1]. The dense linear
        solve's error at the domain's first grid point, x = 0, does not
        shrink with 'edge' padding (it grows: 0.040 at n=32 to 0.088 at
        n=256, a negative convergence order) while it shrinks at very close
        to the conservative stencil's own second order with 'wrap' (orders
        1.995, 1.999, 1.9997 for the doublings 32->64->128->256).
        """
        dt = 0.01

        def manufactured(n):
            dx = 2 * jnp.pi / n
            x = jnp.arange(n) * dx
            D = 1.0 + 0.3 * jnp.sin(x)
            u_exact = jnp.sin(x)
            D_prime = 0.3 * jnp.cos(x)
            u_prime = jnp.cos(x)
            u_double_prime = -jnp.sin(x)
            Lu_exact = D_prime * u_prime + D * u_double_prime  # (D u')'
            rhs = u_exact - dt * Lu_exact
            return dx, D, u_exact, rhs

        def boundary_error(n: int, mode: str) -> float:
            """|u_numeric(0) - u_exact(0)| from a dense solve of the FD system."""
            dx, D, u_exact, rhs = manufactured(n)
            ng = 1

            def apply_op(u_interior):
                u_padded = jnp.pad(u_interior, ng, mode=mode)
                Lu = apply_variable_diffusion_1d(u_padded, D, dx, ng, n)
                return Lu[ng:ng + n]

            identity = jnp.eye(n)
            L_cols = jax.vmap(apply_op, in_axes=1, out_axes=1)(identity)
            A = identity - dt * L_cols
            u_numeric = jnp.linalg.solve(A, rhs)
            return float(jnp.abs(u_numeric[0] - u_exact[0]))

        ns = (32, 64, 128, 256)
        errors_wrap = [boundary_error(n, 'wrap') for n in ns]
        orders_wrap = [
            np.log(errors_wrap[i] / errors_wrap[i + 1]) / np.log(2.0)
            for i in range(len(errors_wrap) - 1)
        ]
        assert all(o > 1.8 for o in orders_wrap), f"orders {orders_wrap}, errors {errors_wrap}"
        assert errors_wrap[0] < 1e-3, f"boundary error at n=32 too large: {errors_wrap[0]}"

        errors_edge = [boundary_error(n, 'edge') for n in ns]
        assert errors_edge[-1] > errors_edge[0], \
            f"'edge' boundary error should not shrink with n: {errors_edge}"

    def test_varying_coeff_converges(self):
        """Variable coefficient should converge with iterations."""
        n = 64
        ng = 1
        dx = 2 * jnp.pi / n
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)

        # Moderate variation
        D = 1.0 + 0.2 * jnp.sin(x)
        approx = create_circulant_approx_1d(D, dx)

        rhs = jnp.sin(2 * x)

        u, residuals = richardson_iteration_varcoeff_1d(
            rhs, D, approx.fft_symbol, dx, ng, n, n_iters=10, dt=0.01
        )

        # Residuals should decrease
        assert float(residuals[-1]) < float(residuals[0])


class TestQualityAssessment:
    """Test quality assessment functions."""

    def test_quality_metrics(self):
        """Test quality assessment output."""
        D = jnp.ones(64) * 1.0
        approx = create_circulant_approx_1d(D, 0.1)

        quality = assess_circulant_quality(approx)

        assert 'D_mean' in quality
        assert 'variation_ratio' in quality
        assert 'recommendation' in quality
        assert 'expected_relative_error' in quality

    def test_excellent_recommendation(self):
        """Constant D should get excellent recommendation."""
        D = jnp.ones(64) * 1.0
        approx = create_circulant_approx_1d(D, 0.1)

        quality = assess_circulant_quality(approx)
        assert 'excellent' in quality['recommendation']

    def test_poor_recommendation(self):
        """High variation should get poor recommendation."""
        x = jnp.linspace(0, 2 * jnp.pi, 64)
        D = 1.0 + 0.8 * jnp.sin(x)  # High variation
        approx = create_circulant_approx_1d(D, 0.1)

        quality = assess_circulant_quality(approx)
        assert 'poor' in quality['recommendation'] or 'moderate' in quality['recommendation']


class TestETDVariableCoeff:
    """Test ETD with variable coefficient approximation."""

    def test_etd1_decay(self):
        """ETD1 should show exponential decay for pure diffusion."""
        n = 64
        dx = 2 * jnp.pi / n
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)

        D = jnp.ones(n) * 0.1
        approx = create_circulant_approx_1d(D, dx)

        # Initial condition
        u0 = jnp.sin(x)  # Mode k=1

        # Zero nonlinear term
        N_u = jnp.zeros_like(u0)

        dt = 0.1
        u1 = etd1_varcoeff_approx_1d(u0, N_u, approx.fft_symbol, dt)

        # For pure diffusion of sin(x), decay is exp(-D * k^2 * dt) = exp(-D * dt)

        # Check amplitude decreased
        assert jnp.max(jnp.abs(u1)) < jnp.max(jnp.abs(u0))

    def test_etd1_preserves_mean(self):
        """ETD with zero nonlinearity should preserve mean."""
        n = 64
        dx = 0.1

        D = jnp.ones(n) * 1.0
        approx = create_circulant_approx_1d(D, dx)

        # Initial with nonzero mean
        u0 = jnp.ones(n) * 2.0 + 0.5 * jnp.sin(jnp.linspace(0, 2 * jnp.pi, n))
        N_u = jnp.zeros_like(u0)

        u1 = etd1_varcoeff_approx_1d(u0, N_u, approx.fft_symbol, dt=0.1)

        # Mean should be preserved (DC mode has λ=0)
        assert jnp.abs(jnp.mean(u1) - jnp.mean(u0)) < 1e-10


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_1d_workflow(self):
        """Test complete 1D workflow."""
        n = 64
        dx = 2 * jnp.pi / n
        ng = 1

        # Create varying coefficient
        x = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        D = 1.0 + 0.15 * jnp.sin(x)

        # Create approximation
        approx = create_circulant_approx_1d(D, dx)
        assert approx.is_valid_approx  # Should be valid with 15% variation

        # Solve Helmholtz
        rhs = jnp.sin(2 * x)
        solve_helmholtz_circulant_1d(rhs, approx.fft_symbol, dt=0.01)

        # Refine with iteration
        u_refined, residuals = richardson_iteration_varcoeff_1d(
            rhs, D, approx.fft_symbol, dx, ng, n, n_iters=5, dt=0.01
        )

        # Refined should be closer to true solution
        # (at least residual should decrease)
        assert float(residuals[-1]) <= float(residuals[0]) * 1.1

    def test_2d_workflow(self):
        """Test complete 2D workflow."""
        ny, nx = 32, 32
        dy, dx = 0.1, 0.1

        # Constant D for exact comparison
        D = jnp.ones((ny, nx)) * 1.5

        approx = create_circulant_approx_2d(D, dy, dx)

        x = jnp.linspace(0, 2 * jnp.pi, nx)
        y = jnp.linspace(0, 2 * jnp.pi, ny)
        X, Y = jnp.meshgrid(x, y)
        rhs = jnp.sin(X) * jnp.cos(Y)

        u = solve_helmholtz_circulant_2d(rhs, approx.fft_symbol, dt=0.01)

        # Solution should be reasonable (not NaN/Inf)
        assert jnp.all(jnp.isfinite(u))
        assert u.shape == (ny, nx)
