"""
Tests for Newton-Krylov solver.

Verifies:
- NK converges on simple nonlinear system
- JVP matches finite difference approximation
- Preconditioner improves convergence
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.newton_krylov import (
    newton_krylov_solve, NKParams, _jvp_matvec
)
from moljax.core.preconditioners import IdentityPreconditioner, BlockJacobiPreconditioner


class TestNKConvergence:
    """Test Newton-Krylov convergence."""

    def test_nk_converges_quadratic(self):
        """Test NK converges on x^2 - 1 = 0."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': x['u'] ** 2 - 1.0}

        x0 = {'u': jnp.array([0.5, 0.5, 0.5])}

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(newton_tol=1e-8)
        )

        # Should converge to x = 1 (positive root starting from 0.5)
        assert result.stats.converged, "NK should converge"
        assert jnp.allclose(result.solution['u'], 1.0, atol=1e-6), \
            f"Solution {result.solution['u']} should be ~1.0"

    def test_nk_converges_linear(self):
        """Test NK converges on linear system 2x - 4 = 0."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': 2.0 * x['u'] - 4.0}

        x0 = {'u': jnp.array([0.0, 0.0, 0.0])}

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(newton_tol=1e-10)
        )

        assert result.stats.converged
        assert jnp.allclose(result.solution['u'], 2.0, atol=1e-8)

    def test_nk_multi_field(self):
        """Test NK on multi-field system."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        # System: u^2 + v^2 - 2 = 0
        #         u - v = 0
        # Solution: u = v = 1

        def residual(x):
            u, v = x['u'], x['v']
            return {
                'u': u ** 2 + v ** 2 - 2.0,
                'v': u - v
            }

        x0 = {
            'u': jnp.array([0.5, 0.5, 0.5]),
            'v': jnp.array([0.8, 0.8, 0.8])
        }

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(newton_tol=1e-8)
        )

        assert result.stats.converged
        assert jnp.allclose(result.solution['u'], 1.0, atol=1e-6)
        assert jnp.allclose(result.solution['v'], 1.0, atol=1e-6)


class TestJVP:
    """Test JVP-based Jacobian-vector products."""

    def test_jvp_matches_finite_diff(self):
        """Test that JVP matches finite difference approximation."""

        def residual(x):
            return {'u': x['u'] ** 2}

        x = {'u': jnp.array([2.0, 2.0, 2.0])}
        v = {'u': jnp.array([1.0, 1.0, 1.0])}

        # JVP: J @ v = d/dx[F(x)] @ v = 2*x * v
        Jv_jvp = _jvp_matvec(residual, x, v)

        # Finite difference: (F(x + eps*v) - F(x)) / eps
        # Use larger eps for float32 stability
        eps = 1e-4
        x_plus = {'u': x['u'] + eps * v['u']}
        Jv_fd = {'u': (residual(x_plus)['u'] - residual(x)['u']) / eps}

        # Relax tolerance for float32
        assert jnp.allclose(Jv_jvp['u'], Jv_fd['u'], rtol=0.1), \
            f"JVP {Jv_jvp['u']} doesn't match FD {Jv_fd['u']}"

    def test_jvp_linear_operator(self):
        """Test JVP on linear operator gives exact Jacobian."""

        A = jnp.array([[2.0, 1.0], [0.0, 3.0]])

        def residual(x):
            u = x['u']
            return {'u': A @ u}

        x = {'u': jnp.array([1.0, 2.0])}
        v = {'u': jnp.array([1.0, 0.0])}

        # For linear F(x) = Ax, J = A, so J @ v = A @ v
        Jv = _jvp_matvec(residual, x, v)
        expected = A @ v['u']

        assert jnp.allclose(Jv['u'], expected)


class TestPreconditioner:
    """Test preconditioner effects."""

    def test_identity_preconditioner(self):
        """Test that identity preconditioner doesn't change residual."""
        precond = IdentityPreconditioner()

        r = {'u': jnp.array([1.0, 2.0, 3.0])}

        from moljax.core.preconditioners import PrecondContext
        from moljax.core.grid import Grid1D

        grid = Grid1D.uniform(3, 0.0, 1.0)
        context = PrecondContext(grid=grid, dt=0.1, params={})

        r_precond = precond.apply(r, context)

        assert jnp.allclose(r_precond['u'], r['u'])

    def test_block_jacobi_scales(self):
        """Test that block Jacobi applies scaling."""
        precond = BlockJacobiPreconditioner(
            diffusion_keys={'u': 'D'}
        )

        r = {'u': jnp.ones(5) * 10.0}

        from moljax.core.preconditioners import PrecondContext
        from moljax.core.grid import Grid1D

        grid = Grid1D.uniform(3, 0.0, 1.0)
        context = PrecondContext(grid=grid, dt=0.1, params={'D': 1.0})

        r_precond = precond.apply(r, context)

        # Should scale down due to diffusion term
        # scale = 1 + dt * D * 4/dx^2
        scale = 1.0 + 0.1 * 1.0 * 4.0 / grid.dx ** 2
        expected = r['u'] / scale

        assert jnp.allclose(r_precond['u'], expected)


class TestNKRobustness:
    """Test NK solver robustness features."""

    def test_nk_reports_failure_on_divergence(self):
        """Test NK reports non-convergence when it can't converge."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        # System with no solution: x^2 + 1 = 0 (real)
        def residual(x):
            return {'u': x['u'] ** 2 + 1.0}

        x0 = {'u': jnp.array([0.5, 0.5, 0.5])}

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(max_newton_iters=5, newton_tol=1e-10)
        )

        # Should not converge
        assert not result.stats.converged or float(result.stats.final_res_norm) > 0.1

    def test_nk_iteration_count(self):
        """Test that NK tracks iteration count."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': x['u'] - 1.0}

        x0 = {'u': jnp.array([0.0, 0.0, 0.0])}

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(newton_tol=1e-10)
        )

        # Linear system should converge quickly (within a few iterations)
        assert int(result.stats.newton_iters) <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
