"""
Tests for Newton-Krylov solver.

Verifies:
- NK converges on simple nonlinear system
- JVP matches finite difference approximation
- Preconditioner improves convergence
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The BDF2 order tests resolve second-order errors down to 1e-4 at
# newton_tol=1e-13, which needs float64.
jax.config.update("jax_enable_x64", True)

from moljax.core.bc import BCType, FieldBCSpec
from moljax.core.grid import Grid1D
from moljax.core.model import MOLModel
from moljax.core.newton_krylov import (
    NKParams,
    _jvp_matvec,
    create_bdf2_residual,
    newton_krylov_solve,
)
from moljax.core.operators import NonlinearOp
from moljax.core.preconditioners import BlockJacobiPreconditioner, IdentityPreconditioner


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

        from moljax.core.grid import Grid1D
        from moljax.core.preconditioners import PrecondContext

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

        from moljax.core.grid import Grid1D
        from moljax.core.preconditioners import PrecondContext

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

    def test_nk_lin_iters_is_the_krylov_budget(self):
        """lin_iters counts the GMRES budget, max_krylov_iters per Newton step.

        jax.scipy.sparse.linalg.gmres returns (x, info) with no iteration
        count, so the statistic is the budget that was made available, not
        the iterations spent. This pins the documented meaning.
        """
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': x['u'] ** 2 - 1.0}

        result = newton_krylov_solve(
            residual_fn=residual,
            x0={'u': jnp.array([0.5, 0.5, 0.5])},
            grid=grid,
            params={},
            nk_params=NKParams(newton_tol=1e-8, max_krylov_iters=7)
        )

        assert int(result.stats.lin_iters) == 7 * int(result.stats.newton_iters)

    def test_nk_converged_on_last_allowed_iteration(self):
        """The flag describes the returned iterate, not the one before it.

        x - 1 = 0 is solved exactly by the first Newton step, so with
        max_newton_iters=1 the returned x is the root and converged must
        be True with a residual norm at rounding level.
        """
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': x['u'] - 1.0}

        x0 = {'u': jnp.array([0.0, 0.0, 0.0])}

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(max_newton_iters=1, newton_tol=1e-10)
        )

        assert bool(result.stats.converged)
        assert int(result.stats.newton_iters) == 1
        assert float(result.stats.final_res_norm) < 1e-10
        assert jnp.allclose(result.solution['u'], 1.0, atol=1e-10)

    def test_nk_reported_residual_is_residual_of_solution(self):
        """final_res_norm equals ||F(x)|| at the returned x, converged or not."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': x['u'] ** 2 - 2.0}

        x0 = {'u': jnp.array([0.5, 3.0, 1.0])}

        for max_iters in (1, 2, 20):
            result = newton_krylov_solve(
                residual_fn=residual,
                x0=x0,
                grid=grid,
                params={},
                nk_params=NKParams(max_newton_iters=max_iters, newton_tol=1e-12)
            )
            true_norm = float(jnp.linalg.norm(residual(result.solution)['u']))
            reported = float(result.stats.final_res_norm)
            assert abs(reported - true_norm) <= 1e-12 * max(1.0, true_norm), \
                f"max_iters={max_iters}: reported {reported:.3e}, actual {true_norm:.3e}"
            assert bool(result.stats.converged) == (true_norm < 1e-12)

    def test_failed_line_search_does_not_increase_residual(self):
        """When no backtracked candidate satisfies the decrease test, the
        iterate must never move to a step with a larger residual than where
        it started.

        atan(x) from x0 = 10 is the classic Newton overshoot example: the
        full Newton step (and every halving tried within max_backtrack=3)
        lands further from the root than the start, so the line search
        finds nothing to accept. The old fallback applied the untried full
        step anyway (the same candidate the line search never even
        evaluated against the decrease test), which increased the residual
        from about 2.548 to about 2.708 (checked by direct reproduction).
        The fix keeps the best candidate actually tried, which here is none
        of them, so the iterate must stay unchanged.
        """
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def residual(x):
            return {'u': jnp.arctan(x['u'])}

        x0 = {'u': jnp.array([10.0, 10.0, 10.0])}
        r0_norm = float(jnp.linalg.norm(residual(x0)['u']))

        result = newton_krylov_solve(
            residual_fn=residual,
            x0=x0,
            grid=grid,
            params={},
            nk_params=NKParams(max_newton_iters=1, max_backtrack=3, newton_tol=1e-12)
        )

        r1_norm = float(result.stats.final_res_norm)
        assert r1_norm <= r0_norm + 1e-10, \
            f"line search increased the residual: {r0_norm:.6f} -> {r1_norm:.6f}"
        # No candidate improved on the start, so the iterate is unchanged.
        assert jnp.allclose(result.solution['u'], x0['u'])
        assert not bool(result.stats.converged)


def decay_model(nx: int = 4) -> MOLModel:
    """y' = -y on every grid point, periodic, so exp(-t) is the exact solution."""
    grid = Grid1D.uniform(nx, 0.0, 1.0)
    op = NonlinearOp(name="decay", apply=lambda s, g, t, p: {'y': -s['y']})
    return MOLModel(
        grid=grid,
        bc_spec={'y': FieldBCSpec(kind=BCType.PERIODIC)},
        params={'dtype': jnp.float64},
        nonlinear_ops=(op,)
    )


def bdf2_run(model: MOLModel, steps: list[float]) -> tuple[float, float]:
    """Advance y' = -y from t=0 with the given step sequence.

    The first step is taken from the exact solution so only BDF2 itself is
    measured. Returns (y at the end, end time).
    """
    n = model.grid.nx_total
    nk = NKParams(newton_tol=1e-13, max_newton_iters=20)

    @jax.jit
    def step(y, y_prev, t_new, dt, dt_prev):
        residual = create_bdf2_residual(model, y, y_prev, t_new, dt, dt_prev)
        guess = {'y': 2.0 * y['y'] - y_prev['y']}
        result = newton_krylov_solve(residual, guess, model.grid, model.params, nk_params=nk)
        return result.solution, result.stats.converged

    t = steps[0]
    y_prev = {'y': jnp.ones(n)}
    y = {'y': jnp.full(n, np.exp(-t))}
    dt_prev = steps[0]
    for dt in steps[1:]:
        y_new, converged = step(y, y_prev, t + dt, dt, dt_prev)
        assert bool(converged)
        y_prev, y = y, y_new
        t, dt_prev = t + dt, dt
    return float(y['y'][1]), t


def observed_orders(errors: list[float], ratio: float = 2.0) -> list[float]:
    return [np.log(errors[i] / errors[i + 1]) / np.log(ratio) for i in range(len(errors) - 1)]


class TestBDF2Residual:
    """create_bdf2_residual must give a second-order method."""

    def test_bdf2_order(self):
        """Constant steps 0.1, 0.05, 0.025 to t=1: consecutive orders at least 1.9."""
        model = decay_model()
        errors = []
        for dt in (0.1, 0.05, 0.025):
            n_steps = int(round(1.0 / dt))
            y_end, t_end = bdf2_run(model, [dt] * n_steps)
            assert abs(t_end - 1.0) < 1e-12
            errors.append(abs(y_end - np.exp(-1.0)))

        orders = observed_orders(errors)
        assert all(o >= 1.9 for o in orders), f"errors {errors}, orders {orders}"

    def test_bdf2_order_alternating_steps(self):
        """Steps alternate 0.1, 0.05 (then halved twice) to t=0.9: orders at least 1.9.

        The variable-step coefficients are exercised on every step since
        dt/dt_prev is 1/2 or 2, never 1.
        """
        model = decay_model()
        errors = []
        for scale in (1.0, 0.5, 0.25):
            n_pairs = int(round(0.9 / (0.15 * scale)))
            steps = [0.1 * scale, 0.05 * scale] * n_pairs
            y_end, t_end = bdf2_run(model, steps)
            assert abs(t_end - 0.9) < 1e-12
            errors.append(abs(y_end - np.exp(-0.9)))

        orders = observed_orders(errors)
        assert all(o >= 1.9 for o in orders), f"errors {errors}, orders {orders}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
