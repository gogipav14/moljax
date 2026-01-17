"""
Tests for time integrators.

Verifies:
- RK4 achieves 4th order convergence on scalar ODE
- BE is stable for large dt on stiff problems
- Explicit methods blow up at large dt for diffusion
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.bc import FieldBCSpec, BCType
from moljax.core.model import MOLModel, create_gray_scott_model
from moljax.core.operators import LinearOp
from moljax.core.stepping import (
    IntegratorType, euler_step, ssprk3_step, rk4_step,
    be_step, cn_step, step_explicit, adaptive_integrate
)
from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.utils import get_interior


class TestExplicitIntegrators:
    """Tests for explicit integrators."""

    def test_euler_linear_decay(self):
        """Test Euler on du/dt = -u (exponential decay)."""
        # Simple ODE model
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def decay_rhs(state, grid, t, params):
            return {'u': -state['u']}

        decay_op = LinearOp(name="decay", apply=decay_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(decay_op,),
            nonlinear_ops=()
        )

        # Initial condition
        y0 = {'u': jnp.array([1.0, 1.0, 1.0])}  # padded

        # Integrate
        dt = 0.1
        y = y0
        for _ in range(10):
            y = euler_step(model, y, 0.0, dt)

        # Expected: u(1) = exp(-1) ~ 0.368
        u_final = float(y['u'][1])
        expected = np.exp(-1.0)

        # Euler is first order, so ~10% error is expected
        assert abs(u_final - expected) < 0.1, f"Euler error: {abs(u_final - expected)}"

    def test_rk4_order_convergence(self):
        """Test RK4 achieves 4th order convergence."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def decay_rhs(state, grid, t, params):
            return {'u': -state['u']}

        decay_op = LinearOp(name="decay", apply=decay_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(decay_op,),
            nonlinear_ops=()
        )

        errors = []
        dts = [0.2, 0.1, 0.05]
        t_end = 1.0
        expected = np.exp(-t_end)

        for dt in dts:
            y = {'u': jnp.array([1.0, 1.0, 1.0])}
            n_steps = int(t_end / dt)

            for _ in range(n_steps):
                y = rk4_step(model, y, 0.0, dt)

            error = abs(float(y['u'][1]) - expected)
            errors.append(error)

        # Check convergence rate ~ 4
        rate1 = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])
        rate2 = np.log(errors[1] / errors[2]) / np.log(dts[1] / dts[2])

        assert rate1 > 3.5, f"RK4 convergence rate too low: {rate1}"
        assert rate2 > 3.5, f"RK4 convergence rate too low: {rate2}"

    def test_ssprk3_stability_advection(self):
        """Test SSPRK3 is stable for advection with CFL dt."""
        nx = 50
        grid = Grid1D.uniform(nx, 0.0, 1.0)

        # Advection: du/dt = -v * du/dx
        v = 1.0

        def advection_rhs(state, grid, t, params):
            from moljax.core.operators import d1_upwind_1d
            from moljax.core.bc import apply_bc
            state = apply_bc(state, grid, {'u': FieldBCSpec.periodic()})
            du_dx = d1_upwind_1d(state['u'], v, grid)
            return {'u': -v * du_dx}

        advection_op = LinearOp(name="advection", apply=advection_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(advection_op,),
            nonlinear_ops=()
        )

        # Initial condition: smooth bump
        x = grid.x_coords(include_ghost=True)
        u0 = jnp.exp(-100 * (x - 0.5) ** 2)
        y = {'u': u0}

        # CFL dt
        dt = 0.5 * grid.dx / v

        # Integrate for one pass across domain
        n_steps = int(1.0 / (v * dt))
        for _ in range(n_steps):
            y = ssprk3_step(model, y, 0.0, dt)

        # Should not blow up
        assert jnp.all(jnp.isfinite(y['u'])), "SSPRK3 produced non-finite values"
        assert float(jnp.max(jnp.abs(y['u']))) < 2.0, "SSPRK3 solution unstable"


class TestImplicitIntegrators:
    """Tests for implicit integrators."""

    def test_be_stable_large_dt(self):
        """Test BE is stable with large dt on stiff problem."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        # Stiff decay: du/dt = -100*u
        lambda_stiff = 100.0

        def stiff_rhs(state, grid, t, params):
            return {'u': -lambda_stiff * state['u']}

        stiff_op = LinearOp(name="stiff", apply=stiff_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(stiff_op,),
            nonlinear_ops=()
        )

        y = {'u': jnp.array([1.0, 1.0, 1.0])}

        # Large dt that would be unstable for explicit
        dt = 0.1  # >> 2/lambda for explicit stability

        # BE step
        y_new, stats = be_step(model, y, 0.0, dt)

        # Should not blow up
        assert jnp.all(jnp.isfinite(y_new['u'])), "BE produced non-finite values"
        # Solution should decay
        assert float(y_new['u'][1]) < float(y['u'][1]), "BE solution should decay"

    def test_cn_second_order(self):
        """Test CN achieves second order convergence."""
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def decay_rhs(state, grid, t, params):
            return {'u': -state['u']}

        decay_op = LinearOp(name="decay", apply=decay_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(decay_op,),
            nonlinear_ops=()
        )

        errors = []
        dts = [0.2, 0.1, 0.05]
        t_end = 1.0
        expected = np.exp(-t_end)

        for dt in dts:
            y = {'u': jnp.array([1.0, 1.0, 1.0])}
            n_steps = int(t_end / dt)

            for i in range(n_steps):
                y, _ = cn_step(model, y, i * dt, dt)

            error = abs(float(y['u'][1]) - expected)
            errors.append(error)

        # Check convergence rate ~ 2
        rate1 = np.log(errors[0] / errors[1]) / np.log(dts[0] / dts[1])

        assert rate1 > 1.8, f"CN convergence rate too low: {rate1}"


class TestStability:
    """Test stability properties of integrators."""

    def test_explicit_unstable_large_dt_diffusion(self):
        """Test that explicit blows up with dt above diffusion CFL."""
        nx = 20
        grid = Grid1D.uniform(nx, 0.0, 1.0)
        D = 1.0

        def diffusion_rhs(state, grid, t, params):
            from moljax.core.operators import laplacian_1d
            from moljax.core.bc import apply_bc
            state = apply_bc(state, grid, {'u': FieldBCSpec.periodic()})
            return {'u': D * laplacian_1d(state['u'], grid)}

        diffusion_op = LinearOp(name="diffusion", apply=diffusion_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(diffusion_op,),
            nonlinear_ops=()
        )

        # Initial condition
        x = grid.x_coords(include_ghost=True)
        u0 = jnp.sin(2 * jnp.pi * x)
        y = {'u': u0}

        # dt above stability limit: dt > 0.5 * dx^2 / D
        dt_stable = 0.5 * grid.dx ** 2 / D
        dt_unstable = 10 * dt_stable  # Well above limit

        # Run many steps with unstable dt
        for _ in range(50):
            y = euler_step(model, y, 0.0, dt_unstable)

        # Should blow up (NaN or very large values)
        is_blown = (
            not jnp.all(jnp.isfinite(y['u'])) or
            float(jnp.max(jnp.abs(y['u']))) > 1e5
        )
        assert is_blown, "Explicit should be unstable with large dt"


class TestAdaptive:
    """Test adaptive integration."""

    def test_adaptive_completes(self):
        """Test that adaptive integration completes successfully."""
        nx = 20
        grid = Grid1D.uniform(nx, 0.0, 1.0)

        def diffusion_rhs(state, grid, t, params):
            from moljax.core.operators import laplacian_1d
            from moljax.core.bc import apply_bc
            state = apply_bc(state, grid, {'u': FieldBCSpec.periodic()})
            D = params.get('D', 0.1)
            return {'u': D * laplacian_1d(state['u'], grid)}

        diffusion_op = LinearOp(name="diffusion", apply=diffusion_rhs)

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'D': 0.1, 'dtype': jnp.float64},
            linear_ops=(diffusion_op,),
            nonlinear_ops=()
        )

        # Initial condition
        x = grid.x_coords(include_ghost=True)
        y0 = {'u': jnp.sin(2 * jnp.pi * x)}

        cfl_params = CFLParams(cfl_diffusion=0.25, dt_max=0.1)
        pid_params = PIDParams(atol=1e-4, rtol=1e-3, dt_max=0.1)

        result = adaptive_integrate(
            model=model,
            y0=y0,
            t0=0.0,
            t_end=0.1,
            dt0=0.001,
            method=IntegratorType.RK4,
            max_steps=1000,
            cfl_params=cfl_params,
            pid_params=pid_params
        )

        # Should complete successfully
        from moljax.core.utils import StatusCode
        assert int(result.status) == StatusCode.SUCCESS, f"Status: {int(result.status)}"
        assert float(result.t_final) >= 0.1 - 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
