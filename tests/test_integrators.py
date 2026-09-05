"""
Tests for time integrators.

Verifies:
- RK4 achieves 4th order convergence on scalar ODE
- BE is stable for large dt on stiff problems
- Explicit methods blow up at large dt for diffusion
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The fixed-step tests compare a compiled loop with an eager one to 1e-14,
# which needs float64.
jax.config.update("jax_enable_x64", True)

from moljax.core.bc import FieldBCSpec
from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.model import MOLModel, create_gray_scott_periodic_fft
from moljax.core.newton_krylov import NKParams
from moljax.core.operators import LinearOp
from moljax.core.stepping import (
    IntegratorType,
    adaptive_integrate,
    bdf2_step,
    be_step,
    cn_step,
    euler_step,
    imex_strang_step,
    integrate_fixed_dt,
    integrate_imex_fixed_dt,
    rk4_step,
    ssprk3_step,
)


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
            from moljax.core.bc import apply_bc
            from moljax.core.operators import d1_upwind_1d
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
            from moljax.core.bc import apply_bc
            from moljax.core.operators import laplacian_1d
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
            from moljax.core.bc import apply_bc
            from moljax.core.operators import laplacian_1d
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


def gray_scott_off_equilibrium():
    """Small periodic Gray-Scott model with a Gaussian dip in u and a bump in v.

    The reaction term is nonzero everywhere, so a wrong step count or a
    skipped step changes the answer visibly.
    """
    grid = Grid2D.uniform(8, 8, 0.0, 2.5, 0.0, 2.5)
    model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

    def bump(X, Y):
        return jnp.exp(-((X - 1.25) ** 2 + (Y - 1.25) ** 2))

    state = model.create_initial_state(init_fns={
        'u': lambda X, Y: 1.0 - 0.5 * bump(X, Y),
        'v': lambda X, Y: 0.25 * bump(X, Y),
    })
    return model, fft_cache, diffusivities, state


def assert_states_match(actual, expected, tol=1e-14):
    for name in expected:
        diff = float(jnp.max(jnp.abs(actual[name] - expected[name])))
        assert diff < tol, f"field {name}: max difference {diff:.3e}"


def assert_history_layout(t_hist, y_hist, y_final, grid, t_end, n_expected):
    """The history holds one entry per save_every steps, interior only, ending at t_end."""
    assert t_hist.shape == (n_expected,)
    assert abs(float(t_hist[-1]) - t_end) < 1e-12
    sl_y, sl_x = grid.interior_slice
    for name, field in y_final.items():
        assert y_hist[name].shape == (n_expected, grid.ny, grid.nx)
        assert float(jnp.max(jnp.abs(y_hist[name][-1] - field[sl_y, sl_x]))) < 1e-14


class TestFixedStep:
    """integrate_fixed_dt and integrate_imex_fixed_dt take exactly (t_end - t0)/dt steps.

    Each run is compared with the same number of eager single steps. With
    t_end=0.5, dt=0.05 and save_every=5 the history must hold two entries,
    at t=0.25 and t=0.5.
    """

    T_END, DT, SAVE_EVERY, N_STEPS = 0.5, 0.05, 5, 10
    NK = NKParams(newton_tol=1e-12)

    def test_rk4_matches_manual_loop(self):
        model, _, _, y0 = gray_scott_off_equilibrium()
        y = y0
        for i in range(self.N_STEPS):
            y = rk4_step(model, y, i * self.DT, self.DT)

        t_hist, y_hist, y_final = integrate_fixed_dt(
            model, y0, 0.0, self.T_END, self.DT,
            method=IntegratorType.RK4, save_every=self.SAVE_EVERY
        )
        assert_states_match(y_final, y)
        assert_history_layout(t_hist, y_hist, y_final, model.grid, self.T_END, 2)
        assert abs(float(t_hist[0]) - 0.25) < 1e-12

    def test_be_matches_manual_loop(self):
        model, _, _, y0 = gray_scott_off_equilibrium()
        y = y0
        for i in range(self.N_STEPS):
            y, _ = be_step(model, y, i * self.DT, self.DT, nk_params=self.NK)

        t_hist, y_hist, y_final = integrate_fixed_dt(
            model, y0, 0.0, self.T_END, self.DT,
            method=IntegratorType.BE, save_every=self.SAVE_EVERY, nk_params=self.NK
        )
        assert_states_match(y_final, y)
        assert_history_layout(t_hist, y_hist, y_final, model.grid, self.T_END, 2)

    def test_bdf2_matches_manual_loop(self):
        model, _, _, y0 = gray_scott_off_equilibrium()
        # BE start, then BDF2 with a constant step.
        y_prev, y = y0, be_step(model, y0, 0.0, self.DT, nk_params=self.NK)[0]
        for i in range(1, self.N_STEPS):
            y_new, _ = bdf2_step(model, y, y_prev, i * self.DT, self.DT, self.DT, nk_params=self.NK)
            y_prev, y = y, y_new

        t_hist, y_hist, y_final = integrate_fixed_dt(
            model, y0, 0.0, self.T_END, self.DT,
            method=IntegratorType.BDF2, save_every=self.SAVE_EVERY, nk_params=self.NK
        )
        assert_states_match(y_final, y)
        assert_history_layout(t_hist, y_hist, y_final, model.grid, self.T_END, 2)

    def test_imex_strang_matches_manual_loop(self):
        model, fft_cache, diffusivities, y0 = gray_scott_off_equilibrium()
        y = y0
        for i in range(self.N_STEPS):
            y = imex_strang_step(model, y, i * self.DT, self.DT, fft_cache, diffusivities)

        t_hist, y_hist, y_final = integrate_imex_fixed_dt(
            model, y0, 0.0, self.T_END, self.DT, fft_cache, diffusivities,
            use_strang=True, save_every=self.SAVE_EVERY
        )
        assert_states_match(y_final, y)
        assert_history_layout(t_hist, y_hist, y_final, model.grid, self.T_END, 2)

    def test_dt_must_divide_interval(self):
        model, fft_cache, diffusivities, y0 = gray_scott_off_equilibrium()
        with pytest.raises(ValueError):
            integrate_fixed_dt(model, y0, 0.0, self.T_END, 0.03, method=IntegratorType.RK4)
        with pytest.raises(ValueError):
            integrate_imex_fixed_dt(model, y0, 0.0, self.T_END, 0.03, fft_cache, diffusivities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
