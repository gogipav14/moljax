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
from moljax.core.model import (
    MOLModel,
    create_gray_scott_model,
    create_gray_scott_periodic_fft,
)
from moljax.core.newton_krylov import NKParams
from moljax.core.operators import LinearOp, NonlinearOp
from moljax.core.stepping import (
    IntegratorType,
    _bdf2_predictor,
    _newton_start,
    adaptive_integrate,
    adaptive_integrate_imex,
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
from moljax.core.utils import StatusCode


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

    def test_adaptive_be_completes(self):
        """Adaptive BE on y' = -y reaches t = 1 with a sensible step count.

        The BE error estimate is the difference to a Crank-Nicolson step,
        which is the BE local error dt^2/2 y''. The earlier estimate
        dt F(y_be) never fell below the tolerance and every step was
        rejected until MAX_STEPS.
        """
        grid = Grid1D.uniform(4, 0.0, 1.0)

        def decay_rhs(state, grid, t, params):
            return {'u': -state['u']}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="decay", apply=decay_rhs),),
            nonlinear_ops=()
        )
        y0 = {'u': jnp.ones(grid.nx_total)}

        result = adaptive_integrate(
            model, y0, 0.0, 1.0, 0.01, method=IntegratorType.BE,
            max_steps=2000, nk_params=NKParams(newton_tol=1e-12)
        )

        assert int(result.status) == 0, f"status {int(result.status)}, t_final {float(result.t_final)}"
        assert abs(float(result.t_final) - 1.0) < 1e-12
        assert int(result.n_accepted) < 200, f"{int(result.n_accepted)} accepted steps"
        assert abs(float(result.y_final['u'][1]) - np.exp(-1.0)) < 5e-3

    def test_adaptive_float32_model_under_x64(self):
        """A float32 model must integrate while x64 is enabled.

        The CFL limit used to come back as float64, so the lax.cond that
        picks the initial dt saw branches of different dtypes.

        t_final is float64 here, not float32: the clock is deliberately
        carried wider than the field (see stepping._time_dtype), because
        a float32 clock stops advancing at large t while a float32 field
        is a considered choice about the field alone. The state's own
        dtype is what this test pins, and it is still float32.
        """
        model = create_gray_scott_model(Grid2D.uniform(8, 8, 0, 1, 0, 1), dtype=jnp.float32)
        y0 = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.1})

        result = adaptive_integrate(model, y0, 0.0, 0.01, 0.001, method=IntegratorType.RK4, max_steps=50)

        assert int(result.status) == 0
        assert result.y_final['u'].dtype == jnp.float32
        assert result.dt_history.dtype == jnp.float32
        assert result.t_final.dtype == jnp.float64

    def test_adaptive_save_every_includes_t_end(self):
        """save_every must land on multiples of save_every and always keep t_end.

        should_save compared state.n_accepted (the pre-increment count,
        starting at 0) against save_every, so a run saved after accepted
        steps 1, 6, 11, 16, 21 instead of 5, 10, 15, 20: with dt0=0.125,
        t_end=2.875 (23 accepted steps) and save_every=5, the saved times
        were [0, 0.125, 0.75, 1.375, 2.0, 2.625] (measured) and the run's
        final state at t_end=2.875 was never saved at all. The fix
        evaluates should_save on the post-increment count and forces a
        save of the final accepted step, giving
        [0, 0.625, 1.25, 1.875, 2.5, 2.875] here: t0, every fifth accepted
        time, and t_end, with no duplicate of the first entry.

        Checked on both adaptive integrators: BE (a Newton-Krylov implicit
        method) on scalar decay, and IMEX Euler on the Gray-Scott model.
        dt0 and t_end are exact binary fractions (multiples of 0.125) so
        the accepted step count and save times are exact, not merely
        close, and PIDParams(dt_max=dt0, atol/rtol huge) keeps dt pinned
        at dt0 and every step accepted so the count is deterministic.
        """
        dt0 = 0.125
        n_steps = 23
        t_end = dt0 * n_steps
        save_every = 5
        expected_t = [0.0, 0.625, 1.25, 1.875, 2.5, 2.875]
        pid_params = PIDParams(dt_max=dt0, atol=1e6, rtol=1e6)

        # BE on scalar decay (Newton-Krylov implicit).
        grid = Grid1D.uniform(4, 0.0, 1.0)

        def decay_rhs(state, grid, t, params):
            return {'u': -state['u']}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="decay", apply=decay_rhs),),
            nonlinear_ops=()
        )
        y0 = {'u': jnp.ones(grid.nx_total)}

        result = adaptive_integrate(
            model, y0, 0.0, t_end, dt0,
            method=IntegratorType.BE,
            max_steps=100,
            pid_params=pid_params,
            nk_params=NKParams(newton_tol=1e-12),
            save_every=save_every
        )

        assert int(result.n_accepted) == n_steps
        t_hist = result.t_history[:int(result.n_steps)]
        assert jnp.allclose(t_hist, jnp.array(expected_t)), f"BE history: {t_hist}"
        assert abs(float(result.t_final) - t_end) < 1e-12
        assert float(t_hist[-1]) == float(result.t_final)
        assert jnp.all(jnp.diff(t_hist) > 0), "duplicate or out-of-order save"

        # IMEX Euler on Gray-Scott (a different accept_step body entirely).
        imex_model, fft_cache, diffusivities, y0_imex = gray_scott_off_equilibrium()

        result_imex = adaptive_integrate_imex(
            imex_model, y0_imex, 0.0, t_end, dt0, fft_cache, diffusivities,
            use_strang=False, max_steps=100, pid_params=pid_params, save_every=save_every
        )

        assert int(result_imex.n_accepted) == n_steps
        t_hist_imex = result_imex.t_history[:int(result_imex.n_steps)]
        assert jnp.allclose(t_hist_imex, jnp.array(expected_t)), f"IMEX history: {t_hist_imex}"
        assert abs(float(result_imex.t_final) - t_end) < 1e-12
        assert float(t_hist_imex[-1]) == float(result_imex.t_final)
        assert jnp.all(jnp.diff(t_hist_imex) > 0), "duplicate or out-of-order save"

    def test_bdf2_startup_is_second_order(self):
        """The BDF2 startup step (step_count < 1) must itself be second order.

        be_only always returned y_be, even on the branch taken for BDF2
        startup (use_be = is_be OR bdf2_startup), where y_cn is already
        computed for the error estimate and is second-order accurate. Since
        this measures a single step's local truncation error against the
        exact solution (one order higher than the corresponding global
        order), a first-order local solution (y_be, LTE O(dt^2)) gives an
        error ratio near 4 across a halving, and the second-order local
        solution (y_cn, LTE O(dt^3)) this fix returns gives a ratio near 8.
        Measured: about 3.69 before the fix, about 7.62 after. Tolerances
        are set loose enough that both steps are accepted at their input dt
        unconditionally.
        """
        grid = Grid1D.uniform(4, 0.0, 1.0)

        def decay_rhs(state, grid, t, params):
            return {'u': -state['u']}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="decay", apply=decay_rhs),),
            nonlinear_ops=()
        )
        nk_params = NKParams(newton_tol=1e-13, max_newton_iters=20)
        pid_params = PIDParams(atol=1.0, rtol=1.0)

        errors = []
        for dt in (0.1, 0.05):
            y0 = {'u': jnp.ones(grid.nx_total)}
            result = adaptive_integrate(
                model, y0, 0.0, dt, dt,
                method=IntegratorType.BDF2,
                max_steps=1,
                pid_params=pid_params,
                nk_params=nk_params
            )
            assert int(result.n_accepted) == 1, "the single startup step must be accepted"
            assert abs(float(result.t_final) - dt) < 1e-12
            errors.append(abs(float(result.y_final['u'][1]) - np.exp(-dt)))

        ratio = errors[0] / errors[1]
        assert 7.0 <= ratio <= 8.5, f"errors {errors}, ratio {ratio:.3f}"

    def test_bdf2_startup_rejects_unconverged_cn(self):
        """A BDF2 startup step must be rejected when its CN solve fails, even if BE converges.

        be_only returns y_cn (the second-order state) on the BDF2 startup
        branch, but until now it still returned BE's NKStats. u' = 2u with
        u0 = 1e-7 and dt = 1 makes BE (one evaluation at t + dt) converge
        to the default Newton tolerance while CN (which must satisfy the
        trapezoidal residual at both endpoints) does not: CN's residual is
        about 3.46e-7, above the 1e-8 default. With BE's converged = True
        standing in for the whole step, the accept/reject check let this
        unconverged CN state through, seeding the BDF2 history with it.

        The fix returns CN's stats alongside y_cn on this branch, so the
        adaptive integrator rejects the dt = 1 attempt and subdivides
        instead. Checked two ways: directly, that be_step converges and
        cn_step does not at this state; and through adaptive_integrate,
        that the startup attempt is rejected at least once and the run
        still lands close to the exact solution once it does converge.
        """
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def growth_rhs(state, grid, t, params):
            return {'u': 2.0 * state['u']}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="growth", apply=growth_rhs),),
            nonlinear_ops=()
        )
        y0 = {'u': jnp.full(grid.nx_total, 1e-7)}
        dt = 1.0
        nk_params = NKParams()  # default newton_tol = 1e-8

        _, stats_be = be_step(model, y0, 0.0, dt, nk_params=nk_params)
        _, stats_cn = cn_step(model, y0, 0.0, dt, nk_params=nk_params)
        assert bool(stats_be.converged), "BE must converge for this case to exercise the regression"
        assert not bool(stats_cn.converged), "CN must fail to converge for this case to exercise the regression"

        result = adaptive_integrate(
            model, y0, 0.0, dt, dt,
            method=IntegratorType.BDF2,
            max_steps=200,
            pid_params=PIDParams(),
            nk_params=nk_params
        )
        assert int(result.n_rejected) >= 1, \
            "the dt = 1 startup attempt must be rejected since its CN solve did not converge"
        exact = 1e-7 * np.exp(2.0)
        error = abs(float(result.y_final['u'][1]) - exact)
        assert error < 2e-7, f"error {error:.3e} too large; startup should have subdivided to converge"


class TestImplicitPredictor:
    """The explicit-Euler predictor is only a Newton start; it must not poison the step."""

    def test_be_step_with_non_finite_predictor(self):
        """A forcing term singular at t = 0.

        y' = -y + t^(-1/2) has an integrable singularity: backward Euler
        evaluates F only at t + dt and is well defined, but the Euler
        predictor y + dt F(y, 0) is infinite. The step must start Newton
        from y instead and return the BE solution.
        """
        grid = Grid1D.uniform(1, 0.0, 1.0)

        def forced_decay(state, grid, t, params):
            return {'u': -state['u'] + 1.0 / jnp.sqrt(t)}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="forced_decay", apply=forced_decay),),
            nonlinear_ops=()
        )
        y0 = {'u': jnp.ones(3)}
        dt = 0.1

        y1, stats = be_step(model, y0, 0.0, dt, nk_params=NKParams(newton_tol=1e-12))

        expected = (1.0 + dt / np.sqrt(dt)) / (1.0 + dt)
        assert bool(jnp.all(jnp.isfinite(y1['u'])))
        assert bool(stats.converged)
        assert abs(float(y1['u'][1]) - expected) < 1e-8

    def test_newton_start_rejects_amplified_predictor(self):
        """A finite but wildly amplified predictor must fall back to y.

        At 100x the explicit diffusion limit, explicit Euler amplifies the
        discrete Nyquist mode ((-1)^j, an eigenfunction of the periodic
        3-point Laplacian) by a factor of about -199: finite, but useless as
        a Newton start. The pre-existing test at this same dt used
        sin(2 pi x) instead, whose amplification factor is only about -0.92,
        which is why it never exercised this guard. A smooth low-mode state
        at the same dt must still return the predictor unchanged.
        """
        nx = 32
        grid = Grid1D.uniform(nx, 0.0, 1.0)
        D = 1.0

        def diffusion_rhs(state, grid, t, params):
            from moljax.core.bc import apply_bc
            from moljax.core.operators import laplacian_1d
            state = apply_bc(state, grid, {'u': FieldBCSpec.periodic()})
            return {'u': D * laplacian_1d(state['u'], grid)}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="diffusion", apply=diffusion_rhs),),
            nonlinear_ops=()
        )
        dt = 100 * 0.5 * grid.dx ** 2 / D

        nyquist = jnp.where(jnp.arange(grid.nx_total) % 2 == 0, 1.0, -1.0)
        y_nyquist = {'u': nyquist}
        start = _newton_start(model, y_nyquist, 0.0, dt)
        assert jnp.array_equal(start['u'], y_nyquist['u']), \
            "an amplified Nyquist predictor should be rejected, returning y exactly"

        x = grid.x_coords(include_ghost=True)
        y_smooth = {'u': jnp.sin(2 * jnp.pi * x)}
        predicted = euler_step(model, y_smooth, 0.0, dt)
        start_smooth = _newton_start(model, y_smooth, 0.0, dt)
        assert jnp.array_equal(start_smooth['u'], predicted['u']), \
            "a smooth low-mode state at the same dt should still return the predictor"

    def test_newton_start_accepts_predictor_from_zero_state(self):
        """A cold start (y = 0) with a nonzero forcing must not reject its own predictor.

        The growth guard compares max-abs(y_pred) against
        _NEWTON_PREDICTOR_MAX_GROWTH * max-abs(y); when y is identically
        zero that bound is exactly 0, so any nonzero, perfectly finite
        predictor used to fail the ratio test and the step silently fell
        back to y = 0 again, discarding the only informative predictor an
        external source term provides. _predictor_is_valid now skips the
        ratio test when max-abs(y) is exactly 0.
        """
        grid = Grid1D.uniform(4, 0.0, 1.0)

        def forced_rhs(state, grid, t, params):
            return {'u': jnp.ones_like(state['u'])}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="forced", apply=forced_rhs),),
            nonlinear_ops=()
        )
        y0 = {'u': jnp.zeros(grid.nx_total)}
        dt = 0.1

        predicted = euler_step(model, y0, 0.0, dt)
        start = _newton_start(model, y0, 0.0, dt)
        assert jnp.array_equal(start['u'], predicted['u']), \
            "a cold start with nonzero forcing should return the Euler predictor, not y"
        assert not jnp.array_equal(start['u'], y0['u'])

    def test_bdf2_predictor_is_ratio_aware(self):
        """(1+w) y_n - w y_{n-1} must beat 2 y_n - y_{n-1} when w != 1.

        On y' = -y with y_prev = 1 at t = 0 and y = exp(-1) at t = 1
        (dt_prev = 1, dt = 0.1, so w = 0.1), the constant-step formula
        2 y_n - y_{n-1} predicts -0.264: the wrong sign against the exact
        y(1.1) = exp(-1.1) = 0.3329. The ratio-aware predictor gives 0.3047,
        an 8.5 percent relative error, well under the 15 percent bound
        checked here. This is only ever a Newton start: on this linear
        problem both predictors reach the same converged solution in one
        Newton step, so the assertion is on the predictor's own value, not
        on iteration counts.
        """
        grid = Grid1D.uniform(4, 0.0, 1.0)
        model = MOLModel(
            grid=grid,
            bc_spec={'y': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
        )
        n = grid.nx_total
        dt_prev, dt = 1.0, 0.1
        y_prev = {'y': jnp.ones(n)}
        y = {'y': jnp.full(n, np.exp(-1.0))}

        pred = _bdf2_predictor(model, y, y_prev, dt, dt_prev)
        exact = np.exp(-1.1)

        assert float(pred['y'][1]) > 0, "predictor has the wrong sign"
        rel_err = abs(float(pred['y'][1]) - exact) / exact
        assert rel_err < 0.15, f"relative error {rel_err:.4f} exceeds 0.15"

    def test_implicit_steps_far_above_explicit_limit(self):
        """BE and CN take a finite step at 100x the explicit diffusion limit."""
        nx = 32
        grid = Grid1D.uniform(nx, 0.0, 1.0)
        D = 1.0

        def diffusion_rhs(state, grid, t, params):
            from moljax.core.bc import apply_bc
            from moljax.core.operators import laplacian_1d
            state = apply_bc(state, grid, {'u': FieldBCSpec.periodic()})
            return {'u': D * laplacian_1d(state['u'], grid)}

        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="diffusion", apply=diffusion_rhs),),
            nonlinear_ops=()
        )
        x = grid.x_coords(include_ghost=True)
        y0 = {'u': jnp.sin(2 * jnp.pi * x)}
        dt = 100 * 0.5 * grid.dx ** 2 / D

        for step in (be_step, cn_step):
            y1, stats = step(model, y0, 0.0, dt, nk_params=NKParams(max_krylov_iters=200))
            assert bool(jnp.all(jnp.isfinite(y1['u'])))
            assert bool(stats.converged)
            assert float(jnp.max(jnp.abs(y1['u']))) <= 1.0 + 1e-8


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



def cubic_decay_model():
    """u' = -u^3 on a one-cell periodic grid, the reviewer's fixed-step reproduction.

    Backward Euler at u0 = 1, dt = 1 needs more than one Newton iteration:
    capped at one, it stops at u = 0.5 with a residual of 0.6495 against
    the 1e-8 tolerance.
    """
    grid = Grid1D.uniform(1, 0.0, 1.0)
    op = NonlinearOp(name="cubic", apply=lambda s, g, t, p: {'u': -s['u'] ** 3})
    model = MOLModel(
        grid=grid,
        bc_spec={'u': FieldBCSpec.periodic()},
        params={'dtype': jnp.float64},
        linear_ops=(),
        nonlinear_ops=(op,)
    )
    return model, {'u': jnp.ones(grid.nx_total)}


class TestFixedStepReportsFailedSolves:
    """integrate_fixed_dt must not return a failed Newton solve as a normal result.

    do_be, do_cn and do_bdf2 each dropped the NKStats their step function
    returns (`y_new, _ = be_step(...)`) and the scan carry had no status
    field at all, so the fixed-step path had nothing to report with, where
    the adaptive path carries a StatusCode and rejects on
    nk_stats.converged. On u' = -u^3, u0 = 1, dt = 1 with
    max_newton_iters = 1, be_step returns u = 0.5 with converged = False
    and residual 0.6495, and integrate_fixed_dt returned that 0.5 with no
    indication whatsoever.
    """

    def test_unconverged_solve_is_reported(self):
        """The reproduction: a raise by default, NK_FAILED with return_status."""
        model, y0 = cubic_decay_model()
        nk = NKParams(max_newton_iters=1)

        y_step, stats = be_step(model, y0, 0.0, 1.0, nk_params=nk)
        assert not bool(stats.converged)
        assert abs(float(y_step['u'][1]) - 0.5) < 1e-12
        assert abs(float(stats.final_res_norm) - 0.6495190528383293) < 1e-12

        with pytest.raises(RuntimeError, match="NK_FAILED"):
            integrate_fixed_dt(
                model, y0, 0.0, 1.0, 1.0, method=IntegratorType.BE, nk_params=nk
            )

        _, _, y_final, status = integrate_fixed_dt(
            model, y0, 0.0, 1.0, 1.0, method=IntegratorType.BE, nk_params=nk,
            return_status=True
        )
        assert int(status) == StatusCode.NK_FAILED
        # The failed step does not advance the state: the 0.5 is not returned.
        assert float(y_final['u'][1]) == 1.0

    def test_failure_freezes_the_remaining_steps(self):
        """Every step after the first failure is a no-op, not a further wrong step."""
        model, y0 = cubic_decay_model()
        nk = NKParams(max_newton_iters=1)
        t_hist, y_hist, y_final, status = integrate_fixed_dt(
            model, y0, 0.0, 5.0, 1.0, method=IntegratorType.BE, nk_params=nk,
            return_status=True
        )
        assert int(status) == StatusCode.NK_FAILED
        assert t_hist.shape == (5,)
        # All five saved states are the frozen initial state.
        assert float(jnp.max(jnp.abs(y_hist['u'] - 1.0))) == 0.0
        assert float(y_final['u'][1]) == 1.0

    def test_converging_run_reports_success_and_is_bit_identical(self):
        """A converging run reports SUCCESS and matches an eager loop exactly.

        Bit-identical, not merely close: the fix only adds a status to the
        scan carry and a lax.cond that keeps the state on failure, so a
        run in which nothing fails must produce exactly the arithmetic it
        produced before.
        """
        model, _, _, y0 = gray_scott_off_equilibrium()
        nk = NKParams(newton_tol=1e-12)
        dt, n_steps = 0.05, 10

        y = y0
        for i in range(n_steps):
            y, stats = be_step(model, y, i * dt, dt, nk_params=nk)
            assert bool(stats.converged)

        t_hist, y_hist, y_final, status = integrate_fixed_dt(
            model, y0, 0.0, dt * n_steps, dt, method=IntegratorType.BE,
            save_every=5, nk_params=nk, return_status=True
        )
        assert int(status) == StatusCode.SUCCESS
        for name in y:
            assert jnp.array_equal(y_final[name], y[name]), f"field {name} not bit-identical"
        assert t_hist.shape == (2,)
        assert y_hist['u'].shape[0] == 2

    def test_explicit_blowup_is_reported(self):
        """A non-finite explicit state stops the run with NON_FINITE_VALUES."""
        grid = Grid1D.uniform(1, 0.0, 1.0)
        op = NonlinearOp(name="blowup", apply=lambda s, g, t, p: {'u': s['u'] ** 3})
        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(),
            nonlinear_ops=(op,)
        )
        y0 = {'u': jnp.full(grid.nx_total, 1e3)}
        _, _, _, status = integrate_fixed_dt(
            model, y0, 0.0, 1.0, 0.1, method=IntegratorType.EULER, return_status=True
        )
        assert int(status) == StatusCode.NON_FINITE_VALUES


def source_term_model(t0, dtype=jnp.float32):
    """u' = t - t0 from u(t0) = 0, whose exact solution at t0 + 1 is 0.5.

    The right-hand side is the time itself, so a clock that does not
    advance is the whole answer: u stays at 0.
    """
    grid = Grid1D.uniform(1, 0.0, 1.0)
    op = NonlinearOp(
        name="source",
        apply=lambda s, g, t, p: {'u': jnp.full_like(s['u'], t - t0)}
    )
    model = MOLModel(
        grid=grid,
        bc_spec={'u': FieldBCSpec.periodic()},
        params={'dtype': dtype},
        linear_ops=(),
        nonlinear_ops=(op,)
    )
    return model, {'u': jnp.zeros(grid.nx_total, dtype=dtype)}


class TestTimeIsNotCarriedInTheStateDtype:
    """The clock must not inherit a float32 field's precision.

    integrate_fixed_dt and adaptive_integrate both built their time from
    model.dtype (jnp.array(t0, dtype=model.dtype)) and accumulated it step
    by step. At t0 = 1e6 in float32 the spacing is 0.0625, so t + 0.01
    rounds straight back to t: the fixed-step run took all 100 steps at
    the same instant and RK4 returned u = 0 instead of 0.5, and the
    adaptive run hit MAX_STEPS_REACHED with t_final still exactly 1e6.
    Time now follows JAX's default float type (float64 under x64), and
    the fixed path takes each timestamp as t0 + i*dt rather than as a
    running sum.
    """

    T0 = 1e6

    def test_fixed_step_advances_at_a_large_t0(self):
        model, y0 = source_term_model(self.T0)
        t_hist, y_hist, y_final = integrate_fixed_dt(
            model, y0, self.T0, self.T0 + 1.0, 0.01, method=IntegratorType.RK4
        )
        assert abs(float(y_final['u'][1]) - 0.5) < 1e-6, float(y_final['u'][1])
        assert abs(float(t_hist[-1]) - (self.T0 + 1.0)) < 1e-9
        assert float(t_hist[0]) > self.T0
        # The field itself is still float32; only the clock is wider.
        assert y_final['u'].dtype == jnp.float32
        assert t_hist.dtype == jnp.float64

    def test_adaptive_advances_at_a_large_t0(self):
        model, y0 = source_term_model(self.T0)
        result = adaptive_integrate(
            model, y0, self.T0, self.T0 + 1.0, 0.01, method=IntegratorType.RK4,
            max_steps=500, pid_params=PIDParams(dt_max=0.01)
        )
        assert int(result.status) == StatusCode.SUCCESS
        assert abs(float(result.t_final) - (self.T0 + 1.0)) < 1e-9
        assert abs(float(result.y_final['u'][1]) - 0.5) < 1e-6
        assert result.y_final['u'].dtype == jnp.float32

    def test_timestamps_do_not_drift_with_the_step_count(self):
        """t0 + i*dt, not a running sum: 1000 steps of 0.001 land exactly on 1.0."""
        model, y0 = source_term_model(0.0, dtype=jnp.float64)
        t_hist, _, _ = integrate_fixed_dt(
            model, y0, 0.0, 1.0, 0.001, method=IntegratorType.RK4, save_every=100
        )
        expected = [0.1 * (i + 1) for i in range(10)]
        for got, want in zip([float(v) for v in t_hist], expected, strict=True):
            assert abs(got - want) < 1e-13, f"{got} vs {want}"
        assert float(t_hist[-1]) == 1.0

    def test_unrepresentable_step_is_rejected_without_x64(self):
        """With no wider type to fall back on, the run is refused at validation."""
        t0 = jnp.array(1e6, dtype=jnp.float32)
        dt = jnp.array(0.01, dtype=jnp.float32)
        assert float(t0 + dt) == float(t0), "the premise of this test no longer holds"

        with jax.enable_x64(False):
            model, y0 = source_term_model(1e6)
            with pytest.raises(ValueError, match="not representable"):
                integrate_fixed_dt(
                    model, y0, 1e6, 1e6 + 1.0, 0.01, method=IntegratorType.RK4
                )
            with pytest.raises(ValueError, match="not representable"):
                adaptive_integrate(
                    model, y0, 1e6, 1e6 + 1.0, 0.01, method=IntegratorType.RK4,
                    max_steps=50
                )


def two_root_model():
    """u' = -2u^2(u - 1/2) from u0 = 1, the reviewer's error-estimate reproduction.

    At dt = 1 with max_newton_iters = 1, Crank-Nicolson converges to 0.5
    (the equation's other steady state) and backward Euler fails at the
    same 0.5, so the embedded estimate y_cn - y_be is exactly zero.
    """
    grid = Grid1D.uniform(1, 0.0, 1.0)
    op = NonlinearOp(
        name="two_root",
        apply=lambda s, g, t, p: {'u': -2.0 * s['u'] ** 2 * (s['u'] - 0.5)}
    )
    model = MOLModel(
        grid=grid,
        bc_spec={'u': FieldBCSpec.periodic()},
        params={'dtype': jnp.float64},
        linear_ops=(),
        nonlinear_ops=(op,)
    )
    return model, {'u': jnp.ones(grid.nx_total)}


# u(1) for two_root_model, from scipy.integrate.solve_ivp at rtol 1e-12,
# atol 1e-14 (the reviewer's independent reproduction reports the same
# 0.6510085678). Hard-coded so the test does not depend on scipy.
TWO_ROOT_EXACT = 0.651008567786174


class TestErrorEstimateRequiresEverySolve:
    """An embedded error estimate is only valid if every solve in it converged.

    cn_with_err and bdf2_with_err threw away the auxiliary backward Euler
    solve's NKStats (`y_be, _ = be_step(...)`) and reported only the
    primary solve's, so a step was accepted on the strength of a
    difference between one converged state and one that had merely
    stopped iterating. When both stall at the same value the difference is
    exactly zero, which the controller reads as a perfect step: on
    u' = -2u^2(u - 1/2), u0 = 1, dt = 1 with max_newton_iters = 1, err was
    0.0 and the run finished with SUCCESS at y = 0.5 against the reference
    0.6510085678. Commit 58c0d1f closed the same hole on the BDF2 startup
    branch; this is the rest of it.
    """

    def test_both_solves_stalling_at_one_value_is_not_a_perfect_step(self):
        """The reproduction, first as the two solves, then through the integrator."""
        model, y0 = two_root_model()
        nk = NKParams(max_newton_iters=1)

        y_cn, stats_cn = cn_step(model, y0, 0.0, 1.0, nk_params=nk)
        y_be, stats_be = be_step(model, y0, 0.0, 1.0, nk_params=nk)
        assert bool(stats_cn.converged)
        assert not bool(stats_be.converged)
        assert float(y_cn['u'][1] - y_be['u'][1]) == 0.0, "the zero estimate is the premise"

        result = adaptive_integrate(
            model, y0, 0.0, 1.0, 1.0, method=IntegratorType.CN, max_steps=500,
            nk_params=nk, pid_params=PIDParams(dt_max=1.0)
        )
        assert int(result.n_rejected) >= 1, "the dt = 1 attempt was accepted again"
        assert int(result.status) == StatusCode.SUCCESS
        assert abs(float(result.y_final['u'][1]) - TWO_ROOT_EXACT) < 1e-3, (
            f"subdivided to {float(result.y_final['u'][1])}, "
            f"reference {TWO_ROOT_EXACT}"
        )

    def test_bdf2_estimate_requires_its_auxiliary_solve(self):
        """bdf2_with_err has the same auxiliary backward Euler solve."""
        model, y0 = two_root_model()
        result = adaptive_integrate(
            model, y0, 0.0, 1.0, 1.0, method=IntegratorType.BDF2, max_steps=500,
            nk_params=NKParams(max_newton_iters=1), pid_params=PIDParams(dt_max=1.0)
        )
        assert int(result.n_rejected) >= 1
        assert int(result.status) == StatusCode.SUCCESS
        assert abs(float(result.y_final['u'][1]) - TWO_ROOT_EXACT) < 1e-3

    def test_be_method_rejects_when_its_auxiliary_cn_fails(self):
        """The plain BE branch reads the same y_cn - y_be difference.

        On u' = 2u from u0 = 1e-7 at dt = 1 with the default tolerances,
        backward Euler converges and Crank-Nicolson does not (residual
        3.46e-7 against the 1e-8 tolerance); the estimate is the
        difference between them, so backward Euler's own success is not
        enough to accept the step. This is the case 58c0d1f used for the
        BDF2 startup branch, here on the BE method itself.
        """
        grid = Grid1D.uniform(1, 0.0, 1.0)
        model = MOLModel(
            grid=grid,
            bc_spec={'u': FieldBCSpec.periodic()},
            params={'dtype': jnp.float64},
            linear_ops=(LinearOp(name="grow", apply=lambda s, g, t, p: {'u': 2.0 * s['u']}),),
            nonlinear_ops=()
        )
        y0 = {'u': jnp.full(grid.nx_total, 1e-7)}

        _, stats_be = be_step(model, y0, 0.0, 1.0)
        _, stats_cn = cn_step(model, y0, 0.0, 1.0)
        assert bool(stats_be.converged)
        assert not bool(stats_cn.converged)

        result = adaptive_integrate(
            model, y0, 0.0, 1.0, 1.0, method=IntegratorType.BE, max_steps=500,
            pid_params=PIDParams(dt_max=1.0)
        )
        assert int(result.n_rejected) >= 1, "the dt = 1 attempt was accepted again"
        assert int(result.status) == StatusCode.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
