"""
Tests for IMEX integrators in MOL-JAX.

Validates:
- IMEX Euler stability
- IMEX Strang accuracy
- IMEX adaptive integration
- dt policy for IMEX (no diffusion limit)
"""

import jax
import jax.numpy as jnp
import numpy as np

# The order and exactness tests resolve errors at 1e-6 and below.
jax.config.update("jax_enable_x64", True)

from moljax.core.bc import BCType, FieldBCSpec
from moljax.core.dt_policy import (
    CFLParams,
    heisenberg_cfl_dt,
    imex_cfl_dt,
)
from moljax.core.fft_solvers import create_fft_cache
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.model import (
    MOLModel,
    create_gray_scott_model,
    create_gray_scott_periodic_fft,
)
from moljax.core.operators import NonlinearOp
from moljax.core.stepping import (
    adaptive_integrate_imex,
    imex_euler_step,
    imex_ssprk2_step,
    imex_strang_step,
    integrate_imex_fixed_dt,
)


class TestIMEXStability:
    """Test IMEX stability for stiff problems."""

    def test_imex_euler_no_nan(self):
        """IMEX Euler should not produce NaN."""
        grid = Grid2D.uniform(32, 32, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        # Initial condition
        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        # Run a few steps with large dt (larger than explicit diffusion CFL)
        dt = 0.1  # Much larger than explicit CFL
        y = state
        for _ in range(10):
            y = imex_euler_step(model, y, 0.0, dt, fft_cache, diffusivities)

        # Check no NaN
        assert jnp.all(jnp.isfinite(y['u']))
        assert jnp.all(jnp.isfinite(y['v']))

    def test_imex_strang_no_nan(self):
        """IMEX Strang should not produce NaN."""
        grid = Grid2D.uniform(32, 32, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        dt = 0.1
        y = state
        for _ in range(10):
            y = imex_strang_step(model, y, 0.0, dt, fft_cache, diffusivities)

        assert jnp.all(jnp.isfinite(y['u']))
        assert jnp.all(jnp.isfinite(y['v']))

    def test_imex_stable_large_dt(self):
        """IMEX should remain stable with dt larger than explicit diffusion CFL."""
        grid = Grid2D.uniform(32, 32, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        # Compute explicit diffusion CFL
        cfl_params = CFLParams()
        dt_explicit = heisenberg_cfl_dt(grid, model.params, cfl_params)

        # Use dt significantly larger than explicit CFL
        dt_imex = float(dt_explicit) * 5.0

        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        # Run 20 steps
        y = state
        for _ in range(20):
            y = imex_strang_step(model, y, 0.0, dt_imex, fft_cache, diffusivities)

        # Should still be finite
        assert jnp.all(jnp.isfinite(y['u']))
        assert jnp.all(jnp.isfinite(y['v']))


def linear_reaction_diffusion_1d(D: float = 1.0, r: float = 0.3, nx: int = 32):
    """u_t = D u_xx - r u on [0, 2 pi], periodic, started from cos x.

    The FFT diffusion solve uses the discrete symbol, so the mode decays as
    exp((D lambda_1 - r) t) with lambda_1 = (2 cos dx - 2)/dx^2, and the
    only error left in an IMEX step is the splitting error.
    """
    grid = Grid1D.uniform(nx, 0.0, 2.0 * np.pi)
    reaction = NonlinearOp(name="linear_reaction", apply=lambda s, g, t, p: {'u': -r * s['u']})
    model = MOLModel(
        grid=grid,
        bc_spec={'u': FieldBCSpec(kind=BCType.PERIODIC)},
        params={'dtype': jnp.float64},
        nonlinear_ops=(reaction,)
    )
    fft_cache = create_fft_cache(grid)
    u0 = {'u': jnp.cos(grid.x_coords(include_ghost=True))}
    lam1 = (2.0 * np.cos(grid.dx) - 2.0) / grid.dx ** 2

    def exact(t_end):
        return np.exp((D * lam1 - r) * t_end) * np.cos(np.asarray(grid.x_coords()))

    return model, fft_cache, {'u': D}, u0, exact


def imex_errors_and_orders(step, dts=(0.2, 0.1, 0.05, 0.025), t_end=1.0):
    model, fft_cache, diffusivities, u0, exact = linear_reaction_diffusion_1d()
    errors = []
    for dt in dts:
        u = u0
        for i in range(int(round(t_end / dt))):
            u = step(model, u, i * dt, dt, fft_cache, diffusivities)
        errors.append(float(np.max(np.abs(np.asarray(u['u'][1:-1]) - exact(t_end)))))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    return errors, orders


class TestIMEXOrder:
    """The two second-order IMEX steps must show second order on a linear problem."""

    def test_imex_strang_order(self):
        errors, orders = imex_errors_and_orders(imex_strang_step)
        assert all(o > 1.8 for o in orders), f"errors {errors}, orders {orders}"

    def test_imex_ssprk2_order(self):
        errors, orders = imex_errors_and_orders(imex_ssprk2_step)
        assert all(o > 1.8 for o in orders), f"errors {errors}, orders {orders}"


class TestIMEXAccuracy:
    """Test IMEX accuracy."""

    def test_pure_diffusion_exact(self):
        """Strang is exact for pure diffusion; IMEX Euler is first order.

        The Strang half-steps apply exp(dt/2 D Laplacian) through the FFT,
        so with no reaction the step is the exact discrete decay up to
        rounding. IMEX Euler solves (I - dt D Laplacian), a first-order
        approximation with local error (dt D lambda)^2 / 2.
        """
        grid = Grid2D.uniform(32, 32, 0, 2*jnp.pi, 0, 2*jnp.pi, n_ghost=1)
        base = create_gray_scott_model(grid, Du=0.1, Dv=0.05, bc_type=BCType.PERIODIC)
        # F = k = 0 would still leave the u v^2 term, so drop the reaction operator.
        model = MOLModel(
            grid=base.grid, bc_spec=base.bc_spec, params=base.params,
            linear_ops=base.linear_ops, nonlinear_ops=()
        )
        fft_cache = create_fft_cache(grid)
        diffusivities = {'u': 0.1, 'v': 0.05}

        X, Y = grid.meshgrid(include_ghost=True)
        state = {'u': jnp.sin(X) * jnp.sin(Y), 'v': jnp.cos(X) * jnp.cos(Y)}

        dt = 0.05
        y_strang = imex_strang_step(model, state, 0.0, dt, fft_cache, diffusivities)
        y_euler = imex_euler_step(model, state, 0.0, dt, fft_cache, diffusivities)

        # Discrete symbol of the (1, 1) mode
        lam = ((2.0 * np.cos(grid.dx) - 2.0) / grid.dx ** 2
               + (2.0 * np.cos(grid.dy) - 2.0) / grid.dy ** 2)
        sl_y, sl_x = grid.interior_slice
        for name, D in diffusivities.items():
            expected = np.exp(D * lam * dt) * np.asarray(state[name][sl_y, sl_x])
            scale = np.max(np.abs(expected))
            err_strang = np.max(np.abs(np.asarray(y_strang[name][sl_y, sl_x]) - expected)) / scale
            err_euler = np.max(np.abs(np.asarray(y_euler[name][sl_y, sl_x]) - expected)) / scale
            assert err_strang < 1e-6, f"{name}: Strang error {err_strang:.2e}"
            assert err_euler < 1e-3, f"{name}: IMEX Euler error {err_euler:.2e}"


class TestIMEXDTPolicy:
    """Test dt policy for IMEX."""

    def test_imex_dt_larger_than_explicit(self):
        """IMEX CFL dt should be larger than explicit CFL dt."""
        grid = Grid2D.uniform(64, 64, 0, 2.5, 0, 2.5, n_ghost=1)
        params = {
            'Du': 0.16,
            'Dv': 0.08,
            'F': 0.04,
            'k': 0.06,
            'vx': 0.0,
            'vy': 0.0,
        }

        cfl_params = CFLParams(dt_max=10.0)

        # Explicit CFL includes diffusion
        dt_explicit = heisenberg_cfl_dt(grid, params, cfl_params)

        # IMEX CFL does not include diffusion
        dt_imex = imex_cfl_dt(grid, params, cfl_params)

        # IMEX should allow larger dt
        assert float(dt_imex) > float(dt_explicit)

    def test_imex_dt_scales_with_reaction(self):
        """IMEX dt should scale with reaction rate, not diffusion."""
        grid = Grid2D.uniform(32, 32, 0, 1, 0, 1, n_ghost=1)

        # Vary reaction rate
        cfl_params = CFLParams(dt_max=100.0)

        params_slow = {'F': 0.01, 'k': 0.01}
        params_fast = {'F': 0.1, 'k': 0.1}

        dt_slow = imex_cfl_dt(grid, params_slow, cfl_params)
        dt_fast = imex_cfl_dt(grid, params_fast, cfl_params)

        # Faster reaction should give smaller dt
        assert float(dt_fast) < float(dt_slow)


class TestIMEXAdaptive:
    """Test IMEX adaptive integration."""

    def test_adaptive_imex_completes(self):
        """Adaptive IMEX should complete integration."""
        grid = Grid2D.uniform(32, 32, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        # Initial condition with perturbation
        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        # Add small perturbation to v
        key = jax.random.PRNGKey(0)
        perturb = 0.01 * jax.random.normal(key, state['v'].shape)
        state['v'] = state['v'] + perturb

        # Run short integration
        result = adaptive_integrate_imex(
            model, state,
            t0=0.0, t_end=1.0, dt0=0.1,
            fft_cache=fft_cache,
            diffusivities=diffusivities,
            use_strang=True,
            max_steps=1000
        )

        # Should complete
        assert result.status == 0  # SUCCESS
        assert float(result.t_final) >= 0.99
        assert result.n_accepted > 0

    def test_adaptive_imex_accepts_steps(self):
        """Adaptive IMEX should accept most steps."""
        grid = Grid2D.uniform(32, 32, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        result = adaptive_integrate_imex(
            model, state,
            t0=0.0, t_end=0.5, dt0=0.05,
            fft_cache=fft_cache,
            diffusivities=diffusivities,
            use_strang=True,
            max_steps=500
        )

        # Acceptance rate should be reasonable (> 50%)
        accept_rate = result.n_accepted / (result.n_accepted + result.n_rejected + 1e-10)
        assert accept_rate > 0.5

    def test_fixed_dt_imex_matches_steps(self):
        """Fixed dt IMEX should give same result as manual steps."""
        grid = Grid2D.uniform(16, 16, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        dt = 0.05
        n_steps = 10

        # Manual steps
        y_manual = state
        for i in range(n_steps):
            y_manual = imex_strang_step(model, y_manual, i*dt, dt, fft_cache, diffusivities)

        # Fixed dt integration
        t_hist, y_hist, y_fixed = integrate_imex_fixed_dt(
            model, state,
            t0=0.0, t_end=n_steps*dt, dt=dt,
            fft_cache=fft_cache,
            diffusivities=diffusivities,
            use_strang=True
        )

        # Should match
        assert jnp.allclose(y_fixed['u'], y_manual['u'], atol=1e-10)
        assert jnp.allclose(y_fixed['v'], y_manual['v'], atol=1e-10)


class TestIMEXSolutionBounds:
    """Test IMEX solution stays within bounds."""

    def test_gray_scott_bounded(self):
        """Gray-Scott solution should stay within physical bounds."""
        grid = Grid2D.uniform(32, 32, 0, 2.5, 0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        # Initialize with physically meaningful values
        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})

        # Add small perturbation
        key = jax.random.PRNGKey(42)
        v_perturb = state['v'] + 0.1 * jax.random.uniform(key, state['v'].shape)
        state['v'] = v_perturb

        # Run for a while
        result = adaptive_integrate_imex(
            model, state,
            t0=0.0, t_end=5.0, dt0=0.1,
            fft_cache=fft_cache,
            diffusivities=diffusivities,
            use_strang=True,
            max_steps=2000
        )

        # Solution should stay in reasonable bounds
        # For Gray-Scott: u, v should stay in [-0.5, 1.5] for short times
        u_min = jnp.min(result.y_final['u'])
        u_max = jnp.max(result.y_final['u'])
        v_min = jnp.min(result.y_final['v'])
        v_max = jnp.max(result.y_final['v'])

        assert u_min > -1.0, f"u_min = {u_min} too negative"
        assert u_max < 2.0, f"u_max = {u_max} too large"
        assert v_min > -1.0, f"v_min = {v_min} too negative"
        assert v_max < 2.0, f"v_max = {v_max} too large"
