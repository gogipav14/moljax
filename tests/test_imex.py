"""
Tests for IMEX integrators in MOL-JAX.

Validates:
- IMEX Euler stability
- IMEX Strang accuracy
- IMEX adaptive integration
- dt policy for IMEX (no diffusion limit)
"""

import pytest
import jax
import jax.numpy as jnp

from moljax.core.grid import Grid2D
from moljax.core.bc import BCType
from moljax.core.model import (
    create_gray_scott_model,
    create_gray_scott_periodic_fft,
)
from moljax.core.fft_solvers import create_fft_cache
from moljax.core.stepping import (
    imex_euler_step,
    imex_strang_step,
    imex_ssprk2_step,
    adaptive_integrate_imex,
    integrate_imex_fixed_dt,
)
from moljax.core.dt_policy import (
    imex_cfl_dt,
    heisenberg_cfl_dt,
    CFLParams,
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


class TestIMEXAccuracy:
    """Test IMEX accuracy."""

    def test_pure_diffusion_exact(self):
        """IMEX should be exact for pure diffusion."""
        # Create a model with only diffusion (no reaction)
        grid = Grid2D.uniform(32, 32, 0, 2*jnp.pi, 0, 2*jnp.pi, n_ghost=1)
        model = create_gray_scott_model(
            grid, Du=0.1, Dv=0.05, F=0.0, k=0.0,  # F=k=0 means no reaction
            bc_type=BCType.PERIODIC
        )
        fft_cache = create_fft_cache(grid)
        diffusivities = {'u': 0.1, 'v': 0.05}

        # Initial sine wave
        X, Y = grid.meshgrid(include_ghost=True)
        u0 = jnp.sin(X) * jnp.sin(Y)
        v0 = jnp.cos(X) * jnp.cos(Y)
        state = {'u': u0, 'v': v0}

        dt = 0.01
        # IMEX with no reaction should give exact diffusion
        y_imex = imex_euler_step(model, state, 0.0, dt, fft_cache, diffusivities)

        # Analytical diffusion decay: exp(-D*(kx^2+ky^2)*dt) with kx=ky=1
        decay_u = jnp.exp(-0.1 * 2 * dt)
        decay_v = jnp.exp(-0.05 * 2 * dt)

        # Extract interior
        sl_y, sl_x = grid.interior_slice
        u_interior = y_imex['u'][sl_y, sl_x]
        v_interior = y_imex['v'][sl_y, sl_x]
        u0_interior = u0[sl_y, sl_x]
        v0_interior = v0[sl_y, sl_x]

        # Check decay
        expected_u = decay_u * u0_interior
        expected_v = decay_v * v0_interior

        assert jnp.allclose(u_interior, expected_u, rtol=0.01)
        assert jnp.allclose(v_interior, expected_v, rtol=0.01)


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
