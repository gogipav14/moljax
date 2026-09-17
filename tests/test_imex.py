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
import pytest

# The order and exactness tests resolve errors at 1e-6 and below.
jax.config.update("jax_enable_x64", True)

from moljax.core.bc import BCType, FieldBCSpec
from moljax.core.dt_policy import (
    CFLParams,
    heisenberg_cfl_dt,
    imex_cfl_dt,
)
from moljax.core.fft_solvers import (
    apply_diffusion_inverse_fft,
    create_fft_cache,
    diffusion_rhs_fft,
    extract_interior,
)
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.model import (
    MOLModel,
    create_advdiff_periodic_fft,
    create_gray_scott_model,
    create_gray_scott_periodic_fft,
)
from moljax.core.operators import NonlinearOp, laplacian_2d
from moljax.core.state import tree_add, tree_axpy, tree_zeros_like
from moljax.core.stepping import (
    adaptive_integrate_imex,
    imex_euler_step,
    imex_ssprk2_step,
    imex_strang_step,
    integrate_imex_fixed_dt,
    rk4_step,
)
from moljax.core.utils import StatusCode


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


def _imex_ssprk2_step_via_fft(model, y, t, dt, fft_cache, diffusivities):
    """The original imex_ssprk2_step: L1, L2 from a second FFT round trip
    through diffusion_rhs_fft, rather than reading them off the stage
    solve's own spectral coefficients as the current step does. Kept
    here, not as literal arrays, so the reference is an independent
    computation rather than a magic number, and so it still calls
    diffusion_rhs_fft for the call-count check below.
    """
    gamma = 1.0 - 1.0 / 2.0 ** 0.5

    def reaction(state, time):
        if len(model.nonlinear_ops) > 0:
            return model.nonlinear_rhs(state, time)
        return tree_zeros_like(state)

    def diffusion(state):
        return diffusion_rhs_fft(state, model.grid, diffusivities, fft_cache)

    y = model.apply_bcs(y, t)
    U1 = apply_diffusion_inverse_fft(y, model.grid, gamma * dt, diffusivities, fft_cache)
    U1 = model.apply_bcs(U1, t + gamma * dt)
    R1 = reaction(U1, t)
    L1 = diffusion(U1)

    rhs2 = tree_axpy(tree_axpy(y, dt, R1), (1.0 - 2.0 * gamma) * dt, L1)
    U2 = apply_diffusion_inverse_fft(rhs2, model.grid, gamma * dt, diffusivities, fft_cache)
    U2 = model.apply_bcs(U2, t + (1.0 - gamma) * dt)
    R2 = reaction(U2, t + dt)
    L2 = diffusion(U2)

    y_new = tree_axpy(y, 0.5 * dt, tree_add(tree_add(R1, R2), tree_add(L1, L2)))
    y_new = model.apply_bcs(y_new, t + dt)
    return y_new, y, U1, L1


class TestIMEXSSPRK2StageReuse:
    """imex_ssprk2_step must reuse the stage FFT solve's own Laplacian."""

    def test_imex_ssprk2_reuses_stage_laplacian(self, monkeypatch):
        """Matches the original implementation and stops calling diffusion_rhs_fft.

        The algebraic identity dt L U1 = (U1 - y) / gamma (from the stage 1
        solve (I - gamma dt L) U1 = y) is checked directly against
        diffusion_rhs_fft's L1 = L U1 on a Gray-Scott 16x16 state, as a
        sanity check that the identity is mathematically valid; it is not
        what the step itself uses, since at float32 it subtracts nearly
        equal states and divides by a small number, amplifying roundoff
        (see test_imex_ssprk2_float32_stage_laplacian_accuracy). The step
        instead reads L U off the Helmholtz solve's own spectral
        coefficients (apply_diffusion_inverse_fft_with_laplacian), which
        this test does not distinguish from the identity at float64 (both
        agree to rounding). The full step is compared to the original
        implementation (_imex_ssprk2_step_via_fft, an independent
        computation, not a literal array), and finally diffusion_rhs_fft is
        monkeypatched with a call counter to confirm the step never calls
        it for the stage Laplacian.

        The expected count is 2, not 0: the explicit part is now the
        model's right-hand side minus the diffusion the split treats
        implicitly (see _imex_explicit_rhs), and that subtraction is one
        diffusion_rhs_fft call per explicit-part evaluation, of which
        imex_ssprk2_step makes two (one per stage). A regression that went
        back to recomputing L1 and L2 through diffusion_rhs_fft would make
        it 4. The reference step still uses model.nonlinear_rhs for its
        explicit part, which for this reaction-diffusion model (whose only
        linear operator is D * Laplacian) equals the new explicit part to
        roundoff, so the 1e-12 comparison below is unaffected.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 2.5, 0.0, 2.5)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)

        def bump(X, Y):
            return jnp.exp(-((X - 1.25) ** 2 + (Y - 1.25) ** 2))

        y0 = model.create_initial_state(init_fns={
            'u': lambda X, Y: 1.0 - 0.5 * bump(X, Y),
            'v': lambda X, Y: 0.25 * bump(X, Y),
        })
        dt = 0.05
        t = 0.3
        gamma = 1.0 - 1.0 / 2.0 ** 0.5

        y_ref, y_bc, U1_ref, L1_ref = _imex_ssprk2_step_via_fft(
            model, y0, t, dt, fft_cache, diffusivities
        )

        # The identity itself, on the interior (it does not hold on ghost cells).
        identity_max_err = 0.0
        for name in U1_ref:
            u_int = extract_interior(U1_ref[name], grid)
            y_int = extract_interior(y_bc[name], grid)
            lap_from_identity = (u_int - y_int) / (gamma * dt)
            lap_direct = extract_interior(L1_ref[name], grid)
            identity_max_err = max(
                identity_max_err, float(jnp.max(jnp.abs(lap_from_identity - lap_direct)))
            )
        assert identity_max_err < 1e-12, f"identity residual {identity_max_err:.3e}"

        call_count = {"n": 0}
        original = diffusion_rhs_fft

        def counting_diffusion_rhs_fft(*args, **kwargs):
            call_count["n"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            "moljax.core.fft_solvers.diffusion_rhs_fft", counting_diffusion_rhs_fft
        )

        y_new = imex_ssprk2_step(model, y0, t, dt, fft_cache, diffusivities)

        assert call_count["n"] == 2, (
            f"diffusion_rhs_fft was called {call_count['n']} times inside the step; "
            f"2 is one per explicit-part evaluation, 4 would mean the stage "
            f"Laplacians are being recomputed instead of read off the solve"
        )
        for name in y_ref:
            max_diff = float(jnp.max(jnp.abs(y_new[name] - y_ref[name])))
            assert max_diff < 1e-12, f"field {name}: max difference {max_diff:.3e}"


def _cos_diffusion_1d_float32(nx=32):
    """Model, FFT cache and exact discrete solution for the float32 stage-Laplacian check.

    32 periodic cells on [0, 2 pi], u0 = cos(x), D = 1, no reaction: the
    exact discrete solution is cos(x) * exp(lambda_1 * t), lambda_1 the
    discrete symbol of the periodic 3-point Laplacian at wavenumber 1.
    """
    grid = Grid1D.uniform(nx, 0.0, 2.0 * jnp.pi)
    dtype = jnp.float32
    model = MOLModel(
        grid=grid,
        bc_spec={'u': FieldBCSpec.periodic()},
        params={'dtype': dtype},
        linear_ops=(),
        nonlinear_ops=()
    )
    fft_cache = create_fft_cache(grid, dtype=dtype)
    diffusivities = {'u': 1.0}
    x = grid.x_coords(include_ghost=True)
    y0 = {'u': jnp.cos(x).astype(dtype)}
    lam = (2.0 * np.cos(grid.dx) - 2.0) / grid.dx ** 2

    def exact(t_end):
        x_interior = np.asarray(grid.x_coords(include_ghost=False))
        return np.cos(x_interior) * np.exp(lam * t_end)

    return model, fft_cache, diffusivities, y0, exact


class TestIMEXSSPRK2Float32StageLaplacian:
    """imex_ssprk2_step's stage Laplacian must not amplify float32 roundoff.

    The stage_laplacian helper recovered dt * L * U algebraically as
    (U - rhs) / gamma: correct in exact arithmetic, since U solves
    (I - gamma dt L) U = rhs, but it subtracts nearly equal states and
    divides by a small number, which amplifies FFT roundoff in float32.
    The regression signature is refinement making things worse: dt = 1e-4
    measured 7.57e-4 and dt = 1e-5 measured 1.92e-3, both far above the
    1.94e-7 the pre-regression implementation got at dt = 1e-4. The fix
    reads L U off the Helmholtz solve's own spectral coefficients
    (apply_diffusion_inverse_fft_with_laplacian) instead of recomputing it
    from real-space states.
    """

    @pytest.mark.parametrize("dt,max_error", [
        pytest.param(1e-4, 1e-6, id="dt=1e-4"),
        pytest.param(1e-5, 5e-4, id="dt=1e-5"),
    ])
    def test_float32_stage_laplacian_accuracy(self, dt, max_error):
        """Reviewer's reproduction, plus a finer dt to check refinement no longer misbehaves.

        At dt = 1e-4 the fix measures about 4.2e-7 here, matching the
        order of magnitude of the pre-regression commit 5823ad5's 1.94e-7
        and well under the bug's 7.57e-4, asserted below 1e-6.

        At dt = 1e-5 the fix measures about 9.6e-5: this is not smaller
        than the dt = 1e-4 case, since by 100,000 steps float32 rounding
        accumulated over the run dominates the (already tiny) truncation
        error, a generic float32 effect and not the bug. The historical
        diffusion_rhs_fft-based implementation this replaces shows the
        same non-monotonic error across dt to within rounding (checked by
        hand, not asserted here, since it is not the code under test).
        What distinguishes the fix from the bug is magnitude: 9.6e-5 is
        still about 20x below the bug's 1.92e-3 at the same dt, asserted
        below 5e-4.
        """
        model, fft_cache, diffusivities, y0, exact = _cos_diffusion_1d_float32()
        t_end = 1.0
        n_steps = round(t_end / dt)

        step = jax.jit(
            lambda y, t: imex_ssprk2_step(model, y, t, dt, fft_cache, diffusivities)
        )
        y = y0
        t = 0.0
        for _ in range(n_steps):
            y = step(y, t)
            t += dt

        u_interior = np.asarray(y['u'][model.grid.interior_slice])
        err = float(np.max(np.abs(u_interior - exact(t_end))))
        assert err < max_error, f"dt={dt:.0e}: max error {err:.3e}"

def advection_only_model(nx=16, vx=1.0):
    """Pure advection dressed as an advection-diffusion model (D = 0).

    create_advection_diffusion_model puts diffusion and advection in one
    LinearOp, so this model's right-hand side is entirely advection and
    the FFT diffusion solve, with D = 0, is the identity. The IMEX
    steppers must still advance it.
    """
    grid = Grid2D.uniform(nx, nx, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, n_ghost=1)
    model, fft_cache, diffusivities = create_advdiff_periodic_fft(
        grid, field_names=('c',), D=0.0, vx=vx, vy=0.0
    )
    X, _ = grid.meshgrid(include_ghost=True)
    return model, fft_cache, diffusivities, {'c': jnp.sin(X)}


class TestIMEXExplicitPart:
    """The IMEX explicit part is everything the FFT diffusion split does not do.

    The steppers evaluated model.nonlinear_rhs alone, which assumes a
    model's linear operators are exactly the diffusion the FFT solve
    inverts. create_advection_diffusion_model folds advection into the
    same LinearOp, so with D = 0 the solve was the identity, the explicit
    part was zero (the model has no nonlinear operators) and every IMEX
    step returned the state untouched: max-abs change 8.60e-16 on a 16x16
    sine with an advective right-hand side of max-abs 0.9936, and
    adaptive_integrate_imex reported SUCCESS on a state that never moved.
    """

    def test_fft_symbol_matches_the_second_difference_stencil(self):
        """The two Laplacians the split relies on agree to roundoff.

        The explicit part subtracts D * Delta y computed spectrally from a
        model right-hand side whose diffusion is the 5-point stencil, so
        the cancellation is only exact if the FFT symbol is that stencil's
        symbol, (2 cos(k dx) - 2)/dx^2 + (2 cos(k dy) - 2)/dy^2, and not
        -k^2. Checked here rather than assumed.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, n_ghost=1)
        fft_cache = create_fft_cache(grid)
        X, Y = grid.meshgrid(include_ghost=True)
        model = create_gray_scott_model(grid, Du=0.16, Dv=0.08, bc_type=BCType.PERIODIC)
        state = model.apply_bcs({'u': jnp.sin(X) * jnp.cos(2 * Y),
                                 'v': jnp.cos(3 * X) + 0.5 * jnp.sin(Y)}, 0.0)

        spectral = diffusion_rhs_fft(state, grid, {'u': 0.16, 'v': 0.08}, fft_cache)
        for name, D in (('u', 0.16), ('v', 0.08)):
            stencil = D * laplacian_2d(state[name], grid)
            diff = float(jnp.max(jnp.abs(
                extract_interior(spectral[name], grid) - extract_interior(stencil, grid)
            )))
            scale = float(jnp.max(jnp.abs(extract_interior(stencil, grid))))
            assert diff / scale < 1e-12, f"{name}: relative difference {diff / scale:.3e}"

    def test_explicit_part_of_a_reaction_diffusion_model_is_the_reaction(self):
        """Subtracting the split's diffusion leaves exactly what was there before."""
        from moljax.core.stepping import _imex_explicit_rhs

        grid = Grid2D.uniform(16, 16, 0.0, 2.5, 0.0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)
        X, Y = grid.meshgrid(include_ghost=True)
        state = model.apply_bcs({'u': 1.0 - 0.5 * jnp.exp(-((X - 1.25) ** 2 + (Y - 1.25) ** 2)),
                                 'v': 0.25 * jnp.exp(-((X - 1.25) ** 2 + (Y - 1.25) ** 2))}, 0.0)

        explicit = _imex_explicit_rhs(model, state, 0.0, fft_cache, diffusivities)
        reaction = model.nonlinear_rhs(state, 0.0)
        for name in explicit:
            diff = float(jnp.max(jnp.abs(
                extract_interior(explicit[name], grid) - extract_interior(reaction[name], grid)
            )))
            assert diff < 1e-12, f"{name}: explicit part differs from the reaction by {diff:.3e}"

    def test_advection_advances_at_the_right_speed(self):
        """The D = 0 reproduction, against an explicit RK4 reference.

        The semi-discrete problem is the same in both cases (the model's
        own right-hand side), so a second-order IMEX step and a
        fourth-order explicit one differ only by their truncation error
        over the short horizon used here. Compared on the interior:
        rk4_step leaves the ghost cells of its input untouched (model.rhs
        fills them on its own copy) while the IMEX steps return a state
        with boundary conditions applied, so the padding does not compare.
        """
        model, fft_cache, diffusivities, y0 = advection_only_model()
        interior = model.grid.interior_slice
        rhs0 = float(jnp.max(jnp.abs(model.rhs(y0, 0.0)['c'][interior])))
        assert rhs0 > 0.9, f"the test problem is trivial: max-abs rhs {rhs0:.3e}"

        dt, n_steps = 0.005, 10
        y_ref = y0
        for i in range(n_steps):
            y_ref = rk4_step(model, y_ref, i * dt, dt)
        moved = float(jnp.max(jnp.abs(y_ref['c'][interior] - y0['c'][interior])))
        assert moved > 0.04, f"the reference barely moved: {moved:.3e}"

        for step, tol in ((imex_euler_step, 0.05),
                          (imex_strang_step, 1e-4),
                          (imex_ssprk2_step, 1e-4)):
            y = y0
            for i in range(n_steps):
                y = step(model, y, i * dt, dt, fft_cache, diffusivities)
            err = float(jnp.max(jnp.abs(y['c'][interior] - y_ref['c'][interior])))
            assert err / moved < tol, f"{step.__name__}: relative error {err / moved:.3e}"

    def test_adaptive_imex_advances_the_advection_case(self):
        """adaptive_integrate_imex reported SUCCESS on a state that never moved."""
        model, fft_cache, diffusivities, y0 = advection_only_model()
        result = adaptive_integrate_imex(
            model, y0, 0.0, 0.1, 0.01, fft_cache, diffusivities,
            use_strang=True, max_steps=200
        )
        assert int(result.status) == StatusCode.SUCCESS
        interior = model.grid.interior_slice
        moved = float(jnp.max(jnp.abs(result.y_final['c'][interior] - y0['c'][interior])))
        assert moved > 0.05, f"the run advanced by only {moved:.3e}"

    def test_pure_diffusion_is_unchanged(self):
        """With no advection and no reaction, the step is the exact discrete decay."""
        grid = Grid2D.uniform(32, 32, 0.0, 2.0 * np.pi, 0.0, 2.0 * np.pi, n_ghost=1)
        base = create_gray_scott_model(grid, Du=0.1, Dv=0.05, bc_type=BCType.PERIODIC)
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
        lam = ((2.0 * np.cos(grid.dx) - 2.0) / grid.dx ** 2
               + (2.0 * np.cos(grid.dy) - 2.0) / grid.dy ** 2)
        sl_y, sl_x = grid.interior_slice
        for name, D in diffusivities.items():
            expected = np.exp(D * lam * dt) * np.asarray(state[name][sl_y, sl_x])
            err = np.max(np.abs(np.asarray(y_strang[name][sl_y, sl_x]) - expected))
            assert err < 1e-12, f"{name}: error {err:.3e}"

    def test_reaction_diffusion_step_is_unchanged(self):
        """A model whose linear part is only diffusion steps exactly as before.

        The reference is the step the old explicit part produced
        (model.nonlinear_rhs), written out here rather than stored as
        literal arrays.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 2.5, 0.0, 2.5, n_ghost=1)
        model, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)
        X, Y = grid.meshgrid(include_ghost=True)
        bump = jnp.exp(-((X - 1.25) ** 2 + (Y - 1.25) ** 2))
        y0 = {'u': 1.0 - 0.5 * bump, 'v': 0.25 * bump}
        dt, t = 0.05, 0.3

        def old_imex_euler(y):
            y = model.apply_bcs(y, t)
            N = model.nonlinear_rhs(y, t)
            rhs = tree_axpy(y, dt, N)
            y_new = apply_diffusion_inverse_fft(rhs, model.grid, dt, diffusivities, fft_cache)
            return model.apply_bcs(y_new, t + dt)

        y_ref = old_imex_euler(y0)
        y_new = imex_euler_step(model, y0, t, dt, fft_cache, diffusivities)
        for name in y_ref:
            diff = float(jnp.max(jnp.abs(y_new[name] - y_ref[name])))
            assert diff < 1e-12, f"{name}: max difference {diff:.3e}"


class TestIMEXSplitValidation:
    """A split the FFT solve cannot represent is refused, not approximated."""

    @pytest.mark.parametrize("step", [imex_euler_step, imex_strang_step, imex_ssprk2_step])
    def test_non_periodic_bc_raises(self, step):
        grid = Grid2D.uniform(8, 8, 0.0, 1.0, 0.0, 1.0, n_ghost=1)
        base = create_gray_scott_model(grid, bc_type=BCType.DIRICHLET)
        fft_cache = create_fft_cache(grid)
        state = base.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})
        with pytest.raises(ValueError, match="periodic"):
            step(base, state, 0.0, 0.01, fft_cache, {'u': 0.16, 'v': 0.08})

    def test_unknown_field_raises(self):
        grid = Grid2D.uniform(8, 8, 0.0, 2.5, 0.0, 2.5, n_ghost=1)
        model, fft_cache, _ = create_gray_scott_periodic_fft(grid)
        state = model.create_initial_state(fill_values={'u': 1.0, 'v': 0.0})
        with pytest.raises(ValueError, match="which the model does not have"):
            imex_euler_step(model, state, 0.0, 0.01, fft_cache, {'w': 0.16})
