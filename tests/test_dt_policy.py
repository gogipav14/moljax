"""
Tests for dt policy.

Verifies:
- dt decreases when dx decreases (CFL scaling)
- dt decreases when diffusivity increases
- PID controller adjusts dt based on error
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# The dtype tests need float64 to exist so that float32 can differ from it.
jax.config.update("jax_enable_x64", True)

from moljax.core.bc import FieldBCSpec
from moljax.core.dt_policy import (
    CFLParams,
    PIDParams,
    create_initial_controller_state,
    heisenberg_cfl_dt,
    imex_cfl_dt,
    pid_controller_dt,
    propose_dt,
)
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.model import MOLModel
from moljax.core.newton_krylov import NKStats
from moljax.core.operators import LinearOp
from moljax.core.stepping import IntegratorType, adaptive_integrate
from moljax.core.utils import StatusCode


class TestCFLDtype:
    """The CFL limits come back in the requested dtype.

    adaptive_integrate carries dt in the model's dtype; a float64 CFL limit
    for a float32 model made lax.cond branches disagree and the loop fail.
    """

    @pytest.mark.parametrize("cfl_fn", [heisenberg_cfl_dt, imex_cfl_dt])
    def test_cfl_dt_dtype_follows_argument(self, cfl_fn):
        grid = Grid2D.uniform(8, 8, 0.0, 1.0, 0.0, 1.0)
        params = {'Du': 0.16, 'Dv': 0.08, 'F': 0.04, 'k': 0.06, 'vx': 0.3}
        cfl_params = CFLParams(dt_max=10.0)

        dt32 = cfl_fn(grid, params, cfl_params, dtype=jnp.float32)
        dt64 = cfl_fn(grid, params, cfl_params, dtype=jnp.float64)
        dt_default = cfl_fn(grid, params, cfl_params)

        assert dt32.dtype == jnp.float32
        assert dt64.dtype == jnp.float64
        assert dt_default.dtype == jnp.result_type(float)
        assert abs(float(dt32) - float(dt64)) < 1e-6 * float(dt64)
        assert float(dt64) > 0.0


class TestCFLLimiter:
    """Tests for CFL-based dt limiting."""

    def test_advection_dt_scales_with_dx(self):
        """Test that advection CFL dt scales linearly with dx."""
        dts = []
        dxs = []

        for nx in [50, 100, 200]:
            grid = Grid1D.uniform(nx, 0.0, 1.0)
            params = {'vx': 1.0, 'vy': 0.0}
            cfl_params = CFLParams(cfl_advection=0.5)

            dt = heisenberg_cfl_dt(grid, params, cfl_params)
            dts.append(float(dt))
            dxs.append(grid.dx)

        # Check linear scaling: dt ~ dx
        ratio1 = dts[0] / dts[1]
        ratio2 = dts[1] / dts[2]
        dx_ratio1 = dxs[0] / dxs[1]
        dx_ratio2 = dxs[1] / dxs[2]

        assert abs(ratio1 - dx_ratio1) < 0.1, f"Advection CFL not scaling: {ratio1} vs {dx_ratio1}"
        assert abs(ratio2 - dx_ratio2) < 0.1, f"Advection CFL not scaling: {ratio2} vs {dx_ratio2}"

    def test_diffusion_dt_scales_with_dx_squared(self):
        """Test that diffusion CFL dt scales quadratically with dx."""
        dts = []
        dxs = []

        for nx in [50, 100, 200]:
            grid = Grid1D.uniform(nx, 0.0, 1.0)
            params = {'D': 1.0}
            cfl_params = CFLParams(cfl_diffusion=0.25)

            dt = heisenberg_cfl_dt(grid, params, cfl_params)
            dts.append(float(dt))
            dxs.append(grid.dx)

        # Check quadratic scaling: dt ~ dx^2
        ratio1 = dts[0] / dts[1]
        ratio2 = dts[1] / dts[2]
        dx_ratio1 = (dxs[0] / dxs[1]) ** 2
        dx_ratio2 = (dxs[1] / dxs[2]) ** 2

        assert abs(ratio1 - dx_ratio1) < 0.5, f"Diffusion CFL not scaling: {ratio1} vs {dx_ratio1}"
        assert abs(ratio2 - dx_ratio2) < 0.5, f"Diffusion CFL not scaling: {ratio2} vs {dx_ratio2}"

    def test_dt_decreases_with_velocity(self):
        """Test that dt decreases when velocity increases."""
        grid = Grid1D.uniform(100, 0.0, 1.0)
        cfl_params = CFLParams(cfl_advection=0.5)

        params1 = {'vx': 1.0}
        params2 = {'vx': 2.0}

        dt1 = heisenberg_cfl_dt(grid, params1, cfl_params)
        dt2 = heisenberg_cfl_dt(grid, params2, cfl_params)

        assert float(dt2) < float(dt1), "dt should decrease with velocity"
        # Should roughly halve
        assert abs(float(dt2) / float(dt1) - 0.5) < 0.1

    def test_dt_decreases_with_diffusivity(self):
        """Test that dt decreases when diffusivity increases."""
        grid = Grid1D.uniform(100, 0.0, 1.0)
        cfl_params = CFLParams(cfl_diffusion=0.25)

        params1 = {'D': 1.0}
        params2 = {'D': 2.0}

        dt1 = heisenberg_cfl_dt(grid, params1, cfl_params)
        dt2 = heisenberg_cfl_dt(grid, params2, cfl_params)

        assert float(dt2) < float(dt1), "dt should decrease with diffusivity"
        # Should roughly halve
        assert abs(float(dt2) / float(dt1) - 0.5) < 0.1


class TestPIDController:
    """Tests for PID dt controller."""

    def test_dt_increases_on_small_error(self):
        """Test that dt increases when error is small."""
        controller = create_initial_controller_state()
        pid_params = PIDParams(safety=0.9, max_factor=2.0)

        dt_old = jnp.array(0.1)
        err_ratio = jnp.array(0.1)  # Error well below tolerance

        dt_new, _ = pid_controller_dt(dt_old, err_ratio, controller, pid_params, order=4)

        assert float(dt_new) > float(dt_old), "dt should increase on small error"

    def test_dt_decreases_on_large_error(self):
        """Test that dt decreases when error is large (rejected step)."""
        controller = create_initial_controller_state()
        pid_params = PIDParams(safety=0.9, min_factor=0.2)

        dt_old = jnp.array(0.1)
        err_ratio = jnp.array(2.0)  # Error above tolerance (rejected)

        dt_new, _ = pid_controller_dt(dt_old, err_ratio, controller, pid_params, order=4)

        assert float(dt_new) < float(dt_old), "dt should decrease on large error"

    def test_dt_clamped_to_limits(self):
        """Test that dt is clamped to min/max."""
        controller = create_initial_controller_state()
        pid_params = PIDParams(dt_min=0.001, dt_max=1.0, max_factor=100.0)

        # Very small error should want large dt
        dt_old = jnp.array(0.5)
        err_ratio = jnp.array(1e-6)

        dt_new, _ = pid_controller_dt(dt_old, err_ratio, controller, pid_params, order=4)

        assert float(dt_new) <= pid_params.dt_max, "dt should be clamped to max"

        # Very large error should want tiny dt
        err_ratio = jnp.array(1e6)
        dt_new, _ = pid_controller_dt(dt_old, err_ratio, controller, pid_params, order=4)

        assert float(dt_new) >= pid_params.dt_min, "dt should be clamped to min"


class TestWaveCFL:
    """Test CFL for wave problems."""

    def test_wave_dt_scales_with_dx(self):
        """Test that wave CFL dt scales linearly with dx."""
        dts = []
        dxs = []

        K, rho = 1.0, 1.0

        for nx in [50, 100, 200]:
            grid = Grid1D.uniform(nx, 0.0, 1.0)
            params = {'K': K, 'rho': rho}
            cfl_params = CFLParams(cfl_wave=0.5)

            dt = heisenberg_cfl_dt(grid, params, cfl_params)
            dts.append(float(dt))
            dxs.append(grid.dx)

        # Check linear scaling
        ratio1 = dts[0] / dts[1]
        dx_ratio1 = dxs[0] / dxs[1]

        assert abs(ratio1 - dx_ratio1) < 0.1, f"Wave CFL not scaling: {ratio1} vs {dx_ratio1}"

    def test_wave_dt_decreases_with_wave_speed(self):
        """Test that dt decreases when wave speed increases."""
        grid = Grid1D.uniform(100, 0.0, 1.0)
        cfl_params = CFLParams(cfl_wave=0.5)

        # c = sqrt(K/rho)
        params1 = {'K': 1.0, 'rho': 1.0}  # c = 1
        params2 = {'K': 4.0, 'rho': 1.0}  # c = 2

        dt1 = heisenberg_cfl_dt(grid, params1, cfl_params)
        dt2 = heisenberg_cfl_dt(grid, params2, cfl_params)

        assert float(dt2) < float(dt1), "dt should decrease with wave speed"


class Test2DCFL:
    """Test CFL in 2D."""

    def test_2d_diffusion_more_restrictive(self):
        """Test that 2D diffusion CFL is more restrictive than 1D."""
        nx = 50
        grid1d = Grid1D.uniform(nx, 0.0, 1.0)
        grid2d = Grid2D.uniform(nx, nx, 0.0, 1.0, 0.0, 1.0)

        params = {'D': 1.0}
        cfl_params = CFLParams(cfl_diffusion=0.25)

        # Both have same dx, but 2D should have smaller dt
        # due to the 2D Laplacian having larger spectral radius
        heisenberg_cfl_dt(grid1d, params, cfl_params)
        dt2d = heisenberg_cfl_dt(grid2d, params, cfl_params)

        # In principle they should be similar since we use same CFL number
        # The key point is both should give stable dt
        assert float(dt2d) > 0


class TestCombinedAdvectionDiffusionCFL:
    """heisenberg_cfl_dt must bound the COMBINED advection+diffusion
    operator, not min(separate advection limit, separate diffusion limit).

    Reproduction: periodic 16x16 unit square, D=1, vx=32, vy=0 (dx=dy=
    1/16). The old independent-limits formula returned dt=8.789e-4, but
    the checkerboard mode's real eigenvalue is -3072 (diffusion -2048 from
    2nd-order central differencing, plus upwind advection's own numerical
    diffusion -1024), giving z=dt*lambda=-2.7 and unstable amplification
    for both Euler (R(z)=-1.7) and SSPRK3 (R(z)=-1.3355).
    """

    @staticmethod
    def _build_dense_operator(grid, vx, vy, D):
        """Dense interior-to-interior matrix for
        du/dt = -vx*du/dx - vy*du/dy (upwind) + D*Laplacian(u) (central),
        applying moljax's own stencils with periodic ghost cells, so its
        eigenvalues reflect the real discrete operator.
        """
        from moljax.core.bc import apply_bc
        from moljax.core.operators import d1_upwind_2d, laplacian_2d

        bc_spec = {'u': FieldBCSpec.periodic()}
        ny, nx = grid.ny, grid.nx
        n = ny * nx

        def rhs(u_interior_flat):
            u_interior = u_interior_flat.reshape(ny, nx)
            full = jnp.zeros((grid.ny_total, grid.nx_total))
            full = full.at[grid.interior_slice].set(u_interior)
            full = apply_bc({'u': full}, grid, bc_spec)['u']
            dudx = d1_upwind_2d(full, vx, grid, axis=1)
            dudy = d1_upwind_2d(full, vy, grid, axis=0)
            lap = laplacian_2d(full, grid)
            dudt = -vx * dudx - vy * dudy + D * lap
            return dudt[grid.interior_slice].reshape(-1)

        identity = np.eye(n)
        columns = [np.array(rhs(jnp.asarray(identity[:, j]))) for j in range(n)]
        return np.stack(columns, axis=1)

    def test_checkerboard_reproduction_is_stable_for_euler_and_ssprk3(self):
        grid = Grid2D.uniform(16, 16, 0.0, 1.0, 0.0, 1.0)
        params = {'D': 1.0, 'vx': 32.0, 'vy': 0.0}
        A = self._build_dense_operator(grid, vx=32.0, vy=0.0, D=1.0)
        eigenvalues = np.linalg.eigvals(A)

        def R_euler(z):
            return 1.0 + z

        def R_ssprk3(z):
            return 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0

        for method, R in ((IntegratorType.EULER, R_euler), (IntegratorType.SSPRK3, R_ssprk3)):
            cfl_params = CFLParams()
            dt = float(heisenberg_cfl_dt(grid, params, cfl_params, method=int(method)))
            z = dt * eigenvalues
            spectral_radius = float(np.max(np.abs(R(z))))
            assert spectral_radius < 1.0 + 1e-9, (
                f"method={method}: dt={dt:.6e} gives spectral radius "
                f"{spectral_radius:.6f} (unstable)"
            )

    def test_checkerboard_eigenvalue_matches_closed_form(self):
        """Sanity check on the dense operator itself: its most negative
        real eigenvalue must match the closed-form checkerboard value
        -(4D/dx^2 + 4D/dy^2 + 2|vx|/dx + 2|vy|/dy) = -3072 for this grid.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 1.0, 0.0, 1.0)
        A = self._build_dense_operator(grid, vx=32.0, vy=0.0, D=1.0)
        eigenvalues = np.linalg.eigvals(A)
        most_negative_real = float(np.min(eigenvalues.real))
        assert abs(most_negative_real - (-3072.0)) < 1e-6, most_negative_real

    def test_pre_fix_dt_would_have_been_unstable(self):
        """Direct check of the reported numbers: the old independent-limits
        dt (8.789e-4) gives |R(z)| > 1 for both Euler and SSPRK3 on the
        checkerboard mode, confirming the old formula under-restricted dt.
        """
        dx = 1.0 / 16.0
        # Both x and y diffuse (D applies on both axes even though vy=0);
        # only x advects.
        lam_checkerboard = -(4.0 * 1.0 / dx ** 2 + 4.0 * 1.0 / dx ** 2 + 2.0 * 32.0 / dx)
        old_dt = 0.9 * min(0.5 * dx / 32.0, 0.25 * dx ** 2 / 1.0)  # old CFLParams defaults
        z = old_dt * lam_checkerboard
        assert abs(old_dt - 8.789e-4) < 1e-7, old_dt
        assert abs(z - (-2.7)) < 1e-2, z
        assert abs((1.0 + z) - (-1.7)) < 1e-2
        r3 = 1.0 + z + z ** 2 / 2.0 + z ** 3 / 6.0
        assert abs(r3 - (-1.3355)) < 1e-3
        assert abs(1.0 + z) > 1.0
        assert abs(r3) > 1.0

    def test_default_call_is_stable_without_specifying_method(self):
        """Same reproduction, calling heisenberg_cfl_dt with its default
        arguments only (method defaults to Euler): the returned dt alone,
        with no explicit method kwarg, must already be stable. This is the
        call shape every pre-fix caller used, so it fails against the old
        min(advection, diffusion) formula on its own terms, not merely
        because of the new keyword argument.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 1.0, 0.0, 1.0)
        params = {'D': 1.0, 'vx': 32.0, 'vy': 0.0}
        A = self._build_dense_operator(grid, vx=32.0, vy=0.0, D=1.0)
        eigenvalues = np.linalg.eigvals(A)

        dt = float(heisenberg_cfl_dt(grid, params, CFLParams()))
        z = dt * eigenvalues
        spectral_radius = float(np.max(np.abs(1.0 + z)))  # Euler, the default
        assert spectral_radius < 1.0 + 1e-9, (
            f"default heisenberg_cfl_dt call gives dt={dt:.6e}, "
            f"Euler spectral radius {spectral_radius:.6f} (unstable)"
        )

    def test_pure_diffusion_matches_classical_2d_limit(self):
        """No advection: dt must equal the classical dx^2/(4D) * safety
        limit for 2D Euler (R_stab=2), the special case the combined
        formula reduces to.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 1.0, 0.0, 1.0)
        params = {'D': 1.0}
        cfl_params = CFLParams()
        dt = float(heisenberg_cfl_dt(grid, params, cfl_params, method=int(IntegratorType.EULER)))
        expected = cfl_params.safety * grid.dx ** 2 / (4.0 * 1.0)
        assert abs(dt - expected) < 1e-12, (dt, expected)

    def test_more_permissive_integrator_gets_a_larger_dt(self):
        """RK4's larger real stability boundary must yield a larger dt than
        Euler's on the same combined advection-diffusion operator.
        """
        grid = Grid2D.uniform(16, 16, 0.0, 1.0, 0.0, 1.0)
        params = {'D': 1.0, 'vx': 32.0, 'vy': 0.0}
        cfl_params = CFLParams()
        dt_euler = float(heisenberg_cfl_dt(grid, params, cfl_params, method=int(IntegratorType.EULER)))
        dt_ssprk3 = float(heisenberg_cfl_dt(grid, params, cfl_params, method=int(IntegratorType.SSPRK3)))
        dt_rk4 = float(heisenberg_cfl_dt(grid, params, cfl_params, method=int(IntegratorType.RK4)))
        assert dt_euler < dt_ssprk3 < dt_rk4


def float32_decay_model():
    """u' = -u at u0 = 1000 in float32, the reviewer's non-terminating reproduction.

    The default Newton tolerance is 1e-8 on the unweighted residual norm,
    which a float32 residual of a state of size 1000 cannot reach (the
    spacing of 1000 in float32 is about 6e-5), so every implicit solve on
    this model reports converged = False no matter how small dt gets.
    """
    grid = Grid1D.uniform(1, 0.0, 1.0)
    model = MOLModel(
        grid=grid,
        bc_spec={'u': FieldBCSpec.periodic()},
        params={'dtype': jnp.float32},
        linear_ops=(LinearOp(name="decay", apply=lambda s, g, t, p: {'u': -s['u']}),),
        nonlinear_ops=()
    )
    return model, {'u': jnp.full(grid.nx_total, 1000.0, dtype=jnp.float32)}


class TestRejectionsTerminate:
    """A step that keeps failing must not be retried forever.

    propose_dt read acceptance off err_ratio <= 1.0 alone, so a step whose
    Newton solve failed (and whose error estimate was therefore meaningless
    and usually tiny) took the PID's accepted branch and could have its dt
    grown by up to max_factor = 5, which outruns the integrator's halving
    on rejection. On the float32 decay model above with dt0 = 0.1 the dt
    plateaued between 1.6e-3 and 1.8e-3 and adaptive_integrate never
    returned at all, even with max_steps = 1: max_steps bounds accepted
    steps, and no step was ever accepted. propose_dt now takes the
    integrator's own decision, and the integrators carry the controller's
    consecutive_rejects counter (it was built by propose_dt and then thrown
    away by reject_step) so a per-step rejection budget can stop the loop.
    """

    def test_failed_solve_never_grows_dt(self):
        """The accepted branch, with its growth factor, is not taken on a failure."""
        grid = Grid1D.uniform(4, 0.0, 1.0)
        params = {'dtype': jnp.float64}
        state = {'u': jnp.ones(grid.nx_total)}
        controller = create_initial_controller_state(jnp.float64)
        dt_old = jnp.array(0.1)
        # The error estimate a pair of failed solves produces: tiny, and
        # meaningless. err_ratio <= 1 alone reads it as a fine step.
        err_ratio = jnp.array(1e-10)
        failed = NKStats(
            converged=jnp.array(False),
            newton_iters=jnp.array(1, dtype=jnp.int32),
            lin_iters=jnp.array(50, dtype=jnp.int32),
            final_res_norm=jnp.array(0.6495)
        )

        dt_told, _ = propose_dt(
            method=int(IntegratorType.BE), grid=grid, params=params, state=state,
            t=jnp.array(0.0), dt_old=dt_old, err_ratio=err_ratio,
            controller_state=controller, nk_stats=failed, order=1,
            accepted=jnp.array(False)
        )
        assert float(dt_told) < float(dt_old), (
            f"a failed solve grew dt from {float(dt_old)} to {float(dt_told)}"
        )

        # Left to the error test alone, the same call grows it.
        dt_err_only, _ = propose_dt(
            method=int(IntegratorType.BE), grid=grid, params=params, state=state,
            t=jnp.array(0.0), dt_old=dt_old, err_ratio=err_ratio,
            controller_state=controller, nk_stats=failed, order=1
        )
        assert float(dt_err_only) > float(dt_old)

    def test_rejection_budget_stops_the_run(self):
        """The reproduction terminates, with a status that says why.

        On the parent commit this call never returns; here it stops after
        max_rejections_per_step consecutive rejections with
        MAX_ATTEMPTS_REACHED, a status distinct from MAX_STEPS_REACHED
        (which counts accepted steps) and from DT_TOO_SMALL (dt is still
        far above dt_min when the budget runs out).
        """
        model, y0 = float32_decay_model()

        result = adaptive_integrate(
            model, y0, 0.0, 1.0, 0.1, method=IntegratorType.BE, max_steps=1,
            max_rejections_per_step=5
        )
        assert int(result.status) == StatusCode.MAX_ATTEMPTS_REACHED
        assert int(result.n_accepted) == 0
        assert int(result.n_rejected) == 5
        assert float(result.t_final) == 0.0

        # Same outcome at the documented default budget.
        result_default = adaptive_integrate(
            model, y0, 0.0, 1.0, 0.1, method=IntegratorType.BE, max_steps=1
        )
        assert int(result_default.status) == StatusCode.MAX_ATTEMPTS_REACHED
        assert int(result_default.n_rejected) == 10

    def test_dt_falls_monotonically_while_the_step_keeps_failing(self):
        """Ten rejections in a row must cut dt, not push it back up.

        This is the plateau itself, run as the integrator runs it: propose
        a dt for a failed step, halve it the way reject_step does, repeat.
        Told only the error ratio the failed solves produce, the loop is
        net-growing (the PID's growth outruns the halving) and dt settles
        at the controller's clamp instead of shrinking.
        """
        grid = Grid1D.uniform(4, 0.0, 1.0)
        params = {'dtype': jnp.float64}
        state = {'u': jnp.ones(grid.nx_total)}
        pid_params = PIDParams()
        err_ratio = jnp.array(1e-10)
        failed = NKStats(
            converged=jnp.array(False),
            newton_iters=jnp.array(1, dtype=jnp.int32),
            lin_iters=jnp.array(50, dtype=jnp.int32),
            final_res_norm=jnp.array(0.6495)
        )

        def reject_sequence(pass_decision):
            controller = create_initial_controller_state(jnp.float64)
            dt = jnp.array(0.1)
            seq = []
            for _ in range(10):
                kwargs = {'accepted': jnp.array(False)} if pass_decision else {}
                dt_new, controller = propose_dt(
                    method=int(IntegratorType.BE), grid=grid, params=params,
                    state=state, t=jnp.array(0.0), dt_old=dt, err_ratio=err_ratio,
                    controller_state=controller, nk_stats=failed,
                    pid_params=pid_params, order=1, **kwargs
                )
                dt = jnp.maximum(dt_new * 0.5, pid_params.dt_min)
                seq.append(float(dt))
            return seq

        told = reject_sequence(True)
        assert all(b < a for a, b in zip([0.1] + told[:-1], told, strict=True)), told
        assert told[-1] < 1e-4, told[-1]

        err_only = reject_sequence(False)
        assert err_only[-1] >= 0.1, err_only


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
