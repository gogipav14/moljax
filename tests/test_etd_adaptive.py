"""
Tests for the adaptive ETD driver (etd_integrate_adaptive).

The expected values come from sources that are independent of the
adaptive driver: closed-form solutions, hand arithmetic, and the
fixed-step driver etd_integrate. The grids are small.
"""

import jax

# float32 must differ from float64 in the dtype test.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import moljax.core.fft_integrators as fi
from moljax.core.dt_policy import PIDParams
from moljax.core.fft_integrators import (
    clear_compiled_drivers,
    etd_integrate,
    etd_integrate_adaptive,
)
from moljax.core.fft_operators import DiffusionOperator
from moljax.core.grid import Grid1D
from moljax.core.utils import StatusCode


class ConstantOp:
    """A linear operator L = lam * I, given by its constant eigenvalues.

    With L = lam * I, each point is an independent ODE, so a closed-form
    solution of the ODE is the exact solution of the semi-discrete problem.
    """

    def __init__(self, lam, n, dtype=jnp.float64):
        self.eigenvalues = jnp.full(n, lam, dtype=dtype)


# Stiff nonlinear test problem with a known solution (a nonlinear
# Prothero-Robinson problem): u' = -a u - u^2 + a phi + phi' + phi^2 with
# phi(t) = cos(t + x). The exact solution for u(0) = phi(0) is u = phi.
PR_A = 5.0
PR_N = 8
PR_T = 4.0


def pr_problem(dtype=jnp.float64):
    x = jnp.linspace(0.0, 1.0, PR_N, dtype=dtype)

    def rhs(state, t):
        phi = jnp.cos(t + x)
        dphi = -jnp.sin(t + x)
        u = state["u"]
        return {"u": (-u * u + PR_A * phi + dphi + phi * phi).astype(dtype)}

    u0 = {"u": jnp.cos(x)}
    exact = np.cos(PR_T + np.asarray(x, dtype=np.float64))
    return u0, {"u": ConstantOp(-PR_A, PR_N, dtype)}, rhs, exact


# Bernoulli problem u' = -a u + u^2. Exact solution:
# u(t) = a / (1 + (a / u0 - 1) exp(a t)).
BE_A = 10.0


def bernoulli_problem():
    u0 = jnp.linspace(0.2, 0.8, PR_N)

    def rhs(state, t):
        return {"u": state["u"] ** 2}

    def exact(t):
        u0n = np.asarray(u0)
        return BE_A / (1.0 + (BE_A / u0n - 1.0) * np.exp(BE_A * t))

    return {"u": u0}, {"u": ConstantOp(-BE_A, PR_N)}, rhs, exact


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_compiled_drivers()
    yield
    clear_compiled_drivers()


class TestFixedDtMatchesFixedStepDriver:
    """With the controller held at dt0, the result is the fixed-step result.

    A very loose tolerance makes the controller ask for growth, and
    dt_max = dt0 stops that growth, so each step has the size dt0. An
    accepted step keeps the two half steps, so the reference is
    etd_integrate with dt0 / 2.
    """

    @pytest.mark.parametrize("method", ["etd1", "etdrk4"])
    def test_final_state_and_history(self, method):
        grid = Grid1D.uniform(32, x_min=0.0, x_max=1.0)
        x = grid.x_coords(include_ghost=False)
        u0 = {"u": 0.5 + 0.4 * jnp.sin(2 * jnp.pi * x)}
        ops = {"u": DiffusionOperator(grid, 0.01)}

        def rhs(state, t):
            u = state["u"]
            return {"u": u - u ** 3}

        dt0, t_end = 0.05, 0.5
        pid = PIDParams(atol=1e8, rtol=1e8, dt_max=dt0)
        res = etd_integrate_adaptive(u0, (0.0, t_end), dt0, ops, rhs, method=method,
                                     pid_params=pid, save_every=3, max_steps=50)

        assert bool(res.success)
        assert int(res.n_accepted) == 10
        assert int(res.n_rejected) == 0
        assert float(res.t_final) == t_end

        # Saves at accepted steps 3, 6 and 9, and the final step 10.
        n_saved = int(res.n_saved)
        assert n_saved == 5
        np.testing.assert_allclose(np.asarray(res.t_history[:n_saved]),
                                   [0.0, 0.15, 0.3, 0.45, 0.5], rtol=0, atol=1e-14)
        np.testing.assert_allclose(np.asarray(res.dt_history[1:n_saved]), dt0, rtol=1e-12)

        # Fixed step dt0 / 2: the states after 6, 12, 18 and 20 half steps.
        _, hist = etd_integrate(u0, (0.0, t_end), dt0 / 2, ops, rhs, method=method,
                                save_every=1)
        reference = [hist[k]["u"] for k in (0, 6, 12, 18, 20)]
        for i, ref in enumerate(reference):
            np.testing.assert_allclose(np.asarray(res.y_history["u"][i]), np.asarray(ref),
                                       rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(np.asarray(res.y_final["u"]), np.asarray(reference[-1]),
                                   rtol=1e-12, atol=1e-14)


class TestErrorFollowsTolerance:
    """The global error decreases under tolerance refinement.

    For a per-step error control with an O(dt^(p+1)) estimate, the step
    size scales as tol^(1/(p+1)) and the global error as tol^(p/(p+1)):
    a slope of 0.5 for ETD1 and 0.8 for ETDRK4. The reference is the
    exact solution of the nonlinear Prothero-Robinson problem.
    """

    @pytest.mark.parametrize("method, tols, slope_range", [
        ("etd1", (1e-4, 1e-5, 1e-6), (0.4, 0.65)),
        ("etdrk4", (1e-5, 1e-6, 1e-7, 1e-8), (0.65, 1.05)),
    ])
    def test_order_trend(self, method, tols, slope_range):
        u0, ops, rhs, exact = pr_problem()
        errors = []
        for tol in tols:
            res = etd_integrate_adaptive(u0, (0.0, PR_T), 0.01, ops, rhs, method=method,
                                         pid_params=PIDParams(atol=tol, rtol=tol, dt_max=10.0),
                                         max_steps=20000)
            assert bool(res.success), int(res.status)
            assert float(res.t_final) == PR_T
            errors.append(float(np.max(np.abs(np.asarray(res.y_final["u"]) - exact))))

        assert all(b < a for a, b in zip(errors, errors[1:], strict=False)), errors
        slope = np.polyfit(np.log(tols), np.log(errors), 1)[0]
        assert slope_range[0] <= slope <= slope_range[1], (slope, errors)


class TestCompilesOnce:
    """A sweep over dt0 and t_end uses one compiled loop."""

    def test_sweep_over_dt0_and_t_end(self):
        u0, ops, rhs_inner, exact_fn = bernoulli_problem()
        traces = []

        def rhs(state, t):
            # Python code runs only while JAX traces the function.
            traces.append(1)
            return rhs_inner(state, t)

        pid = PIDParams(atol=1e-8, rtol=1e-8, dt_max=10.0)
        first = None
        for dt0 in (0.01, 0.03, 0.2):
            for t_end in (0.5, 1.0, 1.7):
                res = etd_integrate_adaptive(u0, (0.0, t_end), dt0, ops, rhs,
                                             method="etdrk4", pid_params=pid)
                if first is None:
                    first = len(traces)
                assert bool(res.success)
                assert float(res.t_final) == t_end
                np.testing.assert_allclose(np.asarray(res.y_final["u"]), exact_fn(t_end),
                                           rtol=1e-5, atol=1e-10)

        assert first > 0
        assert len(traces) == first
        assert len(fi._ETD_ADAPTIVE_DRIVERS) == 1

    def test_changed_controller_parameters_build_a_new_loop(self):
        u0, ops, rhs, _ = bernoulli_problem()
        for tol in (1e-6, 1e-7):
            etd_integrate_adaptive(u0, (0.0, 0.5), 0.01, ops, rhs, method="etd1",
                                   pid_params=PIDParams(atol=tol, rtol=tol))
        assert len(fi._ETD_ADAPTIVE_DRIVERS) == 2


class TestRejections:
    """The rejection path: error rejections, the budget, dt_min, non-finite steps."""

    def test_large_dt0_is_rejected_and_the_run_recovers(self):
        u0, ops, rhs, exact_fn = bernoulli_problem()
        t_end = 1.0
        res = etd_integrate_adaptive(u0, (0.0, t_end), t_end, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=1e-9, rtol=1e-9, dt_max=10.0))
        assert bool(res.success)
        assert int(res.n_rejected) >= 1
        assert float(res.t_final) == t_end
        np.testing.assert_allclose(np.asarray(res.y_final["u"]), exact_fn(t_end),
                                   rtol=1e-6, atol=1e-12)

    def test_rejection_budget_stops_the_run(self):
        u0, ops, rhs, _ = bernoulli_problem()
        res = etd_integrate_adaptive(u0, (0.0, 1.0), 1.0, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=1e-12, rtol=1e-12, dt_max=10.0),
                                     max_rejections_per_step=1)
        assert int(res.status) == StatusCode.MAX_ATTEMPTS_REACHED
        assert not bool(res.success)
        assert int(res.n_accepted) == 0
        assert int(res.n_rejected) == 1
        assert float(res.t_final) == 0.0
        np.testing.assert_array_equal(np.asarray(res.y_final["u"]), np.asarray(u0["u"]))

    def test_dt_min_stops_the_run(self):
        u0, ops, rhs, _ = bernoulli_problem()
        res = etd_integrate_adaptive(u0, (0.0, 1.0), 1.0, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=1e-12, rtol=1e-12,
                                                          dt_min=0.5, dt_max=10.0))
        assert int(res.status) == StatusCode.DT_TOO_SMALL
        assert not bool(res.success)

    def test_max_steps_stops_the_run(self):
        u0, ops, rhs, _ = bernoulli_problem()
        res = etd_integrate_adaptive(u0, (0.0, 1.0), 0.01, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=1e-8, rtol=1e-8), max_steps=3)
        assert int(res.status) == StatusCode.MAX_STEPS_REACHED
        assert int(res.n_accepted) == 3
        assert float(res.t_final) < 1.0

    @staticmethod
    def gompertz():
        """u' = -u ln(u) with L = 0. Exact: u(t) = exp(ln(u0) exp(-t)).

        With L = 0, ETD1 is the explicit Euler method. From u0 = 2, a half
        step of 2 gives 2 - 2 * 2 ln(2) = -0.77, and ln of a negative
        value is NaN. So the first attempt with dt0 = 4 is non-finite.
        """
        n = 4
        u0 = {"u": jnp.full(n, 2.0)}

        def rhs(state, t):
            u = state["u"]
            return {"u": -u * jnp.log(u)}

        def exact(t):
            return np.exp(np.log(2.0) * np.exp(-t))

        return u0, {"u": ConstantOp(0.0, n)}, rhs, exact

    def test_non_finite_step_is_rejected(self):
        u0, ops, rhs, exact_fn = self.gompertz()
        # Hand check of the first attempt (see gompertz).
        assert 2.0 - 2.0 * 2.0 * np.log(2.0) < 0.0

        # With this loose tolerance, no finite step fails the error test,
        # so each rejection is a non-finite step.
        res = etd_integrate_adaptive(u0, (0.0, 5.0), 4.0, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=1e3, rtol=1e3, dt_max=10.0))
        assert bool(res.success)
        assert int(res.n_rejected) >= 1
        assert float(res.t_final) == 5.0
        n_saved = int(res.n_saved)
        assert np.all(np.isfinite(np.asarray(res.y_history["u"][:n_saved])))

        # With a tight tolerance the run is also accurate.
        res = etd_integrate_adaptive(u0, (0.0, 5.0), 4.0, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=1e-7, rtol=1e-7, dt_max=10.0),
                                     max_steps=20000)
        assert bool(res.success)
        assert int(res.n_rejected) >= 1
        np.testing.assert_allclose(np.asarray(res.y_final["u"]), exact_fn(5.0), rtol=1e-3)


class TestDtypes:
    """The state keeps its dtype; the clock is float64 under x64."""

    @pytest.mark.parametrize("dtype, tol, bound", [
        (jnp.float32, 1e-5, 2e-4),
        (jnp.float64, 1e-7, 1e-6),
    ])
    def test_state_dtype_and_accuracy(self, dtype, tol, bound):
        u0, ops, rhs, exact = pr_problem(dtype)
        res = etd_integrate_adaptive(u0, (0.0, PR_T), 0.01, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=tol, rtol=tol, dt_max=10.0))
        assert bool(res.success)
        assert res.y_final["u"].dtype == dtype
        assert res.y_history["u"].dtype == dtype
        assert res.t_final.dtype == jnp.float64
        assert float(res.t_final) == PR_T
        err = float(np.max(np.abs(np.asarray(res.y_final["u"], dtype=np.float64) - exact)))
        assert err < bound, err


class TestArguments:
    """Arguments that the driver cannot use raise ValueError."""

    @pytest.mark.parametrize("kwargs", [
        {"method": "etd2"},
        {"method": "rk4"},
        {"save_every": 0},
        {"max_steps": 0},
        {"max_rejections_per_step": 0},
    ])
    def test_bad_static_arguments(self, kwargs):
        u0, ops, rhs, _ = bernoulli_problem()
        with pytest.raises(ValueError):
            etd_integrate_adaptive(u0, (0.0, 1.0), 0.1, ops, rhs, **kwargs)

    @pytest.mark.parametrize("t_span, dt0", [((1.0, 1.0), 0.1), ((1.0, 0.0), 0.1),
                                             ((0.0, 1.0), 0.0), ((0.0, 1.0), -0.1)])
    def test_bad_span_or_dt0(self, t_span, dt0):
        u0, ops, rhs, _ = bernoulli_problem()
        with pytest.raises(ValueError):
            etd_integrate_adaptive(u0, t_span, dt0, ops, rhs)
