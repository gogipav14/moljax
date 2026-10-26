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

        # Binary-exact times: the clock adds dt0 ten times with no rounding,
        # so no small extra step appears at the end.
        dt0, t_end = 0.0625, 0.625
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
                                   [0.0, 0.1875, 0.375, 0.5625, 0.625], rtol=0, atol=0)
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


def linear_l0_problem(lam, n=4, dtype=jnp.float64):
    """u' = lam * u with L = 0, so that ETD1 is explicit Euler and ETDRK4
    is classical RK4. The step-doubling difference is then a polynomial
    in z = lam * h that the tests compute by hand.
    """
    def rhs(state, t):
        return {"u": lam * state["u"]}

    return {"u": jnp.ones(n, dtype=dtype)}, {"u": ConstantOp(0.0, n, dtype)}, rhs


def rk4_factor(z):
    return 1.0 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24


class TestFinalStep:
    """The trial step is min(dt, remaining) and it never exceeds dt_max."""

    def test_shifted_times_without_x64(self):
        """At t0 = 1e6 in float32 the old final-step slack forced a step of
        0.25 > dt_max. With atol = 1e-4, that step failed ten times and the
        run stopped.
        """
        jax.config.update("jax_enable_x64", False)
        try:
            u0, ops, rhs = linear_l0_problem(1.0, dtype=jnp.float32)
            t0, t_end, dt0 = 1e6, 1e6 + 0.25, 0.0625
            for atol in (1.0, 1e-4):
                res = etd_integrate_adaptive(
                    u0, (t0, t_end), dt0, ops, rhs, method="etdrk4",
                    pid_params=PIDParams(atol=atol, rtol=0.0, dt_max=dt0),
                    rhs_uses_time=False)
                assert bool(res.success), int(res.status)
                assert int(res.n_accepted) == 4
                assert int(res.n_rejected) == 0
                assert float(res.t_final) == float(jnp.float32(t_end))
                n = int(res.n_saved)
                assert np.all(np.asarray(res.dt_history[1:n]) <= dt0)
                np.testing.assert_allclose(np.asarray(res.y_final["u"]), np.exp(0.25),
                                           rtol=1e-5)
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_shifted_times_with_x64(self):
        """At t0 = 1e15 the clock resolution is 0.125, equal to dt_max.

        The old slack grew with the accepted count and forced a step of
        0.75 after two steps. A loose tolerance holds dt at dt_max, which
        the clock can represent, so the run takes eight steps.
        """
        u0, ops, rhs = linear_l0_problem(1.0)
        t0, dt0 = 1e15, 0.125
        t_end = t0 + 1.0
        res = etd_integrate_adaptive(u0, (t0, t_end), dt0, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=1.0, rtol=0.0, dt_max=dt0),
                                     rhs_uses_time=False)
        assert bool(res.success), int(res.status)
        assert int(res.n_accepted) == 8
        assert float(res.t_final) == t_end
        n = int(res.n_saved)
        assert np.all(np.asarray(res.dt_history[1:n]) <= dt0)
        np.testing.assert_allclose(np.asarray(res.y_final["u"]), np.e, rtol=1e-6)


class TestStateDtypeIsChecked:
    @pytest.mark.parametrize("field", [jnp.full(4, 2), jnp.full(4, 2, dtype=jnp.int32),
                                       jnp.ones(4, dtype=bool)])
    def test_integer_and_boolean_states_raise(self, field):
        _, ops, rhs = linear_l0_problem(1.0)
        with pytest.raises(ValueError, match="floating dtype"):
            etd_integrate_adaptive({"u": field}, (0.0, 0.5), 0.1, ops, rhs)
        assert len(fi._ETD_ADAPTIVE_DRIVERS) == 0


class TestDtMin:
    """DT_TOO_SMALL comes only after a rejected trial at dt_min."""

    def test_trial_at_dt_min_is_taken(self):
        # ETD1 with L = 0 is explicit Euler: the error estimate is z^2 / 4 * u
        # with z = h. At h = 0.02 the ratio is 1e-4 / 6e-5 = 1.67 (reject);
        # at h = dt_min = 0.01 it is 2.5e-5 / 6e-5 = 0.42 (accept).
        u0, ops, rhs = linear_l0_problem(1.0, n=1)
        res = etd_integrate_adaptive(u0, (0.0, 0.05), 0.02, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=6e-5, rtol=0.0, dt_min=0.01))
        assert bool(res.success), int(res.status)
        assert int(res.n_rejected) >= 1
        assert float(res.t_final) == 0.05
        np.testing.assert_allclose(np.asarray(res.y_final["u"]), np.exp(0.05), rtol=1e-3)

    def test_rejected_trial_at_dt_min_stops_the_run(self):
        # atol = 1e-6: the ratio is 100 at h = 0.02 and 25 at h = 0.01.
        u0, ops, rhs = linear_l0_problem(1.0, n=1)
        res = etd_integrate_adaptive(u0, (0.0, 0.05), 0.02, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=1e-6, rtol=0.0, dt_min=0.01))
        assert int(res.status) == StatusCode.DT_TOO_SMALL
        assert int(res.n_rejected) == 2
        assert int(res.n_accepted) == 0


class TestZeroSolution:
    def test_zero_state_with_atol_zero(self):
        """error = 0 and scale = 0 give a zero contribution, not 0/0."""
        n = 4

        def rhs(state, t):
            return {"u": jnp.zeros_like(state["u"])}

        res = etd_integrate_adaptive({"u": jnp.zeros(n)}, (0.0, 1.0), 0.1,
                                     {"u": ConstantOp(-1.0, n)}, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=0.0, rtol=1e-3))
        assert bool(res.success), int(res.status)
        assert int(res.n_rejected) == 0
        assert np.isfinite(float(res.dt_final))
        np.testing.assert_array_equal(np.asarray(res.y_final["u"]), np.zeros(n))


class TestErrorScale:
    """The error scale and the controller order, against hand arithmetic."""

    def test_richardson_divisor_sets_the_acceptance_threshold(self):
        # RK4 on u' = -2u with h = 0.5: z = -1. The step-doubling
        # difference is R(z/2)^2 - R(z); the estimate is that over 2^4 - 1.
        lam, h = -2.0, 0.5
        diff = abs(rk4_factor(lam * h / 2) ** 2 - rk4_factor(lam * h))
        est = diff / 15.0
        u0, ops, rhs = linear_l0_problem(lam)

        # The ratio is est / atol: 0.8 (accept), then 1.25 (reject).
        res = etd_integrate_adaptive(u0, (0.0, h), h, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=est / 0.8, rtol=0.0, dt_max=h))
        assert int(res.n_accepted) == 1 and int(res.n_rejected) == 0
        res = etd_integrate_adaptive(u0, (0.0, h), h, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=est / 1.25, rtol=0.0, dt_max=h),
                                     max_rejections_per_step=1)
        assert int(res.n_accepted) == 0 and int(res.n_rejected) == 1

    @pytest.mark.parametrize("method, lam, h, p, step_factor", [
        ("etd1", -2.0, 0.1, 1, lambda z: 1.0 + z),
        ("etdrk4", -2.0, 0.5, 4, rk4_factor),
    ])
    def test_proposal_uses_order_p_plus_1(self, method, lam, h, p, step_factor):
        z = lam * h
        est = abs(step_factor(z / 2) ** 2 - step_factor(z)) / (2 ** p - 1)
        ratio = 0.5
        pid = PIDParams(atol=est / ratio, rtol=0.0, dt_max=100.0)
        q = p + 1
        # The PI formula with prev_err_ratio = 1 (the initial state).
        factor = pid.safety * ratio ** (-pid.kI / q) * (1.0 / ratio) ** (pid.kP / q)
        expected = h * min(max(factor, pid.min_factor), pid.max_factor)

        u0, ops, rhs = linear_l0_problem(lam)
        res = etd_integrate_adaptive(u0, (0.0, 10.0), h, ops, rhs, method=method,
                                     pid_params=pid, max_steps=1)
        assert int(res.status) == StatusCode.MAX_STEPS_REACHED
        assert int(res.n_accepted) == 1
        np.testing.assert_allclose(float(res.dt_final), expected, rtol=1e-9)


class TestControllerOrder:
    """The rejection fraction of ETD1 with the order p and with p + 1.

    Problem: the nonlinear Prothero-Robinson problem above (a = 5, 8
    points, t in [0, 4]), ETD1, dt0 = 0.01, atol = rtol = 1e-3,
    dt_max = 10. With the order p, the controller oscillates and about
    half of all attempts are rejections.
    """

    def counts(self, monkeypatch, offset):
        monkeypatch.setattr(fi, "_ETD_CONTROLLER_ORDER_OFFSET", offset)
        clear_compiled_drivers()
        u0, ops, rhs, _ = pr_problem()
        res = etd_integrate_adaptive(u0, (0.0, PR_T), 0.01, ops, rhs, method="etd1",
                                     pid_params=PIDParams(atol=1e-3, rtol=1e-3, dt_max=10.0),
                                     max_steps=20000)
        assert bool(res.success)
        return int(res.n_accepted), int(res.n_rejected)

    def test_rejection_fraction(self, monkeypatch):
        acc_p1, rej_p1 = self.counts(monkeypatch, 1)
        acc_p, rej_p = self.counts(monkeypatch, 0)
        print(f"order p + 1: accepted {acc_p1}, rejected {rej_p1}; "
              f"order p: accepted {acc_p}, rejected {rej_p}")
        assert rej_p1 / (acc_p1 + rej_p1) < 0.05
        assert rej_p / (acc_p + rej_p) > 0.3


class TestRelativeClock:
    """The clock keeps the elapsed time also at a large t0.

    An absolute clock lost elapsed time: in float32 at t0 = 1e6 the
    spacing is 0.0625, so t + 0.1 advanced the clock by 0.125, and the run
    reported SUCCESS after too few steps with a wrong value.

    The right-hand side is autonomous, and at these t0 the clock cannot
    resolve the stage times, so the runs declare rhs_uses_time=False.
    """

    @staticmethod
    def run(x64, t0, span, dt0, save_every=1):
        jax.config.update("jax_enable_x64", x64)
        try:
            dtype = jnp.float64 if x64 else jnp.float32
            u0, ops, rhs = linear_l0_problem(1.0, dtype=dtype)
            t_end = t0 + span
            res = etd_integrate_adaptive(u0, (t0, t_end), dt0, ops, rhs, method="etdrk4",
                                         pid_params=PIDParams(atol=1e-6, rtol=0.0, dt_max=dt0),
                                         save_every=save_every, rhs_uses_time=False)
            res = jax.tree.map(np.asarray, res)
            t_end_clock = float(jnp.asarray(t_end, dtype=dtype))
        finally:
            jax.config.update("jax_enable_x64", True)
        return res, t_end_clock

    @pytest.mark.parametrize("x64, t0, span, dt0, n_steps, rtol", [
        (False, 1e6, 0.25, 0.1, 3, 2e-6),
        (True, 1e15, 1.0, 0.2, 5, 1e-6),
    ])
    def test_reproductions(self, x64, t0, span, dt0, n_steps, rtol):
        res, t_end_clock = self.run(x64, t0, span, dt0)
        assert bool(res.success), int(res.status)
        assert int(res.n_accepted) == n_steps
        assert float(res.t_final) == t_end_clock
        np.testing.assert_allclose(res.y_final["u"], np.exp(span), rtol=rtol)
        n = int(res.n_saved)
        np.testing.assert_allclose(np.sum(res.dt_history[1:n], dtype=np.float64), span,
                                   rtol=4 * np.finfo(res.dt_history.dtype).eps)

    def test_matches_fixed_step_driver_at_large_t0(self):
        """x64 at t0 = 1e15 with dt0 = 0.2: the reference is the fixed-step
        driver with dt = 0.1 on [0, 1] (the problem is autonomous)."""
        res, _ = self.run(True, 1e15, 1.0, 0.2)
        u0, ops, rhs = linear_l0_problem(1.0)
        _, hist = etd_integrate(u0, (0.0, 1.0), 0.1, ops, rhs, method="etdrk4")
        np.testing.assert_allclose(res.y_final["u"], np.asarray(hist[-1]["u"]), rtol=1e-12)

    @pytest.mark.parametrize("x64, t0, span, dt0", [
        (False, 1e6, 0.25, 0.1),
        (True, 1e15, 1.0, 0.2),
        (True, 1e15, 1.0, 0.1),
    ])
    def test_history_holds_the_final_state(self, x64, t0, span, dt0):
        res, t_end_clock = self.run(x64, t0, span, dt0, save_every=3)
        assert bool(res.success), int(res.status)
        n = int(res.n_saved)
        assert float(res.t_history[n - 1]) == t_end_clock
        np.testing.assert_array_equal(res.y_history["u"][n - 1], res.y_final["u"])
        # Round-off of tau can leave one more very small step, never fewer.
        assert int(np.ceil(span / dt0 - 1e-9)) <= int(res.n_accepted) <= int(np.ceil(span / dt0)) + 1
        np.testing.assert_allclose(res.y_final["u"], np.exp(span), rtol=2e-6)

    def test_span_below_clock_resolution_raises(self):
        jax.config.update("jax_enable_x64", False)
        try:
            u0, ops, rhs = linear_l0_problem(1.0, dtype=jnp.float32)
            with pytest.raises(ValueError, match="clock resolution"):
                etd_integrate_adaptive(u0, (1e6, 1e6 + 0.01), 0.1, ops, rhs)
        finally:
            jax.config.update("jax_enable_x64", True)


class TestCompletionOnTheUpdatedClock:
    """tau + dt_step can round to the span when dt_step < remaining.

    Case: 0.01 + 0.01 + 0.01 = 0.03 and 0.04 - 0.03 > 0.01, but
    0.03 + 0.01 == 0.04. The run ended after four steps without a save
    at t_end.
    """

    def test_reported_case(self):
        u0, ops, rhs = linear_l0_problem(1.0, n=1)
        res = etd_integrate_adaptive(u0, (0.0, 0.04), 0.01, ops, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=1.0, rtol=0.0, dt_max=0.01),
                                     save_every=3)
        assert bool(res.success)
        assert int(res.n_accepted) == 4
        n = int(res.n_saved)
        assert n == 3
        assert float(res.t_history[n - 1]) == 0.04
        np.testing.assert_array_equal(np.asarray(res.y_history["u"][n - 1]),
                                      np.asarray(res.y_final["u"]))

    @pytest.mark.parametrize("dt0", [0.01, 0.03, 0.07, 0.1])
    def test_sweep_last_entry_is_at_t_end(self, dt0):
        # save_every is larger than any step count, so the history holds
        # only the initial state and the final save. The spans 0.04
        # (dt0 = 0.01), 0.27 (0.03), 4.76 (0.07) and 1.3 (0.1) are spans
        # where the sum of the steps rounds to the span before the final
        # step.
        u0, ops, rhs = linear_l0_problem(1.0, n=1)
        for t_end in (0.04, 0.1, 0.27, 0.7, 1.0, 1.3, 4.76):
            if t_end <= dt0:
                continue
            res = etd_integrate_adaptive(u0, (0.0, t_end), dt0, ops, rhs, method="etdrk4",
                                         pid_params=PIDParams(atol=1.0, rtol=0.0, dt_max=dt0),
                                         save_every=1000)
            assert bool(res.success), (dt0, t_end)
            n = int(res.n_saved)
            assert n == 2, (dt0, t_end, n)
            assert float(res.t_history[1]) == t_end, (dt0, t_end)
            np.testing.assert_array_equal(np.asarray(res.y_history["u"][1]),
                                          np.asarray(res.y_final["u"]))
            # After many steps the round-off of tau can exceed a few ulp and
            # leave one more very small step.
            assert int(res.n_accepted) <= int(np.ceil(t_end / dt0 - 1e-9)) + 1
            np.testing.assert_allclose(np.asarray(res.y_final["u"]), np.exp(t_end), rtol=1e-5)


class TestSpanCast:
    def test_changed_span_raises_without_x64(self):
        """In float32, (1000000.02, 1000000.04) is cast to (1000000.0,
        1000000.0625): the span changes from 0.02 to 0.0625."""
        jax.config.update("jax_enable_x64", False)
        try:
            n = 1
            u0 = {"u": jnp.zeros(n, dtype=jnp.float32)}

            def rhs(state, t):
                return {"u": jnp.ones_like(state["u"])}

            with pytest.raises(ValueError, match="clock resolution"):
                etd_integrate_adaptive(u0, (1000000.02, 1000000.04), 0.1,
                                       {"u": ConstantOp(0.0, n, jnp.float32)}, rhs,
                                       method="etd1")
        finally:
            jax.config.update("jax_enable_x64", True)


class TestStageTimes:
    """A time-dependent right-hand side needs resolvable stage times."""

    T0 = 1e15

    def rhs_time(self, state, t):
        return {"u": jnp.full_like(state["u"], t - self.T0)}

    def test_unresolved_stage_times_stop_the_run(self):
        # The clock spacing at 1e15 is 0.125, so t0 + 0.0625 rounds to t0
        # and the two half steps of ETD1 see the same time.
        n = 1
        res = etd_integrate_adaptive({"u": jnp.zeros(n)}, (self.T0, self.T0 + 0.125), 0.125,
                                     {"u": ConstantOp(0.0, n)}, self.rhs_time, method="etd1",
                                     pid_params=PIDParams(atol=1e-12, rtol=0.0, dt_max=0.125))
        assert int(res.status) == StatusCode.TIME_UNRESOLVED
        assert not bool(res.success)
        assert int(res.n_accepted) == 0
        np.testing.assert_array_equal(np.asarray(res.y_final["u"]), np.zeros(n))

    def test_resolved_stage_times_give_the_exact_value(self):
        # u' = t - t0 from u = 0: u(t0 + 1) = 1/2. At t0 = 1e6 with x64 the
        # spacing is about 1e-10, so the stage times are resolved.
        t0 = 1e6
        n = 1

        def rhs(state, t):
            return {"u": jnp.full_like(state["u"], t - t0)}

        res = etd_integrate_adaptive({"u": jnp.zeros(n)}, (t0, t0 + 1.0), 0.1,
                                     {"u": ConstantOp(0.0, n)}, rhs, method="etdrk4",
                                     pid_params=PIDParams(atol=1e-10, rtol=0.0))
        assert bool(res.success), int(res.status)
        np.testing.assert_allclose(np.asarray(res.y_final["u"]), 0.5, rtol=1e-8)

    def test_autonomous_rhs_at_large_t0(self):
        """u' = u at t0 = 1e15 with dt0 = 0.125 (the clock spacing there).

        With the default rhs_uses_time=True, the stage check stops the run.
        With rhs_uses_time=False the run succeeds, for a plain rhs and for
        jax.jit(rhs). (A trace-based detection of time use rejected the
        jitted rhs, because the outer equation lists t as an input.)
        """
        t0 = self.T0
        u0, ops, rhs = linear_l0_problem(1.0, n=1)
        pid = PIDParams(atol=1.0, rtol=0.0, dt_max=0.125)

        res = etd_integrate_adaptive(u0, (t0, t0 + 1.0), 0.125, ops, rhs, method="etdrk4",
                                     pid_params=pid)
        assert int(res.status) == StatusCode.TIME_UNRESOLVED

        for f in (rhs, jax.jit(rhs)):
            res = etd_integrate_adaptive(u0, (t0, t0 + 1.0), 0.125, ops, f, method="etdrk4",
                                         pid_params=pid, rhs_uses_time=False)
            assert bool(res.success), int(res.status)
            assert int(res.n_accepted) == 8
            np.testing.assert_allclose(np.asarray(res.y_final["u"]), np.e, rtol=1e-6)

    def test_time_dependent_rhs_default_is_checked(self):
        """The default rhs_uses_time=True also checks a jitted rhs."""
        n = 1
        res = etd_integrate_adaptive({"u": jnp.zeros(n)}, (self.T0, self.T0 + 0.125), 0.125,
                                     {"u": ConstantOp(0.0, n)}, jax.jit(self.rhs_time),
                                     method="etd1",
                                     pid_params=PIDParams(atol=1e-12, rtol=0.0, dt_max=0.125))
        assert int(res.status) == StatusCode.TIME_UNRESOLVED


class TestSpanCheckUnderJit:
    """The span check uses host arithmetic, so it also runs inside jax.jit."""

    def test_changed_span_raises_inside_jit(self):
        jax.config.update("jax_enable_x64", False)
        try:
            n = 1
            ops = {"u": ConstantOp(0.0, n, jnp.float32)}

            def rhs(state, t):
                return {"u": jnp.ones_like(state["u"])}

            @jax.jit
            def run(u):
                return etd_integrate_adaptive({"u": u}, (1000000.02, 1000000.04), 0.1, ops, rhs,
                                              method="etd1").y_final["u"]

            with pytest.raises(ValueError, match="clock resolution"):
                run(jnp.zeros(n, dtype=jnp.float32))
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_traced_t_span_raises(self):
        u0, ops, rhs = linear_l0_problem(1.0, n=1)

        @jax.jit
        def run(t_end):
            return etd_integrate_adaptive(u0, (0.0, t_end), 0.1, ops, rhs).y_final["u"]

        with pytest.raises(ValueError, match="t_span must be concrete"):
            run(1.0)

    def test_concrete_t_span_inside_jit_succeeds(self):
        u0, ops, rhs = linear_l0_problem(1.0, n=1)

        @jax.jit
        def run(u):
            return etd_integrate_adaptive({"u": u}, (0.0, 1.0), 0.1, ops, rhs,
                                          pid_params=PIDParams(atol=1e-10, rtol=0.0)).y_final["u"]

        np.testing.assert_allclose(np.asarray(run(u0["u"])), np.e, rtol=1e-8)



class TestStepResolvedInStateDtype:
    """A step that is zero in the state dtype must not give a false success."""

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
    def test_sixteen_bit_states_are_rejected(self, dtype):
        """float16 and bfloat16 states raise ValueError before compilation."""
        u0 = {"u": jnp.zeros(4, dtype=dtype)}
        ops = {"u": ConstantOp(0.0, 4)}
        with pytest.raises(ValueError, match="float32 or float64"):
            etd_integrate_adaptive(
                u0, (0.0, 1e-8), 1e-8, ops, lambda y, t: {"u": 1e4 + 0 * y["u"]},
                method="etd1", rhs_uses_time=False,
            )

    def test_step_below_float32_resolution_stops(self):
        """u' = 1e38 over (0, 1e-40) in float32: the step is not resolved, so the run fails.

        The exact result is 0.01. Before this check the driver returned SUCCESS
        with an unchanged state.
        """
        u0 = {"u": jnp.zeros(4, dtype=jnp.float32)}
        ops = {"u": ConstantOp(0.0, 4, jnp.float32)}
        result = etd_integrate_adaptive(
            u0, (0.0, 1e-40), 1e-12, ops, lambda y, t: {"u": jnp.full_like(y["u"], 1e38)},
            method="etd1", rhs_uses_time=False,
        )
        assert not bool(result.success)
        assert int(result.status) == int(StatusCode.DT_TOO_SMALL)


class TestSubnormalSpan:
    """A span below the smallest normal number of the clock dtype is rejected."""

    def _call(self, span, rhs_value, dtype):
        u0 = {"u": jnp.zeros(4, dtype=dtype)}
        ops = {"u": ConstantOp(0.0, 4, dtype)}
        return etd_integrate_adaptive(
            u0, (0.0, span), 1e-12, ops, lambda y, t: {"u": jnp.full_like(y["u"], rhs_value)},
            method="etd1", rhs_uses_time=False,
        )

    def test_subnormal_span_without_x64(self):
        """float32 clock: the span 2**-130 is subnormal and raises ValueError."""
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(ValueError, match="subnormal|smallest normal"):
                self._call(2.0**-130, 1e38, jnp.float32)
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_subnormal_span_with_x64(self):
        """float64 clock: the span 2**-1030 is subnormal and raises ValueError."""
        with pytest.raises(ValueError, match="subnormal|smallest normal"):
            self._call(2.0**-1030, 1e307, jnp.float64)


class TestLongRunCompletion:
    """Many steps at an ordinary time offset must end at t_end with SUCCESS."""

    @pytest.mark.parametrize("method", ["etd1", "etdrk4"])
    def test_thousand_steps_from_t0_1000(self, method):
        """x64, u' = 1 + 0.001 (t - 1000) on (1000, 1010), dt0 = dt_max = 0.01.

        The exact result is 10 + 0.001 * 10**2 / 2 = 10.05. Before the
        compensated clock, 1000 additions of 0.01 left a residue beyond the
        completion margin, and the run stopped with TIME_UNRESOLVED.
        """
        u0 = {"u": jnp.zeros(1)}
        ops = {"u": ConstantOp(0.0, 1)}
        result = etd_integrate_adaptive(
            u0, (1000.0, 1010.0), 0.01, ops,
            lambda y, t: {"u": jnp.ones_like(y["u"]) * (1.0 + 0.001 * (t - 1000.0))},
            method=method, pid_params=PIDParams(dt_max=0.01),
        )
        assert int(result.status) == int(StatusCode.SUCCESS)
        assert float(result.t_final) == 1010.0
        # ETD1 with L = 0 is explicit Euler (first order); ETDRK4 integrates a
        # forcing that is linear in t exactly.
        rtol = 1e-4 if method == "etd1" else 1e-9
        np.testing.assert_allclose(np.asarray(result.y_final["u"]), [10.05], rtol=rtol)


class TestSubnormalEndpoints:
    """A subnormal endpoint can be flushed to zero, so it is rejected."""

    def _call(self, half, rhs_value, dtype):
        u0 = {"u": jnp.zeros(4, dtype=dtype)}
        ops = {"u": ConstantOp(0.0, 4, dtype)}
        return etd_integrate_adaptive(
            u0, (-half, half), 1e-12, ops, lambda y, t: {"u": jnp.full_like(y["u"], rhs_value)},
            method="etd1", rhs_uses_time=False,
        )

    def test_subnormal_endpoints_without_x64(self):
        jax.config.update("jax_enable_x64", False)
        try:
            with pytest.raises(ValueError, match="subnormal"):
                self._call(2.0**-127, 1e38, jnp.float32)
        finally:
            jax.config.update("jax_enable_x64", True)

    def test_subnormal_endpoints_with_x64(self):
        with pytest.raises(ValueError, match="subnormal"):
            self._call(2.0**-1023, 1e307, jnp.float64)
