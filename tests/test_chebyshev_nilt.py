"""
Tests for the Chebyshev-family NILT methods: Talbot, Weeks, Gaver-Stehfest.
"""

import os
import subprocess
import sys
from fractions import Fraction
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from moljax.laplace.chebyshev_nilt import (
    adaptive_chebyshev_nilt,
    gaver_stehfest_method,
    gaver_stehfest_weights,
    laguerre_coefficients,
    talbot_contour,
    talbot_method,
    weeks_method,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def F_exp(s):
    return 1.0 / (s + 1.0)


def F_sin(s):
    return 1.0 / (s ** 2 + 1.0)


class TestTalbot:
    """Fixed Talbot contour of Weideman and Trefethen (2007)."""

    @pytest.mark.parametrize("n_points, tol", [(16, 1e-8), (32, 1e-10), (64, 1e-10)])
    def test_talbot_reproduces_exponential(self, n_points, tol):
        """exp(-t): 6e-10 at N = 16, rounding level at N = 32 and 64."""
        t = jnp.array([0.5, 1.0, 2.0])
        res = talbot_method(F_exp, t, n_points=n_points)
        err = float(jnp.max(jnp.abs(res.f - jnp.exp(-t))))
        assert err < tol, err

    def test_talbot_reproduces_sine(self):
        """sin t at N = 32 to 1e-9 (poles on the imaginary axis)."""
        t = jnp.array([0.5, 1.0, 2.0, 3.0])
        res = talbot_method(F_sin, t, n_points=32)
        assert float(jnp.max(jnp.abs(res.f - jnp.sin(t)))) < 1e-9

    def test_talbot_nonpositive_times_are_zero(self):
        res = talbot_method(F_exp, jnp.array([-1.0, 0.0, 1.0]), n_points=16)
        assert float(res.f[0]) == 0.0
        assert float(res.f[1]) == 0.0
        assert abs(float(res.f[2]) - np.exp(-1.0)) < 1e-8

    def test_adaptive_talbot_converges(self):
        """Refinement stops once successive results agree and reports the measured difference."""
        t = jnp.linspace(0.1, 5.0, 20)
        res = adaptive_chebyshev_nilt(F_exp, t, method='talbot', tol=1e-10)
        assert float(jnp.max(jnp.abs(res.f - jnp.exp(-t)))) < 1e-8
        assert res.error_estimate < 1e-10
        assert res.n_terms <= 64


class TestWeeks:
    """Weeks' method with the coefficients of Weideman (1999)."""

    @pytest.mark.parametrize("n_terms", [31, 32])
    def test_weeks_reproduces_exponential(self, n_terms):
        """1/(s+1) with sigma = 0.5, b = 1: rounding level for odd and even term counts."""
        t = jnp.array([0.5, 1.0, 2.0, 4.0])
        res = weeks_method(F_exp, n_terms, t, sigma=0.5, b=1.0)
        assert float(jnp.max(jnp.abs(res.f - jnp.exp(-t)))) < 1e-8

    @pytest.mark.parametrize("sigma, b", [(0.0, 1.0), (0.0, 0.5), (0.5, 2.0)])
    def test_weeks_parameter_choices(self, sigma, b):
        t = jnp.array([0.5, 1.0, 2.0])
        res = weeks_method(F_exp, 32, t, sigma=sigma, b=b)
        assert float(jnp.max(jnp.abs(res.f - jnp.exp(-t)))) < 1e-8

    def test_laguerre_coefficients_are_geometric_for_exponential(self):
        """For 1/(s+1), sigma = 0.5, b = 1 the map gives G(z) = 0.8/(1 - 0.2 z), so a_n = 0.8 (0.2)^n."""
        a = laguerre_coefficients(F_exp, 12, sigma=0.5, b=1.0)
        expected = 0.8 * 0.2 ** jnp.arange(12)
        assert float(jnp.max(jnp.abs(a - expected))) < 1e-14


class TestTalbotRemovableSingularity:
    """theta = 0 is on the midpoint grid whenever n_points is odd.

    talbot_contour evaluated cot(alpha theta) there directly, so every
    contour point came out NaN for odd n_points; n_points = 34 produced a
    NaN error estimate on its own, because the estimate halves the point
    count and 17 is odd. Neither count was prohibited by the API.
    """

    @pytest.mark.parametrize("n_points", [31, 32, 33, 34])
    def test_every_point_count_inverts_and_estimates(self, n_points):
        t = jnp.array([0.5, 1.0, 2.0])
        res = talbot_method(F_exp, t, n_points=n_points)
        assert bool(jnp.all(jnp.isfinite(res.f))), res.f
        assert float(jnp.max(jnp.abs(res.f - jnp.exp(-t)))) < 1e-9
        assert np.isfinite(res.error_estimate), res.error_estimate
        assert res.error_estimate < 1e-8

    def test_the_contour_is_finite_at_theta_zero(self):
        """The grid point at theta = 0 takes the analytic limits
        theta cot(alpha theta) -> 1/alpha and cot(u) - u/sin^2(u) -> 0, so
        s is real there and the weight is finite."""
        alpha, beta, delta = 0.6407, 0.5017, 0.6122
        n_points = 31
        s, w = talbot_contour(1.0, n_points, 0.0)
        assert bool(jnp.all(jnp.isfinite(s)))
        assert bool(jnp.all(jnp.isfinite(w)))

        mid = n_points // 2  # 2k + 1 = N here, so theta = 0
        expected_s = n_points * (beta / alpha - delta)
        assert float(jnp.real(s[mid])) == pytest.approx(expected_s, rel=1e-12)
        assert float(jnp.imag(s[mid])) == pytest.approx(0.0, abs=1e-12)

    def test_the_limit_branch_agrees_with_the_cotangent_nearby(self):
        """The small-|u| series and the direct evaluation must join
        smoothly: an odd contour differs from its even neighbours only by
        the usual quadrature error, not by a jump at theta = 0."""
        t = jnp.array([1.0])
        errors = [
            float(jnp.abs(talbot_method(F_exp, t, n_points=n).f[0] - jnp.exp(-1.0)))
            for n in (30, 31, 32)
        ]
        assert max(errors) < 1e-12, errors


class TestWeeksErrorEstimate:
    """The estimate has to carry the e^{sigma t} prefactor.

    weeks_method returned max |coeffs[-3:]|, the truncated tail of the
    Laguerre series before it is multiplied by e^{sigma t}. At sigma = 1 and
    t = 50 that factor is 5.2e21, so an estimate of 9.7e-15 was reported for
    a result that was wrong by 9.0e4.
    """

    def test_estimate_exposes_an_unusable_parameter_choice(self):
        res = weeks_method(F_exp, 32, jnp.array([50.0]))
        true_error = abs(float(res.f[0]) - np.exp(-50.0))
        assert true_error == pytest.approx(9.0057e4, rel=1e-3)
        # The estimate must at least admit that the answer is worthless.
        assert res.error_estimate >= true_error
        assert res.error_estimate >= abs(float(res.f[0]))

    def test_estimate_stays_tight_on_well_chosen_parameters(self):
        """sigma = 0.5, b = 1 on 1/(s+1): the estimate must still bound the
        true error without becoming useless."""
        t = jnp.array([0.5, 1.0, 2.0, 4.0])
        res = weeks_method(F_exp, 32, t, sigma=0.5, b=1.0)
        true_error = float(jnp.max(jnp.abs(res.f - jnp.exp(-t))))
        assert res.error_estimate >= true_error
        assert res.error_estimate < 1e-12

    def test_estimate_scales_with_the_prefactor(self):
        """Same coefficients, later time: the estimate grows by e^{sigma dt}."""
        early = weeks_method(F_exp, 32, jnp.array([1.0]), sigma=0.5, b=1.0)
        late = weeks_method(F_exp, 32, jnp.array([9.0]), sigma=0.5, b=1.0)
        assert late.error_estimate / early.error_estimate == pytest.approx(
            float(np.exp(0.5 * 8.0)), rel=1e-10
        )

    def test_roundoff_term_floors_the_estimate(self):
        """Even with a tail at exactly zero the estimate keeps the eps
        sum|a_n| roundoff term, so it is never reported as exact."""
        res = weeks_method(F_exp, 32, jnp.array([1.0]), sigma=0.5, b=1.0)
        assert res.error_estimate > 0.0


class TestGaverStehfest:
    """Gaver-Stehfest with exact rational weights and the x64 guard."""

    def test_weights_match_exact_rationals(self):
        """Stored float64 weights match an independent Fraction evaluation to 1e-13 relative."""
        n, m = 14, 7
        expected = []
        for k in range(1, n + 1):
            total = Fraction(0)
            for j in range((k + 1) // 2, min(k, m) + 1):
                total += Fraction(
                    j ** m * factorial(2 * j),
                    factorial(m - j) * factorial(j) * factorial(j - 1) * factorial(k - j) * factorial(2 * j - k),
                )
            expected.append((-1) ** (k + m) * total)

        w = gaver_stehfest_weights(n)
        assert isinstance(w, np.ndarray)
        assert w.dtype == np.float64
        assert w.shape == (n,)
        for wk, ek in zip(w, expected, strict=True):
            assert abs(wk - float(ek)) <= 1e-13 * abs(float(ek))
        assert float(np.max(np.abs(w))) > 1e8

    def test_weights_tabulated_n6(self):
        """The n = 6 weights are the tabulated integers (1, -49, 366, -858, 810, -270)."""
        np.testing.assert_array_equal(gaver_stehfest_weights(6), [1.0, -49.0, 366.0, -858.0, 810.0, -270.0])

    def test_gaver_stehfest_exponential_under_x64(self):
        res = gaver_stehfest_method(F_exp, jnp.array([0.5, 1.0, 2.0]), n_terms=14)
        assert abs(float(res.f[1]) - np.exp(-1.0)) < 1e-5
        assert res.diagnostics['weights'].dtype == np.float64
        assert res.n_terms == 14

    def test_odd_term_count_is_rounded_up(self):
        res = gaver_stehfest_method(F_exp, jnp.array([1.0]), n_terms=13)
        assert res.n_terms == 14
        assert abs(float(res.f[0]) - np.exp(-1.0)) < 1e-5

    def test_gaver_stehfest_requires_x64(self):
        """Without x64 the method raises a clear error instead of returning values off by O(1)."""
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "import jax.numpy as jnp\n"
            "from moljax.laplace.chebyshev_nilt import gaver_stehfest_method\n"
            "try:\n"
            "    gaver_stehfest_method(lambda s: 1 / (s + 1), jnp.array([1.0]), 14)\n"
            "except RuntimeError as e:\n"
            "    print('RAISED', e)\n"
        )
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=ROOT)
        out = subprocess.run(
            [sys.executable, '-c', code], capture_output=True, text=True, env=env, cwd=ROOT, check=True
        )
        assert 'RAISED' in out.stdout, out.stdout + out.stderr
        assert '64-bit precision' in out.stdout
        assert 'jax_enable_x64' in out.stdout
