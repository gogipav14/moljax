"""Regression tests for conditioning-diagnostic failure modes.

Each test here pins a defect that would otherwise return a confident but
wrong answer:

  1. an enclosing disk that does not enclose its own boundary at small scale,
  2. an unconverged eigensolve accepted as a numerical-range support point,
  3. a diagnostic run that reports success after the implicit solve failed.

All three corrupt the adequacy verdict rather than raising, so they are the
failure modes worth gating in CI.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmarks.conditioning_decision_demo import (  # noqa: E402
    DemoConfig,
    run_decision_demo,
)
from moljax.conditioning._geometry import _smallest_enclosing_disk  # noqa: E402
from moljax.conditioning.field_of_values import numerical_range  # noqa: E402


class TestEnclosingDiskScaleInvariance:
    """The disk must enclose its points at every magnitude, not just near one."""

    @pytest.mark.parametrize("exponent", [-12, -10, -8, -4, 0, 4, 8, 12])
    def test_disk_encloses_boundary(self, exponent: int) -> None:
        scale = 10.0**exponent
        points = np.exp(2j * np.pi * np.arange(7) / 7) * scale
        center, radius = _smallest_enclosing_disk(points)
        worst = max(abs(point - center) for point in points)
        # A disk that misses its own boundary understates the radius, and with
        # it the disk rate the verdict is read from.
        assert worst <= radius * (1.0 + 1.0e-9)

    @pytest.mark.parametrize("exponent", [-12, -6, 0, 6, 12])
    def test_radius_scales_linearly(self, exponent: int) -> None:
        scale = 10.0**exponent
        points = np.exp(2j * np.pi * np.arange(9) / 9) * scale
        _, radius = _smallest_enclosing_disk(points)
        assert radius == pytest.approx(scale, rel=1.0e-9)

    def test_collinear_points_still_enclosed(self) -> None:
        points = np.asarray([0.0 + 0.0j, 1e-10 + 0.0j, 2e-10 + 0.0j])
        center, radius = _smallest_enclosing_disk(points)
        assert max(abs(p - center) for p in points) <= radius * (1.0 + 1.0e-9)


class TestSupportConvergenceIsChecked:
    """An unconverged support point must be refused, not returned."""

    @staticmethod
    def _diagonal(n: int):
        diag = jnp.asarray(np.linspace(0.01, 1.2, n))
        return (lambda v: diag * v), (lambda v: jnp.conj(diag) * v)

    def test_truncated_budget_raises(self) -> None:
        matvec, adjoint = self._diagonal(120)
        with pytest.raises(RuntimeError, match="did not converge"):
            numerical_range(matvec, adjoint, 120, n_angles=8, max_iters=1)

    @pytest.mark.slow
    def test_converged_budget_recovers_known_interval(self) -> None:
        n = 120
        matvec, adjoint = self._diagonal(n)
        result = numerical_range(matvec, adjoint, n, n_angles=24, max_iters=120)
        # W(A) of a real diagonal operator is the interval [0.01, 1.2].
        assert result.radius == pytest.approx(0.595, abs=5.0e-3)
        assert result.center.real == pytest.approx(0.605, abs=5.0e-3)
        assert result.max_support_residual < 1.0e-3
        assert not result.origin_enclosed


class TestFailedImplicitStepIsNotReportedComplete:
    """A failed Newton solve must not be diagnosed or reported as success."""

    @pytest.mark.slow
    def test_nonconverged_newton_marks_run_failed(self, tmp_path) -> None:
        result = run_decision_demo(
            DemoConfig(
                nx=8,
                ny=8,
                dt=2.0,
                n_states=1,
                n_angles=4,
                fov_max_iters=60,
                arnoldi_steps=3,
                pseudospectrum_points=3,
                overhead_runs=2,
                max_newton_iters=0,
                max_krylov_iters=18,
                figure_dir=str(tmp_path),
            )
        )
        assert result["status"] == "failed"
        assert result["implicit_step_failures"]
        assert result["implicit_step_failures"][0]["converged"] is False
        # No verdict may be emitted for a state the solver never reached.
        assert result["states"] == []


class TestSampledBoundaryIsNotTreatedAsEnclosure:
    """A coarse sweep must not certify an operator whose range contains zero.

    Johnson support points form an *inscribed* polygon of the numerical range.
    Fitting the disk to them understates the radius and testing the origin
    against their hull understates enclosure, so a coarse sweep could certify
    an operator whose range contains the origin.  The disk is therefore fitted
    to the half-plane intersection, which contains the range: since
    ``0 in W`` and ``W subset disk(c, R)`` force ``|c| <= R``, a correct outer
    bound always yields ``disk_rate >= 1`` and can never be adequate.
    """

    @staticmethod
    def _origin_containing_operator(m: int = 24):
        # Eigenvalues on a circle of radius 1 about a centre of modulus 0.9, so
        # the origin lies strictly inside W(A).  The centre sits at -45 degrees
        # so the closest approach to the origin falls between the directions a
        # four-angle sweep samples.
        centre = 0.9 * np.exp(-1j * np.pi / 4)
        diag = jnp.asarray(centre + np.exp(2j * np.pi * np.arange(m) / m))
        return (lambda v: diag * v), (lambda v: jnp.conj(diag) * v), centre

    @pytest.mark.slow
    @pytest.mark.parametrize("n_angles", [4, 6, 8, 32])
    def test_origin_containing_range_is_never_adequate(self, n_angles: int) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner
        from moljax.conditioning.pseudospectra import arnoldi, ritz_values

        m = 24
        matvec, adjoint, centre = self._origin_containing_operator(m)
        result = numerical_range(matvec, adjoint, m, n_angles=n_angles, max_iters=150)
        v0 = jnp.asarray(np.random.default_rng(0).standard_normal(m) + 0j)
        ritz = ritz_values(arnoldi(matvec, v0, 12)[1])
        assessment = assess_preconditioner(result, ritz, epsilon_zero=float(abs(centre)))
        # The outer bound must never understate a range that contains zero.
        assert result.disk_rate >= 1.0
        assert assessment.verdict != "adequate"

    @pytest.mark.slow
    def test_outer_bound_converges_from_above(self) -> None:
        m = 24
        matvec, adjoint, _ = self._origin_containing_operator(m)
        coarse = numerical_range(matvec, adjoint, m, n_angles=4, max_iters=150)
        fine = numerical_range(matvec, adjoint, m, n_angles=32, max_iters=150)
        # Refining directions may only tighten a genuine outer bound.
        assert coarse.disk_rate >= fine.disk_rate

    @pytest.mark.slow
    def test_origin_outside_is_still_certified(self) -> None:
        """The conservative rule must not destroy true negatives."""
        m = 24
        diag = jnp.asarray(np.linspace(0.8, 1.2, m))
        result = numerical_range(
            lambda v: diag * v, lambda v: jnp.conj(diag) * v, m,
            n_angles=16, max_iters=150,
        )
        assert not result.origin_enclosed
        assert result.disk_rate < 1.0


class TestOutlierGateCanActuallyReject:
    """The outlier count must be measured against an independent model.

    ``fov.center``/``fov.radius`` describe a disk that provably contains the
    whole numerical range, and Ritz values of an Arnoldi compression always lie
    in that range, so measured against it the count is identically zero and
    ``max_right_real_outliers`` could never reject anything.
    """

    @pytest.mark.slow
    def test_planted_rightmost_ritz_value_is_rejected(self) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner
        from moljax.conditioning.pseudospectra import arnoldi, ritz_values

        m = 24
        # Tight bulk on [1.00, 1.23] with one far-right eigenvalue planted.
        diag = jnp.asarray(np.concatenate([1.0 + 0.01 * np.arange(m - 1), [6.0]]) + 0j)
        matvec = lambda v: diag * v  # noqa: E731
        adjoint = lambda v: jnp.conj(diag) * v  # noqa: E731
        fov = numerical_range(matvec, adjoint, m, n_angles=16, max_iters=200)
        v0 = jnp.asarray(np.random.default_rng(0).standard_normal(m) + 0j)
        ritz = ritz_values(arnoldi(matvec, v0, 14)[1])

        assessment = assess_preconditioner(fov, ritz, epsilon_zero=0.9)
        assert assessment.n_right_real_outliers >= 1
        assert assessment.verdict != "adequate"

    def test_enclosing_disk_would_be_vacuous(self) -> None:
        """Pin the reason the gate must not use the enclosing disk."""
        from moljax.conditioning._geometry import _smallest_enclosing_disk
        from moljax.conditioning.non_normality import right_real_outliers

        ritz = jnp.asarray(np.concatenate([np.full(10, 1.0), [5.0]]) + 0j)
        center, radius = _smallest_enclosing_disk(np.asarray(ritz))
        # Measured against a disk that encloses every Ritz value, nothing can
        # ever exceed the right edge.
        assert right_real_outliers(ritz, center, radius) == 0


class TestSupportSurvivesAnAdversarialStartBlock:
    """A start block aligned against the dominant eigenspace must not win.

    LOBPCG explores only the space its initial block generates.  A block
    orthogonal to the dominant invariant subspace can converge to a
    subdominant eigenpair with a near-zero residual, which the residual gate
    would accept, understating the support and breaking the outer bound.  The
    block therefore carries a seeded random direction.
    """

    @pytest.mark.slow
    def test_orthogonal_start_block_still_finds_the_dominant_mode(self) -> None:
        n = 16
        theta = 0.0
        init = np.sin(np.arange(n) + theta + 1.0) + 1j * np.cos(
            np.arange(n) + 0.5 * theta + 0.5
        )
        first = np.concatenate([init.real, init.imag])
        second = np.roll(first, 1)

        def constraint_rows(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            real_part, imag_part = vector[:n], vector[n:]
            return (
                np.concatenate([real_part, imag_part]),
                np.concatenate([imag_part, -real_part]),
            )

        rows = np.vstack([*constraint_rows(first), *constraint_rows(second)])
        null_space = np.linalg.svd(rows)[2][4:]
        candidate = null_space[0]
        dominant = candidate[:n] + 1j * candidate[n:]
        dominant /= np.linalg.norm(dominant)

        projector = np.outer(dominant, dominant.conj())
        operator = 10.0 * projector + 1.0 * (np.eye(n) - projector)
        matrix = jnp.asarray(operator)
        result = numerical_range(
            lambda v: matrix @ v,
            lambda v: jnp.conj(matrix).T @ v,
            n,
            n_angles=8,
            max_iters=200,
        )
        # The dominant eigenvalue is 10; a missed mode would report near 1.
        assert float(np.real(np.asarray(result.boundary)[0])) == pytest.approx(
            10.0, abs=1.0e-6
        )


class TestRealBulkOutlierDetection:
    """The right-real gate must not depend on complex clustering succeeding.

    ``_bulk_disk`` requires a candidate outlier set to be near-real relative to
    the bulk radius, and falls back to a disk enclosing every Ritz value when
    none qualifies.  Counting beyond that fallback always returns zero, so a
    bulk-model failure would read as "no outliers" instead of "no model".
    """

    @pytest.mark.parametrize(
        ("name", "values", "expect_outliers"),
        [
            # A real outlier hidden behind a farther complex pair.
            ("real outlier plus complex pair",
             np.concatenate([np.full(17, 1.0), [5.0], [6 + 0.1j, 6 - 0.1j]]), True),
            # A degenerate bulk whose radius is zero under roundoff imaginary parts.
            ("roundoff imaginary bulk",
             np.concatenate([np.full(10, 1.0), [5 + 1e-16j]]), True),
            ("plain real outlier",
             np.concatenate([np.full(10, 1.0), [5.0]]), True),
            # Legitimately tight clusters must not flag their own noise.
            ("tight legitimate cluster", np.linspace(1.0, 1.01, 20) + 0j, False),
            ("identical bulk with roundoff imag", np.full(20, 1.0) + 1e-16j, False),
            ("identical bulk with roundoff real",
             1.0 + 1e-14 * np.arange(20) + 0j, False),
        ],
    )
    def test_detects_real_outliers_without_false_alarms(
        self, name: str, values: np.ndarray, expect_outliers: bool
    ) -> None:
        from moljax.conditioning.non_normality import real_bulk_outliers

        count = real_bulk_outliers(jnp.asarray(values.astype(np.complex128)))
        assert (count > 0) is expect_outliers, f"{name}: got {count}"

    def test_enclosing_fallback_would_hide_them(self) -> None:
        """Pin why the bulk disk cannot be reused for this gate."""
        from moljax.conditioning.non_normality import _bulk_disk, right_real_outliers

        values = jnp.asarray(
            np.concatenate([np.full(17, 1.0), [5.0], [6 + 0.1j, 6 - 0.1j]]).astype(
                np.complex128
            )
        )
        center, radius = _bulk_disk(values)
        assert right_real_outliers(values, center, radius, factor=3.0) == 0


class TestSupportsAreCorroboratedNotCertified:
    """No fixed start can prove it found a global maximum, so say so."""

    @pytest.mark.slow
    def test_dominant_mode_orthogonal_to_every_start_column(self) -> None:
        """Constrain against all three columns of the first start block."""
        n, theta = 24, 0.0
        pad = max(2 * n, 16)
        init = np.sin(np.arange(n) + theta + 1.0) + 1j * np.cos(
            np.arange(n) + 0.5 * theta + 0.5
        )
        first = np.pad(np.concatenate([init.real, init.imag]), (0, pad - 2 * n))
        second = np.roll(first, 1)
        key = jax.random.PRNGKey((int(theta * 1_000_003) + 0) & 0x7FFFFFFF)
        third = np.asarray(jax.random.normal(key, (pad,), dtype=jnp.float64))

        def constraint_rows(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            real_part, imag_part = vector[:n], vector[n : 2 * n]
            return (
                np.concatenate([real_part, imag_part]),
                np.concatenate([imag_part, -real_part]),
            )

        rows = np.vstack(
            [row for column in (first, second, third) for row in constraint_rows(column)]
        )
        dominant = np.linalg.svd(rows)[2][rows.shape[0] :][0]
        dominant = dominant[:n] + 1j * dominant[n:]
        dominant /= np.linalg.norm(dominant)

        operator = np.eye(n) + 9.0 * np.outer(dominant, dominant.conj())
        matrix = jnp.asarray(operator)
        result = numerical_range(
            lambda v: matrix @ v,
            lambda v: jnp.conj(matrix).T @ v,
            n,
            n_angles=8,
            max_iters=200,
        )
        assert float(np.real(np.asarray(result.boundary)[0])) == pytest.approx(
            10.0, abs=1.0e-6
        )

    def test_result_reports_corroboration(self) -> None:
        m = 16
        diag = jnp.asarray(np.linspace(0.8, 1.2, m))
        result = numerical_range(
            lambda v: diag * v, lambda v: jnp.conj(diag) * v, m,
            n_angles=8, max_iters=150, n_restarts=2,
        )
        assert result.supports_corroborated is True

    def test_uncorroborated_supports_cannot_be_adequate(self) -> None:
        """A cleared corroboration flag must force abstention."""
        from moljax.conditioning.non_normality import assess_preconditioner

        m = 16
        diag = jnp.asarray(np.linspace(0.8, 1.2, m))
        good = numerical_range(
            lambda v: diag * v, lambda v: jnp.conj(diag) * v, m,
            n_angles=8, max_iters=150,
        )
        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        assert assess_preconditioner(good, ritz, epsilon_zero=0.8).verdict == "adequate"
        doubtful = good._replace(supports_corroborated=False)
        assert (
            assess_preconditioner(doubtful, ritz, epsilon_zero=0.8).verdict
            == "indeterminate"
        )
