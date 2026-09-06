"""Regression tests for conditioning-diagnostic failure modes.

Each test here pins a defect that would otherwise return a confident but
wrong answer: an enclosing disk that does not enclose its own boundary at
small scale, an unconverged eigensolve accepted as a support point, a run
that reports success after the implicit solve failed, an inscribed boundary
read as an outer bound, an outlier gate that cannot reject or that a short
spectrum slips past, a reading outside its domain scored as a measurement,
a start block orthogonal to the dominant mode, and a verdict that hides
which checks were actually run.

All of them corrupt the verdict rather than raising, so they are the failure
modes worth gating in CI.
"""

from __future__ import annotations

import itertools
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from benchmarks.conditioning_decision_demo import (  # noqa: E402
    DemoConfig,
    run_decision_demo,
)
from moljax.conditioning._geometry import _origin_enclosed, _smallest_enclosing_disk  # noqa: E402
from moljax.conditioning.field_of_values import FieldOfValuesResult, numerical_range  # noqa: E402


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


class TestEnclosingDiskToleratesLargeOffsets:
    """A tight cluster far from the origin must resolve at its own scale.

    ``_smallest_enclosing_disk`` scaled its containment tolerance by the
    points' distance from the origin.  A cluster with a large common offset
    and a tiny spread -- a numerical range far from zero, say -- then read
    every point as already contained in an arbitrary early candidate disk,
    because the tolerance carried by that offset dwarfed the actual radius:
    a disk of true radius 1e-9 centered at 1e6 read back at radius 2e-9, a
    factor of two, before this fix. The fix scales the tolerance by the
    points' own spread instead, and translates to one of the points before
    the circumcenter arithmetic, which squares each coordinate and so loses
    most of its precision to cancellation when the coordinates share a large
    common offset.
    """

    def test_tight_cluster_far_from_origin_matches_a_shifted_reference(self) -> None:
        # float64 cannot represent "1e6 + 1e-9" any more precisely than
        # roughly one part in ulp(1e6) / 1e-9 =~ 12%: adding a radius far
        # below the offset's own rounding granularity perturbs the radius
        # actually realized by each stored point.  A high-precision
        # reference has to be built from the same translate-then-solve
        # arithmetic to be comparable; comparing straight to the
        # mathematically intended radius of 1e-9 would be judging the fix
        # against a target float64 cannot represent at this magnitude.
        theta = 2.0 * np.pi * np.arange(24) / 24
        points = 1.0e6 + 1.0e-9 * np.exp(1j * theta)
        center, radius = _smallest_enclosing_disk(points)

        origin = points[0]
        shifted = points - origin
        candidates = [(value, 0.0) for value in shifted]
        pairs = itertools.combinations(range(len(shifted)), 2)
        candidates.extend((0.5 * (shifted[i] + shifted[j]), abs(shifted[i] - shifted[j]) / 2.0) for i, j in pairs)
        feasible = [
            (candidate_center, candidate_radius)
            for candidate_center, candidate_radius in candidates
            if np.max(np.abs(shifted - candidate_center)) <= candidate_radius * (1.0 + 1.0e-9)
        ]
        reference_center, reference_radius = min(feasible, key=lambda item: item[1])

        assert radius == pytest.approx(reference_radius, rel=1.0e-9)
        assert abs(center - origin - reference_center) <= 1.0e-9 * reference_radius
        # Documents the achievable precision at this magnitude rather than
        # asserting an unrepresentable exact match: the true radius realized
        # by the stored points differs from the mathematically intended 1e-9
        # by close to ulp(1e6), so this stays well short of 1.0 (the old
        # code's factor-of-two error) without pretending float64 can do
        # better than its own precision at this offset.
        assert radius == pytest.approx(1.0e-9, rel=0.2)


class TestOriginEnclosedScaleInvariance:
    """The origin-in-hull test must not depend on the boundary's magnitude.

    ``_origin_enclosed`` floored its coordinate tolerance at a scale of 1.0,
    so a hull well under unit magnitude was judged against a tolerance far
    larger than every one of its coordinates, which can call the origin
    enclosed for a hull nowhere near it.
    """

    @pytest.mark.parametrize("scale", [1.0e-10, 1.0, 1.0e10])
    def test_hull_excluding_the_origin_is_reported_at_every_scale(self, scale: float) -> None:
        # The hull of these four points does not contain the origin (its
        # bounding box does, which is exactly the case a hull test, as
        # opposed to a bounding-box test, has to get right), regardless of
        # how small or large the whole configuration is scaled.
        points = scale * np.asarray([1.0 - 0.1j, -0.1 + 1.0j, 1.0 + 1.0j, 0.6 + 0.6j])
        assert _origin_enclosed(points) is False


class TestSupportConvergenceIsChecked:
    """An unconverged support point must be flagged, and never certified."""

    @staticmethod
    def _diagonal(n: int):
        diag = jnp.asarray(np.linspace(0.01, 1.2, n))
        return (lambda v: diag * v), (lambda v: jnp.conj(diag) * v)

    def test_truncated_budget_is_reported_not_raised(self) -> None:
        """Under-resolved supports return geometry, flagged, rather than raising.

        Refusing to return anything would discard a measurement a regime sweep
        can still use for relative comparison; what must not survive is the
        certification.
        """
        matvec, adjoint = self._diagonal(120)
        result = numerical_range(matvec, adjoint, 120, n_angles=8, max_iters=1)
        assert not result.supports_converged
        assert result.max_support_residual > 1.0e-3
        assert math.isfinite(result.disk_rate)

    def test_unconverged_supports_force_abstention(self) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner

        matvec, adjoint = self._diagonal(120)
        result = numerical_range(matvec, adjoint, 120, n_angles=8, max_iters=1)
        ritz = jnp.asarray(np.linspace(0.01, 1.2, 8) + 0j)
        assert (
            assess_preconditioner(result, ritz, epsilon_zero=0.9).verdict
            == "indeterminate"
        )

    def test_convergence_is_judged_against_the_requested_tolerance(self) -> None:
        """A stricter request must not be overruled by a downstream default.

        The convergence state is decided inside numerical_range, where the
        caller's tolerance is known.  Were the assessment to apply its own
        threshold instead, a caller asking for 1e-6 could be handed an
        adequate verdict on a residual that failed their request.
        """
        matvec, adjoint = self._diagonal(60)
        strict = numerical_range(
            matvec, adjoint, 60, n_angles=8, max_iters=120, residual_tolerance=1.0e-14
        )
        lenient = numerical_range(
            matvec, adjoint, 60, n_angles=8, max_iters=120, residual_tolerance=1.0e-2
        )
        # Same solve, same residual; only the requested tolerance differs.
        assert strict.max_support_residual == pytest.approx(
            lenient.max_support_residual, rel=1e-9
        )
        assert not strict.supports_converged
        assert lenient.supports_converged

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
        # Eigenvalues on a circle of radius 1 about a center of modulus 0.9, so
        # the origin lies strictly inside W(A).  The center sits at -45 degrees
        # so the closest approach to the origin falls between the directions a
        # four-angle sweep samples.
        center = 0.9 * np.exp(-1j * np.pi / 4)
        diag = jnp.asarray(center + np.exp(2j * np.pi * np.arange(m) / m))
        return (lambda v: diag * v), (lambda v: jnp.conj(diag) * v), center

    @pytest.mark.slow
    @pytest.mark.parametrize("n_angles", [4, 6, 8, 32])
    def test_origin_containing_range_is_never_adequate(self, n_angles: int) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner
        from moljax.conditioning.pseudospectra import arnoldi, ritz_values

        m = 24
        matvec, adjoint, center = self._origin_containing_operator(m)
        result = numerical_range(matvec, adjoint, m, n_angles=n_angles, max_iters=150)
        v0 = jnp.asarray(np.random.default_rng(0).standard_normal(m) + 0j)
        ritz = ritz_values(arnoldi(matvec, v0, 12)[1])
        assessment = assess_preconditioner(result, ritz, epsilon_zero=float(abs(center)))
        # The outer bound must never understate a range that contains zero.
        assert result.disk_rate >= 1.0
        assert assessment.verdict == "indeterminate"

    @pytest.mark.slow
    def test_outer_bound_converges_from_above(self) -> None:
        m = 24
        matvec, adjoint, _ = self._origin_containing_operator(m)
        coarse = numerical_range(matvec, adjoint, m, n_angles=4, max_iters=150)
        fine = numerical_range(matvec, adjoint, m, n_angles=32, max_iters=150)
        # Refining directions may only tighten a genuine outer bound.
        assert coarse.disk_rate >= fine.disk_rate

    @pytest.mark.slow
    def test_origin_outside_is_still_separated(self) -> None:
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
        assert assessment.verdict == "investigate"

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
            # A spread without a cluster is the disk-rate gate's business.
            ("spread spectrum without a bulk",
             np.array([23.86, 11.19, 1.59, 0.5]) + 0j, False),
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
            n_angles=8, max_iters=150, n_restarts=2,
        )
        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        assert assess_preconditioner(good, ritz, epsilon_zero=0.8).verdict == "adequate"
        doubtful = good._replace(supports_corroborated=False)
        assert (
            assess_preconditioner(doubtful, ritz, epsilon_zero=0.8).verdict
            == "indeterminate"
        )


class TestCertificationReachesEveryConsumer:
    """Uncertified geometry must not appear authoritative anywhere.

    Two independent conditions invalidate the outer bound: under-resolved
    supports, and restart disagreement.  Consumers that check only one of them
    let uncertified geometry through, which is how a rate derived from a
    boundary that may cut into the numerical range escaped as a convergence
    prediction.  ``supports_consistent`` is the single derived answer.
    """

    @staticmethod
    def _resolved():
        m = 24
        diag = jnp.asarray(np.linspace(0.8, 1.2, m))
        return numerical_range(
            lambda v: diag * v, lambda v: jnp.conj(diag) * v, m,
            n_angles=8, max_iters=150,
        )

    def test_both_conditions_are_required(self) -> None:
        fov = self._resolved()
        assert fov.supports_consistent
        assert not fov._replace(supports_converged=False).supports_consistent
        assert not fov._replace(supports_corroborated=False).supports_consistent

    @pytest.mark.parametrize("flag", ["supports_converged", "supports_corroborated"])
    def test_rates_withhold_the_prediction(self, flag: str) -> None:
        from moljax.conditioning.non_normality import estimate_rates

        fov = self._resolved()
        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        assert estimate_rates(fov, ritz).predicted_gmres_factor is not None

        doubtful = fov._replace(**{flag: False})
        rates = estimate_rates(doubtful, ritz)
        # r1/r2 stay available for relative comparison; the prediction does not.
        assert rates.predicted_gmres_factor is None
        assert not rates.supports_consistent
        assert math.isfinite(rates.r1)

    def test_rates_carry_corroboration_provenance(self) -> None:
        """A serialized estimate must say what evidence backs its prediction.

        With the default single start the prediction is offered, but it rests
        on the residual gate alone.  Once the estimate is passed on without
        the ``FieldOfValuesResult`` it came from, that qualification has to be
        readable from the estimate itself, or a provisional number is
        indistinguishable from a corroborated one.
        """
        from moljax.conditioning.non_normality import estimate_rates

        fov = self._resolved()
        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        single = estimate_rates(fov, ritz)
        assert single.predicted_gmres_factor is not None
        assert single.corroboration_attempted is False
        assert "corroboration_attempted" in single._asdict()

        restarted = estimate_rates(fov._replace(corroboration_attempted=True), ritz)
        assert restarted.corroboration_attempted is True
        assert restarted.predicted_gmres_factor == single.predicted_gmres_factor

    @pytest.mark.parametrize("flag", ["supports_converged", "supports_corroborated"])
    def test_assessment_abstains_for_either_condition(self, flag: str) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner

        fov = self._resolved()
        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        assert assess_preconditioner(fov, ritz, epsilon_zero=0.8).verdict == "provisional"

        doubtful = fov._replace(**{flag: False})
        assessment = assess_preconditioner(doubtful, ritz, epsilon_zero=0.8)
        assert assessment.verdict == "indeterminate"
        assert assessment.predicted_gmres_factor is None
        assert assessment.supports_consistent is False

    def test_assessment_records_whether_corroboration_ran(self) -> None:
        """A caller reading the assessment must be able to see what backed it."""
        from moljax.conditioning.non_normality import assess_preconditioner

        fov = self._resolved()
        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        a1 = assess_preconditioner(fov, ritz, epsilon_zero=0.8)
        assert a1.supports_consistent is True
        assert a1.corroboration_attempted is False  # default n_restarts=1

        a2 = assess_preconditioner(
            fov._replace(corroboration_attempted=True), ritz, epsilon_zero=0.8
        )
        assert a2.corroboration_attempted is True

    @pytest.mark.parametrize("flag", ["supports_converged", "supports_corroborated"])
    def test_plot_labels_uncertified_geometry(self, flag: str) -> None:
        pytest.importorskip("matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        from moljax.conditioning.figures import plot_numerical_range

        doubtful = self._resolved()._replace(**{flag: False})
        figure = plot_numerical_range(doubtful)
        axis = figure.axes[0]
        assert "unresolved" in axis.get_title().lower()
        labels = [text.get_text().lower() for text in axis.get_legend().get_texts()]
        assert not any(label == "enclosing disk" for label in labels)


class TestProvisionalVerdictReflectsCorroboration:
    """A verdict must not hide missing evidence in a side field.

    With ``n_restarts=1`` no restart is run, so the corroboration check that
    could have caught a missed dominant support was not attempted.  The
    threshold gates may still pass, but the strong claim "adequate" is not
    warranted.  The verdict itself has to carry that qualification -- most
    callers act on it and do not read the metadata.
    """

    @pytest.mark.slow
    def test_default_run_yields_provisional_not_adequate(self) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner

        m = 24
        diag = jnp.asarray(np.linspace(0.8, 1.2, m))
        # Default n_restarts=1: no corroboration attempted.
        fov = numerical_range(
            lambda v: diag * v, lambda v: jnp.conj(diag) * v, m,
            n_angles=8, max_iters=150,
        )
        assert fov.supports_consistent  # vacuously
        assert not fov.corroboration_attempted

        ritz = jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j)
        assessment = assess_preconditioner(fov, ritz, epsilon_zero=0.8)
        # The verdict itself, not only a flag, must show that adequate is not
        # earned when the corroboration branch was never tested.
        assert assessment.verdict == "provisional"

    @pytest.mark.slow
    def test_corroborated_run_promotes_to_adequate(self) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner

        m = 24
        diag = jnp.asarray(np.linspace(0.8, 1.2, m))
        fov = numerical_range(
            lambda v: diag * v, lambda v: jnp.conj(diag) * v, m,
            n_angles=8, max_iters=150, n_restarts=2,
        )
        assert fov.corroboration_attempted
        assessment = assess_preconditioner(
            fov, jnp.asarray(np.linspace(0.8, 1.2, 8) + 0j), epsilon_zero=0.8,
        )
        assert assessment.verdict == "adequate"

    def test_investigate_dominates_provisional(self) -> None:
        """A failing threshold gate is a stronger signal than uncorroboration."""
        from moljax.conditioning.non_normality import assess_preconditioner

        # Uncorroborated, but disk_rate above threshold: still "investigate".
        fov = FieldOfValuesResult(
            boundary=jnp.asarray([1.0 + 0.0j, 2.0 + 0.0j], dtype=jnp.complex128),
            center=1.5 + 0.0j, radius=0.9,
            disk_rate=0.6, origin_enclosed=False,
            cp_prefactor=1.0 + math.sqrt(2.0),
            corroboration_attempted=False,
        )
        result = assess_preconditioner(
            fov, jnp.asarray([1.4, 1.5, 1.6, 1.7]) + 0j, epsilon_zero=0.5,
            rate_threshold=0.5,  # tighter than 0.6 disk_rate -> fails
        )
        assert result.verdict == "investigate"


class TestOutlierGateFailsClosed:
    """A short or degraded Ritz spectrum must not read as "no outliers".

    Arnoldi trims its projection after a Krylov breakdown, so a handful of
    Ritz values is a real path.  Quartiles taken over a sample that includes
    the candidate let it inflate its own threshold: for ``[1, 1, 1, 6]`` the
    plain interquartile rule lands exactly on 6 and a strict comparison
    reports nothing.  A corroborated field of values spanning ``[1, 6]`` has
    ``disk_rate ~ 0.71``, so that spectrum would then pass every gate.
    """

    @staticmethod
    def _fov_on(left: float, right: float) -> FieldOfValuesResult:
        center = 0.5 * (left + right)
        radius = 0.5 * (right - left)
        return FieldOfValuesResult(
            boundary=jnp.asarray([left, right], dtype=jnp.complex128),
            center=center + 0.0j, radius=radius,
            disk_rate=radius / abs(center), origin_enclosed=False,
            cp_prefactor=1.0 + math.sqrt(2.0),
            corroboration_attempted=True,
        )

    def test_candidate_cannot_inflate_its_own_threshold(self) -> None:
        from moljax.conditioning.non_normality import real_bulk_outliers

        assert real_bulk_outliers(jnp.asarray([1.0, 1.0, 1.0, 6.0]) + 0j) == 1

    @pytest.mark.parametrize("ritz", [[1.0, 1.0, 1.0, 6.0], [1.0, 1.0, 1.0, 1.0, 6.0]])
    def test_short_spectrum_with_outlier_is_rejected(self, ritz: list[float]) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner

        fov = self._fov_on(1.0, 6.0)
        values = jnp.asarray(ritz) + 0j
        # Every other gate passes; the outlier gate is the only one that can
        # reject this spectrum.
        assert assess_preconditioner(
            fov, values, epsilon_zero=0.9, max_right_real_outliers=1
        ).verdict == "adequate"
        assessment = assess_preconditioner(fov, values, epsilon_zero=0.9)
        assert assessment.n_right_real_outliers == 1
        assert assessment.verdict == "investigate"

    @pytest.mark.parametrize("ritz", [[6.0], [1.0, 6.0], [1.0, 1.0, 6.0]])
    def test_too_few_values_abstain(self, ritz: list[float]) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner, real_bulk_outliers

        values = jnp.asarray(ritz) + 0j
        with pytest.raises(ValueError, match="at least 4"):
            real_bulk_outliers(values)
        assessment = assess_preconditioner(self._fov_on(1.0, 6.0), values, epsilon_zero=0.9)
        assert assessment.verdict == "indeterminate"
        assert assessment.n_right_real_outliers is None
        assert assessment.predicted_gmres_factor is None

    @pytest.mark.parametrize(
        "bad", [math.nan, math.inf, -math.inf, complex(1.0, math.nan)]
    )
    def test_non_finite_values_abstain(self, bad: complex) -> None:
        from moljax.conditioning.non_normality import assess_preconditioner, real_bulk_outliers

        values = jnp.asarray([1.0, 1.0, 1.0, bad], dtype=jnp.complex128)
        with pytest.raises(ValueError, match="non-finite"):
            real_bulk_outliers(values)
        assessment = assess_preconditioner(self._fov_on(1.0, 6.0), values, epsilon_zero=0.9)
        assert assessment.verdict == "indeterminate"
        assert assessment.n_right_real_outliers is None

    @pytest.mark.parametrize(
        ("reading", "value"),
        [
            ("disk_rate", math.nan),
            ("disk_rate", -math.inf),
            ("disk_rate", -0.5),
            ("epsilon_zero", math.nan),
            ("epsilon_zero", math.inf),
            ("epsilon_zero", -math.inf),
            ("epsilon_zero", -1e-3),
        ],
    )
    def test_unusable_reading_abstains(self, reading: str, value: float) -> None:
        """A reading outside its domain is not scored.

        Before this gate, an infinite ``epsilon_zero`` satisfied its lower
        bound and a negative ``disk_rate`` its upper bound, so either
        corrupted reading produced ``adequate`` on an otherwise sound input.
        """
        from moljax.conditioning.non_normality import assess_preconditioner

        fov = self._fov_on(1.0, 1.5)
        ritz = jnp.asarray([1.0, 1.1, 1.2, 1.4]) + 0j
        assert assess_preconditioner(fov, ritz, epsilon_zero=0.9).verdict == "adequate"
        epsilon = 0.9
        if reading == "disk_rate":
            fov = fov._replace(disk_rate=value)
        else:
            epsilon = value
        assessment = assess_preconditioner(fov, ritz, epsilon_zero=epsilon)
        assert assessment.verdict == "indeterminate"
        assert assessment.n_right_real_outliers is None
        assert assessment.predicted_gmres_factor is None

    def test_infinite_disk_rate_is_a_reading(self) -> None:
        """``numerical_range`` reports ``inf`` for a disk centered on the origin.

        That is a measurement, not a defect, and it fails the rate gate like
        any rate of one or more; the gate answers ``investigate``, not
        ``indeterminate``.
        """
        from moljax.conditioning.non_normality import assess_preconditioner

        fov = self._fov_on(1.0, 1.5)._replace(disk_rate=math.inf)
        assessment = assess_preconditioner(
            fov, jnp.asarray([1.0, 1.1, 1.2, 1.4]) + 0j, epsilon_zero=0.9
        )
        assert assessment.verdict == "investigate"
        assert assessment.n_right_real_outliers == 0



class TestCertificateGatesAreScaleInvariant:
    """The residual and corroboration gates must not depend on operator magnitude.

    Both gates normalized by ``max(scale, 1.0)`` and the LOBPCG shift was
    floored at 1.0 as well, so an operator of magnitude 1e-6 was solved as a
    perturbation of the identity: the eigensolver stopped at once, the
    residual gate measured its unconverged vectors against a unit scale and
    passed them, and the restart spread was judged against 1.0 and read as
    agreement.  The same operator at magnitude 1 failed both gates.  A flag
    that flips when the equations are rescaled is not a certificate.
    """

    @staticmethod
    def _actions(diagonal: np.ndarray):
        values = jnp.asarray(diagonal, dtype=jnp.complex128)
        return (lambda v: values * v), (lambda v: jnp.conj(values) * v)

    @pytest.mark.slow
    @pytest.mark.parametrize("scale", [1.0, 1.0e-6, 1.0e-10])
    def test_unconverged_supports_fail_both_gates_at_every_scale(self, scale: float) -> None:
        # 396 eigenvalues in [0.8, 1.0] and four at -1, so W(A) = [-1, 1].
        # Three LOBPCG iterations cannot resolve the supports: both gates
        # must fire, and they must fire regardless of the magnitude.
        m = 400
        diagonal = scale * np.concatenate([np.linspace(0.8, 1.0, m - 4), [-1.0] * 4])
        matvec, adjoint = self._actions(diagonal)
        result = numerical_range(matvec, adjoint, m, n_angles=8, max_iters=3, n_restarts=2)
        assert result.supports_converged is False
        assert result.supports_corroborated is False
        assert result.supports_consistent is False

    @pytest.mark.slow
    @pytest.mark.parametrize("scale", [1.0, 1.0e-6, 1.0e-10])
    def test_enclosed_origin_is_found_at_every_scale(self, scale: float) -> None:
        # Eigenvalues on the unit circle about a center of modulus 0.9, so the
        # origin lies inside W(A).  A resolved sweep must report it enclosed and
        # the outer disk must be the same disk, rescaled: with the unit floor
        # the 1e-10 copy was "solved" in one step and reported a smaller disk
        # with a clean certificate.
        m = 24
        center = 0.9 * np.exp(-1j * np.pi / 4)
        diagonal = scale * (center + np.exp(2j * np.pi * np.arange(m) / m))
        matvec, adjoint = self._actions(diagonal)
        result = numerical_range(matvec, adjoint, m, n_angles=8, max_iters=120, n_restarts=2)
        assert result.supports_consistent
        assert result.origin_enclosed is True
        # Exact eight-direction outer disk of the unscaled operator.
        assert result.disk_rate == pytest.approx(1.2027, abs=1.0e-3)
        assert result.radius / scale == pytest.approx(1.0824, abs=1.0e-3)
