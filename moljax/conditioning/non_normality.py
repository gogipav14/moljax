"""Rate estimates and heuristic conditioning decisions for linear operators.

This module combines numerical-range and Arnoldi diagnostics without
materializing the large operator.  The small dense computations act only on
traced boundary points or Ritz values.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from moljax.conditioning._geometry import _origin_enclosed, _smallest_enclosing_disk
from moljax.conditioning.field_of_values import FieldOfValuesResult

_CP_PREFACTOR = 1.0 + math.sqrt(2.0)
_RATE_AGREEMENT_TOLERANCE = 0.05
_MAX_OUTLIER_FRACTION = 0.2
_BULK_OUTLIER_FACTOR = 3.0
_BULK_IMAGINARY_FRACTION = 0.05


class RateEstimates(NamedTuple):
    """Complementary numerical-range and Ritz-value convergence estimates.

    Attributes:
        r1: Enclosing-disk estimate ``rho / abs(c)``.
        r2: Traced-boundary minimax estimate.
        r3: Bulk Ritz-clustering estimate after self-consistent outlier removal.
        predicted_gmres_factor: The smallest finite estimate among ``r1``,
            ``r2``, and ``r3``, or ``None`` when a defect in the supports was
            detected, or when the numerical range encloses the origin
            (equivalently ``r1 >= 1``).  ``r1`` and ``r2`` derive from the
            numerical range, so an under-resolved or disagreeing boundary
            makes them optimistically small; offering their minimum as a
            convergence prediction would present exactly the number a caller
            is most likely to act on.  An enclosed origin means GMRES is not
            guaranteed to converge at all, yet ``r3``, which only sees the
            Ritz bulk and not the outliers an enclosed origin represents, can
            still be small and finite; withholding the minimum keeps that
            bulk estimate from masquerading as a convergence prediction for
            an operator whose numerical range contains zero. Read it together
            with ``corroboration_attempted``: a value produced without
            restarts rests on the residual gate alone.
        agree: Whether the finite estimates agree within a relative tolerance.
        supports_consistent: Whether ``r1`` and ``r2`` rest on a boundary that
            passed every check that was actually run.  This is not a
            certificate: see ``FieldOfValuesResult.supports_consistent``.
            When false, ``r1`` and ``r2`` remain usable for relative
            comparison between runs sharing a configuration, but not as
            absolute rates.
        corroboration_attempted: Whether the boundary behind ``r1`` and
            ``r2`` was traced with independent eigensolver restarts, so a
            missed dominant support could have been detected.  Carried here
            so the estimate remains self-describing once it is serialized or
            passed on without the ``FieldOfValuesResult`` it came from.
    """

    r1: float
    r2: float
    r3: float
    predicted_gmres_factor: float | None
    agree: bool
    supports_consistent: bool = True
    corroboration_attempted: bool = False


class PreconditionerAssessment(NamedTuple):
    """Heuristic verdict from numerical-range and pseudospectral indicators.

    The default thresholds are generic heuristics, not parameters tuned to a
    particular problem.  Apply the procedure to states actually visited by a
    solver; synthetic stress states can be useful diagnostics but do not by
    themselves establish a production preconditioner's behavior.

    Verdict values:

    ``adequate``
        Every threshold gate passed, and the supports were corroborated by
        independent eigensolver restarts.  This is the strongest verdict the
        procedure emits; use as a positive signal that further preconditioner
        work is unlikely to change the picture on the assessed state.

    ``provisional``
        Every threshold gate passed, but corroboration was not attempted
        (``n_restarts`` was one).  The picture is consistent as far as the
        checks that were run go; a missed dominant support could still change
        it.  Raise ``n_restarts`` on ``numerical_range`` to promote a
        provisional verdict to ``adequate``.

    ``investigate``
        A threshold gate failed.  Further preconditioner work is likely
        warranted on this state.

    ``indeterminate``
        No gate was evaluated.  Either the geometry is not a valid outer
        bound (the supports failed their checks, or the origin is enclosed by
        the numerical range, so the disk-rate reading is not a convergence
        factor), or a diagnostic input was unusable: fewer than four Ritz
        values, a non-finite Ritz value, a NaN or negative ``disk_rate`` or
        ``epsilon_zero``, or an infinite ``epsilon_zero``.  (An infinite
        ``disk_rate`` is a reading: ``numerical_range`` reports it for a disk
        centered on the origin, and it fails the rate gate like any rate of
        one or more.)  In the second case ``n_right_real_outliers`` and
        ``predicted_gmres_factor`` are ``None``.  Unusable inputs abstain
        rather than raise because they are the signature of a degraded
        upstream computation, which must never read as a passed gate.
    """

    verdict: str
    disk_rate: float
    epsilon_zero: float
    n_right_real_outliers: int | None
    predicted_gmres_factor: float | None
    rate_threshold: float
    eps_zero_threshold: float
    max_right_real_outliers: int
    supports_consistent: bool = True
    corroboration_attempted: bool = False


def enclosing_disk_rate(fov: FieldOfValuesResult) -> float:
    """Return ``r1 = rho / abs(c)`` from an enclosing numerical-range disk."""
    return float(fov.disk_rate)


_GOLDEN_SECTION_RATIO = (math.sqrt(5.0) - 1.0) / 2.0


def _golden_section_minimize(
    func: Callable[[float], float], lower: float, upper: float, iterations: int = 100
) -> float:
    """Return the minimum value of a unimodal scalar function on ``[lower, upper]``.

    Golden-section search only needs unimodality, which a convex function of
    one real variable always has.  It is used here, nested, rather than a
    general-purpose optimizer so both the outer and inner searches stay a
    fixed, small number of scalar evaluations with no external dependency.
    """
    left, right = lower, upper
    probe_left = right - _GOLDEN_SECTION_RATIO * (right - left)
    probe_right = left + _GOLDEN_SECTION_RATIO * (right - left)
    value_left, value_right = func(probe_left), func(probe_right)
    for _ in range(iterations):
        if value_left < value_right:
            right, probe_right, value_right = probe_right, probe_left, value_left
            probe_left = right - _GOLDEN_SECTION_RATIO * (right - left)
            value_left = func(probe_left)
        else:
            left, probe_left, value_left = probe_left, probe_right, value_right
            probe_right = left + _GOLDEN_SECTION_RATIO * (right - left)
            value_right = func(probe_right)
    midpoint = 0.5 * (left + right)
    return min(value_left, value_right, func(midpoint))


def traced_boundary_rate(boundary: jax.Array) -> float:
    """Return ``r2 = min_beta max_i |1 - beta * z_i|`` on a traced boundary.

    The objective is convex in ``beta`` (a maximum of moduli of affine
    functions of ``beta``), and its minimum over each real coordinate with
    the other held fixed is itself convex in that coordinate, so a nested
    golden-section search -- outer over ``Re(beta)``, inner over
    ``Im(beta)`` -- finds the global minimum without an external optimizer
    and without materializing the pairwise circle-intersection geometry the
    previous bisection needed.

    That previous approach bisected over the rate ``r`` and tested
    feasibility using pairwise intersections of circles
    ``|beta - 1/z_i| <= r/|z_i|``.  Near tangency, the height of an
    intersection point carries an error of order
    ``eps |z_j| / (min|z| ** 2 * height)``, which blows up as the tangency is
    approached and swamps a fixed tolerance long before it reaches ``64
    eps``: on the interval ``[0.1, 3.9]`` it returned 0.97375 against the
    exact 0.95, and raising the tolerance alone did not fix it, because the
    error is not a rounding margin but a geometric singularity in the
    parametrization.

    If zero lies in the convex hull of the traced points, no finite complex
    ``beta`` can pull every ``beta * z_i`` inside a disk around 1 smaller
    than the whole plane's worth of directions spanned by points on both
    sides of the origin, and the minimax value is exactly 1; this is checked
    directly rather than left to the search, which would need an unbounded
    domain to discover it.
    """
    values = jnp.asarray(boundary, dtype=jnp.complex128)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("boundary must be a nonempty one-dimensional array")
    host_values = np.asarray(values, dtype=np.complex128)
    if _origin_enclosed(host_values):
        return 1.0
    magnitudes = np.abs(host_values)
    if bool(np.any(magnitudes <= np.finfo(np.float64).tiny)):
        return 1.0
    bound = 2.0 / float(np.min(magnitudes))

    def objective(re_beta: float, im_beta: float) -> float:
        beta = complex(re_beta, im_beta)
        return float(np.max(np.abs(1.0 - beta * host_values)))

    def minimize_over_imaginary_part(re_beta: float) -> float:
        return _golden_section_minimize(lambda im_beta: objective(re_beta, im_beta), -bound, bound)

    best = _golden_section_minimize(minimize_over_imaginary_part, -bound, bound)
    return float(min(best, 1.0))


def _robust_enclosing_disk(values: jax.Array) -> tuple[complex, float]:
    """Return an enclosing disk with an exact collinear fast path."""
    if values.size == 0:
        raise ValueError("an enclosing disk needs at least one point")
    if values.size == 1:
        return complex(values[0]), 0.0
    real_span = float(jnp.max(jnp.real(values)) - jnp.min(jnp.real(values)))
    imag_span = float(jnp.max(jnp.imag(values)) - jnp.min(jnp.imag(values)))
    span = max(real_span, imag_span, 1.0)
    if imag_span <= 1.0e-13 * span:
        left = complex(values[int(jnp.argmin(jnp.real(values)))])
        right = complex(values[int(jnp.argmax(jnp.real(values)))])
        center = 0.5 * (left + right)
        return center, abs(right - left) / 2.0
    return _smallest_enclosing_disk(values)


def _bulk_disk(ritz: jax.Array) -> tuple[complex, float]:
    """Select the reference bulk cluster and return its enclosing disk."""
    values = jnp.asarray(ritz, dtype=jnp.complex128)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("ritz must be a nonempty one-dimensional array")
    order = jnp.argsort(jnp.real(values))[::-1]
    maximum_outliers = max(1, int(math.floor(_MAX_OUTLIER_FRACTION * values.size)))
    for count in range(maximum_outliers, 0, -1):
        outlier_indices = order[:count]
        bulk_indices = order[count:]
        if bulk_indices.size < 3:
            continue
        center, radius = _robust_enclosing_disk(values[bulk_indices])
        threshold = center.real + _BULK_OUTLIER_FACTOR * radius
        flags = (jnp.real(values) > threshold) & (
            jnp.abs(jnp.imag(values)) <= _BULK_IMAGINARY_FRACTION * radius
        )
        if {int(index) for index in jnp.flatnonzero(flags)} == {
            int(index) for index in outlier_indices
        }:
            return center, radius
    return _robust_enclosing_disk(values)


def clustering_rate(ritz: jax.Array) -> float:
    """Return ``r3`` from the self-consistent bulk Ritz-value cluster.

    Up to 20% of the rightmost Ritz values may be removed only when they lie
    beyond ``center.real + 3 * radius`` and close to the real axis.  ``r3`` is
    then the bulk disk's ``radius / abs(center)``.
    """
    center, radius = _bulk_disk(ritz)
    return math.inf if abs(center) == 0.0 else float(radius / abs(center))


def crouzeix_palencia_envelope(
    disk_rate: float,
    n_iters: int,
    *,
    prefactor: float = _CP_PREFACTOR,
) -> jax.Array:
    """Return ``prefactor * disk_rate**k`` for iterations ``k = 1, ..., n``."""
    if n_iters < 0:
        raise ValueError("n_iters must be nonnegative")
    iterations = jnp.arange(1, n_iters + 1, dtype=jnp.float64)
    return (
        jnp.asarray(prefactor, dtype=jnp.float64)
        * jnp.asarray(disk_rate, dtype=jnp.float64) ** iterations
    )


def right_real_outliers(
    ritz: jax.Array,
    center: complex,
    radius: float,
    *,
    factor: float = 1.0,
) -> int:
    """Count Ritz values with real part beyond ``center.real + factor * radius``."""
    values = jnp.asarray(ritz, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("ritz must be a one-dimensional array")
    threshold = center.real + factor * radius
    return int(jnp.count_nonzero(jnp.real(values) > threshold))


_REAL_BULK_RELATIVE_FLOOR = 1.0e-6
# An outlier must sit more than this many bulk widths beyond the bulk's top.
# Measured on clean symmetric clusters of eight values, a 3 x IQR rule flags a
# spurious outlier roughly 8% of the time and this rule roughly 0.3%; the
# width of a handful of values is a steadier scale than their quartiles.
_REAL_BULK_WIDTH_FACTOR = 2.0
_MIN_REAL_BULK_SIZE = 3


def _ritz_defect(ritz: jax.Array) -> str | None:
    """Return why ``ritz`` cannot support an outlier count, or ``None``.

    A wrong shape is a programming error and raises.  Too few values or a
    non-finite value are the signature of a degraded upstream computation,
    so they are reported rather than raised, and the caller decides whether
    to raise or to abstain.
    """
    values = jnp.asarray(ritz, dtype=jnp.complex128)
    if values.ndim != 1:
        raise ValueError("ritz must be a one-dimensional array")
    minimum = _MIN_REAL_BULK_SIZE + 1
    if values.size < minimum:
        return f"at least {minimum} Ritz values are needed, got {values.size}"
    if not bool(jnp.all(jnp.isfinite(values))):
        return "ritz contains a non-finite value"
    return None


def _reading_defect(disk_rate: float, epsilon_zero: float) -> str | None:
    """Return why a scalar reading cannot be gated, or ``None``.

    Both readings are non-negative by construction, and a finite matrix has
    a finite smallest singular value, so a NaN, a negative value or an
    infinite ``epsilon_zero`` is not a measurement.  Scored as one, an
    infinite ``epsilon_zero`` or a negative ``disk_rate`` would pass its
    gate.  An infinite ``disk_rate`` is different: ``numerical_range``
    reports it for a disk centered on the origin, and it fails the rate gate
    like any rate of one or more, so it is left to the gate.
    """
    rate = float(disk_rate)
    if math.isnan(rate) or rate < 0.0:
        return f"disk_rate must be a non-negative number, got {rate}"
    eps = float(epsilon_zero)
    if not math.isfinite(eps) or eps < 0.0:
        return f"epsilon_zero must be a finite non-negative number, got {eps}"
    return None


def real_bulk_outliers(ritz: jax.Array, *, factor: float = _REAL_BULK_WIDTH_FACTOR) -> int:
    """Count Ritz values whose real part sits beyond a robust real-part bulk.

    This deliberately ignores imaginary parts and does not reuse ``_bulk_disk``.
    That helper requires a candidate outlier set to be near-real relative to the
    bulk radius, and when no candidate passes it falls back to a disk enclosing
    every Ritz value.  Counting beyond such a disk is guaranteed to return zero,
    so a bulk-model failure would read as "no outliers" rather than as "no
    model".  Two spectra defeat it that way: a tight real bulk carrying
    roundoff-sized imaginary parts, whose bulk radius is zero, and a real
    outlier accompanied by a farther complex pair.

    The bulk is estimated with the candidate outliers left out.  The real parts
    are sorted, and for each admissible outlier count ``k`` (at most 20% of the
    values, always at least one, always leaving a bulk of at least three) the
    remaining values set the threshold ``top + factor * max(width, floor)``,
    where ``top`` and ``width`` are the largest value and the range of the
    bulk; the largest ``k`` whose ``k``-th value exceeds it is returned.  A
    scale estimated from a sample that includes the candidate lets a single far
    value inflate its own threshold: for ``[1, 1, 1, 6]`` a plain interquartile
    rule lands exactly on 6 and a strict comparison reports nothing, and
    Arnoldi does produce spectra that short after an early Krylov breakdown.
    The width is used rather than the interquartile range because Arnoldi
    projections are small samples, on which quartiles are too noisy a scale.
    The relative floor keeps a numerically degenerate bulk, whose width is
    roundoff, from flagging its own roundoff.

    A spread of values without a cluster, such as the spectrum of an
    unpreconditioned operator, is not an outlier pattern and is left to the
    disk-rate gate.

    Fewer than four values leave no bulk to measure against, and a non-finite
    value makes every comparison vacuous; both raise ``ValueError`` here.
    ``assess_preconditioner`` turns the same conditions into an
    ``indeterminate`` verdict instead.
    """
    defect = _ritz_defect(ritz)
    if defect is not None:
        raise ValueError(defect)
    values = jnp.asarray(ritz, dtype=jnp.complex128)
    reals = np.sort(np.asarray(jnp.real(values), dtype=np.float64))[::-1]
    count = reals.size
    maximum_outliers = min(
        max(1, int(math.floor(_MAX_OUTLIER_FRACTION * count))),
        count - _MIN_REAL_BULK_SIZE,
    )
    for n_outliers in range(maximum_outliers, 0, -1):
        bulk = reals[n_outliers:]
        top, bottom = float(bulk[0]), float(bulk[-1])
        width = max(top - bottom, _REAL_BULK_RELATIVE_FLOOR * max(abs(top), abs(bottom), 1.0))
        if reals[n_outliers - 1] > top + factor * width:
            return n_outliers
    return 0


def _rate_agreement(rates: tuple[float, float, float]) -> bool:
    """Return whether finite rate estimates agree to a five-percent tolerance."""
    finite = [rate for rate in rates if math.isfinite(rate)]
    if len(finite) < 2:
        return True
    spread = max(finite) - min(finite)
    scale = max(1.0, max(abs(rate) for rate in finite))
    return spread <= _RATE_AGREEMENT_TOLERANCE * scale


def estimate_rates(fov: FieldOfValuesResult, ritz: jax.Array) -> RateEstimates:
    """Combine the ``r1``, ``r2``, and ``r3`` diagnostics for one operator state."""
    r1 = enclosing_disk_rate(fov)
    r2 = traced_boundary_rate(fov.boundary)
    r3 = clustering_rate(ritz)
    consistent = fov.supports_consistent
    finite = [rate for rate in (r1, r2, r3) if math.isfinite(rate)]
    # r1 and r2 are read off the numerical-range geometry, so without an
    # outer bound their minimum is not a convergence prediction and must not
    # be offered as one.  We withhold it on any detected inconsistency.  A
    # numerical range that encloses the origin (equivalently r1 >= 1, since
    # an outer disk containing the origin has |center| <= radius) is the
    # same abstention for a different reason: r3 is a bulk-clustering
    # estimate that ignores the outliers the enclosed origin represents, so
    # it can be small and finite while the enclosing-disk and traced-boundary
    # rates that see the whole range are at or above one.  Offering the
    # bulk estimate's minimum in that case is exactly the confident wrong
    # number a caller is most likely to act on.  What evidence backs the
    # number when it is offered travels with it as corroboration_attempted,
    # so a serialized estimate stays self-describing.
    safe_to_predict = consistent and not fov.origin_enclosed and r1 < 1.0
    predicted_gmres_factor = (min(finite) if finite else math.inf) if safe_to_predict else None
    return RateEstimates(
        r1=r1,
        r2=r2,
        r3=r3,
        predicted_gmres_factor=predicted_gmres_factor,
        agree=_rate_agreement((r1, r2, r3)),
        supports_consistent=consistent,
        corroboration_attempted=bool(fov.corroboration_attempted),
    )


def assess_preconditioner(
    fov: FieldOfValuesResult,
    ritz: jax.Array,
    epsilon_zero: float,
    *,
    rate_threshold: float = 0.9,
    eps_zero_threshold: float = 0.1,
    max_right_real_outliers: int = 0,
) -> PreconditionerAssessment:
    """Classify a preconditioner with generic, heuristic diagnostic thresholds.

    These defaults are not tuned to a specific problem.  They should be
    assessed on states a solver actually visits; synthetic stress states are
    diagnostic complements rather than a standalone performance verdict.
    """
    # A degraded upstream computation shows up here as a short or non-finite
    # Ritz spectrum or a reading outside its domain.  None of these is a
    # measurement, yet each would be scored as one: a NaN reading fails its
    # gate, which is at least conservative, while a short or non-finite
    # spectrum counted as zero outliers, an infinite epsilon_zero or a
    # negative disk_rate would pass.  Any of them abstains before any gate
    # is evaluated.
    inputs_usable = (
        _ritz_defect(ritz) is None and _reading_defect(fov.disk_rate, epsilon_zero) is None
    )
    rates = estimate_rates(fov, ritz) if inputs_usable else None
    # The outlier count must be measured against an independent cluster model.
    # fov.center/fov.radius describe a disk enclosing the traced outer bound,
    # and Ritz values of an Arnoldi compression lie in the numerical range, so
    # whenever that bound holds no Ritz value can exceed its right edge:
    # measured there, the count is identically zero and max_right_real_outliers
    # can never reject anything.  The Ritz bulk is the model that can.
    n_outliers = real_bulk_outliers(ritz) if inputs_usable else None
    if not inputs_usable:
        verdict = "indeterminate"
    elif not fov.supports_consistent:
        # One derived condition, so this cannot drift out of step with the
        # other consumers.  It covers both under-resolved supports, judged
        # against the tolerance the caller requested rather than a default
        # chosen here, and any detected restart disagreement.  Either way the
        # half-plane intersection may not contain the numerical range.  This
        # is not proof that it does contain it when the flag is True; that is
        # what corroboration_attempted is for.
        verdict = "indeterminate"
    elif fov.origin_enclosed:
        verdict = "indeterminate"
    elif (
        fov.disk_rate <= rate_threshold
        and epsilon_zero >= eps_zero_threshold
        and n_outliers <= max_right_real_outliers
    ):
        # Passing every threshold gate is only a strong claim when the
        # corroboration check that could have caught a missed dominant support
        # was actually run.  Without it the verdict is honest but weak: the
        # caller must see the qualification in the action they take, not only
        # in a side field they might skip.  Raise n_restarts to promote it.
        verdict = "adequate" if fov.corroboration_attempted else "provisional"
    else:
        verdict = "investigate"
    return PreconditionerAssessment(
        verdict=verdict,
        disk_rate=float(fov.disk_rate),
        epsilon_zero=float(epsilon_zero),
        n_right_real_outliers=n_outliers,
        predicted_gmres_factor=None if rates is None else rates.predicted_gmres_factor,
        rate_threshold=float(rate_threshold),
        eps_zero_threshold=float(eps_zero_threshold),
        max_right_real_outliers=max_right_real_outliers,
        supports_consistent=bool(fov.supports_consistent),
        corroboration_attempted=bool(fov.corroboration_attempted),
    )
