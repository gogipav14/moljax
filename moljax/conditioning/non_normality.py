"""Rate estimates and heuristic conditioning decisions for linear operators.

This module combines numerical-range and Arnoldi diagnostics without
materializing the large operator.  The small dense computations act only on
traced boundary points or Ritz values.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from moljax.conditioning._geometry import _smallest_enclosing_disk
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
            ``r2``, and ``r3``.
        agree: Whether the finite estimates agree within a relative tolerance.
    """

    r1: float
    r2: float
    r3: float
    predicted_gmres_factor: float
    agree: bool


class PreconditionerAssessment(NamedTuple):
    """Heuristic verdict from numerical-range and pseudospectral indicators.

    The default thresholds are generic heuristics, not parameters tuned to a
    particular problem.  Apply the procedure to states actually visited by a
    solver; synthetic stress states can be useful diagnostics but do not by
    themselves establish a production preconditioner's behavior.
    """

    verdict: str
    disk_rate: float
    epsilon_zero: float
    n_right_real_outliers: int
    predicted_gmres_factor: float
    rate_threshold: float
    eps_zero_threshold: float
    max_right_real_outliers: int


def enclosing_disk_rate(fov: FieldOfValuesResult) -> float:
    """Return ``r1 = rho / abs(c)`` from an enclosing numerical-range disk."""
    return float(fov.disk_rate)


def _traced_rate_feasible(centers: jax.Array, scales: jax.Array, rate: jax.Array) -> jax.Array:
    """Test feasibility of ``|beta - 1/z| <= rate / |z|`` for all points."""
    count = centers.size
    first, second = jnp.triu_indices(count, k=1)
    left = centers[first]
    right = centers[second]
    left_radius = rate * scales[first]
    right_radius = rate * scales[second]
    difference = right - left
    distance = jnp.abs(difference)
    safe_distance = jnp.where(distance > jnp.finfo(jnp.float64).tiny, distance, 1.0)
    direction = difference / safe_distance
    along = (left_radius**2 - right_radius**2 + distance**2) / (2.0 * safe_distance)
    height_squared = left_radius**2 - along**2
    base = left + along * direction
    height = jnp.sqrt(jnp.maximum(height_squared, 0.0))
    intersections = jnp.concatenate(
        (base + 1j * height * direction, base - 1j * height * direction)
    )
    pair_valid = (distance > jnp.finfo(jnp.float64).tiny) & (
        height_squared >= -64.0 * jnp.finfo(jnp.float64).eps
    )
    candidate_valid = jnp.concatenate((jnp.ones(count, dtype=bool), pair_valid, pair_valid))
    candidates = jnp.concatenate((centers, intersections))
    scaled_distances = jnp.abs(candidates[:, None] - centers[None, :]) / scales[None, :]
    contained = jnp.max(scaled_distances, axis=1) <= rate + 64.0 * jnp.finfo(jnp.float64).eps
    return jnp.any(candidate_valid & contained)


def traced_boundary_rate(boundary: jax.Array) -> float:
    """Return ``r2 = min_alpha max_z |1 - z / alpha|`` on a traced boundary.

    Under ``beta = 1 / alpha``, each bound is a disk
    ``|beta - 1 / z| <= r / |z|``.  Bisection over ``r`` and the finite set of
    disk centers and pairwise circle intersections evaluates this minimax
    problem without an external optimizer.  Keeping every traced point is
    equivalent to restricting to convex-hull vertices for this convex
    objective.
    """
    values = jnp.asarray(boundary, dtype=jnp.complex128)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("boundary must be a nonempty one-dimensional array")
    magnitudes = jnp.abs(values)
    if bool(jnp.any(magnitudes <= jnp.finfo(jnp.float64).tiny)):
        return 1.0
    centers = 1.0 / values
    scales = 1.0 / magnitudes

    def bisect(_: int, interval: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        lower, upper = interval
        middle = 0.5 * (lower + upper)
        feasible = _traced_rate_feasible(centers, scales, middle)
        return jax.lax.cond(
            feasible,
            lambda _: (lower, middle),
            lambda _: (middle, upper),
            operand=None,
        )

    _, result = jax.lax.fori_loop(
        0,
        48,
        bisect,
        (jnp.asarray(0.0), jnp.asarray(1.0)),
    )
    return float(result)


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
    finite = [rate for rate in (r1, r2, r3) if math.isfinite(rate)]
    predicted_gmres_factor = min(finite) if finite else math.inf
    return RateEstimates(
        r1=r1,
        r2=r2,
        r3=r3,
        predicted_gmres_factor=predicted_gmres_factor,
        agree=_rate_agreement((r1, r2, r3)),
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
    rates = estimate_rates(fov, ritz)
    n_outliers = right_real_outliers(ritz, fov.center, fov.radius)
    if fov.origin_enclosed:
        verdict = "indeterminate"
    elif (
        fov.disk_rate <= rate_threshold
        and epsilon_zero >= eps_zero_threshold
        and n_outliers <= max_right_real_outliers
    ):
        verdict = "adequate"
    else:
        verdict = "investigate"
    return PreconditionerAssessment(
        verdict=verdict,
        disk_rate=float(fov.disk_rate),
        epsilon_zero=float(epsilon_zero),
        n_right_real_outliers=n_outliers,
        predicted_gmres_factor=rates.predicted_gmres_factor,
        rate_threshold=float(rate_threshold),
        eps_zero_threshold=float(eps_zero_threshold),
        max_right_real_outliers=max_right_real_outliers,
    )
