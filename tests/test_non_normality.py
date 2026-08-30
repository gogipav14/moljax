"""Dense-reference tests for non-normality rate estimates and decisions."""

from __future__ import annotations

import itertools
import math

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import optimize

from moljax.conditioning import (
    FieldOfValuesResult,
    assess_preconditioner,
    clustering_rate,
    crouzeix_palencia_envelope,
    enclosing_disk_rate,
    estimate_rates,
    right_real_outliers,
)


def _grcar(n: int) -> np.ndarray:
    """Return the order-n Grcar matrix with three superdiagonals."""
    matrix = np.eye(n, dtype=np.complex128)
    for diagonal in range(1, 4):
        matrix += np.diag(np.ones(n - diagonal), k=diagonal)
    matrix += np.diag(-np.ones(n - 1), k=-1)
    return matrix


def _normal_matrix() -> np.ndarray:
    """Return a normal diagonal matrix whose numerical range avoids zero."""
    return np.diag(
        np.array(
            [1.1 - 0.4j, 1.6 + 0.8j, 2.3 - 0.2j, 2.0 + 1.1j, 1.3 - 1.0j],
            dtype=np.complex128,
        )
    )


def _dense_boundary(matrix: np.ndarray, n_angles: int = 14) -> np.ndarray:
    """Trace Johnson support points with dense Hermitian eigensolves."""
    result = np.empty(n_angles, dtype=np.complex128)
    for index in range(n_angles):
        theta = 2.0 * math.pi * index / n_angles
        phase = np.exp(1j * theta)
        hermitian = 0.5 * (phase * matrix + phase.conjugate() * matrix.conj().T)
        _, vectors = np.linalg.eigh(hermitian)
        vector = vectors[:, -1]
        result[index] = np.vdot(vector, matrix @ vector)
    return result


def _circle_from_two(first: complex, second: complex) -> tuple[complex, float]:
    """Return the diameter circle through two points."""
    center = 0.5 * (first + second)
    return center, abs(first - center)


def _circle_from_three(
    first: complex, second: complex, third: complex
) -> tuple[complex, float] | None:
    """Return the circumcircle, or ``None`` for collinear points."""
    ax, ay = first.real, first.imag
    bx, by = second.real, second.imag
    cx, cy = third.real, third.imag
    determinant = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(determinant) <= np.finfo(np.float64).eps:
        return None
    ux = (
        (ax * ax + ay * ay) * (by - cy)
        + (bx * bx + by * by) * (cy - ay)
        + (cx * cx + cy * cy) * (ay - by)
    ) / determinant
    uy = (
        (ax * ax + ay * ay) * (cx - bx)
        + (bx * bx + by * by) * (ax - cx)
        + (cx * cx + cy * cy) * (bx - ax)
    ) / determinant
    center = complex(ux, uy)
    return center, abs(first - center)


def _minimum_enclosing_disk(points: np.ndarray) -> tuple[complex, float]:
    """Enumerate the finite set of circles defining the exact small-point disk."""
    values = [complex(value) for value in points]
    candidates = [(value, 0.0) for value in values]
    candidates.extend(_circle_from_two(*pair) for pair in itertools.combinations(values, 2))
    candidates.extend(
        circle
        for triple in itertools.combinations(values, 3)
        if (circle := _circle_from_three(*triple)) is not None
    )
    scale = max(1.0, *(abs(value) for value in values))
    valid = [
        circle
        for circle in candidates
        if all(abs(point - circle[0]) <= circle[1] + 1.0e-12 * scale for point in values)
    ]
    return min(valid, key=lambda circle: circle[1])


def _reference_traced_rate(values: np.ndarray, disk_center: complex) -> float:
    """Independently minimize the reference ``min_alpha max |1-z/alpha|``."""
    real_span = max(float(np.ptp(values.real)), 1.0e-3)
    imag_span = max(float(np.ptp(values.imag)), 1.0e-3)
    real_mid = float(np.mean(values.real))
    imag_mid = float(np.mean(values.imag))

    def objective(raw: np.ndarray) -> float:
        center = complex(float(raw[0]), float(raw[1]))
        if abs(center) <= np.finfo(np.float64).tiny:
            return 1.0e12
        return float(np.max(np.abs(1.0 - values / center)))

    result = optimize.differential_evolution(
        objective,
        (
            (real_mid - 20.0 * real_span, real_mid + 20.0 * real_span),
            (imag_mid - 20.0 * imag_span, imag_mid + 20.0 * imag_span),
        ),
        seed=20260807,
        maxiter=160,
        popsize=10,
        tol=1.0e-10,
        polish=True,
        workers=1,
        updating="immediate",
    )
    return min(float(result.fun), objective(np.asarray([disk_center.real, disk_center.imag])), 1.0)


def _robust_disk_reference(values: np.ndarray) -> tuple[complex, float]:
    """Mirror the collinear rule while using this test's disk enumerator."""
    if values.size == 1:
        return complex(values[0]), 0.0
    span = max(float(np.ptp(values.real)), float(np.ptp(values.imag)), 1.0)
    if float(np.ptp(values.imag)) <= 1.0e-13 * span:
        left = values[int(np.argmin(values.real))]
        right = values[int(np.argmax(values.real))]
        center = 0.5 * (left + right)
        return complex(center), float(abs(right - left) / 2.0)
    return _minimum_enclosing_disk(values)


def _reference_clustering_rate(ritz: np.ndarray) -> float:
    """Compute the documented 20%-outlier, bulk-disk reference estimate."""
    order = np.argsort(ritz.real)[::-1]
    selected: tuple[complex, float] | None = None
    maximum_outliers = max(1, int(math.floor(0.2 * ritz.size)))
    for count in range(maximum_outliers, 0, -1):
        outliers = order[:count]
        bulk = order[count:]
        if bulk.size < 3:
            continue
        center, radius = _robust_disk_reference(ritz[bulk])
        flags = (ritz.real > center.real + 3.0 * radius) & (np.abs(ritz.imag) <= 0.05 * radius)
        if set(np.flatnonzero(flags)) == set(outliers.tolist()):
            selected = center, radius
            break
    center, radius = selected if selected is not None else _robust_disk_reference(ritz)
    return math.inf if abs(center) == 0.0 else float(radius / abs(center))


def _fov_from_boundary(boundary: np.ndarray) -> FieldOfValuesResult:
    """Build a result with independently constructed enclosing-disk fields."""
    center, radius = _minimum_enclosing_disk(boundary)
    return FieldOfValuesResult(
        boundary=jnp.asarray(boundary, dtype=jnp.complex128),
        center=center,
        radius=radius,
        disk_rate=radius / abs(center),
        origin_enclosed=False,
        cp_prefactor=1.0 + math.sqrt(2.0),
    )


@pytest.mark.slow
@pytest.mark.parametrize("matrix", [_grcar(6), _normal_matrix()], ids=["grcar", "normal"])
def test_rate_estimates_match_independent_dense_references(matrix: np.ndarray):
    """Expected r1/r2/r3 values come from dense supports and dense Ritz values."""
    boundary = _dense_boundary(matrix)
    fov = _fov_from_boundary(boundary)
    ritz = np.linalg.eigvals(matrix)
    rates = estimate_rates(fov, jnp.asarray(ritz, dtype=jnp.complex128))

    expected_r1 = fov.radius / abs(fov.center)
    expected_r2 = _reference_traced_rate(boundary, fov.center)
    expected_r3 = _reference_clustering_rate(ritz)
    assert enclosing_disk_rate(fov) == pytest.approx(expected_r1, abs=1.0e-12)
    assert rates.r1 == pytest.approx(expected_r1, abs=1.0e-12)
    assert rates.r2 == pytest.approx(expected_r2, abs=1.0e-9)
    assert clustering_rate(jnp.asarray(ritz, dtype=jnp.complex128)) == pytest.approx(
        expected_r3, abs=1.0e-12
    )
    assert rates.r3 == pytest.approx(expected_r3, abs=1.0e-12)
    assert rates.predicted_gmres_factor == pytest.approx(min(expected_r1, expected_r2, expected_r3))


def test_crouzeix_palencia_envelope_matches_formula_and_decreases():
    """The envelope is the universal prefactor times the geometric disk rate."""
    disk_rate = 0.4
    actual = np.asarray(crouzeix_palencia_envelope(disk_rate, 5))
    expected = (1.0 + math.sqrt(2.0)) * disk_rate ** np.arange(1, 6)
    np.testing.assert_allclose(actual, expected, atol=1.0e-15, rtol=0.0)
    assert np.all(np.diff(actual) < 0.0)


def test_right_real_outliers_counts_planted_values():
    """Only values beyond the requested real threshold are counted."""
    ritz = jnp.asarray([1.7 + 0.0j, 2.1 + 0.2j, 2.31 - 3.0j, 2.8 + 1.0j])
    assert right_real_outliers(ritz, center=2.0 + 0.0j, radius=0.25) == 2


def _assessment_fov(
    boundary: list[complex], center: complex, radius: float, origin_enclosed: bool
) -> FieldOfValuesResult:
    """Create a small numerical-range result for decision-procedure tests."""
    return FieldOfValuesResult(
        boundary=jnp.asarray(boundary, dtype=jnp.complex128),
        center=center,
        radius=radius,
        disk_rate=radius / abs(center) if center else math.inf,
        origin_enclosed=origin_enclosed,
        cp_prefactor=1.0 + math.sqrt(2.0),
    )


def test_assess_preconditioner_verdicts_cover_all_branches():
    """The heuristic recognizes adequate, investigate, and indeterminate cases."""
    adequate = _assessment_fov([1.9 + 0.0j, 2.1 + 0.0j], 2.0 + 0.0j, 0.1, False)
    broad = _assessment_fov([0.1 + 0.0j, 3.9 + 0.0j], 2.0 + 0.0j, 1.9, False)
    enclosing = _assessment_fov([1.0 + 0.0j, -1.0 + 0.0j], 0.0 + 0.0j, 1.0, True)
    ritz = jnp.asarray([1.95 + 0.0j, 2.05 + 0.0j])

    assert assess_preconditioner(adequate, ritz, epsilon_zero=0.2).verdict == "adequate"
    assert assess_preconditioner(broad, ritz, epsilon_zero=0.2).verdict == "investigate"
    assert assess_preconditioner(enclosing, ritz, epsilon_zero=0.2).verdict == "indeterminate"
