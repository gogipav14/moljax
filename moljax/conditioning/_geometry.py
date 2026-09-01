"""Small dense geometry helpers for conditioning diagnostics."""

from __future__ import annotations

import numpy as np


def _circle_from_two(first: complex, second: complex) -> tuple[complex, float]:
    """Return the diameter circle through two points."""
    center = 0.5 * (first + second)
    return center, abs(first - center)


def _circle_from_three(
    first: complex, second: complex, third: complex
) -> tuple[complex, float] | None:
    """Return a circumcircle, or ``None`` when the points are collinear."""
    ax, ay = first.real, first.imag
    bx, by = second.real, second.imag
    cx, cy = third.real, third.imag
    determinant = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    # The determinant is twice the signed triangle area, so it scales with the
    # square of the coordinate span.  An absolute epsilon therefore declares
    # every well-formed triangle collinear once the points are small enough
    # (below roughly 1e-8), which silently returns a two-point circle that need
    # not contain the third point.  Compare against the span instead.
    span = max(abs(first - second), abs(second - third), abs(third - first))
    if span <= 0.0:
        return None
    if abs(determinant) <= 8.0 * np.finfo(np.float64).eps * span * span:
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


def _contains(circle: tuple[complex, float], point: complex, tolerance: float) -> bool:
    """Return whether ``circle`` contains ``point`` to floating-point tolerance."""
    center, radius = circle
    return abs(point - center) <= radius + tolerance


def _smallest_enclosing_disk(points: np.ndarray) -> tuple[complex, float]:
    """Compute the exact minimum enclosing disk of a finite point set without SciPy."""
    values = [complex(value) for value in np.asarray(points, dtype=np.complex128)]
    if not values:
        raise ValueError("an enclosing disk requires at least one boundary point")
    # Tolerances must track the data.  Flooring the scale at 1.0 makes the
    # containment test meaninglessly loose for small-magnitude spectra.
    scale = max((abs(value) for value in values), default=0.0)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    order = np.random.default_rng(0).permutation(len(values))
    ordered = [values[int(index)] for index in order]
    circle: tuple[complex, float] = (ordered[0], 0.0)
    for first_index, first in enumerate(ordered[1:], start=1):
        if _contains(circle, first, tolerance):
            continue
        circle = (first, 0.0)
        for second_index, second in enumerate(ordered[:first_index]):
            if _contains(circle, second, tolerance):
                continue
            circle = _circle_from_two(first, second)
            for third in ordered[:second_index]:
                if _contains(circle, third, tolerance):
                    continue
                candidate = _circle_from_three(first, second, third)
                if candidate is not None:
                    circle = candidate
    # Safety net: a disk that fails to enclose its own boundary would silently
    # understate the disk rate, which is the primary input to the adequacy
    # verdict.  Inflating the radius keeps the result correct and errs toward
    # the conservative verdict rather than a false certificate.
    center, radius = circle
    worst = max((abs(value - center) for value in values), default=0.0)
    if worst > radius:
        circle = (center, worst)
    return circle


def _convex_hull(points: np.ndarray) -> list[complex]:
    """Return the counter-clockwise convex hull using the monotone-chain algorithm."""
    values = sorted({(float(value.real), float(value.imag)) for value in points})
    if len(values) <= 1:
        return [complex(*value) for value in values]

    def cross(
        origin: tuple[float, float], left: tuple[float, float], right: tuple[float, float]
    ) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return [complex(*point) for point in lower[:-1] + upper[:-1]]


def _origin_enclosed(points: np.ndarray) -> bool:
    """Return whether zero is in the convex hull of the boundary points."""
    scale = max(1.0, *(abs(complex(value)) for value in points))
    coordinate_tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if (
        np.min(points.real) > coordinate_tolerance
        or np.max(points.real) < -coordinate_tolerance
        or np.min(points.imag) > coordinate_tolerance
        or np.max(points.imag) < -coordinate_tolerance
    ):
        return False
    hull = _convex_hull(points)
    if len(hull) == 1:
        return abs(hull[0]) <= 64.0 * np.finfo(np.float64).eps
    if len(hull) == 2:
        first, second = hull
        segment = second - first
        if abs(segment) <= np.finfo(np.float64).eps:
            return abs(first) <= 64.0 * np.finfo(np.float64).eps
        position = -first / segment
        return abs(position.imag) <= 1.0e-12 and 0.0 <= position.real <= 1.0
    tolerance = 64.0 * np.finfo(np.float64).eps * scale * scale
    for first, second in zip(hull, hull[1:] + hull[:1], strict=True):
        if (second - first).real * (-first).imag - (second - first).imag * (
            -first
        ).real < -tolerance:
            return False
    return True


def _support_outer_polygon(
    thetas: np.ndarray, supports: np.ndarray
) -> np.ndarray:
    """Return vertices of the half-plane intersection implied by support values.

    Johnson sampling yields, for each direction ``theta``, the exact support
    value ``h(theta) = max Re(e^{i theta} z)`` over the numerical range.  The
    sampled boundary points themselves form an *inscribed* polygon, which
    under-states the range; the half-planes ``Re(e^{i theta} z) <= h(theta)``
    instead intersect in a polygon that provably *contains* it.

    The distinction decides verdict soundness.  If ``0`` lies in the numerical
    range and the enclosing disk is a genuine outer bound ``(c, R)``, then
    ``|c| = |c - 0| <= R`` forces ``disk_rate >= 1``, so an adequacy threshold
    below one can never certify an operator whose range contains the origin.
    Computing the disk from inscribed points loses that guarantee.
    """
    count = thetas.size
    if count < 3:
        raise ValueError("an outer polygon requires at least three directions")
    vertices: list[complex] = []
    for index in range(count):
        next_index = (index + 1) % count
        first, second = thetas[index], thetas[next_index]
        matrix = np.asarray(
            [
                [np.cos(first), -np.sin(first)],
                [np.cos(second), -np.sin(second)],
            ]
        )
        if abs(np.linalg.det(matrix)) <= 1.0e-12:
            # Parallel supports contribute no finite vertex.
            continue
        point = np.linalg.solve(
            matrix, np.asarray([supports[index], supports[next_index]])
        )
        vertices.append(complex(point[0], point[1]))
    if not vertices:
        raise ValueError("support directions produced no finite outer vertex")
    return np.asarray(vertices, dtype=np.complex128)
