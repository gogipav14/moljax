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


def _contains(circle: tuple[complex, float], point: complex, tolerance: float) -> bool:
    """Return whether ``circle`` contains ``point`` to floating-point tolerance."""
    center, radius = circle
    return abs(point - center) <= radius + tolerance


def _smallest_enclosing_disk(points: np.ndarray) -> tuple[complex, float]:
    """Compute the exact minimum enclosing disk of a finite point set without SciPy."""
    values = [complex(value) for value in np.asarray(points, dtype=np.complex128)]
    if not values:
        raise ValueError("an enclosing disk requires at least one boundary point")
    scale = max(1.0, *(abs(value) for value in values))
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
