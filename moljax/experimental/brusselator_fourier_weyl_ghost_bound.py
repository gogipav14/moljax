"""Fourier--Weyl--ghost lower bound for padded Brusselator BE operators.

This isolated experimental implementation evaluates the rigorous structural
bound proposed for the periodic two-field Brusselator backward-Euler
linearization.  It uses only the physical interior state and never
materializes the large diagnostic operator.  The dense builder at the end of
the module exists solely for small-grid adversarial validation.

The formulas are exact-arithmetic bounds.  Float64 evaluation follows the
same standard as moljax's dense conditioning helper; it is not a
directed-rounding interval enclosure.

The interior perturbation uses the spectral norm in the Weyl--Mirsky
inequality ``sigma_min(X + E) >= sigma_min(X) - ||E||_2``; see L. Mirsky,
"Symmetric gauge functions and unitarily invariant norms," *Quart. J.
Math.* 11 (1960), 50--59, doi:10.1093/qmath/11.1.50.  The ghost formula is
documented at ``ghost_structure_lower_bound``.  For unitarily-invariant-norm
background only (not as the source of either bound), see J.-C. Bourin,
"Matrix subadditivity inequalities and block-matrices," arXiv:0805.1954,
doi:10.48550/arXiv.0805.1954.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Array = np.ndarray
CandidateName = Literal["homogeneous", "mean", "minimax"]


@dataclass(frozen=True)
class BoundCandidate:
    """One valid constant-Jacobian choice in the Fourier--Weyl bound."""

    name: CandidateName
    center: tuple[float, float]
    b0: float
    perturbation_norm: float
    interior_lower_bound: float
    ghost_norm_bound: float
    full_lower_bound: float


@dataclass(frozen=True)
class FourierWeylGhostLowerBound:
    """Certificate data for all valid ``K0`` choices and their maximum."""

    selected: BoundCandidate
    candidates: tuple[BoundCandidate, ...]
    padding: str = "periodic n_ghost=1; ghost multiplicities in {0, 1, 3}"
    floating_point_standard: str = "float64 formulas; no directed-rounding enclosure"


def brusselator_jacobian(u: Array, v: Array, beta: float) -> Array:
    """Return pointwise reaction Jacobians with trailing ``(2, 2)`` axes."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must be same-shape two-dimensional interior arrays")
    product = 2.0 * u * v
    squared = u * u
    result = np.empty(u.shape + (2, 2), dtype=np.float64)
    result[..., 0, 0] = -(beta + 1.0) + product
    result[..., 0, 1] = squared
    result[..., 1, 0] = beta - product
    result[..., 1, 1] = -squared
    return result


def _stable_sigma_min_2x2(matrix: Array) -> float:
    """Return ``sigma_min`` as ``abs(det) / sigma_max`` for a 2-by-2 block."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise ValueError("expected a 2 by 2 matrix")
    sigma_max = float(np.linalg.svd(matrix, compute_uv=False)[0])
    if sigma_max == 0.0:
        return 0.0
    return float(abs(np.linalg.det(matrix)) / sigma_max)


def _fd_laplacian_half_symbol(ny: int, nx: int, dy: float, dx: float) -> Array:
    """Return the FD periodic symbol including zero and Nyquist modes."""
    if ny < 2 or nx < 2 or dy <= 0.0 or dx <= 0.0:
        raise ValueError("grid sizes must be >= 2 and spacings must be positive")
    ky = np.arange(ny, dtype=np.float64)[:, None]
    kx = np.arange(nx // 2 + 1, dtype=np.float64)[None, :]
    return (
        -4.0 * np.sin(np.pi * kx / nx) ** 2 / dx**2
        -4.0 * np.sin(np.pi * ky / ny) ** 2 / dy**2
    )


def _circle_from_two(first: Array, second: Array) -> tuple[Array, float]:
    center = 0.5 * (first + second)
    return center, float(np.linalg.norm(first - center))


def _circle_from_three(first: Array, second: Array, third: Array) -> tuple[Array, float] | None:
    """Return a circumcircle, or ``None`` for collinear inputs."""
    ax, ay = first
    bx, by = second
    cx, cy = third
    determinant = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    scale = max(1.0, float(np.max(np.abs((first, second, third)))))
    if abs(determinant) <= 32.0 * np.finfo(np.float64).eps * scale * scale:
        return None
    first_norm = ax * ax + ay * ay
    second_norm = bx * bx + by * by
    third_norm = cx * cx + cy * cy
    center = np.array(
        (
            (first_norm * (by - cy) + second_norm * (cy - ay) + third_norm * (ay - by))
            / determinant,
            (first_norm * (cx - bx) + second_norm * (ax - cx) + third_norm * (bx - ax))
            / determinant,
        ),
        dtype=np.float64,
    )
    return center, float(np.linalg.norm(first - center))


def _contains(center: Array, radius: float, points: Array) -> bool:
    scale = max(1.0, radius)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale
    return bool(np.all(np.linalg.norm(points - center, axis=1) <= radius + tolerance))


def smallest_enclosing_circle(points: Array) -> tuple[float, float]:
    """Return a deterministic randomized-incremental enclosing-circle center.

    Any returned center is valid for the lower bound.  The subsequent maximum
    distance is recomputed exactly from all points, so microscopic
    non-minimality in finite precision cannot weaken certificate safety.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points must have shape (n, 2) with n > 0")
    ordered = points[np.random.default_rng(0).permutation(len(points))]
    center = ordered[0].copy()
    radius = 0.0
    for outer, point in enumerate(ordered):
        if _contains(center, radius, point[None, :]):
            continue
        center = point.copy()
        radius = 0.0
        for middle in range(outer):
            second = ordered[middle]
            if _contains(center, radius, second[None, :]):
                continue
            center, radius = _circle_from_two(point, second)
            for inner in range(middle):
                third = ordered[inner]
                if _contains(center, radius, third[None, :]):
                    continue
                circumcircle = _circle_from_three(point, second, third)
                if circumcircle is None:
                    pairs = (
                        _circle_from_two(point, second),
                        _circle_from_two(point, third),
                        _circle_from_two(second, third),
                    )
                    viable = [
                        candidate
                        for candidate in pairs
                        if _contains(candidate[0], candidate[1], ordered[: middle + 1])
                    ]
                    center, radius = min(viable, key=lambda candidate: candidate[1])
                else:
                    center, radius = circumcircle
    return float(center[0]), float(center[1])


def _features(u: Array, v: Array) -> Array:
    return np.stack((2.0 * u * v, u * u), axis=-1).reshape(-1, 2)


def _k_from_feature(feature: tuple[float, float], beta: float) -> Array:
    s0, t0 = feature
    return np.array(((-(beta + 1.0) + s0, t0), (beta - s0, -t0)), dtype=np.float64)


def _ghost_multiplicity(ny: int, nx: int) -> Array:
    """Return periodic one-layer ghost-copy counts for physical nodes."""
    multiplicity = np.zeros((ny, nx), dtype=np.float64)
    multiplicity[[0, -1], :] += 1.0
    multiplicity[:, [0, -1]] += 1.0
    multiplicity[0, 0] += 1.0
    multiplicity[0, -1] += 1.0
    multiplicity[-1, 0] += 1.0
    multiplicity[-1, -1] += 1.0
    return multiplicity


def ghost_structure_lower_bound(interior_lower_bound: float, ghost_norm_bound: float) -> float:
    """Return ``g(b,c)`` for ``[[B,0],[C,I]]`` with ``σmin(B)>=b`` and ``||C||<=c``.

    The exact formula follows directly from the inverse block form:
    ``||A^-1||_2 <= ||[[1/b, 0], [c/b, 1]]||_2``.  The reciprocal of the
    scalar matrix's largest singular value is the ``g(b,c)`` below.  For
    related general singular-value inequalities for block-triangular
    matrices, see C.-K. Li and R. Mathias, Theorem 1, equation (1),
    *SIAM J. Matrix Anal. Appl.* 24 (2002), 126--131,
    doi:10.1137/S0895479801398517.  That theorem is contextual rather than
    the direct source of this closed form.
    """
    b = float(interior_lower_bound)
    c = float(ghost_norm_bound)
    if b <= 0.0:
        return 0.0
    if c < 0.0:
        raise ValueError("ghost_norm_bound must be non-negative")
    denominator = np.sqrt((b + 1.0) ** 2 + c * c) + np.sqrt((b - 1.0) ** 2 + c * c)
    return float(2.0 * b / denominator)


def _candidate_bound(
    name: CandidateName,
    center: tuple[float, float],
    *,
    u: Array,
    v: Array,
    du: float,
    dv: float,
    beta: float,
    dt: float,
    domain_length_x: float,
    domain_length_y: float,
) -> BoundCandidate:
    ny, nx = u.shape
    symbol = _fd_laplacian_half_symbol(ny, nx, domain_length_y / ny, domain_length_x / nx)
    k0 = _k_from_feature(center, beta)
    b0 = np.inf
    for ell in symbol.ravel():
        preconditioner = np.diag((1.0 - dt * du * ell, 1.0 - dt * dv * ell))
        constant_jacobian = np.eye(2) - dt * (np.diag((du, dv)) * ell + k0)
        b0 = min(b0, _stable_sigma_min_2x2(np.linalg.solve(preconditioner, constant_jacobian)))
    # K(x)-K0 = [1, -1]^T [delta_s, delta_t], so sqrt(2)*the Euclidean
    # feature distance is exactly its matrix spectral norm (not its radius).
    feature_distance = np.linalg.norm(_features(u, v) - np.asarray(center), axis=1)
    perturbation = float(np.sqrt(2.0) * np.max(feature_distance))
    interior = max(0.0, float(b0 - dt * perturbation))
    # Leading singular values are the pointwise spectral norms required for c.
    jacobian_norms = np.linalg.svd(brusselator_jacobian(u, v, beta), compute_uv=False)[..., 0]
    multiplicity = _ghost_multiplicity(ny, nx)
    ghost = float(dt * np.max(np.sqrt(multiplicity) * jacobian_norms))
    return BoundCandidate(
        name=name,
        center=(float(center[0]), float(center[1])),
        b0=float(b0),
        perturbation_norm=perturbation,
        interior_lower_bound=interior,
        ghost_norm_bound=ghost,
        full_lower_bound=ghost_structure_lower_bound(interior, ghost),
    )


def fourier_weyl_ghost_lower_bound(
    u: Array,
    v: Array,
    *,
    du: float,
    dv: float,
    a: float,
    beta: float,
    dt: float,
    domain_length_x: float = 5.0,
    domain_length_y: float | None = None,
) -> FourierWeylGhostLowerBound:
    """Certify a lower bound on padded ``sigma_min(P^-1 J)`` for Brusselator BE."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must be same-shape two-dimensional interior arrays")
    if min(du, dv, a, dt, domain_length_x) <= 0.0 or beta < 0.0:
        raise ValueError("diffusivities, a, dt, and domain lengths must be positive")
    if domain_length_y is None:
        domain_length_y = domain_length_x
    if domain_length_y <= 0.0:
        raise ValueError("domain_length_y must be positive")
    feature = _features(u, v)
    centers: tuple[tuple[CandidateName, tuple[float, float]], ...] = (
        ("homogeneous", (2.0 * beta, a * a)),
        ("mean", (float(np.mean(feature[:, 0])), float(np.mean(feature[:, 1])))),
        ("minimax", smallest_enclosing_circle(feature)),
    )
    candidates = tuple(
        _candidate_bound(
            name,
            center,
            u=u,
            v=v,
            du=du,
            dv=dv,
            beta=beta,
            dt=dt,
            domain_length_x=domain_length_x,
            domain_length_y=domain_length_y,
        )
        for name, center in centers
    )
    return FourierWeylGhostLowerBound(
        selected=max(candidates, key=lambda candidate: candidate.full_lower_bound),
        candidates=candidates,
    )


def dense_padded_preconditioned_operator(
    u: Array,
    v: Array,
    *,
    du: float,
    dv: float,
    beta: float,
    dt: float,
    domain_length_x: float = 5.0,
    domain_length_y: float | None = None,
) -> Array:
    """Build the exact small dense padded operator, for validation only."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("u and v must be same-shape two-dimensional interior arrays")
    if domain_length_y is None:
        domain_length_y = domain_length_x
    ny, nx = u.shape
    n_physical = nx * ny
    dx = domain_length_x / nx
    dy = domain_length_y / ny
    laplacian = np.zeros((n_physical, n_physical), dtype=np.float64)

    def index(row: int, column: int) -> int:
        return (row % ny) * nx + (column % nx)

    for row in range(ny):
        for column in range(nx):
            current = index(row, column)
            laplacian[current, current] = -2.0 / dx**2 - 2.0 / dy**2
            laplacian[current, index(row, column - 1)] += 1.0 / dx**2
            laplacian[current, index(row, column + 1)] += 1.0 / dx**2
            laplacian[current, index(row - 1, column)] += 1.0 / dy**2
            laplacian[current, index(row + 1, column)] += 1.0 / dy**2
    identity = np.eye(n_physical)
    p_inverse = np.block(
        [
            [np.linalg.inv(identity - dt * du * laplacian), np.zeros_like(identity)],
            [np.zeros_like(identity), np.linalg.inv(identity - dt * dv * laplacian)],
        ]
    )
    jacobian = brusselator_jacobian(u, v, beta).reshape(n_physical, 2, 2)
    kinetic = np.zeros((2 * n_physical, 2 * n_physical), dtype=np.float64)
    for point, block in enumerate(jacobian):
        kinetic[point, point] = block[0, 0]
        kinetic[point, n_physical + point] = block[0, 1]
        kinetic[n_physical + point, point] = block[1, 0]
        kinetic[n_physical + point, n_physical + point] = block[1, 1]
    diffusion = np.block(
        [
            [du * laplacian, np.zeros_like(laplacian)],
            [np.zeros_like(laplacian), dv * laplacian],
        ]
    )
    interior = p_inverse @ (np.eye(2 * n_physical) - dt * (diffusion + kinetic))
    ghost_positions = [
        (row, column)
        for row in range(ny + 2)
        for column in range(nx + 2)
        if row in {0, ny + 1} or column in {0, nx + 1}
    ]
    copies = np.zeros((len(ghost_positions), n_physical), dtype=np.float64)
    for ghost, (row, column) in enumerate(ghost_positions):
        copies[ghost, index(row - 1, column - 1)] = 1.0
    r_operator = np.block([[copies, np.zeros_like(copies)], [np.zeros_like(copies), copies]])
    coupling = -dt * r_operator @ kinetic
    n_ghost = 2 * len(ghost_positions)
    return np.block(
        [
            [interior, np.zeros((2 * n_physical, n_ghost))],
            [coupling, np.eye(n_ghost)],
        ]
    )


__all__ = [
    "BoundCandidate",
    "FourierWeylGhostLowerBound",
    "brusselator_jacobian",
    "dense_padded_preconditioned_operator",
    "fourier_weyl_ghost_lower_bound",
    "ghost_structure_lower_bound",
    "smallest_enclosing_circle",
]
