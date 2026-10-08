"""Dense pseudospectral criterion for small experimental operators.

The field-of-values disk test can abstain on a non-normal operator even when
its spectrum is well separated from zero. This module probes that gap with a
dense pseudospectral construction intended for validation-sized operators.

It samples ``sigma_min(z I - A)`` on a Cartesian grid. A union-find sweep
finds the first sampled level at which every eigenvalue seed belongs to one
connected component. The 1-Lipschitz property of the shifted smallest
singular value supplies a conservative grid-spacing correction to that
connectivity threshold. If the corrected threshold lies below
``sigma_min(A)``, a spectrum-enclosing pseudospectral component exists while
the origin remains outside it.

For levels in that window, a linear program constructs a normalized
polynomial on the sampled component. The Trefethen contour estimate uses the
prefactor ``L(Gamma_epsilon) / (2 pi epsilon)`` and no Crouzeix spectral-set
constant. This is a grid-resolved numerical certificate, not interval
arithmetic. Its dense resolvent-sampling cost is exposed because a certificate
can cost more than the Krylov solve it describes.

The resolvent-contour estimate follows Trefethen and Embree, *Spectra and
Pseudospectra* (Princeton, 2005), DOI 10.1515/9780691213101.  For the GMRES
interpretation and corrected extension of Oxford Technical Report 99/08, see
Embree, "How Descriptive are GMRES Convergence Bounds?", arXiv:2209.01231,
DOI 10.48550/arXiv.2209.01231; there is no SIAM J. Matrix Analysis and
Applications version of that report.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import ceil, log, pi
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy.interpolate
import scipy.linalg
import scipy.ndimage
import scipy.optimize

from moljax._precision import require_x64

Matvec = Callable[[jax.Array], jax.Array]


@dataclass(frozen=True)
class PseudospectralCriterionResult:
    """Result of the dense, grid-resolved pseudospectral criterion."""

    certified: bool
    reason: str
    matrix_dimension: int
    epsilon_zero: float
    eps_connect_grid: float
    eps_connect_upper: float
    selected_epsilon: float | None
    polynomial_degree: int | None
    polynomial_envelope_epsilon: float | None
    polynomial_constraint_point_count: int | None
    polynomial_sampled_max_modulus: float | None
    polynomial_effective_rate: float | None
    polynomial_coefficients: tuple[tuple[float, float], ...]
    theorem_prefactor: float | None
    target_tolerance: float
    predicted_iterations: int | None
    component_grid_point_count: int | None
    contour_arc_length_grid: float | None
    real_grid: tuple[float, ...]
    imag_grid: tuple[float, ...]
    sigma_min_grid: tuple[tuple[float, ...], ...]
    domain_real_min: float
    domain_real_max: float
    domain_imag_min: float
    domain_imag_max: float
    grid_points_per_axis: int
    grid_spacing_real: float
    grid_spacing_imag: float
    sigma_min_evaluations: int
    sigma_min_seconds: float
    polynomial_search_seconds: float


@dataclass(frozen=True)
class _PolynomialCandidate:
    epsilon: float
    degree: int
    envelope_epsilon: float
    constraint_point_count: int
    sampled_max_modulus: float
    effective_rate: float
    coefficients: tuple[tuple[float, float], ...]
    prefactor: float
    predicted_iterations: int
    component_grid_point_count: int
    arc_length: float


def materialize_dense_operator(matvec: Matvec, n: int) -> np.ndarray:
    """Materialize a real/complex linear action in the standard basis."""
    require_x64("dense pseudospectral operator materialization")
    if n < 1:
        raise ValueError("n must be positive")
    basis = jnp.eye(n, dtype=jnp.float64)
    actions = jax.jit(jax.vmap(matvec))(basis)
    return np.asarray(jax.block_until_ready(actions.T), dtype=np.complex128)


def _validated_square_matrix(matrix: np.ndarray) -> np.ndarray:
    square = np.asarray(matrix, dtype=np.complex128)
    if square.ndim != 2 or square.shape[0] != square.shape[1] or square.shape[0] == 0:
        raise ValueError("matrix must be a nonempty square array")
    if not np.all(np.isfinite(square)):
        raise ValueError("matrix must contain only finite values")
    return square


def dense_sigma_min(matrix: np.ndarray, point: complex) -> float:
    """Return ``sigma_min(point * I - matrix)`` for one dense matrix."""
    square = _validated_square_matrix(matrix)
    shifted = complex(point) * np.eye(square.shape[0], dtype=np.complex128) - square
    return float(scipy.linalg.svdvals(shifted, overwrite_a=True, check_finite=False)[-1])


def _pseudospectral_domain(
    eigenvalues: np.ndarray,
    padding_fraction: float,
) -> tuple[float, float, float, float]:
    real_min = float(np.min(eigenvalues.real))
    real_max = float(np.max(eigenvalues.real))
    real_span = max(real_max - real_min, 1.0)
    imag_extent = max(float(np.max(np.abs(eigenvalues.imag))), 0.25 * real_span)
    left = min(real_min - padding_fraction * real_span, -0.1 * real_span)
    right = real_max + padding_fraction * real_span
    imag_bound = (1.0 + padding_fraction) * imag_extent
    return left, right, -imag_bound, imag_bound


def _sample_sigma_min_grid(
    matrix: np.ndarray,
    real_grid: np.ndarray,
    imag_grid: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    values = np.empty((imag_grid.size, real_grid.size), dtype=float)
    real_matrix = float(np.max(np.abs(matrix.imag))) <= 1.0e-13
    started = perf_counter()
    evaluations = 0
    if real_matrix and np.allclose(imag_grid, -imag_grid[::-1], rtol=0.0, atol=1.0e-14):
        middle = imag_grid.size // 2
        for row in range(middle, imag_grid.size):
            for column, real in enumerate(real_grid):
                values[row, column] = dense_sigma_min(matrix, complex(real, imag_grid[row]))
                evaluations += 1
            mirror = imag_grid.size - 1 - row
            if mirror != row:
                values[mirror] = values[row]
    else:
        for row, imag in enumerate(imag_grid):
            for column, real in enumerate(real_grid):
                values[row, column] = dense_sigma_min(matrix, complex(real, imag))
                evaluations += 1
    return values, evaluations, perf_counter() - started


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = int(self.parent[item])
        if parent != item:
            self.parent[item] = self.find(parent)
        return int(self.parent[item])

    def union(self, first: int, second: int) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        if self.rank[root_first] < self.rank[root_second]:
            root_first, root_second = root_second, root_first
        self.parent[root_second] = root_first
        if self.rank[root_first] == self.rank[root_second]:
            self.rank[root_first] += 1


def _eigenvalue_seed_indices(
    real_grid: np.ndarray,
    imag_grid: np.ndarray,
    eigenvalues: np.ndarray,
) -> tuple[np.ndarray, float]:
    points = real_grid[None, :] + 1j * imag_grid[:, None]
    flat_points = points.ravel()
    seeds: list[int] = []
    maximum_distance = 0.0
    for eigenvalue in np.asarray(eigenvalues).ravel():
        seed = int(np.argmin(np.abs(flat_points - eigenvalue)))
        seeds.append(seed)
        maximum_distance = max(maximum_distance, float(abs(flat_points[seed] - eigenvalue)))
    return np.unique(np.asarray(seeds, dtype=int)), maximum_distance


def _grid_connectivity_threshold(
    sigma_min: np.ndarray,
    seeds: np.ndarray,
) -> float:
    """Connect eigenvalue seed nodes through sampled pseudospectral sublevels."""
    rows, columns = sigma_min.shape
    order = np.argsort(sigma_min.ravel(), kind="stable")
    active = np.zeros(rows * columns, dtype=bool)
    disjoint = _DisjointSet(rows * columns)
    seed_set = set(int(seed) for seed in seeds)
    active_seeds = 0
    for flat_index in order:
        flat = int(flat_index)
        active[flat] = True
        if flat in seed_set:
            active_seeds += 1
        row, column = divmod(flat, columns)
        for neighbor_row, neighbor_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if not (0 <= neighbor_row < rows and 0 <= neighbor_column < columns):
                continue
            neighbor = neighbor_row * columns + neighbor_column
            if active[neighbor]:
                disjoint.union(flat, neighbor)
        if active_seeds == seeds.size:
            first_root = disjoint.find(int(seeds[0]))
            if all(disjoint.find(int(seed)) == first_root for seed in seeds[1:]):
                return float(sigma_min.ravel()[flat])
    raise RuntimeError("eigenvalue seed components did not connect on the grid")


def _connected_component(
    sigma_min: np.ndarray,
    epsilon: float,
    seeds: np.ndarray,
) -> np.ndarray | None:
    inside = sigma_min <= epsilon
    labels, _ = scipy.ndimage.label(
        inside,
        structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    )
    seed_labels = np.unique(labels.ravel()[seeds])
    seed_labels = seed_labels[seed_labels != 0]
    if seed_labels.size != 1 or np.any(labels.ravel()[seeds] != seed_labels[0]):
        return None
    component = labels == seed_labels[0]
    if (
        np.any(component[0])
        or np.any(component[-1])
        or np.any(component[:, 0])
        or np.any(component[:, -1])
    ):
        return None
    return component


def _grid_boundary_arc_length(component: np.ndarray, dx: float, dy: float) -> float:
    padded = np.pad(component, 1, constant_values=False)
    horizontal_edges = np.count_nonzero(padded[1:, :] != padded[:-1, :])
    vertical_edges = np.count_nonzero(padded[:, 1:] != padded[:, :-1])
    return float(horizontal_edges * dx + vertical_edges * dy)


def _fit_polynomial(
    points: np.ndarray,
    degree: int,
) -> tuple[float, float, tuple[tuple[float, float], ...]] | None:
    """Minimize a square-envelope surrogate and report its max modulus."""
    scale = max(float(np.max(np.abs(points))), 1.0)
    powers = np.column_stack([(points / scale) ** order for order in range(1, degree + 1)])
    real = powers.real
    imag = powers.imag
    minus_one = -np.ones((points.size, 1))
    constraints = np.block(
        [
            [real, -imag, minus_one],
            [-real, imag, minus_one],
            [imag, real, minus_one],
            [-imag, -real, minus_one],
        ]
    )
    bounds = [(None, None)] * (2 * degree) + [(0.0, None)]
    objective = np.zeros(2 * degree + 1)
    objective[-1] = 1.0
    right_hand_side = np.concatenate(
        [
            -np.ones(points.size),
            np.ones(points.size),
            np.zeros(points.size),
            np.zeros(points.size),
        ]
    )
    result = scipy.optimize.linprog(
        objective,
        A_ub=constraints,
        b_ub=right_hand_side,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        return None
    coefficients = result.x[:degree] + 1j * result.x[degree : 2 * degree]
    polynomial = 1.0 + powers @ coefficients
    maximum = float(np.max(np.abs(polynomial)))
    effective_rate = maximum ** (1.0 / degree)
    serialized = tuple((float(value.real), float(value.imag)) for value in coefficients)
    return maximum, effective_rate, serialized


def _predicted_iterations(
    prefactor: float,
    block_factor: float,
    degree: int,
    tolerance: float,
) -> int | None:
    if not 0.0 <= block_factor < 1.0 or prefactor <= 0.0:
        return None
    if prefactor <= tolerance:
        return 0
    if block_factor == 0.0:
        return degree
    blocks = max(0, ceil(log(tolerance / prefactor) / log(block_factor)))
    return degree * blocks


def _search_polynomial_certificate(
    sigma_min: np.ndarray,
    real_grid: np.ndarray,
    imag_grid: np.ndarray,
    eigenvalues: np.ndarray,
    eps_connect_upper: float,
    epsilon_zero: float,
    target_tolerance: float,
    polynomial_degrees: tuple[int, ...],
    epsilon_samples: int,
) -> _PolynomialCandidate | None:
    dx = float(real_grid[1] - real_grid[0])
    dy = float(imag_grid[1] - imag_grid[0])
    interpolation_allowance = 0.5 * float(np.hypot(dx, dy))
    fine_real = np.linspace(real_grid[0], real_grid[-1], 2 * real_grid.size - 1)
    fine_imag = np.linspace(imag_grid[0], imag_grid[-1], 2 * imag_grid.size - 1)
    fine_points = fine_real[None, :] + 1j * fine_imag[:, None]
    fine_mesh = np.stack(
        np.meshgrid(fine_imag, fine_real, indexing="ij"),
        axis=-1,
    )
    interpolator = scipy.interpolate.RegularGridInterpolator(
        (imag_grid, real_grid),
        sigma_min,
        method="linear",
        bounds_error=True,
    )
    fine_sigma_min = interpolator(fine_mesh)
    fine_seeds, _ = _eigenvalue_seed_indices(fine_real, fine_imag, eigenvalues)
    gap = epsilon_zero - eps_connect_upper
    if gap <= 0.0:
        return None
    candidates: list[_PolynomialCandidate] = []
    fractions = np.linspace(0.15, 0.9, epsilon_samples)
    for fraction in fractions:
        epsilon = eps_connect_upper + float(fraction) * gap
        envelope_epsilon = epsilon + interpolation_allowance
        if envelope_epsilon >= epsilon_zero:
            continue
        component = _connected_component(fine_sigma_min, epsilon, fine_seeds)
        if component is None:
            continue
        envelope = _connected_component(
            fine_sigma_min,
            envelope_epsilon,
            fine_seeds,
        )
        if envelope is None:
            continue
        points = fine_points[envelope]
        arc_length = _grid_boundary_arc_length(component, 0.5 * dx, 0.5 * dy)
        # Trefethen's pseudospectral contour bound is
        # L(Gamma_epsilon) / (2 pi epsilon).  A Crouzeix constant belongs to
        # functional calculus over W(A), not to this resolvent-contour bound.
        prefactor = arc_length / (2.0 * pi * epsilon)
        for degree in polynomial_degrees:
            fitted = _fit_polynomial(points, degree)
            if fitted is None:
                continue
            maximum, effective_rate, coefficients = fitted
            predicted = _predicted_iterations(
                prefactor,
                maximum,
                degree,
                target_tolerance,
            )
            if predicted is None:
                continue
            candidates.append(
                _PolynomialCandidate(
                    epsilon=epsilon,
                    degree=degree,
                    envelope_epsilon=envelope_epsilon,
                    constraint_point_count=points.size,
                    sampled_max_modulus=maximum,
                    effective_rate=effective_rate,
                    coefficients=coefficients,
                    prefactor=prefactor,
                    predicted_iterations=predicted,
                    component_grid_point_count=int(np.count_nonzero(component)),
                    arc_length=arc_length,
                )
            )
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: (
            candidate.predicted_iterations,
            candidate.effective_rate,
            candidate.degree,
        ),
    )


def assess_dense_pseudospectral_criterion(
    matrix: np.ndarray,
    *,
    grid_points_per_axis: int = 61,
    domain_padding_fraction: float = 0.75,
    polynomial_degrees: tuple[int, ...] = (4, 8, 12, 16, 20),
    epsilon_samples: int = 6,
    target_tolerance: float = 1.0e-8,
) -> PseudospectralCriterionResult:
    """Assess one small dense operator with a pseudospectral criterion."""
    require_x64("dense pseudospectral criterion")
    square = _validated_square_matrix(matrix)
    if grid_points_per_axis < 5 or grid_points_per_axis % 2 == 0:
        raise ValueError("grid_points_per_axis must be an odd integer of at least 5")
    if domain_padding_fraction <= 0.0:
        raise ValueError("domain_padding_fraction must be positive")
    if not polynomial_degrees or min(polynomial_degrees) < 1:
        raise ValueError("polynomial_degrees must contain positive values")
    if epsilon_samples < 1:
        raise ValueError("epsilon_samples must be positive")
    if not 0.0 < target_tolerance < 1.0:
        raise ValueError("target_tolerance must lie in (0, 1)")

    eigenvalues = scipy.linalg.eigvals(square, check_finite=False)
    real_min, real_max, imag_min, imag_max = _pseudospectral_domain(
        eigenvalues,
        domain_padding_fraction,
    )
    real_grid = np.linspace(real_min, real_max, grid_points_per_axis)
    imag_grid = np.linspace(imag_min, imag_max, grid_points_per_axis)
    sigma_min, evaluations, sigma_seconds = _sample_sigma_min_grid(
        square,
        real_grid,
        imag_grid,
    )
    seeds, maximum_seed_distance = _eigenvalue_seed_indices(
        real_grid,
        imag_grid,
        eigenvalues,
    )
    eps_connect_grid = _grid_connectivity_threshold(sigma_min, seeds)
    dx = float(real_grid[1] - real_grid[0])
    dy = float(imag_grid[1] - imag_grid[0])
    eps_connect_upper = max(
        eps_connect_grid + 0.5 * max(dx, dy),
        maximum_seed_distance,
    )
    epsilon_zero = dense_sigma_min(square, 0.0)
    search_started = perf_counter()
    candidate = _search_polynomial_certificate(
        sigma_min,
        real_grid,
        imag_grid,
        eigenvalues,
        eps_connect_upper,
        epsilon_zero,
        target_tolerance,
        polynomial_degrees,
        epsilon_samples,
    )
    search_seconds = perf_counter() - search_started
    if eps_connect_upper >= epsilon_zero:
        certified = False
        reason = "spectrum connectivity reaches the origin-exclusion threshold"
    elif candidate is None:
        certified = False
        reason = "the connected window has no decaying sampled polynomial certificate"
    else:
        certified = True
        reason = "connected pseudospectral window excludes the origin"

    return PseudospectralCriterionResult(
        certified=certified,
        reason=reason,
        matrix_dimension=square.shape[0],
        epsilon_zero=epsilon_zero,
        eps_connect_grid=eps_connect_grid,
        eps_connect_upper=eps_connect_upper,
        selected_epsilon=None if candidate is None else candidate.epsilon,
        polynomial_degree=None if candidate is None else candidate.degree,
        polynomial_envelope_epsilon=(None if candidate is None else candidate.envelope_epsilon),
        polynomial_constraint_point_count=(
            None if candidate is None else candidate.constraint_point_count
        ),
        polynomial_sampled_max_modulus=(
            None if candidate is None else candidate.sampled_max_modulus
        ),
        polynomial_effective_rate=None if candidate is None else candidate.effective_rate,
        polynomial_coefficients=() if candidate is None else candidate.coefficients,
        theorem_prefactor=None if candidate is None else candidate.prefactor,
        target_tolerance=target_tolerance,
        predicted_iterations=None if candidate is None else candidate.predicted_iterations,
        component_grid_point_count=(
            None if candidate is None else candidate.component_grid_point_count
        ),
        contour_arc_length_grid=None if candidate is None else candidate.arc_length,
        real_grid=tuple(float(value) for value in real_grid),
        imag_grid=tuple(float(value) for value in imag_grid),
        sigma_min_grid=tuple(tuple(float(value) for value in row) for row in sigma_min),
        domain_real_min=real_min,
        domain_real_max=real_max,
        domain_imag_min=imag_min,
        domain_imag_max=imag_max,
        grid_points_per_axis=grid_points_per_axis,
        grid_spacing_real=dx,
        grid_spacing_imag=dy,
        sigma_min_evaluations=evaluations + 1,
        sigma_min_seconds=sigma_seconds,
        polynomial_search_seconds=search_seconds,
    )


__all__ = [
    "PseudospectralCriterionResult",
    "assess_dense_pseudospectral_criterion",
    "dense_sigma_min",
    "materialize_dense_operator",
]
