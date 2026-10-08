"""Targeted matrix-free estimator for a pseudospectral connectivity window.

The dense pseudospectral criterion traces a full level set.  This
experimental refinement asks a cheaper question: can selected spectral
points be joined by paths on which ``sigma_min(z I - A)`` stays below a level
that excludes the origin?  It uses matrix-free Arnoldi iterations on the
shifted normal operator and adaptively samples only a minimum-spanning tree of
straight paths between spectral anchors.

The shifted smallest-singular-value function is 1-Lipschitz.  Rayleigh
quotients from the normal operator are upper estimates of its minimum
eigenvalue, so sampled values plus a Lipschitz interval bound provide a safe
upper bound along each tested path.  Exact eigenvalues and a trusted lower
bound on ``sigma_min(A)`` therefore give a sufficient qualitative
connectivity certificate for those paths.

A fully matrix-free run is deliberately labelled provisional: ordinary
Arnoldi supplies an upper estimate, not a certified lower bound, for
``sigma_min(A)`` at the origin, and a reduced Ritz spectrum need not contain
the full spectrum.  Also, sparse path probes do not construct a closed
pseudospectral contour, so they cannot supply the arc length required by the
Trefethen residual bound.  The estimator never invents that rate.

References: Trefethen and Embree, *Spectra and Pseudospectra* (Princeton,
2005), DOI 10.1515/9780691213101; Embree, "How Descriptive are GMRES
Convergence Bounds?", arXiv:2209.01231, DOI 10.48550/arXiv.2209.01231.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse.linalg

from moljax._precision import require_x64
from moljax.conditioning import LinearizedOperator, arnoldi, ritz_values


@dataclass(frozen=True)
class MatrixFreeSigmaMinResult:
    """One matrix-free upper estimate of ``sigma_min(z I - A)``."""

    point: complex
    estimate: float
    squared_rayleigh_quotient: float
    normal_residual_norm: float
    solver: str
    converged: bool
    arnoldi_steps: int
    forward_applications: int
    adjoint_applications: int
    forward_adjoint_pairs: int
    operator_applications: int
    elapsed_seconds: float


@dataclass(frozen=True)
class TargetedPseudospectralEstimate:
    """Result of a sparse path-connectivity pseudospectral estimate."""

    certified: bool
    provisional: bool
    verdict: str
    reason: str
    epsilon_zero_estimate: float
    epsilon_zero_lower_bound: float | None
    eps_connect_path_upper: float | None
    window_nonempty: bool
    spectrum_source: str
    spectrum_complete: bool
    spectrum_point_count: int
    anchor_count: int
    tree_edge_count: int
    bisection_iterations: int
    sigma_min_evaluations: int
    arnoldi_steps_per_evaluation: int
    forward_adjoint_pairs: int
    operator_applications: int
    elapsed_seconds: float
    rate_bound_available: bool
    rate_bound_reason: str
    sampled_path_maximum: float | None
    lipschitz_path_upper: float | None
    spectrum_coverage_upper: float
    evaluation_budget_exhausted: bool


def _normal_action(
    operator: LinearizedOperator,
    point: complex,
    vector: jax.Array,
) -> jax.Array:
    shifted = complex(point) * vector - operator.matvec(vector)
    return complex(point).conjugate() * shifted - operator.matvec_adjoint(shifted)


def _deterministic_start(n: int, seed: int) -> jax.Array:
    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed))
    vector = jax.random.normal(key_real, (n,), dtype=jnp.float64)
    vector = vector + 1j * jax.random.normal(key_imag, (n,), dtype=jnp.float64)
    return vector / jnp.linalg.norm(vector)


def estimate_operator_norm(
    operator: LinearizedOperator,
    *,
    power_steps: int = 12,
    seed: int = 20261024,
    safety_factor: float = 1.5,
) -> tuple[float, int, float]:
    """Return a reusable matrix-free scale for shifted normal iterations.

    The power estimate is multiplied by a safety factor to make the affine
    normal-operator transform numerically useful.  It is a scale, not a
    rigorous operator-norm upper bound and is never used as a certificate.
    """
    require_x64("matrix-free pseudospectral operator-norm estimate")
    if power_steps < 1:
        raise ValueError("power_steps must be positive")
    if safety_factor <= 1.0:
        raise ValueError("safety_factor must exceed one")
    vector = _deterministic_start(operator.n, seed)
    started = perf_counter()
    estimate_squared = 0.0
    for _ in range(power_steps):
        action = _normal_action(operator, 0.0j, vector)
        action = jax.block_until_ready(action)
        action_norm = float(jnp.linalg.norm(action))
        if action_norm == 0.0:
            return 0.0, power_steps, perf_counter() - started
        vector = action / action_norm
        estimate_squared = float(jnp.real(jnp.vdot(vector, _normal_action(operator, 0.0j, vector))))
    estimate = safety_factor * sqrt(max(estimate_squared, 0.0))
    return estimate, 4 * power_steps, perf_counter() - started


def estimate_sigma_min_matrix_free(
    operator: LinearizedOperator,
    point: complex,
    *,
    arnoldi_steps: int = 40,
    operator_norm_scale: float,
    seed: int = 20261025,
    refine_with_propack: bool = False,
    propack_tolerance: float = 1.0e-6,
    propack_max_iterations: int = 2000,
) -> MatrixFreeSigmaMinResult:
    """Estimate ``sigma_min(point I - A)`` with matrix-free normal actions.

    By default, Arnoldi is applied to ``I - B*B / scale**2``, where
    ``B = point I - A``.  Set ``refine_with_propack`` to use matrix-free
    Lanczos bidiagonalization when an accurate origin value or dense
    cross-check is required.  In either mode the returned value is recomputed
    as a normal-operator Rayleigh quotient, an upper bound on the true minimum
    squared singular value; the residual reports approximation quality.
    """
    require_x64("matrix-free pseudospectral sigma-min estimate")
    if not 1 <= arnoldi_steps <= operator.n:
        raise ValueError("arnoldi_steps must lie between one and the operator dimension")
    if not np.isfinite(operator_norm_scale) or operator_norm_scale <= 0.0:
        raise ValueError("operator_norm_scale must be positive and finite")
    if propack_tolerance <= 0.0:
        raise ValueError("propack_tolerance must be positive")
    if propack_max_iterations < 1:
        raise ValueError("propack_max_iterations must be positive")

    if refine_with_propack:
        forward_applications = 0
        adjoint_applications = 0

        def shifted_forward(vector: np.ndarray) -> np.ndarray:
            nonlocal forward_applications
            forward_applications += 1
            value = jnp.asarray(vector, dtype=jnp.complex128)
            result = complex(point) * value - operator.matvec(value)
            return np.asarray(jax.block_until_ready(result), dtype=np.complex128)

        def shifted_adjoint(vector: np.ndarray) -> np.ndarray:
            nonlocal adjoint_applications
            adjoint_applications += 1
            value = jnp.asarray(vector, dtype=jnp.complex128)
            result = complex(point).conjugate() * value - operator.matvec_adjoint(value)
            return np.asarray(jax.block_until_ready(result), dtype=np.complex128)

        shifted = scipy.sparse.linalg.LinearOperator(
            (operator.n, operator.n),
            matvec=shifted_forward,
            rmatvec=shifted_adjoint,
            dtype=np.complex128,
        )
        started = perf_counter()
        try:
            _, _, right_adjoint = scipy.sparse.linalg.svds(
                shifted,
                k=1,
                which="SM",
                tol=propack_tolerance,
                maxiter=propack_max_iterations,
                solver="propack",
                return_singular_vectors=True,
                random_state=seed,
            )
        except (ValueError, np.linalg.LinAlgError, scipy.sparse.linalg.ArpackNoConvergence):
            pass
        else:
            vector = np.asarray(right_adjoint.conjugate().T[:, 0], dtype=np.complex128)
            vector_norm = np.linalg.norm(vector)
            if np.isfinite(vector_norm) and vector_norm > 0.0:
                vector /= vector_norm
                forward = shifted_forward(vector)
                normal = shifted_adjoint(forward)
                rayleigh = max(float(np.vdot(vector, normal).real), 0.0)
                residual = float(np.linalg.norm(normal - rayleigh * vector))
                elapsed = perf_counter() - started
                pairs = min(forward_applications, adjoint_applications)
                return MatrixFreeSigmaMinResult(
                    point=complex(point),
                    estimate=sqrt(rayleigh),
                    squared_rayleigh_quotient=rayleigh,
                    normal_residual_norm=residual,
                    solver="propack",
                    converged=True,
                    arnoldi_steps=0,
                    forward_applications=forward_applications,
                    adjoint_applications=adjoint_applications,
                    forward_adjoint_pairs=pairs,
                    operator_applications=forward_applications + adjoint_applications,
                    elapsed_seconds=elapsed,
                )

    shifted_scale = (operator_norm_scale + abs(complex(point))) ** 2

    def transformed_action(vector: jax.Array) -> jax.Array:
        normal = _normal_action(operator, point, vector)
        return vector - normal / shifted_scale

    started = perf_counter()
    coverage = arnoldi(
        transformed_action,
        _deterministic_start(operator.n, seed),
        arnoldi_steps,
    )
    basis = coverage.basis
    hessenberg = coverage.hessenberg
    projection = np.asarray(hessenberg[: hessenberg.shape[1], :], dtype=np.complex128)
    projection = 0.5 * (projection + projection.conjugate().T)
    _, eigenvectors = np.linalg.eigh(projection)
    coefficients = jnp.asarray(eigenvectors[:, -1], dtype=jnp.complex128)
    vector = basis[:, : hessenberg.shape[1]] @ coefficients
    vector = vector / jnp.linalg.norm(vector)
    normal_vector = _normal_action(operator, point, vector)
    normal_vector = jax.block_until_ready(normal_vector)
    rayleigh = max(float(jnp.real(jnp.vdot(vector, normal_vector))), 0.0)
    residual = float(jnp.linalg.norm(normal_vector - rayleigh * vector))
    elapsed = perf_counter() - started
    pairs = int(hessenberg.shape[1]) + 1
    return MatrixFreeSigmaMinResult(
        point=complex(point),
        estimate=sqrt(rayleigh),
        squared_rayleigh_quotient=rayleigh,
        normal_residual_norm=residual,
        solver="normal_arnoldi",
        converged=True,
        arnoldi_steps=int(hessenberg.shape[1]),
        forward_applications=pairs,
        adjoint_applications=pairs,
        forward_adjoint_pairs=pairs,
        operator_applications=2 * pairs,
        elapsed_seconds=elapsed,
    )


def matrix_free_ritz_values(
    operator: LinearizedOperator,
    *,
    arnoldi_steps: int = 12,
    seed: int = 20261026,
) -> tuple[np.ndarray, int, float]:
    """Return a small forward-only Ritz spectrum for provisional runtime use."""
    require_x64("matrix-free pseudospectral Ritz estimate")
    if not 1 <= arnoldi_steps <= operator.n:
        raise ValueError("arnoldi_steps must lie between one and the operator dimension")
    started = perf_counter()
    coverage = arnoldi(
        operator.matvec,
        _deterministic_start(operator.n, seed),
        arnoldi_steps,
    )
    hessenberg = coverage.hessenberg
    values = np.asarray(ritz_values(hessenberg), dtype=np.complex128)
    return values, int(hessenberg.shape[1]), perf_counter() - started


def _farthest_point_anchors(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count < 1:
        raise ValueError("anchor_count must be positive")
    count = min(count, points.size)
    centroid = np.mean(points)
    selected = [int(np.argmax(np.abs(points - centroid)))]
    distances = np.abs(points - points[selected[0]])
    while len(selected) < count:
        candidate = int(np.argmax(distances))
        if candidate in selected:
            break
        selected.append(candidate)
        distances = np.minimum(distances, np.abs(points - points[candidate]))
    indices = np.asarray(selected, dtype=int)
    return points[indices], indices


def _minimum_spanning_tree(points: np.ndarray) -> list[tuple[int, int]]:
    if points.size < 2:
        return []
    in_tree = np.zeros(points.size, dtype=bool)
    in_tree[0] = True
    distances = np.abs(points - points[0])
    parents = np.zeros(points.size, dtype=int)
    distances[0] = np.inf
    edges: list[tuple[int, int]] = []
    while len(edges) < points.size - 1:
        candidates = np.where(in_tree, np.inf, distances)
        second = int(np.argmin(candidates))
        if not np.isfinite(candidates[second]):
            raise RuntimeError("failed to construct a spanning tree")
        edges.append((int(parents[second]), second))
        in_tree[second] = True
        new_distances = np.abs(points - points[second])
        improved = (~in_tree) & (new_distances < distances)
        parents[improved] = second
        distances[improved] = new_distances[improved]
    return edges


def _interval_upper(first: tuple[complex, float], second: tuple[complex, float]) -> float:
    distance = abs(second[0] - first[0])
    return max(first[1], second[1], 0.5 * (first[1] + second[1] + distance))


class _CachedSigmaEstimator:
    def __init__(
        self,
        operator: LinearizedOperator,
        *,
        arnoldi_steps: int,
        operator_norm_scale: float,
        seed: int,
        maximum_evaluations: int,
    ) -> None:
        self.operator = operator
        self.arnoldi_steps = arnoldi_steps
        self.operator_norm_scale = operator_norm_scale
        self.seed = seed
        self.maximum_evaluations = maximum_evaluations
        self.cache: dict[tuple[float, float], MatrixFreeSigmaMinResult] = {}

    @staticmethod
    def _key(point: complex) -> tuple[float, float]:
        value = complex(point)
        return round(value.real, 15), round(value.imag, 15)

    def add_exact_zero(self, point: complex) -> None:
        self.cache[self._key(point)] = MatrixFreeSigmaMinResult(
            point=complex(point),
            estimate=0.0,
            squared_rayleigh_quotient=0.0,
            normal_residual_norm=0.0,
            solver="exact_eigenvalue",
            converged=True,
            arnoldi_steps=0,
            forward_applications=0,
            adjoint_applications=0,
            forward_adjoint_pairs=0,
            operator_applications=0,
            elapsed_seconds=0.0,
        )

    def evaluate(
        self,
        point: complex,
        *,
        refine_with_propack: bool = False,
    ) -> MatrixFreeSigmaMinResult | None:
        key = self._key(point)
        if key in self.cache and (not refine_with_propack or self.cache[key].solver == "propack"):
            return self.cache[key]
        evaluated = sum(result.solver != "exact_eigenvalue" for result in self.cache.values())
        if evaluated >= self.maximum_evaluations:
            return None
        result = estimate_sigma_min_matrix_free(
            self.operator,
            point,
            arnoldi_steps=self.arnoldi_steps,
            operator_norm_scale=self.operator_norm_scale,
            seed=self.seed + evaluated,
            refine_with_propack=refine_with_propack,
        )
        self.cache[key] = result
        return result

    @property
    def computed_results(self) -> list[MatrixFreeSigmaMinResult]:
        return [result for result in self.cache.values() if result.solver != "exact_eigenvalue"]


def _path_below_level(
    first: tuple[complex, float],
    second: tuple[complex, float],
    level: float,
    estimator: _CachedSigmaEstimator,
) -> bool | None:
    if max(first[1], second[1]) > level:
        return False
    if _interval_upper(first, second) <= level:
        return True
    midpoint = 0.5 * (first[0] + second[0])
    result = estimator.evaluate(midpoint)
    if result is None:
        return None
    middle = (midpoint, result.estimate)
    if result.estimate > level:
        return False
    left = _path_below_level(first, middle, level, estimator)
    if left is not True:
        return left
    return _path_below_level(middle, second, level, estimator)


def _tree_below_level(
    anchors: np.ndarray,
    endpoint_values: np.ndarray,
    edges: Sequence[tuple[int, int]],
    level: float,
    coverage_upper: float,
    estimator: _CachedSigmaEstimator,
) -> bool | None:
    if coverage_upper > level:
        return False
    for first_index, second_index in edges:
        status = _path_below_level(
            (complex(anchors[first_index]), float(endpoint_values[first_index])),
            (complex(anchors[second_index]), float(endpoint_values[second_index])),
            level,
            estimator,
        )
        if status is not True:
            return status
    return True


def _sampled_and_lipschitz_maxima(
    anchors: np.ndarray,
    endpoint_values: np.ndarray,
    edges: Sequence[tuple[int, int]],
    estimator: _CachedSigmaEstimator,
) -> tuple[float, float]:
    sampled = float(np.max(endpoint_values)) if endpoint_values.size else 0.0
    upper = 0.0
    for first_index, second_index in edges:
        first_point = complex(anchors[first_index])
        second_point = complex(anchors[second_index])
        points = [(first_point, float(endpoint_values[first_index]))]
        direction = second_point - first_point
        for result in estimator.cache.values():
            if result.arnoldi_steps == 0:
                continue
            if abs(direction) == 0.0:
                continue
            coordinate = (result.point - first_point) / direction
            if abs(coordinate.imag) <= 1.0e-10 and -1.0e-12 <= coordinate.real <= 1.0 + 1.0e-12:
                points.append((result.point, result.estimate))
        points.append((second_point, float(endpoint_values[second_index])))
        points.sort(key=lambda item: abs(item[0] - first_point))
        sampled = max(sampled, *(value for _, value in points))
        upper = max(
            upper,
            *(
                _interval_upper(first, second)
                for first, second in zip(points, points[1:], strict=False)
            ),
        )
    return sampled, upper


def assess_targeted_pseudospectral_connectivity(
    operator: LinearizedOperator,
    spectrum_points: Sequence[complex] | np.ndarray,
    *,
    spectrum_source: str,
    spectrum_complete: bool,
    exact_spectrum_points: bool,
    epsilon_zero_lower_bound: float | None = None,
    arnoldi_steps: int = 40,
    operator_norm_scale: float,
    anchor_count: int = 12,
    maximum_sigma_min_evaluations: int = 64,
    bisection_iterations: int = 10,
    seed: int = 20261027,
    refine_origin_with_propack: bool = True,
) -> TargetedPseudospectralEstimate:
    """Assess a sparse spectrum-connectivity path below the origin threshold.

    ``certified`` can be true only when the supplied spectrum is complete,
    its points are exact eigenvalues, and ``epsilon_zero_lower_bound`` is a
    trusted positive lower bound.  Omitting any of these runs the same search
    as a provisional estimator and cannot produce a theorem-level verdict.
    """
    require_x64("targeted matrix-free pseudospectral assessment")
    points = np.asarray(spectrum_points, dtype=np.complex128).ravel()
    if points.size == 0 or not np.all(np.isfinite(points)):
        raise ValueError("spectrum_points must be a nonempty finite sequence")
    if maximum_sigma_min_evaluations < 1:
        raise ValueError("maximum_sigma_min_evaluations must be positive")
    if bisection_iterations < 1:
        raise ValueError("bisection_iterations must be positive")
    if epsilon_zero_lower_bound is not None and epsilon_zero_lower_bound < 0.0:
        raise ValueError("epsilon_zero_lower_bound must be nonnegative")

    started = perf_counter()
    estimator = _CachedSigmaEstimator(
        operator,
        arnoldi_steps=arnoldi_steps,
        operator_norm_scale=operator_norm_scale,
        seed=seed,
        maximum_evaluations=maximum_sigma_min_evaluations,
    )
    anchors, selected = _farthest_point_anchors(points, anchor_count)
    edges = _minimum_spanning_tree(anchors)
    if exact_spectrum_points:
        endpoint_values = np.zeros(anchors.size, dtype=float)
        for point in anchors:
            estimator.add_exact_zero(complex(point))
        nearest = np.min(np.abs(points[:, None] - anchors[None, :]), axis=1)
        coverage_upper = 0.5 * float(np.max(nearest))
    else:
        endpoint_results = [estimator.evaluate(complex(point)) for point in anchors]
        if any(result is None for result in endpoint_results):
            raise RuntimeError("evaluation budget is smaller than the requested anchor count")
        endpoint_values = np.asarray(
            [result.estimate for result in endpoint_results if result is not None], dtype=float
        )
        coverage_upper = 0.0

    origin_result = estimator.evaluate(
        0.0j,
        refine_with_propack=refine_origin_with_propack,
    )
    if origin_result is None:
        raise RuntimeError("evaluation budget was exhausted before the origin estimate")
    epsilon_estimate = origin_result.estimate
    threshold = (
        float(epsilon_zero_lower_bound)
        if epsilon_zero_lower_bound is not None
        else epsilon_estimate
    )
    high = threshold * (1.0 - 16.0 * np.finfo(float).eps)
    high_status = _tree_below_level(
        anchors,
        endpoint_values,
        edges,
        high,
        coverage_upper,
        estimator,
    )
    budget_exhausted = high_status is None
    path_upper: float | None = None
    if high_status is True:
        low = 0.0
        for _ in range(bisection_iterations):
            midpoint = 0.5 * (low + high)
            status = _tree_below_level(
                anchors,
                endpoint_values,
                edges,
                midpoint,
                coverage_upper,
                estimator,
            )
            if status is True:
                high = midpoint
            else:
                low = midpoint
                budget_exhausted = budget_exhausted or status is None
        path_upper = high

    sampled_maximum, lipschitz_upper = _sampled_and_lipschitz_maxima(
        anchors,
        endpoint_values,
        edges,
        estimator,
    )
    if path_upper is not None:
        path_upper = max(path_upper, coverage_upper)
        sampled_maximum = min(sampled_maximum, path_upper)
        lipschitz_upper = min(lipschitz_upper, path_upper)
    window_nonempty = path_upper is not None and path_upper < threshold
    prerequisite_complete = (
        spectrum_complete and exact_spectrum_points and epsilon_zero_lower_bound is not None
    )
    certified = bool(window_nonempty and prerequisite_complete)
    provisional = not certified
    if certified:
        verdict = "certified"
        reason = "tested spectrum-spanning paths lie below a trusted origin-exclusion level"
    elif window_nonempty:
        verdict = "provisional_window"
        reason = (
            "a sparse path window was found, but reduced spectrum and/or matrix-free "
            "origin estimates do not prove full-spectrum origin exclusion"
        )
    elif budget_exhausted:
        verdict = "unresolved"
        reason = "the sparse path test exhausted its matrix-free evaluation budget"
    else:
        verdict = "not_certified"
        reason = "no tested spectrum-spanning path was established below the origin threshold"

    results = estimator.computed_results
    return TargetedPseudospectralEstimate(
        certified=certified,
        provisional=provisional,
        verdict=verdict,
        reason=reason,
        epsilon_zero_estimate=epsilon_estimate,
        epsilon_zero_lower_bound=epsilon_zero_lower_bound,
        eps_connect_path_upper=path_upper,
        window_nonempty=bool(window_nonempty),
        spectrum_source=spectrum_source,
        spectrum_complete=spectrum_complete,
        spectrum_point_count=int(points.size),
        anchor_count=int(selected.size),
        tree_edge_count=len(edges),
        bisection_iterations=bisection_iterations,
        sigma_min_evaluations=len(results),
        arnoldi_steps_per_evaluation=arnoldi_steps,
        forward_adjoint_pairs=sum(result.forward_adjoint_pairs for result in results),
        operator_applications=sum(result.operator_applications for result in results),
        elapsed_seconds=perf_counter() - started,
        rate_bound_available=False,
        rate_bound_reason=(
            "sparse path probes do not construct the closed contour or arc length "
            "required by the Trefethen pseudospectral residual bound"
        ),
        sampled_path_maximum=sampled_maximum,
        lipschitz_path_upper=lipschitz_upper,
        spectrum_coverage_upper=coverage_upper,
        evaluation_budget_exhausted=budget_exhausted,
    )


__all__ = [
    "MatrixFreeSigmaMinResult",
    "TargetedPseudospectralEstimate",
    "assess_targeted_pseudospectral_connectivity",
    "estimate_operator_norm",
    "estimate_sigma_min_matrix_free",
    "matrix_free_ritz_values",
]
