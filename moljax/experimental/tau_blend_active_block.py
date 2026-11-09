"""Pure dense helpers for the experimental tau-blend active-block study."""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np
import scipy.linalg

from moljax.conditioning import assess_preconditioner, real_bulk_outliers
from moljax.conditioning._geometry import (
    _origin_enclosed,
    _smallest_enclosing_disk,
    _support_outer_polygon,
)


@dataclass(frozen=True)
class ActiveBlockDecomposition:
    """The permuted blocks of ``MJ = [[I + E, X], [F, Y]]``."""

    permutation: np.ndarray
    inverse_permutation: np.ndarray
    inactive_count: int
    E: np.ndarray
    F: np.ndarray
    X: np.ndarray
    Y: np.ndarray
    lemma1_constant: float


@dataclass(frozen=True)
class Lemma6Bound:
    """One evaluated lower bound on the small-eigenvalue tail."""

    modal_threshold: float
    modes_above_threshold: int
    rank_correction: int
    lower_bound: int


@dataclass(frozen=True)
class DenseFovDisk:
    """Exact dense support samples and their enclosing-disk policy inputs."""

    angles: np.ndarray
    supports: np.ndarray
    boundary: np.ndarray
    center: complex
    radius: float
    disk_rate: float
    origin_enclosed: bool


def extract_active_block(
    preconditioned_operator: np.ndarray,
    inactive_mask: np.ndarray,
) -> ActiveBlockDecomposition:
    """Permute inactive inputs first and return all four block deviations."""
    matrix = np.asarray(preconditioned_operator)
    mask = np.asarray(inactive_mask, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("preconditioned_operator must be square")
    if mask.shape != (matrix.shape[0],):
        raise ValueError("inactive_mask must match the operator dimension")
    permutation = np.concatenate((np.flatnonzero(mask), np.flatnonzero(~mask)))
    inverse = np.argsort(permutation)
    permuted = matrix[np.ix_(permutation, permutation)]
    inactive_count = int(mask.sum())
    active_count = matrix.shape[0] - inactive_count
    E = permuted[:inactive_count, :inactive_count] - np.eye(
        inactive_count,
        dtype=matrix.dtype,
    )
    X = permuted[:inactive_count, inactive_count:]
    F = permuted[inactive_count:, :inactive_count]
    Y = permuted[inactive_count:, inactive_count:]
    stacked = np.vstack((X, np.eye(active_count, dtype=matrix.dtype) - Y))
    constant = float(np.linalg.norm(stacked, 2)) if active_count else 0.0
    return ActiveBlockDecomposition(
        permutation=permutation,
        inverse_permutation=inverse,
        inactive_count=inactive_count,
        E=E,
        F=F,
        X=X,
        Y=Y,
        lemma1_constant=constant,
    )


def lemma6_tail_lower_bound(
    diffusion_mode_magnitudes: np.ndarray,
    *,
    diffusivity_floor: float,
    scalar_reference: float,
    dt: float,
    magnitude_threshold: float,
    rank_correction: int,
) -> Lemma6Bound:
    """Evaluate Lemma 6's rank-corrected lower bound."""
    a = float(magnitude_threshold)
    denominator = dt * (scalar_reference * a - diffusivity_floor)
    modal_threshold = math.inf if denominator <= 0.0 else (1.0 - a) / denominator
    modes = int(np.count_nonzero(np.asarray(diffusion_mode_magnitudes) > modal_threshold))
    lower = max(0, modes - int(rank_correction))
    return Lemma6Bound(modal_threshold, modes, int(rank_correction), lower)


def lemma1_iteration_bound(
    *,
    polynomial_degree: int,
    contour_modulus_bound: float,
    theorem_prefactor: float,
    block_constant: float,
    target_tolerance: float,
) -> int | None:
    """Return the Lemma 1' full-operator iteration bound, including its first step."""
    factor = float(contour_modulus_bound)
    combined = float(block_constant) * float(theorem_prefactor)
    if factor < 0.0 or factor >= 1.0 or combined < 0.0 or target_tolerance <= 0.0:
        return None
    if combined <= target_tolerance:
        active_iterations = 0
    elif factor == 0.0:
        active_iterations = int(polynomial_degree)
    else:
        active_iterations = int(polynomial_degree) * max(
            0,
            math.ceil(math.log(target_tolerance / combined) / math.log(factor)),
        )
    return 1 + active_iterations


def unrestarted_gmres(
    operator: np.ndarray,
    rhs: np.ndarray,
    tolerance: float,
    max_iterations: int,
    *,
    secondary: tuple[np.ndarray, np.ndarray] | None = None,
    stop: str = "primary",
    checkpoints: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Run the study's twice-reorthogonalized, unrestarted dense GMRES."""
    if stop not in ("primary", "secondary", "both"):
        raise ValueError("stop must be 'primary', 'secondary' or 'both'")
    if stop != "primary" and secondary is None:
        raise ValueError("this stop rule needs a secondary residual")
    matrix = np.asarray(operator)
    right_hand_side = np.asarray(rhs)
    dimension = matrix.shape[0]
    beta = float(np.linalg.norm(right_hand_side))
    result: dict[str, Any] = {
        "iterations": 0,
        "iterations_primary": None,
        "iterations_secondary": None,
        "breakdown": False,
        "checkpoints": {},
    }
    if beta == 0.0:
        result.update(
            iterations_primary=0,
            iterations_secondary=0,
            x=np.zeros(dimension),
            primary_ratio=0.0,
            secondary_ratio=0.0,
        )
        return result
    if secondary is not None:
        secondary_operator, secondary_rhs = secondary
        secondary_norm = float(np.linalg.norm(secondary_rhs))
        secondary_basis = np.zeros((dimension, max_iterations + 1))
    basis = np.zeros((dimension, max_iterations + 1))
    hessenberg = np.zeros((max_iterations + 1, max_iterations))
    cosines = np.zeros(max_iterations)
    sines = np.zeros(max_iterations)
    rotated_rhs = np.zeros(max_iterations + 1)
    rotated_rhs[0] = beta
    basis[:, 0] = right_hand_side / beta
    if secondary is not None:
        secondary_basis[:, 0] = secondary_operator @ basis[:, 0]
    pending = sorted(set(checkpoints), reverse=True)
    coefficients = np.zeros(0)
    primary_ratio = 1.0
    secondary_ratio = 1.0
    for column in range(max_iterations):
        vector = matrix @ basis[:, column]
        vector_norm = float(np.linalg.norm(vector))
        for _ in range(2):
            projection = basis[:, : column + 1].T @ vector
            vector = vector - basis[:, : column + 1] @ projection
            hessenberg[: column + 1, column] += projection
        hessenberg[column + 1, column] = float(np.linalg.norm(vector))
        breakdown = hessenberg[column + 1, column] <= (
            64.0 * np.finfo(float).eps * max(vector_norm, np.finfo(float).tiny)
        )
        if not breakdown:
            basis[:, column + 1] = vector / hessenberg[column + 1, column]
            if secondary is not None:
                secondary_basis[:, column + 1] = secondary_operator @ basis[:, column + 1]
        for row in range(column):
            temporary = (
                cosines[row] * hessenberg[row, column] + sines[row] * hessenberg[row + 1, column]
            )
            hessenberg[row + 1, column] = (
                -sines[row] * hessenberg[row, column] + cosines[row] * hessenberg[row + 1, column]
            )
            hessenberg[row, column] = temporary
        radius = math.hypot(hessenberg[column, column], hessenberg[column + 1, column])
        if radius == 0.0:
            cosines[column], sines[column] = 1.0, 0.0
        else:
            cosines[column] = hessenberg[column, column] / radius
            sines[column] = hessenberg[column + 1, column] / radius
        hessenberg[column, column] = radius
        hessenberg[column + 1, column] = 0.0
        rotated_rhs[column + 1] = -sines[column] * rotated_rhs[column]
        rotated_rhs[column] = cosines[column] * rotated_rhs[column]
        primary_ratio = abs(rotated_rhs[column + 1]) / beta
        iteration = column + 1
        needs_solution = (
            secondary is not None or pending or breakdown or iteration == max_iterations
        )
        if needs_solution or primary_ratio <= tolerance:
            coefficients = scipy.linalg.solve_triangular(
                hessenberg[:iteration, :iteration],
                rotated_rhs[:iteration],
            )
        if secondary is not None:
            secondary_ratio = (
                float(np.linalg.norm(secondary_rhs - secondary_basis[:, :iteration] @ coefficients))
                / secondary_norm
            )
        while pending and primary_ratio <= pending[0]:
            level = pending.pop(0)
            result["checkpoints"][repr(level)] = {
                "iterations": iteration,
                "primary_ratio": primary_ratio,
                "secondary_ratio": secondary_ratio if secondary is not None else None,
            }
        if result["iterations_primary"] is None and primary_ratio <= tolerance:
            result["iterations_primary"] = iteration
        if (
            secondary is not None
            and result["iterations_secondary"] is None
            and secondary_ratio <= tolerance
        ):
            result["iterations_secondary"] = iteration
        done = {
            "primary": result["iterations_primary"] is not None,
            "secondary": result["iterations_secondary"] is not None,
            "both": result["iterations_primary"] is not None
            and result["iterations_secondary"] is not None,
        }[stop]
        result["iterations"] = iteration
        if done or breakdown:
            result["breakdown"] = bool(breakdown)
            break
    completed = result["iterations"]
    if coefficients.size != completed:
        coefficients = scipy.linalg.solve_triangular(
            hessenberg[:completed, :completed],
            rotated_rhs[:completed],
        )
    solution = basis[:, :completed] @ coefficients
    result["x"] = solution
    result["primary_ratio"] = float(np.linalg.norm(right_hand_side - matrix @ solution) / beta)
    result["primary_ratio_estimate"] = float(primary_ratio)
    if secondary is not None:
        result["secondary_ratio"] = float(
            np.linalg.norm(secondary_rhs - secondary_operator @ solution) / secondary_norm
        )
    return result


def true_residual_gmres(
    preconditioner: np.ndarray,
    operator: np.ndarray,
    rhs: np.ndarray,
    *,
    tolerance: float,
    max_iterations: int | None = None,
) -> dict[str, Any]:
    """Run left-preconditioned GMRES until the true residual reaches tolerance."""
    maximum = operator.shape[0] if max_iterations is None else int(max_iterations)
    result = unrestarted_gmres(
        preconditioner @ operator,
        preconditioner @ rhs,
        tolerance,
        maximum,
        secondary=(operator, rhs),
        stop="secondary",
    )
    result["converged"] = bool(result["secondary_ratio"] <= tolerance)
    result["true_stop_count"] = result["iterations_secondary"]
    result["preconditioned_stop_count"] = result["iterations_primary"]
    return result


def dense_exact_support_fov(
    operator: np.ndarray,
    *,
    n_angles: int = 32,
) -> DenseFovDisk:
    """Compute exact dense support eigenproblems and the enclosing disk."""
    matrix = np.asarray(operator, dtype=np.complex128)
    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    boundary: list[complex] = []
    supports: list[float] = []
    for angle in angles:
        rotation = np.exp(1j * angle)
        hermitian = 0.5 * (rotation * matrix + np.conj(rotation) * matrix.conj().T)
        values, vectors = scipy.linalg.eigh(
            hermitian,
            subset_by_index=(matrix.shape[0] - 1, matrix.shape[0] - 1),
            check_finite=False,
            driver="evr",
        )
        vector = vectors[:, 0]
        supports.append(float(values[0]))
        boundary.append(complex(np.vdot(vector, matrix @ vector)))
    support_array = np.asarray(supports)
    boundary_array = np.asarray(boundary)
    outer = _support_outer_polygon(angles, support_array)
    center, radius = _smallest_enclosing_disk(outer)
    disk_rate = math.inf if center == 0.0 else float(radius / abs(center))
    origin_enclosed = bool(np.all(support_array >= 0.0))
    if not origin_enclosed:
        origin_enclosed = bool(_origin_enclosed(boundary_array))
    return DenseFovDisk(
        angles=angles,
        supports=support_array,
        boundary=boundary_array,
        center=complex(center),
        radius=float(radius),
        disk_rate=disk_rate,
        origin_enclosed=origin_enclosed,
    )


def emulate_merged_policy(
    operator: np.ndarray,
    fov: DenseFovDisk,
) -> dict[str, Any]:
    """Apply merged policy defaults to exact dense substitutes.

    Thresholds are read from :func:`assess_preconditioner` rather than copied.
    Exact eigenvalues substitute for Ritz values; dense ``sigma_min`` supplies
    full-operator evidence; exact support eigensolves are marked corroborated.
    Every substitution is serialized for auditability.
    """
    matrix = np.asarray(operator)
    parameters = inspect.signature(assess_preconditioner).parameters
    rate_threshold = float(parameters["rate_threshold"].default)
    epsilon_threshold = float(parameters["eps_zero_threshold"].default)
    max_outliers = int(parameters["max_right_real_outliers"].default)
    eigenvalues = scipy.linalg.eigvals(matrix, check_finite=False)
    epsilon_zero = float(scipy.linalg.svdvals(matrix, check_finite=False)[-1])
    outliers = int(real_bulk_outliers(jnp.asarray(eigenvalues)))
    if fov.origin_enclosed:
        verdict = "indeterminate"
        reason = "exact dense supports do not separate the origin"
    elif (
        fov.disk_rate <= rate_threshold
        and epsilon_zero >= epsilon_threshold
        and outliers <= max_outliers
    ):
        verdict = "adequate"
        reason = "all merged-policy gates pass under exact dense substitutions"
    else:
        verdict = "investigate"
        failures = []
        if fov.disk_rate > rate_threshold:
            failures.append("disk_rate")
        if epsilon_zero < epsilon_threshold:
            failures.append("epsilon_zero")
        if outliers > max_outliers:
            failures.append("right_real_outliers")
        reason = "failed gates: " + ", ".join(failures)
    return {
        "verdict": verdict,
        "reason": reason,
        "disk_rate": fov.disk_rate,
        "origin_enclosed": fov.origin_enclosed,
        "epsilon_zero": epsilon_zero,
        "n_right_real_outliers": outliers,
        "thresholds": {
            "rate_threshold": rate_threshold,
            "eps_zero_threshold": epsilon_threshold,
            "max_right_real_outliers": max_outliers,
        },
        "policy_emulation_notes": [
            "exact dense eigenvalues substitute for Arnoldi Ritz values",
            "dense sigma_min is full-operator epsilon-zero evidence",
            "exact Hermitian support eigenproblems substitute supports_consistent=True",
            "exact support solves substitute corroboration_attempted=True",
        ],
    }


__all__ = [
    "ActiveBlockDecomposition",
    "DenseFovDisk",
    "Lemma6Bound",
    "dense_exact_support_fov",
    "emulate_merged_policy",
    "extract_active_block",
    "lemma1_iteration_bound",
    "lemma6_tail_lower_bound",
    "true_residual_gmres",
    "unrestarted_gmres",
]
