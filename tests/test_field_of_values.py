"""Dense-reference tests for matrix-free numerical-range diagnostics."""

from __future__ import annotations

import math
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from moljax.conditioning import FieldOfValuesResult, numerical_range


def _matrix_actions(matrix: np.ndarray) -> tuple[Callable[[jax.Array], jax.Array], ...]:
    """Return JAX matrix-free forward and Euclidean-adjoint actions."""
    operator = jnp.asarray(matrix, dtype=jnp.complex128)
    return (
        lambda value: operator @ value,
        lambda value: operator.conj().T @ value,
    )


def _dense_boundary(matrix: np.ndarray, n_angles: int) -> np.ndarray:
    """Return dense Johnson support points as an independent reference."""
    boundary = np.empty(n_angles, dtype=np.complex128)
    for index in range(n_angles):
        theta = 2.0 * math.pi * index / n_angles
        phase = np.exp(1j * theta)
        hermitian = 0.5 * (phase * matrix + phase.conjugate() * matrix.conj().T)
        _, vectors = np.linalg.eigh(hermitian)
        vector = vectors[:, -1]
        boundary[index] = np.vdot(vector, matrix @ vector)
    return boundary


def _result(matrix: np.ndarray, n_angles: int = 12) -> FieldOfValuesResult:
    """Run the public diagnostic with float64 enabled for the test operator."""
    matvec, matvec_adjoint = _matrix_actions(matrix)
    return numerical_range(
        matvec,
        matvec_adjoint,
        matrix.shape[0],
        n_angles=n_angles,
    )


def _grcar(n: int) -> np.ndarray:
    """Return the order-n Grcar matrix with three superdiagonals."""
    matrix = np.eye(n, dtype=np.complex128)
    for diagonal in range(1, 4):
        matrix += np.diag(np.ones(n - diagonal), k=diagonal)
    matrix += np.diag(-np.ones(n - 1), k=-1)
    return matrix


def test_normal_boundary_matches_convex_hull_supports():
    """A normal operator's supports agree with its eigenvalue convex hull."""
    eigenvalues = np.array(
        [
            -1.1 - 0.4j,
            -0.2 + 1.3j,
            0.8 + 0.5j,
            1.9 - 0.2j,
            1.2 - 1.4j,
            -0.6 - 1.1j,
        ],
        dtype=np.complex128,
    )
    matrix = np.diag(eigenvalues)
    result = _result(matrix)
    expected = _dense_boundary(matrix, result.boundary.size)

    np.testing.assert_allclose(np.asarray(result.boundary), expected, atol=1.0e-9, rtol=0.0)
    for index, value in enumerate(np.asarray(result.boundary)):
        theta = 2.0 * math.pi * index / result.boundary.size
        support = np.max(np.real(np.exp(1j * theta) * eigenvalues))
        assert abs(np.real(np.exp(1j * theta) * value) - support) <= 1.0e-9


@pytest.mark.parametrize(
    "matrix",
    [
        _grcar(6),
        np.eye(6, dtype=np.complex128) + np.diag(np.ones(5), k=1),
        np.diag(np.linspace(1.0, 2.0, 6)) + 1.7 * np.diag(np.ones(5), k=1),
    ],
    ids=["grcar", "jordan", "bidiagonal-shift"],
)
def test_nonnormal_boundary_matches_dense_hermitian_reference(matrix: np.ndarray):
    """Matrix-free supports agree with dense Hermitian-part eigensolves."""
    result = _result(matrix)
    expected = _dense_boundary(matrix, result.boundary.size)
    np.testing.assert_allclose(np.asarray(result.boundary), expected, atol=1.0e-9, rtol=0.0)


def test_adjoint_identity():
    """The test actions satisfy the Euclidean complex-adjoint identity."""
    matrix = _grcar(6) + 0.25j * np.diag(np.linspace(1.0, 2.0, 6))
    generator = np.random.default_rng(20260819)
    vector = generator.standard_normal(6) + 1j * generator.standard_normal(6)
    cotangent = generator.standard_normal(6) + 1j * generator.standard_normal(6)

    matvec, matvec_adjoint = _matrix_actions(matrix)
    forward = np.asarray(matvec(jnp.asarray(vector)))
    adjoint = np.asarray(matvec_adjoint(jnp.asarray(cotangent)))

    assert abs(np.vdot(forward, cotangent) - np.vdot(vector, adjoint)) <= 1.0e-12


def test_origin_enclosure_and_disk_rate():
    """Known real intervals exercise origin membership and disk-rate reporting."""
    positive = np.diag(np.linspace(1.0, 6.0, 6)).astype(np.complex128)
    straddling = np.diag(np.linspace(-1.0, 5.0, 6)).astype(np.complex128)

    positive_result = _result(positive, n_angles=8)
    straddling_result = _result(straddling, n_angles=8)

    assert not positive_result.origin_enclosed
    assert positive_result.disk_rate == pytest.approx(5.0 / 7.0, abs=1.0e-12)
    assert straddling_result.origin_enclosed
    assert straddling_result.disk_rate == pytest.approx(1.5, abs=1.0e-12)
    assert positive_result.cp_prefactor == pytest.approx(1.0 + math.sqrt(2.0))
