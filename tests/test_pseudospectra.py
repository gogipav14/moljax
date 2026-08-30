"""Dense-reference tests for forward-only Arnoldi pseudospectra diagnostics."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from moljax.conditioning import (
    arnoldi,
    epsilon_zero,
    pseudospectrum_dense,
    reduced_pseudospectrum,
    ritz_values,
)


def _grcar(n: int) -> np.ndarray:
    """Return the order-n Grcar matrix with three superdiagonals."""
    matrix = np.eye(n, dtype=np.complex128)
    for diagonal in range(1, 4):
        matrix += np.diag(np.ones(n - diagonal), k=diagonal)
    matrix += np.diag(-np.ones(n - 1), k=-1)
    return matrix


def _normal_matrix() -> np.ndarray:
    """Return a normal diagonal matrix with distinct complex eigenvalues."""
    return np.diag(
        np.array(
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
    )


def _matvec(matrix: np.ndarray):
    """Return a JAX-compatible forward action for a dense test matrix."""
    operator = jnp.asarray(matrix, dtype=jnp.complex128)
    return lambda vector: operator @ vector


def _starting_vector(n: int) -> jax.Array:
    """Return a deterministic cyclic starting vector with no zero entries."""
    return jnp.asarray(np.arange(1, n + 1), dtype=jnp.complex128) + 0.5j


def _maximum_matching_error(actual: jax.Array | np.ndarray, expected: np.ndarray) -> float:
    """Return the maximum nearest-neighbour error between two small spectra."""
    unmatched = list(np.asarray(expected, dtype=np.complex128))
    errors: list[float] = []
    for value in np.asarray(actual, dtype=np.complex128):
        index = int(np.argmin(np.abs(np.asarray(unmatched) - value)))
        errors.append(float(abs(value - unmatched.pop(index))))
    return max(errors, default=0.0)


def test_arnoldi_basis_is_orthonormal_and_satisfies_relation():
    """Twice-MGS Arnoldi returns the expected basis and Hessenberg relation."""
    matrix = _grcar(6)
    action = _matvec(matrix)
    basis, hessenberg = arnoldi(action, _starting_vector(6), 5)
    leading_basis = basis[:, :5]
    expected = jnp.asarray(matrix) @ leading_basis
    reconstructed = basis @ hessenberg

    np.testing.assert_allclose(
        np.asarray(leading_basis.conj().T @ leading_basis),
        np.eye(5),
        atol=1.0e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(reconstructed),
        np.asarray(expected),
        atol=1.0e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize("matrix", [_grcar(6), _normal_matrix()], ids=["grcar", "normal"])
def test_full_order_ritz_values_match_dense_eigenvalues(matrix: np.ndarray):
    """A full-order Arnoldi projection is similar to the dense operator."""
    basis, hessenberg = arnoldi(_matvec(matrix), _starting_vector(matrix.shape[0]), matrix.shape[0])
    del basis
    actual = ritz_values(hessenberg)

    assert _maximum_matching_error(actual, np.linalg.eigvals(matrix)) <= 1.0e-9


@pytest.mark.slow
def test_full_order_reduced_grid_matches_dense_grid():
    """Full-order reduced singular values equal the dense reference on a grid."""
    matrix = _grcar(6)
    real_grid = np.linspace(-1.0, 3.0, 7)
    imag_grid = np.linspace(-2.5, 2.5, 6)
    action = _matvec(matrix)
    _, hessenberg = arnoldi(action, _starting_vector(6), 6)
    reduced = reduced_pseudospectrum(hessenberg, real_grid, imag_grid)
    dense = pseudospectrum_dense(action, 6, real_grid, imag_grid)

    np.testing.assert_allclose(
        np.asarray(reduced), np.asarray(dense.sigma_min), atol=1.0e-9, rtol=0.0
    )


def test_epsilon_zero_matches_dense_smallest_singular_value():
    """The zero-entry threshold is the smallest singular value of the operator."""
    matrix = _grcar(6)
    _, hessenberg = arnoldi(_matvec(matrix), _starting_vector(6), 6)
    actual = epsilon_zero(hessenberg)

    expected = np.linalg.svd(matrix, compute_uv=False)[-1]
    assert actual == pytest.approx(expected, abs=1.0e-10)


def test_arnoldi_breakdown_trims_to_the_completed_invariant_block():
    """A start vector confined to one block returns only that block's Ritz data."""
    first_block = np.diag(np.array([1.0, 2.0, 4.0], dtype=np.complex128))
    second_block = np.diag(np.array([7.0, 9.0, 12.0], dtype=np.complex128))
    matrix = np.zeros((6, 6), dtype=np.complex128)
    matrix[:3, :3] = first_block
    matrix[3:, 3:] = second_block
    start = jnp.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=jnp.complex128)

    basis, hessenberg = arnoldi(_matvec(matrix), start, 5)
    actual_ritz = ritz_values(hessenberg)
    actual_epsilon = epsilon_zero(hessenberg)

    assert basis.shape == (6, 4)
    assert hessenberg.shape == (4, 3)
    assert _maximum_matching_error(actual_ritz, np.linalg.eigvals(first_block)) <= 1.0e-9
    expected_epsilon = np.linalg.svd(first_block, compute_uv=False)[-1]
    assert actual_epsilon == pytest.approx(expected_epsilon, abs=1.0e-10)
