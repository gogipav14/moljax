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
    full_operator_epsilon_zero,
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
    result = arnoldi(action, _starting_vector(6), 5)
    basis, hessenberg = result.basis, result.hessenberg
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
    assert result.k_requested == 5
    assert result.k_achieved == 5
    assert result.breakdown is False
    assert result.residual_norm < 1.0e-9


def test_arnoldi_result_supports_legacy_two_field_indexing():
    """``result[0]``/``result[1]`` still give ``basis``/``hessenberg``.

    ``ArnoldiResult`` replaced the bare ``(Q, H)`` tuple so coverage metadata
    travels with the factorization, but code that indexes rather than
    unpacks (``arnoldi(...)[1]``) must keep working.
    """
    matrix = _grcar(6)
    result = arnoldi(_matvec(matrix), _starting_vector(6), 5)
    assert result[0] is result.basis
    assert result[1] is result.hessenberg


@pytest.mark.parametrize("matrix", [_grcar(6), _normal_matrix()], ids=["grcar", "normal"])
def test_full_order_ritz_values_match_dense_eigenvalues(matrix: np.ndarray):
    """A full-order Arnoldi projection is similar to the dense operator."""
    result = arnoldi(_matvec(matrix), _starting_vector(matrix.shape[0]), matrix.shape[0])
    actual = ritz_values(result.hessenberg)

    assert _maximum_matching_error(actual, np.linalg.eigvals(matrix)) <= 1.0e-9
    assert result.k_achieved == matrix.shape[0]


@pytest.mark.slow
def test_full_order_reduced_grid_matches_dense_grid():
    """Full-order reduced singular values equal the dense reference on a grid."""
    matrix = _grcar(6)
    real_grid = np.linspace(-1.0, 3.0, 7)
    imag_grid = np.linspace(-2.5, 2.5, 6)
    action = _matvec(matrix)
    hessenberg = arnoldi(action, _starting_vector(6), 6).hessenberg
    reduced = reduced_pseudospectrum(hessenberg, real_grid, imag_grid)
    dense = pseudospectrum_dense(action, 6, real_grid, imag_grid)

    np.testing.assert_allclose(
        np.asarray(reduced), np.asarray(dense.sigma_min), atol=1.0e-9, rtol=0.0
    )


def test_epsilon_zero_matches_dense_smallest_singular_value():
    """The zero-entry threshold is the smallest singular value of the operator."""
    matrix = _grcar(6)
    hessenberg = arnoldi(_matvec(matrix), _starting_vector(6), 6).hessenberg
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

    result = arnoldi(_matvec(matrix), start, 5)
    basis, hessenberg = result.basis, result.hessenberg
    actual_ritz = ritz_values(hessenberg)
    actual_epsilon = epsilon_zero(hessenberg)

    assert basis.shape == (6, 4)
    assert hessenberg.shape == (4, 3)
    assert _maximum_matching_error(actual_ritz, np.linalg.eigvals(first_block)) <= 1.0e-9
    expected_epsilon = np.linalg.svd(first_block, compute_uv=False)[-1]
    assert actual_epsilon == pytest.approx(expected_epsilon, abs=1.0e-10)
    # Coverage metadata must reflect the trimmed reduction: only 3 of the 6
    # requested (and possible) dimensions were reached, because the start
    # vector never excites the second, decoupled block.
    assert result.k_requested == 5
    assert result.k_achieved == 3
    assert result.breakdown is True
    assert result.residual_norm < 1.0e-9


@pytest.mark.parametrize("matrix", [_grcar(8), _normal_matrix()], ids=["grcar", "normal"])
def test_full_operator_epsilon_zero_matches_the_dense_singular_value(matrix: np.ndarray):
    """The full-operator helper agrees with a direct dense SVD, without Arnoldi."""
    n = matrix.shape[0]
    actual = full_operator_epsilon_zero(_matvec(matrix), n)
    expected = np.linalg.svd(matrix, compute_uv=False)[-1]
    assert actual == pytest.approx(expected, abs=1.0e-10)


def test_full_operator_epsilon_zero_agrees_with_pseudospectrum_dense():
    """The helper's value equals ``pseudospectrum_dense``'s on the same operator.

    This also pins the ``_materialize`` refactor: both call paths must build
    the identical dense matrix and read off the identical smallest singular
    value.
    """
    matrix = _grcar(6)
    action = _matvec(matrix)
    real_grid = np.linspace(-1.0, 3.0, 3)
    imag_grid = np.linspace(-2.5, 2.5, 3)

    actual = full_operator_epsilon_zero(action, 6)
    dense = pseudospectrum_dense(action, 6, real_grid, imag_grid)
    assert actual == dense.epsilon_zero


def test_full_operator_epsilon_zero_sees_the_mode_a_reduced_arnoldi_misses():
    """A decoupled mode a reduced Arnoldi projection never excites is still seen.

    Same operator and start vector as
    ``test_arnoldi_breakdown_does_not_promote_a_reduced_epsilon_zero_to_adequate``
    in ``tests/test_non_normality.py``: ``A`` has a diagonal first entry with
    no coupling out of it, so ``v0 = [0, 1, ..., 1]`` never excites it and
    Arnoldi breaks down at dimension 7 of 8, reading a reduced
    ``epsilon_zero`` of about 0.598 that has nothing to do with the true
    ``sigma_min(A) = 0.05``.  The full-operator helper does not depend on
    ``v0`` at all, so it reads the true value directly.
    """
    diagonal = np.array([0.05, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    matrix = np.diag(diagonal)
    for row in range(1, 7):
        matrix[row, row + 1] = 0.02
    matrix = matrix.astype(np.complex128)
    action = _matvec(matrix)

    v0 = jnp.asarray([0.0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.complex128)
    arnoldi_result = arnoldi(action, v0, 8)
    assert arnoldi_result.k_achieved == 7
    assert arnoldi_result.breakdown is True
    reduced_epsilon = epsilon_zero(arnoldi_result.hessenberg)
    assert reduced_epsilon == pytest.approx(0.598, abs=5.0e-4)

    actual = full_operator_epsilon_zero(action, 8)
    assert actual == pytest.approx(0.05, abs=1.0e-9)


def test_full_operator_epsilon_zero_rejects_invalid_arguments():
    """``n`` must be positive and ``dtype`` must be complex128."""
    action = _matvec(_grcar(4))
    with pytest.raises(ValueError, match="n must be positive"):
        full_operator_epsilon_zero(action, 0)
    with pytest.raises(ValueError, match="dtype=jnp.complex128"):
        full_operator_epsilon_zero(action, 4, dtype=jnp.complex64)


@pytest.mark.parametrize("scale", [1.0, 1.0e-15])
def test_arnoldi_breakdown_tolerance_is_scale_relative(scale: float):
    """Breakdown detection must not depend on the operator's overall magnitude.

    The fixed ``64 * eps`` tolerance that preceded the relative one compared
    every subdiagonal entry to about ``1.4e-14``.  For an operator of scale
    ``1e-15`` every entry is below that, so the old rule declared breakdown
    after the first column whether or not an invariant subspace existed:
    both cases below trimmed to ``(2, 1)``.  Comparing to ``64 * eps`` times
    the pre-orthogonalization norm ``||A q||`` instead makes the decision
    about the operator's own geometry, so a genuine invariant block trims to
    ``(4, 3)`` and an operator without one runs to the full ``(7, 6)`` at
    every scale.  Both operators are axis-aligned so the outcome does not
    rest on rounding noise from a random similarity.
    """
    blocks = np.zeros((6, 6), dtype=np.complex128)
    blocks[:3, :3] = np.diag([1.0, 2.0, 4.0])
    blocks[3:, 3:] = np.diag([7.0, 9.0, 12.0])
    start = jnp.asarray(np.array([1.0, 0.5, 0.25, 0.0, 0.0, 0.0]), dtype=jnp.complex128)

    blocked_result = arnoldi(_matvec(scale * blocks), start, 6)
    assert blocked_result.hessenberg.shape == (4, 3)
    assert blocked_result.k_achieved == 3
    assert blocked_result.breakdown is True

    coupled = blocks + 0.1 * np.ones((6, 6), dtype=np.complex128)
    coupled_result = arnoldi(_matvec(scale * coupled), start, 6)
    assert coupled_result.hessenberg.shape == (7, 6)
    assert coupled_result.k_achieved == 6
    assert coupled_result.k_requested == 6
