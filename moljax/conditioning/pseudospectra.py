"""Forward-only Arnoldi pseudospectra diagnostics for linear operators.

Arnoldi reduces a large operator to a small upper-Hessenberg projection using
only forward matrix-vector products.  Pseudospectral singular values are then
computed densely on that projection, so the large operator is never
materialized during the reduced analysis.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

Matvec = Callable[[jax.Array], jax.Array]


class PseudospectraResult(NamedTuple):
    """Pseudospectral data evaluated on a rectangular complex grid.

    Attributes:
        real_grid: Real-axis coordinates, ordered across columns.
        imag_grid: Imaginary-axis coordinates, ordered across rows.
        sigma_min: ``sigma_min(z I - A)`` on ``imag_grid x real_grid``.
        ritz_values: Eigenvalues of the analysed operator or projection.
        epsilon_zero: Smallest singular value at ``z = 0``.
    """

    real_grid: jax.Array
    imag_grid: jax.Array
    sigma_min: jax.Array
    ritz_values: jax.Array
    epsilon_zero: float


def _complex_action(matvec: Matvec, value: jax.Array) -> jax.Array:
    """Apply a real or complex linear action to a complex vector."""
    real = jnp.asarray(matvec(jnp.real(value)), dtype=jnp.complex128)
    imag = jnp.asarray(matvec(jnp.imag(value)), dtype=jnp.complex128)
    return real + 1j * imag


def _square_hessenberg(hessenberg: jax.Array) -> jax.Array:
    """Return the square Ritz projection from a rectangular Arnoldi array."""
    matrix = jnp.asarray(hessenberg, dtype=jnp.complex128)
    if matrix.ndim != 2:
        raise ValueError("H must be a two-dimensional array")
    rows, columns = matrix.shape
    if columns < 1 or rows < columns:
        raise ValueError("H must have at least as many rows as columns")
    return matrix[:columns, :columns]


def arnoldi(
    matvec: Matvec,
    v0: jax.Array,
    k: int,
    *,
    reorthogonalize: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Return a forward-only modified-Gram--Schmidt Arnoldi factorization.

    Args:
        matvec: Callable computing ``A @ v`` for one vector ``v``.
        v0: Nonzero initial Krylov vector.
        k: Requested Arnoldi steps, with ``1 <= k <= len(v0)``.
        reorthogonalize: Apply a second modified-Gram--Schmidt pass.

    Returns:
        ``(Q, H)`` where ``Q`` has shape ``(n, k_eff + 1)`` and ``H`` has
        shape ``(k_eff + 1, k_eff)``.  ``k_eff`` equals ``k`` unless a Krylov
        breakdown occurs, in which case the factorization is trimmed after
        the completed column.  The rectangular convention is retained, so
        ``A @ Q[:, :k_eff] == Q @ H`` also holds at breakdown.
    """
    if k < 1:
        raise ValueError("k must be positive")

    initial = jnp.asarray(v0, dtype=jnp.complex128)
    if initial.ndim != 1:
        raise ValueError("v0 must be one-dimensional")
    n = initial.size
    if k > n:
        raise ValueError("k must not exceed the operator dimension")
    initial_norm = float(jnp.linalg.norm(initial))
    if initial_norm == 0.0:
        raise ValueError("v0 must be nonzero")

    basis = jnp.zeros((n, k + 1), dtype=jnp.complex128)
    hessenberg = jnp.zeros((k + 1, k), dtype=jnp.complex128)
    basis = basis.at[:, 0].set(initial / initial_norm)
    tolerance = 64.0 * jnp.finfo(jnp.float64).eps
    k_eff = k

    for column in range(k):
        candidate = _complex_action(matvec, basis[:, column])
        coefficients = basis[:, : column + 1].conj().T @ candidate
        candidate = candidate - basis[:, : column + 1] @ coefficients
        if reorthogonalize:
            correction = basis[:, : column + 1].conj().T @ candidate
            coefficients = coefficients + correction
            candidate = candidate - basis[:, : column + 1] @ correction
        hessenberg = hessenberg.at[: column + 1, column].set(coefficients)
        norm = jnp.linalg.norm(candidate)
        hessenberg = hessenberg.at[column + 1, column].set(norm)
        if float(norm) <= tolerance:
            k_eff = column + 1
            break
        basis = basis.at[:, column + 1].set(candidate / norm)

    return basis[:, : k_eff + 1], hessenberg[: k_eff + 1, :k_eff]


def _sigma_min_grid(
    matrix: jax.Array,
    real_grid: jax.Array,
    imag_grid: jax.Array,
) -> jax.Array:
    """Return smallest shifted singular values for one small dense matrix."""
    identity = jnp.eye(matrix.shape[0], dtype=jnp.complex128)
    points = real_grid[None, :] + 1j * imag_grid[:, None]
    shifted = points[..., None, None] * identity - matrix
    return jnp.linalg.svd(shifted, compute_uv=False)[..., -1]


def reduced_pseudospectrum(
    hessenberg: jax.Array,
    real_grid: jax.Array,
    imag_grid: jax.Array,
) -> jax.Array:
    """Evaluate ``sigma_min(z I - H_k)`` for a reduced Arnoldi projection.

    ``hessenberg`` may be the rectangular ``(k + 1, k)`` result of
    :func:`arnoldi`; only its square leading block ``H_k`` is used.  This is
    the Ritz projection whose eigenvalues and shifted singular values define
    the reduced pseudospectrum.
    """
    projection = _square_hessenberg(hessenberg)
    real = jnp.asarray(real_grid, dtype=jnp.float64)
    imag = jnp.asarray(imag_grid, dtype=jnp.float64)
    if real.ndim != 1 or imag.ndim != 1:
        raise ValueError("real_grid and imag_grid must be one-dimensional")
    if real.size == 0 or imag.size == 0:
        raise ValueError("real_grid and imag_grid must be nonempty")
    return _sigma_min_grid(projection, real, imag)


def epsilon_zero(hessenberg: jax.Array) -> float:
    """Return the continuous epsilon at which zero enters the pseudospectrum."""
    projection = _square_hessenberg(hessenberg)
    return float(jnp.linalg.svd(projection, compute_uv=False)[-1])


def ritz_values(hessenberg: jax.Array) -> jax.Array:
    """Return eigenvalues of the square leading Arnoldi projection."""
    return jnp.linalg.eigvals(_square_hessenberg(hessenberg))


def pseudospectrum_dense(
    matvec: Matvec,
    n: int,
    real_grid: jax.Array,
    imag_grid: jax.Array,
) -> PseudospectraResult:
    """Materialize a small operator and evaluate its dense pseudospectrum.

    The operator columns are obtained by applying ``matvec`` to the identity
    vectors.  This validation helper is intended only for small systems and
    figures; use :func:`arnoldi` plus :func:`reduced_pseudospectrum` for large
    matrix-free operators.
    """
    if n < 1:
        raise ValueError("n must be positive")

    real = jnp.asarray(real_grid, dtype=jnp.float64)
    imag = jnp.asarray(imag_grid, dtype=jnp.float64)
    if real.ndim != 1 or imag.ndim != 1:
        raise ValueError("real_grid and imag_grid must be one-dimensional")
    if real.size == 0 or imag.size == 0:
        raise ValueError("real_grid and imag_grid must be nonempty")
    identity = jnp.eye(n, dtype=jnp.complex128)
    columns = jax.vmap(lambda column: _complex_action(matvec, column))(identity.T)
    matrix = columns.T
    return PseudospectraResult(
        real_grid=real,
        imag_grid=imag,
        sigma_min=_sigma_min_grid(matrix, real, imag),
        ritz_values=jnp.linalg.eigvals(matrix),
        epsilon_zero=float(jnp.linalg.svd(matrix, compute_uv=False)[-1]),
    )
