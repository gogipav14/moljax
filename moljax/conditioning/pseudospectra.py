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
import jax.numpy as jnp

from moljax._precision import require_x64

Matvec = Callable[[jax.Array], jax.Array]


class ArnoldiResult(NamedTuple):
    """Forward-only Arnoldi factorization together with its coverage evidence.

    ``arnoldi`` used to return a plain ``(Q, H)`` tuple, which threw away
    exactly the information a caller needs to judge whether ``H`` is a
    trustworthy stand-in for the full operator: how many steps were actually
    completed, and whether they ended in a clean breakdown or something is
    wrong upstream.  A reduced ``H`` from an invariant-subspace breakdown can
    have a much larger smallest singular value than the full operator (an
    exact decoupled degree of freedom the start vector never excites is the
    generic example, not an adversarial one), so ``epsilon_zero(H)`` silently
    stops being a lower bound on ``sigma_min`` of the full operator once
    ``k_achieved < n``.  Carrying ``k_achieved`` and ``n`` (via ``basis``'s
    row count) lets :func:`moljax.conditioning.non_normality.assess_preconditioner`
    tell a full-dimensional projection from a reduced one instead of trusting
    every ``epsilon_zero`` equally.

    Attributes:
        basis: Orthonormal Krylov basis ``Q``, shape ``(n, k_achieved + 1)``.
        hessenberg: Rectangular upper-Hessenberg array ``H``, shape
            ``(k_achieved + 1, k_achieved)``.
        k_requested: The ``k`` passed to :func:`arnoldi`.
        k_achieved: The number of completed Arnoldi columns.  Equals
            ``k_requested`` unless a Krylov breakdown occurred, in which case
            it is the dimension at which the factorization was trimmed.
        breakdown: Whether the factorization stopped early because the next
            Krylov vector's norm fell below the scale-relative breakdown
            tolerance, rather than because ``k_requested`` columns were
            completed.
        residual_norm: ``||A Q[:, :k_achieved] - Q H||``, the explicit
            forward-factorization residual.  It is a coverage indicator, not
            a correctness proof: orthogonalization keeps it near machine
            epsilon whenever the arithmetic behaved, so a value that is not
            small flags a problem with ``matvec`` (e.g. non-finite output or
            an operator that is not actually linear) rather than with the
            reduction itself.
    """

    basis: jax.Array
    hessenberg: jax.Array
    k_requested: int
    k_achieved: int
    breakdown: bool
    residual_norm: float


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
) -> ArnoldiResult:
    """Return a forward-only modified-Gram--Schmidt Arnoldi factorization.

    Args:
        matvec: Callable computing ``A @ v`` for one vector ``v``.
        v0: Nonzero initial Krylov vector.
        k: Requested Arnoldi steps, with ``1 <= k <= len(v0)``.
        reorthogonalize: Apply a second modified-Gram--Schmidt pass.

    Returns:
        An :class:`ArnoldiResult`.  Its ``basis`` (``Q``) has shape
        ``(n, k_eff + 1)`` and its ``hessenberg`` (``H``) has shape
        ``(k_eff + 1, k_eff)``, where ``k_eff`` is ``k_achieved``: equal to
        ``k`` unless a Krylov breakdown occurs, in which case the
        factorization is trimmed after the completed column.  The
        rectangular convention is retained, so ``A @ Q[:, :k_eff] == Q @ H``
        also holds at breakdown; ``residual_norm`` reports how well that
        relation actually closed.  Callers that used to write
        ``Q, H = arnoldi(...)`` must switch to ``result.basis`` /
        ``result.hessenberg`` (``result[0]`` / ``result[1]`` also work,
        since ``ArnoldiResult`` is a ``NamedTuple`` with ``basis`` and
        ``hessenberg`` first); the six-field result no longer unpacks as a
        bare pair.

    Raises:
        RuntimeError: If 64-bit precision is not enabled.
    """
    require_x64("conditioning diagnostics")
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
    breakdown_factor = 64.0 * jnp.finfo(jnp.float64).eps
    k_eff = k
    breakdown_occurred = False

    for column in range(k):
        candidate = _complex_action(matvec, basis[:, column])
        # The pre-orthogonalization norm is the scale of this Krylov step:
        # ``A q`` for the current basis vector ``q``.  A fixed absolute
        # tolerance treats an operator far from unit scale as if every
        # column were a breakdown (the norm never clears the threshold) or
        # never one (rounding noise never reaches it); comparing to this
        # scale instead makes the breakdown test track the operator.
        action_norm = jnp.linalg.norm(candidate)
        coefficients = basis[:, : column + 1].conj().T @ candidate
        candidate = candidate - basis[:, : column + 1] @ coefficients
        if reorthogonalize:
            correction = basis[:, : column + 1].conj().T @ candidate
            coefficients = coefficients + correction
            candidate = candidate - basis[:, : column + 1] @ correction
        hessenberg = hessenberg.at[: column + 1, column].set(coefficients)
        norm = jnp.linalg.norm(candidate)
        hessenberg = hessenberg.at[column + 1, column].set(norm)
        if float(norm) <= breakdown_factor * float(action_norm):
            k_eff = column + 1
            breakdown_occurred = True
            break
        basis = basis.at[:, column + 1].set(candidate / norm)

    trimmed_basis = basis[:, : k_eff + 1]
    trimmed_hessenberg = hessenberg[: k_eff + 1, :k_eff]
    return ArnoldiResult(
        basis=trimmed_basis,
        hessenberg=trimmed_hessenberg,
        k_requested=k,
        k_achieved=k_eff,
        breakdown=breakdown_occurred,
        residual_norm=_forward_residual_norm(matvec, trimmed_basis, trimmed_hessenberg, k_eff),
    )


def _forward_residual_norm(
    matvec: Matvec, basis: jax.Array, hessenberg: jax.Array, k_eff: int
) -> float:
    """Return ``||A Q[:, :k_eff] - Q H||`` for a trimmed Arnoldi factorization.

    Recomputed explicitly rather than reused from the reduction loop (whose
    intermediate products are already the orthogonalized remainders, not the
    raw ``A q`` values) so the check exercises the same ``matvec`` the caller
    supplied and catches a non-finite or inconsistent operator as a large
    residual rather than as a silently corrupted basis.
    """
    columns = basis[:, :k_eff]
    action = jax.vmap(lambda column: _complex_action(matvec, column), in_axes=1, out_axes=1)(
        columns
    )
    return float(jnp.linalg.norm(action - basis @ hessenberg))


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

    Raises:
        RuntimeError: If 64-bit precision is not enabled.
    """
    require_x64("conditioning diagnostics")
    projection = _square_hessenberg(hessenberg)
    real = jnp.asarray(real_grid, dtype=jnp.float64)
    imag = jnp.asarray(imag_grid, dtype=jnp.float64)
    if real.ndim != 1 or imag.ndim != 1:
        raise ValueError("real_grid and imag_grid must be one-dimensional")
    if real.size == 0 or imag.size == 0:
        raise ValueError("real_grid and imag_grid must be nonempty")
    return _sigma_min_grid(projection, real, imag)


def epsilon_zero(hessenberg: jax.Array) -> float:
    """Return the continuous epsilon at which zero enters the pseudospectrum.

    Raises:
        RuntimeError: If 64-bit precision is not enabled.
    """
    require_x64("conditioning diagnostics")
    projection = _square_hessenberg(hessenberg)
    return float(jnp.linalg.svd(projection, compute_uv=False)[-1])


def ritz_values(hessenberg: jax.Array) -> jax.Array:
    """Return eigenvalues of the square leading Arnoldi projection.

    Raises:
        RuntimeError: If 64-bit precision is not enabled.
    """
    require_x64("conditioning diagnostics")
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

    Raises:
        RuntimeError: If 64-bit precision is not enabled.
    """
    require_x64("conditioning diagnostics")
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
