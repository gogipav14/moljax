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
from moljax.conditioning.field_of_values import _CP_PREFACTOR


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
    assert positive_result.cp_prefactor == pytest.approx(_CP_PREFACTOR)


@pytest.mark.slow
def test_numerical_range_scale_invariance():
    """Flags, residuals and geometry are equivariant under rescaling the operator.

    The certificate gates used to floor their scale at 1.0, so an operator of
    magnitude 1e-8 passed the residual gate and the restart corroboration with
    the same unconverged vectors that failed both gates at magnitude 1.  Every
    flag and every relative quantity below has to be identical across sixteen
    decades, and the boundary has to scale exactly.
    """
    m = 24
    center = 0.9 * np.exp(-1j * np.pi / 4)
    diagonal = center + np.exp(2j * np.pi * np.arange(m) / m)
    scales = [1.0e-8, 1.0e-4, 1.0, 1.0e4, 1.0e8]
    for max_iters, expected_resolved in ((1, False), (120, True)):
        results = []
        for scale in scales:
            matvec, matvec_adjoint = _matrix_actions(np.diag(scale * diagonal))
            results.append(
                numerical_range(
                    matvec,
                    matvec_adjoint,
                    m,
                    n_angles=8,
                    max_iters=max_iters,
                    n_restarts=2,
                )
            )
        reference = results[scales.index(1.0)]
        assert reference.supports_converged is expected_resolved
        assert reference.supports_corroborated is expected_resolved
        for scale, result in zip(scales, results, strict=True):
            assert result.supports_converged is expected_resolved
            assert result.supports_corroborated is expected_resolved
            if expected_resolved:
                # The unresolved sweep separates the origin on an unconverged
                # support; whether that separation survives the hull check is
                # a geometry question with its own scale test.
                assert result.origin_enclosed is reference.origin_enclosed
            # At this residual level (order 1e-10 for the resolved sweep) the
            # remaining variation across sixteen decades of scale is floating-
            # point rounding noise, not a defect: matching numerics reference
            # item 1's own measurement of "2.7e-10 at every scale", a generous
            # relative tolerance still catches the original defect, which
            # floored the residual at a fixed scale and disagreed by many
            # orders of magnitude rather than by rounding.
            assert result.max_support_residual == pytest.approx(
                reference.max_support_residual, rel=0.5
            )
            assert result.disk_rate == pytest.approx(reference.disk_rate, rel=1.0e-9)
            np.testing.assert_allclose(
                np.asarray(result.boundary) / scale,
                np.asarray(reference.boundary),
                rtol=1.0e-9,
                atol=0.0,
            )


def test_default_operator_key_ties_the_seed_to_the_operator():
    """Two different operators must not share the same LOBPCG starting seed.

    The seed used to depend only on the sweep angle and restart index, both
    fixed regardless of which operator was being diagnosed, so an operator
    whose dominant eigenspace happened to be orthogonal to those fixed
    columns at every angle and restart a call used was an exact blind spot
    for every user of ``numerical_range``, not just an unlucky one.  Hashing
    the sign pattern of a probe vector's image under the operator ties the
    seed to the operator, so two operators whose probe images disagree in
    sign somewhere get different seeds, while the same operator always gets
    the same one (the sign pattern, unlike the raw floating-point values, is
    also invariant to rescaling the operator by a positive real factor).
    """
    from moljax.conditioning.field_of_values import _default_operator_key

    # Same magnitudes, opposite sign pattern entrywise, so their probe
    # images (equal to the diagonal itself, since the probe is all ones)
    # cannot share a sign pattern.
    diagonal_a = jnp.asarray([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0], dtype=jnp.complex128)
    diagonal_b = jnp.asarray([1.0, 1.0, 1.0, -1.0, -1.0, -1.0], dtype=jnp.complex128)
    key_a = _default_operator_key(lambda v: diagonal_a * v, 6)
    key_b = _default_operator_key(lambda v: diagonal_b * v, 6)
    key_a_again = _default_operator_key(lambda v: diagonal_a * v, 6)

    assert np.asarray(key_a).tolist() == np.asarray(key_a_again).tolist()
    assert np.asarray(key_a).tolist() != np.asarray(key_b).tolist()


def test_numerical_range_accepts_an_explicit_operator_key():
    """A caller-supplied ``operator_key`` overrides the default and stays reproducible."""
    matrix = _grcar(6)
    matvec, matvec_adjoint = _matrix_actions(matrix)

    default_a = numerical_range(matvec, matvec_adjoint, 6, n_angles=8, max_iters=60)
    default_b = numerical_range(matvec, matvec_adjoint, 6, n_angles=8, max_iters=60)
    np.testing.assert_allclose(
        np.asarray(default_a.boundary), np.asarray(default_b.boundary), rtol=0.0, atol=0.0
    )

    explicit = numerical_range(
        matvec,
        matvec_adjoint,
        6,
        n_angles=8,
        max_iters=60,
        operator_key=jax.random.PRNGKey(0),
    )
    explicit_again = numerical_range(
        matvec,
        matvec_adjoint,
        6,
        n_angles=8,
        max_iters=60,
        operator_key=jax.random.PRNGKey(0),
    )
    np.testing.assert_allclose(
        np.asarray(explicit.boundary), np.asarray(explicit_again.boundary), rtol=0.0, atol=0.0
    )
    # A converged, well-resolved boundary does not depend on which valid
    # starting block found it: the default and the explicit key should agree
    # on the (well-resolved) supports even though their seeds differ.
    np.testing.assert_allclose(
        np.asarray(default_a.boundary), np.asarray(explicit.boundary), atol=1.0e-8, rtol=0.0
    )


def _construct_orthogonal_blind_spot_operator(
    n: int, n_angles: int, n_restarts: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the operator family from Codex conditioning.md finding 1.

    Returns ``(A, u, v)`` where ``A = I + 4 u v^T + 1e-7 P D P``,
    ``D = diag(linspace(-1, 1, n))``, ``P = I - u u^T - v v^T``, and ``u``,
    ``v`` are real orthonormal vectors constructed to be orthogonal, in the
    realified representation, to every LOBPCG starting column the *old*
    (pre-fix) seed formula -- deterministic in ``theta`` and ``restart``
    only -- would ever have produced for the given ``n_angles``/``n_restarts``
    sweep.  ``span(u, v)`` is an exact invariant subspace of both ``A`` and
    its adjoint (``A u = u``, ``A v = 4 u + v``), on which ``A`` restricts to
    ``[[1, 4], [0, 1]]``: a numerical range of the disk centered at 1 with
    radius 2, which contains the origin, while the complementary block is a
    tiny (``1e-7``-scale) perturbation of the identity.  A starting block
    confined to the orthogonal complement of ``span(u, v)`` never leaves it
    under the rotated-Hermitian action (the complement is invariant too), so
    a fixed seed exactly orthogonal to ``u`` and ``v`` is a genuine blind
    spot for the mechanism this operator targets, independent of how well
    the resulting (wrong) answer happens to converge in any one trial.
    """
    block_width = 3
    real_dimension = 2 * n
    padded_dimension = max(real_dimension, 5 * block_width + 1)
    assert padded_dimension == real_dimension, "test only covers the unpadded regime"

    def old_columns(theta: float, restart: int) -> list[np.ndarray]:
        initial_complex = jnp.sin(jnp.arange(n, dtype=jnp.float64) + theta + 1.0) + 1j * jnp.cos(
            jnp.arange(n, dtype=jnp.float64) + 0.5 * theta + 0.5
        )
        initial_real = jnp.concatenate((jnp.real(initial_complex), jnp.imag(initial_complex)))
        probe_key = jax.random.PRNGKey((int(theta * 1_000_003) + 7_919 * restart) & 0x7FFFFFFF)
        random_column = jax.random.normal(probe_key, (padded_dimension,), dtype=jnp.float64)
        if restart == 0:
            columns = [initial_real, jnp.roll(initial_real, 1), random_column]
        else:
            extra = jax.random.normal(
                jax.random.fold_in(probe_key, 1), (padded_dimension, 2), dtype=jnp.float64
            )
            columns = [random_column, extra[:, 0], extra[:, 1]]
        return [np.asarray(column) for column in columns]

    vectors = []
    for index in range(n_angles):
        theta = 2.0 * math.pi * index / n_angles
        for restart in range(n_restarts):
            vectors.extend(old_columns(theta, restart))

    # Embedding u, v with zero imaginary part reduces "orthogonal to every
    # 2n-dim starting column" to "orthogonal to every column's first n
    # (real-part) entries", so the null space is computed on that smaller
    # n x n_columns matrix.
    real_parts = np.stack([vector[:n] for vector in vectors])
    _, _, vt = np.linalg.svd(real_parts, full_matrices=True)
    u = vt[len(vectors)] / np.linalg.norm(vt[len(vectors)])
    v = vt[len(vectors) + 1] / np.linalg.norm(vt[len(vectors) + 1])
    for vector in vectors:
        assert abs(np.dot(vector[:n], u)) < 1.0e-8
        assert abs(np.dot(vector[:n], v)) < 1.0e-8
    assert abs(np.dot(u, v)) < 1.0e-8

    diagonal = np.diag(np.linspace(-1.0, 1.0, n))
    projector = np.eye(n) - np.outer(u, u) - np.outer(v, v)
    matrix = np.eye(n) + 4.0 * np.outer(u, v) + 1.0e-7 * (projector @ diagonal @ projector)
    return matrix, u, v


@pytest.mark.slow
def test_orthogonal_blind_spot_construction_no_longer_hides_the_enclosed_origin():
    """The construction that motivated the operator-dependent seed is resolved.

    This is the 64x64 construction described in Codex conditioning.md
    finding 1: an operator whose true numerical range is the disk centered
    at 1 with radius 2 (which contains the origin), built so that ``u`` and
    ``v`` -- spanning the only part of the operator that reaches beyond a
    tiny neighborhood of 1 -- are orthogonal to every column the old,
    operator-independent seed formula would have used for this exact
    ``n_angles=3``, ``n_restarts=2``, ``max_iters=1`` sweep (matching the
    reproduction parameters in the audit note).

    Note for future maintenance: the audit's own verification could not
    turn this mechanism into a stable false "adequate" through
    ``numerical_range``'s full sweep either (it reported the residual gate
    tripping in 35+ trials, falling to "indeterminate" rather than a false
    "adequate"), and this test does not reliably distinguish the pre-fix
    from the post-fix seed formula on its own -- both can stumble onto the
    correct answer once a second restart's fixed seed happens not to be
    blind for this particular construction.  It is kept as a direct
    regression guard on the exact construction the audit describes, and as
    the scenario the softened ``adequate`` docstring exists for; the seed
    tests above are what actually exercise the operator-dependent fix.
    """
    n = 64
    matrix, u, v = _construct_orthogonal_blind_spot_operator(n, n_angles=3, n_restarts=2)
    operator = jnp.asarray(matrix, dtype=jnp.complex128)
    matvec = lambda value: operator @ value  # noqa: E731
    matvec_adjoint = lambda value: operator.conj().T @ value  # noqa: E731

    result = numerical_range(matvec, matvec_adjoint, n, n_angles=3, max_iters=1, n_restarts=2)

    # The true range contains the origin (it is the disk of radius 2 about
    # 1); the softened contract this test protects is that the diagnostic
    # must never report the origin as excluded when it in fact is not.
    assert result.origin_enclosed or not result.supports_consistent
