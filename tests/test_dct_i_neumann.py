"""
Tests for the node-centred Neumann path (DCT-I).

Covers the properties that must hold jointly for the transform, the
endpoint stencil, the normalization and the eigenvalues to be a
consistent set:

  1. parity with scipy.fft.dct(type=1)
  2. exact diagonalization of the node-centred stencil ([-2, 2] end rows)
  3. constant-mode preservation (the lambda_0 = 0 null mode)
  4. inverse normalization (round trip)
  5. multidimensional (separable 2D) composition
  6. JIT compatibility
  7. manufactured convergence at second order in dx
  8. node vs cell layouts stay distinct, and 'cell' reproduces the
     pre-v1.1.0 behaviour

See moljax/core/fft_nonperiodic.py for the layout conventions.
"""

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import scipy.fft as sfft

from moljax.core.fft_nonperiodic import (
    BCType,
    create_nonperiodic_fft_cache,
    dct_I,
    dct_I_2d,
    etd1_neumann,
    etd1_neumann_node,
    idct_I,
    idct_I_2d,
    laplacian_symbol_neumann,
    laplacian_symbol_neumann_cell,
    laplacian_symbol_neumann_node,
    solve_helmholtz_neumann,
    solve_helmholtz_neumann_cell,
    solve_helmholtz_neumann_node,
)


def dense_neumann_node_laplacian(N: int, dx: float) -> np.ndarray:
    """Second-difference operator with node-centred Neumann end rows."""
    A = np.zeros((N, N))
    for i in range(1, N - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = 1.0, -2.0, 1.0
    A[0, 0], A[0, 1] = -2.0, 2.0
    A[N - 1, N - 2], A[N - 1, N - 1] = 2.0, -2.0
    return A / (dx * dx)


class TestTransformParity:
    """DCT-I must agree with the reference implementation."""

    @pytest.mark.parametrize("N", [2, 3, 8, 33, 64])
    def test_matches_scipy_type_1(self, N):
        rng = np.random.default_rng(N)
        x = rng.standard_normal(N)
        got = np.asarray(dct_I(jnp.asarray(x)))
        want = sfft.dct(x, type=1)
        assert np.allclose(got, want, atol=1e-12)

    def test_rejects_degenerate_length(self):
        with pytest.raises(ValueError, match="at least 2 points"):
            dct_I(jnp.ones(1))


class TestInverseNormalization:
    """idct_I must invert dct_I exactly, not merely up to a constant."""

    @pytest.mark.parametrize("N", [2, 8, 17, 65])
    def test_round_trip(self, N):
        rng = np.random.default_rng(100 + N)
        x = jnp.asarray(rng.standard_normal(N))
        assert np.allclose(np.asarray(idct_I(dct_I(x))), np.asarray(x), atol=1e-12)

    @pytest.mark.parametrize("N", [4, 16])
    def test_round_trip_other_order(self, N):
        rng = np.random.default_rng(200 + N)
        x = jnp.asarray(rng.standard_normal(N))
        assert np.allclose(np.asarray(dct_I(idct_I(x))), np.asarray(x), atol=1e-12)


class TestStencilDiagonalization:
    """The DCT-I basis must be an exact eigenbasis of the node stencil."""

    @pytest.mark.parametrize("N", [5, 16, 65])
    def test_basis_vectors_are_eigenvectors(self, N):
        dx = 0.7 / (N - 1)
        A = dense_neumann_node_laplacian(N, dx)
        lam = np.asarray(laplacian_symbol_neumann_node(N, dx))
        n = np.arange(N)
        for k in range(N):
            v = np.cos(np.pi * k * n / (N - 1))
            residual = A @ v - lam[k] * v
            assert np.abs(residual).max() < 1e-11 * max(1.0, abs(lam[k]))

    @pytest.mark.parametrize("N", [5, 16, 65])
    def test_symbol_matches_dense_spectrum(self, N):
        dx = 0.7 / (N - 1)
        A = dense_neumann_node_laplacian(N, dx)
        lam = np.sort(np.asarray(laplacian_symbol_neumann_node(N, dx)))
        eig = np.sort(np.linalg.eigvals(A).real)
        assert np.allclose(lam, eig, atol=1e-8, rtol=1e-8)

    def test_transform_diagonalizes_operator(self, N=32):
        """A u transformed equals lambda times u transformed."""
        dx = 1.0 / (N - 1)
        A = dense_neumann_node_laplacian(N, dx)
        lam = np.asarray(laplacian_symbol_neumann_node(N, dx))
        rng = np.random.default_rng(7)
        u = rng.standard_normal(N)
        lhs = np.asarray(dct_I(jnp.asarray(A @ u)))
        rhs = lam * np.asarray(dct_I(jnp.asarray(u)))
        assert np.allclose(lhs, rhs, atol=1e-9, rtol=1e-9)


class TestConstantMode:
    """lambda_0 = 0, so constants live in the null space and must survive."""

    @pytest.mark.parametrize("N", [4, 16, 33])
    def test_zero_eigenvalue(self, N):
        lam = laplacian_symbol_neumann_node(N, 0.1)
        assert abs(float(lam[0])) < 1e-14

    def test_constant_preserved_by_etd1(self, N=32):
        dx = 1.0 / (N - 1)
        lam = laplacian_symbol_neumann_node(N, dx)
        u = jnp.ones(N) * 2.5
        for _ in range(20):
            u = etd1_neumann_node(u, jnp.zeros_like(u), 0.1 * lam, dt=0.01)
        assert np.allclose(np.asarray(u), 2.5, atol=1e-10)

    def test_constant_preserved_by_helmholtz(self, N=32):
        dx = 1.0 / (N - 1)
        lam = laplacian_symbol_neumann_node(N, dx)
        u = jnp.ones(N) * -1.25
        out = solve_helmholtz_neumann_node(u, lam, dt=0.5, D=1.0)
        assert np.allclose(np.asarray(out), -1.25, atol=1e-10)

    def test_dense_operator_annihilates_constant(self, N=16):
        dx = 1.0 / (N - 1)
        A = dense_neumann_node_laplacian(N, dx)
        assert np.abs(A @ np.ones(N)).max() < 1e-10


class TestMultidimensional:
    """Separable composition along both axes."""

    @pytest.mark.parametrize("shape", [(8, 8), (17, 9), (5, 32)])
    def test_2d_round_trip(self, shape):
        rng = np.random.default_rng(hash(shape) % 2**31)
        X = jnp.asarray(rng.standard_normal(shape))
        assert np.allclose(np.asarray(idct_I_2d(dct_I_2d(X))), np.asarray(X), atol=1e-11)

    def test_2d_matches_scipy(self, shape=(9, 12)):
        rng = np.random.default_rng(3)
        X = rng.standard_normal(shape)
        got = np.asarray(dct_I_2d(jnp.asarray(X)))
        want = sfft.dct(sfft.dct(X, type=1, axis=-1), type=1, axis=-2)
        assert np.allclose(got, want, atol=1e-10)

    def test_2d_separable_eigenvalues(self, N=16):
        """2D Neumann Laplacian symbol is the outer sum of 1D symbols."""
        dx = 1.0 / (N - 1)
        lam = np.asarray(laplacian_symbol_neumann_node(N, dx))
        lam2d = lam[:, None] + lam[None, :]
        A = dense_neumann_node_laplacian(N, dx)
        rng = np.random.default_rng(11)
        U = rng.standard_normal((N, N))
        LU = A @ U + U @ A.T  # separable 2D Laplacian
        lhs = np.asarray(dct_I_2d(jnp.asarray(LU)))
        rhs = lam2d * np.asarray(dct_I_2d(jnp.asarray(U)))
        assert np.allclose(lhs, rhs, atol=1e-8, rtol=1e-8)


class TestJIT:
    """Everything on the node path must trace and compile."""

    def test_transform_jits(self, N=32):
        f = jax.jit(lambda x: idct_I(dct_I(x)))
        rng = np.random.default_rng(5)
        x = jnp.asarray(rng.standard_normal(N))
        assert np.allclose(np.asarray(f(x)), np.asarray(x), atol=1e-11)

    def test_helmholtz_jits(self, N=32):
        dx = 1.0 / (N - 1)
        lam = laplacian_symbol_neumann_node(N, dx)
        rhs = jnp.asarray(np.random.default_rng(6).standard_normal(N))
        out = jax.jit(solve_helmholtz_neumann_node)(rhs, lam, 0.01, 0.1)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_etd1_jits(self, N=32):
        dx = 1.0 / (N - 1)
        lam = laplacian_symbol_neumann_node(N, dx)
        u = jnp.asarray(np.random.default_rng(8).standard_normal(N))
        out = jax.jit(etd1_neumann_node)(u, jnp.zeros_like(u), 0.1 * lam, 0.01)
        assert np.all(np.isfinite(np.asarray(out)))

    def test_grad_flows(self, N=16):
        dx = 1.0 / (N - 1)
        lam = laplacian_symbol_neumann_node(N, dx)

        def loss(u):
            return jnp.sum(solve_helmholtz_neumann_node(u, lam, 0.01, 0.1) ** 2)

        g = jax.grad(loss)(jnp.ones(N))
        assert np.all(np.isfinite(np.asarray(g)))


class TestManufacturedConvergence:
    """
    Heat equation on [0, L] with homogeneous Neumann BCs.

    u(x, 0) = cos(pi x / L)  =>  u(x, t) = exp(-D pi^2 t / L^2) cos(pi x / L)

    ETD1 with a zero nonlinear term is exact in time for this linear
    problem, so the only error is the spatial eigenvalue error, which is
    O(dx^2).
    """

    L = 1.0
    D = 0.1
    T = 0.05

    def _error(self, N):
        dx = self.L / (N - 1)
        x = np.linspace(0.0, self.L, N)
        u0 = np.cos(np.pi * x / self.L)
        lam = laplacian_symbol_neumann_node(N, dx)
        u = etd1_neumann_node(
            jnp.asarray(u0), jnp.zeros(N), self.D * lam, dt=self.T
        )
        exact = np.exp(-self.D * np.pi**2 * self.T / self.L**2) * u0
        return np.abs(np.asarray(u) - exact).max()

    def test_second_order_in_space(self):
        Ns = [17, 33, 65, 129]
        errs = [self._error(N) for N in Ns]
        # errors must decrease monotonically
        assert all(errs[i + 1] < errs[i] for i in range(len(errs) - 1)), errs
        # fitted order on the last refinement pair
        order = np.log2(errs[-2] / errs[-1])
        assert 1.8 < order < 2.2, f"observed order {order:.3f}, errors {errs}"

    def test_absolute_accuracy(self):
        assert self._error(129) < 1e-5


class TestLayoutsStayDistinct:
    """Node and cell layouts must not be silently interchangeable."""

    def test_symbols_differ(self, N=16):
        node = np.asarray(laplacian_symbol_neumann_node(N, 0.1))
        cell = np.asarray(laplacian_symbol_neumann_cell(N, 0.1))
        assert not np.allclose(node, cell)

    def test_default_is_node(self, N=16):
        assert np.allclose(
            np.asarray(laplacian_symbol_neumann(N, 0.1)),
            np.asarray(laplacian_symbol_neumann_node(N, 0.1)),
        )

    def test_cell_reproduces_previous_behaviour(self, N=16):
        """centering='cell' must give exactly the pre-v1.1.0 symbol."""
        k = np.arange(N)
        legacy = -4.0 / (0.1 * 0.1) * np.sin(np.pi * k / (2 * N)) ** 2
        assert np.allclose(
            np.asarray(laplacian_symbol_neumann(N, 0.1, centering='cell')), legacy
        )

    def test_dispatchers_route_correctly(self, N=16):
        dx = 1.0 / (N - 1)
        lam = laplacian_symbol_neumann_node(N, dx)
        rhs = jnp.asarray(np.random.default_rng(9).standard_normal(N))
        assert np.allclose(
            np.asarray(solve_helmholtz_neumann(rhs, lam, 0.01, 0.1)),
            np.asarray(solve_helmholtz_neumann_node(rhs, lam, 0.01, 0.1)),
        )
        lam_c = laplacian_symbol_neumann_cell(N, dx)
        assert np.allclose(
            np.asarray(solve_helmholtz_neumann(rhs, lam_c, 0.01, 0.1, centering='cell')),
            np.asarray(solve_helmholtz_neumann_cell(rhs, lam_c, 0.01, 0.1)),
        )

    def test_bad_centering_rejected(self, N=8):
        lam = laplacian_symbol_neumann_node(N, 0.1)
        with pytest.raises(ValueError, match="centering"):
            solve_helmholtz_neumann(jnp.ones(N), lam, 0.01, 0.1, centering='nodes')
        with pytest.raises(ValueError, match="centering"):
            laplacian_symbol_neumann(N, 0.1, centering='middle')
        with pytest.raises(ValueError, match="centering"):
            etd1_neumann(jnp.ones(N), jnp.zeros(N), lam, 0.01, centering='x')


class TestCache:
    """BCType must select the matching symbol."""

    def test_neumann_is_node_centred(self, N=32):
        cache = create_nonperiodic_fft_cache(N, 0.1, BCType.NEUMANN)
        assert cache.bc_type == 'neumann'
        assert np.allclose(
            np.asarray(cache.laplacian_symbol),
            np.asarray(laplacian_symbol_neumann_node(N, 0.1)),
        )

    def test_neumann_cell_selectable(self, N=32):
        cache = create_nonperiodic_fft_cache(N, 0.1, BCType.NEUMANN_CELL)
        assert cache.bc_type == 'neumann_cell'
        assert np.allclose(
            np.asarray(cache.laplacian_symbol),
            np.asarray(laplacian_symbol_neumann_cell(N, 0.1)),
        )

    def test_string_construction(self, N=16):
        assert create_nonperiodic_fft_cache(N, 0.1, 'neumann_cell').bc_type == 'neumann_cell'
        assert create_nonperiodic_fft_cache(N, 0.1, 'neumann').bc_type == 'neumann'
