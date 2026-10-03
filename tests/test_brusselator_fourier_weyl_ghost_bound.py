"""Adversarial validation for the Fourier--Weyl--ghost certificate."""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moljax.conditioning import linearized_operator
from moljax.core.grid import Grid2D
from moljax.core.model import create_brusselator_periodic_fft
from moljax.core.newton_krylov import create_implicit_residual
from moljax.core.preconditioners import PrecondContext, create_fft_preconditioner
from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
    dense_padded_preconditioned_operator,
    fourier_weyl_ghost_lower_bound,
    ghost_structure_lower_bound,
)


def _pattern(n: int, length: float) -> np.ndarray:
    coordinate = np.arange(n, dtype=np.float64) * length / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="xy")
    return np.cos(4.0 * np.pi * x / length) * np.cos(10.0 * np.pi * y / length)


def _synthetic_state(n: int, alpha: float, length: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    phi = _pattern(n, length)
    return 1.0 + alpha * phi, 1.8 - 0.35 * alpha * phi


def _dense_sigma(
    u: np.ndarray, v: np.ndarray, *, du: float, dv: float, beta: float, dt: float
) -> float:
    matrix = dense_padded_preconditioned_operator(u, v, du=du, dv=dv, beta=beta, dt=dt)
    return float(np.linalg.svd(matrix, compute_uv=False)[-1])


def test_ghost_formula_never_exceeds_dense_random_block_structures() -> None:
    rng = np.random.default_rng(20260928)
    margins = []
    for _ in range(30):
        b_matrix = rng.normal(size=(5, 5))
        c_matrix = rng.normal(size=(7, 5))
        b = float(np.linalg.svd(b_matrix, compute_uv=False)[-1])
        c = float(np.linalg.svd(c_matrix, compute_uv=False)[0])
        matrix = np.block([[b_matrix, np.zeros((5, 7))], [c_matrix, np.eye(7)]])
        dense = float(np.linalg.svd(matrix, compute_uv=False)[-1])
        bound = ghost_structure_lower_bound(b, c)
        margins.append(dense - bound)
        assert bound <= dense + 5.0e-13
    assert min(margins) >= -5.0e-13


@pytest.mark.parametrize(("b", "c"), ((0.1, 0.0), (0.1, 0.8), (0.7, 1.2), (2.0, 4.0)))
def test_ghost_formula_equals_scalar_extremal_block(b: float, c: float) -> None:
    matrix = np.array(((b, 0.0), (c, 1.0)))
    dense = float(np.linalg.svd(matrix, compute_uv=False)[-1])
    assert ghost_structure_lower_bound(b, c) == pytest.approx(dense, abs=5.0e-14)


@pytest.mark.parametrize(
    ("alpha", "expected_dense", "expected_bound", "expected_c"),
    ((0.0, 0.6328, 0.5987, 0.827), (0.1, 0.5783, 0.4998, 0.996), (0.5, 0.4071, 0.1868, 1.685)),
)
def test_pavlov_turing_16_table(
    alpha: float, expected_dense: float, expected_bound: float, expected_c: float
) -> None:
    u, v = _synthetic_state(16, alpha)
    result = fourier_weyl_ghost_lower_bound(u, v, du=0.01, dv=0.1, a=1.0, beta=1.8, dt=0.2)
    mean = next(candidate for candidate in result.candidates if candidate.name == "mean")
    dense = _dense_sigma(u, v, du=0.01, dv=0.1, beta=1.8, dt=0.2)
    assert dense == pytest.approx(expected_dense, abs=5.0e-4)
    assert mean.full_lower_bound == pytest.approx(expected_bound, abs=5.0e-4)
    assert mean.ghost_norm_bound == pytest.approx(expected_c, abs=5.0e-4)
    assert mean.full_lower_bound <= dense + 5.0e-13


@pytest.mark.parametrize("alpha", (0.0, 0.1, 0.5))
def test_dense_builder_matches_actual_padded_linearization(alpha: float) -> None:
    n, length, dt = 16, 5.0, 0.2
    du, dv, beta = 0.01, 0.1, 1.8
    u, v = _synthetic_state(n, alpha, length)
    grid = Grid2D.uniform(n, n, 0.0, length, 0.0, length, n_ghost=1)
    model, fft_cache, _ = create_brusselator_periodic_fft(
        grid, Du=du, Dv=dv, a=1.0, b=beta, dtype=jnp.float64
    )
    padded_u = np.zeros((n + 2, n + 2), dtype=np.float64)
    padded_v = np.zeros_like(padded_u)
    padded_u[1:-1, 1:-1] = u
    padded_v[1:-1, 1:-1] = v
    state = model.apply_bcs({"u": jnp.asarray(padded_u), "v": jnp.asarray(padded_v)}, 0.0)
    residual = create_implicit_residual(model, state, dt, dt, method="be")
    preconditioner = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    operator = linearized_operator(
        residual,
        state,
        preconditioner=preconditioner,
        context=PrecondContext(grid=grid, dt=dt, params=model.params),
    )
    basis = jnp.eye(operator.n, dtype=jnp.float64)
    actual = np.column_stack(
        [np.asarray(operator.matvec(basis[:, column])).real for column in range(operator.n)]
    )
    block = dense_padded_preconditioned_operator(u, v, du=du, dv=dv, beta=beta, dt=dt)
    assert float(np.linalg.svd(actual, compute_uv=False)[-1]) == pytest.approx(
        float(np.linalg.svd(block, compute_uv=False)[-1]), abs=5.0e-13
    )


@pytest.mark.parametrize(
    ("n", "alpha", "du", "dv", "a", "beta", "dt"),
    (
        (16, 0.0, 0.01, 0.1, 1.0, 1.8, 0.2),
        (16, 0.1, 0.01, 0.1, 1.0, 1.8, 0.2),
        (16, 0.5, 0.01, 0.1, 1.0, 1.8, 0.2),
        (16, 0.9, 0.01, 0.1, 1.0, 1.8, 0.2),
        (16, 0.1, 0.01, 0.02, 1.0, 3.4, 0.05),
        (16, 0.8, 0.01, 0.02, 1.0, 3.4, 0.05),
        (32, 0.1, 0.01, 0.1, 1.0, 1.8, 0.2),
        (32, 0.5, 0.01, 0.02, 1.0, 3.4, 0.05),
    ),
)
def test_every_k0_candidate_never_exceeds_dense(
    n: int, alpha: float, du: float, dv: float, a: float, beta: float, dt: float
) -> None:
    phi = _pattern(n, 5.0)
    u = a + alpha * phi
    v = beta / a - 0.35 * alpha * phi
    result = fourier_weyl_ghost_lower_bound(u, v, du=du, dv=dv, a=a, beta=beta, dt=dt)
    dense = _dense_sigma(u, v, du=du, dv=dv, beta=beta, dt=dt)
    for candidate in result.candidates:
        assert candidate.full_lower_bound <= dense + 5.0e-12
    assert result.selected.full_lower_bound <= dense + 5.0e-12
