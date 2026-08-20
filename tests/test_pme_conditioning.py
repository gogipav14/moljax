"""Checks for experimental PME conditioning diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from moljax.core.grid import Grid1D
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_conditioning import (
    assess_pme_state,
    build_pme_linearization,
    measure_gmres_iterations,
)
from moljax.experimental.pme_preconditioner import (
    d0_frozen_mean,
    helmholtz_inverse_relative_residual,
)


def _barenblatt_state(grid: NodeCenteredDirichletGrid) -> jax.Array:
    """Return a compactly supported ``m=2`` state for diagnostics."""
    return barenblatt(grid.x_coords(), 0.1, 2.0, b=0.30)


def _smooth_dirichlet_state(grid: NodeCenteredDirichletGrid) -> jax.Array:
    """Return a smooth positive ``m=2`` state with zero boundary nodes."""
    coordinate = (grid.x_coords() - grid.x_min) / (grid.x_max - grid.x_min)
    return jnp.sin(jnp.pi * coordinate)


def _dense_gmres_iterations(matrix: np.ndarray, rhs: np.ndarray, tol: float, max_iters: int) -> int:
    """Independent dense Arnoldi count used only as a test oracle."""
    beta = np.linalg.norm(rhs)
    if beta == 0.0:
        return 0
    basis = [rhs / beta]
    hessenberg = np.zeros((max_iters + 1, max_iters))
    for column in range(max_iters):
        vector = matrix @ basis[column]
        for row, basis_vector in enumerate(basis):
            coefficient = np.vdot(basis_vector, vector)
            hessenberg[row, column] += coefficient
            vector = vector - coefficient * basis_vector
        for row, basis_vector in enumerate(basis):
            correction = np.vdot(basis_vector, vector)
            hessenberg[row, column] += correction
            vector = vector - correction * basis_vector
        hessenberg[column + 1, column] = np.linalg.norm(vector)
        reduced_rhs = np.zeros(column + 2)
        reduced_rhs[0] = beta
        coefficients = np.linalg.lstsq(
            hessenberg[: column + 2, : column + 1], reduced_rhs, rcond=None
        )[0]
        candidate = sum(
            coefficient * basis_vector
            for coefficient, basis_vector in zip(coefficients, basis, strict=True)
        )
        if np.linalg.norm(rhs - matrix @ candidate) / beta <= tol:
            return column + 1
        if hessenberg[column + 1, column] <= np.sqrt(np.finfo(float).eps):
            break
        basis.append(vector / hessenberg[column + 1, column])
    return len(basis)


@pytest.mark.slow
def test_assess_pme_state_has_a_valid_adjoint_gate_and_verdict() -> None:
    """The experimental adapter exposes a valid matrix-free diagnostic state."""
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(64, -4.0, 4.0)
        result = assess_pme_state(
            _barenblatt_state(grid),
            grid,
            2.0,
            0.02,
            1.0e-5,
            "frozen_bulk",
            n_angles=4,
            fov_max_iters=16,
            arnoldi_steps=6,
        )

    assert result["adjoint_error"] <= 1.0e-8
    assert result["verdict"] in {"adequate", "investigate", "indeterminate"}


@pytest.mark.slow
def test_frozen_mean_preconditioning_tightens_the_m2_numerical_range() -> None:
    """The frozen-D0 variant improves the disk-rate diagnostic over identity."""
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(64, -4.0, 4.0)
        state = _smooth_dirichlet_state(grid)
        frozen = assess_pme_state(
            state,
            grid,
            2.0,
            0.1,
            1.0e-5,
            "frozen_mean",
            n_angles=4,
            fov_max_iters=16,
            arnoldi_steps=6,
        )
        identity = assess_pme_state(
            state,
            grid,
            2.0,
            0.1,
            1.0e-5,
            "identity",
            n_angles=4,
            fov_max_iters=16,
            arnoldi_steps=6,
        )

    assert d0_frozen_mean(state, 1.0) == pytest.approx(1.0)
    assert frozen["disk_rate"] < identity["disk_rate"]


@pytest.mark.slow
def test_counted_gmres_matches_an_independent_dense_reference() -> None:
    """The experimental residual-history count agrees with dense GMRES to one step."""
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(24, -4.0, 4.0)
        state = jnp.exp(-grid.x_coords() ** 2)
        linearization = build_pme_linearization(state, grid, 1.0, 0.002, 0.0, "identity")
        basis = jnp.eye(grid.nx, dtype=jnp.float64)
        matrix = np.asarray(
            jnp.column_stack(
                [jnp.real(linearization.operator.matvec(basis[:, i])) for i in range(grid.nx)]
            )
        )
        rhs = np.asarray(linearization.rhs)
        measured = measure_gmres_iterations(
            state,
            grid,
            1.0,
            0.002,
            0.0,
            "identity",
            tol=1.0e-10,
            max_iters=24,
        )
        expected = _dense_gmres_iterations(matrix, rhs, 1.0e-10, 24)

    assert measured["converged"]
    assert measured["iterations"] <= 24
    assert abs(measured["iterations"] - expected) <= 1


def test_helmholtz_inverse_uses_matching_node_centering() -> None:
    """The legacy cell/DST pairing is material, while the node path is exact."""
    with jax.enable_x64(True):
        cell_grid = Grid1D.uniform(64, -4.0, 4.0)
        node_grid = NodeCenteredDirichletGrid.uniform(64, -4.0, 4.0)
        key = jax.random.PRNGKey(7)
        legacy = helmholtz_inverse_relative_residual(1.0, 0.02, cell_grid, key)
        node = helmholtz_inverse_relative_residual(1.0, 0.02, node_grid, key)

    assert legacy > 1.0e-3
    assert node < 1.0e-11


@pytest.mark.slow
def test_helmholtz_variants_reduce_real_gmres_work_for_linear_control() -> None:
    """A matching frozen coefficient reduces actual iterations on the linear control."""
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(24, -4.0, 4.0)
        state = jnp.exp(-grid.x_coords() ** 2)
        identity = measure_gmres_iterations(
            state, grid, 1.0, 0.02, 0.0, "identity", tol=1.0e-8, max_iters=24
        )
        frozen_mean = measure_gmres_iterations(
            state, grid, 1.0, 0.02, 0.0, "frozen_mean", tol=1.0e-8, max_iters=24
        )
        frozen_bulk = measure_gmres_iterations(
            state, grid, 1.0, 0.02, 0.0, "frozen_bulk", tol=1.0e-8, max_iters=24
        )

    assert identity["converged"]
    assert frozen_mean["converged"]
    assert frozen_bulk["converged"]
    assert identity["iterations"] >= frozen_mean["iterations"]
    assert identity["iterations"] >= frozen_bulk["iterations"]
