"""Small checks of the dense tau-blend mask analysis (N = 32 and 64).

The expected values come from references that are independent of the code
under test: the block algebra of the masked model, the repository DST blend,
a separate GMRES implementation in this file, and the closed-form floor CFL
number.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from benchmarks.tau_blend_mask_analysis import (
    MaskAnalysisConfig,
    blend_matrix,
    block_norms,
    build_cases,
    build_state,
    cfl_mask,
    floor_cfl,
    gaussian_log_weights,
    gmres,
    gmres_two_conventions,
    jacobian,
    one_inverse_matrix,
    quantile_references,
    repository_blend_check,
)
from moljax.experimental.node_centered import NodeCenteredDirichletGrid

CONFIG = MaskAnalysisConfig()


def _state(nx: int, m: int) -> dict:
    return build_state(nx, m, CONFIG.dt, CONFIG)


def _reference_gmres_counts(M, J, b, tol, maxit):
    """Return the two GMRES counts from a separate Arnoldi and least-squares solve."""
    A = M @ J
    rhs = M @ b
    n = A.shape[0]
    V = np.zeros((n, maxit + 1))
    H = np.zeros((maxit + 1, maxit))
    V[:, 0] = rhs / np.linalg.norm(rhs)
    k_prec = k_true = None
    for k in range(maxit):
        w = A @ V[:, k]
        for _ in range(2):  # classical Gram-Schmidt, two passes
            h = V[:, : k + 1].T @ w
            w = w - V[:, : k + 1] @ h
            H[: k + 1, k] += h
        H[k + 1, k] = np.linalg.norm(w)
        V[:, k + 1] = w / H[k + 1, k]
        e1 = np.zeros(k + 2)
        e1[0] = np.linalg.norm(rhs)
        y = np.linalg.lstsq(H[: k + 2, : k + 1], e1, rcond=None)[0]
        x = V[:, : k + 1] @ y
        if k_prec is None and np.linalg.norm(rhs - A @ x) <= tol * np.linalg.norm(rhs):
            k_prec = k + 1
        if k_true is None and np.linalg.norm(b - J @ x) <= tol * np.linalg.norm(b):
            k_true = k + 1
        if k_prec is not None and k_true is not None:
            break
    return k_prec, k_true


@pytest.mark.parametrize("m", [4, 8])
def test_mask_gives_block_triangular_form_when_masked_coefficient_is_zero(m):
    """With D = 0 on the masked nodes, F = 0, E = 0 and n0 eigenvalues are 1."""
    state = _state(32, m)
    D = state["D"]
    masked = cfl_mask(state, D, CONFIG.eta)
    n0 = int(np.count_nonzero(masked))
    assert 0 < n0 < state["nx"]
    J0 = jacobian(state["L"], np.where(masked, 0.0, D), state["dt"])
    refs = quantile_references(D, masked, 3)
    preconditioners = {
        "one_inverse": one_inverse_matrix(state, D, masked),
        "blend": blend_matrix(state, masked, refs, gaussian_log_weights(D, refs, ~masked)),
    }
    for M in preconditioners.values():
        A = M @ J0
        norms = block_norms(A, masked)
        assert norms["block_F_norm"] < 1.0e-12
        assert norms["block_E_norm"] < 1.0e-12
        assert norms["block_X_norm"] > 1.0e-3
        eigenvalues = np.linalg.eigvals(A)
        assert np.count_nonzero(np.abs(eigenvalues - 1.0) < 1.0e-8) >= n0


@pytest.mark.parametrize(("nx", "m"), [(32, 2), (64, 4), (64, 8)])
def test_dense_blend_matches_repository_blend(nx, m):
    """The dense repository-style blend acts as the repository DST blend."""
    state = _state(nx, m)
    cases = {case["case"]: case for case in build_cases(state, CONFIG)}
    for count in (2, 3, 5, 9):
        check = repository_blend_check(state, cases[f"repo_blend_B{count}"], count, seed=count)
        assert check["action_relative_error"] < 1.0e-12
        assert check["reference_max_relative_difference"] < 1.0e-14
        n0 = int(np.count_nonzero(cases[f"repo_blend_B{count}"]["identity_inputs"]))
        assert check["inactive_count_repository"] == n0


def test_gmres_counts_agree_with_a_separate_implementation():
    """Both conventions give the same counts as a separate GMRES implementation."""
    state = _state(32, 4)
    cases = {case["case"]: case for case in build_cases(state, CONFIG)}
    for name in ("scalar_max", "cfl_mask_one_inverse_max", "repo_blend_B3", "cfl_mask_frame"):
        M = cases[name]["M"]
        out = gmres_two_conventions(M, state["J"], state["b"], 1.0e-8, 64)
        expected = _reference_gmres_counts(M, state["J"], state["b"], 1.0e-8, 64)
        assert (out["gmres_preconditioned"], out["gmres_true"]) == expected
        assert out["final_preconditioned_ratio"] <= 1.0e-8
        assert out["final_true_ratio"] <= 1.0e-8


def test_gmres_conventions_satisfy_the_documented_relation():
    """Check the relation of the two residual conventions.

    For left preconditioning, ||r|| / ||b|| <= kappa_2(M) ||M r|| / ||M b||.
    The conventions agree for M = I, do not change when M is scaled, and give
    one iteration for M = J^-1.
    """
    state = _state(32, 4)
    J, b = state["J"], state["b"]
    M = {case["case"]: case for case in build_cases(state, CONFIG)}["scalar_max"]["M"]
    levels = (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-8)
    out = gmres(M @ J, M @ b, 1.0e-8, 64, secondary=(J, b), stop="both", checkpoints=levels)
    singular = np.linalg.svd(M, compute_uv=False)
    kappa_m = singular[0] / singular[-1]
    assert len(out["checkpoints"]) == len(levels)
    for point in out["checkpoints"].values():
        assert point["secondary_ratio"] <= kappa_m * point["primary_ratio"] * (1.0 + 1.0e-10)

    plain = gmres_two_conventions(np.eye(state["nx"]), J, b, 1.0e-8, 64)
    assert plain["gmres_preconditioned"] == plain["gmres_true"]

    base = gmres_two_conventions(M, J, b, 1.0e-8, 64)
    scaled = gmres_two_conventions(7.25 * M, J, b, 1.0e-8, 64)
    assert (base["gmres_preconditioned"], base["gmres_true"]) == (
        scaled["gmres_preconditioned"],
        scaled["gmres_true"],
    )

    exact = gmres_two_conventions(np.linalg.inv(J), J, b, 1.0e-8, 64)
    assert (exact["gmres_preconditioned"], exact["gmres_true"]) == (1, 1)


def test_floor_cfl_value_and_empty_mask_at_n2048():
    """At N = 2048, m = 2, dt = 0.02 the floor CFL is 0.10496, so eta = 0.1 masks no node."""
    grid = NodeCenteredDirichletGrid.uniform(2048, -4.0, 4.0)
    h = 8.0 / 2049.0
    assert grid.dx == pytest.approx(h, rel=1.0e-14)
    expected = 4.0 * 0.02 * 2.0 * 1.0e-5 / h**2
    assert expected == pytest.approx(0.104960025, rel=1.0e-8)
    assert floor_cfl(2, 1.0e-5, 0.02, grid.dx) == pytest.approx(expected, rel=1.0e-14)
    floor_state = {"dx": grid.dx, "dt": 0.02}
    D = np.full(2048, 2.0e-5)
    assert not np.any(cfl_mask(floor_state, D, 0.1))
    assert np.all(cfl_mask(floor_state, D, 0.3))
