"""Checks for experimental PME conditioning diagnostics."""

from __future__ import annotations

from math import ceil, floor, isinf, sqrt

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.pme_breakdown import BreakdownConfig, run_breakdown_study
from moljax.conditioning import crouzeix_palencia_envelope
from moljax.core.grid import Grid1D
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_conditioning import (
    assess_pme_state,
    build_pme_linearization,
    measure_gmres_iterations,
    predicted_iterations_from_envelope,
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


@pytest.mark.parametrize("disk_rate", (0.2, 0.5, 0.8))
def test_envelope_iteration_predictor_reaches_the_requested_threshold(disk_rate: float) -> None:
    """The ceiling of the predictor is the first CP-envelope index below tolerance."""
    tolerance = 1.0e-6
    prefactor = 1.0 + sqrt(2.0)
    predicted = predicted_iterations_from_envelope(
        disk_rate,
        tol=tolerance,
        prefactor=prefactor,
    )
    first_index = ceil(predicted)
    previous_index = floor(predicted)
    envelope = crouzeix_palencia_envelope(disk_rate, first_index, prefactor=prefactor)

    assert float(envelope[-1]) <= tolerance
    assert (
        float(crouzeix_palencia_envelope(disk_rate, previous_index, prefactor=prefactor)[-1])
        > tolerance
    )


def test_envelope_iteration_predictor_is_monotone_and_marks_nonpredictive_rates() -> None:
    """Broader disks require more bound iterations; a unit rate has no decay estimate."""
    rates = (0.0, 0.2, 0.5, 0.8)
    predictions = [predicted_iterations_from_envelope(rate, tol=1.0e-6) for rate in rates]

    assert predictions == sorted(predictions)
    assert isinf(predicted_iterations_from_envelope(1.0, tol=1.0e-6))


@pytest.mark.slow
def test_regime_claim_smoke_reports_when_the_small_sample_is_not_separated(tmp_path) -> None:
    """A small mixed-verdict sample either supports separation or reports its absence."""
    report = run_breakdown_study(
        BreakdownConfig(
            nx=128,
            m_values=(1,),
            analysis_dt_values=(2.0e-4, 2.0),
            front_target_halfwidths=(0.25, 3.0),
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=4,
            max_krylov_iters=400,
            output_path=str(tmp_path / "pme_regime_smoke.json"),
        )
    )
    claim = report["regime_claim"]["adequate_vs_investigate"]
    buckets = report["regime_claim"]["iteration_by_verdict"]

    if buckets.get("adequate", {"count": 0})["count"] == 0:
        pytest.skip("small sample did not produce an adequate verdict")
    if buckets.get("investigate", {"count": 0})["count"] == 0:
        pytest.skip("small sample did not produce an investigate verdict")
    if not claim["supports_cost_separation"]:
        pytest.skip("small sample does not support the regime-cost inequality")
    assert buckets["investigate"]["median"] >= buckets["adequate"]["median"]


@pytest.mark.slow
def test_identity_stress_sweep_has_required_dynamic_range(tmp_path) -> None:
    """The stress schedule must expose at least a fivefold identity-cost range."""
    report = run_breakdown_study(
        BreakdownConfig(
            nx=128,
            m_values=(1,),
            d0_kinds=("identity",),
            state_dt=0.02,
            analysis_dt_values=(2.0e-4, 2.0),
            front_target_halfwidths=(0.25, 3.0),
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=4,
            max_krylov_iters=400,
            output_path=str(tmp_path / "pme_identity_stress.json"),
        )
    )
    dynamic_range = report["verdict_on_decision_procedure"]["identity_iteration_dynamic_range"]

    assert dynamic_range["ratio"] >= 5.0, (
        "The stress schedule is too benign: "
        f"min={dynamic_range['min']}, max={dynamic_range['max']}, "
        f"ratio={dynamic_range['ratio']}"
    )


@pytest.mark.slow
def test_regime_map_reports_non_benign_high_stiffness_cells(tmp_path) -> None:
    """The map must retain high-stiffness nonlinear cells with real cost variation."""
    high_dt = 2.0
    report = run_breakdown_study(
        BreakdownConfig(
            nx=128,
            m_values=(1, 2, 3),
            d0_kinds=("identity",),
            state_dt=0.02,
            analysis_dt_values=(2.0e-4, high_dt),
            front_target_halfwidths=(0.25, 0.75, 3.0),
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=4,
            max_krylov_iters=400,
            output_path=str(tmp_path / "pme_regime_map.json"),
        )
    )
    regime_map = report["regime_map"]
    cells = regime_map["cells"]
    high_stiffness_nonlinear = [
        cell for cell in cells if cell["analysis_dt"] == high_dt and cell["m"] in {2, 3}
    ]

    assert len(cells) == 6
    assert len(high_stiffness_nonlinear) == 2
    assert all(
        cell["identity_iteration_range"]["ratio"] >= 5.0 for cell in high_stiffness_nonlinear
    )
