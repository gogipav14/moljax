"""Checks for experimental PME conditioning diagnostics."""

from __future__ import annotations

import importlib
import json
from math import ceil, floor, isinf, sqrt
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

import benchmarks.pme_breakdown as pme_breakdown
from benchmarks.pme_breakdown import BreakdownConfig, run_breakdown_study
from moljax.conditioning import crouzeix_palencia_envelope
from moljax.core.grid import Grid1D
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_conditioning import (
    _counted_gmres,
    assess_pme_state,
    build_pme_linearization,
    measure_gmres_iterations,
    predicted_iterations_from_envelope,
)
from moljax.experimental.pme_preconditioner import (
    d0_floor,
    d0_frozen_mean,
    helmholtz_inverse_relative_residual,
)
from moljax.experimental.porous_fisher_conditioning import (
    build_porous_fisher_linearization,
    measure_porous_fisher_gmres_iterations,
)


def _barenblatt_state(grid: NodeCenteredDirichletGrid) -> jax.Array:
    """Return a compactly supported ``m=2`` state for diagnostics."""
    return barenblatt(grid.x_coords(), 0.1, 2.0, b=0.30)


def _smooth_dirichlet_state(grid: NodeCenteredDirichletGrid) -> jax.Array:
    """Return a smooth positive ``m=2`` state with zero boundary nodes."""
    coordinate = (grid.x_coords() - grid.x_min) / (grid.x_max - grid.x_min)
    return jnp.sin(jnp.pi * coordinate)


def test_unconverged_reference_state_is_not_assessed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed source solve must produce metadata, not a conditioning verdict."""
    grid = NodeCenteredDirichletGrid.uniform(16, -1.0, 1.0)
    config = BreakdownConfig(
        nx=16,
        d0_kinds=("identity",),
        analysis_dt_values=(0.02,),
    )
    state_solver = {
        "status": "source_state_unusable",
        "converged": False,
        "newton_iters": 1,
        "final_residual_l2": 1.0,
        "newton_tolerance": config.newton_tol,
    }

    def forbidden_assessment(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("conditioning assessment must not run")

    monkeypatch.setattr(pme_breakdown, "assess_pme_state", forbidden_assessment)
    records: list[dict[str, object]] = []
    pme_breakdown._record_state(
        records,
        _barenblatt_state(grid),
        grid,
        2,
        1,
        0.25,
        0.02,
        config,
        {"decision": "test"},
        state_solver,
    )

    assert len(records) == 1
    assert records[0]["source_state_status"] == "source_state_unusable"
    assert records[0]["conditioning_assessed"] is False
    assert "verdict" not in records[0]


def test_persisted_source_state_is_reused_and_hash_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escalation must diagnose the saved source array, never a fresh re-solve."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    stage2_regeneration = importlib.import_module("regenerate_stage2_conditioning")
    calls = 0

    def solve_once() -> tuple[jax.Array, dict[str, object]]:
        nonlocal calls
        calls += 1
        state = jnp.asarray((0.125, 0.5, 0.25), dtype=jnp.float64)
        return state, {
            "status": "converged",
            "converged": True,
            "newton_iters": 3,
            "final_residual_l2": 1.0e-12,
            "newton_tolerance": 1.0e-8,
        }

    fingerprint = stage2_regeneration._pme_source_fingerprint(BreakdownConfig(nx=3), 2, 3, 3.0)
    saved_state, saved_solver = stage2_regeneration._load_or_persist_source_state(
        tmp_path, "pme", "m2_front3", fingerprint, solve_once
    )

    def must_not_resolve() -> tuple[jax.Array, dict[str, object]]:
        raise AssertionError("persisted source state was unexpectedly re-solved")

    loaded_state, loaded_solver = stage2_regeneration._load_or_persist_source_state(
        tmp_path, "pme", "m2_front3", fingerprint, must_not_resolve
    )

    assert calls == 1
    assert np.array_equal(np.asarray(saved_state), np.asarray(loaded_state))
    assert saved_solver["source_state_identity"] == loaded_solver["source_state_identity"]
    artifact = loaded_solver["source_state_artifact"]
    assert artifact["source_state_identity"] == loaded_solver["source_state_identity"]

    artifact_path = tmp_path / artifact["relative_path"]
    np.save(artifact_path, np.zeros(3, dtype=np.float64), allow_pickle=False)
    with pytest.raises(RuntimeError, match="artifact hash mismatch"):
        stage2_regeneration._load_saved_source_state(tmp_path, "pme", "m2_front3", fingerprint)


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
    assert result["verdict"] in {"adequate", "investigate", "indeterminate", "provisional"}
    assert result["epsilon_zero_full_operator_evidence"] is False
    assert result["arnoldi_k_achieved"] == 6


@pytest.mark.slow
def test_assess_pme_state_can_attach_full_operator_epsilon_evidence() -> None:
    """The optional recovery path uses the full dense epsilon-zero helper."""
    grid = NodeCenteredDirichletGrid.uniform(32, -4.0, 4.0)
    result = assess_pme_state(
        _barenblatt_state(grid),
        grid,
        2.0,
        0.02,
        1.0e-5,
        "frozen_mean",
        n_angles=3,
        fov_max_iters=4,
        fov_n_restarts=2,
        arnoldi_steps=6,
        full_operator_epsilon_evidence=True,
    )

    assert result["epsilon_zero_full_operator_evidence"] is True
    assert result["full_operator_epsilon_zero"] is not None
    assert result["full_operator_epsilon_zero_seconds"] is not None
    assert result["epsilon_zero"] == pytest.approx(result["full_operator_epsilon_zero"])


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
    state = jnp.exp(-(grid.x_coords() ** 2))
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


def test_frozen_mean_is_sign_safe_and_matches_the_regularization_floor() -> None:
    """A mean-negative state cannot produce a negative or invalid Helmholtz D0."""
    epsilon = 1.0e-5
    state = jnp.full(8, -1.0e-3, dtype=jnp.float64)
    d0 = d0_frozen_mean(state, 2.0, epsilon=epsilon)

    assert np.isfinite(d0)
    assert d0 > 0.0
    assert d0 == pytest.approx(2.0 * np.sqrt(1.0e-6 + epsilon**2))
    assert d0_floor(2.0, epsilon) == pytest.approx(2.0 * epsilon)


@pytest.mark.slow
def test_helmholtz_variants_reduce_real_gmres_work_for_linear_control() -> None:
    """A matching frozen coefficient reduces actual iterations on the linear control."""
    grid = NodeCenteredDirichletGrid.uniform(24, -4.0, 4.0)
    state = jnp.exp(-(grid.x_coords() ** 2))
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
            nx=64,
            m_values=(2,),
            d0_kinds=("identity",),
            analysis_dt_values=(2.0e-2, 2.0),
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
            nx=64,
            m_values=(2, 3),
            d0_kinds=("identity",),
            state_dt=0.02,
            analysis_dt_values=(2.0e-2, high_dt),
            front_target_halfwidths=(0.25, 3.0),
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

    assert len(cells) == 4
    assert len(high_stiffness_nonlinear) == 2
    assert all(
        cell["identity_iteration_range"]["ratio"] >= 5.0 for cell in high_stiffness_nonlinear
    )


def test_singular_pf_linearization_reports_breakdown_not_convergence() -> None:
    """A vanishing rotated pivot must not be read as a zero GMRES residual."""
    grid = NodeCenteredDirichletGrid.uniform(1, -1.0, 1.0)
    state = jnp.asarray((0.25,), dtype=jnp.float64)
    settings = {"r": 4.5, "dt": 1.0, "epsilon": 0.1875, "d0_kind": "identity"}
    linearization = build_porous_fisher_linearization(state, grid, **settings)
    jacobian = jax.jacfwd(linearization.operator.matvec)(jnp.zeros(1, dtype=jnp.float64))

    assert float(jnp.max(jnp.abs(jacobian))) == 0.0
    assert float(jnp.linalg.norm(linearization.rhs)) > 0.5

    stats = measure_porous_fisher_gmres_iterations(
        state, grid, tol=1.0e-10, max_iters=8, **settings
    )

    assert stats["converged"] is False
    assert stats["final_relative_residual"] == pytest.approx(1.0)
    assert stats["breakdown"] is True


def test_nonsingular_gmres_measurement_is_unchanged_by_breakdown_detection() -> None:
    """Breakdown detection must not move a solvable system's count or residual."""
    grid = NodeCenteredDirichletGrid.uniform(16, -1.0, 1.0)
    state = jnp.maximum(1.0 - grid.x_coords() ** 2, 0.0)
    expected = {
        "identity": 1.1181463722680472e-12,
        "frozen_mean": 3.2488832540148357e-19,
    }
    for d0_kind, residual in expected.items():
        stats = measure_gmres_iterations(
            state,
            grid,
            2.0,
            0.1,
            1.0e-3,
            d0_kind,
            tol=1.0e-10,
            max_iters=64,
        )
        assert stats["converged"] is True
        assert stats["iterations"] == 8
        assert stats["final_relative_residual"] == pytest.approx(residual, rel=1.0e-6)
        assert stats["breakdown"] is False


def test_breakdown_cutoff_is_relative_to_the_operator_scale() -> None:
    """A small invertible operator must not be mistaken for a breakdown.

    ``A = [1e-9]``, ``b = [1]`` has condition number one and is solved exactly
    by one iteration, although its rotated pivot is below an absolute
    ``sqrt(eps)``.  The scaled and unscaled systems must be reported
    identically, while an exactly singular operator still breaks down.
    """
    rhs = jnp.asarray((1.0,), dtype=jnp.float64)
    scaled = _counted_gmres(lambda v: 1.0e-9 * v, rhs, tol=1.0e-10, max_iters=8)
    unscaled = _counted_gmres(lambda v: 1.0 * v, rhs, tol=1.0e-10, max_iters=8)

    assert scaled["breakdown"] is False
    assert scaled["converged"] is True
    assert scaled["iterations"] == 1
    assert scaled["final_relative_residual"] <= 1.0e-10
    assert scaled == unscaled

    singular = _counted_gmres(
        lambda v: jnp.zeros_like(v),
        jnp.asarray((0.61025382,), dtype=jnp.float64),
        tol=1.0e-10,
        max_iters=8,
    )

    assert singular["breakdown"] is True
    assert singular["converged"] is False
    assert singular["iterations"] == 1
    assert singular["final_relative_residual"] == pytest.approx(1.0)


def test_invariant_krylov_space_is_decided_by_the_measured_residual() -> None:
    """At an invariant Krylov space only the measured residual decides.

    ``diag(1, 1e-12)`` against ``b = (1, 1)`` spans its Krylov space in two
    columns, so the second Arnoldi subdiagonal vanishes to roundoff and the
    rotated estimate collapses to about ``1e-20``.  The triangular system is
    nonsingular, but its candidate's measured ``||b - A x|| / ||b||`` is
    about ``1.3e-4``, so ``tol = 1e-10`` is a breakdown reported with that
    measured residual and ``tol = 1e-2`` is convergence.  SciPy's GMRES agrees
    that ``tol = 1e-10`` is not reached.
    """
    diagonal = jnp.asarray((1.0, 1.0e-12), dtype=jnp.float64)
    rhs = jnp.asarray((1.0, 1.0), dtype=jnp.float64)
    refused = _counted_gmres(lambda v: diagonal * v, rhs, tol=1.0e-10, max_iters=8)

    assert refused["breakdown"] is True
    assert refused["converged"] is False
    assert refused["iterations"] == 2
    assert 1.0e-10 < refused["final_relative_residual"] < 1.0

    accepted = _counted_gmres(lambda v: diagonal * v, rhs, tol=1.0e-2, max_iters=8)

    assert accepted["breakdown"] is False
    assert accepted["converged"] is True
    assert accepted["iterations"] == 2
    assert accepted["final_relative_residual"] <= 1.0e-2

    reference = _scipy_gmres(np.diag(np.asarray(diagonal)), np.asarray(rhs), tol=1.0e-10)
    assert reference["converged"] is False
    assert reference["measured"] > 1.0e-10


def _scipy_gmres(matrix: np.ndarray, rhs: np.ndarray, *, tol: float) -> dict[str, object]:
    """Run unrestarted SciPy GMRES and return its count and measured residual."""
    from scipy.sparse.linalg import gmres

    history: list[float] = []
    solution, info = gmres(
        matrix,
        rhs,
        rtol=tol,
        atol=0.0,
        restart=2 * rhs.size,
        maxiter=1,
        callback=history.append,
        callback_type="pr_norm",
    )
    measured = float(np.linalg.norm(rhs - matrix @ solution) / np.linalg.norm(rhs))
    return {"converged": info == 0, "iterations": len(history), "measured": measured}


def test_small_rotated_pivot_does_not_truncate_a_solvable_system() -> None:
    """A small rotated pivot with a nonzero subdiagonal is no reason to stop.

    For ``A = diag(1e9, 1, 2)``, ``b = (1, 1, 1)`` the second rotated pivot
    is about ``3e-9`` times ``||A v_2||`` and far below ``sqrt(eps)`` times
    the largest Hessenberg entry, but the Arnoldi subdiagonal is not small
    relative to that column, so the Krylov space still grows.  The third
    column spans the space; convergence there is decided by the measured
    residual, so ``converged`` already certifies ``||b - A x|| / ||b|| <= tol``.
    """
    diagonal = np.asarray((1.0e9, 1.0, 2.0))
    rhs = np.ones(3)
    stats = _counted_gmres(
        lambda v: jnp.asarray(diagonal) * v, jnp.asarray(rhs), tol=1.0e-7, max_iters=8
    )

    assert stats["converged"] is True
    assert stats["breakdown"] is False
    assert stats["iterations"] == 3
    assert stats["final_relative_residual"] <= 1.0e-7

    reference = _scipy_gmres(np.diag(diagonal), rhs, tol=1.0e-7)
    assert reference["converged"] is True
    assert reference["iterations"] == 3
    assert reference["measured"] <= 1.0e-7


def _diagonal_matvec(diagonal: np.ndarray):
    """Return the matrix-free action of ``diag(diagonal)``."""
    entries = jnp.asarray(diagonal, dtype=jnp.float64)
    return lambda v: entries * v


def test_ill_conditioned_diagonal_tracks_scipy_gmres() -> None:
    """A condition-number-1e8 system converges when SciPy's GMRES does.

    Twenty distinct eigenvalues logspaced over ``[1e-4, 1e4]`` need the full
    Krylov space, whose last column is an invariant-subspace breakdown decided
    by the measured residual.  Widening the spectrum to ``[1e-8, 1e8]``
    (condition number ``1e16``) defeats both implementations at ``1e-8``.
    """
    rhs = np.ones(20)
    for exponent, expected in ((4.0, True), (8.0, False)):
        diagonal = np.logspace(-exponent, exponent, 20)
        stats = _counted_gmres(
            _diagonal_matvec(diagonal), jnp.asarray(rhs), tol=1.0e-8, max_iters=40
        )
        reference = _scipy_gmres(np.diag(diagonal), rhs, tol=1.0e-8)

        assert stats["converged"] is expected
        assert reference["converged"] is expected
        assert stats["breakdown"] is not expected
        assert abs(stats["iterations"] - reference["iterations"]) <= 1
        if expected:
            assert stats["final_relative_residual"] <= 1.0e-8
            assert reference["measured"] <= 1.0e-8


def _stage2_regeneration(monkeypatch: pytest.MonkeyPatch):
    """Import the Stage-2 regeneration entry point the way its runners do."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "benchmarks"))
    return importlib.import_module("regenerate_stage2_conditioning")


def _converged_probe_solver() -> dict[str, object]:
    """Return the minimal converged solver metadata a persisted state needs."""
    return {
        "status": "converged",
        "converged": True,
        "newton_iters": 3,
        "final_residual_l2": 1.0e-12,
        "newton_tolerance": 1.0e-8,
    }


def test_source_state_cache_is_keyed_by_generation_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A state cached under one configuration must not be reused under another."""
    stage2 = _stage2_regeneration(monkeypatch)
    original = jnp.asarray((0.125, 0.5, 0.25), dtype=jnp.float64)
    replacement = jnp.asarray((0.5, 0.25, 0.125), dtype=jnp.float64)
    fingerprint = stage2._pme_source_fingerprint(BreakdownConfig(nx=3), 2, 1, 0.25)
    foreign = stage2._pme_source_fingerprint(BreakdownConfig(nx=3, epsilon=1.0e-3), 2, 1, 0.25)

    assert fingerprint != foreign

    stage2._load_or_persist_source_state(
        tmp_path, "pme", "m2_front1", fingerprint, lambda: (original, _converged_probe_solver())
    )

    def must_not_resolve() -> tuple[jax.Array, dict[str, object]]:
        raise AssertionError("an identical configuration was regenerated")

    reused, _ = stage2._load_or_persist_source_state(
        tmp_path, "pme", "m2_front1", fingerprint, must_not_resolve
    )

    assert np.array_equal(np.asarray(reused), np.asarray(original))

    regenerated, _ = stage2._load_or_persist_source_state(
        tmp_path, "pme", "m2_front1", foreign, lambda: (replacement, _converged_probe_solver())
    )

    assert np.array_equal(np.asarray(regenerated), np.asarray(replacement))
    with pytest.raises(stage2.StaleSourceStateError, match="different configuration"):
        stage2._load_saved_source_state(tmp_path, "pme", "m2_front1", fingerprint)


def test_record_loader_refuses_a_foreign_source_state_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An m2/front1 artifact must fail closed for an m8/front3 record."""
    stage2 = _stage2_regeneration(monkeypatch)
    config = BreakdownConfig(nx=3)
    state = jnp.asarray((0.125, 0.5, 0.25), dtype=jnp.float64)
    fingerprint = stage2._pme_source_fingerprint(config, 2, 1, config.front_target_halfwidths[0])
    _, solver = stage2._persist_source_state(
        tmp_path, "pme", "m2_front1", state, _converged_probe_solver(), fingerprint
    )
    matching = {"m": 2, "front_case": 1, "analysis_dt": 2.0, "d0_kind": "identity"}
    matching["reference_state_solver"] = solver
    foreign = dict(matching, m=8, front_case=3)

    loaded, _ = stage2._load_record_source_state(
        tmp_path, "pme", matching, stage2._pme_record_fingerprint(config, matching)
    )

    assert np.array_equal(np.asarray(loaded), np.asarray(state))
    with pytest.raises(RuntimeError, match="fingerprint differs"):
        stage2._load_record_source_state(
            tmp_path, "pme", foreign, stage2._pme_record_fingerprint(config, foreign)
        )


_PROBE_DIAGNOSTIC = {
    "supports_consistent": True,
    "corroboration_attempted": True,
    "supports_converged": True,
    "supports_corroborated": True,
    "max_support_residual": 1.0e-6,
    "disk_rate": 0.5,
    "epsilon_zero": 0.5,
    "epsilon_zero_reduced_arnoldi": 0.5,
    "full_operator_epsilon_zero": None,
    "full_operator_epsilon_zero_seconds": None,
    "epsilon_zero_full_operator_evidence": False,
    "verdict_reason": None,
    "arnoldi_k_requested": 6,
    "arnoldi_k_achieved": 6,
    "arnoldi_breakdown": False,
    "arnoldi_residual_norm": 0.0,
    "origin_enclosed": False,
    "n_right_real_outliers": 0,
    "predicted_gmres_factor": 0.5,
    "verdict": "adequate",
    "d0": 2.0,
}


def _persisted_const_record(
    stage2, tmp_path: Path, config: BreakdownConfig, d0_used: float
) -> dict[str, object]:
    """Persist a source state and return a const-D0 record that points at it."""
    state = jnp.asarray((0.125, 0.5, 0.25), dtype=jnp.float64)
    fingerprint = stage2._pme_source_fingerprint(config, 2, 1, config.front_target_halfwidths[0])
    _, solver = stage2._persist_source_state(
        tmp_path, "pme", "m2_front1", state, _converged_probe_solver(), fingerprint
    )
    grid = pme_breakdown.NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    analysis_dt = 2.0
    return {
        "m": 2,
        "front_case": 1,
        "analysis_dt": analysis_dt,
        "d0_kind": "const",
        "d0_used": d0_used,
        "sigma": d0_used * analysis_dt / grid.dx**2,
        "reference_state_solver": solver,
    }


def test_record_is_reassessed_on_its_own_recorded_d0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reassessed operator must carry the record's D0, not a fresh default."""
    stage2 = _stage2_regeneration(monkeypatch)
    config = BreakdownConfig(nx=3, const_d0=2.0)
    record = _persisted_const_record(stage2, tmp_path, config, 2.0)
    captured: dict[str, object] = {}

    def capture(state, grid, m, dt, epsilon, d0_kind, **kwargs):
        del state
        captured.update(
            {
                "nx": grid.nx,
                "m": m,
                "dt": dt,
                "epsilon": epsilon,
                "d0_kind": d0_kind,
                "const_value": kwargs["const_value"],
            }
        )
        return dict(_PROBE_DIAGNOSTIC)

    monkeypatch.setattr(stage2.pme_breakdown, "assess_pme_state", capture)
    stage2._assess_pme_record(tmp_path, record, (16, 60, 2), report_config=config._asdict())

    assert captured["const_value"] == 2.0
    assert captured["nx"] == 3
    assert captured["dt"] == 2.0
    assert captured["epsilon"] == config.epsilon


def test_record_whose_d0_contradicts_its_configuration_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record that the stored configuration cannot reproduce must fail closed."""
    stage2 = _stage2_regeneration(monkeypatch)
    config = BreakdownConfig(nx=3, const_d0=1.0)
    record = _persisted_const_record(stage2, tmp_path, config, 2.0)

    def must_not_assess(*args, **kwargs):
        raise AssertionError("a contradictory record was assessed")

    monkeypatch.setattr(stage2.pme_breakdown, "assess_pme_state", must_not_assess)
    with pytest.raises(RuntimeError, match="Reconstructed D0"):
        stage2._assess_pme_record(tmp_path, record, (16, 60, 2), report_config=config._asdict())


class _BatchWasRebuilt(RuntimeError):
    """Raised by a stubbed study runner to show that a batch was regenerated."""


def _pme_batch_config(stage2):
    """Return the batch configuration the PME batch runner builds for ``m = 2``."""
    return stage2.pme_breakdown.BreakdownConfig(m_values=(2,))


def _write_legacy_pme_source_state(stage2, checkpoint_dir: Path, source_key: str) -> dict:
    """Write a pre-fingerprint (v1) source-state artifact the way the old code did."""
    relative, array_path, manifest_path = stage2._source_state_paths(
        checkpoint_dir, "pme", source_key
    )
    state = jnp.asarray((0.125, 0.5, 0.25), dtype=jnp.float64)
    array_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(array_path, np.asarray(state, dtype=np.float64), allow_pickle=False)
    identity = stage2._source_state_identity(state)
    solver = dict(
        _converged_probe_solver(),
        source_state_identity=identity,
        source_state_artifact={
            "relative_path": str(relative),
            "source_state_identity": identity,
        },
    )
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "stage2_conditioning_source_state_v1",
                "study": "pme",
                "source_key": source_key,
                "relative_path": str(relative),
                "source_state_identity": identity,
                "state_solver": solver,
            }
        ),
        encoding="utf-8",
    )
    return solver


def _write_pme_batch(stage2, checkpoint_dir: Path, schema: str, solver: dict) -> Path:
    """Write a completed PME batch checkpoint under the requested batch schema."""
    record = {
        "m": 2,
        "front_case": 1,
        "analysis_dt": 2.0,
        "d0_kind": "identity",
        "actual_gmres": {"iterations": 7, "converged": True},
        "reference_state_solver": solver,
    }
    checkpoint = stage2._pme_checkpoint_path(checkpoint_dir, 2)
    checkpoint.write_text(
        json.dumps(
            {
                "schema": schema,
                "m": 2,
                "config": _pme_batch_config(stage2)._asdict(),
                "records": [record],
                "centering": {"note": "synthetic"},
                "runtime_seconds": 1.0,
                "batch_wall_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def _write_pme_dependent_checkpoint(stage2, checkpoint_dir: Path) -> Path:
    """Write one checkpoint derived from the assembled PME batches."""
    dependent = stage2._bba5a94_audit_checkpoint_path(checkpoint_dir, "pme")
    dependent.write_text(
        json.dumps({"schema": stage2.BBA5A94_AUDIT_SCHEMA, "audited": {}}), encoding="utf-8"
    )
    return dependent


def test_pre_change_pme_batch_checkpoint_is_rebuilt_not_returned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch completed before the artifact schema bump must not be returned.

    Its records were measured on v1 artifacts that carry no generation
    fingerprint, so assembly would republish diagnostics that reassessment
    then rejects.  The batch and the checkpoints derived from it are discarded
    and the batch is regenerated instead.
    """
    stage2 = _stage2_regeneration(monkeypatch)

    def rebuilt(*args: object, **kwargs: object) -> dict[str, object]:
        raise _BatchWasRebuilt("the batch runner regenerated the batch")

    monkeypatch.setattr(stage2.pme_breakdown, "run_breakdown_study", rebuilt)
    solver = _write_legacy_pme_source_state(stage2, tmp_path, "m2_front1")
    checkpoint = _write_pme_batch(stage2, tmp_path, "stage2_conditioning_pme_batch_v3", solver)
    dependent = _write_pme_dependent_checkpoint(stage2, tmp_path)

    with pytest.raises(_BatchWasRebuilt):
        stage2._run_pme_batch(tmp_path, 2)

    assert not checkpoint.is_file()
    assert not dependent.is_file()


def test_pme_batch_on_unfingerprinted_source_states_is_rebuilt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Carrying the current batch schema is not enough to accept a batch."""
    stage2 = _stage2_regeneration(monkeypatch)

    def rebuilt(*args: object, **kwargs: object) -> dict[str, object]:
        raise _BatchWasRebuilt("the batch runner regenerated the batch")

    monkeypatch.setattr(stage2.pme_breakdown, "run_breakdown_study", rebuilt)
    solver = _write_legacy_pme_source_state(stage2, tmp_path, "m2_front1")
    checkpoint = _write_pme_batch(stage2, tmp_path, stage2.PME_BATCH_SCHEMA, solver)
    dependent = _write_pme_dependent_checkpoint(stage2, tmp_path)

    with pytest.raises(_BatchWasRebuilt):
        stage2._run_pme_batch(tmp_path, 2)

    assert not checkpoint.is_file()
    assert not dependent.is_file()


def test_pme_batch_on_matching_source_states_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A batch whose artifacts are current with matching fingerprints is complete."""
    stage2 = _stage2_regeneration(monkeypatch)

    def must_not_rerun(*args: object, **kwargs: object) -> dict[str, object]:
        raise AssertionError("a compatible batch was unexpectedly regenerated")

    monkeypatch.setattr(stage2.pme_breakdown, "run_breakdown_study", must_not_rerun)
    config = _pme_batch_config(stage2)
    fingerprint = stage2._pme_source_fingerprint(
        config, 2, 1, config.front_target_halfwidths[0]
    )
    _, solver = stage2._persist_source_state(
        tmp_path,
        "pme",
        "m2_front1",
        jnp.asarray((0.125, 0.5, 0.25), dtype=jnp.float64),
        _converged_probe_solver(),
        fingerprint,
    )
    checkpoint = _write_pme_batch(stage2, tmp_path, stage2.PME_BATCH_SCHEMA, solver)
    dependent = _write_pme_dependent_checkpoint(stage2, tmp_path)

    assert stage2._run_pme_batch(tmp_path, 2) == checkpoint
    assert checkpoint.is_file()
    assert dependent.is_file()
