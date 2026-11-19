"""Fast algebra and protocol checks for tau-blend active-block helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from benchmarks import tau_blend_active_block as audit
from benchmarks.tau_blend_mask_analysis import (
    build_state,
    cfl_mask,
    helmholtz,
    jacobian,
    one_inverse_matrix,
)
from moljax.experimental.tau_blend_active_block import (
    DenseFovDisk,
    emulate_merged_policy,
    extract_active_block,
    lemma6_tail_lower_bound,
    true_residual_gmres,
)


def test_pending_direct_refresh_preserves_other_evidence(tmp_path: Path, monkeypatch) -> None:
    """The selected-degree refresh changes only the pending direct certificate."""
    source = Path(audit.__file__).parent / "results" / "tau_blend_active_block.json"
    output = tmp_path / source.name
    # Put the copy into the pending state. The test then does not depend on
    # the state of the committed file.
    before = json.loads(source.read_text())
    pending = before["direct_regularized_full_MJ_certificates"][0]["certificate_ladder"][0]
    pending["recomputation"] = {"status": "pending second-host recomputation"}
    before["certificate_recomputation"]["pending_second_host"] = (
        "direct regularized full MJ at N=128"
    )
    output.write_text(json.dumps(before))

    def refresh(row: dict) -> dict:
        selected = next(item for item in row["certificate_ladder"] if item["certified"])
        assert (row["nx"], selected["grid"], selected["degree"]) == (128, 565, 12)
        selected["contour_modulus_bound"] = 0.125
        return row

    monkeypatch.setattr(audit, "refresh_full_protocol", refresh)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau_blend_active_block.py",
            "--resume",
            "--nx",
            "128",
            "--refresh-full-selected",
            "--output",
            str(output),
        ],
    )
    audit.main()
    after = json.loads(output.read_text())
    assert after["cases"] == before["cases"]
    assert after["decision_summary"] == before["decision_summary"]
    assert after["ordering_check"] == before["ordering_check"]
    assert (
        after["direct_regularized_full_MJ_certificates"][1:]
        == before["direct_regularized_full_MJ_certificates"][1:]
    )
    updated = after["direct_regularized_full_MJ_certificates"][0]["certificate_ladder"][0]
    assert updated["contour_modulus_bound"] == 0.125
    assert updated["recomputation"]["status"] == "recomputed"
    assert updated["recomputation"]["host"] == "second host"
    assert "pending_second_host" not in after["certificate_recomputation"]
    assert (
        after["certificate_recomputation"]["recomputed_ladder_entries"]
        == before["certificate_recomputation"]["recomputed_ladder_entries"] + 1
    )


def _state(nx: int, m: int = 4) -> dict:
    config = SimpleNamespace(
        x_min=-4.0,
        x_max=4.0,
        t0=0.1,
        halfwidth=3.0,
        epsilon=1.0e-5,
    )
    return build_state(nx, m, 0.02, config)


def _blocks(nx: int):
    state = _state(nx)
    mask = cfl_mask(state, state["D"], 0.1)
    preconditioner = one_inverse_matrix(state, state["D"], mask)
    zeroed = np.where(mask, 0.0, state["D"])
    j0 = jacobian(state["L"], zeroed, state["dt"])
    return (
        state,
        mask,
        extract_active_block(preconditioner @ j0, mask),
        extract_active_block(preconditioner @ state["J"], mask),
    )


def _polynomial(matrix: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    value = np.zeros_like(matrix)
    power = np.eye(matrix.shape[0])
    for coefficient in coefficients:
        value += coefficient * power
        power = power @ matrix
    return value


def test_j0_has_exact_identity_block_and_regularization_preserves_y() -> None:
    """The block hypotheses hold exactly at both fast test dimensions."""
    for nx in (32, 64):
        _, _, j0, regularized = _blocks(nx)
        assert np.count_nonzero(j0.E) == 0
        assert np.count_nonzero(j0.F) == 0
        assert np.array_equal(j0.Y, regularized.Y)


def test_lemma1_polynomial_block_identity() -> None:
    """The claimed block polynomial identity holds for random q."""
    rng = np.random.default_rng(20261110)
    for nx in (32, 64):
        _, _, blocks, _ = _blocks(nx)
        coefficients = rng.normal(size=5)
        dimension = blocks.E.shape[0] + blocks.Y.shape[0]
        operator = np.block(
            [
                [np.eye(blocks.E.shape[0]), blocks.X],
                [np.zeros_like(blocks.F), blocks.Y],
            ]
        )
        q_operator = _polynomial(operator, coefficients)
        p_operator = (np.eye(dimension) - operator) @ q_operator
        q_y = _polynomial(blocks.Y, coefficients)
        expected = np.block(
            [
                [np.zeros_like(blocks.E), -blocks.X @ q_y],
                [np.zeros_like(blocks.F), (np.eye(blocks.Y.shape[0]) - blocks.Y) @ q_y],
            ]
        )
        assert np.allclose(p_operator, expected, rtol=2.0e-13, atol=2.0e-13)


def test_lemma6_lower_bound_never_exceeds_dense_tail_count() -> None:
    """The rank-corrected lower bound is below the measured scalar tail."""
    for nx in (32, 64):
        state = _state(nx)
        diffusivity = state["D"]
        floor = float(diffusivity.min())
        rank = int(np.count_nonzero(diffusivity > floor))
        reference = float(diffusivity.max())
        preconditioned = helmholtz(state, reference) @ state["J"]
        measured = int(np.count_nonzero(np.abs(np.linalg.eigvals(preconditioned)) < 0.1))
        bound = lemma6_tail_lower_bound(
            -state["lam"],
            diffusivity_floor=floor,
            scalar_reference=reference,
            dt=state["dt"],
            magnitude_threshold=0.1,
            rank_correction=rank,
        )
        assert bound.lower_bound <= measured


def _synthetic_fov(*, rate: float, enclosed: bool) -> DenseFovDisk:
    return DenseFovDisk(
        angles=np.asarray([0.0]),
        supports=np.asarray([1.0]),
        boundary=np.asarray([1.0 + 0.0j]),
        center=1.0 + 0.0j,
        radius=rate,
        disk_rate=rate,
        origin_enclosed=enclosed,
    )


def test_policy_emulation_uses_merged_thresholds() -> None:
    """Synthetic exact inputs exercise adequate, investigate and abstention."""
    operator = np.diag([1.0, 1.1, 1.2, 1.3])
    adequate = emulate_merged_policy(operator, _synthetic_fov(rate=0.2, enclosed=False))
    investigate = emulate_merged_policy(operator, _synthetic_fov(rate=1.1, enclosed=False))
    indeterminate = emulate_merged_policy(operator, _synthetic_fov(rate=0.2, enclosed=True))
    assert adequate["verdict"] == "adequate"
    assert investigate["verdict"] == "investigate"
    assert indeterminate["verdict"] == "indeterminate"
    assert adequate["thresholds"] == {
        "rate_threshold": 0.9,
        "eps_zero_threshold": 0.1,
        "max_right_real_outliers": 0,
    }
    assert len(adequate["policy_emulation_notes"]) == 4


def test_true_residual_protocol_returns_verified_counts() -> None:
    """The protocol records both stop conventions and verifies the true residual."""
    rng = np.random.default_rng(20261110)
    for nx in (32, 64):
        operator = np.diag(np.linspace(0.2, 2.0, nx)) + 0.02 * np.diag(
            np.ones(nx - 1),
            1,
        )
        preconditioner = np.diag(1.0 / np.diag(operator))
        rhs = rng.normal(size=nx)
        result = true_residual_gmres(
            preconditioner,
            operator,
            rhs,
            tolerance=1.0e-10,
        )
        residual = np.linalg.norm(rhs - operator @ result["x"]) / np.linalg.norm(rhs)
        assert result["converged"]
        assert residual == result["secondary_ratio"]
        assert residual <= 1.0e-10
        assert result["true_stop_count"] is not None
        assert result["preconditioned_stop_count"] is not None


def test_single_grid_refresh_truncates_later_entries_and_uses_all_degrees(monkeypatch) -> None:
    """An earlier certificate becomes the selected grid with recorded provenance."""
    state = {"D": np.ones(2), "L": np.eye(2), "J": np.eye(2), "dt": 1.0}
    monkeypatch.setattr(audit, "_state", lambda nx, m: state)
    monkeypatch.setattr(
        audit,
        "_preconditioner",
        lambda state, method: (np.eye(2), np.zeros(2, dtype=bool), np.ones(1)),
    )
    monkeypatch.setattr(audit, "jacobian", lambda *args: np.eye(2))
    monkeypatch.setattr(
        audit,
        "extract_active_block",
        lambda *args: SimpleNamespace(Y=np.eye(2), lemma1_constant=1.0),
    )
    seen_degrees = []

    def assess(matrix, **settings):
        seen_degrees.append(settings["polynomial_degrees"])
        return SimpleNamespace(certified=True, grid=settings["grid_points_per_axis"])

    monkeypatch.setattr(audit, "assess_dense_pseudospectral_criterion", assess)
    monkeypatch.setattr(
        audit,
        "_certificate_summary",
        lambda result, wall, constant: {"certified": result.certified, "grid": result.grid},
    )
    monkeypatch.setattr(audit, "benchmark_provenance", lambda script, wall: {"git_commit": "test"})
    row = {
        "nx": 32,
        "m": 2,
        "method": "cfl_reference_blend_B3",
        "Y_certificate_ladder": [
            {"grid": 121, "certified": False},
            {"grid": 241, "certified": True},
        ],
        "merged_policy": {"verdict": "indeterminate"},
    }
    refreshed = audit.refresh_case_grid(row, 121)
    assert [entry["grid"] for entry in refreshed["Y_certificate_ladder"]] == [121]
    assert refreshed["Y_certificate_selected"]["grid"] == 121
    assert refreshed["union"] == "certified"
    assert refreshed["Y_certificate_selected"]["recomputation"]["git_commit"] == "test"

    full = {"nx": 32, "certificate_ladder": [{"grid": 121, "certified": True}]}
    assert (
        audit.refresh_full_grid(full)["certificate_ladder"][0]["recomputation"]["git_commit"]
        == "test"
    )
    assert seen_degrees == [audit.DEGREES, audit.DEGREES]

    monkeypatch.setattr(
        audit,
        "assess_dense_pseudospectral_criterion",
        lambda *args, **kwargs: SimpleNamespace(certified=False, grid=121),
    )
    with pytest.raises(RuntimeError, match="selected certificate lost"):
        audit.refresh_case_grid(refreshed, 121)
