#!/usr/bin/env python3
"""Regenerate the Stage-2 conditioning studies in resumable fresh-process batches.

Each batch invokes an existing study runner for one independent exponent or
reaction-strength slice.  Completed batch reports are atomically checkpointed
outside the repository, then assembled without further numerical work into the
canonical result JSONs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import pme_breakdown
import porous_fisher_conditioning

DEFAULT_CHECKPOINT_DIR = Path("/tmp/moljax-stage2-conditioning-checkpoints")
BBA5A94_AUDIT_SCHEMA = "stage2_conditioning_bba5a94_verdict_audit_v1"
GEOMETRY_LADDER = ((32, 120, 2), (64, 180, 2), (96, 240, 2))
BASE_GEOMETRY_BUDGET = {"n_angles": 16, "fov_max_iters": 60, "fov_n_restarts": 2}

# These four measurements were completed before the broad escalation pass.  They
# are retained here so the final assembly can use their already-observed results
# without recomputing the priority reaction-axis cases.
PRIORITY_REACTION_RESOLUTIONS: dict[tuple[float, float], tuple[dict[str, Any], ...]] = {
    (0.0, 0.02): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.8310001082106886e-4,
            "disk_rate": 0.9542368997552261,
            "origin_enclosed": False,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 7.159925074859145e-5,
            "disk_rate": 0.9542369444437695,
            "origin_enclosed": False,
            "verdict": "investigate",
        },
    ),
    (0.0, 2.0): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.92385206626124e-4,
            "disk_rate": 1.0013565109857867,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 7.333308483450632e-5,
            "disk_rate": 1.0013565590117444,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 96,
            "fov_max_iters": 240,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 2.072477977866651e-5,
            "disk_rate": 1.0013565500432922,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
    ),
    (1.0, 0.02): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.730617069768882e-4,
            "disk_rate": 0.9551645357445586,
            "origin_enclosed": False,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 1.3566156479233245e-4,
            "disk_rate": 0.9551646300876108,
            "origin_enclosed": False,
            "verdict": "investigate",
        },
    ),
    (1.0, 2.0): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.820089498108155e-4,
            "disk_rate": 1.0023281370758816,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 1.3891133448470182e-4,
            "disk_rate": 1.002328238465557,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 96,
            "fov_max_iters": 240,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 1.7135275935423258e-5,
            "disk_rate": 1.0023282296128921,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
    ),
}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON checkpoint atomically in its target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one completed checkpoint or fail with its exact location."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing completed batch checkpoint: {path}") from error


def _pme_checkpoint_path(checkpoint_dir: Path, m: int) -> Path:
    """Return the checkpoint path for one independent PME exponent batch."""
    return checkpoint_dir / f"pme_m{m}.json"


def _porous_fisher_checkpoint_path(checkpoint_dir: Path, reaction: float) -> Path:
    """Return the checkpoint path for one independent reaction-strength batch."""
    return checkpoint_dir / f"porous_fisher_r{reaction:g}.json"


def _run_pme_batch(checkpoint_dir: Path, m: int) -> Path:
    """Run and atomically checkpoint one PME exponent batch if needed."""
    config = pme_breakdown.BreakdownConfig(m_values=(m,))
    checkpoint = _pme_checkpoint_path(checkpoint_dir, m)
    if checkpoint.exists():
        existing = _load_checkpoint(checkpoint)
        if existing.get("schema") == "stage2_conditioning_pme_batch_v1" and existing.get("m") == m:
            print(f"checkpoint exists: {checkpoint}")
            return checkpoint
        raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")

    working_output = checkpoint.with_suffix(".working.json")
    started_at = perf_counter()
    report = pme_breakdown.run_breakdown_study(config._replace(output_path=str(working_output)))
    working_output.unlink(missing_ok=True)
    payload = {
        "schema": "stage2_conditioning_pme_batch_v1",
        "m": m,
        "config": config._asdict(),
        "records": report["records"],
        "centering": report["centering"],
        "runtime_seconds": report["runtime_seconds"],
        "batch_wall_seconds": perf_counter() - started_at,
    }
    _atomic_write(checkpoint, payload)
    print(f"completed m={m}: records={len(payload['records'])} checkpoint={checkpoint}")
    return checkpoint


def _run_porous_fisher_batch(checkpoint_dir: Path, reaction: float) -> Path:
    """Run and atomically checkpoint one reaction-strength batch if needed."""
    config = porous_fisher_conditioning.ReactionStudyConfig(reaction_values=(reaction,))
    checkpoint = _porous_fisher_checkpoint_path(checkpoint_dir, reaction)
    if checkpoint.exists():
        existing = _load_checkpoint(checkpoint)
        if (
            existing.get("schema") == "stage2_conditioning_porous_fisher_batch_v1"
            and existing.get("reaction") == reaction
        ):
            print(f"checkpoint exists: {checkpoint}")
            return checkpoint
        raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")

    working_output = checkpoint.with_suffix(".working.json")
    started_at = perf_counter()
    report = porous_fisher_conditioning.run_reaction_study(
        config._replace(output_path=str(working_output))
    )
    working_output.unlink(missing_ok=True)
    payload = {
        "schema": "stage2_conditioning_porous_fisher_batch_v1",
        "reaction": reaction,
        "config": config._asdict(),
        "records": report["records"],
        "runtime_seconds": report["runtime_seconds"],
        "batch_wall_seconds": perf_counter() - started_at,
    }
    _atomic_write(checkpoint, payload)
    print(
        "completed "
        f"reaction={reaction:g}: records={len(payload['records'])} checkpoint={checkpoint}"
    )
    return checkpoint


def _assemble_pme(checkpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    """Assemble completed PME batches into the existing canonical report schema."""
    config = pme_breakdown.BreakdownConfig(output_path=str(output_path))
    records: list[dict[str, Any]] = []
    runtimes: list[float] = []
    centering: dict[str, Any] | None = None
    for m in config.m_values:
        checkpoint = _load_checkpoint(_pme_checkpoint_path(checkpoint_dir, m))
        if (
            checkpoint.get("schema") != "stage2_conditioning_pme_batch_v1"
            or checkpoint.get("m") != m
        ):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        records.extend(checkpoint["records"])
        runtimes.append(float(checkpoint["runtime_seconds"]))
        if centering is None:
            centering = checkpoint["centering"]

    if centering is None:
        raise RuntimeError("No PME batch checkpoints were available")
    decision = pme_breakdown._verdict_on_decision_procedure(records)
    dynamic_range = decision["identity_iteration_dynamic_range"]
    if not dynamic_range["meets_tenfold_gate"]:
        raise RuntimeError(
            "Identity GMRES dynamic-range gate failed: "
            f"min={dynamic_range['min']}, max={dynamic_range['max']}, "
            f"ratio={dynamic_range['ratio']}"
        )
    report = {
        "description": (
            "Experimental PME conditioning study. It evaluates a preconditioner decision "
            "procedure; it does not claim to fix stiffness degradation."
        ),
        "config": config._asdict(),
        "metric": {
            "accuracy": "not evaluated in this conditioning-only stress study",
            "timing": "total runtime only",
        },
        "runtime_seconds": sum(runtimes),
        "gmres_measurement_note": (
            "actual_gmres is an explicit residual-history count for the fixed system "
            "P^-1 J delta = P^-1 (-R), not NKStats.lin_iters."
        ),
        "state_schedule_note": (
            "Each state is produced by the stable reference backward-Euler step size state_dt "
            "with the frozen_bulk preconditioner.  Each analysis_dt then defines a separate "
            "fixed linearized backward-Euler system at that genuinely visited state."
        ),
        "centering": centering,
        "regime_claim": pme_breakdown._regime_claim(records),
        "rank_claim": pme_breakdown._rank_claim(records),
        "predictor_quality": pme_breakdown._predictor_quality(records),
        "correlation": pme_breakdown._correlation_pairs(records),
        "regime_map": pme_breakdown._regime_map(records),
        "verdict_on_decision_procedure": decision,
        "records": records,
    }
    _atomic_write(output_path, report)
    print(f"assembled PME: records={len(records)} output={output_path}")
    return report


def _assemble_porous_fisher(checkpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    """Assemble completed reaction batches into the existing canonical report schema."""
    config = porous_fisher_conditioning.ReactionStudyConfig(output_path=str(output_path))
    records: list[dict[str, Any]] = []
    runtimes: list[float] = []
    for reaction in config.reaction_values:
        checkpoint = _load_checkpoint(_porous_fisher_checkpoint_path(checkpoint_dir, reaction))
        if (
            checkpoint.get("schema") != "stage2_conditioning_porous_fisher_batch_v1"
            or checkpoint.get("reaction") != reaction
        ):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        records.extend(checkpoint["records"])
        runtimes.append(float(checkpoint["runtime_seconds"]))

    identity_records = [record for record in records if record["d0_kind"] == "identity"]
    identity_iterations = [
        float(record["actual_gmres"]["iterations"]) for record in identity_records
    ]
    minimum = min(identity_iterations)
    maximum = max(identity_iterations)
    ratio = float("inf") if minimum == 0.0 and maximum > 0.0 else maximum / minimum
    if ratio < 10.0:
        raise RuntimeError(
            f"Identity GMRES dynamic-range gate failed: min={minimum}, max={maximum}, ratio={ratio}"
        )
    report = {
        "description": (
            "Experimental Porous--Fisher conditioning study. It tests whether diagnostics flag "
            "hard states when a logistic reaction remains outside the diffusion preconditioner."
        ),
        "config": config._asdict(),
        "runtime_seconds": sum(runtimes),
        "physical_model": {
            "equation": "u_t = d_xx(u**2 + epsilon**2) + r*u*(1-u)",
            "diffusivity": "D(u)=2*u",
            "preconditioner_scope": "diffusion-only D0 Helmholtz; reaction is unpreconditioned",
        },
        "gmres_measurement_note": (
            "actual_gmres is the explicit residual-history count for P^-1 J delta=P^-1(-R), "
            "not NKStats.lin_iters."
        ),
        "regime_claim": porous_fisher_conditioning._regime_claim(records),
        "verdict_on_decision_procedure": porous_fisher_conditioning._verdict_on_decision_procedure(
            records
        ),
        "reaction_effect": porous_fisher_conditioning._reaction_effect(records),
        "records": records,
    }
    _atomic_write(output_path, report)
    print(f"assembled Porous--Fisher: records={len(records)} output={output_path}")
    return report


def _base_output_path(study: str) -> Path:
    """Return the assembled base report path for one conditioning study."""
    if study == "pme":
        return Path(pme_breakdown.BreakdownConfig().output_path)
    return Path(porous_fisher_conditioning.ReactionStudyConfig().output_path)


def _record_key(study: str, record: dict[str, Any]) -> str:
    """Return a stable key for one record in an assembled result report."""
    if study == "pme":
        return (
            f"m={int(record['m'])};front={int(record['front_case'])};"
            f"dt={float(record['analysis_dt']):.17g};d0={record['d0_kind']}"
        )
    return (
        f"r={float(record['r']):.17g};dt={float(record['analysis_dt']):.17g};d0={record['d0_kind']}"
    )


def _geometry_snapshot(
    record: dict[str, Any],
    *,
    n_angles: int,
    fov_max_iters: int,
    fov_n_restarts: int,
) -> dict[str, Any]:
    """Keep the bounded geometry data needed to audit one ladder attempt."""
    return {
        "n_angles": n_angles,
        "fov_max_iters": fov_max_iters,
        "fov_n_restarts": fov_n_restarts,
        "supports_consistent": bool(record["supports_consistent"]),
        "corroboration_attempted": bool(record["corroboration_attempted"]),
        "supports_converged": bool(record["supports_converged"]),
        "supports_corroborated": bool(record["supports_corroborated"]),
        "max_support_residual": float(record["max_support_residual"]),
        "disk_rate": float(record["disk_rate"]),
        "epsilon_zero": float(record["epsilon_zero"]),
        "origin_enclosed": bool(record["origin_enclosed"]),
        "n_right_real_outliers": (
            None
            if record["n_right_real_outliers"] is None
            else int(record["n_right_real_outliers"])
        ),
        "predicted_gmres_factor": (
            None
            if record["predicted_gmres_factor"] is None
            else float(record["predicted_gmres_factor"])
        ),
        "verdict": str(record["verdict"]),
    }


def _is_certified(snapshot: dict[str, Any]) -> bool:
    """Return whether a resolved two-restart geometry attempt is certified."""
    return bool(snapshot["supports_consistent"] and snapshot["corroboration_attempted"])


def _legacy_snapshot_to_current(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Translate persisted pre-bba5a94 certification fields without changing readings."""
    updated = dict(snapshot)
    legacy_certified = updated.pop("geometry_certified", None)
    if "supports_consistent" not in updated:
        if legacy_certified is None:
            raise RuntimeError("Persisted geometry snapshot lacks a certification field")
        updated["supports_consistent"] = bool(legacy_certified)
    updated.setdefault("corroboration_attempted", updated["fov_n_restarts"] >= 2)
    return updated


def _resolution_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the atomic per-record geometry-resolution checkpoint path."""
    return checkpoint_dir / f"{study}_geometry_resolution_v1.json"


def _base_digest(path: Path) -> str:
    """Return a stable identity for the already-regenerated base result report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base_report(study: str) -> tuple[Path, dict[str, Any]]:
    """Load the local 16/60/2 report without any numerical work."""
    path = _base_output_path(study)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing assembled base result report: {path}") from error


def _load_resolution_checkpoint(
    checkpoint_dir: Path,
    study: str,
    base_path: Path,
) -> dict[str, Any]:
    """Load or initialize one resumable per-record resolution checkpoint."""
    path = _resolution_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(base_path)
    if not path.exists():
        return {
            "schema": "stage2_conditioning_geometry_resolution_v1",
            "study": study,
            "base_result": str(base_path),
            "base_sha256": digest,
            "resolved": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != "stage2_conditioning_geometry_resolution_v1"
        or payload.get("study") != study
        or payload.get("base_result") != str(base_path)
        or payload.get("base_sha256") != digest
    ):
        raise RuntimeError(f"Resolution checkpoint identity mismatch: {path}")
    return payload


def _priority_reaction_resolution(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return a previously observed priority resolution without rerunning it."""
    if "r" not in record or record["d0_kind"] != "identity":
        return None
    key = (float(record["r"]), float(record["analysis_dt"]))
    attempts = PRIORITY_REACTION_RESOLUTIONS.get(key)
    if attempts is None:
        return None
    final = dict(attempts[-1])
    if not _is_certified(final):
        raise RuntimeError(f"Priority resolution is not certified: r={key[0]}, dt={key[1]}")
    return {
        "status": "CERTIFIED",
        "source": "prior_priority_probe",
        "attempts": [dict(attempt) for attempt in attempts],
        "final": final,
        "final_category": f"CERTIFIED_{final['verdict'].upper()}",
        "final_verdict": final["verdict"],
    }


def _assess_pme_record(record: dict[str, Any], budget: tuple[int, int, int]) -> dict[str, Any]:
    """Recreate one PME state and assess it at one requested geometry budget."""
    n_angles, fov_max_iters, fov_n_restarts = budget
    config = pme_breakdown.BreakdownConfig(
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    grid = pme_breakdown.NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    m = int(record["m"])
    target_halfwidth = float(record["target_support_halfwidth"])
    state = pme_breakdown._initial_state(grid, m, config.t0, target_halfwidth)
    state, _ = pme_breakdown._solve_one_step(state, grid, m, config, "frozen_bulk")
    diagnostic = pme_breakdown.assess_pme_state(
        state,
        grid,
        float(m),
        float(record["analysis_dt"]),
        0.0 if m == 1 else config.epsilon,
        str(record["d0_kind"]),
        const_value=config.const_d0,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_residual_tolerance=config.fov_residual_tolerance,
        fov_n_restarts=fov_n_restarts,
        arnoldi_steps=config.arnoldi_steps,
        seed=(
            20260900
            + 1000 * m
            + 10 * int(record["front_case"])
            + config.d0_kinds.index(str(record["d0_kind"]))
        ),
    )
    return _geometry_snapshot(
        diagnostic,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )


def _assess_porous_fisher_record(
    record: dict[str, Any], budget: tuple[int, int, int]
) -> dict[str, Any]:
    """Recreate one reaction-axis state and assess it at one geometry budget."""
    n_angles, fov_max_iters, fov_n_restarts = budget
    config = porous_fisher_conditioning.ReactionStudyConfig(
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    grid = porous_fisher_conditioning.NodeCenteredDirichletGrid.uniform(
        config.nx, config.x_min, config.x_max
    )
    initial = porous_fisher_conditioning.porous_fisher_traveling_wave(
        grid.x_coords(),
        config.initial_time,
        r=config.reference_wave_r,
        c=porous_fisher_conditioning.wave_speed(config.reference_wave_r),
    )
    reaction = float(record["r"])
    state, _ = porous_fisher_conditioning._advance_to_visited_state(
        initial, grid, r=reaction, config=config
    )
    d0_kind = str(record["d0_kind"])
    diagnostic = porous_fisher_conditioning.assess_porous_fisher_state(
        state,
        grid,
        r=reaction,
        dt=float(record["analysis_dt"]),
        epsilon=config.epsilon,
        d0_kind=d0_kind,
        const_value=config.const_d0,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_residual_tolerance=config.fov_residual_tolerance,
        fov_n_restarts=fov_n_restarts,
        arnoldi_steps=config.arnoldi_steps,
        seed=(
            20260880
            + 1000 * int(100 * reaction)
            + 10 * int(100 * float(record["analysis_dt"]))
            + config.d0_kinds.index(d0_kind)
        ),
    )
    return _geometry_snapshot(
        diagnostic,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )


def _resolve_record(study: str, record: dict[str, Any]) -> dict[str, Any]:
    """Run the bounded ladder for one previously uncertified record."""
    assessor = _assess_pme_record if study == "pme" else _assess_porous_fisher_record
    attempts: list[dict[str, Any]] = []
    for budget in GEOMETRY_LADDER:
        attempt = assessor(record, budget)
        attempts.append(attempt)
        if _is_certified(attempt):
            return {
                "status": "CERTIFIED",
                "source": "escalation",
                "attempts": attempts,
                "final": attempt,
                "final_category": f"CERTIFIED_{attempt['verdict'].upper()}",
                "final_verdict": attempt["verdict"],
            }
    return {
        "status": "UNCERTIFIED_AT_CAP",
        "source": "escalation",
        "attempts": attempts,
        "final": attempts[-1],
        "final_category": "UNCERTIFIED_AT_CAP",
        "final_verdict": None,
    }


def _resolve_one_record(checkpoint_dir: Path, study: str, key: str) -> None:
    """Resolve one eligible record and atomically checkpoint its whole ladder."""
    base_path, report = _load_base_report(study)
    records = {_record_key(study, record): record for record in report["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} resolution key: {key}") from error
    if _is_certified(_geometry_snapshot(record, **BASE_GEOMETRY_BUDGET)):
        raise RuntimeError(f"Record is already certified at the base budget: {key}")
    if _priority_reaction_resolution(record) is not None:
        raise RuntimeError(f"Record was already resolved by the priority probe: {key}")
    checkpoint = _load_resolution_checkpoint(checkpoint_dir, study, base_path)
    if key in checkpoint["resolved"]:
        print(f"resolution checkpoint exists: {key}")
        return
    started_at = perf_counter()
    resolution = _resolve_record(study, record)
    resolution["wall_seconds"] = perf_counter() - started_at
    checkpoint["resolved"][key] = resolution
    _atomic_write(_resolution_checkpoint_path(checkpoint_dir, study), checkpoint)
    print(
        f"resolved {key}: category={resolution['final_category']} "
        f"attempts={len(resolution['attempts'])} "
        f"checkpoint={_resolution_checkpoint_path(checkpoint_dir, study)}"
    )


def _resolution_metadata(
    study: str,
    record: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Return final, auditable geometry metadata for one base-report record."""
    base = _geometry_snapshot(record, **BASE_GEOMETRY_BUDGET)
    if _is_certified(base):
        return {
            "status": "CERTIFIED",
            "source": "base",
            "attempts": [base],
            "final": base,
            "final_category": f"CERTIFIED_{base['verdict'].upper()}",
            "final_verdict": base["verdict"],
        }
    priority = _priority_reaction_resolution(record)
    if priority is not None:
        return priority
    key = _record_key(study, record)
    try:
        return checkpoint["resolved"][key]
    except KeyError as error:
        raise RuntimeError(f"Missing completed resolution checkpoint: {key}") from error


def _attach_final_resolution(
    study: str,
    report: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Attach bounded final categories without losing the base measurement fields."""
    final_records: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    budget_counts: dict[str, int] = {}
    cap_corroboration_failures = 0
    cap_residual_failures = 0
    for record in report["records"]:
        updated = dict(record)
        resolution = _resolution_metadata(study, record, checkpoint)
        final = resolution["final"]
        final_budget = f"{final['n_angles']}/{final['fov_max_iters']}/{final['fov_n_restarts']}"
        updated["geometry_resolution"] = resolution
        updated["final_verdict"] = resolution["final_verdict"]
        updated["final_category"] = resolution["final_category"]
        category_counts[resolution["final_category"]] = (
            category_counts.get(resolution["final_category"], 0) + 1
        )
        if resolution["status"] == "CERTIFIED":
            budget_counts[final_budget] = budget_counts.get(final_budget, 0) + 1
        else:
            cap_corroboration_failures += not bool(final["supports_corroborated"])
            cap_residual_failures += not bool(final["supports_converged"])
        final_records.append(updated)
    updated_report = dict(report)
    updated_report["records"] = final_records
    updated_report["geometry_resolution"] = {
        "description": (
            "A record has a final certified category only after support convergence and "
            "two-restart corroboration.  Records still uncertified at 96/240/2 are retained "
            "as UNCERTIFIED_AT_CAP rather than interpreted as origin-enclosure evidence."
        ),
        "base_budget": BASE_GEOMETRY_BUDGET,
        "escalation_ladder": [
            {
                "n_angles": n_angles,
                "fov_max_iters": fov_max_iters,
                "fov_n_restarts": fov_n_restarts,
            }
            for n_angles, fov_max_iters, fov_n_restarts in GEOMETRY_LADDER
        ],
        "final_category_counts": category_counts,
        "certifying_budget_counts": budget_counts,
        "uncertified_at_cap_failures": {
            "corroboration_failed": cap_corroboration_failures,
            "support_not_converged": cap_residual_failures,
        },
    }
    return updated_report


def _assemble_final_resolution(checkpoint_dir: Path, study: str, output_path: Path) -> None:
    """Write one result report with final geometry-resolution metadata attached."""
    base_path, report = _load_base_report(study)
    checkpoint = _load_resolution_checkpoint(checkpoint_dir, study, base_path)
    final_report = _attach_final_resolution(study, report, checkpoint)
    _atomic_write(output_path, final_report)
    print(
        f"assembled final {study} resolution: records={len(final_report['records'])} output={output_path}"
    )


def _bba5a94_audit_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the external checkpoint for one current-main verdict audit."""
    return checkpoint_dir / f"{study}_bba5a94_verdict_audit_v1.json"


def _recorded_final_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    """Return the recorded terminal ladder attempt for one assembled record."""
    try:
        return dict(record["geometry_resolution"]["final"])
    except KeyError as error:
        raise RuntimeError(f"Record lacks final geometry-resolution metadata: {record}") from error


def _recorded_final_budget(record: dict[str, Any]) -> tuple[int, int, int]:
    """Return the already-recorded budget; this refresh never searches the ladder."""
    final = _recorded_final_snapshot(record)
    return (
        int(final["n_angles"]),
        int(final["fov_max_iters"]),
        int(final["fov_n_restarts"]),
    )


def _load_bba5a94_audit_checkpoint(
    checkpoint_dir: Path,
    study: str,
    result_path: Path,
) -> dict[str, Any]:
    """Load or initialize the external per-record current-main audit checkpoint."""
    path = _bba5a94_audit_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(result_path)
    if not path.exists():
        return {
            "schema": BBA5A94_AUDIT_SCHEMA,
            "study": study,
            "result_path": str(result_path),
            "result_sha256": digest,
            "audited": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != BBA5A94_AUDIT_SCHEMA
        or payload.get("study") != study
        or payload.get("result_path") != str(result_path)
        or payload.get("result_sha256") != digest
    ):
        raise RuntimeError(f"Current-main audit checkpoint identity mismatch: {path}")
    return payload


def _category_from_snapshot(snapshot: dict[str, Any]) -> str:
    """Map a current certification observation to the existing final-category contract."""
    if not _is_certified(snapshot):
        return "UNCERTIFIED_AT_CAP"
    return f"CERTIFIED_{snapshot['verdict'].upper()}"


def _recorded_current_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized recorded terminal snapshot for a current-main comparison."""
    expected = _legacy_snapshot_to_current(_recorded_final_snapshot(record))
    expected.setdefault("epsilon_zero", float(record["epsilon_zero"]))
    expected.setdefault("n_right_real_outliers", record["n_right_real_outliers"])
    expected.setdefault("predicted_gmres_factor", record["predicted_gmres_factor"])
    return expected


def _verify_recorded_budget(record: dict[str, Any], observed: dict[str, Any], key: str) -> None:
    """Fail closed if an audit did not use the record's stored terminal budget."""
    expected_budget = _recorded_final_budget(record)
    observed_budget = (
        observed["n_angles"],
        observed["fov_max_iters"],
        observed["fov_n_restarts"],
    )
    if observed_budget != expected_budget:
        raise RuntimeError(
            f"Certification refresh used the wrong budget for {key}: "
            f"expected={expected_budget}, observed={observed_budget}"
        )
    if not observed["corroboration_attempted"]:
        raise RuntimeError(f"Current-main audit did not attempt corroboration: {key}")


def _verdict_shift_reason(
    record: dict[str, Any], expected: dict[str, Any], observed: dict[str, Any]
) -> str | None:
    """Return a closed operational label for a changed current-main verdict."""
    if observed["verdict"] == expected["verdict"]:
        return None
    if observed["n_right_real_outliers"] is None:
        return "domain_guard_null_outliers"
    if not _is_certified(observed):
        return "insufficient_geometry_resolution"
    if observed["origin_enclosed"] and not expected["origin_enclosed"]:
        return "origin_enclosure"
    if observed["n_right_real_outliers"] != record["n_right_real_outliers"]:
        return "new_outlier_gate"
    return "other"


def _audit_one_bba5a94_record(checkpoint_dir: Path, study: str, key: str) -> None:
    """Reassess one stored terminal budget and atomically checkpoint its current reading."""
    result_path, report = _load_base_report(study)
    records = {_record_key(study, record): record for record in report["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} current-main audit key: {key}") from error
    checkpoint = _load_bba5a94_audit_checkpoint(checkpoint_dir, study, result_path)
    if key in checkpoint["audited"]:
        print(f"current-main audit checkpoint exists: {key}")
        return
    assessor = _assess_pme_record if study == "pme" else _assess_porous_fisher_record
    started_at = perf_counter()
    observed = assessor(record, _recorded_final_budget(record))
    _verify_recorded_budget(record, observed, key)
    expected = _recorded_current_snapshot(record)
    reason = _verdict_shift_reason(record, expected, observed)
    checkpoint["audited"][key] = {
        "recorded": expected,
        "observed": observed,
        "recorded_final_category": str(record["final_category"]),
        "current_final_category": _category_from_snapshot(observed),
        "verdict_changed": reason is not None,
        "change_reason": reason,
        "wall_seconds": perf_counter() - started_at,
    }
    path = _bba5a94_audit_checkpoint_path(checkpoint_dir, study)
    _atomic_write(path, checkpoint)
    print(
        f"audited {key}: category={_category_from_snapshot(observed)} "
        f"changed={reason is not None} reason={reason} checkpoint={path}"
    )


def _list_pending_bba5a94_audit(checkpoint_dir: Path, study: str) -> None:
    """Print every terminal record that still needs its current-main audit."""
    result_path, report = _load_base_report(study)
    checkpoint = _load_bba5a94_audit_checkpoint(checkpoint_dir, study, result_path)
    pending = [
        _record_key(study, record)
        for record in report["records"]
        if _record_key(study, record) not in checkpoint["audited"]
    ]
    print(json.dumps(pending, indent=2))


def _migrate_resolution_metadata(
    resolution: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Replace obsolete certification fields with the current terminal observation."""
    updated = deepcopy(resolution)
    updated["attempts"] = [_legacy_snapshot_to_current(attempt) for attempt in updated["attempts"]]
    final = dict(audit["observed"])
    category = str(audit["current_final_category"])
    updated["final"] = final
    updated["status"] = "CERTIFIED" if _is_certified(final) else "UNCERTIFIED_AT_CAP"
    updated["final_category"] = category
    updated["final_verdict"] = final["verdict"] if _is_certified(final) else None
    updated["bba5a94_audit"] = {
        "recorded_final_category": audit["recorded_final_category"],
        "verdict_changed": bool(audit["verdict_changed"]),
        "change_reason": audit["change_reason"],
    }
    return updated


def _assemble_bba5a94_audit(checkpoint_dir: Path, study: str, output_path: Path) -> None:
    """Apply every current-main terminal reading after the complete recorded-budget audit."""
    result_path, report = _load_base_report(study)
    checkpoint = _load_bba5a94_audit_checkpoint(checkpoint_dir, study, result_path)
    records = report["records"]
    expected_keys = {_record_key(study, record) for record in records}
    missing = expected_keys.difference(checkpoint["audited"])
    if missing:
        raise RuntimeError(
            f"Cannot assemble current-main audit; {len(missing)} records remain: {sorted(missing)[:3]}"
        )

    updated_records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    changed_records: list[str] = []
    for record in records:
        key = _record_key(study, record)
        audit = checkpoint["audited"][key]
        observed = audit["observed"]
        updated = dict(record)
        updated.pop("geometry_certified", None)
        if "rates" in updated:
            updated["rates"] = dict(updated["rates"])
            updated["rates"].pop("geometry_certified", None)
        for field in (
            "supports_consistent",
            "corroboration_attempted",
            "supports_converged",
            "supports_corroborated",
            "max_support_residual",
            "disk_rate",
            "epsilon_zero",
            "origin_enclosed",
            "n_right_real_outliers",
            "predicted_gmres_factor",
            "verdict",
        ):
            updated[field] = observed[field]
        updated["geometry_resolution"] = _migrate_resolution_metadata(
            record["geometry_resolution"], audit
        )
        updated["final_category"] = audit["current_final_category"]
        updated["final_verdict"] = observed["verdict"] if _is_certified(observed) else None
        category = str(audit["current_final_category"])
        categories[category] = categories.get(category, 0) + 1
        if audit["verdict_changed"]:
            changed_records.append(key)
            reason = str(audit["change_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        updated_records.append(updated)

    updated_report = dict(report)
    updated_report["records"] = updated_records
    report_resolution = dict(report["geometry_resolution"])
    report_resolution["description"] = (
        "Each final terminal reading was reassessed on bba5a94 at its previously recorded "
        "budget with two restarts. A certified category requires consistent supports and "
        "corroboration attempted; an indeterminate result with certified supports is retained "
        "as a current-main fail-closed verdict, while inconsistent support evidence at the cap "
        "is retained as UNCERTIFIED_AT_CAP."
    )
    report_resolution["final_category_counts"] = categories
    report_resolution["bba5a94_audit"] = {
        "checkpoint_schema": BBA5A94_AUDIT_SCHEMA,
        "recorded_budget_only": True,
        "verdict_changed_count": len(changed_records),
        "change_reason_counts": reason_counts,
        "changed_record_keys": changed_records,
    }
    updated_report["geometry_resolution"] = report_resolution
    serialized = json.dumps(updated_report, sort_keys=True)
    if "geometry_certified" in serialized:
        raise RuntimeError("Current-main audit left an obsolete geometry_certified field")
    _atomic_write(output_path, updated_report)
    print(
        f"assembled current-main verdict audit: study={study} "
        f"records={len(updated_records)} output={output_path}"
    )


def _list_pending_resolution(checkpoint_dir: Path, study: str) -> None:
    """Print deterministic keys for uncertified records not resolved by a prior probe."""
    base_path, report = _load_base_report(study)
    checkpoint = _load_resolution_checkpoint(checkpoint_dir, study, base_path)
    pending = []
    for record in report["records"]:
        if _is_certified(_geometry_snapshot(record, **BASE_GEOMETRY_BUDGET)):
            continue
        if _priority_reaction_resolution(record) is not None:
            continue
        key = _record_key(study, record)
        if key not in checkpoint["resolved"]:
            pending.append(key)
    print(json.dumps(pending, indent=2))


def main() -> None:
    """Run one fresh-process batch or assemble all completed batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("pme", "porous_fisher"), required=True)
    parser.add_argument("--batch", help="PME exponent or Porous--Fisher reaction value")
    parser.add_argument("--assemble", action="store_true", help="Assemble completed checkpoints")
    parser.add_argument(
        "--list-pending-resolution",
        action="store_true",
        help="List base-uncertified records not resolved by an existing checkpoint",
    )
    parser.add_argument(
        "--resolve-record",
        help="Resolve one listed record through the bounded geometry ladder",
    )
    parser.add_argument(
        "--assemble-final-resolution",
        action="store_true",
        help="Attach final certified or at-cap metadata to the result report",
    )
    parser.add_argument(
        "--list-pending-bba5a94-audit",
        action="store_true",
        help="List terminal records not yet audited at their recorded current-main budget",
    )
    parser.add_argument(
        "--audit-bba5a94-record",
        help="Audit one record at its already-recorded terminal geometry budget",
    )
    parser.add_argument(
        "--assemble-bba5a94-audit",
        action="store_true",
        help="Apply all audited current-main verdicts and certification fields",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, help="Override the canonical result path")
    args = parser.parse_args()
    actions = sum(
        bool(action)
        for action in (
            args.batch,
            args.assemble,
            args.list_pending_resolution,
            args.resolve_record,
            args.assemble_final_resolution,
            args.list_pending_bba5a94_audit,
            args.audit_bba5a94_record,
            args.assemble_bba5a94_audit,
        )
    )
    if actions != 1:
        parser.error("Specify exactly one action")

    if args.list_pending_resolution:
        _list_pending_resolution(args.checkpoint_dir, args.study)
        return
    if args.resolve_record:
        _resolve_one_record(args.checkpoint_dir, args.study, args.resolve_record)
        return
    if args.list_pending_bba5a94_audit:
        _list_pending_bba5a94_audit(args.checkpoint_dir, args.study)
        return
    if args.audit_bba5a94_record:
        _audit_one_bba5a94_record(args.checkpoint_dir, args.study, args.audit_bba5a94_record)
        return
    if args.assemble_final_resolution:
        _assemble_final_resolution(
            args.checkpoint_dir,
            args.study,
            args.output or _base_output_path(args.study),
        )
        return
    if args.assemble_bba5a94_audit:
        _assemble_bba5a94_audit(
            args.checkpoint_dir,
            args.study,
            args.output or _base_output_path(args.study),
        )
        return

    if args.study == "pme":
        if args.assemble:
            _assemble_pme(
                args.checkpoint_dir,
                args.output or Path(pme_breakdown.BreakdownConfig().output_path),
            )
            return
        try:
            m = int(args.batch)
        except ValueError as error:
            parser.error(f"PME batch must be an integer exponent: {error}")
        if m not in pme_breakdown.BreakdownConfig().m_values:
            parser.error(f"Unsupported PME exponent: {m}")
        _run_pme_batch(args.checkpoint_dir, m)
        return

    if args.assemble:
        _assemble_porous_fisher(
            args.checkpoint_dir,
            args.output or Path(porous_fisher_conditioning.ReactionStudyConfig().output_path),
        )
        return
    try:
        reaction = float(args.batch)
    except ValueError as error:
        parser.error(f"Porous--Fisher batch must be a float reaction strength: {error}")
    if reaction not in porous_fisher_conditioning.ReactionStudyConfig().reaction_values:
        parser.error(f"Unsupported reaction strength: {reaction:g}")
    _run_porous_fisher_batch(args.checkpoint_dir, reaction)


if __name__ == "__main__":
    main()
