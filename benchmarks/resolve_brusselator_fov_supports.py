#!/usr/bin/env python3
"""Resolve Brusselator FOV support geometry from persisted v4 source states.

The base regeneration deliberately uses a cheap FOV budget.  This companion
runner upgrades only records whose support geometry was unresolved, always in
one fresh numerical process per record.  It never evolves a source state:
``reassess_brusselator_record`` reloads and SHA256-validates the exact v4
artifact before rebuilding the stored operator configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from benchmarks.brusselator_conditioning import (
    HOPF_REGIME,
    PRESETS,
    TURING_REGIME,
    _cache_relative_source_artifact,
    _fixed_transition,
    _records_for,
    _summary,
    reassess_brusselator_record,
)
from moljax.conditioning.non_normality import _reading_defect

STUDIES = (
    "screen_64",
    "developed_64",
    "fixed_dt_256",
    "hopf_continuation_256",
)
OUTPUT_NAMES = {
    "screen_64": "brusselator_conditioning.json",
    "developed_64": "brusselator_conditioning_developed.json",
    "fixed_dt_256": "brusselator_conditioning_fixed_dt.json",
    "hopf_continuation_256": "brusselator_conditioning_hopf_continuation.json",
}
EXPECTED_RECORDS = {
    "screen_64": 8,
    "developed_64": 12,
    "fixed_dt_256": 8,
    "hopf_continuation_256": 4,
}
BASE_BUDGET = (4, 8, 2)
FOV_SUPPORT_LADDER = (
    (32, 120, 2),
    (64, 180, 2),
    (96, 240, 2),
)
RESOLUTION_SCHEMA = "brusselator_fov_support_resolution_v1"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON payload atomically, preserving completed checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _record_key(record: dict[str, Any]) -> str:
    """Return a stable study-local identity for one persisted diagnostic record."""
    state = record.get("trajectory_step", record.get("state_index"))
    if state is None:
        raise RuntimeError("record lacks a source-state position")
    return f"{record['regime']};state={state};preconditioner={record['preconditioner']}"


def _base_path(checkpoint_dir: Path, study: str) -> Path:
    return checkpoint_dir / "results" / OUTPUT_NAMES[study]


def _resolution_path(checkpoint_dir: Path, study: str) -> Path:
    return checkpoint_dir / "fov_support_resolution" / f"{study}.json"


def _result_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base(checkpoint_dir: Path, study: str) -> tuple[Path, dict[str, Any]]:
    path = _base_path(checkpoint_dir, study)
    if not path.is_file():
        raise RuntimeError(f"base study output is missing: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or len(report.get("records", ())) != EXPECTED_RECORDS[study]:
        raise RuntimeError(f"base study output is incomplete: {path}")
    return path, report


def _load_resolution(checkpoint_dir: Path, study: str, base_path: Path) -> dict[str, Any]:
    """Load a resolution checkpoint only for the exact raw result snapshot."""
    path = _resolution_path(checkpoint_dir, study)
    initial = {
        "schema": RESOLUTION_SCHEMA,
        "base_result_path": str(base_path),
        "base_result_sha256": _result_hash(base_path),
        "resolved": {},
    }
    if not path.is_file():
        return initial
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    if checkpoint.get("schema") != RESOLUTION_SCHEMA:
        raise RuntimeError(f"incompatible FOV resolution checkpoint: {path}")
    if checkpoint.get("base_result_path") != str(base_path):
        raise RuntimeError(f"FOV resolution checkpoint points to another base result: {path}")
    if checkpoint.get("base_result_sha256") != initial["base_result_sha256"]:
        raise RuntimeError(f"FOV resolution checkpoint base hash mismatch: {path}")
    if not isinstance(checkpoint.get("resolved"), dict):
        raise RuntimeError(f"FOV resolution checkpoint is malformed: {path}")
    return checkpoint


def _attempt(assessment: dict[str, Any], budget: tuple[int, int, int]) -> dict[str, Any]:
    """Persist one fully auditable terminal observation at an explicit budget."""
    n_angles, fov_max_iters, fov_n_restarts = budget
    return {
        "budget": {
            "n_angles": n_angles,
            "fov_max_iters": fov_max_iters,
            "fov_n_restarts": fov_n_restarts,
        },
        "assessment": assessment,
        "supports_consistent": bool(assessment.get("supports_consistent")),
        "supports_converged": bool(assessment.get("supports_converged")),
        "supports_corroborated": bool(assessment.get("supports_corroborated")),
        "corroboration_attempted": bool(assessment.get("corroboration_attempted")),
    }


def _cap_reason(attempt: dict[str, Any]) -> str:
    """Return the fail-closed reason left after exhausting the FOV ladder."""
    if not attempt["supports_converged"] and not attempt["supports_corroborated"]:
        return "support_not_converged_and_restart_corroboration_failed"
    if not attempt["supports_converged"]:
        return "support_not_converged"
    if not attempt["supports_corroborated"]:
        return "restart_corroboration_failed"
    return "diagnostic_not_certifiable"


def _weak_bound_override_eligible(assessment: dict[str, Any]) -> bool:
    """Return whether a weak certificate may refine an otherwise usable reading."""
    return (
        str(assessment.get("verdict")) in {"investigate", "provisional"}
        and assessment.get("n_right_real_outliers") is not None
    )


def _unrecoverable_reading(assessment: dict[str, Any]) -> str | None:
    """Return why a stored non-abstaining verdict lacks a usable reading, or ``None``.

    ``assess_preconditioner`` measures the right-real outlier count only when
    the Ritz spectrum has enough finite values and the disk rate and
    epsilon_zero are usable readings; otherwise it abstains.  The pre-cce8089
    weak-bound override promoted such abstentions to ``provisional``.  A
    stored verdict resting on such a reading cannot be recovered from the
    checkpoint, so it is not trusted.
    """
    if str(assessment.get("verdict")) not in {"adequate", "provisional", "investigate"}:
        return None
    if assessment.get("n_right_real_outliers") is None:
        return (
            "stored reading has no measured right-real outlier count (too few or "
            "non-finite Ritz values, or an unusable disk_rate/epsilon_zero), so its "
            "stored verdict cannot be recovered"
        )
    try:
        defect = _reading_defect(assessment["disk_rate"], assessment["epsilon_zero"])
    except (KeyError, TypeError, ValueError):
        defect = "disk_rate or epsilon_zero is missing or not a number"
    if defect is not None:
        return f"stored reading is unusable ({defect}), so its stored verdict cannot be recovered"
    return None


def _policy_outcome(assessment: dict[str, Any]) -> tuple[str, str | None]:
    """Apply bound-evidence precedence without erasing an invalid abstention.

    A stored verdict whose reading cannot be recovered (for example one
    promoted by the pre-cce8089 override) is indeterminate.  A weak but valid
    Fourier--Weyl--ghost certificate is distinct from no certificate: if
    geometry is corroborated and the origin is outside, it remains
    provisional only when the underlying reading was a usable
    investigate/provisional result.  Origin enclosure, support failure, and
    invalid/incomplete diagnostics retain fail-closed precedence.  The raw
    stored assessment is left untouched in the attempt history.
    """
    unrecoverable = _unrecoverable_reading(assessment)
    if unrecoverable is not None:
        return "indeterminate", unrecoverable
    certificate = assessment.get("fourier_weyl_ghost_lower_bound")
    if (
        isinstance(certificate, dict)
        and certificate.get("status") == "valid_but_below_adequacy_gate"
        and _weak_bound_override_eligible(assessment)
        and bool(assessment.get("supports_consistent"))
        and not bool(assessment.get("origin_enclosed"))
    ):
        return "provisional", "certification not established by the methods attempted"
    return str(assessment["verdict"]), assessment.get("verdict_reason")


def _policy_category(assessment: dict[str, Any]) -> str:
    """Return the final category for one stored, resolved assessment."""
    return _policy_outcome(assessment)[0]


def _normalise_resolution(resolution: dict[str, Any]) -> dict[str, Any]:
    """Normalize one terminal resolution to its published policy fields."""
    if resolution["status"] == "UNCERTIFIED_AT_CAP":
        resolution["final_category"] = "uncertified_at_cap"
        resolution["final_verdict"] = "uncertified_at_cap"
        resolution["final_verdict_reason"] = resolution["uncertified_reason"]
        return resolution
    category, reason = _policy_outcome(resolution["final"]["assessment"])
    resolution["final_category"] = category
    resolution["final_verdict"] = category
    resolution["final_verdict_reason"] = reason
    return resolution


def _resolve_record(
    record: dict[str, Any],
    *,
    source_state_cache_dir: str,
    original_source_state_cache_dir: str | None = None,
) -> dict[str, Any]:
    """Run the bounded ladder without ever re-solving the source trajectory."""
    attempts: list[dict[str, Any]] = []
    for budget in FOV_SUPPORT_LADDER:
        n_angles, fov_max_iters, fov_n_restarts = budget
        assessment = reassess_brusselator_record(
            record,
            source_state_cache_dir=source_state_cache_dir,
            original_source_state_cache_dir=original_source_state_cache_dir,
            n_angles=n_angles,
            fov_max_iters=fov_max_iters,
            fov_n_restarts=fov_n_restarts,
        )
        attempt = _attempt(assessment, budget)
        attempts.append(attempt)
        if attempt["supports_consistent"]:
            return _normalise_resolution({
                "status": "RESOLVED",
                "source": "fov_support_escalation",
                "attempts": attempts,
                "final": attempt,
            })
    return _normalise_resolution({
        "status": "UNCERTIFIED_AT_CAP",
        "source": "fov_support_escalation",
        "attempts": attempts,
        "final": attempts[-1],
        "uncertified_reason": _cap_reason(attempts[-1]),
    })


def _resolve_one(checkpoint_dir: Path, study: str, key: str) -> None:
    """Resolve one record and atomically persist its complete ladder."""
    base_path, report = _load_base(checkpoint_dir, study)
    records = {_record_key(record): record for record in report["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"unknown {study} record: {key}") from error
    if bool(record.get("supports_consistent")):
        raise RuntimeError(f"record is already resolved at the base budget: {key}")
    checkpoint = _load_resolution(checkpoint_dir, study, base_path)
    if key in checkpoint["resolved"]:
        print(f"RESOLUTION_CHECKPOINT_EXISTS study={study} key={key}", flush=True)
        return
    resolution = _resolve_record(
        record,
        source_state_cache_dir=str(checkpoint_dir / "source_states"),
        original_source_state_cache_dir=_original_cache_root(report),
    )
    checkpoint["resolved"][key] = resolution
    _atomic_json(_resolution_path(checkpoint_dir, study), checkpoint)
    final = resolution["final"]
    print(
        "RESOLUTION_COMPLETE "
        f"study={study} key={key} category={resolution['final_category']} "
        f"budget={final['budget']['n_angles']}/{final['budget']['fov_max_iters']}/"
        f"{final['budget']['fov_n_restarts']} "
        f"checkpoint={_resolution_path(checkpoint_dir, study)}",
        flush=True,
    )


def _base_attempt(record: dict[str, Any]) -> dict[str, Any]:
    """Represent the raw reading as the first, explicitly nonterminal attempt."""
    assessment = {
        key: deepcopy(value)
        for key, value in record.items()
        if key
        in {
            "status",
            "verdict",
            "disk_rate",
            "epsilon_zero",
            "reduced_arnoldi_epsilon_zero",
            "epsilon_zero_full_operator_evidence",
            "predicted_gmres_factor",
            "origin_enclosed",
            "n_right_real_outliers",
            "supports_consistent",
            "corroboration_attempted",
            "verdict_reason",
            "fov_imaginary_extent",
            "rates",
            "lobpcg_sigma_min_upper_estimate",
            "fourier_weyl_ghost_lower_bound",
        }
    }
    return {
        "budget": {
            "n_angles": int(record["record_config"]["n_angles"]),
            "fov_max_iters": int(record["record_config"]["fov_max_iters"]),
            "fov_n_restarts": int(record["record_config"]["fov_n_restarts"]),
        },
        "assessment": assessment,
        "supports_consistent": bool(record["supports_consistent"]),
        "supports_converged": None,
        "supports_corroborated": None,
        "corroboration_attempted": bool(record["corroboration_attempted"]),
    }


def _original_cache_root(report: dict[str, Any]) -> str | None:
    """Return the source-state cache root a base report was generated with."""
    config = report.get("config", {})
    if config.get("source_state_cache_dir") is not None:
        return str(config["source_state_cache_dir"])
    if config.get("output_path") is not None:
        return str(Path(config["output_path"]).parent / "brusselator_source_states")
    return None


def _cache_relative_artifact(
    artifact: dict[str, Any], original_cache_root: str | None
) -> dict[str, Any]:
    """Normalize one legacy artifact path to a safe cache-root-relative path."""
    try:
        fingerprint = artifact["generation_fingerprint"]
        regime_name = fingerprint["regime"]["name"]
    except (KeyError, TypeError) as error:
        raise RuntimeError("record source-state artifact lacks its generation contract") from error
    return _cache_relative_source_artifact(
        deepcopy(artifact),
        str(regime_name),
        fingerprint,
        original_cache_root=original_cache_root,
    )


def _hopf_vs_turing(
    records: list[dict[str, Any]], study: str, *, scope_caveat: str | None = None
) -> dict[str, Any]:
    """Derive the mode-specific Hopf/Turing summary from final policy records."""
    by_regime = {
        regime: [
            record
            for record in records
            if record["regime"] == regime and record["preconditioner"] == "fft_diffusion"
        ]
        for regime in ("hopf", "turing")
    }
    if study == "screen_64":
        adequate = {
            regime: sum(record["verdict"] == "adequate" for record in rows)
            for regime, rows in by_regime.items()
        }
        both = all(adequate[regime] == len(rows) for regime, rows in by_regime.items())
        return {
            "outcome": "both_adequate_under_fft" if both else "fft_regime_assessments_mixed",
            "statement": (
                "The FFT diffusion preconditioner is assessed adequate for both visited-state "
                "regimes."
                if both
                else "The FFT diffusion preconditioner has mixed final-policy outcomes across "
                "the visited-state regimes."
            ),
            "hopf_adequate_fft_records": adequate["hopf"],
            "turing_adequate_fft_records": adequate["turing"],
        }
    if study != "developed_64":
        raise ValueError(f"Hopf/Turing comparison is not defined for {study}")
    all_indeterminate = {
        regime: bool(rows) and all(record["verdict"] == "indeterminate" for record in rows)
        for regime, rows in by_regime.items()
    }
    both = all(all_indeterminate.values())
    hopf_imaginary = sorted(by_regime["hopf"], key=lambda record: record["trajectory_step"])
    summary = {
        "outcome": (
            "both_regimes_indeterminate_on_developed_states"
            if both
            else "developed_fft_regime_assessments_mixed"
        ),
        "statement": (
            "Both evolved regimes are indeterminate at every sampled FFT-preconditioned state "
            "because their numerical ranges enclose the origin; Hopf still has the larger, "
            "growing imaginary extent."
            if both
            else "The developed FFT-preconditioned regimes have mixed final-policy outcomes; "
            "see the per-regime summaries."
        ),
        "hopf_nonadequate_fft_records": sum(
            record["verdict"] != "adequate" for record in by_regime["hopf"]
        ),
        "turing_nonadequate_fft_records": sum(
            record["verdict"] != "adequate" for record in by_regime["turing"]
        ),
        "hopf_origin_enclosed_any": any(
            bool(record["origin_enclosed"]) for record in by_regime["hopf"]
        ),
        "turing_origin_enclosed_any": any(
            bool(record["origin_enclosed"]) for record in by_regime["turing"]
        ),
        "both_regimes_indeterminate": both,
        "hopf_fov_imaginary_extent_grows_over_samples": (
            hopf_imaginary[-1]["fov_imaginary_extent"]
            > hopf_imaginary[0]["fov_imaginary_extent"]
        ),
        "hopf_fov_imaginary_extent_by_time": [
            {"time": record["time"], "fov_imaginary_extent": record["fov_imaginary_extent"]}
            for record in hopf_imaginary
        ],
    }
    if scope_caveat is not None:
        summary["scope_caveat"] = scope_caveat
    return summary


def _recompute_derived_summaries(
    report: dict[str, Any], records: list[dict[str, Any]], study: str
) -> None:
    """Replace every record-derived base summary with one from final policy records."""
    config = PRESETS[study]
    if study in {"screen_64", "developed_64"}:
        report["regime_comparison"] = {
            regime.name: _summary(
                _records_for(records, regime.name),
                regime,
                include_details=study != "screen_64",
            )
            for regime in (HOPF_REGIME, TURING_REGIME)
        }
        previous_comparison = report.get("hopf_vs_turing", {})
        report["hopf_vs_turing"] = _hopf_vs_turing(
            records,
            study,
            scope_caveat=previous_comparison.get("scope_caveat"),
        )
    else:
        report["fixed_dt_transition"] = _fixed_transition(records, config)


def _attach_resolution(
    checkpoint_dir: Path,
    study: str,
) -> dict[str, Any]:
    """Build an honest final report after every unresolved record has a terminal attempt."""
    base_path, report = _load_base(checkpoint_dir, study)
    checkpoint = _load_resolution(checkpoint_dir, study, base_path)
    final_records: list[dict[str, Any]] = []
    tally = {
        "adequate": 0,
        "provisional": 0,
        "investigate": 0,
        "indeterminate": 0,
        "uncertified_at_cap": 0,
    }
    budget_counts: dict[str, int] = {}
    cap_reasons: dict[str, int] = {}
    for record in report["records"]:
        key = _record_key(record)
        if bool(record["supports_consistent"]):
            resolution = _normalise_resolution({
                "status": "RESOLVED",
                "source": "base",
                "attempts": [_base_attempt(record)],
                "final": _base_attempt(record),
            })
        else:
            try:
                resolution = checkpoint["resolved"][key]
            except KeyError as error:
                raise RuntimeError(f"missing FOV resolution for {study}: {key}") from error
        resolution = _normalise_resolution(deepcopy(resolution))
        final_attempt = resolution["final"]
        final_assessment = final_attempt["assessment"]
        updated = deepcopy(record)
        updated.update(final_assessment)
        final_config = dict(updated["record_config"])
        final_config.update(final_attempt["budget"])
        updated["record_config"] = final_config
        updated["source_state_artifact"] = _cache_relative_artifact(
            updated["source_state_artifact"], _original_cache_root(report)
        )
        updated["geometry_resolution"] = resolution
        updated["final_category"] = resolution["final_category"]
        updated["final_verdict"] = resolution["final_verdict"]
        updated["verdict"] = resolution["final_verdict"]
        updated["verdict_reason"] = resolution["final_verdict_reason"]
        final_records.append(updated)
        category = resolution["final_category"]
        if category not in tally:
            raise RuntimeError(f"unexpected final category: {category}")
        tally[category] += 1
        if category == "uncertified_at_cap":
            reason = resolution["uncertified_reason"]
            cap_reasons[reason] = cap_reasons.get(reason, 0) + 1
        else:
            budget = final_attempt["budget"]
            label = f"{budget['n_angles']}/{budget['fov_max_iters']}/{budget['fov_n_restarts']}"
            budget_counts[label] = budget_counts.get(label, 0) + 1
    final_report = deepcopy(report)
    final_report["records"] = final_records
    _recompute_derived_summaries(final_report, final_records, study)
    final_report["geometry_resolution"] = {
        "description": (
            "Each initially unresolved FOV geometry was reassessed from its persisted, "
            "SHA256-validated v4 source state.  A final category requires consistent "
            "supports; records unresolved at 96/240/2 are UNCERTIFIED_AT_CAP rather "
            "than interpreted as origin-enclosure evidence."
        ),
        "base_budget": {
            "n_angles": BASE_BUDGET[0],
            "fov_max_iters": BASE_BUDGET[1],
            "fov_n_restarts": BASE_BUDGET[2],
        },
        "escalation_ladder": [
            {
                "n_angles": n_angles,
                "fov_max_iters": fov_max_iters,
                "fov_n_restarts": fov_n_restarts,
            }
            for n_angles, fov_max_iters, fov_n_restarts in FOV_SUPPORT_LADDER
        ],
        "final_category_counts": tally,
        "certifying_budget_counts": budget_counts,
        "uncertified_at_cap_reason_counts": cap_reasons,
    }
    return final_report


def _assemble(checkpoint_dir: Path, study: str) -> None:
    """Write one resolved report only after its full checkpoint is complete."""
    report = _attach_resolution(checkpoint_dir, study)
    output = checkpoint_dir / "resolved_results" / OUTPUT_NAMES[study]
    _atomic_json(output, report)
    print(
        f"ASSEMBLED_FOV_RESOLUTION study={study} records={len(report['records'])} "
        f"output={output}",
        flush=True,
    )


def _reclassify_resolved(checkpoint_dir: Path, studies: tuple[str, ...]) -> None:
    """Reapply final-category policy to stored observations without numerics."""
    for study in studies:
        base_path, _ = _load_base(checkpoint_dir, study)
        checkpoint = _load_resolution(checkpoint_dir, study, base_path)
        changed = 0
        for resolution in checkpoint["resolved"].values():
            before = json.dumps(resolution, sort_keys=True)
            _normalise_resolution(resolution)
            if json.dumps(resolution, sort_keys=True) != before:
                changed += 1
        if changed:
            _atomic_json(_resolution_path(checkpoint_dir, study), checkpoint)
        _assemble(checkpoint_dir, study)
        print(
            f"RECLASSIFIED_FOV_RESOLUTION study={study} changed={changed}",
            flush=True,
        )


def _controller(checkpoint_dir: Path, studies: tuple[str, ...]) -> None:
    """Run all unresolved records sequentially in fresh child processes."""
    repository = Path(__file__).resolve().parent.parent
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    for study in studies:
        base_path, report = _load_base(checkpoint_dir, study)
        checkpoint = _load_resolution(checkpoint_dir, study, base_path)
        pending = [
            _record_key(record)
            for record in report["records"]
            if not bool(record["supports_consistent"])
            and _record_key(record) not in checkpoint["resolved"]
        ]
        print(f"FOV_RESOLUTION_START study={study} pending={len(pending)}", flush=True)
        for index, key in enumerate(pending, start=1):
            command = (
                sys.executable,
                str(Path(__file__).resolve()),
                "--checkpoint-dir",
                str(checkpoint_dir),
                "--resolve-one",
                study,
                key,
            )
            print(
                f"FOV_RESOLUTION_RECORD study={study} index={index} total={len(pending)} key={key}",
                flush=True,
            )
            subprocess.run(command, cwd=repository, env=environment, check=True)
        _assemble(checkpoint_dir, study)
    print("BRUSSELATOR_FOV_SUPPORT_RESOLUTION_COMPLETE", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    parser.add_argument("--resolve-one", nargs=2, metavar=("STUDY", "RECORD_KEY"))
    parser.add_argument("--reclassify-resolved", action="store_true")
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir.resolve()
    if args.reclassify_resolved:
        if args.resolve_one is not None:
            raise RuntimeError("--reclassify-resolved cannot be combined with --resolve-one")
        _reclassify_resolved(checkpoint_dir, tuple(args.studies))
        return
    if args.resolve_one is not None:
        study, key = args.resolve_one
        _resolve_one(checkpoint_dir, study, key)
        return
    _controller(checkpoint_dir, tuple(args.studies))


if __name__ == "__main__":
    main()
