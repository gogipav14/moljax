"""Finalize the pre-registered fixed-state Brusselator ``dt`` sweep.

The input is the atomically checkpointed raw benchmark payload.  This script
does not run JAX or touch a source state; it applies the decision rule recorded
in ``BRUSSELATOR_DT_SWEEP_PROTOCOL.md`` and writes an auditable analysis block
alongside the unchanged raw cells.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "benchmarks/results/brusselator_dt_sweep.json"
DEVELOPED_FAMILY = "developed64"
FFT = "fft_diffusion"
IDENTITY = "identity"
MEAN_FIELD = "mean_field_coupled_fft"
SPLIT = "split_reaction_after_diffusion"
LOW_DT = (0.05, 0.1, 0.2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dt_text(value: float) -> str:
    return f"{value:g}"


def _cell_key(arm: str, state: str, dt: float, preconditioner: str) -> str:
    return f"{arm}|{state}|dt={_dt_text(dt)}|{preconditioner}"


def _median(values: Iterable[float | int]) -> float:
    return float(statistics.median(values))


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * ((start + 1) + end)
        for position in order[start:end]:
            ranks[position] = rank
        start = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return numerator / denominator


def _spearman(left: list[float], right: list[float]) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def _sign(value: float | int) -> int:
    return (value > 0) - (value < 0)


def _cell_is_geometry_valid(cell: dict[str, Any]) -> bool:
    return bool(cell["supports_consistent"] and cell["supports_converged"])


def _decision_analysis(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    developed = sorted(
        {
            cell["state"]
            for cell in cells.values()
            if cell["arm"] == "main" and cell["family"] == DEVELOPED_FAMILY
        }
    )
    d1_rows: list[dict[str, Any]] = []
    for state in developed:
        low_keys = [_cell_key("main", state, dt, FFT) for dt in LOW_DT]
        dt1_key = _cell_key("main", state, 1.0, FFT)
        low_iterations = [cells[key]["gmres"]["iterations"] for key in low_keys]
        low_median = _median(low_iterations)
        dt1_iterations = int(cells[dt1_key]["gmres"]["iterations"])
        gate_valid = all(
            cells[key]["gmres"]["converged"] and _cell_is_geometry_valid(cells[key])
            for key in (*low_keys, dt1_key)
        )
        ratio = dt1_iterations / low_median if gate_valid else None
        d1_rows.append(
            {
                "state": state,
                "low_dt_cell_keys": low_keys,
                "low_dt_iterations": low_iterations,
                "low_dt_median_iterations": low_median,
                "dt1_cell_key": dt1_key,
                "dt1_iterations": dt1_iterations,
                "gate_valid": gate_valid,
                "ratio": ratio,
                "passes_ratio_threshold": bool(ratio is not None and ratio >= 2.0),
            }
        )
    d1_ratios = [row["ratio"] for row in d1_rows if row["ratio"] is not None]
    d1_median = _median(d1_ratios)
    d1_count = sum(row["passes_ratio_threshold"] for row in d1_rows)
    d1 = {
        "states": d1_rows,
        "ratio_threshold": 2.0,
        "minimum_states_passing": 5,
        "median_ratio": d1_median,
        "states_passing": d1_count,
        "all_rows_gate_valid": len(d1_ratios) == len(developed),
        "passes": bool(len(d1_ratios) == len(developed) and d1_median >= 2.0 and d1_count >= 5),
    }

    dt1_fft = [row["dt1_iterations"] for row in d1_rows]
    d2_median = _median(dt1_fft)
    d2 = {
        "cell_keys": [row["dt1_cell_key"] for row in d1_rows],
        "iterations": dt1_fft,
        "median_iterations": d2_median,
        "threshold": 25.0,
        "all_rows_gate_valid": d1["all_rows_gate_valid"],
        "passes": bool(d1["all_rows_gate_valid"] and d2_median >= 25.0),
    }

    d3_probes: list[dict[str, Any]] = []
    for preconditioner in (MEAN_FIELD, SPLIT):
        rows: list[dict[str, Any]] = []
        for state in developed:
            fft_key = _cell_key("main", state, 1.0, FFT)
            probe_key = _cell_key("probe", state, 1.0, preconditioner)
            fft_cell = cells[fft_key]
            probe_cell = cells[probe_key]
            geometry_valid = _cell_is_geometry_valid(fft_cell) and _cell_is_geometry_valid(
                probe_cell
            )
            rows.append(
                {
                    "state": state,
                    "fft_cell_key": fft_key,
                    "probe_cell_key": probe_key,
                    "fft_iterations": int(fft_cell["gmres"]["iterations"]),
                    "probe_iterations": int(probe_cell["gmres"]["iterations"]),
                    "probe_converged": bool(probe_cell["gmres"]["converged"]),
                    "geometry_valid": geometry_valid,
                    "fft_disk_rate": float(fft_cell["disk_rate"]),
                    "probe_disk_rate": float(probe_cell["disk_rate"]),
                    "probe_disk_rate_lower": bool(
                        geometry_valid and probe_cell["disk_rate"] < fft_cell["disk_rate"]
                    ),
                }
            )
        probe_median = _median(row["probe_iterations"] for row in rows)
        reduction = 1.0 - probe_median / d2_median
        developed_all_converged = all(row["probe_converged"] for row in rows)
        arm_cells = [
            cell
            for cell in cells.values()
            if cell["arm"] == "probe" and cell["preconditioner"] == preconditioner
        ]
        entire_arm_all_converged = all(cell["gmres"]["converged"] for cell in arm_cells)
        lower_count = sum(row["probe_disk_rate_lower"] for row in rows)
        all_geometry_valid = all(row["geometry_valid"] for row in rows)
        d3_probes.append(
            {
                "preconditioner": preconditioner,
                "states": rows,
                "fft_median_iterations": d2_median,
                "probe_median_iterations": probe_median,
                "relative_reduction": reduction,
                "relative_reduction_threshold": 0.4,
                "developed_dt1_all_converged": developed_all_converged,
                "entire_probe_arm_all_converged": entire_arm_all_converged,
                "all_geometry_valid": all_geometry_valid,
                "lower_disk_rate_states": lower_count,
                "lower_disk_rate_states_threshold": 4,
                "passes": bool(
                    reduction >= 0.4
                    and developed_all_converged
                    and all_geometry_valid
                    and lower_count >= 4
                ),
            }
        )
    d3 = {
        "probes": d3_probes,
        "passes": any(probe["passes"] for probe in d3_probes),
    }
    overall = bool(d1["passes"] and d2["passes"] and d3["passes"])
    return {
        "d1_material_degradation": d1,
        "d2_absolute_cost": d2,
        "d3_recoverability": d3,
        "all_conditions_pass": overall,
        "decision": "BUILD" if overall else "DO NOT BUILD",
    }


def _completeness(cells: dict[str, dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    main_planned = (
        len(payload["plan"]["states"])
        * len(payload["plan"]["main_dt"])
        * len(payload["plan"]["main_preconditioners"])
    )
    probe_planned = (
        len(payload["plan"]["states"])
        * len(payload["plan"]["probe_dt"])
        * len(payload["plan"]["probe_preconditioners"])
    )
    failed = [key for key, cell in cells.items() if cell["status"] == "failed"]
    skipped = [key for key, cell in cells.items() if cell["status"] == "skipped"]
    nonconverged = [
        {
            "cell_key": key,
            "iterations": int(cell["gmres"]["iterations"]),
            "final_relative_residual": float(cell["gmres"]["final_relative_residual"]),
        }
        for key, cell in sorted(cells.items())
        if not cell["gmres"]["converged"]
    ]
    unresolved = []
    for key, cell in sorted(cells.items()):
        if cell["supports_consistent"]:
            continue
        final_attempt = cell["fov_attempts"][-1]
        unresolved.append(
            {
                "cell_key": key,
                "n_angles": int(final_attempt["n_angles"]),
                "fov_max_iters": int(final_attempt["fov_max_iters"]),
                "n_restarts": int(final_attempt["n_restarts"]),
                "supports_consistent": False,
            }
        )
    return {
        "main_planned": main_planned,
        "main_completed": sum(cell["arm"] == "main" for cell in cells.values()),
        "probe_planned": probe_planned,
        "probe_completed": sum(cell["arm"] == "probe" for cell in cells.values()),
        "total_planned": main_planned + probe_planned,
        "total_completed": len(cells),
        "remaining_cells": main_planned + probe_planned - len(cells),
        "eta_seconds": 0.0,
        "failed_cells": failed,
        "skipped_cells": skipped,
        "nonconverged_gmres": nonconverged,
        "fov_unresolved_at_cap": unresolved,
        "truncation": payload["truncation"],
        "source_state_gate": {
            "all_records_converged_before_loading": True,
            "all_artifacts_fingerprint_and_sha256_verified": True,
            "evidence": "the benchmark raises before a cell is created on either failure",
        },
    }


def _timing(cells: dict[str, dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    elapsed = float(payload["completed_at_unix"] - payload["started_at_unix"])
    accounted = float(payload["gpu_seconds_accounted"])
    return {
        "gpu_seconds_accounted": accounted,
        "gpu_hours_accounted": accounted / 3600.0,
        "elapsed_seconds": elapsed,
        "elapsed_hours": elapsed / 3600.0,
        "v0_wall_seconds": float(payload["v0"]["wall_seconds"]),
        "main_cell_wall_seconds": sum(
            float(cell["wall_seconds"]) for cell in cells.values() if cell["arm"] == "main"
        ),
        "probe_cell_wall_seconds": sum(
            float(cell["wall_seconds"]) for cell in cells.values() if cell["arm"] == "probe"
        ),
        "diagnostic_only": True,
    }


def _identity_growth(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    states = sorted({cell["state"] for cell in cells.values() if cell["arm"] == "main"})
    rows = []
    for state in states:
        low_keys = [_cell_key("main", state, dt, IDENTITY) for dt in LOW_DT]
        dt1_key = _cell_key("main", state, 1.0, IDENTITY)
        keys = [*low_keys, dt1_key]
        low = [int(cells[key]["gmres"]["iterations"]) for key in low_keys]
        low_median = _median(low)
        dt1_iterations = int(cells[dt1_key]["gmres"]["iterations"])
        valid = all(
            cells[key]["gmres"]["converged"] and _cell_is_geometry_valid(cells[key]) for key in keys
        )
        rows.append(
            {
                "state": state,
                "low_dt_cell_keys": low_keys,
                "low_dt_iterations": low,
                "low_dt_median_iterations": low_median,
                "dt1_cell_key": dt1_key,
                "dt1_iterations": dt1_iterations,
                "gate_valid": valid,
                "growth_ratio": dt1_iterations / low_median if valid else None,
                "censored_recorded_ratio": dt1_iterations / low_median,
            }
        )
    return rows


def _monotonicity(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    dt_grid = (0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
    states = sorted({cell["state"] for cell in cells.values() if cell["arm"] == "main"})
    rows = []
    for preconditioner in (FFT, IDENTITY):
        for state in states:
            keys = [_cell_key("main", state, dt, preconditioner) for dt in dt_grid]
            iterations = [int(cells[key]["gmres"]["iterations"]) for key in keys]
            converged = [bool(cells[key]["gmres"]["converged"]) for key in keys]
            geometry_valid = [_cell_is_geometry_valid(cells[key]) for key in keys]
            interpretable = [
                iteration
                for iteration, did_converge, has_geometry in zip(
                    iterations, converged, geometry_valid, strict=True
                )
                if did_converge and has_geometry
            ]
            prefix_length = next(
                (index for index, value in enumerate(converged) if not value), len(converged)
            )
            prefix = iterations[:prefix_length]
            rows.append(
                {
                    "state": state,
                    "preconditioner": preconditioner,
                    "cell_keys": keys,
                    "dt": list(dt_grid),
                    "iterations": iterations,
                    "converged": converged,
                    "geometry_valid": geometry_valid,
                    "recorded_counts_nondecreasing": all(
                        left <= right
                        for left, right in zip(iterations, iterations[1:], strict=False)
                    ),
                    "converged_prefix_length": prefix_length,
                    "converged_prefix_nondecreasing": all(
                        left <= right for left, right in zip(prefix, prefix[1:], strict=False)
                    ),
                    "all_converged": all(converged),
                    "interpretable_count": len(interpretable),
                    "interpretable_nondecreasing": (
                        all(
                            left <= right
                            for left, right in zip(interpretable, interpretable[1:], strict=False)
                        )
                        if len(interpretable) >= 2
                        else None
                    ),
                }
            )
    return rows


def _cross_state_spearman(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    states = sorted(
        {
            cell["state"]
            for cell in cells.values()
            if cell["arm"] == "main" and cell["grid"]["nx"] == 64
        }
    )
    rows = []
    for state in states:
        key = _cell_key("main", state, 1.0, FFT)
        cell = cells[key]
        rows.append(
            {
                "state": state,
                "cell_key": key,
                "iterations": int(cell["gmres"]["iterations"]),
                "max_reaction_jacobian_2_norm": float(
                    cell["reaction"]["max_reaction_jacobian_2_norm"]
                ),
                "gate_valid": bool(cell["gmres"]["converged"] and _cell_is_geometry_valid(cell)),
            }
        )
    valid = [row for row in rows if row["gate_valid"]]
    return {
        "rows": rows,
        "valid_rows": len(valid),
        "spearman_rho": _spearman(
            [float(row["iterations"]) for row in valid],
            [row["max_reaction_jacobian_2_norm"] for row in valid],
        ),
    }


def _ordering_agreement(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for key, candidate in sorted(cells.items()):
        if candidate["preconditioner"] == FFT:
            continue
        baseline_key = _cell_key("main", candidate["state"], candidate["dt"], FFT)
        baseline = cells[baseline_key]
        eligible = bool(
            candidate["gmres"]["converged"]
            and baseline["gmres"]["converged"]
            and _cell_is_geometry_valid(candidate)
            and _cell_is_geometry_valid(baseline)
        )
        iteration_delta = int(candidate["gmres"]["iterations"]) - int(
            baseline["gmres"]["iterations"]
        )
        disk_delta = float(candidate["disk_rate"] - baseline["disk_rate"])
        agrees = bool(eligible and _sign(iteration_delta) == _sign(disk_delta))
        rows.append(
            {
                "candidate_cell_key": key,
                "fft_cell_key": baseline_key,
                "arm": candidate["arm"],
                "preconditioner": candidate["preconditioner"],
                "state": candidate["state"],
                "dt": float(candidate["dt"]),
                "iteration_delta_candidate_minus_fft": iteration_delta,
                "disk_rate_delta_candidate_minus_fft": disk_delta,
                "eligible": eligible,
                "agrees": agrees if eligible else None,
            }
        )
    summaries = []
    for arm, preconditioner in (
        ("main", IDENTITY),
        ("probe", MEAN_FIELD),
        ("probe", SPLIT),
    ):
        selected = [
            row for row in rows if row["arm"] == arm and row["preconditioner"] == preconditioner
        ]
        eligible = [row for row in selected if row["eligible"]]
        count = sum(bool(row["agrees"]) for row in eligible)
        summaries.append(
            {
                "arm": arm,
                "preconditioner": preconditioner,
                "total_pairs": len(selected),
                "eligible_pairs": len(eligible),
                "agreeing_pairs": count,
                "agreement_rate": count / len(eligible) if eligible else None,
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    count = sum(bool(row["agrees"]) for row in eligible)
    return {
        "definition": (
            "candidate-versus-FFT sign agreement: sign(delta iterations) equals "
            "sign(delta disk_rate); pairs with non-converged GMRES or unresolved FoV are excluded"
        ),
        "rows": rows,
        "by_comparator": summaries,
        "overall": {
            "total_pairs": len(rows),
            "eligible_pairs": len(eligible),
            "agreeing_pairs": count,
            "agreement_rate": count / len(eligible) if eligible else None,
        },
    }


def _split_determinants(cells: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "cell_key": key,
            "state": cell["state"],
            "dt": float(cell["dt"]),
            "minimum_absolute_reaction_determinant": float(
                cell["preconditioner_metadata"]["minimum_absolute_reaction_determinant"]
            ),
            "gmres_converged": bool(cell["gmres"]["converged"]),
        }
        for key, cell in sorted(cells.items())
        if cell["preconditioner"] == SPLIT
    ]


def analyze(payload: dict[str, Any], checkpoint: Path, script: Path) -> dict[str, Any]:
    if payload["status"] != "complete":
        raise RuntimeError(f"checkpoint is not complete: {payload['status']}")
    cells = payload["cells"]
    result = dict(payload)
    result["analysis_schema"] = "brusselator_fixed_state_dt_sweep_analysis_v1"
    result["analysis"] = {
        "provenance": {
            "checkpoint_filename": checkpoint.name,
            "checkpoint_sha256": _sha256(checkpoint),
            "analysis_script_sha256": _sha256(script),
            "analysis_uses_gpu": False,
        },
        "completeness": _completeness(cells, payload),
        "decision": _decision_analysis(cells),
        "diagnostics": {
            "identity_growth": _identity_growth(cells),
            "within_state_monotonicity": _monotonicity(cells),
            "cross_state_spearman_64_dt1": _cross_state_spearman(cells),
            "disk_iteration_ordering": _ordering_agreement(cells),
            "split_minimum_determinants": _split_determinants(cells),
            "timing": _timing(cells, payload),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    result = analyze(payload, args.checkpoint, Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
