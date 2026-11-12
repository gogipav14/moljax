"""Render the fixed-state Brusselator ``dt`` sweep results from committed JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks/results"
INPUT = RESULTS_DIR / "brusselator_dt_sweep.json"
OUTPUT = RESULTS_DIR / "BRUSSELATOR_DT_SWEEP_RESULTS.md"
SOURCE = INPUT.name


def _display(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, list):
        return "[" + ", ".join(_display(item) for item in value) + "]"
    return str(value)


def _ref(value: Any, pointer: str) -> str:
    return f"{_display(value)} ⟦`{SOURCE}#{pointer}`⟧"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


def render(payload: dict[str, Any]) -> str:
    analysis = payload["analysis"]
    completeness = analysis["completeness"]
    decision = analysis["decision"]
    diagnostics = analysis["diagnostics"]
    lines = [
        "# Brusselator fixed-state backward-Euler `dt` sweep results",
        "",
        "Every reported value is generated from the committed JSON; the bracketed selector is "
        "its JSON pointer. Wall times are diagnostic only.",
        "",
        "## Completeness and gates",
        "",
        _table(
            ["arm", "completed", "planned"],
            [
                [
                    "main",
                    _ref(completeness["main_completed"], "/analysis/completeness/main_completed"),
                    _ref(completeness["main_planned"], "/analysis/completeness/main_planned"),
                ],
                [
                    "probe",
                    _ref(completeness["probe_completed"], "/analysis/completeness/probe_completed"),
                    _ref(completeness["probe_planned"], "/analysis/completeness/probe_planned"),
                ],
                [
                    "total",
                    _ref(completeness["total_completed"], "/analysis/completeness/total_completed"),
                    _ref(completeness["total_planned"], "/analysis/completeness/total_planned"),
                ],
            ],
        ),
        "",
        f"Remaining cells: {_ref(completeness['remaining_cells'], '/analysis/completeness/remaining_cells')}. "
        f"ETA: {_ref(completeness['eta_seconds'], '/analysis/completeness/eta_seconds')} seconds. "
        f"Truncation: {_ref(completeness['truncation'], '/analysis/completeness/truncation')}.",
        "",
        f"Failed cells: {_ref(len(completeness['failed_cells']), '/analysis/completeness/failed_cells')} "
        f"(array length). Skipped cells: "
        f"{_ref(len(completeness['skipped_cells']), '/analysis/completeness/skipped_cells')} "
        "(array length).",
        "",
        "The source-state loader required a converged source record and verified its v4 "
        "fingerprint plus SHA256 before creating any cell.",
        "",
        "### Non-converged GMRES cells",
        "",
    ]
    nonconverged_rows = []
    for index, row in enumerate(completeness["nonconverged_gmres"]):
        base = f"/analysis/completeness/nonconverged_gmres/{index}"
        nonconverged_rows.append(
            [
                _ref(row["cell_key"], f"{base}/cell_key"),
                _ref(row["iterations"], f"{base}/iterations"),
                _ref(row["final_relative_residual"], f"{base}/final_relative_residual"),
            ]
        )
    lines.extend(
        [_table(["cell", "iterations", "final true relative residual"], nonconverged_rows), ""]
    )

    lines.extend(["### FoV unresolved at the top rung", ""])
    unresolved_rows = []
    for index, row in enumerate(completeness["fov_unresolved_at_cap"]):
        base = f"/analysis/completeness/fov_unresolved_at_cap/{index}"
        unresolved_rows.append(
            [
                _ref(row["cell_key"], f"{base}/cell_key"),
                _ref(row["n_angles"], f"{base}/n_angles"),
                _ref(row["fov_max_iters"], f"{base}/fov_max_iters"),
                _ref(row["n_restarts"], f"{base}/n_restarts"),
            ]
        )
    lines.extend([_table(["cell", "angles", "max iterations", "restarts"], unresolved_rows), ""])

    lines.extend(["## Pre-registered decision", "", "### D1 — material degradation", ""])
    d1 = decision["d1_material_degradation"]
    d1_rows = []
    for index, row in enumerate(d1["states"]):
        base = f"/analysis/decision/d1_material_degradation/states/{index}"
        d1_rows.append(
            [
                _ref(row["state"], f"{base}/state"),
                _ref(row["low_dt_iterations"], f"{base}/low_dt_iterations"),
                _ref(row["low_dt_median_iterations"], f"{base}/low_dt_median_iterations"),
                _ref(row["dt1_iterations"], f"{base}/dt1_iterations"),
                _ref(row["ratio"], f"{base}/ratio"),
                _ref(row["gate_valid"], f"{base}/gate_valid"),
                _ref(row["passes_ratio_threshold"], f"{base}/passes_ratio_threshold"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "state",
                    "iterations at low dt",
                    "low-dt median",
                    "iterations at dt=1",
                    "ratio",
                    "valid",
                    "passes",
                ],
                d1_rows,
            ),
            "",
            f"Median ratio: {_ref(d1['median_ratio'], '/analysis/decision/d1_material_degradation/median_ratio')}; "
            f"threshold: {_ref(d1['ratio_threshold'], '/analysis/decision/d1_material_degradation/ratio_threshold')}. "
            f"States passing: {_ref(d1['states_passing'], '/analysis/decision/d1_material_degradation/states_passing')}; "
            f"minimum: {_ref(d1['minimum_states_passing'], '/analysis/decision/d1_material_degradation/minimum_states_passing')}. "
            f"D1 passes: {_ref(d1['passes'], '/analysis/decision/d1_material_degradation/passes')}.",
            "",
            "### D2 — absolute cost",
            "",
        ]
    )
    d2 = decision["d2_absolute_cost"]
    lines.extend(
        [
            f"Developed FFT iterations at `dt=1`: {_ref(d2['iterations'], '/analysis/decision/d2_absolute_cost/iterations')}. "
            f"Median: {_ref(d2['median_iterations'], '/analysis/decision/d2_absolute_cost/median_iterations')}; "
            f"threshold: {_ref(d2['threshold'], '/analysis/decision/d2_absolute_cost/threshold')}. "
            f"D2 passes: {_ref(d2['passes'], '/analysis/decision/d2_absolute_cost/passes')}.",
            "",
            "### D3 — recoverability",
            "",
        ]
    )
    d3_rows = []
    for index, probe in enumerate(decision["d3_recoverability"]["probes"]):
        base = f"/analysis/decision/d3_recoverability/probes/{index}"
        d3_rows.append(
            [
                _ref(probe["preconditioner"], f"{base}/preconditioner"),
                _ref(probe["probe_median_iterations"], f"{base}/probe_median_iterations"),
                _ref(probe["relative_reduction"], f"{base}/relative_reduction"),
                _ref(probe["relative_reduction_threshold"], f"{base}/relative_reduction_threshold"),
                _ref(probe["developed_dt1_all_converged"], f"{base}/developed_dt1_all_converged"),
                _ref(
                    probe["entire_probe_arm_all_converged"],
                    f"{base}/entire_probe_arm_all_converged",
                ),
                _ref(probe["lower_disk_rate_states"], f"{base}/lower_disk_rate_states"),
                _ref(
                    probe["lower_disk_rate_states_threshold"],
                    f"{base}/lower_disk_rate_states_threshold",
                ),
                _ref(probe["passes"], f"{base}/passes"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "probe",
                    "median iterations",
                    "reduction",
                    "required reduction",
                    "developed dt=1 all converged",
                    "entire arm all converged",
                    "lower-disk states",
                    "required states",
                    "passes",
                ],
                d3_rows,
            ),
            "",
        ]
    )
    for probe_index, probe in enumerate(decision["d3_recoverability"]["probes"]):
        lines.extend([f"#### {_display(probe['preconditioner'])}", ""])
        rows = []
        for state_index, row in enumerate(probe["states"]):
            base = f"/analysis/decision/d3_recoverability/probes/{probe_index}/states/{state_index}"
            rows.append(
                [
                    _ref(row["state"], f"{base}/state"),
                    _ref(row["fft_iterations"], f"{base}/fft_iterations"),
                    _ref(row["probe_iterations"], f"{base}/probe_iterations"),
                    _ref(row["probe_converged"], f"{base}/probe_converged"),
                    _ref(row["fft_disk_rate"], f"{base}/fft_disk_rate"),
                    _ref(row["probe_disk_rate"], f"{base}/probe_disk_rate"),
                    _ref(row["probe_disk_rate_lower"], f"{base}/probe_disk_rate_lower"),
                ]
            )
        lines.extend(
            [
                _table(
                    [
                        "state",
                        "FFT iter",
                        "probe iter",
                        "converged",
                        "FFT disk",
                        "probe disk",
                        "disk lower",
                    ],
                    rows,
                ),
                "",
            ]
        )
    lines.extend(
        [
            f"D3 passes: {_ref(decision['d3_recoverability']['passes'], '/analysis/decision/d3_recoverability/passes')}.",
            "",
            f"**DECISION: {_ref(decision['decision'], '/analysis/decision/decision')}**. "
            f"All conditions pass: {_ref(decision['all_conditions_pass'], '/analysis/decision/all_conditions_pass')}.",
            "",
            "## Non-gating diagnostics",
            "",
            "### Identity-arm growth",
            "",
        ]
    )
    identity_rows = []
    for index, row in enumerate(diagnostics["identity_growth"]):
        base = f"/analysis/diagnostics/identity_growth/{index}"
        identity_rows.append(
            [
                _ref(row["state"], f"{base}/state"),
                _ref(row["low_dt_iterations"], f"{base}/low_dt_iterations"),
                _ref(row["dt1_iterations"], f"{base}/dt1_iterations"),
                _ref(row["growth_ratio"], f"{base}/growth_ratio"),
                _ref(row["censored_recorded_ratio"], f"{base}/censored_recorded_ratio"),
                _ref(row["gate_valid"], f"{base}/gate_valid"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "state",
                    "low-dt iterations",
                    "dt=1 iterations",
                    "exact ratio",
                    "recorded/censored ratio",
                    "valid",
                ],
                identity_rows,
            ),
            "",
            "A null exact ratio means at least one contributing identity run did not converge or its FoV did not resolve; the recorded/censored ratio is retained but not interpreted.",
            "",
            "### Within-state monotonicity",
            "",
        ]
    )
    monotonic_rows = []
    for index, row in enumerate(diagnostics["within_state_monotonicity"]):
        base = f"/analysis/diagnostics/within_state_monotonicity/{index}"
        monotonic_rows.append(
            [
                _ref(row["state"], f"{base}/state"),
                _ref(row["preconditioner"], f"{base}/preconditioner"),
                _ref(row["iterations"], f"{base}/iterations"),
                _ref(row["recorded_counts_nondecreasing"], f"{base}/recorded_counts_nondecreasing"),
                _ref(row["geometry_valid"], f"{base}/geometry_valid"),
                _ref(row["interpretable_count"], f"{base}/interpretable_count"),
                _ref(
                    row["interpretable_nondecreasing"],
                    f"{base}/interpretable_nondecreasing",
                ),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "state",
                    "preconditioner",
                    "iterations over dt grid",
                    "recorded monotone",
                    "FoV valid over dt grid",
                    "interpretable cells",
                    "interpretable monotone",
                ],
                monotonic_rows,
            ),
            "",
            "The recorded-count column is descriptive only. The interpreted result excludes every "
            "non-converged GMRES cell and every unresolved-FoV cell; null means fewer than two "
            "cells remain.",
            "",
            "### Cross-state reaction correlation at `dt=1` on the 64-square states",
            "",
        ]
    )
    spearman = diagnostics["cross_state_spearman_64_dt1"]
    spearman_rows = []
    for index, row in enumerate(spearman["rows"]):
        base = f"/analysis/diagnostics/cross_state_spearman_64_dt1/rows/{index}"
        spearman_rows.append(
            [
                _ref(row["state"], f"{base}/state"),
                _ref(row["iterations"], f"{base}/iterations"),
                _ref(row["max_reaction_jacobian_2_norm"], f"{base}/max_reaction_jacobian_2_norm"),
                _ref(row["gate_valid"], f"{base}/gate_valid"),
            ]
        )
    lines.extend(
        [
            _table(
                ["state", "FFT iterations", "max reaction Jacobian 2-norm", "valid"], spearman_rows
            ),
            "",
            f"Valid rows: {_ref(spearman['valid_rows'], '/analysis/diagnostics/cross_state_spearman_64_dt1/valid_rows')}. "
            f"Spearman rho: {_ref(spearman['spearman_rho'], '/analysis/diagnostics/cross_state_spearman_64_dt1/spearman_rho')}.",
            "",
            "### Disk/FoV ordering versus measured iteration ordering",
            "",
            diagnostics["disk_iteration_ordering"]["definition"],
            "",
        ]
    )
    ordering_rows = []
    for index, row in enumerate(diagnostics["disk_iteration_ordering"]["by_comparator"]):
        base = f"/analysis/diagnostics/disk_iteration_ordering/by_comparator/{index}"
        ordering_rows.append(
            [
                _ref(row["preconditioner"], f"{base}/preconditioner"),
                _ref(row["total_pairs"], f"{base}/total_pairs"),
                _ref(row["eligible_pairs"], f"{base}/eligible_pairs"),
                _ref(row["agreeing_pairs"], f"{base}/agreeing_pairs"),
                _ref(row["agreement_rate"], f"{base}/agreement_rate"),
            ]
        )
    lines.extend(
        [
            _table(
                ["candidate vs FFT", "all pairs", "eligible", "agreeing", "rate"], ordering_rows
            ),
            "",
        ]
    )
    pair_rows = []
    for index, row in enumerate(diagnostics["disk_iteration_ordering"]["rows"]):
        base = f"/analysis/diagnostics/disk_iteration_ordering/rows/{index}"
        pair_rows.append(
            [
                _ref(row["candidate_cell_key"], f"{base}/candidate_cell_key"),
                _ref(
                    row["iteration_delta_candidate_minus_fft"],
                    f"{base}/iteration_delta_candidate_minus_fft",
                ),
                _ref(
                    row["disk_rate_delta_candidate_minus_fft"],
                    f"{base}/disk_rate_delta_candidate_minus_fft",
                ),
                _ref(row["eligible"], f"{base}/eligible"),
                _ref(row["agrees"], f"{base}/agrees"),
            ]
        )
    lines.extend(
        [
            _table(
                ["candidate cell", "iteration delta", "disk-rate delta", "eligible", "agrees"],
                pair_rows,
            ),
            "",
        ]
    )

    lines.extend(["### Split-probe determinant diagnostic", ""])
    determinant_rows = []
    for index, row in enumerate(diagnostics["split_minimum_determinants"]):
        base = f"/analysis/diagnostics/split_minimum_determinants/{index}"
        determinant_rows.append(
            [
                _ref(row["cell_key"], f"{base}/cell_key"),
                _ref(
                    row["minimum_absolute_reaction_determinant"],
                    f"{base}/minimum_absolute_reaction_determinant",
                ),
                _ref(row["gmres_converged"], f"{base}/gmres_converged"),
            ]
        )
    lines.extend(
        [_table(["cell", "minimum absolute determinant", "GMRES converged"], determinant_rows), ""]
    )

    timing = diagnostics["timing"]
    lines.extend(
        [
            "### Timing (diagnostic only)",
            "",
            f"Accounted GPU time: {_ref(timing['gpu_seconds_accounted'], '/analysis/diagnostics/timing/gpu_seconds_accounted')} seconds, "
            f"or {_ref(timing['gpu_hours_accounted'], '/analysis/diagnostics/timing/gpu_hours_accounted')} hours. "
            f"Elapsed time: {_ref(timing['elapsed_seconds'], '/analysis/diagnostics/timing/elapsed_seconds')} seconds, "
            f"or {_ref(timing['elapsed_hours'], '/analysis/diagnostics/timing/elapsed_hours')} hours. "
            f"V0: {_ref(timing['v0_wall_seconds'], '/analysis/diagnostics/timing/v0_wall_seconds')} seconds; "
            f"main-cell sum: {_ref(timing['main_cell_wall_seconds'], '/analysis/diagnostics/timing/main_cell_wall_seconds')} seconds; "
            f"probe-cell sum: {_ref(timing['probe_cell_wall_seconds'], '/analysis/diagnostics/timing/probe_cell_wall_seconds')} seconds.",
            "",
            "These wall times include compilation and diagnostic work and are not performance claims.",
            "",
        ]
    )
    return "\n".join(line.rstrip() for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    args.output.write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
