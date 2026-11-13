#!/usr/bin/env python3
"""Render the corrected tau-blend baseline facts from committed JSONs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / "benchmarks" / "results"
OUTPUT = RESULTS / "TAU_BLEND_BASELINE_FACTS.md"

SOURCES = {
    "batch_cpu": "pme_dst_tau_blend_batching_audit_cpu.json",
    "batch_gpu": "pme_dst_tau_blend_batching_audit_gpu.json",
    "inexact": "pme_dst_tau_blend_inexact_newton.json",
    "poc": "pme_dst_tau_blend_poc.json",
    "criterion": "pme_dst_tau_blend_pseudospectral_criterion.json",
    "runtime": "pme_dst_tau_blend_pseudospectral_runtime_estimator.json",
    "baseline": "pme_dst_tau_blend_single_reference_baselines.json",
    "spectral": "pme_dst_tau_blend_spectral_analysis.json",
    "stage1": "pme_dst_tau_blend_stage1_validation.json",
    "validation": "pme_dst_tau_blend_validation.json",
}

SCALAR_METHODS = {
    "const",
    "floor",
    "frozen_bulk",
    "frozen_mean",
    "geometric_mean",
    "harmonic_mean",
    "optimized_d0",
}


@dataclass(frozen=True, order=True)
class JsonReference:
    """One committed file and RFC 6901 pointer."""

    source: str
    pointer: str


def _pointer(document: Any, pointer: str) -> Any:
    value = document
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError(f"invalid JSON pointer: {pointer!r}")
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8g}"
    if isinstance(value, list):
        return ", ".join(_display(item) for item in value)
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


class Facts:
    """Loaded evidence and the paths emitted while rendering."""

    def __init__(self, results_dir: Path = RESULTS) -> None:
        self.documents = {
            key: json.loads((results_dir / filename).read_text(encoding="utf-8"))
            for key, filename in SOURCES.items()
        }
        self.references: set[JsonReference] = set()

    def cite(self, source: str, pointer: str, value: Any | None = None) -> str:
        actual = _pointer(self.documents[source], pointer)
        self.references.add(JsonReference(source, pointer))
        shown = actual if value is None else value
        return f"{_display(shown)} ⟦`{SOURCES[source]}#{pointer}`⟧"

    def derived(self, value: Any, source: str, pointer: str, operation: str) -> str:
        _pointer(self.documents[source], pointer)
        self.references.add(JsonReference(source, pointer))
        return f"{_display(value)} ⟦`{SOURCES[source]}#{pointer}`; derived: {operation}⟧"

    def validate(self) -> None:
        for reference in self.references:
            _pointer(self.documents[reference.source], reference.pointer)


def _provenance_rows(facts: Facts) -> list[list[str]]:
    rows = []
    for source, filename in SOURCES.items():
        document = facts.documents[source]
        backend = "not recorded"
        platform = "not recorded"
        if "backend" in document:
            backend = facts.cite(source, "/backend")
            platform = facts.cite(source, "/devices")
        elif "device" in document:
            backend = facts.cite(source, "/device/default_backend")
            platform = facts.cite(source, "/device/devices")
        rows.append(
            [
                filename,
                facts.cite(source, "/provenance/repository_head"),
                facts.cite(source, "/provenance/base_revision"),
                "not recorded",
                platform,
                backend,
                "not recorded",
            ]
        )
    return rows


def _spectral_rows(facts: Facts) -> list[list[str]]:
    baseline = facts.documents["baseline"]
    epsilon = baseline["config"]["epsilon"]
    rows = []
    for index, record in enumerate(baseline["spectral_records"]):
        root = f"/spectral_records/{index}"
        d0 = record["d0"]
        floor = record["m"] * epsilon ** (record["m"] - 1)
        active_maximum = record["coefficient"]["active_maximum"]
        rows.append(
            [
                facts.cite("baseline", f"{root}/m"),
                facts.cite("baseline", f"{root}/method"),
                facts.cite("baseline", f"{root}/d0"),
                (
                    "—"
                    if d0 is None
                    else facts.derived(d0 / floor, "baseline", root, "d0 / (m * epsilon**(m-1))")
                ),
                (
                    "—"
                    if d0 is None
                    else facts.derived(d0 / active_maximum, "baseline", root, "d0 / active_maximum")
                ),
                facts.cite("baseline", f"{root}/spectrum/condition_number_2"),
                facts.cite("baseline", f"{root}/spectrum/near_zero_eigenvalue_count"),
                facts.cite("baseline", f"{root}/spectrum/minimum_eigenvalue_modulus"),
                facts.cite("baseline", f"{root}/spectrum/spectral_abscissa"),
                facts.cite("baseline", f"{root}/spectrum/numerical_abscissa"),
                facts.cite("baseline", f"{root}/spectrum/non_normality_gap"),
                facts.cite("baseline", f"{root}/field_of_values/disk_rate"),
            ]
        )
    return rows


def _work_rows(facts: Facts) -> list[list[str]]:
    rows = []
    for index, record in enumerate(facts.documents["baseline"]["work_precision_records"]):
        root = f"/work_precision_records/{index}"
        for method in sorted(record["methods"]):
            method_root = f"{root}/methods/{method}"
            rows.append(
                [
                    facts.cite("baseline", f"{root}/nx"),
                    facts.cite("baseline", f"{root}/m"),
                    facts.cite("baseline", f"{root}/requested_relative_residual"),
                    method,
                    facts.cite("baseline", f"{method_root}/iterations"),
                    facts.cite("baseline", f"{method_root}/median_seconds"),
                    facts.cite("baseline", f"{method_root}/iqr_seconds"),
                    facts.cite("baseline", f"{method_root}/all_converged"),
                ]
            )
    return rows


def _tight_rows(facts: Facts) -> list[list[str]]:
    rows = []
    records = facts.documents["baseline"]["work_precision_records"]
    for index, record in enumerate(records):
        if record["requested_relative_residual"] != 1e-8:
            continue
        root = f"/work_precision_records/{index}"
        methods = record["methods"]
        blend = methods["tau_blend"]
        oracle = methods["optimized_d0"]
        comparators = {
            method: result
            for method, result in methods.items()
            if method != "tau_blend" and result["all_converged"]
        }
        fastest = (
            min(comparators.items(), key=lambda item: item[1]["median_seconds"])[1]
            if comparators
            else None
        )
        fastest_name = (
            min(comparators, key=lambda method: comparators[method]["median_seconds"])
            if comparators
            else "none"
        )
        rows.append(
            [
                facts.cite("baseline", f"{root}/nx"),
                facts.cite("baseline", f"{root}/m"),
                facts.cite("baseline", f"{root}/requested_relative_residual"),
                facts.cite("baseline", f"{root}/methods/tau_blend/iterations"),
                facts.cite("baseline", f"{root}/methods/tau_blend/median_seconds"),
                facts.cite("baseline", f"{root}/methods/optimized_d0/all_converged"),
                facts.cite("baseline", f"{root}/methods/optimized_d0/iterations"),
                facts.cite("baseline", f"{root}/methods/optimized_d0/median_seconds"),
                (
                    facts.derived(
                        oracle["median_seconds"] / blend["median_seconds"],
                        "baseline",
                        root,
                        "oracle median_seconds / blend median_seconds",
                    )
                    if oracle["all_converged"]
                    else "—"
                ),
                fastest_name,
                (
                    facts.cite("baseline", f"{root}/methods/{fastest_name}/iterations")
                    if fastest is not None
                    else "—"
                ),
                (
                    facts.derived(
                        fastest["median_seconds"] / blend["median_seconds"],
                        "baseline",
                        root,
                        "fastest converged comparator median_seconds / blend median_seconds",
                    )
                    if fastest is not None
                    else "—"
                ),
            ]
        )
    return rows


def _loose_rows(facts: Facts) -> list[list[str]]:
    rows = []
    records = facts.documents["baseline"]["work_precision_records"]
    for index, record in enumerate(records):
        if record["requested_relative_residual"] != 1e-2:
            continue
        root = f"/work_precision_records/{index}"
        blend = record["methods"]["tau_blend"]
        scalars = {
            method: result
            for method, result in record["methods"].items()
            if method in SCALAR_METHODS and result["all_converged"]
        }
        fastest_name, fastest = min(scalars.items(), key=lambda item: item[1]["median_seconds"])
        signed_percent = 100.0 * (blend["median_seconds"] / fastest["median_seconds"] - 1.0)
        rows.append(
            [
                facts.cite("baseline", f"{root}/nx"),
                facts.cite("baseline", f"{root}/m"),
                fastest_name,
                facts.cite("baseline", f"{root}/methods/{fastest_name}/iterations"),
                facts.cite("baseline", f"{root}/methods/{fastest_name}/median_seconds"),
                facts.cite("baseline", f"{root}/methods/tau_blend/iterations"),
                facts.cite("baseline", f"{root}/methods/tau_blend/median_seconds"),
                facts.derived(
                    signed_percent,
                    "baseline",
                    root,
                    "100 * (blend median / fastest scalar median - 1)",
                ),
            ]
        )
    return rows


def _contrast_rows(facts: Facts) -> list[list[str]]:
    rows = []
    for index, record in enumerate(facts.documents["baseline"]["contrast_records"]):
        root = f"/contrast_records/{index}"
        for method in sorted(record["methods"]):
            method_root = f"{root}/methods/{method}"
            rows.append(
                [
                    facts.cite("baseline", f"{root}/m"),
                    facts.cite("baseline", f"{root}/step"),
                    facts.cite("baseline", f"{root}/time"),
                    facts.cite("baseline", f"{root}/coefficient/active_d95_over_d05"),
                    facts.cite("baseline", f"{root}/coefficient/degeneracy_fraction"),
                    method,
                    facts.cite("baseline", f"{method_root}/iterations"),
                    facts.cite("baseline", f"{method_root}/converged"),
                    facts.cite("baseline", f"{root}/optimized_d0_scan/ratio_to_active_maximum"),
                    facts.cite("baseline", f"{root}/optimized_d0_scan/ratio_to_active_minimum"),
                ]
            )
    return rows


def _batching_sections(facts: Facts) -> list[str]:
    lines = []
    for source in ("batch_cpu", "batch_gpu"):
        document = facts.documents[source]
        lines.extend(
            [
                f"### {document['backend'].upper()}",
                "",
                f"Maximum relative action difference: {facts.cite(source, '/maximum_relative_action_difference')}.",
                f"All solve iteration sets identical: {facts.cite(source, '/all_solve_iteration_sets_identical')}.",
                "",
            ]
        )
        apply_rows = []
        for index, _record in enumerate(document["apply_records"]):
            root = f"/apply_records/{index}"
            apply_rows.append(
                [
                    facts.cite(source, f"{root}/nx"),
                    facts.cite(source, f"{root}/m"),
                    facts.cite(source, f"{root}/equivalence/maximum_relative_l2"),
                    facts.cite(
                        source,
                        f"{root}/timing/speedup_batched_over_forced_sequential",
                    ),
                ]
            )
        lines.extend(
            [
                _table(["N", "m", "action relative difference", "apply speed-up"], apply_rows),
                "",
            ]
        )
        solve_rows = []
        mismatch_rows = []
        for index, record in enumerate(document["solve_records"]):
            root = f"/solve_records/{index}"
            row = [
                facts.cite(source, f"{root}/nx"),
                facts.cite(source, f"{root}/m"),
                facts.cite(source, f"{root}/requested_relative_residual"),
                facts.cite(source, f"{root}/phase0_batched/iterations"),
                facts.cite(source, f"{root}/forced_sequential/iterations"),
                facts.cite(source, f"{root}/iteration_sets_identical"),
                facts.cite(source, f"{root}/full_solve_speedup_batched_over_forced_sequential"),
            ]
            solve_rows.append(row)
            if not record["iteration_sets_identical"]:
                mismatch_rows.append(row)
        if solve_rows:
            lines.extend(
                [
                    _table(
                        [
                            "N",
                            "m",
                            "tolerance",
                            "batched iterations",
                            "sequential iterations",
                            "identical",
                            "end-to-end speed-up",
                        ],
                        solve_rows,
                    ),
                    "",
                    "Iteration-count mismatches:",
                    "",
                    (
                        _table(
                            [
                                "N",
                                "m",
                                "tolerance",
                                "batched iterations",
                                "sequential iterations",
                                "identical",
                                "end-to-end speed-up",
                            ],
                            mismatch_rows,
                        )
                        if mismatch_rows
                        else "None."
                    ),
                    "",
                ]
            )
        else:
            lines.extend(["No solve records are stored for this backend.", ""])
    return lines


def _criterion_rows(facts: Facts) -> tuple[list[list[str]], list[list[str]]]:
    criterion_rows = []
    for index, _record in enumerate(facts.documents["criterion"]["records"]):
        root = f"/records/{index}"
        criterion_rows.append(
            [
                facts.cite("criterion", f"{root}/m"),
                facts.cite("criterion", f"{root}/pseudospectral_criterion/certified"),
                facts.cite(
                    "criterion", f"{root}/pseudospectral_criterion/polynomial_effective_rate"
                ),
                facts.cite("criterion", f"{root}/pseudospectral_criterion/predicted_iterations"),
                facts.cite("criterion", f"{root}/actual_counted_gmres/iterations"),
            ]
        )
    runtime_rows = []
    for index, _record in enumerate(facts.documents["runtime"]["records"]):
        root = f"/records/{index}"
        runtime_rows.append(
            [
                facts.cite("runtime", f"{root}/m"),
                facts.cite("runtime", f"{root}/method"),
                facts.cite("runtime", f"{root}/cost/runtime_estimate_time_over_gmres"),
                facts.cite(
                    "runtime",
                    f"{root}/fully_matrix_free_runtime_estimate/verdict",
                ),
                facts.cite(
                    "runtime",
                    f"{root}/fully_matrix_free_runtime_estimate/rate_bound_available",
                ),
                facts.cite(
                    "runtime",
                    f"{root}/exact_spectrum_sparse_paths/verdict",
                ),
                facts.cite(
                    "runtime",
                    f"{root}/exact_spectrum_sparse_paths/rate_bound_available",
                ),
            ]
        )
    return criterion_rows, runtime_rows


def render_facts(results_dir: Path = RESULTS) -> tuple[str, tuple[JsonReference, ...]]:
    """Render the facts sheet and return all emitted references."""
    facts = Facts(results_dir)
    criterion_rows, runtime_rows = _criterion_rows(facts)
    lines = [
        "# Tau-blend corrected single-reference baseline facts",
        "",
        "This file is generated only from the ten JSONs regenerated by commit `78925c5`. "
        "Every displayed measurement has a source file and JSON pointer; aggregates name "
        "their collection and operation.",
        "",
        "## P1 — provenance recorded by each JSON",
        "",
        _table(
            ["JSON", "commit", "base", "host", "platform/device", "backend", "versions"],
            _provenance_rows(facts),
        ),
        "",
        "Host names and Python, JAX, jaxlib, NumPy and SciPy versions are not recorded in "
        "these ten artifacts. Platform/backend fields are shown only where present.",
        "",
        "## P2 — dense spectral table",
        "",
        _table(
            [
                "m",
                "method",
                "d",
                "d/floor",
                "d/active max",
                "kappa_2",
                "count abs(lambda)<0.1",
                "min abs(lambda)",
                "spectral abscissa",
                "numerical abscissa",
                "gap",
                "disk rate",
            ],
            _spectral_rows(facts),
        ),
        "",
        "## P3 — work precision",
        "",
        _table(
            ["N", "m", "tolerance", "method", "iterations", "median s", "IQR s", "converged"],
            _work_rows(facts),
        ),
        "",
        "## P4 — tight-tolerance comparisons",
        "",
        _table(
            [
                "N",
                "m",
                "tolerance",
                "blend iterations",
                "blend median s",
                "oracle converged",
                "oracle iterations",
                "oracle median s",
                "oracle/blend time",
                "fastest comparator",
                "comparator iterations",
                "comparator/blend time",
            ],
            _tight_rows(facts),
        ),
        "",
        "## P5 — loose-tolerance comparisons",
        "",
        "Positive percentages mean that the blend is slower than the fastest converged "
        "scalar reference.",
        "",
        _table(
            [
                "N",
                "m",
                "fastest scalar",
                "scalar iterations",
                "scalar median s",
                "blend iterations",
                "blend median s",
                "signed %",
            ],
            _loose_rows(facts),
        ),
        "",
        "## P6 — contrast sweep and oracle location",
        "",
        _table(
            [
                "m",
                "step",
                "time",
                "D95/D05",
                "degeneracy fraction",
                "method",
                "iterations",
                "converged",
                "oracle/active max",
                "oracle/floor",
            ],
            _contrast_rows(facts),
        ),
        "",
        "## P7 — batching audit",
        "",
        *_batching_sections(facts),
        "## P8 — pseudospectral criterion and runtime estimator",
        "",
        "Dense criterion:",
        "",
        _table(
            ["m", "certified", "certified rate", "predicted iterations", "actual GMRES"],
            criterion_rows,
        ),
        "",
        "Provisional sparse runtime estimator:",
        "",
        _table(
            [
                "m",
                "method",
                "runtime/GMRES",
                "matrix-free verdict",
                "matrix-free rate available",
                "exact-spectrum verdict",
                "exact-spectrum rate available",
            ],
            runtime_rows,
        ),
        "",
        "## P9 — remaining regenerated artifacts",
        "",
        f"- Inexact Newton: {facts.cite('inexact', '/scope')}",
        f"- Validation: {facts.cite('validation', '/scope')}",
        f"- Stage-1 validation: {facts.cite('stage1', '/scope')}",
        f"- Proof of concept: {facts.cite('poc', '/scope')}",
        "",
    ]
    text = "\n".join(lines)
    facts.validate()
    return text, tuple(sorted(facts.references))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    text, _references = render_facts()
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
