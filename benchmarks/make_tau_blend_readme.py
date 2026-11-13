#!/usr/bin/env python3
"""Render the tau-blend mask-analysis report from committed JSON evidence."""

from __future__ import annotations

import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import Any

try:
    from benchmarks.tau_blend_mask_analysis import (
        REPOSITORY_ACTIVE_ABSOLUTE,
        REPOSITORY_ACTIVE_RELATIVE,
    )
except ModuleNotFoundError:  # Direct ``python benchmarks/...`` execution.
    from tau_blend_mask_analysis import REPOSITORY_ACTIVE_ABSOLUTE, REPOSITORY_ACTIVE_RELATIVE
from moljax.experimental.pseudospectral_criterion import assess_dense_pseudospectral_criterion

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
OUTPUT = RESULTS_DIR / "TAU_BLEND_README.md"

FILES = {
    "active": "tau_blend_active_block.json",
    "cfl": "tau_blend_cfl_fidelity.json",
    "cn": "tau_blend_cn_replay.json",
    "lemma": "tau_blend_lemma_checks.json",
    "mask": "tau_blend_mask_analysis.json",
    "reference": "tau_blend_reference_count.json",
    "repro": "tau_blend_reproducibility.json",
    "runtime": "pme_dst_tau_blend_pseudospectral_runtime_estimator.json",
    "baseline": "pme_dst_tau_blend_single_reference_baselines.json",
    "batch_cpu": "pme_dst_tau_blend_batching_audit_cpu.json",
    "batch_gpu": "pme_dst_tau_blend_batching_audit_gpu.json",
    "criterion_legacy": "pme_dst_tau_blend_pseudospectral_criterion.json",
    "criterion_current": "pme_dst_tau_blend_pseudospectral_criterion_current.json",
}

PART_II_METHODS = (
    "identity",
    "frozen_mean",
    "frozen_bulk",
    "floor",
    "const",
    "geometric_mean",
    "harmonic_mean",
    "optimized_d0",
    "tau_blend",
)
PART_II_LABELS = {
    "identity": "identity",
    "frozen_mean": "mean",
    "frozen_bulk": "bulk",
    "floor": "floor",
    "const": "const",
    "geometric_mean": "geometric",
    "harmonic_mean": "harmonic",
    "optimized_d0": "oracle",
    "tau_blend": "blend",
}


def _load(results_dir: Path) -> dict[str, dict[str, Any]]:
    return {
        name: json.loads((results_dir / filename).read_text(encoding="utf-8"))
        for name, filename in FILES.items()
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.8g}".replace("e-0", "e-").replace("e+0", "e+")
    return str(value)


def _sig(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}".replace("e-0", "e-").replace("e+0", "e+")


def _ratio_column(value: float | None) -> str:
    if value is None:
        return "--"
    if value != 0.0 and (abs(value) >= 1.0e4 or abs(value) < 1.0e-2):
        return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.2f}"


def _sci(value: float, significant_digits: int) -> str:
    mantissa, exponent = f"{value:.{significant_digits - 1}e}".split("e")
    return f"{mantissa.rstrip('0').rstrip('.')}e{int(exponent)}"


def _table(
    headers: tuple[str, ...],
    rows: list[tuple[Any, ...]],
    *,
    numeric_columns: frozenset[int] = frozenset(),
) -> str:
    numeric = [
        index in numeric_columns
        or all(value is None or isinstance(value, (int, float)) for value in column)
        for index, column in enumerate(zip(*rows, strict=True))
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---:" if is_numeric else "---" for is_numeric in numeric) + "|",
            *("| " + " | ".join(_fmt(value) for value in row) + " |" for row in rows),
        ]
    )


def _wrap(text: str, *, initial_indent: str = "", subsequent_indent: str = "") -> str:
    """Wrap one prose paragraph after interpolation."""
    protected = re.sub(
        r"`[^`]+`",
        lambda match: match.group(0).replace(" ", "\0"),
        " ".join(text.split()).replace("( ", "("),
    )
    wrapped = textwrap.fill(
        protected,
        width=88,
        initial_indent=initial_indent,
        subsequent_indent=subsequent_indent,
        break_long_words=False,
        break_on_hyphens=False,
    )
    return wrapped.replace("\0", " ")


def _wrap_markdown(text: str) -> str:
    """Pass every prose paragraph through ``_wrap`` without changing Markdown tables."""
    lines = text.strip().splitlines()
    rendered: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            rendered.append("")
            index += 1
            continue
        if line.startswith(("#", "|")):
            rendered.append(line.rstrip())
            index += 1
            continue
        marker = re.match(r"^(\s*(?:[-*]|\d+\.)\s+)(.*)$", line)
        if marker:
            prefix, first = marker.groups()
            paragraph = [first]
            index += 1
            while index < len(lines):
                candidate = lines[index]
                if (
                    not candidate.strip()
                    or candidate.startswith(("#", "|"))
                    or re.match(r"^\s*(?:[-*]|\d+\.)\s+", candidate)
                ):
                    break
                paragraph.append(candidate.strip())
                index += 1
            rendered.append(
                _wrap(
                    " ".join(paragraph), initial_indent=prefix, subsequent_indent=" " * len(prefix)
                )
            )
            continue
        paragraph = [line.strip()]
        index += 1
        while index < len(lines):
            candidate = lines[index]
            if (
                not candidate.strip()
                or candidate.startswith(("#", "|"))
                or re.match(r"^\s*(?:[-*]|\d+\.)\s+", candidate)
            ):
                break
            paragraph.append(candidate.strip())
            index += 1
        rendered.append(_wrap(" ".join(paragraph)))
    return "\n".join(rendered).rstrip() + "\n"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _selected(case: dict[str, Any]) -> dict[str, Any]:
    return case["Y_certificate_selected"]


def _abstention_label(reason: str) -> str:
    if "connectivity" in reason:
        return "connectivity"
    if "no polynomial" in reason:
        return "no polynomial"
    if "compute" in reason:
        return "compute cap"
    raise RuntimeError(f"unknown certificate abstention reason: {reason}")


def _assert_evidence(data: dict[str, dict[str, Any]]) -> None:
    active = data["active"]
    cases = active["cases"]
    lemma = data["lemma"]
    mask = data["mask"]

    _require(
        all(row["passes"] and row["margin"] >= 0 for row in lemma["lemma6"]),
        "Lemma 6 lower-bound evidence contains a failure or negative margin",
    )
    _require(
        not any(not row["passes"] for row in lemma["lemma2"]),
        "Lemma 2 evidence contains a norm-bound violation",
    )
    _require(
        lemma["lemma3"]["violations"] == 0
        and not any(not row["passes"] for row in lemma["lemma3"]["rows"]),
        "Lemma 3 evidence contains a trajectory-bound violation",
    )
    _require(
        all(
            row["J0_blocks"]["max_abs_E"] == 0.0
            and row["J0_blocks"]["max_abs_F"] == 0.0
            and row["regularized_blocks"]["Y_byte_identical_to_J0"]
            for row in cases
        ),
        "active-block evidence violates E0 = F0 = 0 or Y identity",
    )
    _require(
        all(
            row["inactive_count"] == 0
            for row in cases
            if row["m"] == 2 and row["method"] == "repository_blend_B3"
        ),
        "repository mask is non-empty at m = 2",
    )
    _require(
        all(row["repository_masked_count"] == 0 for row in mask["states"] if row["m"] == 2),
        "delivery repository mask is non-empty at m = 2",
    )
    for row in mask["states"]:
        if row["m"] not in {4, 8}:
            continue
        check = row["repository_checks"]["repo_blend_B3"]
        _require(
            row["repository_masked_count"] == row["cfl_masked_count"]
            and check["inactive_count_repository"] == row["cfl_masked_count"]
            and check["reference_max_relative_difference"] == 0.0,
            f"repository and CFL blends differ at N={row['nx']}, m={row['m']}",
        )

    summary = active["decision_summary"]
    certified = [row for row in cases if _selected(row)["certified"]]
    disk_adequate = [row for row in cases if row["merged_policy"]["verdict"] == "adequate"]
    union = [row for row in cases if row["union"] == "certified"]
    _require(
        len(cases) == 21
        and len(certified) == summary["Y_certified_count"] == 15
        and len(union) == summary["union_certified_count"] == 15
        and len(disk_adequate) == summary["disk_adequate_count"] == 2,
        "decision counts are not 15 Y-certified, 15 union-certified and 2 disk-adequate of 21",
    )
    method_counts = {
        method: sum(row["method"] == method for row in certified)
        for method in {
            "one_inverse_cfl_mask",
            "cfl_reference_blend_B3",
            "repository_blend_B3",
        }
    }
    _require(
        method_counts
        == {
            "one_inverse_cfl_mask": 6,
            "cfl_reference_blend_B3": 8,
            "repository_blend_B3": 1,
        },
        "Y-certificate method counts are not 6/8/1",
    )
    reason_counts: dict[str, int] = {}
    for row in cases:
        selected = _selected(row)
        if selected["certified"]:
            continue
        label = _abstention_label(selected["reason"])
        reason_counts[label] = reason_counts.get(label, 0) + 1
    _require(
        reason_counts == {"connectivity": 3, "no polynomial": 2, "compute cap": 1},
        "certificate abstention counts are not 3/2/1",
    )

    target = active["configuration"]["target_tolerance"]
    for row in certified:
        selected = _selected(row)
        degree = selected["degree"]
        rho = selected["rho"]
        product = row["J0_blocks"]["lemma1_constant"] * selected["prefactor"]
        j_star = 0
        while product * rho ** (degree * j_star) > target:
            j_star += 1
        recomputed = 1 + degree * j_star
        _require(
            recomputed == selected["lemma1_full_operator_iteration_bound"],
            "Lemma 1' bound does not recompute for "
            f"N={row['nx']}, m={row['m']}, method={row['method']}",
        )

    groups: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in cases:
        groups.setdefault((row["nx"], row["m"]), []).append(row)
    for (nx, m), rows in groups.items():
        positive = [row for row in rows if row["union"] == "certified"]
        negative = [row for row in rows if row["union"] != "certified"]
        if not positive:
            continue
        if negative:
            _require(
                max(row["gmres_regularized"]["true_stop_count"] for row in positive)
                <= min(row["gmres_regularized"]["true_stop_count"] for row in negative),
                f"certified method is slower than an abstaining method at N={nx}, m={m}",
            )
        fastest = min(row["gmres_regularized"]["true_stop_count"] for row in rows)
        _require(
            any(row["gmres_regularized"]["true_stop_count"] == fastest for row in positive),
            f"fastest method is not certified at N={nx}, m={m}",
        )
    _require(
        all(
            row["gmres_regularized"]["true_stop_count"]
            > row["gmres_regularized"]["preconditioned_stop_count"]
            for row in cases
        ),
        "a true-residual stop count is not greater than its preconditioned stop count",
    )

    repro = data["repro"]
    mask_rows = repro["mask_analysis_subset"]
    e3_rows = repro["drifted_E3_subset"]
    _require(
        len(mask_rows) == 46 and sum(not row["all_equal"] for row in mask_rows) == 4,
        "mask reproducibility count is not 4 of 46",
    )
    _require(
        len(e3_rows) == 96 and sum(not row["all_equal"] for row in e3_rows) == 48,
        "E3 reproducibility count is not 48 of 96",
    )

    baseline = data["baseline"]
    spectral = baseline["spectral_records"]
    epsilon = float(baseline["config"]["epsilon"])
    m_values = sorted({int(row["m"]) for row in spectral})
    low_m = m_values[0]
    for m in m_values[1:]:
        active_scale = [
            row
            for row in spectral
            if int(row["m"]) == m
            and row["d0"] is not None
            and float(row["d0"]) / float(row["coefficient"]["active_maximum"]) >= 0.01
        ]
        _require(
            active_scale
            and all(row["spectrum"]["near_zero_eigenvalue_count"] > 0 for row in active_scale),
            f"an active-scale scalar lacks a near-zero tail at m = {m}",
        )
        oracle = next(row for row in active_scale if row["method"] == "optimized_d0")
        _require(
            oracle["spectrum"]["condition_number_2"]
            == max(row["spectrum"]["condition_number_2"] for row in active_scale)
            and oracle["spectrum"]["minimum_eigenvalue_modulus"]
            == min(row["spectrum"]["minimum_eigenvalue_modulus"] for row in active_scale),
            f"the oracle is not spectrally worst among active-scale scalars at m = {m}",
        )

    low_records = [row for row in spectral if int(row["m"]) == low_m and row["d0"] is not None]
    for row in low_records:
        floor = low_m * epsilon ** (low_m - 1)
        ratio = float(row["d0"]) / floor
        if ratio <= 100:
            _require(
                row["spectrum"]["near_zero_eigenvalue_count"] == 0,
                "an m = 2 near-floor reference has a near-zero tail",
            )
        if ratio >= 1.0e4:
            _require(
                row["spectrum"]["near_zero_eigenvalue_count"] > 0,
                "an m = 2 far-from-floor reference lacks a near-zero tail",
            )

    blend_rows = [row for row in spectral if row["method"] == "tau_blend"]
    _require(
        len(blend_rows) == len(m_values)
        and all(row["spectrum"]["near_zero_eigenvalue_count"] == 0 for row in blend_rows),
        "the blend has a near-zero eigenvalue in the baseline spectral table",
    )
    tight_tolerance = min(float(value) for value in baseline["config"]["work_tolerances"])
    work = {
        (int(row["nx"]), int(row["m"])): row
        for row in baseline["work_precision_records"]
        if float(row["requested_relative_residual"]) == tight_tolerance
    }
    shared = (
        (512, 2, "repository_blend_B3", 0),
        (1024, 2, "repository_blend_B3", 1),
        (512, 4, "cfl_reference_blend_B3", 0),
        (512, 8, "cfl_reference_blend_B3", 0),
        (1024, 4, "cfl_reference_blend_B3", 0),
        (1024, 8, "cfl_reference_blend_B3", 0),
    )
    for nx, m, method, allowance in shared:
        part_i = next(
            row for row in cases if row["nx"] == nx and row["m"] == m and row["method"] == method
        )
        part_ii = work[(nx, m)]["methods"]["tau_blend"]["iterations"][0]
        difference = abs(part_ii - part_i["gmres_regularized"]["preconditioned_stop_count"])
        _require(
            difference <= allowance,
            f"Part I/II blend stopping counts disagree at N={nx}, m={m}",
        )
    low_blend = next(row for row in blend_rows if int(row["m"]) == low_m)
    part_i_low = next(
        row
        for row in cases
        if row["nx"] == baseline["config"]["spectral_nx"]
        and row["m"] == low_m
        and row["method"] == "repository_blend_B3"
    )
    _require(
        abs(low_blend["spectrum"]["spectral_abscissa"] - part_i_low["Y_spectrum"]["maximum_real"])
        <= 1.0e-6 * abs(part_i_low["Y_spectrum"]["maximum_real"]),
        "the m = 2 blend spectral abscissa does not match Part I",
    )

    loose_tolerance = max(float(value) for value in baseline["config"]["work_tolerances"])
    loose = [
        row
        for row in baseline["work_precision_records"]
        if float(row["requested_relative_residual"]) == loose_tolerance
    ]
    loose_percentages = []
    for row in loose:
        scalars = [
            values
            for method, values in row["methods"].items()
            if method not in {"identity", "tau_blend"} and values["all_converged"]
        ]
        fastest = min(scalars, key=lambda values: values["median_seconds"])
        loose_percentages.append(
            100.0
            * (row["methods"]["tau_blend"]["median_seconds"] / fastest["median_seconds"] - 1.0)
        )
    _require(
        len(loose_percentages)
        == len(baseline["config"]["work_nx_values"]) * len(baseline["config"]["work_m_values"])
        and all(value > 0.0 for value in loose_percentages),
        "the blend is not slower than the fastest scalar in every loose-tolerance case",
    )

    mismatches = [
        row for row in data["batch_gpu"]["solve_records"] if not row["iteration_sets_identical"]
    ]
    _require(len(mismatches) == 1, "the batching audit does not contain exactly one mismatch")
    legacy_low = next(row for row in data["criterion_legacy"]["records"] if int(row["m"]) == low_m)
    current_low = next(
        row for row in data["criterion_current"]["records"] if int(row["m"]) == low_m
    )
    _require(
        not legacy_low["pseudospectral_criterion"]["certified"]
        and not current_low["pseudospectral_criterion"]["certified"],
        "the dense criterion unexpectedly certifies m = 2",
    )
    legacy_verdicts = {
        int(row["m"]): bool(row["pseudospectral_criterion"]["certified"])
        for row in data["criterion_legacy"]["records"]
    }
    current_verdicts = {
        int(row["m"]): bool(row["pseudospectral_criterion"]["certified"])
        for row in data["criterion_current"]["records"]
    }
    _require(
        legacy_verdicts == current_verdicts,
        "the current dense-criterion rerun changes a committed verdict",
    )


def _lemma6_table(rows: list[dict[str, Any]]) -> str:
    names = {"max": "max", "median": "median", "geomean": "geometric mean"}
    return _table(
        ("N", "m", "reference", "lower bound", "measured below 0.1"),
        [
            (
                row["nx"],
                row["m"],
                names[row["case"].removeprefix("scalar_")],
                row["lower_bound"],
                row["measured_below_0p1"],
            )
            for row in rows
        ],
    )


def _m2_table(cases: list[dict[str, Any]], mask: dict[str, Any]) -> str:
    floor_by_n = {row["nx"]: row["floor_cfl"] for row in mask["states"] if row["m"] == 2}
    order = {"repository_blend_B3": 0, "one_inverse_cfl_mask": 1, "cfl_reference_blend_B3": 2}
    rows = sorted(
        (row for row in cases if row["m"] == 2 and row["nx"] in floor_by_n),
        key=lambda row: (row["nx"], order[row["method"]]),
    )
    return _table(
        ("N", "method", "masked nodes", "chi_floor", "max Re sigma(Y)", "GMRES"),
        [
            (
                row["nx"],
                f"`{row['method']}`",
                row["inactive_count"],
                floor_by_n[row["nx"]],
                row["Y_spectrum"]["maximum_real"],
                row["gmres_regularized"]["true_stop_count"],
            )
            for row in rows
        ],
    )


def _active_geometry_table(cases: list[dict[str, Any]]) -> str:
    return _table(
        ("N", "m", "method", "sigma_min(Y)", "C"),
        [
            (
                row["nx"],
                row["m"],
                f"`{row['method']}`",
                row["Y_spectrum"]["sigma_min"],
                row["J0_blocks"]["lemma1_constant"],
            )
            for row in cases
        ],
    )


def _decision_table(cases: list[dict[str, Any]]) -> str:
    rows = []
    for row in cases:
        selected = _selected(row)
        certificate = (
            "certified"
            if selected["certified"]
            else f"abstains: {_abstention_label(selected['reason'])}"
        )
        rows.append(
            (
                row["nx"],
                row["m"],
                f"`{row['method']}`",
                row["merged_policy"]["verdict"],
                certificate,
                f"{selected['grid']} / {_fmt(selected.get('degree'))}",
                selected.get("rho"),
                selected.get("prefactor"),
                selected.get("lemma1_full_operator_iteration_bound"),
                row["gmres_J0"]["true_stop_count"],
                row["gmres_regularized"]["true_stop_count"],
            )
        )
    return _table(
        (
            "N",
            "m",
            "method",
            "disk verdict",
            "Y certificate",
            "grid / degree",
            "rho",
            "prefactor",
            "Lemma 1' bound",
            "GMRES `J_0`",
            "GMRES `J`",
        ),
        rows,
    )


def _direct_table(active: dict[str, Any]) -> str:
    gmres = {
        (row["nx"], row["m"], row["method"]): row["gmres_regularized"]["true_stop_count"]
        for row in active["cases"]
    }
    rows = []
    for row in active["direct_regularized_full_MJ_certificates"]:
        selected = next(item for item in row["certificate_ladder"] if item["certified"])
        rows.append(
            (
                row["nx"],
                selected["grid"],
                selected["degree"],
                selected["rho"],
                selected["prefactor"],
                gmres[(row["nx"], row["m"], row["method"])],
            )
        )
    return _table(("N", "grid", "degree", "rho", "prefactor", "GMRES `J`"), rows)


def _reference_table(rows: list[dict[str, Any]]) -> str:
    return _table(
        (
            "N",
            "m",
            "references",
            "sigma_min(Y)",
            "GMRES (true)",
            "GMRES (preconditioned)",
            "certificate",
        ),
        [
            (
                row["nx"],
                row["m"],
                row["reference_count"],
                row["sigma_min"],
                row["gmres"]["true_stop_count"],
                row["gmres"]["preconditioned_stop_count"],
                (
                    "certified"
                    if row["Y_criterion"] is not None and row["Y_criterion"]["certified"]
                    else "not run"
                ),
            )
            for row in rows
        ],
    )


def _lemma4_table(rows: list[dict[str, Any]]) -> str:
    kind_order = {"S": 0, "B3": 1, "Z1": 2, "Zf": 3}
    selected = [
        row
        for row in rows
        if row["dt"] == 0.02
        and (row["eta"] is None or row["eta"] == (0.3 if row["m"] == 2 else 0.1))
    ]
    selected.sort(key=lambda row: (row["m"], kind_order[row["kind"]]))
    by_m = {m: {row["kind"]: row for row in selected if row["m"] == m} for m in (2, 4, 8)}
    return _table(
        (
            "m",
            "scalar (S)",
            "repository blend (B3)",
            "mask, one inverse (Z1)",
            "mask with frame (Zf)",
        ),
        [
            (
                m,
                by_m[m]["S"]["kappa2_M"],
                by_m[m]["B3"]["kappa2_M"],
                by_m[m]["Z1"]["kappa2_M"],
                by_m[m]["Zf"]["kappa2_M"],
            )
            for m in (2, 4, 8)
        ],
    )


def _cn_table(rows: list[dict[str, Any]]) -> str:
    return _table(
        ("m", "BE Newton", "BE GMRES", "CN Newton", "CN GMRES"),
        [
            (
                row["m"],
                row["BE"]["newton_total"],
                row["BE"]["gmres_total"],
                row["CN"]["newton_total"],
                row["CN"]["gmres_total"],
            )
            for row in rows
        ],
    )


def _part_ii_spectral_table(baseline: dict[str, Any]) -> str:
    m_values = sorted({int(row["m"]) for row in baseline["spectral_records"]})
    low_m = m_values[0]
    order = {method: index for index, method in enumerate(PART_II_METHODS)}
    records = sorted(
        (
            row
            for row in baseline["spectral_records"]
            if not (row["method"] == "floor" and int(row["m"]) != low_m)
        ),
        key=lambda row: (int(row["m"]), order[row["method"]]),
    )
    epsilon = float(baseline["config"]["epsilon"])
    rows = []
    for row in records:
        m = int(row["m"])
        d0 = row["d0"]
        ratios = (None, None)
        if d0 is not None and row["method"] != "identity":
            ratios = (
                float(d0) / (m * epsilon ** (m - 1)),
                float(d0) / float(row["coefficient"]["active_maximum"]),
            )
        rows.append(
            (
                m,
                PART_II_LABELS[row["method"]],
                *(_ratio_column(value) for value in ratios),
                _ratio_column(row["spectrum"]["condition_number_2"]),
                row["spectrum"]["near_zero_eigenvalue_count"],
                _ratio_column(row["spectrum"]["minimum_eigenvalue_modulus"]),
            )
        )
    return _table(
        (
            "m",
            "reference",
            "d / floor",
            "d / active max",
            "kappa_2",
            "count below 0.1",
            "min abs(lambda)",
        ),
        rows,
        numeric_columns=frozenset({0, 2, 3, 4, 5, 6}),
    )


def _part_ii_work_tables(baseline: dict[str, Any]) -> tuple[str, str]:
    records = sorted(
        baseline["work_precision_records"],
        key=lambda row: (
            int(row["nx"]),
            int(row["m"]),
            -float(row["requested_relative_residual"]),
        ),
    )
    iteration_rows = []
    timing_rows = []
    for row in records:
        prefix = (row["nx"], row["m"], row["requested_relative_residual"])
        iteration_rows.append(
            (
                *prefix,
                *(
                    values["iterations"][0] if values["all_converged"] else "cap"
                    for values in (row["methods"][method] for method in PART_II_METHODS)
                ),
            )
        )
        timing_rows.append(
            (
                *prefix,
                *(
                    f"{_sig(values['median_seconds'], 3)} [{_sig(values['iqr_seconds'], 2)}]"
                    for values in (row["methods"][method] for method in PART_II_METHODS)
                ),
            )
        )
    headers = (
        "N",
        "m",
        "tol",
        *(PART_II_LABELS[method] for method in PART_II_METHODS),
    )
    numeric = frozenset(range(len(headers)))
    return (
        _table(headers, iteration_rows, numeric_columns=numeric),
        _table(headers, timing_rows, numeric_columns=numeric),
    )


def _part_ii_contrast_table(
    baseline: dict[str, Any],
) -> tuple[str, list[tuple[dict[str, Any], str, int, int]]]:
    order = {method: index for index, method in enumerate(PART_II_METHODS)}
    selected = []
    rows = []
    for row in sorted(baseline["contrast_records"], key=lambda item: (item["m"], item["step"])):
        candidates = {
            method: values
            for method, values in row["methods"].items()
            if method != "tau_blend" and values["converged"]
        }
        best_method = min(
            candidates,
            key=lambda method: (candidates[method]["iterations"], order[method]),
        )
        best_count = int(candidates[best_method]["iterations"])
        blend_count = int(row["methods"]["tau_blend"]["iterations"])
        selected.append((row, best_method, best_count, blend_count))
        rows.append(
            (
                row["m"],
                row["step"],
                row["coefficient"]["degeneracy_fraction"],
                f"{PART_II_LABELS[best_method]} ({best_count})",
                blend_count,
                _ratio_column(best_count / blend_count),
            )
        )
    return (
        _table(
            ("m", "step", "degeneracy fraction", "best single", "blend", "ratio"),
            rows,
            numeric_columns=frozenset({0, 1, 2, 4, 5}),
        ),
        selected,
    )


def _render_part_ii(data: dict[str, dict[str, Any]]) -> str:
    baseline = data["baseline"]
    config = baseline["config"]
    spectral = baseline["spectral_records"]
    m_values = sorted({int(row["m"]) for row in spectral})
    low_m, middle_m, high_m = m_values
    work_n = sorted(int(value) for value in config["work_nx_values"])
    spectral_n = int(config["spectral_nx"])
    criterion_n = int(data["criterion_legacy"]["config"]["nx"])
    tolerances = sorted(float(value) for value in config["work_tolerances"])
    tight_tolerance = tolerances[0]
    loose_tolerance = tolerances[-1]
    max_iterations = int(config["max_krylov_iters"])
    capped_count = max_iterations + 1
    epsilon = float(config["epsilon"])
    certificate_change_count = len(data["active"]["configuration"]["certificate_changes"])

    spectral_table = _part_ii_spectral_table(baseline)
    work_table, timing_table = _part_ii_work_tables(baseline)
    contrast_table, contrast = _part_ii_contrast_table(baseline)

    by_spectral = {(int(row["m"]), row["method"]): row for row in spectral}
    active_ranges = {}
    for m in (middle_m, high_m):
        rows = [
            row
            for row in spectral
            if int(row["m"]) == m
            and row["d0"] is not None
            and float(row["d0"]) / float(row["coefficient"]["active_maximum"]) >= 0.01
        ]
        active_ranges[m] = {
            "ratio": (
                min(float(row["d0"]) / float(row["coefficient"]["active_maximum"]) for row in rows),
                max(float(row["d0"]) / float(row["coefficient"]["active_maximum"]) for row in rows),
            ),
            "count": (
                min(row["spectrum"]["near_zero_eigenvalue_count"] for row in rows),
                max(row["spectrum"]["near_zero_eigenvalue_count"] for row in rows),
            ),
            "kappa": [row["spectrum"]["condition_number_2"] for row in rows],
        }

    low_rows = [
        row
        for row in spectral
        if int(row["m"]) == low_m and row["d0"] is not None and row["method"] != "identity"
    ]
    low_floor = low_m * epsilon ** (low_m - 1)
    low_far = [row for row in low_rows if float(row["d0"]) / low_floor >= 1.0e4]
    low_near = [row for row in low_rows if float(row["d0"]) / low_floor <= 100]
    blend_kappa = [
        by_spectral[(m, "tau_blend")]["spectrum"]["condition_number_2"] for m in m_values
    ]

    work = {
        (int(row["nx"]), int(row["m"]), float(row["requested_relative_residual"])): row
        for row in baseline["work_precision_records"]
    }
    shared_specs = (
        (512, low_m, "repository_blend_B3"),
        (1024, low_m, "repository_blend_B3"),
        (512, middle_m, "cfl_reference_blend_B3"),
        (512, high_m, "cfl_reference_blend_B3"),
        (1024, middle_m, "cfl_reference_blend_B3"),
        (1024, high_m, "cfl_reference_blend_B3"),
    )
    stopping_rows = []
    for nx, m, method in shared_specs:
        part_i = next(
            row
            for row in data["active"]["cases"]
            if row["nx"] == nx and row["m"] == m and row["method"] == method
        )
        part_ii_count = work[(nx, m, tight_tolerance)]["methods"]["tau_blend"]["iterations"][0]
        pre_count = part_i["gmres_regularized"]["preconditioned_stop_count"]
        stopping_rows.append(
            (
                nx,
                m,
                "repository" if method == "repository_blend_B3" else "CFL",
                part_ii_count,
                pre_count,
                part_i["gmres_regularized"]["true_stop_count"],
                part_ii_count - pre_count,
            )
        )
    stopping_table = _table(
        ("N", "m", "Part I blend", "Part II", "Part I preconditioned", "Part I true", "delta"),
        stopping_rows,
        numeric_columns=frozenset({0, 1, 3, 4, 5, 6}),
    )
    tight = [work[(nx, m, tight_tolerance)] for nx in work_n for m in m_values]
    active_scale_methods = ("frozen_mean", "frozen_bulk", "const")
    active_scale_slower_count = sum(
        all(
            row["methods"][method]["iterations"][0] > row["methods"]["identity"]["iterations"][0]
            for method in active_scale_methods
        )
        for row in tight
    )
    blend_counts = {
        nx: [
            work[(nx, m, tight_tolerance)]["methods"]["tau_blend"]["iterations"][0]
            for m in m_values
        ]
        for nx in work_n
    }
    oracle_time_ratios = []
    for row in tight:
        oracle = row["methods"]["optimized_d0"]
        if oracle["all_converged"]:
            oracle_time_ratios.append(
                oracle["median_seconds"] / row["methods"]["tau_blend"]["median_seconds"]
            )
    middle_large = work[(work_n[-1], middle_m, tight_tolerance)]
    middle_identity_ratio = (
        middle_large["methods"]["identity"]["median_seconds"]
        / middle_large["methods"]["tau_blend"]["median_seconds"]
    )

    loose_percentages = []
    loose_rows = []
    for nx in work_n:
        for m in m_values:
            row = work[(nx, m, loose_tolerance)]
            scalars = {
                method: values
                for method, values in row["methods"].items()
                if method not in {"identity", "tau_blend"} and values["all_converged"]
            }
            fastest = min(scalars, key=lambda method: scalars[method]["median_seconds"])
            blend = row["methods"]["tau_blend"]
            percentage = 100.0 * (
                blend["median_seconds"] / scalars[fastest]["median_seconds"] - 1.0
            )
            loose_percentages.append(percentage)
            loose_rows.append((row, fastest))
    loose_active_rows = [
        (row, method) for row, method in loose_rows if int(row["m"]) in {middle_m, high_m}
    ]
    loose_active_scalar_counts = [
        row["methods"][method]["iterations"][0] for row, method in loose_active_rows
    ]
    loose_active_blend_counts = [
        row["methods"]["tau_blend"]["iterations"][0] for row, _ in loose_active_rows
    ]
    low_loose = [work[(nx, low_m, loose_tolerance)] for nx in work_n]
    low_loose_blend_counts = [row["methods"]["tau_blend"]["iterations"][0] for row in low_loose]
    low_loose_best_counts = [
        min(
            values["iterations"][0]
            for method, values in row["methods"].items()
            if method not in {"identity", "tau_blend"} and values["all_converged"]
        )
        for row in low_loose
    ]

    high_contrast = [item for item in contrast if int(item[0]["m"]) == high_m]
    low_contrast = [item for item in contrast if int(item[0]["m"]) == low_m]
    high_ratios = [best / blend for _, _, best, blend in high_contrast]
    low_ratios = [best / blend for _, _, best, blend in low_contrast]
    contrast_values = {
        m: [
            float(row[0]["coefficient"]["active_d95_over_d05"])
            for row in contrast
            if row[0]["m"] == m
        ]
        for m in (low_m, high_m)
    }
    oracle_locations = [
        (
            float(row[0]["optimized_d0_scan"]["ratio_to_active_maximum"]),
            float(row[0]["optimized_d0_scan"]["ratio_to_active_minimum"]),
        )
        for row in contrast
    ]

    gpu = data["batch_gpu"]
    cpu = data["batch_cpu"]
    gpu_apply = [
        row["timing"]["speedup_batched_over_forced_sequential"] for row in gpu["apply_records"]
    ]
    gpu_solve = [
        row["full_solve_speedup_batched_over_forced_sequential"] for row in gpu["solve_records"]
    ]
    cpu_apply = [
        row["timing"]["speedup_batched_over_forced_sequential"] for row in cpu["apply_records"]
    ]
    mismatch = next(row for row in gpu["solve_records"] if not row["iteration_sets_identical"])
    batched_mismatch_count = mismatch["phase0_batched"]["iterations"][0]
    sequential_mismatch_count = mismatch["forced_sequential"]["iterations"][0]

    legacy = {int(row["m"]): row for row in data["criterion_legacy"]["records"]}
    current = {int(row["m"]): row for row in data["criterion_current"]["records"]}
    criterion_rows = []
    for m in m_values:
        row = legacy[m]
        criterion = row["pseudospectral_criterion"]
        criterion_rows.append(
            (
                m,
                "yes" if criterion["certified"] else "no",
                criterion["polynomial_effective_rate"],
                criterion["predicted_iterations"],
                row["actual_counted_gmres"]["iterations"],
            )
        )
    criterion_table = _table(
        ("m", "certified", "rate", "predicted iterations", "GMRES"), criterion_rows
    )
    current_comparison = "; ".join(
        f"`m = {m}`: {_fmt(legacy[m]['pseudospectral_criterion']['polynomial_effective_rate'])} / "
        f"{_fmt(legacy[m]['pseudospectral_criterion']['predicted_iterations'])} to "
        f"{_fmt(current[m]['pseudospectral_criterion']['polynomial_effective_rate'])} / "
        f"{_fmt(current[m]['pseudospectral_criterion']['predicted_iterations'])}"
        for m in m_values
    )
    current_prediction_ratios = {
        m: current[m]["pseudospectral_criterion"]["predicted_iterations"]
        / current[m]["actual_counted_gmres"]["iterations"]
        for m in (middle_m, high_m)
    }

    result = f"""# Part II. Single-reference baseline study (corrected coefficient)

This part restates the earlier single-reference study against its JSONs, regenerated
with the corrected coefficient. Section II.6 lists what changed from the previous
version.

**Scope.** The same one-dimensional problem as Part I, at one wide-front
backward-Euler linearization of the initial state and three later states
(steps {', '.join(str(step) for step in config['evolved_sample_steps'] if step != 0)}),
`m = {', '.join(str(m) for m in m_values)}`, `N = {', '.join(str(n) for n in work_n)}`.
The dense spectral table uses `N = {spectral_n}` and the dense criterion uses
`N = {criterion_n}`. The blend in this part is the repository blend with the
repository mask, so at `m = {low_m}` it has no mask (Part I, Section 3); its
`m = {low_m}` spectrum is the one Part I reports for `repository_blend_B3` at
`N = {spectral_n}`.

**Counts and timings.** Part II solves the same left-preconditioned system as the
PME conditioning benchmark and stops on
`||P^-1 (b - J x)|| / ||P^-1 b||`. Identity is the sole case in which this is the
true residual. Comparisons across preconditioners therefore mix stopping norms, as
Lemma 4 warns; Part I, Section 8 reports both conventions for the shared blend
operators. Their counts are:

{stopping_table}

The JSONs were generated at
`{baseline['provenance']['repository_head']}` (base
`{baseline['provenance']['base_revision']}`) and committed in the correction delivery;
they record no host, device model or library versions, so all timings in this part are
indicative only.

Source: the Part II JSONs, `/provenance` [P1].

## II.1 Scalar references and the Lemma 6 boundary

{spectral_table}

Source: `pme_dst_tau_blend_single_reference_baselines.json`, `/spectral_records/*`
[P2]. At `m = {middle_m}` and `{high_m}` the floor reference
(`{_fmt(middle_m * epsilon ** (middle_m - 1))}`,
`{_fmt(high_m * epsilon ** (high_m - 1))}`) coincides with identity and is omitted.

Lemma 6 explains every row. With `beta_k = (1 + dt f mu_k) / (1 + dt d mu_k)`, a
reference `d` far above the floor value `f` leaves the floor-region modes near `f/d`,
a tail of small eigenvalues; a reference near the floor gives `beta_k` near one and no
tail, but then it barely preconditions the active part.

- **m = {middle_m} and m = {high_m}.** Every scalar reference at the active scale
  (`d` from {_sig(active_ranges[high_m]['ratio'][0])} to
  {_sig(active_ranges[middle_m]['ratio'][1])} times the active maximum) keeps a tail:
  {active_ranges[middle_m]['count'][0]} to {active_ranges[middle_m]['count'][1]}
  eigenvalues below `0.1` at `m = {middle_m}`, and
  {active_ranges[high_m]['count'][0]} to {active_ranges[high_m]['count'][1]} at
  `m = {high_m}`. Their condition numbers lie within
  {max(100 * (max(values['kappa']) - min(values['kappa'])) / min(values['kappa']) for values in active_ranges.values()):.1f}%
  of each other, and the GMRES-optimal oracle is spectrally the worst: the largest
  `kappa_2` and the smallest `min |lambda|`.
- **m = {low_m}.** The references split by distance from the floor. The
  {len(low_far)} at {_fmt(min(float(row['d0']) / low_floor for row in low_far))} to
  {_fmt(max(float(row['d0']) / low_floor for row in low_far))} times the floor keep
  {min(row['spectrum']['near_zero_eigenvalue_count'] for row in low_far)} to
  {max(row['spectrum']['near_zero_eigenvalue_count'] for row in low_far)} small
  eigenvalues. The {len(low_near)} within
  {_fmt(max(float(row['d0']) / low_floor for row in low_near))} times the floor have
  none, but their condition numbers,
  {_fmt(min(row['spectrum']['condition_number_2'] for row in low_near))} to
  {_fmt(max(row['spectrum']['condition_number_2'] for row in low_near))}, are close to
  identity's {_fmt(by_spectral[(low_m, 'identity')]['spectrum']['condition_number_2'])}.
  The geometric and harmonic means land near the floor because the corrected
  repository rule treats every node as active, so the floor nodes enter the means.
- **The blend** has no eigenvalue below `0.1` at any `m`, with `kappa_2`
  {', '.join(_fmt(value) for value in blend_kappa)}. At `m = {low_m}` its larger
  condition number is that of an empty mask; Part I shows that the CFL mask repairs it.

## II.2 Linear-solve work at one linearization

GMRES iterations; `cap` means {max_iterations} iterations without convergence
(reported as {capped_count}, the inherited counter convention).

{work_table}

Median [IQR] seconds, with the unrecorded-environment caveat above:

{timing_table}

Source: `pme_dst_tau_blend_single_reference_baselines.json`,
`/work_precision_records/*` [P3].

**Tight tolerance (`{_fmt(tight_tolerance)}`).** The tail costs more than it saves.
In {active_scale_slower_count} of the {len(tight)} cases, every active-scale scalar
needs more iterations than no preconditioner at all; at
`N = {work_n[-1]}, m = {high_m}` every single reference caps. The blend needs
{', '.join(str(value) for value in blend_counts[work_n[0]])}
iterations at `N = {work_n[0]}` and
{', '.join(str(value) for value in blend_counts[work_n[-1]])} at `N = {work_n[-1]}`.
Against the converged oracle, its time is
{', '.join(_sig(value) for value in oracle_time_ratios[:3])} times smaller at
`N = {work_n[0]}` and {_sig(oracle_time_ratios[-1])} times smaller at
`N = {work_n[-1]}, m = {low_m}`; at `N = {work_n[-1]}, m = {middle_m}` the oracle
caps and the fastest converged comparator, identity, takes
{_sig(middle_identity_ratio)} times longer.

**Loose tolerance (`{_fmt(loose_tolerance)}`).** The blend is slower than the fastest
converged scalar in all {len(loose_percentages)} cases, by
{min(loose_percentages):.1f}% to {max(loose_percentages):.1f}%. At
`m = {middle_m}` and `{high_m}` it needs {min(loose_active_blend_counts)} to
{max(loose_active_blend_counts)} iterations against
{min(loose_active_scalar_counts)} to {max(loose_active_scalar_counts)} for the
fastest scalars, but applies {config['reference_count']} Helmholtz inverses per
iteration. At `m = {low_m}` it needs more iterations in both cases
({low_loose_blend_counts[0]} and {low_loose_blend_counts[1]} against
{low_loose_best_counts[0]}).

Source: `pme_dst_tau_blend_single_reference_baselines.json` [P4, P5].

## II.3 Later states (contrast sweep)

{contrast_table}

GMRES iterations at `{_fmt(config['contrast_tolerance'])}`, `N =
{config['contrast_nx']}`. "Best single" is the minimum converged iteration count over
the {len(PART_II_METHODS) - 1} non-blend comparators, with the selected method named.
Source: `pme_dst_tau_blend_single_reference_baselines.json`, `/contrast_records/*`
[P6].

At `m = {high_m}` the best single reference is identity or the floor in every state,
and the blend's advantage ({', '.join(_sig(value) for value in high_ratios)}) narrows
as the front fills in, without vanishing. At `m = {low_m}` the repository rule marks
no node as degenerate, and the advantage is
{_sig(min(low_ratios))} to {_sig(max(low_ratios))}. The oracle sits at the floor in
the initial `m = {low_m}` state and at the active maximum in the other
{sum(location[0] == 1.0 for location in oracle_locations)} states. The
coefficient-contrast ratio `D95/D05` (about
`{_sig(max(contrast_values[low_m]), 2)}` at `m = {low_m}` and
`{_sig(min(contrast_values[high_m]), 2)}` to `{_sig(max(contrast_values[high_m]), 2)}` at
`m = {high_m}`) does not order these results.

## II.4 Batching audit

The production blend applies its {config['reference_count']} references in one batched
Helmholtz call. On the GPU the batched and sequential actions agree to
`{_sig(gpu['maximum_relative_action_difference'])}` relative; the batched application
is {_sig(min(gpu_apply))} to {_sig(max(gpu_apply))} times faster, and the full solve
{_sig(min(gpu_solve))} to {_sig(max(gpu_solve))} times. The iteration counts agree in
{sum(row['iteration_sets_identical'] for row in gpu['solve_records'])} of
{len(gpu['solve_records'])} solves; at `N = {mismatch['nx']}, m = {mismatch['m']}`,
`{_fmt(mismatch['requested_relative_residual'])}`, the batched solve takes
{batched_mismatch_count} iterations and the sequential one
{sequential_mismatch_count}, a difference at the stopping threshold
that we attribute to round-off (an interpretation). On the CPU the batched application
is {_sig(min(cpu_apply))} to {_sig(max(cpu_apply))} times as fast. Batching therefore
neither helps nor hurts the end-to-end solve at these sizes.

Source: `pme_dst_tau_blend_batching_audit_{{gpu,cpu}}.json` [P7].

## II.5 Dense criterion and runtime estimator

The legacy dense criterion used the public defaults on code that predates the
{certificate_change_count} certificate changes of Part I. It and the current rerun
both use `N = {criterion_n}`.

{criterion_table}

Source: `pme_dst_tau_blend_pseudospectral_criterion.json` [P8].

The current-code rerun preserves every verdict. Old rate/predicted count to current
rate/predicted count: {current_comparison}. Source:
`pme_dst_tau_blend_pseudospectral_criterion_current.json`, `/records/*`.

At `m = {low_m}` the criterion abstains in both runs on the unmasked blend, as Part I's
active-block certificate does. At `m = {middle_m}` and `{high_m}` it certifies, with
current-code predictions {_sig(current_prediction_ratios[middle_m])} and
{_sig(current_prediction_ratios[high_m])} times the measured counts. The provisional
sparse estimator's costs are given in Part I, Section 10; none of its modes provides a
rate bound. Its non-certification of the frozen mean at `m = {high_m}`, at
{_sig(next(row['cost']['runtime_estimate_time_over_gmres'] for row in data['runtime']['records'] if row['method'] == 'frozen_mean'))}
times the cost of a solve, is the failure of a sufficient test, not evidence that
GMRES is slow.

## II.6 What changed from the previous version

- At `m = {low_m}` the regularized coefficient never vanishes
  (`D >= {_fmt(low_floor)}`); the previous statement that it vanishes on a set of
  positive measure holds only for the unregularized equation.
- "Every effective scalar keeps a tail" now holds at `m = {middle_m}` and
  `{high_m}` only; at `m = {low_m}`, references near the floor avoid the tail by not
  preconditioning.
- "The oracle is spectrally worst" holds at `m = {middle_m}` and `{high_m}` only;
  at `m = {low_m}` the oracle selects the floor.
- The blend's `kappa_2` at `m = {low_m}` is {_fmt(blend_kappa[0])}, the empty-mask
  value.
- The tight-tolerance time ratios are {_sig(min(oracle_time_ratios))} to
  {_sig(max(oracle_time_ratios))}, not the previous range.
- At loose tolerance the blend is slower at every `m`, including `m = {high_m}`.
- The batching speed-up is {_sig(min(gpu_apply))} to {_sig(max(gpu_apply))} on the
  GPU, not the previous range, and the iteration counts agree in
  {sum(row['iteration_sets_identical'] for row in gpu['solve_records'])} of
  {len(gpu['solve_records'])} solves, not all.
- The runtime estimator's cost is not a floor (Part I, Section 10).

The previous values are in git history.
"""
    return _wrap_markdown(result).rstrip()


def render(results_dir: Path = RESULTS_DIR) -> str:
    """Render Part I after fail-closed validation of all quoted evidence."""
    data = _load(results_dir)
    _assert_evidence(data)
    active = data["active"]
    cases = active["cases"]
    cfl = data["cfl"]
    cn = data["cn"]
    lemma = data["lemma"]
    mask = data["mask"]
    reference = data["reference"]
    repro = data["repro"]
    runtime = data["runtime"]

    delivery_config = mask["provenance"]["config"]
    dt = delivery_config["dt"]
    epsilon = delivery_config["epsilon"]
    x_min = delivery_config["x_min"]
    x_max = delivery_config["x_max"]
    nx_values = sorted({row["nx"] for row in cases})
    m_values = sorted({row["m"] for row in cases})
    case_count = len(cases)
    reference_count = len(next(row for row in cases if row["method"].endswith("B3"))["references"])
    eta_m2 = next(row["eta"] for row in cases if row["m"] == 2)
    eta_other = next(row["eta"] for row in cases if row["m"] == 4)

    lemma6 = lemma["lemma6"]
    lemma6_ratios = [row["lower_bound"] / row["measured_below_0p1"] for row in lemma6]
    lemma6_min_margin = min(row["margin"] for row in lemma6)
    lemma2_worst_1 = max(max(-row["margin_1"], 0.0) for row in lemma["lemma2"])
    lemma2_worst_inf = max(max(-row["margin_inf"], 0.0) for row in lemma["lemma2"])
    lemma3_rows = lemma["lemma3"]["rows"]
    lemma3_min_margin = min(row["minimum_margin"] for row in lemma3_rows)

    m2_cases = {(row["nx"], row["method"]): row for row in cases if row["m"] == 2}
    floor_records = cfl["floor"]
    floor_failed = next(row for row in floor_records if row["kind"] == "Z1" and row["eta"] == 0.1)
    floor_masked = [row for row in floor_records if row["kind"] == "Z1" and row["eta"] > 0.1]

    identity_rows = {row["m"]: row for row in lemma["identity_block_deviation"]}
    m2_deviations = [
        row["regularized_blocks"]["max_abs_E"]
        for row in cases
        if row["m"] == 2 and row["method"] == "one_inverse_cfl_mask"
    ]
    m2_sizes = [
        row["nx"] for row in cases if row["m"] == 2 and row["method"] == "one_inverse_cfl_mask"
    ]
    m2_scale_ratio = m2_deviations[-1] / m2_deviations[-2]
    m2_expected_ratio = ((m2_sizes[-1] + 1) / (m2_sizes[-2] + 1)) ** 2

    certified = [row for row in cases if _selected(row)["certified"]]
    bounds_to_measured = [
        _selected(row)["lemma1_full_operator_iteration_bound"] / row["gmres_J0"]["true_stop_count"]
        for row in certified
    ]
    min_factor_row = certified[bounds_to_measured.index(min(bounds_to_measured))]
    max_factor_row = certified[bounds_to_measured.index(max(bounds_to_measured))]

    configs = active["configuration"]
    public = inspect.signature(assess_dense_pseudospectral_criterion).parameters
    public_grid = public["grid_points_per_axis"].default
    public_padding = public["domain_padding_fraction"].default
    public_degrees = public["polynomial_degrees"].default
    public_epsilon_samples = public["epsilon_samples"].default
    public_target = public["target_tolerance"].default
    changes = {row["change"]: row["commit"] for row in configs["certificate_changes"]}

    method_counts = {
        method: sum(row["method"] == method and _selected(row)["certified"] for row in cases)
        for method in {"one_inverse_cfl_mask", "cfl_reference_blend_B3", "repository_blend_B3"}
    }
    reason_counts = {"connectivity": 0, "no polynomial": 0, "compute cap": 0}
    for row in cases:
        if not _selected(row)["certified"]:
            reason_counts[_abstention_label(_selected(row)["reason"])] += 1

    reference_rows = reference["rows"]
    by_reference = {(row["nx"], row["m"], row["reference_count"]): row for row in reference_rows}
    true_differences = [
        row["gmres_regularized"]["true_stop_count"]
        - row["gmres_regularized"]["preconditioned_stop_count"]
        for row in cases
    ]

    cn_mask_differences = {
        m: {row["operator_mask_symmetric_difference"] for row in cn["steps"] if row["m"] == m}
        for m in m_values
        if any(row["m"] == m for row in cn["steps"])
    }
    _require(
        all(len(values) == 1 for values in cn_mask_differences.values()),
        "CN mask difference is not constant across replay steps",
    )

    mask_differences = sum(not row["all_equal"] for row in repro["mask_analysis_subset"])
    e3_differences = sum(not row["all_equal"] for row in repro["drifted_E3_subset"])
    status_changes = [
        row for row in repro["drifted_E3_subset"] if not row["fields"]["steps_converged"]["equal"]
    ]
    _require(len(status_changes) == 1, "expected exactly one convergence-status change")
    status = status_changes[0]

    runtime_rows = [row for row in runtime["records"] if row["method"] == "tau_blend"]
    runtime_rows.sort(key=lambda row: row["m"])
    runtime_ratios = ", ".join(
        f"`m = {row['m']}`: {row['cost']['runtime_estimate_time_over_gmres']:.1f}x"
        for row in runtime_rows
    )

    lemma6_table = _lemma6_table(lemma6)
    m2_table = _m2_table(cases, mask)
    geometry_table = _active_geometry_table(cases)
    decision_table = _decision_table(cases)
    direct_table = _direct_table(active)
    reference_table = _reference_table(reference_rows)
    lemma4_table = _lemma4_table(lemma["lemma4"])
    cn_table = _cn_table(cn["summaries"])
    part_ii = _render_part_ii(data)

    nx_text = ", ".join(str(value) for value in nx_values)
    m_text = ", ".join(str(value) for value in m_values)
    public_degree_text = ", ".join(str(value) for value in public_degrees)
    study_degree_text = ", ".join(str(value) for value in configs["polynomial_degrees"])
    grid_text = ", ".join(str(value) for value in configs["grid_ladder"])
    mask_rule = (
        f"`D <= max({_fmt(REPOSITORY_ACTIVE_ABSOLUTE)}, "
        f"{_fmt(REPOSITORY_ACTIVE_RELATIVE)} max(D))`"
    )
    four_hour_cap = configs["case_cap_seconds"] / (60 * 60)
    abstention_count = case_count - len(certified)

    rendered = f"""# PME DST/tau-blend: mask analysis and certified preconditioner decisions

This report replaces the stale-flagged version of this file, whose numbers
predated the corrected diffusion coefficient. The earlier text remains in git
history. Part I covers the mask analysis and the certified decision procedure.
Part II restates the single-reference baseline study against the JSONs regenerated
with the corrected coefficient.

The lemmas cited below are stated and proved in `TAU_BLEND_MASK_LEMMAS.md`
(G. Pavlov). `TAU_BLEND_MASK_CLAIMS.md` (G. Pavlov) maps the numbers of the
earlier mask study to their scripts. `TAU_BLEND_MASK_FACTS.md` prints every
number in Part I with its source file and JSON pointer, and is regenerated from
the committed JSONs by `benchmarks/make_tau_blend_mask_facts.py`.

# Part I. Mask analysis and certified decisions

## 1. Scope and limits

- **Problem.** The one-dimensional, node-centred porous-medium equation with
  homogeneous Dirichlet boundaries on `[{_fmt(x_min)}, {_fmt(x_max)}]` and the
  regularized coefficient `D = m (u^2 + eps^2)^((m-1)/2)`,
  `eps = {_fmt(epsilon)}`. One backward-Euler linearization per state, with
  `dt = {_fmt(dt)}`. The active-block study uses `N = {nx_text}` and
  `m = {m_text}`.
- **Time integration.** All trajectory evidence uses backward Euler. One bounded
  Crank--Nicolson replay is reported in Section 8, and it is not a general claim
  about Crank--Nicolson.
- **Not covered.** IMEX--Strang and ETDRK4 avoid Newton solves and are not affected
  by this work. There is no multigrid baseline, so this report makes no performance
  claim against the solver state of the art. Part I reports iteration counts only,
  no wall times.
- **Counts.** All GMRES counts in Sections 3 to 7 are unrestarted GMRES stopped on
  the true relative residual `||b - J x_k|| <= 1e-8 ||b||`, on the regularized
  operator unless stated otherwise. Section 8 compares them with the
  preconditioned-residual convention. Exact counts can differ by one or a few
  iterations between hosts (Section 9).

Source: `tau_blend_mask_analysis.json`, `/provenance/config`; `tau_blend_active_block.json`,
`/configuration` [F1, F6].

### Preconditioners compared

All have the form `M = Q + sum_w H(d_w) W_w` of Pavlov's lemma setting, with
`H(d) = (I + dt d T)^-1` the DST-I Helmholtz inverse, `Q` the projector onto the
masked set `Z`, and diagonal input weights `W_w` that vanish on `Z`.

| Name | Mask | Helmholtz references |
|---|---|---|
| `repository_blend_B3` | repository rule {mask_rule} | {reference_count} active-support quantiles with Gaussian log-coefficient weights |
| `cfl_reference_blend_B3` | CFL mask `chi_j = 4 dt d_j / h^2 <= eta`, `eta = {_fmt(eta_m2)}` at `m = 2` and `{_fmt(eta_other)}` otherwise | {reference_count} quantiles of the CFL-active coefficients with the same weights |
| `one_inverse_cfl_mask` | the same CFL mask | one inverse at the active maximum |

At `m = 4` and `m = 8` the repository blend and the CFL blend select the same
nodes and references in the committed cross-checks, so the study has {case_count}
unique cases: the two CFL methods at all nine `(N, m)` pairs, and the repository
blend at `m = 2`.

Source: `tau_blend_mask_analysis.json`, `/states/*/repository_checks`;
`tau_blend_active_block.json`, `/cases/*` [F3, F6].

## 2. Why a scalar reference leaves a tail (Lemma 6)

Lemma 6 bounds the number of small eigenvalues of `H(d) J` from below, for any
scalar reference `d`, in terms of the modal CFL numbers of the region where the
coefficient sits at its floor:

`#{{lambda(H(d) J) < a}} >= #{{beta_k < a}} - s`, with
`beta_k = (1 + dt f mu_k) / (1 + dt d mu_k)`.

At level `a = 0.1` the bound holds on all {len(lemma6)} scalar-reference records,
with a minimum margin of {lemma6_min_margin} eigenvalues. The bound accounts for
{100 * min(lemma6_ratios):.1f}% to {100 * max(lemma6_ratios):.1f}% of the measured
tail (bound/measured from {_fmt(min(lemma6_ratios))} to
{_fmt(max(lemma6_ratios))}). The tail is therefore structural: it is forced by
the floor region and the CFL numbers, not by a poor choice of `d`.

{lemma6_table}

Source: `tau_blend_lemma_checks.json`, `/lemma6/*` [F2].

## 3. The mask at m = 2 under the corrected coefficient

Lemma 5 gives every node the floor CFL number
`chi_floor = 4 dt m eps^(m-1) / h^2`. With the corrected coefficient,
`D >= {_fmt(next(row['diffusivity_floor'] for row in lemma6 if row['m'] == 2))}`
at `m = 2`, which lies above the repository threshold {mask_rule}, so the
repository mask selects **no** node at `m = 2`. The CFL mask selects
{m2_cases[(512, 'one_inverse_cfl_mask')]['inactive_count']} and
{m2_cases[(1024, 'one_inverse_cfl_mask')]['inactive_count']} nodes at `N = 512`
and `N = 1024`, exactly the floor nodes (claims map row D17).

{m2_table}

Source: `tau_blend_active_block.json`, `/cases/*`;
`tau_blend_mask_analysis.json`, `/states/*` [F3].

Without a mask, the repository blend's active-block spectrum spreads to maximum
real parts {m2_cases[(512, 'repository_blend_B3')]['Y_spectrum']['maximum_real']:.1f}
and {m2_cases[(1024, 'repository_blend_B3')]['Y_spectrum']['maximum_real']:.1f},
and GMRES needs {m2_cases[(512, 'repository_blend_B3')]['gmres_regularized']['true_stop_count']}
and {m2_cases[(1024, 'repository_blend_B3')]['gmres_regularized']['true_stop_count']}
iterations. The CFL blend restores a compact spectrum (maximum real parts
{m2_cases[(512, 'cfl_reference_blend_B3')]['Y_spectrum']['maximum_real']:.2f}
and {m2_cases[(1024, 'cfl_reference_blend_B3')]['Y_spectrum']['maximum_real']:.2f})
and needs {m2_cases[(512, 'cfl_reference_blend_B3')]['gmres_regularized']['true_stop_count']}
and {m2_cases[(1024, 'cfl_reference_blend_B3')]['gmres_regularized']['true_stop_count']}
iterations. At `N = 128` the counts are
{m2_cases[(128, 'repository_blend_B3')]['gmres_regularized']['true_stop_count']}
(repository) and
{m2_cases[(128, 'cfl_reference_blend_B3')]['gmres_regularized']['true_stop_count']}
(CFL blend).

**Limit (Lemma 5).** At fixed `dt`, `chi_floor` grows as `h^-2`, so any fixed
cutoff `eta` eventually loses the mask under refinement. Pavlov's floor case
shows this: at `m = {floor_failed['m']}`, `N = {floor_failed['nx']}`,
`dt = {_fmt(floor_failed['dt'])}`, `chi_floor = {floor_failed['floor_cfl']:.9g}`
exceeds `eta = {_fmt(floor_failed['eta'])}`, the mask is empty, and the
one-inverse solve does not reach tolerance within {floor_failed['gmres_max']}
iterations; at `eta = {_fmt(floor_masked[0]['eta'])}` or
`{_fmt(floor_masked[-1]['eta'])}` the mask holds {floor_masked[0]['masked']} nodes.
A mesh-robust cutoff rule is not proposed here (claims map rows C10 to C12).

Source: `tau_blend_cfl_fidelity.json`, `/floor/*` [F13].

## 4. Exact deflation and its regularized perturbation (Lemmas 1, 2 and 5)

In the node order `(Z, A)`, write `M J = [[I + E, X], [F, Y]]`. For `J_0`,
which sets the coefficient to zero on `Z`, Lemma 1 gives `E = F = 0` exactly;
the measurements confirm `E_0 = F_0 = 0` in all {case_count} cases. The active
block `Y` is byte-identical between `J_0` and the regularized `J` in all
{case_count} cases.

For the regularized `J`, the identity block deviates by the floor CFL term. The
measured maximum deviation equals `2 dt d_floor / h^2` to
`{identity_rows[2]['absolute_difference']:.2g}` at `N = {identity_rows[2]['nx']},
m = 2`, and to `{identity_rows[4]['absolute_difference']:.2g}` at
`N = {identity_rows[4]['nx']}, m = 4`. It scales with `(N + 1)^2`: at `m = 2`
it is
`{_sci(m2_deviations[0], 8)}`, `{_sci(m2_deviations[1], 8)}` and
`{_sci(m2_deviations[2], 8)}` at `N = {m2_sizes[0]}, {m2_sizes[1]}, {m2_sizes[2]}`,
and the ratio between the last two is {m2_scale_ratio:.3f}, equal to
`(({m2_sizes[-1]} + 1)/({m2_sizes[-2]} + 1))^2 = {m2_expected_ratio:.3f}`. At
`m = 8` the predicted deviation is of order
`{_sci(identity_rows[8]['expected_2_dt_floor_over_h2'], 2)}`,
below the double-precision resolution of the unit diagonal; the measured maximum
is then about half the prediction, consistent with an off-diagonal entry surviving
while the diagonal increment rounds away (an interpretation, not a separate
measurement).

Lemma 2 bounds `||J - J_0||_p <= eta_Z`. Over {len(lemma['lemma2'])} records there
are no violations; the worst excess is `{lemma2_worst_1:.2g}` in the 1-norm and
`{lemma2_worst_inf:.2g}` in the infinity norm, which is round-off. In several
records `||J - J_0||_1` equals `eta_Z` to rounding, so the 1-norm bound is attained
on these states.

Source: `tau_blend_active_block.json`, `/cases/*/J0_blocks`,
`/cases/*/regularized_blocks`; `tau_blend_lemma_checks.json`, `/lemma2`,
`/identity_block_deviation` [F4, F10].

## 5. From the active block to a GMRES bound (Lemma 1', ours)

**Statement.** Let `M J_0 = [[I, X], [0, Y]]` and let `q` be any polynomial with
`q(0) = 1`. Then `p(z) = (1 - z) q(z)` satisfies `p(0) = 1` and

`p(M J_0) = [[0, -X q(Y)], [0, (I - Y) q(Y)]]`,

so for every right-hand side, GMRES on `M J_0` satisfies

`||r_{{k+1}}|| / ||r_0|| <= C min_{{q(0)=1, deg q <= k}} ||q(Y)||_2`, with
`C = ||[X; I - Y]||_2`.

**Proof.** `I - M J_0 = [[0, -X], [0, I - Y]]`, and `q(M J_0)` is block upper
triangular with diagonal blocks `q(1) I` and `q(Y)`. Multiplying gives the
displayed form. GMRES minimizes over all residual polynomials of degree `k + 1`,
so its residual is at most `||p(M J_0)|| ||r_0||` for this `p`.

The bound is a worst-case (ideal-GMRES) bound: it holds for every right-hand side
and can be pessimistic for a particular one. It is exact for `J_0` only; for the
regularized `J` this report certifies the full operator directly (Section 6). The
block identity is checked numerically for random `q` in
`tests/test_tau_blend_active_block.py`.

{geometry_table}

Source: `tau_blend_active_block.json`, `/cases/*` [F5].

When the mask is empty (the repository blend at `m = 2`), `Y = M J` and
`C = ||I - M J||_2`.

## 6. Decision procedure: disk test plus pseudospectral certificate

### The certificate

For a closed contour `Gamma` that encloses `sigma(A)` and excludes the origin
(the boundary of the grid cells of a connected epsilon-pseudospectral component),
let `s_Gamma` be a proven lower bound for `sigma_min(zI - A)` on `Gamma` and
`M_Gamma` a proven upper bound for `|p|` on `Gamma`, for a polynomial `p` of
degree `d` with `p(0) = 1`. The Cauchy integral gives
`||p(A)^j|| <= L(Gamma) / (2 pi s_Gamma) * M_Gamma^j` [Trefethen--Embree]. The
certified rate is `rho = M_Gamma^(1/d)`, and the prefactor is
`L(Gamma) / (2 pi s_Gamma)`.

`s_Gamma` comes from the grid samples of `sigma_min`, its 1-Lipschitz property
along each edge, and an SVD backward-error margin. `M_Gamma` is the sampled
maximum of `|p|` plus three allowances: a derivative enclosure times half the
sample spacing, the floating-point error of evaluating `p`, and, when a centred
basis is used, the error of translating the coefficients. Three changes on this
branch tighten or correct the bound:

1. **Centred derivative enclosure** ({changes['centred derivative enclosure and numerical-range domain fallback']}).
   The enclosure is the minimum of the origin-centred
   `sum_j j |a_j| R^(j-1)` and the same sum in powers of `z - c`, with `c` the
   centre of the spectral bounding box. The sample points are unchanged, so the
   bound can only decrease.
2. **Search-domain fallback** ({changes['centred derivative enclosure and numerical-range domain fallback']}).
   If a level set touches the edge of the search box, the box is enlarged once
   to an outer rectangle of `W(A) + D(0, eps)`, which contains the
   epsilon-pseudospectrum because `sigma_min(zI - A) >= dist(z, W(A))`. The grid
   spacing does not increase.
3. **Floating-point evaluation of `p`** ({changes['Horner floating-point evaluation allowance']}).
   The previous bound covered the variation between samples but not the rounding
   error of the sampled values. The Horner allowance
   `gamma_{{2n}} sum_i |a_i| |z|^i` [Higham] is now added, evaluated in whichever
   of the origin and centred bases gives the tighter valid bound. The term grows
   with the degree and the size of the coefficients; a degree-40 regression test
   with large coefficients exercises a case where it dominates.

Monotonicity and validity are tested in
{changes['soundness and monotonicity validation for the tightened certificate']}:
the derivative bound is never below the sampled `max |p'|`; the new contour bound
never exceeds the old one; `sigma_min(zI - A) >= dist(z, W(A))` holds on random
non-normal matrices. The study settings differ from the public defaults: grids
{grid_text}; degrees {study_degree_text}; {configs['epsilon_samples']} epsilon
samples; target `{_fmt(configs['target_tolerance'])}`. The public defaults are grid
{public_grid}, domain-padding fraction {_fmt(public_padding)}, degrees
{public_degree_text}, {public_epsilon_samples} epsilon samples and target
`{_fmt(public_target)}` (source:
`moljax/experimental/pseudospectral_criterion.py`,
`assess_dense_pseudospectral_criterion` signature).

Source: `tau_blend_active_block.json`, `/configuration` [F6].

### The merged decision

A case counts as certified if the active-block (`Y`) certificate certifies, or if
the merged dense disk policy returns `adequate`. Only `adequate` counts from the
disk policy; `provisional`, `investigate` and `indeterminate` do not.

{decision_table}

Source: `tau_blend_active_block.json`, `/cases/*`, `/decision_summary` [F6].

Abstention reasons: "connectivity" means the pseudospectral component that encloses
the spectrum reaches the origin-exclusion threshold; "no polynomial" means the
connected window has no polynomial with a proven contour bound below one; "compute
cap" means the estimated cumulative cost of the next grid exceeds the
{_fmt(four_hour_cap)}-hour per-case cap.

**Counts.** The `Y` certificate certifies
{active['decision_summary']['Y_certified_count']} of the {case_count} cases. The
disk policy returns `adequate` for {active['decision_summary']['disk_adequate_count']},
both already certified, so the merged procedure also certifies
{active['decision_summary']['union_certified_count']} of {case_count}. By method:
{method_counts['one_inverse_cfl_mask']} of 9 for the one-inverse mask,
{method_counts['cfl_reference_blend_B3']} of 9 for the CFL blend, and
{method_counts['repository_blend_B3']} of 3 for the repository blend at `m = 2`.
The {abstention_count} abstentions are {reason_counts['connectivity']} for connectivity,
{reason_counts['no polynomial']} for no polynomial and
{reason_counts['compute cap']} for the compute cap.

**Ordering.** In every `(N, m)` group with at least one certified method, the
method with the fewest GMRES iterations is certified, and no certified method is
slower than an abstaining one. Where two or more methods certify, the smaller
certified rate belongs to the method with fewer iterations; at `N = 128, m = 2`
the rates of the repository blend and the one-inverse mask differ by
{abs(_selected(m2_cases[(128, 'repository_blend_B3')])['rho'] - _selected(m2_cases[(128, 'one_inverse_cfl_mask')])['rho']):.3f},
a tie under the {_fmt(configs['rate_tie_threshold'])} tie rule. At `N = 1024,
m = 4` no method certifies.

**Conservatism.** The Lemma 1' bound is `1 + d j*`, with `j*` the smallest
integer for which `C * prefactor * rho^(d j) <= 1e-8`. Against the measured
`J_0` counts, it exceeds them by a factor of {min(bounds_to_measured):.1f}
(`N = {min_factor_row['nx']}, m = {min_factor_row['m']}`, CFL blend:
{_selected(min_factor_row)['lemma1_full_operator_iteration_bound']} against
{min_factor_row['gmres_J0']['true_stop_count']}) to {max(bounds_to_measured):.1f}
(`N = {max_factor_row['nx']}, m = {max_factor_row['m']}`, CFL blend:
{_selected(max_factor_row)['lemma1_full_operator_iteration_bound']} against
{max_factor_row['gmres_J0']['true_stop_count']}). The bounds are valid and
conservative; they separate the methods in the measured order.

### Direct certificates for the regularized operator

Lemma 1' is exact for `J_0` only. For the CFL blend at `m = 2`, the full
regularized operator `M J` is certified directly:

{direct_table}

Source: `tau_blend_active_block.json`, `/direct_regularized_full_MJ_certificates/*` [F6].

At `N = 128` the domain fallback produced the selected grid
{configs['full_operator_selected_grid_N128']}, with the grid spacing kept or
reduced. The committed JSON does not record a distinct pre-fallback starting grid.

## 7. Limits of the result

- **Mesh dependence.** The CFL blend's counts grow under refinement:
  {m2_cases[(128, 'cfl_reference_blend_B3')]['gmres_regularized']['true_stop_count']},
  {m2_cases[(512, 'cfl_reference_blend_B3')]['gmres_regularized']['true_stop_count']},
  {m2_cases[(1024, 'cfl_reference_blend_B3')]['gmres_regularized']['true_stop_count']}
  at `m = 2`;
  {next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 128 and row['m'] == 4 and row['method'] == 'cfl_reference_blend_B3')},
  {next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 512 and row['m'] == 4 and row['method'] == 'cfl_reference_blend_B3')},
  {next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 1024 and row['m'] == 4 and row['method'] == 'cfl_reference_blend_B3')}
  at `m = 4`; and
  {next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 128 and row['m'] == 8 and row['method'] == 'cfl_reference_blend_B3')},
  {next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 512 and row['m'] == 8 and row['method'] == 'cfl_reference_blend_B3')},
  {next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 1024 and row['m'] == 8 and row['method'] == 'cfl_reference_blend_B3')}
  at `m = 8`, for `N = {nx_text}`. At `N = 1024, m = 4` nothing certifies.
- **Certified is not fast.** The one-inverse mask certifies at rates
  {min(_selected(row)['rho'] for row in certified if row['method'] == 'one_inverse_cfl_mask' and row['nx'] >= 512):.3f}
  to {max(_selected(row)['rho'] for row in certified if row['method'] == 'one_inverse_cfl_mask' and row['nx'] >= 512):.3f}
  at `N >= 512`, with
  {min(row['gmres_regularized']['true_stop_count'] for row in certified if row['method'] == 'one_inverse_cfl_mask' and row['nx'] >= 512)}
  to {max(row['gmres_regularized']['true_stop_count'] for row in certified if row['method'] == 'one_inverse_cfl_mask' and row['nx'] >= 512)} iterations; the CFL blend certifies at
  {min(_selected(row)['rho'] for row in certified if row['method'] == 'cfl_reference_blend_B3' and row['nx'] == 512):.3f}
  to {max(_selected(row)['rho'] for row in certified if row['method'] == 'cfl_reference_blend_B3' and row['nx'] == 512):.3f}
  at `N = 512`, with
  {min(row['gmres_regularized']['true_stop_count'] for row in certified if row['method'] == 'cfl_reference_blend_B3' and row['nx'] == 512)}
  to {max(row['gmres_regularized']['true_stop_count'] for row in certified if row['method'] == 'cfl_reference_blend_B3' and row['nx'] == 512)} iterations. The procedure decides whether a preconditioner is guaranteed to work and ranks the certified ones; it does not make a slow one fast.
- **Reference count.** For the CFL blend, three references give the fewest
  iterations at `N = 512` ({by_reference[(512, 4, 3)]['gmres']['true_stop_count']}
  at `m = 4`, {by_reference[(512, 8, 3)]['gmres']['true_stop_count']} at `m = 8`).
  At `N = 1024` four references give
  {by_reference[(1024, 4, 4)]['gmres']['true_stop_count']} at `m = 4` and
  {by_reference[(1024, 8, 4)]['gmres']['true_stop_count']} at `m = 8`, against
  {by_reference[(1024, 4, 3)]['gmres']['true_stop_count']} and
  {by_reference[(1024, 8, 3)]['gmres']['true_stop_count']} for three, and five or
  six references are worse than four. The dependence is not monotone, and no rule
  for choosing the count is proposed.

{reference_table}

Source: `tau_blend_reference_count.json`, `/rows/*` [F7].

- **Dimension and baselines.** One-dimensional only; no multigrid comparison;
  backward Euler only for trajectories.

## 8. Fidelity: what the counts and trajectories mean (Lemmas 3 and 4)

**Stopping convention (Lemma 4).** A test on the left-preconditioned residual
bounds the true residual only up to `kappa(M)`. At `N = 512`, `dt = {_fmt(dt)}`:

{lemma4_table}

`kappa_2(M)`; source: `tau_blend_lemma_checks.json`, `/lemma4/*` [F9]. The `Z1`
and `Zf` rows use `eta = {_fmt(eta_other)}`; at `m = 2` the values at
`eta = {_fmt(eta_m2)}` are identical.

In all {case_count} active-block cases the true-residual count exceeds the
preconditioned-residual count, by {min(true_differences)} to
{max(true_differences)} iterations; for example
{m2_cases[(1024, 'repository_blend_B3')]['gmres_regularized']['true_stop_count']}
against
{m2_cases[(1024, 'repository_blend_B3')]['gmres_regularized']['preconditioned_stop_count']}
for the repository blend at `N = 1024, m = 2`, and
{next(row['gmres_regularized']['true_stop_count'] for row in cases if row['nx'] == 512 and row['m'] == 4 and row['method'] == 'one_inverse_cfl_mask')}
against
{next(row['gmres_regularized']['preconditioned_stop_count'] for row in cases if row['nx'] == 512 and row['m'] == 4 and row['method'] == 'one_inverse_cfl_mask')}
for the one-inverse mask at `N = 512, m = 4`. The preconditioned counts of the
CFL blend at `m = 2` (
{m2_cases[(512, 'cfl_reference_blend_B3')]['gmres_regularized']['preconditioned_stop_count']}
and
{m2_cases[(1024, 'cfl_reference_blend_B3')]['gmres_regularized']['preconditioned_stop_count']}
at `N = 512, 1024`) equal those of Pavlov's mask analysis (claims map row D20).

Source: `tau_blend_active_block.json`, `/cases/*/gmres_regularized` [F9].

**A posteriori trajectory bound (Lemma 3).** Backward Euler for the PME is
nonexpansive in the 1-norm, so
`||v^n - u^n||_1 <= ||v^0 - u^0||_1 + sum_k ||rho^k||_1` holds for every
preconditioner and cutoff. It holds on all {len(lemma3_rows)} trajectories,
including the {lemma['lemma3']['failed_step_trajectories']} that contain failed
Newton steps, with no violation and a minimum margin of
`{lemma3_min_margin:.2g}`.

Source: `tau_blend_lemma_checks.json`, `/lemma3` [F8].

**Crank--Nicolson replay.** One bounded replay at `N = {cn['summaries'][0]['nx']}`,
`dt = {_fmt(cn['summaries'][0]['dt'])}`, {cn['protocol']['steps']} steps. The CN
active set is smaller than the BE one by
{next(iter(cn_mask_differences[2]))} nodes at `m = 2` and
{next(iter(cn_mask_differences[4]))} at `m = 4`, at every step. Both integrators
converge.

{cn_table}

Source: `tau_blend_cn_replay.json`, `/steps`, `/summaries` [F11]. This replay
supports no general statement about CN; all trajectory evidence elsewhere is
backward Euler.

## 9. Reproducibility on a second host

Pavlov's delivery (`research/tau-blend-repro` at
{configs['rewrite_base_revision']}) was re-run on a second host with the
same JAX version ({repro['provenance']['jax']}). Convergence outcomes are preserved
except in one row; exact counts move slightly.

- Mask analysis: {mask_differences} of {len(repro['mask_analysis_subset'])} rows
  differ, each by one GMRES iteration.
- E3 trajectories: {e3_differences} of {len(repro['drifted_E3_subset'])} rows differ
  in at least one Newton or linear total.
- One convergence-status change: `m = {status['m']}`, `N = {status['nx']}`,
  `dt = {_fmt(status['dt'])}`, repository blend, {status['side']} preconditioning,
  forcing {_fmt(status['forcing'])}. The delivery converges
  {status['fields']['steps_converged']['delivered']} steps
  ({status['fields']['newton_total']['delivered']} Newton,
  {status['fields']['linear_iterations_total']['delivered']} linear iterations);
  this host converges {status['fields']['steps_converged']['regenerated']}, with one
  line-search failure ({status['fields']['newton_total']['regenerated']} Newton,
  {status['fields']['linear_iterations_total']['regenerated']} linear).

Source: `tau_blend_reproducibility.json`, `/mask_analysis_subset`,
`/drifted_E3_subset` [F12].

## 10. Cost of the certificate, and a corrected claim

The provisional matrix-free runtime estimator (`rate_bound_available=false`)
records these cost ratios relative to one GMRES solve: {runtime_ratios}. These
are not dense-certificate costs. An
earlier version of this report called the range a "fundamental evidence floor".
That claim is withdrawn: no result here establishes a lower bound on the cost of
certification, and an abstention is only the failure of a sufficient test. The
cost is a measured property of that implementation, not a dense-certificate
complexity bound.

Source: `pme_dst_tau_blend_pseudospectral_runtime_estimator.json`, `/records/*/cost`
[II.5].

On the active-block study, the estimated cost of the next grid exceeded the
{_fmt(four_hour_cap)}-hour per-case cap once (`N = 1024, m = 2`, repository
blend), which is reported as an abstention.

{part_ii}

## References

1. G. Pavlov, "moljax: GPU-accelerated method of lines for stiff
   reaction-diffusion PDEs with FFT preconditioning," *Computer Physics
   Communications* 326 (2026) 110205. DOI `10.1016/j.cpc.2026.110205`.
2. L. N. Trefethen and M. Embree, *Spectra and Pseudospectra* (Princeton,
   2005). DOI `10.1515/9780691213101`.
3. M. Embree, "How Descriptive are GMRES Convergence Bounds?",
   arXiv:2209.01231. DOI `10.48550/arXiv.2209.01231`.
4. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
   (SIAM, 2002). DOI `10.1137/1.9780898718027`.
"""
    return _wrap_markdown(rendered)


def main() -> None:
    """Write the deterministic report from committed measurements."""
    OUTPUT.write_text(render(), encoding="utf-8")
    print(OUTPUT.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
