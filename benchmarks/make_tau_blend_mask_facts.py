#!/usr/bin/env python3
"""Render the tau-blend mask facts sheet from committed evidence only."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
RESULTS = REPOSITORY / "benchmarks" / "results"
OUTPUT = RESULTS / "TAU_BLEND_MASK_FACTS.md"

SOURCES = {
    "active": "tau_blend_active_block.json",
    "cn": "tau_blend_cn_replay.json",
    "delivery_cfl": "tau_blend_cfl_fidelity.json",
    "delivery_mask": "tau_blend_mask_analysis.json",
    "lemma": "tau_blend_lemma_checks.json",
    "reference": "tau_blend_reference_count.json",
    "repro": "tau_blend_reproducibility.json",
}


@dataclass(frozen=True, order=True)
class JsonReference:
    """One file and RFC 6901 pointer printed in the facts sheet."""

    source: str
    pointer: str


def _pointer(document: Any, pointer: str) -> Any:
    """Resolve one RFC 6901 JSON pointer."""
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
    """Return compact, deterministic Markdown-safe text."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.8g}"
    text = str(value).replace("|", "\\|").replace("\n", " ")
    return text


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(row) + " |" for row in rows),
        ]
    )


class Facts:
    """Loaded committed evidence plus references emitted while rendering."""

    def __init__(self) -> None:
        self.documents = {
            key: json.loads((RESULTS / filename).read_text(encoding="utf-8"))
            for key, filename in SOURCES.items()
        }
        self.references: set[JsonReference] = set()

    def cite(self, source: str, pointer: str, value: Any | None = None) -> str:
        """Format one value with its source file and existing JSON path."""
        reference = JsonReference(source, pointer)
        actual = _pointer(self.documents[source], pointer)
        self.references.add(reference)
        shown = actual if value is None else value
        filename = SOURCES[source]
        return f"{_display(shown)} ⟦`{filename}#{pointer}`⟧"

    def derived(self, value: Any, source: str, pointer: str, operation: str) -> str:
        """Format a derived value and name the source collection and operation."""
        _pointer(self.documents[source], pointer)
        self.references.add(JsonReference(source, pointer))
        return f"{_display(value)} ⟦`{SOURCES[source]}#{pointer}`; " f"derived: {operation}⟧"

    def validate(self) -> None:
        """Raise if any printed JSON pointer does not resolve."""
        for reference in self.references:
            _pointer(self.documents[reference.source], reference.pointer)


def _claims_rows() -> list[tuple[str, str, str]]:
    """Read row id, output field and regenerated value from Pavlov's claims map."""
    text = (RESULTS / "TAU_BLEND_MASK_CLAIMS.md").read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not re.match(r"^\| [DCR]\d+ \|", line):
            continue
        protected = line.replace("||X||", "NORM_X").replace("||J - J0||", "NORM_J_J0")
        cells = [cell.strip() for cell in protected.strip().strip("|").split("|")]
        cells = [
            cell.replace("NORM_X", "||X||").replace("NORM_J_J0", "||J - J0||") for cell in cells
        ]
        rows.append((cells[0], cells[4], cells[-1]))
    return rows


def render_facts() -> tuple[str, tuple[JsonReference, ...]]:
    """Render the facts sheet and return every emitted JSON reference."""
    facts = Facts()
    active = facts.documents["active"]
    cn = facts.documents["cn"]
    lemma = facts.documents["lemma"]
    reference = facts.documents["reference"]
    repro = facts.documents["repro"]
    lines = [
        "# Tau-blend mask rewrite: committed facts",
        "",
        "This file is generated from committed JSON evidence. Every displayed number is "
        "followed by its source file and JSON pointer. Aggregates name the source collection "
        "and operation. No scratch artifact is read.",
        "",
        "## F1 — provenance",
        "",
        f"Rewrite base: {facts.cite('active', '/configuration/rewrite_base_revision')}",
        "",
    ]

    provenance_rows = []
    for source in SOURCES:
        document = facts.documents[source]
        pointer = "/provenance"
        provenance = document.get("provenance")
        if provenance is None and source == "repro":
            provenance = document["provenance"]
        if source.startswith("delivery_"):
            head_key = "repository_head"
            backend_key = "jax_backend"
            jaxlib = "not recorded"
            scipy = "not recorded"
        else:
            head_key = "git_commit"
            backend_key = "backend"
            jaxlib = facts.cite(source, f"{pointer}/jaxlib")
            scipy = facts.cite(source, f"{pointer}/scipy")
        provenance_rows.append(
            [
                SOURCES[source],
                facts.cite(source, f"{pointer}/{head_key}"),
                facts.cite(source, f"{pointer}/python"),
                facts.cite(source, f"{pointer}/jax"),
                jaxlib,
                facts.cite(source, f"{pointer}/numpy"),
                scipy,
                facts.cite(source, f"{pointer}/{backend_key}"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "JSON",
                    "execution commit",
                    "Python",
                    "JAX",
                    "jaxlib",
                    "NumPy",
                    "SciPy",
                    "backend",
                ],
                provenance_rows,
            ),
            "",
            "## F2 — Lemma 6 tail bound",
            "",
        ]
    )
    lemma6_rows = []
    ratios = []
    for index, row in enumerate(lemma["lemma6"]):
        root = f"/lemma6/{index}"
        ratios.append(row["lower_bound"] / row["measured_below_0p1"])
        lemma6_rows.append(
            [
                facts.cite("lemma", f"{root}/nx"),
                facts.cite("lemma", f"{root}/m"),
                facts.cite("lemma", f"{root}/case"),
                facts.cite("lemma", f"{root}/lower_bound"),
                facts.cite("lemma", f"{root}/measured_below_0p1"),
                facts.cite("lemma", f"{root}/margin"),
                facts.derived(ratios[-1], "lemma", root, "lower_bound / measured"),
                facts.cite("lemma", f"{root}/passes"),
            ]
        )
    min_margin = min(row["margin"] for row in lemma["lemma6"])
    lines.extend(
        [
            _table(
                ["N", "m", "record", "bound", "measured", "margin", "bound/measured", "passes"],
                lemma6_rows,
            ),
            "",
            f"Minimum margin: {facts.derived(min_margin, 'lemma', '/lemma6', 'minimum margin')}. ",
            f"Bound/measured range: {facts.derived(min(ratios), 'lemma', '/lemma6', 'minimum ratio')} "
            f"to {facts.derived(max(ratios), 'lemma', '/lemma6', 'maximum ratio')}.",
            "",
            "## F3 — m=2 masks and regularized work",
            "",
        ]
    )
    delivery_states = {
        (row["nx"], row["m"]): index
        for index, row in enumerate(facts.documents["delivery_mask"]["states"])
    }
    m2_rows = []
    for index, row in enumerate(active["cases"]):
        if row["m"] != 2:
            continue
        root = f"/cases/{index}"
        state_index = delivery_states.get((row["nx"], 2))
        chi_floor = (
            "not recorded at N=128"
            if state_index is None
            else facts.cite("delivery_mask", f"/states/{state_index}/floor_cfl")
        )
        m2_rows.append(
            [
                facts.cite("active", f"{root}/nx"),
                facts.cite("active", f"{root}/method"),
                facts.cite("active", f"{root}/inactive_count"),
                chi_floor,
                facts.cite("active", f"{root}/eta"),
                facts.cite("active", f"{root}/Y_spectrum/maximum_real"),
                facts.cite("active", f"{root}/MJ_spectrum/maximum_real"),
                facts.cite("active", f"{root}/gmres_regularized/true_stop_count"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "N",
                    "method",
                    "masked",
                    "chi_floor",
                    "eta",
                    "max Re sigma(Y)",
                    "max Re sigma(MJ)",
                    "regularized GMRES",
                ],
                m2_rows,
            ),
            "",
            "## F4 — block structure",
            "",
        ]
    )
    block_rows = []
    for index, _row in enumerate(active["cases"]):
        root = f"/cases/{index}"
        block_rows.append(
            [
                facts.cite("active", f"{root}/nx"),
                facts.cite("active", f"{root}/m"),
                facts.cite("active", f"{root}/method"),
                facts.cite("active", f"{root}/J0_blocks/max_abs_E"),
                facts.cite("active", f"{root}/J0_blocks/max_abs_F"),
                facts.cite("active", f"{root}/regularized_blocks/max_abs_E"),
                facts.cite("active", f"{root}/regularized_blocks/max_abs_F"),
                facts.cite("active", f"{root}/regularized_blocks/Y_byte_identical_to_J0"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "N",
                    "m",
                    "method",
                    "||E0||max",
                    "||F0||max",
                    "||E||max",
                    "||F||max",
                    "Y identical",
                ],
                block_rows,
            ),
            "",
            "The regularized identity-block deviation and its `(N+1)^2` prediction are:",
            "",
            _table(
                ["N", "m", "measured deviation", "2 dt d_floor / h^2", "absolute difference"],
                [
                    [
                        facts.cite("lemma", f"/identity_block_deviation/{i}/nx"),
                        facts.cite("lemma", f"/identity_block_deviation/{i}/m"),
                        facts.cite(
                            "lemma",
                            f"/identity_block_deviation/{i}/max_abs_identity_block_deviation",
                        ),
                        facts.cite(
                            "lemma", f"/identity_block_deviation/{i}/expected_2_dt_floor_over_h2"
                        ),
                        facts.cite("lemma", f"/identity_block_deviation/{i}/absolute_difference"),
                    ]
                    for i in range(len(lemma["identity_block_deviation"]))
                ],
            ),
            "",
            "## F5 — active-block conditioning",
            "",
            _table(
                ["N", "m", "method", "sigma_min(Y)", "C = ||[X; I-Y]||2"],
                [
                    [
                        facts.cite("active", f"/cases/{i}/nx"),
                        facts.cite("active", f"/cases/{i}/m"),
                        facts.cite("active", f"/cases/{i}/method"),
                        facts.cite("active", f"/cases/{i}/Y_spectrum/sigma_min"),
                        facts.cite("active", f"/cases/{i}/J0_blocks/lemma1_constant"),
                    ]
                    for i in range(len(active["cases"]))
                ],
            ),
            "",
            "## F6 — certificates and merged decision",
            "",
            f"Unique cases: {facts.cite('active', '/decision_summary/case_count')}; Y-certified: "
            f"{facts.cite('active', '/decision_summary/Y_certified_count')}; union-certified: "
            f"{facts.cite('active', '/decision_summary/union_certified_count')}. The union rule is "
            f"{facts.cite('active', '/decision_summary/union_positive_rule')}.",
            "",
        ]
    )
    certificate_rows = []
    for index, row in enumerate(active["cases"]):
        root = f"/cases/{index}"
        selected = row["Y_certificate_selected"]
        certificate_rows.append(
            [
                facts.cite("active", f"{root}/nx"),
                facts.cite("active", f"{root}/m"),
                facts.cite("active", f"{root}/method"),
                facts.cite("active", f"{root}/merged_policy/verdict"),
                facts.cite("active", f"{root}/merged_policy/disk_rate"),
                facts.cite("active", f"{root}/Y_certificate_selected/certified"),
                facts.cite("active", f"{root}/Y_certificate_selected/reason"),
                facts.cite("active", f"{root}/Y_certificate_selected/grid"),
                (
                    facts.cite("active", f"{root}/Y_certificate_selected/degree")
                    if "degree" in selected
                    else "—"
                ),
                (
                    facts.cite("active", f"{root}/Y_certificate_selected/rho")
                    if "rho" in selected
                    else "—"
                ),
                (
                    facts.cite("active", f"{root}/Y_certificate_selected/prefactor")
                    if "prefactor" in selected
                    else "—"
                ),
                (
                    facts.cite(
                        "active",
                        f"{root}/Y_certificate_selected/lemma1_full_operator_iteration_bound",
                    )
                    if "lemma1_full_operator_iteration_bound" in selected
                    else "—"
                ),
                facts.cite("active", f"{root}/gmres_J0/true_stop_count"),
                facts.cite("active", f"{root}/gmres_regularized/true_stop_count"),
                facts.cite("active", f"{root}/union"),
            ]
        )
    lines.extend(
        [
            _table(
                [
                    "N",
                    "m",
                    "method",
                    "disk",
                    "disk rate",
                    "Y cert",
                    "reason",
                    "grid",
                    "degree",
                    "rho",
                    "prefactor",
                    "Lemma 1' bound",
                    "GMRES J0",
                    "GMRES regularized",
                    "union",
                ],
                certificate_rows,
            ),
            "",
            "Direct regularized full-MJ certificates:",
            "",
        ]
    )
    direct_rows = []
    for index, row in enumerate(active["direct_regularized_full_MJ_certificates"]):
        selected_index = next(
            i for i, item in enumerate(row["certificate_ladder"]) if item["certified"]
        )
        root = f"/direct_regularized_full_MJ_certificates/{index}"
        cert = f"{root}/certificate_ladder/{selected_index}"
        direct_rows.append(
            [
                facts.cite("active", f"{root}/nx"),
                facts.cite("active", f"{cert}/certified"),
                facts.cite("active", f"{cert}/grid"),
                facts.cite("active", f"{cert}/degree"),
                facts.cite("active", f"{cert}/rho"),
                facts.cite("active", f"{cert}/prefactor"),
                facts.cite("active", f"{cert}/reason"),
            ]
        )
    lines.extend(
        [
            _table(["N", "certified", "grid", "degree", "rho", "prefactor", "reason"], direct_rows),
            "",
            "Certificate settings: grids "
            f"{facts.cite('active', '/configuration/grid_ladder')}; degrees "
            f"{facts.cite('active', '/configuration/polynomial_degrees')}; epsilon samples "
            f"{facts.cite('active', '/configuration/epsilon_samples')}; target tolerance "
            f"{facts.cite('active', '/configuration/target_tolerance')}.",
            "",
            "Certificate changes:",
            "",
            *[
                f"- {facts.cite('active', f'/configuration/certificate_changes/{i}/commit')}: "
                f"{facts.cite('active', f'/configuration/certificate_changes/{i}/change')}"
                for i in range(len(active["configuration"]["certificate_changes"]))
            ],
            "",
            "Ordering check (rate differences below "
            f"{facts.cite('active', '/configuration/rate_tie_threshold')} are ties):",
            "",
            _table(
                [
                    "N",
                    "m",
                    "fastest regularized GMRES",
                    "union contains fastest",
                    "certified rates",
                    "rates tied",
                ],
                [
                    [
                        facts.cite("active", f"/ordering_check/{i}/nx"),
                        facts.cite("active", f"/ordering_check/{i}/m"),
                        facts.cite("active", f"/ordering_check/{i}/fastest_true_stop_count"),
                        facts.cite("active", f"/ordering_check/{i}/union_contains_fastest"),
                        facts.cite("active", f"/ordering_check/{i}/certified_rates"),
                        facts.cite(
                            "active",
                            f"/ordering_check/{i}/rate_differences_below_0p05_treated_as_ties",
                        ),
                    ]
                    for i in range(len(active["ordering_check"]))
                ],
            ),
            "",
            "## F7 — reference-count sensitivity",
            "",
            _table(
                [
                    "N",
                    "m",
                    "references",
                    "min Re(Y)",
                    "max Re(Y)",
                    "sigma_min(Y)",
                    "GMRES true",
                    "GMRES preconditioned",
                    "certificate",
                ],
                [
                    [
                        facts.cite("reference", f"/rows/{i}/nx"),
                        facts.cite("reference", f"/rows/{i}/m"),
                        facts.cite("reference", f"/rows/{i}/reference_count"),
                        facts.cite("reference", f"/rows/{i}/minimum_real"),
                        facts.cite("reference", f"/rows/{i}/maximum_real"),
                        facts.cite("reference", f"/rows/{i}/sigma_min"),
                        facts.cite("reference", f"/rows/{i}/gmres/true_stop_count"),
                        facts.cite("reference", f"/rows/{i}/gmres/preconditioned_stop_count"),
                        (
                            facts.cite("reference", f"/rows/{i}/Y_criterion/certified")
                            if row["Y_criterion"] is not None
                            else "not run"
                        ),
                    ]
                    for i, row in enumerate(reference["rows"])
                ],
            ),
            "",
            "## F8 — Lemma 3 a posteriori trajectories",
            "",
        ]
    )
    lemma3_rows = lemma["lemma3"]["rows"]
    min_lemma3_index = min(range(len(lemma3_rows)), key=lambda i: lemma3_rows[i]["minimum_margin"])
    failed_rows = [i for i, row in enumerate(lemma3_rows) if row["failed_steps"]]
    lines.extend(
        [
            "The checked inequality is `||v^n-u^n||_1 <= ||v^0-u^0||_1 + sum_k ||rho^k||_1`.",
            "",
            f"Trajectories: {facts.derived(len(lemma3_rows), 'lemma', '/lemma3/rows', 'list length')}; "
            f"failed-step trajectories: {facts.cite('lemma', '/lemma3/failed_step_trajectories')}; "
            f"violations: {facts.cite('lemma', '/lemma3/violations')}; minimum margin: "
            f"{facts.cite('lemma', f'/lemma3/rows/{min_lemma3_index}/minimum_margin')}.",
            "",
            "Trajectories containing failed steps:",
            "",
            _table(
                ["m", "dt", "kind", "side", "forcing", "failed steps", "minimum margin", "passes"],
                [
                    [
                        facts.cite("lemma", f"/lemma3/rows/{i}/m"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/dt"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/kind"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/side"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/forcing"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/failed_steps"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/minimum_margin"),
                        facts.cite("lemma", f"/lemma3/rows/{i}/passes"),
                    ]
                    for i in failed_rows
                ],
            ),
            "",
            "## F9 — Lemma 4 and stopping conventions",
            "",
            _table(
                ["N", "m", "kind", "dt", "eta", "sigma_min(M)", "kappa_2(M)"],
                [
                    [
                        facts.cite("lemma", f"/lemma4/{i}/nx"),
                        facts.cite("lemma", f"/lemma4/{i}/m"),
                        facts.cite("lemma", f"/lemma4/{i}/kind"),
                        facts.cite("lemma", f"/lemma4/{i}/dt"),
                        facts.cite("lemma", f"/lemma4/{i}/eta"),
                        facts.cite("lemma", f"/lemma4/{i}/sigma_min_M"),
                        facts.cite("lemma", f"/lemma4/{i}/kappa2_M"),
                    ]
                    for i in range(len(lemma["lemma4"]))
                ],
            ),
            "",
            "True-stop versus preconditioned-stop counts (the N=512, m=4 one-inverse row is marked):",
            "",
            _table(
                ["N", "m", "method", "true stop", "preconditioned stop", "highlight"],
                [
                    [
                        facts.cite("active", f"/cases/{i}/nx"),
                        facts.cite("active", f"/cases/{i}/m"),
                        facts.cite("active", f"/cases/{i}/method"),
                        facts.cite("active", f"/cases/{i}/gmres_regularized/true_stop_count"),
                        facts.cite(
                            "active", f"/cases/{i}/gmres_regularized/preconditioned_stop_count"
                        ),
                        (
                            "requested comparison"
                            if row["nx"] == 512
                            and row["m"] == 4
                            and row["method"] == "one_inverse_cfl_mask"
                            else ""
                        ),
                    ]
                    for i, row in enumerate(active["cases"])
                ],
            ),
            "",
            "## F10 — Lemma 2 norm bounds",
            "",
        ]
    )
    lemma2 = lemma["lemma2"]
    excess1 = [row["norm_1"] - row["eta_Z"] for row in lemma2]
    excess_inf = [row["norm_inf"] - row["eta_Z"] for row in lemma2]
    worst_indices = sorted(
        range(len(lemma2)),
        key=lambda i: max(excess1[i], excess_inf[i]),
        reverse=True,
    )[:10]
    lines.extend(
        [
            f"Records: {facts.derived(len(lemma2), 'lemma', '/lemma2', 'list length')}; violations: "
            f"{facts.derived(sum(not row['passes'] for row in lemma2), 'lemma', '/lemma2', 'count passes=false')}; "
            f"worst 1-norm excess: {facts.derived(max(excess1), 'lemma', '/lemma2', 'max(norm_1-eta_Z)')}; "
            f"worst infinity-norm excess: {facts.derived(max(excess_inf), 'lemma', '/lemma2', 'max(norm_inf-eta_Z)')}.",
            "",
            "Ten closest rows to the bound:",
            "",
            _table(
                [
                    "m",
                    "dt",
                    "eta",
                    "masked",
                    "eta_Z",
                    "||J-J0||1",
                    "||J-J0||inf",
                    "margin1",
                    "margin inf",
                ],
                [
                    [
                        facts.cite("lemma", f"/lemma2/{i}/m"),
                        facts.cite("lemma", f"/lemma2/{i}/dt"),
                        facts.cite("lemma", f"/lemma2/{i}/eta"),
                        facts.cite("lemma", f"/lemma2/{i}/masked"),
                        facts.cite("lemma", f"/lemma2/{i}/eta_Z"),
                        facts.cite("lemma", f"/lemma2/{i}/norm_1"),
                        facts.cite("lemma", f"/lemma2/{i}/norm_inf"),
                        facts.cite("lemma", f"/lemma2/{i}/margin_1"),
                        facts.cite("lemma", f"/lemma2/{i}/margin_inf"),
                    ]
                    for i in worst_indices
                ],
            ),
            "",
            "## F11 — CN replay",
            "",
            _table(
                [
                    "m",
                    "step",
                    "BE active",
                    "CN active",
                    "mask delta",
                    "BE Newton",
                    "BE GMRES",
                    "CN Newton",
                    "CN GMRES",
                    "BE min u",
                    "CN min u",
                ],
                [
                    [
                        facts.cite("cn", f"/steps/{i}/m"),
                        facts.cite("cn", f"/steps/{i}/step"),
                        facts.cite("cn", f"/steps/{i}/active_size_BE_operator_mask"),
                        facts.cite("cn", f"/steps/{i}/active_size_CN_operator_mask"),
                        facts.cite("cn", f"/steps/{i}/operator_mask_symmetric_difference"),
                        facts.cite("cn", f"/steps/{i}/BE/newton_iterations"),
                        facts.cite("cn", f"/steps/{i}/BE/gmres_iterations"),
                        facts.cite("cn", f"/steps/{i}/CN/newton_iterations"),
                        facts.cite("cn", f"/steps/{i}/CN/gmres_iterations"),
                        facts.cite("cn", f"/steps/{i}/BE/min_u"),
                        facts.cite("cn", f"/steps/{i}/CN/min_u"),
                    ]
                    for i in range(len(cn["steps"]))
                ],
            ),
            "",
            "Totals:",
            "",
            _table(
                [
                    "N",
                    "m",
                    "BE Newton",
                    "BE GMRES",
                    "CN Newton",
                    "CN GMRES",
                    "BE converged",
                    "CN converged",
                ],
                [
                    [
                        facts.cite("cn", f"/summaries/{i}/nx"),
                        facts.cite("cn", f"/summaries/{i}/m"),
                        facts.cite("cn", f"/summaries/{i}/BE/newton_total"),
                        facts.cite("cn", f"/summaries/{i}/BE/gmres_total"),
                        facts.cite("cn", f"/summaries/{i}/CN/newton_total"),
                        facts.cite("cn", f"/summaries/{i}/CN/gmres_total"),
                        facts.cite("cn", f"/summaries/{i}/BE/all_steps_converged"),
                        facts.cite("cn", f"/summaries/{i}/CN/all_steps_converged"),
                    ]
                    for i in range(len(cn["summaries"]))
                ],
            ),
            "",
            "## F12 — delivery versus this host",
            "",
        ]
    )
    for section, title in (
        ("mask_analysis_subset", "Mask-analysis differences"),
        ("drifted_E3_subset", "E3 differences"),
    ):
        rows = repro[section]
        mismatches = [i for i, row in enumerate(rows) if not row["all_equal"]]
        lines.extend(
            [
                f"### {title}",
                "",
                f"Compared rows: {facts.derived(len(rows), 'repro', f'/{section}', 'list length')}; "
                f"rows differing: {facts.derived(len(mismatches), 'repro', f'/{section}', 'count all_equal=false')}.",
                "",
            ]
        )
        mismatch_rows = []
        for index in mismatches:
            row = rows[index]
            identity = ", ".join(
                f"{key}={_display(value)}"
                for key, value in row.items()
                if key not in {"fields", "all_equal"}
            )
            for field, comparison in row["fields"].items():
                if comparison["equal"]:
                    continue
                root = f"/{section}/{index}/fields/{field}"
                mismatch_rows.append(
                    [
                        identity,
                        field,
                        facts.cite("repro", f"{root}/delivered"),
                        facts.cite("repro", f"{root}/regenerated"),
                    ]
                )
        lines.extend([_table(["case", "field", "delivery", "host"], mismatch_rows), ""])

    lines.extend(
        [
            "## F13 — Pavlov delivery claims map (listed, not recomputed)",
            "",
            "These rows are transcribed from the committed claims map. They are deliberately not "
            "recomputed here; the `source field` column retains that map's selector.",
            "",
            _table(
                ["row id", "value after regeneration", "source field"],
                [
                    [
                        row_id,
                        _display(value),
                        _display(field),
                    ]
                    for row_id, field, value in _claims_rows()
                ],
            ),
            "",
        ]
    )
    facts.validate()
    return "\n".join(lines), tuple(sorted(facts.references))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    text, _ = render_facts()
    args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
