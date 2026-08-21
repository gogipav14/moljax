#!/usr/bin/env python3
"""Generate Stage-2 figures directly from the committed benchmark JSON files.

This script contains no model evaluation and creates no synthetic measurements.
Install the existing visualization extra before running it:

    pip install -e ".[viz]"
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
FIGURES_DIR = REPOSITORY_ROOT / "benchmarks" / "figures"


def _load_json(filename: str) -> dict[str, Any]:
    """Load one committed Stage-2 result file."""
    return json.loads((RESULTS_DIR / filename).read_text())


def _pyplot() -> Any:
    """Import matplotlib lazily so the visualization extra stays optional."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        message = "matplotlib is required; install the existing visualization extra with pip install -e '.[viz]'"
        raise SystemExit(message) from error
    return plt


def _rate_matrix(
    cells: list[dict[str, Any]], m_values: list[int], dt_values: list[float], rate_key: str
) -> list[list[float]]:
    """Return the requested JSON disk-rate medians arranged by exponent and step size."""
    by_cell = {(int(cell["m"]), float(cell["analysis_dt"])): cell for cell in cells}
    matrix: list[list[float]] = []
    for m in m_values:
        row: list[float] = []
        for dt in dt_values:
            value = by_cell[(m, dt)][rate_key]["median"]
            row.append(math.nan if value is None else float(value))
        matrix.append(row)
    return matrix


def _plot_pme_regime_map(pme: dict[str, Any], plt: Any) -> Path:
    """Plot JSON-recorded easy and hard identity disk rates for each PME cell."""
    cells = pme["regime_map"]["cells"]
    m_values = sorted({int(cell["m"]) for cell in cells})
    dt_values = sorted({float(cell["analysis_dt"]) for cell in cells})
    matrices = (
        (
            "Easy-state identity disk rate",
            _rate_matrix(cells, m_values, dt_values, "identity_easy_disk_rate"),
        ),
        (
            "Hard-state identity disk rate",
            _rate_matrix(cells, m_values, dt_values, "identity_hard_disk_rate"),
        ),
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e6e6e6")
    finite_rates = [
        value for _, matrix in matrices for row in matrix for value in row if math.isfinite(value)
    ]
    vmax = max(1.0, max(finite_rates))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, (title, matrix) in zip(axes, matrices, strict=True):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
        axis.set_title(title)
        axis.set_xticks(range(len(dt_values)), [f"{dt:g}" for dt in dt_values])
        axis.set_yticks(range(len(m_values)), [str(m) for m in m_values])
        axis.set_xlabel("candidate implicit step size")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                label = "n/a" if not math.isfinite(value) else f"{value:.3f}"
                color = "white" if math.isfinite(value) and value > 0.55 * vmax else "black"
                axis.text(
                    column_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=8,
                )
        figure.colorbar(image, ax=axis, shrink=0.85, label="enclosing-disk rate")
    axes[0].set_ylabel("porous-medium exponent m")
    figure.suptitle("PME identity geometry separates easy and hard visited states")
    figure.tight_layout()
    output = FIGURES_DIR / "stage2_pme_regime_map.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _hard_verdict_counts(reaction: dict[str, Any], entry: dict[str, Any]) -> Counter[str]:
    """Count JSON verdicts for the identity-hard states at one reaction strength."""
    reaction_strength = float(entry["r"])
    threshold = float(entry["identity_hard_iteration_threshold"])
    hard_records = [
        record
        for record in reaction["records"]
        if float(record["r"]) == reaction_strength
        and record["d0_kind"] == "identity"
        and float(record["actual_gmres"]["iterations"]) >= threshold
    ]
    return Counter(record["verdict"] for record in hard_records)


def _plot_reaction_axis(reaction: dict[str, Any], plt: Any) -> Path:
    """Plot identity iteration ranges and hard-state verdict composition by reaction strength."""
    entries = reaction["reaction_effect"]["by_reaction"]
    positions = list(range(len(entries)))
    labels = [f"r={float(entry['r']):g}" for entry in entries]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    iteration_axis, verdict_axis = axes
    for position, entry in zip(positions, entries, strict=True):
        iteration_range = entry["identity_iteration_range"]
        iteration_axis.vlines(
            position,
            float(iteration_range["min"]),
            float(iteration_range["max"]),
            color="tab:blue",
            linewidth=3,
        )
        iteration_axis.scatter(
            position,
            float(iteration_range["median"]),
            color="black",
            label="median" if position == 0 else None,
            zorder=3,
        )
    iteration_axis.set_xticks(positions, labels)
    iteration_axis.set_ylabel("identity GMRES iterations")
    iteration_axis.set_title("Identity iteration range across visited states")
    iteration_axis.legend()

    verdicts = ("adequate", "investigate", "indeterminate")
    colors = {"adequate": "tab:green", "investigate": "tab:orange", "indeterminate": "tab:red"}
    bottoms = [0.0] * len(entries)
    for verdict in verdicts:
        fractions: list[float] = []
        for entry in entries:
            counts = _hard_verdict_counts(reaction, entry)
            total = sum(counts.values())
            fractions.append(counts[verdict] / total if total else 0.0)
        verdict_axis.bar(positions, fractions, bottom=bottoms, color=colors[verdict], label=verdict)
        for position, fraction, bottom in zip(positions, fractions, bottoms, strict=True):
            if fraction:
                verdict_axis.text(
                    position,
                    bottom + 0.5 * fraction,
                    f"{fraction:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
        bottoms = [bottom + fraction for bottom, fraction in zip(bottoms, fractions, strict=True)]
    verdict_axis.set_xticks(positions, labels)
    verdict_axis.set_ylim(0.0, 1.0)
    verdict_axis.set_ylabel("fraction of identity-hard states")
    verdict_axis.set_title("Hard-state decision verdict composition")
    verdict_axis.legend()
    figure.suptitle("Reaction-axis conditioning evidence from the recorded identity systems")
    figure.tight_layout()
    output = FIGURES_DIR / "stage2_reaction_axis.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _matched_crossover(problem: dict[str, Any]) -> dict[str, Any]:
    """Return the recorded, rather than interpolated, matched-accuracy crossover."""
    return next(
        item for item in problem["matched_accuracy_crossovers"] if item["status"] == "matched"
    )


def _plot_work_precision(work_precision: dict[str, Any], plt: Any) -> Path:
    """Plot each recorded error/runtime point for both methods and both nonlinear problems."""
    problems = work_precision["problems"]
    labels = {"pme_m2": "PME (m=2)", "porous_fisher_r1": "Porous-Fisher (r=1)"}
    styles = {
        "be_jfnk_frozen_bulk": {"color": "tab:blue", "marker": "o", "label": "BE-JFNK"},
        "diffrax_tsit5_pid": {"color": "tab:orange", "marker": "s", "label": "Diffrax Tsit5/PID"},
    }

    figure, axes = plt.subplots(1, len(problems), figsize=(11, 4.5), sharey=True)
    for axis, (name, problem) in zip(axes, problems.items(), strict=True):
        for method, style in styles.items():
            records = problem[method]
            runtimes = [float(record["runtime"]["median_seconds"]) for record in records]
            errors = [float(record["error_inf"]) for record in records]
            axis.loglog(runtimes, errors, linewidth=1.5, **style)
        crossover = _matched_crossover(problem)
        diffrax_record = crossover["diffrax_tsit5_pid"]
        axis.scatter(
            [float(diffrax_record["runtime"]["median_seconds"])],
            [float(diffrax_record["error_inf"])],
            s=110,
            facecolors="none",
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )
        axis.annotate(
            f"{float(crossover['speedup']):.1f}× faster\nat ≤ {float(crossover['target_error']):.0e}",
            xy=(
                float(diffrax_record["runtime"]["median_seconds"]),
                float(diffrax_record["error_inf"]),
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=8,
        )
        axis.text(0.03, 0.04, "lower-left = better", transform=axis.transAxes, fontsize=8)
        axis.set_title(labels[name])
        axis.set_xlabel("median runtime (s)")
        axis.grid(which="both", alpha=0.25)
    axes[0].set_ylabel(r"$\|e\|_\infty$ on common smooth interior")
    axes[0].legend()
    figure.suptitle("Nonlinear work--precision from recorded benchmark data")
    figure.tight_layout()
    output = FIGURES_DIR / "stage2_work_precision.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    """Generate every ignored Stage-2 PNG from committed JSON data."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    pme = _load_json("pme_breakdown.json")
    reaction = _load_json("porous_fisher_conditioning.json")
    work_precision = _load_json("work_precision_nonlinear_diffusion.json")
    outputs = (
        _plot_pme_regime_map(pme, plt),
        _plot_reaction_axis(reaction, plt),
        _plot_work_precision(work_precision, plt),
    )
    for output in outputs:
        print(output.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
