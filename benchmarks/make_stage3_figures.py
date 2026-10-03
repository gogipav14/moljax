#!/usr/bin/env python3
"""Regenerate ignored Stage-3 figures from resolved Brusselator JSON results."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
FIGURES_DIR = REPOSITORY_ROOT / "benchmarks" / "figures"
PRESETS = (
    ("brusselator_conditioning.json", "screen 64"),
    ("brusselator_conditioning_developed.json", "developed 64"),
    ("brusselator_conditioning_fixed_dt.json", "fixed dt 256"),
    ("brusselator_conditioning_hopf_continuation.json", "Hopf continuation 256"),
)
CATEGORIES = ("adequate", "provisional", "investigate", "indeterminate", "uncertified_at_cap")
COLORS = {
    "adequate": "tab:green",
    "provisional": "tab:blue",
    "investigate": "tab:orange",
    "indeterminate": "tab:red",
    "uncertified_at_cap": "tab:gray",
}


def _load(filename: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / filename).read_text())


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        raise SystemExit("matplotlib is required; install the visualization extra") from error
    return plt


def _plot_tally(reports: dict[str, dict[str, Any]], plt: Any) -> Path:
    figure, axis = plt.subplots(figsize=(9.0, 4.8))
    positions = list(range(len(PRESETS)))
    bottoms = [0] * len(PRESETS)
    for category in CATEGORIES:
        values = [
            Counter(report["final_category"] for report in reports[name]["records"])[category]
            for name, _ in PRESETS
        ]
        axis.bar(
            positions,
            values,
            bottom=bottoms,
            color=COLORS[category],
            label=category.replace("_", " "),
        )
        bottoms = [
            bottom + value for bottom, value in zip(bottoms, values, strict=True)
        ]
    axis.set_xticks(positions, [label for _, label in PRESETS])
    axis.set_ylabel("resolved records")
    axis.set_title("Stage-3 final categories after FOV support escalation")
    axis.legend(ncol=2, fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = FIGURES_DIR / "stage3_resolved_tally.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_fixed_dt(fixed_dt: dict[str, Any], plt: Any) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    records = fixed_dt["records"]
    for preconditioner, marker in (("identity", "o"), ("fft_diffusion", "s")):
        subset = [record for record in records if record["preconditioner"] == preconditioner]
        positions = list(range(len(subset)))
        labels = [
            f"{record['regime']}\nstep {record.get('trajectory_step', record.get('state_index'))}"
            for record in subset
        ]
        colors = [COLORS[record["final_category"]] for record in subset]
        axes[0].scatter(
            positions,
            [record["disk_rate"] for record in subset],
            marker=marker,
            s=85,
            c=colors,
            edgecolors="black",
            label=preconditioner,
        )
        axes[1].scatter(
            positions,
            [record["actual_gmres"]["iterations"] for record in subset],
            marker=marker,
            s=85,
            c=colors,
            edgecolors="black",
            label=preconditioner,
        )
        for axis in axes:
            axis.set_xticks(positions, labels)
            axis.grid(axis="y", alpha=0.25)
    axes[0].axhline(
        1.0,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="origin-enclosure threshold",
    )
    axes[0].set_ylabel("resolved FOV disk rate")
    axes[0].set_title("Fixed-dt geometry")
    axes[1].set_ylabel("counted GMRES iterations")
    axes[1].set_title("Measured solve work")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    output = FIGURES_DIR / "stage3_fixed_dt_geometry_and_gmres.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_bound_vs_disk(reports: dict[str, dict[str, Any]], plt: Any) -> Path:
    figure, axis = plt.subplots(figsize=(7.5, 5.0))
    for name, _ in PRESETS:
        for record in reports[name]["records"]:
            certificate = record["fourier_weyl_ghost_lower_bound"]
            axis.scatter(
                certificate["full_lower_bound"],
                record["disk_rate"],
                color=COLORS[record["final_category"]],
                marker="x" if record["origin_enclosed"] else "o",
                s=58,
                alpha=0.85,
            )
    axis.axvline(
        0.1,
        color="tab:blue",
        linestyle="--",
        linewidth=1,
        label="adequacy-bound gate",
    )
    axis.axhline(
        1.0,
        color="tab:red",
        linestyle="--",
        linewidth=1,
        label="origin-enclosure threshold",
    )
    axis.set_xlabel("Fourier–Weyl–ghost lower bound")
    axis.set_ylabel("resolved FOV disk rate")
    axis.set_title("Bound evidence and FOV geometry")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    output = FIGURES_DIR / "stage3_bound_vs_disk.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    """Generate ignored figures from committed resolved JSON inputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    reports = {name: _load(name) for name, _ in PRESETS}
    plt = _pyplot()
    outputs = (
        _plot_tally(reports, plt),
        _plot_fixed_dt(reports["brusselator_conditioning_fixed_dt.json"], plt),
        _plot_bound_vs_disk(reports, plt),
    )
    for output in outputs:
        print(output.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
