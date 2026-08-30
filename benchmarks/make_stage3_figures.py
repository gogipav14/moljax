#!/usr/bin/env python3
"""Regenerate ignored Stage-3 figures from committed Brusselator JSON results.

This script performs no model evaluation. Install the existing visualization
extra before running it:

    pip install -e ".[viz]"
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
FIGURES_DIR = REPOSITORY_ROOT / "benchmarks" / "figures"

VERDICT_COLORS = {
    "adequate": "tab:green",
    "investigate": "tab:orange",
    "indeterminate": "tab:red",
}


def _load_json(filename: str) -> dict[str, Any]:
    """Load one committed Brusselator result file."""
    return json.loads((RESULTS_DIR / filename).read_text())


def _pyplot() -> Any:
    """Import matplotlib lazily so visualization remains an optional extra."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as error:
        message = (
            "matplotlib is required; install the visualization extra with pip install -e '.[viz]'"
        )
        raise SystemExit(message) from error
    return plt


def _format_time(value: float) -> str:
    """Format a recorded time without changing its numerical value."""
    return f"t={value:g}"


def _plot_fixed_dt_transition(fixed_dt: dict[str, Any], plt: Any) -> Path:
    """Plot the recorded early-to-later disk-rate values by regime."""
    regimes = fixed_dt["fixed_dt_transition"]["by_regime"]
    figure, axes = plt.subplots(1, len(regimes), figsize=(10, 4.5), sharey=True)
    for axis, (regime, entries) in zip(axes, regimes.items(), strict=True):
        fft_entries = entries["fft_diffusion"]
        samples = (fft_entries["early"], fft_entries["developed"])
        positions = (0, 1)
        disk_rates = [float(sample["disk_rate"]) for sample in samples]
        colors = [VERDICT_COLORS[str(sample["verdict"])] for sample in samples]
        axis.plot(positions, disk_rates, color="0.45", linewidth=1.5, zorder=1)
        axis.scatter(positions, disk_rates, s=90, color=colors, edgecolors="black", zorder=2)
        axis.axhline(1.0, color="tab:red", linestyle="--", linewidth=1, label="disk-rate threshold")
        axis.set_xticks(positions, [_format_time(float(sample["time"])) for sample in samples])
        axis.set_title(regime.capitalize())
        axis.set_xlabel("visited state")
        axis.grid(axis="y", alpha=0.25)
        for position, sample in zip(positions, samples, strict=True):
            origin = "origin enclosed" if sample["origin_enclosed"] else "origin outside"
            disk_rate = float(sample["disk_rate"])
            high_rate = disk_rate >= 1.0
            axis.annotate(
                f"{sample['verdict']}\n{origin}",
                xy=(position, disk_rate),
                xytext=(-8, -34) if high_rate else (0, 9),
                textcoords="offset points",
                ha="right" if high_rate else "center",
                fontsize=8,
            )
    axes[0].set_ylabel("FFT enclosing-disk rate")
    axes[-1].legend(loc="upper left", fontsize=8)
    figure.suptitle("Fixed-step FFT decision transition from committed states")
    figure.tight_layout()
    output = FIGURES_DIR / "stage3_fixed_dt_transition.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_adequacy_vs_cost(fixed_dt: dict[str, Any], plt: Any) -> Path:
    """Compare recorded FFT and identity counted-GMRES costs for each fixed-step state."""
    regimes = fixed_dt["fixed_dt_transition"]["by_regime"]
    labels: list[str] = []
    fft_counts: list[float] = []
    identity_counts: list[float] = []
    identity_converged: list[bool] = []
    for regime, entries in regimes.items():
        for state_name in ("early", "developed"):
            fft_entry = entries["fft_diffusion"][state_name]
            identity_entry = entries["identity"][state_name]
            labels.append(f"{regime}\n{_format_time(float(fft_entry['time']))}")
            fft_counts.append(float(fft_entry["actual_gmres_iterations"]))
            identity_counts.append(float(identity_entry["actual_gmres_iterations"]))
            identity_converged.append(bool(identity_entry["actual_gmres_converged"]))

    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.5, 4.5))
    axis.bar(
        [position - width / 2 for position in positions],
        fft_counts,
        width,
        color="tab:green",
        label="FFT diffusion",
    )
    identity_bars = axis.bar(
        [position + width / 2 for position in positions],
        identity_counts,
        width,
        color="tab:gray",
        label="identity",
    )
    for bar, converged in zip(identity_bars, identity_converged, strict=True):
        if not converged:
            bar.set_hatch("//")
            axis.annotate(
                "tolerance\nnot met",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    axis.set_xticks(positions, labels)
    axis.set_ylabel("counted GMRES iterations")
    axis.set_title("FFT preconditioning keeps recorded linear solves inexpensive")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    output = FIGURES_DIR / "stage3_adequacy_vs_cost.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _plot_hopf_imaginary_extent(developed: dict[str, Any], plt: Any) -> Path:
    """Plot the recorded Hopf numerical-range imaginary extent at 64 by 64."""
    samples = developed["regime_comparison"]["hopf"]["fov_imaginary_extent_by_time"]
    times = [float(sample["time"]) for sample in samples]
    extents = [float(sample["fov_imaginary_extent"]) for sample in samples]
    figure, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.plot(times, extents, marker="o", color="tab:purple")
    axis.set_xlabel("time")
    axis.set_ylabel("FOV imaginary extent")
    axis.set_title("64 by 64 Hopf imaginary extent along recorded trajectory")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    output = FIGURES_DIR / "stage3_hopf_imaginary_extent.png"
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    """Generate every ignored Stage-3 PNG from the committed result files."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt = _pyplot()
    fixed_dt = _load_json("brusselator_conditioning_fixed_dt.json")
    developed = _load_json("brusselator_conditioning_developed.json")
    outputs = (
        _plot_fixed_dt_transition(fixed_dt, plt),
        _plot_adequacy_vs_cost(fixed_dt, plt),
        _plot_hopf_imaginary_extent(developed, plt),
    )
    for output in outputs:
        print(output.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
