#!/usr/bin/env python3
"""Regenerate ignored tau-blend figures from committed JSON measurements."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
FIGURES_DIR = REPOSITORY_ROOT / "benchmarks" / "figures"
BASELINE_RESULT = "pme_dst_tau_blend_single_reference_baselines.json"
INEXACT_RESULT = "pme_dst_tau_blend_inexact_newton.json"
PSEUDOSPECTRAL_RESULT = "pme_dst_tau_blend_pseudospectral_criterion.json"

METHOD_LABELS = {
    "identity": "identity",
    "frozen_mean": "arithmetic mean",
    "frozen_bulk": "bulk mean",
    "floor": "floor",
    "const": "constant",
    "geometric_mean": "geometric mean",
    "harmonic_mean": "harmonic mean",
    "optimized_d0": "oracle $d_0^*$",
    "tau_blend": "tau blend ($l=3$)",
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


def _save(figure: Any, plt: Any, filename: str) -> Path:
    output = FIGURES_DIR / filename
    figure.tight_layout()
    figure.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(figure)
    return output


def _spectral_equivalence_boundary(baseline: dict[str, Any], plt: Any) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    records = baseline["spectral_records"]
    methods = tuple(METHOD_LABELS)
    for method in methods:
        subset = sorted(
            (record for record in records if record["method"] == method),
            key=lambda record: record["coefficient"]["degeneracy_fraction"],
        )
        if not subset:
            continue
        axis.plot(
            [record["coefficient"]["degeneracy_fraction"] for record in subset],
            [record["spectrum"]["condition_number_2"] for record in subset],
            marker="o",
            linewidth=2.4 if method == "tau_blend" else 1.0,
            label=METHOD_LABELS[method],
        )
    axis.axvspan(0.0, 0.05, color="tab:green", alpha=0.12, label="zero-set-free limit")
    axis.set_yscale("log")
    axis.set_xlabel(r"degeneracy fraction $\#\{D<\mathrm{floor}\}/N$")
    axis.set_ylabel(r"$\kappa_2(P^{-1}J)$")
    axis.set_title("F1. Spectral-equivalence boundary")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncol=2, fontsize=7)
    return _save(figure, plt, "tau_blend_f1_spectral_equivalence_boundary.png")


def _sorted_spectrum(baseline: dict[str, Any], plt: Any) -> Path:
    figure, axis = plt.subplots(figsize=(8.5, 5.0))
    records = [record for record in baseline["spectral_records"] if record["m"] == 8]
    for record in records:
        method = record["method"]
        axis.plot(
            record["spectrum"]["sorted_eigenvalue_magnitudes"],
            linewidth=2.6 if method == "tau_blend" else 1.0,
            alpha=1.0 if method == "tau_blend" else 0.75,
            label=METHOD_LABELS[method],
        )
    axis.axhline(0.1, color="black", linestyle="--", linewidth=1, label=r"$|\lambda|=0.1$")
    axis.set_yscale("log")
    axis.set_xlabel("sorted eigenvalue index")
    axis.set_ylabel(r"$|\lambda|$")
    axis.set_title("F2. Near-zero spectral tail at $m=8$, $N=512$")
    axis.grid(alpha=0.25, which="both")
    axis.legend(ncol=2, fontsize=7)
    return _save(figure, plt, "tau_blend_f2_sorted_eigenvalue_magnitudes.png")


def _spectrum_and_fov(baseline: dict[str, Any], plt: Any) -> Path:
    record = next(
        record
        for record in baseline["spectral_records"]
        if record["m"] == 8 and record["method"] == "tau_blend"
    )
    eigenvalues = np.asarray(record["spectrum"]["eigenvalues"], dtype=float)
    boundary = np.asarray(record["field_of_values"]["boundary"], dtype=float)
    gap = record["spectrum"]["non_normality_gap"]
    figure, axis = plt.subplots(figsize=(7.0, 5.4))
    axis.scatter(eigenvalues[:, 0], eigenvalues[:, 1], s=12, alpha=0.55, label="spectrum")
    axis.plot(
        boundary[:, 0],
        boundary[:, 1],
        color="tab:orange",
        linewidth=1.5,
        label="FOV support boundary",
    )
    axis.scatter([0.0], [0.0], color="black", marker="x", s=60, label="origin")
    axis.annotate(
        f"numerical − spectral abscissa = {gap:.3g}",
        xy=(0.03, 0.95),
        xycoords="axes fraction",
        va="top",
    )
    axis.set_xlabel("real part")
    axis.set_ylabel("imaginary part")
    axis.set_title("F3. Spectrum and field of values ($m=8$, blend)")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    return _save(figure, plt, "tau_blend_f3_spectrum_vs_fov.png")


def _work_precision(baseline: dict[str, Any], plt: Any) -> Path:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for axis, tolerance in zip(axes, (1.0e-2, 1.0e-8), strict=True):
        for nx, linestyle in ((512, "-"), (1024, "--")):
            subset = sorted(
                (
                    record
                    for record in baseline["work_precision_records"]
                    if record["nx"] == nx and record["requested_relative_residual"] == tolerance
                ),
                key=lambda record: record["m"],
            )
            for method in ("frozen_bulk", "optimized_d0", "identity", "tau_blend"):
                axis.plot(
                    [record["m"] for record in subset],
                    [record["methods"][method]["median_seconds"] for record in subset],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{METHOD_LABELS[method]}, N={nx}",
                )
        axis.set_yscale("log")
        axis.set_xticks((2, 4, 8))
        axis.set_xlabel("PME exponent m")
        axis.set_title(f"target relative residual {tolerance:.0e}")
        axis.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("median solve time (s)")
    axes[1].legend(fontsize=6, ncol=2)
    figure.suptitle("F4. Linear-solve work–precision: loose and tight regimes")
    return _save(figure, plt, "tau_blend_f4_work_precision.png")


def _inexact_newton_trace(inexact: dict[str, Any], plt: Any) -> Path:
    record = next(record for record in inexact["records"] if record["m"] == 8)
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    for method, payload in record["methods"].items():
        steps = payload["representative"]["steps"]
        x = [step["iteration"] for step in steps]
        axes[0].plot(
            x,
            [step["residual_norm_after"] for step in steps],
            marker="o",
            label=METHOD_LABELS[method],
        )
        axes[1].plot(
            x, [step["gmres_iterations"] for step in steps], marker="o", label=METHOD_LABELS[method]
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("nonlinear residual after step")
    axes[1].set_ylabel("counted GMRES iterations")
    for axis in axes:
        axis.set_xlabel("Newton iteration")
        axis.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=8)
    figure.suptitle("F5. Inexact-Newton per-step trace ($m=8$)")
    return _save(figure, plt, "tau_blend_f5_inexact_newton_trace.png")


def _pseudospectral_window(pseudospectral: dict[str, Any], plt: Any) -> Path:
    record = next(record for record in pseudospectral["records"] if record["m"] == 8)
    criterion = record["pseudospectral_criterion"]
    real = np.asarray(criterion["real_grid"], dtype=float)
    imag = np.asarray(criterion["imag_grid"], dtype=float)
    sigma = np.asarray(criterion["sigma_min_grid"], dtype=float)
    levels = sorted(
        {
            criterion["eps_connect_upper"],
            criterion["selected_epsilon"],
            criterion["epsilon_zero"],
        }
    )
    figure, axis = plt.subplots(figsize=(7.0, 5.4))
    contours = axis.contour(real, imag, sigma, levels=levels, linewidths=(1.2, 2.0, 1.2))
    axis.clabel(contours, inline=True, fontsize=8, fmt="ε=%.3g")
    axis.scatter([0.0], [0.0], color="black", marker="x", s=60, label="origin")
    axis.set_xlabel("real part")
    axis.set_ylabel("imaginary part")
    axis.set_title("F6. Pseudospectral connectivity and certification window ($m=8$)")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    return _save(figure, plt, "tau_blend_f6_pseudospectral_window.png")


def main() -> None:
    """Generate six ignored figures solely from committed JSON inputs."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _load(BASELINE_RESULT)
    inexact = _load(INEXACT_RESULT)
    pseudospectral = _load(PSEUDOSPECTRAL_RESULT)
    plt = _pyplot()
    outputs = (
        _spectral_equivalence_boundary(baseline, plt),
        _sorted_spectrum(baseline, plt),
        _spectrum_and_fov(baseline, plt),
        _work_precision(baseline, plt),
        _inexact_newton_trace(inexact, plt),
        _pseudospectral_window(pseudospectral, plt),
    )
    for output in outputs:
        print(output.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
