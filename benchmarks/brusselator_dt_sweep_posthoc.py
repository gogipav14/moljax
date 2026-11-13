"""Post-hoc reaction-variation diagnostics for the regenerated CPU states."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import textwrap
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from benchmarks import brusselator_dt_sweep as sweep
from benchmarks import brusselator_dt_sweep_cpu_replication as replication
from moljax.experimental.brusselator_conditioning import build_brusselator_system

jax.config.update("jax_enable_x64", True)

SUMMARY = "benchmarks/results/BRUSSELATOR_DT_SWEEP_SUMMARY.md"


def _correlation(x: list[float], y: list[int | float]) -> dict[str, Any]:
    result = spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": len(x),
        "predictor_ties": len(set(x)) != len(x),
        "iteration_ties": len(set(y)) != len(y),
    }


def _cell_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "state": row["state"],
        "dt": row["dt"],
        "preconditioner": row["preconditioner"],
    }


def _sig(value: float, digits: int) -> str:
    return f"{value:.{digits}g}".replace("e-0", "e-").replace("e+0", "e")


def _wrap(text: str) -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=88,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _wrap_paragraphs(text: str) -> str:
    return "\n\n".join(_wrap(paragraph) for paragraph in text.strip().split("\n\n"))


def _true_residual_descriptive(
    true: dict[str, Any], cpu: dict[str, Any], gpu: dict[str, Any]
) -> dict[str, Any]:
    cells = list(true["cells"].values())
    methods = sorted({row["preconditioner"] for row in cells})
    condition_numbers = {}
    residuals = {}
    stop_differences = {}
    for method in methods:
        rows = [row for row in cells if row["preconditioner"] == method]
        finite = [row for row in rows if row["preconditioner_condition_number_2"] is not None]
        condition_numbers[method] = {
            "minimum": (
                min(row["preconditioner_condition_number_2"] for row in finite) if finite else None
            ),
            "maximum": (
                max(row["preconditioner_condition_number_2"] for row in finite) if finite else None
            ),
            "method": rows[0]["preconditioner_condition_number_method"],
        }
        converged = [row for row in rows if row["preconditioned_converged"]]
        maximum_residual = max(
            converged, key=lambda row: row["true_relative_residual_at_preconditioned_stop"]
        )
        residuals[method] = {
            "maximum": maximum_residual["true_relative_residual_at_preconditioned_stop"],
            "argmax_cell": _cell_identity(maximum_residual),
            "cells": len(converged),
        }
        deltas = [row["true_stop_count"] - row["preconditioned_stop_count"] for row in rows]
        maximum_delta = max(deltas)
        maximum_row = rows[deltas.index(maximum_delta)]
        stop_differences[method] = {
            "maximum_true_minus_preconditioned": maximum_delta,
            "argmax_cell": _cell_identity(maximum_row),
            "negative_cells": sum(delta < 0 for delta in deltas),
        }

    fft = [row for row in cells if row["preconditioner"] == "fft_diffusion"]
    condition_numbers["fft_diffusion"]["by_regime"] = {
        regime: {
            "minimum": min(
                row["preconditioner_condition_number_2"] for row in fft if row["regime"] == regime
            ),
            "maximum": max(
                row["preconditioner_condition_number_2"] for row in fft if row["regime"] == regime
            ),
        }
        for regime in ("hopf", "turing")
    }
    condition_numbers["fft_diffusion"]["turing_to_hopf_kappa_minus_one_by_dt"] = []
    for dt in sorted({row["dt"] for row in fft}):
        values = {
            regime: {
                row["preconditioner_condition_number_2"] - 1.0
                for row in fft
                if row["regime"] == regime and row["dt"] == dt
            }
            for regime in ("hopf", "turing")
        }
        if any(len(items) != 1 for items in values.values()):
            raise RuntimeError(f"FFT condition number varies within a regime at dt={dt}")
        hopf = values["hopf"].pop()
        turing = values["turing"].pop()
        condition_numbers["fft_diffusion"]["turing_to_hopf_kappa_minus_one_by_dt"].append(
            {"dt": dt, "hopf": hopf, "turing": turing, "ratio": turing / hopf}
        )
    mean_field_hopf_dt2 = [
        row["preconditioner_condition_number_2"]
        for row in cells
        if row["preconditioner"] == "mean_field_coupled_fft"
        and row["regime"] == "hopf"
        and row["state"] in {"developed64:hopf:1", "screen64:hopf:0", "screen64:hopf:1"}
        and row["dt"] == 2.0
    ]
    condition_numbers["mean_field_coupled_fft"]["hopf_dt2"] = {
        "minimum": min(mean_field_hopf_dt2),
        "maximum": max(mean_field_hopf_dt2),
        "cells": len(mean_field_hopf_dt2),
    }

    split = sorted(
        (
            row
            for row in cells
            if row["preconditioner"] == "split_reaction_after_diffusion"
            and row["family"] == "developed64"
            and row["dt"] == 1.0
        ),
        key=lambda row: row["state"],
    )
    fft_dt1 = {
        regime: {
            "preconditioned_stop": {
                "minimum": min(
                    row["preconditioned_stop_count"]
                    for row in fft
                    if row["regime"] == regime and row["dt"] == 1.0
                ),
                "maximum": max(
                    row["preconditioned_stop_count"]
                    for row in fft
                    if row["regime"] == regime and row["dt"] == 1.0
                ),
            },
            "true_stop": {
                "minimum": min(
                    row["true_stop_count"]
                    for row in fft
                    if row["regime"] == regime and row["dt"] == 1.0
                ),
                "maximum": max(
                    row["true_stop_count"]
                    for row in fft
                    if row["regime"] == regime and row["dt"] == 1.0
                ),
            },
        }
        for regime in ("hopf", "turing")
    }
    return {
        "note": "descriptive summary of the pre-registered true-residual cells; not a post hoc hypothesis",
        "cell_count": len(cells),
        "cpu_wall_seconds": true["cpu_wall_seconds"],
        "condition_number_2": condition_numbers,
        "physical_relative_residual_at_preconditioned_stop": residuals,
        "stop_count_difference": stop_differences,
        "split_dt1_developed64": {
            "states": [
                {
                    **_cell_identity(row),
                    "preconditioned_stop_count": row["preconditioned_stop_count"],
                    "true_stop_count": row["true_stop_count"],
                    "true_converged": row["true_converged"],
                }
                for row in split
            ],
            "true_stop_median": float(statistics.median(row["true_stop_count"] for row in split)),
            "fft_true_stop_median": true["decision_true_stop"]["d2_absolute_cost"][
                "median_iterations"
            ],
        },
        "fft_dt1_ranges": fft_dt1,
        "reproduction_environment": {
            "gpu": {key: gpu["provenance"][key] for key in ("python", "jax", "jaxlib", "numpy")},
            "cpu": {key: cpu["provenance"][key] for key in ("python", "jax", "jaxlib", "numpy")},
            "regenerated_states": cpu["plan"]["states"],
            "excluded_original_states": sorted(
                set(gpu["plan"]["states"]) - set(cpu["plan"]["states"])
            ),
        },
    }


def _assert_preexisting_unchanged(old: dict[str, Any], new: dict[str, Any]) -> None:
    def compare(previous: Any, current: Any, path: str) -> None:
        if path in {"/provenance/git_commit", "/provenance/script_sha256"}:
            return
        if isinstance(previous, dict):
            for key, value in previous.items():
                if path == "" and key in {
                    "spearman_between_predictors",
                    "true_residual_descriptive",
                }:
                    continue
                compare(value, current[key], f"{path}/{key}")
            return
        if previous != current:
            raise RuntimeError(f"pre-existing post-hoc field changed at {path}")

    compare(old, new, "")


def _replace_section(text: str, heading: str, next_heading: str, body: str) -> str:
    start = text.index(heading) + len(heading)
    end = text.index(next_heading, start)
    return text[:start] + "\n\n" + body.strip() + "\n\n" + text[end:]


def _render_summary(summary: str, payload: dict[str, Any]) -> str:
    descriptive = payload["true_residual_descriptive"]
    fft_ranges = descriptive["fft_dt1_ranges"]
    states = payload["states"]
    correlations = payload["spearman_at_dt_1"]
    near_hopf = [
        row
        for row in states
        if row["state"] in {"developed64:hopf:1", "screen64:hopf:0", "screen64:hopf:1"}
    ]
    screen_hopf = sorted(
        (row for row in near_hopf if row["state"].startswith("screen64:")),
        key=lambda row: row["state"],
    )
    developed_turing = next(row for row in states if row["state"] == "developed64:turing:200")
    ratio_rows = descriptive["condition_number_2"]["fft_diffusion"][
        "turing_to_hopf_kappa_minus_one_by_dt"
    ]
    if not all(abs(row["ratio"] - 5.0) <= 1.0e-12 for row in ratio_rows):
        raise RuntimeError("FFT Turing/Hopf kappa-minus-one ratio is not five")
    section3 = _wrap_paragraphs(f"""
Across the ten 64x64 states at `dt = 1`, the Spearman correlation between FFT
iterations and `max_x ||J_R(x)||_2` is
{correlations['max_reaction_jacobian_2_norm']['fft_diffusion']['preconditioned_stop']['rho']:.3f},
and {correlations['max_reaction_jacobian_2_norm']['fft_diffusion']['true_stop']['rho']:.3f}
when the same Krylov sequences are stopped on the physical residual (Section 5). The
two `screen64` Hopf states have `max ||J_R|| =
{min(row['max_reaction_jacobian_2_norm'] for row in screen_hopf):.2f}` and need
{screen_hopf[0]['iterations_at_dt_1']['fft_diffusion']['preconditioned_stop']} and
{screen_hopf[1]['iterations_at_dt_1']['fft_diffusion']['preconditioned_stop']}
iterations; `developed64:turing:200` has `max ||J_R|| =
{developed_turing['max_reaction_jacobian_2_norm']:.2f}` and needs
{developed_turing['iterations_at_dt_1']['fft_diffusion']['preconditioned_stop']}.
Regime separates the counts completely: at `dt = 1` every Hopf state needs
{fft_ranges['hopf']['preconditioned_stop']['minimum']} to
{fft_ranges['hopf']['preconditioned_stop']['maximum']} FFT iterations and every Turing
state {fft_ranges['turing']['preconditioned_stop']['minimum']} to
{fft_ranges['turing']['preconditioned_stop']['maximum']}
({fft_ranges['hopf']['true_stop']['minimum']} to
{fft_ranges['hopf']['true_stop']['maximum']} and
{fft_ranges['turing']['true_stop']['minimum']} to
{fft_ranges['turing']['true_stop']['maximum']} under the physical criterion). The two
regimes also differ in their reaction parameters and diffusion scaling
(`kappa_2(P_FFT) - 1` is five times larger on the Turing states at every `dt`), so the
separation does not identify a mechanism.

The ten states are fewer than ten independent observations. `developed64:hopf:1` and
the two `screen64` Hopf states are near-identical near-homogeneous states
(`max ||J_R||` {min(row['max_reaction_jacobian_2_norm'] for row in near_hopf):.3f} to
{max(row['max_reaction_jacobian_2_norm'] for row in near_hopf):.3f}, maximum reaction
variation {min(row['max_reaction_variation_2_norm'] for row in near_hopf):.3f} to
{max(row['max_reaction_variation_2_norm'] for row in near_hopf):.3f}), and so are the
two `screen64` Turing states. The correlations here and in Section 6 are descriptive.

Source: `BRUSSELATOR_DT_SWEEP_RESULTS.md`, "Cross-state reaction correlation";
`brusselator_dt_sweep_posthoc.json`,
`/spearman_at_dt_1/max_reaction_jacobian_2_norm/fft_diffusion/true_stop`,
`/true_residual_descriptive`, `/states/*`.
""")
    summary = _replace_section(
        summary, "## 3. Reaction magnitude predicts poorly; regime predicts well", "## 4.", section3
    )

    fft_kappa = descriptive["condition_number_2"]["fft_diffusion"]["by_regime"]
    mean_kappa = descriptive["condition_number_2"]["mean_field_coupled_fft"]
    if mean_kappa["hopf_dt2"]["cells"] != 3:
        raise RuntimeError("expected three near-homogeneous Hopf cells at dt=2")
    residuals = descriptive["physical_relative_residual_at_preconditioned_stop"]
    deltas = descriptive["stop_count_difference"]
    split = descriptive["split_dt1_developed64"]
    hopf20 = next(row for row in split["states"] if row["state"] == "developed64:hopf:20")
    section5_heading = "## 5. The decision is robust to the physical true-residual convention"
    section5_start = summary.index(section5_heading)
    section5_end = summary.index("## 6.", section5_start)
    section5 = summary[section5_start:section5_end]
    anchor = "which combines a Fourier-diagonal inverse with a spatially varying pointwise inverse."
    section5 = section5[: section5.index(anchor) + len(anchor)]
    appended = _wrap(
        f"""Over the {descriptive['cell_count']} cells, `kappa_2(P)` for the FFT preconditioner rises with
        `dt` from {_sig(fft_kappa['hopf']['minimum'], 3)} to
        {_sig(fft_kappa['hopf']['maximum'], 3)} on the Hopf states and from
        {_sig(fft_kappa['turing']['minimum'], 3)} to
        {_sig(fft_kappa['turing']['maximum'], 3)} on the Turing states; for the
        mean-field probe it ranges from {_sig(mean_kappa['minimum'], 3)} to
        {_sig(mean_kappa['maximum'], 3)}. These values bound the gap between the two
        criteria only loosely. Among converged cells, the physical relative residual
        at the preconditioned stop never exceeds
        {_sig(residuals['fft_diffusion']['maximum'], 2)} for FFT or
        {_sig(residuals['mean_field_coupled_fft']['maximum'], 2)} for the mean-field
        probe, including the three `dt = 2` Hopf cells
        where `kappa_2(P_mf)` is {_sig(mean_kappa['hopf_dt2']['minimum'], 2)} to
        {_sig(mean_kappa['hopf_dt2']['maximum'], 2)}. The split probe's own norm is the least faithful: at its
        preconditioned stop the physical residual reaches
        {_sig(residuals['split_reaction_after_diffusion']['maximum'], 2)}
        (`developed64:turing:200`, `dt = 2`), its physical stop comes up to
        {deltas['split_reaction_after_diffusion']['maximum_true_minus_preconditioned']}
        iterations later, and `developed64:hopf:20` at `dt = 1`, which converges in
        {hopf20['preconditioned_stop_count']} iterations under its own norm, does not
        converge within {hopf20['true_stop_count'] - 1} under the physical criterion.
        Its physical-stop median at `dt = 1` is {split['true_stop_median']:.1f} against
        FFT's {split['fft_true_stop_median']:g}, so it still
        fails D3."""
    )
    source = _wrap(f"""Source: `brusselator_dt_sweep_true_residual.json`, `/decision_true_stop`,
        `/decision_comparison`, and `/cells/*`; `BRUSSELATOR_DT_SWEEP_TRUE_RESIDUAL.md`
        gives all {descriptive['cell_count']} cells. The check took
        `{descriptive['cpu_wall_seconds']:.3f}` CPU seconds; this is diagnostic
        laptop timing only; `brusselator_dt_sweep_posthoc.json#/true_residual_descriptive`.""")
    summary = (
        summary[:section5_start]
        + section5.rstrip()
        + "\n"
        + appended
        + "\n\n"
        + source
        + "\n\n"
        + summary[section5_end:]
    )

    predictor_rho = payload["spearman_between_predictors"][
        "max_reaction_jacobian_vs_max_variation"
    ]["rho"]
    mean_pre = correlations["max_reaction_jacobian_2_norm"]["mean_field_coupled_fft"][
        "preconditioned_stop"
    ]["rho"]
    variation_pre = correlations["max_reaction_variation_2_norm"]["mean_field_coupled_fft"][
        "preconditioned_stop"
    ]["rho"]
    near_fft = [row["iterations_at_dt_1"]["fft_diffusion"] for row in near_hopf]
    near_mean = [row["iterations_at_dt_1"]["mean_field_coupled_fft"] for row in near_hopf]
    replacement = _wrap(
        f"""The sample is only `n = {len(states)}`, with the near-duplicate states noted in Section 3, and
the iteration counts contain ties, so these correlations are descriptive. The two
predictors are themselves rank-correlated at {predictor_rho:.3f} over these states, so
{variation_pre:.3f} against {mean_pre:.3f} does not distinguish them. The observation consistent with the
identity is narrower: on the three near-homogeneous Hopf states `max ||J_R||` is about
{min(row['max_reaction_jacobian_2_norm'] for row in near_hopf):.2f} and FFT needs
{min(row['preconditioned_stop'] for row in near_fft)} to
{max(row['preconditioned_stop'] for row in near_fft)} iterations at `dt = 1`
({min(row['true_stop'] for row in near_fft)} to {max(row['true_stop'] for row in near_fft)}
under the physical criterion), while the maximum reaction variation is at most
{max(row['max_reaction_variation_2_norm'] for row in near_hopf):.3f} and the mean-field
probe needs {min(row['preconditioned_stop'] for row in near_mean)} or
{max(row['preconditioned_stop'] for row in near_mean)}. Testing variation as a predictor
would need states chosen for that purpose."""
    )
    section6_start = summary.index("## 6.")
    paragraph_start = summary.index("The sample is only", section6_start)
    paragraph_end = summary.index("\n\nThe nearly singular", paragraph_start)
    summary = summary[:paragraph_start] + replacement + summary[paragraph_end:]
    summary = summary.replace(
        "| predictor | iterations | preconditioned stop | physical stop |",
        "| predictor | preconditioner | preconditioned stop | physical stop |",
    )
    summary = summary.replace(
        "`/spearman_at_dt_1/*`.",
        "`/spearman_at_dt_1/*`, `/spearman_between_predictors`.",
    )

    marker = "therefore classified as a **reproduction** on that state-similarity criterion."
    environment = descriptive["reproduction_environment"]
    if (
        environment["gpu"]["jax"] != environment["cpu"]["jax"]
        or environment["gpu"]["jaxlib"] != environment["cpu"]["jaxlib"]
    ):
        raise RuntimeError("GPU and CPU replication JAX versions differ")
    excluded = environment["excluded_original_states"]
    if len(excluded) != 2 or not all(state.startswith("fixed256:") for state in excluded):
        raise RuntimeError("the regenerated-state exclusion is not the two fixed256 states")
    insertion = (
        marker
        + "\n"
        + _wrap(f"""Both runs used JAX and jaxlib {environment['gpu']['jax']}; the GPU run used
        Python {environment['gpu']['python']} and NumPy {environment['gpu']['numpy']},
        the CPU run Python {environment['cpu']['python']} and NumPy
        {environment['cpu']['numpy']}. The two `fixed256` states were neither
        regenerated nor committed: their rows in Sections 2 and 4 rest on the original
        GPU run alone, and they do not enter D1 to D3.""")
    )
    start = summary.index(marker)
    all_cells = summary.index("All 200 GMRES", start)
    summary = summary[:start] + insertion + "\n\n" + summary[all_cells:]
    return summary


def run(repository: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    cpu_path = repository / "benchmarks/results/brusselator_dt_sweep_cpu_replication.json"
    true_path = repository / "benchmarks/results/brusselator_dt_sweep_true_residual.json"
    gpu_path = repository / "benchmarks/results/brusselator_dt_sweep.json"
    cpu = json.loads(cpu_path.read_text())
    true = json.loads(true_path.read_text())
    gpu = json.loads(gpu_path.read_text())
    states = []
    for spec in sweep.CPU_STATE_SPECS:
        record = sweep._find_record(repository, spec)
        state, source = sweep._load_regenerated_state(source_dir, spec, record)
        grid, regime = sweep._grid_and_regime(record)
        _, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        reaction = sweep._reaction_jacobian(state, grid, regime.b)
        mean = jnp.mean(reaction, axis=(0, 1))
        variation = jnp.linalg.svd(reaction - mean, compute_uv=False)[..., 0]
        maximum = float(jax.block_until_ready(jnp.max(variation)))
        average = float(jax.block_until_ready(jnp.mean(variation)))
        max_reaction = sweep._reaction_metadata(reaction, 1.0)["max_reaction_jacobian_2_norm"]
        symbol = np.asarray(jax.device_get(fft_cache.laplacian_symbol), dtype=np.float64)
        mean_np = np.asarray(jax.device_get(mean), dtype=np.float64)
        minima = {}
        for dt in sweep.PROBE_DT_GRID:
            minimum = float("inf")
            for value in symbol.ravel():
                block = np.eye(2) - dt * (
                    np.diag([diffusivities["u"] * value, diffusivities["v"] * value]) + mean_np
                )
                minimum = min(minimum, float(np.linalg.svd(block, compute_uv=False)[-1]))
            minima[f"dt={dt:g}"] = minimum
        counts = {}
        for method, arm in (("fft_diffusion", "main"), ("mean_field_coupled_fft", "probe")):
            key = replication._cell_key(arm, spec, 1.0, method)
            counts[method] = {
                "preconditioned_stop": cpu["cells"][key]["cpu"]["gmres"]["iterations"],
                "true_stop": true["cells"][key]["true_stop_count"],
            }
        states.append(
            {
                "state": spec.identifier,
                "regime": spec.regime,
                "source_state_sha256": source["state_identity"]["sha256"],
                "max_reaction_jacobian_2_norm": max_reaction,
                "max_reaction_variation_2_norm": maximum,
                "mean_reaction_variation_2_norm": average,
                "mean_field_fourier_block_minimum_singular_value": minima,
                "iterations_at_dt_1": counts,
            }
        )
    correlations = {}
    for predictor, field in (
        ("max_reaction_jacobian_2_norm", "max_reaction_jacobian_2_norm"),
        ("max_reaction_variation_2_norm", "max_reaction_variation_2_norm"),
    ):
        correlations[predictor] = {}
        x = [row[field] for row in states]
        for method in ("fft_diffusion", "mean_field_coupled_fft"):
            correlations[predictor][method] = {
                convention: _correlation(
                    x, [row["iterations_at_dt_1"][method][convention] for row in states]
                )
                for convention in ("preconditioned_stop", "true_stop")
            }
    predictor_correlation = _correlation(
        [row["max_reaction_jacobian_2_norm"] for row in states],
        [row["max_reaction_variation_2_norm"] for row in states],
    )
    predictor_correlation["first_predictor_ties"] = predictor_correlation.pop("predictor_ties")
    predictor_correlation["second_predictor_ties"] = predictor_correlation.pop("iteration_ties")
    payload = {
        "schema": "brusselator_dt_sweep_posthoc_v1",
        "analysis_label": "exploratory post hoc; hypothesis formed after seeing sweep results",
        "provenance": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.x64_enabled),
            "git_commit": sweep._git("rev-parse", "HEAD"),
            "script_sha256": sweep._sha256(Path(__file__)),
            "cpu_replication_sha256": sweep._sha256(cpu_path),
            "true_residual_check_sha256": sweep._sha256(true_path),
        },
        "states": states,
        "spearman_at_dt_1": correlations,
        "spearman_between_predictors": {
            "max_reaction_jacobian_vs_max_variation": predictor_correlation
        },
        "true_residual_descriptive": _true_residual_descriptive(true, cpu, gpu),
    }
    if output.is_file():
        _assert_preexisting_unchanged(json.loads(output.read_text()), payload)
    sweep._atomic_write_json(output, payload)
    summary_path = repository / SUMMARY
    summary_path.write_text(
        _render_summary(summary_path.read_text(encoding="utf-8"), payload),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", required=True)
    parser.add_argument("--output", default="benchmarks/results/brusselator_dt_sweep_posthoc.json")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    run(repository, Path(args.source_state_dir).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
