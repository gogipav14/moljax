#!/usr/bin/env python3
"""Regenerate the tau-blend measurement README from committed JSON files."""

from __future__ import annotations

import json
import math
from pathlib import Path
from textwrap import fill
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPOSITORY_ROOT / "benchmarks" / "results"
OUTPUT = RESULTS_DIR / "TAU_BLEND_README.md"
METHODS = (
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
EFFECTIVE_SCALAR_METHODS = (
    "frozen_mean",
    "frozen_bulk",
    "const",
    "geometric_mean",
    "harmonic_mean",
    "optimized_d0",
)
# The result schema names ``near_zero_eigenvalue_count`` but does not carry the
# threshold itself. Keep this renderer constant synchronized with the benchmark.
NEAR_ZERO_THRESHOLD = 0.1
SHORT = {
    "identity": "identity",
    "frozen_mean": "mean",
    "frozen_bulk": "bulk",
    "floor": "floor",
    "const": "const",
    "geometric_mean": "geometric",
    "harmonic_mean": "harmonic",
    "optimized_d0": "oracle d0*",
    "tau_blend": "blend",
}


def _load(filename: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / filename).read_text())


def _iterations(values: dict[str, Any]) -> int:
    iterations = values["iterations"]
    return int(iterations[0] if isinstance(iterations, list) else iterations)


def _scientific(value: float) -> str:
    """Format one significant digit without a zero-padded exponent."""
    return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def _wrap(text: str) -> str:
    """Wrap one generated prose paragraph deterministically."""
    return fill(
        " ".join(text.split()),
        width=88,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _measurement_cell(values: dict[str, Any]) -> str:
    marker = "" if values["all_converged"] else " cap"
    return (
        f"{_iterations(values)}{marker}; "
        f"{values['median_seconds']:.3f} +/- {values['iqr_seconds']:.3f} s"
    )


def _work_table(baseline: dict[str, Any]) -> str:
    header = "| N | m | tol | " + " | ".join(SHORT[method] for method in METHODS) + " |"
    rule = "|---:|---:|---:|" + "---:|" * len(METHODS)
    rows = [header, rule]
    for record in baseline["work_precision_records"]:
        cells = [_measurement_cell(record["methods"][method]) for method in METHODS]
        rows.append(
            f"| {record['nx']} | {record['m']} | "
            f"{_scientific(record['requested_relative_residual'])} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


def _spectral_table(baseline: dict[str, Any]) -> str:
    rows = [
        f"| m | method | kappa_2 | count abs(lambda)<{NEAR_ZERO_THRESHOLD:g} | "
        "min abs(lambda) | "
        "spectral abscissa | numerical abscissa | gap | disk rate |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in baseline["spectral_records"]:
        spectrum = record["spectrum"]
        rows.append(
            f"| {record['m']} | {SHORT[record['method']]} | "
            f"{spectrum['condition_number_2']:.3g} | "
            f"{spectrum['near_zero_eigenvalue_count']} | "
            f"{spectrum['minimum_eigenvalue_modulus']:.3g} | "
            f"{spectrum['spectral_abscissa']:.3g} | "
            f"{spectrum['numerical_abscissa']:.3g} | "
            f"{spectrum['non_normality_gap']:.3g} | "
            f"{record['field_of_values']['disk_rate']:.4f} |"
        )
    return "\n".join(rows)


def _contrast_table(baseline: dict[str, Any]) -> str:
    rows = [
        "| m | step | D95/D05 | degeneracy fraction | d0*/Dmax(active) | "
        "d0* empirical rank | oracle iterations | blend iterations | oracle/blend |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    summaries = baseline["summaries"]["contrast"]
    for record in baseline["contrast_records"]:
        summary = next(
            row for row in summaries if row["m"] == record["m"] and row["step"] == record["step"]
        )
        scan = record["optimized_d0_scan"]
        ratio = summary["best_single_iterations"] / summary["tau_iterations"]
        rows.append(
            f"| {record['m']} | {record['step']} | "
            f"{record['coefficient']['active_d95_over_d05']:.3f} | "
            f"{record['coefficient']['degeneracy_fraction']:.3f} | "
            f"{scan['ratio_to_active_maximum']:.3f} | "
            f"{scan['active_empirical_quantile_rank']:.3f} | "
            f"{summary['best_single_iterations']} | {summary['tau_iterations']} | "
            f"{ratio:.2f}x |"
        )
    return "\n".join(rows)


def _tight_table(baseline: dict[str, Any]) -> str:
    rows = [
        "| N | m | oracle d0* | blend | active-oracle/blend time | "
        "fastest converged comparator | fastest-comparator/blend time |",
        "|---:|---:|---:|---:|---:|---|---:|",
    ]
    tight_tolerance = min(baseline["config"]["work_tolerances"])
    for record in baseline["work_precision_records"]:
        if record["requested_relative_residual"] != tight_tolerance:
            continue
        methods = record["methods"]
        oracle = methods["optimized_d0"]
        blend = methods["tau_blend"]
        converged = {
            name: values
            for name, values in methods.items()
            if name != "tau_blend" and values["all_converged"]
        }
        fastest_name = (
            min(converged, key=lambda name: converged[name]["median_seconds"])
            if converged
            else None
        )
        oracle_ratio = (
            f"{oracle['median_seconds'] / blend['median_seconds']:.1f}x"
            if oracle["all_converged"]
            else "oracle capped"
        )
        if fastest_name is None:
            fastest_cell = "none converged"
            fastest_ratio = "--"
        else:
            fastest = converged[fastest_name]
            fastest_cell = f"{SHORT[fastest_name]}: {_measurement_cell(fastest)}"
            fastest_ratio = f"{fastest['median_seconds'] / blend['median_seconds']:.1f}x"
        rows.append(
            f"| {record['nx']} | {record['m']} | {_measurement_cell(oracle)} | "
            f"{_measurement_cell(blend)} | {oracle_ratio} | {fastest_cell} | "
            f"{fastest_ratio} |"
        )
    return "\n".join(rows)


def _loose_table(baseline: dict[str, Any]) -> str:
    rows = [
        "| N | m | fastest scalar | scalar time | blend time | blend delta |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    loose_tolerance = max(baseline["config"]["work_tolerances"])
    for record in baseline["work_precision_records"]:
        if record["requested_relative_residual"] != loose_tolerance:
            continue
        methods = record["methods"]
        scalars = {
            name: values
            for name, values in methods.items()
            if name not in {"identity", "tau_blend"} and values["all_converged"]
        }
        best_name = min(scalars, key=lambda name: scalars[name]["median_seconds"])
        best = scalars[best_name]
        blend = methods["tau_blend"]
        delta = 100.0 * (blend["median_seconds"] / best["median_seconds"] - 1.0)
        rows.append(
            f"| {record['nx']} | {record['m']} | {SHORT[best_name]} | "
            f"{best['median_seconds']:.3f} +/- {best['iqr_seconds']:.3f} s | "
            f"{blend['median_seconds']:.3f} +/- {blend['iqr_seconds']:.3f} s | "
            f"{delta:+.1f}% |"
        )
    return "\n".join(rows)


def _batching_summary(batch: dict[str, Any]) -> tuple[float, float, float, float]:
    speedups = [
        row["timing"]["speedup_batched_over_forced_sequential"] for row in batch["apply_records"]
    ]
    full = [
        row["full_solve_speedup_batched_over_forced_sequential"] for row in batch["solve_records"]
    ]
    return min(speedups), max(speedups), min(full), max(full)


def _criterion_summary(runtime: dict[str, Any]) -> tuple[float, float, float, int]:
    tau = [record for record in runtime["records"] if record["method"] == "tau_blend"]
    offline = [record["cost"]["runtime_estimate_time_over_gmres"] for record in tau]
    rejection_record = next(
        record for record in runtime["records"] if record["method"] == "frozen_mean"
    )
    return (
        min(offline),
        max(offline),
        rejection_record["cost"]["runtime_estimate_time_over_gmres"],
        rejection_record["m"],
    )


def _spectral_claims(baseline: dict[str, Any]) -> dict[str, Any]:
    records = baseline["spectral_records"]
    exponents = sorted({record["m"] for record in records})
    effective = {
        m: [
            record
            for record in records
            if record["m"] == m and record["method"] in EFFECTIVE_SCALAR_METHODS
        ]
        for m in exponents
    }
    condition_ranges = []
    spreads = []
    for m in exponents:
        values = [record["spectrum"]["condition_number_2"] for record in effective[m]]
        lower, upper = min(values), max(values)
        condition_ranges.append(f"{lower:.3g}--{upper:.3g}")
        spreads.append(100.0 * (upper / lower - 1.0))
    near_zero = [
        record["spectrum"]["near_zero_eigenvalue_count"]
        for rows in effective.values()
        for record in rows
    ]
    blend = [
        next(record for record in records if record["m"] == m and record["method"] == "tau_blend")
        for m in exponents
    ]
    oracle_is_worst = all(
        next(record for record in effective[m] if record["method"] == "optimized_d0")["spectrum"][
            "condition_number_2"
        ]
        == max(record["spectrum"]["condition_number_2"] for record in effective[m])
        and next(record for record in effective[m] if record["method"] == "optimized_d0")[
            "spectrum"
        ]["minimum_eigenvalue_modulus"]
        == min(record["spectrum"]["minimum_eigenvalue_modulus"] for record in effective[m])
        for m in exponents
    )
    if not oracle_is_worst:
        raise RuntimeError("oracle is no longer spectrally worst among effective scalars")
    return {
        "exponents": "/".join(str(value) for value in exponents),
        "condition_ranges": "/".join(condition_ranges),
        "maximum_condition_spread_percent": max(spreads),
        "near_zero_minimum": min(near_zero),
        "near_zero_maximum": max(near_zero),
        "blend_near_zero_maximum": max(
            record["spectrum"]["near_zero_eigenvalue_count"] for record in blend
        ),
        "blend_condition_numbers": "/".join(
            f"{record['spectrum']['condition_number_2']:.3g}" for record in blend
        ),
    }


def _tight_claims(baseline: dict[str, Any]) -> tuple[str, str]:
    tight_tolerance = min(baseline["config"]["work_tolerances"])
    records = [
        record
        for record in baseline["work_precision_records"]
        if record["requested_relative_residual"] == tight_tolerance
    ]
    ratios = [
        record["methods"]["optimized_d0"]["median_seconds"]
        / record["methods"]["tau_blend"]["median_seconds"]
        for record in records
        if record["methods"]["optimized_d0"]["all_converged"]
    ]
    overview = (
        f"On the clean post-`{baseline['provenance']['base_revision'][:7]}` replay, "
        f"the blend is {min(ratios):.1f}x--{max(ratios):.1f}x faster than the "
        f"converged active-range oracle in the {len(ratios)} cases where that oracle "
        "reaches tolerance."
    )
    capped = []
    for record in records:
        methods = record["methods"]
        if methods["optimized_d0"]["all_converged"]:
            continue
        blend = methods["tau_blend"]
        effective_capped = all(
            not methods[name]["all_converged"] for name in EFFECTIVE_SCALAR_METHODS
        )
        if not effective_capped:
            raise RuntimeError("capped oracle has another converged effective scalar")
        controls = {
            name: methods[name] for name in ("identity", "floor") if methods[name]["all_converged"]
        }
        prefix = (
            f"At `N={record['nx']},m={record['m']}`, no effective active-range scalar "
            "reaches tolerance"
        )
        if controls:
            fastest_name = min(controls, key=lambda name: controls[name]["median_seconds"])
            fastest = controls[fastest_name]
            capped.append(
                f"{prefix}, while {fastest_name} does; the blend takes "
                f"{_iterations(blend)} iterations and about {blend['median_seconds']:.3f} s "
                f"versus {_iterations(fastest)} iterations and about "
                f"{fastest['median_seconds']:.2f} s for that fastest converged control."
            )
        else:
            capped.append(
                f"{prefix}, nor do identity or floor; the blend takes "
                f"{_iterations(blend)} iterations and about {blend['median_seconds']:.3f} s."
            )
    return overview, " ".join(capped)


def _loose_claims(baseline: dict[str, Any]) -> tuple[str, int]:
    loose_tolerance = max(baseline["config"]["work_tolerances"])
    records = [
        record
        for record in baseline["work_precision_records"]
        if record["requested_relative_residual"] == loose_tolerance
    ]
    clauses = []
    for m in sorted({record["m"] for record in records}):
        deltas = []
        for record in records:
            if record["m"] != m:
                continue
            scalars = {
                name: values
                for name, values in record["methods"].items()
                if name not in {"identity", "tau_blend"} and values["all_converged"]
            }
            best = min(scalars.values(), key=lambda values: values["median_seconds"])
            blend = record["methods"]["tau_blend"]
            deltas.append(100.0 * (blend["median_seconds"] / best["median_seconds"] - 1.0))
        if all(delta >= 0.0 for delta in deltas):
            clauses.append(f"{min(deltas):.1f}--{max(deltas):.1f}% slower at `m={m}`")
        elif all(delta <= 0.0 for delta in deltas):
            advantages = [-delta for delta in deltas]
            clauses.append(f"{min(advantages):.1f}--{max(advantages):.1f}% faster at `m={m}`")
        else:
            clauses.append(f"mixed ({min(deltas):+.1f}% to {max(deltas):+.1f}%) at `m={m}`")
    return ", ".join(clauses), len(records)


def _contrast_claims(baseline: dict[str, Any]) -> tuple[str, str]:
    rows = baseline["summaries"]["contrast"]
    ratio_clauses = []
    for m in sorted({row["m"] for row in rows}):
        subset = sorted((row for row in rows if row["m"] == m), key=lambda row: row["step"])
        ratios = [row["best_single_iterations"] / row["tau_iterations"] for row in subset]
        ratio_clauses.append(
            f"from {ratios[0]:.2f}x to {min(ratios[1:]):.2f}--{max(ratios[1:]):.2f}x "
            f"for `m={m}`"
        )
    work = baseline["work_precision_records"]
    contrast = baseline["contrast_records"]
    work_at_max = sum(
        math.isclose(record["optimized_d0_scan"]["ratio_to_active_maximum"], 1.0) for record in work
    )
    contrast_at_max = sum(
        math.isclose(record["optimized_d0_scan"]["ratio_to_active_maximum"], 1.0)
        for record in contrast
    )
    exceptions = sorted(
        record["optimized_d0_scan"]["ratio_to_active_maximum"]
        for record in contrast
        if not math.isclose(record["optimized_d0_scan"]["ratio_to_active_maximum"], 1.0)
    )
    work_count = f"all {len(work)}" if work_at_max == len(work) else f"{work_at_max} of {len(work)}"
    contrast_count = (
        f"all {len(contrast)}"
        if contrast_at_max == len(contrast)
        else f"{contrast_at_max} of {len(contrast)}"
    )
    selection = (
        f"The oracle selects the active-support maximum for {work_count} "
        f"initial-wide-front cases and {contrast_count} contrast states."
    )
    if exceptions:
        selection += (
            f" The remaining {len(exceptions)} are still high in the active range ("
            + " and ".join(f"`{value:.3f}`" for value in exceptions)
            + " of the active maximum), not means."
        )
    return ", and ".join(ratio_clauses), selection


def render() -> str:
    """Render the report after validating that every required input is complete."""
    baseline = _load("pme_dst_tau_blend_single_reference_baselines.json")
    batching = _load("pme_dst_tau_blend_batching_audit_gpu.json")
    criterion = _load("pme_dst_tau_blend_pseudospectral_criterion.json")
    runtime = _load("pme_dst_tau_blend_pseudospectral_runtime_estimator.json")
    for report in (baseline, criterion, runtime):
        if not report.get("complete", False):
            raise RuntimeError(f"incomplete report: {report.get('schema')}")
    apply_min, apply_max, solve_min, solve_max = _batching_summary(batching)
    offline_min, offline_max, rejection, rejection_m = _criterion_summary(runtime)
    spectral = _spectral_claims(baseline)
    tight_overview, tight_capped = _tight_claims(baseline)
    loose_summary, loose_case_count = _loose_claims(baseline)
    contrast_ratios, oracle_selection = _contrast_claims(baseline)
    config = baseline["config"]
    nx_values = "/".join(str(value) for value in config["work_nx_values"])
    m_values = ",".join(str(value) for value in config["work_m_values"])
    tight_tolerance = min(config["work_tolerances"])
    loose_tolerance = max(config["work_tolerances"])
    reference_count = batching["reference_count_static_shape"]
    oracle_scan_count = len(baseline["work_precision_records"]) + len(baseline["contrast_records"])
    oracle_candidate_count = oracle_scan_count * config["oracle_scan_points"]
    blend_tail_count = spectral["blend_near_zero_maximum"]
    blend_tail_description = (
        "zero eigenvalues" if blend_tail_count == 0 else f"at most {blend_tail_count} eigenvalues"
    )
    spectral_paragraph = _wrap(
        f"""The decisive observation is not that the arithmetic mean was chosen poorly.
        Every effective nonzero scalar Helmholtz reference tested here--arithmetic mean,
        bulk mean, constant one, geometric mean, harmonic mean, and the per-case GMRES
        oracle `d0*`--retains a large near-zero spectral tail. At
        `m={spectral['exponents']}`, their condition numbers span only
        `{spectral['condition_ranges']}` and differ by at most
        {spectral['maximum_condition_spread_percent']:.1f}% within an exponent, with
        {spectral['near_zero_minimum']}--{spectral['near_zero_maximum']} eigenvalues below
        magnitude {NEAR_ZERO_THRESHOLD:g}. The oracle `d0*` is spectrally the worst of
        that group: it has the largest condition number and smallest `min|lambda|` at
        every measured exponent. The near-zero `floor` reference and identity do not
        manufacture the tail, but they also provide essentially no useful conditioning
        and have still larger condition numbers. Thus "effective scalar reference"
        means a non-degenerate scalar that actually preconditions: identity and the
        near-zero floor are controls that avoid the tail only by forgoing effective
        conditioning. The {reference_count}-reference blend removes the tail:
        {blend_tail_description} below
        {NEAR_ZERO_THRESHOLD:g}, with `kappa_2={spectral['blend_condition_numbers']}` at
        `m={spectral['exponents']}`."""
    )
    counter_paragraph = _wrap(
        f"""Each cell is `counted GMRES iterations; median +/- IQR seconds`. Timings
        exclude {config['timing_warmups']} warmups, synchronize with
        `jax.block_until_ready`, and include fresh linearization/preconditioner
        construction plus the solve. `cap` means the target was not reached within the
        configured budget. Pavlov identified that the inherited counter reports
        {config['max_krylov_iters'] + 1} at a budget of
        {config['max_krylov_iters']} and explicitly deferred changing that convention
        because doing so would shift recorded counts; this study preserves that known
        item."""
    )
    tight_paragraph = _wrap(f"""The active-support oracle scan uses {config['oracle_scan_points']}
        logarithmically spaced values between `D_min(active)` and `D_max(active)` for
        every case: {oracle_scan_count} independent scans and {oracle_candidate_count}
        candidate solves across the main and contrast studies. {tight_overview}
        {tight_capped}""")
    loose_paragraph = _wrap(
        f"""The result is regime-dependent: the blend is {loose_summary} on this replay.
        These timing separations must be read alongside the table's IQRs.
        {loose_case_count} cases are not a basis for a dispatch rule, and none is
        proposed; an earlier resolution-aware-dispatch exploration was closed as a
        negative result."""
    )
    contrast_paragraph = _wrap(
        f"""The iteration advantage shrinks as compact support fills in but does not
        vanish: the measured oracle/blend ratio falls {contrast_ratios}. It tracks the
        zero-set/degeneracy fraction more coherently than `D95/D05`.
        {oracle_selection} This explains why `frozen_bulk` is generally the strongest
        shipped nontrivial scalar and `frozen_mean` the weakest in these states. It does
        not by itself justify changing moljax's default."""
    )
    batching_paragraph = _wrap(
        f"""The production `l={reference_count}` application was already batched through
        `{batching['production_path']}`. StableHLO confirms a single Helmholtz call over
        the full static `{reference_count} x N` reference tensor rather than a
        per-reference Python loop. The batched and forced-sequential actions agree to
        `{batching['maximum_relative_action_difference']:.2e}` relative error, and all
        solve iteration sets are identical. On GPU, batching speeds up the application
        itself by {apply_min:.2f}--{apply_max:.2f}x; the end-to-end solve speedup is only
        {solve_min:.3f}--{solve_max:.3f}x because the application is not the bottleneck at
        these sizes. Therefore the loose-tolerance deficit is not a hidden
        sequential-transform bug. The `(l,N)` Helmholtz denominator is still formed
        inside each application; hoisting it is an identified, unimplemented
        optimization."""
    )
    criterion_paragraph = _wrap(
        f"""The dense criterion is mathematically sound for these validation-sized
        operators. Its Trefethen resolvent-contour prefactor is exactly
        `L(Gamma_epsilon)/(2*pi*epsilon)`; no Crouzeix spectral-set constant belongs in
        that bound. It certifies the tested origin-enclosed blend cases, but
        certification is offline: the matrix-free runtime estimator costs
        {offline_min:.1f}--{offline_max:.1f}x the corresponding solve, a fundamental
        evidence floor rather than an omitted batching optimization. As a rejection
        gate it can be cheap: rejecting the inadequate `m={rejection_m}` frozen-mean
        reference costs {rejection:.2f}x its solve. Sparse matrix-free Ritz/path mode
        remains provisional and records `rate_bound_available=false`, because reduced
        Ritz values do not prove full spectrum coverage and sparse paths do not provide
        a closed-contour arc length."""
    )
    matrix_free_rate_available = all(
        not record["fully_matrix_free_runtime_estimate"]["rate_bound_available"]
        for record in runtime["records"]
    )
    if not matrix_free_rate_available:
        raise RuntimeError("matrix-free sparse-path records unexpectedly expose a rate bound")

    return f"""# PME DST/tau-blend experimental report

## Scope and claim discipline

This is a measurement report for Newton--Krylov preconditioning of the
node-centred one-dimensional porous-medium equation (PME), using backward Euler
for the linear-solve study and Crank--Nicolson/backward-Euler-compatible residual
and preconditioner machinery. It does **not** apply to IMEX--Strang or ETDRK4:
those methods avoid Newton solves and are unaffected by this work (moljax paper,
Sec. 6.1.3). The experiments use `N={nx_values}`, PME exponents
`m in {{{m_values}}}`, one wide-front backward-Euler linearization for the main
work--precision comparison, and node-centred homogeneous Dirichlet boundaries.

There is no multigrid baseline. Accordingly this report makes **no performance
claim** against the relevant solver state of the art. Wall times below are
capability measurements on this machine, not a recommendation to replace a
multigrid method or moljax's default preconditioner.

## Motivation: where the classical constant reference stops being graceful

Constant-coefficient FFT/DST preconditioning for a variable-coefficient diffusion
operator is classical. QSC/FFT solvers, the variable-coefficient disk solver, and
fast sine-transform analyses obtain mesh-independent or spectrally equivalent
preconditioners when the coefficient is positive and has bounded contrast. The
moljax paper says variable coefficients should "degrade performance gracefully"
(Sec. 6.4, limitation 2). Degenerate PME diffusion is outside that contract:
`D(u)` vanishes on a set of positive measure, so the literal contrast is unbounded.

The tested fixed operator is

`P_tau^-1 r = W0 r + sum_w H(d_w) (W_w r)`,

where the active-support weights form a partition of unity and each `H(d_w)` is a
constant-coefficient DST-I Helmholtz inverse. This is an incremental experimental
remedy, not a claim to a new preconditioner class. Partition-of-unity
preconditioning is established in RBF-PUM and SORAS, while MPGMRES already uses
multiple preconditioners by enlarging the Krylov search space. Unlike MPGMRES, this
blend is one fixed linear operator and retains ordinary GMRES; it does not incur
the multipreconditioned Krylov-space orthogonalisation growth.

## Primary result: the spectral-equivalence boundary

{spectral_paragraph}

{_spectral_table(baseline)}

This is the boundary of the classical bounded-contrast argument in these data:
effective single-reference preconditioning loses spectral equivalence when the
coefficient has a positive-measure zero set; the partitioned blend restores a
zero-free spectrum for the tested states.

## Linear-solve measurements

{counter_paragraph}

{_work_table(baseline)}

### Tight tolerance (`{_scientific(tight_tolerance)}`)

{_tight_table(baseline)}

{tight_paragraph}

These replayed values supersede the preliminary scratch timing headline:
iteration counts for the active oracle mostly reproduce, but the merged GMRES
counter, independent rerun, and current timing samples yield the ratios above.
The committed JSON is authoritative.

### Loose tolerance (`{_scientific(loose_tolerance)}`)

{_loose_table(baseline)}

{loose_paragraph}

## Contrast and degeneracy

{_contrast_table(baseline)}

{contrast_paragraph}

## Batched implementation audit

{batching_paragraph}

## Pseudospectral criterion

{criterion_paragraph}

## Honest limits

- No multigrid baseline; therefore no performance claim.
- One-dimensional, node-centred, homogeneous-Dirichlet PME only, at `N={nx_values}`.
- The main linear study measures one backward-Euler step; inexact-Newton traces
  are supporting evidence, not a full time-to-solution application benchmark.
- Loose-tolerance behavior is regime-dependent, and no switching heuristic is
  proposed.
- The advantage shrinks as the zero set fills in, although it does not vanish in
  the measured contrast sweep.
- A two-dimensional extension is a major build: moljax's `Grid2D` is cell-centred,
  there is no matching 2-D DST-I path, the flux-form linearization differs, and a
  PME front is curve-shaped. It was not attempted here.
- The dense pseudospectral certificate is intentionally offline at
  `N={runtime['config']['nx']}`; its sparse matrix-free approximation is a
  provisional diagnostic only.

## Reproduction and provenance

Run the benchmark entry points with `PYTHONPATH="$PWD"` and Python x64 enabled.
The canonical tables come from
`benchmarks/pme_dst_tau_blend_single_reference_baselines.py`; the batching audit,
dense/sparse pseudospectral studies, validation, spectral analysis, and
inexact-Newton scripts provide the supporting JSONs. Source states use a
generation fingerprint, relocatable cache-relative path, and SHA256 identity;
foreign or damaged artifacts fail closed. Provenance resolves
`merge-base upstream/main -> git describe --tags -> HEAD -> unavailable` and
labels the source. Reassessment reconstructs the stored grid, state, solver, and
preconditioner configuration before accepting an artifact.

Figures are regenerated from the JSONs only:

```bash
PYTHONPATH="$PWD" conda run -n moljax python benchmarks/make_tau_blend_figures.py
```

## References (DOI verified)

1. G. Pavlov, "moljax: GPU-accelerated method of lines for
   stiff reaction-diffusion PDEs with FFT preconditioning," *Computer Physics
   Communications* 326 (2026) 110205. DOI `10.1016/j.cpc.2026.110205`.
2. C. C. Christara and K. S. Ng, "Fast Fourier Transform Solvers and
   Preconditioners for Quadratic Spline Collocation," *BIT Numerical
   Mathematics* 42(4) (2002) 702--739. DOI `10.1023/A:1021944218806`.
3. M.-C. Lai and Y.-H. Tseng, "A fast iterative solver for the variable
   coefficient diffusion equation on a disk," *Journal of Computational Physics*
   208 (2005) 196--205. DOI `10.1016/j.jcp.2005.02.005`.
4. P. De Luca, "Fast Sine-Transform Preconditioning for Global-in-Time
   Fractional Diffusion," *Fractal and Fractional* 10(8) (2026) 573. DOI
   `10.3390/fractalfract10080573`.
5. X. Lin, C. Li, and S. Y. Hon, "Absolute-value based preconditioner for
   complex-shifted Laplacian systems," arXiv:2408.00488 (2024). DOI
   `10.48550/arXiv.2408.00488`.
6. T. Bakhos, P. K. Kitanidis, S. Ladenheim, A. K. Saibaba, and D. B. Szyld,
   "Multipreconditioned GMRES for Shifted Systems," *SIAM Journal on Scientific
   Computing* 39(5) (2017) S222--S247. DOI `10.1137/16M1068694`; preprint DOI
   `10.48550/arXiv.1603.08970`.
7. A. Heryudono, E. Larsson, A. Ramage, and L. von Sydow, "Preconditioning for
   Radial Basis Function Partition of Unity Methods," *Journal of Scientific
   Computing* 67 (2016) 1089--1109. DOI `10.1007/s10915-015-0120-6`.
8. M. Bonazzoli, X. Claeys, F. Nataf, and P.-H. Tournier, "Analysis of the SORAS
   domain decomposition preconditioner for non-self-adjoint or indefinite
   problems," *Journal of Scientific Computing* 89 (2021) 19. DOI
   `10.1007/s10915-021-01631-8`.
9. L. N. Trefethen and M. Embree, *Spectra and Pseudospectra* (Princeton, 2005).
   DOI `10.1515/9780691213101`.
10. M. Embree, "How Descriptive are GMRES Convergence Bounds?", corrected and
    extended from Oxford Technical Report 99/08, arXiv:2209.01231. DOI
    `10.48550/arXiv.2209.01231`. There is no SIAM J. Matrix Analysis and
    Applications version.
"""


def main() -> None:
    """Write the deterministic report from committed measurements."""
    OUTPUT.write_text(render())
    print(OUTPUT.relative_to(REPOSITORY_ROOT))


if __name__ == "__main__":
    main()
