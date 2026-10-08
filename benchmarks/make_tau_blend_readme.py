#!/usr/bin/env python3
"""Regenerate the tau-blend measurement README from committed JSON files."""

from __future__ import annotations

import json
from pathlib import Path
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
SHORT = {
    "identity": "identity",
    "frozen_mean": "mean",
    "frozen_bulk": "bulk",
    "floor": "floor",
    "const": "const",
    "geometric_mean": "geometric",
    "harmonic_mean": "harmonic",
    "optimized_d0": "oracle d0*",
    "tau_blend": "blend l=3",
}


def _load(filename: str) -> dict[str, Any]:
    return json.loads((RESULTS_DIR / filename).read_text())


def _iterations(values: dict[str, Any]) -> int:
    iterations = values["iterations"]
    return int(iterations[0] if isinstance(iterations, list) else iterations)


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
            f"{record['requested_relative_residual']:.0e} | " + " | ".join(cells) + " |"
        )
    return "\n".join(rows)


def _spectral_table(baseline: dict[str, Any]) -> str:
    rows = [
        "| m | method | kappa_2 | count abs(lambda)<0.1 | min abs(lambda) | "
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
    for record in baseline["work_precision_records"]:
        if record["requested_relative_residual"] != 1.0e-8:
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
    for record in baseline["work_precision_records"]:
        if record["requested_relative_residual"] != 1.0e-2:
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


def _criterion_summary(runtime: dict[str, Any]) -> tuple[float, float, float]:
    tau = [record for record in runtime["records"] if record["method"] == "tau_blend"]
    offline = [record["cost"]["runtime_estimate_time_over_gmres"] for record in tau]
    rejection = next(
        record["cost"]["runtime_estimate_time_over_gmres"]
        for record in runtime["records"]
        if record["method"] == "frozen_mean"
    )
    return min(offline), max(offline), rejection


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
    offline_min, offline_max, rejection = _criterion_summary(runtime)
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
Sec. 6.1.3). The experiments use `N=512` and `N=1024`, PME exponents
`m in {{2,4,8}}`, one wide-front backward-Euler linearization for the main
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

The decisive observation is not that the arithmetic mean was chosen poorly. Every
effective nonzero scalar Helmholtz reference tested here--arithmetic mean, bulk
mean, constant one, geometric mean, harmonic mean, and the per-case GMRES oracle
`d0*`--retains a large near-zero spectral tail. At `m=2/4/8`, their condition
numbers are approximately `1.06e3/3.54e3/6.90e3`, with 170--322 eigenvalues below
magnitude 0.1. The oracle `d0*` is spectrally the worst of that group: it has the
largest condition number and smallest `min|lambda|` at all three exponents. The
near-zero `floor` reference and identity do not manufacture the tail, but they
also provide essentially no useful conditioning and have still larger condition
numbers. The three-reference blend removes the tail: zero eigenvalues below 0.1,
with `kappa_2=11.1/54.8/51.4`.

{_spectral_table(baseline)}

This is the boundary of the classical bounded-contrast argument in these data:
effective single-reference preconditioning loses spectral equivalence when the
coefficient has a positive-measure zero set; the partitioned blend restores a
zero-free spectrum for the tested states.

## Linear-solve measurements

Each cell is `counted GMRES iterations; median +/- IQR seconds`. Timings exclude
two warmups, synchronize with `jax.block_until_ready`, and include fresh
linearization/preconditioner construction plus the solve. `cap` means the target
was not reached within the configured budget (the inherited counter reports 401
at a budget of 400; that reporting convention is deliberately left unchanged).

{_work_table(baseline)}

### Tight tolerance (`1e-8`)

{_tight_table(baseline)}

The active-support oracle scan uses 51 logarithmically spaced values between
`D_min(active)` and `D_max(active)` for every case: 26 independent scans and 1326
candidate solves across the main and contrast studies. On the clean post-`d8d4432`
replay, the blend is 16.0x--54.0x faster than the converged active-range oracle in
the four cases where that oracle reaches tolerance. At `N=1024,m=4`, no
active-range scalar reaches tolerance, while identity/floor do; the blend takes
55 iterations and about 0.456 s versus 330 iterations and about 10.45 s for the
fastest converged comparator. At `N=1024,m=8`, no scalar, floor, or identity reaches
tolerance, while the blend takes 33 iterations and about 0.286 s.

These replayed values supersede the scratch Phase-0 timing headline
`18.4x--62.5x`: iteration counts for the active oracle mostly reproduce, but the
merged GMRES counter, independent rerun, and current timing samples yield the
ratios above. The committed JSON is authoritative.

### Loose tolerance (`1e-2`)

{_loose_table(baseline)}

The result is regime-dependent: the blend is about 15--20% slower at `m=2`,
7--13% slower at `m=4`, and about 2--4% faster at `m=8` on this replay. The
`m=4` timing separation is not large relative to run-to-run variability. Six
cases are not a basis for a dispatch rule, and none is proposed; an earlier
resolution-aware-dispatch exploration was closed as a negative result.

## Contrast and degeneracy

{_contrast_table(baseline)}

The iteration advantage shrinks as compact support fills in but does not vanish:
the measured oracle/blend ratio falls from 7.93x to 3.91--4.50x for `m=2`, and
from 11.59x to 8.31--10.50x for `m=8`. It tracks the zero-set/degeneracy fraction
more coherently than `D95/D05`. The oracle selects the active-support maximum for
all 18 initial-wide-front cases and six of eight contrast states. The other two
are still high in the active range (`0.877` and `0.891` of the active maximum),
not means. This explains why `frozen_bulk` is generally the strongest shipped
nontrivial scalar and `frozen_mean` the weakest in these states. It does not by
itself justify changing moljax's default.

## Batched implementation audit

The production `l=3` application was already batched. StableHLO contains one
Helmholtz call on a `3 x N` tensor and two transform invocations, versus three
Helmholtz calls and six transforms in the forced-sequential comparator. The two
actions agree to `{batching['maximum_relative_action_difference']:.2e}` relative
error and all solve iteration sets are identical. On GPU, batching speeds up the
application itself by {apply_min:.2f}--{apply_max:.2f}x; the end-to-end solve
speedup is only {solve_min:.3f}--{solve_max:.3f}x because the application is not
the bottleneck at these sizes. Therefore the loose-tolerance deficit is not a
hidden sequential-transform bug. The `(l,N)` Helmholtz denominator is still
formed inside each application; hoisting it is an identified, unimplemented
optimization.

## Pseudospectral criterion

The dense criterion is mathematically sound for these validation-sized operators.
Its Trefethen resolvent-contour prefactor is exactly
`L(Gamma_epsilon)/(2*pi*epsilon)`; no Crouzeix spectral-set constant belongs in
that bound. It certifies the tested origin-enclosed blend cases, but certification
is offline: the matrix-free runtime estimator costs {offline_min:.1f}--{offline_max:.1f}x
the corresponding solve, a fundamental evidence floor rather than an omitted
batching optimization. As a rejection gate it can be cheap: rejecting the
inadequate `m=8` frozen-mean reference costs {rejection:.2f}x its solve. Sparse
matrix-free Ritz/path mode remains provisional and records
`rate_bound_available=false`, because reduced Ritz values do not prove full
spectrum coverage and sparse paths do not provide a closed-contour arc length.

## Honest limits

- No multigrid baseline; therefore no performance claim.
- One-dimensional, node-centred, homogeneous-Dirichlet PME only, at `N=512/1024`.
- The main linear study measures one backward-Euler step; inexact-Newton traces
  are supporting evidence, not a full time-to-solution application benchmark.
- Loose-tolerance behavior is regime-dependent, and no switching heuristic is
  proposed.
- The advantage shrinks as the zero set fills in, although it does not vanish in
  the measured contrast sweep.
- A two-dimensional extension is a major build: moljax's `Grid2D` is cell-centred,
  there is no matching 2-D DST-I path, the flux-form linearization differs, and a
  PME front is curve-shaped. It was not attempted here.
- The dense pseudospectral certificate is intentionally offline at `N=512`; its
  sparse matrix-free approximation is a provisional diagnostic only.

## Reproduction and provenance

Run the benchmark entry points with `PYTHONPATH="$PWD"` and Python x64 enabled.
The canonical tables come from
`benchmarks/pme_dst_tau_blend_single_reference_baselines.py`; the batching audit,
dense/sparse pseudospectral studies, validation, spectral analysis, and
inexact-Newton scripts provide the supporting JSONs. Source states use a v4
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

1. G. Pavlov and G. Vourvachakis, "moljax: GPU-accelerated method of lines for
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
