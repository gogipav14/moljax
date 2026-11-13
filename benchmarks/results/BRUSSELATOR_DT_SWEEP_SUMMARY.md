# Brusselator fixed-state backward-Euler dt sweep: summary

## Question

Stage 3 left seven Brusselator states on which the shipped FFT-diffusion preconditioner
was `indeterminate`; six of them were assessed at `dt = 1`. This study asks whether
that loss tracks the reaction Jacobian the preconditioner omits, and whether a cheap
reaction-aware variant recovers it. It measures; it does not build. The protocol and
the decision rule were committed before the run
(`benchmarks/BRUSSELATOR_DT_SWEEP_PROTOCOL.md`, 22ebad1).

## Setup

- **States.** Twelve SHA256-verified Stage-3 source states, reused exactly: the seven
  FFT-indeterminate states (six `developed64`, one `fixed256` Turing state) and five
  matched controls (four `screen64`, one `fixed256`). Changing `dt` changes the
  backward-Euler linearization, never the state.
- **Main arm.** `dt` in `{0.05, 0.1, 0.2, 0.5, 1, 1.5, 2}`, shipped FFT-diffusion and
  identity preconditioners: 168 cells.
- **Probe arm.** Two diagnostic preconditioners, local to the benchmark script and not
  shipped, at `dt` in `{0.2, 1, 2}`: 72 cells.
  1. *Mean-field coupled FFT*: per Fourier mode, the exact solve with
     `I - dt diag(D_u lambda_k, D_v lambda_k) - dt Jbar_R`, where `Jbar_R` is the grid
     mean of the pointwise 2x2 reaction Jacobian.
  2. *Split*: the shipped FFT Helmholtz inverse, then `(I - dt J_R(x))^-1` pointwise.
- **Measurement.** Unrestarted GMRES to the relative residual of the
  left-preconditioned system,
  `||P^-1 (b - J x)|| <= 1e-8 ||P^-1 b||`, capped at 100 iterations. Each
  preconditioner therefore uses its own norm; for identity this is the true residual.
  The protocol's phrase "true-relative-residual protocol" names this inherited
  `_counted_gmres` check. A capped run is recorded as 101 and retained as
  non-converged evidence; disk diagnostics use the ladder `32/120/2`, `64/180/2`,
  `96/240/2`.
- **Premise check.** With periodic boundaries the FFT preconditioner `P` inverts the
  diffusion part exactly, so `P^-1 J = I - dt P^-1 J_R`. All 72 checks hold, with a
  maximum relative defect of `3.518680569e-15` (threshold `1e-12`). Any degradation of
  `P^-1 J` is therefore reaction-driven. Source:
  `brusselator_dt_sweep.json#/v0/maximum_relative_defect`.

All 240 cells completed, with no failure, skip or truncation. 51 GMRES runs did not
converge (45 main, 6 probe) and are kept as non-converged evidence. 7 disk assessments
stayed unresolved at the top rung, all identity cells of
`developed64:turing:120`; they are excluded wherever geometry is interpreted.

Source: `brusselator_dt_sweep.json`; `BRUSSELATOR_DT_SWEEP_RESULTS.md`,
"Completeness and gates".

## 1. Pre-registered decision: BUILD

| state | FFT iterations, dt <= 0.2 | median | FFT at dt = 1 | ratio | mean-field probe at dt = 1 |
|---|---|---:|---:|---:|---:|
| `developed64:hopf:1` | 5, 7, 8 | 7 | 37 | 5.29 | 5 |
| `developed64:hopf:10` | 6, 8, 11 | 8 | 50 | 6.25 | 23 |
| `developed64:hopf:20` | 6, 8, 11 | 8 | 39 | 4.88 | 18 |
| `developed64:turing:80` | 5, 5, 7 | 5 | 18 | 3.60 | 5 |
| `developed64:turing:120` | 6, 7, 9 | 7 | 23 | 3.29 | 10 |
| `developed64:turing:200` | 5, 7, 8 | 7 | 20 | 2.86 | 10 |

- **D1 (material degradation): pass.** Median ratio 4.2375 (threshold 2); 6 of 6
  states at least 2 (required 5).
- **D2 (absolute cost): pass.** Median FFT count at `dt = 1` is 30 (threshold 25).
- **D3 (recoverability): pass.** The mean-field probe lowers the median to 10, a
  reduction of 66.7% (required 40%). All six runs converge, and its disk rate is lower
  than FFT's on 6 of 6 states (required 4). The split probe fails D3: median 36.5
  (21.7% *more* than FFT), one non-converged run at `dt = 1`, and a higher disk rate on
  all 6 states.

Source: `brusselator_dt_sweep.json`, `/analysis/decision/*`;
`BRUSSELATOR_DT_SWEEP_RESULTS.md`, "Pre-registered decision".

## 2. The degradation is a dt effect on every state, not a property of the seven

The controls degrade the same way. At `dt = 1` the FFT preconditioner needs 35 and 37
iterations on the two `screen64` Hopf controls, 22 on both `screen64` Turing controls
and 22 on the early `fixed256` Turing state. The counts rise monotonically with `dt` on
all twelve states. On all five Hopf states FFT fails to reach `1e-8` within 100
iterations at `dt = 2`, and on four of them already at `dt = 1.5` (the exception,
`developed64:hopf:20`, needs 72). No Turing state caps with FFT: at most 48 iterations
at `dt = 2`.

So the Stage-3 `indeterminate` verdicts at `dt = 1` record the timestep at which those
states were assessed. They do not single out the states.

Source: `BRUSSELATOR_DT_SWEEP_RESULTS.md`, "Within-state monotonicity",
"Non-converged GMRES cells".

## 3. Reaction magnitude predicts poorly; regime predicts well

Across the ten 64x64 states at `dt = 1`, the Spearman correlation between FFT iterations
and `max_x ||J_R(x)||_2` is 0.372, and 0.523 when the same Krylov sequences are stopped
on the physical residual (Section 5). The two `screen64` Hopf states have `max ||J_R|| =
4.41` and need 35 and 37 iterations; `developed64:turing:200` has `max ||J_R|| = 16.54`
and needs 20. Regime separates the counts completely: at `dt = 1` every Hopf state needs
35 to 50 FFT iterations and every Turing state 18 to 23 (34 to 51 and 19 to 24 under the
physical criterion). The two regimes also differ in their reaction parameters and
diffusion scaling (`kappa_2(P_FFT) - 1` is five times larger on the Turing states at
every `dt`), so the separation does not identify a mechanism.

The ten states are fewer than ten independent observations. `developed64:hopf:1` and the
two `screen64` Hopf states are near-identical near-homogeneous states (`max ||J_R||`
4.411 to 4.417, maximum reaction variation 0.022 to 0.028), and so are the two
`screen64` Turing states. The correlations here and in Section 6 are descriptive.

Source: `BRUSSELATOR_DT_SWEEP_RESULTS.md`, "Cross-state reaction correlation";
`brusselator_dt_sweep_posthoc.json`,
`/spearman_at_dt_1/max_reaction_jacobian_2_norm/fft_diffusion/true_stop`,
`/true_residual_descriptive`, `/states/*`.

## 4. The mean-field probe helps everywhere it was measured

In all 36 probe cells the mean-field coupled FFT needs fewer iterations than FFT, and
all 36 converge, including `dt = 2` on the five Hopf states where FFT caps.

| state | dt = 0.2: FFT / probe | dt = 1: FFT / probe | dt = 2: FFT / probe |
|---|---:|---:|---:|
| `screen64:hopf:0` | 8 / 4 | 35 / 4 | cap / 9 |
| `screen64:hopf:1` | 8 / 4 | 37 / 4 | cap / 9 |
| `screen64:turing:0` | 8 / 3 | 22 / 4 | 38 / 4 |
| `screen64:turing:1` | 9 / 3 | 22 / 4 | 38 / 4 |
| `fixed256:turing:1` | 8 / 3 | 22 / 3 | 38 / 4 |
| `developed64:hopf:1` | 8 / 4 | 37 / 5 | cap / 11 |
| `developed64:hopf:10` | 11 / 9 | 50 / 23 | cap / 57 |
| `developed64:hopf:20` | 11 / 9 | 39 / 18 | cap / 29 |
| `developed64:turing:80` | 7 / 4 | 18 / 5 | 31 / 5 |
| `developed64:turing:120` | 9 / 6 | 23 / 10 | 48 / 15 |
| `developed64:turing:200` | 8 / 6 | 20 / 10 | 36 / 13 |
| `fixed256:turing:1000` | 9 / 6 | 20 / 10 | 37 / 14 |

Source: `brusselator_dt_sweep.json`, `/cells/probe|STATE|dt=DT|mean_field_coupled_fft/gmres`
and the corresponding `/cells/main|STATE|dt=DT|fft_diffusion/gmres` records.

The disk diagnostic agrees with this ranking: for the mean-field probe against FFT,
the sign of the disk-rate change equals the sign of the iteration change in 31 of 31
eligible pairs. It is unreliable for the other comparisons: 27 of 44 for identity
against FFT, and 15 of 30 for the split probe against FFT.

The split probe's failure is not caused by singular pointwise blocks: the minimum
`|det(I - dt J_R(x))|` over its 36 cells is 0.383, and no cell was skipped.

Source: `BRUSSELATOR_DT_SWEEP_RESULTS.md`, "Disk/FoV ordering",
"Split-probe determinant diagnostic".

## 5. The decision is robust to the physical true-residual convention

A pre-registered CPU follow-up continued each of the same 200 Krylov sequences until
both stopping criteria were resolved. Every preconditioned-stop count equals its
committed replication value. Under the physical criterion
`||b - J x|| <= 1e-8 ||b||`, D1 still passes with median ratio `4.4625` and all six
states above two; D2 still passes with median `31`; and D3 still passes because the
mean-field probe lowers that median to `11`, a `64.5%` reduction, with all runs
converged and all six disk rates lower. The decision remains **BUILD**.

The mean-field Hopf cells at `dt = 2` illustrate the small count shift: the
preconditioned/physical counts are `11/12`, `57/58`, and `29/30` for developed steps
1, 10, and 20. Exact `kappa_2(P)` is recorded from the Fourier blocks for FFT and the
mean-field probe. It is unavailable from per-mode blocks for the split composition,
which combines a Fourier-diagonal inverse with a spatially varying pointwise inverse.
Over the 200 cells, `kappa_2(P)` for the FFT preconditioner rises with `dt` from 2.31 to
53.4 on the Hopf states and from 7.55 to 263 on the Turing states; for the mean-field
probe it ranges from 9.25 to 3.62e4. These values bound the gap between the two criteria
only loosely. Among converged cells, the physical relative residual at the
preconditioned stop never exceeds 3.9e-8 for FFT or 2.7e-8 for the mean-field probe,
including the three `dt = 2` Hopf cells where `kappa_2(P_mf)` is 3.5e4 to 3.6e4. The
split probe's own norm is the least faithful: at its preconditioned stop the physical
residual reaches 7.1e-6 (`developed64:turing:200`, `dt = 2`), its physical stop comes up
to 29 iterations later, and `developed64:hopf:20` at `dt = 1`, which converges in 88
iterations under its own norm, does not converge within 100 under the physical
criterion. Its physical-stop median at `dt = 1` is 45.5 against FFT's 31, so it still
fails D3.

Source: `brusselator_dt_sweep_true_residual.json`, `/decision_true_stop`,
`/decision_comparison`, and `/cells/*`; `BRUSSELATOR_DT_SWEEP_TRUE_RESIDUAL.md` gives
all 200 cells. The check took `354.167` CPU seconds; this is diagnostic laptop timing
only; `brusselator_dt_sweep_posthoc.json#/true_residual_descriptive`.

## 6. Exploratory (post hoc): reaction variation and mean-field block conditioning

This hypothesis was formed after seeing the sweep results and is not part of the
pre-registered decision. For
`P_mf = I - dt (D Delta + Jbar_R)`, the exact periodic-grid identity is

`P_mf^-1 J - I = -dt P_mf^-1 (J_R(x) - Jbar_R)`.

The regenerated-state reaction-variation measurements are:

| state | max variation | mean variation | block min, dt=0.2 | dt=1 | dt=2 |
|---|---:|---:|---:|---:|---:|
| `developed64:hopf:1` | 0.0271952 | 0.00593731 | 0.549835 | 0.116935 | 0.00161396 |
| `developed64:hopf:10` | 4.98704 | 1.84528 | 0.648728 | 0.340094 | 0.389519 |
| `developed64:hopf:20` | 6.42996 | 2.45416 | 0.703885 | 0.470331 | 0.631036 |
| `developed64:turing:80` | 0.0856456 | 0.0238243 | 0.860779 | 0.754358 | 0.713572 |
| `developed64:turing:120` | 2.61143 | 0.319198 | 0.865109 | 0.770466 | 0.74499 |
| `developed64:turing:200` | 14.3875 | 2.23165 | 0.992833 | 1.26418 | 1.71374 |
| `screen64:hopf:0` | 0.0281197 | 0.00590851 | 0.549859 | 0.116952 | 0.00154478 |
| `screen64:hopf:1` | 0.0217664 | 0.00478816 | 0.549859 | 0.116953 | 0.00154222 |
| `screen64:turing:0` | 0.0184995 | 0.00320226 | 0.860751 | 0.754258 | 0.713372 |
| `screen64:turing:1` | 0.0132127 | 0.00234353 | 0.860751 | 0.75426 | 0.713376 |

At `dt = 1`, the Spearman correlations over the ten states are:

| predictor | preconditioner | preconditioned stop | physical stop |
|---|---|---:|---:|
| `max ||J_R||_2` | FFT | 0.371958 | 0.522799 |
| `max ||J_R||_2` | mean-field | 0.786724 | 0.786724 |
| `max ||J_R - Jbar_R||_2` | FFT | 0.152442 | 0.303953 |
| `max ||J_R - Jbar_R||_2` | mean-field | 0.862249 | 0.862249 |

The sample is only `n = 10`, with the near-duplicate states noted in Section 3, and the
iteration counts contain ties, so these correlations are descriptive. The two predictors
are themselves rank-correlated at 0.927 over these states, so 0.862 against 0.787 does
not distinguish them. The observation consistent with the identity is narrower: on the
three near-homogeneous Hopf states `max ||J_R||` is about 4.41 and FFT needs 35 to 37
iterations at `dt = 1` (34 to 38 under the physical criterion), while the maximum
reaction variation is at most 0.028 and the mean-field probe needs 4 or 5. Testing
variation as a predictor would need states chosen for that purpose.

The nearly singular mean-field Fourier blocks occur on the near-homogeneous Hopf
states at `dt = 2`, with minimum singular values about `1.5e-3`. Every Turing block
minimum stays above `0.71`. A prospective shipped implementation would therefore need
to check these blocks, but this post-hoc observation is not a performance claim.
The identity above also relies on periodic FFT diagonalisation.

Source: `brusselator_dt_sweep_posthoc.json`, `/states/*` and
`/spearman_at_dt_1/*`, `/spearman_between_predictors`.

## Scope and limits

- Fixed states only: each cell is one backward-Euler linearization, not a trajectory or
  a time-to-solution measurement.
- 64x64 and 256x256 grids; the only 256x256 states are Turing states.
- One tolerance (`1e-8`), unrestarted GMRES capped at 100 iterations.
- The probes are measuring instruments, not tuned preconditioners; there is no
  wall-time comparison. The 3.44 GPU-hours recorded include compilation and the disk
  diagnostics.
- **Relevance.** The only non-conditioning implicit Brusselator benchmark is
  `benchmarks/benchmark_brusselator.py`, whose Newton--Krylov Crank--Nicolson path uses
  `dt = 0.05` (lines 156--163). The conditioning study itself uses backward-Euler
  `dt = 0.05, 0.1, 0.2, 1` in `benchmarks/brusselator_conditioning.py` (preset
  definitions at lines 70--126). There is no implicit Brusselator example under
  `examples/`, and no adaptive Brusselator BE or BDF2 run. At `dt <= 0.2` FFT needs 5
  to 11 iterations on these states, and at `dt = 0.2` the probe saves 2 to 6 per solve.
  The large gains occur at `dt >= 1`, which is not demonstrated by a non-conditioning
  implicit trajectory here.

## Reproduction

The original v4 source states existed only in workstation scratch and were
deleted during the workstation migration. CPU regeneration from the recorded
generation fingerprints does not reproduce their SHA-256 identities; stored
state identities are therefore host-specific. The commands below remain the
record of the original GPU run and continue to fail closed if supplied states
do not match those original identities.

The pre-registered CPU check regenerated the ten `64 x 64` states under new,
host-labelled identities. Although every regenerated SHA-256 differs from the
original, the maximum relative difference in `max_x ||J_R(x)||_2` is
`3.6182827810178155e-12`, below the pre-registered `1e-6` threshold. The run is
therefore classified as a **reproduction** on that state-similarity criterion.
Both runs used JAX and jaxlib 0.11.2; the GPU run used Python 3.12.13 and NumPy 2.5.2,
the CPU run Python 3.12.3 and NumPy 2.5.4. The two `fixed256` states were neither
regenerated nor committed: their rows in Sections 2 and 4 rest on the original GPU run
alone, and they do not enter D1 to D3.

All 200 GMRES iteration counts and convergence flags match the GPU result. D1,
D2 and D3 remain PASS: the developed-state median degradation ratio is
`4.2375`, the `dt = 1` FFT median is `30`, and mean-field coupled FFT reduces
that median to `10` while lowering all six corroborated disk rates. The CPU
decision is consequently still **BUILD**. The recorded end-to-end CPU time is
`1741.084510564804` seconds, including an interrupted-and-resumed checkpoint;
it is diagnostic only and is not comparable with the original GPU timing.
These values are recorded in
`benchmarks/results/brusselator_dt_sweep_cpu_replication.json` and rendered
cell by cell in `benchmarks/results/BRUSSELATOR_DT_SWEEP_CPU_REPLICATION.md`.

The CPU replication entry points are:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH="$PWD" \
.venv/bin/python benchmarks/brusselator_dt_sweep.py \
  --regenerate-source-states benchmarks/results/brusselator_dt_sweep_cpu_states

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH="$PWD" \
.venv/bin/python benchmarks/brusselator_dt_sweep_cpu_replication.py \
  --source-state-dir benchmarks/results/brusselator_dt_sweep_cpu_states \
  --output benchmarks/results/brusselator_dt_sweep_cpu_replication.json \
  --markdown-output benchmarks/results/BRUSSELATOR_DT_SWEEP_CPU_REPLICATION.md

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH="$PWD" \
.venv/bin/python benchmarks/brusselator_dt_sweep_true_residual.py \
  --source-state-dir benchmarks/results/brusselator_dt_sweep_cpu_states

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH="$PWD" \
.venv/bin/python benchmarks/brusselator_dt_sweep_posthoc.py \
  --source-state-dir benchmarks/results/brusselator_dt_sweep_cpu_states
```

The original GPU sweep used the following entry points and explicit paths:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH="$PWD" \
python benchmarks/brusselator_dt_sweep.py \
  --source-state-cache-dir /tmp/moljax-brusselator-stage2-d4047f3-v4-r1/source_states \
  --output /tmp/moljax-brusselator-dt-sweep.json

PYTHONPATH="$PWD" python benchmarks/analyze_brusselator_dt_sweep.py \
  /tmp/moljax-brusselator-dt-sweep.json \
  --output benchmarks/results/brusselator_dt_sweep.json

PYTHONPATH="$PWD" python benchmarks/make_brusselator_dt_sweep_results.py \
  --input benchmarks/results/brusselator_dt_sweep.json \
  --output benchmarks/results/BRUSSELATOR_DT_SWEEP_RESULTS.md
```

The sweep entry point requires a GPU, JAX x64 and
`XLA_PYTHON_CLIENT_PREALLOCATE=false`. The source states are loaded from the Stage-3 v4
cache and verified by fingerprint and SHA256; a mismatch fails closed.
