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
- **Measurement.** Unrestarted GMRES to a true relative residual of `1e-8`, capped at
  100 iterations (a capped run is recorded as 101 and retained as non-converged
  evidence); disk diagnostics on the ladder `32/120/2`, `64/180/2`, `96/240/2`.
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

Across the ten 64x64 states at `dt = 1`, the Spearman correlation between FFT
iterations and `max_x ||J_R(x)||_2` is 0.372. The two `screen64` Hopf states have
`max ||J_R|| = 4.41` and need 35 and 37 iterations; `developed64:turing:200` has
`max ||J_R|| = 16.54` and needs 20. The Hopf states are consistently harder than the
Turing states at comparable or smaller reaction magnitude.

Source: `BRUSSELATOR_DT_SWEEP_RESULTS.md`, "Cross-state reaction correlation".

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

## 5. Interpretation (ours, informal, not tested here)

For the mean-field preconditioner `P_mf = I - dt (D Delta + Jbar_R)`, which is
block-diagonal in Fourier space, the same algebra as the premise check gives

`P_mf^-1 J - I = -dt P_mf^-1 (J_R(x) - Jbar_R)`.

Its departure from the identity is therefore driven by the *spatial variation* of the
reaction Jacobian, not by its size. This is consistent with the table in Section 4: the
near-homogeneous early states (`screen64`, `fixed256:turing:1`) need 3 to 4 iterations
at `dt = 1`, while the developed Hopf states, which carry the most spatial structure,
need 18 to 23. It would also explain why `max ||J_R||` predicts FFT poorly. The sweep
did not record `max_x ||J_R(x) - Jbar_R||`, so this explanation is untested.

Two cautions follow from the same expression. First, `P_mf` can itself be nearly
singular: a Fourier block `I - dt (D lambda_k + Jbar_R)` is singular when
`D lambda_k + Jbar_R` has the real eigenvalue `1/dt`, which the Turing-unstable modes
make possible. A shipped version must check the smallest singular value over the
blocks. Second, the identity depends on the FFT diagonalizing the diffusion operator
exactly, so it holds for periodic boundaries only.

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
