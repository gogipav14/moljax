# Brusselator fixed-state backward-Euler dt sweep protocol

## Question and scope

This measurement asks whether the shipped FFT-diffusion preconditioner's loss
of certification at large backward-Euler timesteps tracks the pointwise
reaction Jacobian that the preconditioner omits.  Every row reuses an exact,
SHA256-verified Stage-3 source state: changing `dt` changes the linearized
backward-Euler operator but never regenerates or advances the state.  The two
reaction-aware preconditioners are diagnostic instruments local to
`benchmarks/brusselator_dt_sweep.py`; neither is a shipped API.

The source-state cache is the Stage-2/3 v4 cache.  The benchmark rejects a
missing artifact, an absolute or traversing artifact path, a generation-
fingerprint mismatch, a manifest mismatch, or a state SHA256 mismatch.

## States

The twelve states are ordered so the six developed 64x64 states run first,
then the two fixed-dt 256x256 Turing states, then the four 64x64 screen
controls.

| ID | tracked report | regime | step/state | source time | role |
|---|---|---:|---:|---:|---|
| `developed64:hopf:1` | `brusselator_conditioning_developed.json` | Hopf | 1 | 1 | FFT-indeterminate |
| `developed64:hopf:10` | same | Hopf | 10 | 10 | FFT-indeterminate |
| `developed64:hopf:20` | same | Hopf | 20 | 20 | FFT-indeterminate |
| `developed64:turing:80` | same | Turing | 80 | 80 | FFT-indeterminate |
| `developed64:turing:120` | same | Turing | 120 | 120 | FFT-indeterminate |
| `developed64:turing:200` | same | Turing | 200 | 200 | FFT-indeterminate |
| `fixed256:turing:1000` | `brusselator_conditioning_fixed_dt.json` | Turing | 1000 | 200 | FFT-indeterminate |
| `fixed256:turing:1` | same | Turing | 1 | 0.2 | matched adequate control |
| `screen64:hopf:0` | `brusselator_conditioning.json` | Hopf | state 0 | 0.1 | adequate control |
| `screen64:hopf:1` | same | Hopf | state 1 | 0.2 | adequate control |
| `screen64:turing:0` | same | Turing | state 0 | 0.1 | adequate control |
| `screen64:turing:1` | same | Turing | state 1 | 0.2 | adequate control |

## Arms and ordering

The V0 premise check runs first.  For every state at `dt` in `{0.2, 1}` and
three deterministic random float64 vectors it verifies

`||(P^-1 J - I)x + dt P^-1 J_R x|| / ||x|| <= 1e-12`,

where `P` is the shipped FFT-diffusion preconditioner and `J_R` is the exact
JVP of the reaction term after the model's periodic boundary map.  Any failure
terminates the benchmark before a sweep cell is run.

The main arm contains exactly 168 cells:

- all twelve states;
- `dt` in `{0.05, 0.1, 0.2, 0.5, 1, 1.5, 2}`;
- shipped FFT-diffusion and identity preconditioners.

The probe arm contains 72 cells: all twelve states at `dt` in `{0.2, 1, 2}`
with each of the following diagnostic-only preconditioners.

1. **Mean-field coupled FFT.**  Let `Jbar_R` be the physical-grid mean of the
   pointwise 2x2 reaction Jacobian.  Each Fourier mode solves
   `I - dt*diag(D_u*lambda_k, D_v*lambda_k) - dt*Jbar_R` exactly.
2. **Split reaction-after-diffusion.**  Apply the shipped FFT Helmholtz inverse
   first, then apply `(I - dt*J_R(x))^-1` pointwise on the physical grid.  A
   cell is skipped, with the determinant recorded, when
   `min_x |det(I - dt*J_R(x))| < 1e-12`.

Execution order is V0; main developed64; main fixed256; main screen64; probe
developed64; probe fixed256; probe screen64.  Within a state, timesteps are
ascending and preconditioners are in the order listed above.

## Per-cell measurements

Every completed cell records the exact state identity and operator dimension;
`dt`; `dt*max_x ||J_R(x)||_2`; `max_x ||J_R(x)||_2`; counted unrestarted GMRES
convergence, iterations, breakdown flag and final true relative residual at
`1e-8`; reduced-Arnoldi epsilon-zero and right-real outlier count; disk rate,
disk center/radius, origin enclosure, support residual and support
convergence/corroboration flags; all attempted FoV budgets; and build, GMRES,
Arnoldi, FoV and total wall seconds.  Probe cells carry the same fields as the
main arm plus their diagnostic-preconditioner metadata.

GMRES uses the Stage-3 true-relative-residual protocol with tolerance `1e-8`
and a 100-iteration cap; a capped run is retained as non-converged evidence.

FoV resolution starts at `32 angles / 120 iterations / 2 restarts`.  Only a
record with inconsistent supports is escalated to `64/180/2`, then
`96/240/2`; the first consistent result is final.  The residual tolerance is
`1e-3`.  All random keys are deterministic.  Output is atomically checkpointed
after V0 and after every cell, so interruption loses at most the in-flight
cell.

## Budget

`XLA_PYTHON_CLIENT_PREALLOCATE=false` is mandatory and no other process may
use the GPU during the sweep.  Total GPU process time is capped at 5.0 hours.
The checkpoint records a budget truncation and all unstarted cell IDs.  The
benchmark runs under an external five-hour timeout as a second hard guard.

## Pre-registered decision rule

Build a reaction-aware FFT preconditioner for backward-Euler Brusselator only if ALL hold:
 D1 (material degradation): on the six developed64 states, the median of I_FFT(dt=1) / median_{dt<=0.2} I_FFT is >= 2, and at least 5 of the 6 states are individually >= 2.
 D2 (absolute cost): the median of I_FFT(dt=1) over the six developed64 states is >= 25.
 D3 (recoverability): at least one probe lowers the developed64 median I(dt=1) by >= 40% relative to FFT, with all of its GMRES runs converged, AND has a lower disk rate than FFT on at least 4 of the 6 developed64 states.
Reported, but not part of the decision: the identity arm; within-state monotonicity in dt (for a fixed state, S_R is proportional to dt, so its correlation with iterations is monotonicity only); the cross-state Spearman correlation over the 64² states at dt = 1 between I_FFT and max_x ||J_R(x)||_2; and per cell, whether the disk/FoV ordering agrees with the measured iteration ordering.
A failed rule is reported as a negative result: either FFT preconditioning remains adequate at the dt moljax uses, or no cheap reaction-aware variant recovers the loss.

The script records raw measurements only.  It does not evaluate this rule
during the run; analysis waits until the checkpoint is complete or budget-
truncated.
