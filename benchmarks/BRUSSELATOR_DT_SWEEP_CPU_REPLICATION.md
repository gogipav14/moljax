# Brusselator fixed-state `dt` sweep: CPU replication protocol

This protocol was committed before regenerating source states or executing any
CPU replication cell.  The original Stage-3 v4 source states existed only in
workstation scratch and were deleted.  This run therefore cannot reproduce
their SHA-256 identities.  Regenerated states receive new, host-labelled
identities and are treated as distinct realisations.

## Scope

The replication covers only the ten tracked `64 x 64` states:

- `developed64:hopf:1`, `developed64:hopf:10`, and
  `developed64:hopf:20`;
- `developed64:turing:80`, `developed64:turing:120`, and
  `developed64:turing:200`;
- `screen64:hopf:0`, `screen64:hopf:1`, `screen64:turing:0`, and
  `screen64:turing:1`.

The two `256 x 256` fixed-step states are excluded.

## Similarity classification

For every state, compute

`abs(S_regenerated - S_GPU) / abs(S_GPU)`,

where `S` is `max_x ||J_R(x)||_2` and the GPU value is read from
`benchmarks/results/brusselator_dt_sweep.json`.  If every relative difference
is at most `1e-6`, call the outcome **reproduction**.  Otherwise call it
**replication on regenerated realisations**.  This classification is fixed
before inspecting the regenerated values.

## Cells and stopping protocol

Use the same counted-GMRES implementation as the GPU sweep, with true relative
residual tolerance `1e-8` and iteration cap `100`.

For all ten states, run GMRES only for:

- FFT diffusion and identity at `dt` in `{0.05, 0.1, 0.2, 0.5, 1, 1.5, 2}`;
- mean-field coupled FFT and split reaction-after-diffusion at `dt` in
  `{0.2, 1, 2}`.

Run disk diagnostics only for the twelve cells required by D3: FFT diffusion
and mean-field coupled FFT on the six developed states at `dt = 1`.  CPU wall
times are diagnostic and are not compared with the original GPU timings.

## Pre-registered decision rule

Build a reaction-aware FFT preconditioner for backward-Euler Brusselator only if ALL hold:
 D1 (material degradation): on the six developed64 states, the median of I_FFT(dt=1) / median_{dt<=0.2} I_FFT is >= 2, and at least 5 of the 6 states are individually >= 2.
 D2 (absolute cost): the median of I_FFT(dt=1) over the six developed64 states is >= 25.
 D3 (recoverability): at least one probe lowers the developed64 median I(dt=1) by >= 40% relative to FFT, with all of its GMRES runs converged, AND has a lower disk rate than FFT on at least 4 of the 6 developed64 states.
Reported, but not part of the decision: the identity arm; within-state monotonicity in dt (for a fixed state, S_R is proportional to dt, so its correlation with iterations is monotonicity only); the cross-state Spearman correlation over the 64² states at dt = 1 between I_FFT and max_x ||J_R(x)||_2; and per cell, whether the disk/FoV ordering agrees with the measured iteration ordering.
A failed rule is reported as a negative result: either FFT preconditioning remains adequate at the dt moljax uses, or no cheap reaction-aware variant recovers the loss.

## Success criterion

The CPU replication succeeds if D1, D2, and D3 have the same PASS/FAIL results
as the original GPU run.  The regenerated-state similarity classification is
reported independently and is not adjusted after seeing the results.
