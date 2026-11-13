# Brusselator `dt` sweep: true-residual convention check

This protocol was committed before executing the convention check. It does not alter
the original pre-registered decision record.

## Scope

Use the same 200 CPU cells on the ten committed regenerated `64 x 64` states as
`BRUSSELATOR_DT_SWEEP_CPU_REPLICATION.md`: FFT diffusion and identity at all seven
recorded `dt` values, and the mean-field and split probes at `dt` in `{0.2, 1, 2}`.

## One-sequence stopping measurements

For each cell, run one unrestarted left-preconditioned Krylov sequence, capped at 100
iterations. Record:

- the preconditioned-stop count, which must equal the committed CPU-replication value;
- the true-stop count, the first `k` satisfying
  `||b - J x_k|| <= 1e-8 ||b||`, with `b = -R`;
- the true relative residual at the preconditioned stop.

## Preconditioner conditioning

Record `kappa_2(P)` exactly from the Fourier-mode blocks for FFT diffusion and the
mean-field coupled FFT probe. Identity has condition number one. The split probe is a
composition of a Fourier-diagonal inverse and a spatially varying pointwise reaction
inverse; it is not block diagonal in either basis, so no exact per-mode condition
number is available and the field is reported as unavailable.

## Decision robustness

Apply D1, D2 and D3 verbatim with the true-stop counts; reuse the committed disk rates,
which are independent of the stopping convention. The robustness criterion is that D1,
D2 and D3 return the same PASS/FAIL results as under the preconditioned convention.
The original decision record is unchanged whether or not this criterion holds.
