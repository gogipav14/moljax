# Changelog

All notable changes to moljax are documented here.

## [1.2.0] - 2026-09-01

### Added

- **`moljax.conditioning`**: matrix-free conditioning diagnostics, contributed
  by **Georgios Vakis (Vourvachakis)** (IACM and IESL, Foundation for Research
  and Technology Hellas), who is added to `CITATION.cff` as a software author
  and joins the project as a maintainer.

  The subpackage provides a numerical range traced by the Johnson support
  construction, forward-only Arnoldi pseudospectra, non-normality rate
  estimates with the Crouzeix-Palencia envelope, a JFNK linearization adapter
  built from public `jvp`/`vjp` actions, and `assess_preconditioner`, which
  answers whether further preconditioner work is warranted on the states a
  solver actually visits.

  Its only dependency on the rest of moljax is `power_iteration_rho` from
  `moljax.laplace.spectral_bounds`, and it adds no runtime dependency:
  matplotlib remains lazily imported behind the existing `viz` extra.

  The origin of the work is a defect he found in this package: moljax
  documented DCT-I for Neumann boundaries while implementing the cell-centered
  DCT-II symbol, fixed in v1.1.0.

### Fixed

- **Importing moljax no longer changes process-wide JAX precision.**
  `moljax/core/gpu_benchmarks.py` called
  `jax.config.update("jax_enable_x64", True)` at module scope, and because
  `moljax.core` imports it eagerly, merely importing moljax overrode
  caller-owned configuration for the remainder of the process, affecting
  unrelated arrays, compilation, accelerator memory and performance. Callers
  that need float64 must now enable it themselves. The full suite passes
  without the implicit enable, so nothing in the package depended on it.

- **The certificate gates in `moljax.conditioning.field_of_values` floored
  their scale at 1.0 and passed everything for small-magnitude operators.**
  The LOBPCG shift, the eigenpair residual normalization, and the restart
  corroboration spread were all compared against `max(..., 1.0)`, so an
  operator whose rotated Hermitian parts sat below unit magnitude was solved
  as a near-identity perturbation and certified after one LOBPCG step
  regardless of whether the supports had actually converged. `numerical_range`
  now estimates the operator's own scale with a floorless power iteration and
  normalizes every scale-dependent quantity by it, so `supports_converged`
  and `supports_corroborated` agree at every magnitude.

- **`assess_preconditioner` could offer a convergence factor for an operator
  whose numerical range contains the origin.** `estimate_rates` withheld
  `predicted_gmres_factor` only when the numerical-range supports failed
  their own consistency checks; the bulk-clustering estimate `r3`, which
  does not see the numerical range, could still be small and finite when the
  range encloses zero, an operator GMRES is not even guaranteed to converge
  on. `predicted_gmres_factor` is now also withheld whenever
  `fov.origin_enclosed` or the enclosing-disk rate is at or above one.

- **`traced_boundary_rate`'s bisection returned a wrong minimax rate near
  tangency.** The pairwise circle-intersection geometry it relied on carries
  an error that grows without bound as two supporting circles approach
  tangency, well past what a tighter tolerance can fix. It is rewritten as a
  direct nested golden-section search over the complex scaling factor, which
  needs only the objective's convexity and returns the origin-enclosed case's
  exact value of 1.0 directly.

- **The decision demo could draw a residual-decay figure for a preconditioner
  state flagged as needing further work.** The residual-envelope figure was
  drawn whenever a convergence factor was available, which an "investigate"
  verdict can still carry when the failing threshold is unrelated to the
  numerical-range rates. It is now drawn only for an "adequate" or
  "provisional" verdict, and the demo's JSON output now maps non-finite
  readings to `null` under strict `allow_nan=False` serialization instead of
  emitting invalid JSON tokens.

- **Arnoldi's Krylov-breakdown check used a fixed absolute tolerance.**
  `moljax.conditioning.pseudospectra.arnoldi` compared the post-orthogonalization
  residual to a constant `64 * eps`, which is too loose for an operator well
  above unit scale and can be too tight for one well below it. The
  comparison is now relative to that step's pre-orthogonalization norm.

- **Two geometry helpers in `moljax.conditioning._geometry` floored their
  tolerance at a fixed scale.** `_origin_enclosed` floored its coordinate
  tolerance at 1.0, which could call the origin enclosed for a small-magnitude
  hull nowhere near it. `_smallest_enclosing_disk` scaled its tolerance by the
  points' distance from the origin rather than their own spread, and its
  circumcenter arithmetic lost precision for a tight cluster far from the
  origin; both are fixed, the second by working relative to one of the points.

- **`FieldOfValuesResult`'s field order did not match its own docstring.**
  Purely cosmetic: every construction site already used keyword arguments.

- **Fixed-step integrators took the wrong number of steps and ignored `save_every`.**
  `integrate_fixed_dt` called `step_explicit` without the method argument, so
  every call raised `TypeError` (traced by `lax.cond` for implicit methods
  too); both it and `integrate_imex_fixed_dt` computed the step count as
  `int((t_end - t0)/dt) + 1`, overshooting `t_end` by one step or more, and
  ignored `save_every`, emitting every step regardless. The step count is now
  `round((t_end - t0)/dt)` with a `ValueError` when `dt` does not divide the
  interval to 1e-9 relative, and `save_every` batches steps through an outer
  `lax.scan` of `lax.fori_loop` blocks that emit once each.

- **`create_bdf2_residual` scaled only the left-hand side of the variable-step
  BDF2 formula.** The right-hand side kept the old `beta = (1+w)/(1+2w) dt`
  instead of `beta = dt`, so at constant step BDF2 integrated `y' = (2/3)
  F(y)` and refining `dt` did not improve the error. `beta = dt` restores the
  correct scheme (observed order rises from about -0.1 to about 1.9-2.0), and
  adaptive BDF2 problems that previously hit `MAX_STEPS` now complete.

- **`newton_krylov_solve` reported `converged` for the residual it was about
  to leave, not the one it returned.** A solve that met the tolerance on its
  last allowed iteration reported `converged = False`, costing one extra
  Newton (and GMRES) iteration just to observe convergence, and a rejected
  backtracking step kept a stale residual norm. The accepted flag from the
  line-search scan is now kept through to the returned iterate, so
  `res_norm` and `converged` always describe the state that is returned;
  Newton iteration counts drop by one across the board.

- **`etdrk4_step` evaluated coupled reactions on a single-field state.**
  It looped over fields and called `nonlinear_rhs` on a one-field dictionary
  at each stage, so any reaction coupling fields (Gray-Scott's `u v^2`, for
  one) raised `KeyError`. Each stage is now formed for every field before
  `nonlinear_rhs` is evaluated on the complete state, matching ETD1 and ETD2;
  single-field results are bit-identical to before.

- **`imex_strang_step` and `imex_ssprk2_step` were first order despite being
  documented as second order.** The Strang half-steps solved diffusion with
  backward Euler instead of the exact exponential, and the SSP2 step
  averaged two backward-Euler stages rather than a consistent IMEX pairing.
  Strang now applies `exp(dt/2 D Laplacian)` through a new
  `apply_diffusion_exp_fft` with Heun stages at `t` and `t + dt`, and
  `imex_ssprk2_step` is the Pareschi-Russo IMEX-SSP2(2,2,2) scheme; observed
  orders rise from about 1.0 to about 2.0-2.1 on a manufactured
  diffusion-reaction problem.

- **Inhomogeneous Neumann boundaries were wrong beyond the first ghost
  layer, and mislabeled as an outward-normal convention.** Every ghost
  layer used the same `2 dx` offset instead of its own distance `(2i+1) dx`
  from the interior, and the flux is `du/dx` in `+x` on both faces, not an
  outward normal. A linear profile is now reproduced exactly at any ghost
  width.

- **The adaptive backward-Euler error estimate could not fall below
  tolerance.** It compared the BE step against `dt * F(y_be)`, the size of
  the update rather than of its error, so every step was rejected until
  `MAX_STEPS`. The estimate is now the difference to a Crank-Nicolson step,
  BE's actual local error; a `y' = -y` run that previously stalled now
  completes in 149 accepted steps.

- **`heisenberg_cfl_dt` and `imex_cfl_dt` always returned the default float
  type.** A float32 model run under x64 fed a float64 CFL limit into a
  `lax.cond` whose other branch was float32, and JAX refused to trace it.
  Both functions take a `dtype` keyword, passed as the model's dtype from
  the adaptive integrators.

- **The implicit steps' explicit-Euler predictor could poison an otherwise
  well-posed step.** `be_step` and `cn_step` seed Newton from an
  explicit-Euler predictor; a right-hand side singular at the start time but
  finite at the end of the step (or an overflow far past the explicit
  stability limit) makes that predictor non-finite, and Newton seeded with
  `inf` or `nan` returns `nan`. The predictor now falls back to the current
  state when it is not finite; it is otherwise unchanged, so existing GMRES
  iteration counts do not move.

- **`build_wavenumbers_2d_rfft` swapped the x and y wavenumbers.** It
  returned the y frequencies under the name `kx` and the x frequencies under
  `ky`; the Laplacian symbol hid the swap by pairing each with the matching
  wrong spacing, but the advection symbol did not, disagreeing with the
  full-spectrum path by 2.1e-3. `FFTDiffusionPreconditioner` and
  `FFTAdvectionDiffusionPreconditioner` also ignored an rfft cache's
  `use_rfft` flag (raising `TypeError` when given one) and computed the
  speed of a 2D velocity tuple with a bare `abs()`. Both preconditioners now
  route through rfft2/irfft2 when the cache calls for it, and a first
  derivative symbol's non-Hermitian Nyquist bin is zeroed on both paths
  (a 5.5e-3 imaginary leak before, at rounding level after).

- **`create_advection_diffusion_model` stored `field_names` as whatever
  sequence the caller passed.** Every other model factory stores a list in
  `metadata['field_names']`; this one kept the caller's tuple, which broke
  callers that serialize or extend the metadata as a list.

- **The `phi` functions used a threshold too small for float32, and were
  duplicated in two modules.** `phi1`/`phi2`/`phi3` switched from a Taylor
  series to a cancelling direct formula at `|z| = 1e-4` regardless of
  precision; in float32 the direct formula is unusable well above that
  point (an 11,000% relative error at `|z| = 1.1e-4`). `jit_kernels.py` now
  has one 16-term Horner-Taylor implementation with a precision-dependent
  switch (`|z| < 0.5` in float64, `|z| < 2.0` in float32), imported by
  `fft_integrators.py` instead of duplicated.

- **The slow `dst_I`/`idst_I` pair was not an inverse of itself.** `dst_I`
  doubled its coefficients while `idst_I` divided by `N + 1` rather than by
  `2/(N + 1)`, so `idst_I(dst_I(x))` returned `2x`.

- **`exact_cfl_dt('explicit')` returned a positive, unstable step for pure
  advection.** It used `2/rho` on the spectrum's magnitude, ignoring that
  forward Euler is unconditionally unstable for a purely imaginary
  eigenvalue. It now takes the per-eigenvalue bound from
  `|1 + dt * lambda| <= 1` and returns 0 when every eigenvalue is purely
  imaginary.

- **Newton-Krylov and preconditioner documentation described behavior the
  code did not have.** `NKStats.lin_iters` is the GMRES budget made
  available, not iterations spent (`jax.scipy.sparse.linalg.gmres` returns
  no count); norms are unweighted 2-norms of the flattened residual with no
  grid weighting; `estimate_error_doubling`'s docstring claimed its error
  was scaled by `1/(2^p - 1)`, which the code does not do.

- **`nilt_solve_linear_pde` now inverts every Fourier mode numerically.** It
  previously returned the closed form `exp(lambda t) u0_hat` as both
  `u_final` and `u_analytical`, and the only inversion it ran was a scalar
  NILT of the `k = 0` mode, so the bridge test passed without any inversion
  taking place and `compare_nilt_vs_timestepping` reported a NILT error of
  zero. Every mode is now inverted in one `nilt_fft_batch` call, with the
  `t = 0` jump subtracted analytically and complex modes handled as two real
  transforms; the result agrees with the closed form to 3e-9 on the 256-point
  diffusion case. `tss_steps` reports the number of steps actually taken.

- **The adaptive tuner's quality sensors can fire.** `tune_nilt_adaptive`
  classified on the imaginary leakage of the ifft, which is rounding noise
  once the spectrum is mirrored into Hermitian symmetry, so a transform
  sampled at dt = 2 (74% error) was "good" and one at dt = 0.02 was "poor".
  `nilt_fft_uniform` now reports `band_edge_ratio` and
  `tail_energy_fraction` from the sampled transform, `QualityTier` carries
  them together with `tail_ratio` and `r_late` (dropping `eps_im_valid`,
  `r_early`, `spike_ratio`), the tier classifier is exported as
  `classify_quality_tier`, and `retune_based_on_diagnostics` returns
  "at max_N, no change" instead of re-running an identical inversion.
  `quality_metrics.classify_quality` takes the two sensors;
  `integrate_with_adaptive_tuner` (no caller) is removed.

- **The CFL-guided tuner converges.** `check_spectral_cfl_conditions`
  sampled the tail on a grid running twice past the Nyquist frequency, one
  scalar at a time, floored the endpoint scale at 1 (passing any transform
  of amplitude below 1%), and used tolerances the default tuner cannot meet;
  `tune_nilt_adaptive_cfl` re-evaluated the endpoint jump after switching to
  half-step sampling and switched again every iteration. The check now uses
  the k = 0..N/2 grid in one vectorized call, a jump relative to the signal
  scale, and defaults (`tau_chi = 2.0`, `tau_tail = 1e-2`) the tuner meets;
  the endpoint condition is settled by the switch. exp(-t) ends "good" at
  iteration 0 instead of "poor" after three.

- Tests that asserted nothing now do: coarse coverage must be worse than fine
  coverage in the second-order example, the Hermitian projection must remove
  a 1% non-Hermitian perturbation, the N-clamp warning must be present, and
  the retuning tests assert the action taken and the parameter it changed.

- **`nilt_with_smoothing` returns the right magnitude.** It sampled the
  transform on a one-sided grid running to twice the Nyquist frequency and
  scaled by `1/(2T)` instead of `N/(2T)`, so its output was about `N` times
  too small (exp(-t) read 3.5e-3 where 0.527 was expected, the unit step
  0.013). It now evaluates the half grid, applies the first `N//2 + 1`
  sigma-factors and inverts through the same Hermitian mirroring and scaling
  as `nilt_fft_uniform`, which it matches to 1e-12 with smoothing off.

- **Talbot, Weeks and Gaver-Stehfest inversions return correct values.**
  `talbot_method` used the wrong contour constant and an extra `pi/N`
  factor (0.098 of the true value at N = 32, divergent at N = 64);
  `weeks_method` integrated through a pole of its integrand and returned NaN;
  `gaver_stehfest_method` formed its 1e8-magnitude weights in float32 when
  x64 was off and returned values off by O(1). Talbot now follows Weideman
  and Trefethen (2007), Weeks follows Weideman (1999) with FFT coefficients
  and the missing `e^{-bt}` factor, and Gaver-Stehfest forms its weights
  exactly and requires 64-bit precision through the new
  `moljax._precision.require_x64`, which the conditioning subpackage now
  uses as well. exp(-t) is reproduced to 5e-14 (Talbot, N = 32), rounding
  (Weeks, 32 terms) and 1e-6 (Gaver-Stehfest, 14 terms).

- **The NILT overflow guard judges the dtype the grid actually has.** With
  x64 off, a float64 request runs in float32, and `exp(a t)` overflowed at
  `a t_max = 88.7` while the guard, reading the declared dtype, allowed up to
  699.8; the tuner's parameters for an unstable operator returned 152
  non-finite values out of 256 without an error. The guard now uses the
  grid's dtype, names it in the error, and also covers the half-step
  variants and `nilt_with_smoothing`, which had none.

- **`nilt_fft_with_pole_at_origin` requires a positive shift, and
  `invert_laplace` tunes its defaults.** The default `a = 0` evaluated
  `s F(s)` at the pole itself and returned NaN everywhere; `a` is now a
  required positive argument. `invert_laplace` replaced its fixed defaults
  (which raised the overflow guard for `t_end` above about 39) with
  `tune_nilt_params`, and accepts `bounds`. `estimate_nilt_truncation_error`
  honors `dtype`; the unused `_nilt_core_jit` and a dead `N == 16384` check
  in `diagnose_tuning` are removed; the `projection_threshold` docstring
  describes what the code does (it labels, it does not gate).

- **Benchmark scripts honor `--backend`.** 25 scripts (11 under
  `benchmarks/`, 14 under `benchmarks/sisc/`) parsed or ignored the flag and
  then called `setup_benchmark(expected_backend="gpu")` regardless, so
  `--backend any` still aborted with "Expected gpu backend, got cpu" and the
  CPU reproduction path documented in REPRODUCE.md did not exist. Every
  script now derives the expected backend from the flag, the seven scripts
  that had no argument parsing gained the standard `--n-reps`/`--backend`
  parser, and `run_all.sh` and `run_sisc_suite.sh` forward `--backend`
  (default `gpu`) to every stage.
- **The SISC suite runs from a clean checkout.** `run_sisc_suite.sh` ran the
  scripts from `benchmarks/sisc/` without putting `benchmarks/` on the path,
  so every stage died with `ModuleNotFoundError: benchmark_utils`; under
  `set -e`, the `((PASSED++))` bookkeeping would also have aborted the suite
  after its first passing stage. The runner now exports `PYTHONPATH`, counts
  without arithmetic-command side effects, creates `benchmarks/figures/`,
  accepts `--backend`, and the scripts write to `benchmarks/results/sisc/`
  (tracked) instead of the gitignored `benchmarks/sisc/results/`.
  REPRODUCE.md names the real paths.

- **The SISC GMRES iteration counts were the number of JAX traces (a
  constant 5), not GMRES iterations**, because the counter lived inside a
  matvec that JAX traces a fixed number of times regardless of the system.
  The six affected scripts now count with `scipy.sparse.linalg.gmres`
  (`callback_type='pr_norm'`) on the same system, keep JAX GMRES for wall
  time where one is reported, and record `iteration_source`; a second,
  independent bug in `bench_jvp_vs_fd_sweep.py` (the right-hand side
  preconditioned twice) is fixed in the same commit, and REPRODUCE.md now
  says the committed `iter_vs_grid.json`/`iter_vs_dim.json` predate the fix.
  Verified on CPU: iteration counts vary sensibly by preconditioner and
  system instead of reading a flat 5 or 200.

- **The README quick start now runs.** Every snippet called constructors,
  factories or keywords that do not exist; it is rewritten against the real
  API and each snippet (and every example) was executed on CPU before
  committing, alongside the corrected bibtex, JAX/CI badges and install
  instructions.
- **The examples silently ran in float32** despite building float64 state;
  each now enables `jax_enable_x64` before creating any array.
- CONTRIBUTING.md and REPRODUCE.md no longer contradict the code or CI:
  lint is required and nothing is deselected (653 tests, verified
  by a full run), and REPRODUCE.md names one GPU (RTX 5060) and the real
  SISC script count (13) and paths.

- **A non-finite or missing NILT quality sensor classified as `'good'`.**
  `adaptive_tuning.classify_quality` decided the tier from `>` comparisons
  against threshold values; a NaN sensor (`F_eval` returning NaN or inf on
  the Bromwich contour, or a diagnostics dict missing the sensor keys)
  fails every such comparison and fell through to the best tier, so
  `tune_nilt_adaptive` could report a non-finite inversion as a successful
  `'good'` result. The same fallthrough existed in the differently-shaped
  `quality_metrics.classify_quality`. Both now treat a missing or
  non-finite sensor as a failure (`'poor'`, and `QualityLevel.FAILED`, an
  enum value that existed but was never returned, in `quality_metrics`),
  naming the offending sensor in the reason; the documented case where
  `assess_nilt_quality` is called without `F_vals` samples still defers to
  the wraparound sensor unchanged.

- **A failed Newton-Krylov line search could apply a step it had already
  shown made the residual worse.** `newton_step`'s backtracking `lax.scan`
  tries `alpha`, `alpha * backtrack_factor`, ... against the Armijo-style
  decrease test, but when none of them was accepted the fallback took
  `x + alpha * dx` at the original, undamped `alpha`: the very first,
  worst candidate the scan had already rejected, applied without ever
  being checked against the decrease test a second time. `newton_step` now
  tracks the best (lowest-residual) candidate seen across the backtracking
  scan, seeded with the starting iterate itself, and falls back to that
  instead: a failed line search can no longer move to a larger residual
  than where it started, and reports unchanged if nothing tried improved
  on it.

- **`nilt_solve_linear_pde` silently mishandled a spectrum that was not
  1D.** The bridge reads `n_modes` from `eigenvalues.shape[0]` and applies
  a 1D `fft`/`irfft` throughout, so a 2D spectrum (e.g. from a 2D
  `DiffusionOperator`) either failed with an opaque broadcasting error deep
  inside `nilt_fft_batch` or, depending on shapes, could reconstruct a
  field of the wrong size instead of failing at all. It now raises a
  `ValueError` up front naming the unsupported shape when `eigenvalues` or
  `u0` is not 1D; the docstring documents the restriction and 2D support
  is out of scope.

- **`tune_nilt_adaptive_cfl` could rate an all-NaN inversion as
  'acceptable'.** At the iteration limit, the tier was picked from
  `len(cfl.violated_conditions)` alone: a NaN `band_edge_ratio` fails the
  `R_tail <= tau_tail` comparison, which counts as exactly one CFL
  violation, so a transform where every sensor was NaN and `result.f` was
  entirely non-finite could report 'acceptable' with no warning at all.
  The finite-sensor rejection `classify_quality` applies lived in a
  different function and never ran on this path. Every return from
  `tune_nilt_adaptive_cfl` now goes through `_quality_from_diagnostics`,
  which shares `classify_quality`'s missing-or-non-finite check
  (`_missing_or_non_finite_reason`) and additionally rejects a non-finite
  `result.f`, overriding the caller's tier to 'poor' and naming the
  offending sensor in the reason.

- **`nilt_solve_linear_pde` did not check that `eigenvalues`, `u0` and
  `source` had matching lengths.** The 1D checks above reject the wrong
  number of dimensions but not the wrong number of modes: a 4-mode
  spectrum with a 1-element `u0` broadcast into a 4-element field with 3
  fabricated modes, a 1-mode spectrum with a 4-element `u0` silently
  returned only 1 element, and a mismatched `source` was not checked at
  all and failed later with an opaque broadcasting error inside
  `nilt_fft_batch`. It now requires a nonempty `eigenvalues`, `u0.shape ==
  eigenvalues.shape`, and, when given, `source.shape == eigenvalues.shape`,
  raising a `ValueError` naming the mismatch before any FFT runs.

- **`newton_krylov_solve` with `max_backtrack=0` silently turned Newton
  into a no-op.** `newton_step`'s best-candidate fallback initializes
  `best_x_flat`/`best_r_norm` from the starting iterate and only updates
  them inside the backtracking `lax.scan`; with `max_backtrack=0` the scan
  runs its body zero times, so the fallback always returned the untouched
  starting iterate no matter what `dx_flat` was, regardless of
  `NKParams.damping`. `max_backtrack=0` now means "no line search": the
  configured damped step is applied unconditionally and its residual
  evaluated, so `res_norm`/`converged` still describe the returned
  iterate.

### Changed

- pyproject.toml: author email matches CITATION.cff, `ruff==0.16.5` in the
  `dev` extra, and E402 waived per file for `tests/**`, `benchmarks/**` and
  `examples/**` instead of project-wide.

### Notes

- The diagnostics report an outer bound on the numerical range that is
  conditional, not certified: it holds only if each sampled support is the
  true maximum in its direction, which no fixed eigensolver start can prove.
  The same condition governs `origin_enclosed`, since each sampled support
  is a Rayleigh quotient and so a lower bound on the true one.
  `FieldOfValuesResult.supports_consistent` records that every check that
  was run passed, `corroboration_attempted` records whether independent
  restarts were among them, and `RateEstimates` and
  `PreconditionerAssessment` carry both so a serialized result stays
  self-describing. `assess_preconditioner` abstains with `indeterminate`
  when a check failed or a diagnostic input is unusable (fewer than four
  Ritz values, a non-finite Ritz value, a NaN or negative reading, an
  infinite `epsilon_zero`), and answers
  `provisional` rather than `adequate` when every gate passed but no
  restart was run.
  `numerical_range(..., n_restarts=n)` raises the number of starts when a
  verdict is load bearing; the default of one attempts no corroboration.

## [1.1.1] - 2026-08-03

### Fixed

- **`etd_integrate` now compiles its time-stepping loop.** It previously
  stepped through an eager Python `for` loop, so every step paid full
  XLA dispatch. Long integrations were pathologically slow: a 3,276-step
  ETDRK4 run took 41 s, and the NILT-bridge comparison tests built on it
  ran for many minutes to hours.

  Stepping now runs inside `lax.fori_loop` when only the endpoint is
  retained and `lax.scan` when intermediate states are saved. The method
  dispatch is hoisted out of the loop; ETD2's first step is still taken
  eagerly because it seeds the `N_prev` term from `None`.

  Output is bit-identical to the previous implementation, verified across
  all three methods and seven `save_every` / step-count combinations.
  `tests/test_fft_nilt_bridge.py` went from hanging past 900 s to passing
  in 87 s.

- `compare_nilt_vs_timestepping` requested every intermediate state and
  then used only the last one, which for long horizons on fine grids
  allocated hundreds of MB it immediately discarded. It now retains only
  the endpoint, and calls `block_until_ready` so its reported timings
  are not measuring asynchronous dispatch.

- **Table 3 had archived data but no script to regenerate it.** No file
  in the repository produced the four-column CuPy / nvmath / JAX
  comparison; `benchmark_cupy_fft.py` covers only CuPy versus JAX. Adds
  `benchmarks/benchmark_fft_3lib.py`, which regenerates
  `results/fft_3lib_comparison.json`. CuPy and nvmath are optional; a
  missing library yields a null column instead of aborting.

  On the current stack (nvmath 1.0.0, JAX 0.9.1 rather than the paper's
  0.8/0.8.2) the regenerated values track the published ones at 256²,
  512² and 1024². They differ at 64² and 128², where the measurement is
  launch-latency dominated and the paper reports interquartile ranges as
  large as the medians. The table's conclusion is unaffected: JAX inside
  a compiled loop is competitive with CuPy, so the speedups are
  algorithmic rather than kernel-level.

### Changed

- The two NILT-bridge tests marked `slow` in 1.1.0 are no longer slow and
  run by default again. `pytest` no longer deselects `slow` tests; the
  marker remains registered for the pre-existing one in
  `test_option_a_vs_b.py`, which also runs by default again.

### Known issues

- `compare_nilt_vs_timestepping` reports `tss_steps` from
  `ceil(t_end / dt)` while `etd_integrate` floors internally, so it
  integrates to `floor(t_end/dt) * dt` rather than exactly `t_end`. This
  is pre-existing and accounts for the residual ~9e-5 relative error in
  the time-stepping leg of an otherwise exact linear propagation.

## [1.1.0] - 2026-08-03

Reproducibility release. Brings the public repository in line with the
published paper (*Computer Physics Communications* **326** (2026) 110205,
[doi:10.1016/j.cpc.2026.110205](https://doi.org/10.1016/j.cpc.2026.110205))
and corrects a boundary-condition discrepancy between the paper and the
code.

Release `v1.0.0` (commit `25cd9a3`) remains the archival artifact cited by
the paper and is unchanged.

### Fixed

- **Neumann boundary conditions now use the node-centered DCT-I described
  in Section 3.1.1 of the paper.** The code previously implemented the
  cell-centered DCT-II symbol, `-4/dx² sin²(πk/(2N))`, while the paper
  specified DCT-I. This was also internally inconsistent: the Dirichlet
  path uses the node-centered DST-I symbol, so a mixed Dirichlet/Neumann
  problem combined two different grid layouts.

  `BCType.NEUMANN` now selects the node-centered DCT-I form,
  `-4/dx² sin²(πk/(2(N-1)))`, whose eigenvectors exactly diagonalize the
  `[-2, 2]/dx²` end-row stencil. The previous behavior is available as
  `BCType.NEUMANN_CELL`, or via `centering='cell'` on the affected
  functions.

  Thanks to Georgios Vakis (Vourvachakis), IACM/IESL-FORTH and University
  of Crete, who identified this while building a coupled two-temperature
  solver on moljax.

- Package version reported `0.1.0` despite the paper citing release
  v1.0.0. Now `1.1.0` in both `pyproject.toml` and `moljax.__version__`.

- README described the paper as submitted to the *Journal of
  Computational Physics* under a previous title. Corrected to the
  published CPC reference with DOI.

- `REPRODUCE.md` listed JAX 0.4.35 / CUDA 12.x / Ubuntu 22.04, which
  matched neither the paper nor the environment the benchmarks ran in.

### Added

- **Node-centred DCT-I transforms**: `dct_I`, `idct_I`, `dct_I_2d`,
  `idct_I_2d`. JAX exposes only the type-2 cosine transform, so these are
  built from the real FFT of the symmetric even extension of length
  `2N-2`. Verified against `scipy.fft.dct(type=1)` to machine precision.

- Layout-explicit Neumann API: `laplacian_symbol_neumann_node` /
  `_cell`, `solve_poisson_neumann_node` / `_cell`,
  `solve_helmholtz_neumann_node` / `_cell`, `etd1_neumann_node` /
  `_cell`, plus a `centering` argument on the original names.

- `tests/test_dct_i_neumann.py` (44 tests): parity with SciPy, exact
  stencil diagonalization, constant-mode preservation, inverse
  normalization, separable 2D composition, JIT and `grad`
  compatibility, second-order manufactured convergence, and confirmation
  that `centering='cell'` reproduces the pre-1.1.0 symbol exactly.

- **Schnakenberg and Brusselator systems** (`create_schnakenberg_model`,
  `create_brusselator_model`, `schnakenberg_reaction_op`,
  `brusselator_reaction_op`, and the periodic-FFT variants). These
  produce Tables 9 and 10 and were absent from the public repository, so
  two of the paper's three benchmark systems could not previously be
  reproduced from it.

- **15 benchmark and figure scripts** that generate published tables and
  figures but were missing here: the Schnakenberg and Brusselator
  benchmarks, the ablation and FFT-vs-sparse studies, the split
  Gray-Scott legs, the work-precision sweeps, and the pattern gallery,
  attractor divergence and reactor steep-gradient figure generators.
  Their result files, including the `wp_schnakenberg.json` cited in the
  paper, are included.

- `benchmarks/run_all.sh` and `benchmarks/plot_main_figures.py`, the two
  entry points named in the paper's reproduction quickstart. Neither
  existed; `run_all.sh` in particular was named `run_all_benchmarks.sh`
  and covered 10 of the benchmarks, omitting Schnakenberg, Brusselator,
  the work-precision sweeps and the OFAT, ablation, CuPy-FFT and
  JIT-factorial studies.

- `environment-current.yml`, tracking the stack moljax is actively
  developed against, alongside the paper-exact `environment.yml`.

- `rfft2` half-spectrum path for real 2D fields (`use_rfft=True` by
  default on `DiffusionOperator` and the 2D FFT cache), with matching
  ETD and Helmholtz kernels.

### Changed

- `benchmarks/run_all_benchmarks.sh` now forwards to `run_all.sh`.

- `pytest` deselects tests marked `slow` by default; run them with
  `pytest -m slow`, or everything with `pytest -m ""`. Two NILT-bridge
  tests are newly marked `slow` because `compare_nilt_vs_timestepping`
  integrates tens of thousands of ETDRK4 steps through an eager Python
  loop and runs for many minutes to hours. A third test in
  `test_option_a_vs_b.py` carried a `slow` marker that was never
  registered or honored before, and is now deselected too.

### Notes

- `use_rfft=True` changes floating-point output at the round-off level
  relative to v1.0.0 for 2D periodic problems. Pass `use_rfft=False` for
  bit-comparable behavior.

- The default Neumann layout change alters results for code that relied
  on `BCType.NEUMANN` meaning cell-centered. The domain length implied by
  `N` and `dx` differs between layouts: `(N-1)·dx` for node-centered
  versus `N·dx` for cell-centered.

## [1.0.0] - 2026-03

Release accompanying the CPC paper. Archival commit `25cd9a3`.
