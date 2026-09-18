# Changelog

All notable changes to moljax are documented here.

## [Unreleased]

### Added

- **The adaptive NILT tuners verify the resolved bandwidth with a second
  inversion at `dt/2` instead of trusting a ratio that cannot see the
  exponential amplification.** `tune_nilt_adaptive`'s accuracy budget
  `A_max = amplification_tolerance / eps_machine` bounds the amplified
  rounding floor, and nothing bounded the amplified *truncation* error: the
  tail of `F` beyond `pi/dt` comes back multiplied by `exp(a t_end)` too, and
  `band_edge_ratio`, a ratio, cannot see a factor that scales its numerator
  and denominator alike. With the numerical abscissa 49 of
  `J = [[-1, 100], [0, -1]]` and its off-diagonal resolvent
  `F(s) = 100/(s + 1)^2`, the `t_end = 1/3` window passed every condition
  (`A_exp = 1.24e9` under the budget `4.5e9`, `band_edge_ratio` 0.011) and
  came back "good, all sensors within normal range" at a relative RMS error
  of 600. No closed-form condition separated that from a healthy grid: the
  rigorous tail bound is one to four orders of magnitude pessimistic there
  and the cheap heuristics correlate with the true error only within two
  decades. The check is therefore a measurement. New
  `compute_richardson_difference` (exported) inverts a second time with `dt`
  halved and `N` doubled at fixed `T` and fixed `a`, which doubles the
  resolved band and leaves the coarse samples at the refined grid's even
  indices, and returns the relative RMS gap between the two on the coarse
  samples in `[0, t_end]`, undamped: on the window above the damped gap
  reads 1.9e-2 and the undamped one 7.7, so damping the comparison, as the
  wraparound sensor does for its own reasons, would hide exactly the term
  this sensor exists to find. The gap tracks the true relative error within a
  factor of three over five decades (0.05 to 1/3 windows of the `J` above:
  measured 2.9e-4, 3.2e-3, 4.5e-2, 6.4e-1, 8.4e+0 against true 4.2e-4,
  3.3e-3, 4.6e-2, 6.2e-1, 8.3e+0) and the new
  `truncation_tolerance` default 0.3 is the geometric midpoint of the gap
  between the loosest transform the tuner rates good today (0.128,
  `e^{-t} cos 2t` over `t_end = 10`) and the tightest window that must fire
  (0.638, `t_end = 0.2`, true error 0.62). On disagreement the ladder applies
  the bandwidth remedy, doubling `N` and re-verifying up to `N_max` and the
  iteration budget, and only then reports `poor` with a reason beginning
  "unresolved truncation:" plus a `UserWarning`, never `good`; the measured
  gap is recorded on `QualityTier.richardson_difference` (new field,
  defaulting to NaN) whether it passes or not. `tune_nilt_adaptive` runs the
  check after the CFL conditions and `classify_quality` both pass;
  `tune_nilt_adaptive_cfl` runs it at the one exit that would otherwise
  report "all CFL conditions satisfied", comparing the uniform pilot rather
  than the half-step inversion, whose grid `t = (n + 1/2) dt` shares no
  sample but `t = 0` with its own refinement. Both take
  `verify_truncation: bool = True`; it costs one extra inversion at twice the
  size per iteration (measured 1.3x total tune time over a twelve-transform
  battery) and is on by default because the alternative is a confident
  verdict on an answer that is hundreds of times wrong.
  `verify_truncation=False` reproduces the previous behavior exactly,
  including the documented gap. The `t_end = 1/3` window is the only verdict
  that changes: `good` to `poor`, with the true error dropping from 6.0e+2 to
  8.6 as the ladder widens the band; the neighboring `t_end = 0.2` window
  stays `good` and its true error falls from 0.62 to 0.075 because one
  doubling resolves it.
  `tests/test_adaptive_tuning_quality.py::TestTruncationVerification` covers
  the caught window, the flag-off reproduction of the gap, the undamped
  choice, the grid alignment, the recorded field, the CFL path and a
  non-finite inversion (inf, not NaN, so no threshold comparison silently
  passes).

### Changed

- **`numerical_range`'s LOBPCG restart seeding now depends on the operator
  being diagnosed, and the `adequate` docstring is honest about what
  restart agreement establishes.** `_largest_hermitian_eigenvector`'s
  random starting columns were seeded only from the sweep angle and the
  restart index, both fixed regardless of which operator was being traced,
  so an operator whose dominant eigenspace happened to be orthogonal to
  those fixed columns at every angle and restart a call used was an exact
  blind spot for every caller, not an unlucky one. A 64x64 construction
  with such a planted blind spot (Codex conditioning.md finding 1,
  2026-09-14: `A = I + 4 u v^T + 1e-7 P D P`, `D = diag(linspace(-1, 1,
  64))`, `P = I - u u^T - v v^T`, `u`, `v` orthogonal to every starting
  column the old seed formula would use) drove `disk_rate` toward zero and
  `origin_enclosed` toward `False` on an operator whose true numerical
  range is the disk of radius 2 centered at 1, which contains the origin;
  the audit's own verification could not turn this into a stable false
  `adequate` (the residual gate tripped first in every trial it ran), so
  the concrete risk was the docstring's promise that agreement across
  restarts is more than corroboration. Fixed by hashing the sign pattern of
  a fixed probe vector's image under the operator (`matvec(ones(n))`) into
  a `numerical_range(..., operator_key=...)` argument that seeds every
  restart, so a fixed construction can no longer be blind to the seed
  without already depending on the very hash it would have to predict; the
  sign pattern (rather than the raw floating-point values) keeps the
  default deterministic under a positive real rescale of the operator; an
  earlier version of this fix hashed the raw values instead, and
  `test_numerical_range_scale_invariance`'s `disk_rate` disagreed across
  its sixteen decades of scale, because a decimal rescale changes a
  float's low bits even though the operator is "the same" up to that
  factor, and hashing is exquisitely sensitive to exactly those bits. The
  `PreconditionerAssessment.adequate` docstring now cross-references
  `FieldOfValuesResult.supports_corroborated`'s existing hedge: agreement
  across restarts is corroboration, not proof.
  `tests/test_field_of_values.py::test_default_operator_key_ties_the_seed_to_the_operator`
  and `::test_numerical_range_accepts_an_explicit_operator_key` exercise the
  new seeding mechanism directly and fail against the pre-fix code (no such
  function/argument existed);
  `::test_orthogonal_blind_spot_construction_no_longer_hides_the_enclosed_origin`
  reproduces the conditioning.md construction as a regression guard, though
  -- consistent with the audit's own experience -- it does not reliably
  distinguish the pre-fix seed formula from the post-fix one on its own,
  since a second restart's fixed seed can happen not to be blind for a
  given construction either way.

- **`pseudospectra.arnoldi` returns a structured `ArnoldiResult`, and
  `non_normality.assess_preconditioner` can be told whether `epsilon_zero`
  covers the full operator.** `arnoldi` used to return a bare `(Q, H)`
  tuple with no record of how many Krylov steps actually completed, so a
  reduced `epsilon_zero` from an Arnoldi breakdown was indistinguishable
  from the full operator's smallest singular value: `assess_preconditioner`
  took bare `ritz` and `epsilon_zero` values and had no way to tell.
  `A = diag([.05, .6, .65, .7, .75, .8, .85, .9])` plus `0.02` on the first
  superdiagonal except the `(0, 1)` entry, `v0 = [0, 1, 1, 1, 1, 1, 1, 1]`,
  `k = 8`: the start vector has no component on the decoupled first degree
  of freedom, so Arnoldi breaks down at dimension 7 with
  `epsilon_zero(H) = 0.598`, while `sigma_min(A) = 0.05`; `numerical_range`
  gives `disk_rate = 0.895`, and the assessment read `adequate` from the
  reduced value where the true value reads `investigate`. This is not
  adversarial: any start vector with no component on a decoupled degree of
  freedom triggers it. `arnoldi` now returns an `ArnoldiResult` recording
  `basis`, `hessenberg`, `k_requested`, `k_achieved`, `breakdown`, and the
  forward-factorization residual `||A Q[:, :k_achieved] - Q H||` as a
  coverage indicator (indexing and the first two fields' order are
  unchanged, so `result[0]`/`result[1]` still give `basis`/`hessenberg`,
  but a plain `Q, H = arnoldi(...)` unpacking no longer works).
  `assess_preconditioner` gained `coverage: ArnoldiResult | None = None`
  and `full_operator_lower_bound: bool = False`: a reduced `epsilon_zero`
  may now support `adequate` only when `coverage.k_achieved` equals the
  operator's dimension or the caller asserts a validated full-operator
  lower bound; otherwise the strongest verdict is `provisional`, with the
  mechanism recorded in the new `verdict_reason` field. The bare-value call
  path (`coverage` omitted) keeps working and is capped at `provisional`
  for the same reason: unknown coverage must never read as `adequate`.
  `epsilon_zero_full_operator_evidence` reports which case applied. Several
  existing tests asserted `adequate` from a bare `epsilon_zero` used only
  to exercise a different gate (corroboration, the outlier count, or a
  corrupted-reading abstention); those now pass
  `full_operator_lower_bound=True` to keep testing that gate in isolation.
  `tests/test_non_normality.py::test_arnoldi_breakdown_does_not_promote_a_reduced_epsilon_zero_to_adequate`
  reproduces the exposure above and checks both the bare-value and
  coverage-aware call paths land on `provisional`/`investigate`, never
  `adequate`;
  `test_full_dimensional_arnoldi_projection_can_still_reach_adequate`
  checks a genuinely full-rank projection can still reach `adequate`.
  `tests/test_pseudospectra.py` gained coverage assertions
  (`k_requested`/`k_achieved`/`breakdown`/`residual_norm`) on the existing
  Arnoldi tests plus a legacy-indexing regression test.

### Fixed

- **The periodic Laplacian symbol built as `(2*cos(k*dx) - 2)/dx^2` catastrophically
  cancelled in float32.** `laplacian_symbol_1d` and both 2D builders
  (`laplacian_symbol_2d`, `laplacian_symbol_2d_rfft`) in `fft_solvers.py` subtracted
  two O(1) values to recover an O(dx^2) result; once `dx` was small enough that
  `cos(k*dx)`'s own float32 rounding error (about 1e-7 relative) was comparable to
  `2 - 2*cos(k*dx)` itself, the symbol lost accuracy or vanished outright. On a unit
  domain the first nonzero eigenvalue came out as -39.5, -40.0, -32.0, and 0.0 (the
  mode lost entirely) at N = 1024, 4096, 16384, 32768 against the exact -39.478, and
  a Helmholtz solve with D = 1, dt = 0.1 at N = 32768 retained the fundamental at
  amplitude 1.000 instead of the correct 0.2021. Fixed by using the algebraically
  identical `-4*sin(k*dx/2)^2/dx^2` (via `cos(theta) = 1 - 2*sin^2(theta/2)`), which
  never forms the cancelling subtraction; float64 results are unchanged to rounding
  (about 1e-9 relative at the worst mode). `tests/test_fft_solver.py::TestLaplacianSymbolFloat32Precision`
  adds the float32 eigenvalue and Helmholtz reproductions (subprocess, x64 off), plus
  confirmations that the symbol still matches the second-difference stencil on a
  Fourier mode and that float64 changes only at rounding level.

- **The Jacobi preconditioner (`preconditioners.py`'s `DiffusionPreconditioner._jacobi_solve`)
  double-counted the diagonal.** For `A x = rhs`, `A = I - dt*D*Laplacian`, the update
  was `x_new = (1-omega)*x + omega*(rhs + dt*D*Laplacian(x))/diag`, which divides by
  `diag` (already the diagonal of `A`) without ever subtracting `x`'s own contribution
  to `Laplacian(x)` from the numerator. For `nx = 8`, `dx = 1/8`, `dt = D = 1`, and an
  all-ones padded right-hand side (exact solution `x = 1`, since `Laplacian(1) = 0`),
  this drove `x = 1` to 0.01183 after 5 iterations and toward 1/129 in the limit; more
  iterations made it worse. Fixed by using the textbook damped Jacobi splitting
  `A = M - N`, `M = diag(A)`: `x_new = x + omega*(rhs - x + dt*D*Laplacian(x))/diag`.
  `tests/test_preconditioners.py::TestDiffusionPreconditionerJacobi` adds the constant-field
  reproduction above (now an exact fixed point to 1e-14) and a random-right-hand-side
  check that the residual decreases monotonically over 20 iterations and the fixed
  point matches a dense solve to 1e-8.

- **`variable_coeff.py`'s Richardson iteration used a fixed `omega = 1` and a validity
  flag (`is_valid_approx = std(D)/mean(D) < threshold`) that bounds nothing about
  convergence.** `nx = 128`, `D = 1` except `D[64] = 4` (a single large spike) passed as
  "valid" (variation_ratio 0.258) while the residual grew from about 21 to 5.2e5 over 30
  iterations at `dt = 1`; `residuals[-1]` also described the previous iterate rather than
  the one actually returned. Fixed by deriving a spectral bound for the damped Richardson
  iteration preconditioned by the circulant (mean-`D`) Helmholtz solve: writing
  `A = I - dt*L_D` and `M = I - dt*D_ref*Laplacian` (both symmetric positive definite,
  since the conservative discretization is self-adjoint), the iteration contracts for
  `0 < omega < 2/lambda_max(M^-1*A)`, and `lambda_max(M^-1*A) < D_max/D_ref` for any `D`
  bounded away from 0 (derivation in `richardson_omega_bound`'s docstring). `omega` now
  defaults to `0.9 * 2*D_ref/D_max` (a safety margin below that bound) while still
  accepting an explicit override; the returned `(solution, residual_history, converged)`
  reports the *returned* iterate's true residual at `residual_history[-1]` (previously one
  step stale) and a convergence flag; `is_valid_approx` now reacts to `D`'s maximum
  deviation from the mean, which correctly flags the spike case invalid. `richardson_iteration_varcoeff_1d/2d`'s
  return signature grew a third element, so the two existing call sites in
  `tests/test_variable_coeff.py` were updated to unpack it.
  `tests/test_variable_coeff.py::TestRichardsonSpikeCoefficient` adds the spike
  reproduction (now converges below 1e-8 relative within 30 iterations), a check that
  `residuals[-1]` matches the returned iterate's true residual, and a check that an
  explicit `omega` override is respected.

- **The README's spatial operators table overstated the FD stencils' accuracy with
  non-periodic boundaries, and `tests/test_variable_coeff.py` failed when run in
  isolation.** The table listed "O(Δx²) to O(Δx⁶)" under "Any" boundary condition
  support, but the 4th-order stencil with the current 2nd-order Dirichlet ghost closure
  converges at 2nd order overall (measured 4.13e-4, 1.03e-4, 2.58e-5, 6.45e-6 on
  `u = x(1 - x)` under grid refinement); higher-order stencils reach their nominal order
  only with periodic boundaries. Added the same one-sentence caveat to
  `operators_ho.py`'s module docstring. Separately, `tests/test_variable_coeff.py` never
  called `jax.config.update("jax_enable_x64", True)` itself, relying on another test
  module to flip the global flag first; five of its tests failed on float32 rounding when
  the file was run in isolation. Added the same `jax_enable_x64` enable the other test
  modules use.

- **`nilt_fft.integrate_discrete` gave the initial sample its full weight,
  leaving a `dt g[0]/2` offset in every cumulative value after the first.**
  The cumulative trapezoid is
  `f[k] = dt (cumsum(g)[k] - g[0]/2 - g[k]/2)`; the implementation
  subtracted only `g[k]/2`, and resetting `f[0]` to zero hid the offset at
  exactly one point. `integrate_discrete(jnp.ones(5), 0.1)` returned
  `[0, 0.15, 0.25, 0.35, 0.45]` instead of `[0, 0.1, 0.2, 0.3, 0.4]`, and
  the "simpson" branch repeated the same formula (it never applied a
  Simpson correction; the two branches are now one). The offset went into
  `nilt_fft_with_pole_at_origin`. The rule is now exact on constant and
  linear inputs.
  `tests/test_nilt_pairs.py::TestIntegrateDiscrete` covers both rules on
  constants, lines, a nonzero first sample and an unknown rule name.
  `TestPoleAtOrigin::test_pole_at_origin_requires_positive_shift` had its
  tolerance raised from 5e-3 to 1.2e-2: the bug was cancelling a real
  error. For `F(s) = 1/(s(s+1))` at `dt = 0.05`, `N = 1024`, `a = 0.5` the
  inverted `g` has `g(0) = 0.4924` (the half-jump) and `g(0.05) = 1.0439`
  against `exp(-t)`, which leaves the cumulative result 1.04e-2 low; the
  `+0.0123` offset cancelled most of that and the old tolerance was
  calibrated on the cancellation. The same quadrature on the exact
  `exp(-t)` is accurate to 2.1e-4.

- **`chebyshev_nilt.talbot_method` returned all NaNs for every odd
  `n_points`, and a NaN error estimate for every even one whose half is
  odd.** The midpoint grid `theta_k = -pi + (2k + 1) pi / N` contains
  `theta = 0` exactly when `N` is odd, and `talbot_contour` evaluated
  `theta cot(alpha theta)` and its derivative `cot(u) - u/sin^2(u)` there
  directly, dividing by `sin(0)`. `talbot_method(lambda s: 1/(s + 1),
  jnp.array([0.5, 1, 2]), n_points=31)` returned `[nan, nan, nan]`;
  `n_points = 34` returned the right values but a NaN `error_estimate`,
  since the estimate halves the point count and 17 is odd. Neither count
  was prohibited by the API. All four of `n_points` 31, 32, 33, 34 now
  return `exp(-t)` to better than 1e-9 with a finite estimate. Fixed by
  taking the removable singularity analytically on a small-`|alpha theta|`
  branch of a `jnp.where`, using the series `theta cot(alpha theta) =
  (1 - u^2/3 - u^4/45)/alpha` and `cot(u) - u/sin^2(u) = -2u/3 - 4u^3/45`
  rather than the bare limits, so a `theta` that lands near but not exactly
  on zero is as accurate as the rest of the contour.
  `tests/test_chebyshev_nilt.py::TestTalbotRemovableSingularity` covers the
  four point counts, the contour value at `theta = 0`
  (`s = N (beta/alpha - delta)`, real), and the smooth join between the two
  branches.

- **`chebyshev_nilt.weeks_method` reported its error estimate before the
  `e^{sigma t}` prefactor, so a result that was wrong by 9.0e4 came back
  with an estimate of 9.7e-15.** The estimate was `max |coeffs[-3:]|`, the
  truncated tail of the Laguerre series, and the series is multiplied by
  `e^{sigma t}`: at `sigma = 1` and `t = 50` that factor is 5.2e21.
  `weeks_method(lambda s: 1/(s + 1), 32, jnp.array([50.0]))` returns
  90057.28 against the exact 1.93e-22; the estimate is now 5.16e7, above
  both the true error and the returned value. Fixed by multiplying the tail
  by `max_t e^{sigma t}` and adding a roundoff term
  `eps * sum|a_n| * e^{sigma t}`, which is admissible because the Laguerre
  functions satisfy `|e^{-x/2} L_n(x)| <= 1` for `x >= 0`. The estimate is
  absolute, not relative, and the docstring now says so and states that a
  large estimate means the caller must retune `sigma` and `b` (per
  Weideman 1999), since only the caller knows where the singularities of
  `F` are. On well-chosen parameters the estimate stays tight: 1.7e-15
  against a true error of 1.1e-16 for `1/(s + 1)` with `sigma = 0.5`,
  `b = 1` at `t <= 4`.
  `tests/test_chebyshev_nilt.py::TestWeeksErrorEstimate` covers both, the
  scaling with the prefactor, and the roundoff floor.

- **`quality_metrics.compute_eps_im` computed the wraparound tail ratio
  from the imaginary part of the inverted signal, which is zero by
  construction for a Hermitian spectrum, so `assess_nilt_quality` graded a
  badly periodized inversion "excellent".** The spectrum is mirrored into
  exact Hermitian symmetry before the IFFT, so `late_leakage / norm_real`
  measures rounding noise and nothing else. For `F(s) = 1/(s + 0.01)^2`
  with `N = 256`, `dt = 0.01`, `a = 0` and `t_end = 0.64` it read
  `tail_ratio = 2.77e-21` and returned `excellent` while `f(0.64)` came out
  3906.3 instead of 0.6359; `nilt_fft.py`'s own classifier
  (`compute_imaginary_leakage` feeding `classify_quality_tier`) correctly
  said `poor` on the same parameters. The sensor now reads
  `tail_ratio = 1.0000545` and the verdict is `poor`. Fixed by extracting
  the damped-tail sensor `nilt_fft.compute_wraparound_tail_ratio` (real
  plus imaginary energy beyond `t_end` relative to `[0, t_end]`, returning
  the new `WraparoundTail`), having `compute_imaginary_leakage` call it,
  and replacing `compute_eps_im`'s imaginary-part formula with the same
  call, so the standalone assessment and the uniform inversion now report
  the same number. `r_early` and `r_late` keep their old definitions: they
  localize the leakage and decide nothing. When `t` is supplied without a
  `t_end`, the half-period `t[N // 2]` is used, matching
  `nilt_fft_uniform`'s own default.
  `tests/test_adaptive_tuning_quality.py::TestStandaloneWraparoundSensor`
  adds the reproduction, the agreement between the two implementations, and
  a well-resolved case that must stay `excellent`.

- **`tune_nilt_adaptive` never called `check_spectral_cfl_conditions`, and
  its only limit on the Bromwich shift was the overflow budget, so it
  reported "good" on an inversion whose every digit was amplified rounding
  noise.** `band_edge_ratio`, `tail_energy_fraction` and `tail_ratio` are
  all ratios, and `exp(a t)` scales the signal and its error alike, so none
  of them moves when the shift turns the answer into noise; the feasibility
  limit in `tune_nilt_params` only keeps `exp(a t)` representable, which
  leaves the whole band between the overflow budget and the accuracy budget
  open. For `J = [[-1, 100], [0, -1]]` (numerical abscissa 49, off-diagonal
  resolvent `F(s) = 100/(s + 1)^2`), `t_end = 1`,
  `bounds = {rho: 1, re_max: 49, im_max: 0}` selected `a = 53.605`,
  `N = 256` and reported "good, all sensors within normal range" while
  returning -5.44e19 where `100 t exp(-t)` is 36.79; `A_exp = 1.9e23`
  leaves a float64 rounding floor of 4.2e7. The call now returns a `poor`
  verdict reading "infeasible: amplification A_exp=1.91e+23 leaves an error
  floor of eps*A_exp=4.24e+07, above the accuracy budget A_max=4.50e+09;
  ... Split t_end=1 into 3 windows of 0.333 ..." and raises a
  `UserWarning`. Fixed by running the same `check_spectral_cfl_conditions`
  the CFL tuner uses on every pilot inversion and gating on its
  conditioning and spectral placement conditions, with the amplification
  threshold derived from the working precision as
  `A_max = amplification_tolerance / eps_machine` (new keyword
  `amplification_tolerance`, default 1e-6, so `eps_machine exp(a t_end) <=
  1e-6`) instead of a fixed constant; and by clamping step 3 of
  `retune_based_on_diagnostics`, which halved `a` for "general
  degradation", at `required_abscissa(bounds.re_max, T)`, so the ladder
  falls through to doubling `N` rather than crossing a pole. The window
  length in the refusal solves the wraparound and amplification conditions
  together, so splitting at it brings `A_exp` inside the budget. The budget
  bounds amplified rounding noise only: the bandwidth truncation error is
  amplified by `exp(a t_end)` as well and the normalized sensors still
  cannot see that, which is now stated in the docstring.
  `tests/test_adaptive_tuning_quality.py::TestAmplificationBudget` adds the
  reproduction, the clamped ladder, the recommended window, and a shift
  well inside the budget that must keep passing.

- **`tune_nilt_adaptive_cfl` accepted a retuned Bromwich shift without
  rechecking where the contour had landed, so the conditioning guard could
  halve `a` across a pole and still report "good, all CFL conditions
  satisfied".** `SpectralCFLConditions` had no placement field, and the
  four conditions it did carry are all blind to the pole: tail energy,
  phase step, amplification and endpoint jump are no larger for a divergent
  inversion than for a convergent one. `F(s) = 1/(s - 10)`, `t_end = 1`,
  `bounds = {rho: 10, re_max: 10, im_max: 0}` tuned `a = 14.605`, hit the
  `A_exp = 2.2e6 > A_max = 1e6` conditioning violation, halved `a` to
  7.303, crossed the pole at 10, and returned -0.393 at `t = 0.977` where
  `exp(10 t)` is 17424; the verdict was `good`. It now returns 17454.8 at
  the same point (0.17% relative error) with the verdict `acceptable` and
  the reason "no further adjustments available". Fixed by adding the
  spectral placement condition `a >= sigma + max(delta_min,
  ln(1/eps_tail)/(2T))` to `SpectralCFLConditions` (fields `sigma`,
  `a_required`, `spectral_placement_ok`) and to
  `check_spectral_cfl_conditions`, which takes the abscissa as a new
  `sigma` argument; by clamping the conditioning remedy in
  `suggest_parameter_adjustments` at that floor (new
  `endpoint_diagnostics.required_abscissa`) instead of halving `a`
  unconditionally; and by re-running the placement check in
  `tune_nilt_adaptive_cfl` against the re-normalized triad after every
  adjustment, refusing the window with a `poor` verdict and a `UserWarning`
  when the adjusted shift falls through the floor. The tuner passes the
  abscissa `tune_nilt_params` recorded in `diagnostics['alpha']`, so no
  caller has to supply it. `tests/test_adaptive_tuning_quality.py::TestSpectralPlacementGuard`
  adds the reproduction plus unit coverage of the new condition and of the
  clamped remedy.

- **An embedded error estimate built from a failed Newton solve was
  treated as a real one.** `cn_with_err` and `bdf2_with_err` threw away
  the auxiliary backward Euler solve's `NKStats`
  (`y_be, _ = be_step(...)`) and reported only the primary solve's, so the
  accept test saw a difference between one converged state and one that
  had merely stopped iterating. When both stall at the same value that
  difference is exactly zero, which the controller reads as a perfect
  step: on `u' = -2u^2(u - 1/2)`, `u0 = 1`, `dt = 1` with
  `max_newton_iters = 1`, Crank-Nicolson converges to 0.5, backward Euler
  fails at the same 0.5, `err` is 0.0 and `adaptive_integrate` finished
  with `SUCCESS` at `y = 0.5` against the reference 0.6510085678. The
  estimate is now valid only if every solve contributing to it converged
  to a finite state; otherwise the step is rejected and its `dt` cut,
  exactly as for a failed primary solve. The same reproduction now rejects
  the `dt = 1` attempt and subdivides to 0.650999 (CN) and 0.650972 (BDF2)
  after 48 rejections. The plain backward Euler branch is covered too: its
  estimate is the same `y_cn - y_be` difference, so an unconverged
  Crank-Nicolson auxiliary rejects the step even when backward Euler
  itself converged. Commit 58c0d1f closed this hole on the BDF2 startup
  branch only; `test_bdf2_startup_rejects_unconverged_cn` still passes.
  `tests/test_integrators.py::TestErrorEstimateRequiresEverySolve` covers
  all three branches.

- **`check_compatibility_neumann` and `project_to_compatible` used a plain
  sum/mean, which is the wrong measure for the default node-centred (DCT-I)
  Neumann layout.** The node-centred Laplacian's end rows are `[-2, 2]/dx**2`
  rather than the interior `[1, -2, 1]/dx**2`, so its left null vector is
  trapezoidal, `(1, 2, ..., 2, 1)` (half weight at each endpoint), not
  uniform. With `N = 4`, `dx = 1`: `rhs = [1, -1, 0, 0]` has plain sum zero
  and was accepted, but its trapezoidal-weighted sum is `-0.5`, and solving
  it anyway leaves a residual of exactly `1/6` at every point; the solvable
  `rhs = [2, -1, 0, 0]` has plain sum `1` and was rejected, but its
  trapezoidal-weighted sum is exactly `0`. The cell-centred (DCT-II) layout
  was unaffected: its null vector is already uniform. Fixed by adding a
  `centering` argument (`'node'` default, matching `solve_poisson_neumann`'s
  default; `'cell'` for DCT-II) to both functions, using trapezoidal weights
  for node centering and the previous plain sum/mean for cell centering.
  `tests/test_fft_nonperiodic.py::TestPoissonSolver::test_node_centered_compatibility_uses_trapezoidal_weights`
  reproduces both flips (rejecting `[1, -1, 0, 0]`, accepting
  `[2, -1, 0, 0]`) and the `1/6` residual;
  `::test_projected_random_rhs_solves_at_rounding_level` checks a random
  vector's projection solves to rounding error for both centerings;
  `::test_cell_centered_compatibility_is_unaffected_by_the_fix` pins that
  `centering='cell'` still matches the historical (centering-less) behavior.

- **`nilt_solve_linear_pde` let a real eigenvalue into its transient mask
  whenever the mode's residual was nonzero, inflating the Bromwich shift
  and ruining the inversion of the genuinely complex modes.** For a real
  `lambda_k`, `c_k = -Re(lambda_k) = -lambda_k`, so `H_k(s)`'s two poles
  coincide and its inverse is identically zero regardless of the mode's
  weight `w_k = r_k/lambda_k`; the mask (`transient_mask = w != 0`) did not
  check for this and let such a mode in whenever `w_k` was nonzero. That
  mode's own contribution was still zero, but it entered
  `sigma_H = max Re(lambda_k)` over the mask and the tuner's
  `re_max_override` anyway. `eigenvalues = [20, -1+5j, 0, -1-5j]`,
  `u0 = [1, 0, -1, 0]`, `source = full(4, 1e-12)`, `t_end = 1`: the k = 0
  mode (`lambda = 20`, `u0_hat = 0`, `f_hat = 4e-12`) had a tiny but
  nonzero residual, pushed `sigma_H` to 20 and the tuned shift `a` to
  24.605 instead of 3.605, and the max error against the exact field (below
  0.36) reached 349033. This exposure predates a518612's residual
  weighting (a real mode with nonzero `u0_hat` could already enter the mask
  under the old `w_k = u0_k`); a518612 widened it to forcing-only real
  modes by also making the weight nonzero from the source alone.
  `eigenvalues = full(4, 200)`, `u0 = 0`, `source = 1`, `t_end = 1` raised
  `NILT-CFL infeasible` even though nothing needed inverting, since the
  spectrum's lone real mode still entered the mask; it now returns the
  closed form `t phi1(200 t) f` (`expm1(200)/200`, about `3.61e84`)
  directly with no NILT grid built. Fixed by requiring
  `Im(lambda_k) != 0` in `transient_mask` in addition to `w != 0`, and
  using the same mask for `sigma_H`, the tuner override, the
  `a <= sigma_H` check, and the transfer function's denominators.
  `tests/test_fft_nilt_bridge.py::TestRealEigenvaluesAreNeverInverted`
  adds the two reproductions above, the pre-existing nonzero-`u0`
  exposure, and a purely imaginary pair that must still be inverted.
  `TestSmallEigenvalueReconstruction::test_exactly_zero_matches_the_tiny_eigenvalue_limit`
  and
  `TestClosedFormKeepsTheSourceUnderALargeInitialCondition::test_real_decay_keeps_a_source_16_decades_under_u0`
  are updated: both used a real spectrum and asserted a grid was built once
  `|lambda_k| t_end` cleared tau, which a real eigenvalue no longer does
  regardless of tau.

- **`etd_integrate`'s ETD2 path anchored saved snapshots to the wrong
  absolute step.** The 266c898 rewrite (an outer `lax.scan` over
  save-sized blocks) sized every block at `save_every` steps, but ETD2
  takes its first step eagerly to seed the multistep history before that
  loop, so the first block started one step late: `t_span = (0, 1)`,
  `dt = 0.25`, `save_every = 2` returned history times `[0, 0.75, 1]`
  instead of `[0, 0.5, 1]`, and `save_every = 1` dropped the `t = 0.25`
  snapshot entirely. ETD1 and ETDRK4, which have no eager seed step, were
  unaffected. Fixed by sizing the first compiled block to
  `save_every - 1` steps (or, when `save_every == 1`, treating the seed
  step's own state as the first snapshot) so every later save lands on
  the same absolute-step grid ETD1 and ETDRK4 use; the total step count
  and the always-returned final state at `t_end` are unchanged.
  `tests/test_fft_operators.py::TestETDIntegrateStepSchedule` adds
  `test_etd2_history_times_match_etd1` (`save_every` in `{1, 2, 3}`,
  the last a non-divisor of the 4-step schedule, checked against ETD1's
  history times on the same inputs) and
  `test_etd2_trajectory_unchanged_by_save_every` (ETD2's final state on a
  reaction-diffusion problem agrees to `1e-13` across `save_every` in
  `{1, 2, 3}`).

- **`_odd_symbol_wavenumber` identified the Nyquist mode by comparing
  floating-point wavenumbers with a relative tolerance, which float32
  rounding defeats.** fftfreq's own float32 rounding error at the Nyquist
  bin is about 1.4e-6 relative, well past the helper's 1e-12 tolerance, so
  the mask never fired in float32: on `Grid1D.uniform(10, 0, 1)`, `v = 1`,
  `D = 0`, float32, x64 disabled, the Nyquist advection eigenvalue stayed
  `31.4159j` instead of `0`, `exp_matvec((-1)**i, 0.1)` returned `-u`
  instead of `u`, and the Helmholtz solve residual reached `0.908` instead
  of float32 rounding level. Fixed by identifying the Nyquist bin by
  integer index and axis parity instead of by value: for an even-length
  axis of size `n` it is index `n // 2`, whether `k` is a full fftfreq
  spectrum of length `n` or an rfftfreq half-spectrum of length
  `n // 2 + 1` (its last index); an odd-length axis has no bin exactly at
  `+-pi/dx`, so nothing is masked there. `_odd_symbol_wavenumber` now
  takes the axis length `n` and an `axis` argument instead of the grid
  spacing. `tests/test_preconditioners.py` covers the masking helper
  directly (a float32 array whose Nyquist entry is perturbed by a
  relative 2e-7, both full- and half-spectrum layouts, the odd-length
  no-op case, and a 2D broadcast array masked along a named axis) and, in
  a subprocess with x64 left disabled (the main suite runs with x64
  enabled, where the bug does not show), the exact 1D reproduction above
  plus a 2D case with the Nyquist mode along both axes.

- **The Neumann ETD1 coefficients and both Neumann Poisson solvers divided
  by a possibly-zero eigenvalue before `jnp.where` masked the result,
  poisoning reverse-mode gradients.** Every Neumann layout has a k = 0
  mode with eigenvalue exactly zero (the constant mode), and `jnp.where`
  differentiates both of its branches regardless of which one is selected
  in the forward pass. `_etd1_coefficients` computed `(exp(z)-1)/z`
  unmasked before selecting it via `jnp.where`, and
  `solve_poisson_neumann_node`/`solve_poisson_neumann_cell` divided by the
  raw `laplacian_symbol` before masking the k = 0 component to zero: on
  `N = 4`, `dx = 1`, `u = N_u = ones(4)`, `dt = 0.1`, `d/dD` of the summed
  ETD1 Neumann step was NaN for every `D` for both centerings (expected
  `0.0`, since the k = 0 mode never contributes to the sum), and `d/drhs`
  of the squared Poisson solution norm at `rhs = [2, -1, 0, 0]` was four
  NaNs. `_etd1_coefficients` now calls `jit_kernels.phi1` (safe on both
  branches after the phi-function fix above) instead of duplicating the
  formula, and the Poisson solves mask the denominator to a safe
  placeholder before dividing, then mask the k = 0 component of the
  quotient to zero, the idiom already used at `fft_solvers.py:338`.
  `tests/test_fft_nonperiodic.py::TestNeumannGradients` covers all four
  gradients (both centerings, ETD1 and Poisson) against a finite-difference
  reference.

- **`phi1`, `phi2`, and `phi3` fed their unused Taylor branch a raw, unmasked
  `z`, so its gradient could be NaN even where the direct branch was
  selected and finite.** `jnp.where` evaluates both branches and their
  cotangents; the direct branch already substitutes a safe placeholder for
  `z` where it is not selected, but the Taylor branch's degree-15 Horner
  polynomial (`_phi_taylor`) still received the real `z` regardless of
  which branch was chosen (commit `a7040ad` masked only the direct
  branch). At float32 `z = -1e4` the polynomial overflows to `-inf`, and
  `jnp.where`'s zero cotangent times `inf` is NaN: `jax.grad(phi1)` at
  that point was NaN (forward value `1e-4`, finite) instead of matching
  the analytic derivative `1/z^2 = 1e-8`. `phi2` and `phi3` had the same
  defect. Fixed by substituting a safe placeholder (`z` where the branch
  is selected, `0.0` otherwise) into the Taylor branch too, mirroring the
  direct branch's existing idiom. `tests/test_phi.py::test_phi_gradients_finite_at_large_negative_z_float32`
  covers `phi1`/`phi2`/`phi3` at float32 `z = -1e4` and `z = -1e2` against
  the closed-form derivative, jitted and not;
  `tests/test_jit_kernels.py::TestETDGradients::test_etd1_kernel_gradient_wrt_diffusion_stiff_mode_float32`
  covers the same defect through `etd1_kernel_1d`.

- **The NILT bridge reconstructed each mode as a cancellation of two
  `1/lambda_k` terms.** `nilt_solve_linear_pde` split every mode with
  `|lambda_k|` above a relative spectral-zero threshold
  (`max(1e-12 max|lambda|, 1e-300)`) into a transient weight
  `w_k = u0_k + f_k/lambda_k` and a particular constant `-f_k/lambda_k`, and
  added the two back as plain floats. For a small but not spectrally zero
  `lambda_k` both are `O(1/lambda_k)` and have to cancel to an `O(1)`
  answer, which floating point cannot do: eight eigenvalues `-1e-8` with
  `u0 = 0`, `source = 1`, `t_end = 1` returned `0` instead of about `1` in
  float32, and `lambda = -1e-16` returned `2` instead of `1` in float64.
  The reconstruction now uses the form the exact mode solution already has,
  `u_k(t) = e^{lambda_k t} u0_k + t phi1(lambda_k t) f_k` with
  `phi1(z) = (e^z - 1)/z` (`moljax.core.jit_kernels.phi1`, whose Taylor
  branch below `|z| = 0.5` in double precision carries the small-argument
  limit), which contains no `1/lambda_k`: the forced part is exact for a
  source constant in time and is evaluated in the time domain, and the NILT
  inverts the transient alone,
  `H_k(s) = w_k [1/(s - lambda_k) - 1/(s + c_k)]`, with a weight that is no
  longer `u0_k + f_k/lambda_k`. The same formula covers `lambda_k = 0`
  (`phi1(0) = 1`), so there is no spectral-zero branch in the
  reconstruction; the one threshold left decides only whether a mode is
  inverted at all, and compares `|lambda_k| t_end` against the working
  precision rather than a fraction of `max|lambda_k|`. (That commit set
  `w_k = u0_k` and put the threshold at the working precision's epsilon;
  the entry below corrects the weight to `r_k/lambda_k` and the threshold
  to `tau = 1e-2` on `|lambda_k| t_end`. The numbers here are unaffected:
  every spectrum involved is real, so `H_k` is identically zero under
  either weight.) On a
  real spectrum spanning `1e-12` to `1e3` in magnitude with nonzero `u0` and
  source, the max error against the per-mode closed form falls from 4.0e-10
  to 4.4e-16 at `t = 0.05`, and the jump across the old spectral-zero
  threshold (`lambda = 0` against `lambda = -1e-14`, `t_end = 2`) falls from
  7.8e-3 to 3.0e-14.
  `tests/test_fft_nilt_bridge.py::TestSmallEigenvalueReconstruction` covers
  all of these.

- **A stationary mode had its transient inverted numerically, with nothing
  left to cancel the inversion's error.** Once the forced response was
  evaluated in closed form (previous entry), `nilt_solve_linear_pde`
  inverted `H_k(s) = u0_k [1/(s - lambda_k) - 1/(s + c_k)]`, weighted by
  the initial condition. A stationary mode, one whose residual
  `r_k = lambda_k u0_k + f_k` is zero and whose solution is therefore
  `u_k(t) = u0_k` for every `t`, still had `e^{lambda_k t} u0_k` inverted,
  and the particular term `-f_k/lambda_k` whose own NILT error used to
  cancel it was gone. On `eigenvalues = [0, -1+100j, 0, -1-100j]`,
  `u0 = [1, 0, -1, 0]`, `source = [1, 100, -1, -100]`, `t_end = 1` (every
  `r_k` exactly zero), the bridge returned
  `[0.98499821, 0.00121119, -0.98499821, -0.00121119]` instead of `u0`, a
  max error of 1.5e-2, the raw NILT error on the lightly damped `100j`
  pair. The inverted weight is now residual-proportional,
  `w_k = r_k/lambda_k` for modes with `|lambda_k| t_end > tau` and `w_k = 0`
  below, `r_k` being formed as `lambda_k u0_k + f_k` so that a stationary
  mode's weight is zero to the last bit. Everything added back in closed
  form stays in `u0_k` and `f_k`:
  `u_k(t) - w_k (e^{lambda_k t} - e^{-c_k t}) = u0_k e^{-c_k t}
  + f_k t phi1(-c_k t) Re(lambda_k)/lambda_k` on the inverted modes and the
  whole `e^{lambda_k t} u0_k + t phi1(lambda_k t) f_k` below `tau`, which
  is also what `u_analytical` reports. Writing that remainder in the
  residual instead reads `u0_k` against a second term of size
  `|r_k/lambda_k| = |u0_k|` and loses `f_k` whenever
  `|lambda_k u0_k| >> |f_k|`: four eigenvalues `-1` with `u0 = 1e16`,
  `f = 1` and `t_end = 50` returned 0 everywhere, `u_analytical` included,
  against the exact `1e16 e^{-50} + (1 - e^{-50}) = 1.000001928749848`, and
  a `lambda = -1 + 5j` mode with `u0_k = 1e12`, `f_k = 1` at `t_end = 30`
  was 1.3e-4 out relative. In the `u0_k`, `f_k` form both are exact to
  rounding: the `u0_k` term is a plain decay, `Re(lambda_k)/lambda_k` is
  bounded by 1 in modulus, and the division by `lambda_k` happens only
  above `tau`.
  `tau = TRANSIENT_TAU = 1e-2` on `|lambda_k| t_end`, a module constant,
  up from `sqrt(eps)`. Both branches are exact in exact arithmetic, so
  `tau` decides only which modes are charged the inversion's own error. It
  is tempting to read `w_k = r_k/lambda_k` as amplifying the NILT's
  truncation error by `1/(|lambda_k| t_end)`, but the two poles of `H_k`
  coalesce in the same limit --
  `H_k(s) = r_k [i Im(lambda_k)/lambda_k]/((s - lambda_k)(s + c_k))` -- so
  the transform handed over is `O(|r_k|)` and its error is flat: with
  `lambda = +-ib`, `u0 = 0`, `f_hat = 2`, `t_end = 1`, the inverted branch
  is 1.861e-5 from the exact answer at every `b` from 1e-7 to 3, where the
  closed form is exact. A flat error cannot be made small by moving `tau`,
  so `tau` goes where the jump it creates matches the answer's own
  variation across the band it separates, `(|lambda_k| t_end)^2/6 =
  1.667e-5` at 1e-2 (the first-order term is a phase and cancels against
  the conjugate partner). `sqrt(eps)` put the same 1.9e-5 jump where the
  exact answer varies by 4e-17: `b = 1.49e-8` returned 1.0 and
  `b = 1.491e-8` returned 0.9999813. What the weight does amplify is the
  cancellation in forming `1/(s - lambda_k) - 1/(s + c_k)`, relative size
  `eps |s|/|Im(lambda_k)|` at the contour's top frequency, 3e-6 at
  `sqrt(eps)` and 5e-12 at 1e-2; the old threshold bounded that correctly,
  it was simply not the binding term.
  The stationary reproduction above now returns `u0` to rounding (2.8e-17
  against 1.5e-2), a near-stationary case at a residual `1e-6` of the
  source falls from 1.5e-2 to 1.5e-8 and at `1e-8` to 1.5e-10 (the error
  scales with `r_k`, as the weight does), a stationary mode sitting inside
  an otherwise live spectrum comes back at 1.2e-16 against 3.0e-2, and the
  same spectrum with `source = 0`, which is not stationary, is unchanged at
  1.500179e-2.
  `tests/test_fft_nilt_bridge.py::TestStationaryModeIsNotInverted` covers
  the stationary field, the empty-transient report when every mode is
  stationary, the residual scaling, and the mode-by-mode check beside live
  modes;
  `TestClosedFormKeepsTheSourceUnderALargeInitialCondition` covers the two
  large-`u0` cases; and `TestSmallEigenvalueReconstruction` gains
  `test_imaginary_pair_straddling_tau_agrees_on_both_sides`, which measures
  the jump at `tau` against that variation. The threshold moved, so the
  `lambda = -1e-14`, `t_end = 2` leg of
  `test_exactly_zero_matches_the_tiny_eigenvalue_limit` is now
  `-4.999999e-3` and `-5.000001e-3`, one on each side of `tau`, both
  checked against their own exact solution.

- **`moljax.conditioning.pseudospectra` and `moljax.conditioning.non_normality`
  never checked for 64-bit precision.** `numerical_range`
  (`field_of_values.py`) and `linearized_operator` (`linearization.py`) both
  call `moljax._precision.require_x64` before doing any work, but none of
  `pseudospectra.py`'s public functions (`arnoldi`, `epsilon_zero`,
  `pseudospectrum_dense`, `reduced_pseudospectrum`, `ritz_values`) or
  `non_normality.py`'s (`assess_preconditioner`, `estimate_rates`,
  `clustering_rate`, `crouzeix_palencia_envelope`, `enclosing_disk_rate`,
  `real_bulk_outliers`, `right_real_outliers`, `traced_boundary_rate`) did.
  With x64 disabled, `epsilon_zero(np.full((2, 2), 1e8))` returned
  `11.313709` for an exactly singular matrix instead of raising, and a
  `complex128` request was silently downgraded to `complex64` throughout
  both modules. Every public entry point in both modules now calls
  `require_x64("conditioning diagnostics")` first, matching
  `numerical_range` and `linearized_operator`; internal callers between
  public functions (`estimate_rates` calling `enclosing_disk_rate`,
  `assess_preconditioner` calling `estimate_rates`, and so on) each pay one
  extra, cheap guard call rather than skip it, and no guard sits inside a
  loop. `tests/test_conditioning_precision_guard.py` runs a fresh
  subprocess with x64 left disabled and checks that `epsilon_zero`,
  `pseudospectrum_dense`, `arnoldi`, and `assess_preconditioner` each raise
  `require_x64`'s `RuntimeError`.

- **`etd_integrate` floored its step count and could allocate history
  proportional to every step taken instead of every step saved.** The step
  count was `int((t_end - t_start) / dt)`, which truncates rather than
  rounds: `t_span = (0, 0.3)`, `dt = 0.1` took 2 steps and stopped at
  `t = 0.2` instead of 3 steps to `t = 0.3` (`0.3 / 0.1` is
  `2.9999999999999996` in floating point). Because the returned state was
  also only ever a step already in the fixed history list, `t_span = (0, 1)`,
  `dt = 0.25`, `save_every = 10` returned only `u0`: none of the 4 computed
  steps landed on a `(step + 1) % save_every == 0` boundary, so the final
  state at `t = 1` was silently discarded. Separately, whenever any
  intermediate history was requested the compiled loop was a single
  `lax.scan` over every step, stacking the full state at every step before
  `save_every` thinned it: 1000 steps of two 8x8 float64 fields with
  `save_every = 500` allocated 1,024,000 bytes internally for 3,072 bytes
  returned, and a 1e5-step, two-256x256-field run would have needed about
  98 GiB. `etd_integrate` now takes `round((t_end - t_start) / dt)` steps
  and raises `ValueError` when `dt` does not divide the interval exactly,
  always returns the final state as the last history entry regardless of
  `save_every`, and runs an outer `lax.scan` over saved snapshots with an
  inner `lax.fori_loop` of `save_every` steps (the same shape
  `moljax.core.stepping.integrate_fixed_dt` already used), so the stacked
  history scales with the number of snapshots kept, not the number of
  steps taken. ETD2's carried nonlinear term threads through both loop
  levels unchanged. `tests/test_fft_operators.py::TestETDIntegrateStepSchedule`
  covers the schedule, the forced endpoint, the divisibility check, the
  scan allocation, and agreement across `etd1`/`etd2`/`etdrk4` and
  `save_every` values.

- **`imex_ssprk2_step`'s stage Laplacian amplified float32 roundoff.** The
  stage Laplacian was recovered algebraically as `dt L U = (U - rhs) /
  gamma` on the interior, which subtracts nearly equal states and divides
  by a small number: exact in infinite precision, since `U` solves
  `(I - gamma dt L) U = rhs`, but this amplifies FFT roundoff badly in
  float32, and gets worse rather than better as `dt` shrinks. On 32
  periodic cells on `[0, 2 pi]`, `u0 = cos(x)`, `D = 1`, float32,
  integrated to `t = 1`: the max error against the exact discrete solution
  was 1.94e-7 at `dt = 1e-4` before the regression (commit `5823ad5`) but
  7.57e-4 after it, and `dt = 1e-5` made it worse still (1.92e-3) instead
  of better. The stage Laplacian is now read off the Helmholtz solve's own
  spectral coefficients (`apply_diffusion_inverse_fft_with_laplacian` in
  `fft_solvers.py`, one inverse FFT reusing the `u_hat` the solve already
  computed) instead of recovered from real-space states, matching the
  pre-regression accuracy without paying for a second full FFT round trip.
  `tests/test_imex.py::TestIMEXSSPRK2Float32StageLaplacian::test_float32_stage_laplacian_accuracy`
  covers this at `dt = 1e-4` and `dt = 1e-5`.

- **The BDF2 startup step accepted a failed Crank-Nicolson solve using
  backward Euler's convergence status.** `be_only`, the branch adaptive
  BDF2 shares with backward Euler startup, returns the Crank-Nicolson
  state `y_cn` on the startup branch (it is second order, `y_be` is not),
  but still returned backward Euler's `NKStats` regardless of branch. On
  `u' = 2u`, `u0 = 1e-7`, `dt = 1` with default tolerances, Crank-Nicolson
  does not converge (residual about 3.46e-7 against the 1e-8 tolerance)
  while backward Euler does, and the accept/reject check reads
  `nk_stats.converged` from whichever stats came back: with backward
  Euler's `converged = True` standing in, the adaptive integrator accepted
  the unconverged Crank-Nicolson state and seeded its history with it.
  `be_only` now returns Crank-Nicolson's own stats alongside `y_cn` on the
  startup branch, so a non-converged startup solve is rejected the same
  way a failed backward Euler solve is; the plain backward Euler path is
  unchanged. `tests/test_integrators.py::TestAdaptive::test_bdf2_startup_rejects_unconverged_cn`
  covers this.

- **`apply_variable_diffusion_1d`/`2d` and the Richardson-iteration residual
  closures padded ghost cells with `'edge'` replication instead of periodic
  `'wrap'`, inconsistent with the circulant FFT preconditioner
  (`create_circulant_approx_1d`/`2d`) that treats the domain as periodic.**
  The boundary stencil's left neighbor used `D[0]` (replicated) instead of
  the correct periodic neighbor `D[n-1]`, an inconsistency, not a
  discretization error: on a manufactured periodic solution
  (`u = sin(x)`, `D = 1 + 0.3 sin(x)` on `[0, 2*pi)`), the domain's first
  grid point's error grew with resolution under `'edge'` padding (0.040 at
  n=32 to 0.088 at n=256, a negative convergence order) instead of shrinking
  at the conservative stencil's own second order, which `'wrap'` padding now
  gives (orders 1.995, 1.999, 1.9997 across 32→64→128→256). No caller
  documents a non-periodic (Neumann) use of these functions, so the padding
  mode is changed outright rather than gated behind a new parameter.
  `tests/test_variable_coeff.py::TestRichardsonIteration::test_constant_coeff_converges_quickly`
  (which noted this exact mismatch in a comment) is renamed
  `test_variable_diffusion_matches_periodic_solution` and now asserts the
  convergence order directly.

- **`bdf2_step`'s predictor, `2 y_n - y_{n-1}`, ignored the step ratio and
  had no finite guard.** This is only correct extrapolation at a constant
  step; at a ratio `w = dt/dt_prev` away from 1 it silently assumes `w = 1`.
  On `y' = -y` at `w = 0.1` (`y_prev = 1` at `t = 0`, `y = exp(-1)` at
  `t = 1`) it predicted `-0.264`, the wrong sign, against the exact
  `y(1.1) = 0.3329`. `bdf2_step` now uses the ratio-aware
  `(1+w) y_n - w y_{n-1}` (`0.3047` on the same example, 8.5 percent
  relative error), guarded against non-finite or wildly amplified values
  the same way as `_newton_start`. A linear problem hides this (Newton
  reaches the same converged solution in one step from either predictor
  start); a nonlinear residual's first evaluation does not.

- **`_newton_start`'s growth guard rejected every nonzero predictor when `y`
  was identically zero.** The guard compares `max-abs(y_pred)` against
  `10 * max-abs(y)`; at `y = 0` (a cold start with only an external source
  term, `F(y, t) != 0`) that bound is exactly 0, so any nonzero, perfectly
  finite predictor failed the ratio test and the step fell back to `y = 0`
  again, discarding the only informative predictor available. The shared
  guard (`_predictor_is_valid`, now also used by the BDF2 predictor above)
  skips the ratio test when `max-abs(y)` is exactly 0 and accepts any
  finite predictor there instead; the guard is unchanged away from `y = 0`.
  A fixed floor was considered and rejected because it would make the guard
  depend on the state's units.

- **`phi1`, `phi2` and `phi3` returned NaN under `jax.grad` at `z = 0`,
  including through `etd1_step` and `etdrk4_step` in `dt` or `D`.** Every
  periodic grid's `k = 0` Fourier mode gives `z = 0` on every step, so the
  bug was not exotic. `jnp.where` evaluates and differentiates both
  branches, and the direct formula's `0/0` at `z = 0` produced a NaN
  cotangent even though the series branch was the one selected there. Each
  direct branch now evaluates at a safe placeholder `z` (1.0) instead of the
  true `z` when the series branch is selected, the idiom already used for
  the Helmholtz denominators (`fft_solvers.py:338`). `jax.grad(phi1)(0.0)`
  is now `0.5` (was NaN), `phi2` and `phi3` similarly finite and matching
  the series derivative; forward values are bit-identical.

- **A Newton step that stagnates behind a rejected line search now exits
  instead of repeating itself to `max_newton_iters`.** The fallback added to
  keep the best line-search candidate (see the v1.2.0 entry below) left
  `newton_cond` checking only `iter_count` and `converged`, so a step whose
  line search accepted no candidate and made no progress (`best_r_norm >=
  r_norm`) returned the same iterate unchanged, and the next iteration
  recomputed the identical residual, JVP and rejected GMRES step. `atan(x)`
  from `x0 = [10, 10, 10]` with `max_backtrack=3` ran the full
  `max_newton_iters=20` doing nothing; a new `stagnated` field on
  `NewtonState` now exits the loop after 1 iteration, `converged` still
  honestly `False`.

- **`_newton_start` could seed Newton with a predictor amplified about 199x
  by an unstable explicit-Euler stage.** Only a finiteness check guarded the
  explicit-Euler predictor; at dt well past the explicit CFL limit, the
  discrete Nyquist mode is amplified rather than damped, and a finite but
  wildly oscillatory predictor wasted Newton iterations undoing the
  overshoot instead of benefiting from a good start. `_newton_start` now
  also falls back to `y` when the predictor's max-abs exceeds 10 times the
  max-abs of `y`.

- **`adaptive_integrate` and `adaptive_integrate_imex` saved history entries
  one step early and dropped the final accepted state.** `should_save`
  compared the pre-increment accepted-step count against `save_every`, so a
  run saved after accepted steps 1, 6, 11, ... instead of 5, 10, 15, ..., and
  `t_end` was never written to the history unless it happened to land on
  such a boundary. `should_save` now uses the post-increment count, and the
  final accepted step is always saved regardless of alignment.

- **The BDF2 startup step in `adaptive_integrate` was first order, not
  second.** `be_only` always returned `y_be`, including on the branch also
  taken for the BDF2 startup step, where `y_cn` (second order, already
  computed for the error estimate) was available but discarded. The PID
  controller assumes order 2 for BDF2 everywhere, so it scaled a first-order
  local error as if it were second order. `be_only` now returns `y_cn` on
  the startup branch and keeps `y_be` for the BE method itself.

### Added

- **`nilt_solve_linear_pde` and `compare_nilt_vs_timestepping` require
  64-bit precision.** They now call `moljax._precision.require_x64`, as
  `gaver_stehfest_method` and the rest of the NILT stack do, so a float32
  call raises the same clear `RuntimeError` naming the entry point instead
  of running the inversion at a precision the Bromwich contour's `e^{a t}`
  factor (about 100 at the tuned shift) immediately spends.
  `tests/test_fft_nilt_bridge.py::TestSmallEigenvalueReconstruction::test_bridge_requires_x64`
  covers this.

### Removed

- **The unused inner `step` closure in `make_etd1_integrator`.** `integrate`
  builds its own `lax.scan` body inline and never called it; the closure's
  own carry unpacking (`u, t = carry`) did not match how it then indexed
  `carry[2]`, so any future caller would have hit an `IndexError`
  immediately. Dead code with no callers (checked at the bytecode level in
  `test_jit_kernels.py::test_etd1_integrator_has_no_dead_step`).

### Changed

- **Time is no longer carried in the state's dtype.** `integrate_fixed_dt`,
  `integrate_imex_fixed_dt`, `adaptive_integrate` and
  `adaptive_integrate_imex` all built their clock from `model.dtype`
  (`jnp.array(t0, dtype=model.dtype)`) and accumulated it one step at a
  time. At `t0 = 1e6` in float32 the spacing is 0.0625, so `t + 0.01`
  rounds straight back to `t`: the fixed-step run took all 100 steps at
  the same instant and RK4 returned `u = 0` instead of 0.5 on
  `u' = t - t0`, and the adaptive run stopped with `MAX_STEPS_REACHED` and
  `t_final` still exactly `1e6`. A float32 state is a choice about the
  field, not a statement that the clock fits in 24 bits of mantissa, so
  time now follows JAX's default float type (float64 when x64 is enabled)
  regardless of the state's dtype, and the fixed-step path takes each
  timestamp as `t0 + i*dt` rather than as a running sum (one rounding in
  total instead of one per step: 1000 steps of 0.001 now land on exactly
  1.0 instead of 1.0000000000000007). The state itself stays in its own
  dtype: a step's result is cast back, so a right-hand side that uses `t`
  cannot silently widen the field or break a loop carry's dtype. Both
  reproductions now give 0.5. **Public surface:** `AdaptiveResult.t_final`
  and `t_history`, and `integrate_fixed_dt`'s `t_history`, are float64 for
  a float32 model under x64 (`y_final` and `dt_history` are unchanged);
  and when x64 is off, so there is no wider type to fall back on, a `dt`
  that is unrepresentable at `t0` now raises `ValueError` at validation
  instead of running and advancing nothing.
  `tests/test_integrators.py::TestTimeIsNotCarriedInTheStateDtype` covers
  the two reproductions, the drift, and the refusal.
  `TestAdaptive::test_adaptive_float32_model_under_x64` now pins
  `y_final` and `dt_history` as float32 and `t_final` as float64.

- **The IMEX steppers' explicit part is now everything the FFT diffusion
  split does not handle, not just `model.nonlinear_rhs`.**
  `imex_euler_step`, `imex_strang_step` and `imex_ssprk2_step` evaluated
  `model.nonlinear_rhs` for their explicit stages, which assumes a model's
  linear operators are exactly the diffusion the FFT solve inverts.
  `create_advection_diffusion_model` folds advection into the same
  `LinearOp` as the diffusion, so with `D = 0` the FFT solve was the
  identity, the explicit part was zero (the model has no nonlinear
  operators), and every IMEX step returned the state untouched: max-abs
  change 8.60e-16 on a 16x16 sine whose advective right-hand side has
  max-abs 0.9936, with `adaptive_integrate_imex` reporting `SUCCESS` on a
  state that never moved. The explicit part is now `model.rhs` minus the
  diffusion the split treats implicitly, `D * Laplacian(y)`, taken with
  the same `D` and through the same spectral operator the Helmholtz solve
  inverts (`diffusion_rhs_fft`), so the two cancel to roundoff: the FFT
  symbol is `(2 cos(k dx) - 2)/dx^2 + (2 cos(k dy) - 2)/dy^2`, the symbol
  of the same second-difference stencil the models' Laplacian operators
  use, checked numerically to a relative 1e-12. The same 16x16 advection
  case now moves by 9.93e-3 in one step of `dt = 0.01` and tracks an
  explicit RK4 reference to a relative 2e-7 (Strang and SSPRK2) over ten
  steps. A model with no linear operators states a right-hand side that
  contains no diffusion, so nothing is subtracted there and its steps are
  bit-identical. **Public surface:** the three steppers now raise
  `ValueError` when a diffusive field's boundary condition is not
  periodic, which the split's Laplacian cannot represent, or when
  `diffusivities` names a field the model does not have, instead of
  quietly stepping the wrong operator. Cost: one Laplacian evaluation per
  stage on top of the model's own right-hand side. Reaction-diffusion and
  pure-diffusion models are unchanged to 1e-12 (the explicit part of
  Gray-Scott is its reaction, and a diffusion-only Strang step is still
  the exact discrete decay).
  `tests/test_imex.py::TestIMEXExplicitPart` and
  `::TestIMEXSplitValidation` cover all of this.

- **A step whose Newton solve failed no longer takes the PID controller's
  accepted branch, and one step's rejections are now bounded.**
  `propose_dt` and `propose_dt_imex` decided acceptance from
  `err_ratio <= 1.0` alone, but an error estimate built from a failed
  solve can be arbitrarily small, so a failed step was handed to the PID
  controller, whose growth term (up to `max_factor = 5`) outran the
  integrator's halving on rejection. On `u' = -u` at `u0 = 1000` in
  float32, where the default `newton_tol = 1e-8` is below the float32
  spacing of 1000 and no solve can ever converge, `dt` plateaued between
  1.6e-3 and 1.8e-3 and `adaptive_integrate(..., max_steps=1)` never
  returned at all (killed at 90 s): `max_steps` bounds accepted steps and
  no step was ever accepted. Both proposal functions now take the
  integrator's own `accepted` decision (error test **and** finiteness
  **and** solve convergence), so a failed step always takes the rejected
  branch, whose factor is at most 1 and which the implicit robustness
  limiter can only shrink further. **Public surface:** `propose_dt` and
  `propose_dt_imex` gained an optional `accepted` argument, defaulting to
  the old `err_ratio <= 1.0` so existing callers are unchanged;
  `adaptive_integrate` and `adaptive_integrate_imex` gained
  `max_rejections_per_step` (default 10, CVODE's `MXNCF`), the consecutive
  rejections one step may spend before the run stops with the new
  `StatusCode.MAX_ATTEMPTS_REACHED` (6). That budget is independent of
  `max_steps`, which counts accepted steps only. Both integrators also
  keep the controller state the proposal returns on a rejection; they used
  to discard it, which is why `consecutive_rejects` was written but never
  seen by anyone. The reproduction now stops in about 2 s with
  `MAX_ATTEMPTS_REACHED`, 0 accepted steps and 10 rejections. Well-posed
  runs are untouched: the six methods on a logistic model, RK4 and BDF2 on
  a stiff decay that does reject steps, and both IMEX variants on
  Gray-Scott give identical statuses, accept/reject counts, `dt` histories
  and `t` histories before and after.
  `tests/test_dt_policy.py::TestRejectionsTerminate` covers all three.

- **`integrate_fixed_dt` no longer returns a failed Newton solve as an
  ordinary result; it raises.** `do_be`, `do_cn` and `do_bdf2` each dropped
  the `NKStats` their step function returns (`y_new, _ = be_step(...)`) and
  the scan carry had no status field at all, where the adaptive integrator
  carries a `StatusCode` and rejects a step on `nk_stats.converged`. On
  `u' = -u^3`, `u0 = 1`, `dt = 1` with `max_newton_iters = 1`, `be_step`
  returns `u = 0.5` with `converged = False` and residual `0.6495`, and
  `integrate_fixed_dt` returned that `0.5` with no indication of any kind.
  The scan now carries a status: a step whose state is not finite, or whose
  Newton-Krylov solve did not converge, records `NON_FINITE_VALUES` or
  `NK_FAILED`, keeps the last good state, and turns every later step into a
  no-op. **Public surface:** the default return is still the same
  three-element tuple (every caller in the tree and every documented
  example unpacks exactly three values), and a run that did not finish now
  raises `RuntimeError` naming the status. The new `return_status=True`
  keyword returns the `StatusCode` as a fourth element instead of raising,
  which is what a caller tracing this function under `jit` must use, since
  raising on a tracer is not possible. A run in which nothing fails is
  unchanged: the four fixed-step methods on the 8x8 Gray-Scott model
  (`t_end = 0.5`, `dt = 0.05`) produce bit-identical histories and final
  states before and after.
  `tests/test_integrators.py::TestFixedStepReportsFailedSolves` covers the
  reproduction, the frozen tail, the explicit blow-up, and the
  bit-identical converging run.

- **`exact_cfl_dt('imex')` and `exact_cfl_dt('etd')` now raise
  `NotImplementedError` instead of returning `safety * 1.0`.** Both
  branches ignored `op.eigenvalues` entirely and returned a literal
  constant that looked like a real stability bound; no caller in the tree
  uses either branch (grep confirms). A wrong number that resembles a
  real answer is worse than a refusal, so both now name the unimplemented
  branch in the raised error instead. `'explicit'` is unaffected.

- **`cn_step` and `bdf2_step` now hand the preconditioner their own effective
  diffusive step, not the outer `dt`.** `newton_krylov_solve` builds the
  `PrecondContext` from whatever `dt` it is given, but CN's Newton Jacobian
  is `I - (dt/2)*F'(y)` and BDF2's is `alpha0*I - dt*F'(y)`
  (`alpha0 = (1+2w)/(1+w)`, 1.5 at a constant step): passing the outer `dt`
  to a linear FFT diffusion preconditioner built for `(I - dt*D*Laplacian)`
  makes it only approximately invert the actual Jacobian. `cn_step` now
  passes `dt/2`; `bdf2_step` passes `dt/alpha0` and scales the
  preconditioner's output by `1/alpha0`, via two new `newton_krylov_solve`
  keywords, `precond_dt` (default `dt`) and `precond_scale` (default 1).
  Measured on a 16-point periodic grid at `dt*D = 1` (constant step): CN's
  preconditioned eigenvalues were `[0.50, 1.0]`, now exactly `1`; BDF2's
  were `[1.00, 1.5]`, now exactly `1`. Performance only (fewer Krylov
  iterations to converge); no step produces a different result.

- **`imex_ssprk2_step` no longer recomputes each stage's Laplacian through a
  second FFT round trip.** Both stages already solve
  `(I - gamma dt L) U = rhs`, so `L U = (U - rhs) / (gamma * dt)` on the
  interior; the step used this identity for neither and called
  `diffusion_rhs_fft` again instead. No numerical change (the two agree to
  about 3e-14 on a Gray-Scott state); measured about 1.5x faster per step
  with the redundant FFT removed.

- **The numerical range is a 2-spectral set, not a `1 + sqrt(2)`-spectral
  set.** Crouzeix's conjecture was proved in 2026 with the sharp constant 2
  (Jin, "The Numerical Range Is a 2-Spectral Set", Preprints.org,
  doi:10.20944/preprints202607.1919.v4; Lorist and Schwenninger, "A solution
  to Crouzeix's conjecture", arXiv:2608.03841), superseding Crouzeix and
  Palencia, SIAM J. Matrix Anal. Appl. 38(2) 2017, doi:10.1137/17M1116672.
  `_CP_PREFACTOR` is now a single definition in `field_of_values.py`
  (`non_normality.py` carried its own duplicate), imported by
  `non_normality.py` and `figures.py`. The drawn Crouzeix-Palencia envelopes
  tighten by a factor of `(1 + sqrt(2)) / 2`, about 1.21x; no verdict in
  `assess_preconditioner` depends on the constant's value.

## [1.2.0] - 2026-09-06

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

- **`nilt_solve_linear_pde` tuned and inverted a transform with two extra
  poles the tuner never saw.** It inverted `G_k(s) = U_k(s) - u0_k/(s + c)`
  with `c = 1/t_end`, which keeps the source pole at the origin and adds a
  second pole at `-1/t_end` invisible to `tune_nilt_for_fft_operator`; on
  `eigenvalues = full(8, -10)`, `u0 = ones(8)`, `t_end = 1` the tuner picked
  `a = 0` and returned `-0.006816` instead of `e^{-10} = 4.540e-5`. The
  bridge now removes everything the closed form already knows (the
  particular constant `-f_k/lambda_k`, or the whole polynomial
  `u0_k + f_k t` when `lambda_k` is spectrally zero) and inverts only the
  transient `H_k(s) = w_k * [1/(s - lambda_k) - 1/(s + c_k)]` with
  `c_k = -Re(lambda_k)`, tuned against that transform's own abscissa. On a
  real spectrum both poles of `H_k` coincide and the bridge is exact by
  construction; the NILT budget is now spent only on the oscillatory part
  of a spectrum.

- **`AdvectionDiffusionOperator` carried a complex eigenvalue at the
  self-paired Nyquist mode of an even grid.** The odd symbol `-i*v*k` is
  not Hermitian at `k = -pi/dx`, a mode with no distinct conjugate
  partner; on an 8-point grid with `v = 1`, `D = 0`, `u0[j] = (-1)^j`,
  `t_end = 0.1` `nilt_solve_linear_pde` returned `-5.638635` at the first
  point instead of leaving the stationary checkerboard mode unchanged. The
  Nyquist entry is now zeroed, reusing the `_odd_symbol_wavenumber`
  convention already applied to the FFT preconditioners, and
  `nilt_solve_linear_pde` raises if a self-paired mode (index 0, or `N//2`
  on an even grid) is ever handed a nonzero imaginary eigenvalue.

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
