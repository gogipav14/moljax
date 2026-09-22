# Stage 2: nonlinear-diffusion conditioning coverage

## Method and valid-state provenance

This study evaluates fixed backward-Euler Newton systems for one-dimensional,
node-centered Dirichlet nonlinear diffusion. The final audit is bound to the
immutable `v1.2.1` conditioning base
`54852e477ed2d32ac13c1c7e711b28c81f320f13`. Each reading combines the
matrix-free field-of-values procedure, an enclosing-disk decision bound, two
operator-keyed support restarts, scalar-domain guards, and a counted GMRES
solve of the same fixed linear system.

The experimental diffusivity is the smooth nonnegative divergence-form
regularization:

```text
D_epsilon(u) = m * (u**2 + epsilon**2)**((m - 1) / 2).
```

It represents the nonnegative diffusion magnitude at sign-changing and zero
nodes rather than a signed coefficient, so it does not introduce an
anti-diffusive sign flip. In particular,
`D_epsilon(0) = m * epsilon**(m - 1)`, the exact value returned by
`d0_floor`.

All 21 source states converged before assessment, including the five
wide-front PME source states (`m=2,3,4,6,8`, front halfwidth 3). They now
converge using the v1.2.1 default Newton line search (`max_backtrack=8`) and
`max_newton_iters=13`; this audit carries no experimental backtracking
override. The 75 records depending on those wide-front states are therefore
valid measurements. Each converged float64 source array was saved once with
its shape, dtype, and SHA256, then loaded and hash-verified for every
diagnostic; a non-converged source state would fail closed instead of being
diagnosed.

The base geometry budget is 16 angles, 60 field-of-values iterations, and two
restarts. Uncertified base records use the 32/120/2, 64/180/2, and 96/240/2
ladder. Records whose support solve remained marginal at the terminal budget
were additionally checked at 240 and then 480 field-of-values iterations;
restart-corroboration failures remain fail closed because the public result is
boolean at that point.

## Final categories and coverage

- **Certified adequate:** all numerical gates pass, including a
  full-operator lower bound for `epsilon_zero`.
- **Certified investigate:** corroborated support geometry supports a
  cautionary decision without origin enclosure.
- **Certified indeterminate:** corroborated support geometry gives a
  fail-closed indeterminate decision, through origin enclosure or a
  scalar-domain guard.
- **Uncertified at cap:** support convergence or two-restart corroboration is
  absent at the terminal budget. It is a resolution outcome, not a
  preconditioner-inadequacy claim.

| Study | Certified adequate | Certified provisional | Certified investigate | Certified indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| PME (270 records) | 59 | 0 | 154 | 50 | 7 |
| Porous-Fisher (45 records) | 15 | 0 | 13 | 17 | 0 |

No provisional categories remain. The 60 provisional PME records were
reassessed with `full_operator_epsilon_zero(matvec, n)` on their recorded
terminal geometry: 59 become certified adequate and one becomes certified
investigate. All 15 provisional Porous-Fisher records become certified
adequate. This uses the helper's dense route (2n matvecs plus an n-by-n SVD),
not reduced-Arnoldi coverage. The helper took 60.08 seconds total for the 60
PME readings (median 1.204 seconds each) and 9.77 seconds total for the 15
Porous-Fisher readings (median 0.652 seconds each).

Two PME support-convergence at-cap records clear at 480 iterations: the
`m=6`, front-1, `dt=2`, `frozen_mean` record becomes certified indeterminate,
and the `m=6`, front-2, `dt=0.02`, `frozen_mean` record becomes certified
investigate. Seven PME records remain at cap; all have converged support
solves and fail only two-restart corroboration:

| m | Front | dt | Variant | Origin status |
| ---: | ---: | ---: | --- | --- |
| 3 | 1 | 0.0002 | `frozen_bulk` | outside |
| 3 | 1 | 0.0002 | `floor` | outside |
| 3 | 1 | 0.0002 | `identity` | outside |
| 3 | 1 | 0.02 | `frozen_bulk` | outside |
| 3 | 1 | 0.02 | `floor` | outside |
| 3 | 1 | 0.02 | `identity` | outside |
| 3 | 1 | 2 | `identity` | outside |

Relative to the earlier `ef8be4b` tally (PME 0/58/143/47/22 and
Porous-Fisher 0/15/13/17/0 for adequate/provisional/investigate/
indeterminate/at-cap), the final v1.2.1 tally is PME 59/0/154/50/7 and
Porous-Fisher 15/0/13/17/0. The full-operator evidence removes the
provisional category, the support-iteration pass clears two marginal
support-solve records, and the frozen-base regeneration measures the C1-fixed
operator on valid wide-front source states. The remaining differences are
therefore reported from the final v1.2.1 record set rather than attributed to
an unsupported single cause.

## Reaction axis

The reaction-axis headline uses raw counts, not percentages, for the two
identity-hard states at each reaction strength:

| Reaction strength | Certified adequate | Certified investigate | Certified indeterminate |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 100 | 0 | 0 | 2 |

At `r=0` and `r=1`, the `dt=0.02` identity systems are certified investigate;
their origins are outside the field of values and counted GMRES takes 57 and
58 iterations, respectively. The `dt=2` identity systems at `r=0` and `r=1`,
and the `dt=0.02` and `dt=2` hard identity systems at `r=100`, are certified
indeterminate with origin enclosure. These are certified results, not
unresolved base-budget readings.

## Linear controls and nonlinear PME

All 45 `m=1` controls are certified: three identity records are adequate, six
identity records investigate, and 36 frozen-style records are indeterminate.
The 36 indeterminate controls have null `n_right_real_outliers` on a
degenerate near-identity Ritz input; their support geometry is corroborated
and the origin is outside. They are baseline controls, not nonlinear-PME
evidence.

The following breakdown covers the 135 records with `m` in `{2, 4, 8}`.
It aggregates all three front cases for that subset (38 adequate records); the
`m=3` and `m=6` subsets, not shown in this table, contribute 8 and 10
adequate records, respectively. With the 3 adequate `m=1` identity controls,
the full PME adequate count reconciles as `38 + 8 + 10 + 3 = 59`.

| Variant | Adequate | Investigate | Indeterminate | At cap |
| --- | ---: | ---: | ---: | ---: |
| `frozen_mean` | 8 | 14 | 5 | 0 |
| `frozen_bulk` | 7 | 19 | 1 | 0 |
| `floor` | 8 | 19 | 0 | 0 |
| `const` | 7 | 19 | 1 | 0 |
| `identity` | 8 | 17 | 2 | 0 |

The seven unresolved records are all `m=3`, front-1 cases listed above; none
is presented as a certified decision.

## Scope and reproducibility

Coverage is limited to one-dimensional node-centered Dirichlet systems, the
recorded state schedule, and non-uniform per-record geometry budgets. The
result JSONs record convergence status, source-array SHA256, terminal budget,
audit base revision, full-operator evidence when used, and recovery metadata.
Fresh assessments load the same persisted source array and verify its SHA256
before diagnosis.

Regenerate the base studies with `benchmarks/pme_breakdown.py` and
`benchmarks/porous_fisher_conditioning.py`. The persisted-state audit and
recovery workflow is in `benchmarks/regenerate_stage2_conditioning.py`; figures
are generated directly from the JSONs by `benchmarks/make_stage2_figures.py`.
