# Stage 2: nonlinear-diffusion conditioning coverage

## Method and valid-state provenance

This study evaluates fixed backward-Euler Newton systems for one-dimensional,
node-centered Dirichlet nonlinear diffusion. It uses the matrix-free
conditioning procedure present at base revision
`131e6319f3a9a19fee583ad26e77142d0830a8fb`: field-of-values and
pseudospectral measurements, an outer-disk bound, two independent support
restarts, and domain checks for scalar readings. Each geometry result is
paired with a counted GMRES solve of the same fixed linear system.

This regeneration also corrects the experimental diffusion coefficient to the
smooth nonnegative divergence-form derivative. The experimental reference-state
solves use the existing `NKParams` controls with `max_backtrack=6` and
`max_newton_iters=13`; no core solver was changed. All 21 source states
converged before assessment. Each converged float64 source array was persisted
once, identified by shape, dtype, and SHA256, and loaded for every base,
escalation, and reproducibility assessment. The assessment path fails closed
rather than diagnosing a non-converged source state.

The base geometry budget is 16 angles, 60 field-of-values iterations, and two
restarts. Uncertified base records were escalated through 32/120/2, 64/180/2,
and 96/240/2. In PME, 172 certified records used 16/60/2, 75 used 32/120/2,
12 used 64/180/2, and eight used 96/240/2; three additional 96/240/2 attempts
remained uncertified at the cap. Porous-Fisher has 27 certified records at
16/60/2 and 18 at 32/120/2.

## Final categories and coverage

- **Certified adequate:** corroborated support geometry supports the adequate
  decision.
- **Certified investigate:** corroborated support geometry supports a cautious
  warning without origin enclosure.
- **Certified indeterminate:** corroborated support geometry reaches a
  fail-closed indeterminate decision. In the nonlinear and reaction-axis cases
  below, these are origin-enclosed results; the linear-control guard is stated
  separately.
- **Uncertified at cap:** support convergence or two-restart corroboration was
  still absent at 96/240/2. This is a fail-closed resolution outcome, not a
  preconditioner-inadequacy claim.

| Study | Certified adequate | Certified investigate | Certified indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: |
| PME (270 records) | 65 | 153 | 49 | 3 |
| Porous-Fisher (45 records) | 15 | 13 | 17 | 0 |

## Reaction axis

The reaction-axis headline uses the two identity-hard states at each reaction
strength. Its final count table is:

| Reaction strength | Certified adequate | Certified investigate | Certified indeterminate |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 100 | 0 | 0 | 2 |

At `r=0` and `r=1`, the `dt=0.02` identity systems are certified investigate:
their disk rates are 0.954237 and 0.955164, their origins are outside the
field of values, and counted GMRES takes 57 and 58 iterations. The `dt=2`
identity systems at `r=0` and `r=1` are certified indeterminate with origin
enclosure. At `r=100`, both hard identity systems (`dt=0.02` and `dt=2`) are
certified indeterminate with origin enclosure. These are certified results,
not unresolved base-budget readings.

## Linear controls

The final `m=1` data does not support the prior statement that 36 controls
were unresolved. All 45 `m=1` records now have certified categories: three
adequate, six investigate, and 36 indeterminate; none is uncertified at cap.
The identity variant supplies the three adequate and six investigate records.
Each of the four frozen-style variants (`frozen_mean`, `frozen_bulk`, `floor`,
and `const`) contributes nine certified-indeterminate records. For those 36
controls, `n_right_real_outliers` is null on a degenerate near-identity Ritz
input, so the scalar-domain guard returns indeterminate even though the support
geometry is corroborated and the origin is outside. These baseline controls are
not nonlinear-PME evidence.

## Nonlinear PME

The following focused breakdown covers the 135 final records with `m` in
`{2, 4, 8}`. It gives state-dependent decisions for the frozen-coefficient
variants rather than a blanket certificate.

| Variant | Certified adequate | Certified investigate | Certified indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: |
| `frozen_mean` | 8 | 14 | 5 | 0 |
| `frozen_bulk` | 8 | 17 | 1 | 1 |
| `floor` | 8 | 19 | 0 | 0 |
| `const` | 7 | 20 | 0 | 0 |
| `identity` | 8 | 17 | 2 | 0 |

The three PME records that remain uncertified at cap all use converged,
SHA256-verified source states and have origins outside the numerical range:

| `m` | Front case | Analysis `dt` | Variant | Reason at 96/240/2 |
| ---: | ---: | ---: | --- | --- |
| 2 | 2 | 0.02 | `frozen_bulk` | support solve did not converge |
| 3 | 1 | 0.02 | `identity` | two-restart corroboration failed |
| 3 | 1 | 2 | `identity` | two-restart corroboration failed |

They are not domain-guard cases, not non-converged reference states, and not
evidence of origin enclosure. The counted-GMRES measurement remains the
operational counterpart to every geometry record; this summary does not infer
solve difficulty from geometry alone.

## Scope and reproducibility

The coverage is limited to one-dimensional node-centered Dirichlet systems,
the recorded state schedule, and non-uniform per-record geometry budgets.
Fresh assessments of boundary, adequate, investigate, indeterminate, and all
three uncertified-at-cap examples reused the persisted source arrays, matched
their SHA256 identities, and reproduced their final categories and disk rates
to floating-point precision.

Regenerate the base studies with `benchmarks/pme_breakdown.py` and
`benchmarks/porous_fisher_conditioning.py`. The persisted-state audit and
recorded-budget escalation are implemented by
`benchmarks/regenerate_stage2_conditioning.py`; the result files are
`pme_breakdown.json` and `porous_fisher_conditioning.json`. Regenerate the
ignored figures from those JSON files with `benchmarks/make_stage2_figures.py`.
