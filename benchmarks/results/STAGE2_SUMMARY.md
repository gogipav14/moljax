# Stage 2: nonlinear-diffusion conditioning coverage

## Method

This study evaluates fixed backward-Euler Newton systems for one-dimensional,
node-centered Dirichlet nonlinear diffusion.  Each record uses the bba5a94
reformed matrix-free conditioning procedure: field-of-values and pseudospectral
measurements, an outer-disk bound, two independent support restarts, and
domain checks for scalar readings.  The geometry result is always paired with
a counted GMRES solve of the same fixed linear system.

The base geometry budget is 16 angles, 60 field-of-values iterations, and two
restarts.  Records whose support evidence could not be certified were replayed
at their recorded escalation budget: 32/120/2, 64/180/2, or 96/240/2.  The
final records were then reassessed at those stored budgets with two restarts.

## Categories and coverage

The final data distinguishes four categories.

- **Certified adequate:** corroborated support geometry supports the adequate
  decision.
- **Certified investigate:** corroborated support geometry supports a cautious
  non-adequate warning without origin enclosure.
- **Certified indeterminate:** corroborated support geometry leads to a
  fail-closed indeterminate decision.  For the nonlinear and reaction-axis
  cases below this is an origin-enclosed result; the linear-control domain-guard
  case is described separately.
- **Uncertified at cap:** the support evidence was not corroborated even at
  96/240/2.  The procedure declines to certify a geometry verdict rather than
  guessing; this is not a preconditioner-inadequacy claim.

| Study | Certified adequate | Certified investigate | Certified indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: |
| PME (270 records) | 65 | 132 | 53 | 20 |
| Porous-Fisher (45 records) | 15 | 13 | 17 | 0 |

The final PME resolution metadata records 230 results at 16/60/2, 14 at
32/120/2, five at 64/180/2, and one at 96/240/2; the remaining 20 are the
uncertified-at-cap records.  Porous-Fisher records use 36, five, and four
results at those respective certifying budgets.

## Reaction axis

The reaction-axis headline is based on the two identity-hard states at each
reaction strength.  Their raw verdict counts are:

| Reaction strength | Certified adequate | Certified investigate | Certified indeterminate |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 100 | 0 | 0 | 2 |

At `r=0` and `r=1`, the `dt=0.02` identity systems are certified investigate:
their disk rates are 0.954237 and 0.955165, their origins remain outside the
field of values, and counted GMRES takes 57 and 58 iterations.  The `dt=2`
identity systems at `r=0` and `r=1` are certified indeterminate with the
origin enclosed.  At `r=100`, both hard identity states (`dt=0.02` and `dt=2`)
are certified indeterminate with the origin enclosed.  These are certified
current-geometry results, not unresolved base-budget readings.

## Linear controls

All 36 changed records are `m=1` baseline controls using the constant-like
preconditioner variants (`frozen_mean`, `frozen_bulk`, `floor`, and `const`)
over three front cases and three implicit step sizes.  They changed from
adequate to certified indeterminate because the current domain guard receives
`n_right_real_outliers = null` for a degenerate near-identity Ritz input.  The
guard therefore declines to certify adequacy from an uninformative spectral
reading.  These controls are not nonlinear-PME evidence.

## Nonlinear PME

For the nonlinear exponents `m=2`, `m=4`, and `m=8`, the 135 final records are
39 certified adequate, 75 certified investigate, 14 certified indeterminate,
and seven uncertified at cap.  Thus the frozen-coefficient variants receive
state-dependent decisions across the visited systems rather than a blanket
certificate.

| Variant | Certified adequate | Certified investigate | Certified indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: |
| `frozen_mean` | 8 | 9 | 7 | 3 |
| `frozen_bulk` | 8 | 15 | 2 | 2 |
| `floor` | 8 | 14 | 3 | 2 |
| `const` | 7 | 18 | 2 | 0 |
| `identity` | 8 | 19 | 0 | 0 |

The counted-GMRES measurements remain the operational counterpart to every
geometry classification; the summary does not infer solve difficulty from
geometry alone.

## Scope and reproducibility

The coverage is limited to one-dimensional node-centered Dirichlet systems and
to the recorded state schedule and per-record geometry budgets.  The
uncertified-at-cap category is explicit fail-closed behavior, not evidence that
the corresponding preconditioner is inadequate.

Regenerate the studies with `benchmarks/pme_breakdown.py` and
`benchmarks/porous_fisher_conditioning.py`.  The final current-geometry audit
and recorded-budget replay are implemented by
`benchmarks/regenerate_stage2_conditioning.py`; the source JSONs are
`pme_breakdown.json` and `porous_fisher_conditioning.json`.  Figures are
regenerated from those JSONs by `benchmarks/make_stage2_figures.py`.
