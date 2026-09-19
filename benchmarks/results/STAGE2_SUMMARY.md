# Stage 2: nonlinear-diffusion conditioning coverage

## Method and valid-state provenance

This study evaluates fixed backward-Euler Newton systems for one-dimensional,
node-centered Dirichlet nonlinear diffusion. The final audit uses the
matrix-free conditioning procedure at base revision
`ef8be4bd058dc4d03d4a81270f801659fd5b7a96`: field-of-values and reduced
pseudospectral measurements, an outer-disk bound, two operator-keyed support
restarts, scalar-domain guards, and the reduced-Arnoldi coverage gate for
`epsilon_zero`. Each geometry result is paired with a counted GMRES solve of
the same fixed linear system.

The experimental diffusion coefficient is the smooth nonnegative
divergence-form derivative. Reference-state solves use the existing
`NKParams(max_backtrack=6, max_newton_iters=13)` controls; no core solver was
changed. All 21 source states converged before assessment. Each converged
float64 source array was persisted once, identified by shape, dtype, and
SHA256, then loaded for every diagnostic. The path fails closed rather than
diagnosing a non-converged state.

The base geometry budget is 16 angles, 60 field-of-values iterations, and two
restarts. Uncertified base records were escalated through 32/120/2, 64/180/2,
and 96/240/2. The final audit re-read every record at its already-recorded
terminal budget; it neither regenerated source states nor re-searched budgets.

## Final categories and coverage

- **Certified provisional:** every numerical gate passed and support geometry
  was corroborated, but `epsilon_zero` came from a reduced Arnoldi projection
  and is not full-operator evidence.
- **Certified investigate:** corroborated support geometry supports a
  cautionary decision without origin enclosure.
- **Certified indeterminate:** corroborated support geometry produces a
  fail-closed indeterminate decision, commonly through origin enclosure or a
  scalar-domain guard.
- **Uncertified at cap:** support convergence or two-restart corroboration was
  absent at the record's terminal budget. This is a resolution outcome, not a
  preconditioner-inadequacy claim.

| Study | Certified adequate | Certified provisional | Certified investigate | Certified indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| PME (270 records) | 0 | 58 | 143 | 47 | 22 |
| Porous-Fisher (45 records) | 0 | 15 | 13 | 17 | 0 |

The update from the prior `131e631` results is deliberate and fail closed:
73 formerly adequate records are now provisional because their six-step
Arnoldi reductions do not establish full-operator coverage (58 PME and 15
Porous-Fisher). Operator-keyed restart seeding also changed support
corroboration for 21 PME records: 20 now remain uncertified at their recorded
cap, while one prior at-cap record now certifies as investigate. No source
state changed during this audit.

## Reaction axis

The reaction-axis headline uses the two identity-hard states at each reaction
strength. Its final count table is unchanged:

| Reaction strength | Certified provisional | Certified investigate | Certified indeterminate |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 1 |
| 1 | 0 | 1 | 1 |
| 100 | 0 | 0 | 2 |

At `r=0` and `r=1`, the `dt=0.02` identity systems are certified investigate;
their origins are outside the field of values and counted GMRES takes 57 and
58 iterations. The `dt=2` identity systems at `r=0` and `r=1`, and both hard
identity systems at `r=100`, are certified indeterminate with origin
enclosure. These are certified results, not unresolved base-budget readings.

## Linear controls and nonlinear PME

All 45 `m=1` controls have certified categories: three identity records are
provisional, six identity records investigate, and 36 frozen-style records
are indeterminate. The 36 indeterminate controls have null
`n_right_real_outliers` on a degenerate near-identity Ritz input; their support
geometry is corroborated and the origin is outside. They are baseline controls,
not nonlinear-PME evidence.

The following breakdown covers the 135 records with `m` in `{2, 4, 8}`.

| Variant | Provisional | Investigate | Indeterminate | At cap |
| --- | ---: | ---: | ---: | ---: |
| `frozen_mean` | 7 | 12 | 4 | 4 |
| `frozen_bulk` | 8 | 17 | 1 | 1 |
| `floor` | 8 | 19 | 0 | 0 |
| `const` | 7 | 18 | 0 | 2 |
| `identity` | 8 | 17 | 2 | 0 |

The 22 PME records at cap use converged, SHA256-verified source states. Fifteen
have a support solve that did not converge; seven have restart corroboration
failure. Twenty have origins outside the traced field of values and two have
origin enclosure, but none is presented as a certified decision while its
support evidence is inconsistent.

## Scope and reproducibility

Coverage is limited to one-dimensional node-centered Dirichlet systems, the
recorded state schedule, and non-uniform per-record geometry budgets. Fresh
assessments load the same persisted source array and verify its SHA256 before
diagnosis. The result JSONs record the source-state provenance, convergence
status, terminal budget, current audit base revision, and every changed record.

Regenerate the base studies with `benchmarks/pme_breakdown.py` and
`benchmarks/porous_fisher_conditioning.py`. The persisted-state audit is in
`benchmarks/regenerate_stage2_conditioning.py`; figures are regenerated from
the JSONs by `benchmarks/make_stage2_figures.py`.
