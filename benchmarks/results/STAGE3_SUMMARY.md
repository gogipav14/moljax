# Stage 3: resolved Brusselator conditioning study

## Method and provenance

This experimental conditioning study evaluates preconditioned backward-Euler/JFNK systems visited
by moljax's shipped `create_brusselator_periodic_fft` factory. It covers Hopf and Turing regimes
on periodic two-dimensional grids, including 256 by 256 physical-grid cases with a 133128-component
padded two-field operator. The run is based on merged main `d4047f3` and frozen v1.2.1 conditioning.

Each source state is persisted once under the v4 contract with its generation fingerprint,
convergence status, and SHA256 identity. Reassessment reloads and validates that artifact instead
of re-solving a trajectory. Initially unresolved FOV supports are escalated through `32/120/2`,
`64/180/2`, and `96/240/2` (angles/support iterations/restarts); unresolved support at the cap
fails closed. Records retain final FOV configuration, source provenance, and a separate LOBPCG
sigma-min upper estimate.

## Certified adequacy at scale

Dense `full_operator_epsilon_zero` materializes a 133128 by 133128 operator: about 264 GiB before
the cubic SVD cost, so it is infeasible here. Pavlov derived the Fourier--Weyl--ghost lower bound
used as the scalable full-operator route: a constant-coefficient Fourier block bound, a pointwise
Weyl perturbation, and the padded ghost-cell `g(b,c)` correction. Each record stores selected K0,
b0, perturbation norm, c, padding convention, and the final bound.

The perturbation step uses the matrix spectral norm in the Weyl--Mirsky singular-value inequality
(L. Mirsky, *Quart. J. Math.* 11 (1960), 50--59, doi:10.1093/qmath/11.1.50); the exact `g(b,c)`
formula follows from the inverse of the block-triangular ghost structure.

The bound was independently checked on small dense problems and never exceeded dense sigma-min.
It uses the same float64 standard as the dense helper; a strictly rounded mathematical certificate
would additionally require directed rounding or interval arithmetic. LOBPCG is an upper estimate,
not adequacy evidence. A bound at least 0.1, corroborated supports, and an origin-outside FOV
certify `adequate`; without the bound these records would stop at provisional.

## Final resolved tally

| Preset | Adequate | Provisional | Investigate | Indeterminate | Uncertified at cap |
| --- | ---: | ---: | ---: | ---: | ---: |
| screen_64 | 8 | 0 | 0 | 0 | 0 |
| developed_64 | 0 | 0 | 0 | 11 | 1 |
| fixed_dt_256 | 3 | 0 | 4 | 1 | 0 |
| hopf_continuation_256 | 2 | 0 | 2 | 0 | 0 |
| **All 32 records** | **13** | **0** | **6** | **12** | **1** |

All 13 adequate records clear the 0.1 bound gate, corroborate support geometry, and leave the
origin outside. Categories are fail closed:

- `adequate`: all gates, including the full-operator lower bound, pass.
- `provisional`: a valid lower bound is below 0.1, supports corroborate, the origin is outside,
  and every other gate passes, so epsilon zero is the only failed gate.
- `investigate`: supports corroborate and the origin is outside, but a disk-rate caution remains.
- `indeterminate`: the corroborated FOV encloses the origin; this dominates other categories.
- `uncertified_at_cap`: support geometry did not converge at the escalation cap.

No record is provisional. The six investigate records have disk rates from 0.9252 through
0.9980. Five of them clear the bound gate. The sixth is late fixed-dt 256 by 256 Turing with
identity preconditioning: its valid bound is 0.0, its origin is outside, supports corroborate at
`64/180/2`, and its disk rate of 0.9980 fails the 0.9 gate, a caution that a bound below the
adequacy gate cannot remove. The one at-cap
record is developed-64 Turing step 120 with identity preconditioning; its support solve remains
unconverged through `96/240/2`.

## Refined scientific result

Origin enclosure is an empirical property of timestep, visited state/regime, and preconditioner;
it is not an unconditional property of developed Turing states. On the same late 256 by 256,
dt=0.2 Turing state, identity has an origin-outside FOV (disk rate 0.9980, 84 GMRES iterations,
investigate, with a bound of 0.0), while FFT diffusion encloses the origin (disk rate 1.5918,
9 GMRES iterations, indeterminate). The same state therefore has opposite enclosure outcomes
under the two preconditioners.

At dt=1 on the developed-64 exploration, all corroborated Hopf and Turing records enclose the
origin; the developed-64 Turing step-120 identity record is excluded because its support geometry
did not corroborate at the cap, so its origin status is unresolved rather than a counterexample.
At 256 by 256, developed Hopf samples remain origin-outside for both preconditioners at
dt=0.2 and dt=0.05; FFT is adequate while identity is investigate. These configurations change
the trajectories as well as dt, so this is a timestep-dependent empirical pattern, not an isolated
causal claim about dt.

The late 256 by 256 Turing result sharpens the open problem: FFT diffusion is fastest (9 GMRES
iterations versus 84 for identity) precisely where the enclosing-disk criterion must abstain. A
second, non-disk criterion is needed for this origin-enclosed-but-fast-converging regime.
Experimental pseudospectral work is forthcoming after the PRs, as agreed with the maintainer; it
is not included here.

## Scope and reproduction

This is a conditioning study, not an exact-solution error study. It stages experimental evidence
for two-dimensional periodic Brusselator operators and reports the one uncorroborable support case
honestly. Regenerate source studies with `benchmarks/brusselator_conditioning.py`, resolve supports
with `benchmarks/resolve_brusselator_fov_supports.py`, and regenerate ignored figures with
`PYTHONPATH="$PWD" python benchmarks/make_stage3_figures.py`.

Every number in this summary derives from the four committed JSON files beside it.
