# Stage 2 experimental summary

## Thesis

The visited-state field-of-values and pseudospectra decision procedure validated in Stage 1 on an
FFT-preconditioned operator is tested here on nonlinear diffusion (PME) and reaction-diffusion
(Porous--Fisher). These are experimental staging results, not a public API or a claim of a new
preconditioner.

## PME regime map

The table uses identity-preconditioned systems to define hard and easy visited states within each
`(m, dt)` cell. It reports the JSON median enclosing-disk rate for each group and the observed
identity GMRES iteration range. Every nonlinear cell (`m > 1`) has a higher hard-state disk rate
than easy-state disk rate. The `m = 1` linear controls are state-independent within each cell, so
their per-cell Pearson coefficient and easy-state group are undefined.

| m | candidate dt | identity iterations (min--max) | easy disk rate | hard disk rate | per-cell Pearson |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2e-04 | 4--4 | -- | 0.6057 | -- |
| 1 | 2e-02 | 81--81 | -- | 0.9680 | -- |
| 1 | 2e+00 | 202--202 | -- | 0.9738 | -- |
| 2 | 2e-04 | 4--18 | 0.0311 | 0.5863 | 1.00 |
| 2 | 2e-02 | 9--141 | 0.7624 | 0.9930 | 1.00 |
| 2 | 2e+00 | 12--191 | 0.9969 | 0.9999 | 1.00 |
| 3 | 2e-04 | 4--29 | 0.0596 | 0.7464 | 1.00 |
| 3 | 2e-02 | 11--165 | 0.8638 | 0.9966 | 1.00 |
| 3 | 2e+00 | 13--176 | 0.9984 | 1.0000 | 1.00 |
| 4 | 2e-04 | 5--40 | 0.0817 | 0.8107 | 1.00 |
| 4 | 2e-02 | 11--194 | 0.8989 | 0.9977 | 1.00 |
| 4 | 2e+00 | 12--229 | 0.9989 | 1.0000 | 1.00 |
| 6 | 2e-04 | 5--46 | 0.1118 | 0.8648 | 1.00 |
| 6 | 2e-02 | 12--199 | 0.9264 | 0.9984 | 1.00 |
| 6 | 2e+00 | 16--210 | 0.9992 | 1.0000 | 1.00 |
| 8 | 2e-04 | 5--45 | 0.1305 | 0.8872 | 1.00 |
| 8 | 2e-02 | 13--211 | 0.9375 | 0.9987 | 1.00 |
| 8 | 2e+00 | 15--222 | 0.9993 | 1.0000 | 1.00 |

Source: `benchmarks/pme_breakdown.py` (the full raw sweep is regenerated rather than tracked),
`regime_map.cells`.

The 1.00 coefficients are endpoint monotonicity from two front samples per nonlinear cell, not
statistical estimates. The breadth is the result: all 15 nonlinear cells separate their two
visited-state endpoints. The pooled identity Pearson is 0.5107; it is a regime-pooling artifact
and is deliberately not the headline result.

## Reaction axis

For each reaction strength, hard identity states are those at or above that strength's JSON median
identity iteration count. The last column is the hard-state verdict composition in
adequate/investigate/indeterminate order. Separation survives all three reaction strengths, but
the language becomes less specific as reaction increases.

| r | identity iterations (min / median / max) | easy disk rate | hard disk rate | identity Pearson | hard verdicts (A/I/Ind) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 8 / 57 / 133 | 0.1657 | 0.9755 | 0.8270 | 0 / 2 / 0 |
| 1 | 8 / 58 / 135 | 0.1659 | 0.9765 | 0.8281 | 0 / 1 / 1 |
| 100 | 8 / 95 / 113 | 0.1827 | 1.0737 | 0.9939 | 0 / 0 / 2 |

Source: `benchmarks/results/porous_fisher_conditioning.json`, `reaction_effect.by_reaction` and
`records`.

At `r = 0`, hard states are all `investigate`; at `r = 1`, they split evenly between
`investigate` and `indeterminate`; at `r = 100`, they are all `indeterminate`. The hard-state
median disk rate at `r = 100` is 1.0737, but it is not a convergence rate: the enclosing disk
contains the origin, so the field-of-values rate is formally void and the procedure correctly
routes the state to `indeterminate`.

## Work--precision

The comparison uses the same nonlinear node-centred spatial discretization for both methods and
reports max-norm error on the common smooth interior. The recorded matching point is the best
configuration from each method that reaches the stated target; no interpolation is used.

| problem | target | BE-JFNK error / median runtime / nonlinear RHS evaluations | Diffrax error / median runtime / NFE estimate | winner and speedup |
| --- | ---: | ---: | ---: | --- |
| PME, m=2 | 1e-03 | 2.7336e-04 / 5.7019e-02 s / 104 | 1.2120e-04 / 3.7386e-04 s / 13 | Diffrax, 152.52x |
| Porous--Fisher, r=1 | 1e-03 | 5.3739e-04 / 1.9379e-02 s / 34 | 7.5212e-05 / 3.9563e-04 s / 13 | Diffrax, 48.98x |

Source: `benchmarks/results/work_precision_fixedstep_vs_adaptive_diffrax.json`,
`problems.*.matched_accuracy_crossovers`.

The stored `1e-05`, `1e-06`, and `1e-08` crossover entries are unreached by one or both methods.
At the matched `1e-03` target, adaptive Diffrax Tsit5/PID is 49--153x faster. This is not spun as
a solver win: fixed-step backward Euler is O(dt)-limited and is not expected to beat adaptive,
higher-order integration on these smooth nonlinear fronts. The Stage-2 contribution is the
diagnostic that flags hard states, not a faster time integrator.

## Scope and caveats

- Experimental staging only; no Stage-2 module is public API.
- The studies are one-dimensional with homogeneous Dirichlet boundaries and a node-centred,
  DST-consistent diffusion operator.
- The preconditioner freezes a diffusion coefficient; the Porous--Fisher reaction term is not
  preconditioned.
- The recorded adjoint identity errors range from 0 to 5.46e-12 for PME and from 0 to 7.65e-14
  for Porous--Fisher, both below the 1e-08 gate.
- The work--precision comparison changes only the time integrator and nonlinear solver. It is not
  a reproduction of a linear FFT--Crank--Nicolson comparison.

Sources: regenerated output from `benchmarks/pme_breakdown.py`,
`benchmarks/results/porous_fisher_conditioning.json`, and
`benchmarks/results/work_precision_fixedstep_vs_adaptive_diffrax.json`.
