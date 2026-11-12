# PME DST/tau-blend: mask analysis and certified preconditioner decisions

This report replaces the stale-flagged version of this file, whose numbers
predated the corrected diffusion coefficient. The earlier text remains in git
history. Part I covers the mask analysis and the certified decision procedure.
Part II, the single-reference baseline study, is being restated against the
JSONs regenerated with the corrected coefficient.

The lemmas cited below are stated and proved in `TAU_BLEND_MASK_LEMMAS.md`
(G. Pavlov). `TAU_BLEND_MASK_CLAIMS.md` (G. Pavlov) maps the numbers of the
earlier mask study to their scripts. `TAU_BLEND_MASK_FACTS.md` prints every
number in Part I with its source file and JSON pointer, and is regenerated from
the committed JSONs by `benchmarks/make_tau_blend_mask_facts.py`.

# Part I. Mask analysis and certified decisions

## 1. Scope and limits

- **Problem.** The one-dimensional, node-centred porous-medium equation with
  homogeneous Dirichlet boundaries on `[-4, 4]` and the
  regularized coefficient `D = m (u^2 + eps^2)^((m-1)/2)`,
  `eps = 1e-5`. One backward-Euler linearization per state, with
  `dt = 0.02`. The active-block study uses `N = 128, 512, 1024` and
  `m = 2, 4, 8`.
- **Time integration.** All trajectory evidence uses backward Euler. One bounded
  Crank--Nicolson replay is reported in Section 8, and it is not a general claim
  about Crank--Nicolson.
- **Not covered.** IMEX--Strang and ETDRK4 avoid Newton solves and are not affected
  by this work. There is no multigrid baseline, so this report makes no performance
  claim against the solver state of the art. Part I reports iteration counts only,
  no wall times.
- **Counts.** All GMRES counts in Sections 3 to 7 are unrestarted GMRES stopped on
  the true relative residual `||b - J x_k|| <= 1e-8 ||b||`, on the regularized
  operator unless stated otherwise. Section 8 compares them with the
  preconditioned-residual convention. Exact counts can differ by one or a few
  iterations between hosts (Section 9).

Source: `tau_blend_mask_analysis.json`, `/provenance/config`; `tau_blend_active_block.json`,
`/configuration` [F1, F6].

### Preconditioners compared

All have the form `M = Q + sum_w H(d_w) W_w` of Pavlov's lemma setting, with
`H(d) = (I + dt d T)^-1` the DST-I Helmholtz inverse, `Q` the projector onto the
masked set `Z`, and diagonal input weights `W_w` that vanish on `Z`.

| Name | Mask | Helmholtz references |
|---|---|---|
| `repository_blend_B3` | repository rule `D <= max(1e-14, 1e-6 max(D))` | 3 active-support quantiles with Gaussian log-coefficient weights |
| `cfl_reference_blend_B3` | CFL mask `chi_j = 4 dt d_j / h^2 <= eta`, `eta = 0.3` at `m = 2` and `0.1` otherwise | 3 quantiles of the CFL-active coefficients with the same weights |
| `one_inverse_cfl_mask` | the same CFL mask | one inverse at the active maximum |

At `m = 4` and `m = 8` the repository blend and the CFL blend select the same
nodes and references in the committed cross-checks, so the study has 21
unique cases: the two CFL methods at all nine `(N, m)` pairs, and the repository
blend at `m = 2`.

Source: `tau_blend_mask_analysis.json`, `/states/*/repository_checks`;
`tau_blend_active_block.json`, `/cases/*` [F3, F6].

## 2. Why a scalar reference leaves a tail (Lemma 6)

Lemma 6 bounds the number of small eigenvalues of `H(d) J` from below, for any
scalar reference `d`, in terms of the modal CFL numbers of the region where the
coefficient sits at its floor:

`#{lambda(H(d) J) < a} >= #{beta_k < a} - s`, with
`beta_k = (1 + dt f mu_k) / (1 + dt d mu_k)`.

At level `a = 0.1` the bound holds on all 18 scalar-reference records,
with a minimum margin of 16 eigenvalues. The bound accounts for
90.6% to 96.6% of the measured
tail (bound/measured from 0.90625 to
0.96615385). The tail is therefore structural: it is forced by
the floor region and the CFL numbers, not by a poor choice of `d`.

| N | m | reference | lower bound | measured below 0.1 |
|---:|---|---|---|---|
| 512 | 2 | max | 304 | 322 |
| 512 | 2 | median | 300 | 318 |
| 512 | 2 | geometric mean | 294 | 312 |
| 512 | 4 | max | 254 | 272 |
| 512 | 4 | median | 251 | 268 |
| 512 | 4 | geometric mean | 248 | 264 |
| 512 | 8 | max | 203 | 224 |
| 512 | 8 | median | 201 | 220 |
| 512 | 8 | geometric mean | 198 | 216 |
| 1024 | 2 | max | 638 | 666 |
| 1024 | 2 | median | 634 | 658 |
| 1024 | 2 | geometric mean | 628 | 650 |
| 1024 | 4 | max | 522 | 554 |
| 1024 | 4 | median | 519 | 548 |
| 1024 | 4 | geometric mean | 516 | 540 |
| 1024 | 8 | max | 417 | 454 |
| 1024 | 8 | median | 415 | 446 |
| 1024 | 8 | geometric mean | 412 | 438 |

Source: `tau_blend_lemma_checks.json`, `/lemma6/*` [F2].

## 3. The mask at m = 2 under the corrected coefficient

Lemma 5 gives every node the floor CFL number
`chi_floor = 4 dt m eps^(m-1) / h^2`. With the corrected coefficient,
`D >= 2e-5`
at `m = 2`, which lies above the repository threshold `D <= max(1e-14, 1e-6 max(D))`, so the
repository mask selects **no** node at `m = 2`. The CFL mask selects
334 and
668 nodes at `N = 512`
and `N = 1024`, exactly the floor nodes.

| N | method | masked nodes | chi_floor | max Re sigma(Y) | GMRES |
|---:|---|---|---|---|---|
| 512 | `repository_blend_B3` | 0 | 0.006579225 | 38.190625 | 57 |
| 512 | `one_inverse_cfl_mask` | 334 | 0.006579225 | 1 | 40 |
| 512 | `cfl_reference_blend_B3` | 334 | 0.006579225 | 2.0195094 | 15 |
| 1024 | `repository_blend_B3` | 0 | 0.026265625 | 146.88815 | 118 |
| 1024 | `one_inverse_cfl_mask` | 668 | 0.026265625 | 1 | 58 |
| 1024 | `cfl_reference_blend_B3` | 668 | 0.026265625 | 2.7989895 | 19 |

Source: `tau_blend_active_block.json`, `/cases/*`;
`tau_blend_mask_analysis.json`, `/states/*` [F3].

Without a mask, the repository blend's active-block spectrum spreads to maximum
real parts 38.2
and 146.9,
and GMRES needs 57
and 118
iterations. The CFL blend restores a compact spectrum (maximum real parts
2.02
and 2.80)
and needs 15
and 19
iterations. At `N = 128` the counts are
15
(repository) and
10
(CFL blend).

**Limit (Lemma 5).** At fixed `dt`, `chi_floor` grows as `h^-2`, so any fixed
cutoff `eta` eventually loses the mask under refinement. Pavlov's floor case
shows this: at `m = 2`, `N = 2048`,
`dt = 0.02`, `chi_floor = 0.104960025`
exceeds `eta = 0.1`, the mask is empty, and the
one-inverse solve does not reach tolerance within 300
iterations; at `eta = 0.3` or
`1` the mask holds 1334 nodes.
A mesh-robust cutoff rule is not proposed here.

Source: `tau_blend_cfl_fidelity.json`, `/floor/*` [F13].

## 4. Exact deflation and its regularized perturbation (Lemmas 1, 2 and 5)

In the node order `(Z, A)`, write `M J = [[I + E, X], [F, Y]]`. For `J_0`,
which sets the coefficient to zero on `Z`, Lemma 1 gives `E = F = 0` exactly;
the measurements confirm `E_0 = F_0 = 0` in all 21 cases. The active
block `Y` is byte-identical between `J_0` and the regularized `J` in all
21 cases.

For the regularized `J`, the identity block deviates by the floor CFL term. The
measured maximum deviation equals `2 dt d_floor / h^2` to
`7.3e-17` at `N = 512,
m = 2`, and to `4.3e-18` at
`N = 512, m = 4`. It scales with `(N + 1)^2`: at `m = 2`
it is
`2.080125e-4`, `3.2896125e-3` and
`1.3132813e-2` at `N = 128, 512, 1024`,
and the ratio between the last two is 3.992, equal to
`((1024 + 1)/(512 + 1))^2 = 3.992`. At
`m = 8` the predicted deviation is `1e-32`,
below the double-precision resolution of the unit diagonal; the measured maximum
is then about half the prediction, consistent with an off-diagonal entry surviving
while the diagonal increment rounds away (an interpretation, not a separate
measurement).

Lemma 2 bounds `||J - J_0||_p <= eta_Z`. Over 216 records there
are no violations; the worst excess is `3.6e-15` in the 1-norm and
`8.2e-17` in the infinity norm, which is round-off. In several
records `||J - J_0||_1` equals `eta_Z` to rounding, so the 1-norm bound is attained
on these states.

Source: `tau_blend_active_block.json`, `/cases/*/J0_blocks`,
`/cases/*/regularized_blocks`; `tau_blend_lemma_checks.json`, `/lemma2`,
`/identity_block_deviation` [F4, F10].

## 5. From the active block to a GMRES bound (Lemma 1', ours)

**Statement.** Let `M J_0 = [[I, X], [0, Y]]` and let `q` be any polynomial with
`q(0) = 1`. Then `p(z) = (1 - z) q(z)` satisfies `p(0) = 1` and

`p(M J_0) = [[0, -X q(Y)], [0, (I - Y) q(Y)]]`,

so for every right-hand side, GMRES on `M J_0` satisfies

`||r_{k+1}|| / ||r_0|| <= C min_{q(0)=1, deg q <= k} ||q(Y)||_2`, with
`C = ||[X; I - Y]||_2`.

**Proof.** `I - M J_0 = [[0, -X], [0, I - Y]]`, and `q(M J_0)` is block upper
triangular with diagonal blocks `q(1) I` and `q(Y)`. Multiplying gives the
displayed form. GMRES minimizes over all residual polynomials of degree `k + 1`,
so its residual is at most `||p(M J_0)|| ||r_0||` for this `p`.

The bound is a worst-case (ideal-GMRES) bound: it holds for every right-hand side
and can be pessimistic for a particular one. It is exact for `J_0` only; for the
regularized `J` this report certifies the full operator directly (Section 6). The
block identity is checked numerically for random `q` in
`tests/test_tau_blend_active_block.py`.

| N | m | method | sigma_min(Y) | C |
|---:|---|---|---|---|
| 128 | 2 | `one_inverse_cfl_mask` | 0.16473036 | 1.413918 |
| 128 | 2 | `repository_blend_B3` | 0.4569388 | 2.3281802 |
| 128 | 2 | `cfl_reference_blend_B3` | 0.81876371 | 0.96643866 |
| 128 | 4 | `one_inverse_cfl_mask` | 0.03346387 | 0.96936701 |
| 128 | 4 | `cfl_reference_blend_B3` | 0.35099138 | 2.8029638 |
| 128 | 8 | `one_inverse_cfl_mask` | 0.08703789 | 5.309951 |
| 128 | 8 | `cfl_reference_blend_B3` | 0.79943021 | 4.5974814 |
| 512 | 2 | `one_inverse_cfl_mask` | 0.033845096 | 4.6464368 |
| 512 | 2 | `repository_blend_B3` | 0.28807928 | 37.190639 |
| 512 | 2 | `cfl_reference_blend_B3` | 0.71781984 | 3.8700658 |
| 512 | 4 | `one_inverse_cfl_mask` | 0.023444946 | 12.628915 |
| 512 | 4 | `cfl_reference_blend_B3` | 0.73753625 | 11.402013 |
| 512 | 8 | `one_inverse_cfl_mask` | 0.014298407 | 10.322216 |
| 512 | 8 | `cfl_reference_blend_B3` | 0.70463794 | 9.1562358 |
| 1024 | 2 | `one_inverse_cfl_mask` | 0.016596978 | 10.706432 |
| 1024 | 2 | `repository_blend_B3` | 0.20625577 | 145.88816 |
| 1024 | 2 | `cfl_reference_blend_B3` | 0.68563241 | 9.5460263 |
| 1024 | 4 | `one_inverse_cfl_mask` | 0.0035150634 | 1.2116194 |
| 1024 | 4 | `cfl_reference_blend_B3` | 0.17762823 | 33.753656 |
| 1024 | 8 | `one_inverse_cfl_mask` | 0.0049181136 | 5.0481582 |
| 1024 | 8 | `cfl_reference_blend_B3` | 0.46277392 | 12.395966 |

Source: `tau_blend_active_block.json`, `/cases/*` [F5].

When the mask is empty (the repository blend at `m = 2`), `Y = M J` and
`C = ||I - M J||_2`.

## 6. Decision procedure: disk test plus pseudospectral certificate

### The certificate

For a closed contour `Gamma` that encloses `sigma(A)` and excludes the origin
(the boundary of the grid cells of a connected epsilon-pseudospectral component),
let `s_Gamma` be a proven lower bound for `sigma_min(zI - A)` on `Gamma` and
`M_Gamma` a proven upper bound for `|p|` on `Gamma`, for a polynomial `p` of
degree `d` with `p(0) = 1`. The Cauchy integral gives
`||p(A)^j|| <= L(Gamma) / (2 pi s_Gamma) * M_Gamma^j` [Trefethen--Embree]. The
certified rate is `rho = M_Gamma^(1/d)`, and the prefactor is
`L(Gamma) / (2 pi s_Gamma)`.

`s_Gamma` comes from the grid samples of `sigma_min`, its 1-Lipschitz property
along each edge, and an SVD backward-error margin. `M_Gamma` is the sampled
maximum of `|p|` plus three allowances: a derivative enclosure times half the
sample spacing, the floating-point error of evaluating `p`, and, when a centred
basis is used, the error of translating the coefficients. Three changes on this
branch tighten or correct the bound:

1. **Centred derivative enclosure** (61fa907).
   The enclosure is the minimum of the origin-centred
   `sum_j j |a_j| R^(j-1)` and the same sum in powers of `z - c`, with `c` the
   centre of the spectral bounding box. The sample points are unchanged, so the
   bound can only decrease.
2. **Search-domain fallback** (61fa907).
   If a level set touches the edge of the search box, the box is enlarged once
   to an outer rectangle of `W(A) + D(0, eps)`, which contains the
   epsilon-pseudospectrum because `sigma_min(zI - A) >= dist(z, W(A))`. The grid
   spacing does not increase.
3. **Floating-point evaluation of `p`** (a7d6dae).
   The previous bound covered the variation between samples but not the rounding
   error of the sampled values. The Horner allowance
   `gamma_{2n} sum_i |a_i| |z|^i` [Higham] is now added, evaluated in whichever
   of the origin and centred bases gives the tighter valid bound. The term grows
   with the degree and the size of the coefficients; a degree-40 regression test
   with large coefficients exercises a case where it dominates.

Monotonicity and validity are tested in
c9c4a8e:
the derivative bound is never below the sampled `max |p'|`; the new contour bound
never exceeds the old one; `sigma_min(zI - A) >= dist(z, W(A))` holds on random
non-normal matrices. The study settings differ from the public defaults: grids
121, 241, 481; degrees 4, 8, 12, 16, 20, 24, 28, 32, 36, 40; 6 epsilon
samples; target `1e-8`. The public defaults are grid
61, domain-padding fraction 0.75, degrees
4, 8, 12, 16, 20, 6 epsilon samples and target
`1e-8`.

Source: `tau_blend_active_block.json`, `/configuration` [F6].

### The merged decision

A case counts as certified if the active-block (`Y`) certificate certifies, or if
the merged dense disk policy returns `adequate`. Only `adequate` counts from the
disk policy; `provisional`, `investigate` and `indeterminate` do not.

| N | m | method | disk verdict | Y certificate | grid / degree | rho | prefactor | Lemma 1' bound | GMRES `J_0` | GMRES `J` |
|---:|---|---|---|---|---|---|---|---|---|---|
| 128 | 2 | `one_inverse_cfl_mask` | indeterminate | certified | 121 / 4 | 0.6689927 | 7.8016507 | 53 | 17 | 17 |
| 128 | 2 | `repository_blend_B3` | adequate | certified | 121 / 8 | 0.66373192 | 14.415351 | 57 | 15 | 15 |
| 128 | 2 | `cfl_reference_blend_B3` | adequate | certified | 121 / 12 | 0.41734583 | 4.1834557 | 25 | 10 | 10 |
| 128 | 4 | `one_inverse_cfl_mask` | investigate | abstains: connectivity | 481 / -- | -- | -- | -- | 24 | 24 |
| 128 | 4 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 4 | 0.67363048 | 22.173878 | 61 | 17 | 17 |
| 128 | 8 | `one_inverse_cfl_mask` | indeterminate | certified | 121 / 8 | 0.75423384 | 16.893764 | 89 | 24 | 24 |
| 128 | 8 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 12 | 0.56008177 | 6.6467845 | 49 | 11 | 11 |
| 512 | 2 | `one_inverse_cfl_mask` | indeterminate | certified | 121 / 8 | 0.93941072 | 47.660579 | 385 | 40 | 40 |
| 512 | 2 | `repository_blend_B3` | indeterminate | abstains: no polynomial | 481 / -- | -- | -- | -- | 57 | 57 |
| 512 | 2 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 4 | 0.6146668 | 9.6238488 | 49 | 15 | 15 |
| 512 | 4 | `one_inverse_cfl_mask` | indeterminate | certified | 241 / 8 | 0.91228692 | 66.783082 | 281 | 47 | 47 |
| 512 | 4 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 8 | 0.67927653 | 10.191672 | 65 | 17 | 17 |
| 512 | 8 | `one_inverse_cfl_mask` | indeterminate | certified | 481 / 8 | 0.94249057 | 68.797227 | 425 | 55 | 55 |
| 512 | 8 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 16 | 0.78009837 | 12.768788 | 97 | 20 | 20 |
| 1024 | 2 | `one_inverse_cfl_mask` | indeterminate | certified | 241 / 8 | 0.95708509 | 105.1185 | 585 | 57 | 58 |
| 1024 | 2 | `repository_blend_B3` | indeterminate | abstains: compute cap | 241 / -- | -- | -- | -- | 118 | 118 |
| 1024 | 2 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 8 | 0.68121869 | 12.266603 | 65 | 19 | 19 |
| 1024 | 4 | `one_inverse_cfl_mask` | indeterminate | abstains: connectivity | 481 / -- | -- | -- | -- | 78 | 78 |
| 1024 | 4 | `cfl_reference_blend_B3` | indeterminate | abstains: no polynomial | 481 / -- | -- | -- | -- | 67 | 67 |
| 1024 | 8 | `one_inverse_cfl_mask` | indeterminate | abstains: connectivity | 481 / -- | -- | -- | -- | 84 | 84 |
| 1024 | 8 | `cfl_reference_blend_B3` | indeterminate | certified | 121 / 8 | 0.95518159 | 77.995915 | 553 | 38 | 38 |

Source: `tau_blend_active_block.json`, `/cases/*`, `/decision_summary` [F6].

Abstention reasons: "connectivity" means the pseudospectral component that encloses
the spectrum reaches the origin-exclusion threshold; "no polynomial" means the
connected window has no polynomial with a proven contour bound below one; "compute
cap" means the estimated cumulative cost of the next grid exceeds the
4-hour per-case cap.

**Counts.** The `Y` certificate certifies
15 of the 21 cases. The
disk policy returns `adequate` for 2,
both already certified, so the merged procedure also certifies
15 of 21. By method:
6 of 9 for the one-inverse mask,
8 of 9 for the CFL blend, and
1 of 3 for the repository blend at `m = 2`.
The 6 abstentions are 3 for connectivity,
2 for no polynomial and
1 for the compute cap.

**Ordering.** In every `(N, m)` group with at least one certified method, the
method with the fewest GMRES iterations is certified, and no certified method is
slower than an abstaining one. Where two or more methods certify, the smaller
certified rate belongs to the method with fewer iterations; at `N = 128, m = 2`
the rates of the repository blend and the one-inverse mask differ by
0.005,
a tie under the 0.05 tie rule. At `N = 1024,
m = 4` no method certifies.

**Conservatism.** The Lemma 1' bound is `1 + d j*`, with `j*` the smallest
integer for which `C * prefactor * rho^(d j) <= 1e-8`. Against the measured
`J_0` counts, it exceeds them by a factor of 2.5
(`N = 128, m = 2`, CFL blend:
25 against
10) to 14.6
(`N = 1024, m = 8`, CFL blend:
553 against
38). The bounds are valid and
conservative; they separate the methods in the measured order.

### Direct certificates for the regularized operator

Lemma 1' is exact for `J_0` only. For the CFL blend at `m = 2`, the full
regularized operator `M J` is certified directly:

| N | grid | degree | rho | prefactor | GMRES `J` |
|---:|---|---|---|---|---|
| 128 | 565 | 12 | 0.42515174 | 5.2538931 | 10 |
| 512 | 121 | 12 | 0.62776353 | 21.425118 | 15 |
| 1024 | 121 | 8 | 0.72943227 | 55.105304 | 19 |

Source: `tau_blend_active_block.json`, `/direct_regularized_full_MJ_certificates/*` [F6].

At `N = 128` the domain fallback produced the selected grid
565, with the grid spacing kept or
reduced. The committed JSON does not record a distinct pre-fallback starting grid.

## 7. Limits of the result

- **Mesh dependence.** The CFL blend's counts grow under refinement:
  10,
  15,
  19
  at `m = 2`;
  17,
  17,
  67
  at `m = 4`; and
  11,
  20,
  38
  at `m = 8`, for `N = 128, 512, 1024`. At `N = 1024, m = 4` nothing certifies.
- **Certified is not fast.** The one-inverse mask certifies at rates
  0.912
  to 0.957
  at `N >= 512`, with
  40
  to 58 iterations; the CFL blend certifies at
  0.615
  to 0.780
  at `N = 512`, with
  15
  to 20 iterations. The procedure decides whether a preconditioner is guaranteed to work and ranks the certified ones; it does not make a slow one fast.
- **Reference count.** For the CFL blend, three references give the fewest
  iterations at `N = 512` (17
  at `m = 4`, 20 at `m = 8`).
  At `N = 1024` four references give
  33 at `m = 4` and
  33 at `m = 8`, against
  67 and
  38 for three, and five or
  six references are worse than four. The dependence is not monotone, and no rule
  for choosing the count is proposed.

| N | m | references | sigma_min(Y) | GMRES (true) | GMRES (preconditioned) | certificate |
|---:|---|---|---|---|---|---|
| 512 | 4 | 3 | 0.73753625 | 17 | 15 | certified |
| 512 | 4 | 4 | 0.43273641 | 20 | 16 | certified |
| 512 | 4 | 5 | 0.29830289 | 23 | 17 | certified |
| 512 | 4 | 6 | 0.2319451 | 24 | 17 | certified |
| 512 | 8 | 3 | 0.70463794 | 20 | 17 | certified |
| 512 | 8 | 4 | 0.3835468 | 23 | 18 | certified |
| 512 | 8 | 5 | 0.23062782 | 28 | 19 | not run |
| 512 | 8 | 6 | 0.1645761 | 28 | 18 | not run |
| 1024 | 4 | 3 | 0.17762823 | 67 | 55 | not run |
| 1024 | 4 | 4 | 0.08497409 | 33 | 27 | not run |
| 1024 | 4 | 5 | 0.047392477 | 38 | 28 | not run |
| 1024 | 4 | 6 | 0.033333995 | 46 | 32 | not run |
| 1024 | 8 | 3 | 0.46277392 | 38 | 33 | not run |
| 1024 | 8 | 4 | 0.22945654 | 33 | 26 | not run |
| 1024 | 8 | 5 | 0.12636897 | 36 | 27 | not run |
| 1024 | 8 | 6 | 0.079412583 | 40 | 28 | not run |

Source: `tau_blend_reference_count.json`, `/rows/*` [F7].

- **Dimension and baselines.** One-dimensional only; no multigrid comparison;
  backward Euler only for trajectories.

## 8. Fidelity: what the counts and trajectories mean (Lemmas 3 and 4)

**Stopping convention (Lemma 4).** A test on the left-preconditioned residual
bounds the true residual only up to `kappa(M)`. At `N = 512`, `dt = 0.02`:

| m | scalar (S) | repository blend (B3) | mask, one inverse (Z1) | mask with frame (Zf) |
|---:|---|---|---|---|
| 2 | 1053.5444 | 28.931565 | 1121.5069 | 8914.9531 |
| 4 | 3423.3855 | 2108.3833 | 3735.1013 | 27737.091 |
| 8 | 6483.5385 | 2763.9588 | 7277.271 | 95755.909 |

`kappa_2(M)`; source: `tau_blend_lemma_checks.json`, `/lemma4/*` [F9]. The `Z1`
and `Zf` rows use `eta = 0.1`; at `m = 2` the values at
`eta = 0.3` are identical.

In all 21 active-block cases the true-residual count exceeds the
preconditioned-residual count, by 1 to
15 iterations; for example
118
against
104
for the repository blend at `N = 1024, m = 2`, and
47
against
37
for the one-inverse mask at `N = 512, m = 4`. The preconditioned counts of the
CFL blend at `m = 2` (
14
and
17
at `N = 512, 1024`) equal those of Pavlov's mask analysis.

Source: `tau_blend_active_block.json`, `/cases/*/gmres_regularized` [F9].

**A posteriori trajectory bound (Lemma 3).** Backward Euler for the PME is
nonexpansive in the 1-norm, so
`||v^n - u^n||_1 <= ||v^0 - u^0||_1 + sum_k ||rho^k||_1` holds for every
preconditioner and cutoff. It holds on all 216 trajectories,
including the 40 that contain failed
Newton steps, with no violation and a minimum margin of
`2.1e-12`.

Source: `tau_blend_lemma_checks.json`, `/lemma3` [F8].

**Crank--Nicolson replay.** One bounded replay at `N = 512`,
`dt = 0.02`, 5 steps. The CN
active set is smaller than the BE one by
8 nodes at `m = 2` and
2 at `m = 4`, at every step. Both integrators
converge.

| m | BE Newton | BE GMRES | CN Newton | CN GMRES |
|---:|---|---|---|---|
| 2 | 38 | 1802 | 35 | 1210 |
| 4 | 41 | 2152 | 38 | 1471 |

Source: `tau_blend_cn_replay.json`, `/steps`, `/summaries` [F11]. This replay
supports no general statement about CN; all trajectory evidence elsewhere is
backward Euler.

## 9. Reproducibility on a second host

Pavlov's delivery (`research/tau-blend-repro` at
ad478fc) was re-run on a second host with the
same JAX version (0.11.2). Convergence outcomes are preserved
except in one row; exact counts move slightly.

- Mask analysis: 4 of 46 rows
  differ, each by one GMRES iteration.
- E3 trajectories: 48 of 96 rows differ
  in at least one Newton or linear total.
- One convergence-status change: `m = 4`, `N = 512`,
  `dt = 0.02`, repository blend, left preconditioning,
  forcing 0.1. The delivery converges
  5 steps
  (69 Newton,
  623 linear iterations);
  this host converges 4, with one
  line-search failure (58 Newton,
  486 linear).

Source: `tau_blend_reproducibility.json`, `/mask_analysis_subset`,
`/drifted_E3_subset` [F12].

## 10. Cost of the certificate, and a corrected claim

The dense certificate is an offline tool. The provisional matrix-free runtime
estimator records these ratios to the corresponding solve: `m = 2`: 13.9x, `m = 4`: 45.2x, `m = 8`: 41.6x. An
earlier version of this report called the range a "fundamental evidence floor".
That claim is withdrawn: no result here establishes a lower bound on the cost of
certification, and an abstention is only the failure of a sufficient test. The
cost is a measured property of that implementation, not a dense-certificate
complexity bound.

Source: `pme_dst_tau_blend_pseudospectral_runtime_estimator.json`, `/records/*/cost`
[Part II runtime evidence].

On the active-block study, the estimated cost of the next grid exceeded the
4-hour per-case cap once (`N = 1024, m = 2`, repository
blend), which is reported as an abstention.

# Part II. Single-reference baseline study (being restated)

The previous version of this file reported a single-reference baseline study:
spectra and work--precision of scalar Helmholtz references against the blend,
contrast and degeneracy, a batching audit and the pseudospectral criterion. Its
JSONs were regenerated with the corrected coefficient, and several of its claims,
in particular those at `m = 2`, no longer hold. The claims are being audited
against the regenerated JSONs and will be restated here. Until then the previous
text is available in git history; the regenerated JSONs are in this directory.

## References

1. G. Pavlov, "moljax: GPU-accelerated method of lines for stiff
   reaction-diffusion PDEs with FFT preconditioning," *Computer Physics
   Communications* 326 (2026) 110205. DOI `10.1016/j.cpc.2026.110205`.
2. L. N. Trefethen and M. Embree, *Spectra and Pseudospectra* (Princeton,
   2005). DOI `10.1515/9780691213101`.
3. M. Embree, "How Descriptive are GMRES Convergence Bounds?",
   arXiv:2209.01231. DOI `10.48550/arXiv.2209.01231`.
4. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.
   (SIAM, 2002). DOI `10.1137/1.9780898718027`.
