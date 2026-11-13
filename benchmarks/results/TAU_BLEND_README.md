# PME DST/tau-blend: mask analysis and certified preconditioner decisions

This report replaces the stale-flagged version of this file, whose numbers predated the
corrected diffusion coefficient. The earlier text remains in git history. Part I covers
the mask analysis and the certified decision procedure. Part II restates the
single-reference baseline study against the JSONs regenerated with the corrected
coefficient.

The lemmas cited below are stated and proved in `TAU_BLEND_MASK_LEMMAS.md` (G. Pavlov).
`TAU_BLEND_MASK_CLAIMS.md` (G. Pavlov) maps the numbers of the earlier mask study to
their scripts. `TAU_BLEND_MASK_FACTS.md` prints every number in Part I with its source
file and JSON pointer, and is regenerated from the committed JSONs by
`benchmarks/make_tau_blend_mask_facts.py`.

# Part I. Mask analysis and certified decisions

## 1. Scope and limits

- **Problem.** The one-dimensional, node-centred porous-medium equation with homogeneous
  Dirichlet boundaries on `[-4, 4]` and the regularized coefficient
  `D = m (u^2 + eps^2)^((m-1)/2)`, `eps = 1e-5`. One backward-Euler linearization per
  state, with `dt = 0.02`. The active-block study uses `N = 128, 512, 1024` and
  `m = 2, 4, 8`.
- **Time integration.** All trajectory evidence uses backward Euler. One bounded
  Crank--Nicolson replay is reported in Section 8, and it is not a general claim about
  Crank--Nicolson.
- **Not covered.** IMEX--Strang and ETDRK4 avoid Newton solves and are not affected by
  this work. There is no multigrid baseline, so this report makes no performance claim
  against the solver state of the art. Part I reports iteration counts only, no wall
  times.
- **Counts.** All GMRES counts in Sections 3 to 7 are unrestarted GMRES stopped on the
  true relative residual `||b - J x_k|| <= 1e-8 ||b||`, on the regularized operator
  unless stated otherwise. Section 8 compares them with the preconditioned-residual
  convention. Exact counts can differ by one or a few iterations between hosts (Section
  9).

Source: `tau_blend_mask_analysis.json`, `/provenance/config`;
`tau_blend_active_block.json`, `/configuration` [F1, F6].

### Preconditioners compared

All have the form `M = Q + sum_w H(d_w) W_w` of Pavlov's lemma setting, with
`H(d) = (I + dt d T)^-1` the DST-I Helmholtz inverse, `Q` the projector onto the masked
set `Z`, and diagonal input weights `W_w` that vanish on `Z`.

| Name | Mask | Helmholtz references |
|---|---|---|
| `repository_blend_B3` | repository rule `D <= max(1e-14, 1e-6 max(D))` | 3 active-support quantiles with Gaussian log-coefficient weights |
| `cfl_reference_blend_B3` | CFL mask `chi_j = 4 dt d_j / h^2 <= eta`, `eta = 0.3` at `m = 2` and `0.1` otherwise | 3 quantiles of the CFL-active coefficients with the same weights |
| `one_inverse_cfl_mask` | the same CFL mask | one inverse at the active maximum |

At `m = 4` and `m = 8` the repository blend and the CFL blend select the same nodes and
references in the committed cross-checks, so the study has 21 unique cases: the two CFL
methods at all nine `(N, m)` pairs, and the repository blend at `m = 2`.

Source: `tau_blend_mask_analysis.json`, `/states/*/repository_checks`;
`tau_blend_active_block.json`, `/cases/*` [F3, F6].

## 2. Why a scalar reference leaves a tail (Lemma 6)

Lemma 6 bounds the number of small eigenvalues of `H(d) J` from below, for any scalar
reference `d`, in terms of the modal CFL numbers of the region where the coefficient
sits at its floor:

`#{lambda(H(d) J) < a} >= #{beta_k < a} - s`, with
`beta_k = (1 + dt f mu_k) / (1 + dt d mu_k)`.

At level `a = 0.1` the bound holds on all 18 scalar-reference records, with a minimum
margin of 16 eigenvalues. The bound accounts for 90.6% to 96.6% of the measured tail
(bound/measured from 0.90625 to 0.96615385). The tail is therefore structural: it is
forced by the floor region and the CFL numbers, not by a poor choice of `d`.

| N | m | reference | lower bound | measured below 0.1 |
|---:|---:|---|---:|---:|
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

Lemma 5 gives every node the floor CFL number `chi_floor = 4 dt m eps^(m-1) / h^2`. With
the corrected coefficient, `D >= 2e-5` at `m = 2`, which lies above the repository
threshold `D <= max(1e-14, 1e-6 max(D))`, so the repository mask selects **no** node at
`m = 2`. The CFL mask selects 334 and 668 nodes at `N = 512` and `N = 1024`, exactly the
floor nodes (claims map row D17).

| N | method | masked nodes | chi_floor | max Re sigma(Y) | GMRES |
|---:|---|---:|---:|---:|---:|
| 512 | `repository_blend_B3` | 0 | 0.006579225 | 38.190625 | 57 |
| 512 | `one_inverse_cfl_mask` | 334 | 0.006579225 | 1 | 40 |
| 512 | `cfl_reference_blend_B3` | 334 | 0.006579225 | 2.0195094 | 15 |
| 1024 | `repository_blend_B3` | 0 | 0.026265625 | 146.88815 | 118 |
| 1024 | `one_inverse_cfl_mask` | 668 | 0.026265625 | 1 | 58 |
| 1024 | `cfl_reference_blend_B3` | 668 | 0.026265625 | 2.7989895 | 19 |

Source: `tau_blend_active_block.json`, `/cases/*`; `tau_blend_mask_analysis.json`,
`/states/*` [F3].

Without a mask, the repository blend's active-block spectrum spreads to maximum real
parts 38.2 and 146.9, and GMRES needs 57 and 118 iterations. The CFL blend restores a
compact spectrum (maximum real parts 2.02 and 2.80) and needs 15 and 19 iterations. At
`N = 128` the counts are 15 (repository) and 10 (CFL blend).

**Limit (Lemma 5).** At fixed `dt`, `chi_floor` grows as `h^-2`, so any fixed cutoff
`eta` eventually loses the mask under refinement. Pavlov's floor case shows this: at
`m = 2`, `N = 2048`, `dt = 0.02`, `chi_floor = 0.104960025` exceeds `eta = 0.1`, the
mask is empty, and the one-inverse solve does not reach tolerance within 300 iterations;
at `eta = 0.3` or `1` the mask holds 1334 nodes. A mesh-robust cutoff rule is not
proposed here (claims map rows C10 to C12).

Source: `tau_blend_cfl_fidelity.json`, `/floor/*` [F13].

## 4. Exact deflation and its regularized perturbation (Lemmas 1, 2 and 5)

In the node order `(Z, A)`, write `M J = [[I + E, X], [F, Y]]`. For `J_0`, which sets
the coefficient to zero on `Z`, Lemma 1 gives `E = F = 0` exactly; the measurements
confirm `E_0 = F_0 = 0` in all 21 cases. The active block `Y` is byte-identical between
`J_0` and the regularized `J` in all 21 cases.

For the regularized `J`, the identity block deviates by the floor CFL term. The measured
maximum deviation equals `2 dt d_floor / h^2` to `7.3e-17` at `N = 512, m = 2`, and to
`4.3e-18` at `N = 512, m = 4`. It scales with `(N + 1)^2`: at `m = 2` it is
`2.080125e-4`, `3.2896125e-3` and `1.3132813e-2` at `N = 128, 512, 1024`, and the ratio
between the last two is 3.992, equal to `((1024 + 1)/(512 + 1))^2 = 3.992`. At `m = 8`
the predicted deviation is of order `1.3e-32`, below the double-precision resolution of
the unit diagonal; the measured maximum is then about half the prediction, consistent
with an off-diagonal entry surviving while the diagonal increment rounds away (an
interpretation, not a separate measurement).

Lemma 2 bounds `||J - J_0||_p <= eta_Z`. Over 216 records there are no violations; the
worst excess is `3.6e-15` in the 1-norm and `8.2e-17` in the infinity norm, which is
round-off. In several records `||J - J_0||_1` equals `eta_Z` to rounding, so the 1-norm
bound is attained on these states.

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

**Proof.** `I - M J_0 = [[0, -X], [0, I - Y]]`, and `q(M J_0)` is block upper triangular
with diagonal blocks `q(1) I` and `q(Y)`. Multiplying gives the displayed form. GMRES
minimizes over all residual polynomials of degree `k + 1`, so its residual is at most
`||p(M J_0)|| ||r_0||` for this `p`.

The bound is a worst-case (ideal-GMRES) bound: it holds for every right-hand side and
can be pessimistic for a particular one. It is exact for `J_0` only; for the regularized
`J` this report certifies the full operator directly (Section 6). The block identity is
checked numerically for random `q` in `tests/test_tau_blend_active_block.py`.

| N | m | method | sigma_min(Y) | C |
|---:|---:|---|---:|---:|
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

For a closed contour `Gamma` that encloses `sigma(A)` and excludes the origin (the
boundary of the grid cells of a connected epsilon-pseudospectral component), let
`s_Gamma` be a proven lower bound for `sigma_min(zI - A)` on `Gamma` and `M_Gamma` a
proven upper bound for `|p|` on `Gamma`, for a polynomial `p` of degree `d` with
`p(0) = 1`. The Cauchy integral gives
`||p(A)^j|| <= L(Gamma) / (2 pi s_Gamma) * M_Gamma^j` [Trefethen--Embree]. The certified
rate is `rho = M_Gamma^(1/d)`, and the prefactor is `L(Gamma) / (2 pi s_Gamma)`.

`s_Gamma` comes from the grid samples of `sigma_min`, its 1-Lipschitz property along
each edge, and an SVD backward-error margin. `M_Gamma` is the sampled maximum of `|p|`
plus three allowances: a derivative enclosure times half the sample spacing, the
floating-point error of evaluating `p`, and, when a centred basis is used, the error of
translating the coefficients. Three changes on this branch tighten or correct the bound:

1. **Centred derivative enclosure** (61fa907). The enclosure is the minimum of the
   origin-centred `sum_j j |a_j| R^(j-1)` and the same sum in powers of `z - c`, with
   `c` the centre of the spectral bounding box. The sample points are unchanged, so the
   bound can only decrease.
2. **Search-domain fallback** (61fa907). If a level set touches the edge of the search
   box, the box is enlarged once to an outer rectangle of `W(A) + D(0, eps)`, which
   contains the epsilon-pseudospectrum because `sigma_min(zI - A) >= dist(z, W(A))`. The
   grid spacing does not increase.
3. **Floating-point evaluation of `p`** (a7d6dae). The previous bound covered the
   variation between samples but not the rounding error of the sampled values. The
   Horner allowance `gamma_{2n} sum_i |a_i| |z|^i` [Higham] is now added, evaluated in
   whichever of the origin and centred bases gives the tighter valid bound. The term
   grows with the degree and the size of the coefficients; a degree-40 regression test
   with large coefficients exercises a case where it dominates.

Monotonicity and validity are tested in c9c4a8e: the derivative bound is never below the
sampled `max |p'|`; the new contour bound never exceeds the old one;
`sigma_min(zI - A) >= dist(z, W(A))` holds on random non-normal matrices. The study
settings differ from the public defaults: grids 121, 241, 481; degrees 4, 8, 12, 16, 20,
24, 28, 32, 36, 40; 6 epsilon samples; target `1e-8`. The public defaults are grid 61,
domain-padding fraction 0.75, degrees 4, 8, 12, 16, 20, 6 epsilon samples and target
`1e-8` (source: `moljax/experimental/pseudospectral_criterion.py`,
`assess_dense_pseudospectral_criterion` signature).

Source: `tau_blend_active_block.json`, `/configuration` [F6].

### The merged decision

A case counts as certified if the active-block (`Y`) certificate certifies, or if the
merged dense disk policy returns `adequate`. Only `adequate` counts from the disk
policy; `provisional`, `investigate` and `indeterminate` do not.

| N | m | method | disk verdict | Y certificate | grid / degree | rho | prefactor | Lemma 1' bound | GMRES `J_0` | GMRES `J` |
|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|
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

Abstention reasons: "connectivity" means the pseudospectral component that encloses the
spectrum reaches the origin-exclusion threshold; "no polynomial" means the connected
window has no polynomial with a proven contour bound below one; "compute cap" means the
estimated cumulative cost of the next grid exceeds the 4-hour per-case cap.

**Counts.** The `Y` certificate certifies 15 of the 21 cases. The disk policy returns
`adequate` for 2, both already certified, so the merged procedure also certifies 15 of
21. By method: 6 of 9 for the one-inverse mask, 8 of 9 for the CFL blend, and 1 of 3 for
the repository blend at `m = 2`. The 6 abstentions are 3 for connectivity, 2 for no
polynomial and 1 for the compute cap.

**Ordering.** In every `(N, m)` group with at least one certified method, the method
with the fewest GMRES iterations is certified, and no certified method is slower than an
abstaining one. Where two or more methods certify, the smaller certified rate belongs to
the method with fewer iterations; at `N = 128, m = 2` the rates of the repository blend
and the one-inverse mask differ by 0.005, a tie under the 0.05 tie rule. At
`N = 1024, m = 4` no method certifies.

**Conservatism.** The Lemma 1' bound is `1 + d j*`, with `j*` the smallest integer for
which `C * prefactor * rho^(d j) <= 1e-8`. Against the measured `J_0` counts, it exceeds
them by a factor of 2.5 (`N = 128, m = 2`, CFL blend: 25 against 10) to 14.6
(`N = 1024, m = 8`, CFL blend: 553 against 38). The bounds are valid and conservative;
they separate the methods in the measured order.

### Direct certificates for the regularized operator

Lemma 1' is exact for `J_0` only. For the CFL blend at `m = 2`, the full regularized
operator `M J` is certified directly:

| N | grid | degree | rho | prefactor | GMRES `J` |
|---:|---:|---:|---:|---:|---:|
| 128 | 565 | 12 | 0.42515174 | 5.2538931 | 10 |
| 512 | 121 | 12 | 0.62776353 | 21.425118 | 15 |
| 1024 | 121 | 8 | 0.72943227 | 55.105304 | 19 |

Source: `tau_blend_active_block.json`, `/direct_regularized_full_MJ_certificates/*`
[F6].

At `N = 128` the domain fallback produced the selected grid 565, with the grid spacing
kept or reduced. The committed JSON does not record a distinct pre-fallback starting
grid.

## 7. Limits of the result

- **Mesh dependence.** The CFL blend's counts grow under refinement: 10, 15, 19 at
  `m = 2`; 17, 17, 67 at `m = 4`; and 11, 20, 38 at `m = 8`, for `N = 128, 512, 1024`.
  At `N = 1024, m = 4` nothing certifies.
- **Certified is not fast.** The one-inverse mask certifies at rates 0.912 to 0.957 at
  `N >= 512`, with 40 to 58 iterations; the CFL blend certifies at 0.615 to 0.780 at
  `N = 512`, with 15 to 20 iterations. The procedure decides whether a preconditioner is
  guaranteed to work and ranks the certified ones; it does not make a slow one fast.
- **Reference count.** For the CFL blend, three references give the fewest iterations at
  `N = 512` (17 at `m = 4`, 20 at `m = 8`). At `N = 1024` four references give 33 at
  `m = 4` and 33 at `m = 8`, against 67 and 38 for three, and five or six references are
  worse than four. The dependence is not monotone, and no rule for choosing the count is
  proposed.

| N | m | references | sigma_min(Y) | GMRES (true) | GMRES (preconditioned) | certificate |
|---:|---:|---:|---:|---:|---:|---|
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

- **Dimension and baselines.** One-dimensional only; no multigrid comparison; backward
  Euler only for trajectories.

## 8. Fidelity: what the counts and trajectories mean (Lemmas 3 and 4)

**Stopping convention (Lemma 4).** A test on the left-preconditioned residual bounds the
true residual only up to `kappa(M)`. At `N = 512`, `dt = 0.02`:

| m | scalar (S) | repository blend (B3) | mask, one inverse (Z1) | mask with frame (Zf) |
|---:|---:|---:|---:|---:|
| 2 | 1053.5444 | 28.931565 | 1121.5069 | 8914.9531 |
| 4 | 3423.3855 | 2108.3833 | 3735.1013 | 27737.091 |
| 8 | 6483.5385 | 2763.9588 | 7277.271 | 95755.909 |

`kappa_2(M)`; source: `tau_blend_lemma_checks.json`, `/lemma4/*` [F9]. The `Z1` and `Zf`
rows use `eta = 0.1`; at `m = 2` the values at `eta = 0.3` are identical.

In all 21 active-block cases the true-residual count exceeds the preconditioned-residual
count, by 1 to 15 iterations; for example 118 against 104 for the repository blend at
`N = 1024, m = 2`, and 47 against 37 for the one-inverse mask at `N = 512, m = 4`. The
preconditioned counts of the CFL blend at `m = 2` (14 and 17 at `N = 512, 1024`) equal
those of Pavlov's mask analysis (claims map row D20).

Source: `tau_blend_active_block.json`, `/cases/*/gmres_regularized` [F9].

**A posteriori trajectory bound (Lemma 3).** Backward Euler for the PME is nonexpansive
in the 1-norm, so `||v^n - u^n||_1 <= ||v^0 - u^0||_1 + sum_k ||rho^k||_1` holds for
every preconditioner and cutoff. It holds on all 216 trajectories, including the 40 that
contain failed Newton steps, with no violation and a minimum margin of `2.1e-12`.

Source: `tau_blend_lemma_checks.json`, `/lemma3` [F8].

**Crank--Nicolson replay.** One bounded replay at `N = 512`, `dt = 0.02`, 5 steps. The
CN active set is smaller than the BE one by 8 nodes at `m = 2` and 2 at `m = 4`, at
every step. Both integrators converge.

| m | BE Newton | BE GMRES | CN Newton | CN GMRES |
|---:|---:|---:|---:|---:|
| 2 | 38 | 1802 | 35 | 1210 |
| 4 | 41 | 2152 | 38 | 1471 |

Source: `tau_blend_cn_replay.json`, `/steps`, `/summaries` [F11]. This replay supports
no general statement about CN; all trajectory evidence elsewhere is backward Euler.

## 9. Reproducibility on a second host

Pavlov's delivery (`research/tau-blend-repro` at ad478fc) was re-run on a second host
with the same JAX version (0.11.2). Convergence outcomes are preserved except in one
row; exact counts move slightly.

- Mask analysis: 4 of 46 rows differ, each by one GMRES iteration.
- E3 trajectories: 48 of 96 rows differ in at least one Newton or linear total.
- One convergence-status change: `m = 4`, `N = 512`, `dt = 0.02`, repository blend, left
  preconditioning, forcing 0.1. The delivery converges 5 steps (69 Newton, 623 linear
  iterations); this host converges 4, with one line-search failure (58 Newton, 486
  linear).

Source: `tau_blend_reproducibility.json`, `/mask_analysis_subset`, `/drifted_E3_subset`
[F12].

## 10. Cost of the certificate, and a corrected claim

The provisional matrix-free runtime estimator (`rate_bound_available=false`) records
these cost ratios relative to one GMRES solve: `m = 2`: 13.9x, `m = 4`: 45.2x, `m = 8`:
41.6x. These are not dense-certificate costs. An earlier version of this report called
the range a "fundamental evidence floor". That claim is withdrawn: no result here
establishes a lower bound on the cost of certification, and an abstention is only the
failure of a sufficient test. The cost is a measured property of that implementation,
not a dense-certificate complexity bound.

Source: `pme_dst_tau_blend_pseudospectral_runtime_estimator.json`, `/records/*/cost`
[II.5].

On the active-block study, the estimated cost of the next grid exceeded the 4-hour
per-case cap once (`N = 1024, m = 2`, repository blend), which is reported as an
abstention.

# Part II. Single-reference baseline study (corrected coefficient)

This part restates the earlier single-reference study against its JSONs, regenerated
with the corrected coefficient. Section II.6 lists what changed from the previous
version.

**Scope.** The same one-dimensional problem as Part I, at one wide-front backward-Euler
linearization of the initial state (and, in II.3, 4 later states), `m = 2, 4, 8`,
`N = 512, 1024`. The dense spectral table uses `N = 512` and the dense criterion uses
`N = 512`. The blend in this part is the repository blend with the repository mask, so
at `m = 2` it has no mask (Part I, Section 3); its `m = 2` spectrum is the one Part I
reports for `repository_blend_B3` at `N = 512`.

**Counts and timings.** The benchmark calls `_counted_gmres`, which claims convergence
only after the measured true relative residual `||b - A x|| / ||b||` reaches the
requested tolerance. Its stopping convention therefore matches Part I's true-residual
convention. The JSONs were generated at `39f40910839326d376f669d7d242d6ab9be56e08` (base
`v1.2.1`) and committed in the correction delivery; they record no host, device model or
library versions, so all timings in this part are indicative only.

Source: the Part II JSONs, `/provenance` [P1].

## II.1 Scalar references and the Lemma 6 boundary

| m | reference | d / floor | d / active max | kappa_2 | count below 0.1 | min abs(lambda) |
|---:|---|---:|---:|---:|---:|---:|
| 2 | identity | -- | -- | 1773.661 | 0 | 1.0000001 |
| 2 | mean | 37573.225 | 0.23253991 | 1056.6956 | 294 | 0.004055825 |
| 2 | bulk | 148299.38 | 0.91782175 | 1056.9678 | 320 | 0.0010306872 |
| 2 | floor | 1 | 6.1889792e-6 | 1763.1363 | 0 | 1 |
| 2 | const | 50000 | 0.30944896 | 1056.7537 | 300 | 0.0030508637 |
| 2 | geometric | 52.594462 | 0.00032550603 | 1368.0925 | 0 | 0.74782939 |
| 2 | harmonic | 1.5329185 | 9.4872006e-6 | 1757.5863 | 0 | 0.99652912 |
| 2 | oracle | 1 | 6.1889792e-6 | 1763.1363 | 0 | 1 |
| 2 | blend | -- | -- | 132.56989 | 0 | 0.7046761 |
| 4 | identity | -- | -- | 8296.1512 | 0 | 1 |
| 4 | mean | 1.7033408e+14 | 0.063387724 | 3535.5269 | 236 | 0.0044424078 |
| 4 | bulk | 2.5676733e+15 | 0.95552788 | 3536.4039 | 272 | 0.00029592747 |
| 4 | const | 2.5e+14 | 0.093034409 | 3535.7019 | 242 | 0.0030310647 |
| 4 | geometric | 1.4809793e+15 | 0.55112813 | 3536.2418 | 264 | 0.00051295796 |
| 4 | harmonic | 8.5261811e+14 | 0.31729129 | 3536.1046 | 258 | 0.00089066022 |
| 4 | oracle | 2.6871778e+15 | 1 | 3588.5801 | 272 | 0.00028277067 |
| 4 | blend | -- | -- | 54.777113 | 0 | 0.84821246 |
| 8 | identity | -- | -- | 18574.596 | 0 | 1 |
| 8 | mean | 3.3622679e+33 | 0.012819804 | 6899.3202 | 170 | 0.011177481 |
| 8 | bulk | 2.5452575e+35 | 0.97046704 | 6901.6747 | 224 | 0.00014930053 |
| 8 | const | 1.25e+34 | 0.047660552 | 6900.4961 | 192 | 0.0030313036 |
| 8 | geometric | 1.4202832e+35 | 0.54153184 | 6901.4776 | 216 | 0.00026752648 |
| 8 | harmonic | 6.6115579e+34 | 0.2520884 | 6901.2558 | 208 | 0.00057451918 |
| 8 | oracle | 2.622714e+35 | 1 | 7002.2834 | 224 | 0.00014489188 |
| 8 | blend | -- | -- | 51.443028 | 0 | 0.83255408 |

Source: `pme_dst_tau_blend_single_reference_baselines.json`, `/spectral_records/*` [P2].
At `m = 4` and `8` the floor reference (`4e-15`, `8e-35`) coincides with identity and is
omitted.

Lemma 6 explains every row. With `beta_k = (1 + dt f mu_k) / (1 + dt d mu_k)`, a
reference `d` far above the floor value `f` leaves the floor-region modes near `f/d`, a
tail of small eigenvalues; a reference near the floor gives `beta_k` near one and no
tail, but then it barely preconditions the active part.

- **m = 4 and m = 8.** Every scalar reference at the active scale (`d` from 0.012819804
  to 1 times the active maximum) keeps a tail: 236 to 272 eigenvalues below `0.1` at
  `m = 4`, and 170 to 224 at `m = 8`. Their condition numbers lie within 1.5005742% of
  each other, and the GMRES-optimal oracle is spectrally the worst: the largest
  `kappa_2` and the smallest `min |lambda|`.
- **m = 2.** The references split by distance from the floor. The 3 at 37573.225 to
  148299.38 times the floor keep 294 to 320 small eigenvalues. The 4 within 52.594462
  times the floor have none, but their condition numbers, 1368.0925 to 1763.1363, are
  close to identity's 1773.661. The geometric and harmonic means land near the floor
  because the corrected repository rule treats every node as active, so the floor nodes
  enter the means.
- **The blend** has no eigenvalue below `0.1` at any `m`, with `kappa_2` 132.56989,
  54.777113, 51.443028. At `m = 2` its larger condition number is that of an empty mask;
  Part I shows that the CFL mask repairs it.

## II.2 Linear-solve work at one linearization

GMRES iterations; `cap` means 400 iterations without convergence (reported as 401, the
inherited counter convention).

| N | m | tol | identity | mean | bulk | floor | const | geometric | harmonic | oracle | blend |
|---:|---:|---:|---|---|---|---|---|---|---|---|---:|
| 512 | 2 | 0.01 | 29 | 13 | 7 | 28 | 11 | 42 | 28 | 7 | 8 |
| 512 | 2 | 1e-5 | 111 | 111 | 101 | 96 | 109 | 112 | 111 | 96 | 28 |
| 512 | 2 | 1e-8 | 121 | 181 | 166 | 111 | 178 | 131 | 121 | 111 | 50 |
| 512 | 4 | 0.01 | 82 | 48 | 16 | 82 | 41 | 20 | 25 | 16 | 4 |
| 512 | 4 | 1e-5 | 151 | 185 | 143 | 151 | 175 | 149 | 153 | 142 | 11 |
| 512 | 4 | 1e-8 | 156 | 285 | 245 | 156 | 277 | 253 | 260 | 245 | 15 |
| 512 | 8 | 0.01 | 102 | 84 | 18 | 102 | 63 | 23 | 32 | 17 | 5 |
| 512 | 8 | 1e-5 | 190 | 253 | 170 | 190 | 227 | 182 | 194 | 170 | 13 |
| 512 | 8 | 1e-8 | 196 | 334 | 277 | 196 | 317 | 282 | 292 | 277 | 17 |
| 1024 | 2 | 0.01 | 50 | 12 | 7 | 48 | 11 | 99 | 49 | 7 | 18 |
| 1024 | 2 | 1e-5 | 213 | 188 | 165 | 177 | 183 | 213 | 210 | 164 | 62 |
| 1024 | 2 | 1e-8 | 253 | 353 | 321 | 228 | 344 | 297 | 244 | 228 | 105 |
| 1024 | 4 | 0.01 | 196 | 47 | 15 | 196 | 41 | 20 | 55 | 15 | 11 |
| 1024 | 4 | 1e-5 | 323 | 304 | 216 | 323 | 290 | 231 | 321 | 215 | 33 |
| 1024 | 4 | 1e-8 | 330 | cap | cap | 330 | cap | cap | cap | cap | 55 |
| 1024 | 8 | 0.01 | 170 | 119 | 19 | 170 | 67 | 23 | 40 | 18 | 8 |
| 1024 | 8 | 1e-5 | 386 | cap | 293 | 386 | 393 | 315 | 368 | 292 | 21 |
| 1024 | 8 | 1e-8 | cap | cap | cap | cap | cap | cap | cap | cap | 33 |

Median [IQR] seconds, with the unrecorded-environment caveat above:

| N | m | tol | identity | mean | bulk | floor | const | geometric | harmonic | oracle | blend |
|---:|---:|---:|---|---|---|---|---|---|---|---|---|
| 512 | 2 | 0.01 | 0.10132594 [0.017403931] | 0.095396729 [0.014263316] | 0.087377926 [0.014671128] | 0.11522334 [0.043575743] | 0.085698517 [0.0040937079] | 0.15898793 [0.11867331] | 0.1408555 [0.039906832] | 0.08280787 [0.0043434061] | 0.11854856 [0.025709503] |
| 512 | 2 | 1e-5 | 0.68522133 [0.16664332] | 0.70845633 [0.097772314] | 0.56672659 [0.15100866] | 0.55210125 [0.14552211] | 0.6861409 [0.087679542] | 0.73716489 [0.18109118] | 0.72243508 [0.13325866] | 0.58160118 [0.12226561] | 0.1765324 [0.039689487] |
| 512 | 2 | 1e-8 | 0.58197142 [0.087744969] | 1.4160055 [0.063296856] | 1.1728565 [0.15418544] | 0.57646483 [0.026693835] | 1.40928 [0.1569963] | 0.74675852 [0.073402172] | 0.68224203 [0.06923097] | 0.54075975 [0.095081697] | 0.23148801 [0.016896598] |
| 512 | 4 | 0.01 | 0.33870324 [0.03656309] | 0.17532577 [0.031977885] | 0.087095706 [0.005002493] | 0.36096145 [0.020054752] | 0.1627172 [0.012743085] | 0.10019781 [0.026585919] | 0.10908153 [0.0110457] | 0.088572462 [0.0043255539] | 0.10502802 [0.00582109] |
| 512 | 4 | 1e-5 | 1.027712 [0.11539538] | 1.6038514 [0.13167227] | 0.97819695 [0.060242431] | 1.0336724 [0.069405525] | 1.4219629 [0.11433043] | 1.1003219 [0.081686926] | 1.0994023 [0.064320354] | 0.97894558 [0.066313446] | 0.11659306 [0.003220528] |
| 512 | 4 | 1e-8 | 1.1544524 [0.073318303] | 3.884865 [0.20017932] | 2.7141054 [0.11935271] | 1.1470286 [0.10351339] | 3.445285 [0.19889315] | 2.8139011 [0.14253115] | 3.0855899 [0.27130407] | 2.8064512 [0.22622456] | 0.12672133 [0.009723886] |
| 512 | 8 | 0.01 | 0.52012315 [0.10557911] | 0.39746284 [0.043011146] | 0.10110094 [0.003741883] | 0.56267664 [0.092494774] | 0.28531677 [0.031062597] | 0.11266081 [0.008858735] | 0.12992439 [0.021582735] | 0.10067479 [0.0037962859] | 0.12567804 [0.023573305] |
| 512 | 8 | 1e-5 | 1.8004765 [0.08353349] | 3.0256741 [0.17948889] | 1.3829166 [0.091824694] | 1.7939963 [0.090106239] | 2.3115109 [0.072279431] | 1.5288917 [0.058376535] | 1.7015094 [0.11052372] | 1.3222088 [0.10264503] | 0.13315101 [0.0035802349] |
| 512 | 8 | 1e-8 | 1.7202972 [0.081947934] | 4.9229426 [0.23044587] | 3.4068548 [0.23595175] | 1.6710508 [0.102614] | 4.6753775 [0.15675381] | 3.4626601 [0.15770534] | 3.683741 [0.23963729] | 5.4720419 [0.18077466] | 0.14976526 [0.004024702] |
| 1024 | 2 | 0.01 | 0.16023289 [0.006531737] | 0.091888352 [0.004047751] | 0.096301899 [0.022417241] | 0.19416995 [0.015062223] | 0.093156089 [0.0029950551] | 0.4742628 [0.10469807] | 0.19688651 [0.0093091379] | 0.085959019 [0.00091315102] | 0.14081486 [0.02366129] |
| 1024 | 2 | 1e-5 | 1.4885382 [0.061567497] | 1.357419 [0.069886274] | 1.0156231 [0.040052768] | 1.1248108 [0.031381986] | 1.365277 [0.075038789] | 2.0169919 [0.23916973] | 2.0205977 [0.056597653] | 1.144732 [0.065708436] | 0.24998773 [0.024388336] |
| 1024 | 2 | 1e-8 | 2.0873333 [0.12167476] | 4.3995622 [0.16208933] | 3.6197534 [0.078480449] | 1.9308502 [0.20253135] | 4.0444963 [0.079745532] | 3.1876063 [0.062590683] | 2.2287464 [0.048920912] | 1.924745 [0.087255295] | 0.46547898 [0.035378674] |
| 1024 | 4 | 0.01 | 1.2305533 [0.051316196] | 0.14780109 [0.007425895] | 0.090453567 [0.003247001] | 1.4785179 [0.005038875] | 0.13594167 [0.002124897] | 0.095069455 [0.026961944] | 0.17102925 [0.0078445339] | 0.089585185 [0.003972927] | 0.12138882 [0.005146566] |
| 1024 | 4 | 1e-5 | 3.2542124 [0.050310587] | 3.3041769 [0.035024395] | 1.7771906 [0.072976768] | 3.6270462 [0.077902852] | 3.1079898 [0.060226483] | 2.0773905 [0.085129602] | 3.6826261 [0.23274514] | 1.7328361 [0.031457959] | 0.16029007 [0.004906491] |
| 1024 | 4 | 1e-8 | 3.3166325 [0.10879123] | 5.3815439 [0.1533669] | 5.4608246 [0.1073439] | 3.7431456 [0.19982194] | 5.3254341 [0.16376072] | 5.4825062 [0.095226919] | 5.5371508 [0.10973787] | 5.4244503 [0.10869012] | 0.21870502 [0.023326903] |
| 1024 | 8 | 0.01 | 0.96416339 [0.051558246] | 0.52036537 [0.028207003] | 0.1017692 [0.004376568] | 1.0673766 [0.025207148] | 0.21996508 [0.02035422] | 0.10932517 [0.071305308] | 0.13862393 [0.003894677] | 0.099612728 [0.005238935] | 0.12592035 [0.004026093] |
| 1024 | 8 | 1e-5 | 4.585032 [0.049657349] | 5.5178823 [0.034335504] | 3.0490585 [0.016246985] | 4.9976861 [0.075437428] | 5.2784774 [0.25530652] | 3.5023911 [0.11983667] | 4.7525094 [0.10789742] | 3.0366353 [0.039597591] | 0.14081512 [0.01691104] |
| 1024 | 8 | 1e-8 | 4.7405763 [0.098961748] | 5.5555601 [0.23958248] | 5.3718649 [0.24787472] | 5.4642094 [0.24117503] | 5.8183401 [0.4343362] | 5.6987729 [0.47795977] | 5.706572 [0.43066981] | 5.7925592 [0.24592199] | 0.16555858 [0.006595258] |

Source: `pme_dst_tau_blend_single_reference_baselines.json`, `/work_precision_records/*`
[P3].

**Tight tolerance (`1e-8`).** The tail costs more than it saves. In 5 of the 6 cases,
every active-scale scalar needs more iterations than no preconditioner at all; at
`N = 1024, m = 8` every single reference caps. The blend needs 50, 15, 17 iterations at
`N = 512` and 105, 55, 33 at `N = 1024`. Against the converged oracle, its time is
2.3360162, 22.146636, 36.537458 times smaller at `N = 512` and 4.1349772 times smaller
at `N = 1024, m = 2`; at `N = 1024, m = 4` the oracle caps and the fastest converged
comparator, identity, takes 15.164867 times longer.

**Loose tolerance (`0.01`).** The blend is slower than the fastest converged scalar in
all 6 cases, by 20.589209% to 63.816267%. At `m = 4` and `8` it needs 4 to 11 iterations
against 15 to 18 for the fastest scalars, but applies 3 Helmholtz inverses per
iteration; at `m = 2` it also needs more iterations in the `N = 1024` case.

Source: `pme_dst_tau_blend_single_reference_baselines.json` [P4, P5].

## II.3 Later states (contrast sweep)

| m | step | degeneracy fraction | best single | blend | ratio |
|---:|---:|---:|---|---:|---:|
| 2 | 0 | 0 | floor (111) | 50 | 2.22 |
| 2 | 2 | 0 | bulk (131) | 54 | 2.4259259 |
| 2 | 5 | 0 | oracle (133) | 49 | 2.7142857 |
| 2 | 8 | 0 | bulk (135) | 48 | 2.8125 |
| 8 | 0 | 0.41796875 | identity (196) | 17 | 11.529412 |
| 8 | 2 | 0.40625 | identity (210) | 20 | 10.5 |
| 8 | 5 | 0.390625 | identity (216) | 26 | 8.3076923 |
| 8 | 8 | 0.37890625 | identity (219) | 22 | 9.9545455 |

GMRES iterations at `1e-8`, `N = 512`. "Best single" is the minimum converged iteration
count over the 8 non-blend comparators, with the selected method named. Source:
`pme_dst_tau_blend_single_reference_baselines.json`, `/contrast_records/*` [P6].

At `m = 8` the best single reference is identity or the floor in every state, and the
blend's advantage (11.529412, 10.5, 8.3076923, 9.9545455) narrows as the front fills in,
without vanishing. At `m = 2` the repository rule marks no node as degenerate, and the
advantage is 2.22 to 2.8125. The oracle sits at the floor in the initial `m = 2` state
and at the active maximum in the other 7 states. The coefficient-contrast ratio
`D95/D05` (about `158126.27` at `m = 2` and `10.228362` to `10.960587` at `m = 8`) does
not order these results.

## II.4 Batching audit

The production blend applies its 3 references in one batched Helmholtz call. On the GPU
the batched and sequential actions agree to `1.99345e-16` relative; the batched
application is 1.0040106 to 1.0693409 times faster, and the full solve 0.97993245 to
1.043602 times. The iteration counts agree in 11 of 12 solves; at `N = 1024, m = 2`,
`1e-8`, the batched solve takes 105 iterations and the sequential one 104, a difference
at the stopping threshold that we attribute to round-off (an interpretation). On the CPU
the batched application is 0.67044136 to 0.97270426 times as fast. Batching therefore
neither helps nor hurts the end-to-end solve at these sizes.

Source: `pme_dst_tau_blend_batching_audit_{gpu,cpu}.json` [P7].

## II.5 Dense criterion and runtime estimator

The legacy dense criterion used the public defaults on code that predates the 3
certificate changes of Part I. It and the current rerun both use `N = 512`.

| m | certified | rate | predicted iterations | GMRES |
|---:|---|---:|---:|---:|
| 2 | no | -- | -- | 50 |
| 4 | yes | 0.79420796 | 104 | 15 |
| 8 | yes | 0.93724655 | 368 | 17 |

Source: `pme_dst_tau_blend_pseudospectral_criterion.json` [P8].

The current-code rerun preserves every verdict. Old rate/predicted count to current
rate/predicted count: `m = 2`: -- / -- to -- / --; `m = 4`: 0.79420796 / 104 to
0.79373709 / 104; `m = 8`: 0.93724655 / 368 to 0.93367218 / 352. Source:
`pme_dst_tau_blend_pseudospectral_criterion_current.json`, `/records/*`.

At `m = 2` the criterion abstains on the unmasked blend, as Part I's active-block
certificate does. At `m = 4` and `8` it certifies. The provisional sparse estimator's
costs are given in Part I, Section 10; none of its modes provides a rate bound. Its
non-certification of the frozen mean at `m = 8`, at 0.33219555 times the cost of a
solve, is the failure of a sufficient test, not evidence that GMRES is slow.

## II.6 What changed from the previous version

- At `m = 2` the regularized coefficient never vanishes (`D >= 2e-5`); the previous
  statement that it vanishes on a set of positive measure holds only for the
  unregularized equation.
- "Every effective scalar keeps a tail" now holds at `m = 4` and `8` only; at `m = 2`,
  references near the floor avoid the tail by not preconditioning.
- "The oracle is spectrally worst" holds at `m = 4` and `8` only; at `m = 2` the oracle
  selects the floor.
- The blend's `kappa_2` at `m = 2` is 132.56989, the empty-mask value.
- The tight-tolerance time ratios are 2.3360162 to 36.537458, not the previous range.
- At loose tolerance the blend is slower at every `m`, including `m = 8`.
- The batching speed-up is 1.0040106 to 1.0693409 on the GPU, not the previous range,
  and the iteration counts agree in 11 of 12 solves, not all.
- The runtime estimator's cost is not a floor (Part I, Section 10).

The previous values are in git history.

## References

1. G. Pavlov, "moljax: GPU-accelerated method of lines for stiff reaction-diffusion PDEs
   with FFT preconditioning," *Computer Physics Communications* 326 (2026) 110205. DOI
   `10.1016/j.cpc.2026.110205`.
2. L. N. Trefethen and M. Embree, *Spectra and Pseudospectra* (Princeton, 2005). DOI
   `10.1515/9780691213101`.
3. M. Embree, "How Descriptive are GMRES Convergence Bounds?", arXiv:2209.01231. DOI
   `10.48550/arXiv.2209.01231`.
4. N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed. (SIAM, 2002).
   DOI `10.1137/1.9780898718027`.
