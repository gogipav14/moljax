# Tau-blend mask claims: map from each number to its source

This table maps each number in two sections of reply 15 to a script, a case
and an output field. The two sections are "THE DOMINANT EFFECT IN THE BLEND"
and "CFL ANALYSIS OF THE MASK". The last section of this file also maps the
CFL numbers of the "DIRECTION" section, because the script for E1 also
gives them.

## Scripts and outputs

| Label | Script | Output |
|---|---|---|
| MA | `benchmarks/tau_blend_mask_analysis.py` | `benchmarks/results/tau_blend_mask_analysis.json` and `.csv`, list `cases`, list `states` |
| CF | `benchmarks/tau_blend_cfl_fidelity.py` | `benchmarks/results/tau_blend_cfl_fidelity.json` and `.csv`, lists `e1_e2`, `e2_best_cutoffs`, `e4`, `floor`, `e3`, `e3_steps` |
| T | `tests/test_tau_blend_mask_analysis.py` | test result |

Regenerate the results with these commands:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python benchmarks/tau_blend_mask_analysis.py
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python benchmarks/tau_blend_cfl_fidelity.py
```

## Conventions

- "prec" is the field `gmres_preconditioned`: the first k with
  ||M (b - J x_k)|| <= 1e-8 ||M b||. The repository benchmarks use this
  convention.
- "true" is the field `gmres_true`: the first k with
  ||b - J x_k|| <= 1e-8 ||b||, from the same Krylov sequence.
- The email gives one count for each item. The column "Expected" gives the
  convention that agrees with the saved private data. The private A9 and A10
  data use "prec" only. The N = 2048 floor case is the exception (row C9).
- "MA row (N, m, case)" selects one entry of the list `cases`.
- The CFL mask is chi_j = 4 dt D_j / h^2 <= eta, with eta = 0.1 in MA. The
  private study A9 used dt D_j lambda_max <= 0.1. The MA field
  `a9_rule_and_cfl_rule_agree` shows if the two rules select the same nodes.
- The expected values come from the private CSV files
  `a9_blend_ablation.csv` (moljax 39f4091) and `a10_cfl_mask.csv`
  (moljax 78925c5 or later), unless the column "Source" gives a different
  source. These runs are not in this repository. Thus the expected values
  are not verified by a run of the committed scripts.
- MA and CF use their own GMRES (Givens rotations). A10 used a
  least-squares solve in each iteration. Small differences in the counts
  are possible. A difference must be reported, not corrected by hand.

## Section "THE DOMINANT EFFECT IN THE BLEND"

| # | Email number | Script | Case or selector | Output field | Expected | Source | Value after regeneration |
|---|---|---|---|---|---|---|---|
| D1 | Item 1: P^-1 J = [[I, X], [0, Y]] and spectrum {1}^n0 with sigma(Y) when D = 0 on the inactive set | T | `test_mask_gives_block_triangular_form_when_masked_coefficient_is_zero` | test passes | pass | algebra (Lemma) | |
| D2 | "90 to 97 percent" of the GMRES reduction | MA | N = 512, 1024; m = 4, 8; (`scalar_max` - `cfl_mask_one_inverse_max`) / (`scalar_max` - `repo_blend_B3`) | `gmres_preconditioned` | 0.905, 0.897, 0.973, 0.921 | A9 CSV | |
| D3 | N = 512, m = 4: "246 to 37 to 15" | MA | (512, 4): `scalar_max`, `cfl_mask_one_inverse_max`, `repo_blend_B3` | `gmres_preconditioned` | 246, 37, 15 | A9 CSV | |
| D4 | N = 512, m = 8: "278 to 44 to 17" | MA | (512, 8): same three cases | `gmres_preconditioned` | 278, 44, 17 | A9 CSV | |
| D5 | N = 1024, m = 4: "425 to 65 to 55" | MA | (1024, 4): same three cases | `gmres_preconditioned` | 425, 65, 55 | A9 CSV | |
| D6 | N = 1024, m = 8: "488 to 69 to 33" | MA | (1024, 8): same three cases | `gmres_preconditioned` | 488, 69, 33 | A9 CSV | |
| D7 | "10 and 14 eigenvalues below 0.1" (mask and one inverse, N = 512) | MA | (512, 4), (512, 8): `cfl_mask_one_inverse_max` | `eigenvalues_below_0p1` | 10, 14 | A9 CSV | |
| D8 | "kappa_2 stays at 1.6e3 to 2.1e3" | MA | (512, 4), (512, 8): `cfl_mask_one_inverse_max` | `kappa2` | 1597, 2059 | A9 CSV | |
| D9 | "against 55 and 51 for your blend" | MA | (512, 4), (512, 8): `repo_blend_B3` | `kappa2` | 54.78, 51.44 | A9 CSV | |
| D10 | Blend without the mask: "148 to 226 eigenvalues stay below 0.1" in three of four cases | MA | `repo_blend_B3_no_mask` at (512, 4), (512, 8), (1024, 8) | `eigenvalues_below_0p1` | 196, 148, 226 | A9 CSV | |
| D11 | Blend without the mask: "89 to 135 iterations" | MA | same three entries | `gmres_preconditioned` | 89, 99, 135 | A9 CSV | |
| D12 | N = 1024, m = 4: "the references alone remove the tail", "96 iterations, against 55" | MA | (1024, 4): `repo_blend_B3_no_mask`; `repo_blend_B3` | `eigenvalues_below_0p1`; `gmres_preconditioned` | 0; 96 and 55 | A9 CSV | |
| D13 | m = 2: "With the references taken off the floor and no mask, 170 and 466 small eigenvalues stay" | MA | (512, 2), (1024, 2): `cfl_refs_no_mask_B3` | `eigenvalues_below_0p1` | 170, 466 | A9 CSV | |
| D14 | "an additional GMRES gain of a factor of approximately 2.5 at N = 512" | MA | (512, 4), (512, 8): `cfl_mask_one_inverse_max` / `repo_blend_B3` | `gmres_preconditioned` | 2.47, 2.59 | A9 CSV | |
| D15 | "For m = 4 this factor decreases to 1.2 at N = 1024" | MA | (1024, 4): same ratio | `gmres_preconditioned` | 1.18 | A9 CSV | |
| D16 | "from 50, 15 and 17 to 105, 55 and 33" (blend under refinement) | MA | `repo_blend_B3`, m = 2, 4, 8 at N = 512 and 1024 | `gmres_preconditioned` | 50, 15, 17; 105, 55, 33 | A9 CSV | |
| D17 | Item 3: the CFL mask "selects exactly the floor nodes" (m = 2) | MA | `states`, (512, 2), (1024, 2) | `cfl_mask_equals_floor_nodes`, `cfl_masked_count` | true; 334, 668 | A9 CSV (inactive count) | |
| D18 | "the mask alone still needs 50 and 105 iterations" | MA | (512, 2), (1024, 2): `cfl_mask_repo_refs_B3` | `gmres_preconditioned` | 50, 105 | private check; not in the A9 CSV | |
| D19 | Related control, not in the email: the mask with one inverse at m = 2 | MA | (512, 2), (1024, 2): `cfl_mask_one_inverse_max` | `gmres_preconditioned` | 33, 45 | A9 CSV | |
| D20 | "the counts become 14 and 17 again" | MA | (512, 2), (1024, 2): `cfl_blend_B3` | `gmres_preconditioned` | 14, 17 | A9 CSV | |
| D21 | "Your earlier, incorrect coefficient did both of these things" | none | none | none | none | the original branch with the old coefficient | |
| D22 | Item 4: "||X|| is 9 to 12.6 at N = 512" | MA | (512, 4), (512, 8): `cfl_mask_one_inverse_*` and `repo_blend_B3` | `block_X_norm` | 8.94 to 12.6 | A9 CSV | |
| D23 | "negative eigenvalues, down to approximately -4" (Hermitian part) | MA | (512, 4), (512, 8): `repo_blend_B3` | `hermitian_part_min_eigenvalue` | -3.96, -2.89 | A9 CSV | |
| D24 | "This agrees with the abstention of the disk test" | none | none | `disk_only_verdict` | "indeterminate" | `benchmarks/results/pme_dst_tau_blend_spectral_analysis.json` | |
| D25 | "At m = 2 there is no mask and no X" | MA | (512, 2), (1024, 2): `repo_blend_B3` | `identity_input_count`, `block_X_norm` | 0, null | A9 CSV | |
| D26 | Item 5: mask, one inverse and frame: "12 to 14 GMRES iterations" at N = 512 and 1024, all m | MA | `cfl_mask_frame` (eta = 0.1) and `cfl_mask_frame_eta1` | `gmres_preconditioned` | N = 512: 12, 12, 14. N = 1024: no saved value | A10 CSV (N = 512, E1E2, dt = 0.02, Zf) | |
| D27 | "In each case the count is smaller than the count of the blend" | MA | `cfl_mask_frame` against `repo_blend_B3` | `gmres_preconditioned` | frame < blend for all six states | D16, D26 | |
| D28 | "kappa_2 of this variant is large (up to 2e5)" | MA | (1024, 4), (1024, 8): `cfl_mask_frame_eta1` and `cfl_mask_frame` | `kappa2` | 1.86e5, 1.23e5 (eta = 1) | private CFL study of 2026-09 (N = 1024, eta = 1); no saved CSV | |
| D29 | "at m = 8 it has negative eigenvalues" | MA | (512, 8), (1024, 8): `cfl_mask_frame`, `cfl_mask_frame_eta1` | `min_real_eigenvalue` | < 0 (-15.17 at N = 512, eta = 1) | private CFL study of 2026-09; no saved CSV | |
| D30 | "The symmetric version ... controls kappa: the diag(D)^(1/2) similarity ..., then PCG" | none | none | none | none | no measurement; a statement of method | |

## Section "CFL ANALYSIS OF THE MASK"

| # | Email number | Script | Case or selector | Output field | Expected | Source | Value after regeneration |
|---|---|---|---|---|---|---|---|
| C1 | "each variant gives exact backward Euler to 5e-9 or better" (when Newton converges on the true residual) | CF | `e3` rows with `converged` true | max of `relative_l2_difference_to_reference` | 4.9e-9 (m = 8, dt = 0.2, right, forcing 0.1, S) | A10 CSV | |
| C2 | "The cutoff bounds the mask perturbation: ||J - J0|| <= eta" | CF | all `e1_e2` rows of kind Z1 and Zf | `J_minus_J0_norm` <= `eta` | true for all rows | algebra (Lemma); the A10 CSV has no such field | |
| C3 | "by 10 to 1000 times in single-step measurements", "The difference increases with the CFL number" | CF | `e4`, tolerance 1e-8 | `true_over_preconditioned` (normalized) and `absolute_ratio_r_over_Mr` | 10 to 1000 | no saved source; see "Numbers without a saved source" | |
| C4 | "With loose forcing (0.1), the true linear residual can then be more than one" | CF | `e3`, side left, forcing 0.1 | `max_true_linear_ratio` | > 1 in several rows (for example 2.9, m = 2, dt = 0.2, S) | A10 CSV | |
| C5 | "the front was behind: by 1.2 in x at m = 4, c = 3.5e4" | CF | `e3`, m = 4, dt = 0.2, left, forcing 0.1, kinds S and Z1 | `front_halfwidth_difference` | -1.2 | A10 CSV | |
| C6 | "and by 1.9 for the frame at m = 2, c = 1.1e4" | CF | `e3`, m = 2, dt = 0.2, left, forcing 0.1, Zf | `front_halfwidth_difference` | -1.9 | A10 CSV | |
| C7 | "Your blend converged in several runs with ratios up to 5" | CF | `e3`, kind B3, `converged` true | `max_true_linear_ratio` | 5.3 (m = 4, dt = 0.2, left, forcing 0.1) | A10 CSV | |
| C8 | "It failed only two of five steps at m = 8, c = 6.9e4" | CF | `e3`, m = 8, dt = 0.2, left, forcing 0.1, B3 | `steps_converged`, `line_search_failures` | 3, 2 | A10 CSV | |
| C9 | "Your inexact-Newton harness already checks the true residual" | none | none | none | none | `moljax/experimental/pme_inexact_newton.py` (code) | |
| C10 | "4 dt m eps^(m-1)/h^2 ... At m = 2, N = 2048, dt = 0.02, this value is 0.105" | CF; T | `floor` rows; `test_floor_cfl_value_and_empty_mask_at_n2048` | `floor_cfl` | 0.104960025 | closed form | |
| C11 | "eta = 0.1 loses the mask (more than 300 iterations)" | CF | `floor`, eta = 0.1, Z1 and Zf | `masked`, `gmres_true`, `gmres_preconditioned` | 0; true: none within 300 (both kinds); prec: none within 300 (Z1), 269 (Zf) | run of 2026-10-25 (see below) | |
| C12 | "eta = 0.3 or 1 keeps the mask (97 iterations with one inverse, 20 with the frame)" | CF | `floor`, eta = 0.3 and 1 | `masked`, `gmres_true` | 1334; 97 (Z1), 20 (Zf) | run of 2026-10-25 (see below) | |
| C13 | Same item, preconditioned convention (not in the email) | CF | `floor`, eta = 0.3 and 1 | `gmres_preconditioned` | 76 (Z1), 16 (Zf) | run of 2026-10-25; an earlier private check | |
| C14 | "approximately 4, 8 and 18 iterations for each step to 1e-8 at c of approximately 1e2, 1e3 and 1e4" | CF | `e3`, `reference_newton_to_1e-8_per_step`; `e3_steps` with kind "direct" | mean and max over the five steps | means 3.8 to 4.2, 7.6 to 8.2, 17.6 to 19.4; max per step at dt = 0.2: 38, 39, 30 | A10 CSV (means and max only) | |
| C15 | "With right preconditioning and forcing 0.1, the four preconditioners usually stay within 20 percent of each other" | CF | `e3`, side right, forcing 0.1 | `newton_total` | qualitative | A10 CSV | |
| C16 | "at m = 2, c = 1.1e4, they need 54, 65, 69 and 119 Newton iterations over five steps (scalar, blend, mask, mask plus frame)" | CF | `e3`, m = 2, dt = 0.2, right, forcing 0.1; S, B3, Z1, Zf | `newton_total` | 54, 65, 69, 119 | A10 CSV | |
| C17 | "At m = 8, c = 6.9e4, ... 17002 with the scalar, 6312 with the mask, 1971 with your blend and 1396 with the mask plus the frame" | CF | `e3`, m = 8, dt = 0.2, right, forcing 0.1; S, Z1, B3, Zf | `linear_iterations_total` | 17002, 6312, 1971, 1396 | A10 CSV | |
| C18 | The values of c: 3.5e4 (m = 4), 1.1e4 (m = 2), 6.9e4 (m = 8) at dt = 0.2; 1e2, 1e3, 1e4 at dt = 2e-3, 2e-2, 0.2 | CF | `e3` | `global_cfl` | 35359, 10631, 69022 | A10 CSV | |

## Section "DIRECTION" (CFL numbers that E1 also gives)

| # | Email number | Script | Case or selector | Output field | Expected | Source | Value after regeneration |
|---|---|---|---|---|---|---|---|
| R1 | "When c increases from 0.1 to 1e5, the count of the scalar increases from approximately 5 to more than 300" | CF | `e1_e2`, kind S, dt = 2e-6 and 2 | `gmres_true` | 5 (m = 2, c = 0.106); 306 and 354 (m = 4, 8, dt = 2) | private CFL study of 2026-09 (both conventions); no saved CSV | |
| R2 | "to approximately 55 with one inverse" | CF | `e2_best_cutoffs`, kind Z1, dt = 2 | `best_gmres_true` | 55, 48, 54 | private CFL study of 2026-09; no saved CSV | |
| R3 | "and to 21 to 23 with the frame" | CF | `e2_best_cutoffs`, kind Zf, dt = 2 | `best_gmres_true` | 23, 17, 19 (frame); 21 for the CFL blend at m = 2 | private CFL study of 2026-09; no saved CSV | |
| R4 | "Up to c of approximately 10, the best cutoff often masks all nodes" | CF | `e2_best_cutoffs`, c <= 10 | `best_masked_gmres_true` = N | several entries | private CFL study of 2026-09 | |

## Result of the N = 2048 floor case

The floor case ran one time on 2026-10-25 (moljax 9ecfa7f plus these
scripts, CPU, float64, one thread). Command:
`python benchmarks/tau_blend_cfl_fidelity.py --parts floor`.

| eta | Masked nodes | Z1 prec | Z1 true | Zf prec | Zf true |
|---:|---:|---:|---:|---:|---:|
| 0.1 | 0 | > 300 | > 300 | 269 | > 300 |
| 0.3 | 1334 | 76 | 97 | 16 | 20 |
| 1 | 1334 | 76 | 97 | 16 | 20 |

The email counts 97 and 20 use the true residual. The counts 76 and 16 of
the saved scripts use the preconditioned residual. Thus the two values do
not contradict each other. At 300 iterations and eta = 0.1, the true
relative residuals were 1.68e-2 (Z1) and 1.83e-7 (Zf). With the
preconditioned convention, the frame converges at eta = 0.1 in 269
iterations. The statement "more than 300 iterations" is thus correct for
the true residual only.

## Numbers without a saved source

1. C3, "10 to 1000 times": no saved run gives this range. E4 gives the
   normalized ratio and the absolute ratio ||r|| / ||M r||. The absolute
   ratio changes when M is multiplied by a scalar. Use the normalized ratio
   for a claim, or state the scale of M.
2. D18, "50 and 105": the A9 CSV has no case with the CFL mask and the
   original references. A private check gave these
   counts. MA now has the case `cfl_mask_repo_refs_B3`.
3. D26 to D29 at N = 1024: no saved CSV has the frame at N = 1024. The
   values come from the private CFL study of 2026-09 with eta = 1.
4. D21, D24, D30 and C9 are not measurements of these scripts.
5. R1 to R4: the private CFL study of 2026-09 required both conventions
   and selected the best eta in a larger set (up to eta = 1000). CF uses
   eta <= 100. The best-cutoff values can thus change.
