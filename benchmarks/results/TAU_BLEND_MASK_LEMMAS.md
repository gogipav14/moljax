# Lemmas for the mask analysis of the DST/tau blend

This section states the lemmas behind the mask analysis, with their
assumptions and proofs. The scripts `benchmarks/tau_blend_mask_analysis.py`
and `benchmarks/tau_blend_cfl_fidelity.py` measure the quantities that the
lemmas name. `TAU_BLEND_MASK_CLAIMS.md` maps each number to its source.

## Setting and notation

- Grid: `N` interior nodes of a node-centred grid with spacing `h` and
  homogeneous Dirichlet boundary values.
- `L` is the second-difference matrix: `-2/h^2` on the diagonal and `1/h^2`
  on the two neighbor diagonals. `T = -L` is symmetric positive definite,
  with eigenvalues `mu_k` in `(0, 4/h^2)`.
- `D = diag(d_1, ..., d_N)` with `d_j >= 0`. For the regularized PME,
  `d_j = m (u_j^2 + eps^2)^((m-1)/2) >= m eps^(m-1)`.
- Backward Euler Jacobian: `J = I + dt T D` (equal to `I - dt L diag(D)`).
- Local diffusive CFL number: `chi_j = 4 dt d_j / h^2`. Global CFL number:
  `c = 4 dt max_j d_j / h^2`.
- Mask: `Z` is the set of masked nodes (`chi_j <= eta` for a cutoff `eta`),
  and `A` is its complement. `Q` is the diagonal 0/1 projector onto `Z`.
- `H(d) = (I + dt d T)^-1` is the DST Helmholtz inverse for a scalar `d > 0`.
- Preconditioner with input weights (left preconditioning, `M J`):
  `M = Q + sum_w H(d_w) W_w`, where each `W_w` is diagonal and zero on `Z`.
  The repository blend, the mask with one inverse (`W = I - Q`) and the frame
  variant (`W = diag(Dmax/d_j)` on `A`) all have this form.
- Norms: `||.||_p` is the vector p-norm and the induced matrix norm. Residual
  ratios are relative: `||r|| / ||b||` and `||M r|| / ||M b||`.

## Lemma 1. The mask gives an exact identity block

Assumptions: `d_j = 0` for each `j` in `Z`, and `M e_j = e_j` for each `j` in
`Z` (the identity on masked inputs).

Statement: in the node order `(Z, A)`,

    M J = [[I, X], [0, Y]],

with `X` the rows of `Z` and `Y` the rows of `A`, both in the columns of `A`.
Thus the spectrum of `M J` is `{1}` with multiplicity `|Z|`, together with the
spectrum of `Y`.

Proof: for `j` in `Z`, `J e_j = e_j + dt T D e_j = e_j`, because `d_j = 0`. Thus
`M J e_j = M e_j = e_j`. So each column of `M J` in `Z` is the identity column,
and the block form follows. The determinant of a block triangular matrix gives
the spectrum.

Limits: the lemma gives no information about `Y`, and no bound on the
condition number: a large `X` can make `kappa_2(M J)` large and can put the
origin inside the field of values while the spectrum stays at `1` and in
`sigma(Y)`. The regularized PME has `d_j > 0`; Lemma 2 bounds the difference.

## Lemma 2. The cutoff bounds the mask perturbation

Assumptions: the setting above. Let `J_0 = I + dt T D (I - Q)` (the
coefficient set to zero on `Z`), and `eta_Z = max_{j in Z} chi_j <= eta`.

Statement:

1. `||J - J_0||_p <= eta_Z` for `p = 1, 2, infinity`.
2. `||M (J - J_0)||_2 <= ||M||_2 eta_Z`. (The scripts measure this left side.)
3. `J` and `J_0` are M-matrices with `J^-1 >= 0` and `||J^-1||_1 <= 1`. If
   `J x = b` and `J_0 x_0 = b`, then `||x_0 - x||_1 <= eta_Z ||Q x||_1`.

Proof:
1. `J - J_0 = dt T D Q`. Column `j` of this matrix is `dt d_j T e_j` for `j` in
   `Z` and zero otherwise. Its absolute sum is at most `dt d_j (2 + 1 + 1)/h^2
   = chi_j`. Each row has at most three nonzero entries of `T` (absolute sum
   `<= 4/h^2`) times `dt d_j` with `j` in `Z`, so each absolute row sum is also
   at most `eta_Z`. The bound for `p = 2` follows from
   `||B||_2 <= sqrt(||B||_1 ||B||_inf)`.
2. Submultiplicativity.
3. The off-diagonal entries of `J` are `-dt d_j / h^2 <= 0`, and the sum of
   column `j` is `1 + dt d_j (2 - k_j)/h^2 >= 1`, where `k_j <= 2` is the number
   of neighbors. Thus `J` is a column diagonally dominant Z-matrix with margin
   at least one, so `J^-1 >= 0` and `||J^-1||_1 <= 1`. The same holds for
   `J_0`. Then `x_0 - x = J_0^-1 (J - J_0) x`, and the 1-norm bound of part 1
   applied to `Q x` gives the result.

## Lemma 3. Backward Euler for the PME is nonexpansive in the 1-norm

Assumptions: `F(v) = v - dt L phi(v)`, where `phi` acts componentwise, is
continuously differentiable, and has `phi' = d >= 0` (the regularized PME
potential satisfies this).

Statement: for all `u`, `v`, `||u - v||_1 <= ||F(u) - F(v)||_1`. Thus, for the
exact trajectory `F(u^n) = u^(n-1)` and an approximate trajectory
`F(v^n) = v^(n-1) + rho^n` (`rho^n` is the nonlinear residual of step `n`),

    ||v^n - u^n||_1 <= ||v^0 - u^0||_1 + sum_{k=1..n} ||rho^k||_1.

The bound holds for each preconditioner, each cutoff and each CFL number.

Proof: `F(u) - F(v) = (I + dt T diag(dbar)) (u - v)`, with the secant
coefficients `dbar_j = integral_0^1 phi'(v_j + s (u_j - v_j)) ds >= 0`. By part 3
of Lemma 2 this matrix has an inverse with 1-norm at most one. Apply the
inequality at each step and sum.

## Lemma 4. Left preconditioning can hide the true residual

Assumptions: `M` is nonsingular.

Statement: `||r|| / ||b|| <= kappa(M) ||M r|| / ||M b||`, with
`kappa(M) = ||M|| ||M^-1||` in the same norm.

Proof: `||r|| <= ||M^-1|| ||M r||` and `||M b|| <= ||M|| ||b||`.

Consequence: a stopping test on the left-preconditioned residual does not bound
the true residual, unless `kappa(M)` is small. Right preconditioning, or a
check of the true residual, removes the gap.

## Lemma 5. The mask disappears above the floor

Assumptions: the regularized PME coefficient.

Statement: each `chi_j >= chi_floor = 4 dt m eps^(m-1) / h^2`. If
`eta < chi_floor`, the mask is empty.

Proof: `d_j >= m eps^(m-1)`.

Consequence: at fixed `dt`, `chi_floor` grows as `h^-2`, so a fixed cutoff loses
the mask on a fine enough grid. At `m = 2`, `N = 2048`, `dt = 0.02` on
`[-4, 4]`, `chi_floor = 0.105`.

## Lemma 6. A scalar reference keeps small eigenvalues on a floor region

Assumptions: `d > 0`, and `D = f I + Delta` with `f > 0`, `Delta >= 0` diagonal
of rank `s`.

Statement: the eigenvalues of `H(d) J` are real and positive, and

    #{lambda(H(d) J) < a} >= #{beta_k < a} - s,
    beta_k = (1 + dt f mu_k) / (1 + dt d mu_k).

Proof: `H(d) J = (T^-1 + dt d I)^-1 (T^-1 + dt D)`, which is similar to the
symmetric positive definite matrix
`(T^-1 + dt d I)^(-1/2) (T^-1 + dt D) (T^-1 + dt d I)^(-1/2)`. This matrix is
a positive semidefinite update of rank `s` of a matrix with eigenvalues
`beta_k`. A positive semidefinite update of rank `s` moves at most `s`
eigenvalues across any level.

Consequence: for `f` much smaller than `d`, the high-frequency `beta_k` are
near `f/d`, so a scalar reference near the maximum keeps a tail of small
eigenvalues, one for almost each degree of freedom in the floor region.

## Lemma 7. The frame gain for one mode (constant-coefficient model)

Assumptions: a model with commuting operators: one eigenvector with the
eigenvalue `a > 0` of `dt Dmax T` and a constant ratio `s = d/Dmax` in
`(0, 1]`; no mask boundary.

Statement: the mask with one inverse gives the eigenvalue
`b_1 = (1 + a s)/(1 + a)`, the frame gives `b_f = (1 + a s)/(s (1 + a))`, and
`|b_f - 1| / |b_1 - 1| = 1/(a s)`.

Proof: substitute the eigenvector.

Limits: for the high frequencies, `a s` is near `chi_j`. The frame helps where
the modal CFL number is more than one and amplifies the modes where it is
less than one. With a variable coefficient, a commutator term and the mask
interface also enter; the lemma does not cover them.

## Statements without a proof here

- The symmetric version of the frame (the `diag(D)^(1/2)` similarity on the
  active block, then PCG): a symmetric positive definite transformation
  permits PCG, but it does not by itself bound the condition number. The
  prediction in the private note (kappa about 3.8) is not tested against the
  corrected coefficient.
- All GMRES iteration counts are measurements, not consequences of these
  lemmas.
