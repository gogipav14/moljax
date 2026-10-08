# PME DST/tau-blend experimental report

## Scope and claim discipline

This is a measurement report for Newton--Krylov preconditioning of the
node-centred one-dimensional porous-medium equation (PME), using backward Euler
for the linear-solve study and Crank--Nicolson/backward-Euler-compatible residual
and preconditioner machinery. It does **not** apply to IMEX--Strang or ETDRK4:
those methods avoid Newton solves and are unaffected by this work (moljax paper,
Sec. 6.1.3). The experiments use `N=512` and `N=1024`, PME exponents
`m in {2,4,8}`, one wide-front backward-Euler linearization for the main
work--precision comparison, and node-centred homogeneous Dirichlet boundaries.

There is no multigrid baseline. Accordingly this report makes **no performance
claim** against the relevant solver state of the art. Wall times below are
capability measurements on this machine, not a recommendation to replace a
multigrid method or moljax's default preconditioner.

## Motivation: where the classical constant reference stops being graceful

Constant-coefficient FFT/DST preconditioning for a variable-coefficient diffusion
operator is classical. QSC/FFT solvers, the variable-coefficient disk solver, and
fast sine-transform analyses obtain mesh-independent or spectrally equivalent
preconditioners when the coefficient is positive and has bounded contrast. The
moljax paper says variable coefficients should "degrade performance gracefully"
(Sec. 6.4, limitation 2). Degenerate PME diffusion is outside that contract:
`D(u)` vanishes on a set of positive measure, so the literal contrast is unbounded.

The tested fixed operator is

`P_tau^-1 r = W0 r + sum_w H(d_w) (W_w r)`,

where the active-support weights form a partition of unity and each `H(d_w)` is a
constant-coefficient DST-I Helmholtz inverse. This is an incremental experimental
remedy, not a claim to a new preconditioner class. Partition-of-unity
preconditioning is established in RBF-PUM and SORAS, while MPGMRES already uses
multiple preconditioners by enlarging the Krylov search space. Unlike MPGMRES, this
blend is one fixed linear operator and retains ordinary GMRES; it does not incur
the multipreconditioned Krylov-space orthogonalisation growth.

## Primary result: the spectral-equivalence boundary

The decisive observation is not that the arithmetic mean was chosen poorly. Every
effective nonzero scalar Helmholtz reference tested here--arithmetic mean, bulk
mean, constant one, geometric mean, harmonic mean, and the per-case GMRES oracle
`d0*`--retains a large near-zero spectral tail. At `m=2/4/8`, their condition
numbers are approximately `1.06e3/3.54e3/6.90e3`, with 170--322 eigenvalues below
magnitude 0.1. The oracle `d0*` is spectrally the worst of that group: it has the
largest condition number and smallest `min|lambda|` at all three exponents. The
near-zero `floor` reference and identity do not manufacture the tail, but they
also provide essentially no useful conditioning and have still larger condition
numbers. The three-reference blend removes the tail: zero eigenvalues below 0.1,
with `kappa_2=11.1/54.8/51.4`.

| m | method | kappa_2 | count abs(lambda)<0.1 | min abs(lambda) | spectral abscissa | numerical abscissa | gap | disk rate |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | identity | 1.77e+03 | 0 | 1 | 1.06e+03 | 1.06e+03 | 9.33e-05 | 1.0003 |
| 2 | mean | 1.06e+03 | 294 | 0.00406 | 4.29 | 4.29 | 1.52e-10 | 0.9981 |
| 2 | bulk | 1.06e+03 | 320 | 0.00103 | 1.09 | 1.09 | 5.33e-10 | 0.9981 |
| 2 | floor | 1.76e+03 | 0 | 1 | 1.05e+03 | 1.05e+03 | 9.18e-05 | 0.9981 |
| 2 | const | 1.06e+03 | 300 | 0.00305 | 3.22 | 3.22 | 8.5e-11 | 0.9981 |
| 2 | geometric | 1.06e+03 | 280 | 0.00707 | 7.47 | 7.47 | 5.2e-10 | 0.9981 |
| 2 | harmonic | 1.06e+03 | 302 | 0.00292 | 3.09 | 3.09 | 7.81e-11 | 0.9981 |
| 2 | oracle d0* | 1.07e+03 | 322 | 0.000946 | 1 | 1.01 | 0.0107 | 0.9981 |
| 2 | blend l=3 | 11.1 | 0 | 0.799 | 2.02 | 3.25 | 1.23 | 1.4749 |
| 4 | identity | 8.3e+03 | 0 | 1 | 3.52e+03 | 3.52e+03 | 0.000124 | 0.9994 |
| 4 | mean | 3.54e+03 | 236 | 0.00444 | 15.7 | 15.7 | 2.05e-10 | 0.9994 |
| 4 | bulk | 3.54e+03 | 272 | 0.000296 | 1.05 | 1.05 | 4.04e-09 | 0.9994 |
| 4 | floor | 8.3e+03 | 0 | 1 | 3.52e+03 | 3.52e+03 | 0.000124 | 0.9994 |
| 4 | const | 3.54e+03 | 242 | 0.00303 | 10.7 | 10.7 | 8.42e-11 | 0.9994 |
| 4 | geometric | 3.54e+03 | 246 | 0.00235 | 8.3 | 8.3 | 4.7e-11 | 0.9994 |
| 4 | harmonic | 3.54e+03 | 258 | 0.000891 | 3.15 | 3.15 | 6.29e-12 | 0.9994 |
| 4 | oracle d0* | 3.59e+03 | 272 | 0.000283 | 1 | 1.01 | 0.014 | 0.9994 |
| 4 | blend l=3 | 54.8 | 0 | 0.848 | 2.32 | 7.36 | 5.04 | 3.3735 |
| 8 | identity | 1.86e+04 | 0 | 1 | 6.88e+03 | 6.88e+03 | 0.000131 | 0.9997 |
| 8 | mean | 6.9e+03 | 170 | 0.0112 | 77.1 | 77.1 | 1.91e-09 | 0.9997 |
| 8 | bulk | 6.9e+03 | 224 | 0.000149 | 1.03 | 1.03 | 7.35e-09 | 0.9997 |
| 8 | floor | 1.86e+04 | 0 | 1 | 6.88e+03 | 6.88e+03 | 0.000131 | 0.9997 |
| 8 | const | 6.9e+03 | 192 | 0.00303 | 20.9 | 20.9 | 8.23e-11 | 0.9997 |
| 8 | geometric | 6.9e+03 | 198 | 0.00186 | 12.9 | 12.9 | 2.64e-11 | 0.9997 |
| 8 | harmonic | 6.9e+03 | 208 | 0.000575 | 3.96 | 3.96 | 2.08e-12 | 0.9997 |
| 8 | oracle d0* | 7e+03 | 224 | 0.000145 | 1 | 1.01 | 0.0135 | 0.9997 |
| 8 | blend l=3 | 51.4 | 0 | 0.833 | 3.5 | 6.15 | 2.66 | 2.8932 |

This is the boundary of the classical bounded-contrast argument in these data:
effective single-reference preconditioning loses spectral equivalence when the
coefficient has a positive-measure zero set; the partitioned blend restores a
zero-free spectrum for the tested states.

## Linear-solve measurements

Each cell is `counted GMRES iterations; median +/- IQR seconds`. Timings exclude
two warmups, synchronize with `jax.block_until_ready`, and include fresh
linearization/preconditioner construction plus the solve. `cap` means the target
was not reached within the configured budget (the inherited counter reports 401
at a budget of 400; that reporting convention is deliberately left unchanged).

| N | m | tol | identity | mean | bulk | floor | const | geometric | harmonic | oracle d0* | blend l=3 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 2 | 1e-02 | 29; 0.155 +/- 0.013 s | 13; 0.126 +/- 0.012 s | 7; 0.114 +/- 0.015 s | 29; 0.202 +/- 0.006 s | 11; 0.113 +/- 0.028 s | 16; 0.142 +/- 0.028 s | 11; 0.122 +/- 0.018 s | 7; 0.118 +/- 0.013 s | 4; 0.130 +/- 0.005 s |
| 512 | 2 | 1e-05 | 111; 1.251 +/- 0.061 s | 111; 1.322 +/- 0.058 s | 101; 1.094 +/- 0.040 s | 96; 0.996 +/- 0.049 s | 109; 1.278 +/- 0.018 s | 114; 1.376 +/- 0.057 s | 109; 1.274 +/- 0.037 s | 99; 1.060 +/- 0.043 s | 10; 0.146 +/- 0.020 s |
| 512 | 2 | 1e-08 | 121; 1.413 +/- 0.049 s | 182; 3.202 +/- 0.063 s | 166; 2.707 +/- 0.068 s | 111; 1.276 +/- 0.096 s | 178; 3.058 +/- 0.098 s | 190; 3.471 +/- 0.062 s | 178; 3.089 +/- 0.079 s | 166; 2.711 +/- 0.048 s | 14; 0.169 +/- 0.020 s |
| 512 | 4 | 1e-02 | 82; 0.729 +/- 0.031 s | 48; 0.349 +/- 0.018 s | 16; 0.148 +/- 0.016 s | 82; 0.757 +/- 0.017 s | 41; 0.291 +/- 0.024 s | 37; 0.252 +/- 0.012 s | 25; 0.180 +/- 0.004 s | 16; 0.143 +/- 0.038 s | 4; 0.154 +/- 0.026 s |
| 512 | 4 | 1e-05 | 150; 2.184 +/- 0.073 s | 184; 3.329 +/- 0.145 s | 143; 2.034 +/- 0.026 s | 150; 2.277 +/- 0.083 s | 176; 3.051 +/- 0.094 s | 169; 2.856 +/- 0.107 s | 153; 2.314 +/- 0.087 s | 142; 2.003 +/- 0.041 s | 11; 0.176 +/- 0.024 s |
| 512 | 4 | 1e-08 | 156; 2.542 +/- 0.022 s | 284; 8.125 +/- 0.207 s | 245; 6.122 +/- 0.101 s | 156; 2.609 +/- 0.138 s | 278; 7.908 +/- 0.204 s | 273; 7.584 +/- 0.120 s | 261; 6.912 +/- 0.049 s | 245; 6.071 +/- 0.089 s | 15; 0.169 +/- 0.023 s |
| 512 | 8 | 1e-02 | 102; 1.158 +/- 0.056 s | 84; 0.885 +/- 0.021 s | 18; 0.174 +/- 0.016 s | 102; 1.236 +/- 0.043 s | 63; 0.545 +/- 0.068 s | 52; 0.424 +/- 0.012 s | 32; 0.233 +/- 0.017 s | 17; 0.161 +/- 0.016 s | 5; 0.158 +/- 0.018 s |
| 512 | 8 | 1e-05 | 190; 3.687 +/- 0.055 s | 253; 6.305 +/- 0.427 s | 170; 2.855 +/- 0.062 s | 190; 3.587 +/- 0.074 s | 228; 4.941 +/- 0.040 s | 216; 4.503 +/- 0.175 s | 194; 3.627 +/- 0.100 s | 170; 2.844 +/- 0.099 s | 13; 0.196 +/- 0.029 s |
| 512 | 8 | 1e-08 | 197; 3.706 +/- 0.176 s | 334; 10.426 +/- 0.143 s | 278; 7.352 +/- 0.113 s | 197; 3.736 +/- 0.147 s | 317; 9.539 +/- 0.161 s | 310; 9.095 +/- 0.164 s | 292; 8.082 +/- 0.094 s | 277; 7.270 +/- 0.120 s | 17; 0.189 +/- 0.007 s |
| 1024 | 2 | 1e-02 | 50; 0.329 +/- 0.006 s | 12; 0.130 +/- 0.026 s | 7; 0.120 +/- 0.014 s | 48; 0.363 +/- 0.053 s | 11; 0.121 +/- 0.016 s | 20; 0.149 +/- 0.005 s | 11; 0.124 +/- 0.030 s | 7; 0.114 +/- 0.010 s | 5; 0.137 +/- 0.007 s |
| 1024 | 2 | 1e-05 | 213; 4.503 +/- 0.133 s | 188; 3.488 +/- 0.037 s | 165; 2.739 +/- 0.223 s | 176; 3.135 +/- 0.135 s | 183; 3.367 +/- 0.075 s | 201; 4.100 +/- 0.064 s | 185; 3.453 +/- 0.075 s | 164; 2.831 +/- 0.247 s | 12; 0.153 +/- 0.008 s |
| 1024 | 2 | 1e-08 | 253; 6.164 +/- 0.226 s | 352; 11.989 +/- 0.119 s | 318; 9.835 +/- 0.154 s | 228; 5.132 +/- 0.223 s | 342; 11.229 +/- 0.302 s | 369; 13.146 +/- 0.147 s | 343; 11.374 +/- 0.313 s | 317; 9.759 +/- 0.111 s | 17; 0.181 +/- 0.033 s |
| 1024 | 4 | 1e-02 | 196; 3.807 +/- 0.087 s | 47; 0.344 +/- 0.032 s | 15; 0.142 +/- 0.004 s | 196; 3.828 +/- 0.130 s | 41; 0.304 +/- 0.064 s | 74; 0.691 +/- 0.004 s | 55; 0.454 +/- 0.009 s | 15; 0.142 +/- 0.017 s | 11; 0.161 +/- 0.017 s |
| 1024 | 4 | 1e-05 | 322; 10.132 +/- 0.086 s | 303; 8.967 +/- 0.006 s | 216; 4.714 +/- 0.100 s | 322; 9.985 +/- 0.259 s | 290; 8.164 +/- 0.152 s | 357; 12.163 +/- 0.206 s | 319; 9.896 +/- 0.023 s | 215; 4.572 +/- 0.044 s | 33; 0.256 +/- 0.009 s |
| 1024 | 4 | 1e-08 | 330; 10.584 +/- 0.392 s | 401 cap; 15.336 +/- 0.236 s | 401 cap; 15.415 +/- 0.222 s | 330; 10.453 +/- 0.403 s | 401 cap; 15.244 +/- 0.260 s | 401 cap; 15.345 +/- 0.218 s | 401 cap; 15.411 +/- 0.186 s | 401 cap; 15.270 +/- 0.261 s | 55; 0.456 +/- 0.015 s |
| 1024 | 8 | 1e-02 | 170; 2.951 +/- 0.047 s | 119; 1.555 +/- 0.011 s | 19; 0.162 +/- 0.003 s | 170; 2.929 +/- 0.048 s | 67; 0.584 +/- 0.022 s | 89; 0.931 +/- 0.045 s | 40; 0.298 +/- 0.025 s | 18; 0.199 +/- 0.034 s | 8; 0.155 +/- 0.041 s |
| 1024 | 8 | 1e-05 | 387; 14.359 +/- 0.323 s | 401 cap; 15.430 +/- 0.146 s | 296; 8.674 +/- 0.193 s | 387; 14.326 +/- 0.340 s | 394; 14.901 +/- 0.320 s | 401 cap; 15.491 +/- 0.451 s | 364; 13.070 +/- 0.133 s | 293; 8.425 +/- 0.126 s | 21; 0.221 +/- 0.010 s |
| 1024 | 8 | 1e-08 | 401 cap; 15.328 +/- 0.327 s | 401 cap; 15.501 +/- 0.170 s | 401 cap; 15.398 +/- 0.088 s | 401 cap; 16.978 +/- 1.482 s | 401 cap; 16.476 +/- 0.205 s | 401 cap; 16.625 +/- 0.167 s | 401 cap; 16.739 +/- 0.077 s | 401 cap; 16.385 +/- 0.181 s | 33; 0.286 +/- 0.020 s |

### Tight tolerance (`1e-8`)

| N | m | oracle d0* | blend | active-oracle/blend time | fastest converged comparator | fastest-comparator/blend time |
|---:|---:|---:|---:|---:|---|---:|
| 512 | 2 | 166; 2.711 +/- 0.048 s | 14; 0.169 +/- 0.020 s | 16.0x | floor: 111; 1.276 +/- 0.096 s | 7.5x |
| 512 | 4 | 245; 6.071 +/- 0.089 s | 15; 0.169 +/- 0.023 s | 35.9x | identity: 156; 2.542 +/- 0.022 s | 15.0x |
| 512 | 8 | 277; 7.270 +/- 0.120 s | 17; 0.189 +/- 0.007 s | 38.5x | identity: 197; 3.706 +/- 0.176 s | 19.6x |
| 1024 | 2 | 317; 9.759 +/- 0.111 s | 17; 0.181 +/- 0.033 s | 54.0x | floor: 228; 5.132 +/- 0.223 s | 28.4x |
| 1024 | 4 | 401 cap; 15.270 +/- 0.261 s | 55; 0.456 +/- 0.015 s | oracle capped | floor: 330; 10.453 +/- 0.403 s | 22.9x |
| 1024 | 8 | 401 cap; 16.385 +/- 0.181 s | 33; 0.286 +/- 0.020 s | oracle capped | none converged | -- |

The active-support oracle scan uses 51 logarithmically spaced values between
`D_min(active)` and `D_max(active)` for every case: 26 independent scans and 1326
candidate solves across the main and contrast studies. On the clean post-`d8d4432`
replay, the blend is 16.0x--54.0x faster than the converged active-range oracle in
the four cases where that oracle reaches tolerance. At `N=1024,m=4`, no
active-range scalar reaches tolerance, while identity/floor do; the blend takes
55 iterations and about 0.456 s versus 330 iterations and about 10.45 s for the
fastest converged comparator. At `N=1024,m=8`, no scalar, floor, or identity reaches
tolerance, while the blend takes 33 iterations and about 0.286 s.

These replayed values supersede the scratch Phase-0 timing headline
`18.4x--62.5x`: iteration counts for the active oracle mostly reproduce, but the
merged GMRES counter, independent rerun, and current timing samples yield the
ratios above. The committed JSON is authoritative.

### Loose tolerance (`1e-2`)

| N | m | fastest scalar | scalar time | blend time | blend delta |
|---:|---:|---|---:|---:|---:|
| 512 | 2 | const | 0.113 +/- 0.028 s | 0.130 +/- 0.005 s | +15.1% |
| 512 | 4 | oracle d0* | 0.143 +/- 0.038 s | 0.154 +/- 0.026 s | +7.3% |
| 512 | 8 | oracle d0* | 0.161 +/- 0.016 s | 0.158 +/- 0.018 s | -1.6% |
| 1024 | 2 | oracle d0* | 0.114 +/- 0.010 s | 0.137 +/- 0.007 s | +19.7% |
| 1024 | 4 | bulk | 0.142 +/- 0.004 s | 0.161 +/- 0.017 s | +13.4% |
| 1024 | 8 | bulk | 0.162 +/- 0.003 s | 0.155 +/- 0.041 s | -3.9% |

The result is regime-dependent: the blend is about 15--20% slower at `m=2`,
7--13% slower at `m=4`, and about 2--4% faster at `m=8` on this replay. The
`m=4` timing separation is not large relative to run-to-run variability. Six
cases are not a basis for a dispatch rule, and none is proposed; an earlier
resolution-aware-dispatch exploration was closed as a negative result.

## Contrast and degeneracy

| m | step | D95/D05 | degeneracy fraction | d0*/Dmax(active) | d0* empirical rank | oracle iterations | blend iterations | oracle/blend |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0 | 9.548 | 0.652 | 1.000 | 1.000 | 111 | 14 | 7.93x |
| 2 | 2 | 34.877 | 0.609 | 1.000 | 1.000 | 130 | 32 | 4.06x |
| 2 | 5 | 42.535 | 0.578 | 1.000 | 1.000 | 133 | 34 | 3.91x |
| 2 | 8 | 32.326 | 0.555 | 1.000 | 1.000 | 135 | 30 | 4.50x |
| 8 | 0 | 10.228 | 0.418 | 1.000 | 1.000 | 197 | 17 | 11.59x |
| 8 | 2 | 10.845 | 0.406 | 0.891 | 0.671 | 210 | 20 | 10.50x |
| 8 | 5 | 10.961 | 0.391 | 0.877 | 0.654 | 216 | 26 | 8.31x |
| 8 | 8 | 10.477 | 0.379 | 1.000 | 1.000 | 219 | 22 | 9.95x |

The iteration advantage shrinks as compact support fills in but does not vanish:
the measured oracle/blend ratio falls from 7.93x to 3.91--4.50x for `m=2`, and
from 11.59x to 8.31--10.50x for `m=8`. It tracks the zero-set/degeneracy fraction
more coherently than `D95/D05`. The oracle selects the active-support maximum for
all 18 initial-wide-front cases and six of eight contrast states. The other two
are still high in the active range (`0.877` and `0.891` of the active maximum),
not means. This explains why `frozen_bulk` is generally the strongest shipped
nontrivial scalar and `frozen_mean` the weakest in these states. It does not by
itself justify changing moljax's default.

## Batched implementation audit

The production `l=3` application was already batched. StableHLO contains one
Helmholtz call on a `3 x N` tensor and two transform invocations, versus three
Helmholtz calls and six transforms in the forced-sequential comparator. The two
actions agree to `1.01e-16` relative
error and all solve iteration sets are identical. On GPU, batching speeds up the
application itself by 1.70--1.98x; the end-to-end solve
speedup is only 0.918--1.079x because the application is not
the bottleneck at these sizes. Therefore the loose-tolerance deficit is not a
hidden sequential-transform bug. The `(l,N)` Helmholtz denominator is still
formed inside each application; hoisting it is an identified, unimplemented
optimization.

## Pseudospectral criterion

The dense criterion is mathematically sound for these validation-sized operators.
Its Trefethen resolvent-contour prefactor is exactly
`L(Gamma_epsilon)/(2*pi*epsilon)`; no Crouzeix spectral-set constant belongs in
that bound. It certifies the tested origin-enclosed blend cases, but certification
is offline: the matrix-free runtime estimator costs 58.0--73.9x
the corresponding solve, a fundamental evidence floor rather than an omitted
batching optimization. As a rejection gate it can be cheap: rejecting the
inadequate `m=8` frozen-mean reference costs 0.32x its solve. Sparse
matrix-free Ritz/path mode remains provisional and records
`rate_bound_available=false`, because reduced Ritz values do not prove full
spectrum coverage and sparse paths do not provide a closed-contour arc length.

## Honest limits

- No multigrid baseline; therefore no performance claim.
- One-dimensional, node-centred, homogeneous-Dirichlet PME only, at `N=512/1024`.
- The main linear study measures one backward-Euler step; inexact-Newton traces
  are supporting evidence, not a full time-to-solution application benchmark.
- Loose-tolerance behavior is regime-dependent, and no switching heuristic is
  proposed.
- The advantage shrinks as the zero set fills in, although it does not vanish in
  the measured contrast sweep.
- A two-dimensional extension is a major build: moljax's `Grid2D` is cell-centred,
  there is no matching 2-D DST-I path, the flux-form linearization differs, and a
  PME front is curve-shaped. It was not attempted here.
- The dense pseudospectral certificate is intentionally offline at `N=512`; its
  sparse matrix-free approximation is a provisional diagnostic only.

## Reproduction and provenance

Run the benchmark entry points with `PYTHONPATH="$PWD"` and Python x64 enabled.
The canonical tables come from
`benchmarks/pme_dst_tau_blend_single_reference_baselines.py`; the batching audit,
dense/sparse pseudospectral studies, validation, spectral analysis, and
inexact-Newton scripts provide the supporting JSONs. Source states use a v4
generation fingerprint, relocatable cache-relative path, and SHA256 identity;
foreign or damaged artifacts fail closed. Provenance resolves
`merge-base upstream/main -> git describe --tags -> HEAD -> unavailable` and
labels the source. Reassessment reconstructs the stored grid, state, solver, and
preconditioner configuration before accepting an artifact.

Figures are regenerated from the JSONs only:

```bash
PYTHONPATH="$PWD" conda run -n moljax python benchmarks/make_tau_blend_figures.py
```

## References (DOI verified)

1. G. Pavlov and G. Vourvachakis, "moljax: GPU-accelerated method of lines for
   stiff reaction-diffusion PDEs with FFT preconditioning," *Computer Physics
   Communications* 326 (2026) 110205. DOI `10.1016/j.cpc.2026.110205`.
2. C. C. Christara and K. S. Ng, "Fast Fourier Transform Solvers and
   Preconditioners for Quadratic Spline Collocation," *BIT Numerical
   Mathematics* 42(4) (2002) 702--739. DOI `10.1023/A:1021944218806`.
3. M.-C. Lai and Y.-H. Tseng, "A fast iterative solver for the variable
   coefficient diffusion equation on a disk," *Journal of Computational Physics*
   208 (2005) 196--205. DOI `10.1016/j.jcp.2005.02.005`.
4. P. De Luca, "Fast Sine-Transform Preconditioning for Global-in-Time
   Fractional Diffusion," *Fractal and Fractional* 10(8) (2026) 573. DOI
   `10.3390/fractalfract10080573`.
5. X. Lin, C. Li, and S. Y. Hon, "Absolute-value based preconditioner for
   complex-shifted Laplacian systems," arXiv:2408.00488 (2024). DOI
   `10.48550/arXiv.2408.00488`.
6. T. Bakhos, P. K. Kitanidis, S. Ladenheim, A. K. Saibaba, and D. B. Szyld,
   "Multipreconditioned GMRES for Shifted Systems," *SIAM Journal on Scientific
   Computing* 39(5) (2017) S222--S247. DOI `10.1137/16M1068694`; preprint DOI
   `10.48550/arXiv.1603.08970`.
7. A. Heryudono, E. Larsson, A. Ramage, and L. von Sydow, "Preconditioning for
   Radial Basis Function Partition of Unity Methods," *Journal of Scientific
   Computing* 67 (2016) 1089--1109. DOI `10.1007/s10915-015-0120-6`.
8. M. Bonazzoli, X. Claeys, F. Nataf, and P.-H. Tournier, "Analysis of the SORAS
   domain decomposition preconditioner for non-self-adjoint or indefinite
   problems," *Journal of Scientific Computing* 89 (2021) 19. DOI
   `10.1007/s10915-021-01631-8`.
9. L. N. Trefethen and M. Embree, *Spectra and Pseudospectra* (Princeton, 2005).
   DOI `10.1515/9780691213101`.
10. M. Embree, "How Descriptive are GMRES Convergence Bounds?", corrected and
    extended from Oxford Technical Report 99/08, arXiv:2209.01231. DOI
    `10.48550/arXiv.2209.01231`. There is no SIAM J. Matrix Analysis and
    Applications version.
