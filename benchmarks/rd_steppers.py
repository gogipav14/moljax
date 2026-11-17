"""Time steppers for the two-field reaction-diffusion benchmarks.

System (periodic, pseudo-spectral Laplacian with eigenvalues ``lap_eig``):

    du/dt = Du * lap(u) + Ru(u, v)
    dv/dt = Dv * lap(v) + Rv(u, v)

The benchmark scripts had one hand-written copy of each stepper. Three of
the copies were wrong:

1. ETDRK4: the final update used ``f2 * (Na + Nb)``. Cox and Matthews
   (2002) and Kassam and Trefethen (2005) have ``2 * f2 * (Na + Nb)``.
   Without the factor the four reaction weights sum to 4h/6, not h, so the scheme was
   not consistent.
2. IMEX-Strang: the reaction half steps were forward Euler, which made the
   composition first order.
3. Crank-Nicolson: the "Newton" loop was the fixed-point iteration
   ``w <- w - P^-1 F(w)``. It diverges when the spectral radius of
   ``(dt/2) P^-1 J_R`` is above one (Schnakenberg at dt = 0.01).

This module has one implementation of each stepper. The tests in
``tests/test_rd_benchmark_steppers.py`` measure the order of each one.

All functions take and return real arrays ``u`` and ``v`` of shape (N, N).
"""

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

Reaction = Callable[[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]

# Reaction evaluations for each step of the explicit and split steppers.
NFE_PER_STEP = {"rk4": 4, "imex_strang": 2, "etdrk4": 4}


def _fft2(x):
    return jnp.fft.fft2(x)


def _ifft2_real(x_hat):
    return jnp.real(jnp.fft.ifft2(x_hat))


# ---------------------------------------------------------------------------
# RK4
# ---------------------------------------------------------------------------
def make_rk4_step(lap_eig, Du: float, Dv: float, reaction: Reaction):
    """Classical RK4 on the full right-hand side. Returns step(u, v, dt)."""
    lap = jnp.asarray(lap_eig)

    def rhs(u, v):
        ru, rv = reaction(u, v)
        return (Du * _ifft2_real(lap * _fft2(u)) + ru,
                Dv * _ifft2_real(lap * _fft2(v)) + rv)

    @jax.jit
    def step(u, v, dt):
        k1u, k1v = rhs(u, v)
        k2u, k2v = rhs(u + 0.5 * dt * k1u, v + 0.5 * dt * k1v)
        k3u, k3v = rhs(u + 0.5 * dt * k2u, v + 0.5 * dt * k2v)
        k4u, k4v = rhs(u + dt * k3u, v + dt * k3v)
        return (u + dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u),
                v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v))

    return step


# ---------------------------------------------------------------------------
# IMEX-Strang: D(h/2) Heun(h) D(h/2)
# ---------------------------------------------------------------------------
def make_imex_strang_step(lap_eig, Du: float, Dv: float, reaction: Reaction, dt: float):
    """Strang splitting, second order. Returns step(u, v).

    1. Half diffusion step, exact in Fourier space.
    2. Full reaction step with Heun's method (two reaction evaluations).
    3. Half diffusion step, exact in Fourier space.

    This is the scheme of ``moljax.core.stepping.imex_strang_step``.
    """
    half_u = jnp.asarray(np.exp(0.5 * dt * Du * np.asarray(lap_eig)))
    half_v = jnp.asarray(np.exp(0.5 * dt * Dv * np.asarray(lap_eig)))

    @jax.jit
    def step(u, v):
        u = _ifft2_real(half_u * _fft2(u))
        v = _ifft2_real(half_v * _fft2(v))
        r1u, r1v = reaction(u, v)
        r2u, r2v = reaction(u + dt * r1u, v + dt * r1v)
        u = u + 0.5 * dt * (r1u + r2u)
        v = v + 0.5 * dt * (r1v + r2v)
        u = _ifft2_real(half_u * _fft2(u))
        v = _ifft2_real(half_v * _fft2(v))
        return u, v

    return step


def make_imex_strang_solve(lap_eig, Du: float, Dv: float, reaction: Reaction, dt: float):
    """Compiled solve(u0, v0, n_steps) for IMEX-Strang with combined diffusion steps.

    Two adjacent half diffusion steps are one full diffusion step, so n steps
    are: D(h/2), then n - 1 times [Heun(h), D(h)], then Heun(h), D(h/2).
    The result is the same as n calls of ``make_imex_strang_step`` up to
    rounding. The cost is one FFT pair for each step, not two.
    """
    lap = np.asarray(lap_eig)
    half_u, half_v = jnp.asarray(np.exp(0.5 * dt * Du * lap)), jnp.asarray(np.exp(0.5 * dt * Dv * lap))
    full_u, full_v = jnp.asarray(np.exp(dt * Du * lap)), jnp.asarray(np.exp(dt * Dv * lap))

    def heun(u, v):
        r1u, r1v = reaction(u, v)
        r2u, r2v = reaction(u + dt * r1u, v + dt * r1v)
        return u + 0.5 * dt * (r1u + r2u), v + 0.5 * dt * (r1v + r2v)

    def diffuse(u, v, eu, ev):
        return _ifft2_real(eu * _fft2(u)), _ifft2_real(ev * _fft2(v))

    @jax.jit
    def solve(u0, v0, n_steps):
        u, v = diffuse(u0, v0, half_u, half_v)
        u, v = jax.lax.fori_loop(
            0, n_steps - 1, lambda _, uv: diffuse(*heun(*uv), full_u, full_v), (u, v))
        return diffuse(*heun(u, v), half_u, half_v)

    return solve


# ---------------------------------------------------------------------------
# ETDRK4 (Cox and Matthews 2002; contour form of Kassam and Trefethen 2005)
# ---------------------------------------------------------------------------
def etdrk4_coefficients(L, dt: float, M: int = 32):
    """Return E, E2, Q, f1, f2, f3 for the linear eigenvalues ``L`` (real).

    The final update is
    ``E u + f1 N(u) + 2 f2 (N(a) + N(b)) + f3 N(c)``.
    For L -> 0: Q -> dt/2 and f1, f2, f3 -> dt/6.
    """
    L = np.asarray(L, dtype=np.float64)
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * L[..., np.newaxis] + r
    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2)
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    return E, E2, Q, f1, f2, f3


def make_etdrk4_step(lap_eig, Du: float, Dv: float, reaction: Reaction, dt: float):
    """ETDRK4, fourth order. Returns step(u, v)."""
    lap = np.asarray(lap_eig)
    Eu, E2u, Qu, f1u, f2u, f3u = [jnp.asarray(x) for x in etdrk4_coefficients(Du * lap, dt)]
    Ev, E2v, Qv, f1v, f2v, f3v = [jnp.asarray(x) for x in etdrk4_coefficients(Dv * lap, dt)]

    def n_hat(u, v):
        ru, rv = reaction(u, v)
        return _fft2(ru), _fft2(rv)

    @jax.jit
    def step(u, v):
        uh, vh = _fft2(u), _fft2(v)
        nu, nv = n_hat(u, v)

        au, av = E2u * uh + Qu * nu, E2v * vh + Qv * nv
        nau, nav = n_hat(_ifft2_real(au), _ifft2_real(av))

        bu, bv = E2u * uh + Qu * nau, E2v * vh + Qv * nav
        nbu, nbv = n_hat(_ifft2_real(bu), _ifft2_real(bv))

        cu, cv = E2u * au + Qu * (2 * nbu - nu), E2v * av + Qv * (2 * nbv - nv)
        ncu, ncv = n_hat(_ifft2_real(cu), _ifft2_real(cv))

        uh_new = Eu * uh + f1u * nu + 2 * f2u * (nau + nbu) + f3u * ncu
        vh_new = Ev * vh + f1v * nv + 2 * f2v * (nav + nbv) + f3v * ncv
        return _ifft2_real(uh_new), _ifft2_real(vh_new)

    return step


# ---------------------------------------------------------------------------
# Crank-Nicolson with Newton-GMRES
# ---------------------------------------------------------------------------
class CNStats(NamedTuple):
    """Counts for one or more Crank-Nicolson steps (they add)."""
    newton_iters: jnp.ndarray   # Newton iterations
    krylov_iters: jnp.ndarray   # Jacobian-vector products in GMRES
    residual_evals: jnp.ndarray  # evaluations of the nonlinear residual
    failed_steps: jnp.ndarray   # steps that stopped above the tolerance
    max_residual: jnp.ndarray   # largest final residual (max norm)

    @staticmethod
    def zero():
        z = jnp.zeros((), dtype=jnp.int64 if jax.config.jax_enable_x64 else jnp.int32)
        return CNStats(z, z, z, z, jnp.zeros(()))

    def add(self, other: "CNStats") -> "CNStats":
        return CNStats(
            self.newton_iters + other.newton_iters,
            self.krylov_iters + other.krylov_iters,
            self.residual_evals + other.residual_evals,
            self.failed_steps + other.failed_steps,
            jnp.maximum(self.max_residual, other.max_residual),
        )

    def nfe(self):
        """Reaction evaluations: residuals, Jacobian products, one explicit part for each step."""
        return self.residual_evals + self.krylov_iters


def _gmres(matvec, b, m: int, tol_abs):
    """GMRES without restart, at most ``m`` iterations, zero first guess.

    Stops when the 2-norm of the linear residual is at most ``tol_abs``.
    Returns (x, iterations, residual norm). ``b`` has any shape.
    """
    beta = jnp.linalg.norm(b)
    V0 = jnp.zeros((m + 1,) + b.shape, dtype=b.dtype).at[0].set(b / jnp.where(beta > 0, beta, 1.0))
    H0 = jnp.zeros((m + 1, m), dtype=b.dtype)
    cs0 = jnp.ones((m,), dtype=b.dtype)
    sn0 = jnp.zeros((m,), dtype=b.dtype)
    g0 = jnp.zeros((m + 1,), dtype=b.dtype).at[0].set(beta)
    axes = tuple(range(1, b.ndim + 1))

    def cond(state):
        _, _, _, _, g, k = state
        return (k < m) & (jnp.abs(g[k]) > tol_abs)

    def body(state):
        V, H, cs, sn, g, k = state
        w = matvec(V[k])
        # Modified Gram-Schmidt against all vectors; the rows above k are zero.
        def mgs(j, carry):
            w, h = carry
            hj = jnp.where(j <= k, jnp.sum(V[j] * w), 0.0)
            return w - hj * V[j], h.at[j].set(hj)
        w, h = jax.lax.fori_loop(0, m, mgs, (w, jnp.zeros((m + 1,), dtype=b.dtype)))
        # One pass of reorthogonalization keeps the basis orthogonal.
        corr = jnp.where(jnp.arange(m + 1) <= k, jnp.sum(V * w[None], axis=axes), 0.0)
        w = w - jnp.tensordot(corr, V, axes=1)
        h = h + corr
        hk1 = jnp.linalg.norm(w)
        h = h.at[k + 1].set(hk1)
        V = V.at[k + 1].set(w / jnp.where(hk1 > 0, hk1, 1.0))

        # Apply the previous Givens rotations to the new column.
        def rot(j, h):
            t = cs[j] * h[j] + sn[j] * h[j + 1]
            hj1 = -sn[j] * h[j] + cs[j] * h[j + 1]
            apply = j < k
            return h.at[j].set(jnp.where(apply, t, h[j])).at[j + 1].set(jnp.where(apply, hj1, h[j + 1]))
        h = jax.lax.fori_loop(0, m, rot, h)
        denom = jnp.sqrt(h[k] ** 2 + h[k + 1] ** 2)
        denom = jnp.where(denom > 0, denom, 1.0)
        c, s = h[k] / denom, h[k + 1] / denom
        h = h.at[k].set(c * h[k] + s * h[k + 1]).at[k + 1].set(0.0)
        g = g.at[k + 1].set(-s * g[k]).at[k].set(c * g[k])
        return V, H.at[:, k].set(h), cs.at[k].set(c), sn.at[k].set(s), g, k + 1

    V, H, _, _, g, k = jax.lax.while_loop(cond, body, (V0, H0, cs0, sn0, g0, 0))
    # Solve the k-by-k triangular system; the unused part is the identity.
    idx = jnp.arange(m)
    used = idx < k
    R = jnp.where(used[:, None] & used[None, :], H[:m, :], jnp.eye(m, dtype=b.dtype))
    y = jax.scipy.linalg.solve_triangular(R, jnp.where(used, g[:m], 0.0), lower=False)
    x = jnp.tensordot(y, V[:m], axes=1)
    return x, k, jnp.abs(g[k])


def make_cn_step(
    lap_eig, Du: float, Dv: float, reaction: Reaction, dt: float,
    newton_tol: float = 1e-8, max_newton: int = 20,
    krylov_dim: int = 30, krylov_rtol: float = 1e-3, max_backtrack: int = 8,
    preconditioner: str = "diffusion",
):
    """Crank-Nicolson with a Newton-GMRES solve. Returns step(u, v) -> (u, v, CNStats).

    The residual for the new state w = (u, v) is
    ``F(w) = w - w_old - (dt/2) (D lap w + R(w)) - (dt/2) (D lap w_old + R(w_old))``.
    Each Newton iteration solves ``J dw = -F`` with GMRES. The Jacobian
    product comes from ``jax.linearize``. The right preconditioner is applied
    with the FFT. ``preconditioner="diffusion"`` is the exact inverse of the
    diffusion part, ``(I - (dt/2) D lap)^-1``. ``preconditioner="mean_field"``
    also includes the reaction Jacobian at the mean state (a 2 x 2 solve for
    each Fourier mode).

    ``newton_tol`` is compared with ``max|F_u| + max|F_v|`` (the norm of the
    earlier benchmark code). The first guess is the old state. A backtracking
    line search halves the Newton step until the residual 2-norm decreases
    (at most ``max_backtrack`` times). A step that
    does not reach the tolerance in ``max_newton`` iterations counts as a
    failed step; the caller must check ``CNStats.failed_steps``.
    """
    lap = jnp.asarray(lap_eig)
    D = jnp.stack([Du * lap, Dv * lap])            # (2, N, N)
    precond = 1.0 / (1.0 - 0.5 * dt * D)

    def explicit_part(w):
        ru, rv = reaction(w[0], w[1])
        return jnp.real(jnp.fft.ifft2(D * jnp.fft.fft2(w))) + jnp.stack([ru, rv])

    def apply_diffusion_precond(x):
        return jnp.real(jnp.fft.ifft2(precond * jnp.fft.fft2(x)))

    def make_mean_field_precond(w):
        """Exact inverse of I - (dt/2) (D lap + J_R) with J_R at the mean state.

        This is a 2 x 2 solve for each Fourier mode. It is the exact inverse
        of the Jacobian when the state is uniform.
        """
        mean = jnp.mean(w, axis=(1, 2))
        jac = jax.jacfwd(lambda m: jnp.stack(reaction(m[0], m[1])))(mean)
        h = 0.5 * dt
        m11 = 1.0 - h * (D[0] + jac[0, 0])
        m22 = 1.0 - h * (D[1] + jac[1, 1])
        m12, m21 = -h * jac[0, 1], -h * jac[1, 0]
        det = m11 * m22 - m12 * m21

        def apply(x):
            xh = jnp.fft.fft2(x)
            yu = (m22 * xh[0] - m12 * xh[1]) / det
            yv = (m11 * xh[1] - m21 * xh[0]) / det
            return jnp.real(jnp.fft.ifft2(jnp.stack([yu, yv])))
        return apply

    def res_norm(F):
        return jnp.max(jnp.abs(F[0])) + jnp.max(jnp.abs(F[1]))

    def step(u, v):
        w_old = jnp.stack([u, v])
        rhs_old = w_old + 0.5 * dt * explicit_part(w_old)

        def residual(w):
            return w - 0.5 * dt * explicit_part(w) - rhs_old

        F0 = residual(w_old)

        def cond(state):
            _, F, it, _, _ = state
            return (res_norm(F) > newton_tol) & (it < max_newton)

        def body(state):
            w, F, it, kry, evals = state
            apply_precond = (make_mean_field_precond(w) if preconditioner == "mean_field"
                             else apply_diffusion_precond)
            _, jvp_at_w = jax.linearize(residual, w)
            dy, k, _ = _gmres(lambda y: jvp_at_w(apply_precond(y)), -F,
                              krylov_dim, krylov_rtol * jnp.linalg.norm(F))
            dw = apply_precond(dy)
            f_norm = jnp.linalg.norm(F)

            # Backtracking: halve the step until the residual 2-norm decreases.
            def ls_cond(ls):
                alpha, _, F_try, n_try = ls
                bad = ~(jnp.linalg.norm(F_try) < (1.0 - 1e-4 * alpha) * f_norm)
                return bad & (n_try < max_backtrack)

            def ls_body(ls):
                alpha, _, _, n_try = ls
                alpha = 0.5 * alpha
                w_try = w + alpha * dw
                return alpha, w_try, residual(w_try), n_try + 1

            w_full = w + dw
            _, w_new, F_new, n_try = jax.lax.while_loop(
                ls_cond, ls_body, (jnp.ones(()), w_full, residual(w_full), 0))
            return w_new, F_new, it + 1, kry + k, evals + 2 + n_try

        w, F, it, kry, evals = jax.lax.while_loop(cond, body, (w_old, F0, 0, 0, 2))
        final = res_norm(F)
        zero = CNStats.zero()
        stats = CNStats(
            newton_iters=zero.newton_iters + it,
            krylov_iters=zero.krylov_iters + kry,
            # one explicit part for rhs_old, one residual at the first guess;
            # for each iteration one linearization point, one new residual
            # and one residual for each backtracking trial
            residual_evals=zero.residual_evals + evals,
            failed_steps=zero.failed_steps + jnp.where(final > newton_tol, 1, 0),
            max_residual=jnp.where(jnp.isfinite(final), final, jnp.inf),
        )
        return w[0], w[1], stats

    return step


def make_cn_solve(cn_step):
    """Return a compiled solve(u0, v0, n_steps) -> (u, v, CNStats, steps_done).

    The loop stops after the first step that does not reach the Newton
    tolerance, so ``steps_done < n_steps`` shows a failed run.
    """
    @jax.jit
    def solve(u0, v0, n_steps):
        def cond(carry):
            _, _, stats, i = carry
            return (i < n_steps) & (stats.failed_steps == 0)

        def body(carry):
            u, v, stats, i = carry
            u, v, s = cn_step(u, v)
            return u, v, stats.add(s), i + 1

        return jax.lax.while_loop(cond, body, (u0, v0, CNStats.zero(), 0))
    return solve


def make_solve(step):
    """Return a compiled solve(u0, v0, n_steps) for a step(u, v) function."""
    @jax.jit
    def solve(u0, v0, n_steps):
        return jax.lax.fori_loop(0, n_steps, lambda _, uv: step(*uv), (u0, v0))
    return solve


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
def field_summary(u, v) -> dict:
    """Finiteness and statistics of a final state, for the result files."""
    u, v = np.asarray(u), np.asarray(v)
    finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
    out = {"finite": finite}
    if finite:
        for name, x in (("u", u), ("v", v)):
            out[f"{name}_min"] = float(x.min())
            out[f"{name}_max"] = float(x.max())
            out[f"{name}_mean"] = float(x.mean())
            out[f"{name}_std"] = float(x.std())
    return out


def cn_stats_summary(stats: CNStats, n_steps: int) -> dict:
    """CNStats as a dictionary for the result files."""
    return {
        "newton_iters": int(stats.newton_iters),
        "krylov_iters": int(stats.krylov_iters),
        "nfe": int(stats.nfe()),
        "nfe_per_step": float(stats.nfe()) / n_steps,
        "newton_per_step": float(stats.newton_iters) / n_steps,
        "krylov_per_newton": float(stats.krylov_iters) / max(int(stats.newton_iters), 1),
        "failed_steps": int(stats.failed_steps),
        "max_residual": float(stats.max_residual),
    }


# ---------------------------------------------------------------------------
# Benchmark driver (the three benchmark scripts call this)
# ---------------------------------------------------------------------------
def env_float(name: str, default: float) -> float:
    """Value of an environment variable as a float, or the default."""
    return float(os.environ.get(name, default))


def _timed(run, n_reps: int):
    """Call run() n_reps times. Return (median, IQR, last result)."""
    times, out = [], None
    for _ in range(n_reps):
        start = time.perf_counter()
        out = run()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    q25, q75 = np.percentile(times, [25, 75])
    return float(np.median(times)), float(q75 - q25), out


def run_benchmark(
    system: str, lap_eig, Du: float, Dv: float, reaction: Reaction, u0, v0,
    t_final: float, dt_rk4: float, dt_cn: float, dt_imex: float, dt_etd: float,
    n_reps: int, rk4_short_steps: int | None = None, config: dict | None = None,
    cn_newton_tol: float = 1e-8,
):
    """Time RK4, Crank-Nicolson, IMEX-Strang and ETDRK4 and write the result file.

    Every method runs in one compiled loop. The first call of each loop
    (10 steps) is the warm-up and is not timed. ``rk4_short_steps`` gives
    the length of a timed RK4 burst; the full RK4 time is then an
    extrapolation. Environment variables: RD_DT_CN, RD_DT_IMEX, RD_DT_ETD,
    RD_N_REPS, RD_T_FINAL, RD_OUT_DIR, RD_TAG (a suffix for the file names)
    and RD_SKIP (a comma list of rk4, cn, imex, etdrk4 to omit).
    """
    t_final = env_float("RD_T_FINAL", t_final)
    dt_cn = env_float("RD_DT_CN", dt_cn)
    dt_imex = env_float("RD_DT_IMEX", dt_imex)
    dt_etd = env_float("RD_DT_ETD", dt_etd)
    n_reps = int(os.environ.get("RD_N_REPS", n_reps))
    tag = os.environ.get("RD_TAG", "")
    skip = {x.strip() for x in os.environ.get("RD_SKIP", "").split(",") if x.strip()}
    out_dir = Path(os.environ.get("RD_OUT_DIR", Path(__file__).parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    u0, v0 = jnp.asarray(u0), jnp.asarray(v0)
    methods, finals = [], {}

    def record(name, dt, steps, median, iqr, u, v, nfe, extra=None):
        entry = {
            "name": name, "time_s": median, "iqr_s": iqr, "steps": int(steps),
            "dt": float(dt), "t_final": float(steps * dt), "nfe": int(nfe),
        }
        entry.update(field_summary(u, v))
        entry.update(extra or {})
        methods.append(entry)
        finals[name] = (np.asarray(u), np.asarray(v))
        print(f"   {name}: {median:.2f} +/- {iqr:.3f} s, steps {int(steps):,}, "
              f"dt {dt:g}, NFE {int(nfe):,}, finite {entry['finite']}", flush=True)
        if not entry["finite"]:
            print(f"   WARNING: {name} is not finite. The time is not valid.", flush=True)

    if "rk4" not in skip:
        # 1. RK4
        print("\n1. RK4 (explicit)...", flush=True)
        rk4 = make_rk4_step(lap_eig, Du, Dv, reaction)
        solve = make_solve(lambda u, v: rk4(u, v, dt_rk4))
        n_full = int(np.ceil(t_final / dt_rk4))
        n_run = n_full if rk4_short_steps is None else min(rk4_short_steps, n_full)
        jax.block_until_ready(solve(u0, v0, 10))
        median, iqr, (u, v) = _timed(lambda: solve(u0, v0, n_run), n_reps)
        scale = n_full / n_run
        record("RK4 (explicit)", dt_rk4, n_full, median * scale, iqr * scale, u, v,
               NFE_PER_STEP["rk4"] * n_full,
               {"extrapolated": n_run < n_full, "measured_steps": n_run,
                "measured_time_s": median, "state_is_at_t": n_run * dt_rk4})

    if "cn" not in skip:
        # 2. Crank-Nicolson
        print("\n2. CN (Newton-GMRES, FFT preconditioner)...", flush=True)
        n_cn = int(np.ceil(t_final / dt_cn))
        cn_solve = make_cn_solve(make_cn_step(lap_eig, Du, Dv, reaction, dt_cn, newton_tol=cn_newton_tol))
        jax.block_until_ready(cn_solve(u0, v0, 10))
        start = time.perf_counter()
        u, v, stats, done = cn_solve(u0, v0, n_cn)
        jax.block_until_ready(u)
        first = time.perf_counter() - start
        if int(done) < n_cn or int(stats.failed_steps) > 0:
            print(f"   CN FAILED: Newton did not converge at step {int(done)} of {n_cn:,} "
                  f"(residual {float(stats.max_residual):.3e}). No time is recorded.", flush=True)
            entry = {"name": "CN (Newton-Krylov)", "time_s": None, "iqr_s": None, "steps": n_cn,
                     "dt": float(dt_cn), "t_final": float(n_cn * dt_cn), "nfe": None,
                     "valid": False, "steps_done": int(done), "time_to_failure_s": first}
            entry.update(field_summary(u, v))
            entry.update(cn_stats_summary(stats, max(int(done), 1)))
            methods.append(entry)
        else:
            times = [first]
            for _ in range(n_reps - 1):
                start = time.perf_counter()
                u, v, stats, done = cn_solve(u0, v0, n_cn)
                jax.block_until_ready(u)
                times.append(time.perf_counter() - start)
            q25, q75 = np.percentile(times, [25, 75])
            median, iqr = float(np.median(times)), float(q75 - q25)
            extra = {"valid": True, "newton_tol": cn_newton_tol}
            extra.update(cn_stats_summary(stats, n_cn))
            record("CN (Newton-Krylov)", dt_cn, n_cn, median, iqr, u, v, int(stats.nfe()), extra)

    if "imex" not in skip:
        # 3. IMEX-Strang
        print("\n3. IMEX-Strang (half diffusion, Heun reaction, half diffusion; combined)...", flush=True)
        n_imex = int(np.ceil(t_final / dt_imex))
        solve = make_imex_strang_solve(lap_eig, Du, Dv, reaction, dt_imex)
        jax.block_until_ready(solve(u0, v0, 10))
        median, iqr, (u, v) = _timed(lambda: solve(u0, v0, n_imex), n_reps)
        record("IMEX-Strang", dt_imex, n_imex, median, iqr, u, v, NFE_PER_STEP["imex_strang"] * n_imex)

    if "etdrk4" not in skip:
        # 4. ETDRK4
        print("\n4. ETDRK4...", flush=True)
        n_etd = int(np.ceil(t_final / dt_etd))
        solve = make_solve(make_etdrk4_step(lap_eig, Du, Dv, reaction, dt_etd))
        jax.block_until_ready(solve(u0, v0, 10))
        median, iqr, (u, v) = _timed(lambda: solve(u0, v0, n_etd), n_reps)
        record("ETDRK4", dt_etd, n_etd, median, iqr, u, v, NFE_PER_STEP["etdrk4"] * n_etd)

    results = {"methods": methods, "config": dict(config or {})}
    results["config"].update({
        "system": system, "t_final": t_final, "n_reps": n_reps,
        "dtype": "float64" if jax.config.jax_enable_x64 else "float32",
        "device": str(jax.devices()[0]), "jax_version": jax.__version__,
        "laplacian": "pseudo-spectral (-k^2)",
        "loop": "one compiled loop for each method; warm-up of 10 steps not timed",
        "steppers": "benchmarks/rd_steppers.py",
    })
    stem = system.lower().replace("-", "_").replace(" ", "_") + tag
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / f"{stem}.csv", "w") as f:
        keys = ["name", "dt", "steps", "time_s", "iqr_s", "nfe", "finite", "valid",
                "failed_steps", "u_min", "u_max", "u_mean", "u_std", "v_min", "v_max", "v_mean", "v_std"]
        f.write(",".join(keys) + "\n")
        for m in methods:
            f.write(",".join("" if m.get(k) is None else str(m.get(k)) for k in keys) + "\n")
    if "ETDRK4" in finals:
        np.save(out_dir / f"{stem}_u_final_etd.npy", finals["ETDRK4"][0])
        np.save(out_dir / f"{stem}_v_final_etd.npy", finals["ETDRK4"][1])

    print("\n" + "=" * 96)
    print(f"SUMMARY - {system}, {results['config']['dtype']}, {results['config']['device']}")
    print("=" * 96)
    print(f"{'Method':<20} {'Steps':>12} {'dt':>12} {'Time (s)':>18} {'NFE':>14} {'Finite':>7}")
    for m in methods:
        t = "failed" if m["time_s"] is None else f"{m['time_s']:.2f} +/- {m['iqr_s']:.2f}"
        nfe = "-" if m["nfe"] is None else f"{m['nfe']:,}"
        print(f"{m['name']:<20} {m['steps']:>12,} {m['dt']:>12.6g} {t:>18} {nfe:>14} {str(m['finite']):>7}")
    print("=" * 96)
    print(f"Results saved to {out_dir / (stem + '.json')}")
    return results
