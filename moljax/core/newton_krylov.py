"""
Newton-Krylov solver for implicit time stepping in MOL-JAX.

This module provides a JIT-compatible Newton-Krylov solver using:
- JVP-based matrix-free Jacobian-vector products
- GMRES or BiCGSTAB for inner linear solves
- Damping/backtracking line search for robustness
- lax.while_loop for Newton iterations (JIT-friendly)

Design decisions:
- All Newton iterations bounded by max_iters (static for JIT)
- Backtracking uses bounded tries (static for JIT)
- Residual and JVP operate on StateDict PyTrees
- Flatten/unflatten bridges PyTree to flat vector for Krylov
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

from moljax.core.grid import GridType
from moljax.core.preconditioners import IdentityPreconditioner, PrecondContext, Preconditioner
from moljax.core.state import StateDict, tree_add, tree_scale, tree_sub

if TYPE_CHECKING:
    from moljax.core.model import MOLModel


class NKParams(NamedTuple):
    """Newton-Krylov solver parameters.

    newton_tol is compared with the unweighted 2-norm of the flattened
    residual (all fields, all points, no grid weighting), so it scales
    with sqrt(number of unknowns). max_krylov_iters is the total Krylov
    budget per Newton step: the number of matrix-vector products GMRES is
    allowed, across every restart cycle combined.

    jax.scipy.sparse.linalg.gmres itself does not take a total matvec
    budget; it takes a Krylov subspace size ("restart", 20 by default) and
    a cycle count ("maxiter", rebuilding that subspace from the previous
    cycle's result each time). _gmres_solve derives both from
    max_krylov_iters and restart:

        restart_used = min(max_krylov_iters, restart)
        maxiter = ceil(max_krylov_iters / restart_used)

    so the enforced budget restart_used * maxiter equals max_krylov_iters
    exactly whenever max_krylov_iters <= restart or is a multiple of it,
    and otherwise overshoots by fewer than restart_used matvecs (GMRES
    also stops early on its own convergence test, so this is a ceiling,
    not a target). Before this derivation existed, max_krylov_iters was
    passed straight through as gmres's maxiter with restart left at its
    default of 20, so the true budget was 20x the documented value and
    there was no way to cap the Krylov space itself below 20 (see
    NKStats.lin_iters for what the reported statistic means).

    restart is the Krylov subspace dimension gmres builds per cycle
    (gmres's own "restart" parameter; default 20 to match it), capped by
    max_krylov_iters so a budget smaller than the default restart size
    still limits the subspace built in a single cycle instead of being
    silently widened back out to 20.

    max_backtrack = 0 means "no line search": the damped step
    x + damping * dx is applied unconditionally every Newton iteration,
    with no candidate ever checked against the Armijo-style decrease test.
    This differs from max_backtrack = 1, which does check that one
    candidate and falls back to the starting iterate if it fails.

    max_backtrack defaults to 8, not 3. With backtrack_factor = 0.5, depth
    d reaches alpha = 2 ** -(d - 1) before the line search gives up on a
    step: 3 stops at 0.25, 6 at 0.03125, 8 at 0.0078125. A stiff wide-front
    porous-medium solve stalled at the old default of 3 (the accepted
    alpha never got below 0.25) and converged once max_backtrack was
    raised to 6; 8 leaves one more binary decade of margin past that
    reading. backtrack_step stops evaluating candidates once one is
    accepted, so a step that accepts at depth j costs j residual
    evaluations regardless of how high max_backtrack is set, and the extra
    depth from 3 to 8 costs nothing on any step that accepts. It only
    costs anything on a step where no candidate ever satisfies the
    decrease test: one additional residual evaluation per unit of extra
    depth, on that step alone.

    The symptom that says to raise max_backtrack further: a step reporting
    converged = False with newton_iters == 1 and final_res_norm about
    equal to the residual the solve entered with. That is the stagnation
    exit (NewtonState.stagnated, see test_stagnated_newton_exits_early):
    every candidate down to alpha = 2 ** -(max_backtrack - 1) failed the
    Armijo-style decrease test, so the loop exited rather than repeat the
    identical rejected line search. Raising newton_tol or max_newton_iters
    does not help here, since a stagnated step is a fixed point of the
    whole Newton loop, not something more iterations would move past.
    Going past 8 makes sense for a good but very long Newton direction
    that only pays off after many halvings, typical of a stiff wide
    diffusion front; the cost is the one extra residual evaluation per
    extra depth described above, paid only on steps that reach the
    fallback.
    """
    max_newton_iters: int = 10
    max_krylov_iters: int = 50
    restart: int = 20
    newton_tol: float = 1e-8
    krylov_tol: float = 1e-6
    damping: float = 1.0
    backtrack_factor: float = 0.5
    max_backtrack: int = 8
    min_residual_decrease: float = 1e-4


class NKStats(NamedTuple):
    """Newton-Krylov solver statistics.

    lin_iters is the number of Krylov iterations that the GMRES solves
    used, summed over the Newton iterations (see _gmres_solve). It is at
    most newton_iters times the budget of NKParams.max_krylov_iters, plus
    less than one restart cycle for each Newton iteration.
    final_res_norm is the unweighted 2-norm of the residual at the
    returned solution and converged is final_res_norm < newton_tol.
    """
    converged: jnp.ndarray
    newton_iters: jnp.ndarray
    lin_iters: jnp.ndarray
    final_res_norm: jnp.ndarray


class NKResult(NamedTuple):
    """Newton-Krylov solver result."""
    solution: StateDict
    stats: NKStats


def _flatten_state(state: StateDict) -> tuple[jnp.ndarray, Callable]:
    """Flatten StateDict to 1D vector and get unravel function."""
    flat, unravel = ravel_pytree(state)
    return flat, unravel


def _jvp_matvec(
    residual_fn: Callable[[StateDict], StateDict],
    x: StateDict,
    v: StateDict
) -> StateDict:
    """
    Compute Jacobian-vector product J(x) @ v using forward-mode AD.

    J @ v = d/deps [F(x + eps*v)] |_{eps=0}

    Args:
        residual_fn: Function F(x) -> residual
        x: Point at which to evaluate Jacobian
        v: Vector to multiply

    Returns:
        J @ v as StateDict
    """
    _, jvp_result = jax.jvp(residual_fn, (x,), (v,))
    return jvp_result


def _gmres_cycle(matvec, r0, x0, m: int, tol_abs):
    """One GMRES cycle of at most m iterations that stops at the tolerance.

    Arnoldi with modified Gram-Schmidt, one reorthogonalization pass and
    Givens rotations, so the residual norm is known after each iteration and
    the loop stops when it is at most tol_abs. r0 is the residual b -
    matvec(x0) of the start vector. The caller computes it, so the cycle
    does not use an operator product for it. Returns (x, iterations).
    r0 and x0 are flat vectors. They can be real or complex.

    The inner products use the complex conjugate (V.conj() @ w), and the
    Givens rotations are unitary. For a real dtype, the arithmetic is the
    same as the real algorithm.
    """
    beta = jnp.linalg.norm(r0)
    n = r0.shape[0]
    dtype = r0.dtype
    V0 = jnp.zeros((m + 1, n), dtype=dtype).at[0].set(r0 / jnp.where(beta > 0, beta, 1.0))
    H0 = jnp.zeros((m + 1, m), dtype=dtype)
    # The product of the Givens rotations so far. One matrix product then
    # applies them to a new column. A loop of scalar updates for the same
    # work costs more than the matrix-vector product on a GPU.
    Q0 = jnp.eye(m + 1, dtype=dtype)
    g0 = jnp.zeros((m + 1,), dtype=dtype).at[0].set(beta)
    index = jnp.arange(m + 1)

    def cond(state):
        _, _, _, g, k = state
        return jnp.logical_and(k < m, jnp.abs(g[k]) > tol_abs)

    def body(state):
        V, H, Q, g, k = state
        w = matvec(V[k])
        active = (index <= k).astype(dtype)
        # Two Gram-Schmidt passes against the vectors built so far. The
        # projection coefficient on v_i is <v_i, w> = conj(v_i) . w.
        h = active * (V.conj() @ w)
        w = w - h @ V
        corr = active * (V.conj() @ w)
        w = w - corr @ V
        h = h + corr
        h_next = jnp.linalg.norm(w)
        h = h.at[k + 1].set(h_next)
        V = V.at[k + 1].set(w / jnp.where(h_next > 0, h_next, 1.0))

        # Apply the earlier Givens rotations to the new column. Row k + 1
        # of Q is still the unit row, so h[k + 1] stays h_next.
        h = Q @ h

        # A unitary rotation G with rows (conj(c), s) and (-s, c) on the
        # rows k, k + 1. a = h[k] is complex, d = h_next is real and not
        # negative. With r = sqrt(|a|^2 + d^2), c = a / r and s = d / r,
        # G maps (a, d) to (r, 0): conj(c) a + s d = (|a|^2 + d^2) / r = r
        # and -s a + c d = (-d a + a d) / r = 0. G is unitary because
        # |c|^2 + s^2 = 1 and conj(c) (-s) + s conj(c) = 0. For real
        # values, conj(c) = c and G is the real Givens rotation.
        a = h[k]
        d = h_next
        r = jnp.sqrt(jnp.real(a * jnp.conj(a)) + d * d)
        r = jnp.where(r > 0, r, 1.0)
        c, s_ = a / r, d / r
        c_bar = jnp.conj(c)
        h = h.at[k].set(c_bar * a + s_ * d).at[k + 1].set(0.0)
        g = g.at[k + 1].set(-s_ * g[k]).at[k].set(c_bar * g[k])
        row_k, row_next = Q[k], Q[k + 1]
        Q = Q.at[k].set(c_bar * row_k + s_ * row_next).at[k + 1].set(-s_ * row_k + c * row_next)
        return V, H.at[:, k].set(h), Q, g, k + 1

    V, H, _, g, k = lax.while_loop(
        cond, body, (V0, H0, Q0, g0, jnp.array(0, dtype=jnp.int32)))

    # Solve the k-by-k triangular system; the unused part is the identity.
    used = jnp.arange(m) < k
    R = jnp.where(used[:, None] & used[None, :], H[:m, :], jnp.eye(m, dtype=dtype))
    y = jax.scipy.linalg.solve_triangular(R, jnp.where(used, g[:m], 0.0), lower=False)
    return x0 + y @ V[:m], k


def _gmres_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    tol: float,
    max_iters: int,
    restart: int = 20,
    M: Callable[[jnp.ndarray], jnp.ndarray] | None = None
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Restarted GMRES that stops at the tolerance and counts its iterations.

    jax.scipy.sparse.linalg.gmres in its default ("batched") mode builds the
    complete Krylov space of the restart size in each cycle, and it returns
    no iteration count. A solve that needs two iterations then costs
    ``restart`` matrix-vector products (measured: one Crank-Nicolson step on
    a 128 x 128 Brusselator took 908 ms at restart 30 and 8.6 ms at restart
    2). This solver stops inside a cycle when the residual estimate reaches
    the tolerance. After each cycle, it calculates the true residual of the
    iterate and stops only when the true residual reaches the tolerance.
    The arithmetic is correct for real and complex systems.

    Args:
        matvec: Matrix-vector product function
        b: Right-hand side
        x0: Initial guess
        tol: Relative tolerance: the solve stops when the (preconditioned)
            residual 2-norm is at most tol times the 2-norm of the
            (preconditioned) right-hand side, as gmres does with atol = 0.
        max_iters: Total Krylov budget (NKParams.max_krylov_iters). The
            cycle size is restart_used = min(max_iters, restart) and the
            number of cycles is ceil(max_iters / restart_used), so the
            budget is exceeded by fewer than restart_used products.
        restart: Requested Krylov subspace size per restart cycle, capped
            by max_iters (NKParams.restart).
        M: Optional left preconditioner

    Returns:
        Tuple of (solution, iterations). iterations is the number of Krylov
        iterations that the solve used (one matrix-vector product each).
        The solve also uses one product for the initial residual and one
        product for the true residual after each cycle.
    """
    restart_used = max(1, min(max_iters, restart))
    maxiter = -(-max_iters // restart_used)  # ceil(max_iters / restart_used)

    if M is not None:
        # Left preconditioning: solve M^-1 A x = M^-1 b
        def op(v):
            return M(matvec(v))
        rhs = M(b)
    else:
        op = matvec
        rhs = b

    tol_abs = tol * jnp.linalg.norm(rhs)

    # The outer loop stops on the true residual norm(rhs - op(x)), not on
    # the estimate |g[k]| of the cycle. In float32 the estimate can be far
    # below the true residual. The residual vector of the returned iterate
    # is also the start residual of the next cycle, so this test does not
    # add an operator product. If the working precision cannot reach
    # tol_abs, the loop stops at the cycle budget.
    def cond(state):
        _, _, res, _, cycle = state
        return jnp.logical_and(cycle < maxiter, res > tol_abs)

    def body(state):
        x, r, _, iters, cycle = state
        x, k = _gmres_cycle(op, r, x, restart_used, tol_abs)
        r = rhs - op(x)
        return x, r, jnp.linalg.norm(r), iters + k, cycle + 1

    r0 = rhs - op(x0)
    init = (x0, r0, jnp.linalg.norm(r0), jnp.array(0, dtype=jnp.int32),
            jnp.array(0, dtype=jnp.int32))
    x, _, _, iters, _ = lax.while_loop(cond, body, init)
    return x, iters


def _bicgstab_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    tol: float,
    max_iters: int,
    M: Callable[[jnp.ndarray], jnp.ndarray] | None = None
) -> tuple[jnp.ndarray, int]:
    """
    BiCGSTAB solver using jax.scipy.sparse.linalg.

    Args:
        matvec: Matrix-vector product function
        b: Right-hand side
        x0: Initial guess
        tol: Convergence tolerance
        max_iters: Maximum iterations
        M: Optional preconditioner

    Returns:
        Tuple of (solution, iterations)
    """
    from jax.scipy.sparse.linalg import bicgstab

    if M is not None:
        # Left preconditioning
        def precond_matvec(v):
            return M(matvec(v))
        b_precond = M(b)
        result, info = bicgstab(precond_matvec, b_precond, x0=x0, tol=tol, maxiter=max_iters)
    else:
        result, info = bicgstab(matvec, b, x0=x0, tol=tol, maxiter=max_iters)

    return result, max_iters


def newton_krylov_solve(
    residual_fn: Callable[[StateDict], StateDict],
    x0: StateDict,
    grid: GridType,
    params: dict[str, Any],
    preconditioner: Preconditioner | None = None,
    nk_params: NKParams | None = None,
    dt: float = 1.0,
    precond_dt: float | None = None,
    precond_scale: float = 1.0
) -> NKResult:
    """
    Solve F(x) = 0 using Newton-Krylov method.

    Uses damped Newton iteration with Krylov inner solve:
        1. Compute residual r = F(x)
        2. Solve J(x) @ dx = -r using GMRES with JVP
        3. Line search: x_new = x + alpha * dx
        4. Repeat until converged or max iterations

    Args:
        residual_fn: Function F: StateDict -> StateDict to solve F(x) = 0
        x0: Initial guess (StateDict)
        grid: Grid handed to the preconditioner through PrecondContext.
            Norms are unweighted 2-norms of the flattened residual and do
            not use the grid.
        params: Model parameters
        preconditioner: Optional preconditioner
        nk_params: Solver parameters (default created if None)
        dt: Time step, used for both the residual's own bookkeeping (none,
            currently) and as the default preconditioner time step
        precond_dt: Time step the preconditioner's context sees; defaults to
            dt. A method whose Jacobian is alpha0 * I - dt * F'(y) (BE, CN
            with alpha0 = 1 and an effective step other than dt, BDF2 with
            alpha0 != 1) should pass the step that actually appears in that
            Jacobian's diffusive term, e.g. dt/2 for CN or dt/alpha0 for
            BDF2, so a linear FFT preconditioner built from
            (I - precond_dt*D*Laplacian)^-1 matches the operator it
            preconditions instead of only approximating it.
        precond_scale: Scalar applied to the preconditioner's output after
            precond_dt corrects its argument; BDF2's alpha0 != 1 multiplies
            the whole left-hand side, so M^-1 must be scaled by 1/alpha0 to
            approximate J^-1 = (1/alpha0)(I - (dt/alpha0)*D*Laplacian)^-1
            rather than (I - (dt/alpha0)*D*Laplacian)^-1 alone.

    Returns:
        NKResult containing solution and statistics (see NKStats for what
        lin_iters and final_res_norm measure)

    Example:
        >>> def residual(x):
        ...     return {'u': x['u']**2 - 1.0}
        >>> x0 = {'u': jnp.array([0.5])}
        >>> result = newton_krylov_solve(residual, x0, grid, params)
        >>> result.solution['u']  # Should be ~1.0
    """
    if nk_params is None:
        nk_params = NKParams()
    if preconditioner is None:
        preconditioner = IdentityPreconditioner()
    if precond_dt is None:
        precond_dt = dt

    # Create preconditioner context using the method's effective diffusive
    # step, which may differ from the outer dt (see precond_dt above).
    precond_context = PrecondContext(grid=grid, dt=precond_dt, params=params)

    # Get flattening structure from x0
    flat_x0, unravel = ravel_pytree(x0)

    # Define matvec for Krylov solver (operates on flat vectors)
    def krylov_matvec(x_flat: StateDict, v_flat: jnp.ndarray) -> jnp.ndarray:
        """J(x) @ v where x and v are in flat representation."""
        x = unravel(x_flat)
        v = unravel(v_flat)
        Jv = _jvp_matvec(residual_fn, x, v)
        Jv_flat, _ = ravel_pytree(Jv)
        return Jv_flat

    # Define preconditioner for Krylov (operates on flat vectors)
    def precond_apply_flat(v_flat: jnp.ndarray) -> jnp.ndarray:
        v = unravel(v_flat)
        Mv = preconditioner.apply(v, precond_context)
        Mv_flat, _ = ravel_pytree(Mv)
        return precond_scale * Mv_flat

    # Newton iteration state. Invariant at every loop boundary: res_norm is
    # ||F(x)|| at the current x and converged is res_norm < newton_tol, so
    # the statistics returned describe the iterate that is returned.
    # stagnated marks a line search that accepted no candidate and whose
    # best-tried residual is no better than the entering one (best_r_norm
    # >= r_norm): the returned iterate is then identical to state.x_flat, so
    # every subsequent Newton step would recompute the same residual, the
    # same GMRES solve and the same rejected candidates. Without this flag
    # newton_cond only checks iter_count and converged, so such a step
    # repeats until max_newton_iters (see test_stagnated_newton_exits_early).
    class NewtonState(NamedTuple):
        x_flat: jnp.ndarray
        res_norm: jnp.ndarray
        iter_count: jnp.ndarray
        lin_iters_total: jnp.ndarray
        converged: jnp.ndarray
        stagnated: jnp.ndarray

    def newton_step(state: NewtonState) -> NewtonState:
        """Single Newton iteration."""
        x = unravel(state.x_flat)

        # Compute residual
        r = residual_fn(x)
        r_flat, _ = ravel_pytree(r)
        r_norm = jnp.linalg.norm(r_flat)

        # Solve J @ dx = -r using Krylov
        def matvec_at_x(v_flat):
            return krylov_matvec(state.x_flat, v_flat)

        dx_flat, lin_iters = _gmres_solve(
            matvec=matvec_at_x,
            b=-r_flat,
            x0=jnp.zeros_like(r_flat),
            tol=nk_params.krylov_tol,
            max_iters=nk_params.max_krylov_iters,
            restart=nk_params.restart,
            M=precond_apply_flat if not isinstance(preconditioner, IdentityPreconditioner) else None
        )

        # Line search / damping
        alpha = nk_params.damping

        def backtrack_step(carry, _):
            alpha_bt, x_bt_flat, r_bt_norm, accepted, best_x_flat, best_r_norm = carry

            def try_candidate():
                # Candidate update
                x_new_flat = x_bt_flat + alpha_bt * dx_flat
                x_new = unravel(x_new_flat)
                r_new = residual_fn(x_new)
                r_new_flat, _ = ravel_pytree(r_new)
                r_new_norm = jnp.linalg.norm(r_new_flat)

                # Accept if residual decreased sufficiently
                accept = r_new_norm < (1.0 - nk_params.min_residual_decrease * alpha_bt) * r_bt_norm

                # Update if accepted, otherwise reduce alpha
                new_alpha = lax.cond(
                    jnp.logical_or(accept, accepted),
                    lambda: alpha_bt,
                    lambda: alpha_bt * nk_params.backtrack_factor
                )
                new_x_flat = lax.cond(
                    jnp.logical_and(accept, jnp.logical_not(accepted)),
                    lambda: x_new_flat,
                    lambda: x_bt_flat
                )
                new_r_norm = lax.cond(
                    jnp.logical_and(accept, jnp.logical_not(accepted)),
                    lambda: r_new_norm,
                    lambda: r_bt_norm
                )
                new_accepted = jnp.logical_or(accepted, accept)

                # Track the best candidate tried so far (by residual norm), so
                # that if no candidate ever satisfies the Armijo-style decrease
                # test, the loop still has something no worse than the starting
                # iterate to fall back on instead of the untried full step.
                is_better = r_new_norm < best_r_norm
                new_best_x_flat = lax.cond(is_better, lambda: x_new_flat, lambda: best_x_flat)
                new_best_r_norm = lax.cond(is_better, lambda: r_new_norm, lambda: best_r_norm)

                return (new_alpha, new_x_flat, new_r_norm, new_accepted, new_best_x_flat, new_best_r_norm)

            # Once accepted is True, every remaining scan iteration paid for
            # a residual_fn(x_new) evaluation whose result was then only
            # used to (maybe) update best_x_flat/best_r_norm: the other four
            # carry entries are pinned once accepted (new_alpha/new_x_flat/
            # new_r_norm all short-circuit to their pre-candidate values via
            # the `accept, accepted` and `accepted` conditions above, and
            # new_accepted stays True). But best_x_flat/best_r_norm are only
            # ever read below through lax.cond(accepted, ..., best_x_flat),
            # i.e. exactly when accepted is False -- and once this iteration
            # sees accepted already True, that branch is never taken again
            # for the rest of the scan. So skipping try_candidate() once
            # accepted leaves x_final_flat/r_final bit-identical while
            # saving one residual evaluation per remaining depth.
            #
            # Under vmap, lax.cond lowers to a select and both branches
            # still execute for every batch element, so this only saves
            # work per solve (accepted is a scalar), not per batch element;
            # correctness is unaffected either way.
            return lax.cond(accepted, lambda: carry, try_candidate), None

        # max_backtrack is a static Python int (a NKParams field, not a
        # traced value), so branching on it in Python is fine under
        # lax.while_loop. length=0 means "no line search": lax.scan would
        # run the body zero times, leaving best_x_flat/best_r_norm at their
        # initial values (the starting iterate, alpha = 0) and turning the
        # Newton step into a no-op that never applies the computed dx_flat.
        # Apply the configured damped step unconditionally instead, and
        # evaluate its residual so res_norm/converged still describe the
        # returned iterate (the invariant NewtonState documents).
        if nk_params.max_backtrack == 0:
            x_final_flat = state.x_flat + alpha * dx_flat
            r_final_pytree = residual_fn(unravel(x_final_flat))
            r_final_flat, _ = ravel_pytree(r_final_pytree)
            r_final = jnp.linalg.norm(r_final_flat)
            # No line search to stagnate: the damped step is always applied.
            stagnated = jnp.array(False)
        else:
            # Run backtracking. best_x_flat/best_r_norm start at the current
            # iterate (alpha = 0), so a candidate only replaces them when it
            # is strictly better than where we started.
            init_carry = (alpha, state.x_flat, r_norm, jnp.array(False), state.x_flat, r_norm)
            (_, x_bt_flat, r_bt_norm, accepted, best_x_flat, best_r_norm), _ = lax.scan(
                backtrack_step, init_carry, None, length=nk_params.max_backtrack
            )

            # If a candidate satisfied the decrease test, take it (the
            # accepted branch reuses the norm the line search already
            # computed). If none did, never apply the untried full step: it
            # was never checked against the decrease test and can be the
            # worst candidate of all (the whole reason backtracking exists).
            # Instead keep the best candidate actually tried, which is never
            # worse than the starting iterate because best_r_norm is
            # initialized from it.
            x_final_flat = lax.cond(accepted, lambda: x_bt_flat, lambda: best_x_flat)
            r_final = lax.cond(accepted, lambda: r_bt_norm, lambda: best_r_norm)
            # accepted is False and best_r_norm >= r_norm means no candidate
            # tried in the line search improved on the entering iterate, so
            # x_final_flat is exactly state.x_flat: the step is a no-op.
            stagnated = jnp.logical_and(jnp.logical_not(accepted), best_r_norm >= r_norm)

        return NewtonState(
            x_flat=x_final_flat,
            res_norm=r_final,
            iter_count=state.iter_count + 1,
            lin_iters_total=state.lin_iters_total + lin_iters,
            converged=r_final < nk_params.newton_tol,
            stagnated=stagnated
        )

    def newton_cond(state: NewtonState) -> jnp.ndarray:
        """Continue condition for Newton loop.

        stagnated exits the loop early (converged stays False) instead of
        repeating an unchanged iterate, the same GMRES solve and the same
        rejected line-search candidates until max_newton_iters.
        """
        return jnp.logical_and(
            state.iter_count < nk_params.max_newton_iters,
            jnp.logical_not(jnp.logical_or(state.converged, state.stagnated))
        )

    # Initial residual norm
    r0 = residual_fn(x0)
    r0_flat, _ = ravel_pytree(r0)
    r0_norm = jnp.linalg.norm(r0_flat)

    # Run Newton iteration
    init_state = NewtonState(
        x_flat=flat_x0,
        res_norm=r0_norm,
        iter_count=jnp.array(0, dtype=jnp.int32),
        lin_iters_total=jnp.array(0, dtype=jnp.int32),
        converged=r0_norm < nk_params.newton_tol,
        stagnated=jnp.array(False)
    )

    final_state = lax.while_loop(newton_cond, newton_step, init_state)

    # Reconstruct solution
    solution = unravel(final_state.x_flat)

    stats = NKStats(
        converged=final_state.converged,
        newton_iters=final_state.iter_count,
        lin_iters=final_state.lin_iters_total,
        final_res_norm=final_state.res_norm
    )

    return NKResult(solution=solution, stats=stats)


def create_implicit_residual(
    model: "MOLModel",
    y_old: StateDict,
    t_new: float,
    dt: float,
    method: str = "be"
) -> Callable[[StateDict], StateDict]:
    """
    Create residual function for implicit time stepping.

    BE: R(y_new) = y_new - y_old - dt * F(y_new, t_new)
    CN: R(y_new) = y_new - y_old - dt/2 * (F(y_old, t_old) + F(y_new, t_new))

    Args:
        model: MOLModel with rhs method
        y_old: Previous solution
        t_new: New time
        dt: Time step
        method: "be" (Backward Euler) or "cn" (Crank-Nicolson)

    Returns:
        Residual function R: StateDict -> StateDict
    """
    t_old = t_new - dt

    if method == "be":
        def residual(y_new: StateDict) -> StateDict:
            F_new = model.rhs(y_new, t_new)
            # R = y_new - y_old - dt * F(y_new)
            return tree_sub(tree_sub(y_new, y_old), tree_scale(F_new, dt))
    elif method == "cn":
        F_old = model.rhs(y_old, t_old)

        def residual(y_new: StateDict) -> StateDict:
            F_new = model.rhs(y_new, t_new)
            # R = y_new - y_old - dt/2 * (F_old + F_new)
            F_avg = tree_add(F_old, F_new)
            return tree_sub(tree_sub(y_new, y_old), tree_scale(F_avg, 0.5 * dt))
    else:
        raise ValueError(f"Unknown method: {method}")

    return residual


def bdf2_alpha0(dt: float, dt_prev: float) -> float:
    """The y_new coefficient (1+2w)/(1+w) of the scaled BDF2 residual, w = dt/dt_prev.

    Shared with create_bdf2_residual so the Newton Jacobian's y_new
    coefficient and the preconditioner's effective step (stepping.py's
    bdf2_step passes dt/alpha0, scaled by 1/alpha0) never drift apart: the
    Jacobian of the residual below is alpha0*I - dt*F'(y_new), so a linear
    preconditioner approximating (I - dt_eff*D*Laplacian)^-1 needs
    dt_eff = dt/alpha0 and a 1/alpha0 output scale to approximate J^-1.
    """
    omega = dt / dt_prev
    return (1.0 + 2.0 * omega) / (1.0 + omega)


def create_bdf2_residual(
    model: "MOLModel",
    y_n: StateDict,
    y_nm1: StateDict,
    t_new: float,
    dt: float,
    dt_prev: float
) -> Callable[[StateDict], StateDict]:
    """
    Create residual function for BDF2 time stepping.

    Variable step BDF2 (Hairer and Wanner, Solving ODEs II, III.5), with
    w = dt / dt_prev:

        y_{n+1} - (1+w)^2/(1+2w) y_n + w^2/(1+2w) y_{n-1} = (1+w)/(1+2w) dt F(y_{n+1})

    Multiplying through by (1+2w)/(1+w) gives the form used here,

        (1+2w)/(1+w) y_{n+1} - (1+w) y_n + w^2/(1+w) y_{n-1} = dt F(y_{n+1}),

    whose right-hand side coefficient is exactly dt. Scaling only the
    left-hand side and keeping (1+w)/(1+2w) dt on the right, as an earlier
    version did, integrates y' = (2/3) F(y) at constant step: on y' = -y it
    converged to exp(-2t/3) instead of exp(-t), a first-order error that
    does not shrink with dt.

    Args:
        model: MOLModel with rhs method
        y_n: Current solution
        y_nm1: Previous solution
        t_new: New time
        dt: Current time step
        dt_prev: Previous time step

    Returns:
        Residual function R: StateDict -> StateDict
    """
    omega = dt / dt_prev
    alpha0 = bdf2_alpha0(dt, dt_prev)
    alpha1 = -(1.0 + omega)
    alpha2 = omega ** 2 / (1.0 + omega)
    beta = dt

    def residual(y_new: StateDict) -> StateDict:
        F_new = model.rhs(y_new, t_new)
        # R = alpha0 * y_new + alpha1 * y_n + alpha2 * y_nm1 - beta * F(y_new)
        result = tree_scale(y_new, alpha0)
        result = tree_add(result, tree_scale(y_n, alpha1))
        result = tree_add(result, tree_scale(y_nm1, alpha2))
        result = tree_sub(result, tree_scale(F_new, beta))
        return result

    return residual
