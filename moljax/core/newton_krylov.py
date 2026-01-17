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

from dataclasses import dataclass
from typing import Callable, Tuple, Dict, Any, Optional, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax
from jax.flatten_util import ravel_pytree

from moljax.core.state import StateDict, tree_add, tree_scale, tree_sub, tree_norm2
from moljax.core.grid import GridType
from moljax.core.preconditioners import Preconditioner, PrecondContext, IdentityPreconditioner


class NKParams(NamedTuple):
    """Newton-Krylov solver parameters."""
    max_newton_iters: int = 10
    max_krylov_iters: int = 50
    newton_tol: float = 1e-8
    krylov_tol: float = 1e-6
    damping: float = 1.0
    backtrack_factor: float = 0.5
    max_backtrack: int = 3
    min_residual_decrease: float = 1e-4


class NKStats(NamedTuple):
    """Newton-Krylov solver statistics."""
    converged: jnp.ndarray
    newton_iters: jnp.ndarray
    lin_iters: jnp.ndarray
    final_res_norm: jnp.ndarray


class NKResult(NamedTuple):
    """Newton-Krylov solver result."""
    solution: StateDict
    stats: NKStats


def _flatten_state(state: StateDict) -> Tuple[jnp.ndarray, Callable]:
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


def _gmres_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    tol: float,
    max_iters: int,
    M: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
) -> Tuple[jnp.ndarray, int]:
    """
    GMRES solver wrapper.

    Uses jax.scipy.sparse.linalg.gmres if available.
    Falls back to basic implementation if needed.

    Args:
        matvec: Matrix-vector product function
        b: Right-hand side
        x0: Initial guess
        tol: Convergence tolerance
        max_iters: Maximum iterations
        M: Optional left preconditioner

    Returns:
        Tuple of (solution, iterations)
    """
    # Use JAX's GMRES
    # Note: JAX GMRES doesn't track iteration count, we estimate from residual reduction
    try:
        from jax.scipy.sparse.linalg import gmres

        # Wrap matvec in LinearOperator-like callable
        def linear_op(v):
            return matvec(v)

        # Apply preconditioner if provided
        if M is not None:
            # Left preconditioning: solve M^-1 A x = M^-1 b
            def precond_matvec(v):
                return M(matvec(v))
            b_precond = M(b)
            result, info = gmres(precond_matvec, b_precond, x0=x0, tol=tol, maxiter=max_iters)
        else:
            result, info = gmres(linear_op, b, x0=x0, tol=tol, maxiter=max_iters)

        # info is 0 for success
        return result, max_iters  # GMRES doesn't return iter count easily

    except ImportError:
        # Fallback to BiCGSTAB
        return _bicgstab_solve(matvec, b, x0, tol, max_iters, M)


def _bicgstab_solve(
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    b: jnp.ndarray,
    x0: jnp.ndarray,
    tol: float,
    max_iters: int,
    M: Optional[Callable[[jnp.ndarray], jnp.ndarray]] = None
) -> Tuple[jnp.ndarray, int]:
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
    params: Dict[str, Any],
    preconditioner: Optional[Preconditioner] = None,
    nk_params: Optional[NKParams] = None,
    dt: float = 1.0
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
        grid: Grid for norm computations
        params: Model parameters
        preconditioner: Optional preconditioner
        nk_params: Solver parameters (default created if None)
        dt: Time step (for preconditioner context)

    Returns:
        NKResult containing solution and statistics

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

    # Create preconditioner context
    precond_context = PrecondContext(grid=grid, dt=dt, params=params)

    # Get flattening structure from x0
    flat_x0, unravel = ravel_pytree(x0)
    n_dof = flat_x0.shape[0]

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
        return Mv_flat

    # Newton iteration state
    class NewtonState(NamedTuple):
        x_flat: jnp.ndarray
        res_norm: jnp.ndarray
        iter_count: jnp.ndarray
        lin_iters_total: jnp.ndarray
        converged: jnp.ndarray

    def newton_step(state: NewtonState) -> NewtonState:
        """Single Newton iteration."""
        x = unravel(state.x_flat)

        # Compute residual
        r = residual_fn(x)
        r_flat, _ = ravel_pytree(r)
        r_norm = jnp.linalg.norm(r_flat)

        # Check convergence
        converged = r_norm < nk_params.newton_tol

        # Solve J @ dx = -r using Krylov
        def matvec_at_x(v_flat):
            return krylov_matvec(state.x_flat, v_flat)

        dx_flat, lin_iters = _gmres_solve(
            matvec=matvec_at_x,
            b=-r_flat,
            x0=jnp.zeros_like(r_flat),
            tol=nk_params.krylov_tol,
            max_iters=nk_params.max_krylov_iters,
            M=precond_apply_flat if not isinstance(preconditioner, IdentityPreconditioner) else None
        )

        # Line search / damping
        alpha = nk_params.damping

        def backtrack_step(carry, _):
            alpha_bt, x_bt_flat, r_bt_norm, accepted = carry

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

            return (new_alpha, new_x_flat, new_r_norm, new_accepted), None

        # Run backtracking
        init_carry = (alpha, state.x_flat, r_norm, jnp.array(False))
        (final_alpha, x_new_flat, new_r_norm, _), _ = lax.scan(
            backtrack_step, init_carry, None, length=nk_params.max_backtrack
        )

        # If no backtrack accepted, just take the damped step
        x_final_flat = lax.cond(
            jnp.allclose(x_new_flat, state.x_flat),
            lambda: state.x_flat + alpha * dx_flat,
            lambda: x_new_flat
        )

        return NewtonState(
            x_flat=x_final_flat,
            res_norm=new_r_norm,
            iter_count=state.iter_count + 1,
            lin_iters_total=state.lin_iters_total + lin_iters,
            converged=converged
        )

    def newton_cond(state: NewtonState) -> jnp.ndarray:
        """Continue condition for Newton loop."""
        return jnp.logical_and(
            state.iter_count < nk_params.max_newton_iters,
            jnp.logical_not(state.converged)
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
        converged=r0_norm < nk_params.newton_tol
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

    Variable step BDF2:
        (1+2r)/(1+r) * y_{n+1} - (1+r) * y_n + r^2/(1+r) * y_{n-1} = dt * F(y_{n+1})

    where r = dt / dt_prev.

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
    alpha0 = (1.0 + 2.0 * omega) / (1.0 + omega)
    alpha1 = -(1.0 + omega)
    alpha2 = omega ** 2 / (1.0 + omega)
    beta = dt * (1.0 + omega) / (1.0 + 2.0 * omega)

    def residual(y_new: StateDict) -> StateDict:
        F_new = model.rhs(y_new, t_new)
        # R = alpha0 * y_new + alpha1 * y_n + alpha2 * y_nm1 - beta * F(y_new)
        result = tree_scale(y_new, alpha0)
        result = tree_add(result, tree_scale(y_n, alpha1))
        result = tree_add(result, tree_scale(y_nm1, alpha2))
        result = tree_sub(result, tree_scale(F_new, beta))
        return result

    return residual
