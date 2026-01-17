"""
Time stepping schemes for MOL-JAX.

This module provides time integration methods:
- Explicit: Euler, SSPRK3, RK4
- Implicit: BE, CN, BDF2 (using Newton-Krylov)
- IMEX: Strang splitting and IMEX-Euler (diffusion implicit via FFT, reaction explicit)
- Adaptive integration via JIT-friendly lax.while_loop

Design decisions:
- All integrators are pure functions suitable for JIT
- Method selection via IntEnum + lax.switch (no recompilation)
- Adaptive stepping uses static-shape output buffers
- Error estimation via step doubling or embedded pairs
- Accept/reject logic via lax.cond (no Python branching)
- IMEX integrators use FFT for implicit diffusion, explicit RK for reactions
"""

from enum import IntEnum
from typing import Dict, Any, Optional, Tuple, Callable, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.grid import GridType
from moljax.core.state import (
    StateDict, tree_add, tree_sub, tree_scale, tree_axpy,
    tree_zeros_like, scaled_error_norm
)
from moljax.core.bc import apply_bc
from moljax.core.model import MOLModel
from moljax.core.newton_krylov import (
    newton_krylov_solve, create_implicit_residual, create_bdf2_residual,
    NKParams, NKStats, NKResult
)
from moljax.core.preconditioners import Preconditioner, IdentityPreconditioner
from moljax.core.dt_policy import (
    CFLParams, PIDParams, ControllerState,
    create_initial_controller_state, propose_dt, compute_error_order,
    heisenberg_cfl_dt
)
from moljax.core.utils import (
    StatusCode, allocate_scalar_history, allocate_state_history,
    save_to_history, is_finite, get_interior
)


class IntegratorType(IntEnum):
    """Time integrator types."""
    EULER = 0
    SSPRK3 = 1
    RK4 = 2
    BE = 3          # Backward Euler
    CN = 4          # Crank-Nicolson
    BDF2 = 5
    IMEX_EULER = 6  # IMEX with Euler for reaction, implicit diffusion
    IMEX_STRANG = 7 # IMEX Strang splitting (2nd order)


# =============================================================================
# Explicit Integrators
# =============================================================================

def euler_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float
) -> StateDict:
    """
    Forward Euler step: y_{n+1} = y_n + dt * F(y_n, t_n).

    First-order accurate. Simple but conditionally stable.

    Args:
        model: MOLModel with rhs method
        y: Current state (with BCs applied)
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    F = model.rhs(y, t)
    return tree_axpy(y, dt, F)


def ssprk3_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float
) -> StateDict:
    """
    Strong Stability Preserving RK3 (Shu-Osher form).

    Third-order accurate. Good for hyperbolic problems.

    Stage 1: y1 = y + dt * F(y, t)
    Stage 2: y2 = 3/4*y + 1/4*y1 + 1/4*dt*F(y1, t+dt)
    Stage 3: y_new = 1/3*y + 2/3*y2 + 2/3*dt*F(y2, t+dt/2)

    Args:
        model: MOLModel with rhs method
        y: Current state
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    # Stage 1
    F0 = model.rhs(y, t)
    y1 = tree_axpy(y, dt, F0)

    # Stage 2
    F1 = model.rhs(y1, t + dt)
    y2 = tree_add(tree_scale(y, 0.75), tree_scale(y1, 0.25))
    y2 = tree_axpy(y2, 0.25 * dt, F1)

    # Stage 3
    F2 = model.rhs(y2, t + 0.5 * dt)
    y_new = tree_add(tree_scale(y, 1.0 / 3.0), tree_scale(y2, 2.0 / 3.0))
    y_new = tree_axpy(y_new, 2.0 / 3.0 * dt, F2)

    return y_new


def rk4_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float
) -> StateDict:
    """
    Classical 4th-order Runge-Kutta.

    k1 = F(y, t)
    k2 = F(y + dt/2*k1, t + dt/2)
    k3 = F(y + dt/2*k2, t + dt/2)
    k4 = F(y + dt*k3, t + dt)
    y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        model: MOLModel with rhs method
        y: Current state
        t: Current time
        dt: Time step

    Returns:
        New state at t + dt
    """
    k1 = model.rhs(y, t)
    k2 = model.rhs(tree_axpy(y, 0.5 * dt, k1), t + 0.5 * dt)
    k3 = model.rhs(tree_axpy(y, 0.5 * dt, k2), t + 0.5 * dt)
    k4 = model.rhs(tree_axpy(y, dt, k3), t + dt)

    # y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    update = tree_add(k1, tree_scale(k2, 2.0))
    update = tree_add(update, tree_scale(k3, 2.0))
    update = tree_add(update, k4)

    return tree_axpy(y, dt / 6.0, update)


# =============================================================================
# Implicit Integrators
# =============================================================================

def be_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    preconditioner: Optional[Preconditioner] = None,
    nk_params: Optional[NKParams] = None
) -> Tuple[StateDict, NKStats]:
    """
    Backward Euler step: y_{n+1} = y_n + dt * F(y_{n+1}, t_{n+1}).

    First-order accurate but A-stable (unconditionally stable for linear).
    Solves nonlinear system using Newton-Krylov.

    Args:
        model: MOLModel with rhs method
        y: Current state
        t: Current time
        dt: Time step
        preconditioner: Optional preconditioner for NK solver
        nk_params: Newton-Krylov parameters

    Returns:
        Tuple of (new_state, nk_stats)
    """
    t_new = t + dt

    # Create residual: R(y_new) = y_new - y - dt * F(y_new, t_new)
    residual_fn = create_implicit_residual(model, y, t_new, dt, method="be")

    # Initial guess: explicit Euler
    y_init = euler_step(model, y, t, dt)

    # Solve
    result = newton_krylov_solve(
        residual_fn=residual_fn,
        x0=y_init,
        grid=model.grid,
        params=model.params,
        preconditioner=preconditioner,
        nk_params=nk_params,
        dt=dt
    )

    return result.solution, result.stats


def cn_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    preconditioner: Optional[Preconditioner] = None,
    nk_params: Optional[NKParams] = None
) -> Tuple[StateDict, NKStats]:
    """
    Crank-Nicolson step: y_{n+1} = y_n + dt/2 * (F(y_n, t_n) + F(y_{n+1}, t_{n+1})).

    Second-order accurate and A-stable.

    Args:
        model: MOLModel with rhs method
        y: Current state
        t: Current time
        dt: Time step
        preconditioner: Optional preconditioner
        nk_params: Newton-Krylov parameters

    Returns:
        Tuple of (new_state, nk_stats)
    """
    t_new = t + dt

    # Create residual for CN
    residual_fn = create_implicit_residual(model, y, t_new, dt, method="cn")

    # Initial guess: explicit Euler
    y_init = euler_step(model, y, t, dt)

    # Solve
    result = newton_krylov_solve(
        residual_fn=residual_fn,
        x0=y_init,
        grid=model.grid,
        params=model.params,
        preconditioner=preconditioner,
        nk_params=nk_params,
        dt=dt
    )

    return result.solution, result.stats


def bdf2_step(
    model: MOLModel,
    y: StateDict,
    y_prev: StateDict,
    t: float,
    dt: float,
    dt_prev: float,
    preconditioner: Optional[Preconditioner] = None,
    nk_params: Optional[NKParams] = None
) -> Tuple[StateDict, NKStats]:
    """
    BDF2 step (variable step size).

    Second-order accurate and A-stable.

    Args:
        model: MOLModel with rhs method
        y: Current state (y_n)
        y_prev: Previous state (y_{n-1})
        t: Current time
        dt: Current time step
        dt_prev: Previous time step
        preconditioner: Optional preconditioner
        nk_params: Newton-Krylov parameters

    Returns:
        Tuple of (new_state, nk_stats)
    """
    t_new = t + dt

    # Create residual for BDF2
    residual_fn = create_bdf2_residual(model, y, y_prev, t_new, dt, dt_prev)

    # Initial guess: linear extrapolation
    y_init = tree_axpy(tree_scale(y, 2.0), -1.0, y_prev)

    # Solve
    result = newton_krylov_solve(
        residual_fn=residual_fn,
        x0=y_init,
        grid=model.grid,
        params=model.params,
        preconditioner=preconditioner,
        nk_params=nk_params,
        dt=dt
    )

    return result.solution, result.stats


# =============================================================================
# IMEX Integrators
# =============================================================================

def imex_euler_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    fft_cache,
    diffusivities: Dict[str, float]
) -> StateDict:
    """
    IMEX Euler step: diffusion implicit (FFT), reaction explicit.

    The split is:
    1. Compute explicit reaction: R(y, t)
    2. Form RHS: y + dt * R(y, t)
    3. Solve (I - dt * D * Δ) y_new = RHS for each diffusive field

    First-order accurate overall.

    Args:
        model: MOLModel with diffusion and reaction terms
        y: Current state (with BCs applied)
        t: Current time
        dt: Time step
        fft_cache: Precomputed FFT cache for diffusion solve
        diffusivities: Dict mapping field name to diffusion coefficient

    Returns:
        New state at t + dt
    """
    from moljax.core.fft_solvers import apply_diffusion_inverse_fft

    # Apply boundary conditions
    y = model.apply_bcs(y, t)

    # Compute explicit RHS (reactions only, diffusion handled implicitly)
    # We need the nonlinear part only
    if len(model.nonlinear_ops) > 0:
        N = model.nonlinear_rhs(y, t)
    else:
        N = tree_zeros_like(y)

    # Form RHS for diffusion solve: y + dt * N
    rhs = tree_axpy(y, dt, N)

    # Solve (I - dt * D * Δ) y_new = rhs using FFT
    y_new = apply_diffusion_inverse_fft(rhs, model.grid, dt, diffusivities, fft_cache)

    # Apply boundary conditions to result
    y_new = model.apply_bcs(y_new, t + dt)

    return y_new


def imex_strang_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    fft_cache,
    diffusivities: Dict[str, float]
) -> StateDict:
    """
    IMEX Strang splitting step (2nd order).

    The split is:
    1. Half diffusion step (implicit FFT): y* = (I - dt/2 * D * Δ)^-1 y
    2. Full reaction step (explicit RK2): y** = y* + dt * R(y*, t)
       Using Heun's method for 2nd order
    3. Half diffusion step (implicit FFT): y_new = (I - dt/2 * D * Δ)^-1 y**

    Second-order accurate for time-independent operators.

    Args:
        model: MOLModel with diffusion and reaction terms
        y: Current state
        t: Current time
        dt: Time step
        fft_cache: Precomputed FFT cache
        diffusivities: Dict mapping field name to diffusion coefficient

    Returns:
        New state at t + dt
    """
    from moljax.core.fft_solvers import apply_diffusion_inverse_fft

    dt_half = dt / 2.0

    # Apply boundary conditions
    y = model.apply_bcs(y, t)

    # Step 1: Half diffusion step (implicit)
    y_star = apply_diffusion_inverse_fft(y, model.grid, dt_half, diffusivities, fft_cache)
    y_star = model.apply_bcs(y_star, t + dt_half)

    # Step 2: Full reaction step using Heun's method (explicit 2nd order)
    if len(model.nonlinear_ops) > 0:
        # k1 = R(y*, t + dt/2)
        R1 = model.nonlinear_rhs(y_star, t + dt_half)

        # y_tilde = y* + dt * k1
        y_tilde = tree_axpy(y_star, dt, R1)
        y_tilde = model.apply_bcs(y_tilde, t + dt)

        # k2 = R(y_tilde, t + dt)
        R2 = model.nonlinear_rhs(y_tilde, t + dt)

        # y** = y* + dt/2 * (k1 + k2)
        R_avg = tree_add(R1, R2)
        y_double = tree_axpy(y_star, 0.5 * dt, R_avg)
    else:
        y_double = y_star

    y_double = model.apply_bcs(y_double, t + dt)

    # Step 3: Half diffusion step (implicit)
    y_new = apply_diffusion_inverse_fft(y_double, model.grid, dt_half, diffusivities, fft_cache)
    y_new = model.apply_bcs(y_new, t + dt)

    return y_new


def imex_ssprk2_step(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    fft_cache,
    diffusivities: Dict[str, float]
) -> StateDict:
    """
    IMEX SSP-RK2 step (2nd order).

    A two-stage IMEX method:
    Stage 1: y1 = (I - dt*D*Δ)^-1 (y + dt * N(y, t))
    Stage 2: y_new = 1/2 * y + 1/2 * (I - dt*D*Δ)^-1 (y1 + dt * N(y1, t+dt))

    This is the IMEX analogue of SSPRK2/Heun's method.

    Args:
        model: MOLModel with diffusion and reaction terms
        y: Current state
        t: Current time
        dt: Time step
        fft_cache: Precomputed FFT cache
        diffusivities: Dict mapping field name to diffusion coefficient

    Returns:
        New state at t + dt
    """
    from moljax.core.fft_solvers import apply_diffusion_inverse_fft

    # Apply boundary conditions
    y = model.apply_bcs(y, t)

    # Stage 1: Forward Euler style
    if len(model.nonlinear_ops) > 0:
        N0 = model.nonlinear_rhs(y, t)
    else:
        N0 = tree_zeros_like(y)

    rhs1 = tree_axpy(y, dt, N0)
    y1 = apply_diffusion_inverse_fft(rhs1, model.grid, dt, diffusivities, fft_cache)
    y1 = model.apply_bcs(y1, t + dt)

    # Stage 2: Averaging
    if len(model.nonlinear_ops) > 0:
        N1 = model.nonlinear_rhs(y1, t + dt)
    else:
        N1 = tree_zeros_like(y1)

    rhs2 = tree_axpy(y1, dt, N1)
    y2_full = apply_diffusion_inverse_fft(rhs2, model.grid, dt, diffusivities, fft_cache)
    y2_full = model.apply_bcs(y2_full, t + dt)

    # Average: y_new = 1/2 * y + 1/2 * y2_full
    y_new = tree_add(tree_scale(y, 0.5), tree_scale(y2_full, 0.5))
    y_new = model.apply_bcs(y_new, t + dt)

    return y_new


# =============================================================================
# Error Estimation
# =============================================================================

def estimate_error_doubling(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    step_fn: Callable
) -> Tuple[StateDict, StateDict]:
    """
    Estimate error using step doubling.

    Takes one full step of size dt and two half steps of size dt/2.
    Error estimate: (y_half - y_full) / (2^p - 1) where p is the order.

    Args:
        model: MOLModel
        y: Current state
        t: Current time
        dt: Time step
        step_fn: Step function (model, y, t, dt) -> y_new

    Returns:
        Tuple of (y_new, error_estimate)
    """
    # Full step
    y_full = step_fn(model, y, t, dt)

    # Two half steps
    y_half1 = step_fn(model, y, t, dt / 2.0)
    y_half2 = step_fn(model, y_half1, t + dt / 2.0, dt / 2.0)

    # Error estimate (unscaled by order factor)
    error = tree_sub(y_half2, y_full)

    # Return the more accurate solution (half steps) and error
    return y_half2, error


def estimate_error_implicit(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    y_be: StateDict,
    y_cn: StateDict
) -> StateDict:
    """
    Estimate error for implicit methods by comparing BE and CN.

    Since BE is O(dt) and CN is O(dt^2), the difference is an error estimate.

    Args:
        model: MOLModel
        y: Current state
        t: Current time
        dt: Time step
        y_be: BE solution
        y_cn: CN solution

    Returns:
        Error estimate
    """
    return tree_sub(y_cn, y_be)


def estimate_error_imex_doubling(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    fft_cache,
    diffusivities: Dict[str, float],
    use_strang: bool = True
) -> Tuple[StateDict, StateDict]:
    """
    Estimate error for IMEX methods using step doubling.

    Takes one full step of size dt and two half steps of size dt/2.

    Args:
        model: MOLModel
        y: Current state
        t: Current time
        dt: Time step
        fft_cache: FFT cache
        diffusivities: Diffusion coefficients
        use_strang: Use Strang splitting (True) or IMEX Euler (False)

    Returns:
        Tuple of (y_new, error_estimate)
    """
    if use_strang:
        step_fn = lambda m, y, t, dt: imex_strang_step(m, y, t, dt, fft_cache, diffusivities)
    else:
        step_fn = lambda m, y, t, dt: imex_euler_step(m, y, t, dt, fft_cache, diffusivities)

    # Full step
    y_full = step_fn(model, y, t, dt)

    # Two half steps
    y_half1 = step_fn(model, y, t, dt / 2.0)
    y_half2 = step_fn(model, y_half1, t + dt / 2.0, dt / 2.0)

    # Error estimate
    error = tree_sub(y_half2, y_full)

    # Return the more accurate solution (half steps)
    return y_half2, error


# =============================================================================
# Unified Step Function
# =============================================================================

def step_explicit(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    method: int
) -> StateDict:
    """
    Take a single explicit step using specified method.

    Uses lax.switch to select method without recompilation.

    Args:
        model: MOLModel
        y: Current state
        t: Current time
        dt: Time step
        method: IntegratorType enum value (0, 1, or 2)

    Returns:
        New state
    """
    step_fns = [
        lambda: euler_step(model, y, t, dt),
        lambda: ssprk3_step(model, y, t, dt),
        lambda: rk4_step(model, y, t, dt),
    ]

    return lax.switch(method, step_fns)


def step_explicit_with_error(
    model: MOLModel,
    y: StateDict,
    t: float,
    dt: float,
    method: int
) -> Tuple[StateDict, StateDict]:
    """
    Take explicit step with error estimation via step doubling.

    Args:
        model: MOLModel
        y: Current state
        t: Current time
        dt: Time step
        method: IntegratorType enum value

    Returns:
        Tuple of (new_state, error_estimate)
    """
    def euler_with_err():
        return estimate_error_doubling(model, y, t, dt, euler_step)

    def ssprk3_with_err():
        return estimate_error_doubling(model, y, t, dt, ssprk3_step)

    def rk4_with_err():
        return estimate_error_doubling(model, y, t, dt, rk4_step)

    step_fns = [euler_with_err, ssprk3_with_err, rk4_with_err]
    return lax.switch(method, step_fns)


# =============================================================================
# Adaptive Integration with lax.while_loop
# =============================================================================

class AdaptiveState(NamedTuple):
    """State for adaptive integration while_loop."""
    # Current solution
    t: jnp.ndarray
    y: StateDict
    dt: jnp.ndarray

    # History for BDF2
    y_prev: StateDict
    dt_prev: jnp.ndarray
    step_count: jnp.ndarray  # For BDF2 startup

    # Controller state
    controller: ControllerState

    # Output buffers (pre-allocated)
    t_history: jnp.ndarray
    y_history: Dict[str, jnp.ndarray]
    dt_history: jnp.ndarray
    write_idx: jnp.ndarray

    # Statistics
    n_accepted: jnp.ndarray
    n_rejected: jnp.ndarray
    status: jnp.ndarray


class AdaptiveResult(NamedTuple):
    """Result of adaptive integration."""
    t_final: jnp.ndarray
    y_final: StateDict
    t_history: jnp.ndarray
    y_history: Dict[str, jnp.ndarray]
    dt_history: jnp.ndarray
    n_steps: jnp.ndarray
    n_accepted: jnp.ndarray
    n_rejected: jnp.ndarray
    status: jnp.ndarray


def adaptive_integrate(
    model: MOLModel,
    y0: StateDict,
    t0: float,
    t_end: float,
    dt0: float,
    method: int = IntegratorType.RK4,
    max_steps: int = 10000,
    cfl_params: Optional[CFLParams] = None,
    pid_params: Optional[PIDParams] = None,
    preconditioner: Optional[Preconditioner] = None,
    nk_params: Optional[NKParams] = None,
    save_every: int = 1
) -> AdaptiveResult:
    """
    Adaptive time integration using lax.while_loop.

    This is the main driver for solving PDEs. It uses:
    - Error estimation via step doubling (explicit) or method comparison (implicit)
    - PID controller for dt adjustment
    - CFL limiter for explicit methods
    - Accept/reject logic via lax.cond

    All array shapes are static; output buffers are pre-allocated.

    Args:
        model: MOLModel to integrate
        y0: Initial state (with ghost cells)
        t0: Initial time
        t_end: Final time
        dt0: Initial time step
        method: IntegratorType enum value
        max_steps: Maximum number of steps (for buffer allocation)
        cfl_params: CFL parameters (default created if None)
        pid_params: PID parameters (default created if None)
        preconditioner: Preconditioner for implicit methods
        nk_params: NK parameters for implicit methods
        save_every: Save every N accepted steps to history

    Returns:
        AdaptiveResult with final state and histories
    """
    dtype = model.dtype

    if cfl_params is None:
        cfl_params = CFLParams()
    if pid_params is None:
        pid_params = PIDParams()
    if preconditioner is None:
        preconditioner = IdentityPreconditioner()
    if nk_params is None:
        nk_params = NKParams()

    # Get method order for error scaling
    order = compute_error_order(method)

    # Compute initial CFL-limited dt for explicit methods
    is_explicit = method < 3
    dt_cfl = heisenberg_cfl_dt(model.grid, model.params, cfl_params)
    dt_init = lax.cond(
        is_explicit,
        lambda: jnp.minimum(dt0, dt_cfl),
        lambda: jnp.array(dt0, dtype=dtype)
    )

    # Allocate output buffers
    max_saves = max_steps // save_every + 1
    t_history = allocate_scalar_history(max_saves, dtype)
    y_history = allocate_state_history(y0, model.grid, max_saves, interior_only=True, dtype=dtype)
    dt_history = allocate_scalar_history(max_saves, dtype)

    # Initialize state
    controller = create_initial_controller_state(dtype)

    init_state = AdaptiveState(
        t=jnp.array(t0, dtype=dtype),
        y=y0,
        dt=dt_init,
        y_prev=y0,  # Will be updated after first step
        dt_prev=dt_init,
        step_count=jnp.array(0, dtype=jnp.int32),
        controller=controller,
        t_history=t_history.at[0].set(t0),
        y_history=save_to_history(y_history, y0, model.grid, 0, interior_only=True),
        dt_history=dt_history,
        write_idx=jnp.array(1, dtype=jnp.int32),
        n_accepted=jnp.array(0, dtype=jnp.int32),
        n_rejected=jnp.array(0, dtype=jnp.int32),
        status=jnp.array(StatusCode.RUNNING, dtype=jnp.int32)
    )

    def cond_fn(state: AdaptiveState) -> jnp.ndarray:
        """Continue while t < t_end and not at max steps or error."""
        running = state.status == StatusCode.RUNNING
        not_done = state.t < t_end
        space_left = state.write_idx < max_saves
        return jnp.logical_and(running, jnp.logical_and(not_done, space_left))

    def body_fn(state: AdaptiveState) -> AdaptiveState:
        """Single step of adaptive integration."""

        # Clamp dt to not overshoot
        dt_clamped = jnp.minimum(state.dt, t_end - state.t)

        # Take step based on method type
        def explicit_step():
            y_new, err = step_explicit_with_error(model, state.y, state.t, dt_clamped, method)
            # Create dummy NK stats
            nk_stats = NKStats(
                converged=jnp.array(True),
                newton_iters=jnp.array(0, dtype=jnp.int32),
                lin_iters=jnp.array(0, dtype=jnp.int32),
                final_res_norm=jnp.array(0.0, dtype=dtype)
            )
            return y_new, err, nk_stats

        def implicit_step():
            # For implicit: use BE + CN comparison for error
            # Or for BDF2 after startup, use BE as error estimator
            def be_only():
                y_be, stats = be_step(model, state.y, state.t, dt_clamped, preconditioner, nk_params)
                # Use F(y) - F(y_be) as crude error estimate
                err = tree_scale(model.rhs(y_be, state.t + dt_clamped), dt_clamped)
                return y_be, err, stats

            def cn_with_err():
                y_cn, stats = cn_step(model, state.y, state.t, dt_clamped, preconditioner, nk_params)
                y_be, _ = be_step(model, state.y, state.t, dt_clamped, preconditioner, nk_params)
                err = tree_sub(y_cn, y_be)
                return y_cn, err, stats

            def bdf2_with_err():
                y_bdf2, stats = bdf2_step(
                    model, state.y, state.y_prev, state.t, dt_clamped, state.dt_prev,
                    preconditioner, nk_params
                )
                # Compare with BE for error
                y_be, _ = be_step(model, state.y, state.t, dt_clamped, preconditioner, nk_params)
                err = tree_sub(y_bdf2, y_be)
                return y_bdf2, err, stats

            # Select based on method and step count
            is_be = method == 3
            is_cn = method == 4
            is_bdf2 = method == 5
            bdf2_startup = jnp.logical_and(is_bdf2, state.step_count < 1)

            # Use BE for: BE method OR BDF2 startup
            use_be = jnp.logical_or(is_be, bdf2_startup)

            return lax.cond(
                use_be,
                be_only,
                lambda: lax.cond(is_cn, cn_with_err, bdf2_with_err)
            )

        y_new, err, nk_stats = lax.cond(method < 3, explicit_step, implicit_step)

        # Compute error ratio
        err_ratio = scaled_error_norm(err, y_new, model.grid, pid_params.atol, pid_params.rtol)
        err_ratio = jnp.maximum(err_ratio, 1e-10)  # Avoid division issues

        # Accept/reject decision
        accept = err_ratio <= 1.0

        # Check for finite values
        finite = is_finite(y_new)
        accept = jnp.logical_and(accept, finite)

        # For implicit: also check NK convergence
        accept = lax.cond(
            method >= 3,
            lambda: jnp.logical_and(accept, nk_stats.converged),
            lambda: accept
        )

        # Compute new dt
        dt_new, new_controller = propose_dt(
            method=method,
            grid=model.grid,
            params=model.params,
            state=state.y,
            t=state.t,
            dt_old=dt_clamped,
            err_ratio=err_ratio,
            controller_state=state.controller,
            nk_stats=nk_stats,
            cfl_params=cfl_params,
            pid_params=pid_params,
            order=order
        )

        # Update state based on accept/reject
        def accept_step():
            # Update history if it's time to save
            should_save = (state.n_accepted % save_every) == 0
            new_t_hist = lax.cond(
                should_save,
                lambda: state.t_history.at[state.write_idx].set(state.t + dt_clamped),
                lambda: state.t_history
            )
            new_y_hist = lax.cond(
                should_save,
                lambda: save_to_history(state.y_history, y_new, model.grid, state.write_idx, True),
                lambda: state.y_history
            )
            new_dt_hist = lax.cond(
                should_save,
                lambda: state.dt_history.at[state.write_idx - 1].set(dt_clamped),
                lambda: state.dt_history
            )
            new_write_idx = lax.cond(
                should_save,
                lambda: state.write_idx + 1,
                lambda: state.write_idx
            )

            return AdaptiveState(
                t=state.t + dt_clamped,
                y=y_new,
                dt=dt_new,
                y_prev=state.y,
                dt_prev=dt_clamped,
                step_count=state.step_count + 1,
                controller=new_controller,
                t_history=new_t_hist,
                y_history=new_y_hist,
                dt_history=new_dt_hist,
                write_idx=new_write_idx,
                n_accepted=state.n_accepted + 1,
                n_rejected=state.n_rejected,
                status=state.status
            )

        def reject_step():
            # Keep state, update dt and reject count
            # More aggressive shrink on reject
            dt_reject = jnp.maximum(dt_new * 0.5, pid_params.dt_min)

            # Check for dt too small
            new_status = lax.cond(
                dt_reject <= pid_params.dt_min * 1.1,
                lambda: jnp.array(StatusCode.DT_TOO_SMALL, dtype=jnp.int32),
                lambda: state.status
            )

            return AdaptiveState(
                t=state.t,
                y=state.y,
                dt=dt_reject,
                y_prev=state.y_prev,
                dt_prev=state.dt_prev,
                step_count=state.step_count,
                controller=state.controller,
                t_history=state.t_history,
                y_history=state.y_history,
                dt_history=state.dt_history,
                write_idx=state.write_idx,
                n_accepted=state.n_accepted,
                n_rejected=state.n_rejected + 1,
                status=new_status
            )

        new_state = lax.cond(accept, accept_step, reject_step)

        # Check for non-finite
        new_state = lax.cond(
            jnp.logical_not(finite),
            lambda: new_state._replace(status=jnp.array(StatusCode.NON_FINITE_VALUES, dtype=jnp.int32)),
            lambda: new_state
        )

        return new_state

    # Run integration
    final_state = lax.while_loop(cond_fn, body_fn, init_state)

    # Set final status
    final_status = lax.cond(
        final_state.t >= t_end,
        lambda: jnp.array(StatusCode.SUCCESS, dtype=jnp.int32),
        lambda: lax.cond(
            final_state.write_idx >= max_saves,
            lambda: jnp.array(StatusCode.MAX_STEPS_REACHED, dtype=jnp.int32),
            lambda: final_state.status
        )
    )

    return AdaptiveResult(
        t_final=final_state.t,
        y_final=final_state.y,
        t_history=final_state.t_history,
        y_history=final_state.y_history,
        dt_history=final_state.dt_history,
        n_steps=final_state.write_idx,
        n_accepted=final_state.n_accepted,
        n_rejected=final_state.n_rejected,
        status=final_status
    )


# =============================================================================
# Fixed-Step Integration (simpler, faster for known stable dt)
# =============================================================================

def integrate_fixed_dt(
    model: MOLModel,
    y0: StateDict,
    t0: float,
    t_end: float,
    dt: float,
    method: int = IntegratorType.RK4,
    save_every: int = 1,
    preconditioner: Optional[Preconditioner] = None,
    nk_params: Optional[NKParams] = None
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], StateDict]:
    """
    Fixed time step integration using lax.scan.

    Faster than adaptive for problems where stable dt is known.

    Args:
        model: MOLModel
        y0: Initial state
        t0: Initial time
        t_end: Final time
        dt: Fixed time step
        method: IntegratorType
        save_every: Save every N steps
        preconditioner: For implicit methods
        nk_params: For implicit methods

    Returns:
        Tuple of (t_history, y_history, final_state)
    """
    if preconditioner is None:
        preconditioner = IdentityPreconditioner()
    if nk_params is None:
        nk_params = NKParams()

    n_steps = int((t_end - t0) / dt) + 1
    n_saves = n_steps // save_every + 1

    # Allocate output
    t_history = allocate_scalar_history(n_saves, model.dtype)
    y_history = allocate_state_history(y0, model.grid, n_saves, interior_only=True, dtype=model.dtype)

    class ScanState(NamedTuple):
        t: jnp.ndarray
        y: StateDict
        y_prev: StateDict
        dt_prev: jnp.ndarray
        step: jnp.ndarray

    def scan_body(carry: ScanState, _) -> Tuple[ScanState, Tuple[jnp.ndarray, StateDict]]:
        """Single fixed step."""

        def explicit():
            return step_explicit(model, carry.y, carry.t, dt), None

        def implicit():
            is_be = method == 3
            is_cn = method == 4

            def do_be():
                y_new, _ = be_step(model, carry.y, carry.t, dt, preconditioner, nk_params)
                return y_new

            def do_cn():
                y_new, _ = cn_step(model, carry.y, carry.t, dt, preconditioner, nk_params)
                return y_new

            def do_bdf2():
                # BDF2 with BE startup for first step
                use_be = carry.step < 1
                y_new = lax.cond(
                    use_be,
                    lambda: be_step(model, carry.y, carry.t, dt, preconditioner, nk_params)[0],
                    lambda: bdf2_step(model, carry.y, carry.y_prev, carry.t, dt, carry.dt_prev, preconditioner, nk_params)[0]
                )
                return y_new

            y_new = lax.cond(
                is_be,
                do_be,
                lambda: lax.cond(is_cn, do_cn, do_bdf2)
            )
            return y_new, None

        y_new, _ = lax.cond(method < 3, explicit, implicit)

        new_carry = ScanState(
            t=carry.t + dt,
            y=y_new,
            y_prev=carry.y,
            dt_prev=jnp.array(dt, dtype=model.dtype),
            step=carry.step + 1
        )

        # Output for saving
        should_save = (carry.step % save_every) == 0
        t_out = carry.t + dt
        y_out = y_new

        return new_carry, (t_out, y_out)

    init_carry = ScanState(
        t=jnp.array(t0, dtype=model.dtype),
        y=y0,
        y_prev=y0,
        dt_prev=jnp.array(dt, dtype=model.dtype),
        step=jnp.array(0, dtype=jnp.int32)
    )

    final_carry, (t_outs, y_outs) = lax.scan(scan_body, init_carry, None, length=n_steps)

    return t_outs, y_outs, final_carry.y


# =============================================================================
# IMEX Adaptive Integration
# =============================================================================

def adaptive_integrate_imex(
    model: MOLModel,
    y0: StateDict,
    t0: float,
    t_end: float,
    dt0: float,
    fft_cache,
    diffusivities: Dict[str, float],
    use_strang: bool = True,
    max_steps: int = 10000,
    cfl_params: Optional[CFLParams] = None,
    pid_params: Optional[PIDParams] = None,
    save_every: int = 1
) -> AdaptiveResult:
    """
    Adaptive IMEX time integration using lax.while_loop.

    Uses IMEX splitting: diffusion implicit (FFT), reaction explicit.
    The dt is NOT limited by diffusion CFL since diffusion is implicit.

    Args:
        model: MOLModel to integrate
        y0: Initial state (with ghost cells)
        t0: Initial time
        t_end: Final time
        dt0: Initial time step
        fft_cache: Precomputed FFT cache for diffusion solve
        diffusivities: Dict mapping field name to diffusion coefficient
        use_strang: Use Strang splitting (True) or IMEX Euler (False)
        max_steps: Maximum number of steps
        cfl_params: CFL parameters (for advection/reaction limiting)
        pid_params: PID parameters
        save_every: Save every N accepted steps

    Returns:
        AdaptiveResult with final state and histories
    """
    from moljax.core.dt_policy import propose_dt_imex, imex_cfl_dt

    dtype = model.dtype

    if cfl_params is None:
        cfl_params = CFLParams()
    if pid_params is None:
        pid_params = PIDParams()

    # Order for error estimation
    order = 2 if use_strang else 1

    # Compute initial IMEX CFL-limited dt (no diffusion limit)
    dt_cfl = imex_cfl_dt(model.grid, model.params, cfl_params)
    dt_init = jnp.minimum(jnp.array(dt0, dtype=dtype), dt_cfl)

    # Allocate output buffers
    max_saves = max_steps // save_every + 1
    t_history = allocate_scalar_history(max_saves, dtype)
    y_history = allocate_state_history(y0, model.grid, max_saves, interior_only=True, dtype=dtype)
    dt_history = allocate_scalar_history(max_saves, dtype)

    # Initialize state
    controller = create_initial_controller_state(dtype)

    init_state = AdaptiveState(
        t=jnp.array(t0, dtype=dtype),
        y=y0,
        dt=dt_init,
        y_prev=y0,
        dt_prev=dt_init,
        step_count=jnp.array(0, dtype=jnp.int32),
        controller=controller,
        t_history=t_history.at[0].set(t0),
        y_history=save_to_history(y_history, y0, model.grid, 0, interior_only=True),
        dt_history=dt_history,
        write_idx=jnp.array(1, dtype=jnp.int32),
        n_accepted=jnp.array(0, dtype=jnp.int32),
        n_rejected=jnp.array(0, dtype=jnp.int32),
        status=jnp.array(StatusCode.RUNNING, dtype=jnp.int32)
    )

    def cond_fn(state: AdaptiveState) -> jnp.ndarray:
        running = state.status == StatusCode.RUNNING
        not_done = state.t < t_end
        space_left = state.write_idx < max_saves
        return jnp.logical_and(running, jnp.logical_and(not_done, space_left))

    def body_fn(state: AdaptiveState) -> AdaptiveState:
        # Clamp dt to not overshoot
        dt_clamped = jnp.minimum(state.dt, t_end - state.t)

        # Take IMEX step with error estimation
        if use_strang:
            y_new, err = estimate_error_imex_doubling(
                model, state.y, state.t, dt_clamped, fft_cache, diffusivities, use_strang=True
            )
        else:
            y_new, err = estimate_error_imex_doubling(
                model, state.y, state.t, dt_clamped, fft_cache, diffusivities, use_strang=False
            )

        # Compute error ratio
        err_ratio = scaled_error_norm(err, y_new, model.grid, pid_params.atol, pid_params.rtol)
        err_ratio = jnp.maximum(err_ratio, 1e-10)

        # Accept/reject decision
        accept = err_ratio <= 1.0

        # Check for finite values
        finite = is_finite(y_new)
        accept = jnp.logical_and(accept, finite)

        # Compute new dt using IMEX policy (no diffusion limit)
        dt_new, new_controller = propose_dt_imex(
            grid=model.grid,
            params=model.params,
            state=state.y,
            t=state.t,
            dt_old=dt_clamped,
            err_ratio=err_ratio,
            controller_state=state.controller,
            cfl_params=cfl_params,
            pid_params=pid_params,
            order=order
        )

        # Update state based on accept/reject
        def accept_step():
            should_save = (state.n_accepted % save_every) == 0
            new_t_hist = lax.cond(
                should_save,
                lambda: state.t_history.at[state.write_idx].set(state.t + dt_clamped),
                lambda: state.t_history
            )
            new_y_hist = lax.cond(
                should_save,
                lambda: save_to_history(state.y_history, y_new, model.grid, state.write_idx, True),
                lambda: state.y_history
            )
            new_dt_hist = lax.cond(
                should_save,
                lambda: state.dt_history.at[state.write_idx - 1].set(dt_clamped),
                lambda: state.dt_history
            )
            new_write_idx = lax.cond(
                should_save,
                lambda: state.write_idx + 1,
                lambda: state.write_idx
            )

            return AdaptiveState(
                t=state.t + dt_clamped,
                y=y_new,
                dt=dt_new,
                y_prev=state.y,
                dt_prev=dt_clamped,
                step_count=state.step_count + 1,
                controller=new_controller,
                t_history=new_t_hist,
                y_history=new_y_hist,
                dt_history=new_dt_hist,
                write_idx=new_write_idx,
                n_accepted=state.n_accepted + 1,
                n_rejected=state.n_rejected,
                status=state.status
            )

        def reject_step():
            dt_reject = jnp.maximum(dt_new * 0.5, pid_params.dt_min)
            new_status = lax.cond(
                dt_reject <= pid_params.dt_min * 1.1,
                lambda: jnp.array(StatusCode.DT_TOO_SMALL, dtype=jnp.int32),
                lambda: state.status
            )

            return AdaptiveState(
                t=state.t,
                y=state.y,
                dt=dt_reject,
                y_prev=state.y_prev,
                dt_prev=state.dt_prev,
                step_count=state.step_count,
                controller=state.controller,
                t_history=state.t_history,
                y_history=state.y_history,
                dt_history=state.dt_history,
                write_idx=state.write_idx,
                n_accepted=state.n_accepted,
                n_rejected=state.n_rejected + 1,
                status=new_status
            )

        new_state = lax.cond(accept, accept_step, reject_step)

        # Check for non-finite
        new_state = lax.cond(
            jnp.logical_not(finite),
            lambda: new_state._replace(status=jnp.array(StatusCode.NON_FINITE_VALUES, dtype=jnp.int32)),
            lambda: new_state
        )

        return new_state

    # Run integration
    final_state = lax.while_loop(cond_fn, body_fn, init_state)

    # Set final status
    final_status = lax.cond(
        final_state.t >= t_end,
        lambda: jnp.array(StatusCode.SUCCESS, dtype=jnp.int32),
        lambda: lax.cond(
            final_state.write_idx >= max_saves,
            lambda: jnp.array(StatusCode.MAX_STEPS_REACHED, dtype=jnp.int32),
            lambda: final_state.status
        )
    )

    return AdaptiveResult(
        t_final=final_state.t,
        y_final=final_state.y,
        t_history=final_state.t_history,
        y_history=final_state.y_history,
        dt_history=final_state.dt_history,
        n_steps=final_state.write_idx,
        n_accepted=final_state.n_accepted,
        n_rejected=final_state.n_rejected,
        status=final_status
    )


def integrate_imex_fixed_dt(
    model: MOLModel,
    y0: StateDict,
    t0: float,
    t_end: float,
    dt: float,
    fft_cache,
    diffusivities: Dict[str, float],
    use_strang: bool = True,
    save_every: int = 1
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray], StateDict]:
    """
    Fixed time step IMEX integration using lax.scan.

    Args:
        model: MOLModel
        y0: Initial state
        t0: Initial time
        t_end: Final time
        dt: Fixed time step
        fft_cache: FFT cache
        diffusivities: Diffusion coefficients
        use_strang: Use Strang splitting or IMEX Euler
        save_every: Save every N steps

    Returns:
        Tuple of (t_history, y_history, final_state)
    """
    n_steps = int((t_end - t0) / dt) + 1
    n_saves = n_steps // save_every + 1

    # Allocate output
    t_history = allocate_scalar_history(n_saves, model.dtype)
    y_history = allocate_state_history(y0, model.grid, n_saves, interior_only=True, dtype=model.dtype)

    class ScanState(NamedTuple):
        t: jnp.ndarray
        y: StateDict
        step: jnp.ndarray

    def scan_body(carry: ScanState, _) -> Tuple[ScanState, Tuple[jnp.ndarray, StateDict]]:
        if use_strang:
            y_new = imex_strang_step(model, carry.y, carry.t, dt, fft_cache, diffusivities)
        else:
            y_new = imex_euler_step(model, carry.y, carry.t, dt, fft_cache, diffusivities)

        new_carry = ScanState(
            t=carry.t + dt,
            y=y_new,
            step=carry.step + 1
        )

        return new_carry, (carry.t + dt, y_new)

    init_carry = ScanState(
        t=jnp.array(t0, dtype=model.dtype),
        y=y0,
        step=jnp.array(0, dtype=jnp.int32)
    )

    final_carry, (t_outs, y_outs) = lax.scan(scan_body, init_carry, None, length=n_steps)

    return t_outs, y_outs, final_carry.y
