"""
Time step policy for MOL-JAX.

This module provides automatic time step selection:
- Heisenberg/CFL limiter for explicit methods
- PID/PI controller for adaptive stepping
- Implicit robustness limiter based on Newton-Krylov stats
- IMEX-aware dt limiting (no diffusion limit when diffusion is implicit)

Design decisions:
- All policies are JIT-compatible (no Python branching)
- Controller state is passed through carry for while_loop compatibility
- Conservative default constants for stability
- dt is always clamped to [dt_min, dt_max]
- IMEX methods: diffusion CFL is not limiting when treated implicitly
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, NamedTuple
import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.grid import Grid1D, Grid2D, GridType
from moljax.core.state import StateDict


@dataclass(frozen=True)
class CFLParams:
    """
    Parameters for CFL-based time step limiting.

    Attributes:
        safety: Safety factor (default 0.9)
        dt_min: Minimum allowed dt (default 1e-12)
        dt_max: Maximum allowed dt (default 1.0)
        cfl_advection: CFL number for advection (default 0.5)
        cfl_diffusion: CFL number for diffusion (default 0.25 for 2D)
        cfl_wave: CFL number for wave/acoustic (default 0.5)
    """
    safety: float = 0.9
    dt_min: float = 1e-12
    dt_max: float = 1.0
    cfl_advection: float = 0.5
    cfl_diffusion: float = 0.25
    cfl_wave: float = 0.5


@dataclass(frozen=True)
class PIDParams:
    """
    Parameters for PID/PI adaptive time step controller.

    The controller computes:
        dt_new = dt * safety * err_ratio^(-kI) * prev_err_ratio^(kP)

    where err_ratio = error_norm (should be <= 1 for accepted step).

    Attributes:
        safety: Safety factor (default 0.9)
        kI: Integral coefficient (default 0.7/order for PI controller)
        kP: Proportional coefficient (default 0.4/order for PI controller)
        dt_min: Minimum allowed dt (default 1e-12)
        dt_max: Maximum allowed dt (default 1.0)
        max_factor: Maximum dt growth factor (default 5.0)
        min_factor: Maximum dt shrink factor (default 0.2)
        atol: Absolute tolerance for error (default 1e-6)
        rtol: Relative tolerance for error (default 1e-4)
    """
    safety: float = 0.9
    kI: float = 0.7  # Scaled by 1/order in controller
    kP: float = 0.4  # Scaled by 1/order in controller
    dt_min: float = 1e-12
    dt_max: float = 1.0
    max_factor: float = 5.0
    min_factor: float = 0.2
    atol: float = 1e-6
    rtol: float = 1e-4


class ControllerState(NamedTuple):
    """
    State for adaptive dt controller, passed through while_loop carry.

    Attributes:
        prev_err_ratio: Error ratio from previous step
        prevprev_err_ratio: Error ratio from two steps ago (for PID)
        consecutive_rejects: Number of consecutive rejected steps
    """
    prev_err_ratio: jnp.ndarray
    prevprev_err_ratio: jnp.ndarray
    consecutive_rejects: jnp.ndarray


def create_initial_controller_state(dtype: jnp.dtype = jnp.float64) -> ControllerState:
    """Create initial controller state."""
    return ControllerState(
        prev_err_ratio=jnp.array(1.0, dtype=dtype),
        prevprev_err_ratio=jnp.array(1.0, dtype=dtype),
        consecutive_rejects=jnp.array(0, dtype=jnp.int32)
    )


def heisenberg_cfl_dt(
    grid: GridType,
    params: Dict[str, Any],
    cfl_params: CFLParams
) -> jnp.ndarray:
    """
    Compute CFL-limited time step (Heisenberg limiter).

    Considers:
    - Advection: dt <= CFL * dx / v_max
    - Diffusion: dt <= CFL * dx^2 / D_max
    - Wave: dt <= CFL * dx / c

    Args:
        grid: Grid defining spacing
        params: Dict with velocities, diffusivities, wave speed
        cfl_params: CFL parameters

    Returns:
        Maximum stable dt
    """
    dx = grid.min_dx
    dx2 = grid.min_dx2

    # Start with dt_max
    dt = jnp.array(cfl_params.dt_max)

    # Advection CFL
    vx = jnp.abs(params.get('vx', 0.0))
    vy = jnp.abs(params.get('vy', 0.0))
    v_max = vx + vy
    dt_adv = lax.cond(
        v_max > 1e-14,
        lambda: cfl_params.cfl_advection * dx / v_max,
        lambda: cfl_params.dt_max
    )
    dt = jnp.minimum(dt, dt_adv)

    # Diffusion CFL
    D = params.get('D', 0.0)
    Du = params.get('Du', 0.0)
    Dv = params.get('Dv', 0.0)
    D_max = jnp.maximum(D, jnp.maximum(Du, Dv))
    dt_diff = lax.cond(
        D_max > 1e-14,
        lambda: cfl_params.cfl_diffusion * dx2 / D_max,
        lambda: cfl_params.dt_max
    )
    dt = jnp.minimum(dt, dt_diff)

    # Wave/acoustic CFL
    K = params.get('K', 0.0)
    rho = params.get('rho', 1.0)
    c_wave = params.get('wave_speed', 0.0)
    c_acoustic = lax.cond(
        K > 1e-14,
        lambda: jnp.sqrt(K / rho),
        lambda: 0.0
    )
    c = jnp.maximum(c_wave, c_acoustic)
    dt_wave = lax.cond(
        c > 1e-14,
        lambda: cfl_params.cfl_wave * dx / c,
        lambda: cfl_params.dt_max
    )
    dt = jnp.minimum(dt, dt_wave)

    # Apply safety factor and clamp
    dt = cfl_params.safety * dt
    dt = jnp.clip(dt, cfl_params.dt_min, cfl_params.dt_max)

    return dt


def pid_controller_dt(
    dt_old: jnp.ndarray,
    err_ratio: jnp.ndarray,
    controller_state: ControllerState,
    pid_params: PIDParams,
    order: int = 4
) -> Tuple[jnp.ndarray, ControllerState]:
    """
    PID/PI controller for adaptive time step.

    Computes new dt based on error ratio using PI formula:
        factor = (1/err_ratio)^(kI/order) * (prev_err/err_ratio)^(kP/order)
        dt_new = dt_old * safety * clip(factor, min_factor, max_factor)

    Args:
        dt_old: Current time step
        err_ratio: Current error norm (should be <= 1 for accepted step)
        controller_state: Previous controller state
        pid_params: PID parameters
        order: Order of the method (for scaling coefficients)

    Returns:
        Tuple of (dt_new, new_controller_state)
    """
    # Scale coefficients by order
    kI = pid_params.kI / order
    kP = pid_params.kP / order

    # Avoid division by zero
    err_safe = jnp.maximum(err_ratio, 1e-10)
    prev_err_safe = jnp.maximum(controller_state.prev_err_ratio, 1e-10)

    # PI formula
    factor = jnp.power(1.0 / err_safe, kI) * jnp.power(prev_err_safe / err_safe, kP)

    # Clip factor
    factor = jnp.clip(factor * pid_params.safety, pid_params.min_factor, pid_params.max_factor)

    # Compute new dt
    dt_new = dt_old * factor
    dt_new = jnp.clip(dt_new, pid_params.dt_min, pid_params.dt_max)

    # Update controller state
    new_state = ControllerState(
        prev_err_ratio=err_ratio,
        prevprev_err_ratio=controller_state.prev_err_ratio,
        consecutive_rejects=jnp.array(0, dtype=jnp.int32)
    )

    return dt_new, new_state


def handle_rejected_step(
    dt_old: jnp.ndarray,
    err_ratio: jnp.ndarray,
    controller_state: ControllerState,
    pid_params: PIDParams,
    order: int = 4
) -> Tuple[jnp.ndarray, ControllerState]:
    """
    Handle a rejected step by reducing dt more aggressively.

    Args:
        dt_old: Current time step that was rejected
        err_ratio: Error ratio that caused rejection (> 1)
        controller_state: Current controller state
        pid_params: PID parameters
        order: Order of the method

    Returns:
        Tuple of (dt_new, new_controller_state)
    """
    # Simple reduction based on error
    # dt_new = dt_old * safety * (1/err_ratio)^(1/order)
    kI = pid_params.kI / order
    err_safe = jnp.maximum(err_ratio, 1.0)
    factor = pid_params.safety * jnp.power(1.0 / err_safe, kI)
    factor = jnp.clip(factor, pid_params.min_factor, 1.0)  # Never grow on reject

    dt_new = dt_old * factor
    dt_new = jnp.clip(dt_new, pid_params.dt_min, pid_params.dt_max)

    # Update state: increment reject counter
    n_rejects = controller_state.consecutive_rejects + 1
    new_state = ControllerState(
        prev_err_ratio=controller_state.prev_err_ratio,  # Keep old
        prevprev_err_ratio=controller_state.prevprev_err_ratio,
        consecutive_rejects=n_rejects
    )

    return dt_new, new_state


class NKStats(NamedTuple):
    """Newton-Krylov solver statistics for implicit dt control."""
    converged: jnp.ndarray
    newton_iters: jnp.ndarray
    lin_iters: jnp.ndarray
    final_res_norm: jnp.ndarray


def implicit_robustness_dt(
    dt_old: jnp.ndarray,
    nk_stats: NKStats,
    dt_min: float = 1e-12,
    dt_max: float = 1.0,
    growth_factor: float = 1.2,
    shrink_factor: float = 0.5,
    fast_iters_threshold: int = 3
) -> jnp.ndarray:
    """
    Adjust dt based on Newton-Krylov convergence.

    - Shrink dt if NK didn't converge
    - Shrink dt if NK took many iterations
    - Grow dt if NK converged quickly

    Args:
        dt_old: Current time step
        nk_stats: Statistics from NK solver
        dt_min: Minimum dt
        dt_max: Maximum dt
        growth_factor: Factor to grow dt on fast convergence
        shrink_factor: Factor to shrink dt on slow/failed convergence
        fast_iters_threshold: Threshold for "fast" convergence

    Returns:
        New dt
    """
    # Shrink if not converged
    dt_new = lax.cond(
        nk_stats.converged,
        lambda: dt_old,
        lambda: dt_old * shrink_factor
    )

    # Adjust based on iteration count (only if converged)
    dt_new = lax.cond(
        jnp.logical_and(nk_stats.converged, nk_stats.newton_iters <= fast_iters_threshold),
        lambda: jnp.minimum(dt_new * growth_factor, dt_max),
        lambda: dt_new
    )

    return jnp.clip(dt_new, dt_min, dt_max)


def propose_dt(
    method: int,
    grid: GridType,
    params: Dict[str, Any],
    state: StateDict,
    t: jnp.ndarray,
    dt_old: jnp.ndarray,
    err_ratio: jnp.ndarray,
    controller_state: ControllerState,
    nk_stats: Optional[NKStats] = None,
    cfl_params: Optional[CFLParams] = None,
    pid_params: Optional[PIDParams] = None,
    order: int = 4
) -> Tuple[jnp.ndarray, ControllerState]:
    """
    Unified dt proposal function for all methods.

    Args:
        method: Integrator type (0-2 explicit, 3-5 implicit)
        grid: Grid defining spacing
        params: Model parameters
        state: Current state
        t: Current time
        dt_old: Previous time step
        err_ratio: Error ratio from current step
        controller_state: Controller state
        nk_stats: NK statistics (for implicit methods)
        cfl_params: CFL parameters (default created if None)
        pid_params: PID parameters (default created if None)
        order: Method order

    Returns:
        Tuple of (proposed_dt, new_controller_state)
    """
    if cfl_params is None:
        cfl_params = CFLParams()
    if pid_params is None:
        pid_params = PIDParams()

    # Check if step was accepted
    accepted = err_ratio <= 1.0

    # Compute base dt from controller
    def accepted_dt():
        dt_pid, new_state = pid_controller_dt(
            dt_old, err_ratio, controller_state, pid_params, order
        )
        return dt_pid, new_state

    def rejected_dt():
        dt_rej, new_state = handle_rejected_step(
            dt_old, err_ratio, controller_state, pid_params, order
        )
        return dt_rej, new_state

    dt_adaptive, new_controller_state = lax.cond(
        accepted,
        accepted_dt,
        rejected_dt
    )

    # For explicit methods (0-2), also enforce CFL
    is_explicit = method < 3
    dt_cfl = heisenberg_cfl_dt(grid, params, cfl_params)

    dt_final = lax.cond(
        is_explicit,
        lambda: jnp.minimum(dt_adaptive, dt_cfl),
        lambda: dt_adaptive
    )

    # For implicit methods with NK stats, apply robustness limiter
    if nk_stats is not None:
        is_implicit = method >= 3
        dt_robust = implicit_robustness_dt(dt_final, nk_stats)
        dt_final = lax.cond(
            is_implicit,
            lambda: dt_robust,
            lambda: dt_final
        )

    # Final clamp
    dt_final = jnp.clip(dt_final, pid_params.dt_min, pid_params.dt_max)

    return dt_final, new_controller_state


def compute_error_order(method: int) -> int:
    """
    Get the error estimation order for a given method.

    Args:
        method: Integrator type enum value

    Returns:
        Order used for error estimation
    """
    # EULER=0, SSPRK3=1, RK4=2, BE=3, CN=4, BDF2=5, IMEX_EULER=6, IMEX_SSPRK3=7
    orders = [1, 3, 4, 1, 2, 2, 1, 2]
    return orders[min(method, len(orders) - 1)]


# =============================================================================
# IMEX-Specific CFL Limits
# =============================================================================

def imex_cfl_dt(
    grid,
    params: Dict[str, Any],
    cfl_params: CFLParams
) -> jnp.ndarray:
    """
    Compute CFL-limited time step for IMEX methods.

    For IMEX with implicit diffusion, the diffusion CFL is NOT limiting.
    Only advection/wave CFL and reaction stiffness limits apply.

    Args:
        grid: Grid defining spacing
        params: Dict with velocities, wave speed, reaction rates
        cfl_params: CFL parameters

    Returns:
        Maximum stable dt for explicit part of IMEX
    """
    dx = grid.min_dx

    # Start with dt_max
    dt = jnp.array(cfl_params.dt_max)

    # Advection CFL (still explicit in IMEX)
    vx = jnp.abs(params.get('vx', 0.0))
    vy = jnp.abs(params.get('vy', 0.0))
    v_max = vx + vy
    dt_adv = lax.cond(
        v_max > 1e-14,
        lambda: cfl_params.cfl_advection * dx / v_max,
        lambda: cfl_params.dt_max
    )
    dt = jnp.minimum(dt, dt_adv)

    # Wave/acoustic CFL (still explicit if hyperbolic)
    K = params.get('K', 0.0)
    rho = params.get('rho', 1.0)
    c_wave = params.get('wave_speed', 0.0)
    c_acoustic = lax.cond(
        K > 1e-14,
        lambda: jnp.sqrt(K / rho),
        lambda: 0.0
    )
    c = jnp.maximum(c_wave, c_acoustic)
    dt_wave = lax.cond(
        c > 1e-14,
        lambda: cfl_params.cfl_wave * dx / c,
        lambda: cfl_params.dt_max
    )
    dt = jnp.minimum(dt, dt_wave)

    # Reaction stiffness limit (heuristic)
    # For Gray-Scott: F+k ~ 0.1, so dt ~ 5
    # This is a soft limit; reactions are explicit in IMEX
    F = params.get('F', 0.0)
    k = params.get('k', 0.0)
    r = params.get('r', 0.0)  # For Fisher-KPP
    reaction_rate = jnp.maximum(F + k, r)
    dt_react = lax.cond(
        reaction_rate > 1e-14,
        lambda: 0.5 / reaction_rate,
        lambda: cfl_params.dt_max
    )
    dt = jnp.minimum(dt, dt_react)

    # NOTE: Diffusion CFL is NOT included here since diffusion is implicit

    # Apply safety factor and clamp
    dt = cfl_params.safety * dt
    dt = jnp.clip(dt, cfl_params.dt_min, cfl_params.dt_max)

    return dt


def estimate_reaction_stiffness(
    state: StateDict,
    params: Dict[str, Any]
) -> jnp.ndarray:
    """
    Estimate reaction stiffness for dt limiting.

    Returns max |df/du| across all fields, which limits the explicit
    reaction time step.

    Args:
        state: Current state
        params: Model parameters

    Returns:
        Estimated reaction stiffness bound
    """
    # Common reaction stiffness estimates
    stiffness = jnp.array(0.0)

    # Gray-Scott: |df/du| ~ v^2 + F, |df/dv| ~ 2*u*v + (F+k)
    if 'F' in params and 'k' in params:
        F = params['F']
        k = params['k']
        stiffness = jnp.maximum(stiffness, F + k)
        if 'u' in state and 'v' in state:
            # More refined estimate using current state
            v_max = jnp.max(jnp.abs(state['v']))
            u_max = jnp.max(jnp.abs(state['u']))
            stiffness = jnp.maximum(stiffness, v_max**2 + F)
            stiffness = jnp.maximum(stiffness, 2*u_max*v_max + F + k)

    # Fisher-KPP: |df/du| = r * |1 - 2u| <= r
    if 'r' in params:
        stiffness = jnp.maximum(stiffness, params['r'])

    return stiffness


def propose_dt_imex(
    grid,
    params: Dict[str, Any],
    state: StateDict,
    t: jnp.ndarray,
    dt_old: jnp.ndarray,
    err_ratio: jnp.ndarray,
    controller_state: ControllerState,
    cfl_params: Optional[CFLParams] = None,
    pid_params: Optional[PIDParams] = None,
    order: int = 2
) -> Tuple[jnp.ndarray, ControllerState]:
    """
    Propose dt for IMEX methods.

    Similar to propose_dt but uses IMEX CFL (no diffusion limit).

    Args:
        grid: Grid defining spacing
        params: Model parameters
        state: Current state
        t: Current time
        dt_old: Previous time step
        err_ratio: Error ratio from current step
        controller_state: Controller state
        cfl_params: CFL parameters
        pid_params: PID parameters
        order: Method order

    Returns:
        Tuple of (proposed_dt, new_controller_state)
    """
    if cfl_params is None:
        cfl_params = CFLParams()
    if pid_params is None:
        pid_params = PIDParams()

    # Check if step was accepted
    accepted = err_ratio <= 1.0

    # Compute base dt from controller
    def accepted_dt():
        dt_pid, new_state = pid_controller_dt(
            dt_old, err_ratio, controller_state, pid_params, order
        )
        return dt_pid, new_state

    def rejected_dt():
        dt_rej, new_state = handle_rejected_step(
            dt_old, err_ratio, controller_state, pid_params, order
        )
        return dt_rej, new_state

    dt_adaptive, new_controller_state = lax.cond(
        accepted,
        accepted_dt,
        rejected_dt
    )

    # Apply IMEX CFL limit (no diffusion constraint)
    dt_cfl = imex_cfl_dt(grid, params, cfl_params)
    dt_final = jnp.minimum(dt_adaptive, dt_cfl)

    # Final clamp
    dt_final = jnp.clip(dt_final, pid_params.dt_min, pid_params.dt_max)

    return dt_final, new_controller_state
