"""
Exponential Time Differencing (ETD) integrators using FFT-diagonalized operators.

ETD methods are exponential integrators for semi-linear PDEs of the form:
    u_t = L*u + N(u)

where L is a linear operator (FFT-diagonalizable) and N(u) is nonlinear.

Key methods:
- ETD1 (Exponential Euler): 1st order, simple and stable
- ETD2 (Exponential Trapezoidal): 2nd order
- ETDRK4 (Cox-Matthews): 4th order Runge-Kutta style

The key advantage of ETD methods is that the linear part L is solved exactly
via exp(dt*L), removing stability restrictions from the linear operator.
This allows much larger timesteps for stiff problems.

All methods work with FFTLinearOperator instances for efficient computation.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Optional

import jax.numpy as jnp
from jax import lax

from moljax.core.fft_operators import FFTLinearOperator
from moljax.core.state import StateDict


class ETDStepResult(NamedTuple):
    """Result of an ETD step."""
    state: StateDict
    t: float
    diagnostics: dict


def _phi1(z: jnp.ndarray) -> jnp.ndarray:
    """Compute φ₁(z) = (exp(z) - 1) / z with regularization at z=0.

    Uses Taylor expansion for |z| < 1e-4 to avoid numerical instability.
    φ₁(z) = 1 + z/2 + z²/6 + z³/24 + O(z⁴)
    """
    # Taylor expansion for small |z|
    taylor = 1.0 + z/2.0 + z**2/6.0 + z**3/24.0

    # Direct formula for larger |z|
    exp_z = jnp.exp(z)
    direct = (exp_z - 1.0) / z

    return jnp.where(jnp.abs(z) < 1e-4, taylor, direct)


def _phi2(z: jnp.ndarray) -> jnp.ndarray:
    """Compute φ₂(z) = (exp(z) - 1 - z) / z² with regularization at z=0.

    φ₂(z) = 1/2 + z/6 + z²/24 + z³/120 + O(z⁴)
    """
    taylor = 0.5 + z/6.0 + z**2/24.0 + z**3/120.0

    exp_z = jnp.exp(z)
    direct = (exp_z - 1.0 - z) / (z * z)

    return jnp.where(jnp.abs(z) < 1e-4, taylor, direct)


def _phi3(z: jnp.ndarray) -> jnp.ndarray:
    """Compute φ₃(z) = (exp(z) - 1 - z - z²/2) / z³ with regularization.

    φ₃(z) = 1/6 + z/24 + z²/120 + z³/720 + O(z⁴)
    """
    taylor = 1.0/6.0 + z/24.0 + z**2/120.0 + z**3/720.0

    exp_z = jnp.exp(z)
    direct = (exp_z - 1.0 - z - z**2/2.0) / (z**3)

    return jnp.where(jnp.abs(z) < 1e-4, taylor, direct)


def _etdrk4_coefficients(z: jnp.ndarray) -> tuple:
    """Compute ETDRK4 (Cox-Matthews) coefficients.

    Returns (E, E2, a21, a31, a32, a41, a42, a43, b1, b2, b3, b4)
    for the 4-stage RK scheme.

    Reference: Cox & Matthews (2002), "Exponential Time Differencing for
    Stiff Systems", J. Comput. Phys. 176, 430-455.
    """
    exp_z = jnp.exp(z)
    exp_z2 = jnp.exp(z / 2.0)

    # E = exp(z), E2 = exp(z/2)
    E = exp_z
    E2 = exp_z2

    # φ functions at z and z/2
    phi1_z = _phi1(z)
    phi2_z = _phi2(z)
    phi3_z = _phi3(z)

    phi1_z2 = _phi1(z / 2.0)
    phi2_z2 = _phi2(z / 2.0)

    # Stage coefficients (simplified Cox-Matthews)
    # a coefficients for stages
    a21 = phi1_z2  # dt/2 * phi1(z/2)

    # b coefficients for final combination
    # Using the standard ETDRK4 formula
    b1 = phi1_z - 3*phi2_z + 4*phi3_z
    b2 = 2*phi2_z - 4*phi3_z
    b3 = 2*phi2_z - 4*phi3_z
    b4 = -phi2_z + 4*phi3_z

    return E, E2, phi1_z2, phi1_z, b1, b2, b3, b4


def etd1_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: Dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
) -> StateDict:
    """ETD1 (Exponential Euler) step for u_t = L*u + N(u).

    Update formula:
        u_{n+1} = exp(dt*L)*u_n + φ₁(dt*L)*dt*N(u_n)

    where φ₁(z) = (exp(z) - 1) / z

    Args:
        state: Current state (interior values only, no ghost cells)
        t: Current time
        dt: Time step
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u) for each field

    Returns:
        New state after one ETD1 step
    """
    # Compute nonlinear term at current state
    N_state = nonlinear_rhs(state, t)

    result = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)

        if op is not None:
            lam = op.eigenvalues
            z = dt * lam

            # Exponential and phi_1 factors
            exp_z = jnp.exp(z)
            phi1 = _phi1(z)

            # FFT of current state and nonlinear term
            if u_field.ndim == 1:
                u_hat = jnp.fft.fft(u_field)
                N_hat = jnp.fft.fft(N_state[name])

                # ETD1 update in Fourier space
                u_new_hat = exp_z * u_hat + dt * phi1 * N_hat
                result[name] = jnp.real(jnp.fft.ifft(u_new_hat))
            else:
                u_hat = jnp.fft.fft2(u_field)
                N_hat = jnp.fft.fft2(N_state[name])

                u_new_hat = exp_z * u_hat + dt * phi1 * N_hat
                result[name] = jnp.real(jnp.fft.ifft2(u_new_hat))
        else:
            # No linear operator: explicit Euler for nonlinear part
            result[name] = u_field + dt * N_state[name]

    return result


def etd2_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: Dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
    N_prev: Optional[StateDict] = None,
) -> tuple[StateDict, StateDict]:
    """ETD2 (Exponential Adams-Bashforth 2) step.

    Update formula:
        u_{n+1} = exp(dt*L)*u_n + φ₁(dt*L)*dt*N_n + φ₂(dt*L)*dt*(N_n - N_{n-1})

    This is a 2nd order multi-step method requiring N_{n-1} from previous step.

    Args:
        state: Current state
        t: Current time
        dt: Time step
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u)
        N_prev: Nonlinear term from previous step (None for first step)

    Returns:
        Tuple of (new_state, N_current) where N_current should be saved for next step
    """
    N_curr = nonlinear_rhs(state, t)

    # If no previous N, fall back to ETD1
    if N_prev is None:
        return etd1_step(state, t, dt, linear_ops, nonlinear_rhs), N_curr

    result = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)

        if op is not None:
            lam = op.eigenvalues
            z = dt * lam

            exp_z = jnp.exp(z)
            phi1 = _phi1(z)
            phi2 = _phi2(z)

            N_n = N_curr[name]
            N_nm1 = N_prev[name]

            if u_field.ndim == 1:
                u_hat = jnp.fft.fft(u_field)
                N_n_hat = jnp.fft.fft(N_n)
                N_nm1_hat = jnp.fft.fft(N_nm1)

                # ETD2 update
                u_new_hat = (exp_z * u_hat
                            + dt * phi1 * N_n_hat
                            + dt * phi2 * (N_n_hat - N_nm1_hat))
                result[name] = jnp.real(jnp.fft.ifft(u_new_hat))
            else:
                u_hat = jnp.fft.fft2(u_field)
                N_n_hat = jnp.fft.fft2(N_n)
                N_nm1_hat = jnp.fft.fft2(N_nm1)

                u_new_hat = (exp_z * u_hat
                            + dt * phi1 * N_n_hat
                            + dt * phi2 * (N_n_hat - N_nm1_hat))
                result[name] = jnp.real(jnp.fft.ifft2(u_new_hat))
        else:
            # Explicit Adams-Bashforth 2
            result[name] = u_field + dt * (1.5 * N_curr[name] - 0.5 * N_prev[name])

    return result, N_curr


def etdrk4_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: Dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
) -> StateDict:
    """ETDRK4 (Cox-Matthews) 4th order exponential integrator.

    4-stage Runge-Kutta style ETD method achieving O(dt⁴) accuracy.
    The linear part is solved exactly; error is only from nonlinear treatment.

    Reference: Cox & Matthews (2002), J. Comput. Phys. 176, 430-455.

    Update stages:
        a = exp(dt*L/2)*u_n + (dt/2)*φ₁(dt*L/2)*N(u_n)
        b = exp(dt*L/2)*u_n + (dt/2)*φ₁(dt*L/2)*N(a)
        c = exp(dt*L/2)*a + (dt/2)*φ₁(dt*L/2)*(2*N(b) - N(u_n))
        u_{n+1} = exp(dt*L)*u_n + dt*(b₁*N_n + b₂*(N_a + N_b) + b₄*N_c)

    Args:
        state: Current state
        t: Current time
        dt: Time step
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u)

    Returns:
        New state after one ETDRK4 step
    """
    # Stage 1: N at current state
    N_n = nonlinear_rhs(state, t)

    result = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)

        if op is not None:
            lam = op.eigenvalues
            z = dt * lam

            # Get ETDRK4 coefficients
            E, E2, phi1_z2, phi1_z, b1, b2, b3, b4 = _etdrk4_coefficients(z)

            if u_field.ndim == 1:
                fft_func = jnp.fft.fft
                ifft_func = jnp.fft.ifft
            else:
                fft_func = jnp.fft.fft2
                ifft_func = jnp.fft.ifft2

            u_hat = fft_func(u_field)
            N_n_hat = fft_func(N_n[name])

            # Stage a: half step
            a_hat = E2 * u_hat + (dt/2) * phi1_z2 * N_n_hat
            a_field = jnp.real(ifft_func(a_hat))

            # Evaluate N(a)
            N_a = nonlinear_rhs({name: a_field}, t + dt/2)[name]
            N_a_hat = fft_func(N_a)

            # Stage b: another half step starting from u_n
            b_hat = E2 * u_hat + (dt/2) * phi1_z2 * N_a_hat
            b_field = jnp.real(ifft_func(b_hat))

            # Evaluate N(b)
            N_b = nonlinear_rhs({name: b_field}, t + dt/2)[name]
            N_b_hat = fft_func(N_b)

            # Stage c: half step from a with modified N
            c_hat = E2 * a_hat + (dt/2) * phi1_z2 * (2*N_b_hat - N_n_hat)
            c_field = jnp.real(ifft_func(c_hat))

            # Evaluate N(c)
            N_c = nonlinear_rhs({name: c_field}, t + dt)[name]
            N_c_hat = fft_func(N_c)

            # Final combination
            u_new_hat = E * u_hat + dt * (b1 * N_n_hat + b2 * (N_a_hat + N_b_hat) + b4 * N_c_hat)
            result[name] = jnp.real(ifft_func(u_new_hat))
        else:
            # Fallback to classical RK4 for fields without linear operator
            k1 = N_n[name]
            k2 = nonlinear_rhs({name: u_field + dt/2 * k1}, t + dt/2)[name]
            k3 = nonlinear_rhs({name: u_field + dt/2 * k2}, t + dt/2)[name]
            k4 = nonlinear_rhs({name: u_field + dt * k3}, t + dt)[name]
            result[name] = u_field + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    return result


def etd_integrate(
    u0: StateDict,
    t_span: tuple[float, float],
    dt: float,
    linear_ops: Dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
    method: str = 'etd1',
    save_every: int = 1,
) -> tuple[jnp.ndarray, list[StateDict]]:
    """Integrate u_t = L*u + N(u) using ETD methods.

    Args:
        u0: Initial state (interior values, no ghost cells)
        t_span: (t_start, t_end)
        dt: Time step
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u)
        method: 'etd1', 'etd2', or 'etdrk4'
        save_every: Save solution every N steps

    Returns:
        Tuple of (t_array, state_history)
    """
    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    t_history = [t_start]
    state_history = [u0]

    state = u0
    t = t_start
    N_prev = None

    for step in range(n_steps):
        if method == 'etd1':
            state = etd1_step(state, t, dt, linear_ops, nonlinear_rhs)
        elif method == 'etd2':
            state, N_prev = etd2_step(state, t, dt, linear_ops, nonlinear_rhs, N_prev)
        elif method == 'etdrk4':
            state = etdrk4_step(state, t, dt, linear_ops, nonlinear_rhs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'etd1', 'etd2', or 'etdrk4'")

        t += dt

        if (step + 1) % save_every == 0:
            t_history.append(t)
            state_history.append(state)

    return jnp.array(t_history), state_history


# =============================================================================
# Multi-field Batched FFT Operations
# =============================================================================

def batched_fft_matvec(
    state: StateDict,
    linear_ops: Dict[str, FFTLinearOperator],
) -> StateDict:
    """Apply L*u for all fields using batched FFT when operators are shared.

    Args:
        state: Current state dict
        linear_ops: Dict mapping field name to FFTLinearOperator

    Returns:
        Dict of L*u for each field
    """
    result = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)
        if op is not None:
            result[name] = op.matvec(u_field)
        else:
            result[name] = jnp.zeros_like(u_field)
    return result


def batched_fft_solve(
    rhs: StateDict,
    linear_ops: Dict[str, FFTLinearOperator],
    dt: float,
) -> StateDict:
    """Solve (I - dt*L)*u = rhs for all fields.

    Args:
        rhs: Right-hand side dict
        linear_ops: Dict mapping field name to FFTLinearOperator
        dt: Time step

    Returns:
        Solution dict
    """
    result = {}
    for name, rhs_field in rhs.items():
        op = linear_ops.get(name)
        if op is not None:
            result[name] = op.solve(rhs_field, dt)
        else:
            result[name] = rhs_field
    return result


def batched_fft_exp_matvec(
    state: StateDict,
    linear_ops: Dict[str, FFTLinearOperator],
    dt: float,
) -> StateDict:
    """Apply exp(dt*L)*u for all fields.

    Args:
        state: Current state dict
        linear_ops: Dict mapping field name to FFTLinearOperator
        dt: Time step

    Returns:
        Dict of exp(dt*L)*u for each field
    """
    result = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)
        if op is not None:
            result[name] = op.exp_matvec(u_field, dt)
        else:
            result[name] = u_field
    return result


def stacked_fft_solve_shared_op(
    rhs: StateDict,
    op: FFTLinearOperator,
    dt: float,
) -> StateDict:
    """Solve (I - dt*L)*u = rhs for all fields with SHARED operator.

    More efficient than batched_fft_solve when all fields use the same L.
    Uses a single batched FFT operation.

    Args:
        rhs: Right-hand side dict (all fields same shape)
        op: Single FFTLinearOperator shared by all fields
        dt: Time step

    Returns:
        Solution dict
    """
    field_names = list(rhs.keys())
    if not field_names:
        return {}

    # Stack into single array: shape (n_fields, *spatial_shape)
    rhs_stacked = jnp.stack([rhs[name] for name in field_names], axis=0)

    lam = op.eigenvalues
    denom = 1.0 / (1.0 - dt * lam)

    if rhs_stacked.ndim == 2:  # 1D fields: (n_fields, nx)
        rhs_hat = jnp.fft.fft(rhs_stacked, axis=-1)
        u_hat = denom * rhs_hat
        u_stacked = jnp.real(jnp.fft.ifft(u_hat, axis=-1))
    else:  # 2D fields: (n_fields, ny, nx)
        rhs_hat = jnp.fft.fft2(rhs_stacked, axes=(-2, -1))
        u_hat = denom * rhs_hat
        u_stacked = jnp.real(jnp.fft.ifft2(u_hat, axes=(-2, -1)))

    return {name: u_stacked[i] for i, name in enumerate(field_names)}


def diffusion_only_etd1(
    u0: jnp.ndarray,
    t_end: float,
    dt: float,
    op: FFTLinearOperator,
) -> jnp.ndarray:
    """Simplified ETD1 for pure diffusion (no nonlinear term).

    For u_t = L*u with no N(u), the solution is simply:
        u(t) = exp(t*L)*u0

    This is useful for testing and benchmarking.

    Args:
        u0: Initial condition (interior, no ghost cells)
        t_end: Final time
        dt: Time step (for accuracy, use multiple steps)
        op: FFT linear operator

    Returns:
        Solution at t_end
    """
    n_steps = max(1, int(t_end / dt))
    actual_dt = t_end / n_steps

    u = u0
    for _ in range(n_steps):
        u = op.exp_matvec(u, actual_dt)

    return u


def imex_euler_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: Dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
) -> StateDict:
    """IMEX-Euler step: implicit diffusion, explicit reaction.

    Update: (I - dt*L)*u_{n+1} = u_n + dt*N(u_n)

    This treats L implicitly (via FFT solve) and N explicitly.
    More stable than ETD for some problems, but requires solving linear system.

    Args:
        state: Current state
        t: Current time
        dt: Time step
        linear_ops: FFT operators for implicit solve
        nonlinear_rhs: Explicit nonlinear term

    Returns:
        New state after IMEX-Euler step
    """
    N_state = nonlinear_rhs(state, t)

    result = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)

        # RHS = u_n + dt*N(u_n)
        rhs = u_field + dt * N_state[name]

        if op is not None:
            # Solve (I - dt*L)*u_{n+1} = rhs
            result[name] = op.solve(rhs, dt)
        else:
            # No linear operator: explicit Euler
            result[name] = rhs

    return result
