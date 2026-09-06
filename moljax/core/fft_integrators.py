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

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from moljax.core.fft_operators import FFTLinearOperator
from moljax.core.jit_kernels import phi1, phi2, phi3
from moljax.core.state import StateDict


class ETDStepResult(NamedTuple):
    """Result of an ETD step."""
    state: StateDict
    t: float
    diagnostics: dict


def _etdrk4_coefficients(z: jnp.ndarray) -> tuple:
    """Compute ETDRK4 (Cox-Matthews) coefficients.

    Returns (E, E2, phi1_z2, b1, b2, b4): E = exp(z) and E2 = exp(z/2)
    propagate the linear part over a step and a half step, phi1_z2 =
    φ₁(z/2) weights the nonlinear term in the three internal stages, and
    b1, b2, b4 weight N_n, N_a + N_b and N_c in the final combination
    (Cox-Matthews' b3 equals b2, so the two middle stages share one
    coefficient). The φ functions come from jit_kernels so there is a
    single implementation.

    Reference: Cox & Matthews (2002), "Exponential Time Differencing for
    Stiff Systems", J. Comput. Phys. 176, 430-455.
    """
    E = jnp.exp(z)
    E2 = jnp.exp(z / 2.0)

    phi1_z2 = phi1(z / 2.0)
    phi2_z = phi2(z)
    phi3_z = phi3(z)

    b1 = phi1(z) - 3*phi2_z + 4*phi3_z
    b2 = 2*phi2_z - 4*phi3_z
    b4 = -phi2_z + 4*phi3_z

    return E, E2, phi1_z2, b1, b2, b4


def _fft_pair(op: FFTLinearOperator, u_field: jnp.ndarray) -> tuple[Callable, Callable]:
    """Forward and inverse transforms matching the operator's spectral layout.

    The eigenvalues of an operator built with rfft cover the half spectrum
    (ny, nx//2 + 1), so a field paired with such an operator must go through
    rfft2/irfft2; the full-spectrum and 1D layouts take the real part of the
    complex inverse. Every ETD step needs this choice per field, so it is
    made in one place.
    """
    if u_field.ndim == 1:
        return jnp.fft.fft, lambda x: jnp.real(jnp.fft.ifft(x))
    if getattr(op, '_is_rfft', False):
        ny, nx = u_field.shape
        return jnp.fft.rfft2, lambda x: jnp.fft.irfft2(x, s=(ny, nx))
    return jnp.fft.fft2, lambda x: jnp.real(jnp.fft.ifft2(x))


def etd1_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: dict[str, FFTLinearOperator],
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
            z = dt * op.eigenvalues
            exp_z = jnp.exp(z)
            phi1_z = phi1(z)

            fft_func, ifft_func = _fft_pair(op, u_field)
            u_hat = fft_func(u_field)
            N_hat = fft_func(N_state[name])

            # ETD1 update in Fourier space
            u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
            result[name] = ifft_func(u_new_hat)
        else:
            # No linear operator: explicit Euler for nonlinear part
            result[name] = u_field + dt * N_state[name]

    return result


def etd2_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
    N_prev: StateDict | None = None,
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
            z = dt * op.eigenvalues
            exp_z = jnp.exp(z)
            phi1_z = phi1(z)
            phi2_z = phi2(z)

            fft_func, ifft_func = _fft_pair(op, u_field)
            u_hat = fft_func(u_field)
            N_n_hat = fft_func(N_curr[name])
            N_nm1_hat = fft_func(N_prev[name])

            # ETD2 update
            u_new_hat = (exp_z * u_hat
                        + dt * phi1_z * N_n_hat
                        + dt * phi2_z * (N_n_hat - N_nm1_hat))
            result[name] = ifft_func(u_new_hat)
        else:
            # Explicit Adams-Bashforth 2
            result[name] = u_field + dt * (1.5 * N_curr[name] - 0.5 * N_prev[name])

    return result, N_curr


class _ETDRK4Field(NamedTuple):
    """Per-field spectral data for one ETDRK4 step: transforms, u_hat, and coefficients."""
    fft: Callable
    ifft: Callable
    u_hat: jnp.ndarray
    E: jnp.ndarray
    E2: jnp.ndarray
    phi1_z2: jnp.ndarray
    b1: jnp.ndarray
    b2: jnp.ndarray
    b4: jnp.ndarray


def etdrk4_step(
    state: StateDict,
    t: float,
    dt: float,
    linear_ops: dict[str, FFTLinearOperator],
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

    Each stage is formed for every field before N is evaluated on the full
    stage state: a reaction term such as Gray-Scott's u v^2 couples the
    fields, so N(a) needs a for both u and v. Fields without a linear
    operator take the classical RK4 stages (a = u + dt/2 k1, and so on) in
    the same sweep, so they too see the coupled stage state.

    Args:
        state: Current state
        t: Current time
        dt: Time step
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u)

    Returns:
        New state after one ETDRK4 step
    """
    # Spectral data for the fields that have a linear operator; the others
    # take the RK4 fallback.
    spectral: dict[str, _ETDRK4Field] = {}
    for name, u_field in state.items():
        op = linear_ops.get(name)
        if op is not None:
            fft_func, ifft_func = _fft_pair(op, u_field)
            E, E2, phi1_z2, b1, b2, b4 = _etdrk4_coefficients(dt * op.eigenvalues)
            spectral[name] = _ETDRK4Field(
                fft_func, ifft_func, fft_func(u_field), E, E2, phi1_z2, b1, b2, b4
            )
    explicit = [name for name in state if name not in spectral]
    u_hat = {name: f.u_hat for name, f in spectral.items()}

    def transform(N_state):
        return {name: f.fft(N_state[name]) for name, f in spectral.items()}

    def half_stage(base_hat, N_hat, explicit_stage):
        """exp(z/2) base + (dt/2) phi1(z/2) N for the ETD fields, joined with the RK4 stages."""
        stage_hat = {
            name: f.E2 * base_hat[name] + (dt/2) * f.phi1_z2 * N_hat[name]
            for name, f in spectral.items()
        }
        stage = {
            name: spectral[name].ifft(stage_hat[name]) if name in spectral else explicit_stage[name]
            for name in state
        }
        return stage_hat, stage

    # Stage 1: N at current state
    N_n = nonlinear_rhs(state, t)
    N_n_hat = transform(N_n)

    # Stage a: half step from u_n with N_n (RK4: u + dt/2 k1)
    a_hat, a_state = half_stage(
        u_hat, N_n_hat, {name: state[name] + dt/2 * N_n[name] for name in explicit}
    )
    N_a = nonlinear_rhs(a_state, t + dt/2)
    N_a_hat = transform(N_a)

    # Stage b: another half step from u_n with N_a (RK4: u + dt/2 k2)
    _b_hat, b_state = half_stage(
        u_hat, N_a_hat, {name: state[name] + dt/2 * N_a[name] for name in explicit}
    )
    N_b = nonlinear_rhs(b_state, t + dt/2)
    N_b_hat = transform(N_b)

    # Stage c: half step from a with 2 N_b - N_n (RK4: u + dt k3)
    _c_hat, c_state = half_stage(
        a_hat,
        {name: 2*N_b_hat[name] - N_n_hat[name] for name in spectral},
        {name: state[name] + dt * N_b[name] for name in explicit}
    )
    N_c = nonlinear_rhs(c_state, t + dt)
    N_c_hat = transform(N_c)

    # Final combination
    result = {}
    for name, u_field in state.items():
        if name in spectral:
            f = spectral[name]
            u_new_hat = f.E * f.u_hat + dt * (
                f.b1 * N_n_hat[name] + f.b2 * (N_a_hat[name] + N_b_hat[name]) + f.b4 * N_c_hat[name]
            )
            result[name] = f.ifft(u_new_hat)
        else:
            # Classical RK4 with k1 = N_n, k2 = N_a, k3 = N_b, k4 = N_c
            result[name] = u_field + dt/6 * (N_n[name] + 2*N_a[name] + 2*N_b[name] + N_c[name])

    return result


def etd_integrate(
    u0: StateDict,
    t_span: tuple[float, float],
    dt: float,
    linear_ops: dict[str, FFTLinearOperator],
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

    Notes:
        The time-stepping loop is compiled. Previously this was an eager
        Python ``for`` loop, so every step paid full XLA dispatch and
        long integrations (tens of thousands of steps) took many minutes
        to hours. Stepping now runs inside ``lax.fori_loop`` when only
        the endpoint is retained, and ``lax.scan`` when intermediate
        states are saved.

        When ``save_every >= n_steps`` no intermediate state is
        materialized, which matters for long horizons: collecting every
        step of a 256-point, 130k-step run would allocate hundreds of MB
        that the caller usually discards.
    """
    if method not in ('etd1', 'etd2', 'etdrk4'):
        raise ValueError(f"Unknown method: {method}. Use 'etd1', 'etd2', or 'etdrk4'")

    t_start, t_end = t_span
    n_steps = int((t_end - t_start) / dt)

    if n_steps <= 0:
        return jnp.array([t_start]), [u0]

    # Hoist the method dispatch out of the loop. ETD2 carries the previous
    # nonlinear term; its first step seeds that term from None, so it is
    # taken eagerly and the compiled loop covers the remainder.
    if method == 'etd2':
        seed_state, seed_N = etd2_step(u0, t_start, dt, linear_ops, nonlinear_rhs, None)
        carry = (seed_state, seed_N)
        eager_states = [seed_state]

        def advance(c, t):
            return etd2_step(c[0], t, dt, linear_ops, nonlinear_rhs, c[1])

        def state_of(c):
            return c[0]
    else:
        step_impl = etd1_step if method == 'etd1' else etdrk4_step
        carry = u0
        eager_states = []

        def advance(c, t):
            return step_impl(c, t, dt, linear_ops, nonlinear_rhs)

        def state_of(c):
            return c

    n_done = len(eager_states)
    n_rem = n_steps - n_done

    # Step indices (0-based) whose result is retained, matching the
    # original `(step + 1) % save_every == 0` rule.
    save_steps = [s for s in range(n_steps) if (s + 1) % save_every == 0]

    needs_history = any(s >= n_done and s != n_steps - 1 for s in save_steps)

    if not needs_history:
        # Only the endpoint (at most) is kept: loop without collecting.
        if n_rem > 0:
            def fori_body(i, c):
                return advance(c, t_start + dt * (n_done + i))

            carry = lax.fori_loop(0, n_rem, fori_body, carry)

        def state_after(step_idx):
            if step_idx < n_done:
                return eager_states[step_idx]
            return state_of(carry)
    else:
        ts = t_start + dt * (n_done + jnp.arange(n_rem, dtype=jnp.result_type(float)))

        def scan_body(c, t):
            c_new = advance(c, t)
            return c_new, state_of(c_new)

        carry, stacked = lax.scan(scan_body, carry, ts)

        def state_after(step_idx):
            if step_idx < n_done:
                return eager_states[step_idx]
            return jax.tree.map(lambda a: a[step_idx - n_done], stacked)

    t_history = [t_start]
    state_history = [u0]
    for s in save_steps:
        t_history.append(t_start + (s + 1) * dt)
        state_history.append(state_after(s))

    return jnp.array(t_history), state_history


# =============================================================================
# Multi-field Batched FFT Operations
# =============================================================================

def batched_fft_matvec(
    state: StateDict,
    linear_ops: dict[str, FFTLinearOperator],
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
    linear_ops: dict[str, FFTLinearOperator],
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
    linear_ops: dict[str, FFTLinearOperator],
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

    is_rfft = getattr(op, '_is_rfft', False)
    if rhs_stacked.ndim == 2:  # 1D fields: (n_fields, nx)
        rhs_hat = jnp.fft.fft(rhs_stacked, axis=-1)
        u_hat = denom * rhs_hat
        u_stacked = jnp.real(jnp.fft.ifft(u_hat, axis=-1))
    elif is_rfft:  # 2D fields with rfft: (n_fields, ny, nx)
        spatial_shape = rhs_stacked.shape[-2:]
        rhs_hat = jnp.fft.rfft2(rhs_stacked, axes=(-2, -1))
        u_hat = denom * rhs_hat
        u_stacked = jnp.fft.irfft2(u_hat, s=spatial_shape, axes=(-2, -1))
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
    linear_ops: dict[str, FFTLinearOperator],
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
