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

import math
from collections.abc import Callable
from fractions import Fraction
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from moljax.core.dt_policy import (
    ControllerState,
    PIDParams,
    create_initial_controller_state,
    propose_dt_etd,
)
from moljax.core.fft_operators import FFTLinearOperator
from moljax.core.jit_kernels import phi1, phi2, phi3
from moljax.core.model import CompiledDriverCache, state_cache_key, static_cache_key

# Re-exported so that a caller holding on to ETD drivers can drop them from
# the module they came from; the registry lives with the identity helpers in
# model, and this clears moljax.core.stepping's drivers too.
from moljax.core.model import clear_compiled_drivers as clear_compiled_drivers
from moljax.core.state import StateDict, scaled_error_norm_interior
from moljax.core.stepping import (
    _DEFAULT_MAX_REJECTIONS_PER_STEP,
    _time_dtype,
)
from moljax.core.utils import StatusCode, is_finite


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
    return _etdrk4_step_at(state, t, t + dt/2, t + dt, dt, linear_ops, nonlinear_rhs)


def _etdrk4_step_at(
    state: StateDict,
    t: float,
    t_half: float,
    t_full: float,
    dt: float,
    linear_ops: dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
) -> StateDict:
    """One ETDRK4 step with given stage times (see etdrk4_step).

    t_half and t_full are the times of the stages at t + dt/2 and t + dt.
    etd_integrate_adaptive gives each of them as one rounding of the
    exact value. The expressions t + dt/2 and t + dt round twice when t
    is itself a rounded stage time.
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
    N_a = nonlinear_rhs(a_state, t_half)
    N_a_hat = transform(N_a)

    # Stage b: another half step from u_n with N_a (RK4: u + dt/2 k2)
    _b_hat, b_state = half_stage(
        u_hat, N_a_hat, {name: state[name] + dt/2 * N_a[name] for name in explicit}
    )
    N_b = nonlinear_rhs(b_state, t_half)
    N_b_hat = transform(N_b)

    # Stage c: half step from a with 2 N_b - N_n (RK4: u + dt k3)
    _c_hat, c_state = half_stage(
        a_hat,
        {name: 2*N_b_hat[name] - N_n_hat[name] for name in spectral},
        {name: state[name] + dt * N_b[name] for name in explicit}
    )
    N_c = nonlinear_rhs(c_state, t_full)
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


def _etd_step_count(t0: float, t_end: float, dt: float) -> int:
    """
    Number of ETD steps of size dt that cover [t0, t_end] exactly.

    Same schedule as moljax.core.stepping._fixed_step_count: rounding
    absorbs the floating point error of the division ((0.3 - 0.0) / 0.1 is
    2.9999999999999996) and the check below rejects a genuine mismatch,
    rather than truncating int((t_end - t0) / dt), which silently falls
    short of the endpoint (the same span and dt gave 2 steps instead of 3).
    """
    span = t_end - t0
    if dt <= 0.0 or span <= 0.0:
        raise ValueError(f"need dt > 0 and t_end > t0, got dt={dt}, t0={t0}, t_end={t_end}")
    n_steps = int(round(span / dt))
    if n_steps < 1 or abs(n_steps * dt - span) > 1e-9 * span:
        raise ValueError(
            f"dt={dt} does not divide the interval [{t0}, {t_end}] into whole steps "
            f"({span / dt:.12g} steps); pick dt = (t_end - t0)/n for an integer n"
        )
    return n_steps


class _ETDSchedule(NamedTuple):
    """
    How etd_integrate splits n_steps into an eager seed and compiled blocks.

    Derived from the method, the step count and save_every alone, so it is
    the same for every call that shares a compiled driver, and both the
    driver (which needs the block sizes) and the caller (which needs to
    label the snapshots) can work it out for themselves.

    n_done is the steps taken before the blocks begin: one for ETD2, whose
    first step seeds the previous nonlinear term from None and is therefore
    a Python-level branch, and none for ETD1 and ETDRK4. lead is the extra
    steps needed to reach the first save boundary, n_saves the number of
    save_every-sized blocks the outer scan runs, and n_tail the steps left
    over after them.
    """
    n_done: int
    lead: int
    n_done_aligned: int
    record_lead: bool
    n_saves: int
    n_tail: int


def _etd_schedule(method: str, n_steps: int, save_every: int) -> _ETDSchedule:
    """
    Work out etd_integrate's block structure.

    ETD2's eager seed step (n_done == 1) lands on absolute step 1, which is
    a save boundary only when save_every == 1; for any larger save_every it
    sits mid-block. `lead` is the number of extra steps needed to reach the
    next boundary at a multiple of save_every (0 for ETD1/ETDRK4, whose
    n_done == 0 is already a boundary). Running it as its own short block
    first re-anchors every subsequent save to the same absolute-step grid
    ETD1/ETDRK4 use, instead of offsetting all of them by the seed step
    (the bug this fixes: save_every == 2 saved at steps 3, 5, ... instead
    of 2, 4, ...).

    record_lead is true only when the lead block reaches a genuine
    intermediate save point; if it consumes every remaining step,
    n_done_aligned == n_steps and that state is the final state, recorded
    by etd_integrate's closing block instead (this is also why a plain
    save_every == 1 seed step, lead == 0, is still recorded: n_done_aligned
    == n_done == 1 is itself already the first save boundary).
    """
    n_done = 1 if method == 'etd2' else 0
    n_rem = n_steps - n_done
    lead = min((save_every - n_done % save_every) % save_every, n_rem)
    n_done_aligned = n_done + lead
    record_lead = 0 < n_done_aligned < n_steps
    n_rem -= lead
    n_saves = n_rem // save_every
    n_tail = n_rem - n_saves * save_every
    return _ETDSchedule(n_done, lead, n_done_aligned, record_lead, n_saves, n_tail)


# The compiled ETD loops, one per set of static parameters. The loop used to
# be built as a fresh closure on every call, so it was a new Python object
# each time and JAX's compilation cache could not recognize it: three
# identical etd_integrate calls traced and compiled the same scan three
# times. Keeping the closure alive across calls is what lets that cache hit.
_ETD_DRIVERS = CompiledDriverCache("fft_integrators.etd_integrate")


def _build_etd_driver(dt, n_steps, save_every, linear_ops, nonlinear_rhs, method):
    """
    Compile etd_integrate's stepping loop for one set of static parameters.

    dt, the step count, save_every and the method are closure constants:
    they are what the loop is built out of, and dt fixes the step count in
    the first place. The state and the start time are arguments, so a run
    that only differs in its initial condition, or one that continues from
    where the last one stopped, reuses the same executable.

    Returns (lead_state, stacked, final_state), with lead_state None unless
    the lead block lands on an intermediate save point and stacked None when
    there are no full blocks to scan over.
    """
    sched = _etd_schedule(method, n_steps, save_every)

    def run(u0, t_start):
        # Hoist the method dispatch out of the loop. ETD2 carries the previous
        # nonlinear term; its first step seeds that term from None, which is a
        # Python-level branch inside etd2_step, so it is taken before the
        # compiled loop rather than inside it. The loop itself only ever sees a
        # concrete N_prev array threaded through the carry, both across
        # fori_loop steps and across scan iterations, so it is never re-seeded
        # at a block boundary.
        if method == 'etd2':
            seed_state, seed_N = etd2_step(u0, t_start, dt, linear_ops, nonlinear_rhs, None)
            carry = (seed_state, seed_N)

            def advance(c, t):
                return etd2_step(c[0], t, dt, linear_ops, nonlinear_rhs, c[1])

            def state_of(c):
                return c[0]
        else:
            step_impl = etd1_step if method == 'etd1' else etdrk4_step
            carry = u0

            def advance(c, t):
                return step_impl(c, t, dt, linear_ops, nonlinear_rhs)

            def state_of(c):
                return c

        def run_block(c, step_offset, count):
            """Advance `c` by `count` steps. The i-th (0-based) step lands on
            absolute step n_done + step_offset + i, so the time passed to
            `advance` matches the pre-rewrite single scan's per-step time
            exactly (t_start + dt * absolute_step_index)."""

            def body(i, c):
                t = t_start + dt * (sched.n_done + step_offset + i)
                return advance(c, t)

            return lax.fori_loop(0, count, body, c)

        if sched.lead > 0:
            carry = run_block(carry, 0, sched.lead)
        lead_state = state_of(carry) if sched.record_lead else None

        if sched.n_saves > 0:
            def save_body(c, block_index):
                c = run_block(c, sched.lead + block_index * save_every, save_every)
                return c, state_of(c)

            carry, stacked = lax.scan(save_body, carry, jnp.arange(sched.n_saves))
        else:
            stacked = None

        if sched.n_tail > 0:
            carry = run_block(carry, sched.lead + sched.n_saves * save_every, sched.n_tail)

        return lead_state, stacked, state_of(carry)

    return jax.jit(run)


def _etd_driver(u0, dt, n_steps, save_every, linear_ops, nonlinear_rhs, method):
    """
    Look up, or build, the compiled ETD loop for these parameters.

    The key holds everything that determines the trace and nothing that does
    not: the state's structure and avals but not its values, dt with its
    Python type (a np.float32 dt is strongly typed and promotes differently
    from a Python float of the same value), the step schedule, the linear
    operators and the nonlinear right-hand side by identity, and the x64
    flag, which decides what dtype the times are carried in. See
    moljax.core.model.static_cache_key for why keying by identity is safe
    while the entry is alive.
    """
    key = (
        'etd_integrate',
        state_cache_key(u0),
        static_cache_key(dt),
        n_steps,
        save_every,
        static_cache_key(linear_ops),
        static_cache_key(nonlinear_rhs),
        method,
        bool(jax.config.jax_enable_x64),
    )
    return _ETD_DRIVERS.get_or_build(key, lambda: _build_etd_driver(
        dt, n_steps, save_every, linear_ops, nonlinear_rhs, method
    ))


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
        t_span: (t_start, t_end). dt must divide the span into a whole
            number of steps (see _etd_step_count); a non-divisible span
            raises ValueError rather than silently falling short of t_end.
        dt: Time step
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u)
        method: 'etd1', 'etd2', or 'etdrk4'
        save_every: Save solution every N steps

    Returns:
        Tuple of (t_array, state_history). The last entry of state_history
        is always the state at t_end, independent of save_every.

    Notes:
        The compiled loop is cached and reused. A call whose method, dt,
        step count, save_every, linear operators and nonlinear right-hand
        side match an earlier one's, with a state of the same structure,
        shapes and dtypes, runs the executable that call built rather than
        tracing a new one; the state and ``t_span[0]`` are arguments of
        that executable, so continuing a run from where the last one
        stopped compiles nothing. The cache holds 32 entries, each pinning
        what its loop closes over; ``clear_compiled_drivers`` empties it.

        The time-stepping loop is compiled as an outer ``lax.scan`` over
        saved snapshots, each covering ``save_every`` steps taken by an
        inner ``lax.fori_loop``, with any leftover steps (when
        ``save_every`` does not divide the step count evenly) run in a
        final ``lax.fori_loop`` after the scan. This is the same shape
        moljax.core.stepping's ``integrate_fixed_dt`` uses (see
        ``_run_fixed_steps``).

        Previously the compiled loop was a single ``lax.scan`` over every
        step whenever intermediate history was requested, so the stacked
        output held every step's full state before ``save_every`` was
        applied: memory scaled with the total step count rather than the
        number of saved snapshots. 1000 steps of two 8x8 float64 fields
        with save_every=500 allocated 1,024,000 bytes internally for 3,072
        bytes returned, and a 1e5-step, two-256x256-field run would have
        needed about 98 GiB. The outer scan now stacks one snapshot per
        ``save_every`` steps, so memory scales with the number of
        snapshots kept, not the number of steps taken.
    """
    if method not in ('etd1', 'etd2', 'etdrk4'):
        raise ValueError(f"Unknown method: {method}. Use 'etd1', 'etd2', or 'etdrk4'")
    if save_every < 1:
        raise ValueError(f"save_every must be at least 1, got {save_every}")

    t_start, t_end = t_span
    n_steps = _etd_step_count(t_start, t_end, dt)
    sched = _etd_schedule(method, n_steps, save_every)

    driver = _etd_driver(u0, dt, n_steps, save_every, linear_ops, nonlinear_rhs, method)
    lead_state, stacked, final_state = driver(u0, t_start)

    t_history = [t_start]
    state_history = [u0]

    if sched.record_lead:
        t_history.append(t_start + dt * sched.n_done_aligned)
        state_history.append(lead_state)

    for block_index in range(sched.n_saves):
        step_count = sched.n_done_aligned + (block_index + 1) * save_every
        t_history.append(t_start + dt * step_count)
        state_history.append(jax.tree.map(lambda a, i=block_index: a[i], stacked))

    # The last entry is always the true final state at t_end: either the
    # last block above already reached it (n_tail == 0), or it is appended
    # here regardless of where the last save_every boundary fell. Either
    # way it represents exactly n_steps steps, so its label is the caller's
    # own t_end rather than a recomputed t_start + n_steps * dt, which can
    # differ from t_end at the last bit or two of precision.
    if sched.n_saves > 0 and sched.n_tail == 0:
        t_history[-1] = t_end
    else:
        t_history.append(t_end)
        state_history.append(final_state)

    return jnp.array(t_history), state_history


# =============================================================================
# Adaptive ETD Integration
# =============================================================================

# Order p of each method that etd_integrate_adaptive can step. ETD2 is not
# here: it is a multistep method, and a variable step needs a changed
# formula and its own history for the full step and the half steps.
_ETD_ADAPTIVE_ORDER = {'etd1': 1, 'etdrk4': 4}

# The controller gets the order p + _ETD_CONTROLLER_ORDER_OFFSET. The
# step-doubling estimate is O(dt^(p+1)), so the offset is 1. With the
# offset 0 (the order p), the controller exponents are too large for ETD1:
# on the stiff test problem of
# tests/test_etd_adaptive.py::TestControllerOrder, about half of all
# attempts are rejections. The value is read when a loop is compiled; that
# test changes it to show the difference.
_ETD_CONTROLLER_ORDER_OFFSET = 1

# The compiled adaptive ETD loops, kept apart from _ETD_DRIVERS so that a
# sweep of one family cannot remove the drivers of the other family.
_ETD_ADAPTIVE_DRIVERS = CompiledDriverCache("fft_integrators.etd_integrate_adaptive")


class ETDAdaptiveResult(NamedTuple):
    """
    Result of etd_integrate_adaptive.

    Attributes:
        t_final: Time at the end of the run (equal to t_end on success)
        y_final: State at t_final (interior values only)
        t_history: Times of the saved states, shape (max_saves,), with
            max_saves = max_steps // save_every + 2. Only the first
            n_saved entries are valid; the remaining entries are zero.
        y_history: Saved states, one array of shape (max_saves, *shape)
            for each field. Only the first n_saved entries are valid.
        dt_history: Size of the accepted step that ends at each saved
            time. Entry 0 (the initial state) is zero.
        n_saved: Number of valid history entries, the initial state included
        n_accepted: Number of accepted steps
        n_rejected: Number of rejected steps
        dt_final: The step size that the controller proposes for a next step
        status: A moljax.core.utils.StatusCode value
        success: True if the run reached t_end (status SUCCESS)
    """
    t_final: jnp.ndarray
    y_final: StateDict
    t_history: jnp.ndarray
    y_history: dict[str, jnp.ndarray]
    dt_history: jnp.ndarray
    n_saved: jnp.ndarray
    n_accepted: jnp.ndarray
    n_rejected: jnp.ndarray
    dt_final: jnp.ndarray
    status: jnp.ndarray
    success: jnp.ndarray


class _ETDAdaptiveState(NamedTuple):
    """
    Carry of the lax.while_loop of etd_integrate_adaptive.

    Declared at module level for the same reason as
    moljax.core.stepping._FixedScanState: a NamedTuple that is built for
    each call is a different pytree node type each time.

    k is the elapsed time as an integer number of ticks (see
    _etd_tick_plan), not a floating-point time.
    """
    k: jnp.ndarray
    y: StateDict
    dt: jnp.ndarray
    controller: ControllerState
    t_history: jnp.ndarray
    y_history: dict[str, jnp.ndarray]
    dt_history: jnp.ndarray
    write_idx: jnp.ndarray
    n_accepted: jnp.ndarray
    n_rejected: jnp.ndarray
    status: jnp.ndarray


class _TickPlan(NamedTuple):
    """
    The tick grid of one run of etd_integrate_adaptive (see _etd_tick_plan).

    All entries are arguments of the compiled loop, so a new t_span does
    not compile a new loop. Positions are absolute times for a
    nonlinear_rhs that reads t, and elapsed times t - t0 otherwise.
    """
    t0: jnp.ndarray            # time dtype
    t_end: jnp.ndarray         # time dtype
    lo: jnp.ndarray            # position of tick 0
    hi: jnp.ndarray            # position of tick n_ticks
    unit: jnp.ndarray          # the tick length G, a power of two
    base: jnp.ndarray          # tick k (0 < k < n_ticks) is at (base + k) * unit
    n_ticks: jnp.ndarray
    r_lo: jnp.ndarray          # the partial tick at tick 0 (zero if lo is on the grid)
    r_hi: jnp.ndarray          # the partial tick at tick n_ticks (zero if hi is on the grid)
    lo_partial: jnp.ndarray
    hi_partial: jnp.ndarray
    n_min: jnp.ndarray         # the smallest step, in ticks
    n_min_first: jnp.ndarray   # the smallest step from tick 0
    n_min_last: jnp.ndarray    # the smallest step to tick n_ticks
    k_max: jnp.ndarray         # the largest step, in ticks
    k_max_first: jnp.ndarray   # the largest step from tick 0
    k_max_last: jnp.ndarray    # the largest step to tick n_ticks


def _build_etd_adaptive_driver(linear_ops, nonlinear_rhs, method, pid_params, max_steps,
                               save_every, max_rejections_per_step, rhs_uses_time):
    """
    Compile the while loop of etd_integrate_adaptive for one set of static parameters.

    The method, the controller parameters, max_steps, save_every and the
    rejection budget are closure constants. max_steps and save_every set
    the size of the history buffers. The initial state, the tick plan and
    dt0 are arguments, so a sweep over dt0 or t_end compiles one loop.
    """
    order = _ETD_ADAPTIVE_ORDER[method]
    # Richardson: the error of the two half steps is
    # (y_half2 - y_full) / (2^p - 1) to leading order.
    richardson = 1.0 / (2.0 ** order - 1.0)
    # The step-doubling estimate is O(dt^(p+1)), so the controller gets
    # the order p + 1 (see _ETD_CONTROLLER_ORDER_OFFSET).
    controller_order = order + _ETD_CONTROLLER_ORDER_OFFSET
    max_saves = max_steps // save_every + 2

    def run(u0, plan: _TickPlan, dt0):
        time_dtype = _time_dtype()
        tick_dtype = plan.n_ticks.dtype
        names = list(u0)
        field_dtypes = {name: jnp.result_type(u0[name]) for name in names}
        # The dtype of the step arithmetic. dt is cast to it, so that a
        # float32 state is not widened by a float64 dt.
        compute_dtype = jnp.result_type(*[u0[name] for name in names])

        def to_field_dtypes(state):
            return {name: state[name].astype(field_dtypes[name]) for name in names}

        def position(k):
            """The exact position of tick k (a float: see _etd_tick_plan)."""
            interior = (plan.base + k).astype(time_dtype) * plan.unit
            return jnp.where(k == 0, plan.lo, jnp.where(k == plan.n_ticks, plan.hi, interior))

        def absolute(x):
            """The time that nonlinear_rhs and t_history get for the position x."""
            return x if rhs_uses_time else plan.t0 + x

        def history_time(k):
            return jnp.where(k == plan.n_ticks, plan.t_end, absolute(position(k)))

        def step_length(k, n):
            """The exact length of the step from tick k over n ticks."""
            partial_lo = (k == 0) & plan.lo_partial
            partial_hi = (k + n == plan.n_ticks) & plan.hi_partial
            whole = n - partial_lo.astype(tick_dtype) - partial_hi.astype(tick_dtype)
            return (whole.astype(time_dtype) * plan.unit
                    + jnp.where(partial_lo, plan.r_lo, 0.0).astype(time_dtype)
                    + jnp.where(partial_hi, plan.r_hi, 0.0).astype(time_dtype))

        def stage_times(k, n, h):
            """The stage times of one doubled step, each one rounding of its exact value.

            The start and the end are exact tick positions. The other
            times add h/4 or h/2 (exact: h is a float and the factor is a
            power of two) to the start, or subtract h/4 from the end, so
            each is one rounding of its exact value. Nested expressions
            such as (t + h/2) + h/4 round twice.
            """
            a = position(k)
            b = position(k + n)
            mid = a + h / 2
            if method == 'etd1':
                return absolute(a), absolute(mid), absolute(b), None, None
            return absolute(a), absolute(mid), absolute(b), absolute(a + h / 4), absolute(b - h / 4)

        def doubled_step(y, k, n, h):
            """One full step and two half steps. Returns (y_full, y_half2)."""
            hc = h.astype(compute_dtype)
            t_a, t_mid, t_b, t_q1, t_q3 = stage_times(k, n, h)
            if method == 'etd1':
                y_full = etd1_step(y, t_a, hc, linear_ops, nonlinear_rhs)
                y_half1 = etd1_step(y, t_a, hc / 2, linear_ops, nonlinear_rhs)
                y_half2 = etd1_step(to_field_dtypes(y_half1), t_mid, hc / 2, linear_ops,
                                    nonlinear_rhs)
            else:
                y_full = _etdrk4_step_at(y, t_a, t_mid, t_b, hc, linear_ops, nonlinear_rhs)
                y_half1 = _etdrk4_step_at(y, t_a, t_q1, t_mid, hc / 2, linear_ops,
                                          nonlinear_rhs)
                y_half2 = _etdrk4_step_at(to_field_dtypes(y_half1), t_mid, t_q3, t_b, hc / 2,
                                          linear_ops, nonlinear_rhs)
            return to_field_dtypes(y_full), to_field_dtypes(y_half2)

        def choose_ticks(k, dt):
            """Return (n, admissible): the ticks of the next trial for the proposal dt.

            The rule keeps the remaining count R at zero or at least
            n_min_last, so that the end stays reachable (see
            moljax_question/etd_global_ticks.md, section 4):
            1. n* = floor(dt / unit), capped at k_max.
            2. The whole remainder, if it fits and exceeds n* by at most n*/8
               (a decimal dt then gives no very short final step).
            3. If the remainder after n* is in (0, n_min_last), leave
               exactly n_min_last ticks, or take the whole remainder if that
               leaves less than one step.
            The trial is admissible when n* is at least the smallest step.
            """
            remaining = plan.n_ticks - k
            first = k == 0
            cap = jnp.where(first, plan.k_max_first, plan.k_max)
            cap_end = jnp.minimum(cap, plan.k_max_last)
            n_lo = jnp.where(first, plan.n_min_first, plan.n_min)
            dt_cap = jnp.minimum(dt, cap.astype(time_dtype) * plan.unit)
            n_star = jnp.floor(dt_cap / plan.unit).astype(tick_dtype)
            n = jnp.minimum(n_star, cap)
            take_all = (remaining <= cap_end) & ((n >= remaining) | (8 * (remaining - n) <= n))
            short_tail = remaining - n < plan.n_min_last
            n_split = remaining - plan.n_min_last
            n = jnp.where(take_all, remaining,
                          jnp.where(short_tail, jnp.where(n_split >= n_lo, n_split, remaining), n))
            return n, n_star >= n_lo

        dt_min = jnp.asarray(pid_params.dt_min, dtype=time_dtype)
        dt_init = jnp.clip(dt0, pid_params.dt_min, pid_params.dt_max).astype(time_dtype)

        t_history = jnp.zeros(max_saves, dtype=time_dtype).at[0].set(plan.t0)
        y_history = {
            name: jnp.zeros((max_saves,) + jnp.shape(u0[name]), dtype=field_dtypes[name])
            .at[0].set(u0[name])
            for name in names
        }
        dt_history = jnp.zeros(max_saves, dtype=time_dtype)

        init = _ETDAdaptiveState(
            k=jnp.zeros((), dtype=tick_dtype),
            y=to_field_dtypes(u0),
            dt=dt_init,
            controller=create_initial_controller_state(time_dtype),
            t_history=t_history,
            y_history=y_history,
            dt_history=dt_history,
            write_idx=jnp.array(1, dtype=jnp.int32),
            n_accepted=jnp.array(0, dtype=jnp.int32),
            n_rejected=jnp.array(0, dtype=jnp.int32),
            status=jnp.array(StatusCode.RUNNING, dtype=jnp.int32),
        )

        def cond_fn(s: _ETDAdaptiveState) -> jnp.ndarray:
            running = s.status == StatusCode.RUNNING
            return running & (s.k < plan.n_ticks) & (s.n_accepted < max_steps)

        def body_fn(s: _ETDAdaptiveState) -> _ETDAdaptiveState:
            n, admissible = choose_ticks(s.k, s.dt)
            # A proposal below the smallest step stops the run before the
            # step. For a nonlinear_rhs that reads t, the smallest step is
            # set by the clock resolution (TIME_UNRESOLVED).
            too_small = StatusCode.TIME_UNRESOLVED if rhs_uses_time else StatusCode.DT_TOO_SMALL
            return lax.cond(
                admissible,
                lambda: _step_body(s, n),
                lambda: s._replace(status=jnp.array(too_small, dtype=jnp.int32)),
            )

        def _step_body(s: _ETDAdaptiveState, n) -> _ETDAdaptiveState:
            h = step_length(s.k, n)
            y_full, y_half2 = doubled_step(s.y, s.k, n, h)
            err = {name: (y_half2[name] - y_full[name]) * richardson for name in names}

            finite = is_finite(y_full) & is_finite(y_half2)
            err_ratio = scaled_error_norm_interior(err, y_half2, pid_params.atol, pid_params.rtol)
            err_ratio = err_ratio.astype(time_dtype)
            # A non-finite error ratio makes the controller give a NaN dt.
            # A step with non-finite values, or with a non-finite ratio
            # (a nonzero error where atol + rtol*|y| is zero), gets a large
            # finite ratio instead, so the controller cuts dt by min_factor.
            finite = finite & jnp.isfinite(err_ratio)
            err_ratio = jnp.where(finite, jnp.maximum(err_ratio, 1e-10), 1e10)
            accept = finite & (err_ratio <= 1.0)

            dt_new, controller = propose_dt_etd(
                h, err_ratio, s.controller, pid_params, controller_order, accepted=accept
            )

            def accept_step():
                n_accepted = s.n_accepted + 1
                # The clock is the integer tick count, so the accepted
                # lengths sum exactly to t_end - t0 and the last tick is
                # t_end (no remainder, no margin).
                k_new = s.k + n
                done = k_new == plan.n_ticks
                save = ((n_accepted % save_every) == 0) | done
                idx = s.write_idx

                def write(buffer, value):
                    # Select the value, not the buffer, so that the update
                    # stays in place and costs one entry, not the buffer.
                    return buffer.at[idx].set(jnp.where(save, value, buffer[idx]))

                t_hist = write(s.t_history, history_time(k_new))
                dt_hist = write(s.dt_history, h)
                y_hist = {name: write(s.y_history[name], y_half2[name]) for name in names}
                return _ETDAdaptiveState(
                    k=k_new, y=y_half2, dt=dt_new, controller=controller,
                    t_history=t_hist, y_history=y_hist, dt_history=dt_hist,
                    write_idx=idx + save.astype(jnp.int32),
                    n_accepted=n_accepted, n_rejected=s.n_rejected, status=s.status,
                )

            def reject_step():
                # Halve the length of the rejected trial, not an older and
                # larger proposal: each rejection then halves the ticks of
                # the next trial (the same extra cut as the adaptive drivers
                # of moljax.core.stepping).
                dt_reject = jnp.maximum(jnp.minimum(dt_new, h) * 0.5,
                                        pid_params.dt_min).astype(time_dtype)
                # Stop only when the rejected trial itself was at dt_min (or
                # smaller, for a short final step). A proposal that reaches
                # dt_min gets one trial at dt_min first.
                status = jnp.where(
                    h <= dt_min,
                    jnp.array(StatusCode.DT_TOO_SMALL, dtype=jnp.int32),
                    jnp.where(
                        controller.consecutive_rejects >= max_rejections_per_step,
                        jnp.array(StatusCode.MAX_ATTEMPTS_REACHED, dtype=jnp.int32),
                        s.status,
                    ),
                )
                return s._replace(dt=dt_reject, controller=controller,
                                  n_rejected=s.n_rejected + 1, status=status)

            return lax.cond(accept, accept_step, reject_step)

        final = lax.while_loop(cond_fn, body_fn, init)

        reached = final.k == plan.n_ticks
        status = jnp.where(
            reached,
            jnp.array(StatusCode.SUCCESS, dtype=jnp.int32),
            jnp.where(
                final.status == StatusCode.RUNNING,
                jnp.array(StatusCode.MAX_STEPS_REACHED, dtype=jnp.int32),
                final.status,
            ),
        )
        return ETDAdaptiveResult(
            t_final=history_time(final.k),
            y_final=final.y,
            t_history=final.t_history,
            y_history=final.y_history,
            dt_history=final.dt_history,
            n_saved=final.write_idx,
            n_accepted=final.n_accepted,
            n_rejected=final.n_rejected,
            dt_final=final.dt,
            status=status,
            success=status == StatusCode.SUCCESS,
        )

    return jax.jit(run)


def _etd_adaptive_driver(u0, linear_ops, nonlinear_rhs, method, pid_params, max_steps,
                         save_every, max_rejections_per_step, rhs_uses_time):
    """
    Look up, or build, the compiled adaptive ETD loop for these parameters.

    The key holds the structure, shapes and dtypes of the state, the
    linear operators and the nonlinear right-hand side by identity, the
    method, the controller parameters, max_steps, save_every, the
    rejection budget, rhs_uses_time and the x64 flag. The key does not hold dt0 or t_end:
    they are arguments of the compiled loop.
    """
    key = (
        'etd_integrate_adaptive',
        state_cache_key(u0),
        static_cache_key(linear_ops),
        static_cache_key(nonlinear_rhs),
        method,
        static_cache_key(pid_params),
        max_steps,
        save_every,
        max_rejections_per_step,
        rhs_uses_time,
        bool(jax.config.jax_enable_x64),
    )
    return _ETD_ADAPTIVE_DRIVERS.get_or_build(key, lambda: _build_etd_adaptive_driver(
        linear_ops, nonlinear_rhs, method, pid_params, max_steps, save_every,
        max_rejections_per_step, rhs_uses_time
    ))


def _floor_log2(x: Fraction) -> int:
    """Return floor(log2(x)) for a positive rational x."""
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if Fraction(2) ** e > x:
        e -= 1
    return e


def _dyadic_valuation(x: Fraction) -> int:
    """Return v with x = odd * 2^v, for a nonzero dyadic rational x."""
    num = abs(x.numerator)
    return (num & -num).bit_length() - x.denominator.bit_length()


def _etd_tick_plan(t0, t_end, dt0, method, rhs_uses_time, compute_dtype,
                   pid_params) -> _TickPlan:
    """
    Build the tick grid of etd_integrate_adaptive on the host, with exact rationals.

    The run covers the cast endpoints fl(t0) and fl(t_end) of the time
    dtype (p bits). The grid unit is the power of two
    G = 2^(floor(log2(top)) + 1 - p), where top is the largest position:
    max(|t0|, |t_end|) for a nonlinear_rhs that reads t (positions are
    absolute times), and t_end - t0 otherwise (positions are elapsed
    times). Every multiple of G up to top is a float, so every tick
    position is exact. An endpoint that is not a multiple of G gives a
    partial tick at that end; only the step that touches that end
    includes it.

    A step of n ticks is resolved when n >= m (m = 2 for ETD1 and 4 for
    ETDRK4; m + 1 for a step with a partial tick): its stage times are
    then distinct, because each is one rounding of an exact value and
    adjacent exact values differ by more than G (or are exact). For an
    elapsed grid there is no resolution limit: the smallest step is one
    tick (two with a partial tick, so that h/4 stays normal).

    With k_max >= n_min_first + n_min_last - 1, every remaining count of
    at least n_min_last ticks is reachable, so the span is feasible
    exactly when it holds at least max(n_min_first, n_min_last) ticks.
    The derivation and the checks are in moljax_question/etd_global_ticks.md
    of the research notes.

    Raises:
        ValueError: if the grid unit is below four times the smallest
            normal number of the time or state dtype, if dt_max is below
            n_min_first + n_min_last - 1 ticks, if the span holds fewer
            ticks than the smallest step, or if a concrete dt0 (clamped to
            [dt_min, dt_max]) is below the smallest first step.
    """
    clock = np.float64 if _time_dtype() == jnp.float64 else np.float32
    p = np.finfo(clock).nmant + 1
    lo_c, hi_c = clock(float(t0)), clock(float(t_end))
    start, end = Fraction(float(lo_c)), Fraction(float(hi_c))
    if rhs_uses_time:
        lo, hi = start, end
        top = max(abs(start), abs(end))
        m = 2 if method == 'etd1' else 4
        n_min, n_min_partial = m, m + 1
    else:
        lo, hi = Fraction(0), end - start
        top = hi
        n_min, n_min_partial = 1, 2
    unit = Fraction(2) ** (_floor_log2(top) + 1 - p)
    tiny = max(float(np.finfo(clock).tiny), float(np.finfo(compute_dtype).tiny))
    if unit < 4 * Fraction(tiny):
        raise ValueError(
            f"etd_integrate_adaptive: the clock tick {float(unit)!r} for t_span "
            f"[{t0}, {t_end}] is below four times the smallest normal number; the "
            f"CPU can flush the stage times to zero. Shift the time origin."
        )

    first = math.ceil(lo / unit)
    last = math.floor(hi / unit)
    lo_partial = first * unit != lo
    hi_partial = last * unit != hi
    base = first - int(lo_partial)
    n_ticks = last - first + int(lo_partial) + int(hi_partial)
    r_lo = first * unit - lo if lo_partial else Fraction(0)
    r_hi = hi - last * unit if hi_partial else Fraction(0)
    n_min_first = n_min_partial if lo_partial else n_min
    n_min_last = n_min_partial if hi_partial else n_min
    need = n_min_first + n_min_last - 1

    dt_max = float(pid_params.dt_max)
    k_max = 2 ** p if not math.isfinite(dt_max) else min(math.floor(Fraction(dt_max) / unit), 2 ** p)

    def partial_cap(r):
        """The largest step count whose length (n - 1) G + r is a float, or k_max."""
        if Fraction(float(clock(float(r)))) != r:
            return k_max  # r itself is not a float: the length has one rounding
        cap = 1 + math.floor((Fraction(2) ** (p + _dyadic_valuation(r)) - r) / unit)
        return min(k_max, cap) if cap >= need else k_max

    k_max_first = partial_cap(r_lo) if lo_partial else k_max
    k_max_last = partial_cap(r_hi) if hi_partial else k_max

    if min(k_max, k_max_first, k_max_last) < need:
        raise ValueError(
            f"etd_integrate_adaptive: dt_max={pid_params.dt_max} is below {need} clock "
            f"ticks of {float(unit)!r} at t_span [{t0}, {t_end}]; no step schedule can "
            f"keep t_end reachable. Increase dt_max, enable x64, or shift the time origin."
        )
    if n_ticks < max(n_min_first, n_min_last):
        raise ValueError(
            f"etd_integrate_adaptive: the span [{t0}, {t_end}] holds {n_ticks} clock ticks "
            f"of {float(unit)!r}; one {method} step needs at least "
            f"{max(n_min_first, n_min_last)} ticks so that its stage times are distinct. "
            f"Enable x64, or shift the time origin."
        )
    if not isinstance(dt0, jax.core.Tracer):
        dt_first = min(max(float(dt0), float(pid_params.dt_min)), dt_max)
        if math.floor(Fraction(dt_first) / unit) < n_min_first:
            raise ValueError(
                f"etd_integrate_adaptive: dt0={dt0} (clamped to [dt_min, dt_max]) is below "
                f"{n_min_first} clock ticks of {float(unit)!r} at t0={t0}, the smallest "
                f"first step. Increase dt0, enable x64, or shift the time origin."
            )

    time_dtype = _time_dtype()
    tick_dtype = jnp.int64 if time_dtype == jnp.float64 else jnp.int32

    def time(x):
        return jnp.asarray(clock(float(x)), dtype=time_dtype)

    def ticks(x):
        return jnp.asarray(x, dtype=tick_dtype)

    return _TickPlan(
        t0=jnp.asarray(lo_c, dtype=time_dtype), t_end=jnp.asarray(hi_c, dtype=time_dtype),
        lo=time(lo), hi=time(hi), unit=time(unit),
        base=ticks(base), n_ticks=ticks(n_ticks), r_lo=time(r_lo), r_hi=time(r_hi),
        lo_partial=jnp.asarray(lo_partial), hi_partial=jnp.asarray(hi_partial),
        n_min=ticks(n_min), n_min_first=ticks(n_min_first), n_min_last=ticks(n_min_last),
        k_max=ticks(k_max), k_max_first=ticks(k_max_first), k_max_last=ticks(k_max_last),
    )


# The largest relative change of the span t_end - t0 that the cast of t0
# and t_end to the time dtype can cause before etd_integrate_adaptive
# raises. The cast of each endpoint rounds it to the clock resolution at
# that time. When the resolution is not small against the span, the run
# integrates over a different duration (in float32, (1000000.02,
# 1000000.04) becomes (1000000.0, 1000000.0625)).
_SPAN_RELATIVE_TOLERANCE = 1e-6


def _check_adaptive_span(t0, t_end, dt0) -> None:
    """
    Check the time arguments of etd_integrate_adaptive on the host.

    Raise ValueError if t0 or t_end is not concrete, if t_end <= t0, if
    the cast of t0 and t_end to the time dtype changes the span by more
    than _SPAN_RELATIVE_TOLERANCE (relative), or if a concrete dt0 is not
    positive. _etd_tick_plan checks the clock resolution.

    The check uses NumPy scalars, not jnp, so that it gives the same
    result inside an outer jax.jit. A traced dt0 is permitted: it only
    sets the first trial step.
    """
    for name, value in (("t0", t0), ("t_end", t_end)):
        if isinstance(value, jax.core.Tracer):
            raise ValueError(
                f"etd_integrate_adaptive: t_span must be concrete; {name} is a traced "
                f"value. Pass t_span as Python or NumPy numbers (a static argument "
                f"of an outer jax.jit)."
            )
    t0_f = float(t0)
    t_end_f = float(t_end)
    requested = t_end_f - t0_f
    if not requested > 0.0:
        raise ValueError(f"need t_end > t0, got t0={t0}, t_end={t_end}")

    clock = np.float64 if _time_dtype() == jnp.float64 else np.float32
    cast = float(clock(t_end_f) - clock(t0_f))
    if abs(cast - requested) > _SPAN_RELATIVE_TOLERANCE * requested:
        raise ValueError(
            f"the span [{t0}, {t_end}] changes from {requested!r} to {cast!r} when t0 and "
            f"t_end are cast to {np.dtype(clock).name}: the clock resolution at t0 is too "
            f"coarse. Enable x64 or shift the time origin."
        )
    # A subnormal span can be flushed to zero by the CPU, so the loop would
    # not run and the driver would report success without a step.
    if not np.isfinite(np.spacing(clock(cast))):
        raise ValueError(
            f"the span {cast!r} is too large for the {np.dtype(clock).name} clock "
            f"(its spacing is not finite)."
        )
    tiny = float(np.finfo(clock).tiny)
    for name, value in (("t0", t0_f), ("t_end", t_end_f)):
        if value != 0.0 and abs(value) < tiny:
            raise ValueError(
                f"{name}={value!r} is subnormal in {np.dtype(clock).name}; the CPU can "
                f"flush it to zero. Shift the time origin."
            )
    if cast < tiny:
        raise ValueError(
            f"the span {cast!r} is below the smallest normal {np.dtype(clock).name} "
            f"number; the clock cannot resolve it."
        )

    if isinstance(dt0, jax.core.Tracer):
        return
    if not float(dt0) > 0.0:
        raise ValueError(f"need dt0 > 0, got dt0={dt0}")


def etd_integrate_adaptive(
    u0: StateDict,
    t_span: tuple[float, float],
    dt0: float,
    linear_ops: dict[str, FFTLinearOperator],
    nonlinear_rhs: Callable[[StateDict, float], StateDict],
    method: str = 'etdrk4',
    pid_params: PIDParams | None = None,
    max_steps: int = 10000,
    save_every: int = 1,
    max_rejections_per_step: int = _DEFAULT_MAX_REJECTIONS_PER_STEP,
    rhs_uses_time: bool = True,
) -> ETDAdaptiveResult:
    """Integrate u_t = L*u + N(u) with an adaptive time step (ETD1 or ETDRK4).

    The driver is a lax.while_loop. It works on interior-only states with
    the same linear_ops and nonlinear_rhs as etd_integrate, etd1_step and
    etdrk4_step. It does not use moljax.core.stepping.adaptive_integrate,
    which works on MOLModel states with ghost cells.

    Error estimate: step doubling. Each attempt takes one step of size dt
    (y_full) and two steps of size dt/2 (y_half2). The error estimate is
    (y_half2 - y_full) / (2^p - 1), with p = 1 for ETD1 and p = 4 for
    ETDRK4. This is the Richardson estimate of the error of y_half2, so a
    tolerance has the same meaning for each order. An accepted step keeps
    y_half2 (no extrapolation).

    CAUTION: The step-doubling estimators of moljax.core.stepping
    (estimate_error_doubling and estimate_error_imex_doubling) keep the
    raw difference y_half2 - y_full. For the same atol and rtol, those
    drivers therefore accept a different error than this driver.

    Time-step policy "ETD" (see moljax.core.dt_policy.propose_dt_etd):
    there is no CFL limit, because exp(dt*L) is exact. The PI controller
    of PIDParams sets dt, in the range [dt_min, dt_max]. The controller
    receives the order p + 1 of the local error estimate, not p (the
    drivers of moljax.core.stepping give it p). The driver
    rejects a step that has non-finite values and cuts dt for the next
    attempt. The run stops with StatusCode.DT_TOO_SMALL only when a trial
    at dt_min is also rejected.

    The cost of each attempt is three steps (for ETDRK4, 12 evaluations
    of nonlinear_rhs against 4).

    Args:
        u0: Initial state (interior values, no ghost cells)
        t_span: (t0, t_end), with t_end > t0
        dt0: Initial time step. The driver clamps it to [dt_min, dt_max].
        linear_ops: Dict mapping field name to FFTLinearOperator
        nonlinear_rhs: Function computing N(u, t)
        method: 'etd1' or 'etdrk4'
        pid_params: Controller parameters, tolerances and dt limits
            (default PIDParams())
        max_steps: Maximum number of accepted steps. It also sets the size
            of the history buffers.
        save_every: Save the state after each save_every-th accepted step.
            The driver always saves the final accepted step.
        max_rejections_per_step: Consecutive rejections allowed for one
            step before the run stops with StatusCode.MAX_ATTEMPTS_REACHED.
        rhs_uses_time: True (the safe default) if nonlinear_rhs can read
            t. The driver then checks the stage times of each step (see
            the CAUTION below). Set False only for an autonomous
            nonlinear_rhs (one that does not read t); the driver then
            does not make the check.

    Returns:
        ETDAdaptiveResult. The history holds accepted steps only. The last
        accepted step ends exactly at t_end.

    Raises:
        ValueError: For an unknown method, save_every < 1, max_steps < 1,
            max_rejections_per_step < 1, t_end <= t0, dt0 <= 0, a field of
            u0 that does not have a real floating dtype, a traced t0 or
            t_end, a span that the cast to the time dtype changes by more
            than 1e-6 (relative), or a span that the clock cannot divide
            into resolved steps: a span shorter than the smallest step, a
            dt_max or a dt0 below it, or a clock tick below four times the
            smallest normal number (see the clock in the Notes).

    Notes:
        The compiled loop is cached and reused. The cache key holds the
        linear operators and nonlinear_rhs (by identity), the method,
        pid_params, max_steps, save_every, max_rejections_per_step,
        rhs_uses_time and the structure, shapes and dtypes of u0. The
        values of u0, t0, t_end and dt0 are arguments of the compiled loop,
        so a sweep over them compiles once. t_span must be concrete (the
        driver checks it on the host before it runs the loop). The cache holds 32 entries;
        clear_compiled_drivers empties it.

        The clock is carried in float64 when x64 is on, also for a float32
        state (see moljax.core.stepping._time_dtype). The step arithmetic
        uses the dtype of the state.

        The clock is an integer number of ticks. A tick is the power of
        two G = ulp(max(|t0|, |t_end|)) in the time dtype when
        rhs_uses_time=True, and G = ulp(t_end - t0) when it is False. Every
        multiple of G in the span is a float, so each step starts and ends
        at an exact time, the accepted lengths (dt_history) sum exactly to
        t_end - t0, and the last time is t_end. No remainder is discarded.
        An endpoint that is not a multiple of G gives a partial tick, which
        the step at that end includes. The controller step is rounded down
        to whole ticks; to keep t_end reachable, a step can also be up to
        1/8 longer than the proposal (never above dt_max), or end a few
        ticks earlier.

        A nonlinear_rhs that reads t (rhs_uses_time=True, the default) gets
        stage times that are each one rounding of their exact values. A
        doubled step needs at least 2 ticks (ETD1) or 4 ticks (ETDRK4), one
        more with a partial tick, so that its stage times are distinct.
        Before compilation, the driver raises ValueError if the span, dt_max
        or dt0 is below that. If the controller later proposes a smaller
        step, the run stops with StatusCode.TIME_UNRESOLVED before the step.
        The stage times keep the resolution G of the time dtype; for a
        time-dependent nonlinear_rhs keep G small against dt (enable x64, or
        shift the time origin). Near the smallest step, the step-doubling
        estimate can underestimate the error from this rounding.

        CAUTION: With rhs_uses_time=False, the ticks are on the elapsed
        time, and the time given to nonlinear_rhs is t0 + elapsed time,
        rounded. Thus an autonomous problem at a large t0 can use small
        steps. Do not set False for a nonlinear_rhs that reads t: the result
        can be wrong with no message.

        The step functions get the step length in the state dtype. For a
        float32 state with a float64 clock (x64), each length is rounded
        once to float32.
    """
    if method not in _ETD_ADAPTIVE_ORDER:
        raise ValueError(
            f"Unknown method: {method}. Use one of {sorted(_ETD_ADAPTIVE_ORDER)}"
        )
    if save_every < 1:
        raise ValueError(f"save_every must be at least 1, got {save_every}")
    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")
    if max_rejections_per_step < 1:
        raise ValueError(
            f"max_rejections_per_step must be at least 1, got {max_rejections_per_step}"
        )
    if pid_params is None:
        pid_params = PIDParams()

    for name, field in u0.items():
        if not jnp.issubdtype(jnp.result_type(field), jnp.floating):
            raise ValueError(
                f"etd_integrate_adaptive: field {name!r} has dtype "
                f"{jnp.result_type(field)}; use a real floating dtype. An integer "
                f"or Boolean state cannot hold the update and truncates it."
            )
        if jnp.dtype(jnp.result_type(field)) in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
            raise ValueError(
                f"etd_integrate_adaptive: field {name!r} has dtype "
                f"{jnp.result_type(field)}; use float32 or float64. In a 16-bit "
                f"state a small step size becomes zero, so the state cannot change."
            )

    t0, t_end = t_span
    _check_adaptive_span(t0, t_end, dt0)

    compute_dtype = jnp.result_type(*[u0[name] for name in u0])
    plan = _etd_tick_plan(t0, t_end, dt0, method, bool(rhs_uses_time), compute_dtype,
                          pid_params)

    driver = _etd_adaptive_driver(u0, linear_ops, nonlinear_rhs, method, pid_params,
                                  max_steps, save_every, max_rejections_per_step,
                                  bool(rhs_uses_time))
    return driver(u0, plan, jnp.asarray(dt0, dtype=_time_dtype()))


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
