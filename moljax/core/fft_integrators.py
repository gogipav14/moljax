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
from moljax.core.model import CompiledDriverCache, state_cache_key, static_cache_key

# Re-exported so that a caller holding on to ETD drivers can drop them from
# the module they came from; the registry lives with the identity helpers in
# model, and this clears moljax.core.stepping's drivers too.
from moljax.core.model import clear_compiled_drivers as clear_compiled_drivers
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
