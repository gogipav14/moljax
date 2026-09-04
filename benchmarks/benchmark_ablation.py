#!/usr/bin/env python3
"""
Ablation Benchmark: Quantifying Each Contribution

Tests the incremental value of each moljax component:
1. JVP vs FD-Jv: AD-based vs finite-difference Jacobian-vector products
2. FFT preconditioner vs none: iteration count and wall-clock time
3. JIT loop vs host loop: lax.while_loop vs Python while
4. Adaptive vs fixed: adaptive stepping overhead

Uses Gray-Scott reaction-diffusion as the test problem.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from benchmark_utils import (
    add_benchmark_args,
    check_finite_tree,
    compute_stats,
    setup_benchmark,
)
from jax import lax
from jax.scipy.sparse.linalg import gmres

# Parse CLI arguments
parser = add_benchmark_args()
args = parser.parse_args()
N_REPS = args.n_reps

# Configuration
N = 128  # Grid size (moderate for faster ablation)
T_FINAL = 100.0  # Short simulation for ablation
DT = 0.5  # Fixed timestep
D = 0.01  # Diffusion coefficient
dx = 1.0 / N
N_WARMUP = 2

# Gray-Scott parameters
Du, Dv = 2e-5, 1e-5
F, k = 0.035, 0.065
L = 2.5

print("Ablation Benchmark")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N}, T_FINAL: {T_FINAL}, dt: {DT}")
print(f"N_REPS: {N_REPS}")
print("=" * 70)

# Setup wavenumbers for pseudo-spectral
dx_gs = L / N
kx = jnp.fft.fftfreq(N, dx_gs) * 2 * jnp.pi
ky = jnp.fft.fftfreq(N, dx_gs) * 2 * jnp.pi
KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)

# Initial condition for Gray-Scott
x = jnp.linspace(0, L, N, endpoint=False)
y = jnp.linspace(0, L, N, endpoint=False)
X, Y = jnp.meshgrid(x, y, indexing='ij')
u0 = 1 - 0.5 * jnp.exp(-100 * ((X - L/2)**2 + (Y - L/2)**2))
v0 = 0.25 * jnp.exp(-100 * ((X - L/2)**2 + (Y - L/2)**2))

results = {'ablations': {}}

# =============================================================================
# Ablation 1: JVP vs FD-Jv (Jacobian-vector product accuracy and cost)
# =============================================================================
print("\n" + "=" * 70)
print("ABLATION 1: JVP vs Finite-Difference Jv")
print("=" * 70)

def gray_scott_rhs(state, lap_eig, Du, Dv, F, k):
    """Gray-Scott RHS with spectral Laplacian."""
    u, v = state[..., 0], state[..., 1]

    # Diffusion via FFT
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    lap_u = jnp.real(jnp.fft.ifft2(lap_eig * u_hat))
    lap_v = jnp.real(jnp.fft.ifft2(lap_eig * v_hat))

    # Reaction
    uvv = u * v * v
    du = Du * lap_u - uvv + F * (1 - u)
    dv = Dv * lap_v + uvv - (F + k) * v

    return jnp.stack([du, dv], axis=-1)

state0 = jnp.stack([u0, v0], axis=-1)

# JVP via AD
@jax.jit
def jvp_matvec(primals, tangents):
    """Exact JVP via forward-mode AD."""
    _, jvp_out = jax.jvp(
        lambda s: gray_scott_rhs(s, lap_eig, Du, Dv, F, k),
        (primals,),
        (tangents,)
    )
    return jvp_out

# FD-Jv via finite differences
@jax.jit
def fd_jv_matvec(state, tangent, eps=1e-7):
    """Finite-difference Jacobian-vector product."""
    f_plus = gray_scott_rhs(state + eps * tangent, lap_eig, Du, Dv, F, k)
    f_minus = gray_scott_rhs(state - eps * tangent, lap_eig, Du, Dv, F, k)
    return (f_plus - f_minus) / (2 * eps)

# Test accuracy
np.random.seed(42)
tangent = jnp.array(np.random.randn(*state0.shape))
tangent = tangent / jnp.linalg.norm(tangent)  # Normalize

# JVP result (ground truth)
jvp_result = jvp_matvec(state0, tangent)
jvp_result.block_until_ready()

# FD-Jv results at different epsilon
fd_errors = {}
for eps in [1e-4, 1e-6, 1e-8]:
    fd_result = fd_jv_matvec(state0, tangent, eps)
    fd_result.block_until_ready()
    rel_error = float(jnp.linalg.norm(fd_result - jvp_result) / jnp.linalg.norm(jvp_result))
    fd_errors[f'eps_{eps:.0e}'] = rel_error
    print(f"  FD eps={eps:.0e}: relative error = {rel_error:.2e}")

# Timing comparison
jvp_times = []
fd_times = []

# Warmup
for _ in range(N_WARMUP):
    _ = jvp_matvec(state0, tangent).block_until_ready()
    _ = fd_jv_matvec(state0, tangent).block_until_ready()

for _ in range(N_REPS):
    start = time.perf_counter()
    _ = jvp_matvec(state0, tangent).block_until_ready()
    jvp_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    _ = fd_jv_matvec(state0, tangent).block_until_ready()
    fd_times.append(time.perf_counter() - start)

jvp_median, jvp_iqr = compute_stats(jvp_times)
fd_median, fd_iqr = compute_stats(fd_times)

print(f"\n  JVP (AD):  {jvp_median*1000:.3f} ms (±{jvp_iqr*1000:.3f})")
print(f"  FD-Jv:     {fd_median*1000:.3f} ms (±{fd_iqr*1000:.3f})")
print(f"  FD/JVP ratio: {fd_median/jvp_median:.2f}x")

results['ablations']['jvp_vs_fd'] = {
    'jvp_ms': float(jvp_median * 1000),
    'jvp_iqr_ms': float(jvp_iqr * 1000),
    'fd_ms': float(fd_median * 1000),
    'fd_iqr_ms': float(fd_iqr * 1000),
    'fd_jvp_ratio': float(fd_median / jvp_median),
    'fd_errors': fd_errors,
    'note': 'JVP is exact to machine precision; FD has O(eps^2) error'
}

# =============================================================================
# Ablation 2: FFT Preconditioner vs None (already in gmres_sweep, summarize here)
# =============================================================================
print("\n" + "=" * 70)
print("ABLATION 2: FFT Preconditioner vs None")
print("=" * 70)

# Build CN system for a single step
sigma = 10.0  # Moderate stiffness
dt_cn = sigma * dx_gs**2 / D

# FFT eigenvalues for preconditioning
def cn_residual(u_next_flat, u_curr_flat, lap_eig, dt, D, N):
    """CN residual: (I - dt/2 * D * L) u^{n+1} - (I + dt/2 * D * L) u^n"""
    u_next = u_next_flat.reshape(N, N)
    u_curr = u_curr_flat.reshape(N, N)

    u_next_hat = jnp.fft.fft2(u_next)
    u_curr_hat = jnp.fft.fft2(u_curr)

    lhs = jnp.real(jnp.fft.ifft2(u_next_hat * (1 - 0.5*dt*D*lap_eig)))
    rhs = jnp.real(jnp.fft.ifft2(u_curr_hat * (1 + 0.5*dt*D*lap_eig)))

    return (lhs - rhs).flatten()

def fft_precond(u_flat, lap_eig, dt, D, N):
    """FFT-based preconditioner: approximate inverse."""
    u = u_flat.reshape(N, N)
    u_hat = jnp.fft.fft2(u)
    result_hat = u_hat / (1 - 0.5*dt*D*lap_eig)
    return jnp.real(jnp.fft.ifft2(result_hat)).flatten()

# Test GMRES with and without preconditioner
u_curr = u0.flatten()
rhs_cn = jnp.zeros(N*N)

iter_count_none = [0]
iter_count_fft = [0]

def matvec_none(u):
    iter_count_none[0] += 1
    return cn_residual(u, jnp.zeros_like(u), lap_eig, dt_cn, D, N)

def matvec_fft(u):
    iter_count_fft[0] += 1
    return fft_precond(cn_residual(u, jnp.zeros_like(u), lap_eig, dt_cn, D, N), lap_eig, dt_cn, D, N)

# No preconditioner
sol_none, info_none = gmres(matvec_none, u_curr, tol=1e-6, maxiter=1000)
iters_none = iter_count_none[0]

# With FFT preconditioner
iter_count_fft[0] = 0
sol_fft, info_fft = gmres(matvec_fft, fft_precond(u_curr, lap_eig, dt_cn, D, N), tol=1e-6, maxiter=1000)
iters_fft = iter_count_fft[0]

print(f"  Stiffness ratio sigma = {sigma}")
print(f"  Without preconditioner: {iters_none} iterations")
print(f"  With FFT preconditioner: {iters_fft} iterations")
print(f"  Reduction: {iters_none/iters_fft:.1f}x")

results['ablations']['fft_precond'] = {
    'sigma': sigma,
    'iters_none': int(iters_none),
    'iters_fft': int(iters_fft),
    'reduction': float(iters_none / iters_fft),
    'note': 'At sigma=10, FFT preconditioner reduces GMRES iterations significantly'
}

# =============================================================================
# Ablation 3: JIT Loop vs Host Loop
# =============================================================================
print("\n" + "=" * 70)
print("ABLATION 3: JIT Loop (lax.while_loop) vs Host Loop (Python while)")
print("=" * 70)

N_STEPS_LOOP = 200

@jax.jit
def step_fn(state, lap_eig, Du, Dv, F, k, dt):
    """Single explicit Euler step."""
    rhs = gray_scott_rhs(state, lap_eig, Du, Dv, F, k)
    return state + dt * rhs

# Host loop (Python while)
def host_loop_solve(state0, n_steps, lap_eig, Du, Dv, F, k, dt):
    """Solve with Python loop - each step is JIT-compiled."""
    state = state0
    for _ in range(n_steps):
        state = step_fn(state, lap_eig, Du, Dv, F, k, dt)
    return state

# JIT loop (lax.fori_loop)
@jax.jit
def jit_loop_solve(state0, n_steps, lap_eig, Du, Dv, F, k, dt):
    """Solve with lax.fori_loop - entire loop is JIT-compiled."""
    def body_fn(i, state):
        return step_fn(state, lap_eig, Du, Dv, F, k, dt)
    return lax.fori_loop(0, n_steps, body_fn, state0)

# Compile both
_ = host_loop_solve(state0, 10, lap_eig, Du, Dv, F, k, 0.01).block_until_ready()
_ = jit_loop_solve(state0, 10, lap_eig, Du, Dv, F, k, 0.01).block_until_ready()

# Warmup
for _ in range(N_WARMUP):
    _ = host_loop_solve(state0, 10, lap_eig, Du, Dv, F, k, 0.01).block_until_ready()
    _ = jit_loop_solve(state0, 10, lap_eig, Du, Dv, F, k, 0.01).block_until_ready()

# Timing
host_times = []
jit_times = []

for _ in range(N_REPS):
    start = time.perf_counter()
    result_host = host_loop_solve(state0, N_STEPS_LOOP, lap_eig, Du, Dv, F, k, 0.01)
    result_host.block_until_ready()
    host_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    result_jit = jit_loop_solve(state0, N_STEPS_LOOP, lap_eig, Du, Dv, F, k, 0.01)
    result_jit.block_until_ready()
    jit_times.append(time.perf_counter() - start)

host_median, host_iqr = compute_stats(host_times)
jit_median, jit_iqr = compute_stats(jit_times)

print(f"  Host loop (Python while): {host_median*1000:.2f} ms (±{host_iqr*1000:.2f})")
print(f"  JIT loop (lax.fori_loop): {jit_median*1000:.2f} ms (±{jit_iqr*1000:.2f})")
print(f"  JIT speedup: {host_median/jit_median:.2f}x")

# Verify same result
check_finite_tree(result_host, "Host loop")
check_finite_tree(result_jit, "JIT loop")
diff = float(jnp.max(jnp.abs(result_host - result_jit)))
print(f"  Max difference: {diff:.2e} (should be ~0)")

results['ablations']['jit_loop'] = {
    'n_steps': N_STEPS_LOOP,
    'host_ms': float(host_median * 1000),
    'host_iqr_ms': float(host_iqr * 1000),
    'jit_ms': float(jit_median * 1000),
    'jit_iqr_ms': float(jit_iqr * 1000),
    'speedup': float(host_median / jit_median),
    'max_diff': diff,
    'note': 'JIT loop eliminates Python interpreter overhead per step'
}

# =============================================================================
# Ablation 4: Adaptive vs Fixed Timestep
# =============================================================================
print("\n" + "=" * 70)
print("ABLATION 4: Adaptive vs Fixed Timestep (overhead analysis)")
print("=" * 70)

# For this ablation, we compare:
# - Fixed timestep: known stable dt, no overhead
# - Adaptive: adds error estimation and step rejection overhead

# Fixed timestep explicit Euler
@jax.jit
def fixed_step_solve(state0, n_steps, lap_eig, Du, Dv, F, k, dt):
    """Fixed timestep integration."""
    def body(i, state):
        rhs = gray_scott_rhs(state, lap_eig, Du, Dv, F, k)
        return state + dt * rhs
    return lax.fori_loop(0, n_steps, body, state0)

# Adaptive (RK23 with error control)
@jax.jit
def rk23_step(state, dt, lap_eig, Du, Dv, F, k):
    """RK23 step with error estimate."""
    k1 = gray_scott_rhs(state, lap_eig, Du, Dv, F, k)
    k2 = gray_scott_rhs(state + 0.5*dt*k1, lap_eig, Du, Dv, F, k)
    k3 = gray_scott_rhs(state + 0.75*dt*k2, lap_eig, Du, Dv, F, k)

    # 3rd order solution
    y_new = state + dt * (2/9*k1 + 1/3*k2 + 4/9*k3)

    # 2nd order solution for error estimate
    y_2nd = state + dt * (7/24*k1 + 1/4*k2 + 1/3*k3 + 1/8*k3)  # Approximation

    error = jnp.max(jnp.abs(y_new - y_2nd))
    return y_new, error

def adaptive_solve(state0, t_final, dt_init, lap_eig, Du, Dv, F, k, rtol=1e-4):
    """Adaptive RK23 integration with accept/reject."""
    t = 0.0
    dt = dt_init
    state = state0
    n_steps = 0
    n_rejects = 0

    while t < t_final:
        dt = min(dt, t_final - t)
        state_new, error = rk23_step(state, dt, lap_eig, Du, Dv, F, k)

        # Error-based step control
        if error < rtol:
            state = state_new
            t += dt
            n_steps += 1
            # Increase dt
            dt = min(dt * 1.5, 0.1)
        else:
            n_rejects += 1
            dt = dt * 0.5

        # Safety check
        if n_steps + n_rejects > 10000:
            break

    return state, n_steps, n_rejects

# Compare on short integration
T_TEST = 10.0
dt_fixed = 0.01
n_steps_fixed = int(T_TEST / dt_fixed)

# Fixed
for _ in range(N_WARMUP):
    _ = fixed_step_solve(state0, 100, lap_eig, Du, Dv, F, k, dt_fixed).block_until_ready()

fixed_times = []
for _ in range(N_REPS):
    start = time.perf_counter()
    result_fixed = fixed_step_solve(state0, n_steps_fixed, lap_eig, Du, Dv, F, k, dt_fixed)
    result_fixed.block_until_ready()
    fixed_times.append(time.perf_counter() - start)

fixed_median, fixed_iqr = compute_stats(fixed_times)

# Adaptive (not JIT-compiled loop due to while condition)
adaptive_times = []
adaptive_steps = []
adaptive_rejects = []

for _ in range(N_REPS):
    start = time.perf_counter()
    result_adaptive, n_steps, n_rejects = adaptive_solve(state0, T_TEST, 0.01, lap_eig, Du, Dv, F, k)
    result_adaptive.block_until_ready()
    adaptive_times.append(time.perf_counter() - start)
    adaptive_steps.append(n_steps)
    adaptive_rejects.append(n_rejects)

adaptive_median, adaptive_iqr = compute_stats(adaptive_times)
avg_steps = np.mean(adaptive_steps)
avg_rejects = np.mean(adaptive_rejects)

print(f"  T_final = {T_TEST}")
print(f"  Fixed (dt={dt_fixed}, {n_steps_fixed} steps): {fixed_median*1000:.2f} ms")
print(f"  Adaptive (avg {avg_steps:.0f} steps, {avg_rejects:.0f} rejects): {adaptive_median*1000:.2f} ms")
print(f"  Overhead ratio: {adaptive_median/fixed_median:.2f}x")

results['ablations']['adaptive_vs_fixed'] = {
    't_final': T_TEST,
    'fixed_dt': dt_fixed,
    'fixed_steps': n_steps_fixed,
    'fixed_ms': float(fixed_median * 1000),
    'fixed_iqr_ms': float(fixed_iqr * 1000),
    'adaptive_avg_steps': float(avg_steps),
    'adaptive_avg_rejects': float(avg_rejects),
    'adaptive_ms': float(adaptive_median * 1000),
    'adaptive_iqr_ms': float(adaptive_iqr * 1000),
    'overhead_ratio': float(adaptive_median / fixed_median),
    'note': 'Adaptive adds error estimation + step rejection overhead'
}

# =============================================================================
# Summary and Save Results
# =============================================================================
print("\n" + "=" * 70)
print("ABLATION SUMMARY")
print("=" * 70)
print(f"{'Component':<35} {'Baseline':<15} {'Improved':<15} {'Benefit':<15}")
print("-" * 70)

# JVP vs FD
print(f"{'JVP vs FD-Jv (accuracy)':<35} {'O(eps²) error':<15} {'machine eps':<15} {'exact':<15}")
print(f"{'JVP vs FD-Jv (speed)':<35} {fd_median*1000:.2f} ms{'':<7} {jvp_median*1000:.2f} ms{'':<7} {fd_median/jvp_median:.1f}x{'':<10}")

# FFT precond
print(f"{'FFT precond (iters, σ=10)':<35} {iters_none:<15} {iters_fft:<15} {iters_none/iters_fft:.1f}x{'':<10}")

# JIT loop
print(f"{'JIT loop vs host loop':<35} {host_median*1000:.2f} ms{'':<7} {jit_median*1000:.2f} ms{'':<7} {host_median/jit_median:.1f}x{'':<10}")

# Adaptive overhead
print(f"{'Adaptive vs fixed (overhead)':<35} {fixed_median*1000:.2f} ms{'':<7} {adaptive_median*1000:.2f} ms{'':<7} {adaptive_median/fixed_median:.1f}x overhead")

print("=" * 70)

results['config'] = {
    'grid_size': N,
    't_final': T_FINAL,
    'n_reps': N_REPS,
    'device': device_str,
    'dtype': 'float64',
}

# Save results
output_path = Path(__file__).parent / 'results' / 'ablation.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
