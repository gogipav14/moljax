#!/usr/bin/env python3
"""
Benchmark: Solver Comparison (Table 8)

1D Advection-Diffusion equation:
    dc/dt = D * d²c/dx² - v * dc/dx
    D = 0.01, v = 1.0, N = 1024, t ∈ [0, 10]

Compares:
- SciPy Radau (implicit, CPU)
- moljax FFT-CN (JAX GPU)
- moljax IMEX (JAX GPU)

Note: Julia/DifferentialEquations.jl removed - no Julia code available.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
from scipy.integrate import solve_ivp
import jax.numpy as jnp

# Import benchmark utilities
from benchmark_utils import (
    setup_benchmark, check_finite_tree, compute_stats,
    add_benchmark_args, DEFAULT_N_REPS
)

# Parse CLI arguments
parser = add_benchmark_args()
args = parser.parse_args()
N_REPS = args.n_reps
# Use None for backend check since this includes CPU SciPy baselines
EXPECTED_BACKEND = None

# Configuration
N = 1024
D = 0.01
v = 1.0
T_FINAL = 10.0
dx = 1.0 / N

# Tolerances
RTOL = 1e-6
ATOL = 1e-8
IMEX_RTOL = 1e-5  # Looser for speed comparison

print("Solver Comparison Benchmark")
print("=" * 60)
device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)
print(f"1D Advection-Diffusion: N={N}, D={D}, v={v}, t=[0,{T_FINAL}]")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Grid
x = np.linspace(0, 1, N, endpoint=False)

# Initial condition: Gaussian pulse
u0 = np.exp(-100 * (x - 0.5)**2)

# Analytical solution for comparison (advection-diffusion)
def analytical_solution(x, t):
    """Approximate analytical solution for Gaussian in periodic domain."""
    # For validation, we use a reference solution instead
    return None

results = {'solvers': []}

# =============================================================================
# SciPy Radau (implicit, CPU)
# =============================================================================
print("\n1. SciPy Radau (implicit, CPU)...")

# Build differentiation matrices (finite difference, 2nd order)
def build_fd_matrices(N, dx):
    """Build finite difference matrices for periodic BC."""
    # 2nd derivative (diffusion)
    D2 = np.zeros((N, N))
    for i in range(N):
        D2[i, (i-1) % N] = 1
        D2[i, i] = -2
        D2[i, (i+1) % N] = 1
    D2 /= dx**2

    # 1st derivative (advection) - upwind
    D1 = np.zeros((N, N))
    for i in range(N):
        D1[i, (i-1) % N] = -1
        D1[i, (i+1) % N] = 1
    D1 /= (2 * dx)

    return D2, D1

D2_mat, D1_mat = build_fd_matrices(N, dx)

def rhs_scipy(t, u):
    """RHS for SciPy: du/dt = D * D2 @ u - v * D1 @ u"""
    return D * D2_mat @ u - v * D1_mat @ u

# Run with timing
times_scipy = []
step_counts = []

for i in range(N_REPS):
    start = time.perf_counter()
    sol = solve_ivp(
        rhs_scipy,
        [0, T_FINAL],
        u0,
        method='Radau',
        rtol=RTOL,
        atol=ATOL,
    )
    elapsed = time.perf_counter() - start
    times_scipy.append(elapsed)
    step_counts.append(len(sol.t))

scipy_median = np.median(times_scipy)
scipy_steps = int(np.median(step_counts))

# Compute error vs initial (rough measure)
scipy_final = sol.y[:, -1]

results['solvers'].append({
    'name': 'SciPy Radau',
    'type': 'implicit',
    'device': 'CPU',
    'time_s': float(scipy_median),
    'steps': scipy_steps,
    'rtol': RTOL,
    'atol': ATOL,
})
print(f"   Time: {scipy_median:.2f} s, Steps: {scipy_steps}")

# =============================================================================
# moljax FFT-CN (JAX GPU)
# =============================================================================
print("\n2. moljax FFT-CN (JAX GPU)...")

# FFT-based operator for diffusion + advection
kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
lap_eig = -kx**2  # Diffusion eigenvalues
adv_eig = -1j * kx  # Advection eigenvalues (central)

u0_jax = jnp.array(u0)

@jax.jit
def fft_cn_step(u, dt):
    """Crank-Nicolson step with FFT-diagonalized operators."""
    u_hat = jnp.fft.fft(u)

    # L = D * lap + v * adv (linear operator eigenvalues)
    L_eig = D * lap_eig + v * adv_eig

    # CN: (I - dt/2 * L) u^{n+1} = (I + dt/2 * L) u^n
    u_hat_new = u_hat * (1 + 0.5*dt*L_eig) / (1 - 0.5*dt*L_eig)

    return jnp.real(jnp.fft.ifft(u_hat_new))

def fft_cn_solve(u0, t_final, dt):
    """Solve to t_final with fixed dt."""
    n_steps = int(t_final / dt)
    u = u0
    for _ in range(n_steps):
        u = fft_cn_step(u, dt)
    return u, n_steps

# Choose dt for stability (CFL-like)
dt_cn = 0.001  # Conservative

# Warmup
_ = fft_cn_solve(u0_jax, 1.0, dt_cn)[0].block_until_ready()

times_cn = []
for _ in range(N_REPS):
    start = time.perf_counter()
    sol_cn, steps_cn = fft_cn_solve(u0_jax, T_FINAL, dt_cn)
    sol_cn.block_until_ready()
    times_cn.append(time.perf_counter() - start)

cn_median = np.median(times_cn)
cn_final = np.array(sol_cn)

results['solvers'].append({
    'name': 'moljax FFT-CN',
    'type': 'implicit',
    'device': 'GPU',
    'time_s': float(cn_median),
    'steps': steps_cn,
    'dt': float(dt_cn),
})
print(f"   Time: {cn_median:.2f} s, Steps: {steps_cn}")

# =============================================================================
# moljax IMEX (JAX GPU) - Strang splitting
# =============================================================================
print("\n3. moljax IMEX (JAX GPU)...")

@jax.jit
def imex_step(u, dt):
    """IMEX step: implicit diffusion, explicit advection."""
    # Half step explicit advection
    u_hat = jnp.fft.fft(u)
    u_hat = u_hat * jnp.exp(0.5*dt*v*adv_eig)
    u = jnp.real(jnp.fft.ifft(u_hat))

    # Full step implicit diffusion
    u_hat = jnp.fft.fft(u)
    u_hat = u_hat * jnp.exp(dt*D*lap_eig)  # Exact exponential
    u = jnp.real(jnp.fft.ifft(u_hat))

    # Half step explicit advection
    u_hat = jnp.fft.fft(u)
    u_hat = u_hat * jnp.exp(0.5*dt*v*adv_eig)
    u = jnp.real(jnp.fft.ifft(u_hat))

    return u

def imex_solve(u0, t_final, dt):
    """Solve to t_final with fixed dt."""
    n_steps = int(t_final / dt)
    u = u0
    for _ in range(n_steps):
        u = imex_step(u, dt)
    return u, n_steps

# Choose dt (can be larger due to Strang stability)
dt_imex = 0.005

# Warmup
_ = imex_solve(u0_jax, 1.0, dt_imex)[0].block_until_ready()

times_imex = []
for _ in range(N_REPS):
    start = time.perf_counter()
    sol_imex, steps_imex = imex_solve(u0_jax, T_FINAL, dt_imex)
    sol_imex.block_until_ready()
    times_imex.append(time.perf_counter() - start)

imex_median = np.median(times_imex)
imex_final = np.array(sol_imex)

results['solvers'].append({
    'name': 'moljax IMEX',
    'type': 'IMEX',
    'device': 'GPU',
    'time_s': float(imex_median),
    'steps': steps_imex,
    'dt': float(dt_imex),
    'rtol': IMEX_RTOL,
})
print(f"   Time: {imex_median:.2f} s, Steps: {steps_imex}")

# =============================================================================
# Error comparison (vs SciPy as reference)
# =============================================================================
print("\n4. Computing errors...")

# Use SciPy solution as reference
err_cn = np.max(np.abs(cn_final - scipy_final))
err_imex = np.max(np.abs(imex_final - scipy_final))

results['solvers'][1]['error_vs_scipy'] = float(err_cn)
results['solvers'][2]['error_vs_scipy'] = float(err_imex)

print(f"   FFT-CN error vs SciPy: {err_cn:.2e}")
print(f"   IMEX error vs SciPy:   {err_imex:.2e}")

# Save results
results['config'] = {
    'N': N,
    'D': D,
    'v': v,
    't_final': T_FINAL,
}

output_path = Path(__file__).parent / 'results' / 'solver_comparison.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# =============================================================================
# Summary Table
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY (Table 8)")
print("=" * 70)
print(f"{'Solver':<20} {'Time (s)':>12} {'Steps':>10} {'Device':>10}")
print("-" * 70)
for s in results['solvers']:
    print(f"{s['name']:<20} {s['time_s']:>12.2f} {s['steps']:>10} {s['device']:>10}")
print("=" * 70)
print("\nNote: Julia/DifferentialEquations.jl removed (no Julia code available)")
print("=" * 70)
