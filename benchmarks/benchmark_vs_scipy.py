#!/usr/bin/env python3
"""
Benchmark comparing JAX FFT-CN against SciPy ODE solvers.
This mirrors the Julia benchmark_diffeq.jl for fair cross-language comparison.

Tests 2D diffusion equation on 128x128 grid for t=[0,1].
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import json
import time
from pathlib import Path
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import jit

# Import benchmark utilities
from benchmark_utils import (
    setup_benchmark, check_finite_tree, compute_stats,
    add_benchmark_args, DEFAULT_N_REPS
)

# Parse CLI arguments
parser = add_benchmark_args()
args = parser.parse_args()
N_RUNS = args.n_reps
# Use None for backend check since this includes CPU SciPy baselines
EXPECTED_BACKEND = None

print("=" * 70)
print("Benchmark: JAX FFT-CN vs SciPy ODE Solvers")
print("=" * 70)
print()

device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)
print()

# =============================================================================
# Problem Setup (same as Julia benchmark)
# =============================================================================

GRID_SIZE = 128
D = 0.01
T_FINAL = 1.0
N_STEPS = 1000
DT = T_FINAL / N_STEPS
N_WARMUP = 2

print("Configuration:")
print(f"  Grid size: {GRID_SIZE} × {GRID_SIZE}")
print(f"  Diffusion coefficient: {D}")
print(f"  Final time: {T_FINAL}")
print(f"  FFT-CN steps: {N_STEPS} (dt = {DT})")
print(f"  N_RUNS: {N_RUNS}")
print()

# Create initial condition
x = np.linspace(0, 1, GRID_SIZE, endpoint=False)
y = np.linspace(0, 1, GRID_SIZE, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')
u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

# Analytical solution
analytical = np.exp(-8 * np.pi**2 * D * T_FINAL) * u0

# =============================================================================
# Method 1: JAX FFT-CN (GPU)
# =============================================================================

def create_cn_factor_jax(N, D, dt):
    """Create Crank-Nicolson factor on GPU."""
    dx = 1.0 / N
    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)
    return (1.0 + 0.5 * dt * D * lap_eig) / (1.0 - 0.5 * dt * D * lap_eig)

@jit
def solve_fft_cn_jax(u0, cn_factor, n_steps):
    """JAX FFT-CN solver using lax.fori_loop for efficiency."""
    def step(i, u):
        u_hat = jnp.fft.fft2(u)
        u_hat = u_hat * cn_factor
        return jnp.real(jnp.fft.ifft2(u_hat))

    return jax.lax.fori_loop(0, n_steps, step, u0)

# =============================================================================
# Method 2: NumPy FFT-CN (CPU)
# =============================================================================

def solve_fft_cn_numpy(u0, D, dt, n_steps):
    """NumPy FFT-CN solver (CPU baseline)."""
    N = u0.shape[0]
    dx = 1.0 / N

    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)
    cn_factor = (1.0 + 0.5 * dt * D * lap_eig) / (1.0 - 0.5 * dt * D * lap_eig)

    u = u0.copy()
    for _ in range(n_steps):
        u_hat = np.fft.fft2(u)
        u_hat = u_hat * cn_factor
        u = np.real(np.fft.ifft2(u_hat))

    return u

# =============================================================================
# Method 3: SciPy solve_ivp with finite differences
# =============================================================================

def create_diffusion_rhs(N, D, dx):
    """Create RHS function for finite-difference diffusion."""
    def rhs(t, u_flat):
        u = u_flat.reshape(N, N)
        du = np.zeros_like(u)

        # Periodic Laplacian via finite differences
        for i in range(N):
            ip1 = (i + 1) % N
            im1 = (i - 1) % N
            for j in range(N):
                jp1 = (j + 1) % N
                jm1 = (j - 1) % N

                lap = (u[ip1, j] + u[im1, j] + u[i, jp1] + u[i, jm1] - 4*u[i, j]) / dx**2
                du[i, j] = D * lap

        return du.flatten()

    return rhs

def solve_scipy(u0, D, t_final, method='Radau'):
    """Solve using SciPy solve_ivp."""
    N = u0.shape[0]
    dx = 1.0 / N

    rhs = create_diffusion_rhs(N, D, dx)

    result = solve_ivp(
        rhs,
        (0, t_final),
        u0.flatten(),
        method=method,
        rtol=1e-3,
        atol=1e-6
    )

    return result.y[:, -1].reshape(N, N)

# =============================================================================
# Run Benchmarks
# =============================================================================

results = {}

# --- JAX FFT-CN (GPU) ---
print("1. JAX FFT-CN (GPU)...")
cn_factor = create_cn_factor_jax(GRID_SIZE, D, DT)
u0_jax = jnp.array(u0)

# Warmup (includes JIT compilation)
for _ in range(N_WARMUP):
    result = solve_fft_cn_jax(u0_jax, cn_factor, N_STEPS)
    result.block_until_ready()

jax_times = []
for i in range(N_RUNS):
    start = time.perf_counter()
    result = solve_fft_cn_jax(u0_jax, cn_factor, N_STEPS)
    result.block_until_ready()
    elapsed = time.perf_counter() - start
    jax_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f} s")

jax_result = np.array(result)
jax_error = np.max(np.abs(jax_result - analytical))
print(f"   Mean: {np.mean(jax_times):.4f} s, Error: {jax_error:.2e}\n")
results["jax_fft_cn"] = {
    "mean_s": float(np.mean(jax_times)),
    "std_s": float(np.std(jax_times)),
    "error": float(jax_error)
}

# --- NumPy FFT-CN (CPU) ---
print("2. NumPy FFT-CN (CPU)...")
for _ in range(N_WARMUP):
    solve_fft_cn_numpy(u0, D, DT, N_STEPS)

numpy_times = []
for i in range(N_RUNS):
    start = time.perf_counter()
    numpy_result = solve_fft_cn_numpy(u0, D, DT, N_STEPS)
    elapsed = time.perf_counter() - start
    numpy_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f} s")

numpy_error = np.max(np.abs(numpy_result - analytical))
print(f"   Mean: {np.mean(numpy_times):.4f} s, Error: {numpy_error:.2e}\n")
results["numpy_fft_cn"] = {
    "mean_s": float(np.mean(numpy_times)),
    "std_s": float(np.std(numpy_times)),
    "error": float(numpy_error)
}

# --- SciPy Radau (implicit, stiff) ---
print("3. SciPy Radau (implicit, finite differences)...")
print("   (This may take a while...)")
for _ in range(1):  # Fewer warmup for slow method
    solve_scipy(u0, D, T_FINAL, method='Radau')

scipy_radau_times = []
for i in range(min(3, N_RUNS)):  # Fewer runs for slow method
    start = time.perf_counter()
    scipy_result = solve_scipy(u0, D, T_FINAL, method='Radau')
    elapsed = time.perf_counter() - start
    scipy_radau_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f} s")

scipy_error = np.max(np.abs(scipy_result - analytical))
print(f"   Mean: {np.mean(scipy_radau_times):.4f} s, Error: {scipy_error:.2e}\n")
results["scipy_radau"] = {
    "mean_s": float(np.mean(scipy_radau_times)),
    "std_s": float(np.std(scipy_radau_times)),
    "error": float(scipy_error)
}

# --- SciPy RK45 (explicit) ---
print("4. SciPy RK45 (explicit, finite differences)...")
for _ in range(1):
    solve_scipy(u0, D, T_FINAL, method='RK45')

scipy_rk45_times = []
for i in range(N_RUNS):
    start = time.perf_counter()
    scipy_rk45_result = solve_scipy(u0, D, T_FINAL, method='RK45')
    elapsed = time.perf_counter() - start
    scipy_rk45_times.append(elapsed)
    print(f"   Run {i+1}: {elapsed:.4f} s")

scipy_rk45_error = np.max(np.abs(scipy_rk45_result - analytical))
print(f"   Mean: {np.mean(scipy_rk45_times):.4f} s, Error: {scipy_rk45_error:.2e}\n")
results["scipy_rk45"] = {
    "mean_s": float(np.mean(scipy_rk45_times)),
    "std_s": float(np.std(scipy_rk45_times)),
    "error": float(scipy_rk45_error)
}

# =============================================================================
# Results Summary
# =============================================================================

print("=" * 70)
print(f"RESULTS SUMMARY ({GRID_SIZE}×{GRID_SIZE} grid, t={T_FINAL})")
print("=" * 70)
print()

baseline = results["scipy_radau"]["mean_s"]

print(f"{'Method':<30}  {'Time (s)':>10}  {'Speedup':>12}  {'Error':>10}")
print("-" * 70)

print(f"{'SciPy Radau (FD)':<30}  {results['scipy_radau']['mean_s']:>10.4f}  {'1.0x (baseline)':>12}  {results['scipy_radau']['error']:>10.2e}")
print(f"{'SciPy RK45 (FD)':<30}  {results['scipy_rk45']['mean_s']:>10.4f}  {baseline/results['scipy_rk45']['mean_s']:>11.1f}x  {results['scipy_rk45']['error']:>10.2e}")
print(f"{'NumPy FFT-CN (CPU)':<30}  {results['numpy_fft_cn']['mean_s']:>10.4f}  {baseline/results['numpy_fft_cn']['mean_s']:>11.1f}x  {results['numpy_fft_cn']['error']:>10.2e}")
print(f"{'JAX FFT-CN (GPU)':<30}  {results['jax_fft_cn']['mean_s']:>10.4f}  {baseline/results['jax_fft_cn']['mean_s']:>11.1f}x  {results['jax_fft_cn']['error']:>10.2e}")

print()

# =============================================================================
# Save Results
# =============================================================================

output = {
    "config": {
        "grid_size": GRID_SIZE,
        "D": D,
        "t_final": T_FINAL,
        "fft_cn_steps": N_STEPS,
        "fft_cn_dt": DT,
        "n_runs": N_RUNS,
        "device": device_str,
        "dtype": "float64"
    },
    "results": results
}

output_path = Path(__file__).parent / "results" / "scipy_comparison.json"
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved to: {output_path}")
