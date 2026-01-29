#!/usr/bin/env python3
"""
Benchmark: JIT Compilation Speedup (Table 3)

2D Diffusion with FFT-based Crank-Nicolson
Grid: 256x256, 1000 time steps

Measures:
- NumPy (vectorized) baseline
- JAX JIT (GPU) with compilation time and steady-state time
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
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
# Use None for backend check since this is a mixed CPU/GPU benchmark
EXPECTED_BACKEND = None

# Configuration
N = 256  # Grid size
N_STEPS = 1000
D = 0.01  # Diffusion coefficient
dx = 1.0 / N
dt = 0.0001
N_WARMUP = 3

print(f"JIT Speedup Benchmark")
print("=" * 60)
device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)
print(f"Grid: {N}x{N}, Steps: {N_STEPS}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Setup wavenumbers
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)

# Initial condition
u0 = np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)[:, None]) * \
     np.sin(2 * np.pi * np.linspace(0, 1, N, endpoint=False)[None, :])

results = {}

# =============================================================================
# NumPy Baseline (vectorized, CPU)
# =============================================================================
print("\n1. NumPy (vectorized CPU)...")

def numpy_cn_solve(u0, n_steps, lap_eig, dt, D):
    """Crank-Nicolson with FFT-diagonalized diffusion."""
    u = u0.copy()
    for _ in range(n_steps):
        u_hat = np.fft.fft2(u)
        # CN: (I - dt/2 * D * L) u^{n+1} = (I + dt/2 * D * L) u^n
        u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
        u = np.real(np.fft.ifft2(u_hat))
    return u

# Warmup
for _ in range(N_WARMUP):
    _ = numpy_cn_solve(u0, 10, lap_eig, dt, D)

# Timing
times = []
for _ in range(N_REPS):
    start = time.perf_counter()
    result = numpy_cn_solve(u0, N_STEPS, lap_eig, dt, D)
    times.append(time.perf_counter() - start)

# Check for NaN/Inf
if not np.all(np.isfinite(result)):
    raise ValueError("NumPy result contains NaN/Inf")

numpy_median, numpy_iqr = compute_stats(times)

results['numpy_cpu'] = {
    'median_s': float(numpy_median),
    'iqr_s': float(numpy_iqr),
    'ms_per_step': float(numpy_median * 1000 / N_STEPS),
}
print(f"   Median: {numpy_median:.2f} +/- {numpy_iqr:.2f} s")
print(f"   Per step: {numpy_median * 1000 / N_STEPS:.3f} ms/step")

# =============================================================================
# JAX JIT (GPU)
# =============================================================================
print("\n2. JAX JIT (GPU)...")

# Convert to JAX arrays on GPU
u0_jax = jnp.array(u0)
lap_eig_jax = jnp.array(lap_eig)

@jax.jit
def jax_cn_step(u, lap_eig, dt, D):
    """Single Crank-Nicolson step."""
    u_hat = jnp.fft.fft2(u)
    u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
    return jnp.real(jnp.fft.ifft2(u_hat))

def jax_cn_solve(u0, n_steps, lap_eig, dt, D):
    """Solve with Python loop (each step is JIT-compiled)."""
    u = u0
    for _ in range(n_steps):
        u = jax_cn_step(u, lap_eig, dt, D)
    return u

# Measure compilation time (first call)
start_compile = time.perf_counter()
result_compile = jax_cn_solve(u0_jax, N_STEPS, lap_eig_jax, dt, D)
result_compile.block_until_ready()
compile_time = time.perf_counter() - start_compile
print(f"   Compile time (first call): {compile_time:.2f} s")

# Warmup (already compiled)
for _ in range(N_WARMUP):
    _ = jax_cn_solve(u0_jax, 10, lap_eig_jax, dt, D).block_until_ready()

# Steady-state timing
times = []
for _ in range(N_REPS):
    start = time.perf_counter()
    result = jax_cn_solve(u0_jax, N_STEPS, lap_eig_jax, dt, D)
    result.block_until_ready()
    times.append(time.perf_counter() - start)

# Check for NaN/Inf
check_finite_tree(result, "JAX result")

jax_median, jax_iqr = compute_stats(times)

results['jax_gpu'] = {
    'compile_s': float(compile_time),
    'median_s': float(jax_median),
    'iqr_s': float(jax_iqr),
    'ms_per_step': float(jax_median * 1000 / N_STEPS),
}
print(f"   Steady-state: {jax_median:.2f} +/- {jax_iqr:.2f} s")
print(f"   Per step: {jax_median * 1000 / N_STEPS:.3f} ms/step")

# =============================================================================
# Summary
# =============================================================================
speedup = numpy_median / jax_median

results['speedup'] = float(speedup)
results['config'] = {
    'grid_size': N,
    'n_steps': N_STEPS,
    'n_reps': N_REPS,
    'D': D,
    'dt': dt,
    'device': device_str,
    'dtype': 'float64',
}

print("\n" + "=" * 60)
print("SUMMARY (Table 5)")
print("=" * 60)
print(f"NumPy (vectorized) CPU: {numpy_median:.2f} +/- {numpy_iqr:.2f} s")
print(f"JAX JIT GPU:            {jax_median:.2f} +/- {jax_iqr:.2f} s")
print(f"Compile time:           {compile_time:.2f} s")
print(f"Speedup:                {speedup:.1f}x")
print("=" * 60)

# Save results
output_path = Path(__file__).parent / 'results' / 'jit_speedup.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Generate figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))

labels = ['NumPy\n(vectorized CPU)', 'JAX JIT\n(GPU)']
times_plot = [numpy_median, jax_median]
errors = [numpy_iqr/2, jax_iqr/2]
colors = ['#1f77b4', '#2ca02c']

bars = ax.bar(labels, times_plot, yerr=errors, capsize=5, color=colors, edgecolor='black')

ax.set_ylabel('Time (seconds)', fontsize=12)
ax.set_title(f'JIT Compilation Speedup\n{N}x{N} grid, {N_STEPS} steps', fontsize=14)
ax.set_ylim(0, max(times_plot) * 1.2)

# Add value labels
for bar, val, err in zip(bars, times_plot, errors):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + err + 0.02,
            f'{val:.2f}s', ha='center', va='bottom', fontsize=11)

# Add speedup annotation
ax.annotate(f'{speedup:.1f}x\nspeedup',
            xy=(1, jax_median), xytext=(1.3, numpy_median/2),
            fontsize=12, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='green'),
            color='green')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'jit_speedup.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
