#!/usr/bin/env python3
"""
Benchmark: Scaling Study (Table 10)

2D Diffusion with Crank-Nicolson + FFT, 1000 steps
Grid sizes: 64x64, 128x128, 256x256, 512x512, 1024x1024

Measures CPU vs GPU timing and speedup ratio.
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
GRID_SIZES = [64, 128, 256, 512, 1024]
N_STEPS = 1000
D = 0.01
N_WARMUP = 2

print("Scaling Study Benchmark")
print("=" * 60)
device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)
print(f"Grid sizes: {GRID_SIZES}")
print(f"Steps per run: {N_STEPS}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

results = {'grids': []}

for N in GRID_SIZES:
    print(f"\n--- Grid: {N}x{N} ({N*N:,} DOFs) ---")

    dx = 1.0 / N
    dt = 0.0001

    # Setup wavenumbers
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)

    # Initial condition
    u0 = np.random.randn(N, N)

    # =========================================================================
    # CPU (NumPy)
    # =========================================================================
    def numpy_solve(u0, n_steps):
        u = u0.copy()
        for _ in range(n_steps):
            u_hat = np.fft.fft2(u)
            u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
            u = np.real(np.fft.ifft2(u_hat))
        return u

    # Warmup
    for _ in range(N_WARMUP):
        _ = numpy_solve(u0, 10)

    # Time
    times_cpu = []
    for _ in range(N_REPS):
        start = time.perf_counter()
        result_cpu = numpy_solve(u0, N_STEPS)
        times_cpu.append(time.perf_counter() - start)

    # Check for NaN/Inf
    if not np.all(np.isfinite(result_cpu)):
        raise ValueError(f"CPU result contains NaN/Inf at N={N}")

    cpu_median, cpu_iqr = compute_stats(times_cpu)
    print(f"  CPU: {cpu_median:.2f} +/- {cpu_iqr:.2f} s")

    # =========================================================================
    # GPU (JAX)
    # =========================================================================
    u0_jax = jnp.array(u0)
    lap_eig_jax = jnp.array(lap_eig)

    @jax.jit
    def jax_step(u, lap_eig):
        u_hat = jnp.fft.fft2(u)
        u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
        return jnp.real(jnp.fft.ifft2(u_hat))

    def jax_solve(u0, n_steps, lap_eig):
        u = u0
        for _ in range(n_steps):
            u = jax_step(u, lap_eig)
        return u

    # Warmup (compile)
    for _ in range(N_WARMUP + 1):
        _ = jax_solve(u0_jax, 10, lap_eig_jax).block_until_ready()

    # Time
    times_gpu = []
    for _ in range(N_REPS):
        start = time.perf_counter()
        result_gpu = jax_solve(u0_jax, N_STEPS, lap_eig_jax)
        result_gpu.block_until_ready()
        times_gpu.append(time.perf_counter() - start)

    # Check for NaN/Inf
    check_finite_tree(result_gpu, f"GPU result N={N}")

    gpu_median, gpu_iqr = compute_stats(times_gpu)

    speedup = cpu_median / gpu_median
    print(f"  GPU: {gpu_median:.2f} +/- {gpu_iqr:.2f} s")
    print(f"  Speedup: {speedup:.1f}x")

    results['grids'].append({
        'N': N,
        'DOFs': N * N,
        'cpu_median_s': float(cpu_median),
        'cpu_iqr_s': float(cpu_iqr),
        'gpu_median_s': float(gpu_median),
        'gpu_iqr_s': float(gpu_iqr),
        'speedup': float(speedup),
    })

# Save results
results['config'] = {
    'n_steps': N_STEPS,
    'n_reps': N_REPS,
    'D': D,
    'device': device_str,
    'dtype': 'float64',
}

output_path = Path(__file__).parent / 'results' / 'scaling.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# =============================================================================
# Summary Table
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY (Table 9)")
print("=" * 70)
print(f"{'Grid':>12} {'DOFs':>12} {'CPU (s)':>12} {'GPU (s)':>12} {'Speedup':>10}")
print("-" * 70)
for g in results['grids']:
    print(f"{g['N']:>5}x{g['N']:<5} {g['DOFs']:>12,} {g['cpu_median_s']:>12.2f} {g['gpu_median_s']:>12.2f} {g['speedup']:>10.1f}x")
print("=" * 70)

# =============================================================================
# Generate Figure
# =============================================================================
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

dofs = [g['DOFs'] for g in results['grids']]
cpu_times = [g['cpu_median_s'] for g in results['grids']]
gpu_times = [g['gpu_median_s'] for g in results['grids']]
speedups = [g['speedup'] for g in results['grids']]

# Left: CPU vs GPU timing (log-log)
ax1.loglog(dofs, cpu_times, 'o-', label='CPU (NumPy)', color='#1f77b4', linewidth=2, markersize=8)
ax1.loglog(dofs, gpu_times, 's-', label='GPU (JAX)', color='#2ca02c', linewidth=2, markersize=8)
ax1.set_xlabel('Degrees of Freedom', fontsize=12)
ax1.set_ylabel('Time (seconds)', fontsize=12)
ax1.set_title('Scaling: CPU vs GPU', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Mark crossover region
ax1.axvline(x=10000, color='gray', linestyle='--', alpha=0.5)
ax1.text(10000, max(cpu_times)*0.5, 'Crossover\n~100x100', ha='center', fontsize=9, color='gray')

# Right: Speedup vs DOFs
ax2.semilogx(dofs, speedups, 'o-', color='#d62728', linewidth=2, markersize=8)
ax2.set_xlabel('Degrees of Freedom', fontsize=12)
ax2.set_ylabel('GPU Speedup (x)', fontsize=12)
ax2.set_title('GPU Speedup vs Problem Size', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

# Add value labels
for x, y in zip(dofs, speedups):
    ax2.annotate(f'{y:.1f}x', (x, y), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'scaling.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
