#!/usr/bin/env python3
"""
Benchmark E11: FP32 vs FP64 Work-Precision

Compare single vs double precision:
- Work-precision at both precisions
- Show where FP32 is acceptable vs where FP64 is needed

Purpose: Precision guidance for practitioners
"""

import jax
# Start with FP64, will also test FP32
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
import jax.numpy as jnp

from benchmark_utils import setup_benchmark, compute_stats

print("E11: FP32 vs FP64 Work-Precision")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print("=" * 60)

# Configuration
N = 128
L = 1.0
D = 0.1
T_FINAL = 0.1
DT_VALUES = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
N_REPS = 10

dx = L / N

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Manufactured solution: u = exp(-8π²Dt) sin(2πx) sin(2πy)
u0_np = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)

# FFT eigenvalues
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX_fft, KY_fft = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX_fft**2 + KY_fft**2)

results = {'fp64': [], 'fp32': [], 'config': {}}

# ========== FP64 ==========
print("\n=== FP64 ===")
print(f"{'dt':>10} {'Steps':>8} {'Time (ms)':>12} {'Error':>14}")
print("-" * 50)

u0_64 = jnp.array(u0_np, dtype=jnp.float64)
lap_eig_64 = jnp.array(lap_eig, dtype=jnp.complex128)

for dt in DT_VALUES:
    n_steps = int(T_FINAL / dt)
    actual_T = n_steps * dt

    # Exact solution at actual_T
    u_exact = u0_np * np.exp(-8*np.pi**2*D*actual_T)

    # CN amplification factor
    G = (1 + 0.5*dt*D*lap_eig_64) / (1 - 0.5*dt*D*lap_eig_64)

    @jax.jit
    def cn_step_64(u):
        u_hat = jnp.fft.fft2(u)
        return jnp.real(jnp.fft.ifft2(G * u_hat))

    @jax.jit
    def integrate_64(u0, n_steps):
        def body(i, u):
            return cn_step_64(u)
        return jax.lax.fori_loop(0, n_steps, body, u0)

    # Warmup
    _ = integrate_64(u0_64, n_steps).block_until_ready()

    # Timing
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        u_final = integrate_64(u0_64, n_steps)
        u_final.block_until_ready()
        times.append(time.perf_counter() - t0)

    error = float(jnp.max(jnp.abs(u_final - u_exact)))
    time_median, time_iqr = compute_stats(times)

    print(f"{dt:>10.5f} {n_steps:>8} {time_median*1000:>12.3f} {error:>14.2e}")

    results['fp64'].append({
        'dt': dt,
        'n_steps': n_steps,
        'time_ms': float(time_median * 1000),
        'time_iqr_ms': float(time_iqr * 1000),
        'error': error,
    })

# ========== FP32 ==========
print("\n=== FP32 ===")
print(f"{'dt':>10} {'Steps':>8} {'Time (ms)':>12} {'Error':>14}")
print("-" * 50)

u0_32 = jnp.array(u0_np, dtype=jnp.float32)
lap_eig_32 = jnp.array(lap_eig, dtype=jnp.complex64)

for dt in DT_VALUES:
    n_steps = int(T_FINAL / dt)
    actual_T = n_steps * dt

    # Exact solution at actual_T
    u_exact = u0_np * np.exp(-8*np.pi**2*D*actual_T)

    # CN amplification factor (FP32)
    G = (1 + 0.5*dt*D*lap_eig_32) / (1 - 0.5*dt*D*lap_eig_32)

    @jax.jit
    def cn_step_32(u):
        u_hat = jnp.fft.fft2(u)
        return jnp.real(jnp.fft.ifft2(G * u_hat)).astype(jnp.float32)

    @jax.jit
    def integrate_32(u0, n_steps):
        def body(i, u):
            return cn_step_32(u)
        return jax.lax.fori_loop(0, n_steps, body, u0)

    # Warmup
    _ = integrate_32(u0_32, n_steps).block_until_ready()

    # Timing
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        u_final = integrate_32(u0_32, n_steps)
        u_final.block_until_ready()
        times.append(time.perf_counter() - t0)

    error = float(jnp.max(jnp.abs(u_final.astype(jnp.float64) - u_exact)))
    time_median, time_iqr = compute_stats(times)

    print(f"{dt:>10.5f} {n_steps:>8} {time_median*1000:>12.3f} {error:>14.2e}")

    results['fp32'].append({
        'dt': dt,
        'n_steps': n_steps,
        'time_ms': float(time_median * 1000),
        'time_iqr_ms': float(time_iqr * 1000),
        'error': error,
    })

# Save results
results['config'] = {
    'grid_size': N,
    'D': D,
    't_final': T_FINAL,
    'n_reps': N_REPS,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'precision_fp32_fp64.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: FP32 vs FP64 Speedup at Matched Accuracy")
print("=" * 70)

# Find crossover point where FP32 error exceeds FP64 truncation error
fp32_floor = min(r['error'] for r in results['fp32'])
print(f"FP32 error floor: {fp32_floor:.2e}")
print(f"FP64 can achieve: {min(r['error'] for r in results['fp64']):.2e}")

# Compute speedups
print(f"\n{'dt':>10} {'FP64 (ms)':>12} {'FP32 (ms)':>12} {'Speedup':>10}")
print("-" * 50)
for r64, r32 in zip(results['fp64'], results['fp32']):
    speedup = r64['time_ms'] / r32['time_ms']
    print(f"{r64['dt']:>10.5f} {r64['time_ms']:>12.3f} {r32['time_ms']:>12.3f} {speedup:>9.2f}x")
print("=" * 70)

# Generate figure
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Work-precision plot
times_64 = [r['time_ms'] for r in results['fp64']]
errors_64 = [r['error'] for r in results['fp64']]
times_32 = [r['time_ms'] for r in results['fp32']]
errors_32 = [r['error'] for r in results['fp32']]

ax1.loglog(times_64, errors_64, 'o-', label='FP64', linewidth=2, markersize=8, color='blue')
ax1.loglog(times_32, errors_32, 's-', label='FP32', linewidth=2, markersize=8, color='orange')

# Mark FP32 floor
ax1.axhline(y=fp32_floor, color='orange', linestyle='--', alpha=0.7, label='FP32 floor')
ax1.axhline(y=1e-7, color='gray', linestyle=':', alpha=0.5, label='Typical FP32 limit')

ax1.set_xlabel('Runtime (ms)', fontsize=12)
ax1.set_ylabel('Max Error', fontsize=12)
ax1.set_title('Work-Precision: FP32 vs FP64', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Speedup bar chart
dts = [r['dt'] for r in results['fp64']]
speedups = [r64['time_ms'] / r32['time_ms'] for r64, r32 in zip(results['fp64'], results['fp32'])]

ax2.bar(range(len(dts)), speedups, color='green', alpha=0.7)
ax2.set_xticks(range(len(dts)))
ax2.set_xticklabels([f'{dt:.4f}' for dt in dts], rotation=45, ha='right')
ax2.set_xlabel('Timestep dt', fontsize=12)
ax2.set_ylabel('Speedup (FP64 time / FP32 time)', fontsize=12)
ax2.set_title('FP32 Speedup over FP64', fontsize=14)
ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
ax2.grid(True, alpha=0.3, axis='y')

for i, (s, dt) in enumerate(zip(speedups, dts)):
    ax2.text(i, s + 0.05, f'{s:.2f}x', ha='center', fontsize=9)

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_precision_fp32_fp64.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
