#!/usr/bin/env python3
"""
Benchmark E13: 3D Feasibility

Test 3D diffusion with FFT-CN at moderate grid sizes.
Show runtime and memory limitations.

Purpose: Shows 3D limits explicitly
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
import jax.numpy as jnp

from benchmark_utils import setup_benchmark, compute_stats, add_benchmark_args

parser = add_benchmark_args()
args = parser.parse_args()

print("E13: 3D Feasibility Test")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print("=" * 60)

# Configuration
GRID_SIZES_3D = [32, 48, 64, 80, 100, 128]
D = 0.1
T_FINAL = 0.01
N_STEPS = 10
N_REPS = 5

results = {'experiments': [], 'config': {}}

print("\n3D Diffusion Benchmark:")
print(f"{'Grid':>12} {'DOF':>12} {'Memory':>12} {'Time (ms)':>12} {'Error':>14}")
print("-" * 70)

for N in GRID_SIZES_3D:
    L = 1.0
    dx = L / N
    dt = T_FINAL / N_STEPS
    dof = N * N * N

    # Memory estimate
    bytes_per_float = 8
    # Solution + eigenvalues (complex) + workspace
    memory_bytes = dof * bytes_per_float * (1 + 2 + 2)  # ~5 arrays
    memory_mb = memory_bytes / (1024**2)

    print(f"{N}³ ", end='', flush=True)

    try:
        # Grid
        x = np.linspace(0, L, N, endpoint=False)
        X, Y, Z = np.meshgrid(x, x, x, indexing='ij')

        # Manufactured solution: u = exp(-12π²Dt) sin(2πx) sin(2πy) sin(2πz)
        u0 = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y) * np.sin(2*np.pi*Z)
        u0_jax = jnp.array(u0)

        actual_T = N_STEPS * dt
        u_exact = u0 * np.exp(-12*np.pi**2*D*actual_T)

        # FFT eigenvalues (3D)
        kx = np.fft.fftfreq(N, dx) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing='ij')
        lap_eig = -(KX**2 + KY**2 + KZ**2)
        lap_eig_jax = jnp.array(lap_eig)

        # CN amplification factor
        G = (1 + 0.5*dt*D*lap_eig_jax) / (1 - 0.5*dt*D*lap_eig_jax)

        @jax.jit
        def cn_step_3d(u):
            u_hat = jnp.fft.fftn(u)
            return jnp.real(jnp.fft.ifftn(G * u_hat))

        @jax.jit
        def integrate_3d(u0, n_steps):
            def body(i, u):
                return cn_step_3d(u)
            return jax.lax.fori_loop(0, n_steps, body, u0)

        # Warmup (may fail for large grids)
        _ = integrate_3d(u0_jax, N_STEPS).block_until_ready()

        # Timing
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            u_final = integrate_3d(u0_jax, N_STEPS)
            u_final.block_until_ready()
            times.append(time.perf_counter() - t0)

        error = float(jnp.max(jnp.abs(u_final - u_exact)))
        time_median, time_iqr = compute_stats(times)

        print(f"{dof:>12,} {memory_mb:>11.1f}MB {time_median*1000:>11.1f}ms {error:>14.2e}")

        results['experiments'].append({
            'grid_size': N,
            'dof': dof,
            'memory_mb': float(memory_mb),
            'time_ms': float(time_median * 1000),
            'time_iqr_ms': float(time_iqr * 1000),
            'error': error,
            'success': True,
        })

        # Clean up
        del u0_jax, lap_eig_jax, G, u_final

    except Exception as e:
        print(f"{dof:>12,} {memory_mb:>11.1f}MB {'FAILED':>11} {str(e)[:30]}")

        results['experiments'].append({
            'grid_size': N,
            'dof': dof,
            'memory_mb': float(memory_mb),
            'time_ms': None,
            'error': None,
            'success': False,
            'error_msg': str(e)[:100],
        })

# Save results
results['config'] = {
    'D': D,
    't_final': T_FINAL,
    'n_steps': N_STEPS,
    'n_reps': N_REPS,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / '3d_feasibility.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: 3D FFT-CN Feasibility")
print("=" * 70)

successful = [e for e in results['experiments'] if e['success']]
failed = [e for e in results['experiments'] if not e['success']]

if successful:
    max_successful = max(successful, key=lambda x: x['dof'])
    print(f"Maximum successful grid: {max_successful['grid_size']}³ "
          f"({max_successful['dof']:,} DOF)")
    print(f"  Memory: {max_successful['memory_mb']:.1f}MB")
    print(f"  Time: {max_successful['time_ms']:.1f}ms for {N_STEPS} steps")
    print(f"  Error: {max_successful['error']:.2e}")

if failed:
    print(f"\nFailed grids: {', '.join(str(e['grid_size']) + '³' for e in failed)}")

# Extrapolate: estimate memory for larger grids
print("\nProjected requirements for larger grids:")
for N in [128, 192, 256, 384, 512]:
    dof = N ** 3
    memory_gb = dof * 8 * 5 / (1024**3)
    print(f"  {N}³: {dof:>12,} DOF, ~{memory_gb:.1f}GB")

print("=" * 70)

# Generate figure
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Runtime vs grid size
successful_grids = [e['grid_size'] for e in successful]
successful_times = [e['time_ms'] for e in successful]
successful_dofs = [e['dof'] for e in successful]

if successful_grids:
    ax1.semilogy(range(len(successful_grids)), successful_times, 'o-',
                linewidth=2, markersize=10, color='green')
    ax1.set_xticks(range(len(successful_grids)))
    ax1.set_xticklabels([f'{N}³\n({dof//1000}K)' for N, dof in zip(successful_grids, successful_dofs)])
    ax1.set_xlabel('Grid Size (DOF)', fontsize=12)
    ax1.set_ylabel('Time (ms, log scale)', fontsize=12)
    ax1.set_title(f'3D FFT-CN Runtime ({N_STEPS} steps)', fontsize=14)
    ax1.grid(True, alpha=0.3)

    for i, (t, g) in enumerate(zip(successful_times, successful_grids)):
        ax1.annotate(f'{t:.1f}ms', (i, t), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)

# Memory vs DOF
all_grids = [e['grid_size'] for e in results['experiments']]
all_memory = [e['memory_mb'] for e in results['experiments']]
all_success = [e['success'] for e in results['experiments']]

colors = ['green' if s else 'red' for s in all_success]
ax2.bar(range(len(all_grids)), all_memory, color=colors, alpha=0.7)
ax2.axhline(y=8000, color='red', linestyle='--', alpha=0.7, label='8GB GPU')
ax2.set_xticks(range(len(all_grids)))
ax2.set_xticklabels([f'{N}³' for N in all_grids])
ax2.set_xlabel('Grid Size', fontsize=12)
ax2.set_ylabel('Estimated Memory (MB)', fontsize=12)
ax2.set_title('3D Memory Requirements\n(Green=success, Red=failed)', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_3d_feasibility.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
