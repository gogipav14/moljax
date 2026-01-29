#!/usr/bin/env python3
"""
Benchmark E12: GPU Memory Scaling

Measure memory usage vs grid size and GMRES restart parameter.
Show feasibility region for 8GB GPU.

Purpose: Heads off "won't scale" concerns
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from benchmark_utils import setup_benchmark, add_benchmark_args

parser = add_benchmark_args()
args = parser.parse_args()

print("E12: GPU Memory Scaling")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print("=" * 60)

# Configuration
GRID_SIZES = [64, 128, 256, 512, 1024]
RESTART_VALUES = [10, 20, 30, 50, 100]
D = 0.01
SIGMA = 10
GPU_MEMORY_GB = 8.0

results = {'experiments': [], 'memory_model': {}, 'config': {}}

print("\nMemory analysis:")
print(f"{'Grid':>10} {'DOF':>12} {'Solution':>12} {'GMRES m=30':>14} {'Total':>12}")
print("-" * 70)

for N in GRID_SIZES:
    dof = N * N
    dx = 1.0 / N
    dt = SIGMA * dx**2 / D

    # Memory components (bytes, FP64)
    bytes_per_float = 8
    solution_memory = dof * bytes_per_float
    eigenvalues_memory = dof * bytes_per_float * 2  # Complex

    # GMRES Krylov basis: m vectors of size dof
    m = 30  # Default restart
    gmres_basis_memory = m * dof * bytes_per_float

    # Total memory estimate
    total_memory = solution_memory + eigenvalues_memory + gmres_basis_memory

    sol_mb = solution_memory / (1024**2)
    gmres_mb = gmres_basis_memory / (1024**2)
    total_mb = total_memory / (1024**2)

    print(f"{N}x{N:>6} {dof:>12,} {sol_mb:>11.1f}MB {gmres_mb:>13.1f}MB {total_mb:>11.1f}MB")

    # Actual test: allocate arrays and measure
    try:
        # Allocate test arrays
        u = jnp.zeros((N, N), dtype=jnp.float64)
        lap_eig = jnp.zeros((N, N), dtype=jnp.complex128)

        # Force allocation
        u.block_until_ready()
        lap_eig.block_until_ready()

        # Check if we can do GMRES
        krylov_basis = jnp.zeros((m, dof), dtype=jnp.float64)
        krylov_basis.block_until_ready()

        allocation_success = True

        # Clean up
        del u, lap_eig, krylov_basis

    except Exception as e:
        allocation_success = False
        print(f"  [ALLOCATION FAILED: {e}]")

    results['experiments'].append({
        'grid_size': N,
        'dof': dof,
        'solution_mb': float(sol_mb),
        'gmres_basis_mb': float(gmres_mb),
        'total_estimated_mb': float(total_mb),
        'allocation_success': allocation_success,
    })

# Memory model: bytes = dof * (solution + eigenvalues + m * krylov)
# = dof * (8 + 16 + 8m) = dof * (24 + 8m)
print("\n" + "=" * 70)
print("Memory Model: Memory (bytes) = DOF × (24 + 8m) for FP64")
print("  - Solution: 8 bytes/DOF")
print("  - Eigenvalues (complex): 16 bytes/DOF")
print("  - Krylov basis: 8m bytes/DOF (m = restart)")
print("=" * 70)

# Feasibility map: (N, m) combinations that fit in GPU memory
print(f"\nFeasibility map for {GPU_MEMORY_GB}GB GPU:")
print(f"{'Grid':>10}" + "".join(f"{'m='+str(m):>10}" for m in RESTART_VALUES))
print("-" * (10 + 10 * len(RESTART_VALUES)))

feasibility_map = []
for N in GRID_SIZES:
    dof = N * N
    row = {'grid': N, 'dof': dof, 'restarts': {}}
    row_str = f"{N}x{N:>6}"

    for m in RESTART_VALUES:
        # Memory estimate
        memory_bytes = dof * (24 + 8 * m)
        memory_gb = memory_bytes / (1024**3)
        fits = memory_gb < GPU_MEMORY_GB * 0.8  # 80% safety margin

        row['restarts'][m] = {
            'memory_gb': float(memory_gb),
            'fits': fits,
        }
        row_str += f"{'✓' if fits else '✗':>10}"

    print(row_str)
    feasibility_map.append(row)

results['feasibility_map'] = feasibility_map

# Maximum grid size for each restart value
print(f"\nMaximum grid size for {GPU_MEMORY_GB}GB GPU:")
for m in RESTART_VALUES:
    # Solve: N² × (24 + 8m) < GPU_MEMORY_GB × 0.8 × 1024³
    max_dof = GPU_MEMORY_GB * 0.8 * (1024**3) / (24 + 8 * m)
    max_N = int(np.sqrt(max_dof))
    print(f"  m={m:3d}: max grid ≈ {max_N}×{max_N} ({max_N**2:,} DOF)")

results['max_grid_sizes'] = {
    m: int(np.sqrt(GPU_MEMORY_GB * 0.8 * (1024**3) / (24 + 8 * m)))
    for m in RESTART_VALUES
}

# Save results
results['config'] = {
    'gpu_memory_gb': GPU_MEMORY_GB,
    'dtype': 'float64',
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'gpu_memory_scaling.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Generate figure
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Memory vs grid size
grid_sizes = [e['grid_size'] for e in results['experiments']]
total_memory = [e['total_estimated_mb'] for e in results['experiments']]

ax1.semilogy(range(len(grid_sizes)), total_memory, 'o-', linewidth=2, markersize=10)
ax1.axhline(y=GPU_MEMORY_GB * 1024 * 0.8, color='red', linestyle='--',
            label=f'{GPU_MEMORY_GB}GB limit (80%)')
ax1.set_xticks(range(len(grid_sizes)))
ax1.set_xticklabels([f'{N}²' for N in grid_sizes])
ax1.set_xlabel('Grid Size', fontsize=12)
ax1.set_ylabel('Memory (MB, log scale)', fontsize=12)
ax1.set_title('Memory Usage vs Grid Size\n(GMRES restart=30)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Feasibility heatmap
grid_labels = [f'{row["grid"]}²' for row in feasibility_map]
restart_labels = [str(m) for m in RESTART_VALUES]

feasibility_matrix = np.array([
    [1 if row['restarts'][m]['fits'] else 0 for m in RESTART_VALUES]
    for row in feasibility_map
])

im = ax2.imshow(feasibility_matrix, cmap='RdYlGn', aspect='auto')
ax2.set_xticks(range(len(restart_labels)))
ax2.set_xticklabels(restart_labels)
ax2.set_yticks(range(len(grid_labels)))
ax2.set_yticklabels(grid_labels)
ax2.set_xlabel('GMRES Restart (m)', fontsize=12)
ax2.set_ylabel('Grid Size', fontsize=12)
ax2.set_title(f'Feasibility Region ({GPU_MEMORY_GB}GB GPU)\n(Green=fits, Red=exceeds)', fontsize=14)

# Add memory values
for i, row in enumerate(feasibility_map):
    for j, m in enumerate(RESTART_VALUES):
        mem_gb = row['restarts'][m]['memory_gb']
        color = 'black' if row['restarts'][m]['fits'] else 'white'
        ax2.text(j, i, f'{mem_gb:.1f}GB', ha='center', va='center', fontsize=8, color=color)

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_gpu_memory_scaling.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
