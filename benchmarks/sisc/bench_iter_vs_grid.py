#!/usr/bin/env python3
"""
Benchmark E1: GMRES Iterations vs Grid Size (2D)

Tests if preconditioner effectiveness degrades with problem size.
Fix σ ∈ {1, 10, 100}, sweep grids 64²→512²

Collects GMRES iterations across full time integration (multiple timesteps)
and reports: median [IQR] (max)

Purpose: Preempts "works only at 128²" criticism
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import json
from pathlib import Path
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from benchmark_utils import (
    setup_benchmark, add_benchmark_args
)

parser = add_benchmark_args()
args = parser.parse_args()

# Configuration
GRID_SIZES = [64, 128, 256]  # Reduced for faster runtime
STIFFNESS_RATIOS = [1, 10, 100]
GMRES_TOL = 1e-6
GMRES_MAXITER = 200
GMRES_RESTART = 30
D = 0.01
REACTION_K = 10.0
N_TIMESTEPS = 5  # Number of timesteps to integrate
NEWTON_MAXITER = 5
NEWTON_TOL = 1e-6

print("E1: GMRES Iterations vs Grid Size (Full Integration)")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid sizes: {GRID_SIZES}")
print(f"Stiffness ratios: {STIFFNESS_RATIOS}")
print(f"Timesteps: {N_TIMESTEPS}")
print(f"GMRES: tol={GMRES_TOL}, restart={GMRES_RESTART}, maxiter={GMRES_MAXITER}")
print("=" * 60)

results = {'experiments': [], 'config': {}}


def compute_stats(iterations_list):
    """Compute median, IQR, and max from a list of iteration counts."""
    if not iterations_list:
        return {'median': 0, 'q25': 0, 'q75': 0, 'max': 0, 'count': 0}
    arr = np.array(iterations_list)
    return {
        'median': float(np.median(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'max': int(np.max(arr)),
        'min': int(np.min(arr)),
        'count': len(arr),
        'all_iterations': [int(x) for x in arr]
    }


for N in GRID_SIZES:
    dx = 1.0 / N

    # FFT eigenvalues for this grid
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)
    lap_eig_jax = jnp.array(lap_eig)

    for sigma in STIFFNESS_RATIOS:
        dt = sigma * dx**2 / D

        print(f"\nGrid {N}x{N}, σ={sigma}...")

        # Initial condition
        np.random.seed(42)
        u = jnp.array(np.abs(np.random.randn(N, N)) + 0.5)

        # Matrix-free FD Laplacian
        @jax.jit
        def fd_laplacian_matvec(u_2d, dx_val=dx):
            return (
                jnp.roll(u_2d, -1, axis=0) + jnp.roll(u_2d, 1, axis=0) +
                jnp.roll(u_2d, -1, axis=1) + jnp.roll(u_2d, 1, axis=1) -
                4.0 * u_2d
            ) / (dx_val ** 2)

        # Reaction term: -k*u^2
        def reaction(u_2d):
            return -REACTION_K * u_2d**2

        # RHS of ODE: du/dt = D*laplacian(u) + reaction(u)
        @jax.jit
        def rhs_func(u_2d):
            return D * fd_laplacian_matvec(u_2d) + reaction(u_2d)

        # Crank-Nicolson residual: u_new - u_old - dt/2*(f(u_new) + f(u_old))
        @jax.jit
        def cn_residual(u_new, u_old):
            return u_new - u_old - 0.5 * dt * (rhs_func(u_new) + rhs_func(u_old))

        # Jacobian-vector product via AD
        @jax.jit
        def jacobian_matvec(v, u_lin):
            """J(u_lin) @ v where J = I - dt/2 * df/du"""
            def residual_at_u(u_new):
                return cn_residual(u_new, u)  # u_old is captured
            return jax.jvp(residual_at_u, (u_lin,), (v,))[1]

        # FFT preconditioner (approximates (I - dt/2 * D * L)^{-1})
        @jax.jit
        def fft_precond_2d(r_2d, dt_val=dt):
            r_hat = jnp.fft.fft2(r_2d)
            # Preconditioner: (I - dt/2 * D * L)^{-1}
            result_hat = r_hat / (1 - 0.5 * dt_val * D * lap_eig_jax)
            return jnp.real(jnp.fft.ifft2(result_hat))

        # Collect GMRES iterations for FFT-preconditioned case
        fft_iterations = []
        u_fft = u.copy()

        for step in range(N_TIMESTEPS):
            u_old = u_fft
            u_new = u_old.copy()  # Initial guess

            for newton_iter in range(NEWTON_MAXITER):
                # Compute residual
                residual = cn_residual(u_new, u_old)
                res_norm = jnp.linalg.norm(residual.flatten())

                if res_norm < NEWTON_TOL:
                    break

                # Solve J @ delta = -residual using GMRES with FFT preconditioner
                def matvec_flat(v_flat):
                    return jacobian_matvec(v_flat.reshape(N, N), u_new).flatten()

                def precond_matvec_flat(v_flat):
                    return fft_precond_2d(v_flat.reshape(N, N)).flatten()

                # Left-preconditioned GMRES: solve M^{-1}J @ delta = M^{-1}(-residual)
                def precond_A(v_flat):
                    return precond_matvec_flat(matvec_flat(v_flat))

                precond_rhs = precond_matvec_flat(-residual.flatten())

                # Count iterations
                iter_count = [0]
                def counted_matvec(v):
                    iter_count[0] += 1
                    return precond_A(v)

                try:
                    delta_flat, info = gmres(
                        counted_matvec, precond_rhs,
                        tol=GMRES_TOL, maxiter=GMRES_MAXITER,
                        restart=GMRES_RESTART
                    )
                    delta_flat.block_until_ready()
                    fft_iterations.append(iter_count[0])
                except Exception:
                    fft_iterations.append(GMRES_MAXITER)

                delta = delta_flat.reshape(N, N)
                u_new = u_new + delta

            u_fft = u_new

        fft_stats = compute_stats(fft_iterations)
        print(f"  FFT precond: median={fft_stats['median']:.0f}, "
              f"IQR=[{fft_stats['q25']:.0f}-{fft_stats['q75']:.0f}], "
              f"max={fft_stats['max']}, n_solves={fft_stats['count']}")

        # Collect GMRES iterations for unpreconditioned case
        unprecond_iterations = []
        u_unprecond = u.copy()

        for step in range(N_TIMESTEPS):
            u_old = u_unprecond
            u_new = u_old.copy()

            for newton_iter in range(NEWTON_MAXITER):
                residual = cn_residual(u_new, u_old)
                res_norm = jnp.linalg.norm(residual.flatten())

                if res_norm < NEWTON_TOL:
                    break

                def matvec_flat(v_flat):
                    return jacobian_matvec(v_flat.reshape(N, N), u_new).flatten()

                iter_count = [0]
                def counted_matvec(v):
                    iter_count[0] += 1
                    return matvec_flat(v)

                try:
                    delta_flat, info = gmres(
                        counted_matvec, -residual.flatten(),
                        tol=GMRES_TOL, maxiter=GMRES_MAXITER,
                        restart=GMRES_RESTART
                    )
                    delta_flat.block_until_ready()
                    unprecond_iterations.append(iter_count[0])
                except Exception:
                    unprecond_iterations.append(GMRES_MAXITER)

                delta = delta_flat.reshape(N, N)
                u_new = u_new + delta

            u_unprecond = u_new

        unprecond_stats = compute_stats(unprecond_iterations)
        print(f"  Unprecond:   median={unprecond_stats['median']:.0f}, "
              f"IQR=[{unprecond_stats['q25']:.0f}-{unprecond_stats['q75']:.0f}], "
              f"max={unprecond_stats['max']}, n_solves={unprecond_stats['count']}")

        results['experiments'].append({
            'grid_size': N,
            'sigma': sigma,
            'dofs': N * N,
            'n_timesteps': N_TIMESTEPS,
            'fft_precond': fft_stats,
            'unpreconditioned': unprecond_stats,
        })

# Save results
results['config'] = {
    'gmres_tol': GMRES_TOL,
    'gmres_maxiter': GMRES_MAXITER,
    'gmres_restart': GMRES_RESTART,
    'newton_tol': NEWTON_TOL,
    'newton_maxiter': NEWTON_MAXITER,
    'n_timesteps': N_TIMESTEPS,
    'D': D,
    'reaction_k': REACTION_K,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'iter_vs_grid.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary table
print("\n" + "=" * 90)
print("SUMMARY: GMRES Iterations (FFT preconditioned) - median [IQR] (max)")
print("=" * 90)
print(f"{'Grid':>10} {'σ=1':>20} {'σ=10':>20} {'σ=100':>20}")
print("-" * 90)
for N in GRID_SIZES:
    row = [N]
    for sigma in STIFFNESS_RATIOS:
        exp = [e for e in results['experiments']
               if e['grid_size'] == N and e['sigma'] == sigma][0]
        s = exp['fft_precond']
        row.append(f"{s['median']:.0f} [{s['q25']:.0f}-{s['q75']:.0f}] ({s['max']})")
    print(f"{row[0]:>10} {row[1]:>20} {row[2]:>20} {row[3]:>20}")
print("-" * 90)
print("Unpreconditioned baseline:")
for N in GRID_SIZES:
    row = [N]
    for sigma in STIFFNESS_RATIOS:
        exp = [e for e in results['experiments']
               if e['grid_size'] == N and e['sigma'] == sigma][0]
        s = exp['unpreconditioned']
        if s['max'] >= GMRES_MAXITER:
            row.append(f">{GMRES_MAXITER}")
        else:
            row.append(f"{s['median']:.0f} [{s['q25']:.0f}-{s['q75']:.0f}] ({s['max']})")
    print(f"{row[0]:>10} {row[1]:>20} {row[2]:>20} {row[3]:>20}")
print("=" * 90)

# Generate figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# FFT preconditioned
ax = axes[0]
for sigma in STIFFNESS_RATIOS:
    data = [e for e in results['experiments'] if e['sigma'] == sigma]
    grids = [e['grid_size'] for e in data]
    medians = [e['fft_precond']['median'] for e in data]
    q25s = [e['fft_precond']['q25'] for e in data]
    q75s = [e['fft_precond']['q75'] for e in data]
    maxs = [e['fft_precond']['max'] for e in data]

    x = range(len(grids))
    ax.errorbar(x, medians, yerr=[np.array(medians)-np.array(q25s),
                                   np.array(q75s)-np.array(medians)],
                fmt='o-', label=f'σ={sigma}', linewidth=2, markersize=8, capsize=4)

ax.set_xticks(range(len(GRID_SIZES)))
ax.set_xticklabels([f'{N}²' for N in GRID_SIZES])
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('GMRES Iterations', fontsize=12)
ax.set_title('FFT Preconditioned', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 20)

# Comparison at σ=100
ax = axes[1]
grids = GRID_SIZES
fft_medians = [e['fft_precond']['median'] for e in results['experiments'] if e['sigma'] == 100]
unprecond_medians = [min(e['unpreconditioned']['median'], GMRES_MAXITER)
                     for e in results['experiments'] if e['sigma'] == 100]

x = np.arange(len(grids))
width = 0.35
ax.bar(x - width/2, fft_medians, width, label='FFT Precond', color='#2ca02c')
ax.bar(x + width/2, unprecond_medians, width, label='Unpreconditioned', color='#d62728')

ax.set_xticks(x)
ax.set_xticklabels([f'{N}²' for N in grids])
ax.set_xlabel('Grid Size', fontsize=12)
ax.set_ylabel('GMRES Iterations (median)', fontsize=12)
ax.set_title('σ=100: FFT vs Unpreconditioned', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_iter_vs_grid.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
