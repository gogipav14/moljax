#!/usr/bin/env python3
"""
Benchmark E1: GMRES Iterations vs Grid Size (2D) - Fast Version

Tests GMRES iterations on representative CN Jacobian systems.
Uses multiple random RHS vectors to get iteration statistics.

Output format: median [IQR] (max)
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import json
from pathlib import Path
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

from benchmark_utils import setup_benchmark, add_benchmark_args

parser = add_benchmark_args()
args = parser.parse_args()

# Configuration
GRID_SIZES = [32, 64, 128]
STIFFNESS_RATIOS = [10, 100, 1000]  # Higher stiffness to show contrast
GMRES_TOL = 1e-10  # Tight tolerance
GMRES_MAXITER = 500
GMRES_RESTART = 50  # Larger restart for unpreconditioned
D = 1.0  # Higher diffusion coefficient
REACTION_K = 0.0  # Pure diffusion - shows preconditioner advantage
N_SAMPLES = 10  # Number of random RHS vectors to test

print("E1: GMRES Iterations vs Grid Size (Fast)")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid sizes: {GRID_SIZES}")
print(f"Stiffness ratios: {STIFFNESS_RATIOS}")
print(f"Samples per config: {N_SAMPLES}")
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

        # Representative linearization point
        np.random.seed(42)
        u0 = jnp.array(np.abs(np.random.randn(N, N)) + 0.5)

        # Matrix-free FD Laplacian
        @jax.jit
        def fd_laplacian(u_2d):
            return (
                jnp.roll(u_2d, -1, axis=0) + jnp.roll(u_2d, 1, axis=0) +
                jnp.roll(u_2d, -1, axis=1) + jnp.roll(u_2d, 1, axis=1) -
                4.0 * u_2d
            ) / (dx ** 2)

        # Jacobian of CN residual: J = I - dt/2 * (D*L + diag(df/du))
        # For f(u) = -k*u^2, df/du = -2*k*u
        reaction_jac = -2.0 * REACTION_K * u0

        @jax.jit
        def jacobian_matvec(v):
            diffusion = D * fd_laplacian(v)
            reaction = reaction_jac * v
            return v - 0.5 * dt * (diffusion + reaction)

        # FFT preconditioner: (I - dt/2 * D * L)^{-1}
        @jax.jit
        def fft_precond(r):
            r_hat = jnp.fft.fft2(r)
            result_hat = r_hat / (1 - 0.5 * dt * D * lap_eig_jax)
            return jnp.real(jnp.fft.ifft2(result_hat))

        def matvec_flat(v_flat):
            return jacobian_matvec(v_flat.reshape(N, N)).flatten()

        def precond_flat(v_flat):
            return fft_precond(v_flat.reshape(N, N)).flatten()

        # Warmup
        _ = matvec_flat(jnp.ones(N*N))
        _ = precond_flat(jnp.ones(N*N))

        # Test FFT-preconditioned GMRES
        fft_iterations = []
        for sample in range(N_SAMPLES):
            np.random.seed(100 + sample)
            rhs = jnp.array(np.random.randn(N, N).flatten())

            def precond_A(v):
                return precond_flat(matvec_flat(v))

            precond_rhs = precond_flat(rhs)

            iter_count = [0]
            def counted_matvec(v):
                iter_count[0] += 1
                return precond_A(v)

            try:
                sol, info = gmres(counted_matvec, precond_rhs,
                                  tol=GMRES_TOL, maxiter=GMRES_MAXITER,
                                  restart=GMRES_RESTART)
                sol.block_until_ready()
                fft_iterations.append(iter_count[0])
            except Exception:
                fft_iterations.append(GMRES_MAXITER)

        fft_stats = compute_stats(fft_iterations)
        print(f"  FFT precond: median={fft_stats['median']:.0f}, "
              f"IQR=[{fft_stats['q25']:.0f}-{fft_stats['q75']:.0f}], "
              f"max={fft_stats['max']}")

        # Test unpreconditioned GMRES
        unprecond_iterations = []
        for sample in range(N_SAMPLES):
            np.random.seed(100 + sample)
            rhs = jnp.array(np.random.randn(N, N).flatten())

            iter_count = [0]
            def counted_matvec(v):
                iter_count[0] += 1
                return matvec_flat(v)

            try:
                sol, info = gmres(counted_matvec, rhs,
                                  tol=GMRES_TOL, maxiter=GMRES_MAXITER,
                                  restart=GMRES_RESTART)
                sol.block_until_ready()
                unprecond_iterations.append(iter_count[0])
            except Exception:
                unprecond_iterations.append(GMRES_MAXITER)

        unprecond_stats = compute_stats(unprecond_iterations)
        print(f"  Unprecond:   median={unprecond_stats['median']:.0f}, "
              f"IQR=[{unprecond_stats['q25']:.0f}-{unprecond_stats['q75']:.0f}], "
              f"max={unprecond_stats['max']}")

        results['experiments'].append({
            'grid_size': N,
            'sigma': sigma,
            'dofs': N * N,
            'n_samples': N_SAMPLES,
            'fft_precond': fft_stats,
            'unpreconditioned': unprecond_stats,
        })

# Save results
results['config'] = {
    'gmres_tol': GMRES_TOL,
    'gmres_maxiter': GMRES_MAXITER,
    'gmres_restart': GMRES_RESTART,
    'n_samples': N_SAMPLES,
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
print("SUMMARY: GMRES Iterations - median [IQR] (max)")
print("=" * 90)
print(f"{'Grid':>10} {'σ=1':>20} {'σ=10':>20} {'σ=100':>20}")
print("-" * 90)
print("FFT Preconditioned:")
for N in GRID_SIZES:
    row = [N]
    for sigma in STIFFNESS_RATIOS:
        exp = [e for e in results['experiments']
               if e['grid_size'] == N and e['sigma'] == sigma][0]
        s = exp['fft_precond']
        row.append(f"{s['median']:.0f} [{s['q25']:.0f}-{s['q75']:.0f}] ({s['max']})")
    print(f"{row[0]:>10} {row[1]:>20} {row[2]:>20} {row[3]:>20}")
print("-" * 90)
print("Unpreconditioned:")
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
