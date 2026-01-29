#!/usr/bin/env python3
"""
Benchmark E2: GMRES Iterations vs Dimension (1D/2D/3D) - Fast Version

Tests GMRES iterations on representative CN Jacobian systems.
Uses multiple random RHS vectors to get iteration statistics.

Output format: median [IQR] (max)
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

# Configuration
CONFIGS = [
    {'dim': 1, 'N': 1000, 'shape': (1000,)},
    {'dim': 2, 'N': 64, 'shape': (64, 64)},
    {'dim': 3, 'N': 16, 'shape': (16, 16, 16)},
]
SIGMA = 10
GMRES_TOL = 1e-6
GMRES_MAXITER = 200
GMRES_RESTART = 30
D = 0.01
REACTION_K = 1.0
N_SAMPLES = 20

print("E2: GMRES Iterations vs Dimension (Fast)")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Stiffness ratio σ = {SIGMA}")
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


for config in CONFIGS:
    dim = config['dim']
    N = config['N']
    shape = config['shape']
    dof = int(np.prod(shape))
    dx = 1.0 / N
    dt = SIGMA * dx**2 / D

    print(f"\n{dim}D: shape={shape}, DOF={dof:,}...")

    np.random.seed(42)

    if dim == 1:
        kx = np.fft.fftfreq(N, dx) * 2 * np.pi
        lap_eig = -kx**2
        lap_eig_jax = jnp.array(lap_eig)
        u0 = jnp.array(np.abs(np.random.randn(N)) + 0.5)

        @jax.jit
        def fd_laplacian(u):
            return (jnp.roll(u, -1) + jnp.roll(u, 1) - 2.0 * u) / (dx ** 2)

        @jax.jit
        def fft_precond(r):
            r_hat = jnp.fft.fft(r)
            result_hat = r_hat / (1 - 0.5 * dt * D * lap_eig_jax)
            return jnp.real(jnp.fft.ifft(result_hat))

    elif dim == 2:
        kx = np.fft.fftfreq(N, dx) * 2 * np.pi
        ky = kx
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        lap_eig = -(KX**2 + KY**2)
        lap_eig_jax = jnp.array(lap_eig)
        u0 = jnp.array(np.abs(np.random.randn(N, N)) + 0.5)

        @jax.jit
        def fd_laplacian(u):
            return (
                jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0) +
                jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1) -
                4.0 * u
            ) / (dx ** 2)

        @jax.jit
        def fft_precond(r):
            r_hat = jnp.fft.fft2(r)
            result_hat = r_hat / (1 - 0.5 * dt * D * lap_eig_jax)
            return jnp.real(jnp.fft.ifft2(result_hat))

    else:  # dim == 3
        kx = np.fft.fftfreq(N, dx) * 2 * np.pi
        ky, kz = kx, kx
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        lap_eig = -(KX**2 + KY**2 + KZ**2)
        lap_eig_jax = jnp.array(lap_eig)
        u0 = jnp.array(np.abs(np.random.randn(N, N, N)) + 0.5)

        @jax.jit
        def fd_laplacian(u):
            return (
                jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0) +
                jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1) +
                jnp.roll(u, -1, axis=2) + jnp.roll(u, 1, axis=2) -
                6.0 * u
            ) / (dx ** 2)

        @jax.jit
        def fft_precond(r):
            r_hat = jnp.fft.fftn(r)
            result_hat = r_hat / (1 - 0.5 * dt * D * lap_eig_jax)
            return jnp.real(jnp.fft.ifftn(result_hat))

    reaction_jac = -2.0 * REACTION_K * u0

    @jax.jit
    def jacobian_matvec(v):
        diffusion = D * fd_laplacian(v)
        reaction = reaction_jac * v
        return v - 0.5 * dt * (diffusion + reaction)

    def matvec_flat(v_flat):
        return jacobian_matvec(v_flat.reshape(shape)).flatten()

    def precond_flat(v_flat):
        return fft_precond(v_flat.reshape(shape)).flatten()

    # Warmup
    _ = matvec_flat(jnp.ones(dof))
    _ = precond_flat(jnp.ones(dof))

    # Test FFT-preconditioned GMRES
    fft_iterations = []
    t0 = time.perf_counter()
    for sample in range(N_SAMPLES):
        np.random.seed(100 + sample)
        rhs = jnp.array(np.random.randn(*shape).flatten())

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

    fft_time = time.perf_counter() - t0
    fft_stats = compute_stats(fft_iterations)
    print(f"  FFT precond: median={fft_stats['median']:.0f}, "
          f"IQR=[{fft_stats['q25']:.0f}-{fft_stats['q75']:.0f}], "
          f"max={fft_stats['max']}, time={fft_time:.2f}s")

    # Test unpreconditioned GMRES
    unprecond_iterations = []
    t0 = time.perf_counter()
    for sample in range(N_SAMPLES):
        np.random.seed(100 + sample)
        rhs = jnp.array(np.random.randn(*shape).flatten())

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

    unprecond_time = time.perf_counter() - t0
    unprecond_stats = compute_stats(unprecond_iterations)
    print(f"  Unprecond:   median={unprecond_stats['median']:.0f}, "
          f"IQR=[{unprecond_stats['q25']:.0f}-{unprecond_stats['q75']:.0f}], "
          f"max={unprecond_stats['max']}, time={unprecond_time:.2f}s")

    results['experiments'].append({
        'dim': dim,
        'N': N,
        'shape': list(shape),
        'dof': dof,
        'n_samples': N_SAMPLES,
        'fft_precond': {**fft_stats, 'time_s': fft_time},
        'unpreconditioned': {**unprecond_stats, 'time_s': unprecond_time},
    })

# Save results
results['config'] = {
    'sigma': SIGMA,
    'gmres_tol': GMRES_TOL,
    'gmres_maxiter': GMRES_MAXITER,
    'gmres_restart': GMRES_RESTART,
    'n_samples': N_SAMPLES,
    'D': D,
    'reaction_k': REACTION_K,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'iter_vs_dim.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary
print("\n" + "=" * 90)
print("SUMMARY: GMRES Iterations vs Dimension (σ=10) - median [IQR] (max)")
print("=" * 90)
print(f"{'Dim':>5} {'Shape':>15} {'DOF':>10} {'FFT Precond':>25} {'Unprecond':>25}")
print("-" * 90)
for exp in results['experiments']:
    shape_str = 'x'.join(map(str, exp['shape']))
    fft = exp['fft_precond']
    unp = exp['unpreconditioned']
    fft_str = f"{fft['median']:.0f} [{fft['q25']:.0f}-{fft['q75']:.0f}] ({fft['max']})"
    if unp['max'] >= GMRES_MAXITER:
        unp_str = f">{GMRES_MAXITER}"
    else:
        unp_str = f"{unp['median']:.0f} [{unp['q25']:.0f}-{unp['q75']:.0f}] ({unp['max']})"
    print(f"{exp['dim']:>5} {shape_str:>15} {exp['dof']:>10,} {fft_str:>25} {unp_str:>25}")
print("=" * 90)
