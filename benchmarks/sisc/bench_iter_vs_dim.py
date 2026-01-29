#!/usr/bin/env python3
"""
Benchmark E2: GMRES Iterations vs Dimension (1D/2D/3D)

Tests dimensionality effects on Toeplitz preconditioning.
Protocol: ~1M DOF across dimensions (1D: N=1e6, 2D: 1024², 3D: 100³)

Collects GMRES iterations across full time integration (multiple timesteps)
and reports: median [IQR] (max)

Purpose: Validate preconditioning across dimensions
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

# Configuration - scaled DOF per dimension for reasonable runtime
CONFIGS = [
    {'dim': 1, 'N': 1000, 'shape': (1000,)},        # 1D: 1K points
    {'dim': 2, 'N': 64, 'shape': (64, 64)},         # 2D: ~4K points
    {'dim': 3, 'N': 16, 'shape': (16, 16, 16)},     # 3D: 4K points
]
SIGMA = 10  # Fixed stiffness ratio
GMRES_TOL = 1e-6
GMRES_MAXITER = 200
GMRES_RESTART = 30
D = 0.01
REACTION_K = 1.0
N_TIMESTEPS = 5
NEWTON_MAXITER = 5
NEWTON_TOL = 1e-6

print("E2: GMRES Iterations vs Dimension (Full Integration)")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Stiffness ratio σ = {SIGMA}")
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
        # 1D FFT eigenvalues
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
        # 2D FFT eigenvalues
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
        # 3D FFT eigenvalues
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

    # Reaction term
    def reaction(u):
        return -REACTION_K * u**2

    @jax.jit
    def rhs_func(u):
        return D * fd_laplacian(u) + reaction(u)

    @jax.jit
    def cn_residual(u_new, u_old):
        return u_new - u_old - 0.5 * dt * (rhs_func(u_new) + rhs_func(u_old))

    # Collect GMRES iterations for FFT-preconditioned case
    fft_iterations = []
    u_fft = u0.copy()

    t0 = time.perf_counter()
    for step in range(N_TIMESTEPS):
        u_old = u_fft
        u_new = u_old.copy()

        for newton_iter in range(NEWTON_MAXITER):
            residual = cn_residual(u_new, u_old)
            res_norm = jnp.linalg.norm(residual.flatten())

            if res_norm < NEWTON_TOL:
                break

            # Jacobian-vector product via AD
            def residual_at_u(u_trial):
                return cn_residual(u_trial, u_old)

            def jacobian_matvec(v):
                return jax.jvp(residual_at_u, (u_new,), (v,))[1]

            def matvec_flat(v_flat):
                return jacobian_matvec(v_flat.reshape(shape)).flatten()

            def precond_flat(v_flat):
                return fft_precond(v_flat.reshape(shape)).flatten()

            def precond_A(v_flat):
                return precond_flat(matvec_flat(v_flat))

            precond_rhs = precond_flat(-residual.flatten())

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

            delta = delta_flat.reshape(shape)
            u_new = u_new + delta

        u_fft = u_new

    fft_time = time.perf_counter() - t0
    fft_stats = compute_stats(fft_iterations)
    print(f"  FFT precond: median={fft_stats['median']:.0f}, "
          f"IQR=[{fft_stats['q25']:.0f}-{fft_stats['q75']:.0f}], "
          f"max={fft_stats['max']}, n_solves={fft_stats['count']}, time={fft_time:.2f}s")

    # Collect GMRES iterations for unpreconditioned case
    unprecond_iterations = []
    u_unprecond = u0.copy()

    t0 = time.perf_counter()
    for step in range(N_TIMESTEPS):
        u_old = u_unprecond
        u_new = u_old.copy()

        for newton_iter in range(NEWTON_MAXITER):
            residual = cn_residual(u_new, u_old)
            res_norm = jnp.linalg.norm(residual.flatten())

            if res_norm < NEWTON_TOL:
                break

            def residual_at_u(u_trial):
                return cn_residual(u_trial, u_old)

            def jacobian_matvec(v):
                return jax.jvp(residual_at_u, (u_new,), (v,))[1]

            def matvec_flat(v_flat):
                return jacobian_matvec(v_flat.reshape(shape)).flatten()

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

            delta = delta_flat.reshape(shape)
            u_new = u_new + delta

        u_unprecond = u_new

    unprecond_time = time.perf_counter() - t0
    unprecond_stats = compute_stats(unprecond_iterations)
    print(f"  Unprecond:   median={unprecond_stats['median']:.0f}, "
          f"IQR=[{unprecond_stats['q25']:.0f}-{unprecond_stats['q75']:.0f}], "
          f"max={unprecond_stats['max']}, n_solves={unprecond_stats['count']}, time={unprecond_time:.2f}s")

    results['experiments'].append({
        'dim': dim,
        'N': N,
        'shape': list(shape),
        'dof': dof,
        'n_timesteps': N_TIMESTEPS,
        'fft_precond': {**fft_stats, 'time_s': fft_time},
        'unpreconditioned': {**unprecond_stats, 'time_s': unprecond_time},
    })

# Save results
results['config'] = {
    'sigma': SIGMA,
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

# Generate figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# FFT preconditioned by dimension
ax = axes[0]
dims = [e['dim'] for e in results['experiments']]
medians = [e['fft_precond']['median'] for e in results['experiments']]
q25s = [e['fft_precond']['q25'] for e in results['experiments']]
q75s = [e['fft_precond']['q75'] for e in results['experiments']]
dofs = [e['dof'] for e in results['experiments']]

bars = ax.bar(range(len(dims)), medians, color=['#1f77b4', '#2ca02c', '#ff7f0e'],
              yerr=[np.array(medians)-np.array(q25s), np.array(q75s)-np.array(medians)],
              capsize=5)
ax.set_xticks(range(len(dims)))
ax.set_xticklabels([f'{d}D\n({dof//1000}K DOF)' for d, dof in zip(dims, dofs)])
ax.set_ylabel('GMRES Iterations (median)', fontsize=12)
ax.set_title('FFT Preconditioned', fontsize=14)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(medians) + 5)

for bar, m in zip(bars, medians):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{m:.0f}', ha='center', fontsize=11)

# Comparison: FFT vs Unpreconditioned
ax = axes[1]
fft_medians = [e['fft_precond']['median'] for e in results['experiments']]
unprecond_medians = [min(e['unpreconditioned']['median'], GMRES_MAXITER)
                     for e in results['experiments']]

x = np.arange(len(dims))
width = 0.35
ax.bar(x - width/2, fft_medians, width, label='FFT Precond', color='#2ca02c')
ax.bar(x + width/2, unprecond_medians, width, label='Unpreconditioned', color='#d62728')

ax.set_xticks(x)
ax.set_xticklabels([f'{d}D' for d in dims])
ax.set_ylabel('GMRES Iterations (median)', fontsize=12)
ax.set_title('FFT vs Unpreconditioned', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
ax.set_yscale('log')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_iter_vs_dim.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
