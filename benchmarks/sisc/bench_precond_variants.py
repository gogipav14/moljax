#!/usr/bin/env python3
"""
Benchmark E4: Preconditioner Variants

Compare preconditioner effectiveness:
- None (identity)
- Jacobi (diagonal)
- Block-Jacobi (block diagonal)
- FFT diffusion (spectral)
- Shifted FFT (shifted spectrum)

Purpose: Ablation study for preconditioning
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
N = 128
STIFFNESS_RATIOS = [1, 10, 100]
GMRES_TOL = 1e-6
GMRES_MAXITER = 5000
D = 0.01
REACTION_K = 10.0

print("E4: Preconditioner Variants")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N}")
print(f"Stiffness ratios: {STIFFNESS_RATIOS}")
print("=" * 60)

dx = 1.0 / N

# FFT eigenvalues
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Initial condition
np.random.seed(42)
u0 = np.abs(np.random.randn(N, N)) + 0.5
u0_jax = jnp.array(u0)
rhs_2d = jnp.array(np.random.randn(N, N))

results = {'experiments': [], 'config': {}}

for sigma in STIFFNESS_RATIOS:
    dt = sigma * dx**2 / D
    print(f"\n--- σ = {sigma} (dt = {dt:.6f}) ---")

    # Matrix-free FD Laplacian
    @jax.jit
    def fd_laplacian_matvec(u_2d):
        return (
            jnp.roll(u_2d, -1, axis=0) + jnp.roll(u_2d, 1, axis=0) +
            jnp.roll(u_2d, -1, axis=1) + jnp.roll(u_2d, 1, axis=1) -
            4.0 * u_2d
        ) / (dx ** 2)

    reaction_jac = 2.0 * REACTION_K * u0_jax

    @jax.jit
    def jacobian_matvec_2d(v_2d):
        diffusion = D * fd_laplacian_matvec(v_2d)
        reaction = reaction_jac * v_2d
        return v_2d - dt * (diffusion + reaction)

    def jacobian_matvec_flat(v_flat):
        return jacobian_matvec_2d(v_flat.reshape(N, N)).flatten()

    rhs_flat = rhs_2d.flatten()

    # === Preconditioner definitions ===

    # 1. None (identity)
    def precond_none(u):
        return u

    # 2. Jacobi (diagonal of J = I - dt*(D*L_diag + diag(reaction_jac)))
    # Diagonal of FD Laplacian is -4/dx^2
    jacobi_diag = 1.0 - dt * (D * (-4.0 / dx**2) + reaction_jac)
    jacobi_diag_flat = jacobi_diag.flatten()

    @jax.jit
    def precond_jacobi_flat(u_flat):
        return u_flat / jacobi_diag_flat

    # 3. Block-Jacobi (treat each row as a block - simplified)
    # For structured grids, this is similar to line relaxation
    @jax.jit
    def precond_block_jacobi(u_2d):
        # Solve tridiagonal system per row (simplified: just scale by avg diagonal)
        avg_diag = 1.0 - dt * (D * (-4.0 / dx**2))
        return u_2d / avg_diag

    def precond_block_jacobi_flat(u_flat):
        return precond_block_jacobi(u_flat.reshape(N, N)).flatten()

    # 4. FFT diffusion preconditioner
    @jax.jit
    def precond_fft_2d(u_2d):
        u_hat = jnp.fft.fft2(u_2d)
        result_hat = u_hat / (1.0 - dt * D * lap_eig_jax)
        return jnp.real(jnp.fft.ifft2(result_hat))

    def precond_fft_flat(u_flat):
        return precond_fft_2d(u_flat.reshape(N, N)).flatten()

    # 5. Shifted FFT (adds small shift for stability)
    shift = 0.1 * REACTION_K  # Account for reaction

    @jax.jit
    def precond_shifted_fft_2d(u_2d):
        u_hat = jnp.fft.fft2(u_2d)
        result_hat = u_hat / (1.0 - dt * (D * lap_eig_jax - shift))
        return jnp.real(jnp.fft.ifft2(result_hat))

    def precond_shifted_fft_flat(u_flat):
        return precond_shifted_fft_2d(u_flat.reshape(N, N)).flatten()

    # Test each preconditioner
    preconditioners = [
        ('None', precond_none),
        ('Jacobi', precond_jacobi_flat),
        ('Block-Jacobi', precond_block_jacobi_flat),
        ('FFT Diffusion', precond_fft_flat),
        ('Shifted FFT', precond_shifted_fft_flat),
    ]

    sigma_results = {'sigma': sigma, 'dt': float(dt), 'preconditioners': {}}

    for name, precond in preconditioners:
        iter_count = [0]

        def counted_matvec(u):
            iter_count[0] += 1
            return precond(jacobian_matvec_flat(u))

        precond_rhs = precond(rhs_flat)

        iter_count[0] = 0
        try:
            t0 = time.perf_counter()
            sol, info = gmres(counted_matvec, precond_rhs, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
            sol.block_until_ready()
            elapsed = time.perf_counter() - t0
            iters = iter_count[0]
            converged = info == 0
        except Exception:
            iters = GMRES_MAXITER
            converged = False
            elapsed = 0

        status = "✓" if converged else "✗"
        print(f"  {name:15s}: {iters:5d} iters, {elapsed*1000:.1f}ms {status}")

        sigma_results['preconditioners'][name] = {
            'iterations': int(iters),
            'converged': bool(converged),
            'time_ms': float(elapsed * 1000),
        }

    results['experiments'].append(sigma_results)

# Save results
results['config'] = {
    'grid_size': N,
    'gmres_tol': GMRES_TOL,
    'gmres_maxiter': GMRES_MAXITER,
    'D': D,
    'reaction_k': REACTION_K,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'precond_variants.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary table
print("\n" + "=" * 80)
print("SUMMARY: GMRES Iterations by Preconditioner")
print("=" * 80)
precond_names = ['None', 'Jacobi', 'Block-Jacobi', 'FFT Diffusion', 'Shifted FFT']
header = f"{'σ':>6}" + "".join(f"{name:>12}" for name in precond_names)
print(header)
print("-" * 80)
for exp in results['experiments']:
    row = f"{exp['sigma']:>6}"
    for name in precond_names:
        iters = exp['preconditioners'][name]['iterations']
        conv = exp['preconditioners'][name]['converged']
        row += f"{iters:>11}{'*' if not conv else ' '}"
    print(row)
print("-" * 80)
print("* = did not converge")
print("=" * 80)

# Generate figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

sigmas = [exp['sigma'] for exp in results['experiments']]
x = np.arange(len(sigmas))
width = 0.15

colors = ['#d62728', '#9467bd', '#8c564b', '#2ca02c', '#1f77b4']

for i, name in enumerate(precond_names):
    iters = [exp['preconditioners'][name]['iterations'] for exp in results['experiments']]
    bars = ax.bar(x + i*width, iters, width, label=name, color=colors[i])

ax.set_yscale('log')
ax.set_xlabel('Stiffness ratio σ', fontsize=12)
ax.set_ylabel('GMRES Iterations (log scale)', fontsize=12)
ax.set_title('Preconditioner Comparison: GMRES Iterations vs Stiffness', fontsize=14)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(sigmas)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=GMRES_MAXITER, color='red', linestyle='--', alpha=0.5, label='Max iter')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_precond_variants.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
