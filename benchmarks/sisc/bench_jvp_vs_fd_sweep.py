#!/usr/bin/env python3
"""
Benchmark E9: FD-Jv Epsilon Sweep

Compare AD-JVP vs finite-difference JVP accuracy:
- Sweep ε ∈ {1e-2, ..., 1e-12}
- Show U-shaped error curve (truncation vs roundoff)
- Demonstrate FD stagnation regime

Purpose: Shows FD stagnation regime, validates AD advantage
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
N = 64
EPSILONS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12]
D = 0.01
DT = 0.01
REACTION_K = 10.0
GMRES_TOL = 1e-10
GMRES_MAXITER = 200

print("E9: JVP vs Finite-Difference Epsilon Sweep")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N}")
print(f"Epsilons: {EPSILONS}")
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

# Random vector for JVP
v = jnp.array(np.random.randn(N, N))
v_flat = v.flatten()

# Define residual function for Newton iteration
@jax.jit
def fd_laplacian(u):
    return (
        jnp.roll(u, -1, axis=0) + jnp.roll(u, 1, axis=0) +
        jnp.roll(u, -1, axis=1) + jnp.roll(u, 1, axis=1) -
        4.0 * u
    ) / (dx ** 2)

@jax.jit
def reaction(u):
    return REACTION_K * u * u * (1 - u)

@jax.jit
def rhs(u):
    return D * fd_laplacian(u) + reaction(u)

# Residual: R(u) = u - u_n - dt * f(u) for backward Euler
u_n = u0_jax  # Previous timestep

@jax.jit
def residual(u):
    return u - u_n - DT * rhs(u)

def residual_flat(u_flat):
    return residual(u_flat.reshape(N, N)).flatten()

# ========== 1. AD-based JVP (reference) ==========
print("\nComputing AD-JVP (reference)...")

@jax.jit
def jvp_ad(u_flat, v_flat):
    """Exact Jacobian-vector product via forward-mode AD."""
    primals, tangents = jax.jvp(residual_flat, (u_flat,), (v_flat,))
    return tangents

u_flat = u0_jax.flatten()

t0 = time.perf_counter()
jvp_exact = jvp_ad(u_flat, v_flat)
jvp_exact.block_until_ready()
time_ad = time.perf_counter() - t0

jvp_exact_norm = float(jnp.linalg.norm(jvp_exact))
print(f"  AD-JVP norm: {jvp_exact_norm:.10e}")
print(f"  Time: {time_ad*1000:.3f}ms")

results = {
    'ad_jvp': {
        'norm': jvp_exact_norm,
        'time_ms': float(time_ad * 1000),
    },
    'fd_jvp': [],
    'config': {},
}

# ========== 2. Finite-difference JVP sweep ==========
print("\nFinite-difference epsilon sweep:")
print(f"{'ε':>12} {'‖Jv‖':>14} {'Rel Error':>14} {'Time':>10}")
print("-" * 60)

for eps in EPSILONS:
    # Forward FD: J*v ≈ (R(u + ε*v) - R(u)) / ε
    @jax.jit
    def jvp_fd_forward(u_flat, v_flat, eps_val):
        return (residual_flat(u_flat + eps_val * v_flat) - residual_flat(u_flat)) / eps_val

    # Central FD: J*v ≈ (R(u + ε*v) - R(u - ε*v)) / (2ε)
    @jax.jit
    def jvp_fd_central(u_flat, v_flat, eps_val):
        return (residual_flat(u_flat + eps_val * v_flat) -
                residual_flat(u_flat - eps_val * v_flat)) / (2 * eps_val)

    # Use central differences (more accurate)
    t0 = time.perf_counter()
    jvp_fd = jvp_fd_central(u_flat, v_flat, eps)
    jvp_fd.block_until_ready()
    time_fd = time.perf_counter() - t0

    jvp_fd_norm = float(jnp.linalg.norm(jvp_fd))
    rel_error = float(jnp.linalg.norm(jvp_fd - jvp_exact) / jnp.linalg.norm(jvp_exact))

    print(f"{eps:>12.0e} {jvp_fd_norm:>14.6e} {rel_error:>14.6e} {time_fd*1000:>9.3f}ms")

    results['fd_jvp'].append({
        'epsilon': float(eps),
        'norm': jvp_fd_norm,
        'relative_error': rel_error,
        'time_ms': float(time_fd * 1000),
    })

# ========== 3. GMRES convergence comparison ==========
print("\nGMRES convergence comparison:")

# FFT preconditioner
@jax.jit
def fft_precond(u_2d):
    u_hat = jnp.fft.fft2(u_2d)
    result_hat = u_hat / (1.0 - DT * D * lap_eig_jax)
    return jnp.real(jnp.fft.ifft2(result_hat))

def precond_flat(u_flat):
    return fft_precond(u_flat.reshape(N, N)).flatten()

rhs_gmres = jnp.array(np.random.randn(N * N))
precond_rhs = precond_flat(rhs_gmres)

# AD-based GMRES
def ad_matvec(v_flat):
    return jvp_ad(u_flat, v_flat)

def precond_ad_matvec(v_flat):
    return precond_flat(ad_matvec(v_flat))

iter_count_ad = [0]
def counted_ad_matvec(v):
    iter_count_ad[0] += 1
    return precond_ad_matvec(v)

iter_count_ad[0] = 0
sol_ad, info_ad = gmres(counted_ad_matvec, precond_flat(precond_rhs),
                        tol=GMRES_TOL, maxiter=GMRES_MAXITER)
sol_ad.block_until_ready()
iters_ad = iter_count_ad[0]

print(f"\n  AD-JVP GMRES: {iters_ad} iterations")

results['gmres_comparison'] = {'ad': {'iterations': iters_ad}}

# FD-based GMRES at different epsilons
test_epsilons = [1e-4, 1e-6, 1e-8, 1e-10]
print(f"  FD-JVP GMRES at various ε:")

for eps in test_epsilons:
    def fd_matvec(v_flat, eps_val=eps):
        return (residual_flat(u_flat + eps_val * v_flat) -
                residual_flat(u_flat - eps_val * v_flat)) / (2 * eps_val)

    def precond_fd_matvec(v_flat):
        return precond_flat(fd_matvec(v_flat))

    iter_count_fd = [0]
    def counted_fd_matvec(v):
        iter_count_fd[0] += 1
        return precond_fd_matvec(v)

    iter_count_fd[0] = 0
    try:
        sol_fd, info_fd = gmres(counted_fd_matvec, precond_flat(precond_rhs),
                               tol=GMRES_TOL, maxiter=GMRES_MAXITER)
        sol_fd.block_until_ready()
        iters_fd = iter_count_fd[0]
        converged = info_fd == 0
    except:
        iters_fd = GMRES_MAXITER
        converged = False

    status = "✓" if converged else "✗"
    print(f"    ε={eps:.0e}: {iters_fd} iterations {status}")

    results['gmres_comparison'][f'fd_eps_{eps}'] = {
        'iterations': iters_fd,
        'converged': converged,
    }

# Save results
results['config'] = {
    'grid_size': N,
    'D': D,
    'dt': DT,
    'reaction_k': REACTION_K,
    'gmres_tol': GMRES_TOL,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'jvp_vs_fd_sweep.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Generate figure
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: Error vs epsilon (U-shaped curve)
epsilons = [d['epsilon'] for d in results['fd_jvp']]
errors = [d['relative_error'] for d in results['fd_jvp']]

ax1.loglog(epsilons, errors, 'bo-', linewidth=2, markersize=8, label='FD relative error')
ax1.axhline(y=1e-15, color='green', linestyle='--', alpha=0.7, label='Machine precision')

# Mark optimal region
min_error_idx = np.argmin(errors)
ax1.axvline(x=epsilons[min_error_idx], color='red', linestyle=':', alpha=0.7)
ax1.annotate(f'Optimal: ε={epsilons[min_error_idx]:.0e}',
            xy=(epsilons[min_error_idx], errors[min_error_idx]),
            xytext=(epsilons[min_error_idx]*10, errors[min_error_idx]*10),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red')

ax1.set_xlabel('Finite difference ε', fontsize=12)
ax1.set_ylabel('Relative error ‖Jv_FD - Jv_AD‖ / ‖Jv_AD‖', fontsize=12)
ax1.set_title('FD-JVP Error vs Perturbation Size\n(U-shaped: truncation ↔ roundoff)', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Annotate regions
ax1.text(1e-3, 1e-2, 'Truncation\ndominated', fontsize=9, ha='center')
ax1.text(1e-11, 1e-2, 'Roundoff\ndominated', fontsize=9, ha='center')

# Right: GMRES iterations comparison
methods = ['AD'] + [f'FD ε={eps:.0e}' for eps in test_epsilons]
iters = [results['gmres_comparison']['ad']['iterations']]
for eps in test_epsilons:
    key = f'fd_eps_{eps}'
    iters.append(results['gmres_comparison'][key]['iterations'])

colors = ['green'] + ['blue' if results['gmres_comparison'][f'fd_eps_{eps}']['converged']
                       else 'red' for eps in test_epsilons]
bars = ax2.bar(range(len(methods)), iters, color=colors)

ax2.set_xticks(range(len(methods)))
ax2.set_xticklabels(methods, rotation=45, ha='right')
ax2.set_ylabel('GMRES Iterations', fontsize=12)
ax2.set_title('GMRES Convergence: AD vs FD JVP', fontsize=14)
ax2.grid(True, alpha=0.3, axis='y')

for bar, it in zip(bars, iters):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            str(it), ha='center', fontsize=10)

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_jvp_vs_fd_sweep.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
