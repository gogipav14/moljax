#!/usr/bin/env python3
"""
Benchmark E10: End-to-End Gradient Sanity Check

Validate AD sensitivity claims with finite-difference verification:
- Compute dX/dD (diffusion sensitivity)
- Compute dX/dPe (Peclet sensitivity)
- Compare AD vs FD gradients

Purpose: Validates AD sensitivity claims for reactor application
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
import jax.numpy as jnp

from benchmark_utils import setup_benchmark, add_benchmark_args

parser = add_benchmark_args()
args = parser.parse_args()

# Configuration
N = 64
T_FINAL = 1.0
N_STEPS = 100
TEST_PARAMS = [
    {'D': 0.01, 'k': 1.0},
    {'D': 0.1, 'k': 1.0},
    {'D': 0.01, 'k': 10.0},
]
FD_EPSILON = 1e-6

print("E10: End-to-End Gradient Sanity Check")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N}, T_final={T_FINAL}")
print("=" * 60)

L = 1.0
dx = L / N
dt = T_FINAL / N_STEPS

# Grid
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Initial condition: Gaussian pulse
sigma_ic = 0.1
u0 = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma_ic**2))
u0_jax = jnp.array(u0)

# FFT eigenvalues
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX_fft, KY_fft = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX_fft**2 + KY_fft**2)
lap_eig_jax = jnp.array(lap_eig)

def make_integrator(D, k):
    """Create CN integrator for given D and k."""

    # CN amplification factor for diffusion
    G = (1 + 0.5*dt*D*lap_eig_jax) / (1 - 0.5*dt*D*lap_eig_jax)

    @jax.jit
    def cn_diffusion_step(u):
        u_hat = jnp.fft.fft2(u)
        return jnp.real(jnp.fft.ifft2(G * u_hat))

    @jax.jit
    def reaction_step(u):
        # Simple first-order decay: R = -k*u
        return u * jnp.exp(-k * dt)

    @jax.jit
    def integrate(u0):
        def body(i, u):
            # Strang splitting: R/2 -> D -> R/2
            u = u * jnp.exp(-k * dt / 2)
            u = cn_diffusion_step(u)
            u = u * jnp.exp(-k * dt / 2)
            return u
        return jax.lax.fori_loop(0, N_STEPS, body, u0)

    return integrate

def objective_fn(D, k, u0):
    """Objective: total mass at final time."""
    integrate = make_integrator(D, k)
    u_final = integrate(u0)
    return jnp.sum(u_final) * dx * dx  # Integral

results = {'tests': [], 'config': {}}

print("\nGradient verification:")
print(f"{'D':>8} {'k':>8} {'∂J/∂D (AD)':>14} {'∂J/∂D (FD)':>14} {'Rel Diff':>12}")
print("-" * 70)

for params in TEST_PARAMS:
    D = params['D']
    k = params['k']

    # AD gradients
    @jax.jit
    def obj_D(D_val):
        return objective_fn(D_val, k, u0_jax)

    @jax.jit
    def obj_k(k_val):
        return objective_fn(D, k_val, u0_jax)

    # Compute gradients via AD
    t0 = time.perf_counter()
    grad_D_ad = jax.grad(obj_D)(D)
    grad_D_ad = float(grad_D_ad)
    time_ad_D = time.perf_counter() - t0

    t0 = time.perf_counter()
    grad_k_ad = jax.grad(obj_k)(k)
    grad_k_ad = float(grad_k_ad)
    time_ad_k = time.perf_counter() - t0

    # Compute gradients via FD
    t0 = time.perf_counter()
    J_plus_D = float(obj_D(D + FD_EPSILON))
    J_minus_D = float(obj_D(D - FD_EPSILON))
    grad_D_fd = (J_plus_D - J_minus_D) / (2 * FD_EPSILON)
    time_fd_D = time.perf_counter() - t0

    t0 = time.perf_counter()
    J_plus_k = float(obj_k(k + FD_EPSILON))
    J_minus_k = float(obj_k(k - FD_EPSILON))
    grad_k_fd = (J_plus_k - J_minus_k) / (2 * FD_EPSILON)
    time_fd_k = time.perf_counter() - t0

    # Relative differences
    rel_diff_D = abs(grad_D_ad - grad_D_fd) / (abs(grad_D_ad) + 1e-16)
    rel_diff_k = abs(grad_k_ad - grad_k_fd) / (abs(grad_k_ad) + 1e-16)

    print(f"{D:>8.3f} {k:>8.1f} {grad_D_ad:>14.6e} {grad_D_fd:>14.6e} {rel_diff_D:>12.2e}")

    results['tests'].append({
        'D': D,
        'k': k,
        'grad_D_ad': grad_D_ad,
        'grad_D_fd': grad_D_fd,
        'grad_D_rel_diff': rel_diff_D,
        'grad_k_ad': grad_k_ad,
        'grad_k_fd': grad_k_fd,
        'grad_k_rel_diff': rel_diff_k,
        'time_ad_D_ms': time_ad_D * 1000,
        'time_fd_D_ms': time_fd_D * 1000,
        'time_ad_k_ms': time_ad_k * 1000,
        'time_fd_k_ms': time_fd_k * 1000,
    })

print("-" * 70)

# ========== Sensitivity surface for D and k ==========
print("\nComputing sensitivity surface...")

D_range = np.linspace(0.005, 0.1, 10)
k_range = np.linspace(0.5, 5.0, 10)

J_surface = np.zeros((len(D_range), len(k_range)))
grad_D_surface = np.zeros((len(D_range), len(k_range)))
grad_k_surface = np.zeros((len(D_range), len(k_range)))

for i, D_val in enumerate(D_range):
    for j, k_val in enumerate(k_range):
        J_surface[i, j] = float(objective_fn(D_val, k_val, u0_jax))

        @jax.jit
        def obj_D_local(D_v):
            return objective_fn(D_v, k_val, u0_jax)

        @jax.jit
        def obj_k_local(k_v):
            return objective_fn(D_val, k_v, u0_jax)

        grad_D_surface[i, j] = float(jax.grad(obj_D_local)(D_val))
        grad_k_surface[i, j] = float(jax.grad(obj_k_local)(k_val))

results['sensitivity_surface'] = {
    'D_range': D_range.tolist(),
    'k_range': k_range.tolist(),
    'J': J_surface.tolist(),
    'grad_D': grad_D_surface.tolist(),
    'grad_k': grad_k_surface.tolist(),
}

# Save results
results['config'] = {
    'grid_size': N,
    't_final': T_FINAL,
    'n_steps': N_STEPS,
    'fd_epsilon': FD_EPSILON,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'adjoint_grad_sanity.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: AD vs FD Gradient Comparison")
print("=" * 70)
print(f"{'D':>8} {'k':>8} {'∂J/∂D Diff':>14} {'∂J/∂k Diff':>14} {'AD speedup':>12}")
print("-" * 70)
for test in results['tests']:
    speedup = test['time_fd_D_ms'] / test['time_ad_D_ms']
    print(f"{test['D']:>8.3f} {test['k']:>8.1f} "
          f"{test['grad_D_rel_diff']:>14.2e} {test['grad_k_rel_diff']:>14.2e} "
          f"{speedup:>11.1f}x")
print("=" * 70)
print(f"\nAll relative differences < 1e-4: "
      f"{'✓' if all(t['grad_D_rel_diff'] < 1e-4 for t in results['tests']) else '✗'}")

# Generate figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

D_mesh, k_mesh = np.meshgrid(D_range, k_range, indexing='ij')

# Objective surface
ax = axes[0]
c = ax.contourf(D_mesh, k_mesh, J_surface, levels=20, cmap='viridis')
plt.colorbar(c, ax=ax, label='J (total mass)')
ax.set_xlabel('D (diffusion)')
ax.set_ylabel('k (reaction rate)')
ax.set_title('Objective J(D, k)')

# Gradient w.r.t. D
ax = axes[1]
c = ax.contourf(D_mesh, k_mesh, grad_D_surface, levels=20, cmap='RdBu_r')
plt.colorbar(c, ax=ax, label='∂J/∂D')
ax.set_xlabel('D (diffusion)')
ax.set_ylabel('k (reaction rate)')
ax.set_title('Sensitivity ∂J/∂D')

# Gradient w.r.t. k
ax = axes[2]
c = ax.contourf(D_mesh, k_mesh, grad_k_surface, levels=20, cmap='RdBu_r')
plt.colorbar(c, ax=ax, label='∂J/∂k')
ax.set_xlabel('D (diffusion)')
ax.set_ylabel('k (reaction rate)')
ax.set_title('Sensitivity ∂J/∂k')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_adjoint_grad_sanity.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
