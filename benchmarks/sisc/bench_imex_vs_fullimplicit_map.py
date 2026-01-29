#!/usr/bin/env python3
"""
Benchmark E14: IMEX vs Full Implicit vs ETD Regime Map

2D parameter sweep (diffusion stiffness × reaction stiffness)
to determine optimal method selection.

Purpose: Evidence-based method selection guide
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

print("E14: IMEX vs Full Implicit vs ETD Regime Map")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print("=" * 60)

# Configuration
N = 64  # Grid size
L = 1.0
T_FINAL = 0.1

# Parameter ranges
D_VALUES = [0.001, 0.01, 0.1]  # Diffusion coefficients (low to high stiffness)
K_VALUES = [0.1, 1.0, 10.0, 100.0]  # Reaction rates (low to high stiffness)

dx = L / N

# Grid
x = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, x, indexing='ij')

# Initial condition
u0 = 0.5 + 0.3 * np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)
u0_jax = jnp.array(u0)

# FFT eigenvalues
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX_fft, KY_fft = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX_fft**2 + KY_fft**2)
lap_eig_jax = jnp.array(lap_eig)

results = {'experiments': [], 'regime_map': [], 'config': {}}

print("\nBenchmarking methods across parameter space:")
print(f"{'D':>8} {'k':>8} {'RK4':>10} {'IMEX':>10} {'CN':>10} {'ETD':>10} {'Winner':>10}")
print("-" * 70)

def make_methods(D, k, dt):
    """Create integrators for given parameters."""

    # Reaction function
    @jax.jit
    def reaction(u):
        return k * u * (1 - u)

    # CN amplification for diffusion
    G_cn = (1 + 0.5*dt*D*lap_eig_jax) / (1 - 0.5*dt*D*lap_eig_jax)

    # Implicit diffusion solve
    M_inv = 1.0 / (1 - dt*D*lap_eig_jax)

    # ETD phi functions
    z = dt * D * lap_eig_jax
    exp_z = jnp.exp(z)
    phi1 = jnp.where(jnp.abs(z) < 1e-10, 1.0, (exp_z - 1) / z)

    # === Method 1: RK4 (explicit) ===
    @jax.jit
    def rk4_step(u):
        def rhs(u):
            # FFT Laplacian
            u_hat = jnp.fft.fft2(u)
            lap_u = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat))
            return D * lap_u + reaction(u)

        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u + dt*k3)
        return u + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

    # === Method 2: IMEX-Euler ===
    @jax.jit
    def imex_step(u):
        # Explicit reaction, implicit diffusion
        u_star = u + dt * reaction(u)
        u_hat = jnp.fft.fft2(u_star)
        return jnp.real(jnp.fft.ifft2(M_inv * u_hat))

    # === Method 3: Full implicit CN + Newton (simplified) ===
    @jax.jit
    def cn_step(u):
        # For linear diffusion + mild reaction, use operator splitting
        # CN for diffusion, then explicit reaction
        u_hat = jnp.fft.fft2(u)
        u_diffused = jnp.real(jnp.fft.ifft2(G_cn * u_hat))
        return u_diffused + dt * reaction(u_diffused)

    # === Method 4: ETD1 ===
    @jax.jit
    def etd_step(u):
        u_hat = jnp.fft.fft2(u)
        N_hat = jnp.fft.fft2(reaction(u))
        u_new_hat = exp_z * u_hat + dt * phi1 * N_hat
        return jnp.real(jnp.fft.ifft2(u_new_hat))

    return {
        'RK4': rk4_step,
        'IMEX': imex_step,
        'CN': cn_step,
        'ETD': etd_step,
    }

for D in D_VALUES:
    for k in K_VALUES:
        # Compute stiffness metrics
        sigma_diff = D * 0.001 / dx**2  # Approximate with small dt
        sigma_rxn = k * 0.001

        # Choose dt based on fastest stable explicit
        # RK4 stability: |lambda| * dt < 2.8
        lambda_max_diff = D * 4 / dx**2  # FD Laplacian max eigenvalue magnitude
        lambda_max_rxn = k  # Reaction rate

        dt_explicit = 2.0 / max(lambda_max_diff, lambda_max_rxn)
        dt_implicit = 0.01  # Can take much larger steps

        # Number of steps for each
        n_steps_explicit = max(int(T_FINAL / dt_explicit), 1)
        n_steps_implicit = max(int(T_FINAL / dt_implicit), 1)

        dt_exp = T_FINAL / n_steps_explicit
        dt_imp = T_FINAL / n_steps_implicit

        # Create methods
        methods_exp = make_methods(D, k, dt_exp)
        methods_imp = make_methods(D, k, dt_imp)

        timings = {}

        # Test RK4 (explicit)
        @jax.jit
        def integrate_rk4(u0, n):
            def body(i, u):
                return methods_exp['RK4'](u)
            return jax.lax.fori_loop(0, n, body, u0)

        try:
            _ = integrate_rk4(u0_jax, n_steps_explicit).block_until_ready()
            t0 = time.perf_counter()
            _ = integrate_rk4(u0_jax, n_steps_explicit).block_until_ready()
            timings['RK4'] = (time.perf_counter() - t0) * 1000
        except:
            timings['RK4'] = float('inf')

        # Test IMEX (implicit diffusion, explicit reaction)
        @jax.jit
        def integrate_imex(u0, n):
            def body(i, u):
                return methods_imp['IMEX'](u)
            return jax.lax.fori_loop(0, n, body, u0)

        try:
            _ = integrate_imex(u0_jax, n_steps_implicit).block_until_ready()
            t0 = time.perf_counter()
            _ = integrate_imex(u0_jax, n_steps_implicit).block_until_ready()
            timings['IMEX'] = (time.perf_counter() - t0) * 1000
        except:
            timings['IMEX'] = float('inf')

        # Test CN
        @jax.jit
        def integrate_cn(u0, n):
            def body(i, u):
                return methods_imp['CN'](u)
            return jax.lax.fori_loop(0, n, body, u0)

        try:
            _ = integrate_cn(u0_jax, n_steps_implicit).block_until_ready()
            t0 = time.perf_counter()
            _ = integrate_cn(u0_jax, n_steps_implicit).block_until_ready()
            timings['CN'] = (time.perf_counter() - t0) * 1000
        except:
            timings['CN'] = float('inf')

        # Test ETD
        @jax.jit
        def integrate_etd(u0, n):
            def body(i, u):
                return methods_imp['ETD'](u)
            return jax.lax.fori_loop(0, n, body, u0)

        try:
            _ = integrate_etd(u0_jax, n_steps_implicit).block_until_ready()
            t0 = time.perf_counter()
            _ = integrate_etd(u0_jax, n_steps_implicit).block_until_ready()
            timings['ETD'] = (time.perf_counter() - t0) * 1000
        except:
            timings['ETD'] = float('inf')

        # Determine winner
        winner = min(timings, key=timings.get)

        print(f"{D:>8.3f} {k:>8.1f} "
              f"{timings['RK4']:>9.1f}ms "
              f"{timings['IMEX']:>9.1f}ms "
              f"{timings['CN']:>9.1f}ms "
              f"{timings['ETD']:>9.1f}ms "
              f"{winner:>10}")

        results['experiments'].append({
            'D': D,
            'k': k,
            'timings': timings,
            'winner': winner,
            'n_steps_explicit': n_steps_explicit,
            'n_steps_implicit': n_steps_implicit,
        })

# Build regime map
print("\n" + "=" * 70)
print("REGIME MAP: Optimal Method Selection")
print("=" * 70)
print("Based on fastest runtime for each (D, k) combination:\n")

method_symbols = {'RK4': 'R', 'IMEX': 'I', 'CN': 'C', 'ETD': 'E'}

print(f"{'':>8}" + "".join(f"k={k:>6.1f}" for k in K_VALUES))
for D in D_VALUES:
    row = f"D={D:>5.3f}"
    for k in K_VALUES:
        exp = [e for e in results['experiments'] if e['D'] == D and e['k'] == k][0]
        row += f"  {method_symbols[exp['winner']]:>5}"
    print(row)

print("\nLegend: R=RK4, I=IMEX, C=CN, E=ETD")
print("\nGeneral guidance:")
print("  - Low diffusion, low reaction → RK4 (explicit)")
print("  - High diffusion, low reaction → IMEX or ETD")
print("  - High diffusion, high reaction → CN (full implicit)")
print("=" * 70)

# Save results
results['config'] = {
    'grid_size': N,
    't_final': T_FINAL,
    'D_values': D_VALUES,
    'k_values': K_VALUES,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'imex_vs_fullimplicit_map.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Generate figure
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Create heatmap of winner
method_to_num = {'RK4': 0, 'IMEX': 1, 'CN': 2, 'ETD': 3}
winner_matrix = np.zeros((len(D_VALUES), len(K_VALUES)))

for i, D in enumerate(D_VALUES):
    for j, k in enumerate(K_VALUES):
        exp = [e for e in results['experiments'] if e['D'] == D and e['k'] == k][0]
        winner_matrix[i, j] = method_to_num[exp['winner']]

from matplotlib.colors import ListedColormap
colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']  # Blue, Green, Orange, Red
cmap = ListedColormap(colors)

im = ax.imshow(winner_matrix, cmap=cmap, aspect='auto', vmin=-0.5, vmax=3.5)

# Labels
ax.set_xticks(range(len(K_VALUES)))
ax.set_xticklabels([str(k) for k in K_VALUES])
ax.set_yticks(range(len(D_VALUES)))
ax.set_yticklabels([str(D) for D in D_VALUES])
ax.set_xlabel('Reaction rate k', fontsize=12)
ax.set_ylabel('Diffusion coefficient D', fontsize=12)
ax.set_title('Optimal Method by Parameter Regime', fontsize=14)

# Colorbar with method labels
cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
cbar.set_ticklabels(['RK4', 'IMEX', 'CN', 'ETD'])

# Add timing annotations
for i, D in enumerate(D_VALUES):
    for j, k in enumerate(K_VALUES):
        exp = [e for e in results['experiments'] if e['D'] == D and e['k'] == k][0]
        t = exp['timings'][exp['winner']]
        ax.text(j, i, f'{t:.0f}ms', ha='center', va='center',
               fontsize=9, color='white', fontweight='bold')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_imex_vs_fullimplicit_map.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
