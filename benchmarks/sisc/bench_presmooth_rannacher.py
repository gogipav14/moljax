#!/usr/bin/env python3
"""
Benchmark E8: Nonsmooth IC with Rannacher Startup

Compare Crank-Nicolson oscillations with Rannacher damping:
- Pure CN: Shows oscillations for discontinuous IC
- CN + Rannacher: 2 BE half-steps damp spurious modes
- Pure BE: No oscillations but only 1st order

Purpose: Addresses CN criticism for nonsmooth data
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
N = 256
D = 0.01
DT = 0.001
N_STEPS = 100
T_FINAL = DT * N_STEPS

print("E8: Nonsmooth IC - Rannacher Startup")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N}, D={D}, dt={DT}")
print("=" * 60)

L = 1.0
dx = L / N

# FFT eigenvalues
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Discontinuous initial condition: unit square in center
x = np.linspace(0, L, N, endpoint=False)
y = np.linspace(0, L, N, endpoint=False)
X, Y = np.meshgrid(x, y, indexing='ij')

# Sharp square pulse
u0 = np.zeros((N, N))
mask = (X > 0.3) & (X < 0.7) & (Y > 0.3) & (Y < 0.7)
u0[mask] = 1.0
u0_jax = jnp.array(u0)

# Amplification factors
G_cn = (1 + 0.5*DT*D*lap_eig_jax) / (1 - 0.5*DT*D*lap_eig_jax)
G_be = 1.0 / (1 - DT*D*lap_eig_jax)
G_be_half = 1.0 / (1 - 0.5*DT*D*lap_eig_jax)

@jax.jit
def cn_step(u):
    u_hat = jnp.fft.fft2(u)
    return jnp.real(jnp.fft.ifft2(G_cn * u_hat))

@jax.jit
def be_step(u):
    u_hat = jnp.fft.fft2(u)
    return jnp.real(jnp.fft.ifft2(G_be * u_hat))

@jax.jit
def be_half_step(u):
    u_hat = jnp.fft.fft2(u)
    return jnp.real(jnp.fft.ifft2(G_be_half * u_hat))

results = {'methods': {}, 'config': {}}

# ========== 1. Pure Crank-Nicolson ==========
print("\n1. Pure Crank-Nicolson:")

@jax.jit
def integrate_cn(u0, n_steps):
    def body(i, u):
        return cn_step(u)
    return jax.lax.fori_loop(0, n_steps, body, u0)

# Store solution history for oscillation analysis
u = u0_jax
solutions_cn = [np.array(u)]
for i in range(N_STEPS):
    u = cn_step(u)
    if i % 10 == 9:
        solutions_cn.append(np.array(u))

u_final_cn = u

# Measure oscillations: check for values outside [0, 1]
osc_min = float(jnp.min(u_final_cn))
osc_max = float(jnp.max(u_final_cn))
osc_amplitude = max(0 - osc_min, osc_max - 1)

print(f"  Final range: [{osc_min:.6f}, {osc_max:.6f}]")
print(f"  Oscillation amplitude: {osc_amplitude:.6f}")

results['methods']['pure_cn'] = {
    'min_value': osc_min,
    'max_value': osc_max,
    'oscillation_amplitude': osc_amplitude,
    'solutions_at_steps': [int(i*10+9) for i in range(len(solutions_cn)-1)] + [N_STEPS],
}

# ========== 2. CN with Rannacher Startup ==========
print("\n2. CN + Rannacher Startup (2 BE half-steps):")

@jax.jit
def integrate_cn_rannacher(u0, n_steps):
    # Rannacher: 2 BE half-steps, then CN
    u = be_half_step(u0)
    u = be_half_step(u)

    def body(i, u):
        return cn_step(u)
    return jax.lax.fori_loop(0, n_steps - 1, body, u)

u = u0_jax
# Rannacher startup
u = be_half_step(u)
u = be_half_step(u)

solutions_rannacher = [np.array(u)]
for i in range(N_STEPS - 1):
    u = cn_step(u)
    if i % 10 == 9:
        solutions_rannacher.append(np.array(u))

u_final_rannacher = u

osc_min_r = float(jnp.min(u_final_rannacher))
osc_max_r = float(jnp.max(u_final_rannacher))
osc_amplitude_r = max(0 - osc_min_r, osc_max_r - 1)

print(f"  Final range: [{osc_min_r:.6f}, {osc_max_r:.6f}]")
print(f"  Oscillation amplitude: {osc_amplitude_r:.6f}")

results['methods']['cn_rannacher'] = {
    'min_value': osc_min_r,
    'max_value': osc_max_r,
    'oscillation_amplitude': osc_amplitude_r,
    'rannacher_steps': 2,
}

# ========== 3. Pure Backward Euler ==========
print("\n3. Pure Backward Euler:")

@jax.jit
def integrate_be(u0, n_steps):
    def body(i, u):
        return be_step(u)
    return jax.lax.fori_loop(0, n_steps, body, u0)

u = u0_jax
solutions_be = [np.array(u)]
for i in range(N_STEPS):
    u = be_step(u)
    if i % 10 == 9:
        solutions_be.append(np.array(u))

u_final_be = u

osc_min_be = float(jnp.min(u_final_be))
osc_max_be = float(jnp.max(u_final_be))
osc_amplitude_be = max(0 - osc_min_be, osc_max_be - 1)

print(f"  Final range: [{osc_min_be:.6f}, {osc_max_be:.6f}]")
print(f"  Oscillation amplitude: {osc_amplitude_be:.6f}")

results['methods']['pure_be'] = {
    'min_value': osc_min_be,
    'max_value': osc_max_be,
    'oscillation_amplitude': osc_amplitude_be,
}

# ========== Compare solutions ==========
# For smooth IC reference, use Gaussian
sigma_ic = 0.1
u0_smooth = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma_ic**2))
u0_smooth_jax = jnp.array(u0_smooth)

print("\n4. Comparison with smooth IC (Gaussian):")
u_smooth_cn = integrate_cn(u0_smooth_jax, N_STEPS)
u_smooth_be = integrate_be(u0_smooth_jax, N_STEPS)

# For smooth IC, both should give similar results
diff_smooth = float(jnp.max(jnp.abs(u_smooth_cn - u_smooth_be)))
print(f"  Max |CN - BE| for smooth IC: {diff_smooth:.6e}")

results['smooth_ic_comparison'] = {
    'max_diff_cn_be': diff_smooth,
}

# Save results
results['config'] = {
    'grid_size': N,
    'D': D,
    'dt': DT,
    'n_steps': N_STEPS,
    't_final': T_FINAL,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'rannacher_startup.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY: Oscillation Amplitude for Discontinuous IC")
print("=" * 70)
print(f"{'Method':>20} {'Min':>12} {'Max':>12} {'Oscillation':>12}")
print("-" * 70)
for name, data in results['methods'].items():
    print(f"{name:>20} {data['min_value']:>12.6f} {data['max_value']:>12.6f} {data['oscillation_amplitude']:>12.6f}")
print("=" * 70)
print("\nNote: Values should be in [0,1] for a properly damped solution.")
print("Oscillation amplitude = max(|min-0|, |max-1|)")

# Generate figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot cross-section at y = 0.5
y_idx = N // 2

ax = axes[0]
ax.set_title('Initial Condition\n(Discontinuous)')
ax.plot(x, u0[:, y_idx], 'k-', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('u')
ax.set_ylim(-0.2, 1.2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.set_title(f'Pure CN (t={T_FINAL})\nOscillation: {results["methods"]["pure_cn"]["oscillation_amplitude"]:.4f}')
ax.plot(x, np.array(u_final_cn)[:, y_idx], 'b-', linewidth=2)
ax.set_xlabel('x')
ax.set_ylim(-0.2, 1.2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.set_title(f'CN + Rannacher (t={T_FINAL})\nOscillation: {results["methods"]["cn_rannacher"]["oscillation_amplitude"]:.4f}')
ax.plot(x, np.array(u_final_rannacher)[:, y_idx], 'g-', linewidth=2)
ax.set_xlabel('x')
ax.set_ylim(-0.2, 1.2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

ax = axes[3]
ax.set_title(f'Pure BE (t={T_FINAL})\nOscillation: {results["methods"]["pure_be"]["oscillation_amplitude"]:.4f}')
ax.plot(x, np.array(u_final_be)[:, y_idx], 'r-', linewidth=2)
ax.set_xlabel('x')
ax.set_ylim(-0.2, 1.2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'fig_rannacher_startup.pdf'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
