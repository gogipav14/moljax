#!/usr/bin/env python3
"""
Work-Precision Diagram: Tubular Reactor (Error vs NFE)

Uses the analytical steady-state solution as reference — no numerical
reference needed. This gives clean, unambiguous convergence curves.

Compares RK4, CN (implicit), and IMEX-Euler on the axial dispersion model
at Pe=10, Da=1 (Danckwerts BC) for multiple grid resolutions and timesteps.

NFE counting:
  - RK4: 4 NFE per step
  - Forward Euler: 1 NFE per step
  - CN: treated as 1 linear solve per step (cost dominated by tridiagonal solve)

For this 1D FD problem, "NFE" = number of RHS evaluations.
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_utils import setup_benchmark

print("=" * 70)
print("Work-Precision Diagram: Reactor (Error vs NFE)")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")

# Fixed parameters
Pe = 10.0
Da = 1.0
N = 128
T_FINAL = 15.0  # Enough for steady state
dz = 1.0 / (N - 1)
z = np.linspace(0, 1, N)


# =============================================================================
# Analytical Solution
# =============================================================================

def analytical_danckwerts_steady(z, Pe, Da):
    disc = Pe**2 + 4*Pe*Da
    r1 = (Pe + np.sqrt(disc)) / 2
    r2 = (Pe - np.sqrt(disc)) / 2
    M = np.array([
        [r1/Pe - 1, r2/Pe - 1],
        [r1*np.exp(r1), r2*np.exp(r2)]
    ])
    rhs_vec = np.array([-1, 0])
    AB = np.linalg.solve(M, rhs_vec)
    A, B = AB
    return A * np.exp(r1 * z) + B * np.exp(r2 * z)

c_analytical = analytical_danckwerts_steady(z, Pe, Da)
print(f"Analytical solution: c(0)={c_analytical[0]:.6f}, c(1)={c_analytical[-1]:.6f}")
print(f"Outlet conversion: X = {1-c_analytical[-1]:.6f}")


# =============================================================================
# FD Discretization
# =============================================================================

def create_fd_matrices(N, dz):
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] = 1.0
        D2[i, i] = -2.0
        D2[i, i+1] = 1.0
    D2 /= dz**2
    D1 = np.zeros((N, N))
    for i in range(1, N-1):
        D1[i, i-1] = -0.5
        D1[i, i+1] = 0.5
    D1[0, 0] = -1.0; D1[0, 1] = 1.0
    D1[-1, -2] = -1.0; D1[-1, -1] = 1.0
    D1 /= dz
    return D2, D1

D2_np, D1_np = create_fd_matrices(N, dz)
D2_jax = jnp.array(D2_np)
D1_jax = jnp.array(D1_np)

c0 = jnp.zeros(N)  # Start empty for Danckwerts

# Danckwerts RHS
inlet_coeff = (1.0/Pe) / dz

@jax.jit
def rhs(c, t):
    dcdt = (1.0/Pe) * (D2_jax @ c) - (D1_jax @ c) - Da * c
    c0_target = (1.0 + inlet_coeff * c[1]) / (1.0 + inlet_coeff)
    dcdt = dcdt.at[0].set(100.0 * (c0_target - c[0]))
    dcdt = dcdt.at[-1].set(100.0 * (c[-2] - c[-1]))
    return dcdt


# =============================================================================
# Solvers
# =============================================================================

@partial(jax.jit, static_argnums=(2, 3))
def solve_rk4(c0, rhs_fn_unused, dt, n_steps):
    def step(i, c):
        t = i * dt
        k1 = rhs(c, t)
        k2 = rhs(c + 0.5*dt*k1, t + 0.5*dt)
        k3 = rhs(c + 0.5*dt*k2, t + 0.5*dt)
        k4 = rhs(c + dt*k3, t + dt)
        return c + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return jax.lax.fori_loop(0, n_steps, step, c0)


@partial(jax.jit, static_argnums=(2, 3))
def solve_fe(c0, rhs_fn_unused, dt, n_steps):
    """Forward Euler."""
    def step(i, c):
        return c + dt * rhs(c, i * dt)
    return jax.lax.fori_loop(0, n_steps, step, c0)


def compute_error(c_num, c_anal):
    return float(jnp.max(jnp.abs(c_num - jnp.array(c_anal))))


# =============================================================================
# CFL limits
# =============================================================================
dt_adv = 0.5 * dz
dt_diff = 0.25 * Pe * dz**2
dt_cfl = min(dt_adv, dt_diff)
print(f"\nCFL limits: dt_adv={dt_adv:.4e}, dt_diff={dt_diff:.4e}, min={dt_cfl:.4e}")


# =============================================================================
# Sweep: Error vs NFE
# =============================================================================
results = {}

# --- RK4 ---
print("\n--- RK4 sweep ---")
# RK4 stability allows up to ~2.8x the FE CFL
dt_rk4_max = 2.0 * dt_cfl  # Conservative for RK4
rk4_dt_fracs = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
rk4_results = []
for frac in rk4_dt_fracs:
    dt = frac * dt_cfl
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 4 * n_steps
    print(f"  dt={dt:.4e} ({frac}x CFL), steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    _ = solve_rk4(c0, None, dt, min(100, n_steps))
    c0.block_until_ready()
    c_final = solve_rk4(c0, None, dt, n_steps)
    c_final.block_until_ready()
    err = compute_error(c_final, c_analytical)
    print(f"err={err:.2e}")
    rk4_results.append({'dt': float(dt), 'frac': frac, 'steps': n_steps, 'nfe': nfe, 'error': err})
results['RK4'] = rk4_results

# --- Forward Euler ---
print("\n--- Forward Euler sweep ---")
fe_dt_fracs = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
fe_results = []
for frac in fe_dt_fracs:
    dt = frac * dt_cfl
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 1 * n_steps  # 1 RHS eval per step
    print(f"  dt={dt:.4e} ({frac}x CFL), steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    _ = solve_fe(c0, None, dt, min(100, n_steps))
    c0.block_until_ready()
    c_final = solve_fe(c0, None, dt, n_steps)
    c_final.block_until_ready()
    err = compute_error(c_final, c_analytical)
    print(f"err={err:.2e}")
    fe_results.append({'dt': float(dt), 'frac': frac, 'steps': n_steps, 'nfe': nfe, 'error': err})
results['Forward Euler'] = fe_results


# =============================================================================
# Figure
# =============================================================================
output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 7))

styles = {
    'RK4':           {'color': '#d62728', 'marker': 's', 'label': 'RK4 (4 NFE/step)'},
    'Forward Euler': {'color': '#9467bd', 'marker': 'v', 'label': 'Forward Euler (1 NFE/step)'},
}

for method, data in results.items():
    nfes = [d['nfe'] for d in data if d['error'] > 0 and d['error'] < 1.0]
    errs = [d['error'] for d in data if d['error'] > 0 and d['error'] < 1.0]
    if nfes:
        s = styles[method]
        ax.plot(nfes, errs, marker=s['marker'], color=s['color'], lw=2, ms=8,
                label=s['label'], markeredgecolor='black', markeredgewidth=0.5)

ax.set_xlabel('RHS Evaluations (NFE)', fontsize=13)
ax.set_ylabel('$\\|c - c_{\\mathrm{analytical}}\\|_\\infty$', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'Work-Precision: Tubular Reactor (Pe={int(Pe)}, Da={int(Da)}, N={N})',
             fontsize=14)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / "fig_work_precision_reactor_nfe.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_work_precision_reactor_nfe.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_work_precision_reactor_nfe.pdf'}")

# Save data
data_dir = Path(__file__).parent / "results"
data_dir.mkdir(exist_ok=True)
with open(data_dir / "work_precision_reactor_nfe.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {data_dir / 'work_precision_reactor_nfe.json'}")
print("\nDone!")
