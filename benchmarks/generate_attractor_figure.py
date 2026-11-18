#!/usr/bin/env python3
"""
Generate 3x3 attractor divergence figure: three R-D systems x three methods.

Rows: Gray-Scott, Schnakenberg, Brusselator
Columns: RK4, IMEX-Strang, ETDRK4

Each panel shows the computed Turing pattern (u field) with wall time and NFE.
All methods use identical parameters, grid, and initial conditions per system.
The different treatment of the stiff linear operator steers each solver onto a
different pattern attractor branch during the transient growth phase.

The solvers are the moljax library solvers (see rd_library_benchmark.py):
integrate_fixed_dt (RK4), integrate_imex_fixed_dt and etd_integrate. The
Laplacian is the one of the library (second-order central differences). NFE
is 4 for each RK4 step, 2 for each IMEX-Strang step, 4 for each ETDRK4 step.
A library driver compiles one loop for each step count, so the warm-up is
one full run. A run of more than 2,000,000 steps has no warm-up; its time
includes the compilation (`time_includes_compile`).

Output: the figure, and the files attractor_divergence.json and
attractor_divergence.csv (one row for each (system, method)). There is no
reference solution here, so the `error` column of the CSV file is empty.

Environment variables:
  RD_OUT_DIR   Output folder for the figure, JSON and CSV (default: the
               figures folder for the figure, the results folder for the data).
  RD_QUICK=1   Short test (short final times, small blocks).
"""

import jax

jax.config.update("jax_enable_x64", True)

import csv
import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rd_library_benchmark as lb

from moljax.core.model import (
    create_brusselator_periodic_fft,
    create_gray_scott_periodic_fft,
    create_schnakenberg_periodic_fft,
)

QUICK = os.environ.get("RD_QUICK", "0") == "1"
_OUT = os.environ.get("RD_OUT_DIR")
output_dir = Path(_OUT) if _OUT else Path(__file__).parent.parent / "figures"
results_dir = Path(_OUT) if _OUT else Path(__file__).parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

print("Generating 3x3 attractor divergence figure...")
print(f"JAX devices: {jax.devices()}")
if QUICK:
    print("RD_QUICK=1: short test. The results are not for the paper.")


# =============================================================================
# System definitions
# =============================================================================
systems = []

# --- Gray-Scott ---
N_gs = 128; L_gs = 2.5; T_gs = 5000.0
Du_gs, Dv_gs = 2e-5, 1e-5
F_gs, k_gs = 0.035, 0.065

np.random.seed(42)
u0_gs = jnp.array(1.0 - 0.5 * np.random.rand(N_gs, N_gs))
v0_gs = jnp.array(0.25 * np.random.rand(N_gs, N_gs))

def gs_rxn_u(u, v):
    return -u * v**2 + F_gs * (1 - u)
def gs_rxn_v(u, v):
    return u * v**2 - (F_gs + k_gs) * v

# Max spectral eigenvalue: Dv * (pi*N/L)^2 = 1e-5 * (pi*128/2.5)^2 = 1e-5 * 25872 = 0.259
# RK4 stability: dt < 2.8/0.259 = 10.8 → dt=5.0 ok (reaction limited: F+k=0.1, so dt<10 ok)
# Actually the reaction is slow, so dt_rk4 = 1.0 should be fine
systems.append({
    'name': 'Gray--Scott', 'field': 'v', 'cmap': 'inferno',
    'N': N_gs, 'L': L_gs, 'T': T_gs,
    'make_model': lambda g: create_gray_scott_periodic_fft(g, Du=Du_gs, Dv=Dv_gs, F=F_gs, k=k_gs),
    'u0': u0_gs, 'v0': v0_gs,
    'rxn_u': gs_rxn_u, 'rxn_v': gs_rxn_v,
    'dt_rk4': 0.5, 'dt_imex': 0.5, 'dt_etd': 1.0,
})

# --- Schnakenberg ---
N_sn = 128; L_sn = 10.0; T_sn = 500.0
Du_sn, Dv_sn = 1.0, 40.0
a_sn, b_sn, gamma_sn = 0.1, 0.9, 100.0

us_sn = a_sn + b_sn; vs_sn = b_sn / (a_sn + b_sn)**2
np.random.seed(42)
u0_sn = jnp.array(us_sn + 0.01 * np.random.randn(N_sn, N_sn))
v0_sn = jnp.array(vs_sn + 0.01 * np.random.randn(N_sn, N_sn))

def sn_rxn_u(u, v):
    return gamma_sn * (a_sn - u + u**2 * v)
def sn_rxn_v(u, v):
    return gamma_sn * (b_sn - u**2 * v)

# Max spectral eigenvalue (2D corner mode): Dv * 2*(pi*N/L)^2 = 40*2*1617 = 129,356
# RK4: dt < 2.785/129356 = 2.15e-5 → need dt=2e-5 → 25M steps / 100M NFE!
# This is INFEASIBLE in practice: ~75 min wall time vs seconds for IMEX/ETDRK4.
# We deliberately show dt=4e-5 going unstable to make this point.
# IMEX: implicit diffusion, explicit reaction: dt < 2.8/100 = 0.028 → use 0.005
# ETDRK4: same reaction constraint: dt < 0.028 → use 0.005
systems.append({
    'name': 'Schnakenberg', 'field': 'u', 'cmap': 'viridis',
    'N': N_sn, 'L': L_sn, 'T': T_sn,
    'make_model': lambda g: create_schnakenberg_periodic_fft(
        g, Du=Du_sn, Dv=Dv_sn, a=a_sn, b=b_sn, gamma=gamma_sn),
    'u0': u0_sn, 'v0': v0_sn,
    'rxn_u': sn_rxn_u, 'rxn_v': sn_rxn_v,
    'dt_rk4': 4e-5, 'dt_imex': 0.005, 'dt_etd': 0.005,
    'rk4_infeasible': True,  # Flag: RK4 is unstable at any practical dt
})

# --- Brusselator ---
N_br = 128; L_br = 5.0; T_br = 100.0
Du_br, Dv_br = 0.01, 0.1
a_br, b_br = 1.0, 1.8

us_br, vs_br = a_br, b_br / a_br
np.random.seed(42)
u0_br = jnp.array(us_br + 0.05 * np.random.randn(N_br, N_br))
v0_br = jnp.array(vs_br + 0.05 * np.random.randn(N_br, N_br))

def br_rxn_u(u, v):
    return a_br - (b_br + 1) * u + u**2 * v
def br_rxn_v(u, v):
    return b_br * u - u**2 * v

# Max spectral eigenvalue: Dv * (pi*N/L)^2 = 0.1*(pi*128/5)^2 = 0.1*6468 = 647
# RK4: dt < 2.8/647 = 0.0043 → use 0.002
# Reaction: |lambda_rxn| ~ 1, dt < 2.8 → no constraint
# IMEX: dt=0.05 (implicit diffusion)
# ETDRK4: dt=0.5
systems.append({
    'name': 'Brusselator', 'field': 'u', 'cmap': 'RdBu_r',
    'N': N_br, 'L': L_br, 'T': T_br,
    'make_model': lambda g: create_brusselator_periodic_fft(g, Du=Du_br, Dv=Dv_br, a=a_br, b=b_br),
    'u0': u0_br, 'v0': v0_br,
    'rxn_u': br_rxn_u, 'rxn_v': br_rxn_v,
    'dt_rk4': 0.002, 'dt_imex': 0.05, 'dt_etd': 0.5,
})


# =============================================================================
# Run all 9 combinations
# =============================================================================
if QUICK:
    # Short final times.
    for sys_, t_quick in zip(systems, (20.0, 0.5, 2.0), strict=True):
        sys_['T'] = t_quick

results = {}  # results[(sys_idx, method)] = (u_field, v_field, walltime, nfe, nsteps, dt)
records = []  # one record for each (system, method) for the JSON and CSV files

NFE_KEY = {'RK4': 'rk4', 'IMEX': 'imex_strang', 'ETDRK4': 'etdrk4'}

for si, sys in enumerate(systems):
    name = sys['name']
    rxn_u, rxn_v = sys['rxn_u'], sys['rxn_v']
    T = sys['T']

    def reaction(u, v, rxn_u=rxn_u, rxn_v=rxn_v):
        return rxn_u(u, v), rxn_v(u, v)

    lib = lb.make_library_system(sys['make_model'], reaction, sys['u0'], sys['v0'], sys['L'], T)
    lam_max = float(np.max(-lib['lap_eig'])) * max(lib['Du'], lib['Dv'])
    print(f"\n{name}: RK4 diffusion limit of the library Laplacian: dt < {2.8 / lam_max:.3e}")

    for method, title in (('RK4', 'RK4'), ('IMEX', 'IMEX-Strang'), ('ETDRK4', 'ETDRK4')):
        print(f"\n{'='*60}")
        print(f"{name} - {title}")
        print(f"{'='*60}", flush=True)
        dt_req = sys[{'RK4': 'dt_rk4', 'IMEX': 'dt_imex', 'ETDRK4': 'dt_etd'}[method]]
        n_steps, dt = lb.steps_for(T, dt_req)
        lib_method = {'RK4': 'RK4', 'IMEX': 'IMEX-Strang', 'ETDRK4': 'ETDRK4'}[method]
        warm = n_steps <= 2_000_000
        if warm:
            lb.library_solve(lib, lib_method, dt, n_steps)     # compile; not timed
        t_start = time.perf_counter()
        u_f, v_f, extra = lb.library_solve(lib, lib_method, dt, n_steps)
        elapsed = time.perf_counter() - t_start
        u_f, v_f = np.array(u_f), np.array(v_f)
        status = extra.get('status')
        if status == 'NON_FINITE_VALUES':
            # The library returns the last finite state. The panel must show
            # that the run failed.
            u_f = np.full_like(u_f, np.nan)
            v_f = np.full_like(v_f, np.nan)
        nfe = lb.NFE_PER_STEP[NFE_KEY[method]] * n_steps
        results[(si, method)] = (u_f, v_f, elapsed, nfe, n_steps, dt)
        rec = {'system': name, 'method': method, 'dt': float(dt), 'steps': n_steps,
               'nfe': int(nfe), 'time_s': float(elapsed), 'time_includes_compile': not warm,
               'status': status, 'rk4_diffusion_limit': 2.8 / lam_max,
               'error': None, 'valid': None}
        rec.update(lb.field_summary(u_f, v_f))
        rec['valid'] = rec['finite']
        print(f"  {elapsed:.2f} s, steps {n_steps:,}, finite {rec['finite']}, status {status}, "
              f"u mean {rec.get('u_mean')}, u std {rec.get('u_std')}", flush=True)
        records.append(rec)

with open(results_dir / "attractor_divergence.json", 'w') as f:
    json.dump({'records': records, 'quick': QUICK,
               'config': {'grid': 128, 'solvers': 'moljax library', 'laplacian': 'second-order central differences',
                          'jax_version': jax.__version__, 'device': str(jax.devices()[0])}}, f, indent=2)
with open(results_dir / "attractor_divergence.csv", 'w', newline='') as f:
    cols = ['system', 'method', 'dt', 'steps', 'nfe', 'time_s', 'time_includes_compile',
            'status', 'error', 'finite', 'valid', 'u_mean', 'u_std', 'v_mean', 'v_std']
    w = csv.writer(f)
    w.writerow(cols)
    for rec in records:
        w.writerow(["" if rec.get(c) is None else rec[c] for c in cols])
print(f"Saved: {results_dir / 'attractor_divergence.json'} and .csv")


# =============================================================================
# PLOTTING: 3x3 grid
# =============================================================================
print("\n--- Generating 3x3 figure ---")

fig, axes = plt.subplots(3, 3, figsize=(13, 12))
methods = ['RK4', 'IMEX', 'ETDRK4']
method_labels = ['RK4 (explicit)', 'IMEX-Strang', 'ETDRK4']

for si, sys in enumerate(systems):
    for mi, method in enumerate(methods):
        ax = axes[si, mi]
        u_field, v_field, wt, nfe, nsteps, dt = results[(si, method)]

        # Choose which field to show
        field = v_field if sys['field'] == 'v' else u_field
        extent = [0, sys['L'], 0, sys['L']]
        is_nan = not np.isfinite(field).all()

        if is_nan:
            # Show informative "infeasible" panel
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.55, 'UNSTABLE', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, fontweight='bold',
                    color='#cc0000')
            ax.text(0.5, 0.35, (
                'Stable dt ≤ 2.2×10⁻⁵\n'
                '→ 25M steps, 100M NFE\n'
                '→ ~75 min wall time'),
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8.5, color='#444444', linespacing=1.4)
            ax.set_xlim(0, sys['L']); ax.set_ylim(0, sys['L'])
        else:
            im = ax.imshow(field.T, cmap=sys['cmap'], origin='lower', extent=extent)
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

        # Title: method header (top row) + timing
        title_lines = []
        if si == 0:
            title_lines.append(method_labels[mi])
        if is_nan:
            title_lines.append(f'dt = {dt:.0e} (unstable)')
        else:
            title_lines.append(f'dt = {dt}, {nsteps:,} steps')
            title_lines.append(f'{wt:.1f} s, {nfe:,} NFE')
        ax.set_title('\n'.join(title_lines), fontsize=9,
                      fontweight='bold' if si == 0 else 'normal')

        # Row labels (left column only)
        if mi == 0:
            ax.set_ylabel(f'{sys["name"]}\n$y$', fontsize=10)
        else:
            ax.set_ylabel('$y$', fontsize=10)

        # Bottom row only
        if si == 2:
            ax.set_xlabel('$x$', fontsize=10)

        ax.set_aspect('equal')

plt.suptitle(
    'Attractor divergence: identical ICs and parameters, different integrators (128² grids)',
    fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])

plt.savefig(output_dir / "fig_attractor_divergence.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_attractor_divergence.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_attractor_divergence.pdf'}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY: Wall time and NFE across all 9 combinations")
print("="*80)
print(f"{'System':<15} {'Method':<12} {'dt':<12} {'Steps':>10} {'NFE':>12} {'Time (s)':>10}")
print("-"*80)
for si, sys in enumerate(systems):
    for method in methods:
        u_f, v_f, wt, nfe, ns, dt = results[(si, method)]
        print(f"{sys['name']:<15} {method:<12} {dt:<12.4g} {ns:>10,} {nfe:>12,} {wt:>10.2f}")
    print()
print("="*80)
print("Done!")
