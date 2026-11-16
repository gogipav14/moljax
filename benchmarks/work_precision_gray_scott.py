#!/usr/bin/env python3
"""
Work-Precision Diagram: Gray-Scott System

Plots Error vs. NFE (Nonlinear Function Evaluations) for each method.
This is the implementation-independent "gold standard" for algorithmic comparison.

The solvers are the moljax library solvers. The sweep engine is run_wp_system() in
generate_all_wp_nfe.py. This file has no hand-written stepper.

NFE counting (counted in the run):
  - RK4: 4 NFE per step (4 RK stages, each evaluates reaction)
  - CN+FFT: the counted value of CNStats.nfe() (residuals plus Jacobian products)
  - IMEX-Strang: 2 NFE per step (Heun reaction step)
  - ETDRK4: 4 NFE per step (4 nonlinear evaluations at different stages)

Reference solution: ETDRK4 with dt=0.01 (10,000 steps for t=100).
Second reference: RK4 at dt=0.01. The relative L2 difference of the two is
stored as `reference_cross_check`. The error is the relative L2 error.
A point has `above_10x_cross_check` = true if its error is above 10 times
the cross-check value.

A CN run with a failed Newton step has no error value and `valid` = false.

Environment variables:
  RD_OUT_DIR   Output folder for JSON, CSV and figures (default: results and figures folders).
  RD_QUICK=1   Short test (t=10, fewer dt values). The backend check is off.
"""

import jax

jax.config.update("jax_enable_x64", True)

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_utils import add_benchmark_args, setup_benchmark
from generate_all_wp_nfe import (
    OUTPUT_DIR,
    QUICK,
    YLABEL_REL_L2,
    make_2d_system,
    method_items,
    pick,
    run_wp_system,
    write_json_csv,
)

from moljax.core.model import create_gray_scott_periodic_fft

parser = add_benchmark_args()
args = parser.parse_args()
EXPECTED_BACKEND = args.backend if args.backend != 'any' else None
if QUICK:
    EXPECTED_BACKEND = None  # The quick test runs on any backend.

print("=" * 70)
print("Work-Precision Diagram: Gray-Scott")
print("=" * 70)
device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)

# Configuration
N = 256
Du = 2e-5
Dv = 1e-5
F = 0.035
k = 0.065
L = 1.0
T_FINAL = 10.0 if QUICK else 100.0  # Early transient: pattern formation actively underway

# Initial condition (same as main benchmark)
np.random.seed(42)
u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1
cx, cy = N//2, N//2
r = 10
u0[cx-r:cx+r, cy-r:cy+r] = 0.5
v0[cx-r:cx+r, cy-r:cy+r] = 0.25


def reaction(u, v):
    uvv = u * v * v
    return -uvv + F * (1 - u), uvv - (F + k) * v


system = make_2d_system(lambda g: create_gray_scott_periodic_fft(g, Du=Du, Dv=Dv, F=F, k=k),
                        reaction, u0, v0, L, T_FINAL)

# Stability limit for RK4
lambda_max = np.max(-system['lap_eig'])
dt_rk4_max = 0.9 * 2.8 / (max(Du, Dv) * lambda_max)

# RK4 has a MAXIMUM dt from stability. We sweep from that max down.
sweeps = {
    'RK4': [frac * dt_rk4_max for frac in pick([0.5, 0.6, 0.7, 0.8, 0.9])],
    'IMEX-Strang': pick([0.1, 0.2, 0.5, 1.0, 2.0, 5.0]),
    'ETDRK4': pick([0.2, 0.5, 1.0, 2.0, 5.0]),
    'CN+FFT': pick([0.1, 0.2, 0.5, 1.0, 2.0]),
}

print(f"\nRK4 stability limit (0.9 of the limit): dt_max = {dt_rk4_max:.4f}")
print(f"T_FINAL = {T_FINAL}")
results = run_wp_system(system, sweeps, ref_dt=0.01, dt_rk4_stable=dt_rk4_max)


# =============================================================================
# Generate Figure: Work-Precision Diagram
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

styles = {
    'RK4':         {'color': '#d62728', 'marker': 's', 'label': 'RK4 (explicit)'},
    'CN+FFT':      {'color': '#2ca02c', 'marker': '^', 'label': 'CN + FFT precond'},
    'IMEX-Strang': {'color': '#1f77b4', 'marker': 'o', 'label': 'IMEX-Strang'},
    'ETDRK4':      {'color': '#ff7f0e', 'marker': 'D', 'label': 'ETDRK4'},
}

for method, data in method_items(results):
    pts = [d for d in data if d['error'] is not None and d['error'] > 0]
    if not pts:
        continue
    nfes = [d['nfe'] for d in pts]
    errs = [d['error'] for d in pts]
    s = styles[method]
    ax.plot(nfes, errs, marker=s['marker'], color=s['color'], lw=2, ms=8,
            label=s['label'], markeredgecolor='black', markeredgewidth=0.5)

if results.get('reference_cross_check'):
    ax.axhline(10 * results['reference_cross_check'], color='gray', ls='--', lw=1,
               label='10 x reference cross-check')

ax.set_xlabel('Nonlinear Function Evaluations (NFE)', fontsize=13)
ax.set_ylabel(YLABEL_REL_L2, fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'Work-Precision Diagram: Gray-Scott ($256^2$, $t = {int(T_FINAL)}$)', fontsize=14)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_work_precision_gray_scott.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig_work_precision_gray_scott.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'fig_work_precision_gray_scott.pdf'}")

# Save data (JSON and CSV)
write_json_csv(results, 'gray_scott', 'work_precision_gray_scott')

print("\nDone!")
