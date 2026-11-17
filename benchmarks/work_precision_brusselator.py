#!/usr/bin/env python3
"""
Work-Precision Diagram: Brusselator System

Plots Error vs. NFE (Nonlinear Function Evaluations) for each method.
This is the implementation-independent "gold standard" for algorithmic comparison.

The solvers are the moljax library solvers. The sweep engine is run_wp_system() in
generate_all_wp_nfe.py. This file has no hand-written stepper.

Strategy:
- Reference: ETDRK4 at dt=0.001 (50,000 steps).
- Second reference: RK4 at 0.9 of its stability limit (about 515,000 steps).
  The relative L2 difference of the two is stored as `reference_cross_check`.
- RK4: single point at stability limit (about 2M NFE)
- IMEX/ETDRK4: sweep over dt to show convergence curves

The error is the relative L2 error. A point has `above_10x_cross_check` =
true if its error is above 10 times the cross-check value.
NFE is counted in the run: 4 for each RK4 step, 2 for each IMEX-Strang
step, 4 for each ETDRK4 step.

Environment variables:
  RD_OUT_DIR   Output folder for JSON, CSV and figures (default: results and figures folders).
  RD_QUICK=1   Short test (t=0.5, fewer dt values). The backend check is off.
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

from moljax.core.model import create_brusselator_periodic_fft

parser = add_benchmark_args()
args = parser.parse_args()
EXPECTED_BACKEND = args.backend if args.backend != 'any' else None
if QUICK:
    EXPECTED_BACKEND = None  # The quick test runs on any backend.

print("=" * 70)
print("Work-Precision Diagram: Brusselator")
print("=" * 70)
device_str = setup_benchmark(expected_backend=EXPECTED_BACKEND)

# Configuration
N = 256
Du = 0.01
Dv = 0.02
a = 1.0
b = 3.4
L = 1.0
T_FINAL = 0.5 if QUICK else 50.0

# Initial condition: perturbed steady state
np.random.seed(42)
u_ss, v_ss = a, b / a
u0 = u_ss + 0.01 * np.random.randn(N, N)
v0 = v_ss + 0.01 * np.random.randn(N, N)


def reaction(u, v):
    u2v = u * u * v
    return a - (b + 1.0) * u + u2v, b * u - u2v


system = make_2d_system(lambda g: create_brusselator_periodic_fft(g, Du=Du, Dv=Dv, a=a, b=b),
                        reaction, u0, v0, L, T_FINAL)

# Stability limit
lambda_max = np.max(-system['lap_eig'])
dt_rk4_max = 0.9 * 2.8 / (max(Du, Dv) * lambda_max)
print(f"RK4 stability limit: dt_max = {dt_rk4_max:.6e}")
print(f"Steps for T={T_FINAL}: {int(np.ceil(T_FINAL / dt_rk4_max)):,}")

# RK4: single point at the stability limit.
sweeps = {
    'RK4': [dt_rk4_max],
    'IMEX-Strang': pick([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
    'ETDRK4': pick([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]),
}
results = run_wp_system(system, sweeps, ref_dt=0.001, dt_rk4_stable=dt_rk4_max)


# =============================================================================
# Generate Figure
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

styles = {
    'RK4':         {'color': '#d62728', 'marker': 's', 'ms': 12, 'label': 'RK4 (explicit, stability-limited)'},
    'IMEX-Strang': {'color': '#1f77b4', 'marker': 'o', 'ms': 8, 'label': 'IMEX-Strang (2 NFE/step)'},
    'ETDRK4':      {'color': '#ff7f0e', 'marker': 'D', 'ms': 8, 'label': 'ETDRK4 (4 NFE/step)'},
}

for method, data in method_items(results):
    pts = [d for d in data if d['error'] is not None and d['error'] > 0]
    if not pts:
        continue
    nfes = [d['nfe'] for d in pts]
    errs = [d['error'] for d in pts]
    s = styles[method]
    ax.plot(nfes, errs, marker=s['marker'], color=s['color'], lw=2, ms=s['ms'],
            label=s['label'], markeredgecolor='black', markeredgewidth=0.5)

if results.get('reference_cross_check'):
    ax.axhline(10 * results['reference_cross_check'], color='gray', ls='--', lw=1,
               label='10 x reference cross-check')

# Add annotation for the key result
rk4_ok = [d for d in results['RK4'] if d['error'] is not None and d['error'] > 0]
if rk4_ok:
    rk4_pt = rk4_ok[0]
    ax.annotate(f'RK4: {rk4_pt["nfe"]:,} NFE\n(stability-limited)',
                xy=(rk4_pt['nfe'], rk4_pt['error']),
                xytext=(rk4_pt['nfe']/5, rk4_pt['error']*3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')

ax.set_xlabel('Nonlinear Function Evaluations (NFE)', fontsize=13)
ax.set_ylabel(YLABEL_REL_L2, fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'Work-Precision Diagram: Brusselator ($256^2$, $t = {int(T_FINAL)}$)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig_work_precision_brusselator.pdf", dpi=300, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig_work_precision_brusselator.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {OUTPUT_DIR / 'fig_work_precision_brusselator.pdf'}")

# Save data (JSON and CSV)
write_json_csv(results, 'brusselator', 'work_precision_brusselator')
print("\nDone!")
