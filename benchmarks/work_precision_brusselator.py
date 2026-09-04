#!/usr/bin/env python3
"""
Work-Precision Diagram: Brusselator System

Plots Error vs. NFE (Nonlinear Function Evaluations) for each method.
This is the implementation-independent "gold standard" for algorithmic comparison.

Strategy:
- Reference: ETDRK4 at dt=0.001 (50,000 steps) — very accurate, fast (implicit)
- RK4: single point at stability limit (513K steps, ~2M NFE)
- IMEX/ETDRK4/CN: sweep over dt to show convergence curves

The key result: IMEX-Strang at 2,000 NFE achieves comparable accuracy to
RK4 at 2,053,380 NFE — a >1000x reduction in algorithmic work.
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from benchmark_utils import setup_benchmark

print("=" * 70)
print("Work-Precision Diagram: Brusselator")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")

# Configuration
N = 256
Du = 0.01
Dv = 0.02
a = 1.0
b = 3.4
L = 1.0
dx = L / N
T_FINAL = 50.0

# Wavenumbers
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Initial condition: perturbed steady state
np.random.seed(42)
u_ss, v_ss = a, b / a
u0 = u_ss + 0.01 * np.random.randn(N, N)
v0 = v_ss + 0.01 * np.random.randn(N, N)
u0_jax = jnp.array(u0)
v0_jax = jnp.array(v0)

# Stability limit
lambda_max = np.max(-lap_eig)
dt_rk4_max = 0.9 * 2.8 / (max(Du, Dv) * lambda_max)
print(f"RK4 stability limit: dt_max = {dt_rk4_max:.6e}")
print(f"Steps for T={T_FINAL}: {int(np.ceil(T_FINAL / dt_rk4_max)):,}")


# =============================================================================
# Solver Definitions
# =============================================================================

@jax.jit
def rk4_step(u, v, dt):
    def rhs(u, v):
        u_hat = jnp.fft.fft2(u)
        v_hat = jnp.fft.fft2(v)
        lap_u = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat))
        lap_v = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat))
        u2v = u * u * v
        return Du * lap_u + a - (b + 1.0) * u + u2v, Dv * lap_v + b * u - u2v

    k1u, k1v = rhs(u, v)
    k2u, k2v = rhs(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
    k3u, k3v = rhs(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
    k4u, k4v = rhs(u + dt*k3u, v + dt*k3v)
    return u + dt/6 * (k1u + 2*k2u + 2*k3u + k4u), v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)

def rk4_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = rk4_step(u, v, dt)
    return u, v


@jax.jit
def imex_step(u, v, dt):
    u2v = u * u * v
    u = u + 0.5*dt * (a - (b + 1.0) * u + u2v)
    v = v + 0.5*dt * (b * u - u2v)
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    u_hat = u_hat * jnp.exp(dt * Du * lap_eig_jax)
    v_hat = v_hat * jnp.exp(dt * Dv * lap_eig_jax)
    u = jnp.real(jnp.fft.ifft2(u_hat))
    v = jnp.real(jnp.fft.ifft2(v_hat))
    u2v = u * u * v
    u = u + 0.5*dt * (a - (b + 1.0) * u + u2v)
    v = v + 0.5*dt * (b * u - u2v)
    return u, v

def imex_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = imex_step(u, v, dt)
    return u, v


def make_etdrk4_solver(dt_etd):
    Lu = Du * lap_eig
    Lv = Dv * lap_eig

    def compute_coeffs(L_op, dt, M=32):
        r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
        LR = dt * L_op[..., np.newaxis] + r
        E = np.exp(dt * L_op)
        E2 = np.exp(dt * L_op / 2)
        Q = dt * np.mean((np.exp(LR/2) - 1) / LR, axis=-1)
        f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1)
        f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1)
        f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1)
        return E, E2, Q, f1, f2, f3

    Eu, E2u, Qu, f1u, f2u, f3u = [jnp.array(x) for x in compute_coeffs(Lu, dt_etd)]
    Ev, E2v, Qv, f1v, f2v, f3v = [jnp.array(x) for x in compute_coeffs(Lv, dt_etd)]

    @jax.jit
    def step(u, v):
        uh = jnp.fft.fft2(u)
        vh = jnp.fft.fft2(v)
        def Nu(u, v): return a - (b+1)*u + u*u*v
        def Nv(u, v): return b*u - u*u*v

        nu = jnp.fft.fft2(Nu(u, v)); nv = jnp.fft.fft2(Nv(u, v))
        au = E2u*uh + Qu*nu; av = E2v*vh + Qv*nv
        ar = jnp.real(jnp.fft.ifft2(au)); avr = jnp.real(jnp.fft.ifft2(av))
        nau = jnp.fft.fft2(Nu(ar, avr)); nav = jnp.fft.fft2(Nv(ar, avr))
        bu = E2u*uh + Qu*nau; bv = E2v*vh + Qv*nav
        br = jnp.real(jnp.fft.ifft2(bu)); bvr = jnp.real(jnp.fft.ifft2(bv))
        nbu = jnp.fft.fft2(Nu(br, bvr)); nbv = jnp.fft.fft2(Nv(br, bvr))
        cu = E2u*au + Qu*(2*nbu-nu); cv = E2v*av + Qv*(2*nbv-nv)
        cr = jnp.real(jnp.fft.ifft2(cu)); cvr = jnp.real(jnp.fft.ifft2(cv))
        ncu = jnp.fft.fft2(Nu(cr, cvr)); ncv = jnp.fft.fft2(Nv(cr, cvr))
        uhn = Eu*uh + f1u*nu + f2u*(nau+nbu) + f3u*ncu
        vhn = Ev*vh + f1v*nv + f2v*(nav+nbv) + f3v*ncv
        return jnp.real(jnp.fft.ifft2(uhn)), jnp.real(jnp.fft.ifft2(vhn))

    def solve(u0, v0, n_steps):
        u, v = u0, v0
        for _ in range(n_steps):
            u, v = step(u, v)
        return u, v
    return solve


# =============================================================================
# Reference Solution: ETDRK4 at very small dt
# =============================================================================
ref_dt = 0.001
ref_steps = int(np.ceil(T_FINAL / ref_dt))
print(f"\nGenerating reference solution (ETDRK4, dt={ref_dt}, {ref_steps} steps)...")
ref_solver = make_etdrk4_solver(ref_dt)
_ = ref_solver(u0_jax, v0_jax, 10)
u0_jax.block_until_ready()
t0 = time.perf_counter()
u_ref, v_ref = ref_solver(u0_jax, v0_jax, ref_steps)
u_ref.block_until_ready()
print(f"  Reference computed in {time.perf_counter() - t0:.1f}s")
print(f"  u range: [{float(jnp.min(u_ref)):.6f}, {float(jnp.max(u_ref)):.6f}]")
print(f"  v range: [{float(jnp.min(v_ref)):.6f}, {float(jnp.max(v_ref)):.6f}]")


def validate_solution(*arrays, label=""):
    """Check that all arrays are finite (no NaN/Inf). Return True if valid."""
    for arr in arrays:
        if not bool(jnp.all(jnp.isfinite(arr))):
            print(f"  WARNING: NaN/Inf in {label}!")
            return False
    return True


def compute_error(u, v):
    if not validate_solution(u, v, label="solution"):
        return float('inf')
    return max(float(jnp.max(jnp.abs(u - u_ref))), float(jnp.max(jnp.abs(v - v_ref))))


# =============================================================================
# Sweep
# =============================================================================
results = {}

# --- RK4 (single point at stability limit) ---
print("\n--- RK4 (stability-limited) ---")
dt_rk4 = dt_rk4_max
n_steps_rk4 = int(np.ceil(T_FINAL / dt_rk4))
nfe_rk4 = 4 * n_steps_rk4
print(f"  dt={dt_rk4:.6e}, steps={n_steps_rk4:,}, NFE={nfe_rk4:,}...", end=" ", flush=True)
_ = rk4_solve(u0_jax, v0_jax, 100, dt_rk4)
u0_jax.block_until_ready()
u_rk4, v_rk4 = rk4_solve(u0_jax, v0_jax, n_steps_rk4, dt_rk4)
u_rk4.block_until_ready()
err_rk4 = compute_error(u_rk4, v_rk4)
print(f"err={err_rk4:.2e}")
results['RK4'] = [{'dt': float(dt_rk4), 'steps': n_steps_rk4, 'nfe': nfe_rk4, 'error': err_rk4}]

# --- IMEX-Strang ---
print("\n--- IMEX-Strang sweep ---")
imex_dts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
imex_results = []
for dt in imex_dts:
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 2 * n_steps
    print(f"  dt={dt}, steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    _ = imex_solve(u0_jax, v0_jax, 10, dt)
    u0_jax.block_until_ready()
    u_sol, v_sol = imex_solve(u0_jax, v0_jax, n_steps, dt)
    u_sol.block_until_ready()
    err = compute_error(u_sol, v_sol)
    print(f"err={err:.2e}")
    imex_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe, 'error': err})
results['IMEX-Strang'] = imex_results

# --- ETDRK4 ---
print("\n--- ETDRK4 sweep ---")
etd_dts = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
etd_results = []
for dt in etd_dts:
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 4 * n_steps
    print(f"  dt={dt}, steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    solver = make_etdrk4_solver(dt)
    _ = solver(u0_jax, v0_jax, 10)
    u0_jax.block_until_ready()
    u_sol, v_sol = solver(u0_jax, v0_jax, n_steps)
    u_sol.block_until_ready()
    err = compute_error(u_sol, v_sol)
    print(f"err={err:.2e}")
    etd_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe, 'error': err})
results['ETDRK4'] = etd_results


# =============================================================================
# Generate Figure
# =============================================================================
output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 7))

styles = {
    'RK4':         {'color': '#d62728', 'marker': 's', 'ms': 12, 'label': 'RK4 (explicit, stability-limited)'},
    'IMEX-Strang': {'color': '#1f77b4', 'marker': 'o', 'ms': 8, 'label': 'IMEX-Strang (2 NFE/step)'},
    'ETDRK4':      {'color': '#ff7f0e', 'marker': 'D', 'ms': 8, 'label': 'ETDRK4 (4 NFE/step)'},
}

for method, data in results.items():
    nfes = [d['nfe'] for d in data]
    errs = [d['error'] for d in data]
    s = styles[method]
    ax.plot(nfes, errs, marker=s['marker'], color=s['color'], lw=2, ms=s['ms'],
            label=s['label'], markeredgecolor='black', markeredgewidth=0.5)

# Add annotation for the key result
if results.get('IMEX-Strang') and results.get('RK4'):
    imex_best = results['IMEX-Strang'][0]  # Smallest dt = best accuracy
    rk4_pt = results['RK4'][0]
    nfe_ratio = rk4_pt['nfe'] / imex_best['nfe']
    ax.annotate(f'RK4: {rk4_pt["nfe"]:,} NFE\n(stability-limited)',
                xy=(rk4_pt['nfe'], rk4_pt['error']),
                xytext=(rk4_pt['nfe']/5, rk4_pt['error']*3),
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=10, ha='center')

ax.set_xlabel('Nonlinear Function Evaluations (NFE)', fontsize=13)
ax.set_ylabel('$\\|e\\|_\\infty$ vs reference', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'Work-Precision Diagram: Brusselator ($256^2$, $t = {int(T_FINAL)}$)', fontsize=14)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / "fig_work_precision_brusselator.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_work_precision_brusselator.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_work_precision_brusselator.pdf'}")

# Save data
data_dir = Path(__file__).parent / "results"
data_dir.mkdir(exist_ok=True)
with open(data_dir / "work_precision_brusselator.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {data_dir / 'work_precision_brusselator.json'}")
print("\nDone!")
