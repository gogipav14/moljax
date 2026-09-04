#!/usr/bin/env python3
"""
Work-Precision Diagram: Gray-Scott System

Plots Error vs. NFE (Nonlinear Function Evaluations) for each method.
This is the implementation-independent "gold standard" for algorithmic comparison.

NFE counting:
  - RK4: 4 NFE per step (4 RK stages, each evaluates reaction)
  - CN+FFT: ~3 NFE per step (Newton iterations, each evaluates full residual)
  - IMEX-Strang: 2 NFE per step (2 half-step reaction evaluations)
  - ETDRK4: 4 NFE per step (4 nonlinear evaluations at different stages)

Reference solution: ETDRK4 with dt=0.1 (100,000 steps for t=10000).
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
print("Work-Precision Diagram: Gray-Scott")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")

# Configuration
N = 256
Du = 2e-5
Dv = 1e-5
F = 0.035
k = 0.065
dx = 1.0 / N
T_FINAL = 100.0  # Early transient: pattern formation actively underway

# Wavenumbers
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Initial condition (same as main benchmark)
np.random.seed(42)
u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1
cx, cy = N//2, N//2
r = 10
u0[cx-r:cx+r, cy-r:cy+r] = 0.5
v0[cx-r:cx+r, cy-r:cy+r] = 0.25
u0_jax = jnp.array(u0)
v0_jax = jnp.array(v0)

# Stability limit for RK4
lambda_max = np.max(-lap_eig)
dt_rk4_max = 0.9 * 2.8 / (max(Du, Dv) * lambda_max)


# =============================================================================
# Solver Definitions (reused from benchmark_gray_scott.py)
# =============================================================================

@jax.jit
def rk4_rhs(u, v):
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    lap_u = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat))
    lap_v = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat))
    uvv = u * v * v
    du = Du * lap_u - uvv + F * (1 - u)
    dv = Dv * lap_v + uvv - (F + k) * v
    return du, dv

@jax.jit
def rk4_step(u, v, dt):
    k1u, k1v = rk4_rhs(u, v)
    k2u, k2v = rk4_rhs(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
    k3u, k3v = rk4_rhs(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
    k4u, k4v = rk4_rhs(u + dt*k3u, v + dt*k3v)
    u_new = u + dt/6 * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
    return u_new, v_new

def rk4_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = rk4_step(u, v, dt)
    return u, v


@jax.jit
def imex_step(u, v, dt):
    uvv = u * v * v
    u = u + 0.5*dt * (-uvv + F * (1 - u))
    v = v + 0.5*dt * (uvv - (F + k) * v)
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    u_hat = u_hat * jnp.exp(dt * Du * lap_eig_jax)
    v_hat = v_hat * jnp.exp(dt * Dv * lap_eig_jax)
    u = jnp.real(jnp.fft.ifft2(u_hat))
    v = jnp.real(jnp.fft.ifft2(v_hat))
    uvv = u * v * v
    u = u + 0.5*dt * (-uvv + F * (1 - u))
    v = v + 0.5*dt * (uvv - (F + k) * v)
    return u, v

def imex_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = imex_step(u, v, dt)
    return u, v


def make_etdrk4_solver(dt_etd):
    """Create ETDRK4 solver for given dt."""
    Lu = Du * lap_eig
    Lv = Dv * lap_eig

    def compute_etdrk4_coeffs(L, dt, M=32):
        r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
        LR = dt * L[..., np.newaxis] + r
        E = np.exp(dt * L)
        E2 = np.exp(dt * L / 2)
        Q = dt * np.mean((np.exp(LR/2) - 1) / LR, axis=-1)
        f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1)
        f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1)
        f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1)
        return E, E2, Q, f1, f2, f3

    E_u, E2_u, Q_u, f1_u, f2_u, f3_u = compute_etdrk4_coeffs(Lu, dt_etd)
    E_v, E2_v, Q_v, f1_v, f2_v, f3_v = compute_etdrk4_coeffs(Lv, dt_etd)
    E_u, E2_u, Q_u, f1_u, f2_u, f3_u = [jnp.array(x) for x in [E_u, E2_u, Q_u, f1_u, f2_u, f3_u]]
    E_v, E2_v, Q_v, f1_v, f2_v, f3_v = [jnp.array(x) for x in [E_v, E2_v, Q_v, f1_v, f2_v, f3_v]]

    @jax.jit
    def etdrk4_step(u, v):
        u_hat = jnp.fft.fft2(u)
        v_hat = jnp.fft.fft2(v)

        def N_u(u, v): return -u*v*v + F*(1-u)
        def N_v(u, v): return u*v*v - (F+k)*v

        Nu = jnp.fft.fft2(N_u(u, v))
        Nv = jnp.fft.fft2(N_v(u, v))

        a_u = E2_u * u_hat + Q_u * Nu
        a_v = E2_v * v_hat + Q_v * Nv
        a_u_r = jnp.real(jnp.fft.ifft2(a_u))
        a_v_r = jnp.real(jnp.fft.ifft2(a_v))

        Na_u = jnp.fft.fft2(N_u(a_u_r, a_v_r))
        Na_v = jnp.fft.fft2(N_v(a_u_r, a_v_r))

        b_u = E2_u * u_hat + Q_u * Na_u
        b_v = E2_v * v_hat + Q_v * Na_v
        b_u_r = jnp.real(jnp.fft.ifft2(b_u))
        b_v_r = jnp.real(jnp.fft.ifft2(b_v))

        Nb_u = jnp.fft.fft2(N_u(b_u_r, b_v_r))
        Nb_v = jnp.fft.fft2(N_v(b_u_r, b_v_r))

        c_u = E2_u * a_u + Q_u * (2*Nb_u - Nu)
        c_v = E2_v * a_v + Q_v * (2*Nb_v - Nv)
        c_u_r = jnp.real(jnp.fft.ifft2(c_u))
        c_v_r = jnp.real(jnp.fft.ifft2(c_v))

        Nc_u = jnp.fft.fft2(N_u(c_u_r, c_v_r))
        Nc_v = jnp.fft.fft2(N_v(c_u_r, c_v_r))

        u_hat_new = E_u * u_hat + f1_u * Nu + f2_u * (Na_u + Nb_u) + f3_u * Nc_u
        v_hat_new = E_v * v_hat + f1_v * Nv + f2_v * (Na_v + Nb_v) + f3_v * Nc_v

        return jnp.real(jnp.fft.ifft2(u_hat_new)), jnp.real(jnp.fft.ifft2(v_hat_new))

    def etdrk4_solve(u0, v0, n_steps):
        u, v = u0, v0
        for _ in range(n_steps):
            u, v = etdrk4_step(u, v)
        return u, v

    return etdrk4_solve


def make_cn_solver(dt_cn):
    """Create CN solver for given dt."""
    cn_precond_u = 1.0 / (1.0 - 0.5 * dt_cn * Du * lap_eig_jax)
    cn_precond_v = 1.0 / (1.0 - 0.5 * dt_cn * Dv * lap_eig_jax)
    CN_TOL = 1e-8
    CN_MAX_ITER = 10

    @jax.jit
    def cn_newton_step(u_old, v_old):
        def compute_residual(u_new, v_new):
            u_hat_new = jnp.fft.fft2(u_new)
            v_hat_new = jnp.fft.fft2(v_new)
            u_hat_old = jnp.fft.fft2(u_old)
            v_hat_old = jnp.fft.fft2(v_old)
            lap_u_new = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat_new))
            lap_v_new = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat_new))
            lap_u_old = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat_old))
            lap_v_old = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat_old))
            uvv_new = u_new * v_new * v_new
            uvv_old = u_old * v_old * v_old
            R_u_new = -uvv_new + F * (1 - u_new)
            R_v_new = uvv_new - (F + k) * v_new
            R_u_old = -uvv_old + F * (1 - u_old)
            R_v_old = uvv_old - (F + k) * v_old
            res_u = u_new - u_old - 0.5 * dt_cn * (Du * (lap_u_new + lap_u_old) + R_u_new + R_u_old)
            res_v = v_new - v_old - 0.5 * dt_cn * (Dv * (lap_v_new + lap_v_old) + R_v_new + R_v_old)
            return res_u, res_v

        def apply_precond(du, dv):
            du_hat = jnp.fft.fft2(du)
            dv_hat = jnp.fft.fft2(dv)
            return jnp.real(jnp.fft.ifft2(cn_precond_u * du_hat)), jnp.real(jnp.fft.ifft2(cn_precond_v * dv_hat))

        def cond_fn(state):
            u_new, v_new, res_norm, i = state
            return (res_norm > CN_TOL) & (i < CN_MAX_ITER)

        def body_fn(state):
            u_new, v_new, _, i = state
            res_u, res_v = compute_residual(u_new, v_new)
            du, dv = apply_precond(-res_u, -res_v)
            u_new = u_new + du
            v_new = v_new + dv
            res_norm = jnp.max(jnp.abs(res_u)) + jnp.max(jnp.abs(res_v))
            return (u_new, v_new, res_norm, i + 1)

        init_state = (u_old, v_old, jnp.array(1.0), 0)
        final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
        return final_state[0], final_state[1]

    def cn_solve(u0, v0, n_steps):
        u, v = u0, v0
        for _ in range(n_steps):
            u, v = cn_newton_step(u, v)
        return u, v

    return cn_solve


# =============================================================================
# Generate Reference Solution (ETDRK4 with very small dt)
# =============================================================================
ref_dt = 0.01
ref_steps = int(np.ceil(T_FINAL / ref_dt))
print(f"\nGenerating reference solution (ETDRK4, dt={ref_dt}, {ref_steps} steps)...")
ref_solver = make_etdrk4_solver(ref_dt)
# Warmup
_ = ref_solver(u0_jax, v0_jax, 10)
u0_jax.block_until_ready()
# Solve
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


def compute_error(u, v, u_ref, v_ref):
    """L-infinity error vs reference."""
    if not validate_solution(u, v, label="solution"):
        return float('inf')
    err_u = float(jnp.max(jnp.abs(u - u_ref)))
    err_v = float(jnp.max(jnp.abs(v - v_ref)))
    return max(err_u, err_v)


# =============================================================================
# Sweep: Error vs NFE for each method
# =============================================================================
results = {}

# --- RK4 ---
# RK4 has a MAXIMUM dt from stability. We sweep from that max down.
# dt values: 0.5*max, 0.6*max, 0.7*max, 0.8*max, 0.9*max
print("\n--- RK4 sweep ---")
rk4_fracs = [0.5, 0.6, 0.7, 0.8, 0.9]
rk4_results = []
for frac in rk4_fracs:
    dt = frac * dt_rk4_max
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 4 * n_steps
    print(f"  dt={dt:.4f}, steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    # Warmup
    _ = rk4_solve(u0_jax, v0_jax, 100, dt)
    u0_jax.block_until_ready()
    # Solve
    u_sol, v_sol = rk4_solve(u0_jax, v0_jax, n_steps, dt)
    u_sol.block_until_ready()
    err = compute_error(u_sol, v_sol, u_ref, v_ref)
    print(f"err={err:.2e}")
    rk4_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe, 'error': err})
results['RK4'] = rk4_results

# --- IMEX-Strang ---
print("\n--- IMEX-Strang sweep ---")
imex_dts = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
imex_results = []
for dt in imex_dts:
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 2 * n_steps  # 2 reaction evaluations per Strang step
    print(f"  dt={dt}, steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    _ = imex_solve(u0_jax, v0_jax, 10, dt)
    u0_jax.block_until_ready()
    u_sol, v_sol = imex_solve(u0_jax, v0_jax, n_steps, dt)
    u_sol.block_until_ready()
    err = compute_error(u_sol, v_sol, u_ref, v_ref)
    print(f"err={err:.2e}")
    imex_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe, 'error': err})
results['IMEX-Strang'] = imex_results

# --- ETDRK4 ---
print("\n--- ETDRK4 sweep ---")
etd_dts = [0.2, 0.5, 1.0, 2.0, 5.0]
etd_results = []
for dt in etd_dts:
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe = 4 * n_steps  # 4 nonlinear evaluations per ETDRK4 step
    print(f"  dt={dt}, steps={n_steps:,}, NFE={nfe:,}...", end=" ", flush=True)
    solver = make_etdrk4_solver(dt)
    _ = solver(u0_jax, v0_jax, 10)
    u0_jax.block_until_ready()
    u_sol, v_sol = solver(u0_jax, v0_jax, n_steps)
    u_sol.block_until_ready()
    err = compute_error(u_sol, v_sol, u_ref, v_ref)
    print(f"err={err:.2e}")
    etd_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe, 'error': err})
results['ETDRK4'] = etd_results

# --- CN+FFT ---
print("\n--- CN+FFT sweep ---")
cn_dts = [0.1, 0.2, 0.5, 1.0, 2.0]
cn_results = []
for dt in cn_dts:
    n_steps = int(np.ceil(T_FINAL / dt))
    nfe_per_step = 3  # ~3 Newton iterations per step
    nfe = nfe_per_step * n_steps
    print(f"  dt={dt}, steps={n_steps:,}, NFE~{nfe:,}...", end=" ", flush=True)
    solver = make_cn_solver(dt)
    _ = solver(u0_jax, v0_jax, 10)
    u0_jax.block_until_ready()
    u_sol, v_sol = solver(u0_jax, v0_jax, n_steps)
    u_sol.block_until_ready()
    err = compute_error(u_sol, v_sol, u_ref, v_ref)
    print(f"err={err:.2e}")
    cn_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe, 'error': err})
results['CN+FFT'] = cn_results


# =============================================================================
# Generate Figure: Work-Precision Diagram
# =============================================================================
output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 7))

styles = {
    'RK4':         {'color': '#d62728', 'marker': 's', 'label': 'RK4 (explicit)'},
    'CN+FFT':      {'color': '#2ca02c', 'marker': '^', 'label': 'CN + FFT precond'},
    'IMEX-Strang': {'color': '#1f77b4', 'marker': 'o', 'label': 'IMEX-Strang'},
    'ETDRK4':      {'color': '#ff7f0e', 'marker': 'D', 'label': 'ETDRK4'},
}

for method, data in results.items():
    nfes = [d['nfe'] for d in data]
    errs = [d['error'] for d in data]
    s = styles[method]
    ax.plot(nfes, errs, marker=s['marker'], color=s['color'], lw=2, ms=8,
            label=s['label'], markeredgecolor='black', markeredgewidth=0.5)

ax.set_xlabel('Nonlinear Function Evaluations (NFE)', fontsize=13)
ax.set_ylabel('$\\|e\\|_\\infty$ vs reference', fontsize=13)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title(f'Work-Precision Diagram: Gray-Scott ($256^2$, $t = {int(T_FINAL)}$)', fontsize=14)
ax.legend(fontsize=11, loc='best')
ax.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(output_dir / "fig_work_precision_gray_scott.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_work_precision_gray_scott.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_work_precision_gray_scott.pdf'}")

# Save data
data_dir = Path(__file__).parent / "results"
data_dir.mkdir(exist_ok=True)
with open(data_dir / "work_precision_gray_scott.json", 'w') as f:
    json.dump(results, f, indent=2)
print(f"Saved: {data_dir / 'work_precision_gray_scott.json'}")

print("\nDone!")
