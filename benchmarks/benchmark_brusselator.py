#!/usr/bin/env python3
"""
Benchmark: Brusselator Reaction-Diffusion (Oscillatory Instability)

System:
    du/dt = Du * lap(u) + a - (b+1)*u + u^2*v
    dv/dt = Dv * lap(v) + b*u - u^2*v

Parameters: a=1, b=3.4 (above Hopf, oscillatory), Du=0.01, Dv=0.02
Grid: 256x256, domain [0, 1]^2, t in [0, 50]

Methods compared:
- RK4 (explicit)
- Crank-Nicolson (Newton-Krylov with FFT preconditioner)
- IMEX-Strang
- ETDRK4

This problem exhibits OSCILLATORY stiffness: at b > 1+a^2 = 2 (Hopf threshold),
the homogeneous steady state is unstable through oscillation. The combination of
oscillatory instability + diffusion-driven pattern formation creates doubly stiff
dynamics — both eigenvalue magnitudes AND imaginary parts are large.
"""

import json
import time
from pathlib import Path

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Configuration
N = 256
Du = 0.01
Dv = 0.02
a = 1.0
b = 3.4  # Above Hopf threshold (b > 1 + a^2 = 2)
L = 1.0
dx = L / N

T_FINAL = 50.0
N_REPS = 5  # Reduced for faster execution; can increase for publication

print("Brusselator Benchmark")
print(f"Grid: {N}x{N}, Du={Du}, Dv={Dv}, a={a}, b={b}")
print(f"Hopf threshold: b > 1 + a^2 = {1 + a**2} (b={b}, ABOVE)")
print(f"Simulation time: t=0 to t={T_FINAL}")
print(f"JAX devices: {jax.devices()}")
print(f"Float64 enabled: {jax.config.jax_enable_x64}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Setup wavenumbers (pseudo-spectral Laplacian)
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Initial condition: perturbation from homogeneous steady state
# Steady state: u_s = a, v_s = b/a
u_s = a
v_s = b / a
np.random.seed(42)
u0 = u_s + 0.01 * np.random.randn(N, N)
v0 = v_s + 0.01 * np.random.randn(N, N)

u0_jax = jnp.array(u0)
v0_jax = jnp.array(v0)

results = {'methods': []}

# =============================================================================
# 1. RK4 (Explicit) — Extrapolated from short run
# =============================================================================
print("\n1. RK4 (explicit)...")

# Stability limits
lambda_max = np.max(-lap_eig)
dt_diff = 2.8 / (max(Du, Dv) * lambda_max)
dt_react = 0.5 / (b + 1.0)
dt_rk4 = 0.9 * min(dt_diff, dt_react)
n_steps_rk4_full = int(np.ceil(T_FINAL / dt_rk4))
print(f"   dt_diff={dt_diff:.6e}, dt_react={dt_react:.6e}")
print(f"   Using dt={dt_rk4:.6e}")
print(f"   Full steps needed: {n_steps_rk4_full:,}")

# For timing, run a short burst then extrapolate
N_SHORT = min(5000, n_steps_rk4_full)

@jax.jit
def rk4_rhs_brus(u, v):
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    lap_u = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat))
    lap_v = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat))

    u2v = u * u * v
    du = Du * lap_u + a - (b + 1.0) * u + u2v
    dv = Dv * lap_v + b * u - u2v
    return du, dv

@jax.jit
def rk4_step_brus(u, v, dt):
    k1u, k1v = rk4_rhs_brus(u, v)
    k2u, k2v = rk4_rhs_brus(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
    k3u, k3v = rk4_rhs_brus(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
    k4u, k4v = rk4_rhs_brus(u + dt*k3u, v + dt*k3v)

    u_new = u + dt/6 * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
    return u_new, v_new

def rk4_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = rk4_step_brus(u, v, dt)
    return u, v

# Warmup
_ = rk4_solve(u0_jax, v0_jax, 10, dt_rk4)
u0_jax.block_until_ready()

times_rk4 = []
for _ in range(N_REPS):
    start = time.perf_counter()
    u_rk4, v_rk4 = rk4_solve(u0_jax, v0_jax, N_SHORT, dt_rk4)
    u_rk4.block_until_ready()
    times_rk4.append(time.perf_counter() - start)

rk4_short_median = np.median(times_rk4)
rk4_per_step = rk4_short_median / N_SHORT
rk4_extrapolated = rk4_per_step * n_steps_rk4_full
rk4_q25, rk4_q75 = np.percentile(times_rk4, [25, 75])
rk4_iqr_short = rk4_q75 - rk4_q25
rk4_iqr_extrap = rk4_iqr_short / N_SHORT * n_steps_rk4_full

results['methods'].append({
    'name': 'RK4 (explicit)',
    'time_s': float(rk4_extrapolated),
    'iqr_s': float(rk4_iqr_extrap),
    'steps': int(n_steps_rk4_full),
    'dt': float(dt_rk4),
    't_final': float(T_FINAL),
    'extrapolated': True,
    'measured_steps': int(N_SHORT),
    'measured_time_s': float(rk4_short_median),
    'per_step_s': float(rk4_per_step),
})
print(f"   Measured: {rk4_short_median:.3f}s for {N_SHORT:,} steps ({rk4_per_step:.6e} s/step)")
print(f"   Extrapolated: {rk4_extrapolated:.1f}s for {n_steps_rk4_full:,} steps")

# =============================================================================
# 2. Crank-Nicolson (Newton-Krylov with FFT preconditioner)
# =============================================================================
print("\n2. CN (Newton-Krylov + FFT preconditioner)...")

dt_cn = 0.05
n_steps_cn = int(np.ceil(T_FINAL / dt_cn))
t_actual_cn = n_steps_cn * dt_cn
print(f"   Steps: {n_steps_cn:,}, dt={dt_cn}")

cn_precond_u = 1.0 / (1.0 - 0.5 * dt_cn * Du * lap_eig_jax)
cn_precond_v = 1.0 / (1.0 - 0.5 * dt_cn * Dv * lap_eig_jax)

CN_TOL = 1e-8
CN_MAX_ITER = 15

@jax.jit
def cn_newton_step_brus(u_old, v_old):
    def compute_residual(u_new, v_new):
        u_hat_new = jnp.fft.fft2(u_new)
        v_hat_new = jnp.fft.fft2(v_new)
        u_hat_old = jnp.fft.fft2(u_old)
        v_hat_old = jnp.fft.fft2(v_old)

        lap_u_new = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat_new))
        lap_v_new = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat_new))
        lap_u_old = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat_old))
        lap_v_old = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat_old))

        u2v_new = u_new * u_new * v_new
        u2v_old = u_old * u_old * v_old

        R_u_new = a - (b + 1.0) * u_new + u2v_new
        R_v_new = b * u_new - u2v_new
        R_u_old = a - (b + 1.0) * u_old + u2v_old
        R_v_old = b * u_old - u2v_old

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

@jax.jit
def cn_solve_brus(u0, v0, n_steps):
    def body(i, uv):
        u, v = uv
        return cn_newton_step_brus(u, v)
    return jax.lax.fori_loop(0, n_steps, body, (u0, v0))

# Warmup
_ = cn_solve_brus(u0_jax, v0_jax, 10)
u0_jax.block_until_ready()

times_cn = []
for _ in range(N_REPS):
    start = time.perf_counter()
    u_cn, v_cn = cn_solve_brus(u0_jax, v0_jax, n_steps_cn)
    u_cn.block_until_ready()
    times_cn.append(time.perf_counter() - start)

cn_median = np.median(times_cn)
cn_q25, cn_q75 = np.percentile(times_cn, [25, 75])
cn_iqr = cn_q75 - cn_q25

results['methods'].append({
    'name': 'CN (Newton-Krylov)',
    'time_s': float(cn_median),
    'iqr_s': float(cn_iqr),
    'steps': int(n_steps_cn),
    'dt': float(dt_cn),
    't_final': float(t_actual_cn),
})
print(f"   Time: {cn_median:.2f} +/- {cn_iqr:.3f} s (median+/-IQR)")

# =============================================================================
# 3. IMEX-Strang (Splitting)
# =============================================================================
print("\n3. IMEX-Strang...")

dt_imex = 0.05
n_steps_imex = int(np.ceil(T_FINAL / dt_imex))
t_actual_imex = n_steps_imex * dt_imex
print(f"   Steps: {n_steps_imex:,}, dt={dt_imex}")

@jax.jit
def imex_step_brus(u, v, dt):
    # Half step: explicit reaction
    u2v = u * u * v
    u = u + 0.5*dt * (a - (b + 1.0) * u + u2v)
    v = v + 0.5*dt * (b * u - u2v)

    # Full step: implicit diffusion (exact via FFT)
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    u_hat = u_hat * jnp.exp(dt * Du * lap_eig_jax)
    v_hat = v_hat * jnp.exp(dt * Dv * lap_eig_jax)
    u = jnp.real(jnp.fft.ifft2(u_hat))
    v = jnp.real(jnp.fft.ifft2(v_hat))

    # Half step: explicit reaction
    u2v = u * u * v
    u = u + 0.5*dt * (a - (b + 1.0) * u + u2v)
    v = v + 0.5*dt * (b * u - u2v)

    return u, v

def imex_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = imex_step_brus(u, v, dt)
    return u, v

# Warmup
_ = imex_solve(u0_jax, v0_jax, 10, dt_imex)
u0_jax.block_until_ready()

times_imex = []
for _ in range(N_REPS):
    start = time.perf_counter()
    u_imex, v_imex = imex_solve(u0_jax, v0_jax, n_steps_imex, dt_imex)
    u_imex.block_until_ready()
    times_imex.append(time.perf_counter() - start)

imex_median = np.median(times_imex)
imex_q25, imex_q75 = np.percentile(times_imex, [25, 75])
imex_iqr = imex_q75 - imex_q25

results['methods'].append({
    'name': 'IMEX-Strang',
    'time_s': float(imex_median),
    'iqr_s': float(imex_iqr),
    'steps': int(n_steps_imex),
    'dt': float(dt_imex),
    't_final': float(t_actual_imex),
})
print(f"   Time: {imex_median:.2f} +/- {imex_iqr:.3f} s (median+/-IQR)")

# =============================================================================
# 4. ETDRK4 (Exponential Time Differencing)
# =============================================================================
print("\n4. ETDRK4...")

dt_etd = 0.05
n_steps_etd = int(np.ceil(T_FINAL / dt_etd))
t_actual_etd = n_steps_etd * dt_etd
print(f"   Steps: {n_steps_etd:,}, dt={dt_etd}")

# Precompute ETDRK4 coefficients (Cox & Matthews 2002)
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

Lu = Du * lap_eig
Lv = Dv * lap_eig

E_u, E2_u, Q_u, f1_u, f2_u, f3_u = compute_etdrk4_coeffs(Lu, dt_etd)
E_v, E2_v, Q_v, f1_v, f2_v, f3_v = compute_etdrk4_coeffs(Lv, dt_etd)

E_u, E2_u, Q_u, f1_u, f2_u, f3_u = [jnp.array(x) for x in [E_u, E2_u, Q_u, f1_u, f2_u, f3_u]]
E_v, E2_v, Q_v, f1_v, f2_v, f3_v = [jnp.array(x) for x in [E_v, E2_v, Q_v, f1_v, f2_v, f3_v]]

@jax.jit
def etdrk4_step_brus(u, v):
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)

    def N_u(u, v):
        return a - (b + 1.0) * u + u*u*v

    def N_v(u, v):
        return b * u - u*u*v

    # Stage 1
    Nu = jnp.fft.fft2(N_u(u, v))
    Nv = jnp.fft.fft2(N_v(u, v))

    a_u = E2_u * u_hat + Q_u * Nu
    a_v = E2_v * v_hat + Q_v * Nv
    a_u_real = jnp.real(jnp.fft.ifft2(a_u))
    a_v_real = jnp.real(jnp.fft.ifft2(a_v))

    # Stage 2
    Na_u = jnp.fft.fft2(N_u(a_u_real, a_v_real))
    Na_v = jnp.fft.fft2(N_v(a_u_real, a_v_real))

    b_u = E2_u * u_hat + Q_u * Na_u
    b_v = E2_v * v_hat + Q_v * Na_v
    b_u_real = jnp.real(jnp.fft.ifft2(b_u))
    b_v_real = jnp.real(jnp.fft.ifft2(b_v))

    # Stage 3
    Nb_u = jnp.fft.fft2(N_u(b_u_real, b_v_real))
    Nb_v = jnp.fft.fft2(N_v(b_u_real, b_v_real))

    c_u = E2_u * a_u + Q_u * (2*Nb_u - Nu)
    c_v = E2_v * a_v + Q_v * (2*Nb_v - Nv)
    c_u_real = jnp.real(jnp.fft.ifft2(c_u))
    c_v_real = jnp.real(jnp.fft.ifft2(c_v))

    # Stage 4
    Nc_u = jnp.fft.fft2(N_u(c_u_real, c_v_real))
    Nc_v = jnp.fft.fft2(N_v(c_u_real, c_v_real))

    # Final update
    u_hat_new = E_u * u_hat + f1_u * Nu + f2_u * (Na_u + Nb_u) + f3_u * Nc_u
    v_hat_new = E_v * v_hat + f1_v * Nv + f2_v * (Na_v + Nb_v) + f3_v * Nc_v

    return jnp.real(jnp.fft.ifft2(u_hat_new)), jnp.real(jnp.fft.ifft2(v_hat_new))

def etdrk4_solve(u0, v0, n_steps):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = etdrk4_step_brus(u, v)
    return u, v

# Warmup
_ = etdrk4_solve(u0_jax, v0_jax, 10)
u0_jax.block_until_ready()

times_etd = []
for _ in range(N_REPS):
    start = time.perf_counter()
    u_etd, v_etd = etdrk4_solve(u0_jax, v0_jax, n_steps_etd)
    u_etd.block_until_ready()
    times_etd.append(time.perf_counter() - start)

etd_median = np.median(times_etd)
etd_q25, etd_q75 = np.percentile(times_etd, [25, 75])
etd_iqr = etd_q75 - etd_q25

results['methods'].append({
    'name': 'ETDRK4',
    'time_s': float(etd_median),
    'iqr_s': float(etd_iqr),
    'steps': int(n_steps_etd),
    'dt': float(dt_etd),
    't_final': float(t_actual_etd),
})
print(f"   Time: {etd_median:.2f} +/- {etd_iqr:.3f} s (median+/-IQR)")

# =============================================================================
# Save and Summary
# =============================================================================
device = str(jax.devices()[0])
dtype_str = 'float64' if jax.config.jax_enable_x64 else 'float32'

results['config'] = {
    'N': N,
    'Du': Du,
    'Dv': Dv,
    'a': a,
    'b': b,
    'L': L,
    't_final': T_FINAL,
    'hopf_threshold': 1 + a**2,
    'n_reps': N_REPS,
    'dtype': dtype_str,
    'device': device,
    'laplacian': 'pseudo-spectral (-k^2)',
    'system': 'Brusselator',
}

output_path = Path(__file__).parent / 'results' / 'brusselator.json'
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Save final pattern for figure
np.save(output_path.parent / 'brusselator_u_final_etd.npy', np.array(u_etd))
np.save(output_path.parent / 'brusselator_v_final_etd.npy', np.array(v_etd))

print("\n" + "=" * 80)
print(f"SUMMARY - Brusselator, {dtype_str}, {device}")
print("=" * 80)
print(f"{'Method':<20} {'Steps':>12} {'Time (s)':>14} {'dt':>12}")
print("-" * 80)
for m in results['methods']:
    time_str = f"{m['time_s']:.2f} +/- {m['iqr_s']:.2f}"
    print(f"{m['name']:<20} {m['steps']:>12,} {time_str:>14} {m['dt']:>12.6f}")
print("=" * 80)
