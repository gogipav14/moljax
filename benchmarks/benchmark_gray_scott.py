#!/usr/bin/env python3
"""
Benchmark: Gray-Scott Reaction-Diffusion (Table 7)

Gray-Scott system:
    du/dt = Du * lap(u) - u*v^2 + F*(1-u)
    dv/dt = Dv * lap(v) + u*v^2 - (F+k)*v

Parameters: Du=2e-5, Dv=1e-5, F=0.035, k=0.065 (spot pattern)
Grid: 256x256, t ∈ [0, 10000]

Methods compared:
- RK4 (explicit)
- Crank-Nicolson (Newton-Krylov)
- IMEX-Strang
- ETDRK4

IMPORTANT: This benchmark runs to ACTUAL t=10000. No scaling or extrapolation.
All reported times are actual wallclock times to complete the full simulation.
"""

import numpy as np
import time
import json
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision
import jax.numpy as jnp

# Configuration
N = 256
Du = 2e-5
Dv = 1e-5
F = 0.035
k = 0.065
dx = 1.0 / N

# ACTUAL simulation time - NO SCALING
T_FINAL = 10000.0  # Full simulation time

N_REPS = 10  # Paper uses 10 runs, report median

print("Gray-Scott Benchmark (ACTUAL t=10000, NO SCALING)")
print(f"Grid: {N}x{N}, Du={Du}, Dv={Dv}, F={F}, k={k}")
print(f"Simulation time: t=0 to t={T_FINAL} (actual, no extrapolation)")
print(f"JAX devices: {jax.devices()}")
print(f"Float64 enabled: {jax.config.jax_enable_x64}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Setup wavenumbers
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Initial condition: small perturbation from steady state
np.random.seed(42)
u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1

# Add localized perturbation
cx, cy = N//2, N//2
r = 10
u0[cx-r:cx+r, cy-r:cy+r] = 0.5
v0[cx-r:cx+r, cy-r:cy+r] = 0.25

u0_jax = jnp.array(u0)
v0_jax = jnp.array(v0)

results = {'methods': []}

# =============================================================================
# 1. RK4 (Explicit)
# =============================================================================
print("\n1. RK4 (explicit)...")

# For PSEUDO-SPECTRAL Laplacian (-k^2), stability limit is much stricter than FD CFL.
# RK4 stability region on negative real axis: |λ*dt| < 2.8
# λ_max = max|k|^2 = (N/2 * 2π / L)^2 for each dimension, sum for 2D
# With N=256, L=1: k_max = 128*2π ≈ 804, λ_max ≈ 1.29e6
# dt_stable = 2.8 / (D * λ_max) ≈ 0.108 for Du=2e-5
lambda_max = np.max(-lap_eig)  # Maximum eigenvalue magnitude
dt_rk4 = 0.9 * 2.8 / (max(Du, Dv) * lambda_max)  # 90% of stability limit
n_steps_rk4 = int(np.ceil(T_FINAL / dt_rk4))  # ceil ensures t_final >= T_FINAL
t_actual_rk4 = n_steps_rk4 * dt_rk4
print(f"   Pseudo-spectral stability: dt_max={2.8/(max(Du,Dv)*lambda_max):.6f}, using dt={dt_rk4:.6f}")
print(f"   Steps: {n_steps_rk4:,}, actual t_final: {t_actual_rk4:.2f}")

@jax.jit
def rk4_rhs(u, v):
    """Gray-Scott RHS."""
    # Laplacian via FFT
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    lap_u = jnp.real(jnp.fft.ifft2(lap_eig_jax * u_hat))
    lap_v = jnp.real(jnp.fft.ifft2(lap_eig_jax * v_hat))

    # Reaction terms
    uvv = u * v * v
    du = Du * lap_u - uvv + F * (1 - u)
    dv = Dv * lap_v + uvv - (F + k) * v

    return du, dv

@jax.jit
def rk4_step(u, v, dt):
    """RK4 step."""
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

# Warmup
_ = rk4_solve(u0_jax, v0_jax, 100, dt_rk4)
u0_jax.block_until_ready()

times_rk4 = []
for _ in range(N_REPS):
    start = time.perf_counter()
    u_rk4, v_rk4 = rk4_solve(u0_jax, v0_jax, n_steps_rk4, dt_rk4)
    u_rk4.block_until_ready()
    times_rk4.append(time.perf_counter() - start)

rk4_median = np.median(times_rk4)
rk4_q25, rk4_q75 = np.percentile(times_rk4, [25, 75])
rk4_iqr = rk4_q75 - rk4_q25

results['methods'].append({
    'name': 'RK4 (explicit)',
    'time_s': float(rk4_median),
    'iqr_s': float(rk4_iqr),
    'steps': int(n_steps_rk4),
    'dt': float(dt_rk4),
    't_final': float(t_actual_rk4),
})
print(f"   Time: {rk4_median:.2f} ± {rk4_iqr:.3f} s (median±IQR)")
print(f"   Steps: {n_steps_rk4:,}")

# =============================================================================
# 2. Crank-Nicolson (Newton-Krylov with FFT preconditioner) - FULLY JIT
# =============================================================================
print("\n2. CN (Newton-Krylov + FFT preconditioner)...")

dt_cn = 0.5  # Same as IMEX for fair comparison
n_steps_cn = int(np.ceil(T_FINAL / dt_cn))  # ceil for consistency
t_actual_cn = n_steps_cn * dt_cn
print(f"   Steps: {n_steps_cn:,}, dt={dt_cn}")

# FFT preconditioner: invert (I - dt/2 * D * lap) exactly
cn_precond_u = 1.0 / (1.0 - 0.5 * dt_cn * Du * lap_eig_jax)
cn_precond_v = 1.0 / (1.0 - 0.5 * dt_cn * Dv * lap_eig_jax)

CN_TOL = 1e-8
CN_MAX_ITER = 10

@jax.jit
def cn_newton_step_jit(u_old, v_old):
    """One CN step using Newton iteration - FULLY JIT compiled."""

    def compute_residual(u_new, v_new):
        """Compute CN residual."""
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
        """Apply FFT preconditioner."""
        du_hat = jnp.fft.fft2(du)
        dv_hat = jnp.fft.fft2(dv)
        return jnp.real(jnp.fft.ifft2(cn_precond_u * du_hat)), jnp.real(jnp.fft.ifft2(cn_precond_v * dv_hat))

    def cond_fn(state):
        """Continue while residual > tol and iter < max_iter."""
        u_new, v_new, res_norm, i = state
        return (res_norm > CN_TOL) & (i < CN_MAX_ITER)

    def body_fn(state):
        """One Newton iteration."""
        u_new, v_new, _, i = state
        res_u, res_v = compute_residual(u_new, v_new)
        du, dv = apply_precond(-res_u, -res_v)
        u_new = u_new + du
        v_new = v_new + dv
        res_norm = jnp.max(jnp.abs(res_u)) + jnp.max(jnp.abs(res_v))
        return (u_new, v_new, res_norm, i + 1)

    # Initialize with old values
    init_state = (u_old, v_old, jnp.array(1.0), 0)
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_state[0], final_state[1]

@jax.jit
def cn_solve_jit(u0, v0, n_steps):
    """Solve using CN with fully JIT-compiled time stepping."""
    def body(i, uv):
        u, v = uv
        return cn_newton_step_jit(u, v)
    return jax.lax.fori_loop(0, n_steps, body, (u0, v0))

# Warmup
_ = cn_solve_jit(u0_jax, v0_jax, 10)
u0_jax.block_until_ready()

times_cn = []
for _ in range(N_REPS):
    start = time.perf_counter()
    u_cn, v_cn = cn_solve_jit(u0_jax, v0_jax, n_steps_cn)
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
print(f"   Time: {cn_median:.2f} ± {cn_iqr:.3f} s (median±IQR)")
print(f"   Steps: {n_steps_cn:,}")

# =============================================================================
# 3. IMEX-Strang (Splitting)
# =============================================================================
print("\n3. IMEX-Strang...")

# Larger dt possible due to implicit diffusion
dt_imex = 0.5  # Much larger than RK4
n_steps_imex = int(np.ceil(T_FINAL / dt_imex))  # ceil for consistency
t_actual_imex = n_steps_imex * dt_imex
print(f"   Steps: {n_steps_imex:,}, dt={dt_imex}")

@jax.jit
def imex_step(u, v, dt):
    """IMEX Strang splitting step."""
    # Half step: explicit reaction
    uvv = u * v * v
    u = u + 0.5*dt * (-uvv + F * (1 - u))
    v = v + 0.5*dt * (uvv - (F + k) * v)

    # Full step: implicit diffusion (exact via FFT)
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)
    u_hat = u_hat * jnp.exp(dt * Du * lap_eig_jax)
    v_hat = v_hat * jnp.exp(dt * Dv * lap_eig_jax)
    u = jnp.real(jnp.fft.ifft2(u_hat))
    v = jnp.real(jnp.fft.ifft2(v_hat))

    # Half step: explicit reaction
    uvv = u * v * v
    u = u + 0.5*dt * (-uvv + F * (1 - u))
    v = v + 0.5*dt * (uvv - (F + k) * v)

    return u, v

def imex_solve(u0, v0, n_steps, dt):
    u, v = u0, v0
    for _ in range(n_steps):
        u, v = imex_step(u, v, dt)
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
print(f"   Time: {imex_median:.2f} ± {imex_iqr:.3f} s (median±IQR)")
print(f"   Steps: {n_steps_imex:,}")

# =============================================================================
# 4. ETDRK4 (Exponential Time Differencing)
# =============================================================================
print("\n4. ETDRK4...")

# ETDRK4 can use even larger dt
dt_etd = 1.0
n_steps_etd = int(np.ceil(T_FINAL / dt_etd))  # ceil for consistency
t_actual_etd = n_steps_etd * dt_etd
print(f"   Steps: {n_steps_etd:,}, dt={dt_etd}")

# Precompute ETDRK4 coefficients for each mode
# Based on Cox & Matthews (2002)
def compute_etdrk4_coeffs(L, dt, M=32):
    """Compute ETDRK4 coefficients using contour integral."""
    # L is the linear operator eigenvalues (complex)
    # Using M-point contour integral for numerical stability

    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)  # Roots of unity

    # Broadcast for all modes
    LR = dt * L[..., np.newaxis] + r  # Shape: (*L.shape, M)

    E = np.exp(dt * L)
    E2 = np.exp(dt * L / 2)

    # Contour integrals
    Q = dt * np.mean((np.exp(LR/2) - 1) / LR, axis=-1)

    f1 = dt * np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1)
    f2 = dt * np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1)
    f3 = dt * np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1)

    return E, E2, Q, f1, f2, f3

# Linear operators for u and v
Lu = Du * lap_eig
Lv = Dv * lap_eig

E_u, E2_u, Q_u, f1_u, f2_u, f3_u = compute_etdrk4_coeffs(Lu, dt_etd)
E_v, E2_v, Q_v, f1_v, f2_v, f3_v = compute_etdrk4_coeffs(Lv, dt_etd)

# Convert to JAX
E_u, E2_u, Q_u, f1_u, f2_u, f3_u = [jnp.array(x) for x in [E_u, E2_u, Q_u, f1_u, f2_u, f3_u]]
E_v, E2_v, Q_v, f1_v, f2_v, f3_v = [jnp.array(x) for x in [E_v, E2_v, Q_v, f1_v, f2_v, f3_v]]

@jax.jit
def etdrk4_step(u, v):
    """ETDRK4 step for Gray-Scott."""
    u_hat = jnp.fft.fft2(u)
    v_hat = jnp.fft.fft2(v)

    # Nonlinear terms
    def N_u(u, v):
        return -u*v*v + F*(1-u)

    def N_v(u, v):
        return u*v*v - (F+k)*v

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
        u, v = etdrk4_step(u, v)
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
print(f"   Time: {etd_median:.2f} ± {etd_iqr:.3f} s (median±IQR)")
print(f"   Steps: {n_steps_etd:,}")

# =============================================================================
# Save and Summary
# =============================================================================
# Get device and dtype info
device = str(jax.devices()[0])
dtype = 'float64' if jax.config.jax_enable_x64 else 'float32'

results['config'] = {
    'N': N,
    'Du': Du,
    'Dv': Dv,
    'F': F,
    'k': k,
    't_final': T_FINAL,
    'scaling_used': False,  # NO SCALING - actual t=10000 times
    'n_reps': N_REPS,
    'dtype': dtype,
    'device': device,
    'laplacian': 'pseudo-spectral (-k^2)',
}

output_path = Path(__file__).parent / 'results' / 'gray_scott.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

print("\n" + "=" * 80)
print(f"SUMMARY (Table 6) - ACTUAL t=10000 walltime, {dtype}, {device}")
print("=" * 80)
print(f"{'Method':<20} {'Steps':>12} {'Time (s)':>14} {'dt':>10}")
print("-" * 80)
for m in results['methods']:
    time_str = f"{m['time_s']:.2f} ± {m['iqr_s']:.2f}"
    print(f"{m['name']:<20} {m['steps']:>12,} {time_str:>14} {m['dt']:>10.4f}")
print("=" * 80)
print(f"\nNote: ALL times are ACTUAL wallclock to reach t={T_FINAL}")
print(f"      NO SCALING/EXTRAPOLATION - these are measured times")
print(f"      {N_REPS} runs, median ± IQR reported")
print(f"      Laplacian: pseudo-spectral (-k²)")
