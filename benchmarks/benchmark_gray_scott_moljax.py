#!/usr/bin/env python3
"""
Gray-Scott Benchmark using ACTUAL moljax library.

This uses the proper moljax API with JIT-compiled adaptive integrators,
Newton-Krylov solvers, and FFT preconditioners.

Grid: 256x256, t ∈ [0, 10000] (scaled from t=100 benchmark)
"""

import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Add moljax to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "moljax_public"))

from moljax.core.bc import BCType
from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.grid import Grid2D
from moljax.core.model import create_gray_scott_model, create_gray_scott_periodic_fft
from moljax.core.newton_krylov import NKParams
from moljax.core.preconditioners import create_gray_scott_fft_preconditioner
from moljax.core.stepping import (
    IntegratorType,
    adaptive_integrate,
    adaptive_integrate_imex,
)

print("=" * 70)
print("Gray-Scott Benchmark (moljax library)")
print("=" * 70)
print(f"JAX devices: {jax.devices()}")

# Configuration
N = 256
T_BENCH = 100.0  # Benchmark time
T_PAPER = 10000.0  # Paper time
SCALE_FACTOR = T_PAPER / T_BENCH

# Gray-Scott parameters (spot pattern)
Du, Dv = 2e-5, 1e-5
F, k = 0.035, 0.065

# Grid setup
Lx, Ly = 1.0, 1.0
grid = Grid2D.uniform(N, N, 0.0, Lx, 0.0, Ly, n_ghost=1)

print(f"\nGrid: {N}x{N}")
print(f"Du={Du}, Dv={Dv}, F={F}, k={k}")
print(f"Benchmark time: {T_BENCH}, Paper time: {T_PAPER}")
print(f"dx = {grid.dx:.6f}")

# Create models
model = create_gray_scott_model(
    grid=grid, Du=Du, Dv=Dv, F=F, k=k,
    bc_type=BCType.PERIODIC, dtype=jnp.float64
)

model_fft, fft_cache, diffusivities = create_gray_scott_periodic_fft(
    grid=grid, Du=Du, Dv=Dv, F=F, k=k, dtype=jnp.float64
)

# Initial condition
def create_ic(grid, dtype=jnp.float64):
    X, Y = grid.meshgrid(include_ghost=True)
    u = jnp.ones((grid.ny_total, grid.nx_total), dtype=dtype)
    v = jnp.zeros((grid.ny_total, grid.nx_total), dtype=dtype)

    # Square perturbation in center
    cx, cy = 0.5 * Lx, 0.5 * Ly
    mask = (jnp.abs(X - cx) < 0.05 * Lx) & (jnp.abs(Y - cy) < 0.05 * Ly)
    u = jnp.where(mask, 0.5, u)
    v = jnp.where(mask, 0.25, v)

    # Small noise
    key = jax.random.PRNGKey(42)
    u = u + 0.01 * jax.random.uniform(key, u.shape, minval=-1, maxval=1)
    v = v + 0.01 * jax.random.uniform(jax.random.split(key)[0], v.shape, minval=-1, maxval=1)

    return {'u': u.astype(dtype), 'v': v.astype(dtype)}

y0 = create_ic(grid)

results = {'methods': []}

# CFL dt for explicit
dt_cfl = 0.25 * grid.min_dx2 / max(Du, Dv)
print(f"\nDiffusion CFL dt ~ {dt_cfl:.6f}")

# =============================================================================
# 1. RK4 (Explicit) - short run due to CFL
# =============================================================================
print("\n1. RK4 (explicit, CFL-limited)...")

cfl_params = CFLParams(safety=0.9, cfl_diffusion=0.25)
pid_params = PIDParams(safety=0.9, atol=1e-4, rtol=1e-3, dt_min=1e-8, dt_max=dt_cfl)

# Warmup
_ = adaptive_integrate(
    model=model, y0=y0, t0=0.0, t_end=1.0, dt0=dt_cfl*0.5,
    method=IntegratorType.RK4, max_steps=1000,
    cfl_params=cfl_params, pid_params=pid_params, save_every=10000
)

start = time.perf_counter()
result_rk4 = adaptive_integrate(
    model=model, y0=y0, t0=0.0, t_end=T_BENCH, dt0=dt_cfl*0.5,
    method=IntegratorType.RK4, max_steps=500000,
    cfl_params=cfl_params, pid_params=pid_params, save_every=10000
)
jax.block_until_ready(result_rk4.y_final['u'])
rk4_time = time.perf_counter() - start
rk4_scaled = rk4_time * SCALE_FACTOR

print(f"   Bench time: {rk4_time:.2f}s, Scaled: {rk4_scaled:.0f}s")
print(f"   Steps: {int(result_rk4.n_accepted)}")

results['methods'].append({
    'name': 'RK4 (explicit)',
    'time_s': float(rk4_scaled),
    'time_bench_s': float(rk4_time),
    'steps': int(result_rk4.n_accepted * SCALE_FACTOR),
})

# =============================================================================
# 2. CN + FFT Preconditioner
# =============================================================================
print("\n2. CN (Newton-Krylov + FFT precond)...")

pid_params_cn = PIDParams(safety=0.9, atol=1e-4, rtol=1e-3, dt_min=1e-6, dt_max=10.0)
nk_params = NKParams(max_newton_iters=10, max_krylov_iters=50, newton_tol=1e-6, krylov_tol=1e-5)
precond = create_gray_scott_fft_preconditioner(fft_cache)

# Warmup
_ = adaptive_integrate(
    model=model, y0=y0, t0=0.0, t_end=10.0, dt0=1.0,
    method=IntegratorType.CN, max_steps=100,
    pid_params=pid_params_cn, preconditioner=precond, nk_params=nk_params, save_every=10000
)

start = time.perf_counter()
result_cn = adaptive_integrate(
    model=model, y0=y0, t0=0.0, t_end=T_BENCH, dt0=1.0,
    method=IntegratorType.CN, max_steps=10000,
    pid_params=pid_params_cn, preconditioner=precond, nk_params=nk_params, save_every=10000
)
jax.block_until_ready(result_cn.y_final['u'])
cn_time = time.perf_counter() - start
cn_scaled = cn_time * SCALE_FACTOR

print(f"   Bench time: {cn_time:.2f}s, Scaled: {cn_scaled:.0f}s")
print(f"   Steps: {int(result_cn.n_accepted)}, Rejected: {int(result_cn.n_rejected)}")

results['methods'].append({
    'name': 'CN + FFT precond',
    'time_s': float(cn_scaled),
    'time_bench_s': float(cn_time),
    'steps': int(result_cn.n_accepted * SCALE_FACTOR),
})

# =============================================================================
# 3. IMEX-Strang
# =============================================================================
print("\n3. IMEX-Strang...")

pid_params_imex = PIDParams(safety=0.9, atol=1e-5, rtol=1e-4, dt_min=1e-6, dt_max=5.0)

# Warmup
_ = adaptive_integrate_imex(
    model=model_fft, y0=y0, t0=0.0, t_end=10.0, dt0=0.5,
    fft_cache=fft_cache, diffusivities=diffusivities, use_strang=True,
    max_steps=100, pid_params=pid_params_imex, save_every=10000
)

start = time.perf_counter()
result_imex = adaptive_integrate_imex(
    model=model_fft, y0=y0, t0=0.0, t_end=T_BENCH, dt0=0.5,
    fft_cache=fft_cache, diffusivities=diffusivities, use_strang=True,
    max_steps=10000, pid_params=pid_params_imex, save_every=10000
)
jax.block_until_ready(result_imex.y_final['u'])
imex_time = time.perf_counter() - start
imex_scaled = imex_time * SCALE_FACTOR

print(f"   Bench time: {imex_time:.2f}s, Scaled: {imex_scaled:.0f}s")
print(f"   Steps: {int(result_imex.n_accepted)}, Rejected: {int(result_imex.n_rejected)}")

results['methods'].append({
    'name': 'IMEX-Strang',
    'time_s': float(imex_scaled),
    'time_bench_s': float(imex_time),
    'steps': int(result_imex.n_accepted * SCALE_FACTOR),
})

# =============================================================================
# Save Results
# =============================================================================
results['config'] = {
    'N': N,
    'Du': Du, 'Dv': Dv, 'F': F, 'k': k,
    't_bench': T_BENCH, 't_paper': T_PAPER,
    'scale_factor': SCALE_FACTOR,
    'library': 'moljax',
}

output_path = Path(__file__).parent / 'results' / 'gray_scott_moljax.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY (scaled to t=10000)")
print("=" * 70)
print(f"{'Method':<25} {'Time (s)':>12} {'Steps':>10}")
print("-" * 70)
for m in results['methods']:
    print(f"{m['name']:<25} {m['time_s']:>12.0f} {m['steps']:>10,}")
print("=" * 70)
print(f"\nResults saved to {output_path}")
