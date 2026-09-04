#!/usr/bin/env python3
"""
Benchmark: JIT x Device 2x2 Factorial (revised Table 6)

2D Diffusion with FFT-based Crank-Nicolson
Grid: 256x256, 1000 time steps

Measures 4 JAX configurations + NumPy reference:
  1. JAX CPU eager (no JIT, CPU)
  2. JAX CPU JIT  (JIT, CPU)
  3. JAX GPU eager (no JIT, GPU)
  4. JAX GPU JIT  (JIT, GPU)
  5. NumPy CPU (reference)

This isolates the JIT effect from the GPU effect.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from benchmark_utils import add_benchmark_args, compute_stats

parser = add_benchmark_args()
args = parser.parse_args()
N_REPS = args.n_reps

# Configuration
N = 256
N_STEPS = 1000
D = 0.01
dx = 1.0 / N
dt = 0.0001
N_WARMUP = 3

print("JIT x Device 2x2 Factorial Benchmark")
print("=" * 60)
print(f"Grid: {N}x{N}, Steps: {N_STEPS}, N_REPS: {N_REPS}")
print(f"JAX: {jax.__version__}, Devices: {jax.devices()}")
print("=" * 60)

# Setup wavenumbers (numpy)
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig_np = -(KX**2 + KY**2)

# Initial condition
x = np.linspace(0, 1, N, endpoint=False)
u0_np = np.sin(2 * np.pi * x[:, None]) * np.sin(2 * np.pi * x[None, :])

results = {}


# =============================================================================
# 1. NumPy CPU (reference)
# =============================================================================
print("\n1. NumPy (vectorized CPU)...")

def numpy_cn_solve(u0, n_steps, lap_eig, dt, D):
    u = u0.copy()
    for _ in range(n_steps):
        u_hat = np.fft.fft2(u)
        u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
        u = np.real(np.fft.ifft2(u_hat))
    return u

# Warmup
for _ in range(N_WARMUP):
    numpy_cn_solve(u0_np, 10, lap_eig_np, dt, D)

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    result = numpy_cn_solve(u0_np, N_STEPS, lap_eig_np, dt, D)
    times.append(time.perf_counter() - t0)

numpy_median, numpy_iqr = compute_stats(times)
results['numpy_cpu'] = {'median_s': numpy_median, 'iqr_s': numpy_iqr}
print(f"   {numpy_median:.3f} +/- {numpy_iqr:.3f} s")


# =============================================================================
# JAX CN step function (NOT jit-decorated, so we can control JIT per-config)
# =============================================================================
def jax_cn_step(u, lap_eig, dt, D):
    u_hat = jnp.fft.fft2(u)
    u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
    return jnp.real(jnp.fft.ifft2(u_hat))

jax_cn_step_jit = jax.jit(jax_cn_step)


def run_jax_config(label, device, use_jit):
    """Run one JAX configuration and return timing results."""
    step_fn = jax_cn_step_jit if use_jit else jax_cn_step

    with jax.default_device(device):
        u0_dev = jnp.array(u0_np)
        lap_dev = jnp.array(lap_eig_np)

        def solve():
            u = u0_dev
            for _ in range(N_STEPS):
                u = step_fn(u, lap_dev, dt, D)
            return u

        # Warmup (compile if JIT, trace if eager)
        for _ in range(N_WARMUP):
            r = solve()
            r.block_until_ready()

        # Measure compile time (first call after clearing cache)
        if use_jit:
            # Clear compiled cache for this config
            jax_cn_step_jit.clear_cache()
            t0 = time.perf_counter()
            r = solve()
            r.block_until_ready()
            compile_s = time.perf_counter() - t0
        else:
            compile_s = None

        # Re-warmup after cache clear
        for _ in range(N_WARMUP):
            r = solve()
            r.block_until_ready()

        # Timed runs
        times = []
        for _ in range(N_REPS):
            t0 = time.perf_counter()
            r = solve()
            r.block_until_ready()
            times.append(time.perf_counter() - t0)

    median, iqr = compute_stats(times)
    return {
        'median_s': float(median),
        'iqr_s': float(iqr),
        'compile_s': float(compile_s) if compile_s is not None else None,
        'ms_per_step': float(median * 1000 / N_STEPS),
    }


# =============================================================================
# 2. JAX CPU eager
# =============================================================================
print("\n2. JAX CPU eager (no JIT)...")
cpu_device = jax.devices('cpu')[0]
results['jax_cpu_eager'] = run_jax_config("JAX CPU eager", cpu_device, use_jit=False)
print(f"   {results['jax_cpu_eager']['median_s']:.3f} +/- {results['jax_cpu_eager']['iqr_s']:.3f} s")

# =============================================================================
# 3. JAX CPU JIT
# =============================================================================
print("\n3. JAX CPU JIT...")
results['jax_cpu_jit'] = run_jax_config("JAX CPU JIT", cpu_device, use_jit=True)
print(f"   {results['jax_cpu_jit']['median_s']:.3f} +/- {results['jax_cpu_jit']['iqr_s']:.3f} s")
if results['jax_cpu_jit']['compile_s']:
    print(f"   Compile: {results['jax_cpu_jit']['compile_s']:.2f} s")

# =============================================================================
# 4. JAX GPU eager
# =============================================================================
print("\n4. JAX GPU eager (no JIT)...")
gpu_device = jax.devices('gpu')[0]
results['jax_gpu_eager'] = run_jax_config("JAX GPU eager", gpu_device, use_jit=False)
print(f"   {results['jax_gpu_eager']['median_s']:.3f} +/- {results['jax_gpu_eager']['iqr_s']:.3f} s")

# =============================================================================
# 5. JAX GPU JIT
# =============================================================================
print("\n5. JAX GPU JIT...")
results['jax_gpu_jit'] = run_jax_config("JAX GPU JIT", gpu_device, use_jit=True)
print(f"   {results['jax_gpu_jit']['median_s']:.3f} +/- {results['jax_gpu_jit']['iqr_s']:.3f} s")
if results['jax_gpu_jit']['compile_s']:
    print(f"   Compile: {results['jax_gpu_jit']['compile_s']:.2f} s")


# =============================================================================
# Factor decomposition
# =============================================================================
cpu_eager = results['jax_cpu_eager']['median_s']
cpu_jit = results['jax_cpu_jit']['median_s']
gpu_eager = results['jax_gpu_eager']['median_s']
gpu_jit = results['jax_gpu_jit']['median_s']

results['factors'] = {
    'jit_factor_cpu': float(cpu_eager / cpu_jit),
    'jit_factor_gpu': float(gpu_eager / gpu_jit),
    'gpu_factor_eager': float(cpu_eager / gpu_eager),
    'gpu_factor_jit': float(cpu_jit / gpu_jit),
    'vs_numpy': float(numpy_median / gpu_jit),
}

results['config'] = {
    'grid_size': N,
    'n_steps': N_STEPS,
    'n_reps': N_REPS,
    'D': D,
    'dt': dt,
    'dtype': 'float64',
    'jax_version': jax.__version__,
    'gpu': str(jax.devices('gpu')[0]),
    'cpu': str(jax.devices('cpu')[0]),
}

print("\n" + "=" * 60)
print("SUMMARY: 2x2 Factorial")
print("=" * 60)
print(f"{'Config':<25} {'Time (s)':>12} {'vs NumPy':>10}")
print("-" * 50)
print(f"{'NumPy CPU (ref)':<25} {numpy_median:>10.3f}   {'1.0x':>8}")
print(f"{'JAX CPU eager':<25} {cpu_eager:>10.3f}   {numpy_median/cpu_eager:>7.1f}x")
print(f"{'JAX CPU JIT':<25} {cpu_jit:>10.3f}   {numpy_median/cpu_jit:>7.1f}x")
print(f"{'JAX GPU eager':<25} {gpu_eager:>10.3f}   {numpy_median/gpu_eager:>7.1f}x")
print(f"{'JAX GPU JIT':<25} {gpu_jit:>10.3f}   {numpy_median/gpu_jit:>7.1f}x")
print()
print(f"JIT factor (CPU):  {results['factors']['jit_factor_cpu']:.2f}x")
print(f"JIT factor (GPU):  {results['factors']['jit_factor_gpu']:.2f}x")
print(f"GPU factor (eager): {results['factors']['gpu_factor_eager']:.2f}x")
print(f"GPU factor (JIT):  {results['factors']['gpu_factor_jit']:.2f}x")
print("=" * 60)

# Save
output_path = Path(__file__).parent / 'results' / 'jit_factorial.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
