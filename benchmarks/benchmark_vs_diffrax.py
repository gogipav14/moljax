#!/usr/bin/env python3
"""
Benchmark: moljax FFT-CN vs Diffrax solvers for 2D diffusion.

This is the critical comparison - Diffrax is the established JAX ODE/PDE library.
We need to show whether FFT-diagonalized implicit solves provide actual speedup
over Diffrax's general-purpose implicit solvers.

Problem: 2D diffusion equation
    du/dt = D * (d²u/dx² + d²u/dy²)
    Domain: [0,1]² with periodic BC
    IC: sin(2πx) * sin(2πy)
    Analytical: exp(-8π²Dt) * sin(2πx) * sin(2πy)
"""

import time
import json
from pathlib import Path
import numpy as np

import jax
import jax.numpy as jnp
from jax import jit
import diffrax

print("=" * 70)
print("Benchmark: moljax FFT-CN vs Diffrax")
print("=" * 70)
print(f"JAX devices: {jax.devices()}")
print(f"Diffrax version: {diffrax.__version__}")
print()

# =============================================================================
# Problem Setup
# =============================================================================

D = 0.1  # Diffusion coefficient
T_FINAL = 0.1
N_WARMUP = 3
N_RUNS = 5


def analytical_solution(X, Y, t, D):
    """Exact solution for 2D diffusion with sin(2πx)*sin(2πy) IC."""
    return jnp.exp(-8 * jnp.pi**2 * D * t) * jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)


# =============================================================================
# Method 1: FFT-CN (moljax approach) - O(N log N) implicit solve
# =============================================================================

def make_fft_cn_solver(N, D, dt):
    """Create JIT-compiled FFT-CN stepper."""
    dx = 1.0 / N

    # Wavenumbers
    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)

    # CN amplification factor
    cn_factor = (1 + 0.5 * dt * D * lap_eig) / (1 - 0.5 * dt * D * lap_eig)

    @jit
    def step(u):
        u_hat = jnp.fft.fft2(u)
        u_hat = u_hat * cn_factor
        return jnp.real(jnp.fft.ifft2(u_hat))

    return step


def solve_fft_cn(u0, D, dt, n_steps):
    """Solve with FFT-CN."""
    N = u0.shape[0]
    step = make_fft_cn_solver(N, D, dt)

    u = u0
    for _ in range(n_steps):
        u = step(u)
    return u


@jit
def solve_fft_cn_jit(u0, D, dt, n_steps):
    """Fully JIT-compiled FFT-CN with lax.fori_loop."""
    N = u0.shape[0]
    dx = 1.0 / N

    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)
    cn_factor = (1 + 0.5 * dt * D * lap_eig) / (1 - 0.5 * dt * D * lap_eig)

    def body(_, u):
        u_hat = jnp.fft.fft2(u)
        u_hat = u_hat * cn_factor
        return jnp.real(jnp.fft.ifft2(u_hat))

    return jax.lax.fori_loop(0, n_steps, body, u0)


# =============================================================================
# Method 2: Diffrax with spectral Laplacian (fairest comparison)
# =============================================================================

def make_diffrax_spectral_rhs(N, D):
    """RHS using spectral Laplacian - same spatial accuracy as FFT-CN."""
    dx = 1.0 / N
    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)

    def rhs(t, u, args):
        u_hat = jnp.fft.fft2(u)
        lap_u_hat = lap_eig * u_hat
        lap_u = jnp.real(jnp.fft.ifft2(lap_u_hat))
        return D * lap_u

    return rhs


def make_diffrax_fd_rhs(N, D):
    """RHS using 2nd-order finite differences - typical Diffrax usage."""
    dx = 1.0 / N

    def rhs(t, u, args):
        # Periodic Laplacian via FD
        lap_u = (
            jnp.roll(u, 1, axis=0) + jnp.roll(u, -1, axis=0) +
            jnp.roll(u, 1, axis=1) + jnp.roll(u, -1, axis=1) - 4 * u
        ) / dx**2
        return D * lap_u

    return rhs


def solve_diffrax(u0, D, T_final, solver, stepsize_controller, rhs_type='spectral'):
    """Solve with Diffrax."""
    N = u0.shape[0]

    if rhs_type == 'spectral':
        rhs = make_diffrax_spectral_rhs(N, D)
    else:
        rhs = make_diffrax_fd_rhs(N, D)

    term = diffrax.ODETerm(rhs)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T_final,
        dt0=0.001,
        y0=u0,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=100000,
    )

    return sol.ys[0]


# =============================================================================
# Benchmark Runner
# =============================================================================

def benchmark_method(name, solve_fn, u0, analytical, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Benchmark a solver method."""
    print(f"\n{name}...")

    # Warmup
    for _ in range(n_warmup):
        _ = solve_fn()
        jax.block_until_ready(_)

    # Timed runs
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result = solve_fn()
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.4f} s")

    result = solve_fn()
    jax.block_until_ready(result)
    error = float(jnp.max(jnp.abs(result - analytical)))

    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"  Mean: {mean_time:.4f} ± {std_time:.4f} s, Error: {error:.2e}")

    return {
        'mean_s': mean_time,
        'std_s': std_time,
        'times_s': times,
        'error': error,
    }


# =============================================================================
# Run Benchmarks
# =============================================================================

results = {}

for N in [64, 128, 256]:
    print("\n" + "=" * 70)
    print(f"Grid size: {N} × {N} ({N**2} DOFs)")
    print("=" * 70)

    # Setup
    dx = 1.0 / N
    x = jnp.linspace(0, 1, N, endpoint=False)
    y = jnp.linspace(0, 1, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    u0 = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)
    analytical = analytical_solution(X, Y, T_FINAL, D)

    grid_results = {}

    # --- FFT-CN (fixed dt) ---
    dt = 0.001
    n_steps = int(T_FINAL / dt)
    grid_results['fft_cn'] = benchmark_method(
        f"FFT-CN (dt={dt}, {n_steps} steps)",
        lambda: solve_fft_cn_jit(u0, D, dt, n_steps),
        u0, analytical
    )

    # --- Diffrax Tsit5 (explicit, adaptive) with spectral Laplacian ---
    grid_results['diffrax_tsit5_spectral'] = benchmark_method(
        "Diffrax Tsit5 (spectral Laplacian, adaptive)",
        lambda: solve_diffrax(u0, D, T_FINAL,
                             diffrax.Tsit5(),
                             diffrax.PIDController(rtol=1e-5, atol=1e-7),
                             rhs_type='spectral'),
        u0, analytical
    )

    # --- Diffrax Dopri5 (explicit, adaptive) with spectral Laplacian ---
    grid_results['diffrax_dopri5_spectral'] = benchmark_method(
        "Diffrax Dopri5 (spectral Laplacian, adaptive)",
        lambda: solve_diffrax(u0, D, T_FINAL,
                             diffrax.Dopri5(),
                             diffrax.PIDController(rtol=1e-5, atol=1e-7),
                             rhs_type='spectral'),
        u0, analytical
    )

    # --- Diffrax Tsit5 with FD Laplacian (typical usage) ---
    grid_results['diffrax_tsit5_fd'] = benchmark_method(
        "Diffrax Tsit5 (FD Laplacian, adaptive)",
        lambda: solve_diffrax(u0, D, T_FINAL,
                             diffrax.Tsit5(),
                             diffrax.PIDController(rtol=1e-5, atol=1e-7),
                             rhs_type='fd'),
        u0, analytical
    )

    results[N] = grid_results

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: FFT-CN vs Diffrax")
print("=" * 70)

print(f"\n{'Grid':<8} {'Method':<35} {'Time (s)':<12} {'Error':<12} {'vs FFT-CN':<10}")
print("-" * 80)

for N in results:
    fft_time = results[N]['fft_cn']['mean_s']
    for method, data in results[N].items():
        speedup = data['mean_s'] / fft_time if method != 'fft_cn' else 1.0
        speedup_str = f"{speedup:.2f}x slower" if speedup > 1 else f"{1/speedup:.2f}x faster"
        if method == 'fft_cn':
            speedup_str = "(baseline)"
        print(f"{N:<8} {method:<35} {data['mean_s']:<12.4f} {data['error']:<12.2e} {speedup_str:<10}")
    print()

# =============================================================================
# Save Results
# =============================================================================

output = {
    'config': {
        'D': D,
        'T_final': T_FINAL,
        'n_warmup': N_WARMUP,
        'n_runs': N_RUNS,
    },
    'results': {str(k): v for k, v in results.items()},
}

output_dir = Path(__file__).parent / 'results'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'diffrax_comparison.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved to: {output_path}")
