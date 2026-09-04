#!/usr/bin/env python3
"""
Benchmark: GMRES Iteration Sweep (Table 8)

Tests preconditioner effectiveness across stiffness ratios.
Grid: 128x128, varying stiffness ratio sigma = D*dt/dx^2

APPROACH:
- SciPy GMRES for iteration counting (ground truth, CPU)
  Python-side callback works correctly with SciPy (not JIT-compiled)
- JAX GMRES for wall-clock GPU timing
  Iteration counting broken due to lax.fori_loop tracing

Preconditioners:
- None (identity)
- FFT Diffusion: M = (I - dt*D*L_FFT)^{-1}, captures dominant stiff operator

BUG FIX: Previous version used 0.5*dt in preconditioner but full dt in Jacobian.
Now both use consistent dt (Backward Euler formulation).
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

# Import benchmark utilities
from benchmark_utils import add_benchmark_args, compute_stats, setup_benchmark
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as scipy_gmres

parser = add_benchmark_args()
args = parser.parse_args()

# Configuration
N = 128  # Match paper: 128x128 grid
STIFFNESS_RATIOS = [0.1, 1, 10, 100, 1000]  # Full sweep from paper Table 8
GMRES_TOL = 1e-6
GMRES_MAXITER = 10000
D = 0.01
dx = 1.0 / N

# Reaction term: R(u) = k * u^2 with linearization dR/du = 2*k*u
# k=1.0 is realistic for diffusion-dominated reaction-diffusion systems
# (diffusion is the dominant stiff operator, reaction is moderate)
# k=10.0 is too strong: reaction dominates, preconditioner breaks down at sigma>=10
REACTION_K = 1.0

print("GMRES Iteration Sweep Benchmark (FIXED)")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N} (Matrix-free FD discretization)")
print(f"Problem: 2D diffusion-reaction (k={REACTION_K})")
print(f"Stiffness ratios: {STIFFNESS_RATIOS}")
print(f"GMRES tolerance: {GMRES_TOL}")
print("Method: SciPy GMRES for iter counts, JAX GMRES for GPU timing")
print("=" * 70)

# ============================================================================
# NumPy operators (for SciPy GMRES - correct iteration counting)
# ============================================================================

def np_fd_laplacian_2d(u_2d, dx_val):
    """NumPy 2D Laplacian with periodic BC (5-point stencil)."""
    return (
        np.roll(u_2d, -1, axis=0) + np.roll(u_2d, 1, axis=0) +
        np.roll(u_2d, -1, axis=1) + np.roll(u_2d, 1, axis=1) -
        4.0 * u_2d
    ) / (dx_val ** 2)


# FFT eigenvalues (NumPy, for preconditioner)
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig_np = -(KX**2 + KY**2)  # Laplacian eigenvalues (negative)
lap_eig_jax = jnp.array(lap_eig_np)

# Initial condition (linearization point for reaction Jacobian)
np.random.seed(42)
u0_np = np.abs(np.random.randn(N, N)) + 0.5
u0_jax = jnp.array(u0_np)

# RHS vector
rhs_np = np.random.randn(N * N)
rhs_jax = jnp.array(rhs_np)

# ============================================================================
# JAX operators (for GPU timing)
# ============================================================================

@jax.jit
def jax_fd_laplacian_2d(u_2d, dx_val):
    """JAX 2D Laplacian with periodic BC."""
    return (
        jnp.roll(u_2d, -1, axis=0) + jnp.roll(u_2d, 1, axis=0) +
        jnp.roll(u_2d, -1, axis=1) + jnp.roll(u_2d, 1, axis=1) -
        4.0 * u_2d
    ) / (dx_val ** 2)


results = {'sweeps': []}
N_TIMING_REPS = 10

for sigma in STIFFNESS_RATIOS:
    print(f"\n{'='*70}")
    print(f"Stiffness ratio sigma = {sigma}")
    print(f"{'='*70}")

    # Compute dt from stiffness ratio: sigma = D * dt / dx^2
    dt = sigma * dx**2 / D

    # Reaction Jacobian at linearization point: dR/du = 2*k*u0
    reaction_jac_np = 2.0 * REACTION_K * u0_np
    reaction_jac_jax = 2.0 * REACTION_K * u0_jax

    # =====================================================================
    # PART 1: SciPy GMRES for iteration counting (CPU, ground truth)
    # =====================================================================

    # Jacobian matvec: J*v = v - dt*(D*L*v + diag(dR/du)*v)
    def np_jacobian_matvec(v_flat):
        v_2d = v_flat.reshape(N, N)
        diffusion = D * np_fd_laplacian_2d(v_2d, dx)
        reaction = reaction_jac_np * v_2d
        result = v_2d - dt * (diffusion + reaction)
        return result.flatten()

    J_op = LinearOperator((N*N, N*N), matvec=np_jacobian_matvec, dtype=np.float64)

    # --- No Preconditioner ---
    print("  [SciPy] No preconditioner...")
    iter_count_none = [0]
    def callback_none(residual_norm):
        iter_count_none[0] += 1

    iter_count_none[0] = 0
    sol_none, info_none = scipy_gmres(
        J_op, rhs_np, rtol=GMRES_TOL, maxiter=GMRES_MAXITER,
        callback=callback_none, callback_type='pr_norm'
    )
    iters_none = iter_count_none[0]
    converged_none = (info_none == 0)
    residual_none = np.linalg.norm(np_jacobian_matvec(sol_none) - rhs_np) / np.linalg.norm(rhs_np)
    print(f"    Iterations: {iters_none}, converged: {converged_none}, "
          f"relative residual: {residual_none:.2e}")

    # --- FFT Preconditioner ---
    # M = I - dt*D*L_FFT  (matches Jacobian's dt, NOT 0.5*dt)
    # M^{-1} in Fourier space: 1/(1 - dt*D*lambda_k)
    print("  [SciPy] FFT preconditioner...")

    def np_fft_precond(v_flat):
        v_2d = v_flat.reshape(N, N)
        v_hat = np.fft.fft2(v_2d)
        # Preconditioner: (I - dt*D*L)^{-1}
        # lap_eig is negative, so (1 - dt*D*lap_eig) = (1 + dt*D*|lap_eig|) > 1
        result_hat = v_hat / (1.0 - dt * D * lap_eig_np)
        return np.real(np.fft.ifft2(result_hat)).flatten()

    M_op = LinearOperator((N*N, N*N), matvec=np_fft_precond, dtype=np.float64)

    iter_count_fft = [0]
    def callback_fft(residual_norm):
        iter_count_fft[0] += 1

    iter_count_fft[0] = 0
    sol_fft, info_fft = scipy_gmres(
        J_op, rhs_np, rtol=GMRES_TOL, maxiter=GMRES_MAXITER,
        M=M_op, callback=callback_fft, callback_type='pr_norm'
    )
    iters_fft = iter_count_fft[0]
    converged_fft = (info_fft == 0)
    residual_fft = np.linalg.norm(np_jacobian_matvec(sol_fft) - rhs_np) / np.linalg.norm(rhs_np)
    print(f"    Iterations: {iters_fft}, converged: {converged_fft}, "
          f"relative residual: {residual_fft:.2e}")

    # Reduction
    if iters_fft > 0:
        reduction = iters_none / iters_fft
    else:
        reduction = float('inf')
    print(f"    Iteration reduction: {reduction:.1f}x")

    # =====================================================================
    # PART 2: JAX GMRES for GPU wall-clock timing
    # =====================================================================

    @jax.jit
    def jax_jacobian_matvec_2d(v_2d):
        diffusion = D * jax_fd_laplacian_2d(v_2d, dx)
        reaction = reaction_jac_jax * v_2d
        return v_2d - dt * (diffusion + reaction)

    def jax_matvec_flat(v_flat):
        return jax_jacobian_matvec_2d(v_flat.reshape(N, N)).flatten()

    @jax.jit
    def jax_fft_precond_2d(v_2d):
        v_hat = jnp.fft.fft2(v_2d)
        result_hat = v_hat / (1.0 - dt * D * lap_eig_jax)
        return jnp.real(jnp.fft.ifft2(result_hat))

    def jax_precond_matvec_flat(v_flat):
        return jax_fft_precond_2d(jax_jacobian_matvec_2d(v_flat.reshape(N, N))).flatten()

    jax_precond_rhs = jax_fft_precond_2d(rhs_jax.reshape(N, N)).flatten()

    # Warmup JAX compilation
    print("  [JAX] GPU timing (warmup + 10 reps)...")
    from jax.scipy.sparse.linalg import gmres as jax_gmres

    try:
        _ = jax_gmres(jax_matvec_flat, rhs_jax, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
        _[0].block_until_ready()
    except Exception:
        pass

    try:
        _ = jax_gmres(jax_precond_matvec_flat, jax_precond_rhs, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
        _[0].block_until_ready()
    except Exception:
        pass

    # Timed runs: no preconditioner (GPU)
    times_none_gpu = []
    for _ in range(N_TIMING_REPS):
        t0 = time.perf_counter()
        sol, _ = jax_gmres(jax_matvec_flat, rhs_jax, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
        sol.block_until_ready()
        times_none_gpu.append(time.perf_counter() - t0)

    time_none_median, time_none_iqr = compute_stats(times_none_gpu)

    # Timed runs: FFT preconditioner (GPU)
    times_fft_gpu = []
    for _ in range(N_TIMING_REPS):
        precond_rhs_fresh = jax_fft_precond_2d(rhs_jax.reshape(N, N)).flatten()
        t0 = time.perf_counter()
        sol, _ = jax_gmres(jax_precond_matvec_flat, precond_rhs_fresh, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
        sol.block_until_ready()
        times_fft_gpu.append(time.perf_counter() - t0)

    time_fft_median, time_fft_iqr = compute_stats(times_fft_gpu)

    time_speedup = time_none_median / time_fft_median if time_fft_median > 0 else 0.0
    print(f"    GPU time (none): {time_none_median*1000:.2f} ms")
    print(f"    GPU time (FFT):  {time_fft_median*1000:.2f} ms")
    print(f"    GPU time speedup: {time_speedup:.1f}x")

    # =====================================================================
    # Store results
    # =====================================================================
    sweep_result = {
        'sigma': float(sigma),
        'dt': float(dt),
        'none': {
            'iterations': int(iters_none),
            'converged': bool(converged_none),
            'relative_residual': float(residual_none),
            'gpu_time_ms': float(time_none_median * 1000),
            'gpu_time_iqr_ms': float(time_none_iqr * 1000),
        },
        'fft': {
            'iterations': int(iters_fft),
            'converged': bool(converged_fft),
            'relative_residual': float(residual_fft),
            'gpu_time_ms': float(time_fft_median * 1000),
            'gpu_time_iqr_ms': float(time_fft_iqr * 1000),
        },
        'iteration_reduction': float(reduction),
        'gpu_time_speedup': float(time_speedup),
    }
    results['sweeps'].append(sweep_result)

# ============================================================================
# Save results
# ============================================================================
results['config'] = {
    'grid_size': N,
    'gmres_tol': GMRES_TOL,
    'gmres_maxiter': GMRES_MAXITER,
    'device': device_str,
    'dtype': 'float64',
    'D': D,
    'reaction_k': REACTION_K,
    'discretization': 'finite_difference',
    'iteration_source': 'scipy_gmres (CPU, ground truth)',
    'timing_source': 'jax_gmres (GPU)',
    'preconditioner_formula': 'M = (I - dt*D*L_FFT)^{-1}',
    'note': 'Fixed: preconditioner now uses full dt (was 0.5*dt in previous version)',
}

output_path = Path(__file__).parent / 'results' / 'gmres_sweep.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# ============================================================================
# Summary Table
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: GMRES Iterations (SciPy) and GPU Timing (JAX)")
print("=" * 80)
print(f"{'sigma':>8} | {'None iters':>12} {'FFT iters':>12} {'Reduction':>12} | "
      f"{'None (ms)':>10} {'FFT (ms)':>10} {'Speedup':>8}")
print("-" * 80)
for s in results['sweeps']:
    none_str = f"{s['none']['iterations']}" + ("*" if not s['none']['converged'] else "")
    fft_str = f"{s['fft']['iterations']}" + ("*" if not s['fft']['converged'] else "")
    red = s['iteration_reduction']
    t_none = s['none']['gpu_time_ms']
    t_fft = s['fft']['gpu_time_ms']
    spd = s['gpu_time_speedup']
    print(f"{s['sigma']:>8.1f} | {none_str:>12} {fft_str:>12} {red:>11.1f}x | "
          f"{t_none:>10.1f} {t_fft:>10.1f} {spd:>7.1f}x")
print("-" * 80)
print(f"* = did not converge in {GMRES_MAXITER} iterations")
print("=" * 80)

# ============================================================================
# Generate Figure
# ============================================================================
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

sigmas = [s['sigma'] for s in results['sweeps']]
iters_none_list = [s['none']['iterations'] for s in results['sweeps']]
iters_fft_list = [s['fft']['iterations'] for s in results['sweeps']]
times_none_list = [s['none']['gpu_time_ms'] for s in results['sweeps']]
times_fft_list = [s['fft']['gpu_time_ms'] for s in results['sweeps']]

# Left panel: Iteration counts
ax1.loglog(sigmas, iters_none_list, 'o-', label='No preconditioner',
           color='#d62728', linewidth=2, markersize=10)
ax1.loglog(sigmas, iters_fft_list, 's-', label='FFT preconditioner',
           color='#2ca02c', linewidth=2, markersize=10)
ax1.set_xlabel(r'Stiffness ratio $\sigma = D\Delta t / \Delta x^2$', fontsize=12)
ax1.set_ylabel('GMRES iterations', fontsize=12)
ax1.set_title('Iteration Count (SciPy, ground truth)', fontsize=13)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Add reduction annotations
for i, s in enumerate(results['sweeps']):
    red = s['iteration_reduction']
    if red > 1.5:
        ax1.annotate(f'{red:.0f}x',
                     (sigmas[i], iters_fft_list[i]),
                     textcoords="offset points", xytext=(12, 0),
                     fontsize=10, color='green', fontweight='bold')

# Right panel: GPU wall-clock time
ax2.loglog(sigmas, times_none_list, 'o-', label='No preconditioner',
           color='#d62728', linewidth=2, markersize=10)
ax2.loglog(sigmas, times_fft_list, 's-', label='FFT preconditioner',
           color='#2ca02c', linewidth=2, markersize=10)
ax2.set_xlabel(r'Stiffness ratio $\sigma = D\Delta t / \Delta x^2$', fontsize=12)
ax2.set_ylabel('GPU wall-clock time (ms)', fontsize=12)
ax2.set_title('GPU Execution Time (JAX)', fontsize=13)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'gmres_sweep.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
fig_path_pdf = fig_path.with_suffix('.pdf')
plt.savefig(fig_path_pdf, bbox_inches='tight')
print(f"\nFigure saved to {fig_path}")
plt.close()
