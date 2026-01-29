#!/usr/bin/env python3
"""
Benchmark: GMRES Iteration Sweep (Table 8)

Tests preconditioner effectiveness across stiffness ratios.
Grid: 128x128, varying stiffness ratio sigma = D*dt/dx^2

Uses matrix-free FINITE DIFFERENCE matvec to show
preconditioner impact.

Preconditioners:
- None (identity)
- FFT Diffusion (approximate inverse via FFT)

NOTE: This benchmark measures ACTUAL wall-clock times, not estimates.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
import jax.numpy as jnp
from jax.scipy.sparse.linalg import gmres

# Import benchmark utilities
from benchmark_utils import (
    setup_benchmark, check_finite_tree, compute_stats,
    add_benchmark_args, DEFAULT_N_REPS
)

# Parse CLI arguments (n_reps not used for GMRES sweep, but include for consistency)
parser = add_benchmark_args()
args = parser.parse_args()

# Configuration
N = 128  # Match paper: 128x128 grid
STIFFNESS_RATIOS = [1, 10, 100]  # Reduced set for faster testing
GMRES_TOL = 1e-6
GMRES_MAXITER = 10000
D = 0.01
dx = 1.0 / N

# Nonlinear reaction coefficient (creates challenging off-diagonal Jacobian structure)
# Reaction term: R(u) = k * u^2 with linearization dR/du = 2*k*u
# Higher k = more nonlinearity = more GMRES iterations needed
REACTION_K = 10.0  # Reaction rate coefficient (moderate for reasonable iterations)

print("GMRES Iteration Sweep Benchmark")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N} (Matrix-free FD discretization)")
print(f"Problem: 2D diffusion with nonlinear reaction (k={REACTION_K})")
print(f"Stiffness ratios: {STIFFNESS_RATIOS}")
print(f"GMRES tolerance: {GMRES_TOL}")
print("=" * 60)

# Matrix-free FD Laplacian matvec (memory efficient)
@jax.jit
def fd_laplacian_matvec(u_2d, dx_val):
    """Matrix-free 2D Laplacian with periodic BC."""
    # 5-point stencil: (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx^2
    laplacian = (
        jnp.roll(u_2d, -1, axis=0) + jnp.roll(u_2d, 1, axis=0) +
        jnp.roll(u_2d, -1, axis=1) + jnp.roll(u_2d, 1, axis=1) -
        4.0 * u_2d
    ) / (dx_val ** 2)
    return laplacian

# FFT eigenvalues for preconditioning
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Initial condition (used for Jacobian linearization point)
np.random.seed(42)
u0 = np.abs(np.random.randn(N, N)) + 0.5  # Positive values for reaction
u0_jax = jnp.array(u0)

# RHS for GMRES
rhs_2d = jnp.array(np.random.randn(N, N))

results = {'sweeps': []}
N_TIMING_REPS = 10  # Number of runs for timing statistics

for sigma in STIFFNESS_RATIOS:
    print(f"\n--- Stiffness ratio sigma = {sigma} ---")

    # Compute dt from stiffness ratio: sigma = D * dt / dx^2
    dt = sigma * dx**2 / D

    # Reaction Jacobian: dR/du = 2*k*u at linearization point u0
    # This is diagonal but varies spatially, creating structure FFT can't capture
    reaction_jac = 2.0 * REACTION_K * u0_jax

    # Matrix-free Newton Jacobian matvec for reaction-diffusion:
    # J = I - dt*(D*L + diag(dR/du))
    # J*v = v - dt*D*L*v - dt*diag(dR/du)*v
    @jax.jit
    def jacobian_matvec_2d(v_2d):
        """Matrix-free Jacobian: (I - dt*(D*L + dR/du))*v"""
        diffusion = D * fd_laplacian_matvec(v_2d, dx)
        reaction = reaction_jac * v_2d  # diagonal reaction Jacobian
        return v_2d - dt * (diffusion + reaction)

    def jacobian_matvec_flat(v_flat):
        """Flattened version for GMRES."""
        v_2d = v_flat.reshape(N, N)
        result = jacobian_matvec_2d(v_2d)
        return result.flatten()

    rhs_flat = rhs_2d.flatten()

    sweep_result = {'sigma': sigma, 'dt': float(dt)}

    # =========================================================================
    # No Preconditioner (FD matvec) - with wall-clock timing
    # =========================================================================
    print("  Testing: No preconditioner (FD)...")

    iter_count_none = [0]

    def fd_matvec_counted(u):
        iter_count_none[0] += 1
        return jacobian_matvec_flat(u)

    # Warmup run
    try:
        iter_count_none[0] = 0
        sol_none, info_none = gmres(
            fd_matvec_counted,
            rhs_flat,
            tol=GMRES_TOL,
            maxiter=GMRES_MAXITER,
        )
        sol_none.block_until_ready()
        iters_none = iter_count_none[0]
        converged_none = info_none == 0
    except Exception as e:
        iters_none = GMRES_MAXITER
        converged_none = False

    # Timed runs (only if converged)
    times_none = []
    if converged_none:
        for _ in range(N_TIMING_REPS):
            iter_count_none[0] = 0
            t0 = time.perf_counter()
            sol, _ = gmres(fd_matvec_counted, rhs_flat, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
            sol.block_until_ready()
            times_none.append(time.perf_counter() - t0)

    if times_none:
        time_median_none, time_iqr_none = compute_stats(times_none)
    else:
        time_median_none, time_iqr_none = None, None

    sweep_result['none'] = {
        'iterations': int(iters_none),
        'converged': bool(converged_none),
        'time_ms': float(time_median_none * 1000) if time_median_none else None,
        'time_iqr_ms': float(time_iqr_none * 1000) if time_iqr_none else None,
    }
    status = "" if converged_none else "*"
    time_str = f", {time_median_none*1000:.2f} ms" if time_median_none else ""
    print(f"    Iterations: {iters_none}{status}{time_str}")

    # =========================================================================
    # FFT Preconditioner - with wall-clock timing
    # =========================================================================
    print("  Testing: FFT preconditioner...")

    # FFT-based preconditioner: approximate inverse of (I - 0.5*dt*D*L)
    # using spectral eigenvalues instead of FD
    @jax.jit
    def fft_precond_2d(u_2d):
        """FFT preconditioner in 2D."""
        u_hat = jnp.fft.fft2(u_2d)
        # M^{-1} approximates (I - 0.5*dt*D*L)^{-1} using FFT eigenvalues
        result_hat = u_hat / (1 - 0.5*dt*D*lap_eig_jax)
        return jnp.real(jnp.fft.ifft2(result_hat))

    def fft_precond_flat(u_flat):
        u_2d = u_flat.reshape(N, N)
        return fft_precond_2d(u_2d).flatten()

    iter_count_fft = [0]

    def precond_fd_matvec(u):
        iter_count_fft[0] += 1
        # Left-preconditioned: M^{-1} A u
        return fft_precond_flat(jacobian_matvec_flat(u))

    precond_rhs = fft_precond_flat(rhs_flat)

    # Warmup run
    try:
        iter_count_fft[0] = 0
        sol_fft, info_fft = gmres(
            precond_fd_matvec,
            precond_rhs,
            tol=GMRES_TOL,
            maxiter=GMRES_MAXITER,
        )
        sol_fft.block_until_ready()
        iters_fft = iter_count_fft[0]
        converged_fft = info_fft == 0
    except Exception as e:
        iters_fft = GMRES_MAXITER
        converged_fft = False

    # Timed runs (only if converged)
    times_fft = []
    if converged_fft:
        for _ in range(N_TIMING_REPS):
            iter_count_fft[0] = 0
            precond_rhs_fresh = fft_precond_flat(rhs_flat)
            t0 = time.perf_counter()
            sol, _ = gmres(precond_fd_matvec, precond_rhs_fresh, tol=GMRES_TOL, maxiter=GMRES_MAXITER)
            sol.block_until_ready()
            times_fft.append(time.perf_counter() - t0)

    if times_fft:
        time_median_fft, time_iqr_fft = compute_stats(times_fft)
    else:
        time_median_fft, time_iqr_fft = None, None

    sweep_result['fft'] = {
        'iterations': int(iters_fft),
        'converged': bool(converged_fft),
        'time_ms': float(time_median_fft * 1000) if time_median_fft else None,
        'time_iqr_ms': float(time_iqr_fft * 1000) if time_iqr_fft else None,
    }
    status = "" if converged_fft else "*"
    time_str = f", {time_median_fft*1000:.2f} ms" if time_median_fft else ""
    print(f"    Iterations: {iters_fft}{status}{time_str}")

    # Reduction ratio
    if iters_none > 0 and iters_fft > 0:
        reduction = iters_none / iters_fft
        sweep_result['reduction'] = float(reduction)
        print(f"    Reduction: {reduction:.1f}x")

    # Time speedup
    if time_median_none and time_median_fft:
        time_speedup = time_median_none / time_median_fft
        sweep_result['time_speedup'] = float(time_speedup)
        print(f"    Time speedup: {time_speedup:.1f}x")

    results['sweeps'].append(sweep_result)

# Save results
results['config'] = {
    'grid_size': N,
    'gmres_tol': GMRES_TOL,
    'gmres_maxiter': GMRES_MAXITER,
    'device': device_str,
    'dtype': 'float64',
    'D': D,
    'discretization': 'finite_difference',
}

output_path = Path(__file__).parent / 'results' / 'gmres_sweep.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# =============================================================================
# Summary Table
# =============================================================================
print("\n" + "=" * 70)
print("SUMMARY (Table 10)")
print("=" * 70)
print(f"{'sigma':>10} {'None':>12} {'FFT':>12} {'Reduction':>12}")
print("-" * 70)
for s in results['sweeps']:
    none_str = f"{s['none']['iterations']}" + ("*" if not s['none']['converged'] else "")
    fft_str = f"{s['fft']['iterations']}" + ("*" if not s['fft']['converged'] else "")
    reduction = s.get('reduction', 0)
    print(f"{s['sigma']:>10} {none_str:>12} {fft_str:>12} {reduction:>11.1f}x")
print("-" * 70)
print(f"* = did not converge in {GMRES_MAXITER} iterations")
print("=" * 70)

# =============================================================================
# Generate Figure
# =============================================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

sigmas = [s['sigma'] for s in results['sweeps']]
iters_none = [s['none']['iterations'] for s in results['sweeps']]
iters_fft = [s['fft']['iterations'] for s in results['sweeps']]

ax.semilogy(range(len(sigmas)), iters_none, 'o-', label='No preconditioner (FD)', color='#1f77b4', linewidth=2, markersize=10)
ax.semilogy(range(len(sigmas)), iters_fft, 's-', label='FFT preconditioner', color='#2ca02c', linewidth=2, markersize=10)

ax.set_xticks(range(len(sigmas)))
ax.set_xticklabels([str(s) for s in sigmas])
ax.set_xlabel(r'Stiffness ratio $\sigma = D\Delta t / \Delta x^2$', fontsize=12)
ax.set_ylabel('GMRES iterations', fontsize=12)
ax.set_title('GMRES Iterations vs Stiffness Ratio\n(Finite Difference discretization, FFT Preconditioner)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add GMRES restart threshold
ax.axhline(y=GMRES_MAXITER, color='red', linestyle='--', alpha=0.5)
ax.text(len(sigmas)-0.5, GMRES_MAXITER*1.1, f'Restart threshold ({GMRES_MAXITER})',
        fontsize=9, color='red', ha='right')

# Add reduction annotations
for i, s in enumerate(results['sweeps']):
    if 'reduction' in s and s['reduction'] > 1.5:
        ax.annotate(f'{s["reduction"]:.0f}x',
                   (i, iters_fft[i]), textcoords="offset points",
                   xytext=(10, 5), fontsize=9, color='green')

plt.tight_layout()
fig_path = Path(__file__).parent.parent / 'figures' / 'gmres_sweep.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to {fig_path}")
plt.close()
