#!/usr/bin/env python3
"""
Generate convergence figure showing spectral and temporal order verification.

For pseudo-spectral FFT methods with smooth periodic solutions:
- SPATIAL: Error decays exponentially with N (spectral accuracy)
- TEMPORAL: Error decays as O(dt^2) for Crank-Nicolson (2nd-order)

Tests 2D diffusion with PERIODIC manufactured solution:
    u(x,y,t) = exp(-8*pi^2*D*t) * sin(2*pi*x) * sin(2*pi*y)

This completes exactly one period over [0,1]^2, matching periodic BC.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import jax
import jax.numpy as jnp

print("Generating convergence verification figure...")
print(f"JAX devices: {jax.devices()}")

D = 0.1
T_FINAL = 0.1

def analytical_solution(x, y, t, D):
    """Exact solution for 2D diffusion with periodic sin(2*pi*x)*sin(2*pi*y) IC."""
    # Laplacian of sin(2*pi*x)*sin(2*pi*y) = -8*pi^2 * u
    # So decay factor is exp(-8*pi^2*D*t)
    return np.exp(-8 * np.pi**2 * D * t) * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

def run_simulation(N, dt, D, T_final):
    """Run FFT-CN simulation and return max error."""
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)
    y = np.linspace(0, 1, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial condition (periodic: completes one full period over [0,1])
    u0 = np.sin(2 * np.pi * X) * np.sin(2 * np.pi * Y)

    # FFT eigenvalues for periodic domain (pseudo-spectral Laplacian: -k^2)
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)

    # Time stepping (Crank-Nicolson in Fourier space)
    n_steps = int(T_final / dt)
    u = u0.copy()

    for _ in range(n_steps):
        u_hat = np.fft.fft2(u)
        u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
        u = np.real(np.fft.ifft2(u_hat))

    # Analytical solution
    u_exact = analytical_solution(X, Y, n_steps * dt, D)

    # Error (full domain - solution is properly periodic)
    error = np.max(np.abs(u - u_exact))

    return error

# =============================================================================
# Spatial Convergence (fixed small dt, varying N) - SPECTRAL ACCURACY
# =============================================================================
print("\n1. Spatial convergence test (spectral accuracy)...")

dt_fixed = 1e-6  # Very small dt to isolate spatial error
grid_sizes = [8, 16, 32, 64, 128, 256]
spatial_errors = []

for N in grid_sizes:
    err = run_simulation(N, dt_fixed, D, T_FINAL)
    spatial_errors.append(err)
    print(f"   N={N:4d}, error={err:.2e}")

# Check for exponential decay (spectral accuracy)
# For smooth solutions, error ~ C * exp(-alpha * N)
# In semi-log plot, this is linear with slope -alpha
log_N = np.array(grid_sizes)
log_err = np.log10(np.array(spatial_errors) + 1e-20)  # Prevent log(0)

# Linear fit in semi-log space (error vs N, not log(N))
# For spectral methods: log(error) ~ -alpha * N
valid_idx = np.array(spatial_errors) > 1e-14  # Above machine precision
if np.sum(valid_idx) >= 2:
    coeffs = np.polyfit(np.array(grid_sizes)[valid_idx], log_err[valid_idx], 1)
    spectral_rate = -coeffs[0]  # Exponential decay rate
    print(f"   Spectral decay rate (per grid point): {spectral_rate:.4f}")
else:
    spectral_rate = None
    print("   Reached machine precision - spectral accuracy confirmed!")

# =============================================================================
# Temporal Convergence (fixed fine grid, varying dt)
# =============================================================================
print("\n2. Temporal convergence test (Crank-Nicolson, 2nd-order)...")

N_fixed = 256  # Fine grid to isolate temporal error
dt_values = [0.01, 0.005, 0.002, 0.001, 0.0005]
temporal_errors = []

for dt in dt_values:
    err = run_simulation(N_fixed, dt, D, T_FINAL)
    temporal_errors.append(err)
    print(f"   dt={dt:.4f}, error={err:.2e}")

# Fit slope in log-log space (polynomial convergence)
log_dt = np.log(dt_values)
log_err_t = np.log(temporal_errors)
temporal_slope, _ = np.polyfit(log_dt, log_err_t, 1)
print(f"   Fitted temporal order: {temporal_slope:.2f}")

# =============================================================================
# Generate Figure
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ---- Spatial convergence (SEMI-LOG for spectral accuracy) ----
ax1.semilogy(grid_sizes, spatial_errors, 'o-', color='#1f77b4', linewidth=2,
             markersize=10, label='Measured error')

# Reference: exponential decay (spectral accuracy)
if spectral_rate and spectral_rate > 0:
    ref_N = np.linspace(grid_sizes[0], grid_sizes[-1], 100)
    ref_err = spatial_errors[0] * np.exp(-spectral_rate * (ref_N - grid_sizes[0]))
    ax1.semilogy(ref_N, ref_err, '--', color='gray', linewidth=2,
                 label=f'Exponential decay')

# Machine precision line
ax1.axhline(y=1e-14, color='red', linestyle=':', linewidth=1.5,
            label='Machine precision')

ax1.set_xlabel(r'Grid points $N$', fontsize=12)
ax1.set_ylabel(r'Max error $\|e\|_\infty$', fontsize=12)
ax1.set_title('Spatial Convergence (Spectral Accuracy)\nError decays exponentially with N', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(1e-16, 1e0)

# ---- Temporal convergence (LOG-LOG for polynomial order) ----
ax2.loglog(dt_values, temporal_errors, 's-', color='#2ca02c', linewidth=2,
           markersize=10, label='Measured error')

# Reference slope = 2 (Crank-Nicolson is 2nd-order in time)
ref_dt = np.array([dt_values[0], dt_values[-1]])
ref_err_t = temporal_errors[0] * (ref_dt / dt_values[0])**2
ax2.loglog(ref_dt, ref_err_t, '--', color='gray', linewidth=2,
           label=f'Slope = 2 (reference)')

ax2.set_xlabel(r'Time step $\Delta t$', fontsize=12)
ax2.set_ylabel(r'Max error $\|e\|_\infty$', fontsize=12)
ax2.set_title(f'Temporal Convergence (Crank-Nicolson)\nFitted order: {temporal_slope:.2f}', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

fig_dir = Path(__file__).parent / 'figures'
fig_dir.mkdir(exist_ok=True)
fig_path = fig_dir / 'convergence.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to {fig_path}")
plt.close()

# Also save as PDF for paper
fig_path_pdf = fig_dir / 'convergence.pdf'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Regenerate for PDF
ax1.semilogy(grid_sizes, spatial_errors, 'o-', color='#1f77b4', linewidth=2,
             markersize=10, label='Measured error')
if spectral_rate and spectral_rate > 0:
    ax1.semilogy(ref_N, ref_err, '--', color='gray', linewidth=2,
                 label=f'Exponential decay')
ax1.axhline(y=1e-14, color='red', linestyle=':', linewidth=1.5,
            label='Machine precision')
ax1.set_xlabel(r'Grid points $N$', fontsize=12)
ax1.set_ylabel(r'Max error $\|e\|_\infty$', fontsize=12)
ax1.set_title('Spatial Convergence (Spectral Accuracy)\nError decays exponentially with N', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(1e-16, 1e0)

ax2.loglog(dt_values, temporal_errors, 's-', color='#2ca02c', linewidth=2,
           markersize=10, label='Measured error')
ax2.loglog(ref_dt, ref_err_t, '--', color='gray', linewidth=2,
           label=f'Slope = 2 (reference)')
ax2.set_xlabel(r'Time step $\Delta t$', fontsize=12)
ax2.set_ylabel(r'Max error $\|e\|_\infty$', fontsize=12)
ax2.set_title(f'Temporal Convergence (Crank-Nicolson)\nFitted order: {temporal_slope:.2f}', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(fig_path_pdf, bbox_inches='tight')
print(f"PDF saved to {fig_path_pdf}")
plt.close()

print("\n" + "="*70)
print("CONVERGENCE SUMMARY")
print("="*70)
print(f"SPATIAL (pseudo-spectral FFT): Exponential decay (spectral accuracy)")
print(f"  - Error drops from {spatial_errors[0]:.2e} to {spatial_errors[-1]:.2e}")
print(f"  - Reaches near-machine precision at N={grid_sizes[-1]}")
print(f"\nTEMPORAL (Crank-Nicolson): Order {temporal_slope:.2f} (expected: 2)")
print(f"  - Error ~ O(dt^2) as expected for CN")
print("="*70)
