#!/usr/bin/env python3
"""
Method Comparison Benchmarks for C&CE Paper

Part 1: Tubular Reactor Method Comparison (RK4 vs CN vs IMEX)
Part 2: Diffusion Work-Precision with FD Baseline (PERIODIC BC for fair comparison)

This addresses reviewer concerns about:
1. Reactor case study needs method comparison across stiffness regimes
2. FFT-CN comparison needs fair FD baseline with SAME BCs (not different problems)
"""

import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt

# JAX setup
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit
from functools import partial

print("=" * 70)
print("Method Comparison Benchmarks")
print("=" * 70)
print(f"JAX devices: {jax.devices()}")
print()

# Output directories (local to benchmarks/)
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# =============================================================================
# PART 1: TUBULAR REACTOR METHOD COMPARISON (RK4 vs CN vs IMEX)
# =============================================================================
#
# PDE: ∂c/∂t = (1/Pe) ∂²c/∂z² - ∂c/∂z - Da*c
#
# Nondimensional parameters:
#   Pe = vL/D (Peclet number)
#   Da = kL/v (Damköhler number, first-order reaction k*c)
#
# BCs: c(0,t) = 1 (inlet), ∂c/∂z(1,t) = 0 (outlet)
# IC: c(z,0) = 0 (empty reactor)
#
# Analytical steady-state for Pe >> 1 (plug flow limit):
#   c_ss(z) ≈ exp(-Da*z)
#   X_out = 1 - c_ss(1) = 1 - exp(-Da)
# =============================================================================

def create_reactor_system(N, Pe, Da):
    """
    Create discretized reactor system using second-order FD.

    Returns L matrix for: dc/dt = L @ c + source
    Also returns split operators for IMEX: L_diff (dispersion) and L_adv_rxn (advection + reaction)
    """
    dz = 1.0 / (N - 1)
    z = np.linspace(0, 1, N)

    # Second derivative (central difference, second-order)
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] = 1.0
        D2[i, i] = -2.0
        D2[i, i+1] = 1.0
    D2 /= dz**2

    # First derivative (upwind for stability)
    D1 = np.zeros((N, N))
    for i in range(1, N):
        D1[i, i-1] = -1.0
        D1[i, i] = 1.0
    D1 /= dz

    # Combined linear operator (interior)
    L = (1.0/Pe) * D2 - D1 - Da * np.eye(N)

    # Split for IMEX: diffusion implicit, advection+reaction explicit
    L_diff = (1.0/Pe) * D2.copy()
    L_adv_rxn = -D1 - Da * np.eye(N)

    # Boundary conditions:
    # z=0: c = 1 (Dirichlet) - implemented via source term
    # z=1: dc/dz = 0 (Neumann) - ghost point: c[N] = c[N-1]

    # At z=0: set row to zero (will add source term)
    L[0, :] = 0
    L_diff[0, :] = 0
    L_adv_rxn[0, :] = 0

    # At z=1: use one-sided difference for Neumann BC
    L[-1, :] = 0
    L[-1, -2] = (1.0/Pe) / dz**2
    L[-1, -1] = -(1.0/Pe) / dz**2 - 1.0/dz - Da

    L_diff[-1, :] = 0
    L_diff[-1, -2] = (1.0/Pe) / dz**2
    L_diff[-1, -1] = -(1.0/Pe) / dz**2

    L_adv_rxn[-1, :] = 0
    L_adv_rxn[-1, -2] = 0
    L_adv_rxn[-1, -1] = -1.0/dz - Da

    return jnp.array(L), jnp.array(L_diff), jnp.array(L_adv_rxn), jnp.array(z)


def run_reactor_rk4(c0, L, dt, n_steps, c_inlet=1.0):
    """RK4 for reactor with Dirichlet inlet BC."""
    @jit
    def rhs(c):
        dcdt = L @ c
        dcdt = dcdt.at[0].set(0.0)
        return dcdt

    @jit
    def step(c, _):
        k1 = rhs(c)
        k2 = rhs(c + 0.5*dt*k1)
        k3 = rhs(c + 0.5*dt*k2)
        k4 = rhs(c + dt*k3)
        c_new = c + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        c_new = c_new.at[0].set(c_inlet)
        return c_new, None

    c_final, _ = jax.lax.scan(step, c0, None, length=n_steps)
    return c_final


def run_reactor_cn(c0, L, dt, n_steps, c_inlet=1.0):
    """
    Crank-Nicolson for reactor.

    (I - 0.5*dt*L) c^{n+1} = (I + 0.5*dt*L) c^n
    """
    N = len(c0)
    I = jnp.eye(N)

    A_lhs = I - 0.5 * dt * L
    A_rhs = I + 0.5 * dt * L

    # Modify for Dirichlet BC at inlet
    A_lhs = A_lhs.at[0, :].set(0)
    A_lhs = A_lhs.at[0, 0].set(1)
    A_rhs = A_rhs.at[0, :].set(0)

    @jit
    def step(c, _):
        rhs = A_rhs @ c
        rhs = rhs.at[0].set(c_inlet)
        c_new = jnp.linalg.solve(A_lhs, rhs)
        return c_new, None

    c_final, _ = jax.lax.scan(step, c0, None, length=n_steps)
    return c_final


def run_reactor_imex(c0, L_diff, L_adv_rxn, dt, n_steps, c_inlet=1.0):
    """
    IMEX for reactor: diffusion implicit, advection+reaction explicit.

    (I - dt*L_diff) c^{n+1} = c^n + dt*L_adv_rxn @ c^n
    """
    N = len(c0)
    I = jnp.eye(N)

    A_impl = I - dt * L_diff

    # Modify for Dirichlet BC at inlet
    A_impl = A_impl.at[0, :].set(0)
    A_impl = A_impl.at[0, 0].set(1)

    @jit
    def step(c, _):
        # Explicit part: advection + reaction
        rhs = c + dt * (L_adv_rxn @ c)
        rhs = rhs.at[0].set(c_inlet)
        # Implicit solve for diffusion
        c_new = jnp.linalg.solve(A_impl, rhs)
        return c_new, None

    c_final, _ = jax.lax.scan(step, c0, None, length=n_steps)
    return c_final


def benchmark_reactor_methods(Pe, Da, t_final=5.0, N=128):
    """Compare methods on reactor for specific Pe, Da."""
    L, L_diff, L_adv_rxn, z = create_reactor_system(N, Pe, Da)
    dz = 1.0 / (N - 1)

    # Initial condition: empty reactor (c=0), inlet at c=1
    c0 = jnp.zeros(N)
    c0 = c0.at[0].set(1.0)

    results = {}

    # Stability constraints:
    # RK4 advection CFL: dt < dz (v=1)
    # RK4 diffusion: dt < Pe * dz^2 / 2
    # RK4 reaction: dt < 1/Da

    dt_cfl = 0.5 * dz
    dt_diff = 0.4 * Pe * dz**2 if Pe > 0 else 1.0
    dt_rxn = 0.5 / Da if Da > 0 else 1.0

    # RK4: use smallest stability limit
    dt_rk4 = min(dt_cfl, dt_diff, dt_rxn, 0.001)
    n_steps_rk4 = int(t_final / dt_rk4)

    # CN: unconditionally stable, can use larger timestep
    dt_cn = min(10 * dz, 0.05, t_final / 100)
    n_steps_cn = int(t_final / dt_cn)

    # IMEX: advection CFL limited, but can handle diffusion implicitly
    dt_imex = min(dt_cfl * 2, 0.02, t_final / 100)
    n_steps_imex = int(t_final / dt_imex)

    # Warmup JIT
    _ = run_reactor_rk4(c0, L, dt_rk4, 10)
    _ = run_reactor_cn(c0, L, dt_cn, 10)
    _ = run_reactor_imex(c0, L_diff, L_adv_rxn, dt_imex, 10)

    # RK4 benchmark
    start = time.perf_counter()
    c_rk4 = run_reactor_rk4(c0, L, dt_rk4, n_steps_rk4)
    c_rk4.block_until_ready()
    time_rk4 = time.perf_counter() - start
    conv_rk4 = 1.0 - float(c_rk4[-1])

    results['RK4'] = {
        'time': time_rk4,
        'conversion': conv_rk4,
        'steps': n_steps_rk4,
        'dt': dt_rk4
    }

    # CN benchmark
    start = time.perf_counter()
    c_cn = run_reactor_cn(c0, L, dt_cn, n_steps_cn)
    c_cn.block_until_ready()
    time_cn = time.perf_counter() - start
    conv_cn = 1.0 - float(c_cn[-1])

    results['CN'] = {
        'time': time_cn,
        'conversion': conv_cn,
        'steps': n_steps_cn,
        'dt': dt_cn
    }

    # IMEX benchmark
    start = time.perf_counter()
    c_imex = run_reactor_imex(c0, L_diff, L_adv_rxn, dt_imex, n_steps_imex)
    c_imex.block_until_ready()
    time_imex = time.perf_counter() - start
    conv_imex = 1.0 - float(c_imex[-1])

    results['IMEX'] = {
        'time': time_imex,
        'conversion': conv_imex,
        'steps': n_steps_imex,
        'dt': dt_imex
    }

    # Analytical plug flow approximation (Pe -> inf)
    conv_analytical = 1.0 - np.exp(-Da)

    results['analytical'] = {'conversion': conv_analytical}
    results['RK4']['error'] = abs(conv_rk4 - conv_analytical)
    results['CN']['error'] = abs(conv_cn - conv_analytical)
    results['IMEX']['error'] = abs(conv_imex - conv_analytical)

    return results, np.array(z), np.array(c_rk4), np.array(c_cn), np.array(c_imex)


# =============================================================================
# PART 2: DIFFUSION WORK-PRECISION WITH PERIODIC FD BASELINE
# =============================================================================
#
# FAIR COMPARISON: Both methods use PERIODIC BC on same problem!
#
# PDE: ∂u/∂t = D ∂²u/∂x²
# IC: u(x,0) = sin(2πx) (smooth, periodic)
# Exact: u(x,t) = sin(2πx) exp(-4π²Dt)
#
# FFT: uses spectral eigenvalues λ_k = -(2πk)² (exact for smooth)
# FD:  uses FD eigenvalues λ_k = -(4/dx²)sin²(πk dx) (O(dx²) error)
#
# Both use periodic BC, same IC, same exact solution - only spatial operator differs.
# =============================================================================

def solve_diffusion_fft_cn(u0, D, dx, dt, n_steps):
    """FFT-CN with pseudo-spectral Laplacian (periodic BC)."""
    N = len(u0)

    # Wavenumbers for periodic domain [0, 1)
    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    lap_eig = -kx**2  # SPECTRAL: continuous symbol

    # CN factor in spectral space
    cn_factor = (1 + 0.5 * dt * D * lap_eig) / (1 - 0.5 * dt * D * lap_eig)

    @jit
    def step(u, _):
        u_hat = jnp.fft.fft(u)
        u_hat = u_hat * cn_factor
        return jnp.real(jnp.fft.ifft(u_hat)), None

    u_final, _ = jax.lax.scan(step, u0, None, length=n_steps)
    return u_final


def solve_diffusion_fd_cn_periodic(u0, D, dx, dt, n_steps):
    """
    FD-CN with second-order central differences (PERIODIC BC).
    Uses FFT diagonalization with FD eigenvalues.

    This is the fair comparison: same BC as FFT-CN, but FD spatial accuracy.
    """
    N = len(u0)

    # FD eigenvalues for periodic Laplacian (circulant matrix)
    # λ_k = (2/dx²)(cos(2πk/N) - 1) = -(4/dx²)sin²(πk/N)
    k = jnp.fft.fftfreq(N, 1.0/N)  # Modes 0, 1, ..., N/2-1, -N/2, ..., -1
    lap_eig_fd = (2.0 / dx**2) * (jnp.cos(2 * jnp.pi * k / N) - 1)

    # CN factor in spectral space (with FD eigenvalues)
    cn_factor = (1 + 0.5 * dt * D * lap_eig_fd) / (1 - 0.5 * dt * D * lap_eig_fd)

    @jit
    def step(u, _):
        u_hat = jnp.fft.fft(u)
        u_hat = u_hat * cn_factor
        return jnp.real(jnp.fft.ifft(u_hat)), None

    u_final, _ = jax.lax.scan(step, u0, None, length=n_steps)
    return u_final


def benchmark_diffusion_methods(N=128, D=0.1, t_final=0.1):
    """
    Compare FFT-CN (spectral) and FD-CN (finite-difference) for diffusion.

    FAIR COMPARISON: Both use periodic BC, same IC, same exact solution.
    Only difference is spatial operator: spectral (-k²) vs FD eigenvalues.
    """

    # Both methods: periodic domain [0, 1) with N points
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)

    # Same initial condition for both (smooth, periodic)
    u0 = jnp.sin(2 * np.pi * x)

    # Same exact solution for both
    def u_exact(t):
        return np.sin(2 * np.pi * x) * np.exp(-4 * np.pi**2 * D * t)

    results = {}
    dt_values = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]

    for dt in dt_values:
        n_steps = int(t_final / dt)

        # FFT-CN (pseudo-spectral eigenvalues)
        _ = solve_diffusion_fft_cn(u0, D, dx, dt, 10)  # Warmup
        start = time.perf_counter()
        u_fft = solve_diffusion_fft_cn(u0, D, dx, dt, n_steps)
        u_fft.block_until_ready()
        time_fft = time.perf_counter() - start
        error_fft = np.max(np.abs(np.array(u_fft) - u_exact(t_final)))

        # FD-CN (finite-difference eigenvalues, but same periodic BC)
        _ = solve_diffusion_fd_cn_periodic(u0, D, dx, dt, 10)  # Warmup
        start = time.perf_counter()
        u_fd = solve_diffusion_fd_cn_periodic(u0, D, dx, dt, n_steps)
        u_fd.block_until_ready()
        time_fd = time.perf_counter() - start
        error_fd = np.max(np.abs(np.array(u_fd) - u_exact(t_final)))

        results[dt] = {
            'FFT-CN': {'time': time_fft, 'error': error_fft},
            'FD-CN': {'time': time_fd, 'error': error_fd},
            'n_steps': n_steps
        }

    return results, x


# =============================================================================
# PLOTTING
# =============================================================================

def plot_reactor_comparison(results_by_regime, save_path):
    """Plot reactor method comparison across Pe-Da regimes (now with IMEX)."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    regimes = list(results_by_regime.keys())
    methods = ['RK4', 'CN', 'IMEX']
    colors = {'RK4': '#1f77b4', 'CN': '#2ca02c', 'IMEX': '#ff7f0e'}

    # Runtime comparison
    ax = axes[0]
    x = np.arange(len(regimes))
    width = 0.25

    for i, method in enumerate(methods):
        times = [results_by_regime[r][method]['time'] for r in regimes]
        ax.bar(x + (i-1)*width, times, width, label=method, color=colors[method])

    ax.set_ylabel('Runtime (s)', fontsize=12)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('\n', '\n') for r in regimes], fontsize=9)
    ax.legend(fontsize=10)
    ax.set_title('(a) Runtime', fontsize=12)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Speedup over RK4
    ax = axes[1]
    width = 0.35
    speedups_cn = [results_by_regime[r]['RK4']['time']/results_by_regime[r]['CN']['time'] for r in regimes]
    speedups_imex = [results_by_regime[r]['RK4']['time']/results_by_regime[r]['IMEX']['time'] for r in regimes]

    bars1 = ax.bar(x - width/2, speedups_cn, width, label='CN', color=colors['CN'])
    bars2 = ax.bar(x + width/2, speedups_imex, width, label='IMEX', color=colors['IMEX'])
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Speedup over RK4', fontsize=12)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('\n', '\n') for r in regimes], fontsize=9)
    ax.legend(fontsize=10)
    ax.set_title('(b) Speedup', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, speedup in zip(bars1, speedups_cn):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{speedup:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')
    for bar, speedup in zip(bars2, speedups_imex):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{speedup:.1f}×', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Conversion comparison
    ax = axes[2]
    width = 0.2
    for i, method in enumerate(methods):
        conv = [results_by_regime[r][method]['conversion'] for r in regimes]
        ax.bar(x + (i-1)*width, conv, width, label=method, color=colors[method])
    # Add analytical
    conv_ana = [results_by_regime[r]['analytical']['conversion'] for r in regimes]
    ax.bar(x + 1.5*width, conv_ana, width, label='Plug flow', color='gray', alpha=0.5)

    ax.set_ylabel('Conversion $X$', fontsize=12)
    ax.set_xlabel('Regime', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([r.replace('\n', '\n') for r in regimes], fontsize=9)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_title('(c) Outlet Conversion', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_work_precision(results, save_path):
    """Plot work-precision diagram for diffusion methods."""
    fig, ax = plt.subplots(figsize=(8, 6))

    dt_values = sorted(results.keys(), reverse=True)

    times_fft = [results[dt]['FFT-CN']['time'] for dt in dt_values]
    errors_fft = [results[dt]['FFT-CN']['error'] for dt in dt_values]

    times_fd = [results[dt]['FD-CN']['time'] for dt in dt_values]
    errors_fd = [results[dt]['FD-CN']['error'] for dt in dt_values]

    ax.loglog(times_fft, errors_fft, 'o-', color='#1f77b4', linewidth=2,
              markersize=10, label='FFT-CN (spectral eigenvalues)')
    ax.loglog(times_fd, errors_fd, 's-', color='#2ca02c', linewidth=2,
              markersize=10, label='FD-CN (FD eigenvalues)')

    ax.set_xlabel('Runtime (s)', fontsize=12)
    ax.set_ylabel('Max Error', fontsize=12)
    ax.set_title('Work-Precision: Spectral vs FD (Same Periodic BC)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Machine precision line
    ax.axhline(y=1e-14, color='red', linestyle=':', linewidth=1.5, label='Machine precision')

    # Annotate timesteps
    for i, dt in enumerate(dt_values):
        if i % 2 == 0:
            ax.annotate(f'dt={dt}', (times_fft[i], errors_fft[i]),
                       textcoords="offset points", xytext=(10, 5), fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.savefig(save_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("\n" + "="*70)
    print("PART 1: Tubular Reactor Method Comparison (RK4 vs CN vs IMEX)")
    print("="*70)

    # Test across Pe-Da regimes
    regimes = {
        'Pe=10\nDa=0.5': (10, 0.5),       # Dispersion-dominated, mild reaction
        'Pe=50\nDa=2': (50, 2),            # Balanced
        'Pe=100\nDa=5': (100, 5),          # Near plug-flow, stiff kinetics
    }

    reactor_results = {}
    for regime_name, (Pe, Da) in regimes.items():
        print(f"\nRunning {regime_name.replace(chr(10), ', ')}...")
        results, z, c_rk4, c_cn, c_imex = benchmark_reactor_methods(Pe, Da)
        reactor_results[regime_name] = results

        print(f"  RK4:  {results['RK4']['time']:.4f}s, {results['RK4']['steps']} steps, X={results['RK4']['conversion']:.4f}")
        print(f"  CN:   {results['CN']['time']:.4f}s, {results['CN']['steps']} steps, X={results['CN']['conversion']:.4f}")
        print(f"  IMEX: {results['IMEX']['time']:.4f}s, {results['IMEX']['steps']} steps, X={results['IMEX']['conversion']:.4f}")
        print(f"  Analytical (plug flow): X={results['analytical']['conversion']:.4f}")
        print(f"  Speedup: CN={results['RK4']['time']/results['CN']['time']:.1f}×, IMEX={results['RK4']['time']/results['IMEX']['time']:.1f}×")

    # Plot reactor comparison
    plot_reactor_comparison(reactor_results, FIGURES_DIR / "fig_reactor_method_comparison.pdf")

    print("\n" + "="*70)
    print("PART 2: Diffusion Work-Precision (Fair Comparison: Same Periodic BC)")
    print("="*70)

    diffusion_results, x = benchmark_diffusion_methods(N=128, D=0.1, t_final=0.1)

    print("\nWork-Precision Results (N=128, periodic BC for both):")
    print(f"{'dt':<10} {'FFT-CN time':<15} {'FFT-CN error':<15} {'FD-CN time':<15} {'FD-CN error':<15}")
    print("-"*70)
    for dt in sorted(diffusion_results.keys(), reverse=True):
        r = diffusion_results[dt]
        print(f"{dt:<10.4f} {r['FFT-CN']['time']:<15.4f} {r['FFT-CN']['error']:<15.2e} {r['FD-CN']['time']:<15.4f} {r['FD-CN']['error']:<15.2e}")

    # Plot work-precision
    plot_work_precision(diffusion_results, FIGURES_DIR / "fig_diffusion_work_precision.pdf")

    # Save results
    all_results = {
        'reactor_methods': {k.replace('\n', ' '): v for k, v in reactor_results.items()},
        'diffusion_work_precision': {str(k): v for k, v in diffusion_results.items()}
    }

    with open(RESULTS_DIR / "method_comparison_results.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\nResults saved to: {RESULTS_DIR / 'method_comparison_results.json'}")

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nReactor Method Comparison (RK4 vs CN vs IMEX):")
    for regime_name, results in reactor_results.items():
        speedup_cn = results['RK4']['time'] / results['CN']['time']
        speedup_imex = results['RK4']['time'] / results['IMEX']['time']
        print(f"  {regime_name.replace(chr(10), ', ')}: CN={speedup_cn:.1f}×, IMEX={speedup_imex:.1f}× faster than RK4")

    print("\nDiffusion Work-Precision (FAIR: same periodic BC):")
    print("  FFT-CN (spectral): Error dominated by temporal O(dt²)")
    print("  FD-CN (2nd-order FD eigenvalues): Spatial O(dx²) dominates")
    print("  Both use periodic BC - only spatial discretization differs!")
    print("="*70)
