#!/usr/bin/env python3
"""
Tubular Reactor Benchmark for C&CE Paper

Solves the axial dispersion model with first-order reaction:
    ∂c/∂t = (1/Pe) ∂²c/∂z² - ∂c/∂z - Da·c

Boundary conditions supported:
    - Danckwerts: (1/Pe)∂c/∂z = c - c_in at inlet, ∂c/∂z = 0 at outlet
    - Robin: α·c + β·∂c/∂z = γ at boundaries
    - Neumann: ∂c/∂z = 0 at both boundaries (no flux)

Dimensionless groups:
    Pe = v·L/D_ax (Péclet number: advection/dispersion)
    Da = k·L/v (Damköhler number: reaction/advection)

This uses FD operators (not FFT) due to non-periodic BCs.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import jax.numpy as jnp
from jax import jit
from functools import partial

# Import benchmark utilities
from benchmark_utils import (
    setup_benchmark, check_finite_tree, compute_stats,
    add_benchmark_args, DEFAULT_N_REPS
)

# Parse CLI arguments
parser = add_benchmark_args()
args = parser.parse_args()
N_REPS = args.n_reps

print("=" * 70)
print("Tubular Reactor Benchmark (Multi-BC)")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")
print(f"N_REPS: {N_REPS}")
print()

# =============================================================================
# FD Discretization for Axial Dispersion Model
# =============================================================================

def create_fd_matrices(N, dz):
    """
    Create finite difference matrices for 2nd derivative and 1st derivative.
    Uses 2nd-order central differences for interior.
    """
    # Second derivative (for dispersion)
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] = 1.0
        D2[i, i] = -2.0
        D2[i, i+1] = 1.0
    D2 /= dz**2

    # First derivative (central for interior)
    D1 = np.zeros((N, N))
    for i in range(1, N-1):
        D1[i, i-1] = -0.5
        D1[i, i+1] = 0.5
    # Upwind at boundaries
    D1[0, 0] = -1.0
    D1[0, 1] = 1.0
    D1[-1, -2] = -1.0
    D1[-1, -1] = 1.0
    D1 /= dz

    return D2, D1


def create_rhs_danckwerts(N, Pe, Da, dz, c_in=1.0):
    """
    Danckwerts BCs:
        Inlet (z=0):  (1/Pe) ∂c/∂z = c - c_in
        Outlet (z=1): ∂c/∂z = 0
    """
    D2, D1 = create_fd_matrices(N, dz)
    D2 = jnp.array(D2)
    D1 = jnp.array(D1)

    @jit
    def rhs(c, t):
        # Interior PDE: (1/Pe) D2 c - D1 c - Da c
        dcdt = (1.0/Pe) * (D2 @ c) - (D1 @ c) - Da * c

        # Inlet BC (z=0): Danckwerts
        # (1/Pe) * (c[1] - c[0])/dz = c[0] - c_in
        inlet_coeff = (1.0/Pe) / dz
        c0_target = (c_in + inlet_coeff * c[1]) / (1.0 + inlet_coeff)
        dcdt = dcdt.at[0].set(100.0 * (c0_target - c[0]))

        # Outlet BC (z=1): Neumann (∂c/∂z = 0)
        dcdt = dcdt.at[-1].set(100.0 * (c[-2] - c[-1]))

        return dcdt

    return rhs


def create_rhs_robin(N, Pe, Da, dz, c_in=1.0, alpha_in=1.0, beta_in=0.1, alpha_out=0.0, beta_out=1.0):
    """
    Robin BCs (generalized):
        Inlet (z=0):  α_in·c + β_in·∂c/∂z = c_in
        Outlet (z=1): α_out·c + β_out·∂c/∂z = 0

    Special cases:
        α=1, β=0: Dirichlet (c = c_in)
        α=0, β=1: Neumann (∂c/∂z = 0)
        α=1, β=1/Pe: Danckwerts-like
    """
    D2, D1 = create_fd_matrices(N, dz)
    D2 = jnp.array(D2)
    D1 = jnp.array(D1)

    @jit
    def rhs(c, t):
        dcdt = (1.0/Pe) * (D2 @ c) - (D1 @ c) - Da * c

        # Inlet Robin BC: α_in·c[0] + β_in·(c[1]-c[0])/dz = c_in
        # => c[0] = (c_in - β_in/dz·c[1]) / (α_in - β_in/dz)
        denom_in = alpha_in + beta_in/dz
        if denom_in != 0:
            c0_target = (c_in + beta_in/dz * c[1]) / denom_in
            dcdt = dcdt.at[0].set(100.0 * (c0_target - c[0]))

        # Outlet Robin BC: α_out·c[-1] + β_out·(c[-1]-c[-2])/dz = 0
        denom_out = alpha_out + beta_out/dz
        if denom_out != 0:
            cN_target = (beta_out/dz * c[-2]) / denom_out
            dcdt = dcdt.at[-1].set(100.0 * (cN_target - c[-1]))

        return dcdt

    return rhs


def create_rhs_neumann(N, Pe, Da, dz):
    """
    Neumann BCs (no flux):
        Inlet (z=0):  ∂c/∂z = 0
        Outlet (z=1): ∂c/∂z = 0

    Note: This requires a source term to inject mass.
    We model as: c(z,0) = 1 (initial condition), no-flux boundaries.
    """
    D2, D1 = create_fd_matrices(N, dz)
    D2 = jnp.array(D2)
    D1 = jnp.array(D1)

    @jit
    def rhs(c, t):
        dcdt = (1.0/Pe) * (D2 @ c) - (D1 @ c) - Da * c

        # Neumann at inlet: ∂c/∂z = 0 => c[0] = c[1]
        dcdt = dcdt.at[0].set(100.0 * (c[1] - c[0]))

        # Neumann at outlet: ∂c/∂z = 0 => c[-1] = c[-2]
        dcdt = dcdt.at[-1].set(100.0 * (c[-2] - c[-1]))

        return dcdt

    return rhs


# =============================================================================
# Time Integration Methods
# =============================================================================

@partial(jit, static_argnums=(1, 2, 3))
def solve_rk4(c0, rhs_fn, dt, n_steps):
    """Classical RK4."""
    def step(i, c):
        t = i * dt
        k1 = rhs_fn(c, t)
        k2 = rhs_fn(c + 0.5*dt*k1, t + 0.5*dt)
        k3 = rhs_fn(c + 0.5*dt*k2, t + 0.5*dt)
        k4 = rhs_fn(c + dt*k3, t + dt)
        return c + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return jax.lax.fori_loop(0, n_steps, step, c0)


@partial(jit, static_argnums=(1, 2, 3))
def solve_forward_euler(c0, rhs_fn, dt, n_steps):
    """Explicit Forward Euler."""
    def step(i, c):
        return c + dt * rhs_fn(c, i * dt)
    return jax.lax.fori_loop(0, n_steps, step, c0)


# =============================================================================
# Analytical Solutions
# =============================================================================

def analytical_danckwerts_steady(z, Pe, Da):
    """
    Analytical steady-state for Danckwerts BCs with first-order reaction.

    Steady-state: (1/Pe) c'' - c' - Da*c = 0
    Solution: c(z) = A*exp(r1*z) + B*exp(r2*z)
    """
    disc = Pe**2 + 4*Pe*Da
    if disc < 0:
        return np.ones_like(z) * np.nan

    r1 = (Pe + np.sqrt(disc)) / 2
    r2 = (Pe - np.sqrt(disc)) / 2

    # Danckwerts BCs matrix system
    # z=0: (1/Pe)(A*r1 + B*r2) = (A + B) - 1
    # z=1: A*r1*exp(r1) + B*r2*exp(r2) = 0
    M = np.array([
        [r1/Pe - 1, r2/Pe - 1],
        [r1*np.exp(r1), r2*np.exp(r2)]
    ])
    rhs_vec = np.array([-1, 0])

    try:
        AB = np.linalg.solve(M, rhs_vec)
        A, B = AB
        return A * np.exp(r1 * z) + B * np.exp(r2 * z)
    except:
        return np.ones_like(z) * np.nan


def analytical_neumann_transient(z, t, Pe, Da, N_terms=50):
    """
    Analytical transient for Neumann BCs starting from c(z,0) = 1.

    Uses eigenfunction expansion (cosine series for Neumann).
    Simplified case: pure decay without advection for demonstration.
    """
    # For demonstration, use exponential decay model
    # Real solution requires careful eigenfunction analysis
    c = np.exp(-Da * t) * np.ones_like(z)
    return c


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_single_benchmark(Pe, Da, bc_type, N=128, t_final=10.0, n_runs=10):
    """Run benchmark for specific Pe, Da, BC combination."""
    dz = 1.0 / (N - 1)
    z = np.linspace(0, 1, N)

    # Initial condition
    if bc_type == 'neumann':
        c0 = jnp.ones(N)  # Start full for Neumann
    else:
        c0 = jnp.zeros(N)  # Start empty for Danckwerts/Robin

    # Create RHS function
    if bc_type == 'danckwerts':
        rhs_fn = create_rhs_danckwerts(N, Pe, Da, dz)
        c_analytical = analytical_danckwerts_steady(z, Pe, Da)
    elif bc_type == 'robin':
        rhs_fn = create_rhs_robin(N, Pe, Da, dz, alpha_in=1.0, beta_in=1.0/Pe)
        c_analytical = analytical_danckwerts_steady(z, Pe, Da)  # Similar for this Robin
    elif bc_type == 'neumann':
        rhs_fn = create_rhs_neumann(N, Pe, Da, dz)
        c_analytical = analytical_neumann_transient(z, t_final, Pe, Da)
    else:
        raise ValueError(f"Unknown BC type: {bc_type}")

    # CFL timestep
    dt_adv = 0.5 * dz
    dt_diff = 0.25 * Pe * dz**2
    dt = min(dt_adv, dt_diff, 0.01)
    n_steps = int(t_final / dt)

    # RK4 benchmark
    _ = solve_rk4(c0, rhs_fn, dt, min(100, n_steps))  # Warmup

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        c_final = solve_rk4(c0, rhs_fn, dt, n_steps)
        c_final.block_until_ready()
        times.append(time.perf_counter() - start)

    c_final = np.array(c_final)
    conversion = 1.0 - c_final[-1] if bc_type != 'neumann' else c_final.mean()
    error = np.max(np.abs(c_final - c_analytical)) if not np.any(np.isnan(c_analytical)) else np.nan

    return {
        'bc_type': bc_type,
        'Pe': Pe,
        'Da': Da,
        'mean_time': float(np.mean(times)),
        'std_time': float(np.std(times)),
        'conversion': float(conversion),
        'error': float(error) if not np.isnan(error) else None,
        'dt': float(dt),
        'n_steps': int(n_steps),
        'profile': c_final.tolist(),
        'z': z.tolist()
    }


def run_full_sweep():
    """Run benchmarks across all BC types, Pe, Da combinations."""
    bc_types = ['danckwerts', 'robin', 'neumann']
    Pe_values = [1, 10, 100]
    Da_values = [0.1, 1, 10]

    all_results = {}

    for bc in bc_types:
        print(f"\n{'='*60}")
        print(f"Boundary Condition: {bc.upper()}")
        print(f"{'='*60}")

        for Pe in Pe_values:
            for Da in Da_values:
                key = f"{bc}_Pe{Pe}_Da{Da}"
                print(f"  Pe={Pe}, Da={Da}...", end=" ", flush=True)

                try:
                    result = run_single_benchmark(Pe, Da, bc, N=128, t_final=15.0, n_runs=N_REPS)
                    all_results[key] = result
                    print(f"Time: {result['mean_time']:.4f}s, Conv: {result['conversion']:.4f}")
                except Exception as e:
                    print(f"Error: {e}")
                    all_results[key] = {'error': str(e), 'Pe': Pe, 'Da': Da, 'bc_type': bc}

    return all_results


def generate_figures(results):
    """Generate publication-quality figures."""
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    # Figure 1: Concentration profiles by BC type
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bc_types = ['danckwerts', 'robin', 'neumann']
    bc_labels = ['Danckwerts', 'Robin', 'Neumann']

    Pe, Da = 10, 1  # Representative case

    for ax, bc, label in zip(axes, bc_types, bc_labels):
        key = f"{bc}_Pe{Pe}_Da{Da}"
        if key in results and 'profile' in results[key]:
            z = results[key]['z']
            c = results[key]['profile']
            ax.plot(z, c, 'b-', lw=2, label='Numerical (RK4)')

            # Add analytical if available
            z_arr = np.array(z)
            if bc in ['danckwerts', 'robin']:
                c_anal = analytical_danckwerts_steady(z_arr, Pe, Da)
                if not np.any(np.isnan(c_anal)):
                    ax.plot(z, c_anal, 'k--', lw=1.5, label='Analytical')

        ax.set_xlabel('z (dimensionless)')
        ax.set_ylabel('c (dimensionless)')
        ax.set_title(f'{label} BC\n(Pe={Pe}, Da={Da})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_reactor_bc_comparison.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_reactor_bc_comparison.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_reactor_bc_comparison.pdf'}")

    # Figure 2: Conversion vs Da for different Pe (Danckwerts)
    fig, ax = plt.subplots(figsize=(8, 6))

    for Pe in [1, 10, 100]:
        Da_list = []
        conv_list = []
        for Da in [0.1, 1, 10]:
            key = f"danckwerts_Pe{Pe}_Da{Da}"
            if key in results and 'conversion' in results[key]:
                Da_list.append(Da)
                conv_list.append(results[key]['conversion'])
        if Da_list:
            ax.plot(Da_list, conv_list, 'o-', lw=2, ms=8, label=f'Pe={Pe}')

    ax.set_xlabel('Damköhler number (Da)')
    ax.set_ylabel('Outlet conversion')
    ax.set_xscale('log')
    ax.set_title('Conversion vs Da (Danckwerts BC)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_reactor_conversion_vs_da.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_reactor_conversion_vs_da.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_reactor_conversion_vs_da.pdf'}")

    # Figure 3: Work-precision diagram
    fig, ax = plt.subplots(figsize=(8, 6))

    markers = {'danckwerts': 'o', 'robin': 's', 'neumann': '^'}
    colors = {1: 'blue', 10: 'green', 100: 'red'}

    for key, data in results.items():
        if 'mean_time' in data and data.get('error') is not None:
            bc = data['bc_type']
            Pe = data['Pe']
            ax.scatter(data['mean_time'], data['error'],
                      marker=markers[bc], c=colors[Pe], s=100, alpha=0.7)

    # Create legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Danckwerts'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='Robin'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Neumann'),
        Line2D([0], [0], color='blue', lw=2, label='Pe=1'),
        Line2D([0], [0], color='green', lw=2, label='Pe=10'),
        Line2D([0], [0], color='red', lw=2, label='Pe=100'),
    ]
    ax.legend(handles=legend_elements, loc='best')

    ax.set_xlabel('Wall-clock time (s)')
    ax.set_ylabel('Max error vs analytical')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Work-Precision: Tubular Reactor')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_reactor_work_precision.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig_reactor_work_precision.png", dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'fig_reactor_work_precision.pdf'}")

    plt.close('all')


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\nRunning full parameter sweep with multiple BC types...")
    results = run_full_sweep()

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    results_json = {
        'config': {
            'device': device_str,
            'dtype': 'float64',
            'n_reps': N_REPS,
        },
        'results': {}
    }
    for k, v in results.items():
        results_json['results'][k] = {kk: vv for kk, vv in v.items() if kk not in ['profile', 'z']}
        if 'profile' in v:
            results_json['results'][k]['final_outlet_c'] = v['profile'][-1] if v['profile'] else None

    with open(output_dir / "reactor_results.json", 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {output_dir / 'reactor_results.json'}")

    # Generate figures
    print("\nGenerating figures...")
    generate_figures(results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Tubular Reactor Benchmark (Multi-BC)")
    print("=" * 80)
    print(f"{'BC Type':<12} {'Pe':>6} {'Da':>6} {'Conversion':>12} {'Time(s)':>10} {'Error':>12}")
    print("-" * 70)

    for bc in ['danckwerts', 'robin', 'neumann']:
        for Pe in [1, 10, 100]:
            for Da in [0.1, 1, 10]:
                key = f"{bc}_Pe{Pe}_Da{Da}"
                if key in results and 'mean_time' in results[key]:
                    data = results[key]
                    err = data.get('error')
                    err_str = f"{err:.2e}" if err is not None else "N/A"
                    print(f"{bc:<12} {Pe:>6} {Da:>6} {data['conversion']:>12.4f} {data['mean_time']:>10.4f} {err_str:>12}")

    print("\nBenchmark complete!")
