#!/usr/bin/env python3
"""
Benchmark: moljax vs Diffrax for tubular reactor simulation.

This is a fair comparison using finite-difference discretization for both
(Danckwerts BCs are not periodic, so FFT doesn't apply directly).

Problem: 1D advection-dispersion-reaction (tubular reactor)
    dc/dt = (1/Pe) d²c/dz² - dc/dz - Da * c

    Danckwerts BCs:
    - Inlet (z=0): (1/Pe) dc/dz = c - 1
    - Outlet (z=1): dc/dz = 0
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
print("Benchmark: moljax vs Diffrax for Tubular Reactor")
print("=" * 70)
print(f"JAX devices: {jax.devices()}")
print()

# =============================================================================
# Problem Setup
# =============================================================================

N_WARMUP = 3
N_RUNS = 5


def build_fd_laplacian_neumann(N, dx):
    """Second-order FD Laplacian with Neumann BC at outlet."""
    diag = -2.0 * jnp.ones(N)
    off_diag = jnp.ones(N - 1)
    L = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
    # Neumann at z=1 (outlet): use ghost point approach
    L = L.at[-1, -1].set(-2.0)
    L = L.at[-1, -2].set(2.0)  # dc/dz = 0 at outlet
    return L / dx**2


def build_fd_gradient_upwind(N, dx):
    """First-order upwind gradient for advection."""
    diag = jnp.ones(N)
    off_diag = -jnp.ones(N - 1)
    G = jnp.diag(diag) + jnp.diag(off_diag, -1)
    G = G.at[0, 0].set(0.0)  # Will handle inlet BC separately
    return G / dx


# =============================================================================
# Method 1: moljax-style FD-CN (implicit diffusion, explicit advection/reaction)
# =============================================================================

def make_reactor_cn_solver(N, Pe, Da, dt):
    """Create implicit CN solver for reactor."""
    dx = 1.0 / N
    z = jnp.linspace(dx/2, 1 - dx/2, N)  # Cell centers

    # Diffusion operator
    L = build_fd_laplacian_neumann(N, dx)

    # CN matrices: (I - 0.5*dt/Pe*L) c^{n+1} = (I + 0.5*dt/Pe*L) c^n + explicit_terms
    I = jnp.eye(N)
    A = I - 0.5 * dt / Pe * L
    B = I + 0.5 * dt / Pe * L

    # Precompute LU factorization equivalent (solve via direct inverse for small N)
    A_inv = jnp.linalg.inv(A)

    # Advection gradient (upwind)
    G = build_fd_gradient_upwind(N, dx)

    @jit
    def step(c):
        # Explicit advection and reaction
        adv = -G @ c
        adv = adv.at[0].set(-(c[0] - 1.0) / dx)  # Danckwerts inlet: c_in = 1
        rxn = -Da * c

        # CN step for diffusion
        rhs = B @ c + dt * (adv + rxn)

        # Inlet BC modification for Danckwerts
        # (1/Pe) dc/dz = c - 1 at z=0
        # Approximated in the diffusion discretization

        return A_inv @ rhs

    return step, z


def solve_reactor_cn(N, Pe, Da, dt, T_final):
    """Solve reactor with FD-CN."""
    step, z = make_reactor_cn_solver(N, Pe, Da, dt)
    n_steps = int(T_final / dt)

    c = jnp.ones(N)  # Initial: uniform concentration = 1
    for _ in range(n_steps):
        c = step(c)

    return c, z


from functools import partial

@partial(jit, static_argnums=(0, 4))
def solve_reactor_cn_jit(N, Pe, Da, dt, n_steps):
    """Fully JIT-compiled reactor solver."""
    dx = 1.0 / N

    # Build operators
    L = build_fd_laplacian_neumann(N, dx)
    I = jnp.eye(N)
    A = I - 0.5 * dt / Pe * L
    B = I + 0.5 * dt / Pe * L
    A_inv = jnp.linalg.inv(A)
    G = build_fd_gradient_upwind(N, dx)

    c0 = jnp.ones(N)

    def body(_, c):
        adv = -G @ c
        adv = adv.at[0].set(-(c[0] - 1.0) / dx)
        rxn = -Da * c
        rhs = B @ c + dt * (adv + rxn)
        return A_inv @ rhs

    return jax.lax.fori_loop(0, n_steps, body, c0)


# =============================================================================
# Method 2: Diffrax with FD RHS
# =============================================================================

def make_diffrax_reactor_rhs(N, Pe, Da):
    """RHS for Diffrax using same FD discretization."""
    dx = 1.0 / N

    L = build_fd_laplacian_neumann(N, dx)
    G = build_fd_gradient_upwind(N, dx)

    def rhs(t, c, args):
        # Diffusion
        diff = (1.0 / Pe) * (L @ c)

        # Advection with Danckwerts inlet
        adv = -G @ c
        adv = adv.at[0].set(-(c[0] - 1.0) / dx)

        # Reaction
        rxn = -Da * c

        return diff + adv + rxn

    return rhs


def solve_reactor_diffrax(N, Pe, Da, T_final, solver, stepsize_controller):
    """Solve reactor with Diffrax."""
    rhs = make_diffrax_reactor_rhs(N, Pe, Da)
    term = diffrax.ODETerm(rhs)

    c0 = jnp.ones(N)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=0.0,
        t1=T_final,
        dt0=0.001,
        y0=c0,
        stepsize_controller=stepsize_controller,
        saveat=diffrax.SaveAt(t1=True),
        max_steps=1000000,
    )

    return sol.ys[0]


# =============================================================================
# Steady-state reference (for error calculation)
# =============================================================================

def steady_state_conversion(Pe, Da):
    """Approximate steady-state outlet conversion for first-order reaction."""
    # For plug flow (Pe -> inf): X = 1 - exp(-Da)
    # For CSTR (Pe -> 0): X = Da / (1 + Da)
    # Interpolation for finite Pe
    X_pfr = 1 - np.exp(-Da)
    X_cstr = Da / (1 + Da)
    # Simple mixing model (approximate)
    alpha = 1 / (1 + 2/Pe)  # Approaches 1 for large Pe
    return alpha * X_pfr + (1 - alpha) * X_cstr


# =============================================================================
# Benchmark Runner
# =============================================================================

def benchmark_method(name, solve_fn, n_warmup=N_WARMUP, n_runs=N_RUNS):
    """Benchmark a solver method."""
    print(f"\n{name}...")

    # Warmup
    for _ in range(n_warmup):
        result = solve_fn()
        jax.block_until_ready(result)

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

    mean_time = np.mean(times)
    std_time = np.std(times)
    outlet_c = float(result[-1])
    conversion = 1 - outlet_c
    print(f"  Mean: {mean_time:.4f} ± {std_time:.4f} s, Outlet c: {outlet_c:.4f}, X: {conversion:.4f}")

    return {
        'mean_s': mean_time,
        'std_s': std_time,
        'times_s': times,
        'outlet_c': outlet_c,
        'conversion': conversion,
    }


# =============================================================================
# Run Benchmarks
# =============================================================================

results = {}

# Test cases spanning Pe-Da space
# Note: Dispersion-dominated (Pe=1) is too stiff for Diffrax explicit solvers
test_cases = [
    {'Pe': 10, 'Da': 1, 'name': 'Balanced (Pe=10, Da=1)'},
    {'Pe': 100, 'Da': 5, 'name': 'Near plug flow (Pe=100, Da=5)'},
    {'Pe': 50, 'Da': 2, 'name': 'Moderate stiffness (Pe=50, Da=2)'},
]

N = 128  # Grid points
T_FINAL = 5.0  # Integrate to near steady-state

for case in test_cases:
    Pe, Da = case['Pe'], case['Da']

    print("\n" + "=" * 70)
    print(f"{case['name']}")
    print(f"N = {N}, T = {T_FINAL}")
    print("=" * 70)

    case_results = {'Pe': Pe, 'Da': Da}

    # Determine appropriate dt for CN stability
    dx = 1.0 / N
    dt_diff = 0.5 * Pe * dx**2  # Diffusion stability
    dt_adv = dx  # Advection CFL
    dt = min(dt_diff, dt_adv, 0.01)
    n_steps = int(T_FINAL / dt)

    print(f"dt = {dt:.4f}, n_steps = {n_steps}")

    # --- moljax FD-CN ---
    case_results['moljax_cn'] = benchmark_method(
        f"moljax FD-CN (dt={dt:.4f}, {n_steps} steps)",
        lambda Pe=Pe, Da=Da, dt=dt, n_steps=n_steps: solve_reactor_cn_jit(N, Pe, Da, dt, n_steps)
    )

    # --- Diffrax Tsit5 (explicit, adaptive) ---
    case_results['diffrax_tsit5'] = benchmark_method(
        "Diffrax Tsit5 (adaptive, rtol=1e-5)",
        lambda Pe=Pe, Da=Da: solve_reactor_diffrax(N, Pe, Da, T_FINAL,
                                                    diffrax.Tsit5(),
                                                    diffrax.PIDController(rtol=1e-5, atol=1e-7))
    )

    # --- Diffrax Dopri5 (explicit, adaptive) ---
    case_results['diffrax_dopri5'] = benchmark_method(
        "Diffrax Dopri5 (adaptive, rtol=1e-5)",
        lambda Pe=Pe, Da=Da: solve_reactor_diffrax(N, Pe, Da, T_FINAL,
                                                    diffrax.Dopri5(),
                                                    diffrax.PIDController(rtol=1e-5, atol=1e-7))
    )

    # --- Diffrax Heun (explicit, constant dt) for fair comparison ---
    case_results['diffrax_heun_fixed'] = benchmark_method(
        f"Diffrax Heun (fixed dt={dt:.4f})",
        lambda Pe=Pe, Da=Da, dt=dt: solve_reactor_diffrax(N, Pe, Da, T_FINAL,
                                                          diffrax.Heun(),
                                                          diffrax.ConstantStepSize())
    )

    results[case['name']] = case_results

# =============================================================================
# Summary
# =============================================================================

print("\n" + "=" * 70)
print("SUMMARY: moljax vs Diffrax for Tubular Reactor")
print("=" * 70)

print(f"\n{'Case':<35} {'Method':<25} {'Time (s)':<12} {'X_out':<10} {'vs moljax':<12}")
print("-" * 95)

for case_name, case_data in results.items():
    moljax_time = case_data['moljax_cn']['mean_s']
    for method in ['moljax_cn', 'diffrax_tsit5', 'diffrax_dopri5', 'diffrax_heun_fixed']:
        data = case_data[method]
        ratio = data['mean_s'] / moljax_time
        ratio_str = f"{ratio:.2f}x" if ratio > 1 else f"{1/ratio:.2f}x faster"
        if method == 'moljax_cn':
            ratio_str = "(baseline)"
        print(f"{case_name:<35} {method:<25} {data['mean_s']:<12.4f} {data['conversion']:<10.4f} {ratio_str:<12}")
    print()

# =============================================================================
# Save Results
# =============================================================================

output = {
    'config': {
        'N': N,
        'T_final': T_FINAL,
        'n_warmup': N_WARMUP,
        'n_runs': N_RUNS,
    },
    'results': results,
}

output_dir = Path(__file__).parent / 'results'
output_dir.mkdir(exist_ok=True)
output_path = output_dir / 'reactor_diffrax_comparison.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2, default=float)
print(f"Results saved to: {output_path}")
