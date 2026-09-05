#!/usr/bin/env python3
"""
Gray-Scott benchmark: moljax vs Diffrax at ACTUAL t=10000.

This benchmark compares moljax exponential integrators against Diffrax ODE solvers
for the stiff Gray-Scott reaction-diffusion system.

IMPORTANT: If Diffrax is slow or fails, that is a VALID RESULT showing moljax's
advantage for stiff reaction-diffusion problems. NO SCALING - actual walltime to t=10000.

Gray-Scott system:
    du/dt = Du * lap(u) - u*v^2 + F*(1-u)
    dv/dt = Dv * lap(v) + u*v^2 - (F+k)*v

Parameters: Du=2e-5, Dv=1e-5, F=0.035, k=0.065 (spot pattern)
Grid: 256x256, t in [0, 10000]
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from benchmark_utils import add_benchmark_args, compute_stats, setup_benchmark

# Try to import Diffrax
try:
    import diffrax
    from diffrax import (
        Dopri5,
        Kvaerno5,
        ODETerm,
        PIDController,
        SaveAt,
        Tsit5,
    )
    HAS_DIFFRAX = True
except ImportError:
    HAS_DIFFRAX = False
    print("WARNING: Diffrax not installed. Run: pip install diffrax")

# Configuration - same as benchmark_gray_scott.py
N = 256
Du = 2e-5
Dv = 1e-5
F = 0.035
k = 0.065
L = 1.0  # Domain length [0, L]
dx = L / N

# ACTUAL simulation time - NO SCALING
T_FINAL = 10000.0

parser = add_benchmark_args()
args = parser.parse_args()
EXPECTED_BACKEND = args.backend if args.backend != 'any' else None
N_REPS = args.n_reps  # Paper uses 10 runs, report median

# Max steps for Diffrax (prevent infinite loops)
MAX_STEPS = 100_000_000  # 100 million steps max


def make_gray_scott_diffrax_rhs(N, L, Du, Dv, F, k):
    """Create Gray-Scott RHS using PS-FFT Laplacian for Diffrax.

    Uses the SAME pseudo-spectral Laplacian as moljax for fair comparison.
    """
    dx = L / N
    kx = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    ky = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    KX, KY = jnp.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)  # PS-FFT Laplacian eigenvalues

    def rhs(t, state, args):
        """Gray-Scott RHS: state shape (N, N, 2) for [u, v]."""
        u = state[..., 0]
        v = state[..., 1]

        # Diffusion via FFT (spectral) - same as moljax
        u_hat = jnp.fft.fft2(u)
        v_hat = jnp.fft.fft2(v)
        lap_u = jnp.real(jnp.fft.ifft2(lap_eig * u_hat))
        lap_v = jnp.real(jnp.fft.ifft2(lap_eig * v_hat))

        # Reaction terms
        uvv = u * v * v
        du = Du * lap_u - uvv + F * (1 - u)
        dv = Dv * lap_v + uvv - (F + k) * v

        return jnp.stack([du, dv], axis=-1)

    return rhs


def make_initial_condition(N, L, seed=42):
    """Create Gray-Scott initial condition (same as benchmark_gray_scott.py)."""
    np.random.seed(seed)
    u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
    v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1

    # Add localized perturbation
    cx, cy = N // 2, N // 2
    r = 10
    u0[cx-r:cx+r, cy-r:cy+r] = 0.5
    v0[cx-r:cx+r, cy-r:cy+r] = 0.25

    return jnp.array(u0), jnp.array(v0)


def run_diffrax_solver(solver, controller, rtol, atol, u0, v0, max_steps=MAX_STEPS):
    """Run Diffrax solver to t=10000.

    Returns dict with timing results or failure info.
    """
    rhs = make_gray_scott_diffrax_rhs(N, L, Du, Dv, F, k)
    term = ODETerm(rhs)

    # Stack u, v for Diffrax
    y0 = jnp.stack([u0, v0], axis=-1)

    # Only save final state
    saveat = SaveAt(t1=True)

    solver_name = solver.__class__.__name__
    print(f"\n   Running {solver_name} (rtol={rtol}, atol={atol})...")

    # Warmup (compile)
    try:
        sol_warmup = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=10.0, dt0=0.01,  # Short warmup
            y0=y0, saveat=saveat, stepsize_controller=controller,
            max_steps=10000
        )
        sol_warmup.ys.block_until_ready()
    except Exception as e:
        print(f"   Warmup failed: {e}")
        return {
            'solver': solver_name,
            'status': 'warmup_failed',
            'error': str(e),
        }

    # Timed runs
    times = []
    final_steps = None
    for rep in range(N_REPS):
        try:
            t0 = time.perf_counter()
            sol = diffrax.diffeqsolve(
                term, solver, t0=0.0, t1=T_FINAL, dt0=0.01,
                y0=y0, saveat=saveat, stepsize_controller=controller,
                max_steps=max_steps
            )
            sol.ys.block_until_ready()
            elapsed = time.perf_counter() - t0

            # Check for NaN
            if not jnp.all(jnp.isfinite(sol.ys)):
                print(f"   Rep {rep+1}: NaN detected!")
                return {
                    'solver': solver_name,
                    'status': 'nan_detected',
                    'rep': rep + 1,
                }

            times.append(elapsed)
            final_steps = int(sol.stats['num_steps'])
            print(f"   Rep {rep+1}: {elapsed:.2f}s ({final_steps:,} steps)")

        except Exception as e:
            print(f"   Rep {rep+1} failed: {e}")
            return {
                'solver': solver_name,
                'status': 'failed',
                'error': str(e),
                'rep': rep + 1,
            }

    median, iqr = compute_stats(times)
    return {
        'solver': solver_name,
        'status': 'completed',
        'time_s': float(median),
        'iqr_s': float(iqr),
        'steps': final_steps,
        't_final': T_FINAL,
        'rtol': rtol,
        'atol': atol,
    }


def main():
    print("=" * 70)
    print("Gray-Scott Diffrax Benchmark (ACTUAL t=10000, NO SCALING)")
    print("=" * 70)

    device = setup_benchmark(expected_backend=EXPECTED_BACKEND)

    print(f"\nGrid: {N}x{N}, Du={Du}, Dv={Dv}, F={F}, k={k}")
    print(f"Simulation: t=0 to t={T_FINAL} (actual, no extrapolation)")
    print(f"N_REPS: {N_REPS}")
    print(f"MAX_STEPS: {MAX_STEPS:,}")

    # Create initial condition
    u0, v0 = make_initial_condition(N, L)

    results = {'methods': []}

    # Load moljax results for comparison
    moljax_results_path = Path(__file__).parent / 'results' / 'gray_scott.json'
    if moljax_results_path.exists():
        with open(moljax_results_path) as f:
            moljax_data = json.load(f)
        print("\n--- moljax reference (from gray_scott.json) ---")
        for m in moljax_data['methods']:
            print(f"   {m['name']}: {m['time_s']:.2f}s ({m['steps']:,} steps)")
            results['methods'].append({
                'solver': m['name'],
                'source': 'moljax',
                'time_s': m['time_s'],
                'iqr_s': m.get('iqr_s', 0),
                'steps': m['steps'],
                't_final': m['t_final'],
            })
    else:
        print(f"\nWARNING: moljax results not found at {moljax_results_path}")
        print("         Run benchmark_gray_scott.py first!")

    if not HAS_DIFFRAX:
        print("\nERROR: Diffrax not available. Cannot run comparison.")
        results['diffrax_available'] = False
    else:
        print("\n--- Diffrax solvers ---")
        results['diffrax_available'] = True

        # Define solvers and tolerances to test
        # Note: Gray-Scott is STIFF, so explicit solvers may struggle

        # 1. Kvaerno5 (implicit, good for stiff) with adaptive stepping
        print("\n1. Diffrax Kvaerno5 (implicit, adaptive)...")
        controller = PIDController(rtol=1e-4, atol=1e-6)
        result = run_diffrax_solver(
            Kvaerno5(), controller, rtol=1e-4, atol=1e-6, u0=u0, v0=v0
        )
        results['methods'].append(result)

        # 2. Tsit5 (explicit, may be slow for stiff)
        # Note: This is expected to be slow or fail for stiff Gray-Scott
        print("\n2. Diffrax Tsit5 (explicit, adaptive)...")
        print("   (Warning: explicit solver on stiff problem - may be very slow)")
        controller = PIDController(rtol=1e-4, atol=1e-6)
        result = run_diffrax_solver(
            Tsit5(), controller, rtol=1e-4, atol=1e-6, u0=u0, v0=v0,
            max_steps=MAX_STEPS
        )
        results['methods'].append(result)

        # 3. Dopri5 with looser tolerance (another explicit option)
        print("\n3. Diffrax Dopri5 (explicit, looser tol)...")
        controller = PIDController(rtol=1e-3, atol=1e-5)
        result = run_diffrax_solver(
            Dopri5(), controller, rtol=1e-3, atol=1e-5, u0=u0, v0=v0,
            max_steps=MAX_STEPS
        )
        results['methods'].append(result)

    # Save results
    results['config'] = {
        'N': N,
        'Du': Du,
        'Dv': Dv,
        'F': F,
        'k': k,
        't_final': T_FINAL,
        'scaling_used': False,
        'max_steps': MAX_STEPS,
        'n_reps': N_REPS,
        'device': device,
        'dtype': 'float64',
        'laplacian': 'pseudo-spectral (-k^2)',
    }

    output_path = Path(__file__).parent / 'results' / 'gray_scott_diffrax.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Gray-Scott t=10000 (actual walltime)")
    print("=" * 70)
    print(f"{'Method':<30} {'Status':<15} {'Time (s)':>12} {'Steps':>12}")
    print("-" * 70)

    for m in results['methods']:
        status = m.get('status', 'completed')
        if status == 'completed':
            time_str = f"{m['time_s']:.2f}"
            steps_str = f"{m['steps']:,}"
        else:
            time_str = "---"
            steps_str = status

        solver = m.get('solver', m.get('name', 'Unknown'))
        source = m.get('source', 'diffrax')
        label = f"{solver} ({source})"
        print(f"{label:<30} {status:<15} {time_str:>12} {steps_str:>12}")

    print("=" * 70)
    print("\nNote: If Diffrax explicit solvers failed or are very slow,")
    print("      this demonstrates moljax's advantage for stiff R-D systems.")


if __name__ == '__main__':
    main()
