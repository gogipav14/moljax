#!/usr/bin/env python3
"""
Benchmark E3: Newton Policy Ablation

Compare step rejection policies when Newton fails:
- terminate: Abort simulation
- ÷2: Halve timestep
- ÷4: Quarter timestep (default)
- ÷8: Reduce by factor of 8

Purpose: Validates Newton failure handling novelty
"""

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import time
import json
from pathlib import Path
import jax.numpy as jnp
from jax import lax

from benchmark_utils import setup_benchmark, add_benchmark_args

parser = add_benchmark_args()
args = parser.parse_args()

# Configuration
N = 64
SHRINK_FACTORS = [None, 2, 4, 8]  # None = terminate on failure
D = 0.01
REACTION_K = 50.0  # Strong nonlinearity to trigger Newton failures
T_FINAL = 1.0
DT_INIT = 0.1  # Intentionally large to trigger failures
NEWTON_TOL = 1e-8
NEWTON_MAXITER = 10
GMRES_TOL = 1e-6
N_TRIALS = 5

print("E3: Newton Policy Ablation")
print("=" * 60)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N}x{N}, T_final={T_FINAL}")
print(f"Shrink factors: {SHRINK_FACTORS}")
print("=" * 60)

# Setup problem
dx = 1.0 / N
np.random.seed(42)

# FFT eigenvalues
kx = np.fft.fftfreq(N, dx) * 2 * np.pi
ky = np.fft.fftfreq(N, dx) * 2 * np.pi
KX, KY = np.meshgrid(kx, ky, indexing='ij')
lap_eig = -(KX**2 + KY**2)
lap_eig_jax = jnp.array(lap_eig)

# Operators
@jax.jit
def laplacian_fft(u):
    """FFT-based Laplacian."""
    u_hat = jnp.fft.fft2(u)
    result_hat = u_hat * lap_eig_jax
    return jnp.real(jnp.fft.ifft2(result_hat))

@jax.jit
def reaction(u):
    """Nonlinear reaction: k * u^2 * (1 - u)"""
    return REACTION_K * u * u * (1.0 - u)

@jax.jit
def rhs(u):
    return D * laplacian_fft(u) + reaction(u)

@jax.jit
def fft_solve(u, dt):
    """Solve (I - dt*D*L) x = u via FFT."""
    u_hat = jnp.fft.fft2(u)
    result_hat = u_hat / (1.0 - dt * D * lap_eig_jax)
    return jnp.real(jnp.fft.ifft2(result_hat))

def simple_newton_step(u_n, dt, max_iter=NEWTON_MAXITER, tol=NEWTON_TOL):
    """Simple Newton solver for backward Euler.

    Returns (u_new, converged, n_iters)
    """
    # Backward Euler: u^{n+1} = u^n + dt * f(u^{n+1})
    # Residual: R(u) = u - u^n - dt * f(u)

    u = u_n.copy()

    for k in range(max_iter):
        # Residual
        R = u - u_n - dt * rhs(u)
        res_norm = jnp.linalg.norm(R.flatten())

        if res_norm < tol:
            return u, True, k + 1

        # Simplified Newton: use FFT preconditioner as approximate inverse
        # This is not exact but demonstrates the concept
        du = -fft_solve(R, dt)
        u = u + du

    return u, False, max_iter

results = {'experiments': [], 'config': {}}

for shrink in SHRINK_FACTORS:
    shrink_name = f"÷{shrink}" if shrink else "terminate"
    print(f"\nPolicy: {shrink_name}")

    trial_results = []

    for trial in range(N_TRIALS):
        # Initial condition with some randomness
        np.random.seed(42 + trial)
        u0 = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(N)[:, None] / N) * \
             np.sin(2 * np.pi * np.arange(N)[None, :] / N)
        u0 += 0.05 * np.random.randn(N, N)
        u0 = np.clip(u0, 0.01, 0.99)
        u = jnp.array(u0)

        t = 0.0
        dt = DT_INIT
        n_steps = 0
        n_accepted = 0
        n_rejected = 0
        n_newton_failures = 0
        success = True

        t0 = time.perf_counter()

        while t < T_FINAL and n_steps < 10000:
            u_new, converged, n_newton = simple_newton_step(u, dt)

            if not converged:
                n_newton_failures += 1
                n_rejected += 1

                if shrink is None:
                    # Terminate policy
                    success = False
                    break
                else:
                    # Shrink and retry
                    dt = dt / shrink
                    if dt < 1e-10:
                        success = False
                        break
                    continue

            # Accept step
            u = u_new
            t = t + dt
            n_steps += 1
            n_accepted += 1

            # Grow dt cautiously
            dt = min(dt * 1.2, DT_INIT)

        elapsed = time.perf_counter() - t0

        trial_results.append({
            'success': success and t >= T_FINAL * 0.99,
            'final_t': float(t),
            'n_accepted': n_accepted,
            'n_rejected': n_rejected,
            'n_newton_failures': n_newton_failures,
            'time_s': elapsed,
        })

    # Aggregate trial results
    n_success = sum(1 for r in trial_results if r['success'])
    avg_accepted = np.mean([r['n_accepted'] for r in trial_results])
    avg_rejected = np.mean([r['n_rejected'] for r in trial_results])
    avg_failures = np.mean([r['n_newton_failures'] for r in trial_results])
    avg_time = np.mean([r['time_s'] for r in trial_results])

    print(f"  Success rate: {n_success}/{N_TRIALS}")
    print(f"  Avg accepted steps: {avg_accepted:.1f}")
    print(f"  Avg rejected steps: {avg_rejected:.1f}")
    print(f"  Avg Newton failures: {avg_failures:.1f}")
    print(f"  Avg runtime: {avg_time:.3f}s")

    results['experiments'].append({
        'policy': shrink_name,
        'shrink_factor': shrink,
        'success_rate': n_success / N_TRIALS,
        'avg_accepted': float(avg_accepted),
        'avg_rejected': float(avg_rejected),
        'avg_newton_failures': float(avg_failures),
        'avg_time_s': float(avg_time),
        'trials': trial_results,
    })

# Save results
results['config'] = {
    'grid_size': N,
    'D': D,
    'reaction_k': REACTION_K,
    't_final': T_FINAL,
    'dt_init': DT_INIT,
    'newton_tol': NEWTON_TOL,
    'newton_maxiter': NEWTON_MAXITER,
    'n_trials': N_TRIALS,
    'device': device_str,
}

output_path = Path(__file__).parent / 'results' / 'newton_policy_ablation.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY: Newton Failure Policy Ablation")
print("=" * 70)
print(f"{'Policy':>12} {'Success':>10} {'Accepted':>10} {'Rejected':>10} {'Time':>10}")
print("-" * 70)
for exp in results['experiments']:
    print(f"{exp['policy']:>12} {exp['success_rate']*100:>9.0f}% "
          f"{exp['avg_accepted']:>10.1f} {exp['avg_rejected']:>10.1f} "
          f"{exp['avg_time_s']:>9.3f}s")
print("=" * 70)
