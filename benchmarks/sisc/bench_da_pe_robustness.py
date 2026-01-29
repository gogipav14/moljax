#!/usr/bin/env python3
"""
Benchmark: Robustness across Damköhler and Péclet regimes.

Sweeps (Da, Pe) parameter space for the advection-diffusion-reaction equation:
    u_t = D * Lap(u) - v * grad(u) - k * u

where:
    Da = k * L^2 / D    (Damköhler number: reaction/diffusion timescale)
    Pe = v * L / D      (Péclet number: advection/diffusion ratio)

Reports GMRES iteration statistics with FFT diffusion preconditioner.

Output: results/da_pe_robustness.json
"""

import json
import os
import time
from pathlib import Path
from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np

# Ensure 64-bit precision
jax.config.update("jax_enable_x64", True)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Problem parameters
N = 128  # Grid size (N x N)
L = 2 * jnp.pi  # Domain length
dx = L / N

# Fixed base diffusivity (we vary k and v to achieve target Da and Pe)
D = 0.01

# Parameter sweep - use larger values to stress the preconditioner
DA_VALUES = [0.1, 1.0, 10.0, 100.0]
PE_VALUES = [0, 10, 100, 1000, 10000]

# Time stepping - use larger dt to increase stiffness
DT = 0.01  # Larger timestep to stress preconditioner
N_STEPS = 20  # More timesteps for better statistics
N_SOLVES_PER_STEP = 4  # Newton iterations per step (approximate)

GMRES_TOL = 1e-8
GMRES_MAXITER = 500
GMRES_RESTART = 30


def get_device():
    """Get available device."""
    try:
        devices = jax.devices('gpu')
        if devices:
            return devices[0]
    except:
        pass
    return jax.devices('cpu')[0]


DEVICE = get_device()
print(f"Using device: {DEVICE}")


def build_laplacian_eigenvalues(n, dx):
    """FFT eigenvalues for periodic Laplacian."""
    k = jnp.fft.fftfreq(n, d=dx / (2 * jnp.pi))
    kx, ky = jnp.meshgrid(k, k, indexing='ij')
    return -(kx**2 + ky**2)


def build_advection_eigenvalues(n, dx, vx, vy):
    """FFT eigenvalues for periodic advection operator: -v . grad."""
    k = jnp.fft.fftfreq(n, d=dx / (2 * jnp.pi))
    kx, ky = jnp.meshgrid(k, k, indexing='ij')
    # d/dx in Fourier: i*kx, d/dy in Fourier: i*ky
    # -v . grad u = -(vx * du/dx + vy * du/dy)
    # In Fourier: -i*(vx*kx + vy*ky)
    return -1j * (vx * kx + vy * ky)


def fft_diffusion_preconditioner(r, lap_eig, dt, D):
    """Apply FFT-based diffusion preconditioner: (I - dt*D*L)^{-1} r."""
    r_hat = jnp.fft.fft2(r)
    denom = 1.0 - dt * D * lap_eig
    z_hat = r_hat / denom
    return jnp.real(jnp.fft.ifft2(z_hat))


def gmres_iteration_count(matvec, b, precond, tol, maxiter, restart):
    """
    Count GMRES iterations to convergence.
    Uses restarted GMRES with preconditioning.
    Returns (iterations, converged, final_residual_ratio).
    """
    n = b.size
    x = jnp.zeros_like(b)

    b_norm = jnp.linalg.norm(b)
    if b_norm < 1e-14:
        return 0, True, 0.0

    total_iters = 0
    converged = False

    for restart_cycle in range(maxiter // restart + 1):
        r = b - matvec(x)
        r_norm = jnp.linalg.norm(r)

        if r_norm < tol * b_norm:
            converged = True
            break

        # Apply preconditioner
        z = precond(r)
        beta = jnp.linalg.norm(z)

        if beta < 1e-14:
            converged = True
            break

        # Arnoldi process with restart
        m = min(restart, maxiter - total_iters)
        V = jnp.zeros((m + 1, n))
        H = jnp.zeros((m + 1, m))
        V = V.at[0].set(z / beta)

        for j in range(m):
            # Matrix-vector product with preconditioning
            w = precond(matvec(V[j]))

            # Modified Gram-Schmidt
            for i in range(j + 1):
                H = H.at[i, j].set(jnp.dot(w, V[i]))
                w = w - H[i, j] * V[i]

            h_jp1 = jnp.linalg.norm(w)
            H = H.at[j + 1, j].set(h_jp1)

            if h_jp1 > 1e-14:
                V = V.at[j + 1].set(w / h_jp1)

            total_iters += 1

            # Check convergence via least squares residual
            e1 = jnp.zeros(j + 2).at[0].set(beta)
            y, _, _, _ = jnp.linalg.lstsq(H[:j+2, :j+1], e1, rcond=None)

            # Approximate residual
            res_approx = jnp.linalg.norm(H[:j+2, :j+1] @ y - e1)

            if res_approx < tol * b_norm:
                # Compute actual solution
                x = x + V[:j+1].T @ y
                r = b - matvec(x)
                if jnp.linalg.norm(r) < tol * b_norm:
                    converged = True
                    break

        if converged:
            break

        # Update solution for restart
        e1 = jnp.zeros(m + 1).at[0].set(beta)
        y, _, _, _ = jnp.linalg.lstsq(H[:m+1, :m], e1, rcond=None)
        x = x + V[:m].T @ y

        if total_iters >= maxiter:
            break

    final_r = b - matvec(x)
    final_ratio = float(jnp.linalg.norm(final_r) / b_norm)

    return total_iters, converged, final_ratio


def solve_one_newton_step(u, lap_eig, adv_eig, D, k, dt):
    """
    Perform one CN step for advection-diffusion-reaction and count GMRES iterations.

    PDE: u_t = D*Lap(u) - v*grad(u) - k*u
    CN: (I - 0.5*dt*(D*L + A - k*I)) u^{n+1} = (I + 0.5*dt*(D*L + A - k*I)) u^n

    where A is the advection operator.
    """
    shape = u.shape
    n = u.size

    def apply_operators_fft(u_2d):
        """Apply D*Lap + Advection - k*I in Fourier space."""
        u_hat = jnp.fft.fft2(u_2d)
        # Diffusion + Advection
        result_hat = D * lap_eig * u_hat + adv_eig * u_hat
        result = jnp.real(jnp.fft.ifft2(result_hat))
        # Reaction
        result = result - k * u_2d
        return result

    def apply_jacobian(x):
        """Apply (I - 0.5*dt*(D*L + A - k*I)) x."""
        x_2d = x.reshape(shape)
        op_x = apply_operators_fft(x_2d)
        return (x_2d - 0.5 * dt * op_x).ravel()

    def apply_rhs(u_2d):
        """Apply (I + 0.5*dt*(D*L + A - k*I)) u."""
        op_u = apply_operators_fft(u_2d)
        return (u_2d + 0.5 * dt * op_u).ravel()

    def preconditioner(r):
        """Diffusion-only preconditioner: (I - 0.5*dt*D*L)^{-1}."""
        r_2d = r.reshape(shape)
        return fft_diffusion_preconditioner(r_2d, lap_eig, 0.5 * dt, D).ravel()

    # RHS
    b = apply_rhs(u)

    # Count GMRES iterations
    iters, converged, residual = gmres_iteration_count(
        apply_jacobian, b, preconditioner, GMRES_TOL, GMRES_MAXITER, GMRES_RESTART
    )

    # Compute actual solution (direct FFT for linear problem)
    b_2d = b.reshape(shape)
    b_hat = jnp.fft.fft2(b_2d)
    denom = 1.0 - 0.5 * dt * (D * lap_eig + adv_eig) + 0.5 * dt * k
    u_new_hat = b_hat / denom
    u_new = jnp.real(jnp.fft.ifft2(u_new_hat))

    return u_new, iters, converged


def run_da_pe_sweep():
    """Run the (Da, Pe) parameter sweep."""
    print("=" * 70)
    print("Damköhler-Péclet Robustness Sweep")
    print("=" * 70)
    print(f"Grid: {N}x{N}, D={D}, dt={DT}, tol={GMRES_TOL}")
    print(f"Da values: {DA_VALUES}")
    print(f"Pe values: {PE_VALUES}")
    print()

    # Build grid
    x = jnp.linspace(0, L - dx, N)
    X, Y = jnp.meshgrid(x, x, indexing='ij')

    # Eigenvalues for Laplacian (fixed)
    lap_eig = build_laplacian_eigenvalues(N, dx)
    lap_eig = jax.device_put(lap_eig, DEVICE)

    # Initial condition: smooth Gaussian
    u0 = jnp.exp(-((X - jnp.pi)**2 + (Y - jnp.pi)**2) / 0.5)
    u0 = jax.device_put(u0, DEVICE)

    results = {
        'experiments': [],
        'config': {
            'N': N,
            'D': D,
            'L': float(L),
            'dt': DT,
            'n_timesteps': N_STEPS,
            'gmres_tol': GMRES_TOL,
            'gmres_maxiter': GMRES_MAXITER,
            'gmres_restart': GMRES_RESTART,
            'device': str(DEVICE)
        }
    }

    # Store for table
    iteration_table = {}

    for Da in DA_VALUES:
        iteration_table[Da] = {}
        for Pe in PE_VALUES:
            # Compute physical parameters from dimensionless groups
            # Da = k * L^2 / D  =>  k = Da * D / L^2
            # Pe = v * L / D    =>  v = Pe * D / L
            k = Da * D / (L**2)
            v = Pe * D / L

            # Advection in x-direction
            vx, vy = v, 0.0

            # Build advection eigenvalues
            adv_eig = build_advection_eigenvalues(N, dx, vx, vy)
            adv_eig = jax.device_put(adv_eig, DEVICE)

            print(f"Da={Da:6.1f}, Pe={Pe:6.0f}: k={k:.4e}, v={v:.4e}")

            # Collect iteration counts over multiple timesteps
            all_iters = []
            u = u0.copy()

            for step in range(N_STEPS):
                u, iters, converged = solve_one_newton_step(
                    u, lap_eig, adv_eig, D, k, DT
                )
                all_iters.append(iters)

                if not converged:
                    print(f"  WARNING: Step {step} did not converge (iters={iters})")

            # Statistics
            all_iters = np.array(all_iters)
            median_iters = float(np.median(all_iters))
            q25 = float(np.percentile(all_iters, 25))
            q75 = float(np.percentile(all_iters, 75))
            max_iters = int(np.max(all_iters))
            min_iters = int(np.min(all_iters))

            print(f"  GMRES iterations: median={median_iters:.0f}, IQR=[{q25:.0f}-{q75:.0f}], max={max_iters}")

            iteration_table[Da][Pe] = {
                'median': median_iters,
                'q25': q25,
                'q75': q75,
                'max': max_iters,
                'min': min_iters
            }

            results['experiments'].append({
                'Da': Da,
                'Pe': Pe,
                'k': float(k),
                'v': float(v),
                'n_linear_solves': N_STEPS,
                'gmres_stats': {
                    'median': median_iters,
                    'q25': q25,
                    'q75': q75,
                    'max': max_iters,
                    'min': min_iters,
                    'all_iterations': all_iters.tolist()
                }
            })

    # Print summary table
    print("\n" + "=" * 70)
    print("Summary Table: GMRES Iterations (median [IQR])")
    print("=" * 70)

    # Header
    header = f"{'Da':>8} |"
    for Pe in PE_VALUES:
        header += f" Pe={Pe:>4} |"
    print(header)
    print("-" * len(header))

    # Data rows
    for Da in DA_VALUES:
        row = f"{Da:8.1f} |"
        for Pe in PE_VALUES:
            stats = iteration_table[Da][Pe]
            cell = f"{stats['median']:.0f} [{stats['q25']:.0f}-{stats['q75']:.0f}]"
            row += f" {cell:>10} |"
        print(row)

    # Save results
    output_file = RESULTS_DIR / "da_pe_robustness.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == "__main__":
    run_da_pe_sweep()
