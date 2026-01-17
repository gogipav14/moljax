#!/usr/bin/env python3
"""
Gray-Scott Reaction-Diffusion Example (2D, stiff).

This example demonstrates:
- Multi-field state (u, v fields)
- Reaction-diffusion system with stiffness
- Comparison of explicit vs implicit vs IMEX time stepping
- FFT-based diffusion preconditioner for implicit methods
- Adaptive dt control

The Gray-Scott equations:
    du/dt = Du * Laplacian(u) - u*v^2 + F*(1-u)
    dv/dt = Dv * Laplacian(v) + u*v^2 - (F+k)*v

Run: python -m moljax.examples.gray_scott_2d
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from moljax.core.grid import Grid2D
from moljax.core.bc import BCType
from moljax.core.model import create_gray_scott_model, create_gray_scott_periodic_fft
from moljax.core.stepping import (
    IntegratorType, adaptive_integrate, integrate_fixed_dt,
    adaptive_integrate_imex, integrate_imex_fixed_dt
)
from moljax.core.preconditioners import (
    create_gray_scott_preconditioner,
    create_gray_scott_fft_preconditioner
)
from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.newton_krylov import NKParams
from moljax.core.fft_solvers import create_fft_cache
from moljax.core.utils import get_interior


def create_initial_condition(grid: Grid2D, dtype=jnp.float64):
    """
    Create Gray-Scott initial condition.

    u = 1 everywhere
    v = 0 everywhere except a square perturbation in the center
    """
    X, Y = grid.meshgrid(include_ghost=True)

    # u starts at 1.0
    u = jnp.ones((grid.ny_total, grid.nx_total), dtype=dtype)

    # v starts at 0 with a square perturbation
    v = jnp.zeros((grid.ny_total, grid.nx_total), dtype=dtype)

    # Add perturbation in center
    cx = 0.5 * (grid.x_min + grid.x_max)
    cy = 0.5 * (grid.y_min + grid.y_max)
    Lx = grid.x_max - grid.x_min
    Ly = grid.y_max - grid.y_min

    # Square perturbation
    mask = (jnp.abs(X - cx) < 0.1 * Lx) & (jnp.abs(Y - cy) < 0.1 * Ly)
    u = jnp.where(mask, 0.5, u)
    v = jnp.where(mask, 0.25, v)

    # Add some noise for pattern formation
    key = jax.random.PRNGKey(42)
    u = u + 0.01 * jax.random.uniform(key, u.shape, minval=-1, maxval=1)
    v = v + 0.01 * jax.random.uniform(jax.random.split(key)[0], v.shape, minval=-1, maxval=1)

    return {'u': u.astype(dtype), 'v': v.astype(dtype)}


def run_explicit(model, y0, t_end, dt, method=IntegratorType.RK4):
    """Run with explicit integrator (fixed dt)."""
    print(f"\nRunning explicit {IntegratorType(method).name} with dt={dt:.6f}...")

    start_time = time.time()

    cfl_params = CFLParams(safety=0.9, cfl_diffusion=0.25)
    pid_params = PIDParams(
        safety=0.9, atol=1e-4, rtol=1e-3,
        dt_min=1e-6, dt_max=dt * 2
    )

    result = adaptive_integrate(
        model=model,
        y0=y0,
        t0=0.0,
        t_end=t_end,
        dt0=dt,
        method=method,
        max_steps=50000,
        cfl_params=cfl_params,
        pid_params=pid_params,
        save_every=100
    )

    elapsed = time.time() - start_time

    print(f"  Final t: {float(result.t_final):.4f}")
    print(f"  Accepted steps: {int(result.n_accepted)}")
    print(f"  Rejected steps: {int(result.n_rejected)}")
    print(f"  Status: {int(result.status)}")
    print(f"  Wall time: {elapsed:.2f} s")

    return result, elapsed


def run_implicit(model, y0, t_end, dt, method=IntegratorType.CN, use_fft_precond=False):
    """Run with implicit integrator (adaptive dt)."""
    precond_name = "FFT" if use_fft_precond else "BlockJacobi"
    print(f"\nRunning implicit {IntegratorType(method).name} ({precond_name} precond) with dt0={dt:.6f}...")

    start_time = time.time()

    pid_params = PIDParams(
        safety=0.9, atol=1e-4, rtol=1e-3,
        dt_min=1e-6, dt_max=10.0
    )
    nk_params = NKParams(
        max_newton_iters=10, max_krylov_iters=50,
        newton_tol=1e-6, krylov_tol=1e-5
    )

    if use_fft_precond:
        fft_cache = create_fft_cache(model.grid)
        precond = create_gray_scott_fft_preconditioner(fft_cache)
    else:
        precond = create_gray_scott_preconditioner()

    result = adaptive_integrate(
        model=model,
        y0=y0,
        t0=0.0,
        t_end=t_end,
        dt0=dt,
        method=method,
        max_steps=10000,
        pid_params=pid_params,
        preconditioner=precond,
        nk_params=nk_params,
        save_every=10
    )

    elapsed = time.time() - start_time

    print(f"  Final t: {float(result.t_final):.4f}")
    print(f"  Accepted steps: {int(result.n_accepted)}")
    print(f"  Rejected steps: {int(result.n_rejected)}")
    print(f"  Status: {int(result.status)}")
    print(f"  Wall time: {elapsed:.2f} s")

    return result, elapsed


def run_imex(model, y0, t_end, dt0, fft_cache, diffusivities, use_strang=True):
    """Run with IMEX integrator (diffusion implicit via FFT, reaction explicit)."""
    method_name = "IMEX-Strang" if use_strang else "IMEX-Euler"
    print(f"\nRunning {method_name} with dt0={dt0:.6f}...")

    start_time = time.time()

    pid_params = PIDParams(
        safety=0.9, atol=1e-4, rtol=1e-3,
        dt_min=1e-6, dt_max=5.0  # Can use larger dt than explicit
    )

    result = adaptive_integrate_imex(
        model=model,
        y0=y0,
        t0=0.0,
        t_end=t_end,
        dt0=dt0,
        fft_cache=fft_cache,
        diffusivities=diffusivities,
        use_strang=use_strang,
        max_steps=10000,
        pid_params=pid_params,
        save_every=10
    )

    elapsed = time.time() - start_time

    print(f"  Final t: {float(result.t_final):.4f}")
    print(f"  Accepted steps: {int(result.n_accepted)}")
    print(f"  Rejected steps: {int(result.n_rejected)}")
    print(f"  Status: {int(result.status)}")
    print(f"  Wall time: {elapsed:.2f} s")

    return result, elapsed


def main():
    print("=" * 60)
    print("Gray-Scott Reaction-Diffusion Example")
    print("=" * 60)

    # Setup
    nx, ny = 64, 64
    grid = Grid2D.uniform(nx, ny, 0.0, 2.5, 0.0, 2.5, n_ghost=1)

    # Gray-Scott parameters for interesting patterns
    Du, Dv = 0.16, 0.08
    F, k = 0.04, 0.06

    # Create models
    model = create_gray_scott_model(
        grid=grid, Du=Du, Dv=Dv, F=F, k=k,
        bc_type=BCType.PERIODIC, dtype=jnp.float64
    )

    model_fft, fft_cache, diffusivities = create_gray_scott_periodic_fft(
        grid=grid, Du=Du, Dv=Dv, F=F, k=k, dtype=jnp.float64
    )

    print(f"\nGrid: {nx}x{ny}")
    print(f"Domain: [{grid.x_min}, {grid.x_max}] x [{grid.y_min}, {grid.y_max}]")
    print(f"dx = {grid.dx:.6f}, dy = {grid.dy:.6f}")

    # Initial condition
    y0 = create_initial_condition(grid, dtype=jnp.float64)

    # Integration time
    t_end = 100.0

    # Compute CFL-limited dt for explicit
    dt_diffusion = 0.25 * grid.min_dx2 / max(Du, Dv)
    print(f"\nDiffusion CFL dt ~ {dt_diffusion:.6f}")

    results = {}

    # 1. Run explicit (short run due to CFL constraint)
    result_explicit, time_explicit = run_explicit(
        model, y0, t_end=min(t_end, 10.0),
        dt=dt_diffusion * 0.5,
        method=IntegratorType.RK4
    )
    results['explicit'] = {
        'result': result_explicit,
        'time': time_explicit,
        't_final': float(result_explicit.t_final)
    }

    # 2. Run implicit CN with standard preconditioner
    result_implicit, time_implicit = run_implicit(
        model, y0, t_end=t_end,
        dt=1.0,
        method=IntegratorType.CN,
        use_fft_precond=False
    )
    results['implicit_cn'] = {
        'result': result_implicit,
        'time': time_implicit,
        't_final': float(result_implicit.t_final)
    }

    # 3. Run implicit CN with FFT preconditioner
    result_implicit_fft, time_implicit_fft = run_implicit(
        model, y0, t_end=t_end,
        dt=1.0,
        method=IntegratorType.CN,
        use_fft_precond=True
    )
    results['implicit_cn_fft'] = {
        'result': result_implicit_fft,
        'time': time_implicit_fft,
        't_final': float(result_implicit_fft.t_final)
    }

    # 4. Run IMEX Strang
    result_imex, time_imex = run_imex(
        model_fft, y0, t_end=t_end,
        dt0=0.5,  # Can start with larger dt than explicit
        fft_cache=fft_cache,
        diffusivities=diffusivities,
        use_strang=True
    )
    results['imex_strang'] = {
        'result': result_imex,
        'time': time_imex,
        't_final': float(result_imex.t_final)
    }

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, data in results.items():
        t_final = data['t_final']
        wall_time = data['time']
        n_steps = int(data['result'].n_accepted)
        sim_rate = t_final / wall_time if wall_time > 0 else 0
        print(f"{name:20s}: t={t_final:6.1f}, steps={n_steps:5d}, time={wall_time:6.2f}s, rate={sim_rate:.2f} t/s")

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Extract interior for saving
    np.savez(
        output_dir / "gray_scott_comparison.npz",
        # Grid info
        x=np.array(grid.meshgrid(include_ghost=False)[0][0, :]),
        y=np.array(grid.meshgrid(include_ghost=False)[1][:, 0]),
        # Explicit
        u_explicit=np.array(get_interior(result_explicit.y_final['u'], grid)),
        v_explicit=np.array(get_interior(result_explicit.y_final['v'], grid)),
        t_explicit=results['explicit']['t_final'],
        time_explicit=results['explicit']['time'],
        # Implicit CN
        u_implicit=np.array(get_interior(result_implicit.y_final['u'], grid)),
        v_implicit=np.array(get_interior(result_implicit.y_final['v'], grid)),
        t_implicit=results['implicit_cn']['t_final'],
        time_implicit=results['implicit_cn']['time'],
        dt_history_implicit=np.array(result_implicit.dt_history[:int(result_implicit.n_steps)]),
        # Implicit CN with FFT precond
        u_implicit_fft=np.array(get_interior(result_implicit_fft.y_final['u'], grid)),
        v_implicit_fft=np.array(get_interior(result_implicit_fft.y_final['v'], grid)),
        t_implicit_fft=results['implicit_cn_fft']['t_final'],
        time_implicit_fft=results['implicit_cn_fft']['time'],
        # IMEX
        u_imex=np.array(get_interior(result_imex.y_final['u'], grid)),
        v_imex=np.array(get_interior(result_imex.y_final['v'], grid)),
        t_imex=results['imex_strang']['t_final'],
        time_imex=results['imex_strang']['time'],
        dt_history_imex=np.array(result_imex.dt_history[:int(result_imex.n_steps)]),
    )
    print(f"\nResults saved to {output_dir / 'gray_scott_comparison.npz'}")

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Plot u and v for each method
        methods = [
            ('Explicit RK4', result_explicit, results['explicit']['t_final']),
            ('Implicit CN', result_implicit, results['implicit_cn']['t_final']),
            ('Implicit CN+FFT', result_implicit_fft, results['implicit_cn_fft']['t_final']),
            ('IMEX Strang', result_imex, results['imex_strang']['t_final']),
        ]

        for i, (name, result, t_final) in enumerate(methods):
            u = np.array(get_interior(result.y_final['u'], grid))
            v = np.array(get_interior(result.y_final['v'], grid))

            im0 = axes[0, i].imshow(u, origin='lower', cmap='viridis', vmin=0, vmax=1)
            axes[0, i].set_title(f'u ({name}, t={t_final:.1f})')
            plt.colorbar(im0, ax=axes[0, i])

            im1 = axes[1, i].imshow(v, origin='lower', cmap='viridis', vmin=0, vmax=0.5)
            axes[1, i].set_title(f'v ({name})')
            plt.colorbar(im1, ax=axes[1, i])

        plt.tight_layout()
        plt.savefig(output_dir / "gray_scott_comparison.png", dpi=150)
        print(f"Plot saved to {output_dir / 'gray_scott_comparison.png'}")
        plt.close()

    except ImportError:
        print("(matplotlib not available, skipping plots)")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
