#!/usr/bin/env python3
"""
Multi-Species Advection-Diffusion Example (2D).

This example demonstrates:
- Multiple transported fields (c1, c2)
- Advection with upwind scheme
- Diffusion with Laplacian
- CFL-limited explicit time stepping
- dt scaling with velocity and diffusivity

The advection-diffusion equation for each species:
    dc_i/dt = D * Laplacian(c_i) - v . grad(c_i)

Run: python -m moljax.examples.advdiff_multispecies
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from moljax.core.grid import Grid2D
from moljax.core.bc import BCType
from moljax.core.model import create_advection_diffusion_model
from moljax.core.stepping import IntegratorType, adaptive_integrate
from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.utils import get_interior


def create_initial_condition(grid: Grid2D, dtype=jnp.float64):
    """
    Create initial condition with two Gaussian blobs.

    c1: centered at (0.25, 0.5)
    c2: centered at (0.75, 0.5)
    """
    X, Y = grid.meshgrid(include_ghost=True)

    sigma = 0.08

    # Species 1: Gaussian centered at (0.25, 0.5)
    x1, y1 = 0.25, 0.5
    c1 = jnp.exp(-((X - x1)**2 + (Y - y1)**2) / (2 * sigma**2))

    # Species 2: Gaussian centered at (0.75, 0.5)
    x2, y2 = 0.75, 0.5
    c2 = jnp.exp(-((X - x2)**2 + (Y - y2)**2) / (2 * sigma**2))

    return {'c1': c1.astype(dtype), 'c2': c2.astype(dtype)}


def run_advdiff(model, y0, t_end, method=IntegratorType.SSPRK3):
    """Run advection-diffusion simulation."""
    print(f"\nRunning {IntegratorType(method).name}...")

    # Use CFL-limited stepping
    cfl_params = CFLParams(
        safety=0.8,
        cfl_advection=0.5,
        cfl_diffusion=0.25,
        dt_min=1e-8,
        dt_max=0.1
    )
    pid_params = PIDParams(
        safety=0.9,
        atol=1e-5,
        rtol=1e-4,
        dt_min=1e-8,
        dt_max=0.1
    )

    # Compute expected CFL dt
    vx = model.params['vx']
    vy = model.params['vy']
    D = model.params['D']
    v_max = abs(vx) + abs(vy)

    dt_adv = model.grid.min_dx / (v_max + 1e-14) * cfl_params.cfl_advection
    dt_diff = model.grid.min_dx2 / (D + 1e-14) * cfl_params.cfl_diffusion
    dt_cfl = min(dt_adv, dt_diff)
    print(f"  Expected CFL dt: {dt_cfl:.6e}")
    print(f"    (advection: {dt_adv:.6e}, diffusion: {dt_diff:.6e})")

    result = adaptive_integrate(
        model=model,
        y0=y0,
        t0=0.0,
        t_end=t_end,
        dt0=dt_cfl,
        method=method,
        max_steps=50000,
        cfl_params=cfl_params,
        pid_params=pid_params,
        save_every=100
    )

    print(f"  Final t: {float(result.t_final):.4f}")
    print(f"  Accepted steps: {int(result.n_accepted)}")
    print(f"  Rejected steps: {int(result.n_rejected)}")
    print(f"  Status: {int(result.status)}")

    return result


def test_dt_scaling():
    """
    Demonstrate that CFL dt scales correctly with dx.

    For advection: dt ~ dx / v
    For diffusion: dt ~ dx^2 / D
    """
    print("\n" + "=" * 60)
    print("Testing dt scaling with grid resolution")
    print("=" * 60)

    resolutions = [32, 64, 128]
    vx, vy = 1.0, 0.5
    D = 0.01

    print(f"\nParameters: vx={vx}, vy={vy}, D={D}")
    print("-" * 50)
    print(f"{'nx':>6} {'dx':>10} {'dt_adv':>12} {'dt_diff':>12} {'dt_ratio':>10}")
    print("-" * 50)

    dt_advs = []
    dt_diffs = []

    for nx in resolutions:
        grid = Grid2D.uniform(nx, nx, 0.0, 1.0, 0.0, 1.0)
        dx = grid.min_dx

        # CFL limits
        v_max = abs(vx) + abs(vy)
        dt_adv = 0.5 * dx / v_max
        dt_diff = 0.25 * dx**2 / D

        dt_advs.append(dt_adv)
        dt_diffs.append(dt_diff)

        # Ratio to coarsest grid
        if len(dt_advs) > 1:
            ratio_adv = dt_advs[0] / dt_adv
            ratio_diff = dt_diffs[0] / dt_diff
            ratio_str = f"{ratio_adv:.2f}/{ratio_diff:.2f}"
        else:
            ratio_str = "1.00/1.00"

        print(f"{nx:>6} {dx:>10.6f} {dt_adv:>12.6e} {dt_diff:>12.6e} {ratio_str:>10}")

    print("-" * 50)
    print(f"Expected scaling: dt_adv ~ dx (ratio ~2x), dt_diff ~ dx^2 (ratio ~4x)")


def main():
    print("=" * 60)
    print("Multi-Species Advection-Diffusion Example")
    print("=" * 60)

    # Setup
    nx, ny = 64, 64
    grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0, n_ghost=1)

    # Parameters
    D = 0.01    # Diffusion coefficient
    vx = 1.0    # x-velocity
    vy = 0.5    # y-velocity

    model = create_advection_diffusion_model(
        grid=grid,
        field_names=['c1', 'c2'],
        D=D,
        vx=vx,
        vy=vy,
        bc_type=BCType.PERIODIC,
        use_upwind=True,
        dtype=jnp.float64
    )

    print(f"\nGrid: {nx}x{ny}")
    print(f"Domain: [{grid.x_min}, {grid.x_max}] x [{grid.y_min}, {grid.y_max}]")
    print(f"dx = {grid.dx:.6f}, dy = {grid.dy:.6f}")
    print(f"Velocity: ({vx}, {vy})")
    print(f"Diffusion: {D}")

    # Initial condition
    y0 = create_initial_condition(grid, dtype=jnp.float64)

    # Run simulation
    # Time to advect across domain: L / v ~ 1.0 / sqrt(1^2 + 0.5^2) ~ 0.9
    t_end = 0.5

    result = run_advdiff(model, y0, t_end, method=IntegratorType.SSPRK3)

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Extract interior for saving
    c1_init = np.array(get_interior(y0['c1'], grid))
    c2_init = np.array(get_interior(y0['c2'], grid))
    c1_final = np.array(get_interior(result.y_final['c1'], grid))
    c2_final = np.array(get_interior(result.y_final['c2'], grid))

    np.savez(
        output_dir / "advdiff_results.npz",
        # Grid info
        x=np.array(grid.meshgrid(include_ghost=False)[0][0, :]),
        y=np.array(grid.meshgrid(include_ghost=False)[1][:, 0]),
        # Initial
        c1_init=c1_init,
        c2_init=c2_init,
        # Final
        t_final=float(result.t_final),
        c1_final=c1_final,
        c2_final=c2_final,
        n_steps=int(result.n_accepted),
        t_history=np.array(result.t_history[:int(result.n_steps)]),
    )
    print(f"\nResults saved to {output_dir / 'advdiff_results.npz'}")

    # Test dt scaling
    test_dt_scaling()

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        # Initial conditions
        im0 = axes[0, 0].imshow(c1_init, origin='lower', cmap='Blues',
                                 extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
        axes[0, 0].set_title('c1 (t=0)')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0, 0])

        im1 = axes[0, 1].imshow(c2_init, origin='lower', cmap='Oranges',
                                 extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
        axes[0, 1].set_title('c2 (t=0)')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 1])

        # Final conditions
        im2 = axes[1, 0].imshow(c1_final, origin='lower', cmap='Blues',
                                 extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
        axes[1, 0].set_title(f'c1 (t={float(result.t_final):.2f})')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im2, ax=axes[1, 0])

        im3 = axes[1, 1].imshow(c2_final, origin='lower', cmap='Oranges',
                                 extent=[grid.x_min, grid.x_max, grid.y_min, grid.y_max])
        axes[1, 1].set_title(f'c2 (t={float(result.t_final):.2f})')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 1])

        # Add velocity arrows
        for ax in axes.flat:
            ax.quiver(0.1, 0.1, vx, vy, color='red', scale=5)

        plt.suptitle(f'Advection-Diffusion: v=({vx},{vy}), D={D}')
        plt.tight_layout()
        plt.savefig(output_dir / "advdiff_plot.png", dpi=150)
        print(f"Plot saved to {output_dir / 'advdiff_plot.png'}")
        plt.close()

    except ImportError:
        print("(matplotlib not available, skipping plots)")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
