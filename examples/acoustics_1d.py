#!/usr/bin/env python3
"""
1D Linear Acoustics Example (Hyperbolic).

This example demonstrates:
- Coupled multi-field system (pressure p, velocity v)
- Wave equation formulation
- CFL-limited explicit time stepping
- dt scales with dx (wave speed)

The 1D linear acoustics equations:
    dp/dt = -K * dv/dx
    dv/dt = -(1/rho) * dp/dx

Wave speed c = sqrt(K/rho).
CFL condition: dt <= dx / c.

Run: python -m moljax.examples.acoustics_1d
"""

import jax
import jax.numpy as jnp
import numpy as np
from pathlib import Path

from moljax.core.grid import Grid1D
from moljax.core.bc import BCType
from moljax.core.model import create_acoustics_1d_model
from moljax.core.stepping import IntegratorType, adaptive_integrate
from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.utils import get_interior


def create_initial_condition(grid: Grid1D, dtype=jnp.float64):
    """
    Create initial condition: Gaussian pressure pulse, zero velocity.
    """
    x = grid.x_coords(include_ghost=True)

    # Gaussian pressure pulse centered at x=0.3
    x0 = 0.3
    sigma = 0.05
    p = jnp.exp(-((x - x0)**2) / (2 * sigma**2))

    # Zero initial velocity
    v = jnp.zeros_like(x)

    return {'p': p.astype(dtype), 'v': v.astype(dtype)}


def run_acoustics(model, y0, t_end, method=IntegratorType.SSPRK3):
    """Run acoustics simulation."""
    print(f"\nRunning {IntegratorType(method).name}...")

    # Wave speed
    K = model.params['K']
    rho = model.params['rho']
    c = np.sqrt(K / rho)

    # CFL dt
    cfl = 0.5
    dt_cfl = cfl * model.grid.dx / c
    print(f"  Wave speed c = {c:.4f}")
    print(f"  CFL dt = {dt_cfl:.6e}")

    cfl_params = CFLParams(
        safety=0.9,
        cfl_wave=cfl,
        dt_min=1e-10,
        dt_max=dt_cfl * 2
    )
    pid_params = PIDParams(
        safety=0.9,
        atol=1e-6,
        rtol=1e-5,
        dt_min=1e-10,
        dt_max=dt_cfl * 2
    )

    result = adaptive_integrate(
        model=model,
        y0=y0,
        t0=0.0,
        t_end=t_end,
        dt0=dt_cfl,
        method=method,
        max_steps=20000,
        cfl_params=cfl_params,
        pid_params=pid_params,
        save_every=50
    )

    print(f"  Final t: {float(result.t_final):.4f}")
    print(f"  Accepted steps: {int(result.n_accepted)}")
    print(f"  Rejected steps: {int(result.n_rejected)}")
    print(f"  Status: {int(result.status)}")

    return result


def test_cfl_scaling():
    """
    Demonstrate that CFL dt scales linearly with dx.

    For waves: dt ~ dx / c
    """
    print("\n" + "=" * 60)
    print("Testing CFL dt scaling with grid resolution")
    print("=" * 60)

    resolutions = [50, 100, 200, 400]
    K, rho = 1.0, 1.0
    c = np.sqrt(K / rho)
    cfl = 0.5

    print(f"\nWave speed c = {c:.4f}, CFL number = {cfl}")
    print("-" * 50)
    print(f"{'nx':>6} {'dx':>10} {'dt_cfl':>12} {'ratio':>10}")
    print("-" * 50)

    dt_cfls = []

    for nx in resolutions:
        grid = Grid1D.uniform(nx, 0.0, 1.0)
        dx = grid.dx
        dt_cfl = cfl * dx / c
        dt_cfls.append(dt_cfl)

        # Ratio to coarsest grid
        if len(dt_cfls) > 1:
            ratio = dt_cfls[0] / dt_cfl
            ratio_str = f"{ratio:.2f}"
        else:
            ratio_str = "1.00"

        print(f"{nx:>6} {dx:>10.6f} {dt_cfl:>12.6e} {ratio_str:>10}")

    print("-" * 50)
    print(f"Expected scaling: dt ~ dx (ratio doubles with nx)")


def main():
    print("=" * 60)
    print("1D Linear Acoustics Example")
    print("=" * 60)

    # Setup
    nx = 200
    grid = Grid1D.uniform(nx, 0.0, 1.0, n_ghost=1)

    # Parameters
    K = 1.0      # Bulk modulus
    rho = 1.0    # Density
    c = np.sqrt(K / rho)  # Wave speed = 1.0

    model = create_acoustics_1d_model(
        grid=grid,
        K=K,
        rho=rho,
        bc_type=BCType.PERIODIC,
        dtype=jnp.float64
    )

    print(f"\nGrid: nx={nx}")
    print(f"Domain: [{grid.x_min}, {grid.x_max}]")
    print(f"dx = {grid.dx:.6f}")
    print(f"Wave speed c = {c:.4f}")

    # Initial condition
    y0 = create_initial_condition(grid, dtype=jnp.float64)

    # Run simulation
    # Time for wave to travel across domain: L / c = 1.0 / 1.0 = 1.0
    # Run for 2 periods
    t_end = 2.0

    result = run_acoustics(model, y0, t_end, method=IntegratorType.SSPRK3)

    # Save results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Extract interior for saving
    x = np.array(grid.x_coords(include_ghost=False))
    p_init = np.array(get_interior(y0['p'], grid))
    v_init = np.array(get_interior(y0['v'], grid))
    p_final = np.array(get_interior(result.y_final['p'], grid))
    v_final = np.array(get_interior(result.y_final['v'], grid))

    np.savez(
        output_dir / "acoustics_results.npz",
        # Grid info
        x=x,
        wave_speed=c,
        # Initial
        p_init=p_init,
        v_init=v_init,
        # Final
        t_final=float(result.t_final),
        p_final=p_final,
        v_final=v_final,
        n_steps=int(result.n_accepted),
        t_history=np.array(result.t_history[:int(result.n_steps)]),
    )
    print(f"\nResults saved to {output_dir / 'acoustics_results.npz'}")

    # Test CFL scaling
    test_cfl_scaling()

    # Compare initial vs final (should be similar for periodic BC after integer periods)
    p_error = np.max(np.abs(p_final - p_init))
    v_error = np.max(np.abs(v_final - v_init))
    print(f"\n  Error after {t_end:.1f} time units (2 periods):")
    print(f"    max|p_final - p_init| = {p_error:.6e}")
    print(f"    max|v_final - v_init| = {v_error:.6e}")

    # Try to plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Pressure
        axes[0].plot(x, p_init, 'b--', label='Initial', linewidth=2)
        axes[0].plot(x, p_final, 'r-', label=f'Final (t={float(result.t_final):.2f})', linewidth=1)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('Pressure p')
        axes[0].set_title('Pressure Wave')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Velocity
        axes[1].plot(x, v_init, 'b--', label='Initial', linewidth=2)
        axes[1].plot(x, v_final, 'r-', label=f'Final (t={float(result.t_final):.2f})', linewidth=1)
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('Velocity v')
        axes[1].set_title('Velocity Wave')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'1D Acoustics: c={c:.2f}, {int(result.n_accepted)} steps')
        plt.tight_layout()
        plt.savefig(output_dir / "acoustics_plot.png", dpi=150)
        print(f"Plot saved to {output_dir / 'acoustics_plot.png'}")
        plt.close()

    except ImportError:
        print("(matplotlib not available, skipping plots)")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
