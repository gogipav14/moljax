#!/usr/bin/env python3
"""
Generate steep-gradient concentration profiles for the tubular reactor.

Shows that moljax captures physically challenging features:
- Thin inlet boundary layers at high Pe (width ~ 1/Pe)
- Sharp reaction fronts at high Da
- Steep gradients that test numerical resolution

This is the "hard physics" figure: concentration profiles, not just errors.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

from functools import partial

import jax.numpy as jnp
import matplotlib
import numpy as np
from jax import jit

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
from benchmark_utils import setup_benchmark

print("=" * 70)
print("Reactor Steep-Gradient Concentration Profiles")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")


# =============================================================================
# FD Discretization (reused from benchmark_tubular_reactor.py)
# =============================================================================

def create_fd_matrices(N, dz):
    """Create FD matrices for 2nd and 1st derivatives."""
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] = 1.0
        D2[i, i] = -2.0
        D2[i, i+1] = 1.0
    D2 /= dz**2

    D1 = np.zeros((N, N))
    for i in range(1, N-1):
        D1[i, i-1] = -0.5
        D1[i, i+1] = 0.5
    D1[0, 0] = -1.0
    D1[0, 1] = 1.0
    D1[-1, -2] = -1.0
    D1[-1, -1] = 1.0
    D1 /= dz

    return D2, D1


def create_rhs_danckwerts(N, Pe, Da, dz, c_in=1.0):
    """Danckwerts BC RHS."""
    D2, D1 = create_fd_matrices(N, dz)
    D2 = jnp.array(D2)
    D1 = jnp.array(D1)

    @jit
    def rhs(c, t):
        dcdt = (1.0/Pe) * (D2 @ c) - (D1 @ c) - Da * c
        inlet_coeff = (1.0/Pe) / dz
        c0_target = (c_in + inlet_coeff * c[1]) / (1.0 + inlet_coeff)
        dcdt = dcdt.at[0].set(100.0 * (c0_target - c[0]))
        dcdt = dcdt.at[-1].set(100.0 * (c[-2] - c[-1]))
        return dcdt

    return rhs


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


def analytical_danckwerts_steady(z, Pe, Da):
    """Analytical steady-state for Danckwerts BCs."""
    disc = Pe**2 + 4*Pe*Da
    r1 = (Pe + np.sqrt(disc)) / 2
    r2 = (Pe - np.sqrt(disc)) / 2

    M = np.array([
        [r1/Pe - 1, r2/Pe - 1],
        [r1*np.exp(r1), r2*np.exp(r2)]
    ])
    rhs_vec = np.array([-1, 0])

    try:
        AB = np.linalg.solve(M, rhs_vec)
        A, B = AB
        return A * np.exp(r1 * z) + B * np.exp(r2 * z)
    except Exception:
        return np.ones_like(z) * np.nan


def solve_to_steady_state(Pe, Da, N=512, t_final=20.0):
    """Solve reactor to steady state with appropriate grid and timestep."""
    dz = 1.0 / (N - 1)
    z = np.linspace(0, 1, N)
    c0 = jnp.zeros(N)

    rhs_fn = create_rhs_danckwerts(N, Pe, Da, dz)

    # CFL-limited timestep
    dt_adv = 0.4 * dz
    dt_diff = 0.2 * Pe * dz**2
    dt = min(dt_adv, dt_diff, 0.005)
    n_steps = int(t_final / dt)

    print(f"  Pe={Pe}, Da={Da}: N={N}, dt={dt:.2e}, n_steps={n_steps}")

    # Warmup + solve
    _ = solve_rk4(c0, rhs_fn, dt, min(100, n_steps))
    c_final = solve_rk4(c0, rhs_fn, dt, n_steps)
    c_final.block_until_ready()

    c_num = np.array(c_final)
    c_ana = analytical_danckwerts_steady(z, Pe, Da)

    return z, c_num, c_ana


# =============================================================================
# Generate Figure: Steep Gradient Profiles
# =============================================================================

output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

# Configuration: show physically challenging cases
cases = [
    # (Pe, Da, title_suffix, description)
    (10,  1,   'Moderate mixing',     'Baseline: smooth profile'),
    (100, 1,   'Weak mixing',         'Thin inlet boundary layer'),
    (100, 10,  'Fast reaction',       'Steep reaction front + boundary layer'),
    (500, 5,   'Near plug-flow',      'Very thin boundary layer (width ~ 1/Pe)'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, (Pe, Da, subtitle, _desc) in enumerate(cases):
    ax = axes[idx]
    print(f"\nCase {idx+1}: Pe={Pe}, Da={Da} ({subtitle})")

    z, c_num, c_ana = solve_to_steady_state(Pe, Da, N=512, t_final=20.0)

    ax.plot(z, c_num, 'b-', lw=2.0, label='moljax (FD, N=512)')
    if not np.any(np.isnan(c_ana)):
        ax.plot(z, c_ana, 'k--', lw=1.5, alpha=0.8, label='Analytical')
        max_err = np.max(np.abs(c_num - c_ana))
        ax.text(0.97, 0.97, f'$\\|e\\|_\\infty = {max_err:.1e}$',
                transform=ax.transAxes, ha='right', va='top',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    ax.set_xlabel('$z$ (dimensionless position)', fontsize=11)
    ax.set_ylabel('$c$ (concentration)', fontsize=11)
    ax.set_title(f'Pe = {Pe}, Da = {Da}: {subtitle}', fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    # Add inset for high-Pe cases showing the boundary layer
    if Pe >= 100:
        # Zoom into inlet boundary layer region
        bl_width = min(5.0 / Pe, 0.15)  # Approximate boundary layer width
        inset_ax = ax.inset_axes([0.35, 0.45, 0.55, 0.45])
        inset_ax.plot(z, c_num, 'b-', lw=1.5)
        if not np.any(np.isnan(c_ana)):
            inset_ax.plot(z, c_ana, 'k--', lw=1.0, alpha=0.8)
        inset_ax.set_xlim(0, bl_width * 3)
        # Find y range in the boundary layer
        mask = z <= bl_width * 3
        if np.any(mask):
            y_min = min(c_num[mask].min(), 0)
            y_max = c_num[mask].max() * 1.1
            inset_ax.set_ylim(y_min, y_max)
        inset_ax.set_title('Inlet boundary layer', fontsize=9)
        inset_ax.grid(True, alpha=0.3)
        ax.indicate_inset_zoom(inset_ax, edgecolor='red', lw=1.5)

plt.suptitle('Tubular Reactor: Concentration Profiles with Steep Gradients\n'
             '(Axial dispersion model, Danckwerts BC, steady-state)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "fig_reactor_steep_gradients.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_reactor_steep_gradients.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_reactor_steep_gradients.pdf'}")


# =============================================================================
# Figure 2: Pe progression (same Da, increasing Pe) — boundary layer thinning
# =============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
Da_fixed = 5
Pe_values = [10, 50, 100, 200, 500]
colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(Pe_values)))

print(f"\n\nPe Progression (Da={Da_fixed} fixed):")
for Pe, color in zip(Pe_values, colors, strict=False):
    print(f"  Pe={Pe}...")
    z, c_num, c_ana = solve_to_steady_state(Pe, Da_fixed, N=512, t_final=20.0)
    ax2.plot(z, c_num, '-', color=color, lw=2.0, label=f'Pe = {Pe}')

ax2.set_xlabel('$z$ (dimensionless position)', fontsize=12)
ax2.set_ylabel('$c$ (concentration)', fontsize=12)
ax2.set_title(f'Effect of P\u00e9clet Number on Concentration Profile\n'
              f'(Da = {Da_fixed}, Danckwerts BC, N = 512)',
              fontsize=13, fontweight='bold')
ax2.legend(fontsize=11, title='P\u00e9clet number')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)
ax2.set_ylim(bottom=0)

# Add inset zooming into inlet
inset = ax2.inset_axes([0.4, 0.4, 0.5, 0.5])
for Pe, color in zip(Pe_values, colors, strict=False):
    z, c_num, _ = solve_to_steady_state(Pe, Da_fixed, N=512, t_final=20.0)
    inset.plot(z, c_num, '-', color=color, lw=1.5)
inset.set_xlim(0, 0.1)
inset.set_title('Inlet region (z < 0.1)', fontsize=9)
inset.grid(True, alpha=0.3)
ax2.indicate_inset_zoom(inset, edgecolor='red', lw=1.5)

plt.tight_layout()
plt.savefig(output_dir / "fig_reactor_pe_progression.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_reactor_pe_progression.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_reactor_pe_progression.pdf'}")

print("\nDone! All reactor steep-gradient figures generated.")
