#!/usr/bin/env python3
"""
Generate pattern visualization figures for Gray-Scott, Schnakenberg, and Brusselator.

Shows the physical phenomena captured by moljax: Turing pattern formation across
three canonical reaction-diffusion systems. Uses the moljax ETDRK4 solver
(etd_integrate) with the Laplacian of the library (second-order central differences).
RD_OUT_DIR sets the output folder.

Domain sizes are chosen to accommodate several Turing wavelengths for visible patterns.
Schnakenberg uses gamma=100 (gamma=1000 requires implicit reaction treatment).
Brusselator uses b=1.8 (Turing regime; b=3.4 from benchmarks is Hopf/oscillatory).
"""

import jax

jax.config.update("jax_enable_x64", True)

from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

from moljax.core.fft_integrators import etd_integrate
from moljax.core.fft_operators import DiffusionOperator
from moljax.core.grid import Grid2D

matplotlib.use('Agg')
import os

import matplotlib.pyplot as plt

output_dir = Path(os.environ.get("RD_OUT_DIR", Path(__file__).parent.parent / "figures"))
output_dir.mkdir(exist_ok=True)

print("Generating pattern visualization figures...")
print(f"JAX devices: {jax.devices()}")


# =============================================================================
# Helpers
# =============================================================================
def run_etdrk4(N, L, Du, Dv, reaction, dt, state, n_total, label):
    """Run n_total steps of the moljax ETDRK4 solver. Return (u, v)."""
    grid = Grid2D.uniform(N, N, 0.0, L, 0.0, L, n_ghost=1)
    linear_ops = {"u": DiffusionOperator(grid, Du, use_rfft=False),
                  "v": DiffusionOperator(grid, Dv, use_rfft=False)}

    def nonlinear_rhs(y, t):
        ru, rv = reaction(y["u"], y["v"])
        return {"u": ru, "v": rv}

    y = etd_integrate({"u": state[0], "v": state[1]}, (0.0, n_total * dt), dt, linear_ops,
                      nonlinear_rhs, method="etdrk4", save_every=n_total)[1][-1]
    u, v = y["u"], y["v"]
    u_min, u_max = float(u.min()), float(u.max())
    if not np.isfinite(u_min):
        print(f"  WARNING: NaN in {label}!")
    print(f"  [{label}] Final u: [{u_min:.4f}, {u_max:.4f}]")
    return u, v


# =============================================================================
# 1. GRAY-SCOTT: Autocatalytic spot pattern
#    F=0.035, k=0.065, Du=2e-5, Dv=1e-5, L=2.5
#    Distributed IC: random perturbation across entire domain for
#    domain-filling Turing pattern (verified against RK4+FD conventional method)
# =============================================================================
print("\n--- Gray-Scott (spot pattern, distributed IC) ---")

N = 256; L = 2.5; dt = 1.0; T = 10000.0
Du, Dv = 2e-5, 1e-5
F_gs, k_gs = 0.035, 0.065


# Distributed random IC: perturb (u=1, v=0) trivial state across entire domain
# This produces domain-filling Turing patterns (verified against conventional
# RK4+FD Laplacian in verify_patterns_conventional.py)
np.random.seed(42)
u0 = jnp.array(1.0 - 0.5 * np.random.rand(N, N))
v0 = jnp.array(0.25 * np.random.rand(N, N))

def gs_reaction(u, v):
    return (-u * v**2 + F_gs * (1 - u), u * v**2 - (F_gs + k_gs) * v)


n_steps = int(round(T / dt))
print(f"Running {n_steps} steps (dt={dt})...")
state_gs = run_etdrk4(N, L, Du, Dv, gs_reaction, dt, (u0, v0), n_steps, "GS")


# =============================================================================
# 2. SCHNAKENBERG: Turing morphogenesis (stripes/spots)
#    gamma=100, Du=1, Dv=40, L=10 (fits ~6 Turing wavelengths)
#    Turing wavelength: lambda_T = 2*pi/k_c ~ 1.67, so L=10 fits ~6
# =============================================================================
print("\n--- Schnakenberg (Turing stripes, L=10) ---")

N = 256; L = 10.0; dt = 0.005; T = 500.0
Du_sn, Dv_sn = 1.0, 40.0
a_sn, b_sn, gamma_sn = 0.1, 0.9, 100.0


u_s = a_sn + b_sn; v_s = b_sn / (a_sn + b_sn)**2
np.random.seed(42)
u0 = jnp.array(u_s + 0.01 * np.random.randn(N, N))
v0 = jnp.array(v_s + 0.01 * np.random.randn(N, N))

def sn_reaction(u, v):
    return (gamma_sn * (a_sn - u + u**2 * v), gamma_sn * (b_sn - u**2 * v))


n_steps = int(round(T / dt))
print(f"Running {n_steps} steps (dt={dt})...")
state_sn = run_etdrk4(N, L, Du_sn, Dv_sn, sn_reaction, dt, (u0, v0), n_steps, "Snak")


# =============================================================================
# 3. BRUSSELATOR: Turing pattern formation
#    b=1.8 (Turing regime: b < 1+a^2=2 for stable homogeneous state,
#    D_v/D_u=10 sufficient for diffusion-driven instability)
#    L=5, Du=0.01, Dv=0.1
#    Turing wavelength: lambda_T ~ 0.59, so L=5 fits ~8.5
# =============================================================================
print("\n--- Brusselator (Turing patterns, b=1.8, L=5) ---")

N = 256; L = 5.0; dt = 0.01; T = 200.0
Du_br, Dv_br = 0.01, 0.1
a_br, b_br = 1.0, 1.8


u_s = a_br; v_s = b_br / a_br
np.random.seed(42)
u0 = jnp.array(u_s + 0.05 * np.random.randn(N, N))
v0 = jnp.array(v_s + 0.05 * np.random.randn(N, N))

def br_reaction(u, v):
    return (a_br - (b_br + 1) * u + u**2 * v, b_br * u - u**2 * v)


n_steps = int(round(T / dt))
print(f"Running {n_steps} steps (dt={dt})...")
state_br = run_etdrk4(N, L, Du_br, Dv_br, br_reaction, dt, (u0, v0), n_steps, "Brus")


# =============================================================================
# PLOTTING
# =============================================================================
print("\n--- Generating figure ---")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel (a): Gray-Scott v field
v_gs = np.array(state_gs[1])
if not np.isnan(v_gs).any() and v_gs.max() - v_gs.min() > 1e-6:
    im0 = axes[0].imshow(v_gs.T, cmap='inferno', origin='lower', extent=[0, 2.5, 0, 2.5])
    axes[0].set_title('(a) Gray--Scott: $v$\n'
                       '$F{=}0.035$, $k{=}0.065$, $t{=}10{,}000$', fontsize=11)
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
else:
    axes[0].text(0.5, 0.5, 'No pattern', ha='center', va='center',
                 transform=axes[0].transAxes, fontsize=14)

# Panel (b): Schnakenberg u field
u_sn = np.array(state_sn[0])
if not np.isnan(u_sn).any() and u_sn.max() - u_sn.min() > 1e-4:
    im1 = axes[1].imshow(u_sn.T, cmap='viridis', origin='lower', extent=[0, 10, 0, 10])
    axes[1].set_title('(b) Schnakenberg: $u$\n'
                       '$\\gamma{=}100$, $D_v/D_u{=}40$, $t{=}500$', fontsize=11)
    plt.colorbar(im1, ax=axes[1], shrink=0.8)
else:
    msg = 'NaN' if np.isnan(u_sn).any() else 'No pattern'
    axes[1].text(0.5, 0.5, msg, ha='center', va='center',
                 transform=axes[1].transAxes, fontsize=14)

# Panel (c): Brusselator u field
u_br = np.array(state_br[0])
if not np.isnan(u_br).any() and u_br.max() - u_br.min() > 1e-4:
    im2 = axes[2].imshow(u_br.T, cmap='RdBu_r', origin='lower', extent=[0, 5, 0, 5])
    axes[2].set_title('(c) Brusselator: $u$\n'
                       '$b{=}1.8$, $D_v/D_u{=}10$, $t{=}200$', fontsize=11)
    plt.colorbar(im2, ax=axes[2], shrink=0.8)
else:
    msg = 'NaN' if np.isnan(u_br).any() else 'No pattern'
    axes[2].text(0.5, 0.5, msg, ha='center', va='center',
                 transform=axes[2].transAxes, fontsize=14)

for ax in axes:
    ax.set_xlabel('$x$', fontsize=11)
    ax.set_ylabel('$y$', fontsize=11)
    ax.set_aspect('equal')

plt.suptitle('Reaction-Diffusion Pattern Formation (ETDRK4, $256 \\times 256$)',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(output_dir / "fig_pattern_gallery.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_pattern_gallery.png", dpi=150, bbox_inches='tight')
print(f"Saved: {output_dir / 'fig_pattern_gallery.pdf'}")
print("Done!")
