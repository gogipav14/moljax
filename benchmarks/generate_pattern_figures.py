#!/usr/bin/env python3
"""
Generate pattern visualization figures for Gray-Scott, Schnakenberg, and Brusselator.

Shows the physical phenomena captured by moljax: Turing pattern formation across
three canonical reaction-diffusion systems. Uses ETDRK4 (exponential integrator).

Domain sizes are chosen to accommodate several Turing wavelengths for visible patterns.
Schnakenberg uses gamma=100 (gamma=1000 requires implicit reaction treatment).
Brusselator uses b=1.8 (Turing regime; b=3.4 from benchmarks is Hopf/oscillatory).
"""

import jax

jax.config.update("jax_enable_x64", True)

from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

print("Generating pattern visualization figures...")
print(f"JAX devices: {jax.devices()}")


# =============================================================================
# Helpers
# =============================================================================
def etdrk4_coeffs(L_hat, dt):
    E = jnp.exp(dt * L_hat)
    E2 = jnp.exp(dt * L_hat / 2)
    M = 32
    r = jnp.exp(1j * jnp.pi * (jnp.arange(1, M+1) - 0.5) / M)
    LR = dt * L_hat[..., None] + r[None, None, :]
    Q = dt * jnp.real(jnp.mean((jnp.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = dt * jnp.real(jnp.mean((-4 - LR + jnp.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1))
    f2 = dt * jnp.real(jnp.mean((2 + LR + jnp.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = dt * jnp.real(jnp.mean((-4 - 3*LR - LR**2 + jnp.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    return E, E2, Q, f1, f2, f3


def make_lap(N, L):
    dx = L / N
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return jnp.array(-(KX**2 + KY**2))


def make_etdrk4_stepper(nonlin_u_fn, nonlin_v_fn, Eu, E2u, Qu, f1u, f2u, f3u,
                         Ev, E2v, Qv, f1v, f2v, f3v):
    @jax.jit
    def step(state, _):
        u, v = state
        u_hat, v_hat = jnp.fft.fft2(u), jnp.fft.fft2(v)
        Nu_a, Nv_a = jnp.fft.fft2(nonlin_u_fn(u, v)), jnp.fft.fft2(nonlin_v_fn(u, v))

        u_a = jnp.fft.ifft2(E2u * u_hat + Qu * Nu_a).real
        v_a = jnp.fft.ifft2(E2v * v_hat + Qv * Nv_a).real
        Nu_b, Nv_b = jnp.fft.fft2(nonlin_u_fn(u_a, v_a)), jnp.fft.fft2(nonlin_v_fn(u_a, v_a))

        u_b = jnp.fft.ifft2(E2u * u_hat + Qu * Nu_b).real
        v_b = jnp.fft.ifft2(E2v * v_hat + Qv * Nv_b).real
        Nu_c, Nv_c = jnp.fft.fft2(nonlin_u_fn(u_b, v_b)), jnp.fft.fft2(nonlin_v_fn(u_b, v_b))

        u_c = jnp.fft.ifft2(E2u * (E2u * u_hat + Qu * (2*Nu_c - Nu_a))).real
        v_c = jnp.fft.ifft2(E2v * (E2v * v_hat + Qv * (2*Nv_c - Nv_a))).real
        Nu_d, Nv_d = jnp.fft.fft2(nonlin_u_fn(u_c, v_c)), jnp.fft.fft2(nonlin_v_fn(u_c, v_c))

        u_new = jnp.fft.ifft2(Eu * u_hat + Nu_a * f1u + 2*(Nu_b + Nu_c) * f2u + Nu_d * f3u).real
        v_new = jnp.fft.ifft2(Ev * v_hat + Nv_a * f1v + 2*(Nv_b + Nv_c) * f2v + Nv_d * f3v).real
        return (u_new, v_new), None
    return step


def run_etdrk4(step_fn, state, n_total, block_size, label):
    _ = step_fn(state, None)
    state[0].block_until_ready()

    n_blocks = n_total // block_size
    remainder = n_total - n_blocks * block_size

    @partial(jax.jit, static_argnums=(1,))
    def run_block(s, n):
        return jax.lax.scan(step_fn, s, None, length=n)[0]

    for i in range(n_blocks):
        state = run_block(state, block_size)
        state[0].block_until_ready()
        u_min, u_max = float(state[0].min()), float(state[0].max())
        if (i+1) % max(1, n_blocks // 5) == 0 or np.isnan(u_min):
            print(f"  [{label}] Block {i+1}/{n_blocks}, u: [{u_min:.4f}, {u_max:.4f}]")
        if np.isnan(u_min):
            print(f"  WARNING: NaN in {label}!")
            return state

    if remainder > 0:
        state = run_block(state, remainder)
        state[0].block_until_ready()

    u_min, u_max = float(state[0].min()), float(state[0].max())
    print(f"  [{label}] Final u: [{u_min:.4f}, {u_max:.4f}]")
    return state


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

lap = make_lap(N, L)
coeffs_u = etdrk4_coeffs(Du * lap, dt)
coeffs_v = etdrk4_coeffs(Dv * lap, dt)

# Distributed random IC: perturb (u=1, v=0) trivial state across entire domain
# This produces domain-filling Turing patterns (verified against conventional
# RK4+FD Laplacian in verify_patterns_conventional.py)
np.random.seed(42)
u0 = jnp.array(1.0 - 0.5 * np.random.rand(N, N))
v0 = jnp.array(0.25 * np.random.rand(N, N))

gs_step = make_etdrk4_stepper(
    lambda u, v: -u * v**2 + F_gs * (1 - u),
    lambda u, v: u * v**2 - (F_gs + k_gs) * v,
    *coeffs_u, *coeffs_v)

n_steps = int(T / dt)
print(f"Running {n_steps} steps (dt={dt})...")
state_gs = run_etdrk4(gs_step, (u0, v0), n_steps, 1000, "GS")


# =============================================================================
# 2. SCHNAKENBERG: Turing morphogenesis (stripes/spots)
#    gamma=100, Du=1, Dv=40, L=10 (fits ~6 Turing wavelengths)
#    Turing wavelength: lambda_T = 2*pi/k_c ~ 1.67, so L=10 fits ~6
# =============================================================================
print("\n--- Schnakenberg (Turing stripes, L=10) ---")

N = 256; L = 10.0; dt = 0.005; T = 500.0
Du_sn, Dv_sn = 1.0, 40.0
a_sn, b_sn, gamma_sn = 0.1, 0.9, 100.0

lap = make_lap(N, L)
coeffs_u = etdrk4_coeffs(Du_sn * lap, dt)
coeffs_v = etdrk4_coeffs(Dv_sn * lap, dt)

u_s = a_sn + b_sn; v_s = b_sn / (a_sn + b_sn)**2
np.random.seed(42)
u0 = jnp.array(u_s + 0.01 * np.random.randn(N, N))
v0 = jnp.array(v_s + 0.01 * np.random.randn(N, N))

sn_step = make_etdrk4_stepper(
    lambda u, v: gamma_sn * (a_sn - u + u**2 * v),
    lambda u, v: gamma_sn * (b_sn - u**2 * v),
    *coeffs_u, *coeffs_v)

n_steps = int(T / dt)
print(f"Running {n_steps} steps (dt={dt})...")
state_sn = run_etdrk4(sn_step, (u0, v0), n_steps, 5000, "Snak")


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

lap = make_lap(N, L)
coeffs_u = etdrk4_coeffs(Du_br * lap, dt)
coeffs_v = etdrk4_coeffs(Dv_br * lap, dt)

u_s = a_br; v_s = b_br / a_br
np.random.seed(42)
u0 = jnp.array(u_s + 0.05 * np.random.randn(N, N))
v0 = jnp.array(v_s + 0.05 * np.random.randn(N, N))

br_step = make_etdrk4_stepper(
    lambda u, v: a_br - (b_br + 1) * u + u**2 * v,
    lambda u, v: b_br * u - u**2 * v,
    *coeffs_u, *coeffs_v)

n_steps = int(T / dt)
print(f"Running {n_steps} steps (dt={dt})...")
state_br = run_etdrk4(br_step, (u0, v0), n_steps, 2000, "Brus")


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
