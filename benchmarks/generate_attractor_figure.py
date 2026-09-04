#!/usr/bin/env python3
"""
Generate 3x3 attractor divergence figure: three R-D systems x three methods.

Rows: Gray-Scott, Schnakenberg, Brusselator
Columns: RK4, IMEX-Strang, ETDRK4

Each panel shows the computed Turing pattern (u field) with wall time and NFE.
All methods use identical parameters, grid, and initial conditions per system.
The different treatment of the stiff linear operator steers each solver onto a
different pattern attractor branch during the transient growth phase.
"""

import jax

jax.config.update("jax_enable_x64", True)

import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)

print("Generating 3x3 attractor divergence figure...")
print(f"JAX devices: {jax.devices()}")


# =============================================================================
# Helpers
# =============================================================================
def make_lap(N, L):
    dx = L / N
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return jnp.array(-(KX**2 + KY**2))


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


def run_scan(step_fn, state, n_total, block_size, label):
    """Run a JIT-scanned integration with timing."""
    # Warmup (exclude from timing)
    _ = step_fn(state, None)
    state[0].block_until_ready()

    @partial(jax.jit, static_argnums=(1,))
    def run_block(s, n):
        return jax.lax.scan(step_fn, s, None, length=n)[0]

    # Warmup the scan
    if n_total > block_size:
        warmup_state = run_block(state, block_size)
        warmup_state[0].block_until_ready()

    # Timed run
    t0 = time.perf_counter()
    n_blocks = n_total // block_size
    remainder = n_total - n_blocks * block_size
    s = state
    for i in range(n_blocks):
        s = run_block(s, block_size)
        s[0].block_until_ready()
        umin, umax = float(s[0].min()), float(s[0].max())
        if np.isnan(umin):
            print(f"  [{label}] NaN at block {i+1}!")
            return s, time.perf_counter() - t0
        if (i + 1) % max(1, n_blocks // 3) == 0:
            print(f"  [{label}] Block {i+1}/{n_blocks}, u: [{umin:.4f}, {umax:.4f}]")
    if remainder > 0:
        s = run_block(s, remainder)
        s[0].block_until_ready()
    elapsed = time.perf_counter() - t0
    umin, umax = float(s[0].min()), float(s[0].max())
    print(f"  [{label}] Done in {elapsed:.2f}s, u: [{umin:.4f}, {umax:.4f}]")
    return s, elapsed


# =============================================================================
# System definitions
# =============================================================================
systems = []

# --- Gray-Scott ---
N_gs = 128; L_gs = 2.5; T_gs = 5000.0
Du_gs, Dv_gs = 2e-5, 1e-5
F_gs, k_gs = 0.035, 0.065
lap_gs = make_lap(N_gs, L_gs)

np.random.seed(42)
u0_gs = jnp.array(1.0 - 0.5 * np.random.rand(N_gs, N_gs))
v0_gs = jnp.array(0.25 * np.random.rand(N_gs, N_gs))

def gs_rxn_u(u, v):
    return -u * v**2 + F_gs * (1 - u)
def gs_rxn_v(u, v):
    return u * v**2 - (F_gs + k_gs) * v

# Max spectral eigenvalue: Dv * (pi*N/L)^2 = 1e-5 * (pi*128/2.5)^2 = 1e-5 * 25872 = 0.259
# RK4 stability: dt < 2.8/0.259 = 10.8 → dt=5.0 ok (reaction limited: F+k=0.1, so dt<10 ok)
# Actually the reaction is slow, so dt_rk4 = 1.0 should be fine
systems.append({
    'name': 'Gray--Scott', 'field': 'v', 'cmap': 'inferno',
    'N': N_gs, 'L': L_gs, 'T': T_gs,
    'Du': Du_gs, 'Dv': Dv_gs, 'lap': lap_gs,
    'u0': u0_gs, 'v0': v0_gs,
    'rxn_u': gs_rxn_u, 'rxn_v': gs_rxn_v,
    'dt_rk4': 0.5, 'dt_imex': 0.5, 'dt_etd': 1.0,
    'nfe_per_step': {'RK4': 4, 'IMEX': 2, 'ETDRK4': 4},
    'block_rk4': 2000, 'block_imex': 2000, 'block_etd': 1000,
})

# --- Schnakenberg ---
N_sn = 128; L_sn = 10.0; T_sn = 500.0
Du_sn, Dv_sn = 1.0, 40.0
a_sn, b_sn, gamma_sn = 0.1, 0.9, 100.0
lap_sn = make_lap(N_sn, L_sn)

us_sn = a_sn + b_sn; vs_sn = b_sn / (a_sn + b_sn)**2
np.random.seed(42)
u0_sn = jnp.array(us_sn + 0.01 * np.random.randn(N_sn, N_sn))
v0_sn = jnp.array(vs_sn + 0.01 * np.random.randn(N_sn, N_sn))

def sn_rxn_u(u, v):
    return gamma_sn * (a_sn - u + u**2 * v)
def sn_rxn_v(u, v):
    return gamma_sn * (b_sn - u**2 * v)

# Max spectral eigenvalue (2D corner mode): Dv * 2*(pi*N/L)^2 = 40*2*1617 = 129,356
# RK4: dt < 2.785/129356 = 2.15e-5 → need dt=2e-5 → 25M steps / 100M NFE!
# This is INFEASIBLE in practice: ~75 min wall time vs seconds for IMEX/ETDRK4.
# We deliberately show dt=4e-5 going unstable to make this point.
# IMEX: implicit diffusion, explicit reaction: dt < 2.8/100 = 0.028 → use 0.005
# ETDRK4: same reaction constraint: dt < 0.028 → use 0.005
systems.append({
    'name': 'Schnakenberg', 'field': 'u', 'cmap': 'viridis',
    'N': N_sn, 'L': L_sn, 'T': T_sn,
    'Du': Du_sn, 'Dv': Dv_sn, 'lap': lap_sn,
    'u0': u0_sn, 'v0': v0_sn,
    'rxn_u': sn_rxn_u, 'rxn_v': sn_rxn_v,
    'dt_rk4': 4e-5, 'dt_imex': 0.005, 'dt_etd': 0.005,
    'nfe_per_step': {'RK4': 4, 'IMEX': 2, 'ETDRK4': 4},
    'block_rk4': 50000, 'block_imex': 5000, 'block_etd': 5000,
    'rk4_infeasible': True,  # Flag: RK4 is unstable at any practical dt
})

# --- Brusselator ---
N_br = 128; L_br = 5.0; T_br = 100.0
Du_br, Dv_br = 0.01, 0.1
a_br, b_br = 1.0, 1.8
lap_br = make_lap(N_br, L_br)

us_br, vs_br = a_br, b_br / a_br
np.random.seed(42)
u0_br = jnp.array(us_br + 0.05 * np.random.randn(N_br, N_br))
v0_br = jnp.array(vs_br + 0.05 * np.random.randn(N_br, N_br))

def br_rxn_u(u, v):
    return a_br - (b_br + 1) * u + u**2 * v
def br_rxn_v(u, v):
    return b_br * u - u**2 * v

# Max spectral eigenvalue: Dv * (pi*N/L)^2 = 0.1*(pi*128/5)^2 = 0.1*6468 = 647
# RK4: dt < 2.8/647 = 0.0043 → use 0.002
# Reaction: |lambda_rxn| ~ 1, dt < 2.8 → no constraint
# IMEX: dt=0.05 (implicit diffusion)
# ETDRK4: dt=0.5
systems.append({
    'name': 'Brusselator', 'field': 'u', 'cmap': 'RdBu_r',
    'N': N_br, 'L': L_br, 'T': T_br,
    'Du': Du_br, 'Dv': Dv_br, 'lap': lap_br,
    'u0': u0_br, 'v0': v0_br,
    'rxn_u': br_rxn_u, 'rxn_v': br_rxn_v,
    'dt_rk4': 0.002, 'dt_imex': 0.05, 'dt_etd': 0.5,
    'nfe_per_step': {'RK4': 4, 'IMEX': 2, 'ETDRK4': 4},
    'block_rk4': 5000, 'block_imex': 2000, 'block_etd': 200,
})


# =============================================================================
# Run all 9 combinations
# =============================================================================
results = {}  # results[(sys_idx, method)] = (u_field, v_field, walltime, nfe, nsteps, dt)

for si, sys in enumerate(systems):
    name = sys['name']
    lap = sys['lap']
    Du, Dv = sys['Du'], sys['Dv']
    rxn_u, rxn_v = sys['rxn_u'], sys['rxn_v']
    u0, v0 = sys['u0'], sys['v0']
    T = sys['T']

    # --- RK4 ---
    print(f"\n{'='*60}")
    print(f"{name} - RK4")
    print(f"{'='*60}")
    dt = sys['dt_rk4']
    n_steps = int(T / dt)
    nfe = n_steps * 4

    @jax.jit
    def rk4_step(state, _, dt_=dt, Du_=Du, Dv_=Dv, lap_=lap):
        u, v = state
        def rhs(uu, vv):
            lu = jnp.fft.ifft2(lap_ * jnp.fft.fft2(uu)).real
            lv = jnp.fft.ifft2(lap_ * jnp.fft.fft2(vv)).real
            return Du_ * lu + rxn_u(uu, vv), Dv_ * lv + rxn_v(uu, vv)
        k1u, k1v = rhs(u, v)
        k2u, k2v = rhs(u + 0.5*dt_*k1u, v + 0.5*dt_*k1v)
        k3u, k3v = rhs(u + 0.5*dt_*k2u, v + 0.5*dt_*k2v)
        k4u, k4v = rhs(u + dt_*k3u, v + dt_*k3v)
        return (u + (dt_/6)*(k1u + 2*k2u + 2*k3u + k4u),
                v + (dt_/6)*(k1v + 2*k2v + 2*k3v + k4v)), None

    state, elapsed = run_scan(rk4_step, (u0, v0), n_steps, sys['block_rk4'], f"{name}/RK4")
    results[(si, 'RK4')] = (np.array(state[0]), np.array(state[1]), elapsed, nfe, n_steps, dt)

    # --- IMEX-Strang ---
    print(f"\n{'='*60}")
    print(f"{name} - IMEX-Strang")
    print(f"{'='*60}")
    dt = sys['dt_imex']
    n_steps = int(T / dt)
    nfe = n_steps * 2

    imex_fu_half = 1.0 / (1.0 - (dt / 2) * Du * lap)
    imex_fv_half = 1.0 / (1.0 - (dt / 2) * Dv * lap)

    @jax.jit
    def imex_step(state, _, dt_=dt, fu_=imex_fu_half, fv_=imex_fv_half):
        u, v = state
        u_s = jnp.fft.ifft2(jnp.fft.fft2(u) * fu_).real
        v_s = jnp.fft.ifft2(jnp.fft.fft2(v) * fv_).real
        u_m = u_s + dt_ * rxn_u(u_s, v_s)
        v_m = v_s + dt_ * rxn_v(u_s, v_s)
        u_n = jnp.fft.ifft2(jnp.fft.fft2(u_m) * fu_).real
        v_n = jnp.fft.ifft2(jnp.fft.fft2(v_m) * fv_).real
        return (u_n, v_n), None

    state, elapsed = run_scan(imex_step, (u0, v0), n_steps, sys['block_imex'], f"{name}/IMEX")
    results[(si, 'IMEX')] = (np.array(state[0]), np.array(state[1]), elapsed, nfe, n_steps, dt)

    # --- ETDRK4 ---
    print(f"\n{'='*60}")
    print(f"{name} - ETDRK4")
    print(f"{'='*60}")
    dt = sys['dt_etd']
    n_steps = int(T / dt)
    nfe = n_steps * 4

    Eu, E2u, Qu, f1u, f2u, f3u = etdrk4_coeffs(Du * lap, dt)
    Ev, E2v, Qv, f1v, f2v, f3v = etdrk4_coeffs(Dv * lap, dt)

    @jax.jit
    def etd_step(state, _,
                 Eu_=Eu, E2u_=E2u, Qu_=Qu, f1u_=f1u, f2u_=f2u, f3u_=f3u,
                 Ev_=Ev, E2v_=E2v, Qv_=Qv, f1v_=f1v, f2v_=f2v, f3v_=f3v):
        u, v = state
        uh, vh = jnp.fft.fft2(u), jnp.fft.fft2(v)
        Na, Nb_ = jnp.fft.fft2(rxn_u(u, v)), jnp.fft.fft2(rxn_v(u, v))

        ua = jnp.fft.ifft2(E2u_ * uh + Qu_ * Na).real
        va = jnp.fft.ifft2(E2v_ * vh + Qv_ * Nb_).real
        Nb, Nc_ = jnp.fft.fft2(rxn_u(ua, va)), jnp.fft.fft2(rxn_v(ua, va))

        ub = jnp.fft.ifft2(E2u_ * uh + Qu_ * Nb).real
        vb = jnp.fft.ifft2(E2v_ * vh + Qv_ * Nc_).real
        Nc, Nd_ = jnp.fft.fft2(rxn_u(ub, vb)), jnp.fft.fft2(rxn_v(ub, vb))

        uc = jnp.fft.ifft2(E2u_ * (E2u_ * uh + Qu_ * (2*Nc - Na))).real
        vc = jnp.fft.ifft2(E2v_ * (E2v_ * vh + Qv_ * (2*Nd_ - Nb_))).real
        Nd, Ne_ = jnp.fft.fft2(rxn_u(uc, vc)), jnp.fft.fft2(rxn_v(uc, vc))

        un = jnp.fft.ifft2(Eu_ * uh + Na*f1u_ + 2*(Nb+Nc)*f2u_ + Nd*f3u_).real
        vn = jnp.fft.ifft2(Ev_ * vh + Nb_*f1v_ + 2*(Nc_+Nd_)*f2v_ + Ne_*f3v_).real
        return (un, vn), None

    state, elapsed = run_scan(etd_step, (u0, v0), n_steps, sys['block_etd'], f"{name}/ETDRK4")
    results[(si, 'ETDRK4')] = (np.array(state[0]), np.array(state[1]), elapsed, nfe, n_steps, dt)


# =============================================================================
# PLOTTING: 3x3 grid
# =============================================================================
print("\n--- Generating 3x3 figure ---")

fig, axes = plt.subplots(3, 3, figsize=(13, 12))
methods = ['RK4', 'IMEX', 'ETDRK4']
method_labels = ['RK4 (explicit)', 'IMEX-Strang', 'ETDRK4']

for si, sys in enumerate(systems):
    for mi, method in enumerate(methods):
        ax = axes[si, mi]
        u_field, v_field, wt, nfe, nsteps, dt = results[(si, method)]

        # Choose which field to show
        field = v_field if sys['field'] == 'v' else u_field
        extent = [0, sys['L'], 0, sys['L']]
        is_nan = np.isnan(field).any()

        if is_nan:
            # Show informative "infeasible" panel
            ax.set_facecolor('#f0f0f0')
            ax.text(0.5, 0.55, 'UNSTABLE', ha='center', va='center',
                    transform=ax.transAxes, fontsize=16, fontweight='bold',
                    color='#cc0000')
            ax.text(0.5, 0.35, (
                'Stable dt ≤ 2.2×10⁻⁵\n'
                '→ 25M steps, 100M NFE\n'
                '→ ~75 min wall time'),
                ha='center', va='center', transform=ax.transAxes,
                fontsize=8.5, color='#444444', linespacing=1.4)
            ax.set_xlim(0, sys['L']); ax.set_ylim(0, sys['L'])
        else:
            im = ax.imshow(field.T, cmap=sys['cmap'], origin='lower', extent=extent)
            fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)

        # Title: method header (top row) + timing
        title_lines = []
        if si == 0:
            title_lines.append(method_labels[mi])
        if is_nan:
            title_lines.append(f'dt = {dt:.0e} (unstable)')
        else:
            title_lines.append(f'dt = {dt}, {nsteps:,} steps')
            title_lines.append(f'{wt:.1f} s, {nfe:,} NFE')
        ax.set_title('\n'.join(title_lines), fontsize=9,
                      fontweight='bold' if si == 0 else 'normal')

        # Row labels (left column only)
        if mi == 0:
            ax.set_ylabel(f'{sys["name"]}\n$y$', fontsize=10)
        else:
            ax.set_ylabel('$y$', fontsize=10)

        # Bottom row only
        if si == 2:
            ax.set_xlabel('$x$', fontsize=10)

        ax.set_aspect('equal')

plt.suptitle(
    'Attractor divergence: identical ICs and parameters, different integrators (128² grids)',
    fontsize=12, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.98])

plt.savefig(output_dir / "fig_attractor_divergence.pdf", dpi=300, bbox_inches='tight')
plt.savefig(output_dir / "fig_attractor_divergence.png", dpi=150, bbox_inches='tight')
print(f"\nSaved: {output_dir / 'fig_attractor_divergence.pdf'}")

# Print summary table
print("\n" + "="*80)
print("SUMMARY: Wall time and NFE across all 9 combinations")
print("="*80)
print(f"{'System':<15} {'Method':<12} {'dt':<12} {'Steps':>10} {'NFE':>12} {'Time (s)':>10}")
print("-"*80)
for si, sys in enumerate(systems):
    for method in methods:
        u_f, v_f, wt, nfe, ns, dt = results[(si, method)]
        print(f"{sys['name']:<15} {method:<12} {dt:<12.4g} {ns:>10,} {nfe:>12,} {wt:>10.2f}")
    print()
print("="*80)
print("Done!")
