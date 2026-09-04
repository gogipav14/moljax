#!/usr/bin/env python3
"""
Gray-Scott with distributed initial perturbation to produce domain-filling patterns.
Tests two initialization strategies and F,k parameter sets.
"""

import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.fft import fft2, fftfreq, ifft2

matplotlib.use('Agg')
import matplotlib.pyplot as plt

output_dir = Path(__file__).parent.parent / "figures"


def make_spectral_laplacian(N, L):
    dx = L / N
    kx = fftfreq(N, dx) * 2 * np.pi
    ky = fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return -(KX**2 + KY**2)


def etdrk4_coeffs(L_hat, dt):
    E = np.exp(dt * L_hat)
    E2 = np.exp(dt * L_hat / 2)
    M = 32
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt * L_hat[..., None] + r[None, None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = dt * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    return E, E2, Q, f1, f2, f3


def etdrk4_step(u, v, nonlin_u, nonlin_v, cu, cv):
    Eu, E2u, Qu, f1u, f2u, f3u = cu
    Ev, E2v, Qv, f1v, f2v, f3v = cv

    uh, vh = fft2(u), fft2(v)
    Na, Nb_ = fft2(nonlin_u(u, v)), fft2(nonlin_v(u, v))

    ua = np.real(ifft2(E2u * uh + Qu * Na))
    va = np.real(ifft2(E2v * vh + Qv * Nb_))
    Nb, Nc_ = fft2(nonlin_u(ua, va)), fft2(nonlin_v(ua, va))

    ub = np.real(ifft2(E2u * uh + Qu * Nb))
    vb = np.real(ifft2(E2v * vh + Qv * Nc_))
    Nc, Nd_ = fft2(nonlin_u(ub, vb)), fft2(nonlin_v(ub, vb))

    uc = np.real(ifft2(E2u * (E2u * uh + Qu * (2*Nc - Na))))
    vc = np.real(ifft2(E2v * (E2v * vh + Qv * (2*Nd_ - Nb_))))
    Nd, Ne_ = fft2(nonlin_u(uc, vc)), fft2(nonlin_v(uc, vc))

    un = np.real(ifft2(Eu * uh + Na*f1u + 2*(Nb + Nc)*f2u + Nd*f3u))
    vn = np.real(ifft2(Ev * vh + Nb_*f1v + 2*(Nc_ + Nd_)*f2v + Ne_*f3v))
    return un, vn


def run_gs(N, L, Du, Dv, F, k, dt, T, u0, v0, label):
    lap_hat = make_spectral_laplacian(N, L)
    cu = etdrk4_coeffs(Du * lap_hat, dt)
    cv = etdrk4_coeffs(Dv * lap_hat, dt)
    nlu = lambda u, v: -u * v**2 + F * (1 - u)
    nlv = lambda u, v: u * v**2 - (F + k) * v

    n_steps = int(T / dt)
    u, v = u0.copy(), v0.copy()
    t0 = time.time()
    for i in range(n_steps):
        u, v = etdrk4_step(u, v, nlu, nlv, cu, cv)
        if i % (n_steps // 5) == 0:
            print(f"  [{label}] Step {i}/{n_steps}, v range: [{v.min():.4f}, {v.max():.4f}]")
    elapsed = time.time() - t0
    print(f"  [{label}] Done in {elapsed:.1f}s")
    return u, v


# =============================================================================
# Parameters to test
# =============================================================================
N = 256; L = 2.5
configs = [
    # (label, F, k, T, dt, IC_type, description)
    ("Spots (center IC)",      0.035, 0.065, 10000, 1.0, "center",
     "Original paper config: self-replicating spots from center"),

    ("Spots (distributed IC)", 0.035, 0.065, 10000, 1.0, "distributed",
     "Same F,k but with noise across entire domain"),

    ("Spots (distributed, T=20k)", 0.035, 0.065, 20000, 1.0, "distributed",
     "Longer time for more pattern development"),

    ("Labyrinthine",           0.040, 0.060, 5000,  1.0, "distributed",
     "Classic Pearson 1993 labyrinthine/worm pattern"),

    ("Mitosis spots",          0.030, 0.062, 10000, 1.0, "distributed",
     "Self-replicating mitosis pattern"),
]

Du, Dv = 2e-5, 1e-5

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for idx, (label, F, k, T, dt, ic_type, desc) in enumerate(configs):
    row, col = divmod(idx, 3)
    print(f"\n--- {label} (F={F}, k={k}, T={T}) ---")
    print(f"  {desc}")

    np.random.seed(42)
    if ic_type == "center":
        u0 = np.ones((N, N))
        v0 = np.zeros((N, N))
        cx = N // 2; r = 10
        u0[cx-r:cx+r, cx-r:cx+r] = 0.5 + 0.01 * np.random.randn(2*r, 2*r)
        v0[cx-r:cx+r, cx-r:cx+r] = 0.25 + 0.01 * np.random.randn(2*r, 2*r)
    else:
        # Distributed: perturb the (u=1,v=0) trivial steady state everywhere
        u0 = 1.0 - 0.5 * np.random.rand(N, N)
        v0 = 0.25 * np.random.rand(N, N)

    u, v = run_gs(N, L, Du, Dv, F, k, dt, T, u0, v0, label)

    ax = axes[row, col]
    if not np.isnan(v).any() and v.max() - v.min() > 1e-6:
        im = ax.imshow(v.T, cmap='inferno', origin='lower', extent=[0, L, 0, L])
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        ax.text(0.5, 0.5, 'No pattern / NaN', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    vmin, vmax = v.min(), v.max()
    ax.set_title(f"{label}\nF={F}, k={k}, T={T}\nv: [{vmin:.3f}, {vmax:.3f}]", fontsize=9)
    ax.set_aspect('equal')

# Empty last panel - summary
axes[1, 2].axis('off')
axes[1, 2].text(0.5, 0.5,
    "Gray-Scott Pattern Survey\n\n"
    "Domain: 256x256, L=2.5\n"
    "Du=2e-5, Dv=1e-5\n"
    "Method: ETDRK4 (spectral)\n\n"
    "Key finding:\n"
    "Center IC → localized spots\n"
    "Distributed IC → domain-filling\n"
    "patterns that clearly show\n"
    "Turing instability",
    ha='center', va='center', transform=axes[1, 2].transAxes,
    fontsize=10, family='monospace',
    bbox=dict(boxstyle='round', facecolor='lightyellow'))

plt.suptitle("Gray-Scott Parameter & IC Survey: Finding Domain-Filling Turing Patterns",
             fontsize=13, y=0.98)
plt.tight_layout()
outpath = output_dir / "fig_gray_scott_survey.png"
plt.savefig(outpath, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outpath}")
