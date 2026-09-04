#!/usr/bin/env python3
"""
Independent verification of Turing pattern formation using conventional methods.

This script uses ONLY numpy and scipy (NO JAX, NO moljax) to verify that:
1. The same equations produce Turing patterns with textbook methods
2. The ETDRK4 results match conventional methods qualitatively
3. Pattern wavelengths match linear stability theory predictions

Methods used:
- RK4 + FD Laplacian (pure explicit, textbook) for Gray-Scott and Brusselator
- IMEX-Euler (FFT diffusion + explicit reaction) for Schnakenberg (stiff)
- ETDRK4 (spectral, same as paper) for comparison

All implementations are pure numpy/scipy.
"""

import time
from pathlib import Path

import matplotlib
import numpy as np
from scipy.fft import fft2, fftfreq, ifft2

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

output_dir = Path(__file__).parent.parent / "figures"
output_dir.mkdir(exist_ok=True)


# =============================================================================
# Conventional building blocks (pure numpy)
# =============================================================================

def fd_laplacian_2d(u, dx):
    """Standard 5-point FD Laplacian with periodic BC."""
    return (np.roll(u, 1, 0) + np.roll(u, -1, 0) +
            np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4*u) / dx**2


def rk4_step_2field(rhs_fn, u, v, dt):
    """Classical RK4 for a 2-field system. Textbook implementation."""
    k1u, k1v = rhs_fn(u, v)
    k2u, k2v = rhs_fn(u + 0.5*dt*k1u, v + 0.5*dt*k1v)
    k3u, k3v = rhs_fn(u + 0.5*dt*k2u, v + 0.5*dt*k2v)
    k4u, k4v = rhs_fn(u + dt*k3u, v + dt*k3v)
    u_new = u + dt/6 * (k1u + 2*k2u + 2*k3u + k4u)
    v_new = v + dt/6 * (k1v + 2*k2v + 2*k3v + k4v)
    return u_new, v_new


def imex_euler_step(u, v, rxn_u_fn, rxn_v_fn, L_hat_u, L_hat_v, dt):
    """IMEX-Euler: implicit diffusion (FFT), explicit reaction.
    Standard splitting method (Ascher et al. 1997)."""
    # Explicit reaction
    Ru, Rv = rxn_u_fn(u, v), rxn_v_fn(u, v)
    u_star = u + dt * Ru
    v_star = v + dt * Rv
    # Implicit diffusion via FFT
    u_hat = fft2(u_star) / (1 - dt * L_hat_u)
    v_hat = fft2(v_star) / (1 - dt * L_hat_v)
    return np.real(ifft2(u_hat)), np.real(ifft2(v_hat))


def make_spectral_laplacian(N, L):
    """Pseudo-spectral Laplacian eigenvalues."""
    dx = L / N
    kx = fftfreq(N, dx) * 2 * np.pi
    ky = fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    return -(KX**2 + KY**2)


def etdrk4_coeffs(L_hat, dt):
    """ETDRK4 coefficients via contour integral (Kassam & Trefethen 2005)."""
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


def etdrk4_step(u, v, nonlin_u, nonlin_v, coeffs_u, coeffs_v):
    """Single ETDRK4 step (Cox-Matthews 2002)."""
    Eu, E2u, Qu, f1u, f2u, f3u = coeffs_u
    Ev, E2v, Qv, f1v, f2v, f3v = coeffs_v

    u_hat, v_hat = fft2(u), fft2(v)
    Nu_a, Nv_a = fft2(nonlin_u(u, v)), fft2(nonlin_v(u, v))

    u_a = np.real(ifft2(E2u * u_hat + Qu * Nu_a))
    v_a = np.real(ifft2(E2v * v_hat + Qv * Nv_a))
    Nu_b, Nv_b = fft2(nonlin_u(u_a, v_a)), fft2(nonlin_v(u_a, v_a))

    u_b = np.real(ifft2(E2u * u_hat + Qu * Nu_b))
    v_b = np.real(ifft2(E2v * v_hat + Qv * Nv_b))
    Nu_c, Nv_c = fft2(nonlin_u(u_b, v_b)), fft2(nonlin_v(u_b, v_b))

    u_c = np.real(ifft2(E2u * (E2u * u_hat + Qu * (2*Nu_c - Nu_a))))
    v_c = np.real(ifft2(E2v * (E2v * v_hat + Qv * (2*Nv_c - Nv_a))))
    Nu_d, Nv_d = fft2(nonlin_u(u_c, v_c)), fft2(nonlin_v(u_c, v_c))

    u_new = np.real(ifft2(Eu * u_hat + Nu_a*f1u + 2*(Nu_b + Nu_c)*f2u + Nu_d*f3u))
    v_new = np.real(ifft2(Ev * v_hat + Nv_a*f1v + 2*(Nv_b + Nv_c)*f2v + Nv_d*f3v))
    return u_new, v_new


def turing_analysis(f_u, f_v, g_u, g_v, Du, Dv, L, N_modes=100):
    """Linear stability analysis for Turing instability.
    Returns critical wavenumber and growth rate."""
    k_vals = np.arange(1, N_modes) * 2 * np.pi / L
    growth_rates = []
    for k in k_vals:
        k2 = k**2
        J = np.array([[f_u - Du*k2, f_v],
                       [g_u, g_v - Dv*k2]])
        eigs = np.linalg.eigvals(J)
        growth_rates.append(np.max(np.real(eigs)))
    growth_rates = np.array(growth_rates)
    idx_max = np.argmax(growth_rates)
    return k_vals, growth_rates, k_vals[idx_max], growth_rates[idx_max]


def pattern_contrast(field):
    """Measure pattern contrast: (max-min)/(max+min)."""
    fmin, fmax = field.min(), field.max()
    if fmax + fmin == 0:
        return 0
    return (fmax - fmin) / (abs(fmax) + abs(fmin))


def dominant_wavelength(field, L):
    """Compute dominant spatial wavelength from 2D FFT."""
    N = field.shape[0]
    dx = L / N
    spectrum = np.abs(fft2(field - field.mean()))**2
    kx = fftfreq(N, dx) * 2 * np.pi
    ky = fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K = np.sqrt(KX**2 + KY**2)
    # Radial binning
    k_bins = np.linspace(0, K.max()/2, 50)
    power = np.zeros(len(k_bins)-1)
    for i in range(len(k_bins)-1):
        mask = (K >= k_bins[i]) & (K < k_bins[i+1])
        if mask.any():
            power[i] = spectrum[mask].mean()
    k_centers = 0.5*(k_bins[:-1] + k_bins[1:])
    # Skip k=0 mode
    idx_peak = np.argmax(power[1:]) + 1
    k_dom = k_centers[idx_peak]
    lambda_dom = 2*np.pi/k_dom if k_dom > 0 else np.inf
    return lambda_dom, k_centers, power


# =============================================================================
# 1. GRAY-SCOTT VERIFICATION
# =============================================================================
def run_gray_scott():
    print("=" * 70)
    print("GRAY-SCOTT VERIFICATION")
    print("=" * 70)

    N = 128; L = 2.5; dx = L / N
    Du, Dv = 2e-5, 1e-5
    F, k = 0.035, 0.065

    # --- Linear stability analysis ---
    # Steady state: (u_s, v_s) = (1, 0) is trivial; the patterning arises
    # from the (F/(F+k), (F+k)^2*k/F^2) steady state, but Gray-Scott is
    # more complex (excitable, not classical Turing). Skip formal LSA.

    # --- Initial conditions (same seed) ---
    np.random.seed(42)
    u0 = np.ones((N, N))
    v0 = np.zeros((N, N))
    cx = N // 2; r = 10
    u0[cx-r:cx+r, cx-r:cx+r] = 0.5 + 0.01 * np.random.randn(2*r, 2*r)
    v0[cx-r:cx+r, cx-r:cx+r] = 0.25 + 0.01 * np.random.randn(2*r, 2*r)

    # --- Method 1: RK4 + FD Laplacian (conventional) ---
    def rhs_gs(u, v):
        lap_u = fd_laplacian_2d(u, dx)
        lap_v = fd_laplacian_2d(v, dx)
        uvv = u * v**2
        du = Du * lap_u - uvv + F * (1 - u)
        dv = Dv * lap_v + uvv - (F + k) * v
        return du, dv

    dt_rk4 = 0.5
    T = 10000.0
    n_steps = int(T / dt_rk4)

    # CFL check
    max_eig = 8 * max(Du, Dv) / dx**2  # 2D FD Laplacian spectral radius * D
    print(f"  FD CFL limit: dt < {2/max_eig:.2f}, using dt={dt_rk4}")
    print(f"  Running RK4+FD: {n_steps} steps, T={T}...")

    u_rk4, v_rk4 = u0.copy(), v0.copy()
    t0 = time.time()
    for step in range(n_steps):
        u_rk4, v_rk4 = rk4_step_2field(rhs_gs, u_rk4, v_rk4, dt_rk4)
        if step % 5000 == 0:
            print(f"    Step {step}/{n_steps}, u: [{u_rk4.min():.4f}, {u_rk4.max():.4f}], "
                  f"v: [{v_rk4.min():.6f}, {v_rk4.max():.6f}]")
        if np.isnan(u_rk4).any():
            print(f"    NaN at step {step}! Reducing dt...")
            break
    t_rk4 = time.time() - t0
    print(f"  RK4+FD done in {t_rk4:.1f}s")

    # --- Method 2: ETDRK4 (spectral, same as paper) ---
    np.random.seed(42)
    u0_etd = np.ones((N, N))
    v0_etd = np.zeros((N, N))
    u0_etd[cx-r:cx+r, cx-r:cx+r] = 0.5 + 0.01 * np.random.randn(2*r, 2*r)
    v0_etd[cx-r:cx+r, cx-r:cx+r] = 0.25 + 0.01 * np.random.randn(2*r, 2*r)

    lap_hat = make_spectral_laplacian(N, L)
    dt_etd = 1.0
    coeffs_u = etdrk4_coeffs(Du * lap_hat, dt_etd)
    coeffs_v = etdrk4_coeffs(Dv * lap_hat, dt_etd)
    n_etd = int(T / dt_etd)

    def nonlin_u(u, v):
        return -u * v**2 + F * (1 - u)
    def nonlin_v(u, v):
        return u * v**2 - (F + k) * v

    print(f"  Running ETDRK4: {n_etd} steps, dt={dt_etd}...")
    u_etd, v_etd = u0_etd.copy(), v0_etd.copy()
    t0 = time.time()
    for step in range(n_etd):
        u_etd, v_etd = etdrk4_step(u_etd, v_etd, nonlin_u, nonlin_v, coeffs_u, coeffs_v)
        if step % 2000 == 0:
            print(f"    Step {step}/{n_etd}, u: [{u_etd.min():.4f}, {u_etd.max():.4f}]")
    t_etd = time.time() - t0
    print(f"  ETDRK4 done in {t_etd:.1f}s")

    # --- Analysis ---
    contrast_rk4 = pattern_contrast(v_rk4)
    contrast_etd = pattern_contrast(v_etd)
    lambda_rk4, _, _ = dominant_wavelength(v_rk4, L)
    lambda_etd, _, _ = dominant_wavelength(v_etd, L)

    print(f"\n  Pattern contrast (v field): RK4={contrast_rk4:.4f}, ETDRK4={contrast_etd:.4f}")
    print(f"  Dominant wavelength: RK4={lambda_rk4:.4f}, ETDRK4={lambda_etd:.4f}")

    return {
        'rk4': {'u': u_rk4, 'v': v_rk4, 'time': t_rk4, 'label': 'RK4 + FD Laplacian'},
        'etd': {'u': u_etd, 'v': v_etd, 'time': t_etd, 'label': 'ETDRK4 (spectral)'},
        'params': {'N': N, 'L': L, 'Du': Du, 'Dv': Dv, 'F': F, 'k': k, 'T': T},
    }


# =============================================================================
# 2. SCHNAKENBERG VERIFICATION
# =============================================================================
def run_schnakenberg():
    print("\n" + "=" * 70)
    print("SCHNAKENBERG VERIFICATION")
    print("=" * 70)

    N = 64; L = 10.0
    Du, Dv = 1.0, 40.0
    a, b, gamma = 0.1, 0.9, 100.0

    # --- Linear stability analysis ---
    u_s = a + b; v_s = b / (a + b)**2
    f_u = gamma * (-1 + 2*u_s*v_s)
    f_v = gamma * u_s**2
    g_u = gamma * (-2*u_s*v_s)
    g_v = gamma * (-u_s**2)

    k_vals, growth, k_crit, sigma_max = turing_analysis(f_u, f_v, g_u, g_v, Du, Dv, L)
    lambda_crit = 2*np.pi / k_crit
    print(f"  Steady state: u_s={u_s:.3f}, v_s={v_s:.5f}")
    print(f"  Jacobian: f_u={f_u:.1f}, f_v={f_v:.1f}, g_u={g_u:.1f}, g_v={g_v:.1f}")
    print(f"  Turing analysis: k_crit={k_crit:.2f}, lambda_crit={lambda_crit:.3f}, "
          f"sigma_max={sigma_max:.2f}")
    print(f"  Expected wavelengths in domain: L/lambda = {L/lambda_crit:.1f}")

    # --- Initial conditions ---
    np.random.seed(42)
    u0 = u_s + 0.01 * np.random.randn(N, N)
    v0 = v_s + 0.01 * np.random.randn(N, N)

    # --- Method 1: IMEX-Euler (conventional splitting) ---
    # Too stiff for pure explicit (Dv=40, CFL ~ 3.8e-5)
    # IMEX: diffusion implicit via FFT, reaction explicit
    lap_hat = make_spectral_laplacian(N, L)
    L_hat_u = Du * lap_hat
    L_hat_v = Dv * lap_hat

    def rxn_u(u, v):
        return gamma * (a - u + u**2 * v)
    def rxn_v(u, v):
        return gamma * (b - u**2 * v)

    dt_imex = 0.005
    T = 500.0
    n_steps = int(T / dt_imex)

    print(f"  Running IMEX-Euler: {n_steps} steps, dt={dt_imex}, T={T}...")
    u_imex, v_imex = u0.copy(), v0.copy()
    t0 = time.time()
    for step in range(n_steps):
        u_imex, v_imex = imex_euler_step(u_imex, v_imex, rxn_u, rxn_v,
                                          L_hat_u, L_hat_v, dt_imex)
        if step % 20000 == 0:
            print(f"    Step {step}/{n_steps}, u: [{u_imex.min():.4f}, {u_imex.max():.4f}]")
        if np.isnan(u_imex).any():
            print(f"    NaN at step {step}!")
            break
    t_imex = time.time() - t0
    print(f"  IMEX-Euler done in {t_imex:.1f}s")

    # --- Method 2: ETDRK4 ---
    np.random.seed(42)
    u0_etd = u_s + 0.01 * np.random.randn(N, N)
    v0_etd = v_s + 0.01 * np.random.randn(N, N)

    dt_etd = 0.005
    coeffs_u = etdrk4_coeffs(L_hat_u, dt_etd)
    coeffs_v = etdrk4_coeffs(L_hat_v, dt_etd)
    n_etd = int(T / dt_etd)

    print(f"  Running ETDRK4: {n_etd} steps, dt={dt_etd}...")
    u_etd, v_etd = u0_etd.copy(), v0_etd.copy()
    t0 = time.time()
    for step in range(n_etd):
        u_etd, v_etd = etdrk4_step(u_etd, v_etd, rxn_u, rxn_v, coeffs_u, coeffs_v)
        if step % 20000 == 0:
            print(f"    Step {step}/{n_etd}, u: [{u_etd.min():.4f}, {u_etd.max():.4f}]")
    t_etd = time.time() - t0
    print(f"  ETDRK4 done in {t_etd:.1f}s")

    # --- Analysis ---
    contrast_imex = pattern_contrast(u_imex)
    contrast_etd = pattern_contrast(u_etd)
    lambda_imex, _, _ = dominant_wavelength(u_imex, L)
    lambda_etd, _, _ = dominant_wavelength(u_etd, L)

    print(f"\n  Pattern contrast (u field): IMEX={contrast_imex:.4f}, ETDRK4={contrast_etd:.4f}")
    print(f"  Dominant wavelength: IMEX={lambda_imex:.3f}, ETDRK4={lambda_etd:.3f}")
    print(f"  Theory prediction: lambda_crit={lambda_crit:.3f}")

    return {
        'imex': {'u': u_imex, 'v': v_imex, 'time': t_imex, 'label': 'IMEX-Euler (FFT diffusion)'},
        'etd': {'u': u_etd, 'v': v_etd, 'time': t_etd, 'label': 'ETDRK4 (spectral)'},
        'theory': {'k_vals': k_vals, 'growth': growth, 'k_crit': k_crit,
                   'lambda_crit': lambda_crit, 'sigma_max': sigma_max},
        'params': {'N': N, 'L': L, 'Du': Du, 'Dv': Dv, 'a': a, 'b': b,
                   'gamma': gamma, 'T': T},
    }


# =============================================================================
# 3. BRUSSELATOR VERIFICATION
# =============================================================================
def run_brusselator():
    print("\n" + "=" * 70)
    print("BRUSSELATOR VERIFICATION")
    print("=" * 70)

    N = 128; L = 5.0; dx = L / N
    Du, Dv = 0.01, 0.1
    a_br, b_br = 1.0, 1.8

    # --- Linear stability analysis ---
    u_s = a_br; v_s = b_br / a_br
    f_u = b_br - 1           # = 0.8
    f_v = a_br**2             # = 1.0
    g_u = -b_br               # = -1.8
    g_v = -a_br**2            # = -1.0

    k_vals, growth, k_crit, sigma_max = turing_analysis(f_u, f_v, g_u, g_v, Du, Dv, L)
    lambda_crit = 2*np.pi / k_crit
    print(f"  Steady state: u_s={u_s:.3f}, v_s={v_s:.3f}")
    print(f"  Jacobian: f_u={f_u:.1f}, f_v={f_v:.1f}, g_u={g_u:.1f}, g_v={g_v:.1f}")
    print(f"  Trace={f_u+g_v:.2f} (<0: stable), Det={f_u*g_v - f_v*g_u:.2f} (>0: stable)")
    print(f"  Turing analysis: k_crit={k_crit:.2f}, lambda_crit={lambda_crit:.3f}, "
          f"sigma_max={sigma_max:.3f}")
    print(f"  Expected wavelengths in domain: L/lambda = {L/lambda_crit:.1f}")

    # --- Initial conditions ---
    np.random.seed(42)
    u0 = u_s + 0.05 * np.random.randn(N, N)
    v0 = v_s + 0.05 * np.random.randn(N, N)

    # --- Method 1: RK4 + FD Laplacian ---
    def rhs_brus(u, v):
        lap_u = fd_laplacian_2d(u, dx)
        lap_v = fd_laplacian_2d(v, dx)
        u2v = u**2 * v
        du = Du * lap_u + a_br - (b_br + 1) * u + u2v
        dv = Dv * lap_v + b_br * u - u2v
        return du, dv

    max_eig = 8 * max(Du, Dv) / dx**2
    dt_rk4 = min(0.005, 1.8 / max_eig)  # Safety factor 0.9 on CFL
    T = 200.0
    n_steps = int(T / dt_rk4)

    print(f"  FD CFL limit: dt < {2/max_eig:.5f}, using dt={dt_rk4}")
    print(f"  Running RK4+FD: {n_steps} steps, T={T}...")

    u_rk4, v_rk4 = u0.copy(), v0.copy()
    t0 = time.time()
    for step in range(n_steps):
        u_rk4, v_rk4 = rk4_step_2field(rhs_brus, u_rk4, v_rk4, dt_rk4)
        if step % 10000 == 0:
            print(f"    Step {step}/{n_steps}, u: [{u_rk4.min():.4f}, {u_rk4.max():.4f}]")
        if np.isnan(u_rk4).any():
            print(f"    NaN at step {step}!")
            break
    t_rk4 = time.time() - t0
    print(f"  RK4+FD done in {t_rk4:.1f}s")

    # --- Method 2: ETDRK4 ---
    np.random.seed(42)
    u0_etd = u_s + 0.05 * np.random.randn(N, N)
    v0_etd = v_s + 0.05 * np.random.randn(N, N)

    lap_hat = make_spectral_laplacian(N, L)
    dt_etd = 0.01
    coeffs_u = etdrk4_coeffs(Du * lap_hat, dt_etd)
    coeffs_v = etdrk4_coeffs(Dv * lap_hat, dt_etd)
    n_etd = int(T / dt_etd)

    def nonlin_u(u, v):
        return a_br - (b_br + 1) * u + u**2 * v
    def nonlin_v(u, v):
        return b_br * u - u**2 * v

    print(f"  Running ETDRK4: {n_etd} steps, dt={dt_etd}...")
    u_etd, v_etd = u0_etd.copy(), v0_etd.copy()
    t0 = time.time()
    for step in range(n_etd):
        u_etd, v_etd = etdrk4_step(u_etd, v_etd, nonlin_u, nonlin_v, coeffs_u, coeffs_v)
        if step % 5000 == 0:
            print(f"    Step {step}/{n_etd}, u: [{u_etd.min():.4f}, {u_etd.max():.4f}]")
    t_etd = time.time() - t0
    print(f"  ETDRK4 done in {t_etd:.1f}s")

    # --- Analysis ---
    contrast_rk4 = pattern_contrast(u_rk4)
    contrast_etd = pattern_contrast(u_etd)
    lambda_rk4, _, _ = dominant_wavelength(u_rk4, L)
    lambda_etd, _, _ = dominant_wavelength(u_etd, L)

    print(f"\n  Pattern contrast (u field): RK4={contrast_rk4:.4f}, ETDRK4={contrast_etd:.4f}")
    print(f"  Dominant wavelength: RK4={lambda_rk4:.3f}, ETDRK4={lambda_etd:.3f}")
    print(f"  Theory prediction: lambda_crit={lambda_crit:.3f}")

    return {
        'rk4': {'u': u_rk4, 'v': v_rk4, 'time': t_rk4, 'label': 'RK4 + FD Laplacian'},
        'etd': {'u': u_etd, 'v': v_etd, 'time': t_etd, 'label': 'ETDRK4 (spectral)'},
        'theory': {'k_vals': k_vals, 'growth': growth, 'k_crit': k_crit,
                   'lambda_crit': lambda_crit, 'sigma_max': sigma_max},
        'params': {'N': N, 'L': L, 'Du': Du, 'Dv': Dv, 'a': a_br, 'b': b_br, 'T': T},
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_comparison(gs_results, sn_results, br_results):
    """Create comprehensive comparison figure."""
    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.35, wspace=0.3)

    # --- Row 1: Gray-Scott ---
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(gs_results['rk4']['v'].T, cmap='inferno', origin='lower',
                     extent=[0, gs_results['params']['L']]*2)
    ax1.set_title(f"Gray-Scott: {gs_results['rk4']['label']}\n"
                  f"({gs_results['rk4']['time']:.0f}s)", fontsize=9)
    plt.colorbar(im, ax=ax1, shrink=0.8)

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(gs_results['etd']['v'].T, cmap='inferno', origin='lower',
                     extent=[0, gs_results['params']['L']]*2)
    ax2.set_title(f"Gray-Scott: {gs_results['etd']['label']}\n"
                  f"({gs_results['etd']['time']:.0f}s)", fontsize=9)
    plt.colorbar(im, ax=ax2, shrink=0.8)

    # Power spectra
    ax3 = fig.add_subplot(gs[0, 2:])
    _, k_rk4, p_rk4 = dominant_wavelength(gs_results['rk4']['v'], gs_results['params']['L'])
    _, k_etd, p_etd = dominant_wavelength(gs_results['etd']['v'], gs_results['params']['L'])
    ax3.semilogy(k_rk4, p_rk4 / max(p_rk4.max(), 1e-30), 'b-', label='RK4+FD', linewidth=1.5)
    ax3.semilogy(k_etd, p_etd / max(p_etd.max(), 1e-30), 'r--', label='ETDRK4', linewidth=1.5)
    ax3.set_xlabel('Wavenumber k')
    ax3.set_ylabel('Normalized power')
    ax3.set_title('Gray-Scott: Power Spectrum Comparison')
    ax3.legend()
    ax3.set_xlim(0, 40)

    # --- Row 2: Schnakenberg ---
    ax4 = fig.add_subplot(gs[1, 0])
    im = ax4.imshow(sn_results['imex']['u'].T, cmap='viridis', origin='lower',
                     extent=[0, sn_results['params']['L']]*2)
    ax4.set_title(f"Schnakenberg: {sn_results['imex']['label']}\n"
                  f"({sn_results['imex']['time']:.0f}s)", fontsize=9)
    plt.colorbar(im, ax=ax4, shrink=0.8)

    ax5 = fig.add_subplot(gs[1, 1])
    im = ax5.imshow(sn_results['etd']['u'].T, cmap='viridis', origin='lower',
                     extent=[0, sn_results['params']['L']]*2)
    ax5.set_title(f"Schnakenberg: {sn_results['etd']['label']}\n"
                  f"({sn_results['etd']['time']:.0f}s)", fontsize=9)
    plt.colorbar(im, ax=ax5, shrink=0.8)

    ax6 = fig.add_subplot(gs[1, 2:])
    _, k_imex, p_imex = dominant_wavelength(sn_results['imex']['u'], sn_results['params']['L'])
    _, k_etd2, p_etd2 = dominant_wavelength(sn_results['etd']['u'], sn_results['params']['L'])
    ax6.semilogy(k_imex, p_imex / max(p_imex.max(), 1e-30), 'b-', label='IMEX-Euler', linewidth=1.5)
    ax6.semilogy(k_etd2, p_etd2 / max(p_etd2.max(), 1e-30), 'r--', label='ETDRK4', linewidth=1.5)
    ax6.axvline(sn_results['theory']['k_crit'], color='green', linestyle=':', linewidth=2,
                label=f"Theory k_crit={sn_results['theory']['k_crit']:.1f}")
    ax6.set_xlabel('Wavenumber k')
    ax6.set_ylabel('Normalized power')
    ax6.set_title('Schnakenberg: Power Spectrum vs Theory')
    ax6.legend()
    ax6.set_xlim(0, 20)

    # --- Row 3: Brusselator ---
    ax7 = fig.add_subplot(gs[2, 0])
    im = ax7.imshow(br_results['rk4']['u'].T, cmap='RdBu_r', origin='lower',
                     extent=[0, br_results['params']['L']]*2)
    ax7.set_title(f"Brusselator: {br_results['rk4']['label']}\n"
                  f"({br_results['rk4']['time']:.0f}s)", fontsize=9)
    plt.colorbar(im, ax=ax7, shrink=0.8)

    ax8 = fig.add_subplot(gs[2, 1])
    im = ax8.imshow(br_results['etd']['u'].T, cmap='RdBu_r', origin='lower',
                     extent=[0, br_results['params']['L']]*2)
    ax8.set_title(f"Brusselator: {br_results['etd']['label']}\n"
                  f"({br_results['etd']['time']:.0f}s)", fontsize=9)
    plt.colorbar(im, ax=ax8, shrink=0.8)

    ax9 = fig.add_subplot(gs[2, 2:])
    _, k_rk42, p_rk42 = dominant_wavelength(br_results['rk4']['u'], br_results['params']['L'])
    _, k_etd3, p_etd3 = dominant_wavelength(br_results['etd']['u'], br_results['params']['L'])
    ax9.semilogy(k_rk42, p_rk42 / max(p_rk42.max(), 1e-30), 'b-', label='RK4+FD', linewidth=1.5)
    ax9.semilogy(k_etd3, p_etd3 / max(p_etd3.max(), 1e-30), 'r--', label='ETDRK4', linewidth=1.5)
    ax9.axvline(br_results['theory']['k_crit'], color='green', linestyle=':', linewidth=2,
                label=f"Theory k_crit={br_results['theory']['k_crit']:.1f}")
    ax9.set_xlabel('Wavenumber k')
    ax9.set_ylabel('Normalized power')
    ax9.set_title('Brusselator: Power Spectrum vs Theory')
    ax9.legend()
    ax9.set_xlim(0, 20)

    # --- Row 4: Turing dispersion relations ---
    ax10 = fig.add_subplot(gs[3, 0:2])
    k_sn = sn_results['theory']['k_vals']
    g_sn = sn_results['theory']['growth']
    ax10.plot(k_sn, g_sn, 'b-', linewidth=2, label='Schnakenberg')
    ax10.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax10.axvline(sn_results['theory']['k_crit'], color='b', linestyle=':', alpha=0.5)
    ax10.fill_between(k_sn, 0, g_sn, where=g_sn > 0, alpha=0.2, color='blue')
    ax10.set_xlabel('Wavenumber k')
    ax10.set_ylabel('Growth rate σ(k)')
    ax10.set_title('Schnakenberg: Turing Dispersion Relation')
    ax10.set_xlim(0, 15)
    ax10.legend()

    ax11 = fig.add_subplot(gs[3, 2:])
    k_br = br_results['theory']['k_vals']
    g_br = br_results['theory']['growth']
    ax11.plot(k_br, g_br, 'r-', linewidth=2, label='Brusselator')
    ax11.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax11.axvline(br_results['theory']['k_crit'], color='r', linestyle=':', alpha=0.5)
    ax11.fill_between(k_br, 0, g_br, where=g_br > 0, alpha=0.2, color='red')
    ax11.set_xlabel('Wavenumber k')
    ax11.set_ylabel('Growth rate σ(k)')
    ax11.set_title('Brusselator: Turing Dispersion Relation')
    ax11.set_xlim(0, 15)
    ax11.legend()

    fig.suptitle('Pattern Formation Verification: Conventional Methods vs ETDRK4\n'
                 '(All conventional methods use pure numpy/scipy — no JAX, no moljax)',
                 fontsize=14, y=0.98)

    outpath = output_dir / "fig_pattern_verification.png"
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {outpath}")
    return outpath


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("Pattern Formation Verification Script")
    print("Using ONLY numpy/scipy (no JAX, no moljax)")
    print()

    t_total = time.time()

    gs = run_gray_scott()
    sn = run_schnakenberg()
    br = run_brusselator()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nGray-Scott:")
    c1 = pattern_contrast(gs['rk4']['v'])
    c2 = pattern_contrast(gs['etd']['v'])
    print(f"  RK4+FD contrast={c1:.4f}, ETDRK4 contrast={c2:.4f}")
    print(f"  Both show spots: {c1 > 0.01 and c2 > 0.01}")

    print("\nSchnakenberg:")
    c1 = pattern_contrast(sn['imex']['u'])
    c2 = pattern_contrast(sn['etd']['u'])
    l1, _, _ = dominant_wavelength(sn['imex']['u'], sn['params']['L'])
    l2, _, _ = dominant_wavelength(sn['etd']['u'], sn['params']['L'])
    lt = sn['theory']['lambda_crit']
    print(f"  IMEX contrast={c1:.4f}, ETDRK4 contrast={c2:.4f}")
    print(f"  Wavelength: IMEX={l1:.3f}, ETDRK4={l2:.3f}, Theory={lt:.3f}")
    print(f"  Agreement: |IMEX-Theory|/Theory = {abs(l1-lt)/lt*100:.1f}%, "
          f"|ETDRK4-Theory|/Theory = {abs(l2-lt)/lt*100:.1f}%")

    print("\nBrusselator:")
    c1 = pattern_contrast(br['rk4']['u'])
    c2 = pattern_contrast(br['etd']['u'])
    l1, _, _ = dominant_wavelength(br['rk4']['u'], br['params']['L'])
    l2, _, _ = dominant_wavelength(br['etd']['u'], br['params']['L'])
    lt = br['theory']['lambda_crit']
    print(f"  RK4+FD contrast={c1:.4f}, ETDRK4 contrast={c2:.4f}")
    print(f"  Wavelength: RK4={l1:.3f}, ETDRK4={l2:.3f}, Theory={lt:.3f}")
    print(f"  Agreement: |RK4-Theory|/Theory = {abs(l1-lt)/lt*100:.1f}%, "
          f"|ETDRK4-Theory|/Theory = {abs(l2-lt)/lt*100:.1f}%")

    figpath = plot_comparison(gs, sn, br)

    print(f"\nTotal time: {time.time() - t_total:.0f}s")
    print(f"\nVerification figure: {figpath}")
