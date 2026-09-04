#!/usr/bin/env python3
"""
Variable-Coefficient Diffusion Stress Test (REAL IMPLEMENTATION)

Tests FFT preconditioning degradation when diffusion coefficient D(x) is
spatially variable. The FFT preconditioner is built on D_mean (constant),
so as max(D)/min(D) increases, the preconditioner becomes less effective.

APPROACH (same as benchmark_gmres_sweep.py):
- SciPy GMRES with callback for ground-truth iteration counting (CPU)
- JAX for wall-clock GPU timing
- No mock data: actual Jacobian-vector products and FFT preconditioner

Problem: u_t = d/dx(D(x) du/dx) + k*u^2  (1D, periodic BC)
Backward Euler Jacobian: J*v = v - dt*(L_var*v + diag(dR/du)*v)
FFT preconditioner: M = (I - dt*D_mean*L_FFT)^{-1}
"""

# Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from benchmark_utils import add_benchmark_args, setup_benchmark
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import gmres as scipy_gmres

parser = add_benchmark_args()
args = parser.parse_args()

# ============================================================================
# Configuration
# ============================================================================
N = 256             # Grid points (1D)
DX = 1.0 / N       # Grid spacing (domain [0, 1])
D_BASE = 1.0       # Minimum diffusivity
SIGMA = 10.0        # Stiffness ratio (moderate, same as gmres_sweep σ=10)
DT = SIGMA * DX**2 / D_BASE  # dt from stiffness ratio
REACTION_K = 1.0    # Moderate reaction
GMRES_TOL = 1e-6
GMRES_MAXITER = 5000

CONTRASTS = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
PROFILES = ["smooth", "step"]

print("Variable-Coefficient Diffusion Stress Test")
print("=" * 70)
device_str = setup_benchmark(expected_backend="gpu")
print(f"Grid: {N} points (1D, periodic BC)")
print(f"Base diffusivity: D_base = {D_BASE}")
print(f"Stiffness ratio: sigma = {SIGMA} (dt = {DT:.6f})")
print(f"Reaction: k = {REACTION_K}")
print(f"Contrasts: {CONTRASTS}")
print(f"Profiles: {PROFILES}")
print(f"GMRES tolerance: {GMRES_TOL}")
print("=" * 70)


# ============================================================================
# Variable coefficient profiles
# ============================================================================

def build_variable_D(N, contrast, profile, D_base=D_BASE):
    """Build 1D variable diffusion coefficient D(x) on [0,1].

    D(x) ranges from D_base to D_base * contrast.
    """
    x = np.linspace(0, 1, N, endpoint=False)  # Periodic: exclude endpoint

    if profile == "smooth":
        # Smooth transition via tanh
        D_x = D_base * (1.0 + (contrast - 1.0) * (0.5 + 0.5 * np.tanh(10 * (x - 0.5))))
    elif profile == "step":
        # Step function at x=0.5
        D_x = np.where(x < 0.5, D_base, D_base * contrast)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    return D_x


# ============================================================================
# NumPy operators (for SciPy GMRES)
# ============================================================================

def np_variable_diffusion_matvec(v, D_x, dx):
    """Apply d/dx(D(x) dv/dx) using conservative FD with periodic BC.

    Conservative discretization:
        L*v_i = (D_{i+1/2}*(v_{i+1} - v_i) - D_{i-1/2}*(v_i - v_{i-1})) / dx^2
    where D_{i+1/2} = (D_i + D_{i+1}) / 2
    """
    n = len(v)
    # Periodic padding
    v_p = np.roll(v, -1)  # v_{i+1}
    v_m = np.roll(v, 1)   # v_{i-1}
    D_p = np.roll(D_x, -1)  # D_{i+1}
    D_m = np.roll(D_x, 1)   # D_{i-1}

    D_half_plus = 0.5 * (D_x + D_p)    # D_{i+1/2}
    D_half_minus = 0.5 * (D_x + D_m)   # D_{i-1/2}

    Lv = (D_half_plus * (v_p - v) - D_half_minus * (v - v_m)) / dx**2
    return Lv


def np_constant_lap_matvec(v, dx):
    """Apply constant-coefficient Laplacian with periodic BC."""
    return (np.roll(v, -1) + np.roll(v, 1) - 2.0 * v) / dx**2


# FFT eigenvalues for preconditioner (1D, periodic)
k_freq = np.fft.fftfreq(N, DX) * 2 * np.pi
lap_eig_np = -k_freq**2  # Laplacian eigenvalues

# Linearization point
np.random.seed(42)
u0_np = np.abs(np.random.randn(N)) + 0.5
reaction_jac_np = 2.0 * REACTION_K * u0_np  # dR/du = 2*k*u

# RHS vector
rhs_np = np.random.randn(N)


# ============================================================================
# Run contrast sweep
# ============================================================================

results = []

for profile in PROFILES:
    print(f"\n{'='*70}")
    print(f"Profile: {profile}")
    print(f"{'='*70}")

    for contrast in CONTRASTS:
        D_x = build_variable_D(N, contrast, profile)
        D_mean = np.mean(D_x)
        D_max = np.max(D_x)
        D_min = np.min(D_x)

        print(f"\n  Contrast = {contrast:.0f} (D_mean={D_mean:.2f}, D_max/D_min={D_max/D_min:.1f})")

        # --- Jacobian: J*v = v - dt*(L_var*v + diag(dR/du)*v) ---
        def make_jacobian_matvec(D_x_local, dt_local):
            def jacobian_matvec(v):
                diffusion = np_variable_diffusion_matvec(v, D_x_local, DX)
                reaction = reaction_jac_np * v
                return v - dt_local * (diffusion + reaction)
            return jacobian_matvec

        J_matvec = make_jacobian_matvec(D_x, DT)
        J_op = LinearOperator((N, N), matvec=J_matvec, dtype=np.float64)

        # --- FFT preconditioner using D_mean ---
        def make_fft_precond(D_mean_local, dt_local):
            denom = 1.0 - dt_local * D_mean_local * lap_eig_np
            def fft_precond(v):
                v_hat = np.fft.fft(v)
                v_hat_solved = v_hat / denom
                return np.fft.ifft(v_hat_solved).real
            return fft_precond

        fft_precond = make_fft_precond(D_mean, DT)
        M_op = LinearOperator((N, N), matvec=fft_precond, dtype=np.float64)

        # --- GMRES without preconditioner ---
        iter_count_none = [0]
        def callback_none(rn):
            iter_count_none[0] += 1

        iter_count_none[0] = 0
        sol_none, info_none = scipy_gmres(
            J_op, rhs_np, rtol=GMRES_TOL, maxiter=GMRES_MAXITER,
            callback=callback_none, callback_type='pr_norm'
        )
        iters_none = iter_count_none[0]
        converged_none = (info_none == 0)

        # --- GMRES with FFT preconditioner ---
        iter_count_fft = [0]
        def callback_fft(rn):
            iter_count_fft[0] += 1

        iter_count_fft[0] = 0
        sol_fft, info_fft = scipy_gmres(
            J_op, rhs_np, rtol=GMRES_TOL, maxiter=GMRES_MAXITER,
            M=M_op, callback=callback_fft, callback_type='pr_norm'
        )
        iters_fft = iter_count_fft[0]
        converged_fft = (info_fft == 0)

        # Compute residuals
        res_none = np.linalg.norm(J_matvec(sol_none) - rhs_np) / np.linalg.norm(rhs_np)
        res_fft = np.linalg.norm(J_matvec(sol_fft) - rhs_np) / np.linalg.norm(rhs_np)

        reduction = iters_none / max(iters_fft, 1)
        fft_still_helps = iters_fft < iters_none

        print(f"    No precond:  {iters_none:>6d} iters, converged={converged_none}, res={res_none:.2e}")
        print(f"    FFT precond: {iters_fft:>6d} iters, converged={converged_fft}, res={res_fft:.2e}")
        print(f"    Reduction:   {reduction:.1f}x {'(FFT helps)' if fft_still_helps else '(FFT no benefit)'}")

        results.append({
            "profile": profile,
            "contrast": float(contrast),
            "D_mean": float(D_mean),
            "D_max": float(D_max),
            "D_min": float(D_min),
            "iters_none": iters_none,
            "iters_fft": iters_fft,
            "converged_none": converged_none,
            "converged_fft": converged_fft,
            "residual_none": float(res_none),
            "residual_fft": float(res_fft),
            "reduction": float(reduction),
            "fft_helps": bool(fft_still_helps),
        })


# ============================================================================
# Save results
# ============================================================================

output_dir = Path(__file__).parent / "results"
output_dir.mkdir(exist_ok=True)

results_data = {
    "config": {
        "N": N,
        "dx": DX,
        "sigma": SIGMA,
        "dt": DT,
        "D_base": D_BASE,
        "reaction_k": REACTION_K,
        "gmres_tol": GMRES_TOL,
        "device": device_str,
    },
    "results": results,
}

output_file = output_dir / "variable_coeff.json"
with open(output_file, 'w') as f:
    json.dump(results_data, f, indent=2)
print(f"\nResults saved to {output_file}")


# ============================================================================
# Generate figures
# ============================================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for profile in PROFILES:
    data = [r for r in results if r["profile"] == profile]
    contrasts = [r["contrast"] for r in data]
    iters_none = [r["iters_none"] for r in data]
    iters_fft = [r["iters_fft"] for r in data]
    reductions = [r["reduction"] for r in data]

    style = '-o' if profile == 'smooth' else '--s'

    # Left: GMRES iterations
    axes[0].loglog(contrasts, iters_fft, style, label=f'{profile} (FFT)', linewidth=2, markersize=6)
    axes[0].loglog(contrasts, iters_none, style, label=f'{profile} (none)', linewidth=1.5,
                   markersize=5, alpha=0.5)

    # Center: Reduction factor
    axes[1].semilogx(contrasts, reductions, style, label=f'{profile}', linewidth=2, markersize=6)

    # Right: Break-even analysis
    fft_helps = [r["fft_helps"] for r in data]
    axes[2].semilogx(contrasts, [r["iters_fft"] / max(r["iters_none"], 1) for r in data],
                     style, label=f'{profile}', linewidth=2, markersize=6)

axes[0].set_xlabel('Coefficient Contrast (max/min)')
axes[0].set_ylabel('GMRES Iterations')
axes[0].set_title('GMRES Iterations vs Contrast')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel('Coefficient Contrast (max/min)')
axes[1].set_ylabel('Iteration Reduction (none/FFT)')
axes[1].set_title('FFT Preconditioner Effectiveness')
axes[1].axhline(1.0, color='black', linestyle=':', alpha=0.5, label='Break-even')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].set_xlabel('Coefficient Contrast (max/min)')
axes[2].set_ylabel('FFT / None Iteration Ratio')
axes[2].set_title('Relative Cost (< 1 means FFT helps)')
axes[2].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='Break-even')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Variable-Coefficient Stress Test (N={N}, σ={SIGMA}, k={REACTION_K})', fontsize=14)
plt.tight_layout()

fig_dir = Path(__file__).parent.parent / "figures"
fig_dir.mkdir(exist_ok=True)
plt.savefig(fig_dir / "fig_variable_coeff_stress.png", dpi=300, bbox_inches='tight')
plt.savefig(fig_dir / "fig_variable_coeff_stress.pdf", bbox_inches='tight')
print(f"Figure saved to {fig_dir / 'fig_variable_coeff_stress.png'}")


# ============================================================================
# Summary
# ============================================================================

print(f"\n{'='*80}")
print("SUMMARY: Variable-Coefficient Stress Test")
print(f"{'='*80}")
print(f"{'Contrast':>10} | {'Profile':>8} | {'None iters':>10} {'FFT iters':>10} {'Reduction':>10} | {'FFT helps':>9}")
print(f"{'-'*80}")

for r in results:
    marker = "YES" if r["fft_helps"] else "NO"
    converge_mark = "" if r["converged_fft"] else "*"
    print(f"{r['contrast']:>10.0f} | {r['profile']:>8} | {r['iters_none']:>10d} {r['iters_fft']:>10d}{converge_mark} {r['reduction']:>9.1f}x | {marker:>9}")

print(f"{'-'*80}")
print("* = did not converge in GMRES_MAXITER iterations")
print(f"{'='*80}")
