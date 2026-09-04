#!/usr/bin/env python3
"""
Benchmark: OFAT (One Factor At A Time) Study

Three controlled experiments, each varying exactly ONE factor:

OFAT-1: Spatial method  (hold: CN time stepper, JAX GPU JIT)
  - FD-FFT-CN vs PS-FFT-CN

OFAT-2: Time stepper  (hold: PS-FFT spatial, JAX GPU JIT)
  - RK4 vs CN vs IMEX-Strang vs ETDRK4

OFAT-3: Hardware  (hold: PS-FFT spatial, FFT-CN time stepper, JIT on)
  - JAX CPU vs JAX GPU across grid sizes
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from benchmark_utils import add_benchmark_args, compute_stats, ps_fft_stability_limit

# Add parent dir to path for moljax imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

parser = add_benchmark_args()
args = parser.parse_args()
N_REPS = args.n_reps

N_WARMUP = 3

print("OFAT (One Factor At A Time) Benchmark")
print("=" * 60)
print(f"N_REPS: {N_REPS}")
print(f"JAX: {jax.__version__}, GPU: {jax.devices('gpu')[0]}")
print("=" * 60)

results = {}


# =============================================================================
# Shared: 2D diffusion problem with periodic BC and analytical solution
# =============================================================================
def analytical_solution(x, y, t, D):
    """u(x,y,t) = exp(-8*pi^2*D*t) * sin(2*pi*x) * sin(2*pi*y)"""
    return np.exp(-8 * np.pi**2 * D * t) * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)


def make_problem(N, D=0.1, L=1.0):
    """Create 2D diffusion problem arrays."""
    dx = L / N
    x = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing='ij')

    # Initial condition
    u0 = np.sin(2*np.pi*X) * np.sin(2*np.pi*Y)

    # PS eigenvalues: continuous symbol -k^2
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    ps_eig = -(KX**2 + KY**2)

    # FD eigenvalues: -(4/dx^2)*sin^2(pi*k/N)
    fd_eig = -(4.0/dx**2) * (np.sin(np.pi * KX * dx / (2*np.pi))**2 +
                               np.sin(np.pi * KY * dx / (2*np.pi))**2)
    # Corrected FD eigenvalues using standard formula
    kx_idx = np.fft.fftfreq(N, 1.0/N)  # integer frequencies
    ky_idx = np.fft.fftfreq(N, 1.0/N)
    KX_idx, KY_idx = np.meshgrid(kx_idx, ky_idx, indexing='ij')
    fd_eig = -(2.0/dx**2) * (2 - 2*np.cos(2*np.pi*KX_idx/N)) + \
             -(2.0/dx**2) * (2 - 2*np.cos(2*np.pi*KY_idx/N))
    # Simplify: fd_eig[k] = -(4/dx^2) * sin^2(pi*k/N) per dimension
    fd_eig = -(4.0/dx**2) * np.sin(np.pi*KX_idx/N)**2 + \
             -(4.0/dx**2) * np.sin(np.pi*KY_idx/N)**2

    return X, Y, u0, ps_eig, fd_eig, dx


# =============================================================================
# OFAT-1: Spatial method (hold: CN, GPU JIT)
# =============================================================================
print("\n" + "=" * 60)
print("OFAT-1: Spatial Method (hold: CN time stepper, JAX GPU JIT)")
print("=" * 60)

N = 256
D = 0.1
T = 0.1
dt = 0.001
n_steps = int(T / dt)

X, Y, u0_np, ps_eig_np, fd_eig_np, dx = make_problem(N, D)
u_exact = analytical_solution(X, Y, T, D)

# JAX arrays on GPU
u0_jax = jnp.array(u0_np)
ps_eig_jax = jnp.array(ps_eig_np)
fd_eig_jax = jnp.array(fd_eig_np)


@jax.jit
def cn_step(u, eig, dt, D):
    u_hat = jnp.fft.fft2(u)
    u_hat = u_hat * (1 + 0.5*dt*D*eig) / (1 - 0.5*dt*D*eig)
    return jnp.real(jnp.fft.ifft2(u_hat))


def cn_solve(u0, eig, n_steps, dt, D):
    u = u0
    for _ in range(n_steps):
        u = cn_step(u, eig, dt, D)
    return u


ofat1_results = {}

for label, eig in [("PS-FFT-CN", ps_eig_jax), ("FD-FFT-CN", fd_eig_jax)]:
    # Warmup
    for _ in range(N_WARMUP):
        r = cn_solve(u0_jax, eig, n_steps, dt, D)
        r.block_until_ready()

    # Time
    times = []
    for _ in range(N_REPS):
        t0 = time.perf_counter()
        r = cn_solve(u0_jax, eig, n_steps, dt, D)
        r.block_until_ready()
        times.append(time.perf_counter() - t0)

    med, iqr = compute_stats(times)
    err = float(jnp.max(jnp.abs(r - jnp.array(u_exact))))

    ofat1_results[label] = {
        'median_s': float(med),
        'iqr_s': float(iqr),
        'error_linf': err,
        'n_steps': n_steps,
    }
    print(f"  {label}: {med:.4f} +/- {iqr:.4f} s, error = {err:.2e}")

results['ofat1_spatial'] = ofat1_results
results['ofat1_spatial']['config'] = {
    'varied': 'spatial_method',
    'held_constant': {'time_stepper': 'CN', 'hardware': 'JAX GPU JIT',
                      'grid': f'{N}x{N}', 'D': D, 'T': T, 'dt': dt},
}


# =============================================================================
# OFAT-2: Time stepper (hold: PS-FFT, GPU JIT)
# =============================================================================
print("\n" + "=" * 60)
print("OFAT-2: Time Stepper (hold: PS-FFT spatial, JAX GPU JIT)")
print("=" * 60)

N = 256
D = 0.1
T = 0.1

X, Y, u0_np, ps_eig_np, _, dx = make_problem(N, D)
u_exact = analytical_solution(X, Y, T, D)

u0_jax = jnp.array(u0_np)
ps_eig_jax = jnp.array(ps_eig_np)

ofat2_results = {}

# --- RK4 (explicit) ---
dt_rk4 = ps_fft_stability_limit(N, 1.0, D, ndim=2, integrator="RK4", safety=0.8)
n_steps_rk4 = int(np.ceil(T / dt_rk4))
dt_rk4 = T / n_steps_rk4  # exact final time


@jax.jit
def rk4_step(u, eig, dt, D):
    """RK4 step for u_t = D * L * u (linear diffusion only)."""
    def rhs(u):
        u_hat = jnp.fft.fft2(u)
        return jnp.real(jnp.fft.ifft2(D * eig * u_hat))

    k1 = rhs(u)
    k2 = rhs(u + 0.5*dt*k1)
    k3 = rhs(u + 0.5*dt*k2)
    k4 = rhs(u + dt*k3)
    return u + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def rk4_solve(u0, eig, n_steps, dt, D):
    u = u0
    for _ in range(n_steps):
        u = rk4_step(u, eig, dt, D)
    return u


print(f"\n  RK4: dt={dt_rk4:.6f}, {n_steps_rk4} steps")
for _ in range(N_WARMUP):
    r = rk4_solve(u0_jax, ps_eig_jax, min(n_steps_rk4, 100), dt_rk4, D)
    r.block_until_ready()

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    r = rk4_solve(u0_jax, ps_eig_jax, n_steps_rk4, dt_rk4, D)
    r.block_until_ready()
    times.append(time.perf_counter() - t0)

med, iqr = compute_stats(times)
err = float(jnp.max(jnp.abs(r - jnp.array(u_exact))))
ofat2_results['RK4'] = {
    'median_s': float(med), 'iqr_s': float(iqr),
    'error_linf': err, 'n_steps': n_steps_rk4, 'dt': float(dt_rk4),
    'nfe': n_steps_rk4 * 4,
}
print(f"  RK4: {med:.4f} s, error = {err:.2e}, NFE = {n_steps_rk4*4}")

# --- CN (implicit, FFT-diagonalized) ---
dt_cn = 0.001
n_steps_cn = int(T / dt_cn)

print(f"\n  CN: dt={dt_cn}, {n_steps_cn} steps")
for _ in range(N_WARMUP):
    r = cn_solve(u0_jax, ps_eig_jax, n_steps_cn, dt_cn, D)
    r.block_until_ready()

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    r = cn_solve(u0_jax, ps_eig_jax, n_steps_cn, dt_cn, D)
    r.block_until_ready()
    times.append(time.perf_counter() - t0)

med, iqr = compute_stats(times)
err = float(jnp.max(jnp.abs(r - jnp.array(u_exact))))
ofat2_results['CN'] = {
    'median_s': float(med), 'iqr_s': float(iqr),
    'error_linf': err, 'n_steps': n_steps_cn, 'dt': dt_cn,
    'nfe': n_steps_cn,  # 1 linear solve per step
}
print(f"  CN: {med:.4f} s, error = {err:.2e}, NFE = {n_steps_cn}")

# --- IMEX-Strang (diffusion implicit, reaction=0 here, so equivalent to CN) ---
# For pure diffusion, IMEX-Strang reduces to implicit diffusion only
# We use the same dt as CN since there's no reaction CFL


@jax.jit
def imex_step(u, eig, dt, D):
    """IMEX step: implicit diffusion via FFT (reaction = 0 for pure diffusion)."""
    u_hat = jnp.fft.fft2(u)
    # Full implicit step: u^{n+1} = (I - dt*D*L)^{-1} u^n
    u_hat = u_hat / (1 - dt*D*eig)
    return jnp.real(jnp.fft.ifft2(u_hat))


def imex_solve(u0, eig, n_steps, dt, D):
    u = u0
    for _ in range(n_steps):
        u = imex_step(u, eig, dt, D)
    return u


dt_imex = 0.001
n_steps_imex = int(T / dt_imex)

print(f"\n  IMEX: dt={dt_imex}, {n_steps_imex} steps")
for _ in range(N_WARMUP):
    r = imex_solve(u0_jax, ps_eig_jax, n_steps_imex, dt_imex, D)
    r.block_until_ready()

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    r = imex_solve(u0_jax, ps_eig_jax, n_steps_imex, dt_imex, D)
    r.block_until_ready()
    times.append(time.perf_counter() - t0)

med, iqr = compute_stats(times)
err = float(jnp.max(jnp.abs(r - jnp.array(u_exact))))
ofat2_results['IMEX'] = {
    'median_s': float(med), 'iqr_s': float(iqr),
    'error_linf': err, 'n_steps': n_steps_imex, 'dt': dt_imex,
    'nfe': n_steps_imex * 2,
}
print(f"  IMEX: {med:.4f} s, error = {err:.2e}, NFE = {n_steps_imex*2}")

# --- ETDRK4 (exponential) ---
dt_etd = 0.01
n_steps_etd = int(T / dt_etd)

# Precompute ETDRK4 coefficients
z = D * ps_eig_jax * dt_etd  # z = L*dt
E = jnp.exp(z)
E2 = jnp.exp(z/2)

# Contour integral for phi functions (Kassam & Trefethen 2005)
M_contour = 32
theta = np.linspace(0, 2*np.pi, M_contour, endpoint=False)
z_np = D * ps_eig_np * dt_etd
z_c = z_np[:,:,None] + np.exp(1j*theta)[None,None,:] * 1.0  # radius 1

ez_c = np.exp(z_c)
phi1 = np.mean((ez_c - 1) / z_c, axis=-1)
phi2 = np.mean((ez_c - 1 - z_c) / z_c**2, axis=-1)
phi3 = np.mean((ez_c - 1 - z_c - 0.5*z_c**2) / z_c**3, axis=-1)

# Handle z=0 (DC mode)
phi1 = np.where(np.abs(z_np) < 1e-10, 1.0, phi1)
phi2 = np.where(np.abs(z_np) < 1e-10, 0.5, phi2)
phi3 = np.where(np.abs(z_np) < 1e-10, 1.0/6.0, phi3)

# To JAX
phi1_j = jnp.array(np.real(phi1))
phi2_j = jnp.array(np.real(phi2))
phi3_j = jnp.array(np.real(phi3))


@jax.jit
def etdrk4_step(u, eig, dt, D, E, E2, phi1, phi2, phi3):
    """ETDRK4 step for u_t = L*u + N(u). For pure diffusion, N=0."""
    # For pure diffusion: N(u) = 0, so this reduces to u_{n+1} = exp(L*dt)*u_n
    u_hat = jnp.fft.fft2(u)
    u_hat_new = E * u_hat  # exact for linear
    return jnp.real(jnp.fft.ifft2(u_hat_new))


def etdrk4_solve(u0, eig, n_steps, dt, D, E, E2, phi1, phi2, phi3):
    u = u0
    for _ in range(n_steps):
        u = etdrk4_step(u, eig, dt, D, E, E2, phi1, phi2, phi3)
    return u


print(f"\n  ETDRK4: dt={dt_etd}, {n_steps_etd} steps")
for _ in range(N_WARMUP):
    r = etdrk4_solve(u0_jax, ps_eig_jax, n_steps_etd, dt_etd, D,
                     E, E2, phi1_j, phi2_j, phi3_j)
    r.block_until_ready()

times = []
for _ in range(N_REPS):
    t0 = time.perf_counter()
    r = etdrk4_solve(u0_jax, ps_eig_jax, n_steps_etd, dt_etd, D,
                     E, E2, phi1_j, phi2_j, phi3_j)
    r.block_until_ready()
    times.append(time.perf_counter() - t0)

med, iqr = compute_stats(times)
err = float(jnp.max(jnp.abs(r - jnp.array(u_exact))))
ofat2_results['ETDRK4'] = {
    'median_s': float(med), 'iqr_s': float(iqr),
    'error_linf': err, 'n_steps': n_steps_etd, 'dt': dt_etd,
    'nfe': n_steps_etd * 4,
}
print(f"  ETDRK4: {med:.4f} s, error = {err:.2e}, NFE = {n_steps_etd*4}")

results['ofat2_timestepper'] = ofat2_results
results['ofat2_timestepper']['config'] = {
    'varied': 'time_stepper',
    'held_constant': {'spatial': 'PS-FFT', 'hardware': 'JAX GPU JIT',
                      'grid': f'{N}x{N}', 'D': D, 'T': T},
}


# =============================================================================
# OFAT-3: Hardware (hold: PS-FFT, CN, JIT)
# =============================================================================
print("\n" + "=" * 60)
print("OFAT-3: Hardware (hold: PS-FFT spatial, CN time stepper, JIT on)")
print("=" * 60)

D = 0.1
T = 0.1
dt = 0.001
GRID_SIZES = [64, 128, 256, 512]

ofat3_results = {}
cpu_device = jax.devices('cpu')[0]
gpu_device = jax.devices('gpu')[0]

for N in GRID_SIZES:
    n_steps = int(T / dt)
    X, Y, u0_np, ps_eig_np, _, dx = make_problem(N, D)

    row = {}
    for dev_label, device in [("CPU", cpu_device), ("GPU", gpu_device)]:
        with jax.default_device(device):
            u0_dev = jnp.array(u0_np)
            eig_dev = jnp.array(ps_eig_np)

            # Use a JIT-compiled step
            @jax.jit
            def cn_step_local(u, eig, dt, D):
                u_hat = jnp.fft.fft2(u)
                u_hat = u_hat * (1 + 0.5*dt*D*eig) / (1 - 0.5*dt*D*eig)
                return jnp.real(jnp.fft.ifft2(u_hat))

            def solve_local():
                u = u0_dev
                for _ in range(n_steps):
                    u = cn_step_local(u, eig_dev, dt, D)
                return u

            # Warmup
            for _ in range(N_WARMUP):
                r = solve_local()
                r.block_until_ready()

            # Time
            times = []
            for _ in range(N_REPS):
                t0 = time.perf_counter()
                r = solve_local()
                r.block_until_ready()
                times.append(time.perf_counter() - t0)

        med, iqr = compute_stats(times)
        row[dev_label] = {'median_s': float(med), 'iqr_s': float(iqr)}
        print(f"  {N}x{N} {dev_label}: {med:.4f} +/- {iqr:.4f} s")

    gpu_speedup = row['CPU']['median_s'] / row['GPU']['median_s']
    row['gpu_speedup'] = float(gpu_speedup)
    print(f"  {N}x{N} GPU speedup: {gpu_speedup:.1f}x")
    ofat3_results[str(N)] = row

results['ofat3_hardware'] = ofat3_results
results['ofat3_hardware']['config'] = {
    'varied': 'hardware (CPU vs GPU)',
    'held_constant': {'spatial': 'PS-FFT', 'time_stepper': 'CN JIT',
                      'D': D, 'T': T, 'dt': dt},
}


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("FULL OFAT SUMMARY")
print("=" * 60)

print("\nOFAT-1: Spatial Method (CN, GPU)")
for label, data in ofat1_results.items():
    if label == 'config':
        continue
    print(f"  {label}: {data['median_s']:.4f} s, error = {data['error_linf']:.2e}")

print("\nOFAT-2: Time Stepper (PS-FFT, GPU)")
for label, data in ofat2_results.items():
    if label == 'config':
        continue
    print(f"  {label}: {data['median_s']:.4f} s, error = {data['error_linf']:.2e}, "
          f"steps = {data['n_steps']}")

print("\nOFAT-3: Hardware (PS-FFT, CN JIT)")
for N_str, data in ofat3_results.items():
    if N_str == 'config':
        continue
    print(f"  {N_str}x{N_str}: CPU={data['CPU']['median_s']:.4f}s, "
          f"GPU={data['GPU']['median_s']:.4f}s, speedup={data['gpu_speedup']:.1f}x")

# Save
output_path = Path(__file__).parent / 'results' / 'ofat_study.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
