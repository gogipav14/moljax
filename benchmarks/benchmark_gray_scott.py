#!/usr/bin/env python3
"""
Benchmark: Gray-Scott Reaction-Diffusion (Table 7)

Gray-Scott system:
    du/dt = Du * lap(u) - u*v^2 + F*(1-u)
    dv/dt = Dv * lap(v) + u*v^2 - (F+k)*v

Parameters: Du=2e-5, Dv=1e-5, F=0.035, k=0.065 (spot pattern)
Grid: 256x256, t ∈ [0, 10000]

Methods compared:
- RK4 (explicit)
- Crank-Nicolson (moljax Newton-Krylov with the FFT diffusion preconditioner)
- IMEX-Strang (moljax; half diffusion, Heun reaction, half diffusion)
- ETDRK4 (moljax)

All four methods are the moljax library solvers; see rd_library_benchmark.py.
The Laplacian is the second-order central difference operator of the library.

IMPORTANT: This benchmark runs to ACTUAL t=10000. No scaling or extrapolation.
All reported times are actual wallclock times to complete the full simulation.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)  # Enable float64 precision
import rd_library_benchmark as lb

from moljax.core.model import create_gray_scott_periodic_fft

# Configuration
N = 256
Du = 2e-5
Dv = 1e-5
F = 0.035
k = 0.065
dx = 1.0 / N

# ACTUAL simulation time - NO SCALING
T_FINAL = 10000.0  # Full simulation time

N_REPS = 10  # Paper uses 10 runs, report median

print("Gray-Scott Benchmark (ACTUAL t=10000, NO SCALING)")
print(f"Grid: {N}x{N}, Du={Du}, Dv={Dv}, F={F}, k={k}")
print(f"Simulation time: t=0 to t={T_FINAL} (actual, no extrapolation)")
print(f"JAX devices: {jax.devices()}")
print(f"Float64 enabled: {jax.config.jax_enable_x64}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Initial condition: small perturbation from steady state
np.random.seed(42)
u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1

# Add localized perturbation
cx, cy = N//2, N//2
r = 10
u0[cx-r:cx+r, cy-r:cy+r] = 0.5
v0[cx-r:cx+r, cy-r:cy+r] = 0.25

def reaction(u, v):
    uvv = u * v * v
    return -uvv + F * (1 - u), uvv - (F + k) * v


dt_cn = 0.5
dt_imex = 0.5
dt_etd = 1.0

lb.run_library_benchmark(
    system="gray_scott",
    make_model=lambda grid: create_gray_scott_periodic_fft(grid, Du=Du, Dv=Dv, F=F, k=k),
    reaction=reaction, u0=u0, v0=v0, domain_length=1.0,
    t_final=T_FINAL, rk4_reaction_dt=float("inf"), dt_cn=dt_cn, dt_imex=dt_imex, dt_etd=dt_etd,
    n_reps=N_REPS, rk4_short_steps=None,
    config={'Du': Du, 'Dv': Dv, 'F': F, 'k': k},
)
