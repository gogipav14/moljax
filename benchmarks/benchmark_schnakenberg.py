#!/usr/bin/env python3
"""
Benchmark: Schnakenberg Reaction-Diffusion (Turing Morphogenesis)

System:
    du/dt = Du * lap(u) + gamma * (a - u + u^2*v)
    dv/dt = Dv * lap(v) + gamma * (b - u^2*v)

Parameters: a=0.1, b=0.9, gamma=1000, Du=1, Dv=40 (stripe regime)
Grid: 256x256, domain [0, 1]^2, t in [0, 100]

Methods compared:
- RK4 (explicit)
- Crank-Nicolson (moljax Newton-Krylov with the FFT diffusion preconditioner)
- IMEX-Strang (moljax; half diffusion, Heun reaction, half diffusion)
- ETDRK4 (moljax)

All four methods are the moljax library solvers; see rd_library_benchmark.py.
The Laplacian is the second-order central difference operator of the library.

This problem is STIFFER than Gray-Scott: gamma=1000 drives reaction eigenvalues
~1000x larger, requiring much smaller explicit time steps.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import rd_library_benchmark as lb

from moljax.core.model import create_schnakenberg_periodic_fft

# Configuration
N = 256
Du = 1.0
Dv = 40.0
a = 0.1
b = 0.9
gamma = 1000.0
L = 1.0
dx = L / N

T_FINAL = 100.0
N_REPS = 5  # Reduced for faster execution; can increase for publication

print("Schnakenberg Benchmark")
print(f"Grid: {N}x{N}, Du={Du}, Dv={Dv}, a={a}, b={b}, gamma={gamma}")
print(f"Simulation time: t=0 to t={T_FINAL}")
print(f"JAX devices: {jax.devices()}")
print(f"Float64 enabled: {jax.config.jax_enable_x64}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Initial condition: small random perturbation from homogeneous steady state
# Steady state: u_s = a + b, v_s = b / (a + b)^2
u_s = a + b
v_s = b / (u_s**2)
np.random.seed(42)
u0 = u_s + 0.01 * np.random.randn(N, N)
v0 = v_s + 0.01 * np.random.randn(N, N)

def reaction(u, v):
    u2v = u * u * v
    return gamma * (a - u + u2v), gamma * (b - u2v)


dt_cn = 0.01
dt_imex = 0.001
dt_etd = 0.001

lb.run_library_benchmark(
    system="schnakenberg",
    make_model=lambda grid: create_schnakenberg_periodic_fft(grid, Du=Du, Dv=Dv, a=a, b=b, gamma=gamma),
    reaction=reaction, u0=u0, v0=v0, domain_length=L,
    t_final=T_FINAL, rk4_reaction_dt=1.0 / gamma, dt_cn=dt_cn, dt_imex=dt_imex, dt_etd=dt_etd,
    n_reps=N_REPS, rk4_short_steps=1000,
    config={'Du': Du, 'Dv': Dv, 'a': a, 'b': b, 'gamma': gamma},
)
