#!/usr/bin/env python3
"""
Benchmark: Brusselator Reaction-Diffusion (Oscillatory Instability)

System:
    du/dt = Du * lap(u) + a - (b+1)*u + u^2*v
    dv/dt = Dv * lap(v) + b*u - u^2*v

Parameters: a=1, b=3.4 (above Hopf, oscillatory), Du=0.01, Dv=0.02
Grid: 256x256, domain [0, 1]^2, t in [0, 50]

Methods compared:
- RK4 (explicit)
- Crank-Nicolson (moljax Newton-Krylov with the FFT diffusion preconditioner)
- IMEX-Strang (moljax; half diffusion, Heun reaction, half diffusion)
- ETDRK4 (moljax)

All four methods are the moljax library solvers; see rd_library_benchmark.py.
The Laplacian is the second-order central difference operator of the library.

This problem exhibits OSCILLATORY stiffness: at b > 1+a^2 = 2 (Hopf threshold),
the homogeneous steady state is unstable through oscillation. The combination of
oscillatory instability + diffusion-driven pattern formation creates doubly stiff
dynamics — both eigenvalue magnitudes AND imaginary parts are large.
"""

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
import rd_library_benchmark as lb

from moljax.core.model import create_brusselator_periodic_fft

# Configuration
N = 256
Du = 0.01
Dv = 0.02
a = 1.0
b = 3.4  # Above Hopf threshold (b > 1 + a^2 = 2)
L = 1.0
dx = L / N

T_FINAL = 50.0
N_REPS = 5  # Reduced for faster execution; can increase for publication

print("Brusselator Benchmark")
print(f"Grid: {N}x{N}, Du={Du}, Dv={Dv}, a={a}, b={b}")
print(f"Hopf threshold: b > 1 + a^2 = {1 + a**2} (b={b}, ABOVE)")
print(f"Simulation time: t=0 to t={T_FINAL}")
print(f"JAX devices: {jax.devices()}")
print(f"Float64 enabled: {jax.config.jax_enable_x64}")
print(f"N_REPS: {N_REPS}")
print("=" * 60)

# Initial condition: perturbation from homogeneous steady state
# Steady state: u_s = a, v_s = b/a
u_s = a
v_s = b / a
np.random.seed(42)
u0 = u_s + 0.01 * np.random.randn(N, N)
v0 = v_s + 0.01 * np.random.randn(N, N)

def reaction(u, v):
    u2v = u * u * v
    return a - (b + 1.0) * u + u2v, b * u - u2v


dt_cn = 0.05
dt_imex = 0.05
dt_etd = 0.05

lb.run_library_benchmark(
    system="brusselator",
    make_model=lambda grid: create_brusselator_periodic_fft(grid, Du=Du, Dv=Dv, a=a, b=b),
    reaction=reaction, u0=u0, v0=v0, domain_length=L,
    t_final=T_FINAL, rk4_reaction_dt=0.5 / (b + 1.0), dt_cn=dt_cn, dt_imex=dt_imex, dt_etd=dt_etd,
    n_reps=N_REPS, rk4_short_steps=5000,
    config={'Du': Du, 'Dv': Dv, 'a': a, 'b': b},
)
