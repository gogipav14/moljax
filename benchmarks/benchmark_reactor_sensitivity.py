#!/usr/bin/env python3
"""
Sensitivity of the tubular reactor conversion with automatic differentiation.

Model and discretization: the same as benchmark_tubular_reactor.py
(Danckwerts conditions, N = 128, second-order differences, RK4 with the CFL
step, start from an empty reactor, t_final = 15).

The script computes X_out = 1 - c(z=1, t_final) and its derivatives with
respect to Da and Pe:

  reverse mode:  jax.value_and_grad of the complete time loop
  forward mode:  jax.jacfwd of the complete time loop

Each derivative has two checks:

  1. a central difference of the same numerical model;
  2. a central difference of the analytical steady state.

The step dt comes from the nominal Pe and is a constant of the
differentiated function, so the derivative is that of the discrete model at
a fixed step.

Timing: one warm-up call (compilation), then N_REPS timed calls; median.

Environment variables:
  RD_OUT_DIR   Output folder for JSON and CSV (default: the results folder).
  RD_N_REPS    Number of timed calls (default 5).
"""

import jax

jax.config.update("jax_enable_x64", True)

import csv
import json
import os
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

N = 128
T_FINAL = 15.0
N_REPS = int(os.environ.get("RD_N_REPS", 5))
CASES = [(10.0, 0.5), (10.0, 1.0), (10.0, 2.0), (50.0, 1.0), (100.0, 1.0)]
OUT_DIR = Path(os.environ.get("RD_OUT_DIR", Path(__file__).parent / "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fd_matrices(n, dz):
    """Difference matrices of benchmark_tubular_reactor.py."""
    d2 = np.zeros((n, n))
    d1 = np.zeros((n, n))
    for i in range(1, n - 1):
        d2[i, i - 1], d2[i, i], d2[i, i + 1] = 1.0, -2.0, 1.0
        d1[i, i - 1], d1[i, i + 1] = -0.5, 0.5
    d1[0, 0], d1[0, 1] = -1.0, 1.0
    d1[-1, -2], d1[-1, -1] = -1.0, 1.0
    return jnp.array(d2 / dz**2), jnp.array(d1 / dz)


DZ = 1.0 / (N - 1)
D2, D1 = fd_matrices(N, DZ)


def rhs(c, pe, da):
    """Right-hand side with Danckwerts conditions (relaxation form, c_in = 1)."""
    dcdt = (1.0 / pe) * (D2 @ c) - (D1 @ c) - da * c
    inlet = (1.0 / pe) / DZ
    c0_target = (1.0 + inlet * c[1]) / (1.0 + inlet)
    dcdt = dcdt.at[0].set(100.0 * (c0_target - c[0]))
    return dcdt.at[-1].set(100.0 * (c[-2] - c[-1]))


def make_conversion(dt, n_steps):
    """X_out(pe, da) for a fixed step and a fixed step count."""

    def conversion(pe, da):
        def step(_, c):
            k1 = rhs(c, pe, da)
            k2 = rhs(c + 0.5 * dt * k1, pe, da)
            k3 = rhs(c + 0.5 * dt * k2, pe, da)
            k4 = rhs(c + dt * k3, pe, da)
            return c + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        c = jax.lax.fori_loop(0, n_steps, step, jnp.zeros(N))
        return 1.0 - c[-1]

    return conversion


def analytical_conversion(pe, da):
    """X_out of the analytical steady state (Danckwerts conditions)."""
    disc = pe**2 + 4 * pe * da
    r1 = (pe + np.sqrt(disc)) / 2
    r2 = (pe - np.sqrt(disc)) / 2
    # Unknowns: A*exp(r1) and B. This form does not overflow for large Pe.
    m = np.array([[(r1 / pe - 1) * np.exp(-r1), r2 / pe - 1], [r1, r2 * np.exp(r2)]])
    a, b = np.linalg.solve(m, np.array([-1.0, 0.0]))
    return 1.0 - (a + b * np.exp(r2))


def central(f, pe, da, h=1e-5):
    """Central differences of f(pe, da) for Da and for Pe."""
    return (float((f(pe, da + h) - f(pe, da - h)) / (2 * h)),
            float((f(pe + h, da) - f(pe - h, da)) / (2 * h)))


def timed(fn, *args):
    """One warm-up call, then N_REPS timed calls. Return (median seconds, result)."""
    out = fn(*args)
    jax.block_until_ready(out)
    times = []
    for _ in range(N_REPS):
        start = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    return float(np.median(times)), out


print("Reactor sensitivity with automatic differentiation")
print(f"JAX {jax.__version__}, devices {jax.devices()}, N = {N}, t_final = {T_FINAL}, "
      f"N_REPS = {N_REPS}")

records = []
for pe, da in CASES:
    dt = min(0.5 * DZ, 0.25 * pe * DZ**2, 0.01)
    n_steps = int(T_FINAL / dt)
    conversion = make_conversion(dt, n_steps)
    forward = jax.jit(conversion)
    reverse = jax.jit(jax.value_and_grad(conversion, argnums=(0, 1)))
    forward_mode = jax.jit(jax.jacfwd(conversion, argnums=(0, 1)))

    t_forward, x_out = timed(forward, pe, da)
    t_reverse, (_, (dx_dpe, dx_dda)) = timed(reverse, pe, da)
    t_fwd_mode, (fx_dpe, fx_dda) = timed(forward_mode, pe, da)
    fd_dda, fd_dpe = central(forward, pe, da)
    an_dda, an_dpe = central(analytical_conversion, pe, da)

    rec = {
        "Pe": pe, "Da": da, "dt": dt, "steps": n_steps,
        "X_out": float(x_out), "X_out_analytical": float(analytical_conversion(pe, da)),
        "dX_dDa_reverse": float(dx_dda), "dX_dDa_forward": float(fx_dda),
        "dX_dDa_fd_model": fd_dda, "dX_dDa_analytical": an_dda,
        "dX_dPe_reverse": float(dx_dpe), "dX_dPe_forward": float(fx_dpe),
        "dX_dPe_fd_model": fd_dpe, "dX_dPe_analytical": an_dpe,
        "time_forward_s": t_forward, "time_reverse_s": t_reverse,
        "time_forward_mode_s": t_fwd_mode,
        "reverse_over_forward": t_reverse / t_forward,
        "forward_mode_over_forward": t_fwd_mode / t_forward,
    }
    records.append(rec)
    print(f"Pe={pe:g} Da={da:g} steps={n_steps:,}: X={rec['X_out']:.4f} "
          f"(analytical {rec['X_out_analytical']:.4f}); "
          f"dX/dDa={rec['dX_dDa_reverse']:.4f} (model FD {fd_dda:.4f}, analytical {an_dda:.4f}); "
          f"dX/dPe={rec['dX_dPe_reverse']:.5f} (model FD {fd_dpe:.5f}, analytical {an_dpe:.5f}); "
          f"forward {t_forward:.3f} s, reverse {t_reverse:.3f} s, "
          f"forward mode {t_fwd_mode:.3f} s", flush=True)

config = {"N": N, "t_final": T_FINAL, "n_reps": N_REPS, "dtype": "float64",
          "device": str(jax.devices()[0]), "jax_version": jax.__version__,
          "discretization": "benchmark_tubular_reactor.py (Danckwerts, RK4, CFL step)"}
with open(OUT_DIR / "reactor_sensitivity.json", "w") as f:
    json.dump({"config": config, "cases": records}, f, indent=2)
with open(OUT_DIR / "reactor_sensitivity.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print(f"Results saved to {OUT_DIR / 'reactor_sensitivity.json'} and .csv")
