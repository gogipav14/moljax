#!/usr/bin/env python3
"""
Step statistics of the moljax adaptive integrators on Gray-Scott.

Solvers (moljax library):
  IMEX-Strang:  adaptive_integrate_imex (error estimate from step doubling)
  RK4:          adaptive_integrate (error estimate from step doubling, CFL limit)

Two setups on a 256 x 256 periodic grid, F = 0.035, k = 0.065, Du = 2e-5,
Dv = 1e-5, t in [0, 10000]:

  pattern:  L = 2.5, random initial condition (the setup of the pattern
            figure; it forms a spot pattern)
  table:    L = 1, the initial condition of benchmark_gray_scott.py (the
            setup of the timing table)

Controller: PIDParams with rtol = 1e-4, atol = 1e-6 (the library defaults),
dt_max = 5, first step 0.01.

Recorded for each run: accepted steps, rejected steps, accept rate, the
status code, the wall time of the second call (the first call compiles) and
the error against a reference. The library stores dt only at each
SAVE_EVERY-th accepted step, so the dt range is the range of these samples.

Reference: moljax ETDRK4 at the fixed step 0.25. The cross-check is the
difference between this reference and ETDRK4 at the step 0.5.

Environment variables:
  RD_OUT_DIR   Output folder for JSON and CSV (default: the results folder).
  RD_T_FINAL   Final time (default 10000).
  RD_SETUPS    Comma list of pattern, table (default: both).
"""

import jax

jax.config.update("jax_enable_x64", True)

import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import rd_library_benchmark as lb

from moljax.core.dt_policy import CFLParams, PIDParams
from moljax.core.fft_solvers import extract_interior
from moljax.core.model import create_gray_scott_periodic_fft
from moljax.core.stepping import IntegratorType, adaptive_integrate, adaptive_integrate_imex
from moljax.core.utils import StatusCode

N = 256
DU, DV, F, K = 2e-5, 1e-5, 0.035, 0.065
T_FINAL = float(os.environ.get("RD_T_FINAL", 10000.0))
RTOL, ATOL, DT_MAX, DT0 = 1e-4, 1e-6, 5.0, 0.01
MAX_STEPS, SAVE_EVERY = 60_000, 50
REF_DT = 0.25
SETUPS = [s.strip() for s in os.environ.get("RD_SETUPS", "pattern,table").split(",")]
OUT_DIR = Path(os.environ.get("RD_OUT_DIR", Path(__file__).parent / "results"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATUS_NAMES = {v: k for k, v in vars(StatusCode).items() if k.isupper()}


def reaction(u, v):
    uvv = u * v * v
    return -uvv + F * (1 - u), uvv - (F + K) * v


def initial_condition(setup):
    np.random.seed(42)
    if setup == "pattern":
        return 2.5, 1.0 - 0.5 * np.random.rand(N, N), 0.25 * np.random.rand(N, N)
    u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
    v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1
    c, r = N // 2, 10
    u0[c - r:c + r, c - r:c + r] = 0.5
    v0[c - r:c + r, c - r:c + r] = 0.25
    return 1.0, u0, v0


def errors(u, v, u_ref, v_ref):
    """(largest absolute difference, relative L2 difference) of the pair (u, v)."""
    u, v, u_ref, v_ref = (np.asarray(x, dtype=np.float64) for x in (u, v, u_ref, v_ref))
    linf = max(np.max(np.abs(u - u_ref)), np.max(np.abs(v - v_ref)))
    num = np.sqrt(np.sum((u - u_ref) ** 2) + np.sum((v - v_ref) ** 2))
    den = np.sqrt(np.sum(u_ref ** 2) + np.sum(v_ref ** 2))
    return float(linf), float(num / den)


print("Adaptive step statistics, Gray-Scott, moljax solvers")
print(f"JAX {jax.__version__}, devices {jax.devices()}, t_final = {T_FINAL:g}, "
      f"rtol = {RTOL:g}, atol = {ATOL:g}, dt_max = {DT_MAX:g}")

pid = PIDParams(rtol=RTOL, atol=ATOL, dt_max=DT_MAX)
cfl = CFLParams(dt_max=DT_MAX)
records, traces = [], []

for setup in SETUPS:
    length, u0, v0 = initial_condition(setup)
    system = lb.make_library_system(
        lambda g: create_gray_scott_periodic_fft(g, Du=DU, Dv=DV, F=F, k=K),
        reaction, u0, v0, length, T_FINAL)
    model, grid = system["model"], system["grid"]

    print(f"\n=== setup {setup} (L = {length:g}) ===", flush=True)
    n_ref, dt_ref = lb.steps_for(T_FINAL, REF_DT)
    u_ref, v_ref, _ = lb.library_solve(system, "ETDRK4", dt_ref, n_ref)
    u_ref2, v_ref2, _ = lb.library_solve(system, "ETDRK4", 2 * dt_ref, n_ref // 2)
    cross_linf, cross_l2 = errors(u_ref2, v_ref2, u_ref, v_ref)
    summary = lb.field_summary(u_ref, v_ref)
    print(f"reference: ETDRK4 dt {dt_ref:g}; cross-check with dt {2 * dt_ref:g}: "
          f"Linf {cross_linf:.2e}, relative L2 {cross_l2:.2e}; "
          f"v in [{summary['v_min']:.4f}, {summary['v_max']:.4f}]", flush=True)

    def run_imex(system=system, model=model):
        return adaptive_integrate_imex(
            model, system["y0"], 0.0, T_FINAL, DT0, system["fft_cache"],
            system["diffusivities"], use_strang=True, max_steps=MAX_STEPS,
            cfl_params=cfl, pid_params=pid, save_every=SAVE_EVERY)

    def run_rk4(system=system, model=model):
        return adaptive_integrate(
            model, system["y0"], 0.0, T_FINAL, DT0, method=IntegratorType.RK4,
            max_steps=MAX_STEPS, cfl_params=cfl, pid_params=pid, save_every=SAVE_EVERY)

    for method, run in (("IMEX-Strang", run_imex), ("RK4", run_rk4)):
        result = run()                                   # compile; not timed
        jax.block_until_ready(result.y_final)
        start = time.perf_counter()
        result = run()
        jax.block_until_ready(result.y_final)
        elapsed = time.perf_counter() - start

        u = extract_interior(result.y_final["u"], grid)
        v = extract_interior(result.y_final["v"], grid)
        accepted, rejected = int(result.n_accepted), int(result.n_rejected)
        status = STATUS_NAMES.get(int(result.status), str(int(result.status)))
        dts = np.asarray(result.dt_history)
        ts = np.asarray(result.t_history)
        valid = dts > 0
        dts, ts = dts[valid], ts[valid]
        reached = float(result.t_final)
        complete = status == "SUCCESS" and abs(reached - T_FINAL) < 1e-9 * T_FINAL
        linf, rel_l2 = errors(u, v, u_ref, v_ref) if complete else (None, None)
        rec = {
            "setup": setup, "method": method, "status": status, "t_reached": reached,
            "accepted": accepted, "rejected": rejected,
            "accept_rate": accepted / max(accepted + rejected, 1),
            "dt_first": DT0,
            "dt_sample_min": float(dts.min()) if dts.size else None,
            "dt_sample_max": float(dts.max()) if dts.size else None,
            "dt_sample_median": float(np.median(dts)) if dts.size else None,
            "dt_samples": int(dts.size), "save_every": SAVE_EVERY,
            "time_s": elapsed, "error_linf": linf, "error_rel_l2": rel_l2,
            "reference_cross_check_linf": cross_linf,
            "reference_cross_check_rel_l2": cross_l2,
            "rtol": RTOL, "atol": ATOL, "dt_max": DT_MAX,
        }
        records.append(rec)
        traces.extend({"setup": setup, "method": method, "t": float(t), "dt": float(d)}
                      for t, d in zip(ts, dts, strict=True))
        print(f"{method}: status {status}, accepted {accepted:,}, rejected {rejected:,}, "
              f"accept rate {100 * rec['accept_rate']:.1f} percent, dt samples "
              f"[{rec['dt_sample_min']}, {rec['dt_sample_max']}], time {elapsed:.1f} s, "
              f"Linf error {linf}, relative L2 error {rel_l2}", flush=True)

config = {"N": N, "Du": DU, "Dv": DV, "F": F, "k": K, "t_final": T_FINAL,
          "max_steps": MAX_STEPS, "reference": f"moljax ETDRK4, dt {REF_DT}",
          "device": str(jax.devices()[0]), "jax_version": jax.__version__,
          "laplacian": "second-order central differences (moljax), periodic"}
with open(OUT_DIR / "adaptive_gray_scott.json", "w") as f:
    json.dump({"config": config, "runs": records}, f, indent=2)
with open(OUT_DIR / "adaptive_gray_scott.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
with open(OUT_DIR / "adaptive_gray_scott_dt_trace.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["setup", "method", "t", "dt"])
    writer.writeheader()
    writer.writerows(traces)
print(f"\nResults saved to {OUT_DIR / 'adaptive_gray_scott.json'}, .csv and the dt trace")
