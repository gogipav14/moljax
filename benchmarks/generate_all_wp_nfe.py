#!/usr/bin/env python3
"""
Generate ALL Work-Precision Diagrams (Error vs NFE) with runtime data.

Systems:
  1. Tubular Reactor (1D, analytical reference)
  2. Diffusion (1D, analytical reference)
  3. Gray-Scott (2D, numerical reference)
  4. Schnakenberg (2D, numerical reference, gamma=100)
  5. Brusselator (2D, numerical reference, b=1.8 Turing regime)

All diagrams use NFE (Nonlinear Function Evaluations) on x-axis.
Runtime is recorded and annotated on figures.

NFE counting (counted, not estimated):
  RK4:         4 NFE/step (4 stages)
  Forward Euler: 1 NFE/step
  CN (linear): 1 solve/step (reported as 1 NFE equivalent)
  IMEX-Strang: 2 NFE/step (Heun reaction step between two half diffusion steps)
  ETDRK4:      4 NFE/step (4 nonlinear evaluations)
  CN+FFT (2D): counted Newton iterations plus counted Krylov iterations

The three 2D systems run on the moljax solvers (see rd_library_benchmark.py):
integrate_fixed_dt (RK4, CN), integrate_imex_fixed_dt and etd_integrate. The
Laplacian is the one of the library (second-order central differences). The
function run_wp_system() is the engine for the 2D systems. The scripts
work_precision_gray_scott.py and work_precision_brusselator.py import it.

References of the 2D systems:
  Primary:   ETDRK4 at a small dt.
  Secondary: RK4 at its stable step (if the cost is acceptable), or else
             IMEX-Strang at a dt at least 8 times smaller than the smallest
             tested dt.
  The relative L2 difference of the two is stored as `reference_cross_check`.
  The error of a method is meaningful only above this value. Each point has
  the flag `above_10x_cross_check`.
  The error of a point is the relative L2 error of the pair (u, v).

Environment variables:
  RD_OUT_DIR   Folder for all output (JSON, CSV, figures). Default: the
               benchmarks/results folder and the figures folder.
  RD_QUICK=1   Short test: fewer dt values and a short final time.
"""

import jax

jax.config.update("jax_enable_x64", True)

import csv
import json
import os
import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np
import rd_library_benchmark as lb

from moljax.core.model import (
    create_brusselator_periodic_fft,
    create_gray_scott_periodic_fft,
    create_schnakenberg_periodic_fft,
)

matplotlib.use('Agg')
import matplotlib.pyplot as plt

QUICK = os.environ.get("RD_QUICK", "0") == "1"
_OUT = os.environ.get("RD_OUT_DIR")
OUTPUT_DIR = Path(_OUT) if _OUT else Path(__file__).parent.parent / "figures"
RESULTS_DIR = Path(_OUT) if _OUT else Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"JAX devices: {jax.devices()}")
if QUICK:
    print("RD_QUICK=1: short test. The results are not for the paper.")

# Common plot style
STYLES = {
    'RK4':           {'color': '#d62728', 'marker': 's', 'label': 'RK4 (4 NFE/step)'},
    'Forward Euler': {'color': '#9467bd', 'marker': 'v', 'label': 'Forward Euler (1 NFE/step)'},
    'CN+FFT':        {'color': '#2ca02c', 'marker': '^', 'label': 'CN+FFT (1 solve/step)'},
    'IMEX-Strang':   {'color': '#1f77b4', 'marker': 'o', 'label': 'IMEX-Strang (2 NFE/step)'},
    'ETDRK4':        {'color': '#ff7f0e', 'marker': 'D', 'label': 'ETDRK4 (4 NFE/step)'},
    'FFT-CN':        {'color': '#1f77b4', 'marker': 'o', 'label': 'PS-FFT-CN (spectral)'},
    'FD-CN':         {'color': '#2ca02c', 'marker': 's', 'label': 'FD-FFT-CN (2nd-order)'},
    'CN':            {'color': '#2ca02c', 'marker': '^', 'label': 'CN (implicit)'},
    'IMEX':          {'color': '#ff7f0e', 'marker': 'D', 'label': 'IMEX (1st-order)'},
}

YLABEL_LINF = '$\\|e\\|_\\infty$ vs reference'
YLABEL_REL_L2 = 'relative $L^2$ error vs reference'


def validate_solution(*arrays, label=""):
    """Check that all arrays are finite (no NaN/Inf). Return True if valid."""
    for arr in arrays:
        if not bool(jnp.all(jnp.isfinite(arr))):
            print(f"  WARNING: NaN/Inf in {label}!")
            return False
    return True


def time_solve(solve_fn, n_reps=3):
    """Time a solver function. Returns (result, median_time_s)."""
    # Warmup
    result = solve_fn()
    if isinstance(result, tuple):
        result[0].block_until_ready()
    else:
        result.block_until_ready()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = solve_fn()
        if isinstance(result, tuple):
            result[0].block_until_ready()
        else:
            result.block_until_ready()
        times.append(time.perf_counter() - t0)

    return result, float(np.median(times))


def method_items(data):
    """The (method, points) pairs of a result dictionary. Skips the reference keys."""
    return [(m, pts) for m, pts in data.items() if not m.startswith('reference_')]


def plot_wp(data, title, filename, xlabel='Nonlinear Function Evaluations (NFE)',
            ylabel=YLABEL_LINF):
    """Generate a work-precision figure.

    A point with no error value (not finite, or a failed CN run) is not drawn.
    If data has `reference_cross_check`, a dashed line shows 10 times this value.
    Errors below the line are not meaningful.
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    for method_name, points in method_items(data):
        valid = [(p['nfe'], p['error'], p.get('time_s') or 0) for p in points
                 if p['error'] is not None and p['error'] > 0 and np.isfinite(p['error'])]
        if not valid:
            continue
        nfes, errs, times = zip(*valid, strict=False)
        s = STYLES.get(method_name, {'color': 'gray', 'marker': 'x', 'label': method_name})
        ax.plot(nfes, errs, marker=s['marker'], color=s['color'], lw=2, ms=8,
                label=s['label'], markeredgecolor='black', markeredgewidth=0.5)

        # Annotate selected points with runtime
        for i, (nfe, err, t) in enumerate(valid):
            if i == 0 or i == len(valid) - 1:  # First and last points
                if t > 0:
                    ax.annotate(f'{t:.2f}s', (nfe, err),
                                textcoords="offset points", xytext=(8, 5),
                                fontsize=7, color=s['color'], alpha=0.8)

    floor = data.get('reference_cross_check')
    if floor:
        ax.axhline(10 * floor, color='gray', ls='--', lw=1,
                   label='10 x reference cross-check')

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{filename}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / f"{filename}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {OUTPUT_DIR / filename}.pdf")


def print_results_table(data, system_name):
    """Print a summary table of results."""
    print(f"\n  {'Method':<15} {'dt':<12} {'Steps':<10} {'NFE':<12} {'Error':<12} {'Time (s)':<10}")
    print(f"  {'-'*71}")
    for method, points in method_items(data):
        for p in points:
            err = p['error']
            err_str = f"{err:.2e}" if err is not None and np.isfinite(err) else "none"
            t_str = f"{p.get('time_s') or 0:.3f}" if (p.get('time_s') or 0) > 0 else "---"
            nfe = p['nfe']
            nfe_str = f"{nfe:,}" if nfe is not None else "---"
            print(f"  {method:<15} {p['dt']:<12.4e} {p['steps']:<10,} {nfe_str:<12} {err_str:<12} {t_str:<10}")


CSV_COLUMNS = ["system", "method", "dt", "steps", "nfe", "time_s", "error", "finite",
               "valid", "above_10x_cross_check", "status",
               "reference_cross_check"]


def write_json_csv(data, system, stem):
    """Write RESULTS_DIR/<stem>.json and <stem>.csv (one CSV row for each (system, method, dt))."""
    with open(RESULTS_DIR / f"{stem}.json", 'w') as f:
        json.dump(data, f, indent=2)
    cross = data.get('reference_cross_check')
    with open(RESULTS_DIR / f"{stem}.csv", 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(CSV_COLUMNS)
        for method, pts in method_items(data):
            for p in pts:
                err = p.get('error')
                finite = p.get('finite', err is not None and bool(np.isfinite(err)))
                row = {
                    "system": system, "method": method, "dt": p['dt'], "steps": p['steps'],
                    "nfe": p['nfe'], "time_s": p.get('time_s'), "error": err,
                    "finite": finite, "valid": p.get('valid', finite),
                    "above_10x_cross_check": p.get('above_10x_cross_check'),
                    "status": p.get("status"),
                    "reference_cross_check": cross,
                }
                w.writerow(["" if row[c] is None else row[c] for c in CSV_COLUMNS])
    print(f"  Saved: {RESULTS_DIR / stem}.json and .csv")


# =============================================================================
# Engine for the 2D reaction-diffusion systems (moljax solvers through rd_library_benchmark.py)
# =============================================================================

def pick(values, n=4):
    """In quick mode: n values spread over the list. Otherwise: the list."""
    values = list(values)
    if not QUICK or len(values) <= n:
        return values
    idx = sorted({int(round(i)) for i in np.linspace(0, len(values) - 1, n)})
    return [values[i] for i in idx]


def rel_l2(u, v, u_ref, v_ref):
    """Relative L2 difference of the pair (u, v) to the pair (u_ref, v_ref)."""
    u, v, u_ref, v_ref = (np.asarray(x, dtype=np.float64) for x in (u, v, u_ref, v_ref))
    num = np.sqrt(np.sum((u - u_ref) ** 2) + np.sum((v - v_ref) ** 2))
    den = np.sqrt(np.sum(u_ref ** 2) + np.sum(v_ref ** 2))
    return float(num / den)


def linf(u, v, u_ref, v_ref):
    u, v, u_ref, v_ref = (np.asarray(x, dtype=np.float64) for x in (u, v, u_ref, v_ref))
    return float(max(np.max(np.abs(u - u_ref)), np.max(np.abs(v - v_ref))))


def run_method(method, dt_req, system, n_reps, warm=True):
    """Run one moljax solver with one dt over [0, T]. Return (u, v, record).

    The step count is n = ceil(T / dt). The step is dt_eff = T / n, so the
    run ends exactly at T. `record` has the fields that do not need a reference.

    A library driver compiles one loop for each step count, so the warm-up
    is one full run. With warm=False there is no warm-up and the one run is
    timed with its compilation (the record has time_includes_compile).
    """
    T = system["T"]
    n = int(np.ceil(T / dt_req - 1e-9))
    dt = T / n
    if warm:
        lb.library_solve(system, method, dt, n)      # compile; not timed
    else:
        n_reps = 1
    times, out = [], None
    for _ in range(n_reps):
        t0 = time.perf_counter()
        out = lb.library_solve(system, method, dt, n)
        times.append(time.perf_counter() - t0)

    u, v, extra = out
    if method == 'CN+FFT':
        # Counted by the solver: one residual evaluation for each Newton
        # iteration and one Jacobian product for each Krylov iteration. The
        # evaluations of the first guess and of the line search are not in it.
        nfe = extra['newton_iters'] + extra['krylov_iters']
    else:
        nfe = lb.NFE_PER_STEP[{'RK4': 'rk4', 'IMEX-Strang': 'imex_strang',
                               'ETDRK4': 'etdrk4'}[method]] * n
    u, v = np.asarray(u), np.asarray(v)
    rec = {'dt': float(dt), 'dt_requested': float(dt_req), 'steps': n, 'nfe': nfe,
           'time_s': float(np.median(times)), 'time_includes_compile': not warm}
    rec.update(extra)
    return u, v, rec


def run_wp_system(system, sweeps, ref_dt, dt_rk4_stable, n_reps=2, rk4_ref_cap=2_000_000):
    """Run the reference runs and the sweeps of one 2D system.

    system:        dict with lap_eig, Du, Dv, reaction, u0, v0, T.
    sweeps:        {method: [dt, ...]} with the methods RK4, IMEX-Strang, ETDRK4, CN+FFT.
    ref_dt:        dt of the primary reference (ETDRK4).
    dt_rk4_stable: dt at the RK4 stability limit.
    Return a dictionary: {method: [point, ...], 'reference_cross_check', 'reference_info'}.
    """
    T = system['T']
    # --- Primary reference: ETDRK4 at ref_dt ---
    print(f"  Primary reference: ETDRK4, dt={ref_dt} ...", flush=True)
    u_ref, v_ref, rec1 = run_method('ETDRK4', ref_dt, system, 1, warm=False)
    assert np.all(np.isfinite(u_ref)) and np.all(np.isfinite(v_ref)), "Primary reference is not finite!"
    print(f"    {rec1['steps']:,} steps, {rec1['time_s']:.1f} s, u in "
          f"[{u_ref.min():.4f}, {u_ref.max():.4f}]")

    # --- Secondary reference: RK4 at its stable step, else IMEX-Strang ---
    dt_rk4_ref = min(ref_dt, 0.9 * dt_rk4_stable)
    n_rk4_ref = int(np.ceil(T / dt_rk4_ref))
    if n_rk4_ref <= rk4_ref_cap:
        sec_method, sec_dt = 'RK4', dt_rk4_ref
    else:
        smallest = min(min(v) for v in sweeps.values() if v)
        sec_method, sec_dt = 'IMEX-Strang', smallest / 8
    print(f"  Secondary reference: {sec_method}, dt={sec_dt:.4g} ...", flush=True)
    u2, v2, rec2 = run_method(sec_method, sec_dt, system, 1, warm=False)
    sec_finite = bool(np.all(np.isfinite(u2)) and np.all(np.isfinite(v2)))
    cross = rel_l2(u2, v2, u_ref, v_ref) if sec_finite else None
    print(f"    {rec2['steps']:,} steps, {rec2['time_s']:.1f} s, "
          f"cross-check = {cross if cross is None else format(cross, '.3e')}")

    data = {}
    for method, dts in sweeps.items():
        print(f"  --- {method} ---", flush=True)
        pts = []
        for dt_req in dts:
            reps = n_reps if rec_cost(method, T / dt_req) <= 400_000 else 1
            u, v, rec = run_method(method, dt_req, system, reps)
            finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
            valid = finite and rec.get('status', 'SUCCESS') == 'SUCCESS'
            err = rel_l2(u, v, u_ref, v_ref) if valid else None
            rec.update({
                'error': err,
                'error_linf': linf(u, v, u_ref, v_ref) if valid else None,
                'finite': finite, 'valid': valid,
                'above_10x_cross_check': (bool(err > 10 * cross)
                                          if err is not None and cross is not None else None),
            })
            note = "" if valid else "  (NO ERROR VALUE: not finite or the run failed)"
            print(f"    dt={rec['dt']:.4g}, steps={rec['steps']:,}, NFE={rec['nfe']:,}, "
                  f"err={'none' if err is None else format(err, '.2e')}, "
                  f"{rec['time_s']:.2f}s{note}", flush=True)
            pts.append(rec)
        data[method] = pts

    data['reference_cross_check'] = cross
    data['reference_info'] = {
        'error_norm': 'relative L2 of (u, v)',
        'primary': {'method': 'ETDRK4', 'dt': rec1['dt'], 'steps': rec1['steps'],
                    'time_s': rec1['time_s']},
        'secondary': {'method': sec_method, 'dt': rec2['dt'], 'steps': rec2['steps'],
                      'time_s': rec2['time_s'], 'finite': sec_finite},
        'flag': 'above_10x_cross_check: error > 10 * reference_cross_check',
        'quick': QUICK,
        't_final': T,
        'solvers': 'moljax library (integrate_fixed_dt, integrate_imex_fixed_dt, etd_integrate)',
        'laplacian': 'second-order central differences (moljax), periodic',
        'jax_version': jax.__version__, 'device': str(jax.devices()[0]),
    }
    return data


def rec_cost(method, n_steps):
    """Rough cost of one run in reaction evaluations (for the number of timing repeats)."""
    return {'RK4': 4, 'IMEX-Strang': 2, 'ETDRK4': 4, 'CN+FFT': 40}[method] * n_steps


def make_2d_system(make_model, reaction, u0, v0, L, T):
    """Collect the fields of a 2D periodic system for run_wp_system.

    make_model: ``grid -> (model, fft_cache, diffusivities)`` (a moljax factory).
    reaction:   ``(u, v) -> (Ru, Rv)``; it must be the reaction of the model.
    """
    return lb.make_library_system(make_model, reaction, u0, v0, L, T)


# =============================================================================
# 1. REACTOR: Error vs NFE with analytical reference
# =============================================================================

def run_reactor_wp():
    print("\n" + "=" * 70)
    print("1. REACTOR Work-Precision (analytical reference)")
    print("=" * 70)

    Pe, Da, N = 10.0, 1.0, 128
    T_FINAL = 0.3 if QUICK else 15.0
    dz = 1.0 / (N - 1)
    z = np.linspace(0, 1, N)

    # Analytical steady state (Danckwerts BC)
    disc = Pe**2 + 4*Pe*Da
    r1 = (Pe + np.sqrt(disc)) / 2
    r2 = (Pe - np.sqrt(disc)) / 2
    M = np.array([[r1/Pe - 1, r2/Pe - 1], [r1*np.exp(r1), r2*np.exp(r2)]])
    AB = np.linalg.solve(M, np.array([-1.0, 0.0]))
    c_analytical = AB[0] * np.exp(r1 * z) + AB[1] * np.exp(r2 * z)
    c_anal_jax = jnp.array(c_analytical)

    # FD matrices
    D2 = np.zeros((N, N))
    for i in range(1, N-1):
        D2[i, i-1] = 1.0; D2[i, i] = -2.0; D2[i, i+1] = 1.0
    D2 /= dz**2
    D1 = np.zeros((N, N))
    for i in range(1, N-1):
        D1[i, i-1] = -0.5; D1[i, i+1] = 0.5
    D1[0, 0] = -1.0; D1[0, 1] = 1.0
    D1[-1, -2] = -1.0; D1[-1, -1] = 1.0
    D1 /= dz
    D2_jax, D1_jax = jnp.array(D2), jnp.array(D1)

    inlet_coeff = (1.0/Pe) / dz
    c0 = jnp.zeros(N)

    @jax.jit
    def reactor_rhs(c, t):
        dcdt = (1.0/Pe) * (D2_jax @ c) - (D1_jax @ c) - Da * c
        c0_target = (1.0 + inlet_coeff * c[1]) / (1.0 + inlet_coeff)
        dcdt = dcdt.at[0].set(100.0 * (c0_target - c[0]))
        dcdt = dcdt.at[-1].set(100.0 * (c[-2] - c[-1]))
        return dcdt

    # CFL limits
    dt_adv = 0.5 * dz
    dt_diff = 0.25 * Pe * dz**2
    dt_cfl = min(dt_adv, dt_diff)
    print(f"  CFL limit: {dt_cfl:.4e}")

    # ---- RK4 ----
    @partial(jax.jit, static_argnums=(1, 2))
    def solve_rk4_reactor(c0_in, dt, n_steps):
        def step(i, c):
            t = i * dt
            k1 = reactor_rhs(c, t)
            k2 = reactor_rhs(c + 0.5*dt*k1, t + 0.5*dt)
            k3 = reactor_rhs(c + 0.5*dt*k2, t + 0.5*dt)
            k4 = reactor_rhs(c + dt*k3, t + dt)
            return c + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return jax.lax.fori_loop(0, n_steps, step, c0_in)

    # ---- Forward Euler ----
    @partial(jax.jit, static_argnums=(1, 2))
    def solve_fe_reactor(c0_in, dt, n_steps):
        def step(i, c):
            return c + dt * reactor_rhs(c, i * dt)
        return jax.lax.fori_loop(0, n_steps, step, c0_in)

    data = {}

    # RK4 sweep
    print("  --- RK4 ---")
    rk4_results = []
    for frac in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        dt = frac * dt_cfl
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 4 * n_steps
        result, t_s = time_solve(lambda dt=dt, n=n_steps: solve_rk4_reactor(c0, dt, n))
        if validate_solution(result, label=f"RK4 dt={dt:.4e}"):
            err = float(jnp.max(jnp.abs(result - c_anal_jax)))
            print(f"    dt={dt:.4e} ({frac}x CFL), NFE={nfe:,}, err={err:.2e}, {t_s:.3f}s")
            rk4_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                'error': err, 'time_s': t_s})
    data['RK4'] = rk4_results

    # Forward Euler sweep
    print("  --- Forward Euler ---")
    fe_results = []
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9]:
        dt = frac * dt_cfl
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 1 * n_steps
        result, t_s = time_solve(lambda dt=dt, n=n_steps: solve_fe_reactor(c0, dt, n))
        if validate_solution(result, label=f"FE dt={dt:.4e}"):
            err = float(jnp.max(jnp.abs(result - c_anal_jax)))
            print(f"    dt={dt:.4e} ({frac}x CFL), NFE={nfe:,}, err={err:.2e}, {t_s:.3f}s")
            fe_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                               'error': err, 'time_s': t_s})
    data['Forward Euler'] = fe_results

    print_results_table(data, "Reactor")
    plot_wp(data,
            f'Work-Precision: Tubular Reactor (Pe={int(Pe)}, Da={int(Da)}, N={N})',
            'fig_reactor_work_precision')

    write_json_csv(data, 'reactor', 'wp_reactor')

    return data


# =============================================================================
# 2. DIFFUSION: PS-FFT-CN vs FD-FFT-CN, NFE-based
# =============================================================================

def run_diffusion_wp():
    print("\n" + "=" * 70)
    print("2. DIFFUSION Work-Precision (analytical reference)")
    print("=" * 70)

    N, D, t_final = 128, 0.1, 0.1
    dx = 1.0 / N
    x = np.linspace(0, 1, N, endpoint=False)

    u0 = jnp.sin(2 * np.pi * x)

    def u_exact(t):
        return np.sin(2 * np.pi * x) * np.exp(-4 * np.pi**2 * D * t)

    # Spectral eigenvalues
    kx_spec = jnp.fft.fftfreq(N, dx) * 2 * jnp.pi
    lap_eig_spec = -kx_spec**2

    # FD eigenvalues (circulant)
    k_fd = jnp.fft.fftfreq(N, 1.0/N)
    lap_eig_fd = (2.0 / dx**2) * (jnp.cos(2 * jnp.pi * k_fd / N) - 1)

    data = {}

    for method_name, lap_eig in [('FFT-CN', lap_eig_spec), ('FD-CN', lap_eig_fd)]:
        print(f"  --- {method_name} ---")
        results = []

        for dt in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]:
            n_steps = int(np.ceil(t_final / dt))
            t_actual = n_steps * dt
            nfe = n_steps  # 1 solve per step

            cn_factor = (1 + 0.5 * dt * D * lap_eig) / (1 - 0.5 * dt * D * lap_eig)

            @jax.jit
            def diffusion_step(u_in, _, cn_f=cn_factor):
                u_hat = jnp.fft.fft(u_in)
                return jnp.real(jnp.fft.ifft(u_hat * cn_f)), None

            def solve():
                return jax.lax.scan(diffusion_step, u0, None, length=n_steps)[0]

            result, t_s = time_solve(solve)
            if validate_solution(result, label=f"{method_name} dt={dt}"):
                err = float(jnp.max(jnp.abs(result - jnp.array(u_exact(t_actual)))))
                print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.4f}s")
                results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                'error': err, 'time_s': t_s})

        data[method_name] = results

    print_results_table(data, "Diffusion")
    plot_wp(data,
            'Work-Precision: Spectral vs FD Diffusion (N=128, periodic)',
            'fig_diffusion_work_precision',
            xlabel='Linear Solves (steps)')

    write_json_csv(data, 'diffusion', 'wp_diffusion')

    return data


# =============================================================================
# 3. GRAY-SCOTT: Error vs NFE
# =============================================================================

def run_gray_scott_wp():
    print("\n" + "=" * 70)
    print("3. GRAY-SCOTT Work-Precision (numerical reference)")
    print("=" * 70)

    N = 256
    Du_gs, Dv_gs = 2e-5, 1e-5
    F_gs, k_gs = 0.035, 0.065
    L_gs = 1.0
    T_FINAL = 10.0 if QUICK else 100.0  # Early transient for clean convergence

    # Initial condition
    np.random.seed(42)
    u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
    v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1
    cx, cy = N//2, N//2
    r = 10
    u0[cx-r:cx+r, cy-r:cy+r] = 0.5
    v0[cx-r:cx+r, cy-r:cy+r] = 0.25

    # Reaction terms
    def reaction(u, v):
        uvv = u * v * v
        return -uvv + F_gs * (1 - u), uvv - (F_gs + k_gs) * v

    system = make_2d_system(
        lambda g: create_gray_scott_periodic_fft(g, Du=Du_gs, Dv=Dv_gs, F=F_gs, k=k_gs),
        reaction, u0, v0, L_gs, T_FINAL)

    # Stability: diffusion CFL
    lambda_max = float(np.max(-system['lap_eig']))
    dt_rk4_max = 2.8 / (max(Du_gs, Dv_gs) * lambda_max)
    print(f"  RK4 stability limit: dt_max = {dt_rk4_max:.4f}")
    print("  Reaction eigenvalues: mild (~0.1), no reaction stability issue")

    sweeps = {
        'RK4': [frac * dt_rk4_max for frac in pick([0.5, 0.6, 0.7, 0.8, 0.9])],
        'IMEX-Strang': pick([0.1, 0.2, 0.5, 1.0, 2.0, 5.0]),
        'ETDRK4': pick([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]),
    }
    data = run_wp_system(system, sweeps, ref_dt=0.01, dt_rk4_stable=dt_rk4_max)

    print_results_table(data, "Gray-Scott")
    plot_wp(data,
            f'Work-Precision: Gray-Scott ($256^2$, $t={int(T_FINAL)}$)',
            'fig_work_precision_gray_scott', ylabel=YLABEL_REL_L2)
    write_json_csv(data, 'gray_scott', 'wp_gray_scott')
    return data


# =============================================================================
# 4. SCHNAKENBERG: Error vs NFE (gamma=100)
# =============================================================================

def run_schnakenberg_wp():
    print("\n" + "=" * 70)
    print("4. SCHNAKENBERG Work-Precision (gamma=100, numerical reference)")
    print("=" * 70)

    N = 256
    Du_sn, Dv_sn = 1.0, 40.0
    a_sn, b_sn, gamma_sn = 0.1, 0.9, 100.0
    L_sn = 10.0
    T_FINAL = 0.1 if QUICK else 5.0  # Early Turing growth phase

    # Steady state and IC
    u_s = a_sn + b_sn
    v_s = b_sn / (a_sn + b_sn)**2
    np.random.seed(42)
    u0 = u_s + 0.01 * np.random.randn(N, N)
    v0 = v_s + 0.01 * np.random.randn(N, N)

    # Reaction terms
    def reaction(u, v):
        return gamma_sn * (a_sn - u + u**2 * v), gamma_sn * (b_sn - u**2 * v)

    system = make_2d_system(
        lambda g: create_schnakenberg_periodic_fft(g, Du=Du_sn, Dv=Dv_sn, a=a_sn, b=b_sn,
                                                   gamma=gamma_sn),
        reaction, u0, v0, L_sn, T_FINAL)

    # Stability analysis
    lambda_max = float(np.max(-system['lap_eig']))
    dt_rk4_max = 2.8 / (Dv_sn * lambda_max)
    print(f"  RK4 stability limit (diffusion): dt_max = {dt_rk4_max:.6e}")

    # Reaction eigenvalues at steady state
    fu = gamma_sn * (-1 + 2*u_s*v_s)
    fv = gamma_sn * (u_s**2)
    gu = gamma_sn * (-2*u_s*v_s)
    gv = gamma_sn * (-u_s**2)
    tr_rxn = fu + gv
    det_rxn = fu*gv - fv*gu
    disc_rxn = tr_rxn**2 - 4*det_rxn
    lam_rxn = complex(tr_rxn/2, np.sqrt(abs(disc_rxn))/2 if disc_rxn < 0 else 0)
    print(f"  Reaction eigenvalues: {tr_rxn/2:.1f} +/- {np.sqrt(abs(disc_rxn))/2:.1f}i")
    print(f"  |lambda_rxn| ~ {abs(lam_rxn):.1f}")

    # IMEX stability: need |1 + (dt/2)*lambda| <= 1 for Strang half-step
    # Conservative limit for IMEX-Strang
    dt_imex_max = 1.8 / abs(lam_rxn)  # ~0.018 for gamma=100
    print(f"  IMEX-Strang stability limit (reaction): dt_max ~ {dt_imex_max:.4f}")

    # ETDRK4: reaction treated with RK4-like scheme, limit ~ 2.8/|Im(lambda)|
    im_part = np.sqrt(abs(disc_rxn))/2
    dt_etdrk4_max = 2.8 / im_part if im_part > 0 else 1.0
    print(f"  ETDRK4 stability limit (reaction): dt_max ~ {dt_etdrk4_max:.4f}")

    # The old sweep had no RK4 point. The RK4 stable step is the secondary
    # reference if the cost is acceptable (cap in quick mode: 10,000 steps).
    sweeps = {
        'IMEX-Strang': pick([0.0002, 0.0005, 0.001, 0.0015]),
        'ETDRK4': pick([0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]),
    }
    data = run_wp_system(system, sweeps, ref_dt=0.0001, dt_rk4_stable=dt_rk4_max,
                         rk4_ref_cap=10_000 if QUICK else 2_000_000)

    print_results_table(data, "Schnakenberg")
    plot_wp(data,
            f'Work-Precision: Schnakenberg ($256^2$, $\\gamma=100$, $t={int(T_FINAL)}$)',
            'fig_work_precision_schnakenberg', ylabel=YLABEL_REL_L2)
    write_json_csv(data, 'schnakenberg', 'wp_schnakenberg')
    return data


# =============================================================================
# 5. BRUSSELATOR: Error vs NFE (b=1.8, Turing regime)
# =============================================================================

def run_brusselator_wp():
    print("\n" + "=" * 70)
    print("5. BRUSSELATOR Work-Precision (b=1.8, Turing regime, numerical ref)")
    print("=" * 70)

    N = 256
    Du_br, Dv_br = 0.01, 0.1
    a_br, b_br = 1.0, 1.8
    L_br = 5.0
    T_FINAL = 2.0 if QUICK else 100.0  # Long enough for Turing growth

    # Steady state and IC
    u_s = a_br
    v_s = b_br / a_br
    np.random.seed(42)
    u0 = u_s + 0.05 * np.random.randn(N, N)
    v0 = v_s + 0.05 * np.random.randn(N, N)

    # Reaction terms
    def reaction(u, v):
        u2v = u**2 * v
        return a_br - (b_br + 1) * u + u2v, b_br * u - u2v

    system = make_2d_system(
        lambda g: create_brusselator_periodic_fft(g, Du=Du_br, Dv=Dv_br, a=a_br, b=b_br),
        reaction, u0, v0, L_br, T_FINAL)

    # Stability analysis
    lambda_max = float(np.max(-system['lap_eig']))
    dt_rk4_max = 2.8 / (Dv_br * lambda_max)
    print(f"  RK4 stability limit (diffusion): dt_max = {dt_rk4_max:.6e}")

    # Reaction eigenvalues
    fu = -(b_br+1) + 2*u_s*v_s   # = -2.8 + 3.6 = 0.8
    fv = u_s**2                    # = 1.0
    gu = b_br - 2*u_s*v_s         # = 1.8 - 3.6 = -1.8
    gv = -u_s**2                   # = -1.0
    tr_rxn = fu + gv
    det_rxn = fu*gv - fv*gu
    disc_rxn = tr_rxn**2 - 4*det_rxn
    im_part = np.sqrt(abs(disc_rxn))/2 if disc_rxn < 0 else 0
    re_part = tr_rxn / 2
    print(f"  Reaction eigenvalues: {re_part:.2f} +/- {im_part:.2f}i (mild)")

    # IMEX stability: |1 + (dt/2)*lambda| <= 1
    # Conservative: dt/2 * |lambda| < 1 => dt < 2/|lambda|
    abs_lam = np.sqrt(re_part**2 + im_part**2)
    dt_imex_max = 1.8 / abs_lam if abs_lam > 0 else 10.0
    print(f"  IMEX-Strang stability limit: dt_max ~ {dt_imex_max:.2f}")

    # RK4: single point, as before (0.9 of the stability limit).
    sweeps = {
        'RK4': [0.9 * dt_rk4_max],
        'IMEX-Strang': pick([0.01, 0.02, 0.05, 0.1, 0.2]),
        'ETDRK4': pick([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]),
    }
    data = run_wp_system(system, sweeps, ref_dt=0.001, dt_rk4_stable=dt_rk4_max)

    print_results_table(data, "Brusselator")
    plot_wp(data,
            f'Work-Precision: Brusselator ($256^2$, $b=1.8$, $t={int(T_FINAL)}$)',
            'fig_work_precision_brusselator', ylabel=YLABEL_REL_L2)
    write_json_csv(data, 'brusselator', 'wp_brusselator')
    return data


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    all_results = {}

    t_total_start = time.perf_counter()

    all_results['reactor'] = run_reactor_wp()
    all_results['diffusion'] = run_diffusion_wp()
    all_results['gray_scott'] = run_gray_scott_wp()
    all_results['schnakenberg'] = run_schnakenberg_wp()
    all_results['brusselator'] = run_brusselator_wp()

    total_time = time.perf_counter() - t_total_start
    print(f"\n{'=' * 70}")
    print(f"ALL WORK-PRECISION DIAGRAMS COMPLETE in {total_time:.0f}s")
    print(f"{'=' * 70}")
    print(f"Figures saved to: {OUTPUT_DIR}")
    print(f"Data saved to: {RESULTS_DIR}")

    # Summary: count points per system, and points with no error value
    for system, data in all_results.items():
        pts = [p for _, v in method_items(data) for p in v]
        n_none = sum(1 for p in pts if p['error'] is None)
        status = "OK" if n_none == 0 else f"{n_none} point(s) with no error value"
        cross = data.get('reference_cross_check')
        cross_str = "" if cross is None else f", cross-check {cross:.2e}"
        print(f"  {system}: {len(pts)} data points, {status}{cross_str}")
