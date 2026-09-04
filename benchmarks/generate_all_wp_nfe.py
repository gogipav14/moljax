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

NFE counting:
  RK4:         4 NFE/step (4 stages)
  Forward Euler: 1 NFE/step
  CN (linear): 1 solve/step (reported as 1 NFE equivalent)
  IMEX-Strang: 2 NFE/step (2 half-step reaction evaluations)
  ETDRK4:      4 NFE/step (4 nonlinear evaluations)
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from functools import partial
from pathlib import Path

import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

print(f"JAX devices: {jax.devices()}")

OUTPUT_DIR = Path(__file__).parent.parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

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


def plot_wp(data, title, filename, xlabel='Nonlinear Function Evaluations (NFE)'):
    """Generate a work-precision figure."""
    fig, ax = plt.subplots(figsize=(10, 7))

    for method_name, points in data.items():
        valid = [(p['nfe'], p['error'], p.get('time_s', 0)) for p in points
                 if p['error'] > 0 and np.isfinite(p['error'])]
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

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel('$\\|e\\|_\\infty$ vs reference', fontsize=13)
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
    for method, points in data.items():
        for p in points:
            err_str = f"{p['error']:.2e}" if np.isfinite(p['error']) else "NaN"
            t_str = f"{p.get('time_s', 0):.3f}" if p.get('time_s', 0) > 0 else "---"
            print(f"  {method:<15} {p['dt']:<12.4e} {p['steps']:<10,} {p['nfe']:<12,} {err_str:<12} {t_str:<10}")


# =============================================================================
# ETDRK4 builder (reused for Gray-Scott, Schnakenberg, Brusselator)
# =============================================================================

def compute_etdrk4_coeffs_2d(L_2d, dt_val, M=32):
    """Compute ETDRK4 coefficients for 2D linear operator."""
    r = np.exp(1j * np.pi * (np.arange(1, M+1) - 0.5) / M)
    LR = dt_val * L_2d[..., np.newaxis] + r[np.newaxis, np.newaxis, :]
    E = np.exp(dt_val * L_2d)
    E2 = np.exp(dt_val * L_2d / 2)
    Q = dt_val * np.real(np.mean((np.exp(LR/2) - 1) / LR, axis=-1))
    f1 = dt_val * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3*LR + LR**2)) / LR**3, axis=-1))
    f2 = dt_val * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=-1))
    f3 = dt_val * np.real(np.mean((-4 - 3*LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=-1))
    return [jnp.array(x) for x in [E, E2, Q, f1, f2, f3]]


def make_2d_etdrk4(Lu, Lv, dt_val, Nu_fn, Nv_fn):
    """Create an ETDRK4 stepper for a 2-field system."""
    Eu, E2u, Qu, f1u, f2u, f3u = compute_etdrk4_coeffs_2d(Lu, dt_val)
    Ev, E2v, Qv, f1v, f2v, f3v = compute_etdrk4_coeffs_2d(Lv, dt_val)

    @jax.jit
    def step(state, _):
        u, v = state
        uh, vh = jnp.fft.fft2(u), jnp.fft.fft2(v)
        nu, nv = jnp.fft.fft2(Nu_fn(u, v)), jnp.fft.fft2(Nv_fn(u, v))

        au = jnp.real(jnp.fft.ifft2(E2u * uh + Qu * nu))
        av = jnp.real(jnp.fft.ifft2(E2v * vh + Qv * nv))
        nau, nav = jnp.fft.fft2(Nu_fn(au, av)), jnp.fft.fft2(Nv_fn(au, av))

        bu = jnp.real(jnp.fft.ifft2(E2u * uh + Qu * nau))
        bv = jnp.real(jnp.fft.ifft2(E2v * vh + Qv * nav))
        nbu, nbv = jnp.fft.fft2(Nu_fn(bu, bv)), jnp.fft.fft2(Nv_fn(bu, bv))

        cu = jnp.real(jnp.fft.ifft2(E2u * (E2u * uh + Qu * (2 * nbu - nu))))
        cv = jnp.real(jnp.fft.ifft2(E2v * (E2v * vh + Qv * (2 * nbv - nv))))
        ncu, ncv = jnp.fft.fft2(Nu_fn(cu, cv)), jnp.fft.fft2(Nv_fn(cu, cv))

        u_new = jnp.real(jnp.fft.ifft2(Eu * uh + f1u * nu + f2u * (nau + nbu) + f3u * ncu))
        v_new = jnp.real(jnp.fft.ifft2(Ev * vh + f1v * nv + f2v * (nav + nbv) + f3v * ncv))
        return (u_new, v_new), None
    return step


def make_2d_imex_strang(lap_eig, Du, Dv, Nu_fn, Nv_fn, dt_val):
    """Create IMEX-Strang stepper for a 2-field system."""
    exp_u = jnp.array(np.exp(dt_val * Du * lap_eig))
    exp_v = jnp.array(np.exp(dt_val * Dv * lap_eig))

    @jax.jit
    def step(state, _):
        u, v = state
        # Half reaction
        du, dv = Nu_fn(u, v), Nv_fn(u, v)
        u = u + 0.5 * dt_val * du
        v = v + 0.5 * dt_val * dv
        # Full diffusion (exact in Fourier space)
        u = jnp.real(jnp.fft.ifft2(exp_u * jnp.fft.fft2(u)))
        v = jnp.real(jnp.fft.ifft2(exp_v * jnp.fft.fft2(v)))
        # Half reaction
        du, dv = Nu_fn(u, v), Nv_fn(u, v)
        u = u + 0.5 * dt_val * du
        v = v + 0.5 * dt_val * dv
        return (u, v), None
    return step


def make_2d_rk4(lap_eig_jax, Du, Dv, Nu_fn, Nv_fn, dt_val):
    """Create RK4 stepper for a 2-field system (full RHS including diffusion)."""
    @jax.jit
    def rhs(u, v):
        uh, vh = jnp.fft.fft2(u), jnp.fft.fft2(v)
        lap_u = jnp.real(jnp.fft.ifft2(lap_eig_jax * uh))
        lap_v = jnp.real(jnp.fft.ifft2(lap_eig_jax * vh))
        return Du * lap_u + Nu_fn(u, v), Dv * lap_v + Nv_fn(u, v)

    @jax.jit
    def step(state, _):
        u, v = state
        k1u, k1v = rhs(u, v)
        k2u, k2v = rhs(u + 0.5*dt_val*k1u, v + 0.5*dt_val*k1v)
        k3u, k3v = rhs(u + 0.5*dt_val*k2u, v + 0.5*dt_val*k2v)
        k4u, k4v = rhs(u + dt_val*k3u, v + dt_val*k3v)
        u_new = u + dt_val/6 * (k1u + 2*k2u + 2*k3u + k4u)
        v_new = v + dt_val/6 * (k1v + 2*k2v + 2*k3v + k4v)
        return (u_new, v_new), None
    return step


def run_scan_solver(step_fn, state, n_steps, block_size=2000):
    """Run a lax.scan-based solver in blocks to avoid memory issues."""
    @partial(jax.jit, static_argnums=(1,))
    def run_block(s, n):
        return jax.lax.scan(step_fn, s, None, length=n)[0]

    # Warmup
    _ = run_block(state, min(10, n_steps))
    if isinstance(state, tuple):
        state[0].block_until_ready()
    else:
        state.block_until_ready()

    n_blocks = n_steps // block_size
    remainder = n_steps - n_blocks * block_size

    for _i in range(n_blocks):
        state = run_block(state, block_size)
        if isinstance(state, tuple):
            state[0].block_until_ready()
        else:
            state.block_until_ready()

    if remainder > 0:
        state = run_block(state, remainder)
        if isinstance(state, tuple):
            state[0].block_until_ready()
        else:
            state.block_until_ready()

    return state


# =============================================================================
# 1. REACTOR: Error vs NFE with analytical reference
# =============================================================================

def run_reactor_wp():
    print("\n" + "=" * 70)
    print("1. REACTOR Work-Precision (analytical reference)")
    print("=" * 70)

    Pe, Da, N = 10.0, 1.0, 128
    T_FINAL = 15.0
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

    with open(RESULTS_DIR / "wp_reactor.json", 'w') as f:
        json.dump(data, f, indent=2)

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

    with open(RESULTS_DIR / "wp_diffusion.json", 'w') as f:
        json.dump(data, f, indent=2)

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
    dx = L_gs / N
    T_FINAL = 100.0  # Early transient for clean convergence

    # Wavenumbers
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)
    lap_eig_jax = jnp.array(lap_eig)

    # Initial condition
    np.random.seed(42)
    u0 = np.ones((N, N)) - 0.5 * np.random.rand(N, N) * 0.1
    v0 = 0.25 * np.ones((N, N)) + 0.5 * np.random.rand(N, N) * 0.1
    cx, cy = N//2, N//2
    r = 10
    u0[cx-r:cx+r, cy-r:cy+r] = 0.5
    v0[cx-r:cx+r, cy-r:cy+r] = 0.25
    u0_jax, v0_jax = jnp.array(u0), jnp.array(v0)

    # Reaction terms
    def Nu_gs(u, v): return -u * v * v + F_gs * (1 - u)
    def Nv_gs(u, v): return u * v * v - (F_gs + k_gs) * v

    # Stability: diffusion CFL
    lambda_max = float(np.max(-lap_eig))
    dt_rk4_max = 2.8 / (max(Du_gs, Dv_gs) * lambda_max)
    print(f"  RK4 stability limit: dt_max = {dt_rk4_max:.4f}")
    print("  Reaction eigenvalues: mild (~0.1), no reaction stability issue")

    # --- Reference: ETDRK4 dt=0.01 ---
    ref_dt = 0.01
    ref_steps = int(np.ceil(T_FINAL / ref_dt))
    print(f"  Computing reference (ETDRK4 dt={ref_dt}, {ref_steps} steps)...")
    Lu_gs = Du_gs * lap_eig
    Lv_gs = Dv_gs * lap_eig
    ref_step = make_2d_etdrk4(Lu_gs, Lv_gs, ref_dt, Nu_gs, Nv_gs)
    ref_state = run_scan_solver(ref_step, (u0_jax, v0_jax), ref_steps, block_size=2000)
    u_ref, v_ref = ref_state
    print(f"  Reference u: [{float(jnp.min(u_ref)):.4f}, {float(jnp.max(u_ref)):.4f}]")
    assert validate_solution(u_ref, v_ref, label="GS reference"), "Reference is NaN!"

    def gs_error(u, v):
        return max(float(jnp.max(jnp.abs(u - u_ref))), float(jnp.max(jnp.abs(v - v_ref))))

    data = {}

    # RK4 sweep (stability-limited)
    print("  --- RK4 ---")
    rk4_results = []
    for frac in [0.5, 0.6, 0.7, 0.8, 0.9]:
        dt = frac * dt_rk4_max
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 4 * n_steps
        step_fn = make_2d_rk4(lap_eig_jax, Du_gs, Dv_gs, Nu_gs, Nv_gs, dt)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0_jax, v0_jax), ns, block_size=2000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"RK4 dt={dt:.4f}"):
            err = gs_error(*result)
            print(f"    dt={dt:.4f}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            rk4_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                'error': err, 'time_s': t_s})
    data['RK4'] = rk4_results

    # IMEX-Strang sweep
    print("  --- IMEX-Strang ---")
    imex_results = []
    for dt in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 2 * n_steps
        step_fn = make_2d_imex_strang(lap_eig, Du_gs, Dv_gs, Nu_gs, Nv_gs, dt)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0_jax, v0_jax), ns, block_size=2000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"IMEX dt={dt}"):
            err = gs_error(*result)
            print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            imex_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                 'error': err, 'time_s': t_s})
    data['IMEX-Strang'] = imex_results

    # ETDRK4 sweep
    print("  --- ETDRK4 ---")
    etd_results = []
    for dt in [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 4 * n_steps
        step_fn = make_2d_etdrk4(Lu_gs, Lv_gs, dt, Nu_gs, Nv_gs)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0_jax, v0_jax), ns, block_size=2000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"ETDRK4 dt={dt}"):
            err = gs_error(*result)
            print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            etd_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                'error': err, 'time_s': t_s})
    data['ETDRK4'] = etd_results

    print_results_table(data, "Gray-Scott")
    plot_wp(data,
            f'Work-Precision: Gray-Scott ($256^2$, $t={int(T_FINAL)}$)',
            'fig_work_precision_gray_scott')

    with open(RESULTS_DIR / "wp_gray_scott.json", 'w') as f:
        json.dump(data, f, indent=2)

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
    dx = L_sn / N
    T_FINAL = 5.0  # Early Turing growth phase

    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)

    # Steady state and IC
    u_s = a_sn + b_sn
    v_s = b_sn / (a_sn + b_sn)**2
    np.random.seed(42)
    u0 = jnp.array(u_s + 0.01 * np.random.randn(N, N))
    v0 = jnp.array(v_s + 0.01 * np.random.randn(N, N))

    # Reaction terms
    def Nu_sn(u, v): return gamma_sn * (a_sn - u + u**2 * v)
    def Nv_sn(u, v): return gamma_sn * (b_sn - u**2 * v)

    # Stability analysis
    lambda_max = float(np.max(-lap_eig))
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

    # --- Reference: ETDRK4 dt=0.0001 ---
    ref_dt = 0.0001
    ref_steps = int(np.ceil(T_FINAL / ref_dt))
    print(f"  Computing reference (ETDRK4 dt={ref_dt}, {ref_steps} steps)...")
    Lu_sn = Du_sn * lap_eig
    Lv_sn = Dv_sn * lap_eig
    ref_step = make_2d_etdrk4(Lu_sn, Lv_sn, ref_dt, Nu_sn, Nv_sn)
    ref_state = run_scan_solver(ref_step, (u0, v0), ref_steps, block_size=5000)
    u_ref, v_ref = ref_state
    print(f"  Reference u: [{float(jnp.min(u_ref)):.4f}, {float(jnp.max(u_ref)):.4f}]")
    assert validate_solution(u_ref, v_ref, label="Snak reference"), "Reference is NaN!"

    def sn_error(u, v):
        return max(float(jnp.max(jnp.abs(u - u_ref))), float(jnp.max(jnp.abs(v - v_ref))))

    data = {}

    # IMEX-Strang sweep (dt up to ~0.002)
    print("  --- IMEX-Strang ---")
    imex_results = []
    for dt in [0.0002, 0.0005, 0.001, 0.0015]:
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 2 * n_steps
        step_fn = make_2d_imex_strang(lap_eig, Du_sn, Dv_sn, Nu_sn, Nv_sn, dt)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0, v0), ns, block_size=5000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"IMEX dt={dt}"):
            err = sn_error(*result)
            print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            imex_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                 'error': err, 'time_s': t_s})
        else:
            print(f"    dt={dt} SKIPPED (NaN)")
    data['IMEX-Strang'] = imex_results

    # ETDRK4 sweep (dt up to ~0.02)
    print("  --- ETDRK4 ---")
    etd_results = []
    for dt in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]:
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 4 * n_steps
        step_fn = make_2d_etdrk4(Lu_sn, Lv_sn, dt, Nu_sn, Nv_sn)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0, v0), ns, block_size=5000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"ETDRK4 dt={dt}"):
            err = sn_error(*result)
            print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            etd_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                'error': err, 'time_s': t_s})
        else:
            print(f"    dt={dt} SKIPPED (NaN)")
    data['ETDRK4'] = etd_results

    print_results_table(data, "Schnakenberg")
    plot_wp(data,
            f'Work-Precision: Schnakenberg ($256^2$, $\\gamma=100$, $t={int(T_FINAL)}$)',
            'fig_work_precision_schnakenberg')

    with open(RESULTS_DIR / "wp_schnakenberg.json", 'w') as f:
        json.dump(data, f, indent=2)

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
    dx = L_br / N
    T_FINAL = 100.0  # Long enough for Turing growth

    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig = -(KX**2 + KY**2)
    lap_eig_jax = jnp.array(lap_eig)

    # Steady state and IC
    u_s = a_br
    v_s = b_br / a_br
    np.random.seed(42)
    u0 = jnp.array(u_s + 0.05 * np.random.randn(N, N))
    v0 = jnp.array(v_s + 0.05 * np.random.randn(N, N))

    # Reaction terms
    def Nu_br(u, v): return a_br - (b_br + 1) * u + u**2 * v
    def Nv_br(u, v): return b_br * u - u**2 * v

    # Stability analysis
    lambda_max = float(np.max(-lap_eig))
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
    # For Strang half-step with lambda = re + i*im
    # Conservative: dt/2 * |lambda| < 1 => dt < 2/|lambda|
    abs_lam = np.sqrt(re_part**2 + im_part**2)
    dt_imex_max = 1.8 / abs_lam if abs_lam > 0 else 10.0
    print(f"  IMEX-Strang stability limit: dt_max ~ {dt_imex_max:.2f}")

    Lu_br = Du_br * lap_eig
    Lv_br = Dv_br * lap_eig

    # --- Reference: ETDRK4 dt=0.001 ---
    ref_dt = 0.001
    ref_steps = int(np.ceil(T_FINAL / ref_dt))
    print(f"  Computing reference (ETDRK4 dt={ref_dt}, {ref_steps} steps)...")
    ref_step = make_2d_etdrk4(Lu_br, Lv_br, ref_dt, Nu_br, Nv_br)
    ref_state = run_scan_solver(ref_step, (u0, v0), ref_steps, block_size=5000)
    u_ref, v_ref = ref_state
    print(f"  Reference u: [{float(jnp.min(u_ref)):.4f}, {float(jnp.max(u_ref)):.4f}]")
    assert validate_solution(u_ref, v_ref, label="Brus reference"), "Reference is NaN!"

    def br_error(u, v):
        return max(float(jnp.max(jnp.abs(u - u_ref))), float(jnp.max(jnp.abs(v - v_ref))))

    data = {}

    # RK4 sweep (single point at stability limit - too expensive for sweep)
    print("  --- RK4 (single point, stability-limited) ---")
    rk4_results = []
    dt_rk4 = 0.9 * dt_rk4_max
    n_steps_rk4 = int(np.ceil(T_FINAL / dt_rk4))
    nfe_rk4 = 4 * n_steps_rk4
    print(f"    dt={dt_rk4:.6e}, steps={n_steps_rk4:,}, NFE={nfe_rk4:,}")
    step_fn = make_2d_rk4(lap_eig_jax, Du_br, Dv_br, Nu_br, Nv_br, dt_rk4)

    def solve_rk4(sf=step_fn, ns=n_steps_rk4):
        return run_scan_solver(sf, (u0, v0), ns, block_size=5000)

    result, t_s = time_solve(solve_rk4, n_reps=2)
    if validate_solution(*result, label="RK4"):
        err = br_error(*result)
        print(f"    err={err:.2e}, {t_s:.2f}s")
        rk4_results.append({'dt': float(dt_rk4), 'steps': n_steps_rk4, 'nfe': nfe_rk4,
                            'error': err, 'time_s': t_s})
    data['RK4'] = rk4_results

    # IMEX-Strang sweep
    print("  --- IMEX-Strang ---")
    imex_results = []
    for dt in [0.01, 0.02, 0.05, 0.1, 0.2]:
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 2 * n_steps
        step_fn = make_2d_imex_strang(lap_eig, Du_br, Dv_br, Nu_br, Nv_br, dt)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0, v0), ns, block_size=5000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"IMEX dt={dt}"):
            err = br_error(*result)
            print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            imex_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                 'error': err, 'time_s': t_s})
        else:
            print(f"    dt={dt} SKIPPED (NaN)")
    data['IMEX-Strang'] = imex_results

    # ETDRK4 sweep
    print("  --- ETDRK4 ---")
    etd_results = []
    for dt in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        n_steps = int(np.ceil(T_FINAL / dt))
        nfe = 4 * n_steps
        step_fn = make_2d_etdrk4(Lu_br, Lv_br, dt, Nu_br, Nv_br)

        def solve(sf=step_fn, ns=n_steps):
            return run_scan_solver(sf, (u0, v0), ns, block_size=5000)

        result, t_s = time_solve(solve, n_reps=2)
        if validate_solution(*result, label=f"ETDRK4 dt={dt}"):
            err = br_error(*result)
            print(f"    dt={dt}, NFE={nfe:,}, err={err:.2e}, {t_s:.2f}s")
            etd_results.append({'dt': float(dt), 'steps': n_steps, 'nfe': nfe,
                                'error': err, 'time_s': t_s})
        else:
            print(f"    dt={dt} SKIPPED (NaN)")
    data['ETDRK4'] = etd_results

    print_results_table(data, "Brusselator")
    plot_wp(data,
            f'Work-Precision: Brusselator ($256^2$, $b=1.8$, $t={int(T_FINAL)}$)',
            'fig_work_precision_brusselator')

    with open(RESULTS_DIR / "wp_brusselator.json", 'w') as f:
        json.dump(data, f, indent=2)

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

    # Summary: count valid points per system
    for system, data in all_results.items():
        total_points = sum(len(v) for v in data.values())
        nan_free = all(np.isfinite(p['error']) for pts in data.values() for p in pts)
        status = "OK" if nan_free else "HAS NaN"
        print(f"  {system}: {total_points} data points, {status}")
