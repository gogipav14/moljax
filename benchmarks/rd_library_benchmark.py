"""Reaction-diffusion benchmark driver that calls the moljax solvers.

The three benchmark scripts time the library itself:

- RK4:          ``moljax.core.stepping.integrate_fixed_dt`` (RK4)
- CN:           ``integrate_fixed_dt`` (CN) with ``newton_krylov_solve`` and
                the FFT diffusion preconditioner
- IMEX-Strang:  ``integrate_imex_fixed_dt`` (half diffusion, Heun, half diffusion)
- ETDRK4:       ``moljax.core.fft_integrators.etd_integrate``

The spatial operator is the one of the library: the second-order central
difference Laplacian on a periodic grid, which the FFT diagonalizes exactly.

Each library driver compiles one loop for a given step count, so the
warm-up is one full run. It is not timed.

Environment variables: RD_DT_CN, RD_DT_IMEX, RD_DT_ETD, RD_N_REPS,
RD_T_FINAL, RD_OUT_DIR, RD_TAG (a suffix for the file names) and RD_SKIP
(a comma list of rk4, cn, imex, etdrk4 to omit).
"""

import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import moljax
from moljax.core.fft_integrators import etd_integrate
from moljax.core.fft_operators import DiffusionOperator
from moljax.core.fft_solvers import extract_interior
from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.core.preconditioners import create_fft_preconditioner
from moljax.core.stepping import IntegratorType, integrate_fixed_dt, integrate_imex_fixed_dt
from moljax.core.utils import StatusCode

# Reaction evaluations for each step. The library does not count the
# evaluations of the Newton-Krylov solve, so CN has no NFE value.
NFE_PER_STEP = {"rk4": 4, "imex_strang": 2, "etdrk4": 4}

STATUS_NAMES = {
    int(StatusCode.SUCCESS): "SUCCESS",
    int(StatusCode.NK_FAILED): "NK_FAILED",
    int(StatusCode.NON_FINITE_VALUES): "NON_FINITE_VALUES",
}


def field_summary(u, v) -> dict:
    """Finiteness and statistics of a final state."""
    u, v = np.asarray(u), np.asarray(v)
    finite = bool(np.all(np.isfinite(u)) and np.all(np.isfinite(v)))
    out = {"finite": finite}
    if finite:
        for name, x in (("u", u), ("v", v)):
            out[f"{name}_min"] = float(x.min())
            out[f"{name}_max"] = float(x.max())
            out[f"{name}_mean"] = float(x.mean())
            out[f"{name}_std"] = float(x.std())
    return out


CSV_KEYS = ["name", "dt", "steps", "time_s", "iqr_s", "nfe", "finite", "status",
            "extrapolated", "state_is_at_t",
            "newton_iters_per_step", "krylov_iters_per_step",
            "u_min", "u_max", "u_mean", "u_std", "v_min", "v_max", "v_mean", "v_std"]


def write_csv(methods: list, path) -> None:
    """Write one CSV row for each method record.

    ``extrapolated`` is true when the time is an extrapolation from a short
    run. ``state_is_at_t`` is the time of the field statistics: the end of
    the short run for an extrapolated row, t_final for all other rows.
    """
    with open(path, "w") as f:
        f.write(",".join(CSV_KEYS) + "\n")
        for m in methods:
            row = dict(m)
            row.setdefault("extrapolated", False)
            row.setdefault("state_is_at_t", m.get("t_final"))
            f.write(",".join("" if row.get(k) is None else str(row[k]) for k in CSV_KEYS) + "\n")


def steps_for(t_final: float, dt: float) -> tuple[int, float]:
    """Step count and the step that divides t_final exactly."""
    n = int(np.ceil(t_final / dt - 1e-9))
    return n, t_final / n


def padded_state(model, grid, u0, v0) -> dict:
    """State with ghost cells from two interior arrays."""
    sl_y, sl_x = grid.interior_slice
    state = {}
    for name, x in (("u", u0), ("v", v0)):
        pad = jnp.zeros((grid.ny_total, grid.nx_total), dtype=jnp.float64)
        state[name] = pad.at[sl_y, sl_x].set(jnp.asarray(x, dtype=jnp.float64))
    return model.apply_bcs(state, 0.0)


def _timed(run, n_reps: int):
    """One warm-up call, then n_reps timed calls. Return (median, IQR, last result)."""
    out = run()
    jax.block_until_ready(out)
    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        out = run()
        jax.block_until_ready(out)
        times.append(time.perf_counter() - start)
    q25, q75 = np.percentile(times, [25, 75])
    return float(np.median(times)), float(q75 - q25), out


def make_library_system(make_model, reaction, u0, v0, domain_length: float,
                        t_final: float, nk_params: NKParams | None = None) -> dict:
    """Collect one 2D periodic system for ``library_solve``.

    ``lap_eig`` is the Laplacian symbol of the library (second-order central
    differences). Use it for the RK4 stability limit.
    """
    n = np.asarray(u0).shape[0]
    grid = Grid2D.uniform(n, n, 0.0, domain_length, 0.0, domain_length, n_ghost=1)
    model, fft_cache, diffusivities = make_model(grid)
    if nk_params is None:
        nk_params = NKParams(max_newton_iters=20, max_krylov_iters=60, restart=30,
                             newton_tol=1e-6, krylov_tol=1e-3)
    return {
        "grid": grid, "model": model, "fft_cache": fft_cache,
        "diffusivities": diffusivities, "reaction": reaction, "nk_params": nk_params,
        "Du": diffusivities["u"], "Dv": diffusivities["v"],
        "lap_eig": np.asarray(fft_cache.laplacian_symbol),
        "u0": jnp.asarray(u0, dtype=jnp.float64), "v0": jnp.asarray(v0, dtype=jnp.float64),
        "y0": padded_state(model, grid, u0, v0), "T": float(t_final),
    }


def library_solve(system: dict, method: str, dt: float, n_steps: int):
    """Run n_steps of one moljax solver from the initial state of the system.

    method is 'RK4', 'CN+FFT', 'IMEX-Strang' or 'ETDRK4'. Return
    (u, v, extra) with interior fields. ``extra`` has the status of the run
    and, for CN, the counted Newton and Krylov totals. A failed run returns
    the last good state; examine ``extra['status']``.
    """
    model, grid = system["model"], system["grid"]
    t_end = n_steps * dt
    extra = {}
    if method in ("RK4", "CN+FFT"):
        kwargs = {}
        if method == "CN+FFT":
            kwargs = {"preconditioner": create_fft_preconditioner({"u": "Du", "v": "Dv"},
                                                                  system["fft_cache"]),
                      "nk_params": system["nk_params"]}
        _, _, y, status, totals = integrate_fixed_dt(
            model, system["y0"], 0.0, t_end, dt,
            method=IntegratorType.RK4 if method == "RK4" else IntegratorType.CN,
            save_every=n_steps, return_status=True, return_stats=True, **kwargs)
        u, v = extract_interior(y["u"], grid), extract_interior(y["v"], grid)
        extra["status"] = STATUS_NAMES.get(int(status), str(int(status)))
        if method == "CN+FFT":
            extra["newton_iters"] = int(totals.newton_iters)
            extra["krylov_iters"] = int(totals.lin_iters)
    elif method == "IMEX-Strang":
        _, _, y = integrate_imex_fixed_dt(model, system["y0"], 0.0, t_end, dt,
                                          system["fft_cache"], system["diffusivities"],
                                          use_strang=True, save_every=n_steps)
        u, v = extract_interior(y["u"], grid), extract_interior(y["v"], grid)
    elif method == "ETDRK4":
        reaction = system["reaction"]
        linear_ops = {"u": DiffusionOperator(grid, system["Du"], use_rfft=False),
                      "v": DiffusionOperator(grid, system["Dv"], use_rfft=False)}

        def nonlinear_rhs(state, t):
            ru, rv = reaction(state["u"], state["v"])
            return {"u": ru, "v": rv}

        y = etd_integrate({"u": system["u0"], "v": system["v0"]}, (0.0, t_end), dt,
                          linear_ops, nonlinear_rhs, method="etdrk4",
                          save_every=n_steps)[1][-1]
        u, v = y["u"], y["v"]
    else:
        raise ValueError(method)
    jax.block_until_ready((u, v))
    return u, v, extra


def run_library_benchmark(
    system: str, make_model, reaction, u0, v0, domain_length: float,
    t_final: float, rk4_reaction_dt: float, dt_cn: float, dt_imex: float, dt_etd: float,
    n_reps: int, rk4_short_steps: int | None = None, config: dict | None = None,
    nk_params: NKParams | None = None,
):
    """Time the four moljax solvers on one system and write the result files.

    Args:
        make_model: ``grid -> (model, fft_cache, diffusivities)``, for example
            ``lambda g: create_gray_scott_periodic_fft(g, Du=..., ...)``.
        reaction: ``(u, v) -> (Ru, Rv)``; the nonlinear part for ETDRK4. It
            must be the reaction of the model.
        u0, v0: interior initial fields, shape (N, N).
        rk4_reaction_dt: largest RK4 step that the reaction allows
            (``inf`` when diffusion sets the limit).
        rk4_short_steps: length of a timed RK4 burst. The full RK4 time is
            then an extrapolation.
    """
    t_final = float(os.environ.get("RD_T_FINAL", t_final))
    dt_cn = float(os.environ.get("RD_DT_CN", dt_cn))
    dt_imex = float(os.environ.get("RD_DT_IMEX", dt_imex))
    dt_etd = float(os.environ.get("RD_DT_ETD", dt_etd))
    n_reps = int(os.environ.get("RD_N_REPS", n_reps))
    tag = os.environ.get("RD_TAG", "")
    skip = {x.strip() for x in os.environ.get("RD_SKIP", "").split(",") if x.strip()}
    out_dir = Path(os.environ.get("RD_OUT_DIR", Path(__file__).parent / "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    if nk_params is None:
        nk_params = NKParams(max_newton_iters=20, max_krylov_iters=60, restart=30,
                             newton_tol=1e-6, krylov_tol=1e-3)

    n = np.asarray(u0).shape[0]
    grid = Grid2D.uniform(n, n, 0.0, domain_length, 0.0, domain_length, n_ghost=1)
    model, fft_cache, diffusivities = make_model(grid)
    Du, Dv = diffusivities["u"], diffusivities["v"]
    y0 = padded_state(model, grid, u0, v0)
    methods, finals = [], {}

    def interior(state):
        return (np.asarray(extract_interior(state["u"], grid)),
                np.asarray(extract_interior(state["v"], grid)))

    def record(name, dt, steps, median, iqr, u, v, nfe, extra=None):
        entry = {"name": name, "time_s": median, "iqr_s": iqr, "steps": int(steps),
                 "dt": float(dt), "t_final": float(steps * dt),
                 "nfe": None if nfe is None else int(nfe)}
        entry.update(field_summary(u, v))
        entry.update(extra or {})
        methods.append(entry)
        finals[name] = (np.asarray(u), np.asarray(v))
        t = "no time" if median is None else f"{median:.2f} +/- {iqr:.3f} s"
        print(f"   {name}: {t}, steps {int(steps):,}, dt {dt:g}, finite {entry['finite']}"
              + (f", {extra}" if extra else ""), flush=True)

    # 1. RK4
    if "rk4" not in skip:
        print("\n1. RK4 (moljax integrate_fixed_dt)...", flush=True)
        lam_max = float(jnp.max(-fft_cache.laplacian_symbol)) * max(Du, Dv)
        dt_limit = 0.9 * min(2.8 / lam_max, rk4_reaction_dt)
        n_full, dt_rk4 = steps_for(t_final, dt_limit)
        n_run = n_full if rk4_short_steps is None else min(rk4_short_steps, n_full)

        def run_rk4():
            return integrate_fixed_dt(model, y0, 0.0, n_run * dt_rk4, dt_rk4,
                                      method=IntegratorType.RK4, save_every=n_run,
                                      return_status=True)
        median, iqr, (_, _, y, status) = _timed(run_rk4, n_reps)
        scale = n_full / n_run
        record("RK4 (explicit)", dt_rk4, n_full, median * scale, iqr * scale, *interior(y),
               NFE_PER_STEP["rk4"] * n_full,
               {"extrapolated": n_run < n_full, "measured_steps": n_run,
                "measured_time_s": median, "state_is_at_t": n_run * dt_rk4,
                "status": STATUS_NAMES.get(int(status), str(int(status)))})

    # 2. Crank-Nicolson
    if "cn" not in skip:
        print("\n2. CN (moljax Newton-Krylov, FFT diffusion preconditioner)...", flush=True)
        n_cn, dt = steps_for(t_final, dt_cn)
        precond = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)

        def run_cn():
            return integrate_fixed_dt(model, y0, 0.0, t_final, dt, method=IntegratorType.CN,
                                      save_every=n_cn, preconditioner=precond,
                                      nk_params=nk_params, return_status=True,
                                      return_stats=True)
        start = time.perf_counter()
        _, _, y, status, nk_totals = run_cn()
        jax.block_until_ready(y)
        first = time.perf_counter() - start
        status_name = STATUS_NAMES.get(int(status), str(int(status)))
        extra = {"status": status_name, "valid": status_name == "SUCCESS",
                 "nk_params": nk_params._asdict(),
                 # Counted by the solver. On a failed run the totals stop at
                 # the failed step.
                 "newton_iters_total": int(nk_totals.newton_iters),
                 "krylov_iters_total": int(nk_totals.lin_iters),
                 "newton_iters_per_step": int(nk_totals.newton_iters) / n_cn,
                 "krylov_iters_per_step": int(nk_totals.lin_iters) / n_cn}
        if status_name != "SUCCESS":
            print(f"   CN FAILED with status {status_name}. No time is recorded.", flush=True)
            extra["time_with_compile_to_failure_s"] = first
            # The run did not complete all steps, so a value for each step
            # has no meaning. The field statistics are those of the last
            # state before the failed step; its time is not recorded.
            extra["newton_iters_per_step"] = None
            extra["krylov_iters_per_step"] = None
            extra["state_is_at_t"] = None
            record("CN (Newton-Krylov)", dt, n_cn, None, None, *interior(y), None, extra)
        else:
            times = []
            for _ in range(n_reps):
                start = time.perf_counter()
                _, _, y, status, nk_totals = run_cn()
                jax.block_until_ready(y)
                times.append(time.perf_counter() - start)
            q25, q75 = np.percentile(times, [25, 75])
            record("CN (Newton-Krylov)", dt, n_cn, float(np.median(times)), float(q75 - q25),
                   *interior(y), None, extra)

    # 3. IMEX-Strang
    if "imex" not in skip:
        print("\n3. IMEX-Strang (moljax integrate_imex_fixed_dt)...", flush=True)
        n_imex, dt = steps_for(t_final, dt_imex)

        def run_imex():
            return integrate_imex_fixed_dt(model, y0, 0.0, t_final, dt, fft_cache,
                                           diffusivities, use_strang=True, save_every=n_imex)
        median, iqr, (_, _, y) = _timed(run_imex, n_reps)
        record("IMEX-Strang", dt, n_imex, median, iqr, *interior(y),
               NFE_PER_STEP["imex_strang"] * n_imex)

    # 4. ETDRK4
    if "etdrk4" not in skip:
        print("\n4. ETDRK4 (moljax etd_integrate)...", flush=True)
        n_etd, dt = steps_for(t_final, dt_etd)
        linear_ops = {"u": DiffusionOperator(grid, Du, use_rfft=False),
                      "v": DiffusionOperator(grid, Dv, use_rfft=False)}

        def nonlinear_rhs(state, t):
            ru, rv = reaction(state["u"], state["v"])
            return {"u": ru, "v": rv}

        w0 = {"u": jnp.asarray(u0, dtype=jnp.float64), "v": jnp.asarray(v0, dtype=jnp.float64)}

        def run_etd():
            return etd_integrate(w0, (0.0, t_final), dt, linear_ops, nonlinear_rhs,
                                 method="etdrk4", save_every=n_etd)[1][-1]
        median, iqr, y = _timed(run_etd, n_reps)
        record("ETDRK4", dt, n_etd, median, iqr, y["u"], y["v"], NFE_PER_STEP["etdrk4"] * n_etd)

    results = {"methods": methods, "config": dict(config or {})}
    results["config"].update({
        "system": system, "N": n, "L": domain_length, "t_final": t_final, "n_reps": n_reps,
        "dtype": "float64", "device": str(jax.devices()[0]), "jax_version": jax.__version__,
        "moljax_version": getattr(moljax, "__version__", "unknown"),
        "laplacian": "second-order central differences (moljax), periodic",
        "solvers": "moljax library (integrate_fixed_dt, integrate_imex_fixed_dt, etd_integrate)",
        "timing": "one full warm-up run, then n_reps timed runs; median and IQR",
    })
    stem = system + tag
    with open(out_dir / f"{stem}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    write_csv(methods, out_dir / f"{stem}.csv")
    for name, (u, v) in finals.items():
        short = name.split()[0].lower().replace("-", "_")
        np.save(out_dir / f"{stem}_u_final_{short}.npy", u)
        np.save(out_dir / f"{stem}_v_final_{short}.npy", v)

    print("\n" + "=" * 96)
    print(f"SUMMARY - {system}, moljax solvers, {results['config']['device']}")
    print("=" * 96)
    print(f"{'Method':<20} {'Steps':>14} {'dt':>12} {'Time (s)':>20} {'NFE':>14} {'Finite':>7}")
    for m in methods:
        t = "failed" if m["time_s"] is None else f"{m['time_s']:.2f} +/- {m['iqr_s']:.2f}"
        nfe = "-" if m["nfe"] is None else f"{m['nfe']:,}"
        print(f"{m['name']:<20} {m['steps']:>14,} {m['dt']:>12.6g} {t:>20} {nfe:>14} {str(m['finite']):>7}")
    print("=" * 96)
    print(f"Results saved to {out_dir / (stem + '.json')}")
    return results, finals


if __name__ == "__main__":
    # Write the CSV file of each given result JSON file again.
    import sys

    for json_path in map(Path, sys.argv[1:]):
        with open(json_path) as f:
            write_csv(json.load(f)["methods"], json_path.with_suffix(".csv"))
        print(f"Wrote {json_path.with_suffix('.csv')}")
