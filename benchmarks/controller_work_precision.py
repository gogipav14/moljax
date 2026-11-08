"""Work--precision comparison for the adaptive-controller error exponent.

Configuration ``method_order`` reproduces the inherited controller table;
``estimator_order`` uses the power of ``h`` in the actual local error
estimate.  Work is counted in right-hand-side evaluations.  ``NKStats`` does
not expose spent Krylov iterations, so the JSON keeps its configured budget
separate from Newton iterations and does not present it as measured work.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from moljax.core.bc import BCType, FieldBCSpec
from moljax.core.dt_policy import CFLParams, PIDParams, heisenberg_cfl_dt, imex_cfl_dt
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.model import MOLModel, create_gray_scott_periodic_fft
from moljax.core.newton_krylov import NKParams
from moljax.core.operators import LinearOp, NonlinearOp, laplacian_1d
from moljax.core.state import scaled_error_norm, tree_sub
from moljax.core.stepping import (
    bdf2_step,
    be_step,
    cn_step,
    euler_step,
    imex_euler_step,
    imex_strang_step,
    rk4_step,
    ssprk3_step,
)

METHODS = {
    "euler": 0,
    "ssprk3": 1,
    "rk4": 2,
    "be": 3,
    "cn": 4,
    "bdf2": 5,
    "imex_euler": 6,
    "imex_strang": 7,
}
METHOD_ORDER = [1, 3, 4, 1, 2, 2, 1, 2]
ESTIMATOR_ORDER = [2, 4, 5, 2, 2, 2, 2, 3]
STAGES = {"euler": 1, "ssprk3": 3, "rk4": 4}


@dataclass
class Result:
    problem: str
    method: str
    config: str
    tolerance: float
    accepted: int
    rejected: int
    rejection_fraction: float
    rhs_evaluations: int
    newton_iterations: int
    gmres_budget: int
    final_error: float
    t_final: float
    success: bool


class RuntimeCounter:
    def __init__(self) -> None:
        self.value = 0

    def hit(self, _value) -> None:
        self.value += 1

    def reset(self) -> None:
        self.value = 0


def block_tree(tree):
    return jax.tree_util.tree_map(lambda x: x.block_until_ready(), tree)


def finite_tree(tree) -> bool:
    leaves = jax.tree_util.tree_leaves(tree)
    return all(bool(jnp.all(jnp.isfinite(x))) for x in leaves)


def interior_rms_error(y, ref, grid) -> float:
    if isinstance(grid, Grid1D):
        sl = grid.interior_slice
    else:
        sl = grid.interior_slice
    num = 0.0
    den = 0.0
    n = 0
    for key in y:
        d = np.asarray(y[key][sl] - ref[key][sl], dtype=np.float64)
        r = np.asarray(ref[key][sl], dtype=np.float64)
        num += float(np.sum(d * d))
        den += float(np.sum(r * r))
        n += d.size
    rms = math.sqrt(num / n)
    scale = max(math.sqrt(den / n), 1.0e-14)
    return rms / scale


def instrument_model(model: MOLModel, counter: RuntimeCounter) -> MOLModel:
    """Count one callback per model.rhs call through the first available op."""
    if model.linear_ops:
        first = model.linear_ops[0]
        original = first.apply

        def counted(state, grid, t, params):
            jax.debug.callback(counter.hit, jnp.asarray(t))
            return original(state, grid, t, params)

        replacement = LinearOp(first.name, counted, first.dt_bound)
        return MOLModel(
            grid=model.grid,
            bc_spec=model.bc_spec,
            params=model.params,
            linear_ops=(replacement,) + model.linear_ops[1:],
            nonlinear_ops=model.nonlinear_ops,
            metadata=model.metadata,
        )
    first = model.nonlinear_ops[0]
    original = first.apply

    def counted(state, grid, t, params):
        jax.debug.callback(counter.hit, jnp.asarray(t))
        return original(state, grid, t, params)

    replacement = NonlinearOp(first.name, counted, first.dt_bound)
    return MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops,
        nonlinear_ops=(replacement,) + model.nonlinear_ops[1:],
        metadata=model.metadata,
    )


@dataclass
class Problem:
    name: str
    model: MOLModel
    y0: dict
    t0: float
    t_end: float
    dt0: float
    reference: dict
    counter: RuntimeCounter
    fft_cache: object | None = None
    diffusivities: dict | None = None


def make_pr_problem() -> Problem:
    grid = Grid1D.uniform(8, 0.0, 1.0)
    x = grid.x_coords(include_ghost=True)
    a = 5.0

    def linear_apply(state, grid, t, params):
        return {"u": -a * state["u"]}

    def nonlinear_apply(state, grid, t, params):
        phi = jnp.cos(t + x)
        dphi = -jnp.sin(t + x)
        u = state["u"]
        return {"u": -u * u + a * phi + dphi + phi * phi}

    base = MOLModel(
        grid=grid,
        bc_spec={"u": FieldBCSpec(kind=BCType.PERIODIC)},
        params={"dtype": jnp.float64},
        linear_ops=(LinearOp("minus_five", linear_apply),),
        nonlinear_ops=(NonlinearOp("forcing", nonlinear_apply),),
        metadata={"pde_type": "ode_array"},
    )
    y0 = base.apply_bcs({"u": jnp.cos(x)}, 0.0)
    ref = base.apply_bcs({"u": jnp.cos(4.0 + x)}, 4.0)
    counter = RuntimeCounter()
    return Problem(
        "prothero_robinson", instrument_model(base, counter), y0, 0.0, 4.0, 0.01, ref, counter
    )


def gray_reference(model: MOLModel, y0: dict, t_end: float) -> dict:
    n = 8192
    dt = t_end / n

    def body(i, y):
        return rk4_step(model, y, jnp.asarray(i, jnp.float64) * dt, dt)

    run = jax.jit(lambda y: jax.lax.fori_loop(0, n, body, y))
    return block_tree(run(y0))


def make_gray_problem() -> Problem:
    grid = Grid2D.uniform(8, 8, 0.0, 2.5, 0.0, 2.5)
    base, fft_cache, diffusivities = create_gray_scott_periodic_fft(grid)
    X, Y = grid.meshgrid(include_ghost=True)
    bump = jnp.exp(-8.0 * ((X - 1.25) ** 2 + (Y - 1.25) ** 2))
    y0 = base.apply_bcs({"u": 1.0 - 0.5 * bump, "v": 0.25 * bump}, 0.0)
    ref = gray_reference(base, y0, 1.0)
    counter = RuntimeCounter()
    model = instrument_model(base, counter)
    return Problem(
        "gray_scott_8x8", model, y0, 0.0, 1.0, 0.05, ref, counter, fft_cache, diffusivities
    )


def make_heat_problem() -> Problem:
    grid = Grid1D.uniform(256, 0.0, 1.0)
    x = grid.x_coords(include_ghost=True)

    def diffusion_apply(state, grid, t, params):
        return {"u": params["D"] * laplacian_1d(state["u"], grid)}

    def diffusion_bound(grid, params):
        return 0.5 * grid.dx**2 / params["D"]

    base = MOLModel(
        grid=grid,
        bc_spec={"u": FieldBCSpec(kind=BCType.PERIODIC)},
        params={"D": 1.0, "dtype": jnp.float64},
        linear_ops=(LinearOp("heat", diffusion_apply, diffusion_bound),),
        nonlinear_ops=(),
        metadata={"pde_type": "diffusion"},
    )
    # Fundamental mode plus a stiff checkerboard mode.  The latter makes
    # dt0*|lambda_max| ~= 5.2e3 and is essentially gone by t_end.
    idx = jnp.arange(grid.nx_total)
    fundamental = jnp.sin(2.0 * jnp.pi * x)
    checker = 0.1 * (-1.0) ** idx
    y0 = base.apply_bcs({"u": fundamental + checker}, 0.0)
    lam1 = -4.0 / grid.dx**2 * math.sin(math.pi / grid.nx) ** 2
    lammax = -4.0 / grid.dx**2
    t_end = 0.0002
    ref = base.apply_bcs(
        {"u": fundamental * jnp.exp(lam1 * t_end) + checker * jnp.exp(lammax * t_end)},
        t_end,
    )
    counter = RuntimeCounter()
    return Problem(
        "heat_1d_256", instrument_model(base, counter), y0, 0.0, t_end, 0.02, ref, counter
    )


class Stepper:
    def __init__(self, problem: Problem):
        m = problem.model
        nk = NKParams()
        self.euler = jax.jit(lambda y, t, dt: euler_step(m, y, t, dt))
        self.ssprk3 = jax.jit(lambda y, t, dt: ssprk3_step(m, y, t, dt))
        self.rk4 = jax.jit(lambda y, t, dt: rk4_step(m, y, t, dt))
        self.be = jax.jit(lambda y, t, dt: be_step(m, y, t, dt, nk_params=nk))
        self.cn = jax.jit(lambda y, t, dt: cn_step(m, y, t, dt, nk_params=nk))
        self.bdf2 = jax.jit(lambda y, yp, t, dt, dtp: bdf2_step(m, y, yp, t, dt, dtp, nk_params=nk))
        if problem.fft_cache is not None:
            cache = problem.fft_cache
            diffs = problem.diffusivities
            self.imex_euler = jax.jit(lambda y, t, dt: imex_euler_step(m, y, t, dt, cache, diffs))
            self.imex_strang = jax.jit(lambda y, t, dt: imex_strang_step(m, y, t, dt, cache, diffs))


def aggregate_stats(stats):
    if not stats:
        return True, 0, 0, None
    vals = []
    for s in stats:
        vals.append((bool(s.converged), int(s.newton_iters), int(s.lin_iters), s))
    ok = all(v[0] for v in vals)
    return ok, sum(v[1] for v in vals), sum(v[2] for v in vals), vals[0][3]


def attempt(problem, stepper, method, y, y_prev, t, dt, dt_prev, step_count):
    zt = jnp.asarray(t, jnp.float64)
    zh = jnp.asarray(dt, jnp.float64)
    stats = []
    if method in STAGES:
        fn = getattr(stepper, method)
        full = fn(y, zt, zh)
        half1 = fn(y, zt, zh / 2)
        half2 = fn(half1, zt + zh / 2, zh / 2)
        y_new, err = half2, tree_sub(half2, full)
    elif method.startswith("imex"):
        fn = getattr(stepper, method)
        full = fn(y, zt, zh)
        half1 = fn(y, zt, zh / 2)
        half2 = fn(half1, zt + zh / 2, zh / 2)
        y_new, err = half2, tree_sub(half2, full)
    elif method == "be":
        y_be, sbe = stepper.be(y, zt, zh)
        y_cn, scn = stepper.cn(y, zt, zh)
        y_new, err, stats = y_be, tree_sub(y_cn, y_be), [sbe, scn]
    elif method == "cn":
        y_cn, scn = stepper.cn(y, zt, zh)
        y_be, sbe = stepper.be(y, zt, zh)
        y_new, err, stats = y_cn, tree_sub(y_cn, y_be), [scn, sbe]
    elif method == "bdf2" and step_count == 0:
        y_be, sbe = stepper.be(y, zt, zh)
        y_cn, scn = stepper.cn(y, zt, zh)
        y_new, err, stats = y_cn, tree_sub(y_cn, y_be), [scn, sbe]
    else:
        y_bdf, sb = stepper.bdf2(y, y_prev, zt, zh, jnp.asarray(dt_prev))
        y_be, sbe = stepper.be(y, zt, zh)
        y_new, err, stats = y_bdf, tree_sub(y_bdf, y_be), [sb, sbe]

    block_tree((y_new, err, stats))
    ok, newton, gmres, primary = aggregate_stats(stats)
    return y_new, err, ok and finite_tree(y_new), newton, gmres, primary


def controller_dt(dt, err, prev_err, order, accepted, implicit_primary, pid):
    err_safe = max(err, 1.0e-10)
    if accepted:
        factor = (
            pid.safety
            * err_safe ** (-(pid.kI + pid.kP) / order)
            * max(prev_err, 1e-10) ** (pid.kP / order)
        )
        factor = min(max(factor, pid.min_factor), pid.max_factor)
    else:
        factor = pid.safety * max(err_safe, 1.0) ** (-pid.kI / order)
        factor = min(max(factor, pid.min_factor), 1.0)
    out = dt * factor
    if implicit_primary is not None:
        if not bool(implicit_primary.converged):
            out *= 0.5
        elif int(implicit_primary.newton_iters) <= 3:
            out *= 1.2
    out = min(max(out, pid.dt_min), pid.dt_max)
    if not accepted:
        out = max(out * 0.5, pid.dt_min)
    return out


def run_one(problem, stepper, method, config, tol):
    method_id = METHODS[method]
    order = METHOD_ORDER[method_id] if config == "method_order" else ESTIMATOR_ORDER[method_id]
    pid = PIDParams(atol=tol, rtol=tol)
    cfl = CFLParams()
    y = problem.y0
    y_prev = problem.y0
    t = problem.t0
    dt = min(problem.dt0, problem.t_end - problem.t0)
    dt_prev = dt
    step_count = 0
    accepted = rejected = newton = gmres = 0
    prev_err = 1.0
    consecutive_rejects = 0
    problem.counter.reset()
    max_attempts = 50000
    if method_id < 3:
        cfl_limit = float(
            heisenberg_cfl_dt(problem.model.grid, problem.model.params, cfl, method=method_id)
        )
    elif method_id >= 6:
        cfl_limit = float(imex_cfl_dt(problem.model.grid, problem.model.params, cfl))
    else:
        cfl_limit = math.inf
    while t < problem.t_end - 1e-14 and accepted + rejected < max_attempts:
        h = min(dt, problem.t_end - t)
        y_new, err, solves_ok, ni, gi, primary = attempt(
            problem, stepper, method, y, y_prev, t, h, dt_prev, step_count
        )
        newton += ni
        gmres += gi
        ratio = float(scaled_error_norm(err, y_new, problem.model.grid, tol, tol))
        ratio = max(ratio, 1e-10)
        take = ratio <= 1.0 and solves_ok
        new_dt = controller_dt(h, ratio, prev_err, order, take, primary, pid)
        new_dt = min(new_dt, cfl_limit)
        if take:
            old_y = y
            t += h
            y = y_new
            y_prev = old_y
            dt_prev = h
            step_count += 1
            accepted += 1
            prev_err = ratio
            consecutive_rejects = 0
        else:
            rejected += 1
            consecutive_rejects += 1
            if consecutive_rejects >= 10:
                break
        dt = new_dt
    block_tree(y)
    if method in STAGES:
        rhs = (accepted + rejected) * 3 * STAGES[method]
    elif method == "imex_euler":
        rhs = (accepted + rejected) * 3
    elif method == "imex_strang":
        rhs = (accepted + rejected) * 6
    else:
        rhs = problem.counter.value
    success = t >= problem.t_end - 1e-12 and consecutive_rejects < 10
    error = interior_rms_error(y, problem.reference, problem.model.grid) if success else math.inf
    total = accepted + rejected
    return Result(
        problem.name,
        method,
        config,
        tol,
        accepted,
        rejected,
        rejected / total if total else 0.0,
        rhs,
        newton,
        gmres,
        error,
        t,
        success,
    )


def plot_results(results, outdir):
    marker_by_config = {"method_order": "o", "estimator_order": "s"}
    config_order = ["method_order", "estimator_order"]
    problems = sorted({r.problem for r in results})
    for problem in problems:
        methods = sorted({r.method for r in results if r.problem == problem})
        fig, axes = plt.subplots(
            2,
            max(1, math.ceil(len(methods) / 2)),
            figsize=(4.2 * math.ceil(len(methods) / 2), 7.0),
            squeeze=False,
        )
        for ax, method in zip(axes.flat, methods, strict=False):
            configs = [
                c
                for c in config_order
                if any(
                    r.config == c and r.problem == problem and r.method == method for r in results
                )
            ]
            successful_errors = [
                r.final_error
                for r in results
                if r.problem == problem and r.method == method and r.success
            ]
            fail_x = max(successful_errors) * 1.5 if successful_errors else 1.0
            for index, config in enumerate(configs):
                rows = [
                    r
                    for r in results
                    if r.problem == problem
                    and r.method == method
                    and r.config == config
                    and r.success
                ]
                rows.sort(key=lambda r: r.final_error)
                # Tiny display-only offset separates exactly coincident series.
                offset = 10.0 ** ((index - (len(configs) - 1) / 2.0) * 0.008)
                ax.loglog(
                    [r.final_error * offset for r in rows],
                    [r.rhs_evaluations for r in rows],
                    marker=marker_by_config[config],
                    linestyle="-",
                    label=config,
                )
                failed = [
                    r
                    for r in results
                    if r.problem == problem
                    and r.method == method
                    and r.config == config
                    and not r.success
                ]
                for failed_index, row in enumerate(failed):
                    y = row.rhs_evaluations * (1.0 + 0.025 * (index + failed_index))
                    ax.plot(fail_x * offset, y, marker="X", markersize=8, linestyle="None")
                    ax.annotate(
                        f"{config} FAIL {row.tolerance:.1e}",
                        (fail_x * offset, y),
                        fontsize=6,
                        rotation=35,
                    )
            ax.set_title(method)
            ax.set_xlabel("relative RMS final error")
            ax.set_ylabel("RHS evaluations")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend()
            if problem == "heat_1d_256" and method in STAGES:
                ax.text(
                    0.03,
                    0.04,
                    "CFL-limited, controller inactive",
                    transform=ax.transAxes,
                    fontsize=8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
                )
        for ax in axes.flat[len(methods) :]:
            ax.axis("off")
        fig.suptitle(f"Controller work-precision: {problem}")
        fig.tight_layout()
        fig.savefig(outdir / f"controller_work_precision_{problem}.png", dpi=160)
        plt.close(fig)


def matched_error_summary(results):
    """Interpolate log NFE at common errors for estimator/method-order ratios."""
    summary = []
    pairs = sorted({(row.problem, row.method) for row in results})
    for problem, method in pairs:
        series = {}
        for config in ("method_order", "estimator_order"):
            selected = [
                row
                for row in results
                if row.problem == problem
                and row.method == method
                and row.config == config
                and row.success
            ]
            best = {}
            for row in selected:
                best[row.final_error] = min(
                    best.get(row.final_error, float("inf")), row.rhs_evaluations
                )
            errors = sorted(best)
            series[config] = (
                np.log(np.asarray(errors)),
                np.log(np.asarray([best[error] for error in errors])),
            )
        x_old, y_old = series["method_order"]
        x_new, y_new = series["estimator_order"]
        lower = max(float(x_old.min()), float(x_new.min()))
        upper = min(float(x_old.max()), float(x_new.max()))
        if upper <= lower:
            continue
        common = np.linspace(lower, upper, 101)
        ratio = np.exp(np.interp(common, x_new, y_new) - np.interp(common, x_old, y_old))
        summary.append(
            {
                "problem": problem,
                "method": method,
                "common_error_min": float(np.exp(lower)),
                "common_error_max": float(np.exp(upper)),
                "estimator_over_method_order_nfe_median": float(np.median(ratio)),
                "estimator_over_method_order_nfe_min": float(np.min(ratio)),
                "estimator_over_method_order_nfe_max": float(np.max(ratio)),
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/controller_work_precision.json"),
    )
    args = parser.parse_args()
    problems = [make_pr_problem(), make_gray_problem(), make_heat_problem()]
    tolerances = list(np.logspace(-2, -5, 7)) if not args.quick else [1e-3]
    results = []
    for problem in problems:
        stepper = Stepper(problem)
        methods = ["euler", "ssprk3", "rk4", "be", "cn", "bdf2"]
        if problem.fft_cache is not None:
            methods += ["imex_euler", "imex_strang"]
        for method in methods:
            configs = ["method_order", "estimator_order"]
            for config in configs:
                for tol in tolerances:
                    result = run_one(problem, stepper, method, config, tol)
                    results.append(result)
                    print(json.dumps(asdict(result), sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "configurations": {
            "method_order": METHOD_ORDER,
            "estimator_order": ESTIMATOR_ORDER,
        },
        "tolerances": tolerances,
        "work_metric": "right-hand-side evaluations",
        "gmres_budget_note": (
            "gmres_budget is NKStats.lin_iters: the configured Krylov budget, "
            "not iterations spent; Newton iterations are reported separately."
        ),
        "results": [asdict(result) for result in results],
        "matched_error_nfe": matched_error_summary(results),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    plot_results(results, args.output.parent)


if __name__ == "__main__":
    main()
