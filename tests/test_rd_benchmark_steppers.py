"""Order and consistency tests for benchmarks/rd_steppers.py.

Each stepper is compared with an RK4 run at a very small step (an
independent method), on three reaction terms and a 32 x 32 periodic grid.
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import rd_steppers as rd  # noqa: E402

N = 32
DU, DV = 1e-3, 2e-3
T_END = 1.0


def _lap_eig():
    k = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    return -(kx**2 + ky**2)


def gray_scott(u, v):
    uvv = u * v * v
    return -uvv + 0.035 * (1 - u), uvv - (0.035 + 0.065) * v


def schnakenberg(u, v):
    u2v = u * u * v
    return 0.1 - u + u2v, 0.9 - u2v


def brusselator(u, v):
    u2v = u * u * v
    return 1.0 - 4.0 * u + u2v, 3.0 * u - u2v


REACTIONS = {
    "gray_scott": (gray_scott, 0.5, 0.25),
    "schnakenberg": (schnakenberg, 1.0, 0.9),
    "brusselator": (brusselator, 1.0, 3.0),
}


def _initial(u_mean, v_mean):
    x = np.arange(N) / N
    X, Y = np.meshgrid(x, x, indexing="ij")
    u0 = u_mean + 0.1 * np.cos(2 * np.pi * X) * np.sin(4 * np.pi * Y)
    v0 = v_mean + 0.1 * np.sin(2 * np.pi * (X + Y))
    return jnp.asarray(u0), jnp.asarray(v0)


def _run(step, u, v, n):
    def body(_, uv):
        return step(*uv)
    return jax.lax.fori_loop(0, n, body, (u, v))


@pytest.fixture(scope="module")
def references():
    lap = _lap_eig()
    out = {}
    for name, (reaction, um, vm) in REACTIONS.items():
        u0, v0 = _initial(um, vm)
        rk4 = rd.make_rk4_step(lap, DU, DV, reaction)
        dt = 5e-4
        n = int(round(T_END / dt))
        out[name] = _run(lambda u, v, rk4=rk4, dt=dt: rk4(u, v, dt), u0, v0, n)
    return out


def _error(u, v, ref):
    return float(jnp.sqrt(jnp.sum((u - ref[0]) ** 2 + (v - ref[1]) ** 2)
                          / jnp.sum(ref[0] ** 2 + ref[1] ** 2)))


def _orders(make_step, name, references, dts):
    reaction, um, vm = REACTIONS[name]
    u0, v0 = _initial(um, vm)
    errors = []
    for dt in dts:
        step = make_step(reaction, dt)
        u, v = _run(step, u0, v0, int(round(T_END / dt)))
        errors.append(_error(u, v, references[name]))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    return errors, orders


@pytest.mark.parametrize("name", list(REACTIONS))
def test_imex_strang_is_second_order(name, references):
    lap = _lap_eig()
    errors, orders = _orders(
        lambda r, dt: rd.make_imex_strang_step(lap, DU, DV, r, dt),
        name, references, [0.1, 0.05, 0.025])
    assert all(1.8 < p < 2.2 for p in orders), (errors, orders)


@pytest.mark.parametrize("name", list(REACTIONS))
def test_etdrk4_is_fourth_order(name, references):
    lap = _lap_eig()
    errors, orders = _orders(
        lambda r, dt: rd.make_etdrk4_step(lap, DU, DV, r, dt),
        name, references, [0.2, 0.1, 0.05])
    assert all(3.6 < p < 4.4 for p in orders), (errors, orders)


@pytest.mark.parametrize("name", list(REACTIONS))
def test_crank_nicolson_is_second_order_and_converges(name, references):
    lap = _lap_eig()
    reaction, um, vm = REACTIONS[name]
    u0, v0 = _initial(um, vm)
    errors = []
    for dt in [0.1, 0.05, 0.025]:
        solve = rd.make_cn_solve(rd.make_cn_step(lap, DU, DV, reaction, dt, newton_tol=1e-11))
        n = int(round(T_END / dt))
        u, v, stats, done = solve(u0, v0, n)
        assert int(done) == n
        assert int(stats.failed_steps) == 0
        assert float(stats.max_residual) <= 1e-11
        assert int(stats.newton_iters) >= n
        errors.append(_error(u, v, references[name]))
    orders = [np.log2(errors[i] / errors[i + 1]) for i in range(2)]
    assert all(1.8 < p < 2.2 for p in orders), (errors, orders)


def test_etdrk4_weights_sum_to_phi1():
    """f1 + 4 f2 + f3 = dt * phi1(dt L), so a constant N is integrated exactly."""
    dt = 0.3
    L = np.array([0.0, -1e-8, -0.5, -40.0, -1e4])
    _, _, _, f1, f2, f3 = rd.etdrk4_coefficients(L, dt)
    z = dt * L
    phi1 = np.where(np.abs(z) < 1e-6, 1 + z / 2, np.expm1(z) / np.where(z == 0, 1, z))
    np.testing.assert_allclose(f1 + 4 * f2 + f3, dt * phi1, rtol=1e-10)


def _stiff_schnakenberg(n=64):
    k = np.fft.fftfreq(n, 1.0 / n) * 2 * np.pi
    kx, ky = np.meshgrid(k, k, indexing="ij")
    lap = -(kx**2 + ky**2)

    def reaction(u, v):
        u2v = u * u * v
        return 1000.0 * (0.1 - u + u2v), 1000.0 * (0.9 - u2v)

    rng = np.random.default_rng(42)
    u0 = jnp.asarray(1.0 + 0.01 * rng.standard_normal((n, n)))
    v0 = jnp.asarray(0.9 + 0.01 * rng.standard_normal((n, n)))
    return lap, reaction, u0, v0


def test_cn_newton_converges_for_stiff_schnakenberg_and_both_preconditioners_agree():
    """Schnakenberg, gamma = 1000, dt = 0.002 (the benchmark parameters on a 64 x 64 grid)."""
    lap, reaction, u0, v0 = _stiff_schnakenberg()
    out = {}
    for pc in ("diffusion", "mean_field"):
        solve = rd.make_cn_solve(rd.make_cn_step(lap, 1.0, 40.0, reaction, 0.002, preconditioner=pc))
        u, v, stats, done = solve(u0, v0, 50)
        assert int(done) == 50 and int(stats.failed_steps) == 0
        assert float(stats.max_residual) <= 1e-8
        # The system amplifies a difference by exp(409 t), so compare the
        # two solutions after 5 steps only.
        u5, v5, _, _ = solve(u0, v0, 5)
        out[pc] = (np.asarray(u5), np.asarray(v5), int(stats.krylov_iters))
    np.testing.assert_allclose(out["mean_field"][0], out["diffusion"][0], atol=1e-5)
    np.testing.assert_allclose(out["mean_field"][1], out["diffusion"][1], atol=1e-5)
    assert out["mean_field"][2] < out["diffusion"][2]


def test_cn_solve_stops_and_reports_a_step_that_does_not_converge():
    """At dt = 0.01 Newton does not reach the tolerance; the solve must say so."""
    lap, reaction, u0, v0 = _stiff_schnakenberg()
    solve = rd.make_cn_solve(rd.make_cn_step(lap, 1.0, 40.0, reaction, 0.01))
    u, v, stats, done = solve(u0, v0, 50)
    assert int(done) < 50
    assert int(stats.failed_steps) == 1
    assert float(stats.max_residual) > 1e-8


def test_combined_imex_solve_equals_the_single_steps():
    lap = _lap_eig()
    reaction, um, vm = REACTIONS["brusselator"]
    u0, v0 = _initial(um, vm)
    dt, n = 0.05, 20
    u1, v1 = _run(rd.make_imex_strang_step(lap, DU, DV, reaction, dt), u0, v0, n)
    u2, v2 = rd.make_imex_strang_solve(lap, DU, DV, reaction, dt)(u0, v0, n)
    np.testing.assert_allclose(np.asarray(u2), np.asarray(u1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(v2), np.asarray(v1), rtol=0, atol=1e-12)
    u3, v3 = rd.make_imex_strang_solve(lap, DU, DV, reaction, dt)(u0, v0, 1)
    u4, v4 = rd.make_imex_strang_step(lap, DU, DV, reaction, dt)(u0, v0)
    np.testing.assert_allclose(np.asarray(u3), np.asarray(u4), rtol=0, atol=1e-13)


# ---------------------------------------------------------------------------
# The moljax solvers against the independent steppers of this module
# ---------------------------------------------------------------------------
def _library_cases():
    from moljax.core.model import (
        create_brusselator_periodic_fft,
        create_gray_scott_periodic_fft,
        create_schnakenberg_periodic_fft,
    )
    return {
        "gray_scott": (
            lambda g: create_gray_scott_periodic_fft(g, Du=2e-5, Dv=1e-5, F=0.035, k=0.065),
            gray_scott, 0.5, 0.25, 0.5, 20),
        "schnakenberg": (
            lambda g: create_schnakenberg_periodic_fft(g, Du=1e-3, Dv=2e-3, a=0.1, b=0.9, gamma=1.0),
            schnakenberg, 1.0, 0.9, 0.05, 20),
        "brusselator": (
            lambda g: create_brusselator_periodic_fft(g, Du=1e-3, Dv=2e-3, a=1.0, b=3.0),
            brusselator, 1.0, 3.0, 0.05, 20),
    }


@pytest.mark.parametrize("name", ["gray_scott", "schnakenberg", "brusselator"])
def test_moljax_solvers_match_the_independent_steppers(name):
    """IMEX-Strang, ETDRK4 and RK4 of the library agree with rd_steppers to rounding."""
    import rd_library_benchmark as lb

    from moljax.core.fft_integrators import etd_integrate
    from moljax.core.fft_operators import DiffusionOperator
    from moljax.core.fft_solvers import extract_interior
    from moljax.core.grid import Grid2D
    from moljax.core.stepping import IntegratorType, integrate_fixed_dt, integrate_imex_fixed_dt

    make_model, reaction, um, vm, dt, n = _library_cases()[name]
    grid = Grid2D.uniform(N, N, 0.0, 1.0, 0.0, 1.0, n_ghost=1)
    model, cache, diff = make_model(grid)
    Du, Dv = diff["u"], diff["v"]
    lap = np.asarray(cache.laplacian_symbol)
    u0, v0 = _initial(um, vm)
    y0 = lb.padded_state(model, grid, u0, v0)
    t_end = n * dt

    def interior(y):
        return np.asarray(extract_interior(y["u"], grid)), np.asarray(extract_interior(y["v"], grid))

    def close(mine, theirs, tol):
        np.testing.assert_allclose(np.asarray(mine[0]), theirs[0], rtol=0, atol=tol)
        np.testing.assert_allclose(np.asarray(mine[1]), theirs[1], rtol=0, atol=tol)

    _, _, y = integrate_imex_fixed_dt(model, y0, 0.0, t_end, dt, cache, diff, use_strang=True, save_every=n)
    close(rd.make_imex_strang_solve(lap, Du, Dv, reaction, dt)(u0, v0, n), interior(y), 1e-12)

    ops = {"u": DiffusionOperator(grid, Du, use_rfft=False), "v": DiffusionOperator(grid, Dv, use_rfft=False)}
    hist = etd_integrate({"u": u0, "v": v0}, (0.0, t_end), dt, ops,
                         lambda s, t: dict(zip(("u", "v"), reaction(s["u"], s["v"]), strict=True)),
                         method="etdrk4", save_every=n)[1]
    close(_run(rd.make_etdrk4_step(lap, Du, Dv, reaction, dt), u0, v0, n), (hist[-1]["u"], hist[-1]["v"]), 1e-11)

    _, _, y = integrate_fixed_dt(model, y0, 0.0, t_end, dt, method=IntegratorType.RK4, save_every=n)
    rk4 = rd.make_rk4_step(lap, Du, Dv, reaction)
    close(_run(lambda u, v: rk4(u, v, dt), u0, v0, n), interior(y), 1e-12)


@pytest.mark.parametrize("n_steps,save_every", [(1, 1), (7, 1), (12, 4), (11, 4), (9, 9)])
def test_fixed_imex_strang_driver_equals_repeated_single_steps(n_steps, save_every):
    """The driver combines adjacent half diffusion steps; the result must not change."""
    import rd_library_benchmark as lb

    from moljax.core.fft_solvers import extract_interior
    from moljax.core.grid import Grid2D
    from moljax.core.stepping import imex_strang_step, integrate_imex_fixed_dt

    make_model, _, um, vm, dt, _ = _library_cases()["brusselator"]
    grid = Grid2D.uniform(N, N, 0.0, 1.0, 0.0, 1.0, n_ghost=1)
    model, cache, diff = make_model(grid)
    u0, v0 = _initial(um, vm)
    y0 = lb.padded_state(model, grid, u0, v0)

    t_hist, y_hist, y_final = integrate_imex_fixed_dt(
        model, y0, 0.0, n_steps * dt, dt, cache, diff, use_strang=True, save_every=save_every)

    y, saved = y0, []
    for i in range(n_steps):
        y = imex_strang_step(model, y, i * dt, dt, cache, diff)
        if (i + 1) % save_every == 0:
            saved.append(y)
    assert y_hist["u"].shape[0] == len(saved) == n_steps // save_every
    for name in ("u", "v"):
        np.testing.assert_allclose(np.asarray(y_final[name]), np.asarray(y[name]), rtol=0, atol=1e-12)
        for j, ref in enumerate(saved):
            np.testing.assert_allclose(np.asarray(y_hist[name][j]),
                                       np.asarray(extract_interior(ref[name], grid)), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(t_hist), dt * save_every * np.arange(1, len(saved) + 1), rtol=1e-12)


@pytest.mark.parametrize("name", ["gray_scott", "schnakenberg", "brusselator"])
def test_imex_direct_explicit_path_equals_the_subtraction_path(name):
    """A *_periodic_fft model takes its nonlinear part as the explicit part."""
    import rd_library_benchmark as lb

    from moljax.core.grid import Grid2D
    from moljax.core.model import MOLModel
    from moljax.core.stepping import _imex_explicit_rhs, imex_strang_step

    make_model, _, um, vm, dt, _ = _library_cases()[name]
    grid = Grid2D.uniform(N, N, 0.0, 1.0, 0.0, 1.0, n_ghost=1)
    model, cache, diff = make_model(grid)
    assert model.metadata["linear_ops_are_fft_diffusion"] is True
    plain_meta = {k: v for k, v in model.metadata.items() if k != "linear_ops_are_fft_diffusion"}
    plain = MOLModel(grid=model.grid, bc_spec=model.bc_spec, params=model.params,
                     linear_ops=model.linear_ops, nonlinear_ops=model.nonlinear_ops,
                     metadata=plain_meta)
    y0 = lb.padded_state(model, grid, *_initial(um, vm))
    sl = grid.interior_slice

    direct = _imex_explicit_rhs(model, y0, 0.0, cache, diff)
    subtracted = _imex_explicit_rhs(plain, y0, 0.0, cache, diff)
    y_direct = imex_strang_step(model, y0, 0.0, dt, cache, diff)
    y_subtracted = imex_strang_step(plain, y0, 0.0, dt, cache, diff)
    for f in ("u", "v"):
        np.testing.assert_allclose(np.asarray(direct[f][sl]), np.asarray(subtracted[f][sl]),
                                   rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.asarray(y_direct[f][sl]), np.asarray(y_subtracted[f][sl]),
                                   rtol=0, atol=1e-11)

    # Other diffusivities than those of the model: the direct path must not be used.
    other = {"u": diff["u"]}
    part = _imex_explicit_rhs(model, y0, 0.0, cache, other)
    part_plain = _imex_explicit_rhs(plain, y0, 0.0, cache, other)
    np.testing.assert_allclose(np.asarray(part["v"][sl]), np.asarray(part_plain["v"][sl]), rtol=0, atol=0)
    assert float(jnp.max(jnp.abs(part["v"][sl] - direct["v"][sl]))) > 1e-6
