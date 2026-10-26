#!/usr/bin/env python3
"""Dense ablation of the DST/tau blend: which effect does the GMRES work.

This script is the committed form of the private study "A9".  It builds every
preconditioner as a dense matrix from the eigendecomposition of the Dirichlet
second difference ``L``.  It does not use the JAX implementation for the
operators.  It uses moljax only for the state, the coefficient and the
residual, and it compares the dense blend with the repository blend
(``moljax.experimental.pme_dst_tau_blend``) to round-off.

Model.  The Newton matrix of one backward-Euler step is
``J = I - dt L diag(D)``, with ``D = porous_medium_diffusivity(u, m)``.  Each
preconditioner has the input-weighted form

    M = diag(w0) + sum_k H(d_k) diag(w_k),    H(d) = (I - dt d L)^-1.

The identity weight ``w0`` is one on the masked ("inactive") inputs.  The
analysis operator is ``A = M J`` and the right-hand side is ``M b`` with
``b = -R(u)``.

Mask rule.  Every CFL mask in this script uses the local diffusive CFL number

    chi_j = 4 dt D_j / h^2,    node j is masked when chi_j <= eta.

The private study A9 used ``dt D_j lambda_max <= 0.1`` with
``lambda_max = (4 / h^2) cos^2(pi / (2 (N + 1)))``, the largest eigenvalue of
``-L``.  The two rules differ by the factor ``cos^2(pi / (2 (N + 1)))``, which
is ``1 - 9.4e-6`` at N = 512.  The output records this change.  The
repository blend keeps its own rule ``D_j <= max(1e-14, 1e-6 max(D))``; the
case names that start with ``repo_`` use that rule.

Block form.  Put the masked nodes first.  Then

    A = [[I + E, X], [F, Y]].

When D is zero on the masked nodes, each masked column of J is an identity
column, thus E = 0 and F = 0, and the spectrum of A is {1}^n0 together with
the spectrum of Y.  The output gives ||X||_2, ||F||_2 and ||E||_2 for each
case.

GMRES.  Unrestarted GMRES on ``A x = M b`` from ``x = 0``, with modified
Gram-Schmidt applied two times.  The output gives two iteration counts:

    (a) "preconditioned": the first k with ||M (b - J x_k)|| <= tol ||M b||,
    (b) "true":           the first k with ||b - J x_k|| <= tol ||b||.

Both counts come from the same Krylov sequence.  Count (a) is the convention
of the repository benchmarks.

Usage (a full run takes some minutes for each state at N = 1024)::

    JAX_PLATFORMS=cpu python benchmarks/tau_blend_mask_analysis.py
    JAX_PLATFORMS=cpu python benchmarks/tau_blend_mask_analysis.py --nx 64 --m 4 \
        --output /tmp/check.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular

from benchmarks.pme_breakdown import _initial_state
from benchmarks.tau_blend_reproducibility import provenance_revisions, state_identity
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import (
    porous_medium_diffusivity,
    regularized_porous_medium_potential,
)
from moljax.experimental.pme_conditioning import make_backward_euler_residual
from moljax.experimental.pme_dst_tau_blend import pme_dst_tau_blend_preconditioner

SCHEMA = "tau_blend_mask_analysis_v1"
NEAR_ZERO = 0.1
REPOSITORY_ACTIVE_RELATIVE = 1.0e-6
REPOSITORY_ACTIVE_ABSOLUTE = 1.0e-14


@dataclass(frozen=True)
class MaskAnalysisConfig:
    """The full configuration of the dense ablation."""

    nx: tuple[int, ...] = (512, 1024)
    m: tuple[int, ...] = (2, 4, 8)
    dt: float = 0.02
    epsilon: float = 1.0e-5
    t0: float = 0.1
    halfwidth: float = 3.0
    x_min: float = -4.0
    x_max: float = 4.0
    eta: float = 0.1
    frame_extra_eta: tuple[float, ...] = (1.0,)
    gmres_tol: float = 1.0e-8
    gmres_max: int = 1200
    probe_seed: int = 20261025
    output: str = "benchmarks/results/tau_blend_mask_analysis.json"
    mask_rule: str = "chi_j = 4 dt D_j / h^2 <= eta"
    mask_rule_change: str = (
        "The private study A9 masked node j when dt D_j lambda_max <= 0.1, with "
        "lambda_max = (4/h^2) cos^2(pi/(2(N+1))). This script masks node j when "
        "4 dt D_j / h^2 <= eta. The two rules differ by the factor "
        "cos^2(pi/(2(N+1))) in the cutoff."
    )
    reference_counts: tuple[int, ...] = field(default=(2, 5, 9))


# ---------------------------------------------------------------------------
# State and dense operators
# ---------------------------------------------------------------------------


def second_difference(nx: int, dx: float) -> np.ndarray:
    """Return the dense Dirichlet second difference on nx interior nodes."""
    main = -2.0 * np.ones(nx)
    off = np.ones(nx - 1)
    return (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / dx**2


def build_state(nx: int, m: int, dt: float, config: Any) -> dict[str, Any]:
    """Return the wide Barenblatt state, its coefficient and its dense operators.

    The state is the initial state of
    ``benchmarks/pme_dst_tau_blend_spectral_analysis.py``.
    """
    grid = NodeCenteredDirichletGrid.uniform(nx, config.x_min, config.x_max)
    u = np.asarray(_initial_state(grid, m, config.t0, config.halfwidth), dtype=np.float64)
    return state_from_values(u, grid, m, dt, config.epsilon)


def state_from_values(
    u: np.ndarray,
    grid: NodeCenteredDirichletGrid,
    m: int,
    dt: float,
    epsilon: float,
) -> dict[str, Any]:
    """Return the dense model of one Newton system at the state u."""
    nx = grid.nx
    dx = float(grid.dx)
    D = np.asarray(porous_medium_diffusivity(jnp.asarray(u), float(m), epsilon=epsilon))
    L = second_difference(nx, dx)
    lam, Q = np.linalg.eigh(L)
    residual = make_backward_euler_residual(jnp.asarray(u), grid, float(m), dt, epsilon)
    b = -np.asarray(residual(jnp.asarray(u)), dtype=np.float64)
    phi = np.asarray(
        regularized_porous_medium_potential(jnp.asarray(u), float(m), epsilon=epsilon)
    )
    b_dense = dt * (L @ phi)
    return {
        "grid": grid,
        "nx": nx,
        "m": m,
        "dt": dt,
        "epsilon": epsilon,
        "dx": dx,
        "u": u,
        "D": D,
        "L": L,
        "lam": lam,
        "Q": Q,
        "J": jacobian(L, D, dt),
        "b": b,
        "rhs_dense_relative_difference": float(
            np.linalg.norm(b_dense - b) / max(np.linalg.norm(b), np.finfo(float).tiny)
        ),
    }


def jacobian(L: np.ndarray, D: np.ndarray, dt: float) -> np.ndarray:
    """Return J = I - dt L diag(D)."""
    return np.eye(L.shape[0]) - dt * L * D[None, :]


def helmholtz(state: dict[str, Any], d: float, dt: float | None = None) -> np.ndarray:
    """Return the dense inverse H(d) = (I - dt d L)^-1 from the DST-I modes."""
    step = state["dt"] if dt is None else dt
    Q, lam = state["Q"], state["lam"]
    return (Q / (1.0 - step * d * lam)) @ Q.T


def local_cfl(state: dict[str, Any], D: np.ndarray, dt: float | None = None) -> np.ndarray:
    """Return chi_j = 4 dt D_j / h^2."""
    step = state["dt"] if dt is None else dt
    return 4.0 * step * D / state["dx"] ** 2


def floor_cfl(m: int, epsilon: float, dt: float, dx: float) -> float:
    """Return the CFL number of the coefficient floor, 4 dt m eps^(m-1) / h^2."""
    return 4.0 * dt * m * epsilon ** (m - 1) / dx**2


def cfl_mask(state: dict[str, Any], D: np.ndarray, eta: float, dt: float | None = None) -> np.ndarray:
    """Return the CFL mask (True on masked nodes)."""
    return local_cfl(state, D, dt) <= eta


def repository_mask(D: np.ndarray) -> np.ndarray:
    """Return the inactive set of the repository blend (True on masked nodes)."""
    nonnegative = np.maximum(D, 0.0)
    threshold = max(REPOSITORY_ACTIVE_ABSOLUTE, REPOSITORY_ACTIVE_RELATIVE * float(nonnegative.max()))
    return ~(nonnegative > threshold)


def quantile_references(D: np.ndarray, masked: np.ndarray, count: int) -> np.ndarray:
    """Return the repository quantile references of the unmasked coefficients."""
    return np.quantile(D[~masked], np.linspace(0.0, 1.0, count))


def gaussian_log_weights(D: np.ndarray, refs: np.ndarray, weighted: np.ndarray) -> np.ndarray:
    """Return the repository Gaussian log-coefficient weights, shape (refs, n).

    Inputs where ``weighted`` is False get zero weight.
    """
    refs = np.asarray(refs, dtype=np.float64)
    span = float(np.ptp(np.log(refs)))
    if refs.size == 1 or span == 0.0:
        W = np.zeros((D.size, refs.size))
        W[:, 0] = 1.0
    else:
        logs = np.log(np.maximum(D, np.finfo(float).tiny))[:, None]
        width = span / (refs.size - 1)
        exponent = -(((logs - np.log(refs)[None, :]) / width) ** 2)
        exponent = exponent - exponent.max(axis=1, keepdims=True)
        W = np.exp(exponent)
        W = W / W.sum(axis=1, keepdims=True)
    return np.where(weighted[:, None], W, 0.0).T


def hard_weights(D: np.ndarray, refs: np.ndarray, weighted: np.ndarray) -> np.ndarray:
    """Return nearest-reference weights in log D, shape (refs, n)."""
    logs = np.log(np.maximum(D, np.finfo(float).tiny))[:, None]
    nearest = np.argmin(np.abs(logs - np.log(refs)[None, :]), axis=1)
    return np.stack([np.where(weighted & (nearest == k), 1.0, 0.0) for k in range(len(refs))])


def blend_matrix(
    state: dict[str, Any],
    identity_inputs: np.ndarray,
    refs: np.ndarray,
    weights: np.ndarray,
    dt: float | None = None,
) -> np.ndarray:
    """Return M = diag(w0) + sum_k H(d_k) diag(w_k)."""
    M = np.diag(identity_inputs.astype(np.float64))
    for d, w in zip(refs, weights, strict=True):
        M = M + helmholtz(state, float(d), dt) * w[None, :]
    return M


# ---------------------------------------------------------------------------
# GMRES with two residual conventions
# ---------------------------------------------------------------------------


def gmres(
    A: np.ndarray,
    rhs: np.ndarray,
    tol: float,
    maxit: int,
    *,
    secondary: tuple[np.ndarray, np.ndarray] | None = None,
    stop: str = "primary",
    checkpoints: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Run unrestarted GMRES on ``A x = rhs`` from ``x = 0``.

    The "primary" residual is ``||rhs - A x|| / ||rhs||``; GMRES minimizes it.
    The optional ``secondary = (B, c)`` gives a second residual
    ``||c - B x|| / ||c||``.  For left preconditioning, use ``A = M J``,
    ``rhs = M b`` and ``secondary = (J, b)``: then the primary residual is the
    preconditioned residual and the secondary residual is the true residual.

    ``stop`` is "primary", "secondary" or "both".  The run stops when the
    selected residuals are at or below ``tol``, at a breakdown, or after
    ``maxit`` iterations.  ``checkpoints`` gives tolerances: for each, the
    result records the first k with a primary residual at or below it, and the
    secondary residual at that k.
    """
    if stop not in ("primary", "secondary", "both"):
        raise ValueError("stop must be 'primary', 'secondary' or 'both'")
    if stop != "primary" and secondary is None:
        raise ValueError("this stop rule needs a secondary residual")
    n = A.shape[0]
    beta = float(np.linalg.norm(rhs))
    result: dict[str, Any] = {
        "iterations": 0,
        "iterations_primary": None,
        "iterations_secondary": None,
        "breakdown": False,
        "checkpoints": {},
    }
    if beta == 0.0:
        result.update(
            iterations_primary=0,
            iterations_secondary=0,
            x=np.zeros(n),
            primary_ratio=0.0,
            secondary_ratio=0.0,
        )
        return result
    if secondary is not None:
        B, c = secondary
        c_norm = float(np.linalg.norm(c))
        BV = np.zeros((n, maxit + 1))
    V = np.zeros((n, maxit + 1))
    H = np.zeros((maxit + 1, maxit))
    cs = np.zeros(maxit)
    sn = np.zeros(maxit)
    g = np.zeros(maxit + 1)
    g[0] = beta
    V[:, 0] = rhs / beta
    if secondary is not None:
        BV[:, 0] = B @ V[:, 0]
    pending = sorted(set(checkpoints), reverse=True)
    y = np.zeros(0)
    primary_ratio = 1.0
    secondary_ratio = 1.0 if secondary is None else float(np.linalg.norm(c)) / c_norm
    for k in range(maxit):
        w = A @ V[:, k]
        w_norm = float(np.linalg.norm(w))
        for _ in range(2):
            h = V[:, : k + 1].T @ w
            w = w - V[:, : k + 1] @ h
            H[: k + 1, k] += h
        H[k + 1, k] = float(np.linalg.norm(w))
        breakdown = H[k + 1, k] <= 64.0 * np.finfo(float).eps * max(w_norm, np.finfo(float).tiny)
        if not breakdown:
            V[:, k + 1] = w / H[k + 1, k]
            if secondary is not None:
                BV[:, k + 1] = B @ V[:, k + 1]
        for i in range(k):
            t = cs[i] * H[i, k] + sn[i] * H[i + 1, k]
            H[i + 1, k] = -sn[i] * H[i, k] + cs[i] * H[i + 1, k]
            H[i, k] = t
        r = math.hypot(H[k, k], H[k + 1, k])
        cs[k], sn[k] = (1.0, 0.0) if r == 0.0 else (H[k, k] / r, H[k + 1, k] / r)
        H[k, k] = r
        H[k + 1, k] = 0.0
        g[k + 1] = -sn[k] * g[k]
        g[k] = cs[k] * g[k]
        primary_ratio = abs(g[k + 1]) / beta
        iteration = k + 1
        need_y = secondary is not None or pending or breakdown or iteration == maxit
        if need_y or primary_ratio <= tol:
            y = solve_triangular(H[: k + 1, : k + 1], g[: k + 1])
        if secondary is not None:
            secondary_ratio = float(np.linalg.norm(c - BV[:, : k + 1] @ y)) / c_norm
        while pending and primary_ratio <= pending[0]:
            level = pending.pop(0)
            result["checkpoints"][repr(level)] = {
                "iterations": iteration,
                "primary_ratio": primary_ratio,
                "secondary_ratio": secondary_ratio if secondary is not None else None,
            }
        if result["iterations_primary"] is None and primary_ratio <= tol:
            result["iterations_primary"] = iteration
        if (
            secondary is not None
            and result["iterations_secondary"] is None
            and secondary_ratio <= tol
        ):
            result["iterations_secondary"] = iteration
        done = {
            "primary": result["iterations_primary"] is not None,
            "secondary": result["iterations_secondary"] is not None,
            "both": result["iterations_primary"] is not None
            and result["iterations_secondary"] is not None,
        }[stop]
        result["iterations"] = iteration
        if done or breakdown:
            result["breakdown"] = bool(breakdown)
            break
    k_done = result["iterations"]
    x = V[:, :k_done] @ y if y.size == k_done else V[:, :k_done] @ solve_triangular(
        H[:k_done, :k_done], g[:k_done]
    )
    result["x"] = x
    # Verified residuals of the returned iterate (not the Givens estimate).
    result["primary_ratio"] = float(np.linalg.norm(rhs - A @ x)) / beta
    result["primary_ratio_estimate"] = float(primary_ratio)
    if secondary is not None:
        result["secondary_ratio"] = float(np.linalg.norm(c - B @ x)) / c_norm
    return result


def gmres_two_conventions(
    M: np.ndarray,
    J: np.ndarray,
    b: np.ndarray,
    tol: float,
    maxit: int,
    checkpoints: tuple[float, ...] = (),
) -> dict[str, Any]:
    """Run left-preconditioned GMRES until both residual conventions reach tol."""
    out = gmres(
        M @ J,
        M @ b,
        tol,
        maxit,
        secondary=(J, b),
        stop="both",
        checkpoints=checkpoints,
    )
    return {
        "gmres_preconditioned": out["iterations_primary"],
        "gmres_true": out["iterations_secondary"],
        "gmres_iterations_run": out["iterations"],
        "gmres_breakdown": out["breakdown"],
        "final_preconditioned_ratio": out["primary_ratio"],
        "final_true_ratio": out["secondary_ratio"],
        "checkpoints": out["checkpoints"],
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


def _case(name: str, M: np.ndarray, identity_inputs: np.ndarray, refs: Any, **extra: Any) -> dict:
    return {
        "case": name,
        "M": M,
        "identity_inputs": np.asarray(identity_inputs, dtype=bool),
        "references": [float(d) for d in np.atleast_1d(refs)],
        **extra,
    }


def build_cases(state: dict[str, Any], config: MaskAnalysisConfig) -> list[dict[str, Any]]:
    """Return every ablation case of one state.

    Each control changes one thing against a named parent case:

    - ``scalar_*``: H(d) on all inputs; d from the CFL-active coefficients.
    - ``cfl_mask_one_inverse_*``: the CFL mask, then the same H(d).
    - ``repo_blend_B3``: the repository blend (its own mask and references).
    - ``repo_blend_B3_no_mask``: the same references and weights, no identity.
    - ``repo_blend_B3_hard``: nearest-reference weights instead of Gaussian.
    - ``repo_blend_B{2,5,9}``: another reference count.
    - ``repo_blend_B3_logspaced``: log-spaced references between the extremes.
    - ``cfl_mask_repo_refs_B3``: the CFL mask with the repository references.
    - ``cfl_blend_*``: the CFL mask with references from the CFL-active nodes.
    - ``cfl_refs_no_mask_B3``: the references of ``cfl_blend_B3``, no identity.
    - ``cfl_mask_frame``: the CFL mask, H(Dmax) and diag(Dmax / D) on the
      active inputs.
    """
    D = state["D"]
    n = state["nx"]
    eta = config.eta
    everywhere = np.ones(n, dtype=bool)
    nowhere = np.zeros(n, dtype=bool)
    q_cfl = cfl_mask(state, D, eta)
    q_repo = repository_mask(D)
    cases: list[dict[str, Any]] = []

    active = D[~q_cfl]
    scalars = {
        "max": float(active.max()),
        "median": float(np.median(active)),
        "geomean": float(np.exp(np.mean(np.log(active)))),
    }
    for label, d in scalars.items():
        cases.append(
            _case(f"scalar_{label}", helmholtz(state, d), nowhere, d, family="scalar", mask_rule="none")
        )
    for label, d in scalars.items():
        weights = (~q_cfl).astype(float)[None, :]
        cases.append(
            _case(
                f"cfl_mask_one_inverse_{label}",
                blend_matrix(state, q_cfl, [d], weights),
                q_cfl,
                d,
                family="cfl",
                mask_rule="cfl",
            )
        )

    for family, mask in (("repo", q_repo), ("cfl", q_cfl)):
        prefix = f"{family}_blend"
        refs3 = quantile_references(D, mask, 3)
        cases.append(
            _case(
                f"{prefix}_B3",
                blend_matrix(state, mask, refs3, gaussian_log_weights(D, refs3, ~mask)),
                mask,
                refs3,
                family=family,
                mask_rule=family,
            )
        )
        cases.append(
            _case(
                f"{prefix}_B3_hard",
                blend_matrix(state, mask, refs3, hard_weights(D, refs3, ~mask)),
                mask,
                refs3,
                family=family,
                mask_rule=family,
            )
        )
        for count in config.reference_counts:
            refs = quantile_references(D, mask, count)
            cases.append(
                _case(
                    f"{prefix}_B{count}",
                    blend_matrix(state, mask, refs, gaussian_log_weights(D, refs, ~mask)),
                    mask,
                    refs,
                    family=family,
                    mask_rule=family,
                )
            )
        unmasked = D[~mask]
        refs_log = np.exp(np.linspace(np.log(unmasked.min()), np.log(unmasked.max()), 3))
        cases.append(
            _case(
                f"{prefix}_B3_logspaced",
                blend_matrix(state, mask, refs_log, gaussian_log_weights(D, refs_log, ~mask)),
                mask,
                refs_log,
                family=family,
                mask_rule=family,
            )
        )

    refs_repo = quantile_references(D, q_repo, 3)
    cases.append(
        _case(
            "repo_blend_B3_no_mask",
            blend_matrix(state, nowhere, refs_repo, gaussian_log_weights(D, refs_repo, everywhere)),
            nowhere,
            refs_repo,
            family="repo",
            mask_rule="none",
        )
    )
    cases.append(
        _case(
            "cfl_mask_repo_refs_B3",
            blend_matrix(state, q_cfl, refs_repo, gaussian_log_weights(D, refs_repo, ~q_cfl)),
            q_cfl,
            refs_repo,
            family="cfl",
            mask_rule="cfl",
        )
    )
    refs_cfl = quantile_references(D, q_cfl, 3)
    cases.append(
        _case(
            "cfl_refs_no_mask_B3",
            blend_matrix(state, nowhere, refs_cfl, gaussian_log_weights(D, refs_cfl, everywhere)),
            nowhere,
            refs_cfl,
            family="cfl",
            mask_rule="none",
        )
    )

    for frame_eta in (eta, *config.frame_extra_eta):
        mask = cfl_mask(state, D, frame_eta)
        name = "cfl_mask_frame" if frame_eta == eta else f"cfl_mask_frame_eta{frame_eta:g}"
        cases.append(
            _case(
                name,
                frame_matrix(state, D, mask),
                mask,
                float(D[~mask].max()) if np.any(~mask) else [],
                family="cfl",
                mask_rule="cfl",
                eta=frame_eta,
            )
        )
    for case in cases:
        case.setdefault("eta", eta if case["mask_rule"] == "cfl" else None)
    return cases


def frame_matrix(state: dict[str, Any], D: np.ndarray, masked: np.ndarray, dt: float | None = None) -> np.ndarray:
    """Return Q + H(Dmax) diag(Dmax / D) (I - Q), with Dmax over the active nodes."""
    if not np.any(~masked):
        return np.eye(state["nx"])
    dmax = float(D[~masked].max())
    gain = np.where(masked, 0.0, dmax / D)
    return np.diag(masked.astype(float)) + helmholtz(state, dmax, dt) * gain[None, :]


def one_inverse_matrix(state: dict[str, Any], D: np.ndarray, masked: np.ndarray, dt: float | None = None) -> np.ndarray:
    """Return Q + H(Dmax) (I - Q), with Dmax over the active nodes."""
    if not np.any(~masked):
        return np.eye(state["nx"])
    dmax = float(D[~masked].max())
    return np.diag(masked.astype(float)) + helmholtz(state, dmax, dt) * (~masked).astype(float)[None, :]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def block_norms(A: np.ndarray, identity_inputs: np.ndarray) -> dict[str, float | None]:
    """Return ||X||, ||F|| and ||E|| of A = [[I + E, X], [F, Y]] (masked first)."""
    q = identity_inputs
    n0 = int(np.count_nonzero(q))
    if n0 == 0 or n0 == q.size:
        return {"block_X_norm": None, "block_F_norm": None, "block_E_norm": None}
    return {
        "block_X_norm": float(np.linalg.norm(A[np.ix_(q, ~q)], 2)),
        "block_F_norm": float(np.linalg.norm(A[np.ix_(~q, q)], 2)),
        "block_E_norm": float(np.linalg.norm(A[np.ix_(q, q)] - np.eye(n0), 2)),
    }


def spectral_metrics(A: np.ndarray, identity_inputs: np.ndarray) -> dict[str, Any]:
    """Return the dense spectral metrics of A = M J."""
    eigenvalues = np.linalg.eigvals(A)
    singular = np.linalg.svd(A, compute_uv=False)
    hermitian_min = float(np.linalg.eigvalsh(0.5 * (A + A.T))[0])
    n0 = int(np.count_nonzero(identity_inputs))
    distance = np.sort(np.abs(eigenvalues - 1.0))
    return {
        "kappa2": float(singular[0] / singular[-1]),
        "eigenvalues_below_0p1": int(np.count_nonzero(np.abs(eigenvalues) < NEAR_ZERO)),
        "min_abs_eigenvalue": float(np.min(np.abs(eigenvalues))),
        "max_abs_eigenvalue": float(np.max(np.abs(eigenvalues))),
        "min_real_eigenvalue": float(np.min(eigenvalues.real)),
        "max_abs_imag_eigenvalue": float(np.max(np.abs(eigenvalues.imag))),
        "hermitian_part_min_eigenvalue": hermitian_min,
        "origin_outside_field_of_values": bool(hermitian_min > 0.0),
        # The n0 eigenvalues nearest to 1: the largest of their distances.
        "inactive_cluster_max_distance_from_1": float(distance[n0 - 1]) if n0 else None,
        **block_norms(A, identity_inputs),
    }


def repository_blend_check(
    state: dict[str, Any],
    case: dict[str, Any],
    reference_count: int,
    seed: int,
) -> dict[str, float]:
    """Compare a dense repository-style blend with the repository blend action."""
    pre = pme_dst_tau_blend_preconditioner(
        jnp.asarray(state["u"]),
        float(state["m"]),
        state["dt"],
        state["grid"],
        state["epsilon"],
        reference_count=reference_count,
    )
    probe = np.random.default_rng(seed).normal(size=state["nx"])
    source = np.asarray(pre.apply(jnp.asarray(probe)))
    dense = case["M"] @ probe
    refs_repo = np.asarray(pre.reference_values)
    refs_dense = np.asarray(case["references"])
    return {
        "action_relative_error": float(np.linalg.norm(dense - source) / np.linalg.norm(source)),
        "reference_max_relative_difference": float(
            np.max(np.abs(refs_repo - refs_dense) / np.abs(refs_repo))
        ),
        "inactive_count_repository": int(np.count_nonzero(np.asarray(pre.inactive_weight))),
    }


# ---------------------------------------------------------------------------
# Provenance and output
# ---------------------------------------------------------------------------


def provenance(config: Any) -> dict[str, Any]:
    """Return the provenance block of an output file."""
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPOSITORY,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return {
        **provenance_revisions(),
        "working_tree_clean": status.strip() == "",
        "command_line": [sys.executable, *sys.argv],
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "jax": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "config": asdict(config),
    }


def _plain(value: Any) -> Any:
    """Convert numpy scalars and arrays for JSON output."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    return value


def write_outputs(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """Write the JSON payload and a flat CSV of the rows beside it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(payload), indent=1) + "\n")
    flat = []
    for row in rows:
        flat.append(
            {
                k: (";".join(repr(float(x)) for x in v) if isinstance(v, list) else v)
                for k, v in row.items()
                if not isinstance(v, dict)
            }
        )
    keys: list[str] = []
    for row in flat:
        keys.extend(k for k in row if k not in keys)
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def analyze_state(nx: int, m: int, config: MaskAnalysisConfig) -> tuple[dict[str, Any], list[dict]]:
    """Return the state record and the case rows of one (N, m) state."""
    state = build_state(nx, m, config.dt, config)
    D = state["D"]
    q_cfl = cfl_mask(state, D, config.eta)
    q_repo = repository_mask(D)
    lambda_max = float(np.max(-state["lam"]))
    q_a9 = state["dt"] * D * lambda_max <= 0.1
    # Nodes where D equals the coefficient floor m eps^(m-1) to round-off.
    floor_nodes = D <= m * config.epsilon ** (m - 1) * (1.0 + 1.0e-9)
    record = {
        "nx": nx,
        "m": m,
        "dt": config.dt,
        "h": state["dx"],
        "state_identity": state_identity(jnp.asarray(state["u"])),
        "D_max": float(D.max()),
        "D_min": float(D.min()),
        "global_cfl": float(local_cfl(state, D).max()),
        "floor_cfl": floor_cfl(m, config.epsilon, config.dt, state["dx"]),
        "cfl_masked_count": int(np.count_nonzero(q_cfl)),
        "floor_node_count": int(np.count_nonzero(floor_nodes)),
        "cfl_mask_equals_floor_nodes": bool(np.array_equal(q_cfl, floor_nodes)),
        "repository_masked_count": int(np.count_nonzero(q_repo)),
        "a9_rule_masked_count": int(np.count_nonzero(q_a9)),
        "a9_rule_and_cfl_rule_agree": bool(np.array_equal(q_a9, q_cfl)) if config.eta == 0.1 else None,
        "rhs_dense_relative_difference": state["rhs_dense_relative_difference"],
        "repository_checks": {},
    }
    rows = []
    for case in build_cases(state, config):
        A = case["M"] @ state["J"]
        metrics = spectral_metrics(A, case["identity_inputs"])
        counts = gmres_two_conventions(case["M"], state["J"], state["b"], config.gmres_tol, config.gmres_max)
        counts.pop("checkpoints")
        row = {
            "nx": nx,
            "m": m,
            "dt": config.dt,
            "case": case["case"],
            "family": case["family"],
            "mask_rule": case["mask_rule"],
            "eta": case["eta"],
            "identity_input_count": int(np.count_nonzero(case["identity_inputs"])),
            "references": case["references"],
            **metrics,
            **counts,
        }
        if case["case"].startswith("repo_blend_B") and case["case"][12:].isdigit():
            count = len(case["references"])
            check = repository_blend_check(state, case, count, config.probe_seed)
            record["repository_checks"][case["case"]] = check
            row["repository_action_relative_error"] = check["action_relative_error"]
        rows.append(row)
        print(
            f"N={nx:5d} m={m} {case['case']:30s} n0={row['identity_input_count']:5d} "
            f"kappa={row['kappa2']:10.4g} nz={row['eigenvalues_below_0p1']:4d} "
            f"hmin={row['hermitian_part_min_eigenvalue']:+.3g} "
            f"gmres(prec/true)={row['gmres_preconditioned']}/{row['gmres_true']}",
            flush=True,
        )
    return record, rows


def main(argv: list[str] | None = None) -> None:
    """Run the dense ablation and write the JSON and CSV results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--nx", type=int, nargs="+", default=list(MaskAnalysisConfig.nx))
    parser.add_argument("--m", type=int, nargs="+", default=list(MaskAnalysisConfig.m))
    parser.add_argument("--eta", type=float, default=MaskAnalysisConfig.eta)
    parser.add_argument("--gmres-max", type=int, default=MaskAnalysisConfig.gmres_max)
    parser.add_argument("--output", default=MaskAnalysisConfig.output)
    args = parser.parse_args(argv)
    config = MaskAnalysisConfig(
        nx=tuple(args.nx),
        m=tuple(args.m),
        eta=args.eta,
        gmres_max=args.gmres_max,
        output=args.output,
    )
    states, rows = [], []
    for nx in config.nx:
        for m in config.m:
            record, state_rows = analyze_state(nx, m, config)
            states.append(record)
            rows.extend(state_rows)
    output = Path(config.output)
    if not output.is_absolute():
        output = REPOSITORY / output
    payload = {
        "schema": SCHEMA,
        "provenance": provenance(config),
        "residual_conventions": {
            "gmres_preconditioned": "first k with ||M(b - J x_k)|| <= tol ||M b|| (left preconditioning)",
            "gmres_true": "first k with ||b - J x_k|| <= tol ||b||, same Krylov sequence",
        },
        "states": states,
        "cases": rows,
    }
    write_outputs(output, payload, rows)
    print(f"wrote {output} and {output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
