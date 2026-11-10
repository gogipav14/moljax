#!/usr/bin/env python3
"""Reference-count sensitivity of the CFL-mask tau blend."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import scipy.linalg

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from benchmarks.tau_blend_benchmark_support import benchmark_provenance, write_json
from benchmarks.tau_blend_mask_analysis import (
    blend_matrix,
    build_state,
    cfl_mask,
    gaussian_log_weights,
    quantile_references,
)
from moljax.experimental.pseudospectral_criterion import (
    assess_dense_pseudospectral_criterion,
)
from moljax.experimental.tau_blend_active_block import (
    extract_active_block,
    true_residual_gmres,
)


def _state(nx: int, m: int) -> dict[str, Any]:
    config = SimpleNamespace(
        x_min=-4.0,
        x_max=4.0,
        t0=0.1,
        halfwidth=3.0,
        epsilon=1.0e-5,
    )
    return build_state(nx, m, 0.02, config)


def run_case(nx: int, m: int, reference_count: int) -> dict[str, Any]:
    """Return spectrum, singular value, and true-residual work for one case."""
    state = _state(nx, m)
    mask = cfl_mask(state, state["D"], 0.1)
    references = quantile_references(state["D"], mask, reference_count)
    weights = gaussian_log_weights(state["D"], references, ~mask)
    preconditioner = blend_matrix(state, mask, references, weights)
    blocks = extract_active_block(preconditioner @ state["J"], mask)
    eigenvalues = scipy.linalg.eigvals(blocks.Y, check_finite=False)
    singular = scipy.linalg.svdvals(blocks.Y, check_finite=False)
    gmres = true_residual_gmres(
        preconditioner,
        state["J"],
        state["b"],
        tolerance=1.0e-8,
    )
    criterion = None
    if gmres["iterations"] < 25:
        assessment = assess_dense_pseudospectral_criterion(
            blocks.Y,
            polynomial_degrees=tuple(range(4, 41, 4)),
            target_tolerance=1.0e-8,
        )
        criterion = asdict(assessment)
    return {
        "nx": nx,
        "m": m,
        "reference_count": reference_count,
        "reference_rule": "active-set diffusivity quantiles",
        "references": [float(value) for value in references],
        "active_count": int((~mask).sum()),
        "minimum_real": float(eigenvalues.real.min()),
        "maximum_real": float(eigenvalues.real.max()),
        "maximum_abs_imag": float(np.abs(eigenvalues.imag).max()),
        "sigma_min": float(singular[-1]),
        "gmres": {
            "iterations": gmres["iterations"],
            "true_stop_count": gmres["true_stop_count"],
            "preconditioned_stop_count": gmres["preconditioned_stop_count"],
            "true_relative_residual": gmres["secondary_ratio"],
            "preconditioned_relative_residual": gmres["primary_ratio"],
            "converged": gmres["converged"],
        },
        "Y_criterion": criterion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--m", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--reference-count", type=int, nargs="+", default=[3, 4, 5, 6])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/tau_blend_reference_count.json"),
    )
    args = parser.parse_args()
    started = perf_counter()
    rows = [
        run_case(nx, m, count) for nx in args.nx for m in args.m for count in args.reference_count
    ]
    payload = {
        "schema": "tau_blend_reference_count_v1",
        "configuration": {
            "nx": args.nx,
            "m": args.m,
            "reference_count": args.reference_count,
            "eta": 0.1,
            "true_residual_tolerance": 1.0e-8,
            "certificate_trigger": "measured GMRES iterations < 25",
        },
        "rows": rows,
        "provenance": benchmark_provenance(__file__, perf_counter() - started),
    }
    write_json(args.output, payload)


if __name__ == "__main__":
    main()
