"""Check preconditioned and physical residual stops on the CPU replication cells."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks import brusselator_dt_sweep as sweep
from benchmarks import brusselator_dt_sweep_cpu_replication as replication
from moljax.experimental.brusselator_conditioning import build_brusselator_system

jax.config.update("jax_enable_x64", True)

SCHEMA = "brusselator_dt_sweep_true_residual_check_v1"
PROTOCOL = "benchmarks/BRUSSELATOR_DT_SWEEP_TRUE_RESIDUAL_CHECK.md"
CPU_RESULT = "benchmarks/results/brusselator_dt_sweep_cpu_replication.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate(
    rhs: jax.Array,
    basis: list[jax.Array],
    triangular: list[list[float]],
    rotated_rhs: list[float],
) -> jax.Array:
    columns = len(triangular)
    weights = [0.0] * columns
    for row in reversed(range(columns)):
        total = rotated_rhs[row]
        for column in range(row + 1, columns):
            total -= triangular[column][row] * weights[column]
        weights[row] = total / triangular[row][row]
    result = jnp.zeros_like(rhs)
    for weight, vector in zip(weights, basis[:columns], strict=True):
        result = result + weight * vector
    return result


def _dual_stop_gmres(
    preconditioned_matvec: Callable[[jax.Array], jax.Array],
    preconditioned_rhs: jax.Array,
    physical_matvec: Callable[[jax.Array], jax.Array],
    physical_rhs: jax.Array,
) -> dict[str, Any]:
    """Measure both residual conventions from one unrestarted Krylov sequence."""
    rhs = jnp.asarray(preconditioned_rhs, dtype=jnp.float64)
    physical_rhs = jnp.asarray(physical_rhs, dtype=jnp.float64)
    rhs_norm = float(jnp.linalg.norm(rhs))
    physical_norm = float(jnp.linalg.norm(physical_rhs))
    basis = [rhs / rhs_norm]
    cosines: list[float] = []
    sines: list[float] = []
    triangular: list[list[float]] = []
    rotated_rhs = [rhs_norm] + [0.0] * sweep.GMRES_MAX_ITERS
    breakdown_factor = 64.0 * float(jnp.finfo(jnp.float64).eps)
    preconditioned_stop: int | None = None
    true_stop: int | None = None
    true_at_preconditioned: float | None = None
    preconditioned_at_stop: float | None = None

    for column in range(sweep.GMRES_MAX_ITERS):
        image = jnp.real(preconditioned_matvec(basis[column]))
        column_scale = float(jnp.linalg.norm(image))
        coefficients: list[float] = []
        remainder = image
        for vector in basis:
            coefficient = float(jnp.vdot(vector, remainder))
            coefficients.append(coefficient)
            remainder = remainder - coefficient * vector
        for row, vector in enumerate(basis):
            correction = float(jnp.vdot(vector, remainder))
            coefficients[row] += correction
            remainder = remainder - correction * vector
        subdiagonal = float(jnp.linalg.norm(remainder))
        invariant = subdiagonal <= breakdown_factor * column_scale
        h_column = coefficients + [subdiagonal]
        for row, (cosine, sine) in enumerate(zip(cosines, sines, strict=True)):
            upper = cosine * h_column[row] + sine * h_column[row + 1]
            h_column[row + 1] = -sine * h_column[row] + cosine * h_column[row + 1]
            h_column[row] = upper
        diagonal = h_column[column]
        normalization = math.hypot(diagonal, subdiagonal)
        if normalization == 0.0:
            break
        cosine, sine = diagonal / normalization, subdiagonal / normalization
        cosines.append(cosine)
        sines.append(sine)
        h_column[column] = normalization
        h_column[column + 1] = 0.0
        triangular.append(h_column[: column + 1])
        previous = rotated_rhs[column]
        rotated_rhs[column] = cosine * previous
        rotated_rhs[column + 1] = -sine * previous

        candidate = _candidate(rhs, basis, triangular, rotated_rhs)
        true_relative = float(
            jnp.linalg.norm(physical_rhs - jnp.real(physical_matvec(candidate))) / physical_norm
        )
        rotated_relative = abs(rotated_rhs[column + 1]) / rhs_norm
        iteration = column + 1
        if preconditioned_stop is None and rotated_relative <= sweep.GMRES_TOLERANCE:
            pre_relative = float(
                jnp.linalg.norm(rhs - jnp.real(preconditioned_matvec(candidate))) / rhs_norm
            )
            if pre_relative <= sweep.GMRES_TOLERANCE:
                preconditioned_stop = iteration
                preconditioned_at_stop = pre_relative
                true_at_preconditioned = true_relative
        if true_stop is None and true_relative <= sweep.GMRES_TOLERANCE:
            true_stop = iteration
        if preconditioned_stop is not None and true_stop is not None:
            break
        if invariant:
            break
        basis.append(remainder / subdiagonal)

    return {
        "preconditioned_stop_count": preconditioned_stop or sweep.GMRES_MAX_ITERS + 1,
        "preconditioned_converged": preconditioned_stop is not None,
        "true_stop_count": true_stop or sweep.GMRES_MAX_ITERS + 1,
        "true_converged": true_stop is not None,
        "preconditioned_relative_residual_at_preconditioned_stop": preconditioned_at_stop,
        "true_relative_residual_at_preconditioned_stop": true_at_preconditioned,
    }


def _condition_number(
    preconditioner: str,
    fft_cache: Any,
    diffusivities: dict[str, float],
    reaction_jacobian: jax.Array,
    dt: float,
) -> tuple[float | None, str]:
    if preconditioner == "identity":
        return 1.0, "identity"
    symbol = np.asarray(jax.device_get(fft_cache.laplacian_symbol), dtype=np.float64)
    if preconditioner == "fft_diffusion":
        values = np.concatenate(
            [np.abs(1.0 - dt * diffusivities[field] * symbol).ravel() for field in ("u", "v")]
        )
        return float(np.max(values) / np.min(values)), "exact Fourier diagonal"
    if preconditioner == "mean_field_coupled_fft":
        mean = np.asarray(jax.device_get(jnp.mean(reaction_jacobian, axis=(0, 1))))
        maximum = 0.0
        minimum = math.inf
        for value in symbol.ravel():
            block = np.eye(2) - dt * (
                np.diag([diffusivities["u"] * value, diffusivities["v"] * value]) + mean
            )
            singular = np.linalg.svd(block, compute_uv=False)
            maximum = max(maximum, float(singular[0]))
            minimum = min(minimum, float(singular[-1]))
        return maximum / minimum, "exact Fourier 2x2 blocks"
    return None, "unavailable: split composition is not block diagonal in one basis"


def _decision(cells: dict[str, dict[str, Any]], committed: dict[str, Any]) -> dict[str, Any]:
    proxy = json.loads(json.dumps(committed["cells"]))
    for key, row in cells.items():
        proxy[key]["cpu"]["gmres"] = {
            "iterations": row["true_stop_count"],
            "converged": row["true_converged"],
        }
    return replication._decision(proxy)


def run(repository: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    committed_path = repository / CPU_RESULT
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    protocol = repository / PROTOCOL
    started = time.time()
    cells: dict[str, dict[str, Any]] = {}
    if output.is_file():
        checkpoint = json.loads(output.read_text(encoding="utf-8"))
        if checkpoint.get("schema") != SCHEMA:
            raise RuntimeError("true-residual checkpoint schema mismatch")
        cells = checkpoint.get("cells", {})
    for spec in sweep.CPU_STATE_SPECS:
        record = sweep._find_record(repository, spec)
        state, source = sweep._load_regenerated_state(source_dir, spec, record)
        grid, regime = sweep._grid_and_regime(record)
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        reaction = sweep._reaction_jacobian(state, grid, regime.b)
        for arm, planned, dt, preconditioner in replication._cell_specs():
            if planned != spec:
                continue
            key = replication._cell_key(arm, spec, dt, preconditioner)
            if key in cells:
                continue
            linearization, _ = sweep._build_cell_linearization(
                state,
                model,
                fft_cache,
                diffusivities,
                reaction,
                dt,
                float(record["time"]),
                preconditioner,
            )
            if linearization is None:
                raise RuntimeError(f"unexpected skipped CPU cell: {key}")
            physical = sweep.build_brusselator_linearization(
                state,
                model,
                fft_cache,
                diffusivities,
                dt,
                time_value=float(record["time"]),
                preconditioner_kind="identity",
            )

            measured = _dual_stop_gmres(
                linearization.operator.matvec,
                linearization.rhs,
                physical.operator.matvec,
                physical.rhs,
            )
            committed_count = committed["cells"][key]["cpu"]["gmres"]["iterations"]
            if measured["preconditioned_stop_count"] != committed_count:
                raise RuntimeError(
                    f"preconditioned-stop count mismatch for {key}: "
                    f"{measured['preconditioned_stop_count']} != {committed_count}"
                )
            kappa, method = _condition_number(
                preconditioner, fft_cache, diffusivities, reaction, dt
            )
            cells[key] = {
                "state": spec.identifier,
                "family": spec.family,
                "regime": spec.regime,
                "dt": dt,
                "preconditioner": preconditioner,
                "source_state_sha256": source["state_identity"]["sha256"],
                "committed_preconditioned_stop_count": committed_count,
                **measured,
                "preconditioner_condition_number_2": kappa,
                "preconditioner_condition_number_method": method,
            }
            sweep._atomic_write_json(
                output,
                {"schema": SCHEMA, "status": "running", "cells": cells},
            )
    true_decision = _decision(cells, committed)
    original = committed["decision"]
    comparison = {
        name: {
            "preconditioned_passes": bool(original[key]["passes"]),
            "true_passes": bool(true_decision[key]["passes"]),
            "matches": bool(original[key]["passes"] == true_decision[key]["passes"]),
        }
        for name, key in (
            ("D1", "d1_material_degradation"),
            ("D2", "d2_absolute_cost"),
            ("D3", "d3_recoverability"),
        )
    }
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "provenance": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.x64_enabled),
            "git_commit": sweep._git("rev-parse", "HEAD"),
            "script_sha256": _sha256(Path(__file__)),
            "protocol_sha256": _sha256(protocol),
            "cpu_replication_sha256": _sha256(committed_path),
        },
        "plan": {"cells": len(cells), "tolerance": sweep.GMRES_TOLERANCE, "cap": 100},
        "cells": cells,
        "decision_true_stop": true_decision,
        "decision_comparison": comparison,
        "robustness_criterion_met": all(row["matches"] for row in comparison.values()),
        "cpu_wall_seconds": time.time() - started,
    }
    sweep._atomic_write_json(output, payload)
    return payload


def render(payload: dict[str, Any]) -> str:
    rows = []
    for _, row in sorted(payload["cells"].items()):
        kappa = row["preconditioner_condition_number_2"]
        residual = row["true_relative_residual_at_preconditioned_stop"]
        residual_text = f"{residual:.6g}" if residual is not None else "n/a"
        rows.append(
            f"| {row['state']} | {row['dt']:g} | {row['preconditioner']} | "
            f"{row['preconditioned_stop_count']} | {row['true_stop_count']} | "
            f"{residual_text} | "
            f"{kappa:.6g} |"
            if kappa is not None
            else f"| {row['state']} | {row['dt']:g} | {row['preconditioner']} | "
            f"{row['preconditioned_stop_count']} | {row['true_stop_count']} | "
            f"{residual_text} | unavailable |"
        )
    decision = payload["decision_true_stop"]
    comparison = payload["decision_comparison"]
    hopf_dt2 = [
        row
        for row in payload["cells"].values()
        if row["family"] == "developed64"
        and row["regime"] == "hopf"
        and row["dt"] == 2.0
        and row["preconditioner"] == "mean_field_coupled_fft"
    ]
    return "\n".join(
        [
            "# Brusselator `dt` sweep: physical true-residual check",
            "",
            "CPU wall times are diagnostic only.",
            "",
            "| State | dt | Preconditioner | Preconditioned stop | True stop | True residual at preconditioned stop | kappa_2(P) |",
            "|---|---:|---|---:|---:|---:|---:|",
            *rows,
            "",
            "## Decision under the physical true-residual convention",
            "",
            f"D1: **{'PASS' if decision['d1_material_degradation']['passes'] else 'FAIL'}**, "
            f"median ratio {decision['d1_material_degradation']['median_ratio']:.6g}.",
            f"D2: **{'PASS' if decision['d2_absolute_cost']['passes'] else 'FAIL'}**, "
            f"median {decision['d2_absolute_cost']['median_iterations']:.6g} iterations.",
            f"D3: **{'PASS' if decision['d3_recoverability']['passes'] else 'FAIL'}**; "
            f"decision **{decision['decision']}**.",
            f"D1/D2/D3 match the preconditioned convention: "
            f"{comparison['D1']['matches']}/{comparison['D2']['matches']}/{comparison['D3']['matches']}. "
            f"Robustness criterion met: **{payload['robustness_criterion_met']}**.",
            "",
            "## Developed Hopf mean-field cells at dt = 2",
            "",
            "| State | Preconditioned stop | True stop | True residual at preconditioned stop |",
            "|---|---:|---:|---:|",
            *(
                f"| {row['state']} | {row['preconditioned_stop_count']} | "
                f"{row['true_stop_count']} | {row['true_relative_residual_at_preconditioned_stop']:.6g} |"
                for row in sorted(hopf_dt2, key=lambda item: item["state"])
            ),
            "",
            f"The run took {payload['cpu_wall_seconds']:.6g} CPU seconds (diagnostic only).",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", required=True)
    parser.add_argument(
        "--output", default="benchmarks/results/brusselator_dt_sweep_true_residual.json"
    )
    parser.add_argument(
        "--markdown-output",
        default="benchmarks/results/BRUSSELATOR_DT_SWEEP_TRUE_RESIDUAL.md",
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    payload = run(repository, Path(args.source_state_dir).resolve(), output)
    Path(args.markdown_output).resolve().write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
