"""Run and render the pre-registered 64-square CPU replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from benchmarks import brusselator_dt_sweep as sweep
from moljax.experimental.brusselator_conditioning import build_brusselator_system

jax.config.update("jax_enable_x64", True)

SCHEMA = "brusselator_dt_sweep_cpu_replication_v1"
PROTOCOL = "benchmarks/BRUSSELATOR_DT_SWEEP_CPU_REPLICATION.md"
GPU_RESULT = "benchmarks/results/brusselator_dt_sweep.json"
DEVELOPED = "developed64"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pointer(path: str) -> str:
    return f"`brusselator_dt_sweep_cpu_replication.json#{path}`"


def _median(values: list[int | float]) -> float:
    return float(statistics.median(values))


def _cell_specs() -> list[tuple[str, sweep.StateSpec, float, str]]:
    cells: list[tuple[str, sweep.StateSpec, float, str]] = []
    for spec in sweep.CPU_STATE_SPECS:
        for dt in sweep.DT_GRID:
            for preconditioner in sweep.MAIN_PRECONDITIONERS:
                cells.append(("main", spec, dt, preconditioner))
        for dt in sweep.PROBE_DT_GRID:
            for preconditioner in sweep.PROBE_PRECONDITIONERS:
                cells.append(("probe", spec, dt, preconditioner))
    return cells


def _needs_disk(spec: sweep.StateSpec, dt: float, preconditioner: str) -> bool:
    return (
        spec.family == DEVELOPED
        and dt == 1.0
        and preconditioner in {"fft_diffusion", "mean_field_coupled_fft"}
    )


def _decision(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    states = sorted({cell["state"] for cell in cells.values() if cell["family"] == DEVELOPED})
    d1_rows = []
    for state in states:
        low_keys = [sweep._cell_key("main", state, dt, "fft_diffusion") for dt in (0.05, 0.1, 0.2)]
        dt1_key = sweep._cell_key("main", state, 1.0, "fft_diffusion")
        low = [int(cells[key]["cpu"]["gmres"]["iterations"]) for key in low_keys]
        dt1 = int(cells[dt1_key]["cpu"]["gmres"]["iterations"])
        converged = all(cells[key]["cpu"]["gmres"]["converged"] for key in (*low_keys, dt1_key))
        low_median = _median(low)
        ratio = dt1 / low_median if converged else None
        d1_rows.append(
            {
                "state": state,
                "low_dt_cell_keys": low_keys,
                "low_dt_iterations": low,
                "low_dt_median_iterations": low_median,
                "dt1_cell_key": dt1_key,
                "dt1_iterations": dt1,
                "all_converged": converged,
                "ratio": ratio,
                "passes_ratio_threshold": bool(ratio is not None and ratio >= 2.0),
            }
        )
    ratios = [row["ratio"] for row in d1_rows if row["ratio"] is not None]
    d1 = {
        "states": d1_rows,
        "ratio_threshold": 2.0,
        "minimum_states_passing": 5,
        "median_ratio": _median(ratios),
        "states_passing": sum(row["passes_ratio_threshold"] for row in d1_rows),
        "passes": bool(
            len(ratios) == 6
            and _median(ratios) >= 2.0
            and sum(row["passes_ratio_threshold"] for row in d1_rows) >= 5
        ),
    }
    dt1_values = [int(row["dt1_iterations"]) for row in d1_rows]
    d2 = {
        "iterations": dt1_values,
        "median_iterations": _median(dt1_values),
        "threshold": 25.0,
        "passes": bool(
            all(row["all_converged"] for row in d1_rows) and _median(dt1_values) >= 25.0
        ),
    }
    probes = []
    for preconditioner in sweep.PROBE_PRECONDITIONERS:
        rows = []
        for state in states:
            fft_key = sweep._cell_key("main", state, 1.0, "fft_diffusion")
            probe_key = sweep._cell_key("probe", state, 1.0, preconditioner)
            fft = cells[fft_key]["cpu"]
            probe = cells[probe_key]["cpu"]
            geometry = fft.get("disk") and probe.get("disk")
            geometry_valid = bool(
                geometry
                and fft["disk"]["supports_consistent"]
                and fft["disk"]["supports_converged"]
                and probe["disk"]["supports_consistent"]
                and probe["disk"]["supports_converged"]
            )
            rows.append(
                {
                    "state": state,
                    "fft_cell_key": fft_key,
                    "probe_cell_key": probe_key,
                    "fft_iterations": int(fft["gmres"]["iterations"]),
                    "probe_iterations": int(probe["gmres"]["iterations"]),
                    "probe_converged": bool(probe["gmres"]["converged"]),
                    "geometry_valid": geometry_valid,
                    "fft_disk_rate": fft.get("disk", {}).get("disk_rate"),
                    "probe_disk_rate": probe.get("disk", {}).get("disk_rate"),
                    "probe_disk_rate_lower": bool(
                        geometry_valid and probe["disk"]["disk_rate"] < fft["disk"]["disk_rate"]
                    ),
                }
            )
        fft_median = _median([row["fft_iterations"] for row in rows])
        probe_median = _median([row["probe_iterations"] for row in rows])
        reduction = 1.0 - probe_median / fft_median
        all_probe_cells = [
            cell for cell in cells.values() if cell["preconditioner"] == preconditioner
        ]
        all_converged = all(
            cell["cpu"]["status"] == "completed" and cell["cpu"]["gmres"]["converged"]
            for cell in all_probe_cells
        )
        lower = sum(row["probe_disk_rate_lower"] for row in rows)
        geometry_valid = all(row["geometry_valid"] for row in rows)
        probes.append(
            {
                "preconditioner": preconditioner,
                "states": rows,
                "fft_median_iterations": fft_median,
                "probe_median_iterations": probe_median,
                "relative_reduction": reduction,
                "relative_reduction_threshold": 0.4,
                "all_gmres_runs_converged": all_converged,
                "all_geometry_valid": geometry_valid,
                "lower_disk_rate_states": lower,
                "lower_disk_rate_states_threshold": 4,
                "passes": bool(
                    reduction >= 0.4 and all_converged and geometry_valid and lower >= 4
                ),
            }
        )
    d3 = {"probes": probes, "passes": any(probe["passes"] for probe in probes)}
    return {
        "d1_material_degradation": d1,
        "d2_absolute_cost": d2,
        "d3_recoverability": d3,
        "all_conditions_pass": bool(d1["passes"] and d2["passes"] and d3["passes"]),
        "decision": "BUILD" if d1["passes"] and d2["passes"] and d3["passes"] else "DO NOT BUILD",
    }


def _initial_payload(repository: Path, source_dir: Path, gpu_path: Path) -> dict[str, Any]:
    script = Path(__file__).resolve()
    protocol = repository / PROTOCOL
    source_manifest = source_dir / "regenerated_source_states_manifest.json"
    return {
        "schema": SCHEMA,
        "status": "running",
        "provenance": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "x64_enabled": bool(jax.config.x64_enabled),
            "git_commit": sweep._git("rev-parse", "HEAD"),
            "script_sha256": _sha256(script),
            "protocol_sha256": _sha256(protocol),
            "gpu_result_sha256": _sha256(gpu_path),
            "regenerated_source_manifest_sha256": _sha256(source_manifest),
        },
        "plan": {
            "states": [spec.identifier for spec in sweep.CPU_STATE_SPECS],
            "gmres_cells": len(_cell_specs()),
            "disk_cells": 12,
            "gmres_tolerance": sweep.GMRES_TOLERANCE,
            "gmres_max_iters": sweep.GMRES_MAX_ITERS,
        },
        "started_at_unix": time.time(),
        "similarity": {},
        "cells": {},
    }


def run(repository: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    if jax.default_backend() != "cpu" or not jax.config.x64_enabled:
        raise RuntimeError("the CPU replication requires the CPU backend and JAX x64")
    gpu_path = repository / GPU_RESULT
    gpu = json.loads(gpu_path.read_text(encoding="utf-8"))
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
        if payload.get("schema") != SCHEMA:
            raise RuntimeError("CPU replication checkpoint schema mismatch")
    else:
        payload = _initial_payload(repository, source_dir, gpu_path)

    for spec in sweep.CPU_STATE_SPECS:
        record = sweep._find_record(repository, spec)
        state, source = sweep._load_regenerated_state(source_dir, spec, record)
        grid, regime = sweep._grid_and_regime(record)
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        reaction_jacobian = sweep._reaction_jacobian(state, grid, regime.b)
        regenerated_strength = sweep._reaction_metadata(reaction_jacobian, 1.0)[
            "max_reaction_jacobian_2_norm"
        ]
        gpu_key = sweep._cell_key("main", spec.identifier, 1.0, "fft_diffusion")
        gpu_strength = float(gpu["cells"][gpu_key]["reaction"]["max_reaction_jacobian_2_norm"])
        relative = abs(regenerated_strength - gpu_strength) / abs(gpu_strength)
        payload["similarity"][spec.identifier] = {
            "gpu_cell_key": gpu_key,
            "gpu_max_reaction_jacobian_2_norm": gpu_strength,
            "regenerated_max_reaction_jacobian_2_norm": regenerated_strength,
            "relative_difference": relative,
            "threshold": 1.0e-6,
            "within_threshold": relative <= 1.0e-6,
            "source": source,
        }

        for arm, planned_spec, dt, preconditioner in _cell_specs():
            if planned_spec != spec:
                continue
            key = sweep._cell_key(arm, spec.identifier, dt, preconditioner)
            if key in payload["cells"]:
                continue
            started = time.perf_counter()
            linearization, preconditioner_metadata = sweep._build_cell_linearization(
                state,
                model,
                fft_cache,
                diffusivities,
                reaction_jacobian,
                dt,
                float(record["time"]),
                preconditioner,
            )
            if linearization is None:
                cpu = {
                    "status": "skipped",
                    "skip_reason": "pointwise reaction inverse determinant below floor",
                    "preconditioner_metadata": preconditioner_metadata,
                    "wall_seconds": time.perf_counter() - started,
                }
            else:
                gmres_started = time.perf_counter()
                gmres = sweep._measure_gmres(linearization)
                cpu = {
                    "status": "completed",
                    "gmres": gmres,
                    "gmres_seconds": time.perf_counter() - gmres_started,
                    "preconditioner_metadata": preconditioner_metadata,
                }
                if _needs_disk(spec, dt, preconditioner):
                    cpu["disk"] = sweep._measure_disk(
                        linearization,
                        sweep._stable_seed(f"cpu:{arm}:{spec.identifier}:{dt}:{preconditioner}"),
                    )
                cpu["wall_seconds"] = time.perf_counter() - started
            gpu_cell = gpu["cells"][key]
            payload["cells"][key] = {
                "arm": arm,
                "state": spec.identifier,
                "family": spec.family,
                "regime": spec.regime,
                "dt": dt,
                "preconditioner": preconditioner,
                "gpu_json_pointer": f"/cells/{key.replace('~', '~0').replace('/', '~1')}",
                "gpu": {
                    "status": gpu_cell["status"],
                    "gmres": gpu_cell.get("gmres"),
                    "disk_rate": gpu_cell.get("disk_rate"),
                },
                "cpu": cpu,
            }
            sweep._atomic_write_json(output, payload)

    differences = [row["relative_difference"] for row in payload["similarity"].values()]
    payload["similarity_classification"] = {
        "maximum_relative_difference": max(differences),
        "threshold": 1.0e-6,
        "all_within_threshold": all(value <= 1.0e-6 for value in differences),
        "label": (
            "reproduction"
            if all(value <= 1.0e-6 for value in differences)
            else "replication on regenerated realisations"
        ),
    }
    payload["decision"] = _decision(payload["cells"])
    gpu_decision = gpu["analysis"]["decision"]
    payload["decision_comparison"] = {
        name: {
            "gpu_passes": bool(gpu_decision[gpu_key]["passes"]),
            "cpu_passes": bool(payload["decision"][cpu_key]["passes"]),
            "matches": bool(
                gpu_decision[gpu_key]["passes"] == payload["decision"][cpu_key]["passes"]
            ),
        }
        for name, gpu_key, cpu_key in (
            ("D1", "d1_material_degradation", "d1_material_degradation"),
            ("D2", "d2_absolute_cost", "d2_absolute_cost"),
            ("D3", "d3_recoverability", "d3_recoverability"),
        )
    }
    payload["success_criterion_met"] = all(
        row["matches"] for row in payload["decision_comparison"].values()
    )
    payload["completed_at_unix"] = time.time()
    payload["cpu_wall_seconds"] = payload["completed_at_unix"] - payload["started_at_unix"]
    payload["status"] = "complete"
    sweep._atomic_write_json(output, payload)
    return payload


def render(payload: dict[str, Any]) -> str:
    similarity_rows = []
    for state, row in sorted(payload["similarity"].items()):
        base = f"/similarity/{state.replace('~', '~0').replace('/', '~1')}"
        similarity_rows.append(
            f"| {state} | {row['gpu_max_reaction_jacobian_2_norm']:.12g} | "
            f"{row['regenerated_max_reaction_jacobian_2_norm']:.12g} | "
            f"{row['relative_difference']:.6g} | {row['within_threshold']} | "
            f"{_pointer(base)} |"
        )
    d1_rows = []
    for index, row in enumerate(payload["decision"]["d1_material_degradation"]["states"]):
        d1_rows.append(
            f"| {row['state']} | {row['low_dt_iterations']} | {row['dt1_iterations']} | "
            f"{row['ratio']:.6g} | {row['passes_ratio_threshold']} | "
            f"{_pointer(f'/decision/d1_material_degradation/states/{index}')} |"
        )
    d3_rows = []
    for index, probe in enumerate(payload["decision"]["d3_recoverability"]["probes"]):
        d3_rows.append(
            f"| {probe['preconditioner']} | {probe['fft_median_iterations']:.6g} | "
            f"{probe['probe_median_iterations']:.6g} | {probe['relative_reduction']:.6g} | "
            f"{probe['lower_disk_rate_states']} | {probe['all_gmres_runs_converged']} | "
            f"{probe['passes']} | {_pointer(f'/decision/d3_recoverability/probes/{index}')} |"
        )
    cell_rows = []
    for key, cell in sorted(payload["cells"].items()):
        cpu = cell["cpu"]
        gpu_gmres = cell["gpu"]["gmres"]
        cpu_gmres = cpu.get("gmres")
        cell_rows.append(
            f"| {cell['state']} | {cell['dt']:g} | {cell['preconditioner']} | "
            f"{gpu_gmres['iterations'] if gpu_gmres else 'skip'} | "
            f"{cpu_gmres['iterations'] if cpu_gmres else 'skip'} | "
            f"{gpu_gmres['converged'] if gpu_gmres else 'n/a'} | "
            f"{cpu_gmres['converged'] if cpu_gmres else 'n/a'} | "
            f"{cpu['wall_seconds']:.6g} | {_pointer('/cells/' + key.replace('~', '~0').replace('/', '~1'))} |"
        )
    comparison = payload["decision_comparison"]
    return "\n".join(
        [
            "# Brusselator `dt` sweep: CPU replication on regenerated states",
            "",
            "CPU wall times below are diagnostic only and are not comparable with the original GPU run.",
            "",
            "## Similarity classification",
            "",
            f"Classification: **{payload['similarity_classification']['label']}**; maximum relative difference "
            f"{payload['similarity_classification']['maximum_relative_difference']:.6g} against threshold "
            f"{payload['similarity_classification']['threshold']:.6g}. "
            f"Source: {_pointer('/similarity_classification')}.",
            "",
            "| State | GPU max reaction norm | CPU max reaction norm | Relative difference | Within threshold | Source |",
            "|---|---:|---:|---:|:---:|---|",
            *similarity_rows,
            "",
            "## Pre-registered decision",
            "",
            "| State | CPU FFT counts at dt <= 0.2 | CPU FFT count at dt=1 | Ratio | Pass | Source |",
            "|---|---|---:|---:|:---:|---|",
            *d1_rows,
            "",
            f"D1: **{'PASS' if payload['decision']['d1_material_degradation']['passes'] else 'FAIL'}**; "
            f"median ratio {payload['decision']['d1_material_degradation']['median_ratio']:.6g}. "
            f"Source: {_pointer('/decision/d1_material_degradation')}.",
            "",
            f"D2: **{'PASS' if payload['decision']['d2_absolute_cost']['passes'] else 'FAIL'}**; "
            f"median count {payload['decision']['d2_absolute_cost']['median_iterations']:.6g}. "
            f"Source: {_pointer('/decision/d2_absolute_cost')}.",
            "",
            "| Probe | FFT median | Probe median | Reduction | Lower disk rates | All runs converged | Pass | Source |",
            "|---|---:|---:|---:|---:|:---:|:---:|---|",
            *d3_rows,
            "",
            f"D3: **{'PASS' if payload['decision']['d3_recoverability']['passes'] else 'FAIL'}**. "
            f"Source: {_pointer('/decision/d3_recoverability')}.",
            "",
            f"CPU decision: **{payload['decision']['decision']}**. D1/D2/D3 match GPU: "
            f"{comparison['D1']['matches']}/{comparison['D2']['matches']}/{comparison['D3']['matches']}. "
            f"Success criterion met: {payload['success_criterion_met']}. "
            f"Source: {_pointer('/decision_comparison')}.",
            "",
            "## Cell-by-cell comparison",
            "",
            "| State | dt | Preconditioner | GPU iterations | CPU iterations | GPU converged | CPU converged | CPU seconds | Source |",
            "|---|---:|---|---:|---:|:---:|:---:|---:|---|",
            *cell_rows,
            "",
            "## Diagnostic CPU cost",
            "",
            f"The complete CPU replication took {payload['cpu_wall_seconds']:.6g} seconds. "
            f"Source: {_pointer('/cpu_wall_seconds')}.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", required=True)
    parser.add_argument(
        "--output", default="benchmarks/results/brusselator_dt_sweep_cpu_replication.json"
    )
    parser.add_argument(
        "--markdown-output",
        default="benchmarks/results/BRUSSELATOR_DT_SWEEP_CPU_REPLICATION.md",
    )
    arguments = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    output = Path(arguments.output).resolve()
    payload = run(repository, Path(arguments.source_state_dir).resolve(), output)
    markdown = Path(arguments.markdown_output).resolve()
    markdown.write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
