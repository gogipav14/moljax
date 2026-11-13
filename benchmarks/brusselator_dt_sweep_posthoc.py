"""Post-hoc reaction-variation diagnostics for the regenerated CPU states."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import spearmanr

from benchmarks import brusselator_dt_sweep as sweep
from benchmarks import brusselator_dt_sweep_cpu_replication as replication
from moljax.experimental.brusselator_conditioning import build_brusselator_system

jax.config.update("jax_enable_x64", True)


def _correlation(x: list[float], y: list[int]) -> dict[str, Any]:
    result = spearmanr(x, y)
    return {
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
        "n": len(x),
        "predictor_ties": len(set(x)) != len(x),
        "iteration_ties": len(set(y)) != len(y),
    }


def run(repository: Path, source_dir: Path, output: Path) -> dict[str, Any]:
    cpu_path = repository / "benchmarks/results/brusselator_dt_sweep_cpu_replication.json"
    true_path = repository / "benchmarks/results/brusselator_dt_sweep_true_residual.json"
    cpu = json.loads(cpu_path.read_text())
    true = json.loads(true_path.read_text())
    states = []
    for spec in sweep.CPU_STATE_SPECS:
        record = sweep._find_record(repository, spec)
        state, source = sweep._load_regenerated_state(source_dir, spec, record)
        grid, regime = sweep._grid_and_regime(record)
        _, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        reaction = sweep._reaction_jacobian(state, grid, regime.b)
        mean = jnp.mean(reaction, axis=(0, 1))
        variation = jnp.linalg.svd(reaction - mean, compute_uv=False)[..., 0]
        maximum = float(jax.block_until_ready(jnp.max(variation)))
        average = float(jax.block_until_ready(jnp.mean(variation)))
        max_reaction = sweep._reaction_metadata(reaction, 1.0)["max_reaction_jacobian_2_norm"]
        symbol = np.asarray(jax.device_get(fft_cache.laplacian_symbol), dtype=np.float64)
        mean_np = np.asarray(jax.device_get(mean), dtype=np.float64)
        minima = {}
        for dt in sweep.PROBE_DT_GRID:
            minimum = float("inf")
            for value in symbol.ravel():
                block = np.eye(2) - dt * (
                    np.diag([diffusivities["u"] * value, diffusivities["v"] * value]) + mean_np
                )
                minimum = min(minimum, float(np.linalg.svd(block, compute_uv=False)[-1]))
            minima[f"dt={dt:g}"] = minimum
        counts = {}
        for method, arm in (("fft_diffusion", "main"), ("mean_field_coupled_fft", "probe")):
            key = replication._cell_key(arm, spec, 1.0, method)
            counts[method] = {
                "preconditioned_stop": cpu["cells"][key]["cpu"]["gmres"]["iterations"],
                "true_stop": true["cells"][key]["true_stop_count"],
            }
        states.append(
            {
                "state": spec.identifier,
                "regime": spec.regime,
                "source_state_sha256": source["state_identity"]["sha256"],
                "max_reaction_jacobian_2_norm": max_reaction,
                "max_reaction_variation_2_norm": maximum,
                "mean_reaction_variation_2_norm": average,
                "mean_field_fourier_block_minimum_singular_value": minima,
                "iterations_at_dt_1": counts,
            }
        )
    correlations = {}
    for predictor, field in (
        ("max_reaction_jacobian_2_norm", "max_reaction_jacobian_2_norm"),
        ("max_reaction_variation_2_norm", "max_reaction_variation_2_norm"),
    ):
        correlations[predictor] = {}
        x = [row[field] for row in states]
        for method in ("fft_diffusion", "mean_field_coupled_fft"):
            correlations[predictor][method] = {
                convention: _correlation(
                    x, [row["iterations_at_dt_1"][method][convention] for row in states]
                )
                for convention in ("preconditioned_stop", "true_stop")
            }
    payload = {
        "schema": "brusselator_dt_sweep_posthoc_v1",
        "analysis_label": "exploratory post hoc; hypothesis formed after seeing sweep results",
        "provenance": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": jax.lib.__version__,
            "numpy": np.__version__,
            "backend": jax.default_backend(),
            "x64_enabled": bool(jax.config.x64_enabled),
            "git_commit": sweep._git("rev-parse", "HEAD"),
            "script_sha256": sweep._sha256(Path(__file__)),
            "cpu_replication_sha256": sweep._sha256(cpu_path),
            "true_residual_check_sha256": sweep._sha256(true_path),
        },
        "states": states,
        "spearman_at_dt_1": correlations,
    }
    sweep._atomic_write_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state-dir", required=True)
    parser.add_argument("--output", default="benchmarks/results/brusselator_dt_sweep_posthoc.json")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    run(repository, Path(args.source_state_dir).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
