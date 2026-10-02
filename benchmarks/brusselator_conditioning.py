#!/usr/bin/env python3
"""Parameterized Brusselator conditioning studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from statistics import median
from tempfile import NamedTemporaryFile
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    _integrate_visited_states,
    assess_brusselator_state,
    build_brusselator_system,
    measure_brusselator_gmres,
    sampled_visited_states,
    state_developedness,
)

jax.config.update("jax_enable_x64", True)


SOURCE_STATE_ARTIFACT_SCHEMA = "brusselator_conditioning_source_state_v4"
SOURCE_STATE_MODEL_VERSION = "brusselator_periodic_fft_be_v1"
RECORD_CHECKPOINT_SCHEMA = "brusselator_conditioning_record_checkpoint_v1"


class BrusselatorConditioningConfig(NamedTuple):
    """Configuration shared by all three benchmark presets."""

    mode: str
    nx: int
    ny: int
    dt: float
    perturbation: float
    seed: int
    n_angles: int
    fov_max_iters: int
    fov_residual_tolerance: float
    fov_n_restarts: int
    arnoldi_steps: int
    compute_lobpcg_upper_estimate: bool
    max_newton_iters: int
    max_krylov_iters: int
    newton_tol: float
    krylov_tol: float
    n_states: int = 0
    hopf_sample_steps: tuple[int, ...] = ()
    turing_sample_steps: tuple[int, ...] = ()
    regimes: tuple[str, ...] = ("hopf", "turing")
    output_path: str = "benchmarks/results/brusselator_conditioning.json"
    source_state_cache_dir: str | None = None
    record_checkpoint_path: str | None = None


def _config(mode: str, **kwargs: Any) -> BrusselatorConditioningConfig:
    defaults = dict(
        mode=mode,
        nx=64,
        ny=64,
        dt=0.1,
        perturbation=1.0e-3,
        seed=20260821,
        n_angles=4,
        fov_max_iters=8,
        fov_residual_tolerance=1.0e-3,
        fov_n_restarts=2,
        arnoldi_steps=6,
        compute_lobpcg_upper_estimate=True,
        max_newton_iters=10,
        max_krylov_iters=80,
        newton_tol=1.0e-8,
        krylov_tol=1.0e-8,
    )
    defaults.update(kwargs)
    return BrusselatorConditioningConfig(**defaults)


SCREEN_64 = _config(
    "screen_64", n_states=2, output_path="benchmarks/results/brusselator_conditioning.json"
)
DEVELOPED_64 = _config(
    "developed_64",
    dt=1.0,
    seed=20260822,
    max_newton_iters=15,
    max_krylov_iters=100,
    hopf_sample_steps=(1, 10, 20),
    turing_sample_steps=(80, 120, 200),
    output_path="benchmarks/results/brusselator_conditioning_developed.json",
)
FIXED_DT_256 = _config(
    "fixed_dt_256",
    nx=256,
    ny=256,
    dt=0.2,
    seed=20260823,
    max_newton_iters=15,
    max_krylov_iters=100,
    hopf_sample_steps=(1, 50),
    turing_sample_steps=(1, 1000),
    output_path="benchmarks/results/brusselator_conditioning_fixed_dt.json",
)
HOPF_CONTINUATION_256 = _config(
    "hopf_continuation_256",
    nx=256,
    ny=256,
    dt=0.05,
    seed=20260823,
    max_newton_iters=15,
    max_krylov_iters=100,
    hopf_sample_steps=(4, 400),
    regimes=("hopf",),
    output_path="benchmarks/results/brusselator_conditioning_hopf_continuation.json",
)
PRESETS = {
    "screen_64": SCREEN_64,
    "developed_64": DEVELOPED_64,
    "fixed_dt_256": FIXED_DT_256,
    "hopf_continuation_256": HOPF_CONTINUATION_256,
}


def _git_revision(*args: str) -> str | None:
    """Return an optional local Git revision without requiring a remote."""
    repository = Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _provenance_revisions() -> dict[str, str]:
    """Capture portable Git provenance without making a study depend on remotes."""
    repository_head = _git_revision("rev-parse", "HEAD")
    base_revision = _git_revision("merge-base", "HEAD", "upstream/main")
    if base_revision is not None:
        return {
            "repository_head": repository_head or "unavailable",
            "base_revision": base_revision,
            "base_revision_source": "merge-base-upstream",
        }
    tag_revision = _git_revision("describe", "--tags", "--abbrev=0")
    if tag_revision is not None:
        return {
            "repository_head": repository_head or "unavailable",
            "base_revision": tag_revision,
            "base_revision_source": "describe-tags",
        }
    if repository_head is not None:
        return {
            "repository_head": repository_head,
            "base_revision": repository_head,
            "base_revision_source": "head-fallback",
        }
    return {
        "repository_head": "unavailable",
        "base_revision": "unavailable",
        "base_revision_source": "unavailable",
    }


def _source_state_identity(state: dict[str, jax.Array]) -> dict[str, Any]:
    """Return a SHA256 identity for a two-field persisted source state."""
    digest = hashlib.sha256()
    fields: dict[str, dict[str, Any]] = {}
    for field in ("u", "v"):
        values = np.asarray(jax.device_get(state[field]), dtype=np.float64)
        digest.update(field.encode("utf-8"))
        digest.update(values.dtype.str.encode("utf-8"))
        digest.update(values.tobytes(order="C"))
        fields[field] = {"shape": list(values.shape), "dtype": values.dtype.str}
    return {"sha256": digest.hexdigest(), "fields": fields}


def _source_state_fingerprint(
    config: BrusselatorConditioningConfig,
    regime: Any,
    sample_steps: tuple[int, ...],
    seed: int,
) -> dict[str, Any]:
    """Return the v4 generation contract required to reuse a source trajectory."""
    return {
        "schema": SOURCE_STATE_ARTIFACT_SCHEMA,
        "regime": regime._asdict(),
        "grid": {"nx": config.nx, "ny": config.ny, "n_ghost": 1},
        "domain": {"x": [0.0, regime.domain_length], "y": [0.0, regime.domain_length]},
        "state_dt": config.dt,
        "seed": seed,
        "perturbation": config.perturbation,
        "sample_steps": list(sample_steps),
        "solver": {
            "max_newton_iters": config.max_newton_iters,
            "max_krylov_iters": config.max_krylov_iters,
            "newton_tol": config.newton_tol,
            "krylov_tol": config.krylov_tol,
            "max_backtrack": "NKParams-default-8",
        },
        "model_version": SOURCE_STATE_MODEL_VERSION,
    }


def _cache_directory(config: BrusselatorConditioningConfig) -> Path:
    """Return the source-state artifact directory for this explicit run."""
    if config.source_state_cache_dir is not None:
        return Path(config.source_state_cache_dir)
    return Path(config.output_path).parent / "brusselator_source_states"


def _source_fingerprint_key(fingerprint: dict[str, Any]) -> str:
    """Return a stable filesystem key for one full source-generation contract."""
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _cache_paths(
    config: BrusselatorConditioningConfig, regime: Any, fingerprint: dict[str, Any]
) -> tuple[Path, Path]:
    root = _cache_directory(config)
    stem = f"{regime.name}-{_source_fingerprint_key(fingerprint)}"
    return root / f"{stem}.npz", root / f"{stem}.json"


def _cache_relative_source_path(
    recorded: Any,
    expected_name: str,
    *,
    original_cache_root: str | None = None,
) -> str:
    """Return a recorded source-artifact path in cache-root-relative form.

    Revisions up to 8be9ef8 recorded the cache-prefixed path
    ``<cache root>/<regime>-<fingerprint key>.npz``, often absolute.  Such a
    legacy path is migrated to its cache-relative file name only when it has
    exactly that content-addressed name and, when the original cache root is
    known, lies directly under it.  The legacy location is never read: the
    caller loads the artifact from its own cache root, where the generation
    fingerprint and SHA256 identities are checked as for any other record.
    Traversal is always rejected.
    """
    if not isinstance(recorded, str) or not recorded:
        raise RuntimeError("record source-state artifact path is invalid")
    path = Path(recorded)
    if not path.parts or ".." in path.parts:
        raise RuntimeError("record source-state artifact path must be cache-root-relative")
    if not path.is_absolute() and len(path.parts) == 1:
        return str(path)
    if path.name != expected_name:
        raise RuntimeError("record source-state artifact path must be cache-root-relative")
    if original_cache_root is not None and path.parent != Path(original_cache_root):
        raise RuntimeError(
            "legacy source-state artifact path is outside its recorded cache root"
        )
    return path.name


def _cache_relative_source_artifact(
    artifact: dict[str, Any],
    regime_name: str,
    fingerprint: dict[str, Any],
    *,
    original_cache_root: str | None = None,
) -> dict[str, Any]:
    """Return a copy of one source artifact with a migrated cache-relative path."""
    expected_name = f"{regime_name}-{_source_fingerprint_key(fingerprint)}.npz"
    return {
        **artifact,
        "relative_path": _cache_relative_source_path(
            artifact.get("relative_path"),
            expected_name,
            original_cache_root=original_cache_root,
        ),
    }


def _diagnostic_contract(config: BrusselatorConditioningConfig) -> dict[str, Any]:
    """Return the complete, JSON-stable contract for diagnostic checkpoints."""
    return {
        "source_artifact_schema": SOURCE_STATE_ARTIFACT_SCHEMA,
        "mode": config.mode,
        "grid": {"nx": config.nx, "ny": config.ny, "n_ghost": 1},
        "dt": config.dt,
        "perturbation": config.perturbation,
        "seed": config.seed,
        "n_angles": config.n_angles,
        "fov_max_iters": config.fov_max_iters,
        "fov_residual_tolerance": config.fov_residual_tolerance,
        "fov_n_restarts": config.fov_n_restarts,
        "arnoldi_steps": config.arnoldi_steps,
        "compute_lobpcg_upper_estimate": config.compute_lobpcg_upper_estimate,
        "max_newton_iters": config.max_newton_iters,
        "max_krylov_iters": config.max_krylov_iters,
        "newton_tol": config.newton_tol,
        "krylov_tol": config.krylov_tol,
        "n_states": config.n_states,
        "hopf_sample_steps": list(config.hopf_sample_steps),
        "turing_sample_steps": list(config.turing_sample_steps),
        "regimes": list(config.regimes),
        "model_version": SOURCE_STATE_MODEL_VERSION,
    }


def _diagnostic_fingerprint(config: BrusselatorConditioningConfig) -> str:
    """Hash the immutable contract required to reuse diagnostic records."""
    encoded = json.dumps(_diagnostic_contract(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _load_record_checkpoint(config: BrusselatorConditioningConfig) -> dict[str, dict[str, Any]]:
    """Load completed records only when the exact diagnostic contract matches."""
    if config.record_checkpoint_path is None:
        return {}
    path = Path(config.record_checkpoint_path)
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != RECORD_CHECKPOINT_SCHEMA:
        raise RuntimeError(f"incompatible record checkpoint schema: {path}")
    if payload.get("diagnostic_fingerprint") != _diagnostic_fingerprint(config):
        raise RuntimeError(f"record checkpoint diagnostic fingerprint mismatch: {path}")
    records = payload.get("records")
    if not isinstance(records, dict) or not all(isinstance(row, dict) for row in records.values()):
        raise RuntimeError(f"record checkpoint is malformed: {path}")
    return records


def _persist_record_checkpoint(
    config: BrusselatorConditioningConfig, records: dict[str, dict[str, Any]]
) -> None:
    """Atomically persist each completed diagnostic record for process-safe resume."""
    if config.record_checkpoint_path is None:
        return
    _atomic_write_json(
        Path(config.record_checkpoint_path),
        {
            "schema": RECORD_CHECKPOINT_SCHEMA,
            "diagnostic_fingerprint": _diagnostic_fingerprint(config),
            "records": records,
        },
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_save_states(path: Path, states: list[dict[str, jax.Array]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"{field}_{index}": np.asarray(jax.device_get(state[field]), dtype=np.float64)
        for index, state in enumerate(states)
        for field in ("u", "v")
    }
    with NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_cached_states(
    config: BrusselatorConditioningConfig,
    regime: Any,
    fingerprint: dict[str, Any],
) -> tuple[list[dict[str, jax.Array]], list[dict[str, Any]]] | None:
    """Load a v4 source trajectory only when its contract and hashes match."""
    array_path, manifest_path = _cache_paths(config, regime, fingerprint)
    if not array_path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_STATE_ARTIFACT_SCHEMA:
        raise RuntimeError(f"stale source artifact schema: {manifest_path}")
    if manifest.get("generation_fingerprint") != fingerprint:
        raise RuntimeError(f"source-state fingerprint mismatch: {manifest_path}")
    identities = manifest.get("source_state_identities")
    if not isinstance(identities, list):
        raise RuntimeError(f"source artifact lacks identities: {manifest_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        states = [
            {
                "u": jax.block_until_ready(jnp.asarray(arrays[f"u_{index}"], dtype=jnp.float64)),
                "v": jax.block_until_ready(jnp.asarray(arrays[f"v_{index}"], dtype=jnp.float64)),
            }
            for index in range(len(identities))
        ]
    observed = [_source_state_identity(state) for state in states]
    if observed != identities:
        raise RuntimeError(f"source-state SHA256 mismatch: {array_path}")
    return states, identities


def _persist_source_states(
    config: BrusselatorConditioningConfig,
    regime: Any,
    fingerprint: dict[str, Any],
    states: list[dict[str, jax.Array]],
) -> tuple[list[dict[str, jax.Array]], list[dict[str, Any]]]:
    """Persist, hash, and reload a successful source trajectory exactly once."""
    array_path, manifest_path = _cache_paths(config, regime, fingerprint)
    identities = [_source_state_identity(state) for state in states]
    _atomic_save_states(array_path, states)
    _atomic_write_json(
        manifest_path,
        {
            "schema": SOURCE_STATE_ARTIFACT_SCHEMA,
            "generation_fingerprint": fingerprint,
            "array_path": array_path.name,
            "source_state_identities": identities,
        },
    )
    loaded = _load_cached_states(config, regime, fingerprint)
    if loaded is None:
        raise RuntimeError(f"failed to reload persisted source states: {array_path}")
    return loaded


def _nk(config: BrusselatorConditioningConfig) -> NKParams:
    return NKParams(
        max_newton_iters=config.max_newton_iters,
        max_krylov_iters=config.max_krylov_iters,
        newton_tol=config.newton_tol,
        krylov_tol=config.krylov_tol,
    )


def _records(config: BrusselatorConditioningConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    checkpoint_records = _load_record_checkpoint(config)
    regimes = tuple(
        regime for regime in (HOPF_REGIME, TURING_REGIME) if regime.name in config.regimes
    )
    if not regimes:
        raise ValueError("regimes must select at least one known Brusselator regime")
    for regime_index, regime in enumerate(regimes):
        grid = Grid2D.uniform(config.nx, config.ny, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        source_seed = config.seed + regime_index
        if config.mode == "screen_64":
            sample_steps = tuple(range(1, config.n_states + 1))
        else:
            sample_steps = (
                config.hopf_sample_steps if regime.name == "hopf" else config.turing_sample_steps
            )
        fingerprint = _source_state_fingerprint(config, regime, sample_steps, source_seed)
        cached = _load_cached_states(config, regime, fingerprint)
        if cached is None:
            if config.mode == "screen_64":
                generated = _integrate_visited_states(
                    regime,
                    model,
                    fft_cache,
                    n_steps=config.n_states,
                    dt=config.dt,
                    perturbation=config.perturbation,
                    seed=source_seed,
                    nk_params=_nk(config),
                )
                source_states, identities = _persist_source_states(
                    config, regime, fingerprint, generated
                )
                samples = [
                    (index, (index + 1) * config.dt, state, None)
                    for index, state in enumerate(source_states)
                ]
            else:
                visited = sampled_visited_states(
                    regime,
                    grid=grid,
                    sample_steps=sample_steps,
                    dt=config.dt,
                    perturbation=config.perturbation,
                    seed=source_seed,
                    nk_params=_nk(config),
                )
                source_states, identities = _persist_source_states(
                    config, regime, fingerprint, [sample.state for sample in visited]
                )
                samples = [
                    (sample.step, sample.time, state, state_developedness(state, grid, regime))
                    for sample, state in zip(visited, source_states, strict=True)
                ]
        else:
            source_states, identities = cached
            if config.mode == "screen_64":
                samples = [
                    (index, (index + 1) * config.dt, state, None)
                    for index, state in enumerate(source_states)
                ]
            else:
                samples = [
                    (step, step * config.dt, state, state_developedness(state, grid, regime))
                    for step, state in zip(sample_steps, source_states, strict=True)
                ]
        array_path, _ = _cache_paths(config, regime, fingerprint)
        relative_array_path = array_path.relative_to(_cache_directory(config))
        for sample_position, (index, time_value, state, developedness) in enumerate(samples):
            source_artifact = {
                "schema": SOURCE_STATE_ARTIFACT_SCHEMA,
                "relative_path": str(relative_array_path),
                "sample_position": sample_position,
                "source_state_identity": identities[sample_position],
                "generation_fingerprint": fingerprint,
                "converged": True,
            }
            for kind in ("identity", "fft_diffusion"):
                record_key = f"{regime.name}:{sample_position}:{kind}"
                completed = checkpoint_records.get(record_key)
                if completed is not None:
                    # Base-revision checkpoints carry cache-prefixed paths.  A
                    # path that cannot be migrated is left as recorded, so it
                    # fails the contract comparison below.
                    if isinstance(completed.get("source_state_artifact"), dict):
                        try:
                            completed = {
                                **completed,
                                "source_state_artifact": _cache_relative_source_artifact(
                                    completed["source_state_artifact"], regime.name, fingerprint
                                ),
                            }
                        except RuntimeError:
                            pass
                    if (
                        completed.get("source_state_artifact") != source_artifact
                        or completed.get("record_config", {}).get("preconditioner_kind") != kind
                        or completed.get("record_config", {}).get("analysis_dt") != config.dt
                    ):
                        raise RuntimeError(
                            f"record checkpoint does not match its source/operator contract: {record_key}"
                        )
                    records.append(completed)
                    checkpoint_records[record_key] = completed
                    continue
                assessment = assess_brusselator_state(
                    state,
                    model,
                    fft_cache,
                    diffusivities,
                    config.dt,
                    regime,
                    preconditioner_kind=kind,
                    time_value=time_value,
                    n_angles=config.n_angles,
                    fov_max_iters=config.fov_max_iters,
                    fov_residual_tolerance=config.fov_residual_tolerance,
                    fov_n_restarts=config.fov_n_restarts,
                    arnoldi_steps=config.arnoldi_steps,
                    compute_lobpcg_upper_estimate=config.compute_lobpcg_upper_estimate,
                    seed=config.seed + 100 * regime_index + 10 * index,
                )
                gmres = None
                if assessment["status"] == "completed":
                    gmres = measure_brusselator_gmres(
                        state,
                        model,
                        fft_cache,
                        diffusivities,
                        config.dt,
                        regime,
                        tol=config.krylov_tol,
                        max_iters=config.max_krylov_iters,
                        time_value=time_value,
                        preconditioner_kind=kind,
                    )
                row = {
                    **assessment,
                    "time": float(time_value),
                    "actual_gmres": gmres,
                    "source_state_artifact": source_artifact,
                    "record_config": {
                        "regime": regime._asdict(),
                        "grid": {"nx": config.nx, "ny": config.ny, "n_ghost": 1},
                        "domain_length": regime.domain_length,
                        "analysis_dt": config.dt,
                        "preconditioner_kind": kind,
                        "n_angles": config.n_angles,
                        "fov_max_iters": config.fov_max_iters,
                        "fov_residual_tolerance": config.fov_residual_tolerance,
                        "fov_n_restarts": config.fov_n_restarts,
                        "arnoldi_steps": config.arnoldi_steps,
                        "compute_lobpcg_upper_estimate": config.compute_lobpcg_upper_estimate,
                        "assessment_seed": config.seed + 100 * regime_index + 10 * index,
                        "source_state_fingerprint": fingerprint,
                    },
                    **_provenance_revisions(),
                }
                if config.mode == "screen_64":
                    row["state_index"] = index
                else:
                    row["trajectory_step"] = index
                    row["developedness"] = developedness
                records.append(row)
                checkpoint_records[record_key] = row
                _persist_record_checkpoint(config, checkpoint_records)
    return records


def _distribution(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {"adequate": 0, "investigate": 0, "indeterminate": 0, "skipped": 0}
    for record in records:
        result[record["verdict"]] = result.get(record["verdict"], 0) + 1
    return result


def _summary(
    records: list[dict[str, Any]], regime: Any, *, include_details: bool = True
) -> dict[str, Any]:
    rows = [r for r in records if r["preconditioner"] == "fft_diffusion"]
    complete = [r for r in rows if r["status"] == "completed"]

    def values(key: str) -> list[float]:
        return [float(r[key]) for r in complete]

    def sample_index(record: dict[str, Any]) -> int:
        return int(record.get("trajectory_step", record.get("state_index", 0)))

    iterations = [float(r["actual_gmres"]["iterations"]) for r in rows if r["actual_gmres"]]
    summary = {
        "parameters": regime._asdict(),
        "fft_records": len(rows),
        "verdict_distribution": _distribution(rows),
        "median_disk_rate": float(median(values("disk_rate"))) if complete else None,
        "median_fov_imaginary_extent": (
            float(median(values("fov_imaginary_extent"))) if complete else None
        ),
        "origin_enclosed_fraction": (
            float(sum(bool(r["origin_enclosed"]) for r in complete) / len(complete))
            if complete
            else None
        ),
        "median_actual_fft_gmres_iterations": float(median(iterations)) if iterations else None,
    }
    if include_details:
        summary.update(
            {
                "fov_imaginary_extent_by_time": [
                    {"time": r["time"], "fov_imaginary_extent": r["fov_imaginary_extent"]}
                    for r in sorted(complete, key=sample_index)
                ],
                "fft_gmres_iterations_by_time": [
                    {
                        "time": r["time"],
                        "iterations": r["actual_gmres"]["iterations"],
                        "converged": r["actual_gmres"]["converged"],
                    }
                    for r in sorted(rows, key=sample_index)
                    if r["actual_gmres"]
                ],
                "developedness_by_time": [
                    {"time": r["time"], **r["developedness"]}
                    for r in sorted(complete, key=sample_index)
                ],
            }
        )
    return summary


def _records_for(records: list[dict[str, Any]], regime: str) -> list[dict[str, Any]]:
    return [r for r in records if r["regime"] == regime]


_ABSTAINING_VERDICTS = frozenset({"indeterminate", "uncertified_at_cap", "skipped"})


def _unresolved_record(record: dict[str, Any]) -> bool:
    """Return whether a record abstains or rests on inconsistent FOV supports."""
    return str(record["verdict"]) in _ABSTAINING_VERDICTS or not bool(
        record.get("supports_consistent")
    )


def _fft_regime_status(rows: list[dict[str, Any]]) -> str:
    """Classify one regime's FFT records as adequate, unresolved, or not_adequate."""
    if rows and all(record["verdict"] == "adequate" for record in rows):
        return "adequate"
    if not rows or any(_unresolved_record(record) for record in rows):
        return "unresolved"
    return "not_adequate"


def _hopf_vs_turing(
    records: list[dict[str, Any]], study: str, *, scope_caveat: str | None = None
) -> dict[str, Any]:
    """Derive the mode-specific Hopf/Turing summary from the records' verdicts.

    This is the single source of the ``hopf_vs_turing`` conclusion: the base
    study report derives it from its raw records and the FOV-support resolver
    rebuilds it from final-policy records, so neither can assert an outcome
    the records do not support.
    """
    by_regime = {
        regime: [
            record
            for record in records
            if record["regime"] == regime and record["preconditioner"] == "fft_diffusion"
        ]
        for regime in ("hopf", "turing")
    }
    if study == "screen_64":
        adequate = {
            regime: sum(record["verdict"] == "adequate" for record in rows)
            for regime, rows in by_regime.items()
        }
        status = {regime: _fft_regime_status(rows) for regime, rows in by_regime.items()}
        both = all(value == "adequate" for value in status.values())
        if both:
            return {
                "outcome": "both_adequate_under_fft",
                "statement": (
                    "The FFT diffusion preconditioner is assessed adequate for both "
                    "visited-state regimes."
                ),
                "hopf_adequate_fft_records": adequate["hopf"],
                "turing_adequate_fft_records": adequate["turing"],
            }
        unresolved = all(value == "unresolved" for value in status.values())
        return {
            "outcome": (
                "fft_regime_assessments_unresolved"
                if unresolved
                else "fft_regime_assessments_mixed"
            ),
            "statement": (
                "No visited-state regime is assessed adequate under the FFT diffusion "
                "preconditioner: each regime has at least one unresolved FFT record "
                "(abstaining verdict or inconsistent FOV supports). No adequacy "
                "conclusion is drawn; see the per-regime statuses."
                if unresolved
                else "The FFT diffusion preconditioner has mixed outcomes across the "
                "visited-state regimes; see the per-regime statuses."
            ),
            "hopf_adequate_fft_records": adequate["hopf"],
            "turing_adequate_fft_records": adequate["turing"],
            "fft_status_by_regime": status,
            "unresolved_fft_records_by_regime": {
                regime: sum(_unresolved_record(record) for record in rows)
                for regime, rows in by_regime.items()
            },
        }
    if study != "developed_64":
        raise ValueError(f"Hopf/Turing comparison is not defined for {study}")
    all_indeterminate = {
        regime: bool(rows) and all(record["verdict"] == "indeterminate" for record in rows)
        for regime, rows in by_regime.items()
    }
    both = all(all_indeterminate.values())
    hopf_imaginary = sorted(by_regime["hopf"], key=lambda record: record["trajectory_step"])
    summary = {
        "outcome": (
            "both_regimes_indeterminate_on_developed_states"
            if both
            else "developed_fft_regime_assessments_mixed"
        ),
        "statement": (
            "Both evolved regimes are indeterminate at every sampled FFT-preconditioned state "
            "because their numerical ranges enclose the origin; Hopf still has the larger, "
            "growing imaginary extent."
            if both
            else "The developed FFT-preconditioned regimes have mixed outcomes; "
            "see the per-regime summaries."
        ),
        "hopf_nonadequate_fft_records": sum(
            record["verdict"] != "adequate" for record in by_regime["hopf"]
        ),
        "turing_nonadequate_fft_records": sum(
            record["verdict"] != "adequate" for record in by_regime["turing"]
        ),
        "hopf_origin_enclosed_any": any(
            bool(record["origin_enclosed"]) for record in by_regime["hopf"]
        ),
        "turing_origin_enclosed_any": any(
            bool(record["origin_enclosed"]) for record in by_regime["turing"]
        ),
        "both_regimes_indeterminate": both,
        "hopf_fov_imaginary_extent_grows_over_samples": (
            hopf_imaginary[-1]["fov_imaginary_extent"]
            > hopf_imaginary[0]["fov_imaginary_extent"]
        ),
        "hopf_fov_imaginary_extent_by_time": [
            {"time": record["time"], "fov_imaginary_extent": record["fov_imaginary_extent"]}
            for record in hopf_imaginary
        ],
    }
    if scope_caveat is not None:
        summary["scope_caveat"] = scope_caveat
    return summary


def _fixed_row(record: dict[str, Any]) -> dict[str, Any]:
    gmres = record["actual_gmres"]
    return {
        key: record[key]
        for key in (
            "trajectory_step",
            "time",
            "developedness",
            "verdict",
            "disk_rate",
            "epsilon_zero",
            "origin_enclosed",
            "fov_imaginary_extent",
            "n_right_real_outliers",
            "adjoint_error",
        )
    } | {
        "actual_gmres_iterations": None if gmres is None else gmres["iterations"],
        "actual_gmres_converged": None if gmres is None else gmres["converged"],
        "actual_gmres_final_relative_residual": (
            None if gmres is None else gmres["final_relative_residual"]
        ),
    }


def _fixed_transition(
    records: list[dict[str, Any]], config: BrusselatorConditioningConfig
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for regime in (HOPF_REGIME, TURING_REGIME):
        if regime.name not in config.regimes:
            continue
        result[regime.name] = {}
        for kind in ("identity", "fft_diffusion"):
            rows = sorted(
                (r for r in records if r["regime"] == regime.name and r["preconditioner"] == kind),
                key=lambda r: r["trajectory_step"],
            )
            early, developed = _fixed_row(rows[0]), _fixed_row(rows[-1])
            result[regime.name][kind] = {
                "early": early,
                "developed": developed,
                "adequate_to_indeterminate": early["verdict"] == "adequate"
                and developed["verdict"] == "indeterminate",
            }
    transitions = [result[name]["fft_diffusion"]["adequate_to_indeterminate"] for name in result]
    both = all(transitions)
    if len(transitions) == 1:
        transitioned = transitions[0]
        return {
            "outcome": (
                "fft_adequate_to_indeterminate_at_fixed_dt"
                if transitioned
                else "fft_verdict_stable_at_fixed_dt"
            ),
            "statement": (
                "At fixed backward-Euler dt, the FFT-preconditioned verdict changes from adequate at the early state to indeterminate at the developed state."
                if transitioned
                else "At fixed backward-Euler dt, the FFT-preconditioned verdict is unchanged between the early and developed states."
            ),
            "fixed_dt": config.dt,
            "same_discretized_operator_family": "Every early/developed pair uses the same periodic grid, shipped FFT preconditioner, and backward-Euler timestep. The state-dependent Jacobian changes between visited states by design; no comparison changes dt.",
            "by_regime": result,
        }
    return {
        "outcome": (
            "fft_adequate_to_indeterminate_in_both_regimes_at_fixed_dt"
            if both
            else "fft_adequate_to_indeterminate_in_one_regime_at_fixed_dt"
        ),
        "statement": (
            "At fixed backward-Euler dt, the FFT-preconditioned verdict changes from adequate at the early state to indeterminate at the developed state in both regimes."
            if both
            else "At fixed backward-Euler dt, at least one FFT-preconditioned regime changes from adequate early to indeterminate after its state develops; see the per-regime rows."
        ),
        "fixed_dt": config.dt,
        "same_discretized_operator_family": "Every early/developed pair uses the same periodic grid, shipped FFT preconditioner, and backward-Euler timestep. The state-dependent Jacobian changes between visited states by design; no comparison changes dt.",
        "by_regime": result,
    }


def _result(config: BrusselatorConditioningConfig, records: list[dict[str, Any]]) -> dict[str, Any]:
    comparison = {
        regime.name: _summary(
            _records_for(records, regime.name),
            regime,
            include_details=config.mode != "screen_64",
        )
        for regime in (HOPF_REGIME, TURING_REGIME)
        if regime.name in config.regimes
    }
    config_json = {key: value for key, value in config._asdict().items() if key != "mode"}
    if config.regimes == ("hopf", "turing"):
        config_json.pop("regimes")
    model = {
        "name": "brusselator",
        "grid": [config.nx, config.ny],
        "boundary": "periodic",
        "spatial_operator": "moljax shipped periodic FFT-preconditioned path",
        "state_generation": "backward_euler_newton_krylov",
        "diagnostic_preconditioners": ["identity", "fft_diffusion"],
        "exact_solution_error": "not applicable: conditioning study",
    }
    provenance = _provenance_revisions()
    if config.mode == "screen_64":
        return {
            "schema_version": "brusselator_conditioning_v1",
            "status": "completed",
            "config": {
                k: v
                for k, v in config_json.items()
                if k not in {"hopf_sample_steps", "turing_sample_steps"}
            },
            "model": model,
            "provenance": provenance,
            "records": records,
            "regime_comparison": comparison,
            "hopf_vs_turing": _hopf_vs_turing(records, config.mode),
        }
    if config.mode == "developed_64":
        config_json.pop("n_states", None)
        return {
            "schema_version": "brusselator_conditioning_developed_v1",
            "status": "completed",
            "config": config_json,
            "model": model,
            "provenance": provenance,
            "records": records,
            "regime_comparison": comparison,
            "hopf_vs_turing": _hopf_vs_turing(
                records,
                config.mode,
                scope_caveat="This is a 64x64 screen with BE dt=1; Hopf reaches t=20 and Turing reaches t=200, below the 256x256 target scale. The FOV values use the dt=1 BE operator.",
            ),
        }
    model["domain_length"] = 5.0
    model["spatial_operator"] = "moljax shipped periodic pseudo-spectral FFT path"
    model["state_generation"] = "FFT-preconditioned backward_euler_newton_krylov"
    config_json.pop("n_states", None)
    hopf_time = config.dt * config.hopf_sample_steps[-1] if config.hopf_sample_steps else None
    turing_time = config.dt * config.turing_sample_steps[-1] if config.turing_sample_steps else None
    if config.mode == "hopf_continuation_256":
        return {
            "schema_version": "brusselator_conditioning_hopf_continuation_v1",
            "status": "completed",
            "config": config_json,
            "model": model,
            "provenance": provenance,
            "records": records,
            "fixed_dt_transition": _fixed_transition(records, config),
            "scope": {
                "grid_resolution": "256x256 physical periodic grid at L=5",
                "hopf_developed_time": hopf_time,
                "same_dt_for_early_and_developed": True,
                "caveat": "This Hopf-only continuation uses a smaller fixed BE timestep to reach a developed state beyond the previous dt=0.2 continuation limit.",
            },
        }
    return {
        "schema_version": "brusselator_conditioning_fixed_dt_v1",
        "status": "completed",
        "config": config_json,
        "model": model,
        "provenance": provenance,
        "records": records,
        "fixed_dt_transition": _fixed_transition(records, config),
        "scope": {
            "grid_resolution": "256x256 physical periodic grid at L=5",
            "hopf_developed_time": hopf_time,
            "turing_developed_time": turing_time,
            "turing_reaches_t200": turing_time == 200.0,
            "caveat": "The Turing developed state reaches t=200. The Hopf sample is a developed state at the stated time, not a claim to reproduce a full long-time attractor.",
        },
    }


def run_brusselator_conditioning_study(config: BrusselatorConditioningConfig) -> dict[str, Any]:
    """Run one preset and return its JSON-ready result."""
    return _result(config, _records(config))


def reassess_brusselator_record(
    record: dict[str, Any],
    *,
    source_state_cache_dir: str,
    original_source_state_cache_dir: str | None = None,
    n_angles: int | None = None,
    fov_max_iters: int | None = None,
    fov_n_restarts: int | None = None,
) -> dict[str, Any]:
    """Reassess one record from its complete stored configuration, fail closed.

    This recovery path deliberately never consults a benchmark preset or a
    default.  It reconstructs the grid, regime, and preconditioner from the
    serialized record, then reloads the exact v4 source artifact after
    checking its fingerprint and SHA256 identity.  Optional FOV controls are
    an explicit diagnostic-resolution override only; they never affect the
    persisted state-generation contract.  A base-revision cache-prefixed
    artifact path is migrated to cache-relative form first; when
    ``original_source_state_cache_dir`` is given, it must lie directly under
    that recorded root.
    """
    try:
        stored = record["record_config"]
        artifact = record["source_state_artifact"]
        regime = type(HOPF_REGIME)(**stored["regime"])
        grid_values = stored["grid"]
        fingerprint = stored["source_state_fingerprint"]
        position = int(artifact["sample_position"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("record lacks complete reassessment provenance") from error
    if artifact.get("schema") != SOURCE_STATE_ARTIFACT_SCHEMA:
        raise RuntimeError("record uses an incompatible source-state artifact schema")
    if artifact.get("generation_fingerprint") != fingerprint:
        raise RuntimeError("record source-state fingerprint is internally inconsistent")
    if stored["analysis_dt"] <= 0.0 or stored["preconditioner_kind"] not in {
        "identity",
        "fft_diffusion",
    }:
        raise RuntimeError("record contains an invalid replay operator configuration")
    resolved_n_angles = int(stored["n_angles"] if n_angles is None else n_angles)
    resolved_fov_max_iters = int(
        stored["fov_max_iters"] if fov_max_iters is None else fov_max_iters
    )
    resolved_fov_n_restarts = int(
        stored["fov_n_restarts"] if fov_n_restarts is None else fov_n_restarts
    )
    if resolved_n_angles < 3 or resolved_fov_max_iters < 1 or resolved_fov_n_restarts < 1:
        raise ValueError("FOV reassessment controls must be positive and use at least 3 angles")
    replay = BrusselatorConditioningConfig(
        mode="reassess",
        nx=int(grid_values["nx"]),
        ny=int(grid_values["ny"]),
        dt=float(stored["analysis_dt"]),
        perturbation=0.0,
        seed=0,
        n_angles=resolved_n_angles,
        fov_max_iters=resolved_fov_max_iters,
        fov_residual_tolerance=float(stored["fov_residual_tolerance"]),
        fov_n_restarts=resolved_fov_n_restarts,
        arnoldi_steps=int(stored["arnoldi_steps"]),
        compute_lobpcg_upper_estimate=bool(
            stored["compute_lobpcg_upper_estimate"]
        ),
        max_newton_iters=0,
        max_krylov_iters=0,
        newton_tol=0.0,
        krylov_tol=0.0,
        source_state_cache_dir=source_state_cache_dir,
    )
    expected_path, _ = _cache_paths(replay, regime, fingerprint)
    relative_path = Path(
        _cache_relative_source_path(
            artifact.get("relative_path"),
            expected_path.name,
            original_cache_root=original_source_state_cache_dir,
        )
    )
    if _cache_directory(replay) / relative_path != expected_path:
        raise RuntimeError("record source-state artifact path does not match the replay cache")
    loaded = _load_cached_states(replay, regime, fingerprint)
    if loaded is None:
        raise RuntimeError("record source-state artifact is missing")
    states, identities = loaded
    if position < 0 or position >= len(states) or identities[position] != artifact["source_state_identity"]:
        raise RuntimeError("record source-state identity does not match the persisted artifact")
    state = states[position]
    if tuple(state["u"].shape) != (replay.ny + 2, replay.nx + 2):
        raise RuntimeError("persisted state shape does not match the recorded grid")
    grid = Grid2D.uniform(
        replay.nx,
        replay.ny,
        0.0,
        float(stored["domain_length"]),
        0.0,
        float(stored["domain_length"]),
        n_ghost=int(grid_values["n_ghost"]),
    )
    model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
    return assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        replay.dt,
        regime,
        preconditioner_kind=str(stored["preconditioner_kind"]),
        time_value=float(record["time"]),
        n_angles=replay.n_angles,
        fov_max_iters=replay.fov_max_iters,
        fov_residual_tolerance=replay.fov_residual_tolerance,
        fov_n_restarts=replay.fov_n_restarts,
        arnoldi_steps=replay.arnoldi_steps,
        compute_lobpcg_upper_estimate=replay.compute_lobpcg_upper_estimate,
        seed=int(stored["assessment_seed"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=sorted(PRESETS), default="screen_64")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-state-cache-dir",
        type=Path,
        help="persisted v4 source-state directory; reuse is SHA256/fingerprint verified",
    )
    parser.add_argument(
        "--record-checkpoint",
        type=Path,
        help="atomic per-record diagnostic checkpoint for resumable regeneration",
    )
    args = parser.parse_args()
    config = PRESETS[args.study]._replace(
        source_state_cache_dir=(
            None if args.source_state_cache_dir is None else str(args.source_state_cache_dir)
        ),
        record_checkpoint_path=(
            None if args.record_checkpoint is None else str(args.record_checkpoint)
        ),
    )
    output = args.output or Path(config.output_path)
    result = run_brusselator_conditioning_study(config)
    _atomic_write_json(output, result)
    print(json.dumps(result, indent=2))
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
