#!/usr/bin/env python3
"""Regenerate the Stage-2 conditioning studies in resumable fresh-process batches.

Each batch invokes an existing study runner for one independent exponent or
reaction-strength slice.  Completed batch reports are atomically checkpointed
outside the repository, then assembled without further numerical work into the
canonical result JSONs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pme_breakdown
import porous_fisher_conditioning

import moljax

DEFAULT_CHECKPOINT_DIR = Path("/tmp/moljax-stage2-conditioning-checkpoints")
# A batch is only as current as the source states its records were measured on,
# so the two batch schemas move with SOURCE_STATE_ARTIFACT_SCHEMA.  Leaving them
# behind would let an already-completed batch return its pre-fingerprint records
# to assembly, which then republishes diagnostics that reassessment refuses.
PME_BATCH_SCHEMA = "stage2_conditioning_pme_batch_v4"
POROUS_FISHER_BATCH_SCHEMA = "stage2_conditioning_porous_fisher_batch_v4"
SOURCE_STATE_ARTIFACT_SCHEMA = "stage2_conditioning_source_state_v2"
SOURCE_STATE_FINGERPRINT_SCHEMA = "stage2_conditioning_source_state_fingerprint_v1"
BBA5A94_AUDIT_SCHEMA = "stage2_conditioning_bba5a94_verdict_audit_v1"
C1_OPTION_A_AUDIT_SCHEMA = "stage2_conditioning_c1_option_a_audit_v1"
CURRENT_MAIN_AUDIT_SCHEMA = "stage2_conditioning_current_main_verdict_audit_v1"
V121_RECOVERY_SCHEMA = "stage2_conditioning_v1_2_1_recovery_v1"
GEOMETRY_LADDER = ((32, 120, 2), (64, 180, 2), (96, 240, 2))
SUPPORT_ITERATION_LADDER = (240, 480)
BASE_GEOMETRY_BUDGET = {"n_angles": 16, "fov_max_iters": 60, "fov_n_restarts": 2}

# These four measurements were completed before the broad escalation pass.  They
# are retained here so the final assembly can use their already-observed results
# without recomputing the priority reaction-axis cases.
PRIORITY_REACTION_RESOLUTIONS: dict[tuple[float, float], tuple[dict[str, Any], ...]] = {
    (0.0, 0.02): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.8310001082106886e-4,
            "disk_rate": 0.9542368997552261,
            "origin_enclosed": False,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 7.159925074859145e-5,
            "disk_rate": 0.9542369444437695,
            "origin_enclosed": False,
            "verdict": "investigate",
        },
    ),
    (0.0, 2.0): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.92385206626124e-4,
            "disk_rate": 1.0013565109857867,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 7.333308483450632e-5,
            "disk_rate": 1.0013565590117444,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 96,
            "fov_max_iters": 240,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 2.072477977866651e-5,
            "disk_rate": 1.0013565500432922,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
    ),
    (1.0, 0.02): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.730617069768882e-4,
            "disk_rate": 0.9551645357445586,
            "origin_enclosed": False,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 1.3566156479233245e-4,
            "disk_rate": 0.9551646300876108,
            "origin_enclosed": False,
            "verdict": "investigate",
        },
    ),
    (1.0, 2.0): (
        {
            "n_angles": 32,
            "fov_max_iters": 120,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 3.820089498108155e-4,
            "disk_rate": 1.0023281370758816,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 64,
            "fov_max_iters": 180,
            "fov_n_restarts": 2,
            "supports_consistent": False,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": False,
            "max_support_residual": 1.3891133448470182e-4,
            "disk_rate": 1.002328238465557,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
        {
            "n_angles": 96,
            "fov_max_iters": 240,
            "fov_n_restarts": 2,
            "supports_consistent": True,
            "corroboration_attempted": True,
            "supports_converged": True,
            "supports_corroborated": True,
            "max_support_residual": 1.7135275935423258e-5,
            "disk_rate": 1.0023282296128921,
            "origin_enclosed": True,
            "verdict": "indeterminate",
        },
    ),
}


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON checkpoint atomically in its target directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _source_state_identity(state: jax.Array | np.ndarray) -> dict[str, Any]:
    """Return the exact identity of the state supplied to a diagnostic."""
    values = np.asarray(jax.device_get(state), dtype=np.float64)
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode())
    digest.update(values.dtype.str.encode())
    digest.update(values.tobytes(order="C"))
    return {
        "sha256": digest.hexdigest(),
        "shape": list(values.shape),
        "dtype": values.dtype.str,
    }


class StaleSourceStateError(RuntimeError):
    """A persisted source state was generated under a different configuration."""


def _pme_source_fingerprint(
    config: pme_breakdown.BreakdownConfig,
    m: int,
    front_case: int,
    target_halfwidth: float,
) -> dict[str, Any]:
    """Return every parameter that determines one PME source-state solve.

    The cache key is only ``m`` and the front case, so the cache is reusable
    across runs of the same study and misleading across any other.  This
    fingerprint records the rest: the regularization, the reference step size,
    the grid, the initial-state parameters and the library revision.  The
    geometry budget is deliberately absent because it does not enter the solve.
    """
    return {
        "schema": SOURCE_STATE_FINGERPRINT_SCHEMA,
        "study": "pme",
        "m": int(m),
        "front_case": int(front_case),
        "front_target_halfwidth": float(target_halfwidth),
        "epsilon": 0.0 if int(m) == 1 else float(config.epsilon),
        "state_dt": float(config.state_dt),
        "state_d0_kind": "frozen_bulk",
        "const_d0": float(config.const_d0),
        "initial_state": {
            "kind": "m1_heat_kernel" if int(m) == 1 else "barenblatt",
            "t0": float(config.t0),
        },
        "nx": int(config.nx),
        "x_min": float(config.x_min),
        "x_max": float(config.x_max),
        "newton_tol": float(config.newton_tol),
        "krylov_tol": float(config.krylov_tol),
        "max_newton_iters": int(config.max_newton_iters),
        "max_krylov_iters": int(config.max_krylov_iters),
        "moljax_version": str(moljax.__version__),
    }


def _porous_fisher_source_fingerprint(
    config: porous_fisher_conditioning.ReactionStudyConfig,
    reaction: float,
) -> dict[str, Any]:
    """Return every parameter that determines one reaction-axis source solve."""
    return {
        "schema": SOURCE_STATE_FINGERPRINT_SCHEMA,
        "study": "porous_fisher",
        "r": float(reaction),
        "epsilon": float(config.epsilon),
        "state_dt": float(config.state_dt),
        "state_d0_kind": "frozen_bulk",
        "const_d0": float(config.const_d0),
        "initial_state": {
            "kind": "porous_fisher_traveling_wave",
            "initial_time": float(config.initial_time),
            "reference_wave_r": float(config.reference_wave_r),
        },
        "nx": int(config.nx),
        "x_min": float(config.x_min),
        "x_max": float(config.x_max),
        "newton_tol": float(config.newton_tol),
        "krylov_tol": float(config.krylov_tol),
        "max_newton_iters": int(config.max_newton_iters),
        "max_krylov_iters": int(config.max_krylov_iters),
        "moljax_version": str(moljax.__version__),
    }


def _pme_record_fingerprint(
    config: pme_breakdown.BreakdownConfig,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Return the source-state fingerprint one PME record must have been built on."""
    front_case = int(record["front_case"])
    halfwidths = config.front_target_halfwidths
    if not 1 <= front_case <= len(halfwidths):
        raise RuntimeError(f"Unknown PME front case: {front_case}")
    return _pme_source_fingerprint(config, int(record["m"]), front_case, halfwidths[front_case - 1])


def _porous_fisher_record_fingerprint(
    config: porous_fisher_conditioning.ReactionStudyConfig,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Return the source-state fingerprint one reaction-axis record was built on."""
    return _porous_fisher_source_fingerprint(config, float(record["r"]))


def _atomic_save_array(path: Path, state: jax.Array) -> None:
    """Persist one source array atomically without serializing it through JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.asarray(jax.device_get(state), dtype=np.float64), allow_pickle=False)
    os.replace(temporary, path)


def _source_state_paths(
    checkpoint_dir: Path,
    study: str,
    source_key: str,
) -> tuple[Path, Path, Path]:
    """Return the artifact paths for one stable source-state key."""
    if study not in {"pme", "porous_fisher"}:
        raise ValueError(f"Unsupported source-state study: {study}")
    if not source_key.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Unsafe source-state key: {source_key!r}")
    relative = Path("source_states") / f"{study}_{source_key}.npy"
    array_path = checkpoint_dir / relative
    return relative, array_path, array_path.with_suffix(".json")


def _load_saved_source_state(
    checkpoint_dir: Path,
    study: str,
    source_key: str,
    fingerprint: dict[str, Any],
) -> tuple[jax.Array, dict[str, Any]]:
    """Load, hash-verify and configuration-verify one persisted source state.

    The array hash alone only proves that the file was not corrupted.  A
    manifest written under a different epsilon, reference step size, grid,
    initial state or library revision carries a different generation
    fingerprint, and reusing it would silently mix configurations, so it raises
    :class:`StaleSourceStateError` for the caller to regenerate or fail closed.
    """
    expected_relative, expected_array, manifest_path = _source_state_paths(
        checkpoint_dir, study, source_key
    )
    manifest = _load_checkpoint(manifest_path)
    if (
        manifest.get("study") != study
        or manifest.get("source_key") != source_key
        or manifest.get("relative_path") != str(expected_relative)
    ):
        raise RuntimeError(f"Source-state artifact identity mismatch: {manifest_path}")
    if manifest.get("schema") != SOURCE_STATE_ARTIFACT_SCHEMA:
        raise StaleSourceStateError(
            f"Source-state artifact schema is {manifest.get('schema')!r}, "
            f"not {SOURCE_STATE_ARTIFACT_SCHEMA!r}: {manifest_path}"
        )
    if manifest.get("generation_fingerprint") != fingerprint:
        raise StaleSourceStateError(
            f"Source-state artifact was generated under a different configuration: "
            f"{manifest_path}"
        )
    if not expected_array.is_file():
        raise RuntimeError(f"Missing source-state array: {expected_array}")
    values = np.load(expected_array, allow_pickle=False)
    state = jax.block_until_ready(jnp.asarray(values, dtype=jnp.float64))
    observed_identity = _source_state_identity(state)
    expected_identity = manifest.get("source_state_identity")
    if observed_identity != expected_identity:
        raise RuntimeError(f"Source-state artifact hash mismatch: {expected_array}")
    state_solver = dict(manifest.get("state_solver", {}))
    artifact = state_solver.get("source_state_artifact")
    if (
        not bool(state_solver.get("converged"))
        or state_solver.get("source_state_identity") != expected_identity
        or artifact is None
        or artifact.get("relative_path") != str(expected_relative)
        or artifact.get("source_state_identity") != expected_identity
        or artifact.get("generation_fingerprint") != fingerprint
    ):
        raise RuntimeError(f"Source-state provenance mismatch: {manifest_path}")
    return state, state_solver


def _persist_source_state(
    checkpoint_dir: Path,
    study: str,
    source_key: str,
    state: jax.Array,
    state_solver: dict[str, Any],
    fingerprint: dict[str, Any],
) -> tuple[jax.Array, dict[str, Any]]:
    """Save a converged source state, then reload its verified diagnostic array."""
    if not bool(state_solver.get("converged")):
        return state, state_solver
    relative, array_path, manifest_path = _source_state_paths(checkpoint_dir, study, source_key)
    identity = _source_state_identity(state)
    solver = deepcopy(state_solver)
    solver_identity = solver.get("source_state_identity")
    if solver_identity is not None and solver_identity != identity:
        raise RuntimeError(
            f"Solved {study} source state differs from its solver identity: {source_key}"
        )
    artifact = {
        "relative_path": str(relative),
        "source_state_identity": identity,
        "generation_fingerprint": fingerprint,
    }
    solver["source_state_identity"] = identity
    solver["source_state_artifact"] = artifact
    _atomic_save_array(array_path, state)
    _atomic_write(
        manifest_path,
        {
            "schema": SOURCE_STATE_ARTIFACT_SCHEMA,
            "study": study,
            "source_key": source_key,
            "relative_path": str(relative),
            "source_state_identity": identity,
            "generation_fingerprint": fingerprint,
            "state_solver": solver,
        },
    )
    return _load_saved_source_state(checkpoint_dir, study, source_key, fingerprint)


def _load_or_persist_source_state(
    checkpoint_dir: Path,
    study: str,
    source_key: str,
    fingerprint: dict[str, Any],
    solve: Callable[[], tuple[jax.Array, dict[str, Any]]],
) -> tuple[jax.Array, dict[str, Any]]:
    """Reuse an exact source artifact, or solve once and make it authoritative.

    An artifact whose generation fingerprint does not match the requested
    configuration is stale, not reusable, so it is regenerated in place.
    """
    _, _, manifest_path = _source_state_paths(checkpoint_dir, study, source_key)
    if manifest_path.is_file():
        try:
            return _load_saved_source_state(checkpoint_dir, study, source_key, fingerprint)
        except StaleSourceStateError as stale:
            print(f"regenerating stale source state: {stale}")
    state, state_solver = solve()
    return _persist_source_state(
        checkpoint_dir, study, source_key, state, state_solver, fingerprint
    )


def _load_record_source_state(
    checkpoint_dir: Path,
    study: str,
    record: dict[str, Any],
    fingerprint: dict[str, Any],
) -> tuple[jax.Array, dict[str, Any]]:
    """Load the exact persisted state named by a regenerated result record.

    This path fails closed.  A record whose stored artifact was generated under
    another configuration, or an artifact whose own fingerprint has since moved,
    is refused rather than regenerated: the record's published numbers were
    produced on the original state and cannot be reconciled with a new one.
    """
    solver_key = "reference_state_solver" if study == "pme" else "state_solver"
    try:
        stored_solver = record[solver_key]
        artifact = stored_solver["source_state_artifact"]
        relative = Path(artifact["relative_path"])
    except KeyError as error:
        raise RuntimeError(
            f"Record lacks persisted source-state provenance: {_record_key(study, record)}"
        ) from error
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError(f"Unsafe source-state artifact path: {relative}")
    filename = relative.name
    prefix = f"{study}_"
    suffix = ".npy"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        raise RuntimeError(f"Unexpected source-state artifact name: {relative}")
    source_key = filename[len(prefix) : -len(suffix)]
    if artifact.get("generation_fingerprint") != fingerprint:
        raise RuntimeError(
            "Record source-state fingerprint differs from the requested configuration: "
            f"{_record_key(study, record)}"
        )
    try:
        state, state_solver = _load_saved_source_state(
            checkpoint_dir, study, source_key, fingerprint
        )
    except StaleSourceStateError as stale:
        raise RuntimeError(
            f"Refusing to reassess {_record_key(study, record)} on a stale source state: {stale}"
        ) from stale
    if (
        stored_solver.get("source_state_identity") != state_solver["source_state_identity"]
        or artifact.get("source_state_identity") != state_solver["source_state_identity"]
    ):
        raise RuntimeError(
            f"Persisted source-state identity differs from record: {_record_key(study, record)}"
        )
    return state, state_solver


def _load_checkpoint(path: Path) -> dict[str, Any]:
    """Load one completed checkpoint or fail with its exact location."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing completed batch checkpoint: {path}") from error


def _pme_checkpoint_path(checkpoint_dir: Path, m: int) -> Path:
    """Return the checkpoint path for one independent PME exponent batch."""
    return checkpoint_dir / f"pme_m{m}.json"


def _porous_fisher_checkpoint_path(checkpoint_dir: Path, reaction: float) -> Path:
    """Return the checkpoint path for one independent reaction-strength batch."""
    return checkpoint_dir / f"porous_fisher_r{reaction:g}.json"


def _dependent_checkpoint_paths(checkpoint_dir: Path, study: str) -> tuple[Path, ...]:
    """Return every checkpoint derived from one study's assembled batches."""
    return (
        _resolution_checkpoint_path(checkpoint_dir, study),
        _bba5a94_audit_checkpoint_path(checkpoint_dir, study),
        _c1_option_a_checkpoint_path(checkpoint_dir, study),
        _current_main_audit_checkpoint_path(checkpoint_dir, study),
        _v121_recovery_checkpoint_path(checkpoint_dir, study),
    )


def _incompatible_batch_source_states(
    checkpoint_dir: Path,
    study: str,
    existing: dict[str, Any],
    record_fingerprint: Callable[[dict[str, Any]], dict[str, Any]],
) -> str | None:
    """Return why a completed batch is not reusable, or ``None`` if it is.

    A batch is complete only when every source state its records were measured
    on still exists under the current artifact schema with a generation
    fingerprint matching this batch's configuration.  Anything else is a batch
    whose numbers this run can neither reproduce nor verify: returning it lets
    assembly republish diagnostics that reassessment then refuses, so the
    caller rebuilds it instead.
    """
    solver_key = "reference_state_solver" if study == "pme" else "state_solver"
    records = existing.get("records")
    if not isinstance(records, list) or not records:
        return "the batch holds no records"
    verified: set[tuple[str, str]] = set()
    for record in records:
        key = _record_key(study, record)
        stored_solver = record.get(solver_key)
        artifact = (
            stored_solver.get("source_state_artifact")
            if isinstance(stored_solver, dict)
            else None
        )
        if not isinstance(artifact, dict):
            return f"{key} lacks persisted source-state provenance"
        try:
            fingerprint = record_fingerprint(record)
        except (KeyError, RuntimeError) as error:
            return f"{key} has no reproducible source-state fingerprint: {error}"
        if artifact.get("generation_fingerprint") != fingerprint:
            return f"{key} was measured on a differently configured source state"
        cached = (str(artifact.get("relative_path")), json.dumps(fingerprint, sort_keys=True))
        if cached in verified:
            continue
        try:
            _load_record_source_state(checkpoint_dir, study, record, fingerprint)
        except RuntimeError as error:
            return f"{key} has no current source-state artifact: {error}"
        verified.add(cached)
    return None


def _invalidate_batch(checkpoint_dir: Path, study: str, checkpoint: Path, reason: str) -> None:
    """Discard an incompatible batch checkpoint and everything derived from it."""
    print(f"invalidating batch checkpoint {checkpoint}: {reason}")
    checkpoint.unlink(missing_ok=True)
    checkpoint.with_suffix(".working.json").unlink(missing_ok=True)
    for dependent in _dependent_checkpoint_paths(checkpoint_dir, study):
        if dependent.is_file():
            print(f"invalidating dependent checkpoint: {dependent}")
            dependent.unlink()


def _run_pme_batch(checkpoint_dir: Path, m: int) -> Path:
    """Run and atomically checkpoint one PME exponent batch if needed.

    An existing checkpoint counts as complete only after its source states are
    validated against this configuration.  A batch written under an older
    schema, or one whose artifacts no longer match, is invalidated along with
    the checkpoints derived from it and rebuilt rather than returned.
    """
    config = pme_breakdown.BreakdownConfig(m_values=(m,))
    checkpoint = _pme_checkpoint_path(checkpoint_dir, m)
    if checkpoint.exists():
        existing = _load_checkpoint(checkpoint)
        if existing.get("m") != m:
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        if existing.get("schema") != PME_BATCH_SCHEMA:
            _invalidate_batch(
                checkpoint_dir,
                "pme",
                checkpoint,
                f"batch schema is {existing.get('schema')!r}, not {PME_BATCH_SCHEMA!r}",
            )
        else:
            incompatible = _incompatible_batch_source_states(
                checkpoint_dir,
                "pme",
                existing,
                lambda record: _pme_record_fingerprint(config, record),
            )
            if incompatible is None:
                print(f"checkpoint exists: {checkpoint}")
                return checkpoint
            _invalidate_batch(checkpoint_dir, "pme", checkpoint, incompatible)

    working_output = checkpoint.with_suffix(".working.json")
    started_at = perf_counter()

    def source_state_provider(
        initial_state: jax.Array,
        grid: pme_breakdown.NodeCenteredDirichletGrid,
        exponent: int,
        front_case: int,
        target_halfwidth: float,
        provider_config: pme_breakdown.BreakdownConfig,
    ) -> tuple[jax.Array, dict[str, Any]]:
        return _load_or_persist_source_state(
            checkpoint_dir,
            "pme",
            f"m{exponent}_front{front_case}",
            _pme_source_fingerprint(provider_config, exponent, front_case, target_halfwidth),
            lambda: pme_breakdown._solve_one_step(
                initial_state, grid, exponent, provider_config, "frozen_bulk"
            ),
        )

    report = pme_breakdown.run_breakdown_study(
        config._replace(output_path=str(working_output)),
        source_state_provider=source_state_provider,
    )
    working_output.unlink(missing_ok=True)
    payload = {
        "schema": PME_BATCH_SCHEMA,
        "m": m,
        "config": config._asdict(),
        "records": report["records"],
        "centering": report["centering"],
        "runtime_seconds": report["runtime_seconds"],
        "batch_wall_seconds": perf_counter() - started_at,
    }
    _atomic_write(checkpoint, payload)
    if report["source_state_status"]["source_state_unusable_records"]:
        raise RuntimeError(
            f"PME m={m} has unconverged reference states; conditioning was not assessed"
        )
    print(f"completed m={m}: records={len(payload['records'])} checkpoint={checkpoint}")
    return checkpoint


def _run_porous_fisher_batch(checkpoint_dir: Path, reaction: float) -> Path:
    """Run and atomically checkpoint one reaction-strength batch if needed.

    As in the PME batches, a completed checkpoint is reused only when every
    source state it depends on still carries the current artifact schema and a
    fingerprint matching this configuration.
    """
    config = porous_fisher_conditioning.ReactionStudyConfig(reaction_values=(reaction,))
    checkpoint = _porous_fisher_checkpoint_path(checkpoint_dir, reaction)
    if checkpoint.exists():
        existing = _load_checkpoint(checkpoint)
        if existing.get("reaction") != reaction:
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        if existing.get("schema") != POROUS_FISHER_BATCH_SCHEMA:
            _invalidate_batch(
                checkpoint_dir,
                "porous_fisher",
                checkpoint,
                f"batch schema is {existing.get('schema')!r}, "
                f"not {POROUS_FISHER_BATCH_SCHEMA!r}",
            )
        else:
            incompatible = _incompatible_batch_source_states(
                checkpoint_dir,
                "porous_fisher",
                existing,
                lambda record: _porous_fisher_record_fingerprint(config, record),
            )
            if incompatible is None:
                print(f"checkpoint exists: {checkpoint}")
                return checkpoint
            _invalidate_batch(checkpoint_dir, "porous_fisher", checkpoint, incompatible)

    working_output = checkpoint.with_suffix(".working.json")
    started_at = perf_counter()

    def source_state_provider(
        initial_state: jax.Array,
        grid: porous_fisher_conditioning.NodeCenteredDirichletGrid,
        source_reaction: float,
        provider_config: porous_fisher_conditioning.ReactionStudyConfig,
    ) -> tuple[jax.Array, dict[str, Any]]:
        return _load_or_persist_source_state(
            checkpoint_dir,
            "porous_fisher",
            f"r{source_reaction:g}",
            _porous_fisher_source_fingerprint(provider_config, source_reaction),
            lambda: porous_fisher_conditioning._advance_to_visited_state(
                initial_state, grid, r=source_reaction, config=provider_config
            ),
        )

    report = porous_fisher_conditioning.run_reaction_study(
        config._replace(output_path=str(working_output)),
        source_state_provider=source_state_provider,
    )
    working_output.unlink(missing_ok=True)
    payload = {
        "schema": POROUS_FISHER_BATCH_SCHEMA,
        "reaction": reaction,
        "config": config._asdict(),
        "records": report["records"],
        "runtime_seconds": report["runtime_seconds"],
        "batch_wall_seconds": perf_counter() - started_at,
    }
    _atomic_write(checkpoint, payload)
    if report["source_state_status"]["source_state_unusable_records"]:
        raise RuntimeError(
            f"Porous--Fisher r={reaction:g} has an unconverged reference state; "
            "conditioning was not assessed"
        )
    print(
        "completed "
        f"reaction={reaction:g}: records={len(payload['records'])} checkpoint={checkpoint}"
    )
    return checkpoint


def _assemble_pme(checkpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    """Assemble completed PME batches into the existing canonical report schema."""
    config = pme_breakdown.BreakdownConfig(output_path=str(output_path))
    records: list[dict[str, Any]] = []
    runtimes: list[float] = []
    centering: dict[str, Any] | None = None
    for m in config.m_values:
        checkpoint = _load_checkpoint(_pme_checkpoint_path(checkpoint_dir, m))
        if checkpoint.get("schema") != PME_BATCH_SCHEMA or checkpoint.get("m") != m:
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        records.extend(checkpoint["records"])
        runtimes.append(float(checkpoint["runtime_seconds"]))
        if centering is None:
            centering = checkpoint["centering"]

    if centering is None:
        raise RuntimeError("No PME batch checkpoints were available")
    decision = pme_breakdown._verdict_on_decision_procedure(records)
    dynamic_range = decision["identity_iteration_dynamic_range"]
    if not dynamic_range["meets_tenfold_gate"]:
        raise RuntimeError(
            "Identity GMRES dynamic-range gate failed: "
            f"min={dynamic_range['min']}, max={dynamic_range['max']}, "
            f"ratio={dynamic_range['ratio']}"
        )
    report = {
        "description": (
            "Experimental PME conditioning study. It evaluates a preconditioner decision "
            "procedure; it does not claim to fix stiffness degradation."
        ),
        "config": config._asdict(),
        "metric": {
            "accuracy": "not evaluated in this conditioning-only stress study",
            "timing": "total runtime only",
        },
        "runtime_seconds": sum(runtimes),
        "source_state_status": {
            "all_converged": True,
            "converged_records": len(records),
            "source_state_unusable_records": 0,
        },
        "gmres_measurement_note": (
            "actual_gmres is an explicit residual-history count for the fixed system "
            "P^-1 J delta = P^-1 (-R), not NKStats.lin_iters."
        ),
        "state_schedule_note": (
            "Each state is produced by the stable reference backward-Euler step size state_dt "
            "with the frozen_bulk preconditioner.  Each analysis_dt then defines a separate "
            "fixed linearized backward-Euler system at that genuinely visited state."
        ),
        "centering": centering,
        "regime_claim": pme_breakdown._regime_claim(records),
        "rank_claim": pme_breakdown._rank_claim(records),
        "predictor_quality": pme_breakdown._predictor_quality(records),
        "correlation": pme_breakdown._correlation_pairs(records),
        "regime_map": pme_breakdown._regime_map(records),
        "verdict_on_decision_procedure": decision,
        "records": records,
    }
    _atomic_write(output_path, report)
    print(f"assembled PME: records={len(records)} output={output_path}")
    return report


def _assemble_porous_fisher(checkpoint_dir: Path, output_path: Path) -> dict[str, Any]:
    """Assemble completed reaction batches into the existing canonical report schema."""
    config = porous_fisher_conditioning.ReactionStudyConfig(output_path=str(output_path))
    records: list[dict[str, Any]] = []
    runtimes: list[float] = []
    for reaction in config.reaction_values:
        checkpoint = _load_checkpoint(_porous_fisher_checkpoint_path(checkpoint_dir, reaction))
        if (
            checkpoint.get("schema") != POROUS_FISHER_BATCH_SCHEMA
            or checkpoint.get("reaction") != reaction
        ):
            raise RuntimeError(f"Checkpoint identity mismatch: {checkpoint}")
        records.extend(checkpoint["records"])
        runtimes.append(float(checkpoint["runtime_seconds"]))

    identity_records = [record for record in records if record["d0_kind"] == "identity"]
    identity_iterations = [
        float(record["actual_gmres"]["iterations"]) for record in identity_records
    ]
    minimum = min(identity_iterations)
    maximum = max(identity_iterations)
    ratio = float("inf") if minimum == 0.0 and maximum > 0.0 else maximum / minimum
    if ratio < 10.0:
        raise RuntimeError(
            f"Identity GMRES dynamic-range gate failed: min={minimum}, max={maximum}, ratio={ratio}"
        )
    report = {
        "description": (
            "Experimental Porous--Fisher conditioning study. It tests whether diagnostics flag "
            "hard states when a logistic reaction remains outside the diffusion preconditioner."
        ),
        "config": config._asdict(),
        "runtime_seconds": sum(runtimes),
        "source_state_status": {
            "all_converged": True,
            "converged_records": len(records),
            "source_state_unusable_records": 0,
        },
        "physical_model": {
            "equation": "u_t = d_xx(Phi_epsilon(u)) + r*u*(1-u)",
            "diffusivity": "D_epsilon(u)=2*sqrt(u**2 + epsilon**2)",
            "preconditioner_scope": "diffusion-only D0 Helmholtz; reaction is unpreconditioned",
        },
        "gmres_measurement_note": (
            "actual_gmres is the explicit residual-history count for P^-1 J delta=P^-1(-R), "
            "not NKStats.lin_iters."
        ),
        "regime_claim": porous_fisher_conditioning._regime_claim(records),
        "verdict_on_decision_procedure": porous_fisher_conditioning._verdict_on_decision_procedure(
            records
        ),
        "reaction_effect": porous_fisher_conditioning._reaction_effect(records),
        "records": records,
    }
    _atomic_write(output_path, report)
    print(f"assembled Porous--Fisher: records={len(records)} output={output_path}")
    return report


def _base_output_path(study: str) -> Path:
    """Return the assembled base report path for one conditioning study."""
    if study == "pme":
        return Path(pme_breakdown.BreakdownConfig().output_path)
    return Path(porous_fisher_conditioning.ReactionStudyConfig().output_path)


def _record_key(study: str, record: dict[str, Any]) -> str:
    """Return a stable key for one record in an assembled result report."""
    if study == "pme":
        return (
            f"m={int(record['m'])};front={int(record['front_case'])};"
            f"dt={float(record['analysis_dt']):.17g};d0={record['d0_kind']}"
        )
    return (
        f"r={float(record['r']):.17g};dt={float(record['analysis_dt']):.17g};d0={record['d0_kind']}"
    )


def _require_converged_state(
    study: str,
    record: dict[str, Any],
    state_solver: dict[str, Any],
) -> None:
    """Fail closed before assessing a source state that did not converge."""
    key = _record_key(study, record)
    if not bool(state_solver["converged"]):
        raise RuntimeError(
            f"Refusing to assess {key}: source-state solve did not converge "
            f"(residual={state_solver['final_residual_l2']})"
        )


def _geometry_snapshot(
    record: dict[str, Any],
    *,
    n_angles: int,
    fov_max_iters: int,
    fov_n_restarts: int,
) -> dict[str, Any]:
    """Keep the bounded geometry data needed to audit one ladder attempt."""
    return {
        "n_angles": n_angles,
        "fov_max_iters": fov_max_iters,
        "fov_n_restarts": fov_n_restarts,
        "supports_consistent": bool(record["supports_consistent"]),
        "corroboration_attempted": bool(record["corroboration_attempted"]),
        "supports_converged": bool(record["supports_converged"]),
        "supports_corroborated": bool(record["supports_corroborated"]),
        "max_support_residual": float(record["max_support_residual"]),
        "disk_rate": float(record["disk_rate"]),
        "epsilon_zero": float(record["epsilon_zero"]),
        "epsilon_zero_reduced_arnoldi": (
            None
            if record.get("epsilon_zero_reduced_arnoldi") is None
            else float(record["epsilon_zero_reduced_arnoldi"])
        ),
        "full_operator_epsilon_zero": (
            None
            if record.get("full_operator_epsilon_zero") is None
            else float(record["full_operator_epsilon_zero"])
        ),
        "full_operator_epsilon_zero_seconds": (
            None
            if record.get("full_operator_epsilon_zero_seconds") is None
            else float(record["full_operator_epsilon_zero_seconds"])
        ),
        "epsilon_zero_full_operator_evidence": bool(
            record.get("epsilon_zero_full_operator_evidence", False)
        ),
        "verdict_reason": record.get("verdict_reason"),
        "arnoldi_k_requested": record.get("arnoldi_k_requested"),
        "arnoldi_k_achieved": record.get("arnoldi_k_achieved"),
        "arnoldi_breakdown": record.get("arnoldi_breakdown"),
        "arnoldi_residual_norm": record.get("arnoldi_residual_norm"),
        "origin_enclosed": bool(record["origin_enclosed"]),
        "n_right_real_outliers": (
            None
            if record["n_right_real_outliers"] is None
            else int(record["n_right_real_outliers"])
        ),
        "predicted_gmres_factor": (
            None
            if record["predicted_gmres_factor"] is None
            else float(record["predicted_gmres_factor"])
        ),
        "verdict": str(record["verdict"]),
    }


def _is_certified(snapshot: dict[str, Any]) -> bool:
    """Return whether a resolved two-restart geometry attempt is certified."""
    return bool(snapshot["supports_consistent"] and snapshot["corroboration_attempted"])


def _legacy_snapshot_to_current(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Translate persisted pre-bba5a94 certification fields without changing readings."""
    updated = dict(snapshot)
    legacy_certified = updated.pop("geometry_certified", None)
    if "supports_consistent" not in updated:
        if legacy_certified is None:
            raise RuntimeError("Persisted geometry snapshot lacks a certification field")
        updated["supports_consistent"] = bool(legacy_certified)
    updated.setdefault("corroboration_attempted", updated["fov_n_restarts"] >= 2)
    return updated


def _resolution_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the atomic per-record geometry-resolution checkpoint path."""
    return checkpoint_dir / f"{study}_geometry_resolution_v1.json"


def _base_digest(path: Path) -> str:
    """Return a stable identity for the already-regenerated base result report."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_base_report(study: str) -> tuple[Path, dict[str, Any]]:
    """Load the local 16/60/2 report without any numerical work."""
    path = _base_output_path(study)
    try:
        return path, json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing assembled base result report: {path}") from error


def _load_resolution_checkpoint(
    checkpoint_dir: Path,
    study: str,
    base_path: Path,
) -> dict[str, Any]:
    """Load or initialize one resumable per-record resolution checkpoint."""
    path = _resolution_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(base_path)
    if not path.exists():
        return {
            "schema": "stage2_conditioning_geometry_resolution_v1",
            "study": study,
            "base_result": str(base_path),
            "base_sha256": digest,
            "resolved": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != "stage2_conditioning_geometry_resolution_v1"
        or payload.get("study") != study
        or payload.get("base_result") != str(base_path)
        or payload.get("base_sha256") != digest
    ):
        raise RuntimeError(f"Resolution checkpoint identity mismatch: {path}")
    return payload


def _priority_reaction_resolution(record: dict[str, Any]) -> dict[str, Any] | None:
    """Disable reuse of pre-C1 priority probes during a full regeneration."""
    del record
    return None


def _assess_pme_record(
    checkpoint_dir: Path,
    record: dict[str, Any],
    budget: tuple[int, int, int],
    *,
    full_operator_epsilon_evidence: bool = False,
) -> dict[str, Any]:
    """Load one persisted PME state and assess it at one geometry budget."""
    n_angles, fov_max_iters, fov_n_restarts = budget
    config = pme_breakdown.BreakdownConfig(
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    grid = pme_breakdown.NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    m = int(record["m"])
    state, state_solver = _load_record_source_state(
        checkpoint_dir, "pme", record, _pme_record_fingerprint(config, record)
    )
    _require_converged_state("pme", record, state_solver)
    diagnostic = pme_breakdown.assess_pme_state(
        state,
        grid,
        float(m),
        float(record["analysis_dt"]),
        0.0 if m == 1 else config.epsilon,
        str(record["d0_kind"]),
        const_value=config.const_d0,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_residual_tolerance=config.fov_residual_tolerance,
        fov_n_restarts=fov_n_restarts,
        arnoldi_steps=config.arnoldi_steps,
        full_operator_epsilon_evidence=full_operator_epsilon_evidence,
        seed=(
            20260900
            + 1000 * m
            + 10 * int(record["front_case"])
            + config.d0_kinds.index(str(record["d0_kind"]))
        ),
    )
    return _geometry_snapshot(
        diagnostic,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )


def _assess_porous_fisher_record(
    checkpoint_dir: Path,
    record: dict[str, Any],
    budget: tuple[int, int, int],
    *,
    full_operator_epsilon_evidence: bool = False,
) -> dict[str, Any]:
    """Load one persisted reaction-axis state and assess it at one geometry budget."""
    n_angles, fov_max_iters, fov_n_restarts = budget
    config = porous_fisher_conditioning.ReactionStudyConfig(
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    grid = porous_fisher_conditioning.NodeCenteredDirichletGrid.uniform(
        config.nx, config.x_min, config.x_max
    )
    reaction = float(record["r"])
    state, state_solver = _load_record_source_state(
        checkpoint_dir,
        "porous_fisher",
        record,
        _porous_fisher_record_fingerprint(config, record),
    )
    _require_converged_state("porous_fisher", record, state_solver)
    d0_kind = str(record["d0_kind"])
    diagnostic = porous_fisher_conditioning.assess_porous_fisher_state(
        state,
        grid,
        r=reaction,
        dt=float(record["analysis_dt"]),
        epsilon=config.epsilon,
        d0_kind=d0_kind,
        const_value=config.const_d0,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_residual_tolerance=config.fov_residual_tolerance,
        fov_n_restarts=fov_n_restarts,
        arnoldi_steps=config.arnoldi_steps,
        full_operator_epsilon_evidence=full_operator_epsilon_evidence,
        seed=(
            20260880
            + 1000 * int(100 * reaction)
            + 10 * int(100 * float(record["analysis_dt"]))
            + config.d0_kinds.index(d0_kind)
        ),
    )
    return _geometry_snapshot(
        diagnostic,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )


def _c1_option_a_measure_pme_record(
    checkpoint_dir: Path,
    record: dict[str, Any],
    budget: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one persisted PME state and retain its final-budget work evidence."""

    n_angles, fov_max_iters, fov_n_restarts = budget
    config = pme_breakdown.BreakdownConfig(
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    grid = pme_breakdown.NodeCenteredDirichletGrid.uniform(config.nx, config.x_min, config.x_max)
    m = int(record["m"])
    state, state_solver = _load_record_source_state(
        checkpoint_dir, "pme", record, _pme_record_fingerprint(config, record)
    )
    _require_converged_state("pme", record, state_solver)
    epsilon = 0.0 if m == 1 else config.epsilon
    d0_kind = str(record["d0_kind"])
    diagnostic = pme_breakdown.assess_pme_state(
        state,
        grid,
        float(m),
        float(record["analysis_dt"]),
        epsilon,
        d0_kind,
        const_value=config.const_d0,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_residual_tolerance=config.fov_residual_tolerance,
        fov_n_restarts=fov_n_restarts,
        arnoldi_steps=config.arnoldi_steps,
        seed=(
            20260900 + 1000 * m + 10 * int(record["front_case"]) + config.d0_kinds.index(d0_kind)
        ),
    )
    actual_gmres = pme_breakdown.measure_gmres_iterations(
        state,
        grid,
        float(m),
        float(record["analysis_dt"]),
        epsilon,
        d0_kind,
        tol=config.krylov_tol,
        max_iters=config.max_krylov_iters,
        const_value=config.const_d0,
    )
    snapshot = _geometry_snapshot(
        diagnostic,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    return snapshot, {
        "d0_used": diagnostic["d0"],
        "sigma": float(diagnostic["d0"] * float(record["analysis_dt"]) / grid.dx**2),
        "adjoint_identity": diagnostic["adjoint_error"],
        "adjoint_tolerance": diagnostic["adjoint_tolerance"],
        "rates": diagnostic["rates"],
        "actual_gmres": actual_gmres,
        "front_max_gradient": float(
            jnp.max(jnp.abs(jnp.diff(pme_breakdown.interior_values(state, grid)))) / grid.dx
        ),
        "reference_state_solver": dict(state_solver),
    }


def _c1_option_a_measure_porous_fisher_record(
    checkpoint_dir: Path,
    record: dict[str, Any],
    budget: tuple[int, int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load one persisted Porous--Fisher state and retain final-budget work evidence."""
    n_angles, fov_max_iters, fov_n_restarts = budget
    config = porous_fisher_conditioning.ReactionStudyConfig(
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    grid = porous_fisher_conditioning.NodeCenteredDirichletGrid.uniform(
        config.nx, config.x_min, config.x_max
    )
    reaction = float(record["r"])
    state, state_solver = _load_record_source_state(
        checkpoint_dir,
        "porous_fisher",
        record,
        _porous_fisher_record_fingerprint(config, record),
    )
    _require_converged_state("porous_fisher", record, state_solver)
    d0_kind = str(record["d0_kind"])
    diagnostic = porous_fisher_conditioning.assess_porous_fisher_state(
        state,
        grid,
        r=reaction,
        dt=float(record["analysis_dt"]),
        epsilon=config.epsilon,
        d0_kind=d0_kind,
        const_value=config.const_d0,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_residual_tolerance=config.fov_residual_tolerance,
        fov_n_restarts=fov_n_restarts,
        arnoldi_steps=config.arnoldi_steps,
        seed=(
            20260880
            + 1000 * int(100 * reaction)
            + 10 * int(100 * float(record["analysis_dt"]))
            + config.d0_kinds.index(d0_kind)
        ),
    )
    actual_gmres = porous_fisher_conditioning.measure_porous_fisher_gmres_iterations(
        state,
        grid,
        r=reaction,
        dt=float(record["analysis_dt"]),
        epsilon=config.epsilon,
        d0_kind=d0_kind,
        tol=config.krylov_tol,
        max_iters=config.max_krylov_iters,
        const_value=config.const_d0,
    )
    snapshot = _geometry_snapshot(
        diagnostic,
        n_angles=n_angles,
        fov_max_iters=fov_max_iters,
        fov_n_restarts=fov_n_restarts,
    )
    return snapshot, {
        "d0_used": diagnostic["d0"],
        "sigma": float(diagnostic["d0"] * float(record["analysis_dt"]) / grid.dx**2),
        "adjoint_identity": diagnostic["adjoint_error"],
        "adjoint_tolerance": diagnostic["adjoint_tolerance"],
        "rates": diagnostic["rates"],
        "actual_gmres": actual_gmres,
        "state_solver": dict(state_solver),
    }


def _resolve_record(
    checkpoint_dir: Path,
    study: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Run the bounded ladder for one previously uncertified record."""
    assessor = _assess_pme_record if study == "pme" else _assess_porous_fisher_record
    attempts: list[dict[str, Any]] = []
    for budget in GEOMETRY_LADDER:
        attempt = assessor(checkpoint_dir, record, budget)
        attempts.append(attempt)
        if _is_certified(attempt):
            return {
                "status": "CERTIFIED",
                "source": "escalation",
                "attempts": attempts,
                "final": attempt,
                "final_category": f"CERTIFIED_{attempt['verdict'].upper()}",
                "final_verdict": attempt["verdict"],
            }
    return {
        "status": "UNCERTIFIED_AT_CAP",
        "source": "escalation",
        "attempts": attempts,
        "final": attempts[-1],
        "final_category": "UNCERTIFIED_AT_CAP",
        "final_verdict": None,
    }


def _resolve_one_record(checkpoint_dir: Path, study: str, key: str) -> None:
    """Resolve one eligible record and atomically checkpoint its whole ladder."""
    base_path, report = _load_base_report(study)
    records = {_record_key(study, record): record for record in report["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} resolution key: {key}") from error
    if _is_certified(_geometry_snapshot(record, **BASE_GEOMETRY_BUDGET)):
        raise RuntimeError(f"Record is already certified at the base budget: {key}")
    if _priority_reaction_resolution(record) is not None:
        raise RuntimeError(f"Record was already resolved by the priority probe: {key}")
    checkpoint = _load_resolution_checkpoint(checkpoint_dir, study, base_path)
    if key in checkpoint["resolved"]:
        print(f"resolution checkpoint exists: {key}")
        return
    started_at = perf_counter()
    resolution = _resolve_record(checkpoint_dir, study, record)
    resolution["wall_seconds"] = perf_counter() - started_at
    checkpoint["resolved"][key] = resolution
    _atomic_write(_resolution_checkpoint_path(checkpoint_dir, study), checkpoint)
    print(
        f"resolved {key}: category={resolution['final_category']} "
        f"attempts={len(resolution['attempts'])} "
        f"checkpoint={_resolution_checkpoint_path(checkpoint_dir, study)}"
    )


def _resolution_metadata(
    study: str,
    record: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Return final, auditable geometry metadata for one base-report record."""
    base = _geometry_snapshot(record, **BASE_GEOMETRY_BUDGET)
    if _is_certified(base):
        return {
            "status": "CERTIFIED",
            "source": "base",
            "attempts": [base],
            "final": base,
            "final_category": f"CERTIFIED_{base['verdict'].upper()}",
            "final_verdict": base["verdict"],
        }
    priority = _priority_reaction_resolution(record)
    if priority is not None:
        return priority
    key = _record_key(study, record)
    try:
        return checkpoint["resolved"][key]
    except KeyError as error:
        raise RuntimeError(f"Missing completed resolution checkpoint: {key}") from error


def _attach_final_resolution(
    study: str,
    report: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Attach bounded final categories without losing the base measurement fields."""
    final_records: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    budget_counts: dict[str, int] = {}
    cap_corroboration_failures = 0
    cap_residual_failures = 0
    for record in report["records"]:
        updated = dict(record)
        resolution = _resolution_metadata(study, record, checkpoint)
        final = resolution["final"]
        final_budget = f"{final['n_angles']}/{final['fov_max_iters']}/{final['fov_n_restarts']}"
        _replace_geometry_fields(updated, final)
        updated["geometry_budget"] = {
            "n_angles": final["n_angles"],
            "fov_max_iters": final["fov_max_iters"],
            "fov_n_restarts": final["fov_n_restarts"],
        }
        if study == "pme":
            updated["predicted_iterations_from_envelope"] = (
                pme_breakdown.predicted_iterations_from_envelope(
                    float(final["disk_rate"]),
                    tol=float(report["config"]["krylov_tol"]),
                )
            )
        updated["geometry_resolution"] = resolution
        updated["final_verdict"] = resolution["final_verdict"]
        updated["final_category"] = resolution["final_category"]
        category_counts[resolution["final_category"]] = (
            category_counts.get(resolution["final_category"], 0) + 1
        )
        if resolution["status"] == "CERTIFIED":
            budget_counts[final_budget] = budget_counts.get(final_budget, 0) + 1
        else:
            cap_corroboration_failures += not bool(final["supports_corroborated"])
            cap_residual_failures += not bool(final["supports_converged"])
        final_records.append(updated)
    updated_report = dict(report)
    updated_report["records"] = final_records
    updated_report["geometry_resolution"] = {
        "description": (
            "A record has a final certified category only after support convergence and "
            "two-restart corroboration.  Records still uncertified at 96/240/2 are retained "
            "as UNCERTIFIED_AT_CAP rather than interpreted as origin-enclosure evidence."
        ),
        "base_budget": BASE_GEOMETRY_BUDGET,
        "escalation_ladder": [
            {
                "n_angles": n_angles,
                "fov_max_iters": fov_max_iters,
                "fov_n_restarts": fov_n_restarts,
            }
            for n_angles, fov_max_iters, fov_n_restarts in GEOMETRY_LADDER
        ],
        "final_category_counts": category_counts,
        "certifying_budget_counts": budget_counts,
        "uncertified_at_cap_failures": {
            "corroboration_failed": cap_corroboration_failures,
            "support_not_converged": cap_residual_failures,
        },
    }
    if study == "pme":
        updated_report["regime_claim"] = pme_breakdown._regime_claim(final_records)
        updated_report["rank_claim"] = pme_breakdown._rank_claim(final_records)
        updated_report["predictor_quality"] = pme_breakdown._predictor_quality(final_records)
        updated_report["correlation"] = pme_breakdown._correlation_pairs(final_records)
        updated_report["regime_map"] = pme_breakdown._regime_map(final_records)
        updated_report["verdict_on_decision_procedure"] = (
            pme_breakdown._verdict_on_decision_procedure(final_records)
        )
    else:
        updated_report["regime_claim"] = porous_fisher_conditioning._regime_claim(final_records)
        updated_report["verdict_on_decision_procedure"] = (
            porous_fisher_conditioning._verdict_on_decision_procedure(final_records)
        )
        updated_report["reaction_effect"] = porous_fisher_conditioning._reaction_effect(
            final_records
        )
    return updated_report


def _assemble_final_resolution(checkpoint_dir: Path, study: str, output_path: Path) -> None:
    """Write one result report with final geometry-resolution metadata attached."""
    base_path, report = _load_base_report(study)
    checkpoint = _load_resolution_checkpoint(checkpoint_dir, study, base_path)
    final_report = _attach_final_resolution(study, report, checkpoint)
    _atomic_write(output_path, final_report)
    print(
        f"assembled final {study} resolution: records={len(final_report['records'])} output={output_path}"
    )


def _bba5a94_audit_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the external checkpoint for one current-main verdict audit."""
    return checkpoint_dir / f"{study}_bba5a94_verdict_audit_v1.json"


def _recorded_final_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    """Return the recorded terminal ladder attempt for one assembled record."""
    try:
        return dict(record["geometry_resolution"]["final"])
    except KeyError as error:
        raise RuntimeError(f"Record lacks final geometry-resolution metadata: {record}") from error


def _recorded_final_budget(record: dict[str, Any]) -> tuple[int, int, int]:
    """Return the already-recorded budget; this refresh never searches the ladder."""
    final = _recorded_final_snapshot(record)
    return (
        int(final["n_angles"]),
        int(final["fov_max_iters"]),
        int(final["fov_n_restarts"]),
    )


def _load_bba5a94_audit_checkpoint(
    checkpoint_dir: Path,
    study: str,
    result_path: Path,
) -> dict[str, Any]:
    """Load or initialize the external per-record current-main audit checkpoint."""
    path = _bba5a94_audit_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(result_path)
    if not path.exists():
        return {
            "schema": BBA5A94_AUDIT_SCHEMA,
            "study": study,
            "result_path": str(result_path),
            "result_sha256": digest,
            "audited": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != BBA5A94_AUDIT_SCHEMA
        or payload.get("study") != study
        or payload.get("result_path") != str(result_path)
        or payload.get("result_sha256") != digest
    ):
        raise RuntimeError(f"Current-main audit checkpoint identity mismatch: {path}")
    return payload


def _category_from_snapshot(snapshot: dict[str, Any]) -> str:
    """Map a current certification observation to the existing final-category contract."""
    if not _is_certified(snapshot):
        return "UNCERTIFIED_AT_CAP"
    return f"CERTIFIED_{snapshot['verdict'].upper()}"


def _recorded_current_snapshot(record: dict[str, Any]) -> dict[str, Any]:
    """Return a normalized recorded terminal snapshot for a current-main comparison."""
    expected = _legacy_snapshot_to_current(_recorded_final_snapshot(record))
    expected.setdefault("epsilon_zero", float(record["epsilon_zero"]))
    expected.setdefault("n_right_real_outliers", record["n_right_real_outliers"])
    expected.setdefault("predicted_gmres_factor", record["predicted_gmres_factor"])
    return expected


def _verify_recorded_budget(record: dict[str, Any], observed: dict[str, Any], key: str) -> None:
    """Fail closed if an audit did not use the record's stored terminal budget."""
    expected_budget = _recorded_final_budget(record)
    observed_budget = (
        observed["n_angles"],
        observed["fov_max_iters"],
        observed["fov_n_restarts"],
    )
    if observed_budget != expected_budget:
        raise RuntimeError(
            f"Certification refresh used the wrong budget for {key}: "
            f"expected={expected_budget}, observed={observed_budget}"
        )
    if not observed["corroboration_attempted"]:
        raise RuntimeError(f"Current-main audit did not attempt corroboration: {key}")


def _verdict_shift_reason(
    record: dict[str, Any], expected: dict[str, Any], observed: dict[str, Any]
) -> str | None:
    """Return a closed operational label for a changed current-main verdict."""
    recorded_category = str(record["final_category"])
    observed_category = _category_from_snapshot(observed)
    if recorded_category != observed_category:
        if recorded_category == "UNCERTIFIED_AT_CAP":
            return "support_certification_recovered"
        if observed_category == "UNCERTIFIED_AT_CAP":
            return "support_certification_lost"
    if observed["verdict"] == expected["verdict"]:
        return None
    if observed["n_right_real_outliers"] is None:
        return "domain_guard_null_outliers"
    if observed["origin_enclosed"] != expected["origin_enclosed"]:
        return "origin_enclosure_change"
    if observed["n_right_real_outliers"] != record["n_right_real_outliers"]:
        return "new_outlier_gate"
    return "geometry_verdict_shift"


def _current_main_shift_reason(
    record: dict[str, Any], expected: dict[str, Any], observed: dict[str, Any]
) -> str | None:
    """Classify a changed final reading under the current conditioning semantics."""
    if (
        expected["verdict"] == "adequate"
        and observed["verdict"] == "provisional"
        and not observed["epsilon_zero_full_operator_evidence"]
    ):
        return "reduced_arnoldi_epsilon_zero_coverage_gate"
    if (
        expected["supports_consistent"] != observed["supports_consistent"]
        or expected["supports_corroborated"] != observed["supports_corroborated"]
    ):
        return "operator_tied_restart_seed_corroboration_change"
    return _verdict_shift_reason(record, expected, observed)


def _audit_one_bba5a94_record(checkpoint_dir: Path, study: str, key: str) -> None:
    """Reassess one stored terminal budget and atomically checkpoint its current reading."""
    result_path, report = _load_base_report(study)
    records = {_record_key(study, record): record for record in report["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} current-main audit key: {key}") from error
    checkpoint = _load_bba5a94_audit_checkpoint(checkpoint_dir, study, result_path)
    if key in checkpoint["audited"]:
        print(f"current-main audit checkpoint exists: {key}")
        return
    assessor = _assess_pme_record if study == "pme" else _assess_porous_fisher_record
    started_at = perf_counter()
    observed = assessor(checkpoint_dir, record, _recorded_final_budget(record))
    _verify_recorded_budget(record, observed, key)
    expected = _recorded_current_snapshot(record)
    reason = _verdict_shift_reason(record, expected, observed)
    checkpoint["audited"][key] = {
        "recorded": expected,
        "observed": observed,
        "recorded_final_category": str(record["final_category"]),
        "current_final_category": _category_from_snapshot(observed),
        "verdict_changed": reason is not None,
        "change_reason": reason,
        "wall_seconds": perf_counter() - started_at,
    }
    path = _bba5a94_audit_checkpoint_path(checkpoint_dir, study)
    _atomic_write(path, checkpoint)
    print(
        f"audited {key}: category={_category_from_snapshot(observed)} "
        f"changed={reason is not None} reason={reason} checkpoint={path}"
    )


def _list_pending_bba5a94_audit(checkpoint_dir: Path, study: str) -> None:
    """Print every terminal record that still needs its current-main audit."""
    result_path, report = _load_base_report(study)
    checkpoint = _load_bba5a94_audit_checkpoint(checkpoint_dir, study, result_path)
    pending = [
        _record_key(study, record)
        for record in report["records"]
        if _record_key(study, record) not in checkpoint["audited"]
    ]
    print(json.dumps(pending, indent=2))


def _migrate_resolution_metadata(
    resolution: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Replace obsolete certification fields with the current terminal observation."""
    updated = deepcopy(resolution)
    updated["attempts"] = [_legacy_snapshot_to_current(attempt) for attempt in updated["attempts"]]
    final = dict(audit["observed"])
    category = str(audit["current_final_category"])
    updated["final"] = final
    updated["status"] = "CERTIFIED" if _is_certified(final) else "UNCERTIFIED_AT_CAP"
    updated["final_category"] = category
    updated["final_verdict"] = final["verdict"] if _is_certified(final) else None
    updated["bba5a94_audit"] = {
        "recorded_final_category": audit["recorded_final_category"],
        "verdict_changed": bool(audit["verdict_changed"]),
        "change_reason": audit["change_reason"],
    }
    return updated


def _assemble_bba5a94_audit(checkpoint_dir: Path, study: str, output_path: Path) -> None:
    """Apply every current-main terminal reading after the complete recorded-budget audit."""
    result_path, report = _load_base_report(study)
    checkpoint = _load_bba5a94_audit_checkpoint(checkpoint_dir, study, result_path)
    records = report["records"]
    expected_keys = {_record_key(study, record) for record in records}
    missing = expected_keys.difference(checkpoint["audited"])
    if missing:
        raise RuntimeError(
            f"Cannot assemble current-main audit; {len(missing)} records remain: {sorted(missing)[:3]}"
        )

    updated_records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    changed_records: list[str] = []
    for record in records:
        key = _record_key(study, record)
        audit = checkpoint["audited"][key]
        observed = audit["observed"]
        updated = dict(record)
        updated.pop("geometry_certified", None)
        if "rates" in updated:
            updated["rates"] = dict(updated["rates"])
            updated["rates"].pop("geometry_certified", None)
        for field in (
            "supports_consistent",
            "corroboration_attempted",
            "supports_converged",
            "supports_corroborated",
            "max_support_residual",
            "disk_rate",
            "epsilon_zero",
            "origin_enclosed",
            "n_right_real_outliers",
            "predicted_gmres_factor",
            "verdict",
        ):
            updated[field] = observed[field]
        updated["geometry_resolution"] = _migrate_resolution_metadata(
            record["geometry_resolution"], audit
        )
        updated["final_category"] = audit["current_final_category"]
        updated["final_verdict"] = observed["verdict"] if _is_certified(observed) else None
        category = str(audit["current_final_category"])
        categories[category] = categories.get(category, 0) + 1
        if audit["verdict_changed"]:
            changed_records.append(key)
            reason = str(audit["change_reason"])
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        updated_records.append(updated)

    updated_report = dict(report)
    updated_report["records"] = updated_records
    report_resolution = dict(report["geometry_resolution"])
    report_resolution["description"] = (
        "Each final terminal reading was reassessed on bba5a94 at its previously recorded "
        "budget with two restarts. A certified category requires consistent supports and "
        "corroboration attempted; an indeterminate result with certified supports is retained "
        "as a current-main fail-closed verdict, while inconsistent support evidence at the cap "
        "is retained as UNCERTIFIED_AT_CAP."
    )
    report_resolution["final_category_counts"] = categories
    report_resolution["bba5a94_audit"] = {
        "checkpoint_schema": BBA5A94_AUDIT_SCHEMA,
        "recorded_budget_only": True,
        "verdict_changed_count": len(changed_records),
        "change_reason_counts": reason_counts,
        "changed_record_keys": changed_records,
    }
    updated_report["geometry_resolution"] = report_resolution
    serialized = json.dumps(updated_report, sort_keys=True)
    if "geometry_certified" in serialized:
        raise RuntimeError("Current-main audit left an obsolete geometry_certified field")
    _atomic_write(output_path, updated_report)
    print(
        f"assembled current-main verdict audit: study={study} "
        f"records={len(updated_records)} output={output_path}"
    )


def _load_result(path: Path) -> dict[str, Any]:
    """Load one completed result JSON used as an immutable audit reference."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing audit result JSON: {path}") from error


def _c1_option_a_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the external per-record checkpoint for the Option-A re-audit."""
    return checkpoint_dir / f"{study}_c1_option_a_audit_v1.json"


def _load_c1_option_a_checkpoint(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
) -> dict[str, Any]:
    """Load or initialize a recorded-budget C1 audit checkpoint."""
    path = _c1_option_a_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(reference_path)
    if not path.exists():
        return {
            "schema": C1_OPTION_A_AUDIT_SCHEMA,
            "study": study,
            "reference_result": str(reference_path),
            "reference_sha256": digest,
            "audited": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != C1_OPTION_A_AUDIT_SCHEMA
        or payload.get("study") != study
        or payload.get("reference_result") != str(reference_path)
        or payload.get("reference_sha256") != digest
    ):
        raise RuntimeError(f"C1 audit checkpoint identity mismatch: {path}")
    return payload


def _c1_option_a_audit_one_record(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
    key: str,
) -> None:
    """Reassess one pre-C1 terminal record at its recorded final budget."""
    reference = _load_result(reference_path)
    records = {_record_key(study, record): record for record in reference["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} C1 audit key: {key}") from error
    checkpoint = _load_c1_option_a_checkpoint(checkpoint_dir, study, reference_path)
    if key in checkpoint["audited"]:
        print(f"C1 audit checkpoint exists: {key}")
        return

    measure = (
        _c1_option_a_measure_pme_record
        if study == "pme"
        else _c1_option_a_measure_porous_fisher_record
    )
    started_at = perf_counter()
    observed, measurement = measure(checkpoint_dir, record, _recorded_final_budget(record))
    _verify_recorded_budget(record, observed, key)
    recorded = _recorded_current_snapshot(record)
    recorded_category = str(record["final_category"])
    current_category = _category_from_snapshot(observed)
    change_reason = _verdict_shift_reason(record, recorded, observed)
    if change_reason is None and current_category != recorded_category:
        change_reason = "support_reproducibility_change"
    checkpoint["audited"][key] = {
        "recorded": recorded,
        "observed": observed,
        "recorded_final_category": recorded_category,
        "current_final_category": current_category,
        "measurement": measurement,
        "verdict_changed": observed["verdict"] != recorded["verdict"],
        "category_changed": current_category != recorded_category,
        "change_reason": change_reason,
        "wall_seconds": perf_counter() - started_at,
    }
    path = _c1_option_a_checkpoint_path(checkpoint_dir, study)
    _atomic_write(path, checkpoint)
    print(
        f"C1 audited {key}: category={current_category} "
        f"changed={checkpoint['audited'][key]['category_changed']} checkpoint={path}"
    )


def _list_pending_c1_option_a_audit(checkpoint_dir: Path, study: str, reference_path: Path) -> None:
    """List every pre-C1 terminal record not yet observed under Option A."""
    reference = _load_result(reference_path)
    checkpoint = _load_c1_option_a_checkpoint(checkpoint_dir, study, reference_path)
    pending = [
        _record_key(study, record)
        for record in reference["records"]
        if _record_key(study, record) not in checkpoint["audited"]
    ]
    print(json.dumps(pending, indent=2))


def _c1_option_a_resolution(record: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Build C1 terminal metadata without presenting pre-fix attempts as current."""
    observed = dict(audit["observed"])
    category = _category_from_snapshot(observed)
    recorded = _recorded_current_snapshot(record)
    category_changed = category != str(record["final_category"])
    reason = _verdict_shift_reason(record, recorded, observed)
    return {
        "status": "CERTIFIED" if _is_certified(observed) else "UNCERTIFIED_AT_CAP",
        "source": "c1_option_a_recorded_budget_audit",
        "description": (
            "The Option-A operator was reassessed once at the pre-C1 record's terminal "
            "budget with two restarts; this audit deliberately does not re-search budgets."
        ),
        "recorded_pre_c1_budget": _recorded_final_snapshot(record),
        "attempts": [observed],
        "final": observed,
        "final_category": category,
        "final_verdict": observed["verdict"] if _is_certified(observed) else None,
        "pre_c1": {
            "final_category": str(record["final_category"]),
            "verdict_changed": observed["verdict"] != recorded["verdict"],
            "category_changed": category_changed,
            "change_reason": reason,
        },
    }


def _replace_geometry_fields(record: dict[str, Any], observed: dict[str, Any]) -> None:
    """Replace every persisted geometry reading with its C1 observation."""
    for field in (
        "supports_consistent",
        "corroboration_attempted",
        "supports_converged",
        "supports_corroborated",
        "max_support_residual",
        "disk_rate",
        "epsilon_zero",
        "epsilon_zero_reduced_arnoldi",
        "full_operator_epsilon_zero",
        "full_operator_epsilon_zero_seconds",
        "epsilon_zero_full_operator_evidence",
        "verdict_reason",
        "arnoldi_k_requested",
        "arnoldi_k_achieved",
        "arnoldi_breakdown",
        "arnoldi_residual_norm",
        "origin_enclosed",
        "n_right_real_outliers",
        "predicted_gmres_factor",
        "verdict",
    ):
        record[field] = observed[field]


def _assemble_c1_option_a_audit(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
    output_path: Path,
) -> None:
    """Build final Option-A results from recorded-budget fresh measurements."""
    reference = _load_result(reference_path)
    checkpoint = _load_c1_option_a_checkpoint(checkpoint_dir, study, reference_path)
    reference_by_key = {_record_key(study, record): record for record in reference["records"]}
    missing = set(reference_by_key).difference(checkpoint["audited"])
    if missing:
        raise RuntimeError(
            f"Cannot assemble C1 audit; {len(missing)} records remain: {sorted(missing)[:3]}"
        )

    records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    budget_counts: dict[str, int] = {}
    changed_keys: list[str] = []
    reason_counts: dict[str, int] = {}
    for key in sorted(reference_by_key):
        reference_record = reference_by_key[key]
        audit = checkpoint["audited"][key]
        observed = audit["observed"]
        updated = dict(reference_record)
        updated.update(audit["measurement"])
        _replace_geometry_fields(updated, observed)
        if study == "pme":
            updated["predicted_iterations_from_envelope"] = (
                pme_breakdown.predicted_iterations_from_envelope(
                    float(observed["disk_rate"]),
                    tol=float(reference["config"]["krylov_tol"]),
                )
            )
        resolution = _c1_option_a_resolution(reference_record, audit)
        updated["geometry_resolution"] = resolution
        updated["final_category"] = resolution["final_category"]
        updated["final_verdict"] = resolution["final_verdict"]
        category = str(resolution["final_category"])
        categories[category] = categories.get(category, 0) + 1
        final = resolution["final"]
        if resolution["status"] == "CERTIFIED":
            budget = f"{final['n_angles']}/{final['fov_max_iters']}/{final['fov_n_restarts']}"
            budget_counts[budget] = budget_counts.get(budget, 0) + 1
        if resolution["pre_c1"]["category_changed"]:
            changed_keys.append(key)
            reason = resolution["pre_c1"]["change_reason"] or "certification_status_changed"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        records.append(updated)

    report = dict(reference)
    report["records"] = records
    report["runtime_seconds"] = sum(
        float(checkpoint["audited"][key]["wall_seconds"]) for key in reference_by_key
    )
    report["geometry_resolution"] = {
        "description": (
            "Option-A C1 re-audit at each record's pre-C1 terminal budget with two restarts. "
            "No budget ladder was rerun; uncorroborated current evidence remains "
            "UNCERTIFIED_AT_CAP."
        ),
        "source_schema": C1_OPTION_A_AUDIT_SCHEMA,
        "reference_result": str(reference_path),
        "reference_sha256": _base_digest(reference_path),
        "recorded_budget_only": True,
        "final_category_counts": categories,
        "certifying_budget_counts": budget_counts,
        "category_changed_count": len(changed_keys),
        "change_reason_counts": reason_counts,
        "changed_record_keys": changed_keys,
    }
    if study == "pme":
        report["regime_claim"] = pme_breakdown._regime_claim(records)
        report["rank_claim"] = pme_breakdown._rank_claim(records)
        report["predictor_quality"] = pme_breakdown._predictor_quality(records)
        report["correlation"] = pme_breakdown._correlation_pairs(records)
        report["regime_map"] = pme_breakdown._regime_map(records)
        report["verdict_on_decision_procedure"] = pme_breakdown._verdict_on_decision_procedure(
            records
        )
    else:
        report["regime_claim"] = porous_fisher_conditioning._regime_claim(records)
        report["verdict_on_decision_procedure"] = (
            porous_fisher_conditioning._verdict_on_decision_procedure(records)
        )
        report["reaction_effect"] = porous_fisher_conditioning._reaction_effect(records)
    if study == "porous_fisher":
        report["physical_model"] = {
            "equation": "u_t = d_xx(Phi_epsilon(u)) + r*u*(1-u)",
            "diffusivity": "D_epsilon(u)=2*sqrt(u**2 + epsilon**2)",
            "preconditioner_scope": "diffusion-only D0 Helmholtz; reaction is unpreconditioned",
        }
    _atomic_write(output_path, report)
    print(f"assembled C1 Option-A audit: study={study} records={len(records)} output={output_path}")


def _current_main_audit_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the independent per-record checkpoint for a rebased-main audit."""
    return checkpoint_dir / f"{study}_current_main_verdict_audit_v1.json"


def _load_current_main_audit_checkpoint(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
    audit_base_revision: str,
) -> dict[str, Any]:
    """Load or initialize a resumable recorded-budget audit on the current base."""
    path = _current_main_audit_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(reference_path)
    if not path.exists():
        return {
            "schema": CURRENT_MAIN_AUDIT_SCHEMA,
            "study": study,
            "reference_result": str(reference_path),
            "reference_sha256": digest,
            "audit_base_revision": audit_base_revision,
            "audited": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != CURRENT_MAIN_AUDIT_SCHEMA
        or payload.get("study") != study
        or payload.get("reference_result") != str(reference_path)
        or payload.get("reference_sha256") != digest
        or payload.get("audit_base_revision") != audit_base_revision
    ):
        raise RuntimeError(f"Current-main audit checkpoint identity mismatch: {path}")
    return payload


def _audit_one_current_main_record(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
    audit_base_revision: str,
    key: str,
) -> None:
    """Reassess one persisted state at its final recorded geometry budget only."""
    reference = _load_result(reference_path)
    records = {_record_key(study, record): record for record in reference["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} current-main audit key: {key}") from error
    checkpoint = _load_current_main_audit_checkpoint(
        checkpoint_dir, study, reference_path, audit_base_revision
    )
    if key in checkpoint["audited"]:
        print(f"current-main audit checkpoint exists: {key}")
        return

    assessor = _assess_pme_record if study == "pme" else _assess_porous_fisher_record
    started_at = perf_counter()
    observed = assessor(checkpoint_dir, record, _recorded_final_budget(record))
    _verify_recorded_budget(record, observed, key)
    recorded = _recorded_current_snapshot(record)
    recorded_category = str(record["final_category"])
    current_category = _category_from_snapshot(observed)
    reason = _current_main_shift_reason(record, recorded, observed)
    category_changed = current_category != recorded_category
    checkpoint["audited"][key] = {
        "recorded": recorded,
        "observed": observed,
        "recorded_final_category": recorded_category,
        "current_final_category": current_category,
        "verdict_changed": observed["verdict"] != recorded["verdict"],
        "category_changed": category_changed,
        "change_reason": reason,
        "wall_seconds": perf_counter() - started_at,
    }
    path = _current_main_audit_checkpoint_path(checkpoint_dir, study)
    _atomic_write(path, checkpoint)
    print(
        f"audited {key}: category={current_category} changed={category_changed} "
        f"reason={reason} checkpoint={path}"
    )


def _list_pending_current_main_audit(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
    audit_base_revision: str,
) -> None:
    """List records not yet re-read under the current conditioning revision."""
    reference = _load_result(reference_path)
    checkpoint = _load_current_main_audit_checkpoint(
        checkpoint_dir, study, reference_path, audit_base_revision
    )
    pending = [
        _record_key(study, record)
        for record in reference["records"]
        if _record_key(study, record) not in checkpoint["audited"]
    ]
    print(json.dumps(pending, indent=2))


def _current_main_resolution(record: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Attach the current-base reading without relabeling it as a budget search."""
    observed = dict(audit["observed"])
    category = str(audit["current_final_category"])
    return {
        "status": "CERTIFIED" if _is_certified(observed) else "UNCERTIFIED_AT_CAP",
        "source": "current_main_recorded_budget_audit",
        "description": (
            "The persisted, SHA256-verified converged source state was reassessed once at its "
            "already-recorded terminal budget with two restarts after rebasing onto the current "
            "conditioning revision. This audit does not re-solve source states or re-search "
            "the geometry budget."
        ),
        "recorded_pre_current_main_budget": _recorded_final_snapshot(record),
        "attempts": [observed],
        "final": observed,
        "final_category": category,
        "final_verdict": observed["verdict"] if _is_certified(observed) else None,
        "pre_current_main": {
            "final_category": audit["recorded_final_category"],
            "verdict_changed": bool(audit["verdict_changed"]),
            "category_changed": bool(audit["category_changed"]),
            "change_reason": audit["change_reason"],
        },
    }


def _assemble_current_main_audit(
    checkpoint_dir: Path,
    study: str,
    reference_path: Path,
    audit_base_revision: str,
    output_path: Path,
) -> None:
    """Assemble all recorded-budget readings into the current-base result report."""
    reference = _load_result(reference_path)
    checkpoint = _load_current_main_audit_checkpoint(
        checkpoint_dir, study, reference_path, audit_base_revision
    )
    reference_by_key = {_record_key(study, record): record for record in reference["records"]}
    missing = set(reference_by_key).difference(checkpoint["audited"])
    if missing:
        raise RuntimeError(
            f"Cannot assemble current-main audit; {len(missing)} records remain: {sorted(missing)[:3]}"
        )

    records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    budget_counts: dict[str, int] = {}
    changed_keys: list[str] = []
    reason_counts: dict[str, int] = {}
    for record in reference["records"]:
        key = _record_key(study, record)
        audit = checkpoint["audited"][key]
        observed = audit["observed"]
        updated = dict(record)
        _replace_geometry_fields(updated, observed)
        if study == "pme":
            updated["predicted_iterations_from_envelope"] = (
                pme_breakdown.predicted_iterations_from_envelope(
                    float(observed["disk_rate"]),
                    tol=float(reference["config"]["krylov_tol"]),
                )
            )
        resolution = _current_main_resolution(record, audit)
        updated["geometry_resolution"] = resolution
        updated["final_category"] = resolution["final_category"]
        updated["final_verdict"] = resolution["final_verdict"]
        category = str(resolution["final_category"])
        categories[category] = categories.get(category, 0) + 1
        final = resolution["final"]
        if resolution["status"] == "CERTIFIED":
            budget = f"{final['n_angles']}/{final['fov_max_iters']}/{final['fov_n_restarts']}"
            budget_counts[budget] = budget_counts.get(budget, 0) + 1
        if audit["category_changed"]:
            changed_keys.append(key)
            reason = audit["change_reason"] or "certification_status_changed"
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        records.append(updated)

    report = dict(reference)
    report["records"] = records
    report_resolution = dict(reference["geometry_resolution"])
    report_resolution.update(
        {
            "description": (
                "Final geometry readings were re-audited against the current conditioning "
                "revision at each record's already-recorded terminal budget with two restarts. "
                "Persisted source states were loaded and SHA256-verified; no source state or "
                "budget ladder was regenerated."
            ),
            "source_schema": CURRENT_MAIN_AUDIT_SCHEMA,
            "reference_result": str(reference_path),
            "reference_sha256": _base_digest(reference_path),
            "audit_base_revision": audit_base_revision,
            "recorded_budget_only": True,
            "final_category_counts": categories,
            "certifying_budget_counts": budget_counts,
            "category_changed_count": len(changed_keys),
            "change_reason_counts": reason_counts,
            "changed_record_keys": changed_keys,
        }
    )
    report["geometry_resolution"] = report_resolution
    report["conditioning_audit_base_revision"] = audit_base_revision
    if study == "pme":
        report["regime_claim"] = pme_breakdown._regime_claim(records)
        report["rank_claim"] = pme_breakdown._rank_claim(records)
        report["predictor_quality"] = pme_breakdown._predictor_quality(records)
        report["correlation"] = pme_breakdown._correlation_pairs(records)
        report["regime_map"] = pme_breakdown._regime_map(records)
        report["verdict_on_decision_procedure"] = pme_breakdown._verdict_on_decision_procedure(
            records
        )
    else:
        report["regime_claim"] = porous_fisher_conditioning._regime_claim(records)
        report["verdict_on_decision_procedure"] = (
            porous_fisher_conditioning._verdict_on_decision_procedure(records)
        )
        report["reaction_effect"] = porous_fisher_conditioning._reaction_effect(records)
    _atomic_write(output_path, report)
    print(
        f"assembled current-main audit: study={study} records={len(records)} output={output_path}"
    )


def _v121_recovery_checkpoint_path(checkpoint_dir: Path, study: str) -> Path:
    """Return the resumable v1.2.1 evidence-recovery checkpoint for one study."""
    return checkpoint_dir / f"{study}_v1_2_1_recovery_v1.json"


def _load_v121_recovery_checkpoint(
    checkpoint_dir: Path,
    study: str,
    result_path: Path,
    audit_base_revision: str,
) -> dict[str, Any]:
    """Load or initialize recovery evidence bound to one resolved base report."""
    path = _v121_recovery_checkpoint_path(checkpoint_dir, study)
    digest = _base_digest(result_path)
    if not path.exists():
        return {
            "schema": V121_RECOVERY_SCHEMA,
            "study": study,
            "reference_result": str(result_path),
            "reference_sha256": digest,
            "audit_base_revision": audit_base_revision,
            "recovered": {},
        }
    payload = _load_checkpoint(path)
    if (
        payload.get("schema") != V121_RECOVERY_SCHEMA
        or payload.get("study") != study
        or payload.get("reference_result") != str(result_path)
        or payload.get("reference_sha256") != digest
        or payload.get("audit_base_revision") != audit_base_revision
    ):
        raise RuntimeError(f"v1.2.1 recovery checkpoint identity mismatch: {path}")
    return payload


def _v121_recovery_required(record: dict[str, Any]) -> bool:
    """Return whether a record needs full-evidence or support-cap recovery."""
    category = str(record["final_category"])
    if category == "CERTIFIED_PROVISIONAL":
        return True
    return category == "UNCERTIFIED_AT_CAP" and not bool(
        _recorded_final_snapshot(record)["supports_converged"]
    )


def _assess_v121_record(
    checkpoint_dir: Path,
    study: str,
    record: dict[str, Any],
    budget: tuple[int, int, int],
    *,
    full_operator_epsilon_evidence: bool = False,
) -> dict[str, Any]:
    """Read one persisted state at one budget, optionally with full epsilon evidence."""
    assessor = _assess_pme_record if study == "pme" else _assess_porous_fisher_record
    return assessor(
        checkpoint_dir,
        record,
        budget,
        full_operator_epsilon_evidence=full_operator_epsilon_evidence,
    )


def _recover_v121_record(
    checkpoint_dir: Path,
    study: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Recover only support-cap or reduced-Arnoldi evidence on one final record."""
    original = _recorded_final_snapshot(record)
    final = dict(original)
    support_attempts: list[dict[str, Any]] = []
    category_before = str(record["final_category"])

    if category_before == "UNCERTIFIED_AT_CAP" and not final["supports_converged"]:
        n_angles, recorded_max_iters, n_restarts = _recorded_final_budget(record)
        for max_iters in SUPPORT_ITERATION_LADDER:
            if max_iters <= recorded_max_iters:
                continue
            attempt = _assess_v121_record(
                checkpoint_dir, study, record, (n_angles, max_iters, n_restarts)
            )
            support_attempts.append(attempt)
            final = attempt
            if final["supports_converged"]:
                break

    full_evidence_attempt = None
    if _is_certified(final) and final["verdict"] == "provisional":
        full_evidence_attempt = _assess_v121_record(
            checkpoint_dir,
            study,
            record,
            (
                int(final["n_angles"]),
                int(final["fov_max_iters"]),
                int(final["fov_n_restarts"]),
            ),
            full_operator_epsilon_evidence=True,
        )
        final = full_evidence_attempt

    return {
        "category_before": category_before,
        "initial": original,
        "support_escalation_attempts": support_attempts,
        "full_operator_evidence_attempt": full_evidence_attempt,
        "final": final,
        "final_category": _category_from_snapshot(final),
        "final_verdict": final["verdict"] if _is_certified(final) else None,
        "support_recovery_attempted": bool(support_attempts),
        "full_operator_evidence_attempted": full_evidence_attempt is not None,
    }


def _recover_one_v121_record(
    checkpoint_dir: Path,
    study: str,
    audit_base_revision: str,
    key: str,
) -> None:
    """Recover one eligible v1.2.1 record and atomically checkpoint it."""
    result_path, report = _load_base_report(study)
    records = {_record_key(study, record): record for record in report["records"]}
    try:
        record = records[key]
    except KeyError as error:
        raise RuntimeError(f"Unknown {study} v1.2.1 recovery key: {key}") from error
    if not _v121_recovery_required(record):
        raise RuntimeError(f"Record does not need v1.2.1 recovery: {key}")
    checkpoint = _load_v121_recovery_checkpoint(
        checkpoint_dir, study, result_path, audit_base_revision
    )
    if key in checkpoint["recovered"]:
        print(f"v1.2.1 recovery checkpoint exists: {key}")
        return
    started_at = perf_counter()
    recovery = _recover_v121_record(checkpoint_dir, study, record)
    recovery["wall_seconds"] = perf_counter() - started_at
    checkpoint["recovered"][key] = recovery
    path = _v121_recovery_checkpoint_path(checkpoint_dir, study)
    _atomic_write(path, checkpoint)
    print(
        f"v1.2.1 recovered {key}: category={recovery['final_category']} "
        f"support_attempts={len(recovery['support_escalation_attempts'])} "
        f"full_evidence={recovery['full_operator_evidence_attempted']} checkpoint={path}"
    )


def _list_pending_v121_recovery(
    checkpoint_dir: Path,
    study: str,
    audit_base_revision: str,
) -> None:
    """List only records eligible for the bounded v1.2.1 recovery operations."""
    result_path, report = _load_base_report(study)
    checkpoint = _load_v121_recovery_checkpoint(
        checkpoint_dir, study, result_path, audit_base_revision
    )
    pending = [
        _record_key(study, record)
        for record in report["records"]
        if _v121_recovery_required(record)
        and _record_key(study, record) not in checkpoint["recovered"]
    ]
    print(json.dumps(pending, indent=2))


def _v121_resolution(record: dict[str, Any], recovery: dict[str, Any] | None) -> dict[str, Any]:
    """Retain the geometry ladder while appending v1.2.1 evidence recovery."""
    resolution = deepcopy(record["geometry_resolution"])
    if recovery is None:
        return resolution
    final = dict(recovery["final"])
    resolution.update(
        {
            "status": "CERTIFIED" if _is_certified(final) else "UNCERTIFIED_AT_CAP",
            "source": "v1_2_1_evidence_recovery",
            "final": final,
            "final_category": recovery["final_category"],
            "final_verdict": recovery["final_verdict"],
            "v1_2_1_recovery": recovery,
        }
    )
    return resolution


def _assemble_v121_recovery(
    checkpoint_dir: Path,
    study: str,
    audit_base_revision: str,
    output_path: Path,
) -> None:
    """Write final v1.2.1 results after bounded evidence-only recovery."""
    result_path, report = _load_base_report(study)
    checkpoint = _load_v121_recovery_checkpoint(
        checkpoint_dir, study, result_path, audit_base_revision
    )
    required = {
        _record_key(study, record)
        for record in report["records"]
        if _v121_recovery_required(record)
    }
    missing = required.difference(checkpoint["recovered"])
    if missing:
        raise RuntimeError(
            f"Cannot assemble v1.2.1 recovery; {len(missing)} records remain: {sorted(missing)[:3]}"
        )

    records: list[dict[str, Any]] = []
    categories: dict[str, int] = {}
    budget_counts: dict[str, int] = {}
    at_cap_support_failures = 0
    at_cap_corroboration_failures = 0
    support_cleared = 0
    full_evidence_attempts = 0
    full_evidence_seconds: list[float] = []
    changed_keys: list[str] = []
    for record in report["records"]:
        key = _record_key(study, record)
        recovery = checkpoint["recovered"].get(key)
        resolution = _v121_resolution(record, recovery)
        final = resolution["final"]
        updated = dict(record)
        _replace_geometry_fields(updated, final)
        updated["geometry_budget"] = {
            "n_angles": final["n_angles"],
            "fov_max_iters": final["fov_max_iters"],
            "fov_n_restarts": final["fov_n_restarts"],
        }
        if study == "pme":
            updated["predicted_iterations_from_envelope"] = (
                pme_breakdown.predicted_iterations_from_envelope(
                    float(final["disk_rate"]),
                    tol=float(report["config"]["krylov_tol"]),
                )
            )
        updated["geometry_resolution"] = resolution
        updated["final_category"] = str(resolution["final_category"])
        updated["final_verdict"] = resolution["final_verdict"]
        category = str(resolution["final_category"])
        categories[category] = categories.get(category, 0) + 1
        if resolution["status"] == "CERTIFIED":
            budget = f"{final['n_angles']}/{final['fov_max_iters']}/{final['fov_n_restarts']}"
            budget_counts[budget] = budget_counts.get(budget, 0) + 1
        else:
            at_cap_support_failures += not bool(final["supports_converged"])
            at_cap_corroboration_failures += bool(final["supports_converged"]) and not bool(
                final["supports_corroborated"]
            )
        if recovery is not None:
            if recovery["category_before"] != category:
                changed_keys.append(key)
            if recovery["support_recovery_attempted"] and _is_certified(final):
                support_cleared += 1
            if recovery["full_operator_evidence_attempted"]:
                full_evidence_attempts += 1
                seconds = final.get("full_operator_epsilon_zero_seconds")
                if seconds is not None:
                    full_evidence_seconds.append(float(seconds))
        records.append(updated)

    final_report = dict(report)
    final_report["records"] = records
    final_report["conditioning_audit_base_revision"] = audit_base_revision
    final_report["geometry_resolution"] = {
        **dict(report["geometry_resolution"]),
        "description": (
            "v1.2.1 final categories use persisted, SHA256-verified converged source states; "
            "the normal 16/60/2-to-96/240/2 geometry ladder; full-operator epsilon-zero only "
            "for certified provisional records; and 240/480 support-iteration recovery only "
            "for support-convergence at-cap records. Restart-corroboration failures remain "
            "UNCERTIFIED_AT_CAP."
        ),
        "source_schema": V121_RECOVERY_SCHEMA,
        "reference_result": str(result_path),
        "reference_sha256": _base_digest(result_path),
        "audit_base_revision": audit_base_revision,
        "final_category_counts": categories,
        "certifying_budget_counts": budget_counts,
        "at_cap_failures": {
            "support_not_converged": at_cap_support_failures,
            "restart_corroboration_failed": at_cap_corroboration_failures,
        },
        "support_escalation": {
            "max_iters_ladder": list(SUPPORT_ITERATION_LADDER),
            "certified_after_escalation": support_cleared,
        },
        "full_operator_epsilon_zero": {
            "method": "dense materialization via full_operator_epsilon_zero",
            "attempted_records": full_evidence_attempts,
            "total_seconds": float(sum(full_evidence_seconds)),
            "median_seconds": (
                None
                if not full_evidence_seconds
                else float(np.median(np.asarray(full_evidence_seconds)))
            ),
        },
        "changed_record_keys": changed_keys,
    }
    if study == "pme":
        final_report["regime_claim"] = pme_breakdown._regime_claim(records)
        final_report["rank_claim"] = pme_breakdown._rank_claim(records)
        final_report["predictor_quality"] = pme_breakdown._predictor_quality(records)
        final_report["correlation"] = pme_breakdown._correlation_pairs(records)
        final_report["regime_map"] = pme_breakdown._regime_map(records)
        final_report["verdict_on_decision_procedure"] = (
            pme_breakdown._verdict_on_decision_procedure(records)
        )
    else:
        final_report["regime_claim"] = porous_fisher_conditioning._regime_claim(records)
        final_report["verdict_on_decision_procedure"] = (
            porous_fisher_conditioning._verdict_on_decision_procedure(records)
        )
        final_report["reaction_effect"] = porous_fisher_conditioning._reaction_effect(records)
    _atomic_write(output_path, final_report)
    print(f"assembled v1.2.1 recovery: study={study} records={len(records)} output={output_path}")


def _list_pending_resolution(checkpoint_dir: Path, study: str) -> None:
    """Print deterministic keys for uncertified records not resolved by a prior probe."""
    base_path, report = _load_base_report(study)
    checkpoint = _load_resolution_checkpoint(checkpoint_dir, study, base_path)
    pending = []
    for record in report["records"]:
        if _is_certified(_geometry_snapshot(record, **BASE_GEOMETRY_BUDGET)):
            continue
        if _priority_reaction_resolution(record) is not None:
            continue
        key = _record_key(study, record)
        if key not in checkpoint["resolved"]:
            pending.append(key)
    print(json.dumps(pending, indent=2))


def main() -> None:
    """Run one fresh-process batch or assemble all completed batches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("pme", "porous_fisher"), required=True)
    parser.add_argument("--batch", help="PME exponent or Porous--Fisher reaction value")
    parser.add_argument("--assemble", action="store_true", help="Assemble completed checkpoints")
    parser.add_argument(
        "--list-pending-resolution",
        action="store_true",
        help="List base-uncertified records not resolved by an existing checkpoint",
    )
    parser.add_argument(
        "--resolve-record",
        help="Resolve one listed record through the bounded geometry ladder",
    )
    parser.add_argument(
        "--assemble-final-resolution",
        action="store_true",
        help="Attach final certified or at-cap metadata to the result report",
    )
    parser.add_argument(
        "--list-pending-bba5a94-audit",
        action="store_true",
        help="List terminal records not yet audited at their recorded current-main budget",
    )
    parser.add_argument(
        "--audit-bba5a94-record",
        help="Audit one record at its already-recorded terminal geometry budget",
    )
    parser.add_argument(
        "--assemble-bba5a94-audit",
        action="store_true",
        help="Apply all audited current-main verdicts and certification fields",
    )
    parser.add_argument(
        "--list-pending-c1-option-a-audit",
        action="store_true",
        help="List pre-C1 terminal records not yet reassessed under Option A",
    )
    parser.add_argument(
        "--audit-c1-option-a-record",
        help="Reassess one pre-C1 terminal record under Option A at its stored budget",
    )
    parser.add_argument(
        "--assemble-c1-option-a-audit",
        action="store_true",
        help="Assemble all recorded-budget Option-A readings into final results",
    )
    parser.add_argument(
        "--list-pending-current-main-audit",
        action="store_true",
        help="List records not yet re-audited on the specified current conditioning revision",
    )
    parser.add_argument(
        "--audit-current-main-record",
        help="Reassess one persisted source state at its stored terminal geometry budget",
    )
    parser.add_argument(
        "--assemble-current-main-audit",
        action="store_true",
        help="Assemble every re-audited current-main reading into the result report",
    )
    parser.add_argument(
        "--list-pending-v121-recovery",
        action="store_true",
        help="List provisional and support-at-cap records eligible for v1.2.1 recovery",
    )
    parser.add_argument(
        "--recover-v121-record",
        help="Recover one provisional or support-at-cap record on the v1.2.1 base",
    )
    parser.add_argument(
        "--assemble-v121-recovery",
        action="store_true",
        help="Assemble v1.2.1 full-evidence and support-iteration recovery",
    )
    parser.add_argument(
        "--reference-result",
        type=Path,
        help="Immutable prior final result JSON used only for record keys and budgets",
    )
    parser.add_argument(
        "--audit-base-revision",
        help="Immutable conditioning base revision being measured by a current-main audit",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--output", type=Path, help="Override the canonical result path")
    args = parser.parse_args()
    actions = sum(
        bool(action)
        for action in (
            args.batch,
            args.assemble,
            args.list_pending_resolution,
            args.resolve_record,
            args.assemble_final_resolution,
            args.list_pending_bba5a94_audit,
            args.audit_bba5a94_record,
            args.assemble_bba5a94_audit,
            args.list_pending_c1_option_a_audit,
            args.audit_c1_option_a_record,
            args.assemble_c1_option_a_audit,
            args.list_pending_current_main_audit,
            args.audit_current_main_record,
            args.assemble_current_main_audit,
            args.list_pending_v121_recovery,
            args.recover_v121_record,
            args.assemble_v121_recovery,
        )
    )
    if actions != 1:
        parser.error("Specify exactly one action")

    if args.list_pending_resolution:
        _list_pending_resolution(args.checkpoint_dir, args.study)
        return
    if args.resolve_record:
        _resolve_one_record(args.checkpoint_dir, args.study, args.resolve_record)
        return
    if args.list_pending_bba5a94_audit:
        _list_pending_bba5a94_audit(args.checkpoint_dir, args.study)
        return
    if args.audit_bba5a94_record:
        _audit_one_bba5a94_record(args.checkpoint_dir, args.study, args.audit_bba5a94_record)
        return
    if args.list_pending_c1_option_a_audit:
        if args.reference_result is None:
            parser.error("--list-pending-c1-option-a-audit requires --reference-result")
        _list_pending_c1_option_a_audit(args.checkpoint_dir, args.study, args.reference_result)
        return
    if args.audit_c1_option_a_record:
        if args.reference_result is None:
            parser.error("--audit-c1-option-a-record requires --reference-result")
        _c1_option_a_audit_one_record(
            args.checkpoint_dir,
            args.study,
            args.reference_result,
            args.audit_c1_option_a_record,
        )
        return
    if args.assemble_final_resolution:
        _assemble_final_resolution(
            args.checkpoint_dir,
            args.study,
            args.output or _base_output_path(args.study),
        )
        return
    if args.assemble_bba5a94_audit:
        _assemble_bba5a94_audit(
            args.checkpoint_dir,
            args.study,
            args.output or _base_output_path(args.study),
        )
        return
    if args.assemble_c1_option_a_audit:
        if args.reference_result is None:
            parser.error("--assemble-c1-option-a-audit requires --reference-result")
        _assemble_c1_option_a_audit(
            args.checkpoint_dir,
            args.study,
            args.reference_result,
            args.output or _base_output_path(args.study),
        )
        return
    if args.list_pending_current_main_audit:
        if args.reference_result is None or args.audit_base_revision is None:
            parser.error(
                "--list-pending-current-main-audit requires --reference-result and "
                "--audit-base-revision"
            )
        _list_pending_current_main_audit(
            args.checkpoint_dir,
            args.study,
            args.reference_result,
            args.audit_base_revision,
        )
        return
    if args.list_pending_v121_recovery:
        if args.audit_base_revision is None:
            parser.error("--list-pending-v121-recovery requires --audit-base-revision")
        _list_pending_v121_recovery(args.checkpoint_dir, args.study, args.audit_base_revision)
        return
    if args.recover_v121_record:
        if args.audit_base_revision is None:
            parser.error("--recover-v121-record requires --audit-base-revision")
        _recover_one_v121_record(
            args.checkpoint_dir,
            args.study,
            args.audit_base_revision,
            args.recover_v121_record,
        )
        return
    if args.assemble_v121_recovery:
        if args.audit_base_revision is None:
            parser.error("--assemble-v121-recovery requires --audit-base-revision")
        _assemble_v121_recovery(
            args.checkpoint_dir,
            args.study,
            args.audit_base_revision,
            args.output or _base_output_path(args.study),
        )
        return
    if args.audit_current_main_record:
        if args.reference_result is None or args.audit_base_revision is None:
            parser.error(
                "--audit-current-main-record requires --reference-result and --audit-base-revision"
            )
        _audit_one_current_main_record(
            args.checkpoint_dir,
            args.study,
            args.reference_result,
            args.audit_base_revision,
            args.audit_current_main_record,
        )
        return
    if args.assemble_current_main_audit:
        if args.reference_result is None or args.audit_base_revision is None:
            parser.error(
                "--assemble-current-main-audit requires --reference-result and "
                "--audit-base-revision"
            )
        _assemble_current_main_audit(
            args.checkpoint_dir,
            args.study,
            args.reference_result,
            args.audit_base_revision,
            args.output or _base_output_path(args.study),
        )
        return

    if args.study == "pme":
        if args.assemble:
            _assemble_pme(
                args.checkpoint_dir,
                args.output or Path(pme_breakdown.BreakdownConfig().output_path),
            )
            return
        try:
            m = int(args.batch)
        except ValueError as error:
            parser.error(f"PME batch must be an integer exponent: {error}")
        if m not in pme_breakdown.BreakdownConfig().m_values:
            parser.error(f"Unsupported PME exponent: {m}")
        _run_pme_batch(args.checkpoint_dir, m)
        return

    if args.assemble:
        _assemble_porous_fisher(
            args.checkpoint_dir,
            args.output or Path(porous_fisher_conditioning.ReactionStudyConfig().output_path),
        )
        return
    try:
        reaction = float(args.batch)
    except ValueError as error:
        parser.error(f"Porous--Fisher batch must be a float reaction strength: {error}")
    if reaction not in porous_fisher_conditioning.ReactionStudyConfig().reaction_values:
        parser.error(f"Unsupported reaction strength: {reaction:g}")
    _run_porous_fisher_batch(args.checkpoint_dir, reaction)


if __name__ == "__main__":
    main()
