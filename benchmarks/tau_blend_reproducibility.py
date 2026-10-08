"""Fail-closed provenance and source-state persistence for tau-blend studies."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

SOURCE_STATE_SCHEMA = "pme_tau_blend_source_states_v4"


def _git_revision(*args: str) -> str | None:
    """Return one optional local Git identity without requiring a remote."""
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


def provenance_revisions() -> dict[str, str]:
    """Capture merge-base, tag, or HEAD provenance without making it fatal."""
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


def state_identity(state: jax.Array) -> dict[str, Any]:
    """Return a deterministic SHA256 identity for one float64 state."""
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


def fingerprint_digest(fingerprint: dict[str, Any]) -> str:
    """Return the stable filename digest for a generation contract."""
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_paths(
    cache_root: str | Path,
    label: str,
    fingerprint: dict[str, Any],
) -> tuple[Path, Path]:
    safe_label = "".join(character if character.isalnum() else "_" for character in label)
    stem = f"{safe_label}-{fingerprint_digest(fingerprint)[:20]}"
    root = Path(cache_root).resolve()
    return root / f"{stem}.npz", root / f"{stem}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_save_states(path: Path, states: list[jax.Array]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        f"state_{index}": np.asarray(jax.device_get(state), dtype=np.float64)
        for index, state in enumerate(states)
    }
    with NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_cached_states(
    cache_root: str | Path,
    label: str,
    fingerprint: dict[str, Any],
) -> tuple[list[jax.Array], list[dict[str, Any]], list[dict[str, Any]]] | None:
    """Load states only when schema, fingerprint, count, and SHA256 all match."""
    array_path, manifest_path = _cache_paths(cache_root, label, fingerprint)
    if not array_path.is_file() or not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SOURCE_STATE_SCHEMA:
        raise RuntimeError(f"stale source-state schema: {manifest_path}")
    if manifest.get("generation_fingerprint") != fingerprint:
        raise RuntimeError(f"source-state fingerprint mismatch: {manifest_path}")
    identities = manifest.get("source_state_identities")
    metadata = manifest.get("source_state_metadata")
    if not isinstance(identities, list) or not isinstance(metadata, list):
        raise RuntimeError(f"source-state manifest is incomplete: {manifest_path}")
    if len(identities) != len(metadata):
        raise RuntimeError(f"source-state manifest lengths disagree: {manifest_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        expected_keys = {f"state_{index}" for index in range(len(identities))}
        if set(arrays.files) != expected_keys:
            raise RuntimeError(f"source-state array members disagree: {array_path}")
        states = [
            jax.block_until_ready(jnp.asarray(arrays[f"state_{index}"], dtype=jnp.float64))
            for index in range(len(identities))
        ]
    if [state_identity(state) for state in states] != identities:
        raise RuntimeError(f"source-state SHA256 mismatch: {array_path}")
    return states, identities, metadata


def load_or_generate_states(
    cache_root: str | Path,
    label: str,
    fingerprint: dict[str, Any],
    generator: Callable[[], tuple[list[jax.Array], list[dict[str, Any]]]],
) -> tuple[list[jax.Array], list[dict[str, Any]], list[dict[str, Any]], bool]:
    """Load one exact trajectory or generate, persist, and reload it once."""
    cached = load_cached_states(cache_root, label, fingerprint)
    if cached is not None:
        states, identities, metadata = cached
        return states, identities, metadata, True
    states, metadata = generator()
    if not states or len(states) != len(metadata):
        raise RuntimeError("generated source states and metadata must be nonempty and aligned")
    identities = [state_identity(state) for state in states]
    array_path, manifest_path = _cache_paths(cache_root, label, fingerprint)
    _atomic_save_states(array_path, states)
    _atomic_write_json(
        manifest_path,
        {
            "schema": SOURCE_STATE_SCHEMA,
            "generation_fingerprint": fingerprint,
            "array_path": array_path.name,
            "source_state_identities": identities,
            "source_state_metadata": metadata,
        },
    )
    loaded = load_cached_states(cache_root, label, fingerprint)
    if loaded is None:
        raise RuntimeError(f"persisted source states could not be reloaded: {array_path}")
    loaded_states, loaded_identities, loaded_metadata = loaded
    return loaded_states, loaded_identities, loaded_metadata, False


def source_state_artifact(
    cache_root: str | Path,
    label: str,
    fingerprint: dict[str, Any],
    index: int,
    identity: dict[str, Any],
) -> dict[str, Any]:
    """Return relocatable provenance for one state in a v4 artifact."""
    array_path, _ = _cache_paths(cache_root, label, fingerprint)
    root = Path(cache_root).resolve()
    return {
        "schema": SOURCE_STATE_SCHEMA,
        "relative_path": str(array_path.relative_to(root)),
        "sample_position": index,
        "source_state_identity": identity,
        "generation_fingerprint": fingerprint,
        "converged": True,
    }


def replay_source_state(
    record: dict[str, Any],
    *,
    cache_root: str | Path,
    expected_record_config: dict[str, Any],
) -> jax.Array:
    """Replay a record from its exact config and relocatable v4 artifact."""
    if record.get("record_config") != expected_record_config:
        raise RuntimeError("record configuration mismatch during reassessment")
    artifact = record.get("source_state_artifact")
    if not isinstance(artifact, dict) or artifact.get("schema") != SOURCE_STATE_SCHEMA:
        raise RuntimeError("record lacks a compatible source-state artifact")
    fingerprint = artifact.get("generation_fingerprint")
    if not isinstance(fingerprint, dict):
        raise RuntimeError("record lacks a source-state fingerprint")
    relative = Path(str(artifact.get("relative_path", "")))
    if relative.is_absolute() or ".." in relative.parts:
        raise RuntimeError("source-state artifact path is not cache-root-relative")
    label = str(fingerprint.get("label", ""))
    expected_array, _ = _cache_paths(cache_root, label, fingerprint)
    if Path(cache_root).resolve() / relative != expected_array:
        raise RuntimeError("source-state artifact path disagrees with its fingerprint")
    cached = load_cached_states(cache_root, label, fingerprint)
    if cached is None:
        raise RuntimeError("source-state artifact is missing")
    states, identities, _ = cached
    position = int(artifact.get("sample_position", -1))
    if position < 0 or position >= len(states):
        raise RuntimeError("source-state sample position is invalid")
    if identities[position] != artifact.get("source_state_identity"):
        raise RuntimeError("source-state identity mismatch during reassessment")
    return states[position]


__all__ = [
    "SOURCE_STATE_SCHEMA",
    "fingerprint_digest",
    "load_cached_states",
    "load_or_generate_states",
    "provenance_revisions",
    "replay_source_state",
    "source_state_artifact",
    "state_identity",
]
