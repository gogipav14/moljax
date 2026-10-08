"""Regression tests for fail-closed tau-blend source-state persistence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from benchmarks.tau_blend_reproducibility import (
    load_or_generate_states,
    replay_source_state,
    source_state_artifact,
)

jax.config.update("jax_enable_x64", True)


def _fingerprint() -> dict[str, object]:
    return {
        "schema": "pme_tau_blend_generation_fingerprint_v4",
        "label": "test-state",
        "study": "tau-blend-test",
        "grid": {"nx": 4, "domain": [-1.0, 1.0]},
        "m": 4,
        "state_dt": 0.01,
        "solver": {"max_backtrack": 8},
        "model_version": "test",
    }


def _persist(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    fingerprint = _fingerprint()

    def generate() -> tuple[list[jax.Array], list[dict[str, object]]]:
        return [jnp.arange(4, dtype=jnp.float64)], [{"step": 0}]

    _, identities, _, _ = load_or_generate_states(root, "test-state", fingerprint, generate)
    artifact = source_state_artifact(root, "test-state", fingerprint, 0, identities[0])
    record = {"record_config": {"m": 4}, "source_state_artifact": artifact}
    return fingerprint, record


def test_source_state_cache_replays_after_relocation(tmp_path: Path) -> None:
    """A relative artifact remains replayable after moving its cache root."""
    original = tmp_path / "original"
    relocated = tmp_path / "relocated"
    _, record = _persist(original)
    shutil.copytree(original, relocated)
    replayed = replay_source_state(record, cache_root=relocated, expected_record_config={"m": 4})
    assert jnp.array_equal(replayed, jnp.arange(4, dtype=jnp.float64))


def test_source_state_cache_rejects_foreign_fingerprint(tmp_path: Path) -> None:
    """A foreign generation contract is a miss, never cross-config reuse."""
    fingerprint, _ = _persist(tmp_path)
    foreign = {**fingerprint, "m": 8}

    def generate() -> tuple[list[jax.Array], list[dict[str, object]]]:
        return [jnp.full((4,), 8.0)], [{"step": 0}]

    states, _, _, reused = load_or_generate_states(tmp_path, "test-state", foreign, generate)
    assert reused is False
    assert jnp.array_equal(states[0], jnp.full((4,), 8.0))


def test_source_state_cache_rejects_sha256_corruption(tmp_path: Path) -> None:
    """Manifest identity corruption fails closed during replay."""
    _, record = _persist(tmp_path)
    manifest = next(tmp_path.glob("*.json"))
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["source_state_identities"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        replay_source_state(record, cache_root=tmp_path, expected_record_config={"m": 4})


def test_source_state_reassessment_requires_exact_record_config(tmp_path: Path) -> None:
    """Reassessment cannot silently rebuild from a default configuration."""
    _, record = _persist(tmp_path)
    with pytest.raises(RuntimeError, match="configuration mismatch"):
        replay_source_state(record, cache_root=tmp_path, expected_record_config={"m": 2})
