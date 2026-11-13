"""Tests for host-labelled Brusselator source-state repair."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import brusselator_dt_sweep as sweep


def test_regenerated_loader_keeps_a_distinct_host_identity(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    spec = sweep.CPU_STATE_SPECS[0]
    record = sweep._find_record(repository, spec)
    state = {
        "u": jnp.zeros((66, 66), dtype=jnp.float64),
        "v": jnp.ones((66, 66), dtype=jnp.float64),
    }
    identity = sweep._state_identity(state)
    relative = Path("host-labelled.npz")
    sweep._atomic_write_states(tmp_path / relative, [state])
    fingerprint = record["record_config"]["source_state_fingerprint"]
    key = sweep._fingerprint_key(fingerprint)
    manifest = {
        "schema": sweep.REGENERATED_SOURCE_SCHEMA,
        "identity_scope": "host-labelled regenerated realisations; not Stage-3 identities",
        "host_label": "test-host",
        "trajectories": {
            key: {
                "relative_path": str(relative),
                "npz_sha256": sweep._sha256(tmp_path / relative),
                "generation_fingerprint": fingerprint,
                "regenerated_state_identities": [identity],
            }
        },
        "records": {
            spec.identifier: {
                "original_stage3": {
                    "relative_path": record["source_state_artifact"]["relative_path"],
                    "source_state_identity": record["source_state_artifact"][
                        "source_state_identity"
                    ],
                },
                "regenerated": {
                    "relative_path": str(relative),
                    "sample_position": 0,
                    "source_state_identity": identity,
                },
                "generation_fingerprint": fingerprint,
            }
        },
    }
    sweep._atomic_write_json(tmp_path / "regenerated_source_states_manifest.json", manifest)

    loaded, source = sweep._load_regenerated_state(tmp_path, spec, record)

    np.testing.assert_array_equal(np.asarray(loaded["u"]), np.asarray(state["u"]))
    assert source["state_identity"] == identity
    assert source["original_stage3_identity"] != identity
    with pytest.raises(RuntimeError, match="source artifact is missing"):
        sweep._load_state(tmp_path, record)


def test_regeneration_refuses_to_overwrite_an_existing_directory(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[1]
    with pytest.raises(FileExistsError, match="already exists"):
        sweep._regenerate_source_states(repository, tmp_path)
