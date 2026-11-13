"""Tests for host-labelled Brusselator source-state repair."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks import brusselator_dt_sweep as sweep
from benchmarks import brusselator_dt_sweep_cpu_replication as cpu_replication


def test_cpu_replication_cell_key_accepts_a_spec_or_identifier() -> None:
    spec = sweep.CPU_STATE_SPECS[0]
    expected = f"main|{spec.identifier}|dt=1|fft_diffusion"
    assert cpu_replication._cell_key("main", spec, 1.0, "fft_diffusion") == expected
    assert cpu_replication._cell_key("main", spec.identifier, 1.0, "fft_diffusion") == expected


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


def test_cpu_replication_report_is_current_and_reproduces_the_decision() -> None:
    repository = Path(__file__).resolve().parents[1]
    result = repository / "benchmarks/results/brusselator_dt_sweep_cpu_replication.json"
    report = repository / "benchmarks/results/BRUSSELATOR_DT_SWEEP_CPU_REPLICATION.md"
    payload = json.loads(result.read_text(encoding="utf-8"))

    assert len(payload["cells"]) == 200
    assert payload["similarity_classification"]["label"] == "reproduction"
    assert payload["success_criterion_met"]
    assert all(row["matches"] for row in payload["decision_comparison"].values())
    assert all(
        cell["gpu"]["gmres"]["iterations"] == cell["cpu"]["gmres"]["iterations"]
        and cell["gpu"]["gmres"]["converged"] == cell["cpu"]["gmres"]["converged"]
        for cell in payload["cells"].values()
    )
    assert report.read_text(encoding="utf-8") == cpu_replication.render(payload)
