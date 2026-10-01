"""Smoke tests for the experimental two-regime Brusselator conditioning study."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

import pytest

from benchmarks import brusselator_conditioning as benchmark
from benchmarks import resolve_brusselator_fov_supports as resolver
from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams
from moljax.experimental.brusselator_conditioning import (
    HOPF_REGIME,
    TURING_REGIME,
    assess_brusselator_state,
    build_brusselator_system,
    sampled_visited_states,
    visited_states,
)
from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
    dense_padded_preconditioned_operator,
)


def _homogeneous_state(regime, grid):
    model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
    state = model.apply_bcs(
        {
            "u": jax.numpy.full(
                (grid.ny_total, grid.nx_total), regime.a, dtype=jax.numpy.float64
            ),
            "v": jax.numpy.full(
                (grid.ny_total, grid.nx_total), regime.b / regime.a, dtype=jax.numpy.float64
            ),
        },
        0.0,
    )
    return state, model, fft_cache, diffusivities


@pytest.fixture(scope="module")
def tiny_fft_records():
    """Evaluate the minimal FFT-only two-regime smoke configuration once."""
    records = {}
    for seed, regime in enumerate((HOPF_REGIME, TURING_REGIME), start=1):
        grid = Grid2D.uniform(8, 8, 0.0, regime.domain_length, 0.0, regime.domain_length)
        state = visited_states(
            regime,
            grid=grid,
            n_steps=1,
            dt=0.1,
            perturbation=1.0e-3,
            seed=seed,
        )[0]
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        records[regime.name] = assess_brusselator_state(
            state,
            model,
            fft_cache,
            diffusivities,
            0.1,
            regime,
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=3,
            seed=seed,
        )
    return records


@pytest.mark.slow
def test_hopf_visited_state_passes_adjoint_gate_and_has_a_verdict(tiny_fft_records):
    """A tiny FFT-preconditioned Hopf state is valid input to the toolbox."""
    record = tiny_fft_records["hopf"]
    assert record["status"] == "completed"
    assert record["adjoint_error"] <= 1.0e-8
    assert record["verdict"] in {"adequate", "investigate", "indeterminate"}


@pytest.mark.slow
def test_both_regimes_record_structural_discriminators(tiny_fft_records):
    """The outcome is data, but both physical-regime record fields must exist."""
    for regime in ("hopf", "turing"):
        record = tiny_fft_records[regime]
        assert record["status"] == "completed"
        assert record["adjoint_error"] <= 1.0e-8
        assert isinstance(record["origin_enclosed"], bool)
        assert math.isfinite(record["fov_imaginary_extent"])
        assert record["fov_imaginary_extent"] >= 0.0


@pytest.mark.slow
def test_developed_hopf_sample_leaves_the_fixed_point_and_passes_adjoint_gate():
    """A late sampled Hopf state is developed rather than a seed perturbation."""
    perturbation = 1.0e-3
    grid = Grid2D.uniform(
        8,
        8,
        0.0,
        HOPF_REGIME.domain_length,
        0.0,
        HOPF_REGIME.domain_length,
    )
    samples = sampled_visited_states(
        HOPF_REGIME,
        grid=grid,
        sample_steps=(1, 5, 10),
        dt=1.0,
        perturbation=perturbation,
        seed=20260822,
        nk_params=NKParams(
            max_newton_iters=15,
            max_krylov_iters=100,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        ),
    )
    late = samples[-1]
    model, fft_cache, diffusivities = build_brusselator_system(HOPF_REGIME, grid)
    assessment = assess_brusselator_state(
        late.state,
        model,
        fft_cache,
        diffusivities,
        1.0,
        HOPF_REGIME,
        n_angles=3,
        fov_max_iters=4,
        arnoldi_steps=3,
        seed=20260822,
    )

    departure = max(late.developedness.values())
    assert departure > 20.0 * perturbation
    assert assessment["status"] == "completed"
    assert assessment["adjoint_error"] <= 1.0e-8


@pytest.mark.slow
def test_fourier_weyl_bound_certifies_a_small_homogeneous_turing_state():
    """The integrated full-operator bound unlocks adequacy only with all gates clear."""
    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.01,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        compute_lobpcg_upper_estimate=True,
        seed=20260821,
    )
    certificate = assessment["fourier_weyl_ghost_lower_bound"]
    interior = np.asarray(state["u"])[1:-1, 1:-1]
    interior_v = np.asarray(state["v"])[1:-1, 1:-1]
    dense = float(
        np.linalg.svd(
            dense_padded_preconditioned_operator(
                interior,
                interior_v,
                du=TURING_REGIME.du,
                dv=TURING_REGIME.dv,
                beta=TURING_REGIME.b,
                dt=0.01,
            ),
            compute_uv=False,
        )[-1]
    )
    assert assessment["verdict"] == "adequate"
    assert assessment["epsilon_zero_full_operator_evidence"] is True
    assert certificate["status"] == "clears_adequacy_gate"
    assert certificate["full_lower_bound"] >= 0.1
    assert certificate["full_lower_bound"] <= dense + 5.0e-13
    assert assessment["lobpcg_sigma_min_upper_estimate"] is not None


@pytest.mark.slow
@pytest.mark.parametrize("weak_bound", [0.0, 0.05])
def test_valid_weak_fourier_weyl_bound_preserves_provisional(monkeypatch, weak_bound):
    """A valid but insufficient lower bound cannot promote a provisional reading."""
    import moljax.experimental.brusselator_conditioning as conditioning

    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    genuine = conditioning._fourier_weyl_bound(state, model, TURING_REGIME, 0.01)
    weak_selected = replace(genuine.selected, full_lower_bound=weak_bound)
    monkeypatch.setattr(
        conditioning,
        "_fourier_weyl_bound",
        lambda *_args, **_kwargs: replace(genuine, selected=weak_selected),
    )
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.01,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        seed=20260821,
    )
    certificate = assessment["fourier_weyl_ghost_lower_bound"]
    assert genuine.selected.full_lower_bound >= 0.1
    assert assessment["verdict"] == "provisional"
    assert assessment["epsilon_zero_full_operator_evidence"] is False
    assert certificate["status"] == "valid_but_below_adequacy_gate"
    assert certificate["full_lower_bound"] == pytest.approx(weak_bound)


@pytest.mark.slow
def test_origin_enclosure_remains_indeterminate_despite_a_valid_bound():
    """The full-operator lower bound cannot override an origin-enclosed FOV."""
    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state = visited_states(
        TURING_REGIME,
        grid=grid,
        n_steps=1,
        dt=0.2,
        perturbation=0.8,
        seed=20260928,
    )[0]
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    assessment = assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.2,
        TURING_REGIME,
        n_angles=8,
        fov_max_iters=60,
        arnoldi_steps=6,
        seed=20260928,
    )
    assert assessment["fourier_weyl_ghost_lower_bound"]["full_lower_bound"] >= 0.0
    assert assessment["origin_enclosed"] is True
    assert assessment["verdict"] == "indeterminate"


def test_v4_source_cache_rejects_a_foreign_generation_fingerprint(tmp_path):
    """A foreign source-generation contract is a safe cache miss."""
    config = benchmark._config(
        "screen_64",
        nx=4,
        ny=4,
        n_states=1,
        source_state_cache_dir=str(tmp_path),
    )
    fingerprint = benchmark._source_state_fingerprint(config, TURING_REGIME, (1,), 7)
    state = {
        "u": jax.numpy.ones((6, 6), dtype=jax.numpy.float64),
        "v": jax.numpy.full((6, 6), 1.8, dtype=jax.numpy.float64),
    }
    benchmark._persist_source_states(config, TURING_REGIME, fingerprint, [state])
    foreign = {**fingerprint, "seed": 8}
    assert benchmark._load_cached_states(config, TURING_REGIME, foreign) is None


def _replay_fixture(tmp_path, cache_root=None):
    """Persist one minimal source artifact and return its exact replay record."""
    cache_root = tmp_path / "original-cache" if cache_root is None else cache_root
    config = benchmark._config(
        "screen_64",
        nx=4,
        ny=4,
        n_states=1,
        source_state_cache_dir=str(cache_root),
    )
    fingerprint = benchmark._source_state_fingerprint(config, TURING_REGIME, (1,), 7)
    state = {
        "u": jax.numpy.ones((6, 6), dtype=jax.numpy.float64),
        "v": jax.numpy.full((6, 6), 1.8, dtype=jax.numpy.float64),
    }
    _, identities = benchmark._persist_source_states(config, TURING_REGIME, fingerprint, [state])
    array_path, _ = benchmark._cache_paths(config, TURING_REGIME, fingerprint)
    record = {
        "time": config.dt,
        "record_config": {
            "regime": TURING_REGIME._asdict(),
            "grid": {"nx": 4, "ny": 4, "n_ghost": 1},
            "source_state_fingerprint": fingerprint,
            "analysis_dt": config.dt,
            "preconditioner_kind": "identity",
            "n_angles": config.n_angles,
            "fov_max_iters": config.fov_max_iters,
            "fov_residual_tolerance": config.fov_residual_tolerance,
            "fov_n_restarts": config.fov_n_restarts,
            "arnoldi_steps": config.arnoldi_steps,
            "compute_lobpcg_upper_estimate": config.compute_lobpcg_upper_estimate,
            "assessment_seed": config.seed,
            "domain_length": TURING_REGIME.domain_length,
        },
        "source_state_artifact": {
            "schema": benchmark.SOURCE_STATE_ARTIFACT_SCHEMA,
            "relative_path": array_path.name,
            "sample_position": 0,
            "source_state_identity": identities[0],
            "generation_fingerprint": fingerprint,
            "converged": True,
        },
    }
    return cache_root, record, state


def test_v4_source_cache_replay_survives_relocation(tmp_path, monkeypatch):
    """A cache-root-relative artifact replays after an intact cache is moved."""
    cache_root, record, original_state = _replay_fixture(tmp_path)
    relocated = tmp_path / "relocated-cache"
    shutil.move(str(cache_root), relocated)
    expected_identity = benchmark._source_state_identity(original_state)
    monkeypatch.setattr(
        benchmark,
        "assess_brusselator_state",
        lambda state, *_args, **_kwargs: {
            "loaded_source_identity": benchmark._source_state_identity(state)
        },
    )

    replayed = benchmark.reassess_brusselator_record(
        record, source_state_cache_dir=str(relocated)
    )

    assert replayed["loaded_source_identity"] == expected_identity


@pytest.mark.parametrize("bad_path", ["/tmp/foreign-state.npz", "../foreign-state.npz"])
def test_v4_source_cache_replay_rejects_nonrelative_artifact_paths(tmp_path, bad_path):
    """Relocatable provenance never permits an absolute or traversing path."""
    cache_root, record, _ = _replay_fixture(tmp_path)
    record["source_state_artifact"]["relative_path"] = bad_path

    with pytest.raises(RuntimeError, match="cache-root-relative"):
        benchmark.reassess_brusselator_record(record, source_state_cache_dir=str(cache_root))


def _legacy_path(cache_root, record):
    """Return the 8be9ef8 artifact path form: the cache-prefixed array path."""
    return str(cache_root / record["source_state_artifact"]["relative_path"])


def _loaded_identity(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "assess_brusselator_state",
        lambda state, *_args, **_kwargs: {
            "loaded_source_identity": benchmark._source_state_identity(state)
        },
    )


@pytest.mark.parametrize("record_original_root", [False, True])
def test_legacy_cache_prefixed_artifact_replays_from_relocated_cache(
    tmp_path, monkeypatch, record_original_root
):
    """An 8be9ef8 absolute artifact path migrates when the relocated artifact is intact."""
    cache_root, record, original_state = _replay_fixture(tmp_path)
    record["source_state_artifact"]["relative_path"] = _legacy_path(cache_root, record)
    assert Path(record["source_state_artifact"]["relative_path"]).is_absolute()
    relocated = tmp_path / "relocated-cache"
    shutil.move(str(cache_root), relocated)
    _loaded_identity(monkeypatch)

    replayed = benchmark.reassess_brusselator_record(
        record,
        source_state_cache_dir=str(relocated),
        original_source_state_cache_dir=str(cache_root) if record_original_root else None,
    )

    assert replayed["loaded_source_identity"] == benchmark._source_state_identity(
        original_state
    )


@pytest.mark.parametrize("tamper", ["sha256", "fingerprint"])
def test_legacy_cache_prefixed_artifact_fails_closed_on_a_tampered_artifact(
    tmp_path, monkeypatch, tamper
):
    """Migrating a legacy path never weakens the fingerprint or SHA256 checks."""
    cache_root, record, _ = _replay_fixture(tmp_path)
    record["source_state_artifact"]["relative_path"] = _legacy_path(cache_root, record)
    relocated = tmp_path / "relocated-cache"
    shutil.move(str(cache_root), relocated)
    array_name = Path(record["source_state_artifact"]["relative_path"]).name
    if tamper == "sha256":
        np.savez_compressed(
            relocated / array_name, u_0=np.full((6, 6), 1.5), v_0=np.full((6, 6), 1.8)
        )
        message = "SHA256 mismatch"
    else:
        manifest_path = (relocated / array_name).with_suffix(".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["generation_fingerprint"]["seed"] += 1
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        message = "fingerprint mismatch"
    _loaded_identity(monkeypatch)

    with pytest.raises(RuntimeError, match=message):
        benchmark.reassess_brusselator_record(record, source_state_cache_dir=str(relocated))


@pytest.mark.parametrize("variant", ["foreign_root", "foreign_name", "traversal"])
def test_legacy_artifact_path_outside_its_cache_root_is_rejected(tmp_path, variant):
    """A legacy path migrates only from its recorded cache root, never by traversal."""
    cache_root, record, _ = _replay_fixture(tmp_path)
    name = Path(record["source_state_artifact"]["relative_path"]).name
    recorded = {
        "foreign_root": str(tmp_path / "elsewhere" / name),
        "foreign_name": str(cache_root / "turing-0000000000000000.npz"),
        "traversal": str(cache_root / ".." / "original-cache" / name),
    }[variant]
    record["source_state_artifact"]["relative_path"] = recorded

    with pytest.raises(RuntimeError, match="cache root|cache-root-relative"):
        benchmark.reassess_brusselator_record(
            record,
            source_state_cache_dir=str(cache_root),
            original_source_state_cache_dir=str(cache_root),
        )


def _tiny_regeneration_config(root):
    return benchmark._config(
        "screen_64",
        nx=4,
        ny=4,
        n_states=1,
        regimes=("turing",),
        compute_lobpcg_upper_estimate=False,
        output_path=str(root / "results" / "brusselator_conditioning.json"),
        source_state_cache_dir=str(root / "source_states"),
        record_checkpoint_path=str(root / "record_checkpoints" / "screen_64.json"),
    )


@pytest.mark.parametrize("relocate", [False, True])
def test_partial_regeneration_resumes_from_a_base_revision_checkpoint(
    tmp_path, monkeypatch, relocate
):
    """An 8be9ef8 record checkpoint resumes; its artifacts migrate to relative paths."""
    calls = []

    def fake_assessment(*_args, preconditioner_kind, **_kwargs):
        calls.append(preconditioner_kind)
        return {"status": "skipped", "verdict": "skipped"}

    monkeypatch.setattr(benchmark, "assess_brusselator_state", fake_assessment)
    config = _tiny_regeneration_config(tmp_path / "run")
    benchmark._records(config)
    checkpoint_path = Path(config.record_checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    for row in payload["records"].values():
        row["source_state_artifact"]["relative_path"] = str(
            Path(config.source_state_cache_dir) / row["source_state_artifact"]["relative_path"]
        )
    del payload["records"]["turing:0:fft_diffusion"]
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")
    if relocate:
        relocated = tmp_path / "relocated" / "source_states"
        shutil.copytree(config.source_state_cache_dir, relocated)
        shutil.rmtree(config.source_state_cache_dir)
        config = config._replace(source_state_cache_dir=str(relocated))
    calls.clear()

    records = benchmark._records(config)

    assert calls == ["fft_diffusion"]
    assert len(records) == 2
    persisted = json.loads(checkpoint_path.read_text(encoding="utf-8"))["records"]
    for row in [*records, *persisted.values()]:
        path = Path(row["source_state_artifact"]["relative_path"])
        assert not path.is_absolute() and len(path.parts) == 1


def test_partial_regeneration_rejects_a_legacy_checkpoint_for_another_artifact(
    tmp_path, monkeypatch
):
    """A legacy path naming a different artifact still fails the checkpoint contract."""
    monkeypatch.setattr(
        benchmark,
        "assess_brusselator_state",
        lambda *_args, **_kwargs: {"status": "skipped", "verdict": "skipped"},
    )
    config = _tiny_regeneration_config(tmp_path / "run")
    benchmark._records(config)
    checkpoint_path = Path(config.record_checkpoint_path)
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    payload["records"]["turing:0:identity"]["source_state_artifact"]["relative_path"] = str(
        Path(config.source_state_cache_dir) / "turing-0000000000000000.npz"
    )
    checkpoint_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match its source/operator contract"):
        benchmark._records(config)


def _fake_resolved_assessment(*_args, **_kwargs):
    return {
        "status": "completed",
        "verdict": "investigate",
        "verdict_reason": None,
        "disk_rate": 0.95,
        "epsilon_zero": 0.2,
        "n_right_real_outliers": 0,
        "supports_consistent": True,
        "supports_converged": True,
        "supports_corroborated": True,
        "corroboration_attempted": True,
        "origin_enclosed": False,
        "fov_imaginary_extent": 0.1,
        "fourier_weyl_ghost_lower_bound": {"status": "clears_adequacy_gate"},
    }


def test_unresolved_fov_checkpoint_in_the_base_format_resumes(tmp_path, monkeypatch):
    """A base report and partial FOV checkpoint written by 8be9ef8 resume and assemble."""
    checkpoint_dir = tmp_path / "checkpoint"
    cache_root = checkpoint_dir / "source_states"
    _, base_record, _ = _replay_fixture(tmp_path, cache_root)
    base_record["source_state_artifact"]["relative_path"] = _legacy_path(cache_root, base_record)
    records = []
    for kind in ("identity", "fft_diffusion"):
        record = json.loads(json.dumps(base_record))
        record["record_config"]["preconditioner_kind"] = kind
        record.update(
            {
                "regime": "turing",
                "preconditioner": kind,
                "state_index": 0,
                "status": "completed",
                "verdict": "indeterminate",
                "verdict_reason": None,
                "disk_rate": 0.5,
                "epsilon_zero": 0.2,
                "n_right_real_outliers": 0,
                "supports_consistent": False,
                "corroboration_attempted": True,
                "origin_enclosed": False,
                "fov_imaginary_extent": 0.1,
                "actual_gmres": None,
            }
        )
        records.append(record)
    report = {
        "status": "completed",
        "config": {"source_state_cache_dir": str(cache_root)},
        "records": records,
    }
    monkeypatch.setitem(resolver.EXPECTED_RECORDS, "screen_64", len(records))
    base_path = resolver._base_path(checkpoint_dir, "screen_64")
    resolver._atomic_json(base_path, report)
    identity_key, fft_key = (resolver._record_key(record) for record in records)
    attempt = resolver._attempt(_fake_resolved_assessment(), resolver.FOV_SUPPORT_LADDER[0])
    resolver._atomic_json(
        resolver._resolution_path(checkpoint_dir, "screen_64"),
        {
            "schema": resolver.RESOLUTION_SCHEMA,
            "base_result_path": str(base_path),
            "base_result_sha256": resolver._result_hash(base_path),
            "resolved": {
                identity_key: {
                    "status": "RESOLVED",
                    "source": "fov_support_escalation",
                    "attempts": [attempt],
                    "final": attempt,
                    "final_category": "investigate",
                    "final_verdict": "investigate",
                }
            },
        },
    )
    monkeypatch.setattr(benchmark, "assess_brusselator_state", _fake_resolved_assessment)

    resolver._resolve_one(checkpoint_dir, "screen_64", fft_key)
    final = resolver._attach_resolution(checkpoint_dir, "screen_64")

    assert [record["final_category"] for record in final["records"]] == [
        "investigate",
        "investigate",
    ]
    for record in final["records"]:
        assert record["source_state_artifact"]["relative_path"] == Path(
            base_record["source_state_artifact"]["relative_path"]
        ).name


def _short_arnoldi_counterexample():
    """Return Pavlov's deterministic incomplete-reading counterexample."""
    rng = np.random.default_rng(0)
    for _ in range(25):
        u = 1.0 + rng.uniform(-0.8, 0.8, (4, 4))
        v = 1.8 + rng.uniform(-1.0, 1.0, (4, 4))
    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    state = model.apply_bcs(
        {
            "u": jax.numpy.zeros((6, 6), dtype=jax.numpy.float64)
            .at[1:-1, 1:-1]
            .set(jax.numpy.asarray(u)),
            "v": jax.numpy.zeros((6, 6), dtype=jax.numpy.float64)
            .at[1:-1, 1:-1]
            .set(jax.numpy.asarray(v)),
        },
        0.0,
    )
    return assess_brusselator_state(
        state,
        model,
        fft_cache,
        diffusivities,
        0.2,
        TURING_REGIME,
        n_angles=4,
        fov_max_iters=60,
        fov_n_restarts=2,
        arnoldi_steps=1,
        seed=0,
    )


@pytest.mark.slow
def test_weak_bound_never_overrides_a_short_arnoldi_abstention():
    """Incomplete Ritz evidence remains indeterminate in module and resolver policy."""
    assessment = _short_arnoldi_counterexample()

    assert assessment["fourier_weyl_ghost_lower_bound"]["full_lower_bound"] < 0.1
    assert assessment["n_right_real_outliers"] is None
    assert assessment["verdict"] == "indeterminate"
    assert resolver._policy_category(assessment) == "indeterminate"


def test_weak_bound_never_overrides_a_nonfinite_reading():
    """A non-finite/invalid reading remains an abstention in both policy sites."""
    invalid = {
        "verdict": "indeterminate",
        "verdict_reason": "ritz contains a non-finite value",
        "disk_rate": float("nan"),
        "epsilon_zero": float("nan"),
        "n_right_real_outliers": None,
        "supports_consistent": True,
        "origin_enclosed": False,
        "fourier_weyl_ghost_lower_bound": {"status": "valid_but_below_adequacy_gate"},
    }
    module_assessment = SimpleNamespace(
        verdict="indeterminate", n_right_real_outliers=None
    )

    assert resolver._policy_category(invalid) == "indeterminate"
    assert resolver._policy_outcome(invalid) == ("indeterminate", invalid["verdict_reason"])
    from moljax.experimental import brusselator_conditioning as conditioning

    assert conditioning._weak_bound_override_eligible(module_assessment) is False


def _promoted_by_old_override(assessment):
    """Apply the pre-cce8089 weak-bound override to a stored assessment."""
    promoted = dict(assessment)
    promoted["verdict"] = "provisional"
    promoted["verdict_reason"] = "certification not established by the methods attempted"
    return promoted


def _stored_resolution(assessment):
    attempt = resolver._attempt(assessment, resolver.FOV_SUPPORT_LADDER[0])
    return {
        "status": "RESOLVED",
        "source": "fov_support_escalation",
        "attempts": [attempt],
        "final": attempt,
        "final_category": assessment["verdict"],
        "final_verdict": assessment["verdict"],
    }


@pytest.mark.parametrize(
    "defect",
    [
        {"n_right_real_outliers": None},
        {"disk_rate": float("nan")},
        {"epsilon_zero": float("inf")},
    ],
)
def test_reclassification_demotes_a_legacy_provisional_without_a_valid_reading(defect):
    """A stored verdict promoted from an unusable reading reclassifies to indeterminate."""
    stored = _promoted_by_old_override(
        {
            "status": "completed",
            "verdict": "indeterminate",
            "verdict_reason": None,
            "disk_rate": 1.25,
            "epsilon_zero": 1.04,
            "n_right_real_outliers": 0,
            "supports_consistent": True,
            "origin_enclosed": False,
            "fourier_weyl_ghost_lower_bound": {"status": "valid_but_below_adequacy_gate"},
            **defect,
        }
    )
    resolution = _stored_resolution(stored)

    normalised = resolver._normalise_resolution(json.loads(json.dumps(resolution)))

    assert normalised["final_category"] == normalised["final_verdict"] == "indeterminate"
    assert "cannot be recovered" in normalised["final_verdict_reason"]
    assert normalised["attempts"] == json.loads(json.dumps(resolution["attempts"]))
    assert normalised["final"]["assessment"]["verdict"] == "provisional"


def test_reclassification_keeps_a_legacy_provisional_with_a_valid_reading():
    """A provisional verdict backed by a measured reading is unchanged."""
    stored = _promoted_by_old_override(
        {
            "status": "completed",
            "verdict": "investigate",
            "verdict_reason": None,
            "disk_rate": 0.95,
            "epsilon_zero": 0.02,
            "n_right_real_outliers": 0,
            "supports_consistent": True,
            "origin_enclosed": False,
            "fourier_weyl_ghost_lower_bound": {"status": "valid_but_below_adequacy_gate"},
        }
    )

    normalised = resolver._normalise_resolution(_stored_resolution(stored))

    assert normalised["final_category"] == normalised["final_verdict"] == "provisional"
    assert normalised["final_verdict_reason"] == stored["verdict_reason"]


@pytest.mark.slow
def test_reclassification_demotes_the_promoted_short_arnoldi_counterexample():
    """The counterexample, promoted by the old override and stored, reclassifies."""
    stored = _promoted_by_old_override(_short_arnoldi_counterexample())
    assert stored["n_right_real_outliers"] is None

    normalised = resolver._normalise_resolution(_stored_resolution(stored))

    assert normalised["final_category"] == "indeterminate"
    assert normalised["attempts"][0]["assessment"]["verdict"] == "provisional"


def _resolved_reports() -> dict[str, dict]:
    """Load the four promoted final-policy reports committed with the study."""
    root = Path(__file__).resolve().parents[1] / "benchmarks" / "results"
    names = {
        "screen_64": "brusselator_conditioning.json",
        "developed_64": "brusselator_conditioning_developed.json",
        "fixed_dt_256": "brusselator_conditioning_fixed_dt.json",
        "hopf_continuation_256": "brusselator_conditioning_hopf_continuation.json",
    }
    import json

    return {study: json.loads((root / name).read_text()) for study, name in names.items()}


def test_promoted_resolved_reports_match_final_policy_and_tally():
    """Every published record and the aggregate tally use the terminal policy."""
    reports = _resolved_reports()
    tally = {
        "adequate": 0,
        "provisional": 0,
        "investigate": 0,
        "indeterminate": 0,
        "uncertified_at_cap": 0,
    }
    for report in reports.values():
        for record in report["records"]:
            assert record["verdict"] == record["final_verdict"] == record["final_category"]
            assert not Path(record["source_state_artifact"]["relative_path"]).is_absolute()
            assert ".." not in Path(record["source_state_artifact"]["relative_path"]).parts
            tally[record["verdict"]] += 1
    assert tally == {
        "adequate": 13,
        "provisional": 1,
        "investigate": 5,
        "indeterminate": 12,
        "uncertified_at_cap": 1,
    }


def test_promoted_resolved_report_aggregates_match_final_records():
    """All record-derived summaries are recomputed after FOV resolution."""
    reports = _resolved_reports()
    for study, report in reports.items():
        rebuilt = dict(report)
        resolver._recompute_derived_summaries(rebuilt, report["records"], study)
        for key in ("regime_comparison", "hopf_vs_turing", "fixed_dt_transition"):
            if key in report:
                assert report[key] == rebuilt[key]

    screen = reports["screen_64"]
    assert screen["regime_comparison"]["hopf"]["verdict_distribution"]["adequate"] == 2
    assert screen["regime_comparison"]["turing"]["verdict_distribution"]["adequate"] == 2
    assert screen["regime_comparison"]["hopf"]["median_disk_rate"] == pytest.approx(
        0.4404371091303861
    )
    assert screen["hopf_vs_turing"]["hopf_adequate_fft_records"] == 2
    assert screen["hopf_vs_turing"]["turing_adequate_fft_records"] == 2

    fixed = reports["fixed_dt_256"]
    for regime in ("hopf", "turing"):
        for kind in ("identity", "fft_diffusion"):
            rows = sorted(
                (
                    record
                    for record in fixed["records"]
                    if record["regime"] == regime and record["preconditioner"] == kind
                ),
                key=lambda record: record["trajectory_step"],
            )
            transition = fixed["fixed_dt_transition"]["by_regime"][regime][kind]
            assert transition["early"]["verdict"] == rows[0]["verdict"]
            assert transition["developed"]["verdict"] == rows[-1]["verdict"]
