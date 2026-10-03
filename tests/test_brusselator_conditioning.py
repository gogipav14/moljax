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
    linearization = conditioning.build_brusselator_linearization(
        state, model, fft_cache, diffusivities, 0.01
    )
    genuine = conditioning._fourier_weyl_bound(
        state,
        model,
        TURING_REGIME,
        0.01,
        preconditioner=linearization.preconditioner,
        context=linearization.context,
    )
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
    """A provisional verdict whose only failed gate is epsilon zero is unchanged."""
    stored = _promoted_by_old_override(
        {
            "status": "completed",
            "verdict": "investigate",
            "verdict_reason": None,
            "disk_rate": 0.5,
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


def _weak_bound_reading(**overrides):
    reading = {
        "status": "completed",
        "verdict": "investigate",
        "verdict_reason": None,
        "disk_rate": 0.5,
        "epsilon_zero": 0.02,
        "n_right_real_outliers": 0,
        "supports_consistent": True,
        "origin_enclosed": False,
        "fourier_weyl_ghost_lower_bound": {"status": "valid_but_below_adequacy_gate"},
    }
    reading.update(overrides)
    return reading


def test_weak_bound_promotes_investigate_only_when_epsilon_zero_is_the_failed_gate():
    """Synthetic investigate readings: only an epsilon-zero failure becomes provisional."""
    assert resolver.RATE_THRESHOLD == 0.9
    assert resolver.MAX_RIGHT_REAL_OUTLIERS == 0
    assert resolver._policy_outcome(_weak_bound_reading()) == (
        "provisional",
        "certification not established by the methods attempted",
    )
    # A failed disk-rate or outlier gate keeps investigate and its reason.
    for failed in ({"disk_rate": 0.95}, {"n_right_real_outliers": 2}):
        reading = _weak_bound_reading(verdict_reason="measured caution", **failed)
        assert resolver._policy_outcome(reading) == ("investigate", "measured caution")
    # Exactly at the rate threshold the disk gate passes.
    assert resolver._policy_category(_weak_bound_reading(disk_rate=0.9)) == "provisional"
    # Fail-closed gates keep their precedence.
    enclosed = _weak_bound_reading(verdict="indeterminate", origin_enclosed=True)
    assert resolver._policy_category(enclosed) == "indeterminate"


@pytest.mark.parametrize("failed", [{"disk_rate": 0.95}, {"n_right_real_outliers": 1}])
def test_reclassification_restores_investigate_for_a_legacy_promotion_past_a_gate(failed):
    """A stored provisional promoted past a failed threshold gate is investigate."""
    stored = _promoted_by_old_override(_weak_bound_reading(**failed))

    normalised = resolver._normalise_resolution(_stored_resolution(stored))

    assert normalised["final_category"] == normalised["final_verdict"] == "investigate"
    assert "failed threshold gate" in normalised["final_verdict_reason"]
    assert normalised["final"]["assessment"]["verdict"] == "provisional"


def test_committed_late_turing_identity_record_is_investigate():
    """The review reproduction: bound 0, disk_rate 0.998 above the 0.9 gate."""
    report = _resolved_reports()["fixed_dt_256"]
    (record,) = [
        record
        for record in report["records"]
        if record["regime"] == "turing"
        and record["preconditioner"] == "identity"
        and record["trajectory_step"] == 1000
    ]
    assessment = record["geometry_resolution"]["final"]["assessment"]
    assert assessment["verdict"] == "investigate"
    assert assessment["fourier_weyl_ghost_lower_bound"]["status"] == (
        "valid_but_below_adequacy_gate"
    )
    assert assessment["fourier_weyl_ghost_lower_bound"]["full_lower_bound"] == 0.0
    assert assessment["disk_rate"] == pytest.approx(0.9980383387, abs=1.0e-10)
    assert resolver._policy_category(assessment) == "investigate"
    assert record["verdict"] == record["final_verdict"] == "investigate"


def test_module_weak_bound_eligibility_reads_the_assessment_gates():
    """The module policy uses the thresholds the assessment applied."""
    from moljax.experimental import brusselator_conditioning as conditioning

    def assessment(**overrides):
        values = {
            "verdict": "investigate",
            "n_right_real_outliers": 0,
            "supports_consistent": True,
            "disk_rate": 0.5,
            "rate_threshold": 0.9,
            "max_right_real_outliers": 0,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    assert conditioning._weak_bound_override_eligible(assessment()) is True
    assert conditioning._weak_bound_override_eligible(assessment(verdict="provisional"))
    assert conditioning._weak_bound_override_eligible(assessment(disk_rate=0.95)) is False
    assert (
        conditioning._weak_bound_override_eligible(assessment(disk_rate=0.95, rate_threshold=1.0))
        is True
    )
    assert (
        conditioning._weak_bound_override_eligible(assessment(n_right_real_outliers=1)) is False
    )
    assert (
        conditioning._weak_bound_override_eligible(assessment(supports_consistent=False))
        is False
    )


@pytest.mark.slow
def test_weak_bound_keeps_a_disk_rate_investigate_in_the_module():
    """Module reproduction: 8x8 identity at dt=0.4 has disk rate 0.958 > 0.9."""
    import moljax.experimental.brusselator_conditioning as conditioning

    grid = Grid2D.uniform(8, 8, 0.0, 5.0, 0.0, 5.0)
    state, model, fft_cache, diffusivities = _homogeneous_state(TURING_REGIME, grid)
    linearization = conditioning.build_brusselator_linearization(
        state, model, fft_cache, diffusivities, 0.4, preconditioner_kind="identity"
    )
    genuine = conditioning._fourier_weyl_bound(
        state,
        model,
        TURING_REGIME,
        0.4,
        preconditioner=linearization.preconditioner,
        context=linearization.context,
    )
    weak = replace(genuine, selected=replace(genuine.selected, full_lower_bound=0.0))
    original = conditioning._fourier_weyl_bound
    conditioning._fourier_weyl_bound = lambda *_args, **_kwargs: weak
    try:
        assessment = assess_brusselator_state(
            state,
            model,
            fft_cache,
            diffusivities,
            0.4,
            TURING_REGIME,
            preconditioner_kind="identity",
            n_angles=8,
            fov_max_iters=60,
            arnoldi_steps=6,
            seed=20260821,
        )
    finally:
        conditioning._fourier_weyl_bound = original
    assert assessment["fourier_weyl_ghost_lower_bound"]["status"] == (
        "valid_but_below_adequacy_gate"
    )
    assert assessment["disk_rate"] > 0.9
    assert assessment["origin_enclosed"] is False
    assert assessment["verdict"] == "investigate"
    assert resolver._policy_category(assessment) == "investigate"


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
        "provisional": 0,
        "investigate": 6,
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


# --- Fourier--Weyl--ghost certificate operator guards -------------------------


def _patterned_state(model, grid, alpha):
    """Return a padded state: steady state plus a smooth periodic pattern."""
    regime = TURING_REGIME
    rows, columns = np.meshgrid(
        np.arange(grid.ny_total), np.arange(grid.nx_total), indexing="ij"
    )
    phi = np.cos(2.0 * np.pi * columns / grid.nx) * np.cos(2.0 * np.pi * rows / grid.ny)
    return model.apply_bcs(
        {
            "u": jax.numpy.asarray(regime.a + alpha * phi),
            "v": jax.numpy.asarray(regime.b / regime.a - 0.35 * alpha * phi),
        },
        0.0,
    )


def _certificate_and_dense(grid, dt, *, alpha=0.0, kind="fft_diffusion"):
    """Return the helper's certificate and the actual padded operator's sigma_min."""
    from moljax.experimental import brusselator_conditioning as conditioning

    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    state = _patterned_state(model, grid, alpha)
    linearization = conditioning.build_brusselator_linearization(
        state, model, fft_cache, diffusivities, dt, preconditioner_kind=kind
    )
    operator = linearization.operator
    basis = jax.numpy.eye(operator.n, dtype=jax.numpy.float64)
    matrix = np.column_stack(
        [np.asarray(operator.matvec(basis[:, column])) for column in range(operator.n)]
    )
    dense = float(np.linalg.svd(matrix, compute_uv=False)[-1])

    def certificate():
        return conditioning._fourier_weyl_bound(
            state,
            model,
            TURING_REGIME,
            dt,
            preconditioner=linearization.preconditioner,
            context=linearization.context,
        )

    return certificate, dense, state, model, fft_cache, diffusivities


def test_certificate_refuses_three_ghost_layers_instead_of_certifying():
    """The n_ghost=3 counterexample: the one-layer formula would overstate sigma_min."""
    from moljax.experimental.brusselator_conditioning import CertificateNotApplicable
    from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
        fourier_weyl_ghost_lower_bound,
    )

    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=3)
    certificate, dense, state, model, fft_cache, diffusivities = _certificate_and_dense(
        grid, 0.2
    )
    interior_y, interior_x = grid.interior_slice
    one_layer = fourier_weyl_ghost_lower_bound(
        np.asarray(state["u"])[interior_y, interior_x],
        np.asarray(state["v"])[interior_y, interior_x],
        du=TURING_REGIME.du,
        dv=TURING_REGIME.dv,
        a=TURING_REGIME.a,
        beta=TURING_REGIME.b,
        dt=0.2,
    ).selected.full_lower_bound
    assert dense == pytest.approx(0.540558, abs=5.0e-7)
    assert one_layer == pytest.approx(0.598697, abs=5.0e-7)
    assert one_layer > dense
    with pytest.raises(CertificateNotApplicable, match="n_ghost == 1"):
        certificate()

    assessment = assess_brusselator_state(
        state, model, fft_cache, diffusivities, 0.2, TURING_REGIME, seed=20260821
    )
    record = assessment["fourier_weyl_ghost_lower_bound"]
    assert record["status"] == "not_applicable"
    assert "n_ghost" in record["reason"]
    assert record["full_lower_bound"] is None
    assert assessment["epsilon_zero_full_operator_evidence"] is False
    assert assessment["epsilon_zero"] == assessment["reduced_arnoldi_epsilon_zero"]
    assert assessment["verdict"] != "adequate"
    assert resolver._policy_outcome(assessment)[0] == assessment["verdict"]


@pytest.mark.parametrize("kind", ["fft_diffusion", "identity"])
def test_certificate_on_a_nonsquare_grid_uses_its_dimensions(kind):
    """A 4x6 grid with dx != dy gets a bound from its own geometry, below dense."""
    from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
        fourier_weyl_ghost_lower_bound,
    )

    grid = Grid2D.uniform(4, 6, 0.0, 5.0, 0.0, 3.0, n_ghost=1)
    assert grid.dx != grid.dy
    certificate, dense, state, _, _, _ = _certificate_and_dense(
        grid, 0.2, alpha=0.3, kind=kind
    )
    bound = certificate()
    expected = fourier_weyl_ghost_lower_bound(
        np.asarray(state["u"])[1:-1, 1:-1],
        np.asarray(state["v"])[1:-1, 1:-1],
        du=TURING_REGIME.du,
        dv=TURING_REGIME.dv,
        a=TURING_REGIME.a,
        beta=TURING_REGIME.b,
        dt=0.2,
        domain_length_x=5.0,
        domain_length_y=3.0,
    )
    assert bound.selected.full_lower_bound == expected.selected.full_lower_bound
    for candidate in bound.candidates:
        assert candidate.full_lower_bound <= dense + 5.0e-13


def test_certificate_uses_the_grid_length_not_the_regime_length():
    """At L=1 the regime's L=5 symbol would certify above the true sigma_min."""
    from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
        fourier_weyl_ghost_lower_bound,
    )

    grid = Grid2D.uniform(4, 4, 0.0, 1.0, 0.0, 1.0, n_ghost=1)
    certificate, dense, state, _, _, _ = _certificate_and_dense(grid, 1.0)
    regime_length = fourier_weyl_ghost_lower_bound(
        np.asarray(state["u"])[1:-1, 1:-1],
        np.asarray(state["v"])[1:-1, 1:-1],
        du=TURING_REGIME.du,
        dv=TURING_REGIME.dv,
        a=TURING_REGIME.a,
        beta=TURING_REGIME.b,
        dt=1.0,
        domain_length_x=TURING_REGIME.domain_length,
    ).selected.full_lower_bound
    bound = certificate().selected.full_lower_bound
    assert regime_length > dense
    assert bound <= dense + 5.0e-13
    assert bound == pytest.approx(0.129543, abs=5.0e-7)


def _guard_inputs(grid, *, regime=TURING_REGIME, dt=0.2, kind="fft_diffusion"):
    from moljax.core.preconditioners import PrecondContext
    from moljax.experimental import brusselator_conditioning as conditioning

    model, fft_cache, _ = build_brusselator_system(regime, grid)
    preconditioner = conditioning._preconditioner(kind, fft_cache)
    context = PrecondContext(grid=model.grid, dt=dt, params=model.params)
    return model, preconditioner, context


def _refused(model, regime, dt, preconditioner, context, match):
    from moljax.experimental import brusselator_conditioning as conditioning

    with pytest.raises(conditioning.CertificateNotApplicable, match=match):
        conditioning._validate_certificate_operator(model, regime, dt, preconditioner, context)


def test_certificate_refuses_operators_outside_its_assumptions():
    """Parameter, dt, boundary, and preconditioner mismatches are all refused."""
    from moljax.core.bc import BCType
    from moljax.core.fft_solvers import create_fft_cache
    from moljax.core.model import MOLModel, create_brusselator_model
    from moljax.core.preconditioners import (
        BlockJacobiPreconditioner,
        PrecondContext,
        create_fft_preconditioner,
    )
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, preconditioner, context = _guard_inputs(grid)
    assert conditioning._validate_certificate_operator(
        model, TURING_REGIME, 0.2, preconditioner, context
    ) == grid

    # Reaction parameters and diffusivities must be the certificate inputs.
    _refused(model, HOPF_REGIME, 0.2, preconditioner, context, "model parameter")
    _refused(model, TURING_REGIME._replace(du=0.02), 0.2, preconditioner, context, "Du")
    # dt must be the linearization's dt.
    _refused(model, TURING_REGIME, 0.1, preconditioner, context, "dt")
    # The context must carry the model's diffusivities and grid.
    foreign_params = PrecondContext(grid=grid, dt=0.2, params={**model.params, "Dv": 0.5})
    _refused(model, TURING_REGIME, 0.2, preconditioner, foreign_params, "context Dv")
    # Non-periodic boundaries are outside the bound.
    neumann = create_brusselator_model(
        grid, Du=0.01, Dv=0.1, a=1.0, b=1.8, bc_type=BCType.NEUMANN
    )
    _refused(neumann, TURING_REGIME, 0.2, preconditioner, context, "periodic")
    # Extra operators change the Jacobian.
    extra = MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops + model.linear_ops,
        nonlinear_ops=model.nonlinear_ops,
        metadata=model.metadata,
    )
    _refused(extra, TURING_REGIME, 0.2, preconditioner, context, "shipped Brusselator")
    # Preconditioners other than identity and the matching FFT symbol are refused.
    _refused(
        model,
        TURING_REGIME,
        0.2,
        BlockJacobiPreconditioner(),
        context,
        "BlockJacobiPreconditioner",
    )
    swapped = create_fft_preconditioner({"u": "Dv", "v": "Du"}, preconditioner.fft_cache)
    _refused(model, TURING_REGIME, 0.2, swapped, context, "map u to Du")
    other_grid = Grid2D.uniform(4, 4, 0.0, 2.0, 0.0, 2.0, n_ghost=1)
    foreign_cache = create_fft_preconditioner(
        {"u": "Du", "v": "Dv"}, create_fft_cache(other_grid)
    )
    _refused(model, TURING_REGIME, 0.2, foreign_cache, context, "symbol")


def test_certificate_guard_accepts_every_committed_adequate_record():
    """All 13 adequate published records have an operator the bound covers."""
    from moljax.experimental import brusselator_conditioning as conditioning

    adequate = [
        record
        for report in _resolved_reports().values()
        for record in report["records"]
        if record["verdict"] == "adequate"
    ]
    assert len(adequate) == 13
    for record in adequate:
        stored = record["record_config"]
        regime = conditioning.BrusselatorRegime(**stored["regime"])
        length = float(stored["domain_length"])
        grid = Grid2D.uniform(
            int(stored["grid"]["nx"]),
            int(stored["grid"]["ny"]),
            0.0,
            length,
            0.0,
            length,
            n_ghost=int(stored["grid"]["n_ghost"]),
        )
        dt = float(stored["analysis_dt"])
        model, preconditioner, context = _guard_inputs(
            grid, regime=regime, dt=dt, kind=stored["preconditioner_kind"]
        )
        assert (
            conditioning._validate_certificate_operator(
                model, regime, dt, preconditioner, context
            )
            == grid
        )
        assert record["fourier_weyl_ghost_lower_bound"]["status"] == "clears_adequacy_gate"


def _with_reaction(model, apply):
    """Return ``model`` with its reaction action replaced, name and params kept."""
    from moljax.core.model import MOLModel

    reaction = replace(model.nonlinear_ops[0], apply=apply)
    return MOLModel(
        grid=model.grid,
        bc_spec=model.bc_spec,
        params=model.params,
        linear_ops=model.linear_ops,
        nonlinear_ops=(reaction,),
        metadata=model.metadata,
    )


def _guard_bound_and_dense(model, dt, preconditioner, state=None):
    """Return (guard outcome, dense sigma_min), by default for a homogeneous Turing state."""
    from moljax.conditioning import linearized_operator
    from moljax.core.newton_krylov import create_implicit_residual
    from moljax.core.preconditioners import PrecondContext
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = model.grid
    if state is None:
        shape = (grid.ny_total, grid.nx_total)
        state = model.apply_bcs(
            {
                "u": jax.numpy.full(shape, TURING_REGIME.a),
                "v": jax.numpy.full(shape, TURING_REGIME.b / TURING_REGIME.a),
            },
            0.0,
        )
    context = PrecondContext(grid=grid, dt=dt, params=model.params)
    residual = create_implicit_residual(model, state, dt, dt, method="be")
    operator = linearized_operator(
        residual, state, preconditioner=preconditioner, context=context
    )
    basis = jax.numpy.eye(operator.n, dtype=jax.numpy.float64)
    matrix = np.column_stack(
        [np.asarray(operator.matvec(basis[:, column])) for column in range(operator.n)]
    )
    dense = float(np.linalg.svd(matrix, compute_uv=False)[-1])
    try:
        outcome = conditioning._fourier_weyl_bound(
            state,
            model,
            TURING_REGIME,
            dt,
            preconditioner=preconditioner,
            context=context,
        ).selected.full_lower_bound
    except conditioning.CertificateNotApplicable as refusal:
        outcome = refusal
    return outcome, dense


def test_certificate_refuses_replaced_actions_with_shipped_names():
    """Review reproductions: same-name replacement actions are refused, not certified."""
    from moljax.core.preconditioners import IdentityPreconditioner
    from moljax.experimental import brusselator_conditioning as conditioning

    class ScaledIdentity(IdentityPreconditioner):
        def apply(self, r, context):
            return {name: 0.01 * value for name, value in r.items()}

    # (a) the original reaction plus 5 times each field, 4x4, L=5, dt=0.2.
    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, _ = build_brusselator_system(TURING_REGIME, grid)
    original = model.nonlinear_ops[0].apply

    def plus_five(state, grid, t, params):
        image = original(state, grid, t, params)
        return {name: image[name] + 5.0 * state[name] for name in image}

    fft = conditioning._preconditioner("fft_diffusion", fft_cache)
    outcome, dense = _guard_bound_and_dense(_with_reaction(model, plus_five), 0.2, fft)
    # The old unguarded certificate was about 0.599, above this true dense value.
    # Keep that counterexample without tying the SVD to one XLA implementation.
    assert 0.0 < dense < 0.5986968893
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "reaction action" in str(outcome)

    # (b) the reaction replaced by 5 * state, 3x3, L=5, dt=0.2.
    grid3 = Grid2D.uniform(3, 3, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model3, fft_cache3, _ = build_brusselator_system(TURING_REGIME, grid3)
    scaled = _with_reaction(
        model3, lambda state, grid, t, params: {name: 5.0 * state[name] for name in state}
    )
    fft3 = conditioning._preconditioner("fft_diffusion", fft_cache3)
    outcome, dense = _guard_bound_and_dense(scaled, 0.2, fft3)
    assert dense < 1.0e-12
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "reaction action" in str(outcome)

    # (c) an IdentityPreconditioner subclass that applies 0.01 * r, 3x3.
    outcome, dense = _guard_bound_and_dense(model3, 0.2, ScaledIdentity())
    # This too is far below the old spurious approximately-0.599 certificate.
    assert 0.0 < dense < 0.5986968893
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "ScaledIdentity" in str(outcome)

    # A replaced diffusion action and an instance-level apply are refused too.
    diffusion = replace(
        model.linear_ops[0], apply=lambda state, grid, t, params: dict(state)
    )
    swapped = replace(model, linear_ops=(diffusion,))
    outcome, _ = _guard_bound_and_dense(swapped, 0.2, fft)
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "diffusion action" in str(outcome)
    shadowed = IdentityPreconditioner()
    object.__setattr__(shadowed, "apply", ScaledIdentity().apply)
    _refused(model, TURING_REGIME, 0.2, shadowed, _guard_inputs(grid)[2], "Identity")

    # The shipped operators on the same grids are accepted, below dense.
    for shipped_model, shipped_cache in ((model, fft_cache), (model3, fft_cache3)):
        for kind in ("fft_diffusion", "identity"):
            preconditioner = conditioning._preconditioner(kind, shipped_cache)
            outcome, dense = _guard_bound_and_dense(shipped_model, 0.2, preconditioner)
            assert not isinstance(outcome, Exception)
            assert outcome <= dense + 5.0e-13


def _plus_five_state(original):
    """Return a replacement for a model method: its image plus 5 times each field."""

    def replaced(state, t=0.0):
        image = original(state, t)
        return {name: image[name] + 5.0 * state[name] for name in image}

    return replaced


def test_certificate_refuses_an_instance_level_rhs_replacement():
    """Review reproduction: model.rhs + 5 * state on 4x4, L=5, dt=0.2 was certified."""
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, _ = build_brusselator_system(TURING_REGIME, grid)
    fft = conditioning._preconditioner("fft_diffusion", fft_cache)
    shipped_bound, shipped_dense = _guard_bound_and_dense(model, 0.2, fft)
    assert not isinstance(shipped_bound, Exception)
    assert shipped_bound <= shipped_dense + 5.0e-13

    object.__setattr__(model, "rhs", _plus_five_state(model.rhs))
    outcome, dense = _guard_bound_and_dense(model, 0.2, fft)
    # The old guard returned the shipped bound (about 0.599) for this operator,
    # whose dense sigma_min is numerically zero.
    assert 0.0 <= dense < 1.0e-8
    assert dense < shipped_bound
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "instance-level replacement of MOLModel.rhs" in str(outcome)


@pytest.mark.parametrize("name", ["rhs", "apply_bcs", "linear_rhs", "nonlinear_rhs"])
def test_certificate_refuses_replaced_model_methods(name, monkeypatch):
    """Instance attributes, rebound methods, and class patches are all refused."""
    from moljax.core.model import MOLModel
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(4, 4, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, preconditioner, context = _guard_inputs(grid)
    object.__setattr__(model, name, _plus_five_state(getattr(model, name)))
    _refused(model, TURING_REGIME, 0.2, preconditioner, context, "instance-level")

    # The shipped function bound to another model is still an instance attribute.
    model, preconditioner, context = _guard_inputs(grid)
    other, _, _ = _guard_inputs(grid)
    object.__setattr__(model, name, getattr(other, name))
    _refused(model, TURING_REGIME, 0.2, preconditioner, context, "instance-level")

    # A replacement on the class itself is refused for every model.
    model, preconditioner, context = _guard_inputs(grid)
    monkeypatch.setattr(MOLModel, name, _plus_five_state(conditioning._SHIPPED_MODEL_ACTIONS[name]))
    _refused(model, TURING_REGIME, 0.2, preconditioner, context, "shipped class function")
    monkeypatch.undo()
    assert (
        conditioning._validate_certificate_operator(
            model, TURING_REGIME, 0.2, preconditioner, context
        )
        == grid
    )


def _constant_state(shape, dtype=None):
    """Return the constant fields u=1, v=1.8 with the given full array shape."""
    dtype = jax.numpy.float64 if dtype is None else dtype
    return {
        "u": jax.numpy.full(shape, TURING_REGIME.a, dtype=dtype),
        "v": jax.numpy.full(shape, TURING_REGIME.b / TURING_REGIME.a, dtype=dtype),
    }


def test_certificate_refuses_oversized_state_arrays():
    """Review reproduction: 6x6 arrays on a 2x2 grid passed the interior-slice check."""
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(2, 2, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    fft = conditioning._preconditioner("fft_diffusion", fft_cache)

    # Correctly sized control: (ny + 2, nx + 2) = (4, 4) still certifies below dense.
    bound, dense = _guard_bound_and_dense(
        model, 0.2, fft, state=_constant_state((grid.ny_total, grid.nx_total))
    )
    assert not isinstance(bound, Exception)
    assert 0.0 < bound <= dense + 5.0e-13

    # Oversized arrays: the old guard returned the same bound (about 0.599), above
    # the dense sigma_min of the actual 72-component operator.
    outcome, oversized_dense = _guard_bound_and_dense(
        model, 0.2, fft, state=_constant_state((6, 6))
    )
    assert 0.0 < oversized_dense < bound
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "shape (6, 6)" in str(outcome)

    # The study entry points refuse the state before any linearization.
    with pytest.raises(ValueError, match="requires \\(4, 4\\)"):
        conditioning.build_brusselator_linearization(
            _constant_state((6, 6)), model, fft_cache, diffusivities, 0.2
        )
    with pytest.raises(ValueError, match="shape"):
        assess_brusselator_state(
            _constant_state((6, 6)), model, fft_cache, diffusivities, 0.2, TURING_REGIME
        )


def test_certificate_and_linearization_refuse_a_wrong_state_layout():
    """Field names, undersized arrays, and a non-float64 dtype are refused."""
    from moljax.core.preconditioners import PrecondContext
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(2, 2, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, diffusivities = build_brusselator_system(TURING_REGIME, grid)
    fft = conditioning._preconditioner("fft_diffusion", fft_cache)
    context = PrecondContext(grid=grid, dt=0.2, params=model.params)
    correct = _constant_state((4, 4))
    cases = (
        ({**correct, "w": correct["u"]}, "exactly the fields u and v"),
        ({"u": correct["u"], "v": correct["v"][:, :3]}, "field v has shape"),
        (_constant_state((4, 4), jax.numpy.float32), "float64 is required"),
    )
    for state, match in cases:
        with pytest.raises(conditioning.CertificateNotApplicable, match=match):
            conditioning._fourier_weyl_bound(
                state, model, TURING_REGIME, 0.2, preconditioner=fft, context=context
            )
        with pytest.raises(ValueError, match=match):
            conditioning.build_brusselator_linearization(
                state, model, fft_cache, diffusivities, 0.2
            )


@pytest.mark.parametrize("n, zero_mode", [(4, -100.0), (4, 40.0), (3, -50.0)])
def test_certificate_refuses_a_wrong_zero_mode_under_a_large_symbol(n, zero_mode):
    """Review reproductions: L=1e-6 makes a global tolerance admit a wrong zero mode."""
    from moljax.core.preconditioners import create_fft_preconditioner
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(n, n, 0.0, 1.0e-6, 0.0, 1.0e-6, n_ghost=1)
    model, fft_cache, _ = build_brusselator_system(TURING_REGIME, grid)
    bad = fft_cache._replace(
        laplacian_symbol=fft_cache.laplacian_symbol.at[0, 0].set(zero_mode)
    )
    preconditioner = create_fft_preconditioner({"u": "Du", "v": "Dv"}, bad)
    outcome, dense = _guard_bound_and_dense(model, 0.2, preconditioner)
    if zero_mode < 0.0:
        # The old global-tolerance check could certify approximately 0.599 here;
        # any positive dense value below it demonstrates the counterexample without
        # depending on a hardware-specific dense-SVD bit pattern.
        assert 0.0 < dense < 0.5986968893
    else:
        # The +40 zero mode gives |1 - dt Dv l_0|^-1 = 1 / |1 - 0.8| = 5.
        assert 1.0 / abs(1.0 - 0.2 * TURING_REGIME.dv * zero_mode) == pytest.approx(5.0)
    assert isinstance(outcome, conditioning.CertificateNotApplicable)
    assert "zero mode" in str(outcome)


def test_certificate_refuses_a_small_symbol_entry_error_and_accepts_shipped_caches():
    """Every symbol entry has its own tolerance; shipped caches pass on several grids."""
    from moljax.core.fft_solvers import create_fft_cache_2d_rfft
    from moljax.core.preconditioners import PrecondContext, create_fft_preconditioner
    from moljax.experimental import brusselator_conditioning as conditioning

    grid = Grid2D.uniform(64, 64, 0.0, 5.0, 0.0, 5.0, n_ghost=1)
    model, fft_cache, _ = build_brusselator_system(TURING_REGIME, grid)
    context = PrecondContext(grid=grid, dt=0.2, params=model.params)
    symbol = np.asarray(fft_cache.laplacian_symbol)
    # An absolute error of half the old global tolerance (1e-12 times the
    # largest entry) in the smallest nonzero entry is a relative error of
    # about 4e-10 there: the old check admitted it, the per-entry check
    # refuses it.
    smallest = np.where(symbol == 0.0, -np.inf, symbol)
    row, column = np.unravel_index(np.argmax(smallest), symbol.shape)
    scale = float(np.max(np.abs(symbol)))
    wrong = symbol.copy()
    wrong[row, column] += 0.5e-12 * scale
    assert np.allclose(wrong, symbol, rtol=1.0e-12, atol=1.0e-12 * scale)
    assert abs(wrong[row, column] - symbol[row, column]) > 1.0e-10 * abs(symbol[row, column])
    perturbed = fft_cache._replace(laplacian_symbol=jax.numpy.asarray(wrong))
    _refused(
        model,
        TURING_REGIME,
        0.2,
        create_fft_preconditioner({"u": "Du", "v": "Dv"}, perturbed),
        context,
        "finite-difference Laplacian symbol",
    )

    shipped_grids = (
        (3, 3, 5.0),
        (4, 6, 1.0e-6),
        (16, 16, 1.0),
        (64, 64, 5.0),
        (256, 256, 5.0),
    )
    for nx, ny, length in shipped_grids:
        shipped_grid = Grid2D.uniform(nx, ny, 0.0, length, 0.0, length, n_ghost=1)
        for dt in (0.01, 0.2, 1.0):
            shipped_model, preconditioner, shipped_context = _guard_inputs(shipped_grid, dt=dt)
            assert (
                conditioning._validate_certificate_operator(
                    shipped_model, TURING_REGIME, dt, preconditioner, shipped_context
                )
                == shipped_grid
            )
        rfft = create_fft_preconditioner(
            {"u": "Du", "v": "Dv"}, create_fft_cache_2d_rfft(shipped_grid)
        )
        assert (
            conditioning._validate_certificate_operator(
                shipped_model, TURING_REGIME, dt, rfft, shipped_context
            )
            == shipped_grid
        )


# --- Base-report Hopf/Turing conclusion ----------------------------------------


def _synthetic_screen_record(regime, state_index, kind, verdict, supports_consistent):
    return {
        "regime": regime,
        "state_index": state_index,
        "preconditioner": kind,
        "status": "completed",
        "verdict": verdict,
        "supports_consistent": supports_consistent,
        "disk_rate": 0.5,
        "fov_imaginary_extent": 0.1,
        "origin_enclosed": False,
        "time": 0.1 * (state_index + 1),
        "actual_gmres": {"iterations": 5, "converged": True},
    }


def _screen_records(verdicts):
    """Build screen_64-shaped records from ``{regime: (verdict, supports_consistent)}``."""
    return [
        _synthetic_screen_record(regime, index, kind, *verdicts[regime])
        for regime in ("hopf", "turing")
        for index in range(2)
        for kind in ("identity", "fft_diffusion")
    ]


def test_base_screen_report_declares_adequacy_only_when_records_are_adequate():
    report = benchmark._result(
        benchmark.SCREEN_64,
        _screen_records({"hopf": ("adequate", True), "turing": ("adequate", True)}),
    )
    assert report["hopf_vs_turing"]["outcome"] == "both_adequate_under_fft"
    assert report["hopf_vs_turing"]["hopf_adequate_fft_records"] == 2
    assert report["hopf_vs_turing"]["turing_adequate_fft_records"] == 2


def test_base_screen_report_does_not_declare_adequacy_for_unresolved_records():
    """The screen_64 reproduction: indeterminate with inconsistent supports."""
    records = _screen_records(
        {"hopf": ("indeterminate", False), "turing": ("indeterminate", False)}
    )
    summary = benchmark._result(benchmark.SCREEN_64, records)["hopf_vs_turing"]
    assert summary["outcome"] == "fft_regime_assessments_unresolved"
    assert "adequate" not in summary["outcome"]
    assert summary["fft_status_by_regime"] == {"hopf": "unresolved", "turing": "unresolved"}
    assert summary["unresolved_fft_records_by_regime"] == {"hopf": 2, "turing": 2}
    assert summary["hopf_adequate_fft_records"] == summary["turing_adequate_fft_records"] == 0


@pytest.mark.parametrize(
    "verdicts",
    [
        {"hopf": ("adequate", True), "turing": ("indeterminate", False)},
        {"hopf": ("investigate", True), "turing": ("adequate", True)},
    ],
)
def test_base_screen_report_reports_mixed_outcomes_per_regime(verdicts):
    summary = benchmark._result(benchmark.SCREEN_64, _screen_records(verdicts))[
        "hopf_vs_turing"
    ]
    assert summary["outcome"] == "fft_regime_assessments_mixed"
    statuses = summary["fft_status_by_regime"]
    for regime, (verdict, _) in verdicts.items():
        expected = {
            "adequate": "adequate",
            "indeterminate": "unresolved",
            "investigate": "not_adequate",
        }[verdict]
        assert statuses[regime] == expected


def test_base_developed_report_derives_its_conclusion_from_records():
    """The developed_64 base conclusion is no longer hard-coded either."""
    records = []
    for regime, verdict in (("hopf", "indeterminate"), ("turing", "adequate")):
        for step in (1, 2):
            for kind in ("identity", "fft_diffusion"):
                record = _synthetic_screen_record(regime, 0, kind, verdict, True)
                del record["state_index"]
                record["trajectory_step"] = step
                record["developedness"] = {"max_abs_u_minus_steady": 0.1}
                record["origin_enclosed"] = verdict == "indeterminate"
                records.append(record)
    summary = benchmark._result(benchmark.DEVELOPED_64, records)["hopf_vs_turing"]
    assert summary["outcome"] == "developed_fft_regime_assessments_mixed"
    assert summary["both_regimes_indeterminate"] is False
    assert summary["turing_nonadequate_fft_records"] == 0
    assert summary["turing_origin_enclosed_any"] is False


def _developed_fft_records(enclosed, consistent, extents):
    """Indeterminate developed_64 FFT records with the given geometry and extents."""
    return [
        {
            "regime": regime,
            "preconditioner": "fft_diffusion",
            "verdict": "indeterminate",
            "supports_consistent": consistent[regime],
            "origin_enclosed": enclosed[regime],
            "n_right_real_outliers": None,
            "trajectory_step": step,
            "time": float(step),
            "fov_imaginary_extent": extent,
        }
        for regime in ("hopf", "turing")
        for step, extent in enumerate(extents[regime], start=1)
    ]


def test_developed_report_does_not_claim_enclosure_or_extent_without_evidence():
    """Review reproduction: unresolved geometry and equal constant extents."""
    records = _developed_fft_records(
        {"hopf": False, "turing": False},
        {"hopf": False, "turing": False},
        {"hopf": (1.0, 1.0, 1.0), "turing": (1.0, 1.0, 1.0)},
    )
    summary = benchmark._hopf_vs_turing(records, "developed_64")
    assert summary["both_regimes_indeterminate"] is True
    assert summary["fft_origin_enclosure_corroborated"] is False
    assert summary["hopf_fov_imaginary_extent_exceeds_turing_at_every_sample"] is False
    assert summary["hopf_fov_imaginary_extent_grows_over_samples"] is False
    assert "corroborated FOV diagnostics enclose" not in summary["statement"]
    assert "unresolved" in summary["statement"]
    assert "larger" not in summary["statement"]
    assert "neither exceeds" in summary["statement"]


@pytest.mark.parametrize(
    ("enclosed", "consistent"),
    [
        ({"hopf": True, "turing": False}, {"hopf": True, "turing": True}),
        ({"hopf": True, "turing": True}, {"hopf": True, "turing": False}),
    ],
)
def test_developed_report_needs_every_record_for_the_enclosure_claim(enclosed, consistent):
    records = _developed_fft_records(
        enclosed, consistent, {"hopf": (3.0, 4.0, 5.0), "turing": (1.0, 2.0, 2.5)}
    )
    summary = benchmark._hopf_vs_turing(records, "developed_64")
    assert summary["fft_origin_enclosure_corroborated"] is False
    assert "corroborated FOV diagnostics enclose" not in summary["statement"]
    assert summary["statement"].endswith(
        "Hopf has the larger imaginary extent at every sample, and its extent grows "
        "over the samples."
    )


@pytest.mark.parametrize(
    ("extents", "larger", "grows", "ending"),
    [
        ({"hopf": (3.0, 4.0, 5.0), "turing": (1.0, 2.0, 2.5)}, True, True, "grows over the samples."),
        ({"hopf": (5.0, 4.0, 3.0), "turing": (1.0, 2.0, 2.5)}, True, False, "does not grow over the samples."),
        ({"hopf": (3.0, 4.0, 5.0), "turing": (1.0, 2.0, 7.0)}, False, True, "not larger than the Turing extent at every sample."),
        ({"hopf": (3.0, 3.0, 3.0), "turing": (3.0, 3.0, 3.0)}, False, False, "nor grows over the samples."),
    ],
)
def test_developed_report_derives_each_extent_claim(extents, larger, grows, ending):
    both = {"hopf": True, "turing": True}
    summary = benchmark._hopf_vs_turing(
        _developed_fft_records(both, both, extents), "developed_64"
    )
    assert summary["fft_origin_enclosure_corroborated"] is True
    assert "The corroborated FOV diagnostics enclose the origin" in summary["statement"]
    assert summary["hopf_fov_imaginary_extent_exceeds_turing_at_every_sample"] is larger
    assert summary["hopf_fov_imaginary_extent_grows_over_samples"] is grows
    assert summary["statement"].endswith(ending)


def test_committed_developed_report_states_only_supported_claims():
    """All 6 FFT records enclose with consistent supports; Turing t=200 has the max extent."""
    summary = _resolved_reports()["developed_64"]["hopf_vs_turing"]
    assert summary["fft_origin_enclosure_corroborated"] is True
    assert summary["hopf_fov_imaginary_extent_grows_over_samples"] is True
    assert summary["hopf_fov_imaginary_extent_exceeds_turing_at_every_sample"] is False
    assert "not larger than the Turing extent" in summary["statement"]


def _fixed_dt_records(verdicts):
    """Two-regime fixed_dt records from ``{regime: (early, developed)}`` verdicts."""
    return [
        {
            "regime": regime,
            "preconditioner": kind,
            "trajectory_step": step,
            "time": float(step),
            "developedness": {},
            "verdict": verdict,
            "disk_rate": 0.5,
            "epsilon_zero": 0.5,
            "origin_enclosed": False,
            "fov_imaginary_extent": 0.1,
            "n_right_real_outliers": 0,
            "adjoint_error": 0.0,
            "actual_gmres": None,
        }
        for regime, (early, developed) in verdicts.items()
        for kind in ("identity", "fft_diffusion")
        for step, verdict in ((1, early), (2, developed))
    ]


@pytest.mark.parametrize(
    ("verdicts", "outcome", "phrase"),
    [
        (
            {"hopf": ("indeterminate",) * 2, "turing": ("indeterminate",) * 2},
            "no_fft_adequate_to_indeterminate_transition_at_fixed_dt",
            "no FFT-preconditioned regime changes",
        ),
        (
            {"hopf": ("adequate", "adequate"), "turing": ("adequate", "indeterminate")},
            "fft_adequate_to_indeterminate_in_one_regime_at_fixed_dt",
            "at least one FFT-preconditioned regime changes",
        ),
        (
            {"hopf": ("adequate", "indeterminate"), "turing": ("adequate", "indeterminate")},
            "fft_adequate_to_indeterminate_in_both_regimes_at_fixed_dt",
            "in both regimes",
        ),
    ],
)
def test_fixed_dt_summary_counts_its_transitions(verdicts, outcome, phrase):
    """Review reproduction: zero transitions were reported as one."""
    summary = benchmark._fixed_transition(
        _fixed_dt_records(verdicts), benchmark.PRESETS["fixed_dt_256"]
    )
    assert summary["outcome"] == outcome
    assert phrase in summary["statement"]


@pytest.mark.parametrize(
    ("early", "developed", "outcome"),
    [
        ("adequate", "adequate", "fft_verdict_stable_at_fixed_dt"),
        ("investigate", "investigate", "fft_verdict_stable_at_fixed_dt"),
        ("adequate", "investigate", "fft_verdict_changed_at_fixed_dt"),
        ("adequate", "indeterminate", "fft_adequate_to_indeterminate_at_fixed_dt"),
        ("investigate", "adequate", "fft_verdict_changed_at_fixed_dt"),
    ],
)
def test_single_regime_fixed_dt_summary_compares_both_verdicts(early, developed, outcome):
    """Review reproduction: adequate to investigate was reported as unchanged."""
    config = benchmark.PRESETS["hopf_continuation_256"]
    assert config.regimes == ("hopf",)
    summary = benchmark._fixed_transition(
        _fixed_dt_records({"hopf": (early, developed)}), config
    )
    rows = summary["by_regime"]["hopf"]["fft_diffusion"]
    assert (rows["early"]["verdict"], rows["developed"]["verdict"]) == (early, developed)
    assert summary["outcome"] == outcome
    if early == developed:
        assert "is unchanged between the early and developed states" in summary["statement"]
    else:
        assert "unchanged" not in summary["statement"]
        assert (
            f"changes from {early} at the early state to {developed} at the developed state"
            in summary["statement"]
        )


def test_committed_hopf_continuation_transition_is_unchanged():
    """The committed adequate-to-adequate Hopf continuation keeps its stable statement."""
    report = _resolved_reports()["hopf_continuation_256"]
    stored = report["fixed_dt_transition"]
    rebuilt = benchmark._fixed_transition(
        report["records"], benchmark.PRESETS["hopf_continuation_256"]
    )
    assert rebuilt == stored
    rows = stored["by_regime"]["hopf"]["fft_diffusion"]
    assert (rows["early"]["verdict"], rows["developed"]["verdict"]) == ("adequate", "adequate")
    assert stored["outcome"] == "fft_verdict_stable_at_fixed_dt"
    assert stored["statement"] == (
        "At fixed backward-Euler dt, the FFT-preconditioned verdict is unchanged "
        "between the early and developed states."
    )
