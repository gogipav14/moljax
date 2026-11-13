"""Checkpointed fixed-state Brusselator backward-Euler ``dt`` sweep.

The benchmark measures the shipped identity and FFT-diffusion
preconditioners, then two diagnostic-only reaction-aware probes.  Its default
sweep never regenerates a trajectory: every source state is loaded from the
v4 Stage-3 cache and checked against both its manifest and record-level SHA256
identity.  The explicit ``--regenerate-source-states`` repair mode instead
writes separately labelled host artefacts and never replaces a Stage-3 cache.

See ``benchmarks/BRUSSELATOR_DT_SWEEP_PROTOCOL.md`` for the pre-registered
states, execution order, measurements, budget, and decision rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from moljax.conditioning import (
    arnoldi,
    assess_preconditioner,
    epsilon_zero,
    linearized_operator,
    numerical_range,
    ritz_values,
)
from moljax.core.grid import Grid2D
from moljax.core.newton_krylov import NKParams, create_implicit_residual
from moljax.core.preconditioners import PrecondContext, create_fft_preconditioner
from moljax.experimental.brusselator_conditioning import (
    BrusselatorLinearization,
    BrusselatorRegime,
    build_brusselator_linearization,
    build_brusselator_system,
    sampled_visited_states,
)
from moljax.experimental.pme_conditioning import _counted_gmres

jax.config.update("jax_enable_x64", True)

SCHEMA = "brusselator_fixed_state_dt_sweep_v1"
SOURCE_SCHEMA = "brusselator_conditioning_source_state_v4"
REGENERATED_SOURCE_SCHEMA = "brusselator_dt_sweep_regenerated_source_states_v1"
DT_GRID = (0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0)
PROBE_DT_GRID = (0.2, 1.0, 2.0)
V0_DT_GRID = (0.2, 1.0)
MAIN_PRECONDITIONERS = ("fft_diffusion", "identity")
PROBE_PRECONDITIONERS = ("mean_field_coupled_fft", "split_reaction_after_diffusion")
FOV_LADDER = ((32, 120, 2), (64, 180, 2), (96, 240, 2))
FOV_RESIDUAL_TOLERANCE = 1.0e-3
GMRES_TOLERANCE = 1.0e-8
GMRES_MAX_ITERS = 100
V0_TOLERANCE = 1.0e-12
SPLIT_DETERMINANT_FLOOR = 1.0e-12
MAX_GPU_HOURS = 5.0


@dataclass(frozen=True)
class StateSpec:
    """One tracked Stage-3 source state in execution order."""

    family: str
    result_name: str
    regime: str
    selector: str
    value: int

    @property
    def identifier(self) -> str:
        return f"{self.family}:{self.regime}:{self.value}"


STATE_SPECS = (
    StateSpec(
        "developed64", "brusselator_conditioning_developed.json", "hopf", "trajectory_step", 1
    ),
    StateSpec(
        "developed64", "brusselator_conditioning_developed.json", "hopf", "trajectory_step", 10
    ),
    StateSpec(
        "developed64", "brusselator_conditioning_developed.json", "hopf", "trajectory_step", 20
    ),
    StateSpec(
        "developed64", "brusselator_conditioning_developed.json", "turing", "trajectory_step", 80
    ),
    StateSpec(
        "developed64", "brusselator_conditioning_developed.json", "turing", "trajectory_step", 120
    ),
    StateSpec(
        "developed64", "brusselator_conditioning_developed.json", "turing", "trajectory_step", 200
    ),
    StateSpec(
        "fixed256", "brusselator_conditioning_fixed_dt.json", "turing", "trajectory_step", 1000
    ),
    StateSpec("fixed256", "brusselator_conditioning_fixed_dt.json", "turing", "trajectory_step", 1),
    StateSpec("screen64", "brusselator_conditioning.json", "hopf", "state_index", 0),
    StateSpec("screen64", "brusselator_conditioning.json", "hopf", "state_index", 1),
    StateSpec("screen64", "brusselator_conditioning.json", "turing", "state_index", 0),
    StateSpec("screen64", "brusselator_conditioning.json", "turing", "state_index", 1),
)
CPU_STATE_SPECS = tuple(spec for spec in STATE_SPECS if spec.family != "fixed256")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_identity(state: dict[str, jax.Array]) -> dict[str, Any]:
    digest = hashlib.sha256()
    fields: dict[str, dict[str, Any]] = {}
    for field in ("u", "v"):
        values = np.asarray(jax.device_get(state[field]), dtype=np.float64)
        digest.update(field.encode("utf-8"))
        digest.update(values.dtype.str.encode("utf-8"))
        digest.update(values.tobytes(order="C"))
        fields[field] = {"shape": list(values.shape), "dtype": values.dtype.str}
    return {"sha256": digest.hexdigest(), "fields": fields}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_write_states(path: Path, states: list[dict[str, jax.Array]]) -> None:
    payload = {
        f"{field}_{index}": np.asarray(jax.device_get(state[field]), dtype=np.float64)
        for index, state in enumerate(states)
        for field in ("u", "v")
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", suffix=".npz", dir=path.parent, delete=False) as handle:
        np.savez_compressed(handle, **payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), text=True, stderr=subprocess.DEVNULL
    ).strip()


def _stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")


def _plan() -> dict[str, Any]:
    return {
        "states": [spec.identifier for spec in STATE_SPECS],
        "main_dt": list(DT_GRID),
        "probe_dt": list(PROBE_DT_GRID),
        "main_preconditioners": list(MAIN_PRECONDITIONERS),
        "probe_preconditioners": list(PROBE_PRECONDITIONERS),
        "fov_ladder": [list(budget) for budget in FOV_LADDER],
        "fov_residual_tolerance": FOV_RESIDUAL_TOLERANCE,
        "gmres_tolerance": GMRES_TOLERANCE,
        "gmres_max_iters": GMRES_MAX_ITERS,
        "v0_tolerance": V0_TOLERANCE,
        "split_determinant_floor": SPLIT_DETERMINANT_FLOOR,
        "max_gpu_hours": MAX_GPU_HOURS,
    }


def _plan_fingerprint() -> str:
    encoded = json.dumps(_plan(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _provenance(script_path: Path, protocol_path: Path) -> dict[str, Any]:
    try:
        git_head = _git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    return {
        "python": platform.python_version(),
        "jax": jax.__version__,
        "jaxlib": jax.lib.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "x64_enabled": bool(jax.config.x64_enabled),
        "xla_python_client_preallocate": os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        "git_commit": git_head,
        "script_sha256": _sha256(script_path),
        "protocol_sha256": _sha256(protocol_path),
    }


def _find_record(repository: Path, spec: StateSpec) -> dict[str, Any]:
    path = repository / "benchmarks" / "results" / spec.result_name
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        record
        for record in payload["records"]
        if record["regime"] == spec.regime
        and record.get(spec.selector) == spec.value
        and record["preconditioner"] == "fft_diffusion"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one tracked record for {spec.identifier}")
    record = matches[0]
    if not record["source_state_artifact"].get("converged", False):
        raise RuntimeError(f"source state did not converge: {spec.identifier}")
    return record


def _fingerprint_key(fingerprint: dict[str, Any]) -> str:
    encoded = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _generate_fingerprint_states(
    fingerprint: dict[str, Any],
) -> list[dict[str, jax.Array]]:
    """Regenerate one trajectory from its recorded v4 generation contract."""
    if fingerprint.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError("source generation fingerprint schema mismatch")
    if fingerprint.get("model_version") != "brusselator_periodic_fft_be_v1":
        raise RuntimeError("unsupported source generation model version")
    grid_config = fingerprint["grid"]
    domain = fingerprint["domain"]
    if domain["x"] != domain["y"]:
        raise RuntimeError("the recorded Brusselator source domain must be square")
    grid = Grid2D.uniform(
        int(grid_config["nx"]),
        int(grid_config["ny"]),
        float(domain["x"][0]),
        float(domain["x"][1]),
        float(domain["y"][0]),
        float(domain["y"][1]),
        n_ghost=int(grid_config["n_ghost"]),
    )
    solver = fingerprint["solver"]
    if solver.get("max_backtrack") != "NKParams-default-8":
        raise RuntimeError("unsupported recorded Newton backtracking contract")
    samples = sampled_visited_states(
        BrusselatorRegime(**fingerprint["regime"]),
        grid=grid,
        sample_steps=tuple(int(step) for step in fingerprint["sample_steps"]),
        dt=float(fingerprint["state_dt"]),
        perturbation=float(fingerprint["perturbation"]),
        seed=int(fingerprint["seed"]),
        nk_params=NKParams(
            max_newton_iters=int(solver["max_newton_iters"]),
            max_krylov_iters=int(solver["max_krylov_iters"]),
            newton_tol=float(solver["newton_tol"]),
            krylov_tol=float(solver["krylov_tol"]),
        ),
    )
    return [sample.state for sample in samples]


def _regenerate_source_states(repository: Path, output_dir: Path) -> Path:
    """Write new host-labelled 64-square states without replacing Stage-3 identities."""
    if output_dir.exists():
        raise FileExistsError(f"regeneration output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    host = platform.node() or "unknown-host"
    host_label = "".join(character if character.isalnum() else "-" for character in host)
    grouped: dict[str, list[tuple[StateSpec, dict[str, Any]]]] = {}
    fingerprints: dict[str, dict[str, Any]] = {}
    for spec in CPU_STATE_SPECS:
        record = _find_record(repository, spec)
        fingerprint = record["record_config"]["source_state_fingerprint"]
        key = _fingerprint_key(fingerprint)
        grouped.setdefault(key, []).append((spec, record))
        fingerprints[key] = fingerprint

    manifest_records: dict[str, Any] = {}
    trajectories: dict[str, Any] = {}
    for key, members in sorted(grouped.items()):
        fingerprint = fingerprints[key]
        states = _generate_fingerprint_states(fingerprint)
        identities = [_state_identity(state) for state in states]
        regime_name = str(fingerprint["regime"]["name"])
        relative = Path(f"{regime_name}-{key}-{host_label}.npz")
        array_path = output_dir / relative
        _atomic_write_states(array_path, states)
        trajectories[key] = {
            "relative_path": str(relative),
            "npz_sha256": _sha256(array_path),
            "generation_fingerprint": fingerprint,
            "regenerated_state_identities": identities,
        }
        for spec, record in members:
            artifact = record["source_state_artifact"]
            position = int(artifact["sample_position"])
            if position >= len(states):
                raise RuntimeError(f"regenerated trajectory is incomplete: {spec.identifier}")
            manifest_records[spec.identifier] = {
                "state": spec.identifier,
                "source_report": spec.result_name,
                "original_stage3": {
                    "relative_path": artifact["relative_path"],
                    "source_state_identity": artifact["source_state_identity"],
                },
                "regenerated": {
                    "relative_path": str(relative),
                    "sample_position": position,
                    "source_state_identity": identities[position],
                },
                "generation_fingerprint": fingerprint,
            }

    manifest_path = output_dir / "regenerated_source_states_manifest.json"
    _atomic_write_json(
        manifest_path,
        {
            "schema": REGENERATED_SOURCE_SCHEMA,
            "identity_scope": "host-labelled regenerated realisations; not Stage-3 identities",
            "host_label": host,
            "provenance": {
                "python": platform.python_version(),
                "jax": jax.__version__,
                "jaxlib": jax.lib.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "backend": jax.default_backend(),
                "x64_enabled": bool(jax.config.x64_enabled),
                "git_commit": _git("rev-parse", "HEAD"),
            },
            "trajectories": trajectories,
            "records": manifest_records,
        },
    )
    return manifest_path


def _load_regenerated_state(
    cache_root: Path, spec: StateSpec, record: dict[str, Any]
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Load a host-labelled state while retaining its distinct identity."""
    manifest_path = cache_root / "regenerated_source_states_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != REGENERATED_SOURCE_SCHEMA:
        raise RuntimeError("regenerated source-state manifest schema mismatch")
    entry = manifest.get("records", {}).get(spec.identifier)
    if entry is None:
        raise RuntimeError(f"regenerated source state is missing: {spec.identifier}")
    fingerprint = record["record_config"]["source_state_fingerprint"]
    if entry.get("generation_fingerprint") != fingerprint:
        raise RuntimeError(f"regenerated source fingerprint mismatch: {spec.identifier}")
    regenerated = entry["regenerated"]
    relative = Path(regenerated["relative_path"])
    if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
        raise RuntimeError("regenerated artifact path must be a single cache-relative file name")
    array_path = cache_root / relative
    key = _fingerprint_key(fingerprint)
    trajectory = manifest["trajectories"][key]
    if _sha256(array_path) != trajectory["npz_sha256"]:
        raise RuntimeError(f"regenerated trajectory SHA256 mismatch: {array_path}")
    position = int(regenerated["sample_position"])
    with np.load(array_path, allow_pickle=False) as arrays:
        state = {
            field: jax.block_until_ready(
                jnp.asarray(arrays[f"{field}_{position}"], dtype=jnp.float64)
            )
            for field in ("u", "v")
        }
    observed = _state_identity(state)
    if observed != regenerated["source_state_identity"]:
        raise RuntimeError(f"regenerated source-state identity mismatch: {array_path}")
    return state, {
        "identity_scope": manifest["identity_scope"],
        "host_label": manifest["host_label"],
        "relative_path": str(relative),
        "npz_sha256": trajectory["npz_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "state_identity": observed,
        "original_stage3_identity": entry["original_stage3"]["source_state_identity"],
        "sample_position": position,
        "generation_fingerprint": fingerprint,
    }


def _load_state(
    cache_root: Path, record: dict[str, Any]
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    artifact = record["source_state_artifact"]
    relative = Path(artifact["relative_path"])
    if relative.is_absolute() or len(relative.parts) != 1 or ".." in relative.parts:
        raise RuntimeError("source artifact path must be a single cache-relative file name")
    if artifact.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError("source artifact schema mismatch")
    array_path = cache_root / relative
    manifest_path = array_path.with_suffix(".json")
    if not array_path.is_file() or not manifest_path.is_file():
        raise RuntimeError(f"source artifact is missing: {array_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = artifact["generation_fingerprint"]
    if manifest.get("schema") != SOURCE_SCHEMA:
        raise RuntimeError(f"source manifest schema mismatch: {manifest_path}")
    if manifest.get("generation_fingerprint") != fingerprint:
        raise RuntimeError(f"source generation fingerprint mismatch: {manifest_path}")
    if record["record_config"]["source_state_fingerprint"] != fingerprint:
        raise RuntimeError("record and artifact generation fingerprints differ")
    position = int(artifact["sample_position"])
    identities = manifest.get("source_state_identities")
    if not isinstance(identities, list) or position >= len(identities):
        raise RuntimeError(f"source manifest identities are incomplete: {manifest_path}")
    with np.load(array_path, allow_pickle=False) as arrays:
        state = {
            field: jax.block_until_ready(
                jnp.asarray(arrays[f"{field}_{position}"], dtype=jnp.float64)
            )
            for field in ("u", "v")
        }
    observed = _state_identity(state)
    if observed != identities[position] or observed != artifact["source_state_identity"]:
        raise RuntimeError(f"source-state SHA256 mismatch: {array_path}")
    return state, {
        "relative_path": str(relative),
        "npz_sha256": _sha256(array_path),
        "manifest_sha256": _sha256(manifest_path),
        "state_identity": observed,
        "sample_position": position,
        "generation_fingerprint": fingerprint,
    }


def _grid_and_regime(record: dict[str, Any]) -> tuple[Grid2D, BrusselatorRegime]:
    config = record["record_config"]
    grid_config = config["grid"]
    length = float(config["domain_length"])
    grid = Grid2D.uniform(
        int(grid_config["nx"]),
        int(grid_config["ny"]),
        0.0,
        length,
        0.0,
        length,
        n_ghost=int(grid_config["n_ghost"]),
    )
    return grid, BrusselatorRegime(**config["regime"])


def _reaction_jacobian(state: dict[str, jax.Array], grid: Grid2D, beta: float) -> jax.Array:
    y_slice, x_slice = grid.interior_slice
    u = state["u"][y_slice, x_slice]
    v = state["v"][y_slice, x_slice]
    return jnp.stack(
        (
            jnp.stack((-(beta + 1.0) + 2.0 * u * v, u * u), axis=-1),
            jnp.stack((beta - 2.0 * u * v, -(u * u)), axis=-1),
        ),
        axis=-2,
    )


def _reaction_metadata(jacobian: jax.Array, dt: float) -> dict[str, float]:
    largest = jnp.linalg.svd(jacobian, compute_uv=False)[..., 0]
    maximum = float(jax.block_until_ready(jnp.max(largest)))
    mean = np.asarray(jax.device_get(jnp.mean(jacobian, axis=(0, 1))), dtype=np.float64)
    return {
        "max_reaction_jacobian_2_norm": maximum,
        "reaction_strength": dt * maximum,
        "mean_j00": float(mean[0, 0]),
        "mean_j01": float(mean[0, 1]),
        "mean_j10": float(mean[1, 0]),
        "mean_j11": float(mean[1, 1]),
    }


def _interior(field: jax.Array, grid: Grid2D) -> jax.Array:
    y_slice, x_slice = grid.interior_slice
    return field[y_slice, x_slice]


def _embed(field: jax.Array, interior: jax.Array, grid: Grid2D) -> jax.Array:
    y_slice, x_slice = grid.interior_slice
    return field.at[y_slice, x_slice].set(interior.astype(field.dtype))


@dataclass(frozen=True)
class MeanFieldCoupledFFTPreconditioner:
    """Diagnostic Fourier inverse using the spatially averaged reaction block."""

    fft_cache: Any
    reaction_mean: jax.Array
    name: str = "mean_field_coupled_fft"

    def apply(
        self, residual: dict[str, jax.Array], context: PrecondContext
    ) -> dict[str, jax.Array]:
        grid = context.grid
        if not isinstance(grid, Grid2D):
            raise TypeError("the mean-field probe requires Grid2D")
        if getattr(self.fft_cache, "use_rfft", False):
            raise ValueError("the mean-field probe expects the shipped full-spectrum FFT cache")
        symbol = self.fft_cache.laplacian_symbol
        u_hat = jnp.fft.fft2(_interior(residual["u"], grid))
        v_hat = jnp.fft.fft2(_interior(residual["v"], grid))
        dt = context.dt
        jacobian = self.reaction_mean
        a00 = 1.0 - dt * context.params["Du"] * symbol - dt * jacobian[0, 0]
        a01 = -dt * jacobian[0, 1]
        a10 = -dt * jacobian[1, 0]
        a11 = 1.0 - dt * context.params["Dv"] * symbol - dt * jacobian[1, 1]
        determinant = a00 * a11 - a01 * a10
        u_solved = jnp.fft.ifft2((a11 * u_hat - a01 * v_hat) / determinant)
        v_solved = jnp.fft.ifft2((-a10 * u_hat + a00 * v_hat) / determinant)
        if not jnp.issubdtype(residual["u"].dtype, jnp.complexfloating):
            u_solved = jnp.real(u_solved)
            v_solved = jnp.real(v_solved)
        return {
            "u": _embed(residual["u"], u_solved, grid),
            "v": _embed(residual["v"], v_solved, grid),
        }


@dataclass(frozen=True)
class SplitReactionAfterDiffusionPreconditioner:
    """Diagnostic pointwise reaction inverse after the shipped FFT inverse."""

    diffusion: Any
    reaction_jacobian: jax.Array
    name: str = "split_reaction_after_diffusion"

    def apply(
        self, residual: dict[str, jax.Array], context: PrecondContext
    ) -> dict[str, jax.Array]:
        grid = context.grid
        if not isinstance(grid, Grid2D):
            raise TypeError("the split probe requires Grid2D")
        diffused = self.diffusion.apply(residual, context)
        jacobian = self.reaction_jacobian
        dt = context.dt
        a00 = 1.0 - dt * jacobian[..., 0, 0]
        a01 = -dt * jacobian[..., 0, 1]
        a10 = -dt * jacobian[..., 1, 0]
        a11 = 1.0 - dt * jacobian[..., 1, 1]
        determinant = a00 * a11 - a01 * a10
        u_value = _interior(diffused["u"], grid)
        v_value = _interior(diffused["v"], grid)
        u_solved = (a11 * u_value - a01 * v_value) / determinant
        v_solved = (-a10 * u_value + a00 * v_value) / determinant
        return {
            "u": _embed(diffused["u"], u_solved, grid),
            "v": _embed(diffused["v"], v_solved, grid),
        }


def _split_minimum_determinant(jacobian: jax.Array, dt: float) -> float:
    a00 = 1.0 - dt * jacobian[..., 0, 0]
    a01 = -dt * jacobian[..., 0, 1]
    a10 = -dt * jacobian[..., 1, 0]
    a11 = 1.0 - dt * jacobian[..., 1, 1]
    determinant = a00 * a11 - a01 * a10
    return float(jax.block_until_ready(jnp.min(jnp.abs(determinant))))


def _custom_linearization(
    state: dict[str, jax.Array],
    model: Any,
    preconditioner: Any,
    dt: float,
    time_value: float,
) -> BrusselatorLinearization:
    residual = create_implicit_residual(model, state, time_value + dt, dt, method="be")
    context = PrecondContext(grid=model.grid, dt=dt, params=model.params)
    operator = linearized_operator(
        residual,
        state,
        preconditioner=preconditioner,
        context=context,
    )
    negative_residual = jax.tree_util.tree_map(lambda value: -value, residual(state))
    rhs_state = preconditioner.apply(negative_residual, context)
    rhs, _ = ravel_pytree(rhs_state)
    return BrusselatorLinearization(operator, residual, preconditioner, context, rhs)


def _build_cell_linearization(
    state: dict[str, jax.Array],
    model: Any,
    fft_cache: Any,
    diffusivities: dict[str, float],
    reaction_jacobian: jax.Array,
    dt: float,
    time_value: float,
    preconditioner_name: str,
) -> tuple[BrusselatorLinearization | None, dict[str, Any]]:
    if preconditioner_name in MAIN_PRECONDITIONERS:
        return (
            build_brusselator_linearization(
                state,
                model,
                fft_cache,
                diffusivities,
                dt,
                time_value=time_value,
                preconditioner_kind=preconditioner_name,
            ),
            {},
        )
    diffusion = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    if preconditioner_name == "mean_field_coupled_fft":
        reaction_mean = jnp.mean(reaction_jacobian, axis=(0, 1))
        preconditioner = MeanFieldCoupledFFTPreconditioner(fft_cache, reaction_mean)
        return (
            _custom_linearization(state, model, preconditioner, dt, time_value),
            {"reaction_mean": np.asarray(jax.device_get(reaction_mean)).tolist()},
        )
    if preconditioner_name == "split_reaction_after_diffusion":
        minimum = _split_minimum_determinant(reaction_jacobian, dt)
        metadata = {
            "minimum_absolute_reaction_determinant": minimum,
            "composition_order": "reaction_inverse_after_fft_diffusion_inverse",
        }
        if minimum < SPLIT_DETERMINANT_FLOOR:
            return None, metadata
        preconditioner = SplitReactionAfterDiffusionPreconditioner(diffusion, reaction_jacobian)
        return _custom_linearization(state, model, preconditioner, dt, time_value), metadata
    raise ValueError(f"unknown preconditioner {preconditioner_name!r}")


def _diagnose(
    linearization: BrusselatorLinearization, seed: int
) -> tuple[dict[str, Any], dict[str, float]]:
    timings: dict[str, float] = {}
    started = time.perf_counter()
    gmres = _measure_gmres(linearization)
    timings["gmres_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed))
    start = jax.random.normal(key_real, (linearization.operator.n,), dtype=jnp.float64)
    start = start + 1j * jax.random.normal(key_imag, (linearization.operator.n,), dtype=jnp.float64)
    coverage = arnoldi(
        linearization.operator.matvec,
        start,
        min(6, linearization.operator.n),
    )
    ritz = ritz_values(coverage.hessenberg)
    reduced_epsilon = epsilon_zero(coverage.hessenberg)
    timings["arnoldi_seconds"] = time.perf_counter() - started

    attempts: list[dict[str, Any]] = []
    field_of_values = None
    fov_started = time.perf_counter()
    for n_angles, max_iters, n_restarts in FOV_LADDER:
        attempt_started = time.perf_counter()
        field_of_values = numerical_range(
            linearization.operator.matvec,
            linearization.operator.matvec_adjoint,
            linearization.operator.n,
            n_angles=n_angles,
            max_iters=max_iters,
            residual_tolerance=FOV_RESIDUAL_TOLERANCE,
            n_restarts=n_restarts,
            operator_key=jax.random.PRNGKey(seed + 1),
        )
        jax.block_until_ready(field_of_values.boundary)
        attempts.append(
            {
                "n_angles": n_angles,
                "fov_max_iters": max_iters,
                "n_restarts": n_restarts,
                "wall_seconds": time.perf_counter() - attempt_started,
                "disk_rate": float(field_of_values.disk_rate),
                "origin_enclosed": bool(field_of_values.origin_enclosed),
                "max_support_residual": float(field_of_values.max_support_residual),
                "supports_converged": bool(field_of_values.supports_converged),
                "supports_corroborated": bool(field_of_values.supports_corroborated),
                "supports_consistent": bool(field_of_values.supports_consistent),
            }
        )
        if field_of_values.supports_consistent:
            break
    if field_of_values is None:
        raise AssertionError("the FOV ladder is empty")
    timings["fov_seconds"] = time.perf_counter() - fov_started
    assessment = assess_preconditioner(
        field_of_values,
        ritz,
        reduced_epsilon,
        coverage=coverage,
    )
    result = {
        "gmres": gmres,
        "reduced_arnoldi_epsilon_zero": float(reduced_epsilon),
        "n_right_real_outliers": assessment.n_right_real_outliers,
        "reduced_assessment_verdict": assessment.verdict,
        "reduced_assessment_reason": assessment.verdict_reason,
        "disk_rate": float(field_of_values.disk_rate),
        "disk_center_real": float(complex(field_of_values.center).real),
        "disk_center_imag": float(complex(field_of_values.center).imag),
        "disk_radius": float(field_of_values.radius),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "max_support_residual": float(field_of_values.max_support_residual),
        "supports_converged": bool(field_of_values.supports_converged),
        "supports_corroborated": bool(field_of_values.supports_corroborated),
        "supports_consistent": bool(field_of_values.supports_consistent),
        "corroboration_attempted": bool(field_of_values.corroboration_attempted),
        "fov_attempts": attempts,
    }
    return result, timings


def _measure_gmres(linearization: BrusselatorLinearization) -> dict[str, Any]:
    """Run the sweep's counted-GMRES protocol without other diagnostics."""
    return _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=GMRES_TOLERANCE,
        max_iters=GMRES_MAX_ITERS,
    )


def _measure_disk(linearization: BrusselatorLinearization, seed: int) -> dict[str, Any]:
    """Run only the pre-registered FoV ladder and return its disk evidence."""
    attempts: list[dict[str, Any]] = []
    field_of_values = None
    started = time.perf_counter()
    for n_angles, max_iters, n_restarts in FOV_LADDER:
        attempt_started = time.perf_counter()
        field_of_values = numerical_range(
            linearization.operator.matvec,
            linearization.operator.matvec_adjoint,
            linearization.operator.n,
            n_angles=n_angles,
            max_iters=max_iters,
            residual_tolerance=FOV_RESIDUAL_TOLERANCE,
            n_restarts=n_restarts,
            operator_key=jax.random.PRNGKey(seed + 1),
        )
        jax.block_until_ready(field_of_values.boundary)
        attempts.append(
            {
                "n_angles": n_angles,
                "fov_max_iters": max_iters,
                "n_restarts": n_restarts,
                "wall_seconds": time.perf_counter() - attempt_started,
                "disk_rate": float(field_of_values.disk_rate),
                "origin_enclosed": bool(field_of_values.origin_enclosed),
                "max_support_residual": float(field_of_values.max_support_residual),
                "supports_converged": bool(field_of_values.supports_converged),
                "supports_corroborated": bool(field_of_values.supports_corroborated),
                "supports_consistent": bool(field_of_values.supports_consistent),
            }
        )
        if field_of_values.supports_consistent:
            break
    if field_of_values is None:
        raise AssertionError("the FOV ladder is empty")
    return {
        "disk_rate": float(field_of_values.disk_rate),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "max_support_residual": float(field_of_values.max_support_residual),
        "supports_converged": bool(field_of_values.supports_converged),
        "supports_corroborated": bool(field_of_values.supports_corroborated),
        "supports_consistent": bool(field_of_values.supports_consistent),
        "fov_attempts": attempts,
        "fov_seconds": time.perf_counter() - started,
    }


def _reaction_jvp(
    model: Any,
    state: dict[str, jax.Array],
    tangent: dict[str, jax.Array],
    time_value: float,
) -> dict[str, jax.Array]:
    def reaction(candidate: dict[str, jax.Array]) -> dict[str, jax.Array]:
        bounded = model.apply_bcs(candidate, time_value)
        return model.nonlinear_rhs(bounded, time_value)

    return jax.jvp(reaction, (state,), (tangent,))[1]


def _run_v0(
    repository: Path,
    cache_root: Path,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for spec in STATE_SPECS:
        record = _find_record(repository, spec)
        state, source = _load_state(cache_root, record)
        grid, regime = _grid_and_regime(record)
        model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
        for dt in V0_DT_GRID:
            linearization = build_brusselator_linearization(
                state,
                model,
                fft_cache,
                diffusivities,
                dt,
                time_value=float(record["time"]),
                preconditioner_kind="fft_diffusion",
            )
            _, unravel = ravel_pytree(state)
            for sample in range(3):
                key = jax.random.PRNGKey(_stable_seed(f"v0:{spec.identifier}:{dt}:{sample}"))
                vector = jax.random.normal(key, (linearization.operator.n,), dtype=jnp.float64)
                tangent = unravel(vector)
                reaction_image = _reaction_jvp(
                    model,
                    state,
                    tangent,
                    float(record["time"]) + dt,
                )
                preconditioned = linearization.preconditioner.apply(
                    reaction_image, linearization.context
                )
                flat_reaction, _ = ravel_pytree(preconditioned)
                defect = linearization.operator.matvec(vector) - vector + dt * flat_reaction
                relative = float(
                    jax.block_until_ready(jnp.linalg.norm(defect) / jnp.linalg.norm(vector))
                )
                row = {
                    "state": spec.identifier,
                    "state_sha256": source["state_identity"]["sha256"],
                    "dt": dt,
                    "sample": sample,
                    "relative_defect": relative,
                    "passed": relative <= V0_TOLERANCE,
                }
                checks.append(row)
                if not row["passed"]:
                    raise RuntimeError(
                        "V0 premise check failed: "
                        f"{spec.identifier}, dt={dt}, sample={sample}, defect={relative:.3e}"
                    )
    return checks


def _cell_plan() -> list[tuple[str, StateSpec, float, str]]:
    cells: list[tuple[str, StateSpec, float, str]] = []
    for spec in STATE_SPECS:
        for dt in DT_GRID:
            for preconditioner in MAIN_PRECONDITIONERS:
                cells.append(("main", spec, dt, preconditioner))
    for spec in STATE_SPECS:
        for dt in PROBE_DT_GRID:
            for preconditioner in PROBE_PRECONDITIONERS:
                cells.append(("probe", spec, dt, preconditioner))
    return cells


def _cell_key(arm: str, spec: StateSpec, dt: float, preconditioner: str) -> str:
    return f"{arm}|{spec.identifier}|dt={dt:g}|{preconditioner}"


def _initial_payload(script: Path, protocol: Path) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "initialized",
        "plan": _plan(),
        "plan_fingerprint": _plan_fingerprint(),
        "provenance": _provenance(script, protocol),
        "started_at_unix": time.time(),
        "gpu_seconds_accounted": 0.0,
        "v0": {"status": "pending", "checks": []},
        "cells": {},
        "truncation": None,
    }


def _load_or_initialize(output: Path, script: Path, protocol: Path) -> dict[str, Any]:
    if not output.is_file():
        return _initial_payload(script, protocol)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise RuntimeError("checkpoint schema mismatch")
    if payload.get("plan_fingerprint") != _plan_fingerprint():
        raise RuntimeError("checkpoint plan mismatch")
    return payload


def _run_cell(
    repository: Path,
    cache_root: Path,
    arm: str,
    spec: StateSpec,
    dt: float,
    preconditioner_name: str,
) -> dict[str, Any]:
    cell_started = time.perf_counter()
    record = _find_record(repository, spec)
    state, source = _load_state(cache_root, record)
    grid, regime = _grid_and_regime(record)
    model, fft_cache, diffusivities = build_brusselator_system(regime, grid)
    reaction_jacobian = _reaction_jacobian(state, grid, regime.b)
    reaction = _reaction_metadata(reaction_jacobian, dt)
    build_started = time.perf_counter()
    linearization, preconditioner_metadata = _build_cell_linearization(
        state,
        model,
        fft_cache,
        diffusivities,
        reaction_jacobian,
        dt,
        float(record["time"]),
        preconditioner_name,
    )
    build_seconds = time.perf_counter() - build_started
    common = {
        "arm": arm,
        "state": spec.identifier,
        "family": spec.family,
        "regime": spec.regime,
        "selector": spec.selector,
        "selector_value": spec.value,
        "source_time": float(record["time"]),
        "source_report": spec.result_name,
        "source": source,
        "grid": {
            "nx": grid.nx,
            "ny": grid.ny,
            "n_ghost": grid.n_ghost,
            "domain_length": regime.domain_length,
        },
        "dt": dt,
        "preconditioner": preconditioner_name,
        "reaction": reaction,
        "preconditioner_metadata": preconditioner_metadata,
        "build_seconds": build_seconds,
    }
    if linearization is None:
        return {
            **common,
            "status": "skipped",
            "skip_reason": "pointwise reaction inverse determinant below floor",
            "wall_seconds": time.perf_counter() - cell_started,
        }
    diagnostics, timings = _diagnose(
        linearization,
        _stable_seed(f"cell:{arm}:{spec.identifier}:{dt}:{preconditioner_name}"),
    )
    return {
        **common,
        "status": "completed",
        "operator_dimension": linearization.operator.n,
        **diagnostics,
        **timings,
        "wall_seconds": time.perf_counter() - cell_started,
    }


def _unstarted_keys(payload: dict[str, Any]) -> list[str]:
    completed = payload["cells"]
    return [
        _cell_key(arm, spec, dt, preconditioner)
        for arm, spec, dt, preconditioner in _cell_plan()
        if _cell_key(arm, spec, dt, preconditioner) not in completed
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-state-cache-dir",
        default="/tmp/moljax-brusselator-stage2-d4047f3-v4-r1/source_states",
    )
    parser.add_argument("--output", default="/tmp/moljax-brusselator-dt-sweep.json")
    parser.add_argument("--list-plan", action="store_true")
    parser.add_argument(
        "--regenerate-source-states",
        metavar="DIR",
        help="regenerate the ten tracked 64-square states with new host-labelled identities",
    )
    arguments = parser.parse_args()

    if arguments.list_plan:
        print(json.dumps(_plan(), indent=2, sort_keys=True))
        return
    script = Path(__file__).resolve()
    repository = script.parents[1]
    if arguments.regenerate_source_states:
        if not jax.config.x64_enabled:
            raise RuntimeError("source-state regeneration requires JAX x64")
        manifest = _regenerate_source_states(
            repository, Path(arguments.regenerate_source_states).resolve()
        )
        print(manifest)
        return
    if os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() != "false":
        raise RuntimeError("set XLA_PYTHON_CLIENT_PREALLOCATE=false before the sweep")
    if jax.default_backend() != "gpu":
        raise RuntimeError("the pre-registered sweep must run on the GPU backend")
    if not jax.config.x64_enabled:
        raise RuntimeError("the pre-registered sweep requires JAX x64")

    protocol = repository / "benchmarks" / "BRUSSELATOR_DT_SWEEP_PROTOCOL.md"
    output = Path(arguments.output).resolve()
    cache_root = Path(arguments.source_state_cache_dir).resolve()
    payload = _load_or_initialize(output, script, protocol)
    payload["status"] = "running"
    _atomic_write_json(output, payload)

    stop_requested = False

    def request_stop(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    print(
        json.dumps(
            {
                "event": "START",
                "output": str(output),
                "provenance": payload["provenance"],
                "completed_cells": len(payload["cells"]),
                "planned_cells": len(_cell_plan()),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    if payload["v0"]["status"] != "passed":
        v0_started = time.perf_counter()
        try:
            checks = _run_v0(repository, cache_root)
        except Exception as error:
            payload["v0"] = {
                "status": "failed",
                "checks": [],
                "error": f"{type(error).__name__}: {error}",
            }
            payload["status"] = "v0_failed"
            payload["gpu_seconds_accounted"] += time.perf_counter() - v0_started
            _atomic_write_json(output, payload)
            print(json.dumps({"event": "V0_FAILED", "error": str(error)}), flush=True)
            raise
        payload["v0"] = {
            "status": "passed",
            "checks": checks,
            "maximum_relative_defect": max(row["relative_defect"] for row in checks),
            "wall_seconds": time.perf_counter() - v0_started,
        }
        payload["gpu_seconds_accounted"] += payload["v0"]["wall_seconds"]
        _atomic_write_json(output, payload)
        print(
            json.dumps(
                {
                    "event": "V0_COMPLETE",
                    "checks": len(checks),
                    "maximum_relative_defect": payload["v0"]["maximum_relative_defect"],
                    "wall_seconds": payload["v0"]["wall_seconds"],
                    "pre_registered_eta_hours": 4.5,
                    "hard_cap_hours": MAX_GPU_HOURS,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    budget_seconds = MAX_GPU_HOURS * 3600.0
    for arm, spec, dt, preconditioner in _cell_plan():
        key = _cell_key(arm, spec, dt, preconditioner)
        if key in payload["cells"]:
            continue
        if stop_requested or payload["gpu_seconds_accounted"] >= budget_seconds:
            payload["status"] = "budget_truncated"
            payload["truncation"] = {
                "reason": "signal" if stop_requested else "five_hour_gpu_budget",
                "gpu_seconds_accounted": payload["gpu_seconds_accounted"],
                "unstarted_cell_keys": _unstarted_keys(payload),
            }
            _atomic_write_json(output, payload)
            print(json.dumps({"event": "TRUNCATED", **payload["truncation"]}), flush=True)
            return
        print(json.dumps({"event": "CELL_START", "key": key}), flush=True)
        started = time.perf_counter()
        try:
            row = _run_cell(
                repository,
                cache_root,
                arm,
                spec,
                dt,
                preconditioner,
            )
        except Exception as error:
            row = {
                "arm": arm,
                "state": spec.identifier,
                "dt": dt,
                "preconditioner": preconditioner,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        elapsed = time.perf_counter() - started
        row["checkpoint_cell_seconds"] = elapsed
        payload["gpu_seconds_accounted"] += elapsed
        payload["cells"][key] = row
        _atomic_write_json(output, payload)
        print(
            json.dumps(
                {
                    "event": "CELL_COMPLETE",
                    "key": key,
                    "status": row["status"],
                    "seconds": elapsed,
                    "completed": len(payload["cells"]),
                    "planned": len(_cell_plan()),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    payload["status"] = "complete"
    payload["completed_at_unix"] = time.time()
    payload["truncation"] = None
    _atomic_write_json(output, payload)
    print(
        json.dumps(
            {
                "event": "BRUSSELATOR_DT_SWEEP_COMPLETE",
                "completed": len(payload["cells"]),
                "gpu_seconds_accounted": payload["gpu_seconds_accounted"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
