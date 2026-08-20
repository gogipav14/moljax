"""Smoke checks for experimental PME conditioning diagnostics."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from moljax.core.grid import Grid1D
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_conditioning import assess_pme_state, padded_values
from moljax.experimental.pme_preconditioner import d0_frozen_mean


def _barenblatt_state(grid: Grid1D) -> jax.Array:
    """Return a compactly supported padded ``m=2`` state for diagnostics."""
    return padded_values(barenblatt(grid.x_coords(), 1.0, 2.0, b=0.30), grid)


def _smooth_dirichlet_state(grid: Grid1D) -> jax.Array:
    """Return a smooth positive m=2 state with zero face data."""
    coordinate = (grid.x_coords() - grid.x_min) / (grid.x_max - grid.x_min)
    return padded_values(jnp.sin(jnp.pi * coordinate), grid)


@pytest.mark.slow
def test_assess_pme_state_has_a_valid_adjoint_gate_and_verdict() -> None:
    """The experimental adapter exposes a valid matrix-free diagnostic state."""
    with jax.enable_x64(True):
        grid = Grid1D.uniform(64, -4.0, 4.0)
        result = assess_pme_state(
            _barenblatt_state(grid),
            grid,
            2.0,
            0.01,
            1.0e-5,
            "frozen_mean",
            n_angles=4,
            fov_max_iters=16,
            arnoldi_steps=6,
        )

    assert result["adjoint_error"] <= 1.0e-8
    assert result["verdict"] in {"adequate", "investigate", "indeterminate"}


@pytest.mark.slow
def test_frozen_mean_preconditioning_tightens_the_m2_numerical_range() -> None:
    """The frozen-D0 variant improves the disk-rate diagnostic over identity."""
    with jax.enable_x64(True):
        grid = Grid1D.uniform(64, -4.0, 4.0)
        state = _smooth_dirichlet_state(grid)
        frozen = assess_pme_state(
            state,
            grid,
            2.0,
            0.1,
            1.0e-5,
            "frozen_mean",
            n_angles=4,
            fov_max_iters=16,
            arnoldi_steps=6,
        )
        identity = assess_pme_state(
            state,
            grid,
            2.0,
            0.1,
            1.0e-5,
            "identity",
            n_angles=4,
            fov_max_iters=16,
            arnoldi_steps=6,
        )

    assert d0_frozen_mean(state[grid.interior_slice], 1.0) == pytest.approx(1.0)
    assert frozen["disk_rate"] < identity["disk_rate"]
