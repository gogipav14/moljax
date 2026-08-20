"""Validation of the experimental degenerate Porous--Fisher operator."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from benchmarks.porous_fisher_conditioning import ReactionStudyConfig, run_reaction_study
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.porous_fisher import (
    porous_fisher_rhs,
    porous_fisher_traveling_wave,
    wave_front_position,
    wave_speed,
)


@pytest.mark.slow
def test_sharp_traveling_wave_satisfies_the_discrete_ode_away_from_edge_and_boundaries() -> None:
    """The second-order spatial operator agrees with the exact wave ODE."""
    r = 1.0
    c = wave_speed(r)
    t0 = 0.25
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(2001, -8.0, 8.0)
        x = grid.x_coords()
        wave = porous_fisher_traveling_wave(x, t0, r=r, c=c)
        numerical_rhs = porous_fisher_rhs(wave, grid, r=r)
        xi = x - c * t0
        derivative = -0.5 * c * jnp.exp(0.5 * c * xi)
        traveling_wave_rhs = -c * derivative
        smooth_interior = (xi < -0.2) & (x > grid.x_min + 0.5)
        residual = float(
            jnp.max(jnp.abs(numerical_rhs[smooth_interior] - traveling_wave_rhs[smooth_interior]))
        )

    assert residual < 2.0e-4


def test_sharp_wave_is_positive_compact_ahead_and_stays_inside_the_domain() -> None:
    """The finite-speed front remains interior over the selected window."""
    r = 1.0
    c = wave_speed(r)
    t_start = 0.0
    t_end = 0.5
    with jax.enable_x64(True):
        grid = NodeCenteredDirichletGrid.uniform(401, -8.0, 8.0)
        x = grid.x_coords()
        wave = porous_fisher_traveling_wave(x, t_end, r=r, c=c)
        edge = wave_front_position(t_end, c=c)

    assert grid.x_min + 1.0 < wave_front_position(t_start, c=c)
    assert edge < grid.x_max - 1.0
    assert float(jnp.min(wave)) >= 0.0
    assert float(jnp.max(jnp.abs(wave[x >= edge]))) <= 1.0e-12


@pytest.mark.slow
def test_reaction_axis_study_has_real_identity_dynamic_range_and_adjoint_gates(tmp_path) -> None:
    """The reaction benchmark records genuine linear work for each reaction level."""
    report = run_reaction_study(
        ReactionStudyConfig(
            nx=128,
            reaction_values=(0.0, 1.0, 100.0),
            d0_kinds=("frozen_bulk", "identity"),
            n_angles=3,
            fov_max_iters=4,
            arnoldi_steps=4,
            output_path=str(tmp_path / "porous_fisher_conditioning.json"),
        )
    )
    identity_range = report["verdict_on_decision_procedure"]["identity_iteration_range"]
    reaction_effect = report["reaction_effect"]

    assert identity_range["ratio"] >= 10.0
    assert len(reaction_effect["by_reaction"]) == 3
    assert all(record["adjoint_identity"] <= 1.0e-8 for record in report["records"])
