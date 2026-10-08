"""Algebraic smoke tests for the staged PME DST/tau blend."""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import build_pme_linearization
from moljax.experimental.pme_dst_tau_blend import (
    build_pme_single_reference_linearization,
    pme_active_single_reference_values,
    pme_dst_tau_blend_preconditioner,
)
from moljax.experimental.pme_preconditioner import pme_helmholtz_preconditioner


def test_tau_blend_is_exact_for_constant_diffusivity() -> None:
    """A constant coefficient makes every DST reference inverse identical."""
    grid = NodeCenteredDirichletGrid.uniform(32, -1.0, 1.0)
    state = jnp.full((grid.nx,), 0.75, dtype=jnp.float64)
    rhs = jax.random.normal(jax.random.PRNGKey(20261020), (grid.nx,), dtype=jnp.float64)
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=2.0,
        dt=0.05,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=5,
    )
    expected = pme_helmholtz_preconditioner(1.5, 0.05, grid).apply(rhs)
    assert jnp.allclose(blend.apply(rhs), expected, rtol=1.0e-12, atol=1.0e-12)


def test_tau_blend_passes_compact_support_zero_region_through() -> None:
    """The explicit zero-coefficient policy is identity on an all-zero state."""
    grid = NodeCenteredDirichletGrid.uniform(32, -1.0, 1.0)
    state = jnp.zeros((grid.nx,), dtype=jnp.float64)
    rhs = jax.random.normal(jax.random.PRNGKey(20261021), (grid.nx,), dtype=jnp.float64)
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=2.0,
        dt=0.05,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=3,
    )
    assert jnp.array_equal(blend.apply(rhs), rhs)


def test_tau_blend_reference_quantiles_exclude_zero_region() -> None:
    """Reference coefficients are sampled only from active diffusion nodes."""
    grid = NodeCenteredDirichletGrid.uniform(32, -1.0, 1.0)
    state = (
        jnp.zeros((grid.nx,), dtype=jnp.float64)
        .at[10:13]
        .set(jnp.array([0.5, 1.0, 1.5], dtype=jnp.float64))
    )
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=2.0,
        dt=0.05,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=3,
    )
    assert jnp.allclose(blend.reference_values, jnp.array([1.0, 2.0, 3.0]))


def test_active_single_references_use_geometric_and_harmonic_diffusivity_means() -> None:
    """Single references are computed only from the tau blend's active support."""
    grid = NodeCenteredDirichletGrid.uniform(6, -1.0, 1.0)
    state = jnp.array([0.0, 0.5, 1.0, 2.0, 0.0, 0.0], dtype=jnp.float64)
    references = pme_active_single_reference_values(
        state,
        m=2.0,
        grid=grid,
        epsilon=1.0e-5,
    )
    expected = jnp.array([1.0, 2.0, 4.0], dtype=jnp.float64)
    assert references["active_node_count"] == 3
    assert references["inactive_node_count"] == 3
    assert references["geometric_mean"] == pytest.approx(2.0)
    assert references["harmonic_mean"] == pytest.approx(
        float(expected.size / jnp.sum(1.0 / expected))
    )


def test_explicit_single_reference_uses_the_existing_dst_i_path() -> None:
    """An explicit d0 gives the same operator and RHS as a named frozen variant."""
    grid = NodeCenteredDirichletGrid.uniform(32, -1.0, 1.0)
    state = jnp.linspace(0.1, 1.0, grid.nx, dtype=jnp.float64)
    named = build_pme_linearization(
        state,
        grid,
        m=2.0,
        dt=0.02,
        epsilon=1.0e-5,
        d0_kind="const",
        const_value=0.75,
    )
    explicit = build_pme_single_reference_linearization(
        state,
        grid,
        m=2.0,
        dt=0.02,
        epsilon=1.0e-5,
        d0=0.75,
    )
    vector = jax.random.normal(jax.random.PRNGKey(20261022), (grid.nx,), dtype=jnp.float64)
    assert jnp.allclose(explicit.rhs, named.rhs, rtol=0.0, atol=0.0)
    assert jnp.allclose(explicit.operator.matvec(vector), named.operator.matvec(vector))
