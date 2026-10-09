"""Algebraic smoke tests for the staged PME DST/tau blend."""

from __future__ import annotations

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import regularized_porous_medium_potential
from moljax.experimental.pme_conditioning import build_pme_linearization
from moljax.experimental.pme_dst_tau_blend import (
    build_pme_single_reference_linearization,
    pme_active_single_reference_values,
    pme_dst_tau_blend_preconditioner,
    porous_medium_diffusivity,
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
    d0 = float(porous_medium_diffusivity(state, 2.0, 1.0e-5)[0])
    expected = pme_helmholtz_preconditioner(d0, 0.05, grid).apply(rhs)
    assert jnp.allclose(blend.apply(rhs), expected, rtol=1.0e-12, atol=1.0e-12)


def test_tau_blend_passes_compact_support_zero_region_through() -> None:
    """The identity policy applies where the floor is below the threshold.

    At m = 4 the regularization floor ``4 * epsilon**3 = 4e-15`` lies below
    the default absolute threshold, so every node of the zero state is
    inactive.
    """
    grid = NodeCenteredDirichletGrid.uniform(32, -1.0, 1.0)
    state = jnp.zeros((grid.nx,), dtype=jnp.float64)
    rhs = jax.random.normal(jax.random.PRNGKey(20261021), (grid.nx,), dtype=jnp.float64)
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=4.0,
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
        m=4.0,
        dt=0.05,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=3,
    )
    expected = porous_medium_diffusivity(jnp.array([0.5, 1.0, 1.5]), 4.0, 1.0e-5)
    assert jnp.allclose(blend.reference_values, expected, rtol=1.0e-12, atol=0.0)


def test_active_single_references_use_geometric_and_harmonic_diffusivity_means() -> None:
    """Single references are computed only from the tau blend's active support."""
    grid = NodeCenteredDirichletGrid.uniform(6, -1.0, 1.0)
    state = jnp.array([0.0, 0.5, 1.0, 1.5, 0.0, 0.0], dtype=jnp.float64)
    references = pme_active_single_reference_values(
        state,
        m=4.0,
        grid=grid,
        epsilon=1.0e-5,
    )
    # Asymmetric on a log scale: the extrema mean sqrt(0.5 * 13.5) = 2.598
    # differs from the active-value mean (0.5 * 4 * 13.5)**(1/3) = 3.
    expected = porous_medium_diffusivity(jnp.array([0.5, 1.0, 1.5]), 4.0, 1.0e-5)
    assert references["active_node_count"] == 3
    assert references["inactive_node_count"] == 3
    assert references["geometric_mean"] == pytest.approx(
        float(jnp.exp(jnp.mean(jnp.log(expected)))), rel=1.0e-12
    )
    assert references["geometric_mean"] == pytest.approx(3.0, rel=1.0e-9)
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


@pytest.mark.parametrize("m", [2.0, 3.0, 4.0, 8.0])
def test_tau_blend_diffusivity_is_the_derivative_of_the_residual_potential(m: float) -> None:
    """The blend coefficient equals the Jacobian coefficient, sign and floor included."""
    values = jnp.array([-0.5, -1.0e-3, 0.0, 1.0e-6, 0.25, 1.0], dtype=jnp.float64)
    derivative = jax.vmap(
        jax.grad(lambda u: regularized_porous_medium_potential(u, m, epsilon=1.0e-5))
    )(values)
    assert jnp.allclose(
        porous_medium_diffusivity(values, m, 1.0e-5), derivative, rtol=1.0e-12, atol=0.0
    )


def test_tau_blend_keeps_the_m2_regularization_floor_active() -> None:
    """At m = 2 the floor 2 * epsilon exceeds the threshold, so no node is inactive."""
    grid = NodeCenteredDirichletGrid.uniform(16, -1.0, 1.0)
    state = jnp.zeros((grid.nx,), dtype=jnp.float64)
    rhs = jax.random.normal(jax.random.PRNGKey(20261023), (grid.nx,), dtype=jnp.float64)
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=2.0,
        dt=0.05,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=3,
    )
    assert not bool(jnp.any(blend.inactive_weight))
    expected = pme_helmholtz_preconditioner(2.0e-5, 0.05, grid).apply(rhs)
    assert jnp.allclose(blend.apply(rhs), expected, rtol=1.0e-12, atol=1.0e-12)


def test_tau_blend_single_reference_has_unit_active_weights() -> None:
    """One reference on a variable state gives finite unit weights, not 0/0."""
    grid = NodeCenteredDirichletGrid.uniform(32, -1.0, 1.0)
    state = jnp.linspace(0.1, 1.0, grid.nx, dtype=jnp.float64)
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=2.0,
        dt=0.02,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=1,
    )
    rhs = jnp.ones((grid.nx,), dtype=jnp.float64)
    assert jnp.array_equal(blend.input_weights, jnp.ones((1, grid.nx)))
    expected = pme_helmholtz_preconditioner(float(blend.reference_values[0]), 0.02, grid).apply(rhs)
    assert jnp.allclose(blend.apply(rhs), expected, rtol=1.0e-12, atol=1.0e-12)


def test_tau_blend_weights_stay_finite_far_from_every_reference() -> None:
    """Log-space normalization keeps distant coefficients finite and normalized."""
    grid = NodeCenteredDirichletGrid.uniform(64, -1.0, 1.0)
    state = jnp.geomspace(1.0e-4, 1.0, grid.nx, dtype=jnp.float64)
    blend = pme_dst_tau_blend_preconditioner(
        state,
        m=8.0,
        dt=0.02,
        grid=grid,
        epsilon=1.0e-5,
        reference_count=2,
    )
    active = blend.inactive_weight == 0.0
    sums = jnp.sum(blend.input_weights, axis=0)
    assert bool(jnp.all(jnp.isfinite(blend.input_weights)))
    assert jnp.allclose(sums[active], 1.0, rtol=0.0, atol=1.0e-14)
