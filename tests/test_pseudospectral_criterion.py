"""Smoke tests for the dense pseudospectral criterion."""

from __future__ import annotations

from math import pi

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moljax.conditioning import pseudospectrum_dense
from moljax.experimental.pseudospectral_criterion import (
    assess_dense_pseudospectral_criterion,
    dense_sigma_min,
    materialize_dense_operator,
)


def test_dense_sigma_min_matches_existing_dense_pseudospectrum() -> None:
    """The dense specialization preserves the conditioning API definition."""
    matrix = np.array([[1.0, 0.4], [0.0, 1.5]])

    def matvec(vector: jax.Array) -> jax.Array:
        return jnp.asarray(matrix) @ vector

    materialized = materialize_dense_operator(matvec, 2)
    existing = pseudospectrum_dense(
        matvec,
        2,
        jnp.array([0.0, 1.0]),
        jnp.array([0.0]),
    )
    assert np.allclose(materialized, matrix)
    assert dense_sigma_min(materialized, 0.0) == pytest.approx(existing.epsilon_zero)
    assert dense_sigma_min(materialized, 1.0) == pytest.approx(float(existing.sigma_min[0, 1]))


def test_clustered_normal_operator_is_certified_conservatively() -> None:
    """A separated connected cluster has a nonempty pseudospectral window."""
    matrix = np.diag([1.0, 1.2, 1.4])
    result = assess_dense_pseudospectral_criterion(
        matrix,
        grid_points_per_axis=41,
        domain_padding_fraction=0.75,
        polynomial_degrees=(2, 3, 4),
        target_tolerance=1.0e-8,
    )
    assert result.certified
    assert result.eps_connect_upper < result.selected_epsilon < result.epsilon_zero
    assert 0.0 < result.polynomial_effective_rate < 1.0
    assert result.theorem_prefactor is not None
    assert result.contour_arc_length_grid is not None
    assert result.selected_epsilon is not None
    assert result.theorem_prefactor == pytest.approx(
        result.contour_arc_length_grid / (2.0 * pi * result.selected_epsilon)
    )
    assert result.predicted_iterations is not None


def test_origin_eigenvalue_fails_closed() -> None:
    """A spectrum containing zero cannot receive an origin-exclusion certificate."""
    result = assess_dense_pseudospectral_criterion(
        np.diag([0.0, 1.0]),
        grid_points_per_axis=21,
    )
    assert not result.certified
    assert result.epsilon_zero == pytest.approx(0.0, abs=1.0e-14)
    assert result.selected_epsilon is None
    assert result.predicted_iterations is None
