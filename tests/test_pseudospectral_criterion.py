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
    _contour_modulus_bound,
    _contour_sigma_min_lower_bound,
    _fit_polynomial,
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
    assert result.contour_sigma_min_lower_bound is not None
    # The resolvent floor is proven on the contour, not assumed from epsilon.
    # Both reviews measured sigma_min = 0.2397 on this contour, below the
    # selected level 0.2536 that the old prefactor divided by.
    assert 0.0 < result.contour_sigma_min_lower_bound < 0.239736845
    assert result.theorem_prefactor == pytest.approx(
        result.contour_arc_length_grid / (2.0 * pi * result.contour_sigma_min_lower_bound)
    )
    assert result.contour_modulus_bound is not None
    assert 0.0 < result.contour_modulus_bound < 1.0
    assert result.predicted_iterations is not None


def test_contour_sigma_floor_holds_between_samples() -> None:
    """The edge floor never exceeds the true resolvent floor anywhere on the edge."""
    eigenvalues = np.array([1.0, 1.2 + 0.1j, 1.4])
    matrix = np.diag(eigenvalues)
    real_grid = np.linspace(0.0, 2.0, 21)
    imag_grid = np.linspace(-0.6, 0.6, 13)
    sigma = np.min(
        np.abs(
            (real_grid[None, :, None] + 1j * imag_grid[:, None, None]) - eigenvalues[None, None, :]
        ),
        axis=-1,
    )
    rng = np.random.default_rng(20261009)
    starts = rng.uniform(0.2, 1.8, 64) + 1j * rng.uniform(-0.4, 0.4, 64)
    ends = starts + 0.05 * np.exp(2j * pi * rng.uniform(size=64))
    floor = _contour_sigma_min_lower_bound(starts, ends, sigma, real_grid, imag_grid, 0.0)
    fractions = np.linspace(0.0, 1.0, 401)
    points = starts[:, None] + fractions[None, :] * (ends - starts)[:, None]
    true_minimum = min(dense_sigma_min(matrix, complex(point)) for point in points.ravel())
    assert floor <= true_minimum


def test_contour_modulus_bound_covers_points_between_samples() -> None:
    """A polynomial that vanishes at its samples still gets a nonzero bound.

    The square-envelope fit on {0.9, 1.1} reports a sampled maximum of zero,
    although the fitted polynomial has modulus 0.0101 at z = 1.
    """
    fitted = _fit_polynomial(np.array([0.9, 1.1], dtype=np.complex128), 2)
    assert fitted is not None
    sampled, _, coefficients, scale = fitted
    assert sampled == pytest.approx(0.0, abs=1.0e-12)
    starts = np.array([0.9 + 0.0j])
    ends = np.array([1.1 + 0.0j])
    bound, _ = _contour_modulus_bound(starts, ends, coefficients, scale)
    points = np.linspace(0.9, 1.1, 20001) / scale
    values = 1.0 + sum(
        (real + 1j * imag) * points ** (order + 1)
        for order, (real, imag) in enumerate(coefficients)
    )
    true_maximum = float(np.max(np.abs(values)))
    assert true_maximum == pytest.approx(1.0 / 99.0, rel=1.0e-6)
    assert bound >= true_maximum


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
