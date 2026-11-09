"""Smoke tests for the dense pseudospectral criterion."""

from __future__ import annotations

from math import ceil, comb, pi

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moljax.conditioning import pseudospectrum_dense
from moljax.experimental.pseudospectral_criterion import (
    _centered_polynomial_derivative_bound,
    _contour_modulus_bound,
    _contour_sigma_min_lower_bound,
    _fit_polynomial,
    _horner_modulus_upper,
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


def test_degree_40_evaluation_roundoff_is_included_in_contour_bound() -> None:
    """Large cancelling coefficients cannot hide Horner evaluation error."""
    degree = 40
    # p(z) = (1 - z)**40 vanishes at z=1, but its monomial absolute sum is
    # 2**40 there.  The gamma_80 evaluation term must therefore dominate the
    # computed zero and enter the rigorous contour maximum.
    coefficients = np.asarray(
        [(-1) ** order * comb(degree, order) for order in range(degree + 1)],
        dtype=np.complex128,
    )
    upper, absolute_sum, evaluation_error = _horner_modulus_upper(
        np.asarray([1.0 + 0.0j]),
        coefficients,
    )
    assert absolute_sum[0] >= 2.0**degree
    assert evaluation_error[0] > 1.0e-3
    assert upper[0] >= evaluation_error[0]

    serialized = tuple((float(value.real), float(value.imag)) for value in coefficients[1:])
    bound, _ = _contour_modulus_bound(
        np.asarray([1.0 + 0.0j]),
        np.asarray([1.0 + 0.0j]),
        serialized,
        1.0,
    )
    assert bound >= evaluation_error[0]


def test_centered_derivative_bound_covers_random_fitted_polynomials() -> None:
    """The translated enclosure covers dense contour samples through degree 40."""
    rng = np.random.default_rng(20261109)
    angles = np.linspace(0.0, 2.0 * pi, 24, endpoint=False)
    successful_fits = {degree: 0 for degree in range(4, 41, 4)}
    for _ in range(4):
        center = (0.8 + 0.3 * rng.normal()) + 0.2j * rng.normal()
        radii = 0.2 + 0.12 * rng.random(angles.size)
        starts = center + radii * np.exp(1j * angles)
        ends = np.roll(starts, -1)
        fitting_points = center + (
            0.18 * np.sqrt(rng.random(96)) * np.exp(2j * pi * rng.random(96))
        )
        fractions = np.linspace(0.0, 1.0, 100 * angles.size + 1)
        contour = starts[:, None] + fractions[None, :] * (ends - starts)[:, None]
        for degree in range(4, 41, 4):
            fitted = _fit_polynomial(fitting_points, degree)
            if fitted is None:
                continue
            successful_fits[degree] += 1
            _, _, serialized, scale = fitted
            coefficients = np.asarray([complex(*value) for value in serialized])
            derivative_bound = _centered_polynomial_derivative_bound(
                starts,
                ends,
                coefficients,
                scale,
            )
            scaled = contour / scale
            derivative = sum(
                order * coefficient * scaled ** (order - 1) / scale
                for order, coefficient in enumerate(coefficients, start=1)
            )
            assert derivative_bound >= float(np.max(np.abs(derivative)))
            new_modulus_bound, _ = _contour_modulus_bound(
                starts,
                ends,
                serialized,
                scale,
            )
            orders = np.arange(1, degree + 1)
            radius = float(np.max(np.abs(np.concatenate((starts, ends))))) / scale
            origin_derivative_bound = float(
                np.sum(orders * np.abs(coefficients) * radius ** (orders - 1)) / scale
            )
            edge_length = float(np.max(np.abs(ends - starts)))
            subdivisions = min(
                max(8, ceil(origin_derivative_bound * edge_length / (2.0e-3))),
                1024,
            )
            old_fractions = np.linspace(0.0, 1.0, subdivisions + 1)
            old_points = starts[:, None] + old_fractions[None, :] * (ends - starts)[:, None]
            old_scaled = old_points / scale
            old_values = 1.0 + sum(
                coefficient * old_scaled**order
                for order, coefficient in enumerate(coefficients, start=1)
            )
            old_modulus_bound = float(np.max(np.abs(old_values))) + (
                origin_derivative_bound * edge_length / (2.0 * subdivisions)
            )
            assert new_modulus_bound <= old_modulus_bound
    assert min(successful_fits.values()) >= 3


def test_shifted_sigma_min_dominates_distance_to_numerical_range() -> None:
    """Random non-normal matrices satisfy the numerical-range distance lemma."""
    rng = np.random.default_rng(20261109)
    dimension = 40
    for _ in range(2):
        matrix = np.triu(rng.normal(size=(dimension, dimension)))
        matrix += 0.05 * np.tril(rng.normal(size=(dimension, dimension)), -1)
        matrix = matrix.astype(np.complex128)
        matrix += 0.2j * np.triu(rng.normal(size=(dimension, dimension)), 1)
        directions = np.linspace(0.0, 2.0 * pi, 720, endpoint=False)
        supports = np.empty(directions.size)
        for index, theta in enumerate(directions):
            rotation = np.exp(1j * theta)
            hermitian = 0.5 * (rotation * matrix + np.conj(rotation) * matrix.conj().T)
            supports[index] = np.linalg.eigvalsh(hermitian)[-1]
        real_points = np.linspace(-7.0, 7.0, 7)
        imag_points = np.linspace(-5.0, 5.0, 5)
        identity = np.eye(dimension, dtype=np.complex128)
        for real in real_points:
            for imag in imag_points:
                point = complex(real, imag)
                shifted = point * identity - matrix
                _, singular_values, right_adjoint = np.linalg.svd(shifted)
                sigma_min = float(singular_values[-1])
                right = right_adjoint[-1].conj()
                rayleigh_point = complex(np.vdot(right, matrix @ right))
                # The support-grid maximum is a lower approximation to
                # dist(point, W(A)); the Rayleigh witness proves the exact lemma.
                sampled_distance = max(
                    0.0,
                    float(np.max(np.real(np.exp(1j * directions) * point) - supports)),
                )
                assert sigma_min + 2.0e-12 >= sampled_distance
                assert sigma_min + 2.0e-12 >= abs(point - rayleigh_point)


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
