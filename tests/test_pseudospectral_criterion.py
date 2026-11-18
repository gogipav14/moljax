"""Smoke tests for the dense pseudospectral criterion."""

from __future__ import annotations

from fractions import Fraction
from math import comb, pi

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
    assert result.contour_modulus_bound_origin_only is not None
    assert result.contour_modulus_bound <= result.contour_modulus_bound_origin_only
    assert result.minimum_origin_only_contour_bound is not None
    assert result.minimum_origin_only_contour_bound <= result.contour_modulus_bound_origin_only
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
    # 2**40 there.  The gamma_160 evaluation term must therefore dominate the
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


def _exact_modulus_squared(
    coefficients: np.ndarray,
    point: tuple[Fraction, Fraction],
    perturbations: np.ndarray | None = None,
) -> Fraction:
    """Evaluate an ascending complex polynomial exactly from its float inputs."""
    real = Fraction(0)
    imag = Fraction(0)
    x, y = point
    for index in range(coefficients.size - 1, -1, -1):
        coefficient = coefficients[index]
        a = Fraction.from_float(float(coefficient.real))
        b = Fraction.from_float(float(coefficient.imag))
        if perturbations is not None:
            a += Fraction.from_float(float(perturbations[index].real))
            b += Fraction.from_float(float(perturbations[index].imag))
        real, imag = real * x - imag * y + a, real * y + imag * x + b
    return real * real + imag * imag


def _fraction_point(point: complex) -> tuple[Fraction, Fraction]:
    return Fraction.from_float(float(point.real)), Fraction.from_float(float(point.imag))


def test_horner_upper_encloses_exact_complex_polynomials() -> None:
    """Check random, cancelling and coefficient-uncertain cases with rationals."""
    rng = np.random.default_rng(20261118)
    cases: list[tuple[np.ndarray, complex]] = []
    for _ in range(300):
        degree = int(rng.integers(1, 41))
        moduli = 10.0 ** rng.uniform(-8.0, 8.0, degree + 1)
        coefficients = moduli * np.exp(2j * pi * rng.random(degree + 1))
        point = 10.0 ** rng.uniform(-3.0, 3.0) * np.exp(2j * pi * rng.random())
        cases.append((coefficients, complex(point)))
    for degree in (8, 16, 24, 32, 40):
        radius = float(rng.uniform(0.5, 2.0))
        coefficients = np.asarray(
            [(-1) ** order * comb(degree, order) / radius**order for order in range(degree + 1)],
            dtype=np.complex128,
        )
        cases.append((coefficients, complex(radius * (1.0 + 1.0e-6))))
    cases.append(
        (
            np.asarray(
                [(-1) ** order * comb(40, order) for order in range(41)],
                dtype=np.complex128,
            ),
            1.0 + 0.0j,
        )
    )
    for index, (coefficients, point) in enumerate(cases):
        upper, _, _ = _horner_modulus_upper(np.asarray([point]), coefficients)
        assert np.isfinite(upper[0])
        bound_squared = Fraction.from_float(float(upper[0])) ** 2
        assert bound_squared >= _exact_modulus_squared(coefficients, _fraction_point(point))
        if index % 5 == 0:
            errors = 0.01 * np.abs(coefficients)
            perturbations = 0.25 * errors * rng.choice((-1.0, 1.0), coefficients.size)
            uncertain_upper, _, _ = _horner_modulus_upper(
                np.asarray([point]), coefficients, coefficient_errors=errors
            )
            assert all(
                abs(Fraction.from_float(float(error))) <= Fraction.from_float(float(limit))
                for error, limit in zip(perturbations, errors, strict=True)
            )
            assert Fraction.from_float(float(uncertain_upper[0])) ** 2 >= _exact_modulus_squared(
                coefficients, _fraction_point(point), perturbations
            )


def test_horner_upper_encloses_subnormal_terms() -> None:
    """Cover gradual underflow in the complex Horner recurrence."""
    subnormal = np.nextafter(0.0, 1.0)
    coefficients = np.asarray(
        [complex((-1) ** index * subnormal, subnormal) for index in range(41)],
        dtype=np.complex128,
    )
    point = 0.5 + 0.25j
    upper, _, evaluation_error = _horner_modulus_upper(np.asarray([point]), coefficients)
    assert evaluation_error[0] >= 8 * 40 * subnormal
    assert Fraction.from_float(float(upper[0])) ** 2 >= _exact_modulus_squared(
        coefficients, _fraction_point(point)
    )


def test_centred_enclosure_falls_back_on_actual_underflow() -> None:
    """A nonzero translation product becoming subnormal uses the origin enclosure."""
    starts = np.array([0.0, 2.0e-60, 2.0e-60 + 1.0e-60j, 1.0e-60j])
    ends = np.roll(starts, -1)
    coefficients = np.array([0.0, 1.0e-200], dtype=np.complex128)
    product = coefficients[1] * (1.0e-60 + 0.5e-60j) ** 2
    assert 0.0 < abs(product) < np.finfo(np.float64).tiny
    assert _centered_polynomial_derivative_bound(starts, ends, coefficients, 1.0) == float("inf")
    serialized = tuple((float(value.real), float(value.imag)) for value in coefficients)
    assert _contour_modulus_bound(starts, ends, serialized, 1.0) == _contour_modulus_bound(
        starts, ends, serialized, 1.0, basis="origin"
    )


def test_tiny_normal_coefficient_keeps_previous_centred_bound_bitwise() -> None:
    """A small coefficient is safe when the products actually used stay normal."""
    starts = np.array([0.4 - 0.1j, 0.6 - 0.1j, 0.6 + 0.1j, 0.4 + 0.1j])
    ends = np.roll(starts, -1)
    coefficients = np.zeros(40, dtype=np.complex128)
    coefficients[0] = 0.1
    coefficients[-1] = 1.0e-30
    derivative = _centered_polynomial_derivative_bound(starts, ends, coefficients, 1.0)
    assert derivative.hex() == "0x1.99999999ae9d0p-4"
    serialized = tuple((float(value.real), float(value.imag)) for value in coefficients)
    bound, lipschitz = _contour_modulus_bound(starts, ends, serialized, 1.0)
    assert bound.hex() == "0x1.0f9ad455c5f2dp+0"
    assert float(lipschitz).hex() == "0x1.99999999999e3p-4"


def test_contour_upper_encloses_exact_points_between_samples() -> None:
    """Check 100 random rectangles at 65 rational positions on each edge."""
    rng = np.random.default_rng(20261118)
    for _ in range(100):
        center = complex(*rng.uniform(-1.0, 1.0, 2))
        halfwidth, halfheight = rng.uniform(0.01, 0.5, 2)
        corners = np.asarray(
            [
                center + complex(-halfwidth, -halfheight),
                center + complex(halfwidth, -halfheight),
                center + complex(halfwidth, halfheight),
                center + complex(-halfwidth, halfheight),
            ]
        )
        degree = int(rng.integers(1, 25))
        coefficients = np.asarray(
            [1.0, *(rng.normal(size=degree) + 1j * rng.normal(size=degree))],
            dtype=np.complex128,
        )
        serialized = tuple((float(z.real), float(z.imag)) for z in coefficients[1:])
        bound, _ = _contour_modulus_bound(corners, np.roll(corners, -1), serialized, 1.0)
        absolute_sum_bound, _ = _contour_modulus_bound(
            corners, np.roll(corners, -1), serialized, 1.0, basis="absolute_sum"
        )
        origin_bound, _ = _contour_modulus_bound(
            corners, np.roll(corners, -1), serialized, 1.0, basis="origin"
        )
        assert np.isfinite(bound)
        assert bound <= absolute_sum_bound
        assert bound <= origin_bound
        bound_squared = Fraction.from_float(float(bound)) ** 2
        absolute_sum_bound_squared = Fraction.from_float(float(absolute_sum_bound)) ** 2
        for start, end in zip(corners, np.roll(corners, -1), strict=True):
            start_exact = _fraction_point(start)
            end_exact = _fraction_point(end)
            for index in range(65):
                ratio = Fraction(index, 64)
                point = (
                    start_exact[0] + ratio * (end_exact[0] - start_exact[0]),
                    start_exact[1] + ratio * (end_exact[1] - start_exact[1]),
                )
                assert bound_squared >= _exact_modulus_squared(coefficients, point)
                assert absolute_sum_bound_squared >= _exact_modulus_squared(coefficients, point)


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
            origin_modulus_bound, _ = _contour_modulus_bound(
                starts, ends, serialized, scale, basis="origin"
            )
            assert new_modulus_bound <= origin_modulus_bound
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
