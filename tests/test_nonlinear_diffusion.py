"""Analytic checks for the experimental regularized porous-medium operator."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from moljax.core.grid import Grid1D
from moljax.experimental.nonlinear_diffusion import (
    barenblatt,
    porous_medium_flux_rhs,
    support_halfwidth,
)


@pytest.mark.parametrize(("m", "b"), [(2.0, 0.30), (3.0, 0.34)])
def test_regularization_error_is_small_against_coarse_grid_scale(m: float, b: float) -> None:
    """The state regularization is negligible before coarse-grid truncation error."""
    fine_grid = Grid1D.uniform(2001, -4.0, 4.0)
    u = barenblatt(fine_grid.x_coords(), 1.0, m, b=b)
    phi_true = u**m
    phi_regularized = (u**2 + 1.0e-10) ** (m / 2.0)

    regularization_error = float(jnp.max(jnp.abs(phi_regularized - phi_true)))
    coarse_h = 8.0 / 200.0
    coarse_grid_scale = float(coarse_h**2 * jnp.max(jnp.abs(phi_true)))
    ratio = regularization_error / coarse_grid_scale

    print(
        f"m={m:.0f}: regularization_error={regularization_error:.3e}, "
        f"coarse_grid_scale={coarse_grid_scale:.3e}, ratio={ratio:.3e}"
    )
    assert regularization_error < 1.0e-4 * coarse_grid_scale


def _heat_gaussian(x: jnp.ndarray, t: float) -> jnp.ndarray:
    """Whole-line heat kernel, exponentially small at the selected boundaries."""
    return jnp.exp(-(x**2) / (4.0 * t)) / jnp.sqrt(4.0 * jnp.pi * t)


def _heat_gaussian_laplacian(x: jnp.ndarray, t: float) -> jnp.ndarray:
    """Analytic second spatial derivative of ``_heat_gaussian``."""
    u = _heat_gaussian(x, t)
    return (x**2 / (4.0 * t**2) - 1.0 / (2.0 * t)) * u


def test_linear_control_has_second_order_max_norm_accuracy() -> None:
    """The three-point transformed-field stencil is second order for ``m=1``."""
    t = 0.1
    sizes = (101, 201, 401, 801)
    errors = []
    spacings = []

    for nx in sizes:
        grid = Grid1D.uniform(nx, -4.0, 4.0)
        x = grid.x_coords()
        u = (
            jnp.zeros(grid.nx_total, dtype=jnp.float64)
            .at[grid.interior_slice]
            .set(_heat_gaussian(x, t))
        )
        numerical = porous_medium_flux_rhs(u, grid, 1.0, epsilon=0.0)[grid.interior_slice]
        exact = _heat_gaussian_laplacian(x, t)
        errors.append(float(jnp.max(jnp.abs(numerical - exact))))
        spacings.append(grid.dx)

    order = float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])
    print(f"linear-control max-norm errors={errors}, observed_order={order:.6f}")
    assert 1.7 <= order <= 2.3


# Chosen so R(2) is approximately 2.4, leaving a margin inside [-4, 4].
@pytest.mark.parametrize(("m", "b"), [(2.0, 0.30), (3.0, 0.34)])
def test_barenblatt_support_is_compact_and_contained(m: float, b: float) -> None:
    """The exact profile is non-negative, finite-mass, and zero outside its support."""
    radius = float(support_halfwidth(2.0, m, b))
    x = jnp.linspace(-4.0, 4.0, 4001, dtype=jnp.float64)
    u = barenblatt(x, 2.0, m, b=b)
    mass = float(jnp.trapezoid(u, x))
    outside = jnp.abs(x) > radius

    assert radius <= 3.6
    assert np.isfinite(mass)
    assert mass > 0.0
    assert float(jnp.min(u)) >= 0.0
    assert float(jnp.max(jnp.abs(u[outside]))) <= 1.0e-12
