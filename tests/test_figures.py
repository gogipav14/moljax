"""Smoke tests for the optional conditioning-report plotting API."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg", force=True)

import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from moljax.conditioning import (
    FieldOfValuesResult,
    PseudospectraResult,
    plot_numerical_range,
    plot_pseudospectrum,
    plot_rate_scaling,
    plot_residual_envelope,
)


def _field_of_values() -> FieldOfValuesResult:
    """Return a small synthetic numerical-range result for plotting."""
    angles = np.linspace(0.0, 2.0 * math.pi, 9, endpoint=False)
    boundary = 2.0 + 0.25 * np.exp(1j * angles)
    return FieldOfValuesResult(
        boundary=jnp.asarray(boundary),
        center=2.0 + 0.0j,
        radius=0.25,
        disk_rate=0.125,
        origin_enclosed=False,
        cp_prefactor=1.0 + math.sqrt(2.0),
    )


def _pseudospectra() -> PseudospectraResult:
    """Return a compact synthetic pseudospectrum for plotting."""
    real = jnp.linspace(0.5, 2.5, 4)
    imag = jnp.linspace(-1.0, 1.0, 3)
    sigma_min = jnp.asarray([[0.8, 0.4, 0.8, 1.2], [0.5, 0.1, 0.5, 1.0], [0.8, 0.4, 0.8, 1.2]])
    return PseudospectraResult(
        real_grid=real,
        imag_grid=imag,
        sigma_min=sigma_min,
        ritz_values=jnp.asarray([1.2 + 0.1j, 1.8 - 0.2j]),
        epsilon_zero=0.1,
    )


def _near_real_field_of_values() -> FieldOfValuesResult:
    """Return a nearly real result that must not collapse under equal aspect."""
    boundary = 2.0 + np.linspace(-0.2, 0.2, 7) + 1.0e-8j * np.linspace(-1.0, 1.0, 7)
    return FieldOfValuesResult(
        boundary=jnp.asarray(boundary),
        center=2.0 + 0.0j,
        radius=1.0e-8,
        disk_rate=5.0e-9,
        origin_enclosed=False,
        cp_prefactor=1.0 + math.sqrt(2.0),
    )


def _clustered_pseudospectra() -> PseudospectraResult:
    """Return constant-level data with tightly clustered Ritz values."""
    real = jnp.linspace(-20.0, 20.0, 5)
    imag = jnp.linspace(-1.0e-6, 1.0e-6, 3)
    return PseudospectraResult(
        real_grid=real,
        imag_grid=imag,
        sigma_min=jnp.ones((imag.size, real.size)),
        ritz_values=jnp.asarray([1.0 + 1.0e-8j, 1.0 - 1.0e-8j]),
        epsilon_zero=1.0,
    )


def _assert_finite_non_degenerate_limits(figure: Figure) -> None:
    """Check the primary plot axes have finite, nonzero view ranges."""
    axis = figure.axes[0]
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    assert np.isfinite(x_limits).all()
    assert np.isfinite(y_limits).all()
    assert x_limits[1] > x_limits[0]
    assert y_limits[1] > y_limits[0]


def _assert_external_legend(figure: Figure) -> None:
    """Check that a numerical-range legend cannot obscure plotted diagnostics."""
    figure.canvas.draw()
    axis = figure.axes[0]
    legend = axis.get_legend()
    assert legend is not None
    assert legend.get_window_extent().x0 >= axis.get_window_extent().x1


def test_conditioning_figures_return_populated_figures():
    """Each public plotting helper returns a matplotlib figure with axes."""
    numerical_range = plot_numerical_range(_near_real_field_of_values())
    pseudospectrum = plot_pseudospectrum(_clustered_pseudospectra())
    rate_scaling = plot_rate_scaling([16, 64], [0.4, 0.2], [0.3, 0.15], [0.2, 0.1], measured=[8, 5])
    residual_envelope = plot_residual_envelope([1.0, 0.2, 0.04], 0.25)
    figures = [numerical_range, pseudospectrum, rate_scaling, residual_envelope]
    try:
        assert all(isinstance(figure, Figure) and figure.axes for figure in figures)
        _assert_finite_non_degenerate_limits(numerical_range)
        _assert_finite_non_degenerate_limits(pseudospectrum)
        _assert_external_legend(numerical_range)
        assert numerical_range.axes[0].get_aspect() == "auto"
        assert numerical_range.axes[0].get_box_aspect() == 0.55
        assert pseudospectrum.axes[0].collections
        assert all(text.get_text() != "origin" for text in pseudospectrum.axes[0].texts)
        _, rate_labels = rate_scaling.axes[0].get_legend_handles_labels()
        assert rate_labels[:3] == [
            "enclosing-disk rate",
            "traced-boundary rate",
            "bulk-clustering rate",
        ]
        x_limits = pseudospectrum.axes[0].get_xlim()
        y_limits = pseudospectrum.axes[0].get_ylim()
        ritz = np.asarray(_clustered_pseudospectra().ritz_values)
        assert ritz.size > 0
        assert np.all((x_limits[0] <= ritz.real) & (ritz.real <= x_limits[1]))
        assert np.all((y_limits[0] <= ritz.imag) & (ritz.imag <= y_limits[1]))
    finally:
        for figure in figures:
            plt.close(figure)
