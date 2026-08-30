"""Matplotlib figures for already-computed conditioning diagnostics.

The functions in this module consume diagnostic arrays and result objects; they
never evaluate an operator or run a numerical diagnostic themselves.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from moljax.conditioning.field_of_values import FieldOfValuesResult
from moljax.conditioning.pseudospectra import PseudospectraResult


def _plotting_backend() -> tuple[Any, Any]:
    """Import matplotlib lazily and explain how to install the optional extra."""
    try:
        from matplotlib import pyplot as plt
        from matplotlib.patches import Circle
    except ModuleNotFoundError as error:
        raise ImportError(
            "conditioning figures require the optional visualization dependencies; "
            "install moljax with `pip install -e '.[viz]'`."
        ) from error
    return plt, Circle


def _axes(ax: Any | None) -> tuple[Any, Any]:
    """Return a figure and axes, creating one when the caller supplies none."""
    plt, _ = _plotting_backend()
    if ax is None:
        return plt.subplots(constrained_layout=True)
    return ax.figure, ax


def _padded_limits(values: np.ndarray, minimum_span: float) -> tuple[float, float]:
    """Return finite limits with padding and a minimum visible span."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("plot coordinates must contain at least one finite value")
    lower = float(np.min(finite))
    upper = float(np.max(finite))
    span = max(upper - lower, minimum_span, np.finfo(np.float64).eps)
    padding = 0.08 * span
    midpoint = 0.5 * (lower + upper)
    half_span = 0.5 * span + padding
    return midpoint - half_span, midpoint + half_span


def _set_visible_limits(
    axis: Any,
    real_values: np.ndarray,
    imag_values: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Set padded limits and avoid equal-aspect collapse for nearly real data."""
    real_extent = float(np.ptp(real_values))
    imag_extent = float(np.ptp(imag_values))
    dominant_extent = max(real_extent, imag_extent, np.finfo(np.float64).eps)
    minimum_span = 0.10 * dominant_extent
    x_limits = _padded_limits(real_values, minimum_span)
    y_limits = _padded_limits(imag_values, minimum_span)
    axis.set_xlim(*x_limits)
    axis.set_ylim(*y_limits)
    degenerate = min(real_extent, imag_extent) / dominant_extent < 0.15
    if degenerate:
        # A circular artist can otherwise make matplotlib retain a very shallow
        # data-aspect box.  Keep the displayed data limits, but reserve a
        # readable plotting height for a nearly real spectrum.
        axis.set_aspect("auto")
        axis.set_box_aspect(0.55)
    else:
        axis.set_box_aspect(None)
        axis.set_aspect("equal", adjustable="box")
    return x_limits, y_limits


def plot_numerical_range(fov: FieldOfValuesResult, *, ax: Any | None = None) -> Any:
    """Plot a numerical-range boundary, its enclosing disk, and the origin."""
    figure, axis = _axes(ax)
    _, circle = _plotting_backend()
    boundary = np.asarray(fov.boundary, dtype=np.complex128)
    if boundary.size == 0:
        raise ValueError("numerical-range boundary must be nonempty")
    center = complex(fov.center)
    radius = float(fov.radius)
    disk_real = np.asarray([center.real - radius, center.real + radius])
    disk_imag = np.asarray([center.imag - radius, center.imag + radius])
    view_real = np.concatenate((boundary.real, disk_real, np.asarray([center.real, 0.0])))
    view_imag = np.concatenate((boundary.imag, disk_imag, np.asarray([center.imag, 0.0])))
    axis.plot(boundary.real, boundary.imag, "o-", markersize=3, label="boundary")
    axis.add_patch(
        circle(
            (center.real, center.imag),
            radius,
            fill=False,
            linestyle="--",
            color="tab:orange",
            label="enclosing disk",
        )
    )
    axis.scatter([0.0], [0.0], marker="+", s=80, color="black", label="origin", zorder=3)
    axis.scatter(
        [center.real],
        [center.imag],
        marker="x",
        color="tab:orange",
        label="disk center",
        zorder=3,
    )
    _set_visible_limits(axis, view_real, view_imag)
    axis.set_xlabel("Re z")
    axis.set_ylabel("Im z")
    axis.set_title(f"Numerical range (rho / |c| = {fov.disk_rate:.3e})")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    return figure


def _pseudospectrum_arrays(
    result_or_grid: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract grids, singular values, and Ritz values from supported containers."""
    if isinstance(result_or_grid, PseudospectraResult):
        real = result_or_grid.real_grid
        imag = result_or_grid.imag_grid
        sigma_min = result_or_grid.sigma_min
        ritz = result_or_grid.ritz_values
    elif isinstance(result_or_grid, tuple) and len(result_or_grid) == 4:
        real, imag, sigma_min, ritz = result_or_grid
    elif isinstance(result_or_grid, dict):
        real = result_or_grid["real_grid"]
        imag = result_or_grid["imag_grid"]
        sigma_min = result_or_grid["sigma_min"]
        ritz = result_or_grid["ritz_values"]
    else:
        try:
            real = result_or_grid.real_grid
            imag = result_or_grid.imag_grid
            sigma_min = result_or_grid.sigma_min
            ritz = result_or_grid.ritz_values
        except AttributeError as error:
            raise TypeError(
                "result_or_grid must be a PseudospectraResult, a four-tuple, "
                "a mapping, or an object with pseudospectrum fields"
            ) from error
    return (
        np.asarray(real, dtype=np.float64),
        np.asarray(imag, dtype=np.float64),
        np.asarray(sigma_min, dtype=np.float64),
        np.asarray(ritz, dtype=np.complex128),
    )


def _pseudospectrum_view_values(
    real: np.ndarray,
    imag: np.ndarray,
    ritz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a plotted view from the grid, with a local zoom for Ritz clusters."""
    grid_real_span = float(np.ptp(real))
    grid_imag_span = float(np.ptp(imag))
    grid_span = max(grid_real_span, grid_imag_span, np.finfo(np.float64).eps)
    finite_ritz = ritz[np.isfinite(ritz.real) & np.isfinite(ritz.imag)]
    if finite_ritz.size:
        ritz_real_span = float(np.ptp(finite_ritz.real))
        ritz_imag_span = float(np.ptp(finite_ritz.imag))
        ritz_span = max(ritz_real_span, ritz_imag_span)
        if ritz_span < 0.25 * grid_span:
            half_span = max(0.5 * ritz_span, 0.05 * grid_span)
            midpoint = np.mean(finite_ritz)
            view_real = np.asarray(
                [
                    midpoint.real - half_span,
                    midpoint.real + half_span,
                    *finite_ritz.real,
                    0.0,
                ]
            )
            view_imag = np.asarray(
                [
                    midpoint.imag - half_span,
                    midpoint.imag + half_span,
                    *finite_ritz.imag,
                    0.0,
                ]
            )
            return view_real, view_imag
    return (
        np.concatenate((real, finite_ritz.real, np.asarray([0.0]))),
        np.concatenate((imag, finite_ritz.imag, np.asarray([0.0]))),
    )


def plot_pseudospectrum(result_or_grid: Any, *, ax: Any | None = None) -> Any:
    """Plot shifted-smallest-singular-value contours, Ritz values, and the origin.

    ``result_or_grid`` may be a :class:`PseudospectraResult`, a mapping with
    ``real_grid``, ``imag_grid``, ``sigma_min``, and ``ritz_values`` keys, or
    a four-tuple in that order.
    """
    figure, axis = _axes(ax)
    real, imag, sigma_min, ritz = _pseudospectrum_arrays(result_or_grid)
    if sigma_min.shape != (imag.size, real.size):
        raise ValueError("sigma_min must have shape (len(imag_grid), len(real_grid))")
    logged = np.log10(np.maximum(sigma_min, np.finfo(np.float64).tiny))
    low = float(np.min(logged))
    high = float(np.max(logged))
    if math.isclose(low, high, rel_tol=1.0e-10, abs_tol=1.0e-12):
        if real.size > 1 and imag.size > 1:
            artist = axis.pcolormesh(real, imag, logged, shading="auto", cmap="viridis")
        else:
            real_mesh, imag_mesh = np.meshgrid(real, imag)
            artist = axis.scatter(
                real_mesh.reshape(-1),
                imag_mesh.reshape(-1),
                c=logged.reshape(-1),
                cmap="viridis",
            )
    else:
        artist = axis.contour(real, imag, logged, levels=np.linspace(low, high, 8), cmap="viridis")
    figure.colorbar(artist, ax=axis, label="log10 sigma_min(zI - A)")
    axis.scatter(
        ritz.real,
        ritz.imag,
        s=52,
        color="tab:red",
        edgecolor="black",
        linewidth=0.6,
        label="Ritz values",
        zorder=4,
    )
    axis.scatter([0.0], [0.0], marker="+", s=80, color="black", label="origin", zorder=4)
    view_real, view_imag = _pseudospectrum_view_values(real, imag, ritz)
    _set_visible_limits(axis, view_real, view_imag)
    axis.set_xlabel("Re z")
    axis.set_ylabel("Im z")
    axis.set_title("Reduced pseudospectrum")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8, loc="upper left")
    return figure


def plot_rate_scaling(
    sizes: Any,
    r1: Any,
    r2: Any,
    r3: Any,
    *,
    measured: Any | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot rate estimates across a multi-size sweep and optional Krylov work."""
    figure, axis = _axes(ax)
    sizes_array = np.asarray(sizes, dtype=np.float64).reshape(-1)
    rates = {
        "enclosing-disk rate": np.asarray(r1, dtype=np.float64).reshape(-1),
        "traced-boundary rate": np.asarray(r2, dtype=np.float64).reshape(-1),
        "bulk-clustering rate": np.asarray(r3, dtype=np.float64).reshape(-1),
    }
    if sizes_array.size == 0 or any(values.size != sizes_array.size for values in rates.values()):
        raise ValueError("sizes and all rate arrays must be nonempty and have equal length")
    if sizes_array.size < 2:
        raise ValueError("plot_rate_scaling expects a size sweep with at least two points")
    for label, values in rates.items():
        axis.plot(sizes_array, values, "o-", label=label)
    axis.set_xscale("log")
    axis.set_xlabel("problem size")
    axis.set_ylabel("predicted convergence rate")
    axis.set_title("Conditioning rate estimates")
    axis.grid(True, alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    if measured is not None:
        measured_array = np.asarray(measured, dtype=np.float64).reshape(-1)
        if measured_array.size != sizes_array.size:
            raise ValueError("measured must have the same length as sizes")
        secondary = axis.twinx()
        secondary.plot(
            sizes_array,
            measured_array,
            "x--",
            color="0.35",
            label="measured Krylov work",
        )
        secondary.set_ylabel("measured Krylov work", color="0.35")
        handles_secondary, labels_secondary = secondary.get_legend_handles_labels()
        handles += handles_secondary
        labels += labels_secondary
    axis.legend(handles, labels, fontsize=8)
    return figure


def plot_residual_envelope(
    residuals: Any,
    disk_rate: float,
    *,
    prefactor: float = 1.0 + math.sqrt(2.0),
    ax: Any | None = None,
) -> Any:
    """Plot supplied residuals against the Crouzeix--Palencia disk envelope."""
    figure, axis = _axes(ax)
    observed = np.asarray(residuals, dtype=np.float64).reshape(-1)
    if observed.size == 0:
        raise ValueError("residuals must be nonempty")
    iterations = np.arange(observed.size)
    envelope = prefactor * float(disk_rate) ** iterations
    floor = np.finfo(np.float64).tiny
    axis.semilogy(iterations, np.maximum(observed, floor), "o-", label="supplied residuals")
    axis.semilogy(iterations, np.maximum(envelope, floor), ":", label="CP envelope")
    axis.set_xlabel("Krylov iteration")
    axis.set_ylabel("relative residual")
    axis.set_title("Residuals and Crouzeix--Palencia envelope")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8)
    return figure
