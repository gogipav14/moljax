"""Smoke tests for the targeted matrix-free pseudospectral estimator."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from moljax.conditioning import LinearizedOperator
from moljax.experimental.pseudospectral_criterion_matrix_free import (
    assess_targeted_pseudospectral_connectivity,
    estimate_sigma_min_matrix_free,
    matrix_free_ritz_values,
)


def _diagonal_operator(entries: list[float]) -> LinearizedOperator:
    diagonal = jnp.asarray(entries, dtype=jnp.float64)

    def action(vector: jax.Array) -> jax.Array:
        return diagonal * vector

    return LinearizedOperator(action, action, diagonal.size)


def test_matrix_free_sigma_min_matches_dense_diagonal_value() -> None:
    """Full-step Arnoldi recovers the shifted minimum singular value."""
    entries = [1.0, 1.2, 1.4, 1.6]
    operator = _diagonal_operator(entries)
    point = 1.3 + 0.2j
    result = estimate_sigma_min_matrix_free(
        operator,
        point,
        arnoldi_steps=operator.n,
        operator_norm_scale=2.0,
    )
    dense = np.diag(entries)
    truth = np.linalg.svd(point * np.eye(operator.n) - dense, compute_uv=False)[-1]
    assert result.estimate == pytest.approx(truth, rel=1.0e-12, abs=1.0e-12)
    assert result.normal_residual_norm < 1.0e-12
    assert result.operator_applications == 2 * (result.arnoldi_steps + 1)


def test_robust_refinement_is_matrix_free_and_matches_dense() -> None:
    """The robust path, including its fallback, never requires a dense matrix."""
    entries = np.linspace(0.01, 4.0, 16).tolist()
    operator = _diagonal_operator(entries)
    result = estimate_sigma_min_matrix_free(
        operator,
        0.0j,
        arnoldi_steps=operator.n,
        operator_norm_scale=5.0,
        refine_with_propack=True,
    )
    assert result.solver in {"propack", "normal_arnoldi"}
    assert result.estimate == pytest.approx(0.01, rel=1.0e-10, abs=1.0e-12)
    assert result.forward_applications > 0
    assert result.adjoint_applications > 0


def test_exact_spectrum_paths_give_qualitative_certificate_without_rate() -> None:
    """Complete spectral authority certifies paths but not an untraced contour."""
    entries = np.array([1.0, 1.2, 1.4, 1.6])
    operator = _diagonal_operator(entries.tolist())
    result = assess_targeted_pseudospectral_connectivity(
        operator,
        entries,
        spectrum_source="exact test spectrum",
        spectrum_complete=True,
        exact_spectrum_points=True,
        epsilon_zero_lower_bound=1.0,
        arnoldi_steps=operator.n,
        operator_norm_scale=2.0,
        anchor_count=entries.size,
        maximum_sigma_min_evaluations=32,
    )
    assert result.certified
    assert not result.provisional
    assert result.window_nonempty
    assert result.eps_connect_path_upper is not None
    assert result.eps_connect_path_upper < result.epsilon_zero_lower_bound
    assert not result.rate_bound_available
    assert "Trefethen" in result.rate_bound_reason


def test_reduced_ritz_mode_is_explicitly_provisional() -> None:
    """A matrix-free reduced spectrum cannot be promoted to a certificate."""
    entries = np.array([1.0, 1.2, 1.4, 1.6])
    operator = _diagonal_operator(entries.tolist())
    ritz, _, _ = matrix_free_ritz_values(operator, arnoldi_steps=operator.n)
    result = assess_targeted_pseudospectral_connectivity(
        operator,
        ritz,
        spectrum_source="test Ritz values",
        spectrum_complete=False,
        exact_spectrum_points=False,
        arnoldi_steps=operator.n,
        operator_norm_scale=2.0,
        anchor_count=ritz.size,
        maximum_sigma_min_evaluations=32,
    )
    assert not result.certified
    assert result.provisional
    assert result.verdict == "provisional_window"
    assert result.window_nonempty


def test_near_zero_separated_spectrum_is_not_certified() -> None:
    """Connecting a wide normal spectrum engulfs a near-origin eigenvalue."""
    entries = np.array([0.01, 1.0])
    operator = _diagonal_operator(entries.tolist())
    result = assess_targeted_pseudospectral_connectivity(
        operator,
        entries,
        spectrum_source="exact test spectrum",
        spectrum_complete=True,
        exact_spectrum_points=True,
        epsilon_zero_lower_bound=0.01,
        arnoldi_steps=operator.n,
        operator_norm_scale=1.5,
        anchor_count=entries.size,
        maximum_sigma_min_evaluations=16,
    )
    assert not result.certified
    assert result.verdict == "not_certified"
    assert not result.window_nonempty
