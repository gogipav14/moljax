"""Smoke tests for the experimental adaptive inexact-Newton PME harness."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.nonlinear_diffusion import barenblatt
from moljax.experimental.pme_inexact_newton import (
    InexactNewtonConfig,
    PMEBackwardEulerProblem,
    solve_pme_inexact_newton,
)


def _wide_state(grid: NodeCenteredDirichletGrid) -> jax.Array:
    """Return the m=2 wide-front state without importing a benchmark module."""
    return barenblatt(grid.x_coords() / 3.0, 0.1, 2.0, b=1.0)


def test_inexact_newton_adapts_forcing_and_reaches_the_same_root() -> None:
    """Adaptive forcing remains non-vacuous and preconditioner-independent."""
    grid = NodeCenteredDirichletGrid.uniform(64, -4.0, 4.0)
    previous = _wide_state(grid)
    problem = PMEBackwardEulerProblem(previous, grid, 2.0, 2.0e-2, 1.0e-5)
    config = InexactNewtonConfig(
        nonlinear_tolerance=1.0e-8,
        max_newton_iters=20,
        max_gmres_iters=128,
    )
    results = {
        method: solve_pme_inexact_newton(problem, method=method, config=config)
        for method in ("tau_blend", "frozen_mean", "frozen_bulk", "identity")
    }

    reference = results["tau_blend"].solution
    for result in results.values():
        assert result.converged
        assert result.final_residual_norm <= config.nonlinear_tolerance
        assert all(step.forcing_target_met for step in result.steps)
        assert len({round(step.eta, 12) for step in result.steps}) > 1
        assert all(config.eta_min <= step.next_eta <= config.eta_max for step in result.steps)
        assert jnp.linalg.norm(result.solution - reference) <= 1.0e-7 * jnp.linalg.norm(reference)
