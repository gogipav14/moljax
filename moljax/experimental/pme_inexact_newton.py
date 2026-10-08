"""Inexact-Newton measurements for experimental node-centred PME systems.

This module measures a complete backward-Euler nonlinear solve while varying
only the linear preconditioner.  It uses Eisenstat--Walker choice 1 to set a
genuinely adaptive inner tolerance from the mismatch between the previous
linear model and the residual obtained after the accepted Newton step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from math import hypot, sqrt
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np

from moljax._precision import require_x64
from moljax.core.preconditioners import PrecondContext
from moljax.experimental.node_centered import NodeCenteredDirichletGrid
from moljax.experimental.pme_conditioning import (
    make_backward_euler_residual,
    pme_preconditioner_variant,
)
from moljax.experimental.pme_dst_tau_blend import pme_dst_tau_blend_preconditioner

PreconditionerName = Literal["tau_blend", "frozen_mean", "frozen_bulk", "identity"]


@dataclass(frozen=True)
class InexactNewtonConfig:
    """Numerical controls for one backward-Euler inexact-Newton solve."""

    nonlinear_tolerance: float = 1.0e-8
    max_newton_iters: int = 12
    max_gmres_iters: int = 400
    eta_initial: float = 0.5
    eta_min: float = 1.0e-10
    eta_max: float = 0.5
    ew_safeguard_threshold: float = 0.1
    armijo_constant: float = 1.0e-4
    backtrack_factor: float = 0.5
    max_backtracks: int = 10


@dataclass(frozen=True)
class InexactNewtonStep:
    """Observable work and forcing data for one accepted Newton step."""

    iteration: int
    eta: float
    residual_norm_before: float
    gmres_iterations: int
    gmres_info: int
    achieved_linear_relative_residual: float
    forcing_target_met: bool
    step_length: float
    linear_model_residual_norm: float
    residual_norm_after: float
    next_eta: float


@dataclass(frozen=True)
class InexactNewtonResult:
    """Result of one matched-accuracy nonlinear solve."""

    converged: bool
    nonlinear_iterations: int
    total_gmres_iterations: int
    initial_residual_norm: float
    final_residual_norm: float
    solution: jax.Array
    steps: tuple[InexactNewtonStep, ...]


@dataclass(frozen=True)
class PMEBackwardEulerProblem:
    """One fixed node-centred backward-Euler PME problem."""

    previous_state: jax.Array
    grid: NodeCenteredDirichletGrid
    m: float
    dt: float
    epsilon: float
    _residual: Callable[[jax.Array], jax.Array] = field(init=False, repr=False)
    _jacobian_action: Callable[[jax.Array, jax.Array], jax.Array] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        residual = make_backward_euler_residual(
            self.previous_state,
            self.grid,
            self.m,
            self.dt,
            self.epsilon,
        )

        def jacobian_action(state: jax.Array, vector: jax.Array) -> jax.Array:
            return jax.jvp(residual, (state,), (vector,))[1]

        object.__setattr__(self, "_residual", jax.jit(residual))
        object.__setattr__(self, "_jacobian_action", jax.jit(jacobian_action))

    def residual(self, state: jax.Array) -> jax.Array:
        """Evaluate the backward-Euler residual."""
        return self._residual(state)

    def jacobian_action(self, state: jax.Array, vector: jax.Array) -> jax.Array:
        """Apply the exact JAX Jacobian at ``state`` to ``vector``."""
        return self._jacobian_action(state, vector)


def _build_preconditioner(
    state: jax.Array,
    problem: PMEBackwardEulerProblem,
    method: PreconditionerName,
    reference_count: int,
) -> tuple[Any | None, PrecondContext]:
    """Build one state-dependent preconditioner for the current Newton iterate."""
    context = PrecondContext(grid=problem.grid, dt=problem.dt, params={})
    if method == "identity":
        return None, context
    if method == "tau_blend":
        preconditioner = pme_dst_tau_blend_preconditioner(
            state,
            problem.m,
            problem.dt,
            problem.grid,
            problem.epsilon,
            reference_count=reference_count,
        )
        return preconditioner, context
    preconditioner, _ = pme_preconditioner_variant(
        state,
        problem.grid,
        problem.m,
        problem.dt,
        problem.epsilon,
        method,
    )
    return preconditioner, context


def _solve_linear_system(
    problem: PMEBackwardEulerProblem,
    state: jax.Array,
    residual: jax.Array,
    method: PreconditionerName,
    eta: float,
    config: InexactNewtonConfig,
    reference_count: int,
) -> tuple[jax.Array, int, int, float, bool]:
    """Solve one Newton equation by explicit left-preconditioned GMRES.

    The Arnoldi/Givens iteration mirrors the counted Stage-2 implementation,
    but also retains the triangular factor needed to recover the Newton
    update.  A small triangular solve is attempted once the preconditioned
    residual reaches the forcing target; acceptance is based on the true
    unpreconditioned residual ``||J s + F|| / ||F||``.
    """
    preconditioner, context = _build_preconditioner(
        state,
        problem,
        method,
        reference_count,
    )

    def apply_preconditioner(vector: jax.Array) -> jax.Array:
        if preconditioner is None:
            return jnp.asarray(vector, dtype=jnp.float64)
        return preconditioner.apply(vector, context)

    right_hand_side = -jnp.asarray(residual, dtype=jnp.float64)
    preconditioned_rhs = jax.block_until_ready(apply_preconditioner(right_hand_side))
    rhs_norm = float(jnp.linalg.norm(right_hand_side))
    preconditioned_rhs_norm = float(jnp.linalg.norm(preconditioned_rhs))
    if rhs_norm == 0.0:
        return jnp.zeros_like(state), 0, 0, 0.0, True
    if preconditioned_rhs_norm == 0.0:
        return jnp.zeros_like(state), 0, 1, 1.0, False

    basis = [preconditioned_rhs / preconditioned_rhs_norm]
    transformed_columns: list[list[float]] = []
    cosines: list[float] = []
    sines: list[float] = []
    rotated_rhs = [preconditioned_rhs_norm] + [0.0] * config.max_gmres_iters
    breakdown_threshold = float(jnp.sqrt(jnp.finfo(jnp.float64).eps))
    last_update = jnp.zeros_like(state)
    last_true_relative_residual = 1.0

    for column in range(config.max_gmres_iters):
        jacobian_vector = problem.jacobian_action(state, basis[column])
        candidate_vector = jnp.real(apply_preconditioner(jacobian_vector))
        coefficients: list[float] = []
        for basis_vector in basis:
            coefficient = float(jnp.vdot(basis_vector, candidate_vector))
            coefficients.append(coefficient)
            candidate_vector = candidate_vector - coefficient * basis_vector
        for row, basis_vector in enumerate(basis):
            correction = float(jnp.vdot(basis_vector, candidate_vector))
            coefficients[row] = coefficients[row] + correction
            candidate_vector = candidate_vector - correction * basis_vector

        arnoldi_subdiagonal = float(jnp.linalg.norm(candidate_vector))
        hessenberg_column = coefficients + [arnoldi_subdiagonal]
        for row, (cosine, sine) in enumerate(zip(cosines, sines, strict=True)):
            upper = cosine * hessenberg_column[row] + sine * hessenberg_column[row + 1]
            hessenberg_column[row + 1] = (
                -sine * hessenberg_column[row] + cosine * hessenberg_column[row + 1]
            )
            hessenberg_column[row] = upper

        diagonal = hessenberg_column[column]
        subdiagonal = hessenberg_column[column + 1]
        normalization = hypot(diagonal, subdiagonal)
        if normalization <= breakdown_threshold:
            cosine, sine = 1.0, 0.0
        else:
            cosine, sine = diagonal / normalization, subdiagonal / normalization
        cosines.append(cosine)
        sines.append(sine)
        hessenberg_column[column] = normalization
        hessenberg_column[column + 1] = 0.0
        transformed_columns.append(hessenberg_column[: column + 1])

        previous_rhs = rotated_rhs[column]
        rotated_rhs[column] = cosine * previous_rhs
        rotated_rhs[column + 1] = -sine * previous_rhs
        preconditioned_relative_residual = abs(rotated_rhs[column + 1]) / preconditioned_rhs_norm
        should_test_true_residual = (
            preconditioned_relative_residual <= eta
            or arnoldi_subdiagonal <= breakdown_threshold
            or column + 1 == config.max_gmres_iters
        )
        if should_test_true_residual:
            dimension = column + 1
            upper_triangular = np.zeros((dimension, dimension), dtype=np.float64)
            for column_index, values in enumerate(transformed_columns):
                upper_triangular[: column_index + 1, column_index] = values
            coefficients_array, *_ = np.linalg.lstsq(
                upper_triangular,
                np.asarray(rotated_rhs[:dimension], dtype=np.float64),
                rcond=None,
            )
            last_update = sum(
                (
                    float(weight) * basis_vector
                    for weight, basis_vector in zip(
                        coefficients_array,
                        basis[:dimension],
                        strict=True,
                    )
                ),
                start=jnp.zeros_like(state),
            )
            true_linear_residual = problem.jacobian_action(state, last_update) - right_hand_side
            last_true_relative_residual = float(jnp.linalg.norm(true_linear_residual)) / rhs_norm
            if last_true_relative_residual <= eta * (1.0 + 1.0e-10):
                return (
                    jax.block_until_ready(last_update),
                    dimension,
                    0,
                    last_true_relative_residual,
                    True,
                )

        if arnoldi_subdiagonal <= breakdown_threshold:
            return (
                jax.block_until_ready(last_update),
                column + 1,
                1,
                last_true_relative_residual,
                False,
            )
        basis.append(candidate_vector / arnoldi_subdiagonal)

    return (
        jax.block_until_ready(last_update),
        config.max_gmres_iters,
        1,
        last_true_relative_residual,
        False,
    )


def _eisenstat_walker_next_eta(
    eta: float,
    residual_norm_before: float,
    residual_norm_after: float,
    linear_model_residual_norm: float,
    config: InexactNewtonConfig,
) -> float:
    """Return safeguarded Eisenstat--Walker choice 1 for the next step.

    The raw choice compares the achieved nonlinear residual norm with the norm
    predicted by the local linear model.  The power safeguard prevents an
    abrupt oversolve after a poor early model, and the final clamp provides
    deterministic lower and upper forcing limits.
    """
    raw = abs(residual_norm_after - linear_model_residual_norm) / max(
        residual_norm_before,
        np.finfo(np.float64).tiny,
    )
    golden_ratio = 0.5 * (1.0 + sqrt(5.0))
    safeguard = eta**golden_ratio
    if safeguard > config.ew_safeguard_threshold:
        raw = max(raw, safeguard)
    return min(config.eta_max, max(config.eta_min, raw))


def solve_pme_inexact_newton(
    problem: PMEBackwardEulerProblem,
    *,
    method: PreconditionerName,
    config: InexactNewtonConfig | None = None,
    reference_count: int = 3,
    initial_guess: jax.Array | None = None,
) -> InexactNewtonResult:
    """Solve one PME backward-Euler step with adaptive inexact Newton.

    Explicit Arnoldi/Givens GMRES supplies a trustworthy inner-iteration count
    and enforces the true, unpreconditioned linear residual target.  The same
    nonlinear residual, line search, forcing schedule, and stopping test are
    used for all four preconditioner variants.
    """
    require_x64("PME inexact-Newton solve")
    controls = config or InexactNewtonConfig()
    if reference_count < 1:
        raise ValueError("reference_count must be positive")
    if not 0.0 < controls.eta_min <= controls.eta_initial <= controls.eta_max < 1.0:
        raise ValueError("forcing terms must satisfy 0 < eta_min <= eta_initial <= eta_max < 1")
    if controls.nonlinear_tolerance <= 0.0:
        raise ValueError("nonlinear_tolerance must be positive")

    state = jnp.asarray(
        problem.previous_state if initial_guess is None else initial_guess,
        dtype=jnp.float64,
    )
    residual = jax.block_until_ready(problem.residual(state))
    initial_residual_norm = float(jnp.linalg.norm(residual))
    eta = controls.eta_initial
    records: list[InexactNewtonStep] = []

    for iteration in range(controls.max_newton_iters):
        residual_norm_before = float(jnp.linalg.norm(residual))
        if residual_norm_before <= controls.nonlinear_tolerance:
            break

        update, gmres_iterations, info, relative_linear_residual, forcing_met = (
            _solve_linear_system(
                problem,
                state,
                residual,
                method,
                eta,
                controls,
                reference_count,
            )
        )
        if not forcing_met:
            raise RuntimeError(
                "GMRES did not meet the Eisenstat--Walker forcing target: "
                f"method={method}, eta={eta:.3e}, achieved={relative_linear_residual:.3e}, "
                f"info={info}"
            )

        jacobian_update = jax.block_until_ready(problem.jacobian_action(state, update))
        step_length = 1.0
        accepted_state = state + update
        accepted_residual = jax.block_until_ready(problem.residual(accepted_state))
        accepted_norm = float(jnp.linalg.norm(accepted_residual))
        for _ in range(controls.max_backtracks):
            target = (1.0 - controls.armijo_constant * step_length) * residual_norm_before
            if accepted_norm <= target:
                break
            step_length *= controls.backtrack_factor
            accepted_state = state + step_length * update
            accepted_residual = jax.block_until_ready(problem.residual(accepted_state))
            accepted_norm = float(jnp.linalg.norm(accepted_residual))
        else:
            raise RuntimeError(
                f"Newton line search failed for method={method} at iteration={iteration}"
            )

        linear_model_residual = residual + step_length * jacobian_update
        linear_model_residual_norm = float(jnp.linalg.norm(linear_model_residual))
        next_eta = _eisenstat_walker_next_eta(
            eta,
            residual_norm_before,
            accepted_norm,
            linear_model_residual_norm,
            controls,
        )
        records.append(
            InexactNewtonStep(
                iteration=iteration,
                eta=eta,
                residual_norm_before=residual_norm_before,
                gmres_iterations=gmres_iterations,
                gmres_info=info,
                achieved_linear_relative_residual=relative_linear_residual,
                forcing_target_met=forcing_met,
                step_length=step_length,
                linear_model_residual_norm=linear_model_residual_norm,
                residual_norm_after=accepted_norm,
                next_eta=next_eta,
            )
        )
        state = jax.block_until_ready(accepted_state)
        residual = accepted_residual
        eta = next_eta

    final_residual_norm = float(jnp.linalg.norm(residual))
    return InexactNewtonResult(
        converged=final_residual_norm <= controls.nonlinear_tolerance,
        nonlinear_iterations=len(records),
        total_gmres_iterations=sum(record.gmres_iterations for record in records),
        initial_residual_norm=initial_residual_norm,
        final_residual_norm=final_residual_norm,
        solution=state,
        steps=tuple(records),
    )


__all__ = [
    "InexactNewtonConfig",
    "InexactNewtonResult",
    "InexactNewtonStep",
    "PMEBackwardEulerProblem",
    "PreconditionerName",
    "solve_pme_inexact_newton",
]
