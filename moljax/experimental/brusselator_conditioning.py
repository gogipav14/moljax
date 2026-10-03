"""Visited-state conditioning study helpers for the periodic Brusselator.

This experimental module deliberately reuses moljax's shipped Brusselator
factory, periodic FFT diffusion preconditioner, and generic conditioning
toolbox.  It contributes only the study wiring: a two-field backward-Euler
linearization, diagnostic evaluation, and a counted GMRES measurement on the
same left-preconditioned Newton system.
"""

from __future__ import annotations

import hashlib
import types
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import lobpcg_standard
from jax.flatten_util import ravel_pytree

from moljax.conditioning import (
    LinearizedOperator,
    adjoint_identity,
    arnoldi,
    assess_preconditioner,
    epsilon_zero,
    estimate_rates,
    linearized_operator,
    numerical_range,
    ritz_values,
)
from moljax.core.bc import BCType, FieldBCSpec
from moljax.core.fft_solvers import (
    FFTCache2D,
    laplacian_symbol_2d,
    laplacian_symbol_2d_rfft,
)
from moljax.core.grid import Grid2D
from moljax.core.model import (
    MOLModel,
    create_brusselator_model,
    create_brusselator_periodic_fft,
)
from moljax.core.newton_krylov import NKParams, create_implicit_residual
from moljax.core.operators import LinearOp, NonlinearOp, brusselator_reaction_op
from moljax.core.preconditioners import (
    FFTDiffusionPreconditioner,
    IdentityPreconditioner,
    PrecondContext,
    Preconditioner,
    create_fft_preconditioner,
)
from moljax.core.state import StateDict
from moljax.core.stepping import be_step
from moljax.experimental.brusselator_fourier_weyl_ghost_bound import (
    FourierWeylGhostLowerBound,
    fourier_weyl_ghost_lower_bound,
)
from moljax.experimental.pme_conditioning import _counted_gmres

jax.config.update("jax_enable_x64", True)


class BrusselatorRegime(NamedTuple):
    """Physical parameters and reference horizon for one Brusselator regime."""

    name: str
    du: float
    dv: float
    a: float
    b: float
    domain_length: float
    reference_final_time: float


HOPF_REGIME = BrusselatorRegime(
    name="hopf",
    du=0.01,
    dv=0.02,
    a=1.0,
    b=3.4,
    domain_length=5.0,
    reference_final_time=50.0,
)
"""Oscillatory Brusselator regime with ``b > 1 + a**2``."""

TURING_REGIME = BrusselatorRegime(
    name="turing",
    du=0.01,
    dv=0.1,
    a=1.0,
    b=1.8,
    domain_length=5.0,
    reference_final_time=200.0,
)
"""Diffusion-driven Brusselator regime with the paper's ``L=5, t=200`` target."""

REGIMES = {HOPF_REGIME.name: HOPF_REGIME, TURING_REGIME.name: TURING_REGIME}


class BrusselatorLinearization(NamedTuple):
    """A BE linearization and its consistently left-preconditioned RHS."""

    operator: LinearizedOperator
    residual: Any
    preconditioner: Preconditioner
    context: PrecondContext
    rhs: jax.Array


class TrajectorySample(NamedTuple):
    """One converged trajectory state and its distance from the steady state."""

    step: int
    time: float
    state: StateDict
    developedness: dict[str, float]


def _state_identifier(state: StateDict, grid: Grid2D) -> dict[str, Any]:
    """Return a SHA256 identity for the physical fields used by the bound."""
    interior_y, interior_x = grid.interior_slice
    digest = hashlib.sha256()
    shapes: dict[str, list[int]] = {}
    for field in ("u", "v"):
        values = np.asarray(
            jax.device_get(state[field][interior_y, interior_x]), dtype=np.float64
        )
        shapes[field] = list(values.shape)
        digest.update(field.encode("utf-8"))
        digest.update(values.dtype.str.encode("utf-8"))
        digest.update(values.tobytes(order="C"))
    return {"sha256": digest.hexdigest(), "interior_shapes": shapes, "dtype": "<f8"}


def _state_layout_error(state: Any, grid: Grid2D) -> str | None:
    """Return why ``state`` is not a padded two-field float64 state, or ``None``.

    The linearization differentiates the residual at the full padded arrays,
    so the operator has one component for each array entry.  The bound's
    ghost multiplicities {0, 1, 3} hold only when each field has exactly one
    ghost layer around the grid interior.  A larger array passes an interior
    slice check, but the boundary copies then reach more entries.
    """
    if not isinstance(state, Mapping) or set(state) != {"u", "v"}:
        return "the state must contain exactly the fields u and v"
    expected = (grid.ny + 2 * grid.n_ghost, grid.nx + 2 * grid.n_ghost)
    for field in ("u", "v"):
        values = state[field]
        shape = tuple(getattr(values, "shape", ()))
        if shape != expected:
            return (
                f"the state field {field} has shape {shape}, but the grid requires "
                f"{expected} (interior plus {grid.n_ghost} ghost layer(s))"
            )
        dtype = getattr(values, "dtype", None)
        if dtype is None or np.dtype(dtype) != np.dtype(np.float64):
            return f"the state field {field} has dtype {dtype}, but float64 is required"
    return None


class CertificateNotApplicable(ValueError):
    """The analyzed operator lies outside the Fourier--Weyl--ghost assumptions."""


def _same_value(first: Any, second: float) -> bool:
    """Return whether a model parameter equals a certificate input (finite, close)."""
    try:
        value = float(first)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(value)) and bool(
        np.isclose(value, float(second), rtol=1.0e-12, atol=0.0)
    )


_SYMBOL_RTOL = 1.0e-12
"""Per-entry relative tolerance for the FFT preconditioner symbol checks."""


def _same_function(candidate: Callable[..., Any], shipped: Callable[..., Any]) -> bool:
    """Return whether ``candidate`` runs exactly the shipped function's code.

    The shipped Brusselator actions are nested functions, so every factory
    call creates a new function object.  All of them share one code object,
    one globals dictionary, and no closure, defaults, or keyword defaults.  A
    replacement callable cannot satisfy this comparison by name or signature
    alone: it needs the identical code object.
    """
    if type(candidate) is not types.FunctionType or type(shipped) is not types.FunctionType:
        return False
    if candidate.__code__ is not shipped.__code__:
        return False
    if candidate.__globals__ is not shipped.__globals__:
        return False
    if candidate.__defaults__ != shipped.__defaults__:
        return False
    if candidate.__kwdefaults__ != shipped.__kwdefaults__:
        return False
    candidate_cells = candidate.__closure__ or ()
    shipped_cells = shipped.__closure__ or ()
    return len(candidate_cells) == len(shipped_cells) and all(
        first.cell_contents is second.cell_contents
        for first, second in zip(candidate_cells, shipped_cells, strict=True)
    )


_MODEL_ACTIONS = ("rhs", "apply_bcs", "linear_rhs", "nonlinear_rhs")
"""The ``MOLModel`` methods that the backward-Euler residual calls.

``create_implicit_residual`` calls ``model.rhs``, which calls ``apply_bcs``,
``linear_rhs``, and ``nonlinear_rhs``.  The linearization differentiates that
residual, so a replacement of any of these methods changes the operator.
"""

_SHIPPED_MODEL_ACTIONS = {name: MOLModel.__dict__[name] for name in _MODEL_ACTIONS}
"""The class functions of ``MOLModel`` as defined when this module was imported."""


def _validate_shipped_model_methods(model: MOLModel) -> None:
    """Refuse a model whose residual methods are not the shipped ``MOLModel`` code.

    The class check alone does not cover an instance attribute that shadows a
    method (``object.__setattr__`` writes through the frozen dataclass), nor a
    replacement of the method on the class itself.  Each method that the
    residual calls must resolve to a bound method of this model whose function
    is the shipped class function.
    """
    instance_attributes = getattr(model, "__dict__", {})
    for name in _MODEL_ACTIONS:
        shipped = _SHIPPED_MODEL_ACTIONS[name]
        if name in instance_attributes:
            raise CertificateNotApplicable(
                f"the model has an instance-level replacement of MOLModel.{name}"
            )
        if MOLModel.__dict__.get(name) is not shipped:
            raise CertificateNotApplicable(
                f"MOLModel.{name} is not the shipped class function"
            )
        bound = getattr(model, name)
        if (
            type(bound) is not types.MethodType
            or bound.__func__ is not shipped
            or bound.__self__ is not model
        ):
            raise CertificateNotApplicable(
                f"model.{name} does not resolve to the shipped MOLModel.{name}"
            )


def _validate_shipped_brusselator_actions(model: MOLModel, grid: Grid2D) -> None:
    """Refuse any model whose operator actions are not the shipped Brusselator code.

    Names are labels only: a replacement action with the shipped name and
    parameters changes the Jacobian that the bound describes.  The model, the
    operator wrappers, and the boundary specifications must also be the exact
    shipped classes, because a subclass can override the action they apply.
    """
    if type(model) is not MOLModel:
        raise CertificateNotApplicable(
            f"the bound covers only the shipped MOLModel class, got {type(model).__name__}"
        )
    _validate_shipped_model_methods(model)
    if any(type(spec) is not FieldBCSpec for spec in model.bc_spec.values()):
        raise CertificateNotApplicable(
            "the bound requires the shipped FieldBCSpec boundary specification"
        )
    linear_ops = tuple(model.linear_ops)
    nonlinear_ops = tuple(model.nonlinear_ops)
    if (
        len(linear_ops) != 1
        or len(nonlinear_ops) != 1
        or type(linear_ops[0]) is not LinearOp
        or type(nonlinear_ops[0]) is not NonlinearOp
        or linear_ops[0].name != "brusselator_diffusion"
        or nonlinear_ops[0].name != "brusselator_reaction"
    ):
        raise CertificateNotApplicable(
            "the bound covers only the shipped Brusselator diffusion and reaction operators"
        )
    shipped_diffusion = create_brusselator_model(grid).linear_ops[0].apply
    shipped_reaction = brusselator_reaction_op().apply
    if not _same_function(linear_ops[0].apply, shipped_diffusion):
        raise CertificateNotApplicable(
            "the diffusion action is not the shipped Brusselator diffusion implementation"
        )
    if not _same_function(nonlinear_ops[0].apply, shipped_reaction):
        raise CertificateNotApplicable(
            "the reaction action is not the shipped Brusselator reaction implementation"
        )


def _validate_fft_symbol(
    cache: Any,
    grid: Grid2D,
    dt: float,
    diffusivities: tuple[float, float],
) -> None:
    """Check the FFT preconditioner symbol entry by entry, or refuse.

    The bound assumes ``P_k = diag(1 - dt D_u l_k, 1 - dt D_v l_k)`` on the
    grid's own finite-difference symbol ``l_k``, with ``l_0 = 0`` exactly
    (so that ``||P^-1||_2 = 1``).  A tolerance scaled by the largest symbol
    entry admits large errors in the small entries, the zero mode included.
    Each entry of ``l`` and of ``1 - dt D l`` is therefore compared with a
    tolerance relative to that entry's own expected value.
    """
    if type(cache) is not FFTCache2D:
        raise CertificateNotApplicable(
            f"the bound covers only the shipped FFTCache2D, got {type(cache).__name__}"
        )
    use_rfft = cache.use_rfft
    if type(use_rfft) is not bool:
        raise CertificateNotApplicable("the FFT cache use_rfft flag is not a bool")
    symbol_builder = laplacian_symbol_2d_rfft if use_rfft else laplacian_symbol_2d
    expected_symbol = np.asarray(
        symbol_builder(grid.ny, grid.nx, grid.dy, grid.dx, jnp.float64), dtype=np.float64
    )
    actual_symbol = np.asarray(jax.device_get(cache.laplacian_symbol))
    if actual_symbol.shape != expected_symbol.shape:
        raise CertificateNotApplicable(
            "the FFT preconditioner symbol shape does not match the grid"
        )
    if not np.all(np.isfinite(actual_symbol)) or np.iscomplexobj(actual_symbol):
        raise CertificateNotApplicable("the FFT preconditioner symbol is not finite and real")
    actual_symbol = actual_symbol.astype(np.float64)
    if actual_symbol[0, 0] != 0.0:
        raise CertificateNotApplicable(
            "the FFT preconditioner symbol zero mode is not exactly zero"
        )
    if not np.all(
        np.abs(actual_symbol - expected_symbol) <= _SYMBOL_RTOL * np.abs(expected_symbol)
    ):
        raise CertificateNotApplicable(
            "the FFT preconditioner symbol is not the grid's finite-difference Laplacian symbol"
        )
    for diffusivity in diffusivities:
        expected_denominator = 1.0 - dt * diffusivity * expected_symbol
        actual_denominator = 1.0 - dt * diffusivity * actual_symbol
        if not np.all(
            np.abs(actual_denominator - expected_denominator)
            <= _SYMBOL_RTOL * np.abs(expected_denominator)
        ):
            raise CertificateNotApplicable(
                "the FFT preconditioner denominator 1 - dt D l differs from the grid's"
            )


def _validate_certificate_operator(
    model: MOLModel,
    regime: BrusselatorRegime,
    dt: float,
    preconditioner: Preconditioner,
    context: PrecondContext,
) -> Grid2D:
    """Check every operator assumption of the bound, or raise ``CertificateNotApplicable``.

    The Fourier--Weyl--ghost bound is derived for one operator family only:
    the shipped two-field Brusselator (5-point periodic Laplacian diffusion
    plus Brusselator kinetics) on a ``Grid2D`` that is periodic in both axes
    with exactly one ghost layer (the ghost multiplicities in {0, 1, 3}),
    linearized for backward Euler at ``dt``, and left-preconditioned either by
    the identity or by the FFT diffusion preconditioner with symbol
    ``P_k = diag(1 - dt D_u l_k, 1 - dt D_v l_k)`` on the grid's own
    finite-difference Laplacian symbol ``l_k``.  The identity is covered
    because every ``P_k >= I`` (``l_k <= 0``), so the interior bound for
    ``P^-1 J`` is also a lower bound for ``J``; the ghost block ``[C, I]`` is
    the same for both.  Anything else is refused rather than certified.
    """
    grid = model.grid
    if type(grid) is not Grid2D:
        raise CertificateNotApplicable("the bound requires a two-dimensional Grid2D")
    if grid.n_ghost != 1:
        raise CertificateNotApplicable(
            f"the bound's ghost multiplicities assume n_ghost == 1, got {grid.n_ghost}"
        )
    if grid.nx < 2 or grid.ny < 2 or not (grid.dx > 0.0 and grid.dy > 0.0):
        raise CertificateNotApplicable("the bound requires nx, ny >= 2 and positive spacings")
    if set(model.bc_spec) != {"u", "v"} or any(
        spec.kind != BCType.PERIODIC for spec in model.bc_spec.values()
    ):
        raise CertificateNotApplicable(
            "the bound requires periodic boundary conditions on exactly the fields u and v"
        )
    _validate_shipped_brusselator_actions(model, grid)
    params = model.params
    for key, expected in (
        ("Du", regime.du),
        ("Dv", regime.dv),
        ("a", regime.a),
        ("b", regime.b),
    ):
        if not _same_value(params.get(key), expected):
            raise CertificateNotApplicable(
                f"model parameter {key}={params.get(key)!r} does not match the "
                f"certificate input {expected!r}"
            )
    if not (np.isfinite(dt) and dt > 0.0) or not _same_value(context.dt, dt):
        raise CertificateNotApplicable(
            f"the linearization dt={context.dt!r} does not match the certificate dt={dt!r}"
        )
    if context.grid != grid:
        raise CertificateNotApplicable("the preconditioner context grid is not the model grid")
    for key in ("Du", "Dv"):
        if not _same_value(context.params.get(key), float(params[key])):
            raise CertificateNotApplicable(
                f"the preconditioner context {key} does not match the model"
            )
    # Exact classes only: a subclass can override ``apply`` (for example a
    # scaled identity), and then ``||P^-1||_2 = 1`` no longer holds.  The
    # same holds for an instance attribute that shadows the class method.
    if type(preconditioner) not in (IdentityPreconditioner, FFTDiffusionPreconditioner) or (
        "apply" in getattr(preconditioner, "__dict__", {})
    ):
        raise CertificateNotApplicable(
            f"the bound does not cover the {type(preconditioner).__name__} preconditioner"
        )
    if type(preconditioner) is IdentityPreconditioner:
        return grid
    if dict(preconditioner.field_diffusivity_keys or {}) != {"u": "Du", "v": "Dv"}:
        raise CertificateNotApplicable(
            "the FFT preconditioner must map u to Du and v to Dv"
        )
    _validate_fft_symbol(
        preconditioner.fft_cache,
        grid,
        float(dt),
        (float(params["Du"]), float(params["Dv"])),
    )
    return grid


def _fourier_weyl_bound(
    state: StateDict,
    model: MOLModel,
    regime: BrusselatorRegime,
    dt: float,
    *,
    preconditioner: Preconditioner,
    context: PrecondContext,
) -> FourierWeylGhostLowerBound:
    """Evaluate the closed-form padded-operator lower bound on one state.

    Raises ``CertificateNotApplicable`` when the operator is outside the
    bound's assumptions.  Grid sizes and domain lengths come from the model's
    grid, never from the regime.
    """
    grid = _validate_certificate_operator(model, regime, dt, preconditioner, context)
    layout_error = _state_layout_error(state, grid)
    if layout_error is not None:
        raise CertificateNotApplicable(layout_error)
    interior_y, interior_x = grid.interior_slice
    u = np.asarray(jax.device_get(state["u"][interior_y, interior_x]), dtype=np.float64)
    v = np.asarray(jax.device_get(state["v"][interior_y, interior_x]), dtype=np.float64)
    if u.shape != (grid.ny, grid.nx) or v.shape != (grid.ny, grid.nx):
        raise CertificateNotApplicable("the state interior does not match the model grid")
    return fourier_weyl_ghost_lower_bound(
        u,
        v,
        du=regime.du,
        dv=regime.dv,
        a=regime.a,
        beta=regime.b,
        dt=dt,
        domain_length_x=grid.x_max - grid.x_min,
        domain_length_y=grid.y_max - grid.y_min,
    )


def _weak_bound_override_eligible(assessment: Any) -> bool:
    """Return whether a weak certificate may refine an otherwise usable reading.

    ``assess_preconditioner`` uses ``indeterminate`` for incomplete or invalid
    diagnostics (including a short or non-finite Ritz spectrum).  A valid but
    sub-threshold full-operator bound cannot erase that abstention.  It only
    refines an ordinary investigate/provisional reading whose outlier count was
    actually measured, and only when epsilon zero is the sole failed gate:
    supports consistent, disk rate at most the assessment's rate threshold, and
    right-real outliers at most its allowed number.  A disk-rate or outlier
    ``investigate`` is a measured caution that a weak bound cannot remove.
    The thresholds are the ones ``assess_preconditioner`` applied, read back
    from the assessment.  The caller checks origin enclosure.
    """
    if assessment.verdict not in {"investigate", "provisional"}:
        return False
    if assessment.n_right_real_outliers is None:
        return False
    return (
        bool(assessment.supports_consistent)
        and float(assessment.disk_rate) <= float(assessment.rate_threshold)
        and int(assessment.n_right_real_outliers) <= int(assessment.max_right_real_outliers)
    )


def _lobpcg_sigma_min_upper_estimate(
    operator: LinearizedOperator,
    seed: int,
    *,
    max_iters: int = 12,
) -> float | None:
    """Return a non-certifying LOBPCG upper estimate for ``sigma_min(A)``.

    The largest Ritz value of ``-A^*A`` is no larger than its true largest
    eigenvalue.  Negating it therefore estimates ``sigma_min(A)`` from above.
    It is deliberately serialized only as a diagnostic and is never supplied
    to ``assess_preconditioner`` as lower-bound evidence.
    """
    if 2 * operator.n <= 15:
        return None

    def negative_realified_normal(value: jax.Array) -> jax.Array:
        def apply_vector(column: jax.Array) -> jax.Array:
            complex_column = column[: operator.n] + 1j * column[operator.n :]
            normal_image = operator.matvec_adjoint(operator.matvec(complex_column))
            return -jnp.concatenate((jnp.real(normal_image), jnp.imag(normal_image)))

        if value.ndim == 1:
            return apply_vector(value)
        return jax.vmap(apply_vector, in_axes=1, out_axes=1)(value)

    initial = jax.random.normal(
        jax.random.PRNGKey(seed + 17), (2 * operator.n, 3), dtype=jnp.float64
    )
    try:
        eigenvalues, _, _ = lobpcg_standard(
            negative_realified_normal, initial, m=max_iters, tol=1.0e-5
        )
    except (RuntimeError, ValueError):
        return None
    estimate_squared = max(0.0, -float(eigenvalues[0]))
    return float(np.sqrt(estimate_squared))


def resolve_regime(regime: str | BrusselatorRegime) -> BrusselatorRegime:
    """Return a named standard regime or pass through an explicit parameter set."""
    if isinstance(regime, BrusselatorRegime):
        return regime
    try:
        return REGIMES[regime]
    except KeyError as error:
        choices = ", ".join(sorted(REGIMES))
        raise ValueError(
            f"Unknown Brusselator regime {regime!r}; choose one of {choices}"
        ) from error


def build_brusselator_system(
    regime: str | BrusselatorRegime,
    grid: Grid2D,
) -> tuple[MOLModel, Any, dict[str, float]]:
    """Build the shipped periodic-FFT Brusselator model for one study regime."""
    selected = resolve_regime(regime)
    return create_brusselator_periodic_fft(
        grid,
        Du=selected.du,
        Dv=selected.dv,
        a=selected.a,
        b=selected.b,
        dtype=jnp.float64,
    )


def _ready_state(state: StateDict) -> StateDict:
    """Synchronize every array in a PyTree state before returning it."""
    return jax.tree_util.tree_map(jax.block_until_ready, state)


def _initial_state(
    model: MOLModel,
    regime: BrusselatorRegime,
    perturbation: float,
    seed: int,
) -> StateDict:
    """Return a small, reproducible perturbation of the homogeneous steady state."""
    if perturbation <= 0.0:
        raise ValueError("perturbation must be positive so the analysed trajectory is dynamical")
    grid = model.grid
    if not isinstance(grid, Grid2D):
        raise TypeError("The Brusselator study requires a two-dimensional grid")
    key_u, key_v = jax.random.split(jax.random.PRNGKey(seed))
    shape = (grid.ny_total, grid.nx_total)
    u = jnp.full(shape, regime.a, dtype=jnp.float64)
    v = jnp.full(shape, regime.b / regime.a, dtype=jnp.float64)
    u = u + perturbation * jax.random.normal(key_u, shape, dtype=jnp.float64)
    v = v + perturbation * jax.random.normal(key_v, shape, dtype=jnp.float64)
    return model.apply_bcs({"u": u, "v": v}, 0.0)


def _integrate_visited_states(
    regime: BrusselatorRegime,
    model: MOLModel,
    fft_cache: Any,
    *,
    n_steps: int,
    dt: float,
    perturbation: float,
    seed: int,
    nk_params: NKParams,
) -> list[StateDict]:
    """Advance a perturbed state by converged FFT-preconditioned BE steps."""
    if n_steps < 1:
        raise ValueError("n_steps must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    state = _initial_state(model, regime, perturbation, seed)
    preconditioner = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    visited: list[StateDict] = []
    time_value = 0.0
    for _ in range(n_steps):
        state, stats = be_step(
            model,
            state,
            time_value,
            dt,
            preconditioner=preconditioner,
            nk_params=nk_params,
        )
        state = _ready_state(state)
        if not bool(stats.converged):
            raise RuntimeError(
                "The FFT-preconditioned backward-Euler step did not converge "
                f"at t={time_value + dt:g}; no nonconverged iterate is analysed."
            )
        visited.append(state)
        time_value += dt
    return visited


def visited_states(
    regime: str | BrusselatorRegime,
    *,
    grid: Grid2D,
    n_steps: int,
    dt: float,
    perturbation: float,
    seed: int,
) -> list[StateDict]:
    """Return genuinely visited states from FFT-preconditioned BE integration.

    The starting state is the homogeneous Brusselator steady state
    ``(a, b / a)`` plus a small deterministic random perturbation.  Each
    returned state is the converged result of one shipped
    :func:`moljax.core.stepping.be_step`, not an analytic fixed point.
    """
    selected = resolve_regime(regime)
    model, fft_cache, _ = build_brusselator_system(selected, grid)
    return _integrate_visited_states(
        selected,
        model,
        fft_cache,
        n_steps=n_steps,
        dt=dt,
        perturbation=perturbation,
        seed=seed,
        nk_params=NKParams(
            max_newton_iters=10,
            max_krylov_iters=50,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        ),
    )


def state_developedness(
    state: StateDict,
    grid: Grid2D,
    regime: str | BrusselatorRegime,
) -> dict[str, float]:
    """Measure interior departure from the homogeneous Brusselator steady state."""
    selected = resolve_regime(regime)
    slice_y, slice_x = grid.interior_slice
    u_interior = jnp.asarray(state["u"], dtype=jnp.float64)[slice_y, slice_x]
    v_interior = jnp.asarray(state["v"], dtype=jnp.float64)[slice_y, slice_x]
    return {
        "max_abs_u_minus_steady": float(jnp.max(jnp.abs(u_interior - selected.a))),
        "max_abs_v_minus_steady": float(jnp.max(jnp.abs(v_interior - selected.b / selected.a))),
    }


def _sampled_visited_states(
    regime: BrusselatorRegime,
    model: MOLModel,
    fft_cache: Any,
    *,
    sample_steps: tuple[int, ...],
    dt: float,
    perturbation: float,
    seed: int,
    nk_params: NKParams,
) -> list[TrajectorySample]:
    """Advance BE steps and retain only requested, converged trajectory states."""
    if not sample_steps:
        raise ValueError("sample_steps must be nonempty")
    if any(step < 1 for step in sample_steps):
        raise ValueError("sample_steps must contain positive step numbers")
    if tuple(sorted(set(sample_steps))) != sample_steps:
        raise ValueError("sample_steps must be sorted and contain no duplicates")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not isinstance(model.grid, Grid2D):
        raise TypeError("The Brusselator study requires a two-dimensional grid")

    state = _initial_state(model, regime, perturbation, seed)
    preconditioner = create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    requested = set(sample_steps)
    samples: list[TrajectorySample] = []
    time_value = 0.0
    for step in range(1, sample_steps[-1] + 1):
        state, stats = be_step(
            model,
            state,
            time_value,
            dt,
            preconditioner=preconditioner,
            nk_params=nk_params,
        )
        state = _ready_state(state)
        time_value += dt
        if not bool(stats.converged):
            raise RuntimeError(
                "The FFT-preconditioned backward-Euler step did not converge "
                f"at t={time_value:g}; no nonconverged iterate is analysed."
            )
        if step in requested:
            samples.append(
                TrajectorySample(
                    step=step,
                    time=time_value,
                    state=state,
                    developedness=state_developedness(state, model.grid, regime),
                )
            )
    return samples


def sampled_visited_states(
    regime: str | BrusselatorRegime,
    *,
    grid: Grid2D,
    sample_steps: tuple[int, ...],
    dt: float,
    perturbation: float,
    seed: int,
    nk_params: NKParams | None = None,
) -> list[TrajectorySample]:
    """Return selected converged states along a developed BE trajectory.

    This retains only the requested samples while integrating every preceding
    step.  The per-sample ``developedness`` metric is the maximum interior
    departure from ``(a, b/a)``, ensuring that a diagnostic can distinguish a
    developed state from a nearly unchanged perturbation.
    """
    selected = resolve_regime(regime)
    model, fft_cache, _ = build_brusselator_system(selected, grid)
    if nk_params is None:
        nk_params = NKParams(
            max_newton_iters=15,
            max_krylov_iters=100,
            newton_tol=1.0e-8,
            krylov_tol=1.0e-8,
        )
    return _sampled_visited_states(
        selected,
        model,
        fft_cache,
        sample_steps=sample_steps,
        dt=dt,
        perturbation=perturbation,
        seed=seed,
        nk_params=nk_params,
    )


def _preconditioner(
    kind: str,
    fft_cache: Any,
) -> FFTDiffusionPreconditioner | IdentityPreconditioner:
    """Return the requested baseline or shipped diffusion preconditioner."""
    if kind == "fft_diffusion":
        return create_fft_preconditioner({"u": "Du", "v": "Dv"}, fft_cache)
    if kind == "identity":
        return IdentityPreconditioner()
    raise ValueError("preconditioner_kind must be 'fft_diffusion' or 'identity'")


def build_brusselator_linearization(
    state: StateDict,
    model: MOLModel,
    fft_cache: Any,
    diffusivities: Mapping[str, float],
    dt: float,
    *,
    time_value: float = 0.0,
    preconditioner_kind: str = "fft_diffusion",
) -> BrusselatorLinearization:
    """Build ``P^-1 J`` and ``P^-1(-R)`` for the next BE Brusselator solve.

    The state remains a two-field ``StateDict`` through the public residual,
    JVP/VJP, and preconditioner actions.  The shared
    :func:`moljax.conditioning.linearized_operator` performs the only
    flattening, at the conditioning-toolbox boundary.
    """
    if set(diffusivities) != {"u", "v"}:
        raise ValueError("Brusselator FFT diffusivities must contain exactly 'u' and 'v'")
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if not isinstance(model.grid, Grid2D):
        raise TypeError("The Brusselator conditioning study requires a two-dimensional grid")
    layout_error = _state_layout_error(state, model.grid)
    if layout_error is not None:
        raise ValueError(layout_error)
    residual = create_implicit_residual(model, state, time_value + dt, dt, method="be")
    preconditioner = _preconditioner(preconditioner_kind, fft_cache)
    context = PrecondContext(grid=model.grid, dt=dt, params=model.params)
    operator = linearized_operator(
        residual,
        state,
        preconditioner=preconditioner,
        context=context,
    )
    negative_residual = jax.tree_util.tree_map(lambda value: -value, residual(state))
    rhs_state = preconditioner.apply(negative_residual, context)
    rhs, _ = ravel_pytree(rhs_state)
    return BrusselatorLinearization(operator, residual, preconditioner, context, rhs)


def assess_brusselator_state(
    state: StateDict,
    model: MOLModel,
    fft_cache: Any,
    diffusivities: Mapping[str, float],
    dt: float,
    regime_params: str | BrusselatorRegime,
    *,
    preconditioner_kind: str = "fft_diffusion",
    time_value: float = 0.0,
    n_angles: int = 4,
    fov_max_iters: int = 8,
    fov_residual_tolerance: float = 1.0e-3,
    fov_n_restarts: int = 2,
    arnoldi_steps: int = 6,
    compute_lobpcg_upper_estimate: bool = False,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Apply the generic diagnostics to one visited two-field Brusselator state.

    The adjoint identity is a required gate for numerical-range diagnostics.
    If it fails, the result is explicitly flagged and no field-of-values
    verdict is inferred.
    """
    selected = resolve_regime(regime_params)
    linearization = build_brusselator_linearization(
        state,
        model,
        fft_cache,
        diffusivities,
        dt,
        time_value=time_value,
        preconditioner_kind=preconditioner_kind,
    )
    operator = linearization.operator
    adjoint_error = adjoint_identity(operator, jax.random.PRNGKey(seed), operator.n)
    common = {
        "regime": selected.name,
        "preconditioner": preconditioner_kind,
        "operator_dimension": operator.n,
        "adjoint_error": float(adjoint_error),
        "adjoint_tolerance": 1.0e-8,
    }
    if adjoint_error > 1.0e-8:
        return {
            **common,
            "status": "adjoint_failed",
            "verdict": "skipped",
            "disk_rate": None,
            "epsilon_zero": None,
            "reduced_arnoldi_epsilon_zero": None,
            "epsilon_zero_full_operator_evidence": False,
            "predicted_gmres_factor": None,
            "origin_enclosed": None,
            "n_right_real_outliers": None,
            "supports_consistent": None,
            "corroboration_attempted": None,
            "verdict_reason": "adjoint identity gate failed",
            "fov_imaginary_extent": None,
            "lobpcg_sigma_min_upper_estimate": None,
            "fourier_weyl_ghost_lower_bound": None,
        }

    if not isinstance(model.grid, Grid2D):
        raise TypeError("The Brusselator conditioning study requires a two-dimensional grid")

    key_real, key_imag = jax.random.split(jax.random.PRNGKey(seed + 1))
    start = jax.random.normal(key_real, (operator.n,), dtype=jnp.float64)
    start = start + 1j * jax.random.normal(key_imag, (operator.n,), dtype=jnp.float64)
    arnoldi_result = arnoldi(operator.matvec, start, min(arnoldi_steps, operator.n))
    ritz = ritz_values(arnoldi_result.hessenberg)
    reduced_epsilon_at_zero = epsilon_zero(arnoldi_result.hessenberg)
    field_of_values = numerical_range(
        operator.matvec,
        operator.matvec_adjoint,
        operator.n,
        n_angles=n_angles,
        max_iters=fov_max_iters,
        residual_tolerance=fov_residual_tolerance,
        n_restarts=fov_n_restarts,
    )
    rates = estimate_rates(field_of_values, ritz)
    try:
        lower_bound = _fourier_weyl_bound(
            state,
            model,
            selected,
            dt,
            preconditioner=linearization.preconditioner,
            context=linearization.context,
        )
        refusal_reason = None
    except CertificateNotApplicable as refusal:
        lower_bound = None
        refusal_reason = str(refusal)
    selected_bound = None if lower_bound is None else lower_bound.selected
    # A valid lower bound below the 0.1 adequacy gate cannot be allowed to
    # turn a reduced-Arnoldi provisional reading into ``investigate``.  In
    # that case retain the original, coverage-qualified assessment and record
    # that the attempted certificate was insufficient.  A bound that clears
    # the gate is full-operator evidence and unlocks adequate when every other
    # fail-closed gate passes.  An operator outside the bound's assumptions
    # gets no certificate at all: the reduced-Arnoldi assessment stands
    # unrefined, exactly as if no bound had been attempted.
    if selected_bound is None:
        epsilon_for_assessment = reduced_epsilon_at_zero
        assessment = assess_preconditioner(
            field_of_values,
            ritz,
            epsilon_for_assessment,
            coverage=arnoldi_result,
        )
        bound_status = "not_applicable"
    elif selected_bound.full_lower_bound >= 0.1:
        epsilon_for_assessment = selected_bound.full_lower_bound
        assessment = assess_preconditioner(
            field_of_values,
            ritz,
            epsilon_for_assessment,
            coverage=arnoldi_result,
            full_operator_lower_bound=True,
        )
        bound_status = "clears_adequacy_gate"
    else:
        epsilon_for_assessment = reduced_epsilon_at_zero
        assessment = assess_preconditioner(
            field_of_values,
            ritz,
            epsilon_for_assessment,
            coverage=arnoldi_result,
        )
        bound_status = "valid_but_below_adequacy_gate"
    verdict = assessment.verdict
    verdict_reason = assessment.verdict_reason
    if (
        bound_status == "valid_but_below_adequacy_gate"
        and _weak_bound_override_eligible(assessment)
        and assessment.supports_consistent
        and not field_of_values.origin_enclosed
    ):
        # A valid full-operator lower bound below the adequacy threshold is
        # evidence of neither adequacy nor inadequacy.  When epsilon zero is
        # the only failed gate, the reading stays provisional.  A disk-rate or
        # outlier ``investigate`` is not eligible and keeps its verdict.  The
        # origin-enclosure and support-consistency gates retain their normal
        # fail-closed precedence.
        verdict = "provisional"
        verdict_reason = "certification not established by the methods attempted"
    lobpcg_upper_estimate = (
        _lobpcg_sigma_min_upper_estimate(operator, seed)
        if compute_lobpcg_upper_estimate
        else None
    )
    jax.block_until_ready(field_of_values.boundary)
    if selected_bound is None:
        certificate = {
            "evidence": "fourier_weyl_ghost_lower_bound",
            "status": bound_status,
            "reason": refusal_reason,
            "selected_k0": None,
            "k0_feature_center": None,
            "b0": None,
            "perturbation_norm": None,
            "interior_lower_bound": None,
            "c": None,
            "full_lower_bound": None,
            "padding": None,
            "floating_point_standard": None,
        }
    else:
        certificate = {
            "evidence": "fourier_weyl_ghost_lower_bound",
            "status": bound_status,
            "selected_k0": selected_bound.name,
            "k0_feature_center": list(selected_bound.center),
            "b0": selected_bound.b0,
            "perturbation_norm": selected_bound.perturbation_norm,
            "interior_lower_bound": selected_bound.interior_lower_bound,
            "c": selected_bound.ghost_norm_bound,
            "full_lower_bound": selected_bound.full_lower_bound,
            "padding": lower_bound.padding,
            "floating_point_standard": lower_bound.floating_point_standard,
        }
    certificate["state_identifier"] = _state_identifier(state, model.grid)

    return {
        **common,
        "status": "completed",
        "verdict": verdict,
        "disk_rate": float(assessment.disk_rate),
        "epsilon_zero": float(assessment.epsilon_zero),
        "reduced_arnoldi_epsilon_zero": float(reduced_epsilon_at_zero),
        "epsilon_zero_full_operator_evidence": bool(
            assessment.epsilon_zero_full_operator_evidence
        ),
        "predicted_gmres_factor": (
            None
            if assessment.predicted_gmres_factor is None
            else float(assessment.predicted_gmres_factor)
        ),
        "origin_enclosed": bool(field_of_values.origin_enclosed),
        "n_right_real_outliers": (
            None
            if assessment.n_right_real_outliers is None
            else int(assessment.n_right_real_outliers)
        ),
        "supports_consistent": bool(assessment.supports_consistent),
        "supports_converged": bool(field_of_values.supports_converged),
        "supports_corroborated": bool(field_of_values.supports_corroborated),
        "corroboration_attempted": bool(assessment.corroboration_attempted),
        "verdict_reason": verdict_reason,
        "fov_imaginary_extent": float(jnp.max(jnp.abs(jnp.imag(field_of_values.boundary)))),
        "rates": rates._asdict(),
        "lobpcg_sigma_min_upper_estimate": lobpcg_upper_estimate,
        "fourier_weyl_ghost_lower_bound": certificate,
    }


def measure_brusselator_gmres(
    state: StateDict,
    model: MOLModel,
    fft_cache: Any,
    diffusivities: Mapping[str, float],
    dt: float,
    regime_params: str | BrusselatorRegime,
    *,
    tol: float,
    max_iters: int,
    time_value: float = 0.0,
    preconditioner_kind: str = "fft_diffusion",
) -> dict[str, Any]:
    """Measure true GMRES work on the same ``P^-1 J`` system as the diagnostics.

    This directly reuses the experimental residual-history GMRES loop from
    the nonlinear-diffusion study.  Only the state PyTree and its flattening
    differ; those are handled by the shared linearization adapter.
    """
    selected = resolve_regime(regime_params)
    linearization = build_brusselator_linearization(
        state,
        model,
        fft_cache,
        diffusivities,
        dt,
        time_value=time_value,
        preconditioner_kind=preconditioner_kind,
    )
    measurement = _counted_gmres(
        linearization.operator.matvec,
        linearization.rhs,
        tol=tol,
        max_iters=max_iters,
    )
    return {
        **measurement,
        "regime": selected.name,
        "preconditioner": preconditioner_kind,
        "operator_dimension": linearization.operator.n,
    }


__all__ = [
    "BrusselatorLinearization",
    "BrusselatorRegime",
    "CertificateNotApplicable",
    "HOPF_REGIME",
    "REGIMES",
    "TURING_REGIME",
    "TrajectorySample",
    "assess_brusselator_state",
    "build_brusselator_linearization",
    "build_brusselator_system",
    "measure_brusselator_gmres",
    "resolve_regime",
    "sampled_visited_states",
    "state_developedness",
    "visited_states",
]
