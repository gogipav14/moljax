"""Matrix-free numerical-range diagnostics for linear operators.

The numerical range, also called the field of values, is traced with the
Johnson support-function construction.  For each direction ``theta``, this
module finds a dominant eigenvector of the rotated Hermitian part
``(exp(i theta) A + exp(-i theta) A*) / 2`` and evaluates its Rayleigh value
under ``A``.  The resulting boundary supplies an enclosing-disk estimate for
stationary-iteration behavior.

The ``cp_prefactor`` result field records the universal Crouzeix--Palencia
spectral-set constant ``1 + sqrt(2)``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import lobpcg_standard

from moljax.conditioning._geometry import (
    _origin_enclosed,
    _smallest_enclosing_disk,
    _support_outer_polygon,
)
from moljax.laplace.spectral_bounds import power_iteration_rho

Matvec = Callable[[jax.Array], jax.Array]
_CP_PREFACTOR = 1.0 + math.sqrt(2.0)


class FieldOfValuesResult(NamedTuple):
    """Numerical-range boundary and enclosing-disk diagnostics.

    Attributes:
        boundary: Johnson support points ordered by their sweep direction.
        center: Centre of the minimum disk enclosing the support half-plane
            intersection, which contains the numerical range.
        radius: Radius of that disk.  This is an outer bound, so it never
            understates the range.
        disk_rate: ``radius / abs(center)``; infinity when the centre is zero.
        origin_enclosed: Whether the origin fails to be certified outside the
            numerical range.  ``False`` means a sampled support direction
            separates the origin from the range, which is a proof; ``True``
            means no such certificate was found and the caller should abstain.
        cp_prefactor: The Crouzeix--Palencia spectral-set prefactor.
        max_support_residual: Largest relative eigenpair residual over the
            support directions.  Reports how well the boundary is resolved so
            that the disk rate can be read with its convergence quality in
            view, rather than trusting an unqualified number.
        supports_corroborated: Whether no eigensolver restart disagreed on a
            support value.  No fixed starting block can *prove* it found a
            global maximum, so the outer-bound property is conditional on the
            supports being true maxima.  Disagreement shows that condition is
            unmet; agreement is corroboration, not proof.  With the default
            ``n_restarts=1`` no restart is run and the flag is vacuously true.
    """

    boundary: jax.Array
    center: complex
    radius: float
    disk_rate: float
    origin_enclosed: bool
    cp_prefactor: float
    max_support_residual: float = 0.0
    supports_corroborated: bool = True


def _complex_action(action: Matvec, value: jax.Array) -> jax.Array:
    """Apply a real or complex linear action to a complex vector."""
    real = jnp.asarray(action(jnp.real(value)), dtype=jnp.complex128)
    imag = jnp.asarray(action(jnp.imag(value)), dtype=jnp.complex128)
    return real + 1j * imag


def _rotated_hermitian_action(matvec: Matvec, matvec_adjoint: Matvec, theta: float) -> Matvec:
    """Return the Johnson rotated Hermitian-part action for one direction."""
    phase = jnp.exp(jnp.asarray(1j * theta, dtype=jnp.complex128))

    def action(value: jax.Array) -> jax.Array:
        forward = _complex_action(matvec, value)
        adjoint = _complex_action(matvec_adjoint, value)
        return 0.5 * (phase * forward + jnp.conj(phase) * adjoint)

    return action


def _realified_action(action: Matvec, n: int) -> Callable[[jax.Array], jax.Array]:
    """Realify a complex Hermitian action for JAX's matrix-free LOBPCG solver."""

    def apply_vector(value: jax.Array) -> jax.Array:
        complex_value = value[:n] + 1j * value[n : 2 * n]
        complex_result = action(complex_value)
        return jnp.concatenate((jnp.real(complex_result), jnp.imag(complex_result)))

    def realified(value: jax.Array) -> jax.Array:
        if value.ndim == 1:
            return apply_vector(value)
        return jax.vmap(apply_vector, in_axes=1, out_axes=1)(value)

    return realified


def _largest_hermitian_eigenvector(
    action: Matvec,
    n: int,
    theta: float,
    max_iters: int,
    tolerance: float,
    residual_tolerance: float,
    restart: int,
) -> tuple[jax.Array, float]:
    """Find a dominant eigenvector and its relative eigenpair residual."""
    real_dimension = 2 * n
    # Realification represents every complex eigenvector by two real vectors,
    # so the block needs two columns to resolve that unavoidable multiplicity,
    # plus a third random direction guarding against a start block aligned
    # against the dominant eigenspace.  lobpcg_standard requires
    # 5 * block_width < dimension, so pad to satisfy that.
    block_width = 3
    padded_dimension = max(real_dimension, 5 * block_width + 1)
    real_action = _realified_action(action, n)
    initial_complex = jnp.sin(jnp.arange(n, dtype=jnp.float64) + theta + 1.0) + 1j * jnp.cos(
        jnp.arange(n, dtype=jnp.float64) + 0.5 * theta + 0.5
    )
    initial_real = jnp.concatenate((jnp.real(initial_complex), jnp.imag(initial_complex)))
    spectral_scale = power_iteration_rho(
        real_action,
        initial_real,
        max_iters=max(50, min(max_iters, 100)),
        tol=tolerance,
    )
    shift = max(2.0 * abs(spectral_scale), 1.0)

    def shifted_action(value: jax.Array) -> jax.Array:
        active = value[:real_dimension]
        shifted = (real_action(active) + shift * active) / shift
        if value.ndim == 1:
            return jnp.pad(shifted, (0, padded_dimension - real_dimension))
        return jnp.pad(shifted, ((0, padded_dimension - real_dimension), (0, 0)))

    padded_initial = jnp.pad(initial_real, (0, padded_dimension - real_dimension))
    # A fixed starting block can, in principle, be orthogonal to the dominant
    # invariant subspace, in which case LOBPCG converges to a subdominant
    # eigenpair with a small residual and the residual gate below accepts it.
    # The reported support would then be too small and the half-plane would no
    # longer contain the numerical range, breaking the outer-bound guarantee.
    # A third pseudo-random direction, seeded from the sweep angle so runs stay
    # reproducible, makes such an alignment a probability-zero event.
    probe_key = jax.random.PRNGKey(
        (int(theta * 1_000_003) + 7_919 * restart) & 0x7FFFFFFF
    )
    random_column = jax.random.normal(probe_key, (padded_dimension,), dtype=jnp.float64)
    if restart == 0:
        initial_block = jnp.column_stack(
            (padded_initial, jnp.roll(padded_initial, 1), random_column)
        )
    else:
        # Later attempts start from an entirely independent space, so a
        # dominant eigenspace orthogonal to the first attempt's columns is not
        # also orthogonal to these.
        extra = jax.random.normal(
            jax.random.fold_in(probe_key, 1), (padded_dimension, 2), dtype=jnp.float64
        )
        initial_block = jnp.column_stack((random_column, extra))
    _, vectors, _ = lobpcg_standard(
        shifted_action,
        initial_block,
        m=max_iters,
        tol=tolerance,
    )
    candidate = vectors[:, 0]
    # lobpcg_standard reports neither convergence nor a residual, so an
    # iteration budget that is too small returns a vector that is not an
    # eigenvector at all.  Accepting it would place the support point in the
    # interior of the numerical range, understating the radius and, with it,
    # the disk rate that the adequacy verdict is read from.  Validate the
    # eigenpair before trusting it.
    action_value = shifted_action(candidate)
    rayleigh = jnp.real(jnp.vdot(candidate, action_value))
    residual = float(jnp.linalg.norm(action_value - rayleigh * candidate))
    relative = residual / max(float(abs(rayleigh)), 1.0)
    if not math.isfinite(relative) or relative > residual_tolerance:
        raise RuntimeError(
            "numerical-range support did not converge at theta="
            f"{theta:.6f}: relative eigenpair residual {relative:.3e} exceeds "
            f"{residual_tolerance:.3e} after {max_iters} LOBPCG iterations. "
            "Increase max_iters, or loosen residual_tolerance only if an "
            "approximate boundary is acceptable."
        )
    real_vector = candidate[:real_dimension]
    vector = real_vector[:n] + 1j * real_vector[n:]
    return vector / jnp.linalg.norm(vector), relative


def numerical_range(
    matvec: Matvec,
    matvec_adjoint: Matvec,
    n: int,
    *,
    n_angles: int = 180,
    dtype: jnp.dtype = jnp.complex128,
    max_iters: int = 120,
    tolerance: float = 1.0e-13,
    residual_tolerance: float = 1.0e-3,
    n_restarts: int = 1,
) -> FieldOfValuesResult:
    """Trace a matrix-free numerical-range boundary using Johnson supports.

    Args:
        matvec: Callable computing ``A @ v`` for a vector ``v``.
        matvec_adjoint: Callable computing the Euclidean adjoint ``A* @ v``.
        n: Dimension of the linear operator.
        n_angles: Number of equally spaced support directions.
        dtype: Complex working dtype.  Numerical-range diagnostics require
            ``complex128`` to provide float64 accuracy.
        max_iters: Maximum matrix-free LOBPCG iterations per direction.
        tolerance: Relative eigensolver tolerance.
        residual_tolerance: Largest relative eigenpair residual accepted for a
            support direction.  Exceeding it raises rather than returning a
            boundary point that is not on the boundary.
        n_restarts: Independent eigensolver starts per direction.  A single
            fixed start cannot establish that it found a global maximum;
            additional starts corroborate it, and disagreement clears
            ``supports_corroborated``.  The default of one keeps the
            diagnostic cheap, in which case no corroboration is attempted and
            the flag is vacuously true.  Raise it when a verdict is load
            bearing.

    Returns:
        A numerical-range boundary and enclosing-disk diagnostics.

    Raises:
        ValueError: If the dimension, angle count, dtype, or iteration count
            is invalid.
        RuntimeError: If a support direction fails to converge within
            ``max_iters`` to ``residual_tolerance``.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if n_angles < 3:
        raise ValueError("n_angles must be at least three")
    if max_iters < 1:
        raise ValueError("max_iters must be positive")
    if jnp.dtype(dtype) != jnp.dtype(jnp.complex128):
        raise ValueError("numerical-range diagnostics require dtype=jnp.complex128")
    if residual_tolerance <= 0.0:
        raise ValueError("residual_tolerance must be positive")
    if n_restarts < 1:
        raise ValueError("n_restarts must be positive")
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "conditioning diagnostics require 64-bit precision; enable it with "
            'jax.config.update("jax_enable_x64", True) before calling.'
        )

    boundary: list[jax.Array] = []
    thetas: list[float] = []
    worst_residual = 0.0
    corroborated = True
    for index in range(n_angles):
        theta = 2.0 * math.pi * index / n_angles
        thetas.append(theta)
        hermitian = _rotated_hermitian_action(matvec, matvec_adjoint, theta)
        rotation = complex(math.cos(theta), math.sin(theta))
        best_vector: jax.Array | None = None
        best_support = -math.inf
        attempt_supports: list[float] = []
        for restart in range(n_restarts):
            candidate, support_residual = _largest_hermitian_eigenvector(
                hermitian,
                n,
                theta,
                max_iters,
                tolerance,
                residual_tolerance,
                restart,
            )
            worst_residual = max(worst_residual, support_residual)
            value = complex(jnp.vdot(candidate, _complex_action(matvec, candidate)))
            attempt = (rotation * value).real
            attempt_supports.append(attempt)
            # Every Rayleigh quotient is a lower bound on the true support, so
            # keeping the largest attempt is always the safe direction.
            if attempt > best_support:
                best_support, best_vector = attempt, candidate
        if len(attempt_supports) > 1:
            spread = max(attempt_supports) - min(attempt_supports)
            if spread > residual_tolerance * max(abs(best_support), 1.0):
                corroborated = False
        vector = best_vector
        boundary.append(jnp.vdot(vector, _complex_action(matvec, vector)))
    boundary_array = jnp.asarray(boundary, dtype=jnp.complex128)

    boundary_host = np.asarray(boundary_array, dtype=np.complex128)
    theta_host = np.asarray(thetas, dtype=np.float64)

    # The sampled boundary points are an inscribed approximation of the
    # numerical range, so a disk fitted to them can be smaller than the range
    # itself and the origin can fall outside their hull while lying inside the
    # range.  Both errors push the verdict toward a false certificate.  Fit the
    # disk to the half-plane intersection instead, which provably contains the
    # range, and certify the origin only when a sampled direction separates it.
    supports = np.real(np.exp(1j * theta_host) * boundary_host)
    outer = _support_outer_polygon(theta_host, supports)
    center, radius = _smallest_enclosing_disk(outer)
    center_magnitude = abs(center)
    disk_rate = math.inf if center_magnitude == 0.0 else radius / center_magnitude
    # A negative support value is a separating half-plane and therefore a
    # certificate that the origin lies outside.  Without one, report the origin
    # as enclosed so the caller abstains rather than certifies.
    origin_outside_certified = bool(np.min(supports) < 0.0)
    origin_enclosed = not origin_outside_certified
    if not origin_enclosed:
        # Sanity: an inscribed-hull enclosure would contradict the certificate.
        origin_enclosed = bool(_origin_enclosed(boundary_host))
    return FieldOfValuesResult(
        boundary=boundary_array,
        center=center,
        radius=float(radius),
        disk_rate=float(disk_rate),
        origin_enclosed=origin_enclosed,
        cp_prefactor=_CP_PREFACTOR,
        max_support_residual=worst_residual,
        supports_corroborated=corroborated,
    )
