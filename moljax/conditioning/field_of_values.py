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

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse.linalg import lobpcg_standard

from moljax.conditioning._geometry import _origin_enclosed, _smallest_enclosing_disk
from moljax.laplace.spectral_bounds import power_iteration_rho

Matvec = Callable[[jax.Array], jax.Array]
_CP_PREFACTOR = 1.0 + math.sqrt(2.0)


class FieldOfValuesResult(NamedTuple):
    """Numerical-range boundary and enclosing-disk diagnostics.

    Attributes:
        boundary: Johnson support points ordered by their sweep direction.
        center: Centre of the minimum enclosing disk of ``boundary``.
        radius: Radius of the minimum enclosing disk of ``boundary``.
        disk_rate: ``radius / abs(center)``; infinity when the centre is zero.
        origin_enclosed: Whether the origin belongs to the convex boundary.
        cp_prefactor: The Crouzeix--Palencia spectral-set prefactor.
    """

    boundary: jax.Array
    center: complex
    radius: float
    disk_rate: float
    origin_enclosed: bool
    cp_prefactor: float


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
) -> jax.Array:
    """Find a dominant eigenvector using a scaled matrix-free LOBPCG solve."""
    real_dimension = 2 * n
    # Realification represents every complex eigenvector by two real vectors,
    # so request a two-vector block to resolve that unavoidable multiplicity.
    padded_dimension = max(real_dimension, 11)
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
    initial_block = jnp.column_stack((padded_initial, jnp.roll(padded_initial, 1)))
    _, vectors, _ = lobpcg_standard(
        shifted_action,
        initial_block,
        m=max_iters,
        tol=tolerance,
    )
    real_vector = vectors[:real_dimension, 0]
    vector = real_vector[:n] + 1j * real_vector[n:]
    return vector / jnp.linalg.norm(vector)


def numerical_range(
    matvec: Matvec,
    matvec_adjoint: Matvec,
    n: int,
    *,
    n_angles: int = 180,
    dtype: jnp.dtype = jnp.complex128,
    max_iters: int = 120,
    tolerance: float = 1.0e-13,
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

    Returns:
        A numerical-range boundary and enclosing-disk diagnostics.

    Raises:
        ValueError: If the dimension, angle count, dtype, or iteration count
            is invalid.
    """
    if n < 1:
        raise ValueError("n must be positive")
    if n_angles < 3:
        raise ValueError("n_angles must be at least three")
    if max_iters < 1:
        raise ValueError("max_iters must be positive")
    if jnp.dtype(dtype) != jnp.dtype(jnp.complex128):
        raise ValueError("numerical-range diagnostics require dtype=jnp.complex128")

    boundary: list[jax.Array] = []
    for index in range(n_angles):
        theta = 2.0 * math.pi * index / n_angles
        hermitian = _rotated_hermitian_action(matvec, matvec_adjoint, theta)
        vector = _largest_hermitian_eigenvector(
            hermitian,
            n,
            theta,
            max_iters,
            tolerance,
        )
        boundary.append(jnp.vdot(vector, _complex_action(matvec, vector)))
    boundary_array = jnp.asarray(boundary, dtype=jnp.complex128)

    boundary_host = np.asarray(boundary_array, dtype=np.complex128)
    center, radius = _smallest_enclosing_disk(boundary_host)
    center_magnitude = abs(center)
    disk_rate = math.inf if center_magnitude == 0.0 else radius / center_magnitude
    return FieldOfValuesResult(
        boundary=boundary_array,
        center=center,
        radius=float(radius),
        disk_rate=float(disk_rate),
        origin_enclosed=_origin_enclosed(boundary_host),
        cp_prefactor=_CP_PREFACTOR,
    )
