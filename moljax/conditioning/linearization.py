"""Generic matrix-free linearization adapters for JFNK diagnostics.

The public adapter keeps the solver state as a PyTree internally and exposes a
flat complex vector interface at the conditioning-toolbox boundary.  Complex
actions are the real-linear extension: the real and imaginary parts are
applied independently to the real JVP, VJP, and preconditioner actions.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, NamedTuple

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

State = Any
Residual = Callable[[State], State]
Action = Callable[[jax.Array], jax.Array]


class LinearizedOperator(NamedTuple):
    """Flat complex matrix-free operator and its Euclidean adjoint."""

    matvec: Action
    matvec_adjoint: Action
    n: int


def _complex_action(action: Action, value: jax.Array) -> jax.Array:
    """Apply a real-linear action to a flat complex vector."""
    complex_value = jnp.asarray(value, dtype=jnp.complex128)
    real = jnp.asarray(action(jnp.real(complex_value)), dtype=jnp.complex128)
    imag = jnp.asarray(action(jnp.imag(complex_value)), dtype=jnp.complex128)
    return real + 1j * imag


def linearized_operator(
    residual_fn: Residual,
    x: State,
    *,
    preconditioner: Any = None,
    context: Any = None,
) -> LinearizedOperator:
    """Build ``P^-1 J`` and ``J^T P^-T`` from a public residual callable.

    ``residual_fn`` is differentiated directly with :func:`jax.jvp` and
    :func:`jax.vjp`; no solver-private JVP helper is required.  The state and
    residual PyTrees are flattened with :func:`jax.flatten_util.ravel_pytree`.
    The returned toolbox vectors are one-dimensional ``complex128`` arrays,
    with the real and imaginary parts representing two independent real
    applications of the underlying operator.

    Args:
        residual_fn: Public JFNK residual ``State -> State``.
        x: Linearization point, normally a real-valued state PyTree.
        preconditioner: Optional object exposing ``apply(state, context)``.
        context: The preconditioner's :class:`PrecondContext`.

    Raises:
        ValueError: If the residual changes the flattened dimension or a
            preconditioner is supplied without its context.
    """
    if preconditioner is not None and context is None:
        raise ValueError("context is required when preconditioner is supplied")

    flat_x, unravel_x = ravel_pytree(x)
    residual_at_x = residual_fn(x)
    flat_residual, unravel_residual = ravel_pytree(residual_at_x)
    if flat_x.size != flat_residual.size:
        raise ValueError(
            "JFNK linearization must be square after flattening: "
            f"state has {flat_x.size} entries, residual has {flat_residual.size}"
        )

    _, residual_pullback = jax.vjp(residual_fn, x)

    def jacobian_flat(value: jax.Array) -> jax.Array:
        tangent = unravel_x(value)
        _, tangent_residual = jax.jvp(residual_fn, (x,), (tangent,))
        return ravel_pytree(tangent_residual)[0]

    if preconditioner is None:

        def precondition_flat(value: jax.Array) -> jax.Array:
            return value

    else:

        def precondition_flat(value: jax.Array) -> jax.Array:
            residual = unravel_residual(value)
            preconditioned = preconditioner.apply(residual, context)
            return ravel_pytree(preconditioned)[0]

    jacobian_flat = jax.jit(jacobian_flat)
    precondition_flat = jax.jit(precondition_flat)

    transpose_rule = jax.linear_transpose(
        precondition_flat,
        jnp.zeros_like(flat_residual),
    )

    def precondition_transpose_flat(value: jax.Array) -> jax.Array:
        return transpose_rule(value)[0]

    def jacobian_transpose_flat(value: jax.Array) -> jax.Array:
        cotangent = unravel_residual(value)
        cotangent_x = residual_pullback(cotangent)[0]
        return ravel_pytree(cotangent_x)[0]

    precondition_transpose_flat = jax.jit(precondition_transpose_flat)
    jacobian_transpose_flat = jax.jit(jacobian_transpose_flat)

    def real_matvec(value: jax.Array) -> jax.Array:
        return precondition_flat(jacobian_flat(value))

    def real_matvec_adjoint(value: jax.Array) -> jax.Array:
        return jacobian_transpose_flat(precondition_transpose_flat(value))

    real_matvec = jax.jit(real_matvec)
    real_matvec_adjoint = jax.jit(real_matvec_adjoint)

    def matvec(value: jax.Array) -> jax.Array:
        return _complex_action(real_matvec, value)

    def matvec_adjoint(value: jax.Array) -> jax.Array:
        return _complex_action(real_matvec_adjoint, value)

    return LinearizedOperator(matvec=matvec, matvec_adjoint=matvec_adjoint, n=int(flat_x.size))


def adjoint_identity(op: LinearizedOperator, key: jax.Array, n: int) -> float:
    """Return the absolute complex inner-product adjoint residual.

    The random vectors are normalized so the returned value is an absolute
    error on unit-scale test vectors:
    ``|<A v, w> - <v, A* w>|``.
    """
    if n != op.n:
        raise ValueError(f"n={n} does not match operator dimension {op.n}")

    key_v_real, key_v_imag, key_w_real, key_w_imag = jax.random.split(key, 4)
    v = jax.random.normal(key_v_real, (n,), dtype=jnp.float64)
    w = jax.random.normal(key_w_real, (n,), dtype=jnp.float64)
    v = v + 1j * jax.random.normal(key_v_imag, (n,), dtype=jnp.float64)
    w = w + 1j * jax.random.normal(key_w_imag, (n,), dtype=jnp.float64)
    v = v / jnp.linalg.norm(v)
    w = w / jnp.linalg.norm(w)
    forward = op.matvec(v)
    adjoint = op.matvec_adjoint(w)
    error = jnp.abs(jnp.vdot(forward, w) - jnp.vdot(v, adjoint))
    return float(error)


__all__ = ["LinearizedOperator", "adjoint_identity", "linearized_operator"]
