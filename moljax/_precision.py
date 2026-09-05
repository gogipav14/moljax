"""Precision guards shared by the subpackages."""

from __future__ import annotations

import jax


def require_x64(what: str) -> None:
    """
    Raise RuntimeError unless JAX is running with 64-bit precision.

    JAX builds float32 arrays without a word when float64 is requested with
    x64 off, so code whose answer depends on float64 (eigenvector residuals
    at 1e-10, Gaver-Stehfest weights near 1e8 with alternating signs) would
    return confident wrong numbers instead of failing. Callers name what
    needs the precision so the message says which call to fix.
    """
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            f"{what}: 64-bit precision is required; enable it with "
            'jax.config.update("jax_enable_x64", True) before calling.'
        )
