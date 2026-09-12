"""
Tests for the phi functions phi_n(z) = (exp(z) - sum_{j<n} z^j/j!) / z^n.

The direct formula loses n digits of relative accuracy per decade below
|z| = 1 through cancellation, so the implementation switches to a Taylor
series below a dtype-dependent threshold. Reference values come from the
series in exact rational arithmetic.
"""

from fractions import Fraction
from math import factorial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from moljax.core.jit_kernels import phi1, phi2, phi3

MAGNITUDES = [1e-8, 1e-6, 1e-4, 3e-4, 1e-3, 1e-2, 1e-1, 0.5, 2.0]
PHI = {1: phi1, 2: phi2, 3: phi3}


def phi_reference(z: float, n: int) -> float:
    """phi_n(z) from 60 series terms in Fraction arithmetic (z is taken exactly)."""
    zf = Fraction(z)
    total = Fraction(0)
    for j in range(60):
        total += zf ** j / factorial(j + n)
    return float(total)


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_phi_float64(n, magnitude, sign):
    z = jnp.array([sign * magnitude], dtype=jnp.float64)
    value = float(PHI[n](z)[0])
    ref = phi_reference(float(z[0]), n)
    assert abs(value - ref) / abs(ref) < 1e-12, f"phi{n}({float(z[0])}): {value} vs {ref}"


@pytest.mark.parametrize("n", [1, 2, 3])
@pytest.mark.parametrize("magnitude", MAGNITUDES)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_phi_float32(n, magnitude, sign):
    z = jnp.array([sign * magnitude], dtype=jnp.float32)
    value = float(PHI[n](z)[0])
    ref = phi_reference(float(z[0]), n)
    assert abs(value - ref) / abs(ref) < 1e-6, f"phi{n}({float(z[0])}): {value} vs {ref}"


@pytest.mark.parametrize("n", [1, 2, 3])
def test_phi_complex_point(n):
    """Advection eigenvalues are complex; the series must accept them."""
    z = jnp.array([-0.02 + 0.03j], dtype=jnp.complex128)
    value = complex(PHI[n](z)[0])
    ref = sum(complex(z[0]) ** j / factorial(j + n) for j in range(40))
    assert abs(value - ref) / abs(ref) < 1e-12


def test_phi_matches_direct_formula_away_from_zero():
    """Where cancellation is harmless the direct formula is the reference."""
    z = jnp.array([-10.0, -3.0, 3.0, 10.0], dtype=jnp.float64)
    ez = np.exp(np.asarray(z))
    zz = np.asarray(z)
    expected = {
        1: (ez - 1) / zz,
        2: (ez - 1 - zz) / zz**2,
        3: (ez - 1 - zz - zz**2 / 2) / zz**3,
    }
    for n, fn in PHI.items():
        assert np.allclose(np.asarray(fn(z)), expected[n], rtol=1e-13, atol=0)


# phi1(z) = 1 + z/2 + z^2/6 + ..., so phi1'(0) = 1/2; phi2'(0) = 1/6;
# phi3'(0) = 1/24 from the same series shifted by n. Before the fix, the
# direct branch's 0/0 at z = 0 produced a NaN cotangent through jnp.where
# even though the series branch is the one selected there.
EXPECTED_GRAD_AT_ZERO = {1: 0.5, 2: 1.0 / 6.0, 3: 1.0 / 24.0}


@pytest.mark.parametrize("n", [1, 2, 3])
def test_phi_gradients_finite_at_zero(n):
    """jax.grad(phi_n) at z = 0 must be finite and match the series derivative."""
    fn = PHI[n]

    def scalar(z):
        return fn(jnp.array([z], dtype=jnp.float64))[0]

    grad = jax.grad(scalar)(0.0)
    assert jnp.isfinite(grad), f"phi{n}'(0) is not finite: {grad}"
    assert abs(float(grad) - EXPECTED_GRAD_AT_ZERO[n]) < 1e-12

    grad_jit = jax.jit(jax.grad(scalar))(0.0)
    assert jnp.isfinite(grad_jit), f"jitted phi{n}'(0) is not finite: {grad_jit}"
    assert abs(float(grad_jit) - EXPECTED_GRAD_AT_ZERO[n]) < 1e-12
