"""
Canonical Laplace-domain transfer functions and their known inverses.

Provides test oracles for validating NILT implementations.
"""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp


class LaplacePair(NamedTuple):
    """A Laplace transform pair with F(s) and f(t)."""
    name: str
    F: Callable[[jnp.ndarray], jnp.ndarray]  # F(s) in Laplace domain
    f: Callable[[jnp.ndarray], jnp.ndarray]  # f(t) in time domain
    has_pole_at_origin: bool = False  # True if F has pole at s=0


# =============================================================================
# Elementary Laplace pairs
# =============================================================================

def exponential_decay_F(s: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """F(s) = 1/(s + alpha) -> f(t) = exp(-alpha*t)"""
    return 1.0 / (s + alpha)


def exponential_decay_f(t: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    """f(t) = exp(-alpha*t)"""
    return jnp.exp(-alpha * t)


def cosine_F(s: jnp.ndarray, omega: float = 1.0) -> jnp.ndarray:
    """F(s) = s/(s^2 + omega^2) -> f(t) = cos(omega*t)"""
    return s / (s**2 + omega**2)


def cosine_f(t: jnp.ndarray, omega: float = 1.0) -> jnp.ndarray:
    """f(t) = cos(omega*t)"""
    return jnp.cos(omega * t)


def sine_F(s: jnp.ndarray, omega: float = 1.0) -> jnp.ndarray:
    """F(s) = omega/(s^2 + omega^2) -> f(t) = sin(omega*t)"""
    return omega / (s**2 + omega**2)


def sine_f(t: jnp.ndarray, omega: float = 1.0) -> jnp.ndarray:
    """f(t) = sin(omega*t)"""
    return jnp.sin(omega * t)


def step_F(s: jnp.ndarray) -> jnp.ndarray:
    """F(s) = 1/s -> f(t) = 1 (unit step, requires convolution handling)"""
    return 1.0 / s


def step_f(t: jnp.ndarray) -> jnp.ndarray:
    """f(t) = 1 for t >= 0 (Heaviside step)"""
    return jnp.ones_like(t)


def ramp_F(s: jnp.ndarray) -> jnp.ndarray:
    """F(s) = 1/s^2 -> f(t) = t"""
    return 1.0 / (s**2)


def ramp_f(t: jnp.ndarray) -> jnp.ndarray:
    """f(t) = t"""
    return t


def damped_sine_F(s: jnp.ndarray, alpha: float = 1.0,
                   omega: float = 1.0) -> jnp.ndarray:
    """F(s) = omega/((s+alpha)^2 + omega^2) -> f(t) = exp(-alpha*t)*sin(omega*t)"""
    return omega / ((s + alpha)**2 + omega**2)


def damped_sine_f(t: jnp.ndarray, alpha: float = 1.0,
                  omega: float = 1.0) -> jnp.ndarray:
    """f(t) = exp(-alpha*t) * sin(omega*t)"""
    return jnp.exp(-alpha * t) * jnp.sin(omega * t)


def damped_cosine_F(s: jnp.ndarray, alpha: float = 1.0,
                     omega: float = 1.0) -> jnp.ndarray:
    """F(s) = (s+alpha)/((s+alpha)^2 + omega^2) -> f(t) = exp(-alpha*t)*cos(omega*t)"""
    return (s + alpha) / ((s + alpha)**2 + omega**2)


def damped_cosine_f(t: jnp.ndarray, alpha: float = 1.0,
                    omega: float = 1.0) -> jnp.ndarray:
    """f(t) = exp(-alpha*t) * cos(omega*t)"""
    return jnp.exp(-alpha * t) * jnp.cos(omega * t)


# =============================================================================
# Second-order damping transfer function (main test case)
# =============================================================================

def second_order_damping_F(s: jnp.ndarray) -> jnp.ndarray:
    """
    F(s) = 1/(s^2 + s + 1)

    This is the transfer function of a damped second-order system:
    - Natural frequency: omega_n = 1
    - Damping ratio: zeta = 0.5

    Poles at s = -0.5 ± i*sqrt(3)/2
    """
    return 1.0 / (s**2 + s + 1.0)


def second_order_damping_f(t: jnp.ndarray) -> jnp.ndarray:
    """
    Inverse Laplace transform of F(s) = 1/(s^2 + s + 1).

    f(t) = (2/sqrt(3)) * exp(-t/2) * sin(sqrt(3)*t/2)

    Derivation:
    - Poles: s = -1/2 ± i*sqrt(3)/2
    - omega_d = sqrt(3)/2 (damped frequency)
    - sigma = 1/2 (decay rate)
    """
    omega_d = jnp.sqrt(3.0) / 2.0  # Damped natural frequency
    sigma = 0.5  # Decay rate

    return (2.0 / jnp.sqrt(3.0)) * jnp.exp(-sigma * t) * jnp.sin(omega_d * t)


# =============================================================================
# Collection of standard pairs for testing
# =============================================================================

def get_standard_laplace_pairs() -> list[LaplacePair]:
    """
    Return a list of standard Laplace transform pairs for testing.
    """
    return [
        LaplacePair(
            name="exponential_decay",
            F=lambda s: exponential_decay_F(s, alpha=1.0),
            f=lambda t: exponential_decay_f(t, alpha=1.0),
            has_pole_at_origin=False
        ),
        LaplacePair(
            name="cosine",
            F=lambda s: cosine_F(s, omega=1.0),
            f=lambda t: cosine_f(t, omega=1.0),
            has_pole_at_origin=False
        ),
        LaplacePair(
            name="sine",
            F=lambda s: sine_F(s, omega=1.0),
            f=lambda t: sine_f(t, omega=1.0),
            has_pole_at_origin=False
        ),
        LaplacePair(
            name="damped_sine",
            F=lambda s: damped_sine_F(s, alpha=0.5, omega=2.0),
            f=lambda t: damped_sine_f(t, alpha=0.5, omega=2.0),
            has_pole_at_origin=False
        ),
        LaplacePair(
            name="second_order_damping",
            F=second_order_damping_F,
            f=second_order_damping_f,
            has_pole_at_origin=False
        ),
        LaplacePair(
            name="step",
            F=step_F,
            f=step_f,
            has_pole_at_origin=True  # Pole at s=0
        ),
        LaplacePair(
            name="ramp",
            F=ramp_F,
            f=ramp_f,
            has_pole_at_origin=True  # Double pole at s=0
        ),
    ]


# =============================================================================
# Distributed-parameter transfer functions (engineering benchmarks)
# =============================================================================

def first_order_lag_F(s: jnp.ndarray, K: float = 1.0,
                      tau: float = 1.0) -> jnp.ndarray:
    """
    First-order lag: F(s) = K/(tau*s + 1)
    f(t) = (K/tau) * exp(-t/tau)
    """
    return K / (tau * s + 1.0)


def first_order_lag_f(t: jnp.ndarray, K: float = 1.0,
                      tau: float = 1.0) -> jnp.ndarray:
    """f(t) = (K/tau) * exp(-t/tau)"""
    return (K / tau) * jnp.exp(-t / tau)


def first_order_plus_delay_F(s: jnp.ndarray, K: float = 1.0,
                              tau: float = 1.0, theta: float = 0.5
                              ) -> jnp.ndarray:
    """
    First-order plus dead time (FOPDT): F(s) = K * exp(-theta*s) / (tau*s + 1)
    Common model for process control systems.
    """
    return K * jnp.exp(-theta * s) / (tau * s + 1.0)


def first_order_plus_delay_f(t: jnp.ndarray, K: float = 1.0,
                              tau: float = 1.0, theta: float = 0.5
                              ) -> jnp.ndarray:
    """
    f(t) = (K/tau) * exp(-(t-theta)/tau) * H(t-theta)
    where H is the Heaviside step function.
    """
    return jnp.where(
        t >= theta,
        (K / tau) * jnp.exp(-(t - theta) / tau),
        0.0
    )


def diffusion_semi_infinite_F(s: jnp.ndarray, D: float = 1.0,
                               x: float = 1.0) -> jnp.ndarray:
    """
    Transfer function for diffusion in semi-infinite domain.
    F(s) = exp(-x * sqrt(s/D)) / s

    This represents the concentration at position x for a step
    change in boundary concentration at x=0.
    """
    return jnp.exp(-x * jnp.sqrt(s / D)) / s


def diffusion_semi_infinite_f(t: jnp.ndarray, D: float = 1.0,
                               x: float = 1.0) -> jnp.ndarray:
    """
    f(t) = erfc(x / (2*sqrt(D*t)))

    Complementary error function solution for diffusion.
    """
    from jax.scipy.special import erfc
    # Avoid division by zero at t=0
    t_safe = jnp.maximum(t, 1e-10)
    return erfc(x / (2.0 * jnp.sqrt(D * t_safe)))


def packed_bed_dispersion_F(s: jnp.ndarray, Pe: float = 10.0,
                             L: float = 1.0) -> jnp.ndarray:
    """
    Axial dispersion model for packed bed.
    F(s) = exp(Pe/2 * (1 - sqrt(1 + 4*s*L/Pe)))

    Parameters:
        Pe: Peclet number (v*L/D_ax)
        L: Bed length

    This is the breakthrough curve transfer function.
    """
    # Dimensionless formulation
    q = jnp.sqrt(1.0 + 4.0 * s / Pe)
    return jnp.exp(Pe / 2.0 * (1.0 - q))


# =============================================================================
# Utilities for creating parametrized pairs
# =============================================================================

def make_exponential_pair(alpha: float) -> LaplacePair:
    """Create exponential decay pair with given decay rate."""
    return LaplacePair(
        name=f"exp_decay_alpha{alpha}",
        F=lambda s, a=alpha: exponential_decay_F(s, a),
        f=lambda t, a=alpha: exponential_decay_f(t, a),
        has_pole_at_origin=False
    )


def make_oscillatory_pair(omega: float, alpha: float = 0.0) -> LaplacePair:
    """Create damped oscillatory pair."""
    if alpha == 0:
        return LaplacePair(
            name=f"sine_omega{omega}",
            F=lambda s, w=omega: sine_F(s, w),
            f=lambda t, w=omega: sine_f(t, w),
            has_pole_at_origin=False
        )
    else:
        return LaplacePair(
            name=f"damped_sine_a{alpha}_w{omega}",
            F=lambda s, a=alpha, w=omega: damped_sine_F(s, a, w),
            f=lambda t, a=alpha, w=omega: damped_sine_f(t, a, w),
            has_pole_at_origin=False
        )


def make_second_order_pair(omega_n: float, zeta: float) -> LaplacePair:
    """
    Create general second-order transfer function pair.

    F(s) = omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)

    Parameters:
        omega_n: Natural frequency
        zeta: Damping ratio (0 < zeta < 1 for underdamped)
    """
    def F(s, wn=omega_n, z=zeta):
        return wn**2 / (s**2 + 2*z*wn*s + wn**2)

    def f(t, wn=omega_n, z=zeta):
        if z < 1:
            # Underdamped
            omega_d = wn * jnp.sqrt(1 - z**2)
            return (wn / jnp.sqrt(1 - z**2)) * jnp.exp(-z*wn*t) * jnp.sin(omega_d*t)
        elif z == 1:
            # Critically damped
            return wn**2 * t * jnp.exp(-wn*t)
        else:
            # Overdamped
            r1 = -z*wn + wn*jnp.sqrt(z**2 - 1)
            r2 = -z*wn - wn*jnp.sqrt(z**2 - 1)
            return wn**2 / (r1 - r2) * (jnp.exp(r1*t) - jnp.exp(r2*t))

    return LaplacePair(
        name=f"second_order_wn{omega_n}_zeta{zeta}",
        F=F,
        f=f,
        has_pole_at_origin=False
    )
