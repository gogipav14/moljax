"""
JIT-compiled kernels for FFT operations.

This module provides optimized, JIT-compiled versions of core FFT computations
for maximum performance. These kernels avoid Python overhead and enable
GPU acceleration.

Key optimizations:
- @jax.jit decorators on pure numerical functions
- Static argument handling for array shapes
- Fused operations to minimize memory bandwidth
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax


# =============================================================================
# φ-functions (JIT-compiled)
# =============================================================================

@jax.jit
def phi1(z: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled φ₁(z) = (exp(z) - 1) / z."""
    taylor = 1.0 + z/2.0 + z**2/6.0 + z**3/24.0
    exp_z = jnp.exp(z)
    direct = (exp_z - 1.0) / z
    return jnp.where(jnp.abs(z) < 1e-4, taylor, direct)


@jax.jit
def phi2(z: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled φ₂(z) = (exp(z) - 1 - z) / z²."""
    taylor = 0.5 + z/6.0 + z**2/24.0 + z**3/120.0
    exp_z = jnp.exp(z)
    direct = (exp_z - 1.0 - z) / (z * z)
    return jnp.where(jnp.abs(z) < 1e-4, taylor, direct)


@jax.jit
def phi3(z: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled φ₃(z) = (exp(z) - 1 - z - z²/2) / z³."""
    taylor = 1.0/6.0 + z/24.0 + z**2/120.0 + z**3/720.0
    exp_z = jnp.exp(z)
    direct = (exp_z - 1.0 - z - z**2/2.0) / (z**3)
    return jnp.where(jnp.abs(z) < 1e-4, taylor, direct)


# =============================================================================
# ETD Step Kernels (JIT-compiled)
# =============================================================================

@jax.jit
def etd1_kernel_1d(
    u: jnp.ndarray,
    N: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled ETD1 kernel for 1D fields.

    Computes: u_new = ifft(exp(dt*λ) * fft(u) + dt * φ₁(dt*λ) * fft(N))

    Args:
        u: Current field values
        N: Nonlinear term values
        eigenvalues: FFT eigenvalues λ(k)
        dt: Time step

    Returns:
        Updated field values
    """
    z = dt * eigenvalues
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)

    u_hat = jnp.fft.fft(u)
    N_hat = jnp.fft.fft(N)

    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
    return jnp.real(jnp.fft.ifft(u_new_hat))


@jax.jit
def etd1_kernel_2d(
    u: jnp.ndarray,
    N: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled ETD1 kernel for 2D fields."""
    z = dt * eigenvalues
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)

    u_hat = jnp.fft.fft2(u)
    N_hat = jnp.fft.fft2(N)

    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
    return jnp.real(jnp.fft.ifft2(u_new_hat))


@jax.jit
def etd2_kernel_1d(
    u: jnp.ndarray,
    N_curr: jnp.ndarray,
    N_prev: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled ETD2 kernel for 1D fields.

    ETD2 formula:
        u_{n+1} = exp(dt*L)*u_n + φ₁(dt*L)*dt*N_n + φ₂(dt*L)*dt*(N_n - N_{n-1})
    """
    z = dt * eigenvalues
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)
    phi2_z = phi2(z)

    u_hat = jnp.fft.fft(u)
    N_curr_hat = jnp.fft.fft(N_curr)
    N_prev_hat = jnp.fft.fft(N_prev)

    u_new_hat = (exp_z * u_hat +
                 dt * phi1_z * N_curr_hat +
                 dt * phi2_z * (N_curr_hat - N_prev_hat))
    return jnp.real(jnp.fft.ifft(u_new_hat))


@jax.jit
def etdrk4_kernel_1d(
    u: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
    N_func_vals: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
) -> jnp.ndarray:
    """JIT-compiled ETDRK4 kernel for 1D fields.

    Cox-Matthews 4th order scheme. N_func_vals contains (N_a, N_b, N_c, N_d)
    evaluated at the four stages.
    """
    N_a, N_b, N_c, N_d = N_func_vals
    z = dt * eigenvalues
    z2 = z / 2.0

    exp_z = jnp.exp(z)
    exp_z2 = jnp.exp(z2)

    phi1_z = phi1(z)
    phi2_z = phi2(z)
    phi3_z = phi3(z)
    phi1_z2 = phi1(z2)

    # Coefficients
    b1 = phi1_z - 3*phi2_z + 4*phi3_z
    b2 = 2*phi2_z - 4*phi3_z
    b3 = 2*phi2_z - 4*phi3_z
    b4 = -phi2_z + 4*phi3_z

    u_hat = jnp.fft.fft(u)
    N_a_hat = jnp.fft.fft(N_a)
    N_b_hat = jnp.fft.fft(N_b)
    N_c_hat = jnp.fft.fft(N_c)
    N_d_hat = jnp.fft.fft(N_d)

    u_new_hat = (exp_z * u_hat +
                 dt * (b1 * N_a_hat + b2 * N_b_hat + b3 * N_c_hat + b4 * N_d_hat))
    return jnp.real(jnp.fft.ifft(u_new_hat))


# =============================================================================
# FFT Solve Kernels (JIT-compiled)
# =============================================================================

@jax.jit
def helmholtz_solve_1d(
    rhs: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float,
) -> jnp.ndarray:
    """JIT-compiled Helmholtz solve: (I - dt*D*Δ)u = rhs."""
    rhs_hat = jnp.fft.fft(rhs)
    denom = 1.0 - dt * D * laplacian_symbol
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft(u_hat))


@jax.jit
def helmholtz_solve_2d(
    rhs: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float,
) -> jnp.ndarray:
    """JIT-compiled 2D Helmholtz solve."""
    rhs_hat = jnp.fft.fft2(rhs)
    denom = 1.0 - dt * D * laplacian_symbol
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft2(u_hat))


@jax.jit
def advdiff_solve_1d(
    rhs: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled advection-diffusion solve with complex eigenvalues."""
    rhs_hat = jnp.fft.fft(rhs)
    denom = 1.0 - dt * eigenvalues
    denom = jnp.where(jnp.abs(denom) < 1e-14, 1e-14, denom)
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft(u_hat))


# =============================================================================
# Batched Operations (JIT-compiled)
# =============================================================================

@jax.jit
def batched_fft_solve_1d(
    rhs_stack: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float,
) -> jnp.ndarray:
    """JIT-compiled batched Helmholtz solve for multiple fields.

    Args:
        rhs_stack: Shape (n_fields, nx) - stacked RHS arrays
        laplacian_symbol: Shape (nx,) - shared laplacian symbol
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution stack of shape (n_fields, nx)
    """
    rhs_hat = jnp.fft.fft(rhs_stack, axis=-1)
    denom = 1.0 - dt * D * laplacian_symbol
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft(u_hat, axis=-1))


@jax.jit
def batched_exp_matvec_1d(
    u_stack: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """JIT-compiled batched exp(dt*L)*u for multiple fields."""
    u_hat = jnp.fft.fft(u_stack, axis=-1)
    exp_lam = jnp.exp(dt * eigenvalues)
    result_hat = exp_lam * u_hat
    return jnp.real(jnp.fft.ifft(result_hat, axis=-1))


# =============================================================================
# Integration Loop (JIT-compiled with lax.fori_loop)
# =============================================================================

def make_etd1_integrator(eigenvalues: jnp.ndarray, nonlinear_fn):
    """Create a JIT-compiled ETD1 integrator for a specific problem.

    Args:
        eigenvalues: FFT eigenvalues (fixed for this integrator)
        nonlinear_fn: Function u -> N(u) (must be JIT-compatible)

    Returns:
        JIT-compiled function (u0, t0, dt, n_steps) -> (u_final, t_final)
    """
    @jax.jit
    def step(carry, _):
        u, t = carry
        N_u = nonlinear_fn(u)
        u_new = etd1_kernel_1d(u, N_u, eigenvalues, carry[2])
        return (u_new, t + carry[2], carry[2]), None

    @partial(jax.jit, static_argnums=(3,))
    def integrate(u0: jnp.ndarray, t0: float, dt: float, n_steps: int):
        """Run n_steps of ETD1."""
        init = (u0, t0, dt)
        (u_final, t_final, _), _ = lax.scan(
            lambda carry, _: (
                (etd1_kernel_1d(carry[0], nonlinear_fn(carry[0]), eigenvalues, dt),
                 carry[1] + dt, dt),
                None
            ),
            init,
            None,
            length=n_steps
        )
        return u_final, t_final

    return integrate


# =============================================================================
# Benchmark Utilities
# =============================================================================

def benchmark_jit_speedup(
    u: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float = 0.01,
    n_iterations: int = 100,
) -> dict:
    """Benchmark JIT vs non-JIT performance.

    Returns timing comparison for various operations.
    """
    import time

    # Warmup JIT compilation
    _ = helmholtz_solve_1d(u, eigenvalues.real, dt, 0.01)
    _ = etd1_kernel_1d(u, u, eigenvalues, dt)

    results = {}

    # Benchmark Helmholtz solve
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = helmholtz_solve_1d(u, eigenvalues.real, dt, 0.01)
    jax.block_until_ready(_)
    results['helmholtz_jit_ms'] = (time.perf_counter() - t0) / n_iterations * 1000

    # Benchmark ETD1 kernel
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = etd1_kernel_1d(u, u, eigenvalues, dt)
    jax.block_until_ready(_)
    results['etd1_jit_ms'] = (time.perf_counter() - t0) / n_iterations * 1000

    # Benchmark batched solve
    u_stack = jnp.stack([u, u, u, u], axis=0)
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        _ = batched_fft_solve_1d(u_stack, eigenvalues.real, dt, 0.01)
    jax.block_until_ready(_)
    results['batched_solve_jit_ms'] = (time.perf_counter() - t0) / n_iterations * 1000

    return results


# =============================================================================
# Batched Operations with vmap (Multi-field PDEs)
# =============================================================================

@jax.jit
def batched_etd1_kernel_1d(
    u_stack: jnp.ndarray,
    N_stack: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Batched ETD1 kernel for multiple fields sharing the same linear operator.

    Args:
        u_stack: Shape (n_fields, nx) - stacked field values
        N_stack: Shape (n_fields, nx) - stacked nonlinear terms
        eigenvalues: Shape (nx,) - shared eigenvalues
        dt: Time step

    Returns:
        Updated field stack of shape (n_fields, nx)
    """
    z = dt * eigenvalues
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)

    # Batched FFT along last axis
    u_hat = jnp.fft.fft(u_stack, axis=-1)
    N_hat = jnp.fft.fft(N_stack, axis=-1)

    # Broadcasting: eigenvalues (nx,) broadcasts with (n_fields, nx)
    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
    return jnp.real(jnp.fft.ifft(u_new_hat, axis=-1))


@jax.jit
def batched_etd2_kernel_1d(
    u_stack: jnp.ndarray,
    N_curr_stack: jnp.ndarray,
    N_prev_stack: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Batched ETD2 kernel for multiple fields.

    Args:
        u_stack: Shape (n_fields, nx)
        N_curr_stack: Shape (n_fields, nx) - current nonlinear terms
        N_prev_stack: Shape (n_fields, nx) - previous nonlinear terms
        eigenvalues: Shape (nx,)
        dt: Time step

    Returns:
        Updated field stack of shape (n_fields, nx)
    """
    z = dt * eigenvalues
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)
    phi2_z = phi2(z)

    u_hat = jnp.fft.fft(u_stack, axis=-1)
    N_curr_hat = jnp.fft.fft(N_curr_stack, axis=-1)
    N_prev_hat = jnp.fft.fft(N_prev_stack, axis=-1)

    u_new_hat = (exp_z * u_hat +
                 dt * phi1_z * N_curr_hat +
                 dt * phi2_z * (N_curr_hat - N_prev_hat))
    return jnp.real(jnp.fft.ifft(u_new_hat, axis=-1))


def _make_batched_etd1_step(eigenvalues: jnp.ndarray, n_fields: int):
    """Create a vmap-based ETD1 step for n_fields.

    Returns a JIT-compiled function that steps n_fields simultaneously,
    each with potentially different nonlinear terms.
    """
    # Single-field ETD1 step (without @jax.jit to allow vmap)
    def single_etd1(u, N):
        z = 0.01 * eigenvalues  # dt fixed for closure
        exp_z = jnp.exp(z)
        phi1_z = phi1(z)
        u_hat = jnp.fft.fft(u)
        N_hat = jnp.fft.fft(N)
        u_new_hat = exp_z * u_hat + 0.01 * phi1_z * N_hat
        return jnp.real(jnp.fft.ifft(u_new_hat))

    # vmap over batch dimension
    batched_step = jax.vmap(single_etd1, in_axes=(0, 0))

    return jax.jit(batched_step)


@jax.jit
def multi_operator_etd1_1d(
    u_stack: jnp.ndarray,
    N_stack: jnp.ndarray,
    eigenvalues_stack: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """ETD1 for multiple fields with different operators.

    Unlike batched_etd1_kernel_1d, this handles the case where each field
    has its own linear operator (e.g., different diffusion coefficients).

    Args:
        u_stack: Shape (n_fields, nx)
        N_stack: Shape (n_fields, nx)
        eigenvalues_stack: Shape (n_fields, nx) - each field has own eigenvalues
        dt: Time step

    Returns:
        Updated field stack
    """
    z = dt * eigenvalues_stack  # (n_fields, nx)
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)

    u_hat = jnp.fft.fft(u_stack, axis=-1)
    N_hat = jnp.fft.fft(N_stack, axis=-1)

    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
    return jnp.real(jnp.fft.ifft(u_new_hat, axis=-1))


# =============================================================================
# 2D Batched Operations
# =============================================================================

@jax.jit
def batched_etd1_kernel_2d(
    u_stack: jnp.ndarray,
    N_stack: jnp.ndarray,
    eigenvalues: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """Batched 2D ETD1 kernel for multiple fields.

    Args:
        u_stack: Shape (n_fields, ny, nx)
        N_stack: Shape (n_fields, ny, nx)
        eigenvalues: Shape (ny, nx) - shared 2D eigenvalues
        dt: Time step

    Returns:
        Updated field stack of shape (n_fields, ny, nx)
    """
    z = dt * eigenvalues
    exp_z = jnp.exp(z)
    phi1_z = phi1(z)

    # Batched 2D FFT over last two axes
    u_hat = jnp.fft.fft2(u_stack, axes=(-2, -1))
    N_hat = jnp.fft.fft2(N_stack, axes=(-2, -1))

    u_new_hat = exp_z * u_hat + dt * phi1_z * N_hat
    return jnp.real(jnp.fft.ifft2(u_new_hat, axes=(-2, -1)))


@jax.jit
def batched_helmholtz_solve_2d(
    rhs_stack: jnp.ndarray,
    laplacian_symbol: jnp.ndarray,
    dt: float,
    D: float,
) -> jnp.ndarray:
    """Batched 2D Helmholtz solve for multiple fields.

    Args:
        rhs_stack: Shape (n_fields, ny, nx)
        laplacian_symbol: Shape (ny, nx)
        dt: Time step
        D: Diffusion coefficient

    Returns:
        Solution stack of shape (n_fields, ny, nx)
    """
    rhs_hat = jnp.fft.fft2(rhs_stack, axes=(-2, -1))
    denom = 1.0 - dt * D * laplacian_symbol
    u_hat = rhs_hat / denom
    return jnp.real(jnp.fft.ifft2(u_hat, axes=(-2, -1)))


# =============================================================================
# Batched Operations Benchmark
# =============================================================================

def benchmark_batched_scaling(
    nx: int = 256,
    n_fields_list: list = None,
    n_iterations: int = 50,
) -> dict:
    """Benchmark batched operations scaling with number of fields.

    Args:
        nx: Grid size
        n_fields_list: List of number of fields to test
        n_iterations: Number of timed iterations

    Returns:
        Dictionary with timing results
    """
    import time

    if n_fields_list is None:
        n_fields_list = [1, 2, 4, 8, 16, 32]

    dx = 1.0 / nx
    D = 0.01
    dt = 0.01

    # Create eigenvalues
    k = jnp.fft.fftfreq(nx, d=dx) * 2 * jnp.pi
    lap_sym = -k**2
    eigenvalues = D * lap_sym

    x = jnp.linspace(0, 1, nx, endpoint=False)
    u_base = jnp.sin(2 * jnp.pi * x)
    N_base = 0.1 * u_base**2

    results = {
        'nx': nx,
        'n_fields': n_fields_list,
        'shared_op_ms': [],
        'multi_op_ms': [],
        'sequential_ms': [],
    }

    for n_fields in n_fields_list:
        # Setup stacks
        u_stack = jnp.stack([u_base] * n_fields, axis=0)
        N_stack = jnp.stack([N_base] * n_fields, axis=0)
        eig_stack = jnp.stack([eigenvalues] * n_fields, axis=0)

        # Warmup
        _ = batched_etd1_kernel_1d(u_stack, N_stack, eigenvalues, dt)
        _ = multi_operator_etd1_1d(u_stack, N_stack, eig_stack, dt)
        jax.block_until_ready(_)

        # Shared operator (batched)
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            result = batched_etd1_kernel_1d(u_stack, N_stack, eigenvalues, dt)
        jax.block_until_ready(result)
        results['shared_op_ms'].append((time.perf_counter() - t0) / n_iterations * 1000)

        # Multiple operators
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            result = multi_operator_etd1_1d(u_stack, N_stack, eig_stack, dt)
        jax.block_until_ready(result)
        results['multi_op_ms'].append((time.perf_counter() - t0) / n_iterations * 1000)

        # Sequential (for comparison)
        t0 = time.perf_counter()
        for _ in range(n_iterations):
            outputs = []
            for i in range(n_fields):
                out = etd1_kernel_1d(u_stack[i], N_stack[i], eigenvalues, dt)
                outputs.append(out)
            result = jnp.stack(outputs, axis=0)
        jax.block_until_ready(result)
        results['sequential_ms'].append((time.perf_counter() - t0) / n_iterations * 1000)

    return results


def print_batched_scaling_report(results: dict):
    """Print formatted report of batched scaling benchmark."""
    print("\n" + "=" * 70)
    print("Batched Operations Scaling Report")
    print("=" * 70)
    print(f"Grid size: {results['nx']}")
    print()
    print(f"{'Fields':>8} | {'Shared Op':>12} | {'Multi Op':>12} | {'Sequential':>12} | {'Speedup':>8}")
    print("-" * 70)

    for i, n in enumerate(results['n_fields']):
        shared = results['shared_op_ms'][i]
        multi = results['multi_op_ms'][i]
        seq = results['sequential_ms'][i]
        speedup = seq / shared if shared > 0 else 0

        print(f"{n:>8} | {shared:>12.4f} | {multi:>12.4f} | {seq:>12.4f} | {speedup:>8.1f}x")

    print("-" * 70)
    print("Shared Op: All fields share same linear operator (most efficient)")
    print("Multi Op:  Each field has different linear operator")
    print("Sequential: Loop over fields (baseline)")
    print("Speedup: Sequential / Shared Op")
