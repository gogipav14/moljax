# benchmarks/benchmark_utils.py
"""Shared benchmark infrastructure with publication-grade guardrails.

IMPORTANT: Each benchmark script must call jax.config.update("jax_enable_x64", True)
at the VERY TOP before any other JAX imports. Alternatively, set JAX_ENABLE_X64=True
in the shell environment.

Example usage at top of benchmark scripts:
    import jax
    jax.config.update("jax_enable_x64", True)
    # Now safe to import other modules
    import jax.numpy as jnp
    from benchmark_utils import setup_benchmark, time_call, check_finite_tree
"""
import argparse
import time
import numpy as np

# NOTE: Do NOT import jax.numpy here at module scope!
# x64 config must be set before jax.numpy is imported.


def setup_benchmark(expected_backend="gpu"):
    """Standard benchmark setup - call AFTER setting x64 config.

    Args:
        expected_backend: "gpu", "cpu", or None (skip backend check).
                         Use None for mixed CPU/GPU scripts (e.g., SciPy baselines).

    Returns:
        Device string for JSON metadata.
    """
    import jax

    # Verify x64 was already set
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "Float64 not enabled! Add jax.config.update('jax_enable_x64', True) "
            "at the VERY TOP of your script, before any jax.numpy imports."
        )

    # Backend check (flexible for CPU baselines)
    actual_backend = jax.default_backend()
    if expected_backend is not None and actual_backend != expected_backend:
        raise RuntimeError(f"Expected {expected_backend} backend, got {actual_backend}")

    device_str = str(jax.devices()[0])
    print(f"Device: {device_str}")
    print(f"Backend: {actual_backend}")
    print(f"Float64: {jax.config.jax_enable_x64}")
    return device_str


def check_finite_tree(pytree, name="state"):
    """Check all leaves of a PyTree for NaN/Inf.

    Args:
        pytree: JAX PyTree (dict, tuple, or array)
        name: Name for error messages

    Raises:
        ValueError if any leaf contains NaN or Inf
    """
    import jax
    import jax.numpy as jnp

    leaves = jax.tree_util.tree_leaves(pytree)
    for i, leaf in enumerate(leaves):
        if not jnp.all(jnp.isfinite(leaf)):
            raise ValueError(f"{name} leaf {i} contains NaN or Inf!")
    return True


def compute_stats(times):
    """Compute median and IQR from timing array."""
    median = float(np.median(times))
    q25, q75 = np.percentile(times, [25, 75])
    iqr = float(q75 - q25)
    return median, iqr


def ps_fft_stability_limit(N, L, D, ndim=2, integrator="RK4", safety=0.8):
    """Compute stability-limited dt for pseudo-spectral FFT Laplacian.

    Args:
        N: Grid points per dimension
        L: Domain length
        D: Diffusion coefficient
        ndim: Number of spatial dimensions (default 2 for 2D problems)
        integrator: "RK4" (bound=2.8), "FE" (bound=2.0), "SSPRK3" (bound=1.0)
        safety: Safety factor (default 0.8, i.e., 80% of theoretical limit)

    Returns:
        Stable timestep dt

    Note:
        For 2D: lambda_max = k_x,max^2 + k_y,max^2 = ndim * k_max^2
        where k_max = (N/2) * (2*pi/L)
    """
    stability_bounds = {"RK4": 2.8, "FE": 2.0, "SSPRK3": 1.0}
    bound = stability_bounds.get(integrator, 2.8)

    # PS-FFT: k_max = (N/2) * (2*pi/L)
    k_max = (N / 2) * (2 * np.pi / L)
    # For ndim dimensions: lambda_max = ndim * k_max^2
    lambda_max = ndim * (k_max ** 2)

    dt_max = safety * bound / (D * lambda_max)
    return dt_max


def time_call(fn, n_reps=10):
    """Standardized timing with warmup and synchronization.

    Protocol:
    1. One warmup call (discarded, ensures JIT compilation)
    2. n_reps timed calls with block_until_ready() sync
    3. Returns median and IQR

    Args:
        fn: Callable that returns a JAX array or PyTree
        n_reps: Number of repetitions (default 10)

    Returns:
        (result, median_time, iqr_time, times_array)
    """
    import jax

    # Warmup (compile + discard)
    result = fn()
    if hasattr(result, 'block_until_ready'):
        result.block_until_ready()
    else:
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)

    # Timed runs
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        result = fn()
        # Synchronize before stopping clock
        if hasattr(result, 'block_until_ready'):
            result.block_until_ready()
        else:
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), result)
        times.append(time.perf_counter() - t0)

    median, iqr = compute_stats(times)
    return result, median, iqr, np.array(times)


def add_benchmark_args(parser=None):
    """Add standard benchmark CLI arguments.

    Args:
        parser: Existing ArgumentParser or None to create new one

    Returns:
        ArgumentParser with --n-reps and --backend args
    """
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('--n-reps', type=int, default=10,
                        help='Number of repetitions (default: 10 for paper)')
    parser.add_argument('--backend', type=str, default='gpu',
                        choices=['gpu', 'cpu', 'any'],
                        help='Expected backend (default: gpu)')
    return parser


# Constants
DEFAULT_N_REPS = 10  # Paper requires 10 reps for publication
