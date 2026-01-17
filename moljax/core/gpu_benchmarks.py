"""
GPU acceleration benchmarks for FFT operations.

This module provides utilities to benchmark FFT operations across different
devices (CPU, GPU) and grid sizes to measure scaling performance.

Key metrics:
- CPU vs GPU speedup
- Scaling with grid size (target: 100x speedup for N>2048)
- Memory transfer overhead
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import jax
import jax.numpy as jnp

# Enable float64 for precision
jax.config.update("jax_enable_x64", True)


@dataclass
class DeviceInfo:
    """Information about available JAX devices."""
    cpu_count: int
    gpu_count: int
    tpu_count: int
    default_backend: str
    gpu_names: List[str]

    @classmethod
    def detect(cls) -> "DeviceInfo":
        """Detect available JAX devices."""
        cpus = jax.devices("cpu")
        try:
            gpus = jax.devices("gpu")
        except RuntimeError:
            gpus = []
        try:
            tpus = jax.devices("tpu")
        except RuntimeError:
            tpus = []

        gpu_names = [str(g) for g in gpus]

        return cls(
            cpu_count=len(cpus),
            gpu_count=len(gpus),
            tpu_count=len(tpus),
            default_backend=jax.default_backend(),
            gpu_names=gpu_names,
        )

    def has_gpu(self) -> bool:
        """Check if GPU is available."""
        return self.gpu_count > 0

    def __str__(self) -> str:
        lines = [
            f"Default backend: {self.default_backend}",
            f"CPUs: {self.cpu_count}",
            f"GPUs: {self.gpu_count}",
            f"TPUs: {self.tpu_count}",
        ]
        if self.gpu_names:
            lines.append(f"GPU devices: {', '.join(self.gpu_names)}")
        return "\n".join(lines)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    operation: str
    grid_size: int
    device: str
    first_call_ms: float
    avg_ms: float
    std_ms: float
    n_iterations: int

    @property
    def ops_per_sec(self) -> float:
        """Operations per second."""
        return 1000.0 / self.avg_ms if self.avg_ms > 0 else 0.0


@dataclass
class ScalingBenchmark:
    """Benchmark results across multiple grid sizes."""
    operation: str
    device: str
    grid_sizes: List[int]
    times_ms: List[float]
    first_call_ms: List[float]

    def print_table(self):
        """Print scaling table."""
        print(f"\n{self.operation} - {self.device}")
        print("-" * 60)
        print(f"{'Grid Size':>10} | {'First Call (ms)':>15} | {'Avg (ms)':>12} | {'Ops/sec':>10}")
        print("-" * 60)
        for n, first, avg in zip(self.grid_sizes, self.first_call_ms, self.times_ms):
            ops = 1000.0 / avg if avg > 0 else 0
            print(f"{n:>10} | {first:>15.2f} | {avg:>12.4f} | {ops:>10.1f}")


def _benchmark_operation(
    op_fn,
    setup_fn,
    grid_size: int,
    n_warmup: int = 5,
    n_iterations: int = 50,
    device: Optional[str] = None,
) -> BenchmarkResult:
    """Benchmark a single operation.

    Args:
        op_fn: Function to benchmark, takes arrays from setup_fn
        setup_fn: Function(grid_size) -> arrays to pass to op_fn
        grid_size: Grid size for setup
        n_warmup: Number of warmup iterations
        n_iterations: Number of timed iterations
        device: Device string (cpu/gpu) or None for default

    Returns:
        BenchmarkResult with timing statistics
    """
    # Setup arrays
    arrays = setup_fn(grid_size)

    # First call (includes compilation)
    t0 = time.perf_counter()
    result = op_fn(*arrays)
    jax.block_until_ready(result)
    first_call_ms = (time.perf_counter() - t0) * 1000

    # Warmup
    for _ in range(n_warmup):
        result = op_fn(*arrays)
    jax.block_until_ready(result)

    # Timed iterations
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        result = op_fn(*arrays)
        jax.block_until_ready(result)
        times.append((time.perf_counter() - t0) * 1000)

    times_arr = jnp.array(times)
    avg_ms = float(jnp.mean(times_arr))
    std_ms = float(jnp.std(times_arr))

    device_name = device or jax.default_backend()

    return BenchmarkResult(
        operation=op_fn.__name__ if hasattr(op_fn, '__name__') else 'unknown',
        grid_size=grid_size,
        device=device_name,
        first_call_ms=first_call_ms,
        avg_ms=avg_ms,
        std_ms=std_ms,
        n_iterations=n_iterations,
    )


def benchmark_fft_scaling(
    grid_sizes: List[int] = None,
    n_warmup: int = 3,
    n_iterations: int = 20,
) -> Dict[str, ScalingBenchmark]:
    """Benchmark FFT operations across grid sizes.

    Args:
        grid_sizes: List of grid sizes to test (default: powers of 2)
        n_warmup: Warmup iterations
        n_iterations: Timed iterations

    Returns:
        Dictionary of operation name -> ScalingBenchmark
    """
    from moljax.core.jit_kernels import helmholtz_solve_1d, etd1_kernel_1d
    from moljax.core.fft_solvers import laplacian_symbol_1d

    if grid_sizes is None:
        grid_sizes = [64, 128, 256, 512, 1024, 2048, 4096]

    device = jax.default_backend()
    results = {}

    # Helmholtz solve benchmark
    def helmholtz_setup(n):
        dx = 1.0 / n
        lap_sym = laplacian_symbol_1d(n, dx)
        x = jnp.linspace(0, 1, n, endpoint=False)
        rhs = jnp.sin(2 * jnp.pi * x)
        return (rhs, lap_sym, 0.01, 0.01)

    @jax.jit
    def helmholtz_op(rhs, lap_sym, dt, D):
        return helmholtz_solve_1d(rhs, lap_sym, dt, D)

    helm_times = []
    helm_first = []
    for n in grid_sizes:
        result = _benchmark_operation(
            helmholtz_op, helmholtz_setup, n,
            n_warmup=n_warmup, n_iterations=n_iterations
        )
        helm_times.append(result.avg_ms)
        helm_first.append(result.first_call_ms)

    results['helmholtz_1d'] = ScalingBenchmark(
        operation='Helmholtz Solve 1D',
        device=device,
        grid_sizes=grid_sizes,
        times_ms=helm_times,
        first_call_ms=helm_first,
    )

    # ETD1 benchmark
    def etd1_setup(n):
        dx = 1.0 / n
        D = 0.01
        lap_sym = laplacian_symbol_1d(n, dx)
        eigenvalues = D * lap_sym
        x = jnp.linspace(0, 1, n, endpoint=False)
        u = jnp.sin(2 * jnp.pi * x)
        N = 0.1 * u**2
        return (u, N, eigenvalues, 0.01)

    @jax.jit
    def etd1_op(u, N, eigenvalues, dt):
        return etd1_kernel_1d(u, N, eigenvalues, dt)

    etd1_times = []
    etd1_first = []
    for n in grid_sizes:
        result = _benchmark_operation(
            etd1_op, etd1_setup, n,
            n_warmup=n_warmup, n_iterations=n_iterations
        )
        etd1_times.append(result.avg_ms)
        etd1_first.append(result.first_call_ms)

    results['etd1_1d'] = ScalingBenchmark(
        operation='ETD1 Kernel 1D',
        device=device,
        grid_sizes=grid_sizes,
        times_ms=etd1_times,
        first_call_ms=etd1_first,
    )

    # Raw FFT benchmark for comparison
    def fft_setup(n):
        x = jnp.linspace(0, 1, n, endpoint=False)
        u = jnp.sin(2 * jnp.pi * x) + 0.5 * jnp.cos(4 * jnp.pi * x)
        return (u,)

    @jax.jit
    def fft_op(u):
        return jnp.fft.fft(u)

    fft_times = []
    fft_first = []
    for n in grid_sizes:
        result = _benchmark_operation(
            fft_op, fft_setup, n,
            n_warmup=n_warmup, n_iterations=n_iterations
        )
        fft_times.append(result.avg_ms)
        fft_first.append(result.first_call_ms)

    results['fft_1d'] = ScalingBenchmark(
        operation='Raw FFT 1D',
        device=device,
        grid_sizes=grid_sizes,
        times_ms=fft_times,
        first_call_ms=fft_first,
    )

    return results


def benchmark_2d_scaling(
    grid_sizes: List[int] = None,
    n_warmup: int = 3,
    n_iterations: int = 20,
) -> Dict[str, ScalingBenchmark]:
    """Benchmark 2D FFT operations.

    Args:
        grid_sizes: List of grid sizes (N for NxN grid)
        n_warmup: Warmup iterations
        n_iterations: Timed iterations

    Returns:
        Dictionary of operation name -> ScalingBenchmark
    """
    from moljax.core.jit_kernels import helmholtz_solve_2d, etd1_kernel_2d
    from moljax.core.fft_solvers import laplacian_symbol_2d

    if grid_sizes is None:
        grid_sizes = [32, 64, 128, 256, 512]

    device = jax.default_backend()
    results = {}

    # 2D Helmholtz
    def helmholtz_2d_setup(n):
        dx = dy = 1.0 / n
        lap_sym = laplacian_symbol_2d(n, n, dx, dy)
        x = jnp.linspace(0, 1, n, endpoint=False)
        y = jnp.linspace(0, 1, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        rhs = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)
        return (rhs, lap_sym, 0.01, 0.01)

    @jax.jit
    def helmholtz_2d_op(rhs, lap_sym, dt, D):
        return helmholtz_solve_2d(rhs, lap_sym, dt, D)

    helm_times = []
    helm_first = []
    for n in grid_sizes:
        result = _benchmark_operation(
            helmholtz_2d_op, helmholtz_2d_setup, n,
            n_warmup=n_warmup, n_iterations=n_iterations
        )
        helm_times.append(result.avg_ms)
        helm_first.append(result.first_call_ms)

    results['helmholtz_2d'] = ScalingBenchmark(
        operation='Helmholtz Solve 2D',
        device=device,
        grid_sizes=grid_sizes,
        times_ms=helm_times,
        first_call_ms=helm_first,
    )

    # 2D FFT
    def fft2_setup(n):
        x = jnp.linspace(0, 1, n, endpoint=False)
        y = jnp.linspace(0, 1, n, endpoint=False)
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u = jnp.sin(2 * jnp.pi * X) * jnp.cos(2 * jnp.pi * Y)
        return (u,)

    @jax.jit
    def fft2_op(u):
        return jnp.fft.fft2(u)

    fft_times = []
    fft_first = []
    for n in grid_sizes:
        result = _benchmark_operation(
            fft2_op, fft2_setup, n,
            n_warmup=n_warmup, n_iterations=n_iterations
        )
        fft_times.append(result.avg_ms)
        fft_first.append(result.first_call_ms)

    results['fft_2d'] = ScalingBenchmark(
        operation='Raw FFT 2D',
        device=device,
        grid_sizes=grid_sizes,
        times_ms=fft_times,
        first_call_ms=fft_first,
    )

    return results


def run_full_benchmark(
    grid_sizes_1d: List[int] = None,
    grid_sizes_2d: List[int] = None,
    n_iterations: int = 20,
) -> Dict[str, Any]:
    """Run full benchmark suite.

    Returns:
        Dictionary with device info and all benchmark results
    """
    # Detect devices
    device_info = DeviceInfo.detect()

    print("=" * 70)
    print("FFT Performance Benchmark Suite")
    print("=" * 70)
    print()
    print("Device Information:")
    print(device_info)
    print()

    # 1D benchmarks
    print("-" * 70)
    print("1D Benchmarks")
    print("-" * 70)
    results_1d = benchmark_fft_scaling(grid_sizes_1d, n_iterations=n_iterations)

    for name, bench in results_1d.items():
        bench.print_table()

    # 2D benchmarks
    print()
    print("-" * 70)
    print("2D Benchmarks")
    print("-" * 70)
    results_2d = benchmark_2d_scaling(grid_sizes_2d, n_iterations=n_iterations)

    for name, bench in results_2d.items():
        bench.print_table()

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    if device_info.has_gpu():
        print(f"GPU acceleration available: {device_info.gpu_names[0]}")
        print("For large grids (N>2048), expect 10-100x speedup vs CPU")
    else:
        print("No GPU detected - running on CPU")
        print("To enable GPU acceleration, install JAX with CUDA support:")
        print("  pip install --upgrade 'jax[cuda12_pip]' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html")

    return {
        'device_info': device_info,
        '1d': results_1d,
        '2d': results_2d,
    }


def print_benchmark_report(results: Dict[str, Any]):
    """Print formatted benchmark report."""
    device_info = results['device_info']

    print("\n" + "=" * 70)
    print("BENCHMARK REPORT: FFT Performance")
    print("=" * 70)
    print(f"\nBackend: {device_info.default_backend}")
    print(f"GPUs: {device_info.gpu_count}")
    print()

    # 1D table
    print("1D Operations (time in ms):")
    print("-" * 70)

    results_1d = results['1d']
    grid_sizes = results_1d['fft_1d'].grid_sizes

    header = f"{'N':>8}"
    for name in ['fft_1d', 'helmholtz_1d', 'etd1_1d']:
        header += f" | {name:>12}"
    print(header)
    print("-" * 70)

    for i, n in enumerate(grid_sizes):
        row = f"{n:>8}"
        for name in ['fft_1d', 'helmholtz_1d', 'etd1_1d']:
            t = results_1d[name].times_ms[i]
            row += f" | {t:>12.4f}"
        print(row)

    # 2D table
    print()
    print("2D Operations (time in ms):")
    print("-" * 70)

    results_2d = results['2d']
    grid_sizes_2d = results_2d['fft_2d'].grid_sizes

    header = f"{'NxN':>8}"
    for name in ['fft_2d', 'helmholtz_2d']:
        header += f" | {name:>15}"
    print(header)
    print("-" * 70)

    for i, n in enumerate(grid_sizes_2d):
        row = f"{n}x{n}".rjust(8)
        for name in ['fft_2d', 'helmholtz_2d']:
            t = results_2d[name].times_ms[i]
            row += f" | {t:>15.4f}"
        print(row)


if __name__ == "__main__":
    results = run_full_benchmark()
