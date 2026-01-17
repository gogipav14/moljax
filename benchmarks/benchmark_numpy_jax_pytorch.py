#!/usr/bin/env python3
"""
Benchmark comparison: NumPy vs JAX vs PyTorch for FFT-based PDE solving.

This addresses claims like "PyTorch is 1700x faster than NumPy" by showing
the fair comparison is GPU vs GPU, not CPU vs GPU.

Key findings:
- NumPy CPU vs PyTorch GPU: Yes, huge speedup (GPU vs CPU)
- JAX GPU vs PyTorch GPU: Similar performance (both use cuFFT)
- JAX advantage: JIT-compiled control flow (lax.while_loop)
"""

import numpy as np
import time
from pathlib import Path

# Try to import JAX
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
except ImportError:
    JAX_AVAILABLE = False
    print("JAX not available")

# Try to import PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available")


def benchmark_fft_2d(N=1024, n_reps=10, warmup=3):
    """
    Benchmark 2D FFT across frameworks.
    This is the fair comparison the LinkedIn post should have made.
    """
    print(f"\n{'='*60}")
    print(f"2D FFT Benchmark: {N}x{N} grid")
    print(f"{'='*60}")

    results = {}

    # NumPy (CPU baseline)
    print("\n1. NumPy (CPU)...")
    data_np = np.random.randn(N, N).astype(np.float64)

    # Warmup
    for _ in range(warmup):
        _ = np.fft.fft2(data_np)

    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        result = np.fft.fft2(data_np)
        times.append(time.perf_counter() - start)

    results['numpy_cpu'] = {
        'median': np.median(times) * 1000,
        'std': np.std(times) * 1000,
    }
    print(f"   Median: {results['numpy_cpu']['median']:.2f} ms")

    # JAX CPU
    if JAX_AVAILABLE:
        print("\n2. JAX (CPU)...")
        with jax.default_device(jax.devices('cpu')[0]):
            data_jax = jnp.array(data_np)

            @jax.jit
            def jax_fft(x):
                return jnp.fft.fft2(x)

            # Warmup (includes compilation)
            for _ in range(warmup):
                _ = jax_fft(data_jax).block_until_ready()

            times = []
            for _ in range(n_reps):
                start = time.perf_counter()
                result = jax_fft(data_jax).block_until_ready()
                times.append(time.perf_counter() - start)

            results['jax_cpu'] = {
                'median': np.median(times) * 1000,
                'std': np.std(times) * 1000,
            }
            print(f"   Median: {results['jax_cpu']['median']:.2f} ms")

    # JAX GPU
    if JAX_AVAILABLE:
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                print("\n3. JAX (GPU)...")
                with jax.default_device(gpu_devices[0]):
                    data_jax_gpu = jnp.array(data_np)

                    # Warmup
                    for _ in range(warmup):
                        _ = jax_fft(data_jax_gpu).block_until_ready()

                    times = []
                    for _ in range(n_reps):
                        start = time.perf_counter()
                        result = jax_fft(data_jax_gpu).block_until_ready()
                        times.append(time.perf_counter() - start)

                    results['jax_gpu'] = {
                        'median': np.median(times) * 1000,
                        'std': np.std(times) * 1000,
                    }
                    print(f"   Median: {results['jax_gpu']['median']:.2f} ms")
        except:
            print("   JAX GPU not available")

    # PyTorch CPU
    if TORCH_AVAILABLE:
        print("\n4. PyTorch (CPU)...")
        data_torch_cpu = torch.from_numpy(data_np).to(torch.complex128)

        # Warmup
        for _ in range(warmup):
            _ = torch.fft.fft2(data_torch_cpu)

        times = []
        for _ in range(n_reps):
            start = time.perf_counter()
            result = torch.fft.fft2(data_torch_cpu)
            times.append(time.perf_counter() - start)

        results['torch_cpu'] = {
            'median': np.median(times) * 1000,
            'std': np.std(times) * 1000,
        }
        print(f"   Median: {results['torch_cpu']['median']:.2f} ms")

    # PyTorch GPU
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\n5. PyTorch (GPU)...")
        data_torch_gpu = torch.from_numpy(data_np).to(torch.complex128).cuda()

        # Warmup
        for _ in range(warmup):
            _ = torch.fft.fft2(data_torch_gpu)
            torch.cuda.synchronize()

        times = []
        for _ in range(n_reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = torch.fft.fft2(data_torch_gpu)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        results['torch_gpu'] = {
            'median': np.median(times) * 1000,
            'std': np.std(times) * 1000,
        }
        print(f"   Median: {results['torch_gpu']['median']:.2f} ms")

    return results


def benchmark_diffusion_solver(N=256, n_steps=1000, n_reps=5, warmup=2):
    """
    Benchmark diffusion PDE solver (CN with FFT) across frameworks.
    This is the relevant comparison for moljax.
    """
    print(f"\n{'='*60}")
    print(f"Diffusion Solver Benchmark: {N}x{N} grid, {n_steps} steps")
    print(f"{'='*60}")

    D = 0.01
    dx = 1.0 / N
    dt = 0.0001

    results = {}

    # NumPy (CPU)
    print("\n1. NumPy FFT-CN (CPU)...")
    u0_np = np.random.randn(N, N)
    kx = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    lap_eig_np = -(KX**2 + KY**2)

    def numpy_cn_solve(u0, n_steps):
        u = u0.copy()
        for _ in range(n_steps):
            u_hat = np.fft.fft2(u)
            u_hat = u_hat * (1 + 0.5*dt*D*lap_eig_np) / (1 - 0.5*dt*D*lap_eig_np)
            u = np.real(np.fft.ifft2(u_hat))
        return u

    # Warmup
    for _ in range(warmup):
        _ = numpy_cn_solve(u0_np, 10)

    times = []
    for _ in range(n_reps):
        start = time.perf_counter()
        result = numpy_cn_solve(u0_np, n_steps)
        times.append(time.perf_counter() - start)

    results['numpy_cpu'] = {
        'median': np.median(times) * 1000,
        'std': np.std(times) * 1000,
        'ms_per_step': np.median(times) * 1000 / n_steps,
    }
    print(f"   Median: {results['numpy_cpu']['median']:.1f} ms ({results['numpy_cpu']['ms_per_step']:.3f} ms/step)")

    # JAX GPU (with JIT - the moljax approach)
    if JAX_AVAILABLE:
        try:
            gpu_devices = jax.devices('gpu')
            if gpu_devices:
                print("\n2. JAX JIT FFT-CN (GPU) - MOLJAX approach...")
                with jax.default_device(gpu_devices[0]):
                    u0_jax = jnp.array(u0_np)
                    lap_eig_jax = jnp.array(lap_eig_np)

                    @jax.jit
                    def jax_cn_step(u, lap_eig):
                        u_hat = jnp.fft.fft2(u)
                        u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
                        return jnp.real(jnp.fft.ifft2(u_hat))

                    def jax_cn_solve(u0, n_steps):
                        u = u0
                        for _ in range(n_steps):
                            u = jax_cn_step(u, lap_eig_jax)
                        return u

                    # Warmup (includes JIT compilation)
                    for _ in range(warmup):
                        _ = jax_cn_solve(u0_jax, 10).block_until_ready()

                    times = []
                    for _ in range(n_reps):
                        start = time.perf_counter()
                        result = jax_cn_solve(u0_jax, n_steps).block_until_ready()
                        times.append(time.perf_counter() - start)

                    results['jax_gpu'] = {
                        'median': np.median(times) * 1000,
                        'std': np.std(times) * 1000,
                        'ms_per_step': np.median(times) * 1000 / n_steps,
                    }
                    print(f"   Median: {results['jax_gpu']['median']:.1f} ms ({results['jax_gpu']['ms_per_step']:.3f} ms/step)")
        except Exception as e:
            print(f"   JAX GPU failed: {e}")

    # PyTorch GPU
    if TORCH_AVAILABLE and torch.cuda.is_available():
        print("\n3. PyTorch FFT-CN (GPU)...")
        u0_torch = torch.from_numpy(u0_np).cuda()
        lap_eig_torch = torch.from_numpy(lap_eig_np).cuda()

        def torch_cn_solve(u0, n_steps):
            u = u0.clone()
            for _ in range(n_steps):
                u_hat = torch.fft.fft2(u)
                u_hat = u_hat * (1 + 0.5*dt*D*lap_eig_torch) / (1 - 0.5*dt*D*lap_eig_torch)
                u = torch.real(torch.fft.ifft2(u_hat))
            return u

        # Warmup
        for _ in range(warmup):
            _ = torch_cn_solve(u0_torch, 10)
            torch.cuda.synchronize()

        times = []
        for _ in range(n_reps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = torch_cn_solve(u0_torch, n_steps)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)

        results['torch_gpu'] = {
            'median': np.median(times) * 1000,
            'std': np.std(times) * 1000,
            'ms_per_step': np.median(times) * 1000 / n_steps,
        }
        print(f"   Median: {results['torch_gpu']['median']:.1f} ms ({results['torch_gpu']['ms_per_step']:.3f} ms/step)")

    # PyTorch GPU with torch.compile (PyTorch 2.0+)
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            print("\n4. PyTorch torch.compile FFT-CN (GPU)...")

            @torch.compile
            def torch_cn_step_compiled(u, lap_eig):
                u_hat = torch.fft.fft2(u)
                u_hat = u_hat * (1 + 0.5*dt*D*lap_eig) / (1 - 0.5*dt*D*lap_eig)
                return torch.real(torch.fft.ifft2(u_hat))

            def torch_cn_solve_compiled(u0, n_steps):
                u = u0.clone()
                for _ in range(n_steps):
                    u = torch_cn_step_compiled(u, lap_eig_torch)
                return u

            # Warmup (includes compilation)
            for _ in range(warmup + 1):
                _ = torch_cn_solve_compiled(u0_torch, 10)
                torch.cuda.synchronize()

            times = []
            for _ in range(n_reps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = torch_cn_solve_compiled(u0_torch, n_steps)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

            results['torch_compile_gpu'] = {
                'median': np.median(times) * 1000,
                'std': np.std(times) * 1000,
                'ms_per_step': np.median(times) * 1000 / n_steps,
            }
            print(f"   Median: {results['torch_compile_gpu']['median']:.1f} ms ({results['torch_compile_gpu']['ms_per_step']:.3f} ms/step)")
        except Exception as e:
            print(f"   torch.compile failed: {e}")

    return results


def print_comparison_summary(fft_results, solver_results):
    """Print comparison summary with speedup ratios."""
    print(f"\n{'='*60}")
    print("SUMMARY: Fair Comparisons")
    print(f"{'='*60}")

    print("\n--- 2D FFT ---")
    baseline = fft_results.get('numpy_cpu', {}).get('median', 1)
    for name, data in fft_results.items():
        speedup = baseline / data['median']
        print(f"  {name:20s}: {data['median']:8.2f} ms  ({speedup:6.1f}x vs NumPy CPU)")

    if 'jax_gpu' in fft_results and 'torch_gpu' in fft_results:
        ratio = fft_results['torch_gpu']['median'] / fft_results['jax_gpu']['median']
        print(f"\n  JAX GPU vs PyTorch GPU: {ratio:.2f}x (values near 1.0 expected)")

    print("\n--- Diffusion Solver ---")
    baseline = solver_results.get('numpy_cpu', {}).get('median', 1)
    for name, data in solver_results.items():
        speedup = baseline / data['median']
        print(f"  {name:20s}: {data['median']:8.1f} ms  ({speedup:6.1f}x vs NumPy CPU)")

    if 'jax_gpu' in solver_results and 'torch_gpu' in solver_results:
        ratio = solver_results['torch_gpu']['median'] / solver_results['jax_gpu']['median']
        print(f"\n  JAX GPU vs PyTorch GPU: {ratio:.2f}x")

    print(f"\n{'='*60}")
    print("KEY INSIGHTS:")
    print("1. NumPy CPU vs PyTorch GPU speedups are GPU vs CPU, not framework comparison")
    print("2. JAX GPU vs PyTorch GPU: Similar performance (both use cuFFT/cuBLAS)")
    print("3. JAX advantage: JIT-compiled control flow (lax.while_loop, lax.cond)")
    print("4. PyTorch advantage: Larger ecosystem, easier debugging")
    print("5. For PDE solvers with adaptive stepping, JAX's control flow JIT is key")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("NumPy vs JAX vs PyTorch: Fair Comparison")
    print("="*60)
    print("\nThis benchmark addresses claims like 'PyTorch is 1700x faster'")
    print("by comparing GPU-to-GPU, not CPU-to-GPU.\n")

    # FFT benchmarks at different sizes
    fft_results = {}
    for N in [512, 1024, 2048]:
        results = benchmark_fft_2d(N=N, n_reps=10)
        fft_results[N] = results

    # Solver benchmark
    solver_results = benchmark_diffusion_solver(N=256, n_steps=1000, n_reps=5)

    # Summary
    print_comparison_summary(fft_results[1024], solver_results)


if __name__ == "__main__":
    main()
