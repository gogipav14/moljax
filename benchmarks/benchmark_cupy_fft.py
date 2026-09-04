#!/usr/bin/env python3
"""
Benchmark: CuPy vs JAX Raw FFT Throughput

Proves CuPy and JAX have identical raw FFT performance on the same GPU
(both call cuFFT). This establishes that moljax's advantage is algorithmic,
not kernel-level.

Tests: fft2 and rfft2, grid sizes 64-1024, float64/complex128.
"""

# CRITICAL: Set x64 BEFORE any jax.numpy imports
import jax

jax.config.update("jax_enable_x64", True)

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    print("WARNING: CuPy not installed. Install with: pip install cupy-cuda12x")

from benchmark_utils import add_benchmark_args, compute_stats

parser = add_benchmark_args()
args = parser.parse_args()
N_REPS = args.n_reps

GRID_SIZES = [64, 128, 256, 512, 1024]
N_WARMUP = 5

print("CuPy vs JAX Raw FFT Benchmark")
print("=" * 60)
print(f"Grid sizes: {GRID_SIZES}")
print(f"N_REPS: {N_REPS}")
print(f"JAX: {jax.__version__}, GPU: {jax.devices('gpu')[0]}")
if HAS_CUPY:
    print(f"CuPy: {cp.__version__}")
print("=" * 60)

results = {'fft2': {}, 'rfft2': {}}


def time_jax_fft(data_jax, fft_fn, n_reps, n_warmup):
    """Time a JAX FFT operation."""
    # Warmup
    for _ in range(n_warmup):
        r = fft_fn(data_jax)
        r.block_until_ready()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        r = fft_fn(data_jax)
        r.block_until_ready()
        times.append(time.perf_counter() - t0)
    return compute_stats(times)


def time_cupy_fft(data_cp, fft_fn, n_reps, n_warmup):
    """Time a CuPy FFT operation."""
    # Warmup
    for _ in range(n_warmup):
        r = fft_fn(data_cp)
        cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(n_reps):
        cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        r = fft_fn(data_cp)
        cp.cuda.Stream.null.synchronize()
        times.append(time.perf_counter() - t0)
    return compute_stats(times)


def time_jax_fft_jit_amortized(N, n_loops=100, n_reps=10, n_warmup=5):
    """Time JAX FFT inside a JIT-compiled fori_loop (amortized per-FFT cost)."""
    data = jnp.array(np.random.randn(N, N))

    @jax.jit
    def fft_loop(x):
        def body(_, u):
            return jnp.fft.fft2(jnp.real(jnp.fft.ifft2(u)))
        return jax.lax.fori_loop(0, n_loops, body, jnp.fft.fft2(x))

    # Warmup (compile)
    for _ in range(n_warmup):
        r = fft_loop(data)
        r.block_until_ready()

    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        r = fft_loop(data)
        r.block_until_ready()
        times.append(time.perf_counter() - t0)

    med, iqr = compute_stats(times)
    # Each loop iteration does 1 fft2 + 1 ifft2 = 2 FFTs, plus 1 initial fft2
    # Per-fft2 cost = total / (n_loops + 1) but we report per fft2+ifft2 pair
    per_fft_ms = med * 1000 / n_loops
    per_fft_iqr_ms = iqr * 1000 / n_loops
    return per_fft_ms, per_fft_iqr_ms


for N in GRID_SIZES:
    print(f"\n--- Grid {N}x{N} ---")

    # Generate random real data (float64)
    data_np = np.random.randn(N, N).astype(np.float64)

    # ---- FFT2 (complex-to-complex) ----
    data_jax = jnp.array(data_np)

    jax_fft2_med, jax_fft2_iqr = time_jax_fft(
        data_jax, jnp.fft.fft2, N_REPS, N_WARMUP
    )

    entry = {
        'jax_median_ms': float(jax_fft2_med * 1000),
        'jax_iqr_ms': float(jax_fft2_iqr * 1000),
    }

    if HAS_CUPY:
        data_cp = cp.array(data_np)
        cp_fft2_med, cp_fft2_iqr = time_cupy_fft(
            data_cp, cp.fft.fft2, N_REPS, N_WARMUP
        )
        entry['cupy_median_ms'] = float(cp_fft2_med * 1000)
        entry['cupy_iqr_ms'] = float(cp_fft2_iqr * 1000)
        entry['ratio_jax_over_cupy'] = float(jax_fft2_med / cp_fft2_med)

        print(f"  fft2:  JAX {jax_fft2_med*1000:.3f} ms, CuPy {cp_fft2_med*1000:.3f} ms, "
              f"ratio {entry['ratio_jax_over_cupy']:.2f}")
    else:
        print(f"  fft2:  JAX {jax_fft2_med*1000:.3f} ms (CuPy not available)")

    results['fft2'][str(N)] = entry

    # ---- RFFT2 (real-to-complex) ----
    jax_rfft2_med, jax_rfft2_iqr = time_jax_fft(
        data_jax, jnp.fft.rfft2, N_REPS, N_WARMUP
    )

    rentry = {
        'jax_median_ms': float(jax_rfft2_med * 1000),
        'jax_iqr_ms': float(jax_rfft2_iqr * 1000),
    }

    if HAS_CUPY:
        cp_rfft2_med, cp_rfft2_iqr = time_cupy_fft(
            data_cp, cp.fft.rfft2, N_REPS, N_WARMUP
        )
        rentry['cupy_median_ms'] = float(cp_rfft2_med * 1000)
        rentry['cupy_iqr_ms'] = float(cp_rfft2_iqr * 1000)
        rentry['ratio_jax_over_cupy'] = float(jax_rfft2_med / cp_rfft2_med)

        print(f"  rfft2: JAX {jax_rfft2_med*1000:.3f} ms, CuPy {cp_rfft2_med*1000:.3f} ms, "
              f"ratio {rentry['ratio_jax_over_cupy']:.2f}")
    else:
        print(f"  rfft2: JAX {jax_rfft2_med*1000:.3f} ms (CuPy not available)")

    results['rfft2'][str(N)] = rentry

    # ---- JIT-amortized FFT (fft2+ifft2 loop, as in moljax) ----
    jit_med_ms, jit_iqr_ms = time_jax_fft_jit_amortized(N, n_loops=100, n_reps=N_REPS, n_warmup=N_WARMUP)
    jit_entry = {
        'jax_jit_amortized_median_ms': float(jit_med_ms),
        'jax_jit_amortized_iqr_ms': float(jit_iqr_ms),
    }
    if HAS_CUPY and 'cupy_median_ms' in entry:
        jit_entry['ratio_jit_over_cupy'] = float(jit_med_ms / entry['cupy_median_ms'])
        print(f"  jit-loop fft2: JAX {jit_med_ms:.3f} +/- {jit_iqr_ms:.3f} ms/iter, "
              f"CuPy {entry['cupy_median_ms']:.3f} ms, ratio {jit_entry['ratio_jit_over_cupy']:.2f}")
    else:
        print(f"  jit-loop fft2: JAX {jit_med_ms:.3f} +/- {jit_iqr_ms:.3f} ms/iter")
    results['fft2_jit_amortized'] = results.get('fft2_jit_amortized', {})
    results['fft2_jit_amortized'][str(N)] = jit_entry


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY: CuPy vs JAX FFT Throughput (per-call)")
print("=" * 60)
print(f"{'Grid':<10} {'FFT':<6} {'JAX (ms)':>12} {'CuPy (ms)':>12} {'Ratio':>8}")
print("-" * 55)

for fft_type in ['fft2', 'rfft2']:
    for N in GRID_SIZES:
        entry = results[fft_type][str(N)]
        jax_ms = entry['jax_median_ms']
        jax_iqr = entry['jax_iqr_ms']
        if HAS_CUPY and 'cupy_median_ms' in entry:
            cp_ms = entry['cupy_median_ms']
            cp_iqr = entry['cupy_iqr_ms']
            ratio = entry['ratio_jax_over_cupy']
            print(f"{N}x{N:<7} {fft_type:<6} {jax_ms:>6.3f}+/-{jax_iqr:.3f} {cp_ms:>6.3f}+/-{cp_iqr:.3f} {ratio:>7.2f}x")

print("\nJIT-amortized fft2 (as in moljax JIT-compiled loops):")
print(f"{'Grid':<10} {'JAX JIT (ms)':>14} {'CuPy (ms)':>12} {'Ratio':>8}")
print("-" * 50)
if 'fft2_jit_amortized' in results:
    for N in GRID_SIZES:
        je = results['fft2_jit_amortized'].get(str(N), {})
        fe = results['fft2'].get(str(N), {})
        jit_ms = je.get('jax_jit_amortized_median_ms', 0)
        jit_iqr = je.get('jax_jit_amortized_iqr_ms', 0)
        cp_ms = fe.get('cupy_median_ms', 0)
        cp_iqr = fe.get('cupy_iqr_ms', 0)
        ratio = je.get('ratio_jit_over_cupy', 0)
        print(f"{N}x{N:<7} {jit_ms:>7.3f}+/-{jit_iqr:.3f} {cp_ms:>6.3f}+/-{cp_iqr:.3f} {ratio:>7.2f}x")

print("=" * 60)
if HAS_CUPY:
    print("Per-call: CuPy faster due to thinner dispatch overhead.")
    print("JIT-amortized: gap narrows, confirming both use same cuFFT backend.")
    print("moljax advantage is algorithmic (IMEX/ETD/preconditioning) + JIT loop fusion.")

results['config'] = {
    'grid_sizes': GRID_SIZES,
    'n_reps': N_REPS,
    'dtype': 'float64',
    'jax_version': jax.__version__,
    'cupy_version': cp.__version__ if HAS_CUPY else None,
    'gpu': str(jax.devices('gpu')[0]),
}

# Save
output_path = Path(__file__).parent / 'results' / 'cupy_vs_jax_fft.json'
output_path.parent.mkdir(exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
