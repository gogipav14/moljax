#!/usr/bin/env python3
"""
Benchmark: GPU FFT throughput across CuPy, nvmath-python and JAX.

Reproduces Table 3 (Section 5.1.4) of

    Pavlov, G. "moljax: GPU-accelerated method of lines for stiff
    reaction-diffusion PDEs with FFT preconditioning."
    Computer Physics Communications 326 (2026) 110205.
    doi:10.1016/j.cpc.2026.110205

Four columns are measured on the same hardware in float64/complex128:

    cupy       per-call fft2 from Python (thin wrapper around cuFFT)
    nvmath     per-call fft2 via nvmath-python (also wraps cuFFT)
    jax_eager  per-call fft2 from Python under JAX
    jax_jit    fft2 inside a JIT-compiled lax.fori_loop, which is how
               moljax actually executes FFTs during time stepping

The point of the table is that JAX inside a compiled loop is
competitive with CuPy, so moljax's speedups are algorithmic rather than
kernel-level.

Writes benchmarks/results/fft_3lib_comparison.json.

CuPy and nvmath are optional; missing libraries are reported as null and
their columns are skipped rather than aborting the run.
"""

import jax

jax.config.update("jax_enable_x64", True)

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import lax

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    cp = None
    HAS_CUPY = False
    print("WARNING: CuPy not installed; skipping the cupy column.")
    print("         pip install cupy-cuda13x")

try:
    import nvmath.fft as nvfft
    HAS_NVMATH = True
except ImportError:
    nvfft = None
    HAS_NVMATH = False
    print("WARNING: nvmath-python not installed; skipping the nvmath column.")
    print("         pip install nvmath-python")

GRID_SIZES = [64, 128, 256, 512, 1024]
LOOP_ITERS = 100


def stats(times_ms):
    """Median and interquartile range, matching the paper's reporting."""
    a = np.asarray(times_ms, dtype=float)
    q1, q3 = np.percentile(a, [25, 75])
    return {"median_ms": round(float(np.median(a)), 3),
            "iqr_ms": round(float(q3 - q1), 3)}


def time_cupy(n, n_reps, n_warmup):
    a = cp.random.rand(n, n) + 1j * cp.random.rand(n, n)
    for _ in range(n_warmup):
        cp.fft.fft2(a)
    cp.cuda.Device().synchronize()
    out = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        cp.fft.fft2(a)
        cp.cuda.Device().synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_nvmath(n, n_reps, n_warmup):
    a = cp.random.rand(n, n) + 1j * cp.random.rand(n, n)
    for _ in range(n_warmup):
        nvfft.fft(a)
    cp.cuda.Device().synchronize()
    out = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        nvfft.fft(a)
        cp.cuda.Device().synchronize()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_jax_eager(n, n_reps, n_warmup, key):
    a = jax.random.normal(key, (n, n), dtype=jnp.float64).astype(jnp.complex128)
    for _ in range(n_warmup):
        jnp.fft.fft2(a).block_until_ready()
    out = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jnp.fft.fft2(a).block_until_ready()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def time_jax_jit(n, n_reps, n_warmup, key):
    """fft2 inside a compiled fori_loop; reported per FFT."""
    a = jax.random.normal(key, (n, n), dtype=jnp.float64).astype(jnp.complex128)

    @jax.jit
    def loop(x):
        def body(_, acc):
            return jnp.fft.fft2(acc)
        return lax.fori_loop(0, LOOP_ITERS, body, x)

    loop(a).block_until_ready()  # compile
    for _ in range(n_warmup):
        loop(a).block_until_ready()
    out = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        loop(a).block_until_ready()
        out.append((time.perf_counter() - t0) * 1e3 / LOOP_ITERS)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-reps", type=int, default=200)
    p.add_argument("--n-warmup", type=int, default=20)
    args = p.parse_args()

    print(f"Devices: {jax.devices()}")
    print(f"CuPy: {'yes' if HAS_CUPY else 'no'}   nvmath: {'yes' if HAS_NVMATH else 'no'}")
    print(f"reps={args.n_reps} warmup={args.n_warmup} loop_iters={LOOP_ITERS}\n")

    key = jax.random.PRNGKey(0)
    results = {}

    header = f"{'grid':>8} {'cupy':>16} {'nvmath':>16} {'jax_eager':>16} {'jax_jit':>16}"
    print(header)
    print("-" * len(header))

    for n in GRID_SIZES:
        key, k1, k2 = jax.random.split(key, 3)
        entry = {}

        entry["cupy"] = stats(time_cupy(n, args.n_reps, args.n_warmup)) if HAS_CUPY else None
        entry["nvmath"] = (
            stats(time_nvmath(n, args.n_reps, args.n_warmup))
            if (HAS_NVMATH and HAS_CUPY) else None
        )
        entry["jax_eager"] = stats(time_jax_eager(n, args.n_reps, args.n_warmup, k1))
        entry["jax_jit"] = stats(time_jax_jit(n, args.n_reps, args.n_warmup, k2))

        results[str(n)] = entry

        def fmt(col):
            v = entry[col]
            return "n/a" if v is None else f"{v['median_ms']:.3f}+-{v['iqr_ms']:.3f}"

        print(f"{n:>6}^2 {fmt('cupy'):>16} {fmt('nvmath'):>16} "
              f"{fmt('jax_eager'):>16} {fmt('jax_jit'):>16}")

    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "fft_3lib_comparison.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
