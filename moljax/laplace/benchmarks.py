"""
Benchmark harness for NILT performance evaluation.

Compares NILT against MOL time stepping and generates
performance reports saved as NPZ files.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp

from .nilt_fft import nilt_fft_uniform, NILTResult
from .tuning import tune_nilt_params, TunedNILTParams
from .diagnostics import compute_nilt_diagnostics, compare_to_reference
from .spectral_bounds import BoundContext, compute_spectral_bounds


@dataclass
class BenchmarkConfig:
    """Configuration for NILT benchmark run."""
    name: str
    t_end: float
    F_eval: Callable[[jnp.ndarray], jnp.ndarray]
    f_reference: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    bounds: Any = None  # SpectralBounds, dict, or BoundContext
    manual_params: dict | None = None  # Override autotuned params
    output_dir: Path | None = None


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    config: BenchmarkConfig
    tuned_params: TunedNILTParams
    nilt_result: NILTResult
    diagnostics: dict
    timing: dict
    errors: dict | None = None
    comparison: dict | None = None


def run_nilt_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """
    Run a single NILT benchmark.

    Args:
        config: Benchmark configuration

    Returns:
        BenchmarkResult with all metrics
    """
    # Timing dictionary
    timing = {}

    # Step 1: Tune parameters
    t0 = time.perf_counter()

    if config.manual_params is not None:
        # Use manual parameters
        tuned = TunedNILTParams(
            dt=config.manual_params.get('dt', 0.01),
            N=config.manual_params.get('N', 1024),
            T=config.manual_params.get('T', config.t_end * 2),
            a=config.manual_params.get('a', 0.5),
            omega_max=jnp.pi / config.manual_params.get('dt', 0.01),
            omega_req=0.0,
            bound_sources={'all': 'manual'},
            warnings=['Using manual parameters'],
            diagnostics={}
        )
    elif config.bounds is not None:
        if isinstance(config.bounds, BoundContext):
            bounds = compute_spectral_bounds(config.bounds)
        else:
            bounds = config.bounds
        tuned = tune_nilt_params(t_end=config.t_end, bounds=bounds)
    else:
        # Auto-tune with defaults
        tuned = tune_nilt_params(t_end=config.t_end)

    timing['tuning'] = time.perf_counter() - t0

    # Step 2: Run NILT
    t0 = time.perf_counter()
    nilt_result = nilt_fft_uniform(
        config.F_eval,
        dt=tuned.dt,
        N=tuned.N,
        a=tuned.a
    )
    timing['nilt_compute'] = time.perf_counter() - t0

    # Step 3: Compute diagnostics
    t0 = time.perf_counter()
    diag = compute_nilt_diagnostics(
        config.F_eval,
        nilt_result.f,
        nilt_result.t,
        tuned.a,
        tuned.T,
        config.t_end
    )
    diagnostics = {
        'frequency_coverage': diag.frequency_coverage,
        'frequency_slope': diag.frequency_slope,
        'ringing_metric': diag.ringing_metric,
        'wraparound_metric': diag.wraparound_metric,
        'energy_ratio': diag.energy_ratio,
        'quality_score': diag.quality_score,
        'warnings': diag.warnings
    }
    timing['diagnostics'] = time.perf_counter() - t0

    # Step 4: Compare to reference if available
    errors = None
    comparison = None
    if config.f_reference is not None:
        t0 = time.perf_counter()
        f_ref = config.f_reference(nilt_result.t)
        errors = compare_to_reference(
            nilt_result.f, f_ref, nilt_result.t, config.t_end
        )
        comparison = {
            't': nilt_result.t,
            'f_computed': nilt_result.f,
            'f_reference': f_ref,
            'abs_error': jnp.abs(nilt_result.f - f_ref)
        }
        timing['comparison'] = time.perf_counter() - t0

    timing['total'] = sum(timing.values())

    return BenchmarkResult(
        name=config.name,
        config=config,
        tuned_params=tuned,
        nilt_result=nilt_result,
        diagnostics=diagnostics,
        timing=timing,
        errors=errors,
        comparison=comparison
    )


def run_sensitivity_benchmark(
    config: BenchmarkConfig,
    dt_factors: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2, 2.0),
    a_factors: tuple[float, ...] = (0.5, 0.8, 1.0, 1.5, 2.0),
    N_factors: tuple[int, ...] = (256, 512, 1024, 2048, 4096)
) -> dict:
    """
    Run sensitivity study varying NILT parameters.

    Args:
        config: Base benchmark configuration
        dt_factors: Factors to multiply baseline dt
        a_factors: Factors to multiply baseline a
        N_factors: FFT sizes to test

    Returns:
        Dict with error surfaces for each parameter
    """
    # Get baseline parameters
    if config.bounds is not None:
        if isinstance(config.bounds, BoundContext):
            bounds = compute_spectral_bounds(config.bounds)
        else:
            bounds = config.bounds
        base_params = tune_nilt_params(t_end=config.t_end, bounds=bounds)
    else:
        base_params = tune_nilt_params(t_end=config.t_end)

    base_dt = base_params.dt
    base_a = base_params.a
    base_N = base_params.N

    results = {
        'base_params': {
            'dt': base_dt,
            'a': base_a,
            'N': base_N
        },
        'dt_sweep': {'factors': dt_factors, 'errors': []},
        'a_sweep': {'factors': a_factors, 'errors': []},
        'N_sweep': {'sizes': N_factors, 'errors': []}
    }

    # Sweep dt
    for factor in dt_factors:
        dt = base_dt * factor
        nilt_result = nilt_fft_uniform(
            config.F_eval, dt=dt, N=base_N, a=base_a
        )
        if config.f_reference is not None:
            f_ref = config.f_reference(nilt_result.t)
            mask = nilt_result.t <= config.t_end
            error = float(jnp.sqrt(jnp.mean((nilt_result.f[mask] - f_ref[mask])**2)))
        else:
            error = float('nan')
        results['dt_sweep']['errors'].append(error)

    # Sweep a
    for factor in a_factors:
        a = base_a * factor if base_a > 0 else factor * 0.1
        nilt_result = nilt_fft_uniform(
            config.F_eval, dt=base_dt, N=base_N, a=a
        )
        if config.f_reference is not None:
            f_ref = config.f_reference(nilt_result.t)
            mask = nilt_result.t <= config.t_end
            error = float(jnp.sqrt(jnp.mean((nilt_result.f[mask] - f_ref[mask])**2)))
        else:
            error = float('nan')
        results['a_sweep']['errors'].append(error)

    # Sweep N
    for N in N_factors:
        # Adjust dt to maintain similar T
        T = base_params.T
        dt = 2 * T / N
        nilt_result = nilt_fft_uniform(
            config.F_eval, dt=dt, N=N, a=base_a
        )
        if config.f_reference is not None:
            f_ref = config.f_reference(nilt_result.t)
            mask = nilt_result.t <= config.t_end
            error = float(jnp.sqrt(jnp.mean((nilt_result.f[mask] - f_ref[mask])**2)))
        else:
            error = float('nan')
        results['N_sweep']['errors'].append(error)

    return results


def save_benchmark_results(
    result: BenchmarkResult,
    output_path: Path | str
) -> None:
    """
    Save benchmark results to NPZ file.

    Args:
        result: BenchmarkResult to save
        output_path: Path for output file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'name': result.name,
        't': result.nilt_result.t,
        'f': result.nilt_result.f,
        'dt': result.tuned_params.dt,
        'N': result.tuned_params.N,
        'T': result.tuned_params.T,
        'a': result.tuned_params.a,
        'omega_max': result.tuned_params.omega_max,
        'omega_req': result.tuned_params.omega_req,
        'quality_score': result.diagnostics['quality_score'],
        'ringing_metric': result.diagnostics['ringing_metric'],
        'wraparound_metric': result.diagnostics['wraparound_metric'],
        'timing_total': result.timing['total'],
        'timing_nilt': result.timing['nilt_compute'],
    }

    if result.errors is not None:
        save_dict['l2_error'] = result.errors['l2_error']
        save_dict['linf_error'] = result.errors['linf_error']
        save_dict['l2_relative'] = result.errors['l2_relative']

    if result.comparison is not None:
        save_dict['f_reference'] = result.comparison['f_reference']
        save_dict['abs_error'] = result.comparison['abs_error']

    jnp.savez(str(output_path), **save_dict)


def save_sensitivity_results(
    results: dict,
    output_path: Path | str
) -> None:
    """
    Save sensitivity study results to NPZ file.

    Args:
        results: Sensitivity study results dict
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        'base_dt': results['base_params']['dt'],
        'base_a': results['base_params']['a'],
        'base_N': results['base_params']['N'],
        'dt_factors': jnp.array(results['dt_sweep']['factors']),
        'dt_errors': jnp.array(results['dt_sweep']['errors']),
        'a_factors': jnp.array(results['a_sweep']['factors']),
        'a_errors': jnp.array(results['a_sweep']['errors']),
        'N_sizes': jnp.array(results['N_sweep']['sizes']),
        'N_errors': jnp.array(results['N_sweep']['errors']),
    }

    jnp.savez(str(output_path), **save_dict)


# =============================================================================
# Comparison with MOL time stepping
# =============================================================================

def compare_nilt_vs_mol(
    nilt_F_eval: Callable[[jnp.ndarray], jnp.ndarray],
    mol_integrate: Callable[[float], tuple[jnp.ndarray, jnp.ndarray]],
    t_end: float,
    reference_f: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    nilt_bounds: Any = None
) -> dict:
    """
    Compare NILT and MOL time stepping performance.

    Args:
        nilt_F_eval: Laplace domain function for NILT
        mol_integrate: Function(t_end) -> (t_array, u_array) for MOL
        t_end: End time for comparison
        reference_f: Optional analytic reference
        nilt_bounds: Spectral bounds for NILT tuning

    Returns:
        Dict with comparison metrics
    """
    results = {
        'nilt': {},
        'mol': {},
        'comparison': {}
    }

    # Run NILT
    t0 = time.perf_counter()
    tuned = tune_nilt_params(t_end=t_end, bounds=nilt_bounds)
    nilt_result = nilt_fft_uniform(
        nilt_F_eval, dt=tuned.dt, N=tuned.N, a=tuned.a
    )
    nilt_time = time.perf_counter() - t0

    results['nilt']['time'] = nilt_time
    results['nilt']['dt'] = tuned.dt
    results['nilt']['N'] = tuned.N
    results['nilt']['n_points'] = tuned.N

    # Run MOL
    t0 = time.perf_counter()
    t_mol, u_mol = mol_integrate(t_end)
    mol_time = time.perf_counter() - t0

    results['mol']['time'] = mol_time
    results['mol']['n_points'] = len(t_mol)

    # Compare
    if reference_f is not None:
        # NILT error
        f_ref_nilt = reference_f(nilt_result.t)
        mask_nilt = nilt_result.t <= t_end
        nilt_l2 = float(jnp.sqrt(jnp.mean(
            (nilt_result.f[mask_nilt] - f_ref_nilt[mask_nilt])**2
        )))
        results['nilt']['l2_error'] = nilt_l2

        # MOL error
        f_ref_mol = reference_f(t_mol)
        mol_l2 = float(jnp.sqrt(jnp.mean((u_mol - f_ref_mol)**2)))
        results['mol']['l2_error'] = mol_l2

        results['comparison']['error_ratio'] = nilt_l2 / (mol_l2 + 1e-20)

    results['comparison']['time_ratio'] = nilt_time / (mol_time + 1e-20)

    return results


# =============================================================================
# Standard benchmark suite
# =============================================================================

def run_standard_benchmark_suite(
    output_dir: Path | str = 'benchmark_results'
) -> dict:
    """
    Run standard benchmark suite with various test cases.

    Args:
        output_dir: Directory for output files

    Returns:
        Dict with all benchmark results
    """
    from .transfer_functions import (
        second_order_damping_F, second_order_damping_f,
        exponential_decay_F, exponential_decay_f,
        damped_sine_F, damped_sine_f
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Benchmark 1: Second-order damping
    config = BenchmarkConfig(
        name='second_order_damping',
        t_end=20.0,
        F_eval=second_order_damping_F,
        f_reference=second_order_damping_f
    )
    result = run_nilt_benchmark(config)
    save_benchmark_results(result, output_dir / 'second_order_damping.npz')
    all_results['second_order_damping'] = result

    # Benchmark 2: Exponential decay
    config = BenchmarkConfig(
        name='exponential_decay',
        t_end=10.0,
        F_eval=lambda s: exponential_decay_F(s, alpha=1.0),
        f_reference=lambda t: exponential_decay_f(t, alpha=1.0)
    )
    result = run_nilt_benchmark(config)
    save_benchmark_results(result, output_dir / 'exponential_decay.npz')
    all_results['exponential_decay'] = result

    # Benchmark 3: Damped oscillation
    config = BenchmarkConfig(
        name='damped_oscillation',
        t_end=15.0,
        F_eval=lambda s: damped_sine_F(s, alpha=0.3, omega=2.0),
        f_reference=lambda t: damped_sine_f(t, alpha=0.3, omega=2.0)
    )
    result = run_nilt_benchmark(config)
    save_benchmark_results(result, output_dir / 'damped_oscillation.npz')
    all_results['damped_oscillation'] = result

    # Sensitivity study on second-order damping
    config = BenchmarkConfig(
        name='second_order_sensitivity',
        t_end=20.0,
        F_eval=second_order_damping_F,
        f_reference=second_order_damping_f
    )
    sensitivity = run_sensitivity_benchmark(config)
    save_sensitivity_results(sensitivity, output_dir / 'sensitivity_study.npz')
    all_results['sensitivity'] = sensitivity

    return all_results
