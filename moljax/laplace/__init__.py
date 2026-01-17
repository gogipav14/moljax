"""
NILT (Numerical Inverse Laplace Transform) subpackage.

This is a separate implementation track from the MOL stepping core,
providing FFT-based Laplace transform inversion with automatic
parameter tuning based on hybrid spectral bounds.

Main components:
- nilt_fft: FFT-based numerical inverse Laplace transform
- tuning: Spectral autotuner for NILT parameters
- spectral_bounds: Hybrid analytic/matrix-free spectral bounds
- transfer_functions: Canonical Laplace pairs for testing
- diagnostics: Quality metrics for NILT results
- benchmarks: Performance comparison harness
"""

from .nilt_fft import (
    NILTResult,
    nilt_fft_uniform,
    nilt_fft_halfstep,
    nilt_fft_halfstep_ivt,
    nilt_fft_signed_omega,
    nilt_fft_times,
    nilt_fft_with_pole_at_origin,
    nilt_fft_batch,
    integrate_discrete,
    invert_laplace,
    estimate_nilt_truncation_error,
    estimate_aliasing_error,
)

from .tuning import (
    TunedNILTParams,
    tune_nilt_params,
    tune_nilt_from_mol_model,
    tune_for_diffusion,
    tune_for_advection,
    tune_for_advection_diffusion,
    diagnose_tuning,
    next_power_of_two,
)

from .adaptive_tuning import (
    QualityTier,
    AdaptiveTuningResult,
    classify_quality,
    retune_based_on_diagnostics,
    tune_nilt_adaptive,
    tune_nilt_adaptive_cfl,
)

from .endpoint_diagnostics import (
    SpectralCFLConditions,
    compute_ivt,
    check_spectral_cfl_conditions,
    suggest_parameter_adjustments,
    print_cfl_diagnostic_report,
)

from .spectral_bounds import (
    SpectralBounds,
    BoundContext,
    compute_spectral_bounds,
    bounds_from_mol_model,
    fd_laplacian_bounds,
    fd_advection_upwind_bounds,
    fd_advection_centered_bounds,
    wave_operator_bounds,
    gershgorin_bound_from_stencil,
    power_iteration_rho,
    symmetric_part_bound_remax,
    antisymmetric_part_bound_immax,
)

from .transfer_functions import (
    LaplacePair,
    second_order_damping_F,
    second_order_damping_f,
    exponential_decay_F,
    exponential_decay_f,
    cosine_F,
    cosine_f,
    sine_F,
    sine_f,
    damped_sine_F,
    damped_sine_f,
    damped_cosine_F,
    damped_cosine_f,
    step_F,
    step_f,
    ramp_F,
    ramp_f,
    first_order_lag_F,
    first_order_lag_f,
    first_order_plus_delay_F,
    first_order_plus_delay_f,
    diffusion_semi_infinite_F,
    diffusion_semi_infinite_f,
    packed_bed_dispersion_F,
    get_standard_laplace_pairs,
    make_exponential_pair,
    make_oscillatory_pair,
    make_second_order_pair,
)

from .diagnostics import (
    NILTDiagnostics,
    frequency_coverage_metric,
    estimate_truncation_error,
    ringing_metric,
    wraparound_metric,
    energy_distribution,
    oscillation_count,
    compute_nilt_diagnostics,
    compare_to_reference,
    sensitivity_analysis,
)

from .benchmarks import (
    BenchmarkConfig,
    BenchmarkResult,
    run_nilt_benchmark,
    run_sensitivity_benchmark,
    save_benchmark_results,
    save_sensitivity_results,
    compare_nilt_vs_mol,
    run_standard_benchmark_suite,
)

from .fft_nilt_bridge import (
    FFTSpectralBounds,
    exact_spectral_bounds_from_fft,
    fft_bounds_to_spectral_bounds,
    tune_nilt_for_fft_operator,
    create_transfer_function_from_fft_operator,
    nilt_solve_linear_pde,
    NILTvsTSSComparison,
    compare_nilt_vs_timestepping,
    print_comparison_table,
)

from .spectral_smoothing import (
    SmoothingMethod,
    SmoothingResult,
    fejer_sigma,
    lanczos_sigma,
    hamming_window,
    raised_cosine_sigma,
    exponential_sigma,
    get_sigma_factors,
    apply_spectral_smoothing,
    nilt_with_smoothing,
    compare_smoothing_methods,
    print_smoothing_comparison,
)

from .quality_metrics import (
    QualityLevel,
    RetuningAction,
    QualityMetrics,
    AutotunerFeedback,
    compute_eps_im,
    compute_eps_sym,
    classify_quality,
    assess_nilt_quality,
    generate_autotuner_feedback,
    integrate_with_adaptive_tuner,
    print_quality_report,
    quality_meets_threshold,
)

from .chebyshev_nilt import (
    ChebyshevNILTResult,
    chebyshev_nodes,
    chebyshev_coefficients,
    chebyshev_eval,
    laguerre_coefficients,
    laguerre_eval,
    weeks_method,
    talbot_contour,
    talbot_method,
    gaver_stehfest_weights,
    gaver_stehfest_method,
    adaptive_chebyshev_nilt,
    compare_chebyshev_vs_fft,
    print_chebyshev_report,
)


__all__ = [
    # NILT core
    'NILTResult',
    'nilt_fft_uniform',
    'nilt_fft_halfstep',
    'nilt_fft_halfstep_ivt',
    'nilt_fft_signed_omega',
    'nilt_fft_times',
    'nilt_fft_with_pole_at_origin',
    'nilt_fft_batch',
    'integrate_discrete',
    'invert_laplace',
    'estimate_nilt_truncation_error',
    'estimate_aliasing_error',

    # Tuning
    'TunedNILTParams',
    'tune_nilt_params',
    'tune_nilt_from_mol_model',
    'tune_for_diffusion',
    'tune_for_advection',
    'tune_for_advection_diffusion',
    'diagnose_tuning',
    'next_power_of_two',

    # Adaptive tuning (closed-loop)
    'QualityTier',
    'AdaptiveTuningResult',
    'classify_quality',
    'retune_based_on_diagnostics',
    'tune_nilt_adaptive',
    'tune_nilt_adaptive_cfl',

    # Endpoint diagnostics (CFL-like spectral guardrails)
    'SpectralCFLConditions',
    'compute_ivt',
    'check_spectral_cfl_conditions',
    'suggest_parameter_adjustments',
    'print_cfl_diagnostic_report',

    # Spectral bounds
    'SpectralBounds',
    'BoundContext',
    'compute_spectral_bounds',
    'bounds_from_mol_model',
    'fd_laplacian_bounds',
    'fd_advection_upwind_bounds',
    'fd_advection_centered_bounds',
    'wave_operator_bounds',
    'gershgorin_bound_from_stencil',
    'power_iteration_rho',
    'symmetric_part_bound_remax',
    'antisymmetric_part_bound_immax',

    # Transfer functions
    'LaplacePair',
    'second_order_damping_F',
    'second_order_damping_f',
    'exponential_decay_F',
    'exponential_decay_f',
    'cosine_F',
    'cosine_f',
    'sine_F',
    'sine_f',
    'damped_sine_F',
    'damped_sine_f',
    'damped_cosine_F',
    'damped_cosine_f',
    'step_F',
    'step_f',
    'ramp_F',
    'ramp_f',
    'first_order_lag_F',
    'first_order_lag_f',
    'first_order_plus_delay_F',
    'first_order_plus_delay_f',
    'diffusion_semi_infinite_F',
    'diffusion_semi_infinite_f',
    'packed_bed_dispersion_F',
    'get_standard_laplace_pairs',
    'make_exponential_pair',
    'make_oscillatory_pair',
    'make_second_order_pair',

    # Diagnostics
    'NILTDiagnostics',
    'frequency_coverage_metric',
    'estimate_truncation_error',
    'ringing_metric',
    'wraparound_metric',
    'energy_distribution',
    'oscillation_count',
    'compute_nilt_diagnostics',
    'compare_to_reference',
    'sensitivity_analysis',

    # Benchmarks
    'BenchmarkConfig',
    'BenchmarkResult',
    'run_nilt_benchmark',
    'run_sensitivity_benchmark',
    'save_benchmark_results',
    'save_sensitivity_results',
    'compare_nilt_vs_mol',
    'run_standard_benchmark_suite',

    # FFT-NILT Bridge (Milestone 4)
    'FFTSpectralBounds',
    'exact_spectral_bounds_from_fft',
    'fft_bounds_to_spectral_bounds',
    'tune_nilt_for_fft_operator',
    'create_transfer_function_from_fft_operator',
    'nilt_solve_linear_pde',
    'NILTvsTSSComparison',
    'compare_nilt_vs_timestepping',
    'print_comparison_table',

    # Spectral Smoothing (Gibbs artifact reduction)
    'SmoothingMethod',
    'SmoothingResult',
    'fejer_sigma',
    'lanczos_sigma',
    'hamming_window',
    'raised_cosine_sigma',
    'exponential_sigma',
    'get_sigma_factors',
    'apply_spectral_smoothing',
    'nilt_with_smoothing',
    'compare_smoothing_methods',
    'print_smoothing_comparison',

    # Quality Metrics (unified quality assessment)
    'QualityLevel',
    'RetuningAction',
    'QualityMetrics',
    'AutotunerFeedback',
    'compute_eps_im',
    'compute_eps_sym',
    'classify_quality',
    'assess_nilt_quality',
    'generate_autotuner_feedback',
    'integrate_with_adaptive_tuner',
    'print_quality_report',
    'quality_meets_threshold',

    # Chebyshev NILT (alternative methods)
    'ChebyshevNILTResult',
    'chebyshev_nodes',
    'chebyshev_coefficients',
    'chebyshev_eval',
    'laguerre_coefficients',
    'laguerre_eval',
    'weeks_method',
    'talbot_contour',
    'talbot_method',
    'gaver_stehfest_weights',
    'gaver_stehfest_method',
    'adaptive_chebyshev_nilt',
    'compare_chebyshev_vs_fft',
    'print_chebyshev_report',
]
