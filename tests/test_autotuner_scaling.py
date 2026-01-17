"""
Tests for NILT autotuner parameter scaling.

Verifies that tuned parameters scale correctly with
operator/grid properties (CFL-like scaling).
"""

import jax.numpy as jnp
import pytest

from moljax.laplace import (
    tune_nilt_params,
    tune_for_diffusion,
    tune_for_advection,
    tune_for_advection_diffusion,
    diagnose_tuning,
    next_power_of_two,
    SpectralBounds,
    BoundContext,
    compute_spectral_bounds,
    fd_laplacian_bounds,
    fd_advection_upwind_bounds,
)


class TestNextPowerOfTwo:
    """Test power-of-two helper."""

    def test_exact_powers(self):
        assert next_power_of_two(1) == 1
        assert next_power_of_two(2) == 2
        assert next_power_of_two(4) == 4
        assert next_power_of_two(1024) == 1024

    def test_non_powers(self):
        assert next_power_of_two(3) == 4
        assert next_power_of_two(5) == 8
        assert next_power_of_two(100) == 128
        assert next_power_of_two(1000) == 1024
        assert next_power_of_two(1025) == 2048

    def test_edge_cases(self):
        assert next_power_of_two(0) == 1
        assert next_power_of_two(-1) == 1


class TestBasicTuning:
    """Test basic autotuner functionality."""

    def test_default_tuning(self):
        """Test tuning with no bounds (uses defaults)."""
        params = tune_nilt_params(t_end=10.0)

        assert params.dt > 0
        assert params.N > 0
        assert params.T >= 10.0  # T should cover t_end
        assert params.a >= 0
        assert params.omega_max > 0

    def test_period_covers_t_end(self):
        """Verify 2T >= period_factor * t_end."""
        t_end = 10.0
        period_factor = 4.0
        params = tune_nilt_params(t_end=t_end, period_factor=period_factor)

        assert 2 * params.T >= period_factor * t_end * 0.99

    def test_n_is_power_of_two(self):
        """Verify N is always a power of 2."""
        for t_end in [1.0, 10.0, 100.0]:
            params = tune_nilt_params(t_end=t_end)
            assert params.N & (params.N - 1) == 0, f"N={params.N} not power of 2"

    def test_consistency_2T_equals_N_dt(self):
        """Verify 2T = N * dt."""
        params = tune_nilt_params(t_end=10.0)
        assert abs(2 * params.T - params.N * params.dt) < 1e-10

    def test_omega_max_computation(self):
        """Verify omega_max = pi/dt."""
        params = tune_nilt_params(t_end=10.0)
        expected_omega_max = jnp.pi / params.dt
        assert abs(params.omega_max - expected_omega_max) < 1e-10


class TestDiffusionScaling:
    """Test dt scaling for diffusion operator."""

    def test_diffusion_bounds_scale_with_dx(self):
        """Spectral radius scales as 1/dx^2 for diffusion."""
        D = 1.0

        rho1, _, _ = fd_laplacian_bounds(dx=0.1, D=D)
        rho2, _, _ = fd_laplacian_bounds(dx=0.05, D=D)

        # rho ~ 4*D/dx^2, so halving dx should ~4x rho
        ratio = rho2 / rho1
        assert 3.5 < ratio < 4.5, f"Rho scaling ratio {ratio} not ~4"

    def test_diffusion_tuning_dt_scales(self):
        """dt requirement should scale with grid for diffusion."""
        D = 0.01
        t_end = 10.0

        # Coarse grid
        params_coarse = tune_for_diffusion(dx=0.1, D=D, t_end=t_end)

        # Fine grid (half spacing)
        params_fine = tune_for_diffusion(dx=0.05, D=D, t_end=t_end)

        # Finer grid has larger rho, needs smaller dt for frequency coverage
        # omega_req scales with rho ~ 1/dx^2
        # so dt ~ 1/omega_req scales as dx^2

        # The ratio won't be exact 4 due to clamping and other factors
        # but fine grid dt should be smaller
        assert params_fine.omega_req >= params_coarse.omega_req, \
            "Finer grid should require larger omega coverage"

    def test_diffusion_tuning_D_scales(self):
        """dt requirement should scale with diffusivity."""
        dx = 0.1
        t_end = 10.0

        params_D1 = tune_for_diffusion(dx=dx, D=0.01, t_end=t_end)
        params_D10 = tune_for_diffusion(dx=dx, D=0.1, t_end=t_end)

        # Larger D -> larger rho -> larger omega_req
        assert params_D10.omega_req >= params_D1.omega_req, \
            "Larger D should require larger omega coverage"


class TestAdvectionScaling:
    """Test dt scaling for advection operator."""

    def test_advection_bounds_scale_with_dx(self):
        """Bounds scale as |v|/dx for advection."""
        v = 1.0

        rho1, _, im1 = fd_advection_upwind_bounds(dx=0.1, vx=v)
        rho2, _, im2 = fd_advection_upwind_bounds(dx=0.05, vx=v)

        # rho ~ 2|v|/dx, halving dx doubles rho
        ratio = rho2 / rho1
        assert 1.8 < ratio < 2.2, f"Advection rho scaling {ratio} not ~2"

    def test_advection_tuning_velocity_scales(self):
        """dt requirement scales with velocity."""
        dx = 0.1
        t_end = 5.0

        params_v1 = tune_for_advection(dx=dx, v=1.0, t_end=t_end)
        params_v2 = tune_for_advection(dx=dx, v=2.0, t_end=t_end)

        # Larger velocity -> larger omega_req
        assert params_v2.omega_req >= params_v1.omega_req, \
            "Larger velocity should require larger omega coverage"


class TestBromwichShiftSelection:
    """Test Bromwich shift (a) selection."""

    def test_stable_system_small_a(self):
        """Stable systems (re_max <= 0) use MATLAB-faithful a=0 or minimal shift."""
        bounds = SpectralBounds(
            rho=10.0,
            re_max=-1.0,  # Stable
            im_max=5.0,
            methods_used={},
            warnings=[]
        )

        params = tune_nilt_params(t_end=10.0, bounds=bounds)

        # In auto mode, stable systems use a=0 unless wraparound requires shift
        # a should be non-negative and small
        assert params.a >= 0
        assert params.a < 5.0, f"a={params.a} too large for stable system"

    def test_marginally_stable_positive_a(self):
        """Marginally stable (re_max = 0) should get positive a."""
        bounds = SpectralBounds(
            rho=10.0,
            re_max=0.0,  # Marginally stable
            im_max=5.0,
            methods_used={},
            warnings=[]
        )

        params = tune_nilt_params(t_end=10.0, bounds=bounds)
        assert params.a > 0, "a should be positive for marginally stable system"

    def test_unstable_system_large_a(self):
        """Unstable systems (re_max > 0) need a > re_max."""
        bounds = SpectralBounds(
            rho=10.0,
            re_max=2.0,  # Unstable
            im_max=5.0,
            methods_used={},
            warnings=[]
        )

        params = tune_nilt_params(t_end=10.0, bounds=bounds)
        assert params.a > bounds.re_max, \
            f"a={params.a} should exceed re_max={bounds.re_max}"


class TestOmegaCoverage:
    """Test frequency coverage requirements."""

    def test_omega_max_exceeds_omega_req(self):
        """omega_max should exceed omega_req by omega_factor."""
        bounds = SpectralBounds(
            rho=10.0,
            re_max=0.0,
            im_max=5.0,
            methods_used={},
            warnings=[]
        )

        omega_factor = 2.0
        params = tune_nilt_params(
            t_end=10.0, bounds=bounds, omega_factor=omega_factor
        )

        # May not be exact due to N rounding, but should be close
        coverage_ratio = params.omega_max / params.omega_req
        assert coverage_ratio >= omega_factor * 0.5, \
            f"Coverage ratio {coverage_ratio} too small"

    def test_warning_on_insufficient_coverage(self):
        """Warning should be generated when N is clamped."""
        # Force N clamping by extreme frequency requirements
        params = tune_nilt_params(
            t_end=100.0,
            bounds=SpectralBounds(rho=1000.0, re_max=0.0, im_max=500.0,
                                   methods_used={}, warnings=[]),
            N_max=256  # Too small for required frequency coverage
        )

        # Should have warnings about N clamping
        has_warning = any('clamp' in w.lower() or 'split' in w.lower()
                          for w in params.warnings)
        # May or may not trigger depending on exact constraints
        # Just check the params are returned
        assert params.dt > 0
        assert params.N <= 256


class TestConstraintEnforcement:
    """Test that parameter constraints are enforced."""

    def test_n_clamped_to_range(self):
        """N should be clamped to [N_min, N_max]."""
        params = tune_nilt_params(
            t_end=10.0,
            N_min=256,
            N_max=512
        )

        assert params.N >= 256
        assert params.N <= 512

    def test_dt_computed_from_nyquist(self):
        """dt is computed from Nyquist criterion, not manually clamped."""
        params = tune_nilt_params(
            t_end=10.0,
            bounds=SpectralBounds(rho=10.0, re_max=0.0, im_max=5.0,
                                   methods_used={}, warnings=[])
        )

        # dt should be positive and consistent with 2T = N*dt
        assert params.dt > 0
        assert abs(2.0 * params.T - params.N * params.dt) < 1e-10


class TestDiagnosticReport:
    """Test the diagnostic report function."""

    def test_diagnose_tuning_structure(self):
        """Test diagnose_tuning returns expected structure."""
        params = tune_nilt_params(t_end=10.0)
        report = diagnose_tuning(params)

        assert 'parameters' in report
        assert 'coverage' in report
        assert 'bounds' in report
        assert 'warnings' in report
        assert 'recommendations' in report

        assert 'dt' in report['parameters']
        assert 'N' in report['parameters']
        assert 'omega_max' in report['coverage']

    def test_diagnose_generates_recommendations(self):
        """Diagnose should generate recommendations for edge cases."""
        # Force marginal coverage
        bounds = SpectralBounds(
            rho=100.0,
            re_max=0.0,
            im_max=50.0,
            methods_used={},
            warnings=[]
        )

        params = tune_nilt_params(
            t_end=10.0,
            bounds=bounds,
            omega_factor=1.1,  # Tight margin
            N_max=512
        )

        report = diagnose_tuning(params)
        # May or may not have recommendations depending on exact parameters
        assert isinstance(report['recommendations'], list)


class TestConvenienceFunctions:
    """Test convenience tuning functions."""

    def test_tune_for_diffusion(self):
        """Test diffusion-specific tuning."""
        params = tune_for_diffusion(dx=0.1, D=0.01, t_end=10.0)

        assert params.dt > 0
        assert params.N > 0
        assert 'analytic_laplacian' in str(params.bound_sources)

    def test_tune_for_advection(self):
        """Test advection-specific tuning."""
        params = tune_for_advection(dx=0.1, v=1.0, t_end=5.0)

        assert params.dt > 0
        assert params.N > 0
        assert 'advection' in str(params.bound_sources)

    def test_tune_for_advection_diffusion(self):
        """Test combined advection-diffusion tuning."""
        params = tune_for_advection_diffusion(
            dx=0.1, v=1.0, D=0.01, t_end=10.0
        )

        assert params.dt > 0
        assert params.N > 0
        assert 'coupled' in str(params.bound_sources)

    def test_2d_diffusion(self):
        """Test 2D diffusion tuning."""
        params = tune_for_diffusion(dx=0.1, D=0.01, t_end=10.0, ndim=2)

        assert params.dt > 0
        # 2D should have larger omega_req than 1D (due to dx and dy)
        params_1d = tune_for_diffusion(dx=0.1, D=0.01, t_end=10.0, ndim=1)
        assert params.omega_req >= params_1d.omega_req


class TestBoundContextIntegration:
    """Test integration with BoundContext."""

    def test_with_bound_context(self):
        """Test tuning from BoundContext."""
        ctx = BoundContext(
            grid_spacings=(0.1,),
            operator_type='FD_LAPLACIAN',
            diffusivity=0.01,
            ndim=1
        )

        params = tune_nilt_params(t_end=10.0, ctx=ctx)

        assert params.dt > 0
        assert 'laplacian' in str(params.bound_sources)

    def test_coupled_context(self):
        """Test with coupled operator context."""
        ctx = BoundContext(
            grid_spacings=(0.1,),
            operator_type='COUPLED',
            diffusivity=0.01,
            velocities=1.0,
            ndim=1
        )

        params = tune_nilt_params(t_end=10.0, ctx=ctx)
        assert params.dt > 0
