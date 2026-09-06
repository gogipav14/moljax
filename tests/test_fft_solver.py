"""
Tests for FFT-based solvers in MOL-JAX.

Validates:
- FFT Laplacian symbol matches FD Laplacian
- Helmholtz solver correctness: (I - dt*D*Δ)u = rhs
- Multi-field FFT diffusion inverse
- Spectral filters
"""

import jax
import jax.numpy as jnp
import numpy as np

# The preconditioner comparisons are made to 1e-12, which needs float64.
jax.config.update("jax_enable_x64", True)

from moljax.core.bc import BCType, FieldBCSpec, apply_bc
from moljax.core.fft_solvers import (
    apply_diffusion_inverse_fft,
    apply_spectral_filter_interior,
    create_fft_cache_1d,
    create_fft_cache_2d,
    create_fft_cache_2d_rfft,
    exponential_filter_1d,
    exponential_filter_2d,
    laplacian_symbol_1d,
    laplacian_symbol_2d,
    solve_helmholtz_1d,
    solve_helmholtz_2d,
)
from moljax.core.grid import Grid1D, Grid2D
from moljax.core.operators import laplacian_2d
from moljax.core.preconditioners import (
    FFTAdvectionDiffusionPreconditioner,
    FFTDiffusionPreconditioner,
    PrecondContext,
)


class TestRfftCacheLayout:
    """The rfft cache is the first nx//2 + 1 columns of the full cache.

    kx varies along columns (axis 1) and ky along rows (axis 0) in both
    layouts; the rfft cache once carried the y frequencies under the name
    kx and vice versa, which the Laplacian symbol hid (both were paired
    with the wrong spacing, consistently) but the advection symbol did not.

    kx itself matches full.kx column for column except at the Nyquist bin
    of an even nx: fftfreq (the full cache) stores it as -pi/dx and
    rfftfreq (the half cache) as +pi/dx, the same physical frequency with
    opposite sign by numpy convention. That is exactly why the advection
    preconditioner zeroes the odd symbol there (K8); the Laplacian symbol
    is unaffected because cos(kx dx) does not see the sign.
    """

    def test_rfft_wavenumbers_are_half_of_full(self):
        grid = Grid2D.uniform(24, 16, 0.0, 1.0, 0.0, 2.0)
        full = create_fft_cache_2d(grid)
        half = create_fft_cache_2d_rfft(grid)
        n_r = grid.nx // 2 + 1

        assert half.kx.shape == half.ky.shape == (grid.ny, n_r)
        # All columns but the Nyquist one agree exactly; the Nyquist column
        # (present because nx=24 is even) agrees up to the fftfreq/rfftfreq
        # sign convention.
        assert np.array_equal(np.asarray(half.kx[:, :-1]), np.asarray(full.kx[:, :n_r - 1]))
        assert np.array_equal(np.abs(np.asarray(half.kx[:, -1])), np.abs(np.asarray(full.kx[:, n_r - 1])))
        assert np.array_equal(np.asarray(half.ky), np.asarray(full.ky[:, :n_r]))
        assert np.array_equal(np.asarray(half.laplacian_symbol),
                              np.asarray(full.laplacian_symbol[:, :n_r]))
        assert np.allclose(np.asarray(half.kx[0]), 2 * np.pi * np.fft.rfftfreq(grid.nx, grid.dx))
        assert np.allclose(np.asarray(half.ky[:, 0]), 2 * np.pi * np.fft.fftfreq(grid.ny, grid.dy))


class TestFFTPreconditionersRfft:
    """Both FFT preconditioners give the same answer with an rfft or a full cache."""

    def test_fft_preconditioners_rfft_matches_full(self):
        grid = Grid2D.uniform(24, 16, 0.0, 1.0, 0.0, 2.0)
        D, dt, v = 0.3, 0.05, (1.0, 0.5)
        sl_y, sl_x = grid.interior_slice

        rng = np.random.default_rng(0)
        r_interior = rng.standard_normal((grid.ny, grid.nx))
        r_padded = jnp.zeros((grid.ny_total, grid.nx_total)).at[sl_y, sl_x].set(r_interior)
        context = PrecondContext(grid=grid, dt=dt, params={'D': D, 'v': v})

        results = {}
        for label, cache in (('full', create_fft_cache_2d(grid)),
                             ('rfft', create_fft_cache_2d_rfft(grid))):
            diffusion = FFTDiffusionPreconditioner(field_diffusivity_keys={'u': 'D'}, fft_cache=cache)
            advection = FFTAdvectionDiffusionPreconditioner(
                field_diffusivity_keys={'u': 'D'}, field_velocity_keys={'u': 'v'}, fft_cache=cache
            )
            results[label] = (
                np.asarray(diffusion.apply({'u': r_padded}, context)['u'][sl_y, sl_x]),
                np.asarray(advection.apply({'u': r_padded}, context)['u'][sl_y, sl_x]),
            )

        for i, name in enumerate(("diffusion", "advection-diffusion")):
            diff = np.max(np.abs(results['rfft'][i] - results['full'][i]))
            assert diff < 1e-12, f"{name}: rfft and full caches differ by {diff:.2e}"

        # Independent numpy reference. The first-derivative symbol at the
        # Nyquist frequency is set to zero, the standard convention for an
        # odd symbol on a real grid function (it keeps the spectrum
        # Hermitian, so the inverse transform is real).
        kx = 2 * np.pi * np.fft.fftfreq(grid.nx, grid.dx)[None, :]
        ky = 2 * np.pi * np.fft.fftfreq(grid.ny, grid.dy)[:, None]
        lap = (2 * np.cos(kx * grid.dx) - 2) / grid.dx**2 + (2 * np.cos(ky * grid.dy) - 2) / grid.dy**2
        kx_adv = np.where(np.isclose(np.abs(kx), np.pi / grid.dx), 0.0, kx)
        ky_adv = np.where(np.isclose(np.abs(ky), np.pi / grid.dy), 0.0, ky)
        lam_adv = -1j * (v[0] * kx_adv + v[1] * ky_adv)
        r_hat = np.fft.fft2(r_interior)
        x_diff_ref = np.real(np.fft.ifft2(r_hat / (1 - dt * D * lap)))
        x_adv_ref = np.real(np.fft.ifft2(r_hat / (1 - dt * (D * lap + lam_adv))))

        assert np.max(np.abs(results['full'][0] - x_diff_ref)) < 1e-12
        assert np.max(np.abs(results['full'][1] - x_adv_ref)) < 1e-12


class TestLaplacianSymbol1D:
    """Test 1D Laplacian symbol."""

    def test_symbol_shape(self):
        """Symbol should have correct shape."""
        nx = 32
        dx = 0.1
        lam = laplacian_symbol_1d(nx, dx)
        assert lam.shape == (nx,)

    def test_symbol_zero_mode(self):
        """Zero mode (k=0) should give lam=0."""
        nx = 32
        dx = 0.1
        lam = laplacian_symbol_1d(nx, dx)
        # First element is k=0 mode
        assert jnp.abs(lam[0]) < 1e-10

    def test_symbol_negative(self):
        """All non-zero modes should have negative eigenvalues."""
        nx = 32
        dx = 0.1
        lam = laplacian_symbol_1d(nx, dx)
        # All non-zero modes should be negative
        assert jnp.all(lam[1:] < 0)

    def test_symbol_matches_fd_laplacian(self):
        """FFT Laplacian should match FD Laplacian for sine mode."""
        nx = 64
        dx = 2.0 * jnp.pi / nx

        # Create a single Fourier mode: sin(k*x)
        k_mode = 3
        x = jnp.linspace(0, 2*jnp.pi, nx, endpoint=False)
        u_interior = jnp.sin(k_mode * x)

        # Analytical Laplacian: -k^2 * sin(k*x)
        lap_exact = -k_mode**2 * u_interior

        # Compute using FFT symbol
        lam = laplacian_symbol_1d(nx, dx)
        u_hat = jnp.fft.fft(u_interior)
        lap_hat = lam * u_hat
        lap_fft = jnp.fft.ifft(lap_hat).real

        # They should match closely (relaxed for float32)
        assert jnp.allclose(lap_fft, lap_exact, rtol=0.05, atol=0.1)


class TestLaplacianSymbol2D:
    """Test 2D Laplacian symbol."""

    def test_symbol_shape(self):
        """Symbol should have correct shape."""
        ny, nx = 32, 48
        dy, dx = 0.1, 0.15
        lam = laplacian_symbol_2d(ny, nx, dy, dx)
        assert lam.shape == (ny, nx)

    def test_symbol_zero_mode(self):
        """Zero mode should give lam=0."""
        ny, nx = 32, 32
        dy, dx = 0.1, 0.1
        lam = laplacian_symbol_2d(ny, nx, dy, dx)
        assert jnp.abs(lam[0, 0]) < 1e-10

    def test_symbol_matches_fd_laplacian(self):
        """FFT Laplacian should match FD Laplacian for 2D sine mode."""
        ny, nx = 32, 32
        Ly, Lx = 2*jnp.pi, 2*jnp.pi
        dy, dx = Ly/ny, Lx/nx

        # Create 2D Fourier mode
        kx, ky = 2, 3
        x = jnp.linspace(0, Lx, nx, endpoint=False)
        y = jnp.linspace(0, Ly, ny, endpoint=False)
        X, Y = jnp.meshgrid(x, y)
        u_interior = jnp.sin(kx * X) * jnp.sin(ky * Y)

        # Analytical Laplacian: -(kx^2 + ky^2) * u
        lap_exact = -(kx**2 + ky**2) * u_interior

        # Compute using FFT symbol
        lam = laplacian_symbol_2d(ny, nx, dy, dx)
        u_hat = jnp.fft.fft2(u_interior)
        lap_hat = lam * u_hat
        lap_fft = jnp.fft.ifft2(lap_hat).real

        # Should match closely (relaxed for float32)
        assert jnp.allclose(lap_fft, lap_exact, rtol=0.05, atol=0.1)


class TestHelmholtzSolver1D:
    """Test 1D Helmholtz solver."""

    def test_solve_identity(self):
        """With D=0, solution should equal RHS."""
        nx = 32
        dx = 0.1
        cache = create_fft_cache_1d(Grid1D.uniform(nx, 0, nx*dx, n_ghost=1))

        rhs = jnp.sin(jnp.linspace(0, 2*jnp.pi, nx))
        u = solve_helmholtz_1d(rhs, cache.laplacian_symbol, dt=1.0, D=0.0)

        assert jnp.allclose(u, rhs, atol=1e-5)

    def test_solve_correctness(self):
        """Verify (I - dt*D*Δ)u = rhs."""
        nx = 64
        grid = Grid1D.uniform(nx, 0, 2*jnp.pi, n_ghost=1)
        cache = create_fft_cache_1d(grid)

        dt = 0.1
        D = 0.5

        # Random RHS
        key = jax.random.PRNGKey(42)
        rhs = jax.random.normal(key, (nx,))

        # Solve
        u = solve_helmholtz_1d(rhs, cache.laplacian_symbol, dt, D)

        # Verify: compute (I - dt*D*Δ)u
        u_hat = jnp.fft.fft(u)
        residual_hat = u_hat - dt * D * cache.laplacian_symbol * u_hat
        residual = jnp.fft.ifft(residual_hat).real

        # Should equal rhs (relaxed for float32)
        assert jnp.allclose(residual, rhs, atol=1e-5)


class TestHelmholtzSolver2D:
    """Test 2D Helmholtz solver."""

    def test_solve_identity(self):
        """With D=0, solution should equal RHS."""
        ny, nx = 32, 32
        dy, dx = 0.1, 0.1
        grid = Grid2D.uniform(nx, ny, 0, nx*dx, 0, ny*dy, n_ghost=1)
        cache = create_fft_cache_2d(grid)

        rhs = jnp.ones((ny, nx))
        u = solve_helmholtz_2d(rhs, cache.laplacian_symbol, dt=1.0, D=0.0)

        assert jnp.allclose(u, rhs, atol=1e-5)

    def test_solve_correctness(self):
        """Verify (I - dt*D*Δ)u = rhs."""
        ny, nx = 32, 32
        Ly, Lx = 2*jnp.pi, 2*jnp.pi
        _dy, _dx = Ly/ny, Lx/nx
        grid = Grid2D.uniform(nx, ny, 0, Lx, 0, Ly, n_ghost=1)
        cache = create_fft_cache_2d(grid)

        dt = 0.05
        D = 0.3

        # Random RHS
        key = jax.random.PRNGKey(123)
        rhs = jax.random.normal(key, (ny, nx))

        # Solve
        u = solve_helmholtz_2d(rhs, cache.laplacian_symbol, dt, D)

        # Verify: compute (I - dt*D*Δ)u
        u_hat = jnp.fft.fft2(u)
        residual_hat = u_hat - dt * D * cache.laplacian_symbol * u_hat
        residual = jnp.fft.ifft2(residual_hat).real

        # Should equal rhs (relaxed for float32)
        assert jnp.allclose(residual, rhs, atol=1e-5)

    def test_solve_vs_fd_residual(self):
        """FFT solution should have small FD residual."""
        ny, nx = 32, 32
        Lx, Ly = 1.0, 1.0
        grid = Grid2D.uniform(nx, ny, 0, Lx, 0, Ly, n_ghost=1)
        cache = create_fft_cache_2d(grid)

        dt = 0.01
        D = 0.1

        # Create RHS with ghost cells
        rhs_interior = jnp.sin(2*jnp.pi*jnp.linspace(0, 1, nx)) * jnp.ones((ny, 1))
        rhs_padded = jnp.zeros((grid.ny_total, grid.nx_total))
        sl_y, sl_x = grid.interior_slice
        rhs_padded = rhs_padded.at[sl_y, sl_x].set(rhs_interior)

        # Apply periodic BC
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state_rhs = {'f': rhs_padded}
        state_rhs = apply_bc(state_rhs, grid, bc_spec, t=0.0, params={})
        rhs_padded = state_rhs['f']

        # Solve using FFT
        u_interior = solve_helmholtz_2d(rhs_interior, cache.laplacian_symbol, dt, D)

        # Embed into padded array and apply BC
        u_padded = rhs_padded.at[sl_y, sl_x].set(u_interior)
        state_u = {'f': u_padded}
        state_u = apply_bc(state_u, grid, bc_spec, t=0.0, params={})
        u_padded = state_u['f']

        # Compute FD residual: (I - dt*D*Δ)u - rhs
        lap_u = laplacian_2d(u_padded, grid)
        residual = u_padded - dt * D * lap_u - rhs_padded

        # Extract interior residual norm
        res_interior = residual[sl_y, sl_x]
        res_norm = jnp.linalg.norm(res_interior)

        # Should be small (relaxed tolerance for float32)
        assert res_norm < 1e-3, f"Residual norm too large: {res_norm}"


class TestMultiFieldFFT:
    """Test multi-field FFT diffusion inverse."""

    def test_multifield_solve(self):
        """Multiple fields with different diffusivities."""
        ny, nx = 32, 32
        grid = Grid2D.uniform(nx, ny, 0, 1, 0, 1, n_ghost=1)
        cache = create_fft_cache_2d(grid)

        dt = 0.01
        diffusivities = {'u': 0.1, 'v': 0.2, 'w': 0.0}

        # Create state
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        state = {
            'u': jax.random.normal(keys[0], (grid.ny_total, grid.nx_total)),
            'v': jax.random.normal(keys[1], (grid.ny_total, grid.nx_total)),
            'w': jax.random.normal(keys[2], (grid.ny_total, grid.nx_total)),
        }

        # Apply FFT diffusion inverse
        result = apply_diffusion_inverse_fft(state, grid, dt, diffusivities, cache)

        # w should be unchanged (D=0)
        assert jnp.allclose(result['w'], state['w'])

        # u and v should be different
        assert not jnp.allclose(result['u'], state['u'])
        assert not jnp.allclose(result['v'], state['v'])


class TestSpectralFilter:
    """Test spectral filters."""

    def test_filter_preserves_low_modes(self):
        """Low modes should be nearly unchanged."""
        nx = 64
        sigma = exponential_filter_1d(nx, strength=36.0, order=36)

        # Low modes (first few) should be ~1
        assert jnp.all(sigma[:5] > 0.99)

    def test_filter_damps_high_modes(self):
        """High modes should be damped."""
        nx = 64
        sigma = exponential_filter_1d(nx, strength=36.0, order=36)

        # High modes (near Nyquist) should be small
        assert sigma[nx//2] < 0.1

    def test_filter_smoothes_noise(self):
        """Filter should smooth high-frequency noise."""
        nx = 64
        sigma = exponential_filter_1d(nx, strength=10.0, order=8)

        # Create signal with high-frequency noise
        x = jnp.linspace(0, 2*jnp.pi, nx)
        smooth = jnp.sin(x)
        key = jax.random.PRNGKey(0)
        noise = 0.1 * jax.random.normal(key, (nx,))
        noisy = smooth + noise

        # Apply filter
        filtered = apply_spectral_filter_interior(noisy, sigma)

        # Filtered should be closer to smooth than noisy
        err_noisy = jnp.linalg.norm(noisy - smooth)
        err_filtered = jnp.linalg.norm(filtered - smooth)
        assert err_filtered < err_noisy

    def test_filter_2d(self):
        """2D filter should work."""
        ny, nx = 32, 48
        sigma = exponential_filter_2d(ny, nx, strength=20.0, order=16)

        assert sigma.shape == (ny, nx)
        # Corner (low mode) should be ~1
        assert sigma[0, 0] > 0.99
