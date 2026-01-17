"""
Tests for higher-order finite difference operators.

Tests verify:
1. 4th-order accuracy convergence
2. FFT symbol consistency with finite difference stencils
3. Comparison with 2nd-order operators
"""

import pytest
import jax.numpy as jnp
import numpy as np
from functools import partial

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.operators_ho import (
    d1_fourth_order_1d,
    d2_fourth_order_1d,
    d1_ho_1d,
    d2_ho_1d,
    laplacian_ho_1d,
    d1_ho_2d,
    d2_ho_2d,
    laplacian_ho_2d,
    fd_laplacian_symbol_ho_1d,
    fd_laplacian_symbol_ho_2d,
    fd_d1_symbol_ho_1d,
    d2_sixth_order_1d,
    fd_laplacian_symbol_6th_1d,
    get_laplacian_symbol_1d,
    get_laplacian_symbol_2d,
    OperatorOrder,
)
from moljax.core.operators import (
    d1_central_1d,
    d2_central_1d,
    fd_laplacian_symbol_1d,
)


class TestFourthOrder1D:
    """Test 4th-order 1D operators."""

    def test_d1_smooth_function(self):
        """Test D1 on sin(x)."""
        n = 64
        ng = 2
        dx = 2 * jnp.pi / n

        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        f = jnp.sin(x)
        df_exact = jnp.cos(x)

        df_num = d1_fourth_order_1d(f, dx, ng, n)

        # Check interior accuracy (4th-order error scales as dx^4)
        error = jnp.max(jnp.abs(df_num[ng:ng + n] - df_exact[ng:ng + n]))
        assert error < 1e-5, f"D1 error too large: {error}"

    def test_d2_smooth_function(self):
        """Test D2 on sin(x)."""
        n = 64
        ng = 2
        dx = 2 * jnp.pi / n

        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        f = jnp.sin(x)
        d2f_exact = -jnp.sin(x)

        d2f_num = d2_fourth_order_1d(f, dx, ng, n)

        # 4th-order accuracy
        error = jnp.max(jnp.abs(d2f_num[ng:ng + n] - d2f_exact[ng:ng + n]))
        assert error < 1e-5, f"D2 error too large: {error}"

    def test_d1_convergence_order(self):
        """Verify 4th-order convergence for D1."""
        errors = []
        ns = [16, 32, 64, 128]

        for n in ns:
            ng = 2
            dx = 2 * jnp.pi / n

            x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
            f = jnp.sin(x)
            df_exact = jnp.cos(x)

            df_num = d1_fourth_order_1d(f, dx, ng, n)
            error = jnp.max(jnp.abs(df_num[ng:ng + n] - df_exact[ng:ng + n]))
            errors.append(float(error))

        # Compute convergence rates
        rates = []
        for i in range(1, len(errors)):
            rate = jnp.log2(errors[i - 1] / errors[i])
            rates.append(float(rate))

        # Should be close to 4
        avg_rate = sum(rates) / len(rates)
        assert avg_rate > 3.8, f"Convergence rate {avg_rate} not 4th-order"

    def test_d2_convergence_order(self):
        """Verify 4th-order convergence for D2."""
        errors = []
        ns = [16, 32, 64, 128]

        for n in ns:
            ng = 2
            dx = 2 * jnp.pi / n

            x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
            f = jnp.sin(x)
            d2f_exact = -jnp.sin(x)

            d2f_num = d2_fourth_order_1d(f, dx, ng, n)
            error = jnp.max(jnp.abs(d2f_num[ng:ng + n] - d2f_exact[ng:ng + n]))
            errors.append(float(error))

        # Compute convergence rates
        rates = []
        for i in range(1, len(errors)):
            rate = jnp.log2(errors[i - 1] / errors[i])
            rates.append(float(rate))

        avg_rate = sum(rates) / len(rates)
        assert avg_rate > 3.8, f"Convergence rate {avg_rate} not 4th-order"

    def test_grid_interface(self):
        """Test Grid1D interface functions."""
        grid = Grid1D(nx=64, x_min=0.0, x_max=2 * jnp.pi, n_ghost=2)
        x = grid.x_coords(include_ghost=True)
        f = jnp.sin(x)

        df = d1_ho_1d(f, grid)
        d2f = d2_ho_1d(f, grid)
        lap_f = laplacian_ho_1d(f, grid)

        # Laplacian should equal d2 in 1D
        assert jnp.allclose(d2f, lap_f)

    def test_requires_ghost_cells(self):
        """Test that 4th-order requires n_ghost >= 2."""
        grid = Grid1D(nx=64, x_min=0.0, x_max=1.0, n_ghost=1)
        f = jnp.zeros(grid.nx + 2 * grid.n_ghost)

        with pytest.raises(ValueError, match="n_ghost >= 2"):
            d1_ho_1d(f, grid)


class TestFourthOrder2D:
    """Test 4th-order 2D operators."""

    def test_d1_x_smooth(self):
        """Test D1 in x-direction on sin(x)*cos(y)."""
        grid = Grid2D(nx=32, ny=32, x_min=0.0, x_max=2*jnp.pi,
                      y_min=0.0, y_max=2*jnp.pi, n_ghost=2)

        x, y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(x) * jnp.cos(y)
        df_dx_exact = jnp.cos(x) * jnp.cos(y)

        df_dx = d1_ho_2d(f, grid, axis=1)

        ng = grid.n_ghost
        interior = (slice(ng, ng + grid.ny), slice(ng, ng + grid.nx))
        error = jnp.max(jnp.abs(df_dx[interior] - df_dx_exact[interior]))
        assert error < 1e-4, f"D1_x error: {error}"

    def test_d1_y_smooth(self):
        """Test D1 in y-direction on sin(x)*cos(y)."""
        grid = Grid2D(nx=32, ny=32, x_min=0.0, x_max=2*jnp.pi,
                      y_min=0.0, y_max=2*jnp.pi, n_ghost=2)

        x, y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(x) * jnp.cos(y)
        df_dy_exact = -jnp.sin(x) * jnp.sin(y)

        df_dy = d1_ho_2d(f, grid, axis=0)

        ng = grid.n_ghost
        interior = (slice(ng, ng + grid.ny), slice(ng, ng + grid.nx))
        error = jnp.max(jnp.abs(df_dy[interior] - df_dy_exact[interior]))
        assert error < 1e-4, f"D1_y error: {error}"

    def test_laplacian_2d(self):
        """Test 2D Laplacian on sin(x)*sin(y)."""
        grid = Grid2D(nx=32, ny=32, x_min=0.0, x_max=2*jnp.pi,
                      y_min=0.0, y_max=2*jnp.pi, n_ghost=2)

        x, y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(x) * jnp.sin(y)
        # Laplacian of sin(x)*sin(y) = -2*sin(x)*sin(y)
        lap_exact = -2.0 * f

        lap_num = laplacian_ho_2d(f, grid)

        ng = grid.n_ghost
        interior = (slice(ng, ng + grid.ny), slice(ng, ng + grid.nx))
        error = jnp.max(jnp.abs(lap_num[interior] - lap_exact[interior]))
        assert error < 1e-4, f"Laplacian error: {error}"

    def test_2d_convergence(self):
        """Test convergence in 2D."""
        errors = []
        ns = [16, 32, 64]

        for n in ns:
            grid = Grid2D(nx=n, ny=n, x_min=0.0, x_max=2*jnp.pi,
                          y_min=0.0, y_max=2*jnp.pi, n_ghost=2)

            x, y = grid.meshgrid(include_ghost=True)
            f = jnp.sin(x) * jnp.sin(y)
            lap_exact = -2.0 * f

            lap_num = laplacian_ho_2d(f, grid)

            ng = grid.n_ghost
            interior = (slice(ng, ng + grid.ny), slice(ng, ng + grid.nx))
            error = jnp.max(jnp.abs(lap_num[interior] - lap_exact[interior]))
            errors.append(float(error))

        # Check convergence rate
        rates = []
        for i in range(1, len(errors)):
            rate = jnp.log2(errors[i - 1] / errors[i])
            rates.append(float(rate))

        avg_rate = sum(rates) / len(rates)
        assert avg_rate > 3.5, f"2D convergence rate {avg_rate} not 4th-order"


class TestFFTSymbols:
    """Test FFT symbols match finite difference stencils."""

    def test_laplacian_symbol_matches_fd_1d(self):
        """Verify FFT symbol gives same result as FD stencil."""
        n = 64
        dx = 2 * jnp.pi / n
        ng = 2

        # Create test function (single Fourier mode)
        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        k_mode = 3
        f = jnp.cos(k_mode * x)

        # FD result
        d2f_fd = d2_fourth_order_1d(f, dx, ng, n)
        d2f_fd_interior = d2f_fd[ng:ng + n]

        # FFT result
        f_interior = f[ng:ng + n]
        f_hat = jnp.fft.fft(f_interior)
        lam = fd_laplacian_symbol_ho_1d(n, dx)
        d2f_fft = jnp.real(jnp.fft.ifft(lam * f_hat))

        error = jnp.max(jnp.abs(d2f_fd_interior - d2f_fft))
        assert error < 1e-10, f"FFT symbol mismatch: {error}"

    def test_laplacian_symbol_matches_fd_2d(self):
        """Verify 2D FFT symbol matches FD stencil."""
        nx, ny = 32, 32
        dx = 2 * jnp.pi / nx
        dy = 2 * jnp.pi / ny
        ng = 2

        grid = Grid2D(nx=nx, ny=ny, x_min=0.0, x_max=2*jnp.pi,
                      y_min=0.0, y_max=2*jnp.pi, n_ghost=ng)

        x, y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(2 * x) * jnp.sin(3 * y)

        # FD result
        lap_fd = laplacian_ho_2d(f, grid)
        lap_fd_interior = lap_fd[ng:ng + ny, ng:ng + nx]

        # FFT result
        f_interior = f[ng:ng + ny, ng:ng + nx]
        f_hat = jnp.fft.fft2(f_interior)
        lam = fd_laplacian_symbol_ho_2d(ny, nx, dy, dx)
        lap_fft = jnp.real(jnp.fft.ifft2(lam * f_hat))

        error = jnp.max(jnp.abs(lap_fd_interior - lap_fft))
        assert error < 1e-10, f"2D FFT symbol mismatch: {error}"

    def test_d1_symbol_matches_fd(self):
        """Test first derivative symbol."""
        n = 64
        dx = 2 * jnp.pi / n
        ng = 2

        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        k_mode = 4
        f = jnp.sin(k_mode * x)

        # FD result
        df_fd = d1_fourth_order_1d(f, dx, ng, n)
        df_fd_interior = df_fd[ng:ng + n]

        # FFT result
        f_interior = f[ng:ng + n]
        f_hat = jnp.fft.fft(f_interior)
        lam = fd_d1_symbol_ho_1d(n, dx)
        df_fft = jnp.real(jnp.fft.ifft(lam * f_hat))

        error = jnp.max(jnp.abs(df_fd_interior - df_fft))
        assert error < 1e-10, f"D1 FFT symbol mismatch: {error}"


class TestComparisonWithSecondOrder:
    """Compare 4th-order accuracy with 2nd-order."""

    def test_4th_order_more_accurate(self):
        """4th-order should be more accurate than 2nd-order for same grid."""
        n = 32
        ng = 2
        dx = 2 * jnp.pi / n

        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        f = jnp.sin(x)
        d2f_exact = -jnp.sin(x)

        # 4th-order
        d2f_4th = d2_fourth_order_1d(f, dx, ng, n)
        error_4th = jnp.max(jnp.abs(d2f_4th[ng:ng + n] - d2f_exact[ng:ng + n]))

        # 2nd-order (uses ng=1 but we have ng=2)
        grid = Grid1D(nx=n, x_min=0.0, x_max=2 * jnp.pi, n_ghost=ng)
        d2f_2nd = d2_central_1d(f, grid)
        error_2nd = jnp.max(jnp.abs(d2f_2nd[ng:ng + n] - d2f_exact[ng:ng + n]))

        # 4th-order should be significantly more accurate
        assert error_4th < error_2nd / 10, f"4th-order not better: {error_4th} vs {error_2nd}"

    def test_symbol_difference(self):
        """Compare FFT symbols at high wavenumbers."""
        n = 64
        dx = 1.0 / n

        lam_2nd = fd_laplacian_symbol_1d(n, dx)
        lam_4th = fd_laplacian_symbol_ho_1d(n, dx)

        # At low k, they should be similar
        assert jnp.allclose(lam_2nd[:5], lam_4th[:5], rtol=0.1)

        # At high k, they differ (4th-order is more accurate)
        # The exact Laplacian symbol is -k^2
        k = 2 * jnp.pi * jnp.fft.fftfreq(n, d=dx)
        lam_exact = -k ** 2

        # Compare mid-range wavenumbers
        mid = n // 4
        err_2nd = jnp.abs(lam_2nd[mid] - lam_exact[mid])
        err_4th = jnp.abs(lam_4th[mid] - lam_exact[mid])

        assert err_4th < err_2nd, f"4th-order symbol not better at mid-k"


class TestSixthOrder:
    """Test 6th-order operators."""

    def test_d2_sixth_order_convergence(self):
        """Verify 6th-order convergence."""
        errors = []
        ns = [16, 32, 64]

        for n in ns:
            ng = 3
            dx = 2 * jnp.pi / n

            x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
            f = jnp.sin(x)
            d2f_exact = -jnp.sin(x)

            d2f_num = d2_sixth_order_1d(f, dx, ng, n)
            error = jnp.max(jnp.abs(d2f_num[ng:ng + n] - d2f_exact[ng:ng + n]))
            errors.append(float(error))

        # Check convergence rate
        rates = []
        for i in range(1, len(errors)):
            rate = jnp.log2(errors[i - 1] / errors[i])
            rates.append(float(rate))

        avg_rate = sum(rates) / len(rates)
        assert avg_rate > 5.5, f"Convergence rate {avg_rate} not 6th-order"

    def test_6th_order_symbol(self):
        """Test 6th-order FFT symbol."""
        n = 64
        dx = 2 * jnp.pi / n
        ng = 3

        x = jnp.linspace(-ng * dx, 2 * jnp.pi + ng * dx, n + 2 * ng, endpoint=False)
        k_mode = 2
        f = jnp.cos(k_mode * x)

        # FD result
        d2f_fd = d2_sixth_order_1d(f, dx, ng, n)
        d2f_fd_interior = d2f_fd[ng:ng + n]

        # FFT result
        f_interior = f[ng:ng + n]
        f_hat = jnp.fft.fft(f_interior)
        lam = fd_laplacian_symbol_6th_1d(n, dx)
        d2f_fft = jnp.real(jnp.fft.ifft(lam * f_hat))

        error = jnp.max(jnp.abs(d2f_fd_interior - d2f_fft))
        assert error < 1e-10, f"6th-order symbol mismatch: {error}"


class TestSymbolSelectors:
    """Test get_laplacian_symbol functions."""

    def test_get_symbol_1d_orders(self):
        """Test 1D symbol selector."""
        n = 64
        dx = 0.1

        lam_2 = get_laplacian_symbol_1d(n, dx, order=2)
        lam_4 = get_laplacian_symbol_1d(n, dx, order=4)
        lam_6 = get_laplacian_symbol_1d(n, dx, order=6)

        assert lam_2.shape == (n,)
        assert lam_4.shape == (n,)
        assert lam_6.shape == (n,)

        # They should be different
        assert not jnp.allclose(lam_2, lam_4)
        assert not jnp.allclose(lam_4, lam_6)

    def test_get_symbol_2d_orders(self):
        """Test 2D symbol selector."""
        ny, nx = 32, 32
        dy, dx = 0.1, 0.1

        lam_2 = get_laplacian_symbol_2d(ny, nx, dy, dx, order=2)
        lam_4 = get_laplacian_symbol_2d(ny, nx, dy, dx, order=4)

        assert lam_2.shape == (ny, nx)
        assert lam_4.shape == (ny, nx)
        assert not jnp.allclose(lam_2, lam_4)

    def test_invalid_order(self):
        """Test error on invalid order."""
        with pytest.raises(ValueError, match="Unsupported order"):
            get_laplacian_symbol_1d(64, 0.1, order=3)

        with pytest.raises(ValueError, match="Unsupported order"):
            get_laplacian_symbol_2d(32, 32, 0.1, 0.1, order=6)


class TestOperatorOrder:
    """Test OperatorOrder constants."""

    def test_constants(self):
        """Test order constants."""
        assert OperatorOrder.SECOND == 2
        assert OperatorOrder.FOURTH == 4
        assert OperatorOrder.SIXTH == 6
