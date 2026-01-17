"""
Tests for spatial operators.

Verifies that D1 and D2 operators give expected results on sine waves
for periodic BCs (known derivatives).
"""

import pytest
import jax.numpy as jnp
import numpy as np

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.bc import FieldBCSpec, BCType, apply_bc
from moljax.core.operators import (
    d1_central_1d, d1_upwind_1d, d2_central_1d, laplacian_1d,
    d1_central_2d, d1_upwind_2d, d2_central_2d, laplacian_2d
)
from moljax.core.utils import get_interior


class TestOperators1D:
    """Tests for 1D operators."""

    def test_d1_central_sine_periodic(self):
        """Test D1 on sin(2*pi*x) with periodic BC.

        d/dx[sin(2*pi*x)] = 2*pi*cos(2*pi*x)
        """
        nx = 100
        grid = Grid1D.uniform(nx, 0.0, 1.0)

        # Create padded array with sin wave
        x_full = grid.x_coords(include_ghost=True)
        f = jnp.sin(2 * jnp.pi * x_full)

        # Apply periodic BC
        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        # Compute derivative
        df = d1_central_1d(f, grid)
        df_interior = get_interior(df, grid)

        # Expected: 2*pi*cos(2*pi*x)
        x = grid.x_coords(include_ghost=False)
        expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)

        # L-inf error should be O(dx^2) for central difference
        error = jnp.max(jnp.abs(df_interior - expected))
        assert error < 0.01, f"D1 central error too large: {error}"

    def test_d2_central_sine_periodic(self):
        """Test D2 on sin(2*pi*x) with periodic BC.

        d^2/dx^2[sin(2*pi*x)] = -(2*pi)^2 * sin(2*pi*x)
        """
        nx = 100
        grid = Grid1D.uniform(nx, 0.0, 1.0)

        x_full = grid.x_coords(include_ghost=True)
        f = jnp.sin(2 * jnp.pi * x_full)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        d2f = d2_central_1d(f, grid)
        d2f_interior = get_interior(d2f, grid)

        x = grid.x_coords(include_ghost=False)
        expected = -(2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * x)

        error = jnp.max(jnp.abs(d2f_interior - expected))
        assert error < 0.1, f"D2 central error too large: {error}"

    def test_d1_upwind_constant_velocity(self):
        """Test upwind scheme with constant positive velocity on periodic function."""
        nx = 50
        grid = Grid1D.uniform(nx, 0.0, 1.0)

        # Use sine wave which is periodic
        x_full = grid.x_coords(include_ghost=True)
        f = jnp.sin(2 * jnp.pi * x_full)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        # Positive velocity: uses backward difference
        v = 1.0
        df = d1_upwind_1d(f, v, grid)
        df_interior = get_interior(df, grid)

        x = grid.x_coords(include_ghost=False)
        expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)

        error = jnp.max(jnp.abs(df_interior - expected))
        # Upwind is first-order, so expect larger error but bounded
        assert error < 0.5, f"D1 upwind error too large: {error}"


class TestOperators2D:
    """Tests for 2D operators."""

    def test_d1_central_2d_x_sine(self):
        """Test D1 in x-direction on sin(2*pi*x)."""
        nx, ny = 50, 50
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0)

        X, Y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(2 * jnp.pi * X)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        df_dx = d1_central_2d(f, grid, axis=1)
        df_dx_interior = get_interior(df_dx, grid)

        X_int, _ = grid.meshgrid(include_ghost=False)
        expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * X_int)

        error = jnp.max(jnp.abs(df_dx_interior - expected))
        assert error < 0.02, f"D1 x-direction error too large: {error}"

    def test_d1_central_2d_y_sine(self):
        """Test D1 in y-direction on sin(2*pi*y)."""
        nx, ny = 50, 50
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0)

        X, Y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(2 * jnp.pi * Y)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        df_dy = d1_central_2d(f, grid, axis=0)
        df_dy_interior = get_interior(df_dy, grid)

        _, Y_int = grid.meshgrid(include_ghost=False)
        expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * Y_int)

        error = jnp.max(jnp.abs(df_dy_interior - expected))
        assert error < 0.02, f"D1 y-direction error too large: {error}"

    def test_laplacian_2d_sine(self):
        """Test Laplacian on sin(2*pi*x)*sin(2*pi*y).

        Laplacian = -2*(2*pi)^2 * sin(2*pi*x)*sin(2*pi*y)
        """
        nx, ny = 50, 50
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0)

        X, Y = grid.meshgrid(include_ghost=True)
        f = jnp.sin(2 * jnp.pi * X) * jnp.sin(2 * jnp.pi * Y)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        lap_f = laplacian_2d(f, grid)
        lap_f_interior = get_interior(lap_f, grid)

        X_int, Y_int = grid.meshgrid(include_ghost=False)
        expected = -2 * (2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * X_int) * jnp.sin(2 * jnp.pi * Y_int)

        error = jnp.max(jnp.abs(lap_f_interior - expected))
        assert error < 0.5, f"Laplacian error too large: {error}"

    def test_d1_upwind_2d(self):
        """Test upwind scheme in 2D with constant velocity on periodic function."""
        nx, ny = 30, 30
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0)

        X, Y = grid.meshgrid(include_ghost=True)
        # Use sine wave which is periodic
        f = jnp.sin(2 * jnp.pi * X)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
        state = apply_bc(state, grid, bc_spec)
        f = state['f']

        # df/dx = 2*pi*cos(2*pi*x)
        vx = 1.0
        df_dx = d1_upwind_2d(f, vx, grid, axis=1)
        df_dx_interior = get_interior(df_dx, grid)

        X_int, _ = grid.meshgrid(include_ghost=False)
        expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * X_int)
        error = jnp.max(jnp.abs(df_dx_interior - expected))
        # Upwind is first-order, so expect larger error on coarse grid
        assert error < 1.0, f"Upwind 2D error too large: {error}"


class TestOperatorConvergence:
    """Test that operators converge at expected rates."""

    def test_d1_second_order_convergence(self):
        """Verify D1 central is 2nd order accurate."""
        errors = []
        dxs = []

        for nx in [20, 40, 80]:
            grid = Grid1D.uniform(nx, 0.0, 1.0)
            x_full = grid.x_coords(include_ghost=True)
            f = jnp.sin(2 * jnp.pi * x_full)

            state = {'f': f}
            bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
            state = apply_bc(state, grid, bc_spec)
            f = state['f']

            df = d1_central_1d(f, grid)
            df_interior = get_interior(df, grid)

            x = grid.x_coords(include_ghost=False)
            expected = 2 * jnp.pi * jnp.cos(2 * jnp.pi * x)

            error = float(jnp.max(jnp.abs(df_interior - expected)))
            errors.append(error)
            dxs.append(grid.dx)

        # Check convergence rate ~ 2
        rate1 = np.log(errors[0] / errors[1]) / np.log(dxs[0] / dxs[1])
        rate2 = np.log(errors[1] / errors[2]) / np.log(dxs[1] / dxs[2])

        assert rate1 > 1.8, f"D1 convergence rate too low: {rate1}"
        assert rate2 > 1.8, f"D1 convergence rate too low: {rate2}"

    def test_d2_second_order_convergence(self):
        """Verify D2 central is 2nd order accurate."""
        errors = []
        dxs = []

        for nx in [20, 40, 80]:
            grid = Grid1D.uniform(nx, 0.0, 1.0)
            x_full = grid.x_coords(include_ghost=True)
            f = jnp.sin(2 * jnp.pi * x_full)

            state = {'f': f}
            bc_spec = {'f': FieldBCSpec(kind=BCType.PERIODIC)}
            state = apply_bc(state, grid, bc_spec)
            f = state['f']

            d2f = d2_central_1d(f, grid)
            d2f_interior = get_interior(d2f, grid)

            x = grid.x_coords(include_ghost=False)
            expected = -(2 * jnp.pi) ** 2 * jnp.sin(2 * jnp.pi * x)

            error = float(jnp.max(jnp.abs(d2f_interior - expected)))
            errors.append(error)
            dxs.append(grid.dx)

        rate1 = np.log(errors[0] / errors[1]) / np.log(dxs[0] / dxs[1])
        rate2 = np.log(errors[1] / errors[2]) / np.log(dxs[1] / dxs[2])

        assert rate1 > 1.7, f"D2 convergence rate too low: {rate1}"
        assert rate2 > 1.7, f"D2 convergence rate too low: {rate2}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
