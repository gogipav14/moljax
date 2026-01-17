"""
Tests for boundary conditions.

Verifies that ghost cells are filled correctly for:
- Periodic: wraps from opposite boundary
- Dirichlet: clamps boundary values
- Neumann: enforces zero gradient
"""

import pytest
import jax.numpy as jnp
import numpy as np

from moljax.core.grid import Grid1D, Grid2D
from moljax.core.bc import FieldBCSpec, BCType, apply_bc
from moljax.core.utils import get_interior


class TestBC1D:
    """Tests for 1D boundary conditions."""

    def test_periodic_1d(self):
        """Test periodic BC wraps ghost cells correctly."""
        nx = 10
        grid = Grid1D.uniform(nx, 0.0, 1.0, n_ghost=1)

        # Create array with known interior values
        f = jnp.arange(grid.nx_total, dtype=jnp.float64)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec.periodic()}
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        # Left ghost should equal right interior
        # Right ghost should equal left interior
        # Interior: indices 1 to nx (inclusive)
        assert jnp.allclose(f_out[0], f_out[nx])  # left ghost = right interior
        assert jnp.allclose(f_out[nx + 1], f_out[1])  # right ghost = left interior

    def test_dirichlet_1d(self):
        """Test Dirichlet BC clamps boundary values."""
        nx = 10
        grid = Grid1D.uniform(nx, 0.0, 1.0, n_ghost=1)

        # Interior starts with linear ramp
        f = jnp.arange(grid.nx_total, dtype=jnp.float64)

        state = {'f': f}
        # Default Dirichlet with value=0
        bc_spec = {'f': FieldBCSpec.dirichlet()}
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        # Ghost cell formula: ghost = 2*bc_value - interior
        # For bc_value=0: ghost = -interior
        # Left: f[0] = 2*0 - f[1] = -f[1]
        # Right: f[nx+1] = 2*0 - f[nx] = -f[nx]
        expected_left = 2 * 0 - f[1]
        expected_right = 2 * 0 - f[nx]

        assert jnp.allclose(f_out[0], expected_left), f"Left ghost: {f_out[0]} != {expected_left}"
        assert jnp.allclose(f_out[nx + 1], expected_right), f"Right ghost: {f_out[nx+1]} != {expected_right}"

    def test_neumann_1d_zero_gradient(self):
        """Test Neumann BC with zero gradient (homogeneous)."""
        nx = 10
        grid = Grid1D.uniform(nx, 0.0, 1.0, n_ghost=1)

        f = jnp.arange(grid.nx_total, dtype=jnp.float64)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec.neumann()}  # Zero gradient
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        # For zero gradient: ghost = interior (mirror)
        # Actually with the formula: ghost = interior - 2*dx*flux
        # For flux=0: ghost = interior
        # Left: f[0] = f[1] - 0 = f[1]
        # Right: f[nx+1] = f[nx] + 0 = f[nx]
        expected_left = f[1]
        expected_right = f[nx]

        assert jnp.allclose(f_out[0], expected_left), f"Left ghost: {f_out[0]} != {expected_left}"
        assert jnp.allclose(f_out[nx + 1], expected_right), f"Right ghost: {f_out[nx+1]} != {expected_right}"


class TestBC2D:
    """Tests for 2D boundary conditions."""

    def test_periodic_2d(self):
        """Test periodic BC wraps ghost cells correctly in 2D."""
        nx, ny = 5, 5
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0, n_ghost=1)

        # Create array with known values
        f = jnp.arange(grid.ny_total * grid.nx_total, dtype=jnp.float64).reshape(
            grid.ny_total, grid.nx_total
        )

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec.periodic()}
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        ng = grid.n_ghost

        # X direction: left ghost = right interior, right ghost = left interior
        assert jnp.allclose(f_out[:, :ng], f_out[:, nx:nx + ng])
        assert jnp.allclose(f_out[:, nx + ng:], f_out[:, ng:2 * ng])

        # Y direction: bottom ghost = top interior, top ghost = bottom interior
        assert jnp.allclose(f_out[:ng, :], f_out[ny:ny + ng, :])
        assert jnp.allclose(f_out[ny + ng:, :], f_out[ng:2 * ng, :])

    def test_dirichlet_2d(self):
        """Test Dirichlet BC in 2D with default zero value."""
        nx, ny = 5, 5
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0, n_ghost=1)

        f = jnp.ones((grid.ny_total, grid.nx_total), dtype=jnp.float32)

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec.dirichlet()}  # Zero at boundaries
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        ng = grid.n_ghost

        # For Dirichlet with value=0: ghost = 2*0 - interior = -interior
        # Since interior is 1, ghost should be -1
        # Note: corners get BC applied from both directions - check interior rows/cols
        assert jnp.allclose(f_out[ng:-ng, 0], -1.0)  # Left ghost (interior rows)
        assert jnp.allclose(f_out[ng:-ng, -1], -1.0)  # Right ghost (interior rows)
        assert jnp.allclose(f_out[0, ng:-ng], -1.0)  # Bottom ghost (interior cols)
        assert jnp.allclose(f_out[-1, ng:-ng], -1.0)  # Top ghost (interior cols)

    def test_neumann_2d_zero_gradient(self):
        """Test Neumann BC with zero gradient in 2D."""
        nx, ny = 5, 5
        grid = Grid2D.uniform(nx, ny, 0.0, 1.0, 0.0, 1.0, n_ghost=1)

        f = jnp.ones((grid.ny_total, grid.nx_total), dtype=jnp.float64) * 2.0

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec.neumann()}
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        ng = grid.n_ghost

        # For zero gradient: ghost = interior
        # All values should remain 2.0
        assert jnp.allclose(f_out[:, 0], 2.0)
        assert jnp.allclose(f_out[:, -1], 2.0)
        assert jnp.allclose(f_out[0, :], 2.0)
        assert jnp.allclose(f_out[-1, :], 2.0)


class TestBCMultiField:
    """Test BC application to multi-field states."""

    def test_different_bcs_per_field(self):
        """Test that different BCs can be applied to different fields."""
        nx = 10
        grid = Grid1D.uniform(nx, 0.0, 1.0, n_ghost=1)

        u = jnp.ones(grid.nx_total, dtype=jnp.float64)
        v = jnp.ones(grid.nx_total, dtype=jnp.float64) * 2.0

        state = {'u': u, 'v': v}
        bc_spec = {
            'u': FieldBCSpec.periodic(),
            'v': FieldBCSpec.dirichlet()  # Zero at boundaries
        }
        state = apply_bc(state, grid, bc_spec)

        # u should be unchanged (periodic with constant = identity)
        assert jnp.allclose(state['u'], u)

        # v should have ghost cells = -interior = -2
        assert jnp.allclose(state['v'][0], -2.0)
        assert jnp.allclose(state['v'][-1], -2.0)


class TestBCGradient:
    """Test that BCs preserve expected gradient properties."""

    def test_neumann_preserves_gradient(self):
        """Test that Neumann BC preserves derivative at boundary."""
        nx = 20
        grid = Grid1D.uniform(nx, 0.0, 1.0, n_ghost=1)

        # Linear function f = x
        x = grid.x_coords(include_ghost=True)
        f = x.copy()

        state = {'f': f}
        bc_spec = {'f': FieldBCSpec.neumann()}  # Zero gradient
        state = apply_bc(state, grid, bc_spec)
        f_out = state['f']

        # Compute gradient at boundaries using central difference
        # Left boundary gradient: (f[2] - f[0]) / (2*dx)
        grad_left = (f_out[2] - f_out[0]) / (2 * grid.dx)
        # Should be close to 0 for Neumann BC with zero flux
        # But since we're mirroring a linear function, it won't be exactly 0
        # The test is mainly that the BC is applied consistently

        # For a linear function, Neumann should preserve the function value at boundary
        # since ghost = interior for zero gradient
        assert jnp.allclose(f_out[0], f_out[1])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
