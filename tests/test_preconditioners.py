"""
Tests for the Nyquist-mode masking in _odd_symbol_wavenumber
(moljax/core/preconditioners.py), used by both the FFT preconditioners
and AdvectionDiffusionOperator.

_odd_symbol_wavenumber zeroes the derivative of the self-paired Nyquist
mode of an even-length axis. It used to identify that mode by comparing
floating-point wavenumbers against pi/dx with a relative tolerance of
1e-12; fftfreq's own float32 rounding error at that bin is about 1.4e-6
relative, which swamped the tolerance and let the mode through unmasked.
The fix identifies the bin by integer index instead.
"""

import os
import subprocess
import sys

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from moljax.core.preconditioners import _odd_symbol_wavenumber

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestOddSymbolWavenumberIndexBased:
    """Direct tests of the masking helper on synthetic float32 arrays."""

    def test_full_spectrum_nyquist_masked_despite_float32_perturbation(self):
        """A relative 2e-7 perturbation at the Nyquist bin of a float32,
        full-spectrum k array (as fftfreq's own float32 rounding would
        produce) must not stop the bin from being masked.
        """
        n, dx = 10, 0.1
        k = (2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)).astype(jnp.float32)
        nyquist_idx = n // 2
        perturbed = k.at[nyquist_idx].multiply(1.0 + 2e-7)

        masked = _odd_symbol_wavenumber(perturbed, n)

        assert float(masked[nyquist_idx]) == 0.0
        other = jnp.arange(n) != nyquist_idx
        assert jnp.array_equal(masked[other], perturbed[other])

    def test_rfft_half_spectrum_nyquist_masked_despite_float32_perturbation(self):
        """Same check for an rfft half-spectrum array, where the Nyquist
        bin is the array's last index rather than its middle one.
        """
        n, dx = 10, 0.1
        k = (2.0 * jnp.pi * jnp.fft.rfftfreq(n, d=dx)).astype(jnp.float32)
        assert k.shape[0] == n // 2 + 1
        nyquist_idx = k.shape[0] - 1
        perturbed = k.at[nyquist_idx].multiply(1.0 + 2e-7)

        masked = _odd_symbol_wavenumber(perturbed, n)

        assert float(masked[nyquist_idx]) == 0.0
        other = jnp.arange(k.shape[0]) != nyquist_idx
        assert jnp.array_equal(masked[other], perturbed[other])

    def test_odd_length_axis_has_no_nyquist_bin(self):
        """An odd-length axis has no bin exactly at +-pi/dx, so nothing is masked."""
        n, dx = 9, 0.1
        k = (2.0 * jnp.pi * jnp.fft.fftfreq(n, d=dx)).astype(jnp.float32)
        masked = _odd_symbol_wavenumber(k, n)
        assert jnp.array_equal(masked, k)

    def test_2d_broadcast_nyquist_masked_along_named_axis_float32(self):
        """A 2D broadcast kx/ky array (as build_wavenumbers_2d produces) is
        masked only along the axis carrying the transformed dimension, and
        the Nyquist mode along either axis is caught: kx along the last
        axis (columns), ky along the first (rows).
        """
        ny, nx = 6, 8  # both even, so both axes have a Nyquist bin
        kx_1d = (2.0 * jnp.pi * jnp.fft.fftfreq(nx, d=0.1)).astype(jnp.float32)
        ky_1d = (2.0 * jnp.pi * jnp.fft.fftfreq(ny, d=0.2)).astype(jnp.float32)
        kx = jnp.broadcast_to(kx_1d, (ny, nx))
        ky = jnp.broadcast_to(ky_1d[:, None], (ny, nx))

        masked_kx = _odd_symbol_wavenumber(kx, nx, axis=-1)
        masked_ky = _odd_symbol_wavenumber(ky, ny, axis=0)

        assert jnp.all(masked_kx[:, nx // 2] == 0.0)
        assert jnp.all(masked_ky[ny // 2, :] == 0.0)

        cols = jnp.arange(nx) != nx // 2
        rows = jnp.arange(ny) != ny // 2
        assert jnp.array_equal(masked_kx[:, cols], kx[:, cols])
        assert jnp.array_equal(masked_ky[rows, :], ky[rows, :])


class TestNyquistFloat32Subprocess:
    """The main suite runs with jax_enable_x64 True, where float64
    fftfreq's rounding error at the Nyquist bin (about 1e-16 relative)
    never triggered the old relative-tolerance bug. These run in a fresh
    subprocess with x64 left disabled so JAX genuinely computes in
    float32, matching the report's reproduction.
    """

    def _run(self, code: str) -> dict:
        env = dict(os.environ, JAX_PLATFORMS='cpu', PYTHONPATH=ROOT)
        out = subprocess.run(
            [sys.executable, '-c', code],
            capture_output=True, text=True, env=env, cwd=ROOT, check=True,
        )
        lines = [line for line in out.stdout.strip().splitlines() if line.strip()]
        result = {}
        for line in lines:
            key, _, value = line.partition(' ')
            result[key] = value
        assert result, f"no output from subprocess: {out.stdout!r} {out.stderr!r}"
        return result

    def test_1d_grid_nyquist_eigenvalue_and_exp_matvec_and_helmholtz(self):
        """Grid1D.uniform(10, 0, 1), v=1, D=0, float32: reproduces the
        exact scenario from the defect report.

        Before the fix: Nyquist eigenvalue stayed 31.4159j,
        exp_matvec((-1)**i, 0.1) returned -u instead of u, and the
        Helmholtz solve residual reached 0.908.
        """
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "import numpy as np\n"
            "import jax.numpy as jnp\n"
            "from moljax.core.grid import Grid1D\n"
            "from moljax.core.fft_operators import AdvectionDiffusionOperator\n"
            "grid = Grid1D.uniform(10, 0.0, 1.0)\n"
            "op = AdvectionDiffusionOperator(grid, v=1.0, D=0.0, dtype=jnp.float32)\n"
            "lam = op.eigenvalues\n"
            "nyquist_idx = grid.nx // 2\n"
            "print('NYQUIST_EIG_ABS', float(jnp.abs(lam[nyquist_idx])))\n"
            "u = jnp.asarray(((-1.0) ** np.arange(10)).astype(np.float32))\n"
            "em = op.exp_matvec(u, 0.1)\n"
            "print('EXP_MATVEC_MAX_DIFF_FROM_U', float(jnp.max(jnp.abs(em - u))))\n"
            "dt = 0.1\n"
            "sol = op.solve(u, dt)\n"
            "resid = sol - dt * op.matvec(sol) - u\n"
            "print('HELMHOLTZ_RESIDUAL', float(jnp.max(jnp.abs(resid))))\n"
        )
        result = self._run(code)
        assert float(result['NYQUIST_EIG_ABS']) == 0.0, result
        assert float(result['EXP_MATVEC_MAX_DIFF_FROM_U']) < 1e-5, result
        assert float(result['HELMHOLTZ_RESIDUAL']) < 1e-4, result

    def test_2d_grid_nyquist_along_either_axis_float32(self):
        """2D float32 case with the Nyquist mode present along both axes
        (even nx and ny): the full-spectrum and half-spectrum (rfft)
        preconditioner caches must mask it the same way and agree.
        """
        code = (
            "import jax\n"
            "jax.config.update('jax_enable_x64', False)\n"
            "import numpy as np\n"
            "import jax.numpy as jnp\n"
            "from moljax.core.grid import Grid2D\n"
            "from moljax.core.fft_solvers import create_fft_cache_2d, create_fft_cache_2d_rfft\n"
            "from moljax.core.preconditioners import (\n"
            "    FFTAdvectionDiffusionPreconditioner, PrecondContext, _odd_symbol_wavenumber,\n"
            ")\n"
            "grid = Grid2D.uniform(10, 8, 0.0, 1.0, 0.0, 1.0)\n"
            "cache_full = create_fft_cache_2d(grid, jnp.float32)\n"
            "cache_rfft = create_fft_cache_2d_rfft(grid, jnp.float32)\n"
            "kx_full = _odd_symbol_wavenumber(cache_full.kx, grid.nx, axis=-1)\n"
            "ky_full = _odd_symbol_wavenumber(cache_full.ky, grid.ny, axis=0)\n"
            "print('KX_NYQUIST_COL_ABS', float(jnp.max(jnp.abs(kx_full[:, grid.nx // 2]))))\n"
            "print('KY_NYQUIST_ROW_ABS', float(jnp.max(jnp.abs(ky_full[grid.ny // 2, :]))))\n"
            "rng = np.random.default_rng(0)\n"
            "field = jnp.asarray(rng.standard_normal((grid.ny, grid.nx)).astype(np.float32))\n"
            "ctx = PrecondContext(grid=grid, dt=0.05, params={'D': 0.3, 'v': (1.0, 0.5)})\n"
            "precond_full = FFTAdvectionDiffusionPreconditioner(\n"
            "    field_diffusivity_keys={'u': 'D'}, field_velocity_keys={'u': 'v'}, fft_cache=cache_full)\n"
            "precond_rfft = FFTAdvectionDiffusionPreconditioner(\n"
            "    field_diffusivity_keys={'u': 'D'}, field_velocity_keys={'u': 'v'}, fft_cache=cache_rfft)\n"
            "out_full = precond_full.apply({'u': field}, ctx)['u']\n"
            "out_rfft = precond_rfft.apply({'u': field}, ctx)['u']\n"
            "print('FULL_VS_RFFT_MAX_DIFF', float(jnp.max(jnp.abs(out_full - out_rfft))))\n"
        )
        result = self._run(code)
        assert float(result['KX_NYQUIST_COL_ABS']) == 0.0, result
        assert float(result['KY_NYQUIST_ROW_ABS']) == 0.0, result
        assert float(result['FULL_VS_RFFT_MAX_DIFF']) < 1e-4, result
