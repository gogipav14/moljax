"""CPU smoke tests for the FFT-preconditioned conditioning decision demo."""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import pytest

from benchmarks.conditioning_decision_demo import DemoConfig, run_decision_demo


@pytest.mark.slow
def test_small_diffusion_decision_demo(tmp_path):
    """Visited states pass the adjoint gate and cover both decision verdicts.

    The demo uses ``n_restarts=1`` (its default), so the strong verdict a
    diffusion-preconditioned state earns here is ``provisional`` rather than
    ``adequate``.  Raising ``n_restarts`` promotes it, at double the
    eigensolver cost per direction; the demo does not, on the argument that a
    smoke test should reflect the default configuration.

    ``arnoldi_steps`` is likewise left at the demo default.  The outlier gate
    abstains below four Ritz values, and with four the diffusion-preconditioned
    state's rightmost value sits about 1.6 bulk widths out against a threshold
    of two, too close to pin a verdict on.
    """
    result = run_decision_demo(
        DemoConfig(
            nx=8,
            ny=8,
            dt=2.0,
            n_states=1,
            n_angles=4,
            fov_max_iters=60,
            arnoldi_steps=8,
            pseudospectrum_points=3,
            overhead_runs=5,
            max_newton_iters=10,
            max_krylov_iters=18,
            figure_dir=str(tmp_path),
        )
    )

    assert all(record["status"] == "completed" for record in result["states"])
    assert all(record["implicit_step"]["converged"] is True for record in result["states"])
    assert all(record["adjoint_identity"] <= 1.0e-8 for record in result["states"])
    assert {record["assessment"]["verdict"] for record in result["states"]} == {
        "provisional",
        "investigate",
    }
    # Both preconditioners get a numerical-range and a pseudospectrum figure
    # (four), but only the "provisional" fft_diffusion state also gets a
    # residual-envelope figure: the "investigate" identity state fails a
    # threshold gate, and drawing a decaying envelope for a state flagged as
    # needing further preconditioner work would be exactly the false
    # confidence the verdict warns against.
    assert len(result["figures"]) == 5
    envelope_figures = [path for path in result["figures"] if "residual_envelope" in path]
    assert len(envelope_figures) == 1
    assert "identity" not in envelope_figures[0]
