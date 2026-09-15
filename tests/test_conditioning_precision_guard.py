"""x64 precision guard coverage for the pseudospectra and non_normality modules.

Before this fix, none of arnoldi, epsilon_zero, pseudospectrum_dense,
reduced_pseudospectrum, or ritz_values (moljax/conditioning/pseudospectra.py)
and none of the public functions in moljax/conditioning/non_normality.py
called moljax._precision.require_x64, unlike numerical_range
(field_of_values.py) and linearized_operator (linearization.py). With x64
disabled, epsilon_zero(np.full((2, 2), 1e8)) returned 11.313709 for an
exactly singular matrix instead of raising.

This has to run in a fresh subprocess with x64 left disabled: every other
conditioning test module enables x64 at import time, and jax.config is
process-global, so an in-process test here would either flip x64 on for the
rest of the suite or race whichever test happens to run first.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_CHECK_SCRIPT = textwrap.dedent(
    """
    import jax
    jax.config.update("jax_enable_x64", False)

    import numpy as np

    from moljax.conditioning.non_normality import assess_preconditioner
    from moljax.conditioning.pseudospectra import arnoldi, epsilon_zero, pseudospectrum_dense

    def check(name, fn):
        try:
            fn()
        except RuntimeError as exc:
            print(f"{name}: RAISED {exc}")
        else:
            print(f"{name}: NOT_RAISED")

    check("epsilon_zero", lambda: epsilon_zero(np.full((2, 2), 1e8)))
    check(
        "pseudospectrum_dense",
        lambda: pseudospectrum_dense(lambda x: x, 2, np.array([0.0]), np.array([0.0])),
    )
    check("arnoldi", lambda: arnoldi(lambda x: x, np.ones(2), 1))
    check("assess_preconditioner", lambda: assess_preconditioner(None, np.ones(4), 0.1))
    """
)


def test_public_entry_points_require_x64_in_a_fresh_process():
    """epsilon_zero, pseudospectrum_dense, arnoldi, and assess_preconditioner
    must each raise RuntimeError (require_x64's error) with x64 disabled,
    rather than silently computing on float32 data.
    """
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=ROOT)
    result = subprocess.run(
        [sys.executable, "-c", _CHECK_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
        check=True,
    )
    lines = {
        line.split(":", 1)[0]: line
        for line in result.stdout.strip().splitlines()
        if ":" in line
    }
    assert set(lines) == {
        "epsilon_zero",
        "pseudospectrum_dense",
        "arnoldi",
        "assess_preconditioner",
    }, result.stdout + result.stderr
    for name, line in lines.items():
        assert "RAISED" in line, (
            f"{name} did not raise with x64 disabled: {line}\n{result.stderr}"
        )
        assert "64-bit precision is required" in line, line
