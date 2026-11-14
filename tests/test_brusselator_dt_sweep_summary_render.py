"""The source-state-free summary renderer preserves the published report."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks/brusselator_dt_sweep_posthoc.py"
SUMMARY = ROOT / "benchmarks/results/BRUSSELATOR_DT_SWEEP_SUMMARY.md"


def test_summary_render_is_current_and_idempotent(tmp_path: Path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=str(ROOT))
    subprocess.run(
        [sys.executable, str(SCRIPT), "--render-summary-only", str(first)],
        cwd=ROOT,
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--render-summary-only",
            str(second),
            "--summary-input",
            str(first),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    assert first.read_bytes() == SUMMARY.read_bytes()
    assert second.read_bytes() == first.read_bytes()
