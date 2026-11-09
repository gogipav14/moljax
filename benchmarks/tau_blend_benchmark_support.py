"""Shared provenance and JSON helpers for tau-blend rewrite benchmarks."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax
import jaxlib
import numpy as np
import scipy

REPOSITORY = Path(__file__).resolve().parents[1]


def sha256_file(path: str | Path) -> str:
    """Return one file's SHA256 digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    """Return the local revision used for a benchmark run."""
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def benchmark_provenance(script: str | Path, wall_seconds: float) -> dict[str, Any]:
    """Return the required environment, source, and timing provenance."""
    script_path = Path(script).resolve()
    return {
        "python": sys.version,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "backend": jax.default_backend(),
        "git_commit": git_head(),
        "script": str(script_path.relative_to(REPOSITORY)),
        "script_sha256": sha256_file(script_path),
        "wall_seconds": float(wall_seconds),
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write deterministic, human-auditable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


__all__ = ["benchmark_provenance", "git_head", "sha256_file", "write_json"]
