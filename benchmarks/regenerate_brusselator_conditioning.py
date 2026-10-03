#!/usr/bin/env python3
"""Sequential, resumable regeneration of every Brusselator conditioning preset.

Each preset runs in a fresh numerical process.  Its source states are shared
only through v4 fingerprint-and-SHA256 validated artifacts, while each
diagnostic record is atomically checkpointed by ``brusselator_conditioning``.
This controller is deliberately single-process for numerical work: it never
starts a second preset until the first child has exited successfully.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

STUDIES = (
    "screen_64",
    "developed_64",
    "fixed_dt_256",
    "hopf_continuation_256",
)
EXPECTED_RECORDS = {
    "screen_64": 8,
    "developed_64": 12,
    "fixed_dt_256": 8,
    "hopf_continuation_256": 4,
}
OUTPUT_NAMES = {
    "screen_64": "brusselator_conditioning.json",
    "developed_64": "brusselator_conditioning_developed.json",
    "fixed_dt_256": "brusselator_conditioning_fixed_dt.json",
    "hopf_continuation_256": "brusselator_conditioning_hopf_continuation.json",
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _completed(path: Path, study: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "completed" and len(payload.get("records", ())) == EXPECTED_RECORDS[study]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--studies", nargs="+", choices=STUDIES, default=list(STUDIES))
    args = parser.parse_args()

    repository = Path(__file__).resolve().parent.parent
    benchmark = repository / "benchmarks" / "brusselator_conditioning.py"
    checkpoint_dir = args.checkpoint_dir.resolve()
    source_cache = checkpoint_dir / "source_states"
    record_checkpoints = checkpoint_dir / "record_checkpoints"
    # Keep regenerated artifacts outside the tracked result directory until
    # Stage 3 has checked their tally and is ready to regenerate prose/plots.
    results_dir = checkpoint_dir / "results"
    manifest_path = checkpoint_dir / "regeneration_manifest.json"
    manifest: dict[str, Any] = {
        "schema": "brusselator_conditioning_regeneration_v1",
        "studies": {},
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repository)
    for study in args.studies:
        output = results_dir / OUTPUT_NAMES[study]
        if _completed(output, study):
            print(f"SKIP completed study={study} output={output}", flush=True)
            manifest["studies"][study] = {"status": "completed", "output": str(output)}
            _atomic_json(manifest_path, manifest)
            continue
        command = (
            sys.executable,
            str(benchmark),
            "--study",
            study,
            "--output",
            str(output),
            "--source-state-cache-dir",
            str(source_cache),
            "--record-checkpoint",
            str(record_checkpoints / f"{study}.json"),
        )
        print(f"START study={study} command={' '.join(command)}", flush=True)
        subprocess.run(command, cwd=repository, env=environment, check=True)
        if not _completed(output, study):
            raise RuntimeError(f"completed child wrote an invalid result: {output}")
        manifest["studies"][study] = {"status": "completed", "output": str(output)}
        _atomic_json(manifest_path, manifest)
        print(f"COMPLETE study={study} output={output}", flush=True)
    print("BRUSSELATOR_REGENERATION_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
