"""Regression tests for the fixed-state Brusselator dt-sweep report."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "benchmarks/results/brusselator_dt_sweep.json"
REPORT = ROOT / "benchmarks/results/BRUSSELATOR_DT_SWEEP_RESULTS.md"
RENDERER = ROOT / "benchmarks/make_brusselator_dt_sweep_results.py"
REFERENCE = re.compile(r"⟦`brusselator_dt_sweep\.json#([^`]*)`⟧")


def _load_renderer() -> ModuleType:
    spec = importlib.util.spec_from_file_location("make_brusselator_dt_sweep_results", RENDERER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve(document: object, pointer: str) -> object:
    value = document
    if not pointer:
        return value
    assert pointer.startswith("/")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]  # type: ignore[index]
    return value


def test_dt_sweep_report_is_deterministic_current_and_fully_referenced() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    renderer = _load_renderer()
    first = renderer.render(payload)
    second = renderer.render(payload)
    assert first == second
    assert REPORT.read_text(encoding="utf-8") == first
    pointers = REFERENCE.findall(first)
    assert pointers
    for pointer in pointers:
        _resolve(payload, pointer)


def test_dt_sweep_pre_registered_decision_is_auditable() -> None:
    payload = json.loads(RESULT.read_text(encoding="utf-8"))
    decision = payload["analysis"]["decision"]
    assert decision["d1_material_degradation"]["passes"]
    assert decision["d2_absolute_cost"]["passes"]
    assert decision["d3_recoverability"]["passes"]
    assert decision["all_conditions_pass"]
    assert decision["decision"] == "BUILD"
