"""Fail-closed rendering checks for the tau-blend mask-analysis report."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from benchmarks.make_tau_blend_mask_facts import render_facts
from benchmarks.make_tau_blend_readme import FILES, OUTPUT, _fmt, render


def _copy_results(destination: Path) -> None:
    source = OUTPUT.parent
    for filename in FILES.values():
        shutil.copy2(source / filename, destination / filename)


def _mutate(destination: Path, filename: str, change) -> None:
    path = destination / filename
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_readme_is_deterministic_current_and_whitespace_clean() -> None:
    """Two renders agree and reproduce the committed README exactly."""
    first = render()
    second = render()
    assert first == second
    assert all(line == line.rstrip() for line in first.splitlines())
    assert "[RENDERER:" not in first
    assert "<!--" not in first
    assert "`N = 1024, m = 8` every single reference caps" in first
    assert OUTPUT.read_text(encoding="utf-8") == first


def test_lemma6_failure_is_rejected(tmp_path: Path) -> None:
    """A failed scalar-tail lower bound cannot enter the report."""
    _copy_results(tmp_path)

    def fail(document: dict) -> None:
        document["lemma6"][0]["passes"] = False
        document["lemma6"][0]["margin"] = -1

    _mutate(tmp_path, FILES["lemma"], fail)
    with pytest.raises(RuntimeError, match="Lemma 6 lower-bound evidence"):
        render(tmp_path)


def test_certified_slower_than_abstained_is_rejected(tmp_path: Path) -> None:
    """The ordering claim fails closed when a certified method becomes slower."""
    _copy_results(tmp_path)

    def slow(document: dict) -> None:
        row = next(
            item
            for item in document["cases"]
            if item["nx"] == 128 and item["m"] == 4 and item["method"] == "cfl_reference_blend_B3"
        )
        row["gmres_regularized"]["true_stop_count"] = 25

    _mutate(tmp_path, FILES["active"], slow)
    with pytest.raises(RuntimeError, match="certified method is slower"):
        render(tmp_path)


def test_nonempty_m2_repository_mask_is_rejected(tmp_path: Path) -> None:
    """The m=2 motivation cannot render if the repository mask becomes non-empty."""
    _copy_results(tmp_path)

    def unmask(document: dict) -> None:
        row = next(
            item
            for item in document["cases"]
            if item["m"] == 2 and item["method"] == "repository_blend_B3"
        )
        row["inactive_count"] = 1

    _mutate(tmp_path, FILES["active"], unmask)
    with pytest.raises(RuntimeError, match="repository mask is non-empty at m = 2"):
        render(tmp_path)


def test_baseline_blend_near_zero_tail_is_rejected(tmp_path: Path) -> None:
    """Part II cannot claim a zero-free blend after a spectral regression."""
    _copy_results(tmp_path)

    def add_tail(document: dict) -> None:
        row = next(item for item in document["spectral_records"] if item["method"] == "tau_blend")
        row["spectrum"]["near_zero_eigenvalue_count"] = 1

    _mutate(tmp_path, FILES["baseline"], add_tail)
    with pytest.raises(RuntimeError, match="blend has a near-zero eigenvalue"):
        render(tmp_path)


def test_loose_tolerance_speed_claim_is_rejected(tmp_path: Path) -> None:
    """Part II fails closed if the blend becomes faster in a loose-tolerance case."""
    _copy_results(tmp_path)

    def make_blend_fast(document: dict) -> None:
        row = next(
            item
            for item in document["work_precision_records"]
            if item["nx"] == 512 and item["m"] == 2 and item["requested_relative_residual"] == 0.01
        )
        row["methods"]["tau_blend"]["median_seconds"] = 0.0

    _mutate(tmp_path, FILES["baseline"], make_blend_fast)
    with pytest.raises(RuntimeError, match="not slower than the fastest scalar"):
        render(tmp_path)


def test_key_values_match_the_facts_renderer() -> None:
    """Representative Part-I values are shared with the facts sheet evidence."""
    readme = render()
    facts, _ = render_facts()
    active = json.loads((OUTPUT.parent / FILES["active"]).read_text(encoding="utf-8"))
    lemma = json.loads((OUTPUT.parent / FILES["lemma"]).read_text(encoding="utf-8"))
    values = (
        active["decision_summary"]["Y_certified_count"],
        active["cases"][9]["Y_spectrum"]["sigma_min"],
        active["direct_regularized_full_MJ_certificates"][0]["certificate_ladder"][0]["rho"],
        min(row["margin"] for row in lemma["lemma6"]),
    )
    for value in values:
        shown = _fmt(value)
        assert shown in readme
        assert shown in facts
