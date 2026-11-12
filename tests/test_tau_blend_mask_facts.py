"""Determinism and source-path checks for the tau-blend mask facts sheet."""

from __future__ import annotations

import json

from benchmarks.make_tau_blend_mask_facts import OUTPUT, render_facts


def test_tau_blend_mask_facts_are_deterministic_and_current() -> None:
    """Two renders agree and reproduce the committed facts sheet byte-for-byte."""
    first, first_references = render_facts()
    second, second_references = render_facts()
    assert first == second
    assert first_references == second_references
    assert first_references
    assert all(line == line.rstrip() for line in first.splitlines())
    assert OUTPUT.read_text(encoding="utf-8") == first


def test_every_printed_json_path_resolves() -> None:
    """Rendering validates every emitted file/path pair before returning."""
    text, references = render_facts()
    assert len(references) > 100
    for reference in references:
        assert f"#{reference.pointer}`" in text


def test_committed_union_and_gmres_fields_are_self_consistent() -> None:
    """The artifact carries the corrected 15-case union and both operators."""
    result = json.loads(OUTPUT.with_name("tau_blend_active_block.json").read_text(encoding="utf-8"))
    rows = result["cases"]
    summary = result["decision_summary"]
    assert summary["Y_certified_count"] == 15
    assert summary["union_certified_count"] == 15
    assert summary["disk_positive_categories"] == ["adequate"]
    assert summary["union_certified_count"] == sum(row["union"] == "certified" for row in rows)
    assert all(row["gmres"]["operator"] == "regularized_J (headline)" for row in rows)
    assert all("gmres_J0" in row and "gmres_regularized" in row for row in rows)
