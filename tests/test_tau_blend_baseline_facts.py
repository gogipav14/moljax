"""Determinism and source-path checks for the tau-blend baseline facts."""

from benchmarks.make_tau_blend_baseline_facts import OUTPUT, render_facts


def test_tau_blend_baseline_facts_are_deterministic_and_current() -> None:
    first, first_references = render_facts()
    second, second_references = render_facts()
    assert first == second
    assert first_references == second_references
    assert first_references
    assert all(line == line.rstrip() for line in first.splitlines())
    assert OUTPUT.read_text(encoding="utf-8") == first


def test_every_baseline_fact_pointer_resolves() -> None:
    text, references = render_facts()
    assert len(references) > 100
    for reference in references:
        assert f"#{reference.pointer}`" in text
