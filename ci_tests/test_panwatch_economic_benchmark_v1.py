from __future__ import annotations

from benchmarks.panwatch_v1.economic_runner import (
    CASES,
    run_economic_benchmark,
)


def test_economic_track_has_all_ten_frozen_cases():
    ids = [case_id for case_id, _category, _points, _evaluator in CASES]
    assert len(ids) == 10
    assert len(ids) == len(set(ids))


def test_economic_evidence_integrity_regression_gate():
    report = run_economic_benchmark()
    assert report["track"] == "economic_evidence_integrity"
    assert report["total_cases"] == 10
    assert report["regression_gate_percent"] == 90.0
    assert report["regression_gate_passed"] is True, report


def test_economic_track_does_not_claim_external_research_superiority():
    report = run_economic_benchmark()
    limitations = " ".join(report["limitations"]).lower()
    assert "forecasting skill" in limitations
    assert "external llm" in limitations
