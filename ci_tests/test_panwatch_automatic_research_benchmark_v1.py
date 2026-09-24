from __future__ import annotations

from benchmarks.panwatch_v1.automatic_research_runner import (
    CASES,
    run_automatic_research_benchmark,
)


def test_automatic_research_track_has_frozen_case_coverage():
    ids = [case_id for case_id, _category, _points, _evaluator in CASES]
    assert len(ids) == 6
    assert len(ids) == len(set(ids))


def test_automatic_research_track_regression_gate():
    report = run_automatic_research_benchmark()
    assert report["version"] == "1.7.0"
    assert report["track"] == "automatic_research_loop"
    assert report["total_cases"] == 6
    assert report["regression_gate_percent"] == 90.0
    assert report["regression_gate_passed"] is True, report
