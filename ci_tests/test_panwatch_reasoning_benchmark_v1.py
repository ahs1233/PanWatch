from __future__ import annotations

from benchmarks.panwatch_v1.reasoning_runner import CASES, run_reasoning_benchmark


def test_reasoning_track_has_frozen_case_coverage():
    ids = [case_id for case_id, _category, _points, _evaluator in CASES]
    assert len(ids) == 8
    assert len(ids) == len(set(ids))


def test_reasoning_track_regression_gate():
    report = run_reasoning_benchmark()
    assert report["track"] == "claim_graph_falsification"
    assert report["total_cases"] == 8
    assert report["regression_gate_percent"] == 90.0
    assert report["regression_gate_passed"] is True, report


def test_reasoning_track_is_explicit_about_remaining_limitations():
    report = run_reasoning_benchmark()
    limitations = " ".join(report["limitations"]).lower()
    assert "automatic research loop" in limitations
    assert "persistent belief state" in limitations
