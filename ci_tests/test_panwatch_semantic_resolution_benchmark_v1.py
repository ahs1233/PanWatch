from benchmarks.panwatch_v1.semantic_resolution_runner import (
    CASES,
    run_semantic_resolution_benchmark,
)


def test_semantic_resolution_track_has_frozen_coverage():
    ids = [case_id for case_id, _points, _evaluator in CASES]
    assert len(ids) == 5
    assert len(ids) == len(set(ids))


def test_semantic_resolution_track_regression_gate():
    report = run_semantic_resolution_benchmark()
    assert report["version"] == "1.7.0"
    assert report["track"] == "semantic_claim_resolution"
    assert report["total_cases"] == 5
    assert report["regression_gate_percent"] == 90.0
    assert report["regression_gate_passed"] is True, report
