from __future__ import annotations

from benchmarks.panwatch_v1.runner import CASES, run_benchmark


def test_panwatch_benchmark_v1_has_meaningful_case_coverage():
    ids = [case_id for case_id, _category, _points, _evaluator in CASES]
    assert len(ids) >= 15
    assert len(ids) == len(set(ids))


def test_panwatch_benchmark_v1_regression_gate():
    report = run_benchmark()
    assert report["total_cases"] >= 15
    assert report["regression_gate_percent"] == 95.0
    assert report["regression_gate_passed"] is True, report


def test_panwatch_benchmark_v1_is_explicit_about_unautomated_tracks():
    report = run_benchmark()
    coverage = report["track_coverage"]
    assert coverage == {
        "automated": 5,
        "specified_not_automated": 1,
        "total": 6,
    }
    assert any("not automated" in item for item in report["limitations"])
