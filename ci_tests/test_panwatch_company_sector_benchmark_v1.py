from benchmarks.panwatch_v1.company_sector_runner import (
    CASES,
    run_company_sector_benchmark,
)


def test_company_sector_track_has_frozen_case_coverage():
    ids = [case_id for case_id, _category, _points, _evaluator in CASES]
    assert len(ids) == 10
    assert len(ids) == len(set(ids))


def test_company_sector_track_regression_gate():
    report = run_company_sector_benchmark()
    assert report["version"] == "1.7.0"
    assert report["track"] == "company_sector_research"
    assert report["total_cases"] == 10
    assert report["passed_cases"] == 10
    assert report["regression_gate_percent"] == 90.0
    assert report["regression_gate_passed"] is True, report
