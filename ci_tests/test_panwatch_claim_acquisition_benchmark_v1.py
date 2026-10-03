from benchmarks.panwatch_v1.claim_acquisition_runner import (
    CASES,
    run_claim_acquisition_benchmark,
)


def test_claim_acquisition_track_has_frozen_coverage():
    ids=[case_id for case_id,_category,_points,_evaluator in CASES]
    assert len(ids)==5
    assert len(ids)==len(set(ids))


def test_claim_acquisition_track_regression_gate():
    report=run_claim_acquisition_benchmark()
    assert report["version"]=="1.7.0"
    assert report["track"]=="general_claim_acquisition"
    assert report["regression_gate_percent"]==90.0
    assert report["regression_gate_passed"] is True, report
