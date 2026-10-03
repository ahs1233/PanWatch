from __future__ import annotations

from benchmarks.panwatch_v1.belief_runner import CASES, run_belief_benchmark


def test_persistent_belief_track_has_frozen_case_coverage():
    ids = [case_id for case_id, _category, _points, _evaluator in CASES]
    assert len(ids) == 8
    assert len(ids) == len(set(ids))


def test_persistent_belief_track_regression_gate():
    report = run_belief_benchmark()
    assert report["track"] == "persistent_belief_state"
    assert report["total_cases"] == 8
    assert report["regression_gate_percent"] == 90.0
    assert report["regression_gate_passed"] is True, report


def test_persistent_belief_track_states_remaining_boundaries():
    report = run_belief_benchmark()
    text = " ".join(report["limitations"]).lower()
    assert "claim extraction" in text
    assert "automatic research loop" in text
