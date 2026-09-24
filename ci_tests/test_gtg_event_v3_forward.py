from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.modules.xau.gtg_event_v3_forward import (
    CANDIDATE_VERSION,
    FEATURE_SCHEMA_VERSION,
    FORWARD_START_UTC,
    GTGForwardShadowLedger,
    QUALIFIED_CANDIDATE,
    QUALIFIED_EVENT,
    ForwardPathOutcome,
    ForwardPathPrediction,
)


UTC = timezone.utc


def prediction(
    event_id: str,
    event_time: datetime,
    *,
    recorded_at: datetime | None = None,
    expected_mfe: float = 1.0,
    expected_mae: float = 2.0,
    baseline_mfe: float = 1.5,
    baseline_mae: float = 2.5,
) -> ForwardPathPrediction:
    return ForwardPathPrediction(
        event_id=event_id,
        event_time_utc=event_time,
        recorded_at_utc=recorded_at or event_time + timedelta(seconds=10),
        expected_mfe_atr=expected_mfe,
        expected_mae_atr=expected_mae,
        baseline_mfe_atr=baseline_mfe,
        baseline_mae_atr=baseline_mae,
        source_commit="cafb37b3b5cffc8e8530067b6c70bf8610c7e7ab",
        event_name=QUALIFIED_EVENT,
        candidate=QUALIFIED_CANDIDATE,
        candidate_version=CANDIDATE_VERSION,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
    )


def outcome(
    event_id: str,
    event_time: datetime,
    *,
    actual_mfe: float = 1.1,
    actual_mae: float = 2.1,
) -> ForwardPathOutcome:
    complete = event_time + timedelta(hours=4)
    return ForwardPathOutcome(
        event_id=event_id,
        recorded_at_utc=complete + timedelta(seconds=5),
        horizon_complete_at_utc=complete,
        actual_mfe_atr=actual_mfe,
        actual_mae_atr=actual_mae,
    )


def test_rejects_historical_and_late_backfill(tmp_path):
    ledger = GTGForwardShadowLedger(tmp_path / "ledger.sqlite3")

    with pytest.raises(ValueError, match="historical/backfilled"):
        ledger.record_prediction(
            prediction("old", FORWARD_START_UTC - timedelta(minutes=5))
        )

    event_time = FORWARD_START_UTC + timedelta(hours=1)
    with pytest.raises(ValueError, match="late/backfilled"):
        ledger.record_prediction(
            prediction(
                "late",
                event_time,
                recorded_at=event_time + timedelta(minutes=6),
            )
        )


def test_prediction_and_outcome_are_append_only(tmp_path):
    ledger = GTGForwardShadowLedger(tmp_path / "ledger.sqlite3")
    event_time = FORWARD_START_UTC + timedelta(hours=1)
    p = prediction("evt-1", event_time)

    digest = ledger.record_prediction(p)
    assert len(digest) == 64
    assert ledger.pending_count() == 1

    with pytest.raises(ValueError, match="immutable"):
        ledger.record_prediction(p)

    too_early = ForwardPathOutcome(
        event_id="evt-1",
        recorded_at_utc=event_time + timedelta(hours=3),
        horizon_complete_at_utc=event_time + timedelta(hours=3),
        actual_mfe_atr=1.2,
        actual_mae_atr=2.2,
    )
    with pytest.raises(ValueError, match="horizon is incomplete"):
        ledger.record_outcome(too_early)

    ledger.record_outcome(outcome("evt-1", event_time))
    assert ledger.pending_count() == 0

    with pytest.raises(ValueError, match="append-only"):
        ledger.record_outcome(outcome("evt-1", event_time))


def test_outcome_requires_pre_registered_prediction(tmp_path):
    ledger = GTGForwardShadowLedger(tmp_path / "ledger.sqlite3")
    event_time = FORWARD_START_UTC + timedelta(hours=1)
    with pytest.raises(ValueError, match="pre-registered"):
        ledger.record_outcome(outcome("missing", event_time))


def test_only_qualified_down_candidate_is_admitted(tmp_path):
    ledger = GTGForwardShadowLedger(tmp_path / "ledger.sqlite3")
    event_time = FORWARD_START_UTC + timedelta(hours=1)

    bad = prediction("evt-up", event_time)
    bad = ForwardPathPrediction(**{**bad.__dict__, "side": "up"})
    with pytest.raises(ValueError, match="only the validation-qualified DOWN"):
        ledger.record_prediction(bad)


def test_review_gate_requires_fresh_sample_span_and_both_path_wins(tmp_path):
    ledger = GTGForwardShadowLedger(tmp_path / "ledger.sqlite3")

    # 60 immutable forward predictions spread across more than 30 days.
    # Candidate errors are intentionally lower than the locked baseline errors.
    for i in range(60):
        event_time = FORWARD_START_UTC + timedelta(days=i * 31 / 59)
        event_id = f"evt-{i:03d}"
        ledger.record_prediction(
            prediction(
                event_id,
                event_time,
                expected_mfe=1.0,
                expected_mae=2.0,
                baseline_mfe=1.6,
                baseline_mae=2.8,
            )
        )
        ledger.record_outcome(
            outcome(
                event_id,
                event_time,
                actual_mfe=1.1,
                actual_mae=2.1,
            )
        )

    report = ledger.evidence_summary()
    assert report["completed_events"] == 60
    assert report["span_days"] >= 30
    assert report["mfe_ratio_to_baseline"] < 1.0
    assert report["mae_ratio_to_baseline"] < 1.0
    assert report["beats_both_path_baselines"] is True
    assert report["review_ready"] is True
    assert report["promotion_authorized"] is False


def test_empty_ledger_is_never_review_ready(tmp_path):
    report = GTGForwardShadowLedger(tmp_path / "ledger.sqlite3").evidence_summary()
    assert report["completed_events"] == 0
    assert report["review_ready"] is False
    assert report["promotion_authorized"] is False
