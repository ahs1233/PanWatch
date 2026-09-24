"""Forward-only evidence ledger for GTG Event v3 path research.

This module has deliberately *no* order, position, sizing, stop, target-routing,
or execution APIs.  It exists only to preserve a tamper-evident-enough research
contract for the first truly fresh GTG v3 path holdout after the historical OOS
period became known.

Only the validation-selected DOWN path candidate is admitted:
    knn_embedding_median_k64

A prediction must be recorded close to the event timestamp and before the
4-hour path label can exist.  Outcomes are append-only and cannot be recorded
until the full horizon has elapsed.  A summary may mark a sample "review ready"
but can never authorize promotion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import json
import math
import sqlite3


UTC = timezone.utc

FORWARD_START_UTC = datetime(2026, 9, 24, 0, 0, tzinfo=UTC)
PATH_HORIZON = timedelta(hours=4)
MAX_PREDICTION_DELAY = timedelta(minutes=5)
CLOCK_SKEW = timedelta(seconds=60)

QUALIFIED_SIDE = "down"
QUALIFIED_EVENT = "bearish_14_50_to_200"
QUALIFIED_CANDIDATE = "knn_embedding_median_k64"
CANDIDATE_VERSION = "gtg-event-v3-down-path-forward-v1"
FEATURE_SCHEMA_VERSION = "gtg-event-v3-feature-v1"

MIN_REVIEW_EVENTS = 60
MIN_REVIEW_SPAN_DAYS = 30


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return _utc(value).isoformat()


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _finite_nonnegative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class ForwardPathPrediction:
    event_id: str
    event_time_utc: datetime
    recorded_at_utc: datetime
    expected_mfe_atr: float
    expected_mae_atr: float
    baseline_mfe_atr: float
    baseline_mae_atr: float
    source_commit: str
    side: str = QUALIFIED_SIDE
    event_name: str = QUALIFIED_EVENT
    candidate: str = QUALIFIED_CANDIDATE
    candidate_version: str = CANDIDATE_VERSION
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class ForwardPathOutcome:
    event_id: str
    recorded_at_utc: datetime
    horizon_complete_at_utc: datetime
    actual_mfe_atr: float
    actual_mae_atr: float
    metadata: dict[str, Any] | None = None


class GTGForwardShadowLedger:
    """SQLite append-only contract for fresh GTG path evidence."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        return con

    def _init(self) -> None:
        with self._connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    event_id TEXT PRIMARY KEY,
                    event_time_utc TEXT NOT NULL,
                    recorded_at_utc TEXT NOT NULL,
                    label_not_before_utc TEXT NOT NULL,
                    side TEXT NOT NULL,
                    event_name TEXT NOT NULL,
                    candidate TEXT NOT NULL,
                    candidate_version TEXT NOT NULL,
                    feature_schema_version TEXT NOT NULL,
                    expected_mfe_atr REAL NOT NULL,
                    expected_mae_atr REAL NOT NULL,
                    baseline_mfe_atr REAL NOT NULL,
                    baseline_mae_atr REAL NOT NULL,
                    source_commit TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    prediction_hash TEXT NOT NULL UNIQUE
                );

                CREATE TABLE IF NOT EXISTS outcomes (
                    event_id TEXT PRIMARY KEY,
                    recorded_at_utc TEXT NOT NULL,
                    horizon_complete_at_utc TEXT NOT NULL,
                    actual_mfe_atr REAL NOT NULL,
                    actual_mae_atr REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    FOREIGN KEY(event_id) REFERENCES predictions(event_id)
                );
                """
            )

    def record_prediction(self, prediction: ForwardPathPrediction) -> str:
        event_id = str(prediction.event_id).strip()
        if not event_id:
            raise ValueError("event_id is required")

        event_time = _utc(prediction.event_time_utc)
        recorded_at = _utc(prediction.recorded_at_utc)
        if event_time < FORWARD_START_UTC:
            raise ValueError("historical/backfilled events are forbidden")
        if recorded_at < event_time - CLOCK_SKEW:
            raise ValueError("prediction cannot materially predate the event")
        if recorded_at > event_time + MAX_PREDICTION_DELAY:
            raise ValueError("late/backfilled prediction is forbidden")

        if prediction.side != QUALIFIED_SIDE:
            raise ValueError("only the validation-qualified DOWN candidate is admitted")
        if prediction.event_name != QUALIFIED_EVENT:
            raise ValueError("event grammar mismatch")
        if prediction.candidate != QUALIFIED_CANDIDATE:
            raise ValueError("candidate mismatch")
        if prediction.candidate_version != CANDIDATE_VERSION:
            raise ValueError("candidate version mismatch")
        if prediction.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError("feature schema mismatch")

        source_commit = str(prediction.source_commit).strip()
        if not source_commit:
            raise ValueError("source_commit is required")

        expected_mfe = _finite_nonnegative(
            "expected_mfe_atr", prediction.expected_mfe_atr
        )
        expected_mae = _finite_nonnegative(
            "expected_mae_atr", prediction.expected_mae_atr
        )
        baseline_mfe = _finite_nonnegative(
            "baseline_mfe_atr", prediction.baseline_mfe_atr
        )
        baseline_mae = _finite_nonnegative(
            "baseline_mae_atr", prediction.baseline_mae_atr
        )
        label_not_before = event_time + PATH_HORIZON
        metadata_json = json.dumps(
            prediction.metadata or {},
            sort_keys=True,
            separators=(",", ":"),
        )

        canonical = {
            **asdict(prediction),
            "event_time_utc": _iso(event_time),
            "recorded_at_utc": _iso(recorded_at),
            "metadata": prediction.metadata or {},
            "label_not_before_utc": _iso(label_not_before),
        }
        digest = sha256(
            json.dumps(
                canonical,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

        try:
            with self._connect() as con:
                con.execute(
                    """
                    INSERT INTO predictions (
                        event_id, event_time_utc, recorded_at_utc,
                        label_not_before_utc, side, event_name, candidate,
                        candidate_version, feature_schema_version,
                        expected_mfe_atr, expected_mae_atr,
                        baseline_mfe_atr, baseline_mae_atr,
                        source_commit, metadata_json, prediction_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        _iso(event_time),
                        _iso(recorded_at),
                        _iso(label_not_before),
                        prediction.side,
                        prediction.event_name,
                        prediction.candidate,
                        prediction.candidate_version,
                        prediction.feature_schema_version,
                        expected_mfe,
                        expected_mae,
                        baseline_mfe,
                        baseline_mae,
                        source_commit,
                        metadata_json,
                        digest,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("prediction is immutable and event_id must be unique") from exc
        return digest

    def record_outcome(self, outcome: ForwardPathOutcome) -> None:
        event_id = str(outcome.event_id).strip()
        if not event_id:
            raise ValueError("event_id is required")

        recorded_at = _utc(outcome.recorded_at_utc)
        horizon_complete_at = _utc(outcome.horizon_complete_at_utc)
        actual_mfe = _finite_nonnegative("actual_mfe_atr", outcome.actual_mfe_atr)
        actual_mae = _finite_nonnegative("actual_mae_atr", outcome.actual_mae_atr)

        with self._connect() as con:
            prediction = con.execute(
                "SELECT * FROM predictions WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if prediction is None:
                raise ValueError("outcome has no pre-registered prediction")

            label_not_before = _parse(prediction["label_not_before_utc"])
            if horizon_complete_at < label_not_before:
                raise ValueError("outcome horizon is incomplete")
            if recorded_at < label_not_before:
                raise ValueError("outcome cannot be recorded before label availability")

            metadata_json = json.dumps(
                outcome.metadata or {},
                sort_keys=True,
                separators=(",", ":"),
            )
            try:
                con.execute(
                    """
                    INSERT INTO outcomes (
                        event_id, recorded_at_utc, horizon_complete_at_utc,
                        actual_mfe_atr, actual_mae_atr, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        _iso(recorded_at),
                        _iso(horizon_complete_at),
                        actual_mfe,
                        actual_mae,
                        metadata_json,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("outcome is append-only and cannot be replaced") from exc

    def pending_count(self) -> int:
        with self._connect() as con:
            row = con.execute(
                """
                SELECT COUNT(*) AS n
                FROM predictions p
                LEFT JOIN outcomes o ON o.event_id = p.event_id
                WHERE o.event_id IS NULL
                """
            ).fetchone()
        return int(row["n"])

    def evidence_summary(self) -> dict[str, Any]:
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                    p.event_id,
                    p.event_time_utc,
                    p.expected_mfe_atr,
                    p.expected_mae_atr,
                    p.baseline_mfe_atr,
                    p.baseline_mae_atr,
                    o.actual_mfe_atr,
                    o.actual_mae_atr
                FROM predictions p
                JOIN outcomes o ON o.event_id = p.event_id
                ORDER BY p.event_time_utc
                """
            ).fetchall()

        if not rows:
            return {
                "completed_events": 0,
                "pending_events": self.pending_count(),
                "review_ready": False,
                "promotion_authorized": False,
                "reason": "no completed fresh-forward events",
            }

        candidate_mfe_error = [
            abs(float(r["expected_mfe_atr"]) - float(r["actual_mfe_atr"]))
            for r in rows
        ]
        candidate_mae_error = [
            abs(float(r["expected_mae_atr"]) - float(r["actual_mae_atr"]))
            for r in rows
        ]
        baseline_mfe_error = [
            abs(float(r["baseline_mfe_atr"]) - float(r["actual_mfe_atr"]))
            for r in rows
        ]
        baseline_mae_error = [
            abs(float(r["baseline_mae_atr"]) - float(r["actual_mae_atr"]))
            for r in rows
        ]

        def mean(values: list[float]) -> float:
            return float(sum(values) / len(values))

        candidate_mfe = mean(candidate_mfe_error)
        candidate_mae = mean(candidate_mae_error)
        baseline_mfe = mean(baseline_mfe_error)
        baseline_mae = mean(baseline_mae_error)
        mfe_ratio = candidate_mfe / max(baseline_mfe, 1e-12)
        mae_ratio = candidate_mae / max(baseline_mae, 1e-12)

        first = _parse(rows[0]["event_time_utc"])
        last = _parse(rows[-1]["event_time_utc"])
        span_days = max(0.0, (last - first).total_seconds() / 86400.0)
        enough_sample = len(rows) >= MIN_REVIEW_EVENTS
        enough_span = span_days >= MIN_REVIEW_SPAN_DAYS
        beats_both = mfe_ratio < 1.0 and mae_ratio < 1.0

        return {
            "candidate": QUALIFIED_CANDIDATE,
            "candidate_version": CANDIDATE_VERSION,
            "forward_start_utc": _iso(FORWARD_START_UTC),
            "completed_events": len(rows),
            "pending_events": self.pending_count(),
            "span_days": span_days,
            "candidate_mfe_mae_atr": candidate_mfe,
            "baseline_mfe_mae_atr": baseline_mfe,
            "mfe_ratio_to_baseline": mfe_ratio,
            "candidate_mae_mae_atr": candidate_mae,
            "baseline_mae_mae_atr": baseline_mae,
            "mae_ratio_to_baseline": mae_ratio,
            "beats_both_path_baselines": beats_both,
            "minimum_review_events": MIN_REVIEW_EVENTS,
            "minimum_review_span_days": MIN_REVIEW_SPAN_DAYS,
            "review_ready": bool(enough_sample and enough_span and beats_both),
            "promotion_authorized": False,
            "reason": (
                "fresh-forward evidence is review-ready; promotion still requires "
                "an explicit separate governance decision"
                if enough_sample and enough_span and beats_both
                else "fresh-forward evidence gate is not yet satisfied"
            ),
        }
