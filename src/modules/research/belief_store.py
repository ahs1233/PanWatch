"""Persistence for PanWatch belief snapshots, change events and cycles."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from src.platform.persistence.models import (
    ResearchBeliefCycleRecord,
    ResearchBeliefEventRecord,
    ResearchBeliefSnapshotRecord,
)

from .belief_state import (
    BeliefEvent,
    BeliefEventType,
    BeliefSnapshot,
)
from .claim_graph import ClaimStatus
from .evidence import utc


def _snapshot_from_row(row: ResearchBeliefSnapshotRecord) -> BeliefSnapshot:
    return BeliefSnapshot(
        snapshot_id=row.snapshot_id,
        claim_id=row.claim_id,
        claim_key=row.claim_key,
        evaluated_at=utc(row.evaluated_at),
        base_status=ClaimStatus(row.base_status),
        final_status=ClaimStatus(row.final_status),
        base_confidence=float(row.base_confidence),
        final_confidence=float(row.final_confidence),
        support_score=float(row.support_score),
        contradiction_score=float(row.contradiction_score),
        falsification_coverage=float(row.falsification_coverage),
        evidence_ids=tuple(row.evidence_ids or []),
        triggered_rules=tuple(row.triggered_rules or []),
        untestable_rules=tuple(row.untestable_rules or []),
        dependency_failures=tuple(row.dependency_failures or []),
        reasons=tuple(row.reasons or []),
        input_fingerprint=row.input_fingerprint,
        metadata=dict(row.meta or {}),
    )


def load_latest_belief_snapshot(
    db: Session,
    *,
    claim_id: str,
    before_or_at: datetime | None = None,
) -> BeliefSnapshot | None:
    q = db.query(ResearchBeliefSnapshotRecord).filter(
        ResearchBeliefSnapshotRecord.claim_id == claim_id
    )
    if before_or_at is not None:
        q = q.filter(
            ResearchBeliefSnapshotRecord.evaluated_at <= utc(before_or_at)
        )
    row = (
        q.order_by(
            ResearchBeliefSnapshotRecord.evaluated_at.desc(),
            ResearchBeliefSnapshotRecord.snapshot_id.desc(),
        )
        .first()
    )
    return _snapshot_from_row(row) if row is not None else None


def load_belief_history(
    db: Session,
    *,
    claim_id: str,
    limit: int = 100,
) -> list[BeliefSnapshot]:
    rows = (
        db.query(ResearchBeliefSnapshotRecord)
        .filter(ResearchBeliefSnapshotRecord.claim_id == claim_id)
        .order_by(
            ResearchBeliefSnapshotRecord.evaluated_at.asc(),
            ResearchBeliefSnapshotRecord.snapshot_id.asc(),
        )
        .limit(max(1, int(limit)))
        .all()
    )
    return [_snapshot_from_row(row) for row in rows]


def load_belief_events(
    db: Session,
    *,
    claim_id: str | None = None,
    event_type: BeliefEventType | None = None,
    limit: int = 200,
) -> list[BeliefEvent]:
    q = db.query(ResearchBeliefEventRecord)
    if claim_id:
        q = q.filter(ResearchBeliefEventRecord.claim_id == claim_id)
    if event_type is not None:
        q = q.filter(
            ResearchBeliefEventRecord.event_type
            == BeliefEventType(event_type).value
        )
    rows = (
        q.order_by(
            ResearchBeliefEventRecord.occurred_at.asc(),
            ResearchBeliefEventRecord.event_id.asc(),
        )
        .limit(max(1, int(limit)))
        .all()
    )
    return [
        BeliefEvent(
            event_id=row.event_id,
            claim_id=row.claim_id,
            event_type=BeliefEventType(row.event_type),
            occurred_at=utc(row.occurred_at),
            previous_snapshot_id=row.previous_snapshot_id,
            current_snapshot_id=row.current_snapshot_id,
            previous_status=(
                ClaimStatus(row.previous_status)
                if row.previous_status
                else None
            ),
            current_status=ClaimStatus(row.current_status),
            confidence_delta=float(row.confidence_delta),
            detail=row.detail or "",
            metadata=dict(row.meta or {}),
        )
        for row in rows
    ]


def persist_belief_cycle(
    db: Session,
    *,
    cycle,
    metadata: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Persist one PanWatch cycle atomically and idempotently."""
    cycle_existing = db.get(ResearchBeliefCycleRecord, cycle.cycle_id)
    if cycle_existing is None:
        db.add(
            ResearchBeliefCycleRecord(
                cycle_id=cycle.cycle_id,
                started_at=cycle.started_at,
                completed_at=cycle.completed_at,
                claim_count=cycle.claim_count,
                changed_count=cycle.changed_count,
                falsified_count=cycle.falsified_count,
                probe_count=cycle.probe_count,
                meta=dict(metadata or {}),
            )
        )

    snapshots_inserted = 0
    events_inserted = 0
    for update in cycle.updates:
        snapshot = update.snapshot
        existing = db.get(
            ResearchBeliefSnapshotRecord,
            snapshot.snapshot_id,
        )
        if existing is None:
            db.add(
                ResearchBeliefSnapshotRecord(
                    snapshot_id=snapshot.snapshot_id,
                    cycle_id=cycle.cycle_id,
                    claim_id=snapshot.claim_id,
                    claim_key=snapshot.claim_key,
                    evaluated_at=snapshot.evaluated_at,
                    base_status=snapshot.base_status.value,
                    final_status=snapshot.final_status.value,
                    base_confidence=snapshot.base_confidence,
                    final_confidence=snapshot.final_confidence,
                    support_score=snapshot.support_score,
                    contradiction_score=snapshot.contradiction_score,
                    falsification_coverage=snapshot.falsification_coverage,
                    evidence_ids=list(snapshot.evidence_ids),
                    triggered_rules=list(snapshot.triggered_rules),
                    untestable_rules=list(snapshot.untestable_rules),
                    dependency_failures=list(
                        snapshot.dependency_failures
                    ),
                    reasons=list(snapshot.reasons),
                    input_fingerprint=snapshot.input_fingerprint,
                    meta=dict(snapshot.metadata or {}),
                )
            )
            snapshots_inserted += 1

    db.flush()

    for event in cycle.events:
        existing = db.get(ResearchBeliefEventRecord, event.event_id)
        if existing is not None:
            continue
        db.add(
            ResearchBeliefEventRecord(
                event_id=event.event_id,
                cycle_id=cycle.cycle_id,
                claim_id=event.claim_id,
                event_type=event.event_type.value,
                occurred_at=event.occurred_at,
                previous_snapshot_id=event.previous_snapshot_id,
                current_snapshot_id=event.current_snapshot_id,
                previous_status=(
                    event.previous_status.value
                    if event.previous_status is not None
                    else None
                ),
                current_status=event.current_status.value,
                confidence_delta=event.confidence_delta,
                detail=event.detail,
                meta=dict(event.metadata or {}),
            )
        )
        events_inserted += 1

    db.commit()
    return {
        "cycle_inserted": 1 if cycle_existing is None else 0,
        "snapshots_inserted": snapshots_inserted,
        "events_inserted": events_inserted,
    }


def list_recent_belief_cycles(
    db: Session,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    rows = (
        db.query(ResearchBeliefCycleRecord)
        .order_by(ResearchBeliefCycleRecord.started_at.desc())
        .limit(max(1, int(limit)))
        .all()
    )
    return [
        {
            "cycle_id": row.cycle_id,
            "started_at": utc(row.started_at).isoformat(),
            "completed_at": utc(row.completed_at).isoformat(),
            "claim_count": int(row.claim_count),
            "changed_count": int(row.changed_count),
            "falsified_count": int(row.falsified_count),
            "probe_count": int(row.probe_count),
            "meta": dict(row.meta or {}),
        }
        for row in rows
    ]
