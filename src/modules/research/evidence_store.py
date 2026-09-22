"""SQLite/SQLAlchemy persistence bridge for the research evidence ledger."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.platform.persistence.models import (
    ResearchEvidenceRecord,
    ResearchSourceRecord,
)

from .evidence import (
    EvidenceRecord,
    EvidenceRelation,
    ObservationKind,
    SourceProvenance,
    SourceTier,
    utc,
)
from .ledger import EvidenceLedger


def persist_source(db: Session, source: SourceProvenance) -> bool:
    existing = db.get(ResearchSourceRecord, source.source_id)
    if existing is not None:
        if (
            existing.canonical_url != source.canonical_url
            or existing.content_hash != source.content_hash
        ):
            raise ValueError(
                f"source identity collision for {source.source_id}"
            )
        return False

    db.add(
        ResearchSourceRecord(
            source_id=source.source_id,
            url=source.url,
            canonical_url=source.canonical_url,
            domain=source.domain,
            publisher=source.publisher,
            title=source.title,
            source_tier=source.source_tier.value,
            source_family=source.source_family,
            independence_key=source.independence_key,
            published_at=source.published_at,
            retrieved_at=source.retrieved_at,
            observed_at=source.observed_at,
            content_hash=source.content_hash,
            parent_source_id=source.parent_source_id,
            tool_name=source.tool_name,
            meta=source.metadata,
        )
    )
    return True


def persist_evidence(db: Session, record: EvidenceRecord) -> bool:
    existing = db.get(ResearchEvidenceRecord, record.evidence_id)
    if existing is not None:
        if (
            existing.source_id != record.source_id
            or existing.claim_key != record.claim_key
            or existing.content_hash != record.content_hash
        ):
            raise ValueError(
                f"evidence identity collision for {record.evidence_id}"
            )
        return False

    if db.get(ResearchSourceRecord, record.source_id) is None:
        raise ValueError(
            f"source {record.source_id} must be persisted before evidence"
        )

    db.add(
        ResearchEvidenceRecord(
            evidence_id=record.evidence_id,
            claim_key=record.claim_key,
            source_id=record.source_id,
            statement=record.statement,
            relation=record.relation.value,
            observation_kind=record.observation_kind.value,
            event_time=record.event_time,
            observed_at=record.observed_at,
            recorded_at=record.recorded_at,
            confidence=record.confidence,
            content_hash=record.content_hash,
            numeric_value=record.numeric_value,
            unit=record.unit,
            period=record.period,
            revision_of=record.revision_of,
            supersedes=record.supersedes,
            meta=record.metadata,
        )
    )
    return True


def persist_ledger(db: Session, ledger: EvidenceLedger) -> dict[str, int]:
    source_count = 0
    evidence_count = 0
    for source in ledger.sources:
        if persist_source(db, source):
            source_count += 1
    db.flush()
    for record in ledger.records:
        if persist_evidence(db, record):
            evidence_count += 1
    db.commit()
    return {
        "sources_inserted": source_count,
        "evidence_inserted": evidence_count,
    }


def load_claim_ledger(db: Session, claim_key: str) -> EvidenceLedger:
    rows = (
        db.query(ResearchEvidenceRecord)
        .filter(ResearchEvidenceRecord.claim_key == claim_key)
        .order_by(
            ResearchEvidenceRecord.recorded_at.asc(),
            ResearchEvidenceRecord.evidence_id.asc(),
        )
        .all()
    )
    ledger = EvidenceLedger()
    if not rows:
        return ledger

    source_ids = {row.source_id for row in rows}
    sources = (
        db.query(ResearchSourceRecord)
        .filter(ResearchSourceRecord.source_id.in_(source_ids))
        .all()
    )
    for row in sources:
        ledger.register_source(
            SourceProvenance(
                source_id=row.source_id,
                url=row.url or "",
                canonical_url=row.canonical_url or "",
                domain=row.domain or "",
                publisher=row.publisher or "",
                title=row.title or "",
                source_tier=SourceTier(row.source_tier or "unknown"),
                source_family=row.source_family or "",
                independence_key=row.independence_key or "",
                published_at=utc(row.published_at)
                if row.published_at is not None
                else None,
                retrieved_at=utc(row.retrieved_at),
                observed_at=utc(row.observed_at),
                content_hash=row.content_hash or "",
                parent_source_id=row.parent_source_id,
                tool_name=row.tool_name or "",
                metadata=dict(row.meta or {}),
            )
        )

    for row in rows:
        ledger.append(
            EvidenceRecord(
                evidence_id=row.evidence_id,
                claim_key=row.claim_key,
                source_id=row.source_id,
                statement=row.statement,
                relation=EvidenceRelation(row.relation),
                observation_kind=ObservationKind(row.observation_kind),
                event_time=utc(row.event_time)
                if row.event_time is not None
                else None,
                observed_at=utc(row.observed_at),
                recorded_at=utc(row.recorded_at),
                confidence=float(row.confidence),
                content_hash=row.content_hash,
                numeric_value=(
                    float(row.numeric_value)
                    if row.numeric_value is not None
                    else None
                ),
                unit=row.unit or "",
                period=row.period or "",
                revision_of=row.revision_of,
                supersedes=row.supersedes,
                metadata=dict(row.meta or {}),
            )
        )
    return ledger
