from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.modules.research import (
    EvidenceLedger,
    EvidenceRelation,
    ObservationKind,
    SourceTier,
    build_evidence,
    build_source,
    canonicalize_url,
)
from src.modules.research.evidence_store import (
    load_claim_ledger,
    persist_ledger,
)
from src.platform.persistence.migrations import _m127_research_evidence_foundation
from src.platform.persistence.models import (
    ResearchEvidenceRecord,
    ResearchSourceRecord,
)


UTC = timezone.utc
NOW = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def _source(
    slug: str,
    *,
    tier: SourceTier = SourceTier.PRIMARY,
    upstream_origin: str = "",
    content: str | None = None,
):
    return build_source(
        url=f"https://{slug}.example.com/report?utm_source=test&lang=en",
        content=content or f"{slug} content",
        publisher=slug,
        source_tier=tier,
        source_family=slug,
        upstream_origin=upstream_origin,
        published_at=NOW,
        retrieved_at=NOW + timedelta(minutes=1),
    )


def test_canonicalize_url_removes_tracking_but_preserves_semantic_query():
    url = "HTTPS://Example.COM/report/?utm_source=x&lang=en&gclid=1#top"
    assert canonicalize_url(url) == "https://example.com/report?lang=en"


def test_ledger_is_append_only_and_idempotent_for_same_evidence_identity():
    ledger = EvidenceLedger()
    source = _source("official")
    ledger.register_source(source)
    record = build_evidence(
        claim_key="macro.cpi",
        source=source,
        statement="CPI was 3.1 percent",
        numeric_value=3.1,
        unit="%",
    )
    assert ledger.append(record) is True
    assert ledger.append(record) is False
    assert len(ledger.records) == 1


def test_evidence_requires_registered_source():
    ledger = EvidenceLedger()
    source = _source("orphan")
    record = build_evidence(
        claim_key="macro.cpi",
        source=source,
        statement="CPI was 3.1 percent",
    )
    with pytest.raises(ValueError, match="unknown source_id"):
        ledger.append(record)


def test_forecast_never_satisfies_actual_guard():
    ledger = EvidenceLedger()
    source = _source("forecast")
    ledger.register_source(source)
    ledger.append(
        build_evidence(
            claim_key="macro.cpi",
            source=source,
            statement="Forecast 3.2 percent",
            observation_kind=ObservationKind.FORECAST,
            numeric_value=3.2,
            unit="%",
        )
    )
    result = ledger.actual_forecast_guard("macro.cpi")
    assert result.passed is False
    assert result.code == "forecast_only"


def test_only_linked_revision_can_supersede_actual():
    ledger = EvidenceLedger()
    source = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    actual = build_evidence(
        claim_key="macro.payrolls",
        source=source,
        statement="Payrolls 100k",
        observation_kind=ObservationKind.ACTUAL,
        numeric_value=100,
        unit="k",
        period="2026-08",
    )
    ledger.append(actual)

    unlinked = build_evidence(
        claim_key="macro.payrolls",
        source=source,
        statement="Unlinked revision says 70k",
        observation_kind=ObservationKind.REVISION,
        numeric_value=70,
        unit="k",
        period="2026-08",
    )
    ledger.append(unlinked)
    assert ledger.resolve_numeric("macro.payrolls").value == 100

    linked = build_evidence(
        claim_key="macro.payrolls",
        source=source,
        statement="Linked revision says 82k",
        observation_kind=ObservationKind.REVISION,
        numeric_value=82,
        unit="k",
        period="2026-08",
        revision_of=actual.evidence_id,
        supersedes=actual.evidence_id,
    )
    ledger.append(linked)
    resolved = ledger.resolve_numeric("macro.payrolls")
    assert resolved.status == "resolved_revision"
    assert resolved.value == 82


def test_syndicated_copies_count_as_one_independent_source():
    ledger = EvidenceLedger()
    for idx in range(3):
        source = _source(
            f"copy-{idx}",
            upstream_origin="wire:story-123",
            content="Same wire story",
        )
        ledger.register_source(source)
        ledger.append(
            build_evidence(
                claim_key="rates.hold",
                source=source,
                statement="Rates were held unchanged",
            )
        )
    assert ledger.independent_source_count("rates.hold") == 1
    assert ledger.source_independence_ratio("rates.hold") == pytest.approx(1 / 3)


def test_numeric_conflict_is_exposed_not_silently_averaged():
    ledger = EvidenceLedger()
    official = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
    secondary = _source("secondary", tier=SourceTier.SECONDARY)
    for source, value in ((official, 2.1), (secondary, 1.7)):
        ledger.register_source(source)
        ledger.append(
            build_evidence(
                claim_key="macro.gdp",
                source=source,
                statement=f"GDP {value}",
                numeric_value=value,
                unit="%",
                period="2026-Q2",
            )
        )
    resolved = ledger.resolve_numeric("macro.gdp")
    assert resolved.status == "conflict"
    assert resolved.value == 2.1
    assert len(resolved.conflicts) == 1


def test_freshness_guard_uses_event_time_not_retrieval_time():
    ledger = EvidenceLedger()
    source = build_source(
        url="https://example.com/old",
        content="old report",
        source_tier=SourceTier.PRIMARY,
        published_at=NOW - timedelta(days=5),
        retrieved_at=NOW,
    )
    ledger.register_source(source)
    ledger.append(
        build_evidence(
            claim_key="macro.now",
            source=source,
            statement="Old data",
            event_time=NOW - timedelta(days=5),
        )
    )
    result = ledger.freshness_guard(
        "macro.now",
        max_age=timedelta(hours=24),
        as_of=NOW,
    )
    assert result.passed is False
    assert result.code == "stale"


def test_persistence_roundtrip_preserves_semantics_and_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    ResearchSourceRecord.__table__.create(engine)
    ResearchEvidenceRecord.__table__.create(engine)
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        ledger = EvidenceLedger()
        source = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
        ledger.register_source(source)
        record = build_evidence(
            claim_key="macro.cpi",
            source=source,
            statement="CPI was 3.4 percent",
            observation_kind=ObservationKind.ACTUAL,
            relation=EvidenceRelation.SUPPORTS,
            event_time=NOW,
            numeric_value=3.4,
            unit="%",
            period="2026-08",
        )
        ledger.append(record)

        first = persist_ledger(db, ledger)
        second = persist_ledger(db, ledger)
        assert first == {"sources_inserted": 1, "evidence_inserted": 1}
        assert second == {"sources_inserted": 0, "evidence_inserted": 0}

        loaded = load_claim_ledger(db, "macro.cpi")
        assert len(loaded.sources) == 1
        assert len(loaded.records) == 1
        assert loaded.records[0].numeric_value == 3.4
        assert loaded.records[0].observation_kind is ObservationKind.ACTUAL
    finally:
        db.close()


def test_migration_127_creates_evidence_tables_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
    inspector = inspect(engine)
    assert "research_sources" in inspector.get_table_names()
    assert "research_evidence" in inspector.get_table_names()
    source_indexes = {row["name"] for row in inspector.get_indexes("research_sources")}
    evidence_indexes = {row["name"] for row in inspector.get_indexes("research_evidence")}
    assert "ix_research_source_independence" in source_indexes
    assert "ix_research_evidence_claim_kind_time" in evidence_indexes
