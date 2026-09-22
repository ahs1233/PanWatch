from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.modules.research import (
    ClaimGraph,
    ClaimKind,
    EvidenceLedger,
    FalsificationEngine,
    SourceTier,
    build_claim,
    build_evidence,
    build_source,
)
from src.modules.research.belief_scheduler import PersistentBeliefScheduler
from src.modules.research.evidence_store import persist_ledger
from src.modules.research.reasoning_store import (
    persist_claim_graph,
    persist_falsification_engine,
)
from src.platform.persistence.migrations import (
    _m127_research_evidence_foundation,
    _m128_claim_graph_and_falsification,
    _m129_persistent_belief_state,
)
from src.platform.persistence.models import ResearchBeliefCycleRecord


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _session_factory():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
    return sessionmaker(bind=engine)


def _seed(Session):
    db = Session()
    try:
        source = build_source(
            url="https://official.example.com/report",
            content="Initial evidence",
            publisher="official",
            source_tier=SourceTier.OFFICIAL_PRIMARY,
            source_family="official",
            published_at=T0,
            retrieved_at=T0,
        )
        evidence = build_evidence(
            claim_key="thesis.runtime",
            source=source,
            statement="Initial evidence supports the runtime thesis",
            recorded_at=T0,
            event_time=T0,
        )
        ledger = EvidenceLedger()
        ledger.register_source(source)
        ledger.append(evidence)
        persist_ledger(db, ledger)

        graph = ClaimGraph()
        claim = build_claim(
            claim_key="thesis.runtime",
            statement="Runtime belief monitor is active",
            kind=ClaimKind.FACT,
            created_at=T0,
        )
        graph.register_claim(claim)
        graph.link_evidence(
            claim_id=claim.claim_id,
            evidence_id=evidence.evidence_id,
            ledger=ledger,
        )
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, FalsificationEngine())
        return claim.claim_id
    finally:
        db.close()


def test_scheduler_skips_when_research_inputs_are_unchanged(monkeypatch):
    Session = _session_factory()
    _seed(Session)

    import src.modules.research.belief_scheduler as scheduler_module

    monkeypatch.setattr(
        scheduler_module,
        "open_research_session",
        lambda: Session(),
    )
    scheduler = PersistentBeliefScheduler(interval_seconds=60)

    asyncio.run(scheduler._scan())
    db = Session()
    try:
        first_count = db.query(ResearchBeliefCycleRecord).count()
    finally:
        db.close()

    asyncio.run(scheduler._scan())
    db = Session()
    try:
        second_count = db.query(ResearchBeliefCycleRecord).count()
    finally:
        db.close()

    assert first_count == 1
    assert second_count == 1


def test_scheduler_runs_again_after_evidence_and_link_change(monkeypatch):
    Session = _session_factory()
    claim_id = _seed(Session)

    import src.modules.research.belief_scheduler as scheduler_module
    from src.modules.research.evidence_store import load_all_evidence_ledger
    from src.modules.research.reasoning_store import load_claim_graph

    monkeypatch.setattr(scheduler_module, "SessionLocal", Session)
    scheduler = PersistentBeliefScheduler(interval_seconds=60)
    asyncio.run(scheduler._scan())

    db = Session()
    try:
        ledger = load_all_evidence_ledger(db)
        source = build_source(
            url="https://second.example.com/report",
            content="Second evidence",
            publisher="second",
            source_tier=SourceTier.PRIMARY,
            source_family="second",
            published_at=T0,
            retrieved_at=T0,
        )
        record = build_evidence(
            claim_key="thesis.runtime",
            source=source,
            statement="Second independent evidence",
        )
        ledger.register_source(source)
        ledger.append(record)
        persist_ledger(db, ledger)

        graph = load_claim_graph(db, ledger=ledger)
        graph.link_evidence(
            claim_id=claim_id,
            evidence_id=record.evidence_id,
            ledger=ledger,
        )
        persist_claim_graph(db, graph)
    finally:
        db.close()

    asyncio.run(scheduler._scan())
    db = Session()
    try:
        assert db.query(ResearchBeliefCycleRecord).count() == 2
    finally:
        db.close()
