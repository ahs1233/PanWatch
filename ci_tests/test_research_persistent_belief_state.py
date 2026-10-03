from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.modules.research import (
    BeliefEventType,
    BeliefStateEngine,
    ClaimGraph,
    ClaimKind,
    ClaimRelation,
    ClaimStatus,
    EvidenceLedger,
    EvidenceRelation,
    FalsificationEngine,
    FalsificationRuleType,
    ObservationKind,
    PanWatchBeliefMonitor,
    SourceTier,
    build_claim,
    build_evidence,
    build_falsification_rule,
    build_source,
)
from src.modules.research.belief_store import (
    load_belief_events,
    load_belief_history,
    load_latest_belief_snapshot,
    list_recent_belief_cycles,
)
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


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _source(slug: str, tier: SourceTier = SourceTier.PRIMARY):
    return build_source(
        url=f"https://{slug}.example.com/report",
        content=f"{slug} content",
        publisher=slug,
        source_tier=tier,
        source_family=slug,
        published_at=T0,
        retrieved_at=T0 + timedelta(minutes=1),
    )


def _setup_db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
    Session = sessionmaker(bind=engine)
    return engine, Session()


def _basic_monitor():
    ledger = EvidenceLedger()
    source = _source("official", SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    support = build_evidence(
        claim_key="macro.activity",
        source=source,
        statement="Activity remains resilient",
        event_time=T0,
        recorded_at=T0,
        confidence=0.95,
    )
    ledger.append(support)

    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis.soft_landing",
        statement="Soft landing remains intact",
        kind=ClaimKind.CONCLUSION,
        prior_confidence=0.5,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=support.evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.SUPPORTS,
    )

    falsification = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Soft landing fails if unemployment exceeds 6 percent",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        hard_fail=True,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
    )
    falsification.register_rule(rule, graph=graph)
    return ledger, graph, falsification, claim, source


def test_first_cycle_initializes_and_persists_belief():
    _engine, db = _setup_db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic_monitor()
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)

        cycle = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        ).run_cycle(db=db, evaluated_at=T0 + timedelta(hours=1))

        assert cycle.claim_count == 1
        assert cycle.changed_count == 1
        assert cycle.events[0].event_type is BeliefEventType.INITIALIZED

        latest = load_latest_belief_snapshot(db, claim_id=claim.claim_id)
        assert latest is not None
        assert latest.snapshot_id == cycle.updates[0].snapshot.snapshot_id
        assert len(load_belief_history(db, claim_id=claim.claim_id)) == 1
        assert len(list_recent_belief_cycles(db)) == 1
    finally:
        db.close()


def test_identical_next_cycle_suppresses_material_change_events():
    _engine, db = _setup_db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic_monitor()
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        monitor.run_cycle(db=db, evaluated_at=T0 + timedelta(hours=1))
        second = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=2),
        )
        assert second.changed_count == 0
        assert second.events == ()
        assert len(load_belief_history(db, claim_id=claim.claim_id)) == 2
    finally:
        db.close()


def test_new_contradiction_creates_explainable_confidence_drop():
    _engine, db = _setup_db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic_monitor()
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        first = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=1),
        )

        other = _source("secondary", SourceTier.SECONDARY)
        ledger.register_source(other)
        contradiction = build_evidence(
            claim_key="macro.activity",
            source=other,
            statement="Activity has deteriorated sharply",
            relation=EvidenceRelation.CONTRADICTS,
            event_time=T0 + timedelta(hours=2),
            recorded_at=T0 + timedelta(hours=2),
            confidence=0.95,
        )
        ledger.append(contradiction)
        graph.link_evidence(
            claim_id=claim.claim_id,
            evidence_id=contradiction.evidence_id,
            ledger=ledger,
            relation=EvidenceRelation.CONTRADICTS,
        )
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)

        second = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=3),
        )
        assert (
            second.updates[0].snapshot.final_confidence
            < first.updates[0].snapshot.final_confidence
        )
        event_types = {event.event_type for event in second.events}
        assert BeliefEventType.CONFIDENCE_LOWERED in event_types
        assert BeliefEventType.EVIDENCE_CHANGED in event_types
        events = load_belief_events(db, claim_id=claim.claim_id)
        assert any(
            event.event_type is BeliefEventType.CONFIDENCE_LOWERED
            for event in events
        )
    finally:
        db.close()


def test_falsification_then_revision_recovery_preserves_history():
    _engine, db = _setup_db()
    try:
        ledger, graph, falsification, claim, source = _basic_monitor()
        original = build_evidence(
            claim_key="macro.unemployment",
            source=source,
            statement="Unemployment reached 6.2 percent",
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0 + timedelta(hours=1),
            recorded_at=T0 + timedelta(hours=1),
            numeric_value=6.2,
            unit="%",
            period="2026-08",
        )
        ledger.append(original)

        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        first = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=2),
        )
        assert (
            first.updates[0].snapshot.final_status
            is ClaimStatus.FALSIFIED
        )

        revision = build_evidence(
            claim_key="macro.unemployment",
            source=source,
            statement="Unemployment revised to 5.8 percent",
            observation_kind=ObservationKind.REVISION,
            event_time=T0 + timedelta(hours=3),
            recorded_at=T0 + timedelta(hours=3),
            numeric_value=5.8,
            unit="%",
            period="2026-08",
            revision_of=original.evidence_id,
            supersedes=original.evidence_id,
        )
        ledger.append(revision)
        persist_ledger(db, ledger)

        second = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=4),
        )
        assert (
            second.updates[0].snapshot.final_status
            is not ClaimStatus.FALSIFIED
        )
        assert any(
            event.event_type is BeliefEventType.RECOVERED
            for event in second.events
        )

        history = load_belief_history(db, claim_id=claim.claim_id)
        assert history[0].final_status is ClaimStatus.FALSIFIED
        assert history[1].final_status is not ClaimStatus.FALSIFIED
    finally:
        db.close()


def test_future_revision_does_not_rewrite_past_belief():
    ledger, graph, falsification, claim, source = _basic_monitor()
    original = build_evidence(
        claim_key="macro.unemployment",
        source=source,
        statement="Unemployment reached 6.2 percent",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0 + timedelta(hours=1),
        recorded_at=T0 + timedelta(hours=1),
        numeric_value=6.2,
        unit="%",
        period="2026-08",
    )
    ledger.append(original)
    revision = build_evidence(
        claim_key="macro.unemployment",
        source=source,
        statement="Unemployment revised to 5.8 percent",
        observation_kind=ObservationKind.REVISION,
        event_time=T0 + timedelta(hours=3),
        recorded_at=T0 + timedelta(hours=3),
        numeric_value=5.8,
        unit="%",
        period="2026-08",
        revision_of=original.evidence_id,
        supersedes=original.evidence_id,
    )
    ledger.append(revision)

    early = falsification.evaluate(
        claim.claim_id,
        graph=graph,
        ledger=ledger,
        as_of=T0 + timedelta(hours=2),
    )
    late = falsification.evaluate(
        claim.claim_id,
        graph=graph,
        ledger=ledger,
        as_of=T0 + timedelta(hours=4),
    )
    assert early.final_status is ClaimStatus.FALSIFIED
    assert late.final_status is not ClaimStatus.FALSIFIED


def test_dependency_failure_updates_downstream_belief():
    _engine, db = _setup_db()
    try:
        ledger = EvidenceLedger()
        source = _source("official", SourceTier.OFFICIAL_PRIMARY)
        ledger.register_source(source)
        premise_support = build_evidence(
            claim_key="premise",
            source=source,
            statement="Premise initially supported",
            event_time=T0,
            recorded_at=T0,
            confidence=0.95,
        )
        ledger.append(premise_support)

        graph = ClaimGraph()
        premise = build_claim(
            claim_key="premise",
            statement="Critical premise",
            kind=ClaimKind.HYPOTHESIS,
            created_at=T0,
        )
        conclusion = build_claim(
            claim_key="conclusion",
            statement="Conclusion depends on premise",
            kind=ClaimKind.CONCLUSION,
            created_at=T0,
        )
        graph.register_claim(premise)
        graph.register_claim(conclusion)
        graph.link_evidence(
            claim_id=premise.claim_id,
            evidence_id=premise_support.evidence_id,
            ledger=ledger,
        )
        graph.connect(
            premise.claim_id,
            conclusion.claim_id,
            ClaimRelation.DEPENDS_ON,
            required=True,
        )

        falsification = FalsificationEngine()
        premise_rule = build_falsification_rule(
            claim_id=premise.claim_id,
            description="Premise fails if counter evidence appears",
            rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
            hard_fail=True,
            min_sources=1,
        )
        conclusion_rule = build_falsification_rule(
            claim_id=conclusion.claim_id,
            description="Conclusion fails if premise collapses",
            rule_type=FalsificationRuleType.DEPENDENCY_FAILURE,
            related_claim_id=premise.claim_id,
            threshold=0.35,
        )
        falsification.register_rule(premise_rule, graph=graph)
        falsification.register_rule(conclusion_rule, graph=graph)

        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        monitor.run_cycle(db=db, evaluated_at=T0 + timedelta(hours=1))

        counter = _source("counter", SourceTier.PRIMARY)
        ledger.register_source(counter)
        contradiction = build_evidence(
            claim_key="premise",
            source=counter,
            statement="Premise contradicted",
            relation=EvidenceRelation.CONTRADICTS,
            event_time=T0 + timedelta(hours=2),
            recorded_at=T0 + timedelta(hours=2),
            confidence=1.0,
        )
        ledger.append(contradiction)
        graph.link_evidence(
            claim_id=premise.claim_id,
            evidence_id=contradiction.evidence_id,
            ledger=ledger,
            relation=EvidenceRelation.CONTRADICTS,
        )
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)

        cycle = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=3),
        )
        conclusion_update = next(
            update
            for update in cycle.updates
            if update.snapshot.claim_id == conclusion.claim_id
        )
        assert (
            conclusion_update.snapshot.dependency_failures
            == (premise.claim_id,)
        )
        assert conclusion.claim_id in cycle.impacted_claim_ids or premise.claim_id in {
            update.snapshot.claim_id for update in cycle.updates if update.changed
        }
    finally:
        db.close()


def test_migration_129_creates_belief_tables_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
    inspector = inspect(engine)
    names = set(inspector.get_table_names())
    assert {
        "research_belief_cycles",
        "research_belief_snapshots",
        "research_belief_events",
    }.issubset(names)
    snapshot_indexes = {
        row["name"]
        for row in inspector.get_indexes("research_belief_snapshots")
    }
    event_indexes = {
        row["name"]
        for row in inspector.get_indexes("research_belief_events")
    }
    assert "ix_research_belief_claim_evaluated" in snapshot_indexes
    assert "ix_research_belief_event_claim_time" in event_indexes
