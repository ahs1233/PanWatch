from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.modules.research import (
    ClaimGraph,
    ClaimKind,
    ClaimRelation,
    ClaimStatus,
    EvidenceLedger,
    EvidenceRelation,
    FalsificationEngine,
    FalsificationRuleType,
    FalsificationState,
    ObservationKind,
    SourceTier,
    build_claim,
    build_evidence,
    build_falsification_rule,
    build_source,
)
from src.modules.research.evidence_store import persist_ledger
from src.modules.research.reasoning_store import (
    load_claim_graph,
    load_falsification_engine,
    persist_claim_graph,
    persist_falsification_engine,
)
from src.platform.persistence.migrations import (
    _m127_research_evidence_foundation,
    _m128_claim_graph_and_falsification,
)
from src.platform.persistence.models import (
    ResearchClaimEdgeRecord,
    ResearchClaimEvidenceLinkRecord,
    ResearchClaimRecord,
    ResearchEvidenceRecord,
    ResearchFalsificationRuleRecord,
    ResearchSourceRecord,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _source(
    slug: str,
    *,
    tier: SourceTier = SourceTier.PRIMARY,
    upstream_origin: str = "",
):
    return build_source(
        url=f"https://{slug}.example.com/report",
        content=f"{slug} content",
        publisher=slug,
        source_tier=tier,
        source_family=slug,
        upstream_origin=upstream_origin,
        published_at=T0,
        retrieved_at=T0 + timedelta(minutes=1),
    )


def _evidence_ledger() -> tuple[EvidenceLedger, dict[str, object]]:
    ledger = EvidenceLedger()
    official = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
    secondary = _source("secondary", tier=SourceTier.SECONDARY)
    ledger.register_source(official)
    ledger.register_source(secondary)
    support = build_evidence(
        claim_key="macro.inflation.cooling",
        source=official,
        statement="Inflation cooled materially",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        confidence=0.95,
    )
    contradict = build_evidence(
        claim_key="macro.inflation.cooling",
        source=secondary,
        statement="Services inflation remains sticky",
        relation=EvidenceRelation.CONTRADICTS,
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        confidence=0.8,
    )
    ledger.append(support)
    ledger.append(contradict)
    return ledger, {
        "official": official,
        "secondary": secondary,
        "support": support,
        "contradict": contradict,
    }


def test_claim_identity_is_stable_and_registration_is_idempotent():
    graph = ClaimGraph()
    a = build_claim(
        claim_key="thesis.soft_landing",
        statement="The economy is moving toward a soft landing",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    b = build_claim(
        claim_key="thesis.soft_landing",
        statement="The economy is moving toward a soft landing",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0 + timedelta(hours=1),
    )
    assert a.claim_id == b.claim_id
    assert graph.register_claim(a).claim_id == a.claim_id
    assert graph.register_claim(a).claim_id == a.claim_id
    assert graph.register_claim(b).claim_id == a.claim_id


def test_reasoning_cycle_is_rejected():
    graph = ClaimGraph()
    a = build_claim(claim_key="a", statement="A", created_at=T0)
    b = build_claim(claim_key="b", statement="B", created_at=T0)
    graph.register_claim(a)
    graph.register_claim(b)
    graph.connect(a.claim_id, b.claim_id, ClaimRelation.SUPPORTS)
    with pytest.raises(ValueError, match="reasoning cycle"):
        graph.connect(b.claim_id, a.claim_id, ClaimRelation.DEPENDS_ON)


def test_independent_evidence_links_drive_claim_assessment():
    ledger, rows = _evidence_ledger()
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis.inflation",
        statement="Inflation is cooling enough to reduce policy pressure",
        kind=ClaimKind.HYPOTHESIS,
        prior_confidence=0.5,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=rows["support"].evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.SUPPORTS,
    )
    assessment = graph.assess(claim.claim_id, ledger)
    assert assessment.confidence > 0.7
    assert assessment.status is ClaimStatus.SUPPORTED
    assert assessment.independent_source_count == 1


def test_contradictory_evidence_reduces_claim_confidence():
    ledger, rows = _evidence_ledger()
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis.inflation",
        statement="Inflation is cooling enough to reduce policy pressure",
        kind=ClaimKind.HYPOTHESIS,
        prior_confidence=0.5,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=rows["support"].evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.SUPPORTS,
    )
    before = graph.assess(claim.claim_id, ledger).confidence
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=rows["contradict"].evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.CONTRADICTS,
    )
    after = graph.assess(claim.claim_id, ledger).confidence
    assert after < before


def test_required_dependency_failure_propagates_downstream():
    ledger = EvidenceLedger()
    graph = ClaimGraph()
    premise = build_claim(
        claim_key="premise",
        statement="Critical premise",
        prior_confidence=0.1,
        created_at=T0,
    )
    conclusion = build_claim(
        claim_key="conclusion",
        statement="Conclusion depends on premise",
        kind=ClaimKind.CONCLUSION,
        prior_confidence=0.6,
        created_at=T0,
    )
    graph.register_claim(premise)
    graph.register_claim(conclusion)
    graph.connect(
        premise.claim_id,
        conclusion.claim_id,
        ClaimRelation.DEPENDS_ON,
        required=True,
    )
    assessment = graph.assess(conclusion.claim_id, ledger)
    assert premise.claim_id in assessment.dependency_failures
    assert assessment.confidence < 0.35
    assert conclusion.claim_id in graph.impact_set(premise.claim_id)


def test_hypothesis_without_falsification_rule_cannot_finish_supported():
    ledger, rows = _evidence_ledger()
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis.inflation",
        statement="Inflation cooling will persist",
        kind=ClaimKind.HYPOTHESIS,
        prior_confidence=0.5,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=rows["support"].evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.SUPPORTS,
    )
    assert graph.assess(claim.claim_id, ledger).status is ClaimStatus.SUPPORTED

    engine = FalsificationEngine()
    report = engine.evaluate(claim.claim_id, graph=graph, ledger=ledger)
    assert report.final_status is ClaimStatus.OPEN
    assert report.coverage == 0.0
    assert report.probes[0].rule_id == "missing_falsification_rule"


def test_hard_numeric_falsifier_can_kill_supported_claim():
    ledger = EvidenceLedger()
    official = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(official)
    support = build_evidence(
        claim_key="macro.activity",
        source=official,
        statement="Activity remains resilient",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        confidence=0.95,
    )
    unemployment = build_evidence(
        claim_key="macro.unemployment",
        source=official,
        statement="Unemployment reached 6.2 percent",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        numeric_value=6.2,
        unit="%",
        period="2026-08",
    )
    ledger.append(support)
    ledger.append(unemployment)

    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis.soft_landing",
        statement="A soft landing remains intact",
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
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Soft landing fails if unemployment exceeds 6 percent",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        hard_fail=True,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
        required_kinds=(ObservationKind.ACTUAL,),
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(
        claim.claim_id,
        graph=graph,
        ledger=ledger,
        as_of=T0 + timedelta(hours=1),
    )
    assert report.final_status is ClaimStatus.FALSIFIED
    assert report.adjusted_confidence <= 0.05
    assert rule.rule_id in report.triggered_rules


def test_untestable_numeric_rule_generates_research_probe():
    ledger = EvidenceLedger()
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis",
        statement="Testable thesis",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    graph.register_claim(claim)
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Fails if unemployment exceeds 6 percent",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(claim.claim_id, graph=graph, ledger=ledger)
    assert report.results[0].state is FalsificationState.UNTESTABLE
    assert report.coverage == 0.0
    assert "Find a current" in report.probes[0].instruction


def test_counterclaim_confidence_can_falsify_target():
    ledger = EvidenceLedger()
    source = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    counter_evidence = build_evidence(
        claim_key="counter",
        source=source,
        statement="Counter thesis is supported",
        confidence=1.0,
    )
    ledger.append(counter_evidence)

    graph = ClaimGraph()
    target = build_claim(
        claim_key="target",
        statement="Target thesis",
        kind=ClaimKind.HYPOTHESIS,
        prior_confidence=0.6,
        created_at=T0,
    )
    counter = build_claim(
        claim_key="counter",
        statement="Counter thesis",
        kind=ClaimKind.HYPOTHESIS,
        prior_confidence=0.5,
        created_at=T0,
    )
    graph.register_claim(target)
    graph.register_claim(counter)
    graph.link_evidence(
        claim_id=counter.claim_id,
        evidence_id=counter_evidence.evidence_id,
        ledger=ledger,
    )
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=target.claim_id,
        description="Target fails if counter thesis becomes highly credible",
        rule_type=FalsificationRuleType.COUNTERCLAIM_CONFIDENCE,
        hard_fail=True,
        related_claim_id=counter.claim_id,
        threshold=0.7,
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(target.claim_id, graph=graph, ledger=ledger)
    assert report.final_status is ClaimStatus.FALSIFIED


def test_reasoning_persistence_roundtrip():
    engine_sql = create_engine("sqlite:///:memory:")
    ResearchSourceRecord.__table__.create(engine_sql)
    ResearchEvidenceRecord.__table__.create(engine_sql)
    ResearchClaimRecord.__table__.create(engine_sql)
    ResearchClaimEdgeRecord.__table__.create(engine_sql)
    ResearchClaimEvidenceLinkRecord.__table__.create(engine_sql)
    ResearchFalsificationRuleRecord.__table__.create(engine_sql)
    Session = sessionmaker(bind=engine_sql)
    db = Session()
    try:
        ledger, rows = _evidence_ledger()
        persist_ledger(db, ledger)

        graph = ClaimGraph()
        premise = build_claim(
            claim_key="premise",
            statement="Premise",
            kind=ClaimKind.FACT,
            created_at=T0,
        )
        conclusion = build_claim(
            claim_key="conclusion",
            statement="Conclusion",
            kind=ClaimKind.CONCLUSION,
            created_at=T0,
        )
        graph.register_claim(premise)
        graph.register_claim(conclusion)
        graph.connect(
            premise.claim_id,
            conclusion.claim_id,
            ClaimRelation.SUPPORTS,
            weight=0.8,
        )
        graph.link_evidence(
            claim_id=premise.claim_id,
            evidence_id=rows["support"].evidence_id,
            ledger=ledger,
        )
        first = persist_claim_graph(db, graph)
        second = persist_claim_graph(db, graph)
        assert first == {
            "claims_inserted": 2,
            "edges_inserted": 1,
            "links_inserted": 1,
        }
        assert second == {
            "claims_inserted": 0,
            "edges_inserted": 0,
            "links_inserted": 0,
        }

        fal = FalsificationEngine()
        rule = build_falsification_rule(
            claim_id=conclusion.claim_id,
            description="Conclusion fails if premise confidence collapses",
            rule_type=FalsificationRuleType.DEPENDENCY_FAILURE,
            related_claim_id=premise.claim_id,
            threshold=0.35,
        )
        fal.register_rule(rule, graph=graph)
        assert persist_falsification_engine(db, fal) == 1
        assert persist_falsification_engine(db, fal) == 0

        loaded_graph = load_claim_graph(db, ledger=ledger)
        loaded_fal = load_falsification_engine(db, graph=loaded_graph)
        assert len(loaded_graph.claims) == 2
        assert len(loaded_graph.edges) == 1
        assert len(loaded_graph.evidence_links) == 1
        assert len(loaded_fal.rules) == 1
        assert loaded_fal.rules[0].rule_id == rule.rule_id
    finally:
        db.close()


def test_migration_128_creates_reasoning_tables_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
    inspector = inspect(engine)
    names = set(inspector.get_table_names())
    assert {
        "research_claims",
        "research_claim_edges",
        "research_claim_evidence_links",
        "research_falsification_rules",
    }.issubset(names)
    edge_indexes = {
        row["name"]
        for row in inspector.get_indexes("research_claim_edges")
    }
    rule_indexes = {
        row["name"]
        for row in inspector.get_indexes("research_falsification_rules")
    }
    assert "ix_research_claim_edge_target" in edge_indexes
    assert "ix_research_falsification_claim" in rule_indexes
