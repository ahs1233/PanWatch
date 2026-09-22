from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.modules.research.automatic_research import (
    AhmedToolboxResearchGateway,
    AutomaticResearchLoop,
    ExtractedFinding,
    QuoteGroundedEvidenceExtractor,
    ResearchDocument,
)
from src.modules.research.claim_graph import ClaimGraph, ClaimKind, build_claim
from src.modules.research.evidence import (
    EvidenceRelation,
    ObservationKind,
    SourceTier,
    build_evidence,
    build_source,
)
from src.modules.research.evidence_store import persist_ledger
from src.modules.research.falsification import (
    FalsificationEngine,
    FalsificationRuleType,
    build_falsification_rule,
)
from src.modules.research.ledger import EvidenceLedger
from src.modules.research.reasoning_store import (
    persist_claim_graph,
    persist_falsification_engine,
)
from src.platform.persistence.migrations import (
    _m127_research_evidence_foundation,
    _m128_claim_graph_and_falsification,
    _m129_persistent_belief_state,
    _m130_automatic_research_loop,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
        _m130_automatic_research_loop(conn)
    Session = sessionmaker(bind=engine)
    return engine, Session()


def _base_graph(*, rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE):
    ledger = EvidenceLedger()
    source = build_source(
        url="https://internal.example.com/state",
        content="Internal state says the thesis is intact.",
        publisher="internal",
        source_tier=SourceTier.AGGREGATOR,
        source_family="internal",
        independence_key="internal",
        published_at=T0,
        retrieved_at=T0,
        observed_at=T0,
    )
    ledger.register_source(source)
    support = build_evidence(
        claim_key="macro.thesis",
        source=source,
        statement="Internal state says the thesis is intact.",
        relation=EvidenceRelation.SUPPORTS,
        observation_kind=ObservationKind.ESTIMATE,
        event_time=T0,
        observed_at=T0,
        recorded_at=T0,
        confidence=0.7,
    )
    ledger.append(support)

    graph = ClaimGraph()
    claim = build_claim(
        claim_key="macro.thesis",
        statement="The macro thesis is intact.",
        kind=ClaimKind.HYPOTHESIS,
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
    if rule_type is FalsificationRuleType.CONTRADICTORY_EVIDENCE:
        rule = build_falsification_rule(
            claim_id=claim.claim_id,
            description="Independent contradiction weakens the thesis.",
            rule_type=rule_type,
            min_sources=1,
            weight=1.0,
        )
    else:
        rule = build_falsification_rule(
            claim_id=claim.claim_id,
            description="Need two independent supporting sources.",
            rule_type=rule_type,
            evidence_claim_key="macro.thesis",
            min_sources=3,
            weight=1.0,
        )
    falsification.register_rule(rule, graph=graph)
    return ledger, graph, falsification, claim


class _FakeAI:
    def __init__(self, payload):
        self.payload = payload

    async def chat(self, *_args, **_kwargs):
        return self.payload


@pytest.mark.asyncio
async def test_quote_grounding_accepts_exact_source_quote():
    extractor = QuoteGroundedEvidenceExtractor(
        _FakeAI(
            '{"findings":[{"quote":"activity contracted sharply",'
            '"relation":"contradicts","observation_kind":"actual",'
            '"confidence":0.8,"numeric_value":null,"unit":"","period":""}]}'
        )
    )
    doc = ResearchDocument(
        url="https://source.example.com/a",
        title="A",
        text="The official report says activity contracted sharply in August.",
        tool_name="reach_read_url",
    )
    probe = type(
        "Probe",
        (),
        {"instruction": "find contradiction"},
    )()
    findings = await extractor.extract(
        probe=probe,
        rule=None,
        claim_statement="Activity is resilient.",
        evidence_claim_key="macro.activity",
        desired_relation=EvidenceRelation.CONTRADICTS,
        document=doc,
        max_findings=2,
    )
    assert len(findings) == 1
    assert findings[0].relation is EvidenceRelation.CONTRADICTS


@pytest.mark.asyncio
async def test_quote_grounding_rejects_hallucinated_quote():
    extractor = QuoteGroundedEvidenceExtractor(
        _FakeAI(
            '{"findings":[{"quote":"GDP collapsed by 20 percent",'
            '"relation":"contradicts","observation_kind":"actual",'
            '"confidence":0.99}]}'
        )
    )
    doc = ResearchDocument(
        url="https://source.example.com/a",
        title="A",
        text="The report says activity was unchanged.",
        tool_name="reach_read_url",
    )
    probe = type("Probe", (), {"instruction": "find contradiction"})()
    findings = await extractor.extract(
        probe=probe,
        rule=None,
        claim_statement="Activity is resilient.",
        evidence_claim_key="macro.activity",
        desired_relation=EvidenceRelation.CONTRADICTS,
        document=doc,
        max_findings=2,
    )
    assert findings == []


class _FakeGateway:
    def __init__(self, documents):
        self.documents = documents
        self.tool_calls = 0
        self.search_calls = 0

    async def search(self, query, *, max_sources):
        self.search_calls += 1
        self.tool_calls += 1
        return self.documents[:max_sources]


class _StaticExtractor:
    def __init__(self, findings):
        self.findings = findings
        self.calls = 0

    async def extract(self, **_kwargs):
        self.calls += 1
        return list(self.findings)


@pytest.mark.asyncio
async def test_counter_research_adds_evidence_and_changes_belief():
    _engine, db = _db()
    try:
        ledger, graph, falsification, claim = _base_graph()
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)

        gateway = _FakeGateway(
            [
                ResearchDocument(
                    url="https://independent.example.com/report",
                    title="Independent report",
                    text="Independent evidence says the thesis is no longer intact.",
                    tool_name="reach_read_url",
                    publisher="Independent",
                )
            ]
        )
        extractor = _StaticExtractor(
            [
                ExtractedFinding(
                    quote="the thesis is no longer intact",
                    relation=EvidenceRelation.CONTRADICTS,
                    observation_kind=ObservationKind.ACTUAL,
                    confidence=0.85,
                )
            ]
        )
        result = await AutomaticResearchLoop(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            gateway=gateway,
            extractor=extractor,
            max_probes=1,
            max_sources_per_probe=1,
            cooldown_minutes=180,
        ).run(db=db, evaluated_at=T0 + timedelta(minutes=5))

        assert result.status == "success"
        assert result.probes_executed == 1
        assert result.evidence_added == 1
        assert result.beliefs_changed >= 1
        assert gateway.search_calls == 1
        assert any(
            row.relation is EvidenceRelation.CONTRADICTS
            for row in ledger.records
        )
    finally:
        db.close()


@pytest.mark.asyncio
async def test_no_evidence_attempt_enters_cooldown_and_suppresses_repeat():
    _engine, db = _db()
    try:
        ledger, graph, falsification, _claim = _base_graph(
            rule_type=FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT
        )
        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)

        gateway = _FakeGateway(
            [
                ResearchDocument(
                    url="https://independent.example.com/report",
                    title="Independent report",
                    text="This document is irrelevant to the thesis.",
                    tool_name="reach_read_url",
                )
            ]
        )
        extractor = _StaticExtractor([])
        loop = AutomaticResearchLoop(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            gateway=gateway,
            extractor=extractor,
            max_probes=1,
            max_sources_per_probe=1,
            cooldown_minutes=180,
        )
        first = await loop.run(
            db=db,
            evaluated_at=T0 + timedelta(minutes=5),
        )
        second = await loop.run(
            db=db,
            evaluated_at=T0 + timedelta(minutes=15),
        )

        assert first.probes_executed == 1
        assert second.probes_executed == 0
        assert second.probes_skipped_cooldown >= 1
        assert gateway.search_calls == 1
    finally:
        db.close()


def test_gateway_tool_budget_blocks_runaway_calls():
    gateway = AhmedToolboxResearchGateway.__new__(AhmedToolboxResearchGateway)
    gateway.max_tool_calls = 1
    gateway.tool_calls = 0
    gateway._budget()
    assert gateway.tool_calls == 1
    with pytest.raises(RuntimeError, match="budget_exhausted"):
        gateway._budget()


@pytest.mark.asyncio
async def test_no_claims_consumes_no_research_tools():
    _engine, db = _db()
    try:
        gateway = _FakeGateway([])
        result = await AutomaticResearchLoop(
            graph=ClaimGraph(),
            ledger=EvidenceLedger(),
            falsification=FalsificationEngine(),
            gateway=gateway,
            extractor=_StaticExtractor([]),
        ).run(db=db, evaluated_at=T0)
        assert result.status == "no_claims"
        assert result.tool_calls == 0
        assert gateway.search_calls == 0
    finally:
        db.close()


def test_migration_130_creates_loop_tables_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
        _m130_automatic_research_loop(conn)
    inspector = inspect(engine)
    assert {
        "research_loop_runs",
        "research_probe_attempts",
    }.issubset(set(inspector.get_table_names()))
    indexes = {
        row["name"]
        for row in inspector.get_indexes("research_probe_attempts")
    }
    assert "ix_research_probe_key_attempted" in indexes
