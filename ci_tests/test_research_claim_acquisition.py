from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from src.modules.research.automatic_research import ResearchDocument
from src.modules.research.claim_acquisition import (
    ClaimCandidate,
    GeneralClaimAcquisition,
    GroundedClaimExtractor,
)
from src.modules.research.claim_graph import ClaimGraph, ClaimKind
from src.modules.research.evidence import ObservationKind
from src.modules.research.falsification import (
    FalsificationEngine,
    FalsificationRuleType,
)
from src.modules.research.ledger import EvidenceLedger
from src.platform.persistence.migrations import (
    _m127_research_evidence_foundation,
    _m128_claim_graph_and_falsification,
    _m129_persistent_belief_state,
    _m130_automatic_research_loop,
    _m131_general_claim_acquisition,
)
from src.platform.persistence.models import ResearchClaimCandidateRecord


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
        _m130_automatic_research_loop(conn)
        _m131_general_claim_acquisition(conn)
    Session = sessionmaker(bind=engine)
    return engine, Session()


class _Extractor:
    def __init__(self, by_url):
        self.by_url = by_url

    async def extract(self, *, document, topic_hint, max_candidates):
        return list(self.by_url.get(document.url, []))[:max_candidates]


def _candidate(
    *,
    quote,
    statement,
    key="entity.metric",
    kind=ClaimKind.FACT,
    observation=ObservationKind.ACTUAL,
    confidence=0.8,
    testable=True,
    supersedes=False,
    time_sensitive=False,
):
    return ClaimCandidate(
        quote=quote,
        statement=statement,
        proposed_claim_key=key,
        kind=kind,
        observation_kind=observation,
        confidence=confidence,
        testable=testable,
        valid_from=T0,
        valid_until=None,
        time_sensitive=time_sensitive,
        freshness_seconds=3600 if time_sensitive else None,
        supersedes_previous=supersedes,
    )


def _doc(url, text):
    return ResearchDocument(
        url=url,
        title="Source",
        text=text,
        tool_name="fixture",
        publisher="Fixture Publisher",
        published_at=T0,
    )


@pytest.mark.asyncio
async def test_grounded_candidate_becomes_claim_evidence_rules_and_belief():
    _engine, db = _db()
    try:
        doc = _doc(
            "https://economy.example/a",
            "Official data show inflation fell to 3.2 percent in August.",
        )
        candidate = _candidate(
            quote="inflation fell to 3.2 percent in August",
            statement="Inflation was 3.2 percent in August.",
            key="economy.inflation.2026-08",
            time_sensitive=True,
        )
        graph = ClaimGraph()
        ledger = EvidenceLedger()
        falsification = FalsificationEngine()
        result = await GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=_Extractor({doc.url: [candidate]}),
        ).run(db=db, documents=[doc], seed_topic="inflation")

        assert result.claims_accepted == 1
        assert result.rejected == 0
        assert len(graph.claims) == 1
        assert len(ledger.records) == 1
        rules = falsification.rules_for_claim(graph.claims[0].claim_id)
        types = {rule.rule_type for rule in rules}
        assert FalsificationRuleType.CONTRADICTORY_EVIDENCE in types
        assert FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT in types
        assert FalsificationRuleType.FRESHNESS_FAILURE in types
        assert result.belief_changes >= 1
    finally:
        db.close()


@pytest.mark.asyncio
async def test_duplicate_statement_reuses_claim_and_adds_independent_evidence():
    _engine, db = _db()
    try:
        statement = "Company revenue was 10 billion dollars in 2025."
        d1 = _doc(
            "https://company-a.example/filing",
            "The filing states revenue was 10 billion dollars in 2025.",
        )
        d2 = _doc(
            "https://auditor.example/report",
            "The audit confirms revenue was 10 billion dollars in 2025.",
        )
        c1 = _candidate(
            quote="revenue was 10 billion dollars in 2025",
            statement=statement,
            key="company.revenue.2025",
        )
        c2 = _candidate(
            quote="revenue was 10 billion dollars in 2025",
            statement=statement,
            key="company.revenue.2025",
        )
        graph = ClaimGraph()
        ledger = EvidenceLedger()
        falsification = FalsificationEngine()
        result = await GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=_Extractor({d1.url: [c1], d2.url: [c2]}),
        ).run(db=db, documents=[d1, d2], seed_topic="company revenue")

        assert len(graph.claims) == 1
        assert len(ledger.records) == 2
        assert result.claims_accepted == 1
        assert result.duplicates == 1
        rows = db.query(ResearchClaimCandidateRecord).all()
        assert {row.decision for row in rows} == {"accepted", "duplicate"}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_explicit_revision_supersedes_previous_claim():
    _engine, db = _db()
    try:
        d1 = _doc(
            "https://stats.example/original",
            "The agency reported unemployment at 6.2 percent.",
        )
        d2 = _doc(
            "https://stats.example/revision",
            "The agency revised unemployment to 5.8 percent.",
        )
        c1 = _candidate(
            quote="unemployment at 6.2 percent",
            statement="Unemployment was 6.2 percent.",
            key="macro.unemployment.current",
        )
        c2 = _candidate(
            quote="revised unemployment to 5.8 percent",
            statement="Unemployment was 5.8 percent.",
            key="macro.unemployment.current",
            observation=ObservationKind.REVISION,
            supersedes=True,
        )
        graph = ClaimGraph()
        ledger = EvidenceLedger()
        falsification = FalsificationEngine()
        engine = GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=_Extractor({d1.url: [c1], d2.url: [c2]}),
        )
        first = await engine.run(
            db=db,
            documents=[d1],
            evaluated_at=T0,
        )
        second = await engine.run(
            db=db,
            documents=[d2],
            evaluated_at=T0 + timedelta(hours=1),
        )

        assert first.claims_accepted == 1
        assert second.superseded == 1
        latest = max(graph.claims, key=lambda item: item.created_at)
        assert latest.supersedes is not None
    finally:
        db.close()


@pytest.mark.asyncio
async def test_supersede_flag_without_revision_cue_is_denied():
    _engine, db = _db()
    try:
        d1 = _doc("https://science.example/a", "The trial found efficacy of 61 percent.")
        d2 = _doc("https://science.example/b", "A second trial found efficacy of 58 percent.")
        c1 = _candidate(
            quote="efficacy of 61 percent",
            statement="Trial efficacy was 61 percent.",
            key="science.trial.efficacy",
        )
        c2 = _candidate(
            quote="efficacy of 58 percent",
            statement="Trial efficacy was 58 percent.",
            key="science.trial.efficacy",
            supersedes=True,
        )
        graph = ClaimGraph()
        ledger = EvidenceLedger()
        falsification = FalsificationEngine()
        engine = GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=_Extractor({d1.url: [c1], d2.url: [c2]}),
        )
        await engine.run(
            db=db,
            documents=[d1],
            evaluated_at=T0,
        )
        result = await engine.run(
            db=db,
            documents=[d2],
            evaluated_at=T0 + timedelta(hours=1),
        )

        assert result.superseded == 0
        assert len(graph.claims) == 2
        newest = max(graph.claims, key=lambda item: item.created_at)
        assert newest.supersedes is None
        assert newest.metadata["supersede_denied"] is True
    finally:
        db.close()


@pytest.mark.asyncio
async def test_non_testable_candidate_is_rejected_and_audited():
    _engine, db = _db()
    try:
        doc = _doc(
            "https://opinion.example/a",
            "The author says the future feels exciting and mysterious.",
        )
        candidate = _candidate(
            quote="future feels exciting and mysterious",
            statement="The future is exciting and mysterious.",
            key="opinion.future",
            kind=ClaimKind.INTERPRETATION,
            testable=False,
        )
        graph = ClaimGraph()
        ledger = EvidenceLedger()
        falsification = FalsificationEngine()
        result = await GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=_Extractor({doc.url: [candidate]}),
        ).run(db=db, documents=[doc])

        assert result.rejected == 1
        assert len(graph.claims) == 0
        row = db.query(ResearchClaimCandidateRecord).one()
        assert row.decision == "rejected"
        assert row.reason == "not_testable"
    finally:
        db.close()


@pytest.mark.asyncio
async def test_domain_neutral_kinds_work_across_company_science_and_policy():
    _engine, db = _db()
    try:
        docs = [
            _doc("https://co.example/a", "Management forecast revenue growth of 8 percent next year."),
            _doc("https://lab.example/a", "The experiment found the catalyst reduced reaction time by 12 percent."),
            _doc("https://gov.example/a", "The ministry announced the tariff will remain at 5 percent."),
        ]
        candidates = {
            docs[0].url: [_candidate(
                quote="forecast revenue growth of 8 percent next year",
                statement="Management forecasts revenue growth of 8 percent next year.",
                key="company.revenue_growth.next_year",
                kind=ClaimKind.FORECAST,
                observation=ObservationKind.GUIDANCE,
            )],
            docs[1].url: [_candidate(
                quote="catalyst reduced reaction time by 12 percent",
                statement="The catalyst reduced reaction time by 12 percent.",
                key="science.catalyst.reaction_time",
            )],
            docs[2].url: [_candidate(
                quote="tariff will remain at 5 percent",
                statement="The tariff will remain at 5 percent.",
                key="policy.tariff.current",
                observation=ObservationKind.POLICY_STATEMENT,
            )],
        }
        graph = ClaimGraph()
        result = await GeneralClaimAcquisition(
            graph=graph,
            ledger=EvidenceLedger(),
            falsification=FalsificationEngine(),
            extractor=_Extractor(candidates),
        ).run(db=db, documents=docs)

        assert result.claims_accepted == 3
        assert {claim.kind for claim in graph.claims} == {
            ClaimKind.FACT,
            ClaimKind.FORECAST,
        }
        assert {claim.claim_key for claim in graph.claims} == {
            "company.revenue_growth.next_year",
            "science.catalyst.reaction_time",
            "policy.tariff.current",
        }
    finally:
        db.close()


def test_migration_131_creates_acquisition_tables_and_indexes():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
        _m130_automatic_research_loop(conn)
        _m131_general_claim_acquisition(conn)
    inspector = inspect(engine)
    assert {
        "research_acquisition_runs",
        "research_claim_candidates",
    }.issubset(set(inspector.get_table_names()))
    indexes = {
        row["name"]
        for row in inspector.get_indexes("research_claim_candidates")
    }
    assert "ix_research_candidate_fingerprint" in indexes
    assert "ix_research_candidate_decision" in indexes



class _FakeAI:
    def __init__(self, payload):
        self.payload = payload

    async def chat_multi(self, *_args, **_kwargs):
        return self.payload


@pytest.mark.asyncio
async def test_grounded_extractor_rejects_hallucinated_quote():
    extractor = GroundedClaimExtractor(
        _FakeAI(
            '{"claims":[{"quote":"profits doubled overnight",'
            '"statement":"Profits doubled overnight.",'
            '"claim_key":"company.profits.current","kind":"fact",'
            '"observation_kind":"actual","confidence":0.9,"testable":true}]}'
        )
    )
    doc = _doc(
        "https://company.example/report",
        "The company reported stable profits for the quarter.",
    )
    rows = await extractor.extract(
        document=doc,
        topic_hint="company profits",
        max_candidates=3,
    )
    assert rows == []


@pytest.mark.asyncio
async def test_grounded_extractor_accepts_exact_quote():
    extractor = GroundedClaimExtractor(
        _FakeAI(
            '{"claims":[{"quote":"revenue increased by 9 percent",'
            '"statement":"Revenue increased by 9 percent.",'
            '"claim_key":"company.revenue.growth","kind":"fact",'
            '"observation_kind":"actual","confidence":0.8,"testable":true}]}'
        )
    )
    doc = _doc(
        "https://company.example/report",
        "The company said revenue increased by 9 percent during the quarter.",
    )
    rows = await extractor.extract(
        document=doc,
        topic_hint="company revenue",
        max_candidates=3,
    )
    assert len(rows) == 1
    assert rows[0].proposed_claim_key == "company.revenue.growth"



class _TimeoutAI:
    async def chat_multi(self, *_args, **_kwargs):
        raise TimeoutError("provider timeout")


@pytest.mark.asyncio
async def test_timeout_fallback_emits_only_verbatim_grounded_claims():
    extractor = GroundedClaimExtractor(
        _TimeoutAI(),
        timeout_seconds=15,
        max_source_chars=7000,
    )
    doc = _doc(
        "https://semiconductor.example/report",
        (
            "The industry is undergoing rapid change. "
            "Global semiconductor sales reached $627.6 billion in 2025. "
            "Analysts expect advanced packaging capacity to grow by 18 percent by 2027. "
            "Subscribe to our newsletter for weekly updates."
        ),
    )
    rows = await extractor.extract(
        document=doc,
        topic_hint="semiconductor supply chain",
        max_candidates=3,
    )
    assert len(rows) == 2
    source = doc.text.casefold()
    for row in rows:
        assert row.quote == row.statement
        assert row.quote.casefold() in source
        assert row.metadata["extractor"] == "deterministic_grounded_fallback_v1"
        assert row.metadata["fallback_reason"] == "TimeoutError"
        assert row.confidence == 0.42
    assert any(row.kind is ClaimKind.FORECAST for row in rows)
    assert any(row.kind is ClaimKind.FACT for row in rows)


@pytest.mark.asyncio
async def test_timeout_fallback_rejects_boilerplate_and_non_numeric_opinion():
    extractor = GroundedClaimExtractor(_TimeoutAI())
    doc = _doc(
        "https://example.com/opinion",
        (
            "The future of technology feels extremely exciting. "
            "Subscribe to our newsletter for 20 weekly updates. "
            "Privacy Policy 2026 applies to this website."
        ),
    )
    rows = await extractor.extract(
        document=doc,
        topic_hint="technology",
        max_candidates=3,
    )
    assert rows == []
