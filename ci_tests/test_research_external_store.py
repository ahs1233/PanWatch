from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from sqlalchemy import inspect

from src.modules.research import (
    ClaimGraph,
    ClaimKind,
    EvidenceLedger,
    FalsificationEngine,
    FalsificationRuleType,
    PanWatchBeliefMonitor,
    SourceTier,
    build_claim,
    build_evidence,
    build_falsification_rule,
    build_source,
)
from src.modules.research.belief_store import load_belief_history
from src.modules.research.evidence_store import persist_ledger
from src.modules.research.reasoning_store import (
    persist_claim_graph,
    persist_falsification_engine,
)
from src.modules.research.research_store import (
    init_research_store,
    open_research_session,
    research_store_dialect,
    research_store_is_external,
)


UTC = timezone.utc
NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _settings(url: str):
    return SimpleNamespace(research_database_url=url)


def test_external_research_store_provisions_all_tables(tmp_path):
    url = f"sqlite:///{tmp_path / 'research.db'}"
    try:
        assert init_research_store(_settings(url)) is True
        assert research_store_is_external() is True
        assert research_store_dialect() == "sqlite"

        db = open_research_session()
        try:
            tables = set(inspect(db.get_bind()).get_table_names())
        finally:
            db.close()

        assert {
            "research_sources",
            "research_evidence",
            "research_claims",
            "research_claim_edges",
            "research_claim_evidence_links",
            "research_falsification_rules",
            "research_belief_cycles",
            "research_belief_snapshots",
            "research_belief_events",
            "research_loop_runs",
            "research_probe_attempts",
            "research_acquisition_runs",
            "research_claim_candidates",
            "research_claim_resolutions",
        }.issubset(tables)
    finally:
        init_research_store(_settings(""))


def test_persistent_cycle_uses_configured_research_store(tmp_path):
    url = f"sqlite:///{tmp_path / 'research-cycle.db'}"
    try:
        assert init_research_store(_settings(url)) is True

        ledger = EvidenceLedger()
        source = build_source(
            url="https://official.example.com/release",
            content="Official release",
            publisher="official",
            source_tier=SourceTier.OFFICIAL_PRIMARY,
            source_family="official",
            published_at=NOW,
            retrieved_at=NOW,
        )
        ledger.register_source(source)
        evidence = build_evidence(
            claim_key="macro.activity",
            source=source,
            statement="Activity remains resilient",
            event_time=NOW,
            recorded_at=NOW,
            confidence=0.95,
        )
        ledger.append(evidence)

        graph = ClaimGraph()
        claim = build_claim(
            claim_key="thesis.activity",
            statement="Activity is resilient",
            kind=ClaimKind.CONCLUSION,
            created_at=NOW,
        )
        graph.register_claim(claim)
        graph.link_evidence(
            claim_id=claim.claim_id,
            evidence_id=evidence.evidence_id,
            ledger=ledger,
        )

        falsification = FalsificationEngine()
        rule = build_falsification_rule(
            claim_id=claim.claim_id,
            description="Claim fails if independent support disappears",
            rule_type=FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT,
            evidence_claim_key="macro.activity",
            min_sources=1,
        )
        falsification.register_rule(rule, graph=graph)

        db = open_research_session()
        try:
            persist_ledger(db, ledger)
            persist_claim_graph(db, graph)
            persist_falsification_engine(db, falsification)
        finally:
            db.close()

        cycle = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        ).run_persistent_cycle(evaluated_at=NOW)

        assert cycle.claim_count == 1

        db = open_research_session()
        try:
            history = load_belief_history(db, claim_id=claim.claim_id)
        finally:
            db.close()
        assert len(history) == 1
        assert history[0].claim_id == claim.claim_id
    finally:
        init_research_store(_settings(""))


def test_missing_external_url_falls_back_to_local_store():
    assert init_research_store(_settings("")) is False
    assert research_store_is_external() is False
    assert research_store_dialect() == "sqlite"
