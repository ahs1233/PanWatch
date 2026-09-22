"""Automatic Research Loop benchmark track v1.4."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine
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


@dataclass(frozen=True)
class AutomaticResearchCase:
    case_id: str
    category: str
    passed: bool
    points: float
    earned: float
    detail: str


def _db():
    engine = create_engine("sqlite:///:memory:")
    with engine.begin() as conn:
        _m127_research_evidence_foundation(conn)
        _m128_claim_graph_and_falsification(conn)
        _m129_persistent_belief_state(conn)
        _m130_automatic_research_loop(conn)
    Session = sessionmaker(bind=engine)
    return Session()


def _state(rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE):
    ledger = EvidenceLedger()
    source = build_source(
        url="https://internal.example.com/state",
        content="Internal estimate supports the thesis.",
        publisher="internal",
        source_tier=SourceTier.AGGREGATOR,
        source_family="internal",
        independence_key="internal",
        published_at=T0,
        retrieved_at=T0,
        observed_at=T0,
    )
    ledger.register_source(source)
    record = build_evidence(
        claim_key="macro.thesis",
        source=source,
        statement="Internal estimate supports the thesis.",
        relation=EvidenceRelation.SUPPORTS,
        observation_kind=ObservationKind.ESTIMATE,
        event_time=T0,
        observed_at=T0,
        recorded_at=T0,
        confidence=0.7,
    )
    ledger.append(record)
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="macro.thesis",
        statement="The macro thesis is intact.",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=record.evidence_id,
        ledger=ledger,
    )
    falsification = FalsificationEngine()
    if rule_type is FalsificationRuleType.CONTRADICTORY_EVIDENCE:
        rule = build_falsification_rule(
            claim_id=claim.claim_id,
            description="Independent contradiction weakens the thesis.",
            rule_type=rule_type,
            min_sources=1,
        )
    else:
        rule = build_falsification_rule(
            claim_id=claim.claim_id,
            description="Need three independent supporting sources.",
            rule_type=rule_type,
            evidence_claim_key="macro.thesis",
            min_sources=3,
        )
    falsification.register_rule(rule, graph=graph)
    return ledger, graph, falsification, claim


class _AI:
    def __init__(self, text: str):
        self.text = text

    async def chat(self, *_args, **_kwargs):
        return self.text

    async def chat_multi(self, *_args, **_kwargs):
        return self.text


class _Gateway:
    def __init__(self, docs):
        self.docs = list(docs)
        self.tool_calls = 0
        self.search_calls = 0

    async def search(self, _query, *, max_sources):
        self.search_calls += 1
        self.tool_calls += 1
        return self.docs[:max_sources]


class _Extractor:
    def __init__(self, findings):
        self.findings = list(findings)

    async def extract(self, **_kwargs):
        return list(self.findings)


def _grounded_quote() -> tuple[bool, str]:
    async def run():
        extractor = QuoteGroundedEvidenceExtractor(
            _AI(
                '{"findings":[{"quote":"activity contracted sharply",'
                '"relation":"contradicts","observation_kind":"actual",'
                '"confidence":0.8}]}'
            )
        )
        probe = type("Probe", (), {"instruction": "find contradiction"})()
        findings = await extractor.extract(
            probe=probe,
            rule=None,
            claim_statement="Activity is resilient.",
            evidence_claim_key="macro.activity",
            desired_relation=EvidenceRelation.CONTRADICTS,
            document=ResearchDocument(
                url="https://source.example/a",
                title="",
                text="The release says activity contracted sharply in August.",
                tool_name="read",
            ),
            max_findings=2,
        )
        return len(findings) == 1
    ok = asyncio.run(run())
    return ok, "exact source quote must be accepted"


def _hallucinated_quote_rejected() -> tuple[bool, str]:
    async def run():
        extractor = QuoteGroundedEvidenceExtractor(
            _AI(
                '{"findings":[{"quote":"GDP collapsed by 20 percent",'
                '"relation":"contradicts","observation_kind":"actual",'
                '"confidence":0.99}]}'
            )
        )
        probe = type("Probe", (), {"instruction": "find contradiction"})()
        findings = await extractor.extract(
            probe=probe,
            rule=None,
            claim_statement="Activity is resilient.",
            evidence_claim_key="macro.activity",
            desired_relation=EvidenceRelation.CONTRADICTS,
            document=ResearchDocument(
                url="https://source.example/a",
                title="",
                text="The release says activity was unchanged.",
                tool_name="read",
            ),
            max_findings=2,
        )
        return findings == []
    ok = asyncio.run(run())
    return ok, "ungrounded quote must never enter Evidence Ledger"


def _probe_to_belief() -> tuple[bool, str]:
    async def run():
        db = _db()
        try:
            ledger, graph, falsification, _claim = _state()
            persist_ledger(db, ledger)
            persist_claim_graph(db, graph)
            persist_falsification_engine(db, falsification)
            gateway = _Gateway(
                [
                    ResearchDocument(
                        url="https://independent.example/report",
                        title="Independent report",
                        text="Independent evidence says the thesis is no longer intact.",
                        tool_name="read",
                    )
                ]
            )
            result = await AutomaticResearchLoop(
                graph=graph,
                ledger=ledger,
                falsification=falsification,
                gateway=gateway,
                extractor=_Extractor(
                    [
                        ExtractedFinding(
                            quote="the thesis is no longer intact",
                            relation=EvidenceRelation.CONTRADICTS,
                            observation_kind=ObservationKind.ACTUAL,
                            confidence=0.85,
                        )
                    ]
                ),
                max_probes=1,
                max_sources_per_probe=1,
            ).run(db=db, evaluated_at=T0 + timedelta(minutes=5))
            return (
                result.evidence_added == 1
                and result.beliefs_changed >= 1
                and gateway.search_calls == 1
            ), result
        finally:
            db.close()
    ok, result = asyncio.run(run())
    return ok, (
        f"evidence={result.evidence_added}, "
        f"belief_changes={result.beliefs_changed}, "
        f"tool_calls={result.tool_calls}"
    )


def _cooldown_suppresses_repeat() -> tuple[bool, str]:
    async def run():
        db = _db()
        try:
            ledger, graph, falsification, _claim = _state(
                FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT
            )
            persist_ledger(db, ledger)
            persist_claim_graph(db, graph)
            persist_falsification_engine(db, falsification)
            gateway = _Gateway(
                [
                    ResearchDocument(
                        url="https://independent.example/report",
                        title="",
                        text="Irrelevant document.",
                        tool_name="read",
                    )
                ]
            )
            loop = AutomaticResearchLoop(
                graph=graph,
                ledger=ledger,
                falsification=falsification,
                gateway=gateway,
                extractor=_Extractor([]),
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
            return (
                first.probes_executed == 1
                and second.probes_executed == 0
                and second.probes_skipped_cooldown >= 1
                and gateway.search_calls == 1
            ), (first, second)
        finally:
            db.close()
    ok, pair = asyncio.run(run())
    return ok, (
        f"first={pair[0].probes_executed}, "
        f"second={pair[1].probes_executed}, "
        f"cooldown={pair[1].probes_skipped_cooldown}"
    )


def _tool_budget() -> tuple[bool, str]:
    gateway = AhmedToolboxResearchGateway.__new__(
        AhmedToolboxResearchGateway
    )
    gateway.max_tool_calls = 1
    gateway.tool_calls = 0
    gateway._budget()
    try:
        gateway._budget()
    except RuntimeError:
        return True, "second call blocked at budget=1"
    return False, "budget failed to block second call"


def _no_claims_no_calls() -> tuple[bool, str]:
    async def run():
        db = _db()
        try:
            gateway = _Gateway([])
            result = await AutomaticResearchLoop(
                graph=ClaimGraph(),
                ledger=EvidenceLedger(),
                falsification=FalsificationEngine(),
                gateway=gateway,
                extractor=_Extractor([]),
            ).run(db=db, evaluated_at=T0)
            return result, gateway.search_calls
        finally:
            db.close()
    result, calls = asyncio.run(run())
    return (
        result.status == "no_claims"
        and result.tool_calls == 0
        and calls == 0
    ), f"status={result.status}, calls={calls}"


CASES: list[tuple[str, str, float, Callable[[], tuple[bool, str]]]] = [
    ("grounded_quote", "evidence_grounding", 20.0, _grounded_quote),
    ("hallucinated_quote_rejected", "evidence_grounding", 20.0, _hallucinated_quote_rejected),
    ("probe_to_evidence_to_belief", "closed_loop", 25.0, _probe_to_belief),
    ("cooldown_suppresses_repeat", "runaway_prevention", 15.0, _cooldown_suppresses_repeat),
    ("tool_budget_enforced", "runaway_prevention", 10.0, _tool_budget),
    ("no_claims_no_calls", "cost_safety", 10.0, _no_claims_no_calls),
]


def run_automatic_research_benchmark() -> dict:
    results: list[AutomaticResearchCase] = []
    for case_id, category, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            AutomaticResearchCase(
                case_id=case_id,
                category=category,
                passed=passed,
                points=points,
                earned=points if passed else 0.0,
                detail=detail,
            )
        )
    total = sum(item.points for item in results)
    earned = sum(item.earned for item in results)
    score = round((earned / total) * 100.0, 2) if total else 0.0
    return {
        "benchmark": "PanWatch Benchmark v1",
        "version": "1.4.0",
        "track": "automatic_research_loop",
        "score_percent": score,
        "passed_cases": sum(1 for item in results if item.passed),
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 90.0,
        "regression_gate_passed": score >= 90.0,
        "limitations": [
            "The deterministic track validates loop integrity, not web-search quality.",
            "Production live verification is performed separately against Ahmed ToolBox and Neon.",
            "Automatic claim extraction from arbitrary long-form documents is still outside this track.",
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Automatic Research benchmark")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    report = run_automatic_research_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return 0 if args.no_gate or report["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
