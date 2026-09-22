"""General Claim Acquisition benchmark track v1.5."""

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

from src.modules.research.automatic_research import ResearchDocument
from src.modules.research.claim_acquisition import (
    ClaimCandidate,
    GeneralClaimAcquisition,
    GroundedClaimExtractor,
)
from src.modules.research.claim_graph import ClaimGraph, ClaimKind
from src.modules.research.evidence import ObservationKind
from src.modules.research.falsification import FalsificationEngine
from src.modules.research.ledger import EvidenceLedger
from src.platform.persistence.migrations import (
    _m127_research_evidence_foundation,
    _m128_claim_graph_and_falsification,
    _m129_persistent_belief_state,
    _m130_automatic_research_loop,
    _m131_general_claim_acquisition,
    _m132_semantic_claim_resolution,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class AcquisitionCase:
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
        _m131_general_claim_acquisition(conn)
        _m132_semantic_claim_resolution(conn)
    Session = sessionmaker(bind=engine)
    return Session()


def _doc(url: str, text: str):
    return ResearchDocument(
        url=url,
        title="Source",
        text=text,
        tool_name="fixture",
        publisher="Fixture",
        published_at=T0,
    )


def _candidate(
    quote: str,
    statement: str,
    key: str,
    *,
    kind: ClaimKind = ClaimKind.FACT,
    observation: ObservationKind = ObservationKind.ACTUAL,
    testable: bool = True,
    supersedes: bool = False,
):
    return ClaimCandidate(
        quote=quote,
        statement=statement,
        proposed_claim_key=key,
        kind=kind,
        observation_kind=observation,
        confidence=0.8,
        testable=testable,
        valid_from=T0,
        supersedes_previous=supersedes,
    )


class _Extractor:
    def __init__(self, mapping):
        self.mapping = mapping

    async def extract(self, *, document, topic_hint, max_candidates):
        return list(self.mapping.get(document.url, []))[:max_candidates]


class _AI:
    def __init__(self, payload: str):
        self.payload = payload

    async def chat_multi(self, *_args, **_kwargs):
        return self.payload


def _quote_grounding() -> tuple[bool, str]:
    async def run():
        good = GroundedClaimExtractor(
            _AI(
                '{"claims":[{"quote":"revenue increased by 9 percent",'
                '"statement":"Revenue increased by 9 percent.",'
                '"claim_key":"company.revenue.growth","kind":"fact",'
                '"observation_kind":"actual","confidence":0.8,"testable":true}]}'
            )
        )
        bad = GroundedClaimExtractor(
            _AI(
                '{"claims":[{"quote":"profits tripled instantly",'
                '"statement":"Profits tripled instantly.",'
                '"claim_key":"company.profits","kind":"fact",'
                '"observation_kind":"actual","confidence":0.9,"testable":true}]}'
            )
        )
        doc = _doc(
            "https://company.example/report",
            "The company said revenue increased by 9 percent during the quarter.",
        )
        good_rows = await good.extract(
            document=doc,
            topic_hint="revenue",
            max_candidates=3,
        )
        bad_rows = await bad.extract(
            document=doc,
            topic_hint="profits",
            max_candidates=3,
        )
        source = doc.text.casefold()
        hallucination_blocked = all(
            "profits tripled instantly" not in row.statement.casefold()
            and row.quote.casefold() in source
            for row in bad_rows
        )
        return len(good_rows) == 1 and hallucination_blocked
    ok = asyncio.run(run())
    return ok, (
        "exact quote admitted; hallucinated claim rejected; "
        "any fallback remains verbatim source-grounded"
    )


def _duplicate_reuse() -> tuple[bool, str]:
    async def run():
        db = _db()
        try:
            statement = "Revenue was 10 billion dollars in 2025."
            d1 = _doc("https://a.example/report", "The report says revenue was 10 billion dollars in 2025.")
            d2 = _doc("https://b.example/report", "The audit confirms revenue was 10 billion dollars in 2025.")
            c1 = _candidate("revenue was 10 billion dollars in 2025", statement, "company.revenue.2025")
            c2 = _candidate("revenue was 10 billion dollars in 2025", statement, "company.revenue.2025")
            graph = ClaimGraph()
            ledger = EvidenceLedger()
            result = await GeneralClaimAcquisition(
                graph=graph,
                ledger=ledger,
                falsification=FalsificationEngine(),
                extractor=_Extractor({d1.url:[c1], d2.url:[c2]}),
            ).run(db=db, documents=[d1,d2])
            return len(graph.claims) == 1 and len(ledger.records) == 2 and result.duplicates == 1
        finally:
            db.close()
    ok = asyncio.run(run())
    return ok, "two sources strengthen one claim instead of creating duplicate claims"


def _revision_semantics() -> tuple[bool, str]:
    async def run():
        db = _db()
        try:
            d1 = _doc("https://stats.example/a", "The agency reported unemployment at 6.2 percent.")
            d2 = _doc("https://stats.example/b", "The agency revised unemployment to 5.8 percent.")
            c1 = _candidate("unemployment at 6.2 percent", "Unemployment was 6.2 percent.", "macro.unemployment.current")
            c2 = _candidate(
                "revised unemployment to 5.8 percent",
                "Unemployment was 5.8 percent.",
                "macro.unemployment.current",
                observation=ObservationKind.REVISION,
                supersedes=True,
            )
            graph=ClaimGraph()
            engine=GeneralClaimAcquisition(
                graph=graph,
                ledger=EvidenceLedger(),
                falsification=FalsificationEngine(),
                extractor=_Extractor({d1.url:[c1],d2.url:[c2]}),
            )
            await engine.run(db=db, documents=[d1], evaluated_at=T0)
            second=await engine.run(db=db, documents=[d2], evaluated_at=T0+timedelta(hours=1))
            newest=max(graph.claims,key=lambda x:x.created_at)
            return second.superseded == 1 and newest.supersedes is not None
        finally:
            db.close()
    ok=asyncio.run(run())
    return ok, "explicit revision cue creates a superseding claim version"


def _false_supersession_blocked() -> tuple[bool, str]:
    async def run():
        db=_db()
        try:
            d1=_doc("https://lab.example/a","The first trial found efficacy of 61 percent.")
            d2=_doc("https://lab.example/b","A second trial found efficacy of 58 percent.")
            c1=_candidate("efficacy of 61 percent","Trial efficacy was 61 percent.","science.trial.efficacy")
            c2=_candidate("efficacy of 58 percent","Trial efficacy was 58 percent.","science.trial.efficacy",supersedes=True)
            graph=ClaimGraph()
            engine=GeneralClaimAcquisition(
                graph=graph,
                ledger=EvidenceLedger(),
                falsification=FalsificationEngine(),
                extractor=_Extractor({d1.url:[c1],d2.url:[c2]}),
            )
            await engine.run(db=db,documents=[d1],evaluated_at=T0)
            second=await engine.run(db=db,documents=[d2],evaluated_at=T0+timedelta(hours=1))
            newest=max(graph.claims,key=lambda x:x.created_at)
            return second.superseded == 0 and newest.supersedes is None and newest.metadata.get("supersede_denied") is True
        finally:
            db.close()
    ok=asyncio.run(run())
    return ok, "same semantic key without revision cue cannot rewrite history"


def _domain_neutral() -> tuple[bool, str]:
    async def run():
        db=_db()
        try:
            docs=[
                _doc("https://co.example/a","Management forecast revenue growth of 8 percent next year."),
                _doc("https://lab.example/a","The catalyst reduced reaction time by 12 percent."),
                _doc("https://gov.example/a","The ministry announced the tariff will remain at 5 percent."),
            ]
            mapping={
                docs[0].url:[_candidate("forecast revenue growth of 8 percent next year","Management forecasts revenue growth of 8 percent next year.","company.revenue_growth.next_year",kind=ClaimKind.FORECAST,observation=ObservationKind.GUIDANCE)],
                docs[1].url:[_candidate("catalyst reduced reaction time by 12 percent","The catalyst reduced reaction time by 12 percent.","science.catalyst.reaction_time")],
                docs[2].url:[_candidate("tariff will remain at 5 percent","The tariff will remain at 5 percent.","policy.tariff.current",observation=ObservationKind.POLICY_STATEMENT)],
            }
            graph=ClaimGraph()
            result=await GeneralClaimAcquisition(
                graph=graph,
                ledger=EvidenceLedger(),
                falsification=FalsificationEngine(),
                extractor=_Extractor(mapping),
            ).run(db=db,documents=docs,seed_topic="mixed domains")
            return result.claims_accepted == 3 and len({c.claim_key.split(".")[0] for c in graph.claims}) == 3
        finally:
            db.close()
    ok=asyncio.run(run())
    return ok, "same engine admits company, science, and policy claims"


CASES: list[tuple[str,str,float,Callable[[],tuple[bool,str]]]] = [
    ("quote_grounding","grounding",25.0,_quote_grounding),
    ("duplicate_reuse","deduplication",20.0,_duplicate_reuse),
    ("revision_semantics","versioning",20.0,_revision_semantics),
    ("false_supersession_blocked","versioning",20.0,_false_supersession_blocked),
    ("domain_neutral","generality",15.0,_domain_neutral),
]


def run_claim_acquisition_benchmark() -> dict:
    results=[]
    for case_id,category,points,evaluator in CASES:
        try:
            passed,detail=evaluator()
        except Exception as exc:
            passed,detail=False,f"{type(exc).__name__}: {exc}"
        results.append(AcquisitionCase(case_id,category,passed,points,points if passed else 0.0,detail))
    total=sum(r.points for r in results)
    earned=sum(r.earned for r in results)
    score=round(earned/total*100.0,2) if total else 0.0
    return {
        "benchmark":"PanWatch Benchmark v1",
        "version":"1.6.0",
        "track":"general_claim_acquisition",
        "score_percent":score,
        "passed_cases":sum(1 for r in results if r.passed),
        "total_cases":len(results),
        "points_earned":earned,
        "points_total":total,
        "regression_gate_percent":90.0,
        "regression_gate_passed":score>=90.0,
        "limitations":[
            "The deterministic track measures admission integrity, not extraction recall over arbitrary corpora.",
            "Live search/provider quality is verified separately in production.",
            "Cross-document semantic clustering beyond exact statement/key versioning remains conservative by design."
        ],
        "cases":[asdict(r) for r in results],
    }


def main() -> int:
    parser=argparse.ArgumentParser(description="Run General Claim Acquisition benchmark")
    parser.add_argument("--json",dest="json_path")
    parser.add_argument("--no-gate",action="store_true")
    args=parser.parse_args()
    report=run_claim_acquisition_benchmark()
    print(json.dumps(report,indent=2,sort_keys=True))
    if args.json_path:
        path=Path(args.json_path); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return 0 if args.no_gate or report["regression_gate_passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
