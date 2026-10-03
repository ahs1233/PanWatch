"""Persistent Belief State + PanWatch integration benchmark v1.3."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.modules.research import (
    BeliefEventType,
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


@dataclass(frozen=True)
class BeliefCaseResult:
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
    Session = sessionmaker(bind=engine)
    return Session()


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


def _basic():
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
    )
    falsification = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Soft landing fails above six percent unemployment",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        hard_fail=True,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
    )
    falsification.register_rule(rule, graph=graph)
    return ledger, graph, falsification, claim, source


def _persist_base(db, ledger, graph, falsification):
    persist_ledger(db, ledger)
    persist_claim_graph(db, graph)
    persist_falsification_engine(db, falsification)


def _restart_continuity() -> tuple[bool, str]:
    db = _db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic()
        _persist_base(db, ledger, graph, falsification)
        PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        ).run_cycle(db=db, evaluated_at=T0 + timedelta(hours=1))

        restarted_monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        second = restarted_monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=2),
        )
        history = load_belief_history(db, claim_id=claim.claim_id)
        ok = (
            len(history) == 2
            and second.changed_count == 0
            and history[0].input_fingerprint == history[1].input_fingerprint
        )
        return ok, f"history={len(history)}, changed={second.changed_count}"
    finally:
        db.close()


def _unchanged_cycle_no_noise() -> tuple[bool, str]:
    db = _db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic()
        _persist_base(db, ledger, graph, falsification)
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
        ok = second.events == () and second.changed_count == 0
        return ok, f"events={len(second.events)}, changed={second.changed_count}"
    finally:
        db.close()


def _confidence_drop_explained() -> tuple[bool, str]:
    db = _db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic()
        _persist_base(db, ledger, graph, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        first = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=1),
        )

        other = _source("counter", SourceTier.PRIMARY)
        ledger.register_source(other)
        contradiction = build_evidence(
            claim_key="macro.activity",
            source=other,
            statement="Activity deteriorated sharply",
            relation=EvidenceRelation.CONTRADICTS,
            event_time=T0 + timedelta(hours=2),
            recorded_at=T0 + timedelta(hours=2),
            confidence=1.0,
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
        types = {event.event_type for event in second.events}
        ok = (
            second.updates[0].snapshot.final_confidence
            < first.updates[0].snapshot.final_confidence
            and BeliefEventType.CONFIDENCE_LOWERED in types
            and BeliefEventType.EVIDENCE_CHANGED in types
        )
        return ok, f"events={sorted(item.value for item in types)}"
    finally:
        db.close()


def _falsification_transition() -> tuple[bool, str]:
    db = _db()
    try:
        ledger, graph, falsification, claim, source = _basic()
        _persist_base(db, ledger, graph, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        monitor.run_cycle(db=db, evaluated_at=T0 + timedelta(minutes=30))

        value = build_evidence(
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
        ledger.append(value)
        persist_ledger(db, ledger)
        cycle = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=2),
        )
        types = {event.event_type for event in cycle.events}
        ok = (
            cycle.updates[0].snapshot.final_status is ClaimStatus.FALSIFIED
            and BeliefEventType.FALSIFIED in types
        )
        return ok, f"status={cycle.updates[0].snapshot.final_status.value}, events={sorted(i.value for i in types)}"
    finally:
        db.close()


def _revision_recovery() -> tuple[bool, str]:
    db = _db()
    try:
        ledger, graph, falsification, claim, source = _basic()
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
        _persist_base(db, ledger, graph, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        first = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=2),
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
        types = {event.event_type for event in second.events}
        ok = (
            first.updates[0].snapshot.final_status is ClaimStatus.FALSIFIED
            and second.updates[0].snapshot.final_status is not ClaimStatus.FALSIFIED
            and BeliefEventType.RECOVERED in types
        )
        return ok, f"before={first.updates[0].snapshot.final_status.value}, after={second.updates[0].snapshot.final_status.value}"
    finally:
        db.close()


def _no_lookahead_rewrite() -> tuple[bool, str]:
    ledger, graph, falsification, claim, source = _basic()
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
    ledger.append(original)
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
    ok = (
        early.final_status is ClaimStatus.FALSIFIED
        and late.final_status is not ClaimStatus.FALSIFIED
    )
    return ok, f"early={early.final_status.value}, late={late.final_status.value}"


def _required_dependency_propagation() -> tuple[bool, str]:
    db = _db()
    try:
        ledger = EvidenceLedger()
        source = _source("official", SourceTier.OFFICIAL_PRIMARY)
        ledger.register_source(source)
        premise_support = build_evidence(
            claim_key="premise",
            source=source,
            statement="Premise supported",
            event_time=T0,
            recorded_at=T0,
            confidence=0.95,
        )
        breaker = build_evidence(
            claim_key="premise.breaker",
            source=source,
            statement="Breaker value is 1",
            event_time=T0 + timedelta(hours=2),
            recorded_at=T0 + timedelta(hours=2),
            numeric_value=1.0,
            unit="flag",
        )
        ledger.append(premise_support)
        ledger.append(breaker)

        graph = ClaimGraph()
        premise = build_claim(
            claim_key="premise",
            statement="Required premise",
            kind=ClaimKind.HYPOTHESIS,
            created_at=T0,
        )
        conclusion = build_claim(
            claim_key="conclusion",
            statement="Conclusion requires premise",
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
            description="Premise fails when breaker exceeds zero",
            rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
            hard_fail=True,
            evidence_claim_key="premise.breaker",
            operator=">",
            threshold=0.0,
        )
        conclusion_rule = build_falsification_rule(
            claim_id=conclusion.claim_id,
            description="Conclusion requires premise",
            rule_type=FalsificationRuleType.DEPENDENCY_FAILURE,
            related_claim_id=premise.claim_id,
            threshold=0.35,
        )
        falsification.register_rule(premise_rule, graph=graph)
        falsification.register_rule(conclusion_rule, graph=graph)

        _persist_base(db, ledger, graph, falsification)
        cycle = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        ).run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=3),
        )
        conclusion_update = next(
            item for item in cycle.updates
            if item.snapshot.claim_id == conclusion.claim_id
        )
        ok = (
            conclusion_update.snapshot.final_status is ClaimStatus.FALSIFIED
            and premise.claim_id in conclusion_update.snapshot.dependency_failures
        )
        return ok, f"status={conclusion_update.snapshot.final_status.value}, failures={conclusion_update.snapshot.dependency_failures}"
    finally:
        db.close()


def _cycle_persistence_idempotency() -> tuple[bool, str]:
    db = _db()
    try:
        ledger, graph, falsification, claim, _source_obj = _basic()
        _persist_base(db, ledger, graph, falsification)
        monitor = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        )
        first = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=1),
        )
        second = monitor.run_cycle(
            db=db,
            evaluated_at=T0 + timedelta(hours=1),
        )
        history = load_belief_history(db, claim_id=claim.claim_id)
        events = load_belief_events(db, claim_id=claim.claim_id)
        ok = (
            first.cycle_id == second.cycle_id
            and len(history) == 1
            and len(events) == 1
        )
        return ok, f"history={len(history)}, events={len(events)}"
    finally:
        db.close()


CASES: list[tuple[str, str, float, Callable[[], tuple[bool, str]]]] = [
    ("restart_continuity", "persistence", 15.0, _restart_continuity),
    ("unchanged_cycle_no_noise", "change_detection", 10.0, _unchanged_cycle_no_noise),
    ("confidence_drop_explained", "change_explanation", 15.0, _confidence_drop_explained),
    ("falsification_transition", "belief_transition", 15.0, _falsification_transition),
    ("revision_recovery", "belief_transition", 15.0, _revision_recovery),
    ("no_lookahead_rewrite", "temporal_integrity", 15.0, _no_lookahead_rewrite),
    ("required_dependency_propagation", "panwatch_integration", 10.0, _required_dependency_propagation),
    ("cycle_persistence_idempotency", "persistence", 5.0, _cycle_persistence_idempotency),
]


def run_belief_benchmark() -> dict:
    results: list[BeliefCaseResult] = []
    for case_id, category, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            BeliefCaseResult(
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
        "version": "1.7.0",
        "track": "persistent_belief_state",
        "score_percent": score,
        "passed_cases": sum(1 for item in results if item.passed),
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 90.0,
        "regression_gate_passed": score >= 90.0,
        "limitations": [
            "Belief cycles operate on registered claims; generalized automatic claim extraction from arbitrary documents remains outside this track.",
            "Live counter-research dispatch is scored separately in the Automatic Research Loop track.",
            "The benchmark uses deterministic fixtures and does not prove external research superiority."
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Persistent Belief State benchmark")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    report = run_belief_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if args.no_gate or report["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
