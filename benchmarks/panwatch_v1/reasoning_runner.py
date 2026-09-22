"""Deterministic Claim Graph + Falsification benchmark for PanWatch v1.2."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

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


UTC = timezone.utc
T0 = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class ReasoningCaseResult:
    case_id: str
    category: str
    passed: bool
    points: float
    earned: float
    detail: str


def _source(slug: str, tier: SourceTier = SourceTier.PRIMARY):
    return build_source(
        url=f"https://{slug}.example.com/report",
        content=f"{slug} benchmark content",
        publisher=slug,
        source_tier=tier,
        source_family=slug,
        published_at=T0,
        retrieved_at=T0 + timedelta(minutes=1),
    )


def _multi_hop_support_propagation() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    source = _source("official", SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    evidence = build_evidence(
        claim_key="macro.inflation.cooling",
        source=source,
        statement="Inflation cooled materially",
        confidence=0.95,
        event_time=T0,
    )
    ledger.append(evidence)

    graph = ClaimGraph()
    premise = build_claim(
        claim_key="macro.inflation.cooling",
        statement="Inflation is cooling",
        kind=ClaimKind.FACT,
        created_at=T0,
    )
    conclusion = build_claim(
        claim_key="rates.pressure",
        statement="Rate pressure is easing",
        kind=ClaimKind.CONCLUSION,
        created_at=T0,
    )
    graph.register_claim(premise)
    graph.register_claim(conclusion)
    graph.link_evidence(
        claim_id=premise.claim_id,
        evidence_id=evidence.evidence_id,
        ledger=ledger,
    )
    graph.connect(
        premise.claim_id,
        conclusion.claim_id,
        ClaimRelation.SUPPORTS,
    )
    a = graph.assess(premise.claim_id, ledger)
    b = graph.assess(conclusion.claim_id, ledger)
    ok = (
        a.status is ClaimStatus.SUPPORTED
        and b.status is ClaimStatus.SUPPORTED
        and b.confidence > 0.70
    )
    return ok, f"premise={a.confidence:.4f}, conclusion={b.confidence:.4f}"


def _required_dependency_collapse() -> tuple[bool, str]:
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
        statement="Conclusion depends on critical premise",
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
    result = graph.assess(conclusion.claim_id, ledger)
    ok = (
        premise.claim_id in result.dependency_failures
        and result.confidence < 0.35
    )
    return ok, f"confidence={result.confidence:.4f}, failures={result.dependency_failures}"


def _hard_numeric_falsifier() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    source = _source("official", SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    support = build_evidence(
        claim_key="macro.activity",
        source=source,
        statement="Activity remains resilient",
        confidence=0.95,
        event_time=T0,
    )
    unemployment = build_evidence(
        claim_key="macro.unemployment",
        source=source,
        statement="Unemployment is 6.2 percent",
        numeric_value=6.2,
        unit="%",
        period="2026-08",
        event_time=T0,
    )
    ledger.append(support)
    ledger.append(unemployment)
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="soft_landing",
        statement="Soft landing remains intact",
        kind=ClaimKind.CONCLUSION,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=support.evidence_id,
        ledger=ledger,
    )
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Soft landing fails above six percent unemployment",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        hard_fail=True,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(claim.claim_id, graph=graph, ledger=ledger, as_of=T0 + timedelta(hours=1))
    return (
        report.final_status is ClaimStatus.FALSIFIED,
        f"status={report.final_status.value}, confidence={report.adjusted_confidence:.4f}",
    )


def _counterclaim_falsifier() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    source = _source("official", SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    evidence = build_evidence(
        claim_key="hard_landing",
        source=source,
        statement="Hard-landing evidence strengthened",
        confidence=1.0,
        event_time=T0,
    )
    ledger.append(evidence)
    graph = ClaimGraph()
    target = build_claim(
        claim_key="soft_landing",
        statement="Soft landing remains intact",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    counter = build_claim(
        claim_key="hard_landing",
        statement="Hard landing is underway",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    graph.register_claim(target)
    graph.register_claim(counter)
    graph.link_evidence(
        claim_id=counter.claim_id,
        evidence_id=evidence.evidence_id,
        ledger=ledger,
    )
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=target.claim_id,
        description="Soft landing fails if hard-landing thesis exceeds confidence threshold",
        rule_type=FalsificationRuleType.COUNTERCLAIM_CONFIDENCE,
        hard_fail=True,
        related_claim_id=counter.claim_id,
        threshold=0.70,
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(target.claim_id, graph=graph, ledger=ledger)
    return (
        report.final_status is ClaimStatus.FALSIFIED,
        f"triggered={report.triggered_rules}",
    )


def _reasoning_cycle_rejected() -> tuple[bool, str]:
    graph = ClaimGraph()
    a = build_claim(claim_key="a", statement="A", created_at=T0)
    b = build_claim(claim_key="b", statement="B", created_at=T0)
    graph.register_claim(a)
    graph.register_claim(b)
    graph.connect(a.claim_id, b.claim_id, ClaimRelation.SUPPORTS)
    try:
        graph.connect(b.claim_id, a.claim_id, ClaimRelation.DEPENDS_ON)
    except ValueError:
        return True, "cycle rejected"
    return False, "cycle accepted unexpectedly"


def _untestable_rule_generates_probe() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis",
        statement="A testable thesis",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    graph.register_claim(claim)
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Fails if unemployment exceeds six percent",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(claim.claim_id, graph=graph, ledger=ledger)
    ok = (
        report.results[0].state is FalsificationState.UNTESTABLE
        and report.coverage == 0.0
        and len(report.probes) == 1
    )
    return ok, f"coverage={report.coverage:.2f}, probes={len(report.probes)}"


def _unfalsifiable_supported_claim_guard() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    source = _source("official", SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(source)
    evidence = build_evidence(
        claim_key="thesis",
        source=source,
        statement="Evidence supports thesis",
        confidence=1.0,
        event_time=T0,
    )
    ledger.append(evidence)
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="thesis",
        statement="A hypothesis with no failure condition",
        kind=ClaimKind.HYPOTHESIS,
        created_at=T0,
    )
    graph.register_claim(claim)
    graph.link_evidence(
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        ledger=ledger,
    )
    before = graph.assess(claim.claim_id, ledger)
    report = FalsificationEngine().evaluate(
        claim.claim_id,
        graph=graph,
        ledger=ledger,
    )
    ok = (
        before.status is ClaimStatus.SUPPORTED
        and report.final_status is ClaimStatus.OPEN
        and report.probes
    )
    return ok, f"before={before.status.value}, after={report.final_status.value}"


def _numeric_conflict_blocks_false_falsification() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    official = _source("official", SourceTier.OFFICIAL_PRIMARY)
    other = _source("other", SourceTier.PRIMARY)
    ledger.register_source(official)
    ledger.register_source(other)
    ledger.append(
        build_evidence(
            claim_key="macro.unemployment",
            source=official,
            statement="Unemployment 6.2 percent",
            numeric_value=6.2,
            unit="%",
            period="2026-08",
            event_time=T0,
        )
    )
    ledger.append(
        build_evidence(
            claim_key="macro.unemployment",
            source=other,
            statement="Unemployment 5.8 percent",
            numeric_value=5.8,
            unit="%",
            period="2026-08",
            event_time=T0,
        )
    )
    graph = ClaimGraph()
    claim = build_claim(
        claim_key="soft_landing",
        statement="Soft landing remains intact",
        kind=ClaimKind.CONCLUSION,
        created_at=T0,
    )
    graph.register_claim(claim)
    engine = FalsificationEngine()
    rule = build_falsification_rule(
        claim_id=claim.claim_id,
        description="Fails if unemployment exceeds six percent",
        rule_type=FalsificationRuleType.NUMERIC_THRESHOLD,
        hard_fail=True,
        evidence_claim_key="macro.unemployment",
        operator=">",
        threshold=6.0,
    )
    engine.register_rule(rule, graph=graph)
    report = engine.evaluate(claim.claim_id, graph=graph, ledger=ledger)
    ok = (
        report.results[0].state is FalsificationState.UNTESTABLE
        and report.final_status is not ClaimStatus.FALSIFIED
    )
    return ok, f"rule_state={report.results[0].state.value}, final={report.final_status.value}"


CASES: list[tuple[str, str, float, Callable[[], tuple[bool, str]]]] = [
    ("multi_hop_support_propagation", "claim_graph", 15.0, _multi_hop_support_propagation),
    ("required_dependency_collapse", "claim_graph", 15.0, _required_dependency_collapse),
    ("hard_numeric_falsifier", "falsification", 15.0, _hard_numeric_falsifier),
    ("counterclaim_falsifier", "falsification", 15.0, _counterclaim_falsifier),
    ("reasoning_cycle_rejected", "graph_integrity", 10.0, _reasoning_cycle_rejected),
    ("untestable_rule_generates_probe", "research_gap", 10.0, _untestable_rule_generates_probe),
    ("unfalsifiable_supported_claim_guard", "falsifiability", 10.0, _unfalsifiable_supported_claim_guard),
    ("numeric_conflict_blocks_false_falsification", "conflict_handling", 10.0, _numeric_conflict_blocks_false_falsification),
]


def run_reasoning_benchmark() -> dict:
    results: list[ReasoningCaseResult] = []
    for case_id, category, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            ReasoningCaseResult(
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
    score = round(earned / total * 100.0, 2) if total else 0.0
    return {
        "benchmark": "PanWatch Benchmark v1",
        "version": "1.6.0",
        "track": "claim_graph_falsification",
        "score_percent": score,
        "passed_cases": sum(1 for item in results if item.passed),
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 90.0,
        "regression_gate_passed": score >= 90.0,
        "limitations": [
            "This track measures deterministic reasoning integrity, not open-ended intelligence.",
            "Probe execution is scored separately in the Automatic Research Loop track.",
            "Persistent belief-state history is scored separately in the Persistent Belief State track.",
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Claim Graph + Falsification benchmark")
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    report = run_reasoning_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if args.no_gate or report["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
