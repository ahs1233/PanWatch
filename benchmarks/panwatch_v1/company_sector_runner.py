"""Company / sector comparative research benchmark.

This track tests whether the existing Research Engine can preserve company-level
research semantics across primary/secondary sources, guidance, rumors,
restatements, KPI definitions, periods, conflicts and thesis updates.

It intentionally includes one entity-isolation trap: two different companies
with nearly identical KPI wording must NOT be merged into one claim.
"""

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
    EvidenceLedger,
    EvidenceRelation,
    FalsificationEngine,
    FalsificationRuleType,
    ObservationKind,
    SourceTier,
    build_claim,
    build_evidence,
    build_falsification_rule,
    build_source,
)
from src.modules.research.claim_resolution import (
    ConservativeClaimResolver,
    SemanticClaimRelation,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class CompanySectorCase:
    case_id: str
    category: str
    passed: bool
    points: float
    earned: float
    detail: str


def _source(
    slug: str,
    *,
    tier: SourceTier = SourceTier.SECONDARY,
    content: str | None = None,
    upstream_origin: str = "",
    published_at: datetime = T0,
    family: str | None = None,
):
    body = content or f"company benchmark source {slug}"
    return build_source(
        url=f"https://{slug}.example.com/report",
        content=body,
        publisher=slug,
        title=f"{slug} report",
        source_tier=tier,
        source_family=family or slug,
        upstream_origin=upstream_origin,
        published_at=published_at,
        retrieved_at=published_at + timedelta(minutes=2),
        observed_at=published_at + timedelta(minutes=2),
        tool_name="company-sector-benchmark",
    )


def _earnings_release_vs_secondary_summary() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    primary = _source("issuer-ir", tier=SourceTier.OFFICIAL_PRIMARY)
    secondary = _source("news-summary", tier=SourceTier.SECONDARY)
    ledger.register_source(primary)
    ledger.register_source(secondary)

    for source, confidence in ((secondary, 0.75), (primary, 0.95)):
        ledger.append(
            build_evidence(
                claim_key="alpha.revenue.2026-q2",
                source=source,
                statement="Alpha Q2 revenue was 12.4 billion dollars.",
                observation_kind=ObservationKind.ACTUAL,
                event_time=T0,
                numeric_value=12.4,
                unit="USD bn",
                period="2026-Q2",
                confidence=confidence,
            )
        )

    resolved = ledger.resolve_numeric("alpha.revenue.2026-q2")
    selected = next(
        row
        for row in ledger.records
        if row.evidence_id == resolved.selected_evidence_id
    )
    selected_source = ledger.source_for(selected)
    ok = (
        resolved.status == "resolved"
        and resolved.value == 12.4
        and selected_source.source_tier is SourceTier.OFFICIAL_PRIMARY
        and resolved.independent_source_count == 2
    )
    return (
        ok,
        f"status={resolved.status}, selected_tier={selected_source.source_tier.value}, "
        f"independent={resolved.independent_source_count}",
    )


def _guidance_vs_consensus() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    issuer = _source("issuer-guidance", tier=SourceTier.OFFICIAL_PRIMARY)
    consensus = _source("street-consensus", tier=SourceTier.SECONDARY)
    ledger.register_source(issuer)
    ledger.register_source(consensus)

    ledger.append(
        build_evidence(
            claim_key="alpha.revenue.next-quarter",
            source=issuer,
            statement="Management guides revenue to 13.0 billion dollars.",
            observation_kind=ObservationKind.GUIDANCE,
            event_time=T0,
            numeric_value=13.0,
            unit="USD bn",
            period="2026-Q3",
        )
    )
    ledger.append(
        build_evidence(
            claim_key="alpha.revenue.next-quarter",
            source=consensus,
            statement="Consensus forecasts revenue of 13.5 billion dollars.",
            observation_kind=ObservationKind.FORECAST,
            event_time=T0,
            numeric_value=13.5,
            unit="USD bn",
            period="2026-Q3",
        )
    )

    separated = ledger.kind_separation_guard(
        "alpha.revenue.next-quarter",
        ObservationKind.GUIDANCE,
        ObservationKind.FORECAST,
    )
    actual_guard = ledger.actual_forecast_guard(
        "alpha.revenue.next-quarter"
    )
    ok = (
        separated.passed
        and actual_guard.passed is False
        and actual_guard.code == "forecast_only"
    )
    return ok, f"separation={separated.code}, actual_guard={actual_guard.code}"


def _duplicate_press_release_independence() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    content = "Alpha announced a 2 billion dollar repurchase authorization."
    for idx in range(3):
        source = _source(
            f"syndicated-pr-{idx}",
            content=content,
            upstream_origin="press-release:alpha:buyback-2026-09",
        )
        ledger.register_source(source)
        ledger.append(
            build_evidence(
                claim_key="alpha.buyback.authorization",
                source=source,
                statement=content,
                observation_kind=ObservationKind.ACTUAL,
                event_time=T0,
                confidence=0.85,
            )
        )

    independent = ledger.independent_source_count(
        "alpha.buyback.authorization"
    )
    ratio = ledger.source_independence_ratio(
        "alpha.buyback.authorization"
    )
    ok = independent == 1 and round(ratio, 4) == round(1 / 3, 4)
    return ok, f"independent={independent}, ratio={ratio:.4f}"


def _restatement_correction() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    filing = _source("alpha-filing", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(filing)

    original = build_evidence(
        claim_key="alpha.revenue.2025",
        source=filing,
        statement="Alpha reported 2025 revenue of 40.0 billion dollars.",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        numeric_value=40.0,
        unit="USD bn",
        period="2025",
    )
    ledger.append(original)
    revision = build_evidence(
        claim_key="alpha.revenue.2025",
        source=filing,
        statement="Alpha restated 2025 revenue to 39.2 billion dollars.",
        observation_kind=ObservationKind.REVISION,
        event_time=T0 + timedelta(days=30),
        numeric_value=39.2,
        unit="USD bn",
        period="2025",
        revision_of=original.evidence_id,
        supersedes=original.evidence_id,
    )
    ledger.append(revision)

    resolved = ledger.resolve_numeric("alpha.revenue.2025")
    ok = (
        resolved.status == "resolved_revision"
        and resolved.value == 39.2
        and resolved.selected_evidence_id == revision.evidence_id
    )
    return ok, f"status={resolved.status}, value={resolved.value}"


def _filing_vs_acquisition_rumor() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    filing = _source("alpha-8k", tier=SourceTier.OFFICIAL_PRIMARY)
    rumor = _source("social-rumor", tier=SourceTier.SOCIAL)
    ledger.register_source(filing)
    ledger.register_source(rumor)

    ledger.append(
        build_evidence(
            claim_key="alpha.acquisition.agreement",
            source=filing,
            statement="The filing states no definitive acquisition agreement has been signed.",
            relation=EvidenceRelation.SUPPORTS,
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0,
            confidence=0.95,
        )
    )
    ledger.append(
        build_evidence(
            claim_key="alpha.acquisition.agreement",
            source=rumor,
            statement="A social account claims an acquisition agreement was signed.",
            relation=EvidenceRelation.CONTRADICTS,
            observation_kind=ObservationKind.OPINION,
            event_time=T0 + timedelta(minutes=10),
            confidence=0.70,
        )
    )

    separated = ledger.kind_separation_guard(
        "alpha.acquisition.agreement",
        ObservationKind.ACTUAL,
        ObservationKind.OPINION,
    )
    confidence = ledger.claim_confidence("alpha.acquisition.agreement")
    ok = separated.passed and confidence > 0.50
    return ok, f"separation={separated.code}, confidence={confidence:.4f}"


def _cross_company_kpi_definition() -> tuple[bool, str]:
    graph = ClaimGraph()
    alpha = build_claim(
        claim_key="alpha.adjusted_ebitda_margin.2025",
        statement="Alpha adjusted EBITDA margin was 31 percent in 2025.",
        kind=ClaimKind.FACT,
        created_at=T0,
    )
    graph.register_claim(alpha)

    resolution = ConservativeClaimResolver().resolve(
        statement="Beta adjusted EBITDA margin was 31 percent in 2025.",
        claim_key="beta.adjusted_ebitda_margin.2025",
        graph=graph,
    )
    ok = resolution.relation is SemanticClaimRelation.DISTINCT
    return (
        ok,
        f"relation={resolution.relation.value}, score={resolution.score}, "
        f"matched={resolution.matched_claim_id}",
    )


def _period_mismatch_detection() -> tuple[bool, str]:
    graph = ClaimGraph()
    prior = build_claim(
        claim_key="alpha.revenue",
        statement="Alpha revenue was 40 billion dollars in 2025.",
        kind=ClaimKind.FACT,
        created_at=T0,
    )
    graph.register_claim(prior)

    resolution = ConservativeClaimResolver().resolve(
        statement="Alpha revenue was 44 billion dollars in 2026.",
        claim_key="alpha.revenue",
        graph=graph,
    )
    ok = (
        resolution.relation is SemanticClaimRelation.DISTINCT
        and resolution.period_match is False
        and resolution.reason == "different_explicit_period"
    )
    return (
        ok,
        f"relation={resolution.relation.value}, period_match={resolution.period_match}, "
        f"reason={resolution.reason}",
    )


def _contradictory_revenue_numbers() -> tuple[bool, str]:
    graph = ClaimGraph()
    prior = build_claim(
        claim_key="alpha.revenue.2025",
        statement="Alpha revenue was 40.0 billion dollars in 2025.",
        kind=ClaimKind.FACT,
        created_at=T0,
    )
    graph.register_claim(prior)

    resolution = ConservativeClaimResolver().resolve(
        statement="Alpha revenue was 39.2 billion dollars in 2025.",
        claim_key="alpha.revenue.2025",
        graph=graph,
    )
    ok = (
        resolution.relation is SemanticClaimRelation.CONTRADICTION
        and resolution.numeric_match is False
        and resolution.period_match is True
    )
    return (
        ok,
        f"relation={resolution.relation.value}, numeric_match={resolution.numeric_match}, "
        f"period_match={resolution.period_match}",
    )


def _source_recency_precedence() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    older = _source(
        "alpha-filing-old",
        tier=SourceTier.OFFICIAL_PRIMARY,
        upstream_origin="alpha:revenue-current",
        published_at=T0,
    )
    newer = _source(
        "alpha-filing-new",
        tier=SourceTier.OFFICIAL_PRIMARY,
        upstream_origin="alpha:revenue-current",
        published_at=T0 + timedelta(days=7),
    )
    ledger.register_source(older)
    ledger.register_source(newer)

    ledger.append(
        build_evidence(
            claim_key="alpha.revenue.current",
            source=older,
            statement="Alpha revenue was 40.0 billion dollars.",
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0,
            numeric_value=40.0,
            unit="USD bn",
            period="TTM",
            confidence=0.9,
        )
    )
    ledger.append(
        build_evidence(
            claim_key="alpha.revenue.current",
            source=newer,
            statement="Alpha revenue was 41.0 billion dollars.",
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0 + timedelta(days=7),
            numeric_value=41.0,
            unit="USD bn",
            period="TTM",
            confidence=0.9,
        )
    )

    resolved = ledger.resolve_numeric("alpha.revenue.current")
    ok = (
        resolved.status == "resolved"
        and resolved.value == 41.0
        and resolved.independent_source_count == 1
    )
    return (
        ok,
        f"status={resolved.status}, value={resolved.value}, "
        f"independent={resolved.independent_source_count}",
    )


def _thesis_update_after_new_filing() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    graph = ClaimGraph()
    falsification = FalsificationEngine()

    thesis = build_claim(
        claim_key="alpha.margin.expansion.thesis",
        statement="Alpha margins are expanding.",
        kind=ClaimKind.HYPOTHESIS,
        prior_confidence=0.55,
        created_at=T0,
    )
    graph.register_claim(thesis)

    support_source = _source(
        "alpha-q1-filing",
        tier=SourceTier.OFFICIAL_PRIMARY,
        published_at=T0,
    )
    contradict_source = _source(
        "alpha-q2-filing",
        tier=SourceTier.OFFICIAL_PRIMARY,
        published_at=T0 + timedelta(days=90),
    )
    ledger.register_source(support_source)
    ledger.register_source(contradict_source)

    support = build_evidence(
        claim_key=thesis.claim_key,
        source=support_source,
        statement="Q1 gross margin increased year over year.",
        relation=EvidenceRelation.SUPPORTS,
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        recorded_at=T0,
        confidence=0.9,
    )
    ledger.append(support)
    graph.link_evidence(
        claim_id=thesis.claim_id,
        evidence_id=support.evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.SUPPORTS,
        weight=0.9,
    )

    rule = build_falsification_rule(
        claim_id=thesis.claim_id,
        description="A new company filing showing margin contraction weakens the thesis.",
        rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
        min_sources=1,
        weight=1.0,
    )
    falsification.register_rule(rule, graph=graph)

    before = falsification.evaluate(
        thesis.claim_id,
        graph=graph,
        ledger=ledger,
        as_of=T0 + timedelta(days=1),
    )

    contradict = build_evidence(
        claim_key=thesis.claim_key,
        source=contradict_source,
        statement="Q2 filing shows gross margin contracted year over year.",
        relation=EvidenceRelation.CONTRADICTS,
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0 + timedelta(days=90),
        recorded_at=T0 + timedelta(days=90),
        confidence=0.95,
    )
    ledger.append(contradict)
    graph.link_evidence(
        claim_id=thesis.claim_id,
        evidence_id=contradict.evidence_id,
        ledger=ledger,
        relation=EvidenceRelation.CONTRADICTS,
        weight=1.0,
    )

    after = falsification.evaluate(
        thesis.claim_id,
        graph=graph,
        ledger=ledger,
        as_of=T0 + timedelta(days=91),
    )
    ok = (
        after.adjusted_confidence < before.adjusted_confidence
        and bool(after.triggered_rules)
        and after.final_status != before.final_status
    )
    return (
        ok,
        f"before={before.final_status.value}/{before.adjusted_confidence:.4f}, "
        f"after={after.final_status.value}/{after.adjusted_confidence:.4f}, "
        f"triggered={len(after.triggered_rules)}",
    )


CASES: list[
    tuple[str, str, float, Callable[[], tuple[bool, str]]]
] = [
    ("earnings_release_vs_secondary_summary", "source_precedence", 10.0, _earnings_release_vs_secondary_summary),
    ("guidance_vs_consensus", "semantic_separation", 10.0, _guidance_vs_consensus),
    ("duplicate_press_release_independence", "source_independence", 10.0, _duplicate_press_release_independence),
    ("restatement_correction", "revision_integrity", 10.0, _restatement_correction),
    ("filing_vs_acquisition_rumor", "source_quality", 10.0, _filing_vs_acquisition_rumor),
    ("cross_company_kpi_definition", "entity_isolation", 10.0, _cross_company_kpi_definition),
    ("period_mismatch_detection", "period_integrity", 10.0, _period_mismatch_detection),
    ("contradictory_revenue_numbers", "conflict_handling", 10.0, _contradictory_revenue_numbers),
    ("source_recency_precedence", "recency", 10.0, _source_recency_precedence),
    ("thesis_update_after_new_filing", "thesis_update", 10.0, _thesis_update_after_new_filing),
]


def run_company_sector_benchmark() -> dict:
    results: list[CompanySectorCase] = []
    for case_id, category, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            CompanySectorCase(
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
        "benchmark": "PanWatch Company/Sector Research",
        "version": "1.0.0-candidate",
        "track": "company_sector_research",
        "score_percent": score,
        "passed_cases": sum(1 for item in results if item.passed),
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 90.0,
        "regression_gate_passed": score >= 90.0,
        "limitations": [
            "This is a deterministic frozen-fixture benchmark, not a live-source retrieval benchmark.",
            "It tests research semantics and comparative integrity, not stock-picking or valuation skill.",
            "Cross-company entity isolation is intentionally adversarial and must fail closed rather than merge entities."
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PanWatch company/sector research benchmark"
    )
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    report = run_company_sector_benchmark()
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
