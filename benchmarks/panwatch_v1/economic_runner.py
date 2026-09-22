"""Economic evidence-integrity benchmark for PanWatch Benchmark v1.

This track does not measure forecasting skill or live web research quality. It
measures whether the research foundation preserves economic evidence semantics
that are prerequisites for trustworthy research.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from src.modules.research import (
    EvidenceLedger,
    EvidenceRelation,
    ObservationKind,
    SourceTier,
    build_evidence,
    build_source,
)


UTC = timezone.utc
T0 = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


@dataclass(frozen=True)
class EconomicCaseResult:
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
):
    body = content or f"economic benchmark source {slug}"
    return build_source(
        url=f"https://{slug}.example.com/report?utm_source=benchmark",
        content=body,
        publisher=slug,
        title=f"{slug} report",
        source_tier=tier,
        source_family=slug,
        upstream_origin=upstream_origin,
        published_at=published_at,
        retrieved_at=published_at + timedelta(minutes=2),
        observed_at=published_at + timedelta(minutes=2),
        tool_name="benchmark",
    )


def _actual_vs_forecast_macro_release() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    consensus = _source("consensus")
    official = _source("official", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(consensus)
    ledger.register_source(official)
    forecast = build_evidence(
        claim_key="macro.cpi.yoy",
        source=consensus,
        statement="Consensus forecast is 3.2 percent",
        observation_kind=ObservationKind.FORECAST,
        event_time=T0,
        numeric_value=3.2,
        unit="%",
        period="2026-08",
    )
    ledger.append(forecast)
    before = ledger.actual_forecast_guard("macro.cpi.yoy")
    actual = build_evidence(
        claim_key="macro.cpi.yoy",
        source=official,
        statement="CPI rose 3.5 percent year over year",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0 + timedelta(hours=1),
        numeric_value=3.5,
        unit="%",
        period="2026-08",
    )
    ledger.append(actual)
    after = ledger.actual_forecast_guard("macro.cpi.yoy")
    resolved = ledger.resolve_numeric("macro.cpi.yoy")
    ok = (
        before.passed is False
        and before.code == "forecast_only"
        and after.passed is True
        and resolved.value == 3.5
    )
    return ok, f"before={before.code}, after={after.code}, value={resolved.value}"


def _revision_vs_original_release() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    official = _source("statistics-office", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(official)
    original = build_evidence(
        claim_key="macro.payrolls",
        source=official,
        statement="Payrolls increased by 100 thousand",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
        numeric_value=100.0,
        unit="k",
        period="2026-07",
    )
    ledger.append(original)
    revision = build_evidence(
        claim_key="macro.payrolls",
        source=official,
        statement="Payrolls revised to 82 thousand",
        observation_kind=ObservationKind.REVISION,
        event_time=T0 + timedelta(days=30),
        numeric_value=82.0,
        unit="k",
        period="2026-07",
        revision_of=original.evidence_id,
        supersedes=original.evidence_id,
    )
    ledger.append(revision)
    resolved = ledger.resolve_numeric("macro.payrolls")
    ok = (
        resolved.status == "resolved_revision"
        and resolved.selected_evidence_id == revision.evidence_id
        and resolved.value == 82.0
    )
    return ok, f"status={resolved.status}, value={resolved.value}"


def _duplicate_wire_source_independence() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    content = "Central bank kept the policy rate unchanged."
    sources = [
        _source(
            f"syndicate-{idx}",
            content=content,
            upstream_origin="wire:story-001",
        )
        for idx in range(3)
    ]
    for source in sources:
        ledger.register_source(source)
        ledger.append(
            build_evidence(
                claim_key="rates.policy.unchanged",
                source=source,
                statement=content,
                observation_kind=ObservationKind.ACTUAL,
                event_time=T0,
                confidence=0.9,
            )
        )
    independent = ledger.independent_source_count("rates.policy.unchanged")
    ratio = ledger.source_independence_ratio("rates.policy.unchanged")
    confidence = ledger.claim_confidence("rates.policy.unchanged")
    ok = independent == 1 and round(ratio, 4) == round(1 / 3, 4) and confidence == 1.0
    return ok, f"independent={independent}, ratio={ratio:.4f}, confidence={confidence:.4f}"


def _stale_macro_report_trap() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    stale_source = _source(
        "stale-report",
        tier=SourceTier.PRIMARY,
        published_at=T0 - timedelta(days=5),
    )
    ledger.register_source(stale_source)
    ledger.append(
        build_evidence(
            claim_key="macro.activity.now",
            source=stale_source,
            statement="Activity remains strong",
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0 - timedelta(days=5),
        )
    )
    stale = ledger.freshness_guard(
        "macro.activity.now",
        max_age=timedelta(hours=24),
        as_of=T0,
    )

    fresh_source = _source(
        "fresh-report",
        tier=SourceTier.OFFICIAL_PRIMARY,
        published_at=T0 - timedelta(hours=2),
    )
    ledger.register_source(fresh_source)
    ledger.append(
        build_evidence(
            claim_key="macro.activity.now",
            source=fresh_source,
            statement="Latest activity data softened",
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0 - timedelta(hours=2),
        )
    )
    fresh = ledger.freshness_guard(
        "macro.activity.now",
        max_age=timedelta(hours=24),
        as_of=T0,
    )
    return (
        stale.passed is False and stale.code == "stale" and fresh.passed is True,
        f"stale={stale.code}, refreshed={fresh.code}",
    )


def _conflicting_numeric_claims() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    official = _source("official-gdp", tier=SourceTier.OFFICIAL_PRIMARY)
    secondary = _source("secondary-gdp", tier=SourceTier.SECONDARY)
    ledger.register_source(official)
    ledger.register_source(secondary)
    for source, value in ((official, 2.1), (secondary, 1.7)):
        ledger.append(
            build_evidence(
                claim_key="macro.gdp.qoq",
                source=source,
                statement=f"GDP growth {value} percent",
                observation_kind=ObservationKind.ACTUAL,
                event_time=T0,
                numeric_value=value,
                unit="%",
                period="2026-Q2",
            )
        )
    resolved = ledger.resolve_numeric("macro.gdp.qoq")
    ok = (
        resolved.status == "conflict"
        and resolved.value == 2.1
        and len(resolved.conflicts) == 1
        and resolved.independent_source_count == 2
    )
    return ok, f"status={resolved.status}, selected={resolved.value}, conflicts={len(resolved.conflicts)}"


def _event_chronology_reconstruction() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    source = _source("chronology", tier=SourceTier.PRIMARY)
    ledger.register_source(source)
    times = [T0 + timedelta(hours=2), T0, T0 + timedelta(hours=1)]
    ids = []
    for idx, when in enumerate(times):
        record = build_evidence(
            claim_key="event.timeline",
            source=source,
            statement=f"event step {idx}",
            observation_kind=ObservationKind.ACTUAL,
            event_time=when,
        )
        ids.append(record.evidence_id)
        ledger.append(record)
    ordered = ledger.chronology("event.timeline")
    ordered_times = [item.event_time for item in ordered]
    ok = ordered_times == sorted(ordered_times)
    return ok, " -> ".join(item.statement for item in ordered)


def _policy_statement_vs_market_pricing() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    bank = _source("central-bank", tier=SourceTier.OFFICIAL_PRIMARY)
    market = _source("market-pricing", tier=SourceTier.PRIMARY)
    ledger.register_source(bank)
    ledger.register_source(market)
    ledger.append(
        build_evidence(
            claim_key="rates.path",
            source=bank,
            statement="Policy will remain restrictive",
            observation_kind=ObservationKind.POLICY_STATEMENT,
            event_time=T0,
        )
    )
    ledger.append(
        build_evidence(
            claim_key="rates.path",
            source=market,
            statement="Futures price a cut within two meetings",
            observation_kind=ObservationKind.MARKET_PRICING,
            event_time=T0 + timedelta(minutes=5),
        )
    )
    guard = ledger.kind_separation_guard(
        "rates.path",
        ObservationKind.POLICY_STATEMENT,
        ObservationKind.MARKET_PRICING,
    )
    return guard.passed, f"{guard.code}: {guard.detail}"


def _oil_shock_delayed_transmission() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    oil = _source("oil-source", tier=SourceTier.PRIMARY)
    macro = _source("macro-source", tier=SourceTier.PRIMARY)
    ledger.register_source(oil)
    ledger.register_source(macro)
    shock = build_evidence(
        claim_key="oil.shock",
        source=oil,
        statement="Oil price shock begins",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0,
    )
    impact = build_evidence(
        claim_key="inflation.transmission",
        source=macro,
        statement="Fuel-price pass-through appears later",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0 + timedelta(hours=8),
    )
    ledger.append(impact)
    ledger.append(shock)
    lag = ledger.lag_seconds(shock.evidence_id, impact.evidence_id)
    return lag == 8 * 3600, f"lag_seconds={lag:.0f}"


def _forecast_to_realization_update() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    forecaster = _source("forecaster")
    official = _source("release", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(forecaster)
    ledger.register_source(official)
    forecast = build_evidence(
        claim_key="macro.retail.sales",
        source=forecaster,
        statement="Retail sales forecast plus 0.4 percent",
        observation_kind=ObservationKind.FORECAST,
        event_time=T0,
        numeric_value=0.4,
        unit="%",
        period="2026-08",
    )
    ledger.append(forecast)
    actual = build_evidence(
        claim_key="macro.retail.sales",
        source=official,
        statement="Retail sales rose 0.1 percent",
        observation_kind=ObservationKind.ACTUAL,
        event_time=T0 + timedelta(days=1),
        numeric_value=0.1,
        unit="%",
        period="2026-08",
    )
    ledger.append(actual)
    resolved = ledger.resolve_numeric("macro.retail.sales")
    kinds = {
        item.observation_kind
        for item in ledger.for_claim("macro.retail.sales")
    }
    ok = (
        resolved.value == 0.1
        and ObservationKind.FORECAST in kinds
        and ObservationKind.ACTUAL in kinds
    )
    return ok, f"resolved={resolved.value}, preserved_kinds={sorted(k.value for k in kinds)}"


def _confidence_change_after_new_evidence() -> tuple[bool, str]:
    ledger = EvidenceLedger()
    support = _source("support", tier=SourceTier.PRIMARY)
    contradict = _source("contradict", tier=SourceTier.OFFICIAL_PRIMARY)
    ledger.register_source(support)
    ledger.register_source(contradict)
    ledger.append(
        build_evidence(
            claim_key="macro.soft-landing",
            source=support,
            statement="Indicators support a soft-landing thesis",
            relation=EvidenceRelation.SUPPORTS,
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0,
            recorded_at=T0,
            confidence=0.85,
        )
    )
    before = ledger.claim_confidence(
        "macro.soft-landing",
        as_of=T0 + timedelta(minutes=1),
    )
    ledger.append(
        build_evidence(
            claim_key="macro.soft-landing",
            source=contradict,
            statement="New official data contradict the soft-landing thesis",
            relation=EvidenceRelation.CONTRADICTS,
            observation_kind=ObservationKind.ACTUAL,
            event_time=T0 + timedelta(hours=2),
            recorded_at=T0 + timedelta(hours=2),
            confidence=0.95,
        )
    )
    after = ledger.claim_confidence(
        "macro.soft-landing",
        as_of=T0 + timedelta(hours=3),
    )
    return after < before, f"before={before:.4f}, after={after:.4f}"


CASES: list[
    tuple[str, str, float, Callable[[], tuple[bool, str]]]
] = [
    ("actual_vs_forecast_macro_release", "actual_forecast", 12.0, _actual_vs_forecast_macro_release),
    ("revision_vs_original_release", "revision_integrity", 12.0, _revision_vs_original_release),
    ("duplicate_wire_source_independence", "source_independence", 12.0, _duplicate_wire_source_independence),
    ("stale_macro_report_trap", "freshness", 10.0, _stale_macro_report_trap),
    ("conflicting_numeric_claims", "numeric_conflict", 12.0, _conflicting_numeric_claims),
    ("event_chronology_reconstruction", "chronology", 8.0, _event_chronology_reconstruction),
    ("policy_statement_vs_market_pricing", "semantic_separation", 8.0, _policy_statement_vs_market_pricing),
    ("oil_shock_delayed_transmission", "temporal_reasoning", 8.0, _oil_shock_delayed_transmission),
    ("forecast_to_realization_update", "state_update", 10.0, _forecast_to_realization_update),
    ("confidence_change_after_new_evidence", "confidence_update", 8.0, _confidence_change_after_new_evidence),
]


def run_economic_benchmark() -> dict:
    results: list[EconomicCaseResult] = []
    for case_id, category, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            EconomicCaseResult(
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
        "version": "1.5.0",
        "track": "economic_evidence_integrity",
        "score_percent": score,
        "passed_cases": sum(1 for item in results if item.passed),
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 90.0,
        "regression_gate_passed": score >= 90.0,
        "limitations": [
            "This track measures evidence semantics, not economic forecasting skill.",
            "Cases use deterministic frozen fixtures; they do not prove live-source retrieval quality.",
            "No external LLM or human researcher baseline is scored yet.",
            "Claim Graph and generic falsification are scored in a separate v1.2 track; live counter-research execution remains outside this track.",
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PanWatch economic evidence-integrity benchmark"
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Write machine-readable result JSON",
    )
    parser.add_argument(
        "--no-gate",
        action="store_true",
        help="Do not fail on regression gate",
    )
    args = parser.parse_args()
    report = run_economic_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.no_gate:
        return 0
    return 0 if report["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
