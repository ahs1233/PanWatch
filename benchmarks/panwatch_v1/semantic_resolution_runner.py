"""Semantic Claim Resolution benchmark track v1.6."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from src.modules.research.claim_graph import ClaimGraph, ClaimKind, build_claim
from src.modules.research.claim_resolution import (
    ConservativeClaimResolver,
    SemanticClaimRelation,
)


@dataclass(frozen=True)
class ResolutionCase:
    case_id: str
    passed: bool
    points: float
    earned: float
    detail: str


def _graph(statement: str, key: str) -> ClaimGraph:
    graph = ClaimGraph()
    graph.register_claim(
        build_claim(
            claim_key=key,
            statement=statement,
            kind=ClaimKind.FACT,
        )
    )
    return graph


def _case_paraphrase() -> tuple[bool, str]:
    graph = _graph(
        "Data centres consumed around 415 TWh of electricity globally in 2024.",
        "energy.data_centres.electricity.2024",
    )
    result = ConservativeClaimResolver().resolve(
        statement=(
            "Global data centres consumed approximately 415 TWh of "
            "electricity in 2024."
        ),
        claim_key="energy.data_centres.electricity.2024",
        graph=graph,
    )
    return (
        result.relation is SemanticClaimRelation.PARAPHRASE,
        f"relation={result.relation.value} score={result.score}",
    )


def _case_numeric_conflict() -> tuple[bool, str]:
    graph = _graph(
        "Company revenue was 10 billion dollars in 2025.",
        "company.revenue.2025",
    )
    result = ConservativeClaimResolver().resolve(
        statement="Company revenue was 8 billion dollars in 2025.",
        claim_key="company.revenue.2025",
        graph=graph,
    )
    return (
        result.relation is SemanticClaimRelation.CONTRADICTION,
        f"relation={result.relation.value} numeric_match={result.numeric_match}",
    )


def _case_period_separation() -> tuple[bool, str]:
    graph = _graph(
        "Company revenue was 10 billion dollars in 2024.",
        "company.revenue",
    )
    result = ConservativeClaimResolver().resolve(
        statement="Company revenue was 12 billion dollars in 2025.",
        claim_key="company.revenue",
        graph=graph,
    )
    return (
        result.relation is SemanticClaimRelation.DISTINCT
        and result.period_match is False,
        f"relation={result.relation.value} period_match={result.period_match}",
    )


def _case_polarity_conflict() -> tuple[bool, str]:
    graph = _graph(
        "Electricity demand increased in 2025.",
        "energy.electricity_demand.2025",
    )
    result = ConservativeClaimResolver().resolve(
        statement="Electricity demand decreased in 2025.",
        claim_key="energy.electricity_demand.2025",
        graph=graph,
    )
    return (
        result.relation is SemanticClaimRelation.CONTRADICTION,
        f"relation={result.relation.value} polarity_match={result.polarity_match}",
    )


def _case_revision() -> tuple[bool, str]:
    graph = _graph(
        "The agency reported unemployment at 6.2 percent.",
        "macro.unemployment.current",
    )
    result = ConservativeClaimResolver().resolve(
        statement="The agency revised unemployment to 5.8 percent.",
        claim_key="macro.unemployment.current",
        graph=graph,
        supersedes_previous=True,
        revision_explicit=True,
    )
    return (
        result.relation is SemanticClaimRelation.REVISION,
        f"relation={result.relation.value} key_match={result.key_match}",
    )


CASES = [
    ("paraphrase_merge", 25.0, _case_paraphrase),
    ("numeric_conflict", 25.0, _case_numeric_conflict),
    ("period_separation", 20.0, _case_period_separation),
    ("polarity_conflict", 20.0, _case_polarity_conflict),
    ("explicit_revision", 10.0, _case_revision),
]


def run_semantic_resolution_benchmark() -> dict:
    results: list[ResolutionCase] = []
    for case_id, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            ResolutionCase(
                case_id=case_id,
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
        "track": "semantic_claim_resolution",
        "score_percent": score,
        "passed_cases": sum(1 for item in results if item.passed),
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 90.0,
        "regression_gate_passed": score >= 90.0,
        "limitations": [
            "Resolver is deliberately conservative and does not use embedding-only merges.",
            "Entity linking and unit normalization beyond lexical/numeric signals remain future work.",
            "Ambiguous claims stay separate instead of being silently merged."
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Semantic Claim Resolution benchmark"
    )
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--no-gate", action="store_true")
    args = parser.parse_args()
    report = run_semantic_resolution_benchmark()
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
