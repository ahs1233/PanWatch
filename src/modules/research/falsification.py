"""Falsification engine for explicit, testable claim failure conditions."""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

from .claim_graph import ClaimGraph, ClaimStatus
from .evidence import EvidenceRelation, ObservationKind, normalize_text, utc
from .ledger import EvidenceLedger


class FalsificationRuleType(StrEnum):
    CONTRADICTORY_EVIDENCE = "contradictory_evidence"
    NUMERIC_THRESHOLD = "numeric_threshold"
    DEPENDENCY_FAILURE = "dependency_failure"
    FRESHNESS_FAILURE = "freshness_failure"
    MISSING_INDEPENDENT_SUPPORT = "missing_independent_support"
    COUNTERCLAIM_CONFIDENCE = "counterclaim_confidence"


class FalsificationState(StrEnum):
    TRIGGERED = "triggered"
    NOT_TRIGGERED = "not_triggered"
    UNTESTABLE = "untestable"


@dataclass(frozen=True)
class FalsificationRule:
    rule_id: str
    claim_id: str
    description: str
    rule_type: FalsificationRuleType
    hard_fail: bool
    weight: float
    evidence_claim_key: str = ""
    operator: str = ""
    threshold: float | None = None
    min_sources: int = 1
    max_age_seconds: int | None = None
    related_claim_id: str | None = None
    required_kinds: tuple[ObservationKind, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rule_type"] = self.rule_type.value
        payload["required_kinds"] = [
            kind.value for kind in self.required_kinds
        ]
        return payload


@dataclass(frozen=True)
class FalsificationResult:
    rule_id: str
    state: FalsificationState
    severity: float
    observed: Any
    detail: str


@dataclass(frozen=True)
class FalsificationProbe:
    rule_id: str
    claim_id: str
    priority: float
    instruction: str
    missing_requirement: str


@dataclass(frozen=True)
class FalsificationReport:
    claim_id: str
    base_status: ClaimStatus
    final_status: ClaimStatus
    base_confidence: float
    adjusted_confidence: float
    coverage: float
    triggered_rules: tuple[str, ...]
    untestable_rules: tuple[str, ...]
    results: tuple[FalsificationResult, ...]
    probes: tuple[FalsificationProbe, ...]


def _stable_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"fal_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def build_falsification_rule(
    *,
    claim_id: str,
    description: str,
    rule_type: FalsificationRuleType,
    hard_fail: bool = False,
    weight: float = 1.0,
    evidence_claim_key: str = "",
    operator: str = "",
    threshold: float | None = None,
    min_sources: int = 1,
    max_age_seconds: int | None = None,
    related_claim_id: str | None = None,
    required_kinds: tuple[ObservationKind, ...] | list[ObservationKind] = (),
    metadata: dict[str, Any] | None = None,
) -> FalsificationRule:
    text = normalize_text(description)
    if not claim_id:
        raise ValueError("claim_id is required")
    if not text:
        raise ValueError("description is required")
    kinds = tuple(ObservationKind(kind) for kind in required_kinds)
    rule_type = FalsificationRuleType(rule_type)
    normalized_operator = normalize_text(operator)
    if rule_type is FalsificationRuleType.NUMERIC_THRESHOLD:
        if normalized_operator not in {">", ">=", "<", "<=", "==", "!="}:
            raise ValueError("numeric threshold rule requires a valid operator")
        if threshold is None:
            raise ValueError("numeric threshold rule requires threshold")
        if not evidence_claim_key:
            raise ValueError(
                "numeric threshold rule requires evidence_claim_key"
            )
    if rule_type is FalsificationRuleType.FRESHNESS_FAILURE:
        if max_age_seconds is None or max_age_seconds <= 0:
            raise ValueError(
                "freshness rule requires positive max_age_seconds"
            )
        if not evidence_claim_key:
            raise ValueError(
                "freshness rule requires evidence_claim_key"
            )
    if rule_type is FalsificationRuleType.COUNTERCLAIM_CONFIDENCE:
        if not related_claim_id:
            raise ValueError(
                "counterclaim rule requires related_claim_id"
            )
        if threshold is None:
            raise ValueError("counterclaim rule requires threshold")
    if rule_type is FalsificationRuleType.DEPENDENCY_FAILURE:
        if not related_claim_id:
            raise ValueError(
                "dependency rule requires related_claim_id"
            )
        if threshold is None:
            threshold = 0.35

    rule_id = _stable_id(
        {
            "claim_id": claim_id,
            "description": text,
            "rule_type": rule_type.value,
            "evidence_claim_key": evidence_claim_key,
            "operator": normalized_operator,
            "threshold": threshold,
            "related_claim_id": related_claim_id,
        }
    )
    return FalsificationRule(
        rule_id=rule_id,
        claim_id=claim_id,
        description=text,
        rule_type=rule_type,
        hard_fail=bool(hard_fail),
        weight=max(0.0, min(1.0, float(weight))),
        evidence_claim_key=normalize_text(evidence_claim_key),
        operator=normalized_operator,
        threshold=float(threshold) if threshold is not None else None,
        min_sources=max(1, int(min_sources)),
        max_age_seconds=(
            int(max_age_seconds)
            if max_age_seconds is not None
            else None
        ),
        related_claim_id=related_claim_id,
        required_kinds=kinds,
        metadata=dict(metadata or {}),
    )


class FalsificationEngine:
    """Evaluates explicit disconfirmation conditions before accepting a thesis."""

    def __init__(self) -> None:
        self._rules: dict[str, FalsificationRule] = {}

    @property
    def rules(self) -> tuple[FalsificationRule, ...]:
        return tuple(self._rules.values())

    def register_rule(
        self,
        rule: FalsificationRule,
        *,
        graph: ClaimGraph,
    ) -> FalsificationRule:
        graph.get_claim(rule.claim_id)
        if rule.related_claim_id:
            graph.get_claim(rule.related_claim_id)
        existing = self._rules.get(rule.rule_id)
        if existing is not None:
            if existing != rule:
                raise ValueError(
                    f"falsification rule identity collision: {rule.rule_id}"
                )
            return existing
        self._rules[rule.rule_id] = rule
        return rule

    def rules_for_claim(self, claim_id: str) -> list[FalsificationRule]:
        return [
            rule for rule in self._rules.values()
            if rule.claim_id == claim_id
        ]

    def evaluate(
        self,
        claim_id: str,
        *,
        graph: ClaimGraph,
        ledger: EvidenceLedger,
        as_of: datetime | None = None,
    ) -> FalsificationReport:
        assessment = graph.assess(claim_id, ledger, as_of=as_of)
        rules = self.rules_for_claim(claim_id)
        results = tuple(
            self._evaluate_rule(
                rule,
                graph=graph,
                ledger=ledger,
                as_of=as_of,
            )
            for rule in rules
        )
        testable = [
            result for result in results
            if result.state is not FalsificationState.UNTESTABLE
        ]
        coverage = (
            len(testable) / len(results)
            if results
            else 0.0
        )
        triggered = [
            (rule, result)
            for rule, result in zip(rules, results)
            if result.state is FalsificationState.TRIGGERED
        ]
        hard_trigger = any(
            rule.hard_fail for rule, _result in triggered
        )
        total_weight = sum(
            rule.weight
            for rule, result in zip(rules, results)
            if result.state is not FalsificationState.UNTESTABLE
        )
        pressure = 0.0
        if total_weight > 0:
            pressure = sum(
                rule.weight * result.severity
                for rule, result in zip(rules, results)
                if result.state is FalsificationState.TRIGGERED
            ) / total_weight
        adjusted = assessment.confidence
        if hard_trigger:
            adjusted = min(adjusted, 0.05)
        else:
            adjusted *= 1.0 - min(0.85, pressure * 0.75)
        adjusted = round(max(0.0, min(1.0, adjusted)), 4)

        if hard_trigger or adjusted <= 0.20:
            final_status = ClaimStatus.FALSIFIED
        elif triggered and adjusted < 0.55:
            final_status = ClaimStatus.CONTESTED
        elif assessment.status is ClaimStatus.SUPPORTED and adjusted >= 0.70:
            final_status = ClaimStatus.SUPPORTED
        elif assessment.status is ClaimStatus.INSUFFICIENT_EVIDENCE:
            final_status = ClaimStatus.INSUFFICIENT_EVIDENCE
        else:
            final_status = ClaimStatus.OPEN

        probes = tuple(
            self._probe_for(rule, result)
            for rule, result in zip(rules, results)
            if result.state is FalsificationState.UNTESTABLE
        )
        return FalsificationReport(
            claim_id=claim_id,
            base_status=assessment.status,
            final_status=final_status,
            base_confidence=assessment.confidence,
            adjusted_confidence=adjusted,
            coverage=round(coverage, 4),
            triggered_rules=tuple(
                result.rule_id
                for result in results
                if result.state is FalsificationState.TRIGGERED
            ),
            untestable_rules=tuple(
                result.rule_id
                for result in results
                if result.state is FalsificationState.UNTESTABLE
            ),
            results=results,
            probes=probes,
        )

    def _evaluate_rule(
        self,
        rule: FalsificationRule,
        *,
        graph: ClaimGraph,
        ledger: EvidenceLedger,
        as_of: datetime | None,
    ) -> FalsificationResult:
        if rule.rule_type is FalsificationRuleType.CONTRADICTORY_EVIDENCE:
            return self._contradictory_evidence(
                rule,
                graph=graph,
                ledger=ledger,
                as_of=as_of,
            )
        if rule.rule_type is FalsificationRuleType.NUMERIC_THRESHOLD:
            return self._numeric_threshold(
                rule,
                ledger=ledger,
            )
        if rule.rule_type is FalsificationRuleType.DEPENDENCY_FAILURE:
            assessment = graph.assess(
                str(rule.related_claim_id),
                ledger,
                as_of=as_of,
            )
            threshold = float(rule.threshold or 0.35)
            triggered = (
                assessment.status is ClaimStatus.FALSIFIED
                or assessment.confidence < threshold
            )
            return FalsificationResult(
                rule_id=rule.rule_id,
                state=(
                    FalsificationState.TRIGGERED
                    if triggered
                    else FalsificationState.NOT_TRIGGERED
                ),
                severity=(
                    1.0
                    if assessment.status is ClaimStatus.FALSIFIED
                    else max(
                        0.0,
                        min(1.0, (threshold - assessment.confidence) / threshold),
                    )
                    if triggered and threshold > 0
                    else 0.0
                ),
                observed=assessment.confidence,
                detail=(
                    f"dependency confidence={assessment.confidence:.4f}, "
                    f"threshold={threshold:.4f}"
                ),
            )
        if rule.rule_type is FalsificationRuleType.FRESHNESS_FAILURE:
            guard = ledger.freshness_guard(
                rule.evidence_claim_key,
                max_age=timedelta(seconds=int(rule.max_age_seconds or 0)),
                as_of=as_of,
                kinds=rule.required_kinds or None,
            )
            if guard.code == "evidence_missing":
                return FalsificationResult(
                    rule.rule_id,
                    FalsificationState.UNTESTABLE,
                    0.0,
                    None,
                    guard.detail,
                )
            return FalsificationResult(
                rule.rule_id,
                (
                    FalsificationState.TRIGGERED
                    if not guard.passed
                    else FalsificationState.NOT_TRIGGERED
                ),
                1.0 if not guard.passed else 0.0,
                guard.code,
                guard.detail,
            )
        if rule.rule_type is FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT:
            key = rule.evidence_claim_key or graph.get_claim(
                rule.claim_id
            ).claim_key
            rows = ledger.for_claim(
                key,
                kinds=rule.required_kinds or None,
                as_of=as_of,
            )
            if not rows:
                return FalsificationResult(
                    rule.rule_id,
                    FalsificationState.TRIGGERED,
                    1.0,
                    0,
                    "no supporting evidence available",
                )
            source_keys = {
                ledger.source_for(record).independence_key
                for record in rows
                if record.relation is EvidenceRelation.SUPPORTS
            }
            observed = len(source_keys)
            triggered = observed < rule.min_sources
            severity = (
                max(0.0, min(1.0, 1 - observed / rule.min_sources))
                if triggered
                else 0.0
            )
            return FalsificationResult(
                rule.rule_id,
                (
                    FalsificationState.TRIGGERED
                    if triggered
                    else FalsificationState.NOT_TRIGGERED
                ),
                severity,
                observed,
                (
                    f"independent_support={observed}, "
                    f"required={rule.min_sources}"
                ),
            )
        if rule.rule_type is FalsificationRuleType.COUNTERCLAIM_CONFIDENCE:
            assessment = graph.assess(
                str(rule.related_claim_id),
                ledger,
                as_of=as_of,
            )
            threshold = float(rule.threshold or 0.5)
            triggered = assessment.confidence >= threshold
            severity = (
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            assessment.confidence - threshold
                        ) / max(0.0001, 1.0 - threshold),
                    ),
                )
                if triggered
                else 0.0
            )
            return FalsificationResult(
                rule.rule_id,
                (
                    FalsificationState.TRIGGERED
                    if triggered
                    else FalsificationState.NOT_TRIGGERED
                ),
                severity,
                assessment.confidence,
                (
                    f"counterclaim confidence={assessment.confidence:.4f}, "
                    f"threshold={threshold:.4f}"
                ),
            )
        raise ValueError(f"unsupported rule type: {rule.rule_type}")

    def _contradictory_evidence(
        self,
        rule: FalsificationRule,
        *,
        graph: ClaimGraph,
        ledger: EvidenceLedger,
        as_of: datetime | None,
    ) -> FalsificationResult:
        links = graph.links_for_claim(rule.claim_id)
        record_by_id = {
            record.evidence_id: record
            for record in ledger.records
            if as_of is None or record.recorded_at <= utc(as_of)
        }
        contradictions: dict[str, float] = {}
        for link in links:
            if link.relation is not EvidenceRelation.CONTRADICTS:
                continue
            record = record_by_id.get(link.evidence_id)
            if record is None:
                continue
            key = ledger.source_for(record).independence_key
            contradictions[key] = max(
                contradictions.get(key, 0.0),
                record.confidence * link.weight,
            )
        if not links:
            return FalsificationResult(
                rule.rule_id,
                FalsificationState.UNTESTABLE,
                0.0,
                None,
                "claim has no evidence links",
            )
        count = len(contradictions)
        triggered = count >= rule.min_sources
        severity = (
            min(
                1.0,
                (count / rule.min_sources)
                * (
                    sum(contradictions.values()) / count
                    if count
                    else 0.0
                ),
            )
            if triggered
            else 0.0
        )
        return FalsificationResult(
            rule.rule_id,
            (
                FalsificationState.TRIGGERED
                if triggered
                else FalsificationState.NOT_TRIGGERED
            ),
            severity,
            count,
            f"independent_contradictions={count}",
        )

    def _numeric_threshold(
        self,
        rule: FalsificationRule,
        *,
        ledger: EvidenceLedger,
    ) -> FalsificationResult:
        preferred = (
            rule.required_kinds[0]
            if rule.required_kinds
            else ObservationKind.ACTUAL
        )
        resolved = ledger.resolve_numeric(
            rule.evidence_claim_key,
            preferred_kind=preferred,
        )
        if resolved.value is None:
            return FalsificationResult(
                rule.rule_id,
                FalsificationState.UNTESTABLE,
                0.0,
                None,
                resolved.detail,
            )
        if resolved.status == "conflict":
            return FalsificationResult(
                rule.rule_id,
                FalsificationState.UNTESTABLE,
                0.0,
                resolved.value,
                "numeric evidence remains conflicted",
            )
        ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "<": operator.lt,
            "<=": operator.le,
            "==": operator.eq,
            "!=": operator.ne,
        }
        threshold = float(rule.threshold)
        triggered = bool(ops[rule.operator](resolved.value, threshold))
        distance = abs(resolved.value - threshold)
        scale = max(1.0, abs(threshold))
        severity = min(1.0, distance / scale) if triggered else 0.0
        if triggered and severity == 0.0:
            severity = 1.0
        return FalsificationResult(
            rule.rule_id,
            (
                FalsificationState.TRIGGERED
                if triggered
                else FalsificationState.NOT_TRIGGERED
            ),
            round(severity, 4),
            resolved.value,
            (
                f"observed={resolved.value} "
                f"{rule.operator} threshold={threshold}"
            ),
        )

    def _probe_for(
        self,
        rule: FalsificationRule,
        result: FalsificationResult,
    ) -> FalsificationProbe:
        if rule.rule_type is FalsificationRuleType.NUMERIC_THRESHOLD:
            instruction = (
                "Find a current, independently sourced actual value for "
                f"{rule.evidence_claim_key} and test whether it is "
                f"{rule.operator} {rule.threshold}."
            )
        elif rule.rule_type is FalsificationRuleType.FRESHNESS_FAILURE:
            instruction = (
                "Find a fresh observation for "
                f"{rule.evidence_claim_key} within "
                f"{rule.max_age_seconds} seconds."
            )
        elif rule.rule_type is FalsificationRuleType.CONTRADICTORY_EVIDENCE:
            instruction = (
                "Actively search for independent evidence that contradicts "
                "the target claim."
            )
        else:
            instruction = rule.description
        return FalsificationProbe(
            rule_id=rule.rule_id,
            claim_id=rule.claim_id,
            priority=round(rule.weight * (1.5 if rule.hard_fail else 1.0), 4),
            instruction=instruction,
            missing_requirement=result.detail,
        )
