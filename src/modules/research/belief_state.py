"""Persistent belief-state semantics for PanWatch.

A belief snapshot is an immutable observation of what PanWatch believed about a
claim at a specific evaluation time. Belief events explain why the state changed
between snapshots. This layer never rewrites history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from .claim_graph import ClaimGraph, ClaimStatus
from .evidence import utc
from .falsification import FalsificationEngine
from .ledger import EvidenceLedger


class BeliefEventType(StrEnum):
    INITIALIZED = "initialized"
    STATUS_CHANGED = "status_changed"
    CONFIDENCE_RAISED = "confidence_raised"
    CONFIDENCE_LOWERED = "confidence_lowered"
    FALSIFIED = "falsified"
    RECOVERED = "recovered"
    EVIDENCE_CHANGED = "evidence_changed"
    FALSIFICATION_CHANGED = "falsification_changed"
    DEPENDENCY_CHANGED = "dependency_changed"


@dataclass(frozen=True)
class BeliefSnapshot:
    snapshot_id: str
    claim_id: str
    claim_key: str
    evaluated_at: datetime
    base_status: ClaimStatus
    final_status: ClaimStatus
    base_confidence: float
    final_confidence: float
    support_score: float
    contradiction_score: float
    falsification_coverage: float
    evidence_ids: tuple[str, ...]
    triggered_rules: tuple[str, ...]
    untestable_rules: tuple[str, ...]
    dependency_failures: tuple[str, ...]
    reasons: tuple[str, ...]
    input_fingerprint: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["base_status"] = self.base_status.value
        payload["final_status"] = self.final_status.value
        payload["evaluated_at"] = utc(self.evaluated_at).isoformat()
        return payload


@dataclass(frozen=True)
class BeliefEvent:
    event_id: str
    claim_id: str
    event_type: BeliefEventType
    occurred_at: datetime
    previous_snapshot_id: str | None
    current_snapshot_id: str
    previous_status: ClaimStatus | None
    current_status: ClaimStatus
    confidence_delta: float
    detail: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["event_type"] = self.event_type.value
        payload["occurred_at"] = utc(self.occurred_at).isoformat()
        payload["previous_status"] = (
            self.previous_status.value
            if self.previous_status is not None
            else None
        )
        payload["current_status"] = self.current_status.value
        return payload


@dataclass(frozen=True)
class BeliefUpdate:
    snapshot: BeliefSnapshot
    previous_snapshot: BeliefSnapshot | None
    events: tuple[BeliefEvent, ...]

    @property
    def changed(self) -> bool:
        return bool(self.events)


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BeliefStateEngine:
    """Turns current graph/evidence state into immutable belief history."""

    def __init__(self, *, confidence_event_threshold: float = 0.03) -> None:
        self.confidence_event_threshold = max(
            0.0,
            min(1.0, float(confidence_event_threshold)),
        )

    def evaluate(
        self,
        claim_id: str,
        *,
        graph: ClaimGraph,
        ledger: EvidenceLedger,
        falsification: FalsificationEngine,
        previous_snapshot: BeliefSnapshot | None = None,
        evaluated_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BeliefUpdate:
        when = utc(evaluated_at)
        claim = graph.get_claim(claim_id)
        assessment = graph.assess(claim_id, ledger, as_of=when)
        falsification_report = falsification.evaluate(
            claim_id,
            graph=graph,
            ledger=ledger,
            as_of=when,
        )

        available_evidence = {
            record.evidence_id
            for record in ledger.records
            if record.recorded_at <= when
        }
        evidence_ids = tuple(
            sorted(
                link.evidence_id
                for link in graph.links_for_claim(claim_id)
                if link.evidence_id in available_evidence
            )
        )
        triggered_rules = tuple(
            sorted(falsification_report.triggered_rules)
        )
        untestable_rules = tuple(
            sorted(falsification_report.untestable_rules)
        )
        dependency_failures = tuple(
            sorted(assessment.dependency_failures)
        )
        reasons = tuple(
            sorted(
                set(assessment.reasons)
                | {
                    f"falsification:{item.detail}"
                    for item in falsification_report.results
                    if item.rule_id in triggered_rules
                }
            )
        )
        input_fingerprint = _fingerprint(
            {
                "claim_id": claim_id,
                "evidence_ids": evidence_ids,
                "triggered_rules": triggered_rules,
                "untestable_rules": untestable_rules,
                "dependency_failures": dependency_failures,
                "base_status": assessment.status.value,
                "final_status": falsification_report.final_status.value,
                "base_confidence": assessment.confidence,
                "final_confidence": falsification_report.adjusted_confidence,
            }
        )
        snapshot_id = _stable_id(
            "belief",
            {
                "claim_id": claim_id,
                "evaluated_at": when.isoformat(),
                "input_fingerprint": input_fingerprint,
            },
        )
        snapshot = BeliefSnapshot(
            snapshot_id=snapshot_id,
            claim_id=claim_id,
            claim_key=claim.claim_key,
            evaluated_at=when,
            base_status=assessment.status,
            final_status=falsification_report.final_status,
            base_confidence=round(assessment.confidence, 4),
            final_confidence=round(
                falsification_report.adjusted_confidence,
                4,
            ),
            support_score=round(assessment.support_score, 4),
            contradiction_score=round(
                assessment.contradiction_score,
                4,
            ),
            falsification_coverage=round(
                falsification_report.coverage,
                4,
            ),
            evidence_ids=evidence_ids,
            triggered_rules=triggered_rules,
            untestable_rules=untestable_rules,
            dependency_failures=dependency_failures,
            reasons=reasons,
            input_fingerprint=input_fingerprint,
            metadata=dict(metadata or {}),
        )
        events = self._diff(previous_snapshot, snapshot)
        return BeliefUpdate(
            snapshot=snapshot,
            previous_snapshot=previous_snapshot,
            events=events,
        )

    def _diff(
        self,
        previous: BeliefSnapshot | None,
        current: BeliefSnapshot,
    ) -> tuple[BeliefEvent, ...]:
        if previous is None:
            return (
                self._event(
                    BeliefEventType.INITIALIZED,
                    previous,
                    current,
                    detail=(
                        f"belief initialized as {current.final_status.value} "
                        f"at confidence={current.final_confidence:.4f}"
                    ),
                ),
            )

        events: list[BeliefEvent] = []
        status_changed = previous.final_status is not current.final_status
        if (
            current.final_status is ClaimStatus.FALSIFIED
            and previous.final_status is not ClaimStatus.FALSIFIED
        ):
            events.append(
                self._event(
                    BeliefEventType.FALSIFIED,
                    previous,
                    current,
                    detail=(
                        f"belief became falsified from "
                        f"{previous.final_status.value}"
                    ),
                )
            )
        elif (
            previous.final_status is ClaimStatus.FALSIFIED
            and current.final_status is not ClaimStatus.FALSIFIED
        ):
            events.append(
                self._event(
                    BeliefEventType.RECOVERED,
                    previous,
                    current,
                    detail=(
                        f"belief recovered from falsified to "
                        f"{current.final_status.value}"
                    ),
                )
            )
        elif status_changed:
            events.append(
                self._event(
                    BeliefEventType.STATUS_CHANGED,
                    previous,
                    current,
                    detail=(
                        f"status {previous.final_status.value} -> "
                        f"{current.final_status.value}"
                    ),
                )
            )

        confidence_delta = round(
            current.final_confidence - previous.final_confidence,
            4,
        )
        if abs(confidence_delta) >= self.confidence_event_threshold:
            events.append(
                self._event(
                    (
                        BeliefEventType.CONFIDENCE_RAISED
                        if confidence_delta > 0
                        else BeliefEventType.CONFIDENCE_LOWERED
                    ),
                    previous,
                    current,
                    detail=(
                        f"confidence {previous.final_confidence:.4f} -> "
                        f"{current.final_confidence:.4f}"
                    ),
                )
            )

        if previous.evidence_ids != current.evidence_ids:
            added = sorted(
                set(current.evidence_ids) - set(previous.evidence_ids)
            )
            removed = sorted(
                set(previous.evidence_ids) - set(current.evidence_ids)
            )
            events.append(
                self._event(
                    BeliefEventType.EVIDENCE_CHANGED,
                    previous,
                    current,
                    detail=(
                        f"evidence changed; added={added}, removed={removed}"
                    ),
                    metadata={"added": added, "removed": removed},
                )
            )

        if (
            previous.triggered_rules != current.triggered_rules
            or previous.untestable_rules != current.untestable_rules
        ):
            events.append(
                self._event(
                    BeliefEventType.FALSIFICATION_CHANGED,
                    previous,
                    current,
                    detail=(
                        "falsification state changed; "
                        f"triggered={list(current.triggered_rules)}, "
                        f"untestable={list(current.untestable_rules)}"
                    ),
                )
            )

        if previous.dependency_failures != current.dependency_failures:
            events.append(
                self._event(
                    BeliefEventType.DEPENDENCY_CHANGED,
                    previous,
                    current,
                    detail=(
                        "dependency failures changed; "
                        f"current={list(current.dependency_failures)}"
                    ),
                )
            )

        return tuple(events)

    def _event(
        self,
        event_type: BeliefEventType,
        previous: BeliefSnapshot | None,
        current: BeliefSnapshot,
        *,
        detail: str,
        metadata: dict[str, Any] | None = None,
    ) -> BeliefEvent:
        previous_id = previous.snapshot_id if previous else None
        previous_status = previous.final_status if previous else None
        delta = round(
            current.final_confidence
            - (previous.final_confidence if previous else 0.0),
            4,
        )
        event_id = _stable_id(
            "bevent",
            {
                "claim_id": current.claim_id,
                "event_type": event_type.value,
                "previous": previous_id,
                "current": current.snapshot_id,
            },
        )
        return BeliefEvent(
            event_id=event_id,
            claim_id=current.claim_id,
            event_type=event_type,
            occurred_at=current.evaluated_at,
            previous_snapshot_id=previous_id,
            current_snapshot_id=current.snapshot_id,
            previous_status=previous_status,
            current_status=current.final_status,
            confidence_delta=delta,
            detail=detail,
            metadata=dict(metadata or {}),
        )
