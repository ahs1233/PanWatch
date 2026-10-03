"""Deterministic claim graph built on top of the Evidence Foundation.

The graph makes research conclusions inspectable. Evidence supports or
contradicts claims, claims depend on other claims, and confidence can be
propagated without hiding the dependency structure inside an LLM response.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from .evidence import EvidenceRelation, normalize_text, utc
from .ledger import EvidenceLedger


class ClaimKind(StrEnum):
    FACT = "fact"
    INTERPRETATION = "interpretation"
    HYPOTHESIS = "hypothesis"
    FORECAST = "forecast"
    MECHANISM = "mechanism"
    SCENARIO = "scenario"
    CONCLUSION = "conclusion"


class ClaimStatus(StrEnum):
    OPEN = "open"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    FALSIFIED = "falsified"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    SUPERSEDED = "superseded"


class ClaimRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DEPENDS_ON = "depends_on"
    CAUSED_BY = "caused_by"
    IMPLIES = "implies"
    ALTERNATIVE_TO = "alternative_to"
    REFINES = "refines"
    INVALIDATES = "invalidates"


_ACYCLIC_RELATIONS = {
    ClaimRelation.SUPPORTS,
    ClaimRelation.DEPENDS_ON,
    ClaimRelation.CAUSED_BY,
    ClaimRelation.IMPLIES,
    ClaimRelation.REFINES,
}


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class ClaimNode:
    claim_id: str
    claim_key: str
    statement: str
    kind: ClaimKind
    prior_confidence: float
    created_at: datetime
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    supersedes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        for key in ("created_at", "valid_from", "valid_until"):
            value = payload[key]
            payload[key] = utc(value).isoformat() if value is not None else None
        return payload


@dataclass(frozen=True)
class ClaimEdge:
    edge_id: str
    source_claim_id: str
    target_claim_id: str
    relation: ClaimRelation
    weight: float
    required: bool
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relation"] = self.relation.value
        payload["created_at"] = utc(self.created_at).isoformat()
        return payload


@dataclass(frozen=True)
class ClaimEvidenceLink:
    link_id: str
    claim_id: str
    evidence_id: str
    relation: EvidenceRelation
    weight: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relation"] = self.relation.value
        payload["created_at"] = utc(self.created_at).isoformat()
        return payload


@dataclass(frozen=True)
class ClaimAssessment:
    claim_id: str
    claim_key: str
    status: ClaimStatus
    prior_confidence: float
    direct_confidence: float
    propagated_confidence: float
    support_score: float
    contradiction_score: float
    evidence_count: int
    independent_source_count: int
    dependency_failures: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def confidence(self) -> float:
        return self.propagated_confidence


def build_claim(
    *,
    claim_key: str,
    statement: str,
    kind: ClaimKind = ClaimKind.HYPOTHESIS,
    prior_confidence: float = 0.5,
    created_at: datetime | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    supersedes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> ClaimNode:
    key = normalize_text(claim_key)
    text = normalize_text(statement)
    if not key:
        raise ValueError("claim_key is required")
    if not text:
        raise ValueError("statement is required")
    prior = max(0.0, min(1.0, float(prior_confidence)))
    claim_id = _stable_id(
        "clm",
        {
            "claim_key": key,
            "statement": text,
            "kind": ClaimKind(kind).value,
            "valid_from": utc(valid_from).isoformat() if valid_from else None,
            "supersedes": supersedes,
        },
    )
    return ClaimNode(
        claim_id=claim_id,
        claim_key=key,
        statement=text,
        kind=ClaimKind(kind),
        prior_confidence=prior,
        created_at=utc(created_at),
        valid_from=utc(valid_from) if valid_from else None,
        valid_until=utc(valid_until) if valid_until else None,
        supersedes=supersedes,
        metadata=dict(metadata or {}),
    )


class ClaimGraph:
    """Inspectable directed graph connecting claims and evidence."""

    def __init__(self) -> None:
        self._claims: dict[str, ClaimNode] = {}
        self._edges: dict[str, ClaimEdge] = {}
        self._evidence_links: dict[str, ClaimEvidenceLink] = {}

    @property
    def claims(self) -> tuple[ClaimNode, ...]:
        return tuple(self._claims.values())

    @property
    def edges(self) -> tuple[ClaimEdge, ...]:
        return tuple(self._edges.values())

    @property
    def evidence_links(self) -> tuple[ClaimEvidenceLink, ...]:
        return tuple(self._evidence_links.values())

    def get_claim(self, claim_id: str) -> ClaimNode:
        try:
            return self._claims[claim_id]
        except KeyError as exc:
            raise KeyError(f"unknown claim_id: {claim_id}") from exc

    def register_claim(self, claim: ClaimNode) -> ClaimNode:
        existing = self._claims.get(claim.claim_id)
        if existing is not None:
            same_identity = (
                existing.claim_key == claim.claim_key
                and existing.statement == claim.statement
                and existing.kind is claim.kind
                and existing.prior_confidence == claim.prior_confidence
                and existing.valid_from == claim.valid_from
                and existing.valid_until == claim.valid_until
                and existing.supersedes == claim.supersedes
            )
            if not same_identity:
                raise ValueError(f"claim identity collision: {claim.claim_id}")
            return existing
        if claim.supersedes and claim.supersedes not in self._claims:
            raise ValueError(
                f"superseded claim must exist first: {claim.supersedes}"
            )
        self._claims[claim.claim_id] = claim
        return claim

    def connect(
        self,
        source_claim_id: str,
        target_claim_id: str,
        relation: ClaimRelation,
        *,
        weight: float = 1.0,
        required: bool = False,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClaimEdge:
        if source_claim_id not in self._claims:
            raise ValueError(f"unknown source claim: {source_claim_id}")
        if target_claim_id not in self._claims:
            raise ValueError(f"unknown target claim: {target_claim_id}")
        if source_claim_id == target_claim_id:
            raise ValueError("self-referential claim edge is not allowed")
        relation = ClaimRelation(relation)
        weight = max(0.0, min(1.0, float(weight)))
        if relation in _ACYCLIC_RELATIONS and self._path_exists(
            target_claim_id,
            source_claim_id,
            relations=_ACYCLIC_RELATIONS,
        ):
            raise ValueError("claim edge would create a reasoning cycle")
        edge_id = _stable_id(
            "edge",
            {
                "source": source_claim_id,
                "target": target_claim_id,
                "relation": relation.value,
            },
        )
        edge = ClaimEdge(
            edge_id=edge_id,
            source_claim_id=source_claim_id,
            target_claim_id=target_claim_id,
            relation=relation,
            weight=weight,
            required=bool(required),
            created_at=utc(created_at),
            metadata=dict(metadata or {}),
        )
        existing = self._edges.get(edge_id)
        if existing is not None:
            if existing != edge:
                raise ValueError(f"claim edge identity collision: {edge_id}")
            return existing
        self._edges[edge_id] = edge
        return edge

    def link_evidence(
        self,
        *,
        claim_id: str,
        evidence_id: str,
        ledger: EvidenceLedger,
        relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
        weight: float = 1.0,
        created_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ClaimEvidenceLink:
        if claim_id not in self._claims:
            raise ValueError(f"unknown claim: {claim_id}")
        record_by_id = {record.evidence_id: record for record in ledger.records}
        if evidence_id not in record_by_id:
            raise ValueError(f"unknown evidence_id: {evidence_id}")
        relation = EvidenceRelation(relation)
        weight = max(0.0, min(1.0, float(weight)))
        link_id = _stable_id(
            "lnk",
            {
                "claim": claim_id,
                "evidence": evidence_id,
                "relation": relation.value,
            },
        )
        link = ClaimEvidenceLink(
            link_id=link_id,
            claim_id=claim_id,
            evidence_id=evidence_id,
            relation=relation,
            weight=weight,
            created_at=utc(created_at),
            metadata=dict(metadata or {}),
        )
        existing = self._evidence_links.get(link_id)
        if existing is not None:
            if existing != link:
                raise ValueError(f"evidence link identity collision: {link_id}")
            return existing
        self._evidence_links[link_id] = link
        return link

    def incoming(self, claim_id: str) -> list[ClaimEdge]:
        return [
            edge
            for edge in self._edges.values()
            if edge.target_claim_id == claim_id
        ]

    def outgoing(self, claim_id: str) -> list[ClaimEdge]:
        return [
            edge
            for edge in self._edges.values()
            if edge.source_claim_id == claim_id
        ]

    def links_for_claim(self, claim_id: str) -> list[ClaimEvidenceLink]:
        return [
            link
            for link in self._evidence_links.values()
            if link.claim_id == claim_id
        ]

    def impact_set(self, claim_id: str) -> tuple[str, ...]:
        """Claims downstream from a claim if it changes or is falsified."""
        seen: set[str] = set()
        stack = [claim_id]
        while stack:
            current = stack.pop()
            for edge in self.outgoing(current):
                if edge.relation not in {
                    ClaimRelation.SUPPORTS,
                    ClaimRelation.DEPENDS_ON,
                    ClaimRelation.CAUSED_BY,
                    ClaimRelation.IMPLIES,
                    ClaimRelation.INVALIDATES,
                    ClaimRelation.REFINES,
                }:
                    continue
                target = edge.target_claim_id
                if target not in seen and target != claim_id:
                    seen.add(target)
                    stack.append(target)
        return tuple(sorted(seen))

    def assess(
        self,
        claim_id: str,
        ledger: EvidenceLedger,
        *,
        as_of: datetime | None = None,
    ) -> ClaimAssessment:
        memo: dict[str, ClaimAssessment] = {}
        return self._assess(
            claim_id,
            ledger,
            as_of=as_of,
            memo=memo,
            stack=set(),
        )

    def _assess(
        self,
        claim_id: str,
        ledger: EvidenceLedger,
        *,
        as_of: datetime | None,
        memo: dict[str, ClaimAssessment],
        stack: set[str],
    ) -> ClaimAssessment:
        if claim_id in memo:
            return memo[claim_id]
        if claim_id in stack:
            raise ValueError("cycle encountered during claim assessment")
        claim = self.get_claim(claim_id)
        stack = set(stack)
        stack.add(claim_id)

        record_by_id = {
            record.evidence_id: record
            for record in ledger.records
            if as_of is None or record.recorded_at <= utc(as_of)
        }

        best_by_independence: dict[str, tuple[ClaimEvidenceLink, Any]] = {}
        for link in self.links_for_claim(claim_id):
            record = record_by_id.get(link.evidence_id)
            if record is None:
                continue
            source = ledger.source_for(record)
            key = source.independence_key
            strength = record.confidence * link.weight
            current = best_by_independence.get(key)
            if current is None:
                best_by_independence[key] = (link, record)
                continue
            current_strength = current[1].confidence * current[0].weight
            if strength > current_strength:
                best_by_independence[key] = (link, record)

        direct_support = 0.0
        direct_contradiction = 0.0
        direct_count = 0
        reasons: list[str] = []
        for link, record in best_by_independence.values():
            strength = record.confidence * link.weight
            if link.relation is EvidenceRelation.SUPPORTS:
                direct_support += strength
            elif link.relation is EvidenceRelation.CONTRADICTS:
                direct_contradiction += strength
            direct_count += 1

        direct_total = direct_support + direct_contradiction
        if direct_total > 0:
            direct_confidence = (
                claim.prior_confidence + direct_support
            ) / (1.0 + direct_total)
        else:
            direct_confidence = claim.prior_confidence

        graph_support = 0.0
        graph_contradiction = 0.0
        dependency_failures: list[str] = []
        for edge in self.incoming(claim_id):
            source_assessment = self._assess(
                edge.source_claim_id,
                ledger,
                as_of=as_of,
                memo=memo,
                stack=stack,
            )
            strength = edge.weight * source_assessment.confidence
            if edge.relation in {
                ClaimRelation.SUPPORTS,
                ClaimRelation.CAUSED_BY,
                ClaimRelation.IMPLIES,
                ClaimRelation.REFINES,
            }:
                graph_support += strength
            elif edge.relation in {
                ClaimRelation.CONTRADICTS,
                ClaimRelation.INVALIDATES,
                ClaimRelation.ALTERNATIVE_TO,
            }:
                graph_contradiction += strength
            elif edge.relation is ClaimRelation.DEPENDS_ON:
                graph_support += strength
                if (
                    edge.required
                    and (
                        source_assessment.status is ClaimStatus.FALSIFIED
                        or source_assessment.confidence < 0.35
                    )
                ):
                    dependency_failures.append(edge.source_claim_id)

        support_score = direct_support + graph_support
        contradiction_score = direct_contradiction + graph_contradiction
        propagated = (
            claim.prior_confidence + support_score
        ) / (1.0 + support_score + contradiction_score)

        if dependency_failures:
            propagated *= 0.35
            reasons.append(
                "required_dependency_failed:"
                + ",".join(sorted(dependency_failures))
            )

        propagated = round(max(0.0, min(1.0, propagated)), 4)
        direct_confidence = round(
            max(0.0, min(1.0, direct_confidence)),
            4,
        )

        has_inputs = bool(direct_count or self.incoming(claim_id))
        if claim.supersedes:
            reasons.append(f"supersedes:{claim.supersedes}")
        if not has_inputs:
            status = ClaimStatus.INSUFFICIENT_EVIDENCE
        elif propagated <= 0.20:
            status = ClaimStatus.FALSIFIED
        elif contradiction_score > 0 and propagated < 0.55:
            status = ClaimStatus.CONTESTED
        elif propagated >= 0.70 and support_score > 0:
            status = ClaimStatus.SUPPORTED
        else:
            status = ClaimStatus.OPEN

        assessment = ClaimAssessment(
            claim_id=claim.claim_id,
            claim_key=claim.claim_key,
            status=status,
            prior_confidence=claim.prior_confidence,
            direct_confidence=direct_confidence,
            propagated_confidence=propagated,
            support_score=round(support_score, 4),
            contradiction_score=round(contradiction_score, 4),
            evidence_count=direct_count,
            independent_source_count=len(best_by_independence),
            dependency_failures=tuple(sorted(dependency_failures)),
            reasons=tuple(reasons),
        )
        memo[claim_id] = assessment
        return assessment

    def _path_exists(
        self,
        start: str,
        target: str,
        *,
        relations: set[ClaimRelation],
    ) -> bool:
        seen: set[str] = set()
        stack = [start]
        while stack:
            current = stack.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            for edge in self.outgoing(current):
                if edge.relation in relations:
                    stack.append(edge.target_claim_id)
        return False
