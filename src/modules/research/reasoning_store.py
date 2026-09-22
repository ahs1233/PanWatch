"""Persistence bridge for Claim Graph and Falsification Engine."""

from __future__ import annotations

from sqlalchemy.orm import Session

from src.platform.persistence.models import (
    ResearchClaimEdgeRecord,
    ResearchClaimEvidenceLinkRecord,
    ResearchClaimRecord,
    ResearchFalsificationRuleRecord,
)

from .claim_graph import (
    ClaimGraph,
    ClaimKind,
    ClaimNode,
    ClaimRelation,
)
from .evidence import EvidenceRelation, ObservationKind, utc
from .falsification import (
    FalsificationEngine,
    FalsificationRule,
    FalsificationRuleType,
)
from .ledger import EvidenceLedger


def persist_claim_graph(db: Session, graph: ClaimGraph) -> dict[str, int]:
    claims_inserted = 0
    edges_inserted = 0
    links_inserted = 0

    for claim in graph.claims:
        existing = db.get(ResearchClaimRecord, claim.claim_id)
        if existing is not None:
            if (
                existing.claim_key != claim.claim_key
                or existing.statement != claim.statement
                or existing.kind != claim.kind.value
            ):
                raise ValueError(
                    f"claim identity collision for {claim.claim_id}"
                )
            continue
        db.add(
            ResearchClaimRecord(
                claim_id=claim.claim_id,
                claim_key=claim.claim_key,
                statement=claim.statement,
                kind=claim.kind.value,
                prior_confidence=claim.prior_confidence,
                created_at=claim.created_at,
                valid_from=claim.valid_from,
                valid_until=claim.valid_until,
                supersedes=claim.supersedes,
                meta=claim.metadata,
            )
        )
        claims_inserted += 1

    db.flush()

    for edge in graph.edges:
        existing = db.get(ResearchClaimEdgeRecord, edge.edge_id)
        if existing is not None:
            if (
                existing.source_claim_id != edge.source_claim_id
                or existing.target_claim_id != edge.target_claim_id
                or existing.relation != edge.relation.value
            ):
                raise ValueError(
                    f"claim edge identity collision for {edge.edge_id}"
                )
            continue
        db.add(
            ResearchClaimEdgeRecord(
                edge_id=edge.edge_id,
                source_claim_id=edge.source_claim_id,
                target_claim_id=edge.target_claim_id,
                relation=edge.relation.value,
                weight=edge.weight,
                required=edge.required,
                created_at=edge.created_at,
                meta=edge.metadata,
            )
        )
        edges_inserted += 1

    db.flush()

    for link in graph.evidence_links:
        existing = db.get(ResearchClaimEvidenceLinkRecord, link.link_id)
        if existing is not None:
            if (
                existing.claim_id != link.claim_id
                or existing.evidence_id != link.evidence_id
                or existing.relation != link.relation.value
            ):
                raise ValueError(
                    f"claim evidence link identity collision for {link.link_id}"
                )
            continue
        db.add(
            ResearchClaimEvidenceLinkRecord(
                link_id=link.link_id,
                claim_id=link.claim_id,
                evidence_id=link.evidence_id,
                relation=link.relation.value,
                weight=link.weight,
                created_at=link.created_at,
                meta=link.metadata,
            )
        )
        links_inserted += 1

    db.commit()
    return {
        "claims_inserted": claims_inserted,
        "edges_inserted": edges_inserted,
        "links_inserted": links_inserted,
    }


def persist_falsification_engine(
    db: Session,
    engine: FalsificationEngine,
) -> int:
    inserted = 0
    for rule in engine.rules:
        existing = db.get(ResearchFalsificationRuleRecord, rule.rule_id)
        if existing is not None:
            if (
                existing.claim_id != rule.claim_id
                or existing.rule_type != rule.rule_type.value
                or existing.description != rule.description
            ):
                raise ValueError(
                    f"falsification rule identity collision for {rule.rule_id}"
                )
            continue
        db.add(
            ResearchFalsificationRuleRecord(
                rule_id=rule.rule_id,
                claim_id=rule.claim_id,
                description=rule.description,
                rule_type=rule.rule_type.value,
                hard_fail=rule.hard_fail,
                weight=rule.weight,
                evidence_claim_key=rule.evidence_claim_key,
                operator=rule.operator,
                threshold=rule.threshold,
                min_sources=rule.min_sources,
                max_age_seconds=rule.max_age_seconds,
                related_claim_id=rule.related_claim_id,
                required_kinds=[
                    kind.value for kind in rule.required_kinds
                ],
                meta=rule.metadata,
            )
        )
        inserted += 1
    db.commit()
    return inserted


def load_claim_graph(
    db: Session,
    *,
    ledger: EvidenceLedger,
) -> ClaimGraph:
    graph = ClaimGraph()
    claims = (
        db.query(ResearchClaimRecord)
        .order_by(ResearchClaimRecord.created_at.asc())
        .all()
    )

    pending = {row.claim_id: row for row in claims}
    while pending:
        progressed = False
        for claim_id, row in list(pending.items()):
            if row.supersedes and row.supersedes not in {
                claim.claim_id for claim in graph.claims
            }:
                continue
            graph.register_claim(
                ClaimNode(
                    claim_id=row.claim_id,
                    claim_key=row.claim_key,
                    statement=row.statement,
                    kind=ClaimKind(row.kind),
                    prior_confidence=float(row.prior_confidence),
                    created_at=utc(row.created_at),
                    valid_from=(
                        utc(row.valid_from)
                        if row.valid_from is not None
                        else None
                    ),
                    valid_until=(
                        utc(row.valid_until)
                        if row.valid_until is not None
                        else None
                    ),
                    supersedes=row.supersedes,
                    metadata=dict(row.meta or {}),
                )
            )
            pending.pop(claim_id)
            progressed = True
        if not progressed:
            raise ValueError(
                "could not resolve persisted claim supersession order"
            )

    edge_rows = (
        db.query(ResearchClaimEdgeRecord)
        .order_by(ResearchClaimEdgeRecord.created_at.asc())
        .all()
    )
    for row in edge_rows:
        edge = graph.connect(
            row.source_claim_id,
            row.target_claim_id,
            ClaimRelation(row.relation),
            weight=float(row.weight),
            required=bool(row.required),
            created_at=utc(row.created_at),
            metadata=dict(row.meta or {}),
        )
        if edge.edge_id != row.edge_id:
            raise ValueError(
                f"persisted edge id mismatch: {row.edge_id}"
            )

    evidence_ids = {record.evidence_id for record in ledger.records}
    link_rows = (
        db.query(ResearchClaimEvidenceLinkRecord)
        .order_by(ResearchClaimEvidenceLinkRecord.created_at.asc())
        .all()
    )
    for row in link_rows:
        if row.evidence_id not in evidence_ids:
            continue
        link = graph.link_evidence(
            claim_id=row.claim_id,
            evidence_id=row.evidence_id,
            ledger=ledger,
            relation=EvidenceRelation(row.relation),
            weight=float(row.weight),
            created_at=utc(row.created_at),
            metadata=dict(row.meta or {}),
        )
        if link.link_id != row.link_id:
            raise ValueError(
                f"persisted evidence-link id mismatch: {row.link_id}"
            )

    return graph


def load_falsification_engine(
    db: Session,
    *,
    graph: ClaimGraph,
) -> FalsificationEngine:
    engine = FalsificationEngine()
    rows = (
        db.query(ResearchFalsificationRuleRecord)
        .order_by(ResearchFalsificationRuleRecord.created_at.asc())
        .all()
    )
    for row in rows:
        rule = FalsificationRule(
            rule_id=row.rule_id,
            claim_id=row.claim_id,
            description=row.description,
            rule_type=FalsificationRuleType(row.rule_type),
            hard_fail=bool(row.hard_fail),
            weight=float(row.weight),
            evidence_claim_key=row.evidence_claim_key or "",
            operator=row.operator or "",
            threshold=(
                float(row.threshold)
                if row.threshold is not None
                else None
            ),
            min_sources=int(row.min_sources or 1),
            max_age_seconds=(
                int(row.max_age_seconds)
                if row.max_age_seconds is not None
                else None
            ),
            related_claim_id=row.related_claim_id,
            required_kinds=tuple(
                ObservationKind(value)
                for value in (row.required_kinds or [])
            ),
            metadata=dict(row.meta or {}),
        )
        engine.register_rule(rule, graph=graph)
    return engine
