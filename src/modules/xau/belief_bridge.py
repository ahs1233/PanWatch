"""Bridge live XAU research state into the generic Ahmed Research Engine.

The bridge is deliberately deterministic and side-effect limited:
- it consumes technical/macro/fusion state already computed by PanWatch;
- it never calls an LLM, web search, broker or execution venue;
- it writes only Evidence/Claim/Falsification/Belief records;
- claim identities remain stable while evidence observations evolve over time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.modules.research import (
    ClaimGraph,
    ClaimKind,
    ClaimRelation,
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
from src.modules.research.evidence_store import persist_ledger
from src.modules.research.reasoning_store import (
    persist_claim_graph,
    persist_falsification_engine,
)
from src.modules.research.research_store import research_session


TECHNICAL_CLAIM_KEY = "xau.runtime.technical_ready"
MACRO_CLAIM_KEY = "xau.runtime.macro_ready"
THESIS_CLAIM_KEY = "xau.runtime.directional_thesis"

TECHNICAL_STATEMENT = (
    "Current XAU technical sensing is sufficiently reliable for directional research."
)
MACRO_STATEMENT = (
    "Current XAU macro context is sufficiently verified for directional research."
)
THESIS_STATEMENT = (
    "Current XAU decision stack supports an eligible directional research thesis."
)


@dataclass(frozen=True)
class XAUBeliefWriteResult:
    cycle_id: str
    changed_count: int
    falsified_count: int
    event_count: int
    probe_count: int
    claim_count: int
    thesis_status: str
    thesis_confidence: float
    evidence_inserted: int
    sources_inserted: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "changed_count": self.changed_count,
            "falsified_count": self.falsified_count,
            "event_count": self.event_count,
            "probe_count": self.probe_count,
            "claim_count": self.claim_count,
            "thesis_status": self.thesis_status,
            "thesis_confidence": self.thesis_confidence,
            "evidence_inserted": self.evidence_inserted,
            "sources_inserted": self.sources_inserted,
        }


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return max(0.0, min(1.0, float(default)))


def _confidence_bucket(value: Any) -> str:
    score = _confidence(value, 0.0)
    if score >= 0.80:
        return "very_high"
    if score >= 0.65:
        return "high"
    if score >= 0.50:
        return "medium"
    if score >= 0.30:
        return "low"
    return "very_low"


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def xau_material_signature(
    technical: dict[str, Any],
    macro: dict[str, Any],
    fusion: dict[str, Any],
) -> tuple[str, ...]:
    """Categorical signature used to suppress write/call storms.

    Prices and small confidence movements are intentionally excluded. Confidence
    enters as a coarse bucket, so only material calibration changes create a new
    belief cycle.
    """
    micro = technical.get("micro") or {}
    return (
        str(technical.get("status") or ""),
        str(technical.get("candidate") or ""),
        str(technical.get("alignment") or ""),
        str(technical.get("technical_mode") or ""),
        str(bool(technical.get("blocked"))),
        str(micro.get("direction") or ""),
        str(fusion.get("state") or ""),
        str(fusion.get("base_state") or ""),
        str(fusion.get("macro_relation") or ""),
        str(fusion.get("regime") or ""),
        str(fusion.get("meta_decision") or ""),
        str(bool(fusion.get("paper_entry_allowed"))),
        str(bool(fusion.get("event_risk"))),
        str(fusion.get("execution_status") or ""),
        _confidence_bucket(fusion.get("cognitive_confidence")),
        str(bool(macro.get("calendar_ok"))),
        str(bool(macro.get("search_ok"))),
        str(bool(macro.get("synthesis_ok"))),
        str(macro.get("bias_label") or ""),
    )


def xau_belief_persistence_ready(macro: dict[str, Any]) -> bool:
    """Ignore short cache-refresh transitions rather than memorialising noise."""
    return not bool(
        macro.get("refresh_pending")
        and macro.get("cache_stale")
    )


def _build_state(
    technical: dict[str, Any],
    macro: dict[str, Any],
    fusion: dict[str, Any],
    *,
    observed_at: datetime,
) -> tuple[
    EvidenceLedger,
    ClaimGraph,
    FalsificationEngine,
    str,
]:
    ledger = EvidenceLedger()
    graph = ClaimGraph()
    falsification = FalsificationEngine()

    cognition = fusion.get("cognition") or {}
    data_quality = (cognition.get("data_quality") or {}).get("score")
    technical_ready = not bool(technical.get("blocked"))
    macro_ready = bool(fusion.get("macro_ready"))
    candidate = str(fusion.get("technical_candidate") or "none")
    thesis_eligible = bool(
        fusion.get("paper_entry_allowed")
        and candidate in {"long_setup", "short_setup"}
        and not fusion.get("event_risk")
        and fusion.get("execution_allowed") is False
    )

    technical_payload = {
        "status": technical.get("status"),
        "candidate": technical.get("candidate"),
        "alignment": technical.get("alignment"),
        "technical_mode": technical.get("technical_mode"),
        "blocked": bool(technical.get("blocked")),
        "block_reasons": list(technical.get("block_reasons") or []),
        "micro_direction": (technical.get("micro") or {}).get("direction"),
        "data_quality_bucket": _confidence_bucket(data_quality),
    }
    macro_payload = {
        "macro_ready": macro_ready,
        "bias": macro.get("bias"),
        "bias_label": macro.get("bias_label"),
        "confidence_bucket": _confidence_bucket(macro.get("confidence")),
        "event_risk": bool(macro.get("event_risk")),
        "event_kind": macro.get("event_kind"),
        "event_name": macro.get("event_name"),
        "event_validation": macro.get("event_validation"),
        "calendar_ok": bool(macro.get("calendar_ok")),
        "search_ok": bool(macro.get("search_ok")),
        "synthesis_ok": bool(macro.get("synthesis_ok")),
    }
    thesis_payload = {
        "eligible": thesis_eligible,
        "candidate": candidate,
        "state": fusion.get("state"),
        "base_state": fusion.get("base_state"),
        "macro_relation": fusion.get("macro_relation"),
        "regime": fusion.get("regime"),
        "meta_decision": fusion.get("meta_decision"),
        "event_risk": bool(fusion.get("event_risk")),
        "confidence_bucket": _confidence_bucket(
            fusion.get("cognitive_confidence")
        ),
        "execution_allowed": bool(fusion.get("execution_allowed")),
        "execution_status": fusion.get("execution_status"),
        "reasons": list(fusion.get("reasons") or []),
    }

    sources = {
        "technical": build_source(
            url="panwatch://xau/technical-state",
            content=_canonical(technical_payload),
            publisher="PanWatch XAU Technical Engine",
            title="Derived XAU technical research state",
            source_tier=SourceTier.UNKNOWN,
            source_family="panwatch_xau_technical",
            independence_key="panwatch_xau_technical",
            retrieved_at=observed_at,
            observed_at=observed_at,
            tool_name="XAUIntradayEngine",
            metadata={
                "derived": True,
                "execution_eligible": False,
                "payload": technical_payload,
            },
        ),
        "macro": build_source(
            url="panwatch://xau/macro-state",
            content=_canonical(macro_payload),
            publisher="PanWatch XAU Macro Research",
            title="Derived XAU macro research state",
            source_tier=SourceTier.UNKNOWN,
            source_family="panwatch_xau_macro",
            independence_key="panwatch_xau_macro",
            retrieved_at=observed_at,
            observed_at=observed_at,
            tool_name="get_macro_context",
            metadata={
                "derived": True,
                "query": str(macro.get("query") or ""),
                "summary": str(macro.get("summary") or ""),
                "drivers": list(macro.get("drivers") or []),
                "event_source_url": macro.get("event_source_url"),
                "payload": macro_payload,
            },
        ),
        "fusion": build_source(
            url="panwatch://xau/decision-fusion",
            content=_canonical(thesis_payload),
            publisher="PanWatch XAU Decision Fusion",
            title="Derived XAU decision fusion state",
            source_tier=SourceTier.UNKNOWN,
            source_family="panwatch_xau_fusion",
            independence_key="panwatch_xau_fusion",
            retrieved_at=observed_at,
            observed_at=observed_at,
            tool_name="build_decision_fusion",
            metadata={
                "derived": True,
                "execution_eligible": False,
                "payload": thesis_payload,
            },
        ),
    }
    for source in sources.values():
        ledger.register_source(source)

    period = observed_at.isoformat()

    technical_evidence = build_evidence(
        claim_key=TECHNICAL_CLAIM_KEY,
        source=sources["technical"],
        statement=(
            "Technical sensing is ready for directional research"
            if technical_ready
            else "Technical sensing is blocked or unreliable for directional research"
        )
        + f"; status={technical_payload['status']}; "
        + f"candidate={technical_payload['candidate']}; "
        + f"alignment={technical_payload['alignment']}.",
        relation=(
            EvidenceRelation.SUPPORTS
            if technical_ready
            else EvidenceRelation.CONTRADICTS
        ),
        observation_kind=ObservationKind.MARKET_PRICING,
        event_time=observed_at,
        observed_at=observed_at,
        recorded_at=observed_at,
        confidence=(
            _confidence(data_quality, 0.80)
            if technical_ready
            else 1.0
        ),
        period=period,
        metadata=technical_payload,
    )
    ledger.append(technical_evidence)

    macro_evidence = build_evidence(
        claim_key=MACRO_CLAIM_KEY,
        source=sources["macro"],
        statement=(
            "Macro context is verified for directional research"
            if macro_ready
            else "Macro context is not sufficiently verified for directional research"
        )
        + f"; bias={macro_payload['bias_label']}; "
        + f"event_risk={macro_payload['event_risk']}.",
        relation=(
            EvidenceRelation.SUPPORTS
            if macro_ready
            else EvidenceRelation.CONTRADICTS
        ),
        observation_kind=ObservationKind.ESTIMATE,
        event_time=observed_at,
        observed_at=observed_at,
        recorded_at=observed_at,
        confidence=(
            max(0.55, _confidence(macro.get("confidence"), 0.55))
            if macro_ready
            else 1.0
        ),
        period=period,
        metadata=macro_payload,
    )
    ledger.append(macro_evidence)

    if candidate in {"long_setup", "short_setup"}:
        thesis_relation = (
            EvidenceRelation.SUPPORTS
            if thesis_eligible
            else EvidenceRelation.CONTRADICTS
        )
        thesis_statement = (
            "Eligible directional thesis"
            if thesis_eligible
            else "Directional setup exists but is not eligible"
        )
    else:
        thesis_relation = EvidenceRelation.CONTEXT
        thesis_statement = "No directional setup is currently present"

    thesis_evidence = build_evidence(
        claim_key=THESIS_CLAIM_KEY,
        source=sources["fusion"],
        statement=(
            f"{thesis_statement}; candidate={candidate}; "
            f"state={thesis_payload['state']}; "
            f"regime={thesis_payload['regime']}; "
            f"meta={thesis_payload['meta_decision']}."
        ),
        relation=thesis_relation,
        observation_kind=ObservationKind.ESTIMATE,
        event_time=observed_at,
        observed_at=observed_at,
        recorded_at=observed_at,
        confidence=max(
            0.55,
            _confidence(fusion.get("cognitive_confidence"), 0.55),
        ),
        period=period,
        metadata=thesis_payload,
    )
    ledger.append(thesis_evidence)

    technical_claim = build_claim(
        claim_key=TECHNICAL_CLAIM_KEY,
        statement=TECHNICAL_STATEMENT,
        kind=ClaimKind.INTERPRETATION,
        prior_confidence=0.50,
        created_at=observed_at,
        metadata={"domain": "xau", "role": "runtime_gate"},
    )
    macro_claim = build_claim(
        claim_key=MACRO_CLAIM_KEY,
        statement=MACRO_STATEMENT,
        kind=ClaimKind.INTERPRETATION,
        prior_confidence=0.50,
        created_at=observed_at,
        metadata={"domain": "xau", "role": "runtime_gate"},
    )
    thesis_claim = build_claim(
        claim_key=THESIS_CLAIM_KEY,
        statement=THESIS_STATEMENT,
        kind=ClaimKind.CONCLUSION,
        prior_confidence=0.50,
        created_at=observed_at,
        metadata={"domain": "xau", "role": "directional_thesis"},
    )

    for claim in (technical_claim, macro_claim, thesis_claim):
        graph.register_claim(claim)

    graph.link_evidence(
        claim_id=technical_claim.claim_id,
        evidence_id=technical_evidence.evidence_id,
        ledger=ledger,
        relation=technical_evidence.relation,
        created_at=observed_at,
    )
    graph.link_evidence(
        claim_id=macro_claim.claim_id,
        evidence_id=macro_evidence.evidence_id,
        ledger=ledger,
        relation=macro_evidence.relation,
        created_at=observed_at,
    )
    graph.link_evidence(
        claim_id=thesis_claim.claim_id,
        evidence_id=thesis_evidence.evidence_id,
        ledger=ledger,
        relation=thesis_evidence.relation,
        created_at=observed_at,
    )

    graph.connect(
        technical_claim.claim_id,
        thesis_claim.claim_id,
        ClaimRelation.DEPENDS_ON,
        weight=0.95,
        required=True,
        created_at=observed_at,
    )
    graph.connect(
        macro_claim.claim_id,
        thesis_claim.claim_id,
        ClaimRelation.DEPENDS_ON,
        weight=0.80,
        required=True,
        created_at=observed_at,
    )

    technical_rule = build_falsification_rule(
        claim_id=technical_claim.claim_id,
        description=(
            "Technical readiness fails when the current technical observation "
            "explicitly contradicts readiness."
        ),
        rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
        hard_fail=True,
        min_sources=1,
        metadata={"domain": "xau"},
    )
    macro_rule = build_falsification_rule(
        claim_id=macro_claim.claim_id,
        description=(
            "Macro readiness fails when the current macro observation "
            "explicitly contradicts readiness."
        ),
        rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
        hard_fail=True,
        min_sources=1,
        metadata={"domain": "xau"},
    )
    thesis_rule = build_falsification_rule(
        claim_id=thesis_claim.claim_id,
        description=(
            "The directional thesis fails when the current decision fusion "
            "explicitly rejects eligibility."
        ),
        rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
        hard_fail=True,
        min_sources=1,
        metadata={"domain": "xau"},
    )
    technical_dependency_rule = build_falsification_rule(
        claim_id=thesis_claim.claim_id,
        description="The directional thesis requires reliable technical sensing.",
        rule_type=FalsificationRuleType.DEPENDENCY_FAILURE,
        hard_fail=True,
        related_claim_id=technical_claim.claim_id,
        threshold=0.40,
        metadata={"domain": "xau"},
    )
    macro_dependency_rule = build_falsification_rule(
        claim_id=thesis_claim.claim_id,
        description="The directional thesis requires verified macro context.",
        rule_type=FalsificationRuleType.DEPENDENCY_FAILURE,
        hard_fail=True,
        related_claim_id=macro_claim.claim_id,
        threshold=0.40,
        metadata={"domain": "xau"},
    )
    for rule in (
        technical_rule,
        macro_rule,
        thesis_rule,
        technical_dependency_rule,
        macro_dependency_rule,
    ):
        falsification.register_rule(rule, graph=graph)

    return ledger, graph, falsification, thesis_claim.claim_id


def persist_xau_belief_state(
    technical: dict[str, Any],
    macro: dict[str, Any],
    fusion: dict[str, Any],
    *,
    observed_at: datetime | None = None,
) -> XAUBeliefWriteResult:
    """Persist one material XAU state transition into the durable belief store."""
    when = _utc(observed_at)
    ledger, graph, falsification, thesis_claim_id = _build_state(
        technical,
        macro,
        fusion,
        observed_at=when,
    )

    with research_session() as db:
        evidence_counts = persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)
        cycle = PanWatchBeliefMonitor(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
        ).run_cycle(
            db=db,
            evaluated_at=when,
            metadata={
                "domain": "xau",
                "source": "xau_live_scheduler",
                "execution_allowed": False,
                "material_signature": list(
                    xau_material_signature(technical, macro, fusion)
                ),
            },
        )

    thesis_update = next(
        item
        for item in cycle.updates
        if item.snapshot.claim_id == thesis_claim_id
    )
    return XAUBeliefWriteResult(
        cycle_id=cycle.cycle_id,
        changed_count=cycle.changed_count,
        falsified_count=cycle.falsified_count,
        event_count=len(cycle.events),
        probe_count=cycle.probe_count,
        claim_count=cycle.claim_count,
        thesis_status=thesis_update.snapshot.final_status.value,
        thesis_confidence=thesis_update.snapshot.final_confidence,
        evidence_inserted=int(evidence_counts["evidence_inserted"]),
        sources_inserted=int(evidence_counts["sources_inserted"]),
    )
