"""XAU production bridge into the domain-neutral Research Engine.

This module is only an adapter. The research core remains domain-neutral.
It converts the current XAU macro synthesis into a falsifiable claim with
explicit provenance so AutomaticResearchLoop has real production claims to test.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .claim_graph import ClaimKind, build_claim
from .evidence import (
    EvidenceRelation,
    ObservationKind,
    SourceTier,
    build_evidence,
    build_source,
)
from .evidence_store import load_full_ledger, persist_ledger
from .falsification import (
    FalsificationRuleType,
    build_falsification_rule,
)
from .reasoning_store import (
    load_claim_graph,
    load_falsification_engine,
    persist_claim_graph,
    persist_falsification_engine,
)
from .research_store import research_session


_XAU_MACRO_KEY = "xau.macro.directional_bias"


def _latest_claim_for_key(graph, claim_key: str):
    rows = [claim for claim in graph.claims if claim.claim_key == claim_key]
    if not rows:
        return None
    return max(rows, key=lambda item: (item.created_at, item.claim_id))


async def sync_xau_macro_claim() -> dict[str, Any]:
    """Persist the current XAU macro thesis as a falsifiable durable claim."""
    from src.modules.xau.service import get_macro_context

    macro = await get_macro_context(force=True)
    bias = int(max(-1, min(1, int(macro.get("bias", 0) or 0))))
    label = "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral"
    summary = str(macro.get("summary") or "").strip()
    observed_raw = str(macro.get("observed_at") or "")
    try:
        observed_at = datetime.fromisoformat(observed_raw.replace("Z", "+00:00"))
    except ValueError:
        observed_at = datetime.now(timezone.utc)
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=timezone.utc)
    observed_at = observed_at.astimezone(timezone.utc)

    statement = f"The current macro backdrop for XAU/USD is {label}."

    with research_session() as db:
        ledger = load_full_ledger(db)
        graph = load_claim_graph(db, ledger=ledger)
        falsification = load_falsification_engine(db, graph=graph)

        previous = _latest_claim_for_key(graph, _XAU_MACRO_KEY)
        supersedes = None
        if previous is not None and previous.statement != statement:
            supersedes = previous.claim_id

        claim = build_claim(
            claim_key=_XAU_MACRO_KEY,
            statement=statement,
            kind=ClaimKind.HYPOTHESIS,
            prior_confidence=0.5,
            created_at=observed_at,
            supersedes=supersedes,
            metadata={
                "domain": "xau",
                "bridge": "xau_macro",
                "time_sensitive": True,
            },
        )
        if previous is not None and previous.statement == statement:
            claim = previous
        else:
            graph.register_claim(claim)

        source_payload = {
            "observed_at": observed_at.isoformat(),
            "bias": bias,
            "bias_label": label,
            "confidence": macro.get("confidence"),
            "summary": summary,
            "drivers": macro.get("drivers") or [],
            "event_risk": bool(macro.get("event_risk")),
            "event_name": macro.get("event_name"),
            "event_kind": macro.get("event_kind"),
            "search_ok": bool(macro.get("search_ok")),
            "synthesis_ok": bool(macro.get("synthesis_ok")),
        }
        source_text = json.dumps(
            source_payload,
            sort_keys=True,
            ensure_ascii=False,
        )
        source = build_source(
            url="https://panwatch.internal/xau/macro",
            content=source_text,
            publisher="PanWatch",
            title="XAU macro synthesis",
            source_tier=SourceTier.AGGREGATOR,
            source_family="panwatch_internal_macro",
            independence_key="panwatch_internal_macro",
            published_at=observed_at,
            retrieved_at=observed_at,
            observed_at=observed_at,
            tool_name="xau_macro_bridge",
            metadata={
                "internal_derived": True,
                "search_ok": bool(macro.get("search_ok")),
                "synthesis_ok": bool(macro.get("synthesis_ok")),
            },
        )
        ledger.register_source(source)
        try:
            confidence = max(
                0.05,
                min(0.85, float(macro.get("confidence", 0.5) or 0.5)),
            )
        except (TypeError, ValueError):
            confidence = 0.5
        evidence = build_evidence(
            claim_key=_XAU_MACRO_KEY,
            source=source,
            statement=summary or statement,
            relation=EvidenceRelation.SUPPORTS,
            observation_kind=ObservationKind.ESTIMATE,
            event_time=observed_at,
            observed_at=observed_at,
            recorded_at=observed_at,
            confidence=confidence,
            metadata={
                "internal_derived": True,
                "macro_bias": bias,
                "macro_bias_label": label,
            },
        )
        inserted = ledger.append(evidence)
        graph.link_evidence(
            claim_id=claim.claim_id,
            evidence_id=evidence.evidence_id,
            ledger=ledger,
            relation=EvidenceRelation.SUPPORTS,
            weight=min(0.75, confidence),
            metadata={"bridge": "xau_macro"},
        )

        existing_rules = {
            rule.rule_type
            for rule in falsification.rules_for_claim(claim.claim_id)
        }
        if FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT not in existing_rules:
            falsification.register_rule(
                build_falsification_rule(
                    claim_id=claim.claim_id,
                    description=(
                        "The current XAU macro thesis requires at least two "
                        "independent source families."
                    ),
                    rule_type=FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT,
                    hard_fail=False,
                    weight=0.9,
                    evidence_claim_key=_XAU_MACRO_KEY,
                    min_sources=2,
                ),
                graph=graph,
            )
        if FalsificationRuleType.CONTRADICTORY_EVIDENCE not in existing_rules:
            falsification.register_rule(
                build_falsification_rule(
                    claim_id=claim.claim_id,
                    description=(
                        "Independent current evidence may contradict the "
                        "directional XAU macro thesis."
                    ),
                    rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
                    hard_fail=False,
                    weight=1.0,
                    min_sources=1,
                ),
                graph=graph,
            )
        if FalsificationRuleType.FRESHNESS_FAILURE not in existing_rules:
            falsification.register_rule(
                build_falsification_rule(
                    claim_id=claim.claim_id,
                    description=(
                        "The XAU macro thesis must be supported by evidence "
                        "observed within the last 30 minutes."
                    ),
                    rule_type=FalsificationRuleType.FRESHNESS_FAILURE,
                    hard_fail=True,
                    weight=1.0,
                    evidence_claim_key=_XAU_MACRO_KEY,
                    max_age_seconds=1800,
                ),
                graph=graph,
            )

        persist_ledger(db, ledger)
        persist_claim_graph(db, graph)
        persist_falsification_engine(db, falsification)

        return {
            "claim_id": claim.claim_id,
            "claim_key": claim.claim_key,
            "statement": claim.statement,
            "bias": label,
            "evidence_inserted": bool(inserted),
            "supersedes": supersedes,
            "search_ok": bool(macro.get("search_ok")),
            "synthesis_ok": bool(macro.get("synthesis_ok")),
        }
