"""Feature snapshot adapter for GEN1.1 research replay.

Only point-in-time inputs already visible to the GEN1 decision stack are copied.
No future labels or outcome fields are added here.
"""
from __future__ import annotations

from typing import Any


def session_name_utc(observed_at) -> str:
    hour = int(observed_at.hour)
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def build_gen11_feature_snapshot(
    technical: dict[str, Any],
    cognition: dict[str, Any],
    fusion: dict[str, Any],
    evidence: dict[str, Any],
    macro: dict[str, Any],
    *,
    observed_at,
    previous_session: str | None = None,
) -> dict[str, Any]:
    current_session = session_name_utc(observed_at)
    directional = dict(evidence.get("directional_evidence") or {})
    context = dict(technical.get("market_context") or {})
    frames = {
        str(name): dict(payload or {})
        for name, payload in dict(technical.get("frames") or {}).items()
    }
    return {
        "version": "gen1.1-feature-snapshot-v1",
        "lookahead_protected": True,
        "observed_at": observed_at.isoformat(),
        "session": current_session,
        "session_transition": bool(previous_session and current_session != previous_session),
        "previous_session": previous_session,
        "alignment": technical.get("alignment"),
        "frames": frames,
        "market_context": context,
        "macro": {
            "bias": macro.get("bias"),
            "bias_label": macro.get("bias_label"),
            "confidence": macro.get("confidence"),
            "proxy_score": macro.get("proxy_score"),
            "drivers": list(macro.get("drivers") or []),
            "observed_at": macro.get("observed_at"),
        },
        "cognition": {
            "regime": dict(cognition.get("regime") or {}),
            "market_state": dict(cognition.get("market_state") or {}),
            "confidence": dict(cognition.get("confidence") or {}),
            "directional_state": dict(cognition.get("directional_state") or {}),
            "execution_plan": dict(cognition.get("execution_plan") or {}),
        },
        "fusion": {
            "state": fusion.get("state"),
            "paper_entry_allowed": fusion.get("paper_entry_allowed"),
            "cognitive_confidence": fusion.get("cognitive_confidence"),
        },
        "evidence_score": evidence.get("score"),
        "evidence_direction": evidence.get("direction"),
        "evidence_coverage": evidence.get("coverage"),
        "evidence_agreement": evidence.get("agreement_ratio"),
        "evidence_conflict_score": directional.get("conflict_score"),
        "directional_confidence": directional.get("confidence"),
        "evidence_dominant_side": directional.get("dominant_side"),
        "evidence_families": list(evidence.get("families") or []),
        "feature_only": True,
        "outcome_fields_present": False,
    }
