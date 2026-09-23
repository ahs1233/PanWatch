"""Compact no-lookahead feature snapshot for GEN1.1 research replay."""
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


def _pick(payload: dict[str, Any] | None, keys: tuple[str, ...]) -> dict[str, Any]:
    payload = dict(payload or {})
    return {key: payload.get(key) for key in keys}


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
    frames = {
        str(name): _pick(payload, (
            "close", "ema_fast", "ema_slow", "rsi14", "atr14", "atr_pct",
            "breakout", "direction", "recent_swing_high", "recent_swing_low",
        ))
        for name, payload in dict(technical.get("frames") or {}).items()
        if str(name) in {"1m", "5m", "15m"}
    }
    context = dict(technical.get("market_context") or {})
    bias = dict(context.get("bias") or {})
    compact_bias = {
        name: _pick(
            bias.get(name),
            ("direction", "score", "close", "ema", "slope_20", "available", "bar_count"),
        )
        for name in ("monthly", "weekly", "daily", "h4", "h1")
    }
    compact_bias.update({
        "composite_score": bias.get("composite_score"),
        "composite_direction": bias.get("composite_direction"),
        "today_score": bias.get("today_score"),
        "today_direction": bias.get("today_direction"),
    })
    flow = dict(context.get("cash_flow") or {})
    spot_flow = dict(flow.get("spot_tick") or {})
    profile = dict(context.get("volume_profile") or {})
    smart = dict(context.get("smart_money") or {})
    dealing = dict(smart.get("dealing_range") or {})
    liquidity = dict(context.get("liquidity") or {})
    levels = list(liquidity.get("levels") or [])[:4]
    directional = dict(evidence.get("directional_evidence") or {})
    confidence = dict(cognition.get("confidence") or {})
    execution_plan = dict(cognition.get("execution_plan") or {})
    directional_state = dict(cognition.get("directional_state") or {})

    return {
        "version": "gen1.1-feature-snapshot-v1",
        "lookahead_protected": True,
        "feature_only": True,
        "outcome_fields_present": False,
        "observed_at": observed_at.isoformat(),
        "session": current_session,
        "session_transition": bool(previous_session and current_session != previous_session),
        "previous_session": previous_session,
        "alignment": technical.get("alignment"),
        "frames": frames,
        "htf_bias": compact_bias,
        "flow": {
            "direction": flow.get("direction"),
            "score": flow.get("score"),
            "agreement": flow.get("agreement"),
            "cmf20": spot_flow.get("cmf20"),
            "signed_tick_volume_imbalance": spot_flow.get("signed_tick_volume_imbalance"),
            "obv_slope_proxy": spot_flow.get("obv_slope_proxy"),
        },
        "volume_profile": _pick(
            profile,
            ("available", "poc", "vah", "val", "location", "range_low", "range_high", "total_tick_volume"),
        ),
        "smart_money": {
            "bias": smart.get("bias"),
            "score": smart.get("score"),
            "break_of_structure": smart.get("break_of_structure"),
            "liquidity_sweep": smart.get("liquidity_sweep"),
            "displacement": smart.get("displacement"),
            "dealing_zone": dealing.get("zone"),
        },
        "liquidity": {
            "nearest_levels": [
                _pick(item, ("name", "price", "side", "distance"))
                for item in levels
            ],
            "equal_high_count": len(liquidity.get("equal_highs") or []),
            "equal_low_count": len(liquidity.get("equal_lows") or []),
            "tolerance": liquidity.get("tolerance"),
        },
        "macro": {
            "bias": macro.get("bias"),
            "bias_label": macro.get("bias_label"),
            "confidence": macro.get("confidence"),
            "proxy_score": macro.get("proxy_score"),
            "drivers": list(macro.get("drivers") or []),
            "observed_at": macro.get("observed_at"),
        },
        "cognition": {
            "calibrated_confidence": confidence.get("calibrated_confidence"),
            "regime": dict(cognition.get("regime") or {}),
            "execution_action": execution_plan.get("action"),
            "execution_side": execution_plan.get("side"),
            "directional_classification": directional_state.get("classification"),
            "directional_edge": directional_state.get("directional_edge"),
            "source_conflicts": list(directional_state.get("source_conflicts") or []),
        },
        "fusion": {
            "state": fusion.get("state"),
            "paper_entry_allowed": fusion.get("paper_entry_allowed"),
            "cognitive_confidence": fusion.get("cognitive_confidence"),
        },
        "evidence": {
            "score": evidence.get("score"),
            "direction": evidence.get("direction"),
            "coverage": evidence.get("coverage"),
            "agreement_ratio": evidence.get("agreement_ratio"),
            "decision_confidence": evidence.get("decision_confidence"),
            "conflict_score": directional.get("conflict_score"),
            "directional_confidence": directional.get("confidence"),
            "dominant_side": directional.get("dominant_side"),
        },
    }
