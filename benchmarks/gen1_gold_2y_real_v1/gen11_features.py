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



def flatten_gen11_episode(ep) -> dict[str, Any]:
    meta = dict(ep.meta or {})
    snap = dict(meta.get("gen11_features") or {})
    row: dict[str, Any] = {
        "observed_at": ep.observed_at.isoformat(),
        "outcome_at": ep.outcome_at.isoformat(),
        "candidate": ep.candidate,
        "regime": ep.regime,
        "cognitive_confidence": ep.confidence,
        "gen1_decision": meta.get("gen1_decision"),
        "gen1_confidence": meta.get("gen1_decision_confidence"),
        "fusion_state": meta.get("gen1_fusion_state"),
        "entry_price": ep.entry_price,
        "outcome_price": ep.outcome_price,
        "gross_directional_bps": ep.directional_return_bps,
        "net_spread_directional_bps": meta.get("net_spread_directional_return_bps"),
        "session": snap.get("session"),
        "session_transition": snap.get("session_transition"),
        "alignment": snap.get("alignment"),
    }
    for tf in ("1m", "5m", "15m"):
        frame = dict((snap.get("frames") or {}).get(tf) or {})
        for key in ("direction", "rsi14", "atr14", "atr_pct", "breakout"):
            row[f"{tf}_{key}"] = frame.get(key)
        close, fast, slow, atr = frame.get("close"), frame.get("ema_fast"), frame.get("ema_slow"), frame.get("atr14")
        swing_high, swing_low = frame.get("recent_swing_high"), frame.get("recent_swing_low")
        try:
            atr_value = float(atr)
            row[f"{tf}_fast_distance_atr"] = (float(close) - float(fast)) / atr_value if atr_value else None
            row[f"{tf}_slow_distance_atr"] = (float(close) - float(slow)) / atr_value if atr_value else None
            row[f"{tf}_swing_high_distance_atr"] = (float(close) - float(swing_high)) / atr_value if atr_value and swing_high is not None else None
            row[f"{tf}_swing_low_distance_atr"] = (float(close) - float(swing_low)) / atr_value if atr_value and swing_low is not None else None
        except (TypeError, ValueError, ZeroDivisionError):
            row[f"{tf}_fast_distance_atr"] = None
            row[f"{tf}_slow_distance_atr"] = None
            row[f"{tf}_swing_high_distance_atr"] = None
            row[f"{tf}_swing_low_distance_atr"] = None
    htf = dict(snap.get("htf_bias") or {})
    for tf in ("monthly", "weekly", "daily", "h4", "h1"):
        state = dict(htf.get(tf) or {})
        row[f"{tf}_direction"] = state.get("direction")
        row[f"{tf}_score"] = state.get("score")
        row[f"{tf}_slope20"] = state.get("slope_20")
        close = state.get("close")
        emas = dict(state.get("ema") or {})
        for period in (9, 21, 50, 200, 1000):
            ema = emas.get(str(period))
            try:
                row[f"{tf}_ema{period}_distance_pct"] = (float(close) - float(ema)) / float(ema) if float(ema) else None
            except (TypeError, ValueError, ZeroDivisionError):
                row[f"{tf}_ema{period}_distance_pct"] = None
    for key in ("composite_score", "composite_direction", "today_score", "today_direction"):
        row[f"htf_{key}"] = htf.get(key)
    flow = dict(snap.get("flow") or {})
    for key in ("direction", "score", "agreement", "cmf20", "signed_tick_volume_imbalance", "obv_slope_proxy"):
        row[f"flow_{key}"] = flow.get(key)
    profile = dict(snap.get("volume_profile") or {})
    for key in ("poc", "vah", "val", "location", "range_low", "range_high", "total_tick_volume"):
        row[f"volume_{key}"] = profile.get(key)
    smart = dict(snap.get("smart_money") or {})
    for key in ("bias", "score", "break_of_structure", "liquidity_sweep", "displacement", "dealing_zone"):
        row[f"smart_{key}"] = smart.get(key)
    liquidity = dict(snap.get("liquidity") or {})
    row["liquidity_equal_high_count"] = liquidity.get("equal_high_count")
    row["liquidity_equal_low_count"] = liquidity.get("equal_low_count")
    row["liquidity_tolerance"] = liquidity.get("tolerance")
    nearest = list(liquidity.get("nearest_levels") or [])
    row["liquidity_nearest_name"] = (nearest[0] or {}).get("name") if nearest else None
    row["liquidity_nearest_distance"] = (nearest[0] or {}).get("distance") if nearest else None
    macro = dict(snap.get("macro") or {})
    row["macro_bias"] = macro.get("bias")
    row["macro_bias_label"] = macro.get("bias_label")
    row["macro_confidence"] = macro.get("confidence")
    row["macro_proxy_score"] = macro.get("proxy_score")
    for driver in list(macro.get("drivers") or []):
        name = str((driver or {}).get("name") or "")
        if name in {"dxy_5d", "vix_5d", "spx_5d", "oil_5d", "us10y_5d_delta"}:
            row[f"macro_{name}_gold_score"] = (driver or {}).get("gold_score")
            row[f"macro_{name}_value"] = (driver or {}).get("value")
    evidence = dict(snap.get("evidence") or {})
    for key in ("score", "direction", "coverage", "agreement_ratio", "decision_confidence", "conflict_score", "directional_confidence", "dominant_side"):
        row[f"evidence_{key}"] = evidence.get(key)
    cognition = dict(snap.get("cognition") or {})
    for key in ("calibrated_confidence", "execution_action", "execution_side", "directional_classification", "directional_edge"):
        row[f"cognition_{key}"] = cognition.get(key)
    ranges = dict(meta.get("range_outcomes") or {})
    row["range_max_up_usd"] = ranges.get("max_up_usd")
    row["range_max_down_usd"] = ranges.get("max_down_usd")
    levels = dict(ranges.get("levels") or {})
    side = "up" if ep.candidate == "long_setup" else "down"
    for distance in (10, 20, 30):
        first = str((levels.get(f"pm{distance}") or {}).get("first_hit") or "none")
        row[f"range_pm{distance}_first"] = first
        row[f"range_pm{distance}_favorable_first"] = first == side
    return row


def write_gen11_feature_csv(path, episodes) -> None:
    import csv
    rows = [flatten_gen11_episode(ep) for ep in episodes]
    if not rows:
        return
    fieldnames = list(rows[0])
    extras = sorted({key for row in rows for key in row}.difference(fieldnames))
    fieldnames.extend(extras)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
