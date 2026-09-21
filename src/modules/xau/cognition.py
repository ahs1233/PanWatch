"""Deterministic multi-layer cognition for XAU/USD research and paper trading.

This module is deliberately LLM-free so it can run in the hot path.  It turns the
current technical/macro snapshot plus closed-trade memory into an inspectable
reasoning stack: perception -> regime -> hypotheses -> adversarial review ->
confidence calibration -> execution plan -> meta-controller.

It never places live orders and never marks indicative data execution-eligible.
"""

from __future__ import annotations

from math import exp
from typing import Any

COGNITION_VERSION = "1.0.0"


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _direction_value(value: Any) -> int:
    text = str(value or "").lower()
    return 1 if text == "bullish" else -1 if text == "bearish" else 0


def _softmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    ceiling = max(scores.values())
    transformed = {key: exp(value - ceiling) for key, value in scores.items()}
    total = sum(transformed.values()) or 1.0
    return {key: round(value / total, 4) for key, value in transformed.items()}


def _data_quality(technical: dict[str, Any]) -> tuple[float, list[str]]:
    score = 1.0
    issues: list[str] = []
    spot = technical.get("indicative_spot") or {}
    micro = technical.get("micro") or {}
    frames = technical.get("frames") or {}

    if technical.get("blocked"):
        score -= 0.35
        issues.append("technical_data_gate")
    if not spot:
        score -= 0.35
        issues.append("spot_missing")
    else:
        if spot.get("is_stale"):
            score -= 0.30
            issues.append("spot_stale")
        if spot.get("bid") is None or spot.get("ask") is None:
            score -= 0.15
            issues.append("bid_ask_missing")
    if not micro or micro.get("status") == "blocked" or micro.get("is_stale"):
        score -= 0.15
        issues.append("micro_unreliable")

    missing_frames = [name for name in ("1m", "5m", "15m") if name not in frames]
    if missing_frames:
        score -= 0.08 * len(missing_frames)
        issues.extend(f"{name}_missing" for name in missing_frames)

    if any("fallback" in str(item) for item in technical.get("warnings") or []):
        score -= 0.08
        issues.append("fallback_source_active")

    return round(_clip(score), 4), list(dict.fromkeys(issues))


def _perception(technical: dict[str, Any]) -> dict[str, Any]:
    frames = technical.get("frames") or {}
    micro = technical.get("micro") or {}
    five = frames.get("5m") or {}
    fifteen = frames.get("15m") or {}
    one = frames.get("1m") or {}

    directions = [
        _direction_value(one.get("direction")),
        _direction_value(five.get("direction")),
        _direction_value(fifteen.get("direction")),
        _direction_value(micro.get("direction")),
    ]
    directional_pressure = sum(directions) / max(1, len(directions))
    ret10 = _number(micro.get("return_10m_pct"), 0.0)
    ret30 = _number(micro.get("return_30m_pct"), 0.0)
    acceleration = ret10 - (ret30 / 3.0 if ret30 else 0.0)

    atr5 = _number(five.get("atr_pct"), 0.0)
    atr15 = _number(fifteen.get("atr_pct"), 0.0)
    volatility_pct = max(atr5, atr15)
    breakout = str(five.get("breakout") or "none")
    if breakout == "none":
        breakout = str(one.get("breakout") or "none")

    rsi5 = _number(five.get("rsi14"), 50.0)
    rsi15 = _number(fifteen.get("rsi14"), 50.0)
    momentum = "neutral"
    if directional_pressure >= 0.5 and ret10 >= 0:
        momentum = "bullish"
    elif directional_pressure <= -0.5 and ret10 <= 0:
        momentum = "bearish"

    return {
        "directional_pressure": round(directional_pressure, 4),
        "momentum": momentum,
        "return_10m_pct": round(ret10, 5),
        "return_30m_pct": round(ret30, 5),
        "acceleration": round(acceleration, 5),
        "volatility_pct": round(volatility_pct, 5),
        "breakout": breakout,
        "rsi_5m": round(rsi5, 2),
        "rsi_15m": round(rsi15, 2),
    }


def _regime(perception: dict[str, Any], technical: dict[str, Any]) -> dict[str, Any]:
    pressure = _number(perception.get("directional_pressure"))
    vol = _number(perception.get("volatility_pct"))
    breakout = str(perception.get("breakout") or "none")
    alignment = str(technical.get("alignment") or "mixed")

    if technical.get("blocked"):
        label, confidence = "data_uncertain", 0.95
    elif breakout in {"up", "down"} and alignment in {"bullish", "bearish"}:
        label, confidence = "breakout_expansion", 0.82
    elif abs(pressure) >= 0.6 and alignment in {"bullish", "bearish"}:
        label, confidence = (
            "trend_bull" if pressure > 0 else "trend_bear",
            0.74 + min(0.16, abs(pressure) * 0.16),
        )
    elif abs(pressure) <= 0.25 and vol <= 0.18:
        label, confidence = "compression_range", 0.72
    elif alignment == "mixed":
        label, confidence = "transition", 0.66
    else:
        label, confidence = "range", 0.60

    return {
        "label": label,
        "confidence": round(_clip(confidence), 4),
        "volatility_pct": round(vol, 5),
        "alignment": alignment,
    }


def _hypotheses(
    technical: dict[str, Any],
    macro: dict[str, Any],
    perception: dict[str, Any],
    regime: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate = str(technical.get("candidate") or "none")
    setup_dir = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    macro_bias = int(max(-1, min(1, _number(macro.get("bias"), 0.0))))
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    pressure = _number(perception.get("directional_pressure"))
    acceleration = _number(perception.get("acceleration"))
    breakout = str(perception.get("breakout") or "none")
    rsi5 = _number(perception.get("rsi_5m"), 50.0)
    regime_label = str(regime.get("label") or "transition")

    continuation = 0.0
    continuation += 1.2 if setup_dir else -0.4
    continuation += 0.9 * abs(pressure)
    continuation += 0.55 if regime_label in {"trend_bull", "trend_bear", "breakout_expansion"} else 0.0
    continuation += 0.45 if breakout in {"up", "down"} else 0.0
    if setup_dir and macro_bias == setup_dir:
        continuation += 0.45 * max(0.35, macro_conf)
    elif setup_dir and macro_bias == -setup_dir:
        continuation -= 0.55 * max(0.35, macro_conf)

    mean_reversion = 0.15
    mean_reversion += 0.75 if regime_label in {"range", "compression_range", "transition"} else 0.0
    if rsi5 >= 70 or rsi5 <= 30:
        mean_reversion += 0.75
    if setup_dir and acceleration * setup_dir < 0:
        mean_reversion += 0.4

    sweep = 0.10
    sweep += 0.55 if breakout in {"up", "down"} else 0.0
    sweep += 0.4 if regime_label in {"transition", "breakout_expansion"} else 0.0
    if rsi5 >= 75 or rsi5 <= 25:
        sweep += 0.35

    weights = _softmax(
        {
            "trend_continuation": continuation,
            "mean_reversion": mean_reversion,
            "liquidity_sweep": sweep,
        }
    )
    direction = "long" if setup_dir > 0 else "short" if setup_dir < 0 else "none"
    opposing = "short" if direction == "long" else "long" if direction == "short" else "none"
    ordered = sorted(weights.items(), key=lambda item: item[1], reverse=True)
    out = []
    for name, weight in ordered:
        out.append(
            {
                "name": name,
                "weight": weight,
                "direction": direction if name == "trend_continuation" else opposing,
            }
        )
    return out


def _memory_adjustment(memory: dict[str, Any] | None) -> dict[str, Any]:
    memory = memory or {}
    trade_count = int(_number(memory.get("trade_count"), 0.0))
    expectancy = _number(memory.get("expectancy_r"), 0.0)
    profit_factor = memory.get("profit_factor")
    profit_factor_value = _number(profit_factor, 1.0) if profit_factor is not None else 1.0

    strength = _clip(trade_count / 30.0)
    edge = _clip(expectancy / 1.5, -1.0, 1.0)
    pf_edge = _clip((profit_factor_value - 1.0) / 2.0, -1.0, 1.0)
    adjustment = (0.06 * edge + 0.04 * pf_edge) * strength

    return {
        "trade_count": trade_count,
        "expectancy_r": round(expectancy, 4),
        "profit_factor": profit_factor,
        "learning_strength": round(strength, 4),
        "confidence_adjustment": round(adjustment, 4),
    }


def _adversarial_review(
    technical: dict[str, Any],
    macro: dict[str, Any],
    perception: dict[str, Any],
    regime: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    data_quality: float,
) -> dict[str, Any]:
    candidate = str(technical.get("candidate") or "none")
    setup_dir = 1 if candidate == "long_setup" else -1 if candidate == "short_setup" else 0
    macro_bias = int(max(-1, min(1, _number(macro.get("bias"), 0.0))))
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    primary = hypotheses[0] if hypotheses else {"name": "none", "weight": 0.0}
    rsi5 = _number(perception.get("rsi_5m"), 50.0)

    counter_evidence: list[str] = []
    hard_veto = False

    if data_quality < 0.58:
        counter_evidence.append("insufficient_data_quality")
        hard_veto = True
    if bool(macro.get("event_risk")):
        counter_evidence.append("active_high_impact_event")
        hard_veto = True
    if setup_dir and macro_bias == -setup_dir and macro_conf >= 0.65:
        counter_evidence.append("high_confidence_macro_conflict")
    if primary.get("name") != "trend_continuation":
        counter_evidence.append("continuation_not_primary_hypothesis")
    if _number(primary.get("weight")) < 0.46:
        counter_evidence.append("weak_primary_hypothesis")
    if setup_dir > 0 and rsi5 >= 74:
        counter_evidence.append("long_setup_overextended_rsi")
    elif setup_dir < 0 and rsi5 <= 26:
        counter_evidence.append("short_setup_overextended_rsi")
    if regime.get("label") in {"transition", "data_uncertain"}:
        counter_evidence.append("unstable_market_regime")

    veto_score = 0.0
    veto_score += 0.45 if hard_veto else 0.0
    veto_score += min(0.45, 0.09 * len(counter_evidence))
    veto_score = _clip(veto_score)
    veto = hard_veto or veto_score >= 0.64

    return {
        "veto": veto,
        "veto_score": round(veto_score, 4),
        "counter_evidence": list(dict.fromkeys(counter_evidence)),
    }


def _confidence(
    data_quality: float,
    regime: dict[str, Any],
    hypotheses: list[dict[str, Any]],
    macro: dict[str, Any],
    adversarial: dict[str, Any],
    memory_state: dict[str, Any],
) -> dict[str, Any]:
    primary_weight = _number((hypotheses[0] if hypotheses else {}).get("weight"), 0.0)
    regime_conf = _number(regime.get("confidence"), 0.0)
    macro_conf = _clip(_number(macro.get("confidence"), 0.0))
    event_penalty = 0.22 if macro.get("event_risk") else 0.0
    adversarial_penalty = 0.30 * _number(adversarial.get("veto_score"), 0.0)

    raw = (
        0.30 * data_quality
        + 0.30 * primary_weight
        + 0.20 * regime_conf
        + 0.10 * macro_conf
        + 0.10
        - event_penalty
        - adversarial_penalty
    )
    # With little history, shrink confidence toward 0.50 rather than pretend
    # that a deterministic score is already empirically calibrated.
    learning_strength = _number(memory_state.get("learning_strength"), 0.0)
    shrink = 0.20 * (1.0 - learning_strength)
    calibrated = raw * (1.0 - shrink) + 0.50 * shrink
    calibrated += _number(memory_state.get("confidence_adjustment"), 0.0)
    calibrated = _clip(calibrated, 0.05, 0.95)

    band = "high" if calibrated >= 0.72 else "medium" if calibrated >= 0.58 else "low"
    return {
        "raw_confidence": round(_clip(raw), 4),
        "calibrated_confidence": round(calibrated, 4),
        "band": band,
        "calibration_basis": "closed_trade_memory+deterministic_evidence",
    }


def _execution_plan(
    technical: dict[str, Any],
    perception: dict[str, Any],
    confidence: dict[str, Any],
    adversarial: dict[str, Any],
    min_confidence: float,
) -> dict[str, Any]:
    candidate = str(technical.get("candidate") or "none")
    side = "long" if candidate == "long_setup" else "short" if candidate == "short_setup" else None
    spot = technical.get("indicative_spot") or {}
    price = _number(spot.get("price"), 0.0)
    atr = max(0.0, _number(technical.get("atr_reference"), 0.0))
    five = (technical.get("frames") or {}).get("5m") or {}
    ema_fast = _number(five.get("ema_fast"), price)
    extension_atr = abs(price - ema_fast) / atr if atr > 0 and price > 0 else 0.0
    calibrated = _number(confidence.get("calibrated_confidence"), 0.0)

    reasons: list[str] = []
    action = "STAND_DOWN"
    if side is None:
        reasons.append("no_directional_setup")
    elif adversarial.get("veto"):
        reasons.append("adversarial_veto")
    elif calibrated < min_confidence:
        action = "WAIT"
        reasons.append("confidence_below_threshold")
    elif extension_atr >= 0.80:
        action = "WAIT_PULLBACK"
        reasons.append("price_extended_from_5m_fast_ema")
    else:
        action = "ENTER_NOW"
        reasons.append("cognitive_gates_passed")

    if side and price > 0 and atr > 0:
        if side == "long":
            entry_zone = [round(price - 0.22 * atr, 4), round(price + 0.05 * atr, 4)]
            invalidation = technical.get("swing_low_reference")
        else:
            entry_zone = [round(price - 0.05 * atr, 4), round(price + 0.22 * atr, 4)]
            invalidation = technical.get("swing_high_reference")
    else:
        entry_zone = None
        invalidation = None

    return {
        "action": action,
        "side": side,
        "entry_zone": entry_zone,
        "invalidation_reference": invalidation,
        "extension_atr": round(extension_atr, 4),
        "reasons": reasons,
        "execution_allowed": False,
    }


def build_cognitive_state(
    technical: dict[str, Any],
    macro: dict[str, Any],
    *,
    memory: dict[str, Any] | None = None,
    min_confidence: float = 0.58,
) -> dict[str, Any]:
    """Build the complete inspectable XAU cognition stack."""

    min_confidence = _clip(float(min_confidence), 0.50, 0.90)
    data_quality, quality_issues = _data_quality(technical)
    perception = _perception(technical)
    regime = _regime(perception, technical)
    hypotheses = _hypotheses(technical, macro, perception, regime)
    memory_state = _memory_adjustment(memory)
    adversarial = _adversarial_review(
        technical,
        macro,
        perception,
        regime,
        hypotheses,
        data_quality,
    )
    confidence = _confidence(
        data_quality,
        regime,
        hypotheses,
        macro,
        adversarial,
        memory_state,
    )
    plan = _execution_plan(
        technical,
        perception,
        confidence,
        adversarial,
        min_confidence,
    )

    action = str(plan.get("action") or "STAND_DOWN")
    if action == "ENTER_NOW":
        decision = "eligible"
    elif action in {"WAIT", "WAIT_PULLBACK"}:
        decision = "wait"
    elif technical.get("candidate") in {"long_setup", "short_setup"}:
        decision = "veto"
    else:
        decision = "observe"

    return {
        "version": COGNITION_VERSION,
        "data_quality": {
            "score": data_quality,
            "issues": quality_issues,
        },
        "perception": perception,
        "regime": regime,
        "hypotheses": hypotheses,
        "memory": memory_state,
        "adversarial": adversarial,
        "confidence": confidence,
        "execution_plan": plan,
        "meta_controller": {
            "decision": decision,
            "paper_entry_allowed": decision == "eligible",
            "min_confidence": round(min_confidence, 4),
            "live_execution_allowed": False,
        },
    }
