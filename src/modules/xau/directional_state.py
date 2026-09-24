"""Shared directional state for GEN1 XAU reasoning.

Deterministic and side-effect free. It prevents HTF structure, intraday
cognition, scenarios and gold microstructure from independently inventing
incompatible directions.
"""
from __future__ import annotations

from typing import Any

DIRECTION_THRESHOLD = 0.10
STRONG_HTF_THRESHOLD = 0.25
COUNTER_FLOW_COMPONENT_THRESHOLD = 0.20
COUNTER_FLOW_AGREEMENT_THRESHOLD = 65.0
COUNTER_FLOW_MIN_QUALITY = 0.55


def _num(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def direction_from_score(score: float, threshold: float = DIRECTION_THRESHOLD) -> str:
    value = _clip(score)
    if value >= threshold:
        return "bullish"
    if value <= -threshold:
        return "bearish"
    return "neutral"


def _agreement_scope(gold: dict[str, Any]) -> str:
    explicit = str(gold.get("agreement_scope") or "").strip()
    if explicit:
        return explicit
    coverage = gold.get("coverage_state") or {}
    flows = coverage.get("flow") or {}
    ready = {tf for tf, status in flows.items() if status == "ready"}
    if ready & {"1d", "1w"}:
        return "multi_timeframe"
    if ready & {"1h", "4h"}:
        return "intraday_multiframe"
    if ready:
        return "short_term_only"
    return "collecting"


def gold_microstructure_signal(technical: dict[str, Any]) -> dict[str, Any]:
    gold = technical.get("gold_market_fusion") or {}
    if not gold or gold.get("status") not in {"ready", "degraded"}:
        return {
            "available": False,
            "score": 0.0,
            "effective_score": 0.0,
            "flow_score": 0.0,
            "footprint_score": 0.0,
            "liquidity_score": 0.0,
            "microstructure_score": 0.0,
            "microstructure_agreement_score": None,
            "timeframe_agreement_score": None,
            "agreement_scope": "collecting",
            "independent_source_count": 0,
            "quality": 0.0,
            "quality_state": "unavailable",
            "counter_flow_reliable": False,
        }

    flow_score = _clip(_num(gold.get("composite_flow_score"), 0.0))
    footprint_score = _clip(_num(gold.get("composite_footprint_score"), 0.0))
    liquidity_score = _clip(_num(gold.get("composite_liquidity_score"), 0.0))
    microstructure_score = _clip(_num(gold.get("composite_microstructure_score"), 0.0))
    raw_score = _clip(
        0.45 * flow_score
        + 0.30 * footprint_score
        + 0.25 * liquidity_score
    )

    independent = max(0, int(_num(gold.get("independent_source_count"), 0)))
    agreement = _num(
        gold.get("microstructure_agreement_score"),
        _num(gold.get("market_agreement_score"), 50.0),
    )
    timeframe_agreement = gold.get("timeframe_agreement_score")
    scope = _agreement_scope(gold)

    flow5 = (gold.get("flow") or {}).get("5m") or {}
    venue_rows = list(flow5.get("venues") or [])
    if venue_rows:
        sample_quality = sum(_num(x.get("sample_quality"), 0.5) for x in venue_rows) / len(venue_rows)
        freshness = sum(_num(x.get("freshness"), 0.5) for x in venue_rows) / len(venue_rows)
        feed_health = sum(_num(x.get("feed_health"), 0.5) for x in venue_rows) / len(venue_rows)
    else:
        sample_quality = 0.45
        freshness = 0.45
        feed_health = 0.45

    freshness_state = str(gold.get("freshness_state") or "unavailable")
    freshness_floor = {
        "fresh": 1.0,
        "mixed": 0.80,
        "stale_risk": 0.50,
    }.get(freshness_state, 0.35)
    freshness = min(freshness, freshness_floor)

    scope_factor = {
        "multi_timeframe": 1.0,
        "intraday_multiframe": 0.88,
        "short_term_only": 0.68,
        "collecting": 0.48,
    }.get(scope, 0.55)
    independence_factor = 1.0 if independent >= 2 else 0.62 if independent == 1 else 0.35
    quality = (
        0.28 * _clip(sample_quality, 0.0, 1.0)
        + 0.22 * _clip(freshness, 0.0, 1.0)
        + 0.18 * _clip(feed_health, 0.0, 1.0)
        + 0.17 * scope_factor
        + 0.15 * independence_factor
    )
    quality = max(0.0, min(1.0, quality))

    return {
        "available": True,
        "score": round(raw_score, 4),
        "effective_score": round(raw_score * quality, 4),
        "flow_score": round(flow_score, 4),
        "footprint_score": round(footprint_score, 4),
        "liquidity_score": round(liquidity_score, 4),
        "microstructure_score": round(microstructure_score, 4),
        "microstructure_agreement_score": round(agreement, 2),
        "timeframe_agreement_score": (
            round(_num(timeframe_agreement), 2)
            if timeframe_agreement is not None
            else None
        ),
        "agreement_scope": scope,
        "independent_source_count": independent,
        "sample_quality": round(sample_quality, 4),
        "freshness": round(freshness, 4),
        "feed_health": round(feed_health, 4),
        "quality": round(quality, 4),
        "quality_state": (
            "strong" if quality >= 0.75
            else "usable" if quality >= COUNTER_FLOW_MIN_QUALITY
            else "weak"
        ),
        "counter_flow_reliable": bool(
            independent >= 2
            and agreement >= COUNTER_FLOW_AGREEMENT_THRESHOLD
            and quality >= COUNTER_FLOW_MIN_QUALITY
        ),
        "coverage_state": gold.get("coverage_state") or {},
        "freshness_state": freshness_state,
    }


def _classify(htf: str, intraday: str, gold: str) -> str:
    if htf == "bullish":
        if intraday == "bullish" and gold != "bearish":
            return "aligned_bullish"
        if intraday == "bearish" and gold == "bullish":
            return "bearish_pullback_inside_bullish_structure"
        if intraday == "bearish" and gold == "bearish":
            return "possible_bearish_transition"
        return "bullish_structure_mixed_timing"
    if htf == "bearish":
        if intraday == "bearish" and gold != "bullish":
            return "aligned_bearish"
        if intraday == "bullish" and gold == "bearish":
            return "bullish_pullback_inside_bearish_structure"
        if intraday == "bullish" and gold == "bullish":
            return "possible_bullish_transition"
        return "bearish_structure_mixed_timing"
    if intraday == "bullish" and gold == "bullish":
        return "bullish_intraday_candidate"
    if intraday == "bearish" and gold == "bearish":
        return "bearish_intraday_candidate"
    if (
        intraday in {"bullish", "bearish"}
        and gold in {"bullish", "bearish"}
        and intraday != gold
    ):
        return "intraday_gold_conflict"
    return "direction_unresolved"


def build_directional_state(
    technical: dict[str, Any],
    macro: dict[str, Any],
    *,
    edge: dict[str, Any],
    regime: dict[str, Any],
    data_quality: float,
) -> dict[str, Any]:
    context = technical.get("market_context") or {}
    legacy_cash = _clip(_num((context.get("cash_flow") or {}).get("score"), 0.0))
    intraday_score = _clip(_num(edge.get("score"), 0.0))
    htf_score = _clip(_num(edge.get("higher_timeframe_score"), 0.0))
    smc_score = _clip(_num(edge.get("smart_money_score"), 0.0))
    macro_score = _clip(_num(macro.get("bias"), 0.0)) * max(
        0.0, min(1.0, _num(macro.get("confidence"), 0.0))
    )
    gold = gold_microstructure_signal(technical)

    if gold.get("available"):
        gold_weight = 0.75 * max(0.20, _num(gold.get("quality"), 0.0))
        legacy_weight = 0.25
        unified_flow = (
            _num(gold.get("score"), 0.0) * gold_weight
            + legacy_cash * legacy_weight
        ) / max(1e-9, gold_weight + legacy_weight)
    else:
        unified_flow = legacy_cash

    intraday_direction = direction_from_score(intraday_score)
    htf_direction = direction_from_score(htf_score)
    gold_direction = (
        direction_from_score(_num(gold.get("flow_score"), 0.0))
        if gold.get("available")
        else "neutral"
    )
    classification = _classify(htf_direction, intraday_direction, gold_direction)

    if htf_direction in {"bullish", "bearish"}:
        continuation_direction = htf_direction
        continuation_basis = "higher_timeframe_structure"
    elif intraday_direction in {"bullish", "bearish"} and intraday_direction == gold_direction:
        continuation_direction = intraday_direction
        continuation_basis = "intraday_gold_agreement"
    elif intraday_direction in {"bullish", "bearish"} and gold_direction == "neutral":
        continuation_direction = intraday_direction
        continuation_basis = "intraday_edge"
    elif gold_direction in {"bullish", "bearish"} and intraday_direction == "neutral":
        continuation_direction = gold_direction
        continuation_basis = "gold_microstructure"
    else:
        continuation_direction = "neutral"
        continuation_basis = "unresolved"

    agreement = _num(gold.get("microstructure_agreement_score"), 0.0)
    reliable = bool(gold.get("counter_flow_reliable"))
    counter_flow_short = bool(
        reliable
        and _num(gold.get("flow_score")) >= COUNTER_FLOW_COMPONENT_THRESHOLD
        and _num(gold.get("footprint_score")) >= COUNTER_FLOW_COMPONENT_THRESHOLD
        and agreement >= COUNTER_FLOW_AGREEMENT_THRESHOLD
    )
    counter_flow_long = bool(
        reliable
        and _num(gold.get("flow_score")) <= -COUNTER_FLOW_COMPONENT_THRESHOLD
        and _num(gold.get("footprint_score")) <= -COUNTER_FLOW_COMPONENT_THRESHOLD
        and agreement >= COUNTER_FLOW_AGREEMENT_THRESHOLD
    )

    conflicts: list[str] = []
    if htf_direction != "neutral" and intraday_direction != "neutral" and htf_direction != intraday_direction:
        conflicts.append("htf_vs_intraday")
    if intraday_direction != "neutral" and gold_direction != "neutral" and intraday_direction != gold_direction:
        conflicts.append("intraday_vs_gold_flow")
    if htf_direction != "neutral" and gold_direction != "neutral" and htf_direction != gold_direction:
        conflicts.append("htf_vs_gold_flow")
    if gold.get("available") and gold.get("quality_state") == "weak":
        conflicts.append("gold_signal_quality_insufficient")

    return {
        "version": "directional-state-v1",
        "classification": classification,
        "continuation_direction": continuation_direction,
        "continuation_basis": continuation_basis,
        "intraday_direction": intraday_direction,
        "intraday_score": round(intraday_score, 4),
        "htf_direction": htf_direction,
        "htf_score": round(htf_score, 4),
        "htf_strong": abs(htf_score) >= STRONG_HTF_THRESHOLD,
        "gold_flow_direction": gold_direction,
        "gold_flow_score": gold.get("flow_score", 0.0),
        "footprint_direction": direction_from_score(_num(gold.get("footprint_score"), 0.0)),
        "footprint_score": gold.get("footprint_score", 0.0),
        "liquidity_direction": direction_from_score(_num(gold.get("liquidity_score"), 0.0)),
        "liquidity_score": gold.get("liquidity_score", 0.0),
        "gold_microstructure_score": gold.get("microstructure_score", 0.0),
        "gold_signal_quality": gold.get("quality", 0.0),
        "gold_signal_quality_state": gold.get("quality_state"),
        "legacy_cash_flow_score": round(legacy_cash, 4),
        "unified_flow_score": round(_clip(unified_flow), 4),
        "macro_direction": direction_from_score(macro_score),
        "macro_score": round(macro_score, 4),
        "smc_direction": direction_from_score(smc_score),
        "smc_score": round(smc_score, 4),
        "regime": str(regime.get("label") or "unknown"),
        "data_quality": round(max(0.0, min(1.0, float(data_quality))), 4),
        "coverage_state": gold.get("coverage_state") or {},
        "agreement_scope": gold.get("agreement_scope"),
        "microstructure_agreement_score": gold.get("microstructure_agreement_score"),
        "timeframe_agreement_score": gold.get("timeframe_agreement_score"),
        "source_conflicts": conflicts,
        "independent_source_count": gold.get("independent_source_count", 0),
        "counter_flow_short": counter_flow_short,
        "counter_flow_long": counter_flow_long,
        "counter_flow_reliable": reliable,
        "gold": gold,
    }


def flow_opposition_for_side(state: dict[str, Any], side: str | None) -> dict[str, Any]:
    score = _clip(_num(state.get("unified_flow_score"), 0.0))
    if side == "short":
        opposition = score
        hard_counter = bool(state.get("counter_flow_short"))
    elif side == "long":
        opposition = -score
        hard_counter = bool(state.get("counter_flow_long"))
    else:
        opposition = 0.0
        hard_counter = False

    if opposition >= 0.30:
        status = "failed"
    elif opposition >= 0.10:
        status = "pending"
    else:
        status = "satisfied"

    return {
        "status": status,
        "opposition_score": round(opposition, 4),
        "unified_flow_score": round(score, 4),
        "hard_counter_flow": hard_counter,
        "threshold_pending": 0.10,
        "threshold_failed": 0.30,
    }
