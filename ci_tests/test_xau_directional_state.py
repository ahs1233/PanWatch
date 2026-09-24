from __future__ import annotations

from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.directional_state import (
    build_directional_state,
    direction_from_score,
    flow_opposition_for_side,
)


def _gold(flow=0.25, footprint=0.25, liquidity=0.15, agreement=75.0, independent=2, scope="short_term_only"):
    coverage = {
        "flow": {
            "5m": "ready",
            "15m": "ready" if scope != "collecting" else "partial",
            "30m": "ready" if scope != "collecting" else "partial",
            "1h": "ready" if scope in {"intraday_multiframe", "multi_timeframe"} else "partial",
            "4h": "ready" if scope in {"intraday_multiframe", "multi_timeframe"} else "partial",
            "1d": "ready" if scope == "multi_timeframe" else "partial",
            "1w": "ready" if scope == "multi_timeframe" else "partial",
        },
        "profiles": {"1h": "partial", "4h": "partial", "1d": "partial", "1w": "partial"},
    }
    return {
        "status": "ready",
        "venue_count": 3 if independent >= 2 else 2,
        "independent_source_count": independent,
        "composite_flow_score": flow,
        "composite_footprint_score": footprint,
        "composite_liquidity_score": liquidity,
        "composite_microstructure_score": 0.45 * flow + 0.30 * footprint + 0.25 * liquidity,
        "market_agreement_score": agreement,
        "microstructure_agreement_score": agreement,
        "timeframe_agreement_score": 100.0,
        "agreement_scope": scope,
        "freshness_state": "fresh",
        "coverage_state": coverage,
        "flow": {
            "5m": {
                "decision_eligible": True,
                "venues": [
                    {"sample_quality": 1.0, "freshness": 1.0, "feed_health": 1.0},
                    {"sample_quality": 0.9, "freshness": 1.0, "feed_health": 1.0},
                ],
            }
        },
    }


def _state(htf, intraday, gold_flow, *, gold_footprint=None, independent=2, agreement=75.0, scope="short_term_only"):
    technical = {
        "market_context": {"cash_flow": {"score": 0.0}},
        "gold_market_fusion": _gold(
            flow=gold_flow,
            footprint=gold_flow if gold_footprint is None else gold_footprint,
            independent=independent,
            agreement=agreement,
            scope=scope,
        ),
    }
    return build_directional_state(
        technical,
        {"bias": 0, "confidence": 0.0},
        edge={"score": intraday, "higher_timeframe_score": htf, "smart_money_score": 0.0},
        regime={"label": "transition"},
        data_quality=1.0,
    )


def test_direction_conflict_matrix_core_cases():
    assert _state(0.30, 0.30, 0.30)["classification"] == "aligned_bullish"
    assert _state(-0.30, -0.30, -0.30)["classification"] == "aligned_bearish"
    assert _state(0.18, -0.23, 0.26)["classification"] == "bearish_pullback_inside_bullish_structure"
    assert _state(0.18, -0.23, -0.26)["classification"] == "possible_bearish_transition"
    assert _state(-0.18, 0.23, -0.26)["classification"] == "bullish_pullback_inside_bearish_structure"
    assert _state(-0.18, 0.23, 0.26)["classification"] == "possible_bullish_transition"
    assert _state(0.0, 0.23, 0.26)["classification"] == "bullish_intraday_candidate"
    assert _state(0.0, -0.23, -0.26)["classification"] == "bearish_intraday_candidate"


def test_threshold_boundaries_are_stable():
    assert direction_from_score(0.09) == "neutral"
    for score in (0.11, 0.19, 0.21, 0.24, 0.26, 0.29, 0.31, 0.34, 0.36):
        assert direction_from_score(score) == "bullish"
    assert direction_from_score(-0.09) == "neutral"
    assert direction_from_score(-0.11) == "bearish"


def test_counter_flow_requires_independent_quality_and_agreement():
    state = _state(0.18, -0.23, 0.26, gold_footprint=0.27, independent=2, agreement=79.38)
    assert state["counter_flow_short"] is True
    assert flow_opposition_for_side(state, "short")["hard_counter_flow"] is True

    same_exchange_only = _state(0.18, -0.23, 0.26, gold_footprint=0.27, independent=1, agreement=79.38)
    assert same_exchange_only["counter_flow_short"] is False
    assert same_exchange_only["independent_source_count"] == 1


def test_agreement_scope_does_not_promote_5m_to_htf():
    state = _state(0.18, -0.23, 0.26, scope="short_term_only")
    assert state["agreement_scope"] == "short_term_only"
    assert state["timeframe_agreement_score"] == 100.0
    assert state["coverage_state"]["flow"]["1h"] == "partial"
    assert state["coverage_state"]["flow"]["1d"] == "partial"
    assert state["coverage_state"]["flow"]["1w"] == "partial"


def test_gold_unavailable_falls_back_to_legacy_cash_flow():
    technical = {"market_context": {"cash_flow": {"score": -0.22}}}
    state = build_directional_state(
        technical,
        {"bias": 0, "confidence": 0.0},
        edge={"score": -0.20, "higher_timeframe_score": 0.0, "smart_money_score": 0.0},
        regime={"label": "trend_bear"},
        data_quality=1.0,
    )
    assert state["gold"]["available"] is False
    assert state["unified_flow_score"] == -0.22


def _live_regression_technical():
    price = 4319.156
    return {
        "candidate": "short_setup",
        "blocked": False,
        "alignment": "mixed",
        "warnings": [],
        "atr_reference": 3.2649,
        "swing_high_reference": 4325.522,
        "swing_low_reference": 4315.405,
        "indicative_spot": {"price": price, "bid": price - 0.1, "ask": price + 0.1, "age_seconds": 1.0, "is_stale": False},
        "analysis_reference": {"price": price, "source": "test", "age_seconds": 1.0, "is_stale": False, "kind": "indicative_spot", "execution_eligible": False},
        "micro": {"status": "ready", "direction": "bearish", "return_10m_pct": -0.02778, "return_30m_pct": -0.01158, "is_stale": False},
        "frames": {
            "1m": {"direction": "neutral", "rsi14": 48.0, "atr14": 1.0, "atr_pct": 0.05, "breakout": "none", "ema_fast": 4319.0, "ema_slow": 4319.5},
            "5m": {"direction": "bearish", "rsi14": 48.33, "atr14": 3.2, "atr_pct": 0.08, "breakout": "none", "close": price, "ema_fast": 4318.7186, "ema_slow": 4320.2, "ema50": 4325.85, "ema200": 4334.95},
            "15m": {"direction": "bearish", "rsi14": 26.92, "atr14": 5.75, "atr_pct": 0.13, "breakout": "none", "close": price, "ema_fast": 4321.17, "ema_slow": 4327.39, "ema50": 4335.37},
        },
        "market_context": {
            "bias": {"today_score": 0.185, "today_direction": "bullish", "daily": {"available": True}},
            "smart_money": {"score": -0.12, "bias": "neutral"},
            "cash_flow": {"score": -0.098, "direction": "outflow"},
            "cross_validation": {"library_direction": "bullish", "library_agreement": 0.66, "smc_concordance": "unresolved"},
            "volume_profile": {"location": "inside_value"},
        },
        "gold_market_fusion": _gold(flow=0.263, footprint=0.269, liquidity=0.215, agreement=79.38, independent=2, scope="short_term_only"),
    }


def test_live_regression_unifies_hypothesis_scenario_and_blocks_counter_flow_short():
    state = build_cognitive_state(
        _live_regression_technical(),
        {"bias": -1, "confidence": 0.60, "event_risk": False},
        min_confidence=0.50,
    )
    shared = state["directional_state"]
    assert shared["classification"] == "bearish_pullback_inside_bullish_structure"
    assert shared["agreement_scope"] == "short_term_only"
    continuation = next(x for x in state["hypotheses"] if x["name"] == "trend_continuation")
    scenario = next(x for x in state["scenarios"] if x["name"] == "continuation")
    assert continuation["direction"] == "long"
    assert scenario["direction"] == "bullish"
    assert state["execution_plan"]["setup_type"] == "counter_flow_short"
    assert state["execution_plan"]["entry_allowed"] is False
    assert state["execution_plan"]["action"] == "WAIT_CONFIRMATION"
    assert state["execution_plan"]["hypothesis_scenario_consistent"] is True


def test_gold_flip_releases_counter_flow_when_current_signal_is_reliable():
    bearish = _state(0.18, -0.23, -0.31, gold_footprint=-0.31, independent=2, agreement=80.0)
    assert bearish["counter_flow_short"] is False
    bullish = _state(0.18, -0.23, 0.31, gold_footprint=0.31, independent=2, agreement=80.0)
    assert bullish["counter_flow_short"] is True
