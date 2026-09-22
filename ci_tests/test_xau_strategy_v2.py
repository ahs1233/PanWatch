from __future__ import annotations

from datetime import datetime, timezone

from src.modules.strategy.trading_engine import TradingEngineMode, TradingRunRequest
from src.modules.strategy.trading_engine_adapters import LeanEngineAdapter
from src.modules.strategy.xau_intraday import XAUFrameState, XAUIntradayAssessment
from src.modules.strategy.xau_v2 import (
    XAUFeatureFlags,
    XAUStrategyV2Spec,
    assess_xau_strategy_v2,
    xau_v2_ablation_specs,
)
from src.platform.marketdata.xau_models import XAUTimeframe


NOW = datetime(2026, 9, 23, tzinfo=timezone.utc)


def _frame(tf: XAUTimeframe, direction: str) -> XAUFrameState:
    return XAUFrameState(
        timeframe=tf,
        close=4500.0,
        ema_fast=4495.0,
        ema_slow=4490.0,
        rsi14=60.0,
        atr14=8.0,
        atr_pct=0.18,
        breakout="up" if direction == "bullish" else "down" if direction == "bearish" else "none",
        direction=direction,
        recent_swing_high=4510.0,
        recent_swing_low=4475.0,
        observed_at=NOW,
    )


def _intraday(direction: str = "bullish", blocked: bool = False) -> XAUIntradayAssessment:
    frames = {
        "1m": _frame(XAUTimeframe.M1, direction),
        "5m": _frame(XAUTimeframe.M5, direction),
        "15m": _frame(XAUTimeframe.M15, direction),
    }
    return XAUIntradayAssessment(
        status="blocked" if blocked else "ready",
        candidate=(
            "long_setup" if direction == "bullish"
            else "short_setup" if direction == "bearish"
            else "none"
        ),
        blocked=blocked,
        block_reasons=("event_gate",) if blocked else (),
        warnings=(),
        frame_states=frames,
        macro_bias=0,
        event_risk=blocked,
        spread_bps=1.0,
        atr_reference=8.0,
        swing_high_reference=4510.0,
        swing_low_reference=4475.0,
    )


def _bias_state(sign: int) -> dict:
    if sign > 0:
        ema = {"9": 4495, "21": 4490, "50": 4470, "200": 4400, "1000": 4200}
        close = 4500
        slope = 0.02
    else:
        ema = {"9": 4505, "21": 4510, "50": 4530, "200": 4600, "1000": 4800}
        close = 4500
        slope = -0.02
    return {
        "available": True,
        "close": close,
        "ema": ema,
        "slope_20": slope,
    }


def _context(sign: int = 1) -> dict:
    state = _bias_state(sign)
    return {
        "bias": {
            "monthly": state,
            "weekly": state,
            "daily": state,
            "h4": state,
            "h1": state,
        },
        "smart_money": {
            "available": True,
            "break_of_structure": "bullish" if sign > 0 else "bearish",
            "liquidity_sweep": "sell_side_sweep" if sign > 0 else "buy_side_sweep",
            "displacement": "bullish" if sign > 0 else "bearish",
            "dealing_range": {"zone": "discount" if sign > 0 else "premium"},
            "fair_value_gaps": [],
        },
        "cash_flow": {
            "available": True,
            "score": 0.65 * sign,
            "direction": "inflow" if sign > 0 else "outflow",
            "agreement": "aligned",
            "spot_tick": {"source_type": "mt5_tick_volume_proxy"},
            "gc_futures": {"available": True, "source_type": "futures_volume_proxy"},
        },
        "volume_profile": {
            "available": True,
            "poc": 4490.0,
            "vah": 4515.0,
            "val": 4470.0,
            "location": "inside_value",
            "hvn": [4490.0],
            "lvn": [4478.0],
        },
        "liquidity": {
            "available": True,
            "levels": [{"name": "PDH", "price": 4520.0}],
            "equal_highs": [4518.0],
            "equal_lows": [4472.0],
        },
        "volume_note": "tick volume proxy only",
    }


def test_v2_bullish_stack_produces_research_long_setup():
    spec = XAUStrategyV2Spec()
    result = assess_xau_strategy_v2(
        spec=spec,
        intraday=_intraday("bullish"),
        market_context=_context(1),
    )

    assert result.status == "ready"
    assert result.candidate == "long_setup"
    assert result.score is not None and result.score > spec.score_threshold
    assert result.directional_feature_count == 5
    assert result.live_execution_allowed is False


def test_volume_profile_and_liquidity_are_context_not_directional_votes():
    result = assess_xau_strategy_v2(
        spec=XAUStrategyV2Spec(),
        intraday=_intraday(),
        market_context=_context(),
    )
    profile = result.observations["volume_profile_context"]
    liquidity = result.observations["liquidity_context"]

    assert profile.contributing is False
    assert profile.score is None
    assert profile.detail["directional_vote"] is False
    assert liquidity.contributing is False
    assert liquidity.score is None
    assert liquidity.detail["directional_vote"] is False


def test_flow_proxy_never_claims_centralized_global_spot_order_flow():
    result = assess_xau_strategy_v2(
        spec=XAUStrategyV2Spec(),
        intraday=_intraday(),
        market_context=_context(),
    )
    flow = result.observations["flow_proxy"]
    assert flow.detail["centralized_global_spot_order_flow"] is False

    policy = XAUStrategyV2Spec().to_dict()["evidence_policy"]
    assert policy["spot_tick_volume_is_global_order_flow"] is False
    assert policy["gc_futures_is_xauusd_execution_quote"] is False
    assert policy["same_input_library_votes_are_independent"] is False


def test_missing_features_do_not_silently_become_neutral_evidence():
    spec = XAUStrategyV2Spec(min_directional_features=3)
    result = assess_xau_strategy_v2(
        spec=spec,
        intraday=_intraday(),
        market_context={},
    )
    assert result.status == "blocked"
    assert result.candidate == "none"
    assert result.directional_feature_count == 1
    assert "insufficient_directional_features" in result.block_reasons
    assert "htf_momentum_unavailable" in result.warnings


def test_strong_higher_timeframe_conflict_can_veto_candidate():
    spec = XAUStrategyV2Spec(
        score_threshold=0.05,
        htf_conflict_threshold=0.45,
        intraday_weight=0.60,
        htf_momentum_weight=0.10,
        ema_ladder_weight=0.10,
        smart_money_weight=0.10,
        flow_weight=0.10,
    )
    context = _context(-1)
    context["smart_money"] = _context(1)["smart_money"]
    context["cash_flow"] = _context(1)["cash_flow"]

    result = assess_xau_strategy_v2(
        spec=spec,
        intraday=_intraday("bullish"),
        market_context=context,
    )
    assert result.candidate == "none"
    assert "strong_htf_conflict_long" in result.block_reasons


def test_ablation_variants_are_cumulative_and_have_distinct_fingerprints():
    variants = xau_v2_ablation_specs()
    names = [name for name, _ in variants]
    fingerprints = [spec.fingerprint for _, spec in variants]

    assert names == [
        "baseline_intraday",
        "plus_htf_momentum",
        "plus_ema_ladder",
        "plus_smart_money",
        "plus_flow_and_profile_context",
    ]
    assert len(set(fingerprints)) == len(fingerprints)

    enabled_counts = [
        sum(1 for enabled in spec.features.to_dict().values() if enabled)
        for _, spec in variants
    ]
    assert enabled_counts == sorted(enabled_counts)


def test_feature_toggle_changes_strategy_identity():
    a = XAUStrategyV2Spec()
    b = XAUStrategyV2Spec(
        features=XAUFeatureFlags(flow_proxy=False),
    )
    assert a.fingerprint != b.fingerprint


def test_v2_spec_can_flow_through_engine_agnostic_trading_core():
    spec = XAUStrategyV2Spec()
    request = TradingRunRequest(
        mode=TradingEngineMode.BACKTEST,
        strategy_fingerprint=spec.fingerprint,
        strategy=spec.to_dict(),
        dataset_ref="xau-history-v2",
    )
    plan = LeanEngineAdapter().prepare(request)

    assert plan.engine == "lean"
    assert plan.strategy_fingerprint == spec.fingerprint
    assert plan.package["strategy"] == spec.to_dict()
    assert plan.package["execution"]["live"] is False
    assert plan.package["execution"]["broker_orders_enabled"] is False
