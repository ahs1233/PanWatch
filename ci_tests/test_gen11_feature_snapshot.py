from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from benchmarks.gen1_gold_2y_real_v1.gen11_features import (
    build_gen11_feature_snapshot,
    flatten_gen11_episode,
)


def test_gen11_snapshot_is_point_in_time_and_compact():
    observed = datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)
    technical = {
        "alignment": "bullish",
        "frames": {
            "1m": {"close": 2500.0, "ema_fast": 2499.0, "ema_slow": 2498.0, "rsi14": 58.0, "atr14": 2.0, "atr_pct": 0.08, "breakout": "none", "direction": "bullish"},
            "5m": {"close": 2500.0, "ema_fast": 2498.0, "ema_slow": 2496.0, "rsi14": 60.0, "atr14": 5.0, "atr_pct": 0.20, "breakout": "up", "direction": "bullish"},
            "15m": {"close": 2500.0, "ema_fast": 2496.0, "ema_slow": 2490.0, "rsi14": 62.0, "atr14": 8.0, "atr_pct": 0.32, "breakout": "none", "direction": "bullish"},
        },
        "market_context": {
            "bias": {
                "monthly": {"direction": "bullish", "score": 0.4, "close": 2500.0, "ema": {"9": 2450.0}, "slope_20": 0.03, "available": True, "bar_count": 30},
                "weekly": {"direction": "bullish", "score": 0.3, "close": 2500.0, "ema": {"9": 2460.0}, "slope_20": 0.02, "available": True, "bar_count": 100},
                "daily": {"direction": "bullish", "score": 0.2, "close": 2500.0, "ema": {"9": 2470.0}, "slope_20": 0.01, "available": True, "bar_count": 300},
                "h4": {"direction": "bullish", "score": 0.2, "close": 2500.0, "ema": {"9": 2480.0}, "slope_20": 0.01, "available": True, "bar_count": 500},
                "h1": {"direction": "bullish", "score": 0.1, "close": 2500.0, "ema": {"9": 2490.0}, "slope_20": 0.01, "available": True, "bar_count": 1000},
                "composite_score": 0.31,
                "composite_direction": "bullish",
                "today_score": 0.25,
                "today_direction": "bullish",
            },
            "cash_flow": {
                "direction": "inflow",
                "score": 0.2,
                "agreement": "spot_tick_only",
                "spot_tick": {"cmf20": 0.1, "signed_tick_volume_imbalance": 0.2, "obv_slope_proxy": 0.1},
            },
            "volume_profile": {"available": True, "poc": 2495.0, "vah": 2510.0, "val": 2480.0, "location": "inside_value", "bins": [{"price": 1, "volume": 2}]},
            "smart_money": {"bias": "bullish", "score": 0.3, "break_of_structure": "bullish", "liquidity_sweep": "none", "displacement": "bullish", "dealing_range": {"zone": "discount"}, "fair_value_gaps": [{"low": 1, "high": 2}]},
            "liquidity": {"levels": [{"name": "PDH", "price": 2510.0, "side": "above", "distance": 10.0}], "equal_highs": [2510.0], "equal_lows": [], "tolerance": 0.5},
        },
    }
    cognition = {
        "confidence": {"calibrated_confidence": 0.64},
        "regime": {"label": "trend_bull"},
        "execution_plan": {"action": "WAIT_TRIGGER", "side": "long"},
        "directional_state": {"classification": "aligned", "directional_edge": 0.2, "source_conflicts": []},
    }
    fusion = {"state": "setup_macro_neutral", "paper_entry_allowed": True, "cognitive_confidence": 0.64}
    evidence = {"score": 0.25, "direction": "bullish", "coverage": 0.8, "agreement_ratio": 0.75, "decision_confidence": 0.68, "directional_evidence": {"conflict_score": 0.1, "confidence": 0.7, "dominant_side": "long"}}
    macro = {"bias": 1, "bias_label": "bullish", "confidence": 0.6, "proxy_score": 0.2, "drivers": [{"name": "dxy_5d", "gold_score": 0.3, "value": -0.01}], "observed_at": observed.isoformat()}

    snapshot = build_gen11_feature_snapshot(
        technical,
        cognition,
        fusion,
        evidence,
        macro,
        observed_at=observed,
        previous_session="asia",
    )

    assert snapshot["session"] == "london"
    assert snapshot["session_transition"] is True
    assert snapshot["lookahead_protected"] is True
    assert snapshot["outcome_fields_present"] is False
    assert "bins" not in snapshot["volume_profile"]
    assert "fair_value_gaps" not in snapshot["smart_money"]
    assert snapshot["htf_bias"]["monthly"]["ema"]["9"] == 2450.0


def test_flattened_episode_keeps_labels_separate_from_features():
    observed = datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)
    snapshot = {
        "session": "london",
        "session_transition": False,
        "alignment": "bullish",
        "frames": {},
        "htf_bias": {},
        "flow": {},
        "volume_profile": {},
        "smart_money": {},
        "liquidity": {},
        "macro": {},
        "evidence": {},
        "cognition": {},
    }
    episode = SimpleNamespace(
        observed_at=observed,
        outcome_at=observed,
        candidate="long_setup",
        regime="trend_bull",
        confidence=0.61,
        entry_price=2500.0,
        outcome_price=2510.0,
        directional_return_bps=40.0,
        meta={
            "gen1_decision": "LONG",
            "gen1_decision_confidence": 0.65,
            "gen1_fusion_state": "setup_macro_neutral",
            "net_spread_directional_return_bps": 38.0,
            "gen11_features": snapshot,
            "range_outcomes": {"levels": {"pm10": {"first_hit": "up"}}},
        },
    )
    row = flatten_gen11_episode(episode)
    assert row["session"] == "london"
    assert row["net_spread_directional_bps"] == 38.0
    assert row["range_pm10_favorable_first"] is True
