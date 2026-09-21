from __future__ import annotations

from src.modules.xau.cognition import build_cognitive_state


def _technical(candidate="short_setup", *, blocked=False, rsi=44.0, spot_price=4340.0):
    return {
        "candidate": candidate,
        "blocked": blocked,
        "alignment": "bearish" if candidate == "short_setup" else "bullish",
        "warnings": [],
        "atr_reference": 10.0,
        "swing_high_reference": 4355.0,
        "swing_low_reference": 4320.0,
        "indicative_spot": {
            "price": spot_price,
            "bid": spot_price - 0.1,
            "ask": spot_price + 0.1,
            "is_stale": False,
        },
        "micro": {
            "status": "ready",
            "direction": "bearish" if candidate == "short_setup" else "bullish",
            "return_10m_pct": -0.08 if candidate == "short_setup" else 0.08,
            "return_30m_pct": -0.15 if candidate == "short_setup" else 0.15,
            "is_stale": False,
        },
        "frames": {
            "1m": {
                "direction": "bearish" if candidate == "short_setup" else "bullish",
                "rsi14": rsi,
                "atr_pct": 0.08,
                "breakout": "none",
                "ema_fast": spot_price + (1.0 if candidate == "short_setup" else -1.0),
            },
            "5m": {
                "direction": "bearish" if candidate == "short_setup" else "bullish",
                "rsi14": rsi,
                "atr_pct": 0.14,
                "breakout": "down" if candidate == "short_setup" else "up",
                "ema_fast": spot_price + (2.0 if candidate == "short_setup" else -2.0),
            },
            "15m": {
                "direction": "bearish" if candidate == "short_setup" else "bullish",
                "rsi14": rsi,
                "atr_pct": 0.13,
                "breakout": "none",
                "ema_fast": spot_price + (3.0 if candidate == "short_setup" else -3.0),
            },
        },
    }


def test_cognition_builds_all_layers_and_can_allow_clean_setup():
    state = build_cognitive_state(
        _technical(),
        {
            "bias": -1,
            "confidence": 0.72,
            "event_risk": False,
        },
        memory={"trade_count": 12, "expectancy_r": 0.3, "profit_factor": 1.4},
        min_confidence=0.50,
    )

    assert state["regime"]["label"] in {"trend_bear", "breakout_expansion"}
    assert state["hypotheses"][0]["name"] == "trend_continuation"
    assert state["adversarial"]["veto"] is False
    assert state["confidence"]["calibrated_confidence"] >= 0.50
    assert state["meta_controller"]["decision"] in {"eligible", "wait"}
    assert state["meta_controller"]["live_execution_allowed"] is False


def test_event_risk_hard_vetoes_entry():
    state = build_cognitive_state(
        _technical(),
        {
            "bias": -1,
            "confidence": 0.8,
            "event_risk": True,
        },
        min_confidence=0.50,
    )
    assert state["adversarial"]["veto"] is True
    assert state["meta_controller"]["paper_entry_allowed"] is False


def test_memory_changes_calibrated_confidence_without_claiming_certainty():
    positive = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={"trade_count": 30, "expectancy_r": 0.8, "profit_factor": 2.0},
    )
    negative = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={"trade_count": 30, "expectancy_r": -0.8, "profit_factor": 0.6},
    )

    assert (
        positive["confidence"]["calibrated_confidence"]
        > negative["confidence"]["calibrated_confidence"]
    )
    assert positive["confidence"]["calibrated_confidence"] <= 0.95


def test_overextended_short_is_challenged_by_adversarial_layer():
    state = build_cognitive_state(
        _technical(rsi=20.0),
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    assert "short_setup_overextended_rsi" in state["adversarial"]["counter_evidence"]
