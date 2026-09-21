from __future__ import annotations

from src.modules.xau.cognition import (
    build_cognitive_state,
    build_market_state_vector,
    state_vector_similarity,
)


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
            "age_seconds": 1.0,
            "is_stale": False,
        },
        "analysis_reference": {
            "price": spot_price,
            "source": "test-spot",
            "age_seconds": 1.0,
            "is_stale": False,
            "kind": "indicative_spot",
            "execution_eligible": False,
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
    assert abs(sum(state["regime"]["probabilities"].values()) - 1.0) < 0.01
    assert state["hypotheses"][0]["direction"] in {"short", "none"}
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



def test_market_state_vector_similarity_prefers_nearby_episode():
    technical = _technical()
    macro = {"bias": -1, "confidence": 0.7, "event_risk": False}
    current = build_market_state_vector(technical, macro)
    identical = dict(current)
    distant = dict(current)
    distant.update({
        "candidate": "long_setup",
        "alignment": "bullish",
        "session": "asia",
        "regime": "range_rotation",
        "directional_pressure": 1.0,
        "macro_bias": 1.0,
        "rsi_5m_norm": 0.8,
    })

    assert state_vector_similarity(current, identical) == 1.0
    assert state_vector_similarity(current, distant) < 0.75


def test_probabilistic_regime_exposes_distribution_not_binary_label():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    probabilities = state["regime"]["probabilities"]
    assert len(probabilities) >= 6
    assert state["regime"]["label"] == max(probabilities, key=probabilities.get)
    assert abs(sum(probabilities.values()) - 1.0) < 0.01


def test_historical_probability_changes_calibration_basis():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 20,
            "similar_samples": 20,
            "expectancy_r": 0.4,
            "profit_factor": 1.6,
            "posterior_win_probability": 0.68,
            "calibration_sample_count": 20,
            "brier_score": 0.18,
            "expected_calibration_error": 0.08,
        },
    )
    assert state["confidence"]["calibration_basis"] == "empirical_bayesian_history"
    assert state["confidence"]["historical_probability"] == 0.68
    assert state["confidence"]["sample_count"] == 20


def test_sensor_disagreement_reduces_data_quality():
    technical = _technical()
    technical["spot_minus_proxy_bps"] = 12.0
    degraded = build_cognitive_state(
        technical,
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    clean = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )

    assert degraded["data_quality"]["score"] < clean["data_quality"]["score"]
    assert "spot_structure_disagreement" in degraded["data_quality"]["issues"]



def test_regime_is_probabilistic_not_single_rule_label():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
    )
    probabilities = state["regime"]["probabilities"]
    assert len(probabilities) >= 6
    assert abs(sum(probabilities.values()) - 1.0) < 0.01
    assert state["regime"]["label"] in probabilities
    assert 0.0 < state["regime"]["confidence"] < 1.0


def test_state_vector_similarity_prefers_nearby_market_state():
    technical = _technical()
    macro = {"bias": -1, "confidence": 0.7, "event_risk": False}
    current = build_market_state_vector(technical, macro)
    same = dict(current)
    far = {
        **current,
        "candidate": "long_setup",
        "alignment": "bullish",
        "regime": "range_rotation",
        "directional_pressure": 1.0,
        "return_10m_pct": 0.45,
        "return_30m_pct": 0.80,
        "macro_bias": 1.0,
        "rsi_5m_norm": 0.8,
    }

    assert state_vector_similarity(current, same) > 0.99
    assert state_vector_similarity(current, same) > state_vector_similarity(current, far)


def test_market_state_contains_session_quality_and_microstructure_context():
    vector = build_market_state_vector(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
    )
    assert vector["candidate"] == "short_setup"
    assert vector["session"] in {
        "asia",
        "london_open",
        "london_ny_overlap",
        "new_york",
        "late_us",
    }
    assert "data_quality" in vector
    assert "spread_bps" in vector
    assert "spot_proxy_basis_bps" in vector


def test_low_quality_data_forces_adversarial_veto():
    technical = _technical()
    technical["indicative_spot"]["is_stale"] = True
    technical["indicative_spot"]["age_seconds"] = 120
    technical["blocked"] = True

    state = build_cognitive_state(
        technical,
        {"bias": -1, "confidence": 0.8, "event_risk": False},
    )
    assert state["regime"]["label"] == "data_uncertain"
    assert state["adversarial"]["veto"] is True
    assert state["meta_controller"]["paper_entry_allowed"] is False


def test_empirical_history_changes_confidence_basis():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 30,
            "similar_samples": 30,
            "expectancy_r": 0.4,
            "profit_factor": 1.6,
            "posterior_win_probability": 0.68,
            "calibration_sample_count": 30,
            "brier_score": 0.19,
            "expected_calibration_error": 0.08,
        },
    )
    assert state["confidence"]["calibration_basis"] == "empirical_bayesian_history"
    assert state["confidence"]["sample_count"] == 30
    assert state["confidence"]["brier_score"] == 0.19



def test_calibration_error_raises_effective_entry_threshold():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 20,
            "similar_samples": 20,
            "expectancy_r": 0.1,
            "profit_factor": 1.1,
            "posterior_win_probability": 0.56,
            "calibration_sample_count": 20,
            "brier_score": 0.31,
            "expected_calibration_error": 0.28,
        },
        min_confidence=0.58,
    )
    adaptation = state["meta_controller"]["threshold_adaptation"]
    assert adaptation["effective"] > adaptation["base"]
    assert "calibration_error_raise_threshold" in adaptation["reasons"]


def test_well_calibrated_positive_history_only_relaxes_threshold_slightly():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 40,
            "similar_samples": 40,
            "expectancy_r": 0.5,
            "profit_factor": 1.8,
            "posterior_win_probability": 0.66,
            "calibration_sample_count": 40,
            "brier_score": 0.15,
            "expected_calibration_error": 0.05,
        },
        min_confidence=0.58,
    )
    adaptation = state["meta_controller"]["threshold_adaptation"]
    assert adaptation["base"] - adaptation["effective"] <= 0.0151
    assert adaptation["effective"] >= 0.50



def test_micro_substitute_does_not_count_1m_as_missing_quality_penalty():
    technical = _technical()
    technical["technical_mode"] = "spot_micro_plus_spot_5m_15m"
    technical["frames"].pop("1m", None)
    technical["micro"]["status"] = "ready"
    technical["micro"]["is_stale"] = False
    technical["micro"]["age_seconds"] = 8.0
    technical["indicative_spot"]["age_seconds"] = 4.0

    state = build_cognitive_state(
        technical,
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    issues = state["data_quality"]["issues"]
    assert "1m_missing" not in issues
    assert "micro_substitutes_1m" in issues
    assert state["data_quality"]["score"] >= 0.70



def test_mid_only_slightly_aged_spot_keeps_analytical_quality_but_not_execution_readiness():
    technical = _technical()
    technical["technical_mode"] = "spot_micro_plus_spot_5m_15m"
    technical["frames"].pop("1m", None)
    technical["indicative_spot"].update({
        "bid": None,
        "ask": None,
        "age_seconds": 60.0,
        "is_stale": False,
    })
    technical["analysis_reference"] = {
        "price": 4340.1,
        "source": "test-micro",
        "age_seconds": 12.0,
        "is_stale": False,
        "kind": "micro_fallback",
        "execution_eligible": False,
    }
    technical["micro"]["status"] = "ready"
    technical["micro"]["is_stale"] = False
    technical["micro"]["age_seconds"] = 12.0
    technical["warnings"] = ["biquote_ohlc_fallback_active"]

    state = build_cognitive_state(
        technical,
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    quality = state["data_quality"]
    fill = quality["sensors"]["fill_readiness"]
    assert "analysis_reference_stale" not in quality["issues"]
    assert quality["score"] >= 0.85
    assert fill["ready_for_paper_fill"] is False
    assert "fill_bid_ask_missing" in fill["issues"]


def test_stale_fill_quote_does_not_poison_fresh_micro_analysis():
    technical = _technical()
    technical["technical_mode"] = "spot_micro_plus_spot_5m_15m"
    technical["frames"].pop("1m", None)
    technical["indicative_spot"].update({
        "age_seconds": 600.0,
        "is_stale": True,
        "bid": None,
        "ask": None,
    })
    technical["analysis_reference"] = {
        "price": 4340.2,
        "source": "xaus.com:intraday",
        "age_seconds": 8.0,
        "is_stale": False,
        "kind": "micro_fallback",
        "execution_eligible": False,
    }
    technical["micro"].update({
        "status": "ready",
        "is_stale": False,
        "age_seconds": 8.0,
    })

    state = build_cognitive_state(
        technical,
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    quality = state["data_quality"]
    fill = quality["sensors"]["fill_readiness"]
    assert quality["score"] >= 0.90
    assert state["regime"]["label"] != "data_uncertain"
    assert fill["ready_for_paper_fill"] is False
    assert "fill_quote_stale" in fill["issues"]
    assert "fill_bid_ask_missing" in fill["issues"]



def test_cognition_uses_explicit_rollover_fill_state_without_double_penalty():
    technical = _technical()
    technical["indicative_spot"].update({
        "bid": None,
        "ask": None,
        "fill_state": "market_closed_or_rollover",
        "fill_source": "biquote.io:MT5",
        "fill_age_seconds": 900.0,
        "market_state": "closed",
    })
    state = build_cognitive_state(
        technical,
        {"bias": -1, "confidence": 0.6, "event_risk": False},
    )
    fill = state["data_quality"]["sensors"]["fill_readiness"]
    assert fill["score"] == 0.0
    assert fill["market"]["state"] == "closed"
    assert fill["issues"] == ["fill_market_closed"]
    assert state["data_quality"]["score"] > 0.70



def test_shadow_memory_needs_enough_samples_before_affecting_confidence():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "shadow_memory": {
                "sample_count": 7,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 20.0,
                "average_similarity": 0.9,
                "decision_filtered": True,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    assert state["memory"]["shadow_confidence_adjustment"] == 0.0


def test_shadow_memory_is_bounded_research_prior_not_trade_calibration():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "calibration_sample_count": 0,
            "shadow_memory": {
                "sample_count": 40,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 50.0,
                "average_similarity": 1.0,
                "decision_filtered": True,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    memory = state["memory"]
    assert 0.0 < memory["shadow_confidence_adjustment"] <= 0.015
    assert memory["trade_confidence_adjustment"] == 0.0
    assert memory["calibration_sample_count"] == 0
    assert state["confidence"]["calibration_basis"] == "prior_shrunk_no_history"



def test_replay_memory_requires_larger_sample_and_is_bounded():
    insufficient = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "replay_memory": {
                "sample_count": 14,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 50.0,
                "average_similarity": 1.0,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    assert insufficient["memory"]["replay_confidence_adjustment"] == 0.0

    sufficient = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "replay_memory": {
                "sample_count": 80,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 50.0,
                "average_similarity": 1.0,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    memory = sufficient["memory"]
    assert 0.0 < memory["replay_confidence_adjustment"] <= 0.01
    assert memory["trade_confidence_adjustment"] == 0.0
    assert memory["calibration_sample_count"] == 0
    assert sufficient["confidence"]["calibration_basis"] == "prior_shrunk_no_history"


def test_combined_research_priors_are_clamped():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "shadow_memory": {
                "sample_count": 100,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 100.0,
                "average_similarity": 1.0,
                "decision_filtered": True,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
            "replay_memory": {
                "sample_count": 100,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 100.0,
                "average_similarity": 1.0,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    assert state["memory"]["research_confidence_adjustment"] <= 0.02



def test_shadow_prior_ignores_return_magnitude_and_uses_direction_only():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "calibration_sample_count": 0,
            "shadow_memory": {
                "sample_count": 40,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": -999.0,
                "average_similarity": 1.0,
                "decision_filtered": True,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    memory = state["memory"]
    assert memory["shadow_confidence_adjustment"] == 0.008
    assert memory["trade_confidence_adjustment"] == 0.0


def test_unfiltered_shadow_memory_never_affects_confidence():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "shadow_memory": {
                "sample_count": 100,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 1000.0,
                "average_similarity": 1.0,
                "decision_filtered": False,
                "research_only": True,
            },
        },
    )
    assert state["memory"]["shadow_confidence_adjustment"] == 0.0



def test_unprotected_research_memory_never_affects_confidence():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "shadow_memory": {
                "sample_count": 100,
                "positive_rate": 1.0,
                "average_similarity": 1.0,
                "decision_filtered": True,
                "research_only": True,
                "lookahead_protected": False,
                "temporally_decorrelated": True,
            },
            "replay_memory": {
                "sample_count": 100,
                "positive_rate": 1.0,
                "similarity_weighted_return_bps": 100.0,
                "average_similarity": 1.0,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": False,
            },
        },
    )
    memory = state["memory"]
    assert memory["shadow_confidence_adjustment"] == 0.0
    assert memory["replay_confidence_adjustment"] == 0.0
    assert memory["research_confidence_adjustment"] == 0.0


def test_conflicting_research_priors_are_damped():
    state = build_cognitive_state(
        _technical(),
        {"bias": -1, "confidence": 0.7, "event_risk": False},
        memory={
            "trade_count": 0,
            "similar_samples": 0,
            "shadow_memory": {
                "sample_count": 40,
                "positive_rate": 1.0,
                "average_similarity": 1.0,
                "decision_filtered": True,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
            "replay_memory": {
                "sample_count": 80,
                "positive_rate": 0.0,
                "similarity_weighted_return_bps": -50.0,
                "average_similarity": 1.0,
                "research_only": True,
                "lookahead_protected": True,
                "temporally_decorrelated": True,
            },
        },
    )
    memory = state["memory"]
    assert memory["research_prior_conflict"] is True
    assert abs(memory["research_confidence_adjustment"]) < (
        abs(memory["shadow_confidence_adjustment"])
        + abs(memory["replay_confidence_adjustment"])
    )
