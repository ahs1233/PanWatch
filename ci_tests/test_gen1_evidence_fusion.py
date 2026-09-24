from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion


def _base(candidate="long_setup"):
    technical = {
        "candidate": candidate,
        "market_context": {
            "bias": {
                "today_score": 0.55,
                "today_direction": "bullish",
                "daily": {"available": True},
            },
            "smart_money": {"score": 0.35, "manual_score": 0.35, "bias": "bullish"},
        },
        "xaut_order_flow": {
            "status": "ready",
            "trade_tape": {"trade_count": 500, "coverage_seconds": 1200},
            "basis": {"basis_bps": 2.0},
            "flow": {"5m": {"available": True, "delta_ratio": 0.45}},
            "footprint": {
                "available": True,
                "levels": [
                    {"total_volume": 4.0, "delta": 2.0},
                    {"total_volume": 2.0, "delta": 0.5},
                ],
            },
            "volume_profile": {
                "status": "ready",
                "levels": [
                    {"volume": 4.0, "delta": 2.0},
                    {"volume": 2.0, "delta": 0.5},
                ],
            },
            "raw_book": {"pm10": {"imbalance": 0.35}},
            "absorption": {"state": "none"},
        },
    }
    macro = {"bias": 1, "confidence": 0.75}
    fusion = {
        "technical_candidate": candidate,
        "macro_ready": True,
        "paper_entry_allowed": True,
        "cognitive_confidence": 0.72,
        "cognition": {"directional_edge": {"score": 0.42, "strength": 0.42, "direction": "bullish"}},
    }
    memory = {
        "available": True,
        "similar_samples": 18,
        "calibration_sample_count": 18,
        "trade_count": 18,
        "posterior_win_probability": 0.62,
        "brier_score": 0.21,
        "expected_calibration_error": 0.08,
    }
    return technical, macro, fusion, memory


def test_evidence_fusion_confirms_only_after_independent_family_agreement():
    technical, macro, fusion, memory = _base()
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    assert result["decision"] == "LONG"
    assert result["score"] > 0.10
    assert result["coverage"] > 0.80
    assert result["xaut"]["global_xauusd_order_flow"] is False
    assert result["is_validated_win_probability"] is False


def test_evidence_fusion_xaut_is_one_bounded_family_not_three_votes():
    technical, macro, fusion, memory = _base()
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    xaut_families = [row for row in result["families"] if row["name"] == "xaut_microstructure"]
    assert len(xaut_families) == 1
    detail = xaut_families[0]["detail"]
    assert "executed_flow_5m" in detail["components"]
    assert "footprint_delta" in detail["components"]
    assert "profile_delta" in detail["components"]


def test_evidence_fusion_waits_on_strong_xaut_conflict():
    technical, macro, fusion, memory = _base()
    technical["xaut_order_flow"]["flow"]["5m"]["delta_ratio"] = -0.95
    technical["xaut_order_flow"]["footprint"]["levels"] = [
        {"total_volume": 8.0, "delta": -7.0},
    ]
    technical["xaut_order_flow"]["volume_profile"]["levels"] = [
        {"volume": 8.0, "delta": -7.0},
    ]
    technical["xaut_order_flow"]["raw_book"]["pm10"]["imbalance"] = -0.9
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    assert result["decision"] == "WAIT"
    assert any(row["family"] == "xaut_microstructure" for row in result["conflicts"])



def test_evidence_fusion_applies_bounded_empirical_calibration_when_bin_is_mature():
    technical, macro, fusion, memory = _base()
    memory.update({
        "calibration_sample_count": 40,
        "expected_calibration_error": 0.08,
        "calibration_bins": [
            {"index": 0, "lower": 0.0, "upper": 0.2, "count": 0, "observed_rate": None},
            {"index": 1, "lower": 0.2, "upper": 0.4, "count": 0, "observed_rate": None},
            {"index": 2, "lower": 0.4, "upper": 0.6, "count": 0, "observed_rate": None},
            {"index": 3, "lower": 0.6, "upper": 0.8, "count": 40, "observed_rate": 0.58},
            {"index": 4, "lower": 0.8, "upper": 1.0, "count": 0, "observed_rate": None},
        ],
    })
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    calibration = result["calibration"]
    assert calibration["applied"] is True
    assert calibration["method"] == "paper_cognition_bin_shrinkage_v1"
    assert calibration["calibrated_cognitive_confidence"] < calibration["raw_cognitive_confidence"]
    assert result["is_validated_win_probability"] is False


def test_evidence_fusion_refuses_empirical_calibration_on_small_history():
    technical, macro, fusion, memory = _base()
    memory.update({
        "calibration_sample_count": 10,
        "calibration_bins": [
            {"index": 3, "lower": 0.6, "upper": 0.8, "count": 10, "observed_rate": 0.9},
        ],
    })
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    assert result["calibration"]["applied"] is False
    assert result["confidence_kind"] == "heuristic_unvalidated_score"


def test_evidence_fusion_exposes_directional_support_and_opposition():
    technical, macro, fusion, memory = _base()
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    directional = result["directional_evidence"]
    assert directional["long_support"] > directional["short_support"]
    assert 0.0 <= directional["conflict_score"] <= 1.0
    assert directional["dominant_side"] == "long"


def test_counter_flow_is_classification_not_entry_signal():
    technical, macro, fusion, memory = _base(candidate="short_setup")
    technical["gold_market_fusion"] = {
        "status": "ready",
        "venue_count": 3,
        "independent_source_count": 2,
        "composite_flow_score": 0.40,
        "composite_footprint_score": 0.35,
        "composite_liquidity_score": 0.25,
        "composite_microstructure_score": 0.34,
        "market_agreement_score": 82.0,
    }
    fusion["paper_entry_allowed"] = False
    fusion["cognition"] = {
        "directional_edge": {"score": -0.30, "strength": 0.30, "direction": "bearish"},
        "directional_state": {
            "classification": "bearish_pullback_inside_bullish_structure",
            "counter_flow_short": True,
            "counter_flow_long": False,
            "source_conflicts": ["htf_vs_intraday"],
        },
        "execution_plan": {
            "side": "short",
            "action": "WAIT_CONFIRMATION",
            "setup_type": "counter_flow_short",
        },
    }
    result = build_gen1_evidence_fusion(technical, macro, fusion, memory=memory)
    assert result["decision"] == "COUNTER_FLOW_SHORT"
    assert "counter_flow_short_is_classification_not_entry" in result["reasons"]
    assert result["execution_allowed"] is False
