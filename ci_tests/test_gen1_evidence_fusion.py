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
