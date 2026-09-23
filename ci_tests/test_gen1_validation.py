from types import SimpleNamespace

from src.modules.xau.validation import evaluate_gen1_replay


def _episode(decision, confidence, directional_bps, first="up"):
    return SimpleNamespace(
        candidate="long_setup",
        directional_return_bps=directional_bps,
        meta={
            "gen1_decision": decision,
            "gen1_decision_confidence": confidence,
            "sensor_gaps": ["historical_xaut_microstructure_unavailable"],
            "range_outcomes": {
                "levels": {
                    "pm10": {"first_hit": first},
                    "pm20": {"first_hit": "none"},
                    "pm30": {"first_hit": "none"},
                }
            },
        },
    )


def test_validation_measures_calibration_ranges_and_never_claims_proven_edge():
    episodes = [
        _episode("LONG", 0.70, 12.0, "up"),
        _episode("LONG", 0.65, -8.0, "down"),
        _episode("WAIT", 0.55, 4.0, "none"),
    ]
    result = evaluate_gen1_replay(episodes)
    assert result["episode_count"] == 3
    assert result["directional_decision_count"] == 2
    assert result["directional_positive_rate"] == 0.5
    assert result["calibration"]["sample_count"] == 2
    assert result["range_outcomes"]["pm10"]["favorable_first"] == 1
    assert result["range_outcomes"]["pm10"]["adverse_first"] == 1
    assert result["edge_proven"] is False
    assert result["historical_xaut_microstructure_validated"] is False


def test_validation_requires_large_sample_before_edge_candidate_label():
    episodes = [_episode("LONG", 0.7, 10.0, "up") for _ in range(99)]
    result = evaluate_gen1_replay(episodes)
    assert result["validation_status"] == "insufficient_directional_sample"
    assert result["exploratory_edge_candidate"] is False
    assert result["edge_proven"] is False
