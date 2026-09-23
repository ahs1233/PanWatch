from types import SimpleNamespace

from src.modules.xau.validation import evaluate_gen1_live_observations, evaluate_gen1_replay


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



def test_live_validation_uses_full_pipeline_observations_and_sampled_first_touch():
    rows = [
        SimpleNamespace(meta={
            "decision": "LONG",
            "decision_confidence": 0.7,
            "horizon_outcomes": {"60m": {"directional_return_bps": 10.0}},
            "live_range_outcomes": {"levels": {
                "pm10": {"first_hit": "up"},
                "pm20": {"first_hit": "none"},
                "pm30": {"first_hit": "none"},
            }},
        }),
        SimpleNamespace(meta={
            "decision": "SHORT",
            "decision_confidence": 0.65,
            "horizon_outcomes": {"60m": {"directional_return_bps": -5.0}},
            "live_range_outcomes": {"levels": {
                "pm10": {"first_hit": "up"},
                "pm20": {"first_hit": "none"},
                "pm30": {"first_hit": "none"},
            }},
        }),
    ]
    result = evaluate_gen1_live_observations(rows)
    assert result["full_live_pipeline_including_xaut"] is True
    assert result["completed_60m_directional_count"] == 2
    assert result["directional_positive_rate_60m"] == 0.5
    assert result["range_outcomes"]["pm10"]["favorable_first"] == 1
    assert result["range_outcomes"]["pm10"]["adverse_first"] == 1
    assert result["edge_proven"] is False
    assert result["first_touch_precision"] == "sampled_not_intrabar_exact"



def _live_oos_row(index, *, win, revision="rev-test"):
    from datetime import datetime, timedelta, timezone
    observed = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=20 * index)
    directional = 12.0 if win else -4.0
    return SimpleNamespace(
        observed_at=observed,
        meta={
            "decision": "LONG",
            "decision_confidence": 0.70,
            "strategy_revision": revision,
            "revision_pinning_available": True,
            "missing_layers": [],
            "regime": "trend",
            "observation_source": "scheduled_forward_validation",
            "horizon_outcomes": {"60m": {"directional_return_bps": directional}},
            "live_range_outcomes": {"levels": {
                "pm10": {"first_hit": "up" if win else "down"},
                "pm20": {"first_hit": "none"},
                "pm30": {"first_hit": "none"},
            }},
        },
    )


def test_forward_oos_gate_passes_only_on_revision_pinned_statistical_confirmation():
    rows = [_live_oos_row(i, win=(i % 4 != 0)) for i in range(180)]
    result = evaluate_gen1_live_observations(rows)
    oos = result["forward_oos"]
    assert oos["protocol"] == "prequential_forward_oos_v1"
    assert oos["decorrelated_completed_current_revision"] == 180
    assert oos["holdout_count"] >= 50
    assert oos["wilson_95_lower"] > 0.50
    assert oos["bootstrap_mean_bps_95_lower"] > 0
    assert oos["calibration"]["brier_score"] < 0.25
    assert oos["passed"] is True
    assert oos["status"] == "oos_statistical_confirmation_candidate"
    assert result["edge_proven"] is False


def test_forward_oos_gate_rejects_weak_holdout():
    rows = [_live_oos_row(i, win=(i % 2 == 0)) for i in range(180)]
    result = evaluate_gen1_live_observations(rows)
    oos = result["forward_oos"]
    assert oos["passed"] is False
    assert oos["status"] == "oos_gates_not_passed"
    assert oos["gates"]["win_rate_wilson_lower_above_50pct"] is False


def test_forward_oos_uses_latest_revision_only():
    old = [_live_oos_row(i, win=True, revision="old") for i in range(30)]
    new = [_live_oos_row(i + 100, win=True, revision="new") for i in range(40)]
    result = evaluate_gen1_live_observations(old + new)
    assert result["forward_oos"]["strategy_revision"] == "new"
    assert result["forward_oos"]["decorrelated_completed_current_revision"] == 40
    assert result["forward_oos"]["passed"] is False
