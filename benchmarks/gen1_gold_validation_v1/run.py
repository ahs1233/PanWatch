"""GEN1 Gold validation-contract benchmark.

Synthetic episodes validate plumbing, calibration math, range accounting and
anti-overclaim invariants only. They are NOT evidence of real trading edge.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from src.modules.xau.validation import evaluate_gen1_replay


def episode(index: int, *, good: bool) -> SimpleNamespace:
    if good:
        positive = (index % 10) < 7
        confidence = 0.70 if positive else 0.58
        bps = 12.0 if positive else -8.0
        first = "up" if positive else "down"
    else:
        positive = (index % 2) == 0
        confidence = 0.80
        bps = 7.0 if positive else -9.0
        first = "up" if positive else "down"
    return SimpleNamespace(
        candidate="long_setup",
        directional_return_bps=bps,
        meta={
            "gen1_decision": "LONG",
            "gen1_decision_confidence": confidence,
            "sensor_gaps": ["historical_xaut_microstructure_unavailable"],
            "range_outcomes": {
                "levels": {
                    "pm10": {"first_hit": first},
                    "pm20": {"first_hit": first if positive else "none"},
                    "pm30": {"first_hit": "none"},
                }
            },
        },
    )


def main() -> None:
    positive_fixture = evaluate_gen1_replay(
        [episode(i, good=True) for i in range(120)]
    )
    weak_fixture = evaluate_gen1_replay(
        [episode(i, good=False) for i in range(120)]
    )

    passed = bool(
        positive_fixture["exploratory_edge_candidate"] is True
        and positive_fixture["edge_proven"] is False
        and positive_fixture["directional_positive_rate"] == 0.7
        and positive_fixture["range_outcomes"]["pm10"]["favorable_first_rate"] == 0.7
        and weak_fixture["exploratory_edge_candidate"] is False
        and weak_fixture["edge_proven"] is False
        and positive_fixture["historical_xaut_microstructure_validated"] is False
        and "historical_xaut_microstructure_unavailable"
        in positive_fixture["sensor_gaps"]
    )

    result = {
        "benchmark": "gen1-gold-validation-v1",
        "synthetic_data_warning": (
            "Synthetic benchmark validates validation plumbing only; "
            "it is not evidence of real-world trading edge."
        ),
        "positive_fixture": {
            "status": positive_fixture["validation_status"],
            "directional_n": positive_fixture["directional_decision_count"],
            "positive_rate": positive_fixture["directional_positive_rate"],
            "average_bps": positive_fixture["average_directional_return_bps"],
            "brier": positive_fixture["calibration"]["brier_score"],
            "ece": positive_fixture["calibration"]["expected_calibration_error"],
            "pm10": positive_fixture["range_outcomes"]["pm10"],
            "edge_proven": positive_fixture["edge_proven"],
        },
        "weak_fixture": {
            "status": weak_fixture["validation_status"],
            "directional_n": weak_fixture["directional_decision_count"],
            "positive_rate": weak_fixture["directional_positive_rate"],
            "average_bps": weak_fixture["average_directional_return_bps"],
            "brier": weak_fixture["calibration"]["brier_score"],
            "edge_proven": weak_fixture["edge_proven"],
        },
        "anti_overclaim": {
            "historical_xaut_validated": positive_fixture[
                "historical_xaut_microstructure_validated"
            ],
            "sensor_gaps": positive_fixture["sensor_gaps"],
        },
        "passed": passed,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
