"""Deterministic PanWatch Benchmark v1 runner.

This benchmark intentionally measures only capabilities that can be exercised
offline and reproducibly today.  It does not award points for future research
features such as an Evidence Ledger, Claim Graph, or generic falsification
engine.  Those belong to the cross-domain benchmark tracks in manifest.json.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from src.modules.xau.cognition import (
    build_cognitive_state,
    build_market_state_vector,
    state_vector_similarity,
)


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    points: float
    earned: float
    detail: str


def _technical(
    candidate: str = "short_setup",
    *,
    blocked: bool = False,
    rsi: float = 44.0,
    spot_price: float = 4340.0,
) -> dict:
    direction = "bearish" if candidate == "short_setup" else "bullish"
    return {
        "candidate": candidate,
        "blocked": blocked,
        "alignment": direction,
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
            "source": "benchmark-spot",
            "age_seconds": 1.0,
            "is_stale": False,
            "kind": "indicative_spot",
            "execution_eligible": False,
        },
        "micro": {
            "status": "ready",
            "direction": direction,
            "return_10m_pct": -0.08 if candidate == "short_setup" else 0.08,
            "return_30m_pct": -0.15 if candidate == "short_setup" else 0.15,
            "is_stale": False,
            "age_seconds": 1.0,
        },
        "frames": {
            "1m": {
                "direction": direction,
                "rsi14": rsi,
                "atr_pct": 0.08,
                "breakout": "none",
                "ema_fast": spot_price + (1.0 if candidate == "short_setup" else -1.0),
                "source": "biquote.io:MT5-ohlc",
            },
            "5m": {
                "direction": direction,
                "rsi14": rsi,
                "atr_pct": 0.14,
                "breakout": "down" if candidate == "short_setup" else "up",
                "ema_fast": spot_price + (2.0 if candidate == "short_setup" else -2.0),
                "source": "biquote.io:MT5-ohlc",
            },
            "15m": {
                "direction": direction,
                "rsi14": rsi,
                "atr_pct": 0.13,
                "breakout": "none",
                "ema_fast": spot_price + (3.0 if candidate == "short_setup" else -3.0),
                "source": "biquote.io:MT5-ohlc",
            },
        },
        "technical_mode": "biquote_mt5_1m_5m_15m",
    }


def _state(
    technical: dict | None = None,
    macro: dict | None = None,
    *,
    memory: dict | None = None,
    min_confidence: float = 0.50,
) -> dict:
    return build_cognitive_state(
        technical or _technical(),
        macro or {"bias": -1, "confidence": 0.70, "event_risk": False},
        memory=memory,
        min_confidence=min_confidence,
    )


def _event_risk_veto() -> tuple[bool, str]:
    state = _state(macro={"bias": -1, "confidence": 0.8, "event_risk": True})
    ok = state["adversarial"]["veto"] is True and state["meta_controller"]["paper_entry_allowed"] is False
    return ok, "event risk must veto paper entry"


def _stale_data_veto() -> tuple[bool, str]:
    technical = _technical(blocked=True)
    technical["analysis_reference"]["is_stale"] = True
    technical["analysis_reference"]["age_seconds"] = 300
    state = _state(technical)
    ok = state["regime"]["label"] == "data_uncertain" and state["adversarial"]["veto"] is True
    return ok, "stale/blocked analytical data must force uncertainty and veto"


def _memory_calibration_direction() -> tuple[bool, str]:
    positive = _state(memory={"trade_count": 30, "expectancy_r": 0.8, "profit_factor": 2.0})
    negative = _state(memory={"trade_count": 30, "expectancy_r": -0.8, "profit_factor": 0.6})
    p = positive["confidence"]["calibrated_confidence"]
    n = negative["confidence"]["calibrated_confidence"]
    return p > n and p <= 0.95, f"positive={p:.4f}, negative={n:.4f}"


def _overextension_counter_evidence() -> tuple[bool, str]:
    state = _state(_technical(rsi=20.0), macro={"bias": -1, "confidence": 0.6, "event_risk": False})
    evidence = state["adversarial"]["counter_evidence"]
    return "short_setup_overextended_rsi" in evidence, str(evidence)


def _probabilistic_regime_consistency() -> tuple[bool, str]:
    state = _state()
    probabilities = state["regime"]["probabilities"]
    total = sum(probabilities.values())
    ok = (
        len(probabilities) >= 6
        and abs(total - 1.0) < 0.01
        and state["regime"]["label"] == max(probabilities, key=probabilities.get)
    )
    return ok, f"regimes={len(probabilities)}, total={total:.4f}"


def _sensor_disagreement_penalty() -> tuple[bool, str]:
    degraded_input = _technical()
    degraded_input["spot_minus_proxy_bps"] = 12.0
    degraded = _state(degraded_input)
    clean = _state(_technical())
    dq = degraded["data_quality"]["score"]
    cq = clean["data_quality"]["score"]
    return dq < cq, f"degraded={dq:.4f}, clean={cq:.4f}"


def _research_prior_sample_guard() -> tuple[bool, str]:
    state = _state(memory={
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
    })
    value = state["memory"]["shadow_confidence_adjustment"]
    return value == 0.0, f"adjustment={value}"


def _research_prior_integrity_guard() -> tuple[bool, str]:
    state = _state(memory={
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
            "average_similarity": 1.0,
            "research_only": True,
            "lookahead_protected": True,
            "temporally_decorrelated": False,
        },
    })
    memory = state["memory"]
    ok = memory["shadow_confidence_adjustment"] == 0.0 and memory["replay_confidence_adjustment"] == 0.0
    return ok, f"shadow={memory['shadow_confidence_adjustment']}, replay={memory['replay_confidence_adjustment']}"


def _conflicting_research_priors_damped() -> tuple[bool, str]:
    state = _state(memory={
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
    })
    m = state["memory"]
    raw = abs(m["shadow_confidence_adjustment"]) + abs(m["replay_confidence_adjustment"])
    ok = m["research_prior_conflict"] is True and abs(m["research_confidence_adjustment"]) < raw
    return ok, f"combined={m['research_confidence_adjustment']:.4f}, raw_abs={raw:.4f}"


def _realized_trade_evidence_priority() -> tuple[bool, str]:
    state = _state(memory={
        "trade_count": 40,
        "similar_samples": 40,
        "expectancy_r": 0.9,
        "profit_factor": 2.2,
        "posterior_win_probability": 0.68,
        "calibration_sample_count": 40,
        "brier_score": 0.16,
        "expected_calibration_error": 0.06,
        "shadow_memory": {
            "sample_count": 40,
            "positive_rate": 0.0,
            "average_similarity": 0.90,
            "decision_filtered": True,
            "research_only": True,
            "lookahead_protected": True,
            "temporally_decorrelated": True,
        },
        "replay_memory": {
            "sample_count": 40,
            "positive_rate": 0.0,
            "similarity_weighted_return_bps": -25.0,
            "average_similarity": 0.90,
            "research_only": True,
            "lookahead_protected": True,
            "temporally_decorrelated": True,
        },
    })
    m = state["memory"]
    ok = (
        m["trade_research_conflict"] is True
        and m["trade_confidence_adjustment"] > 0
        and m["research_confidence_adjustment"] < 0
        and abs(m["research_confidence_adjustment"]) < abs(m["trade_confidence_adjustment"])
        and m["confidence_adjustment"] > 0
    )
    return ok, f"trade={m['trade_confidence_adjustment']:.4f}, research={m['research_confidence_adjustment']:.4f}"


def _market_closed_fill_isolation() -> tuple[bool, str]:
    technical = _technical()
    technical["indicative_spot"].update({
        "bid": None,
        "ask": None,
        "fill_state": "market_closed_or_rollover",
        "fill_source": "biquote.io:MT5",
        "fill_age_seconds": 900.0,
        "market_state": "closed",
    })
    state = _state(technical, macro={"bias": -1, "confidence": 0.6, "event_risk": False})
    fill = state["data_quality"]["sensors"]["fill_readiness"]
    quality = state["data_quality"]["score"]
    ok = fill["score"] == 0.0 and fill["market"]["state"] == "closed" and quality > 0.70
    return ok, f"fill={fill['score']:.2f}, analytical_quality={quality:.2f}"


def _adaptive_threshold_on_bad_calibration() -> tuple[bool, str]:
    state = _state(
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
    ok = adaptation["effective"] > adaptation["base"] and "calibration_error_raise_threshold" in adaptation["reasons"]
    return ok, f"base={adaptation['base']:.4f}, effective={adaptation['effective']:.4f}"


def _source_family_discrimination() -> tuple[bool, str]:
    spot = _technical()
    futures = _technical()
    for frame in futures["frames"].values():
        frame["source"] = "yfinance:GC=F"
    futures["technical_mode"] = "mixed_research_fallback_1m_5m_15m"
    macro = {"bias": 0, "confidence": 0.0, "event_risk": False}
    s = build_market_state_vector(spot, macro)
    f = build_market_state_vector(futures, macro)
    ok = s["source_family"] == "xau_spot_structure" and f["source_family"] == "gc_futures_proxy"
    return ok, f"spot={s['source_family']}, futures={f['source_family']}"


def _state_similarity_discrimination() -> tuple[bool, str]:
    macro = {"bias": -1, "confidence": 0.7, "event_risk": False}
    current = build_market_state_vector(_technical(), macro)
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
    near_score = state_vector_similarity(current, same)
    far_score = state_vector_similarity(current, far)
    return near_score > 0.99 and near_score > far_score, f"same={near_score:.4f}, far={far_score:.4f}"


def _live_execution_disabled() -> tuple[bool, str]:
    state = _state()
    allowed = state["meta_controller"]["live_execution_allowed"]
    return allowed is False, f"live_execution_allowed={allowed}"


CASES: list[tuple[str, str, float, Callable[[], tuple[bool, str]]]] = [
    ("event_risk_veto", "safety", 8.0, _event_risk_veto),
    ("stale_data_veto", "data_integrity", 8.0, _stale_data_veto),
    ("memory_calibration_direction", "calibration", 7.0, _memory_calibration_direction),
    ("overextension_counter_evidence", "adversarial_reasoning", 6.0, _overextension_counter_evidence),
    ("probabilistic_regime_consistency", "probabilistic_reasoning", 7.0, _probabilistic_regime_consistency),
    ("sensor_disagreement_penalty", "data_integrity", 7.0, _sensor_disagreement_penalty),
    ("research_prior_sample_guard", "memory_integrity", 6.0, _research_prior_sample_guard),
    ("research_prior_integrity_guard", "memory_integrity", 8.0, _research_prior_integrity_guard),
    ("conflicting_research_priors_damped", "conflict_handling", 7.0, _conflicting_research_priors_damped),
    ("realized_trade_evidence_priority", "evidence_hierarchy", 8.0, _realized_trade_evidence_priority),
    ("market_closed_fill_isolation", "data_integrity", 7.0, _market_closed_fill_isolation),
    ("adaptive_threshold_on_bad_calibration", "calibration", 6.0, _adaptive_threshold_on_bad_calibration),
    ("source_family_discrimination", "source_awareness", 5.0, _source_family_discrimination),
    ("state_similarity_discrimination", "memory_reasoning", 5.0, _state_similarity_discrimination),
    ("live_execution_disabled", "safety", 5.0, _live_execution_disabled),
]


def run_benchmark() -> dict:
    results: list[CaseResult] = []
    for case_id, category, points, evaluator in CASES:
        try:
            passed, detail = evaluator()
        except Exception as exc:  # benchmark must report failures, not hide them
            passed, detail = False, f"{type(exc).__name__}: {exc}"
        results.append(
            CaseResult(
                case_id=case_id,
                category=category,
                passed=passed,
                points=points,
                earned=points if passed else 0.0,
                detail=detail,
            )
        )

    total = sum(item.points for item in results)
    earned = sum(item.earned for item in results)
    score = round((earned / total) * 100.0, 2) if total else 0.0
    passed_cases = sum(1 for item in results if item.passed)

    return {
        "benchmark": "PanWatch Benchmark v1",
        "version": "1.5.0",
        "track": "xau_decision_core",
        "score_percent": score,
        "passed_cases": passed_cases,
        "total_cases": len(results),
        "points_earned": earned,
        "points_total": total,
        "regression_gate_percent": 95.0,
        "regression_gate_passed": score >= 95.0,
        "track_coverage": {
            "automated": 6,
            "specified_not_automated": 1,
            "total": 7,
        },
        "limitations": [
            "This is a deterministic core benchmark, not proof of superior research quality.",
            "Economic evidence integrity, Claim Graph/Falsification, Persistent Belief State, Automatic Research, and General Claim Acquisition are automated separately; company/sector comparative research is still specified but not automated.",
            "No external LLM or human baseline is scored in v1.",
            "Evidence Foundation, Claim Graph/Falsification, Persistent Belief State, Automatic Research, and General Claim Acquisition are scored in separate tracks; external comparative research quality remains outside this core score.",
        ],
        "cases": [asdict(item) for item in results],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PanWatch Benchmark v1")
    parser.add_argument("--json", dest="json_path", help="Write machine-readable result JSON")
    parser.add_argument("--no-gate", action="store_true", help="Do not fail on regression gate")
    args = parser.parse_args()

    report = run_benchmark()
    print(json.dumps(report, indent=2, sort_keys=True))

    if args.json_path:
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.no_gate:
        return 0
    return 0 if report["regression_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
