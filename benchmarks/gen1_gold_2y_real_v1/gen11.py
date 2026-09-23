"""GEN1.1 two-year conditional-edge runner.

Research-only wrapper around the exact GEN1 2Y replay. It captures compact
pre-decision features without changing the Evidence Fusion return value, then
runs selector-blinded chronological discovery/validation/final-holdout analysis.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from benchmarks.gen1_gold_2y_real_v1 import run as base
from benchmarks.gen1_gold_2y_real_v1.edge_analysis import analyze_gen11
from src.platform.marketdata.xau_models import XAUTimeframe


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _feature_snapshot(
    technical: dict[str, Any],
    macro: dict[str, Any],
    fusion: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    context = dict(technical.get("market_context") or {})
    bias = dict(context.get("bias") or {})
    frames = dict(technical.get("frames") or {})
    atr = _num(technical.get("atr_reference"))
    entry = _num((technical.get("analysis_reference") or {}).get("price"))

    intraday = {}
    for name in ("1m", "5m", "15m"):
        frame = dict(frames.get(name) or {})
        intraday[name] = {
            key: frame.get(key)
            for key in (
                "close", "ema_fast", "ema_slow", "rsi14", "atr14", "atr_pct",
                "breakout", "direction", "recent_swing_high", "recent_swing_low",
            )
        }

    htf = {}
    for name in ("monthly", "weekly", "daily", "h4", "h1"):
        state = dict(bias.get(name) or {})
        htf[name] = {
            "direction": state.get("direction"),
            "score": state.get("score"),
            "close": state.get("close"),
            "ema": dict(state.get("ema") or {}),
            "slope_20": state.get("slope_20"),
            "available": bool(state.get("available")),
            "bar_count": state.get("bar_count"),
        }

    smart = dict(context.get("smart_money") or {})
    dealing = dict(smart.get("dealing_range") or {})
    cash = dict(context.get("cash_flow") or {})
    profile = dict(context.get("volume_profile") or {})
    swing_high = _num(technical.get("swing_high_reference"))
    swing_low = _num(technical.get("swing_low_reference"))

    return {
        "technical_alignment": technical.get("alignment"),
        "atr_reference": atr,
        "swing_high_distance_atr": (
            (swing_high - entry) / atr
            if entry is not None and swing_high is not None and atr not in (None, 0.0)
            else None
        ),
        "swing_low_distance_atr": (
            (entry - swing_low) / atr
            if entry is not None and swing_low is not None and atr not in (None, 0.0)
            else None
        ),
        "intraday": intraday,
        "htf": htf,
        "market": {
            "htf_composite_score": bias.get("composite_score"),
            "htf_composite_direction": bias.get("composite_direction"),
            "today_score": bias.get("today_score"),
            "today_direction": bias.get("today_direction"),
            "cash_flow_score": cash.get("score"),
            "cash_flow_direction": cash.get("direction"),
            "smart_money_score": smart.get("score"),
            "smart_money_bias": smart.get("bias"),
            "volume_profile_location": profile.get("location"),
            "dealing_zone": dealing.get("zone"),
            "break_of_structure": smart.get("break_of_structure"),
            "liquidity_sweep": smart.get("liquidity_sweep"),
            "displacement": smart.get("displacement"),
        },
        "macro_bias": macro.get("bias"),
        "macro_bias_label": macro.get("bias_label"),
        "macro_proxy_score": macro.get("proxy_score"),
        "macro_confidence": macro.get("confidence"),
        "macro_drivers": list(macro.get("drivers") or []),
        "fusion_state": fusion.get("state"),
        "evidence_decision": evidence.get("decision"),
        "evidence_confidence": evidence.get("decision_confidence"),
    }


def _parity(report_path: Path, episodes: dict[int, list[Any]]) -> dict[str, Any]:
    previous = json.loads(report_path.read_text(encoding="utf-8"))
    checks, passed = [], True
    for horizon in (60, 240):
        current = {
            "overall": base.stats_for(episodes[horizon], horizon),
            "reference_year": base.stats_for(
                base.split_period(episodes[horizon], base.START, base.SPLIT), horizon
            ),
            "holdout_year": base.stats_for(
                base.split_period(episodes[horizon], base.SPLIT, base.END), horizon
            ),
        }
        expected = previous[f"horizon_{horizon}m"]
        for period in ("overall", "reference_year", "holdout_year"):
            for key in (
                "episode_count_raw", "episode_count_horizon_decorrelated",
                "candidate_net_mean_bps", "final_gen1_directional_n",
                "final_gen1_net_mean_bps", "final_gen1_net_positive_rate",
            ):
                a, b = current[period].get(key), expected[period].get(key)
                if isinstance(a, float) or isinstance(b, float):
                    ok = (a is None and b is None) or (
                        a is not None and b is not None and abs(float(a) - float(b)) <= 1e-4
                    )
                else:
                    ok = a == b
                checks.append({"horizon":horizon,"period":period,"metric":key,
                               "current":a,"baseline_artifact":b,"match":bool(ok)})
                passed = passed and bool(ok)
    return {
        "implementation":"same_native_replay_with_observational_feature_wrapper",
        "passed":passed,"check_count":len(checks),
        "failed_checks":[row for row in checks if not row["match"]],
        "checks":checks,
    }


def main() -> None:
    started = time.monotonic()
    out = base.OUT
    out.mkdir(parents=True, exist_ok=True)
    baseline_report_path = out / "gen1_2y_report.json"
    if not baseline_report_path.exists():
        raise SystemExit("baseline report missing; run run.py before gen11.py")

    m1, quotes, dataset = base.download_dataset(base.START, base.END)
    bars_by_tf = {
        XAUTimeframe.M1: m1,
        XAUTimeframe.M5: base.resample(m1, XAUTimeframe.M5),
        XAUTimeframe.M15: base.resample(m1, XAUTimeframe.M15),
        XAUTimeframe.H1: base.resample(m1, XAUTimeframe.H1),
        XAUTimeframe.H4: base.resample(m1, XAUTimeframe.H4),
        XAUTimeframe.D1: base.resample(m1, XAUTimeframe.D1),
    }
    macro_provider, macro_diagnostics = base.build_historical_macro_proxy(base.START, base.END)

    feature_cache: dict[str, dict[str, Any]] = {}
    original = base.build_gen1_evidence_fusion

    def capture(technical, macro, fusion, memory=None, require_xaut=True):
        evidence = original(
            technical, macro, fusion, memory=memory, require_xaut=require_xaut
        )
        observed = str(technical.get("observed_at") or "")
        if observed:
            feature_cache[observed] = _feature_snapshot(
                technical, macro, fusion, evidence
            )
        return evidence

    base.build_gen1_evidence_fusion = capture
    try:
        episodes = base.run_replay_horizons(
            bars_by_tf, quotes, (60, 240), macro_provider
        )
    finally:
        base.build_gen1_evidence_fusion = original

    missing_features = 0
    for rows in episodes.values():
        for episode in rows:
            key = episode.observed_at.isoformat()
            features = feature_cache.get(key, {})
            episode.meta["research_features"] = features
            if not features:
                missing_features += 1

    parity = _parity(baseline_report_path, episodes)
    result = analyze_gen11(episodes[60], episodes[240], out)
    result["native_baseline_parity"] = parity
    result["feature_capture"] = {
        "unique_snapshots": len(feature_cache),
        "episode_feature_misses": missing_features,
    }
    result["dataset_integrity"] = {
        "m1_bar_count": dataset.get("m1_bar_count"),
        "data_days": dataset.get("data_days"),
        "source": dataset.get("source"),
    }
    result["macro_proxy_diagnostics"] = macro_diagnostics
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    (out / "gen1_1_edge_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(json.dumps({
        "phase": "gen1_1_complete",
        "native_baseline_parity": parity["passed"],
        "feature_capture_misses": missing_features,
        "selected_rule_count": result.get("selected_rule_count"),
        "historical_statistical_confirmation_candidate": result.get("historical_statistical_confirmation_candidate"),
        "edge_proven": result.get("edge_proven"),
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2, sort_keys=True))

    if missing_features:
        raise SystemExit(2)
    if not parity["passed"]:
        raise SystemExit(3)
    if not result.get("independent_validation", {}).get("parity"):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
