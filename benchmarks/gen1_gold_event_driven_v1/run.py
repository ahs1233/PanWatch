"""GEN1 Gold event-driven two-year research backtest.

This benchmark scans closed bars continuously but only opens a trade when a
complete strategy-specific setup exists. It manages one position at a time
minute-by-minute until target, stop, or expiry.

Research-only. It must not place live or paper orders.
"""

from __future__ import annotations

import csv
import json
import os
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.gen1_gold_2y_real_v1 import run as legacy
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion
from src.modules.xau.market_context import build_market_context
from src.modules.xau.service import build_decision_fusion
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe

START = legacy.START
SPLIT = legacy.SPLIT
END = legacy.END
STEP_MINUTES = int(os.getenv("GEN1_EVENT_STEP_MINUTES", "5"))
OUT = Path(os.getenv("GEN1_EVENT_OUT", "artifacts/gen1_gold_event_driven_v1"))

SETUP_POLICY = {
    "trend_pullback": {
        "max_hold_minutes": 180,
        "cooldown_minutes": 180,
        "atr_source": "5m",
        "atr_multiple": 1.00,
        "risk_floor_pct": 0.0007,
        "risk_cap_pct": 0.0030,
        "reward_risk": 1.60,
    },
    "breakout": {
        "max_hold_minutes": 240,
        "cooldown_minutes": 240,
        "atr_source": "5m",
        "atr_multiple": 1.25,
        "risk_floor_pct": 0.0008,
        "risk_cap_pct": 0.0035,
        "reward_risk": 1.80,
    },
    "sweep_reversal": {
        "max_hold_minutes": 240,
        "cooldown_minutes": 240,
        "atr_source": "15m",
        "atr_multiple": 1.10,
        "risk_floor_pct": 0.0010,
        "risk_cap_pct": 0.0045,
        "reward_risk": 2.00,
    },
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _same_side(label: Any, side: int) -> bool:
    return str(label or "") == ("bullish" if side > 0 else "bearish")


def _breakout_side(label: Any, side: int) -> bool:
    return str(label or "") == ("up" if side > 0 else "down")


def classify_setup(features: dict[str, Any], candidate: str) -> str | None:
    side = 1 if candidate == "long_setup" else -1
    desired = "LONG" if side > 0 else "SHORT"
    if str(features.get("gen1_decision") or "WAIT") != desired:
        return None

    cognition = _float(features.get("cognitive_confidence"))
    agreement = _float(features.get("evidence_agreement"))
    conflict = _float(features.get("evidence_conflict_score"), 1.0)
    alignment = _float(features.get("htf_alignment_score")) * side
    rsi5 = _float(features.get("5m_rsi14"), 50.0)
    gap5 = _float(features.get("5m_ema_gap_atr")) * side
    price_fast = _float(features.get("5m_price_fast_atr"), 99.0)

    expected_sweep = "sell_side_sweep" if side > 0 else "buy_side_sweep"
    expected_zone = "discount" if side > 0 else "premium"
    structural_turn = (
        _same_side(features.get("displacement"), side)
        or _same_side(features.get("break_of_structure"), side)
    )
    if (
        cognition >= 0.58
        and agreement >= 0.50
        and conflict <= 0.60
        and str(features.get("liquidity_sweep") or "none") == expected_sweep
        and str(features.get("dealing_zone") or "") == expected_zone
        and structural_turn
        and alignment >= -0.05
    ):
        return "sweep_reversal"

    quality = cognition >= 0.58 and agreement >= 0.55 and conflict <= 0.55
    if not quality:
        return None

    rsi_breakout_ok = 55.0 <= rsi5 <= 75.0 if side > 0 else 25.0 <= rsi5 <= 45.0
    if (
        _same_side(features.get("15m_direction"), side)
        and alignment >= 0.15
        and (
            _breakout_side(features.get("5m_breakout"), side)
            or _breakout_side(features.get("1m_breakout"), side)
        )
        and rsi_breakout_ok
        and gap5 >= 0.10
    ):
        return "breakout"

    rsi_pullback_ok = 48.0 <= rsi5 <= 62.0 if side > 0 else 38.0 <= rsi5 <= 52.0
    if (
        _same_side(features.get("5m_direction"), side)
        and _same_side(features.get("15m_direction"), side)
        and alignment >= 0.15
        and abs(price_fast) <= 0.45
        and rsi_pullback_ok
    ):
        return "trend_pullback"

    return None


def _risk_plan(
    setup_type: str,
    features: dict[str, Any],
    entry_mid: float,
    side: int,
) -> dict[str, float | int]:
    policy = SETUP_POLICY[setup_type]
    tf = str(policy["atr_source"])
    atr = max(0.0, _float(features.get(f"{tf}_atr14")))
    floor = entry_mid * float(policy["risk_floor_pct"])
    cap = entry_mid * float(policy["risk_cap_pct"])
    stop_distance = min(cap, max(floor, atr * float(policy["atr_multiple"])))

    if side > 0:
        swing = _float(features.get(f"{tf}_recent_swing_low"))
        swing_distance = entry_mid - swing if swing > 0 else 0.0
    else:
        swing = _float(features.get(f"{tf}_recent_swing_high"))
        swing_distance = swing - entry_mid if swing > 0 else 0.0
    structural = swing_distance + 0.10 * atr if swing_distance > 0 else 0.0
    if 0.0 < structural <= cap:
        stop_distance = max(stop_distance, structural)

    return {
        "stop_distance": stop_distance,
        "target_distance": stop_distance * float(policy["reward_risk"]),
        "max_hold_minutes": int(policy["max_hold_minutes"]),
        "cooldown_minutes": int(policy["cooldown_minutes"]),
    }


def _quote_at_or_before(
    quotes: dict[datetime, tuple[float, float]],
    quote_times: list[datetime],
    at: datetime,
    max_age_minutes: int = 2,
) -> tuple[float, float] | None:
    direct = quotes.get(at)
    if direct is not None:
        return direct
    index = bisect_right(quote_times, at) - 1
    if index < 0:
        return None
    ts = quote_times[index]
    if at - ts > timedelta(minutes=max_age_minutes):
        return None
    return quotes.get(ts)


def simulate_trade(
    *,
    m1: list[XAUBar],
    m1_available: list[datetime],
    quotes: dict[datetime, tuple[float, float]],
    quote_times: list[datetime],
    entry_time: datetime,
    candidate: str,
    setup_type: str,
    features: dict[str, Any],
) -> dict[str, Any] | None:
    side = 1 if candidate == "long_setup" else -1
    entry_quote = _quote_at_or_before(quotes, quote_times, entry_time)
    if entry_quote is None:
        return None
    entry_bid, entry_ask = entry_quote
    entry_mid = (entry_bid + entry_ask) / 2.0
    entry_exec = entry_ask if side > 0 else entry_bid
    plan = _risk_plan(setup_type, features, entry_mid, side)
    stop_distance = float(plan["stop_distance"])
    target_distance = float(plan["target_distance"])
    stop_mid = entry_mid - side * stop_distance
    target_mid = entry_mid + side * target_distance
    expiry = entry_time + timedelta(minutes=int(plan["max_hold_minutes"]))

    start = bisect_right(m1_available, entry_time)
    end = bisect_right(m1_available, expiry)
    future = m1[start:end]
    if not future:
        return None

    exit_time: datetime | None = None
    exit_reason = "time_expiry"
    exit_mid = float(future[-1].close)
    ambiguous = False

    for bar in future:
        if side > 0:
            target_hit = float(bar.high) >= target_mid
            stop_hit = float(bar.low) <= stop_mid
        else:
            target_hit = float(bar.low) <= target_mid
            stop_hit = float(bar.high) >= stop_mid
        available_at = bar.timestamp + timedelta(minutes=1)
        if target_hit and stop_hit:
            exit_reason = "ambiguous_same_bar_stop"
            exit_mid = stop_mid
            exit_time = available_at
            ambiguous = True
            break
        if stop_hit:
            exit_reason = "stop"
            exit_mid = stop_mid
            exit_time = available_at
            break
        if target_hit:
            exit_reason = "target"
            exit_mid = target_mid
            exit_time = available_at
            break

    if exit_time is None:
        exit_time = min(expiry, future[-1].timestamp + timedelta(minutes=1))

    exit_quote = _quote_at_or_before(quotes, quote_times, exit_time)
    if exit_quote is None:
        return None
    exit_bid, exit_ask = exit_quote
    half_spread = max(0.0, exit_ask - exit_bid) / 2.0
    if exit_reason == "time_expiry":
        exit_exec = exit_bid if side > 0 else exit_ask
    elif side > 0:
        exit_exec = exit_mid - half_spread
    else:
        exit_exec = exit_mid + half_spread

    net_bps = side * ((exit_exec - entry_exec) / entry_exec) * 10000.0
    duration = max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
    return {
        "entry_time": entry_time,
        "exit_time": exit_time,
        "setup_type": setup_type,
        "direction": "LONG" if side > 0 else "SHORT",
        "candidate": candidate,
        "entry_mid": entry_mid,
        "entry_exec": entry_exec,
        "exit_mid": exit_mid,
        "exit_exec": exit_exec,
        "stop_mid": stop_mid,
        "target_mid": target_mid,
        "stop_distance_usd": stop_distance,
        "target_distance_usd": target_distance,
        "planned_rr": target_distance / stop_distance if stop_distance else 0.0,
        "duration_minutes": duration,
        "exit_reason": exit_reason,
        "ambiguous_same_bar": ambiguous,
        "net_bps": net_bps,
        "won": net_bps > 0,
        "cooldown_minutes": int(plan["cooldown_minutes"]),
    }


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trade_count": 0}
    pnl = [float(t["net_bps"]) for t in trades]
    wins = [v for v in pnl if v > 0]
    losses = [v for v in pnl if v < 0]
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for value in pnl:
        cumulative += value
        peak = max(peak, cumulative)
        max_dd = min(max_dd, cumulative - peak)
    gross_loss = abs(sum(losses))
    return {
        "trade_count": len(trades),
        "win_rate": sum(1 for v in pnl if v > 0) / len(pnl),
        "mean_net_bps": fmean(pnl),
        "median_net_bps": median(pnl),
        "cumulative_net_bps": sum(pnl),
        "profit_factor": sum(wins) / gross_loss if gross_loss > 0 else None,
        "max_drawdown_bps": max_dd,
        "median_duration_minutes": median(float(t["duration_minutes"]) for t in trades),
        "target_rate": sum(1 for t in trades if t["exit_reason"] == "target") / len(trades),
        "stop_rate": sum(
            1 for t in trades if t["exit_reason"] in {"stop", "ambiguous_same_bar_stop"}
        ) / len(trades),
        "time_expiry_rate": sum(1 for t in trades if t["exit_reason"] == "time_expiry") / len(trades),
    }


def _subset(
    trades: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> list[dict[str, Any]]:
    return [t for t in trades if start <= t["entry_time"] < end]


def _write_trades(path: Path, trades: list[dict[str, Any]]) -> None:
    if not trades:
        path.write_text("", encoding="utf-8")
        return
    fields = list(trades[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for trade in trades:
            row = dict(trade)
            for key in ("entry_time", "exit_time"):
                if isinstance(row.get(key), datetime):
                    row[key] = row[key].isoformat()
            writer.writerow(row)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    m1, quotes, dataset = legacy.download_dataset(START, END)
    bars_by_tf = {
        XAUTimeframe.M1: m1,
        XAUTimeframe.M5: legacy.resample(m1, XAUTimeframe.M5),
        XAUTimeframe.M15: legacy.resample(m1, XAUTimeframe.M15),
        XAUTimeframe.H1: legacy.resample(m1, XAUTimeframe.H1),
        XAUTimeframe.H4: legacy.resample(m1, XAUTimeframe.H4),
        XAUTimeframe.D1: legacy.resample(m1, XAUTimeframe.D1),
    }
    available = {
        tf: legacy.availability(rows, tf)
        for tf, rows in bars_by_tf.items()
    }
    m1_available = available[XAUTimeframe.M1]
    quote_times = sorted(quotes)
    macro_provider, macro_diag = legacy.build_historical_macro_proxy(START, END)
    prefilter = XAUIntradayEngine(require_execution_data=False)
    context_cache: dict[
        tuple[datetime | None, datetime | None, datetime | None],
        dict[str, Any],
    ] = {}

    def cached_context(
        hourly: list[XAUBar],
        h4: list[XAUBar],
        daily: list[XAUBar],
    ) -> dict[str, Any]:
        key = (
            hourly[-1].timestamp if hourly else None,
            h4[-1].timestamp if h4 else None,
            daily[-1].timestamp if daily else None,
        )
        if key not in context_cache:
            context_cache[key] = build_market_context(hourly, h4, daily)
        return context_cache[key]

    trades: list[dict[str, Any]] = []
    diagnostics: dict[str, int] = defaultdict(int)
    last_eval: datetime | None = None
    busy_until = START
    last_entry_by_key: dict[tuple[str, str], datetime] = {}

    for evaluation_time in m1_available:
        if evaluation_time < START or evaluation_time >= END:
            continue
        if (
            last_eval is not None
            and evaluation_time - last_eval < timedelta(minutes=STEP_MINUTES)
        ):
            continue
        last_eval = evaluation_time
        diagnostics["market_scans"] += 1
        if evaluation_time < busy_until:
            diagnostics["scans_while_position_open"] += 1
            continue

        intraday = {
            XAUTimeframe.M1: legacy.bars_window(
                m1, m1_available, evaluation_time, 300
            ),
            XAUTimeframe.M5: legacy.bars_window(
                bars_by_tf[XAUTimeframe.M5],
                available[XAUTimeframe.M5],
                evaluation_time,
                300,
            ),
            XAUTimeframe.M15: legacy.bars_window(
                bars_by_tf[XAUTimeframe.M15],
                available[XAUTimeframe.M15],
                evaluation_time,
                300,
            ),
        }
        pre = prefilter.analyze(
            intraday,
            event_risk=False,
            macro_bias=0,
            now=evaluation_time,
            assume_sorted=True,
        )
        if pre.blocked or pre.candidate not in {"long_setup", "short_setup"}:
            continue
        diagnostics["intraday_candidates"] += 1

        window = {
            **intraday,
            XAUTimeframe.H1: legacy.bars_window(
                bars_by_tf[XAUTimeframe.H1],
                available[XAUTimeframe.H1],
                evaluation_time,
                1100,
            ),
            XAUTimeframe.H4: legacy.bars_window(
                bars_by_tf[XAUTimeframe.H4],
                available[XAUTimeframe.H4],
                evaluation_time,
                1100,
            ),
            XAUTimeframe.D1: legacy.bars_window(
                bars_by_tf[XAUTimeframe.D1],
                available[XAUTimeframe.D1],
                evaluation_time,
                1100,
            ),
        }
        macro = macro_provider(evaluation_time)
        technical = legacy.build_replay_technical_state(
            window,
            evaluation_time,
            macro_bias=int(macro.get("bias", 0) or 0),
            market_context_builder=cached_context,
            intraday_assessment=pre,
        )
        candidate = str(technical.get("candidate") or "none")
        if technical.get("blocked") or candidate not in {"long_setup", "short_setup"}:
            continue

        cognition = build_cognitive_state(
            technical, macro, memory=None, min_confidence=0.58
        )
        fusion = build_decision_fusion(
            technical,
            macro,
            memory=None,
            min_confidence=0.58,
            as_of=evaluation_time,
        )
        evidence = build_gen1_evidence_fusion(
            technical,
            macro,
            fusion,
            memory=None,
            require_xaut=False,
        )
        features = legacy.extract_research_features(
            technical,
            cognition,
            fusion,
            evidence,
            macro,
            evaluation_time,
            candidate,
        )
        setup_type = classify_setup(features, candidate)
        if setup_type is None:
            continue
        diagnostics[f"classified_{setup_type}"] += 1

        direction = "LONG" if candidate == "long_setup" else "SHORT"
        key = (setup_type, direction)
        previous = last_entry_by_key.get(key)
        cooldown = timedelta(
            minutes=int(SETUP_POLICY[setup_type]["cooldown_minutes"])
        )
        if previous is not None and evaluation_time - previous < cooldown:
            diagnostics["cooldown_rejections"] += 1
            continue

        trade = simulate_trade(
            m1=m1,
            m1_available=m1_available,
            quotes=quotes,
            quote_times=quote_times,
            entry_time=evaluation_time,
            candidate=candidate,
            setup_type=setup_type,
            features=features,
        )
        if trade is None:
            diagnostics["execution_data_rejections"] += 1
            continue

        trade.update({
            "session": features.get("session"),
            "regime": features.get("regime"),
            "gen1_decision": features.get("gen1_decision"),
            "gen1_confidence": features.get("gen1_confidence"),
            "cognitive_confidence": features.get("cognitive_confidence"),
            "evidence_agreement": features.get("evidence_agreement"),
            "evidence_conflict_score": features.get("evidence_conflict_score"),
            "htf_alignment_score": features.get("htf_alignment_score"),
            "htf_composite_direction": features.get("htf_composite_direction"),
            "macro_bias_label": features.get("macro_bias_label"),
        })
        trades.append(trade)
        diagnostics["executed_trades"] += 1
        last_entry_by_key[key] = evaluation_time
        busy_until = trade["exit_time"]

        if len(trades) % 100 == 0:
            print(json.dumps({
                "phase": "event_backtest_progress",
                "trades": len(trades),
                "market_scans": diagnostics["market_scans"],
                "intraday_candidates": diagnostics["intraday_candidates"],
                "evaluation_time": evaluation_time.isoformat(),
                "elapsed_seconds": round(time.monotonic() - started, 2),
            }), flush=True)

    reference = _subset(trades, START, SPLIT)
    holdout = _subset(trades, SPLIT, END)
    by_setup = {
        name: {
            "overall": _metrics(
                [t for t in trades if t["setup_type"] == name]
            ),
            "reference": _metrics(
                [t for t in reference if t["setup_type"] == name]
            ),
            "holdout": _metrics(
                [t for t in holdout if t["setup_type"] == name]
            ),
        }
        for name in SETUP_POLICY
    }
    by_direction = {
        direction: {
            "overall": _metrics(
                [t for t in trades if t["direction"] == direction]
            ),
            "holdout": _metrics(
                [t for t in holdout if t["direction"] == direction]
            ),
        }
        for direction in ("LONG", "SHORT")
    }

    report = {
        "benchmark": "gen1-gold-event-driven-v1",
        "strategy_revision_env": os.getenv("STRATEGY_REVISION")
        or os.getenv("GITHUB_SHA")
        or "unknown",
        "dataset": dataset,
        "period": {
            "start": START.isoformat(),
            "split": SPLIT.isoformat(),
            "end": END.isoformat(),
        },
        "architecture": {
            "mode": "event_driven_single_position",
            "scan_step_minutes": STEP_MINUTES,
            "pipeline": [
                "market_scan",
                "intraday_candidate_prefilter",
                "full_gen1_state",
                "setup_classification",
                "quality_gates",
                "entry",
                "strategy_specific_stop_target",
                "minute_by_minute_trade_management",
                "target_stop_or_time_expiry",
            ],
            "setup_types": SETUP_POLICY,
            "wait_policy": "no complete setup means no trade",
            "overlap_policy": "one position at a time",
            "same_bar_tp_sl_policy": "conservative_stop",
            "lookahead_in_decision": False,
            "outcome_tuning_used": False,
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "overall": _metrics(trades),
        "reference_year": _metrics(reference),
        "holdout_year": _metrics(holdout),
        "by_setup": by_setup,
        "by_direction": by_direction,
        "macro_proxy": macro_diag,
        "data_integrity_passed": bool(
            dataset.get("m1_bar_count", 0) > 400_000
        ),
        "edge_proven": False,
        "promotion_status": "research_only",
        "limitations": [
            "M1 OHLC cannot reveal target/stop ordering when both are crossed inside the same minute; those bars are scored conservatively as stops.",
            "Historical macro uses the same one-day-lagged public proxy as the legacy two-year benchmark.",
            "Historical live news/calendar synthesis and XAUT raw microstructure are not reconstructed.",
            "Strategy-specific thresholds are semantic fixed rules for architecture validation and were not optimized on trade outcomes.",
        ],
    }

    (OUT / "event_driven_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_trades(OUT / "trades.csv", trades)
    (OUT / "SUMMARY.md").write_text(
        "\n".join([
            "# GEN1 Gold — Event-Driven Two-Year Backtest",
            "",
            f"- Trades: {report['overall'].get('trade_count', 0)}",
            f"- Holdout trades: {report['holdout_year'].get('trade_count', 0)}",
            f"- Holdout mean net bps: {report['holdout_year'].get('mean_net_bps')}",
            f"- Holdout win rate: {report['holdout_year'].get('win_rate')}",
            f"- Holdout profit factor: {report['holdout_year'].get('profit_factor')}",
            f"- Holdout max drawdown bps: {report['holdout_year'].get('max_drawdown_bps')}",
            "",
            "Research-only. edge_proven remains false until statistical and forward validation are completed.",
        ]),
        encoding="utf-8",
    )

    print(json.dumps({
        "benchmark": report["benchmark"],
        "data_integrity_passed": report["data_integrity_passed"],
        "diagnostics": report["diagnostics"],
        "overall": report["overall"],
        "holdout_year": report["holdout_year"],
        "by_setup": report["by_setup"],
        "by_direction": report["by_direction"],
        "artifact_dir": str(OUT),
        "edge_proven": False,
        "promotion_status": "research_only",
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
