"""GEN1 Gold + KVO Gold Pulse v1 two-year event-driven backtest.

Same frozen dataset, execution model, spread model, stop/target simulation,
single-position rule and period split as gen1_gold_event_driven_v1.
The changed variable is the KVO confirmation layer.
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from benchmarks.gen1_gold_2y_real_v1 import run as legacy
from benchmarks.gen1_gold_event_driven_v1 import run as base
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.strategy.xau_kvo import assess_kvo_gold
from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion
from src.modules.xau.market_context import build_market_context
from src.modules.xau.service import build_decision_fusion
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe

START, SPLIT, END = base.START, base.SPLIT, base.END
STEP_MINUTES = base.STEP_MINUTES
OUT = Path(os.getenv("GEN1_KVO_OUT", "artifacts/gen1_gold_event_driven_kvo_v1"))


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
    available = {tf: legacy.availability(rows, tf) for tf, rows in bars_by_tf.items()}
    m1_available = available[XAUTimeframe.M1]
    quote_times = sorted(quotes)
    macro_provider, macro_diag = legacy.build_historical_macro_proxy(START, END)
    prefilter = XAUIntradayEngine(require_execution_data=False)
    context_cache: dict[tuple[datetime | None, datetime | None, datetime | None], dict[str, Any]] = {}
    kvo_cache: dict[tuple[datetime | None, datetime | None, str, str], Any] = {}

    def cached_context(hourly: list[XAUBar], h4: list[XAUBar], daily: list[XAUBar]) -> dict[str, Any]:
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
        if last_eval is not None and evaluation_time - last_eval < timedelta(minutes=STEP_MINUTES):
            continue
        last_eval = evaluation_time
        diagnostics["market_scans"] += 1
        if evaluation_time < busy_until:
            diagnostics["scans_while_position_open"] += 1
            continue

        intraday = {
            XAUTimeframe.M1: legacy.bars_window(m1, m1_available, evaluation_time, 300),
            XAUTimeframe.M5: legacy.bars_window(
                bars_by_tf[XAUTimeframe.M5], available[XAUTimeframe.M5], evaluation_time, 300
            ),
            XAUTimeframe.M15: legacy.bars_window(
                bars_by_tf[XAUTimeframe.M15], available[XAUTimeframe.M15], evaluation_time, 300
            ),
        }
        pre = prefilter.analyze(
            intraday, event_risk=False, macro_bias=0, now=evaluation_time, assume_sorted=True
        )
        if pre.blocked or pre.candidate not in {"long_setup", "short_setup"}:
            continue
        diagnostics["intraday_candidates"] += 1

        window = {
            **intraday,
            XAUTimeframe.H1: legacy.bars_window(
                bars_by_tf[XAUTimeframe.H1], available[XAUTimeframe.H1], evaluation_time, 1100
            ),
            XAUTimeframe.H4: legacy.bars_window(
                bars_by_tf[XAUTimeframe.H4], available[XAUTimeframe.H4], evaluation_time, 1100
            ),
            XAUTimeframe.D1: legacy.bars_window(
                bars_by_tf[XAUTimeframe.D1], available[XAUTimeframe.D1], evaluation_time, 1100
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

        cognition = build_cognitive_state(technical, macro, memory=None, min_confidence=0.58)
        fusion = build_decision_fusion(
            technical, macro, memory=None, min_confidence=0.58, as_of=evaluation_time
        )
        evidence = build_gen1_evidence_fusion(
            technical, macro, fusion, memory=None, require_xaut=False
        )
        features = legacy.extract_research_features(
            technical, cognition, fusion, evidence, macro, evaluation_time, candidate
        )
        setup_type = base.classify_setup(features, candidate)
        if setup_type is None:
            continue
        diagnostics[f"base_classified_{setup_type}"] += 1

        kvo_key = (
            window[XAUTimeframe.M15][-1].timestamp if window[XAUTimeframe.M15] else None,
            window[XAUTimeframe.H1][-1].timestamp if window[XAUTimeframe.H1] else None,
            candidate,
            setup_type,
        )
        kvo = kvo_cache.get(kvo_key)
        if kvo is None:
            kvo = assess_kvo_gold(
                window[XAUTimeframe.M15],
                window[XAUTimeframe.H1],
                candidate=candidate,
                setup_type=setup_type,
            )
            kvo_cache[kvo_key] = kvo
        if not kvo.allowed:
            diagnostics[f"kvo_reject_{kvo.reason}"] += 1
            continue
        diagnostics[f"kvo_accept_{setup_type}"] += 1

        direction = "LONG" if candidate == "long_setup" else "SHORT"
        key = (setup_type, direction)
        previous = last_entry_by_key.get(key)
        cooldown = timedelta(minutes=int(base.SETUP_POLICY[setup_type]["cooldown_minutes"]))
        if previous is not None and evaluation_time - previous < cooldown:
            diagnostics["cooldown_rejections"] += 1
            continue

        trade = base.simulate_trade(
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
            "kvo_reason": kvo.reason,
            "kvo_m15": kvo.m15.kvo,
            "kvo_m15_signal": kvo.m15.signal,
            "kvo_m15_histogram": kvo.m15.histogram,
            "kvo_m15_histogram_delta": kvo.m15.histogram_delta,
            "kvo_m15_recent_cross": kvo.m15.recent_cross,
            "kvo_h1": kvo.h1.kvo,
            "kvo_h1_signal": kvo.h1.signal,
            "kvo_h1_histogram": kvo.h1.histogram,
            "kvo_h1_histogram_delta": kvo.h1.histogram_delta,
        })
        trades.append(trade)
        diagnostics["executed_trades"] += 1
        last_entry_by_key[key] = evaluation_time
        busy_until = trade["exit_time"]

    reference = base._subset(trades, START, SPLIT)
    holdout = base._subset(trades, SPLIT, END)
    by_setup = {
        name: {
            "overall": base._metrics([t for t in trades if t["setup_type"] == name]),
            "reference": base._metrics([t for t in reference if t["setup_type"] == name]),
            "holdout": base._metrics([t for t in holdout if t["setup_type"] == name]),
        }
        for name in base.SETUP_POLICY
    }
    by_direction = {
        direction: {
            "overall": base._metrics([t for t in trades if t["direction"] == direction]),
            "holdout": base._metrics([t for t in holdout if t["direction"] == direction]),
        }
        for direction in ("LONG", "SHORT")
    }
    report = {
        "benchmark": "gen1-gold-event-driven-kvo-v1",
        "strategy_revision_env": os.getenv("STRATEGY_REVISION") or os.getenv("GITHUB_SHA") or "unknown",
        "dataset": dataset,
        "period": {"start": START.isoformat(), "split": SPLIT.isoformat(), "end": END.isoformat()},
        "comparison_baseline": "gen1-gold-event-driven-v1",
        "changed_variable": "KVO Gold Pulse v1 confirmation layer only",
        "kvo": {
            "core": {"fast": 34, "slow": 55, "signal": 13},
            "timeframes": ["15m", "1h"],
            "spot_volume_semantics": "Dukascopy tick/activity proxy; not centralized global gold volume",
            "standalone_signal": False,
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "overall": base._metrics(trades),
        "reference_year": base._metrics(reference),
        "holdout_year": base._metrics(holdout),
        "by_setup": by_setup,
        "by_direction": by_direction,
        "macro_proxy": macro_diag,
        "data_integrity_passed": bool(dataset.get("m1_bar_count", 0) > 400_000),
        "edge_proven": False,
        "promotion_status": "research_only",
        "lookahead_in_decision": False,
    }
    (OUT / "event_driven_kvo_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    base._write_trades(OUT / "trades.csv", trades)
    print(json.dumps({
        "benchmark": report["benchmark"],
        "data_integrity_passed": report["data_integrity_passed"],
        "diagnostics": report["diagnostics"],
        "overall": report["overall"],
        "reference_year": report["reference_year"],
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
