"""Two-year real-data GEN1 Gold walk-forward benchmark.

Evaluation window:
    2024-09-23 00:00 UTC -> 2026-09-23 00:00 UTC

Gold data is real Dukascopy XAUUSD tick-derived MID M1 OHLCV (quoted-volume
activity proxy).  M5/M15/H1/H4/D1 are resampled from that same provenance
family to avoid cross-instrument leakage.

Historical macro is a point-in-time, one-day-lagged public-market proxy built
from DXY, US 10Y, VIX, S&P 500 and crude.  It is NOT a replay of the live
Ahmed Toolbox news/calendar synthesis.

Historical raw XAUT order book / executed-trade footprint is unavailable and is
therefore explicitly excluded rather than fabricated.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_right
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from src.modules.xau.replay import walk_forward_replay
from src.modules.xau.validation import evaluate_gen1_replay
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe

UTC = timezone.utc
SOURCE = "dukascopy:datafeed:XAUUSD:mid"
MACRO_SYMBOLS = {
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "vix": "^VIX",
    "spx": "^GSPC",
    "oil": "CL=F",
}


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_m1(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    frames = []
    for path in sorted(input_dir.glob("xauusd_m1_*.csv.gz")):
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise RuntimeError("no Dukascopy monthly M1 artifacts found")
    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
    data = data.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")
    data = data.set_index("timestamp")
    for col in ("open", "high", "low", "close", "volume"):
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["open", "high", "low", "close"])
    diagnostics = []
    for path in sorted(input_dir.glob("diagnostics_*.json")):
        diagnostics.append(json.loads(path.read_text(encoding="utf-8")))
    return data, diagnostics


def resample_ohlcv(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    out = m1.resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    })
    return out.dropna(subset=["open", "high", "low", "close"])


def to_bars(frame: pd.DataFrame, timeframe: XAUTimeframe, suffix: str) -> list[XAUBar]:
    rows = []
    for ts, row in frame.iterrows():
        stamp = ts.to_pydatetime()
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=UTC)
        rows.append(XAUBar(
            timestamp=stamp,
            timeframe=timeframe,
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row["volume"]) if pd.notna(row["volume"]) else None,
            source=f"{SOURCE}:{suffix}",
            symbol="XAUUSD",
            execution_eligible=False,
        ))
    return rows


def build_history(m1: pd.DataFrame) -> dict[XAUTimeframe, list[XAUBar]]:
    return {
        XAUTimeframe.M1: to_bars(m1, XAUTimeframe.M1, "1m"),
        XAUTimeframe.M5: to_bars(resample_ohlcv(m1, "5min"), XAUTimeframe.M5, "resampled-5m"),
        XAUTimeframe.M15: to_bars(resample_ohlcv(m1, "15min"), XAUTimeframe.M15, "resampled-15m"),
        XAUTimeframe.H1: to_bars(resample_ohlcv(m1, "1h"), XAUTimeframe.H1, "resampled-1h"),
        XAUTimeframe.H4: to_bars(resample_ohlcv(m1, "4h"), XAUTimeframe.H4, "resampled-4h"),
        XAUTimeframe.D1: to_bars(resample_ohlcv(m1, "1D"), XAUTimeframe.D1, "resampled-1d"),
    }


def _close_series(symbol: str, start: datetime, end: datetime) -> pd.Series:
    raw = yf.download(
        symbol,
        start=(start - timedelta(days=20)).date().isoformat(),
        end=(end + timedelta(days=2)).date().isoformat(),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float, name=symbol)
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()
    close.index = pd.to_datetime(close.index, utc=True)
    return close.rename(symbol)


def build_macro_proxy(start: datetime, end: datetime):
    series = {name: _close_series(symbol, start, end) for name, symbol in MACRO_SYMBOLS.items()}
    usable = [value for value in series.values() if not value.empty]
    if len(usable) < 3:
        raise RuntimeError("historical macro proxy has fewer than three usable market series")
    frame = pd.concat(series.values(), axis=1).sort_index().ffill()
    frame.columns = list(series.keys())
    frame = frame.dropna(how="all")
    # A daily close becomes usable only the next UTC day: conservative anti-lookahead.
    available = [ts.to_pydatetime() + timedelta(days=1) for ts in frame.index]
    if available and available[0].tzinfo is None:
        available = [item.replace(tzinfo=UTC) for item in available]

    def provider(at: datetime) -> dict[str, Any]:
        at = at.astimezone(UTC)
        idx = bisect_right(available, at) - 1
        if idx < 5:
            return {
                "bias": 0,
                "confidence": 0.25,
                "event_risk": False,
                "observed_at": at.isoformat(),
                "calendar_ok": True,
                "search_ok": True,
                "synthesis_ok": True,
                "historical_macro_proxy": True,
                "proxy_status": "warmup",
            }
        now = frame.iloc[idx]
        prev = frame.iloc[idx - 5]
        score = 0.0
        drivers = []

        def pct(name: str) -> float | None:
            a, b = now.get(name), prev.get(name)
            if pd.isna(a) or pd.isna(b) or not b:
                return None
            return (float(a) - float(b)) / float(b)

        dxy = pct("dxy")
        if dxy is not None:
            component = max(-1.0, min(1.0, -dxy / 0.012))
            score += 0.35 * component
            drivers.append({"name": "dxy_5d", "value": round(dxy, 6), "gold_score": round(component, 4)})

        if pd.notna(now.get("us10y")) and pd.notna(prev.get("us10y")):
            delta = float(now["us10y"]) - float(prev["us10y"])
            component = max(-1.0, min(1.0, -delta / 0.35))
            score += 0.30 * component
            drivers.append({"name": "us10y_5d_delta", "value": round(delta, 4), "gold_score": round(component, 4)})

        vix = pct("vix")
        if vix is not None:
            component = max(-1.0, min(1.0, vix / 0.25))
            score += 0.15 * component
            drivers.append({"name": "vix_5d", "value": round(vix, 6), "gold_score": round(component, 4)})

        spx = pct("spx")
        if spx is not None:
            component = max(-1.0, min(1.0, -spx / 0.04))
            score += 0.10 * component
            drivers.append({"name": "spx_5d", "value": round(spx, 6), "gold_score": round(component, 4)})

        oil = pct("oil")
        if oil is not None:
            component = max(-1.0, min(1.0, oil / 0.08))
            score += 0.10 * component
            drivers.append({"name": "oil_5d", "value": round(oil, 6), "gold_score": round(component, 4)})

        score = max(-1.0, min(1.0, score))
        bias = 1 if score >= 0.15 else -1 if score <= -0.15 else 0
        confidence = min(0.85, 0.35 + abs(score) * 0.55)
        observed = available[idx]
        return {
            "bias": bias,
            "confidence": round(confidence, 4),
            "event_risk": False,
            "observed_at": observed.isoformat(),
            "calendar_ok": True,
            "search_ok": True,
            "synthesis_ok": True,
            "cache_stale": False,
            "refresh_pending": False,
            "historical_macro_proxy": True,
            "proxy_score": round(score, 4),
            "drivers": drivers,
            "limitations": [
                "One-day-lagged market proxy, not replay of live Ahmed Toolbox news/calendar synthesis.",
                "No historical event-risk calendar gate is claimed.",
            ],
        }

    return provider, {
        "symbols": MACRO_SYMBOLS,
        "rows": int(len(frame)),
        "start": frame.index.min().isoformat() if len(frame) else None,
        "end": frame.index.max().isoformat() if len(frame) else None,
        "lookahead_policy": "daily_close_available_next_utc_day",
    }


def session_name(at: datetime) -> str:
    hour = at.astimezone(UTC).hour
    if hour < 8:
        return "asia"
    if hour < 13:
        return "london"
    if hour < 21:
        return "new_york"
    return "off_hours"


def final_directional_bps(episode) -> float | None:
    decision = str(episode.meta.get("gen1_decision") or "WAIT").upper()
    if decision not in {"LONG", "SHORT"}:
        return None
    raw = ((episode.outcome_price - episode.entry_price) / episode.entry_price) * 10_000.0
    return raw if decision == "LONG" else -raw


def independent_directionals(episodes, separation_minutes: int = 60):
    selected = []
    last = None
    for episode in sorted(episodes, key=lambda item: item.observed_at):
        value = final_directional_bps(episode)
        if value is None:
            continue
        if last is None or episode.observed_at - last >= timedelta(minutes=separation_minutes):
            selected.append((episode, value))
            last = episode.observed_at
    return selected


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    centre = p + z2 / (2 * total)
    margin = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total))
    return max(0.0, (centre - margin) / denom)


def summarize_independent(selected):
    values = [value for _, value in selected]
    positive = sum(value > 0 for value in values)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    equity_points = []
    monthly = {}
    sessions = {}
    regimes = {}
    for episode, value in selected:
        equity *= 1.0 + value / 10_000.0
        peak = max(peak, equity)
        max_dd = min(max_dd, equity / peak - 1.0)
        equity_points.append((episode.observed_at, equity))
        month = episode.observed_at.strftime("%Y-%m")
        bucket = monthly.setdefault(month, [])
        bucket.append(value)
        sess = session_name(episode.observed_at)
        sessions.setdefault(sess, []).append(value)
        regimes.setdefault(episode.regime or "unknown", []).append(value)

    def group_payload(groups):
        return {
            key: {
                "n": len(vals),
                "positive_rate": round(sum(v > 0 for v in vals) / len(vals), 4) if vals else None,
                "average_bps": round(sum(vals) / len(vals), 4) if vals else None,
            }
            for key, vals in sorted(groups.items())
        }

    return {
        "count": len(values),
        "positive_rate": round(positive / len(values), 4) if values else None,
        "wilson_95_lower": round(wilson_lower(positive, len(values)), 4) if values else None,
        "average_bps": round(sum(values) / len(values), 4) if values else None,
        "median_bps": round(median(values), 4) if values else None,
        "costless_1x_compounded_return_pct": round((equity - 1.0) * 100.0, 4),
        "costless_1x_max_drawdown_pct": round(max_dd * 100.0, 4),
        "sessions": group_payload(sessions),
        "regimes": group_payload(regimes),
        "months": group_payload(monthly),
    }, equity_points


def make_charts(output_dir: Path, m1: pd.DataFrame, equity_points, validation: dict[str, Any]) -> None:
    daily = resample_ohlcv(m1, "1D")
    fig = plt.figure(figsize=(14, 6))
    plt.plot(daily.index, daily["close"])
    plt.title("XAUUSD — Dukascopy MID daily close (two-year evaluation window)")
    plt.xlabel("Date")
    plt.ylabel("USD/oz")
    plt.tight_layout()
    fig.savefig(output_dir / "xauusd_two_year_close.png", dpi=150)
    plt.close(fig)

    if equity_points:
        fig = plt.figure(figsize=(14, 6))
        plt.plot([x for x, _ in equity_points], [100.0 * y for _, y in equity_points])
        plt.title("GEN1 independent 60m research equity — costless 1x reference")
        plt.xlabel("Date")
        plt.ylabel("Reference equity (start=100)")
        plt.tight_layout()
        fig.savefig(output_dir / "gen1_independent_equity.png", dpi=150)
        plt.close(fig)

    labels, rates = [], []
    for name in ("pm10", "pm20", "pm30"):
        row = (validation.get("range_outcomes") or {}).get(name) or {}
        labels.append(name.replace("pm", "±$"))
        value = row.get("favorable_first_rate")
        rates.append(float(value) * 100.0 if value is not None else 0.0)
    fig = plt.figure(figsize=(8, 5))
    plt.bar(labels, rates)
    plt.ylim(0, 100)
    plt.ylabel("Favorable first-touch rate (%)")
    plt.title("GEN1 directional decisions — ±10/20/30 first-touch")
    plt.tight_layout()
    fig.savefig(output_dir / "gen1_range_first_touch.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--step-minutes", type=int, default=15)
    parser.add_argument("--horizon-minutes", type=int, default=60)
    args = parser.parse_args()

    evaluation_start = parse_iso(args.start)
    evaluation_end = parse_iso(args.end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    m1, acquisition = load_m1(Path(args.input_dir))
    m1 = m1[(m1.index >= pd.Timestamp(evaluation_start)) & (m1.index < pd.Timestamp(evaluation_end))]
    if m1.empty:
        raise RuntimeError("no M1 bars inside requested evaluation window")

    history = build_history(m1)
    macro_provider, macro_diag = build_macro_proxy(evaluation_start, evaluation_end)
    episodes = walk_forward_replay(
        history,
        horizon_minutes=args.horizon_minutes,
        step_minutes=args.step_minutes,
        macro_provider=macro_provider,
        source="dukascopy:XAUUSD:two-year-real-v1",
        min_confidence=0.58,
    )
    validation = evaluate_gen1_replay(episodes)
    independent = independent_directionals(episodes, separation_minutes=args.horizon_minutes)
    independent_summary, equity_points = summarize_independent(independent)

    with (output_dir / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "observed_at", "candidate", "gen1_decision", "gen1_confidence", "regime",
            "entry_price", "outcome_price", "final_directional_bps", "session",
            "pm10_first", "pm20_first", "pm30_first",
        ])
        for episode in episodes:
            levels = (episode.meta.get("range_outcomes") or {}).get("levels") or {}
            writer.writerow([
                episode.observed_at.isoformat(),
                episode.candidate,
                episode.meta.get("gen1_decision"),
                episode.meta.get("gen1_decision_confidence"),
                episode.regime,
                episode.entry_price,
                episode.outcome_price,
                final_directional_bps(episode),
                session_name(episode.observed_at),
                (levels.get("pm10") or {}).get("first_hit"),
                (levels.get("pm20") or {}).get("first_hit"),
                (levels.get("pm30") or {}).get("first_hit"),
            ])

    acquisition_failed = sum(int(row.get("failed_hours") or 0) for row in acquisition)
    acquisition_attempted = sum(int(row.get("attempted_hours") or 0) for row in acquisition)
    acquisition_data_hours = sum(int(row.get("data_hours") or 0) for row in acquisition)

    result = {
        "benchmark": "xau-two-year-real-gen1-v1",
        "evaluation_window": {
            "start": evaluation_start.isoformat(),
            "end": evaluation_end.isoformat(),
            "step_minutes": args.step_minutes,
            "outcome_horizon_minutes": args.horizon_minutes,
        },
        "dataset": {
            "provider": SOURCE,
            "symbol": "XAUUSD",
            "price_basis": "mid",
            "m1_bars": int(len(m1)),
            "first_m1": m1.index.min().isoformat(),
            "last_m1": m1.index.max().isoformat(),
            "attempted_hours": acquisition_attempted,
            "data_hours": acquisition_data_hours,
            "failed_hours": acquisition_failed,
            "volume_semantics": "quoted_bid_plus_ask_activity_proxy",
            "centralized_traded_volume": False,
            "execution_eligible": False,
            "same_source_resampling": ["5m", "15m", "1h", "4h", "1d"],
        },
        "historical_macro_proxy": macro_diag,
        "raw_walk_forward": validation,
        "independent_60m_directionals": independent_summary,
        "historical_scope": {
            "real_xauusd_ohlc": True,
            "real_dukascopy_tick_activity": True,
            "htf_structure": True,
            "ema_ladder_when_history_sufficient": True,
            "volume_profile_activity_proxy": True,
            "liquidity_and_smc_from_ohlc": True,
            "point_in_time_macro_market_proxy": True,
            "historical_xaut_raw_book": False,
            "historical_xaut_footprint": False,
            "historical_live_news_synthesis": False,
            "broker_fill_model": False,
        },
        "limitations": [
            "Historical XAUT raw-book/footprint/CVD is unavailable and is not fabricated.",
            "Macro uses a one-day-lagged market proxy, not the exact live Ahmed Toolbox news/calendar pipeline.",
            "Dukascopy quoted bid+ask size is an activity/liquidity proxy, not centralized global gold traded volume.",
            "Reference returns are costless MID-price research outcomes, not broker fills; spread/slippage/commission are excluded.",
            "Early observations have progressively maturing long-horizon EMA context because the evaluation dataset starts exactly at the requested two-year boundary.",
            "Passing statistical thresholds would be evidence under this replay protocol, not proof of a persistent executable edge.",
        ],
        "edge_proven": False,
    }

    make_charts(output_dir, m1, equity_points, validation)
    (output_dir / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))

    if acquisition_failed:
        raise SystemExit("dataset acquisition contains failed hourly requests")
    if not episodes:
        raise SystemExit("walk-forward produced zero candidate episodes")


if __name__ == "__main__":
    main()
