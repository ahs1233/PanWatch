"""Two-year real-data GEN1 Gold benchmark.

Data
----
Dukascopy public XAUUSD native M1 BID/ASK candle files.
The benchmark downloads 2024-09-23 <= t < 2026-09-23, combines BID/ASK
candles into midpoint research bars, preserves the real bid/ask close spread
for a spread-cost sensitivity check, and resamples all higher timeframes from
the same M1 source.

Methodology
-----------
- Fixed strategy revision under test; no parameter tuning inside the benchmark.
- Walk-forward only: every decision sees closed bars only.
- Evaluation cadence: 5 minutes.
- Headline statistical metrics use horizon-spaced observations (>= 60m or
  >= 240m) so overlapping outcomes do not masquerade as independent evidence.
- The first year is a reference window; the second year is chronological holdout.
- Historical external macro/news and historical raw XAUT order book/tape are
  not reconstructed. Those gaps are explicit in every episode.
- Workflow success means the dataset/backtest completed with acceptable data
  integrity. It does NOT mean the strategy has an edge.
"""

from __future__ import annotations

import csv
import glob
import gzip
import hashlib
import json
import lzma
import math
import os
import pickle
import random
import struct
import sys
import time
from bisect import bisect_left, bisect_right
from urllib.parse import quote
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

import httpx

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion
from src.modules.xau.market_context import build_market_context
from src.modules.xau.replay import ReplayEpisode, build_replay_technical_state
from src.modules.xau.service import build_decision_fusion
from src.modules.xau.validation import evaluate_gen1_replay
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


UTC = timezone.utc
RECORD = struct.Struct(">IIIIIf")
DIVIDER = 1000.0
BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SOURCE = "dukascopy:datafeed:XAUUSD:native-m1-bid-ask-mid"
START = datetime.fromisoformat(os.getenv("GEN1_2Y_START", "2024-09-23")).replace(tzinfo=UTC)
END = datetime.fromisoformat(os.getenv("GEN1_2Y_END", "2026-09-23")).replace(tzinfo=UTC)
SPLIT = datetime.fromisoformat(os.getenv("GEN1_2Y_SPLIT", "2025-09-23")).replace(tzinfo=UTC)
STEP_MINUTES = int(os.getenv("GEN1_2Y_STEP_MINUTES", "5"))
WORKERS = int(os.getenv("GEN1_2Y_DOWNLOAD_WORKERS", "12"))
OUT = Path(os.getenv("GEN1_2Y_OUT", "artifacts/gen1_gold_2y_real_v1"))
MACRO_SYMBOLS = {
    "dxy": "DX-Y.NYB",
    "us10y": "^TNX",
    "vix": "^VIX",
    "spx": "^GSPC",
    "oil": "CL=F",
}


class FetchError(RuntimeError):
    pass


def candle_url(day: date, side: str) -> str:
    month0 = day.month - 1
    return (
        f"{BASE_URL}/XAUUSD/{day.year}/{month0:02d}/{day.day:02d}/"
        f"{side.upper()}_candles_min_1.bi5"
    )


def fetch_bytes(url: str, retries: int = 5) -> tuple[str, bytes]:
    last = None
    for attempt in range(retries):
        try:
            response = httpx.get(
                url,
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "PanWatch-GEN1-2Y-Research/1.0"},
            )
            if response.status_code == 404:
                return "notfound", b""
            if response.status_code == 429 or response.status_code >= 500:
                last = f"http_{response.status_code}"
                if attempt + 1 < retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else min(8.0, 0.5 * (2 ** attempt))
                    except ValueError:
                        delay = min(8.0, 0.5 * (2 ** attempt))
                    time.sleep(max(0.2, delay))
                    continue
            response.raise_for_status()
            return "data", bytes(response.content)
        except httpx.HTTPError as exc:
            last = type(exc).__name__
            if attempt + 1 < retries:
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    raise FetchError(f"{url} failed after {retries} attempts: {last}")


def decode_candles(payload: bytes, day: date) -> list[tuple[datetime, float, float, float, float, float]]:
    if not payload:
        return []
    try:
        raw = lzma.decompress(payload)
    except lzma.LZMAError as exc:
        raise FetchError("invalid LZMA candle payload") from exc
    if len(raw) % RECORD.size:
        raise FetchError(f"invalid candle payload length {len(raw)}")
    base = datetime.combine(day, dtime.min, tzinfo=UTC)
    rows = []
    for offset in range(0, len(raw), RECORD.size):
        sec, open_raw, close_raw, low_raw, high_raw, volume = RECORD.unpack_from(raw, offset)
        if sec >= 86400:
            raise FetchError(f"invalid candle second offset: {sec}")
        o = float(open_raw) / DIVIDER
        c = float(close_raw) / DIVIDER
        lo = float(low_raw) / DIVIDER
        hi = float(high_raw) / DIVIDER
        if lo > hi:
            lo, hi = hi, lo
        if min(o, c, lo, hi) <= 0:
            continue
        rows.append((base + timedelta(seconds=int(sec)), o, hi, lo, c, max(0.0, float(volume))))
    return rows


def fetch_day(day: date) -> dict[str, Any]:
    bid_status, bid_payload = fetch_bytes(candle_url(day, "BID"))
    ask_status, ask_payload = fetch_bytes(candle_url(day, "ASK"))
    if bid_status == "notfound" and ask_status == "notfound":
        return {"day": day.isoformat(), "status": "no_data", "rows": []}
    if bid_status != "data" or ask_status != "data":
        raise FetchError(f"one-sided candle availability on {day}: bid={bid_status} ask={ask_status}")
    bid = decode_candles(bid_payload, day)
    ask = decode_candles(ask_payload, day)
    bid_map = {row[0]: row for row in bid}
    ask_map = {row[0]: row for row in ask}
    common = sorted(set(bid_map).intersection(ask_map))
    rows = []
    crossed = 0
    for ts in common:
        b = bid_map[ts]
        a = ask_map[ts]
        if a[4] < b[4]:
            crossed += 1
            continue
        rows.append(
            (
                ts,
                (b[1] + a[1]) / 2.0,
                (b[2] + a[2]) / 2.0,
                (b[3] + a[3]) / 2.0,
                (b[4] + a[4]) / 2.0,
                b[5] + a[5],
                b[4],
                a[4],
            )
        )
    return {
        "day": day.isoformat(),
        "status": "data",
        "rows": rows,
        "bid_rows": len(bid),
        "ask_rows": len(ask),
        "matched_rows": len(rows),
        "crossed_rows": crossed,
    }


def date_range(start: datetime, end: datetime) -> list[date]:
    out = []
    cursor = start.date()
    while cursor < end.date():
        out.append(cursor)
        cursor += timedelta(days=1)
    return out


def download_day_results(
    start: datetime,
    end: datetime,
    *,
    progress_label: str = "dataset",
) -> list[dict[str, Any]]:
    days = date_range(start, end)
    results: list[dict[str, Any]] = []
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(2, WORKERS)) as pool:
        futures = {pool.submit(fetch_day, day): day for day in days}
        for completed, future in enumerate(as_completed(futures), start=1):
            day = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(json.dumps({
                    "phase": "dataset_day_failed",
                    "label": progress_label,
                    "day": day.isoformat(),
                    "completed_days": completed - 1,
                    "requested_days": len(days),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }), flush=True)
                raise
            if completed % 25 == 0 or completed == len(days):
                print(json.dumps({
                    "phase": "dataset_download_progress",
                    "label": progress_label,
                    "completed_days": completed,
                    "requested_days": len(days),
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }), flush=True)
    results.sort(key=lambda item: item["day"])
    return results


def build_dataset_from_results(
    results: list[dict[str, Any]],
    start: datetime,
    end: datetime,
) -> tuple[list[XAUBar], dict[datetime, tuple[float, float]], dict[str, Any]]:
    raw_rows = []
    quote_closes: dict[datetime, tuple[float, float]] = {}
    data_days = 0
    no_data_days = 0
    crossed = 0
    for result in sorted(results, key=lambda item: item["day"]):
        if result["status"] != "data":
            no_data_days += 1
            continue
        data_days += 1
        crossed += int(result.get("crossed_rows", 0))
        for ts, o, h, l, c, v, bid_close, ask_close in result["rows"]:
            if not (start <= ts < end):
                continue
            raw_rows.append((ts, o, h, l, c, v))
            quote_closes[ts + timedelta(minutes=1)] = (bid_close, ask_close)

    raw_rows.sort(key=lambda x: x[0])
    bars = [
        XAUBar(
            timestamp=ts,
            timeframe=XAUTimeframe.M1,
            open=o,
            high=h,
            low=l,
            close=c,
            volume=v,
            source=SOURCE,
            symbol="XAUUSD",
            execution_eligible=False,
        )
        for ts, o, h, l, c, v in raw_rows
    ]
    if not bars:
        raise FetchError("no M1 bars downloaded")

    hasher = hashlib.sha256()
    for row in raw_rows:
        hasher.update(
            (
                f"{row[0].isoformat()}|{row[1]:.3f}|{row[2]:.3f}|"
                f"{row[3]:.3f}|{row[4]:.3f}|{row[5]:.6f}\n"
            ).encode("utf-8")
        )

    minute_gaps = Counter()
    prior = None
    for bar in bars:
        if prior is not None:
            delta = int((bar.timestamp - prior).total_seconds() // 60)
            if delta > 1:
                minute_gaps[min(delta, 10080)] += 1
        prior = bar.timestamp

    spreads = []
    for bid_close, ask_close in quote_closes.values():
        mid = (bid_close + ask_close) / 2.0
        if mid > 0 and ask_close >= bid_close:
            spreads.append(((ask_close - bid_close) / mid) * 10000.0)

    diag = {
        "requested_days": len(date_range(start, end)),
        "data_days": data_days,
        "no_data_days": no_data_days,
        "m1_bar_count": len(bars),
        "crossed_rows_discarded": crossed,
        "first_bar": bars[0].timestamp.isoformat(),
        "last_bar": bars[-1].timestamp.isoformat(),
        "min_price": round(min(b.low for b in bars), 4),
        "max_price": round(max(b.high for b in bars), 4),
        "dataset_sha256": hasher.hexdigest(),
        "mean_close_spread_bps": round(sum(spreads) / len(spreads), 6) if spreads else None,
        "median_close_spread_bps": round(median(spreads), 6) if spreads else None,
        "gap_histogram_minutes_capped": {str(k): v for k, v in sorted(minute_gaps.items())},
        "source": SOURCE,
        "raw_data_redistributed": False,
    }
    return bars, quote_closes, diag


def save_dataset_shard(start: datetime, end: datetime, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = download_day_results(
        start,
        end,
        progress_label=f"{start.date()}..{end.date()}",
    )
    payload = {
        "format": "gen1_gold_2y_day_results_v1",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "results": results,
    }
    with gzip.open(output_path, "wb", compresslevel=6) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps({
        "phase": "dataset_shard_complete",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "day_count": len(results),
        "output_path": str(output_path),
        "size_bytes": output_path.stat().st_size,
    }), flush=True)


def load_dataset_shards(
    pattern: str,
    start: datetime,
    end: datetime,
) -> tuple[list[XAUBar], dict[datetime, tuple[float, float]], dict[str, Any]]:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FetchError(f"no dataset shards matched: {pattern}")

    by_day: dict[str, dict[str, Any]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("format") != "gen1_gold_2y_day_results_v1":
            raise FetchError(f"unsupported dataset shard format: {path}")
        for result in payload.get("results") or []:
            day = str(result["day"])
            if day in by_day and by_day[day] != result:
                raise FetchError(f"conflicting duplicate dataset day: {day}")
            by_day[day] = result

    expected = {day.isoformat() for day in date_range(start, end)}
    missing = sorted(expected.difference(by_day))
    if missing:
        raise FetchError(
            f"dataset shards missing {len(missing)} days; first={missing[:5]}"
        )
    selected = [by_day[day] for day in sorted(expected)]
    print(json.dumps({
        "phase": "dataset_shards_loaded",
        "shard_count": len(paths),
        "day_count": len(selected),
        "paths": paths,
    }), flush=True)
    return build_dataset_from_results(selected, start, end)


def download_dataset(
    start: datetime,
    end: datetime,
) -> tuple[list[XAUBar], dict[datetime, tuple[float, float]], dict[str, Any]]:
    shard_glob = os.getenv("GEN1_2Y_SHARD_GLOB")
    if shard_glob:
        return load_dataset_shards(shard_glob, start, end)
    return build_dataset_from_results(
        download_day_results(start, end),
        start,
        end,
    )

def bucket_start(ts: datetime, timeframe: XAUTimeframe) -> datetime:
    if timeframe is XAUTimeframe.M5:
        return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
    if timeframe is XAUTimeframe.M15:
        return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
    if timeframe is XAUTimeframe.H1:
        return ts.replace(minute=0, second=0, microsecond=0)
    if timeframe is XAUTimeframe.H4:
        return ts.replace(hour=(ts.hour // 4) * 4, minute=0, second=0, microsecond=0)
    if timeframe is XAUTimeframe.D1:
        return ts.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(timeframe)


def resample(m1: list[XAUBar], timeframe: XAUTimeframe) -> list[XAUBar]:
    groups: dict[datetime, list[XAUBar]] = defaultdict(list)
    for row in m1:
        groups[bucket_start(row.timestamp, timeframe)].append(row)
    out = []
    for ts in sorted(groups):
        rows = groups[ts]
        out.append(
            XAUBar(
                timestamp=ts,
                timeframe=timeframe,
                open=rows[0].open,
                high=max(r.high for r in rows),
                low=min(r.low for r in rows),
                close=rows[-1].close,
                volume=sum(float(r.volume or 0.0) for r in rows),
                source=f"{SOURCE}:resampled-{timeframe.value}",
                symbol="XAUUSD",
                execution_eligible=False,
            )
        )
    return out


TF_DURATION = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
    XAUTimeframe.H1: timedelta(hours=1),
    XAUTimeframe.H4: timedelta(hours=4),
    XAUTimeframe.D1: timedelta(days=1),
}


def availability(rows: list[XAUBar], timeframe: XAUTimeframe) -> list[datetime]:
    d = TF_DURATION[timeframe]
    return [r.timestamp + d for r in rows]


def bars_window(
    rows: list[XAUBar],
    available: list[datetime],
    evaluation_time: datetime,
    max_bars: int,
) -> list[XAUBar]:
    end = bisect_right(available, evaluation_time)
    return rows[max(0, end - max_bars):end]


def fetch_yahoo_daily(symbol: str, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        + quote(symbol, safe="")
        + f"?period1={int((start - timedelta(days=30)).timestamp())}"
        + f"&period2={int((end + timedelta(days=2)).timestamp())}"
        + "&interval=1d&events=history&includeAdjustedClose=true"
    )
    last_error = None
    for attempt in range(5):
        try:
            response = httpx.get(
                url,
                timeout=30.0,
                follow_redirects=True,
                headers={"User-Agent": "PanWatch-GEN1-2Y-Macro/1.0"},
            )
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"http_{response.status_code}"
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
                continue
            response.raise_for_status()
            body = response.json()
            result = ((body.get("chart") or {}).get("result") or [None])[0]
            if not result:
                return []
            timestamps = result.get("timestamp") or []
            quote_rows = (((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or [])
            out = []
            for stamp, close in zip(timestamps, quote_rows):
                if close is None:
                    continue
                # Conservative anti-lookahead: daily close becomes available next UTC day.
                observed = datetime.fromtimestamp(int(stamp), tz=UTC)
                available_at = observed.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                out.append((available_at, float(close)))
            return out
        except (httpx.HTTPError, ValueError, TypeError, json.JSONDecodeError) as exc:
            last_error = type(exc).__name__
            if attempt + 1 < 5:
                time.sleep(min(8.0, 0.5 * (2 ** attempt)))
    raise FetchError(f"macro series {symbol} unavailable: {last_error}")


def build_historical_macro_proxy(start: datetime, end: datetime):
    raw = {}
    for name, symbol in MACRO_SYMBOLS.items():
        try:
            raw[name] = fetch_yahoo_daily(symbol, start, end)
        except Exception:
            raw[name] = []

    usable = {name: rows for name, rows in raw.items() if len(rows) >= 10}
    if len(usable) < 3:
        raise FetchError("fewer than three historical macro proxy series are usable")
    usable_times = {
        name: [item[0] for item in rows]
        for name, rows in usable.items()
    }

    def value_at(
        name: str,
        rows: list[tuple[datetime, float]],
        at: datetime,
        lag: int = 0,
    ):
        times = usable_times[name]
        idx = bisect_right(times, at) - 1 - lag
        if idx < 0:
            return None
        return rows[idx][1], rows[idx][0]

    def provider(at: datetime) -> dict[str, Any]:
        components = []
        drivers = []

        def pct_component(name: str, scale: float, invert: bool = False, weight: float = 0.0):
            rows = usable.get(name)
            if not rows:
                return 0.0
            now_pair = value_at(name, rows, at, 0)
            prev_pair = value_at(name, rows, at, 5)
            if not now_pair or not prev_pair or not prev_pair[0]:
                return 0.0
            change = (now_pair[0] - prev_pair[0]) / prev_pair[0]
            score = max(-1.0, min(1.0, change / scale))
            if invert:
                score = -score
            drivers.append({
                "name": f"{name}_5d",
                "value": round(change, 6),
                "gold_score": round(score, 4),
                "available_at": now_pair[1].isoformat(),
            })
            components.append((weight, score))
            return score

        pct_component("dxy", 0.012, invert=True, weight=0.35)
        pct_component("vix", 0.25, invert=False, weight=0.15)
        pct_component("spx", 0.04, invert=True, weight=0.10)
        pct_component("oil", 0.08, invert=False, weight=0.10)

        rows = usable.get("us10y")
        if rows:
            now_pair = value_at("us10y", rows, at, 0)
            prev_pair = value_at("us10y", rows, at, 5)
            if now_pair and prev_pair:
                delta = now_pair[0] - prev_pair[0]
                score = max(-1.0, min(1.0, -delta / 0.35))
                drivers.append({
                    "name": "us10y_5d_delta",
                    "value": round(delta, 4),
                    "gold_score": round(score, 4),
                    "available_at": now_pair[1].isoformat(),
                })
                components.append((0.30, score))

        total_weight = sum(weight for weight, _ in components)
        score = (
            sum(weight * value for weight, value in components) / total_weight
            if total_weight else 0.0
        )
        score = max(-1.0, min(1.0, score))
        bias = 1 if score >= 0.15 else -1 if score <= -0.15 else 0
        confidence = min(0.85, 0.35 + abs(score) * 0.55)
        return {
            "bias": bias,
            "bias_label": "bullish" if bias > 0 else "bearish" if bias < 0 else "neutral",
            "confidence": round(confidence, 4),
            "event_risk": False,
            "calendar_ok": True,
            "search_ok": True,
            "synthesis_ok": True,
            "cache_stale": False,
            "refresh_pending": False,
            "historical_macro_proxy": True,
            "proxy_score": round(score, 4),
            "drivers": drivers,
            "observed_at": at.isoformat(),
            "limitations": [
                "One-day-lagged public market proxy, not exact Ahmed Toolbox news/calendar replay.",
                "Historical event-risk calendar is not reconstructed.",
            ],
        }

    diagnostics = {
        "provider": "Yahoo public chart daily proxies",
        "symbols": MACRO_SYMBOLS,
        "usable_series": sorted(usable),
        "lookahead_policy": "daily close usable next UTC day",
        "exact_live_news_pipeline_replayed": False,
        "event_calendar_replayed": False,
    }
    return provider, diagnostics


def future_close(
    m1: list[XAUBar],
    m1_available: list[datetime],
    target: datetime,
    tolerance_minutes: int = 2,
) -> tuple[float, datetime] | None:
    idx = bisect_left(m1_available, target)
    if idx >= len(m1):
        return None
    observed = m1_available[idx]
    if observed - target > timedelta(minutes=tolerance_minutes):
        return None
    return float(m1[idx].close), observed


def range_outcomes(
    m1: list[XAUBar],
    m1_available: list[datetime],
    evaluation_time: datetime,
    entry: float,
    horizon_minutes: int,
) -> dict[str, Any]:
    start = bisect_right(m1_available, evaluation_time)
    end = bisect_right(
        m1_available,
        evaluation_time + timedelta(minutes=horizon_minutes),
    )
    future = m1[start:end]
    distances = (10, 20, 30)
    result: dict[str, Any] = {
        "horizon_minutes": horizon_minutes,
        "bar_count": len(future),
        "lookahead_used_for_label_only": True,
        "levels": {},
    }
    if not future:
        return result

    max_up = float("-inf")
    max_down = float("-inf")
    level_state = {
        distance: {
            "up_hit": False,
            "down_hit": False,
            "first_hit": "none",
            "first_hit_at": None,
        }
        for distance in distances
    }
    for bar in future:
        max_up = max(max_up, float(bar.high) - entry)
        max_down = max(max_down, entry - float(bar.low))
        available_at = None
        for distance in distances:
            state = level_state[distance]
            hu = float(bar.high) >= entry + distance
            hd = float(bar.low) <= entry - distance
            state["up_hit"] = state["up_hit"] or hu
            state["down_hit"] = state["down_hit"] or hd
            if state["first_hit"] == "none" and (hu or hd):
                if available_at is None:
                    available_at = (bar.timestamp + timedelta(minutes=1)).isoformat()
                state["first_hit_at"] = available_at
                state["first_hit"] = (
                    "ambiguous_same_bar"
                    if hu and hd
                    else "up"
                    if hu
                    else "down"
                )

    result["max_up_usd"] = round(max_up, 4)
    result["max_down_usd"] = round(max_down, 4)
    for distance in distances:
        state = level_state[distance]
        result["levels"][f"pm{distance}"] = {
            "distance_usd": distance,
            "up_level": round(entry + distance, 4),
            "down_level": round(entry - distance, 4),
            **state,
        }
    return result

def spread_net_bps(
    candidate: str,
    entry_at: datetime,
    outcome_at: datetime,
    quotes: dict[datetime, tuple[float, float]],
) -> float | None:
    entry_quote = quotes.get(entry_at)
    exit_quote = quotes.get(outcome_at)
    if not entry_quote or not exit_quote:
        return None
    entry_bid, entry_ask = entry_quote
    exit_bid, exit_ask = exit_quote
    if candidate == "long_setup":
        if entry_ask <= 0:
            return None
        return ((exit_bid - entry_ask) / entry_ask) * 10000.0
    if candidate == "short_setup":
        if entry_bid <= 0:
            return None
        return ((entry_bid - exit_ask) / entry_bid) * 10000.0
    return None


def run_replay_horizons(
    bars_by_tf: dict[XAUTimeframe, list[XAUBar]],
    quotes: dict[datetime, tuple[float, float]],
    horizons: tuple[int, ...],
    macro_provider,
) -> dict[int, list[ReplayEpisode]]:
    """Run the fixed GEN1 decision stack once per evaluation and label many horizons.

    A cheap prefilter uses the exact same intraday candidate engine on the same
    closed M1/M5/M15 windows. HTF context, cognition and evidence fusion are only
    built when that prefilter produces a candidate, preserving decision semantics
    while avoiding duplicate expensive work.
    """
    horizons = tuple(sorted({max(1, int(value)) for value in horizons}))
    available = {tf: availability(rows, tf) for tf, rows in bars_by_tf.items()}
    m1 = bars_by_tf[XAUTimeframe.M1]
    m1_available = available[XAUTimeframe.M1]
    episodes: dict[int, list[ReplayEpisode]] = {horizon: [] for horizon in horizons}
    last_eval = None
    prefilter = XAUIntradayEngine(require_execution_data=False)
    context_cache: dict[
        tuple[datetime | None, datetime | None, datetime | None],
        dict[str, Any],
    ] = {}

    def cached_market_context(
        hourly: list[XAUBar],
        h4: list[XAUBar],
        daily: list[XAUBar],
    ) -> dict[str, Any]:
        key = (
            hourly[-1].timestamp if hourly else None,
            h4[-1].timestamp if h4 else None,
            daily[-1].timestamp if daily else None,
        )
        cached = context_cache.get(key)
        if cached is not None:
            return cached
        value = build_market_context(hourly, h4, daily)
        context_cache[key] = value
        return value

    replay_started = time.monotonic()
    processed_evaluations = 0
    candidate_count = 0
    profile_enabled = os.getenv("GEN1_2Y_PROFILE", "0") == "1"
    profile_seconds: dict[str, float] = defaultdict(float)

    def profile_add(name: str, started: float) -> None:
        if profile_enabled:
            profile_seconds[name] += time.perf_counter() - started
    for evaluation_time in m1_available:
        if evaluation_time < START or evaluation_time >= END:
            continue
        if last_eval is not None and evaluation_time - last_eval < timedelta(minutes=STEP_MINUTES):
            continue
        last_eval = evaluation_time
        processed_evaluations += 1
        if processed_evaluations % 10000 == 0:
            print(json.dumps({
                "phase": "replay_progress",
                "processed_evaluations": processed_evaluations,
                "candidate_count": candidate_count,
                "elapsed_seconds": round(time.monotonic() - replay_started, 2),
                "evaluation_time": evaluation_time.isoformat(),
                "profile_seconds": {
                    key: round(value, 3)
                    for key, value in sorted(profile_seconds.items())
                },
            }), flush=True)

        stage_started = time.perf_counter()
        intraday_window = {
            XAUTimeframe.M1: bars_window(m1, m1_available, evaluation_time, 300),
            XAUTimeframe.M5: bars_window(
                bars_by_tf[XAUTimeframe.M5],
                available[XAUTimeframe.M5],
                evaluation_time,
                300,
            ),
            XAUTimeframe.M15: bars_window(
                bars_by_tf[XAUTimeframe.M15],
                available[XAUTimeframe.M15],
                evaluation_time,
                300,
            ),
        }
        profile_add("intraday_windows", stage_started)
        stage_started = time.perf_counter()
        pre = prefilter.analyze(
            intraday_window,
            event_risk=False,
            macro_bias=0,
            now=evaluation_time,
            assume_sorted=True,
        )
        profile_add("prefilter", stage_started)
        if pre.blocked or pre.candidate not in {"long_setup", "short_setup"}:
            continue
        candidate_count += 1

        stage_started = time.perf_counter()
        window = {
            **intraday_window,
            XAUTimeframe.H1: bars_window(
                bars_by_tf[XAUTimeframe.H1],
                available[XAUTimeframe.H1],
                evaluation_time,
                1100,
            ),
            XAUTimeframe.H4: bars_window(
                bars_by_tf[XAUTimeframe.H4],
                available[XAUTimeframe.H4],
                evaluation_time,
                1100,
            ),
            XAUTimeframe.D1: bars_window(
                bars_by_tf[XAUTimeframe.D1],
                available[XAUTimeframe.D1],
                evaluation_time,
                1100,
            ),
        }
        profile_add("candidate_windows", stage_started)
        stage_started = time.perf_counter()
        macro = macro_provider(evaluation_time)
        profile_add("macro", stage_started)
        stage_started = time.perf_counter()
        technical = build_replay_technical_state(
            window,
            evaluation_time,
            macro_bias=int(macro.get("bias", 0) or 0),
            market_context_builder=cached_market_context,
            intraday_assessment=pre,
        )
        profile_add("technical_state", stage_started)
        candidate = str(technical.get("candidate") or "none")
        if technical.get("blocked") or candidate not in {"long_setup", "short_setup"}:
            continue

        stage_started = time.perf_counter()
        cognition = build_cognitive_state(
            technical,
            macro,
            memory=None,
            min_confidence=0.58,
        )
        profile_add("cognition", stage_started)
        stage_started = time.perf_counter()
        fusion = build_decision_fusion(
            technical,
            macro,
            memory=None,
            min_confidence=0.58,
            as_of=evaluation_time,
        )
        profile_add("fusion", stage_started)
        stage_started = time.perf_counter()
        evidence = build_gen1_evidence_fusion(
            technical,
            macro,
            fusion,
            memory=None,
            require_xaut=False,
        )
        profile_add("evidence", stage_started)
        entry = float(technical["analysis_reference"]["price"])
        side = 1.0 if candidate == "long_setup" else -1.0
        regime = str((cognition.get("regime") or {}).get("label") or "unknown")
        confidence = float(
            (cognition.get("confidence") or {}).get("calibrated_confidence") or 0.0
        )
        shared_meta = {
            "research_only": True,
            "lookahead_protected": True,
            "gen1_decision": str(evidence.get("decision") or "WAIT"),
            "gen1_decision_confidence": evidence.get("decision_confidence"),
            "gen1_fusion_state": fusion.get("state"),
            "evidence_fusion": evidence,
            "sensor_gaps": [
                "historical_live_news_synthesis_not_reconstructed",
                "historical_event_calendar_not_reconstructed",
                "historical_xaut_raw_book_not_available",
                "historical_xaut_trade_tape_not_available",
            ],
            "historical_macro_proxy": {
                "bias": macro.get("bias"),
                "confidence": macro.get("confidence"),
                "proxy_score": macro.get("proxy_score"),
                "drivers": macro.get("drivers"),
            },
            "replay_scope": (
                "real_xau_price_htf_structure_volume_spread_plus_point_in_time_"
                "macro_proxy_without_historical_xaut_microstructure_or_live_news"
            ),
        }

        stage_started = time.perf_counter()
        for horizon_minutes in horizons:
            future = future_close(
                m1,
                m1_available,
                evaluation_time + timedelta(minutes=horizon_minutes),
            )
            if future is None:
                continue
            outcome_price, outcome_at = future
            gross_bps = ((outcome_price - entry) / entry) * 10000.0 * side
            net_bps = spread_net_bps(
                candidate,
                evaluation_time,
                outcome_at,
                quotes,
            )
            ranges = range_outcomes(
                m1,
                m1_available,
                evaluation_time,
                entry,
                horizon_minutes,
            )
            key = hashlib.sha1(
                (
                    f"{SOURCE}|{evaluation_time.isoformat()}|{candidate}|"
                    f"{horizon_minutes}|{entry:.6f}"
                ).encode()
            ).hexdigest()
            meta = dict(shared_meta)
            meta["range_outcomes"] = ranges
            meta["net_spread_directional_return_bps"] = (
                round(net_bps, 4) if net_bps is not None else None
            )
            episodes[horizon_minutes].append(
                ReplayEpisode(
                    replay_key=key,
                    observed_at=evaluation_time,
                    outcome_at=outcome_at,
                    candidate=candidate,
                    regime=regime,
                    confidence=round(confidence, 6),
                    horizon_minutes=horizon_minutes,
                    entry_price=round(entry, 6),
                    outcome_price=round(outcome_price, 6),
                    directional_return_bps=round(gross_bps, 4),
                    positive=gross_bps > 0,
                    source=SOURCE,
                    state_vector=dict(cognition.get("market_state") or {}),
                    cognition=cognition,
                    meta=meta,
                )
            )
        profile_add("labeling", stage_started)

    print(json.dumps({
        "phase": "replay_complete",
        "processed_evaluations": processed_evaluations,
        "candidate_count": candidate_count,
        "elapsed_seconds": round(time.monotonic() - replay_started, 2),
        "profile_seconds": {
            key: round(value, 3)
            for key, value in sorted(profile_seconds.items())
        },
    }), flush=True)
    return episodes


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float | None:
    if n <= 0:
        return None
    p = successes / n
    z2 = z * z
    denominator = 1 + z2 / n
    centre = p + z2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) / n) + z2 / (4 * n * n))
    return max(0.0, (centre - margin) / denominator)


def bootstrap_mean_ci(values: list[float], iterations: int = 3000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    rng = random.Random(20260923)
    n = len(values)
    means = []
    for _ in range(iterations):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * (len(means) - 1))]
    hi = means[int(0.975 * (len(means) - 1))]
    return lo, hi


def decorrelate(episodes: list[ReplayEpisode], minutes: int) -> list[ReplayEpisode]:
    chosen = []
    last = None
    gap = timedelta(minutes=minutes)
    for episode in sorted(episodes, key=lambda x: x.observed_at):
        if last is None or episode.observed_at - last >= gap:
            chosen.append(episode)
            last = episode.observed_at
    return chosen


def session_name(ts: datetime) -> str:
    h = ts.hour
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 13:
        return "london"
    if 13 <= h < 21:
        return "new_york"
    return "off_hours"


def stats_for(episodes: list[ReplayEpisode], horizon: int) -> dict[str, Any]:
    if not episodes:
        return {"episode_count": 0}
    validation = evaluate_gen1_replay(episodes)
    independent = decorrelate(episodes, horizon)
    net_values = [
        float(ep.meta["net_spread_directional_return_bps"])
        for ep in independent
        if ep.meta.get("net_spread_directional_return_bps") is not None
    ]
    gross_values = [float(ep.directional_return_bps) for ep in independent]
    wins = sum(1 for v in net_values if v > 0)
    lo, hi = bootstrap_mean_ci(net_values)
    final = [
        ep for ep in independent
        if str(ep.meta.get("gen1_decision") or "WAIT").upper() in {"LONG", "SHORT"}
    ]
    final_net = [
        float(ep.meta["net_spread_directional_return_bps"])
        for ep in final if ep.meta.get("net_spread_directional_return_bps") is not None
    ]
    final_wins = sum(1 for v in final_net if v > 0)
    final_lo, final_hi = bootstrap_mean_ci(final_net)

    session_counts = Counter(session_name(ep.observed_at) for ep in independent)
    regime_counts = Counter(ep.regime for ep in independent)
    decision_counts = Counter(str(ep.meta.get("gen1_decision") or "WAIT").upper() for ep in independent)

    def drawdown(values: list[float]) -> float:
        equity = peak = 0.0
        worst = 0.0
        for value in values:
            equity += value
            peak = max(peak, equity)
            worst = min(worst, equity - peak)
        return worst

    return {
        "episode_count_raw": len(episodes),
        "episode_count_horizon_decorrelated": len(independent),
        "headline_independence_gap_minutes": horizon,
        "candidate_net_positive_rate": round(wins / len(net_values), 4) if net_values else None,
        "candidate_net_wilson_95_lower": round(wilson_lower(wins, len(net_values)), 4) if net_values else None,
        "candidate_gross_mean_bps": round(sum(gross_values) / len(gross_values), 4) if gross_values else None,
        "candidate_net_mean_bps": round(sum(net_values) / len(net_values), 4) if net_values else None,
        "candidate_net_median_bps": round(median(net_values), 4) if net_values else None,
        "candidate_net_bootstrap_95_ci_bps": [
            round(lo, 4) if lo is not None else None,
            round(hi, 4) if hi is not None else None,
        ],
        "candidate_cumulative_net_bps": round(sum(net_values), 4),
        "candidate_max_drawdown_bps": round(drawdown(net_values), 4),
        "final_gen1_directional_n": len(final_net),
        "final_gen1_net_positive_rate": round(final_wins / len(final_net), 4) if final_net else None,
        "final_gen1_net_wilson_95_lower": round(wilson_lower(final_wins, len(final_net)), 4) if final_net else None,
        "final_gen1_net_mean_bps": round(sum(final_net) / len(final_net), 4) if final_net else None,
        "final_gen1_net_bootstrap_95_ci_bps": [
            round(final_lo, 4) if final_lo is not None else None,
            round(final_hi, 4) if final_hi is not None else None,
        ],
        "decision_counts_independent": dict(decision_counts),
        "session_counts_independent": dict(session_counts),
        "regime_counts_independent": dict(regime_counts),
        "validation_contract": validation,
    }


def split_period(episodes: list[ReplayEpisode], start: datetime, end: datetime) -> list[ReplayEpisode]:
    return [ep for ep in episodes if start <= ep.observed_at < end]


def write_csv(path: Path, episodes: list[ReplayEpisode]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "observed_at", "outcome_at", "horizon_minutes", "candidate", "regime",
            "cognitive_confidence", "gen1_decision", "gen1_confidence", "entry_price",
            "outcome_price", "gross_directional_bps", "net_spread_directional_bps",
            "session", "fusion_state",
        ])
        for ep in episodes:
            writer.writerow([
                ep.observed_at.isoformat(),
                ep.outcome_at.isoformat(),
                ep.horizon_minutes,
                ep.candidate,
                ep.regime,
                ep.confidence,
                ep.meta.get("gen1_decision"),
                ep.meta.get("gen1_decision_confidence"),
                ep.entry_price,
                ep.outcome_price,
                ep.directional_return_bps,
                ep.meta.get("net_spread_directional_return_bps"),
                session_name(ep.observed_at),
                ep.meta.get("gen1_fusion_state"),
            ])


def make_charts(daily: list[XAUBar], episodes60: list[ReplayEpisode]) -> list[str]:
    import matplotlib.pyplot as plt

    charts = []
    xs = [row.timestamp for row in daily]
    ys = [row.close for row in daily]

    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys)
    ax.set_title("XAUUSD — Dukascopy midpoint daily close — 2Y")
    ax.set_xlabel("Date")
    ax.set_ylabel("USD/oz")
    ax.grid(True, alpha=0.2)
    price_path = OUT / "xauusd_2y_price.png"
    fig.tight_layout()
    fig.savefig(price_path, dpi=150)
    plt.close(fig)
    charts.append(str(price_path))

    independent = decorrelate(episodes60, 60)
    cum = []
    total = 0.0
    cx = []
    for ep in independent:
        value = ep.meta.get("net_spread_directional_return_bps")
        if value is None:
            continue
        total += float(value)
        cx.append(ep.observed_at)
        cum.append(total)
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.plot(cx, cum)
    ax.axhline(0, linewidth=1)
    ax.set_title("Candidate baseline cumulative spread-adjusted directional bps (60m-decorrelated)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative bps")
    ax.grid(True, alpha=0.2)
    path = OUT / "candidate_cumulative_net_bps.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append(str(path))

    monthly: dict[str, list[float]] = defaultdict(list)
    for ep in independent:
        value = ep.meta.get("net_spread_directional_return_bps")
        if value is not None:
            monthly[ep.observed_at.strftime("%Y-%m")].append(float(value))
    labels = sorted(monthly)
    rates = [sum(1 for v in monthly[key] if v > 0) / len(monthly[key]) for key in labels]
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    ax.bar(labels, rates)
    ax.axhline(0.5, linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_title("Monthly candidate positive rate after observed BID/ASK spread — 60m horizon")
    ax.set_xlabel("Month")
    ax.set_ylabel("Positive rate")
    ax.tick_params(axis="x", rotation=70)
    fig.tight_layout()
    path = OUT / "monthly_candidate_positive_rate.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    charts.append(str(path))
    return charts


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    phase_started = time.monotonic()
    m1, quotes, dataset = download_dataset(START, END)
    print(json.dumps({
        "phase": "dataset_download_complete",
        "m1_bar_count": len(m1),
        "data_days": dataset.get("data_days"),
        "elapsed_seconds": round(time.monotonic() - phase_started, 2),
    }), flush=True)
    bars_by_tf = {
        XAUTimeframe.M1: m1,
        XAUTimeframe.M5: resample(m1, XAUTimeframe.M5),
        XAUTimeframe.M15: resample(m1, XAUTimeframe.M15),
        XAUTimeframe.H1: resample(m1, XAUTimeframe.H1),
        XAUTimeframe.H4: resample(m1, XAUTimeframe.H4),
        XAUTimeframe.D1: resample(m1, XAUTimeframe.D1),
    }
    dataset["bar_counts"] = {tf.value: len(rows) for tf, rows in bars_by_tf.items()}
    print(json.dumps({
        "phase": "resample_complete",
        "bar_counts": dataset["bar_counts"],
        "elapsed_seconds": round(time.monotonic() - phase_started, 2),
    }), flush=True)

    macro_provider, macro_diagnostics = build_historical_macro_proxy(START, END)
    print(json.dumps({
        "phase": "macro_proxy_complete",
        "usable_series": macro_diagnostics.get("usable_series"),
        "elapsed_seconds": round(time.monotonic() - phase_started, 2),
    }), flush=True)
    episodes_by_horizon = run_replay_horizons(
        bars_by_tf,
        quotes,
        (60, 240),
        macro_provider,
    )
    episodes60 = episodes_by_horizon[60]
    episodes240 = episodes_by_horizon[240]

    report = {
        "benchmark": "gen1-gold-2y-real-v1",
        "strategy_revision_env": os.getenv("STRATEGY_REVISION") or os.getenv("GITHUB_SHA") or "unknown",
        "period": {
            "start_utc": START.isoformat(),
            "split_utc": SPLIT.isoformat(),
            "end_utc_exclusive": END.isoformat(),
            "reference_window": [START.isoformat(), SPLIT.isoformat()],
            "holdout_window": [SPLIT.isoformat(), END.isoformat()],
        },
        "dataset": dataset,
        "methodology": {
            "evaluation_step_minutes": STEP_MINUTES,
            "walk_forward": True,
            "lookahead_in_decision": False,
            "headline_outcome_overlap_removed": True,
            "spread_adjustment": "real Dukascopy BID/ASK M1 close spread",
            "slippage_model": "not included",
            "historical_macro_market_proxy": True,
            "historical_live_news_synthesis_reconstructed": False,
            "historical_event_calendar_reconstructed": False,
            "historical_xaut_raw_book_reconstructed": False,
            "historical_xaut_trade_tape_reconstructed": False,
            "raw_data_uploaded_as_artifact": False,
        },
        "historical_macro_proxy": macro_diagnostics,
        "horizon_60m": {
            "overall": stats_for(episodes60, 60),
            "reference_year": stats_for(split_period(episodes60, START, SPLIT), 60),
            "holdout_year": stats_for(split_period(episodes60, SPLIT, END), 60),
        },
        "horizon_240m": {
            "overall": stats_for(episodes240, 240),
            "reference_year": stats_for(split_period(episodes240, START, SPLIT), 240),
            "holdout_year": stats_for(split_period(episodes240, SPLIT, END), 240),
        },
        "limitations": [
            "This is a real XAUUSD price/structure/spread replay, not a reconstruction of every live sensor.",
            "Macro uses one-day-lagged public market proxies (DXY, US10Y, VIX, S&P 500, crude); the exact Ahmed Toolbox live news/calendar synthesis is not reconstructed.",
            "Historical raw Bitfinex XAUT order book and complete trade tape are not available in this dataset, so they are not fabricated.",
            "Dukascopy candle volume is an activity proxy and is not centralized global gold traded volume.",
            "Spread is included from observed BID/ASK closes; broker slippage, commissions, latency and fill rejection are not modeled.",
            "Passing the data-integrity benchmark does not prove an executable trading edge.",
        ],
    }

    integrity_ok = bool(
        dataset["m1_bar_count"] > 400_000
        and dataset["data_days"] > 450
        and len(episodes60) > 100
        and len(episodes240) > 100
        and dataset["crossed_rows_discarded"] < max(100, dataset["m1_bar_count"] * 0.001)
    )
    report["completed"] = True
    report["data_integrity_passed"] = integrity_ok
    report["edge_proven"] = False

    report_path = OUT / "gen1_2y_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_csv(OUT / "episodes_60m.csv", episodes60)
    write_csv(OUT / "episodes_240m.csv", episodes240)
    make_charts(bars_by_tf[XAUTimeframe.D1], episodes60)

    summary = [
        "# GEN1 Gold — Two-Year Real-Data Backtest",
        "",
        f"- Period: {START.date()} to {END.date()} (end exclusive)",
        f"- Source: {SOURCE}",
        f"- M1 bars: {dataset['m1_bar_count']:,}",
        f"- Strategy revision: {report['strategy_revision_env']}",
        "",
        "## 60m holdout headline",
        "",
        f"- Horizon-decorrelated N: {report['horizon_60m']['holdout_year'].get('episode_count_horizon_decorrelated')}",
        f"- Candidate positive rate after spread: {report['horizon_60m']['holdout_year'].get('candidate_net_positive_rate')}",
        f"- Candidate Wilson 95% lower: {report['horizon_60m']['holdout_year'].get('candidate_net_wilson_95_lower')}",
        f"- Candidate mean bps after spread: {report['horizon_60m']['holdout_year'].get('candidate_net_mean_bps')}",
        f"- Final GEN1 directional N: {report['horizon_60m']['holdout_year'].get('final_gen1_directional_n')}",
        f"- Final GEN1 positive rate after spread: {report['horizon_60m']['holdout_year'].get('final_gen1_net_positive_rate')}",
        "",
        "## Important",
        "",
        "Historical macro market proxies are point-in-time and one-day lagged. Exact live news/calendar synthesis "
        "and raw XAUT microstructure are not fabricated.",
    ]
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")

    print(json.dumps({
        "benchmark": report["benchmark"],
        "data_integrity_passed": integrity_ok,
        "m1_bars": dataset["m1_bar_count"],
        "episodes_60m": len(episodes60),
        "episodes_240m": len(episodes240),
        "holdout_60m": report["horizon_60m"]["holdout_year"],
        "holdout_240m": report["horizon_240m"]["holdout_year"],
        "artifact_dir": str(OUT),
        "edge_proven": False,
    }, indent=2, sort_keys=True))

    if not integrity_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    if os.getenv("GEN1_2Y_PREPARE_ONLY") == "1":
        shard_start = datetime.fromisoformat(
            os.environ["GEN1_2Y_SHARD_START"]
        ).replace(tzinfo=UTC)
        shard_end = datetime.fromisoformat(
            os.environ["GEN1_2Y_SHARD_END"]
        ).replace(tzinfo=UTC)
        shard_out = Path(os.environ["GEN1_2Y_SHARD_OUT"])
        save_dataset_shard(shard_start, shard_end, shard_out)
        sys.exit(0)
    main()
