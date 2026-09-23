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
import hashlib
import json
import lzma
import math
import os
import random
import struct
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

import httpx

from src.modules.xau.cognition import build_cognitive_state
from src.modules.xau.evidence_fusion import build_gen1_evidence_fusion
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


def download_dataset(start: datetime, end: datetime) -> tuple[list[XAUBar], dict[datetime, tuple[float, float]], dict[str, Any]]:
    days = date_range(start, end)
    results = []
    with ThreadPoolExecutor(max_workers=max(2, WORKERS)) as pool:
        futures = {pool.submit(fetch_day, day): day for day in days}
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda x: x["day"])

    raw_rows = []
    quote_closes: dict[datetime, tuple[float, float]] = {}
    data_days = 0
    no_data_days = 0
    crossed = 0
    for result in results:
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
        "requested_days": len(days),
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


def macro_unavailable(evaluation_time: datetime) -> dict[str, Any]:
    return {
        "bias": 0,
        "bias_label": "neutral",
        "confidence": 0.0,
        "event_risk": False,
        "calendar_ok": False,
        "search_ok": False,
        "synthesis_ok": False,
        "cache_stale": False,
        "refresh_pending": False,
        "historical_gap": "external_macro_not_reconstructed",
        "observed_at": evaluation_time.isoformat(),
    }


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
    end = bisect_right(m1_available, evaluation_time + timedelta(minutes=horizon_minutes))
    future = m1[start:end]
    result: dict[str, Any] = {
        "horizon_minutes": horizon_minutes,
        "bar_count": len(future),
        "lookahead_used_for_label_only": True,
        "levels": {},
    }
    if not future:
        return result
    result["max_up_usd"] = round(max(r.high - entry for r in future), 4)
    result["max_down_usd"] = round(max(entry - r.low for r in future), 4)
    for distance in (10, 20, 30):
        first = "none"
        first_at = None
        up = down = False
        for bar in future:
            hu = bar.high >= entry + distance
            hd = bar.low <= entry - distance
            up = up or hu
            down = down or hd
            if first == "none" and (hu or hd):
                first_at = (bar.timestamp + timedelta(minutes=1)).isoformat()
                first = "ambiguous_same_bar" if hu and hd else "up" if hu else "down"
        result["levels"][f"pm{distance}"] = {
            "distance_usd": distance,
            "up_level": round(entry + distance, 4),
            "down_level": round(entry - distance, 4),
            "up_hit": up,
            "down_hit": down,
            "first_hit": first,
            "first_hit_at": first_at,
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


def run_replay(
    bars_by_tf: dict[XAUTimeframe, list[XAUBar]],
    quotes: dict[datetime, tuple[float, float]],
    horizon_minutes: int,
) -> list[ReplayEpisode]:
    available = {tf: availability(rows, tf) for tf, rows in bars_by_tf.items()}
    m1 = bars_by_tf[XAUTimeframe.M1]
    m1_available = available[XAUTimeframe.M1]
    episodes: list[ReplayEpisode] = []
    last_eval = None

    for evaluation_time in m1_available:
        if evaluation_time < START or evaluation_time >= END:
            continue
        if last_eval is not None and evaluation_time - last_eval < timedelta(minutes=STEP_MINUTES):
            continue
        last_eval = evaluation_time

        window = {
            XAUTimeframe.M1: bars_window(m1, m1_available, evaluation_time, 320),
            XAUTimeframe.M5: bars_window(bars_by_tf[XAUTimeframe.M5], available[XAUTimeframe.M5], evaluation_time, 320),
            XAUTimeframe.M15: bars_window(bars_by_tf[XAUTimeframe.M15], available[XAUTimeframe.M15], evaluation_time, 320),
            XAUTimeframe.H1: bars_window(bars_by_tf[XAUTimeframe.H1], available[XAUTimeframe.H1], evaluation_time, 1100),
            XAUTimeframe.H4: bars_window(bars_by_tf[XAUTimeframe.H4], available[XAUTimeframe.H4], evaluation_time, 1100),
            XAUTimeframe.D1: bars_window(bars_by_tf[XAUTimeframe.D1], available[XAUTimeframe.D1], evaluation_time, 1100),
        }
        technical = build_replay_technical_state(window, evaluation_time, macro_bias=0)
        candidate = str(technical.get("candidate") or "none")
        if technical.get("blocked") or candidate not in {"long_setup", "short_setup"}:
            continue

        macro = macro_unavailable(evaluation_time)
        cognition = build_cognitive_state(technical, macro, memory=None, min_confidence=0.58)
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
        future = future_close(
            m1,
            m1_available,
            evaluation_time + timedelta(minutes=horizon_minutes),
        )
        if future is None:
            continue
        outcome_price, outcome_at = future
        entry = float(technical["analysis_reference"]["price"])
        side = 1.0 if candidate == "long_setup" else -1.0
        gross_bps = ((outcome_price - entry) / entry) * 10000.0 * side
        net_bps = spread_net_bps(candidate, evaluation_time, outcome_at, quotes)
        ranges = range_outcomes(m1, m1_available, evaluation_time, entry, horizon_minutes)
        regime = str((cognition.get("regime") or {}).get("label") or "unknown")
        confidence = float((cognition.get("confidence") or {}).get("calibrated_confidence") or 0.0)
        key = hashlib.sha1(
            f"{SOURCE}|{evaluation_time.isoformat()}|{candidate}|{horizon_minutes}|{entry:.6f}".encode()
        ).hexdigest()
        episodes.append(
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
                meta={
                    "research_only": True,
                    "lookahead_protected": True,
                    "gen1_decision": str(evidence.get("decision") or "WAIT"),
                    "gen1_decision_confidence": evidence.get("decision_confidence"),
                    "gen1_fusion_state": fusion.get("state"),
                    "evidence_fusion": evidence,
                    "range_outcomes": ranges,
                    "net_spread_directional_return_bps": round(net_bps, 4) if net_bps is not None else None,
                    "sensor_gaps": [
                        "historical_external_macro_not_reconstructed",
                        "historical_xaut_raw_book_not_available",
                        "historical_xaut_trade_tape_not_available",
                    ],
                    "replay_scope": "real_xau_price_htf_structure_volume_spread_without_historical_external_macro_or_xaut_microstructure",
                },
            )
        )
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

    m1, quotes, dataset = download_dataset(START, END)
    bars_by_tf = {
        XAUTimeframe.M1: m1,
        XAUTimeframe.M5: resample(m1, XAUTimeframe.M5),
        XAUTimeframe.M15: resample(m1, XAUTimeframe.M15),
        XAUTimeframe.H1: resample(m1, XAUTimeframe.H1),
        XAUTimeframe.H4: resample(m1, XAUTimeframe.H4),
        XAUTimeframe.D1: resample(m1, XAUTimeframe.D1),
    }
    dataset["bar_counts"] = {tf.value: len(rows) for tf, rows in bars_by_tf.items()}

    episodes60 = run_replay(bars_by_tf, quotes, 60)
    episodes240 = run_replay(bars_by_tf, quotes, 240)

    report = {
        "benchmark": "gen1-gold-2y-real-v1",
        "strategy_revision_env": os.getenv("GITHUB_SHA") or "unknown",
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
            "historical_external_macro_reconstructed": False,
            "historical_xaut_raw_book_reconstructed": False,
            "historical_xaut_trade_tape_reconstructed": False,
            "raw_data_uploaded_as_artifact": False,
        },
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
            "Historical Ahmed Toolbox macro/news evidence is not available point-in-time for the full two-year window, so it is not fabricated.",
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
        "Historical external macro/news and raw XAUT microstructure were not fabricated. "
        "This benchmark tests the reconstructable real-data price/HTF/structure/volume/spread stack.",
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
    main()
