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
import threading
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
from src.platform.marketdata.xau_dukascopy import (
    DukascopyTick,
    decode_dukascopy_xau_bi5,
    dukascopy_hour_url,
)


UTC = timezone.utc
RECORD = struct.Struct(">IIIIIf")
DIVIDER = 1000.0
BASE_URL = "https://datafeed.dukascopy.com/datafeed"
SOURCE = "dukascopy:datafeed:XAUUSD:native-m1-bid-ask-mid"
_HTTP_LOCAL = threading.local()


def _download_client() -> httpx.Client:
    client = getattr(_HTTP_LOCAL, "client", None)
    if client is None:
        client = httpx.Client(
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
            headers={"User-Agent": "PanWatch-GEN1-2Y-Research/1.0"},
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=4,
                keepalive_expiry=60.0,
            ),
        )
        _HTTP_LOCAL.client = client
    return client

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


class IncompleteDatasetShard(FetchError):
    def __init__(
        self,
        message: str,
        *,
        partial_results: list[dict[str, Any]],
        failed_days: dict[str, str],
    ) -> None:
        super().__init__(message)
        self.partial_results = partial_results
        self.failed_days = failed_days


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
            response = _download_client().get(url)
            status_code = int(response.status_code)
            if status_code == 404:
                return "notfound", b""
            if status_code >= 400:
                last = f"http_{status_code}"
            if status_code in {403, 408, 425, 429} or status_code >= 500:
                if attempt + 1 < retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else min(8.0, 0.5 * (2 ** attempt))
                    except ValueError:
                        delay = min(8.0, 0.5 * (2 ** attempt))
                    time.sleep(max(0.2, delay) + random.uniform(0.0, 0.25))
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


def _provenance(acquisition: str, *, fallback: bool) -> dict[str, Any]:
    return {
        "vendor": "Dukascopy",
        "symbol": "XAUUSD",
        "acquisition": acquisition,
        "used_fallback": bool(fallback),
        "price_basis": "mid_from_bid_ask",
        "spread_basis": "bid_ask_close",
    }


def fetch_day(day: date) -> dict[str, Any]:
    bid_status, bid_payload = fetch_bytes(candle_url(day, "BID"))
    ask_status, ask_payload = fetch_bytes(candle_url(day, "ASK"))
    if bid_status == "notfound" and ask_status == "notfound":
        return {
            "day": day.isoformat(),
            "status": "no_data",
            "rows": [],
            "provenance": _provenance(
                "dukascopy_native_m1_bid_ask_candles",
                fallback=False,
            ),
        }
    if bid_status != "data" or ask_status != "data":
        raise FetchError(
            f"one-sided candle availability on {day}: "
            f"bid={bid_status} ask={ask_status}"
        )
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
        "provenance": _provenance(
            "dukascopy_native_m1_bid_ask_candles",
            fallback=False,
        ),
    }


def fetch_day_tick_fallback(day: date) -> dict[str, Any]:
    """Reconstruct one UTC day from Dukascopy hourly ticks.

    This is a source-path fallback, not a different market proxy: the symbol
    remains XAUUSD and BID/ASK ticks are aggregated into M1 midpoint bars while
    retaining the last observed BID/ASK spread for each minute.
    """

    start = datetime.combine(day, dtime.min, tzinfo=UTC)
    buckets: dict[datetime, list[DukascopyTick]] = defaultdict(list)
    attempted_hours = 0
    for hour_index in range(24):
        hour = start + timedelta(hours=hour_index)
        status, payload = fetch_bytes(
            dukascopy_hour_url(hour),
            retries=3,
        )
        attempted_hours += 1
        if status == "notfound":
            continue
        if status != "data":
            raise FetchError(
                f"tick fallback unavailable for {day} hour={hour_index}: "
                f"status={status}"
            )
        for tick in decode_dukascopy_xau_bi5(payload, hour):
            minute = tick.timestamp.replace(second=0, microsecond=0)
            buckets[minute].append(tick)

    if not buckets:
        return {
            "day": day.isoformat(),
            "status": "no_data",
            "rows": [],
            "fallback_attempted_hours": attempted_hours,
            "provenance": _provenance(
                "dukascopy_tick_derived_m1",
                fallback=True,
            ),
        }

    rows = []
    for minute in sorted(buckets):
        ticks = buckets[minute]
        mids = [tick.mid for tick in ticks]
        last = ticks[-1]
        rows.append(
            (
                minute,
                mids[0],
                max(mids),
                min(mids),
                mids[-1],
                sum(tick.quoted_volume for tick in ticks),
                float(last.bid),
                float(last.ask),
            )
        )
    return {
        "day": day.isoformat(),
        "status": "data",
        "rows": rows,
        "bid_rows": len(rows),
        "ask_rows": len(rows),
        "matched_rows": len(rows),
        "crossed_rows": 0,
        "fallback_attempted_hours": attempted_hours,
        "provenance": _provenance(
            "dukascopy_tick_derived_m1",
            fallback=True,
        ),
    }

def date_range(start: datetime, end: datetime) -> list[date]:
    out = []
    cursor = start.date()
    while cursor < end.date():
        out.append(cursor)
        cursor += timedelta(days=1)
    return out


def _checkpoint_file(checkpoint_dir: Path, day_text: str) -> Path:
    return checkpoint_dir / f"{day_text}.pkl.gz"


def _write_day_checkpoint(
    checkpoint_dir: Path | None,
    result: dict[str, Any],
) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = _checkpoint_file(checkpoint_dir, str(result["day"]))
    temporary = target.with_name(target.name + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(target)


def _load_day_checkpoints(
    checkpoint_dir: Path | None,
    expected_days: set[str],
) -> dict[str, dict[str, Any]]:
    if checkpoint_dir is None or not checkpoint_dir.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.glob("*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as fh:
                result = pickle.load(fh)
        except (OSError, EOFError, pickle.PickleError):
            continue
        day_text = str((result or {}).get("day") or "")
        if (
            day_text in expected_days
            and (result or {}).get("status") in {"data", "no_data"}
        ):
            out[day_text] = result
    return out


def _load_existing_shard_results(
    path: Path | None,
    expected_days: set[str],
) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    try:
        with gzip.open(path, "rb") as fh:
            payload = pickle.load(fh)
    except (OSError, EOFError, pickle.PickleError):
        return {}
    if payload.get("format") not in {
        "gen1_gold_2y_day_results_v1",
        "gen1_gold_2y_day_results_v2",
    }:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for result in payload.get("results") or []:
        day_text = str(result.get("day") or "")
        if day_text in expected_days and result.get("status") in {"data", "no_data"}:
            out[day_text] = result
    return out


def download_day_results(
    start: datetime,
    end: datetime,
    *,
    progress_label: str = "dataset",
    existing_results: dict[str, dict[str, Any]] | None = None,
    checkpoint_dir: Path | None = None,
) -> list[dict[str, Any]]:
    days = date_range(start, end)
    expected_days = {day.isoformat() for day in days}
    results_by_day: dict[str, dict[str, Any]] = {}
    for day_text, result in (existing_results or {}).items():
        if day_text in expected_days and result.get("status") in {"data", "no_data"}:
            results_by_day[day_text] = result
    results_by_day.update(
        _load_day_checkpoints(checkpoint_dir, expected_days)
    )

    pending_days = [
        day for day in days
        if day.isoformat() not in results_by_day
    ]
    failed_days: dict[date, str] = {}
    started = time.monotonic()

    if results_by_day:
        print(json.dumps({
            "phase": "dataset_resume",
            "label": progress_label,
            "reused_days": len(results_by_day),
            "remaining_days": len(pending_days),
            "requested_days": len(days),
        }), flush=True)

    def collect(day: date, future) -> None:
        try:
            result = future.result()
            results_by_day[day.isoformat()] = result
            _write_day_checkpoint(checkpoint_dir, result)
            failed_days.pop(day, None)
        except Exception as exc:
            failed_days[day] = f"{type(exc).__name__}: {exc}"

    if pending_days:
        with ThreadPoolExecutor(max_workers=max(1, WORKERS)) as pool:
            futures = {pool.submit(fetch_day, day): day for day in pending_days}
            for completed, future in enumerate(as_completed(futures), start=1):
                day = futures[future]
                collect(day, future)
                if completed % 25 == 0 or completed == len(pending_days):
                    print(json.dumps({
                        "phase": "dataset_download_progress",
                        "label": progress_label,
                        "completed_days": completed,
                        "reused_days": len(days) - len(pending_days),
                        "successful_days": len(results_by_day),
                        "failed_days_pending_retry": len(failed_days),
                        "requested_days": len(days),
                        "elapsed_seconds": round(time.monotonic() - started, 2),
                    }), flush=True)

    for retry_round in range(1, 4):
        if not failed_days:
            break
        retry_days = sorted(failed_days)
        failed_days = {}
        time.sleep(0.5 * retry_round + random.uniform(0.0, 0.5))
        retry_workers = max(1, min(2, max(1, WORKERS // 2)))
        with ThreadPoolExecutor(max_workers=retry_workers) as pool:
            futures = {pool.submit(fetch_day, day): day for day in retry_days}
            for future in as_completed(futures):
                collect(futures[future], future)
        print(json.dumps({
            "phase": "dataset_retry_round",
            "label": progress_label,
            "retry_round": retry_round,
            "retried_days": len(retry_days),
            "remaining_failed_days": len(failed_days),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }), flush=True)

    # Independent acquisition path for persistent failures: derive M1 from
    # Dukascopy hourly BID/ASK ticks instead of re-hitting the candle endpoint.
    if failed_days:
        fallback_days = sorted(failed_days)
        failed_days = {}
        fallback_workers = max(1, min(2, WORKERS))
        with ThreadPoolExecutor(max_workers=fallback_workers) as pool:
            futures = {
                pool.submit(fetch_day_tick_fallback, day): day
                for day in fallback_days
            }
            for future in as_completed(futures):
                collect(futures[future], future)
        print(json.dumps({
            "phase": "dataset_fallback_round",
            "label": progress_label,
            "fallback": "dukascopy_tick_derived_m1",
            "attempted_days": len(fallback_days),
            "remaining_failed_days": len(failed_days),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        }), flush=True)

    results = [
        results_by_day[day_text]
        for day_text in sorted(results_by_day)
        if day_text in expected_days
    ]
    if failed_days:
        failed_text = {
            day.isoformat(): failed_days[day]
            for day in sorted(failed_days)
        }
        sample = [
            {"day": day, "error": error}
            for day, error in list(failed_text.items())[:10]
        ]
        raise IncompleteDatasetShard(
            f"dataset shard has {len(failed_days)} unrecoverable days "
            f"after primary retries and tick fallback: {sample}",
            partial_results=results,
            failed_days=failed_text,
        )

    if len(results) != len(days):
        missing = sorted(expected_days.difference(results_by_day))
        raise IncompleteDatasetShard(
            f"dataset shard is missing {len(missing)} days: {missing[:10]}",
            partial_results=results,
            failed_days={day: "missing" for day in missing},
        )
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


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _shard_manifest_path(output_path: Path) -> Path:
    name = output_path.name
    if name.endswith(".pkl.gz"):
        name = name[:-7] + ".manifest.json"
    else:
        name = name + ".manifest.json"
    return output_path.with_name(name)


def _write_shard_payload(
    output_path: Path,
    *,
    start: datetime,
    end: datetime,
    results: list[dict[str, Any]],
    failed_days: dict[str, str],
) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_days = {day.isoformat() for day in date_range(start, end)}
    result_days = {
        str(result.get("day"))
        for result in results
        if result.get("status") in {"data", "no_data"}
    }
    complete = not failed_days and result_days == expected_days
    payload = {
        "format": "gen1_gold_2y_day_results_v2",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "complete": complete,
        "results": sorted(results, key=lambda item: item["day"]),
        "failed_days": dict(sorted(failed_days.items())),
        "source": SOURCE,
        "generated_at": datetime.now(tz=UTC).isoformat(),
    }
    temporary = output_path.with_name(output_path.name + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=6) as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)

    source_counts: Counter[str] = Counter()
    fallback_days = []
    for result in results:
        provenance = result.get("provenance") or {}
        acquisition = str(provenance.get("acquisition") or "legacy_unknown")
        source_counts[acquisition] += 1
        if provenance.get("used_fallback"):
            fallback_days.append(str(result["day"]))
    manifest = {
        "format": "gen1_gold_2y_shard_manifest_v1",
        "payload_format": payload["format"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "complete": complete,
        "expected_day_count": len(expected_days),
        "result_day_count": len(result_days),
        "failed_day_count": len(failed_days),
        "failed_days": dict(sorted(failed_days.items())),
        "fallback_day_count": len(fallback_days),
        "fallback_days": sorted(fallback_days),
        "source_counts": dict(sorted(source_counts.items())),
        "payload_file": output_path.name,
        "payload_sha256": _sha256_file(output_path),
        "size_bytes": output_path.stat().st_size,
    }
    manifest_path = _shard_manifest_path(output_path)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def save_dataset_shard(start: datetime, end: datetime, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_days = {day.isoformat() for day in date_range(start, end)}
    existing_path_raw = os.getenv("GEN1_2Y_EXISTING_SHARD")
    existing_path = Path(existing_path_raw) if existing_path_raw else None
    checkpoint_raw = os.getenv("GEN1_2Y_DAY_CACHE_DIR")
    checkpoint_dir = Path(checkpoint_raw) if checkpoint_raw else None

    existing = _load_existing_shard_results(existing_path, expected_days)
    existing.update(_load_day_checkpoints(checkpoint_dir, expected_days))

    error: IncompleteDatasetShard | None = None
    failed_days: dict[str, str] = {}
    try:
        results = download_day_results(
            start,
            end,
            progress_label=f"{start.date()}..{end.date()}",
            existing_results=existing,
            checkpoint_dir=checkpoint_dir,
        )
    except IncompleteDatasetShard as exc:
        results = exc.partial_results
        failed_days = exc.failed_days
        error = exc

    manifest = _write_shard_payload(
        output_path,
        start=start,
        end=end,
        results=results,
        failed_days=failed_days,
    )
    print(json.dumps({
        "phase": (
            "dataset_shard_complete"
            if manifest["complete"]
            else "dataset_shard_partial"
        ),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "day_count": manifest["result_day_count"],
        "expected_day_count": manifest["expected_day_count"],
        "failed_day_count": manifest["failed_day_count"],
        "fallback_day_count": manifest["fallback_day_count"],
        "output_path": str(output_path),
        "manifest_path": str(_shard_manifest_path(output_path)),
        "payload_sha256": manifest["payload_sha256"],
        "size_bytes": output_path.stat().st_size,
    }), flush=True)
    if error is not None:
        raise error


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
        if payload.get("format") not in {
            "gen1_gold_2y_day_results_v1",
            "gen1_gold_2y_day_results_v2",
        }:
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


def build_dataset_manifest(
    pattern: str,
    start: datetime,
    end: datetime,
    output_path: Path,
) -> dict[str, Any]:
    paths = sorted(Path(path) for path in glob.glob(pattern))
    if not paths:
        raise FetchError(f"no dataset shards matched: {pattern}")

    shard_records = []
    source_counts: Counter[str] = Counter()
    fallback_days: list[str] = []
    for path in paths:
        manifest_path = _shard_manifest_path(path)
        if not manifest_path.exists():
            raise FetchError(f"missing shard manifest: {manifest_path}")
        shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_sha = _sha256_file(path)
        if actual_sha != shard_manifest.get("payload_sha256"):
            raise FetchError(f"shard checksum mismatch: {path}")
        if not shard_manifest.get("complete"):
            raise FetchError(f"incomplete dataset shard: {path}")
        source_counts.update(shard_manifest.get("source_counts") or {})
        fallback_days.extend(shard_manifest.get("fallback_days") or [])
        shard_records.append({
            "file": path.name,
            "manifest_file": manifest_path.name,
            "sha256": actual_sha,
            "start": shard_manifest.get("start"),
            "end": shard_manifest.get("end"),
            "fallback_day_count": shard_manifest.get("fallback_day_count", 0),
        })

    _, _, diag = load_dataset_shards(pattern, start, end)
    combined = hashlib.sha256()
    for record in sorted(shard_records, key=lambda item: item["file"]):
        combined.update(
            f"{record['file']}:{record['sha256']}\n".encode("utf-8")
        )
    manifest = {
        "format": "gen1_gold_2y_dataset_manifest_v1",
        "complete": True,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "source": SOURCE,
        "shard_count": len(shard_records),
        "day_count": len(date_range(start, end)),
        "m1_bar_count": diag["m1_bar_count"],
        "dataset_sha256": diag["dataset_sha256"],
        "bundle_sha256": combined.hexdigest(),
        "source_counts": dict(sorted(source_counts.items())),
        "fallback_day_count": len(set(fallback_days)),
        "fallback_days": sorted(set(fallback_days)),
        "shards": sorted(shard_records, key=lambda item: item["file"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "phase": "dataset_manifest_complete",
        "manifest": str(output_path),
        "shard_count": manifest["shard_count"],
        "day_count": manifest["day_count"],
        "m1_bar_count": manifest["m1_bar_count"],
        "dataset_sha256": manifest["dataset_sha256"],
        "bundle_sha256": manifest["bundle_sha256"],
        "fallback_day_count": manifest["fallback_day_count"],
    }), flush=True)
    return manifest


def validate_dataset_manifest(
    pattern: str,
    manifest_path: Path,
    start: datetime,
    end: datetime,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "gen1_gold_2y_dataset_manifest_v1":
        raise FetchError("unsupported dataset manifest format")
    if not manifest.get("complete"):
        raise FetchError("dataset manifest is not complete")
    if manifest.get("start") != start.isoformat():
        raise FetchError("dataset manifest start mismatch")
    if manifest.get("end") != end.isoformat():
        raise FetchError("dataset manifest end mismatch")

    paths = {Path(path).name: Path(path) for path in glob.glob(pattern)}
    records = manifest.get("shards") or []
    if len(records) != manifest.get("shard_count"):
        raise FetchError("dataset manifest shard count mismatch")
    for record in records:
        name = str(record["file"])
        path = paths.get(name)
        if path is None:
            raise FetchError(f"manifest shard missing locally: {name}")
        if _sha256_file(path) != record.get("sha256"):
            raise FetchError(f"manifest shard checksum mismatch: {name}")
    return manifest


def download_dataset(
    start: datetime,
    end: datetime,
) -> tuple[list[XAUBar], dict[datetime, tuple[float, float]], dict[str, Any]]:
    shard_glob = os.getenv("GEN1_2Y_SHARD_GLOB")
    manifest_raw = os.getenv("GEN1_2Y_MANIFEST")
    manifest = None
    if shard_glob and manifest_raw:
        manifest = validate_dataset_manifest(
            shard_glob,
            Path(manifest_raw),
            start,
            end,
        )
    if shard_glob:
        bars, quotes, diag = load_dataset_shards(shard_glob, start, end)
        if manifest is not None:
            if diag["dataset_sha256"] != manifest.get("dataset_sha256"):
                raise FetchError("dataset content checksum mismatch")
            diag["dataset_manifest"] = str(manifest_raw)
            diag["dataset_bundle_sha256"] = manifest.get("bundle_sha256")
            diag["fallback_day_count"] = manifest.get("fallback_day_count", 0)
            diag["fallback_days"] = manifest.get("fallback_days", [])
            diag["source_counts"] = manifest.get("source_counts", {})
        return bars, quotes, diag
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



def extract_research_features(
    technical: dict[str, Any],
    cognition: dict[str, Any],
    fusion: dict[str, Any],
    evidence: dict[str, Any],
    macro: dict[str, Any],
    evaluation_time: datetime,
    candidate: str,
) -> dict[str, Any]:
    """Flatten decision-time-only features for conditional-edge research."""
    price = float((technical.get("analysis_reference") or {}).get("price") or technical.get("price") or 0.0)
    side = 1.0 if candidate == "long_setup" else -1.0
    context = technical.get("market_context") or {}
    bias = context.get("bias") or {}
    regime = cognition.get("regime") or {}
    directional = evidence.get("directional_evidence") or {}
    market_state = cognition.get("market_state") or {}
    features: dict[str, Any] = {
        "direction": "LONG" if side > 0 else "SHORT",
        "session": session_name(evaluation_time),
        "hour_utc": evaluation_time.hour,
        "day_of_week": evaluation_time.strftime("%a"),
        "month": evaluation_time.month,
        "regime": str(regime.get("label") or "unknown"),
        "regime_confidence": regime.get("confidence"),
        "cognitive_confidence": (cognition.get("confidence") or {}).get("calibrated_confidence"),
        "fusion_state": fusion.get("state"),
        "gen1_decision": evidence.get("decision"),
        "gen1_confidence": evidence.get("decision_confidence"),
        "evidence_score": evidence.get("score"),
        "evidence_coverage": evidence.get("coverage"),
        "evidence_agreement": evidence.get("agreement_ratio"),
        "evidence_long_support": directional.get("long_support"),
        "evidence_short_support": directional.get("short_support"),
        "evidence_conflict_score": directional.get("conflict_score"),
        "evidence_dominant_side": directional.get("dominant_side"),
        "macro_bias": macro.get("bias"),
        "macro_bias_label": macro.get("bias_label"),
        "macro_confidence": macro.get("confidence"),
        "macro_proxy_score": macro.get("proxy_score"),
        "htf_composite_score": bias.get("composite_score"),
        "htf_composite_direction": bias.get("composite_direction"),
        "today_score": bias.get("today_score"),
        "today_direction": bias.get("today_direction"),
    }
    frames = technical.get("frames") or {}
    for frame_name in ("1m", "5m", "15m"):
        frame = frames.get(frame_name) or {}
        atr = float(frame.get("atr14") or 0.0)
        fast = frame.get("ema_fast")
        slow = frame.get("ema_slow")
        features.update({
            f"{frame_name}_direction": frame.get("direction"),
            f"{frame_name}_rsi14": frame.get("rsi14"),
            f"{frame_name}_atr14": frame.get("atr14"),
            f"{frame_name}_atr_pct": frame.get("atr_pct"),
            f"{frame_name}_ema_fast": fast,
            f"{frame_name}_ema_slow": slow,
            f"{frame_name}_breakout": frame.get("breakout"),
            f"{frame_name}_recent_swing_high": frame.get("recent_swing_high"),
            f"{frame_name}_recent_swing_low": frame.get("recent_swing_low"),
            f"{frame_name}_ema_gap_atr": (
                (float(fast) - float(slow)) / atr
                if atr > 0 and fast is not None and slow is not None else None
            ),
            f"{frame_name}_price_fast_atr": (
                (price - float(fast)) / atr if atr > 0 and fast is not None else None
            ),
        })
    weighted_alignment = 0.0
    active_weight = 0.0
    weights = {"monthly": 0.34, "weekly": 0.30, "daily": 0.24, "h4": 0.08, "h1": 0.04}
    for name, weight in weights.items():
        state = bias.get(name) or {}
        features[f"{name}_direction"] = state.get("direction")
        features[f"{name}_score"] = state.get("score")
        features[f"{name}_slope20"] = state.get("slope_20")
        ema = state.get("ema") or {}
        close = float(state.get("close") or price or 0.0)
        for period in (9, 21, 50, 200, 1000):
            value = ema.get(str(period))
            features[f"{name}_ema{period}"] = value
            features[f"{name}_price_to_ema{period}_bps"] = (
                ((close - float(value)) / float(value)) * 10000.0
                if value not in (None, 0) else None
            )
        score = state.get("score")
        if score is not None:
            weighted_alignment += weight * float(score) * side
            active_weight += weight
    alignment = weighted_alignment / active_weight if active_weight else 0.0
    features["htf_alignment_score"] = alignment
    features["htf_alignment"] = "aligned" if alignment >= 0.15 else "conflict" if alignment <= -0.15 else "mixed"
    features["trend_strength"] = abs(float(bias.get("composite_score") or 0.0))

    flow = context.get("cash_flow") or {}
    smart = context.get("smart_money") or {}
    profile = context.get("volume_profile") or {}
    features.update({
        "cash_flow_score": flow.get("score"),
        "cash_flow_direction": flow.get("direction"),
        "smart_money_score": smart.get("score"),
        "smart_money_bias": smart.get("bias"),
        "break_of_structure": smart.get("break_of_structure"),
        "liquidity_sweep": smart.get("liquidity_sweep"),
        "displacement": smart.get("displacement"),
        "dealing_zone": (smart.get("dealing_range") or {}).get("zone"),
        "volume_profile_location": profile.get("location"),
        "volume_profile_poc": profile.get("poc"),
        "volume_profile_vah": profile.get("vah"),
        "volume_profile_val": profile.get("val"),
    })
    for key in ("directional_edge", "intraday_ema_structure", "long_ema_structure", "rsi_impulse"):
        if key in market_state:
            features[f"cognition_{key}"] = market_state.get(key)
    for driver in macro.get("drivers") or []:
        name = str(driver.get("name") or "").replace("-", "_")
        if name:
            features[f"macro_{name}_value"] = driver.get("value")
            features[f"macro_{name}_gold_score"] = driver.get("gold_score")
    return features


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
        research_features = extract_research_features(
            technical,
            cognition,
            fusion,
            evidence,
            macro,
            evaluation_time,
            candidate,
        )
        shared_meta = {
            "research_only": True,
            "research_features": research_features,
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



def write_research_features_csv(path: Path, episodes: list[ReplayEpisode]) -> None:
    """Persist flattened point-in-time features without changing replay semantics."""
    feature_keys = sorted({
        key
        for ep in episodes
        for key in ((ep.meta or {}).get("research_features") or {}).keys()
    })
    fixed = [
        "observed_at", "outcome_at", "horizon_minutes", "candidate",
        "net_spread_directional_bps", "gross_directional_bps",
        "pm10_first", "pm20_first", "pm30_first",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fixed + feature_keys)
        writer.writeheader()
        for ep in episodes:
            row = {
                "observed_at": ep.observed_at.isoformat(),
                "outcome_at": ep.outcome_at.isoformat(),
                "horizon_minutes": ep.horizon_minutes,
                "candidate": ep.candidate,
                "net_spread_directional_bps": (ep.meta or {}).get("net_spread_directional_return_bps"),
                "gross_directional_bps": ep.directional_return_bps,
            }
            side = "up" if ep.candidate == "long_setup" else "down"
            adverse = "down" if side == "up" else "up"
            levels = (((ep.meta or {}).get("range_outcomes") or {}).get("levels") or {})
            for distance in (10, 20, 30):
                first = str((levels.get(f"pm{distance}") or {}).get("first_hit") or "none")
                row[f"pm{distance}_first"] = (
                    "favorable" if first == side else
                    "adverse" if first == adverse else
                    "unresolved"
                )
            row.update((ep.meta or {}).get("research_features") or {})
            writer.writerow(row)


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
    write_research_features_csv(OUT / "episodes_240m_features.csv", episodes240)
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
    if os.getenv("GEN1_2Y_BUILD_MANIFEST") == "1":
        manifest_out = Path(
            os.getenv(
                "GEN1_2Y_MANIFEST_OUT",
                "artifacts/gen1_gold_2y_dataset_manifest.json",
            )
        )
        build_dataset_manifest(
            os.environ["GEN1_2Y_SHARD_GLOB"],
            START,
            END,
            manifest_out,
        )
        sys.exit(0)
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
