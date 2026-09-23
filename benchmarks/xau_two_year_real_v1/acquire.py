"""Acquire one month of real Dukascopy XAUUSD ticks and emit M1 bars.

This uses the repository's already-proven public hourly BI5 endpoint.  The
official Dukascopy archive also exposes a requester-pays S3 export, but CI must
not assume private AWS credentials.  Hourly acquisition is parallelized in a
bounded pool and the output keeps Dukascopy provenance.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.platform.marketdata.xau_dukascopy import DukascopyXAUHistoryProvider

UTC = timezone.utc


def parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def month_bounds(month: str) -> tuple[datetime, datetime]:
    year, mon = [int(part) for part in month.split("-", 1)]
    start = datetime(year, mon, 1, tzinfo=UTC)
    end = datetime(year + (mon == 12), 1 if mon == 12 else mon + 1, 1, tzinfo=UTC)
    return start, end


def iter_hours(start: datetime, end: datetime):
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor < end:
        yield cursor
        cursor += timedelta(hours=1)


def aggregate_hour(fetch, start: datetime, end: datetime):
    if fetch.status != "data":
        return [], fetch.status
    buckets = {}
    for tick in fetch.ticks:
        ts = tick.timestamp.astimezone(UTC)
        if not (start <= ts < end):
            continue
        minute = ts.replace(second=0, microsecond=0)
        row = buckets.get(minute)
        mid = tick.mid
        volume = tick.quoted_volume
        if row is None:
            buckets[minute] = [mid, mid, mid, mid, volume]
        else:
            row[1] = max(row[1], mid)
            row[2] = min(row[2], mid)
            row[3] = mid
            row[4] += volume
    return [(ts, *buckets[ts]) for ts in sorted(buckets)], fetch.status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", required=True)
    parser.add_argument("--global-start", required=True)
    parser.add_argument("--global-end", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    month_start, month_end = month_bounds(args.month)
    global_start = parse_iso(args.global_start)
    global_end = parse_iso(args.global_end)
    start = max(month_start, global_start)
    end = min(month_end, global_end)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "month": args.month,
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "provider": "dukascopy:datafeed:XAUUSD:mid",
        "price_basis": "mid",
        "volume_semantics": "quoted_bid_plus_ask_activity_proxy",
        "execution_eligible": False,
        "attempted_hours": 0,
        "data_hours": 0,
        "empty_hours": 0,
        "not_found_hours": 0,
        "failed_hours": 0,
        "m1_bars": 0,
        "first_bar": None,
        "last_bar": None,
        "passed": False,
    }

    if end <= start:
        result["passed"] = True
        (output_dir / f"diagnostics_{args.month}.json").write_text(json.dumps(result, indent=2))
        return

    provider = DukascopyXAUHistoryProvider(
        timeout_seconds=20.0,
        retries=5,
        retry_backoff_seconds=1.0,
    )
    hours = list(iter_hours(start, end))
    result["attempted_hours"] = len(hours)
    rows = []

    with ThreadPoolExecutor(max_workers=max(1, min(8, args.workers))) as pool:
        futures = {pool.submit(provider.fetch_hour, hour): hour for hour in hours}
        for future in as_completed(futures):
            fetch = future.result()
            bars, status = aggregate_hour(fetch, start, end)
            if status == "data":
                result["data_hours"] += 1
                rows.extend(bars)
            elif status == "empty":
                result["empty_hours"] += 1
            elif status == "notfound":
                result["not_found_hours"] += 1
            else:
                result["failed_hours"] += 1

    rows.sort(key=lambda item: item[0])
    dedup = {}
    for row in rows:
        dedup[row[0]] = row
    rows = [dedup[key] for key in sorted(dedup)]

    out_csv = output_dir / f"xauusd_m1_{args.month}.csv.gz"
    with gzip.open(out_csv, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "source"])
        for ts, open_, high, low, close, volume in rows:
            writer.writerow([
                ts.isoformat(),
                f"{open_:.6f}",
                f"{high:.6f}",
                f"{low:.6f}",
                f"{close:.6f}",
                f"{volume:.6f}",
                provider.source,
            ])

    result["m1_bars"] = len(rows)
    result["first_bar"] = rows[0][0].isoformat() if rows else None
    result["last_bar"] = rows[-1][0].isoformat() if rows else None
    result["passed"] = bool(rows and result["failed_hours"] == 0)
    (output_dir / f"diagnostics_{args.month}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
