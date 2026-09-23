from __future__ import annotations

import gzip
import pickle
from datetime import datetime, timedelta, timezone

from benchmarks.gen1_gold_2y_real_v1 import run as bench
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


UTC = timezone.utc


def _bars(timeframe: XAUTimeframe, count: int, end: datetime) -> list[XAUBar]:
    minutes = {
        XAUTimeframe.M1: 1,
        XAUTimeframe.M5: 5,
        XAUTimeframe.M15: 15,
    }[timeframe]
    start = end - timedelta(minutes=minutes * count)
    rows = []
    price = 2600.0
    for idx in range(count):
        ts = start + timedelta(minutes=minutes * idx)
        drift = idx * 0.17
        close = price + drift + (0.35 if idx % 3 else -0.15)
        rows.append(
            XAUBar(
                timestamp=ts,
                timeframe=timeframe,
                open=close - 0.08,
                high=close + 0.35,
                low=close - 0.40,
                close=close,
                volume=100.0 + idx,
                source="test",
                symbol="XAUUSD",
                execution_eligible=False,
            )
        )
    return rows


def _naive_range_outcomes(m1, available, evaluation_time, entry, horizon_minutes):
    from bisect import bisect_right

    start = bisect_right(available, evaluation_time)
    end = bisect_right(
        available,
        evaluation_time + timedelta(minutes=horizon_minutes),
    )
    future = m1[start:end]
    result = {
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


def test_intraday_assume_sorted_fast_path_is_semantically_identical():
    now = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
    data = {
        XAUTimeframe.M1: _bars(XAUTimeframe.M1, 100, now),
        XAUTimeframe.M5: _bars(XAUTimeframe.M5, 100, now),
        XAUTimeframe.M15: _bars(XAUTimeframe.M15, 100, now),
    }
    engine = XAUIntradayEngine(require_execution_data=False)
    baseline = engine.analyze(data, now=now, assume_sorted=False)
    optimized = engine.analyze(data, now=now, assume_sorted=True)
    assert optimized == baseline


def test_range_outcomes_single_pass_matches_previous_algorithm():
    start = datetime(2026, 9, 23, 8, 0, tzinfo=UTC)
    m1 = []
    for idx in range(320):
        ts = start + timedelta(minutes=idx)
        close = 2600.0 + idx * 0.12
        m1.append(
            XAUBar(
                timestamp=ts,
                timeframe=XAUTimeframe.M1,
                open=close - 0.05,
                high=close + (12.0 if idx == 155 else 0.4),
                low=close - (21.0 if idx == 180 else 0.4),
                close=close,
                volume=50.0,
                source="test",
                symbol="XAUUSD",
                execution_eligible=False,
            )
        )
    available = [bar.timestamp + timedelta(minutes=1) for bar in m1]
    evaluation_time = start + timedelta(minutes=120)
    entry = m1[119].close
    for horizon in (60, 240):
        assert bench.range_outcomes(
            m1,
            available,
            evaluation_time,
            entry,
            horizon,
        ) == _naive_range_outcomes(
            m1,
            available,
            evaluation_time,
            entry,
            horizon,
        )


def test_dataset_shards_round_trip_without_missing_or_duplicate_days(tmp_path):
    start = datetime(2026, 9, 20, tzinfo=UTC)
    split = datetime(2026, 9, 21, tzinfo=UTC)
    end = datetime(2026, 9, 22, tzinfo=UTC)

    def day_result(day, base):
        ts = datetime(day.year, day.month, day.day, 0, 0, tzinfo=UTC)
        return {
            "day": day.isoformat(),
            "status": "data",
            "rows": [
                (ts, base, base + 1.0, base - 1.0, base + 0.2, 10.0, base, base + 0.4),
            ],
            "bid_rows": 1,
            "ask_rows": 1,
            "matched_rows": 1,
            "crossed_rows": 0,
        }

    payloads = [
        ("q1.pkl.gz", start, split, [day_result(start.date(), 2600.0)]),
        ("q2.pkl.gz", split, end, [day_result(split.date(), 2610.0)]),
    ]
    for name, shard_start, shard_end, results in payloads:
        with gzip.open(tmp_path / name, "wb") as fh:
            pickle.dump(
                {
                    "format": "gen1_gold_2y_day_results_v1",
                    "start": shard_start.isoformat(),
                    "end": shard_end.isoformat(),
                    "results": results,
                },
                fh,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    bars, quotes, diag = bench.load_dataset_shards(
        str(tmp_path / "*.pkl.gz"),
        start,
        end,
    )
    assert len(bars) == 2
    assert len(quotes) == 2
    assert diag["requested_days"] == 2
    assert diag["data_days"] == 2
    assert diag["m1_bar_count"] == 2
