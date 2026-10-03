from __future__ import annotations

import gzip
import json
import pickle

import pytest
from datetime import datetime, timedelta, timezone

from benchmarks.gen1_gold_2y_real_v1 import run as bench
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.marketdata.xau_dukascopy import DukascopyTick


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


def test_download_day_results_retries_transient_day_without_aborting_pool(monkeypatch):
    start = datetime(2026, 9, 20, tzinfo=UTC)
    end = datetime(2026, 9, 23, tzinfo=UTC)
    attempts = {}

    def fake_fetch_day(day):
        attempts[day] = attempts.get(day, 0) + 1
        if day == start.date() and attempts[day] == 1:
            raise bench.FetchError("transient dukascopy failure")
        return {
            "day": day.isoformat(),
            "status": "no_data",
            "rows": [],
        }

    monkeypatch.setattr(bench, "WORKERS", 2)
    monkeypatch.setattr(bench, "fetch_day", fake_fetch_day)
    monkeypatch.setattr(bench.time, "sleep", lambda _: None)
    monkeypatch.setattr(bench.random, "uniform", lambda *_: 0.0)

    results = bench.download_day_results(start, end, progress_label="retry-test")

    assert [row["day"] for row in results] == [
        "2026-09-20",
        "2026-09-21",
        "2026-09-22",
    ]
    assert attempts[start.date()] == 2
    assert sum(attempts.values()) == 4



def test_download_day_results_reuses_existing_days_and_checkpoints_new_days(
    monkeypatch,
    tmp_path,
):
    start = datetime(2026, 9, 20, tzinfo=UTC)
    end = datetime(2026, 9, 23, tzinfo=UTC)
    existing = {
        "2026-09-20": {
            "day": "2026-09-20",
            "status": "no_data",
            "rows": [],
            "provenance": bench._provenance(
                "dukascopy_native_m1_bid_ask_candles",
                fallback=False,
            ),
        },
    }
    fetched = []

    def fake_fetch_day(day):
        fetched.append(day.isoformat())
        return {
            "day": day.isoformat(),
            "status": "no_data",
            "rows": [],
            "provenance": bench._provenance(
                "dukascopy_native_m1_bid_ask_candles",
                fallback=False,
            ),
        }

    monkeypatch.setattr(bench, "WORKERS", 1)
    monkeypatch.setattr(bench, "fetch_day", fake_fetch_day)
    results = bench.download_day_results(
        start,
        end,
        existing_results=existing,
        checkpoint_dir=tmp_path,
        progress_label="resume-test",
    )

    assert [row["day"] for row in results] == [
        "2026-09-20",
        "2026-09-21",
        "2026-09-22",
    ]
    assert fetched == ["2026-09-21", "2026-09-22"]
    assert (tmp_path / "2026-09-21.pkl.gz").exists()
    assert (tmp_path / "2026-09-22.pkl.gz").exists()


def test_tick_fallback_builds_m1_bid_ask_rows(monkeypatch):
    day = datetime(2026, 9, 20, tzinfo=UTC).date()
    base = datetime(2026, 9, 20, 0, 0, tzinfo=UTC)

    def fake_fetch_bytes(url, retries=5):
        return "data", b"payload"

    def fake_decode(payload, hour):
        return (
            DukascopyTick(
                timestamp=hour + timedelta(seconds=10),
                bid=2600.0,
                ask=2600.4,
                bid_volume=1.0,
                ask_volume=2.0,
            ),
            DukascopyTick(
                timestamp=hour + timedelta(seconds=50),
                bid=2600.2,
                ask=2600.6,
                bid_volume=2.0,
                ask_volume=3.0,
            ),
        )

    monkeypatch.setattr(bench, "fetch_bytes", fake_fetch_bytes)
    monkeypatch.setattr(bench, "decode_dukascopy_xau_bi5", fake_decode)
    result = bench.fetch_day_tick_fallback(day)

    assert result["status"] == "data"
    assert result["fallback_attempted_hours"] == 24
    assert result["provenance"]["used_fallback"] is True
    assert result["provenance"]["acquisition"] == "dukascopy_tick_derived_m1"
    assert len(result["rows"]) == 24
    first = result["rows"][0]
    assert first[0] == base
    assert first[6] == 2600.2
    assert first[7] == 2600.6


def test_save_dataset_shard_persists_partial_payload_before_raising(
    monkeypatch,
    tmp_path,
):
    start = datetime(2026, 9, 20, tzinfo=UTC)
    end = datetime(2026, 9, 22, tzinfo=UTC)
    partial = [
        {
            "day": "2026-09-20",
            "status": "no_data",
            "rows": [],
            "provenance": bench._provenance(
                "dukascopy_native_m1_bid_ask_candles",
                fallback=False,
            ),
        },
    ]

    def fail_download(*args, **kwargs):
        raise bench.IncompleteDatasetShard(
            "one day still missing",
            partial_results=partial,
            failed_days={"2026-09-21": "ConnectTimeout"},
        )

    monkeypatch.setattr(bench, "download_day_results", fail_download)
    output = tmp_path / "m01.pkl.gz"

    with pytest.raises(bench.IncompleteDatasetShard):
        bench.save_dataset_shard(start, end, output)

    assert output.exists()
    manifest_path = tmp_path / "m01.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["complete"] is False
    assert manifest["result_day_count"] == 1
    assert manifest["failed_day_count"] == 1
    with gzip.open(output, "rb") as fh:
        payload = pickle.load(fh)
    assert payload["format"] == "gen1_gold_2y_day_results_v2"
    assert payload["complete"] is False
    assert payload["results"][0]["day"] == "2026-09-20"


def test_dataset_manifest_detects_checksum_tampering(tmp_path):
    start = datetime(2026, 9, 20, tzinfo=UTC)
    end = datetime(2026, 9, 22, tzinfo=UTC)
    shard = tmp_path / "m01.pkl.gz"
    results = []
    for offset in range(2):
        day = (start + timedelta(days=offset)).date()
        ts = datetime(day.year, day.month, day.day, tzinfo=UTC)
        results.append({
            "day": day.isoformat(),
            "status": "data",
            "rows": [
                (ts, 2600.0, 2601.0, 2599.0, 2600.2, 10.0, 2600.0, 2600.4),
            ],
            "bid_rows": 1,
            "ask_rows": 1,
            "matched_rows": 1,
            "crossed_rows": 0,
            "provenance": bench._provenance(
                "dukascopy_native_m1_bid_ask_candles",
                fallback=False,
            ),
        })
    bench._write_shard_payload(
        shard,
        start=start,
        end=end,
        results=results,
        failed_days={},
    )
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest = bench.build_dataset_manifest(
        str(tmp_path / "m*.pkl.gz"),
        start,
        end,
        manifest_path,
    )
    assert manifest["complete"] is True
    assert manifest["shard_count"] == 1
    assert manifest["day_count"] == 2
    bench.validate_dataset_manifest(
        str(tmp_path / "m*.pkl.gz"),
        manifest_path,
        start,
        end,
    )

    shard.write_bytes(shard.read_bytes() + b"tamper")
    with pytest.raises(bench.FetchError, match="checksum"):
        bench.validate_dataset_manifest(
            str(tmp_path / "m*.pkl.gz"),
            manifest_path,
            start,
            end,
        )
