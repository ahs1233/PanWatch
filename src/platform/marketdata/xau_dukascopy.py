"""Dukascopy public XAUUSD historical tick provider.

Purpose
-------
Deep, research-only XAUUSD history for Strategy v2.  This provider is a
separate provenance domain from Biquote/MT5.  It must never be silently spliced
into another provider's bar stream.

Dukascopy BI5 tick files are hourly LZMA payloads containing 20-byte,
big-endian records:
    int32 ms offset, int32 ask, int32 bid, float32 ask volume, float32 bid volume

For XAUUSD, raw prices use a 1000 divider.  Bars emitted here use MID price
((bid + ask) / 2) for OHLC.  The volume field is the sum of quoted bid/ask
sizes observed in the source ticks; it is an activity/liquidity proxy, not
centralized traded gold volume or global order flow.

No live trading, broker orders, or execution eligibility is provided.
"""

from __future__ import annotations

import lzma
import struct
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx

from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


_RECORD = struct.Struct(">IIIff")
_POINT_DIVIDER = 1000.0
_DEFAULT_BASE_URL = "https://datafeed.dukascopy.com/datafeed"
_SUPPORTED = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
    XAUTimeframe.M30: timedelta(minutes=30),
    XAUTimeframe.H1: timedelta(hours=1),
    XAUTimeframe.H4: timedelta(hours=4),
    XAUTimeframe.D1: timedelta(days=1),
}


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _hour_floor(value: datetime) -> datetime:
    value = _utc(value)
    return value.replace(minute=0, second=0, microsecond=0)


def _iter_hours(start: datetime, end: datetime) -> Iterable[datetime]:
    cursor = _hour_floor(start)
    stop = _utc(end)
    while cursor < stop:
        yield cursor
        cursor += timedelta(hours=1)


def _bucket_start(value: datetime, timeframe: XAUTimeframe) -> datetime:
    value = _utc(value)
    if timeframe is XAUTimeframe.M1:
        return value.replace(second=0, microsecond=0)
    if timeframe is XAUTimeframe.M5:
        return value.replace(minute=(value.minute // 5) * 5, second=0, microsecond=0)
    if timeframe is XAUTimeframe.M15:
        return value.replace(minute=(value.minute // 15) * 15, second=0, microsecond=0)
    if timeframe is XAUTimeframe.M30:
        return value.replace(minute=(value.minute // 30) * 30, second=0, microsecond=0)
    if timeframe is XAUTimeframe.H1:
        return value.replace(minute=0, second=0, microsecond=0)
    if timeframe is XAUTimeframe.H4:
        return value.replace(
            hour=(value.hour // 4) * 4,
            minute=0,
            second=0,
            microsecond=0,
        )
    if timeframe is XAUTimeframe.D1:
        return value.replace(hour=0, minute=0, second=0, microsecond=0)
    raise ValueError(f"unsupported Dukascopy XAU timeframe: {timeframe.value}")


@dataclass(frozen=True)
class DukascopyTick:
    timestamp: datetime
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float

    def __post_init__(self) -> None:
        if self.ask <= 0 or self.bid <= 0:
            raise ValueError("Dukascopy tick prices must be positive")
        if self.ask < self.bid:
            raise ValueError("Dukascopy ask cannot be below bid")
        if self.timestamp.tzinfo is None:
            object.__setattr__(
                self,
                "timestamp",
                self.timestamp.replace(tzinfo=timezone.utc),
            )

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def quoted_volume(self) -> float:
        return max(0.0, self.bid_volume) + max(0.0, self.ask_volume)


@dataclass(frozen=True)
class DukascopyHourFetch:
    hour: datetime
    status: str
    ticks: tuple[DukascopyTick, ...]
    url: str
    attempts: int


@dataclass(frozen=True)
class DukascopyRangeDiagnostics:
    requested_start: datetime
    requested_end: datetime
    attempted_hours: int
    data_hours: int
    empty_hours: int
    not_found_hours: int
    failed_hours: int
    tick_count: int
    source: str = "dukascopy:datafeed:XAUUSD:mid"
    symbol: str = "XAUUSD"
    price_basis: str = "mid"
    volume_semantics: str = "quoted_bid_plus_ask_activity_proxy"
    centralized_order_flow: bool = False


def dukascopy_hour_url(
    hour: datetime,
    *,
    base_url: str = _DEFAULT_BASE_URL,
) -> str:
    hour = _hour_floor(hour)
    month_zero_indexed = hour.month - 1
    return (
        f"{base_url.rstrip('/')}/XAUUSD/{hour.year}/"
        f"{month_zero_indexed:02d}/{hour.day:02d}/"
        f"{hour.hour:02d}h_ticks.bi5"
    )


def decode_dukascopy_xau_bi5(
    payload: bytes,
    hour: datetime,
) -> tuple[DukascopyTick, ...]:
    """Decode one raw hourly XAUUSD BI5 payload without third-party numerics."""

    if not payload:
        return ()
    try:
        raw = lzma.decompress(payload)
    except lzma.LZMAError as exc:
        raise ValueError("invalid Dukascopy BI5 LZMA payload") from exc

    if len(raw) % _RECORD.size:
        raise ValueError(
            f"Dukascopy BI5 payload length {len(raw)} is not a multiple "
            f"of {_RECORD.size}"
        )

    hour = _hour_floor(hour)
    ticks: list[DukascopyTick] = []
    last_ms = -1
    for offset in range(0, len(raw), _RECORD.size):
        ms, ask_raw, bid_raw, ask_volume, bid_volume = _RECORD.unpack_from(raw, offset)
        if not 0 <= ms < 3_600_000:
            raise ValueError(f"invalid Dukascopy millisecond offset: {ms}")
        if ms < last_ms:
            raise ValueError("Dukascopy ticks are not chronological within hour")
        last_ms = ms

        ask = float(ask_raw) / _POINT_DIVIDER
        bid = float(bid_raw) / _POINT_DIVIDER
        # Zero-volume ticks are not useful for our activity proxy and mirror the
        # behavior of common Dukascopy research pipelines.
        if float(ask_volume) == 0.0 and float(bid_volume) == 0.0:
            continue
        ticks.append(
            DukascopyTick(
                timestamp=hour + timedelta(milliseconds=int(ms)),
                bid=bid,
                ask=ask,
                bid_volume=float(bid_volume),
                ask_volume=float(ask_volume),
            )
        )
    return tuple(ticks)


class DukascopyXAUHistoryProvider:
    symbol = "XAUUSD"
    source = "dukascopy:datafeed:XAUUSD:mid"
    execution_eligible = False
    price_basis = "mid"
    volume_semantics = "quoted_bid_plus_ask_activity_proxy"
    centralized_order_flow = False

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_seconds: float = 20.0,
        retries: int = 2,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self.retries = max(0, int(retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def fetch_hour(self, hour: datetime) -> DukascopyHourFetch:
        hour = _hour_floor(hour)
        url = dukascopy_hour_url(hour, base_url=self.base_url)
        attempts = self.retries + 1
        last_error: Exception | None = None

        last_status: int | None = None
        for attempt in range(1, attempts + 1):
            retry_after_seconds: float | None = None
            try:
                response = httpx.get(
                    url,
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                    headers={"User-Agent": "PanWatch-XAU-Research/0.7"},
                )
                last_status = int(response.status_code)
                if response.status_code == 404:
                    return DukascopyHourFetch(hour, "notfound", (), url, attempt)
                if response.status_code in {429} or response.status_code >= 500:
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            retry_after_seconds = max(0.0, float(retry_after))
                        except ValueError:
                            retry_after_seconds = None
                    response.raise_for_status()
                if 400 <= response.status_code < 500:
                    return DukascopyHourFetch(
                        hour,
                        f"failed:http_{response.status_code}",
                        (),
                        url,
                        attempt,
                    )
                response.raise_for_status()
                if not response.content:
                    return DukascopyHourFetch(hour, "empty", (), url, attempt)
                ticks = decode_dukascopy_xau_bi5(response.content, hour)
                return DukascopyHourFetch(
                    hour,
                    "data" if ticks else "empty",
                    ticks,
                    url,
                    attempt,
                )
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc
                if attempt < attempts:
                    exponential = self.retry_backoff_seconds * (2 ** (attempt - 1))
                    delay = min(60.0, max(exponential, retry_after_seconds or 0.0))
                    time.sleep(delay)

        if last_status is not None:
            detail = f"http_{last_status}"
        else:
            detail = type(last_error).__name__ if last_error is not None else "unknown"
        return DukascopyHourFetch(hour, f"failed:{detail}", (), url, attempts)

    def ticks_range(
        self,
        *,
        start: datetime,
        end: datetime,
        max_hours: int = 24 * 32,
    ) -> tuple[tuple[DukascopyTick, ...], DukascopyRangeDiagnostics]:
        start_utc = _utc(start)
        end_utc = _utc(end)
        if end_utc <= start_utc:
            raise ValueError("Dukascopy range end must be after start")

        hours = list(_iter_hours(start_utc, end_utc))
        if len(hours) > max(1, int(max_hours)):
            raise ValueError(
                f"Dukascopy range requires {len(hours)} hourly files; "
                f"safety cap is {max_hours}. Use incremental acquisition."
            )

        ticks: list[DukascopyTick] = []
        counts = defaultdict(int)
        for hour in hours:
            fetched = self.fetch_hour(hour)
            key = "failed" if fetched.status.startswith("failed:") else fetched.status
            counts[key] += 1
            ticks.extend(
                tick
                for tick in fetched.ticks
                if start_utc <= _utc(tick.timestamp) < end_utc
            )

        ticks.sort(key=lambda item: _utc(item.timestamp))
        diagnostics = DukascopyRangeDiagnostics(
            requested_start=start_utc,
            requested_end=end_utc,
            attempted_hours=len(hours),
            data_hours=counts["data"],
            empty_hours=counts["empty"],
            not_found_hours=counts["notfound"],
            failed_hours=counts["failed"],
            tick_count=len(ticks),
        )
        return tuple(ticks), diagnostics

    def bars_range(
        self,
        timeframe: XAUTimeframe,
        *,
        start: datetime,
        end: datetime,
        max_hours: int = 24 * 32,
    ) -> tuple[list[XAUBar], DukascopyRangeDiagnostics]:
        if timeframe not in _SUPPORTED:
            raise ValueError(f"unsupported Dukascopy XAU timeframe: {timeframe.value}")

        ticks, diagnostics = self.ticks_range(
            start=start,
            end=end,
            max_hours=max_hours,
        )
        if diagnostics.failed_hours:
            raise RuntimeError(
                f"Dukascopy range has {diagnostics.failed_hours} failed hourly requests"
            )
        if not ticks:
            return [], diagnostics

        buckets: dict[datetime, list[DukascopyTick]] = defaultdict(list)
        for tick in ticks:
            buckets[_bucket_start(tick.timestamp, timeframe)].append(tick)

        start_utc = _utc(start)
        end_utc = _utc(end)
        bars: list[XAUBar] = []
        for timestamp in sorted(buckets):
            if timestamp < start_utc or timestamp >= end_utc:
                continue
            group = buckets[timestamp]
            mids = [tick.mid for tick in group]
            bars.append(
                XAUBar(
                    timestamp=timestamp,
                    timeframe=timeframe,
                    open=mids[0],
                    high=max(mids),
                    low=min(mids),
                    close=mids[-1],
                    volume=sum(tick.quoted_volume for tick in group),
                    source=self.source,
                    symbol="XAUUSD",
                    execution_eligible=False,
                )
            )
        return bars, diagnostics
