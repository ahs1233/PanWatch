"""Cross-provider consistency audit for XAUUSD historical bars.

This module compares providers; it never stitches them together.  A low
deviation can support confidence that both feeds describe the same market, but
it does not make their bars fungible and it does not erase provenance.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable

from src.platform.marketdata.xau_models import XAUBar


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


@dataclass(frozen=True)
class XAUProviderOverlapAudit:
    left_source: str
    right_source: str
    timeframe: str
    left_bar_count: int
    right_bar_count: int
    overlap_count: int
    overlap_fraction_left: float
    overlap_fraction_right: float
    median_close_deviation_bps: float | None
    p95_close_deviation_bps: float | None
    max_close_deviation_bps: float | None
    median_return_deviation_bps: float | None
    p95_return_deviation_bps: float | None
    same_instrument: bool
    same_timeframe: bool
    auto_merge_allowed: bool = False

    def to_dict(self) -> dict:
        return {
            "left_source": self.left_source,
            "right_source": self.right_source,
            "timeframe": self.timeframe,
            "left_bar_count": self.left_bar_count,
            "right_bar_count": self.right_bar_count,
            "overlap_count": self.overlap_count,
            "overlap_fraction_left": self.overlap_fraction_left,
            "overlap_fraction_right": self.overlap_fraction_right,
            "median_close_deviation_bps": self.median_close_deviation_bps,
            "p95_close_deviation_bps": self.p95_close_deviation_bps,
            "max_close_deviation_bps": self.max_close_deviation_bps,
            "median_return_deviation_bps": self.median_return_deviation_bps,
            "p95_return_deviation_bps": self.p95_return_deviation_bps,
            "same_instrument": self.same_instrument,
            "same_timeframe": self.same_timeframe,
            "auto_merge_allowed": self.auto_merge_allowed,
        }


def audit_xau_provider_overlap(
    left: Iterable[XAUBar],
    right: Iterable[XAUBar],
) -> XAUProviderOverlapAudit:
    left_rows = sorted(left, key=lambda row: row.timestamp)
    right_rows = sorted(right, key=lambda row: row.timestamp)
    if not left_rows or not right_rows:
        raise ValueError("both provider bar sets are required")

    left_symbols = {row.symbol for row in left_rows}
    right_symbols = {row.symbol for row in right_rows}
    left_tfs = {row.timeframe for row in left_rows}
    right_tfs = {row.timeframe for row in right_rows}

    if len(left_tfs) != 1 or len(right_tfs) != 1:
        raise ValueError("provider audit requires one timeframe per side")

    same_instrument = left_symbols == right_symbols == {"XAUUSD"}
    same_timeframe = left_tfs == right_tfs
    if not same_timeframe:
        raise ValueError("provider audit timeframes must match")

    timeframe = next(iter(left_tfs))
    left_map = {row.timestamp: row for row in left_rows}
    right_map = {row.timestamp: row for row in right_rows}
    timestamps = sorted(set(left_map).intersection(right_map))

    close_deviation: list[float] = []
    return_deviation: list[float] = []
    previous_left: float | None = None
    previous_right: float | None = None

    for timestamp in timestamps:
        l_close = float(left_map[timestamp].close)
        r_close = float(right_map[timestamp].close)
        midpoint = (l_close + r_close) / 2.0
        if midpoint > 0:
            close_deviation.append(abs(l_close - r_close) / midpoint * 10_000.0)

        if previous_left is not None and previous_right is not None:
            left_ret = (l_close / previous_left - 1.0) * 10_000.0
            right_ret = (r_close / previous_right - 1.0) * 10_000.0
            return_deviation.append(abs(left_ret - right_ret))

        previous_left = l_close
        previous_right = r_close

    left_source = ",".join(sorted({row.source for row in left_rows}))
    right_source = ",".join(sorted({row.source for row in right_rows}))
    return XAUProviderOverlapAudit(
        left_source=left_source,
        right_source=right_source,
        timeframe=timeframe.value,
        left_bar_count=len(left_rows),
        right_bar_count=len(right_rows),
        overlap_count=len(timestamps),
        overlap_fraction_left=len(timestamps) / len(left_rows),
        overlap_fraction_right=len(timestamps) / len(right_rows),
        median_close_deviation_bps=median(close_deviation) if close_deviation else None,
        p95_close_deviation_bps=_percentile(close_deviation, 0.95),
        max_close_deviation_bps=max(close_deviation) if close_deviation else None,
        median_return_deviation_bps=median(return_deviation) if return_deviation else None,
        p95_return_deviation_bps=_percentile(return_deviation, 0.95),
        same_instrument=same_instrument,
        same_timeframe=same_timeframe,
        auto_merge_allowed=False,
    )
