"""Top-down XAU market structure, volume-profile and liquidity analytics.

All flow metrics in this module are research proxies. Biquote MT5 volume is tick
volume, not centralized spot volume. PanWatch therefore exposes provenance and
never labels these metrics as broker/exchange execution flow.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from math import isfinite
from statistics import fmean
from typing import Any, Iterable

from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _ema_series(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("EMA period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    seed = fmean(values[:period])
    value = seed
    out[period - 1] = value
    multiplier = 2.0 / (period + 1.0)
    for i in range(period, len(values)):
        value = (values[i] - value) * multiplier + value
        out[i] = value
    return out


def ema_stack(rows: list[XAUBar], periods: tuple[int, ...] = (9, 21, 50, 200, 1000)) -> dict[str, Any]:
    closes = [float(row.close) for row in rows]
    values: dict[str, float | None] = {}
    series: dict[str, list[float | None]] = {}
    for period in periods:
        seq = _ema_series(closes, period)
        series[str(period)] = seq
        value = seq[-1] if seq else None
        values[str(period)] = round(value, 6) if value is not None else None
    return {"values": values, "series": series, "bars": len(rows)}


def _slope(values: list[float | None], lookback: int = 5) -> float | None:
    available = [(i, value) for i, value in enumerate(values) if value is not None]
    if len(available) < lookback + 1:
        return None
    end_i, end = available[-1]
    start_i, start = available[-(lookback + 1)]
    if start == 0 or end_i == start_i:
        return None
    return ((float(end) - float(start)) / abs(float(start))) * 100.0


def timeframe_structure(rows: list[XAUBar], label: str) -> dict[str, Any]:
    if not rows:
        return {
            "timeframe": label,
            "status": "unavailable",
            "bias": "neutral",
            "score": 0.0,
            "confidence": 0.0,
            "bar_count": 0,
        }
    rows = sorted(rows, key=lambda item: item.timestamp)
    close = float(rows[-1].close)
    stack = ema_stack(rows)
    emas = stack["values"]
    score = 0.0
    weight = 0.0
    evidence: list[str] = []

    for period, w in ((50, 0.28), (200, 0.34), (1000, 0.18)):
        ema = emas.get(str(period))
        if ema is None:
            continue
        weight += w
        if close > ema:
            score += w
            evidence.append(f"close_above_ema{period}")
        elif close < ema:
            score -= w
            evidence.append(f"close_below_ema{period}")

    e50, e200, e1000 = (emas.get("50"), emas.get("200"), emas.get("1000"))
    if e50 is not None and e200 is not None:
        weight += 0.12
        score += 0.12 if e50 > e200 else -0.12 if e50 < e200 else 0.0
        evidence.append("ema50_above_ema200" if e50 > e200 else "ema50_below_ema200")
    if e200 is not None and e1000 is not None:
        weight += 0.08
        score += 0.08 if e200 > e1000 else -0.08 if e200 < e1000 else 0.0
        evidence.append("ema200_above_ema1000" if e200 > e1000 else "ema200_below_ema1000")

    slope50 = _slope(stack["series"]["50"], 5)
    slope200 = _slope(stack["series"]["200"], 5)
    slope_score = 0.0
    slope_weight = 0.0
    for slope, w in ((slope50, 0.06), (slope200, 0.04)):
        if slope is None:
            continue
        slope_weight += w
        slope_score += w if slope > 0 else -w if slope < 0 else 0.0
    score += slope_score
    weight += slope_weight

    normalized = score / weight if weight else 0.0
    normalized = max(-1.0, min(1.0, normalized))
    bias = "bullish" if normalized >= 0.20 else "bearish" if normalized <= -0.20 else "neutral"
    confidence = min(1.0, weight) * min(1.0, len(rows) / 200.0)

    return {
        "timeframe": label,
        "status": "ready",
        "close": round(close, 6),
        "bias": bias,
        "score": round(normalized, 4),
        "confidence": round(confidence, 4),
        "bar_count": len(rows),
        "ema": emas,
        "ema50_slope_pct": round(slope50, 5) if slope50 is not None else None,
        "ema200_slope_pct": round(slope200, 5) if slope200 is not None else None,
        "evidence": evidence,
        "observed_at": rows[-1].timestamp.isoformat(),
        "source": rows[-1].source,
    }


def aggregate_bars(rows: list[XAUBar], timeframe: XAUTimeframe) -> list[XAUBar]:
    """Aggregate daily bars into weekly or monthly research bars."""
    if timeframe not in {XAUTimeframe.W1, XAUTimeframe.MN1}:
        raise ValueError("aggregate_bars supports W1 and MN1 only")
    buckets: dict[tuple[int, int], list[XAUBar]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: item.timestamp):
        dt = row.timestamp.astimezone(timezone.utc)
        if timeframe == XAUTimeframe.W1:
            iso = dt.isocalendar()
            key = (iso.year, iso.week)
        else:
            key = (dt.year, dt.month)
        buckets[key].append(row)

    out: list[XAUBar] = []
    for items in buckets.values():
        first, last = items[0], items[-1]
        volume_values = [float(item.volume) for item in items if item.volume is not None]
        out.append(
            XAUBar(
                timestamp=first.timestamp,
                timeframe=timeframe,
                open=float(first.open),
                high=max(float(item.high) for item in items),
                low=min(float(item.low) for item in items),
                close=float(last.close),
                volume=sum(volume_values) if volume_values else None,
                source=f"{last.source}:aggregated-{timeframe.value}",
                symbol=last.symbol,
                execution_eligible=False,
            )
        )
    out.sort(key=lambda item: item.timestamp)
    return out


def top_down_bias(
    *,
    h1: list[XAUBar],
    h4: list[XAUBar],
    daily: list[XAUBar],
    weekly: list[XAUBar],
    monthly: list[XAUBar],
) -> dict[str, Any]:
    contexts = {
        "1h": timeframe_structure(h1, "1h"),
        "4h": timeframe_structure(h4, "4h"),
        "1d": timeframe_structure(daily, "1d"),
        "1w": timeframe_structure(weekly, "1w"),
        "1mo": timeframe_structure(monthly, "1mo"),
    }
    weights = {"1h": 0.08, "4h": 0.17, "1d": 0.32, "1w": 0.27, "1mo": 0.16}
    weighted = 0.0
    active = 0.0
    for name, ctx in contexts.items():
        conf = float(ctx.get("confidence") or 0.0)
        if ctx.get("status") != "ready" or conf <= 0:
            continue
        effective = weights[name] * (0.35 + 0.65 * conf)
        weighted += effective * float(ctx.get("score") or 0.0)
        active += effective
    score = weighted / active if active else 0.0
    score = max(-1.0, min(1.0, score))
    bias = "bullish" if score >= 0.18 else "bearish" if score <= -0.18 else "neutral"

    today_parts = [contexts["1d"], contexts["4h"], contexts["1h"]]
    today_weight = {"1d": 0.55, "4h": 0.30, "1h": 0.15}
    today_num = 0.0
    today_den = 0.0
    for ctx in today_parts:
        name = str(ctx.get("timeframe"))
        conf = float(ctx.get("confidence") or 0.0)
        if ctx.get("status") != "ready" or conf <= 0:
            continue
        w = today_weight[name] * (0.4 + 0.6 * conf)
        today_num += w * float(ctx.get("score") or 0.0)
        today_den += w
    today_score = today_num / today_den if today_den else score
    today_bias = "bullish" if today_score >= 0.18 else "bearish" if today_score <= -0.18 else "neutral"

    return {
        "bias": bias,
        "score": round(score, 4),
        "today_bias": today_bias,
        "today_score": round(today_score, 4),
        "coverage": round(active / sum(weights.values()), 4),
        "frames": contexts,
        "method": "weighted_ema_structure_v1",
    }


def volume_profile(rows: list[XAUBar], bins: int = 48, value_area: float = 0.70) -> dict[str, Any]:
    rows = [row for row in rows if row.volume is not None and float(row.volume or 0.0) > 0]
    if len(rows) < 20:
        return {"status": "unavailable", "reason": "insufficient_volume_bars", "bar_count": len(rows)}
    low = min(float(row.low) for row in rows)
    high = max(float(row.high) for row in rows)
    if high <= low:
        return {"status": "unavailable", "reason": "flat_price_range", "bar_count": len(rows)}

    bins = max(16, min(int(bins), 96))
    step = (high - low) / bins
    volumes = [0.0] * bins
    for row in rows:
        volume = float(row.volume or 0.0)
        start = max(0, min(bins - 1, int((float(row.low) - low) / step)))
        end = max(0, min(bins - 1, int((float(row.high) - low) / step)))
        count = max(1, end - start + 1)
        allocation = volume / count
        for idx in range(start, end + 1):
            volumes[idx] += allocation

    total = sum(volumes)
    if total <= 0:
        return {"status": "unavailable", "reason": "zero_volume", "bar_count": len(rows)}
    poc_idx = max(range(bins), key=lambda idx: volumes[idx])
    included = {poc_idx}
    accumulated = volumes[poc_idx]
    left = poc_idx - 1
    right = poc_idx + 1
    target = total * max(0.50, min(value_area, 0.90))
    while accumulated < target and (left >= 0 or right < bins):
        lv = volumes[left] if left >= 0 else -1.0
        rv = volumes[right] if right < bins else -1.0
        if rv > lv:
            included.add(right)
            accumulated += max(0.0, rv)
            right += 1
        else:
            included.add(left)
            accumulated += max(0.0, lv)
            left -= 1

    def center(idx: int) -> float:
        return low + (idx + 0.5) * step

    populated = [idx for idx, value in enumerate(volumes) if value > 0]
    hvn = sorted(populated, key=lambda idx: volumes[idx], reverse=True)[:5]
    lvn = sorted(populated, key=lambda idx: volumes[idx])[:5]
    latest = float(rows[-1].close)
    vah = center(max(included))
    val = center(min(included))
    location = "above_value" if latest > vah else "below_value" if latest < val else "inside_value"

    return {
        "status": "ready",
        "poc": round(center(poc_idx), 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "location": location,
        "total_volume": round(total, 2),
        "bar_count": len(rows),
        "volume_kind": "mt5_tick_volume_proxy",
        "value_area_fraction": round(value_area, 3),
        "high_volume_nodes": [round(center(idx), 4) for idx in hvn],
        "low_volume_nodes": [round(center(idx), 4) for idx in lvn],
        "bins": [
            {
                "price": round(center(idx), 4),
                "volume": round(volumes[idx], 2),
                "share": round(volumes[idx] / total, 6),
            }
            for idx in range(bins)
        ],
        "source": rows[-1].source,
        "observed_at": rows[-1].timestamp.isoformat(),
    }


def _atr(rows: list[XAUBar], period: int = 14) -> float:
    if len(rows) < 2:
        return 0.0
    ranges: list[float] = []
    for idx in range(1, len(rows)):
        row = rows[idx]
        prev = rows[idx - 1]
        ranges.append(max(row.high - row.low, abs(row.high - prev.close), abs(row.low - prev.close)))
    tail = ranges[-period:] if len(ranges) >= period else ranges
    return fmean(tail) if tail else 0.0


def _swing_points(rows: list[XAUBar], width: int = 2) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    for i in range(width, len(rows) - width):
        high = float(rows[i].high)
        low = float(rows[i].low)
        if all(high >= float(rows[j].high) for j in range(i - width, i + width + 1) if j != i):
            highs.append((i, high))
        if all(low <= float(rows[j].low) for j in range(i - width, i + width + 1) if j != i):
            lows.append((i, low))
    return highs, lows


def _cluster_levels(points: list[tuple[int, float]], tolerance: float, side: str) -> list[dict[str, Any]]:
    if not points:
        return []
    clusters: list[list[tuple[int, float]]] = []
    for point in sorted(points, key=lambda item: item[1]):
        if not clusters:
            clusters.append([point])
            continue
        center = fmean(value for _, value in clusters[-1])
        if abs(point[1] - center) <= tolerance:
            clusters[-1].append(point)
        else:
            clusters.append([point])
    out = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        out.append({
            "price": round(fmean(value for _, value in cluster), 4),
            "touches": len(cluster),
            "side": side,
            "strength": round(min(1.0, 0.30 + 0.15 * len(cluster)), 3),
        })
    return out


def liquidity_map(rows: list[XAUBar], daily: list[XAUBar] | None = None, weekly: list[XAUBar] | None = None) -> dict[str, Any]:
    rows = sorted(rows, key=lambda item: item.timestamp)
    if len(rows) < 30:
        return {"status": "unavailable", "reason": "insufficient_bars"}
    current = float(rows[-1].close)
    atr = _atr(rows)
    tolerance = max(0.15, atr * 0.18)
    highs, lows = _swing_points(rows[-180:], width=2)
    buy_side = _cluster_levels(highs, tolerance, "buy_side")
    sell_side = _cluster_levels(lows, tolerance, "sell_side")

    reference_levels: list[dict[str, Any]] = []
    if daily and len(daily) >= 2:
        prev = daily[-2]
        reference_levels.extend([
            {"name": "previous_day_high", "price": round(float(prev.high), 4), "side": "buy_side"},
            {"name": "previous_day_low", "price": round(float(prev.low), 4), "side": "sell_side"},
        ])
    if weekly and len(weekly) >= 2:
        prev = weekly[-2]
        reference_levels.extend([
            {"name": "previous_week_high", "price": round(float(prev.high), 4), "side": "buy_side"},
            {"name": "previous_week_low", "price": round(float(prev.low), 4), "side": "sell_side"},
        ])

    above = sorted(
        [item for item in buy_side + reference_levels if float(item["price"]) > current],
        key=lambda item: float(item["price"]) - current,
    )[:5]
    below = sorted(
        [item for item in sell_side + reference_levels if float(item["price"]) < current],
        key=lambda item: current - float(item["price"]),
    )[:5]

    # Wick sweep proxy: breach recent 20-bar extreme, then close back inside.
    sweeps: list[dict[str, Any]] = []
    for idx in range(max(20, len(rows) - 40), len(rows)):
        prior = rows[max(0, idx - 20):idx]
        if len(prior) < 10:
            continue
        row = rows[idx]
        prior_high = max(float(item.high) for item in prior)
        prior_low = min(float(item.low) for item in prior)
        if float(row.high) > prior_high and float(row.close) < prior_high:
            sweeps.append({"time": row.timestamp.isoformat(), "type": "buy_side_sweep", "level": round(prior_high, 4), "close": round(float(row.close), 4)})
        if float(row.low) < prior_low and float(row.close) > prior_low:
            sweeps.append({"time": row.timestamp.isoformat(), "type": "sell_side_sweep", "level": round(prior_low, 4), "close": round(float(row.close), 4)})

    return {
        "status": "ready",
        "current_price": round(current, 4),
        "atr": round(atr, 4),
        "nearest_above": above,
        "nearest_below": below,
        "equal_high_clusters": buy_side[:8],
        "equal_low_clusters": sell_side[:8],
        "reference_levels": reference_levels,
        "recent_sweeps": sweeps[-6:],
        "method": "swing_cluster_and_reference_levels_v1",
    }


def flow_proxy(rows: list[XAUBar], period: int = 20) -> dict[str, Any]:
    rows = [row for row in sorted(rows, key=lambda item: item.timestamp) if row.volume is not None and float(row.volume or 0.0) >= 0]
    if len(rows) < max(25, period):
        return {"status": "unavailable", "reason": "insufficient_volume_bars", "bar_count": len(rows)}

    window = rows[-period:]
    mfv_sum = 0.0
    vol_sum = 0.0
    signed_volume = 0.0
    obv = 0.0
    obv_series: list[float] = []
    previous_close = float(rows[0].close)
    for row in rows:
        volume = float(row.volume or 0.0)
        high, low, close, open_ = float(row.high), float(row.low), float(row.close), float(row.open)
        if row in window:
            denominator = high - low
            multiplier = ((close - low) - (high - close)) / denominator if denominator > 0 else 0.0
            mfv_sum += multiplier * volume
            vol_sum += volume
            signed_volume += volume if close > open_ else -volume if close < open_ else 0.0
        if close > previous_close:
            obv += volume
        elif close < previous_close:
            obv -= volume
        obv_series.append(obv)
        previous_close = close

    cmf = mfv_sum / vol_sum if vol_sum > 0 else 0.0
    directional_share = signed_volume / vol_sum if vol_sum > 0 else 0.0
    obv_tail = obv_series[-min(30, len(obv_series)):]
    obv_delta = obv_tail[-1] - obv_tail[0] if len(obv_tail) > 1 else 0.0
    obv_scale = sum(float(row.volume or 0.0) for row in rows[-len(obv_tail):]) or 1.0
    obv_impulse = max(-1.0, min(1.0, obv_delta / obv_scale))
    score = max(-1.0, min(1.0, 0.50 * cmf + 0.30 * directional_share + 0.20 * obv_impulse))
    bias = "accumulation" if score >= 0.12 else "distribution" if score <= -0.12 else "balanced"

    return {
        "status": "ready",
        "bias": bias,
        "score": round(score, 4),
        "cmf20": round(cmf, 4),
        "directional_volume_share": round(directional_share, 4),
        "obv_impulse": round(obv_impulse, 4),
        "volume_kind": "mt5_tick_volume_proxy",
        "interpretation": "institutional_flow_proxy_not_true_orderflow",
        "bar_count": len(rows),
        "source": rows[-1].source,
        "observed_at": rows[-1].timestamp.isoformat(),
    }


def fair_value_gaps(rows: list[XAUBar], max_results: int = 8) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda item: item.timestamp)
    out: list[dict[str, Any]] = []
    for i in range(2, len(rows)):
        a, _, c = rows[i - 2], rows[i - 1], rows[i]
        if float(c.low) > float(a.high):
            out.append({
                "type": "bullish_fvg",
                "low": round(float(a.high), 4),
                "high": round(float(c.low), 4),
                "time": c.timestamp.isoformat(),
            })
        elif float(c.high) < float(a.low):
            out.append({
                "type": "bearish_fvg",
                "low": round(float(c.high), 4),
                "high": round(float(a.low), 4),
                "time": c.timestamp.isoformat(),
            })
    return out[-max_results:]
