"""Causal features and volatility-aware directional labels for GTG v2.

This module contains no neural-network code.  Keeping market semantics separate
from model architecture prevents label drift when models are replaced.

Labels:
- UP / DOWN / NEUTRAL via symmetric ATR-scaled triple barriers.
- same-bar two-sided first touches are marked ambiguous and excluded.
- path targets describe favorable/adverse excursion and trend strength.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import math
import numpy as np


DIRECTION_CLASSES = ("down", "neutral", "up")
CLASS_DOWN = 0
CLASS_NEUTRAL = 1
CLASS_UP = 2

HORIZON_BARS = {
    30: 6,
    60: 12,
    120: 24,
    240: 48,
}

# Wider horizons receive wider barriers so "neutral" remains meaningful and
# labels do not collapse into almost-always-active events.
BARRIER_ATR = {
    30: 0.75,
    60: 1.00,
    120: 1.40,
    240: 2.00,
}

FEATURE_NAMES = (
    "logret_1",
    "logret_3",
    "logret_12",
    "logret_24",
    "body_atr",
    "range_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "close_location",
    "volume_z48",
    "rsi14_centered",
    "dist_ma14_atr",
    "dist_ma22_atr",
    "dist_ma50_atr",
    "dist_ma200_atr",
    "dist_ma1000_atr",
    "slope_ma14_atr",
    "slope_ma22_atr",
    "slope_ma50_atr",
    "slope_ma200_atr",
    "slope_ma1000_atr",
    "accel_ma14_atr",
    "accel_ma50_atr",
    "accel_ma200_atr",
    "spacing_14_50_atr",
    "spacing_50_200_atr",
    "spacing_200_1000_atr",
    "realized_vol_12",
    "realized_vol_48",
    "session_sin",
    "session_cos",
)


@dataclass(frozen=True)
class DirectionalLabels:
    direction: np.ndarray       # [N, H] integer class 0/1/2
    direction_mask: np.ndarray  # [N, H] float 0/1
    path: np.ndarray            # [N, 3] up_mfe, down_mfe, trend_tstat_scaled
    path_mask: np.ndarray       # [N, 3]
    time: np.ndarray            # [N, H, 2] normalized first touch: up, down
    time_mask: np.ndarray       # [N, H, 2]
    ambiguous: np.ndarray       # [N, H] same-bar double-touch flag


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values, nan=0.0))])
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    for i in range(period - 1, len(values)):
        w = values[i - period + 1:i + 1]
        if np.isfinite(w).all():
            out[i] = float(np.std(w))
    return out


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    value = float(np.mean(values[:period]))
    out[period - 1] = value
    alpha = 2.0 / (period + 1.0)
    for i in range(period, len(values)):
        value += (float(values[i]) - value) * alpha
        out[i] = value
    return out


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    delta = np.diff(values, prepend=values[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    ag = _rolling_mean(gain, period)
    al = _rolling_mean(loss, period)
    for i in range(len(values)):
        if not (math.isfinite(ag[i]) and math.isfinite(al[i])):
            continue
        if al[i] <= 1e-12:
            out[i] = 100.0 if ag[i] > 0 else 50.0
        else:
            rs = ag[i] / al[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum.reduce((high - low, np.abs(high - prev), np.abs(low - prev)))
    return _rolling_mean(tr, period)


def build_directional_features(m5) -> tuple[np.ndarray, dict[str, np.ndarray], list[Any]]:
    """Return causal feature matrix, raw arrays, and bar-available timestamps."""
    times = [bar.timestamp + timedelta(minutes=5) for bar in m5]
    o = np.asarray([float(b.open) for b in m5], dtype=np.float64)
    h = np.asarray([float(b.high) for b in m5], dtype=np.float64)
    l = np.asarray([float(b.low) for b in m5], dtype=np.float64)
    c = np.asarray([float(b.close) for b in m5], dtype=np.float64)
    v = np.asarray([max(0.0, float(b.volume or 0.0)) for b in m5], dtype=np.float64)

    atr = _atr(h, l, c)
    atr_safe = np.where(atr > 1e-9, atr, np.nan)
    mas = {p: _ema(c, p) for p in (14, 22, 50, 200, 1000)}
    rsi = _rsi(c)

    logret1 = np.full(len(c), np.nan, dtype=np.float64)
    logret1[1:] = np.log(c[1:] / c[:-1])
    feats: list[np.ndarray] = []
    for lag in (1, 3, 12, 24):
        prev = np.roll(c, lag)
        prev[:lag] = np.nan
        feats.append(np.log(c / prev))

    body = (c - o) / atr_safe
    candle_range = (h - l) / atr_safe
    upper = (h - np.maximum(o, c)) / atr_safe
    lower = (np.minimum(o, c) - l) / atr_safe
    candle_span = h - l
    close_ratio = np.zeros(len(c), dtype=np.float64)
    np.divide(
        c - l,
        candle_span,
        out=close_ratio,
        where=candle_span > 1e-9,
    )
    close_location = close_ratio * 2.0 - 1.0
    close_location[candle_span <= 1e-9] = 0.0
    feats.extend([body, candle_range, upper, lower, close_location])

    lv = np.log1p(v)
    vm = _rolling_mean(lv, 48)
    vs = _rolling_std(lv, 48)
    feats.append((lv - vm) / np.where(vs > 1e-9, vs, np.nan))
    feats.append((_rsi(c) - 50.0) / 50.0)

    for p in (14, 22, 50, 200, 1000):
        feats.append((c - mas[p]) / atr_safe)

    slopes: dict[int, np.ndarray] = {}
    for p in (14, 22, 50, 200, 1000):
        older = np.roll(mas[p], 3)
        older[:3] = np.nan
        slope = (mas[p] - older) / atr_safe
        slopes[p] = slope
        feats.append(slope)

    for p in (14, 50, 200):
        older_slope = np.roll(slopes[p], 3)
        older_slope[:3] = np.nan
        feats.append(slopes[p] - older_slope)

    feats.append((mas[14] - mas[50]) / atr_safe)
    feats.append((mas[50] - mas[200]) / atr_safe)
    feats.append((mas[200] - mas[1000]) / atr_safe)

    feats.append(_rolling_std(logret1, 12))
    feats.append(_rolling_std(logret1, 48))

    minute = np.asarray([t.hour * 60 + t.minute for t in times], dtype=np.float64)
    angle = 2.0 * np.pi * minute / 1440.0
    feats.extend([np.sin(angle), np.cos(angle)])

    matrix = np.column_stack(feats).astype(np.float32)
    if matrix.shape[1] != len(FEATURE_NAMES):
        raise RuntimeError(
            f"feature schema mismatch: {matrix.shape[1]} vs {len(FEATURE_NAMES)}"
        )

    raw = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "atr": atr,
        "rsi": rsi,
        **{f"ma{p}": arr for p, arr in mas.items()},
    }
    return matrix, raw, times


def _trend_tstat(values: np.ndarray) -> float:
    """OLS slope t-statistic over one future path, without external deps."""
    n = len(values)
    if n < 5 or not np.isfinite(values).all():
        return float("nan")
    x = np.arange(n, dtype=np.float64)
    xm = float(x.mean())
    ym = float(values.mean())
    dx = x - xm
    sxx = float(np.dot(dx, dx))
    if sxx <= 1e-12:
        return 0.0
    slope = float(np.dot(dx, values - ym) / sxx)
    intercept = ym - slope * xm
    resid = values - (intercept + slope * x)
    dof = n - 2
    if dof <= 0:
        return 0.0
    sigma2 = float(np.dot(resid, resid) / dof)
    if sigma2 <= 1e-18:
        return float(np.sign(slope) * 10.0) if slope else 0.0
    se = math.sqrt(sigma2 / sxx)
    return float(slope / se) if se > 0 else 0.0


def build_directional_labels(
    raw: dict[str, np.ndarray],
    horizons_minutes: tuple[int, ...] = (30, 60, 120, 240),
) -> DirectionalLabels:
    n = len(raw["close"])
    hn = len(horizons_minutes)
    direction = np.full((n, hn), CLASS_NEUTRAL, dtype=np.int64)
    direction_mask = np.zeros((n, hn), dtype=np.float32)
    time_y = np.zeros((n, hn, 2), dtype=np.float32)
    time_mask = np.zeros((n, hn, 2), dtype=np.float32)
    ambiguous = np.zeros((n, hn), dtype=np.float32)
    path = np.zeros((n, 3), dtype=np.float32)
    path_mask = np.zeros((n, 3), dtype=np.float32)

    c = raw["close"]
    h = raw["high"]
    l = raw["low"]
    atr = raw["atr"]
    max_bars = max(HORIZON_BARS[hmin] for hmin in horizons_minutes)

    for i in range(1000, n - max_bars - 1):
        a = float(atr[i])
        px = float(c[i])
        if not math.isfinite(a) or a <= 1e-9 or not math.isfinite(px):
            continue

        for hidx, hmin in enumerate(horizons_minutes):
            bars = HORIZON_BARS[hmin]
            mult = BARRIER_ATR[hmin]
            upper = px + mult * a
            lower = px - mult * a
            first_class = CLASS_NEUTRAL
            first_at = None
            is_ambiguous = False

            for step, j in enumerate(range(i + 1, i + bars + 1), start=1):
                hit_up = float(h[j]) >= upper
                hit_down = float(l[j]) <= lower
                if hit_up and hit_down:
                    is_ambiguous = True
                    break
                if hit_up:
                    first_class = CLASS_UP
                    first_at = step
                    break
                if hit_down:
                    first_class = CLASS_DOWN
                    first_at = step
                    break

            if is_ambiguous:
                ambiguous[i, hidx] = 1.0
                continue

            direction[i, hidx] = first_class
            direction_mask[i, hidx] = 1.0
            if first_at is not None:
                side_idx = 0 if first_class == CLASS_UP else 1
                time_y[i, hidx, side_idx] = float(first_at / bars)
                time_mask[i, hidx, side_idx] = 1.0

        future_h = h[i + 1:i + max_bars + 1]
        future_l = l[i + 1:i + max_bars + 1]
        future_c = c[i + 1:i + max_bars + 1]
        if len(future_c) == max_bars:
            up_mfe = max(0.0, float(np.max(future_h) - px) / a)
            down_mfe = max(0.0, float(px - np.min(future_l)) / a)
            tstat = _trend_tstat(future_c)
            if math.isfinite(tstat):
                path[i, 0] = min(up_mfe, 10.0)
                path[i, 1] = min(down_mfe, 10.0)
                path[i, 2] = float(np.clip(tstat / 5.0, -2.0, 2.0))
                path_mask[i, :] = 1.0

    return DirectionalLabels(
        direction=direction,
        direction_mask=direction_mask,
        path=path,
        path_mask=path_mask,
        time=time_y,
        time_mask=time_mask,
        ambiguous=ambiguous,
    )


def class_distribution(
    labels: DirectionalLabels,
    indices: np.ndarray,
    horizon_index: int,
) -> dict[str, int]:
    valid = labels.direction_mask[indices, horizon_index] > 0.5
    y = labels.direction[indices[valid], horizon_index]
    return {
        name: int((y == idx).sum())
        for idx, name in enumerate(DIRECTION_CLASSES)
    }
