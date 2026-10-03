"""Causal event grammar and multi-timeframe features for GTG Event Experience v3.

This module deliberately contains market semantics only.  It does not contain
training loops, model selection, execution, sizing, stops, or order routing.

Initial v3 scope:
- bullish MA14/MA50 transition state -> fixed MA200 destination
- bearish MA14/MA50 transition state -> fixed MA200 destination
- causal M1/M5/M15/H1 context, aligned by CLOSED-bar availability
- path labels: target success, MFE, MAE, adverse-first, time-to-target,
  clean-path and signed trend strength

The adverse barrier is a research label (1 ATR from the event price), not a GTG
stop loss and never becomes execution authority by itself.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import math
import numpy as np

from src.platform.marketdata.xau_models import XAUTimeframe


FEATURE_SCHEMA_VERSION = "gtg-event-v3-feature-v1"
LABEL_VERSION = "gtg-event-v3-ma200-path-v1"
EVENT_BULL = "bullish_14_50_to_200"
EVENT_BEAR = "bearish_14_50_to_200"
EVENT_NONE = "none"
EVENT_CODE_BULL = 1
EVENT_CODE_BEAR = -1
EVENT_CODE_NONE = 0

HORIZON_BARS = 48          # 4h on M5
RESEARCH_ADVERSE_ATR = 1.0
MIN_EVENT_GAP_BARS = 12    # reduce near-duplicate observations; not tuned on OOS

TF_DURATION = {
    XAUTimeframe.M1: timedelta(minutes=1),
    XAUTimeframe.M5: timedelta(minutes=5),
    XAUTimeframe.M15: timedelta(minutes=15),
    XAUTimeframe.H1: timedelta(hours=1),
}


@dataclass(frozen=True)
class MultiTimeframeFeatures:
    matrix: np.ndarray
    feature_names: tuple[str, ...]
    times: list[Any]
    m5_raw: dict[str, np.ndarray]
    source_indices: dict[str, np.ndarray]
    source_available_times: dict[str, list[Any]]


@dataclass(frozen=True)
class EventLabels:
    event_code: np.ndarray
    success: np.ndarray
    success_mask: np.ndarray
    mfe_atr: np.ndarray
    mae_atr: np.ndarray
    adverse_first: np.ndarray
    adverse_mask: np.ndarray
    time_to_target: np.ndarray
    time_mask: np.ndarray
    clean_path: np.ndarray
    trend_strength: np.ndarray
    ambiguous: np.ndarray
    target_distance_atr: np.ndarray


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    c = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    out[period - 1:] = (c[period:] - c[:-period]) / period
    return out


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    c = np.concatenate([[0.0], np.cumsum(values, dtype=np.float64)])
    c2 = np.concatenate([[0.0], np.cumsum(values * values, dtype=np.float64)])
    s = c[period:] - c[:-period]
    s2 = c2[period:] - c2[:-period]
    mean = s / period
    var = np.maximum(0.0, s2 / period - mean * mean)
    out[period - 1:] = np.sqrt(var)
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


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum.reduce((high - low, np.abs(high - prev), np.abs(low - prev)))
    return _rolling_mean(tr, period)


def _rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    ag = _rolling_mean(gain, period)
    al = _rolling_mean(loss, period)
    out = np.full(len(close), np.nan, dtype=np.float64)
    valid = np.isfinite(ag) & np.isfinite(al)
    for i in np.where(valid)[0]:
        if al[i] <= 1e-12:
            out[i] = 100.0 if ag[i] > 0 else 50.0
        else:
            rs = ag[i] / al[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _lag(values: np.ndarray, amount: int) -> np.ndarray:
    out = np.roll(values, amount)
    out[:amount] = np.nan
    return out


def _trend_tstat(values: np.ndarray) -> float:
    n = len(values)
    if n < 5 or not np.isfinite(values).all():
        return 0.0
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


def _frame_state(rows, timeframe: XAUTimeframe, include_ma1000: bool) -> tuple[
    np.ndarray, tuple[str, ...], dict[str, np.ndarray], list[Any]
]:
    if not rows:
        raise ValueError(f"empty timeframe: {timeframe.value}")

    o = np.asarray([float(b.open) for b in rows], dtype=np.float64)
    h = np.asarray([float(b.high) for b in rows], dtype=np.float64)
    l = np.asarray([float(b.low) for b in rows], dtype=np.float64)
    c = np.asarray([float(b.close) for b in rows], dtype=np.float64)
    v = np.asarray([max(0.0, float(b.volume or 0.0)) for b in rows], dtype=np.float64)
    atr = _atr(h, l, c)
    atr_safe = np.where(atr > 1e-9, atr, np.nan)
    rsi = _rsi(c)

    periods = [14, 22, 50, 200] + ([1000] if include_ma1000 else [])
    mas = {p: _ema(c, p) for p in periods}
    slopes: dict[int, np.ndarray] = {}
    for p in periods:
        slopes[p] = (mas[p] - _lag(mas[p], 3)) / atr_safe

    logret = np.zeros(len(c), dtype=np.float64)
    logret[1:] = np.log(c[1:] / c[:-1])
    rv12 = _rolling_std(logret, 12)
    rv48 = _rolling_std(logret, 48)
    lv = np.log1p(v)
    vm = _rolling_mean(lv, 48)
    vs = _rolling_std(lv, 48)
    vol_z = (lv - vm) / np.where(vs > 1e-9, vs, np.nan)

    rsi_slope = rsi - _lag(rsi, 3)
    rsi_accel = rsi_slope - _lag(rsi_slope, 3)

    names: list[str] = []
    cols: list[np.ndarray] = []

    def add(name: str, arr: np.ndarray) -> None:
        names.append(f"{timeframe.value}_{name}")
        cols.append(arr)

    add("logret_1", logret)
    add("logret_3", np.log(c / _lag(c, 3)))
    add("body_atr", (c - o) / atr_safe)
    add("range_atr", (h - l) / atr_safe)
    add("volume_z48", vol_z)
    add("rsi14", (rsi - 50.0) / 50.0)
    add("rsi_slope3", rsi_slope / 50.0)
    add("rsi_accel3", rsi_accel / 50.0)
    add("rv12", rv12)
    add("rv48", rv48)

    for p in periods:
        add(f"dist_ma{p}_atr", (c - mas[p]) / atr_safe)
    for p in periods:
        add(f"slope_ma{p}_atr", slopes[p])
    for p in (14, 50, 200):
        if p in slopes:
            add(f"accel_ma{p}_atr", slopes[p] - _lag(slopes[p], 3))

    add("spacing_14_50_atr", (mas[14] - mas[50]) / atr_safe)
    add("spacing_50_200_atr", (mas[50] - mas[200]) / atr_safe)
    if include_ma1000:
        add("spacing_200_1000_atr", (mas[200] - mas[1000]) / atr_safe)

    matrix = np.column_stack(cols).astype(np.float32)
    duration = TF_DURATION[timeframe]
    available = [b.timestamp + duration for b in rows]
    raw = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "atr": atr,
        "rsi": rsi,
        "rv12": rv12,
        "rv48": rv48,
        **{f"ma{p}": mas[p] for p in periods},
        **{f"slope_ma{p}": slopes[p] for p in periods},
    }
    return matrix, tuple(names), raw, available


def build_multitimeframe_features(
    m1,
    m5,
    m15,
    h1,
) -> MultiTimeframeFeatures:
    """Build a M5 decision grid using only features known at decision time.

    Higher-timeframe bars are aligned by their *availability time* (bar open +
    duration), never by bar-open timestamp.  Therefore an H1 candle that starts
    at 10:00 cannot be used by a 10:30 decision; it becomes available at 11:00.
    """
    frames = (
        (XAUTimeframe.M1, m1, False),
        (XAUTimeframe.M5, m5, True),
        (XAUTimeframe.M15, m15, False),
        (XAUTimeframe.H1, h1, False),
    )
    decision_times = [b.timestamp + timedelta(minutes=5) for b in m5]
    decision_epoch = np.asarray([t.timestamp() for t in decision_times], dtype=np.float64)

    aligned_parts: list[np.ndarray] = []
    names: list[str] = []
    source_indices: dict[str, np.ndarray] = {}
    source_available_times: dict[str, list[Any]] = {}
    m5_raw: dict[str, np.ndarray] | None = None

    for tf, rows, include_ma1000 in frames:
        state, state_names, raw, available = _frame_state(rows, tf, include_ma1000)
        avail_epoch = np.asarray([t.timestamp() for t in available], dtype=np.float64)
        idx = np.searchsorted(avail_epoch, decision_epoch, side="right") - 1
        aligned = np.full((len(decision_times), state.shape[1]), np.nan, dtype=np.float32)
        valid = idx >= 0
        aligned[valid] = state[idx[valid]]
        aligned_parts.append(aligned)
        names.extend(state_names)
        source_indices[tf.value] = idx.astype(np.int64)
        source_available_times[tf.value] = available
        if tf is XAUTimeframe.M5:
            # Decision grid is M5, so this raw state is index-aligned with rows.
            m5_raw = raw

    minute = np.asarray([t.hour * 60 + t.minute for t in decision_times], dtype=np.float64)
    angle = 2.0 * np.pi * minute / 1440.0
    aligned_parts.append(np.column_stack((np.sin(angle), np.cos(angle))).astype(np.float32))
    names.extend(("session_sin", "session_cos"))

    matrix = np.column_stack(aligned_parts).astype(np.float32)
    if m5_raw is None:
        raise RuntimeError("M5 raw state was not built")
    if matrix.shape[1] != len(names):
        raise RuntimeError("feature schema width mismatch")

    return MultiTimeframeFeatures(
        matrix=matrix,
        feature_names=tuple(names),
        times=decision_times,
        m5_raw=m5_raw,
        source_indices=source_indices,
        source_available_times=source_available_times,
    )


def _event_side(raw: dict[str, np.ndarray], i: int) -> int:
    """Detect a price transition through the MA14/MA50 cluster toward MA200.

    MA slope, acceleration, compression and RSI remain learned context rather
    than hard-coded thresholds. The transition uses only current and
    three-bars-earlier state.
    """
    if i < 3:
        return EVENT_CODE_NONE
    px = float(raw["close"][i])
    old_px = float(raw["close"][i - 3])
    atr = float(raw["atr"][i])
    target = float(raw["ma200"][i])
    ma14 = float(raw["ma14"][i])
    ma50 = float(raw["ma50"][i])
    old14 = float(raw["ma14"][i - 3])
    old50 = float(raw["ma50"][i - 3])
    vals = (px, old_px, atr, target, ma14, ma50, old14, old50)
    if not all(math.isfinite(v) for v in vals) or atr <= 1e-9:
        return EVENT_CODE_NONE

    distance = abs(target - px) / atr
    # Reuse the v1 MA200 destination support; do not tune this band on v3 OOS.
    if not (0.5 <= distance <= 6.0):
        return EVENT_CODE_NONE

    current_top = max(ma14, ma50)
    current_bottom = min(ma14, ma50)
    old_top = max(old14, old50)
    old_bottom = min(old14, old50)

    bullish_transition = (
        target > px
        and px > current_top
        and old_px <= old_top
    )
    bearish_transition = (
        target < px
        and px < current_bottom
        and old_px >= old_bottom
    )
    if bullish_transition:
        return EVENT_CODE_BULL
    if bearish_transition:
        return EVENT_CODE_BEAR
    return EVENT_CODE_NONE


def build_event_labels(raw: dict[str, np.ndarray]) -> EventLabels:
    n = len(raw["close"])
    event_code = np.zeros(n, dtype=np.int8)
    success = np.zeros(n, dtype=np.float32)
    success_mask = np.zeros(n, dtype=np.float32)
    mfe = np.zeros(n, dtype=np.float32)
    mae = np.zeros(n, dtype=np.float32)
    adverse_first = np.zeros(n, dtype=np.float32)
    adverse_mask = np.zeros(n, dtype=np.float32)
    time_to_target = np.zeros(n, dtype=np.float32)
    time_mask = np.zeros(n, dtype=np.float32)
    clean = np.zeros(n, dtype=np.float32)
    trend = np.zeros(n, dtype=np.float32)
    ambiguous = np.zeros(n, dtype=np.float32)
    target_distance = np.zeros(n, dtype=np.float32)

    close = raw["close"]
    high = raw["high"]
    low = raw["low"]
    atr = raw["atr"]
    ma200 = raw["ma200"]
    last_event = {EVENT_CODE_BULL: -10_000, EVENT_CODE_BEAR: -10_000}

    for i in range(1000, n - HORIZON_BARS - 1):
        side = _event_side(raw, i)
        if side == EVENT_CODE_NONE:
            continue
        if i - last_event[side] < MIN_EVENT_GAP_BARS:
            continue
        last_event[side] = i
        event_code[i] = side

        a = float(atr[i])
        px = float(close[i])
        target = float(ma200[i])
        target_distance[i] = abs(target - px) / a
        adverse_level = px - side * RESEARCH_ADVERSE_ATR * a

        target_step: int | None = None
        adverse_step: int | None = None
        is_ambiguous = False
        end = min(n, i + HORIZON_BARS + 1)
        for step, j in enumerate(range(i + 1, end), start=1):
            target_hit = (
                float(high[j]) >= target if side > 0 else float(low[j]) <= target
            )
            adverse_hit = (
                float(low[j]) <= adverse_level if side > 0
                else float(high[j]) >= adverse_level
            )
            if target_hit and target_step is None:
                target_step = step
            if adverse_hit and adverse_step is None:
                adverse_step = step
            if (
                target_step is not None
                and adverse_step is not None
                and target_step == adverse_step
            ):
                is_ambiguous = True
                break
            if target_step is not None and adverse_step is not None:
                break

        ambiguous[i] = float(is_ambiguous)
        if is_ambiguous:
            continue

        hit = target_step is not None
        adv_first = (
            adverse_step is not None
            and (target_step is None or adverse_step < target_step)
        )
        success[i] = float(hit)
        success_mask[i] = 1.0
        adverse_first[i] = float(adv_first)
        adverse_mask[i] = 1.0
        clean[i] = float(hit and not adv_first)

        path_bars = target_step if target_step is not None else HORIZON_BARS
        path_end = min(n, i + path_bars + 1)
        future_h = high[i + 1:path_end]
        future_l = low[i + 1:path_end]
        future_c = close[i + 1:path_end]
        if len(future_c):
            if side > 0:
                mfe[i] = max(0.0, float(np.max(future_h) - px) / a)
                mae[i] = max(0.0, float(px - np.min(future_l)) / a)
            else:
                mfe[i] = max(0.0, float(px - np.min(future_l)) / a)
                mae[i] = max(0.0, float(np.max(future_h) - px) / a)
            trend[i] = float(np.clip(side * _trend_tstat(future_c) / 5.0, -2.0, 2.0))

        if target_step is not None:
            time_to_target[i] = float(target_step / HORIZON_BARS)
            time_mask[i] = 1.0

    return EventLabels(
        event_code=event_code,
        success=success,
        success_mask=success_mask,
        mfe_atr=mfe,
        mae_atr=mae,
        adverse_first=adverse_first,
        adverse_mask=adverse_mask,
        time_to_target=time_to_target,
        time_mask=time_mask,
        clean_path=clean,
        trend_strength=trend,
        ambiguous=ambiguous,
        target_distance_atr=target_distance,
    )


def event_name(code: int) -> str:
    if int(code) == EVENT_CODE_BULL:
        return EVENT_BULL
    if int(code) == EVENT_CODE_BEAR:
        return EVENT_BEAR
    return EVENT_NONE


def session_name(timestamp) -> str:
    hour = int(timestamp.hour)
    if 0 <= hour < 7:
        return "asia"
    if 7 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def assert_closed_bar_alignment(features: MultiTimeframeFeatures) -> None:
    """Raise if any aligned source bar becomes available after decision time."""
    for tf_name, idx in features.source_indices.items():
        available = features.source_available_times[tf_name]
        for pos, source_idx in enumerate(idx):
            if source_idx < 0:
                continue
            if available[int(source_idx)] > features.times[pos]:
                raise AssertionError(
                    f"future {tf_name} bar at decision {features.times[pos]}"
                )
