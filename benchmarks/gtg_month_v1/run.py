"""GTG v1.0 one-month historical-core benchmark.

Evaluates the frozen GTG methodology on one month of the existing two-year
Dukascopy XAUUSD dataset. Warm-up data is historical-only and excluded from
performance. The benchmark never fabricates unavailable footprint/order-book
history.
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from bisect import bisect_right
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, median
from typing import Any

from benchmarks.gen1_gold_2y_real_v1 import run as legacy
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe

UTC = timezone.utc
WARMUP_START = datetime.fromisoformat(os.getenv("GTG_WARMUP_START", "2025-11-23")).replace(tzinfo=UTC)
TEST_START = datetime.fromisoformat(os.getenv("GTG_TEST_START", "2026-08-23")).replace(tzinfo=UTC)
TEST_END = datetime.fromisoformat(os.getenv("GTG_TEST_END", "2026-09-23")).replace(tzinfo=UTC)
STEP_MINUTES = int(os.getenv("GTG_STEP_MINUTES", "5"))
OUT = Path(os.getenv("GTG_OUT", "artifacts/gtg_month_v1"))

MA_PERIODS = (14, 22, 50, 200, 1000)
TF_ORDER = (
    XAUTimeframe.M1,
    XAUTimeframe.M5,
    XAUTimeframe.M15,
    XAUTimeframe.H1,
    XAUTimeframe.H4,
)


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    seed = fmean(values[:period])
    k = 2.0 / (period + 1.0)
    value = seed
    for item in values[period:]:
        value += (item - value) * k
    return float(value)


def _rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    deltas = [values[i] - values[i - 1] for i in range(len(values) - period, len(values))]
    gains = [max(v, 0.0) for v in deltas]
    losses = [max(-v, 0.0) for v in deltas]
    ag = fmean(gains)
    al = fmean(losses)
    if al == 0:
        return 100.0 if ag > 0 else 50.0
    rs = ag / al
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(bars: list[XAUBar], period: int = 14) -> float | None:
    if len(bars) < period + 1:
        return None
    vals: list[float] = []
    for i in range(len(bars) - period, len(bars)):
        cur = bars[i]
        prev = bars[i - 1]
        vals.append(max(
            float(cur.high) - float(cur.low),
            abs(float(cur.high) - float(prev.close)),
            abs(float(cur.low) - float(prev.close)),
        ))
    return fmean(vals)


def _sign(v: float, eps: float = 1e-9) -> int:
    return 1 if v > eps else -1 if v < -eps else 0


def _frame_state(bars: list[XAUBar]) -> dict[str, Any] | None:
    if len(bars) < 55:
        return None
    closes = [float(b.close) for b in bars]
    atr = _atr(bars) or 0.0
    emas: dict[int, float | None] = {p: _ema(closes, p) for p in MA_PERIODS}
    prev: dict[int, float | None] = {p: _ema(closes[:-1], p) for p in MA_PERIODS}
    prev2: dict[int, float | None] = {p: _ema(closes[:-2], p) for p in MA_PERIODS}
    back = 5
    older: dict[int, float | None] = {
        p: _ema(closes[:-back], p) if len(closes) > p + back else None
        for p in MA_PERIODS
    }
    slopes: dict[int, float | None] = {}
    for p in MA_PERIODS:
        if emas[p] is None or older[p] is None or atr <= 0:
            slopes[p] = None
        else:
            slopes[p] = (float(emas[p]) - float(older[p])) / atr

    cross_14_50 = 0
    price_break_14_50 = 0
    if None not in (emas[14], emas[50], prev[14], prev[50]):
        if float(prev[14]) <= float(prev[50]) and float(emas[14]) > float(emas[50]):
            cross_14_50 = 1
        elif float(prev[14]) >= float(prev[50]) and float(emas[14]) < float(emas[50]):
            cross_14_50 = -1

        current_upper = max(float(emas[14]), float(emas[50]))
        current_lower = min(float(emas[14]), float(emas[50]))
        previous_upper = max(float(prev[14]), float(prev[50]))
        previous_lower = min(float(prev[14]), float(prev[50]))
        if closes[-1] > current_upper and closes[-2] <= previous_upper:
            price_break_14_50 = 1
        elif closes[-1] < current_lower and closes[-2] >= previous_lower:
            price_break_14_50 = -1

    accepted_200 = 0
    if None not in (emas[200], prev[200], prev2[200]) and len(closes) >= 3:
        if closes[-1] > float(emas[200]) and closes[-2] > float(prev[200]) and closes[-3] <= float(prev2[200]):
            accepted_200 = 1
        elif closes[-1] < float(emas[200]) and closes[-2] < float(prev[200]) and closes[-3] >= float(prev2[200]):
            accepted_200 = -1

    latest = bars[-1]
    barrier_rejection: dict[int, int] = {200: 0, 1000: 0}
    for p in (200, 1000):
        level = emas[p]
        if level is None:
            continue
        level = float(level)
        if float(latest.low) <= level <= float(latest.high):
            if float(latest.open) > level and float(latest.close) > level:
                barrier_rejection[p] = 1
            elif float(latest.open) < level and float(latest.close) < level:
                barrier_rejection[p] = -1

    prior = bars[-21:-1]
    swing_high = max(float(b.high) for b in prior)
    swing_low = min(float(b.low) for b in prior)
    bos = 1 if float(latest.close) > swing_high else -1 if float(latest.close) < swing_low else 0

    order_block: dict[str, float] | None = None
    if bos:
        for bar in reversed(bars[-13:-1]):
            bearish = float(bar.close) < float(bar.open)
            bullish = float(bar.close) > float(bar.open)
            if (bos > 0 and bearish) or (bos < 0 and bullish):
                order_block = {
                    "side": float(bos),
                    "low": float(bar.low),
                    "high": float(bar.high),
                }
                break

    direction = 0
    if None not in (emas[50], emas[200]):
        if (
            float(latest.close) > float(emas[50]) > float(emas[200])
            and (slopes[50] or 0.0) > 0
        ):
            direction = 1
        elif (
            float(latest.close) < float(emas[50]) < float(emas[200])
            and (slopes[50] or 0.0) < 0
        ):
            direction = -1

    spacing = None
    if None not in (emas[14], emas[50]) and atr > 0:
        spacing = (float(emas[14]) - float(emas[50])) / atr

    return {
        "observed_at": latest.timestamp,
        "close": float(latest.close),
        "open": float(latest.open),
        "high": float(latest.high),
        "low": float(latest.low),
        "volume": float(latest.volume or 0.0),
        "atr": atr,
        "rsi14": _rsi(closes, 14),
        "ema": emas,
        "slope": slopes,
        "spacing_14_50_atr": spacing,
        "cross_14_50": cross_14_50,
        "price_break_14_50": price_break_14_50,
        "accepted_200": accepted_200,
        "barrier_rejection": barrier_rejection,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "bos": bos,
        "order_block": order_block,
        "direction": direction,
    }


def _volume_profile(bars: list[XAUBar], bins: int = 48) -> dict[str, Any] | None:
    rows = bars[-288:]
    if len(rows) < 60:
        return None
    lo = min(float(b.low) for b in rows)
    hi = max(float(b.high) for b in rows)
    if hi <= lo:
        return None
    width = (hi - lo) / bins
    vol = [0.0] * bins
    for b in rows:
        px = (float(b.high) + float(b.low) + float(b.close)) / 3.0
        idx = min(bins - 1, max(0, int((px - lo) / width)))
        vol[idx] += max(0.0, float(b.volume or 0.0))
    total = sum(vol)
    if total <= 0:
        return None
    poc_i = max(range(bins), key=lambda i: vol[i])
    selected = {poc_i}
    acc = vol[poc_i]
    left = poc_i - 1
    right = poc_i + 1
    while acc / total < 0.70 and (left >= 0 or right < bins):
        lv = vol[left] if left >= 0 else -1.0
        rv = vol[right] if right < bins else -1.0
        if rv > lv:
            selected.add(right)
            acc += max(0.0, rv)
            right += 1
        else:
            selected.add(left)
            acc += max(0.0, lv)
            left -= 1
    val_i = min(selected)
    vah_i = max(selected)
    return {
        "poc": lo + (poc_i + 0.5) * width,
        "val": lo + val_i * width,
        "vah": lo + (vah_i + 1) * width,
        "range_low": lo,
        "range_high": hi,
        "volume_semantics": "dukascopy_quoted_tick_activity_proxy",
    }


def _session(ts: datetime) -> str:
    h = ts.hour
    if 0 <= h < 7:
        return "asia"
    if 7 <= h < 12:
        return "london"
    if 12 <= h < 16:
        return "london_newyork_overlap"
    if 16 <= h < 21:
        return "newyork"
    return "off_session"


def _macro_provider() -> tuple[Any, dict[str, Any]]:
    if os.getenv("GTG_MACRO_PROXY", "1") != "1":
        return (lambda _at: {"bias": 0}), {"status": "disabled"}
    try:
        provider, diag = legacy.build_historical_macro_proxy(TEST_START - timedelta(days=45), TEST_END)
        return provider, {"status": "available_proxy", **(diag or {})}
    except Exception as exc:
        return (lambda _at: {"bias": 0}), {"status": "unavailable", "reason": type(exc).__name__}


def _regime(states: dict[str, dict[str, Any]]) -> int:
    score = 0
    for key, weight in (("1h", 1), ("4h", 2)):
        st = states.get(key)
        if not st:
            continue
        score += int(st.get("direction", 0)) * weight
    return 1 if score >= 2 else -1 if score <= -2 else 0


def _zone_bonus(side: int, st: dict[str, Any]) -> float:
    atr = float(st.get("atr") or 0.0)
    if atr <= 0:
        return 0.0
    px = float(st["close"])
    if side > 0 and px <= float(st["swing_low"]) + 0.45 * atr:
        return 0.75
    if side < 0 and px >= float(st["swing_high"]) - 0.45 * atr:
        return 0.75
    ob = st.get("order_block")
    if ob and int(ob["side"]) == side:
        if float(ob["low"]) - 0.15 * atr <= px <= float(ob["high"]) + 0.15 * atr:
            return 0.75
    return 0.0


def _target_ahead(side: int, price: float, target: float | None) -> bool:
    if target is None:
        return False
    return target > price if side > 0 else target < price


def _select_setup(
    states: dict[str, dict[str, Any]],
    vp: dict[str, Any] | None,
    macro_bias: int,
    session: str,
) -> dict[str, Any] | None:
    m5 = states.get("5m")
    m15 = states.get("15m")
    h1 = states.get("1h")
    h4 = states.get("4h")
    if not all((m5, m15, h1, h4)):
        return None
    px = float(m5["close"])
    reg = _regime(states)

    setup = None
    side = 0
    target = None
    barrier_period = None

    price_break = int(m5["price_break_14_50"])
    if price_break:
        ma200 = m5["ema"].get(200)
        if _target_ahead(price_break, px, ma200):
            setup = "ladder_price_14_50_to_200"
            side = price_break
            target = float(ma200)

    if setup is None:
        acc = int(m5["accepted_200"])
        if acc:
            ma1000 = m5["ema"].get(1000)
            if _target_ahead(acc, px, ma1000):
                setup = "ladder_200_to_1000"
                side = acc
                target = float(ma1000)

    if setup is None:
        for p in (1000, 200):
            reject = int(m5["barrier_rejection"].get(p, 0))
            if reject:
                setup = "ma_barrier_rejection"
                side = reject
                barrier_period = p
                target = None
                break

    if setup is None or side == 0:
        return None

    if setup.startswith("ladder_") and reg * side < 1:
        return None
    if setup == "ma_barrier_rejection" and reg * side < 0:
        return None

    score = 2.0
    score += 1.5 if reg * side > 0 else 0.0
    score += 1.0 if int(m15.get("direction", 0)) == side else 0.0

    rsi5 = float(m5.get("rsi14") or 50.0)
    rsi15 = float(m15.get("rsi14") or 50.0)
    if side > 0 and 50.0 <= rsi5 <= 72.0 and rsi15 >= 48.0:
        score += 1.0
    elif side < 0 and 28.0 <= rsi5 <= 50.0 and rsi15 <= 52.0:
        score += 1.0

    if int(m5.get("bos", 0)) == side or int(m15.get("bos", 0)) == side:
        score += 1.0

    if vp is not None:
        poc = float(vp["poc"])
        if (side > 0 and px >= poc) or (side < 0 and px <= poc):
            score += 1.0

    score += _zone_bonus(side, m15)

    if macro_bias * side > 0:
        score += 0.5
    elif macro_bias * side < 0:
        score -= 0.75

    if session in {"london", "london_newyork_overlap", "newyork"}:
        score += 0.25

    atr = float(m5.get("atr") or 0.0)
    if atr <= 0:
        return None

    if setup.startswith("ladder_") and target is not None:
        distance = abs(target - px)
        stop_probe = max(1.05 * atr, px * 0.0008)
        if distance < 1.15 * stop_probe:
            return None

    if score < 5.0:
        return None

    return {
        "setup_type": setup,
        "side": side,
        "score": round(score, 3),
        "next_ma_target": target,
        "barrier_period": barrier_period,
        "regime": reg,
        "rsi5": rsi5,
        "rsi15": rsi15,
        "poc": vp.get("poc") if vp else None,
        "vah": vp.get("vah") if vp else None,
        "val": vp.get("val") if vp else None,
    }


def _quote_at(
    quotes: dict[datetime, tuple[float, float]],
    quote_times: list[datetime],
    at: datetime,
) -> tuple[float, float] | None:
    direct = quotes.get(at)
    if direct is not None:
        return direct
    i = bisect_right(quote_times, at) - 1
    if i < 0:
        return None
    ts = quote_times[i]
    if at - ts > timedelta(minutes=2):
        return None
    return quotes.get(ts)


def _simulate(
    *,
    m1: list[XAUBar],
    m1_available: list[datetime],
    quotes: dict[datetime, tuple[float, float]],
    quote_times: list[datetime],
    at: datetime,
    setup: dict[str, Any],
    m5: dict[str, Any],
) -> dict[str, Any] | None:
    side = int(setup["side"])
    q = _quote_at(quotes, quote_times, at)
    if q is None:
        return None
    bid, ask = q
    mid = (bid + ask) / 2.0
    entry = ask if side > 0 else bid
    atr = float(m5["atr"])
    stop_distance = max(1.05 * atr, mid * 0.0008)
    cap = mid * 0.0035
    stop_distance = min(stop_distance, cap)

    if side > 0:
        structural = mid - float(m5["swing_low"])
    else:
        structural = float(m5["swing_high"]) - mid
    if 0 < structural <= cap:
        stop_distance = max(stop_distance, structural + 0.10 * atr)

    setup_type = str(setup["setup_type"])
    target_ref = setup.get("next_ma_target")
    if setup_type.startswith("ladder_") and target_ref is not None:
        raw_dist = abs(float(target_ref) - mid)
        if raw_dist < 1.15 * stop_distance:
            return None
        target_distance = min(raw_dist, 2.2 * stop_distance)
        max_hold = 480 if setup_type == "ladder_200_to_1000" else 240
    else:
        target_distance = 1.6 * stop_distance
        max_hold = 180

    stop_mid = mid - side * stop_distance
    target_mid = mid + side * target_distance
    expiry = at + timedelta(minutes=max_hold)

    start_i = bisect_right(m1_available, at)
    end_i = bisect_right(m1_available, expiry)
    future = m1[start_i:end_i]
    if not future:
        return None

    exit_time = None
    exit_mid = float(future[-1].close)
    exit_reason = "time_expiry"
    ambiguous = False
    for bar in future:
        hit_target = float(bar.high) >= target_mid if side > 0 else float(bar.low) <= target_mid
        hit_stop = float(bar.low) <= stop_mid if side > 0 else float(bar.high) >= stop_mid
        available_at = bar.timestamp + timedelta(minutes=1)
        if hit_target and hit_stop:
            exit_time = available_at
            exit_mid = stop_mid
            exit_reason = "ambiguous_same_bar_stop"
            ambiguous = True
            break
        if hit_stop:
            exit_time = available_at
            exit_mid = stop_mid
            exit_reason = "stop"
            break
        if hit_target:
            exit_time = available_at
            exit_mid = target_mid
            exit_reason = "target"
            break
    if exit_time is None:
        exit_time = min(expiry, future[-1].timestamp + timedelta(minutes=1))

    xq = _quote_at(quotes, quote_times, exit_time)
    if xq is None:
        return None
    xbid, xask = xq
    half_spread = max(0.0, xask - xbid) / 2.0
    if exit_reason == "time_expiry":
        exit_exec = xbid if side > 0 else xask
    elif side > 0:
        exit_exec = exit_mid - half_spread
    else:
        exit_exec = exit_mid + half_spread

    net_bps = side * ((exit_exec - entry) / entry) * 10000.0
    return {
        "entry_time": at,
        "exit_time": exit_time,
        "setup_type": setup_type,
        "direction": "LONG" if side > 0 else "SHORT",
        "score": setup["score"],
        "session": _session(at),
        "regime": setup["regime"],
        "entry_mid": mid,
        "entry_exec": entry,
        "exit_exec": exit_exec,
        "stop_mid": stop_mid,
        "target_mid": target_mid,
        "next_ma_target_at_entry": target_ref,
        "barrier_period": setup.get("barrier_period"),
        "planned_rr": target_distance / stop_distance,
        "duration_minutes": max(0.0, (exit_time - at).total_seconds() / 60.0),
        "exit_reason": exit_reason,
        "ambiguous_same_bar": ambiguous,
        "net_bps": net_bps,
        "won": net_bps > 0,
        "rsi5": setup["rsi5"],
        "rsi15": setup["rsi15"],
        "poc": setup["poc"],
        "vah": setup["vah"],
        "val": setup["val"],
    }


def _future_target(
    m1: list[XAUBar],
    m1_available: list[datetime],
    at: datetime,
    side: int,
    target: float,
    horizon_minutes: int,
) -> dict[str, Any]:
    start = bisect_right(m1_available, at)
    end = bisect_right(m1_available, at + timedelta(minutes=horizon_minutes))
    for bar in m1[start:end]:
        hit = float(bar.high) >= target if side > 0 else float(bar.low) <= target
        if hit:
            t = bar.timestamp + timedelta(minutes=1)
            return {"hit": True, "minutes": (t - at).total_seconds() / 60.0}
    return {"hit": False, "minutes": None}


def _future_rejection(
    m1: list[XAUBar],
    m1_available: list[datetime],
    at: datetime,
    side: int,
    start_price: float,
    barrier: float,
    atr: float,
) -> dict[str, Any]:
    start = bisect_right(m1_available, at)
    end = bisect_right(m1_available, at + timedelta(minutes=120))
    target = start_price + side * atr
    for bar in m1[start:end]:
        target_hit = float(bar.high) >= target if side > 0 else float(bar.low) <= target
        invalid = float(bar.low) < barrier if side > 0 else float(bar.high) > barrier
        t = bar.timestamp + timedelta(minutes=1)
        if target_hit and invalid:
            return {"hit": False, "reason": "ambiguous", "minutes": None}
        if invalid:
            return {"hit": False, "reason": "re_cross", "minutes": None}
        if target_hit:
            return {"hit": True, "reason": "one_atr_away", "minutes": (t - at).total_seconds() / 60.0}
    return {"hit": False, "reason": "timeout", "minutes": None}


def _metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"trade_count": 0}
    pnl = [float(t["net_bps"]) for t in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    cum = peak = 0.0
    dd = 0.0
    for x in pnl:
        cum += x
        peak = max(peak, cum)
        dd = min(dd, cum - peak)
    gross_loss = abs(sum(losses))
    return {
        "trade_count": len(trades),
        "win_rate": sum(x > 0 for x in pnl) / len(pnl),
        "mean_net_bps": fmean(pnl),
        "median_net_bps": median(pnl),
        "cumulative_net_bps": sum(pnl),
        "profit_factor": sum(wins) / gross_loss if gross_loss > 0 else None,
        "max_drawdown_bps": dd,
        "target_rate": sum(t["exit_reason"] == "target" for t in trades) / len(trades),
        "stop_rate": sum(t["exit_reason"] in {"stop", "ambiguous_same_bar_stop"} for t in trades) / len(trades),
        "median_duration_minutes": median(float(t["duration_minutes"]) for t in trades),
    }


def _event_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"event_count": 0}
    hits = [r for r in rows if r.get("hit")]
    mins = [float(r["minutes"]) for r in hits if r.get("minutes") is not None]
    return {
        "event_count": len(rows),
        "hit_rate": len(hits) / len(rows),
        "median_minutes_to_hit": median(mins) if mins else None,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in list(out.items()):
                if isinstance(value, datetime):
                    out[key] = value.isoformat()
            writer.writerow(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    m1, quotes, dataset = legacy.download_dataset(WARMUP_START, TEST_END)
    bars = {
        XAUTimeframe.M1: m1,
        XAUTimeframe.M5: legacy.resample(m1, XAUTimeframe.M5),
        XAUTimeframe.M15: legacy.resample(m1, XAUTimeframe.M15),
        XAUTimeframe.H1: legacy.resample(m1, XAUTimeframe.H1),
        XAUTimeframe.H4: legacy.resample(m1, XAUTimeframe.H4),
    }
    available = {tf: legacy.availability(rows, tf) for tf, rows in bars.items()}
    m1_available = available[XAUTimeframe.M1]
    quote_times = sorted(quotes)

    macro_provider, macro_diag = _macro_provider()

    diagnostics: dict[str, int] = defaultdict(int)
    trades: list[dict[str, Any]] = []
    ladder_price_14_50: dict[str, list[dict[str, Any]]] = {
        tf.value: [] for tf in TF_ORDER
    }
    ladder_ma_cross_14_50: dict[str, list[dict[str, Any]]] = {
        tf.value: [] for tf in TF_ORDER
    }
    ladder_200_1000: dict[str, list[dict[str, Any]]] = {
        tf.value: [] for tf in TF_ORDER
    }
    rejection_200: list[dict[str, Any]] = []
    rejection_1000: list[dict[str, Any]] = []
    busy_until = TEST_START
    last_eval: datetime | None = None
    seen_price_break_events: set[tuple[str, datetime, int]] = set()
    seen_ma_cross_events: set[tuple[str, datetime, int]] = set()
    seen_acceptance_events: set[tuple[str, datetime, int]] = set()

    for at in m1_available:
        if at < TEST_START or at >= TEST_END:
            continue
        if last_eval is not None and at - last_eval < timedelta(minutes=STEP_MINUTES):
            continue
        last_eval = at
        diagnostics["market_scans"] += 1

        states: dict[str, dict[str, Any]] = {}
        for tf in TF_ORDER:
            window = legacy.bars_window(bars[tf], available[tf], at, 1120)
            state = _frame_state(window)
            if state is not None:
                states[tf.value] = state

        m5 = states.get("5m")
        if not m5:
            diagnostics["insufficient_m5_state"] += 1
            continue

        # Hypothesis measurements are independent from trade filters.
        # The primary event is PRICE crossing/reclaiming the MA14/MA50 pair.
        # MA14 crossing MA50 is preserved separately as a secondary dynamic event.
        horizon_minutes = {
            "1m": 120,
            "5m": 240,
            "15m": 720,
            "1h": 1440,
            "4h": 4320,
        }
        target_1000_minutes = {
            "1m": 240,
            "5m": 480,
            "15m": 1440,
            "1h": 2880,
            "4h": 8640,
        }
        for tf_key, state in states.items():
            event_time = state.get("observed_at")
            pb = int(state.get("price_break_14_50", 0))
            target200 = state["ema"].get(200)
            h200 = int(horizon_minutes[tf_key])
            pb_key = (tf_key, event_time, pb)
            if (
                pb
                and event_time is not None
                and pb_key not in seen_price_break_events
            ):
                seen_price_break_events.add(pb_key)
                if (
                    target200 is not None
                    and _target_ahead(pb, float(state["close"]), target200)
                    and at + timedelta(minutes=h200) <= TEST_END
                ):
                    outcome = _future_target(
                        m1, m1_available, at, pb, float(target200), h200
                    )
                    ladder_price_14_50[tf_key].append({
                        "time": at,
                        "event_bar_time": event_time,
                        "side": pb,
                        "target": float(target200),
                        **outcome,
                    })

            cross = int(state.get("cross_14_50", 0))
            cross_key = (tf_key, event_time, cross)
            if (
                cross
                and event_time is not None
                and cross_key not in seen_ma_cross_events
            ):
                seen_ma_cross_events.add(cross_key)
                if (
                    target200 is not None
                    and _target_ahead(cross, float(state["close"]), target200)
                    and at + timedelta(minutes=h200) <= TEST_END
                ):
                    outcome = _future_target(
                        m1, m1_available, at, cross, float(target200), h200
                    )
                    ladder_ma_cross_14_50[tf_key].append({
                        "time": at,
                        "event_bar_time": event_time,
                        "side": cross,
                        "target": float(target200),
                        **outcome,
                    })

            acc = int(state.get("accepted_200", 0))
            target1000 = state["ema"].get(1000)
            h1000 = int(target_1000_minutes[tf_key])
            acc_key = (tf_key, event_time, acc)
            if (
                acc
                and event_time is not None
                and acc_key not in seen_acceptance_events
            ):
                seen_acceptance_events.add(acc_key)
                if (
                    target1000 is not None
                    and _target_ahead(acc, float(state["close"]), target1000)
                    and at + timedelta(minutes=h1000) <= TEST_END
                ):
                    outcome = _future_target(
                        m1, m1_available, at, acc, float(target1000), h1000
                    )
                    ladder_200_1000[tf_key].append({
                        "time": at,
                        "event_bar_time": event_time,
                        "side": acc,
                        "target": float(target1000),
                        **outcome,
                    })

        for p, bucket in ((200, rejection_200), (1000, rejection_1000)):
            side = int(m5["barrier_rejection"].get(p, 0))
            level = m5["ema"].get(p)
            if side and level is not None and float(m5["atr"]) > 0:
                outcome = _future_rejection(
                    m1, m1_available, at, side, float(m5["close"]), float(level), float(m5["atr"])
                )
                bucket.append({
                    "time": at,
                    "side": side,
                    "barrier": float(level),
                    **outcome,
                })

        if at < busy_until:
            diagnostics["scans_while_position_open"] += 1
            continue

        m5_window = legacy.bars_window(bars[XAUTimeframe.M5], available[XAUTimeframe.M5], at, 320)
        vp = _volume_profile(m5_window)
        try:
            macro = macro_provider(at) or {}
        except Exception:
            macro = {"bias": 0}
            diagnostics["macro_runtime_failures"] += 1
        macro_bias = int(max(-1, min(1, int(macro.get("bias", 0) or 0))))

        setup = _select_setup(states, vp, macro_bias, _session(at))
        if setup is None:
            continue
        diagnostics["qualified_setups"] += 1
        diagnostics[f"qualified_{setup['setup_type']}"] += 1

        trade = _simulate(
            m1=m1,
            m1_available=m1_available,
            quotes=quotes,
            quote_times=quote_times,
            at=at,
            setup=setup,
            m5=m5,
        )
        if trade is None:
            diagnostics["execution_rejections"] += 1
            continue
        trades.append(trade)
        busy_until = trade["exit_time"]
        diagnostics["executed_trades"] += 1

    by_setup = {
        name: _metrics([t for t in trades if t["setup_type"] == name])
        for name in (
            "ladder_price_14_50_to_200",
            "ladder_200_to_1000",
            "ma_barrier_rejection",
        )
    }
    by_direction = {
        d: _metrics([t for t in trades if t["direction"] == d])
        for d in ("LONG", "SHORT")
    }
    by_session = {
        s: _metrics([t for t in trades if t["session"] == s])
        for s in ("asia", "london", "london_newyork_overlap", "newyork", "off_session")
    }

    report = {
        "benchmark": "gtg-month-v1",
        "strategy_revision": os.getenv("GITHUB_SHA") or "unknown",
        "methodology": "docs/GTG_METHOD_V1.md",
        "period": {
            "warmup_start": WARMUP_START.isoformat(),
            "test_start": TEST_START.isoformat(),
            "test_end": TEST_END.isoformat(),
            "warmup_excluded_from_performance": True,
        },
        "dataset": dataset,
        "architecture": {
            "ma_periods": list(MA_PERIODS),
            "rsi_period": 14,
            "timeframes": [tf.value for tf in TF_ORDER],
            "scan_step_minutes": STEP_MINUTES,
            "hypothesis_event_deduplication": "one event per closed source-timeframe bar",
            "price_14_50_break_rule": "first close beyond BOTH MA14 and MA50 after prior close was not beyond both in that direction",
            "acceptance_rule": "two consecutive closes beyond MA200 after a prior close on the other side",
            "same_bar_tp_sl_policy": "conservative_stop",
            "one_position_at_a_time": True,
            "lookahead_in_trade_decision": False,
            "outcome_tuning_used": False,
        },
        "layer_coverage": {
            "rsi": "covered",
            "ma_dynamics": "covered",
            "price_structure_supply_demand_order_block": "covered_deterministic_proxy",
            "heatmap": "unavailable_no_historical_book",
            "volume_profile": "covered_with_dukascopy_activity_proxy",
            "footprint": "unavailable_no_historical_centralized_tape",
            "cash_order_flow": "unavailable_no_historical_centralized_tape",
            "macro_economic": macro_diag.get("status", "unknown"),
            "sessions": "covered",
        },
        "hypotheses": {
            "price_break_14_50_to_200_by_timeframe": {
                tf: _event_metrics(rows)
                for tf, rows in ladder_price_14_50.items()
            },
            "secondary_ma14_cross_ma50_to_200_by_timeframe": {
                tf: _event_metrics(rows)
                for tf, rows in ladder_ma_cross_14_50.items()
            },
            "accepted_200_to_1000_by_timeframe": {
                tf: _event_metrics(rows)
                for tf, rows in ladder_200_1000.items()
            },
            "m5_ma200_rejection_one_atr_before_recross_120m": _event_metrics(rejection_200),
            "m5_ma1000_rejection_one_atr_before_recross_120m": _event_metrics(rejection_1000),
        },
        "diagnostics": dict(sorted(diagnostics.items())),
        "overall": _metrics(trades),
        "by_setup": by_setup,
        "by_direction": by_direction,
        "by_session": by_session,
        "macro_proxy": macro_diag,
        "data_integrity_passed": bool(dataset.get("m1_bar_count", 0) > 100_000),
        "edge_proven": False,
        "promotion_status": "research_only",
        "limitations": [
            "This one-month run is an architecture/hypothesis test and cannot prove an edge.",
            "Dukascopy volume is a quoted/tick activity proxy, not centralized global gold volume.",
            "Historical raw heatmap, footprint, delta/CVD and centralized order-book tape are not present and are not fabricated.",
            "Historical macro may use one-day-lagged public proxy series and does not reconstruct every CPI/NFP/FOMC surprise.",
            "MA1000 is evaluated only where the warm-up provides enough closed bars; Daily MA1000 is outside this two-year dataset.",
        ],
    }

    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8")
    _write_csv(OUT / "trades.csv", trades)
    for tf, rows in ladder_price_14_50.items():
        _write_csv(OUT / f"hypothesis_price_14_50_to_200_{tf}.csv", rows)
    for tf, rows in ladder_ma_cross_14_50.items():
        _write_csv(OUT / f"hypothesis_ma_cross_14_50_to_200_{tf}.csv", rows)
    for tf, rows in ladder_200_1000.items():
        _write_csv(OUT / f"hypothesis_200_to_1000_{tf}.csv", rows)
    _write_csv(OUT / "rejection_200.csv", rejection_200)
    _write_csv(OUT / "rejection_1000.csv", rejection_1000)
    (OUT / "SUMMARY.md").write_text(
        "\n".join([
            "# GTG v1.0 — One-Month Historical Core",
            "",
            f"- Test: {TEST_START.date()} to {TEST_END.date()}",
            f"- Trades: {report['overall'].get('trade_count', 0)}",
            f"- Win rate: {report['overall'].get('win_rate')}",
            f"- Mean net bps: {report['overall'].get('mean_net_bps')}",
            f"- Profit factor: {report['overall'].get('profit_factor')}",
            f"- Max drawdown bps: {report['overall'].get('max_drawdown_bps')}",
            "",
            f"- PRICE breaks 14/50 -> 200 by timeframe: {report['hypotheses']['price_break_14_50_to_200_by_timeframe']}",
            f"- accepted 200 -> 1000 by timeframe: {report['hypotheses']['accepted_200_to_1000_by_timeframe']}",
            f"- MA200 rejection: {report['hypotheses']['m5_ma200_rejection_one_atr_before_recross_120m']}",
            f"- MA1000 rejection: {report['hypotheses']['m5_ma1000_rejection_one_atr_before_recross_120m']}",
            "",
            "Research-only; missing historical microstructure is explicitly excluded rather than synthesized.",
        ]),
        encoding="utf-8",
    )

    print(json.dumps({
        "benchmark": report["benchmark"],
        "period": report["period"],
        "layer_coverage": report["layer_coverage"],
        "hypotheses": report["hypotheses"],
        "overall": report["overall"],
        "by_setup": report["by_setup"],
        "by_direction": report["by_direction"],
        "by_session": report["by_session"],
        "diagnostics": report["diagnostics"],
        "artifact_dir": str(OUT),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
