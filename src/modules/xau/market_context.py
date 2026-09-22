"""Higher-timeframe XAU market context.

Builds structural bias, EMA ladder, tick-volume profile, liquidity map and
order-flow proxies from research-only XAUUSD bars. Spot XAUUSD is OTC, so
Biquote/MT5 tick volume is explicitly treated as activity proxy rather than
centralized traded volume.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from datetime import datetime
from math import isfinite
from statistics import median
from typing import Any

from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
EMA_PERIODS = (9, 21, 50, 200, 1000)


def _ema_last(values: list[float], period: int) -> float | None:
    if len(values) < period or period < 1:
        return None
    seed = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    current = seed
    for value in values[period:]:
        current = (value - current) * alpha + current
    return current


def _atr(rows: list[XAUBar], period: int = 14) -> float:
    if len(rows) < 2:
        return 0.0
    values: list[float] = []
    for index in range(1, len(rows)):
        cur = rows[index]
        prev = rows[index - 1]
        values.append(
            max(
                cur.high - cur.low,
                abs(cur.high - prev.close),
                abs(cur.low - prev.close),
            )
        )
    window = values[-period:]
    return sum(window) / len(window) if window else 0.0


def aggregate_bars(rows: list[XAUBar], timeframe: XAUTimeframe) -> list[XAUBar]:
    if timeframe not in {XAUTimeframe.W1, XAUTimeframe.MN1}:
        raise ValueError("aggregate_bars supports weekly or monthly only")
    groups: dict[tuple[int, int], list[XAUBar]] = defaultdict(list)
    for row in sorted(rows, key=lambda x: x.timestamp):
        if timeframe == XAUTimeframe.W1:
            iso = row.timestamp.isocalendar()
            key = (iso.year, iso.week)
        else:
            key = (row.timestamp.year, row.timestamp.month)
        groups[key].append(row)

    out: list[XAUBar] = []
    for group_rows in groups.values():
        first = group_rows[0]
        last = group_rows[-1]
        volume_values = [float(r.volume or 0.0) for r in group_rows]
        out.append(
            XAUBar(
                timestamp=last.timestamp,
                timeframe=timeframe,
                open=first.open,
                high=max(r.high for r in group_rows),
                low=min(r.low for r in group_rows),
                close=last.close,
                volume=sum(volume_values),
                source=f"{last.source}:aggregated-{timeframe.value}",
                symbol=last.symbol,
                execution_eligible=False,
            )
        )
    return out


def timeframe_bias(rows: list[XAUBar], name: str) -> dict[str, Any]:
    if not rows:
        return {"timeframe": name, "direction": "neutral", "score": 0.0, "available": False}
    closes = [float(row.close) for row in rows]
    latest = closes[-1]
    emas = {str(period): _ema_last(closes, period) for period in EMA_PERIODS}

    components: list[float] = []
    weights: list[float] = []
    for period, weight in ((9, 0.08), (21, 0.12), (50, 0.20), (200, 0.28), (1000, 0.32)):
        value = emas[str(period)]
        if value is None:
            continue
        distance = (latest - value) / value if value else 0.0
        components.append(max(-1.0, min(1.0, distance / 0.012)))
        weights.append(weight)

    ladder = 0.0
    ladder_weight = 0.0
    pairs = ((9, 21, 0.08), (21, 50, 0.12), (50, 200, 0.24), (200, 1000, 0.32))
    for fast, slow, weight in pairs:
        a = emas[str(fast)]
        b = emas[str(slow)]
        if a is None or b is None:
            continue
        ladder += weight * (1.0 if a > b else -1.0 if a < b else 0.0)
        ladder_weight += weight

    positional = (
        sum(value * weight for value, weight in zip(components, weights)) / sum(weights)
        if weights else 0.0
    )
    ladder_score = ladder / ladder_weight if ladder_weight else 0.0

    lookback = min(20, max(2, len(closes) - 1))
    slope = 0.0
    if len(closes) > lookback and closes[-lookback - 1]:
        slope = (closes[-1] - closes[-lookback - 1]) / closes[-lookback - 1]
    slope_score = max(-1.0, min(1.0, slope / 0.04))

    score = max(-1.0, min(1.0, 0.52 * positional + 0.33 * ladder_score + 0.15 * slope_score))
    direction = "bullish" if score >= 0.15 else "bearish" if score <= -0.15 else "neutral"
    return {
        "timeframe": name,
        "direction": direction,
        "score": round(score, 4),
        "close": round(latest, 4),
        "ema": {key: round(value, 4) if value is not None else None for key, value in emas.items()},
        "slope_20": round(slope, 6),
        "available": True,
        "bar_count": len(rows),
        "observed_at": rows[-1].timestamp.isoformat(),
        "source": rows[-1].source,
    }


def volume_profile(rows: list[XAUBar], bins: int = 40, value_area: float = 0.70) -> dict[str, Any]:
    usable = [r for r in rows if r.volume is not None and float(r.volume or 0.0) > 0]
    if len(usable) < 10:
        return {"available": False, "reason": "insufficient_tick_volume"}
    low = min(r.low for r in usable)
    high = max(r.high for r in usable)
    if high <= low:
        return {"available": False, "reason": "flat_range"}
    bins = max(20, min(int(bins), 80))
    step = (high - low) / bins
    profile = [0.0] * bins

    for row in usable:
        volume = float(row.volume or 0.0)
        start = max(0, min(bins - 1, int((row.low - low) / step)))
        end = max(0, min(bins - 1, int((row.high - low) / step)))
        count = max(1, end - start + 1)
        share = volume / count
        for index in range(start, end + 1):
            profile[index] += share

    centers = [low + (i + 0.5) * step for i in range(bins)]
    poc_index = max(range(bins), key=lambda i: profile[i])
    total = sum(profile)
    target = total * value_area
    selected = {poc_index}
    accumulated = profile[poc_index]
    left = poc_index - 1
    right = poc_index + 1
    while accumulated < target and (left >= 0 or right < bins):
        left_value = profile[left] if left >= 0 else -1.0
        right_value = profile[right] if right < bins else -1.0
        if right_value >= left_value:
            selected.add(right)
            accumulated += max(0.0, right_value)
            right += 1
        else:
            selected.add(left)
            accumulated += max(0.0, left_value)
            left -= 1

    hvn = sorted(range(bins), key=lambda i: profile[i], reverse=True)[:4]
    positive = [i for i, value in enumerate(profile) if value > 0]
    lvn = sorted(positive, key=lambda i: profile[i])[:4]
    latest = float(usable[-1].close)
    vah = max(centers[i] for i in selected)
    val = min(centers[i] for i in selected)
    return {
        "available": True,
        "poc": round(centers[poc_index], 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "location": "above_value" if latest > vah else "below_value" if latest < val else "inside_value",
        "hvn": [round(centers[i], 4) for i in hvn],
        "lvn": [round(centers[i], 4) for i in lvn],
        "range_low": round(low, 4),
        "range_high": round(high, 4),
        "total_tick_volume": round(total, 2),
        "bins": [
            {
                "price": round(centers[i], 4),
                "volume": round(profile[i], 2),
                "share": round(profile[i] / total, 6) if total else 0.0,
            }
            for i in range(bins)
        ],
        "source_type": "mt5_tick_volume_proxy",
        "centralized_volume": False,
    }


def cash_flow(rows: list[XAUBar], period: int = 20) -> dict[str, Any]:
    usable = [r for r in rows if r.volume is not None and float(r.volume or 0.0) > 0]
    if len(usable) < period:
        return {"available": False, "reason": "insufficient_tick_volume"}
    window = usable[-period:]
    mfv_sum = 0.0
    volume_sum = 0.0
    signed = 0.0
    obv = 0.0
    obv_path: list[float] = []
    previous_close = window[0].close
    for row in window:
        volume = float(row.volume or 0.0)
        rng = row.high - row.low
        multiplier = (((row.close - row.low) - (row.high - row.close)) / rng) if rng > 0 else 0.0
        mfv_sum += multiplier * volume
        volume_sum += volume
        signed += volume * (1.0 if row.close > row.open else -1.0 if row.close < row.open else 0.0)
        if row.close > previous_close:
            obv += volume
        elif row.close < previous_close:
            obv -= volume
        obv_path.append(obv)
        previous_close = row.close

    cmf = mfv_sum / volume_sum if volume_sum else 0.0
    imbalance = signed / volume_sum if volume_sum else 0.0
    obv_slope = 0.0
    if len(obv_path) >= 2:
        denom = max(1.0, sum(abs(float(r.volume or 0.0)) for r in window))
        obv_slope = (obv_path[-1] - obv_path[0]) / denom

    score = max(-1.0, min(1.0, 0.55 * cmf + 0.30 * imbalance + 0.15 * obv_slope))
    direction = "inflow" if score >= 0.08 else "outflow" if score <= -0.08 else "balanced"
    return {
        "available": True,
        "direction": direction,
        "score": round(score, 4),
        "cmf20": round(cmf, 4),
        "signed_tick_volume_imbalance": round(imbalance, 4),
        "obv_slope_proxy": round(obv_slope, 4),
        "source_type": "mt5_tick_volume_proxy",
        "centralized_volume": False,
    }


def liquidity_map(daily: list[XAUBar], hourly: list[XAUBar]) -> dict[str, Any]:
    if not daily:
        return {"available": False}
    weekly = aggregate_bars(daily, XAUTimeframe.W1)
    monthly = aggregate_bars(daily, XAUTimeframe.MN1)
    current_price = hourly[-1].close if hourly else daily[-1].close

    def previous(rows: list[XAUBar]) -> XAUBar | None:
        return rows[-2] if len(rows) >= 2 else None

    pd = previous(daily)
    pw = previous(weekly)
    pm = previous(monthly)
    levels: list[dict[str, Any]] = []
    for name, row in (("PDH", pd), ("PDL", pd), ("PWH", pw), ("PWL", pw), ("PMH", pm), ("PML", pm)):
        if row is None:
            continue
        price = row.high if name.endswith("H") else row.low
        levels.append({
            "name": name,
            "price": round(price, 4),
            "side": "above" if price > current_price else "below",
            "distance": round(abs(price - current_price), 4),
        })

    equal_highs: list[float] = []
    equal_lows: list[float] = []
    recent = hourly[-120:]
    atr = _atr(recent, 14)
    tolerance = max(0.15, atr * 0.12)
    pivots_high = [r.high for r in recent]
    pivots_low = [r.low for r in recent]
    for values, output in ((pivots_high, equal_highs), (pivots_low, equal_lows)):
        for i in range(len(values)):
            matches = [v for j, v in enumerate(values) if j != i and abs(v - values[i]) <= tolerance]
            if matches:
                center = (values[i] + sum(matches)) / (1 + len(matches))
                if all(abs(center - existing) > tolerance for existing in output):
                    output.append(round(center, 4))
    equal_highs = sorted(equal_highs, key=lambda x: abs(x - current_price))[:4]
    equal_lows = sorted(equal_lows, key=lambda x: abs(x - current_price))[:4]
    return {
        "available": True,
        "reference_price": round(current_price, 4),
        "levels": sorted(levels, key=lambda item: item["distance"]),
        "equal_highs": equal_highs,
        "equal_lows": equal_lows,
        "tolerance": round(tolerance, 4),
    }


def smart_money_structure(hourly: list[XAUBar], h4: list[XAUBar], liquidity: dict[str, Any], profile: dict[str, Any], flow: dict[str, Any]) -> dict[str, Any]:
    if len(hourly) < 30:
        return {"available": False}
    recent = hourly[-40:]
    latest = recent[-1]
    atr = _atr(recent, 14)
    prior = recent[-21:-1]
    prior_high = max(r.high for r in prior)
    prior_low = min(r.low for r in prior)
    bos = "bullish" if latest.close > prior_high else "bearish" if latest.close < prior_low else "none"

    sweep = "none"
    if latest.high > prior_high and latest.close < prior_high:
        sweep = "buy_side_sweep"
    elif latest.low < prior_low and latest.close > prior_low:
        sweep = "sell_side_sweep"

    volumes = [float(r.volume or 0.0) for r in recent if r.volume is not None]
    med_volume = median(volumes) if volumes else 0.0
    displacement = "none"
    if atr > 0 and (latest.high - latest.low) >= 1.5 * atr and float(latest.volume or 0.0) >= 1.35 * med_volume:
        displacement = "bullish" if latest.close > latest.open else "bearish" if latest.close < latest.open else "none"

    fvg: list[dict[str, Any]] = []
    for i in range(max(2, len(recent) - 20), len(recent)):
        left = recent[i - 2]
        cur = recent[i]
        if cur.low > left.high:
            fvg.append({"direction": "bullish", "low": round(left.high, 4), "high": round(cur.low, 4), "time": cur.timestamp.isoformat()})
        elif cur.high < left.low:
            fvg.append({"direction": "bearish", "low": round(cur.high, 4), "high": round(left.low, 4), "time": cur.timestamp.isoformat()})
    fvg = fvg[-5:]

    h4_recent = h4[-30:] if h4 else []
    range_high = max((r.high for r in h4_recent), default=latest.high)
    range_low = min((r.low for r in h4_recent), default=latest.low)
    midpoint = (range_high + range_low) / 2.0
    dealing_zone = "premium" if latest.close > midpoint else "discount" if latest.close < midpoint else "equilibrium"

    score = 0.0
    if bos == "bullish":
        score += 0.30
    elif bos == "bearish":
        score -= 0.30
    if displacement == "bullish":
        score += 0.22
    elif displacement == "bearish":
        score -= 0.22
    if sweep == "sell_side_sweep":
        score += 0.18
    elif sweep == "buy_side_sweep":
        score -= 0.18
    score += 0.20 * float(flow.get("score") or 0.0)
    if profile.get("available"):
        poc = float(profile.get("poc") or latest.close)
        if latest.close > poc:
            score += 0.10
        elif latest.close < poc:
            score -= 0.10
    score = max(-1.0, min(1.0, score))

    return {
        "available": True,
        "bias": "bullish" if score >= 0.15 else "bearish" if score <= -0.15 else "neutral",
        "score": round(score, 4),
        "break_of_structure": bos,
        "liquidity_sweep": sweep,
        "displacement": displacement,
        "dealing_range": {
            "high": round(range_high, 4),
            "low": round(range_low, 4),
            "midpoint": round(midpoint, 4),
            "zone": dealing_zone,
        },
        "fair_value_gaps": fvg,
        "liquidity": liquidity,
    }


def build_market_context(
    hourly: list[XAUBar],
    h4: list[XAUBar],
    daily: list[XAUBar],
    futures_hourly: list[XAUBar] | None = None,
) -> dict[str, Any]:
    weekly = aggregate_bars(daily, XAUTimeframe.W1)
    monthly = aggregate_bars(daily, XAUTimeframe.MN1)

    biases = {
        "1h": timeframe_bias(hourly, "1h"),
        "4h": timeframe_bias(h4, "4h"),
        "1d": timeframe_bias(daily, "1d"),
        "1w": timeframe_bias(weekly, "1w"),
        "1mo": timeframe_bias(monthly, "1mo"),
    }

    htf_weights = {"1mo": 0.34, "1w": 0.30, "1d": 0.24, "4h": 0.08, "1h": 0.04}
    weighted = 0.0
    available_weight = 0.0
    for name, weight in htf_weights.items():
        state = biases[name]
        if not state.get("available"):
            continue
        weighted += weight * float(state.get("score") or 0.0)
        available_weight += weight
    composite = weighted / available_weight if available_weight else 0.0
    composite = max(-1.0, min(1.0, composite))
    composite_direction = "bullish" if composite >= 0.15 else "bearish" if composite <= -0.15 else "neutral"

    profile = volume_profile(hourly[-160:])
    flow = cash_flow(hourly, 20)
    futures_flow = cash_flow(futures_hourly or [], 20)
    liquidity = liquidity_map(daily, hourly)

    spot_flow_score = float(flow.get("score") or 0.0)
    futures_flow_score = float(futures_flow.get("score") or 0.0) if futures_flow.get("available") else 0.0
    if futures_flow.get("available") and flow.get("available"):
        combined_flow_score = 0.55 * spot_flow_score + 0.45 * futures_flow_score
        flow_agreement = (
            "aligned"
            if (spot_flow_score == 0 or futures_flow_score == 0 or spot_flow_score * futures_flow_score > 0)
            else "divergent"
        )
    elif futures_flow.get("available"):
        combined_flow_score = futures_flow_score
        flow_agreement = "futures_only"
    else:
        combined_flow_score = spot_flow_score
        flow_agreement = "spot_tick_only"

    combined_flow = {
        "available": bool(flow.get("available") or futures_flow.get("available")),
        "score": round(max(-1.0, min(1.0, combined_flow_score)), 4),
        "direction": "inflow" if combined_flow_score >= 0.08 else "outflow" if combined_flow_score <= -0.08 else "balanced",
        "agreement": flow_agreement,
        "spot_tick": flow,
        "gc_futures": futures_flow,
    }

    smart_money = smart_money_structure(hourly, h4, liquidity, profile, combined_flow)

    # Library implementations are deliberately NOT fused here. They observe the
    # same OHLC input, so counting them as independent evidence would inflate
    # confidence. service._refresh_market_context attaches them as validation
    # oracles and cognition uses agreement only as a reliability guard.

    smart_score = float(smart_money.get("score") or 0.0)
    smart_money["manual_score"] = round(smart_score, 4)
    smart_money["validation_policy"] = "same-input libraries validate; they do not add directional evidence"

    today_score = 0.54 * composite + 0.28 * smart_score + 0.18 * combined_flow_score
    today_score = max(-1.0, min(1.0, today_score))
    today_direction = "bullish" if today_score >= 0.15 else "bearish" if today_score <= -0.15 else "neutral"

    return {
        "version": "htf-context-v3-validator-separation",
        "bias": {
            "monthly": biases["1mo"],
            "weekly": biases["1w"],
            "daily": biases["1d"],
            "h4": biases["4h"],
            "h1": biases["1h"],
            "composite_score": round(composite, 4),
            "composite_direction": composite_direction,
            "today_score": round(today_score, 4),
            "today_direction": today_direction,
        },
        "ema_ladder": {
            name: state.get("ema", {})
            for name, state in biases.items()
        },
        "volume_profile": profile,
        "cash_flow": combined_flow,
        "spot_tick_flow": flow,
        "futures_flow": futures_flow,
        "liquidity": liquidity,
        "smart_money": smart_money,
        "volume_note": (
            "XAUUSD is OTC; MT5 tick volume is an activity proxy, not centralized exchange volume. "
            "GC futures volume is used only as a cross-market participation proxy. Neither is treated "
            "as literal global spot order flow."
        ),
    }
