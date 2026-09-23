"""Gen1 microstructure analytics for the free Bitfinex XAUT/USD sensor."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from typing import Any

from src.platform.marketdata.xaut_bitfinex import XAUTMicrostructureSnapshot, XAUTTrade


def _round_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 8)


def _window_flow(trades: list[XAUTTrade], *, minutes: int) -> dict[str, Any]:
    if not trades:
        return {"available": False, "minutes": minutes, "reason": "no_trades"}
    end = trades[-1].timestamp
    start = end - timedelta(minutes=minutes)
    window = [trade for trade in trades if trade.timestamp >= start]
    if not window:
        return {"available": False, "minutes": minutes, "reason": "no_window_trades"}
    buy = sum(t.size for t in window if t.amount > 0)
    sell = sum(t.size for t in window if t.amount < 0)
    total = buy + sell
    delta = buy - sell
    notional = sum(t.size * t.price for t in window)
    first = window[0].price
    last = window[-1].price
    return {
        "available": True,
        "minutes": minutes,
        "trade_count": len(window),
        "buy_volume": round(buy, 6),
        "sell_volume": round(sell, 6),
        "delta": round(delta, 6),
        "delta_ratio": round(delta / total, 6) if total else 0.0,
        "buy_share": round(buy / total, 6) if total else 0.0,
        "notional_usd": round(notional, 2),
        "price_change": round(last - first, 6),
        "price_change_bps": round(((last - first) / first) * 10_000.0, 4) if first else 0.0,
    }


def _cvd(trades: list[XAUTTrade]) -> dict[str, Any]:
    if not trades:
        return {"available": False}
    value = 0.0
    path: list[tuple[XAUTTrade, float]] = []
    for trade in trades:
        value += trade.size if trade.amount > 0 else -trade.size
        path.append((trade, value))
    latest = path[-1][1]
    end = path[-1][0].timestamp
    changes = {}
    for minutes in (1, 5, 15, 30):
        cutoff = end - timedelta(minutes=minutes)
        base = next((v for t, v in path if t.timestamp >= cutoff), path[0][1])
        changes[f"change_{minutes}m"] = round(latest - base, 6)
    return {
        "available": True,
        "value": round(latest, 6),
        **changes,
    }


def _footprint(trades: list[XAUTTrade], *, tick: float = 0.1) -> dict[str, Any]:
    levels: dict[float, dict[str, float]] = defaultdict(lambda: {"ask_volume": 0.0, "bid_volume": 0.0})
    for trade in trades:
        price = _round_tick(trade.price, tick)
        if trade.amount > 0:
            # Aggressive buy executes at/through ask.
            levels[price]["ask_volume"] += trade.size
        else:
            # Aggressive sell executes at/through bid.
            levels[price]["bid_volume"] += trade.size
    rows = []
    for price in sorted(levels):
        ask = levels[price]["ask_volume"]
        bid = levels[price]["bid_volume"]
        total = ask + bid
        delta = ask - bid
        same_price_ratio = (
            ask / bid if ask > bid and bid > 0
            else bid / ask if bid > ask and ask > 0
            else float("inf") if total > 0 and min(ask, bid) == 0
            else 1.0
        )
        imbalance = "none"
        if total > 0 and ask > 0 and (bid == 0 or ask >= 3.0 * bid):
            imbalance = "buy"
        elif total > 0 and bid > 0 and (ask == 0 or bid >= 3.0 * ask):
            imbalance = "sell"
        rows.append({
            "price": round(price, 4),
            "ask_volume": round(ask, 6),
            "bid_volume": round(bid, 6),
            "delta": round(delta, 6),
            "total_volume": round(total, 6),
            "delta_ratio": round(delta / total, 6) if total else 0.0,
            "same_price_imbalance": imbalance,
            "imbalance_ratio": None if not isfinite(same_price_ratio) else round(same_price_ratio, 4),
        })
    ranked = sorted(rows, key=lambda x: x["total_volume"], reverse=True)
    return {
        "available": bool(rows),
        "tick_size": tick,
        "levels": rows,
        "highest_activity": ranked[:12],
        "method": "aggressor-signed-trades",
        "note": "same-price imbalance; not a diagonal CME footprint imbalance",
    }


def _volume_profile(
    trades: list[XAUTTrade],
    *,
    tick: float = 0.5,
    value_area_fraction: float = 0.70,
) -> dict[str, Any]:
    """Build a price-by-executed-volume profile from real XAUT trades.

    This is centralized Bitfinex XAUT volume. It is stronger evidence than an
    MT5 tick-volume proxy for this venue, but it is not global OTC XAUUSD volume.
    """
    if not trades:
        return {"available": False, "status": "unavailable", "reason": "no_trades"}

    tick = max(0.01, float(tick))
    value_area_fraction = max(0.50, min(float(value_area_fraction), 0.90))
    buckets: dict[float, dict[str, float]] = defaultdict(
        lambda: {"volume": 0.0, "buy_volume": 0.0, "sell_volume": 0.0, "notional": 0.0}
    )
    for trade in trades:
        price = _round_tick(trade.price, tick)
        row = buckets[price]
        row["volume"] += trade.size
        row["notional"] += trade.size * trade.price
        if trade.amount > 0:
            row["buy_volume"] += trade.size
        else:
            row["sell_volume"] += trade.size

    prices = sorted(buckets)
    total_volume = sum(buckets[price]["volume"] for price in prices)
    total_notional = sum(buckets[price]["notional"] for price in prices)
    if total_volume <= 0:
        return {"available": False, "status": "unavailable", "reason": "zero_volume"}

    latest_price = trades[-1].price
    poc_price = max(
        prices,
        key=lambda price: (
            buckets[price]["volume"],
            -abs(price - latest_price),
            price,
        ),
    )
    poc_idx = prices.index(poc_price)
    included = {poc_idx}
    accumulated = buckets[poc_price]["volume"]
    target = total_volume * value_area_fraction
    left = poc_idx - 1
    right = poc_idx + 1
    while accumulated < target and (left >= 0 or right < len(prices)):
        left_volume = buckets[prices[left]]["volume"] if left >= 0 else -1.0
        right_volume = buckets[prices[right]]["volume"] if right < len(prices) else -1.0
        if right_volume > left_volume:
            included.add(right)
            accumulated += max(0.0, right_volume)
            right += 1
        else:
            included.add(left)
            accumulated += max(0.0, left_volume)
            left -= 1

    val = prices[min(included)]
    vah = prices[max(included)]
    location = (
        "above_value"
        if latest_price > vah
        else "below_value"
        if latest_price < val
        else "inside_value"
    )

    levels = []
    for price in prices:
        row = buckets[price]
        volume = row["volume"]
        buy = row["buy_volume"]
        sell = row["sell_volume"]
        levels.append({
            "price": round(price, 4),
            "volume": round(volume, 6),
            "share": round(volume / total_volume, 6),
            "buy_volume": round(buy, 6),
            "sell_volume": round(sell, 6),
            "delta": round(buy - sell, 6),
            "in_value_area": val <= price <= vah,
        })

    ranked_high = sorted(
        levels,
        key=lambda row: (row["volume"], -abs(row["price"] - poc_price)),
        reverse=True,
    )
    ranked_low = sorted(
        [row for row in levels if row["volume"] > 0],
        key=lambda row: (row["volume"], abs(row["price"] - poc_price)),
    )
    coverage_seconds = (
        (trades[-1].timestamp - trades[0].timestamp).total_seconds()
        if len(trades) > 1
        else 0.0
    )

    return {
        "available": True,
        "status": "ready",
        "poc": round(poc_price, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "location": location,
        "tick_size": tick,
        "value_area_fraction": round(value_area_fraction, 3),
        "value_area_volume_share": round(accumulated / total_volume, 6),
        "total_volume_xaut": round(total_volume, 6),
        "total_notional_usd": round(total_notional, 2),
        "trade_count": len(trades),
        "coverage_seconds": round(coverage_seconds, 3),
        "high_volume_nodes": ranked_high[:8],
        "low_volume_nodes": ranked_low[:8],
        "levels": levels,
        "volume_kind": "executed_xaut_volume",
        "source": "bitfinex:XAUTUSD:executed_trades",
        "centralized_proxy_market": True,
        "global_xauusd_volume_profile": False,
        "note": "real executed XAUT volume profile; not global OTC XAUUSD volume",
    }


def _session_name(value: datetime) -> str:
    hour = value.astimezone(timezone.utc).hour
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 13:
        return "london"
    if 13 <= hour < 21:
        return "new_york"
    return "off_hours"


def _rolling_profiles(
    trades: list[XAUTTrade],
    *,
    tick: float,
    value_area_fraction: float,
) -> dict[str, Any]:
    if not trades:
        return {}
    end = trades[-1].timestamp
    out: dict[str, Any] = {}
    for minutes in (15, 30, 60):
        cutoff = end - timedelta(minutes=minutes)
        subset = [trade for trade in trades if trade.timestamp >= cutoff]
        profile = _volume_profile(
            subset,
            tick=tick,
            value_area_fraction=value_area_fraction,
        )
        profile = dict(profile)
        profile["requested_minutes"] = minutes
        profile["coverage_complete"] = bool(trades and trades[0].timestamp <= cutoff)
        profile["window_start"] = cutoff.isoformat()
        profile["window_end"] = end.isoformat()
        out[f"{minutes}m"] = profile
    return out


def _session_profiles(
    trades: list[XAUTTrade],
    *,
    tick: float,
    value_area_fraction: float,
) -> dict[str, Any]:
    if not trades:
        return {"available": False, "current_session": None, "profiles": []}

    groups: dict[tuple[str, str], list[XAUTTrade]] = defaultdict(list)
    for trade in trades:
        utc_time = trade.timestamp.astimezone(timezone.utc)
        groups[(utc_time.date().isoformat(), _session_name(utc_time))].append(trade)

    latest = trades[-1].timestamp.astimezone(timezone.utc)
    current_key = (latest.date().isoformat(), _session_name(latest))
    rows = []
    for (date_key, session), subset in sorted(groups.items()):
        profile = _volume_profile(
            subset,
            tick=tick,
            value_area_fraction=value_area_fraction,
        )
        profile = dict(profile)
        profile.update({
            "date_utc": date_key,
            "session": session,
            "partial_tape": True,
            "first_trade_at": subset[0].timestamp.isoformat(),
            "last_trade_at": subset[-1].timestamp.isoformat(),
        })
        rows.append(profile)

    current = next(
        (
            row for row in rows
            if row.get("date_utc") == current_key[0] and row.get("session") == current_key[1]
        ),
        None,
    )
    return {
        "available": bool(rows),
        "session_clock": "UTC dominant non-overlapping windows",
        "windows_utc": {
            "asia": "00:00-08:00",
            "london": "08:00-13:00",
            "new_york": "13:00-21:00",
            "off_hours": "21:00-24:00",
        },
        "current_session": current_key[1],
        "current": current,
        "profiles": rows,
        "note": "Profiles cover only trades present in the XAUT tape; partial_tape=true is not a full exchange session claim.",
    }


def _book(snapshot: XAUTMicrostructureSnapshot, distance: float) -> dict[str, Any]:
    mid = snapshot.mid
    lower, upper = mid - distance, mid + distance
    nearby = [o for o in snapshot.raw_book if lower <= o.price <= upper]
    bids = [o for o in nearby if o.amount > 0]
    asks = [o for o in nearby if o.amount < 0]
    bid_qty = sum(o.size for o in bids)
    ask_qty = sum(o.size for o in asks)
    total = bid_qty + ask_qty
    return {
        "distance_usd": distance,
        "bid_order_count": len(bids),
        "ask_order_count": len(asks),
        "bid_quantity": round(bid_qty, 6),
        "ask_quantity": round(ask_qty, 6),
        "imbalance": round((bid_qty - ask_qty) / total, 6) if total else 0.0,
        "largest_bids": [
            {"price": o.price, "size": round(o.size, 6), "order_id": o.order_id}
            for o in sorted(bids, key=lambda x: x.size, reverse=True)[:8]
        ],
        "largest_asks": [
            {"price": o.price, "size": round(o.size, 6), "order_id": o.order_id}
            for o in sorted(asks, key=lambda x: x.size, reverse=True)[:8]
        ],
    }


def _absorption(flow5: dict[str, Any], book10: dict[str, Any]) -> dict[str, Any]:
    if not flow5.get("available"):
        return {"state": "unavailable"}
    delta_ratio = float(flow5.get("delta_ratio") or 0.0)
    book_imbalance = float(book10.get("imbalance") or 0.0)
    if delta_ratio <= -0.35 and book_imbalance >= 0.25:
        return {
            "state": "possible_sell_absorption",
            "confidence": "candidate_only",
            "reason": "aggressive_selling_meets_bid_heavy_raw_book",
            "requires": ["price_stall_or_reclaim", "book_replenishment_confirmation"],
        }
    if delta_ratio >= 0.35 and book_imbalance <= -0.25:
        return {
            "state": "possible_buy_absorption",
            "confidence": "candidate_only",
            "reason": "aggressive_buying_meets_ask_heavy_raw_book",
            "requires": ["price_stall_or_rejection", "book_replenishment_confirmation"],
        }
    return {"state": "none", "confidence": "candidate_only"}


def analyze_xaut_microstructure(
    snapshot: XAUTMicrostructureSnapshot,
    *,
    xau_spot_price: float | None = None,
    footprint_tick: float = 0.1,
    volume_profile_tick: float = 0.5,
    value_area_fraction: float = 0.70,
) -> dict[str, Any]:
    trades = sorted(snapshot.trades, key=lambda x: x.timestamp)
    latest_trade = trades[-1] if trades else None
    oldest_trade = trades[0] if trades else None
    flows = {f"{m}m": _window_flow(trades, minutes=m) for m in (1, 5, 15, 30)}
    books = {f"pm{int(d)}": _book(snapshot, d) for d in (10.0, 20.0, 30.0)}
    basis = None
    basis_bps = None
    forward = None
    if xau_spot_price is not None and xau_spot_price > 0:
        basis = float(xau_spot_price) - snapshot.mid
        basis_bps = (basis / float(xau_spot_price)) * 10_000.0
        forward = {
            f"pm{distance}": {
                "xau_up": round(float(xau_spot_price) + distance, 4),
                "xau_down": round(float(xau_spot_price) - distance, 4),
                "xaut_equivalent_up": round(snapshot.mid + distance, 4),
                "xaut_equivalent_down": round(snapshot.mid - distance, 4),
            }
            for distance in (10, 20, 30)
        }
    book10 = books["pm10"]
    profile = _volume_profile(
        trades,
        tick=volume_profile_tick,
        value_area_fraction=value_area_fraction,
    )
    rolling_profiles = _rolling_profiles(
        trades,
        tick=volume_profile_tick,
        value_area_fraction=value_area_fraction,
    )
    session_profiles = _session_profiles(
        trades,
        tick=volume_profile_tick,
        value_area_fraction=value_area_fraction,
    )
    if profile.get("available") and basis is not None:
        profile = dict(profile)
        profile["xauusd_mapping"] = {
            "available": True,
            "method": "instantaneous_xau_minus_xaut_basis",
            "basis": round(basis, 6),
            "poc": round(float(profile["poc"]) + basis, 4),
            "vah": round(float(profile["vah"]) + basis, 4),
            "val": round(float(profile["val"]) + basis, 4),
            "execution_eligible": False,
        }
    else:
        profile = dict(profile)
        profile["xauusd_mapping"] = {
            "available": False,
            "method": "instantaneous_xau_minus_xaut_basis",
            "execution_eligible": False,
        }
    return {
        "version": "xaut-free-order-flow-v1",
        "status": "ready",
        "source": snapshot.source,
        "proxy_symbol": snapshot.symbol,
        "reference_symbol": snapshot.reference_symbol,
        "execution_eligible": False,
        "centralized_proxy_market": True,
        "global_xauusd_order_flow": False,
        "observed_at": snapshot.observed_at.isoformat(),
        "trade_tape": {
            "trade_count": len(trades),
            "latest_trade_at": latest_trade.timestamp.isoformat() if latest_trade else None,
            "oldest_trade_at": oldest_trade.timestamp.isoformat() if oldest_trade else None,
            "coverage_seconds": (
                round((latest_trade.timestamp - oldest_trade.timestamp).total_seconds(), 3)
                if latest_trade and oldest_trade else None
            ),
        },
        "quote": {
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "mid": round(snapshot.mid, 6),
            "last": snapshot.last,
            "spread": round(snapshot.spread, 6),
        },
        "basis": {
            "available": basis is not None,
            "xau_minus_xaut": round(basis, 6) if basis is not None else None,
            "basis_bps": round(basis_bps, 4) if basis_bps is not None else None,
            "mapping": "instantaneous_offset_only",
        },
        "flow": flows,
        "cvd": _cvd(trades),
        "footprint": _footprint(trades, tick=footprint_tick),
        "volume_profile": profile,
        "volume_profiles": {
            "tape": profile,
            "rolling": rolling_profiles,
            "session": session_profiles,
        },
        "raw_book": books,
        "absorption": _absorption(flows["5m"], book10),
        "forward_range_map": forward,
        "evidence_policy": {
            "trade_sign_is_real_for_xaut": True,
            "raw_book_is_real_for_xaut": True,
            "volume_profile_is_real_executed_xaut_volume": True,
            "xaut_volume_profile_is_global_xauusd_volume": False,
            "xaut_is_xauusd_execution_venue": False,
            "xaut_flow_is_global_spot_gold_flow": False,
            "basis_must_be_recomputed": True,
        },
    }
