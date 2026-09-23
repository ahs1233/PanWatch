"""Multi-venue gold microstructure fusion for GEN1.

Important evidence rule: venue volumes are NEVER summed and described as
"global XAU/USD volume".  Each venue is analyzed independently and only
normalized signals are fused with dynamic reliability weights.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import isfinite, log10
from statistics import median
from typing import Any, Iterable

from src.platform.marketdata.gold_okx import OKXGoldSnapshot


WINDOWS_MINUTES = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
    "1w": 10080,
}
PROFILE_WINDOWS = ("1h", "4h", "1d", "1w")


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _round_tick(price: float, tick: float) -> float:
    return round(round(float(price) / float(tick)) * float(tick), 8)


def _trade_side(trade: Any) -> str:
    side = getattr(trade, "aggressor_side", None)
    if side:
        return str(side)
    amount = getattr(trade, "amount", None)
    return "buy" if float(amount or 0.0) > 0 else "sell"


def _trade_size(trade: Any) -> float:
    value = getattr(trade, "size_xau", None)
    if value is None:
        value = getattr(trade, "size", 0.0)
    return abs(float(value or 0.0))


def _trade_price(trade: Any) -> float:
    return float(getattr(trade, "price"))


def _trade_time(trade: Any) -> datetime:
    value = getattr(trade, "timestamp")
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _window_subset(trades: list[Any], minutes: int) -> tuple[list[Any], datetime | None, datetime | None]:
    if not trades:
        return [], None, None
    end = _trade_time(trades[-1])
    cutoff = end - timedelta(minutes=minutes)
    subset = [t for t in trades if _trade_time(t) >= cutoff]
    return subset, cutoff, end


def _flow(trades: list[Any], *, minutes: int) -> dict[str, Any]:
    subset, cutoff, end = _window_subset(trades, minutes)
    if not subset:
        return {
            "available": False,
            "minutes": minutes,
            "status": "unavailable",
            "decision_eligible": False,
            "coverage_complete": False,
            "reason": "no_window_trades",
        }
    buy = sum(_trade_size(t) for t in subset if _trade_side(t) == "buy")
    sell = sum(_trade_size(t) for t in subset if _trade_side(t) == "sell")
    total = buy + sell
    delta = buy - sell
    notional = sum(_trade_size(t) * _trade_price(t) for t in subset)
    first = _trade_price(subset[0])
    last = _trade_price(subset[-1])
    oldest = _trade_time(trades[0])
    coverage_complete = bool(cutoff and oldest <= cutoff)
    return {
        "available": True,
        "status": "ready" if coverage_complete else "partial",
        "decision_eligible": coverage_complete,
        "minutes": minutes,
        "trade_count": len(subset),
        "buy_volume_xau": round(buy, 8),
        "sell_volume_xau": round(sell, 8),
        "delta_xau": round(delta, 8),
        "delta_ratio": round(delta / total, 6) if total else 0.0,
        "buy_share": round(buy / total, 6) if total else 0.0,
        "notional_usd": round(notional, 2),
        "price_change": round(last - first, 6),
        "price_change_bps": round(((last - first) / first) * 10000.0, 4) if first else 0.0,
        "coverage_complete": coverage_complete,
        "window_start": cutoff.isoformat() if cutoff else None,
        "window_end": end.isoformat() if end else None,
    }


def _footprint(trades: list[Any], *, tick: float) -> dict[str, Any]:
    levels: dict[float, dict[str, float]] = defaultdict(lambda: {"ask": 0.0, "bid": 0.0})
    for t in trades:
        price = _round_tick(_trade_price(t), tick)
        size = _trade_size(t)
        if _trade_side(t) == "buy":
            levels[price]["ask"] += size
        else:
            levels[price]["bid"] += size
    rows: list[dict[str, Any]] = []
    for price in sorted(levels):
        ask = levels[price]["ask"]
        bid = levels[price]["bid"]
        total = ask + bid
        delta = ask - bid
        imbalance = "none"
        if ask > 0 and (bid == 0 or ask >= 3.0 * bid):
            imbalance = "buy"
        elif bid > 0 and (ask == 0 or bid >= 3.0 * ask):
            imbalance = "sell"
        rows.append({
            "price": round(price, 4),
            "ask_volume_xau": round(ask, 8),
            "bid_volume_xau": round(bid, 8),
            "delta_xau": round(delta, 8),
            "total_volume_xau": round(total, 8),
            "delta_ratio": round(delta / total, 6) if total else 0.0,
            "same_price_imbalance": imbalance,
        })
    total = sum(row["total_volume_xau"] for row in rows)
    delta = sum(row["delta_xau"] for row in rows)
    ranked = sorted(rows, key=lambda row: row["total_volume_xau"], reverse=True)
    return {
        "available": bool(rows),
        "method": "aggressor-signed-executed-trades",
        "tick_size": tick,
        "total_volume_xau": round(total, 8),
        "delta_xau": round(delta, 8),
        "delta_ratio": round(delta / total, 6) if total else 0.0,
        "levels": rows,
        "highest_activity": ranked[:12],
        "note": "venue footprint, not CME diagonal footprint and not global OTC XAUUSD",
    }


def _volume_profile(
    trades: list[Any],
    *,
    tick: float,
    value_area_fraction: float = 0.70,
) -> dict[str, Any]:
    if not trades:
        return {"available": False, "status": "unavailable", "reason": "no_trades"}
    buckets: dict[float, dict[str, float]] = defaultdict(
        lambda: {"volume": 0.0, "buy": 0.0, "sell": 0.0, "notional": 0.0}
    )
    for t in trades:
        price = _round_tick(_trade_price(t), tick)
        size = _trade_size(t)
        row = buckets[price]
        row["volume"] += size
        row["notional"] += size * _trade_price(t)
        row["buy" if _trade_side(t) == "buy" else "sell"] += size
    prices = sorted(buckets)
    total = sum(buckets[p]["volume"] for p in prices)
    if total <= 0:
        return {"available": False, "status": "unavailable", "reason": "zero_volume"}
    latest = _trade_price(trades[-1])
    poc = max(prices, key=lambda p: (buckets[p]["volume"], -abs(p - latest)))
    poc_i = prices.index(poc)
    included = {poc_i}
    accumulated = buckets[poc]["volume"]
    target = total * max(0.50, min(float(value_area_fraction), 0.90))
    left, right = poc_i - 1, poc_i + 1
    while accumulated < target and (left >= 0 or right < len(prices)):
        lv = buckets[prices[left]]["volume"] if left >= 0 else -1.0
        rv = buckets[prices[right]]["volume"] if right < len(prices) else -1.0
        if rv > lv:
            included.add(right); accumulated += max(0.0, rv); right += 1
        else:
            included.add(left); accumulated += max(0.0, lv); left -= 1
    val, vah = prices[min(included)], prices[max(included)]
    levels = []
    for p in prices:
        row = buckets[p]
        levels.append({
            "price": round(p, 4),
            "volume_xau": round(row["volume"], 8),
            "share": round(row["volume"] / total, 6),
            "buy_volume_xau": round(row["buy"], 8),
            "sell_volume_xau": round(row["sell"], 8),
            "delta_xau": round(row["buy"] - row["sell"], 8),
            "in_value_area": val <= p <= vah,
        })
    hvn = sorted(levels, key=lambda x: x["volume_xau"], reverse=True)[:8]
    lvn = sorted(levels, key=lambda x: x["volume_xau"])[:8]
    total_buy = sum(buckets[p]["buy"] for p in prices)
    total_sell = sum(buckets[p]["sell"] for p in prices)
    return {
        "available": True,
        "status": "ready",
        "poc": round(poc, 4),
        "vah": round(vah, 4),
        "val": round(val, 4),
        "location": "above_value" if latest > vah else "below_value" if latest < val else "inside_value",
        "tick_size": tick,
        "total_volume_xau": round(total, 8),
        "buy_volume_xau": round(total_buy, 8),
        "sell_volume_xau": round(total_sell, 8),
        "delta_xau": round(total_buy - total_sell, 8),
        "delta_ratio": round((total_buy - total_sell) / total, 6) if total else 0.0,
        "total_notional_usd": round(sum(buckets[p]["notional"] for p in prices), 2),
        "trade_count": len(trades),
        "value_area_fraction": round(value_area_fraction, 3),
        "value_area_volume_share": round(accumulated / total, 6),
        "high_volume_nodes": hvn,
        "low_volume_nodes": lvn,
        "levels": levels,
        "volume_kind": "venue_executed_gold_volume",
        "global_xauusd_volume_profile": False,
    }


def _absorption_candidate(flow: dict[str, Any], book: dict[str, Any]) -> dict[str, Any]:
    if not flow.get("available") or not book.get("available"):
        return {"state": "unavailable", "confidence": "candidate_only"}
    delta_ratio = float(flow.get("delta_ratio") or 0.0)
    book_imbalance = float(book.get("imbalance") or 0.0)
    if delta_ratio <= -0.35 and book_imbalance >= 0.25:
        return {
            "state": "possible_sell_absorption",
            "confidence": "candidate_only",
            "reason": "aggressive_selling_meets_bid_heavy_book",
            "requires": ["price_stall_or_reclaim", "book_replenishment_confirmation"],
        }
    if delta_ratio >= 0.35 and book_imbalance <= -0.25:
        return {
            "state": "possible_buy_absorption",
            "confidence": "candidate_only",
            "reason": "aggressive_buying_meets_ask_heavy_book",
            "requires": ["price_stall_or_rejection", "book_replenishment_confirmation"],
        }
    return {"state": "none", "confidence": "candidate_only"}


def _book(snapshot: OKXGoldSnapshot, *, distance: float = 10.0) -> dict[str, Any]:
    lower, upper = snapshot.mid - distance, snapshot.mid + distance
    bids = [x for x in snapshot.book if x.side == "bid" and lower <= x.price <= upper]
    asks = [x for x in snapshot.book if x.side == "ask" and lower <= x.price <= upper]
    bid = sum(x.size_xau for x in bids)
    ask = sum(x.size_xau for x in asks)
    total = bid + ask
    return {
        "available": bool(bids or asks),
        "distance_usd": distance,
        "bid_volume_xau": round(bid, 8),
        "ask_volume_xau": round(ask, 8),
        "imbalance": round((bid - ask) / total, 6) if total else 0.0,
        "largest_bids": [
            {"price": x.price, "size_xau": round(x.size_xau, 8), "order_count": x.order_count}
            for x in sorted(bids, key=lambda x: x.size_xau, reverse=True)[:8]
        ],
        "largest_asks": [
            {"price": x.price, "size_xau": round(x.size_xau, 8), "order_count": x.order_count}
            for x in sorted(asks, key=lambda x: x.size_xau, reverse=True)[:8]
        ],
    }


def _quality(
    snapshot: OKXGoldSnapshot,
    trade_count: int,
    coverage_seconds: float,
    *,
    latest_trade_at: datetime,
) -> dict[str, Any]:
    spread_bps = (snapshot.spread / snapshot.mid) * 10000.0 if snapshot.mid else 999.0
    liquidity = 0.35
    if snapshot.volume_24h_xau and snapshot.volume_24h_xau > 0:
        liquidity = _clip((log10(snapshot.volume_24h_xau + 1.0) - 1.0) / 4.0, 0.15, 1.0)
    sample = min(1.0, trade_count / 500.0)
    coverage = min(1.0, max(0.0, coverage_seconds) / 3600.0)
    spread = max(0.10, min(1.0, 1.0 - spread_bps / 10.0))
    trade_age = max(0.0, (snapshot.observed_at - latest_trade_at).total_seconds())
    if trade_age <= 30.0:
        freshness = 1.0
        freshness_state = "fresh"
    elif trade_age <= 300.0:
        freshness = 0.85
        freshness_state = "aging"
    elif trade_age <= 1800.0:
        freshness = 0.60
        freshness_state = "stale_risk"
    else:
        freshness = 0.35
        freshness_state = "stale"
    feed_health = 1.0 if snapshot.bid > 0 and snapshot.ask >= snapshot.bid and snapshot.book and snapshot.trades else 0.25
    raw = (
        0.34 * liquidity
        + 0.22 * sample
        + 0.17 * spread
        + 0.10 * coverage
        + 0.12 * freshness
        + 0.05 * feed_health
    )
    return {
        "liquidity": round(liquidity, 4),
        "sample": round(sample, 4),
        "spread": round(spread, 4),
        "coverage": round(coverage, 4),
        "freshness": round(freshness, 4),
        "freshness_state": freshness_state,
        "latest_trade_age_seconds": round(trade_age, 3),
        "feed_health": round(feed_health, 4),
        "raw_weight": round(_clip(raw, 0.10, 1.0), 4),
        "spread_bps": round(spread_bps, 4),
    }


def analyze_okx_venue(
    snapshot: OKXGoldSnapshot,
    *,
    trades: Iterable[Any] | None = None,
    xau_spot_price: float | None = None,
) -> dict[str, Any]:
    tape = sorted(list(trades or snapshot.trades), key=_trade_time)
    if not tape:
        raise ValueError("venue analysis requires trades")
    coverage_seconds = max(0.0, (_trade_time(tape[-1]) - _trade_time(tape[0])).total_seconds())
    flows = {name: _flow(tape, minutes=minutes) for name, minutes in WINDOWS_MINUTES.items()}
    profiles: dict[str, Any] = {}
    tick = 0.1 if snapshot.market_kind == "perpetual" else 0.5
    for name in PROFILE_WINDOWS:
        subset, cutoff, _ = _window_subset(tape, WINDOWS_MINUTES[name])
        profile = _volume_profile(subset, tick=tick)
        profile["coverage_complete"] = bool(cutoff and _trade_time(tape[0]) <= cutoff)
        profile["status"] = (
            "ready" if profile.get("available") and profile["coverage_complete"]
            else "partial" if profile.get("available")
            else profile.get("status", "unavailable")
        )
        profile["decision_eligible"] = bool(profile.get("available") and profile["coverage_complete"])
        profile["timeframe"] = name
        profiles[name] = profile

    footprint_windows = {}
    for name, minutes in WINDOWS_MINUTES.items():
        fp_trades, fp_cutoff, _ = _window_subset(tape, minutes)
        footprint = _footprint(fp_trades, tick=tick)
        footprint["coverage_complete"] = bool(fp_cutoff and _trade_time(tape[0]) <= fp_cutoff)
        footprint["status"] = (
            "ready" if footprint.get("available") and (name in {"5m", "15m", "30m"} or footprint["coverage_complete"])
            else "partial" if footprint.get("available")
            else "unavailable"
        )
        footprint["decision_eligible"] = bool(
            footprint.get("available") and (name in {"5m", "15m", "30m"} or footprint["coverage_complete"])
        )
        footprint_windows[name] = footprint
    basis = None
    if xau_spot_price and xau_spot_price > 0:
        basis = float(xau_spot_price) - snapshot.mid
        for profile in profiles.values():
            if profile.get("available"):
                profile["xauusd_mapping"] = {
                    "available": True,
                    "method": "instantaneous_xau_minus_venue_basis",
                    "basis": round(basis, 6),
                    "poc": round(float(profile["poc"]) + basis, 4),
                    "vah": round(float(profile["vah"]) + basis, 4),
                    "val": round(float(profile["val"]) + basis, 4),
                    "execution_eligible": False,
                }
    quality = _quality(
        snapshot,
        len(tape),
        coverage_seconds,
        latest_trade_at=_trade_time(tape[-1]),
    )
    return {
        "available": True,
        "status": "ready",
        "venue": snapshot.source,
        "source_family": "okx",
        "instrument_id": snapshot.instrument_id,
        "market_kind": snapshot.market_kind,
        "centralized_proxy_market": True,
        "global_xauusd_order_flow": False,
        "execution_eligible": False,
        "observed_at": snapshot.observed_at.isoformat(),
        "quote": {
            "bid": snapshot.bid,
            "ask": snapshot.ask,
            "mid": round(snapshot.mid, 6),
            "last": snapshot.last,
            "spread": round(snapshot.spread, 6),
        },
        "basis": {
            "available": basis is not None,
            "xau_minus_venue": round(basis, 6) if basis is not None else None,
            "basis_bps": round((basis / float(xau_spot_price)) * 10000.0, 4) if basis is not None else None,
        },
        "trade_tape": {
            "trade_count": len(tape),
            "coverage_seconds": round(coverage_seconds, 3),
            "oldest_trade_at": _trade_time(tape[0]).isoformat(),
            "latest_trade_at": _trade_time(tape[-1]).isoformat(),
        },
        "flow": flows,
        "footprint": footprint_windows,
        "volume_profiles": profiles,
        "book": {"pm10": _book(snapshot, distance=10.0)},
        "absorption": _absorption_candidate(flows["5m"], _book(snapshot, distance=10.0)),
        "open_interest": {
            "available": snapshot.open_interest_xau is not None or snapshot.open_interest_usd is not None,
            "xau": snapshot.open_interest_xau,
            "usd": snapshot.open_interest_usd,
        },
        "volume_24h_xau": snapshot.volume_24h_xau,
        "quality": quality,
    }


def _bitfinex_standard(xaut: dict[str, Any]) -> dict[str, Any]:
    if not xaut or xaut.get("status") != "ready":
        return {"available": False, "status": "unavailable", "venue": "bitfinex:XAUTUSD"}
    flow = dict(xaut.get("flow") or {})
    rolling = (xaut.get("volume_profiles") or {}).get("rolling") or {}
    profiles: dict[str, Any] = {}
    if rolling.get("60m"):
        one_hour = dict(rolling["60m"])
        levels = one_hour.get("levels") or []
        buy = sum(float(row.get("buy_volume") or 0.0) for row in levels)
        sell = sum(float(row.get("sell_volume") or 0.0) for row in levels)
        total = buy + sell
        one_hour["buy_volume_xau"] = round(buy, 8)
        one_hour["sell_volume_xau"] = round(sell, 8)
        one_hour["delta_xau"] = round(buy - sell, 8)
        one_hour["delta_ratio"] = round((buy - sell) / total, 6) if total else 0.0
        basis = xaut.get("basis") or {}
        basis_value = basis.get("xau_minus_xaut")
        if one_hour.get("available") and basis.get("available") and basis_value is not None:
            offset = float(basis_value)
            one_hour["xauusd_mapping"] = {
                "available": True,
                "method": "instantaneous_xau_minus_xaut_basis",
                "basis": round(offset, 6),
                "poc": round(float(one_hour["poc"]) + offset, 4),
                "vah": round(float(one_hour["vah"]) + offset, 4),
                "val": round(float(one_hour["val"]) + offset, 4),
                "execution_eligible": False,
            }
        profiles["1h"] = one_hour
    tape_profile = xaut.get("volume_profile") or {}
    # Tape profile is intentionally NOT relabeled 4h/1d/1w without coverage proof.
    footprint = xaut.get("footprint") or {}
    fp_total = sum(float(x.get("total_volume") or 0.0) for x in footprint.get("levels") or [])
    fp_delta = sum(float(x.get("delta") or 0.0) for x in footprint.get("levels") or [])
    return {
        "available": True,
        "status": "ready",
        "venue": "bitfinex:XAUTUSD",
        "source_family": "bitfinex",
        "instrument_id": "XAUTUSD",
        "market_kind": "tokenized_spot_l3",
        "centralized_proxy_market": True,
        "global_xauusd_order_flow": False,
        "execution_eligible": False,
        "observed_at": xaut.get("observed_at"),
        "quote": xaut.get("quote") or {},
        "basis": xaut.get("basis") or {},
        "trade_tape": xaut.get("trade_tape") or {},
        "flow": flow,
        "footprint": {
            "5m": {
                "available": bool(footprint.get("available")),
                "delta_ratio": round(fp_delta / fp_total, 6) if fp_total else 0.0,
                "levels": footprint.get("levels") or [],
                "method": footprint.get("method"),
            }
        },
        "volume_profiles": profiles,
        "book": {"pm10": (xaut.get("raw_book") or {}).get("pm10") or {}},
        "quality": {
            "raw_weight": min(0.72, 0.35 + 0.0015 * int((xaut.get("trade_tape") or {}).get("trade_count") or 0)),
            "basis_bps": abs(float((xaut.get("basis") or {}).get("basis_bps") or 0.0)),
            "freshness": max(0.35, 1.0 - min(float(xaut.get("snapshot_age_seconds") or 0.0), 60.0) / 90.0),
            "freshness_state": "fresh" if float(xaut.get("snapshot_age_seconds") or 0.0) <= 5.0 else "aging",
            "feed_health": 1.0 if xaut.get("status") == "ready" else 0.25,
        },
        "source_payload": xaut,
    }


def _weighted_venue_signal(venues: list[dict[str, Any]], timeframe: str) -> dict[str, Any]:
    rows = []
    for venue in venues:
        flow = (venue.get("flow") or {}).get(timeframe) or {}
        if not flow.get("available"):
            continue
        score = _clip(float(flow.get("delta_ratio") or 0.0))
        quality = venue.get("quality") or {}
        base_weight = max(0.05, float(quality.get("raw_weight") or 0.25))
        trade_count = int(flow.get("trade_count") or 0)
        sample_target = {"5m": 20, "15m": 40, "30m": 60, "1h": 100, "4h": 180, "1d": 300, "1w": 500}.get(timeframe, 100)
        sample_quality = min(1.0, trade_count / float(sample_target))
        freshness = max(0.20, float(quality.get("freshness") or 0.5))
        feed_health = max(0.20, float(quality.get("feed_health") or 0.5))
        weight = max(0.02, base_weight * (0.45 + 0.55 * sample_quality) * freshness * feed_health)
        rows.append({
            "venue": venue.get("venue"),
            "score": score,
            "weight": round(weight, 6),
            "base_weight": round(base_weight, 6),
            "sample_quality": round(sample_quality, 4),
            "freshness": round(freshness, 4),
            "feed_health": round(feed_health, 4),
            "trade_count": trade_count,
            "coverage_complete": bool(flow.get("coverage_complete")),
        })
    complete_rows = [r for r in rows if r["coverage_complete"]]
    # Never promote a partial multi-hour/day/week tape to a complete timeframe claim.
    decision_rows = complete_rows if timeframe in {"1h", "4h", "1d", "1w"} else rows
    denom = sum(r["weight"] for r in decision_rows)
    composite = sum(r["score"] * r["weight"] for r in decision_rows) / denom if denom else 0.0
    directional = [r for r in decision_rows if abs(r["score"]) >= 0.05]
    if directional:
        positive = sum(1 for r in directional if r["score"] > 0)
        negative = len(directional) - positive
        agreement = max(positive, negative) / len(directional)
    else:
        agreement = 0.5
    complete = bool(decision_rows)
    return {
        "available": bool(rows),
        "status": "ready" if complete else "partial" if rows else "unavailable",
        "decision_eligible": complete,
        "score": round(_clip(composite), 4),
        "direction": (
            "bullish" if complete and composite >= 0.08
            else "bearish" if complete and composite <= -0.08
            else "neutral" if complete
            else "collecting"
        ),
        "agreement_ratio": round(agreement, 4),
        "complete_venue_count": len(complete_rows),
        "venues": rows,
    }


def _profile_cluster(venues: list[dict[str, Any]], timeframe: str) -> dict[str, Any]:
    rows = []
    for venue in venues:
        profile = (venue.get("volume_profiles") or {}).get(timeframe) or {}
        mapping = profile.get("xauusd_mapping") or {}
        poc = mapping.get("poc") if mapping.get("available") else profile.get("poc")
        if not profile.get("available") or poc is None:
            continue
        rows.append({
            "venue": venue.get("venue"),
            "poc_xauusd": float(poc),
            "vah_xauusd": float(mapping.get("vah") if mapping.get("available") else profile.get("vah")),
            "val_xauusd": float(mapping.get("val") if mapping.get("available") else profile.get("val")),
            "coverage_complete": bool(profile.get("coverage_complete")),
        })
    if not rows:
        return {"available": False, "status": "unavailable", "decision_eligible": False, "timeframe": timeframe, "venues": []}
    complete_rows = [r for r in rows if r["coverage_complete"]]
    cluster_rows = complete_rows or rows
    pocs = [r["poc_xauusd"] for r in cluster_rows]
    center = median(pocs)
    spread = max(pocs) - min(pocs) if len(pocs) > 1 else 0.0
    cluster_score = 1.0 if len(pocs) == 1 else max(0.0, 1.0 - spread / 8.0)
    status = "ready" if len(complete_rows) >= 2 else "degraded" if len(complete_rows) == 1 else "partial"
    return {
        "available": True,
        "status": status,
        "decision_eligible": bool(complete_rows),
        "coverage_complete": bool(complete_rows),
        "timeframe": timeframe,
        "poc_cluster_center": round(center, 4),
        "poc_cluster_low": round(min(pocs), 4),
        "poc_cluster_high": round(max(pocs), 4),
        "poc_cluster_width_usd": round(spread, 4),
        "cluster_score": round(cluster_score, 4),
        "complete_venue_count": len(complete_rows),
        "partial_venue_count": len(rows) - len(complete_rows),
        "venues": rows,
    }


def build_gold_market_fusion(
    *,
    okx_xau: dict[str, Any] | None,
    okx_xaut: dict[str, Any] | None,
    bitfinex_xaut: dict[str, Any] | None,
) -> dict[str, Any]:
    venues = [v for v in [okx_xau, okx_xaut, _bitfinex_standard(bitfinex_xaut or {})] if v and v.get("available")]
    flows = {tf: _weighted_venue_signal(venues, tf) for tf in WINDOWS_MINUTES}
    profiles = {tf: _profile_cluster(venues, tf) for tf in PROFILE_WINDOWS}

    footprint_rows = []
    book_rows = []
    for venue in venues:
        fp = (venue.get("footprint") or {}).get("5m") or {}
        if fp.get("available"):
            footprint_rows.append({
                "venue": venue.get("venue"),
                "score": _clip(float(fp.get("delta_ratio") or 0.0)),
                "weight": max(0.05, float((venue.get("quality") or {}).get("raw_weight") or 0.25)),
            })
        book = (venue.get("book") or {}).get("pm10") or {}
        if book and "imbalance" in book:
            book_rows.append({
                "venue": venue.get("venue"),
                "score": _clip(float(book.get("imbalance") or 0.0)),
                "weight": max(0.05, float((venue.get("quality") or {}).get("raw_weight") or 0.25)),
            })

    def combine(rows: list[dict[str, Any]]) -> tuple[float, float]:
        if not rows:
            return 0.0, 0.5
        denom = sum(x["weight"] for x in rows)
        score = sum(x["score"] * x["weight"] for x in rows) / denom if denom else 0.0
        directional = [x for x in rows if abs(x["score"]) >= 0.05]
        if not directional:
            return score, 0.5
        pos = sum(1 for x in directional if x["score"] > 0)
        agreement = max(pos, len(directional) - pos) / len(directional)
        return score, agreement

    footprint_score, footprint_agreement = combine(footprint_rows)
    book_score, book_agreement = combine(book_rows)
    flow_agreement = (flows.get("5m") or {}).get("agreement_ratio", 0.5)
    profile_1h = profiles.get("1h") or {}
    profile_score = (
        float(profile_1h.get("cluster_score") or 0.0)
        if profile_1h.get("decision_eligible")
        else 0.5
    )
    agreement_score = 100.0 * (
        0.40 * float(flow_agreement)
        + 0.20 * float(footprint_agreement)
        + 0.15 * float(book_agreement)
        + 0.25 * float(profile_score)
    )

    flow_horizon_weights = {"5m": 0.35, "15m": 0.20, "30m": 0.15, "1h": 0.15, "4h": 0.10, "1d": 0.04, "1w": 0.01}
    eligible_flows = [
        (tf, row, flow_horizon_weights[tf])
        for tf, row in flows.items()
        if tf in flow_horizon_weights and row.get("decision_eligible")
    ]
    flow_weight_total = sum(weight for _, _, weight in eligible_flows)
    composite_flow = (
        sum(float(row.get("score") or 0.0) * weight for _, row, weight in eligible_flows) / flow_weight_total
        if flow_weight_total
        else 0.0
    )
    composite_footprint = _clip(footprint_score)
    composite_liquidity = _clip(book_score)
    composite = _clip(0.52 * composite_flow + 0.28 * composite_footprint + 0.20 * composite_liquidity)
    venue_diagnostics = []
    for venue in venues:
        quality = venue.get("quality") or {}
        venue_diagnostics.append({
            "venue": venue.get("venue"),
            "source_family": venue.get("source_family"),
            "instrument_id": venue.get("instrument_id"),
            "market_kind": venue.get("market_kind"),
            "status": venue.get("status"),
            "quality": quality,
            "basis": venue.get("basis") or {},
            "flow_coverage": {
                tf: {
                    "available": bool(((venue.get("flow") or {}).get(tf) or {}).get("available")),
                    "coverage_complete": bool(((venue.get("flow") or {}).get(tf) or {}).get("coverage_complete")),
                }
                for tf in WINDOWS_MINUTES
            },
            "profile_coverage": {
                tf: {
                    "available": bool(((venue.get("volume_profiles") or {}).get(tf) or {}).get("available")),
                    "coverage_complete": bool(((venue.get("volume_profiles") or {}).get(tf) or {}).get("coverage_complete")),
                }
                for tf in PROFILE_WINDOWS
            },
        })
    coverage_state = {
        "flow": {tf: flows[tf].get("status") for tf in WINDOWS_MINUTES},
        "profiles": {tf: profiles[tf].get("status") for tf in PROFILE_WINDOWS},
    }
    source_families = {str(v.get("source_family") or v.get("venue")) for v in venues}
    independent_source_count = len(source_families)
    freshness_values = [float((v.get("quality") or {}).get("freshness") or 0.0) for v in venues]
    freshness_state = (
        "fresh" if freshness_values and min(freshness_values) >= 0.85
        else "mixed" if freshness_values and max(freshness_values) >= 0.60
        else "stale_risk" if freshness_values
        else "unavailable"
    )

    return {
        "version": "gold-market-fusion-v1",
        "status": (
            "ready" if len(venues) >= 2 and independent_source_count >= 2
            else "degraded" if venues
            else "unavailable"
        ),
        "research_only": True,
        "global_xauusd_volume": False,
        "global_xauusd_order_flow": False,
        "venue_count": len(venues),
        "independent_source_count": independent_source_count,
        "source_families": sorted(source_families),
        "venues": venues,
        "venue_diagnostics": venue_diagnostics,
        "coverage_state": coverage_state,
        "freshness_state": freshness_state,
        "flow": flows,
        "footprint": {
            "score": round(composite_footprint, 4),
            "direction": "bullish" if composite_footprint >= 0.08 else "bearish" if composite_footprint <= -0.08 else "neutral",
            "agreement_ratio": round(footprint_agreement, 4),
            "venues": footprint_rows,
        },
        "liquidity": {
            "score": round(composite_liquidity, 4),
            "direction": "bid_heavy" if composite_liquidity >= 0.08 else "ask_heavy" if composite_liquidity <= -0.08 else "balanced",
            "agreement_ratio": round(book_agreement, 4),
            "venues": book_rows,
        },
        "volume_profile_map": profiles,
        "market_agreement_score": round(max(0.0, min(100.0, agreement_score)), 2),
        "composite_flow_score": round(composite_flow, 4),
        "composite_footprint_score": round(composite_footprint, 4),
        "composite_liquidity_score": round(composite_liquidity, 4),
        "composite_microstructure_score": round(composite, 4),
        "direction": "bullish" if composite >= 0.10 else "bearish" if composite <= -0.10 else "neutral",
        "evidence_policy": {
            "never_sum_cross_venue_raw_volume": True,
            "venue_volume_remains_venue_scoped": True,
            "signals_are_liquidity_weighted": True,
            "tokenized_gold_is_not_otc_xauusd": True,
            "perpetual_gold_is_not_otc_xauusd": True,
            "execution_allowed": False,
        },
    }
