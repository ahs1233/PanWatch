from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.modules.xau.xaut_order_flow import analyze_xaut_microstructure
from src.platform.marketdata.xaut_bitfinex import (
    BitfinexXAUTStream,
    XAUTBookOrder,
    XAUTMicrostructureSnapshot,
    XAUTTrade,
    parse_raw_book_order,
    parse_trade,
)


NOW = datetime(2026, 9, 23, 0, 0, tzinfo=timezone.utc)


def test_public_trade_sign_maps_to_aggressor_side():
    buy = parse_trade([1, 1790118000000, 0.50, 4360.0])
    sell = parse_trade([2, 1790118001000, -0.25, 4360.1])
    assert buy.aggressor_side == "buy"
    assert buy.size == 0.50
    assert sell.aggressor_side == "sell"
    assert sell.size == 0.25


def test_raw_book_sign_maps_to_bid_and_ask():
    bid = parse_raw_book_order([10, 4359.0, 3.0])
    ask = parse_raw_book_order([11, 4361.0, -2.0])
    assert bid.side == "bid"
    assert ask.side == "ask"


def test_stream_raw_book_delete_is_order_id_based():
    stream = BitfinexXAUTStream()
    stream.apply_book_payload([[10, 4359.0, 3.0], [11, 4361.0, -2.0]])
    assert 10 in stream._book
    stream.apply_book_payload([10, 0, -1])
    assert 10 not in stream._book
    assert stream._ask == 4361.0


def _snapshot() -> XAUTMicrostructureSnapshot:
    trades = []
    # Older buy flow, followed by aggressive selling in the last five minutes.
    for i in range(12):
        ts = NOW - timedelta(minutes=20 - i)
        trades.append(XAUTTrade(i + 1, ts, 0.2, 4358.0 + i * 0.1))
    for i in range(8):
        ts = NOW - timedelta(seconds=240 - i * 20)
        trades.append(XAUTTrade(100 + i, ts, -0.5, 4360.0 - i * 0.05))
    book = (
        XAUTBookOrder(1, 4359.0, 8.0),
        XAUTBookOrder(2, 4358.5, 6.0),
        XAUTBookOrder(3, 4361.5, -1.0),
        XAUTBookOrder(4, 4362.0, -1.5),
    )
    return XAUTMicrostructureSnapshot(
        bid=4360.0,
        ask=4361.0,
        last=4360.2,
        observed_at=NOW,
        trades=tuple(trades),
        raw_book=book,
    )


def test_analysis_produces_true_xaut_delta_book_and_basis_without_overclaiming():
    result = analyze_xaut_microstructure(_snapshot(), xau_spot_price=4363.5)
    assert result["status"] == "ready"
    assert result["flow"]["5m"]["delta"] < 0
    assert result["raw_book"]["pm10"]["imbalance"] > 0
    assert result["basis"]["xau_minus_xaut"] == 3.0
    assert result["absorption"]["state"] == "possible_sell_absorption"
    assert result["evidence_policy"]["trade_sign_is_real_for_xaut"] is True
    assert result["evidence_policy"]["xaut_is_xauusd_execution_venue"] is False
    assert result["global_xauusd_order_flow"] is False


def test_forward_range_map_contains_10_20_30_both_directions():
    result = analyze_xaut_microstructure(_snapshot(), xau_spot_price=4363.5)
    forward = result["forward_range_map"]
    assert forward["pm10"]["xau_up"] == 4373.5
    assert forward["pm10"]["xau_down"] == 4353.5
    assert forward["pm20"]["xau_up"] == 4383.5
    assert forward["pm30"]["xau_down"] == 4333.5


def test_footprint_uses_executed_trade_sign_not_candle_direction():
    result = analyze_xaut_microstructure(_snapshot(), xau_spot_price=4363.5)
    levels = result["footprint"]["levels"]
    assert levels
    assert any(level["bid_volume"] > 0 for level in levels)
    assert any(level["ask_volume"] > 0 for level in levels)
    assert result["footprint"]["method"] == "aggressor-signed-trades"


def test_volume_profile_uses_real_executed_xaut_volume_and_value_area():
    result = analyze_xaut_microstructure(
        _snapshot(),
        xau_spot_price=4363.5,
        volume_profile_tick=0.5,
    )
    profile = result["volume_profile"]
    assert profile["status"] == "ready"
    assert profile["volume_kind"] == "executed_xaut_volume"
    assert profile["global_xauusd_volume_profile"] is False
    assert profile["total_volume_xaut"] == 6.4
    assert profile["val"] <= profile["poc"] <= profile["vah"]
    assert profile["value_area_volume_share"] >= 0.70
    assert profile["high_volume_nodes"]
    assert profile["low_volume_nodes"]
    assert profile["levels"]


def test_volume_profile_maps_poc_vah_val_to_xau_by_instantaneous_basis_only():
    result = analyze_xaut_microstructure(
        _snapshot(),
        xau_spot_price=4363.5,
        volume_profile_tick=0.5,
    )
    profile = result["volume_profile"]
    mapping = profile["xauusd_mapping"]
    assert mapping["available"] is True
    assert mapping["method"] == "instantaneous_xau_minus_xaut_basis"
    assert mapping["execution_eligible"] is False
    assert mapping["poc"] == round(profile["poc"] + 3.0, 4)
    assert mapping["vah"] == round(profile["vah"] + 3.0, 4)
    assert mapping["val"] == round(profile["val"] + 3.0, 4)
    assert result["evidence_policy"]["volume_profile_is_real_executed_xaut_volume"] is True
    assert result["evidence_policy"]["xaut_volume_profile_is_global_xauusd_volume"] is False
