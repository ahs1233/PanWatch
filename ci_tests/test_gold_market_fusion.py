from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.modules.xau.gold_market_fusion import analyze_okx_venue, build_gold_market_fusion
from src.modules.xau.gold_tape_store import GoldTapeStore
from src.platform.marketdata.gold_okx import OKXBookLevel, OKXGoldSnapshot, OKXGoldTrade

NOW = datetime(2026, 9, 23, 8, 0, tzinfo=timezone.utc)


def _snapshot(name: str, *, bullish: bool, market_kind: str = "perpetual", count: int = 120, spacing_seconds: int = 20) -> OKXGoldSnapshot:
    trades = []
    for i in range(count):
        side = "buy" if (i % 5 != 0 if bullish else i % 5 == 0) else "sell"
        trades.append(
            OKXGoldTrade(
                trade_id=f"{name}-{i}",
                timestamp=NOW - timedelta(seconds=(count - 1 - i) * spacing_seconds),
                price=4320.0 + i * (0.01 if bullish else -0.01),
                size_xau=0.20,
                aggressor_side=side,
            )
        )
    book = (
        OKXBookLevel(4318.0, 20.0 if bullish else 4.0, "bid", 10),
        OKXBookLevel(4322.0, 4.0 if bullish else 20.0, "ask", 10),
    )
    return OKXGoldSnapshot(
        instrument_id=name,
        market_kind=market_kind,
        bid=4319.9,
        ask=4320.1,
        last=4320.0,
        observed_at=NOW,
        trades=tuple(trades),
        book=book,
        open_interest_xau=1000.0,
        open_interest_usd=4_320_000.0,
        volume_24h_xau=50_000.0 if name == "XAU-USDT-SWAP" else 3_000.0,
        source=f"okx:{name}",
    )


def test_okx_venue_builds_real_flow_footprint_and_profile():
    snap = _snapshot("XAU-USDT-SWAP", bullish=True)
    out = analyze_okx_venue(snap, xau_spot_price=4321.0)
    assert out["flow"]["5m"]["available"] is True
    assert out["flow"]["5m"]["delta_ratio"] > 0
    assert out["footprint"]["5m"]["available"] is True
    assert out["volume_profiles"]["1h"]["status"] == "partial"
    assert out["volume_profiles"]["1h"]["decision_eligible"] is False
    assert out["volume_profiles"]["1h"]["xauusd_mapping"]["available"] is True
    assert out["global_xauusd_order_flow"] is False


def test_fusion_weights_signals_not_cross_venue_raw_volume():
    swap = analyze_okx_venue(_snapshot("XAU-USDT-SWAP", bullish=True), xau_spot_price=4321.0)
    token = analyze_okx_venue(_snapshot("XAUT-USDT", bullish=False, market_kind="tokenized_spot"), xau_spot_price=4321.0)
    bitfinex = {
        "status": "ready",
        "observed_at": NOW.isoformat(),
        "quote": {"mid": 4320.0},
        "basis": {"basis_bps": 2.0},
        "trade_tape": {"trade_count": 300, "coverage_seconds": 1200},
        "flow": {"5m": {"available": True, "delta_ratio": 0.20, "trade_count": 50}},
        "footprint": {"available": True, "levels": [
            {"total_volume": 2.0, "delta": 0.4}
        ]},
        "volume_profiles": {"rolling": {"60m": {
            "available": True, "status": "ready", "poc": 4319.5, "vah": 4321.0, "val": 4318.0,
            "xauusd_mapping": {"available": True, "poc": 4320.5, "vah": 4322.0, "val": 4319.0},
            "coverage_complete": True,
        }}},
        "raw_book": {"pm10": {"imbalance": 0.15}},
    }
    fused = build_gold_market_fusion(okx_xau=swap, okx_xaut=token, bitfinex_xaut=bitfinex)
    assert fused["venue_count"] == 3
    assert fused["status"] == "ready"
    assert fused["independent_source_count"] == 2
    assert fused["evidence_policy"]["never_sum_cross_venue_raw_volume"] is True
    assert 0 <= fused["market_agreement_score"] <= 100
    assert fused["flow"]["5m"]["available"] is True
    assert len(fused["flow"]["5m"]["venues"]) == 3


def test_profiles_never_claim_daily_weekly_complete_without_coverage():
    snap = _snapshot("XAU-USDT-SWAP", bullish=True)
    out = analyze_okx_venue(snap, xau_spot_price=4321.0)
    assert out["volume_profiles"]["1d"]["coverage_complete"] is False
    assert out["volume_profiles"]["1w"]["coverage_complete"] is False


def test_tape_store_persists_and_deduplicates(tmp_path):
    snap = _snapshot("XAU-USDT-SWAP", bullish=True)
    store = GoldTapeStore(str(tmp_path / "tape.sqlite3"))
    n1 = store.ingest_okx(snap.source, snap.trades)
    n2 = store.ingest_okx(snap.source, snap.trades)
    assert n1 == len(snap.trades)
    assert n2 == 0
    rows = store.load(snap.source, minutes=10080)
    assert len(rows) == len(snap.trades)


def test_okx_contract_and_spot_size_normalization():
    swap = __import__("src.platform.marketdata.gold_okx", fromlist=["OKXGoldPublicProvider"]).OKXGoldPublicProvider.xau_swap()
    spot = __import__("src.platform.marketdata.gold_okx", fromlist=["OKXGoldPublicProvider"]).OKXGoldPublicProvider.xaut_spot()
    a = swap._parse_trade({"tradeId":"1","px":"4325","sz":"3106","side":"buy","ts":"1790151369238"})
    b = spot._parse_trade({"tradeId":"2","px":"4323.1","sz":"0.115518","side":"buy","ts":"1790151378917"})
    assert a.size_xau == 3.106
    assert b.size_xau == 0.115518


def test_long_timeframe_flow_is_collecting_until_tape_is_complete():
    swap = analyze_okx_venue(_snapshot("XAU-USDT-SWAP", bullish=True), xau_spot_price=4321.0)
    fused = build_gold_market_fusion(okx_xau=swap, okx_xaut=None, bitfinex_xaut=None)
    assert fused["flow"]["1d"]["status"] == "partial"
    assert fused["flow"]["1d"]["decision_eligible"] is False
    assert fused["flow"]["1d"]["direction"] == "collecting"


def test_profile_summary_has_buy_sell_delta_and_partial_guardrail():
    snap = _snapshot("XAU-USDT-SWAP", bullish=True)
    out = analyze_okx_venue(snap, xau_spot_price=4321.0)
    profile = out["volume_profiles"]["1h"]
    assert profile["buy_volume_xau"] > 0
    assert profile["sell_volume_xau"] > 0
    assert profile["delta_xau"] == round(profile["buy_volume_xau"] - profile["sell_volume_xau"], 8)
    assert profile["coverage_complete"] is False
    assert profile["status"] == "partial"


def test_same_exchange_two_instruments_are_not_two_independent_sources():
    swap = analyze_okx_venue(_snapshot("XAU-USDT-SWAP", bullish=True), xau_spot_price=4321.0)
    token = analyze_okx_venue(_snapshot("XAUT-USDT", bullish=False, market_kind="tokenized_spot"), xau_spot_price=4321.0)
    fused = build_gold_market_fusion(okx_xau=swap, okx_xaut=token, bitfinex_xaut=None)
    assert fused["venue_count"] == 2
    assert fused["independent_source_count"] == 1
    assert fused["status"] == "degraded"


def test_diagnostics_expose_freshness_sample_and_feed_health():
    swap = analyze_okx_venue(_snapshot("XAU-USDT-SWAP", bullish=True), xau_spot_price=4321.0)
    fused = build_gold_market_fusion(okx_xau=swap, okx_xaut=None, bitfinex_xaut=None)
    diag = fused["venue_diagnostics"][0]
    assert "freshness" in diag["quality"]
    assert "feed_health" in diag["quality"]
    row = fused["flow"]["5m"]["venues"][0]
    assert "sample_quality" in row
    assert "freshness" in row
    assert "feed_health" in row


def test_footprint_exposes_all_requested_horizons_and_absorption_candidate():
    out = analyze_okx_venue(_snapshot("XAU-USDT-SWAP", bullish=True), xau_spot_price=4321.0)
    assert set(out["footprint"]) == {"5m", "15m", "30m", "1h", "4h", "1d", "1w"}
    assert out["footprint"]["1d"]["status"] == "partial"
    assert out["absorption"]["confidence"] == "candidate_only"


def test_one_hour_becomes_ready_only_after_full_tape_coverage():
    out = analyze_okx_venue(
        _snapshot("XAU-USDT-SWAP", bullish=True, count=70, spacing_seconds=60),
        xau_spot_price=4321.0,
    )
    assert out["flow"]["1h"]["coverage_complete"] is True
    assert out["flow"]["1h"]["status"] == "ready"
    assert out["flow"]["1h"]["decision_eligible"] is True
    assert out["volume_profiles"]["1h"]["coverage_complete"] is True
    assert out["volume_profiles"]["1h"]["status"] == "ready"
    assert out["volume_profiles"]["1h"]["decision_eligible"] is True
    assert out["volume_profiles"]["4h"]["status"] == "partial"
