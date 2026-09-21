from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.modules.xau.paper import (
    XAUPaperTradingEngine,
    _paper_entry_price,
    _paper_mark_price,
    _pnl,
    _week_key,
)
from src.platform.runtime.config import Settings


def test_long_and_short_use_conservative_bid_ask_fills():
    spot = {"price": 4350.0, "bid": 4349.8, "ask": 4350.2}

    assert _paper_entry_price("long", spot) == 4350.2
    assert _paper_mark_price("long", spot) == 4349.8

    assert _paper_entry_price("short", spot) == 4349.8
    assert _paper_mark_price("short", spot) == 4350.2


def test_pnl_direction_is_symmetric():
    assert _pnl("long", 4300.0, 4310.0, 2.0) == 20.0
    assert _pnl("short", 4300.0, 4290.0, 2.0) == 20.0
    assert _pnl("long", 4300.0, 4290.0, 2.0) == -20.0
    assert _pnl("short", 4300.0, 4310.0, 2.0) == -20.0


def test_week_key_uses_baghdad_timezone():
    settings = Settings(xau_paper_timezone="Asia/Baghdad")
    value = datetime(2026, 9, 21, 21, 30, tzinfo=timezone.utc)

    assert _week_key(settings, value) == "2026-W39"


def test_long_risk_sizing_respects_risk_budget_and_notional_cap():
    settings = Settings(
        xau_paper_initial_capital=10_000.0,
        xau_paper_risk_pct=0.01,
        xau_paper_reward_risk=2.0,
        xau_paper_max_leverage=1.0,
    )
    engine = XAUPaperTradingEngine(settings)
    account = SimpleNamespace(current_equity=10_000.0, initial_capital=10_000.0)

    levels = engine._levels_and_size(
        account,
        "long",
        4350.0,
        {
            "atr_reference": 12.0,
            "swing_low_reference": 4300.0,
            "swing_high_reference": 4400.0,
        },
    )

    assert levels is not None
    stop, target, quantity, risk_usd = levels
    assert stop == 4300.0
    assert target == 4450.0
    assert risk_usd <= 100.0 + 1e-6
    assert quantity * 4350.0 <= 10_000.0 + 1.0


def test_short_falls_back_to_atr_when_swing_is_invalid():
    settings = Settings(
        xau_paper_initial_capital=10_000.0,
        xau_paper_risk_pct=0.01,
        xau_paper_reward_risk=2.0,
        xau_paper_max_leverage=1.0,
    )
    engine = XAUPaperTradingEngine(settings)
    account = SimpleNamespace(current_equity=10_000.0, initial_capital=10_000.0)

    levels = engine._levels_and_size(
        account,
        "short",
        4350.0,
        {
            "atr_reference": 10.0,
            "swing_low_reference": 4300.0,
            "swing_high_reference": 4340.0,
        },
    )

    assert levels is not None
    stop, target, quantity, risk_usd = levels
    assert stop == 4360.0
    assert target == 4330.0
    assert risk_usd > 0
    assert quantity * 4350.0 <= 10_000.0 + 1.0
