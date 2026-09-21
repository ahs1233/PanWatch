from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from src.modules.xau.paper import (
    XAUPaperTradingEngine,
    _paper_entry_price,
    _paper_mark_price,
    _paper_exit_quote,
    _paper_exit_fill_price,
    _performance_metrics,
    _position_age_minutes,
    _spot_spread_bps,
    _pnl,
    _week_key,
)
from src.platform.runtime.config import Settings
from src.modules.xau.paper_store import _sqlalchemy_url


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



def test_paper_postgres_url_uses_psycopg_driver():
    value = _sqlalchemy_url("postgresql://user:pass@example.test/panwatch")
    assert value == "postgresql+psycopg://user:pass@example.test/panwatch"



def test_mid_only_reference_is_not_accepted_as_entry_fill():
    spot = {"price": 4350.0, "bid": None, "ask": None}

    assert _paper_entry_price("long", spot) is None
    assert _paper_entry_price("short", spot) is None
    assert _paper_mark_price("long", spot) == 4350.0
    assert _paper_mark_price("short", spot) == 4350.0



def test_target_fill_does_not_credit_favorable_overshoot():
    assert _paper_exit_fill_price("long", 4405.0, 4300.0, 4400.0, "target_price") == 4400.0
    assert _paper_exit_fill_price("short", 4295.0, 4400.0, 4300.0, "target_price") == 4300.0


def test_stop_fill_keeps_adverse_gap_slippage():
    assert _paper_exit_fill_price("long", 4288.0, 4300.0, 4400.0, "stop_loss") == 4288.0
    assert _paper_exit_fill_price("short", 4412.0, 4400.0, 4300.0, "stop_loss") == 4412.0


def test_setup_key_dedupes_macro_state_changes_inside_same_15m_anchor():
    engine = XAUPaperTradingEngine(Settings())
    account = SimpleNamespace(week_key="2026-W39")
    technical = {
        "frames": {
            "15m": {
                "observed_at": "2026-09-21T13:30:00+00:00",
            }
        }
    }

    support = engine._setup_key(
        account,
        technical,
        {"technical_candidate": "long_setup", "state": "setup_macro_support"},
    )
    neutral = engine._setup_key(
        account,
        technical,
        {"technical_candidate": "long_setup", "state": "setup_macro_neutral"},
    )
    short = engine._setup_key(
        account,
        technical,
        {"technical_candidate": "short_setup", "state": "setup_macro_support"},
    )

    assert support == neutral
    assert support != short



def test_performance_metrics_use_r_and_realized_pnl():
    trades = [
        SimpleNamespace(r_multiple=2.0, pnl=100.0, mfe_usd=140.0, mae_usd=-20.0),
        SimpleNamespace(r_multiple=-1.0, pnl=-50.0, mfe_usd=15.0, mae_usd=-60.0),
        SimpleNamespace(r_multiple=1.0, pnl=50.0, mfe_usd=80.0, mae_usd=-10.0),
    ]

    metrics = _performance_metrics(trades)

    assert metrics["trade_count"] == 3
    assert metrics["average_r"] == round(2.0 / 3.0, 4)
    assert metrics["expectancy_r"] == round(2.0 / 3.0, 4)
    assert metrics["profit_factor"] == 3.0
    assert metrics["average_win_r"] == 1.5
    assert metrics["average_loss_r"] == -1.0
    assert metrics["average_mfe_usd"] == round(235.0 / 3.0, 4)
    assert metrics["average_mae_usd"] == -30.0



def test_spread_bps_is_derived_from_bid_ask_when_missing():
    spot = {"price": 4350.0, "bid": 4349.8, "ask": 4350.2}
    spread = _spot_spread_bps(spot)

    assert spread is not None
    assert spread == round((0.4 / 4350.0) * 10_000.0, 10)


def test_position_age_minutes_handles_naive_and_aware_datetimes():
    opened = datetime(2026, 9, 21, 12, 0, tzinfo=timezone.utc)
    now = datetime(2026, 9, 21, 16, 0, tzinfo=timezone.utc)
    assert _position_age_minutes(opened, now) == 240.0

    assert _position_age_minutes(
        opened.replace(tzinfo=None),
        now.replace(tzinfo=None),
    ) == 240.0



def test_mid_only_reference_cannot_trigger_paper_exit():
    spot = {"price": 4350.0, "bid": None, "ask": None}

    assert _paper_mark_price("long", spot) == 4350.0
    assert _paper_mark_price("short", spot) == 4350.0
    assert _paper_exit_quote("long", spot) is None
    assert _paper_exit_quote("short", spot) is None
