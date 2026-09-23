from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.modules.xau.paper import (
    XAUPaperTradingEngine,
    _entry_gate_reason,
    _paper_entry_price,
    _can_revalidate_signal,
    _calibration_metrics,
    _trade_autopsy,
    _paper_mark_price,
    _paper_context_mark_price,
    _paper_exit_quote,
    _paper_management_quote,
    _weekly_reset_fill_price,
    _paper_exit_fill_price,
    _performance_metrics,
    _shadow_metrics,
    _shadow_research_memory,
    _replay_research_memory,
    _shadow_horizon_due,
    _calibration_metrics,
    _trade_autopsy,
    _position_age_minutes,
    _position_guardian,
    _confirm_reversal_exit,
    _spot_spread_bps,
    _pnl,
    _week_key,
)
from src.platform.runtime.config import Settings
from src.modules.xau.paper_store import _sqlalchemy_url


@pytest.mark.parametrize("offset_hours", [0, 3, -5])
@pytest.mark.parametrize("aware_start", [False, True])
def test_research_overlap_normalizes_mixed_timezones_before_comparison(offset_hours, aware_start):
    from src.modules.xau.paper import _decorrelate_research_episodes

    start = datetime(2026, 9, 22, 4, 0)
    end = start + timedelta(hours=1)
    zone = timezone(timedelta(hours=offset_hours))
    aware = lambda value: value.replace(tzinfo=timezone.utc).astimezone(zone)
    episodes = [
        (1.0, 10.0, "60m", aware(start) if aware_start else start,
         end if aware_start else aware(end)),
        (0.9, -5.0, "60m", start + timedelta(minutes=30),
         aware(end + timedelta(minutes=30))),
        (0.8, 2.0, "60m", aware(end), end + timedelta(hours=1)),
    ]
    chosen, discarded = _decorrelate_research_episodes(episodes, limit=80)
    assert chosen == [episodes[0], episodes[2]]
    assert discarded == 1


@pytest.mark.parametrize("operation", ["scan", "eligibility"])
def test_database_work_does_not_block_event_loop(monkeypatch, operation):
    from src.modules.xau import paper
    monkeypatch.setattr(XAUPaperTradingEngine, "_protect_sync", lambda *args: {"status": "ok", "execution_allowed": False})

    entered, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()
    worker_threads = []

    async def snapshot(**kwargs):
        return {}

    def slow_database_work(*args):
        worker_threads.append(threading.get_ident())
        entered.set()
        assert release.wait(3), "event loop could not release database worker"
        return {"status": "ok", "execution_allowed": False}

    monkeypatch.setattr(paper, "get_xau_snapshot", snapshot)
    monkeypatch.setattr(paper, "get_macro_context", snapshot)
    monkeypatch.setattr(paper, "get_indicative_spot", snapshot)
    engine = XAUPaperTradingEngine(Settings(xau_paper_enabled=True))
    monkeypatch.setattr(engine, "_scan_sync" if operation == "scan" else "_eligibility_sync", slow_database_work)

    async def exercise():
        pending = asyncio.create_task(getattr(engine, operation)())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not pending.done()
            assert len(worker_threads) == 1
            assert worker_threads[0] != event_loop_thread
        finally:
            release.set()
        result = await pending
        assert result["execution_allowed"] is False
        if operation == "scan":
            assert result["timing_ms"]["total"] >= result["timing_ms"]["paper_worker"]

    asyncio.run(exercise())


def test_cancelled_scan_keeps_transaction_guard_until_worker_finishes(monkeypatch):
    from src.modules.xau import paper
    monkeypatch.setattr(XAUPaperTradingEngine, "_protect_sync", lambda *args: {"status": "ok", "execution_allowed": False})

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    async def snapshot(**kwargs):
        return {}

    def transaction(*args):
        entered.set()
        try:
            assert release.wait(3)
            return {"status": "ok", "execution_allowed": False}
        finally:
            finished.set()

    monkeypatch.setattr(paper, "get_xau_snapshot", snapshot)
    monkeypatch.setattr(paper, "get_macro_context", snapshot)
    monkeypatch.setattr(paper, "get_indicative_spot", snapshot)
    first = XAUPaperTradingEngine(Settings(xau_paper_enabled=True))
    second = XAUPaperTradingEngine(Settings(xau_paper_enabled=True))
    monkeypatch.setattr(first, "_scan_sync", transaction)
    monkeypatch.setattr(second, "_scan_sync", lambda *args: {"status": "ok"})

    async def exercise():
        pending = asyncio.create_task(first.scan())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            result = await second.scan()
            assert result["status"] == "busy"
            assert result["execution_allowed"] is False
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 2)

    asyncio.run(exercise())
    # asyncio.run drains its executor, so the first worker has released the guard.
    assert asyncio.run(second.scan())["status"] == "ok"


def test_scan_guard_is_released_after_database_failure(monkeypatch):
    engine = XAUPaperTradingEngine()

    def fail(*args):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(engine, "_scan_sync", fail)
    with pytest.raises(RuntimeError, match="database unavailable"):
        engine._scan_serialized({}, {}, None)
    monkeypatch.setattr(engine, "_scan_sync", lambda *args: {"status": "ok"})
    assert engine._scan_serialized({}, {}, None)["status"] == "ok"


def test_worker_session_commits_paper_state_and_remains_readable(monkeypatch, tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.modules.xau import paper
    from src.platform.persistence.database import Base

    sql_engine = create_engine(f"sqlite:///{tmp_path / 'paper.db'}")
    Base.metadata.create_all(sql_engine)
    factory = sessionmaker(bind=sql_engine)
    owner_thread = threading.get_ident()
    session_threads = []

    def open_session():
        session_threads.append(threading.get_ident())
        assert threading.get_ident() != owner_thread
        return factory()

    async def snapshot(**kwargs):
        return {"blocked": True, "candidate": "none", "frames": {}}

    monkeypatch.setattr(paper, "open_xau_paper_session", open_session)
    monkeypatch.setattr(paper, "open_xau_replay_session", open_session)
    monkeypatch.setattr(paper, "get_xau_snapshot", snapshot)
    monkeypatch.setattr(paper, "get_macro_context", snapshot)
    monkeypatch.setattr(paper, "get_indicative_spot", snapshot)
    engine = XAUPaperTradingEngine(Settings(xau_paper_enabled=True))

    async def exercise():
        result = await engine.scan()
        assert result["status"] == "ok"
        assert result["opened"] is False
        assert result["execution_allowed"] is False
        eligibility = await engine.eligibility()
        assert eligibility["eligible"] is False
        summary = await asyncio.to_thread(engine.summary)
        assert summary["account"]["week_key"] == result["week_key"]
        assert summary["position"] is None

    try:
        asyncio.run(exercise())
        assert session_threads
    finally:
        sql_engine.dispose()


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
    assert metrics["win_rate"] == round(2 / 3, 4)
    assert metrics["average_mfe_usd"] == round(235.0 / 3.0, 4)
    assert metrics["average_mae_usd"] == -30.0



def test_spread_bps_is_derived_from_bid_ask_when_missing():
    spot = {"price": 4350.0, "bid": 4349.8, "ask": 4350.2}
    spread = _spot_spread_bps(spot)

    assert spread is not None
    assert spread == pytest.approx((0.4 / 4350.0) * 10_000.0, rel=1e-10)


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



def test_transient_quote_rejections_can_be_revalidated():
    for reason in (
        "bid_ask_unavailable",
        "indicative_spot_stale",
        "spread_too_wide",
    ):
        assert _can_revalidate_signal(False, reason, True) is True

    assert _can_revalidate_signal(False, "setup_macro_conflict", True) is False
    assert _can_revalidate_signal(False, "position_already_open", True) is False
    assert _can_revalidate_signal(True, "bid_ask_unavailable", True) is False
    assert _can_revalidate_signal(False, "bid_ask_unavailable", False) is False



def test_entry_gate_reason_matches_engine_policy():
    good_spot = {
        "price": 4350.0,
        "bid": 4349.8,
        "ask": 4350.2,
        "is_stale": False,
    }

    assert _entry_gate_reason(
        candidate="none",
        fusion_state="no_setup",
        spot=good_spot,
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (False, "no_setup")

    assert _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot=good_spot,
        has_open_position=True,
        max_spread_bps=3.0,
    ) == (False, "position_already_open")

    assert _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot={**good_spot, "is_stale": True},
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (False, "indicative_spot_stale")


    assert _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot={
            **good_spot,
            "provider_health": [
                {
                    "provider": "BiquoteXAUIndicativeSpotReference",
                    "status": "ok",
                    "has_bid_ask": True,
                    "is_stale": True,
                    "market_state": "closed",
                }
            ],
        },
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (False, "market_closed_or_rollover")

    assert _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot={**good_spot, "bid": None},
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (False, "bid_ask_unavailable")

    assert _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot={"price": 4350.0, "bid": 4349.0, "ask": 4351.0, "is_stale": False},
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (False, "spread_too_wide")

    assert _entry_gate_reason(
        candidate="short_setup",
        fusion_state="setup_macro_conflict",
        spot=good_spot,
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (False, "setup_macro_conflict")

    assert _entry_gate_reason(
        candidate="short_setup",
        fusion_state="setup_macro_neutral",
        spot=good_spot,
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (True, "")

    assert _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot=good_spot,
        has_open_position=False,
        max_spread_bps=3.0,
    ) == (True, "")



def test_calibration_metrics_report_brier_and_ece():
    metrics = _calibration_metrics([(0.8, 1), (0.2, 0)])
    assert metrics["calibration_sample_count"] == 2
    assert metrics["brier_score"] == pytest.approx(0.04, abs=1e-4)
    assert metrics["expected_calibration_error"] == pytest.approx(0.2, abs=1e-4)


def test_trade_autopsy_distinguishes_timing_failure_from_direction_failure():
    position = SimpleNamespace(
        risk_usd=100.0,
        mfe_usd=60.0,
        mae_usd=-100.0,
    )
    signal_meta = {
        "cognitive_confidence": 0.80,
        "cognition": {
            "regime": {"label": "trend_bear"},
            "hypotheses": [{"name": "trend_continuation", "weight": 0.5}],
            "adversarial": {"counter_evidence": ["transition_risk"]},
            "confidence": {"calibrated_confidence": 0.80},
        },
    }

    autopsy = _trade_autopsy(
        position,
        signal_meta,
        exit_reason="stop_loss",
        pnl=-100.0,
        r_multiple=-1.0,
    )

    assert autopsy["outcome"] == "loss"
    assert autopsy["primary_attribution"] == "entry_timing_or_stop_too_tight"
    assert "counter_evidence_present_at_entry" in autopsy["attributions"]
    assert "high_confidence_error" in autopsy["attributions"]
    assert autopsy["mfe_r"] == 0.6
    assert autopsy["calibration_outcome"] == 0



def test_calibration_metrics_compute_brier_and_ece():
    metrics = _calibration_metrics([
        (0.8, 1),
        (0.7, 1),
        (0.3, 0),
        (0.2, 0),
    ])
    assert metrics["calibration_sample_count"] == 4
    assert metrics["brier_score"] is not None
    assert 0.0 <= metrics["brier_score"] <= 1.0
    assert metrics["expected_calibration_error"] is not None
    assert 0.0 <= metrics["expected_calibration_error"] <= 1.0


def test_trade_autopsy_identifies_directional_failure():
    position = SimpleNamespace(
        risk_usd=100.0,
        mfe_usd=10.0,
        mae_usd=-100.0,
    )
    signal_meta = {
        "cognitive_confidence": 0.78,
        "cognition": {
            "regime": {"label": "trend_bear"},
            "hypotheses": [{"name": "trend_continuation", "weight": 0.55}],
            "adversarial": {"counter_evidence": []},
            "confidence": {"calibrated_confidence": 0.78},
        },
    }
    autopsy = _trade_autopsy(
        position,
        signal_meta,
        exit_reason="stop_loss",
        pnl=-100.0,
        r_multiple=-1.0,
    )
    assert autopsy["outcome"] == "loss"
    assert autopsy["primary_attribution"] == "directional_thesis_failed"
    assert "high_confidence_error" in autopsy["attributions"]
    assert autopsy["calibration_outcome"] == 0


def test_trade_autopsy_identifies_entry_timing_when_trade_had_mfe():
    position = SimpleNamespace(
        risk_usd=100.0,
        mfe_usd=70.0,
        mae_usd=-100.0,
    )
    autopsy = _trade_autopsy(
        position,
        {
            "cognitive_confidence": 0.62,
            "cognition": {
                "regime": {"label": "trend_bull"},
                "hypotheses": [{"name": "trend_continuation"}],
                "adversarial": {"counter_evidence": []},
            },
        },
        exit_reason="stop_loss",
        pnl=-100.0,
        r_multiple=-1.0,
    )
    assert autopsy["primary_attribution"] == "entry_timing_or_stop_too_tight"
    assert autopsy["mfe_r"] == 0.7



def test_context_mark_uses_fresh_analysis_reference_when_fill_quote_is_stale():
    price, kind = _paper_context_mark_price(
        "short",
        {
            "price": 4342.8,
            "bid": None,
            "ask": None,
            "is_stale": True,
        },
        {
            "price": 4345.0,
            "is_stale": False,
            "kind": "micro_fallback",
        },
    )
    assert price == 4345.0
    assert kind == "analysis_reference"


def test_context_mark_never_turns_analysis_reference_into_exit_quote():
    spot = {
        "price": 4342.8,
        "bid": None,
        "ask": None,
        "is_stale": True,
    }
    analysis = {"price": 4345.0, "is_stale": False}
    mark, kind = _paper_context_mark_price("long", spot, analysis)

    assert mark == 4345.0
    assert kind == "analysis_reference"
    assert _paper_exit_quote("long", spot) is None
    assert _paper_exit_quote("short", spot) is None


def test_mark_position_updates_equity_from_analysis_reference_without_fill():
    engine = XAUPaperTradingEngine(Settings())
    account = SimpleNamespace(
        initial_capital=10_000.0,
        realized_pnl=0.0,
        current_equity=10_000.0,
        peak_equity=10_000.0,
        max_drawdown_pct=0.0,
    )
    position = SimpleNamespace(
        side="short",
        entry_price=4350.0,
        quantity_oz=2.0,
        current_price=4350.0,
        unrealized_pnl=0.0,
        mfe_usd=0.0,
        mae_usd=0.0,
    )

    mark = engine._mark_position(
        account,
        position,
        {"price": 4350.0, "bid": None, "ask": None, "is_stale": True},
        {"price": 4340.0, "is_stale": False},
    )

    assert mark == 4340.0
    assert position.unrealized_pnl == 20.0
    assert account.current_equity == 10020.0



def test_shadow_metrics_measure_false_negative_opportunity_cost():
    signals = [
        SimpleNamespace(
            candidate="long_setup",
            accepted=False,
            meta={"shadow": {"15m": {"directional_return_bps": 12.0}}},
        ),
        SimpleNamespace(
            candidate="long_setup",
            accepted=False,
            meta={"shadow": {"15m": {"directional_return_bps": -4.0}}},
        ),
        SimpleNamespace(
            candidate="short_setup",
            accepted=True,
            meta={"shadow": {"15m": {"directional_return_bps": 8.0}}},
        ),
    ]

    metrics = _shadow_metrics(signals)

    assert metrics["15m"]["rejected"]["count"] == 2
    assert metrics["15m"]["rejected"]["positive_rate"] == 0.5
    assert metrics["15m"]["rejected"]["average_directional_return_bps"] == 4.0
    assert metrics["15m"]["accepted"]["count"] == 1
    assert metrics["15m"]["accepted"]["positive_rate"] == 1.0


def test_shadow_metrics_ignore_non_directional_observations():
    signals = [
        SimpleNamespace(
            candidate="none",
            accepted=False,
            meta={"shadow": {"15m": {"directional_return_bps": 100.0}}},
        ),
    ]
    assert _shadow_metrics(signals) == {}



def test_shadow_horizon_only_measures_near_target_time():
    due, grace = _shadow_horizon_due(15.5, 15)
    assert due is True
    assert grace == 2.0

    due_late, _ = _shadow_horizon_due(25.0, 15)
    assert due_late is False

    due_60, grace_60 = _shadow_horizon_due(64.0, 60)
    assert due_60 is True
    assert grace_60 == 6.0

    due_restart, _ = _shadow_horizon_due(240.0, 15)
    assert due_restart is False



def _guardian_position(side="long"):
    return SimpleNamespace(
        side=side,
        entry_price=100.0,
        quantity_oz=10.0,
        risk_usd=100.0,
        stop_loss=90.0 if side == "long" else 110.0,
        target_price=120.0 if side == "long" else 80.0,
    )


def test_position_guardian_moves_stop_to_breakeven_after_one_r():
    position = _guardian_position("long")
    result = _position_guardian(
        position,
        {
            "technical_candidate": "none",
            "paper_entry_allowed": False,
            "meta_decision": "observe",
            "cognitive_confidence": 0.55,
        },
        110.0,
    )

    assert result["action"] == "tighten_stop"
    assert result["reason"] == "breakeven_after_1r"
    assert result["new_stop_loss"] == 100.0
    assert result["exit_requested"] is False


def test_position_guardian_locks_half_r_after_one_and_half_r():
    position = _guardian_position("long")
    result = _position_guardian(
        position,
        {
            "technical_candidate": "none",
            "paper_entry_allowed": False,
            "meta_decision": "observe",
            "cognitive_confidence": 0.55,
        },
        116.0,
    )

    assert result["action"] == "tighten_stop"
    assert result["reason"] == "lock_half_r_after_1_5r"
    assert result["new_stop_loss"] == 105.0


def test_position_guardian_requests_exit_only_for_qualified_opposite_thesis():
    position = _guardian_position("long")
    qualified = _position_guardian(
        position,
        {
            "technical_candidate": "short_setup",
            "paper_entry_allowed": True,
            "meta_decision": "eligible",
            "cognitive_confidence": 0.70,
            "cognition": {
                "data_quality": {"score": 0.90},
                "meta_controller": {"min_confidence": 0.58},
            },
        },
        98.0,
    )
    weak = _position_guardian(
        position,
        {
            "technical_candidate": "short_setup",
            "paper_entry_allowed": True,
            "meta_decision": "eligible",
            "cognitive_confidence": 0.55,
            "cognition": {
                "data_quality": {"score": 0.90},
                "meta_controller": {"min_confidence": 0.58},
            },
        },
        98.0,
    )

    assert qualified["action"] == "exit"
    assert qualified["exit_requested"] is True
    assert qualified["exit_reason"] == "thesis_reversal"
    assert weak["exit_requested"] is False


def test_position_guardian_short_side_tightens_symmetrically():
    position = _guardian_position("short")
    result = _position_guardian(
        position,
        {
            "technical_candidate": "none",
            "paper_entry_allowed": False,
            "meta_decision": "observe",
            "cognitive_confidence": 0.55,
        },
        84.0,
    )

    assert result["current_r"] == 1.6
    assert result["new_stop_loss"] == 95.0



def test_contextual_mark_does_not_contaminate_execution_extremes():
    engine = XAUPaperTradingEngine(Settings())
    account = SimpleNamespace(
        initial_capital=10_000.0,
        realized_pnl=0.0,
        current_equity=10_000.0,
        peak_equity=10_000.0,
        max_drawdown_pct=1.25,
    )
    position = SimpleNamespace(
        side="long",
        entry_price=4350.0,
        quantity_oz=2.0,
        current_price=4350.0,
        unrealized_pnl=0.0,
        mfe_usd=25.0,
        mae_usd=-15.0,
    )

    mark = engine._mark_position(
        account,
        position,
        {"price": 4350.0, "bid": None, "ask": None, "is_stale": True},
        {"price": 4325.0, "is_stale": False},
    )

    assert mark == 4325.0
    assert position.unrealized_pnl == -50.0
    assert account.current_equity == 9950.0
    assert position.mfe_usd == 25.0
    assert position.mae_usd == -15.0
    assert account.peak_equity == 10_000.0
    assert account.max_drawdown_pct == 1.25


def test_weekly_reset_never_manufactures_mid_only_or_stale_fill():
    assert _weekly_reset_fill_price(
        "long",
        {"price": 4340.0, "bid": None, "ask": None, "is_stale": False},
    ) is None
    assert _weekly_reset_fill_price(
        "short",
        {"price": 4340.0, "bid": 4339.9, "ask": 4340.1, "is_stale": True},
    ) is None
    assert _weekly_reset_fill_price(
        "long",
        {"price": 4340.0, "bid": 4339.9, "ask": 4340.1, "is_stale": False},
    ) == 4339.9
    assert _weekly_reset_fill_price(
        "short",
        {"price": 4340.0, "bid": 4339.9, "ask": 4340.1, "is_stale": False},
    ) == 4340.1



def test_entry_gate_reports_market_rollover_explicitly():
    allowed, reason = _entry_gate_reason(
        candidate="long_setup",
        fusion_state="setup_macro_support",
        spot={
            "price": 4350.0,
            "bid": None,
            "ask": None,
            "is_stale": False,
            "fill_state": "market_closed_or_rollover",
        },
        has_open_position=False,
        max_spread_bps=3.0,
    )
    assert allowed is False
    assert reason == "market_closed_or_rollover"



def test_position_guardian_requires_quality_for_thesis_reversal():
    position = _guardian_position("long")
    result = _position_guardian(
        position,
        {
            "technical_candidate": "short_setup",
            "paper_entry_allowed": True,
            "meta_decision": "eligible",
            "cognitive_confidence": 0.80,
            "cognition": {
                "data_quality": {"score": 0.60},
                "meta_controller": {"min_confidence": 0.58},
            },
        },
        98.0,
    )
    assert result["exit_requested"] is False


def test_position_guardian_uses_reversal_hysteresis_above_entry_threshold():
    position = _guardian_position("long")
    below = _position_guardian(
        position,
        {
            "technical_candidate": "short_setup",
            "paper_entry_allowed": True,
            "meta_decision": "eligible",
            "cognitive_confidence": 0.67,
            "cognition": {
                "data_quality": {"score": 0.90},
                "meta_controller": {"min_confidence": 0.58},
            },
        },
        98.0,
    )
    above = _position_guardian(
        position,
        {
            "technical_candidate": "short_setup",
            "paper_entry_allowed": True,
            "meta_decision": "eligible",
            "cognitive_confidence": 0.70,
            "cognition": {
                "data_quality": {"score": 0.90},
                "meta_controller": {"min_confidence": 0.58},
            },
        },
        98.0,
    )
    assert below["exit_requested"] is False
    assert above["exit_requested"] is True
    assert above["reversal_threshold"] == 0.68



def test_position_management_requires_explicit_ready_fill_state():
    closed = {
        "price": 4340.0,
        "bid": 4339.9,
        "ask": 4340.1,
        "is_stale": False,
        "fill_state": "market_closed_or_rollover",
    }
    ready = {
        **closed,
        "fill_state": "ready",
    }

    assert _paper_management_quote("long", closed) is None
    assert _paper_management_quote("short", closed) is None
    assert _paper_management_quote("long", ready) == 4339.9
    assert _paper_management_quote("short", ready) == 4340.1
    assert _weekly_reset_fill_price("long", closed) is None
    assert _weekly_reset_fill_price("long", ready) == 4339.9



def _qualified_reversal_management():
    return {
        "action": "exit",
        "reason": "qualified_opposite_thesis",
        "exit_requested": True,
        "exit_reason": "thesis_reversal",
        "opposite_candidate": "short_setup",
        "confidence": 0.75,
        "reversal_threshold": 0.68,
        "data_quality": 0.90,
    }


def test_reversal_exit_requires_two_consecutive_confirmations():
    streaks = {}
    first = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="obs-1",
        required=2,
    )
    second = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="obs-2",
        required=2,
    )

    assert first["exit_requested"] is False
    assert first["reason"] == "opposite_thesis_confirmation_pending"
    assert first["confirmation_streak"] == 1
    assert first["confirmation_required"] == 2
    assert first["proposed_exit_reason"] == "thesis_reversal"

    assert second["exit_requested"] is True
    assert second["exit_reason"] == "thesis_reversal"
    assert second["reason"] == "confirmed_opposite_thesis"
    assert second["confirmation_streak"] == 2
    assert "setup-1" not in streaks


def test_reversal_confirmation_resets_on_interruption():
    streaks = {}
    _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="obs-a",
        required=2,
    )

    neutral = _confirm_reversal_exit(
        "setup-1",
        {
            "action": "hold",
            "reason": "no_management_trigger",
            "exit_requested": False,
            "exit_reason": None,
        },
        streaks,
        required=2,
    )
    restarted = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="obs-b",
        required=2,
    )

    assert neutral["confirmation_streak"] == 0
    assert restarted["exit_requested"] is False
    assert restarted["confirmation_streak"] == 1


def test_reversal_confirmation_resets_when_fill_quote_disappears():
    streaks = {}
    _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="obs-a",
        required=2,
    )
    missing = _confirm_reversal_exit(
        "setup-1",
        None,
        streaks,
        required=2,
    )

    assert missing is None
    assert "setup-1" not in streaks



def test_reversal_confirmation_does_not_double_count_same_market_observation():
    streaks = {}
    first = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="micro-tick-100",
        required=2,
    )
    duplicate = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="micro-tick-100",
        required=2,
    )
    fresh = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="micro-tick-101",
        required=2,
    )

    assert first["confirmation_streak"] == 1
    assert first["exit_requested"] is False
    assert duplicate["confirmation_streak"] == 1
    assert duplicate["exit_requested"] is False
    assert duplicate["confirmation_observation_reused"] is True
    assert duplicate["reason"] == "opposite_thesis_waiting_new_observation"
    assert fresh["confirmation_streak"] == 2
    assert fresh["exit_requested"] is True



def test_reversal_confirmation_never_advances_without_observation_id():
    streaks = {}
    first = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id=None,
        required=2,
    )
    second = _confirm_reversal_exit(
        "setup-1",
        _qualified_reversal_management(),
        streaks,
        observation_id="",
        required=2,
    )

    assert first["exit_requested"] is False
    assert first["confirmation_streak"] == 0
    assert first["confirmation_observation_missing"] is True
    assert first["reason"] == "opposite_thesis_missing_observation"
    assert second["exit_requested"] is False
    assert second["confirmation_streak"] == 0
    assert "setup-1" not in streaks



def test_shadow_research_memory_is_separate_and_similarity_filtered():
    base_vector = {
        "candidate": "long_setup",
        "alignment": "bullish",
        "session": "new_york",
        "regime": "trend_bull",
        "directional_pressure": 0.8,
        "return_10m_pct": 0.1,
        "return_30m_pct": 0.2,
        "acceleration": 0.03,
        "volatility_pct": 0.15,
        "rsi_5m_norm": 0.2,
        "rsi_15m_norm": 0.1,
        "breakout": "up",
        "macro_bias": 1.0,
        "macro_confidence": 0.7,
        "spread_bps": 0.5,
        "spot_proxy_basis_bps": 0.0,
        "data_quality": 0.95,
    }
    signals = [
        SimpleNamespace(
            accepted=False,
            rejection_reason="cognitive_veto",
            meta={
                "state_vector": dict(base_vector),
                "shadow": {
                    "60m": {"directional_return_bps": 12.0},
                },
            }
        ),
        SimpleNamespace(
            accepted=False,
            rejection_reason="cognitive_wait",
            meta={
                "state_vector": {**base_vector, "directional_pressure": 0.7},
                "shadow": {
                    "30m": {"directional_return_bps": -4.0},
                },
            }
        ),
        SimpleNamespace(
            accepted=False,
            rejection_reason="cognitive_observe",
            meta={
                "state_vector": {
                    **base_vector,
                    "candidate": "short_setup",
                    "alignment": "bearish",
                    "directional_pressure": -1.0,
                    "macro_bias": -1.0,
                },
                "shadow": {
                    "60m": {"directional_return_bps": 100.0},
                },
            }
        ),
    ]

    memory = _shadow_research_memory(signals, base_vector)

    assert memory["research_only"] is True
    assert memory["sample_count"] == 2
    assert memory["positive_rate"] == 0.5
    assert memory["nearest_similarity"] >= memory["average_similarity"]
    assert memory["horizon_mix"] == {"60m": 1, "30m": 1}



def test_replay_research_memory_filters_by_state_similarity():
    base_vector = {
        "candidate": "short_setup",
        "alignment": "bearish",
        "session": "new_york",
        "regime": "trend_bear",
        "directional_pressure": -0.8,
        "return_10m_pct": -0.1,
        "return_30m_pct": -0.2,
        "acceleration": -0.03,
        "volatility_pct": 0.15,
        "rsi_5m_norm": -0.2,
        "rsi_15m_norm": -0.1,
        "breakout": "down",
        "macro_bias": -1.0,
        "macro_confidence": 0.7,
        "spread_bps": 0.5,
        "spot_proxy_basis_bps": 0.0,
        "data_quality": 0.95,
    }
    episodes = [
        SimpleNamespace(state_vector=dict(base_vector), directional_return_bps=10.0),
        SimpleNamespace(
            state_vector={**base_vector, "directional_pressure": -0.7},
            directional_return_bps=-2.0,
        ),
        SimpleNamespace(
            state_vector={
                **base_vector,
                "candidate": "long_setup",
                "alignment": "bullish",
                "directional_pressure": 1.0,
                "macro_bias": 1.0,
            },
            directional_return_bps=100.0,
        ),
    ]

    memory = _replay_research_memory(episodes, base_vector)

    assert memory["research_only"] is True
    assert memory["lookahead_protected"] is True
    assert memory["sample_count"] == 2
    assert memory["positive_rate"] == 0.5
    assert memory["nearest_similarity"] >= memory["average_similarity"]



def test_shadow_research_memory_excludes_execution_and_accepted_signals():
    vector = {
        "candidate": "long_setup",
        "alignment": "bullish",
        "session": "new_york",
        "regime": "trend_bull",
        "directional_pressure": 0.8,
        "return_10m_pct": 0.1,
        "return_30m_pct": 0.2,
        "acceleration": 0.03,
        "volatility_pct": 0.15,
        "rsi_5m_norm": 0.2,
        "rsi_15m_norm": 0.1,
        "breakout": "up",
        "macro_bias": 1.0,
        "macro_confidence": 0.7,
        "spread_bps": 0.5,
        "spot_proxy_basis_bps": 0.0,
        "data_quality": 0.95,
    }
    signals = [
        SimpleNamespace(
            accepted=True,
            rejection_reason="",
            meta={"state_vector": dict(vector), "shadow": {"60m": {"directional_return_bps": 99.0}}},
        ),
        SimpleNamespace(
            accepted=False,
            rejection_reason="market_closed_or_rollover",
            meta={"state_vector": dict(vector), "shadow": {"60m": {"directional_return_bps": 88.0}}},
        ),
        SimpleNamespace(
            accepted=False,
            rejection_reason="spread_too_wide",
            meta={"state_vector": dict(vector), "shadow": {"60m": {"directional_return_bps": 77.0}}},
        ),
        SimpleNamespace(
            accepted=False,
            rejection_reason="cognitive_veto",
            meta={"state_vector": dict(vector), "shadow": {"60m": {"directional_return_bps": -6.0}}},
        ),
    ]

    memory = _shadow_research_memory(signals, vector)

    assert memory["sample_count"] == 1
    assert memory["positive_rate"] == 0.0
    assert memory["decision_filtered"] is True
    assert "cognitive_veto" in memory["eligible_rejection_reasons"]



def test_replay_memory_temporally_decorrelates_overlapping_horizons():
    vector = {
        "candidate": "long_setup",
        "alignment": "bullish",
        "session": "new_york",
        "regime": "trend_bull",
        "directional_pressure": 0.8,
        "return_10m_pct": 0.1,
        "return_30m_pct": 0.2,
        "acceleration": 0.03,
        "volatility_pct": 0.15,
        "rsi_5m_norm": 0.2,
        "rsi_15m_norm": 0.1,
        "breakout": "up",
        "macro_bias": 0.0,
        "macro_confidence": 0.0,
        "spread_bps": 0.0,
        "spot_proxy_basis_bps": 0.0,
        "data_quality": 0.95,
    }
    start = datetime(2026, 1, 5, 12, 0)
    episodes = []
    for i in range(12):
        observed = start + timedelta(minutes=5 * i)
        episodes.append(
            SimpleNamespace(
                state_vector=dict(vector),
                directional_return_bps=10.0,
                horizon_minutes=60,
                observed_at=observed,
                outcome_at=observed + timedelta(minutes=60),
                meta={},
            )
        )

    memory = _replay_research_memory(episodes, vector)

    assert memory["raw_sample_count"] == 12
    assert memory["sample_count"] == 1
    assert memory["overlap_discarded"] == 11
    assert memory["temporally_decorrelated"] is True

    # Production compatibility storage combines a naive SQL timestamp with
    # an offset-bearing ISO string inside the JSON replay payload.
    compat = [SimpleNamespace(
        observed_at=episode.observed_at,
        meta={"replay_episode": {
            "state_vector": episode.state_vector,
            "directional_return_bps": episode.directional_return_bps,
            "outcome_at": episode.outcome_at.replace(tzinfo=timezone.utc).isoformat(),
        }},
    ) for episode in episodes]
    compat_memory = _replay_research_memory(compat, vector)
    assert compat_memory["sample_count"] == 1
    assert compat_memory["overlap_discarded"] == 11


def test_shadow_memory_temporally_decorrelates_overlapping_decisions():
    vector = {
        "candidate": "short_setup",
        "alignment": "bearish",
        "session": "new_york",
        "regime": "trend_bear",
        "directional_pressure": -0.8,
        "return_10m_pct": -0.1,
        "return_30m_pct": -0.2,
        "acceleration": -0.03,
        "volatility_pct": 0.15,
        "rsi_5m_norm": -0.2,
        "rsi_15m_norm": -0.1,
        "breakout": "down",
        "macro_bias": -1.0,
        "macro_confidence": 0.7,
        "spread_bps": 0.5,
        "spot_proxy_basis_bps": 0.0,
        "data_quality": 0.95,
    }
    start = datetime(2026, 1, 5, 12, 0)
    signals = []
    for i in range(6):
        observed = start + timedelta(minutes=10 * i)
        signals.append(
            SimpleNamespace(
                accepted=False,
                rejection_reason="cognitive_wait",
                observed_at=observed,
                meta={
                    "state_vector": dict(vector),
                    "shadow": {"60m": {"directional_return_bps": 5.0}},
                },
            )
        )

    memory = _shadow_research_memory(signals, vector)

    assert memory["raw_sample_count"] == 6
    assert memory["sample_count"] == 1
    assert memory["overlap_discarded"] == 5
    assert memory["temporally_decorrelated"] is True



def test_replay_memory_does_not_mix_spot_and_futures_source_families():
    base_vector = {
        "source_family": "xau_spot_structure",
        "candidate": "long_setup",
        "alignment": "bullish",
        "session": "london_ny_overlap",
        "regime": "trend_bull",
        "breakout": "up",
        "directional_pressure": 1.0,
        "return_10m_pct": 0.10,
        "return_30m_pct": 0.20,
        "acceleration": 0.03,
        "volatility_pct": 0.15,
        "rsi_5m_norm": 0.20,
        "rsi_15m_norm": 0.10,
        "macro_bias": 0.0,
        "macro_confidence": 0.0,
        "spread_bps": 0.5,
        "spot_proxy_basis_bps": 0.0,
        "spot_consensus_delta_bps": 0.0,
        "data_quality": 1.0,
    }
    t0 = datetime(2026, 9, 21, 12, 0)
    spot_episode = SimpleNamespace(
        state_vector=dict(base_vector),
        directional_return_bps=12.0,
        observed_at=t0,
        outcome_at=t0 + timedelta(minutes=60),
        horizon_minutes=60,
        source="biquote.io:MT5-ohlc:walk-forward",
        meta={},
    )
    futures_vector = dict(base_vector)
    futures_vector["source_family"] = "gc_futures_proxy"
    futures_episode = SimpleNamespace(
        state_vector=futures_vector,
        directional_return_bps=-30.0,
        observed_at=t0 + timedelta(hours=2),
        outcome_at=t0 + timedelta(hours=3),
        horizon_minutes=60,
        source="yfinance:GC=F:walk-forward",
        meta={},
    )

    memory = _replay_research_memory(
        [spot_episode, futures_episode],
        base_vector,
        similarity_floor=0.70,
        limit=20,
    )

    assert memory["sample_count"] == 1
    assert memory["positive_rate"] == 1.0
    assert memory["source_family"] == "xau_spot_structure"
    assert memory["source_filtered"] is True
    assert memory["source_mismatch_discarded"] == 1



def test_calibration_metrics_expose_empirical_bins_for_gen1_shrinkage():
    metrics = _calibration_metrics(
        [(0.72, 1)] * 10 + [(0.75, 0)] * 4 + [(0.32, 0)] * 10
    )
    assert len(metrics["calibration_bins"]) == 5
    high = metrics["calibration_bins"][3]
    assert high["count"] == 14
    assert high["observed_rate"] == pytest.approx(10 / 14, abs=1e-4)
