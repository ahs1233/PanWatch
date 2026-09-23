import asyncio
import threading
from datetime import datetime, timedelta, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.modules.xau import replay
from src.modules.xau.paper_store import _provision_replay_table
from src.modules.xau.replay import (
    _fetch_default_replay_history,
    build_replay_technical_state,
    persist_replay_episodes,
    persist_replay_episodes_in_paper_store,
    walk_forward_replay,
    _future_range_outcomes,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.persistence.database import Base
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperSignal,
    XAUReplayEpisode,
)


def test_replay_processing_does_not_block_live_event_loop(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    event_loop_thread = threading.get_ident()

    async def history(**kwargs):
        return {}, "test"

    def slow_replay(*args, **kwargs):
        assert threading.get_ident() != event_loop_thread
        entered.set()
        assert release.wait(3), "replay blocked the live event loop"
        return {"research_only": True, "execution_allowed": False}

    async def optional_htf(bars, **kwargs):
        return bars

    monkeypatch.setattr(replay, "_fetch_default_replay_history", history)
    monkeypatch.setattr(replay, "_attach_optional_htf_history", optional_htf)
    monkeypatch.setattr(replay, "_replay_and_persist", slow_replay)

    async def exercise():
        task = asyncio.create_task(replay.refresh_replay_memory())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert not task.done()
        finally:
            release.set()
        assert await task == {"research_only": True, "execution_allowed": False}

    asyncio.run(exercise())


def _bars(timeframe: XAUTimeframe, count: int, *, start: datetime, slope: float = 0.2):
    minutes = {
        XAUTimeframe.M1: 1,
        XAUTimeframe.M5: 5,
        XAUTimeframe.M15: 15,
    }[timeframe]
    rows = []
    for i in range(count):
        base = 2500.0 + i * slope
        rows.append(
            XAUBar(
                timestamp=start + timedelta(minutes=i * minutes),
                timeframe=timeframe,
                open=base,
                high=base + 0.35,
                low=base - 0.25,
                close=base + 0.20,
                volume=100.0 + i,
                source=f"synthetic:{timeframe.value}",
                execution_eligible=False,
            )
        )
    return rows


def _history():
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    return {
        XAUTimeframe.M1: _bars(XAUTimeframe.M1, 520, start=start, slope=0.06),
        XAUTimeframe.M5: _bars(XAUTimeframe.M5, 110, start=start, slope=0.30),
        XAUTimeframe.M15: _bars(XAUTimeframe.M15, 40, start=start, slope=0.90),
    }


def test_walk_forward_replay_generates_research_only_directional_episodes():
    episodes = walk_forward_replay(
        _history(),
        horizon_minutes=30,
        step_minutes=5,
        source="synthetic-test",
    )

    assert episodes
    assert all(item.candidate == "long_setup" for item in episodes)
    assert all(item.meta["research_only"] is True for item in episodes)
    assert all(item.meta["lookahead_protected"] is True for item in episodes)
    assert all(item.horizon_minutes == 30 for item in episodes)
    assert all(item.state_vector for item in episodes)
    assert all(item.outcome_at > item.observed_at for item in episodes)


def test_replay_state_does_not_change_when_future_bars_are_mutated():
    history = _history()
    evaluation = datetime(2026, 1, 5, 6, 30, tzinfo=timezone.utc)
    before = build_replay_technical_state(history, evaluation)

    mutated = {}
    for timeframe, rows in history.items():
        duration = {
            XAUTimeframe.M1: timedelta(minutes=1),
            XAUTimeframe.M5: timedelta(minutes=5),
            XAUTimeframe.M15: timedelta(minutes=15),
        }[timeframe]
        next_rows = []
        for bar in rows:
            available_at = bar.timestamp + duration
            if available_at <= evaluation:
                next_rows.append(bar)
                continue
            shock = 1000.0
            next_rows.append(
                XAUBar(
                    timestamp=bar.timestamp,
                    timeframe=bar.timeframe,
                    open=bar.open + shock,
                    high=bar.high + shock,
                    low=bar.low + shock,
                    close=bar.close + shock,
                    volume=bar.volume,
                    source=bar.source,
                    execution_eligible=False,
                )
            )
        mutated[timeframe] = next_rows

    after = build_replay_technical_state(mutated, evaluation)

    assert before["candidate"] == after["candidate"]
    assert before["alignment"] == after["alignment"]
    assert before["analysis_reference"]["price"] == after["analysis_reference"]["price"]
    assert before["frames"] == after["frames"]


def test_15m_bar_is_not_visible_before_its_close_time():
    history = _history()
    last_15m = history[XAUTimeframe.M15][25]
    before_close = last_15m.timestamp + timedelta(minutes=10)
    state = build_replay_technical_state(history, before_close)

    frame = state["frames"].get("15m")
    assert frame is not None
    assert frame["observed_at"] != last_15m.timestamp.isoformat()
    assert datetime.fromisoformat(frame["observed_at"]) < last_15m.timestamp


def test_replay_keys_are_deterministic():
    history = _history()
    first = walk_forward_replay(
        history,
        horizon_minutes=30,
        step_minutes=10,
        source="deterministic-test",
    )
    second = walk_forward_replay(
        history,
        horizon_minutes=30,
        step_minutes=10,
        source="deterministic-test",
    )

    assert [item.replay_key for item in first] == [item.replay_key for item in second]


def test_persist_replay_episodes_is_idempotent():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine, tables=[XAUReplayEpisode.__table__])
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        episodes = walk_forward_replay(
            _history(),
            horizon_minutes=30,
            step_minutes=30,
            source="persist-test",
        )
        assert episodes

        added_first = persist_replay_episodes(db, episodes[:5])
        db.commit()
        added_second = persist_replay_episodes(db, episodes[:5])
        db.commit()

        assert added_first == 5
        assert added_second == 0
        assert db.query(XAUReplayEpisode).count() == 5
    finally:
        db.close()



def test_default_replay_history_does_not_mix_sources_on_partial_biquote_failure(monkeypatch):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)

    def fake_biquote(self, timeframe, *, limit=240, timeout_seconds=12.0):
        if timeframe == XAUTimeframe.M5:
            raise RuntimeError("partial provider failure")
        count = 100 if timeframe != XAUTimeframe.M15 else 40
        return _bars(timeframe, count, start=start, slope=0.1)

    def fake_yahoo(self, timeframe):
        count = {
            XAUTimeframe.M1: 120,
            XAUTimeframe.M5: 80,
            XAUTimeframe.M15: 40,
        }[timeframe]
        rows = _bars(timeframe, count, start=start, slope=0.2)
        return [
            XAUBar(
                timestamp=row.timestamp,
                timeframe=row.timeframe,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                source="yfinance:GC=F",
                execution_eligible=False,
            )
            for row in rows
        ]

    monkeypatch.setattr(replay.BiquoteXAUOHLCProvider, "bars", fake_biquote)
    monkeypatch.setattr(replay.YahooGoldResearchProvider, "bars", fake_yahoo)

    history, source = asyncio.run(_fetch_default_replay_history(limit=1000))

    assert source == "yfinance:GC=F"
    replay_timeframes = (
        XAUTimeframe.M1,
        XAUTimeframe.M5,
        XAUTimeframe.M15,
    )
    assert all(history[tf] for tf in replay_timeframes)
    assert all(
        row.source == "yfinance:GC=F"
        for tf in replay_timeframes
        for row in history[tf]
    )



def test_paper_signal_compat_replay_storage_is_idempotent_and_isolated():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine,
        tables=[
            XAUPaperAccount.__table__,
            XAUPaperSignal.__table__,
        ],
    )
    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        start = datetime(2026, 1, 5, 0, 0)
        account = XAUPaperAccount(
            week_key="2026-W02",
            initial_capital=10_000.0,
            realized_pnl=0.0,
            current_equity=10_000.0,
            peak_equity=10_000.0,
            max_drawdown_pct=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            status="active",
            started_at=start,
        )
        db.add(account)
        db.commit()

        episodes = walk_forward_replay(
            _history(),
            horizon_minutes=30,
            step_minutes=30,
            source="compat-test",
        )
        assert episodes

        added_first = persist_replay_episodes_in_paper_store(db, episodes[:3])
        db.commit()
        added_second = persist_replay_episodes_in_paper_store(db, episodes[:3])
        db.commit()

        rows = (
            db.query(XAUPaperSignal)
            .filter(XAUPaperSignal.rejection_reason == "historical_replay")
            .all()
        )
        assert added_first == 3
        assert added_second == 0
        assert len(rows) == 3
        assert all(row.accepted is False for row in rows)
        assert all(row.candidate.startswith("replay_") for row in rows)
        assert all(row.setup_key.startswith("replay:") for row in rows)
        assert all((row.meta or {}).get("lookahead_protected") is True for row in rows)
    finally:
        db.close()



def test_replay_table_provisioning_is_non_destructive_and_idempotent():
    engine = create_engine("sqlite:///:memory:")

    assert _provision_replay_table(engine) is True
    assert _provision_replay_table(engine) is True

    Session = sessionmaker(bind=engine)
    db = Session()
    try:
        assert db.query(XAUReplayEpisode).count() == 0
    finally:
        db.close()



def test_replay_prefers_deep_biquote_range_history(monkeypatch):
    start = datetime(2026, 9, 15, 0, 0, tzinfo=timezone.utc)
    counts = {
        XAUTimeframe.M1: 400,
        XAUTimeframe.M5: 100,
        XAUTimeframe.M15: 50,
    }

    def fake_range(
        self,
        timeframe,
        *,
        start,
        end,
        timeout_seconds=12.0,
        max_bars_per_request=900,
        max_chunks=64,
    ):
        return _bars(
            timeframe,
            counts[timeframe],
            start=start,
            slope=0.1,
        )

    def should_not_use_recent(self, timeframe, *, limit=240, timeout_seconds=12.0):
        raise AssertionError("recent fallback should not be used when deep range is healthy")

    monkeypatch.setattr(
        replay.BiquoteXAUOHLCProvider,
        "bars_range",
        fake_range,
    )
    monkeypatch.setattr(
        replay.BiquoteXAUOHLCProvider,
        "bars",
        should_not_use_recent,
    )

    history, source = asyncio.run(
        _fetch_default_replay_history(
            limit=1000,
            lookback_days=5,
        )
    )

    assert source == "biquote.io:MT5-ohlc-range"
    assert len(history[XAUTimeframe.M1]) == 400
    assert len(history[XAUTimeframe.M5]) == 100
    assert len(history[XAUTimeframe.M15]) == 50



def test_future_range_outcomes_records_first_touch_without_lookahead_into_decision():
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    bars = _bars(XAUTimeframe.M1, 90, start=start, slope=0.4)
    evaluation = bars[30].timestamp + timedelta(minutes=1)
    entry = bars[30].close
    result = _future_range_outcomes(
        bars,
        evaluation,
        entry,
        horizon_minutes=45,
    )
    assert result["lookahead_used_for_label_only"] is True
    assert set(result["levels"]) == {"pm10", "pm20", "pm30"}
    assert result["max_up_usd"] > 0


def test_walk_forward_replay_records_gen1_scope_and_sensor_gaps():
    def historical_macro(at):
        return {
            "bias": 1,
            "confidence": 0.7,
            "event_risk": False,
            "observed_at": at.isoformat(),
            "calendar_ok": True,
            "search_ok": True,
            "synthesis_ok": True,
        }

    episodes = walk_forward_replay(
        _history(),
        horizon_minutes=30,
        step_minutes=15,
        source="gen1-replay-test",
        macro_provider=historical_macro,
    )
    assert episodes
    assert all("gen1_decision" in item.meta for item in episodes)
    assert all("range_outcomes" in item.meta for item in episodes)
    assert all(
        "historical_xaut_microstructure_unavailable" in item.meta["sensor_gaps"]
        for item in episodes
    )
    assert all(
        item.meta["replay_scope"] == "full_gen1_except_historical_xaut_microstructure"
        for item in episodes
    )



def test_optional_htf_history_preserves_yahoo_source_family(monkeypatch):
    start = datetime(2026, 1, 5, 0, 0, tzinfo=timezone.utc)
    core = {
        XAUTimeframe.M1: _bars(XAUTimeframe.M1, 60, start=start, slope=0.1),
        XAUTimeframe.M5: _bars(XAUTimeframe.M5, 40, start=start, slope=0.1),
        XAUTimeframe.M15: _bars(XAUTimeframe.M15, 30, start=start, slope=0.1),
    }

    def yahoo(self, timeframe):
        rows = _bars(
            timeframe,
            60 if timeframe == XAUTimeframe.H1 else 40,
            start=start,
            slope=0.1,
        )
        return [
            XAUBar(
                timestamp=row.timestamp,
                timeframe=row.timeframe,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
                source="yfinance:GC=F",
                execution_eligible=False,
            )
            for row in rows
        ]

    def no_biquote(*args, **kwargs):
        raise AssertionError("Biquote must not be mixed into Yahoo replay")

    monkeypatch.setattr(replay.YahooGoldResearchProvider, "bars", yahoo)
    monkeypatch.setattr(replay.BiquoteXAUOHLCProvider, "bars", no_biquote)
    enriched = asyncio.run(
        replay._attach_optional_htf_history(
            core,
            lookback_days=0,
            source="yfinance:GC=F",
        )
    )
    assert enriched[XAUTimeframe.H1]
    assert enriched[XAUTimeframe.D1]
    assert enriched[XAUTimeframe.H4] == []
    assert all(row.source == "yfinance:GC=F" for row in enriched[XAUTimeframe.H1])
