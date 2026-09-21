import asyncio
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
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe
from src.platform.persistence.database import Base
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperSignal,
    XAUReplayEpisode,
)


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
    assert all(history[tf] for tf in XAUTimeframe)
    assert all(
        row.source == "yfinance:GC=F"
        for tf in XAUTimeframe
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
