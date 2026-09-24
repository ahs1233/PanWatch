from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.modules.xau import paper_store
from src.platform.persistence.database import Base, SessionLocal
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperPosition,
    XAUPaperSignal,
    XAUPaperTrade,
    XAUReplayEpisode,
)
from src.platform.runtime.config import Settings


def _source_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(
        bind=engine,
        tables=[
            XAUPaperAccount.__table__,
            XAUPaperSignal.__table__,
            XAUPaperPosition.__table__,
            XAUPaperTrade.__table__,
            XAUReplayEpisode.__table__,
        ],
    )
    return engine, sessionmaker(bind=engine, expire_on_commit=False)


def _seed_source(Session):
    db = Session()
    try:
        account = XAUPaperAccount(
            week_key="2026-W39",
            initial_capital=10_000.0,
            realized_pnl=125.0,
            current_equity=10_125.0,
            peak_equity=10_150.0,
            max_drawdown_pct=0.5,
            total_trades=1,
            winning_trades=1,
            losing_trades=0,
            status="active",
            started_at=datetime(2026, 9, 21, 0, 0),
        )
        db.add(account)
        db.flush()

        signal = XAUPaperSignal(
            account_id=account.id,
            setup_key="setup-1",
            candidate="long_setup",
            fusion_state="setup_macro_support",
            macro_relation="supportive",
            event_risk=False,
            price=4300.0,
            accepted=True,
            rejection_reason="",
            observed_at=datetime(2026, 9, 24, 7, 0),
            meta={"source": "local-primary"},
        )
        position = XAUPaperPosition(
            account_id=account.id,
            setup_key="setup-1",
            side="long",
            quantity_oz=1.0,
            entry_price=4300.0,
            stop_loss=4290.0,
            target_price=4320.0,
            current_price=4310.0,
            unrealized_pnl=10.0,
            mfe_usd=15.0,
            mae_usd=-3.0,
            risk_usd=10.0,
            setup_state="setup_macro_support",
            macro_relation="supportive",
            price_source="test",
            status="closed",
            opened_at=datetime(2026, 9, 24, 7, 0),
            closed_at=datetime(2026, 9, 24, 7, 30),
        )
        trade = XAUPaperTrade(
            account_id=account.id,
            setup_key="setup-1",
            side="long",
            quantity_oz=1.0,
            entry_price=4300.0,
            exit_price=4310.0,
            stop_loss=4290.0,
            target_price=4320.0,
            pnl=10.0,
            pnl_pct_equity=0.1,
            r_multiple=1.0,
            mfe_usd=15.0,
            mae_usd=-3.0,
            risk_usd=10.0,
            exit_reason="time_stop",
            setup_state="setup_macro_support",
            macro_relation="supportive",
            price_source="test",
            opened_at=datetime(2026, 9, 24, 7, 0),
            closed_at=datetime(2026, 9, 24, 7, 30),
            meta={"source": "local-primary"},
        )
        replay = XAUReplayEpisode(
            replay_key="replay-1",
            candidate="long_setup",
            regime="trend_bull",
            confidence=0.7,
            horizon_minutes=60,
            entry_price=4300.0,
            outcome_price=4310.0,
            directional_return_bps=23.25,
            positive=True,
            source="test",
            observed_at=datetime(2026, 9, 23, 6, 0),
            outcome_at=datetime(2026, 9, 23, 7, 0),
            state_vector={"candidate": "long_setup"},
            cognition={"confidence": 0.7},
            meta={"research_only": True},
        )
        db.add_all([signal, position, trade, replay])
        db.commit()
    finally:
        db.close()


def test_local_primary_never_contacts_external_on_startup(monkeypatch):
    monkeypatch.setattr(paper_store, "_external_engine", None)
    monkeypatch.setattr(paper_store, "XAUPaperSessionLocal", SessionLocal)
    monkeypatch.setattr(paper_store, "XAUReplaySessionLocal", SessionLocal)

    def must_not_connect(_url):
        raise AssertionError("local_primary startup must not contact replica")

    monkeypatch.setattr(paper_store, "_create_store_engine", must_not_connect)

    settings = Settings(
        xau_paper_storage_mode="local_primary",
        xau_paper_data_persistent=True,
        xau_paper_database_url="postgresql://replica.invalid/panwatch",
    )

    assert paper_store.init_xau_paper_store(settings) is False
    health = paper_store.paper_storage_health(settings)
    assert health["primary_storage"] == "local_sqlite"
    assert health["primary_persistent"] is True
    assert health["external_replica_configured"] is True
    assert paper_store.paper_store_is_external() is False


def test_replica_sync_is_idempotent_and_updates_existing_rows(monkeypatch, tmp_path):
    source_engine, Source = _source_session_factory()
    _seed_source(Source)
    monkeypatch.setattr(paper_store, "SessionLocal", Source)

    target_path = tmp_path / "replica.sqlite3"
    settings = Settings(
        xau_paper_storage_mode="local_primary",
        xau_paper_data_persistent=True,
        xau_paper_replica_sync_enabled=True,
        xau_paper_database_url=f"sqlite:///{target_path}",
    )

    first = paper_store.sync_xau_paper_replica(settings)
    second = paper_store.sync_xau_paper_replica(settings)

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert first["accounts"]["inserted"] == 1
    assert second["accounts"]["inserted"] == 0
    assert second["accounts"]["updated"] == 1
    assert second["target_conflicts"] == 0

    target_engine = create_engine(f"sqlite:///{target_path}")
    Target = sessionmaker(bind=target_engine)
    db = Target()
    try:
        assert db.query(XAUPaperAccount).count() == 1
        assert db.query(XAUPaperSignal).count() == 1
        assert db.query(XAUPaperPosition).count() == 1
        assert db.query(XAUPaperTrade).count() == 1
        assert db.query(XAUReplayEpisode).count() == 1
        assert db.query(XAUPaperAccount).one().current_equity == 10_125.0
    finally:
        db.close()
        target_engine.dispose()
        source_engine.dispose()


def test_replica_outage_never_breaks_local_primary(monkeypatch):
    monkeypatch.setattr(paper_store, "_external_engine", None)
    monkeypatch.setattr(paper_store, "XAUPaperSessionLocal", SessionLocal)
    monkeypatch.setattr(paper_store, "XAUReplaySessionLocal", SessionLocal)

    def fail_connect(_url):
        raise RuntimeError("replica quota exceeded")

    monkeypatch.setattr(paper_store, "_create_store_engine", fail_connect)

    settings = Settings(
        xau_paper_storage_mode="local_primary",
        xau_paper_data_persistent=True,
        xau_paper_replica_sync_enabled=True,
        xau_paper_database_url="postgresql://replica.invalid/panwatch",
    )

    assert paper_store.init_xau_paper_store(settings) is False
    result = paper_store.sync_xau_paper_replica(settings)

    assert result["status"] == "degraded"
    assert result["error_type"] == "RuntimeError"
    assert paper_store.paper_store_is_external() is False
    health = paper_store.paper_storage_health(settings)
    assert health["primary_storage"] == "local_sqlite"
    assert health["primary_persistent"] is True
    assert health["replica_sync"]["status"] == "degraded"
