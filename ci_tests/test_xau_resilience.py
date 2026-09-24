"""Failure-injection tests for independent risk and durable paper state."""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from src.modules.xau import paper, paper_store, service
from src.platform.persistence.database import Base
from src.platform.persistence.models import XAUPaperPosition, XAUPaperSignal, XAUPaperTrade
from src.platform.runtime.config import Settings


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    sql = create_engine(f"sqlite:///{tmp_path / 'risk.db'}")
    Base.metadata.create_all(sql)
    factory = sessionmaker(bind=sql, expire_on_commit=False)
    monkeypatch.setattr(paper, 'open_xau_paper_session', factory)
    monkeypatch.setattr(paper, 'open_xau_replay_session', factory)
    monkeypatch.setattr(paper, 'replay_store_is_external', lambda: False)
    monkeypatch.setattr(paper_store, '_external_engine', None)
    engine = paper.XAUPaperTradingEngine(Settings(xau_paper_enabled=True, xau_paper_max_hold_minutes=120))
    now = datetime.now(timezone.utc)
    spot = {'price': 100., 'bid': 99.9, 'ask': 100.1, 'is_stale': False, 'fill_state': 'ready'}
    with factory() as db:
        account = engine._ensure_week(db, spot, now=now)
        db.commit()
        account_id = account.id
    yield SimpleNamespace(sql=sql, factory=factory, engine=engine, now=now, spot=spot, account_id=account_id)
    sql.dispose()


def seed_position(rt, side='long', age=5):
    with rt.factory() as db:
        db.add(XAUPaperPosition(account_id=rt.account_id, setup_key='risk-test', side=side,
            quantity_oz=1., entry_price=100., stop_loss=90. if side=='long' else 110.,
            target_price=120. if side=='long' else 80., current_price=100., risk_usd=10.,
            opened_at=paper._utc_naive(rt.now - timedelta(minutes=age))))
        db.add(XAUPaperSignal(account_id=rt.account_id, setup_key='risk-test', candidate='long_setup',
            observed_at=paper._utc_naive(rt.now), meta={}))
        db.commit()


@pytest.mark.parametrize('side,quote,reason,age', [
    ('long',89.,'stop_loss',5), ('short',111.,'stop_loss',5),
    ('long',121.,'target_price',5), ('short',79.,'target_price',5),
    ('long',100.,'time_stop',121), ('short',100.,'time_stop',121),
])
def test_hard_exit_commits_without_research_or_autopsy(runtime, monkeypatch, side, quote, reason, age):
    rt=runtime
    seed_position(rt, side, age)
    async def spot(**kwargs): return {**rt.spot, 'bid':quote, 'ask':quote+.01}
    async def forbidden(**kwargs): raise AssertionError('research must not precede hard exit')
    def fail(*args, **kwargs): raise TypeError('broken research')
    monkeypatch.setattr(paper,'get_indicative_spot',spot)
    monkeypatch.setattr(paper,'get_xau_snapshot',forbidden)
    monkeypatch.setattr(paper,'_trade_autopsy',fail)
    monkeypatch.setattr(rt.engine,'_memory_snapshot',fail)
    result=asyncio.run(rt.engine.scan(now=rt.now))
    assert result['closed_trade']['exit_reason']==reason
    assert result['execution_allowed'] is False
    with rt.factory() as db:
        assert db.query(XAUPaperPosition).one().status=='closed'
        assert db.query(XAUPaperTrade).count()==1
    # Retried protection cannot duplicate the closed trade.
    rt.engine._protect_serialized(rt.spot, rt.now)
    with rt.factory() as db: assert db.query(XAUPaperTrade).count()==1


@pytest.mark.parametrize('failure', ['memory','snapshot','macro'])
def test_research_failure_cannot_rollback_committed_stop_tightening(runtime, monkeypatch, failure):
    rt=runtime
    seed_position(rt)
    async def spot(**kwargs): return {**rt.spot, 'bid':111., 'ask':111.1}
    async def snapshot(**kwargs):
        if failure=='snapshot': raise RuntimeError('provider down')
        return {'indicative_spot': await spot(), 'candidate':'none', 'blocked':True}
    async def macro(**kwargs):
        if failure=='macro': raise RuntimeError('macro down')
        return {}
    def memory(*args): raise TypeError('bad historical record')
    monkeypatch.setattr(paper,'get_indicative_spot',spot)
    monkeypatch.setattr(paper,'get_xau_snapshot',snapshot)
    monkeypatch.setattr(paper,'get_macro_context',macro)
    monkeypatch.setattr(rt.engine,'_memory_snapshot',memory)
    result=asyncio.run(rt.engine.scan(now=rt.now))
    assert result['status']=='degraded'
    assert result['paper_entry_allowed'] is False
    with rt.factory() as db:
        assert db.query(XAUPaperPosition).one().stop_loss==100.
        assert db.query(XAUPaperTrade).count()==0


def test_stale_quote_never_manufactures_protective_fill(runtime):
    seed_position(runtime)
    result=runtime.engine._protect_serialized({**runtime.spot,'bid':50.,'is_stale':True}, runtime.now)
    assert result['closed_trade'] is None
    with runtime.factory() as db: assert db.query(XAUPaperPosition).one().status=='open'


@pytest.mark.parametrize('change', [{}, {'calendar_ok':False}, {'search_ok':False}, {'synthesis_ok':False},
    {'cache_stale':True}, {'refresh_pending':True}, {'observed_at': '2000-01-01T00:00:00Z'}])
def test_unknown_macro_blocks_entry(change):
    ready={'calendar_ok':True,'search_ok':True,'synthesis_ok':True,'observed_at':datetime.now(timezone.utc).isoformat(),'bias':0}
    macro={**ready,**change} if change else {}
    result=service.build_decision_fusion({'candidate':'long_setup','blocked':False},macro)
    assert result['state']=='macro_unavailable'
    assert result['paper_entry_allowed'] is False


def test_verified_neutral_macro_is_distinct_from_missing():
    macro={'calendar_ok':True,'search_ok':True,'synthesis_ok':True,'observed_at':datetime.now(timezone.utc).isoformat(),'bias':0}
    result=service.build_decision_fusion({'candidate':'long_setup','blocked':False},macro)
    assert result['state']=='setup_macro_neutral'
    assert result['macro_ready'] is True


def test_memory_cache_survives_session_close_and_invalidates(runtime):
    statements=[]
    event.listen(runtime.sql,'before_cursor_execute',lambda *args: statements.append(args[2]))
    with runtime.factory() as db: first=runtime.engine._memory_history(db,'long_setup')
    before=len(statements)
    with runtime.factory() as db: second=runtime.engine._memory_history(db,'long_setup')
    assert first is second
    assert len(statements)==before
    runtime.engine._invalidate_memory()
    with runtime.factory() as db: runtime.engine._memory_history(db,'long_setup')
    assert len(statements)>before


def test_reversal_state_survives_engine_recreation_but_expires(runtime):
    rt=runtime
    seed_position(rt)
    management={'exit_requested':True,'exit_reason':'thesis_reversal','action':'exit'}
    with rt.factory() as db:
        pos=db.query(XAUPaperPosition).one()
        first=rt.engine._confirm_persisted_reversal(db,pos,management,'tick-1',rt.now)
        assert first['exit_requested'] is False
        db.commit()
    engine=paper.XAUPaperTradingEngine(rt.engine.settings)
    with rt.factory() as db:
        pos=db.query(XAUPaperPosition).one()
        duplicate=engine._confirm_persisted_reversal(db,pos,management,'tick-1',rt.now+timedelta(seconds=15))
        assert duplicate['exit_requested'] is False
        second=engine._confirm_persisted_reversal(db,pos,management,'tick-2',rt.now+timedelta(seconds=30))
        assert second['exit_requested'] is True
        db.commit()
    with rt.factory() as db:
        expired=engine._confirm_persisted_reversal(db,db.query(XAUPaperPosition).one(),management,'tick-3',rt.now+timedelta(minutes=5))
        assert expired['exit_requested'] is False


@pytest.mark.parametrize('acquired',[False,True])
def test_postgres_writer_lock_uses_transaction_scope_and_rolls_back_on_failure(monkeypatch, acquired):
    calls=[]
    tx_calls=[]
    class Transaction:
        def commit(self): tx_calls.append('commit')
        def rollback(self): tx_calls.append('rollback')
    class Connection:
        def begin(self): return Transaction()
        def close(self): calls.append('close')
        def execute(self,stmt):
            calls.append(str(stmt))
            return SimpleNamespace(scalar=lambda:acquired)
    engine=SimpleNamespace(dialect=SimpleNamespace(name='postgresql'),connect=lambda:Connection())
    monkeypatch.setattr(paper_store,'_external_engine',engine)
    with pytest.raises(RuntimeError):
        with paper_store.paper_writer_guard() as allowed:
            assert allowed is acquired
            raise RuntimeError('worker failed')
    assert any('pg_try_advisory_xact_lock' in call for call in calls)
    assert not any('pg_advisory_unlock' in call for call in calls)
    assert tx_calls == ['rollback']
    assert calls[-1] == 'close'


def test_deferred_autopsy_runs_after_protective_commit(runtime):
    rt=runtime
    seed_position(rt)
    result=rt.engine._protect_serialized({**rt.spot,'bid':89.}, rt.now)
    assert result['closed_trade']
    with rt.factory() as db:
        trade=db.query(XAUPaperTrade).one()
        assert trade.meta['autopsy_status']=='deferred_protective_exit'
        rt.engine._backfill_protective_autopsies(db)
        db.commit()
    with rt.factory() as db:
        trade=db.query(XAUPaperTrade).one()
        assert trade.meta['autopsy_status']=='completed'
        assert trade.meta['autopsy']


@pytest.mark.parametrize('quote',[float('nan'),float('inf'),-1.,0.])
def test_invalid_quotes_cannot_trigger_protection(runtime, quote):
    seed_position(runtime)
    result=runtime.engine._protect_serialized({**runtime.spot,'bid':quote}, runtime.now)
    assert result['closed_trade'] is None


def test_missing_macro_blocks_engine_entry_and_can_be_revalidated(runtime, monkeypatch):
    rt=runtime
    technical={'candidate':'long_setup','blocked':False,'indicative_spot':rt.spot,
               'atr_reference':10.,'swing_low_reference':90.}
    async def spot(**kwargs): return rt.spot
    async def snapshot(**kwargs): return technical
    async def macro(**kwargs): return service._neutral_macro_context()
    monkeypatch.setattr(paper,'get_indicative_spot',spot)
    monkeypatch.setattr(paper,'get_xau_snapshot',snapshot)
    monkeypatch.setattr(paper,'get_macro_context',macro)
    result=asyncio.run(rt.engine.scan(now=rt.now))
    assert result['status']=='ok'
    assert result['opened'] is False
    assert result['fusion']['state']=='macro_unavailable'
    assert paper._can_revalidate_signal(False,'macro_unavailable',True)
    with rt.factory() as db: assert db.query(XAUPaperPosition).count()==0


def test_paper_writer_transaction_lock_uses_callers_transaction(monkeypatch):
    calls = []

    class Bind:
        dialect = SimpleNamespace(name="postgresql")

    class Session:
        def get_bind(self):
            return Bind()
        def execute(self, stmt):
            calls.append(str(stmt))
            return SimpleNamespace(scalar=lambda: True)

    acquired, lock_ms = paper_store.acquire_paper_writer_transaction(
        Session(),
        timeout_seconds=0.0,
    )
    assert acquired is True
    assert lock_ms >= 0.0
    assert len(calls) == 1
    assert "pg_try_advisory_xact_lock" in calls[0]


def test_paper_writer_transaction_lock_fails_fast_without_leaking(monkeypatch):
    calls = []

    class Bind:
        dialect = SimpleNamespace(name="postgresql")

    class Session:
        def get_bind(self):
            return Bind()
        def execute(self, stmt):
            calls.append(str(stmt))
            return SimpleNamespace(scalar=lambda: False)

    acquired, lock_ms = paper_store.acquire_paper_writer_transaction(
        Session(),
        timeout_seconds=0.0,
    )
    assert acquired is False
    assert lock_ms >= 0.0
    assert len(calls) == 1
    assert "pg_try_advisory_xact_lock" in calls[0]
