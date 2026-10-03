from __future__ import annotations

from sqlalchemy.exc import OperationalError

from src.modules.xau import paper_store
from src.platform.persistence.database import SessionLocal
from src.platform.runtime.config import Settings


class _FailingEngine:
    def __init__(self) -> None:
        self.disposed = False

    def dispose(self) -> None:
        self.disposed = True


def test_external_paper_store_outage_falls_back_without_startup_failure(monkeypatch):
    engine = _FailingEngine()

    monkeypatch.setattr(
        paper_store,
        "create_engine",
        lambda *_args, **_kwargs: engine,
    )

    def fail_create_all(*_args, **_kwargs):
        raise OperationalError(
            "CREATE TABLE",
            {},
            RuntimeError("external PostgreSQL quota exceeded"),
        )

    monkeypatch.setattr(
        paper_store.Base.metadata,
        "create_all",
        fail_create_all,
    )

    paper_store._external_engine = None
    paper_store._external_replay_available = True
    paper_store._replay_provisioning_error = None
    paper_store.XAUPaperSessionLocal = object()
    paper_store.XAUReplaySessionLocal = object()

    settings = Settings(
        xau_paper_database_url="postgresql://user:pass@example.test/panwatch"
    )

    external = paper_store.init_xau_paper_store(settings)

    assert external is False
    assert engine.disposed is True
    assert paper_store._external_engine is None
    assert paper_store._external_replay_available is False
    assert paper_store.XAUPaperSessionLocal is SessionLocal
    assert paper_store.XAUReplaySessionLocal is SessionLocal
    assert "OperationalError" in str(paper_store._replay_provisioning_error)
