"""Durable storage boundary for Ahmed Research Engine + PanWatch beliefs.

Production can point PANWATCH_RESEARCH_DATABASE_URL at an external PostgreSQL
database. When unset or unavailable, research persistence falls back to the
legacy local SQLite store without blocking PanWatch startup.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.platform.persistence.database import Base, SessionLocal
from src.platform.persistence.models import (
    ResearchBeliefCycleRecord,
    ResearchBeliefEventRecord,
    ResearchBeliefSnapshotRecord,
    ResearchAcquisitionRunRecord,
    ResearchClaimCandidateRecord,
    ResearchClaimEdgeRecord,
    ResearchClaimEvidenceLinkRecord,
    ResearchClaimRecord,
    ResearchEvidenceRecord,
    ResearchFalsificationRuleRecord,
    ResearchLoopRunRecord,
    ResearchProbeAttemptRecord,
    ResearchSourceRecord,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

_external_engine = None
ResearchSessionLocal = SessionLocal

_RESEARCH_TABLES = [
    ResearchSourceRecord.__table__,
    ResearchEvidenceRecord.__table__,
    ResearchClaimRecord.__table__,
    ResearchClaimEdgeRecord.__table__,
    ResearchClaimEvidenceLinkRecord.__table__,
    ResearchFalsificationRuleRecord.__table__,
    ResearchBeliefCycleRecord.__table__,
    ResearchBeliefSnapshotRecord.__table__,
    ResearchBeliefEventRecord.__table__,
    ResearchLoopRunRecord.__table__,
    ResearchProbeAttemptRecord.__table__,
    ResearchAcquisitionRunRecord.__table__,
    ResearchClaimCandidateRecord.__table__,
]


def _sqlalchemy_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def _build_engine(url: str):
    normalized = _sqlalchemy_url(url)
    kwargs = {"pool_pre_ping": True}
    if normalized.startswith(("postgresql", "postgres")):
        kwargs.update(
            {
                "pool_recycle": 300,
                "pool_size": 3,
                "max_overflow": 3,
            }
        )
    return create_engine(normalized, **kwargs)


def init_research_store(settings: Settings | None = None) -> bool:
    """Initialize the research store.

    Returns True only when the configured external store is healthy and all
    research tables are available. Any external provisioning/connectivity
    failure degrades to local SQLite and is logged explicitly.
    """

    global _external_engine, ResearchSessionLocal

    settings = settings or Settings()
    raw_url = (settings.research_database_url or "").strip()
    if not raw_url:
        _external_engine = None
        ResearchSessionLocal = SessionLocal
        logger.warning(
            "Research store is using local SQLite fallback; belief/evidence "
            "history is not redeploy-durable unless /app/data is persisted"
        )
        return False

    try:
        engine = _build_engine(raw_url)
        Base.metadata.create_all(bind=engine, tables=_RESEARCH_TABLES)
        inspector = inspect(engine)
        missing = [
            table.name
            for table in _RESEARCH_TABLES
            if not inspector.has_table(table.name)
        ]
        if missing:
            raise RuntimeError(
                "research store provisioning incomplete: "
                + ", ".join(sorted(missing))
            )
        _external_engine = engine
        ResearchSessionLocal = sessionmaker(
            bind=engine,
            expire_on_commit=False,
        )
        logger.info(
            "Research store initialized external=True dialect=%s tables=%s",
            engine.dialect.name,
            len(_RESEARCH_TABLES),
        )
        return True
    except (SQLAlchemyError, RuntimeError) as exc:
        logger.error(
            "Research store external initialization failed (%s); "
            "falling back to local SQLite",
            type(exc).__name__,
        )
        try:
            if "engine" in locals():
                engine.dispose()
        except Exception:
            pass
        _external_engine = None
        ResearchSessionLocal = SessionLocal
        return False


def research_store_is_external() -> bool:
    return _external_engine is not None


def research_store_dialect() -> str:
    if _external_engine is None:
        return "sqlite"
    return str(_external_engine.dialect.name)


def open_research_session():
    return ResearchSessionLocal()


@contextmanager
def research_session():
    db = open_research_session()
    try:
        yield db
    finally:
        db.close()
