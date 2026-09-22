"""Durable storage boundary for PanWatch research intelligence.

Research state is self-contained (sources, evidence, claims, falsification and
belief history). Prefer PANWATCH_RESEARCH_DATABASE_URL. For the current XAU
deployment, XAU_PAPER_DATABASE_URL is accepted as a compatibility fallback so
research memory can use the already configured durable PostgreSQL service.
"""

from __future__ import annotations

import logging
import os

from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.platform.persistence.database import Base, SessionLocal
from src.platform.persistence.models import (
    ResearchBeliefCycleRecord,
    ResearchBeliefEventRecord,
    ResearchBeliefSnapshotRecord,
    ResearchClaimEdgeRecord,
    ResearchClaimEvidenceLinkRecord,
    ResearchClaimRecord,
    ResearchEvidenceRecord,
    ResearchFalsificationRuleRecord,
    ResearchSourceRecord,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

_external_engine = None
ResearchSessionLocal = SessionLocal
_external_available = False


def _sqlalchemy_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def _research_tables():
    return [
        ResearchSourceRecord.__table__,
        ResearchEvidenceRecord.__table__,
        ResearchClaimRecord.__table__,
        ResearchClaimEdgeRecord.__table__,
        ResearchClaimEvidenceLinkRecord.__table__,
        ResearchFalsificationRuleRecord.__table__,
        ResearchBeliefCycleRecord.__table__,
        ResearchBeliefSnapshotRecord.__table__,
        ResearchBeliefEventRecord.__table__,
    ]


def init_research_store(settings: Settings | None = None) -> bool:
    """Initialize durable research storage.

    Returns True only when an external PostgreSQL research store is active.
    Failure to provision never blocks PanWatch startup; the local SQLite store
    remains available but is explicitly logged as non-durable across redeploys.
    """

    global _external_engine, ResearchSessionLocal, _external_available

    settings = settings or Settings()
    raw_url = (
        os.environ.get("PANWATCH_RESEARCH_DATABASE_URL", "").strip()
        or (settings.xau_paper_database_url or "").strip()
    )
    if not raw_url:
        ResearchSessionLocal = SessionLocal
        _external_available = False
        logger.warning(
            "Research store uses local SQLite; configure "
            "PANWATCH_RESEARCH_DATABASE_URL for redeploy-durable beliefs"
        )
        return False

    if _external_engine is not None and _external_available:
        return True

    try:
        engine = create_engine(
            _sqlalchemy_url(raw_url),
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=2,
            max_overflow=2,
        )
        tables = _research_tables()
        Base.metadata.create_all(bind=engine, tables=tables)

        inspector = inspect(engine)
        missing = [
            table.name
            for table in tables
            if not inspector.has_table(table.name)
        ]
        if missing:
            raise RuntimeError(
                "research tables not provisioned: " + ",".join(missing)
            )

        _external_engine = engine
        ResearchSessionLocal = sessionmaker(
            bind=engine,
            expire_on_commit=False,
        )
        _external_available = True
        logger.info(
            "Research intelligence store initialized on external PostgreSQL"
        )
        return True
    except (SQLAlchemyError, RuntimeError) as exc:
        ResearchSessionLocal = SessionLocal
        _external_available = False
        logger.warning(
            "External research store unavailable (%s); using local SQLite "
            "fallback, which is not durable across Railway redeploys",
            type(exc).__name__,
        )
        return False


def research_store_is_external() -> bool:
    return bool(_external_available and _external_engine is not None)


def open_research_session():
    return ResearchSessionLocal()
