"""Durable storage boundary for the XAU weekly paper league.

When XAU_PAPER_DATABASE_URL is configured, only the XAU paper tables are stored
in the external PostgreSQL database. The rest of PanWatch can continue using
its existing SQLite database.
"""

from __future__ import annotations

import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.platform.persistence.database import Base, SessionLocal
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperPosition,
    XAUPaperSignal,
    XAUPaperTrade,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

_external_engine = None
XAUPaperSessionLocal = SessionLocal


def _sqlalchemy_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def init_xau_paper_store(settings: Settings | None = None) -> bool:
    """Initialize the paper store. Returns True when external PostgreSQL is used."""

    global _external_engine, XAUPaperSessionLocal

    settings = settings or Settings()
    raw_url = (settings.xau_paper_database_url or "").strip()
    if not raw_url:
        XAUPaperSessionLocal = SessionLocal
        logger.warning(
            "XAU paper store is using local SQLite fallback; history will only be "
            "durable when the host persists /app/data"
        )
        return False

    if _external_engine is None:
        _external_engine = create_engine(
            _sqlalchemy_url(raw_url),
            pool_pre_ping=True,
            pool_recycle=300,
            pool_size=2,
            max_overflow=2,
        )
        tables = [
            XAUPaperAccount.__table__,
            XAUPaperSignal.__table__,
            XAUPaperPosition.__table__,
            XAUPaperTrade.__table__,
        ]
        Base.metadata.create_all(bind=_external_engine, tables=tables)
        XAUPaperSessionLocal = sessionmaker(bind=_external_engine, expire_on_commit=False)
        logger.info("XAU paper store initialized on external PostgreSQL")

    return True


def paper_store_is_external() -> bool:
    return _external_engine is not None
