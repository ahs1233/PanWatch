"""Durable storage boundary for the XAU weekly paper league.

When XAU_PAPER_DATABASE_URL is configured, only the XAU paper tables are stored
in the external PostgreSQL database. The rest of PanWatch can continue using
its existing SQLite database.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from src.platform.persistence.database import Base, SessionLocal
from src.platform.persistence.models import (
    XAUPaperAccount,
    XAUPaperPosition,
    XAUPaperSignal,
    XAUPaperTrade,
    XAUReplayEpisode,
)
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)

_external_engine = None
_external_replay_available = False
XAUPaperSessionLocal = SessionLocal
XAUReplaySessionLocal = SessionLocal


@contextmanager
def paper_writer_guard():
    """One paper writer across PostgreSQL replicas, held across commits.

    The dedicated autocommit connection owns the advisory lock; it is never
    returned to the pool while locked. SQLite deployments use the process lock.
    """
    if _external_engine is None or _external_engine.dialect.name != "postgresql":
        yield True
        return
    with _external_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        acquired = bool(conn.execute(text("SELECT pg_try_advisory_lock(782341905)")).scalar())
        try:
            yield acquired
        finally:
            if acquired:
                try:
                    conn.execute(text("SELECT pg_advisory_unlock(782341905)"))
                except Exception:
                    conn.invalidate()
                    raise


def _sqlalchemy_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def _provision_replay_table(engine) -> bool:
    """Create the replay table non-destructively when DDL is permitted."""
    try:
        if not inspect(engine).has_table("xau_replay_episodes"):
            Base.metadata.create_all(
                bind=engine,
                tables=[XAUReplayEpisode.__table__],
            )
        return inspect(engine).has_table("xau_replay_episodes")
    except SQLAlchemyError as exc:
        logger.warning(
            "XAU replay PostgreSQL provisioning unavailable (%s); "
            "falling back to local SQLite",
            type(exc).__name__,
        )
        return False


def init_xau_paper_store(settings: Settings | None = None) -> bool:
    """Initialize the paper store. Returns True when external PostgreSQL is used."""

    global _external_engine, _external_replay_available, XAUPaperSessionLocal, XAUReplaySessionLocal

    settings = settings or Settings()
    raw_url = (settings.xau_paper_database_url or "").strip()
    if not raw_url:
        XAUPaperSessionLocal = SessionLocal
        XAUReplaySessionLocal = SessionLocal
        _external_replay_available = False
        logger.warning(
            "XAU paper/replay store is using local SQLite fallback; history will only be "
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
        # Existing paper tables are mandatory and already provisioned in production.
        # Replay is optional research memory; never make application startup depend
        # on DDL privileges for a newly introduced table.
        paper_tables = [
            XAUPaperAccount.__table__,
            XAUPaperSignal.__table__,
            XAUPaperPosition.__table__,
            XAUPaperTrade.__table__,
        ]
        Base.metadata.create_all(bind=_external_engine, tables=paper_tables)
        XAUPaperSessionLocal = sessionmaker(bind=_external_engine, expire_on_commit=False)

        _external_replay_available = _provision_replay_table(_external_engine)
        if _external_replay_available:
            XAUReplaySessionLocal = sessionmaker(
                bind=_external_engine,
                expire_on_commit=False,
            )
            logger.info("XAU replay store initialized on external PostgreSQL")
        else:
            # Main SQLite is initialized independently by init_db() and includes
            # XAUReplayEpisode via Base.metadata. This fallback keeps research
            # replay available when the external role lacks DDL privileges.
            XAUReplaySessionLocal = SessionLocal
            logger.warning(
                "XAU replay table is not provisioned in external PostgreSQL; "
                "using local SQLite replay fallback"
            )

        logger.info("XAU paper store initialized on external PostgreSQL")

    return True


def paper_store_is_external() -> bool:
    return _external_engine is not None


def replay_store_is_external() -> bool:
    return bool(_external_engine is not None and _external_replay_available)


def open_xau_paper_session():
    return XAUPaperSessionLocal()


def open_xau_replay_session():
    return XAUReplaySessionLocal()
