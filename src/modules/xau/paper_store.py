"""Durable storage boundary for the XAU weekly paper league.

When XAU_PAPER_DATABASE_URL is configured, only the XAU paper tables are stored
in the external PostgreSQL database. The rest of PanWatch can continue using
its existing SQLite database.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
import threading
import time

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
_replay_provisioning_error = None
_local_writer_lock = threading.Lock()
XAUPaperSessionLocal = SessionLocal
XAUReplaySessionLocal = SessionLocal


@contextmanager
def paper_writer_guard(*, timeout_seconds: float = 0.0):
    """Serialize critical paper writers across replicas without stale session locks.

    PostgreSQL uses transaction-scoped advisory locks, so a killed/cancelled
    request cannot strand a session-level lock in the connection pool. A short
    bounded wait absorbs normal commit overlap while keeping hard protection
    responsive. SQLite/local deployments use a process lock.
    """
    timeout_seconds = max(0.0, float(timeout_seconds))
    if _external_engine is None or _external_engine.dialect.name != "postgresql":
        acquired = _local_writer_lock.acquire(timeout=timeout_seconds) if timeout_seconds else _local_writer_lock.acquire(blocking=False)
        try:
            yield acquired
        finally:
            if acquired:
                _local_writer_lock.release()
        return

    deadline = time.monotonic() + timeout_seconds
    conn = _external_engine.connect()
    tx = conn.begin()
    acquired = False
    try:
        while True:
            acquired = bool(conn.execute(text("SELECT pg_try_advisory_xact_lock(782341905)")).scalar())
            if acquired or time.monotonic() >= deadline:
                break
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
        yield acquired
        if acquired:
            tx.commit()
        else:
            tx.rollback()
    except Exception:
        tx.rollback()
        raise
    finally:
        conn.close()


def acquire_paper_writer_transaction(db, *, timeout_seconds: float = 0.0) -> tuple[bool, float]:
    """Acquire the paper-writer advisory lock inside the caller's DB transaction.

    The PostgreSQL xact lock is released automatically by commit/rollback. This
    lets callers keep the lock around only critical account/position mutations
    instead of wrapping research calculations or research-only metadata writes.
    """
    timeout_seconds = max(0.0, float(timeout_seconds))
    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        return True, 0.0

    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        acquired = bool(
            db.execute(text("SELECT pg_try_advisory_xact_lock(782341905)")).scalar()
        )
        if acquired or time.monotonic() >= deadline:
            return acquired, round((time.monotonic() - started) * 1000.0, 2)
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def _sqlalchemy_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def _provision_replay_table(engine) -> bool:
    """Create the replay table non-destructively when DDL is permitted."""
    global _replay_provisioning_error
    try:
        _replay_provisioning_error = None
        if not inspect(engine).has_table("xau_replay_episodes"):
            Base.metadata.create_all(
                bind=engine,
                tables=[XAUReplayEpisode.__table__],
            )
        return inspect(engine).has_table("xau_replay_episodes")
    except SQLAlchemyError as exc:
        _replay_provisioning_error = f"{type(exc).__name__}: {exc}"
        logger.warning(
            "XAU replay dedicated-table provisioning unavailable (%s); "
            "replay will use the durable external paper-signal compatibility store",
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
            # Dedicated replay-table DDL is optional. Replay persistence falls
            # back to XAUPaperSignal in the same external PostgreSQL store.
            XAUReplaySessionLocal = SessionLocal
            logger.warning(
                "XAU replay dedicated table is unavailable; "
                "storage_mode=external_paper_signal_compat durable_external_store=true"
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


def replay_storage_health() -> dict:
    """Describe the effective replay persistence mode without overstating fallback."""
    if _external_engine is None:
        return {
            "replay_storage_mode": "local_sqlite",
            "replay_storage_persistent": False,
            "replay_dedicated_table_available": False,
            "replay_provisioning_error": _replay_provisioning_error,
        }
    if _external_replay_available:
        return {
            "replay_storage_mode": "external_replay_table",
            "replay_storage_persistent": True,
            "replay_dedicated_table_available": True,
            "replay_provisioning_error": None,
        }
    return {
        "replay_storage_mode": "external_paper_signal_compat",
        "replay_storage_persistent": True,
        "replay_dedicated_table_available": False,
        "replay_provisioning_error": _replay_provisioning_error,
    }
