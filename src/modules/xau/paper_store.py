"""Durable storage boundary for the XAU weekly paper league.

Two storage policies are supported:

* external_preferred (legacy/default): PostgreSQL is the primary when healthy,
  with a local SQLite fallback.
* local_primary: the durable SQLite database on /app/data is the only primary.
  XAU_PAPER_DATABASE_URL becomes an optional one-way replica/archive.

Production PanWatch uses local_primary while a Railway volume is mounted at
/app/data. This deliberately avoids dual-primary failover and split-brain:
Neon can be unavailable, quota-limited, or recovering without changing the
trading engine's source of truth.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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
_replica_sync_lock = threading.Lock()
_primary_mode = "uninitialized"
_replica_sync_state: dict[str, object] = {
    "status": "never",
    "last_attempt_at": None,
    "last_success_at": None,
    "last_error_type": None,
    "last_report": None,
}

PAPER_WRITER_ADVISORY_LOCK_KEY = 782341906
XAUPaperSessionLocal = SessionLocal
XAUReplaySessionLocal = SessionLocal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _storage_mode(settings: Settings) -> str:
    value = str(settings.xau_paper_storage_mode or "").strip().lower()
    if value not in {"external_preferred", "local_primary"}:
        logger.error(
            "Invalid XAU_PAPER_STORAGE_MODE=%s; failing safe to local_primary",
            value,
        )
        return "local_primary"
    return value


def _sqlalchemy_url(url: str) -> str:
    value = (url or "").strip()
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


def _create_store_engine(raw_url: str):
    """Create a bounded-time store engine without leaking credentials to logs."""
    url = _sqlalchemy_url(raw_url)
    kwargs: dict[str, object] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }
    if url.startswith("postgresql"):
        kwargs.update(
            {
                "pool_size": 2,
                "max_overflow": 2,
                "connect_args": {"connect_timeout": 5},
            }
        )
    return create_engine(url, **kwargs)


def _dispose_external_engine() -> None:
    global _external_engine
    engine = _external_engine
    _external_engine = None
    if engine is not None:
        try:
            engine.dispose()
        except Exception:
            logger.debug("XAU external engine dispose failed", exc_info=True)


def _activate_local_primary(settings: Settings, *, reason: str) -> None:
    global _external_replay_available, _primary_mode
    global XAUPaperSessionLocal, XAUReplaySessionLocal

    _dispose_external_engine()
    XAUPaperSessionLocal = SessionLocal
    XAUReplaySessionLocal = SessionLocal
    _external_replay_available = False
    _primary_mode = "local_primary"

    if settings.xau_paper_data_persistent:
        logger.info(
            "XAU paper/replay primary=local_sqlite persistent=true reason=%s",
            reason,
        )
    else:
        logger.warning(
            "XAU paper/replay primary=local_sqlite persistent=false reason=%s; "
            "production must mount /app/data and set XAU_PAPER_DATA_PERSISTENT=true",
            reason,
        )


@contextmanager
def paper_writer_guard(*, timeout_seconds: float = 0.0):
    """Serialize critical paper writers across replicas without stale session locks."""
    timeout_seconds = max(0.0, float(timeout_seconds))
    if _external_engine is None or _external_engine.dialect.name != "postgresql":
        acquired = (
            _local_writer_lock.acquire(timeout=timeout_seconds)
            if timeout_seconds
            else _local_writer_lock.acquire(blocking=False)
        )
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
            acquired = bool(
                conn.execute(
                    text(
                        "SELECT pg_try_advisory_xact_lock("
                        f"{PAPER_WRITER_ADVISORY_LOCK_KEY}"
                        ")"
                    )
                ).scalar()
            )
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


def acquire_paper_writer_transaction(
    db,
    *,
    timeout_seconds: float = 0.0,
) -> tuple[bool, float]:
    """Acquire the PostgreSQL paper-writer xact lock when PostgreSQL is primary."""
    timeout_seconds = max(0.0, float(timeout_seconds))
    bind = db.get_bind()
    if bind.dialect.name != "postgresql":
        return True, 0.0

    started = time.monotonic()
    deadline = started + timeout_seconds
    while True:
        acquired = bool(
            db.execute(
                text(
                    "SELECT pg_try_advisory_xact_lock("
                    f"{PAPER_WRITER_ADVISORY_LOCK_KEY}"
                    ")"
                )
            ).scalar()
        )
        if acquired or time.monotonic() >= deadline:
            return acquired, round((time.monotonic() - started) * 1000.0, 2)
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


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
            "replay will use paper-signal compatibility storage",
            type(exc).__name__,
        )
        return False


def init_xau_paper_store(settings: Settings | None = None) -> bool:
    """Initialize the active XAU store without coupling API startup to Neon."""
    global _external_engine, _external_replay_available, _replay_provisioning_error
    global _primary_mode, XAUPaperSessionLocal, XAUReplaySessionLocal

    settings = settings or Settings()
    mode = _storage_mode(settings)
    raw_url = (settings.xau_paper_database_url or "").strip()

    if mode == "local_primary":
        _activate_local_primary(settings, reason="configured_single_primary")
        return False

    if not raw_url:
        _activate_local_primary(settings, reason="external_not_configured")
        _primary_mode = "local_fallback"
        return False

    if _external_engine is not None:
        _primary_mode = "external"
        return True

    candidate_engine = None
    try:
        candidate_engine = _create_store_engine(raw_url)
        paper_tables = [
            XAUPaperAccount.__table__,
            XAUPaperSignal.__table__,
            XAUPaperPosition.__table__,
            XAUPaperTrade.__table__,
        ]
        Base.metadata.create_all(bind=candidate_engine, tables=paper_tables)

        paper_session_local = sessionmaker(
            bind=candidate_engine,
            expire_on_commit=False,
        )
        replay_available = _provision_replay_table(candidate_engine)
        replay_session_local = (
            sessionmaker(bind=candidate_engine, expire_on_commit=False)
            if replay_available
            else SessionLocal
        )
    except SQLAlchemyError as exc:
        _replay_provisioning_error = f"{type(exc).__name__}: {exc}"
        if candidate_engine is not None:
            try:
                candidate_engine.dispose()
            except Exception:
                pass
        _activate_local_primary(
            settings,
            reason=f"external_{type(exc).__name__}",
        )
        _primary_mode = "local_fallback"
        logger.error(
            "XAU external primary unavailable (%s); local fallback remains active",
            type(exc).__name__,
        )
        return False

    _external_engine = candidate_engine
    XAUPaperSessionLocal = paper_session_local
    _external_replay_available = replay_available
    XAUReplaySessionLocal = replay_session_local
    _primary_mode = "external"

    if _external_replay_available:
        logger.info("XAU replay store initialized on external PostgreSQL")
    else:
        logger.warning(
            "XAU replay dedicated table unavailable; "
            "storage_mode=external_paper_signal_compat"
        )
    logger.info("XAU paper primary initialized on external PostgreSQL")
    return True


def paper_store_is_external() -> bool:
    return _external_engine is not None and _primary_mode == "external"


def replay_store_is_external() -> bool:
    return bool(paper_store_is_external() and _external_replay_available)


def open_xau_paper_session():
    return XAUPaperSessionLocal()


def open_xau_replay_session():
    return XAUReplaySessionLocal()


def _row_payload(row, *, exclude: set[str]) -> dict[str, object]:
    return {
        column.name: deepcopy(getattr(row, column.name))
        for column in row.__table__.columns
        if column.name not in exclude
    }


def _apply_payload(target, payload: dict[str, object]) -> None:
    for key, value in payload.items():
        setattr(target, key, deepcopy(value))


def _sync_accounts(source, target) -> tuple[dict[int, int], int, int]:
    mapping: dict[int, int] = {}
    inserted = 0
    updated = 0
    rows = source.query(XAUPaperAccount).order_by(XAUPaperAccount.id.asc()).all()
    for row in rows:
        dest = (
            target.query(XAUPaperAccount)
            .filter(XAUPaperAccount.week_key == row.week_key)
            .first()
        )
        payload = _row_payload(row, exclude={"id"})
        if dest is None:
            dest = XAUPaperAccount(**payload)
            target.add(dest)
            target.flush()
            inserted += 1
        else:
            _apply_payload(dest, payload)
            target.flush()
            updated += 1
        mapping[int(row.id)] = int(dest.id)
    return mapping, inserted, updated


def _sync_keyed_rows(
    source,
    target,
    model,
    *,
    key_name: str,
    account_mapping: dict[int, int] | None = None,
) -> tuple[int, int, int]:
    inserted = 0
    updated = 0
    conflicts = 0
    key_column = getattr(model, key_name)

    for row in source.query(model).order_by(model.id.asc()).all():
        key = getattr(row, key_name)
        matches = (
            target.query(model)
            .filter(key_column == key)
            .order_by(model.id.asc())
            .all()
        )
        dest = matches[0] if matches else None
        if len(matches) > 1:
            conflicts += len(matches) - 1

        payload = _row_payload(row, exclude={"id"})
        if "account_id" in payload and account_mapping is not None:
            source_account_id = int(row.account_id)
            mapped = account_mapping.get(source_account_id)
            if mapped is None:
                raise RuntimeError(
                    f"missing account mapping for {model.__name__}:{key}"
                )
            payload["account_id"] = mapped

        if dest is None:
            dest = model(**payload)
            target.add(dest)
            inserted += 1
        else:
            _apply_payload(dest, payload)
            updated += 1
    target.flush()
    return inserted, updated, conflicts


def sync_xau_paper_replica(settings: Settings | None = None) -> dict[str, object]:
    """Idempotently copy the local primary to the optional external archive.

    The sync is strictly one-way. It never promotes the replica, never deletes
    target rows, and never changes the active Session factory.
    """
    global _replica_sync_state

    settings = settings or Settings()
    mode = _storage_mode(settings)
    raw_url = (settings.xau_paper_database_url or "").strip()

    if mode != "local_primary":
        return {"status": "skipped", "reason": "primary_not_local"}
    if not settings.xau_paper_replica_sync_enabled:
        return {"status": "skipped", "reason": "replica_sync_disabled"}
    if not raw_url:
        return {"status": "skipped", "reason": "replica_not_configured"}
    if not _replica_sync_lock.acquire(blocking=False):
        return {"status": "busy", "reason": "replica_sync_in_progress"}

    attempted_at = _now_iso()
    _replica_sync_state = {
        **_replica_sync_state,
        "status": "syncing",
        "last_attempt_at": attempted_at,
        "last_error_type": None,
    }

    source = None
    target = None
    replica_engine = None
    try:
        replica_engine = _create_store_engine(raw_url)
        paper_tables = [
            XAUPaperAccount.__table__,
            XAUPaperSignal.__table__,
            XAUPaperPosition.__table__,
            XAUPaperTrade.__table__,
        ]
        Base.metadata.create_all(bind=replica_engine, tables=paper_tables)
        replay_available = _provision_replay_table(replica_engine)

        source = SessionLocal()
        TargetSession = sessionmaker(bind=replica_engine, expire_on_commit=False)
        target = TargetSession()

        account_mapping, accounts_inserted, accounts_updated = _sync_accounts(
            source,
            target,
        )
        signal_stats = _sync_keyed_rows(
            source,
            target,
            XAUPaperSignal,
            key_name="setup_key",
            account_mapping=account_mapping,
        )
        position_stats = _sync_keyed_rows(
            source,
            target,
            XAUPaperPosition,
            key_name="setup_key",
            account_mapping=account_mapping,
        )
        trade_stats = _sync_keyed_rows(
            source,
            target,
            XAUPaperTrade,
            key_name="setup_key",
            account_mapping=account_mapping,
        )
        replay_stats = (0, 0, 0)
        if replay_available:
            replay_stats = _sync_keyed_rows(
                source,
                target,
                XAUReplayEpisode,
                key_name="replay_key",
            )

        target.commit()
        total_conflicts = (
            signal_stats[2]
            + position_stats[2]
            + trade_stats[2]
            + replay_stats[2]
        )
        status = "success" if total_conflicts == 0 else "degraded"
        report = {
            "status": status,
            "primary": "local_sqlite",
            "replica": "external_database",
            "accounts": {
                "inserted": accounts_inserted,
                "updated": accounts_updated,
            },
            "signals": {
                "inserted": signal_stats[0],
                "updated": signal_stats[1],
                "conflicts": signal_stats[2],
            },
            "positions": {
                "inserted": position_stats[0],
                "updated": position_stats[1],
                "conflicts": position_stats[2],
            },
            "trades": {
                "inserted": trade_stats[0],
                "updated": trade_stats[1],
                "conflicts": trade_stats[2],
            },
            "replay": {
                "dedicated_table_available": replay_available,
                "inserted": replay_stats[0],
                "updated": replay_stats[1],
                "conflicts": replay_stats[2],
            },
            "target_conflicts": total_conflicts,
        }
        _replica_sync_state = {
            "status": status,
            "last_attempt_at": attempted_at,
            "last_success_at": _now_iso(),
            "last_error_type": None,
            "last_report": report,
        }
        if status == "success":
            logger.info(
                "XAU replica sync success accounts=%s signals=%s positions=%s trades=%s replay=%s",
                report["accounts"],
                report["signals"],
                report["positions"],
                report["trades"],
                report["replay"],
            )
        else:
            logger.warning(
                "XAU replica sync completed with target conflicts=%s; "
                "no target rows were deleted",
                total_conflicts,
            )
        return report
    except Exception as exc:
        if target is not None:
            try:
                target.rollback()
            except Exception:
                pass
        error_type = type(exc).__name__
        report = {
            "status": "degraded",
            "primary": "local_sqlite",
            "replica": "external_database",
            "error_type": error_type,
        }
        _replica_sync_state = {
            **_replica_sync_state,
            "status": "degraded",
            "last_attempt_at": attempted_at,
            "last_error_type": error_type,
            "last_report": report,
        }
        logger.warning(
            "XAU replica sync unavailable type=%s; local primary remains authoritative",
            error_type,
        )
        return report
    finally:
        if source is not None:
            source.close()
        if target is not None:
            target.close()
        if replica_engine is not None:
            try:
                replica_engine.dispose()
            except Exception:
                pass
        _replica_sync_lock.release()


def paper_storage_health(settings: Settings | None = None) -> dict[str, object]:
    settings = settings or Settings()
    mode = _storage_mode(settings)
    if mode == "local_primary":
        primary = "local_sqlite"
        persistent = bool(settings.xau_paper_data_persistent)
    elif paper_store_is_external():
        primary = "external_postgres"
        persistent = True
    else:
        primary = "local_sqlite_fallback"
        persistent = bool(settings.xau_paper_data_persistent)

    return {
        "configured_mode": mode,
        "primary_mode": _primary_mode,
        "primary_storage": primary,
        "primary_persistent": persistent,
        "external_replica_configured": bool(
            mode == "local_primary" and settings.xau_paper_database_url
        ),
        "external_replica_sync_enabled": bool(
            mode == "local_primary" and settings.xau_paper_replica_sync_enabled
        ),
        "replica_sync": deepcopy(_replica_sync_state),
    }


def replay_storage_health() -> dict:
    """Describe effective replay persistence without overstating availability."""
    settings = Settings()
    storage = paper_storage_health(settings)
    if _storage_mode(settings) == "local_primary":
        return {
            "replay_storage_mode": "local_sqlite_primary",
            "replay_storage_persistent": bool(settings.xau_paper_data_persistent),
            "replay_dedicated_table_available": False,
            "replay_provisioning_error": _replay_provisioning_error,
            "replica_sync": storage["replica_sync"],
        }
    if _external_engine is None:
        return {
            "replay_storage_mode": "local_sqlite",
            "replay_storage_persistent": bool(settings.xau_paper_data_persistent),
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
