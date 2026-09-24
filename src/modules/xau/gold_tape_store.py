"""Small persistent executed-trade tape for free gold microstructure sensors."""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from src.platform.marketdata.gold_okx import OKXGoldTrade


@dataclass(frozen=True)
class StoredGoldTrade:
    venue: str
    trade_id: str
    timestamp: datetime
    price: float
    size_xau: float
    aggressor_side: str


class GoldTapeStore:
    def __init__(self, path: str | None = None) -> None:
        if path is None:
            data_dir = Path(os.environ.get("DATA_DIR") or "./data")
            data_dir.mkdir(parents=True, exist_ok=True)
            path = str(data_dir / "gold_market_tape.sqlite3")
        self.path = path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS gold_trades (
                    venue TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    price REAL NOT NULL,
                    size_xau REAL NOT NULL,
                    side TEXT NOT NULL,
                    PRIMARY KEY (venue, trade_id)
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_gold_trades_venue_ts ON gold_trades(venue, ts_ms)"
            )

    def ingest_okx(self, venue: str, trades: Iterable[OKXGoldTrade]) -> int:
        rows = [
            (
                venue,
                str(t.trade_id),
                int(t.timestamp.timestamp() * 1000),
                float(t.price),
                float(t.size_xau),
                str(t.aggressor_side),
            )
            for t in trades
            if t.size_xau > 0 and t.price > 0 and t.aggressor_side in {"buy", "sell"}
        ]
        if not rows:
            return 0
        before = 0
        with self._connect() as conn:
            before = conn.total_changes
            conn.executemany(
                "INSERT OR IGNORE INTO gold_trades(venue,trade_id,ts_ms,price,size_xau,side) VALUES(?,?,?,?,?,?)",
                rows,
            )
            inserted = conn.total_changes - before
        return int(inserted)

    def prune(self, *, keep_days: int = 8) -> int:
        cutoff = int((datetime.now(timezone.utc) - timedelta(days=max(1, keep_days))).timestamp() * 1000)
        with self._connect() as conn:
            before = conn.total_changes
            conn.execute("DELETE FROM gold_trades WHERE ts_ms < ?", (cutoff,))
            return int(conn.total_changes - before)

    def load(self, venue: str, *, minutes: int) -> list[StoredGoldTrade]:
        now = datetime.now(timezone.utc)
        cutoff = int((now - timedelta(minutes=max(1, minutes))).timestamp() * 1000)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT venue,trade_id,ts_ms,price,size_xau,side
                FROM gold_trades
                WHERE venue=? AND ts_ms>=?
                ORDER BY ts_ms ASC, trade_id ASC
                """,
                (venue, cutoff),
            ).fetchall()
        return [
            StoredGoldTrade(
                venue=str(row[0]),
                trade_id=str(row[1]),
                timestamp=datetime.fromtimestamp(int(row[2]) / 1000.0, tz=timezone.utc),
                price=float(row[3]),
                size_xau=float(row[4]),
                aggressor_side=str(row[5]),
            )
            for row in rows
        ]

    def coverage(self, venue: str) -> dict[str, object]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM gold_trades WHERE venue=?",
                (venue,),
            ).fetchone()
        count = int((row or [0])[0] or 0)
        oldest_ms = (row or [None, None])[1]
        newest_ms = (row or [None, None, None])[2]
        oldest = datetime.fromtimestamp(oldest_ms / 1000.0, tz=timezone.utc) if oldest_ms else None
        newest = datetime.fromtimestamp(newest_ms / 1000.0, tz=timezone.utc) if newest_ms else None
        seconds = (newest - oldest).total_seconds() if oldest and newest else 0.0
        return {
            "trade_count": count,
            "oldest_trade_at": oldest.isoformat() if oldest else None,
            "newest_trade_at": newest.isoformat() if newest else None,
            "coverage_seconds": round(max(0.0, seconds), 3),
        }
