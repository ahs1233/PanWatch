"""Runtime orchestration for multi-venue gold microstructure fusion."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from src.modules.xau.gold_market_fusion import analyze_okx_venue, build_gold_market_fusion
from src.modules.xau.gold_tape_store import GoldTapeStore
from src.modules.xau.xaut_runtime import get_xaut_order_flow
from src.platform.marketdata.gold_okx import OKXGoldPublicProvider

logger = logging.getLogger(__name__)

_CACHE_TTL_SECONDS = 4.0
_cache: tuple[float, dict[str, Any]] | None = None
_lock = asyncio.Lock()
_store: GoldTapeStore | None = None
_warmed: set[str] = set()


def _tape_store() -> GoldTapeStore:
    global _store
    if _store is None:
        _store = GoldTapeStore()
    return _store


async def _fetch_okx(provider: OKXGoldPublicProvider, *, force: bool) -> tuple[Any, list[Any]]:
    snapshot = await asyncio.to_thread(provider.fetch_snapshot, trade_limit=500, book_depth=400)
    store = _tape_store()
    await asyncio.to_thread(store.ingest_okx, snapshot.source, snapshot.trades)

    coverage = await asyncio.to_thread(store.coverage, snapshot.source)
    coverage_seconds = float(coverage.get("coverage_seconds") or 0.0)
    # Warm-start a bounded amount of public history once. Completeness is still
    # derived from actual coverage; this cannot silently create a daily/weekly claim.
    if snapshot.source not in _warmed and coverage_seconds < 3600.0:
        try:
            history = await asyncio.to_thread(
                provider.fetch_history_trades,
                max_pages=20 if force else 8,
                page_size=100,
            )
            await asyncio.to_thread(store.ingest_okx, snapshot.source, history)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OKX history warm-start failed venue=%s error=%s", snapshot.source, type(exc).__name__)
        finally:
            _warmed.add(snapshot.source)

    # Keep one week plus safety margin. Daily/weekly profiles only claim complete
    # coverage after enough actual stored tape exists.
    await asyncio.to_thread(store.prune, keep_days=8)
    trades = await asyncio.to_thread(store.load, snapshot.source, minutes=10080)
    return snapshot, trades


async def get_gold_market_fusion(
    *,
    xau_spot_price: float | None = None,
    force: bool = False,
) -> dict[str, Any]:
    global _cache
    now = time.monotonic()
    if not force and _cache and now - _cache[0] < _CACHE_TTL_SECONDS:
        return _cache[1]

    async with _lock:
        now = time.monotonic()
        if not force and _cache and now - _cache[0] < _CACHE_TTL_SECONDS:
            return _cache[1]

        swap_provider = OKXGoldPublicProvider.xau_swap()
        xaut_provider = OKXGoldPublicProvider.xaut_spot()
        swap_task = asyncio.create_task(_fetch_okx(swap_provider, force=force))
        xaut_task = asyncio.create_task(_fetch_okx(xaut_provider, force=force))
        bitfinex_task = asyncio.create_task(
            get_xaut_order_flow(xau_spot_price=xau_spot_price, force=force)
        )

        stage_errors: dict[str, str] = {}
        swap_analysis = None
        xaut_analysis = None
        bitfinex = None

        try:
            snap, trades = await swap_task
            swap_analysis = analyze_okx_venue(
                snap,
                trades=trades,
                xau_spot_price=xau_spot_price,
            )
        except Exception as exc:  # noqa: BLE001
            stage_errors["okx_xau_swap"] = type(exc).__name__

        try:
            snap, trades = await xaut_task
            xaut_analysis = analyze_okx_venue(
                snap,
                trades=trades,
                xau_spot_price=xau_spot_price,
            )
        except Exception as exc:  # noqa: BLE001
            stage_errors["okx_xaut_spot"] = type(exc).__name__

        try:
            bitfinex = await bitfinex_task
        except Exception as exc:  # noqa: BLE001
            stage_errors["bitfinex_xaut"] = type(exc).__name__

        result = build_gold_market_fusion(
            okx_xau=swap_analysis,
            okx_xaut=xaut_analysis,
            bitfinex_xaut=bitfinex,
        )
        result["stage_errors"] = stage_errors
        result["source_health"] = {
            "okx_xau_swap": {
                "available": bool(swap_analysis),
                "status": "ready" if swap_analysis else "unavailable",
                "error": stage_errors.get("okx_xau_swap"),
            },
            "okx_xaut_spot": {
                "available": bool(xaut_analysis),
                "status": "ready" if xaut_analysis else "unavailable",
                "error": stage_errors.get("okx_xaut_spot"),
            },
            "bitfinex_xaut": {
                "available": bool(bitfinex),
                "status": "ready" if bitfinex else "unavailable",
                "error": stage_errors.get("bitfinex_xaut"),
            },
        }
        if stage_errors and result.get("status") == "ready":
            result["status"] = "degraded"
        _cache = (time.monotonic(), result)
        return result
