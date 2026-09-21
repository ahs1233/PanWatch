"""Background XAU research scanner for the XAU profile.

This scanner produces research state only. It never emits or places execution
orders because the current feed is GC=F and execution_eligible=False.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .service import get_xau_snapshot

logger = logging.getLogger(__name__)


class XAUResearchScheduler:
    def __init__(self, timezone: str = "UTC", interval_seconds: int = 10):
        self.scheduler = AsyncIOScheduler(timezone=timezone)
        self.interval_seconds = max(5, int(interval_seconds))
        self._running = False
        self._last_signature = None

    async def _scan(self):
        if self._running:
            return
        self._running = True
        try:
            snapshot = await get_xau_snapshot(force=False)
            signature = (
                str(snapshot.get("status")),
                str(snapshot.get("candidate")),
                str(snapshot.get("alignment")),
                str((snapshot.get("micro") or {}).get("direction")),
            )
            if signature != self._last_signature:
                logger.info(
                    "[XAU] state status=%s candidate=%s alignment=%s mode=%s micro=%s micro_age=%s spot=%s spot_source=%s spot_stale=%s price_proxy=%s execution=%s gates=%s",
                    snapshot.get("status"),
                    snapshot.get("candidate"),
                    snapshot.get("alignment"),
                    snapshot.get("technical_mode"),
                    (snapshot.get("micro") or {}).get("direction"),
                    (snapshot.get("micro") or {}).get("age_seconds"),
                    (snapshot.get("indicative_spot") or {}).get("price"),
                    (snapshot.get("indicative_spot") or {}).get("source"),
                    (snapshot.get("indicative_spot") or {}).get("is_stale"),
                    snapshot.get("price"),
                    snapshot.get("execution_status"),
                    snapshot.get("block_reasons"),
                )
                self._last_signature = signature
            else:
                logger.debug("[XAU] research state unchanged: %s", signature)
        except Exception as exc:
            logger.warning("[XAU] research scan unavailable: %s", type(exc).__name__)
        finally:
            self._running = False

    def start(self):
        self.scheduler.add_job(
            self._scan,
            "date",
            run_date=datetime.now(self.scheduler.timezone) + timedelta(seconds=3),
            id="xau_research_bootstrap",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._scan,
            "interval",
            seconds=self.interval_seconds,
            id="xau_research_scan",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "XAU research scheduler started, interval=%ss",
            self.interval_seconds,
        )

    def shutdown(self):
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
