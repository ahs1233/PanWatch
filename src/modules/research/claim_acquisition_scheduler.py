"""Bounded scheduler for domain-neutral claim acquisition."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.platform.runtime.config import Settings

from .claim_acquisition import acquire_topic_once

logger = logging.getLogger(__name__)


class ClaimAcquisitionScheduler:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.scheduler = AsyncIOScheduler(
            timezone=self.settings.app_timezone,
        )
        self._running = False

    async def _run(self) -> None:
        if self._running:
            logger.debug(
                "[claim-acquisition] previous cycle still running; skipped"
            )
            return
        self._running = True
        try:
            results = await acquire_topic_once(self.settings)
            if not results:
                logger.debug(
                    "[claim-acquisition] no configured or active topics"
                )
                return
            for result in results:
                logger.info(
                    "[claim-acquisition] run=%s topic=%r docs=%s candidates=%s "
                    "accepted=%s duplicates=%s rejected=%s superseded=%s "
                    "belief_changes=%s tool_calls=%s errors=%s",
                    result.run_id,
                    result.seed_topic,
                    result.documents_seen,
                    result.candidates_extracted,
                    result.claims_accepted,
                    result.duplicates,
                    result.rejected,
                    result.superseded,
                    result.belief_changes,
                    result.tool_calls,
                    list(result.errors),
                )
        except Exception as exc:
            logger.warning(
                "[claim-acquisition] cycle failed error=%s",
                type(exc).__name__,
                exc_info=True,
            )
        finally:
            self._running = False

    async def run_once(self) -> None:
        await self._run()

    def start(self) -> None:
        interval = max(
            30,
            int(self.settings.claim_acquisition_interval_minutes),
        )
        # Start after the faster falsification loop has completed its bootstrap
        # cycle so both subsystems do not hit the AI provider simultaneously.
        self.scheduler.add_job(
            self._run,
            "date",
            run_date=(
                datetime.now(self.scheduler.timezone)
                + timedelta(seconds=90)
            ),
            id="claim_acquisition_bootstrap",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._run,
            "interval",
            minutes=interval,
            jitter=min(300, max(30, interval * 2)),
            id="claim_acquisition_cycle",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "Claim Acquisition scheduler started interval=%sm max_topics=%s "
            "max_documents=%s max_claims_per_document=%s",
            interval,
            self.settings.claim_acquisition_max_topics,
            self.settings.claim_acquisition_max_documents,
            self.settings.claim_acquisition_max_claims_per_document,
        )

    def shutdown(self) -> None:
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
        logger.info("Claim Acquisition scheduler stopped")
