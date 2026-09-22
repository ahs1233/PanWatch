"""Background scheduler for the bounded Automatic Research Loop."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.platform.runtime.config import Settings

from .automatic_research import run_automatic_research_once
from .xau_claim_bridge import sync_xau_macro_claim

logger = logging.getLogger(__name__)


class AutomaticResearchScheduler:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.scheduler = AsyncIOScheduler(timezone=self.settings.app_timezone)
        self._running = False

    async def _run(self) -> None:
        if self._running:
            logger.debug("[auto-research] previous cycle still running; skipped")
            return
        self._running = True
        try:
            if (
                self.settings.panwatch_profile.strip().lower() == "xau"
                and self.settings.auto_research_bootstrap_xau_claims
            ):
                try:
                    bridge = await sync_xau_macro_claim()
                    logger.info(
                        "[auto-research] XAU claim synced claim=%s bias=%s evidence_inserted=%s",
                        bridge.get("claim_id"),
                        bridge.get("bias"),
                        bridge.get("evidence_inserted"),
                    )
                except Exception as exc:
                    logger.warning(
                        "[auto-research] XAU claim bridge failed error=%s",
                        type(exc).__name__,
                    )

            result = await run_automatic_research_once(self.settings)
            level = logging.INFO if result.evidence_added or result.beliefs_changed else logging.DEBUG
            logger.log(
                level,
                "[auto-research] run=%s status=%s probes=%s/%s cooldown=%s tool_calls=%s docs=%s evidence=%s raw=%s classified=%s belief_changes=%s errors=%s",
                result.run_id,
                result.status,
                result.probes_executed,
                result.probes_planned,
                result.probes_skipped_cooldown,
                result.tool_calls,
                result.documents_read,
                result.evidence_added,
                result.raw_evidence_added,
                result.classified_evidence_added,
                result.beliefs_changed,
                list(result.errors),
            )
        except Exception as exc:
            logger.warning(
                "[auto-research] cycle failed error=%s",
                type(exc).__name__,
                exc_info=True,
            )
        finally:
            self._running = False

    async def run_once(self):
        await self._run()

    def start(self) -> None:
        interval = max(15, int(self.settings.auto_research_interval_minutes))
        self.scheduler.add_job(
            self._run,
            "date",
            run_date=datetime.now(self.scheduler.timezone) + timedelta(seconds=20),
            id="automatic_research_bootstrap",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            self._run,
            "interval",
            minutes=interval,
            jitter=min(120, max(15, interval * 2)),
            id="automatic_research_cycle",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "Automatic Research scheduler started interval=%sm max_probes=%s max_sources=%s max_tool_calls=%s cooldown=%sm",
            interval,
            self.settings.auto_research_max_probes,
            self.settings.auto_research_max_sources_per_probe,
            self.settings.auto_research_max_tool_calls,
            self.settings.auto_research_probe_cooldown_minutes,
        )

    def shutdown(self) -> None:
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
        logger.info("Automatic Research scheduler stopped")
