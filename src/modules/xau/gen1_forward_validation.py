"""Automatic forward-validation collector for the complete GEN1 Gold pipeline.

This scheduler is slower than the hot-path scanner. It collects real forward
observations without forcing Ahmed ToolBox on every run; macro uses its normal
cache/background-refresh policy. Research-only; never executes orders.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.modules.xau.gen1_pipeline import run_gen1_trade_gold_pipeline
from src.platform.runtime.config import Settings

logger = logging.getLogger(__name__)


class Gen1ForwardValidationScheduler:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.scheduler = AsyncIOScheduler(timezone=self.settings.xau_paper_timezone)
        self._running = False

    async def _collect(self) -> None:
        if self._running:
            logger.debug("[GEN1 forward] previous collection still running")
            return
        self._running = True
        try:
            result = await run_gen1_trade_gold_pipeline(
                force_macro=False,
                record_observation=True,
                observation_source="scheduled_forward_validation",
            )
            observation = result.get("live_observation") or {}
            logger.info(
                "[GEN1 forward] decision=%s confidence=%s pipeline=%s ledger=%s missing=%s revision=%s",
                result.get("decision"),
                result.get("confidence"),
                result.get("pipeline_status"),
                observation.get("status"),
                result.get("missing_layers"),
                result.get("strategy_revision"),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[GEN1 forward] collection failed type=%s", type(exc).__name__)
        finally:
            self._running = False

    def start(self) -> None:
        if not self.settings.xau_gen1_forward_validation_enabled:
            logger.info("[GEN1 forward] scheduler disabled")
            return
        interval = max(60, int(self.settings.xau_gen1_forward_interval_seconds))
        self.scheduler.add_job(
            self._collect,
            "interval",
            seconds=interval,
            next_run_time=datetime.now(self.scheduler.timezone) + timedelta(seconds=45),
            id="gen1-forward-validation",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "[GEN1 forward] scheduler started interval_seconds=%s first_run_delay_seconds=45",
            interval,
        )

    def shutdown(self) -> None:
        if self.scheduler.running:
            self.scheduler.shutdown(wait=False)
