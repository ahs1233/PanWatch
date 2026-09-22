"""Runtime scheduler for persistent PanWatch belief cycles.

The scheduler is intentionally local/database-only. It performs no network
research itself and runs a cycle only when research inputs changed, preventing
the repeated-call explosion that a naive polling loop could create.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from sqlalchemy import func

from src.platform.persistence.models import (
    ResearchClaimEdgeRecord,
    ResearchClaimEvidenceLinkRecord,
    ResearchClaimRecord,
    ResearchEvidenceRecord,
    ResearchFalsificationRuleRecord,
)

from .evidence_store import load_all_evidence_ledger
from .panwatch_monitor import PanWatchBeliefMonitor
from .reasoning_store import (
    load_claim_graph,
    load_falsification_engine,
)
from .store import open_research_session

logger = logging.getLogger(__name__)


class PersistentBeliefScheduler:
    """Evaluate persisted research beliefs only when their inputs change."""

    def __init__(
        self,
        *,
        timezone: str = "UTC",
        interval_seconds: int = 60,
    ) -> None:
        self.scheduler = AsyncIOScheduler(timezone=timezone)
        self.interval_seconds = max(30, int(interval_seconds))
        self._running = False
        self._last_input_signature: tuple | None = None

    @staticmethod
    def _table_signature(db, model, time_column, id_column) -> tuple:
        row = db.query(
            func.count(id_column),
            func.max(time_column),
            func.max(id_column),
        ).one()
        return (
            int(row[0] or 0),
            str(row[1] or ""),
            str(row[2] or ""),
        )

    def _input_signature(self, db) -> tuple:
        return (
            self._table_signature(
                db,
                ResearchEvidenceRecord,
                ResearchEvidenceRecord.recorded_at,
                ResearchEvidenceRecord.evidence_id,
            ),
            self._table_signature(
                db,
                ResearchClaimRecord,
                ResearchClaimRecord.created_at,
                ResearchClaimRecord.claim_id,
            ),
            self._table_signature(
                db,
                ResearchClaimEdgeRecord,
                ResearchClaimEdgeRecord.created_at,
                ResearchClaimEdgeRecord.edge_id,
            ),
            self._table_signature(
                db,
                ResearchClaimEvidenceLinkRecord,
                ResearchClaimEvidenceLinkRecord.created_at,
                ResearchClaimEvidenceLinkRecord.link_id,
            ),
            self._table_signature(
                db,
                ResearchFalsificationRuleRecord,
                ResearchFalsificationRuleRecord.created_at,
                ResearchFalsificationRuleRecord.rule_id,
            ),
        )

    async def _scan(self) -> None:
        if self._running:
            return
        self._running = True
        db = open_research_session()
        try:
            signature = self._input_signature(db)
            if signature == self._last_input_signature:
                logger.debug(
                    "[belief-monitor] research inputs unchanged; cycle skipped"
                )
                return

            ledger = load_all_evidence_ledger(db)
            graph = load_claim_graph(db, ledger=ledger)
            if not graph.claims:
                self._last_input_signature = signature
                logger.debug(
                    "[belief-monitor] no persisted claims; cycle skipped"
                )
                return

            falsification = load_falsification_engine(
                db,
                graph=graph,
            )
            cycle = PanWatchBeliefMonitor(
                graph=graph,
                ledger=ledger,
                falsification=falsification,
            ).run_cycle(
                db=db,
                metadata={"trigger": "persistent_belief_scheduler"},
            )
            self._last_input_signature = signature
            logger.info(
                "[belief-monitor] cycle=%s claims=%s changed=%s "
                "falsified=%s probes=%s events=%s impacted=%s",
                cycle.cycle_id,
                cycle.claim_count,
                cycle.changed_count,
                cycle.falsified_count,
                cycle.probe_count,
                len(cycle.events),
                len(cycle.impacted_claim_ids),
            )
        except Exception as exc:
            db.rollback()
            logger.warning(
                "[belief-monitor] cycle failed: %s",
                type(exc).__name__,
            )
        finally:
            db.close()
            self._running = False

    def start(self) -> None:
        self.scheduler.add_job(
            self._scan,
            "date",
            run_date=(
                datetime.now(self.scheduler.timezone)
                + timedelta(seconds=8)
            ),
            id="persistent_belief_bootstrap",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._scan,
            "interval",
            seconds=self.interval_seconds,
            id="persistent_belief_scan",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.start()
        logger.info(
            "Persistent belief scheduler started, interval=%ss",
            self.interval_seconds,
        )

    def shutdown(self) -> None:
        try:
            self.scheduler.shutdown(wait=False)
        except Exception:
            pass
