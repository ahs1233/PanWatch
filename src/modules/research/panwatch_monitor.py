"""PanWatch integration for persistent beliefs and change monitoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable

from sqlalchemy.orm import Session

from .belief_state import (
    BeliefEvent,
    BeliefSnapshot,
    BeliefStateEngine,
    BeliefUpdate,
)
from .belief_store import (
    load_latest_belief_snapshot,
    persist_belief_cycle,
)
from .claim_graph import ClaimGraph, ClaimStatus
from .evidence import utc
from .falsification import FalsificationEngine, FalsificationProbe
from .ledger import EvidenceLedger


@dataclass(frozen=True)
class PanWatchBeliefCycle:
    cycle_id: str
    started_at: datetime
    completed_at: datetime
    claim_count: int
    changed_count: int
    falsified_count: int
    probe_count: int
    updates: tuple[BeliefUpdate, ...]
    events: tuple[BeliefEvent, ...]
    probes: tuple[FalsificationProbe, ...]
    impacted_claim_ids: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "claim_count": self.claim_count,
            "changed_count": self.changed_count,
            "falsified_count": self.falsified_count,
            "probe_count": self.probe_count,
            "event_count": len(self.events),
            "impacted_claim_ids": list(self.impacted_claim_ids),
        }


def _cycle_id(started_at: datetime, claim_ids: Iterable[str]) -> str:
    raw = json.dumps(
        {
            "started_at": utc(started_at).isoformat(),
            "claim_ids": sorted(claim_ids),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"cycle_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


class PanWatchBeliefMonitor:
    """Run one persistent PanWatch reasoning cycle over selected claims."""

    def __init__(
        self,
        *,
        graph: ClaimGraph,
        ledger: EvidenceLedger,
        falsification: FalsificationEngine,
        belief_engine: BeliefStateEngine | None = None,
    ) -> None:
        self.graph = graph
        self.ledger = ledger
        self.falsification = falsification
        self.belief_engine = belief_engine or BeliefStateEngine()

    def run_cycle(
        self,
        *,
        db: Session,
        claim_ids: Iterable[str] | None = None,
        evaluated_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PanWatchBeliefCycle:
        started = utc(evaluated_at)
        ids = tuple(
            sorted(
                claim_ids
                if claim_ids is not None
                else (claim.claim_id for claim in self.graph.claims)
            )
        )
        cycle_id = _cycle_id(started, ids)
        updates: list[BeliefUpdate] = []
        all_events: list[BeliefEvent] = []
        probes: list[FalsificationProbe] = []
        impacted: set[str] = set()

        for claim_id in ids:
            previous = load_latest_belief_snapshot(
                db,
                claim_id=claim_id,
                before_or_at=started,
            )
            update = self.belief_engine.evaluate(
                claim_id,
                graph=self.graph,
                ledger=self.ledger,
                falsification=self.falsification,
                previous_snapshot=previous,
                evaluated_at=started,
                metadata={
                    "cycle_id": cycle_id,
                    **dict(metadata or {}),
                },
            )
            updates.append(update)
            all_events.extend(update.events)

            report = self.falsification.evaluate(
                claim_id,
                graph=self.graph,
                ledger=self.ledger,
                as_of=started,
            )
            probes.extend(report.probes)

            if update.changed:
                impacted.update(self.graph.impact_set(claim_id))

        completed = utc(evaluated_at)
        cycle = PanWatchBeliefCycle(
            cycle_id=cycle_id,
            started_at=started,
            completed_at=completed,
            claim_count=len(updates),
            changed_count=sum(1 for item in updates if item.changed),
            falsified_count=sum(
                1
                for item in updates
                if item.snapshot.final_status is ClaimStatus.FALSIFIED
            ),
            probe_count=len(probes),
            updates=tuple(updates),
            events=tuple(all_events),
            probes=tuple(
                sorted(
                    probes,
                    key=lambda item: (-item.priority, item.rule_id),
                )
            ),
            impacted_claim_ids=tuple(sorted(impacted)),
        )
        persist_belief_cycle(
            db,
            cycle=cycle,
            metadata=dict(metadata or {}),
        )
        return cycle
