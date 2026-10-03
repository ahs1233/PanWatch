"""Append-only evidence ledger and deterministic research integrity guards."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable

from .evidence import (
    EvidenceRecord,
    EvidenceRelation,
    ObservationKind,
    SourceProvenance,
    SourceTier,
    utc,
)


_TIER_WEIGHT = {
    SourceTier.OFFICIAL_PRIMARY: 1.00,
    SourceTier.PRIMARY: 0.90,
    SourceTier.SECONDARY: 0.72,
    SourceTier.AGGREGATOR: 0.50,
    SourceTier.SOCIAL: 0.35,
    SourceTier.UNKNOWN: 0.45,
}


@dataclass(frozen=True)
class GuardResult:
    passed: bool
    code: str
    detail: str


@dataclass(frozen=True)
class NumericResolution:
    claim_key: str
    status: str
    selected_evidence_id: str | None
    value: float | None
    unit: str
    period: str
    conflicts: tuple[str, ...]
    independent_source_count: int
    detail: str


class EvidenceLedger:
    """Semantic append-only ledger shared by runtime and benchmarks."""

    def __init__(self) -> None:
        self._sources: dict[str, SourceProvenance] = {}
        self._evidence: dict[str, EvidenceRecord] = {}
        self._order: list[str] = []

    @property
    def sources(self) -> tuple[SourceProvenance, ...]:
        return tuple(self._sources.values())

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._evidence[item] for item in self._order)

    def register_source(self, source: SourceProvenance) -> SourceProvenance:
        existing = self._sources.get(source.source_id)
        if existing is not None:
            return existing
        self._sources[source.source_id] = source
        return source

    def append(self, record: EvidenceRecord) -> bool:
        if record.source_id not in self._sources:
            raise ValueError(f"unknown source_id: {record.source_id}")
        if record.evidence_id in self._evidence:
            return False
        self._evidence[record.evidence_id] = record
        self._order.append(record.evidence_id)
        return True

    def source_for(self, record: EvidenceRecord) -> SourceProvenance:
        return self._sources[record.source_id]

    def for_claim(
        self,
        claim_key: str,
        *,
        kinds: Iterable[ObservationKind] | None = None,
        as_of: datetime | None = None,
    ) -> list[EvidenceRecord]:
        allowed = (
            {ObservationKind(kind) for kind in kinds}
            if kinds is not None
            else None
        )
        cutoff = utc(as_of) if as_of is not None else None
        rows: list[EvidenceRecord] = []
        for record in self.records:
            if record.claim_key != claim_key:
                continue
            if allowed is not None and record.observation_kind not in allowed:
                continue
            if cutoff is not None and record.recorded_at > cutoff:
                continue
            rows.append(record)
        return rows

    def chronology(self, claim_key: str) -> list[EvidenceRecord]:
        return sorted(
            self.for_claim(claim_key),
            key=lambda item: (
                item.event_time or item.observed_at,
                item.recorded_at,
                item.evidence_id,
            ),
        )

    def lag_seconds(
        self,
        earlier_evidence_id: str,
        later_evidence_id: str,
    ) -> float:
        earlier = self._evidence[earlier_evidence_id]
        later = self._evidence[later_evidence_id]
        a = earlier.event_time or earlier.observed_at
        b = later.event_time or later.observed_at
        return (b - a).total_seconds()

    def independent_source_count(self, claim_key: str) -> int:
        keys = {
            self.source_for(record).independence_key
            for record in self.for_claim(claim_key)
        }
        return len(keys)

    def source_independence_ratio(self, claim_key: str) -> float:
        rows = self.for_claim(claim_key)
        if not rows:
            return 0.0
        unique_sources = {record.source_id for record in rows}
        independent = {
            self._sources[source_id].independence_key
            for source_id in unique_sources
        }
        return len(independent) / len(unique_sources)

    def actual_forecast_guard(self, claim_key: str) -> GuardResult:
        rows = self.for_claim(claim_key)
        actual_like = {
            ObservationKind.ACTUAL,
            ObservationKind.REVISION,
            ObservationKind.HISTORICAL,
        }
        if any(record.observation_kind in actual_like for record in rows):
            return GuardResult(
                True,
                "actual_present",
                "actual/revision evidence is present",
            )
        forecast_like = {
            ObservationKind.FORECAST,
            ObservationKind.ESTIMATE,
            ObservationKind.GUIDANCE,
        }
        if any(record.observation_kind in forecast_like for record in rows):
            return GuardResult(
                False,
                "forecast_only",
                "forecast/estimate evidence cannot satisfy an actual claim",
            )
        return GuardResult(False, "actual_missing", "no actual evidence is present")

    def kind_separation_guard(
        self,
        claim_key: str,
        left: ObservationKind,
        right: ObservationKind,
    ) -> GuardResult:
        left_rows = self.for_claim(claim_key, kinds=[left])
        right_rows = self.for_claim(claim_key, kinds=[right])
        if left_rows and right_rows:
            return GuardResult(
                True,
                "kinds_separated",
                f"{left.value} and {right.value} remain distinct",
            )
        return GuardResult(
            False,
            "kind_missing",
            f"need both {left.value} and {right.value}",
        )

    def freshness_guard(
        self,
        claim_key: str,
        *,
        max_age: timedelta,
        as_of: datetime | None = None,
        kinds: Iterable[ObservationKind] | None = None,
    ) -> GuardResult:
        now = utc(as_of)
        rows = self.for_claim(
            claim_key,
            kinds=kinds,
            as_of=as_of,
        )
        if not rows:
            return GuardResult(False, "evidence_missing", "no evidence available")
        freshest = max(rows, key=lambda item: item.event_time or item.observed_at)
        timestamp = freshest.event_time or freshest.observed_at
        age = now - timestamp
        if age <= max_age:
            return GuardResult(
                True,
                "fresh",
                f"freshest evidence age={age.total_seconds():.0f}s",
            )
        return GuardResult(
            False,
            "stale",
            (
                f"freshest evidence age={age.total_seconds():.0f}s "
                f"exceeds {max_age.total_seconds():.0f}s"
            ),
        )

    def claim_confidence(
        self,
        claim_key: str,
        *,
        as_of: datetime | None = None,
    ) -> float:
        """Aggregate independent evidence without double-counting copies."""
        rows = self.for_claim(claim_key, as_of=as_of)
        grouped: dict[str, EvidenceRecord] = {}
        for record in rows:
            if record.relation is EvidenceRelation.CONTEXT:
                continue
            key = self.source_for(record).independence_key
            current = grouped.get(key)
            if current is None or self._weighted_strength(
                record
            ) > self._weighted_strength(current):
                grouped[key] = record
        if not grouped:
            return 0.5
        directional = 0.0
        weight_total = 0.0
        for record in grouped.values():
            strength = self._weighted_strength(record)
            sign = (
                1.0
                if record.relation is EvidenceRelation.SUPPORTS
                else -1.0
            )
            directional += sign * strength
            weight_total += strength
        if weight_total <= 0:
            return 0.5
        normalized = directional / weight_total
        return round(
            max(0.0, min(1.0, 0.5 + normalized * 0.5)),
            4,
        )

    def resolve_numeric(
        self,
        claim_key: str,
        *,
        preferred_kind: ObservationKind = ObservationKind.ACTUAL,
        tolerance: float = 1e-9,
        as_of: datetime | None = None,
    ) -> NumericResolution:
        rows = [
            record
            for record in self.for_claim(claim_key, as_of=as_of)
            if record.numeric_value is not None
        ]
        if not rows:
            return NumericResolution(
                claim_key, "missing", None, None, "", "", (), 0,
                "no numeric evidence",
            )

        eligible = self._eligible_numeric_rows(rows, preferred_kind)
        if not eligible:
            return NumericResolution(
                claim_key,
                "kind_missing",
                None,
                None,
                "",
                "",
                (),
                0,
                f"no {preferred_kind.value} numeric evidence",
            )

        by_independence: dict[str, EvidenceRecord] = {}
        for record in eligible:
            key = self.source_for(record).independence_key
            current = by_independence.get(key)
            if current is None or self._resolution_rank(
                record
            ) > self._resolution_rank(current):
                by_independence[key] = record

        representatives = list(by_independence.values())
        selected = max(representatives, key=self._resolution_rank)
        conflicts = tuple(
            record.evidence_id
            for record in representatives
            if not _numeric_equal(selected, record, tolerance=tolerance)
        )
        if selected.observation_kind is ObservationKind.REVISION:
            status = "resolved_revision"
            detail = "latest/highest-quality revision selected"
        elif conflicts:
            status = "conflict"
            detail = "independent numeric sources disagree"
        else:
            status = "resolved"
            detail = "independent numeric sources agree"

        return NumericResolution(
            claim_key=claim_key,
            status=status,
            selected_evidence_id=selected.evidence_id,
            value=selected.numeric_value,
            unit=selected.unit,
            period=selected.period,
            conflicts=conflicts,
            independent_source_count=len(representatives),
            detail=detail,
        )

    def _eligible_numeric_rows(
        self,
        rows: list[EvidenceRecord],
        preferred_kind: ObservationKind,
    ) -> list[EvidenceRecord]:
        preferred_kind = ObservationKind(preferred_kind)
        if preferred_kind is ObservationKind.ACTUAL:
            revisions = [
                record
                for record in rows
                if (
                    record.observation_kind is ObservationKind.REVISION
                    and record.revision_of
                    and record.revision_of in self._evidence
                )
            ]
            if revisions:
                return revisions
            return [
                record
                for record in rows
                if record.observation_kind
                in {ObservationKind.ACTUAL, ObservationKind.HISTORICAL}
            ]
        return [
            record
            for record in rows
            if record.observation_kind is preferred_kind
        ]

    def _weighted_strength(self, record: EvidenceRecord) -> float:
        source = self.source_for(record)
        return (
            max(0.0, min(1.0, record.confidence))
            * _TIER_WEIGHT[source.source_tier]
        )

    def _resolution_rank(self, record: EvidenceRecord) -> tuple:
        source = self.source_for(record)
        kind_priority = (
            3
            if record.observation_kind is ObservationKind.REVISION
            else 2
            if record.observation_kind is ObservationKind.ACTUAL
            else 1
        )
        timestamp = record.event_time or record.observed_at
        return (
            kind_priority,
            _TIER_WEIGHT[source.source_tier],
            record.confidence,
            timestamp.timestamp(),
            record.recorded_at.timestamp(),
        )


def _numeric_equal(
    left: EvidenceRecord,
    right: EvidenceRecord,
    *,
    tolerance: float,
) -> bool:
    if left.unit and right.unit and left.unit != right.unit:
        return False
    if left.period and right.period and left.period != right.period:
        return False
    assert left.numeric_value is not None
    assert right.numeric_value is not None
    scale = max(1.0, abs(left.numeric_value), abs(right.numeric_value))
    return (
        abs(left.numeric_value - right.numeric_value)
        <= tolerance * scale
    )
