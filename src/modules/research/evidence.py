"""Typed, deterministic source provenance and evidence records.

The module is domain-neutral. XAU, macro research, company research and future
PanWatch domains should all pass through the same evidence identity rules.
No LLM is required on this path: provenance must be inspectable and stable.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_TRACKING_QUERY_PREFIXES = ("utm_",)
_TRACKING_QUERY_KEYS = {
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src",
}
_WHITESPACE_RE = re.compile(r"\s+")


class SourceTier(StrEnum):
    OFFICIAL_PRIMARY = "official_primary"
    PRIMARY = "primary"
    SECONDARY = "secondary"
    AGGREGATOR = "aggregator"
    SOCIAL = "social"
    UNKNOWN = "unknown"


class ObservationKind(StrEnum):
    RAW_SOURCE = "raw_source"
    ACTUAL = "actual"
    FORECAST = "forecast"
    REVISION = "revision"
    ESTIMATE = "estimate"
    GUIDANCE = "guidance"
    POLICY_STATEMENT = "policy_statement"
    MARKET_PRICING = "market_pricing"
    OPINION = "opinion"
    HISTORICAL = "historical"


class EvidenceRelation(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CONTEXT = "context"


def utc(value: datetime | None) -> datetime:
    value = value or datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


def content_digest(value: str) -> str:
    return hashlib.sha256(normalize_text(value).encode("utf-8")).hexdigest()


def canonicalize_url(url: str) -> str:
    """Normalize URLs and remove tracking-only parameters."""
    raw = str(url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    netloc = host
    if port and not (
        (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    ):
        netloc = f"{host}:{port}"
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query_items = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lower = key.lower()
        if lower in _TRACKING_QUERY_KEYS or any(
            lower.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES
        ):
            continue
        query_items.append((key, value))
    query = urlencode(sorted(query_items))
    return urlunsplit((scheme, netloc, path, query, ""))


def _iso(value: datetime | None) -> str:
    return utc(value).isoformat()


def _domain(canonical_url: str) -> str:
    return (urlsplit(canonical_url).hostname or "").lower()


def _stable_id(prefix: str, payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class SourceProvenance:
    source_id: str
    url: str
    canonical_url: str
    domain: str
    publisher: str
    title: str
    source_tier: SourceTier
    source_family: str
    independence_key: str
    published_at: datetime | None
    retrieved_at: datetime
    observed_at: datetime
    content_hash: str
    parent_source_id: str | None = None
    tool_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source_tier"] = self.source_tier.value
        for key in ("published_at", "retrieved_at", "observed_at"):
            value = payload[key]
            payload[key] = utc(value).isoformat() if value is not None else None
        return payload


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    claim_key: str
    source_id: str
    statement: str
    relation: EvidenceRelation
    observation_kind: ObservationKind
    event_time: datetime | None
    observed_at: datetime
    recorded_at: datetime
    confidence: float
    content_hash: str
    numeric_value: float | None = None
    unit: str = ""
    period: str = ""
    revision_of: str | None = None
    supersedes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["relation"] = self.relation.value
        payload["observation_kind"] = self.observation_kind.value
        for key in ("event_time", "observed_at", "recorded_at"):
            value = payload[key]
            payload[key] = utc(value).isoformat() if value is not None else None
        return payload


def build_source(
    *,
    url: str,
    content: str,
    publisher: str = "",
    title: str = "",
    source_tier: SourceTier = SourceTier.UNKNOWN,
    source_family: str = "",
    independence_key: str = "",
    upstream_origin: str = "",
    published_at: datetime | None = None,
    retrieved_at: datetime | None = None,
    observed_at: datetime | None = None,
    parent_source_id: str | None = None,
    tool_name: str = "",
    metadata: dict[str, Any] | None = None,
) -> SourceProvenance:
    canonical = canonicalize_url(url)
    domain = _domain(canonical)
    digest = content_digest(content)
    family = (
        normalize_text(source_family)
        or domain
        or normalize_text(publisher).lower()
        or "unknown"
    )
    independent = (
        normalize_text(upstream_origin).lower()
        or normalize_text(independence_key).lower()
        or family.lower()
        or domain
        or "unknown"
    )
    source_id = _stable_id(
        "src",
        {
            "canonical_url": canonical,
            "content_hash": digest,
            "published_at": _iso(published_at) if published_at else None,
        },
    )
    return SourceProvenance(
        source_id=source_id,
        url=str(url or ""),
        canonical_url=canonical,
        domain=domain,
        publisher=normalize_text(publisher),
        title=normalize_text(title),
        source_tier=SourceTier(source_tier),
        source_family=family,
        independence_key=independent,
        published_at=utc(published_at) if published_at else None,
        retrieved_at=utc(retrieved_at),
        observed_at=utc(observed_at or retrieved_at),
        content_hash=digest,
        parent_source_id=parent_source_id,
        tool_name=normalize_text(tool_name),
        metadata=dict(metadata or {}),
    )


def build_evidence(
    *,
    claim_key: str,
    source: SourceProvenance,
    statement: str,
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
    observation_kind: ObservationKind = ObservationKind.ACTUAL,
    event_time: datetime | None = None,
    observed_at: datetime | None = None,
    recorded_at: datetime | None = None,
    confidence: float = 1.0,
    numeric_value: float | None = None,
    unit: str = "",
    period: str = "",
    revision_of: str | None = None,
    supersedes: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceRecord:
    claim = normalize_text(claim_key)
    if not claim:
        raise ValueError("claim_key is required")
    text = normalize_text(statement)
    if not text:
        raise ValueError("statement is required")
    confidence = max(0.0, min(1.0, float(confidence)))
    digest = content_digest(text)
    evidence_id = _stable_id(
        "ev",
        {
            "claim_key": claim,
            "source_id": source.source_id,
            "statement_hash": digest,
            "kind": ObservationKind(observation_kind).value,
            "relation": EvidenceRelation(relation).value,
            "value": numeric_value,
            "unit": normalize_text(unit),
            "period": normalize_text(period),
            "revision_of": revision_of,
        },
    )
    return EvidenceRecord(
        evidence_id=evidence_id,
        claim_key=claim,
        source_id=source.source_id,
        statement=text,
        relation=EvidenceRelation(relation),
        observation_kind=ObservationKind(observation_kind),
        event_time=utc(event_time) if event_time else None,
        observed_at=utc(observed_at or source.observed_at),
        recorded_at=utc(
            recorded_at
            if recorded_at is not None
            else observed_at
            if observed_at is not None
            else source.observed_at
        ),
        confidence=confidence,
        content_hash=digest,
        numeric_value=float(numeric_value) if numeric_value is not None else None,
        unit=normalize_text(unit),
        period=normalize_text(period),
        revision_of=revision_of,
        supersedes=supersedes,
        metadata=dict(metadata or {}),
    )
