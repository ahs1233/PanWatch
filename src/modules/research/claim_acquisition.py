"""General, domain-neutral claim acquisition for Ahmed Research Engine.

The admission contract is conservative:
- a candidate must be backed by an exact quote from the source text;
- extraction is audited before admission;
- exact semantic duplicates reuse the existing claim and only add evidence;
- supersession requires the same claim key plus an explicit revision cue;
- newly admitted claims receive generic falsification rules automatically;
- every decision is persisted for later audit.

This layer accepts arbitrary text (reports, transcripts, filings, articles) and
can also search a topic through the existing Ahmed ToolBox gateway.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol
from urllib.parse import urlsplit

from sqlalchemy.orm import Session

from src.platform.ai.ai_failover import build_failover_client
from src.platform.persistence.models import (
    ResearchAcquisitionRunRecord,
    ResearchClaimCandidateRecord,
    ResearchClaimResolutionRecord,
)
from src.platform.runtime.config import Settings

from .automatic_research import (
    AhmedToolboxResearchGateway,
    ResearchDocument,
    _json_value,
    _relevant_excerpt,
)
from .claim_graph import (
    ClaimGraph,
    ClaimKind,
    ClaimNode,
    ClaimRelation,
    build_claim,
)
from .evidence import (
    EvidenceRelation,
    ObservationKind,
    SourceTier,
    build_evidence,
    build_source,
    normalize_text,
    utc,
)
from .evidence_store import load_full_ledger, persist_ledger
from .claim_resolution import (
    ConservativeClaimResolver,
    SemanticClaimRelation,
)
from .falsification import (
    FalsificationEngine,
    FalsificationRuleType,
    build_falsification_rule,
)
from .panwatch_monitor import PanWatchBeliefMonitor
from .reasoning_store import (
    load_claim_graph,
    load_falsification_engine,
    persist_claim_graph,
    persist_falsification_engine,
)
from .research_store import research_session

logger = logging.getLogger(__name__)

_ALLOWED_KINDS = {item.value: item for item in ClaimKind}
_ALLOWED_OBSERVATIONS = {item.value: item for item in ObservationKind}
_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._:-]{2,159}$")
_REVISION_CUE_RE = re.compile(
    r"\b(revised|updated|restated|amended|corrected|raised to|lowered to|"
    r"increased to|decreased to|now stands at)\b|"
    r"(تم تعديل|معد.?ل|محدث|محد.?ث|صحح|صُحح|ارتفع إلى|انخفض إلى)",
    re.IGNORECASE,
)
_FUTURE_CUE_RE = re.compile(
    r"\b(will|expected to|expects to|forecast|forecasted|projected|"
    r"plans to|set to|scheduled to|by 20\d{2})\b",
    re.IGNORECASE,
)
_FACT_SIGNAL_RE = re.compile(
    r"(\b20\d{2}\b|\d+(?:\.\d+)?\s*%|\$\s*\d|"
    r"\b\d+(?:\.\d+)?\s*(?:billion|million|trillion|"
    r"units|chips|wafers|months|years|days|percent)\b)",
    re.IGNORECASE,
)
_BOILERPLATE_RE = re.compile(
    r"\b(cookie|privacy policy|terms of use|subscribe|newsletter|"
    r"sign in|log in|contact us|read more|all rights reserved)\b",
    re.IGNORECASE,
)
_PAGE_METADATA_RE = re.compile(
    r"^(?:published(?:\s+time|\s+date)?|last\s+updated|updated(?:\s+at)?|"
    r"date\s+published|reading\s+time|author|byline)\s*:\s*",
    re.IGNORECASE,
)
_STRUCTURAL_NON_CLAIM_RE = re.compile(
    r"^\s*(?:#{1,6}\s+|"
    r"(?:figure|fig\.?|table|chart|source|note|section|contents?)\s*[:#-]\s*)",
    re.IGNORECASE,
)


def _is_structural_non_claim(text: str) -> bool:
    value = normalize_text(str(text or ""))
    if not value:
        return True
    if _STRUCTURAL_NON_CLAIM_RE.search(value):
        return True
    words = re.findall(r"\b[\w%-]+\b", value, flags=re.UNICODE)
    if len(words) <= 10 and not re.search(r"[.!?]$", value):
        if re.match(
            r"^(?:share|breakdown|distribution|overview|summary|"
            r"electricity consumption|market share|capacity|production)\b",
            value,
            flags=re.IGNORECASE,
        ):
            return True
    return False


@dataclass(frozen=True)
class ClaimCandidate:
    quote: str
    statement: str
    proposed_claim_key: str
    kind: ClaimKind
    observation_kind: ObservationKind
    confidence: float
    testable: bool = True
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    time_sensitive: bool = False
    freshness_seconds: int | None = None
    supersedes_previous: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClaimAcquisitionResult:
    run_id: str
    status: str
    seed_topic: str
    documents_seen: int
    candidates_extracted: int
    claims_accepted: int
    duplicates: int
    rejected: int
    superseded: int
    tool_calls: int
    belief_changes: int
    accepted_claim_ids: tuple[str, ...]
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "seed_topic": self.seed_topic,
            "documents_seen": self.documents_seen,
            "candidates_extracted": self.candidates_extracted,
            "claims_accepted": self.claims_accepted,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "superseded": self.superseded,
            "tool_calls": self.tool_calls,
            "belief_changes": self.belief_changes,
            "accepted_claim_ids": list(self.accepted_claim_ids),
            "errors": list(self.errors),
        }


class ClaimCandidateExtractor(Protocol):
    async def extract(
        self,
        *,
        document: ResearchDocument,
        topic_hint: str,
        max_candidates: int,
    ) -> list[ClaimCandidate]: ...


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\n".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _parse_time(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _safe_claim_key(proposed: str, statement: str) -> str:
    raw = normalize_text(proposed).lower()
    raw = re.sub(r"\s+", ".", raw)
    raw = re.sub(r"[^a-z0-9._:-]+", "", raw)
    raw = re.sub(r"\.{2,}", ".", raw).strip(".:-")
    if _KEY_RE.fullmatch(raw):
        return raw[:160]
    digest = hashlib.sha256(
        normalize_text(statement).casefold().encode("utf-8")
    ).hexdigest()[:20]
    return f"auto.claim.{digest}"


def _candidate_fingerprint(candidate: ClaimCandidate) -> str:
    payload = "|".join(
        [
            _safe_claim_key(candidate.proposed_claim_key, candidate.statement),
            normalize_text(candidate.statement).casefold(),
            normalize_text(candidate.quote).casefold(),
            candidate.kind.value,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _active_claims(graph: ClaimGraph) -> list[ClaimNode]:
    superseded = {
        claim.supersedes
        for claim in graph.claims
        if claim.supersedes
    }
    return [
        claim
        for claim in graph.claims
        if claim.claim_id not in superseded
        and not bool(claim.metadata.get("admission_invalidated"))
    ]


def _exact_claim(graph: ClaimGraph, statement: str) -> ClaimNode | None:
    target = normalize_text(statement).casefold()
    for claim in graph.claims:
        if normalize_text(claim.statement).casefold() == target:
            return claim
    return None


def _latest_active_for_key(
    graph: ClaimGraph,
    claim_key: str,
) -> ClaimNode | None:
    rows = [
        claim
        for claim in _active_claims(graph)
        if claim.claim_key == claim_key
    ]
    if not rows:
        return None
    return max(rows, key=lambda item: (item.created_at, item.claim_id))


def _source_tier(document: ResearchDocument) -> SourceTier:
    raw = str(document.metadata.get("source_tier") or "unknown")
    try:
        return SourceTier(raw)
    except ValueError:
        return SourceTier.UNKNOWN


def _deterministic_grounded_candidates(
    source_text: str,
    *,
    max_candidates: int,
    fallback_reason: str,
) -> list[ClaimCandidate]:
    """Extract only verbatim, strongly factual sentences without an LLM.

    This path is intentionally narrow. It admits a sentence only when it has a
    concrete numeric/year signal, rejects obvious page boilerplate, and uses
    the exact sentence as both quote and statement. No paraphrase or causal
    inference is introduced.
    """
    raw = str(source_text or "").strip()
    if not raw:
        return []

    blocks = [
        normalize_text(item)
        for item in re.split(
            r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9])",
            raw,
        )
        if normalize_text(item)
    ]
    ranked: list[tuple[int, int, str]] = []
    for index, sentence in enumerate(blocks):
        if len(sentence) < 40 or len(sentence) > 360:
            continue
        if _BOILERPLATE_RE.search(sentence):
            continue
        if _PAGE_METADATA_RE.search(sentence):
            continue
        if _is_structural_non_claim(sentence):
            continue
        if sentence.count("http") or sentence.count("|") > 2:
            continue
        if not _FACT_SIGNAL_RE.search(sentence):
            continue
        future = bool(_FUTURE_CUE_RE.search(sentence))
        score = 2 if future else 1
        if "%" in sentence or "$" in sentence:
            score += 1
        if re.search(r"\b20\d{2}\b", sentence):
            score += 1
        ranked.append((score, index, sentence))

    selected: list[ClaimCandidate] = []
    seen: set[str] = set()
    for _score, _index, sentence in sorted(
        ranked,
        key=lambda item: (-item[0], item[1]),
    ):
        normalized = sentence.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        is_future = bool(_FUTURE_CUE_RE.search(sentence))
        selected.append(
            ClaimCandidate(
                quote=sentence,
                statement=sentence,
                proposed_claim_key="",
                kind=(
                    ClaimKind.FORECAST
                    if is_future
                    else ClaimKind.FACT
                ),
                observation_kind=(
                    ObservationKind.FORECAST
                    if is_future
                    else ObservationKind.ACTUAL
                ),
                confidence=0.42,
                testable=True,
                time_sensitive=is_future,
                freshness_seconds=(604800 if is_future else None),
                supersedes_previous=False,
                metadata={
                    "extractor": "deterministic_grounded_fallback_v1",
                    "fallback_reason": fallback_reason,
                    "verbatim_statement": True,
                },
            )
        )
        if len(selected) >= max(1, int(max_candidates)):
            break
    return selected


class GroundedClaimExtractor:
    """LLM-based claim parser with exact-quote admission validation."""

    def __init__(
        self,
        ai_client,
        *,
        timeout_seconds: int = 45,
        max_source_chars: int = 7000,
    ) -> None:
        self.ai = ai_client
        self.timeout_seconds = max(15, int(timeout_seconds))
        self.max_source_chars = max(1500, int(max_source_chars))

    async def extract(
        self,
        *,
        document: ResearchDocument,
        topic_hint: str,
        max_candidates: int,
    ) -> list[ClaimCandidate]:
        source_text = _relevant_excerpt(
            document.text,
            signals=(topic_hint, document.title),
            max_chars=self.max_source_chars,
        )
        system = (
            "You extract testable research claims from source text. "
            "Use ONLY the supplied source. Return strict JSON. "
            "Every claim must include a short exact contiguous quote copied "
            "verbatim from source_text. Do not convert rhetoric, navigation, "
            "advertising, or vague opinion into factual claims."
        )
        user = json.dumps(
            {
                "topic_hint": topic_hint,
                "source_url": document.url,
                "source_title": document.title,
                "instructions": {
                    "max_claims": max_candidates,
                    "claim_key": (
                        "stable lowercase ASCII semantic key such as "
                        "entity.metric.period_or_scope; reuse the same key for "
                        "revisions of the same underlying proposition"
                    ),
                    "allowed_kind": sorted(_ALLOWED_KINDS),
                    "allowed_observation_kind": sorted(_ALLOWED_OBSERVATIONS),
                    "schema": {
                        "claims": [
                            {
                                "quote": "exact source quote",
                                "statement": "one concise testable proposition",
                                "claim_key": "stable.semantic.key",
                                "entity_name": "exact entity name from quote or null",
                                "entity_type": "company|organization|person|sector|product|asset|country|location|unknown",
                                "kind": "fact|interpretation|hypothesis|forecast|mechanism|scenario|conclusion",
                                "observation_kind": (
                                    "actual|forecast|revision|estimate|guidance|"
                                    "policy_statement|market_pricing|opinion|historical"
                                ),
                                "confidence": "0..1 extraction confidence",
                                "testable": True,
                                "valid_from": "ISO-8601 or null",
                                "valid_until": "ISO-8601 or null",
                                "time_sensitive": False,
                                "freshness_seconds": "integer or null",
                                "supersedes_previous": False,
                            }
                        ]
                    },
                    "rules": [
                        "Return no claim unless the exact quote supports the full statement.",
                        "Claims must be materially useful for research, not trivial metadata.",
                        "Do not invent entities, dates, causality, numbers, or scope.",
                        "Set entity_name only when that entity name appears explicitly in the quote.",
                        "Use an entity-scoped claim_key when the source clearly identifies the subject.",
                        "Set supersedes_previous only for an explicit revision/update/restatement.",
                        "For interpretation/opinion, phrase the claim as an attributable testable proposition when possible.",
                    ],
                },
                "source_text": source_text,
            },
            ensure_ascii=False,
        )
        try:
            raw = await asyncio.wait_for(
                self.ai.chat_multi(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0,
                    max_tokens=1100,
                ),
                timeout=self.timeout_seconds,
            )
        except Exception as exc:
            error_name = type(exc).__name__
            if error_name not in {
                "TimeoutError",
                "APITimeoutError",
                "RateLimitError",
                "APIConnectionError",
                "InternalServerError",
            }:
                raise
            fallback = _deterministic_grounded_candidates(
                source_text,
                max_candidates=max_candidates,
                fallback_reason=error_name,
            )
            logger.warning(
                "Claim extraction degraded to deterministic grounded fallback "
                "url=%s error=%s candidates=%s",
                document.url,
                error_name,
                len(fallback),
            )
            return fallback

        parsed = _json_value(raw)
        rows = parsed.get("claims") if isinstance(parsed, dict) else []
        if not isinstance(rows, list):
            rows = []

        normalized_source = normalize_text(source_text).casefold()
        candidates: list[ClaimCandidate] = []
        for row in rows[: max(1, int(max_candidates))]:
            if not isinstance(row, dict):
                continue
            quote = normalize_text(str(row.get("quote") or ""))
            statement = normalize_text(str(row.get("statement") or ""))
            if len(quote) < 12 or len(statement) < 12:
                continue
            if quote.casefold() not in normalized_source:
                logger.warning(
                    "Rejected ungrounded claim candidate url=%s",
                    document.url,
                )
                continue
            kind = _ALLOWED_KINDS.get(
                str(row.get("kind") or "fact").lower(),
                ClaimKind.FACT,
            )
            observation = _ALLOWED_OBSERVATIONS.get(
                str(row.get("observation_kind") or "actual").lower(),
                ObservationKind.ACTUAL,
            )
            try:
                confidence = max(
                    0.0,
                    min(0.9, float(row.get("confidence", 0.6))),
                )
            except (TypeError, ValueError):
                confidence = 0.6
            freshness = row.get("freshness_seconds")
            try:
                freshness_seconds = (
                    max(300, min(2_678_400, int(freshness)))
                    if freshness not in (None, "")
                    else None
                )
            except (TypeError, ValueError):
                freshness_seconds = None

            entity_name = normalize_text(
                str(row.get("entity_name") or "")
            )
            entity_type = normalize_text(
                str(row.get("entity_type") or "unknown")
            ).lower()
            grounded_entity = (
                entity_name
                if entity_name
                and entity_name.casefold() in quote.casefold()
                else ""
            )

            candidates.append(
                ClaimCandidate(
                    quote=quote,
                    statement=statement,
                    proposed_claim_key=str(row.get("claim_key") or ""),
                    kind=kind,
                    observation_kind=observation,
                    confidence=confidence,
                    testable=bool(row.get("testable", True)),
                    valid_from=_parse_time(row.get("valid_from")),
                    valid_until=_parse_time(row.get("valid_until")),
                    time_sensitive=bool(row.get("time_sensitive", False)),
                    freshness_seconds=freshness_seconds,
                    supersedes_previous=bool(
                        row.get("supersedes_previous", False)
                    ),
                    metadata={
                        "extractor": "grounded_claim_v1",
                        **(
                            {
                                "entity_name": grounded_entity,
                                "entity_type": entity_type,
                            }
                            if grounded_entity
                            else {}
                        ),
                    },
                )
            )
        if candidates:
            return candidates

        fallback = _deterministic_grounded_candidates(
            source_text,
            max_candidates=max_candidates,
            fallback_reason="empty_or_unusable_llm_output",
        )
        if fallback:
            logger.info(
                "Claim extraction used deterministic grounded fallback "
                "after empty/unusable LLM output url=%s candidates=%s",
                document.url,
                len(fallback),
            )
        return fallback


class GeneralClaimAcquisition:
    def __init__(
        self,
        *,
        graph: ClaimGraph,
        ledger,
        falsification: FalsificationEngine,
        extractor: ClaimCandidateExtractor,
        max_claims_per_document: int = 3,
        resolver: ConservativeClaimResolver | None = None,
    ) -> None:
        self.graph = graph
        self.ledger = ledger
        self.falsification = falsification
        self.extractor = extractor
        self.resolver = resolver or ConservativeClaimResolver()
        self.max_claims_per_document = max(
            1,
            int(max_claims_per_document),
        )

    def _ensure_rules(
        self,
        claim: ClaimNode,
        candidate: ClaimCandidate,
        *,
        source_tier: SourceTier,
    ) -> None:
        existing = {
            rule.rule_type
            for rule in self.falsification.rules_for_claim(claim.claim_id)
        }
        if FalsificationRuleType.CONTRADICTORY_EVIDENCE not in existing:
            self.falsification.register_rule(
                build_falsification_rule(
                    claim_id=claim.claim_id,
                    description=(
                        "Independent evidence that directly contradicts this "
                        "claim must reduce confidence or contest the claim."
                    ),
                    rule_type=FalsificationRuleType.CONTRADICTORY_EVIDENCE,
                    hard_fail=False,
                    weight=1.0,
                    min_sources=1,
                    metadata={"generated_by": "general_claim_acquisition"},
                ),
                graph=self.graph,
            )

        if FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT not in existing:
            required_sources = (
                1
                if source_tier
                in {SourceTier.PRIMARY, SourceTier.OFFICIAL_PRIMARY}
                else 2
            )
            self.falsification.register_rule(
                build_falsification_rule(
                    claim_id=claim.claim_id,
                    description=(
                        f"This claim requires at least {required_sources} "
                        "independent source family/families."
                    ),
                    rule_type=FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT,
                    hard_fail=False,
                    weight=0.8,
                    evidence_claim_key=claim.claim_key,
                    min_sources=required_sources,
                    metadata={"generated_by": "general_claim_acquisition"},
                ),
                graph=self.graph,
            )

        if (
            candidate.time_sensitive
            and FalsificationRuleType.FRESHNESS_FAILURE not in existing
        ):
            max_age = candidate.freshness_seconds or 604_800
            self.falsification.register_rule(
                build_falsification_rule(
                    claim_id=claim.claim_id,
                    description=(
                        "This time-sensitive claim requires evidence no older "
                        f"than {max_age} seconds."
                    ),
                    rule_type=FalsificationRuleType.FRESHNESS_FAILURE,
                    hard_fail=False,
                    weight=0.8,
                    evidence_claim_key=claim.claim_key,
                    max_age_seconds=max_age,
                    metadata={"generated_by": "general_claim_acquisition"},
                ),
                graph=self.graph,
            )

    async def run(
        self,
        *,
        db: Session,
        documents: list[ResearchDocument],
        seed_topic: str = "",
        tool_calls: int = 0,
        evaluated_at: datetime | None = None,
    ) -> ClaimAcquisitionResult:
        started = utc(evaluated_at)
        run_id = _stable_id(
            "acq",
            started.isoformat(),
            seed_topic,
            "|".join(document.url for document in documents),
        )

        stale_before = started - timedelta(minutes=10)
        stale_runs = (
            db.query(ResearchAcquisitionRunRecord)
            .filter(
                ResearchAcquisitionRunRecord.status == "running",
                ResearchAcquisitionRunRecord.started_at < stale_before,
            )
            .all()
        )
        for stale in stale_runs:
            stale.status = "abandoned"
            stale.completed_at = started
            stale.error = "stale_acquisition_recovered"
        if stale_runs:
            db.commit()

        run = ResearchAcquisitionRunRecord(
            run_id=run_id,
            started_at=started,
            status="running",
            seed_topic=normalize_text(seed_topic),
            documents_seen=len(documents),
            tool_calls=max(0, int(tool_calls)),
            meta={"version": "general-claim-acquisition-v1"},
        )
        db.add(run)
        db.commit()

        audits: list[dict[str, Any]] = []
        resolution_audits: list[dict[str, Any]] = []
        accepted_ids: set[str] = set()
        seen_source_fingerprints: set[tuple[str, str]] = set()
        extracted = accepted = duplicates = rejected = superseded = 0
        errors: list[str] = []

        try:
            for document in documents:
                tier = _source_tier(document)
                host = urlsplit(document.url).hostname or document.tool_name
                source = build_source(
                    url=document.url,
                    content=document.text,
                    publisher=document.publisher or host,
                    title=document.title,
                    source_tier=tier,
                    source_family=host,
                    independence_key=host,
                    published_at=document.published_at,
                    retrieved_at=datetime.now(timezone.utc),
                    observed_at=datetime.now(timezone.utc),
                    tool_name=document.tool_name,
                    metadata={
                        **document.metadata,
                        "claim_acquisition_run_id": run_id,
                    },
                )
                self.ledger.register_source(source)

                try:
                    candidates = await self.extractor.extract(
                        document=document,
                        topic_hint=seed_topic,
                        max_candidates=self.max_claims_per_document,
                    )
                except Exception as exc:
                    errors.append(
                        f"{document.url}:{type(exc).__name__}"
                    )
                    continue

                for candidate in candidates:
                    extracted += 1
                    fingerprint = _candidate_fingerprint(candidate)
                    source_fingerprint = (source.source_id, fingerprint)
                    if source_fingerprint in seen_source_fingerprints:
                        duplicates += 1
                        audits.append(
                            {
                                "candidate_id": _stable_id(
                                    "cand",
                                    run_id,
                                    source.source_id,
                                    fingerprint,
                                    "same_source_duplicate",
                                ),
                                "source_id": source.source_id,
                                "fingerprint": fingerprint,
                                "quote": candidate.quote,
                                "statement": candidate.statement,
                                "claim_key": _safe_claim_key(
                                    candidate.proposed_claim_key,
                                    candidate.statement,
                                ),
                                "kind": candidate.kind.value,
                                "observation_kind": candidate.observation_kind.value,
                                "confidence": candidate.confidence,
                                "valid_from": candidate.valid_from,
                                "valid_until": candidate.valid_until,
                                "supersedes_previous": candidate.supersedes_previous,
                                "decision": "duplicate",
                                "reason": "same_source_candidate_duplicate",
                                "accepted_claim_id": None,
                                "meta": {
                                    "source_url": document.url,
                                    "time_sensitive": candidate.time_sensitive,
                                    "freshness_seconds": candidate.freshness_seconds,
                                },
                            }
                        )
                        continue
                    seen_source_fingerprints.add(source_fingerprint)

                    claim_key = _safe_claim_key(
                        candidate.proposed_claim_key,
                        candidate.statement,
                    )
                    decision = "accepted"
                    reason = ""
                    claim: ClaimNode | None = None

                    normalized_quote = normalize_text(candidate.quote)
                    normalized_doc = normalize_text(document.text)
                    normalized_statement = normalize_text(candidate.statement)
                    if normalized_quote.casefold() not in normalized_doc.casefold():
                        decision = "rejected"
                        reason = "ungrounded_quote"
                    elif (
                        _PAGE_METADATA_RE.search(normalized_quote)
                        or _PAGE_METADATA_RE.search(normalized_statement)
                    ):
                        decision = "rejected"
                        reason = "page_metadata"
                    elif (
                        _is_structural_non_claim(normalized_quote)
                        or _is_structural_non_claim(normalized_statement)
                    ):
                        decision = "rejected"
                        reason = "structural_non_claim"
                    elif not candidate.testable:
                        decision = "rejected"
                        reason = "not_testable"
                    elif len(normalized_statement) < 12:
                        decision = "rejected"
                        reason = "statement_too_short"
                    elif normalized_statement.endswith("?"):
                        decision = "rejected"
                        reason = "question_not_claim"

                    candidate_id = _stable_id(
                        "cand",
                        run_id,
                        source.source_id,
                        fingerprint,
                    )
                    resolution = None
                    if decision == "rejected":
                        rejected += 1
                    else:
                        candidate_entity = self.resolver.entity_identity.resolve(
                            claim_key=claim_key,
                            statement=candidate.statement,
                            metadata=candidate.metadata,
                            quote=candidate.quote,
                        )
                        resolution = self.resolver.resolve(
                            statement=candidate.statement,
                            claim_key=claim_key,
                            graph=self.graph,
                            supersedes_previous=candidate.supersedes_previous,
                            revision_explicit=bool(
                                _REVISION_CUE_RE.search(candidate.quote)
                            ),
                            metadata=candidate.metadata,
                            quote=candidate.quote,
                        )
                        matched_claim = (
                            self.graph.get_claim(resolution.matched_claim_id)
                            if resolution.matched_claim_id
                            else None
                        )
                        if resolution.relation in {
                            SemanticClaimRelation.EXACT,
                            SemanticClaimRelation.PARAPHRASE,
                        } and matched_claim is not None:
                            claim = matched_claim
                            decision = "duplicate"
                            reason = (
                                "semantic_"
                                + resolution.relation.value
                                + "_existing_claim:"
                                + matched_claim.claim_id
                            )
                            duplicates += 1
                        else:
                            supersedes_id = None
                            supersede_denied = False
                            if (
                                resolution.relation
                                is SemanticClaimRelation.REVISION
                                and matched_claim is not None
                            ):
                                supersedes_id = matched_claim.claim_id
                                decision = "superseded"
                                reason = (
                                    "semantic_revision_supersedes:"
                                    + matched_claim.claim_id
                                )
                                superseded += 1
                            elif candidate.supersedes_previous:
                                supersede_denied = True

                            claim = build_claim(
                                claim_key=claim_key,
                                statement=candidate.statement,
                                kind=candidate.kind,
                                prior_confidence=max(
                                    0.25,
                                    min(0.65, candidate.confidence),
                                ),
                                created_at=started,
                                valid_from=(
                                    candidate.valid_from
                                    or document.published_at
                                ),
                                valid_until=candidate.valid_until,
                                supersedes=supersedes_id,
                                metadata={
                                    "acquired_automatically": True,
                                    "acquisition_run_id": run_id,
                                    "source_id": source.source_id,
                                    "source_url": document.url,
                                    "supersede_denied": supersede_denied,
                                    "semantic_resolution": resolution.relation.value,
                                    "semantic_resolution_score": resolution.score,
                                    "semantic_match_claim_id": resolution.matched_claim_id,
                                    **candidate.metadata,
                                    **(
                                        candidate_entity.to_metadata()
                                        if candidate_entity is not None
                                        else {}
                                    ),
                                },
                            )
                            self.graph.register_claim(claim)
                            accepted += 1

                            if (
                                resolution.relation
                                is SemanticClaimRelation.CONTRADICTION
                                and matched_claim is not None
                            ):
                                edge_meta = {
                                    "generated_by": "semantic_claim_resolution",
                                    "resolution_score": resolution.score,
                                    "resolution_reason": resolution.reason,
                                }
                                self.graph.connect(
                                    matched_claim.claim_id,
                                    claim.claim_id,
                                    ClaimRelation.CONTRADICTS,
                                    weight=min(0.9, max(0.55, resolution.score)),
                                    created_at=started,
                                    metadata=edge_meta,
                                )
                                accepted_ids.add(matched_claim.claim_id)

                        resolution_audits.append(
                            {
                                "resolution_id": _stable_id(
                                    "res",
                                    candidate_id,
                                    resolution.relation.value,
                                    resolution.matched_claim_id or "",
                                ),
                                "candidate_id": candidate_id,
                                "matched_claim_id": resolution.matched_claim_id,
                                "relation": resolution.relation.value,
                                "score": resolution.score,
                                "lexical_score": resolution.lexical_score,
                                "key_match": resolution.key_match,
                                "numeric_match": resolution.numeric_match,
                                "period_match": resolution.period_match,
                                "polarity_match": resolution.polarity_match,
                                "reason": resolution.reason,
                                "signals": resolution.signals,
                            }
                        )

                        if claim is not None:
                            evidence = build_evidence(
                                claim_key=claim.claim_key,
                                source=source,
                                statement=candidate.quote,
                                relation=EvidenceRelation.SUPPORTS,
                                observation_kind=candidate.observation_kind,
                                event_time=(
                                    candidate.valid_from
                                    or document.published_at
                                ),
                                observed_at=datetime.now(timezone.utc),
                                recorded_at=datetime.now(timezone.utc),
                                confidence=max(
                                    0.2,
                                    min(0.85, candidate.confidence),
                                ),
                                metadata={
                                    "quote_grounded": True,
                                    "claim_acquisition_run_id": run_id,
                                    "candidate_fingerprint": fingerprint,
                                },
                            )
                            self.ledger.append(evidence)
                            self.graph.link_evidence(
                                claim_id=claim.claim_id,
                                evidence_id=evidence.evidence_id,
                                ledger=self.ledger,
                                relation=EvidenceRelation.SUPPORTS,
                                weight=max(
                                    0.2,
                                    min(0.85, candidate.confidence),
                                ),
                                metadata={
                                    "claim_acquisition": True,
                                    "candidate_fingerprint": fingerprint,
                                },
                            )
                            if (
                                resolution is not None
                                and resolution.relation
                                is SemanticClaimRelation.CONTRADICTION
                                and resolution.matched_claim_id
                            ):
                                self.graph.link_evidence(
                                    claim_id=resolution.matched_claim_id,
                                    evidence_id=evidence.evidence_id,
                                    ledger=self.ledger,
                                    relation=EvidenceRelation.CONTRADICTS,
                                    weight=max(
                                        0.2,
                                        min(0.85, candidate.confidence),
                                    ),
                                    metadata={
                                        "claim_acquisition": True,
                                        "semantic_contradiction": True,
                                        "candidate_fingerprint": fingerprint,
                                    },
                                )
                            self._ensure_rules(
                                claim,
                                candidate,
                                source_tier=tier,
                            )
                            accepted_ids.add(claim.claim_id)

                    audits.append(
                        {
                            "candidate_id": candidate_id,
                            "source_id": source.source_id,
                            "fingerprint": fingerprint,
                            "quote": candidate.quote,
                            "statement": candidate.statement,
                            "claim_key": claim_key,
                            "kind": candidate.kind.value,
                            "observation_kind": (
                                candidate.observation_kind.value
                            ),
                            "confidence": candidate.confidence,
                            "valid_from": candidate.valid_from,
                            "valid_until": candidate.valid_until,
                            "supersedes_previous": (
                                candidate.supersedes_previous
                            ),
                            "decision": decision,
                            "reason": reason,
                            "accepted_claim_id": (
                                claim.claim_id if claim is not None else None
                            ),
                            "meta": {
                                "source_url": document.url,
                                "time_sensitive": candidate.time_sensitive,
                                "freshness_seconds": (
                                    candidate.freshness_seconds
                                ),
                                "semantic_resolution": (
                                    resolution.relation.value
                                    if resolution is not None
                                    else None
                                ),
                                "semantic_resolution_score": (
                                    resolution.score
                                    if resolution is not None
                                    else None
                                ),
                                "semantic_match_claim_id": (
                                    resolution.matched_claim_id
                                    if resolution is not None
                                    else None
                                ),
                                "entity_id": (
                                    candidate_entity.canonical_id
                                    if decision != "rejected"
                                    and candidate_entity is not None
                                    else None
                                ),
                                "entity_identity_source": (
                                    candidate_entity.source
                                    if decision != "rejected"
                                    and candidate_entity is not None
                                    else None
                                ),
                            },
                        }
                    )

            persist_ledger(db, self.ledger)
            persist_claim_graph(db, self.graph)
            persist_falsification_engine(db, self.falsification)

            for audit in audits:
                existing_row = db.get(
                    ResearchClaimCandidateRecord,
                    audit["candidate_id"],
                )
                if existing_row is not None:
                    continue
                db.add(
                    ResearchClaimCandidateRecord(
                        candidate_id=audit["candidate_id"],
                        run_id=run_id,
                        source_id=audit["source_id"],
                        fingerprint=audit["fingerprint"],
                        quote=audit["quote"],
                        statement=audit["statement"],
                        proposed_claim_key=audit["claim_key"],
                        kind=audit["kind"],
                        observation_kind=audit["observation_kind"],
                        confidence=audit["confidence"],
                        valid_from=audit["valid_from"],
                        valid_until=audit["valid_until"],
                        supersedes_previous=audit[
                            "supersedes_previous"
                        ],
                        decision=audit["decision"],
                        reason=audit["reason"],
                        accepted_claim_id=audit["accepted_claim_id"],
                        meta=audit["meta"],
                    )
                )

            db.flush()
            for resolution in resolution_audits:
                if db.get(
                    ResearchClaimResolutionRecord,
                    resolution["resolution_id"],
                ) is not None:
                    continue
                db.add(
                    ResearchClaimResolutionRecord(
                        resolution_id=resolution["resolution_id"],
                        candidate_id=resolution["candidate_id"],
                        matched_claim_id=resolution["matched_claim_id"],
                        relation=resolution["relation"],
                        score=resolution["score"],
                        lexical_score=resolution["lexical_score"],
                        key_match=resolution["key_match"],
                        numeric_match=resolution["numeric_match"],
                        period_match=resolution["period_match"],
                        polarity_match=resolution["polarity_match"],
                        reason=resolution["reason"],
                        signals=resolution["signals"],
                    )
                )

            belief_changes = 0
            if accepted_ids:
                cycle = PanWatchBeliefMonitor(
                    graph=self.graph,
                    ledger=self.ledger,
                    falsification=self.falsification,
                ).run_cycle(
                    db=db,
                    claim_ids=tuple(sorted(accepted_ids)),
                    evaluated_at=datetime.now(timezone.utc),
                    metadata={
                        "claim_acquisition_run_id": run_id,
                    },
                )
                belief_changes = cycle.changed_count

            run.status = "success"
            run.completed_at = datetime.now(timezone.utc)
            run.candidates_extracted = extracted
            run.claims_accepted = accepted
            run.duplicates = duplicates
            run.rejected = rejected
            run.superseded = superseded
            run.error = ";".join(errors)[:4000]
            db.commit()

            return ClaimAcquisitionResult(
                run_id=run_id,
                status="success",
                seed_topic=seed_topic,
                documents_seen=len(documents),
                candidates_extracted=extracted,
                claims_accepted=accepted,
                duplicates=duplicates,
                rejected=rejected,
                superseded=superseded,
                tool_calls=max(0, int(tool_calls)),
                belief_changes=belief_changes,
                accepted_claim_ids=tuple(sorted(accepted_ids)),
                errors=tuple(errors),
            )
        except Exception as exc:
            run.status = "error"
            run.completed_at = datetime.now(timezone.utc)
            run.candidates_extracted = extracted
            run.claims_accepted = accepted
            run.duplicates = duplicates
            run.rejected = rejected
            run.superseded = superseded
            run.error = type(exc).__name__
            db.commit()
            raise


def _configured_topics(settings: Settings) -> list[str]:
    raw = str(settings.claim_acquisition_topics or "")
    rows = [
        normalize_text(item)
        for item in re.split(r"[\n;]+", raw)
        if normalize_text(item)
    ]
    return rows[: settings.claim_acquisition_max_topics]


def _derived_topics(graph: ClaimGraph, *, limit: int) -> list[str]:
    rows = sorted(
        _active_claims(graph),
        key=lambda item: (item.created_at, item.claim_id),
        reverse=True,
    )
    topics: list[str] = []
    seen: set[str] = set()
    for claim in rows:
        statement = normalize_text(claim.statement)
        key = statement.casefold()
        if not statement or key in seen:
            continue
        seen.add(key)
        topics.append(statement)
        if len(topics) >= limit:
            break
    return topics


async def acquire_topic_once(
    settings: Settings | None = None,
    *,
    topic: str | None = None,
) -> list[ClaimAcquisitionResult]:
    """Search one or more bounded topics and admit grounded claims."""
    settings = settings or Settings()
    if not settings.ahmed_toolbox_url:
        raise RuntimeError("Ahmed ToolBox URL is not configured")
    if not settings.ai_api_key:
        raise RuntimeError("AI API key is not configured")

    with research_session() as db:
        ledger = load_full_ledger(db)
        graph = load_claim_graph(db, ledger=ledger)
        falsification = load_falsification_engine(
            db,
            graph=graph,
        )

        topics = [normalize_text(topic)] if topic else _configured_topics(settings)
        if (
            not topics
            and settings.claim_acquisition_bootstrap_from_active_claims
        ):
            topics = _derived_topics(
                graph,
                limit=settings.claim_acquisition_max_topics,
            )
        topics = [item for item in topics if item][
            : settings.claim_acquisition_max_topics
        ]
        if not topics:
            return []

        gateway = AhmedToolboxResearchGateway(
            settings,
            max_tool_calls=settings.claim_acquisition_max_tool_calls,
            tool_timeout_seconds=settings.auto_research_tool_timeout_seconds,
        )
        ai = build_failover_client(
            None,
            None,
            settings=settings,
            max_fallbacks=2,
        )
        extractor = GroundedClaimExtractor(
            ai,
            timeout_seconds=(
                settings.claim_acquisition_extraction_timeout_seconds
            ),
            max_source_chars=(
                settings.claim_acquisition_extraction_max_chars
            ),
        )
        engine = GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=extractor,
            max_claims_per_document=(
                settings.claim_acquisition_max_claims_per_document
            ),
        )

        results: list[ClaimAcquisitionResult] = []
        try:
            for seed in topics:
                before_calls = gateway.tool_calls
                try:
                    documents = await gateway.search(
                        seed,
                        max_sources=(
                            settings.claim_acquisition_max_documents
                        ),
                    )
                except Exception as exc:
                    now = datetime.now(timezone.utc)
                    run_id = _stable_id(
                        "acq",
                        now.isoformat(),
                        seed,
                        "search_error",
                    )
                    db.add(
                        ResearchAcquisitionRunRecord(
                            run_id=run_id,
                            started_at=now,
                            completed_at=now,
                            status="error",
                            seed_topic=seed,
                            tool_calls=(
                                gateway.tool_calls - before_calls
                            ),
                            error=type(exc).__name__,
                            meta={
                                "version": "general-claim-acquisition-v1",
                                "phase": "search",
                            },
                        )
                    )
                    db.commit()
                    logger.warning(
                        "Claim acquisition search failed topic=%s error=%s",
                        seed,
                        type(exc).__name__,
                    )
                    continue

                result = await engine.run(
                    db=db,
                    documents=documents,
                    seed_topic=seed,
                    tool_calls=gateway.tool_calls - before_calls,
                )
                results.append(result)
        finally:
            await gateway.close()
        return results


async def acquire_text_document(
    *,
    text: str,
    url: str,
    title: str = "",
    publisher: str = "",
    topic_hint: str = "",
    tool_name: str = "direct_text_ingest",
    published_at: datetime | None = None,
    settings: Settings | None = None,
) -> ClaimAcquisitionResult:
    """Acquire grounded claims from arbitrary supplied text/transcript/report."""
    settings = settings or Settings()
    if not settings.ai_api_key:
        raise RuntimeError("AI API key is not configured")
    clean_text = str(text or "").strip()
    if not clean_text:
        raise ValueError("text is required")
    clean_url = str(url or "").strip() or "https://panwatch.internal/direct-text"

    with research_session() as db:
        ledger = load_full_ledger(db)
        graph = load_claim_graph(db, ledger=ledger)
        falsification = load_falsification_engine(
            db,
            graph=graph,
        )
        ai = build_failover_client(
            None,
            None,
            settings=settings,
            max_fallbacks=2,
        )
        extractor = GroundedClaimExtractor(
            ai,
            timeout_seconds=(
                settings.claim_acquisition_extraction_timeout_seconds
            ),
            max_source_chars=(
                settings.claim_acquisition_extraction_max_chars
            ),
        )
        engine = GeneralClaimAcquisition(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            extractor=extractor,
            max_claims_per_document=(
                settings.claim_acquisition_max_claims_per_document
            ),
        )
        document = ResearchDocument(
            url=clean_url,
            title=title,
            text=clean_text,
            tool_name=tool_name,
            publisher=publisher,
            published_at=published_at,
            metadata={"direct_ingest": True},
        )
        return await engine.run(
            db=db,
            documents=[document],
            seed_topic=topic_hint,
            tool_calls=0,
        )
