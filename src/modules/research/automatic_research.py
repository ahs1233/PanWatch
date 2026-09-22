"""Bounded automatic counter-research loop for PanWatch.

Flow:
1. Evaluate active claims and collect research-worthy falsification probes.
2. Discover/search/read through the existing Ahmed ToolBox MCP gateway.
3. Use quote-grounded extraction: a finding is rejected unless its quote exists
   verbatim (after whitespace normalization) in the fetched source text.
4. Persist source provenance + evidence + evidence-to-claim links.
5. Re-evaluate the persistent belief state and record why it changed.

The loop is deliberately bounded and cooldown-backed to prevent runaway tool
invocations.
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
from src.platform.external_tools.ahmed_toolbox import (
    AhmedToolboxClient,
    AhmedToolboxError,
)
from src.platform.persistence.models import (
    ResearchLoopRunRecord,
    ResearchProbeAttemptRecord,
)
from src.platform.runtime.config import Settings

from .claim_graph import ClaimGraph
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
from .falsification import (
    FalsificationEngine,
    FalsificationProbe,
    FalsificationRule,
    FalsificationRuleType,
    FalsificationState,
)
from .ledger import EvidenceLedger
from .panwatch_monitor import PanWatchBeliefMonitor
from .reasoning_store import (
    load_claim_graph,
    load_falsification_engine,
    persist_claim_graph,
)
from .research_store import research_session

logger = logging.getLogger(__name__)

_URL_RE = re.compile(r"""https?://[^\s<>\]\)\}"']+""")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s\)]+)\)")
_JSON_FENCE_RE = re.compile(r"^\s*~~~(?:json)?\s*|\s*~~~\s*$", re.I)
_ALLOWED_KINDS = {item.value: item for item in ObservationKind}
_ALLOWED_RELATIONS = {item.value: item for item in EvidenceRelation}


@dataclass(frozen=True)
class ResearchDocument:
    url: str
    title: str
    text: str
    tool_name: str
    publisher: str = ""
    published_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedFinding:
    quote: str
    relation: EvidenceRelation
    observation_kind: ObservationKind
    confidence: float
    numeric_value: float | None = None
    unit: str = ""
    period: str = ""


@dataclass(frozen=True)
class AutomaticResearchResult:
    run_id: str
    status: str
    probes_planned: int
    probes_executed: int
    probes_skipped_cooldown: int
    tool_calls: int
    documents_read: int
    evidence_added: int
    beliefs_changed: int
    before_cycle_id: str | None
    after_cycle_id: str | None
    raw_evidence_added: int = 0
    classified_evidence_added: int = 0
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "probes_planned": self.probes_planned,
            "probes_executed": self.probes_executed,
            "probes_skipped_cooldown": self.probes_skipped_cooldown,
            "tool_calls": self.tool_calls,
            "documents_read": self.documents_read,
            "evidence_added": self.evidence_added,
            "beliefs_changed": self.beliefs_changed,
            "before_cycle_id": self.before_cycle_id,
            "after_cycle_id": self.after_cycle_id,
            "raw_evidence_added": self.raw_evidence_added,
            "classified_evidence_added": self.classified_evidence_added,
            "errors": list(self.errors),
        }


class EvidenceExtractor(Protocol):
    async def extract(
        self,
        *,
        probe: FalsificationProbe,
        rule: FalsificationRule | None,
        claim_statement: str,
        evidence_claim_key: str,
        desired_relation: EvidenceRelation,
        document: ResearchDocument,
        max_findings: int,
    ) -> list[ExtractedFinding]: ...


def _hash(prefix: str, *parts: str) -> str:
    raw = "\n".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:24]}"


def _text_from_tool_result(result: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in result.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            value = str(block.get("text") or "").strip()
            if value:
                parts.append(value)
    return "\n\n".join(parts)


def _parse_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return utc(parsed)


def _json_value(text: str) -> Any:
    value = _JSON_FENCE_RE.sub("", text.strip())
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = value.find(opener)
            end = value.rfind(closer)
            if start >= 0 and end > start:
                try:
                    return json.loads(value[start : end + 1])
                except json.JSONDecodeError:
                    pass
    return None


def _candidate_records(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(_candidate_records(item))
        return rows
    if not isinstance(value, dict):
        return rows

    url = value.get("url") or value.get("link") or value.get("href")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        rows.append(value)

    for key in (
        "results",
        "items",
        "data",
        "sources",
        "documents",
        "content",
        "organic",
    ):
        nested = value.get(key)
        if isinstance(nested, (list, dict)):
            rows.extend(_candidate_records(nested))
    return rows


def _documents_from_search_text(text: str, *, tool_name: str) -> list[ResearchDocument]:
    docs: list[ResearchDocument] = []
    parsed = _json_value(text)
    for row in _candidate_records(parsed):
        url = str(row.get("url") or row.get("link") or row.get("href") or "").strip()
        if not url:
            continue
        title = str(row.get("title") or row.get("name") or "").strip()
        snippet = str(
            row.get("snippet")
            or row.get("summary")
            or row.get("text")
            or row.get("description")
            or row.get("content")
            or ""
        ).strip()
        docs.append(
            ResearchDocument(
                url=url,
                title=title,
                text=snippet,
                tool_name=tool_name,
                publisher=str(row.get("publisher") or row.get("source") or "").strip(),
                published_at=_parse_datetime(
                    row.get("published_at")
                    or row.get("publishedDate")
                    or row.get("date")
                ),
                metadata={"search_record": True},
            )
        )

    if not docs:
        for title, url in _MD_LINK_RE.findall(text):
            docs.append(
                ResearchDocument(
                    url=url.rstrip(".,;"),
                    title=title.strip(),
                    text="",
                    tool_name=tool_name,
                )
            )
    if not docs:
        seen: set[str] = set()
        for url in _URL_RE.findall(text):
            cleaned = url.rstrip(".,;")
            if cleaned in seen:
                continue
            seen.add(cleaned)
            docs.append(
                ResearchDocument(
                    url=cleaned,
                    title="",
                    text="",
                    tool_name=tool_name,
                )
            )
    return docs


class AhmedToolboxResearchGateway:
    """Dynamic read-only adapter over the already configured Ahmed ToolBox."""

    def __init__(self, settings: Settings, *, max_tool_calls: int = 12) -> None:
        self.settings = settings
        self.max_tool_calls = max(1, int(max_tool_calls))
        self.tool_calls = 0
        self.client = AhmedToolboxClient(
            settings.ahmed_toolbox_url,
            token=settings.ahmed_toolbox_token,
            timeout_seconds=max(
                10.0,
                float(settings.ahmed_toolbox_timeout_seconds),
            ),
        )
        self._tools: dict[str, dict[str, Any]] | None = None

    async def close(self) -> None:
        await asyncio.to_thread(self.client.close)

    async def _discover(self) -> dict[str, dict[str, Any]]:
        if self._tools is None:
            tools = await asyncio.wait_for(
                asyncio.to_thread(self.client.list_tools),
                timeout=25,
            )
            self._tools = {
                str(item.get("name") or ""): item
                for item in tools
                if isinstance(item, dict) and item.get("name")
            }
        return self._tools

    def _budget(self) -> None:
        if self.tool_calls >= self.max_tool_calls:
            raise RuntimeError("automatic_research_tool_call_budget_exhausted")
        self.tool_calls += 1

    async def _call(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._budget()
        result = await asyncio.wait_for(
            asyncio.to_thread(self.client.call_tool, name, arguments),
            timeout=45,
        )
        if result.get("isError"):
            raise AhmedToolboxError(
                _text_from_tool_result(result)[:500]
                or f"{name} returned isError"
            )
        return result

    @staticmethod
    def _schema_args(
        tool: dict[str, Any],
        *,
        query: str | None = None,
        url: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        schema = tool.get("inputSchema") or {}
        properties = schema.get("properties") or {}
        args: dict[str, Any] = {}

        if query is not None:
            for key in ("query", "q", "search", "text"):
                if key in properties:
                    args[key] = query
                    break
            else:
                args["query"] = query

        if url is not None:
            for key in ("url", "uri", "link"):
                if key in properties:
                    args[key] = url
                    break
            else:
                args["url"] = url

        if limit is not None:
            for key in (
                "num_results",
                "numResults",
                "limit",
                "max_results",
                "count",
            ):
                if key in properties:
                    args[key] = int(limit)
                    break
        return args

    async def search(
        self,
        query: str,
        *,
        max_sources: int,
    ) -> list[ResearchDocument]:
        tools = await self._discover()
        search_name = next(
            (
                name
                for name in (
                    "reach_web_search",
                    "web_search_exa",
                    "search_web",
                )
                if name in tools
            ),
            None,
        )
        if search_name is None:
            search_name = next(
                (
                    name
                    for name in tools
                    if "web_search" in name or name.endswith("_search")
                ),
                None,
            )
        if search_name is None:
            raise RuntimeError("no_web_search_tool_available")

        result = await self._call(
            search_name,
            self._schema_args(
                tools[search_name],
                query=query,
                limit=max_sources,
            ),
        )
        search_text = _text_from_tool_result(result)
        candidates = _documents_from_search_text(
            search_text,
            tool_name=search_name,
        )

        deduped: list[ResearchDocument] = []
        seen: set[str] = set()
        for item in candidates:
            if item.url in seen:
                continue
            seen.add(item.url)
            deduped.append(item)
            if len(deduped) >= max_sources:
                break

        read_name = next(
            (
                name
                for name in ("reach_read_url", "scrapling__fetch")
                if name in tools
            ),
            None,
        )
        if read_name is None:
            return [item for item in deduped if item.text]

        documents: list[ResearchDocument] = []
        for candidate in deduped:
            try:
                read_result = await self._call(
                    read_name,
                    self._schema_args(
                        tools[read_name],
                        url=candidate.url,
                    ),
                )
                body = _text_from_tool_result(read_result).strip()
            except Exception as exc:  # one source must not kill the probe
                logger.debug(
                    "automatic research read failed url=%s error=%s",
                    candidate.url,
                    type(exc).__name__,
                )
                body = candidate.text
            if not body:
                continue
            documents.append(
                ResearchDocument(
                    url=candidate.url,
                    title=candidate.title,
                    text=body[:50_000],
                    tool_name=read_name,
                    publisher=candidate.publisher,
                    published_at=candidate.published_at,
                    metadata={
                        **candidate.metadata,
                        "search_tool": search_name,
                    },
                )
            )
        return documents


def _relevant_excerpt(
    text: str,
    *,
    signals: tuple[str, ...],
    max_chars: int,
) -> str:
    """Select compact source-grounded passages most relevant to a probe."""
    raw = str(text or "").strip()
    if len(raw) <= max_chars:
        return raw

    keywords: set[str] = set()
    for signal in signals:
        for token in re.findall(r"[A-Za-z0-9%.$/-]{4,}", str(signal or "").lower()):
            if token not in {
                "current", "independent", "evidence", "target", "claim",
                "search", "could", "would", "should", "against", "materially",
            }:
                keywords.add(token)

    blocks = [
        block.strip()
        for block in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9])", raw)
        if block.strip()
    ]
    if not blocks:
        return raw[:max_chars]

    scored: list[tuple[int, int, str]] = []
    for index, block in enumerate(blocks):
        lower = block.lower()
        score = sum(1 for keyword in keywords if keyword in lower)
        scored.append((score, index, block))

    selected: list[tuple[int, str]] = []
    used = 0
    for score, index, block in sorted(
        scored,
        key=lambda item: (-item[0], item[1]),
    ):
        if score <= 0 and selected:
            break
        remaining = max_chars - used
        if remaining <= 0:
            break
        piece = block[:remaining]
        selected.append((index, piece))
        used += len(piece) + 2
    if not selected:
        return raw[:max_chars]

    selected.sort(key=lambda item: item[0])
    excerpt = "\n\n".join(block for _idx, block in selected)
    return excerpt[:max_chars]


class QuoteGroundedEvidenceExtractor:
    """LLM parser whose output is accepted only when quotes exist in-source."""

    def __init__(
        self,
        ai_client,
        *,
        timeout_seconds: int = 40,
        max_source_chars: int = 6000,
    ) -> None:
        self.ai = ai_client
        self.timeout_seconds = max(15, int(timeout_seconds))
        self.max_source_chars = max(1500, int(max_source_chars))

    async def extract(
        self,
        *,
        probe: FalsificationProbe,
        rule: FalsificationRule | None,
        claim_statement: str,
        evidence_claim_key: str,
        desired_relation: EvidenceRelation,
        document: ResearchDocument,
        max_findings: int,
    ) -> list[ExtractedFinding]:
        source_text = _relevant_excerpt(
            document.text,
            signals=(
                probe.instruction,
                claim_statement,
                evidence_claim_key,
                rule.description if rule else "",
            ),
            max_chars=self.max_source_chars,
        )
        system = (
            "You are a conservative evidence extraction parser. "
            "Use ONLY the supplied source text. Return strict JSON only. "
            "Never infer a fact that is not stated in the source. "
            "Every finding MUST contain a short exact quote copied from the source."
        )
        user = json.dumps(
            {
                "research_probe": probe.instruction,
                "target_claim": claim_statement,
                "evidence_claim_key": evidence_claim_key,
                "preferred_relation": desired_relation.value,
                "rule_type": rule.rule_type.value if rule else "",
                "source_url": document.url,
                "instructions": {
                    "max_findings": max_findings,
                    "schema": {
                        "findings": [
                            {
                                "quote": "exact contiguous quote from source",
                                "relation": "supports|contradicts|context",
                                "observation_kind": (
                                    "actual|forecast|revision|estimate|guidance|"
                                    "policy_statement|market_pricing|opinion|historical"
                                ),
                                "confidence": "0..1",
                                "numeric_value": "number or null",
                                "unit": "string",
                                "period": "string",
                            }
                        ]
                    },
                    "rules": [
                        "If the source does not materially answer the probe, return findings=[].",
                        "Do not fabricate dates, numbers, entities, or quotes.",
                        "Use context when the source is relevant but neither supports nor contradicts.",
                    ],
                },
                "source_text": source_text,
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            raw = await asyncio.wait_for(
                self.ai.chat_multi(
                    messages,
                    temperature=0,
                    max_tokens=500,
                ),
                timeout=self.timeout_seconds,
            )
        except TimeoutError:
            # One bounded retry with an ultra-compact excerpt. This keeps the
            # research loop alive when the provider is slow on longer pages,
            # without accepting ungrounded content or retrying indefinitely.
            compact_text = _relevant_excerpt(
                source_text,
                signals=(
                    probe.instruction,
                    claim_statement,
                    evidence_claim_key,
                ),
                max_chars=min(2200, self.max_source_chars),
            )
            compact_user = json.dumps(
                {
                    "research_probe": probe.instruction,
                    "target_claim": claim_statement,
                    "preferred_relation": desired_relation.value,
                    "source_url": document.url,
                    "instructions": {
                        "max_findings": min(1, max_findings),
                        "return": (
                            "JSON only: {\"findings\":[{\"quote\":"
                            "\"exact quote\",\"relation\":"
                            "\"supports|contradicts|context\","
                            "\"observation_kind\":\"actual|forecast|revision|"
                            "estimate|guidance|policy_statement|market_pricing|"
                            "opinion|historical\",\"confidence\":0.0}]}"
                        ),
                        "if_uncertain": {"findings": []},
                    },
                    "source_text": compact_text,
                },
                ensure_ascii=False,
            )
            raw = await asyncio.wait_for(
                self.ai.chat_multi(
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": compact_user},
                    ],
                    temperature=0,
                    max_tokens=260,
                ),
                timeout=max(15, self.timeout_seconds // 2),
            )
            source_text = compact_text
        parsed = _json_value(raw)
        rows = parsed.get("findings") if isinstance(parsed, dict) else []
        if not isinstance(rows, list):
            return []

        normalized_source = normalize_text(source_text).casefold()
        findings: list[ExtractedFinding] = []
        for row in rows[: max(1, int(max_findings))]:
            if not isinstance(row, dict):
                continue
            quote = normalize_text(str(row.get("quote") or ""))
            if len(quote) < 8:
                continue
            if normalize_text(quote).casefold() not in normalized_source:
                logger.warning(
                    "Rejected ungrounded research finding url=%s",
                    document.url,
                )
                continue

            relation_raw = str(row.get("relation") or "context").lower()
            relation = _ALLOWED_RELATIONS.get(
                relation_raw,
                EvidenceRelation.CONTEXT,
            )
            kind_raw = str(row.get("observation_kind") or "actual").lower()
            kind = _ALLOWED_KINDS.get(
                kind_raw,
                ObservationKind.ACTUAL,
            )
            try:
                confidence = max(
                    0.0,
                    min(0.85, float(row.get("confidence", 0.6))),
                )
            except (TypeError, ValueError):
                confidence = 0.6
            numeric = row.get("numeric_value")
            try:
                numeric_value = (
                    float(numeric)
                    if numeric is not None and numeric != ""
                    else None
                )
            except (TypeError, ValueError):
                numeric_value = None

            findings.append(
                ExtractedFinding(
                    quote=quote,
                    relation=relation,
                    observation_kind=kind,
                    confidence=confidence,
                    numeric_value=numeric_value,
                    unit=normalize_text(str(row.get("unit") or "")),
                    period=normalize_text(str(row.get("period") or "")),
                )
            )
        return findings


def _raw_source_quote(
    document: ResearchDocument,
    *,
    probe: FalsificationProbe,
    claim_statement: str,
    max_chars: int = 700,
) -> str:
    """Return one grounded raw excerpt without assigning semantic relation.

    The returned text is stored under raw.<claim_id> as RAW_SOURCE and is never
    linked into the Claim Graph until a semantic classifier succeeds later.
    """
    excerpt = _relevant_excerpt(
        document.text,
        signals=(probe.instruction, claim_statement),
        max_chars=max(250, int(max_chars)),
    )
    candidates = [
        normalize_text(item)
        for item in re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z0-9])", excerpt)
        if normalize_text(item)
    ]
    quote = next((item for item in candidates if len(item) >= 40), "")
    if not quote:
        quote = normalize_text(excerpt)[:max_chars]
    return quote if len(quote) >= 20 else ""

def _probe_key(probe: FalsificationProbe) -> str:
    return _hash(
        "probe",
        probe.rule_id,
        probe.claim_id,
        probe.instruction,
    )


def _active_claim_ids(graph: ClaimGraph) -> tuple[str, ...]:
    superseded = {
        claim.supersedes
        for claim in graph.claims
        if claim.supersedes
    }
    return tuple(
        sorted(
            claim.claim_id
            for claim in graph.claims
            if claim.claim_id not in superseded
        )
    )


def _research_probe_for_rule(
    rule: FalsificationRule,
    state: FalsificationState,
) -> FalsificationProbe | None:
    if rule.rule_type is FalsificationRuleType.CONTRADICTORY_EVIDENCE:
        if state is FalsificationState.TRIGGERED:
            return None
        return FalsificationProbe(
            rule_id=rule.rule_id,
            claim_id=rule.claim_id,
            priority=round(rule.weight * 0.8, 4),
            instruction=(
                "Actively search for current independent evidence that could "
                "contradict or materially weaken the target claim."
            ),
            missing_requirement="active adversarial verification",
        )
    if (
        rule.rule_type
        in {
            FalsificationRuleType.FRESHNESS_FAILURE,
            FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT,
        }
        and state is FalsificationState.TRIGGERED
    ):
        return FalsificationProbe(
            rule_id=rule.rule_id,
            claim_id=rule.claim_id,
            priority=round(rule.weight, 4),
            instruction=(
                "Find fresh independent evidence needed to test this rule: "
                + rule.description
            ),
            missing_requirement="triggered rule requires new research",
        )
    return None


class AutomaticResearchLoop:
    def __init__(
        self,
        *,
        graph: ClaimGraph,
        ledger: EvidenceLedger,
        falsification: FalsificationEngine,
        gateway: AhmedToolboxResearchGateway,
        extractor: EvidenceExtractor,
        max_probes: int = 3,
        max_sources_per_probe: int = 3,
        max_findings_per_document: int = 2,
        cooldown_minutes: int = 180,
    ) -> None:
        self.graph = graph
        self.ledger = ledger
        self.falsification = falsification
        self.gateway = gateway
        self.extractor = extractor
        self.max_probes = max(1, int(max_probes))
        self.max_sources_per_probe = max(1, int(max_sources_per_probe))
        self.max_findings_per_document = max(
            1,
            int(max_findings_per_document),
        )
        self.cooldown_minutes = max(1, int(cooldown_minutes))

    def _collect_probes(
        self,
        *,
        as_of: datetime,
    ) -> list[FalsificationProbe]:
        candidates: dict[str, FalsificationProbe] = {}
        for claim_id in _active_claim_ids(self.graph):
            report = self.falsification.evaluate(
                claim_id,
                graph=self.graph,
                ledger=self.ledger,
                as_of=as_of,
            )
            for probe in report.probes:
                if probe.rule_id == "missing_falsification_rule":
                    continue
                candidates[_probe_key(probe)] = probe

            rules = self.falsification.rules_for_claim(claim_id)
            by_rule = {
                result.rule_id: result
                for result in report.results
            }
            for rule in rules:
                result = by_rule.get(rule.rule_id)
                if result is None:
                    continue
                extra = _research_probe_for_rule(
                    rule,
                    result.state,
                )
                if extra is not None:
                    candidates.setdefault(_probe_key(extra), extra)

        return sorted(
            candidates.values(),
            key=lambda item: (-item.priority, item.rule_id, item.claim_id),
        )

    def _cooldown_active(
        self,
        db: Session,
        *,
        probe: FalsificationProbe,
        now: datetime,
    ) -> bool:
        key = _probe_key(probe)
        cutoff = now - timedelta(minutes=self.cooldown_minutes)
        row = (
            db.query(ResearchProbeAttemptRecord)
            .filter(
                ResearchProbeAttemptRecord.probe_key == key,
                ResearchProbeAttemptRecord.attempted_at >= cutoff,
                ResearchProbeAttemptRecord.status.in_(
                    ("success", "raw_only", "no_evidence", "no_documents")
                ),
            )
            .order_by(ResearchProbeAttemptRecord.attempted_at.desc())
            .first()
        )
        return row is not None

    @staticmethod
    def _probe_context(
        graph: ClaimGraph,
        falsification: FalsificationEngine,
        probe: FalsificationProbe,
    ) -> tuple[FalsificationRule | None, str, str, EvidenceRelation]:
        claim = graph.get_claim(probe.claim_id)
        rule = next(
            (
                item
                for item in falsification.rules_for_claim(probe.claim_id)
                if item.rule_id == probe.rule_id
            ),
            None,
        )
        evidence_key = (
            rule.evidence_claim_key
            if rule and rule.evidence_claim_key
            else claim.claim_key
        )
        desired = EvidenceRelation.CONTEXT
        if rule:
            if rule.rule_type is FalsificationRuleType.CONTRADICTORY_EVIDENCE:
                desired = EvidenceRelation.CONTRADICTS
            elif rule.rule_type is FalsificationRuleType.MISSING_INDEPENDENT_SUPPORT:
                desired = EvidenceRelation.SUPPORTS
        return rule, claim.statement, evidence_key, desired

    async def run(
        self,
        *,
        db: Session,
        evaluated_at: datetime | None = None,
    ) -> AutomaticResearchResult:
        started = utc(evaluated_at)
        run_id = _hash("rloop", started.isoformat())
        run_row = ResearchLoopRunRecord(
            run_id=run_id,
            started_at=started,
            status="running",
            meta={"version": "automatic-research-v1"},
        )
        db.add(run_row)
        db.commit()

        errors: list[str] = []
        before_cycle = None
        after_cycle = None
        executed = 0
        skipped = 0
        docs_read = 0
        evidence_added = 0
        raw_evidence_added = 0
        classified_evidence_added = 0

        try:
            active_ids = _active_claim_ids(self.graph)
            if not active_ids:
                run_row.status = "no_claims"
                run_row.completed_at = datetime.now(timezone.utc)
                db.commit()
                return AutomaticResearchResult(
                    run_id=run_id,
                    status="no_claims",
                    probes_planned=0,
                    probes_executed=0,
                    probes_skipped_cooldown=0,
                    tool_calls=0,
                    documents_read=0,
                    evidence_added=0,
                    beliefs_changed=0,
                    before_cycle_id=None,
                    after_cycle_id=None,
                )

            monitor = PanWatchBeliefMonitor(
                graph=self.graph,
                ledger=self.ledger,
                falsification=self.falsification,
            )
            before_cycle = monitor.run_cycle(
                db=db,
                claim_ids=active_ids,
                evaluated_at=started,
                metadata={"automatic_research_run_id": run_id, "phase": "before"},
            )
            probes = self._collect_probes(as_of=started)
            run_row.probes_planned = len(probes)
            db.commit()

            for probe in probes:
                if executed >= self.max_probes:
                    break
                if self._cooldown_active(db, probe=probe, now=started):
                    skipped += 1
                    continue

                executed += 1
                probe_key = _probe_key(probe)
                attempt_id = _hash(
                    "attempt",
                    run_id,
                    probe_key,
                    str(executed),
                )
                rule, claim_statement, evidence_key, desired = self._probe_context(
                    self.graph,
                    self.falsification,
                    probe,
                )
                query = (
                    probe.instruction
                    + "\nTarget claim: "
                    + claim_statement
                )
                attempt = ResearchProbeAttemptRecord(
                    attempt_id=attempt_id,
                    run_id=run_id,
                    probe_key=probe_key,
                    rule_id=probe.rule_id,
                    claim_id=probe.claim_id,
                    attempted_at=datetime.now(timezone.utc),
                    status="running",
                    query=query[:5000],
                    tool_name="reach_web_search",
                )
                db.add(attempt)
                db.commit()

                try:
                    documents = await self.gateway.search(
                        query,
                        max_sources=self.max_sources_per_probe,
                    )
                    docs_read += len(documents)
                    attempt.source_count = len(documents)
                    if not documents:
                        attempt.status = "no_documents"
                        db.commit()
                        continue

                    probe_evidence = 0
                    probe_raw = 0
                    classification_errors: list[str] = []
                    for document in documents:
                        now = datetime.now(timezone.utc)
                        source = build_source(
                            url=document.url,
                            content=document.text,
                            publisher=(
                                document.publisher
                                or (urlsplit(document.url).hostname or "")
                            ),
                            title=document.title,
                            source_tier=SourceTier.UNKNOWN,
                            source_family=(
                                urlsplit(document.url).hostname
                                or document.tool_name
                            ),
                            published_at=document.published_at,
                            retrieved_at=now,
                            observed_at=now,
                            tool_name=document.tool_name,
                            metadata={
                                **document.metadata,
                                "automatic_research_run_id": run_id,
                                "probe_rule_id": probe.rule_id,
                            },
                        )
                        self.ledger.register_source(source)

                        try:
                            findings = await self.extractor.extract(
                                probe=probe,
                                rule=rule,
                                claim_statement=claim_statement,
                                evidence_claim_key=evidence_key,
                                desired_relation=desired,
                                document=document,
                                max_findings=self.max_findings_per_document,
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
                            classification_errors.append(error_name)
                            raw_quote = _raw_source_quote(
                                document,
                                probe=probe,
                                claim_statement=claim_statement,
                            )
                            if raw_quote:
                                raw_record = build_evidence(
                                    claim_key=f"raw.{probe.claim_id}",
                                    source=source,
                                    statement=raw_quote,
                                    relation=EvidenceRelation.CONTEXT,
                                    observation_kind=ObservationKind.RAW_SOURCE,
                                    event_time=document.published_at,
                                    observed_at=now,
                                    recorded_at=now,
                                    confidence=0.0,
                                    metadata={
                                        "unclassified_raw": True,
                                        "target_claim_id": probe.claim_id,
                                        "target_claim_key": evidence_key,
                                        "automatic_research_run_id": run_id,
                                        "probe_rule_id": probe.rule_id,
                                        "classification_pending": True,
                                        "classification_error": error_name,
                                    },
                                )
                                if self.ledger.append(raw_record):
                                    probe_raw += 1
                                    raw_evidence_added += 1
                                    evidence_added += 1
                            findings = []
                            logger.warning(
                                "Automatic research classification deferred to raw evidence "
                                "rule=%s url=%s error=%s",
                                probe.rule_id,
                                document.url,
                                error_name,
                            )

                        for finding in findings:
                            record = build_evidence(
                                claim_key=evidence_key,
                                source=source,
                                statement=finding.quote,
                                relation=finding.relation,
                                observation_kind=finding.observation_kind,
                                event_time=document.published_at,
                                observed_at=now,
                                recorded_at=now,
                                confidence=finding.confidence,
                                numeric_value=finding.numeric_value,
                                unit=finding.unit,
                                period=finding.period,
                                metadata={
                                    "quote_grounded": True,
                                    "automatic_research_run_id": run_id,
                                    "probe_rule_id": probe.rule_id,
                                    "classification_pending": False,
                                    "classification_source": "llm_quote_grounded",
                                },
                            )
                            inserted = self.ledger.append(record)
                            if not inserted:
                                continue
                            probe_evidence += 1
                            classified_evidence_added += 1
                            evidence_added += 1
                            self.graph.link_evidence(
                                claim_id=probe.claim_id,
                                evidence_id=record.evidence_id,
                                ledger=self.ledger,
                                relation=finding.relation,
                                weight=min(0.85, finding.confidence),
                                metadata={
                                    "automatic_research": True,
                                    "probe_rule_id": probe.rule_id,
                                },
                            )

                    if probe_raw or probe_evidence:
                        persist_ledger(db, self.ledger)
                    if probe_evidence:
                        persist_claim_graph(db, self.graph)
                        attempt.status = "success"
                    elif probe_raw:
                        attempt.status = "raw_only"
                    else:
                        attempt.status = "no_evidence"
                    attempt.evidence_count = probe_evidence + probe_raw
                    attempt.error_code = (
                        ",".join(sorted(set(classification_errors)))[:255]
                        if classification_errors
                        else ""
                    )
                    attempt.meta = {
                        "raw_evidence_count": probe_raw,
                        "classified_evidence_count": probe_evidence,
                        "classification_errors": sorted(set(classification_errors)),
                    }
                    if classification_errors:
                        errors.extend(
                            f"{probe.rule_id}:{name}"
                            for name in sorted(set(classification_errors))
                        )
                    db.commit()
                except Exception as exc:
                    logger.warning(
                        "Automatic research probe failed rule=%s error=%s",
                        probe.rule_id,
                        type(exc).__name__,
                    )
                    attempt.status = "error"
                    attempt.error_code = type(exc).__name__
                    db.commit()
                    errors.append(
                        f"{probe.rule_id}:{type(exc).__name__}"
                    )

            after_time = datetime.now(timezone.utc)
            after_cycle = monitor.run_cycle(
                db=db,
                claim_ids=active_ids,
                evaluated_at=after_time,
                metadata={"automatic_research_run_id": run_id, "phase": "after"},
            )
            run_row.status = (
                "success_degraded"
                if errors and classified_evidence_added == 0
                else "success"
            )
            run_row.completed_at = after_time
            run_row.probes_executed = executed
            run_row.tool_calls = self.gateway.tool_calls
            run_row.documents_read = docs_read
            run_row.evidence_added = evidence_added
            run_row.beliefs_changed = after_cycle.changed_count
            run_row.error = ";".join(errors)[:4000]
            run_row.meta = {
                "version": "automatic-research-v1",
                "raw_evidence_added": raw_evidence_added,
                "classified_evidence_added": classified_evidence_added,
            }
            db.commit()
            return AutomaticResearchResult(
                run_id=run_id,
                status=run_row.status,
                probes_planned=len(probes),
                probes_executed=executed,
                probes_skipped_cooldown=skipped,
                tool_calls=self.gateway.tool_calls,
                documents_read=docs_read,
                evidence_added=evidence_added,
                beliefs_changed=after_cycle.changed_count,
                before_cycle_id=before_cycle.cycle_id,
                after_cycle_id=after_cycle.cycle_id,
                raw_evidence_added=raw_evidence_added,
                classified_evidence_added=classified_evidence_added,
                errors=tuple(errors),
            )
        except Exception as exc:
            run_row.status = "error"
            run_row.completed_at = datetime.now(timezone.utc)
            run_row.probes_executed = executed
            run_row.tool_calls = self.gateway.tool_calls
            run_row.documents_read = docs_read
            run_row.evidence_added = evidence_added
            run_row.error = type(exc).__name__
            db.commit()
            raise


async def run_automatic_research_once(
    settings: Settings | None = None,
) -> AutomaticResearchResult:
    """Load durable state, run one bounded research cycle, close clients."""
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
        gateway = AhmedToolboxResearchGateway(
            settings,
            max_tool_calls=settings.auto_research_max_tool_calls,
        )
        ai = build_failover_client(
            None,
            None,
            settings=settings,
            max_fallbacks=2,
        )
        extractor = QuoteGroundedEvidenceExtractor(
            ai,
            timeout_seconds=settings.auto_research_extraction_timeout_seconds,
            max_source_chars=settings.auto_research_extraction_max_chars,
        )
        loop = AutomaticResearchLoop(
            graph=graph,
            ledger=ledger,
            falsification=falsification,
            gateway=gateway,
            extractor=extractor,
            max_probes=settings.auto_research_max_probes,
            max_sources_per_probe=settings.auto_research_max_sources_per_probe,
            max_findings_per_document=settings.auto_research_max_findings_per_document,
            cooldown_minutes=settings.auto_research_probe_cooldown_minutes,
        )
        try:
            return await loop.run(db=db)
        finally:
            await gateway.close()
