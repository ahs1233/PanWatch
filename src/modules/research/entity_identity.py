"""Conservative entity identity for research claims.

Entity identity is a hard semantic boundary. Claims about different entities
must not be merged merely because their wording, metric, period, and numeric
value are similar.

The layer is deliberately deterministic:
- grounded entity metadata is preferred when available;
- hierarchical claim-key scope is used as a conservative fallback;
- common legal suffixes are normalized;
- aliases are explicit, never inferred from fuzzy text similarity;
- unknown identity does not invent an entity.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable

from .evidence import normalize_text


class EntityType(StrEnum):
    COMPANY = "company"
    ORGANIZATION = "organization"
    PERSON = "person"
    SECTOR = "sector"
    PRODUCT = "product"
    ASSET = "asset"
    COUNTRY = "country"
    LOCATION = "location"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EntityIdentity:
    canonical_id: str
    canonical_name: str
    entity_type: EntityType = EntityType.UNKNOWN
    aliases: tuple[str, ...] = ()
    source: str = "unknown"
    confidence: float = 0.0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "entity_id": self.canonical_id,
            "entity_name": self.canonical_name,
            "entity_type": self.entity_type.value,
            "entity_aliases": list(self.aliases),
            "entity_identity_source": self.source,
            "entity_identity_confidence": self.confidence,
        }


@dataclass(frozen=True)
class EntityComparison:
    relation: str
    same_entity: bool | None
    left: EntityIdentity | None
    right: EntityIdentity | None
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


_GENERIC_KEY_NAMESPACES = {
    "auto",
    "claim",
    "macro",
    "economy",
    "economic",
    "energy",
    "science",
    "policy",
    "market",
    "markets",
    "rates",
    "event",
    "events",
    "oil",
    "inflation",
    "company",
    "companies",
    "sector",
    "industry",
    "research",
    "global",
    "world",
    "country",
    "government",
    "technology",
    "tech",
    "commodity",
    "commodities",
    "crypto",
}
_LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "ltd",
    "limited",
    "llc",
    "plc",
    "sa",
    "ag",
    "nv",
    "holdings",
    "holding",
    "group",
}
_ALLOWED_ENTITY_TYPES = {item.value: item for item in EntityType}
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_KEY_PART_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,79}$", re.IGNORECASE)


def _ascii(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", normalize_text(value))
    return normalized.encode("ascii", "ignore").decode("ascii")


def normalize_entity_name(value: str) -> str:
    raw = _ascii(value).casefold()
    raw = re.sub(r"[^a-z0-9& ]+", " ", raw)
    tokens = [token for token in raw.split() if token]
    while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def entity_slug(value: str) -> str:
    normalized = normalize_entity_name(value)
    return _SLUG_RE.sub("-", normalized).strip("-")


def _entity_type(value: Any) -> EntityType:
    return _ALLOWED_ENTITY_TYPES.get(
        str(value or "unknown").strip().lower(),
        EntityType.UNKNOWN,
    )


def _grounded_in_text(name: str, texts: Iterable[str]) -> bool:
    target = normalize_entity_name(name)
    if not target:
        return False
    for text in texts:
        haystack = normalize_entity_name(str(text or ""))
        if target and target in haystack:
            return True
    return False


def _key_scope(claim_key: str) -> str:
    key = normalize_text(claim_key).lower()
    if not key or key.startswith("auto.claim."):
        return ""
    first = re.split(r"[.:/]", key, maxsplit=1)[0].strip()
    if (
        not first
        or first in _GENERIC_KEY_NAMESPACES
        or not _KEY_PART_RE.fullmatch(first)
    ):
        return ""
    return first


class EntityIdentityLayer:
    """Resolve and compare claim subjects without fuzzy entity guessing."""

    def __init__(
        self,
        *,
        alias_map: dict[str, str] | None = None,
    ) -> None:
        self._aliases: dict[str, str] = {}
        for alias, canonical in (alias_map or {}).items():
            alias_slug = entity_slug(alias)
            canonical_slug = entity_slug(canonical)
            if alias_slug and canonical_slug:
                self._aliases[alias_slug] = canonical_slug

    def canonical_slug(self, value: str) -> str:
        slug = entity_slug(value)
        return self._aliases.get(slug, slug)

    def resolve(
        self,
        *,
        claim_key: str,
        statement: str,
        metadata: dict[str, Any] | None = None,
        quote: str = "",
    ) -> EntityIdentity | None:
        meta = dict(metadata or {})
        meta_name = normalize_text(str(meta.get("entity_name") or ""))
        meta_id = normalize_text(str(meta.get("entity_id") or ""))
        entity_type = _entity_type(meta.get("entity_type"))

        if meta_name and _grounded_in_text(
            meta_name,
            (quote, statement),
        ):
            canonical = self.canonical_slug(meta_name)
            if canonical:
                aliases = tuple(
                    sorted(
                        {
                            normalize_text(str(alias))
                            for alias in meta.get("entity_aliases", [])
                            if normalize_text(str(alias))
                        }
                    )
                )
                return EntityIdentity(
                    canonical_id=f"{entity_type.value}:{canonical}",
                    canonical_name=normalize_entity_name(meta_name),
                    entity_type=entity_type,
                    aliases=aliases,
                    source="grounded_metadata",
                    confidence=1.0,
                )

        # Persisted identities are trusted only when they were previously
        # produced by this deterministic layer or grounded extractor.
        source = str(meta.get("entity_identity_source") or "")
        if meta_id and source in {
            "grounded_metadata",
            "claim_key_scope",
            "entity_identity_v1",
        }:
            canonical_name = normalize_entity_name(
                str(meta.get("entity_name") or meta_id.split(":", 1)[-1])
            )
            return EntityIdentity(
                canonical_id=meta_id,
                canonical_name=canonical_name,
                entity_type=entity_type,
                aliases=tuple(meta.get("entity_aliases") or ()),
                source=source,
                confidence=float(meta.get("entity_identity_confidence") or 0.9),
            )

        scope = _key_scope(claim_key)
        if scope:
            canonical = self.canonical_slug(scope)
            return EntityIdentity(
                canonical_id=f"scope:{canonical}",
                canonical_name=canonical,
                entity_type=EntityType.UNKNOWN,
                source="claim_key_scope",
                confidence=0.85,
            )
        return None

    def compare(
        self,
        left: EntityIdentity | None,
        right: EntityIdentity | None,
    ) -> EntityComparison:
        if left is None and right is None:
            return EntityComparison(
                relation="unknown",
                same_entity=None,
                left=None,
                right=None,
                reason="both_entity_identities_unknown",
            )
        if left is None or right is None:
            return EntityComparison(
                relation="unknown",
                same_entity=None,
                left=left,
                right=right,
                reason="one_entity_identity_unknown",
            )

        left_key = left.canonical_id.split(":", 1)[-1]
        right_key = right.canonical_id.split(":", 1)[-1]
        same = left_key == right_key
        return EntityComparison(
            relation="same" if same else "different",
            same_entity=same,
            left=left,
            right=right,
            reason=(
                "canonical_entity_identity_match"
                if same
                else "canonical_entity_identity_mismatch"
            ),
            signals={
                "left_entity_id": left.canonical_id,
                "right_entity_id": right.canonical_id,
                "left_source": left.source,
                "right_source": right.source,
            },
        )
