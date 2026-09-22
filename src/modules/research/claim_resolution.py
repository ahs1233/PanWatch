"""Conservative semantic resolution between newly acquired and existing claims.

This module deliberately avoids embedding-only merging. A merge/contradiction
decision is based on an inspectable combination of:
- semantic claim key when supplied by the extractor;
- lexical overlap and sequence similarity;
- numeric values and explicit years/periods;
- negation and directional polarity;
- explicit revision cues.

When the signals are not strong enough, the resolver returns AMBIGUOUS or
DISTINCT rather than silently merging two claims.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum
from typing import Any, Iterable

from .claim_graph import ClaimGraph, ClaimNode
from .evidence import normalize_text


class SemanticClaimRelation(StrEnum):
    EXACT = "exact"
    PARAPHRASE = "paraphrase"
    CONTRADICTION = "contradiction"
    REVISION = "revision"
    DISTINCT = "distinct"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ClaimResolution:
    relation: SemanticClaimRelation
    matched_claim_id: str | None
    score: float
    lexical_score: float
    key_match: bool
    numeric_match: bool
    period_match: bool
    polarity_match: bool
    reason: str
    signals: dict[str, Any] = field(default_factory=dict)


_STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "by",
    "from", "at", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "with", "its", "their", "it", "they",
    "about", "around", "approximately", "estimated", "estimate",
    "according", "today", "current", "currently",
    "في", "من", "إلى", "الى", "على", "عن", "مع", "هو", "هي", "هذا", "هذه",
    "ذلك", "تلك", "تم", "قد", "ما", "أن", "ان", "و", "أو", "او",
}
_NEGATION = {
    "not", "no", "never", "without", "isn't", "aren't", "wasn't", "weren't",
    "لم", "لن", "ليس", "ليست", "لا", "بدون", "غير",
}
_POSITIVE_DIRECTION = {
    "increase", "increased", "increases", "increasing", "rise", "rose", "rises",
    "rising", "growth", "grew", "grow", "higher", "above", "up", "gain",
    "gained", "expanded", "expansion", "ارتفع", "ارتفعت", "زيادة", "زاد",
    "أعلى", "اعلى", "نمو",
}
_NEGATIVE_DIRECTION = {
    "decrease", "decreased", "decreases", "decreasing", "fall", "fell", "falls",
    "falling", "decline", "declined", "lower", "below", "down", "drop",
    "dropped", "contracted", "contraction", "انخفض", "انخفضت", "انخفاض",
    "تراجع", "تراجعت", "أقل", "اقل",
}
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ\u0600-\u06FF0-9%$€£._/-]+")
_NUMBER_RE = re.compile(
    r"(?<![A-Za-z])[-+]?\d+(?:[.,]\d+)?(?:\s*%|\s*(?:twh|gwh|mwh|kw|mw|gw|"
    r"billion|million|trillion|bn|mn|percent|percentage|usd|eur|iqd|dollars?))?",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
_AUTO_KEY_RE = re.compile(r"^auto\.claim\.", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    values = {
        token.casefold().strip(".,:;()[]{}")
        for token in _TOKEN_RE.findall(normalize_text(text))
    }
    return {
        token
        for token in values
        if len(token) >= 2 and token not in _STOPWORDS and not token.isdigit()
    }


def _numbers(text: str) -> tuple[str, ...]:
    years = set(_YEAR_RE.findall(text))
    rows: list[str] = []
    for raw in _NUMBER_RE.findall(normalize_text(text)):
        value = re.sub(r"\s+", "", raw.casefold()).replace(",", "")
        if value in years:
            continue
        if value:
            rows.append(value)
    return tuple(sorted(set(rows)))


def _years(text: str) -> tuple[str, ...]:
    return tuple(sorted(set(_YEAR_RE.findall(normalize_text(text)))))


def _has_negation(text: str) -> bool:
    tokens = {token.casefold() for token in _TOKEN_RE.findall(normalize_text(text))}
    return bool(tokens & _NEGATION)


def _direction(text: str) -> int:
    tokens = {token.casefold() for token in _TOKEN_RE.findall(normalize_text(text))}
    positive = len(tokens & _POSITIVE_DIRECTION)
    negative = len(tokens & _NEGATIVE_DIRECTION)
    if positive > negative:
        return 1
    if negative > positive:
        return -1
    return 0


def _lexical_similarity(left: str, right: str) -> tuple[float, float, float]:
    left_text = normalize_text(left).casefold()
    right_text = normalize_text(right).casefold()
    if left_text == right_text:
        return 1.0, 1.0, 1.0
    left_tokens = _tokens(left_text)
    right_tokens = _tokens(right_text)
    if left_tokens or right_tokens:
        union = left_tokens | right_tokens
        jaccard = len(left_tokens & right_tokens) / max(1, len(union))
    else:
        jaccard = 0.0
    sequence = SequenceMatcher(None, left_text, right_text).ratio()
    combined = 0.58 * jaccard + 0.42 * sequence
    return round(combined, 4), round(jaccard, 4), round(sequence, 4)


def _explicit_key(key: str) -> bool:
    value = normalize_text(key)
    return bool(value) and not _AUTO_KEY_RE.search(value)


def _active_claims(graph: ClaimGraph) -> list[ClaimNode]:
    superseded = {claim.supersedes for claim in graph.claims if claim.supersedes}
    return [
        claim
        for claim in graph.claims
        if claim.claim_id not in superseded
        and not bool(claim.metadata.get("admission_invalidated"))
    ]


def _period_match(left: str, right: str) -> bool:
    a = _years(left)
    b = _years(right)
    if a and b:
        return a == b
    return True


def _numeric_match(left: str, right: str) -> bool:
    a = _numbers(left)
    b = _numbers(right)
    if a and b:
        return a == b
    return not a and not b


def _numeric_conflict(left: str, right: str) -> bool:
    a = _numbers(left)
    b = _numbers(right)
    return bool(a and b and a != b)


def _polarity_match(left: str, right: str) -> bool:
    negation_match = _has_negation(left) == _has_negation(right)
    left_dir = _direction(left)
    right_dir = _direction(right)
    direction_match = (
        left_dir == 0 or right_dir == 0 or left_dir == right_dir
    )
    return negation_match and direction_match


def _polarity_conflict(left: str, right: str) -> bool:
    if _has_negation(left) != _has_negation(right):
        return True
    left_dir = _direction(left)
    right_dir = _direction(right)
    return bool(left_dir and right_dir and left_dir != right_dir)


class ConservativeClaimResolver:
    """Resolve one candidate against the active Claim Graph conservatively."""

    def resolve(
        self,
        *,
        statement: str,
        claim_key: str,
        graph: ClaimGraph,
        supersedes_previous: bool = False,
        revision_explicit: bool = False,
    ) -> ClaimResolution:
        candidate_text = normalize_text(statement)
        candidate_key = normalize_text(claim_key)
        active = _active_claims(graph)
        if not active:
            return ClaimResolution(
                relation=SemanticClaimRelation.DISTINCT,
                matched_claim_id=None,
                score=0.0,
                lexical_score=0.0,
                key_match=False,
                numeric_match=False,
                period_match=True,
                polarity_match=True,
                reason="no_active_claims",
                signals={},
            )

        best: ClaimResolution | None = None
        best_rank = -1.0

        for existing in active:
            lexical, jaccard, sequence = _lexical_similarity(
                candidate_text,
                existing.statement,
            )
            exact = candidate_text.casefold() == normalize_text(
                existing.statement
            ).casefold()
            key_match = (
                _explicit_key(candidate_key)
                and candidate_key == normalize_text(existing.claim_key)
            )
            period_match = _period_match(candidate_text, existing.statement)
            numeric_match = _numeric_match(candidate_text, existing.statement)
            numeric_conflict = _numeric_conflict(
                candidate_text,
                existing.statement,
            )
            polarity_match = _polarity_match(
                candidate_text,
                existing.statement,
            )
            polarity_conflict = _polarity_conflict(
                candidate_text,
                existing.statement,
            )

            relation = SemanticClaimRelation.DISTINCT
            reason = "insufficient_semantic_overlap"
            score = lexical

            if exact:
                relation = SemanticClaimRelation.EXACT
                reason = "normalized_statement_exact_match"
                score = 1.0
            elif (
                supersedes_previous
                and revision_explicit
                and key_match
                and period_match
            ):
                relation = SemanticClaimRelation.REVISION
                reason = "explicit_revision_same_semantic_key"
                score = max(0.9, lexical)
            elif (
                period_match
                and (
                    (key_match and (numeric_conflict or polarity_conflict))
                    or (
                        lexical >= 0.62
                        and (numeric_conflict or polarity_conflict)
                    )
                )
            ):
                relation = SemanticClaimRelation.CONTRADICTION
                reason = (
                    "same_scope_conflicting_numeric_or_polarity"
                    if key_match
                    else "high_overlap_conflicting_numeric_or_polarity"
                )
                score = max(0.72, lexical)
            elif (
                period_match
                and numeric_match
                and polarity_match
                and (
                    (key_match and lexical >= 0.30)
                    or lexical >= 0.72
                )
            ):
                relation = SemanticClaimRelation.PARAPHRASE
                reason = (
                    "same_key_compatible_paraphrase"
                    if key_match
                    else "high_overlap_compatible_paraphrase"
                )
                score = max(0.75, lexical)
            elif key_match and period_match:
                relation = SemanticClaimRelation.AMBIGUOUS
                reason = "same_semantic_key_but_insufficient_resolution_signal"
                score = max(0.5, lexical)
            elif lexical >= 0.58:
                relation = SemanticClaimRelation.AMBIGUOUS
                reason = "moderate_overlap_requires_separate_claim"
                score = lexical
            elif not period_match:
                relation = SemanticClaimRelation.DISTINCT
                reason = "different_explicit_period"
                score = lexical

            relation_priority = {
                SemanticClaimRelation.EXACT: 6,
                SemanticClaimRelation.REVISION: 5,
                SemanticClaimRelation.CONTRADICTION: 4,
                SemanticClaimRelation.PARAPHRASE: 3,
                SemanticClaimRelation.AMBIGUOUS: 2,
                SemanticClaimRelation.DISTINCT: 1,
            }[relation]
            rank = relation_priority + score
            if rank <= best_rank:
                continue
            best_rank = rank
            best = ClaimResolution(
                relation=relation,
                matched_claim_id=existing.claim_id,
                score=round(score, 4),
                lexical_score=lexical,
                key_match=key_match,
                numeric_match=numeric_match,
                period_match=period_match,
                polarity_match=polarity_match,
                reason=reason,
                signals={
                    "jaccard": jaccard,
                    "sequence": sequence,
                    "candidate_numbers": list(_numbers(candidate_text)),
                    "existing_numbers": list(_numbers(existing.statement)),
                    "candidate_years": list(_years(candidate_text)),
                    "existing_years": list(_years(existing.statement)),
                    "candidate_direction": _direction(candidate_text),
                    "existing_direction": _direction(existing.statement),
                    "candidate_negated": _has_negation(candidate_text),
                    "existing_negated": _has_negation(existing.statement),
                    "existing_claim_key": existing.claim_key,
                },
            )

        assert best is not None
        return best
