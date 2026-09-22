"""Research capability: analysis history, context, evidence and outcomes.

It provides read models, source provenance, evidence-building services and
research integrity guards to other PanWatch capabilities.
"""

from .evidence import (
    EvidenceRecord,
    EvidenceRelation,
    ObservationKind,
    SourceProvenance,
    SourceTier,
    build_evidence,
    build_source,
    canonicalize_url,
)
from .ledger import EvidenceLedger, GuardResult, NumericResolution

__all__ = [
    "EvidenceLedger",
    "EvidenceRecord",
    "EvidenceRelation",
    "GuardResult",
    "NumericResolution",
    "ObservationKind",
    "SourceProvenance",
    "SourceTier",
    "build_evidence",
    "build_source",
    "canonicalize_url",
]
