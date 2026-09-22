"""Research capability: evidence, claims, falsification and outcomes.

It provides source provenance, append-only evidence, an inspectable Claim Graph,
and explicit falsification rules to other PanWatch capabilities.
"""

from .claim_graph import (
    ClaimAssessment,
    ClaimEdge,
    ClaimEvidenceLink,
    ClaimGraph,
    ClaimKind,
    ClaimNode,
    ClaimRelation,
    ClaimStatus,
    build_claim,
)
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
from .falsification import (
    FalsificationEngine,
    FalsificationProbe,
    FalsificationReport,
    FalsificationResult,
    FalsificationRule,
    FalsificationRuleType,
    FalsificationState,
    build_falsification_rule,
)
from .ledger import EvidenceLedger, GuardResult, NumericResolution

__all__ = [
    "ClaimAssessment",
    "ClaimEdge",
    "ClaimEvidenceLink",
    "ClaimGraph",
    "ClaimKind",
    "ClaimNode",
    "ClaimRelation",
    "ClaimStatus",
    "EvidenceLedger",
    "EvidenceRecord",
    "EvidenceRelation",
    "FalsificationEngine",
    "FalsificationProbe",
    "FalsificationReport",
    "FalsificationResult",
    "FalsificationRule",
    "FalsificationRuleType",
    "FalsificationState",
    "GuardResult",
    "NumericResolution",
    "ObservationKind",
    "SourceProvenance",
    "SourceTier",
    "build_claim",
    "build_evidence",
    "build_falsification_rule",
    "build_source",
    "canonicalize_url",
]
