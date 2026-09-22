"""Scientific validation layer for engine-neutral trading research."""

from src.modules.strategy.validation.engine import validate_experiment
from src.modules.strategy.validation.models import (
    DatasetManifest,
    ExperimentSpec,
    GateResult,
    ParameterPoint,
    PerformanceMetrics,
    TradeRecord,
    ValidationPolicy,
    ValidationReport,
    ValidationVerdict,
)

__all__ = [
    "DatasetManifest",
    "ExperimentSpec",
    "GateResult",
    "ParameterPoint",
    "PerformanceMetrics",
    "TradeRecord",
    "ValidationPolicy",
    "ValidationReport",
    "ValidationVerdict",
    "validate_experiment",
]
