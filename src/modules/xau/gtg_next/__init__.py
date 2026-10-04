"""GTG Library-First v1.

This package is intentionally isolated from legacy GTG implementations.
It composes mature libraries and keeps custom code limited to GTG domain logic.
"""

from .contracts import EventPrediction, ForwardOutcome, GtgEvent
from .features import FeatureEngine
from .online_eval import OnlinePathEvaluator
from .path_model import KNNPathModel
from .registry import LibraryRegistry

__all__ = [
    "EventPrediction",
    "FeatureEngine",
    "ForwardOutcome",
    "GtgEvent",
    "KNNPathModel",
    "LibraryRegistry",
    "OnlinePathEvaluator",
]
