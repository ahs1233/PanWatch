"""Engine-agnostic trading core contracts.

Ahmed Trading Intelligence owns market interpretation, thesis, signal and risk.
Trading engines consume an immutable strategy package and provide execution /
validation capabilities.  Engines are replaceable infrastructure, not the
strategy brain.

This v1 is deliberately research-only: live execution and broker order
submission are rejected at the contract boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TradingEngineMode(str, Enum):
    HISTORICAL_REPLAY = "historical_replay"
    BACKTEST = "backtest"
    OPTIMIZATION = "optimization"
    PAPER = "paper"
    LIVE_SHADOW = "live_shadow"
    LIVE = "live"


@dataclass(frozen=True)
class TradingEngineCapabilities:
    historical_replay: bool = False
    backtest: bool = False
    optimization: bool = False
    paper: bool = False
    live_shadow: bool = False
    broker_execution: bool = False

    def supports(self, mode: TradingEngineMode) -> bool:
        return {
            TradingEngineMode.HISTORICAL_REPLAY: self.historical_replay,
            TradingEngineMode.BACKTEST: self.backtest,
            TradingEngineMode.OPTIMIZATION: self.optimization,
            TradingEngineMode.PAPER: self.paper,
            TradingEngineMode.LIVE_SHADOW: self.live_shadow,
            TradingEngineMode.LIVE: self.broker_execution,
        }[mode]


@dataclass(frozen=True)
class TradingRunRequest:
    """One immutable handoff from intelligence to an engine."""

    mode: TradingEngineMode
    strategy_fingerprint: str
    strategy: dict[str, Any]
    dataset_ref: str | None = None
    metadata: dict[str, Any] | None = None
    live: bool = False
    broker_orders_enabled: bool = False

    def __post_init__(self) -> None:
        if not self.strategy_fingerprint:
            raise ValueError("strategy_fingerprint is required")
        if not self.strategy:
            raise ValueError("strategy payload is required")
        if self.live:
            raise ValueError("Trading Core v1 refuses live=true")
        if self.broker_orders_enabled:
            raise ValueError("Trading Core v1 refuses broker order submission")
        if self.mode is TradingEngineMode.LIVE:
            raise ValueError("Trading Core v1 does not permit live mode")


@dataclass(frozen=True)
class TradingRunPlan:
    """Prepared engine-specific plan; execution happens outside the brain."""

    engine: str
    mode: TradingEngineMode
    strategy_fingerprint: str
    dataset_ref: str | None
    package: dict[str, Any]
    external_runtime_required: bool
    runtime: str | None = None
    command_hint: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        execution = dict(self.package.get("execution") or {})
        if execution.get("live") is not False:
            raise ValueError("prepared engine plan must remain live=false")
        if execution.get("broker_orders_enabled") is not False:
            raise ValueError("prepared engine plan must disable broker orders")


class TradingEngine(ABC):
    """Replaceable backend boundary for replay/backtest/optimization/paper."""

    name: str
    capabilities: TradingEngineCapabilities

    def validate_request(self, request: TradingRunRequest) -> None:
        if not self.capabilities.supports(request.mode):
            raise ValueError(
                f"{self.name} does not support mode {request.mode.value!r}"
            )

    @abstractmethod
    def prepare(self, request: TradingRunRequest) -> TradingRunPlan:
        """Translate a brain-owned strategy request into an engine plan."""
