"""Adapters from Trading Core to concrete validation/execution engines."""

from __future__ import annotations

from src.modules.strategy.trading_engine import (
    TradingEngine,
    TradingEngineCapabilities,
    TradingRunPlan,
    TradingRunRequest,
)
from src.modules.strategy.xau_strategy_exporter import validate_research_only_package


class NativeEngineAdapter(TradingEngine):
    """PanWatch's deterministic native research engine."""

    name = "native"
    capabilities = TradingEngineCapabilities(
        historical_replay=True,
        backtest=True,
        optimization=False,
        paper=False,
        live_shadow=False,
        broker_execution=False,
    )

    def prepare(self, request: TradingRunRequest) -> TradingRunPlan:
        self.validate_request(request)
        package = {
            "package_version": "1.0",
            "backend": self.name,
            "strategy_fingerprint": request.strategy_fingerprint,
            "strategy": request.strategy,
            "adapter_contract": {
                "engine_module": "src.modules.strategy.xau_intraday",
                "engine_class": "XAUIntradayEngine",
                "external_runtime_required": False,
            },
            "execution": {
                "live": False,
                "broker_orders_enabled": False,
                "simulation_allowed": True,
            },
        }
        _validate_core_package(package)
        return TradingRunPlan(
            engine=self.name,
            mode=request.mode,
            strategy_fingerprint=request.strategy_fingerprint,
            dataset_ref=request.dataset_ref,
            package=package,
            external_runtime_required=False,
        )


class LeanEngineAdapter(TradingEngine):
    """QuantConnect LEAN as a replaceable validation/execution kernel."""

    name = "lean"
    capabilities = TradingEngineCapabilities(
        historical_replay=True,
        backtest=True,
        optimization=True,
        paper=True,
        live_shadow=True,
        broker_execution=False,
    )

    def __init__(self, runtime_image: str = "quantconnect/lean:latest") -> None:
        self.runtime_image = runtime_image

    def prepare(self, request: TradingRunRequest) -> TradingRunPlan:
        self.validate_request(request)
        package = {
            "package_version": "1.0",
            "backend": self.name,
            "strategy_fingerprint": request.strategy_fingerprint,
            "strategy": request.strategy,
            "adapter_contract": {
                "engine": "QuantConnect LEAN",
                "algorithm_language": "Python",
                "external_runtime_required": True,
                "runtime_image": self.runtime_image,
            },
            "execution": {
                "live": False,
                "broker_orders_enabled": False,
                "simulation_allowed": True,
            },
        }
        _validate_core_package(package)
        return TradingRunPlan(
            engine=self.name,
            mode=request.mode,
            strategy_fingerprint=request.strategy_fingerprint,
            dataset_ref=request.dataset_ref,
            package=package,
            external_runtime_required=True,
            runtime=self.runtime_image,
            command_hint=("docker", "run", "--rm", self.runtime_image),
        )


def _validate_core_package(package: dict) -> None:
    """Reuse the existing fail-closed execution invariants when compatible."""

    # The XAU exporter validator checks the same execution invariants plus the
    # strategy's embedded execution policy.  Trading Core accepts any future
    # strategy schema, so enforce the generic boundary here first.
    execution = dict(package.get("execution") or {})
    if execution.get("live") is not False:
        raise ValueError("engine package must set live=false")
    if execution.get("broker_orders_enabled") is not False:
        raise ValueError("engine package must disable broker orders")

    strategy = package.get("strategy") or {}
    if "execution" in strategy:
        validate_research_only_package(package)
