"""Canonical exporter for the deterministic PanWatch XAU intraday strategy.

The exporter has one job: freeze the strategy semantics once, then hand the
same immutable contract to every research/backtest backend.  It does not place
orders and it cannot enable live trading.

This prevents Native-vs-LEAN comparisons from accidentally comparing two
different strategies because of duplicated configuration or drifting defaults.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from src.modules.strategy.xau_intraday import XAUIntradayEngine


_BACKENDS = {"native", "lean"}
_TIMEFRAMES = ("1m", "5m", "15m")


@dataclass(frozen=True)
class XAUStrategySpec:
    """Immutable, engine-neutral XAU intraday strategy contract."""

    strategy_id: str = "panwatch-xau-intraday"
    schema_version: str = "1.0"
    symbol: str = "XAUUSD"
    timeframes: tuple[str, ...] = _TIMEFRAMES
    fast_ema: int = 9
    slow_ema: int = 21
    rsi_period: int = 14
    atr_period: int = 14
    breakout_lookback: int = 20
    max_spread_bps: float | None = None
    require_execution_data: bool = False
    live: bool = field(default=False, init=False)
    broker_orders_enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.timeframes != _TIMEFRAMES:
            raise ValueError("XAU intraday export contract requires 1m/5m/15m")
        if self.fast_ema < 2 or self.slow_ema <= self.fast_ema:
            raise ValueError("EMA periods must satisfy 2 <= fast < slow")
        if min(self.rsi_period, self.atr_period, self.breakout_lookback) < 2:
            raise ValueError("indicator/lookback periods must be >= 2")
        if self.max_spread_bps is not None and self.max_spread_bps <= 0:
            raise ValueError("max_spread_bps must be positive when configured")

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.strategy_id,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "timeframes": list(self.timeframes),
            "indicators": {
                "ema_fast": self.fast_ema,
                "ema_slow": self.slow_ema,
                "rsi_period": self.rsi_period,
                "atr_period": self.atr_period,
                "breakout_lookback": self.breakout_lookback,
            },
            "entry_rules": {
                "long": [
                    "5m.direction == bullish",
                    "15m.direction == bullish",
                    "1m.direction != bearish",
                ],
                "short": [
                    "5m.direction == bearish",
                    "15m.direction == bearish",
                    "1m.direction != bullish",
                ],
            },
            "gates": {
                "event_risk_blocks": True,
                "required_timeframes_must_be_present": True,
                "required_timeframes_must_be_fresh": True,
                "max_spread_bps": self.max_spread_bps,
                "require_execution_data": self.require_execution_data,
                "execution_quote_max_age_seconds": 30,
                "bar_max_age_seconds": {
                    timeframe.value: int(max_age.total_seconds())
                    for timeframe, max_age in XAUIntradayEngine._MAX_AGE.items()
                    if timeframe.value in self.timeframes
                },
            },
            "execution": {
                "live": self.live,
                "broker_orders_enabled": self.broker_orders_enabled,
                "simulation_allowed": True,
            },
        }

    @property
    def fingerprint(self) -> str:
        canonical = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return sha256(canonical.encode("utf-8")).hexdigest()


def export_xau_intraday_strategy(
    engine: XAUIntradayEngine | None = None,
) -> XAUStrategySpec:
    """Freeze the effective XAUIntradayEngine configuration into one spec."""

    engine = engine or XAUIntradayEngine()
    return XAUStrategySpec(
        fast_ema=int(engine.fast_ema),
        slow_ema=int(engine.slow_ema),
        rsi_period=int(engine.rsi_period),
        atr_period=int(engine.atr_period),
        breakout_lookback=int(engine.breakout_lookback),
        max_spread_bps=(
            float(engine.max_spread_bps)
            if engine.max_spread_bps is not None
            else None
        ),
        require_execution_data=bool(engine.require_execution_data),
    )


def export_backend_package(
    backend: str,
    *,
    spec: XAUStrategySpec | None = None,
) -> dict[str, Any]:
    """Export one canonical strategy package for a research backend.

    The package deliberately carries no live switch.  LEAN receives the same
    strategy fingerprint as Native; only the backend adapter contract differs.
    """

    normalized = str(backend or "").strip().lower()
    if normalized not in _BACKENDS:
        raise ValueError(f"unsupported trading backend: {backend!r}")

    strategy = spec or export_xau_intraday_strategy()
    if strategy.live or strategy.broker_orders_enabled:
        raise ValueError("live/broker execution is forbidden by exporter contract")

    adapter_contract: dict[str, Any]
    if normalized == "native":
        adapter_contract = {
            "engine_module": "src.modules.strategy.xau_intraday",
            "engine_class": "XAUIntradayEngine",
            "input_contract": "dict[XAUTimeframe, list[XAUBar]]",
            "external_runner_required": False,
        }
    else:
        adapter_contract = {
            "engine": "QuantConnect LEAN",
            "algorithm_language": "Python",
            "base_resolution": "1m",
            "consolidations": ["5m", "15m"],
            "external_runner_required": True,
            "live_mode": False,
        }

    package = {
        "package_version": "1.0",
        "backend": normalized,
        "strategy_fingerprint": strategy.fingerprint,
        "strategy": strategy.to_dict(),
        "adapter_contract": adapter_contract,
        "execution": {
            "live": False,
            "broker_orders_enabled": False,
            "simulation_allowed": True,
        },
    }
    validate_research_only_package(package)
    return package


def export_native_lean_bundle(
    spec: XAUStrategySpec | None = None,
) -> dict[str, dict[str, Any]]:
    """Return Native and LEAN packages proven to share one strategy identity."""

    strategy = spec or export_xau_intraday_strategy()
    native = export_backend_package("native", spec=strategy)
    lean = export_backend_package("lean", spec=strategy)
    if native["strategy_fingerprint"] != lean["strategy_fingerprint"]:
        raise RuntimeError("backend exports drifted from the canonical strategy")
    return {"native": native, "lean": lean}


def validate_research_only_package(package: dict[str, Any]) -> None:
    """Fail closed if an exported package could be interpreted as live."""

    execution = dict(package.get("execution") or {})
    strategy_execution = dict((package.get("strategy") or {}).get("execution") or {})
    if execution.get("live") is not False:
        raise ValueError("exported trading package must set live=false")
    if execution.get("broker_orders_enabled") is not False:
        raise ValueError("exported trading package must disable broker orders")
    if strategy_execution.get("live") is not False:
        raise ValueError("strategy contract must set live=false")
    if strategy_execution.get("broker_orders_enabled") is not False:
        raise ValueError("strategy contract must disable broker orders")
