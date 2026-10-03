from __future__ import annotations

import pytest

from src.modules.strategy.trading_engine import (
    TradingEngineMode,
    TradingRunRequest,
)
from src.modules.strategy.trading_engine_adapters import (
    LeanEngineAdapter,
    NativeEngineAdapter,
)
from src.modules.strategy.xau_strategy_exporter import export_xau_intraday_strategy


def _request(mode=TradingEngineMode.BACKTEST):
    spec = export_xau_intraday_strategy()
    return TradingRunRequest(
        mode=mode,
        strategy_fingerprint=spec.fingerprint,
        strategy=spec.to_dict(),
        dataset_ref="benchmarks/xau_engine_parity_v1/frozen_dataset.json",
    )


def test_native_and_lean_share_one_engine_agnostic_request():
    request = _request()

    native = NativeEngineAdapter().prepare(request)
    lean = LeanEngineAdapter().prepare(request)

    assert native.strategy_fingerprint == lean.strategy_fingerprint
    assert native.package["strategy"] == lean.package["strategy"]
    assert native.dataset_ref == lean.dataset_ref
    assert native.engine == "native"
    assert lean.engine == "lean"


def test_lean_is_infrastructure_not_strategy_brain():
    plan = LeanEngineAdapter().prepare(_request())

    assert plan.external_runtime_required is True
    assert plan.runtime == "quantconnect/lean:latest"
    assert plan.package["adapter_contract"]["engine"] == "QuantConnect LEAN"
    assert "entry_rules" in plan.package["strategy"]


def test_native_is_replaceable_without_changing_strategy_spec():
    request = _request(TradingEngineMode.HISTORICAL_REPLAY)
    native = NativeEngineAdapter().prepare(request)

    assert native.external_runtime_required is False
    assert native.package["backend"] == "native"
    assert native.package["strategy_fingerprint"] == request.strategy_fingerprint


@pytest.mark.parametrize(
    "mode",
    [
        TradingEngineMode.BACKTEST,
        TradingEngineMode.OPTIMIZATION,
        TradingEngineMode.PAPER,
        TradingEngineMode.LIVE_SHADOW,
    ],
)
def test_lean_declares_validation_capabilities_without_broker_execution(mode):
    plan = LeanEngineAdapter().prepare(_request(mode))

    assert plan.package["execution"]["live"] is False
    assert plan.package["execution"]["broker_orders_enabled"] is False


@pytest.mark.parametrize(
    "mode",
    [
        TradingEngineMode.OPTIMIZATION,
        TradingEngineMode.PAPER,
        TradingEngineMode.LIVE_SHADOW,
    ],
)
def test_native_rejects_modes_it_does_not_implement(mode):
    with pytest.raises(ValueError, match="does not support mode"):
        NativeEngineAdapter().prepare(_request(mode))


def test_core_contract_rejects_live_mode_before_adapter():
    spec = export_xau_intraday_strategy()

    with pytest.raises(ValueError, match="does not permit live mode"):
        TradingRunRequest(
            mode=TradingEngineMode.LIVE,
            strategy_fingerprint=spec.fingerprint,
            strategy=spec.to_dict(),
        )


def test_core_contract_rejects_broker_orders_before_adapter():
    spec = export_xau_intraday_strategy()

    with pytest.raises(ValueError, match="broker order"):
        TradingRunRequest(
            mode=TradingEngineMode.BACKTEST,
            strategy_fingerprint=spec.fingerprint,
            strategy=spec.to_dict(),
            broker_orders_enabled=True,
        )
