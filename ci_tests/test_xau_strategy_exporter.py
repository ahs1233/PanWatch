from __future__ import annotations

import pytest

from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.modules.strategy.xau_strategy_exporter import (
    export_backend_package,
    export_native_lean_bundle,
    export_xau_intraday_strategy,
    validate_research_only_package,
)


def test_exporter_mirrors_xau_engine_defaults():
    spec = export_xau_intraday_strategy()
    payload = spec.to_dict()

    assert payload["symbol"] == "XAUUSD"
    assert payload["timeframes"] == ["1m", "5m", "15m"]
    assert payload["indicators"] == {
        "ema_fast": 9,
        "ema_slow": 21,
        "rsi_period": 14,
        "atr_period": 14,
        "breakout_lookback": 20,
    }
    assert payload["gates"]["bar_max_age_seconds"] == {
        "1m": 180,
        "5m": 720,
        "15m": 2100,
    }


def test_exporter_tracks_custom_engine_configuration():
    engine = XAUIntradayEngine(
        fast_ema=21,
        slow_ema=50,
        rsi_period=10,
        atr_period=20,
        breakout_lookback=30,
        max_spread_bps=4.5,
        require_execution_data=True,
    )

    spec = export_xau_intraday_strategy(engine)
    payload = spec.to_dict()

    assert payload["indicators"]["ema_fast"] == 21
    assert payload["indicators"]["ema_slow"] == 50
    assert payload["indicators"]["rsi_period"] == 10
    assert payload["indicators"]["atr_period"] == 20
    assert payload["indicators"]["breakout_lookback"] == 30
    assert payload["gates"]["max_spread_bps"] == 4.5
    assert payload["gates"]["require_execution_data"] is True


def test_strategy_fingerprint_is_deterministic_and_configuration_sensitive():
    first = export_xau_intraday_strategy()
    second = export_xau_intraday_strategy()
    changed = export_xau_intraday_strategy(
        XAUIntradayEngine(fast_ema=21, slow_ema=50)
    )

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert len(first.fingerprint) == 64


def test_native_and_lean_packages_share_exact_strategy_identity():
    bundle = export_native_lean_bundle()

    assert bundle["native"]["strategy"] == bundle["lean"]["strategy"]
    assert (
        bundle["native"]["strategy_fingerprint"]
        == bundle["lean"]["strategy_fingerprint"]
    )
    assert bundle["native"]["backend"] == "native"
    assert bundle["lean"]["backend"] == "lean"
    assert bundle["lean"]["adapter_contract"]["external_runner_required"] is True


@pytest.mark.parametrize("backend", ["native", "lean", "NATIVE", " LEAN "])
def test_all_supported_exports_are_research_only(backend):
    package = export_backend_package(backend)

    assert package["execution"] == {
        "live": False,
        "broker_orders_enabled": False,
        "simulation_allowed": True,
    }
    assert package["strategy"]["execution"]["live"] is False
    assert package["strategy"]["execution"]["broker_orders_enabled"] is False


def test_exporter_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unsupported trading backend"):
        export_backend_package("metatrader-live")


def test_package_safety_validator_fails_closed_if_mutated_to_live():
    package = export_backend_package("lean")
    package["execution"]["live"] = True

    with pytest.raises(ValueError, match="live=false"):
        validate_research_only_package(package)
