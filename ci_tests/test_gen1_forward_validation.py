import asyncio

from src.modules.xau import gen1_forward_validation
from src.modules.xau.gen1_forward_validation import Gen1ForwardValidationScheduler
from src.platform.runtime.config import Settings


def test_forward_scheduler_collects_without_forcing_macro(monkeypatch):
    calls = []

    async def fake_pipeline(**kwargs):
        calls.append(kwargs)
        return {
            "decision": "WAIT",
            "confidence": 0.5,
            "pipeline_status": "ready",
            "missing_layers": [],
            "strategy_revision": "rev",
            "live_observation": {"status": "recorded"},
        }

    monkeypatch.setattr(
        gen1_forward_validation,
        "run_gen1_trade_gold_pipeline",
        fake_pipeline,
    )
    scheduler = Gen1ForwardValidationScheduler(
        Settings(
            xau_gen1_forward_validation_enabled=True,
            xau_gen1_forward_interval_seconds=120,
        )
    )
    asyncio.run(scheduler._collect())
    assert calls == [{
        "force_macro": False,
        "record_observation": True,
        "observation_source": "scheduled_forward_validation",
    }]


def test_forward_scheduler_does_not_overlap(monkeypatch):
    scheduler = Gen1ForwardValidationScheduler(Settings())
    scheduler._running = True
    called = False

    async def fail_if_called(**kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(
        gen1_forward_validation,
        "run_gen1_trade_gold_pipeline",
        fail_if_called,
    )
    asyncio.run(scheduler._collect())
    assert called is False
