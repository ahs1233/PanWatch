from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.modules.xau.gtg_next import (
    EventPrediction,
    FeatureEngine,
    GtgEvent,
    KNNPathModel,
    OnlinePathEvaluator,
)


def test_feature_engine_delegates_indicator_primitives() -> None:
    idx = pd.date_range("2026-01-01", periods=400, freq="5min", tz="UTC")
    close = np.linspace(2600.0, 2660.0, len(idx)) + np.sin(np.arange(len(idx)) / 5)
    frame = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
        },
        index=idx,
    )
    features = FeatureEngine().transform(frame)
    expected = {"ema_14", "ema_50", "ema_200", "rsi", "rsi_slope", "ema_14_50_spacing_pct"}
    assert expected.issubset(features.columns)
    assert features["ema_200"].notna().sum() > 0


def test_knn_path_model_uses_library_and_returns_two_targets() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100, 8))
    y = np.column_stack([np.abs(x[:, 0]) + 0.5, np.abs(x[:, 1]) + 1.0])
    model = KNNPathModel(n_neighbors=8).fit(x, y)
    pred = model.predict(x[:3])
    assert pred.shape == (3, 2)
    assert np.all(pred >= 0)


def test_online_evaluator_compares_candidate_with_locked_baseline() -> None:
    evaluator = OnlinePathEvaluator()
    evaluator.update(
        actual_mfe=2.0, actual_mae=1.0,
        candidate_mfe=1.9, candidate_mae=1.1,
        baseline_mfe=1.5, baseline_mae=1.5,
    )
    snapshot = evaluator.snapshot()
    assert snapshot["completed"] == 1
    assert snapshot["mfe_ratio"] is not None
    assert snapshot["mae_ratio"] is not None


def test_contracts_are_immutable_and_validate_domain() -> None:
    event = GtgEvent(
        event_id="evt-1",
        event_time=datetime(2026, 9, 24, tzinfo=UTC),
        side="down",
        price=3750.0,
        atr=12.0,
    )
    with pytest.raises(ValidationError):
        event.price = 1.0

    prediction = EventPrediction(
        event_id=event.event_id,
        recorded_at=event.event_time,
        candidate_version="gtg-next-v1",
        mfe_atr=1.2, mae_atr=0.8,
        baseline_mfe_atr=1.0, baseline_mae_atr=1.0,
    )
    assert prediction.event_id == "evt-1"


def test_feature_engine_rejects_unsorted_input() -> None:
    idx = pd.to_datetime(["2026-01-02", "2026-01-01"], utc=True)
    frame = pd.DataFrame(
        {"open": [1, 1], "high": [2, 2], "low": [0.5, 0.5], "close": [1, 1]},
        index=idx,
    )
    with pytest.raises(ValueError, match="monotonic"):
        FeatureEngine().transform(frame)
