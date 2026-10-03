from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.modules.xau.gtg_directional_data import (
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CLASS_UP,
    FEATURE_NAMES,
    build_directional_features,
    build_directional_labels,
)


@dataclass
class _Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def _bars(n: int = 1300) -> list[_Bar]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows = []
    price = 2600.0
    for i in range(n):
        drift = 0.12 * math.sin(i / 17.0) + 0.04
        o = price
        c = price + drift
        h = max(o, c) + 0.35
        l = min(o, c) - 0.30
        rows.append(_Bar(start + timedelta(minutes=5 * i), o, h, l, c, 100 + i % 11))
        price = c
    return rows


def test_feature_schema_is_stable_and_causal_shape():
    x, raw, times = build_directional_features(_bars())
    assert x.shape[1] == len(FEATURE_NAMES)
    assert x.shape[0] == len(times) == len(raw["close"])
    assert np.isfinite(x[-96:]).all()


def test_directional_labels_are_three_class_and_mask_ambiguous():
    bars = _bars()
    x, raw, _times = build_directional_features(bars)
    labels = build_directional_labels(raw)
    assert labels.direction.shape[1] == 4
    assert set(np.unique(labels.direction)).issubset(
        {CLASS_DOWN, CLASS_NEUTRAL, CLASS_UP}
    )
    assert labels.ambiguous.shape == labels.direction_mask.shape
    # Ambiguous labels must never be trainable.
    assert np.all(labels.direction_mask[labels.ambiguous > 0.5] == 0.0)
    assert x.shape[0] == labels.direction.shape[0]


def test_model_registry_forward_shapes_and_bundle(tmp_path):
    torch = pytest.importorskip("torch")
    from src.modules.xau.gtg_directional_v2 import (
        GTGDirectionalConfig,
        GTGDirectionalEngine,
        build_directional_model,
    )

    cfg = GTGDirectionalConfig(
        feature_count=7,
        sequence_length=32,
        horizons_minutes=(30, 60, 120, 240),
        hidden_size=16,
        embedding_size=12,
        transformer_heads=4,
        transformer_layers=1,
        timesnet_top_k=2,
        timesnet_blocks=1,
    )
    x = torch.randn(3, 32, 7)

    for model_type in ("gru", "patch_transformer", "timesnet_lite"):
        model = build_directional_model(model_type, cfg)
        out = model(x)
        assert out["embedding"].shape == (3, 12)
        assert out["path"].shape == (3, 3)
        assert out["time"].shape == (3, 8)
        for horizon in cfg.horizons_minutes:
            assert out["direction"][str(horizon)].shape == (3, 3)

        engine = GTGDirectionalEngine(
            model,
            feature_mean=[0.0] * 7,
            feature_std=[1.0] * 7,
            temperatures={str(h): 1.0 for h in cfg.horizons_minutes},
        )
        path = tmp_path / f"{model_type}.pt"
        engine.save_bundle(path, metadata={"test": True})
        loaded, metadata = GTGDirectionalEngine.load_bundle(path)
        pred = loaded.predict([[0.0] * 7 for _ in range(32)])
        assert metadata["test"] is True
        assert pred.model_type == model_type
        for horizon in cfg.horizons_minutes:
            probs = pred.direction_probabilities[str(horizon)]
            assert abs(sum(probs.values()) - 1.0) < 1e-5
            assert set(probs) == {"down", "neutral", "up"}
