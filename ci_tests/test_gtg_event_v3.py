from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.modules.xau.gtg_event_v3_data import (
    EVENT_CODE_BULL,
    FEATURE_SCHEMA_VERSION,
    LABEL_VERSION,
    assert_closed_bar_alignment,
    build_event_labels,
    build_multitimeframe_features,
)
from src.platform.marketdata.xau_models import XAUBar, XAUTimeframe


def _m1_bars(n: int = 13_200) -> list[XAUBar]:
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    rows: list[XAUBar] = []
    price = 2600.0
    for i in range(n):
        drift = 0.025 * math.sin(i / 37.0) + 0.012 * math.sin(i / 11.0)
        o = price
        c = price + drift
        h = max(o, c) + 0.08 + 0.01 * (i % 3)
        l = min(o, c) - 0.08 - 0.01 * (i % 2)
        rows.append(
            XAUBar(
                timestamp=start + timedelta(minutes=i),
                timeframe=XAUTimeframe.M1,
                open=o,
                high=h,
                low=l,
                close=c,
                volume=100.0 + (i % 17),
                source="test",
                execution_eligible=False,
            )
        )
        price = c
    return rows


def _bucket(ts: datetime, tf: XAUTimeframe) -> datetime:
    if tf is XAUTimeframe.M5:
        return ts.replace(minute=(ts.minute // 5) * 5, second=0, microsecond=0)
    if tf is XAUTimeframe.M15:
        return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)
    if tf is XAUTimeframe.H1:
        return ts.replace(minute=0, second=0, microsecond=0)
    raise ValueError(tf)


def _resample(rows: list[XAUBar], tf: XAUTimeframe) -> list[XAUBar]:
    groups: dict[datetime, list[XAUBar]] = defaultdict(list)
    for row in rows:
        groups[_bucket(row.timestamp, tf)].append(row)
    out = []
    for ts in sorted(groups):
        bucket = groups[ts]
        out.append(
            XAUBar(
                timestamp=ts,
                timeframe=tf,
                open=bucket[0].open,
                high=max(x.high for x in bucket),
                low=min(x.low for x in bucket),
                close=bucket[-1].close,
                volume=sum(float(x.volume or 0.0) for x in bucket),
                source="test-resample",
                execution_eligible=False,
            )
        )
    return out


@pytest.fixture(scope="module")
def feature_fixture():
    m1 = _m1_bars()
    m5 = _resample(m1, XAUTimeframe.M5)
    m15 = _resample(m1, XAUTimeframe.M15)
    h1 = _resample(m1, XAUTimeframe.H1)
    return m1, m5, m15, h1


def test_feature_schema_and_closed_bar_alignment(feature_fixture):
    features = build_multitimeframe_features(*feature_fixture)
    assert features.matrix.shape[0] == len(feature_fixture[1])
    assert features.matrix.shape[1] == len(features.feature_names)
    assert len(set(features.feature_names)) == len(features.feature_names)
    assert "1m_rsi_slope3" in features.feature_names
    assert "5m_dist_ma1000_atr" in features.feature_names
    assert "1h_dist_ma200_atr" in features.feature_names
    assert np.isfinite(features.matrix[-96:]).all()
    assert_closed_bar_alignment(features)


def test_future_minutes_and_unclosed_h1_bar_cannot_change_past_features(feature_fixture):
    m1, _m5, _m15, _h1 = feature_fixture
    decision_time = m1[0].timestamp + timedelta(hours=205, minutes=35)

    mutated: list[XAUBar] = []
    for row in m1:
        if decision_time <= row.timestamp < decision_time.replace(minute=0) + timedelta(hours=1):
            bump = 75.0
            mutated.append(
                XAUBar(
                    timestamp=row.timestamp,
                    timeframe=row.timeframe,
                    open=row.open + bump,
                    high=row.high + bump,
                    low=row.low + bump,
                    close=row.close + bump,
                    volume=float(row.volume or 0.0) * 5.0,
                    source=row.source,
                    execution_eligible=False,
                )
            )
        else:
            mutated.append(row)

    base = build_multitimeframe_features(
        m1,
        _resample(m1, XAUTimeframe.M5),
        _resample(m1, XAUTimeframe.M15),
        _resample(m1, XAUTimeframe.H1),
    )
    changed = build_multitimeframe_features(
        mutated,
        _resample(mutated, XAUTimeframe.M5),
        _resample(mutated, XAUTimeframe.M15),
        _resample(mutated, XAUTimeframe.H1),
    )
    pos = base.times.index(decision_time)
    np.testing.assert_allclose(
        base.matrix[pos],
        changed.matrix[pos],
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_same_bar_target_and_adverse_touch_is_ambiguous_and_masked():
    n = 1_100
    close = np.full(n, 100.0, dtype=np.float64)
    high = np.full(n, 100.2, dtype=np.float64)
    low = np.full(n, 99.8, dtype=np.float64)
    atr = np.ones(n, dtype=np.float64)
    ma14 = np.full(n, 99.8, dtype=np.float64)
    ma50 = np.full(n, 99.7, dtype=np.float64)
    ma200 = np.full(n, 102.0, dtype=np.float64)

    i = 1000
    close[i - 3] = 99.5
    ma14[i - 3] = 99.6
    ma50[i - 3] = 99.7
    close[i] = 100.0
    high[i + 1] = 102.1
    low[i + 1] = 98.9

    raw = {
        "open": close.copy(),
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n),
        "atr": atr,
        "rsi": np.full(n, 50.0),
        "ma14": ma14,
        "ma22": np.full(n, 99.75),
        "ma50": ma50,
        "ma200": ma200,
        "ma1000": np.full(n, 105.0),
    }
    labels = build_event_labels(raw)
    assert labels.event_code[i] == EVENT_CODE_BULL
    assert labels.ambiguous[i] == 1.0
    assert labels.success_mask[i] == 0.0
    assert labels.adverse_mask[i] == 0.0


def _metadata(architecture: str = "gru") -> dict:
    return {
        "model_version": "gtg-event-conditioned-directional-v3",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_version": LABEL_VERSION,
        "dataset_id": "test-dataset",
        "training_range": ["2024-09-23", "2025-07-23"],
        "validation_range": ["2025-07-23", "2025-09-23"],
        "oos_range": ["2025-09-23", "2026-09-23"],
        "git_commit": "test",
        "seed": 260924,
        "normalization": {"method": "test"},
        "calibration": {"success_temperature": 1.0},
        "architecture": architecture,
        "threshold_policy": {"selected_on": "validation_only"},
    }


def test_model_registry_bundle_cpu_parity_nan_and_schema_rejection(tmp_path):
    torch = pytest.importorskip("torch")
    from src.modules.xau.gtg_event_v3 import (
        GTGEventConfig,
        GTGEventEngine,
        build_expert_model,
    )

    torch.manual_seed(7)
    cfg = GTGEventConfig(
        feature_count=7,
        sequence_length=32,
        hidden_size=16,
        embedding_size=12,
        transformer_heads=4,
        transformer_layers=1,
        tcn_channels=16,
        tcn_layers=2,
    )
    x = torch.randn(3, 32, 7)
    for model_type in ("gru", "tcn", "patch_transformer"):
        model = build_expert_model(model_type, cfg)
        out = model(x)
        assert out["success"].shape == (3,)
        assert out["adverse"].shape == (3,)
        assert out["path"].shape == (3, 3)
        assert out["time"].shape == (3,)
        assert out["embedding"].shape == (3, 12)

    model = build_expert_model("gru", cfg)
    engine = GTGEventEngine(
        model,
        feature_mean=[0.0] * 7,
        feature_std=[1.0] * 7,
        device="cpu",
    )
    sequence = [[0.01 * (r + c) for c in range(7)] for r in range(32)]
    p1 = engine.predict(sequence)
    p2 = engine.predict(sequence)
    assert p1 == p2
    assert 0.0 <= p1.success_probability <= 1.0
    assert 0.0 <= p1.adverse_first_probability <= 1.0
    assert p1.success_probability + (1.0 - p1.success_probability) == pytest.approx(1.0)

    path = tmp_path / "expert.pt"
    engine.save_bundle(path, metadata=_metadata())
    loaded, metadata = GTGEventEngine.load_bundle(path, device="cpu")
    p3 = loaded.predict(sequence)
    assert metadata["feature_schema_version"] == FEATURE_SCHEMA_VERSION
    assert p3.success_probability == pytest.approx(p1.success_probability, abs=1e-8)
    assert p3.expected_mfe_atr == pytest.approx(p1.expected_mfe_atr, abs=1e-8)

    with pytest.raises(ValueError, match="runtime feature schema"):
        GTGEventEngine.load_bundle(
            path,
            device="cpu",
            expected_feature_schema_version="wrong-schema",
        )

    bad = [row[:] for row in sequence]
    bad[-1][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        loaded.predict(bad)

    missing, error = GTGEventEngine.safe_load_bundle(
        tmp_path / "missing.pt",
        device="cpu",
    )
    assert missing is None
    assert error


def test_pr_auc_is_compatible_with_current_numpy():
    from benchmarks.gtg_event_v3.run import pr_auc_score

    y = np.asarray([0, 1, 0, 1, 1], dtype=np.float64)
    p = np.asarray([0.1, 0.9, 0.2, 0.8, 0.7], dtype=np.float64)
    score = pr_auc_score(y, p)
    assert score is not None
    assert 0.0 <= score <= 1.0


def test_up_and_down_experts_are_independent_not_complements():
    torch = pytest.importorskip("torch")
    from src.modules.xau.gtg_event_v3 import (
        GTGEventConfig,
        GTGEventEngine,
        build_expert_model,
    )

    cfg = GTGEventConfig(
        feature_count=4,
        sequence_length=16,
        hidden_size=8,
        embedding_size=8,
        transformer_heads=2,
        transformer_layers=1,
        tcn_channels=8,
        tcn_layers=1,
    )
    up = build_expert_model("gru", cfg)
    down = build_expert_model("gru", cfg)
    with torch.no_grad():
        for model, probability in ((up, 0.31), (down, 0.28)):
            for parameter in model.parameters():
                parameter.zero_()
            logit = math.log(probability / (1.0 - probability))
            model.success[-1].bias.fill_(logit)

    seq = [[0.0] * 4 for _ in range(16)]
    up_p = GTGEventEngine(
        up, feature_mean=[0.0] * 4, feature_std=[1.0] * 4
    ).predict(seq).success_probability
    down_p = GTGEventEngine(
        down, feature_mean=[0.0] * 4, feature_std=[1.0] * 4
    ).predict(seq).success_probability

    assert up_p == pytest.approx(0.31, abs=1e-5)
    assert down_p == pytest.approx(0.28, abs=1e-5)
    assert up_p + down_p < 1.0
