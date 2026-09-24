from __future__ import annotations

import numpy as np

from benchmarks.gtg_event_v3.path_diagnostic import (
    fit_ridge,
    knn_median_path,
    path_metrics,
    predict_ridge,
    select_candidate,
)


def test_ridge_path_recovers_simple_mapping():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 6)).astype(np.float32)
    w = rng.normal(size=(6, 3))
    y = x @ w
    y[:, 0] = np.abs(y[:, 0]) + 0.2
    y[:, 1] = np.abs(y[:, 1]) + 0.1
    y[:, 2] = np.tanh(y[:, 2])

    model = fit_ridge(x, y, alpha=0.1, log_positive_targets=False)
    pred = predict_ridge(model, x)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    assert (pred[:, :2] >= 0).all()


def test_knn_median_path_is_finite_and_bounded():
    train_embedding = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]],
        dtype=np.float32,
    )
    train_path = np.asarray(
        [[1.0, 2.0, 0.5], [1.2, 2.2, 0.6], [3.0, 4.0, -0.5], [3.2, 4.2, -0.6]],
        dtype=np.float32,
    )
    query = np.asarray([[1.0, 0.05], [-1.0, -0.05]], dtype=np.float32)
    pred = knn_median_path(train_embedding, train_path, query, k=2)
    assert pred.shape == (2, 3)
    assert np.isfinite(pred).all()
    assert pred[0, 0] < pred[1, 0]


def test_path_metrics_and_selection_use_worst_baseline_ratio():
    train = np.asarray(
        [[1.0, 2.0, 0.0], [1.2, 2.2, 0.1], [0.8, 1.8, -0.1]],
        dtype=np.float32,
    )
    truth = np.asarray(
        [[1.1, 2.1, 0.0], [1.3, 2.3, 0.1], [0.9, 1.9, -0.1]],
        dtype=np.float32,
    )
    good = truth.copy()
    bad = np.asarray([[3.0, 5.0, 0.0]] * len(truth), dtype=np.float32)

    gm = path_metrics(good, truth, train)
    bm = path_metrics(bad, truth, train)
    assert gm["beats_both_path_baselines"] is True
    assert gm["worst_ratio_to_baseline"] < bm["worst_ratio_to_baseline"]

    selected = select_candidate({"bad": bm, "good": gm})
    assert selected == "good"
