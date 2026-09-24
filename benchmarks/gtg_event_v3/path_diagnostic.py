"""Post-OOS path-model diagnostic for GTG Event v3.

This module deliberately does NOT alter or promote the original v3 benchmark.
The original OOS year has already been observed, so every result emitted here is
labeled diagnostic-only and requires a fresh forward holdout before promotion.

The diagnostic reuses the frozen v3 event grammar, champions and train-only
normalization, then compares simple path estimators selected on validation only:
- original neural multi-task path head
- ridge regression on frozen champion embeddings
- cosine k-nearest-neighbour median path retrieval
- session/volatility bucket medians

The selection target mirrors the original promotion rule: both MFE and MAE must
beat a train-median baseline. Candidate selection minimizes the worst validation
MAE ratio to that baseline; OOS is never used to choose a candidate.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmarks.gen1_gold_2y_real_v1 import run as legacy
from benchmarks.gtg_event_v3 import run as base
from src.modules.xau.gtg_event_v3 import GTGEventEngine
from src.modules.xau.gtg_event_v3_data import (
    EVENT_BEAR,
    EVENT_BULL,
    EVENT_CODE_BEAR,
    EVENT_CODE_BULL,
    build_event_labels,
    build_multitimeframe_features,
    session_name,
)
from src.platform.marketdata.xau_models import XAUTimeframe


BASE_OUT = Path(os.getenv("GTG_V3_OUT", "artifacts/gtg_event_v3"))
OUT = Path(os.getenv("GTG_V3_PATH_OUT", "artifacts/gtg_event_v3_path_diagnostic"))
RIDGE_ALPHAS = tuple(
    float(x)
    for x in os.getenv("GTG_V3_PATH_RIDGE_ALPHAS", "0.1,1,10,100").split(",")
)
KNN_K = tuple(
    int(x)
    for x in os.getenv("GTG_V3_PATH_KNN_K", "8,16,32,64").split(",")
)


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float64), std.astype(np.float64)


def fit_ridge(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
    *,
    log_positive_targets: bool,
) -> dict[str, np.ndarray | float | bool]:
    """Fit deterministic multi-output ridge using training data only."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x_mean, x_std = _standardize_fit(x)
    xs = (x - x_mean) / x_std

    yt = y.copy()
    if log_positive_targets:
        yt[:, :2] = np.log1p(np.clip(yt[:, :2], 0.0, None))

    y_mean = np.mean(yt, axis=0)
    yc = yt - y_mean
    gram = xs.T @ xs
    reg = np.eye(gram.shape[0], dtype=np.float64) * float(alpha)
    weights = np.linalg.solve(gram + reg, xs.T @ yc)
    return {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "weights": weights,
        "alpha": float(alpha),
        "log_positive_targets": bool(log_positive_targets),
    }


def predict_ridge(model: dict[str, Any], x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    xs = (x - model["x_mean"]) / model["x_std"]
    pred = xs @ model["weights"] + model["y_mean"]
    pred = np.asarray(pred, dtype=np.float64)
    if bool(model["log_positive_targets"]):
        pred[:, :2] = np.expm1(pred[:, :2])
    pred[:, :2] = np.clip(pred[:, :2], 0.0, None)
    pred[:, 2] = np.clip(pred[:, 2], -2.0, 2.0)
    return pred.astype(np.float32)


def knn_median_path(
    train_embedding: np.ndarray,
    train_path: np.ndarray,
    query_embedding: np.ndarray,
    k: int,
) -> np.ndarray:
    """Cosine KNN median retrieval with no query-time label access."""

    bank = np.asarray(train_embedding, dtype=np.float32).copy()
    query = np.asarray(query_embedding, dtype=np.float32).copy()
    bank /= np.linalg.norm(bank, axis=1, keepdims=True).clip(min=1e-8)
    query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    k = max(1, min(int(k), len(bank)))
    sim = query @ bank.T
    top = np.argpartition(sim, -k, axis=1)[:, -k:]
    pred = np.median(np.asarray(train_path)[top], axis=1)
    return pred.astype(np.float32)


def _vol_bucket(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def bucket_median_path(
    train_path: np.ndarray,
    train_indices: np.ndarray,
    query_indices: np.ndarray,
    *,
    times,
    raw_feature_matrix: np.ndarray,
    rv_col: int,
    q1: float,
    q2: float,
    min_bucket: int = 12,
) -> np.ndarray:
    """Training-only median conditioned on UTC session and volatility bucket."""

    global_med = np.median(train_path, axis=0)
    groups: dict[tuple[str, str], list[np.ndarray]] = {}
    for row, idx in zip(train_path, train_indices):
        key = (
            session_name(times[int(idx)]),
            _vol_bucket(float(raw_feature_matrix[int(idx), rv_col]), q1, q2),
        )
        groups.setdefault(key, []).append(row)

    medians: dict[tuple[str, str], np.ndarray] = {}
    for key, rows in groups.items():
        if len(rows) >= min_bucket:
            medians[key] = np.median(np.asarray(rows), axis=0)

    out = []
    for idx in query_indices:
        key = (
            session_name(times[int(idx)]),
            _vol_bucket(float(raw_feature_matrix[int(idx), rv_col]), q1, q2),
        )
        out.append(medians.get(key, global_med))
    return np.asarray(out, dtype=np.float32)


def path_metrics(
    pred: np.ndarray,
    truth: np.ndarray,
    train_path: np.ndarray,
) -> dict[str, float | bool]:
    pred = np.asarray(pred, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    med = np.median(np.asarray(train_path, dtype=np.float64), axis=0)

    mfe = float(np.mean(np.abs(pred[:, 0] - truth[:, 0])))
    mae = float(np.mean(np.abs(pred[:, 1] - truth[:, 1])))
    trend = float(np.mean(np.abs(pred[:, 2] - truth[:, 2])))
    base_mfe = float(np.mean(np.abs(med[0] - truth[:, 0])))
    base_mae = float(np.mean(np.abs(med[1] - truth[:, 1])))
    mfe_ratio = mfe / max(base_mfe, 1e-12)
    mae_ratio = mae / max(base_mae, 1e-12)
    return {
        "mfe_mae_atr": mfe,
        "mae_mae_atr": mae,
        "trend_strength_mae": trend,
        "baseline_median_mfe_mae_atr": base_mfe,
        "baseline_median_mae_mae_atr": base_mae,
        "mfe_ratio_to_baseline": mfe_ratio,
        "mae_ratio_to_baseline": mae_ratio,
        "worst_ratio_to_baseline": max(mfe_ratio, mae_ratio),
        "mean_ratio_to_baseline": 0.5 * (mfe_ratio + mae_ratio),
        "beats_both_path_baselines": bool(mfe_ratio < 1.0 and mae_ratio < 1.0),
    }


def select_candidate(
    validation_metrics: dict[str, dict[str, float | bool]]
) -> str:
    """Select on validation only; optimize the original both-head requirement."""

    if not validation_metrics:
        raise ValueError("no path candidates")
    return min(
        validation_metrics,
        key=lambda name: (
            float(validation_metrics[name]["worst_ratio_to_baseline"]),
            float(validation_metrics[name]["mean_ratio_to_baseline"]),
            name,
        ),
    )


def _collect(model, x, labels, indices, device: str) -> dict[str, np.ndarray]:
    ds = base.EventDataset(x, labels, indices)
    loader = DataLoader(
        ds,
        batch_size=base.BATCH * 2,
        shuffle=False,
        num_workers=0,
    )
    return base.collect_expert(model, loader, device)


def _candidate_predictions(
    train: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    *,
    times,
    x_raw: np.ndarray,
    rv_col: int,
    q1: float,
    q2: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    valid_pred: dict[str, np.ndarray] = {"native_neural": valid["path_pred"]}
    test_pred: dict[str, np.ndarray] = {"native_neural": test["path_pred"]}

    for log_targets in (False, True):
        mode = "log" if log_targets else "raw"
        for alpha in RIDGE_ALPHAS:
            name = f"ridge_embedding_{mode}_a{alpha:g}"
            ridge = fit_ridge(
                train["embedding"],
                train["path_y"],
                alpha,
                log_positive_targets=log_targets,
            )
            valid_pred[name] = predict_ridge(ridge, valid["embedding"])
            test_pred[name] = predict_ridge(ridge, test["embedding"])

    for k in KNN_K:
        name = f"knn_embedding_median_k{k}"
        valid_pred[name] = knn_median_path(
            train["embedding"], train["path_y"], valid["embedding"], k
        )
        test_pred[name] = knn_median_path(
            train["embedding"], train["path_y"], test["embedding"], k
        )

    valid_pred["session_vol_bucket_median"] = bucket_median_path(
        train["path_y"],
        train["indices"],
        valid["indices"],
        times=times,
        raw_feature_matrix=x_raw,
        rv_col=rv_col,
        q1=q1,
        q2=q2,
    )
    test_pred["session_vol_bucket_median"] = bucket_median_path(
        train["path_y"],
        train["indices"],
        test["indices"],
        times=times,
        raw_feature_matrix=x_raw,
        rv_col=rv_col,
        q1=q1,
        q2=q2,
    )
    return valid_pred, test_pred


def main() -> None:
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    base.seed_all()

    m1, _quotes, dataset = legacy.download_dataset(base.START, base.TEST_END)
    m5 = legacy.resample(m1, XAUTimeframe.M5)
    m15 = legacy.resample(m1, XAUTimeframe.M15)
    h1 = legacy.resample(m1, XAUTimeframe.H1)

    features = build_multitimeframe_features(m1, m5, m15, h1)
    labels = build_event_labels(features.m5_raw)
    x_raw = features.matrix
    times = features.times
    x, _mean, _std = base.normalize_train_only(
        x_raw, times, base.START, base.TRAIN_END
    )

    rv_col = features.feature_names.index("5m_rv48")
    all_train = base.split_indices(
        times, x, labels, base.START, base.TRAIN_END
    )
    train_rv = x_raw[all_train, rv_col]
    q1, q2 = [float(v) for v in np.quantile(train_rv, [1 / 3, 2 / 3])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    side_specs = (
        ("up", EVENT_CODE_BULL, EVENT_BULL),
        ("down", EVENT_CODE_BEAR, EVENT_BEAR),
    )
    report: dict[str, Any] = {
        "benchmark": "gtg-event-v3-path-post-oos-diagnostic",
        "status": "diagnostic_only_not_promotion_evidence",
        "selection_uses_oos": False,
        "original_oos_already_observed": True,
        "fresh_forward_holdout_required_for_promotion": True,
        "dataset_id": base.DATASET_ID,
        "base_benchmark_output": str(BASE_OUT),
        "git_commit": base.GIT_COMMIT,
        "candidates": {
            "ridge_alphas": list(RIDGE_ALPHAS),
            "knn_k": list(KNN_K),
            "ridge_target_modes": ["raw", "log_positive_mfe_mae"],
            "bucket_conditioning": "utc_session_x_train_volatility_tercile",
        },
        "experts": {},
    }

    for side, code, event in side_specs:
        bundle = BASE_OUT / f"gtg_event_v3_{side}_expert.pt"
        engine, metadata = GTGEventEngine.load_bundle(bundle, device=device)
        train_idx = base.split_indices(
            times, x, labels, base.START, base.TRAIN_END, side=code
        )
        valid_idx = base.split_indices(
            times, x, labels, base.TRAIN_END, base.VALID_END, side=code
        )
        test_idx = base.split_indices(
            times, x, labels, base.VALID_END, base.TEST_END, side=code
        )

        train = _collect(engine.model, x, labels, train_idx, device)
        valid = _collect(engine.model, x, labels, valid_idx, device)
        test = _collect(engine.model, x, labels, test_idx, device)

        valid_pred, test_pred = _candidate_predictions(
            train,
            valid,
            test,
            times=times,
            x_raw=x_raw,
            rv_col=rv_col,
            q1=q1,
            q2=q2,
        )
        validation_metrics = {
            name: path_metrics(pred, valid["path_y"], train["path_y"])
            for name, pred in valid_pred.items()
        }
        selected = select_candidate(validation_metrics)
        oos_metrics = {
            name: path_metrics(pred, test["path_y"], train["path_y"])
            for name, pred in test_pred.items()
        }

        report["experts"][side] = {
            "event": event,
            "architecture_frozen_from_v3": metadata["architecture"],
            "sample_counts": {
                "train": int(len(train_idx)),
                "validation": int(len(valid_idx)),
                "diagnostic_oos": int(len(test_idx)),
            },
            "selected_on": "validation_only",
            "selected_candidate": selected,
            "selected_validation_metrics": validation_metrics[selected],
            "selected_diagnostic_oos_metrics": oos_metrics[selected],
            "native_neural_validation_metrics": validation_metrics["native_neural"],
            "native_neural_diagnostic_oos_metrics": oos_metrics["native_neural"],
            "validation_all_candidates": validation_metrics,
            "diagnostic_oos_all_candidates": oos_metrics,
            "promotion_allowed_from_this_report": False,
        }

    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    path = OUT / "report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
