"""GTG Directional Experience v2 benchmark.

Goals:
1. Learn UP, DOWN and NEUTRAL as first-class outcomes.
2. Learn the path: upward excursion, downward excursion and trend strength.
3. Compare multiple causal architectures under exactly the same labels/features.
4. Select the champion ONLY on validation data.
5. Open the untouched second year once for OOS evaluation.

This benchmark is shadow research.  It cannot route orders or alter GTG risk.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from benchmarks.gen1_gold_2y_real_v1 import run as legacy
from src.modules.xau.gtg_directional_data import (
    BARRIER_ATR,
    CLASS_DOWN,
    CLASS_NEUTRAL,
    CLASS_UP,
    DIRECTION_CLASSES,
    FEATURE_NAMES,
    HORIZON_BARS,
    DirectionalLabels,
    build_directional_features,
    build_directional_labels,
    class_distribution,
)
from src.modules.xau.gtg_directional_v2 import (
    GTGDirectionalConfig,
    GTGDirectionalEngine,
    build_directional_model,
)
from src.platform.marketdata.xau_models import XAUTimeframe

UTC = timezone.utc
START = datetime.fromisoformat(os.getenv("GTG_DIR_START", "2024-09-23")).replace(tzinfo=UTC)
TRAIN_END = datetime.fromisoformat(os.getenv("GTG_DIR_TRAIN_END", "2025-07-23")).replace(tzinfo=UTC)
VALID_END = datetime.fromisoformat(os.getenv("GTG_DIR_VALID_END", "2025-09-23")).replace(tzinfo=UTC)
TEST_END = datetime.fromisoformat(os.getenv("GTG_DIR_TEST_END", "2026-09-23")).replace(tzinfo=UTC)
OUT = Path(os.getenv("GTG_DIR_OUT", "artifacts/gtg_directional_v2"))

SEED = int(os.getenv("GTG_DIR_SEED", "260924"))
SEQ = int(os.getenv("GTG_DIR_SEQUENCE", "96"))
BATCH = int(os.getenv("GTG_DIR_BATCH", "256"))
EPOCHS = int(os.getenv("GTG_DIR_EPOCHS", "5"))
TRAIN_STRIDE = int(os.getenv("GTG_DIR_TRAIN_STRIDE", "4"))
VALID_STRIDE = int(os.getenv("GTG_DIR_VALID_STRIDE", "2"))
TEST_STRIDE = int(os.getenv("GTG_DIR_TEST_STRIDE", "2"))
LR = float(os.getenv("GTG_DIR_LR", "0.001"))
WEIGHT_DECAY = float(os.getenv("GTG_DIR_WEIGHT_DECAY", "0.0001"))
PRIMARY_HORIZON = int(os.getenv("GTG_DIR_PRIMARY_HORIZON", "60"))
MODEL_TYPES = tuple(
    s.strip()
    for s in os.getenv(
        "GTG_DIR_MODELS",
        "gru,patch_transformer,timesnet_lite",
    ).split(",")
    if s.strip()
)
HORIZONS = (30, 60, 120, 240)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class DirectionDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        labels: DirectionalLabels,
        indices: np.ndarray,
    ) -> None:
        self.x = x
        self.labels = labels
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        return (
            torch.from_numpy(self.x[i - SEQ + 1:i + 1]),
            torch.from_numpy(self.labels.direction[i]),
            torch.from_numpy(self.labels.direction_mask[i]),
            torch.from_numpy(self.labels.path[i]),
            torch.from_numpy(self.labels.path_mask[i]),
            torch.from_numpy(self.labels.time[i]),
            torch.from_numpy(self.labels.time_mask[i]),
            i,
        )


def period_indices(
    times: list[datetime],
    x: np.ndarray,
    labels: DirectionalLabels,
    start: datetime,
    end: datetime,
    stride: int,
) -> np.ndarray:
    guard = timedelta(minutes=max(HORIZONS))
    out: list[int] = []
    for i in range(max(1000, SEQ - 1), len(times), max(1, stride)):
        t = times[i]
        if t < start or t + guard >= end:
            continue
        seq = x[i - SEQ + 1:i + 1]
        if not np.isfinite(seq).all():
            continue
        if labels.direction_mask[i].sum() < len(HORIZONS):
            continue
        if labels.path_mask[i].sum() < 3:
            continue
        out.append(i)
    return np.asarray(out, dtype=np.int64)


def class_weights(labels: DirectionalLabels, indices: np.ndarray) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for hidx, hmin in enumerate(HORIZONS):
        valid = labels.direction_mask[indices, hidx] > 0.5
        y = labels.direction[indices[valid], hidx]
        counts = np.asarray([(y == c).sum() for c in range(3)], dtype=np.float64)
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / np.sqrt(counts / counts.sum())
        inv = inv / inv.mean()
        inv = np.clip(inv, 0.55, 1.80)
        result[str(hmin)] = torch.tensor(inv, dtype=torch.float32)
    return result


def _path_prediction(raw: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            nn.functional.softplus(raw[:, 0]),
            nn.functional.softplus(raw[:, 1]),
            raw[:, 2],
        ),
        dim=1,
    )


def loss_fn(
    outputs: dict[str, Any],
    direction: torch.Tensor,
    direction_mask: torch.Tensor,
    path: torch.Tensor,
    path_mask: torch.Tensor,
    time_y: torch.Tensor,
    time_mask: torch.Tensor,
    weights: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    device = direction.device
    direction_loss = torch.tensor(0.0, device=device)
    direction_terms = 0
    for hidx, hmin in enumerate(HORIZONS):
        valid = direction_mask[:, hidx] > 0.5
        if valid.any():
            w = weights[str(hmin)].to(device)
            direction_loss = direction_loss + nn.functional.cross_entropy(
                outputs["direction"][str(hmin)][valid],
                direction[valid, hidx].long(),
                weight=w,
            )
            direction_terms += 1
    direction_loss = direction_loss / max(1, direction_terms)

    pp = _path_prediction(outputs["path"])
    path_valid = path_mask > 0.5
    if path_valid.any():
        path_loss = nn.functional.smooth_l1_loss(
            pp[path_valid], path[path_valid], beta=0.5
        )
    else:
        path_loss = torch.tensor(0.0, device=device)

    tp = torch.sigmoid(outputs["time"]).reshape(
        -1, len(HORIZONS), 2
    )
    time_valid = time_mask > 0.5
    if time_valid.any():
        time_loss = nn.functional.smooth_l1_loss(
            tp[time_valid], time_y[time_valid], beta=0.2
        )
    else:
        time_loss = torch.tensor(0.0, device=device)

    total = direction_loss + 0.25 * path_loss + 0.08 * time_loss
    return total, {
        "direction": float(direction_loss.detach().cpu()),
        "path": float(path_loss.detach().cpu()),
        "time": float(time_loss.detach().cpu()),
    }


def collect(model, loader, device: str) -> dict[str, Any]:
    model.eval()
    logits = {str(h): [] for h in HORIZONS}
    y = []
    masks = []
    path_pred = []
    path_y = []
    time_pred = []
    time_y = []
    time_mask = []
    indices = []
    with torch.no_grad():
        for xb, db, mb, pb, pmb, tb, tmb, ib in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            out = model(xb)
            for h in HORIZONS:
                logits[str(h)].append(out["direction"][str(h)].cpu().numpy())
            y.append(db.numpy())
            masks.append(mb.numpy())
            path_pred.append(_path_prediction(out["path"]).cpu().numpy())
            path_y.append(pb.numpy())
            time_pred.append(
                torch.sigmoid(out["time"]).reshape(-1, len(HORIZONS), 2).cpu().numpy()
            )
            time_y.append(tb.numpy())
            time_mask.append(tmb.numpy())
            indices.append(ib.numpy())
    return {
        "logits": {k: np.concatenate(v) for k, v in logits.items()},
        "y": np.concatenate(y),
        "mask": np.concatenate(masks),
        "path_pred": np.concatenate(path_pred),
        "path_y": np.concatenate(path_y),
        "time_pred": np.concatenate(time_pred),
        "time_y": np.concatenate(time_y),
        "time_mask": np.concatenate(time_mask),
        "indices": np.concatenate(indices).astype(np.int64),
    }


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -50.0, 50.0))
    return e / np.sum(e, axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t = 1.0
    best = float("inf")
    onehot = np.eye(3, dtype=np.float64)[y.astype(np.int64)]
    for t in np.linspace(0.50, 3.00, 51):
        p = softmax_np(logits / float(t))
        brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
        if brier < best:
            best = brier
            best_t = float(t)
    return best_t


def confusion_matrix(y: np.ndarray, pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for a, b in zip(y.astype(int), pred.astype(int)):
        cm[a, b] += 1
    return cm


def multiclass_mcc(cm: np.ndarray) -> float:
    c = float(np.trace(cm))
    s = float(cm.sum())
    pk = cm.sum(axis=0).astype(np.float64)
    tk = cm.sum(axis=1).astype(np.float64)
    num = c * s - float(np.dot(pk, tk))
    den = math.sqrt(
        max(0.0, s * s - float(np.dot(pk, pk)))
        * max(0.0, s * s - float(np.dot(tk, tk)))
    )
    return float(num / den) if den > 1e-12 else 0.0


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    conf = p.max(axis=1)
    pred = p.argmax(axis=1)
    correct = (pred == y).astype(np.float64)
    ece = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if sel.any():
            ece += float(sel.mean()) * abs(
                float(correct[sel].mean()) - float(conf[sel].mean())
            )
    return float(ece)


def classification_metrics(
    y: np.ndarray,
    p: np.ndarray,
    baseline_prior: np.ndarray,
) -> dict[str, Any]:
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred)
    recalls = []
    precisions = []
    f1s = []
    per_class: dict[str, Any] = {}
    for idx, name in enumerate(DIRECTION_CLASSES):
        tp = float(cm[idx, idx])
        fn = float(cm[idx, :].sum() - cm[idx, idx])
        fp = float(cm[:, idx].sum() - cm[idx, idx])
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        per_class[name] = {
            "support": int(cm[idx, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    onehot = np.eye(3, dtype=np.float64)[y.astype(np.int64)]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    bp = np.broadcast_to(baseline_prior.reshape(1, 3), p.shape)
    baseline_brier = float(np.mean(np.sum((bp - onehot) ** 2, axis=1)))

    result = {
        "count": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "mcc": multiclass_mcc(cm),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": baseline_brier - brier,
        "ece": expected_calibration_error(y, p),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }
    conf = p.max(axis=1)
    for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        selected = conf >= threshold
        key = f"conf_ge_{int(threshold * 100)}"
        result[key + "_coverage"] = float(selected.mean())
        result[key + "_accuracy"] = (
            float((pred[selected] == y[selected]).mean()) if selected.any() else None
        )

        directional = selected & (pred != CLASS_NEUTRAL)
        result[key + "_directional_coverage"] = float(directional.mean())
        result[key + "_directional_precision"] = (
            float((pred[directional] == y[directional]).mean())
            if directional.any() else None
        )
    return result


def priors_for(labels: DirectionalLabels, indices: np.ndarray) -> dict[str, np.ndarray]:
    result = {}
    for hidx, h in enumerate(HORIZONS):
        y = labels.direction[indices, hidx]
        counts = np.asarray([(y == c).sum() for c in range(3)], dtype=np.float64)
        result[str(h)] = counts / counts.sum()
    return result


def calibrated_metrics(
    collected: dict[str, Any],
    temperatures: dict[str, float],
    priors: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics = {}
    probabilities = {}
    for hidx, h in enumerate(HORIZONS):
        key = str(h)
        y = collected["y"][:, hidx].astype(np.int64)
        p = softmax_np(collected["logits"][key] / temperatures[key])
        probabilities[key] = p
        metrics[key] = classification_metrics(y, p, priors[key])
    return metrics, probabilities


def path_metrics(collected: dict[str, Any]) -> dict[str, float]:
    pred = collected["path_pred"]
    truth = collected["path_y"]
    return {
        "up_mfe_mae_atr": float(np.mean(np.abs(pred[:, 0] - truth[:, 0]))),
        "down_mfe_mae_atr": float(np.mean(np.abs(pred[:, 1] - truth[:, 1]))),
        "trend_tstat_scaled_mae": float(np.mean(np.abs(pred[:, 2] - truth[:, 2]))),
    }


def validation_score(metrics: dict[str, Any]) -> float:
    rows = list(metrics.values())
    bal = fmean(float(r["balanced_accuracy"]) for r in rows)
    f1 = fmean(float(r["macro_f1"]) for r in rows)
    brier_skill = fmean(float(r["brier_skill"]) for r in rows)
    # Classification quality dominates; calibration skill is a small tie-breaker.
    return float(bal + 0.50 * f1 + 0.25 * brier_skill)


def early_stop_score(collected: dict[str, Any], priors: dict[str, np.ndarray]) -> float:
    hidx = HORIZONS.index(PRIMARY_HORIZON)
    key = str(PRIMARY_HORIZON)
    y = collected["y"][:, hidx].astype(np.int64)
    p = softmax_np(collected["logits"][key])
    m = classification_metrics(y, p, priors[key])
    return float(m["balanced_accuracy"] + 0.5 * m["macro_f1"] - 0.1 * m["brier"])


def choose_selective_threshold(
    y: np.ndarray,
    p: np.ndarray,
) -> dict[str, Any]:
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    best = None
    for threshold in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        selected = (conf >= threshold) & (pred != CLASS_NEUTRAL)
        coverage = float(selected.mean())
        if coverage < 0.02:
            continue
        accuracy = float((pred[selected] == y[selected]).mean())
        wrong_opposite = (
            ((pred[selected] == CLASS_UP) & (y[selected] == CLASS_DOWN))
            | ((pred[selected] == CLASS_DOWN) & (y[selected] == CLASS_UP))
        )
        opposite_rate = float(wrong_opposite.mean()) if selected.any() else 0.0
        utility = (accuracy - opposite_rate) * math.sqrt(coverage)
        row = {
            "threshold": threshold,
            "coverage": coverage,
            "directional_accuracy": accuracy,
            "opposite_rate": opposite_rate,
            "validation_utility": utility,
        }
        if best is None or row["validation_utility"] > best["validation_utility"]:
            best = row
    return best or {
        "threshold": 1.0,
        "coverage": 0.0,
        "directional_accuracy": None,
        "opposite_rate": None,
        "validation_utility": None,
    }


def apply_selective_threshold(
    y: np.ndarray,
    p: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = p.argmax(axis=1)
    conf = p.max(axis=1)
    selected = (conf >= threshold) & (pred != CLASS_NEUTRAL)
    if not selected.any():
        return {"count": 0, "coverage": 0.0}
    yp = y[selected]
    pp = pred[selected]
    correct = pp == yp
    opposite = (
        ((pp == CLASS_UP) & (yp == CLASS_DOWN))
        | ((pp == CLASS_DOWN) & (yp == CLASS_UP))
    )
    neutral = yp == CLASS_NEUTRAL
    r = np.where(correct, 1.0, np.where(opposite, -1.0, 0.0))
    return {
        "count": int(selected.sum()),
        "coverage": float(selected.mean()),
        "directional_accuracy": float(correct.mean()),
        "opposite_rate": float(opposite.mean()),
        "neutral_rate": float(neutral.mean()),
        "event_r_mean_before_costs": float(r.mean()),
        "event_r_sum_before_costs": float(r.sum()),
        "up_calls": int((pp == CLASS_UP).sum()),
        "down_calls": int((pp == CLASS_DOWN).sum()),
    }


def monthly_stability(
    y: np.ndarray,
    p: np.ndarray,
    indices: np.ndarray,
    times: list[datetime],
    baseline_prior: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for pos, idx in enumerate(indices):
        grouped[times[int(idx)].strftime("%Y-%m")].append(pos)
    out = {}
    for month, positions in sorted(grouped.items()):
        arr = np.asarray(positions, dtype=np.int64)
        m = classification_metrics(y[arr], p[arr], baseline_prior)
        s = apply_selective_threshold(y[arr], p[arr], threshold)
        out[month] = {
            "count": m["count"],
            "balanced_accuracy": m["balanced_accuracy"],
            "macro_f1": m["macro_f1"],
            "mcc": m["mcc"],
            "brier_skill": m["brier_skill"],
            "selective": s,
        }
    return out


def train_one(
    model_type: str,
    config: GTGDirectionalConfig,
    train_loader,
    valid_loader,
    weights,
    priors,
    device: str,
) -> tuple[Any, list[dict[str, Any]]]:
    seed_all(SEED)
    model = build_directional_model(model_type, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    history = []
    best_state = None
    best_score = -1e9
    stale = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        components: dict[str, list[float]] = defaultdict(list)
        for xb, db, mb, pb, pmb, tb, tmb, _ib in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            db = db.to(device)
            mb = mb.to(device)
            pb = pb.to(device=device, dtype=torch.float32)
            pmb = pmb.to(device)
            tb = tb.to(device=device, dtype=torch.float32)
            tmb = tmb.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            loss, parts = loss_fn(out, db, mb, pb, pmb, tb, tmb, weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            for k, v in parts.items():
                components[k].append(v)

        val = collect(model, valid_loader, device)
        score = early_stop_score(val, priors)
        row = {
            "epoch": epoch,
            "loss": fmean(losses),
            "direction_loss": fmean(components["direction"]),
            "path_loss": fmean(components["path"]),
            "time_loss": fmean(components["time"]),
            "validation_primary_score": score,
        }
        history.append(row)
        print(json.dumps({"phase": "train", "model": model_type, **row}), flush=True)

        if score > best_score + 1e-4:
            best_score = score
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= 2:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model, history


def main() -> None:
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    m1, _quotes, dataset = legacy.download_dataset(START, TEST_END)
    m5 = legacy.resample(m1, XAUTimeframe.M5)
    x_raw, raw, times = build_directional_features(m5)
    labels = build_directional_labels(raw, HORIZONS)

    train_idx = period_indices(times, x_raw, labels, START, TRAIN_END, TRAIN_STRIDE)
    valid_idx = period_indices(times, x_raw, labels, TRAIN_END, VALID_END, VALID_STRIDE)
    test_idx = period_indices(times, x_raw, labels, VALID_END, TEST_END, TEST_STRIDE)

    # Train-only normalization.
    train_rows = x_raw[train_idx]
    mean = np.mean(train_rows, axis=0).astype(np.float32)
    std = np.std(train_rows, axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    x = ((x_raw - mean) / std).astype(np.float32)

    train_ds = DirectionDataset(x, labels, train_idx)
    valid_ds = DirectionDataset(x, labels, valid_idx)
    test_ds = DirectionDataset(x, labels, test_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)

    weights = class_weights(labels, train_idx)
    priors = priors_for(labels, train_idx)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GTGDirectionalConfig(
        feature_count=len(FEATURE_NAMES),
        sequence_length=SEQ,
        horizons_minutes=HORIZONS,
        hidden_size=64,
        embedding_size=48,
        dropout=0.10,
        patch_length=8,
        patch_stride=4,
        transformer_heads=4,
        transformer_layers=2,
        timesnet_top_k=3,
        timesnet_blocks=2,
    )

    models: dict[str, Any] = {}
    reports: dict[str, Any] = {}

    for model_type in MODEL_TYPES:
        model, history = train_one(
            model_type,
            config,
            train_loader,
            valid_loader,
            weights,
            priors,
            device,
        )
        val = collect(model, valid_loader, device)
        temperatures = {}
        for hidx, h in enumerate(HORIZONS):
            key = str(h)
            temperatures[key] = fit_temperature(
                val["logits"][key],
                val["y"][:, hidx].astype(np.int64),
            )
        val_metrics, val_probs = calibrated_metrics(val, temperatures, priors)
        score = validation_score(val_metrics)
        models[model_type] = model
        reports[model_type] = {
            "history": history,
            "temperatures": temperatures,
            "validation": val_metrics,
            "validation_path": path_metrics(val),
            "validation_selection_score": score,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
        }
        print(json.dumps({
            "phase": "validation_complete",
            "model": model_type,
            "selection_score": score,
        }), flush=True)

    # Champion selection is frozen BEFORE any OOS model comparison is read.
    champion = max(
        MODEL_TYPES,
        key=lambda name: reports[name]["validation_selection_score"],
    )
    print(json.dumps({
        "phase": "champion_frozen",
        "champion": champion,
        "basis": "validation_only",
    }), flush=True)

    oos_collected: dict[str, Any] = {}
    oos_probs: dict[str, dict[str, np.ndarray]] = {}
    for model_type in MODEL_TYPES:
        model = models[model_type]
        test = collect(model, test_loader, device)
        oos_collected[model_type] = test
        metrics, probs = calibrated_metrics(
            test,
            reports[model_type]["temperatures"],
            priors,
        )
        oos_probs[model_type] = probs
        reports[model_type]["oos"] = metrics
        reports[model_type]["oos_path"] = path_metrics(test)

    # Selective confidence threshold is ALSO fit only on validation for champion.
    champion_model = models[champion]
    champion_val = collect(champion_model, valid_loader, device)
    _, champion_val_probs = calibrated_metrics(
        champion_val,
        reports[champion]["temperatures"],
        priors,
    )
    primary_idx = HORIZONS.index(PRIMARY_HORIZON)
    threshold_choice = choose_selective_threshold(
        champion_val["y"][:, primary_idx].astype(np.int64),
        champion_val_probs[str(PRIMARY_HORIZON)],
    )

    champion_test = oos_collected[champion]
    champion_test_probs = oos_probs[champion][str(PRIMARY_HORIZON)]
    oos_selective = apply_selective_threshold(
        champion_test["y"][:, primary_idx].astype(np.int64),
        champion_test_probs,
        float(threshold_choice["threshold"]),
    )
    monthly = monthly_stability(
        champion_test["y"][:, primary_idx].astype(np.int64),
        champion_test_probs,
        champion_test["indices"],
        times,
        priors[str(PRIMARY_HORIZON)],
        float(threshold_choice["threshold"]),
    )

    # Save champion model as a portable bundle.  It remains shadow-only.
    engine = GTGDirectionalEngine(
        champion_model,
        feature_mean=mean.tolist(),
        feature_std=std.tolist(),
        temperatures=reports[champion]["temperatures"],
        device=device,
    )
    engine.save_bundle(
        OUT / "gtg_directional_v2_champion.pt",
        metadata={
            "status": "shadow_research",
            "champion_selected_on": "validation_only",
            "primary_horizon_minutes": PRIMARY_HORIZON,
            "selective_threshold": threshold_choice,
            "training_period": [START.isoformat(), TRAIN_END.isoformat()],
            "validation_period": [TRAIN_END.isoformat(), VALID_END.isoformat()],
            "oos_period": [VALID_END.isoformat(), TEST_END.isoformat()],
            "feature_names": FEATURE_NAMES,
        },
    )

    distributions = {}
    for split_name, idx in (
        ("train", train_idx),
        ("validation", valid_idx),
        ("oos", test_idx),
    ):
        distributions[split_name] = {
            str(h): class_distribution(labels, idx, hidx)
            for hidx, h in enumerate(HORIZONS)
        }

    # Promotion gate is intentionally strict.  Passing it only means "eligible
    # for deterministic GTG ablation", not live trading.
    champ_oos = reports[champion]["oos"][str(PRIMARY_HORIZON)]
    months = list(monthly.values())
    positive_months = sum(
        1 for row in months
        if row["mcc"] > 0 and row["brier_skill"] > 0
    )
    promotion_eligible_for_ablation = bool(
        champ_oos["balanced_accuracy"] >= 0.40
        and champ_oos["mcc"] >= 0.08
        and champ_oos["brier_skill"] > 0
        and oos_selective.get("directional_accuracy", 0.0) >= 0.50
        and positive_months >= max(6, len(months) // 2)
    )

    report = {
        "benchmark": "gtg-directional-experience-v2",
        "mode": "shadow_only",
        "dataset": dataset,
        "split_policy": {
            "train": [START.isoformat(), TRAIN_END.isoformat()],
            "validation": [TRAIN_END.isoformat(), VALID_END.isoformat()],
            "untouched_oos": [VALID_END.isoformat(), TEST_END.isoformat()],
            "random_split": False,
            "future_leakage": False,
            "model_selection_uses_oos": False,
            "threshold_selection_uses_oos": False,
        },
        "label_policy": {
            "classes": DIRECTION_CLASSES,
            "horizons_minutes": HORIZONS,
            "barrier_atr": BARRIER_ATR,
            "same_bar_double_touch": "masked_ambiguous",
            "path_targets": [
                "up_mfe_atr_4h",
                "down_mfe_atr_4h",
                "future_trend_tstat_scaled_4h",
            ],
        },
        "features": {
            "count": len(FEATURE_NAMES),
            "names": FEATURE_NAMES,
            "sequence_bars": SEQ,
            "sequence_minutes": SEQ * 5,
            "normalization": "training_split_only",
        },
        "sample_counts": {
            "train": int(len(train_idx)),
            "validation": int(len(valid_idx)),
            "oos": int(len(test_idx)),
        },
        "class_distributions": distributions,
        "models": reports,
        "champion": {
            "model_type": champion,
            "selected_on": "validation_only",
            "primary_horizon_minutes": PRIMARY_HORIZON,
            "threshold_from_validation": threshold_choice,
            "oos_selective": oos_selective,
            "monthly_primary_horizon": monthly,
            "oos_primary": reports[champion]["oos"][str(PRIMARY_HORIZON)],
            "oos_path": reports[champion]["oos_path"],
        },
        "promotion": {
            "decision_authority": False,
            "eligible_for_deterministic_gtg_ablation": promotion_eligible_for_ablation,
            "positive_stability_months": positive_months,
            "total_oos_months": len(months),
            "next_gate": "integrate champion as shadow evidence into frozen GTG execution and compare PF/drawdown without changing risk rules",
        },
        "limitations": [
            "Historical centralized footprint/order-book/heatmap data is not present in the frozen Dukascopy dataset.",
            "Event-R is a directional barrier score before transaction costs, not trading PnL.",
            "Architecture comparison is limited to native GRU, patch Transformer and TimesNetLite implementations to keep the production dependency surface bounded.",
            "A two-year dataset is sufficient for a first OOS experiment, not proof of durable live edge.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "seed": SEED,
    }

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    (OUT / "SUMMARY.md").write_text(
        "\n".join([
            "# GTG Directional Experience v2",
            "",
            f"- Champion (validation-only): {champion}",
            f"- OOS primary horizon: {PRIMARY_HORIZON} minutes",
            f"- OOS balanced accuracy: {champ_oos['balanced_accuracy']:.4f}",
            f"- OOS macro F1: {champ_oos['macro_f1']:.4f}",
            f"- OOS MCC: {champ_oos['mcc']:.4f}",
            f"- OOS Brier skill: {champ_oos['brier_skill']:.6f}",
            f"- Selective threshold: {threshold_choice['threshold']}",
            f"- Selective OOS: {oos_selective}",
            f"- Eligible for deterministic GTG ablation: {promotion_eligible_for_ablation}",
            "",
            "Shadow research only. No live decision authority.",
        ]),
        encoding="utf-8",
    )

    print(json.dumps({
        "benchmark": report["benchmark"],
        "champion": report["champion"],
        "model_comparison": {
            name: {
                "validation_score": reports[name]["validation_selection_score"],
                "oos_primary": reports[name]["oos"][str(PRIMARY_HORIZON)],
                "oos_path": reports[name]["oos_path"],
            }
            for name in MODEL_TYPES
        },
        "promotion": report["promotion"],
        "sample_counts": report["sample_counts"],
        "elapsed_seconds": report["elapsed_seconds"],
        "artifact_dir": str(OUT),
    }, indent=2, sort_keys=True, default=str), flush=True)


if __name__ == "__main__":
    main()
