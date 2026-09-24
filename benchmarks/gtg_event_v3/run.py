"""GTG Event-Conditioned Directional Experience v3 benchmark.

This benchmark is intentionally event-conditioned and shadow-only.

Protocol:
- frozen Dukascopy XAUUSD dataset
- train 2024-09-23 -> 2025-07-23
- validation/calibration 2025-07-23 -> 2025-09-23
- untouched OOS 2025-09-23 -> 2026-09-23
- architecture and thresholds selected on validation only
- independent bullish and bearish experts
- independent TRADEABLE/ABSTAIN gate
- event-conditioned analog memory
- no order authority, no sizing authority, no stop authority
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from benchmarks.gen1_gold_2y_real_v1 import run as legacy
from src.modules.xau.gtg_event_v3 import (
    GTGEventConfig,
    GTGEventEngine,
    build_expert_model,
    build_tradeability_model,
    save_tradeability_bundle,
)
from src.modules.xau.gtg_event_v3_data import (
    EVENT_BEAR,
    EVENT_BULL,
    EVENT_CODE_BEAR,
    EVENT_CODE_BULL,
    FEATURE_SCHEMA_VERSION,
    HORIZON_BARS,
    LABEL_VERSION,
    MIN_EVENT_GAP_BARS,
    RESEARCH_ADVERSE_ATR,
    assert_closed_bar_alignment,
    build_event_labels,
    build_multitimeframe_features,
    event_name,
    session_name,
)
from src.platform.marketdata.xau_models import XAUTimeframe


UTC = timezone.utc
START = datetime.fromisoformat(os.getenv("GTG_V3_START", "2024-09-23")).replace(tzinfo=UTC)
TRAIN_END = datetime.fromisoformat(os.getenv("GTG_V3_TRAIN_END", "2025-07-23")).replace(tzinfo=UTC)
VALID_END = datetime.fromisoformat(os.getenv("GTG_V3_VALID_END", "2025-09-23")).replace(tzinfo=UTC)
TEST_END = datetime.fromisoformat(os.getenv("GTG_V3_TEST_END", "2026-09-23")).replace(tzinfo=UTC)
OUT = Path(os.getenv("GTG_V3_OUT", "artifacts/gtg_event_v3"))
DATASET_ID = os.getenv(
    "GTG_V3_DATASET_ID",
    "gen1-xau-2y-dataset-v2-20240923-20260923",
)
GIT_COMMIT = os.getenv("GITHUB_SHA", "local-unknown")
SEED = int(os.getenv("GTG_V3_SEED", "260924"))
SEQ = int(os.getenv("GTG_V3_SEQUENCE", "96"))
BATCH = int(os.getenv("GTG_V3_BATCH", "192"))
EPOCHS = int(os.getenv("GTG_V3_EPOCHS", "5"))
LR = float(os.getenv("GTG_V3_LR", "0.001"))
WEIGHT_DECAY = float(os.getenv("GTG_V3_WEIGHT_DECAY", "0.0001"))
ANALOG_BANK = int(os.getenv("GTG_V3_ANALOG_BANK", "6000"))
ANALOG_K = int(os.getenv("GTG_V3_ANALOG_K", "15"))
MODEL_TYPES = tuple(
    x.strip()
    for x in os.getenv(
        "GTG_V3_MODELS",
        "gru,tcn,patch_transformer",
    ).split(",")
    if x.strip()
)


def seed_all(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EventDataset(Dataset):
    def __init__(self, x: np.ndarray, labels, indices: np.ndarray) -> None:
        self.x = x
        self.labels = labels
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        path = np.asarray(
            [
                self.labels.mfe_atr[i],
                self.labels.mae_atr[i],
                self.labels.trend_strength[i],
            ],
            dtype=np.float32,
        )
        return (
            torch.from_numpy(self.x[i - SEQ + 1:i + 1]),
            torch.tensor(self.labels.success[i], dtype=torch.float32),
            torch.tensor(self.labels.adverse_first[i], dtype=torch.float32),
            torch.from_numpy(path),
            torch.tensor(self.labels.time_to_target[i], dtype=torch.float32),
            torch.tensor(self.labels.time_mask[i], dtype=torch.float32),
            torch.tensor(self.labels.clean_path[i], dtype=torch.float32),
            i,
        )


class TradeabilityDataset(Dataset):
    def __init__(self, x: np.ndarray, labels, indices: np.ndarray) -> None:
        self.x = x
        self.labels = labels
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        side = float(self.labels.event_code[i])
        seq = self.x[i - SEQ + 1:i + 1]
        side_col = np.full((SEQ, 1), side, dtype=np.float32)
        seq = np.concatenate((seq, side_col), axis=1)
        return (
            torch.from_numpy(seq),
            torch.tensor(self.labels.clean_path[i], dtype=torch.float32),
            i,
        )


def split_indices(
    times,
    x: np.ndarray,
    labels,
    start: datetime,
    end: datetime,
    *,
    side: int | None = None,
) -> np.ndarray:
    guard = timedelta(minutes=HORIZON_BARS * 5)
    out = []
    for i in range(max(1000, SEQ - 1), len(times)):
        if side is None:
            if int(labels.event_code[i]) == 0:
                continue
        elif int(labels.event_code[i]) != int(side):
            continue
        t = times[i]
        if t < start or t + guard >= end:
            continue
        if labels.success_mask[i] < 0.5 or labels.ambiguous[i] > 0.5:
            continue
        seq = x[i - SEQ + 1:i + 1]
        if not np.isfinite(seq).all():
            continue
        out.append(i)
    return np.asarray(out, dtype=np.int64)


def normalize_train_only(
    x_raw: np.ndarray,
    times,
    start: datetime,
    end: datetime,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        i
        for i, t in enumerate(times)
        if start <= t < end and np.isfinite(x_raw[i]).all()
    ]
    if not rows:
        raise RuntimeError("no finite training rows for v3 normalization")
    train_rows = x_raw[np.asarray(rows, dtype=np.int64)]
    mean = train_rows.mean(axis=0).astype(np.float32)
    std = train_rows.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    x = ((x_raw - mean) / std).astype(np.float32)
    return x, mean, std


def sigmoid_np(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    best_t, best_brier = 1.0, float("inf")
    for t in np.linspace(0.50, 3.00, 51):
        p = sigmoid_np(logits / float(t))
        brier = float(np.mean((p - y) ** 2))
        if brier < best_brier:
            best_t, best_brier = float(t), brier
    return best_t


def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    npos = int(y.sum())
    nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    sp = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sp[end] == sp[start]:
            end += 1
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float(
        (ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    )


def pr_auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    positives = int(y.sum())
    if positives == 0:
        return None
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    recall = tp / positives
    precision = tp / np.maximum(1, tp + fp)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.trapz(precision, recall))


def mcc_binary(y: np.ndarray, pred: np.ndarray) -> float:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = float(np.sum(y & pred))
    tn = float(np.sum((~y) & (~pred)))
    fp = float(np.sum((~y) & pred))
    fn = float(np.sum(y & (~pred)))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / den) if den > 1e-12 else 0.0


def ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if sel.any():
            total += float(sel.mean()) * abs(
                float(y[sel].mean()) - float(p[sel].mean())
            )
    return float(total)


def binary_metrics(y: np.ndarray, p: np.ndarray, train_prior: float) -> dict[str, Any]:
    y = y.astype(np.float64)
    pred = p >= 0.5
    tp = float(np.sum((y == 1) & pred))
    tn = float(np.sum((y == 0) & (~pred)))
    fp = float(np.sum((y == 0) & pred))
    fn = float(np.sum((y == 1) & (~pred)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = float(np.mean((p - y) ** 2))
    base_brier = float(np.mean((float(train_prior) - y) ** 2))
    return {
        "count": int(len(y)),
        "base_rate": float(y.mean()) if len(y) else None,
        "accuracy": float(np.mean(pred == (y >= 0.5))) if len(y) else None,
        "balanced_accuracy": (recall + specificity) / 2.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc_binary(y, pred),
        "roc_auc": auc_score(y, p),
        "pr_auc": pr_auc_score(y, p),
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_skill": base_brier - brier,
        "ece": ece_binary(y, p),
        "confusion": {
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)
        },
    }


def path_prediction(raw: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            nn.functional.softplus(raw[:, 0]),
            nn.functional.softplus(raw[:, 1]),
            torch.tanh(raw[:, 2]) * 2.0,
        ),
        dim=1,
    )


def expert_loss(out, success, adverse, path, time_y, time_mask):
    success_loss = nn.functional.binary_cross_entropy_with_logits(
        out["success"], success
    )
    adverse_loss = nn.functional.binary_cross_entropy_with_logits(
        out["adverse"], adverse
    )
    pp = path_prediction(out["path"])
    path_loss = nn.functional.smooth_l1_loss(pp, path, beta=0.5)
    tp = torch.sigmoid(out["time"])
    valid = time_mask > 0.5
    if valid.any():
        time_loss = nn.functional.smooth_l1_loss(
            tp[valid], time_y[valid], beta=0.2
        )
    else:
        time_loss = torch.tensor(0.0, device=success.device)
    total = success_loss + 0.35 * adverse_loss + 0.20 * path_loss + 0.08 * time_loss
    return total, {
        "success": float(success_loss.detach().cpu()),
        "adverse": float(adverse_loss.detach().cpu()),
        "path": float(path_loss.detach().cpu()),
        "time": float(time_loss.detach().cpu()),
    }


def collect_expert(model, loader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    rows: dict[str, list[np.ndarray]] = defaultdict(list)
    with torch.no_grad():
        for xb, sb, ab, pb, tb, tmb, cb, ib in loader:
            out = model(xb.to(device=device, dtype=torch.float32))
            rows["success_logits"].append(out["success"].cpu().numpy())
            rows["adverse_logits"].append(out["adverse"].cpu().numpy())
            rows["path_pred"].append(path_prediction(out["path"]).cpu().numpy())
            rows["time_pred"].append(torch.sigmoid(out["time"]).cpu().numpy())
            rows["embedding"].append(out["embedding"].cpu().numpy())
            rows["success_y"].append(sb.numpy())
            rows["adverse_y"].append(ab.numpy())
            rows["path_y"].append(pb.numpy())
            rows["time_y"].append(tb.numpy())
            rows["time_mask"].append(tmb.numpy())
            rows["clean_y"].append(cb.numpy())
            rows["indices"].append(ib.numpy())
    return {
        k: np.concatenate(v) if v else np.asarray([])
        for k, v in rows.items()
    }


def validation_model_score(collected: dict[str, np.ndarray], train_prior: float) -> float:
    p = sigmoid_np(collected["success_logits"])
    m = binary_metrics(collected["success_y"], p, train_prior)
    auc = float(m["roc_auc"] or 0.5)
    return float(auc + 1.5 * m["brier_skill"] + 0.25 * m["mcc"])


def train_expert(
    model_type: str,
    config: GTGEventConfig,
    train_loader,
    valid_loader,
    train_prior: float,
    device: str,
):
    seed_all()
    model = build_expert_model(model_type, config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best_score = -1e9
    stale = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        parts: dict[str, list[float]] = defaultdict(list)
        for xb, sb, ab, pb, tb, tmb, _cb, _ib in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            sb = sb.to(device=device, dtype=torch.float32)
            ab = ab.to(device=device, dtype=torch.float32)
            pb = pb.to(device=device, dtype=torch.float32)
            tb = tb.to(device=device, dtype=torch.float32)
            tmb = tmb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss, loss_parts = expert_loss(out, sb, ab, pb, tb, tmb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            for k, v in loss_parts.items():
                parts[k].append(v)
        val = collect_expert(model, valid_loader, device)
        score = validation_model_score(val, train_prior)
        row = {
            "epoch": epoch,
            "loss": fmean(losses),
            "success_loss": fmean(parts["success"]),
            "adverse_loss": fmean(parts["adverse"]),
            "path_loss": fmean(parts["path"]),
            "time_loss": fmean(parts["time"]),
            "validation_score_raw": score,
        }
        history.append(row)
        print(json.dumps({"phase": "expert_train", "model": model_type, **row}), flush=True)
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
    return model, history


def train_tradeability(
    config: GTGEventConfig,
    train_loader,
    valid_loader,
    train_prior: float,
    device: str,
):
    seed_all(SEED + 77)
    model = build_tradeability_model("gru", config).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = None
    best = -1e9
    history = []
    stale = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb, _ib in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                out["tradeable"], yb
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        c = collect_tradeability(model, valid_loader, device)
        p = sigmoid_np(c["logits"])
        m = binary_metrics(c["y"], p, train_prior)
        score = float((m["roc_auc"] or 0.5) + m["brier_skill"])
        row = {"epoch": epoch, "loss": fmean(losses), "validation_score_raw": score}
        history.append(row)
        if score > best + 1e-4:
            best = score
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
    return model, history


def collect_tradeability(model, loader, device: str) -> dict[str, np.ndarray]:
    model.eval()
    logits, ys, idx = [], [], []
    with torch.no_grad():
        for xb, yb, ib in loader:
            out = model(xb.to(device=device, dtype=torch.float32))
            logits.append(out["tradeable"].cpu().numpy())
            ys.append(yb.numpy())
            idx.append(ib.numpy())
    return {
        "logits": np.concatenate(logits),
        "y": np.concatenate(ys),
        "indices": np.concatenate(idx).astype(np.int64),
    }


def calibrated_expert_metrics(
    collected: dict[str, np.ndarray],
    success_temp: float,
    adverse_temp: float,
    success_prior: float,
    adverse_prior: float,
) -> dict[str, Any]:
    sp = sigmoid_np(collected["success_logits"] / success_temp)
    ap = sigmoid_np(collected["adverse_logits"] / adverse_temp)
    return {
        "success": binary_metrics(collected["success_y"], sp, success_prior),
        "adverse_first": binary_metrics(collected["adverse_y"], ap, adverse_prior),
    }


def path_metrics(
    collected: dict[str, np.ndarray],
    train_path: np.ndarray,
) -> dict[str, Any]:
    pred = collected["path_pred"]
    truth = collected["path_y"]
    med = np.median(train_path, axis=0)
    result = {
        "mfe_mae_atr": float(np.mean(np.abs(pred[:, 0] - truth[:, 0]))),
        "mae_mae_atr": float(np.mean(np.abs(pred[:, 1] - truth[:, 1]))),
        "trend_strength_mae": float(np.mean(np.abs(pred[:, 2] - truth[:, 2]))),
        "baseline_median_mfe_mae_atr": float(np.mean(np.abs(med[0] - truth[:, 0]))),
        "baseline_median_mae_mae_atr": float(np.mean(np.abs(med[1] - truth[:, 1]))),
    }
    valid = collected["time_mask"] > 0.5
    result["time_to_target_mae_minutes"] = (
        float(
            np.mean(
                np.abs(
                    collected["time_pred"][valid] - collected["time_y"][valid]
                )
            )
            * HORIZON_BARS
            * 5
        )
        if valid.any()
        else None
    )
    return result


def _vol_bucket(value: float, q1: float, q2: float) -> str:
    if value <= q1:
        return "low"
    if value <= q2:
        return "mid"
    return "high"


def analog_probabilities(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    train_idx: np.ndarray,
    query_emb: np.ndarray,
    query_idx: np.ndarray,
    *,
    times,
    raw_feature_matrix: np.ndarray,
    rv_col: int,
    vol_quantiles: tuple[float, float],
) -> np.ndarray:
    if not len(query_idx) or not len(train_idx):
        return np.full(len(query_idx), 0.5, dtype=np.float32)

    rng = np.random.default_rng(SEED)
    if len(train_idx) > ANALOG_BANK:
        keep = np.sort(
            rng.choice(len(train_idx), size=ANALOG_BANK, replace=False)
        )
        train_emb = train_emb[keep]
        train_y = train_y[keep]
        train_idx = train_idx[keep]

    bank = train_emb.astype(np.float32).copy()
    query = query_emb.astype(np.float32).copy()
    bank /= np.linalg.norm(bank, axis=1, keepdims=True).clip(min=1e-8)
    query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)

    q1, q2 = vol_quantiles
    bank_groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    query_groups: dict[tuple[str, str], list[int]] = defaultdict(list)

    for pos, idx in enumerate(train_idx):
        key = (
            session_name(times[int(idx)]),
            _vol_bucket(float(raw_feature_matrix[int(idx), rv_col]), q1, q2),
        )
        bank_groups[key].append(pos)
    for pos, idx in enumerate(query_idx):
        key = (
            session_name(times[int(idx)]),
            _vol_bucket(float(raw_feature_matrix[int(idx), rv_col]), q1, q2),
        )
        query_groups[key].append(pos)

    out = np.full(len(query_idx), float(np.mean(train_y)), dtype=np.float32)
    all_bank = np.arange(len(train_idx), dtype=np.int64)
    for key, q_positions in query_groups.items():
        b_positions = np.asarray(bank_groups.get(key) or [], dtype=np.int64)
        if len(b_positions) < 5:
            b_positions = all_bank
        k = min(ANALOG_K, len(b_positions))
        if k <= 0:
            continue
        qp = np.asarray(q_positions, dtype=np.int64)
        sim = query[qp] @ bank[b_positions].T
        top_local = np.argpartition(sim, -k, axis=1)[:, -k:]
        labels = train_y[b_positions[top_local]]
        # Beta(1,1) smoothing prevents pathological 0/1 analog probabilities.
        out[qp] = (labels.sum(axis=1) + 1.0) / (k + 2.0)
    return out


def choose_blend_weight(
    y: np.ndarray,
    neural: np.ndarray,
    analog: np.ndarray,
) -> dict[str, float]:
    best = {"analog_weight": 0.0, "brier": float(np.mean((neural - y) ** 2))}
    for w in (0.0, 0.25, 0.50):
        p = (1.0 - w) * neural + w * analog
        brier = float(np.mean((p - y) ** 2))
        if brier < best["brier"] - 1e-9:
            best = {"analog_weight": float(w), "brier": brier}
    return best


def choose_gate(
    y: np.ndarray,
    success_p: np.ndarray,
    trade_p: np.ndarray,
    train_prior: float,
) -> dict[str, Any]:
    best = None
    for s_thr in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        for t_thr in (0.40, 0.50, 0.60, 0.70):
            selected = (success_p >= s_thr) & (trade_p >= t_thr)
            coverage = float(selected.mean())
            if coverage < 0.10:
                continue
            precision = float(y[selected].mean())
            lift = precision - float(train_prior)
            utility = lift * math.sqrt(coverage)
            row = {
                "success_threshold": float(s_thr),
                "tradeability_threshold": float(t_thr),
                "coverage": coverage,
                "precision": precision,
                "precision_lift_vs_train_base": lift,
                "validation_utility": utility,
            }
            if best is None or row["validation_utility"] > best["validation_utility"]:
                best = row
    return best or {
        "success_threshold": 1.0,
        "tradeability_threshold": 1.0,
        "coverage": 0.0,
        "precision": None,
        "precision_lift_vs_train_base": None,
        "validation_utility": None,
    }


def selective_metrics(
    y: np.ndarray,
    success_p: np.ndarray,
    trade_p: np.ndarray,
    gate: dict[str, Any],
    train_prior: float,
) -> dict[str, Any]:
    selected = (
        (success_p >= float(gate["success_threshold"]))
        & (trade_p >= float(gate["tradeability_threshold"]))
    )
    coverage = float(selected.mean()) if len(selected) else 0.0
    if not selected.any():
        return {
            "selected": 0,
            "coverage": coverage,
            "no_trade_rate": 1.0 - coverage,
            "precision": None,
            "precision_lift_vs_train_base": None,
        }
    precision = float(y[selected].mean())
    return {
        "selected": int(selected.sum()),
        "coverage": coverage,
        "no_trade_rate": 1.0 - coverage,
        "precision": precision,
        "precision_lift_vs_train_base": precision - float(train_prior),
    }


def index_probability_map(indices: np.ndarray, values: np.ndarray) -> dict[int, float]:
    return {int(i): float(v) for i, v in zip(indices, values)}


def probabilities_for(indices: np.ndarray, mapping: dict[int, float]) -> np.ndarray:
    return np.asarray([mapping[int(i)] for i in indices], dtype=np.float64)


def stability_report(
    y: np.ndarray,
    p: np.ndarray,
    trade_p: np.ndarray,
    indices: np.ndarray,
    times,
    train_prior: float,
    gate: dict[str, Any],
) -> dict[str, Any]:
    monthly: dict[str, list[int]] = defaultdict(list)
    sessions: dict[str, list[int]] = defaultdict(list)
    for pos, idx in enumerate(indices):
        monthly[times[int(idx)].strftime("%Y-%m")].append(pos)
        sessions[session_name(times[int(idx)])].append(pos)

    def rows(groups):
        out = {}
        for key, positions in sorted(groups.items()):
            arr = np.asarray(positions, dtype=np.int64)
            if len(arr) < 10:
                continue
            out[key] = {
                "metrics": binary_metrics(y[arr], p[arr], train_prior),
                "selective": selective_metrics(
                    y[arr], p[arr], trade_p[arr], gate, train_prior
                ),
            }
        return out

    return {"monthly": rows(monthly), "sessions": rows(sessions)}


def volatility_breakdown(
    y: np.ndarray,
    p: np.ndarray,
    trade_p: np.ndarray,
    indices: np.ndarray,
    raw_feature_matrix: np.ndarray,
    rv_col: int,
    q1: float,
    q2: float,
    train_prior: float,
    gate: dict[str, Any],
) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for pos, idx in enumerate(indices):
        groups[_vol_bucket(float(raw_feature_matrix[int(idx), rv_col]), q1, q2)].append(pos)
    out = {}
    for key, positions in sorted(groups.items()):
        arr = np.asarray(positions, dtype=np.int64)
        if not len(arr):
            continue
        out[key] = {
            "count": int(len(arr)),
            "metrics": binary_metrics(y[arr], p[arr], train_prior),
            "selective": selective_metrics(
                y[arr], p[arr], trade_p[arr], gate, train_prior
            ),
        }
    return out


def bundle_metadata(
    *,
    architecture: str,
    calibration: dict[str, Any],
    threshold_policy: dict[str, Any],
    normalization: dict[str, Any],
    event: str,
) -> dict[str, Any]:
    return {
        "model_version": "gtg-event-conditioned-directional-v3",
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "label_version": LABEL_VERSION,
        "dataset_id": DATASET_ID,
        "training_range": [START.isoformat(), TRAIN_END.isoformat()],
        "validation_range": [TRAIN_END.isoformat(), VALID_END.isoformat()],
        "oos_range": [VALID_END.isoformat(), TEST_END.isoformat()],
        "git_commit": GIT_COMMIT,
        "seed": SEED,
        "normalization": normalization,
        "calibration": calibration,
        "architecture": architecture,
        "threshold_policy": threshold_policy,
        "event": event,
        "decision_authority": False,
    }


def main() -> None:
    seed_all()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    m1, _quotes, dataset = legacy.download_dataset(START, TEST_END)
    m5 = legacy.resample(m1, XAUTimeframe.M5)
    m15 = legacy.resample(m1, XAUTimeframe.M15)
    h1 = legacy.resample(m1, XAUTimeframe.H1)

    features = build_multitimeframe_features(m1, m5, m15, h1)
    assert_closed_bar_alignment(features)
    labels = build_event_labels(features.m5_raw)
    x_raw = features.matrix
    times = features.times
    x, mean, std = normalize_train_only(x_raw, times, START, TRAIN_END)

    rv_name = "5m_rv48"
    rv_col = features.feature_names.index(rv_name)

    all_train = split_indices(times, x, labels, START, TRAIN_END)
    all_valid = split_indices(times, x, labels, TRAIN_END, VALID_END)
    all_test = split_indices(times, x, labels, VALID_END, TEST_END)
    if min(len(all_train), len(all_valid), len(all_test)) == 0:
        raise RuntimeError("one or more v3 splits have zero event samples")

    train_rv = x_raw[all_train, rv_col]
    q1, q2 = [float(v) for v in np.quantile(train_rv, [1 / 3, 2 / 3])]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GTGEventConfig(
        feature_count=x.shape[1],
        sequence_length=SEQ,
        hidden_size=64,
        embedding_size=48,
        dropout=0.10,
        patch_length=8,
        patch_stride=4,
        transformer_heads=4,
        transformer_layers=2,
        tcn_channels=64,
        tcn_layers=4,
    )

    # Independent abstention engine on the union of UP/DOWN events.
    regime_cfg = GTGEventConfig(
        feature_count=x.shape[1] + 1,
        sequence_length=SEQ,
        hidden_size=64,
        embedding_size=48,
        dropout=0.10,
        patch_length=8,
        patch_stride=4,
        transformer_heads=4,
        transformer_layers=2,
        tcn_channels=64,
        tcn_layers=4,
    )
    regime_train_ds = TradeabilityDataset(x, labels, all_train)
    regime_valid_ds = TradeabilityDataset(x, labels, all_valid)
    regime_test_ds = TradeabilityDataset(x, labels, all_test)
    regime_train_loader = DataLoader(regime_train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    regime_valid_loader = DataLoader(regime_valid_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
    regime_test_loader = DataLoader(regime_test_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
    regime_prior = float(labels.clean_path[all_train].mean())
    regime_model, regime_history = train_tradeability(
        regime_cfg,
        regime_train_loader,
        regime_valid_loader,
        regime_prior,
        device,
    )
    regime_valid = collect_tradeability(regime_model, regime_valid_loader, device)
    regime_test = collect_tradeability(regime_model, regime_test_loader, device)
    regime_temp = fit_temperature(regime_valid["logits"], regime_valid["y"])
    regime_valid_p = sigmoid_np(regime_valid["logits"] / regime_temp)
    regime_test_p = sigmoid_np(regime_test["logits"] / regime_temp)
    regime_valid_map = index_probability_map(regime_valid["indices"], regime_valid_p)
    regime_test_map = index_probability_map(regime_test["indices"], regime_test_p)
    regime_report = {
        "architecture": "gru",
        "target": "clean_path_to_ma200_without_1atr_adverse_first",
        "history": regime_history,
        "temperature": regime_temp,
        "validation": binary_metrics(regime_valid["y"], regime_valid_p, regime_prior),
        "oos": binary_metrics(regime_test["y"], regime_test_p, regime_prior),
        "parameter_count": int(sum(p.numel() for p in regime_model.parameters())),
    }

    side_specs = (
        ("up", EVENT_CODE_BULL, EVENT_BULL),
        ("down", EVENT_CODE_BEAR, EVENT_BEAR),
    )
    experts_report: dict[str, Any] = {}
    eligible_experts: list[str] = []

    for side_key, side_code, event_label in side_specs:
        train_idx = split_indices(times, x, labels, START, TRAIN_END, side=side_code)
        valid_idx = split_indices(times, x, labels, TRAIN_END, VALID_END, side=side_code)
        test_idx = split_indices(times, x, labels, VALID_END, TEST_END, side=side_code)
        if min(len(train_idx), len(valid_idx), len(test_idx)) < 20:
            experts_report[side_key] = {
                "event": event_label,
                "status": "insufficient_samples",
                "sample_counts": {
                    "train": len(train_idx),
                    "validation": len(valid_idx),
                    "oos": len(test_idx),
                },
            }
            continue

        train_ds = EventDataset(x, labels, train_idx)
        valid_ds = EventDataset(x, labels, valid_idx)
        test_ds = EventDataset(x, labels, test_idx)
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
        valid_loader = DataLoader(valid_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)

        success_prior = float(labels.success[train_idx].mean())
        adverse_prior = float(labels.adverse_first[train_idx].mean())
        model_rows: dict[str, Any] = {}
        models: dict[str, Any] = {}
        val_collected: dict[str, Any] = {}

        for model_type in MODEL_TYPES:
            model, history = train_expert(
                model_type, config, train_loader, valid_loader, success_prior, device
            )
            val = collect_expert(model, valid_loader, device)
            success_temp = fit_temperature(val["success_logits"], val["success_y"])
            adverse_temp = fit_temperature(val["adverse_logits"], val["adverse_y"])
            metrics = calibrated_expert_metrics(
                val, success_temp, adverse_temp, success_prior, adverse_prior
            )
            score = float(
                (metrics["success"]["roc_auc"] or 0.5)
                + 1.5 * metrics["success"]["brier_skill"]
                + 0.25 * metrics["success"]["mcc"]
            )
            model_rows[model_type] = {
                "history": history,
                "parameter_count": int(sum(p.numel() for p in model.parameters())),
                "success_temperature": success_temp,
                "adverse_temperature": adverse_temp,
                "validation_selection_score": score,
                "validation": metrics,
            }
            models[model_type] = model
            val_collected[model_type] = val

        champion = max(
            MODEL_TYPES,
            key=lambda name: model_rows[name]["validation_selection_score"],
        )
        print(json.dumps({
            "phase": "champion_frozen",
            "side": side_key,
            "champion": champion,
            "basis": "validation_only",
        }), flush=True)

        # Evaluate all architectures on OOS only after champion is frozen.
        test_collected: dict[str, Any] = {}
        for model_type in MODEL_TYPES:
            test = collect_expert(models[model_type], test_loader, device)
            test_collected[model_type] = test
            model_rows[model_type]["oos"] = calibrated_expert_metrics(
                test,
                model_rows[model_type]["success_temperature"],
                model_rows[model_type]["adverse_temperature"],
                success_prior,
                adverse_prior,
            )
            train_eval = collect_expert(
                models[model_type],
                DataLoader(train_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0),
                device,
            )
            model_rows[model_type]["oos_path"] = path_metrics(
                test, train_eval["path_y"]
            )

        champ_model = models[champion]
        champ_val = val_collected[champion]
        champ_test = test_collected[champion]
        success_temp = float(model_rows[champion]["success_temperature"])
        adverse_temp = float(model_rows[champion]["adverse_temperature"])
        neural_val_p = sigmoid_np(champ_val["success_logits"] / success_temp)
        neural_test_p = sigmoid_np(champ_test["success_logits"] / success_temp)

        train_eval_loader = DataLoader(train_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
        champ_train = collect_expert(champ_model, train_eval_loader, device)

        analog_val_p = analog_probabilities(
            champ_train["embedding"],
            champ_train["success_y"],
            champ_train["indices"],
            champ_val["embedding"],
            champ_val["indices"],
            times=times,
            raw_feature_matrix=x_raw,
            rv_col=rv_col,
            vol_quantiles=(q1, q2),
        )
        blend_choice = choose_blend_weight(
            champ_val["success_y"], neural_val_p, analog_val_p
        )
        analog_test_p = analog_probabilities(
            champ_train["embedding"],
            champ_train["success_y"],
            champ_train["indices"],
            champ_test["embedding"],
            champ_test["indices"],
            times=times,
            raw_feature_matrix=x_raw,
            rv_col=rv_col,
            vol_quantiles=(q1, q2),
        )
        w = float(blend_choice["analog_weight"])
        final_val_p = (1.0 - w) * neural_val_p + w * analog_val_p
        final_test_p = (1.0 - w) * neural_test_p + w * analog_test_p

        side_valid_trade = probabilities_for(champ_val["indices"], regime_valid_map)
        side_test_trade = probabilities_for(champ_test["indices"], regime_test_map)
        gate = choose_gate(
            champ_val["success_y"],
            final_val_p,
            side_valid_trade,
            success_prior,
        )
        selective = selective_metrics(
            champ_test["success_y"],
            final_test_p,
            side_test_trade,
            gate,
            success_prior,
        )

        final_metrics = binary_metrics(
            champ_test["success_y"], final_test_p, success_prior
        )
        neural_metrics = binary_metrics(
            champ_test["success_y"], neural_test_p, success_prior
        )
        analog_metrics = binary_metrics(
            champ_test["success_y"], analog_test_p, success_prior
        )

        stability = stability_report(
            champ_test["success_y"],
            final_test_p,
            side_test_trade,
            champ_test["indices"],
            times,
            success_prior,
            gate,
        )
        vol_breakdown = volatility_breakdown(
            champ_test["success_y"],
            final_test_p,
            side_test_trade,
            champ_test["indices"],
            x_raw,
            rv_col,
            q1,
            q2,
            success_prior,
            gate,
        )
        positive_months = sum(
            1
            for row in stability["monthly"].values()
            if row["metrics"]["mcc"] > 0 and row["metrics"]["brier_skill"] > 0
        )
        total_months = len(stability["monthly"])
        path = model_rows[champion]["oos_path"]
        path_better = bool(
            path["mfe_mae_atr"] < path["baseline_median_mfe_mae_atr"]
            and path["mae_mae_atr"] < path["baseline_median_mae_mae_atr"]
        )
        precision = selective.get("precision")
        eligible = bool(
            final_metrics["brier_skill"] > 0
            and final_metrics["mcc"] >= 0.05
            and precision is not None
            and precision >= success_prior + 0.03
            and selective["coverage"] >= 0.10
            and positive_months >= max(4, total_months // 2)
            and path_better
        )
        if eligible:
            eligible_experts.append(side_key)

        calibration = {
            "success_temperature": success_temp,
            "adverse_temperature": adverse_temp,
            "analog_weight_selected_on_validation": w,
            "tradeability_temperature": regime_temp,
        }
        threshold_policy = {
            "selected_on": "validation_only",
            **gate,
        }
        metadata = bundle_metadata(
            architecture=champion,
            calibration=calibration,
            threshold_policy=threshold_policy,
            normalization={
                "method": "train-period-only_mean_std",
                "feature_count": int(len(mean)),
            },
            event=event_label,
        )
        engine = GTGEventEngine(
            champ_model,
            feature_mean=mean.tolist(),
            feature_std=std.tolist(),
            success_temperature=success_temp,
            adverse_temperature=adverse_temp,
            device=device,
        )
        engine.save_bundle(
            OUT / f"gtg_event_v3_{side_key}_expert.pt",
            metadata=metadata,
        )

        experts_report[side_key] = {
            "event": event_label,
            "sample_counts": {
                "train": int(len(train_idx)),
                "validation": int(len(valid_idx)),
                "oos": int(len(test_idx)),
            },
            "train_success_base_rate": success_prior,
            "train_adverse_first_base_rate": adverse_prior,
            "architectures": model_rows,
            "champion": champion,
            "champion_selected_on": "validation_only",
            "analog_memory": {
                "scope": "same_event_same_direction_then_same_session_volatility_bucket",
                "k": ANALOG_K,
                "bank_cap": ANALOG_BANK,
                "validation_blend_choice": blend_choice,
                "oos_neural_only": neural_metrics,
                "oos_analog_only": analog_metrics,
                "oos_validation_selected_blend": final_metrics,
            },
            "abstention": {
                "gate_from_validation": gate,
                "oos": selective,
            },
            "path": path,
            "stability": {
                **stability,
                "positive_mcc_and_brier_skill_months": positive_months,
                "total_months": total_months,
            },
            "volatility_regimes": vol_breakdown,
            "promotion": {
                "decision_authority": False,
                "eligible_for_deterministic_gtg_ablation": eligible,
                "path_beats_train_median_baseline": path_better,
            },
        }

    trade_meta = bundle_metadata(
        architecture="gru",
        calibration={"tradeability_temperature": regime_temp},
        threshold_policy={"selected_per_expert_on_validation": True},
        normalization={
            "method": "expert_train_normalized_plus_event_side_channel",
            "feature_count": int(len(mean) + 1),
        },
        event="bullish_and_bearish_ma14_ma50_to_ma200",
    )
    save_tradeability_bundle(
        OUT / "gtg_event_v3_tradeability.pt",
        regime_model,
        feature_mean=[0.0] * (len(mean) + 1),
        feature_std=[1.0] * (len(mean) + 1),
        temperature=regime_temp,
        metadata=trade_meta,
    )

    report = {
        "benchmark": "gtg-event-conditioned-directional-experience-v3",
        "mode": "shadow_only",
        "dataset": {
            **dataset,
            "dataset_id": DATASET_ID,
        },
        "git_commit": GIT_COMMIT,
        "seed": SEED,
        "split_policy": {
            "train": [START.isoformat(), TRAIN_END.isoformat()],
            "validation_calibration": [TRAIN_END.isoformat(), VALID_END.isoformat()],
            "untouched_oos": [VALID_END.isoformat(), TEST_END.isoformat()],
            "random_split": False,
            "future_leakage": False,
            "architecture_selection_uses_oos": False,
            "threshold_selection_uses_oos": False,
            "normalization_uses_oos": False,
        },
        "event_grammar": {
            "events": [EVENT_BULL, EVENT_BEAR],
            "fixed_destination": "MA200_at_event_timestamp",
            "destination_distance_atr": [0.5, 6.0],
            "ma_transition": "MA14_vs_MA50 geometry plus 3-bar slope sign",
            "horizon_minutes": HORIZON_BARS * 5,
            "research_adverse_barrier_atr": RESEARCH_ADVERSE_ATR,
            "research_adverse_barrier_is_stop_loss": False,
            "min_same_direction_event_gap_bars": MIN_EVENT_GAP_BARS,
        },
        "features": {
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "count": len(features.feature_names),
            "names": list(features.feature_names),
            "sequence_bars_m5": SEQ,
            "sequence_minutes": SEQ * 5,
            "timeframes": ["M1", "M5", "M15", "H1"],
            "h4_status": "deferred_initial_v3_to_limit_warmup_and_compute",
            "htf_policy": "closed_bar_only",
            "normalization": "train_period_only",
        },
        "labels": {
            "label_version": LABEL_VERSION,
            "success": "fixed_MA200_touched_within_4h",
            "mfe": "directional_MFE_in_ATR_until_target_or_horizon",
            "mae": "directional_MAE_in_ATR_until_target_or_horizon",
            "adverse_first": "1ATR_research_adverse_barrier_before_MA200",
            "time_to_target": "fraction_of_48_M5_bars",
            "clean_path": "target_hit_without_research_adverse_first",
        },
        "sample_counts_all_events": {
            "train": int(len(all_train)),
            "validation": int(len(all_valid)),
            "oos": int(len(all_test)),
        },
        "tradeability_engine": regime_report,
        "experts": experts_report,
        "historical_comparison": {
            "v1_ma200_destination": {
                "status": "promising_shadow",
                "oos_auc": 0.6292,
                "oos_brier_skill": 0.00590,
                "p_ge_70_precision": 0.7199,
                "p_ge_70_coverage": 0.0553,
            },
            "v2_generic_direction_60m": {
                "status": "not_promoted",
                "champion": "patch_transformer",
                "oos_balanced_accuracy": 0.4446,
                "oos_mcc": 0.0489,
                "oos_brier_skill": -0.02403,
                "selective_directional_accuracy": 0.4934,
                "positive_mcc_and_brier_skill_months": "0/13",
            },
        },
        "microstructure": {
            "included_in_historical_v3": False,
            "reason": "Dukascopy frozen dataset has no centralized CME L2/L3/aggressor history",
            "future_expert": "venue-qualified CME GC/MGC microstructure forward-test only",
        },
        "economic_features": {
            "included_in_initial_event_A_B": False,
            "reason": "no causally timestamped historical release feed is bundled with frozen two-year dataset",
            "policy": "never infer or backfill actual values before publication timestamps",
        },
        "self_supervised": {
            "included_in_initial_benchmark": False,
            "reason": "first isolate event-conditioning value with bounded native architectures; add masked/contrastive pretraining only as a separate ablation if supervised v3 shows OOS signal",
        },
        "promotion": {
            "decision_authority": False,
            "eligible_experts_for_next_deterministic_ablation": eligible_experts,
            "trading_ablation_run": False,
            "trading_ablation_reason": (
                "run only after an expert passes the explicit OOS/calibration/stability/path gate"
            ),
            "walk_forward_retraining_run": False,
            "walk_forward_reason": (
                "rolling retraining is the next robustness gate only for experts that first pass untouched OOS"
            ),
        },
        "limitations": [
            "No synthetic footprint, delta, CVD, or order-book heatmap is fabricated from OHLC.",
            "Event observations are separated by a 12-M5-bar same-direction cooldown but outcome horizons can still overlap across distinct/opposite events; counts are reported as event observations, not independent trades.",
            "Session partitions are descriptive UTC buckets, not execution rules.",
            "The 1 ATR adverse barrier is a research path label, not a stop-loss recommendation.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }

    (OUT / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    summary = [
        "# GTG Event-Conditioned Directional Experience v3",
        "",
        f"- Commit: {GIT_COMMIT}",
        f"- Dataset: {DATASET_ID}",
        f"- Features: {len(features.feature_names)} causal M1/M5/M15/H1 features",
        f"- All-event samples: train={len(all_train)} validation={len(all_valid)} oos={len(all_test)}",
        f"- Eligible experts for deterministic ablation: {eligible_experts}",
        "",
        "Shadow-only. GTG deterministic core remains authoritative.",
    ]
    for side in ("up", "down"):
        row = experts_report.get(side, {})
        if row.get("champion"):
            fm = row["analog_memory"]["oos_validation_selected_blend"]
            sel = row["abstention"]["oos"]
            summary.extend([
                "",
                f"## {side.upper()} expert",
                f"- Champion: {row['champion']}",
                f"- OOS ROC-AUC: {fm['roc_auc']}",
                f"- OOS MCC: {fm['mcc']:.4f}",
                f"- OOS Brier skill: {fm['brier_skill']:.6f}",
                f"- Selective coverage: {sel['coverage']:.4f}",
                f"- Selective precision: {sel['precision']}",
            ])
    (OUT / "SUMMARY.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(json.dumps({
        "phase": "gtg_v3_complete",
        "sample_counts": report["sample_counts_all_events"],
        "eligible_experts": eligible_experts,
        "experts": {
            side: {
                "champion": row.get("champion"),
                "oos": (
                    row.get("analog_memory", {}).get("oos_validation_selected_blend")
                ),
                "abstention": row.get("abstention", {}).get("oos"),
                "promotion": row.get("promotion"),
            }
            for side, row in experts_report.items()
        },
        "artifact_dir": str(OUT),
        "elapsed_seconds": report["elapsed_seconds"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
