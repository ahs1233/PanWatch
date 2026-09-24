"""GTG Deep Experience v1 benchmark.

Train only on the first historical year, tune calibration on the final two months
of that year, and evaluate untouched on the second year.

The model is not promoted to decision authority. It runs in shadow mode and
learns probabilities plus a latent pattern embedding that can be compared with
historical analogs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
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
from src.modules.xau.gtg_experience import GTGExperienceConfig, GTGSequenceEncoder
from src.platform.marketdata.xau_models import XAUTimeframe

UTC = timezone.utc
START = datetime.fromisoformat(os.getenv("GTG_EXP_START", "2024-09-23")).replace(tzinfo=UTC)
TRAIN_END = datetime.fromisoformat(os.getenv("GTG_EXP_TRAIN_END", "2025-07-23")).replace(tzinfo=UTC)
VALID_END = datetime.fromisoformat(os.getenv("GTG_EXP_VALID_END", "2025-09-23")).replace(tzinfo=UTC)
TEST_END = datetime.fromisoformat(os.getenv("GTG_EXP_TEST_END", "2026-09-23")).replace(tzinfo=UTC)
OUT = Path(os.getenv("GTG_EXP_OUT", "artifacts/gtg_experience_v1"))
SEED = int(os.getenv("GTG_EXP_SEED", "260924"))
SEQ = int(os.getenv("GTG_EXP_SEQUENCE", "96"))
TRAIN_STRIDE = int(os.getenv("GTG_EXP_TRAIN_STRIDE", "4"))
VALID_STRIDE = int(os.getenv("GTG_EXP_VALID_STRIDE", "2"))
TEST_STRIDE = int(os.getenv("GTG_EXP_TEST_STRIDE", "2"))
EPOCHS = int(os.getenv("GTG_EXP_EPOCHS", "6"))
BATCH = int(os.getenv("GTG_EXP_BATCH", "256"))
LR = float(os.getenv("GTG_EXP_LR", "0.001"))
WEIGHT_DECAY = float(os.getenv("GTG_EXP_WEIGHT_DECAY", "0.0001"))
ANALOG_BANK = int(os.getenv("GTG_EXP_ANALOG_BANK", "2500"))
ANALOG_QUERIES = int(os.getenv("GTG_EXP_ANALOG_QUERIES", "6000"))

HEADS = (
    "next_move_up",
    "expansion",
    "ma200_touch",
    "ma1000_touch",
    "ma200_rejection",
    "ma1000_rejection",
)

FEATURE_NAMES = (
    "logret_1",
    "logret_3",
    "logret_12",
    "logret_24",
    "body_atr",
    "range_atr",
    "upper_wick_atr",
    "lower_wick_atr",
    "volume_z48",
    "rsi14_centered",
    "dist_ma14_atr",
    "dist_ma22_atr",
    "dist_ma50_atr",
    "dist_ma200_atr",
    "dist_ma1000_atr",
    "slope_ma14_atr",
    "slope_ma22_atr",
    "slope_ma50_atr",
    "slope_ma200_atr",
    "slope_ma1000_atr",
    "spacing_14_50_atr",
    "spacing_50_200_atr",
    "spacing_200_1000_atr",
    "session_sin",
    "session_cos",
)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(False)


def _ema(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    seed = float(np.mean(values[:period]))
    out[period - 1] = seed
    alpha = 2.0 / (period + 1.0)
    value = seed
    for i in range(period, len(values)):
        value += (float(values[i]) - value) * alpha
        out[i] = value
    return out


def _rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    if len(values) < period:
        return out
    cs = np.concatenate([[0.0], np.cumsum(np.nan_to_num(values, nan=0.0))])
    out[period - 1 :] = (cs[period:] - cs[:-period]) / period
    return out


def _rolling_std(values: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        out[i] = float(np.std(window))
    return out


def _rsi(values: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(values), np.nan, dtype=np.float64)
    delta = np.diff(values, prepend=values[0])
    gains = np.maximum(delta, 0.0)
    losses = np.maximum(-delta, 0.0)
    avg_gain = _rolling_mean(gains, period)
    avg_loss = _rolling_mean(losses, period)
    valid = np.isfinite(avg_gain) & np.isfinite(avg_loss)
    for i in np.where(valid)[0]:
        if avg_loss[i] <= 1e-12:
            out[i] = 100.0 if avg_gain[i] > 0 else 50.0
        else:
            rs = avg_gain[i] / avg_loss[i]
            out[i] = 100.0 - 100.0 / (1.0 + rs)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev), np.abs(low - prev)])
    return _rolling_mean(tr, period)


def build_features(m5) -> tuple[np.ndarray, dict[str, np.ndarray], list[datetime]]:
    times = [b.timestamp + timedelta(minutes=5) for b in m5]
    o = np.array([float(b.open) for b in m5], dtype=np.float64)
    h = np.array([float(b.high) for b in m5], dtype=np.float64)
    l = np.array([float(b.low) for b in m5], dtype=np.float64)
    c = np.array([float(b.close) for b in m5], dtype=np.float64)
    v = np.array([max(0.0, float(b.volume or 0.0)) for b in m5], dtype=np.float64)
    atr = _atr(h, l, c, 14)
    atr_safe = np.where(atr > 1e-9, atr, np.nan)

    mas = {p: _ema(c, p) for p in (14, 22, 50, 200, 1000)}
    rsi = _rsi(c, 14)

    feats: list[np.ndarray] = []
    for lag in (1, 3, 12, 24):
        prev = np.roll(c, lag)
        prev[:lag] = np.nan
        feats.append(np.log(c / prev))

    body = (c - o) / atr_safe
    rng = (h - l) / atr_safe
    upper = (h - np.maximum(o, c)) / atr_safe
    lower = (np.minimum(o, c) - l) / atr_safe
    feats.extend([body, rng, upper, lower])

    lv = np.log1p(v)
    vm = _rolling_mean(lv, 48)
    vs = _rolling_std(lv, 48)
    feats.append((lv - vm) / np.where(vs > 1e-9, vs, np.nan))
    feats.append((rsi - 50.0) / 50.0)

    for p in (14, 22, 50, 200, 1000):
        feats.append((c - mas[p]) / atr_safe)
    for p in (14, 22, 50, 200, 1000):
        older = np.roll(mas[p], 3)
        older[:3] = np.nan
        feats.append((mas[p] - older) / atr_safe)

    feats.append((mas[14] - mas[50]) / atr_safe)
    feats.append((mas[50] - mas[200]) / atr_safe)
    feats.append((mas[200] - mas[1000]) / atr_safe)

    minute = np.array([t.hour * 60 + t.minute for t in times], dtype=np.float64)
    angle = 2.0 * np.pi * minute / 1440.0
    feats.extend([np.sin(angle), np.cos(angle)])

    x = np.column_stack(feats).astype(np.float32)
    raw = {
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": v,
        "atr": atr,
        "rsi": rsi,
        **{f"ma{p}": arr for p, arr in mas.items()},
    }
    return x, raw, times


def _first_barrier(
    high: np.ndarray,
    low: np.ndarray,
    start: int,
    end: int,
    up: float,
    down: float,
) -> int | None:
    for j in range(start, min(end, len(high))):
        uh = high[j] >= up
        dh = low[j] <= down
        if uh and dh:
            return None
        if uh:
            return 1
        if dh:
            return 0
    return None


def build_labels(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n = len(raw["close"])
    y = np.zeros((n, len(HEADS)), dtype=np.float32)
    m = np.zeros((n, len(HEADS)), dtype=np.float32)
    c, h, l, atr = raw["close"], raw["high"], raw["low"], raw["atr"]
    ma200, ma1000 = raw["ma200"], raw["ma1000"]

    for i in range(1000, n - 97):
        a = float(atr[i])
        if not math.isfinite(a) or a <= 1e-9:
            continue
        px = float(c[i])

        # Head 0: which 1 ATR barrier is hit first inside the next 2 hours?
        first = _first_barrier(h, l, i + 1, i + 25, px + a, px - a)
        if first is not None:
            y[i, 0] = float(first)
            m[i, 0] = 1.0

        # Head 1: does the market expand at least 1.5 ATR either way within 2h?
        fh = float(np.max(h[i + 1 : i + 25]))
        fl = float(np.min(l[i + 1 : i + 25]))
        y[i, 1] = float(max(fh - px, px - fl) >= 1.5 * a)
        m[i, 1] = 1.0

        # Head 2: fixed MA200-at-event becomes touched within 4h.
        if math.isfinite(ma200[i]):
            target = float(ma200[i])
            dist = abs(px - target) / a
            if 0.5 <= dist <= 6.0:
                y[i, 2] = float(np.any((l[i + 1 : i + 49] <= target) & (h[i + 1 : i + 49] >= target)))
                m[i, 2] = 1.0

        # Head 3: fixed MA1000-at-event becomes touched within 8h.
        if math.isfinite(ma1000[i]):
            target = float(ma1000[i])
            dist = abs(px - target) / a
            if 0.75 <= dist <= 10.0:
                y[i, 3] = float(np.any((l[i + 1 : i + 97] <= target) & (h[i + 1 : i + 97] >= target)))
                m[i, 3] = 1.0

        # Heads 4/5: when current bar interacts with MA, learn whether it moves
        # 1 ATR away on the same side before recrossing the barrier.
        for head, ma in ((4, ma200), (5, ma1000)):
            target = float(ma[i]) if math.isfinite(ma[i]) else float("nan")
            if not math.isfinite(target) or not (l[i] <= target <= h[i]):
                continue
            side = 1 if raw["open"][i] > target and c[i] > target else -1 if raw["open"][i] < target and c[i] < target else 0
            if side == 0:
                continue
            away = px + side * a
            outcome = None
            for j in range(i + 1, min(i + 25, n)):
                away_hit = h[j] >= away if side > 0 else l[j] <= away
                recross = l[j] < target if side > 0 else h[j] > target
                if away_hit and recross:
                    outcome = None
                    break
                if recross:
                    outcome = 0
                    break
                if away_hit:
                    outcome = 1
                    break
            if outcome is not None:
                y[i, head] = float(outcome)
                m[i, head] = 1.0

    return y, m


class SeqDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        mask: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        self.x = x
        self.y = y
        self.mask = mask
        self.indices = indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, pos: int):
        i = int(self.indices[pos])
        return (
            torch.from_numpy(self.x[i - SEQ + 1 : i + 1]),
            torch.from_numpy(self.y[i]),
            torch.from_numpy(self.mask[i]),
            i,
        )


def indices_for_period(
    times: list[datetime],
    x: np.ndarray,
    masks: np.ndarray,
    start: datetime,
    end: datetime,
    stride: int,
    future_guard_minutes: int = 480,
) -> np.ndarray:
    out = []
    guard = timedelta(minutes=future_guard_minutes)
    for i in range(max(1000, SEQ - 1), len(times), max(1, stride)):
        t = times[i]
        if t < start or t + guard >= end:
            continue
        seq = x[i - SEQ + 1 : i + 1]
        if not np.isfinite(seq).all():
            continue
        if masks[i].sum() <= 0:
            continue
        out.append(i)
    return np.array(out, dtype=np.int64)


def masked_loss(logits: dict[str, torch.Tensor], y, mask) -> torch.Tensor:
    total = torch.tensor(0.0, dtype=torch.float32, device=y.device)
    denom = torch.tensor(0.0, dtype=torch.float32, device=y.device)
    for hidx, name in enumerate(HEADS):
        valid = mask[:, hidx] > 0.5
        if valid.any():
            raw = nn.functional.binary_cross_entropy_with_logits(
                logits[name][valid],
                y[valid, hidx],
                reduction="sum",
            )
            # Equalize heads so sparse rejection tasks are learned rather than
            # drowned by the dense expansion/direction tasks.
            total = total + raw / valid.sum().clamp_min(1)
            denom = denom + 1.0
    return total / denom.clamp_min(1.0)


def collect_logits(model, loader, device: str):
    model.eval()
    logits = {name: [] for name in HEADS}
    targets = {name: [] for name in HEADS}
    masks = {name: [] for name in HEADS}
    indices = []
    embeds = []
    with torch.no_grad():
        for xb, yb, mb, ib in loader:
            xb = xb.to(device=device, dtype=torch.float32)
            out, latent = model(xb)
            for hidx, name in enumerate(HEADS):
                logits[name].append(out[name].detach().cpu().numpy())
                targets[name].append(yb[:, hidx].numpy())
                masks[name].append(mb[:, hidx].numpy())
            indices.append(ib.numpy())
            embeds.append(latent.detach().cpu().numpy())
    return (
        {k: np.concatenate(v) if v else np.array([]) for k, v in logits.items()},
        {k: np.concatenate(v) if v else np.array([]) for k, v in targets.items()},
        {k: np.concatenate(v) if v else np.array([]) for k, v in masks.items()},
        np.concatenate(indices) if indices else np.array([], dtype=np.int64),
        np.concatenate(embeds) if embeds else np.empty((0, 32), dtype=np.float32),
    )


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def auc_score(y: np.ndarray, p: np.ndarray) -> float | None:
    y = y.astype(np.int64)
    npos = int(y.sum())
    nneg = int(len(y) - npos)
    if npos == 0 or nneg == 0:
        return None
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    # Average tied ranks.
    sorted_p = p[order]
    start = 0
    while start < len(p):
        end = start + 1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        if end - start > 1:
            avg = (start + 1 + end) / 2.0
            ranks[order[start:end]] = avg
        start = end
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def best_temperature(logits: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    valid = mask > 0.5
    if valid.sum() < 30:
        return 1.0
    z = logits[valid]
    target = y[valid]
    best_t, best_brier = 1.0, float("inf")
    for t in np.linspace(0.5, 3.0, 51):
        p = sigmoid(z / float(t))
        brier = float(np.mean((p - target) ** 2))
        if brier < best_brier:
            best_t, best_brier = float(t), brier
    return best_t


def head_metrics(logits, y, mask, temperature: float) -> dict[str, Any]:
    valid = mask > 0.5
    if valid.sum() == 0:
        return {"count": 0}
    target = y[valid].astype(np.float64)
    p = sigmoid(logits[valid] / temperature)
    pred = p >= 0.5
    base = float(target.mean())
    brier = float(np.mean((p - target) ** 2))
    base_brier = float(np.mean((base - target) ** 2))
    result: dict[str, Any] = {
        "count": int(len(target)),
        "base_rate": base,
        "accuracy": float(np.mean(pred == (target >= 0.5))),
        "brier": brier,
        "baseline_brier": base_brier,
        "brier_skill": base_brier - brier,
        "auc": auc_score(target, p),
    }
    for threshold in (0.60, 0.65, 0.70, 0.75):
        selected = p >= threshold
        result[f"p_ge_{int(threshold*100)}_coverage"] = float(selected.mean())
        result[f"p_ge_{int(threshold*100)}_precision"] = (
            float(target[selected].mean()) if selected.any() else None
        )
    return result


def build_analog_summary(
    train_emb: np.ndarray,
    train_y: np.ndarray,
    train_mask: np.ndarray,
    test_emb: np.ndarray,
    test_y: np.ndarray,
    test_mask: np.ndarray,
    model_probs: np.ndarray,
) -> dict[str, Any]:
    rng = np.random.default_rng(SEED)
    train_valid = np.where(train_mask[:, 0] > 0.5)[0]
    test_valid = np.where(test_mask[:, 0] > 0.5)[0]
    if len(train_valid) == 0 or len(test_valid) == 0:
        return {"count": 0}
    bank_idx = rng.choice(train_valid, size=min(ANALOG_BANK, len(train_valid)), replace=False)
    query_idx = rng.choice(test_valid, size=min(ANALOG_QUERIES, len(test_valid)), replace=False)

    bank = train_emb[bank_idx].astype(np.float32)
    query = test_emb[query_idx].astype(np.float32)
    bank /= np.linalg.norm(bank, axis=1, keepdims=True).clip(min=1e-8)
    query /= np.linalg.norm(query, axis=1, keepdims=True).clip(min=1e-8)
    bank_y = train_y[bank_idx, 0]

    analog_p = np.zeros(len(query_idx), dtype=np.float32)
    k = min(7, len(bank_idx))
    for start in range(0, len(query_idx), 512):
        q = query[start : start + 512]
        sim = q @ bank.T
        top = np.argpartition(sim, -k, axis=1)[:, -k:]
        top_sim = np.take_along_axis(sim, top, axis=1)
        weights = np.exp((top_sim - top_sim.max(axis=1, keepdims=True)) * 6.0)
        labels = bank_y[top]
        analog_p[start : start + len(q)] = (
            (weights * labels).sum(axis=1) / weights.sum(axis=1).clip(min=1e-8)
        )

    truth = test_y[query_idx, 0]
    neural_p = model_probs[query_idx]
    blend = 0.70 * neural_p + 0.30 * analog_p
    return {
        "count": int(len(query_idx)),
        "k": int(k),
        "neural_brier": float(np.mean((neural_p - truth) ** 2)),
        "analog_brier": float(np.mean((analog_p - truth) ** 2)),
        "blend_brier": float(np.mean((blend - truth) ** 2)),
        "neural_accuracy": float(np.mean((neural_p >= 0.5) == truth)),
        "analog_accuracy": float(np.mean((analog_p >= 0.5) == truth)),
        "blend_accuracy": float(np.mean((blend >= 0.5) == truth)),
        "interpretation": "historical-neighbor memory is advisory; deployment requires OOS benefit over neural-only",
    }


def main() -> None:
    seed_everything()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    m1, _quotes, dataset = legacy.download_dataset(START, TEST_END)
    m5 = legacy.resample(m1, XAUTimeframe.M5)
    x_raw, raw, times = build_features(m5)
    y, mask = build_labels(raw)

    train_idx = indices_for_period(times, x_raw, mask, START, TRAIN_END, TRAIN_STRIDE)
    valid_idx = indices_for_period(times, x_raw, mask, TRAIN_END, VALID_END, VALID_STRIDE)
    test_idx = indices_for_period(times, x_raw, mask, VALID_END, TEST_END, TEST_STRIDE)

    # Training-only normalization. No test statistics leak into the scaler.
    train_feature_rows = x_raw[np.array([i for i, t in enumerate(times) if START <= t < TRAIN_END and i >= 1000])]
    finite_rows = train_feature_rows[np.isfinite(train_feature_rows).all(axis=1)]
    mean = finite_rows.mean(axis=0).astype(np.float32)
    std = finite_rows.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    x = ((x_raw - mean) / std).astype(np.float32)

    train_ds = SeqDataset(x, y, mask, train_idx)
    valid_ds = SeqDataset(x, y, mask, valid_idx)
    test_ds = SeqDataset(x, y, mask, test_idx)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GTGExperienceConfig(
        feature_count=len(FEATURE_NAMES),
        sequence_length=SEQ,
        hidden_size=64,
        num_layers=2,
        dropout=0.10,
        embedding_size=32,
        heads=HEADS,
    )
    model = GTGSequenceEncoder(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history: list[dict[str, Any]] = []
    best_state = None
    best_val = float("inf")
    patience = 2
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for xb, yb, mb, _ib in train_loader:
            xb = xb.to(device=device, dtype=torch.float32)
            yb = yb.to(device=device, dtype=torch.float32)
            mb = mb.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits, _latent = model(xb)
            loss = masked_loss(logits, yb, mb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        v_logits, v_y, v_m, _v_idx, _v_emb = collect_logits(model, valid_loader, device)
        per_head = []
        for hidx, name in enumerate(HEADS):
            valid = v_m[name] > 0.5
            if valid.any():
                p = sigmoid(v_logits[name][valid])
                per_head.append(float(np.mean((p - v_y[name][valid]) ** 2)))
        val_brier = fmean(per_head) if per_head else 1.0
        record = {
            "epoch": epoch,
            "train_loss": fmean(losses) if losses else None,
            "valid_mean_brier": val_brier,
        }
        history.append(record)
        print(json.dumps({"phase": "train", **record}), flush=True)

        if val_brier + 1e-5 < best_val:
            best_val = val_brier
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    train_eval_loader = DataLoader(train_ds, batch_size=BATCH * 2, shuffle=False, num_workers=0)
    tr_logits, tr_y_dict, tr_m_dict, tr_indices, tr_emb = collect_logits(model, train_eval_loader, device)
    v_logits, v_y_dict, v_m_dict, v_indices, v_emb = collect_logits(model, valid_loader, device)
    te_logits, te_y_dict, te_m_dict, te_indices, te_emb = collect_logits(model, test_loader, device)

    temperatures = {
        name: best_temperature(v_logits[name], v_y_dict[name], v_m_dict[name])
        for name in HEADS
    }
    validation = {
        name: head_metrics(v_logits[name], v_y_dict[name], v_m_dict[name], temperatures[name])
        for name in HEADS
    }
    oos = {
        name: head_metrics(te_logits[name], te_y_dict[name], te_m_dict[name], temperatures[name])
        for name in HEADS
    }

    # Rebuild dense target/mask matrices in loader order for analog analysis.
    tr_y = np.column_stack([tr_y_dict[n] for n in HEADS])
    tr_m = np.column_stack([tr_m_dict[n] for n in HEADS])
    te_y = np.column_stack([te_y_dict[n] for n in HEADS])
    te_m = np.column_stack([te_m_dict[n] for n in HEADS])
    te_probs_direction = sigmoid(te_logits["next_move_up"] / temperatures["next_move_up"])
    analog = build_analog_summary(
        tr_emb, tr_y, tr_m, te_emb, te_y, te_m, te_probs_direction
    )

    model_path = OUT / "gtg_experience_model.pt"
    torch.save(
        {
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "temperatures": temperatures,
            "training_period": [START.isoformat(), TRAIN_END.isoformat()],
            "calibration_period": [TRAIN_END.isoformat(), VALID_END.isoformat()],
            "oos_period": [VALID_END.isoformat(), TEST_END.isoformat()],
            "seed": SEED,
        },
        model_path,
    )

    # Small OOS prediction sample for auditability.
    sample_rows = []
    for pos in range(0, min(len(te_indices), 5000), max(1, len(te_indices) // 5000 or 1)):
        i = int(te_indices[pos])
        row: dict[str, Any] = {"time": times[i].isoformat()}
        for hidx, name in enumerate(HEADS):
            row[f"{name}_p"] = float(sigmoid(np.array([te_logits[name][pos] / temperatures[name]]))[0])
            row[f"{name}_y"] = float(te_y[pos, hidx])
            row[f"{name}_mask"] = float(te_m[pos, hidx])
        sample_rows.append(row)
    if sample_rows:
        with (OUT / "oos_predictions_sample.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)

    report = {
        "benchmark": "gtg-deep-experience-v1",
        "mode": "shadow_only",
        "dataset": dataset,
        "split_policy": {
            "training": [START.isoformat(), TRAIN_END.isoformat()],
            "calibration": [TRAIN_END.isoformat(), VALID_END.isoformat()],
            "untouched_oos_test": [VALID_END.isoformat(), TEST_END.isoformat()],
            "random_split_used": False,
            "future_leakage_allowed": False,
        },
        "model": {
            **asdict(config),
            "architecture": "causal_2layer_GRU_plus_multitask_heads",
            "feature_names": FEATURE_NAMES,
            "parameter_count": int(sum(p.numel() for p in model.parameters())),
            "device": device,
            "epochs_completed": len(history),
            "temperatures": temperatures,
        },
        "sample_counts": {
            "train_sequences": int(len(train_idx)),
            "validation_sequences": int(len(valid_idx)),
            "oos_sequences": int(len(test_idx)),
        },
        "training_history": history,
        "validation": validation,
        "oos": oos,
        "analog_memory": analog,
        "promotion": {
            "decision_authority": False,
            "status": "shadow_research",
            "requirements_before_gtg_weighting": [
                "positive OOS calibration skill on target heads",
                "stable walk-forward performance across subperiods",
                "ablation proving incremental value over deterministic GTG",
                "no degradation of drawdown after execution integration",
            ],
        },
        "limitations": [
            "Historical heatmap/centralized footprint/order-book tape is absent from the frozen Dukascopy dataset.",
            "The neural model learns from price/volume-proxy state only in this benchmark; live microstructure can be added as separate venue-qualified features later.",
            "Deep learning does not create an edge by itself; OOS and walk-forward evidence control promotion.",
        ],
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
