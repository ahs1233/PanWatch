"""Chronological IS/OOS and rolling walk-forward split primitives."""

from __future__ import annotations

from dataclasses import dataclass

from src.modules.strategy.validation.models import TradeRecord


@dataclass(frozen=True)
class WalkForwardFold:
    index: int
    train: tuple[TradeRecord, ...]
    test: tuple[TradeRecord, ...]


def chronological_split(
    trades: list[TradeRecord],
    in_sample_fraction: float,
) -> tuple[list[TradeRecord], list[TradeRecord]]:
    ordered = sorted(trades, key=lambda t: (t.closed_at, t.trade_id))
    if len(ordered) < 2:
        return ordered, []

    cut = int(len(ordered) * in_sample_fraction)
    cut = max(1, min(cut, len(ordered) - 1))
    return ordered[:cut], ordered[cut:]


def walk_forward_splits(
    trades: list[TradeRecord],
    *,
    train_size: int,
    test_size: int,
    step: int,
) -> list[WalkForwardFold]:
    ordered = sorted(trades, key=lambda t: (t.closed_at, t.trade_id))
    folds: list[WalkForwardFold] = []
    start = 0
    index = 0

    while start + train_size + test_size <= len(ordered):
        train_end = start + train_size
        test_end = train_end + test_size
        folds.append(
            WalkForwardFold(
                index=index,
                train=tuple(ordered[start:train_end]),
                test=tuple(ordered[train_end:test_end]),
            )
        )
        start += step
        index += 1
    return folds
