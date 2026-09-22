"""Monte Carlo robustness tests over completed-trade sequences."""

from __future__ import annotations

import random
from statistics import fmean

from src.modules.strategy.validation.metrics import max_drawdown_from_pnl
from src.modules.strategy.validation.models import TradeRecord


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    weight = pos - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def bootstrap_trade_sequences(
    trades: list[TradeRecord],
    *,
    iterations: int,
    seed: int,
) -> dict:
    if not trades:
        return {
            "iterations": 0,
            "terminal_pnl_p05": None,
            "terminal_pnl_p50": None,
            "terminal_pnl_p95": None,
            "max_drawdown_p50": None,
            "max_drawdown_p95": None,
            "loss_probability": None,
            "mean_terminal_pnl": None,
        }

    rng = random.Random(seed)
    source = [float(t.pnl) for t in trades]
    n = len(source)
    terminal: list[float] = []
    drawdowns: list[float] = []

    for _ in range(iterations):
        sample = [source[rng.randrange(n)] for _ in range(n)]
        terminal.append(sum(sample))
        drawdowns.append(max_drawdown_from_pnl(sample))

    return {
        "iterations": iterations,
        "terminal_pnl_p05": _percentile(terminal, 0.05),
        "terminal_pnl_p50": _percentile(terminal, 0.50),
        "terminal_pnl_p95": _percentile(terminal, 0.95),
        "max_drawdown_p50": _percentile(drawdowns, 0.50),
        "max_drawdown_p95": _percentile(drawdowns, 0.95),
        "loss_probability": sum(1 for x in terminal if x <= 0) / iterations,
        "mean_terminal_pnl": fmean(terminal),
    }
