"""Performance statistics with deterministic, dependency-free math."""

from __future__ import annotations

from math import sqrt
from statistics import fmean, pstdev

from src.modules.strategy.validation.models import PerformanceMetrics, TradeRecord


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def max_drawdown_from_pnl(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def compute_performance(trades: list[TradeRecord]) -> PerformanceMetrics:
    if not trades:
        return PerformanceMetrics(
            trade_count=0,
            net_pnl=0.0,
            win_rate=None,
            expectancy=None,
            profit_factor=None,
            max_drawdown=0.0,
            sharpe_per_trade=None,
            sortino_per_trade=None,
            avg_mae=None,
            avg_mfe=None,
            avg_holding_seconds=None,
        )

    pnls = [float(t.pnl) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    expectancy = fmean(pnls)
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    if gross_loss == 0:
        profit_factor = float("inf") if gross_profit > 0 else None
    else:
        profit_factor = gross_profit / gross_loss

    sigma = pstdev(pnls)
    sharpe = None if sigma == 0 else expectancy / sigma

    downside = [min(0.0, p) for p in pnls]
    downside_deviation = sqrt(fmean([p * p for p in downside]))
    sortino = None if downside_deviation == 0 else expectancy / downside_deviation

    maes = [float(t.mae) for t in trades if t.mae is not None]
    mfes = [float(t.mfe) for t in trades if t.mfe is not None]
    holding = [(t.closed_at - t.opened_at).total_seconds() for t in trades]

    return PerformanceMetrics(
        trade_count=len(trades),
        net_pnl=sum(pnls),
        win_rate=len(wins) / len(trades),
        expectancy=expectancy,
        profit_factor=profit_factor,
        max_drawdown=max_drawdown_from_pnl(pnls),
        sharpe_per_trade=sharpe,
        sortino_per_trade=sortino,
        avg_mae=fmean(maes) if maes else None,
        avg_mfe=fmean(mfes) if mfes else None,
        avg_holding_seconds=fmean(holding) if holding else None,
    )
