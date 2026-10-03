"""Cost, spread and slippage sensitivity for normalized trades."""

from __future__ import annotations

from dataclasses import replace

from src.modules.strategy.validation.metrics import compute_performance
from src.modules.strategy.validation.models import TradeRecord


def cost_stress(
    trades: list[TradeRecord],
    cost_per_trade_levels: tuple[float, ...] = (0.0, 0.05, 0.10, 0.25, 0.50),
) -> dict:
    scenarios: list[dict] = []
    for cost in cost_per_trade_levels:
        if cost < 0:
            raise ValueError("stress cost cannot be negative")
        stressed = [replace(t, pnl=t.pnl - cost) for t in trades]
        metrics = compute_performance(stressed)
        scenarios.append(
            {
                "cost_per_trade": cost,
                "trade_count": metrics.trade_count,
                "net_pnl": metrics.net_pnl,
                "expectancy": metrics.expectancy,
                "profit_factor": metrics.profit_factor,
                "max_drawdown": metrics.max_drawdown,
            }
        )

    first_non_positive = next(
        (
            s["cost_per_trade"]
            for s in scenarios
            if s["expectancy"] is not None and s["expectancy"] <= 0
        ),
        None,
    )
    return {
        "scenarios": scenarios,
        "expectancy_break_even_cost": first_non_positive,
    }
