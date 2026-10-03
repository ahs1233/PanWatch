"""Session, regime and news-vs-non-news performance segmentation."""

from __future__ import annotations

from collections import defaultdict

from src.modules.strategy.validation.metrics import compute_performance
from src.modules.strategy.validation.models import TradeRecord


def _segment(trades: list[TradeRecord], key_fn) -> dict[str, dict]:
    buckets: dict[str, list[TradeRecord]] = defaultdict(list)
    for trade in trades:
        key = key_fn(trade)
        if key is not None:
            buckets[str(key)].append(trade)

    result: dict[str, dict] = {}
    for key, bucket in sorted(buckets.items()):
        metrics = compute_performance(bucket)
        result[key] = {
            "trade_count": metrics.trade_count,
            "net_pnl": metrics.net_pnl,
            "win_rate": metrics.win_rate,
            "expectancy": metrics.expectancy,
            "profit_factor": metrics.profit_factor,
            "max_drawdown": metrics.max_drawdown,
        }
    return result


def segment_performance(trades: list[TradeRecord]) -> dict:
    return {
        "session": _segment(trades, lambda t: t.session),
        "regime": _segment(trades, lambda t: t.regime),
        "news": _segment(
            trades,
            lambda t: (
                "news_window"
                if t.news_window is True
                else "non_news"
                if t.news_window is False
                else None
            ),
        ),
    }
