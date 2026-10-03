"""XAUUSD-specific TradingAgents helpers.

TradingAgents v0.5.0 normalizes XAUUSD to Yahoo's GC=F internally. PanWatch
should still preserve XAUUSD as the user-facing instrument and pass an explicit
non-stock asset type so downstream prompts do not describe gold as a company.
"""

from __future__ import annotations

from typing import Any

XAU_TRADINGAGENTS_SYMBOL = "XAUUSD"
XAU_TRADINGAGENTS_ASSET_TYPE = "commodity"
XAU_TRADINGAGENTS_ANALYSTS = ["market", "social", "news"]


def configure_xau_tradingagents(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with company-fundamentals analysis disabled for gold."""
    out = dict(config)
    out["selected_analysts"] = list(XAU_TRADINGAGENTS_ANALYSTS)
    return out


def propagate_xau(
    graph: Any,
    trade_date: str,
    *,
    portfolio: Any | None = None,
):
    """Run upstream TradingAgents while preserving XAUUSD identity."""
    return graph.propagate(
        XAU_TRADINGAGENTS_SYMBOL,
        trade_date,
        asset_type=XAU_TRADINGAGENTS_ASSET_TYPE,
        portfolio=portfolio,
    )
