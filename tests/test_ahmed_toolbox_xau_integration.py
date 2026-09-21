from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pan_agent import ToolExposure, ToolRegistry

from src.modules.assistant.xau_tools import register_xau_research_tools
from src.modules.automation.tradingagents.xau_support import (
    XAU_TRADINGAGENTS_ANALYSTS,
    configure_xau_tradingagents,
)
from src.modules.strategy.xau_intraday import XAUIntradayEngine
from src.platform.external_tools.registry import register_ahmed_toolbox_tools
from src.platform.marketdata.xau_models import XAUBar, XAUQuote, XAUTimeframe


class _FakeToolboxClient:
    def list_tools(self):
        return [
            {
                "name": "reach_web_search",
                "description": "Search the public web",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "scrapling__fetch",
                "description": "Fetch a rendered page",
                "inputSchema": {
                    "type": "object",
                    "properties": {"url": {"type": "string"}},
                    "required": ["url"],
                },
            },
        ]

    def call_tool(self, name, arguments):
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"{name}:{arguments}",
                }
            ],
            "isError": False,
        }


def _bullish_bars(
    timeframe: XAUTimeframe,
    now: datetime,
    *,
    execution_eligible: bool = True,
) -> list[XAUBar]:
    spacing = {
        XAUTimeframe.M1: timedelta(minutes=1),
        XAUTimeframe.M5: timedelta(minutes=5),
        XAUTimeframe.M15: timedelta(minutes=15),
    }[timeframe]
    result = []
    for index in range(30):
        price = 4200.0 + index * 1.5
        result.append(
            XAUBar(
                timestamp=now - spacing * (29 - index),
                timeframe=timeframe,
                open=price,
                high=price + 1.2,
                low=price - 0.3,
                close=price + 1.0,
                volume=100 + index,
                source="test",
                execution_eligible=execution_eligible,
            )
        )
    return result


def _all_frames(now: datetime, *, execution_eligible: bool = True):
    return {
        timeframe: _bullish_bars(
            timeframe,
            now,
            execution_eligible=execution_eligible,
        )
        for timeframe in (XAUTimeframe.M1, XAUTimeframe.M5, XAUTimeframe.M15)
    }


def test_external_mcp_tools_are_registered_as_deferred_read_tools():
    registry = ToolRegistry()
    descriptors = register_ahmed_toolbox_tools(
        registry,
        _FakeToolboxClient(),
    )

    specs = {tool.name: tool for tool in registry.registered_tools()}

    assert set(specs) == {
        "ext_reach_web_search",
        "ext_scrapling_fetch",
    }
    assert specs["ext_reach_web_search"].exposure is ToolExposure.DEFERRED
    assert specs["ext_scrapling_fetch"].exposure is ToolExposure.DEFERRED
    assert {item.tool_name for item in descriptors} == set(specs)


def test_xau_research_tool_is_deferred_and_explicitly_non_execution():
    registry = ToolRegistry()
    descriptors = register_xau_research_tools(registry)

    specs = {tool.name: tool for tool in registry.registered_tools()}

    assert set(specs) == {"get_xau_intraday_research"}
    assert specs["get_xau_intraday_research"].exposure is ToolExposure.DEFERRED
    assert descriptors[0].tool_name == "get_xau_intraday_research"
    assert "not valid for execution" in descriptors[0].summary


def test_xau_execution_mode_blocks_research_proxy_data():
    now = datetime.now(timezone.utc)
    result = XAUIntradayEngine(require_execution_data=True).analyze(
        _all_frames(now, execution_eligible=False),
        quote=None,
        now=now,
    )

    assert result.blocked is True
    assert result.candidate == "none"
    assert "execution_quote_missing" in result.block_reasons
    assert "execution_1m_bars_required" in result.block_reasons
    assert "execution_5m_bars_required" in result.block_reasons
    assert "execution_15m_bars_required" in result.block_reasons


def test_xau_macro_conflict_is_warning_not_hard_gate():
    now = datetime.now(timezone.utc)
    quote = XAUQuote(
        bid=4244.0,
        ask=4244.2,
        observed_at=now,
        source="spot-test",
        execution_eligible=True,
    )
    result = XAUIntradayEngine(require_execution_data=True).analyze(
        _all_frames(now),
        quote=quote,
        macro_bias=-1,
        now=now,
    )

    assert result.blocked is False
    assert result.candidate == "long_setup"
    assert "macro_bias_conflicts_long" in result.warnings
    assert "macro_bias_conflicts_long" not in result.block_reasons


def test_tradingagents_xau_config_disables_company_fundamentals():
    original = {
        "selected_analysts": ["market", "social", "news", "fundamentals"],
        "other": "preserved",
    }

    result = configure_xau_tradingagents(original)

    assert result["selected_analysts"] == XAU_TRADINGAGENTS_ANALYSTS
    assert "fundamentals" not in result["selected_analysts"]
    assert result["other"] == "preserved"
    assert original["selected_analysts"][-1] == "fundamentals"
