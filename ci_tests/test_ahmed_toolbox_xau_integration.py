from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import runpy

from pan_agent import ModelMessage, RunRequest, ToolExposure, ToolRegistry

from src.modules.assistant import xau_tools
from src.modules.xau import api as xau_api
from src.modules.xau import gen1_pipeline
from src.modules.assistant.prompt import (
    GEN1_TRADE_GOLD_TOOL,
    build_assistant_messages,
    gen1_trade_gold_request_context,
    is_gen1_trade_gold_trigger,
)
from src.modules.assistant.xau_tools import register_xau_research_tools
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


def test_xau_research_tools_are_deferred_and_explicitly_non_execution():
    registry = ToolRegistry()
    descriptors = register_xau_research_tools(registry)

    specs = {tool.name: tool for tool in registry.registered_tools()}
    descriptor_map = {item.tool_name: item for item in descriptors}

    assert set(specs) == {
        "get_xau_intraday_research",
        "get_xau_decision_fusion",
        "get_xau_paper_league",
        "run_gen1_trade_gold",
    }
    assert specs["get_xau_intraday_research"].exposure is ToolExposure.DEFERRED
    assert specs["get_xau_decision_fusion"].exposure is ToolExposure.DEFERRED
    assert specs["get_xau_paper_league"].exposure is ToolExposure.DEFERRED
    assert specs["run_gen1_trade_gold"].exposure is ToolExposure.DIRECT
    assert set(descriptor_map) == set(specs)
    assert "not valid for execution" in descriptor_map["get_xau_intraday_research"].summary
    assert "research-only" in descriptor_map["get_xau_decision_fusion"].summary.lower()


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
    module = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "src"
            / "modules"
            / "automation"
            / "tradingagents"
            / "xau_support.py"
        )
    )
    configure_xau_tradingagents = module["configure_xau_tradingagents"]
    xau_analysts = module["XAU_TRADINGAGENTS_ANALYSTS"]

    original = {
        "selected_analysts": ["market", "social", "news", "fundamentals"],
        "other": "preserved",
    }

    result = configure_xau_tradingagents(original)

    assert result["selected_analysts"] == xau_analysts
    assert "fundamentals" not in result["selected_analysts"]
    assert result["other"] == "preserved"
    assert original["selected_analysts"][-1] == "fundamentals"



def test_gen1_trade_gold_exact_trigger_freezes_allowed_tool():
    assert is_gen1_trade_gold_trigger("Gen1 trade gold") is True
    assert is_gen1_trade_gold_trigger("  GEN1   TRADE   GOLD  ") is True
    assert is_gen1_trade_gold_trigger("Gen1 trade silver") is False

    messages = [ModelMessage(role="user", content="Gen1 trade gold")]
    context = gen1_trade_gold_request_context(messages)
    assert context["allowed_tool_names"] == [GEN1_TRADE_GOLD_TOOL]
    assert context["gen1_trade_gold_contract"] == "v1"

    built = build_assistant_messages(messages)
    assert any(
        item.role == "system" and "GEN1 TRADE GOLD CONTRACT v1" in item.content
        for item in built
    )


def test_gen1_trade_gold_shared_core_runs_toolbox_panwatch_gen1_in_order(monkeypatch):
    calls = []

    async def fake_macro(force=False):
        calls.append(("ahmed_toolbox", force))
        return {
            "bias": 1, "bias_label": "bullish", "confidence": 0.8,
            "event_risk": False, "search_ok": True, "search_source": "ahmed_toolbox",
            "synthesis_ok": True, "summary": "macro ok", "drivers": ["driver"],
        }

    async def fake_snapshot(force=False):
        calls.append(("panwatch", force))
        return {
            "status": "ready", "candidate": "long_setup", "technical_mode": "test",
            "alignment": "bullish", "blocked": False, "frames": {"1m": {"direction": "bullish"}},
            "market_context": {"status": "ready"}, "market_context_error": None,
            "xaut_order_flow_error": None,
            "xaut_order_flow": {
                "footprint": {"available": True},
                "volume_profile": {"status": "ready"},
                "raw_book": {"pm10": {"bid_quantity": 1.0}},
                "forward_range_map": {"pm10": {"xau_up": 4370.0}},
            },
            "execution_status": "LOCKED_NO_TRADABLE_SPOT_FEED",
        }

    def fake_fusion(technical, macro):
        calls.append(("gen1", technical["candidate"], macro["bias"]))
        return {
            "state": "setup_macro_support", "technical_candidate": "long_setup",
            "regime": "trend", "cognitive_confidence": 0.77, "meta_decision": "eligible",
            "research_ready": True, "paper_entry_allowed": True, "execution_allowed": False,
        }

    monkeypatch.setattr(gen1_pipeline, "get_macro_context", fake_macro)
    monkeypatch.setattr(gen1_pipeline, "get_xau_snapshot", fake_snapshot)
    monkeypatch.setattr(gen1_pipeline, "build_decision_fusion", fake_fusion)

    result = __import__("asyncio").run(gen1_pipeline.run_gen1_trade_gold_pipeline())
    assert calls == [
        ("ahmed_toolbox", True),
        ("panwatch", False),
        ("gen1", "long_setup", 1),
    ]
    assert result["pipeline_order"] == ["ahmed_toolbox", "panwatch", "gen1"]
    assert result["pipeline_status"] == "ready"
    assert result["decision"] == "LONG"
    assert result["missing_layers"] == []
    assert result["answer_contract"]["never_hide_missing_layer"] is True


def test_gen1_trade_gold_chat_and_ui_call_same_shared_core(monkeypatch):
    payload = {
        "contract": "gen1-trade-gold-v1", "pipeline_status": "ready", "decision": "WAIT",
        "stages": {"ahmed_toolbox": {}, "panwatch": {}, "gen1": {}},
        "missing_layers": [], "stage_errors": {}, "macro": {}, "technical": {},
        "fusion": {}, "forward_range_map": {},
        "answer_contract": {"never_hide_missing_layer": True},
    }
    calls = []

    async def fake_core():
        calls.append("core")
        return payload

    monkeypatch.setattr(xau_tools, "run_gen1_trade_gold_pipeline", fake_core)
    monkeypatch.setattr(xau_api, "run_gen1_trade_gold_pipeline", fake_core)

    registry = ToolRegistry()
    register_xau_research_tools(registry)
    request = RunRequest(
        run_id="gen1-shared",
        messages=[ModelMessage(role="user", content="Gen1 trade gold")],
    )
    tool_result = __import__("asyncio").run(
        registry.execute("run_gen1_trade_gold", request, {})
    )
    ui_result = __import__("asyncio").run(xau_api.gen1_gold())

    assert tool_result.data == payload
    assert ui_result == payload
    assert calls == ["core", "core"]
